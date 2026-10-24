"""MAPPO with the rollout and the update CUDA-graph captured. Torch only, no torchrl.

The same algorithm as the torchrl reference trainer (``examples/train_mappo.py``): a
decentralised shared-weight tanh-Gaussian actor, a centralised critic, GAE, clipped
surrogate with clip 0.2, smooth-L1 value loss, a one-sample Monte Carlo entropy bonus,
per-agent advantage normalization, gradient clipping at 1.0 over both networks, one Adam,
and the same ``TrainSpec`` recipes, curriculum gate and ``metrics.csv`` columns. What
differs is where the time goes. Per iteration the host issues:

* per env step, one replay of the *act* graph (actor forward, sample, write the action and
  the rollout slot), :meth:`Environment.step` (its own Warp graph), and one replay of the
  *store* graph (reward/flags/info into the slot);
* one replay of the *prepare* graph (critic over the rollout, GAE, minibatch permutations);
* ``epochs x minibatches`` replays of the *minibatch* graph (forward, loss, backward,
  clip, fused Adam step);
* one device-to-host read of every logged statistic.

Nothing else syncs. A minibatch with a non-finite gradient norm is skipped on device
through Adam's ``found_inf`` (the ``GradScaler`` mechanism), which leaves the weights and
the Adam state exactly as they were -- the same skip the reference trainer does on host.

Two deliberate differences from the reference trainer:

* **Timeouts bootstrap from** ``V(s_t)``. The env runs with ``auto_reset=True`` (in-graph
  where the scenario supports it), so the observation after a truncation is already the
  next episode's and there is no terminal observation. A truncated, non-terminated step
  gets ``r_t + gamma * V(s_t)`` and cuts the trace, the rsl_rl / Isaac Lab convention,
  instead of ``r_t + gamma * V(s_{t+1})``.
* **Curriculum knobs are applied before the first reset**, so the first episodes already
  run at the starting difficulty.

On a non-CUDA device, or with ``graphs=False``, the same bodies run eagerly.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from swarp import Environment
from swarp.rl.networks import (
    Actor,
    Critic,
    PolicyNet,
    StateCritic,
    invert_permutation,
    role_slots,
    tanh_normal_log_prob,
)
from swarp.rl.recipes import (
    ACTORS,
    COMMS,
    CRITIC_INPUTS,
    ROLE_PARAMS,
    SPECS,
    ActionFn,
    PolicyArm,
    SolveRateWindow,
    TrainSpec,
    build_env,
    coerce_scen_kwargs,
    score,
    wilson_ci,
)

#: The format :func:`train` writes. ``swarp.rl/2`` keeps every ``swarp.rl/1`` key and adds
#: the network classes and constructor arguments, so any architecture reloads.
CHECKPOINT_FORMAT = "swarp.rl/2"
#: Every format :func:`load_checkpoint` reads.
CHECKPOINT_FORMATS = ("swarp.rl/1", "swarp.rl/2")
#: The per-iteration statistics a default run returns from :meth:`MAPPO.update`, in order,
#: before the logged ``info()`` metrics.
LEGACY_STAT_NAMES = (
    "mean_reward", "n_terminated", "n_done", "loss_objective", "loss_critic", "entropy",
    "grad_skips",
)
_CLIP_EPS = 0.2
_MAX_GRAD_NORM = 1.0
# torchrl's ``_standardize`` default: float32 resolution.
_ADV_EPS = torch.finfo(torch.float32).resolution


class NonFiniteGradients(RuntimeError):
    """Raised when the run is producing non-finite gradients faster than it can learn."""


class RollbackLimit(RuntimeError):
    """Raised when a run has collapsed more often than :class:`RollbackGuard` rolls back."""


# ----------------------------------------------------------------------------- capture


class _Graphed:
    """Run ``fn`` eagerly, or capture it once and replay it.

    ``fn`` is warmed up on a side stream before capture (lazy cuBLAS, autograd and Adam
    state must not initialize inside the capture). Warm-up runs have real side effects, so
    the tensors in ``carries`` (counters, accumulators) are snapshotted before and restored
    after it; everything else ``fn`` writes must be idempotent for a given carry state. The
    first call after a capture replays it, since capturing does not execute the work.
    """

    def __init__(
        self,
        fn: Callable[[], None],
        enabled: bool,
        carries: Sequence[torch.Tensor] = (),
        warmup: int = 2,
        before_warmup: Callable[[], None] | None = None,
        after_warmup: Callable[[], None] | None = None,
    ) -> None:
        self.fn, self.enabled, self.carries, self.warmup = fn, enabled, carries, warmup
        self.before_warmup, self.after_warmup = before_warmup, after_warmup
        self.graph: torch.cuda.CUDAGraph | None = None

    def invalidate(self) -> None:
        self.graph = None

    def __call__(self) -> None:
        if not self.enabled:
            self.fn()
            return
        if self.graph is None:
            self.graph = self._capture()
        self.graph.replay()

    def _capture(self) -> torch.cuda.CUDAGraph:
        saved = [c.clone() for c in self.carries]
        if self.before_warmup is not None:
            self.before_warmup()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                self.fn()
        torch.cuda.current_stream().wait_stream(side)
        if self.after_warmup is not None:
            self.after_warmup()
        for c, s in zip(self.carries, saved, strict=True):
            c.copy_(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.fn()
        return graph


# ------------------------------------------------------------------------------- GAE


def gae(
    rew: torch.Tensor,
    value: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    lmbda: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generalized advantage estimation over a time-major rollout.

    ``rew`` is ``[T, E, A]``, ``value`` is ``[T + 1, E, A]`` (the last row is the value of
    the observation after the rollout), the flags are ``[T, E]`` in ``{0, 1}``. The trace
    is cut wherever an episode ended. A terminated step does not bootstrap; a truncated,
    non-terminated one bootstraps from ``V(s_t)`` because ``s_{t+1}`` already belongs to
    the next episode (see the module docstring).
    """
    T = rew.shape[0]
    done = torch.maximum(terminated, truncated).unsqueeze(-1)
    timeout = (truncated * (1.0 - terminated)).unsqueeze(-1)
    rew = rew + gamma * value[:T] * timeout
    not_done = 1.0 - done
    out = torch.empty_like(rew) if out is None else out
    acc = torch.zeros_like(value[0])
    for t in reversed(range(T)):
        delta = rew[t] + gamma * value[t + 1] * not_done[t] - value[t]
        acc = delta + gamma * lmbda * not_done[t] * acc
        out[t] = acc
    return out


# ------------------------------------------------------------------------------ the loss


def standardize_advantage(
    adv: torch.Tensor, roles: torch.Tensor | None = None, role_counts: Sequence[int] = ()
) -> torch.Tensor:
    """Per agent: statistics over the minibatch (dim 0), agent dim excluded.

    With ``roles`` ``[B, A]`` and the scenario's ``role_counts``, per role-sorted slot
    instead: the advantages are gathered into role-sorted order (role 0's agents by
    ascending index, then role 1's, ...), standardized per column and scattered back, so
    when roles move between agents every column still holds one role. ``role_counts``
    is static; the permutation is a one-hot cumulative sum, capturable like the rest.
    """
    if roles is None or not role_counts:
        return (adv - adv.mean(0)) / adv.std(0).clamp_min(_ADV_EPS)
    slots = role_slots(roles, len(role_counts))  # agent -> slot
    ranked = adv.gather(-1, invert_permutation(slots))  # slot -> agent's advantage
    ranked = (ranked - ranked.mean(0)) / ranked.std(0).clamp_min(_ADV_EPS)
    return ranked.gather(-1, slots)


def ppo_terms(
    loc: torch.Tensor,
    scale: torch.Tensor,
    u: torch.Tensor,
    logp_old: torch.Tensor,
    adv: torch.Tensor,
    noise: torch.Tensor,
    with_kl: bool = False,
) -> tuple[torch.Tensor, ...]:
    """The policy half of the loss from the actor's outputs: ``(objective, entropy)``.

    ``loc``/``scale`` are the actor's Gaussian over the pre-tanh sample, ``u`` the stored
    sample and ``noise`` a standard normal draw for the entropy's reparametrized sample,
    all ``[B, A, act_dim]``; ``logp_old`` and ``adv`` (already normalized, if at all) are
    ``[B, A]``. Network-free, so any actor (role heads, a recurrent core) reuses it.
    ``with_kl`` appends the detached approximate KL to the rollout policy,
    ``mean((r - 1) - log r)`` with ``r`` the probability ratio (Schulman's k3 estimator:
    unbiased and never negative).
    """
    log_w = tanh_normal_log_prob(u, loc, scale) - logp_old
    gain = torch.minimum(
        log_w.exp() * adv,
        log_w.clamp(math.log1p(-_CLIP_EPS), math.log1p(_CLIP_EPS)).exp() * adv,
    )
    entropy = -tanh_normal_log_prob(loc + scale * noise, loc, scale).mean()
    if not with_kl:
        return -gain.mean(), entropy
    log_w = log_w.detach()
    return -gain.mean(), entropy, (log_w.exp() - 1.0 - log_w).mean()


def policy_outputs(
    actor: torch.nn.Module,
    obs: torch.Tensor,
    roles: torch.Tensor | None = None,
    with_aux: bool = False,
    h0: torch.Tensor | None = None,
    done: torch.Tensor | None = None,
    comm_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """``(loc, scale, aux)`` from either actor: :class:`Actor` reads only ``obs``.

    With ``h0`` ``[S, A, H]`` and ``done`` ``[L, S]``, the rows of ``obs`` (and of
    ``roles``, ``comm_mask``) are ``S`` sequences of ``L`` steps, time-major (row
    ``l * S + s`` is step ``l`` of sequence ``s``), and a recurrent :class:`PolicyNet`
    unrolls each from its stored starting hidden state (:meth:`PolicyNet.unroll`). The
    outputs come back in the same flat row order.
    """
    if h0 is not None:
        L = done.shape[0]

        def seq(x: torch.Tensor | None) -> torch.Tensor | None:
            return None if x is None else x.view(L, -1, *x.shape[1:])

        loc, scale, aux, _ = actor.unroll(
            seq(obs), seq(roles), h0, done, with_aux=with_aux, comm_mask=seq(comm_mask)
        )
        return loc.flatten(0, 1), scale.flatten(0, 1), None if aux is None else aux.flatten(0, 1)
    if isinstance(actor, PolicyNet):
        loc, scale, aux, _ = actor.step(obs, roles, with_aux=with_aux, comm_mask=comm_mask)
        return loc, scale, aux
    loc, scale = actor(obs)
    return loc, scale, None


def critic_values(
    critic: torch.nn.Module,
    obs: torch.Tensor,
    state: torch.Tensor | None = None,
    roles: torch.Tensor | None = None,
) -> torch.Tensor:
    """``[*, A]`` values from either critic: :class:`Critic` reads only ``obs``."""
    if isinstance(critic, StateCritic):
        return critic(obs, state, roles)
    return critic(obs)


def ppo_loss(
    actor: Actor,
    critic: torch.nn.Module,
    obs: torch.Tensor,
    u: torch.Tensor,
    logp_old: torch.Tensor,
    adv: torch.Tensor,
    ret: torch.Tensor,
    noise: torch.Tensor,
    normalize_advantage: bool,
    state: torch.Tensor | None = None,
    roles: torch.Tensor | None = None,
    aux_target: torch.Tensor | None = None,
    role_counts: Sequence[int] = (),
    h0: torch.Tensor | None = None,
    done: torch.Tensor | None = None,
    comm_mask: torch.Tensor | None = None,
    with_kl: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Clipped surrogate, smooth-L1 value loss and MC entropy for one minibatch.

    ``obs`` is ``[B, A, obs_dim]``; ``u`` (the stored pre-tanh sample) and ``noise`` (a
    standard normal draw for the entropy's reparametrized sample) are ``[B, A, act_dim]``;
    the rest are ``[B, A]``. The optional inputs are ``state`` ``[B, state_dim]`` for a
    :class:`StateCritic`, ``roles`` ``[B, A]`` for role-reading networks, and
    ``aux_target`` ``[B, A, aux_dim]`` for a :class:`PolicyNet`'s aux head. Returns
    ``(objective, critic, entropy)``, plus the unweighted auxiliary MSE when
    ``aux_target`` is given, then the approximate KL with ``with_kl``. With ``roles`` and
    ``role_counts`` the advantages are normalized per role-sorted slot
    (:func:`standardize_advantage`). ``comm_mask``
    ``[B, A, A]`` goes to a communicating :class:`PolicyNet`; ``h0`` and ``done`` make the
    ``B`` rows sequences for a recurrent one (see :func:`policy_outputs`), and every other
    input then lists its rows in the same time-major order.

    A thin wrapper over :func:`ppo_terms` that keeps the original op order (advantages,
    actor, policy terms, critic), so a default run launches the same kernels in the same
    sequence.
    """
    if normalize_advantage:
        adv = standardize_advantage(adv, roles, role_counts)
    loc, scale, aux = policy_outputs(
        actor, obs, roles, with_aux=aux_target is not None, h0=h0, done=done,
        comm_mask=comm_mask,
    )
    objective, entropy, *kl = ppo_terms(loc, scale, u, logp_old, adv, noise, with_kl)
    value = critic_values(critic, obs, state, roles)
    if aux_target is None:
        return objective, F.smooth_l1_loss(value, ret), entropy, *kl
    return objective, F.smooth_l1_loss(value, ret), entropy, F.mse_loss(aux, aux_target), *kl


# ------------------------------------------------------------------------------ the core


def metric_names(metrics: Mapping[str, int]) -> list[str]:
    """One column name per logged value: the key for a scalar metric; ``rew_term{i}`` for
    each term of ``multiobj_reward``, ``{key}_{i}`` for any other multi-wide one."""
    names: list[str] = []
    for key, width in metrics.items():
        if width == 1:
            names.append(key)
        elif key == "multiobj_reward":
            names += [f"rew_term{i}" for i in range(width)]
        else:
            names += [f"{key}_{i}" for i in range(width)]
    return names


class MAPPO:
    """Rollout buffers, rollout, GAE and the PPO update over one ``Environment``.

    No logging, no checkpoints, no curriculum: :func:`train` owns those. ``env`` must
    have ``auto_reset=True`` and ``clone_outputs=False``: the captured graphs read the
    env's output buffers by address, which is what makes them fixed.

    The networks and ``spec.aux_coef`` decide what else is recorded, once, here: a
    :class:`StateCritic` with a ``state_dim`` adds a ``[T + 1, E, state_dim]`` buffer of
    ``env.critic_state()``, a role-reading network (or a scenario with roles under
    ``normalize_advantage``, whose statistics are then per role-sorted slot) a
    ``[T + 1, E, A]`` buffer of ``env.agent_roles()``, and ``aux_coef > 0`` (with a
    :class:`PolicyNet` that has an aux head) a ``[T, E, A, aux_dim]`` buffer of
    ``env.aux_targets()`` and the loss term ``aux_coef * mse(aux_head(features),
    targets)``. Each is written next to the observation by the act graph. With
    :class:`Actor`, :class:`Critic`, ``aux_coef=0`` and no roles none exists and the
    graphs are the plain ones.

    A communicating :class:`PolicyNet` on a scenario whose ``comm_mask()`` is not
    ``None`` (decided once, here) adds a ``[T, E, A, A]`` buffer of the mask. A recurrent
    one adds the carried hidden state ``h`` ``[E, A, H]`` and ``h_buf`` ``[T, E, A, H]``,
    the hidden state *entering* each step, written by the act graph; the store graph
    zeroes ``h`` for the envs whose episode just ended, since the next observation already
    belongs to the next episode. The update then draws minibatches of whole sequences:
    each env's ``T`` steps split into chunks of ``L = spec.seq_len`` (``T`` when 0), each
    chunk unrolled from its stored ``h_buf`` row with the hidden state zeroed after every
    done, so the recomputed log-probs are the rollout's. A minibatch holds
    ``(E * T / L) // minibatches`` chunks, the same number of samples as without memory.
    """

    def __init__(
        self,
        env: Environment,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        spec: TrainSpec,
        metrics: Mapping[str, int] = (),
        graphs: bool = True,
        compile: bool = False,
    ) -> None:
        if not env.auto_reset or env.clone_outputs:
            raise ValueError("MAPPO needs an Environment with auto_reset=True, clone_outputs=False")
        self.env, self.actor, self.critic, self.spec = env, actor, critic, spec
        dev = torch.device(env.device)
        self.device = dev
        self.cuda = dev.type == "cuda"
        self.graphs = graphs and self.cuda
        self.metrics = dict(metrics)  # info key -> width
        self.metric_names = metric_names(self.metrics)
        T, E, A = spec.steps_per_batch, env.n_envs, env.n_agents
        Ob, D = env.obs_dim, env.act_dim
        self.T, self.E, self.A = T, E, A
        self.N = N = T * E
        self.M = spec.minibatches
        self.B = N // self.M
        if self.B < 1:
            raise ValueError(f"{N} samples cannot make {self.M} minibatches")

        def z(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            return torch.zeros(*shape, device=dev, dtype=dtype)

        # Rollout, time-major so a step writes one contiguous slot.
        self.obs_buf = z(T + 1, E, A, Ob)
        self.u_buf = z(T, E, A, D)
        self.logp_buf = z(T, E, A)
        self.rew_buf = z(T, E, A)
        self.term_buf = z(T, E)
        self.trunc_buf = z(T, E)
        self.actions = z(E, A, D)
        # The action-rate penalty: the last step's action (zero after a done) and the
        # action box's half-widths, only when it is on.
        self.action_rate = spec.action_rate_coef > 0
        if self.action_rate:
            self.prev_actions = z(E, A, D)
            self.act_half = actor.act_half.to(dev).float()
            #: The weight the captured store graph reads; :func:`train` ramps it.
            self.rate_coef = torch.tensor(spec.action_rate_coef, device=dev)
        self.t = z(1, dtype=torch.long)
        self.metric_acc = z(sum(self.metrics.values()))
        # Optional inputs, Python flags read at capture time. Each buffer is ``None`` unless
        # a network reads it.
        self.state_critic = isinstance(critic, StateCritic)
        self.policy_net = isinstance(actor, PolicyNet)
        self.actor_roles = self.policy_net and actor.uses_roles
        self.use_state = self.state_critic and critic.state_dim > 0
        self.role_counts = tuple(env.role_counts)
        # With roles, the advantages are normalized per role-sorted slot, which reads them
        # too (see standardize_advantage).
        self.role_adv = bool(self.role_counts) and spec.normalize_advantage
        self.use_roles = (
            self.actor_roles or (self.state_critic and critic.uses_roles) or self.role_adv
        )
        self.use_aux = spec.aux_coef > 0
        if self.use_aux and not (self.policy_net and actor.aux_head is not None):
            raise ValueError("aux_coef > 0 needs a PolicyNet with an aux head (aux_dim > 0)")
        if self.use_aux and env.aux_dim != actor.aux_dim:
            raise ValueError(
                f"the actor predicts {actor.aux_dim} aux targets; the env provides "
                f"aux_dim={env.aux_dim}"
            )
        if self.actor_roles and env.role_counts != actor.role_counts:
            raise ValueError(
                f"the actor expects role_counts {actor.role_counts}; the env has "
                f"{env.role_counts}"
            )
        if self.use_state and env.state_dim != critic.state_dim:
            raise ValueError(
                f"the critic reads a {critic.state_dim}-wide state; the env provides "
                f"state_dim={env.state_dim}"
            )
        if self.state_critic and critic.uses_roles and env.role_counts != critic.role_counts:
            raise ValueError(
                f"the critic expects role_counts {critic.role_counts}; the env has "
                f"{env.role_counts}"
            )
        self.state_buf = z(T + 1, E, env.state_dim) if self.use_state else None
        self.role_buf = z(T + 1, E, A, dtype=torch.long) if self.use_roles else None
        self.aux_buf = z(T, E, A, env.aux_dim) if self.use_aux else None
        # Memory and communication.
        self.recurrent = self.policy_net and actor.recurrent
        self.use_mask = self.policy_net and actor.communicates and env.comm_mask() is not None
        self.mask_buf = z(T, E, A, A, dtype=torch.bool) if self.use_mask else None
        self.h = self.h_buf = self.done_buf = None
        # The update draws minibatches of ``mb_units`` units out of ``n_units``: samples,
        # or (recurrent) chunks of L consecutive steps of one env.
        self.L, self.n_units, self.mb_units = 1, N, self.B
        if self.recurrent:
            L = spec.seq_len or T
            if T % L:
                raise ValueError(f"seq_len {L} must divide steps_per_batch {T}")
            self.L, self.n_units = L, E * T // L
            self.mb_units = self.n_units // self.M
            if self.mb_units < 1:
                raise ValueError(
                    f"{self.n_units} chunks of {L} steps cannot make {self.M} minibatches"
                )
            self.B = self.mb_units * L
            H = actor.hidden_dim
            self.h = z(E, A, H)
            self.h_buf = z(T, E, A, H)
            self.done_buf = z(T, E)
            self._steps = torch.arange(L, device=dev)
        # state, roles, aux targets, comm mask
        self._src: tuple[torch.Tensor | None, ...] = (None, None, None, None)
        # Update.
        self.v_buf = z(T + 1, E, A)
        self.adv_buf = z(T, E, A)
        self.ret_buf = z(T, E, A)
        self.perms = z(spec.epochs * self.M, self.mb_units, dtype=torch.long)
        self.k = z(1, dtype=torch.long)
        self.ent_coef = torch.tensor(spec.entropy_coeff, device=dev)
        # [mean reward, n_terminated, n_done] from prepare, then
        # [objective, critic, entropy, (aux,) skips, (KL, KL stops)] summed over the
        # minibatch steps.
        self.n_losses = 4 if self.use_aux else 3
        # The KL is measured for the brake and for the adaptive lr; only the brake trips.
        self.kl_brake = spec.target_kl is not None or spec.adaptive_kl is not None
        self._kl_trip = 1.5 * spec.target_kl if spec.target_kl is not None else math.inf
        self.prep_stats = z(3)
        self.mb_stats = z(self.n_losses + 1 + (2 if self.kl_brake else 0))
        self.found_inf = z(())
        self.force_skip = z(())
        # Sticky within an iteration once a minibatch's KL trips the brake; prepare clears it.
        self.kl_stop = z(()) if self.kl_brake else None
        #: The keys of :meth:`update`'s result, in the order the stats buffers hold them.
        losses = [*LEGACY_STAT_NAMES[:6], *(["loss_aux"] if self.use_aux else [])]
        kl = ["approx_kl", "kl_stops"] if self.kl_brake else []
        self.stat_names = [*losses, "grad_skips", *kl, *self.metric_names]

        params = [*actor.parameters(), *critic.parameters()]
        self.params = params
        # Inductor fuses the loss's elementwise work (bias/tanh/log-prob chains) around the
        # cuBLAS GEMMs; the result is still captured by the minibatch graph below.
        self._loss = torch.compile(ppo_loss, dynamic=False) if compile else ppo_loss
        if self.cuda:
            # fused + capturable: a capturable step, and a fused kernel that honours
            # ``found_inf`` by skipping the update and undoing the step increment. Eager
            # CUDA uses it too, so the skip never syncs.
            # A scheduled lr is a tensor, filled in place: the minibatch graph would bake
            # a float in at capture. A rollback scales it too.
            scheduled = spec.lr_final is not None or spec.rollback or spec.adaptive_kl is not None
            lr = torch.tensor(spec.lr, device=dev) if scheduled else spec.lr
            self.optim = torch.optim.Adam(params, lr=lr, fused=True, capturable=True)
            self.optim.found_inf = self.found_inf
            self.optim.grad_scale = None
        else:
            self.optim = torch.optim.Adam(params, lr=spec.lr)

        self._out: tuple[Any, ...] | None = None
        self._act_ptrs: tuple[int, ...] = ()  # the buffers the act graph reads
        self.act_recaptures = 0  # act-graph recaptures in the current rollout
        self.store_recaptures = 0  # store-graph recaptures in the current rollout
        self._sig: tuple[int, ...] | None = None  # the buffers the store graph reads
        # The warm-up runs the bodies for real: the carried hidden state is restored too.
        hidden = (self.h,) if self.recurrent else ()
        self._act = _Graphed(self._act_body, self.graphs, carries=(self.t, *hidden))
        self._store = _Graphed(
            self._store_body, self.graphs, carries=(self.t, self.metric_acc, *hidden)
        )
        self._prepare = _Graphed(self._prepare_body, self.graphs, carries=(), warmup=1)
        kl_carry = (self.kl_stop,) if self.kl_brake else ()
        self._minibatch = _Graphed(
            self._minibatch_body, self.graphs, carries=(self.k, self.mb_stats, *kl_carry),
            before_warmup=lambda: self.force_skip.fill_(1.0),
            after_warmup=lambda: self.force_skip.fill_(0.0),
        )

    # --------------------------------------------------------------------- rollout

    def reset(self) -> None:
        """Reset every env and point the act graph at the env's observation buffer.

        With ``spec.stagger_starts`` the envs are then desynchronised before the first
        rollout: each is re-reset once at a random step of a ``max_steps``-long warm-up
        under the initial greedy policy, so episode ages spread uniformly over
        ``[0, max_steps)``. Without it, a scenario whose episodes all run to the time
        limit truncates every env on the same step forever, and every batch is the same
        ``steps_per_batch`` slice of the episode in all envs. Only the public
        ``reset_at`` is used; the step counter stays the env's own.

        A recurrent policy's hidden state starts at zero and is carried through the
        warm-up, zeroed in each env whose episode ends or is re-reset, so the first
        rollout continues every env's episode with the memory of it.
        """
        obs = self.env.reset()
        if self.recurrent:
            self.h.zero_()
        if self.action_rate:
            self.prev_actions.zero_()
        horizon = self.env.max_steps or 0
        if self.spec.stagger_starts and horizon > 1:
            gen = torch.Generator(device=self.device).manual_seed(0)
            offset = torch.randint(0, horizon, (self.E,), device=self.device, generator=gen)
            with torch.no_grad():
                for t in range(horizon):
                    obs, _, term, trunc, _ = self.env.step(self._greedy(obs))
                    restart = offset == t
                    if self.recurrent:
                        done = torch.logical_or(torch.logical_or(term, trunc), restart)
                    obs = self.env.reset_at(restart)
                    if self.recurrent:
                        self.h.masked_fill_(done.view(-1, 1, 1), 0.0)
        self._out = (obs, None, None, None, None)
        self._fetch_sources()
        if self.use_roles:
            self._check_role_counts(self._src[1])

    def _check_role_counts(self, roles: torch.Tensor) -> None:
        """Raise unless every env's roles honour ``role_counts`` (one host read, at reset).

        :class:`PolicyNet`'s ``"separate"`` slices and the per-slot statistics assume the
        counts; a row that breaks them would be silently misrouted, not rejected.
        """
        ids = torch.arange(len(self.role_counts), device=roles.device)
        counts = (roles.long().unsqueeze(-1) == ids).sum(-2)  # [E, R]
        want = torch.tensor(self.role_counts, device=roles.device)
        bad = (counts != want).any(-1)
        if bad.any():
            e = int(bad.nonzero()[0, 0])
            raise ValueError(
                f"agent_roles() breaks role_counts {self.role_counts} in "
                f"{int(bad.sum())}/{self.E} envs; env {e} has roles {roles[e].tolist()} "
                f"(per-role counts {counts[e].tolist()}). Every env must hold exactly "
                "role_counts[r] agents in role r, with roles in [0, n_roles)."
            )

    def _signature(self, out: tuple[Any, ...]) -> tuple[int, ...]:
        _, rew, term, trunc, info = out
        return (
            rew.data_ptr(), term.data_ptr(), trunc.data_ptr(),
            *(info[k].data_ptr() for k in self.metrics if k in info),
        )

    def _greedy(self, obs: torch.Tensor) -> torch.Tensor:
        """The greedy action; a recurrent policy advances ``self.h``."""
        if not (self.recurrent or self.use_mask):
            if self.actor_roles:
                return self.actor.greedy(obs, self.env.agent_roles())
            return self.actor.greedy(obs)
        roles = self.env.agent_roles() if self.actor_roles else None
        mask = self.env.comm_mask() if self.use_mask else None
        with torch.no_grad():
            loc, _, _, h = self.actor.step(obs, roles, self.h, comm_mask=mask)
        if self.recurrent:
            self.h.copy_(h)
        return self.actor.squash(loc)

    def _fetch_sources(self) -> None:
        """Point at the env's state/role/aux/mask buffers for the current observation
        (host only)."""
        if not (self.use_state or self.use_roles or self.use_aux or self.use_mask):
            return
        E, A = self.E, self.A
        state = roles = aux = mask = None
        if self.use_state:
            state = self.env.critic_state()
            want = (E, self.env.state_dim)
            if state is None or tuple(state.shape) != want:
                raise ValueError(f"critic_state() must be a {want} tensor, got {_shape(state)}")
        if self.use_roles:
            roles = self.env.agent_roles()
            if roles is None or tuple(roles.shape) != (E, A) or roles.is_floating_point():
                raise ValueError(
                    f"agent_roles() must be an integer {(E, A)} tensor, got {_shape(roles)}"
                )
        if self.use_aux:
            aux = self.env.aux_targets()
            want = (E, A, self.env.aux_dim)
            if aux is None or tuple(aux.shape) != want:
                raise ValueError(f"aux_targets() must be a {want} tensor, got {_shape(aux)}")
        if self.use_mask:
            mask = self.env.comm_mask()
            if mask is None or tuple(mask.shape) != (E, A, A) or mask.dtype != torch.bool:
                raise ValueError(
                    f"comm_mask() must be a bool {(E, A, A)} tensor, got {_shape(mask)}"
                )
        self._src = (state, roles, aux, mask)

    def _act_inputs(self) -> tuple[torch.Tensor, ...]:
        """The env-owned buffers the act graph reads by address."""
        return (self._out[0], *(x for x in self._src if x is not None))

    def _act_body(self) -> None:
        obs = self._out[0]
        self.obs_buf.index_copy_(0, self.t, obs.unsqueeze(0))
        state, roles, aux, mask = self._src
        if self.state_buf is not None:
            self.state_buf.index_copy_(0, self.t, state.unsqueeze(0))
        if self.role_buf is not None:
            roles = roles.long()
            self.role_buf.index_copy_(0, self.t, roles.unsqueeze(0))
        if self.aux_buf is not None:
            self.aux_buf.index_copy_(0, self.t, aux.unsqueeze(0))
        if self.mask_buf is not None:
            self.mask_buf.index_copy_(0, self.t, mask.unsqueeze(0))
        if self.recurrent:
            self.h_buf.index_copy_(0, self.t, self.h.unsqueeze(0))
            loc, scale, _, h = self.actor.step(obs, roles, self.h, comm_mask=mask)
            self.h.copy_(h)
        else:
            loc, scale, _ = policy_outputs(self.actor, obs, roles, comm_mask=mask)
        u = loc + scale * torch.randn_like(loc)
        self.actions.copy_(self.actor.squash(u))
        self.u_buf.index_copy_(0, self.t, u.unsqueeze(0))
        self.logp_buf.index_copy_(0, self.t, tanh_normal_log_prob(u, loc, scale).unsqueeze(0))

    def _store_body(self) -> None:
        _, rew, term, trunc, info = self._out
        rate = None
        if self.action_rate:
            rate = ((self.actions - self.prev_actions) / self.act_half).square().mean(-1)
            rew = rew - self.rate_coef * rate
            done = torch.logical_or(term, trunc).view(-1, 1, 1)
            self.prev_actions.copy_(self.actions.masked_fill(done, 0.0))
        self.rew_buf.index_copy_(0, self.t, rew.unsqueeze(0))
        self.term_buf.index_copy_(0, self.t, term.unsqueeze(0).float())
        self.trunc_buf.index_copy_(0, self.t, trunc.unsqueeze(0).float())
        off = 0
        for key, width in self.metrics.items():
            if key == "action_rate" and rate is not None:
                self.metric_acc[off : off + 1] += rate.mean()
                off += 1
                continue
            x = info[key].float().reshape(-1, width).mean(0)
            self.metric_acc[off : off + width] += x
            off += width
        if self.recurrent:
            # The observation after a done already belongs to the next episode.
            self.h.masked_fill_(torch.logical_or(term, trunc).view(-1, 1, 1), 0.0)
        self.t.add_(1)

    @torch.no_grad()
    def rollout(self) -> None:
        """``steps_per_batch`` env steps into the rollout buffers."""
        if self._out is None:
            self.reset()
        self.t.zero_()
        self.metric_acc.zero_()
        self.act_recaptures = 0
        self.store_recaptures = 0
        env, actions = self.env, self.actions
        for _ in range(self.T):
            # The graphs read the env's output buffers by address. They are fixed in
            # steady state, but a first step or a reallocated buffer means a recapture.
            ptrs = tuple(t.data_ptr() for t in self._act_inputs())
            if ptrs != self._act_ptrs:
                self._act.invalidate()
                self._act_ptrs = ptrs
                self._count_act_recapture()
            self._act()
            out = env.step(actions)
            sig = self._signature(out)
            if sig != self._sig:
                self._store.invalidate()
                self._count_store_recapture(sig)
                self._sig = sig
            self._out = out
            self._fetch_sources()
            self._store()
        self.obs_buf[self.T].copy_(self._out[0])
        state, roles = self._src[:2]
        if self.state_buf is not None:
            self.state_buf[self.T].copy_(state)
        if self.role_buf is not None:
            self.role_buf[self.T].copy_(roles)

    def _count_act_recapture(self) -> None:
        """Raise once the act graph recaptures more than twice within one rollout.

        A first rollout recaptures up to twice (the reset's buffers, then the step's), and
        a buffer that moves once at a rollout boundary costs one more; a hook that hands
        back a fresh tensor every call recaptures on every step, and would otherwise only
        run slowly and silently.
        """
        if not self.graphs:
            return
        self.act_recaptures += 1
        if self.act_recaptures > 2:
            raise RuntimeError(
                f"the act graph was recaptured {self.act_recaptures} times in one rollout: "
                "the observation, critic_state(), agent_roles(), aux_targets() or "
                "comm_mask() returns a "
                "tensor at a new address on every call. Each must return the same "
                "fixed-address buffer, updated in place (see 'Training hooks' in "
                "docs/writing-a-scenario.md)."
            )

    def _count_store_recapture(self, sig: tuple[int, ...]) -> None:
        """Raise once the store graph recaptures more than twice within one rollout,
        naming the outputs that moved: as for the act graph, a reward, flag or logged
        ``info()`` metric handed back at a new address every step would otherwise recapture
        on every step, reading memory the previous step just freed."""
        if not self.graphs:
            return
        self.store_recaptures += 1
        if self.store_recaptures > 2:
            names = ["reward", "terminated", "truncated",
                     *(k for k in self.metrics if k in self._out[4])]
            moved = [n for n, a, b in zip(names, sig, self._sig or (), strict=False) if a != b]
            raise RuntimeError(
                f"the store graph was recaptured {self.store_recaptures} times in one rollout: "
                f"{', '.join(moved) or 'an output'} came back at a new address. The reward, "
                "the done flags and every info() key the recipe logs must be fixed-address "
                "buffers, updated in place."
            )

    # ---------------------------------------------------------------------- update

    def _prepare_body(self) -> None:
        spec, T = self.spec, self.T
        gamma, lmbda = spec.gamma, spec.lmbda
        with torch.no_grad():
            for t in range(T + 1):  # per step, so peak memory is one step's activations
                self.v_buf[t] = self._value(t)
            term = self.term_buf
            gae(self.rew_buf, self.v_buf, term, self.trunc_buf, gamma, lmbda, out=self.adv_buf)
            torch.add(self.adv_buf, self.v_buf[:T], out=self.ret_buf)
            if self.recurrent:
                torch.maximum(term, self.trunc_buf, out=self.done_buf)
            n = self.M * self.mb_units
            for e in range(spec.epochs):
                p = torch.rand(self.n_units, device=self.device).argsort()[:n]
                self.perms[e * self.M : (e + 1) * self.M] = p.view(self.M, self.mb_units)
            self.prep_stats[0] = self.rew_buf.mean()
            self.prep_stats[1] = term.sum()
            self.prep_stats[2] = torch.maximum(term, self.trunc_buf).sum()
            self.mb_stats.zero_()
            self.k.zero_()
            if self.kl_brake:
                self.kl_stop.zero_()

    def _value(self, t: int) -> torch.Tensor:
        """The critic over rollout row ``t``, ``[E, A]``."""
        if not self.state_critic:
            return self.critic(self.obs_buf[t])
        state = None if self.state_buf is None else self.state_buf[t]
        roles = None if self.role_buf is None else self.role_buf[t]
        return self.critic(self.obs_buf[t], state, roles)

    def _sequence_rows(self, chunks: torch.Tensor) -> torch.Tensor:
        """The flat rollout rows ``t * E + e`` of ``S`` chunks, time-major ``[L * S]``.

        Chunk ``c * E + e`` is steps ``[c * L, (c + 1) * L)`` of env ``e``, so the first
        ``S`` rows are the chunks' first steps."""
        E, L = self.E, self.L
        start = torch.div(chunks, E, rounding_mode="floor") * L
        env = chunks.remainder(E)
        return ((start.unsqueeze(0) + self._steps.unsqueeze(1)) * E + env.unsqueeze(0)).view(-1)

    def _minibatch_body(self) -> None:
        A = self.A
        idx = self.perms.index_select(0, self.k).view(-1)
        if self.recurrent:
            n_chunks = idx.numel()
            idx = self._sequence_rows(idx)
        obs = self.obs_buf[: self.T].reshape(self.N, A, -1).index_select(0, idx)
        u = self.u_buf.reshape(self.N, A, -1).index_select(0, idx)
        logp_old = self.logp_buf.reshape(self.N, A).index_select(0, idx)
        adv = self.adv_buf.reshape(self.N, A).index_select(0, idx)
        ret = self.ret_buf.reshape(self.N, A).index_select(0, idx)
        extra = {}
        if self.state_buf is not None:
            extra["state"] = self.state_buf[: self.T].reshape(self.N, -1).index_select(0, idx)
        if self.role_buf is not None:
            extra["roles"] = self.role_buf[: self.T].reshape(self.N, A).index_select(0, idx)
        if self.role_adv:
            extra["role_counts"] = self.role_counts
        if self.aux_buf is not None:
            extra["aux_target"] = self.aux_buf.reshape(self.N, A, -1).index_select(0, idx)
        if self.mask_buf is not None:
            extra["comm_mask"] = self.mask_buf.reshape(self.N, A, A).index_select(0, idx)
        if self.recurrent:
            # The stored hidden state entering each chunk's first step, not a recomputed one.
            h_rows = self.h_buf.reshape(self.N, A, -1)
            extra["h0"] = h_rows.index_select(0, idx[:n_chunks])
            extra["done"] = self.done_buf.reshape(self.N).index_select(0, idx).view(self.L, -1)
        noise = torch.randn_like(u)  # drawn here, not in the (possibly compiled) loss
        if self.kl_brake:
            extra["with_kl"] = True
        losses = self._loss(
            self.actor, self.critic, obs, u, logp_old, adv, ret, noise,
            self.spec.normalize_advantage, **extra,
        )
        if self.kl_brake:
            *losses, kl = losses
            # This minibatch's own step is skipped once its KL trips the brake, and every
            # later one in the iteration with it.
            trip = (kl > self._kl_trip).float()
            torch.maximum(self.kl_stop, trip, out=self.kl_stop)
        loss_obj, loss_crit, entropy = losses[:3]
        loss_ent = -self.ent_coef * entropy

        self.optim.zero_grad(set_to_none=False)
        total = loss_obj + loss_crit + loss_ent
        if self.use_aux:
            total = total + self.spec.aux_coef * losses[3]
        total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            self.params, _MAX_GRAD_NORM, foreach=self.cuda or None
        )
        if self.cuda:
            found = torch.maximum((~torch.isfinite(gnorm)).float(), self.force_skip)
            skip = torch.maximum(found, self.kl_stop) if self.kl_brake else found
            self.found_inf.copy_(skip)
            self.optim.step()
        else:
            found = (~torch.isfinite(gnorm)).float()
            skip = torch.maximum(found, self.kl_stop) if self.kl_brake else found
            if not skip.item():
                self.optim.step()
        kept = torch.stack(losses).detach()
        n = self.n_losses
        self.mb_stats[:n] += torch.where(skip > 0, torch.zeros_like(kept), kept)
        self.mb_stats[n] += found
        if self.kl_brake:
            # The KL over every minibatch with finite gradients, a stopped one included (its
            # policy is the one the brake froze); stops counted apart from non-finite skips.
            self.mb_stats[n + 1] += torch.where(found > 0, torch.zeros_like(kl), kl)
            self.mb_stats[n + 2] += self.kl_stop * (1.0 - found)
        self.k.add_(1)

    def update(self, entropy_coeff: float) -> dict[str, float]:
        """One PPO update over the last rollout. Returns the per-iteration statistics
        keyed by :attr:`stat_names`: the rollout's mean reward and episode-end counts, the
        losses and the skipped-minibatch count summed over minibatches, and the logged
        ``info()`` means. One host sync."""
        self.ent_coef.fill_(entropy_coeff)
        self._prepare()
        for _ in range(self.spec.epochs * self.M):
            self._minibatch()
        stats = torch.cat([self.prep_stats, self.mb_stats, self.metric_acc / self.T])
        return dict(zip(self.stat_names, stats.tolist(), strict=True))

    def snapshot(self) -> list[torch.Tensor]:
        """On-device copies of the weights and the optimizer's per-parameter state, in a
        fixed order, for :meth:`restore`."""
        out = [p.detach().clone() for p in self.params]
        for p in self.params:
            out += [v.clone() for v in self.optim.state.get(p, {}).values()
                    if isinstance(v, torch.Tensor)]
        return out

    def restore(self, snap: Sequence[torch.Tensor]) -> None:
        """Copy a :meth:`snapshot` back in place: the captured graphs and the fused Adam
        read these tensors by address, so they must not be replaced."""
        live = [p.detach() for p in self.params]
        for p in self.params:
            live += [v for v in self.optim.state.get(p, {}).values()
                     if isinstance(v, torch.Tensor)]
        if len(live) != len(snap):
            raise ValueError(f"snapshot holds {len(snap)} tensors, the trainer {len(live)}")
        with torch.no_grad():
            for dst, src in zip(live, snap, strict=True):
                dst.copy_(src)


def _shape(x: torch.Tensor | None) -> object:
    return None if x is None else tuple(x.shape)


# ------------------------------------------------------------------------- checkpoints


#: The network classes a ``swarp.rl/2`` checkpoint names. A checkpoint stores the name and
#: the constructor's keyword arguments as primitives, never a pickled class, so
#: ``torch.load(weights_only=True)`` still reads it.
ARCHS: dict[str, type[torch.nn.Module]] = {
    "Actor": Actor,
    "Critic": Critic,
    "PolicyNet": PolicyNet,
    "StateCritic": StateCritic,
}


def build_arch(name: str, kwargs: Mapping[str, Any]) -> torch.nn.Module:
    """``ARCHS[name](**kwargs)``, with a legible error for an unknown name."""
    if name not in ARCHS:
        raise ValueError(f"unknown network class {name!r}; known: {sorted(ARCHS)}")
    return ARCHS[name](**kwargs)


def load_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    """Actor, critic and metadata from a :func:`train` checkpoint.

    A ``swarp.rl/1`` checkpoint is always an :class:`Actor` and a :class:`Critic`, rebuilt
    from its dims; a ``swarp.rl/2`` one names its classes (see :data:`ARCHS`).
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    fmt = ckpt.get("format")
    if fmt not in CHECKPOINT_FORMATS:
        raise ValueError(f"{path} is not a swarp.rl checkpoint (one of {CHECKPOINT_FORMATS})")
    if fmt == "swarp.rl/1":
        actor = Actor(
            ckpt["obs_dim"], ckpt["act_dim"], ckpt["num_cells"],
            low=ckpt.get("act_low"), high=ckpt.get("act_high"),
        )
        critic = Critic(ckpt["obs_dim"], ckpt["n_agents"], ckpt["num_cells"])
    else:
        actor = build_arch(ckpt["actor_cls"], ckpt["actor_kwargs"])
        critic = build_arch(ckpt["critic_cls"], ckpt["critic_kwargs"])
    actor, critic = actor.to(device), critic.to(device)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    return actor.eval(), critic.eval(), ckpt


def checkpoint_archs(ckpt: Mapping[str, Any]) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """``(actor_cls, actor_kwargs, critic_cls, critic_kwargs)`` of a loaded checkpoint, in
    :func:`network_archs`' form. A ``swarp.rl/1`` checkpoint's are rebuilt from its dims."""
    if ckpt.get("format") != "swarp.rl/1":
        return (
            ckpt["actor_cls"], dict(ckpt["actor_kwargs"]),
            ckpt["critic_cls"], dict(ckpt["critic_kwargs"]),
        )
    D = ckpt["act_dim"]
    low, high = ckpt.get("act_low"), ckpt.get("act_high")
    actor = {
        "obs_dim": ckpt["obs_dim"], "act_dim": D, "num_cells": ckpt["num_cells"],
        "low": [-1.0] * D if low is None else list(low),
        "high": [1.0] * D if high is None else list(high),
    }
    critic = {"obs_dim": ckpt["obs_dim"], "n_agents": ckpt["n_agents"],
              "num_cells": ckpt["num_cells"]}
    return "Actor", actor, "Critic", critic


# Constructor keywords that follow from the scenario and its size, not from a network flag.
_SCENARIO_KEYS = ("obs_dim", "act_dim", "n_agents", "state_dim", "role_counts", "low", "high")


def _network_flags(cls: str, kwargs: Mapping[str, Any]) -> list[str]:
    """The ``python -m swarp.rl`` flags that build a network like ``cls(**kwargs)``."""
    flags = ["--num-cells", str(kwargs["num_cells"])]
    if cls == "Actor":
        return ["--actor", "mlp", *flags]
    if cls == "PolicyNet":
        flags = ["--actor", "policynet", "--role-params", kwargs["role_params"], *flags]
        if kwargs.get("recurrent"):
            flags += ["--recurrent", "--hidden-dim", str(kwargs["hidden_dim"])]
        if kwargs.get("comm", "none") != "none":
            flags += ["--comm", kwargs["comm"], "--comm-dim", str(kwargs["comm_dim"])]
        return [*flags, "--aux-coef", "<positive>"] if kwargs.get("aux_dim") else flags
    if cls == "Critic":
        return ["--critic-input", "obs", "--no-per-agent-value", *flags]
    state, obs = kwargs.get("state_dim", 0) > 0, kwargs.get("use_obs", False)
    critic_input = "state+obs" if state and obs else "state" if state else "obs"
    per_agent = "--per-agent-value" if kwargs.get("per_agent") else "--no-per-agent-value"
    return ["--critic-input", critic_input, per_agent, *flags]


def check_resume(
    path: str | Path,
    ckpt: Mapping[str, Any],
    built: tuple[str, dict[str, Any], str, dict[str, Any]],
    critic_only: bool = False,
) -> None:
    """Raise unless the checkpoint's networks are the ones ``built`` (what
    :func:`network_archs` gives for the current flags) describes, naming the mismatch and
    the flags that rebuild the checkpoint's networks. With ``critic_only`` only the critic
    must match."""
    saved = checkpoint_archs(ckpt)
    nets = [("critic", 2)] if critic_only else [("actor", 0), ("critic", 2)]
    problems: list[str] = []
    for which, i in nets:
        (cls_s, kw_s), (cls_b, kw_b) = saved[i : i + 2], built[i : i + 2]
        keys = [k for k in sorted(set(kw_s) | set(kw_b)) if kw_s.get(k) != kw_b.get(k)]
        if cls_s == cls_b and not keys:
            continue
        diffs = [f"class {cls_s} vs {cls_b}"] if cls_s != cls_b else []
        diffs += [f"{k} {kw_s.get(k)!r} vs {kw_b.get(k)!r}" for k in keys]
        hint = f"pass {' '.join(_network_flags(cls_s, kw_s))}"
        if any(k in _SCENARIO_KEYS for k in keys):
            hint += ", and the scenario settings it was trained with (--n-agents, --scen-kwarg)"
        problems.append(f"{which} (checkpoint vs flags: {', '.join(diffs)}); {hint}")
    if problems:
        raise ValueError(
            f"--resume {path}: the checkpoint's networks differ from the ones these flags "
            f"build: " + "; ".join(problems)
        )


def load_policy(path: str | Path, device: str = "cpu") -> torch.nn.Module:
    """The actor of a :func:`train` checkpoint; ``actor.greedy(obs)`` is the eval action
    (``actor.greedy(obs, roles)`` for a role-aware :class:`PolicyNet`)."""
    return load_checkpoint(path, device)[0]


def greedy_arm(actor: Actor) -> PolicyArm:
    """The policy arm of :func:`swarp.rl.recipes.score` for a :class:`Actor`."""
    return lambda n_envs, device, lo, hi: actor.greedy


def greedy_env_arm(actor: torch.nn.Module) -> Callable[[Environment], ActionFn]:
    """The env-reading policy arm of :func:`~swarp.rl.recipes.score`: the greedy action
    of ``actor`` with the roles ``env.agent_roles()`` and the mask ``env.comm_mask()``
    hold at each call.

    For a recurrent :class:`PolicyNet` the action function is stateful: each ``make(env)``
    starts from a zero hidden state and carries it from call to call. It never resets it,
    which is right for :func:`score`'s one un-reset episode per env; a loop whose env
    resets mid-run must build a fresh action function per episode.
    """

    def make(env: Environment) -> ActionFn:
        uses_roles = getattr(actor, "uses_roles", False)
        if not getattr(actor, "stateful", False):
            if uses_roles:
                return lambda obs: actor.greedy(obs, env.agent_roles())
            return actor.greedy
        h: torch.Tensor | None = None

        @torch.no_grad()
        def act(obs: torch.Tensor) -> torch.Tensor:
            nonlocal h
            roles = env.agent_roles() if uses_roles else None
            mask = env.comm_mask() if actor.communicates else None
            loc, _, _, h_next = actor.step(obs, roles, h, comm_mask=mask)
            h = h_next if actor.recurrent else None
            return actor.squash(loc)

        return act

    return make


def greedy_score_kwargs(actor: torch.nn.Module) -> dict[str, Any]:
    """The keyword :func:`~swarp.rl.recipes.score` takes for ``actor``'s greedy policy:
    ``policy`` for an obs-only actor, ``policy_from_env`` for one that reads roles, carries
    a hidden state or reads a communication mask."""
    if getattr(actor, "uses_roles", False) or getattr(actor, "stateful", False):
        return {"policy_from_env": greedy_env_arm(actor)}
    return {"policy": greedy_arm(actor)}


def network_archs(
    spec: TrainSpec, env: Environment, act_low: torch.Tensor, act_high: torch.Tensor
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """``(actor_cls, actor_kwargs, critic_cls, critic_kwargs)`` for a spec on an env, as
    :data:`ARCHS` names and primitive keywords (what a checkpoint stores).

    The defaults give :class:`Actor` and :class:`Critic`, except on a scenario with roles,
    where the obs critic is a :class:`StateCritic` over the role-sorted observations: the
    plain :class:`Critic` flattens them in agent order, a layout that changes whenever
    the roles move between agents. Raises when the spec asks for a hook the scenario does
    not provide.
    """
    obs_dim, n_agents, cells = env.obs_dim, env.n_agents, spec.num_cells
    roles = list(env.role_counts)
    if roles and sum(roles) != n_agents:
        raise ValueError(f"the scenario's role_counts {roles} do not sum to {n_agents} agents")
    if spec.actor == "mlp":
        actor = "Actor", {
            "obs_dim": obs_dim, "act_dim": env.act_dim, "num_cells": cells,
            "low": act_low.tolist(), "high": act_high.tolist(),
        }
    else:
        if spec.aux_coef > 0 and env.aux_dim == 0:
            raise ValueError("aux_coef > 0, but the scenario provides no aux_targets()")
        actor = "PolicyNet", {
            "obs_dim": obs_dim, "act_dim": env.act_dim, "num_cells": cells,
            "role_counts": roles, "role_params": spec.role_params,
            "aux_dim": env.aux_dim if spec.aux_coef > 0 else 0,
            "low": act_low.tolist(), "high": act_high.tolist(),
        }
        # Only when on, as PolicyNet.config writes them.
        if spec.recurrent:
            actor[1].update(recurrent=True, hidden_dim=spec.hidden_dim)
        if spec.actor_scale_min is not None:
            actor[1].update(scale_min=spec.actor_scale_min)
        if spec.comm != "none":
            if roles and spec.role_params == "separate":
                raise ValueError(
                    f"comm={spec.comm!r} needs one shared trunk; with roles use "
                    "role_params 'heads' or 'shared', not 'separate'"
                )
            actor[1].update(comm=spec.comm, comm_dim=spec.comm_dim)
    use_state = spec.critic_input in ("state", "state+obs")
    if use_state and env.state_dim == 0:
        raise ValueError(
            f"critic_input={spec.critic_input!r}, but the scenario provides no critic_state()"
        )
    if spec.critic_input == "obs" and not spec.per_agent_value and not roles:
        critic = "Critic", {"obs_dim": obs_dim, "n_agents": n_agents, "num_cells": cells}
    else:
        critic = "StateCritic", {
            "state_dim": env.state_dim if use_state else 0, "obs_dim": obs_dim,
            "n_agents": n_agents, "num_cells": cells,
            "use_obs": spec.critic_input in ("obs", "state+obs"), "role_counts": roles,
            "per_agent": spec.per_agent_value,
        }
    return (*actor, *critic)


# ---------------------------------------------------------------------------- adaptive KL

#: Per-iteration factor and floor of :func:`adapt_kl_lr`.
KL_LR_STEP = 1.5
KL_LR_FLOOR = 1.0 / 64.0


def adapt_kl_lr(mult: float, kl: float, target: float) -> float:
    """rsl_rl's adaptive learning rate, once per iteration: the multiplier on the
    scheduled rate after an iteration of mean approximate KL ``kl``, divided by
    :data:`KL_LR_STEP` above ``2 * target`` and multiplied by it below ``target / 2``,
    within ``[KL_LR_FLOOR, 1]``."""
    if kl > 2.0 * target:
        return max(mult / KL_LR_STEP, KL_LR_FLOOR)
    if kl < 0.5 * target:
        return min(mult * KL_LR_STEP, 1.0)
    return mult


# ---------------------------------------------------------------------------- rollback


class RollbackGuard:
    """When to snapshot the trainer and when to roll it back, from the windowed solve rate.

    Pure bookkeeping over ``(iteration, solve rate, ramp position)``, so it is tested on a
    synthetic trace; :func:`train` holds the snapshots themselves. Each call to
    :meth:`step` returns ``"snapshot"``, ``"rollback"`` (restore :attr:`restore_it`),
    ``"stop"`` or ``None``.

    - The best solve rate is kept per ramp position (``bucket``: :func:`train` passes every
      ramp's progress), so a harder condition is never held to an easier one's best.
    - Armed once the rate has passed ``arm``; until then nothing is snapshotted or judged.
      ``arm`` is ``floor + drop`` by default: a collapse is a fall from a mastered level,
      and a policy still learning around the floor is not judged (a Limo run armed at 0.5
      rolled back four times in 100 iterations while rising through 0.38-0.51 and
      stopped).
    - Snapshot every ``every`` iterations while the rate is at least ``keep`` times the
      position's best, keeping the newest ``ring``.
    - Roll back once the rate has stayed below ``max(floor, best - drop)`` for ``patience``
      consecutive iterations, to the newest snapshot at least ``lag + patience``
      iterations old: the rate is a ``lag``-iteration window, so it trails the policy, and
      a snapshot from inside the run of low rates may already hold the falling policy.
      Snapshots newer than the one restored are dropped.
    - After ``max_rollbacks`` rollbacks the next trigger is ``"stop"``.

    On the trace of a staged slim-T Limo run the 5650-5700 dip to 0.53 (a 44-point drop for
    about 75 iterations, then a partial recovery to 0.90) triggers too: it was the onset of
    the instability that collapsed the run to 0.04 by 6000, and rolling back there is the
    cheap reading of it.
    """

    def __init__(
        self,
        every: int = 50,
        ring: int = 4,
        keep: float = 0.9,
        arm: float | None = None,
        drop: float = 0.25,
        floor: float = 0.5,
        patience: int = 25,
        lag: int = 25,
        max_rollbacks: int = 4,
    ) -> None:
        self.every, self.ring, self.keep = every, ring, keep
        self.arm = floor + drop if arm is None else arm
        self.drop, self.floor, self.patience, self.lag = drop, floor, patience, lag
        self.max_rollbacks = max_rollbacks
        self.best: dict[int, float] = {}
        self.snaps: list[int] = []  # snapshot iterations, oldest first
        self.armed = False
        self.below = 0
        self.rollbacks = 0
        self.restore_it: int | None = None

    def step(self, it: int, solve: float | None, bucket: Hashable) -> str | None:
        if solve is None:
            self.below = 0
            return None
        best = max(self.best.get(bucket, 0.0), solve)
        self.best[bucket] = best
        self.armed = self.armed or solve >= self.arm
        if not self.armed:
            return None
        if solve < max(self.floor, best - self.drop):
            self.below += 1
            if self.below < self.patience:
                return None
            self.below = 0
            if self.rollbacks >= self.max_rollbacks:
                return "stop"
            old = [s for s in self.snaps if s <= it - self.lag - self.patience]
            if not old:
                return None  # nothing from before the drop: keep training
            self.restore_it = old[-1]
            self.snaps = old
            self.rollbacks += 1
            return "rollback"
        self.below = 0
        last = self.snaps[-1] if self.snaps else None
        if solve >= self.keep * best and (last is None or it - last >= self.every):
            self.snaps = [*self.snaps, it][-self.ring :]
            return "snapshot"
        return None


# ----------------------------------------------------------------------------- the CLI


@dataclass
class TrainConfig:
    """A resolved spec plus CLI overrides. All primitives, so it serializes."""

    scenario: str
    spec: TrainSpec
    scen_kwargs: Mapping[str, Any]
    device: str
    checkpoint_dir: Path
    resume: str | None = None
    resume_critic_only: bool = False
    """With ``resume``: load only the critic's weights and start a fresh actor."""
    start_difficulty: float = 0.0
    checkpoint_every: int = 250
    eval_every: int = 500
    eval_envs: int = 512
    seed: int = 0
    gate_window: int = 25
    max_skip_frac: float = 0.25
    max_skip_iters: int = 5
    target_solve_rate: float | None = None
    graphs: bool = True
    """CUDA-graph the rollout and the update (CUDA only; ignored elsewhere)."""
    tf32: bool = False
    """TF32 matmuls. Faster on Ampere and later; off by default for parity with the torchrl
    trainer."""
    compile: bool = False
    """``torch.compile`` the minibatch loss (needs triton). Costs startup time, then fuses
    the elementwise work between the GEMMs."""


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    """Parse into a :class:`TrainConfig`. The flags mirror ``examples/train_mappo.py``.

    Hyperparameter flags default to ``None``, meaning "take the spec's value".
    """
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser(
        prog="python -m swarp.rl", description="MAPPO on a swarp scenario, CUDA-graph captured."
    )
    p.add_argument("--scenario", required=True, choices=sorted(SPECS))
    p.add_argument("--device", default=default_device)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-agents", type=int)
    p.add_argument("--dt", type=float)
    p.add_argument("--substeps", type=int)
    p.add_argument("--max-steps", type=int, help="episode length")
    p.add_argument(
        "--scen-kwarg", action="append", default=[], metavar="K=V",
        help="scenario constructor keyword; repeatable",
    )
    p.add_argument("--iters", type=int)
    p.add_argument("--n-envs", type=int)
    p.add_argument("--steps-per-batch", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--minibatches", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--lr-final", type=float, help="linear schedule target")
    p.add_argument("--gamma", type=float)
    p.add_argument("--lmbda", type=float)
    p.add_argument("--entropy-coeff", type=float)
    p.add_argument("--entropy-coeff-final", type=float, help="linear schedule target")
    p.add_argument("--num-cells", type=int, help="MLP width")
    p.add_argument("--actor", choices=ACTORS,
                   help="mlp: obs-only Actor; policynet: role-aware PolicyNet (+ aux head)")
    p.add_argument("--role-params", choices=ROLE_PARAMS,
                   help="how a policynet's parameters depend on the role")
    p.add_argument("--critic-input", choices=CRITIC_INPUTS,
                   help="what the centralised critic reads")
    p.add_argument("--per-agent-value", action="store_true", default=None,
                   help="one value per agent (role-sorted slot) instead of one per team")
    p.add_argument("--no-per-agent-value", dest="per_agent_value", action="store_false")
    p.add_argument("--aux-coef", type=float, help="auxiliary-loss weight (needs policynet)")
    p.add_argument("--action-rate-coef", type=float,
                   help="weight of the action-rate penalty on each agent's reward")
    p.add_argument("--action-rate-ramp-iters", type=int,
                   help="iterations over which the action-rate weight ramps in (0: none)")
    p.add_argument("--recurrent", action="store_true", default=None,
                   help="a GRU after the policynet's trunk, trained by BPTT")
    p.add_argument("--hidden-dim", type=int, help="GRU hidden-state width (with --recurrent)")
    p.add_argument("--seq-len", type=int,
                   help="BPTT chunk length; must divide --steps-per-batch (default: all of it)")
    p.add_argument("--comm", choices=COMMS,
                   help="attn: one round of attention over teammates before the trunk")
    p.add_argument("--comm-dim", type=int, help="message width (with --comm attn)")
    p.add_argument("--normalize-advantage", action="store_true", default=None)
    p.add_argument("--no-normalize-advantage", dest="normalize_advantage",
                   action="store_false")
    p.add_argument("--curriculum-iters", type=int, help="0 disables")
    p.add_argument("--curriculum-gate", type=float)
    p.add_argument("--gate-window", type=int, default=25,
                   help="iterations of episode ends behind ep_solve_rate")
    p.add_argument("--start-difficulty", type=float, default=0.0)
    p.add_argument("--checkpoint-dir")
    p.add_argument("--resume", help="checkpoint to continue from (actor + critic weights)")
    p.add_argument("--resume-critic-only", action="store_true",
                   help="with --resume: load only the critic, start a fresh actor")
    p.add_argument("--checkpoint-every", type=int, default=250, help="0 disables")
    p.add_argument("--eval-every", type=int, default=500, help="0 disables")
    p.add_argument("--eval-envs", type=int, default=512)
    p.add_argument("--target-solve-rate", type=float)
    p.add_argument("--max-skip-frac", type=float, default=0.25,
                   help="abort if this fraction of minibatches has non-finite grads")
    p.add_argument("--max-skip-iters", type=int, default=5,
                   help="consecutive iterations over --max-skip-frac before aborting")
    p.add_argument("--no-graphs", dest="graphs", action="store_false",
                   help="run the rollout and update eagerly")
    p.add_argument("--tf32", action="store_true", help="TF32 matmuls (Ampere and later)")
    p.add_argument("--compile", action="store_true", help="torch.compile the minibatch loss")
    a = p.parse_args(argv)

    spec = SPECS[a.scenario]
    overrides = {
        f: getattr(a, f)
        for f in (
            "dt", "substeps", "max_steps", "n_agents", "iters", "n_envs",
            "steps_per_batch", "epochs", "minibatches", "lr", "lr_final", "gamma", "lmbda",
            "entropy_coeff", "entropy_coeff_final", "num_cells", "normalize_advantage",
            "curriculum_iters", "curriculum_gate", "actor", "role_params", "critic_input",
            "per_agent_value", "aux_coef", "action_rate_coef", "action_rate_ramp_iters",
            "recurrent", "hidden_dim", "seq_len", "comm", "comm_dim",
        )
        if getattr(a, f) is not None
    }
    try:
        spec = replace(spec, **overrides)
    except ValueError as exc:
        p.error(str(exc))
    if spec.n_agents < 1:
        p.error("--n-agents must be >= 1")
    if a.resume_critic_only and not a.resume:
        p.error("--resume-critic-only needs --resume")
    scen_kwargs = dict(spec.scen_kwargs)
    try:
        scen_kwargs.update(coerce_scen_kwargs(a.scenario, a.scen_kwarg))
    except (ValueError, KeyError) as exc:
        p.error(str(exc))

    return TrainConfig(
        scenario=a.scenario, spec=spec, scen_kwargs=scen_kwargs, device=a.device,
        checkpoint_dir=Path(a.checkpoint_dir or f"runs/{a.scenario}"), resume=a.resume,
        resume_critic_only=a.resume_critic_only,
        start_difficulty=a.start_difficulty, checkpoint_every=a.checkpoint_every,
        eval_every=a.eval_every, eval_envs=a.eval_envs, seed=a.seed,
        gate_window=a.gate_window, max_skip_frac=a.max_skip_frac,
        max_skip_iters=a.max_skip_iters, target_solve_rate=a.target_solve_rate,
        graphs=a.graphs, tf32=a.tf32, compile=a.compile,
    )


# --------------------------------------------------------------------------- training


def check_action_box(env: Environment, scenario: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-slot action box ``(low, high)``, each ``[act_dim]``, for :class:`Actor`.

    The actor's weights are shared across agents, so it can only map onto one box: raise
    unless every agent's box agrees with it. Zero-width slots (past an agent's model arity
    in a mixed fleet) are ignored, as the kernels ignore them. Call it before building an
    :class:`Actor` for an env from outside :func:`train`, which runs it.
    """
    lo, hi = env.action_bounds
    nonzero = hi > lo  # zero-width slots exist for mixed fleets and are legitimately 0
    low = torch.where(nonzero, lo, torch.inf).amin(0)
    high = torch.where(nonzero, hi, -torch.inf).amax(0)
    unused = ~nonzero.any(0)
    low = torch.where(unused, -1.0, low)
    high = torch.where(unused, 1.0, high)
    agrees = (lo == low) & (hi == high)
    if not agrees[nonzero].all():
        raise ValueError(
            f"{scenario!r} agents have different action boxes ({lo.tolist()} .. {hi.tolist()}); "
            "a shared-weight actor maps onto one box."
        )
    return low.float().cpu(), high.float().cpu()


def train(cfg: TrainConfig) -> Path:
    """Run the training loop. Returns the final checkpoint path."""
    spec, device = cfg.spec, cfg.device
    n_agents = spec.n_agents
    torch.manual_seed(cfg.seed)
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32

    env = build_env(
        cfg.scenario, n_envs=spec.n_envs, n_agents=n_agents, device=device, dt=spec.dt,
        substeps=spec.substeps, max_steps=spec.max_steps, seed=cfg.seed,
        scen_kwargs=cfg.scen_kwargs, env_kwargs={"auto_reset": True},
    )
    scen = env.scenario
    obs_dim, act_dim = env.obs_dim, env.act_dim
    act_low, act_high = check_action_box(env, cfg.scenario)

    env.reset()  # a probe: info() needs a reset world; the real reset follows the knobs
    info0 = scen.info()
    missing = [k for k in spec.metrics if k not in info0]
    if missing:
        raise ValueError(
            f"{cfg.scenario!r} info() has no key(s) {missing}; available: {sorted(info0)}"
        )
    for knob in spec.curriculum:
        if not hasattr(scen, knob.attr):
            raise ValueError(
                f"curriculum knob {knob.attr!r} is not an attribute of "
                f"{type(scen).__name__}; the ramp would silently do nothing."
            )
    for ramp in spec.ramps:
        if not hasattr(scen, ramp.attr):
            raise ValueError(
                f"ramp {ramp.attr!r} is not an attribute of {type(scen).__name__}; "
                f"the ramp would silently do nothing."
            )
    # A multiobj_reward key is [n_envs, n_agents, n_obj]; each term gets its own column.
    metrics = {
        k: (info0[k].shape[-1] if k == "multiobj_reward" else 1) for k in spec.metrics
    }
    if spec.action_rate_coef > 0:
        metrics["action_rate"] = 1  # computed by the trainer, not read from info
    metric_cols = metric_names(metrics)

    actor_cls, actor_kwargs, critic_cls, critic_kwargs = network_archs(
        spec, env, act_low, act_high
    )
    actor = build_arch(actor_cls, actor_kwargs).to(device)
    critic = build_arch(critic_cls, critic_kwargs).to(device)
    if cfg.resume:
        a, c, ckpt = load_checkpoint(cfg.resume, device)
        built = (actor_cls, actor_kwargs, critic_cls, critic_kwargs)
        check_resume(cfg.resume, ckpt, built, critic_only=cfg.resume_critic_only)
        if not cfg.resume_critic_only:
            actor.load_state_dict(a.state_dict())
        critic.load_state_dict(c.state_dict())
        what = "critic" if cfg.resume_critic_only else "actor + critic"
        print(f"resumed {what} from {cfg.resume}")
    actor.train()
    critic.train()

    difficulty = min(1.0, max(0.0, cfg.start_difficulty))

    def set_curriculum(gate_metric: float | None) -> float:
        """Ramp the knobs while the policy keeps solving; a ``None`` gate stays closed."""
        nonlocal difficulty
        if not spec.curriculum_iters:
            return 1.0
        if gate_metric is not None and gate_metric >= spec.curriculum_gate:
            difficulty = min(1.0, difficulty + 1.0 / spec.curriculum_iters)
        for knob in spec.curriculum:
            setattr(scen, knob.attr, knob.value(difficulty))
        if spec.curriculum_fn is not None:
            spec.curriculum_fn(scen, difficulty)
        return difficulty

    def set_ramps(it: int) -> float:
        """Set the ramped attributes for iteration ``it``; return the least progress."""
        changed = False
        for ramp in spec.ramps:
            v = ramp.value(it)
            if getattr(scen, ramp.attr) != v:
                setattr(scen, ramp.attr, v)
                changed = True
        if changed and env.world.runtime is not None:
            env.world.runtime.release()  # the next step recaptures with the new values
        return min((ramp.progress(it) for ramp in spec.ramps), default=1.0)

    set_curriculum(None)
    set_ramps(0)
    algo = MAPPO(
        env, actor, critic, spec, metrics=metrics, graphs=cfg.graphs, compile=cfg.compile
    )
    algo.reset()  # after the knobs, so the first episodes run at the starting difficulty

    ckpt_dir = cfg.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_file = (ckpt_dir / "metrics.csv").open("w", newline="")
    csv_writer = csv.writer(csv_file)
    # loss_aux only when the aux loss is on, so a default run's header is unchanged.
    aux_col = ["loss_aux"] if algo.use_aux else []
    ramp_col = ["ramp"] if spec.ramps else []
    # The guards' columns only when each is on, likewise.
    clock_col = ["ramp_clock"] if spec.ramp_gate is not None else []
    kl_col = ["approx_kl", "kl_stop_frac"] if algo.kl_brake else []
    rollback_col = ["rollbacks", "lr_mult"] if spec.rollback else []
    adapt_col = ["kl_lr_mult"] if spec.adaptive_kl is not None else []
    csv_writer.writerow(
        ["iter", "frames", "wall_s", "env_sps", "reward_per_step", "ep_solve_rate",
         "ep_ends", "ep_len_mean", "difficulty", *ramp_col, *clock_col, "entropy",
         "loss_objective", "loss_critic", *aux_col, "grad_skips", *kl_col, *rollback_col,
         *adapt_col, "eval_solve_rate", "eval_ci_lo", "eval_ci_hi"]
        + metric_cols
    )
    target = spec.target_solve_rate if cfg.target_solve_rate is None else cfg.target_solve_rate

    def save(tag: str) -> Path:
        path = ckpt_dir / f"{cfg.scenario}_{tag}.pt"
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "actor_cls": actor_cls,
                "actor_kwargs": actor_kwargs,
                "critic_cls": critic_cls,
                "critic_kwargs": critic_kwargs,
                "state_dim": env.state_dim,
                "role_counts": list(env.role_counts),
                "aux_dim": env.aux_dim,
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "n_agents": n_agents,
                "num_cells": spec.num_cells,
                "act_low": act_low.tolist(),
                "act_high": act_high.tolist(),
                "scenario": cfg.scenario,
                "scen_kwargs": {k: str(v) for k, v in cfg.scen_kwargs.items()},
                "dt": spec.dt,
                "substeps": spec.substeps,
                "max_steps": spec.max_steps,
                "difficulty": difficulty,
            },
            path,
        )
        return path

    frames_per_batch = spec.n_envs * spec.steps_per_batch
    n_mb = spec.epochs * spec.minibatches
    window = SolveRateWindow(window=cfg.gate_window)
    total_skips = hot_skip_iters = 0
    # Ranked on (difficulty, ramp progress, solve rate): a later checkpoint wins only at an
    # equal or harder difficulty (see train_mappo.py for the caging run that motivated it).
    best_key = (-1.0, -1.0, -1.0)
    # The ramps' clock: the iteration, or with ramp_gate the solve-gated count of ticks.
    gated = spec.ramp_gate is not None
    clock = 0
    first_start = min((r.start for r in spec.ramps), default=0)
    end_clock = max((r.end for r in spec.ramps), default=0) + spec.hold_iters
    lr_mult = 1.0
    kl_mult = 1.0  # the adaptive-KL factor on top of the schedule and the rollbacks
    guard = RollbackGuard(lag=cfg.gate_window) if spec.rollback else None
    snaps: dict[int, tuple[list[torch.Tensor], int, float]] = {}
    print(
        f"{cfg.scenario}: {n_agents} agents, obs_dim {obs_dim}, act_dim {act_dim}, "
        f"{spec.n_envs} envs x {spec.steps_per_batch} steps, dt {spec.dt}, "
        f"substeps {spec.substeps}, max_steps {spec.max_steps}, target {target:.2f}, "
        f"graphs {'on' if algo.graphs else 'off'}, tf32 {'on' if cfg.tf32 else 'off'}, "
        f"compile {'on' if cfg.compile else 'off'}"
    )
    if (actor_cls, critic_cls) != ("Actor", "Critic") or algo.use_aux:
        print(
            f"  actor {actor_cls} {actor_kwargs.get('role_params', '')}, critic {critic_cls} "
            f"(input {spec.critic_input}, per-agent {spec.per_agent_value}), "
            f"roles {list(env.role_counts)}, aux_coef {spec.aux_coef}"
            + (f", GRU {spec.hidden_dim} (BPTT {algo.L} steps)" if algo.recurrent else "")
            + (f", comm {spec.comm} {spec.comm_dim}" if spec.comm != "none" else "")
            + (" over comm_mask()" if algo.use_mask else "")
        )
    if cfg.scen_kwargs:
        print(f"  scenario kwargs: {cfg.scen_kwargs}")

    for it in range(spec.iters):
        if gated and clock >= end_clock:
            print(f"ramp clock reached {end_clock} (last ramp end + hold_iters) at iter {it}")
            break
        t0 = time.perf_counter()
        if spec.entropy_coeff_final is not None and spec.iters > 1:
            frac = it / (spec.iters - 1)
            coeff = spec.entropy_coeff + frac * (spec.entropy_coeff_final - spec.entropy_coeff)
        else:
            coeff = spec.entropy_coeff
        if not gated:
            clock = it
        progress = set_ramps(clock)
        # The guard's ramp position: every ramp's progress, not the least of them, which
        # stays 0 until the last ramp starts (with a turn-away cost ramping in at 8000, that
        # held every condition before it to the clean world's best, and rolled a Limo run
        # back three times).
        position = tuple(round(r.progress(clock) * 1000) for r in spec.ramps)
        tick = clock  # the clock this iteration ran at
        lr = spec.lr
        if spec.lr_final is not None and gated:
            frac = min(1.0, clock / max(end_clock, 1))
            lr = spec.lr + frac * (spec.lr_final - spec.lr)
        elif spec.lr_final is not None and spec.iters > 1:
            frac = it / (spec.iters - 1)
            lr = spec.lr + frac * (spec.lr_final - spec.lr)
        if spec.lr_final is not None or spec.rollback or spec.adaptive_kl is not None:
            group = algo.optim.param_groups[0]
            if isinstance(group["lr"], torch.Tensor):
                group["lr"].fill_(lr * lr_mult * kl_mult)
            else:
                group["lr"] = lr * lr_mult * kl_mult
        if algo.action_rate and spec.action_rate_ramp_iters > 0:
            ramp = min(1.0, (it + 1) / spec.action_rate_ramp_iters)
            algo.rate_coef.fill_(spec.action_rate_coef * ramp)
        algo.rollout()
        stats = algo.update(coeff)  # the iteration's one host sync
        wall = time.perf_counter() - t0
        if it == 0:
            print(f"  env graph_mode {env.graph_mode}")

        mean_reward, n_term, n_done = stats["mean_reward"], stats["n_terminated"], stats["n_done"]
        obj, crit, ent = stats["loss_objective"], stats["loss_critic"], stats["entropy"]
        skips = int(stats["grad_skips"])
        kl_stops = int(stats.get("kl_stops", 0))
        updates = max(n_mb - skips - kl_stops, 1)
        window.update_counts(int(n_term), int(n_done), frames_per_batch)
        if spec.adaptive_kl is not None:
            kl_lr = kl_mult  # the factor this iteration trained at, for the log
            kl_mult = adapt_kl_lr(
                kl_mult, stats["approx_kl"] / max(n_mb - skips, 1), spec.adaptive_kl
            )
        total_skips += skips
        hot_skip_iters = hot_skip_iters + 1 if skips / n_mb > cfg.max_skip_frac else 0
        if hot_skip_iters >= cfg.max_skip_iters:
            save("nan")
            csv_file.close()
            raise NonFiniteGradients(
                f"{skips}/{n_mb} minibatches had non-finite gradients for "
                f"{hot_skip_iters} consecutive iterations at iter {it}; state saved to "
                f"{ckpt_dir / f'{cfg.scenario}_nan.pt'}."
            )

        rate, ep_len = window.solve_rate, window.ep_len_mean
        f = set_curriculum(rate)  # difficulty for the next batch's resets
        if gated and (clock < first_start or (rate is not None and rate >= spec.ramp_gate)):
            clock += 1
        if guard is not None:
            action = guard.step(it, rate, position)
            if action == "snapshot":
                snaps[it] = (algo.snapshot(), clock, difficulty)
            elif action == "rollback":
                weights, clock, difficulty = snaps[guard.restore_it]
                lr_mult = 0.5 ** guard.rollbacks  # halved per rollback, for the rest of the run
                algo.restore(weights)
                set_curriculum(None)  # re-applies the restored difficulty
                set_ramps(clock)
                algo.reset()  # fresh episodes under the restored policy and conditions
                window = SolveRateWindow(window=cfg.gate_window)
                print(
                    f"  ROLLBACK {guard.rollbacks}/{guard.max_rollbacks} at iter {it}: solve "
                    f"{rate:.3f} collapsed; restored the iter {guard.restore_it} snapshot "
                    f"(ramp clock {clock}), lr x{lr_mult:g}"
                )
            elif action == "stop":
                save("collapsed")
                csv_file.close()
                raise RollbackLimit(
                    f"the solve rate collapsed again at iter {it} after "
                    f"{guard.rollbacks} rollbacks; the best checkpoint is "
                    f"{ckpt_dir / f'{cfg.scenario}_best.pt'}, the collapsed state "
                    f"{ckpt_dir / f'{cfg.scenario}_collapsed.pt'}."
                )
            for k in [k for k in snaps if k not in guard.snaps]:
                del snaps[k]

        eval_rate = eval_lo = eval_hi = None
        if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
            res = score(
                cfg.scenario, **greedy_score_kwargs(actor), n_agents=n_agents,
                n_envs=cfg.eval_envs, steps=spec.max_steps, device=device,
                seed=cfg.seed + 1_000, dt=spec.dt, substeps=spec.substeps,
                scen_kwargs=cfg.scen_kwargs, arms=("policy",),
            )["policy"]
            eval_rate, (eval_lo, eval_hi) = res.solve_rate, res.ci
            print(
                f"  eval @ iter {it + 1}: greedy solve {eval_rate:.3f} "
                f"[{eval_lo:.3f}, {eval_hi:.3f}] over {cfg.eval_envs} envs "
                f"(target {target:.2f})"
            )

        csv_writer.writerow(
            [it, (it + 1) * frames_per_batch, f"{wall:.3f}", f"{frames_per_batch / wall:.0f}",
             f"{mean_reward:.6f}", "" if rate is None else f"{rate:.4f}", window.ends,
             "" if ep_len is None else f"{ep_len:.1f}", f"{f:.3f}",
             *(f"{progress:.3f}" for _ in ramp_col), *(str(tick) for _ in clock_col),
             f"{ent / updates:.4f}", f"{obj / updates:.4f}", f"{crit / updates:.4f}",
             *(f"{stats[k] / updates:.4f}" for k in aux_col), skips,
             *([f"{stats['approx_kl'] / max(n_mb - skips, 1):.5f}",
                f"{kl_stops / n_mb:.3f}"] if kl_col else []),
             *([guard.rollbacks, f"{lr_mult:g}"] if rollback_col else []),
             *([f"{kl_lr:.4g}"] if adapt_col else []),
             "" if eval_rate is None else f"{eval_rate:.4f}",
             "" if eval_lo is None else f"{eval_lo:.4f}",
             "" if eval_hi is None else f"{eval_hi:.4f}"]
            + [f"{stats[k]:.6f}" for k in metric_cols]
        )
        csv_file.flush()
        rate_txt = "  --  " if rate is None else f"{rate:.3f}"
        print(
            f"iter {it:4d}  reward/step {mean_reward:+.4f}  solve {rate_txt}  "
            f"ends {window.ends:5d}  ep_len {0.0 if ep_len is None else ep_len:6.1f}  "
            f"diff {f:.2f}  "
            + (f"ramp {progress:.2f}  " if ramp_col else "")
            + (f"clock {tick}  " if clock_col else "")
            + (f"kl {stats['approx_kl'] / max(n_mb - skips, 1):.4f}  " if kl_col else "")
            + (f"lr x{kl_lr:.3g}  " if adapt_col else "")
            + f"ent {ent / updates:+.3f}  {wall * 1e3:.1f}ms"
            + (f"  skipped {skips}" if skips else "")
        )
        if cfg.checkpoint_every and (it + 1) % cfg.checkpoint_every == 0:
            save(f"iter{it + 1:05d}")
        if rate is not None and (f, progress, rate) > best_key:
            best_key = (f, progress, rate)
            save("best")

    final = save("final")
    csv_file.close()
    rate = window.solve_rate
    print(f"\ncheckpoints + metrics.csv in {ckpt_dir}/")
    if best_key[0] >= 0.0:
        print(
            f"best checkpoint: solve {best_key[2]:.3f} at difficulty {best_key[0]:.2f} "
            f"-> {ckpt_dir / f'{cfg.scenario}_best.pt'}"
        )
    if rate is not None:
        k = int(round(rate * window.ends))
        ci = wilson_ci(k, window.ends)
        verdict = "MET" if rate >= target else "NOT met"
        print(
            f"final windowed ep_solve_rate {rate:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] "
            f"over {window.ends} episodes -- target {target:.2f} {verdict}. "
            f"This is under the stochastic policy; run eval_mappo.py for the greedy number."
        )
    if total_skips:
        print(f"non-finite-gradient minibatches skipped over the run: {total_skips}")
    return final
