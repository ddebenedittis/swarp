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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from swarp import Environment
from swarp.rl.networks import Actor, Critic, tanh_normal_log_prob
from swarp.rl.recipes import (
    SPECS,
    PolicyArm,
    SolveRateWindow,
    TrainSpec,
    build_env,
    coerce_scen_kwargs,
    score,
    wilson_ci,
)

CHECKPOINT_FORMAT = "swarp.rl/1"
_CLIP_EPS = 0.2
_MAX_GRAD_NORM = 1.0
# torchrl's ``_standardize`` default: float32 resolution.
_ADV_EPS = torch.finfo(torch.float32).resolution


class NonFiniteGradients(RuntimeError):
    """Raised when the run is producing non-finite gradients faster than it can learn."""


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


def ppo_loss(
    actor: Actor,
    critic: Critic,
    obs: torch.Tensor,
    u: torch.Tensor,
    logp_old: torch.Tensor,
    adv: torch.Tensor,
    ret: torch.Tensor,
    noise: torch.Tensor,
    normalize_advantage: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clipped surrogate, smooth-L1 value loss and MC entropy for one minibatch.

    ``obs`` is ``[B, A, obs_dim]``; ``u`` (the stored pre-tanh sample) and ``noise`` (a
    standard normal draw for the entropy's reparametrized sample) are ``[B, A, act_dim]``;
    the rest are ``[B, A]``. Returns ``(objective, critic, entropy)``.
    """
    if normalize_advantage:
        # Per agent: statistics over the minibatch, agent dim excluded.
        adv = (adv - adv.mean(0)) / adv.std(0).clamp_min(_ADV_EPS)
    loc, scale = actor(obs)
    log_w = tanh_normal_log_prob(u, loc, scale) - logp_old
    gain = torch.minimum(
        log_w.exp() * adv,
        log_w.clamp(math.log1p(-_CLIP_EPS), math.log1p(_CLIP_EPS)).exp() * adv,
    )
    entropy = -tanh_normal_log_prob(loc + scale * noise, loc, scale).mean()
    return -gain.mean(), F.smooth_l1_loss(critic(obs), ret), entropy


# ------------------------------------------------------------------------------ the core


class MAPPO:
    """Rollout buffers, rollout, GAE and the PPO update over one ``Environment``.

    No logging, no checkpoints, no curriculum: :func:`train` owns those. ``env`` must
    have ``auto_reset=True`` and ``clone_outputs=False``: the captured graphs read the
    env's output buffers by address, which is what makes them fixed.
    """

    def __init__(
        self,
        env: Environment,
        actor: Actor,
        critic: Critic,
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
        self.t = z(1, dtype=torch.long)
        self.metric_acc = z(sum(self.metrics.values()))
        # Update.
        self.v_buf = z(T + 1, E, A)
        self.adv_buf = z(T, E, A)
        self.ret_buf = z(T, E, A)
        self.perms = z(spec.epochs * self.M, self.B, dtype=torch.long)
        self.k = z(1, dtype=torch.long)
        self.ent_coef = torch.tensor(spec.entropy_coeff, device=dev)
        # [mean reward, n_terminated, n_done] from prepare, then
        # [objective, critic, entropy, skips] summed over the minibatch steps.
        self.prep_stats = z(3)
        self.mb_stats = z(4)
        self.found_inf = z(())
        self.force_skip = z(())

        params = [*actor.parameters(), *critic.parameters()]
        self.params = params
        # Inductor fuses the loss's elementwise work (bias/tanh/log-prob chains) around the
        # cuBLAS GEMMs; the result is still captured by the minibatch graph below.
        self._loss = torch.compile(ppo_loss, dynamic=False) if compile else ppo_loss
        if self.cuda:
            # fused + capturable: a capturable step, and a fused kernel that honours
            # ``found_inf`` by skipping the update and undoing the step increment. Eager
            # CUDA uses it too, so the skip never syncs.
            self.optim = torch.optim.Adam(params, lr=spec.lr, fused=True, capturable=True)
            self.optim.found_inf = self.found_inf
            self.optim.grad_scale = None
        else:
            self.optim = torch.optim.Adam(params, lr=spec.lr)

        self._out: tuple[Any, ...] | None = None
        self._obs_ptr = 0  # the obs buffer the act graph reads
        self._sig: tuple[int, ...] | None = None  # the buffers the store graph reads
        self._act = _Graphed(self._act_body, self.graphs, carries=(self.t,))
        self._store = _Graphed(
            self._store_body, self.graphs, carries=(self.t, self.metric_acc)
        )
        self._prepare = _Graphed(self._prepare_body, self.graphs, carries=(), warmup=1)
        self._minibatch = _Graphed(
            self._minibatch_body, self.graphs, carries=(self.k, self.mb_stats),
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
        """
        obs = self.env.reset()
        horizon = self.env.max_steps or 0
        if self.spec.stagger_starts and horizon > 1:
            gen = torch.Generator(device=self.device).manual_seed(0)
            offset = torch.randint(0, horizon, (self.E,), device=self.device, generator=gen)
            with torch.no_grad():
                for t in range(horizon):
                    obs = self.env.step(self.actor.greedy(obs))[0]
                    obs = self.env.reset_at(offset == t)
        self._out = (obs, None, None, None, None)

    def _signature(self, out: tuple[Any, ...]) -> tuple[int, ...]:
        _, rew, term, trunc, info = out
        return (
            rew.data_ptr(), term.data_ptr(), trunc.data_ptr(),
            *(info[k].data_ptr() for k in self.metrics),
        )

    def _act_body(self) -> None:
        obs = self._out[0]
        self.obs_buf.index_copy_(0, self.t, obs.unsqueeze(0))
        loc, scale = self.actor(obs)
        u = loc + scale * torch.randn_like(loc)
        self.actions.copy_(self.actor.squash(u))
        self.u_buf.index_copy_(0, self.t, u.unsqueeze(0))
        self.logp_buf.index_copy_(0, self.t, tanh_normal_log_prob(u, loc, scale).unsqueeze(0))

    def _store_body(self) -> None:
        _, rew, term, trunc, info = self._out
        self.rew_buf.index_copy_(0, self.t, rew.unsqueeze(0))
        self.term_buf.index_copy_(0, self.t, term.unsqueeze(0).float())
        self.trunc_buf.index_copy_(0, self.t, trunc.unsqueeze(0).float())
        off = 0
        for key, width in self.metrics.items():
            x = info[key].float().reshape(-1, width).mean(0)
            self.metric_acc[off : off + width] += x
            off += width
        self.t.add_(1)

    @torch.no_grad()
    def rollout(self) -> None:
        """``steps_per_batch`` env steps into the rollout buffers."""
        if self._out is None:
            self.reset()
        self.t.zero_()
        self.metric_acc.zero_()
        env, actions = self.env, self.actions
        for _ in range(self.T):
            # The graphs read the env's output buffers by address. They are fixed in
            # steady state, but a first step or a reallocated buffer means a recapture.
            ptr = self._out[0].data_ptr()
            if ptr != self._obs_ptr:
                self._act.invalidate()
                self._obs_ptr = ptr
            self._act()
            out = env.step(actions)
            sig = self._signature(out)
            if sig != self._sig:
                self._store.invalidate()
                self._sig = sig
            self._out = out
            self._store()
        self.obs_buf[self.T].copy_(self._out[0])

    # ---------------------------------------------------------------------- update

    def _prepare_body(self) -> None:
        spec, T = self.spec, self.T
        gamma, lmbda = spec.gamma, spec.lmbda
        with torch.no_grad():
            for t in range(T + 1):  # per step, so peak memory is one step's activations
                self.v_buf[t] = self.critic(self.obs_buf[t])
            term = self.term_buf
            gae(self.rew_buf, self.v_buf, term, self.trunc_buf, gamma, lmbda, out=self.adv_buf)
            torch.add(self.adv_buf, self.v_buf[:T], out=self.ret_buf)
            n = self.M * self.B
            for e in range(spec.epochs):
                p = torch.rand(self.N, device=self.device).argsort()[:n]
                self.perms[e * self.M : (e + 1) * self.M] = p.view(self.M, self.B)
            self.prep_stats[0] = self.rew_buf.mean()
            self.prep_stats[1] = term.sum()
            self.prep_stats[2] = torch.maximum(term, self.trunc_buf).sum()
            self.mb_stats.zero_()
            self.k.zero_()

    def _minibatch_body(self) -> None:
        A = self.A
        idx = self.perms.index_select(0, self.k).view(-1)
        obs = self.obs_buf[: self.T].reshape(self.N, A, -1).index_select(0, idx)
        u = self.u_buf.reshape(self.N, A, -1).index_select(0, idx)
        logp_old = self.logp_buf.reshape(self.N, A).index_select(0, idx)
        adv = self.adv_buf.reshape(self.N, A).index_select(0, idx)
        ret = self.ret_buf.reshape(self.N, A).index_select(0, idx)
        noise = torch.randn_like(u)  # drawn here, not in the (possibly compiled) loss
        loss_obj, loss_crit, entropy = self._loss(
            self.actor, self.critic, obs, u, logp_old, adv, ret, noise,
            self.spec.normalize_advantage,
        )
        loss_ent = -self.ent_coef * entropy

        self.optim.zero_grad(set_to_none=False)
        (loss_obj + loss_crit + loss_ent).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            self.params, _MAX_GRAD_NORM, foreach=self.cuda or None
        )
        if self.cuda:
            found = torch.maximum((~torch.isfinite(gnorm)).float(), self.force_skip)
            self.found_inf.copy_(found)
            self.optim.step()
        else:
            found = (~torch.isfinite(gnorm)).float()
            if not found.item():
                self.optim.step()
        kept = torch.stack([loss_obj, loss_crit, entropy]).detach()
        self.mb_stats[:3] += torch.where(found > 0, torch.zeros_like(kept), kept)
        self.mb_stats[3] += found
        self.k.add_(1)

    def update(self, entropy_coeff: float) -> list[float]:
        """One PPO update over the last rollout. Returns the per-iteration statistics:
        ``[mean reward, n_terminated, n_done, objective, critic, entropy, skips,
        *metric means]``, sums over minibatches for the losses. One host sync."""
        self.ent_coef.fill_(entropy_coeff)
        self._prepare()
        for _ in range(self.spec.epochs * self.M):
            self._minibatch()
        stats = torch.cat([self.prep_stats, self.mb_stats, self.metric_acc / self.T])
        return stats.tolist()


# ------------------------------------------------------------------------- checkpoints


def load_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[Actor, Critic, dict[str, Any]]:
    """Actor, critic and metadata from a :func:`train` checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    actor = Actor(
        ckpt["obs_dim"], ckpt["act_dim"], ckpt["num_cells"],
        low=ckpt.get("act_low"), high=ckpt.get("act_high"),
    ).to(device)
    critic = Critic(ckpt["obs_dim"], ckpt["n_agents"], ckpt["num_cells"]).to(device)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    return actor.eval(), critic.eval(), ckpt


def load_policy(path: str | Path, device: str = "cpu") -> Actor:
    """The actor of a :func:`train` checkpoint; ``actor.greedy(obs)`` is the eval action."""
    return load_checkpoint(path, device)[0]


def greedy_arm(actor: Actor) -> PolicyArm:
    """The policy arm of :func:`swarp.rl.recipes.score` for a :class:`Actor`."""
    return lambda n_envs, device, lo, hi: actor.greedy


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
    p.add_argument("--gamma", type=float)
    p.add_argument("--lmbda", type=float)
    p.add_argument("--entropy-coeff", type=float)
    p.add_argument("--entropy-coeff-final", type=float, help="linear schedule target")
    p.add_argument("--num-cells", type=int, help="MLP width")
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
            "steps_per_batch", "epochs", "minibatches", "lr", "gamma", "lmbda",
            "entropy_coeff", "entropy_coeff_final", "num_cells", "normalize_advantage",
            "curriculum_iters", "curriculum_gate",
        )
        if getattr(a, f) is not None
    }
    spec = replace(spec, **overrides)
    if spec.n_agents < 1:
        p.error("--n-agents must be >= 1")
    scen_kwargs = dict(spec.scen_kwargs)
    try:
        scen_kwargs.update(coerce_scen_kwargs(a.scenario, a.scen_kwarg))
    except (ValueError, KeyError) as exc:
        p.error(str(exc))

    return TrainConfig(
        scenario=a.scenario, spec=spec, scen_kwargs=scen_kwargs, device=a.device,
        checkpoint_dir=Path(a.checkpoint_dir or f"runs/{a.scenario}"), resume=a.resume,
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
    # A multiobj_reward key is [n_envs, n_agents, n_obj]; each term gets its own column.
    metrics = {
        k: (info0[k].shape[-1] if k == "multiobj_reward" else 1) for k in spec.metrics
    }
    metric_cols: list[str] = []
    for k, w in metrics.items():
        metric_cols += [f"rew_term{i}" for i in range(w)] if k == "multiobj_reward" else [k]

    actor = Actor(obs_dim, act_dim, spec.num_cells, low=act_low, high=act_high).to(device)
    critic = Critic(obs_dim, n_agents, spec.num_cells).to(device)
    if cfg.resume:
        a, c, _ = load_checkpoint(cfg.resume, device)
        actor.load_state_dict(a.state_dict())
        critic.load_state_dict(c.state_dict())
        print(f"resumed actor + critic from {cfg.resume}")
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

    set_curriculum(None)
    algo = MAPPO(
        env, actor, critic, spec, metrics=metrics, graphs=cfg.graphs, compile=cfg.compile
    )
    algo.reset()  # after the knobs, so the first episodes run at the starting difficulty

    ckpt_dir = cfg.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_file = (ckpt_dir / "metrics.csv").open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["iter", "frames", "wall_s", "env_sps", "reward_per_step", "ep_solve_rate",
         "ep_ends", "ep_len_mean", "difficulty", "entropy", "loss_objective",
         "loss_critic", "grad_skips", "eval_solve_rate", "eval_ci_lo", "eval_ci_hi"]
        + metric_cols
    )
    target = spec.target_solve_rate if cfg.target_solve_rate is None else cfg.target_solve_rate

    def save(tag: str) -> Path:
        path = ckpt_dir / f"{cfg.scenario}_{tag}.pt"
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
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
    # Ranked on (difficulty, solve rate): a later checkpoint wins only at an equal or
    # harder difficulty (see train_mappo.py for the caging run that motivated it).
    best_key = (-1.0, -1.0)
    print(
        f"{cfg.scenario}: {n_agents} agents, obs_dim {obs_dim}, act_dim {act_dim}, "
        f"{spec.n_envs} envs x {spec.steps_per_batch} steps, dt {spec.dt}, "
        f"substeps {spec.substeps}, max_steps {spec.max_steps}, target {target:.2f}, "
        f"graphs {'on' if algo.graphs else 'off'}, tf32 {'on' if cfg.tf32 else 'off'}, "
        f"compile {'on' if cfg.compile else 'off'}"
    )
    if cfg.scen_kwargs:
        print(f"  scenario kwargs: {cfg.scen_kwargs}")

    for it in range(spec.iters):
        t0 = time.perf_counter()
        if spec.entropy_coeff_final is not None and spec.iters > 1:
            frac = it / (spec.iters - 1)
            coeff = spec.entropy_coeff + frac * (spec.entropy_coeff_final - spec.entropy_coeff)
        else:
            coeff = spec.entropy_coeff
        algo.rollout()
        stats = algo.update(coeff)  # the iteration's one host sync
        wall = time.perf_counter() - t0
        if it == 0:
            print(f"  env graph_mode {env.graph_mode}")

        mean_reward, n_term, n_done, obj, crit, ent, skips = stats[:7]
        skips = int(skips)
        updates = max(n_mb - skips, 1)
        window.update_counts(int(n_term), int(n_done), frames_per_batch)
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

        eval_rate = eval_lo = eval_hi = None
        if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
            res = score(
                cfg.scenario, policy=greedy_arm(actor), n_agents=n_agents,
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
             f"{ent / updates:.4f}", f"{obj / updates:.4f}", f"{crit / updates:.4f}", skips,
             "" if eval_rate is None else f"{eval_rate:.4f}",
             "" if eval_lo is None else f"{eval_lo:.4f}",
             "" if eval_hi is None else f"{eval_hi:.4f}"]
            + [f"{x:.6f}" for x in stats[7:]]
        )
        csv_file.flush()
        rate_txt = "  --  " if rate is None else f"{rate:.3f}"
        print(
            f"iter {it:4d}  reward/step {mean_reward:+.4f}  solve {rate_txt}  "
            f"ends {window.ends:5d}  ep_len {0.0 if ep_len is None else ep_len:6.1f}  "
            f"diff {f:.2f}  ent {ent / updates:+.3f}  {wall * 1e3:.1f}ms"
            + (f"  skipped {skips}" if skips else "")
        )
        if cfg.checkpoint_every and (it + 1) % cfg.checkpoint_every == 0:
            save(f"iter{it + 1:05d}")
        if rate is not None and (f, rate) > best_key:
            best_key = (f, rate)
            save("best")

    final = save("final")
    csv_file.close()
    rate = window.solve_rate
    print(f"\ncheckpoints + metrics.csv in {ckpt_dir}/")
    if best_key[0] >= 0.0:
        print(
            f"best checkpoint: solve {best_key[1]:.3f} at difficulty {best_key[0]:.2f} "
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
