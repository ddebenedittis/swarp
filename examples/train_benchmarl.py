"""MAPPO on any swarp scenario through BenchMARL's own ``Experiment``.

A baseline for the end-to-end training comparison: a standard BenchMARL run (its task
class, ``MappoConfig``, ``MlpConfig``, ``ExperimentConfig``, collector, losses, optimizers
and loop, as shipped), configured to match the ``TrainSpec`` recipes as closely as its
configuration allows. Everything swarp-specific is a BenchMARL extension point:

* :class:`SwarpClass` / :class:`SwarpTask` -- a task registration in the style of BenchMARL's
  VMAS tasks. swarp is natively batched, so ``get_env_fun`` returns one env of ``num_envs``.
* :class:`GroupedSwarpEnv` -- ``SwarpEnv`` with the per-agent entries nested under one
  ``"agents"`` group, the layout BenchMARL expects (torchrl's ``VmasEnv``'s).
* :class:`SwarpCallback` -- ``metrics.csv``, checkpoints, the curriculum gate and the
  entropy schedule.

BenchMARL pins ``torchrl < 0.12``, so it gets its own venv (the lock resolves the
``benchmarl`` extra separately from ``torchrl``)::

    UV_PROJECT_ENVIRONMENT=.venv-benchmarl uv sync --extra benchmarl
    .venv-benchmarl/bin/python examples/train_benchmarl.py --scenario navigation

and ``swarp.benchmark.train_throughput`` runs it with ``--python
benchmarl=.venv-benchmarl/bin/python``. Evaluate a checkpoint (from any venv with swarp,
benchmarl not needed) with::

    python examples/eval_mappo.py runs/navigation_benchmarl/navigation_final.pt

Configuration. ``MlpConfig``: ``[num_cells, num_cells]``, Tanh, ``nn.Linear`` for both the
actor and the critic. ``MappoConfig``: shared critic, centralised over the group's joint
observation (no state spec, as for VMAS), clip 0.2, smooth-L1 value loss with coefficient
1.0, ``use_tanh_normal`` with ``biased_softplus_1.0`` scale mapping, GAE over the whole batch.
``ExperimentConfig``: shared policy parameters, ``frames_per_batch = n_envs x
steps_per_batch`` collected from one batched env of ``n_envs``, ``epochs`` passes
(``on_policy_n_minibatch_iters``) of ``minibatches`` minibatches each, Adam with torch's
default ``eps=1e-8`` (BenchMARL's default is 1e-6), grad-norm clip 1.0, sampling, training
and buffer devices all ``--device``, no loggers, no rendering, no BenchMARL checkpoints,
evaluation only with ``--eval-every``.

Deviations from our MAPPO (``swarp.rl`` and ``examples/train_mappo.py``):

* **No advantage normalization.** BenchMARL's ``Mappo`` builds ``ClipPPOLoss`` with
  ``normalize_advantage=False``, hard-coded rather than configurable. Our specs normalize
  per agent everywhere except Push-T.
* **Gradients are clipped per network.** BenchMARL keeps one Adam per loss term (actor:
  objective + entropy; critic) and clips each one's gradient norm to 1.0 separately; ours
  clips the joint norm over both. (Two Adams over disjoint parameters are otherwise one.)
* **No non-finite-gradient skip and no abort.** BenchMARL steps every minibatch. The
  ``grad_nonfinite`` column counts minibatches whose gradient norm was non-finite.
* **The collector runs on torch's current stream.** BenchMARL's ``SyncDataCollector``
  otherwise steps the env, and the training loop, on a side stream that swarp's Warp
  launches are not ordered against (the race ``train_mappo.py`` documents), so the callback
  sets ``collector.no_cuda_sync = True`` before the first rollout. That also drops the
  collector's per-step device syncs, which are no-ops on one device.
* **Curriculum and entropy schedule are callbacks.** BenchMARL has neither. The gate runs
  once per iteration after training, like ours; the starting knobs are applied by
  re-resetting the collector in ``on_setup``, because the collector resets the env when it
  is built, before any callback runs. A ``TrainSpec.entropy_coeff_final`` is written into
  the loss's ``entropy_coeff`` buffer before each update.
* **Timeouts bootstrap from** ``V(s_{t+1})``, the true next observation (``SwarpEnv``
  resets through the collector), as in ``train_mappo.py``; ``swarp.rl`` uses ``V(s_t)``.
* **Log-probs and entropy go through the action.** torchrl's ``TanhNormal`` re-derives the
  pre-tanh sample with ``atanh`` and estimates entropy from one sample, as in
  ``train_mappo.py``; ``swarp.rl`` stores the pre-tanh sample.
* **Extra per-step work, as shipped:** the ``RewardSum`` transform (episode returns) and a
  clone of every collected batch. BenchMARL's per-iteration ``log_collection`` statistics
  also run, but outside ``wall_s``.
* **Evaluation is BenchMARL's** deterministic rollout of ``--eval-envs`` envs for
  ``max_steps`` (and at iteration 0 too); the solve rate is read from its rollouts. It needs
  ``create_json``, so an evaluating run also writes BenchMARL's marl-eval json.

``wall_s`` is BenchMARL's own ``timers/collection_time + timers/training_time``: the rollout,
then GAE, the buffer fill, the update and the policy-weight sync, with a device sync added at
the end of training so the last optimizer step is inside it. BenchMARL's collection
statistics, evaluation, our logging and checkpoints are outside it. The greedy action of a
saved checkpoint is ``tanh(loc)``, BenchMARL's own deterministic sample (``TanhNormal`` has
no analytic mode).
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from benchmarl.algorithms import MappoConfig
from benchmarl.environments import Task, TaskClass
from benchmarl.experiment import Callback, Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase
from torchrl.modules import MultiAgentMLP
from tqdm import tqdm

import swarp
from swarp.benchmark.ablation import sync_device
from swarp.interop.torchrl import SwarpEnv
from swarp.rl.ppo import check_action_box
from swarp.rl.recipes import (
    SPECS,
    UNTRAINABLE,
    SolveRateWindow,
    TrainSpec,
    coerce_scen_kwargs,
    scenario_name,
    wilson_ci,
)

CHECKPOINT_FORMAT = "benchmarl/1"
GROUP = "agents"

# ----------------------------------------------------------------------------- the env


class GroupedSwarpEnv(SwarpEnv):
    """:class:`SwarpEnv` with the per-agent entries nested under one ``"agents"`` group.

    BenchMARL reads every per-agent tensor at ``(group, key)`` with the group's shape
    ``[n_envs, n_agents]``, the layout of torchrl's ``VmasEnv``; ``SwarpEnv`` is flat.
    Observation, action, reward and the scenario's ``info()`` move under the group, and
    ``done``/``terminated``/``truncated`` stay at the root, one per env, as in VMAS. Info
    under the group is what keeps it out of BenchMARL's training batch.
    """

    def _make_specs(
        self,
        low: float | torch.Tensor,
        high: float | torch.Tensor,
        dtype: torch.dtype,
        info: dict[str, torch.Tensor],
    ) -> None:
        ne, na, dev = self._env.n_envs, self.n_agents, self.device
        low, high = self._action_bound(low, dtype), self._action_bound(high, dtype)

        def group(**specs) -> Composite:
            return Composite(**specs, shape=(ne, na), device=dev)

        def root(**specs) -> Composite:
            return Composite(**specs, shape=(ne,), device=dev)

        agents = group(
            observation=Unbounded(shape=(ne, na, self.obs_dim), dtype=dtype, device=dev)
        )
        if info:
            agents["info"] = group(
                **{
                    k: Unbounded(shape=tuple(v.shape), dtype=v.dtype, device=dev)
                    for k, v in self._group_info(info).items()
                }
            )
        self.observation_spec = root(**{GROUP: agents})
        self.action_spec = root(
            **{GROUP: group(action=Bounded(
                low=low, high=high, shape=(ne, na, self.act_dim), dtype=dtype, device=dev
            ))}
        )
        self.reward_spec = root(
            **{GROUP: group(reward=Unbounded(shape=(ne, na, 1), dtype=dtype, device=dev))}
        )
        self.done_spec = root(
            **{
                key: Categorical(n=2, shape=(ne, 1), dtype=torch.bool, device=dev)
                for key in ("done", "terminated", "truncated")
            }
        )

    def _group_info(self, info: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Per-env info (``[n_envs, ...]``) broadcast over the agent dim.

        Materialized rather than an expanded view: the collector updates its carrier in
        place, and a write into a stride-0 tensor is rejected. A per-env tensor whose second
        dim happens to equal ``n_agents`` reads as per-agent; only the layout differs, every
        logged mean is the same.
        """
        ne, na = self._env.n_envs, self.n_agents
        return {
            k: v if v.ndim >= 2 and v.shape[1] == na
            else v.reshape(ne, 1, *v.shape[1:]).expand(ne, na, *v.shape[1:]).contiguous()
            for k, v in info.items()
        }

    def _info_td(self, info: dict[str, torch.Tensor], *, copy: bool = True) -> TensorDict:
        # Per-agent entries are cloned when the env did not already; broadcast per-env ones
        # are fresh tensors either way.
        picked = {k: info[k] for k in self._info_keys}
        grouped = self._group_info(picked)
        if copy:
            grouped = {k: v.clone() if v is picked[k] else v for k, v in grouped.items()}
        return TensorDict(
            grouped, batch_size=(*self.batch_size, self.n_agents), device=self.device
        )

    def _nest(self, out: TensorDict, *keys: str) -> TensorDict:
        keys = (*keys, "info") if self._info_keys else keys
        out.set(
            GROUP,
            TensorDict(
                {k: out.pop(k) for k in keys}, batch_size=(*self.batch_size, self.n_agents),
                device=self.device,
            ),
        )
        return out

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        return self._nest(super()._reset(tensordict, **kwargs), "observation")

    def _step(self, tensordict: TensorDict) -> TensorDict:
        flat = TensorDict(
            {"action": tensordict.get((GROUP, "action"))}, batch_size=self.batch_size,
            device=self.device,
        )
        return self._nest(super()._step(flat), "observation", "reward")


# ---------------------------------------------------------------------------- the task


class SwarpClass(TaskClass):
    """A swarp scenario as a BenchMARL task; ``config`` is ``swarp.make``'s keywords.

    ``{"max_steps", "n_agents", "dt", "substeps", **scen_kwargs}``, the shape of a VMAS
    task's yaml (``max_steps`` plus the scenario's own keywords).
    """

    def get_env_fun(
        self, num_envs: int, continuous_actions: bool, seed: int | None, device: str
    ) -> Callable[[], EnvBase]:
        config = dict(self.config)
        return lambda: GroupedSwarpEnv(
            swarp.make(
                scenario_name(self.name.lower()), n_envs=num_envs, device=device, seed=seed,
                **config,
            )
        )

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"]

    def group_map(self, env: EnvBase) -> dict[str, list[str]]:
        return {GROUP: [f"agent_{i}" for i in range(env.n_agents)]}

    def observation_spec(self, env: EnvBase) -> Composite:
        spec = env.full_observation_spec_unbatched.clone()
        if "info" in spec[GROUP]:
            del spec[(GROUP, "info")]
        return spec

    def info_spec(self, env: EnvBase) -> Composite | None:
        spec = env.full_observation_spec_unbatched.clone()
        if "info" not in spec[GROUP]:
            return None
        del spec[(GROUP, "observation")]
        return spec

    def state_spec(self, env: EnvBase) -> Composite | None:
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched

    def action_mask_spec(self, env: EnvBase) -> Composite | None:
        return None

    @staticmethod
    def env_name() -> str:
        return "swarp"


class SwarpTask(Task):
    """The trainable swarp scenarios (``SPECS``), as BenchMARL task enum members.

    There are no yaml task configs: build one with ``SwarpTask.NAVIGATION.get_task(config)``.
    """

    NAVIGATION = None
    FORMATION = None
    DISCOVERY = None
    TRANSPORT = None
    PUSHT = None
    GIVEWAY = None
    SHEPHERDING = None
    CAGING = None

    @staticmethod
    def associated_class() -> type[TaskClass]:
        return SwarpClass


# -------------------------------------------------------------------------- the config


@dataclass
class TrainConfig:
    """A resolved spec plus CLI overrides. All primitives, so it pickles."""

    scenario: str
    spec: TrainSpec
    scen_kwargs: Mapping[str, Any]
    device: str
    checkpoint_dir: Path
    start_difficulty: float = 0.0
    checkpoint_every: int = 250
    eval_every: int = 0
    eval_envs: int = 512
    seed: int = 0
    gate_window: int = 25
    target_solve_rate: float | None = None


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    """Parse into a :class:`TrainConfig`. The flags mirror ``python -m swarp.rl``'s.

    Hyperparameter flags default to ``None``, meaning "take the spec's value".
    """
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser(description="MAPPO on a swarp scenario through BenchMARL.")
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
    p.add_argument("--curriculum-iters", type=int, help="0 disables")
    p.add_argument("--curriculum-gate", type=float)
    p.add_argument("--gate-window", type=int, default=25,
                   help="iterations of episode ends behind ep_solve_rate")
    p.add_argument("--start-difficulty", type=float, default=0.0)
    p.add_argument("--checkpoint-dir")
    p.add_argument("--checkpoint-every", type=int, default=250, help="0 disables")
    p.add_argument("--eval-every", type=int, default=0, help="0 disables")
    p.add_argument("--eval-envs", type=int, default=512)
    p.add_argument("--target-solve-rate", type=float)
    a = p.parse_args(argv)
    if a.scenario in UNTRAINABLE:
        p.error(f"{a.scenario!r} is not trainable: {UNTRAINABLE[a.scenario]}")

    spec = SPECS[a.scenario]
    overrides = {
        f: getattr(a, f)
        for f in (
            "dt", "substeps", "max_steps", "n_agents", "iters", "n_envs",
            "steps_per_batch", "epochs", "minibatches", "lr", "gamma", "lmbda",
            "entropy_coeff", "entropy_coeff_final", "num_cells", "curriculum_iters",
            "curriculum_gate",
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
        checkpoint_dir=Path(a.checkpoint_dir or f"runs/{a.scenario}_benchmarl"),
        start_difficulty=a.start_difficulty, checkpoint_every=a.checkpoint_every,
        eval_every=a.eval_every, eval_envs=a.eval_envs, seed=a.seed,
        gate_window=a.gate_window, target_solve_rate=a.target_solve_rate,
    )


# ---------------------------------------------------------------------- the callback


def actor_state_dict(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The policy's shared ``MultiAgentMLP`` in :class:`swarp.rl.networks.Actor`'s keys.

    BenchMARL's actor (shared-parameter ``MultiAgentMLP``, ``biased_softplus_1.0``
    ``NormalParamExtractor``, ``TanhNormal`` on ``[-1, 1]``) is that module exactly, so the
    evaluator needs neither benchmarl nor this torchrl version.
    """
    mlp = next(m for m in policy.modules() if isinstance(m, MultiAgentMLP))
    return {
        k.replace("params.", "net.", 1): v.detach().clone()
        for k, v in mlp.state_dict().items()
        if isinstance(v, torch.Tensor)
    }


class SwarpCallback(Callback):
    """``metrics.csv``, checkpoints, the curriculum gate and the entropy schedule.

    BenchMARL has no end-of-iteration hook, so :meth:`on_setup` wraps the experiment
    logger's ``log``: the call carrying ``timers/iteration_time`` comes once per iteration,
    after training and evaluation and before BenchMARL's own checkpoint.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.difficulty = min(1.0, max(0.0, cfg.start_difficulty))

    # ------------------------------------------------------------------ setup

    def on_setup(self) -> None:
        exp, cfg, spec = self.experiment, self.cfg, self.cfg.spec
        exp.collector.no_cuda_sync = True  # see the module docstring
        env = exp.collector.env
        while not isinstance(env, GroupedSwarpEnv):
            env = env.base_env
        self.env = env
        self.scen = scen = env._env.scenario
        # The checkpoint's actor is evaluated as tanh(loc) on [-1, 1]; BenchMARL itself
        # trains on any box, but eval_mappo would then act on the wrong one.
        check_action_box(env._env, cfg.scenario)

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
        self.metrics = {
            k: (info0[k].shape[-1] if k == "multiobj_reward" else 1) for k in spec.metrics
        }
        metric_cols: list[str] = []
        for k, w in self.metrics.items():
            metric_cols += [f"rew_term{i}" for i in range(w)] if k == "multiobj_reward" else [k]

        # The collector reset its env when it was built, before any callback ran: set the
        # knobs and reset again, so the first episodes run at the starting difficulty.
        self._set_curriculum(None)
        exp.collector.reset()

        self.frames = spec.n_envs * spec.steps_per_batch
        self.window = SolveRateWindow(window=cfg.gate_window)
        self.best_key = (-1.0, -1.0)
        self.it = 0
        self.eval_res: tuple[float, float, float] | None = None
        self.csv_file = (cfg.checkpoint_dir / "metrics.csv").open("w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(
            ["iter", "frames", "wall_s", "env_sps", "reward_per_step", "ep_solve_rate",
             "ep_ends", "ep_len_mean", "difficulty", "entropy", "loss_objective",
             "loss_critic", "grad_nonfinite", "eval_solve_rate", "eval_ci_lo", "eval_ci_hi"]
            + metric_cols
        )
        log = exp.logger.log

        def log_and_hook(to_log: dict, step: int | None = None) -> None:
            log(to_log, step=step)
            if "timers/iteration_time" in to_log:
                self._end_iteration(
                    to_log["timers/collection_time"] + to_log["timers/training_time"]
                )

        exp.logger.log = log_and_hook

    def _set_curriculum(self, gate_metric: float | None) -> float:
        """Ramp the knobs while the policy keeps solving; a ``None`` gate stays closed."""
        spec = self.cfg.spec
        if not spec.curriculum_iters:
            return 1.0
        if gate_metric is not None and gate_metric >= spec.curriculum_gate:
            self.difficulty = min(1.0, self.difficulty + 1.0 / spec.curriculum_iters)
        for knob in spec.curriculum:
            setattr(self.scen, knob.attr, knob.value(self.difficulty))
        if spec.curriculum_fn is not None:
            spec.curriculum_fn(self.scen, self.difficulty)
        return self.difficulty

    # -------------------------------------------------------------- per iteration

    def on_batch_collected(self, batch: TensorDictBase) -> None:
        """Statistics of the rollout, and the entropy coefficient for this update.

        Runs between BenchMARL's collection and training timers, so its syncs are not
        timed.
        """
        spec, it = self.cfg.spec, self.it
        if spec.entropy_coeff_final is not None and spec.iters > 1:
            frac = it / (spec.iters - 1)
            coeff = spec.entropy_coeff + frac * (spec.entropy_coeff_final - spec.entropy_coeff)
            self.experiment.losses[GROUP].entropy_coeff.fill_(coeff)
        self.window.update(batch.get(("next", "terminated")), batch.get(("next", "done")))
        self.mean_reward = batch.get(("next", GROUP, "reward")).mean().item()
        self.metric_vals: list[float] = []
        for k in self.metrics:
            v = batch.get(("next", GROUP, "info", k)).float()
            if k == "multiobj_reward":
                self.metric_vals += v.reshape(-1, v.shape[-1]).mean(0).tolist()
            else:
                self.metric_vals.append(v.mean().item())

    def on_train_end(self, training_td: TensorDictBase, group: str) -> None:
        # Inside BenchMARL's training timer: finish the last optimizer step before it stops.
        sync_device(self.cfg.device)
        self.training_td = training_td

    def on_evaluation_end(self, rollouts: list[TensorDictBase]) -> None:
        """Solve rate of BenchMARL's deterministic evaluation rollouts, one per env.

        Its logger has already cut each rollout at the first done, so an env is solved iff
        its first episode terminated rather than timed out.
        """
        solved = sum(int(r.get(("next", "terminated")).any()) for r in rollouts)
        lo, hi = wilson_ci(solved, len(rollouts))
        self.eval_res = (solved / len(rollouts), lo, hi)
        tqdm.write(
            f"  eval @ iter {self.it}: greedy solve {self.eval_res[0]:.3f} "
            f"[{lo:.3f}, {hi:.3f}] over {len(rollouts)} envs"
        )

    def _end_iteration(self, wall: float) -> None:
        cfg, it, frames = self.cfg, self.it, self.frames
        td = self.training_td
        means = {
            k: td.get(k).float().mean().item()
            for k in ("entropy", "loss_objective", "loss_critic")
        }
        nonfinite = sum(
            int((~torch.isfinite(td.get(k))).sum().item())
            for k in ("grad_norm_loss_objective", "grad_norm_loss_critic")
        )
        rate, ep_len = self.window.solve_rate, self.window.ep_len_mean
        f = self._set_curriculum(rate)  # difficulty for the next batch's resets
        ev = self.eval_res
        self.csv.writerow(
            [it, (it + 1) * frames, f"{wall:.3f}", f"{frames / wall:.0f}",
             f"{self.mean_reward:.6f}", "" if rate is None else f"{rate:.4f}",
             self.window.ends, "" if ep_len is None else f"{ep_len:.1f}", f"{f:.3f}",
             f"{means['entropy']:.4f}", f"{means['loss_objective']:.4f}",
             f"{means['loss_critic']:.4f}", nonfinite,
             *(("", "", "") if ev is None else (f"{x:.4f}" for x in ev))]
            + [f"{x:.6f}" for x in self.metric_vals]
        )
        self.csv_file.flush()
        self.eval_res = None
        rate_txt = "  --  " if rate is None else f"{rate:.3f}"
        tqdm.write(
            f"iter {it:4d}  reward/step {self.mean_reward:+.4f}  solve {rate_txt}  "
            f"ends {self.window.ends:5d}  ep_len {0.0 if ep_len is None else ep_len:6.1f}  "
            f"diff {f:.2f}  ent {means['entropy']:+.3f}  {wall * 1e3:.1f}ms"
            + (f"  non-finite grads {nonfinite}" if nonfinite else "")
        )
        if cfg.checkpoint_every and (it + 1) % cfg.checkpoint_every == 0:
            self.save(f"iter{it + 1:05d}")
        if rate is not None and (f, rate) > self.best_key:
            self.best_key = (f, rate)
            self.save("best")
        self.it += 1

    # ---------------------------------------------------------------- checkpoints

    def save(self, tag: str) -> Path:
        cfg, spec, env = self.cfg, self.cfg.spec, self.env
        policy = self.experiment.group_policies[GROUP]
        path = cfg.checkpoint_dir / f"{cfg.scenario}_{tag}.pt"
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "actor": actor_state_dict(policy),
                # BenchMARL's own layout, for reloading into a BenchMARL policy.
                "policy": {
                    k: v for k, v in policy.state_dict().items() if isinstance(v, torch.Tensor)
                },
                "obs_dim": env.obs_dim,
                "act_dim": env.act_dim,
                "n_agents": spec.n_agents,
                "num_cells": spec.num_cells,
                "scenario": cfg.scenario,
                "scen_kwargs": {k: str(v) for k, v in cfg.scen_kwargs.items()},
                "dt": spec.dt,
                "substeps": spec.substeps,
                "max_steps": spec.max_steps,
                "difficulty": self.difficulty,
            },
            path,
        )
        return path


# --------------------------------------------------------------------------- training


def build_experiment(cfg: TrainConfig) -> tuple[Experiment, SwarpCallback]:
    """The BenchMARL ``Experiment`` for ``cfg``; builds the envs, does not train."""
    spec, device = cfg.spec, cfg.device
    task = SwarpTask[cfg.scenario.upper()].get_task(
        config={
            "max_steps": spec.max_steps, "n_agents": spec.n_agents, "dt": spec.dt,
            "substeps": spec.substeps, **cfg.scen_kwargs,
        }
    )
    algorithm = MappoConfig(
        share_param_critic=True,
        clip_epsilon=0.2,
        entropy_coef=spec.entropy_coeff,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
        lmbda=spec.lmbda,
        scale_mapping="biased_softplus_1.0",
        use_tanh_normal=True,
        minibatch_advantage=False,
    )
    model = MlpConfig(
        num_cells=[spec.num_cells, spec.num_cells],
        layer_class=torch.nn.Linear,
        activation_class=torch.nn.Tanh,
    )
    frames = spec.n_envs * spec.steps_per_batch
    evaluate = cfg.eval_every > 0
    config = ExperimentConfig.get_from_yaml()
    for key, value in {
        "sampling_device": device, "train_device": device, "buffer_device": device,
        "share_policy_params": True, "prefer_continuous_actions": True,
        "collect_with_grad": False, "parallel_collection": False,
        "gamma": spec.gamma, "lr": spec.lr, "adam_eps": 1e-8, "adam_extra_kwargs": {},
        "clip_grad_norm": True, "clip_grad_val": 1.0,
        "max_n_iters": spec.iters, "max_n_frames": None,
        "on_policy_collected_frames_per_batch": frames,
        "on_policy_n_envs_per_worker": spec.n_envs,
        "on_policy_n_minibatch_iters": spec.epochs,
        "on_policy_minibatch_size": frames // spec.minibatches,
        "evaluation": evaluate, "render": False,
        "evaluation_interval": max(cfg.eval_every, 1) * frames,
        # The eval env is built even with evaluation off (its specs feed the models).
        "evaluation_episodes": cfg.eval_envs if evaluate else 1,
        "evaluation_deterministic_actions": True, "evaluation_static": True,
        "loggers": [], "create_json": evaluate,
        "save_folder": str(cfg.checkpoint_dir), "restore_file": None,
        "checkpoint_interval": 0, "checkpoint_at_end": False,
        "keep_checkpoints_num": None, "exclude_buffer_from_checkpoint": True,
    }.items():
        setattr(config, key, value)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callback = SwarpCallback(cfg)
    experiment = Experiment(
        task=task, algorithm_config=algorithm, model_config=model, seed=cfg.seed,
        config=config, callbacks=[callback],
    )
    return experiment, callback


def train(cfg: TrainConfig) -> Path:
    """Run the experiment. Returns the final checkpoint path."""
    spec = cfg.spec
    target = spec.target_solve_rate if cfg.target_solve_rate is None else cfg.target_solve_rate
    experiment, cb = build_experiment(cfg)
    print(
        f"{cfg.scenario}: {spec.n_agents} agents, obs_dim {cb.env.obs_dim}, act_dim "
        f"{cb.env.act_dim}, {spec.n_envs} envs x {spec.steps_per_batch} steps, dt {spec.dt}, "
        f"substeps {spec.substeps}, max_steps {spec.max_steps}, target {target:.2f}, "
        f"BenchMARL experiment {experiment.folder_name}"
    )
    if cfg.scen_kwargs:
        print(f"  scenario kwargs: {cfg.scen_kwargs}")
    t0 = time.perf_counter()
    try:
        experiment.run()
    finally:
        cb.csv_file.close()
    final = cb.save("final")
    print(f"\ncheckpoints + metrics.csv in {cfg.checkpoint_dir}/ ({time.perf_counter() - t0:.0f}s)")
    rate = cb.window.solve_rate
    if rate is not None:
        k = int(round(rate * cb.window.ends))
        ci = wilson_ci(k, cb.window.ends)
        verdict = "MET" if rate >= target else "NOT met"
        print(
            f"final windowed ep_solve_rate {rate:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] "
            f"over {cb.window.ends} episodes -- target {target:.2f} {verdict}. "
            f"This is under the stochastic policy; run eval_mappo.py for the greedy number."
        )
    return final


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
