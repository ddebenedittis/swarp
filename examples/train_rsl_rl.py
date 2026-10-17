"""MAPPO on a swarp scenario with rsl_rl's PPO and ``OnPolicyRunner``, agents mapped to envs.

A baseline for ``python -m swarp.benchmark.train_throughput``: the same ``TrainSpec``
recipes, curriculum gate and ``metrics.csv`` columns as ``swarp.rl``, trained by rsl_rl as
shipped (``rsl-rl-lib`` 5.5). Needs the optional extra::

    uv pip install -e '.[rsl-rl]'

Run with::

    python examples/train_rsl_rl.py --scenario navigation
    python examples/eval_mappo.py runs/navigation_rsl_rl/navigation_final.pt --n-envs 2048

rsl_rl is single-agent, so :class:`SwarpVecEnv` presents every agent as an env:
``num_envs = n_envs * n_agents``, row ``e * n_agents + i`` is agent ``i`` of env ``e``. The
``"policy"`` observation group is each agent's own observation (rsl_rl's actor is then a
shared-weight per-agent actor, as ours is) and ``"critic"`` is the env's joint observation,
flattened and repeated for each of its agents; ``obs_groups`` wires them to the actor and
the critic. Rewards are per agent, ``dones`` is ``terminated | truncated`` broadcast to the
env's agents and ``extras["time_outs"]`` is ``truncated & ~terminated``, which rsl_rl
bootstraps as ``r + gamma * V(s_t)`` -- the same convention as ``swarp.rl``.

The loop is ``OnPolicyRunner.learn`` unmodified. Its ``Logger`` is replaced by
:class:`_RunLogger`, which keeps the episode accounting in device counters and does the
CSV, curriculum, entropy schedule, eval and checkpoints in ``log()``, after the runner has
taken its ``collect_time`` and ``learn_time``. ``wall_s`` is their sum. (The shipped
``Logger`` does nothing without a ``log_dir``; with one it reads the dones to the host
every step, which is tensorboard bookkeeping rather than training.)

Deliberate differences from our MAPPO (``swarp.rl`` / ``examples/train_mappo.py``):

* **The critic runs once per agent.** Each agent row carries its own copy of the joint
  observation and the critic evaluates all of them, so critic compute (rollout values,
  returns, every minibatch) and the critic observations in the rollout storage are
  ``n_agents`` times ours, for ``n_agents`` identical values.
* **Gaussian plus clamp, not tanh-Gaussian.** rsl_rl samples an unbounded Gaussian and
  computes the log-prob of the raw sample; the adapter clamps it to the env's action box
  before stepping. The greedy action is ``clamp(mean)``, not ``tanh(loc)``.
* **Std parameterization.** ``HeteroscedasticGaussianDistribution``, a state-dependent
  log-std head (initialised to exactly 1.0) clamped to ``[0.01, 1e6]``, against our
  ``softplus(x + b) + 0.01``. ``--std param`` selects rsl_rl's default state-independent
  learned std instead.
* **Analytic entropy** of the pre-clamp Gaussian, against a one-sample Monte Carlo
  estimate of the tanh-Gaussian's. The ``entropy`` column is therefore on another scale.
* **MSE value loss**, unclipped, against smooth-L1. ``--clipped-value-loss`` selects
  rsl_rl's default (PPO-style value clipping at the policy's clip 0.2).
* **Advantage normalization pooled over agents.** Per minibatch
  (``normalize_advantage_per_mini_batch=True``), as ours, but over all agent rows together
  rather than per agent, with ``+1e-8`` on the std. rsl_rl always normalizes, so a spec
  with ``normalize_advantage=False`` (Push-T) cannot be honoured; the run says so.
  ``--adv-norm rollout`` selects rsl_rl's default (once over the whole rollout).
* **Gradient clipping per network**: actor and critic each to norm 1.0, against one clip
  over both.
* **One minibatch partition for all epochs.** rsl_rl draws a single permutation per
  update; ours draws a fresh one per epoch.
* **No non-finite-gradient skip.** rsl_rl steps on whatever the backward produced, so a
  non-finite loss aborts the run here (after saving ``<scenario>_nan.pt``) instead of
  skipping the minibatch on device.
* **Eager everything.** The torch side is rsl_rl's eager loop: an unfused Adam, three
  ``.item()`` host reads per minibatch, rollout under ``torch.inference_mode``. The env
  still replays its own Warp graph per step.
* **No per-step NaN check.** rsl_rl's ``check_for_nan`` (on by default) costs four
  ``.any()`` host syncs per env step; a user who cares about speed turns it off, and so
  does this script. ``--check-for-nan`` turns it back on.

Matched: shared-weight actor, centralised critic over the joint observation, two Tanh
hidden layers of ``num_cells`` with torch's default ``nn.Linear`` init, no observation
normalization, Adam at the spec's fixed ``lr`` (no adaptive-KL schedule), clip 0.2,
``value_loss_coef`` 1.0, GAE with the spec's gamma and lambda, the same timeout bootstrap,
the spec's linear entropy-coefficient schedule (set on the algorithm between iterations),
the curriculum gate with knobs applied before the first reset, and the samples per
minibatch (``n_envs * steps_per_batch / minibatches`` env-steps, each ``n_agents`` rows).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from swarp import Environment
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

CHECKPOINT_FORMAT = "rsl_rl/1"
_CLIP_EPS = 0.2
_MAX_GRAD_NORM = 1.0
_STD_RANGE = (0.01, 1e6)  # our actor's softplus floor; no ceiling, as ours


class NonFiniteLoss(RuntimeError):
    """Raised when an update produced a non-finite loss (rsl_rl has no skip)."""


# ----------------------------------------------------------------------------- the env


class SwarpVecEnv(VecEnv):
    """A swarp ``Environment`` as an rsl_rl ``VecEnv`` with one env per agent.

    ``env`` must have ``auto_reset=True``; rsl_rl never resets. With ``clone_outputs=False``
    the env overwrites its obs buffer every step, while rsl_rl keeps the observation it
    acted on until after the next ``step`` (``PPO.act`` stores it, ``process_env_step``
    copies it into the rollout), so the observations go to two preallocated TensorDicts
    used alternately. Everything else is a view or an ``out=`` write into a fixed buffer:
    no per-step allocation beyond the env's own, and no host sync.
    """

    def __init__(self, env: Environment) -> None:
        if not env.auto_reset:
            raise ValueError("SwarpVecEnv needs an Environment with auto_reset=True")
        self.env = env
        E, A, Ob, D = env.n_envs, env.n_agents, env.obs_dim, env.act_dim
        self.E, self.A, self.Ob, self.D = E, A, Ob, D
        self.num_envs = E * A
        self.num_actions = D
        self.max_episode_length = env.max_steps
        self.device = env.device
        self.cfg: dict[str, Any] = {}
        dev = torch.device(env.device)
        self.lo, self.hi = env.action_bounds  # [A, D]
        self._actions = torch.zeros(E, A, D, device=dev, dtype=env.dtype)
        self._dones = torch.zeros(E, A, dtype=torch.bool, device=dev)
        self._time_outs = torch.zeros(E, A, dtype=torch.bool, device=dev)
        self._obs = [
            TensorDict(
                {
                    "policy": torch.zeros(E * A, Ob, device=dev),
                    "critic": torch.zeros(E * A, A * Ob, device=dev),
                },
                batch_size=[E * A],
                device=dev,
            )
            for _ in range(2)
        ]
        self._k = 0
        self.extras = {"time_outs": self._time_outs.view(-1)}
        # The last step's per-env flags and info, for the run's own accounting.
        self.terminated: torch.Tensor | None = None
        self.episode_end = self._dones[:, 0]
        self.info: Mapping[str, torch.Tensor] = {}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.step_count.repeat_interleave(self.A)

    def _write_obs(self, obs: torch.Tensor) -> TensorDict:
        E, A, Ob = self.E, self.A, self.Ob
        self._k ^= 1
        td = self._obs[self._k]
        td["policy"].view(E, A, Ob).copy_(obs)
        # Row e*A + i gets env e's joint obs, agent-major: our Critic's obs.flatten(-2).
        td["critic"].view(E, A, A, Ob).copy_(obs.unsqueeze(1).expand(E, A, A, Ob))
        return td

    def reset(self) -> TensorDict:
        return self._write_obs(self.env.reset())

    def get_observations(self) -> TensorDict:
        return self._obs[self._k]

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        E, A = self.E, self.A
        # rsl_rl's Gaussian is unbounded; the env box is applied here.
        torch.clamp(actions.view(E, A, self.D), self.lo, self.hi, out=self._actions)
        obs, rew, term, trunc, info = self.env.step(self._actions)
        term_a, trunc_a = term.unsqueeze(1).expand(E, A), trunc.unsqueeze(1).expand(E, A)
        torch.logical_or(term_a, trunc_a, out=self._dones)
        torch.gt(trunc_a, term_a, out=self._time_outs)  # truncated & ~terminated
        self.terminated, self.info = term, info
        return self._write_obs(obs), rew.view(-1), self._dones.view(-1), self.extras


# ---------------------------------------------------------------------- rsl_rl config


def runner_cfg(spec: TrainSpec, args: argparse.Namespace) -> dict[str, Any]:
    """The ``OnPolicyRunner`` train config for a spec."""
    hidden = [spec.num_cells, spec.num_cells]
    if args.std == "state":
        dist = {
            "class_name": "HeteroscedasticGaussianDistribution",
            "init_std": 1.0, "std_type": "log", "std_range": _STD_RANGE,
        }
    else:
        dist = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_range": _STD_RANGE}
    return {
        "num_steps_per_env": spec.steps_per_batch,
        "save_interval": 0,  # the runner saves only with a writer; this script saves its own
        "check_for_nan": args.check_for_nan,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {
            "class_name": "MLPModel", "hidden_dims": hidden, "activation": "tanh",
            "obs_normalization": False, "distribution_cfg": dist,
        },
        "critic": {
            "class_name": "MLPModel", "hidden_dims": hidden, "activation": "tanh",
            "obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": spec.epochs,
            "num_mini_batches": spec.minibatches,
            "clip_param": _CLIP_EPS,
            "gamma": spec.gamma,
            "lam": spec.lmbda,
            "value_loss_coef": 1.0,
            "entropy_coef": spec.entropy_coeff,
            "learning_rate": spec.lr,
            "max_grad_norm": _MAX_GRAD_NORM,
            "optimizer": "adam",
            "use_clipped_value_loss": args.clipped_value_loss,
            "schedule": "fixed",
            "desired_kl": None,
            "normalize_advantage_per_mini_batch": args.adv_norm == "minibatch",
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


# ------------------------------------------------------------------------- checkpoints


def _build_actor(ckpt: Mapping[str, Any], device: str) -> MLPModel:
    obs = TensorDict(
        {"policy": torch.zeros(1, ckpt["obs_dim"], device=device)}, batch_size=[1], device=device
    )
    cfg = dict(ckpt["actor_cfg"])
    cfg.pop("class_name")
    return MLPModel(obs, {"actor": ["policy"]}, "actor", ckpt["act_dim"], **cfg).to(device)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[MLPModel, dict[str, Any]]:
    """The actor and metadata of a checkpoint written by this script."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    actor = _build_actor(ckpt, device)
    actor.load_state_dict(ckpt["actor"])
    return actor.eval(), ckpt


def greedy_arm(actor: MLPModel) -> PolicyArm:
    """The policy arm of :func:`swarp.rl.recipes.score`: the clamped Gaussian mean.

    Calls the MLP on ``[n_envs, n_agents, obs_dim]`` directly (no observation
    normalization is configured, so that is the whole model).
    """

    def make(n_envs: int, device: str, lo: torch.Tensor, hi: torch.Tensor):
        @torch.no_grad()
        def act(obs: torch.Tensor) -> torch.Tensor:
            mean = actor.distribution.deterministic_output(actor.mlp(obs))
            return mean.clamp(lo, hi)

        return act

    return make


# ----------------------------------------------------------------------------- the CLI


@dataclass
class TrainConfig:
    """A resolved spec plus CLI overrides."""

    scenario: str
    spec: TrainSpec
    scen_kwargs: Mapping[str, Any]
    device: str
    checkpoint_dir: Path
    args: argparse.Namespace
    start_difficulty: float = 0.0
    checkpoint_every: int = 250
    eval_every: int = 500
    eval_envs: int = 512
    seed: int = 0
    gate_window: int = 25
    target_solve_rate: float | None = None


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    """Parse into a :class:`TrainConfig`. The flags mirror ``python -m swarp.rl``.

    Hyperparameter flags default to ``None``, meaning "take the spec's value".
    """
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser(description="MAPPO on a swarp scenario with rsl_rl.")
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
    p.add_argument("--checkpoint-dir", help="default: runs/<scenario>_rsl_rl")
    p.add_argument("--checkpoint-every", type=int, default=250, help="0 disables")
    p.add_argument("--eval-every", type=int, default=500, help="0 disables")
    p.add_argument("--eval-envs", type=int, default=512)
    p.add_argument("--target-solve-rate", type=float)
    # rsl_rl's own knobs; the defaults are the choices argued in the module docstring.
    p.add_argument("--check-for-nan", action="store_true",
                   help="rsl_rl's per-step NaN check (four host syncs per env step)")
    p.add_argument("--clipped-value-loss", action="store_true",
                   help="rsl_rl's clipped value loss instead of plain MSE")
    p.add_argument("--adv-norm", choices=("minibatch", "rollout"), default="minibatch",
                   help="advantage normalization statistics, pooled over agents")
    p.add_argument("--std", choices=("state", "param"), default="state",
                   help="state-dependent std head, or rsl_rl's default learned std vector")
    a = p.parse_args(argv)

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
        checkpoint_dir=Path(a.checkpoint_dir or f"runs/{a.scenario}_rsl_rl"), args=a,
        start_difficulty=a.start_difficulty, checkpoint_every=a.checkpoint_every,
        eval_every=a.eval_every, eval_envs=a.eval_envs, seed=a.seed,
        gate_window=a.gate_window, target_solve_rate=a.target_solve_rate,
    )


# --------------------------------------------------------------------------- training


class _RunLogger:
    """Stands in for rsl_rl's ``Logger`` on the runner.

    ``process_env_step`` adds the step's rewards, per-env terminated/ended flags and the
    spec's ``info()`` metrics into device buffers (a handful of in-place kernels, no sync).
    ``log`` runs after the runner has timed the iteration: it reads everything to the host
    once and does the bookkeeping ``swarp.rl``'s ``train`` does. ``writer`` stays ``None``
    so the runner never saves its own checkpoints.
    """

    writer = None

    def __init__(
        self, cfg: TrainConfig, venv: SwarpVecEnv, runner: OnPolicyRunner,
        info0: Mapping[str, torch.Tensor], metrics: Mapping[str, int],
        metric_cols: list[str], set_curriculum,
    ) -> None:
        self.cfg, self.venv, self.runner = cfg, venv, runner
        self.spec = spec = cfg.spec
        self.metrics = dict(metrics)
        self.set_curriculum = set_curriculum
        dev = torch.device(cfg.device)
        self.rew_acc = torch.zeros(venv.num_envs, device=dev)
        self.term_acc = torch.zeros(venv.E, dtype=torch.int32, device=dev)
        self.done_acc = torch.zeros(venv.E, dtype=torch.int32, device=dev)
        # Allocated here, not on first use: inside the rollout's inference_mode they would
        # be inference tensors, which ``zero_`` outside it rejects.
        self.metric_acc = {k: torch.zeros(info0[k].shape, device=dev) for k in self.metrics}
        self.frames_per_batch = spec.n_envs * spec.steps_per_batch
        self.window = SolveRateWindow(window=cfg.gate_window)
        self.target = (
            spec.target_solve_rate if cfg.target_solve_rate is None else cfg.target_solve_rate
        )
        self.best_key = (-1.0, -1.0)
        self.difficulty = 0.0
        ckpt_dir = cfg.checkpoint_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = (ckpt_dir / "metrics.csv").open("w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["iter", "frames", "wall_s", "env_sps", "reward_per_step", "ep_solve_rate",
             "ep_ends", "ep_len_mean", "difficulty", "entropy", "loss_objective",
             "loss_critic", "collect_s", "learn_s", "eval_solve_rate", "eval_ci_lo",
             "eval_ci_hi"]
            + metric_cols
        )

    # The runner's Logger interface.

    def init_logging_writer(self) -> None:
        pass

    def process_env_step(
        self, rewards: torch.Tensor, dones: torch.Tensor, extras: dict,
        intrinsic_rewards: torch.Tensor | None = None,
    ) -> None:
        venv = self.venv
        self.rew_acc.add_(rewards)
        self.term_acc.add_(venv.terminated)
        self.done_acc.add_(venv.episode_end)
        for key, acc in self.metric_acc.items():
            acc.add_(venv.info[key])

    def log(
        self, it: int, collect_time: float, learn_time: float, loss_dict: dict, **_: Any
    ) -> None:
        cfg, spec, window = self.cfg, self.spec, self.window
        wall = collect_time + learn_time
        T = spec.steps_per_batch
        parts = [
            (self.rew_acc.sum() / (T * self.venv.num_envs)).view(1),
            self.term_acc.sum().view(1).float(),
            self.done_acc.sum().view(1).float(),
            *(
                self.metric_acc[k].reshape(-1, w).mean(0) / T
                for k, w in self.metrics.items()
            ),
        ]
        stats = torch.cat(parts).tolist()  # the iteration's one extra host read
        for acc in (self.rew_acc, self.term_acc, self.done_acc, *self.metric_acc.values()):
            acc.zero_()
        mean_reward, n_term, n_done = stats[:3]
        window.update_counts(int(n_term), int(n_done), self.frames_per_batch)
        obj, crit, ent = loss_dict["surrogate"], loss_dict["value"], loss_dict["entropy"]
        if not all(math.isfinite(x) for x in (obj, crit, ent)):
            self.save("nan")
            self.csv_file.close()
            raise NonFiniteLoss(
                f"non-finite loss at iter {it} (surrogate {obj}, value {crit}, entropy "
                f"{ent}); rsl_rl has no skip, state saved to "
                f"{cfg.checkpoint_dir / f'{cfg.scenario}_nan.pt'}."
            )

        rate, ep_len = window.solve_rate, window.ep_len_mean
        f = self.difficulty = self.set_curriculum(rate)  # difficulty for the next resets
        self.runner.alg.entropy_coef = entropy_coeff(spec, it + 1)

        eval_rate = eval_lo = eval_hi = None
        if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
            res = score(
                cfg.scenario, policy=greedy_arm(self.runner.alg.get_policy()),
                n_agents=spec.n_agents, n_envs=cfg.eval_envs, steps=spec.max_steps,
                device=cfg.device, seed=cfg.seed + 1_000, dt=spec.dt, substeps=spec.substeps,
                scen_kwargs=cfg.scen_kwargs, arms=("policy",),
            )["policy"]
            eval_rate, (eval_lo, eval_hi) = res.solve_rate, res.ci
            print(
                f"  eval @ iter {it + 1}: greedy solve {eval_rate:.3f} "
                f"[{eval_lo:.3f}, {eval_hi:.3f}] over {cfg.eval_envs} envs "
                f"(target {self.target:.2f})"
            )

        self.csv_writer.writerow(
            [it, (it + 1) * self.frames_per_batch, f"{wall:.3f}",
             f"{self.frames_per_batch / wall:.0f}", f"{mean_reward:.6f}",
             "" if rate is None else f"{rate:.4f}", window.ends,
             "" if ep_len is None else f"{ep_len:.1f}", f"{f:.3f}", f"{ent:.4f}",
             f"{obj:.4f}", f"{crit:.4f}", f"{collect_time:.3f}", f"{learn_time:.3f}",
             "" if eval_rate is None else f"{eval_rate:.4f}",
             "" if eval_lo is None else f"{eval_lo:.4f}",
             "" if eval_hi is None else f"{eval_hi:.4f}"]
            + [f"{x:.6f}" for x in stats[3:]]
        )
        self.csv_file.flush()
        rate_txt = "  --  " if rate is None else f"{rate:.3f}"
        print(
            f"iter {it:4d}  reward/step {mean_reward:+.4f}  solve {rate_txt}  "
            f"ends {window.ends:5d}  ep_len {0.0 if ep_len is None else ep_len:6.1f}  "
            f"diff {f:.2f}  ent {ent:+.3f}  {wall * 1e3:.1f}ms "
            f"(collect {collect_time * 1e3:.1f})"
        )
        if cfg.checkpoint_every and (it + 1) % cfg.checkpoint_every == 0:
            self.save(f"iter{it + 1:05d}")
        if rate is not None and (f, rate) > self.best_key:
            self.best_key = (f, rate)
            self.save("best")

    # Checkpoints.

    def save(self, tag: str) -> Path:
        cfg, spec, venv = self.cfg, self.spec, self.venv
        alg = self.runner.alg
        path = cfg.checkpoint_dir / f"{cfg.scenario}_{tag}.pt"
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "actor": alg.get_policy().state_dict(),
                "critic": alg._raw_critic.state_dict(),
                "actor_cfg": self.runner.cfg["actor"],
                "critic_cfg": self.runner.cfg["critic"],
                "obs_dim": venv.Ob,
                "act_dim": venv.D,
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


def entropy_coeff(spec: TrainSpec, it: int) -> float:
    """The spec's linear entropy-coefficient schedule at iteration ``it``."""
    if spec.entropy_coeff_final is None or spec.iters <= 1:
        return spec.entropy_coeff
    frac = min(it, spec.iters - 1) / (spec.iters - 1)
    return spec.entropy_coeff + frac * (spec.entropy_coeff_final - spec.entropy_coeff)


def train(cfg: TrainConfig) -> Path:
    """Run the training loop. Returns the final checkpoint path."""
    spec, device = cfg.spec, cfg.device
    torch.manual_seed(cfg.seed)

    env = build_env(
        cfg.scenario, n_envs=spec.n_envs, n_agents=spec.n_agents, device=device, dt=spec.dt,
        substeps=spec.substeps, max_steps=spec.max_steps, seed=cfg.seed,
        scen_kwargs=cfg.scen_kwargs, env_kwargs={"auto_reset": True},
    )
    scen = env.scenario
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
    metrics = {
        k: (info0[k].shape[-1] if k == "multiobj_reward" else 1) for k in spec.metrics
    }
    metric_cols: list[str] = []
    for k, w in metrics.items():
        metric_cols += [f"rew_term{i}" for i in range(w)] if k == "multiobj_reward" else [k]

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

    venv = SwarpVecEnv(env)
    start = set_curriculum(None)
    venv.reset()  # after the knobs, so the first episodes run at the starting difficulty

    runner = OnPolicyRunner(venv, runner_cfg(spec, cfg.args), log_dir=None, device=device)
    runner.alg.entropy_coef = entropy_coeff(spec, 0)
    log = _RunLogger(cfg, venv, runner, info0, metrics, metric_cols, set_curriculum)
    log.difficulty = start
    runner.logger = log

    print(
        f"{cfg.scenario}: {spec.n_agents} agents, obs_dim {env.obs_dim}, act_dim "
        f"{env.act_dim}, {spec.n_envs} envs x {spec.steps_per_batch} steps "
        f"({venv.num_envs} rsl_rl envs), dt {spec.dt}, substeps {spec.substeps}, "
        f"max_steps {spec.max_steps}, target {log.target:.2f}, std {cfg.args.std}, "
        f"adv-norm {cfg.args.adv_norm}, clipped value loss {cfg.args.clipped_value_loss}, "
        f"check_for_nan {cfg.args.check_for_nan}"
    )
    if cfg.scen_kwargs:
        print(f"  scenario kwargs: {cfg.scen_kwargs}")
    if not spec.normalize_advantage:
        print("  note: the spec disables advantage normalization; rsl_rl always normalizes")

    try:
        runner.learn(spec.iters)
    finally:
        log.csv_file.close()
    final = log.save("final")
    print(f"  env graph_mode {env.graph_mode}")

    window, rate = log.window, log.window.solve_rate
    print(f"\ncheckpoints + metrics.csv in {cfg.checkpoint_dir}/")
    if log.best_key[0] >= 0.0:
        print(
            f"best checkpoint: solve {log.best_key[1]:.3f} at difficulty "
            f"{log.best_key[0]:.2f} -> {cfg.checkpoint_dir / f'{cfg.scenario}_best.pt'}"
        )
    if rate is not None:
        k = int(round(rate * window.ends))
        ci = wilson_ci(k, window.ends)
        verdict = "MET" if rate >= log.target else "NOT met"
        print(
            f"final windowed ep_solve_rate {rate:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] "
            f"over {window.ends} episodes -- target {log.target:.2f} {verdict}. "
            f"This is under the stochastic policy; run eval_mappo.py for the greedy number."
        )
    return final


if __name__ == "__main__":
    train(parse_args())
