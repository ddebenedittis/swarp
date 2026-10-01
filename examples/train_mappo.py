"""MAPPO on any swarp scenario, via TorchRL.

Wraps a registered scenario in the batched TorchRL ``EnvBase`` from
:mod:`swarp.interop.torchrl` and runs an on-policy PPO loop with a decentralised
shared-weight actor and a centralised critic. Everything stays on-device: the collector
drives the vectorized swarp step directly, no per-env Python loop.

Needs the optional torchrl extra::

    uv pip install -e '.[torchrl]'

Run with::

    python examples/train_mappo.py --scenario navigation
    python examples/train_mappo.py --scenario pusht --iters 2100

Per-scenario defaults (episode length, substeps, shaping weights, curriculum, which
``info()`` keys to log) come from the ``SPECS`` table in :mod:`mappo`; every one is
overridable on the command line, and ``--scen-kwarg k=v`` reaches any scenario
constructor keyword without a dedicated flag. ``--scenario pusht`` reproduces the
published Push-T recipe exactly, which a test pins.

What actually matters for getting a policy off the ground, learnt the expensive way on
Push-T and generalized here:

* **Balanced shaping terms.** Push-T's raw 1.0/0.5 weights let the (easier) rotation term
  dominate at ~4x, and the policy then spins the T and ignores position; ``pos_shaping``
  5.0 rebalances them. To make that visible on iteration 1 rather than in hour 3, a
  scenario emitting ``info()["multiobj_reward"]`` gets one ``rew_*`` column per term.
* **A curriculum on the initial-condition distribution.** The terminal bonus has to be
  reachable by a novice policy or it is never experienced at all and the run plateaus on
  the dense term alone. Difficulty rises only while the episode solve rate clears
  ``--curriculum-gate``, so a stalled run holds difficulty and shows a flat column --
  which is the diagnostic. (The original Push-T gate compared a per-*step* termination
  fraction against that threshold after multiplying by ``max_steps``, giving ~4.8 vs 0.35:
  open from the first iterations. That run's curriculum was effectively ungated.)
* **Episodes long enough not to truncate successes.** Watch ``ep_len_mean``: if it hugs
  ``max_steps`` while ``ep_solve_rate`` still climbs, raise ``--max-steps`` first.

The reported ``ep_solve_rate`` is a windowed ``sum(terminated)/sum(done)`` -- an unbiased
estimate of the fraction of episode *starts* that get solved. See
:class:`mappo.SolveRateWindow`; it is not the old ``solved`` column, which was smaller by
a factor of the episode length.

Evaluate a checkpoint against random, zero-action and scripted baselines with::

    python examples/eval_mappo.py runs/navigation/navigation_final.pt --n-envs 2048
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from mappo import (
    DONE_KEY,
    DONE_NAME,
    SPECS,
    TERM_KEY,
    TERM_NAME,
    SolveRateWindow,
    TrainSpec,
    build_critic,
    build_env,
    build_policy,
    coerce_scen_kwargs,
    expand_flag,
    score,
    wilson_ci,
)
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from swarp.benchmark.ablation import sync_device
from swarp.interop.torchrl import SwarpEnv


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


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    """Parse into a :class:`TrainConfig`.

    Hyperparameter flags default to ``None``, meaning "take the spec's value", so a
    user override is distinguishable from a spec value and the effective config can be
    printed and stored.
    """
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", required=True, choices=sorted(SPECS))
    p.add_argument("--device", default=default_device)
    p.add_argument("--seed", type=int, default=0)
    # Simulation
    p.add_argument("--n-agents", type=int)
    p.add_argument("--dt", type=float)
    p.add_argument("--substeps", type=int)
    p.add_argument("--max-steps", type=int, help="episode length")
    p.add_argument(
        "--scen-kwarg", action="append", default=[], metavar="K=V",
        help="scenario constructor keyword; repeatable",
    )
    # PPO
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
    # Curriculum
    p.add_argument("--curriculum-iters", type=int, help="0 disables")
    p.add_argument("--curriculum-gate", type=float)
    p.add_argument("--gate-window", type=int, default=25,
                   help="iterations of episode ends behind ep_solve_rate")
    p.add_argument("--start-difficulty", type=float, default=0.0)
    # Bookkeeping
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
        scenario=a.scenario,
        spec=spec,
        scen_kwargs=scen_kwargs,
        device=a.device,
        checkpoint_dir=Path(a.checkpoint_dir or f"runs/{a.scenario}"),
        resume=a.resume,
        start_difficulty=a.start_difficulty,
        checkpoint_every=a.checkpoint_every,
        eval_every=a.eval_every,
        eval_envs=a.eval_envs,
        seed=a.seed,
        gate_window=a.gate_window,
        max_skip_frac=a.max_skip_frac,
        max_skip_iters=a.max_skip_iters,
        target_solve_rate=a.target_solve_rate,
    )


class NonFiniteGradients(RuntimeError):
    """Raised when the run is producing non-finite gradients faster than it can learn."""


def train(cfg: TrainConfig) -> Path:
    """Run the training loop. Returns the final checkpoint path.

    Split out from ``main`` so ``tests/examples`` can drive a 2-iteration smoke run
    in-process: that test catches the shape/key/spec errors which are most of what breaks
    a training script, and it is only possible because this is a function.
    """
    spec, device = cfg.spec, cfg.device
    n_agents = spec.n_agents

    sim = build_env(
        cfg.scenario, n_envs=spec.n_envs, n_agents=n_agents, device=device, dt=spec.dt,
        substeps=spec.substeps, max_steps=spec.max_steps, seed=cfg.seed,
        scen_kwargs=cfg.scen_kwargs,
    )
    # SwarpEnv specs sim.action_bounds by default -- the limits the kernels actually
    # clamp against. Nothing to pass.
    env = SwarpEnv(sim)
    scen = sim.scenario
    obs_dim, act_dim = env.obs_dim, env.act_dim

    # The actor's TanhNormal box is hardcoded [-1, 1] (see mappo.build_policy) while the
    # env's action spec is sim.action_bounds. They coincide for every holonomic
    # max_speed=1.0 scenario, but a scenario with different limits would have its policy
    # silently throttled to a fraction of the achievable range, so check rather than hope.
    lo, hi = sim.action_bounds
    nonzero = hi > lo  # zero-width slots exist for mixed fleets and are legitimately 0
    if not (
        torch.allclose(lo[nonzero], torch.full_like(lo[nonzero], -1.0))
        and torch.allclose(hi[nonzero], torch.full_like(hi[nonzero], 1.0))
    ):
        raise ValueError(
            f"{cfg.scenario!r} has action bounds ({lo.min().item():g}, {hi.max().item():g}) "
            f"but the actor's TanhNormal is fixed to [-1, 1]; rescale the distribution "
            f"from env.action_spec before training this scenario."
        )

    # Validate the spec against the scenario before the first iteration: a metric key that
    # is not in info() would log blanks forever, and a curriculum knob naming an attribute
    # that does not exist would silently set a *new* attribute and ramp nothing.
    info_keys = set(env._info_keys)
    missing = [k for k in spec.metrics if k not in info_keys]
    if missing:
        raise ValueError(
            f"{cfg.scenario!r} info() has no key(s) {missing}; available: "
            f"{sorted(info_keys)}"
        )
    for knob in spec.curriculum:
        if not hasattr(scen, knob.attr):
            raise ValueError(
                f"curriculum knob {knob.attr!r} is not an attribute of "
                f"{type(scen).__name__}; the ramp would silently do nothing."
            )

    policy = build_policy(obs_dim, act_dim, n_agents, device, num_cells=spec.num_cells)
    critic = build_critic(obs_dim, n_agents, device, num_cells=spec.num_cells)

    frames_per_batch = spec.n_envs * spec.steps_per_batch
    collector = Collector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * spec.iters,
        device=device,
        auto_register_policy_transforms=True,
    )
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=frames_per_batch // spec.minibatches,
    )
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        entropy_coeff=spec.entropy_coeff,
        normalize_advantage=spec.normalize_advantage,
        # Normalize per agent, not pooled across the team. The advantage is
        # [batch, n_agents, 1]; excluding the agent dim keeps each agent's statistics
        # independent, which is what torchrl asks for in multi-agent settings and what
        # BenchMARL's MAPPO does. With shared_reward=True every agent's advantage is
        # identical so it makes no difference; with per-agent rewards, pooling would let
        # one agent's scale distort another's.
        normalize_advantage_exclude_dims=(-2,),
    )
    # Leaf names: the value estimator looks them up under ("next", ...) itself.
    loss_module.set_keys(
        reward="reward", done=DONE_NAME, terminated=TERM_NAME, value="state_value"
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=spec.gamma, lmbda=spec.lmbda
    )
    optim = torch.optim.Adam(loss_module.parameters(), lr=spec.lr)

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt["policy"])
        critic.load_state_dict(ckpt["critic"])
        print(f"resumed actor + critic from {cfg.resume}")

    difficulty = min(1.0, max(0.0, cfg.start_difficulty))

    def set_curriculum(gate_metric: float | None) -> float:
        """Ramp the knobs while the policy keeps solving.

        ``gate_metric`` is the windowed episode solve rate, or ``None`` during warm-up
        (all envs start in phase, so the first few iterations see no episode ends at all)
        -- in which case the gate is closed rather than treated as a zero.
        """
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

    ckpt_dir = cfg.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # A multiobj_reward key is [n_envs, n_agents, n_obj]; give each term its own column so
    # a shaping imbalance is legible immediately.
    n_obj = 0
    if "multiobj_reward" in spec.metrics:
        probe = scen.info()["multiobj_reward"]
        n_obj = probe.shape[-1]
    metric_cols: list[str] = []
    for k in spec.metrics:
        if k == "multiobj_reward":
            metric_cols += [f"rew_term{i}" for i in range(n_obj)]
        else:
            metric_cols.append(k)

    csv_path = ckpt_dir / "metrics.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["iter", "frames", "wall_s", "env_sps", "reward_per_step", "ep_solve_rate",
         "ep_ends", "ep_len_mean", "difficulty", "entropy", "loss_objective",
         "loss_critic", "grad_skips", "eval_solve_rate", "eval_ci_lo", "eval_ci_hi"]
        + metric_cols
    )

    target = cfg.target_solve_rate
    if target is None:
        target = spec.target_solve_rate

    def save(tag: str) -> Path:
        path = ckpt_dir / f"{cfg.scenario}_{tag}.pt"
        torch.save(
            {
                # The original six keys, in order: an older evaluator reads exactly these.
                "policy": policy.state_dict(),
                "critic": critic.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "n_agents": n_agents,
                "num_cells": spec.num_cells,
                # New, primitives only -- weights_only=True rejects pickled classes, so no
                # TrainSpec and no Namespace here.
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

    window = SolveRateWindow(window=cfg.gate_window)
    total_skips = 0
    hot_skip_iters = 0
    # Ranked on (difficulty, solve rate) lexicographically rather than solve rate alone:
    # caging peaked at 0.280 while the curriculum was still at difficulty 0.192 and scored
    # 0.000 once it reached 1.0, so a bare rate comparison would enshrine a policy that
    # only ever saw the easy end of the task. A later checkpoint wins only at an equal or
    # harder difficulty.
    best_key = (-1.0, -1.0)
    eval_lo = eval_hi = eval_rate = None
    print(
        f"{cfg.scenario}: {n_agents} agents, obs_dim {obs_dim}, act_dim {act_dim}, "
        f"{spec.n_envs} envs x {spec.steps_per_batch} steps, dt {spec.dt}, "
        f"substeps {spec.substeps}, max_steps {spec.max_steps}, target {target:.2f}"
    )
    if cfg.scen_kwargs:
        print(f"  scenario kwargs: {cfg.scen_kwargs}")

    for it, batch in enumerate(collector):
        t0 = time.perf_counter()
        # Entropy schedule: broad early exploration, committed late behaviour. Watched via
        # the `entropy` column -- an early collapse is the signature of a policy that has
        # locked into a local optimum.
        if spec.entropy_coeff_final is not None and spec.iters > 1:
            frac = it / (spec.iters - 1)
            coeff = spec.entropy_coeff + frac * (spec.entropy_coeff_final - spec.entropy_coeff)
            loss_module.entropy_coeff = coeff
        else:
            coeff = spec.entropy_coeff

        terminated = batch.get(("next", "terminated"))
        done = batch.get(("next", "done"))
        window.update(terminated, done)
        batch.set(DONE_KEY, expand_flag(done, n_agents))
        batch.set(TERM_KEY, expand_flag(terminated, n_agents))
        with torch.no_grad():
            loss_module.value_estimator(
                batch,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        flat = batch.reshape(-1)
        buffer.extend(flat)
        skips = 0
        updates = 0
        ent_sum = obj_sum = crit_sum = 0.0
        for _ in range(spec.epochs):
            for _ in range(spec.minibatches):
                sub = buffer.sample()
                losses = loss_module(sub)
                total = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
                optim.zero_grad()
                total.backward()
                # `clip_grad_norm_` rescales but does not filter: one non-finite gradient
                # goes straight into Adam, whose moments then turn *every* parameter NaN
                # on the next step. That is unrecoverable, and nothing used to stop the
                # run -- it kept collecting, reporting nan reward, for as long as you let
                # it. A 1-agent Push-T run died this way at iter 351; the exact source was
                # never pinned down. Skipping the minibatch costs one update, and on a
                # healthy gradient this branch never runs.
                gnorm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                if not torch.isfinite(gnorm):
                    optim.zero_grad(set_to_none=True)
                    skips += 1
                    continue
                optim.step()
                updates += 1
                obj_sum += losses["loss_objective"].item()
                crit_sum += losses["loss_critic"].item()
                # torchrl reports the raw differential entropy alongside the weighted
                # loss term (`loss_entropy == -entropy_coeff * entropy`), so read it
                # directly rather than dividing the loss back out -- no epsilon guard, and
                # it stays correct if the coefficient schedule ever hits zero.
                ent = losses.get("entropy", None)
                if ent is None:
                    ent_sum += -losses["loss_entropy"].item() / max(coeff, 1e-12)
                else:
                    ent_sum += float(ent.mean().item())
        buffer.empty()
        total_skips += skips

        n_mb = spec.epochs * spec.minibatches
        if skips / n_mb > cfg.max_skip_frac:
            hot_skip_iters += 1
        else:
            hot_skip_iters = 0
        if hot_skip_iters >= cfg.max_skip_iters:
            save("nan")
            csv_file.close()
            collector.shutdown()
            raise NonFiniteGradients(
                f"{skips}/{n_mb} minibatches had non-finite gradients for "
                f"{hot_skip_iters} consecutive iterations at iter {it}; state saved to "
                f"{ckpt_dir / f'{cfg.scenario}_nan.pt'}. The run is not recovering -- "
                f"lower --lr or check the scenario's reward for a division by a distance."
            )

        sync_device(device)
        wall = time.perf_counter() - t0
        mean_reward = batch.get(("next", "reward")).mean().item()
        rate = window.solve_rate
        ep_len = window.ep_len_mean
        f = set_curriculum(rate)  # difficulty for the next batch's resets

        info_vals: list[str] = []
        for k in spec.metrics:
            v = batch.get(("next", "info", k))
            if k == "multiobj_reward":
                # [..., n_obj] -> one mean per term
                flat_v = v.reshape(-1, v.shape[-1])
                info_vals += [f"{x:.6f}" for x in flat_v.float().mean(dim=0).tolist()]
            else:
                info_vals.append(f"{v.float().mean().item():.6f}")

        if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
            # The collector runs this loop body on its own CUDA stream, and a swarp env
            # built and reset under a user-created stream races: the greedy eval came out
            # 0.000 or 0.500 on checkpoints that eval_mappo.py (default stream) scores
            # 1.000. So evaluate on the default stream, ordered against the side stream in
            # both directions. (The training env is built before the collector exists, on
            # the default stream, and stepping it on the side stream measured clean.)
            on_cuda = device.startswith("cuda")
            side = torch.cuda.current_stream(device) if on_cuda else None
            default = torch.cuda.default_stream(device) if on_cuda else None
            if on_cuda:
                default.wait_stream(side)
            with torch.cuda.stream(default):  # a no-op off CUDA (stream None)
                res = score(
                    cfg.scenario, policy=policy, n_agents=n_agents, n_envs=cfg.eval_envs,
                    steps=spec.max_steps, device=device, seed=cfg.seed + 1_000,
                    dt=spec.dt, substeps=spec.substeps, scen_kwargs=cfg.scen_kwargs,
                    arms=("policy",),
                )["policy"]
            if on_cuda:
                side.wait_stream(default)
            eval_rate, (eval_lo, eval_hi) = res.solve_rate, res.ci
            print(
                f"  eval @ iter {it + 1}: greedy solve {eval_rate:.3f} "
                f"[{eval_lo:.3f}, {eval_hi:.3f}] over {cfg.eval_envs} envs "
                f"(target {target:.2f})"
            )

        csv_writer.writerow(
            [it, (it + 1) * frames_per_batch, f"{wall:.3f}",
             f"{frames_per_batch / wall:.0f}", f"{mean_reward:.6f}",
             "" if rate is None else f"{rate:.4f}", window.ends,
             "" if ep_len is None else f"{ep_len:.1f}", f"{f:.3f}",
             f"{ent_sum / max(updates, 1):.4f}", f"{obj_sum / max(updates, 1):.4f}",
             f"{crit_sum / max(updates, 1):.4f}", skips,
             "" if eval_rate is None else f"{eval_rate:.4f}",
             "" if eval_lo is None else f"{eval_lo:.4f}",
             "" if eval_hi is None else f"{eval_hi:.4f}"]
            + info_vals
        )
        csv_file.flush()
        eval_rate = eval_lo = eval_hi = None  # only on the iteration it was measured

        rate_txt = "  --  " if rate is None else f"{rate:.3f}"
        print(
            f"iter {it:4d}  reward/step {mean_reward:+.4f}  solve {rate_txt}  "
            f"ends {window.ends:5d}  ep_len {0.0 if ep_len is None else ep_len:6.1f}  "
            f"diff {f:.2f}  ent {ent_sum / max(updates, 1):+.3f}  {wall:.2f}s"
            + (f"  skipped {skips}" if skips else "")
        )
        if cfg.checkpoint_every and (it + 1) % cfg.checkpoint_every == 0:
            save(f"iter{it + 1:05d}")

        # `_best.pt` exists so a late entropy collapse cannot destroy the run's result:
        # giveway reached solve 0.9999 at iter 1750 and 0.000 by 2750, and only the
        # fixed-interval snapshots saved it. `rate` is None during warm-up, before any
        # episode has ended.
        if rate is not None and (f, rate) > best_key:
            best_key = (f, rate)
            save("best")

    final = save("final")
    csv_file.close()
    collector.shutdown()
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


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
