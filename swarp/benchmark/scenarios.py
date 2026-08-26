"""Generalized cross-scenario hot-path benchmark.

Sweeps {selected scenarios} x {robot model where supported} x {lidar off / N
rays} and, for each config, times an un-optimized *baseline* (all opts off) vs
the fully *optimized* default (fused obs/reward where available + physics CUDA
graph + neighbor dedupe + slim-2D). Every config is parity-checked before timing
(5-step seeded trajectory, optimized vs baseline, ``allclose`` obs/reward + exact
``done``) — the same gate that caught the CUDA-graph reuse bug. The tolerance comes from
the scenario class (``Scenario.parity_rtol`` / ``parity_atol``), since a scenario whose two
paths run deliberately different physics — push-t integrates its body at
``body_substeps`` in torch and at ``Stepper.substeps`` in the engine — cannot be held to
the ulp-scale bound the others meet.

The scenario registry lives in :mod:`swarp.scenarios` (``SCENARIOS``); this CLI is
one of its consumers, so adding a scenario there (or lighting up its fused
kernels) surfaces it here and in the tests at once.

Run with::

    python -m swarp.benchmark.scenarios --scenarios navigation          # default: one
    python -m swarp.benchmark.scenarios --scenarios all --csv out.csv    # full set
    python -m swarp.benchmark.scenarios --scenarios flocking formation \
        --envs 4096 --agents 16 64 --lidar-rays 0 16 32

Notes:
* ``auto_reset=False`` — navigation's per-step reset resampling otherwise
  dominates and masks the step (a real measurement trap).
* Global JIT/graph warmup runs on tiny envs before timing so the first config
  does not absorb one-time kernel-compilation cost.
* OOM on a config prints and continues.
"""

from __future__ import annotations

import argparse
import csv as csvmod
import time

import torch

from swarp import DynamicsModel, Environment, Lidar
from swarp.benchmark.ablation import parity_ok, sync_device
from swarp.scenarios import (
    SCENARIOS,
    make_scenario,
    resolve_scenarios,
    supports_model,
)

# Backwards-compatible alias for the registry, whose home is swarp.scenarios.
SCENARIO_FACTORIES = SCENARIOS

MODELS: dict[str, DynamicsModel] = {
    "holonomic": DynamicsModel.HOLONOMIC,
    "diff-drive": DynamicsModel.DIFF_DRIVE,
    "bicycle": DynamicsModel.KINEMATIC_BICYCLE,
}

DEF_ENVS = (4096,)
DEF_AGENTS = (16, 64)


# --------------------------------------------------------------- registry helpers


def build_scenario(name: str, n_agents: int, model_name: str | None):
    """Construct a scenario for the sweep.

    ``model_name is None`` means "this scenario is holonomic-only" — :func:`_model_axis`
    has already filtered on :func:`~swarp.scenarios.supports_model`, so no ``model``
    kwarg is passed at all. That is deliberate: ``make_scenario`` *warns* when it has to
    drop one, and a sweep should not be generating warnings it knows are expected.
    """
    kw: dict = {"n_agents": n_agents, "world_size": max(1.0, n_agents**0.5 / 4)}
    if model_name is not None:
        kw["model"] = MODELS[model_name]
    return make_scenario(name, **kw)


def _model_axis(name: str) -> list[str | None]:
    """Robot models to sweep for a scenario (holonomic-only -> [None])."""
    if supports_model(SCENARIOS[name]):
        return list(MODELS)
    return [None]


# --------------------------------------------------------------- env construction


def make_env(name, n_envs, n_agents, model_name, optimized, device) -> Environment:
    scen = build_scenario(name, n_agents, model_name)
    env = Environment(
        scen,
        n_envs=n_envs,
        device=device,
        dt=0.05,
        substeps=1,
        seed=0,
        auto_reset=False,
        use_graph=optimized,
        fused=("auto" if optimized else False),
    )
    if not optimized:
        env.world.stepper.neighbor_reuse = False
        env.world.stepper.enable_slim2d = False
    return env


# ------------------------------------------------------------- parity + timing


def _trajectory(env: Environment, n_agents: int, device: str, n_steps: int = 5) -> dict:
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(12345)
    obs_l, rew_l, done_l = [], [], []
    with torch.no_grad():
        for _ in range(n_steps):
            a = torch.empty(env.n_envs, n_agents, env.world.act_dim, device=device)
            a.uniform_(-1.0, 1.0, generator=gen)
            obs, rew, term, trunc, _ = env.step(a)
            obs_l.append(obs.detach().clone())
            rew_l.append(rew.detach().clone())
            done_l.append((term | trunc).detach().clone())
    return {"obs": obs_l, "reward": rew_l, "done": done_l}


def time_env(env, n_agents, device, steps=60, warmup=20, lidar=None) -> float:
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(0)
    a = torch.empty(env.n_envs, n_agents, env.world.act_dim, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            a.uniform_(-1.0, 1.0, generator=gen)
            env.step(a)
            if lidar is not None:
                lidar.scan(env.world)
        sync_device(device)
        t0 = time.perf_counter()
        for _ in range(steps):
            a.uniform_(-1.0, 1.0, generator=gen)
            env.step(a)
            if lidar is not None:
                lidar.scan(env.world)
        sync_device(device)
    return 1e3 * (time.perf_counter() - t0) / steps


def run_config(
    name, model_name, n_envs, n_agents, n_rays, device, steps, warmup
) -> dict:
    """Time baseline vs optimized for one config; parity-checked before timing."""
    label = f"{name}/{model_name or 'holonomic'}"
    if n_rays:
        label += f"+lidar{n_rays}"
    row = {
        "scenario": name,
        "model": model_name or "holonomic",
        "n_envs": n_envs,
        "n_agents": n_agents,
        "lidar_rays": n_rays,
        "ms_base": None,
        "ms_opt": None,
        "speedup": None,
        "graph_mode": None,
        "status": "ok",
        "label": label,
    }
    try:
        base = make_env(name, n_envs, n_agents, model_name, optimized=False, device=device)
        opt = make_env(name, n_envs, n_agents, model_name, optimized=True, device=device)
        # Parity gate: optimized must match baseline on a 5-step seeded rollout, within the
        # tolerance the *scenario* declares (Scenario.parity_rtol / parity_atol) — the same
        # numbers tests/conftest.py's FusedSpec reads.
        cls = SCENARIOS[name]
        ok, why = parity_ok(
            _trajectory(base, n_agents, device),
            _trajectory(opt, n_agents, device),
            cls.parity_rtol,
            cls.parity_atol,
        )
        if not ok:
            row["status"] = "parity-fail"
            row["detail"] = why
            return row
        lidar = None if not n_rays else Lidar(n_rays=n_rays, max_range=1.0)
        row["ms_base"] = time_env(base, n_agents, device, steps, warmup, lidar)
        row["ms_opt"] = time_env(opt, n_agents, device, steps, warmup, lidar)
        row["speedup"] = row["ms_base"] / row["ms_opt"]
        row["graph_mode"] = "graph" if opt.graph_mode else "eager"
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception as exc:  # one bad config must not sink the sweep
        row["status"] = "ERR"
        row["detail"] = f"{type(exc).__name__}: {str(exc)[:80]}"
    return row


def _warmup_all(selected: list[str], device: str) -> None:
    """Compile kernels / capture graphs on tiny envs before the timed sweep."""
    for name in selected:
        for model_name in _model_axis(name):
            for optimized in (False, True):
                try:
                    env = make_env(name, 64, 8, model_name, optimized, device)
                    time_env(env, 8, device, steps=3, warmup=3)
                except Exception:  # warmup is best-effort
                    pass
    sync_device(device)


# ----------------------------------------------------------------------- CLI


def _fmt(row: dict) -> str:
    if row["status"] != "ok":
        base = f"{row['label']:>34} {row['n_agents']:>5} {row['status']:>11}"
        if "detail" in row:
            base += f"  {row['detail']}"
        return base
    return (
        f"{row['label']:>34} {row['n_agents']:>5} "
        f"{row['ms_base']:>9.3f} {row['ms_opt']:>9.3f} {row['speedup']:>6.2f}x "
        f"{row['graph_mode']:>6}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--envs", type=int, nargs="+", default=list(DEF_ENVS))
    parser.add_argument("--agents", type=int, nargs="+", default=list(DEF_AGENTS))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["navigation"],
        help="scenario names, or the literal 'all' (default: navigation only)",
    )
    parser.add_argument(
        "--lidar-rays",
        type=int,
        nargs="+",
        default=[0],
        help="lidar ray counts to add as a per-step cost (0 = off)",
    )
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    try:
        selected = resolve_scenarios(args.scenarios)
    except ValueError as exc:
        parser.error(str(exc))

    if args.device.startswith("cuda"):
        print(f"device: {args.device} ({torch.cuda.get_device_name(args.device)})")
    else:
        print("device: cpu (throughput will be far below GPU numbers)")
    print(f"scenarios: {selected}")
    print(f"timed steps per config: {args.steps}  (ms/step: baseline vs optimized)\n")

    _warmup_all(selected, args.device)

    header = f"{'config':>34} {'agts':>5} {'base':>9} {'opt':>9} {'speedup':>7} {'mode':>6}"
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    for name in selected:
        for model_name in _model_axis(name):
            for n_envs in args.envs:
                for n_agents in args.agents:
                    for n_rays in args.lidar_rays:
                        row = run_config(
                            name,
                            model_name,
                            n_envs,
                            n_agents,
                            n_rays,
                            args.device,
                            args.steps,
                            args.warmup,
                        )
                        rows.append(row)
                        print(_fmt(row))
        print()

    if args.csv:
        fieldnames = [
            "scenario",
            "model",
            "n_envs",
            "n_agents",
            "lidar_rays",
            "ms_base",
            "ms_opt",
            "speedup",
            "graph_mode",
            "status",
        ]
        with open(args.csv, "w", newline="") as f:
            writer = csvmod.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
