"""End-to-end training throughput: the torchrl reference trainer vs ``swarp.rl``.

Run with::

    python -m swarp.benchmark.train_throughput [--scenarios navigation giveway]
        [--n-envs 1024 4096 16384] [--iters 25] [--warmup 5]

Each configuration runs its trainer's real CLI in a subprocess for ``--iters`` iterations
and reads the per-iteration ``wall_s`` column of the ``metrics.csv`` it writes. Both
trainers time collection *and* update (logging, eval and checkpoints excluded), so the
numbers are what a training run pays. The median over the post-warm-up iterations is
reported as ms/iteration and end-to-end env-steps/s.

Configurations (``--configs``):

``torchrl``
    ``examples/train_mappo.py`` (needs the ``torchrl`` extra; skipped otherwise).
``eager``
    ``python -m swarp.rl --no-graphs``: the lean loop, no CUDA graphs.
``graphs``
    ``python -m swarp.rl``: rollout and update CUDA-graph captured (the default).
``fast``
    ``graphs`` plus ``--compile --tf32``: the fastest configuration.
``rsl_rl``
    ``examples/train_rsl_rl.py``: rsl_rl's PPO and ``OnPolicyRunner`` with one rsl_rl env
    per agent (needs the ``rsl-rl`` extra; skipped otherwise).

``sim`` is the simulator alone over the same rollout (``steps_per_batch`` steps of random
actions at the same ``n_envs``, ``auto_reset=True``): the ceiling no trainer can beat, and
the share of an iteration that is physics.

Every trainer is a CLI that takes the shared flags below and writes ``wall_s`` per
iteration to ``<checkpoint-dir>/metrics.csv``; ``TRAINERS`` maps a config name to its
command. A trainer whose library cannot share this venv runs under its own interpreter
via ``--python CONFIG=PATH``.

``--n-envs`` overrides the spec's batch; ``--minibatch-samples`` then keeps the minibatch
size fixed (so the update stays within memory and the number of minibatches grows with the
batch) instead of the spec's fixed number of minibatches.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from swarp.rl.recipes import SPECS, build_env

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Trainer:
    """A trainer CLI: ``args`` follow the interpreter; ``requires`` is the module whose
    absence skips the config when it runs under the current interpreter."""

    args: tuple[str, ...]
    requires: str | None = None


TRAINERS: dict[str, Trainer] = {
    "torchrl": Trainer((str(REPO / "examples" / "train_mappo.py"),), "torchrl"),
    "eager": Trainer(("-m", "swarp.rl", "--no-graphs")),
    "graphs": Trainer(("-m", "swarp.rl")),
    "fast": Trainer(("-m", "swarp.rl", "--compile", "--tf32")),
    "rsl_rl": Trainer((str(REPO / "examples" / "train_rsl_rl.py"),), "rsl_rl"),
}
CONFIGS = tuple(TRAINERS)


def _config_path(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected CONFIG=PATH, got {s!r}")
    config, path = s.split("=", 1)
    if config not in TRAINERS:
        raise argparse.ArgumentTypeError(f"unknown config {config!r}; choose from {CONFIGS}")
    return config, path


def run_trainer(
    config: str, scenario: str, n_envs: int | None, minibatches: int | None, iters: int,
    warmup: int, device: str, python: str = sys.executable,
) -> float | None:
    """Median seconds per iteration after ``warmup`` iterations; ``None`` on CUDA OOM."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            python, *TRAINERS[config].args,
            "--scenario", scenario, "--device", device, "--iters", str(iters),
            "--eval-every", "0", "--checkpoint-every", "0", "--checkpoint-dir", tmp,
        ]
        if n_envs is not None:
            cmd += ["--n-envs", str(n_envs)]
        if minibatches is not None:
            cmd += ["--minibatches", str(minibatches)]
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if proc.returncode != 0 and "OutOfMemoryError" in proc.stderr:
            return None
        if proc.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stderr[-2000:]}")
        with (Path(tmp) / "metrics.csv").open() as fh:
            walls = [float(r["wall_s"]) for r in csv.DictReader(fh)]
    return statistics.median(walls[warmup:])


def sim_only(scenario: str, n_envs: int, steps: int, device: str, repeats: int = 5) -> float:
    """Seconds for ``steps`` env steps with random actions: the rollout's physics alone."""
    spec = SPECS[scenario]
    env = build_env(
        scenario, n_envs=n_envs, n_agents=spec.n_agents, device=device, dt=spec.dt,
        substeps=spec.substeps, max_steps=spec.max_steps, seed=0,
        scen_kwargs=spec.scen_kwargs, env_kwargs={"auto_reset": True},
    )
    env.reset()
    actions = torch.empty(n_envs, spec.n_agents, env.act_dim, device=device)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    times = []
    with torch.no_grad():
        for r in range(repeats + 1):
            sync()
            t0 = time.perf_counter()
            for _ in range(steps):
                env.step(actions.uniform_(-1.0, 1.0))
            sync()
            if r:  # the first pass captures the env's graph
                times.append(time.perf_counter() - t0)
    env.close()
    return statistics.median(times)


def main(argv: list[str] | None = None) -> None:
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--device", default=default_device)
    p.add_argument("--scenarios", nargs="+", default=["navigation", "giveway"],
                   choices=sorted(SPECS))
    p.add_argument("--n-envs", nargs="+", type=int, help="default: each spec's n_envs")
    p.add_argument("--minibatch-samples", type=int,
                   help="fix the minibatch size (env-steps) instead of the minibatch count")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--python", action="append", type=_config_path, metavar="CONFIG=PATH", default=[],
        help="run a config under another venv's interpreter, e.g. "
             "--python benchmarl=.venv-benchmarl/bin/python",
    )
    a = p.parse_args(argv)

    pythons = dict(a.python)
    configs = []
    for config in a.configs:
        req = TRAINERS[config].requires
        if config not in pythons and req and importlib.util.find_spec(req) is None:
            print(f"{req} not installed: skipping the {config} config")
            continue
        configs.append(config)

    header = f"{'scenario':12s} {'n_envs':>7s} {'mb':>4s} {'config':>8s} " \
             f"{'ms/iter':>9s} {'env-steps/s':>13s} {'vs torchrl':>10s} {'sim share':>9s}"
    print(header)
    print("-" * len(header))
    for scenario in a.scenarios:
        spec = SPECS[scenario]
        for n_envs in a.n_envs or [spec.n_envs]:
            frames = n_envs * spec.steps_per_batch
            minibatches = None
            if a.minibatch_samples:
                minibatches = max(1, frames // a.minibatch_samples)
            mb = minibatches or spec.minibatches
            sim = sim_only(scenario, n_envs, spec.steps_per_batch, a.device)
            base = None
            for config in configs:
                wall = run_trainer(
                    config, scenario, n_envs, minibatches, a.iters, a.warmup, a.device,
                    pythons.get(config, sys.executable),
                )
                if wall is None:
                    print(f"{scenario:12s} {n_envs:7d} {mb:4d} {config:>8s} {'OOM':>9s}",
                          flush=True)
                    continue
                if config == "torchrl":
                    base = wall
                ratio = "" if base is None else f"{base / wall:9.2f}x"
                print(
                    f"{scenario:12s} {n_envs:7d} {mb:4d} {config:>8s} {wall * 1e3:9.1f} "
                    f"{frames / wall:13,.0f} {ratio:>10s} {sim / wall:9.1%}",
                    flush=True,
                )
            print(
                f"{scenario:12s} {n_envs:7d} {'':4s} {'sim':>8s} {sim * 1e3:9.1f} "
                f"{frames / sim:13,.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
