"""Head-to-head benchmark: swarp vs VMAS on the navigation scenario.

VMAS: https://github.com/proroklab/VectorizedMultiAgentSimulator

Compares env-steps/s and peak device memory on the *navigation* scenario (the
one both simulators implement) across a grid of ``(n_envs, n_agents)``. Three
workloads are timed:

* ``swarp``        — NavigationScenario (soft collisions + neighbor-list obs, Warp kernels)
* ``vmas-lidar``  — VMAS ``navigation``, ``collisions=True`` (collisions + 12-ray lidar); default
* ``vmas-simple`` — VMAS ``navigation``, ``collisions=False`` (pure dynamics + goal obs, no lidar)

The comparison is deliberately not perfectly apples-to-apples: the observation
models differ by design (VMAS ships lidar; swarp uses padded neighbor lists),
which is why ``vmas-simple`` is included as a lidar-free lower bound.

Requires the ``bench`` dependency group (installs ``vmas``)::

    uv pip install -e '.[bench]'
    uv run python -m swarp.benchmark.compare_vmas            # throughput grid
    uv run python -m swarp.benchmark.compare_vmas --metric memory
    uv run python -m swarp.benchmark.compare_vmas --metric both

Timing: warmup, ``cuda.synchronize``, ``torch.no_grad``, random actions kept
on-device; ``env-steps/s = n_envs * timed_steps / wall_time``. Memory is peak
driver-level device usage (``mem_get_info`` delta), measured in isolated
subprocesses so it counts both PyTorch's caching allocator and Warp's mempool.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import torch


def _ensure_vmas() -> None:
    """Import guard: fall back to a vendored VMAS checkout if not pip-installed."""
    try:
        import vmas  # noqa: F401

        return
    except ImportError:
        pass
    from pathlib import Path

    # parents[2] is the repo root; the vendored copy lives at <root>/VectorizedMultiAgentSimulator
    vendored = Path(__file__).resolve().parents[2] / "VectorizedMultiAgentSimulator"
    if (vendored / "vmas" / "__init__.py").exists():
        sys.path.insert(0, str(vendored))
        return
    raise ImportError(
        "vmas not found. Install the bench group (`uv pip install -e '.[bench]'`) "
        "or place a VMAS checkout at ./VectorizedMultiAgentSimulator."
    )


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


# --------------------------------------------------------------------- builders


def _make_swarp(n_envs: int, n_agents: int, device: str):
    from swarp import Environment, NavigationScenario

    scenario = NavigationScenario(n_agents=n_agents, world_size=max(1.0, n_agents**0.5 / 4))
    env = Environment(scenario, n_envs=n_envs, device=device, dt=0.05, substeps=1, seed=0)
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(0)
    actions = torch.empty(n_envs, n_agents, 2, device=device)

    def step():
        actions.uniform_(-1.0, 1.0, generator=gen)
        env.step(actions)

    return env, step


def make_vmas(n_envs: int, n_agents: int, device: str, collisions: bool):
    """Build the VMAS ``navigation`` baseline and its step closure.

    Returns ``(env, step)``. Shared with
    :mod:`swarp.benchmark._adapters.vmas_adapter` so the cross-simulator table and this
    module's own comparison time the *same* baseline configuration.
    """
    _ensure_vmas()
    from vmas import make_env

    env = make_env(
        scenario="navigation",
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        n_agents=n_agents,
        collisions=collisions,
    )
    env.reset()
    gen = torch.Generator(device=device).manual_seed(0)
    actions = [torch.empty(n_envs, a.action.action_size, device=device) for a in env.agents]

    def step():
        for a in actions:
            a.uniform_(-1.0, 1.0, generator=gen)
        env.step(actions)

    return env, step


_BUILDERS = {
    "swarp": lambda ne, na, dev: _make_swarp(ne, na, dev),
    "vmas-lidar": lambda ne, na, dev: make_vmas(ne, na, dev, True),
    "vmas-simple": lambda ne, na, dev: make_vmas(ne, na, dev, False),
}
SIMS = tuple(_BUILDERS)


# --------------------------------------------------------------------- throughput


def bench_throughput(sim: str, n_envs: int, n_agents: int, device: str, steps: int, warmup: int):
    env, step = _BUILDERS[sim](n_envs, n_agents, device)
    with torch.no_grad():
        for _ in range(warmup):
            step()
        sync(device)
        t0 = time.perf_counter()
        for _ in range(steps):
            step()
        sync(device)
        elapsed = time.perf_counter() - t0
    del env
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return n_envs * steps / elapsed


def _try(fn, *fn_args):
    try:
        return fn(*fn_args)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None
        raise


def run_throughput(args) -> None:
    print("metric: env-steps/s (higher is better)\n")
    hdr = (
        f"{'n_envs':>7} {'n_agents':>8} | {'swarp':>13} | {'vmas-lidar':>13} "
        f"| {'vmas-simple':>13} | {'swarp/lidar':>11} {'swarp/simple':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for na in args.agents:
        for ne in args.envs:
            r = {
                s: _try(bench_throughput, s, ne, na, args.device, args.steps, args.warmup)
                for s in SIMS
            }

            def f(v):
                return "OOM" if v is None else f"{v:,.0f}"

            w, vl, vs = r["swarp"], r["vmas-lidar"], r["vmas-simple"]
            rl = f"{w / vl:.2f}x" if (w and vl) else "-"
            rs = f"{w / vs:.2f}x" if (w and vs) else "-"
            print(f"{ne:>7} {na:>8} | {f(w):>13} | {f(vl):>13} | {f(vs):>13} | {rl:>11} {rs:>12}")


# --------------------------------------------------------------------- memory


def _free_mib() -> float:
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0] / 2**20


def _mem_child(sim: str, n_envs: int, n_agents: int, steps: int) -> None:
    """Print peak driver-level device usage for one config (own process)."""
    torch.zeros(1, device="cuda:0")  # establish torch cuda context
    if sim == "swarp":
        import warp as wp

        wp.init()
        wp.zeros(1, dtype=wp.float32, device="cuda:0")  # force warp mempool up
    base = _free_mib()
    env, step = _BUILDERS[sim](n_envs, n_agents, "cuda:0")
    lo = _free_mib()
    with torch.no_grad():
        for _ in range(steps):
            step()
            lo = min(lo, _free_mib())
    del env
    print(f"DELTA_MIB {base - lo:.1f}")


def run_memory(args) -> None:
    if not args.device.startswith("cuda"):
        print("memory metric requires a CUDA device; skipping.")
        return
    print("metric: peak device memory used, MiB (lower is better)")
    print("(driver-level mem_get_info delta; counts torch + Warp allocators)\n")
    cols = f"{'swarp':>10} | {'vmas-lidar':>11} | {'vmas-simple':>12}"
    hdr = f"{'n_envs':>7} {'n_agents':>8} | {cols}"
    print(hdr)
    print("-" * len(hdr))
    for na in args.agents:
        for ne in args.envs:
            vals = {}
            for sim in SIMS:
                out = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "swarp.benchmark.compare_vmas",
                        "--_mem_child",
                        sim,
                        str(ne),
                        str(na),
                        str(args.steps),
                    ],
                    capture_output=True,
                    text=True,
                )
                mib = next(
                    (
                        float(line.split()[1])
                        for line in out.stdout.splitlines()
                        if line.startswith("DELTA_MIB")
                    ),
                    None,
                )
                if mib is None:
                    sys.stderr.write(out.stderr[-500:] + "\n")
                vals[sim] = mib

            def f(v):
                return "err" if v is None else f"{v:.1f}"

            print(
                f"{ne:>7} {na:>8} | {f(vals['swarp']):>10} "
                f"| {f(vals['vmas-lidar']):>11} | {f(vals['vmas-simple']):>12}"
            )


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    """The CLI, split out from :func:`main` so a test can build it without running."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", default=default_device)
    p.add_argument("--metric", choices=["throughput", "memory", "both"], default="throughput")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--envs", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    p.add_argument("--agents", type=int, nargs="+", default=[4, 16])
    # Hidden: single-config memory probe run in a child process.
    p.add_argument(
        "--_mem_child",
        nargs=4,
        metavar=("SIM", "N_ENVS", "N_AGENTS", "STEPS"),
        help=argparse.SUPPRESS,
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args._mem_child:
        sim, ne, na, steps = args._mem_child
        _mem_child(sim, int(ne), int(na), int(steps))
        return

    if args.device.startswith("cuda"):
        print(f"device: {args.device} ({torch.cuda.get_device_name(args.device)})")
    else:
        print("device: cpu (throughput will be far below GPU numbers)")

    if args.metric in ("throughput", "both"):
        print(f"timed steps/config: {args.steps} (warmup {args.warmup})")
        run_throughput(args)
    if args.metric in ("memory", "both"):
        print()
        run_memory(args)


if __name__ == "__main__":
    main()
