"""Cross-simulator throughput benchmark: swarp vs VMAS, JaxMARL, and CAMAR.

Compares raw simulator **env-steps/s** and **agent-steps/s** on the one
environment all four implement: continuous 2D navigation-to-goal with collision
avoidance. Only the simulator step is measured — not RL algorithms, wrappers, or
task equivalence.

The default ``--sims`` set is chosen to be as apples-to-apples as possible — all
four run obstacle-free navigation with raycasting-free, relative-position
observations: ``swarp`` (fused kernels + whole-step CUDA graph), ``vmas-nolidar`` (VMAS
navigation with ``collisions=False``, so no 12-ray lidar), ``jaxmarl``
(``MPE_simple_spread_v3`` continuous), and ``camar`` (an open ``string_grid``
arena, ``frameskip=0`` — one world step per env-step, == swarp substeps=1). Each
sim's *native* setup is also available:
``vmas`` (collisions + lidar) and ``camar-grid`` (~800-obstacle ``random_grid``),
plus swarp hot-path ablation configs ``swarp-fused`` / ``swarp-eager``.

Why subprocesses. VMAS pins ``numpy < 2`` (torch stack) while JaxMARL and CAMAR
are JAX-based and want recent numpy/jax — the four cannot coexist in one venv,
and torch + JAX in one process fight over VRAM (JAX preallocates by default). So
each simulator is benchmarked in its own subprocess, launched with an interpreter
from a venv that can import it. The numpy split lives entirely at the subprocess
boundary; the final table is a single combined report anchored on swarp.

Install one Python 3.12 venv per simulator (VMAS shares swarp's)::

    for v in .venv .venv-jaxmarl .venv-camar; do uv venv --python 3.12 $v; done
    VIRTUAL_ENV=.venv         uv pip install -e . --group dev --group bench
    VIRTUAL_ENV=.venv-jaxmarl uv pip install -e . --group bench-jaxmarl
    VIRTUAL_ENV=.venv-camar   uv pip install -e . --group bench-camar

If JaxMARL pulls a CPU-only jaxlib, add the matching CUDA build into its venv,
e.g. ``VIRTUAL_ENV=.venv-jaxmarl uv pip install "jax[cuda12]==<jaxmarl's jax>"``.

Run (single combined command)::

    .venv/bin/python -m swarp.benchmark.compare_sims --device cuda:0 \
        --envs 256 1024 4096 --agents 3 \
        --python jaxmarl=.venv-jaxmarl/bin/python \
        --python camar=.venv-camar/bin/python

The parent process imports only the standard library; each ``--python SIM=PATH``
maps a simulator to its interpreter (default: the current interpreter).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# swarp is the anchor: it appears in every table, ratios are computed against it.
# The three swarp entries are hot-path *configurations* of the same simulator:
#   swarp        — optimized: fused Warp obs/reward kernels folded into a
#                 whole-step CUDA graph (physics + neighbor query + obs/reward
#                 captured together, so each step is one graph replay with no
#                 per-launch host floor)
#   swarp-fused  — fused kernels only (no CUDA graph)
#   swarp-eager  — baseline: torch obs/reward, no fused kernels, no graph
SWARP_CONFIGS = {
    "swarp": dict(fused="auto", use_graph=True),
    "swarp-fused": dict(fused="auto", use_graph=False),
    "swarp-eager": dict(fused=False, use_graph=False),
}
# vmas/camar each have a native and an obs-aligned variant:
#   vmas          native navigation (collision physics + 12-ray lidar obs)
#   vmas-nolidar  collisions+lidar off -> relative-position obs (raycasting-free)
#   camar         open arena (no obstacles, frameskip=0 = one world step) -> swarp navigation
#   camar-grid    native random_grid (~800 obstacles, frameskip=2) -> maze task
ALL_SIMS = (*SWARP_CONFIGS, "vmas", "vmas-nolidar", "jaxmarl", "camar", "camar-grid")
# Default is the obs-aligned, obstacle-free set: same task + raycasting-free obs.
DEFAULT_SIMS = ("swarp", "vmas-nolidar", "jaxmarl", "camar")
SCENARIOS = ("navigation",)


# --------------------------------------------------------------------- child side


def _build(sim: str, scenario: str, n_envs: int, n_agents: int, device: str):
    """Dispatch to the per-sim adapter (imported lazily, in the child only)."""
    if sim in SWARP_CONFIGS:
        from swarp.benchmark._adapters import swarp_adapter

        return swarp_adapter.build(scenario, n_envs, n_agents, device, **SWARP_CONFIGS[sim])
    if sim in ("vmas", "vmas-nolidar"):
        from swarp.benchmark._adapters import vmas_adapter

        return vmas_adapter.build(
            scenario, n_envs, n_agents, device, collisions=(sim == "vmas")
        )
    if sim == "jaxmarl":
        from swarp.benchmark._adapters import jaxmarl_adapter

        return jaxmarl_adapter.build(scenario, n_envs, n_agents, device)
    if sim in ("camar", "camar-grid"):
        from swarp.benchmark._adapters import camar_adapter

        return camar_adapter.build(
            scenario, n_envs, n_agents, device, obstacles=(sim == "camar-grid")
        )
    raise ValueError(f"unknown sim {sim!r}")


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "resource_exhausted" in msg


def bench_throughput(runner, steps: int, warmup: int) -> float:
    """env-steps/s over ``steps`` timed steps. Backend-agnostic.

    Warms up twice: once for generic device spin-up, then once at the *timed*
    length so a JAX backend's fixed-length ``lax.scan`` is already compiled when
    the timer starts (its jit cache is keyed on the step count).
    """
    import time

    runner.rollout(warmup)
    runner.sync()
    runner.rollout(steps)  # compile the timed-length scan (JAX) — still untimed
    runner.sync()
    t0 = time.perf_counter()
    runner.rollout(steps)
    runner.sync()
    elapsed = time.perf_counter() - t0
    return runner.n_envs * steps / elapsed


def _child(sim, scenario, n_envs, n_agents, steps, warmup, device) -> None:
    """Benchmark one (sim, config) and print a single machine-readable line."""
    # XLA reads these at first `import jax`, so set them before the adapter imports.
    if sim in ("jaxmarl", "camar"):
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from swarp.benchmark._adapters import CpuOnlyError

    try:
        runner = _build(sim, scenario, n_envs, n_agents, device)
        eps = bench_throughput(runner, steps, warmup)
        runner.close()
    except CpuOnlyError:
        print("CPU-only")
        return
    except Exception as e:  # noqa: BLE001 — sentinel for the parent; traceback -> stderr
        if _is_oom(e):
            print("OOM")
            return
        raise
    print(f"RESULT eps={eps:.6f} agents={runner.n_agents}")


# --------------------------------------------------------------------- parent side


def _run_child(python: str, sim, scenario, ne, na, steps, warmup, device) -> dict:
    """Spawn a child benchmark and parse its one-line result."""
    cmd = [
        python, "-m", "swarp.benchmark.compare_sims", "--_child",
        sim, scenario, str(ne), str(na), str(steps), str(warmup), device,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT"):
            kv = dict(p.split("=", 1) for p in line.split()[1:])
            return {"eps": float(kv["eps"]), "agents": int(kv["agents"])}
        if line.strip() in ("OOM", "CPU-only"):
            return {"status": line.strip()}
    sys.stderr.write(f"[{sim} {scenario} {ne}x{na}] no RESULT; stderr tail:\n")
    sys.stderr.write(out.stderr[-800:] + "\n")
    return {"status": "ERR"}


def _value(r: dict, per_agent: bool) -> float | None:
    """The metric value for a result cell: env-steps/s, or agent-steps/s."""
    if "eps" not in r:
        return None
    return r["eps"] * r["agents"] if per_agent else r["eps"]


def _fmt_cell(r: dict, requested_agents: int, per_agent: bool) -> str:
    if "status" in r:
        return r["status"]
    s = f"{_value(r, per_agent):,.0f}"
    return s + "*" if r["agents"] != requested_agents else s


def _print_table(label: str, per_agent: bool, sims, grid, configs, _w) -> None:
    comps = [s for s in sims if s != "swarp"]
    has_anchor = "swarp" in sims and bool(comps)
    print(f"metric: {label} (higher is better)" + ("   anchor: swarp" if has_anchor else ""))
    head = f"{'n_envs':>8} {'n_agents':>9} | " + " ".join(f"{s:>{_w}}" for s in sims)
    if has_anchor:
        head += " | " + " ".join(f"{s + '/s':>{_w}}" for s in comps)
    print(head)
    print("-" * len(head))
    for na, ne in configs:
        cells = grid[(na, ne)]
        row = f"{ne:>8} {na:>9} | " + " ".join(
            f"{_fmt_cell(cells[s], na, per_agent):>{_w}}" for s in sims
        )
        if has_anchor:
            anchor = _value(cells["swarp"], per_agent)
            ratios = []
            for s in comps:
                v = _value(cells[s], per_agent)
                ratios.append(f"{v / anchor:.2f}x" if (v and anchor) else "-")
            row += " | " + " ".join(f"{x:>{_w}}" for x in ratios)
        print(row)
    print()


def run(args) -> None:
    py = dict(args.python or [])
    default_py = args.python_default or sys.executable
    sims = list(args.sims)
    _w = max(11, max(len(s) for s in sims) + 2)  # +2 so the "<sim>/s" ratio headers fit

    print("anchor: swarp (Warp).  ratios = competitor / swarp")
    print(f"device: {args.device}  steps {args.steps}  warmup {args.warmup}")
    for sim in sims:
        print(f"  {sim:<12} -> {py.get(sim, default_py)}")
    print()

    mism: dict[str, int] = {}  # sim -> realized agents, when != requested
    configs = [(na, ne) for na in args.agents for ne in args.envs]
    for scenario in args.scenarios:
        grid: dict[tuple[int, int], dict[str, dict]] = {}
        for na, ne in configs:
            cells: dict[str, dict] = {}
            for sim in sims:
                r = _run_child(
                    py.get(sim, default_py), sim, scenario, ne, na,
                    args.steps, args.warmup, args.device,
                )
                cells[sim] = r
                if "agents" in r and r["agents"] != na:
                    mism[sim] = r["agents"]
            grid[(na, ne)] = cells

        print(f"[{scenario}]")
        _print_table("env-steps/s", False, sims, grid, configs, _w)
        _print_table("agent-steps/s", True, sims, grid, configs, _w)

    if mism:
        print("* realized agent count differs from requested (sim-fixed):")
        for sim, n in mism.items():
            print(f"    {sim}: {n} agents")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--envs", type=int, nargs="+", default=[256, 1024, 4096])
    p.add_argument("--agents", type=int, nargs="+", default=[3])
    p.add_argument(
        "--sims", nargs="+", default=list(DEFAULT_SIMS), choices=list(ALL_SIMS),
        help="simulators to benchmark (swarp is the anchor; default: swarp vmas jaxmarl camar)",
    )
    p.add_argument(
        "--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS),
    )
    p.add_argument(
        "--python", action="append", type=_sim_path, metavar="SIM=PATH", default=[],
        help="map a sim to its venv interpreter, e.g. --python jaxmarl=.venv-jaxmarl/bin/python",
    )
    p.add_argument(
        "--python-default", default=None,
        help="interpreter for sims without an explicit --python (default: current)",
    )
    # Hidden: single-config benchmark run in a child process.
    p.add_argument(
        "--_child", nargs=7, default=None,
        metavar=("SIM", "SCENARIO", "N_ENVS", "N_AGENTS", "STEPS", "WARMUP", "DEVICE"),
        help=argparse.SUPPRESS,
    )
    args = p.parse_args()

    if args._child:
        sim, scenario, ne, na, steps, warmup, device = args._child
        _child(sim, scenario, int(ne), int(na), int(steps), int(warmup), device)
        return

    run(args)


def _sim_path(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected SIM=PATH, got {s!r}")
    sim, path = s.split("=", 1)
    if sim not in ALL_SIMS:
        raise argparse.ArgumentTypeError(f"unknown sim {sim!r}; choose from {ALL_SIMS}")
    return sim, path


if __name__ == "__main__":
    main()
