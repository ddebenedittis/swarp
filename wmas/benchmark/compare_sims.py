"""Cross-simulator throughput benchmark: wmas vs VMAS, JaxMARL, and CAMAR.

Compares raw simulator **env-steps/s** on the one environment all four
implement: continuous 2D navigation-to-goal with collision avoidance
(wmas ``NavigationScenario`` / VMAS ``navigation`` / JaxMARL
``MPE_simple_spread_v3`` continuous / CAMAR ``random_grid`` + ``HolonomicDynamic``).
Only the simulator step is measured — not RL algorithms, wrappers, or task
equivalence. Observation and reward models differ across sims by design, exactly
as in ``compare_vmas.py``; this measures how fast each engine advances a batch of
environments, nothing more.

Why subprocesses. VMAS pins ``numpy < 2`` (torch stack) while JaxMARL and CAMAR
are JAX-based and want recent numpy/jax — the four cannot coexist in one venv,
and torch + JAX in one process fight over VRAM (JAX preallocates by default). So
each simulator is benchmarked in its own subprocess, launched with an interpreter
from a venv that can import it. The numpy split lives entirely at the subprocess
boundary; the final table is a single combined report anchored on wmas.

Install one Python 3.12 venv per simulator (VMAS shares wmas's)::

    for v in .venv .venv-jaxmarl .venv-camar; do uv venv --python 3.12 $v; done
    VIRTUAL_ENV=.venv         uv pip install -e . --group dev --group bench
    VIRTUAL_ENV=.venv-jaxmarl uv pip install -e . --group bench-jaxmarl
    VIRTUAL_ENV=.venv-camar   uv pip install -e . --group bench-camar

If JaxMARL pulls a CPU-only jaxlib, add the matching CUDA build into its venv,
e.g. ``VIRTUAL_ENV=.venv-jaxmarl uv pip install "jax[cuda12]==<jaxmarl's jax>"``.

Run (single combined command)::

    .venv/bin/python -m wmas.benchmark.compare_sims --device cuda:0 \
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

# wmas is the anchor: it appears in every table, ratios are computed against it.
# The three wmas entries are hot-path *configurations* of the same simulator:
#   wmas        — optimized: fused Warp obs/reward kernels + CUDA-graph capture
#   wmas-fused  — fused kernels only (no CUDA graph)
#   wmas-eager  — baseline: torch obs/reward, no fused kernels, no graph
WMAS_CONFIGS = {
    "wmas": dict(fused="auto", use_graph=True),
    "wmas-fused": dict(fused="auto", use_graph=False),
    "wmas-eager": dict(fused=False, use_graph=False),
}
ALL_SIMS = (*WMAS_CONFIGS, "vmas", "jaxmarl", "camar")
DEFAULT_SIMS = ("wmas", "vmas", "jaxmarl", "camar")
SCENARIOS = ("navigation",)


# --------------------------------------------------------------------- child side


def _build(sim: str, scenario: str, n_envs: int, n_agents: int, device: str):
    """Dispatch to the per-sim adapter (imported lazily, in the child only)."""
    if sim in WMAS_CONFIGS:
        from wmas.benchmark._adapters import wmas_adapter

        return wmas_adapter.build(scenario, n_envs, n_agents, device, **WMAS_CONFIGS[sim])
    if sim == "vmas":
        from wmas.benchmark._adapters import vmas_adapter

        return vmas_adapter.build(scenario, n_envs, n_agents, device)
    if sim == "jaxmarl":
        from wmas.benchmark._adapters import jaxmarl_adapter

        return jaxmarl_adapter.build(scenario, n_envs, n_agents, device)
    if sim == "camar":
        from wmas.benchmark._adapters import camar_adapter

        return camar_adapter.build(scenario, n_envs, n_agents, device)
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

    from wmas.benchmark._adapters import CpuOnlyError

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
        python, "-m", "wmas.benchmark.compare_sims", "--_child",
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


def _fmt_cell(r: dict, requested_agents: int) -> str:
    if "status" in r:
        return r["status"]
    s = f"{r['eps']:,.0f}"
    return s + "*" if r["agents"] != requested_agents else s


def run(args) -> None:
    py = dict(args.python or [])
    default_py = args.python_default or sys.executable
    sims = list(args.sims)

    print("metric: env-steps/s (higher is better)")
    print("anchor: wmas (Warp).  ratios = competitor / wmas")
    print(f"device: {args.device}  steps {args.steps}  warmup {args.warmup}")
    for sim in sims:
        print(f"  {sim:<11} -> {py.get(sim, default_py)}")
    print()

    mism: dict[str, int] = {}  # sim -> realized agents, when != requested
    for scenario in args.scenarios:
        _w = max(11, max(len(s) for s in sims))
        head = f"{'n_envs':>8} {'n_agents':>9} | " + " ".join(f"{s:>{_w}}" for s in sims)
        comps = [s for s in sims if s != "wmas"]
        if "wmas" in sims and comps:
            head += " | " + " ".join(f"{s + '/w':>{_w}}" for s in comps)
        print(f"[{scenario}]")
        print(head)
        print("-" * len(head))

        for na in args.agents:
            for ne in args.envs:
                cells: dict[str, dict] = {}
                for sim in sims:
                    r = _run_child(
                        py.get(sim, default_py), sim, scenario, ne, na,
                        args.steps, args.warmup, args.device,
                    )
                    cells[sim] = r
                    if "agents" in r and r["agents"] != na:
                        mism[sim] = r["agents"]

                row = f"{ne:>8} {na:>9} | " + " ".join(
                    f"{_fmt_cell(cells[s], na):>{_w}}" for s in sims
                )
                if "wmas" in sims and comps:
                    w = cells["wmas"].get("eps")
                    ratios = []
                    for s in comps:
                        e = cells[s].get("eps")
                        ratios.append(f"{e / w:.2f}x" if (e and w) else "-")
                    row += " | " + " ".join(f"{x:>{_w}}" for x in ratios)
                print(row)
        print()

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
        help="simulators to benchmark (wmas is the anchor; default: wmas vmas jaxmarl camar)",
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
