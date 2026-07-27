"""Throughput benchmark: env-steps/s and agent-steps/s across batch sizes.

Run with:  python -m wmas.benchmark.throughput [--device cuda:0] [--steps 100]

Steps the NavigationScenario hot path (dynamics + hash-grid neighbors + soft
collisions + obs/reward) under ``torch.no_grad()`` with random actions kept
on-device.

``use_graph`` is pinned rather than left at ``Environment``'s ``"auto"`` so this
benchmark keeps measuring one fixed configuration and its numbers stay comparable
across commits. ``--graph`` adds whole-step CUDA-graph capture (what ``"auto"`` now
selects on a CUDA device); ``wmas.benchmark.scenarios`` reports both side by side.
"""

from __future__ import annotations

import argparse
import time

import torch

from wmas import Environment, NavigationScenario

ENV_COUNTS = (1_000, 4_000, 8_000, 16_000)
AGENT_COUNTS = (4, 16, 64)


def bench_one(
    n_envs: int, n_agents: int, device: str, steps: int, warmup: int = 10, use_graph: bool = False
) -> dict:
    scenario = NavigationScenario(n_agents=n_agents, world_size=max(1.0, n_agents**0.5 / 4))
    env = Environment(
        scenario, n_envs=n_envs, device=device, dt=0.05, substeps=1, seed=0, use_graph=use_graph
    )
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(0)
    actions = torch.empty(n_envs, n_agents, 2, device=device)

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            actions.uniform_(-1.0, 1.0, generator=gen)
            env.step(actions)
        sync()
        t0 = time.perf_counter()
        for _ in range(steps):
            actions.uniform_(-1.0, 1.0, generator=gen)
            env.step(actions)
        sync()
        elapsed = time.perf_counter() - t0

    env_steps_s = n_envs * steps / elapsed
    return {
        "n_envs": n_envs,
        "n_agents": n_agents,
        "env_steps_s": env_steps_s,
        "agent_steps_s": env_steps_s * n_agents,
        "ms_per_step": 1e3 * elapsed / steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--graph",
        action="store_true",
        help="fold obs/reward into a whole-step CUDA graph (Environment's 'auto' default)",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        name = torch.cuda.get_device_name(args.device)
        print(f"device: {args.device} ({name})")
    else:
        print("device: cpu (no GPU in use — throughput will be far below GPU numbers)")
    print(f"timed steps per config: {args.steps}")
    print(f"whole-step CUDA graph: {'on' if args.graph else 'off'}\n")

    header = (
        f"{'n_envs':>8} {'n_agents':>9} {'ms/step':>9} {'env-steps/s':>14} {'agent-steps/s':>15}"
    )
    print(header)
    print("-" * len(header))
    for n_envs in ENV_COUNTS:
        for n_agents in AGENT_COUNTS:
            try:
                r = bench_one(n_envs, n_agents, args.device, args.steps, use_graph=args.graph)
            except torch.cuda.OutOfMemoryError:
                print(f"{n_envs:>8} {n_agents:>9} {'OOM':>9}")
                torch.cuda.empty_cache()
                continue
            print(
                f"{r['n_envs']:>8} {r['n_agents']:>9} {r['ms_per_step']:>9.2f} "
                f"{r['env_steps_s']:>14,.0f} {r['agent_steps_s']:>15,.0f}"
            )


if __name__ == "__main__":
    main()
