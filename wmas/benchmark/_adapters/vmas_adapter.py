"""VMAS adapter: reuses the ``_make_vmas`` navigation builder from ``compare_vmas``.

Uses VMAS's *native* navigation config (``collisions=True`` — collision physics
plus the default 12-ray lidar observation), i.e. VMAS as shipped. Observation
models differ across all four simulators by design; this measures step
throughput, not task equivalence.
"""

from __future__ import annotations

from wmas.benchmark._adapters import Runner, make_torch_runner
from wmas.benchmark.compare_vmas import _make_vmas


def build(scenario: str, n_envs: int, n_agents: int, device: str, seed: int = 0) -> Runner:
    if scenario != "navigation":
        raise ValueError(f"vmas adapter only implements 'navigation', got {scenario!r}")
    env, step = _make_vmas(n_envs, n_agents, device, collisions=True)
    return make_torch_runner(env, step, n_envs, len(env.agents), device)
