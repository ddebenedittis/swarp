"""VMAS adapter: reuses the ``make_vmas`` navigation builder from ``compare_vmas``.

``collisions`` toggles VMAS's coupled collisions+lidar:

* ``collisions=True`` (``vmas``) — VMAS's native navigation: collision physics plus
  the default 12-ray lidar observation.
* ``collisions=False`` (``vmas-nolidar``) — no lidar, so observations reduce to
  relative positions (own pose/vel + goal + other agents), matching the
  raycasting-free obs of swarp / JaxMARL / open-arena CAMAR. (VMAS's stock
  navigation gates collisions and lidar on the same flag, so this also drops
  agent-agent collision physics — noted in the report.)
"""

from __future__ import annotations

from swarp.benchmark._adapters import Runner, make_torch_runner
from swarp.benchmark.compare_vmas import make_vmas


def build(
    scenario: str, n_envs: int, n_agents: int, device: str, seed: int = 0, collisions: bool = True
) -> Runner:
    if scenario != "navigation":
        raise ValueError(f"vmas adapter only implements 'navigation', got {scenario!r}")
    env, step = make_vmas(n_envs, n_agents, device, collisions=collisions)
    return make_torch_runner(env, step, n_envs, len(env.agents), device)
