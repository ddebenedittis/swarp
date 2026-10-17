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
    scenario: str,
    n_envs: int,
    n_agents: int,
    device: str,
    seed: int = 0,
    collisions: bool = True,
    n_vehicles: int | None = None,
) -> Runner:
    if scenario == "ringroad":
        return _ringroad(n_envs, n_agents, device, seed, n_vehicles)
    if scenario != "navigation":
        raise ValueError(f"vmas adapter implements 'navigation' and 'ringroad', got {scenario!r}")
    env, step = make_vmas(n_envs, n_agents, device, collisions=collisions)
    return make_torch_runner(env, step, n_envs, len(env.agents), device)


def _ringroad(n_envs: int, n_agents: int, device: str, seed: int, n_vehicles=None) -> Runner:
    """The VMAS port of the ring road (:mod:`.ringroad_vmas`), random actions over its box."""
    import torch
    from vmas import make_env

    from swarp.benchmark._adapters.ringroad_vmas import RingRoadVMAS

    env = make_env(
        scenario=RingRoadVMAS(),
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        seed=seed,
        n_agents=n_agents,
        **({} if n_vehicles is None else {"n_vehicles": n_vehicles}),
    )
    env.reset()
    gen = torch.Generator(device=device).manual_seed(seed)
    hi = [a.action.u_range_tensor.to(device) for a in env.agents]
    actions = [torch.empty(n_envs, 2, device=device) for _ in env.agents]

    def step() -> None:
        for a, h in zip(actions, hi, strict=True):
            a.uniform_(-1.0, 1.0, generator=gen).mul_(h)
        env.step(actions)

    return make_torch_runner(env, step, n_envs, len(env.agents), device)
