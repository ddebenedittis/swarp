"""wmas adapter with explicit hot-path configuration knobs.

Builds ``NavigationScenario`` directly (rather than via ``compare_vmas._make_wmas``)
so the benchmark can compare wmas *configurations*:

* ``fused``     — fused Warp obs/reward kernels on the no-grad hot path
  (``fused="auto"`` follows the scenario; navigation ships them).
* ``use_graph`` — persistent buffers + CUDA-graph capture, which elides per-step
  kernel launch overhead.

The world/action setup mirrors ``compare_vmas._make_wmas`` (same ``world_size``,
seeded on-device action buffer) so numbers are directly comparable, except that
the cross-sim benchmark uses ``neighbor_obs=3`` (vs the scenario default of 2) so
the observation encodes the same number of nearest neighbours as CAMAR's obs at
higher agent counts — an apples-to-apples observation size across the sims.
"""

from __future__ import annotations

from wmas.benchmark._adapters import Runner, make_torch_runner


def build(
    scenario: str,
    n_envs: int,
    n_agents: int,
    device: str,
    seed: int = 0,
    use_graph: bool = False,
    fused: bool | str = "auto",
) -> Runner:
    if scenario != "navigation":
        raise ValueError(f"wmas adapter only implements 'navigation', got {scenario!r}")
    import torch

    from wmas import Environment, NavigationScenario

    sc = NavigationScenario(
        n_agents=n_agents, world_size=max(1.0, n_agents**0.5 / 4), neighbor_obs=3
    )
    env = Environment(
        sc, n_envs=n_envs, device=device, dt=0.05, substeps=1, seed=seed,
        use_graph=use_graph, fused=fused,
    )
    env.reset(seed=seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    actions = torch.empty(n_envs, n_agents, env.world.act_dim, device=device)

    def step() -> None:
        actions.uniform_(-1.0, 1.0, generator=gen)
        env.step(actions)

    return make_torch_runner(env, step, n_envs, env.n_agents, device)
