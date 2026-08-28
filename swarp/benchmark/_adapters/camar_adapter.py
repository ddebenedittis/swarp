"""CAMAR adapter: continuous multi-agent navigation with collision avoidance.

CAMAR is the closest sibling to swarp — a JAX continuous-action navigation /
collision-avoidance sim. Two configurations:

* ``obstacles=False`` (default) — an **open arena** (``string_grid`` of all-free
  cells, no border), matching swarp ``NavigationScenario``'s obstacle-free space.
  Observations reduce to goal + neighbor features (no obstacle raycasting).
  CAMAR integrates ``frameskip + 1`` world steps per ``env.step``, so
  ``frameskip=0`` runs exactly one world step, matching swarp ``substeps=1`` —
  an apples-to-apples navigation task (``frameskip=1`` would double CAMAR's
  per-step physics work).
* ``obstacles=True`` (``camar-grid``) — CAMAR's native ``random_grid`` (~800
  obstacles, ``frameskip=2``): a cluttered-maze task, far heavier per step.

``HolonomicDynamic`` (a 2D force vector) maps to swarp's holonomic point agents.
"""

from __future__ import annotations

from swarp.benchmark._adapters import Runner, assert_jax_gpu, make_jax_runner

_SUPPORTED = {"navigation"}
# All-free 12x12 arena (no border) -> zero obstacles, a ~1.2x1.2 world. Note that this is
# *not* the same arena size as swarp's: the swarp adapter uses
# ``world_size = max(1, sqrt(n_agents)/4)`` with bounds at +-world_size, i.e. 2x2 from 3 to
# 16 agents and 4x4 at 64. CAMAR's map generator is what fixes its extent, so matching it
# exactly is not available here; the density difference is disclosed in docs/benchmarks.md.
_OPEN_GRID = "\n".join(["." * 12 for _ in range(12)])


def build(
    scenario: str, n_envs: int, n_agents: int, device: str, seed: int = 0, obstacles: bool = False
) -> Runner:
    if scenario not in _SUPPORTED:
        raise ValueError(
            f"camar adapter has no analogue for scenario {scenario!r}; "
            f"available: {sorted(_SUPPORTED)}"
        )
    assert_jax_gpu()

    from camar import camar_v0
    from jax import random

    if obstacles:
        env = camar_v0(
            map_generator="random_grid",
            dynamic="HolonomicDynamic",
            map_kwargs={"num_agents": n_agents},
        )
    else:
        env = camar_v0(
            map_generator="string_grid",
            dynamic="HolonomicDynamic",
            frameskip=0,  # frameskip+1 world steps/step -> 0 = one step, == swarp substeps=1
            map_kwargs={"map_str": _OPEN_GRID, "num_agents": n_agents, "add_border": False},
        )
    na = env.num_agents
    act_dim = env.action_size

    def sample_actions(key, ne):
        return random.uniform(key, (ne, na, act_dim), minval=-1.0, maxval=1.0)

    return make_jax_runner(
        reset_one=env.reset,
        step_one=env.step,
        sample_actions=sample_actions,
        n_envs=n_envs,
        n_agents=na,
        seed=seed,
    )
