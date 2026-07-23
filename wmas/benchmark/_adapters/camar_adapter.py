"""CAMAR adapter: continuous multi-agent navigation with collision avoidance.

CAMAR is the closest sibling to wmas — a JAX continuous-action navigation /
collision-avoidance sim. The ``random_grid`` map with ``HolonomicDynamic`` maps
directly to wmas ``NavigationScenario`` (holonomic point agents driving to goals
amid obstacles). ``num_agents`` is set via ``map_kwargs``; actions are a single
``(n_agents, action_size)`` array per env (holonomic: a 2D force vector).
"""

from __future__ import annotations

from wmas.benchmark._adapters import Runner, assert_jax_gpu, make_jax_runner

_SUPPORTED = {"navigation"}


def build(scenario: str, n_envs: int, n_agents: int, device: str, seed: int = 0) -> Runner:
    if scenario not in _SUPPORTED:
        raise ValueError(
            f"camar adapter has no analogue for scenario {scenario!r}; "
            f"available: {sorted(_SUPPORTED)}"
        )
    assert_jax_gpu()

    from camar import camar_v0
    from jax import random

    env = camar_v0(
        map_generator="random_grid",
        dynamic="HolonomicDynamic",
        map_kwargs={"num_agents": n_agents},
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
