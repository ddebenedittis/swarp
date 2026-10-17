"""JaxMARL adapter: MPE cooperative navigation (``simple_spread``), continuous.

``MPE_simple_spread_v3`` is JaxMARL's canonical cooperative-navigation task —
agents spread to cover landmarks while avoiding collisions, the JAX analogue of
swarp ``NavigationScenario`` and VMAS ``navigation``. It is instantiated in
**continuous** action mode (``action_type="Continuous"``, a Box) to match swarp's
continuous control; ``num_agents`` (and matching ``num_landmarks``) are honored,
so the requested agent count is realized exactly.
"""

from __future__ import annotations

from swarp.benchmark._adapters import Runner, assert_jax_gpu, make_jax_runner

_SCENARIO_ENV = {"navigation": "MPE_simple_spread_v3"}


def build(scenario: str, n_envs: int, n_agents: int, device: str, seed: int = 0) -> Runner:
    if scenario == "ringroad":
        return _ringroad(n_envs, n_agents, seed)
    if scenario not in _SCENARIO_ENV:
        raise ValueError(
            f"jaxmarl adapter has no analogue for scenario {scenario!r}; "
            f"available: {sorted(_SCENARIO_ENV)}"
        )
    assert_jax_gpu()

    from jax import random
    from jaxmarl import make

    env = make(
        _SCENARIO_ENV[scenario],
        num_agents=n_agents,
        num_landmarks=n_agents,
        action_type="Continuous",
    )
    agents = tuple(env.agents)
    act_dim = env.action_space(agents[0]).shape[0]

    def sample_actions(key, ne):
        # Continuous MPE actions are a Box in [0, 1]; random actions suffice for a
        # throughput measurement (values don't affect per-step cost).
        a = random.uniform(key, (len(agents), ne, act_dim))
        return {ag: a[i] for i, ag in enumerate(agents)}

    return make_jax_runner(
        reset_one=env.reset,
        step_one=env.step,
        sample_actions=sample_actions,
        n_envs=n_envs,
        n_agents=env.num_agents,
        seed=seed,
    )


def _ringroad(n_envs: int, n_agents: int, seed: int) -> Runner:
    """The JAX port of the ring road (:mod:`.ringroad_jax`), random actions over its box."""
    assert_jax_gpu()

    from jax import random

    from swarp.benchmark._adapters.ringroad_jax import RingRoadJax

    env = RingRoadJax(num_agents=n_agents)
    agents = tuple(env.agents)
    hi = env.action_space(agents[0]).high

    def sample_actions(key, ne):
        a = random.uniform(key, (len(agents), ne, 2), minval=-1.0, maxval=1.0) * hi
        return {ag: a[i] for i, ag in enumerate(agents)}

    return make_jax_runner(
        reset_one=env.reset,
        step_one=env.step,
        sample_actions=sample_actions,
        n_envs=n_envs,
        n_agents=n_agents,
        seed=seed,
    )
