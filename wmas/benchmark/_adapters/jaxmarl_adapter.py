"""JaxMARL adapter: MPE cooperative navigation (``simple_spread``), continuous.

``MPE_simple_spread_v3`` is JaxMARL's canonical cooperative-navigation task —
agents spread to cover landmarks while avoiding collisions, the JAX analogue of
wmas ``NavigationScenario`` and VMAS ``navigation``. It is instantiated in
**continuous** action mode (``action_type="Continuous"``, a Box) to match wmas's
continuous control; ``num_agents`` (and matching ``num_landmarks``) are honored,
so the requested agent count is realized exactly.
"""

from __future__ import annotations

from wmas.benchmark._adapters import Runner, assert_jax_gpu, make_jax_runner

_SCENARIO_ENV = {"navigation": "MPE_simple_spread_v3"}


def build(scenario: str, n_envs: int, n_agents: int, device: str, seed: int = 0) -> Runner:
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
