"""Ported VMAS-style scenarios: API smoke, determinism, and reward-shape checks.

Also covers the registry front door (``wmas.scenarios.make_scenario`` / ``wmas.make``).
"""

import inspect

import pytest
import torch

import wmas
from wmas import DynamicsModel, Environment
from wmas.scenarios import SCENARIOS as REGISTRY
from wmas.scenarios import Scenario, fused_scenarios, make_scenario
from wmas.scenarios.discovery import DiscoveryScenario
from wmas.scenarios.flocking import FlockingScenario
from wmas.scenarios.formation import FormationScenario
from wmas.scenarios.sampling import SamplingScenario

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])

# Registry names -> the small-but-nontrivial construction kwargs these checks use.
# The class comes from the shared registry; only the kwargs (which are
# scenario-specific and not expressible by the registry) live here.
SCENARIO_KWARGS = {
    "sampling": {"n_agents": 4, "n_gaussians": 3, "grid_res": 10},
    "discovery": {"n_agents": 5, "n_targets": 4},
    "flocking": {"n_agents": 8},
    "formation": {"n_agents": 5},
}


def _build(name):
    return make_scenario(name, **SCENARIO_KWARGS[name])


def test_covered_names_are_registered():
    """A registry rename must break this file loudly, not skip silently."""
    assert set(SCENARIO_KWARGS) <= set(REGISTRY)


def test_make_scenario_resolves_from_the_registry():
    scen = make_scenario("sampling", n_agents=4, grid_res=8)
    assert isinstance(scen, REGISTRY["sampling"])
    assert scen.n_agents == 4


def test_make_scenario_unknown_name_lists_valid():
    with pytest.raises(ValueError, match="unknown scenario"):
        make_scenario("bogus")


class _PlainScenario(Scenario):
    """A concrete scenario with no fused kernels (inherits fused_available -> False)."""

    def make_world(self, n_envs, device, dt, substeps, dtype):
        raise NotImplementedError

    def reset_world(self, env_mask=None):
        raise NotImplementedError

    def observation(self, agent_idx):
        raise NotImplementedError


def test_fused_scenarios_is_derived_not_hardcoded(monkeypatch):
    """A registered scenario without fused kernels is excluded, and one with them
    is included, purely from ``fused_available()`` — no hand-maintained list."""
    monkeypatch.setitem(REGISTRY, "plain", _PlainScenario)
    names = fused_scenarios()
    assert "plain" not in names
    assert "navigation" in names


def test_make_kwargs_split_is_unambiguous():
    """``wmas.make`` routes kwargs by name; Environment's and the scenarios'
    parameter names must stay disjoint for that to be well defined."""
    env_params = set(inspect.signature(Environment.__init__).parameters) - {
        "self",
        "scenario",
        "n_envs",
    }
    for name, cls in REGISTRY.items():
        overlap = env_params & set(inspect.signature(cls.__init__).parameters)
        assert not overlap, f"{name} shares kwargs with Environment: {sorted(overlap)}"


@pytest.mark.parametrize("device", DEVICES)
def test_make_builds_a_working_environment(device):
    env = wmas.make("flocking", n_envs=4, n_agents=3, device=device, dt=0.1, seed=0)
    assert isinstance(env, Environment)
    assert env.n_envs == 4 and env.n_agents == 3
    assert env.device == device
    obs = env.reset()
    assert obs.shape[:2] == (4, 3) and torch.isfinite(obs).all()


def test_make_drops_model_for_holonomic_only_scenarios():
    env = wmas.make("flocking", n_envs=2, n_agents=2, device="cpu", model=DynamicsModel.DIFF_DRIVE)
    assert env.n_agents == 2  # constructed despite the unsupported kwarg


def _random_actions(env, gen):
    return (
        torch.rand(env.n_envs, env.n_agents, env.world.act_dim, generator=gen, device=env.device)
        * 2.0
        - 1.0
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", list(SCENARIO_KWARGS))
def test_scenario_api_and_finiteness(device, name):
    env = Environment(_build(name), n_envs=8, device=device, dt=0.1, seed=3)
    obs = env.reset()
    assert obs.shape[:2] == (8, env.n_agents) and torch.isfinite(obs).all()
    gen = torch.Generator(device=device).manual_seed(0)
    for _ in range(10):
        obs, rew, done, info = env.step(_random_actions(env, gen))
        assert obs.shape[:2] == (8, env.n_agents) and torch.isfinite(obs).all()
        assert rew.shape == (8, env.n_agents) and torch.isfinite(rew).all()
        assert done.shape == (8,) and done.dtype == torch.bool


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", list(SCENARIO_KWARGS))
def test_scenario_determinism(device, name):
    """A seeded random-policy rollout is bit-for-bit reproducible."""

    def run():
        env = Environment(_build(name), n_envs=6, device=device, dt=0.1, seed=11)
        obs = env.reset(seed=11)
        gen = torch.Generator(device=device).manual_seed(5)
        traj = [obs]
        for _ in range(8):
            obs, rew, _, _ = env.step(_random_actions(env, gen))
            traj.append(obs)
            traj.append(rew)
        return traj

    a, b = run(), run()
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


@pytest.mark.parametrize("device", DEVICES)
def test_sampling_rewards_nonneg_and_consume(device):
    """Field rewards are non-negative and cells get consumed over time."""
    scenario = SamplingScenario(n_agents=4, grid_res=8)
    env = Environment(scenario, n_envs=8, device=device, dt=0.1, seed=1)
    env.reset()
    gen = torch.Generator(device=device).manual_seed(2)
    frac0 = scenario.consumed.float().mean().item()
    total = 0.0
    for _ in range(15):
        _, rew, _, _ = env.step(_random_actions(env, gen))
        assert (rew >= -1e-6).all()  # sampling reward is field value >= 0
        total += rew.sum().item()
    frac1 = scenario.consumed.float().mean().item()
    assert frac1 > frac0  # visiting cells consumes them
    assert total > 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_discovery_covering_reward(device):
    """Enough agents parked on a target triggers a one-off shared covering reward."""
    scenario = DiscoveryScenario(
        n_agents=4, n_targets=2, agents_per_target=2, covering_range=0.3, covering_reward=1.0
    )
    env = Environment(scenario, n_envs=1, device=device, dt=0.1, seed=0)
    env.reset()
    scenario.covered.zero_()  # isolate from any incidental coverage at spawn
    # Teleport two agents onto target 0, the rest far away.
    tgt0 = scenario.targets[0, 0]
    scenario.world.state.pos.data[0, 0] = tgt0
    scenario.world.state.pos.data[0, 1] = tgt0 + 0.01
    scenario.world.state.pos.data[0, 2:] = tgt0 + 5.0
    scenario._refresh()
    assert scenario._cache["newly"][0, 0].item()  # target 0 newly covered
    gr = scenario.global_reward()
    assert gr[0].item() >= 1.0 - 1e-6
    # covering is one-off: a second refresh with the same coverage yields no new reward
    scenario._refresh()
    assert not scenario._cache["newly"][0, 0].item()


@pytest.mark.parametrize("device", DEVICES)
def test_formation_shaping_rewards_progress(device):
    """Moving an agent toward its slot yields positive position-shaping reward."""
    scenario = FormationScenario(n_agents=4, formation_radius=0.4)
    env = Environment(scenario, n_envs=1, device=device, dt=0.1, seed=0)
    env.reset()
    goal0 = scenario.world.goals[0, 0]
    p0 = scenario.world.state.pos.data[0, 0]
    # step agent 0 a little toward its slot; shaping should be positive
    scenario.world.state.pos.data[0, 0] = p0 + 0.1 * (goal0 - p0)
    scenario._refresh()
    assert scenario.agent_reward(0)[0].item() > 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_formation_info_multiobj_reward(device):
    """FormationScenario.info() emits a stacked objective vector + a scalar diagnostic,
    both on-device, with leading dim n_envs."""
    ne, na = 8, 4
    scenario = FormationScenario(n_agents=na)
    env = Environment(scenario, n_envs=ne, device=device, dt=0.1, seed=0)
    env.reset()  # populates the cache via reset_world -> _refresh
    info = scenario.info()
    assert set(info) >= {"multiobj_reward", "formation_error"}
    mo = info["multiobj_reward"]
    assert mo.shape == (ne, na, 2)  # [shaping, collision] per agent
    assert mo.device.type == torch.device(device).type
    fe = info["formation_error"]
    assert fe.shape == (ne,)
    assert fe.device.type == torch.device(device).type
    # The objective vector must sum (over objectives) to the scalar per-agent reward.
    per_agent = torch.stack([scenario.agent_reward(i) for i in range(na)], dim=1)
    assert torch.allclose(mo.sum(-1), per_agent, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_flocking_separation_penalty(device):
    """Two agents on top of each other incur a crowding (separation) penalty."""
    scenario = FlockingScenario(n_agents=4, separation_dist=0.2)
    env = Environment(scenario, n_envs=1, device=device, dt=0.1, seed=0)
    env.reset()
    scenario.world.state.pos.data[0, 0] = torch.zeros(2, device=device)
    scenario.world.state.pos.data[0, 1] = torch.tensor([0.02, 0.0], device=device)
    scenario._refresh()
    assert scenario._cache["crowd"][0, 0].item() > 0.0
    assert scenario.agent_reward(0)[0].item() < 0.0
