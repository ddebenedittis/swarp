"""Ported VMAS-style scenarios: API smoke, determinism, and reward-shape checks.

Also covers the registry front door (``swarp.scenarios.make_scenario`` / ``swarp.make``).
"""

import inspect
import warnings

import pytest
import torch
from conftest import DEVICES

import swarp
from swarp import DynamicsModel, Environment, Integrator, WorldConfig
from swarp.scenarios import SCENARIOS as REGISTRY
from swarp.scenarios import (
    Scenario,
    fused_scenarios,
    make_scenario,
    register_scenario,
    resolve_scenarios,
    scenario_class,
)
from swarp.scenarios.discovery import DiscoveryScenario
from swarp.scenarios.flocking import FlockingScenario
from swarp.scenarios.formation import FormationScenario
from swarp.scenarios.sampling import SamplingScenario

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
    """A concrete scenario with no fused kernels (inherits fused_available -> False).

    The signatures mirror the ABC exactly — ``world_config`` on ``make_world``,
    ``obs_only`` on ``reset_world``, ``observations`` (the abstract member) rather than
    ``observation`` (the concrete slicing helper). ``test_plain_scenario_matches_the_abc``
    instantiates it, which is what keeps them from drifting again.
    """

    obs_dim = 4

    def make_world(self, n_envs, device, dt, substeps, dtype, world_config=None):
        raise NotImplementedError

    def reset_world(self, env_mask=None, *, obs_only=False):
        raise NotImplementedError

    def observations(self):
        raise NotImplementedError


class _OutOfTreeScenario(REGISTRY["navigation"]):
    """Concrete stand-in for a scenario defined outside the package."""


def test_register_scenario_reaches_the_whole_registry(monkeypatch):
    """An out-of-tree scenario must be reachable by name without editing the package."""
    monkeypatch.setitem(REGISTRY, "mine", _OutOfTreeScenario)  # snapshot/restore
    del REGISTRY["mine"]

    register_scenario("mine", _OutOfTreeScenario)
    assert scenario_class("mine") is _OutOfTreeScenario
    assert isinstance(make_scenario("mine", n_agents=2), _OutOfTreeScenario)
    assert "mine" in resolve_scenarios(["all"])
    assert isinstance(swarp.make("mine", n_envs=2, n_agents=2, device="cpu"), Environment)


def test_register_scenario_rejects_collisions_and_non_scenarios(monkeypatch):
    monkeypatch.setitem(REGISTRY, "mine", _OutOfTreeScenario)
    with pytest.raises(ValueError, match="already registered"):
        register_scenario("mine", _OutOfTreeScenario)
    register_scenario("mine", _OutOfTreeScenario, overwrite=True)  # explicit opt-in is fine
    # Shadowing a built-in needs the same opt-in.
    with pytest.raises(ValueError, match="already registered"):
        register_scenario("navigation", _OutOfTreeScenario)

    with pytest.raises(ValueError, match="non-empty"):
        register_scenario("", _OutOfTreeScenario)
    with pytest.raises(TypeError, match="Scenario subclass"):
        register_scenario("nope", object)
    with pytest.raises(TypeError, match="Scenario subclass"):
        register_scenario("nope", _OutOfTreeScenario(n_agents=2))  # instance, not class
    assert "nope" not in REGISTRY


def test_fused_scenarios_is_derived_not_hardcoded(monkeypatch):
    """A registered scenario without fused kernels is excluded, and one with them
    is included, purely from ``fused_available()`` — no hand-maintained list."""
    monkeypatch.setitem(REGISTRY, "plain", _PlainScenario)
    names = fused_scenarios()
    assert "plain" not in names
    assert "navigation" in names


def test_make_kwargs_split_is_unambiguous():
    """``swarp.make`` routes kwargs by name; Environment's and the scenarios'
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
    env = swarp.make("flocking", n_envs=4, n_agents=3, device=device, dt=0.1, seed=0)
    assert isinstance(env, Environment)
    assert env.n_envs == 4 and env.n_agents == 3
    assert env.device == device
    obs = env.reset()
    assert obs.shape[:2] == (4, 3) and torch.isfinite(obs).all()


def test_make_drops_model_for_holonomic_only_scenarios():
    """The drop still happens — but it warns, so it reads as a design choice rather than
    a request that vanished."""
    with pytest.warns(UserWarning, match="holonomic-only"):
        env = swarp.make(
            "flocking", n_envs=2, n_agents=2, device="cpu", model=DynamicsModel.DIFF_DRIVE
        )
    assert env.n_agents == 2  # constructed despite the unsupported kwarg


def test_make_does_not_warn_when_the_scenario_takes_a_model():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        swarp.make(
            "navigation", n_envs=2, n_agents=2, device="cpu", model=DynamicsModel.DIFF_DRIVE
        )


def test_benchmark_model_sweep_emits_no_drop_warnings():
    """The cross-scenario sweep filters on ``supports_model`` rather than leaning on the
    drop, so turning the drop into a warning must not make the benchmark noisy."""
    from swarp.benchmark.scenarios import _model_axis, build_scenario

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in REGISTRY:
            for model_name in _model_axis(name):
                build_scenario(name, 2, model_name)


@pytest.mark.parametrize("name", ["drone", "diff_drive", "holonomic"])
def test_model_accepts_a_string(name):
    """Every other string-ish option in the API takes a string; ``model`` now does too."""
    scen = make_scenario("navigation", n_agents=2, model=name)
    env = Environment(scen, n_envs=2, device="cpu", seed=0)
    try:
        assert env.world.agent_configs[0].model is DynamicsModel[name.upper()]
    finally:
        env.close()


def test_unknown_model_string_lists_the_valid_names():
    with pytest.raises(ValueError, match="holonomic"):
        Environment(
            make_scenario("navigation", n_agents=2, model="hovercraft"),
            n_envs=2,
            device="cpu",
            seed=0,
        )


# Engine settings no scenario exposes as a constructor argument. All six are set away
# from the WorldConfig() defaults, which is what makes them visible to the merge rule.
UNREACHABLE_OVERRIDE = WorldConfig(
    integrator=Integrator.RK4,
    bounds_mode="clamp",
    neighbor_reuse=False,
    grid_dim=64,
    obstacle_angular_damping=3.5,
    contact_max_overlap=0.25,
)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_scenario_honours_the_world_config_override(name):
    """All ten must end make_world with ``.override_with(world_config)``.

    Nothing in the type system enforces that, and a scenario that quietly dropped the
    argument would leave its users back where they started: subclass or nothing.
    """
    scen = make_scenario(name, **SCENARIO_KWARGS.get(name, {}))
    world = scen.make_world(
        n_envs=2,
        device="cpu",
        dt=0.1,
        substeps=1,
        dtype=torch.float32,
        world_config=UNREACHABLE_OVERRIDE,
    )
    cfg = world.stepper.world
    assert cfg.integrator is Integrator.RK4
    assert cfg.bounds_mode == "clamp"
    assert cfg.neighbor_reuse is False
    assert cfg.grid_dim == 64
    assert cfg.obstacle_angular_damping == 3.5
    assert cfg.contact_max_overlap == 0.25
    # ...and what the scenario computed for itself is still there.
    assert cfg.bounds is not None and cfg.collision_margin != WorldConfig().collision_margin


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_omitting_world_config_leaves_the_scenario_config_untouched(name):
    """The non-breaking half: no argument must mean byte-for-byte the old config."""
    kwargs = dict(n_envs=2, device="cpu", dt=0.1, substeps=1, dtype=torch.float32)
    plain = make_scenario(name, **SCENARIO_KWARGS.get(name, {})).make_world(**kwargs)
    explicit_none = make_scenario(name, **SCENARIO_KWARGS.get(name, {})).make_world(
        **kwargs, world_config=None
    )
    assert plain.stepper.world == explicit_none.stepper.world


def test_make_forwards_world_config():
    """``swarp.make`` routes it by name, so the front door reaches the engine too."""
    env = swarp.make(
        "navigation",
        n_envs=8,
        n_agents=3,
        device="cpu",
        world_config=WorldConfig(collision_k=50.0),
    )
    cfg = env.world.stepper.world
    assert cfg.collision_k == 50.0
    assert cfg.bounds is not None  # navigation's own bounds survived


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
        obs, rew, term, trunc, info = env.step(_random_actions(env, gen))
        assert obs.shape[:2] == (8, env.n_agents) and torch.isfinite(obs).all()
        assert rew.shape == (8, env.n_agents) and torch.isfinite(rew).all()
        for flag in (term, trunc):
            assert flag.shape == (8,) and flag.dtype == torch.bool


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
            obs, rew, *_ = env.step(_random_actions(env, gen))
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
        _, rew, *_ = env.step(_random_actions(env, gen))
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
    # fused=False: the assertion below cross-checks info() against agent_reward(), which
    # only the torch reference path populates (the fused path has no per-agent split).
    env = Environment(scenario, n_envs=ne, device=device, dt=0.1, seed=0, fused=False)
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


def test_plain_scenario_matches_the_abc():
    """``_PlainScenario`` must stay instantiable and signature-compatible with ``Scenario``.

    It drifted once — a stale ``make_world`` without ``world_config`` and an
    ``observation`` where the ABC declares ``observations`` — and nothing noticed, because
    the only test using it read ``fused_available`` off the class and never constructed it.
    """
    scen = _PlainScenario()
    assert scen.fused_available is False
    assert scen.graph_hook() is None
    for name in ("make_world", "reset_world", "observations"):
        theirs = inspect.signature(getattr(_PlainScenario, name))
        base = inspect.signature(getattr(Scenario, name))
        assert theirs.parameters.keys() == base.parameters.keys(), name


@pytest.mark.parametrize(
    ("bad", "hint"),
    [("max_step", "max_steps"), ("devise", "device")],
)
def test_make_suggests_the_environment_keyword_you_meant(bad, hint):
    """A misspelled Environment keyword is routed to the scenario, so ``make`` has to
    catch it itself — with the name it looks like, not a bare unexpected-keyword error."""
    with pytest.raises(TypeError, match=f"Did you mean Environment's '{hint}'"):
        swarp.make("navigation", n_envs=2, device="cpu", **{bad: 4})


def test_make_lists_the_scenario_keywords_for_an_unknown_one():
    with pytest.raises(TypeError, match="Scenario keywords: .*n_agents"):
        swarp.make("navigation", n_envs=2, device="cpu", bogus=1)
