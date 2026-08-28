"""``action_bounds``: the action box the integrate kernels actually clamp against.

The claim under test is that :func:`swarp.dynamics.base.action_bounds` and
:attr:`swarp.core.environment.Environment.action_bounds` report the *same* limits the
kernel enforces — so the checks here are behavioural rather than a second copy of the
table: commanding the reported bound saturates the state, and commanding twice the
bound changes nothing.

The TorchRL half pins the default: ``SwarpEnv`` specs ``env.action_bounds``, so a policy
sampling inside its own action space can actually reach what the kernels allow — and an
explicit scalar pair is still the way to opt back out to a normalized box.
"""

import math

import pytest
import torch
from conftest import DEVICES

from swarp import Environment
from swarp.dynamics.base import (
    AgentConfig,
    ControlMode,
    DynamicsModel,
    action_bounds,
    action_dim,
)
from swarp.dynamics.drone import drone_config
from swarp.scenarios.navigation import NavigationScenario

MODELS = [
    (DynamicsModel.HOLONOMIC, ControlMode.VELOCITY),
    (DynamicsModel.HOLONOMIC, ControlMode.ACCELERATION),
    (DynamicsModel.DIFF_DRIVE, ControlMode.VELOCITY),
    (DynamicsModel.DIFF_DRIVE, ControlMode.ACCELERATION),
    (DynamicsModel.KINEMATIC_BICYCLE, ControlMode.VELOCITY),
]


def _fleet(model, mode, n=3, max_speed=1.7):
    """A navigation scenario whose fleet is ``model``/``mode`` with non-unit limits.

    ``max_speed`` is deliberately not 1.0: that is the single value at which the
    ``[-1, 1]`` convention coincides with the physical box, and a test that used it
    could not tell the two apart.
    """

    class Fleet(NavigationScenario):
        def _agent_configs(self):
            return [
                AgentConfig(
                    model=model,
                    ctrl_mode=mode,
                    radius=self.agent_radius,
                    max_speed=max_speed,
                    max_accel=2.0 * max_speed,
                )
                for _ in range(n)
            ]

    return Fleet(n_agents=n, n_obstacles=0)


def _env(scenario, device, dt=0.05):
    return Environment(
        scenario, n_envs=2, device=device, dt=dt, seed=0, use_graph=False, fused=False
    )


def _step_from_seed(env, action):
    """Reset to a fixed state, apply one action, return the five 2D state fields."""
    env.reset(seed=0)
    with torch.no_grad():
        env.step(action)
    return [t.clone() for t in env.world.state[:5]]


# ------------------------------------------------------------------- the table


@pytest.mark.parametrize("model,mode", MODELS)
def test_bounds_are_the_config_limits(model, mode):
    """Each slot's bound is the ``AgentConfig`` field the kernel's branch clamps to."""
    cfg = AgentConfig(model=model, ctrl_mode=mode, max_speed=1.7, max_accel=3.4)
    low, high = action_bounds(cfg)
    assert len(low) == len(high) == action_dim(model)
    if model == DynamicsModel.HOLONOMIC:
        lim = cfg.max_speed if mode == ControlMode.VELOCITY else cfg.max_accel
        expected = [lim, lim]
    elif model == DynamicsModel.DIFF_DRIVE:
        expected = (
            [cfg.max_speed, cfg.max_ang_vel]
            if mode == ControlMode.VELOCITY
            else [cfg.max_accel, cfg.max_ang_accel]
        )
    else:  # the bicycle ignores ctrl_mode: throttle plus a steering *angle*
        expected = [cfg.max_accel, cfg.max_steer]
    assert high == expected
    assert low == [-h for h in expected]


def test_drone_bounds_are_one_sided():
    """A rotor cannot pull, so the drone's box is ``[0, thrust_max]``, not symmetric."""
    cfg = drone_config(thrust_max=7.0)
    low, high = action_bounds(cfg)
    assert low == [0.0] * 4
    assert high == [7.0] * 4


def test_drone_bounds_admit_hover_while_unit_bounds_do_not():
    """Why a normalized box is the wrong default: ``[-1, 1]`` caps a quadrotor below its
    own weight, so the reported physical box is the only one that can hover."""
    cfg = drone_config()  # mass 1.0, gravity 9.81 -> ~2.45 N per rotor to hover
    _, high = action_bounds(cfg)
    weight = cfg.mass * cfg.gravity
    assert sum(high) > weight  # the physical box can hover
    assert weight > 4 * 1.0  # a unit box cannot


# --------------------------------------------------- the kernel's actual clamp


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model,mode", MODELS)
def test_commanding_past_the_bound_changes_nothing(model, mode, device):
    """Doubling an at-the-bound action is a no-op: the bound is where the kernel saturates."""
    env = _env(_fleet(model, mode), device)
    _, high = env.action_bounds
    at = high.unsqueeze(0).expand(env.n_envs, -1, -1).contiguous()
    beyond = _step_from_seed(env, 4.0 * at)
    for a, b in zip(_step_from_seed(env, at), beyond, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model,mode", MODELS)
def test_the_bound_is_not_loose(model, mode, device):
    """Half the bound is *not* saturated — so the reported box is tight, not merely safe."""
    env = _env(_fleet(model, mode), device)
    _, high = env.action_bounds
    at = high.unsqueeze(0).expand(env.n_envs, -1, -1).contiguous()
    inside = _step_from_seed(env, 0.5 * at)
    outside = _step_from_seed(env, at)
    assert any(not torch.allclose(a, b) for a, b in zip(inside, outside, strict=True))


@pytest.mark.parametrize("device", DEVICES)
def test_holonomic_velocity_saturates_at_max_speed(device):
    """The one exact number, for the default model: an axis-aligned command clamps to it."""
    env = _env(_fleet(DynamicsModel.HOLONOMIC, ControlMode.VELOCITY, max_speed=1.7), device)
    _, high = env.action_bounds
    act = torch.zeros(env.n_envs, env.n_agents, env.act_dim, device=device)
    act[..., 0] = 10.0 * high[0, 0]
    vel = _step_from_seed(env, act)[2]
    torch.testing.assert_close(vel[..., 0], torch.full_like(vel[..., 0], 1.7))


@pytest.mark.parametrize("device", DEVICES)
def test_bicycle_steering_slot_is_an_angle(device):
    """Slot 1 of the bicycle is ``max_steer`` (pi/4), not a speed — ~27% under ``1.0``."""
    env = _env(_fleet(DynamicsModel.KINEMATIC_BICYCLE, ControlMode.VELOCITY), device)
    _, high = env.action_bounds
    assert high[0, 1].item() == pytest.approx(math.pi / 4)
    assert high[0, 1].item() < 1.0  # what a [-1, 1] box would over-declare


@pytest.mark.parametrize("device", DEVICES)
def test_drone_climbs_at_its_bound_and_falls_at_unit_thrust(device):
    """End to end: the physical box reaches hover, the ``[-1, 1]`` box cannot."""

    class Drones(NavigationScenario):
        def _agent_configs(self):
            return [drone_config(radius=self.agent_radius) for _ in range(self.n_agents)]

    env = _env(Drones(n_agents=2, n_obstacles=0), device)
    _, high = env.action_bounds
    at = high.unsqueeze(0).expand(env.n_envs, -1, -1).contiguous()
    env.reset(seed=0)
    with torch.no_grad():
        for _ in range(5):
            env.step(at)
    assert (env.world.state.z > 0).all()
    env.reset(seed=0)
    with torch.no_grad():
        for _ in range(5):
            env.step(torch.ones_like(at))
    assert (env.world.state.z < 0).all()


# ------------------------------------------------------------- padded slot width


@pytest.mark.parametrize("device", DEVICES)
def test_padded_slots_report_zero_width(device):
    """A mixed fleet pads ``act_dim`` to 4; the 2D agents' unused slots read ``[0, 0]``."""

    class Mixed(NavigationScenario):
        def _agent_configs(self):
            return [
                AgentConfig(model=DynamicsModel.HOLONOMIC, radius=self.agent_radius),
                drone_config(radius=self.agent_radius),
            ]

    env = _env(Mixed(n_agents=2, n_obstacles=0), device)
    assert env.act_dim == 4
    low, high = env.action_bounds
    assert low.shape == high.shape == (2, 4)
    assert (low[0, 2:] == 0).all() and (high[0, 2:] == 0).all()  # holonomic ignores 2 and 3
    assert (high[1] > 0).all()  # the drone uses all four


# --------------------------------------------------------------- TorchRL specs


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model,mode", MODELS)
def test_swarpenv_defaults_to_the_physical_box(model, mode, device):
    """The promise: with no bounds argument the spec **is** ``env.action_bounds``.

    ``_fleet`` uses ``max_speed=1.7`` precisely so this can tell the physical box apart
    from the ``[-1, 1]`` convention that used to be the default.
    """
    pytest.importorskip("torchrl")
    from swarp.interop.torchrl import SwarpEnv

    env = _env(_fleet(model, mode), device)
    low, high = env.action_bounds
    space = SwarpEnv(env).action_spec.space
    torch.testing.assert_close(space.low, low.expand_as(space.low).contiguous())
    torch.testing.assert_close(space.high, high.expand_as(space.high).contiguous())
    assert (space.high != 1.0).any()  # not the old unit box


@pytest.mark.parametrize("device", DEVICES)
def test_swarpenv_explicit_scalars_still_override(device):
    """The opt-out: an explicit scalar pair is passed straight through, unit box included."""
    pytest.importorskip("torchrl")
    from swarp.interop.torchrl import SwarpEnv

    env = _env(_fleet(DynamicsModel.HOLONOMIC, ControlMode.VELOCITY), device)
    space = SwarpEnv(env, action_low=-1.0, action_high=1.0).action_spec.space
    assert (space.low == -1.0).all() and (space.high == 1.0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_swarpenv_one_sided_override_keeps_the_physical_other_side(device):
    """Overriding one bound must not silently revert the other to a guess."""
    pytest.importorskip("torchrl")
    from swarp.interop.torchrl import SwarpEnv

    env = _env(_fleet(DynamicsModel.HOLONOMIC, ControlMode.VELOCITY), device)
    _, high = env.action_bounds
    space = SwarpEnv(env, action_low=0.0).action_spec.space
    assert (space.low == 0.0).all()
    torch.testing.assert_close(space.high, high.expand_as(space.high).contiguous())


@pytest.mark.parametrize("device", DEVICES)
def test_swarpenv_default_drone_spec_admits_hover(device):
    """The failure this fix removes: under the old ``[-1, 1]`` default a quadrotor's spec
    capped total thrust at 4 N against a ~9.81 N weight, so no policy sampling inside the
    spec could hover. The default spec must now contain the hover action."""
    pytest.importorskip("torchrl")
    from swarp.interop.torchrl import SwarpEnv

    class Drones(NavigationScenario):
        def _agent_configs(self):
            return [drone_config(radius=self.agent_radius) for _ in range(self.n_agents)]

    env = _env(Drones(n_agents=2, n_obstacles=0), device)
    space = SwarpEnv(env).action_spec.space
    cfg = drone_config()
    hover_per_rotor = cfg.mass * cfg.gravity / 4.0
    assert (space.low <= 0.0).all()  # a rotor cannot pull
    assert (space.high >= hover_per_rotor).all()  # ...but it can hold the aircraft up
    assert space.high.sum(-1).min().item() > cfg.mass * cfg.gravity


@pytest.mark.parametrize("device", DEVICES)
def test_swarpenv_accepts_per_slot_bounds(device):
    """Per-slot tensors reach the spec, which a scalar pair could not express."""
    pytest.importorskip("torchrl")
    from torchrl.envs.utils import check_env_specs

    from swarp.interop.torchrl import SwarpEnv

    env = _env(_fleet(DynamicsModel.KINEMATIC_BICYCLE, ControlMode.VELOCITY), device)
    low, high = env.action_bounds
    tenv = SwarpEnv(env, action_low=low, action_high=high)
    space = tenv.action_spec.space
    expected = high.unsqueeze(0).expand(env.n_envs, -1, -1)
    torch.testing.assert_close(space.high, expected.contiguous())
    torch.testing.assert_close(space.low, -expected.contiguous())
    check_env_specs(tenv)
    assert (tenv.action_spec.rand().abs() <= expected + 1e-6).all()


@pytest.mark.parametrize("device", DEVICES)
def test_swarpenv_rejects_unbroadcastable_bounds(device):
    pytest.importorskip("torchrl")
    from swarp.interop.torchrl import SwarpEnv

    env = _env(_fleet(DynamicsModel.HOLONOMIC, ControlMode.VELOCITY), device)
    with pytest.raises(ValueError, match="not broadcastable"):
        SwarpEnv(env, action_high=torch.ones(7, device=device))
