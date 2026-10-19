"""Agent footprint plumbing: AgentShape, derived box radius, the P_SHAPE/P_HALF_* columns,
and the World mirrors. The contact physics of box agents is tested in test_box_agents.py."""

import math

import numpy as np
import pytest
import torch

from swarp.core.config import WorldConfig
from swarp.core.world import World
from swarp.dynamics import base
from swarp.dynamics.base import AgentConfig, AgentShape, DynamicsModel


def test_disc_is_the_default_and_keeps_its_radius():
    cfg = AgentConfig()
    assert cfg.shape == AgentShape.DISC
    assert cfg.radius == 0.05 and cfg.half_extents == (0.0, 0.0)
    assert AgentConfig(radius=0.2).radius == 0.2


def test_box_radius_is_the_circumscribed_radius():
    cfg = AgentConfig(shape="box", half_extents=(0.3, 0.4))
    assert cfg.shape is AgentShape.BOX
    assert cfg.radius == pytest.approx(0.5)
    # An explicit radius equal to the derived one is accepted.
    assert AgentConfig(shape=AgentShape.BOX, half_extents=(0.3, 0.4), radius=0.5).radius == 0.5
    row = cfg.to_row()
    assert row[base.P_SHAPE] == 1.0 and row[base.P_RADIUS] == pytest.approx(0.5)
    assert (row[base.P_HALF_X], row[base.P_HALF_Y]) == (0.3, 0.4)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(shape="box", half_extents=(0.3, 0.4), radius=0.1), "derived"),
        (dict(shape="box", half_extents=(0.3, 0.0)), "positive half_extents"),
        (dict(shape="box"), "positive half_extents"),
        (dict(shape="box", half_extents=(0.1, 0.1), model=DynamicsModel.DRONE), "drone"),
        (dict(half_extents=(0.1, 0.1)), "only meaningful"),
        (dict(shape="triangle"), "unknown shape"),
    ],
)
def test_invalid_footprints_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AgentConfig(**kwargs)


def test_existing_column_indices_are_unchanged():
    """The footprint columns are appended: every column a kernel already reads keeps its
    index, so the stride grows and nothing else moves."""
    before = [
        "P_RADIUS", "P_MASS", "P_MAX_SPEED", "P_MAX_ACCEL", "P_MAX_ANG_VEL",
        "P_MAX_ANG_ACCEL", "P_LF", "P_LR", "P_MAX_STEER", "P_THRUST_MAX", "P_ARM", "P_IXX",
        "P_IYY", "P_IZZ", "P_KAPPA", "P_GRAVITY", "P_MIN_SPEED",
    ]  # fmt: skip
    assert [getattr(base, c) for c in before] == list(range(17))
    assert (base.P_SHAPE, base.P_HALF_X, base.P_HALF_Y) == (17, 18, 19)
    assert (base.P_CMD_GAIN_V, base.P_CMD_GAIN_W) == (20, 21)
    assert base.NUM_PARAMS == 22


def _world(configs, device="cpu", collisions=True):
    return World(configs, WorldConfig(collisions=collisions), n_envs=2, device=device)


def test_world_mirrors_the_footprint():
    box = AgentConfig(shape="box", half_extents=(0.1, 0.05))
    world = _world([AgentConfig(), box])
    assert world.agent_shape.tolist() == [0, 1]
    assert world.agent_shape.dtype == torch.int32
    torch.testing.assert_close(world.agent_half_extents, torch.tensor([[0.0, 0.0], [0.1, 0.05]]))
    torch.testing.assert_close(world.agent_radius, torch.tensor([0.05, math.hypot(0.1, 0.05)]))
    assert world.stepper.has_box_agents
    assert world.stepper.max_agent_radius == pytest.approx(math.hypot(0.1, 0.05))
    assert not _world([AgentConfig(), AgentConfig()]).stepper.has_box_agents


def test_a_box_agent_world_builds_scans_and_steps():
    from swarp.sensors.lidar import Lidar

    world = _world([AgentConfig(), AgentConfig(shape="box", half_extents=(0.1, 0.05))])
    world.write_state(pos=torch.tensor([[0.0, 0.0], [0.5, 0.0]]))
    world.reset_state()
    world.write_state(pos=torch.tensor([[0.0, 0.0], [0.5, 0.0]]))
    for backend in ("torch", "warp"):
        r = Lidar(n_rays=4, max_range=2.0, body_frame=False, backend=backend).scan(world)
        torch.testing.assert_close(r[:, 0, 0], torch.full((2,), 0.4))
    actions = torch.zeros(2, 2, world.act_dim)
    with torch.no_grad():
        world.step(actions)
    world.step(actions.requires_grad_(True))
    world.state.pos.sum().backward()
    assert torch.isfinite(actions.grad).all()


def test_per_env_params_cannot_change_the_shape():
    box = AgentConfig(shape="box", half_extents=(0.1, 0.05))
    world = _world([AgentConfig(), box])
    floats = base.per_env_float_template(world.agent_configs, world.n_envs)
    world.stepper.set_agent_params_per_env(floats)  # unchanged shapes: accepted
    bad = floats.copy()
    bad[1, 0, base.P_SHAPE] = 1.0
    with pytest.raises(ValueError, match="P_SHAPE"):
        world.stepper.set_agent_params_per_env(bad)
    # A torch first install is checked too (a same-shaped torch refresh after an install
    # is the trusted in-place path, which reads nothing back to the host).
    fresh = _world([AgentConfig(), box])
    with pytest.raises(ValueError, match="P_SHAPE"):
        fresh.stepper.set_agent_params_per_env(torch.as_tensor(np.ascontiguousarray(bad)).float())
