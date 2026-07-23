"""Lidar ranges vs hand-computed ray-circle intersections, plus a gradcheck."""

import math

import numpy as np
import pytest
import torch

from wmas.sensors.lidar import Lidar, lidar_scan

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
MAXR = 2.0


def _scan(pos, theta, agent_radius, **kw):
    kw.setdefault("max_range", MAXR)
    return lidar_scan(
        torch.tensor(pos, dtype=torch.float64),
        torch.tensor(theta, dtype=torch.float64),
        torch.tensor(agent_radius, dtype=torch.float64),
        **kw,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_single_obstacle_head_on(device):
    """One ray straight at a circle: range = distance - radius; back ray misses."""
    pos = torch.zeros(1, 1, 2, dtype=torch.float64, device=device)
    theta = torch.zeros(1, 1, dtype=torch.float64, device=device)
    radius = torch.tensor([0.05], dtype=torch.float64, device=device)
    obs_pos = torch.tensor([[[0.5, 0.0]]], dtype=torch.float64, device=device)
    obs_rad = torch.tensor([0.1], dtype=torch.float64, device=device)
    r = lidar_scan(
        pos,
        theta,
        radius,
        n_rays=4,
        max_range=MAXR,
        body_frame=False,
        obstacle_pos=obs_pos,
        obstacle_radius=obs_rad,
    )  # rays at 0, 90, 180, 270 deg
    assert r.shape == (1, 1, 4)
    np.testing.assert_allclose(r[0, 0, 0].cpu(), 0.4, atol=1e-12)  # +x hits
    np.testing.assert_allclose(r[0, 0, 1].cpu(), MAXR, atol=1e-12)  # +y miss
    np.testing.assert_allclose(r[0, 0, 2].cpu(), MAXR, atol=1e-12)  # -x miss
    np.testing.assert_allclose(r[0, 0, 3].cpu(), MAXR, atol=1e-12)  # -y miss


def test_agent_sees_agent():
    """Two agents on the x-axis: the +x ray range is gap minus the target radius."""
    pos = [[[0.0, 0.0], [0.3, 0.0]]]
    r = _scan(pos, [[0.0, 0.0]], [0.05, 0.05], n_rays=4, body_frame=False)
    np.testing.assert_allclose(r[0, 0, 0], 0.3 - 0.05, atol=1e-12)  # agent0 -> agent1
    np.testing.assert_allclose(r[0, 1, 2], 0.3 - 0.05, atol=1e-12)  # agent1 -x -> agent0


def test_body_frame_rotates_rays():
    """With body_frame, ray 0 points along the agent heading."""
    # agent heading +90deg; obstacle directly north -> ray 0 should hit it.
    pos = torch.zeros(1, 1, 2, dtype=torch.float64)
    theta = torch.full((1, 1), math.pi / 2, dtype=torch.float64)
    radius = torch.tensor([0.05], dtype=torch.float64)
    obs_pos = torch.tensor([[[0.0, 0.5]]], dtype=torch.float64)
    obs_rad = torch.tensor([0.1], dtype=torch.float64)
    r = lidar_scan(
        pos,
        theta,
        radius,
        n_rays=4,
        max_range=MAXR,
        body_frame=True,
        obstacle_pos=obs_pos,
        obstacle_radius=obs_rad,
    )
    np.testing.assert_allclose(r[0, 0, 0], 0.4, atol=1e-12)  # heading ray hits north circle


def test_oblique_ray_circle():
    """A ray at 45deg toward an off-axis circle matches the analytic intersection."""
    c = np.array([0.7, 0.7])
    rad = 0.15
    ang = math.pi / 4  # ray along (1,1)/sqrt2
    d = np.array([math.cos(ang), math.sin(ang)])
    proj = c @ d
    perp2 = c @ c - proj**2
    expected = proj - math.sqrt(rad**2 - perp2)
    r = _scan(
        [[[0.0, 0.0]]],
        [[0.0]],
        [0.05],
        n_rays=1,
        body_frame=False,
        angle_start=ang,
        include_agents=False,
        obstacle_pos=torch.tensor([[[c[0], c[1]]]], dtype=torch.float64),
        obstacle_radius=torch.tensor([rad], dtype=torch.float64),
    )
    np.testing.assert_allclose(r[0, 0, 0], expected, atol=1e-12)


def test_lidar_gradcheck():
    """Ranges are differentiable w.r.t. agent and obstacle positions (head-on hit)."""
    pos = torch.tensor([[[0.0, 0.0], [0.4, 0.1]]], dtype=torch.float64, requires_grad=True)
    theta = torch.zeros(1, 2, dtype=torch.float64)
    radius = torch.tensor([0.05, 0.08], dtype=torch.float64)
    obs_pos = torch.tensor([[[0.6, 0.0]]], dtype=torch.float64, requires_grad=True)
    obs_rad = torch.tensor([0.12], dtype=torch.float64)

    def fn(p, op):
        return lidar_scan(
            p,
            theta,
            radius,
            n_rays=8,
            max_range=MAXR,
            body_frame=False,
            obstacle_pos=op,
            obstacle_radius=obs_rad,
        )

    assert torch.autograd.gradcheck(fn, (pos, obs_pos), eps=1e-6, atol=1e-5)


def test_lidar_component_on_world():
    """The Lidar component scans a real World and concatenates into an observation."""
    from wmas.core.config import WorldConfig
    from wmas.core.world import World
    from wmas.dynamics.base import AgentConfig, DynamicsModel

    cfgs = [AgentConfig(model=DynamicsModel.HOLONOMIC, radius=0.05) for _ in range(3)]
    world = World(cfgs, WorldConfig(collisions=True), n_envs=4, device="cpu", dtype=torch.float32)
    world.state = world.state._replace(pos=torch.rand(4, 3, 2) * 0.4 - 0.2, theta=torch.zeros(4, 3))
    lidar = Lidar(n_rays=6, max_range=1.0)
    ranges = lidar.scan(world)
    assert ranges.shape == (4, 3, 6)
    assert torch.isfinite(ranges).all()
    assert (ranges >= 0).all() and (ranges <= 1.0).all()
