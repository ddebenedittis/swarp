"""Lidar ranges vs hand-computed ray-circle intersections, plus a gradcheck."""

import math

import numpy as np
import pytest
import torch
from conftest import DEVICES

from swarp.sensors.lidar import Lidar, lidar_scan
from swarp.sensors.lidar_kernels import lidar_scan_warp

MAXR = 2.0

# Warp vs torch differ only at transcendental ULP level (own sin/cos/sqrt),
# amplified on near-tangent rays where t = proj - sqrt(rad^2 - perp2) is stiff.
# float64 stays ~1e-10; float32 lands at float32 epsilon.
_TOL = {torch.float64: dict(atol=1e-9, rtol=1e-6), torch.float32: dict(atol=1e-4, rtol=1e-4)}


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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("body_frame", [True, False])
@pytest.mark.parametrize("include_agents", [True, False])
@pytest.mark.parametrize("with_obstacles", [True, False])
def test_warp_matches_torch(device, dtype, body_frame, include_agents, with_obstacles):
    """The Warp backend produces the same ranges as the torch broadcast (exactness
    oracle). Includes the motivating 256-ray case."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    e, a, t, n_rays = 3, 5, 4, 256
    pos = (torch.rand(e, a, 2, generator=gen) * 2 - 1).to(device=device, dtype=dtype)
    theta = (torch.rand(e, a, generator=gen) * 6.28).to(device=device, dtype=dtype)
    rad = (torch.rand(a, generator=gen) * 0.2 + 0.05).to(device=device, dtype=dtype)
    if with_obstacles:
        opos = (torch.rand(e, t, 2, generator=gen) * 2 - 1).to(device=device, dtype=dtype)
        orad = (torch.rand(t, generator=gen) * 0.2 + 0.05).to(device=device, dtype=dtype)
    else:
        opos = orad = None

    kw = dict(
        n_rays=n_rays,
        max_range=2.0,
        body_frame=body_frame,
        angle_start=0.3,
        include_agents=include_agents,
        obstacle_pos=opos,
        obstacle_radius=orad,
    )
    ref = lidar_scan(pos, theta, rad, **kw)
    got = lidar_scan_warp(pos, theta, rad, **kw)
    assert got.shape == (e, a, n_rays)
    torch.testing.assert_close(got, ref, **_TOL[dtype])


def test_lidar_backend_validation():
    with pytest.raises(ValueError, match="backend"):
        Lidar(backend="bogus")


def _tiny_world(dtype=torch.float32):
    from swarp.core.config import WorldConfig
    from swarp.core.world import World
    from swarp.dynamics.base import AgentConfig, DynamicsModel

    cfgs = [AgentConfig(model=DynamicsModel.HOLONOMIC, radius=0.05) for _ in range(3)]
    return World(cfgs, WorldConfig(collisions=True), n_envs=4, device="cpu", dtype=dtype)


def test_warp_grad_fallback():
    """backend='warp' uses the warp path for inference but falls back to the
    differentiable torch path whenever a gradient is being tracked."""
    world = _tiny_world(dtype=torch.float64)
    lidar = Lidar(n_rays=6, max_range=1.0, backend="warp")

    # No grad: warp path -> detached output (no grad_fn).
    world.state = world.state._replace(
        pos=torch.rand(4, 3, 2, dtype=torch.float64) * 0.4 - 0.2,
        theta=torch.zeros(4, 3, dtype=torch.float64),
    )
    assert lidar.scan(world).grad_fn is None

    # Grad tracked: torch fallback -> differentiable output.
    with torch.enable_grad():
        p = (torch.rand(4, 3, 2, dtype=torch.float64) * 0.4 - 0.2).requires_grad_(True)
        world.state = world.state._replace(pos=p, theta=torch.zeros(4, 3, dtype=torch.float64))
        out = lidar.scan(world)
        assert out.grad_fn is not None
        out.sum().backward()
        assert p.grad is not None


@pytest.mark.parametrize("backend", ["torch", "warp"])
def test_lidar_component_on_world(backend):
    """The Lidar component scans a real World and concatenates into an observation."""
    world = _tiny_world()
    world.state = world.state._replace(pos=torch.rand(4, 3, 2) * 0.4 - 0.2, theta=torch.zeros(4, 3))
    lidar = Lidar(n_rays=6, max_range=1.0, backend=backend)
    ranges = lidar.scan(world)
    assert ranges.shape == (4, 3, 6)
    assert torch.isfinite(ranges).all()
    assert (ranges >= 0).all() and (ranges <= 1.0).all()
