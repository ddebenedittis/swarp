"""Lidar ranges and hits vs hand-computed and independent intersections, backend parity,
gradchecks, host-sync freedom and a captured launch."""

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


@pytest.mark.parametrize("backend", ["torch", "warp"])
def test_movable_obstacle_scan_tracks_live_pose(backend):
    """The scan must follow a movable obstacle, not the pose the scenario installed.

    A movable body is integrated in place inside the stepper's own arrays, so
    ``world.obstacle_pos`` stays at the spawn pose forever. Reading it made the lidar
    diverge from the truth by more than a world unit over a pusht rollout.
    """
    from swarp.core.config import ObstacleKind, Obstacles, WorldConfig
    from swarp.core.world import World
    from swarp.dynamics.base import AgentConfig, DynamicsModel

    cfgs = [AgentConfig(model=DynamicsModel.HOLONOMIC, radius=0.05) for _ in range(3)]
    world = World(
        cfgs,
        # Default obstacle damping fully decays the body's velocity at dt=0.1; the
        # substeps and the softer damping are just what makes the puck actually move.
        WorldConfig(collisions=True, obstacle_linear_damping=1.0),
        n_envs=4,
        device="cpu",
        dtype=torch.float64,
        substeps=4,
    )
    n_envs = world.n_envs
    # Agent 0 drives +x into a light circular puck it can shove along; the other two
    # are parked far away so only agent 0's contact matters.
    world.state = world.state._replace(
        pos=torch.tensor([[-0.3, 0.0], [-5.0, 5.0], [5.0, -5.0]], dtype=torch.float64)
        .expand(n_envs, 3, 2)
        .clone(),
        theta=torch.zeros(n_envs, 3, dtype=torch.float64),
    )
    world.set_obstacles(
        Obstacles(
            pos=torch.zeros(n_envs, 1, 2, dtype=torch.float64),
            radius=torch.tensor([0.1], dtype=torch.float64),
            kind=torch.tensor([int(ObstacleKind.MOVABLE)]),
            mass=torch.tensor([0.05], dtype=torch.float64),
        )
    )
    lidar = Lidar(n_rays=4, max_range=MAXR, include_agents=False, backend=backend)
    before = lidar.scan(world).clone()

    actions = torch.zeros(n_envs, 3, world.act_dim, dtype=torch.float64)
    actions[:, 0, 0] = 1.0  # push agent 0 straight at the puck
    with torch.no_grad():
        for _ in range(20):
            world.step(actions)

    live_pos = world.obstacle_state_views()[0]
    assert not torch.allclose(live_pos, world.obstacle_pos)  # the installed pose went stale
    after = lidar.scan(world)
    assert not torch.allclose(after, before)
    # Everything stayed on the x axis, so ray 0 of agent 0 still points at the puck and
    # its range is the live centre-to-surface gap.
    gap = (live_pos[:, 0, 0] - world.state.pos[:, 0, 0] - 0.1).clamp(max=MAXR)
    torch.testing.assert_close(after[:, 0, 0], gap)


# ------------------------------------------------------------------ shapes and classes


def _obstacle_world(obstacles, *, n_agents=3, dtype=torch.float64, device="cpu", configs=None):
    """A collision-free world (so nothing moves unless stepped) with ``obstacles``."""
    from swarp.core.config import WorldConfig
    from swarp.core.world import World
    from swarp.dynamics.base import AgentConfig

    cfgs = configs or [AgentConfig(radius=0.05) for _ in range(n_agents)]
    world = World(cfgs, WorldConfig(collisions=False), n_envs=2, device=device, dtype=dtype)
    if obstacles is not None:
        world.set_obstacles(obstacles)
    return world


def _place(world, pos, theta=None):
    n_envs, n_agents = world.n_envs, world.n_agents
    p = torch.tensor(pos, dtype=world.dtype, device=world.device)
    th = torch.zeros(n_agents, dtype=world.dtype, device=world.device)
    if theta is not None:
        th = torch.tensor(theta, dtype=world.dtype, device=world.device)
    world.state = world.state._replace(
        pos=p.expand(n_envs, n_agents, 2).clone(), theta=th.expand(n_envs, n_agents).clone()
    )


def _obstacles(n_envs, centres, shapes, *, radius=None, half=None, angle=None, kind=None):
    from swarp.core.config import Obstacles

    n = len(centres)
    t = torch.float64
    return Obstacles(
        pos=torch.tensor([centres], dtype=t).expand(n_envs, n, 2).clone(),
        radius=torch.tensor(radius if radius is not None else [0.0] * n, dtype=t),
        shape=torch.tensor([int(s) for s in shapes]),
        half_extents=torch.tensor(half if half is not None else [[0.0, 0.0]] * n, dtype=t),
        angle=None if angle is None else torch.tensor(angle, dtype=t),
        kind=None if kind is None else torch.tensor([int(k) for k in kind]),
    )


def _scan_both(world, **kw):
    """``Lidar.scan`` on both backends; asserts they agree and returns the torch result."""
    kw.setdefault("max_range", MAXR)
    kw.setdefault("body_frame", False)
    kw.setdefault("return_hits", True)
    ref = Lidar(backend="torch", **kw).scan(world)
    got = Lidar(backend="warp", **kw).scan(world)
    for r, g in zip(ref, got, strict=True):
        if r.dtype.is_floating_point:
            torch.testing.assert_close(g, r, **_TOL[r.dtype])
        else:
            assert torch.equal(g, r)
    return ref


@pytest.mark.parametrize("device", DEVICES)
def test_segment_obstacle_is_hit(device):
    """A segment is scanned as the capsule it collides as (core plus radius).

    Down the segment's own axis the first surface is the end cap, at distance
    ``centre - half_len - radius``; across it, the flat side at ``radius`` from the core.
    """
    from swarp.core.config import ObstacleShape

    obs = _obstacles(2, [[1.5, 0.0]], [ObstacleShape.SEGMENT], radius=[0.1], half=[[0.5, 0.0]])
    world = _obstacle_world(obs, device=device)
    _place(world, [[0.0, 0.0], [1.5, -1.0], [5.0, 5.0]])
    ranges, cls, idx = _scan_both(world, n_rays=4, include_agents=False)
    np.testing.assert_allclose(ranges[:, 0, 0].cpu(), 1.5 - 0.5 - 0.1, atol=1e-12)  # axis
    np.testing.assert_allclose(ranges[:, 1, 1].cpu(), 1.0 - 0.1, atol=1e-12)  # flat side
    assert (cls[:, 0, 0] == 2).all() and (idx[:, 0, 0] == 0).all()
    assert (cls[:, 0, 1:] == 0).all() and (idx[:, 0, 1:] == -1).all()


@pytest.mark.parametrize("device", DEVICES)
def test_box_face_hit(device):
    from swarp.core.config import ObstacleShape

    obs = _obstacles(2, [[1.0, 0.0]], [ObstacleShape.BOX], half=[[0.2, 0.1]])
    world = _obstacle_world(obs, device=device)
    _place(world, [[0.0, 0.0], [1.0, -1.0], [1.0, 0.0]])  # agent 2 sits inside the box
    ranges = _scan_both(world, n_rays=4, include_agents=False)[0]
    np.testing.assert_allclose(ranges[:, 0, 0].cpu(), 0.8, atol=1e-12)  # +x onto the -x face
    np.testing.assert_allclose(ranges[:, 1, 1].cpu(), 0.9, atol=1e-12)  # +y onto the -y face
    assert (ranges[:, 0, 1:] == MAXR).all()
    assert (ranges[:, 2] == MAXR).all()  # an origin inside the box misses it


@pytest.mark.parametrize("device", DEVICES)
def test_box_corner_graze(device):
    """A ray just outside a corner misses; one just inside it hits the near face."""
    from swarp.core.config import ObstacleShape

    obs = _obstacles(2, [[1.0, 0.0]], [ObstacleShape.BOX], half=[[0.2, 0.2]])
    world = _obstacle_world(obs, device=device)
    _place(world, [[0.0, 0.2 + 1e-6], [0.0, 0.2 - 1e-6], [0.0, -0.2 - 1e-6]])
    ranges = _scan_both(world, n_rays=1, include_agents=False)[0]
    assert (ranges[:, 0, 0] == MAXR).all()
    np.testing.assert_allclose(ranges[:, 1, 0].cpu(), 0.8, atol=1e-12)
    assert (ranges[:, 2, 0] == MAXR).all()


def _np_ray_polygon(o, d, verts):
    """Independent oracle: nearest ``t > 0`` at which ``o + t d`` crosses an edge."""
    best = np.inf
    for k in range(len(verts)):
        a, b = verts[k], verts[(k + 1) % len(verts)]
        e = b - a
        den = d[0] * (-e[1]) + d[1] * e[0]
        if abs(den) < 1e-15:
            continue
        w = a - o
        t = (w[0] * (-e[1]) + w[1] * e[0]) / den
        s = (d[0] * w[1] - d[1] * w[0]) / den
        if t > 0 and 0 <= s <= 1:
            best = min(best, t)
    return best


def _np_box_verts(c, ang, hx, hy):
    ca, sa = math.cos(ang), math.sin(ang)
    local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    return [np.array([c[0] + ca * x - sa * y, c[1] + sa * x + ca * y]) for x, y in local]


def _np_inside_box(o, c, ang, hx, hy):
    ca, sa = math.cos(ang), math.sin(ang)
    rel = o - np.asarray(c)
    lx, ly = ca * rel[0] + sa * rel[1], -sa * rel[0] + ca * rel[1]
    return abs(lx) < hx and abs(ly) < hy


def _np_ray_circle(o, d, c, r):
    oc = np.asarray(c) - o
    proj = oc @ d
    perp2 = oc @ oc - proj * proj
    if perp2 > r * r or proj <= 0:
        return np.inf
    t = proj - math.sqrt(r * r - perp2)
    return t if t > 0 else np.inf


def _np_ray_capsule(o, d, c, ang, half_len, r):
    u = np.array([math.cos(ang), math.sin(ang)])
    rel = o - np.asarray(c)
    s = np.clip(rel @ u, -half_len, half_len)
    if np.sum((rel - s * u) ** 2) < r * r:
        return np.inf
    rect = _np_box_verts(c, ang, half_len, r)
    ends = [np.asarray(c) + half_len * u, np.asarray(c) - half_len * u]
    return min(_np_ray_polygon(o, d, rect), *(_np_ray_circle(o, d, e, r) for e in ends))


@pytest.mark.parametrize("seed", range(4))
def test_rotated_box_and_capsule_match_independent_oracle(seed):
    """Random rotated boxes and capsules against a ray-vs-edges numpy intersection."""
    from swarp.core.config import ObstacleShape

    rng = np.random.default_rng(seed)
    n_rays = 32
    centre = rng.uniform(-0.3, 0.3, 2)
    ang = float(rng.uniform(0, 2 * math.pi))
    hx, hy = rng.uniform(0.1, 0.4, 2)
    for shape in (ObstacleShape.BOX, ObstacleShape.SEGMENT):
        half = [[hx, 0.0 if shape == ObstacleShape.SEGMENT else hy]]
        obs = _obstacles(2, [centre.tolist()], [shape], radius=[hy * 0.5], half=half, angle=[ang])
        world = _obstacle_world(obs)
        origins = rng.uniform(-1.0, 1.0, (3, 2))
        _place(world, origins.tolist())
        ranges = _scan_both(world, n_rays=n_rays, include_agents=False)[0]
        for a in range(3):
            for r in range(n_rays):
                th = r * (2 * math.pi / n_rays)
                d = np.array([math.cos(th), math.sin(th)])
                if shape == ObstacleShape.BOX:
                    inside = _np_inside_box(origins[a], centre, ang, hx, hy)
                    verts = _np_box_verts(centre, ang, hx, hy)
                    exp = np.inf if inside else _np_ray_polygon(origins[a], d, verts)
                else:
                    exp = _np_ray_capsule(origins[a], d, centre, ang, hx, hy * 0.5)
                np.testing.assert_allclose(float(ranges[0, a, r]), min(exp, MAXR), atol=1e-9)


def _box_agent_configs(n=3, box=(1,), half=(0.2, 0.1)):
    from swarp.dynamics.base import AgentConfig, AgentShape

    return [
        AgentConfig(shape=AgentShape.BOX, half_extents=half) if i in box else AgentConfig()
        for i in range(n)
    ]


@pytest.mark.parametrize("device", DEVICES)
def test_box_agents_are_oriented_by_theta(device):
    """A box agent is hit on its faces as rotated by its heading."""
    world = _obstacle_world(None, configs=_box_agent_configs(), device=device)
    # Agent 1 (0.2 x 0.1 half-extents) turned by 90 deg presents its 0.1 side along x.
    _place(world, [[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]], theta=[0.0, math.pi / 2, 0.0])
    ranges, cls, idx = _scan_both(world, n_rays=4)
    np.testing.assert_allclose(ranges[:, 0, 0].cpu(), 1.0 - 0.1, atol=1e-12)
    assert (cls[:, 0, 0] == 1).all() and (idx[:, 0, 0] == 1).all()
    # The box agent itself sees the disc agent 0 at 1 - 0.05, its own box excluded.
    np.testing.assert_allclose(ranges[:, 1, 2].cpu(), 1.0 - 0.05, atol=1e-12)
    assert (idx[:, 1, 2] == 0).all()
    _place(world, [[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]], theta=[0.0, 0.0, 0.0])
    ranges = _scan_both(world, n_rays=4)[0]
    np.testing.assert_allclose(ranges[:, 0, 0].cpu(), 1.0 - 0.2, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_box_agent_occludes_another_agent(device):
    world = _obstacle_world(None, configs=_box_agent_configs(), device=device)
    _place(world, [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], theta=[0.0, 0.3, 0.0])
    ranges, cls, idx = _scan_both(world, n_rays=4)
    assert (idx[:, 0, 0] == 1).all()  # the box, not the disc behind it
    assert (ranges[:, 0, 0] < 0.5).all()
    assert (idx[:, 2, 2] == 1).all()  # and from the other side too
    _place(world, [[0.0, 0.0], [0.5, 2.0], [1.0, 0.0]])
    ranges, cls, idx = _scan_both(world, n_rays=4)
    assert (idx[:, 0, 0] == 2).all()
    np.testing.assert_allclose(ranges[:, 0, 0].cpu(), 0.95, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_occlusion_order_and_hit_classes(device):
    """Nearest wins across agents, immovable and movable obstacles; ties go to agents,
    then to the lower index."""
    from swarp.core.config import ObstacleKind, ObstacleShape

    im, mv = ObstacleKind.IMMOVABLE, ObstacleKind.MOVABLE
    circle, box = ObstacleShape.CIRCLE, ObstacleShape.BOX
    obs = _obstacles(
        2,
        [[1.0, 0.0], [2.0, 0.0], [0.0, 1.05], [0.0, -1.0], [0.0, -1.0]],
        [circle, box, circle, box, box],
        radius=[0.1, 0.0, 0.05, 0.0, 0.0],
        half=[[0.0, 0.0], [0.1, 0.1], [0.0, 0.0], [0.2, 0.2], [0.2, 0.2]],
        kind=[im, mv, mv, mv, im],
    )
    world = _obstacle_world(obs, device=device)
    # Agent 2 has the same centre and radius as obstacle 2.
    _place(world, [[0.0, 0.0], [3.0, 0.0], [0.0, 1.05]])
    ranges, cls, idx = _scan_both(world, n_rays=4)
    # +x: the immovable circle at 0.9 hides the movable box and agent 1 behind it.
    np.testing.assert_allclose(ranges[:, 0, 0].cpu(), 0.9, atol=1e-12)
    assert (cls[:, 0, 0] == 2).all() and (idx[:, 0, 0] == 0).all()
    # +y: agent 2 and movable circle 2 tie at 1.0, and the agent wins.
    np.testing.assert_allclose(ranges[:, 0, 1].cpu(), 1.0, atol=1e-12)
    assert (cls[:, 0, 1] == 1).all() and (idx[:, 0, 1] == 2).all()
    # -y: two coincident boxes, movable 3 and immovable 4: the lower index wins.
    np.testing.assert_allclose(ranges[:, 0, 3].cpu(), 0.8, atol=1e-12)
    assert (cls[:, 0, 3] == 3).all() and (idx[:, 0, 3] == 3).all()
    # -x: nothing.
    assert (cls[:, 0, 2] == 0).all() and (idx[:, 0, 2] == -1).all()
    # Agent 1 looking back (-x) sees the movable box first.
    np.testing.assert_allclose(ranges[:, 1, 2].cpu(), 0.9, atol=1e-12)
    assert (cls[:, 1, 2] == 3).all() and (idx[:, 1, 2] == 1).all()


def test_fov_endpoints():
    """A partial field of view puts its first and last rays exactly at its endpoints;
    the full circle keeps the historical 2*pi/n spacing."""
    from swarp.sensors.lidar_kernels import ray_step

    lidar = Lidar(n_rays=5, angle_start=-math.pi / 4, angle_end=math.pi / 4)
    off = lidar.ray_offsets(dtype=torch.float64)
    torch.testing.assert_close(
        off, torch.linspace(-math.pi / 4, math.pi / 4, 5, dtype=torch.float64)
    )
    assert ray_step(8, 0.0, None) == 2.0 * math.pi / 8
    assert ray_step(8, 0.0, 2.0 * math.pi) == 2.0 * math.pi / 8  # a full span is a circle
    assert ray_step(1, 0.3, 1.0) == 0.0
    with pytest.raises(ValueError, match="angle_end"):
        Lidar(angle_start=1.0, angle_end=0.5)

    # The endpoint rays hit targets placed exactly at the endpoint angles.
    a = math.pi / 4
    obs_pos = torch.tensor(
        [[[math.cos(a), -math.sin(a)], [math.cos(a), math.sin(a)]]], dtype=torch.float64
    )
    kw = dict(
        n_rays=5,
        max_range=MAXR,
        body_frame=False,
        angle_start=-a,
        angle_end=a,
        obstacle_pos=obs_pos,
        obstacle_radius=torch.tensor([0.1, 0.1], dtype=torch.float64),
    )
    args = (
        torch.zeros(1, 1, 2, dtype=torch.float64),
        torch.zeros(1, 1, dtype=torch.float64),
        torch.tensor([0.05], dtype=torch.float64),
    )
    for fn in (lidar_scan, lidar_scan_warp):
        r = fn(*args, **kw)
        np.testing.assert_allclose(r[0, 0, [0, 4]], 0.9, atol=1e-12)
        assert (r[0, 0, 1:4] == MAXR).all()


def _random_scene(seed, device, dtype, e=3, a=6, t=8):
    """Mixed scene: discs and box agents, circles, boxes and capsules of both kinds."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def u(*shape, lo=0.0, hi=1.0):
        x = torch.rand(*shape, generator=g, dtype=torch.float64) * (hi - lo) + lo
        return x.to(device=device, dtype=dtype)

    shape_a = (torch.rand(a, generator=g) < 0.5).to(torch.int32).to(device)
    half_a = u(a, 2, lo=0.04, hi=0.15) * (shape_a == 1).unsqueeze(-1)
    radius_a = torch.where(shape_a == 1, half_a.norm(dim=-1), u(a, lo=0.04, hi=0.12))
    return dict(
        pos=u(e, a, 2, lo=-1.0, hi=1.0),
        theta=u(e, a, lo=0.0, hi=6.28),
        agent_radius=radius_a,
        agent_shape=shape_a,
        agent_half=half_a,
        obstacle_pos=u(e, t, 2, lo=-1.0, hi=1.0),
        obstacle_radius=u(t, lo=0.02, hi=0.12),
        obstacle_shape=torch.randint(0, 3, (t,), generator=g).to(torch.int32).to(device),
        obstacle_angle=u(e, t, lo=0.0, hi=6.28),
        obstacle_half=u(t, 2, lo=0.03, hi=0.3),
        obstacle_kind=torch.randint(0, 2, (t,), generator=g).to(torch.int32).to(device),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("body_frame", [True, False])
@pytest.mark.parametrize("fov", [None, 2.0])
def test_warp_matches_torch_on_mixed_shapes(device, dtype, body_frame, fov):
    """Warp vs torch on randomized mixed scenes: ranges, classes and indices.

    float64 agrees exactly on what was hit. In float32 a grazing ray can flip between
    "just hit" and "just missed" (or between two near-equal targets) on the backends' own
    sin/cos/sqrt rounding, so there the share of disagreeing rays is bounded instead, and
    ranges are compared on the rays that agree.
    """
    for seed in range(3):
        s = _random_scene(seed, device, dtype)
        pos, theta, rad = s.pop("pos"), s.pop("theta"), s.pop("agent_radius")
        kw = dict(
            n_rays=128,
            max_range=1.5,
            body_frame=body_frame,
            angle_start=-1.0 if fov else 0.3,
            angle_end=(-1.0 + fov) if fov else None,
            return_hits=True,
            **s,
        )
        r_ref, c_ref, i_ref = lidar_scan(pos, theta, rad, **kw)
        r_got, c_got, i_got = lidar_scan_warp(pos, theta, rad, **kw)
        assert c_got.dtype == torch.uint8 and i_got.dtype == torch.int32
        assert set(c_ref.unique().tolist()) == {0, 1, 2, 3}
        same = (c_ref == c_got) & (i_ref == i_got)
        if dtype == torch.float64:
            assert same.all()
        else:
            assert (~same).double().mean() < 2e-3
        torch.testing.assert_close(r_got[same], r_ref[same], **_TOL[dtype])


def test_lidar_gradcheck_boxes_and_capsules():
    """float64 gradcheck through box agents, boxes and capsules at generic poses, with
    respect to agent positions and headings and obstacle positions and angles."""
    from swarp.dynamics.base import AgentShape

    f64 = torch.float64
    pos = torch.tensor([[[0.0, 0.0], [0.55, 0.23], [-0.4, 0.5]]], dtype=f64, requires_grad=True)
    theta = torch.tensor([[0.1, 0.7, -0.4]], dtype=f64, requires_grad=True)
    shape = torch.tensor([0, int(AgentShape.BOX), int(AgentShape.BOX)], dtype=torch.int32)
    half = torch.tensor([[0.0, 0.0], [0.12, 0.07], [0.1, 0.1]], dtype=f64)
    radius = torch.cat([torch.tensor([0.05], dtype=f64), half[1:].norm(dim=-1)])
    obs_pos = torch.tensor(
        [[[0.9, -0.35], [-0.7, -0.6], [0.2, 0.95]]], dtype=f64, requires_grad=True
    )
    obs_angle = torch.tensor([[0.37, -1.1, 2.0]], dtype=f64, requires_grad=True)
    obs_shape = torch.tensor([1, 2, 0], dtype=torch.int32)  # box, capsule, circle
    obs_half = torch.tensor([[0.2, 0.15], [0.3, 0.0], [0.0, 0.0]], dtype=f64)
    obs_rad = torch.tensor([0.0, 0.08, 0.12], dtype=f64)

    def fn(p, th, op, oa):
        return lidar_scan(
            p,
            th,
            radius,
            n_rays=13,
            max_range=MAXR,
            agent_shape=shape,
            agent_half=half,
            obstacle_pos=op,
            obstacle_radius=obs_rad,
            obstacle_shape=obs_shape,
            obstacle_angle=oa,
            obstacle_half=obs_half,
        )

    _, cls, _ = lidar_scan(
        pos,
        theta,
        radius,
        n_rays=13,
        max_range=MAXR,
        agent_shape=shape,
        agent_half=half,
        obstacle_pos=obs_pos,
        obstacle_radius=obs_rad,
        obstacle_shape=obs_shape,
        obstacle_angle=obs_angle,
        obstacle_half=obs_half,
        return_hits=True,
    )
    assert (cls == 1).sum() >= 3 and (cls == 2).sum() >= 6  # the shapes are exercised
    assert torch.autograd.gradcheck(fn, (pos, theta, obs_pos, obs_angle), eps=1e-6, atol=1e-5)


def test_axis_aligned_hits_have_finite_gradients():
    """A ray parallel to a box face (a zero direction component in the box frame) and a
    ray straight at a radius-zero box's centre (a zero-argument sqrt in the unselected
    circle branch) must both backpropagate finite gradients."""
    from swarp.core.config import ObstacleShape

    pos = torch.zeros(1, 1, 2, dtype=torch.float64, requires_grad=True)
    obs_pos = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64, requires_grad=True)
    obs_angle = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    r = lidar_scan(
        pos,
        torch.zeros(1, 1, dtype=torch.float64),
        torch.tensor([0.05], dtype=torch.float64),
        n_rays=4,
        max_range=MAXR,
        obstacle_pos=obs_pos,
        obstacle_radius=torch.tensor([0.0, 0.1], dtype=torch.float64),
        obstacle_shape=torch.tensor([int(ObstacleShape.BOX), int(ObstacleShape.SEGMENT)]),
        obstacle_angle=obs_angle,
        obstacle_half=torch.tensor([[0.2, 0.1], [0.3, 0.0]], dtype=torch.float64),
    )
    np.testing.assert_allclose(r[0, 0, :2].detach(), [0.8, 0.9], atol=1e-12)
    r.sum().backward()
    for t in (pos, obs_pos, obs_angle):
        assert torch.isfinite(t.grad).all()
    np.testing.assert_allclose(pos.grad[0, 0], [-1.0, -1.0], atol=1e-9)


# ----------------------------------------------------------- host sync and capture


def _mixed_world(device, dtype=torch.float64):
    from swarp.core.config import ObstacleShape

    obs = _obstacles(
        2,
        [[0.6, 0.0], [0.0, 0.7], [-0.6, -0.4]],
        [ObstacleShape.BOX, ObstacleShape.SEGMENT, ObstacleShape.CIRCLE],
        radius=[0.0, 0.05, 0.1],
        half=[[0.1, 0.1], [0.3, 0.0], [0.0, 0.0]],
        angle=[0.3, 0.0, 0.0],
    )
    world = _obstacle_world(obs, configs=_box_agent_configs(), device=device, dtype=dtype)
    _place(world, [[0.0, 0.0], [-0.2, 0.3], [0.3, -0.4]], theta=[0.0, 0.4, 1.0])
    return world


@pytest.mark.parametrize("backend", ["torch", "warp"])
def test_scan_has_no_host_sync(backend):
    """A scan over every shape kind reads nothing back to the host.

    The circle-only filter this replaces needed two device->host transfers to decide
    which obstacles to keep; scanning every shape needs none. On CPU the transfer entry
    points are trapped directly (``test_scan_has_no_cuda_sync`` is the CUDA check).
    """
    from unittest import mock

    world = _mixed_world("cpu")
    lidar = Lidar(n_rays=8, max_range=MAXR, return_hits=True, backend=backend)
    warm = [t.clone() for t in lidar.scan(world)]

    def _raise(*a, **kw):
        raise AssertionError("device->host transfer on a lidar scan")

    with (
        mock.patch.object(torch.Tensor, "__bool__", _raise),
        mock.patch.object(torch.Tensor, "item", _raise),
        mock.patch.object(torch.Tensor, "nonzero", _raise),
        mock.patch.object(torch.Tensor, "tolist", _raise),
    ):
        again = lidar.scan(world)
    for w, g in zip(warm, again, strict=True):
        assert torch.equal(w, g)
    assert (again[1] == 2).any() and (again[1] == 1).any()  # obstacles and agents seen


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["torch", "warp"])
def test_scan_has_no_cuda_sync(backend):
    world = _mixed_world("cuda:0", dtype=torch.float32)
    lidar = Lidar(n_rays=8, max_range=MAXR, return_hits=True, backend=backend)
    lidar.scan(world)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        lidar.scan(world)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_launch_requires_bind():
    world = _mixed_world("cpu")
    with pytest.raises(RuntimeError, match="bind"):
        Lidar(backend="warp").launch(world)


def test_bind_is_idempotent():
    world = _mixed_world("cpu")
    lidar = Lidar(n_rays=8, backend="warp")
    lidar.bind(world)
    ptr, version = lidar.ranges.data_ptr(), lidar.version
    lidar.bind(world)
    lidar.scan(world)
    assert (lidar.ranges.data_ptr(), lidar.version) == (ptr, version)


@pytest.mark.gpu
def test_captured_launch_replays_live_geometry():
    """``bind`` outside, ``launch`` inside ``wp.ScopedCapture``: replaying the graph after
    the movable box has been pushed equals an eager scan of the pushed scene."""
    import warp as wp

    from swarp.core.config import ObstacleKind, Obstacles, ObstacleShape, WorldConfig
    from swarp.core.world import World
    from swarp.dynamics.base import AgentConfig

    dev, dt = "cuda:0", torch.float32
    cfgs = [AgentConfig(radius=0.05) for _ in range(3)]
    world = World(
        cfgs,
        WorldConfig(collisions=True, obstacle_linear_damping=1.0),
        n_envs=4,
        device=dev,
        dtype=dt,
        substeps=4,
    )
    n_envs = world.n_envs
    world.state = world.state._replace(
        pos=torch.tensor([[-0.3, 0.0], [-5.0, 5.0], [0.0, -1.0]], device=dev, dtype=dt)
        .expand(n_envs, 3, 2)
        .clone(),
        theta=torch.zeros(n_envs, 3, device=dev, dtype=dt),
    )
    world.set_obstacles(
        Obstacles(
            pos=torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], device=dev, dtype=dt)
            .expand(n_envs, 2, 2)
            .clone(),
            radius=torch.tensor([0.0, 0.05], device=dev, dtype=dt),
            shape=torch.tensor([int(ObstacleShape.BOX), int(ObstacleShape.SEGMENT)]),
            half_extents=torch.tensor([[0.1, 0.1], [0.4, 0.0]], device=dev, dtype=dt),
            kind=torch.tensor([int(ObstacleKind.MOVABLE), int(ObstacleKind.IMMOVABLE)]),
            mass=torch.tensor([0.05, 1.0], device=dev, dtype=dt),
        )
    )
    world.enable_persistent(use_graph=False)  # pointer-stable state, as under capture
    lidar = Lidar(n_rays=16, max_range=MAXR, return_hits=True, backend="warp")
    lidar.bind(world)  # the only allocation and module load: none happen in the capture
    wp.synchronize_device(dev)
    with wp.ScopedCapture(device=dev) as capture:
        lidar.launch(world)

    actions = torch.zeros(n_envs, 3, world.act_dim, device=dev, dtype=dt)
    actions[:, 0, 0] = 1.0  # agent 0 pushes the movable box along +x
    outs = []
    views = (lidar.ranges, lidar.hit_class, lidar.hit_index)
    for _ in range(2):
        with torch.no_grad():
            for _ in range(10):
                world.step(actions)
        eager = [t.clone() for t in lidar.scan(world)]
        for t in views:
            t.zero_()
        wp.capture_launch(capture.graph)
        wp.synchronize_device(dev)
        for e, g in zip(eager, views, strict=True):
            assert torch.equal(e, g)
        outs.append(eager[0])
    assert not torch.equal(outs[0], outs[1])  # the body moved between the two replays
    assert (lidar.hit_class[:, 0, 0] == 3).all()  # agent 0's +x ray is on the movable box


def _random_sweep(seed, device, dtype, e, a):
    """Velocities, turn rates, yaw errors and stamp phases for a sweep scan."""
    g = torch.Generator(device="cpu").manual_seed(100 + seed)

    def u(*shape, lo=0.0, hi=1.0):
        x = torch.rand(*shape, generator=g, dtype=torch.float64) * (hi - lo) + lo
        return x.to(device=device, dtype=dtype)

    return (u(e, a, 2, lo=-0.5, hi=0.5), u(e, a, lo=-2.0, hi=2.0), u(e, a, lo=-0.05, hi=0.05),
            u(e, a), 0.1, -1.0 if seed % 2 else 1.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("body_frame", [True, False])
def test_warp_matches_torch_with_sweep(device, dtype, body_frame):
    """The rotating-scan sweep agrees across the backends on randomized mixed scenes, as
    the plain scan does (float64 exactly on what was hit)."""
    for seed in range(3):
        s = _random_scene(seed, device, dtype)
        pos, theta, rad = s.pop("pos"), s.pop("theta"), s.pop("agent_radius")
        sweep = _random_sweep(seed, device, dtype, *pos.shape[:2])
        kw = dict(n_rays=96, max_range=1.5, body_frame=body_frame, angle_start=-2.0,
                  angle_end=2.0, return_hits=True, sweep=sweep, **s)
        r_ref, c_ref, i_ref = lidar_scan(pos, theta, rad, **kw)
        r_got, c_got, i_got = lidar_scan_warp(pos, theta, rad, **kw)
        plain = lidar_scan(pos, theta, rad, **{**kw, "sweep": None})[0]
        assert not torch.allclose(r_ref, plain)  # the sweep moved the rays
        same = (c_ref == c_got) & (i_ref == i_got)
        if dtype == torch.float64:
            assert same.all()
        else:
            assert (~same).double().mean() < 2e-3
        torch.testing.assert_close(r_got[same], r_ref[same], **_TOL[dtype])


@pytest.mark.parametrize("device", DEVICES)
def test_sweep_casts_each_ray_from_the_pose_at_its_time(device):
    """A robot turning counter-clockwise at 1 rad/s, its scan (0.1 s, counter-clockwise
    from the rear) stamped at the end: the forward ray was taken half a period, 0.05 s,
    earlier, when the robot faced 0.05 rad clockwise of now, so a post dead ahead shows
    up to the left of the forward ray, on the ray at ``a`` with ``a = 1 rad/s * tau(a)``,
    ``tau(a) = (0.5 - a / 2 pi) 0.1 s``. A robot driving forward at 1 m/s instead took it
    0.05 m further back. A sweep at rest is the plain scan."""
    f64 = {"device": device, "dtype": torch.float64}
    pos = torch.zeros(1, 1, 2, **f64)
    theta = torch.zeros(1, 1, **f64)
    rad = torch.tensor([0.05], **f64)
    post = dict(obstacle_pos=torch.tensor([[[1.0, 0.0]]], **f64),
                obstacle_radius=torch.tensor([1e-3], **f64))
    zero = torch.zeros(1, 1, **f64)
    n = 3601  # 0.1 mrad apart over +-0.18 rad
    kw = dict(n_rays=n, max_range=MAXR, angle_start=-0.18, angle_end=0.18, **post)

    def seen(sweep):
        for fn in (lidar_scan, lidar_scan_warp):
            r = fn(pos, theta, rad, sweep=sweep, **kw)[0, 0]
            hit = torch.nonzero(r < MAXR).squeeze(-1).double()
            yield -0.18 + hit.mean().item() * 0.36 / (n - 1), r.min().item()

    turning = (torch.zeros(1, 1, 2, **f64), torch.ones(1, 1, **f64), zero, zero + 1.0, 0.1, 1.0)
    for ang, _ in seen(turning):
        assert ang == pytest.approx(0.05 / (1.0 + 0.1 / (2.0 * math.pi)), abs=2e-4)
    driving = (torch.tensor([[[1.0, 0.0]]], **f64), zero, zero, zero + 1.0, 0.1, 1.0)
    for ang, rng in seen(driving):
        assert ang == pytest.approx(0.0, abs=2e-4)
        assert rng == pytest.approx(1.05 - 1e-3, abs=1e-6)
    rest = (torch.zeros(1, 1, 2, **f64), zero, zero, zero + 0.3, 0.1, 1.0)
    torch.testing.assert_close(lidar_scan(pos, theta, rad, sweep=rest, **kw),
                               lidar_scan(pos, theta, rad, **kw))


# ----------------------------------------------------------------------------- scenery


def _scenery(device, dtype, rows):
    """A one-env :class:`Scenery` tuple from ``(shape, x, y, angle, hx, hy, radius)`` rows."""
    t = torch.tensor(rows, dtype=torch.float64)
    return (
        t[None, :, 1:3].to(device=device, dtype=dtype),
        t[None, :, 3].to(device=device, dtype=dtype),
        t[None, :, 4:6].to(device=device, dtype=dtype),
        t[None, :, 6].to(device=device, dtype=dtype),
        t[None, :, 0].to(device=device, dtype=torch.int32),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn", [lidar_scan, lidar_scan_warp])
def test_scenery_shapes_are_hit_and_empty_slots_are_not(device, fn):
    """Four rays from the origin, one per slot: a circle (r 0.1 at 1 m) reads 0.9, a box
    (half 0.1 x 0.2 at 1 m, ray along its 0.2 side) 0.8, an upright capsule (radius 0.05 at
    -1.5 m) 1.45, and an empty slot is never hit. Scenery is class 2 with index -1, and a
    circle obstacle at the circle's range wins the tie."""
    dt = torch.float64
    sc = _scenery(device, dt, [
        (0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1),
        (1, 0.0, 1.0, 0.0, 0.1, 0.2, 0.0),
        (2, -1.5, 0.0, math.pi / 2, 0.5, 0.0, 0.05),
        (-1, 0.0, -0.5, 0.0, 0.0, 0.0, 0.2),
    ])
    args = (torch.zeros(1, 1, 2, dtype=dt, device=device), torch.zeros(1, 1, dtype=dt,
            device=device), torch.tensor([0.05], dtype=dt, device=device))
    r, c, i = fn(*args, n_rays=4, max_range=MAXR, return_hits=True, scenery=sc)
    np.testing.assert_allclose(r[0, 0].cpu(), [0.9, 0.8, 1.45, MAXR], atol=1e-12)
    assert c[0, 0].tolist() == [2, 2, 2, 0] and i[0, 0].tolist() == [-1, -1, -1, -1]
    tie = dict(obstacle_pos=torch.tensor([[[1.0, 0.0]]], dtype=dt, device=device),
               obstacle_radius=torch.tensor([0.1], dtype=dt, device=device),
               obstacle_kind=torch.tensor([1], dtype=torch.int32, device=device))
    _, c, i = fn(*args, n_rays=4, max_range=MAXR, return_hits=True, scenery=sc, **tie)
    assert (c[0, 0, 0].item(), i[0, 0, 0].item()) == (3, 0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sweep", [False, True])
def test_warp_matches_torch_with_scenery(device, dtype, sweep):
    """Per-env scenery (every shape, its own size in each env, some slots empty) over the
    mixed scenes: the backends agree as they do without it (float64 exactly on what was
    hit), with and without the sweep."""
    for seed in range(3):
        s = _random_scene(seed, device, dtype)
        pos, theta, rad = s.pop("pos"), s.pop("theta"), s.pop("agent_radius")
        e, n = pos.shape[0], 6
        g = torch.Generator(device="cpu").manual_seed(50 + seed)

        def u(*shape, lo=0.0, hi=1.0, g=g):
            x = torch.rand(*shape, generator=g, dtype=torch.float64) * (hi - lo) + lo
            return x.to(device=device, dtype=dtype)

        shape = torch.randint(-1, 3, (e, n), generator=g).to(torch.int32).to(device)
        sc = (u(e, n, 2, lo=-1.2, hi=1.2), u(e, n, lo=0.0, hi=6.28), u(e, n, 2, lo=0.03, hi=0.4),
              u(e, n, lo=0.02, hi=0.15), shape)
        kw = dict(n_rays=96, max_range=1.5, return_hits=True, scenery=sc, **s)
        if sweep:
            kw["sweep"] = _random_sweep(seed, device, dtype, *pos.shape[:2])
        r_ref, c_ref, i_ref = lidar_scan(pos, theta, rad, **kw)
        r_got, c_got, i_got = lidar_scan_warp(pos, theta, rad, **kw)
        bare = lidar_scan(pos, theta, rad, **{**kw, "scenery": None})[1]
        assert ((c_ref == 2) & (bare != 2)).any()  # scenery took rays from the rest
        same = (c_ref == c_got) & (i_ref == i_got)
        if dtype == torch.float64:
            assert same.all()
        else:
            assert (~same).double().mean() < 2e-3
        torch.testing.assert_close(r_got[same], r_ref[same], **_TOL[dtype])
