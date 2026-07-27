"""Immovable vs movable obstacles.

Immovable obstacles are infinite-mass scenery. Movable ones carry a mass/inertia and are
integrated by :mod:`wmas.core.bodies` inside the substep loop from the reaction of the
same agent contacts, so they can be pushed, spun, and shoved into each other.
"""

import pytest
import torch
from conftest import DEVICES

from wmas.core.config import ObstacleKind, ObstacleShape, WorldConfig
from wmas.core.world import World
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel


def _world(device, n_envs=2, substeps=4, **cfg_kw):
    cfgs = [
        AgentConfig(
            model=DynamicsModel.HOLONOMIC,
            ctrl_mode=ControlMode.VELOCITY,
            radius=0.05,
            max_speed=1.0,
        )
    ]
    cfg = WorldConfig(
        collisions=True,
        collision_k=1000.0,
        collision_c=20.0,
        collision_margin=0.01,
        bounds=(-2.0, 2.0, -2.0, 2.0),
        **cfg_kw,
    )
    return World(
        cfgs, cfg, n_envs=n_envs, device=device, dt=0.05, substeps=substeps,
        dtype=torch.float32,
    )


def _install(w, kinds, shape=ObstacleShape.CIRCLE, centers=None, half=0.08):
    n = len(kinds)
    dev, dt = w.device, w.dtype
    if centers is None:
        centers = [[0.0, 0.0]] * n
    pos = torch.tensor(centers, device=dev, dtype=dt).unsqueeze(0).expand(w.n_envs, n, 2)
    w.set_obstacles(
        pos.contiguous(),
        torch.full((n,), 0.1, device=dev, dtype=dt),
        shape=torch.full((n,), int(shape), device=dev, dtype=torch.int32),
        angle=torch.zeros(w.n_envs, n, device=dev, dtype=dt),
        half_extents=torch.full((n, 2), half, device=dev, dtype=dt),
        kind=torch.tensor([int(k) for k in kinds], device=dev, dtype=torch.int32),
        mass=torch.full((n,), 1.0, device=dev, dtype=dt),
        inertia=torch.full((n,), 0.01, device=dev, dtype=dt),
    )


def _push(w, steps=30, start=(-0.4, 0.0), heading=(1.0, 0.0)):
    """Drive the single agent from ``start`` along ``heading`` for ``steps`` steps."""
    dev, dt = w.device, w.dtype
    w.state.pos.data.copy_(
        torch.tensor([start], device=dev, dtype=dt).unsqueeze(0).expand(w.n_envs, 1, 2)
    )
    w.state.vel.data.zero_()
    act = torch.tensor([heading], device=dev, dtype=dt).unsqueeze(0).expand(w.n_envs, 1, 2)
    for _ in range(steps):
        w.step(act.contiguous())
    return w.movable_obstacle_state()


@pytest.mark.parametrize("device", DEVICES)
def test_immovable_never_moves(device):
    w = _world(device)
    _install(w, [ObstacleKind.IMMOVABLE])
    pos, angle, vel, ang_vel = _push(w)
    assert torch.equal(pos, torch.zeros_like(pos))
    assert torch.equal(angle, torch.zeros_like(angle))
    assert torch.equal(vel, torch.zeros_like(vel))
    assert torch.equal(ang_vel, torch.zeros_like(ang_vel))
    # ...and it still repels: the agent cannot walk through it.
    assert w.state.pos.data[:, 0, 0].max().item() < 0.1 + 0.05 + 0.02


@pytest.mark.parametrize("device", DEVICES)
def test_movable_is_pushed_along_the_push_direction(device):
    w = _world(device)
    _install(w, [ObstacleKind.MOVABLE])
    pos, _, vel, _ = _push(w)
    assert pos[:, 0, 0].min().item() > 0.05, "movable obstacle was not pushed"
    assert pos[:, 0, 1].abs().max().item() < 1e-3, "a head-on push should not deflect it"
    assert vel[:, 0, 0].min().item() > 0.0
    assert torch.isfinite(pos).all() and torch.isfinite(vel).all()


@pytest.mark.parametrize("device", DEVICES)
def test_offcentre_push_spins_a_movable_box(device):
    """A normal force applied off the centroid is a torque — that is what makes pose
    tasks (Push-T) solvable. A circle cannot spin: its normal passes through its centre."""
    w = _world(device)
    _install(w, [ObstacleKind.MOVABLE], shape=ObstacleShape.BOX, half=0.15)
    _, angle, _, ang_vel = _push(w, start=(-0.4, 0.1))
    assert angle.abs().max().item() > 1e-3, "off-centre push produced no rotation"
    assert torch.isfinite(ang_vel).all()

    w2 = _world(device)
    _install(w2, [ObstacleKind.MOVABLE], shape=ObstacleShape.CIRCLE)
    _, angle_c, _, _ = _push(w2, start=(-0.4, 0.1))
    assert torch.equal(angle_c, torch.zeros_like(angle_c)), "frictionless circle spun"


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_kinds_in_one_world(device):
    """A movable body shoved into immovable scenery stops at it, not through it."""
    w = _world(device)
    _install(
        w,
        [ObstacleKind.MOVABLE, ObstacleKind.IMMOVABLE],
        centers=[[0.0, 0.0], [1.2, 0.0]],
    )
    pos, _, _, _ = _push(w, steps=80)
    assert torch.equal(pos[:, 1], torch.zeros_like(pos[:, 1]) + pos[:, 1])  # finite
    assert torch.equal(pos[:, 1, 0], torch.full_like(pos[:, 1, 0], 1.2)), "scenery moved"
    # The movable one is somewhere between its start and the wall of scenery.
    assert 0.0 < pos[:, 0, 0].min().item() < 1.2


@pytest.mark.parametrize("device", DEVICES)
def test_movable_box_stops_at_an_immovable_wall(device):
    """A capsule wall must act along its whole spine. Treating it as a disc at its centre
    let a pushed box slide past the wall's ends."""
    w = _world(device, n_envs=1)
    dev, dt = device, torch.float32
    w.set_obstacles(
        torch.tensor([[[0.0, 0.0], [0.6, 0.0]]], device=dev, dtype=dt),
        torch.tensor([0.0, 0.04], device=dev, dtype=dt),
        shape=torch.tensor(
            [int(ObstacleShape.BOX), int(ObstacleShape.SEGMENT)], device=dev, dtype=torch.int32
        ),
        angle=torch.tensor([[0.0, 1.5707963]], device=dev, dtype=dt),  # wall across +x
        half_extents=torch.tensor([[0.1, 0.1], [0.8, 0.0]], device=dev, dtype=dt),
        kind=torch.tensor(
            [int(ObstacleKind.MOVABLE), int(ObstacleKind.IMMOVABLE)], device=dev, dtype=torch.int32
        ),
        mass=torch.tensor([1.0, 1.0], device=dev, dtype=dt),
        inertia=torch.tensor([0.02, 1.0], device=dev, dtype=dt),
    )
    pos, _, _, _ = _push(w, steps=120, start=(-0.35, 0.0))
    # Box half-extent 0.1 + wall radius 0.04: its centre cannot get past ~0.46.
    assert pos[0, 0, 0].item() < 0.5, f"box passed through the wall (x={pos[0, 0, 0].item():.3f})"
    assert pos[0, 0, 0].item() > 0.0, "box was never pushed"


@pytest.mark.parametrize("device", DEVICES)
def test_damping_bounds_a_pushed_body(device):
    """Drag stands in for table friction: a pushed body settles instead of running away."""
    fast = _world(device, obstacle_linear_damping=2.0)
    slow = _world(device, obstacle_linear_damping=40.0)
    for w in (fast, slow):
        _install(w, [ObstacleKind.MOVABLE])
    _, _, v_fast, _ = _push(fast)
    _, _, v_slow, _ = _push(slow)
    assert v_slow[:, 0, 0].max().item() < v_fast[:, 0, 0].max().item()


@pytest.mark.parametrize("device", DEVICES)
def test_compound_body_stays_rigid(device):
    """Two boxes sharing one body must move and rotate as one piece (Push-T's T).

    Both shapes belong to body 0, offset along local y; the invariants are that the
    world distance between the shapes never changes and that they keep one angle.
    """
    w = _world(device, n_envs=1)
    dev, dt = device, torch.float32
    off = torch.tensor([[0.0, 0.1], [0.0, -0.1]], device=dev, dtype=dt)
    centers = torch.tensor([[[0.0, 0.1], [0.0, -0.1]]], device=dev, dtype=dt)
    w.set_obstacles(
        centers,
        torch.zeros(2, device=dev, dtype=dt),
        shape=torch.full((2,), int(ObstacleShape.BOX), device=dev, dtype=torch.int32),
        angle=torch.zeros(1, 2, device=dev, dtype=dt),
        half_extents=torch.tensor([[0.12, 0.05], [0.05, 0.12]], device=dev, dtype=dt),
        kind=torch.full((2,), int(ObstacleKind.MOVABLE), device=dev, dtype=torch.int32),
        mass=torch.full((2,), 1.0, device=dev, dtype=dt),
        inertia=torch.full((2,), 0.02, device=dev, dtype=dt),
        body=torch.zeros(2, device=dev, dtype=torch.int32),  # both shapes -> body 0
        body_offset=off,
    )
    d0 = (centers[0, 0] - centers[0, 1]).norm().item()
    pos, angle, _, _ = _push(w, steps=60, start=(-0.4, 0.06))
    d1 = (pos[0, 0] - pos[0, 1]).norm().item()
    assert d1 == pytest.approx(d0, abs=1e-5), "compound body came apart"
    assert angle[0, 0].item() == pytest.approx(angle[0, 1].item(), abs=1e-6)
    assert pos[0, :, 0].min().item() > 0.02, "compound body was not pushed"
    assert abs(angle[0, 0].item()) > 1e-4, "off-centre push did not rotate the body"


@pytest.mark.parametrize("device", DEVICES)
def test_movable_obstacle_determinism(device):
    def run():
        w = _world(device)
        _install(w, [ObstacleKind.MOVABLE], shape=ObstacleShape.BOX)
        return [t.clone() for t in _push(w, start=(-0.4, 0.07))]

    for a, b in zip(run(), run(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_render_geometry_reports_kind_and_live_pose(device):
    from wmas.render.geometry import extract_geometry

    w = _world(device)
    _install(w, [ObstacleKind.MOVABLE, ObstacleKind.IMMOVABLE], centers=[[0.0, 0.0], [1.2, 0.0]])
    _push(w, steps=20)
    g = extract_geometry(w, 0)
    assert g.obstacle_kind is not None
    assert list(g.obstacle_kind) == [int(ObstacleKind.MOVABLE), int(ObstacleKind.IMMOVABLE)]
    # The pushed body's *live* pose, not the tensor that was installed at reset.
    assert g.obstacle_pos[0][0] > 0.05
    assert g.obstacle_pos[1][0] == pytest.approx(1.2)
