"""Box-agent contact physics: oriented-box contacts against agents, obstacles and walls.

Forces are read back through the step itself: a holonomic velocity-mode agent of unit mass
with a zero action ends one substep with ``vel = F * dt``, so ``F = vel / dt`` exactly up
to rounding. Analytic references are hand-computed from the geometry.
"""

import math

import numpy as np
import pytest
import torch
import warp as wp
from conftest import DEVICES, _core

from swarp.core.config import ObstacleKind, Obstacles, ObstacleShape, WorldConfig
from swarp.core.stepper import Stepper
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.interop.autograd import TorchState, warp_step

DT = 0.1
K = 100.0
MARGIN = 0.02
BIG = 100.0
F64 = torch.float64


def box(hx=0.2, hy=0.1, **kw):
    kw.setdefault("max_speed", BIG)
    return AgentConfig(shape="box", half_extents=(hx, hy), **kw)


def disc(r=0.1, **kw):
    kw.setdefault("max_speed", BIG)
    return AgentConfig(radius=r, **kw)


def _t(x, device):
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=F64, device=device)


def state(pos, theta=None, vel=None, ang_vel=None, device="cpu"):
    """One env: ``pos`` ``[n, 2]``, optional ``theta``/``vel``/``ang_vel``."""
    pos = _t(pos, device)[None]
    n = pos.shape[1]

    def scalar(x):
        return torch.zeros(1, n, dtype=F64, device=device) if x is None else _t(x, device)[None]

    v = torch.zeros_like(pos) if vel is None else _t(vel, device)[None]
    return TorchState(
        pos=pos, theta=scalar(theta), vel=v, speed=scalar(None), ang_vel=scalar(ang_vel)
    )


def stepper(cfgs, c=0.0, bounds=None, mode="soft", obstacles=None, device="cpu", substeps=1, **kw):
    world = WorldConfig(
        collision_k=K,
        collision_c=c,
        collision_margin=MARGIN,
        bounds=bounds,
        bounds_mode=mode,
        **kw,
    )
    st = Stepper(cfgs, dt=DT, substeps=substeps, device=device, dtype=wp.float64, world=world)
    if obstacles is not None:
        st.set_obstacles(obstacles)
    return st


def forces(cfgs, s, device="cpu", **kw):
    """Contact force on every agent of ``s`` (holonomic, velocity mode, zero action)."""
    st = stepper(cfgs, device=device, **kw)
    acts = torch.zeros(1, len(cfgs), 2, dtype=F64, device=device)
    with torch.no_grad():
        out = warp_step(st, s, acts)
    return (out.vel[0] / DT).cpu()


def obstacles(pos, radius, shape, half=None, angle=None, vel=None, ang_vel=None, **kw):
    n = len(radius)
    return Obstacles(
        pos=_t(pos, "cpu")[None],
        radius=_t(radius, "cpu"),
        shape=torch.as_tensor([int(s) for s in shape], dtype=torch.int32),
        half_extents=_t(np.zeros((n, 2)) if half is None else half, "cpu"),
        angle=None if angle is None else _t(angle, "cpu")[None],
        vel=None if vel is None else _t(vel, "cpu")[None],
        ang_vel=None if ang_vel is None else _t(ang_vel, "cpu")[None],
        **kw,
    )


def close(a, b, atol=1e-10):
    np.testing.assert_allclose(np.asarray(a, dtype=np.float64), b, atol=atol, rtol=0)


# ----------------------------------------------------------------- analytic pairs


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("gap", [-0.01, 0.01])
def test_face_face(device, gap):
    """Two aligned boxes, faces parallel: the depth is the x overlap, the normal is x."""
    s = state([[0.0, 0.0], [0.4 + gap, 0.05]], device=device)
    f = forces([box(), box()], s, device=device)
    mag = K * (MARGIN - gap)
    close(f[0], [-mag, 0.0])
    close(f[1], [mag, 0.0])


@pytest.mark.parametrize("gap", [-0.005, 0.005])
def test_corner_face(gap):
    """A square turned 45 degrees drives its corner into a face, on both sides of touching:
    the exact vertex distance outside, the SAT depth inside, the face normal either way."""
    corner = 0.1 * math.sqrt(2.0)
    s = state([[0.0, 0.0], [corner + 0.2 + gap, 0.03]], theta=[math.pi / 4, 0.0])
    f = forces([box(0.1, 0.1), box(0.2, 0.2)], s)
    close(f[0], [-K * (MARGIN - gap), 0.0])
    close(f[1], -f[0].numpy())


def test_disc_box_face_and_corner():
    # disc of radius 0.1 against the box's left face, 0.09 away: depth 0.1 + 0.02 - 0.09
    f = forces([disc(), box()], state([[0.0, 0.0], [0.29, 0.0]]))
    close(f[0], [-K * 0.03, 0.0])
    close(f[1], [K * 0.03, 0.0])
    # past the corner: the normal is the corner-to-centre diagonal
    d = 0.05 * math.sqrt(2.0)
    f = forces([disc(), box()], state([[0.25, 0.15], [0.0, 0.0]]))
    mag = K * (0.1 + MARGIN - d)
    close(f[0], [mag / math.sqrt(2.0), mag / math.sqrt(2.0)])
    close(f[1], -f[0].numpy())


@pytest.mark.parametrize("phi", [0.2, 0.7, 1.2])
def test_capsule_end_cap_is_round(phi):
    """A box corner 0.05 from a segment's end, at any angle: the capsule of radius 0.05 is
    hit at the same depth along the radial direction, so the end cap is a true circle."""
    seg = obstacles([[0.0, 0.0]], [0.05], [ObstacleShape.SEGMENT], half=[[0.3, 0.0]], angle=[0.0])
    corner = np.array([0.3 + 0.05 * math.cos(phi), 0.05 * math.sin(phi)])
    f = forces([box(0.1, 0.1)], state([corner + 0.1]), obstacles=seg)
    close(f[0], K * MARGIN * np.array([math.cos(phi), math.sin(phi)]))


def test_capsule_side_and_box_and_circle_obstacles():
    seg = obstacles([[0.0, 0.0]], [0.05], [ObstacleShape.SEGMENT], half=[[0.3, 0.0]], angle=[0.0])
    # bottom face 0.06 above the core: depth 0.05 + 0.02 - 0.06
    close(forces([box()], state([[0.05, 0.16]]), obstacles=seg)[0], [0.0, K * 0.01])
    obs = obstacles(
        [[0.0, 0.0], [2.0, 0.0]],
        [0.0, 0.1],
        [ObstacleShape.BOX, ObstacleShape.CIRCLE],
        half=[[0.3, 0.2], [0.0, 0.0]],
        angle=[0.0, 0.0],
    )
    # box obstacle: left face of the agent 0.01 into the obstacle's right face
    close(forces([box()], state([[0.49, 0.1]]), obstacles=obs)[0], [K * 0.03, 0.0])
    # circle obstacle against the agent's top face, 0.08 above the face
    close(forces([box()], state([[2.0, -0.18]]), obstacles=obs)[0], [0.0, -K * 0.04])


# ------------------------------------------------------------- symmetry and SAT


def _scene():
    """A crowded mixed scene: three boxes, a disc and one obstacle of each shape."""
    cfgs = [box(0.2, 0.1), box(0.15, 0.06), box(0.1, 0.1), disc(0.08)]
    pos = np.array([[0.0, 0.0], [0.27, 0.12], [0.02, 0.2], [-0.24, -0.06]])
    theta = np.array([0.3, -0.4, 1.1, 0.0])
    vel = np.array([[0.2, -0.1], [-0.3, 0.05], [0.1, 0.2], [0.4, 0.0]])
    ang_vel = np.array([0.5, -1.0, 2.0, 0.0])
    obs = dict(
        pos=np.array([[0.2, -0.2], [-0.3, 0.2], [0.42, 0.3]]),
        radius=[0.0, 0.03, 0.1],
        shape=[ObstacleShape.BOX, ObstacleShape.SEGMENT, ObstacleShape.CIRCLE],
        half=[[0.15, 0.05], [0.2, 0.0], [0.0, 0.0]],
        angle=np.array([0.2, -0.5, 0.0]),
        vel=np.array([[0.1, 0.0], [0.0, -0.2], [0.0, 0.1]]),
        ang_vel=[0.3, -0.4, 1.0],
    )
    return cfgs, pos, theta, vel, ang_vel, obs


def _rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def test_rotation_and_translation_invariance():
    cfgs, pos, theta, vel, ang_vel, obs = _scene()
    f0 = forces(cfgs, state(pos, theta, vel, ang_vel), c=1.0, obstacles=obstacles(**obs))
    assert (f0.norm(dim=-1) > 1e-3).sum() >= 3, "the scene should have several contacts"
    alpha, shift = 0.7, np.array([0.3, -0.5])
    r = _rot(alpha)
    moved = dict(obs, pos=obs["pos"] @ r.T + shift, angle=obs["angle"] + alpha,
                 vel=obs["vel"] @ r.T)  # fmt: skip
    s = state(pos @ r.T + shift, theta + alpha, vel @ r.T, ang_vel)
    f1 = forces(cfgs, s, c=1.0, obstacles=obstacles(**moved))
    close(f1, f0.numpy() @ r.T, atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_agent_forces_cancel(device):
    """Equal masses, damping and spin on: agent-agent forces sum to zero over the env."""
    rng = np.random.default_rng(3)
    cfgs = [box(0.2, 0.08), box(0.1, 0.1), box(0.15, 0.05), box(0.05, 0.2), disc(0.1), disc(0.06)]
    n = len(cfgs)
    s = state(
        rng.random((n, 2)) * 0.35,
        rng.uniform(-3, 3, n),
        rng.normal(size=(n, 2)),
        rng.normal(size=n),
        device=device,
    )
    f = forces(cfgs, s, c=1.0, device=device)
    assert (f.norm(dim=-1) > 1e-3).sum() >= 4
    close(f.sum(0), [0.0, 0.0], atol=1e-12)


def test_plus_crossing_is_detected():
    """Two thin boxes crossing in a '+': no corner of either lies inside the other, so a
    vertex-only test misses it; the separating-axis test finds the overlap."""
    s = state([[0.0, 0.0], [0.01, 0.02]], theta=[0.0, math.pi / 2])
    f = forces([box(0.3, 0.02), box(0.3, 0.02)], s)
    assert f[0].norm() > 10.0
    close(f[0] + f[1], [0.0, 0.0], atol=1e-12)
    # the push is along one of the four face normals
    assert min(abs(float(f[0, 0])), abs(float(f[0, 1]))) < 1e-9


# --------------------------------------------------------------------- bounds


def _extents(hx, hy, th):
    return (hx * abs(math.cos(th)) + hy * abs(math.sin(th)),
            hx * abs(math.sin(th)) + hy * abs(math.cos(th)))  # fmt: skip


def test_walls_see_the_support_extents():
    th = 0.5
    ex, ey = _extents(0.2, 0.1, th)
    s = state([[-1.0 + ex + 0.01, 1.0 - ey - 0.005]], theta=[th])
    f = forces([box()], s, bounds=(-1.0, 1.0, -1.0, 1.0))
    close(f[0], [K * 0.01, -K * 0.015])


def test_wall_damping_reads_the_support_vertex_velocity():
    """No spring, no translation: a box spinning its corner into the left wall is damped
    by that corner's velocity ``w x r``, not by its (zero) centre velocity."""
    th, w, c = 0.5, 3.0, 1.0
    ex, _ = _extents(0.2, 0.1, th)
    r_y = -0.2 * math.sin(th) + 0.1 * math.cos(th)  # lever of the -x support vertex
    s = state([[-1.0 + ex + 0.01, 0.0]], theta=[th], ang_vel=[w])
    st_kw = dict(c=c, bounds=(-1.0, 1.0, -1.0, 1.0))
    f_still = forces([box()], state([[-1.0 + ex + 0.01, 0.0]], theta=[th]), **st_kw)
    f_spin = forces([box()], s, **st_kw)
    vn = -w * r_y  # normal velocity of the touching corner, into the wall when negative
    expected = (K * 0.01 - c * vn) / (1.0 + c * DT)
    close(f_still[0], [K * 0.01 / (1.0 + c * DT), 0.0])
    close(f_spin[0], [expected, 0.0])


def test_clamp_holds_each_footprint_inside():
    """``bounds_mode="clamp"`` insets a box by its support extents at its *new* heading and
    a disc by its radius, in the same world."""
    w = 2.0
    cfgs = [
        box(model=DynamicsModel.DIFF_DRIVE, max_ang_vel=BIG),
        disc(),
    ]
    s = state([[-1.5, 1.5], [1.5, -1.5]], theta=[0.3, 0.0])
    st = stepper(cfgs, bounds=(-1.0, 1.0, -1.0, 1.0), mode="clamp")
    acts = torch.tensor([[[0.0, w], [0.0, 0.0]]], dtype=F64)
    with torch.no_grad():
        out = warp_step(st, s, acts)
    ex, ey = _extents(0.2, 0.1, 0.3 + w * DT)
    close(out.theta[0, 0], 0.3 + w * DT, atol=1e-12)
    close(out.pos[0, 0], [-1.0 + ex, 1.0 - ey], atol=1e-12)
    close(out.pos[0, 1], [0.9, -0.9], atol=1e-12)


# --------------------------------------------------------------- differentiability


def _face_vertex_pose():
    """Box B's support vertex 0.01 deep in box A's top face, a disc against A's bottom
    face: a generic pose, away from every branch switch the finite differences could hit."""
    ha, hb = (0.2, 0.1), (0.15, 0.08)
    tha, thb = 0.3, 1.0
    n = np.array([-math.sin(tha), math.cos(tha)])
    t = np.array([math.cos(tha), math.sin(tha)])
    ub0 = np.array([math.cos(thb), math.sin(thb)])
    ub1 = np.array([-math.sin(thb), math.cos(thb)])
    rb = hb[0] * abs(ub0 @ n) + hb[1] * abs(ub1 @ n)
    pos = np.array([[0.0, 0.0], n * (ha[1] + rb - 0.01), -n * (ha[1] + 0.06 - 0.01) + 0.05 * t])
    cfgs = [
        box(*ha, model=DynamicsModel.DIFF_DRIVE, max_ang_vel=BIG, max_accel=BIG),
        box(*hb, model=DynamicsModel.DIFF_DRIVE, max_ang_vel=BIG, max_accel=BIG,
            ctrl_mode=ControlMode.ACCELERATION, max_ang_accel=BIG),
        disc(0.06, max_accel=BIG),
    ]  # fmt: skip
    return cfgs, pos, np.array([tha, thb, 0.0])


def test_gradcheck_through_box_contacts():
    cfgs, pos, theta = _face_vertex_pose()
    st = stepper(cfgs, c=1.0)
    rng = np.random.default_rng(0)
    s = state(pos, theta, 0.1 * rng.normal(size=(3, 2)), 0.5 * rng.normal(size=3))
    s = TorchState(*(x.clone().requires_grad_(True) for x in _core(s)))
    acts = torch.tensor([[[0.1, 0.4], [0.05, -0.3], [0.1, 0.0]]], dtype=F64, requires_grad=True)

    def fn(p, th, v, sp, w, a):
        return tuple(warp_step(st, TorchState(p, th, v, sp, w), a))[:5]

    assert torch.autograd.gradcheck(fn, (*_core(s), acts), eps=1e-6, atol=1e-5)
    # the contact geometry, headings included, reaches the outputs
    out = warp_step(st, s, acts)
    out.pos[0, 0].sum().backward()
    assert s.theta.grad[0, 1].abs() > 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_taped_and_untaped_forward_match(device):
    cfgs, pos, theta = _face_vertex_pose()
    rng = np.random.default_rng(1)
    s = state(pos, theta, 0.1 * rng.normal(size=(3, 2)), 0.5 * rng.normal(size=3), device=device)
    acts = torch.tensor([[[0.1, 0.4], [0.05, -0.3], [0.1, 0.0]]], dtype=F64, device=device)
    st = stepper(cfgs, c=1.0, bounds=(-1.0, 1.0, -1.0, 1.0), device=device, substeps=2)
    with torch.no_grad():
        ref = warp_step(st, s, acts)
    taped = warp_step(st, TorchState(*(x.clone().requires_grad_(True) for x in _core(s))), acts)
    for a, b in zip(_core(ref), _core(taped), strict=True):
        torch.testing.assert_close(a, b.detach(), rtol=0, atol=1e-13)


@pytest.mark.parametrize("device", DEVICES)
def test_box_world_is_deterministic(device):
    cfgs, pos, theta, vel, ang_vel, obs = _scene()
    cfgs = [*cfgs[:3], box(0.1, 0.05, model=DynamicsModel.DIFF_DRIVE, max_ang_vel=BIG), cfgs[3]]
    pos = np.concatenate([pos[:3], [[-0.1, -0.25]], pos[3:]])

    def run():
        st = stepper(cfgs, c=1.0, bounds=(-1.0, 1.0, -1.0, 1.0), obstacles=obstacles(**obs),
                     device=device, substeps=4)  # fmt: skip
        s = state(pos, np.r_[theta[:3], 0.2, theta[3:]], device=device)
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for _ in range(20):
                a = torch.randn(1, len(cfgs), 2, dtype=F64, generator=gen).to(device)
                s = warp_step(st, s, a)
        return torch.cat([x.flatten() for x in _core(s)])

    first = run()
    assert torch.isfinite(first).all()
    assert torch.equal(first, run())


# ------------------------------------------------------------------ movable bodies


def _movable(device, agent, shape=ObstacleShape.BOX, half=(0.15, 0.15), radius=0.1, **kw):
    cfg = WorldConfig(
        collisions=True,
        collision_k=1000.0,
        collision_c=kw.pop("c", 20.0),
        collision_margin=0.01,
        bounds=(-2.0, 2.0, -2.0, 2.0),
        obstacle_linear_damping=0.0,
        obstacle_angular_damping=0.0,
    )
    w = World([agent], cfg, n_envs=1, device=device, dt=0.05, dtype=F64, **kw)
    w.set_obstacles(
        Obstacles(
            torch.zeros(1, 1, 2, dtype=F64, device=device),
            torch.full((1,), radius, dtype=F64, device=device),
            shape=torch.full((1,), int(shape), dtype=torch.int32, device=device),
            angle=torch.zeros(1, 1, dtype=F64, device=device),
            half_extents=torch.tensor([half], dtype=F64, device=device),
            kind=torch.full((1,), int(ObstacleKind.MOVABLE), dtype=torch.int32, device=device),
            mass=torch.full((1,), 2.0, dtype=F64, device=device),
            inertia=torch.full((1,), 0.03, dtype=F64, device=device),
        )
    )
    return w


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("y, spin", [(0.1, -1.0), (-0.1, 1.0)])
def test_box_agent_spins_a_movable_box_off_centre(device, y, spin):
    """Pushing +x above the centre turns the body clockwise, below it anticlockwise."""
    w = _movable(device, box(0.05, 0.05, max_speed=1.0), substeps=4)
    w.write_state(pos=torch.tensor([[[-0.4, y]]], dtype=F64, device=device))
    act = torch.tensor([[[1.0, 0.0]]], dtype=F64, device=device)
    with torch.no_grad():
        for _ in range(30):
            w.step(act)
    pos, angle, _, _ = w.obstacle_state_views()
    assert pos[0, 0, 0] > 0.05, "the body was not pushed"
    assert spin * angle[0, 0] > 1e-3, f"wrong spin sign: angle {float(angle[0, 0])}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [ObstacleShape.BOX, ObstacleShape.SEGMENT, ObstacleShape.CIRCLE])
def test_newtons_third_law_agent_body(device, shape):
    """One substep from rest, no body damping: the body's momentum change is exactly the
    negative of the agent's contact impulse, and its torque is ``r x F`` about the body
    centre through the contact point (here the middle of the corner-to-face gap)."""
    g, cy = 0.004, 0.07
    corner = 0.05 * math.sqrt(2.0)
    face = {ObstacleShape.BOX: -0.15, ObstacleShape.SEGMENT: -0.3 - 0.05,
            ObstacleShape.CIRCLE: -0.1}[shape]  # fmt: skip
    if shape != ObstacleShape.BOX:
        cy = 0.0  # a round face is only flat at its pole
    half = (0.3, 0.0) if shape == ObstacleShape.SEGMENT else (0.15, 0.15)
    radius = 0.05 if shape == ObstacleShape.SEGMENT else 0.1
    agent = box(0.05, 0.05, mass=1.5)
    w = _movable(device, agent, shape=shape, half=half, radius=radius, substeps=1)
    w.write_state(
        pos=torch.tensor([[[face - g - corner, cy]]], dtype=F64, device=device),
        theta=torch.full((1, 1), math.pi / 4, dtype=F64, device=device),
        vel=torch.tensor([[[0.3, 0.1]]], dtype=F64, device=device),
        ang_vel=torch.full((1, 1), 0.5, dtype=F64, device=device),
    )
    with torch.no_grad():
        w.step(torch.zeros(1, 1, 2, dtype=F64, device=device))
    dt = 0.05
    f_agent = (w.state.vel[0, 0] * 1.5 / dt).cpu().numpy()
    _, _, b_vel, b_om = w.obstacle_state_views()
    f_body = (b_vel[0, 0] * 2.0 / dt).cpu().numpy()
    tau = float(b_om[0, 0]) * 0.03 / dt
    assert f_body[0] > 1.0
    close(f_body, -f_agent, atol=1e-9)
    if shape == ObstacleShape.CIRCLE:
        assert tau == 0.0
    elif shape == ObstacleShape.BOX:
        r = np.array([face - g / 2, cy])
        close(tau, r[0] * f_body[1] - r[1] * f_body[0], atol=1e-9)
