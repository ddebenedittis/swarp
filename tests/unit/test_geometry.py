"""Box and segment static-obstacle collision geometry, with analytic references.

Single holonomic agent (velocity mode, zero action, no damping): the post-step
position is ``p + F * dt^2`` with ``F = k * overlap * n`` (m = 1), so each case
reduces to a hand- or numpy-computed contact normal and overlap.
"""

import numpy as np
import pytest
import torch
import warp as wp
from conftest import DEVICES, _core, _map5

from swarp.core.config import Obstacles, ObstacleShape, WorldConfig
from swarp.core.stepper import Stepper
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.interop.autograd import TorchState, warp_step

K, DT = 100.0, 0.1


def holo(n=1, radius=0.1):
    return [
        AgentConfig(
            model=DynamicsModel.HOLONOMIC,
            ctrl_mode=ControlMode.VELOCITY,
            radius=radius,
            max_speed=100.0,
            max_accel=100.0,
        )
        for _ in range(n)
    ]


def make_state(pos, dtype=torch.float64):
    pos = torch.as_tensor(pos, dtype=dtype).unsqueeze(0)
    n = pos.shape[1]
    z = torch.zeros(1, n, dtype=dtype)
    return TorchState(
        pos=pos, theta=z.clone(), vel=torch.zeros_like(pos), speed=z.clone(), ang_vel=z.clone()
    )


def stepper_with_obstacle(
    device,
    shape,
    center,
    radius=0.0,
    angle=0.0,
    half=(0.0, 0.0),
    n_agents=1,
    agent_radius=0.1,
    margin=0.02,
):
    world = WorldConfig(collision_k=K, collision_c=0.0, collision_margin=margin)
    st = Stepper(holo(n_agents, agent_radius), dt=DT, device=device, dtype=wp.float64, world=world)
    st.set_obstacles(
        Obstacles(
            pos=torch.tensor([center], dtype=torch.float64, device=device).view(1, -1, 2),
            radius=torch.tensor([radius], dtype=torch.float64, device=device),
            shape=torch.tensor([int(shape)], dtype=torch.int32, device=device),
            angle=torch.tensor([angle], dtype=torch.float64, device=device),
            half_extents=torch.tensor([half], dtype=torch.float64, device=device),
        )
    )
    return st


def step_pos(stepper, device, agent_pos):
    state = _map5(make_state([agent_pos]), lambda t: t.to(device))
    actions = torch.zeros(1, 1, 2, dtype=torch.float64, device=device)
    with torch.no_grad():
        out = warp_step(stepper, state, actions)
    return out.pos[0, 0].cpu().numpy()


def box_sdf_normal(p, center, angle, half):
    """Independent numpy reference: signed distance + outward unit normal."""
    ca, sa = np.cos(angle), np.sin(angle)
    d = np.asarray(p) - np.asarray(center)
    lx = ca * d[0] + sa * d[1]
    ly = -sa * d[0] + ca * d[1]
    q = np.array([abs(lx), abs(ly)]) - np.asarray(half)
    cl = np.clip([lx, ly], [-half[0], -half[1]], [half[0], half[1]])
    o = np.array([lx, ly]) - cl
    outd = np.hypot(o[0], o[1])
    if outd > 1e-5:
        s, nl = outd, o / outd
    elif q[0] > q[1]:
        s, nl = q[0], np.array([1.0 if lx >= 0 else -1.0, 0.0])
    else:
        s, nl = q[1], np.array([0.0, 1.0 if ly >= 0 else -1.0])
    n = np.array([ca * nl[0] - sa * nl[1], sa * nl[0] + ca * nl[1]])
    return s, n


def expect_box(p, center, angle, half, reach):
    s, n = box_sdf_normal(p, center, angle, half)
    overlap = reach - s
    return np.asarray(p) + n * (K * overlap) * DT * DT


@pytest.mark.parametrize("device", DEVICES)
def test_box_face_analytic(device):
    st = stepper_with_obstacle(device, ObstacleShape.BOX, [0.3, 0.0], half=(0.1, 0.2))
    # local (-0.2,0): closest face x=0.2, dist 0.1, reach 0.12, overlap 0.02, n=(-1,0)
    # pos = 0.1 + (-1)*(100*0.02)*0.01 = 0.1 - 0.02 = 0.08
    np.testing.assert_allclose(step_pos(st, device, [0.1, 0.0]), [0.08, 0.0], atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_box_corner_analytic(device):
    st = stepper_with_obstacle(device, ObstacleShape.BOX, [0.0, 0.0], half=(0.1, 0.1))
    p = [0.16, 0.16]  # nearest point is the (0.1,0.1) corner
    got = step_pos(st, device, p)
    np.testing.assert_allclose(got, expect_box(p, [0.0, 0.0], 0.0, (0.1, 0.1), 0.12), atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_box_rotated_analytic(device):
    angle = np.pi / 4
    st = stepper_with_obstacle(device, ObstacleShape.BOX, [0.0, 0.0], angle=angle, half=(0.1, 0.1))
    p = [0.18, 0.02]
    got = step_pos(st, device, p)
    np.testing.assert_allclose(got, expect_box(p, [0.0, 0.0], angle, (0.1, 0.1), 0.12), atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_box_interior_pushout(device):
    """The discriminating test: an agent center *inside* a filled box is pushed
    out. A plain closest-point clamp would give ~zero force here (dead-zone)."""
    st = stepper_with_obstacle(device, ObstacleShape.BOX, [0.0, 0.0], half=(0.1, 0.1))
    got = step_pos(st, device, [0.03, 0.0])  # inside, nearest to the +x face
    assert got[0] > 0.12  # ejected well past the +x surface
    assert abs(got[1]) < 1e-9  # purely along +x


@pytest.mark.parametrize("device", DEVICES)
def test_segment_midspan_analytic(device):
    st = stepper_with_obstacle(
        device, ObstacleShape.SEGMENT, [0.0, 0.0], radius=0.05, half=(0.2, 0.0), agent_radius=0.05
    )
    # closest point (0,0), dist 0.1, reach 0.05+0.05+0.02=0.12, overlap 0.02, n=(0,1)
    np.testing.assert_allclose(step_pos(st, device, [0.0, 0.1]), [0.0, 0.12], atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_segment_endpoint_analytic(device):
    st = stepper_with_obstacle(
        device, ObstacleShape.SEGMENT, [0.0, 0.0], radius=0.05, half=(0.2, 0.0), agent_radius=0.05
    )
    p = np.array([0.3, 0.05])
    closest = np.array([0.2, 0.0])  # t clamps to +half_len
    d = p - closest
    dist = np.hypot(*d)
    overlap = 0.12 - dist
    expected = p + (d / dist) * (K * overlap) * DT * DT
    np.testing.assert_allclose(step_pos(st, device, p.tolist()), expected, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_segment_degenerate_matches_circle(device):
    """A zero-length segment (capsule) is exactly a circle obstacle."""
    seg = stepper_with_obstacle(
        device, ObstacleShape.SEGMENT, [0.3, 0.0], radius=0.15, half=(0.0, 0.0)
    )
    circ = stepper_with_obstacle(device, ObstacleShape.CIRCLE, [0.3, 0.0], radius=0.15)
    np.testing.assert_allclose(
        step_pos(seg, device, [0.1, 0.0]), step_pos(circ, device, [0.1, 0.0]), atol=1e-14
    )


@pytest.mark.parametrize("device", DEVICES)
def test_no_activation_outside_reach(device):
    st = stepper_with_obstacle(device, ObstacleShape.BOX, [0.0, 0.0], half=(0.1, 0.1))
    p = [0.5, 0.5]  # far outside the contact reach
    np.testing.assert_allclose(step_pos(st, device, p), p, atol=1e-14)


@pytest.mark.parametrize("device", DEVICES)
def test_per_env_box_angle(device):
    """``angle`` may be per-env ``[n_envs, n_obs]``: the same box at the same centre
    rotated differently in each env produces each env's own analytic contact."""
    angles = [0.0, np.pi / 4.0]
    center, half, p = [0.0, 0.0], (0.1, 0.2), [0.16, 0.0]
    world = WorldConfig(collision_k=K, collision_c=0.0, collision_margin=0.02)
    st = Stepper(holo(1, 0.1), dt=DT, device=device, dtype=wp.float64, world=world)
    st.set_obstacles(
        Obstacles(
            pos=torch.tensor([[center], [center]], dtype=torch.float64, device=device),
            radius=torch.zeros(1, dtype=torch.float64, device=device),  # a box ignores radius
            shape=torch.tensor([int(ObstacleShape.BOX)], dtype=torch.int32, device=device),
            angle=torch.tensor([[angles[0]], [angles[1]]], dtype=torch.float64, device=device),
            half_extents=torch.tensor([half], dtype=torch.float64, device=device),
        )
    )
    state = _map5(
        TorchState(
            pos=torch.tensor([[p], [p]], dtype=torch.float64),
            theta=torch.zeros(2, 1, dtype=torch.float64),
            vel=torch.zeros(2, 1, 2, dtype=torch.float64),
            speed=torch.zeros(2, 1, dtype=torch.float64),
            ang_vel=torch.zeros(2, 1, dtype=torch.float64),
        ),
        lambda t: t.to(device),
    )
    with torch.no_grad():
        out = warp_step(st, state, torch.zeros(2, 1, 2, dtype=torch.float64, device=device))

    for e, angle in enumerate(angles):
        np.testing.assert_allclose(
            out.pos[e, 0].cpu().numpy(), expect_box(p, center, angle, half, 0.12), atol=1e-12
        )
    # the two envs really did see different geometry
    assert not np.allclose(out.pos[0, 0].cpu().numpy(), out.pos[1, 0].cpu().numpy())


@pytest.mark.parametrize("device", DEVICES)
def test_shared_angle_broadcasts_across_envs(device):
    """A 1D ``[n_obs]`` angle is shared by every env (the pre-existing contract)."""
    angle, center, half, p = 0.3, [0.0, 0.0], (0.1, 0.2), [0.16, 0.0]
    world = WorldConfig(collision_k=K, collision_c=0.0, collision_margin=0.02)
    st = Stepper(holo(1, 0.1), dt=DT, device=device, dtype=wp.float64, world=world)
    st.set_obstacles(
        Obstacles(
            pos=torch.tensor([[center], [center]], dtype=torch.float64, device=device),
            radius=torch.zeros(1, dtype=torch.float64, device=device),
            shape=torch.tensor([int(ObstacleShape.BOX)], dtype=torch.int32, device=device),
            angle=torch.tensor([angle], dtype=torch.float64, device=device),  # 1D
            half_extents=torch.tensor([half], dtype=torch.float64, device=device),
        )
    )
    state = _map5(
        TorchState(
            pos=torch.tensor([[p], [p]], dtype=torch.float64),
            theta=torch.zeros(2, 1, dtype=torch.float64),
            vel=torch.zeros(2, 1, 2, dtype=torch.float64),
            speed=torch.zeros(2, 1, dtype=torch.float64),
            ang_vel=torch.zeros(2, 1, dtype=torch.float64),
        ),
        lambda t: t.to(device),
    )
    with torch.no_grad():
        out = warp_step(st, state, torch.zeros(2, 1, 2, dtype=torch.float64, device=device))
    want = expect_box(p, center, angle, half, 0.12)
    for e in range(2):
        np.testing.assert_allclose(out.pos[e, 0].cpu().numpy(), want, atol=1e-12)


def test_gradcheck_box_and_segment():
    """Full pipeline is differentiable with box + segment obstacles (f64 CPU,
    contact comfortably in the exterior region, away from the corner kink)."""
    world = WorldConfig(collision_k=10.0, collision_c=1.0, collision_margin=0.02)
    st = Stepper(holo(1, 0.1), dt=0.1, substeps=2, device="cpu", dtype=wp.float64, world=world)
    st.set_obstacles(
        Obstacles(
            pos=torch.tensor([[[0.22, 0.0], [0.0, -0.2]]], dtype=torch.float64),
            radius=torch.tensor([0.0, 0.05], dtype=torch.float64),
            shape=torch.tensor(
                [int(ObstacleShape.BOX), int(ObstacleShape.SEGMENT)], dtype=torch.int32
            ),
            angle=torch.tensor([0.0, 0.0], dtype=torch.float64),
            half_extents=torch.tensor([[0.1, 0.1], [0.2, 0.0]], dtype=torch.float64),
        )
    )
    state = _map5(make_state([[0.05, -0.05]]), lambda t: t.requires_grad_(True))
    actions = (
        0.05 * torch.randn(1, 1, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    ).requires_grad_(True)

    def fn(pos, theta, vel, speed, ang_vel, act):
        return tuple(warp_step(st, TorchState(pos, theta, vel, speed, ang_vel), act))[:5]

    assert torch.autograd.gradcheck(fn, (*_core(state), actions), eps=1e-6, atol=1e-5)
