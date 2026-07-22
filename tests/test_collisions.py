"""Soft-collision (spring-damper) forces and world bounds, with analytic references."""

import numpy as np
import pytest
import torch
import warp as wp

from wmas.core.config import WorldConfig
from wmas.core.stepper import Stepper
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from wmas.interop.autograd import TorchState, warp_step

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def make_state(pos, vel=None, dtype=torch.float64):
    pos = torch.as_tensor(pos, dtype=dtype).unsqueeze(0)  # [1, n_agents, 2]
    n = pos.shape[1]
    z = torch.zeros(1, n, dtype=dtype)
    v = torch.zeros_like(pos) if vel is None else torch.as_tensor(vel, dtype=dtype).unsqueeze(0)
    return TorchState(pos=pos, theta=z.clone(), vel=v, speed=z.clone(), ang_vel=z.clone())


def holo_cfgs(n, radius=0.1, mode=ControlMode.VELOCITY):
    return [
        AgentConfig(model=DynamicsModel.HOLONOMIC, ctrl_mode=mode, radius=radius,
                    max_speed=100.0, max_accel=100.0)
        for _ in range(n)
    ]


@pytest.mark.parametrize("device", DEVICES)
def test_two_agent_spring_analytic(device):
    """Pure spring, no damping: hand-computed post-step positions."""
    k, margin, dt = 100.0, 0.02, 0.1
    world = WorldConfig(collision_k=k, collision_c=0.0, collision_margin=margin)
    stepper = Stepper(holo_cfgs(2), dt=dt, device=device, dtype=wp.float64, world=world)
    state = make_state([[0.0, 0.0], [0.15, 0.0]])
    actions = torch.zeros(1, 2, 2, dtype=torch.float64)
    with torch.no_grad():
        out = warp_step(stepper, TorchState(*(t.to(device) for t in state)), actions.to(device))
    # overlap = 0.1 + 0.1 + 0.02 - 0.15 = 0.07 ; |F| = k * overlap = 7
    # velocity mode, zero action: v_new = F/m * dt ; pos += v_new * dt
    np.testing.assert_allclose(out.pos[0, 0].cpu(), [-0.07, 0.0], atol=1e-12)
    np.testing.assert_allclose(out.pos[0, 1].cpu(), [0.22, 0.0], atol=1e-12)
    np.testing.assert_allclose(out.vel[0, 0].cpu(), [-0.7, 0.0], atol=1e-12)


def test_two_agent_spring_damper_analytic():
    """Spring + damping with approaching agents, acceleration mode."""
    k, c, margin, dt = 100.0, 2.0, 0.02, 0.1
    world = WorldConfig(collision_k=k, collision_c=c, collision_margin=margin)
    stepper = Stepper(
        holo_cfgs(2, mode=ControlMode.ACCELERATION), dt=dt, device="cpu",
        dtype=wp.float64, world=world,
    )
    state = make_state([[0.0, 0.0], [0.15, 0.0]], vel=[[0.5, 0.0], [-0.5, 0.0]])
    actions = torch.zeros(1, 2, 2, dtype=torch.float64)
    with torch.no_grad():
        out = warp_step(stepper, state, actions)
    # agent0: n = (-1,0), overlap 0.07, rel_v = (1,0), dot(rel_v,n) = -1
    # F0 = k*0.07*(-1,0) - c*(-1)*(-1,0) = (-7-2, 0) = (-9, 0)
    # accel mode, zero action: v_new = v + F/m*dt = 0.5 - 0.9 = -0.4
    np.testing.assert_allclose(out.vel[0, 0].cpu(), [-0.4, 0.0], atol=1e-12)
    np.testing.assert_allclose(out.pos[0, 0].cpu(), [-0.04, 0.0], atol=1e-12)
    np.testing.assert_allclose(out.vel[0, 1].cpu(), [0.4, 0.0], atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_momentum_symmetry(device):
    """Equal agents, pure spring: center of mass stays put over many steps."""
    world = WorldConfig(collision_k=50.0, collision_c=0.5, collision_margin=0.05)
    stepper = Stepper(
        holo_cfgs(4, mode=ControlMode.ACCELERATION), dt=0.05, substeps=2,
        device=device, dtype=wp.float64, world=world,
    )
    rng = np.random.default_rng(0)
    pos0 = rng.random((4, 2)) * 0.2  # cramped -> collisions
    state = TorchState(*(t.to(device) for t in make_state(pos0)))
    actions = torch.zeros(1, 4, 2, dtype=torch.float64, device=device)
    com0 = state.pos.mean(dim=1)
    with torch.no_grad():
        for _ in range(20):
            state = warp_step(stepper, state, actions)
    torch.testing.assert_close(state.pos.mean(dim=1), com0, atol=1e-10, rtol=0)
    # and they were actually pushed apart
    d01 = (state.pos[0, 0] - state.pos[0, 1]).norm()
    assert d01 > np.linalg.norm(pos0[0] - pos0[1])


@pytest.mark.parametrize("device", DEVICES)
def test_obstacle_repulsion_analytic(device):
    k, margin, dt = 100.0, 0.02, 0.1
    world = WorldConfig(collision_k=k, collision_c=0.0, collision_margin=margin)
    stepper = Stepper(holo_cfgs(1), dt=dt, device=device, dtype=wp.float64, world=world)
    stepper.set_obstacles(
        pos=torch.tensor([[[0.3, 0.0]]], dtype=torch.float64, device=device),
        radius=torch.tensor([0.15], dtype=torch.float64, device=device),
    )
    state = TorchState(*(t.to(device) for t in make_state([[0.1, 0.0]])))
    actions = torch.zeros(1, 1, 2, dtype=torch.float64, device=device)
    with torch.no_grad():
        out = warp_step(stepper, state, actions)
    # dist = 0.2, min_dist = 0.1+0.15+0.02 = 0.27, overlap = 0.07, n = (-1,0)
    np.testing.assert_allclose(out.pos[0, 0].cpu(), [0.1 - 0.07, 0.0], atol=1e-12)


def test_soft_wall_analytic():
    k, margin, dt = 100.0, 0.02, 0.1
    world = WorldConfig(
        collision_k=k, collision_c=0.0, collision_margin=margin,
        bounds=(-1.0, 1.0, -1.0, 1.0), bounds_mode="soft",
    )
    stepper = Stepper(
        holo_cfgs(1, radius=0.05), dt=dt, device="cpu", dtype=wp.float64, world=world
    )
    state = make_state([[0.95, 0.0]])
    actions = torch.zeros(1, 1, 2, dtype=torch.float64)
    with torch.no_grad():
        out = warp_step(stepper, state, actions)
    # wall dist = 0.05, overlap = 0.05+0.02-0.05 = 0.02, F = (-2, 0)
    np.testing.assert_allclose(out.pos[0, 0].cpu(), [0.95 - 0.02, 0.0], atol=1e-12)


def test_clamp_bounds():
    world = WorldConfig(bounds=(-1.0, 1.0, -1.0, 1.0), bounds_mode="clamp")
    stepper = Stepper(
        holo_cfgs(1, radius=0.05), dt=0.1, device="cpu", dtype=wp.float64, world=world
    )
    state = make_state([[0.98, 0.0]])
    actions = torch.tensor([[[5.0, 0.0]]], dtype=torch.float64)  # clamped to max_speed=100
    with torch.no_grad():
        out = warp_step(stepper, state, actions)
    np.testing.assert_allclose(out.pos[0, 0], [1.0 - 0.05, 0.0], atol=1e-12)


def test_neighbor_radius_validation():
    with pytest.raises(ValueError, match="neighbor_radius"):
        Stepper(
            holo_cfgs(2, radius=0.2), dt=0.1, device="cpu",
            world=WorldConfig(neighbor_radius=0.1),
        )


def test_gradcheck_with_collisions():
    """Full pipeline (neighbors -> forces -> integrate) is differentiable."""
    world = WorldConfig(
        collision_k=10.0, collision_c=1.0, collision_margin=0.02,
        bounds=(-1.0, 1.0, -1.0, 1.0), bounds_mode="soft",
    )
    stepper = Stepper(
        holo_cfgs(2, mode=ControlMode.ACCELERATION), dt=0.1, substeps=2,
        device="cpu", dtype=wp.float64, world=world,
    )
    # overlapping pair, comfortably inside contact (overlap ~0.04 >> fd eps)
    state = make_state([[0.0, 0.0], [0.18, 0.02]], vel=[[0.1, 0.0], [-0.1, 0.05]])
    state = TorchState(*(t.requires_grad_(True) for t in state))
    actions = (0.1 * torch.randn(1, 2, 2, dtype=torch.float64,
                                 generator=torch.Generator().manual_seed(0))).requires_grad_(True)

    def fn(pos, theta, vel, speed, ang_vel, act):
        return tuple(warp_step(stepper, TorchState(pos, theta, vel, speed, ang_vel), act))

    assert torch.autograd.gradcheck(fn, (*state, actions), eps=1e-6, atol=1e-5)


def test_grad_and_nograd_forward_match_with_collisions():
    world = WorldConfig(collision_k=30.0, collision_c=1.0, collision_margin=0.05)
    stepper = Stepper(
        holo_cfgs(3, mode=ControlMode.ACCELERATION), dt=0.05, substeps=3,
        device="cpu", dtype=wp.float64, world=world,
    )
    rng = np.random.default_rng(1)
    state = make_state(rng.random((3, 2)) * 0.25)
    actions = torch.zeros(1, 3, 2, dtype=torch.float64)
    grad_in = TorchState(*(t.clone().requires_grad_(True) for t in state))
    out_grad = warp_step(stepper, grad_in, actions.clone().requires_grad_(True))
    with torch.no_grad():
        out_fast = warp_step(stepper, state, actions)
    for a, b in zip(out_grad, out_fast, strict=True):
        torch.testing.assert_close(a.detach(), b)
