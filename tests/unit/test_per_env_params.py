"""Per-env parameter randomization: the opt-in ``[n_envs, n_agents, NUM_PARAMS]``
layout and its dedicated kernel variants.

References are hand-computed in float64 on CPU, in the same spirit as
``test_dynamics``/``test_collisions``. The per-env path must (a) reproduce the
shared-param path bit-for-bit when the per-env values are broadcast-equal across
envs, and (b) stay fully differentiable (f64 gradcheck).
"""

import numpy as np
import pytest
import torch
import warp as wp
from conftest import DEVICES, _core, _map5, holo_cfgs

from wmas.core.config import Obstacles, WorldConfig
from wmas.core.stepper import Stepper
from wmas.dynamics.base import (
    NUM_PARAMS,
    P_MASS,
    P_RADIUS,
    AgentConfig,
    ControlMode,
    DynamicsModel,
    per_env_float_template,
)
from wmas.interop.autograd import TorchState, warp_step


def make_state(pos, vel=None, n_envs=1, dtype=torch.float64):
    pos = torch.as_tensor(pos, dtype=dtype)
    if pos.dim() == 2:
        pos = pos.unsqueeze(0).expand(n_envs, *pos.shape).contiguous()
    n = pos.shape[1]
    z = torch.zeros(n_envs, n, dtype=dtype)
    if vel is None:
        v = torch.zeros_like(pos)
    else:
        v = torch.as_tensor(vel, dtype=dtype)
        if v.dim() == 2:
            v = v.unsqueeze(0).expand(n_envs, *v.shape).contiguous()
    return TorchState(pos=pos, theta=z.clone(), vel=v, speed=z.clone(), ang_vel=z.clone())


def test_template_shape_and_values():
    cfgs = holo_cfgs(3, radius=0.07)
    tmpl = per_env_float_template(cfgs, n_envs=5)
    assert tmpl.shape == (5, 3, NUM_PARAMS)
    # every env is a copy of the shared per-agent rows
    for e in range(5):
        np.testing.assert_array_equal(tmpl[e], tmpl[0])
    np.testing.assert_allclose(tmpl[..., P_RADIUS], 0.07)


@pytest.mark.parametrize("device", DEVICES)
def test_per_env_mass_divergence_analytic(device):
    """Two envs, identical state/actions/obstacle, different mass -> the force
    response ``dv = F/m * dt`` scales inversely with mass (hand-computed)."""
    k, margin, dt = 100.0, 0.02, 0.1
    world = WorldConfig(collision_k=k, collision_c=0.0, collision_margin=margin)
    cfgs = holo_cfgs(1, radius=0.1)
    stepper = Stepper(cfgs, dt=dt, device=device, dtype=wp.float64, world=world)
    stepper.set_obstacles(
        Obstacles(
            pos=torch.tensor([[[0.3, 0.0]], [[0.3, 0.0]]], dtype=torch.float64, device=device),
            radius=torch.tensor([0.15], dtype=torch.float64, device=device),
        )
    )
    floats = per_env_float_template(cfgs, n_envs=2)
    floats[0, 0, P_MASS] = 1.0
    floats[1, 0, P_MASS] = 2.0
    stepper.set_agent_params_per_env(torch.as_tensor(floats, device=device))

    state = _map5(make_state([[0.1, 0.0]], n_envs=2), lambda t: t.to(device))
    actions = torch.zeros(2, 1, 2, dtype=torch.float64, device=device)
    with torch.no_grad():
        out = warp_step(stepper, state, actions)
    # dist 0.2, min_dist 0.27, overlap 0.07, |F| = k*overlap = 7 along (-1, 0).
    # velocity mode, zero action: v_new = F/m*dt ; pos += v_new*dt
    np.testing.assert_allclose(out.pos[0, 0].cpu(), [0.1 - 0.07, 0.0], atol=1e-12)  # m=1
    np.testing.assert_allclose(out.pos[1, 0].cpu(), [0.1 - 0.035, 0.0], atol=1e-12)  # m=2


@pytest.mark.parametrize("device", DEVICES)
def test_per_env_radius_changes_contact_analytic(device):
    """Different per-env radius changes the contact distance ``r_i + r_j``: the
    smaller-radius env is out of contact, the larger-radius env overlaps."""
    k, margin, dt = 100.0, 0.02, 0.1
    world = WorldConfig(collision_k=k, collision_c=0.0, collision_margin=margin)
    cfgs = holo_cfgs(2, radius=0.15)  # config radius = the max -> neighbor_radius covers it
    stepper = Stepper(cfgs, dt=dt, device=device, dtype=wp.float64, world=world)
    floats = per_env_float_template(cfgs, n_envs=2)
    floats[0, :, P_RADIUS] = 0.1
    floats[1, :, P_RADIUS] = 0.15
    stepper.set_agent_params_per_env(torch.as_tensor(floats, device=device))

    state = _map5(make_state([[0.0, 0.0], [0.25, 0.0]], n_envs=2), lambda t: t.to(device))
    actions = torch.zeros(2, 2, 2, dtype=torch.float64, device=device)
    with torch.no_grad():
        out = warp_step(stepper, state, actions)
    # env0: min_dist = 0.1+0.1+0.02 = 0.22 < 0.25 -> no contact, agents frozen
    np.testing.assert_allclose(out.pos[0, 0].cpu(), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out.pos[0, 1].cpu(), [0.25, 0.0], atol=1e-12)
    # env1: min_dist = 0.32 > 0.25 -> overlap 0.07, |F| = 7, v = -0.7, pos -= 0.07
    np.testing.assert_allclose(out.pos[1, 0].cpu(), [-0.07, 0.0], atol=1e-12)
    np.testing.assert_allclose(out.pos[1, 1].cpu(), [0.32, 0.0], atol=1e-12)


def _mixed_stepper(world, substeps=1):
    cfgs = [
        AgentConfig(model=DynamicsModel.HOLONOMIC, radius=0.1, max_speed=2.0, max_accel=2.0),
        AgentConfig(model=DynamicsModel.DIFF_DRIVE, radius=0.1, max_speed=2.0, max_ang_vel=2.0),
        AgentConfig(
            model=DynamicsModel.KINEMATIC_BICYCLE,
            radius=0.1,
            max_speed=2.0,
            max_accel=2.0,
            max_steer=0.6,
            l_f=0.16,
            l_r=0.14,
        ),
    ]
    stepper = Stepper(cfgs, dt=0.05, substeps=substeps, device="cpu", dtype=wp.float64, world=world)
    return cfgs, stepper


def test_per_env_broadcast_matches_shared_bit_exact():
    """Broadcasting identical params across envs must reproduce the shared-param
    path bit-for-bit through the full pipeline (neighbors, obstacles, walls)."""
    world = WorldConfig(
        collision_k=30.0,
        collision_c=1.0,
        collision_margin=0.05,
        bounds=(-1.0, 1.0, -1.0, 1.0),
        bounds_mode="soft",
    )
    cfgs, shared = _mixed_stepper(world, substeps=3)
    per_env = Stepper(cfgs, dt=0.05, substeps=3, device="cpu", dtype=wp.float64, world=world)
    for s in (shared, per_env):
        s.set_obstacles(
            Obstacles(
                pos=torch.tensor([[[0.4, 0.4]]] * 4, dtype=torch.float64),
                radius=torch.tensor([0.12], dtype=torch.float64),
            )
        )
    per_env.set_agent_params_per_env(torch.as_tensor(per_env_float_template(cfgs, n_envs=4)))

    rng = np.random.default_rng(0)
    state = make_state(rng.random((3, 2)) * 0.3 - 0.15, n_envs=4)
    actions = torch.as_tensor(0.4 * rng.standard_normal((4, 3, 2)), dtype=torch.float64)
    with torch.no_grad():
        out_shared = warp_step(stepper=shared, state=state, actions=actions)
        out_pe = warp_step(stepper=per_env, state=state, actions=actions)
    for a, b in zip(out_shared, out_pe, strict=True):
        assert torch.equal(a, b)  # bit-exact


def test_gradcheck_per_env_with_collisions():
    """Adjoints of the per-env kernel variants are correct (f64 CPU gradcheck)."""
    world = WorldConfig(
        collision_k=10.0,
        collision_c=1.0,
        collision_margin=0.02,
        bounds=(-1.0, 1.0, -1.0, 1.0),
        bounds_mode="soft",
    )
    cfgs = holo_cfgs(2, radius=0.12, mode=ControlMode.ACCELERATION)
    stepper = Stepper(cfgs, dt=0.1, substeps=2, device="cpu", dtype=wp.float64, world=world)
    floats = per_env_float_template(cfgs, n_envs=2)
    floats[0, :, P_MASS], floats[1, :, P_MASS] = 1.0, 1.7
    floats[0, :, P_RADIUS], floats[1, :, P_RADIUS] = 0.1, 0.12
    stepper.set_agent_params_per_env(torch.as_tensor(floats))

    state = make_state([[0.0, 0.0], [0.18, 0.02]], vel=[[0.1, 0.0], [-0.1, 0.05]], n_envs=2)
    state = _map5(state, lambda t: t.requires_grad_(True))
    actions = (
        0.1 * torch.randn(2, 2, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    ).requires_grad_(True)

    def fn(pos, theta, vel, speed, ang_vel, act):
        return tuple(warp_step(stepper, TorchState(pos, theta, vel, speed, ang_vel), act))[:5]

    assert torch.autograd.gradcheck(fn, (*_core(state), actions), eps=1e-6, atol=1e-5)


def test_per_env_no_grad_matches_grad():
    """The tape-free fast path and the autograd path agree on the per-env path."""
    world = WorldConfig(collision_k=30.0, collision_c=1.0, collision_margin=0.05)
    cfgs = holo_cfgs(3, radius=0.1, mode=ControlMode.ACCELERATION)
    stepper = Stepper(cfgs, dt=0.05, substeps=3, device="cpu", dtype=wp.float64, world=world)
    floats = per_env_float_template(cfgs, n_envs=2)
    floats[1, :, P_MASS] = 2.5
    stepper.set_agent_params_per_env(torch.as_tensor(floats))

    rng = np.random.default_rng(1)
    state = make_state(rng.random((3, 2)) * 0.25, n_envs=2)
    actions = torch.zeros(2, 3, 2, dtype=torch.float64)
    grad_in = _map5(state, lambda t: t.clone().requires_grad_(True))
    out_grad = warp_step(stepper, grad_in, actions.clone().requires_grad_(True))
    with torch.no_grad():
        out_fast = warp_step(stepper, state, actions)
    for a, b in zip(out_grad, out_fast, strict=True):
        torch.testing.assert_close(a.detach(), b)


def test_set_per_env_wrong_shape_raises():
    cfgs = holo_cfgs(2)
    stepper = Stepper(cfgs, dt=0.1, device="cpu", dtype=wp.float64)
    with pytest.raises(ValueError, match="n_agents|NUM_PARAMS|shape"):
        stepper.set_agent_params_per_env(torch.zeros(4, 3, NUM_PARAMS, dtype=torch.float64))
    with pytest.raises(ValueError, match="n_agents|NUM_PARAMS|shape"):
        stepper.set_agent_params_per_env(torch.zeros(4, 2, NUM_PARAMS + 1, dtype=torch.float64))


def test_set_per_env_radius_exceeds_neighbor_reach_raises():
    """A per-env radius larger than what the neighbor grid was sized for would
    silently miss contacts -> must be rejected."""
    world = WorldConfig(collision_k=10.0, collision_margin=0.02)
    cfgs = holo_cfgs(2, radius=0.1)  # neighbor_radius sized for r=0.1
    stepper = Stepper(cfgs, dt=0.1, device="cpu", dtype=wp.float64, world=world)
    floats = per_env_float_template(cfgs, n_envs=2)
    floats[1, :, P_RADIUS] = 0.5  # far too big for the sized grid
    with pytest.raises(ValueError, match="neighbor_radius"):
        stepper.set_agent_params_per_env(torch.as_tensor(floats))
