"""Gradient correctness of the Warp adjoint step exposed through torch.autograd.

Finite-difference gradcheck runs on the CPU device in float64. Limits are set
far from the action values so clamp kinks don't sit under the FD perturbation.
"""

import numpy as np
import pytest
import torch
import warp as wp

from wmas.core.config import WorldConfig
from wmas.core.stepper import Stepper
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel, Integrator
from wmas.interop.autograd import TorchState, rollout, warp_step

BIG = 100.0  # limits far away from any test action


def make_stepper(cfgs, dt=0.1, substeps=1, device="cpu", dtype=wp.float64):
    return Stepper(configs=cfgs, dt=dt, substeps=substeps, device=device, dtype=dtype)


def make_state(n_envs, n_agents, device="cpu", dtype=torch.float64, requires_grad=False, seed=0):
    g = torch.Generator().manual_seed(seed)

    def t(*shape):
        x = torch.randn(*shape, generator=g, dtype=dtype).to(device) * 0.5
        return x.requires_grad_(requires_grad)

    return TorchState(
        pos=t(n_envs, n_agents, 2),
        theta=t(n_envs, n_agents),
        vel=t(n_envs, n_agents, 2),
        speed=t(n_envs, n_agents),
        ang_vel=t(n_envs, n_agents),
    )


MODEL_CASES = [
    ("holonomic-vel", DynamicsModel.HOLONOMIC, ControlMode.VELOCITY),
    ("holonomic-acc", DynamicsModel.HOLONOMIC, ControlMode.ACCELERATION),
    ("diffdrive-vel", DynamicsModel.DIFF_DRIVE, ControlMode.VELOCITY),
    ("diffdrive-acc", DynamicsModel.DIFF_DRIVE, ControlMode.ACCELERATION),
    ("bicycle", DynamicsModel.KINEMATIC_BICYCLE, ControlMode.ACCELERATION),
]


@pytest.mark.parametrize("name,model,mode", MODEL_CASES, ids=[c[0] for c in MODEL_CASES])
def test_gradcheck_single_step(name, model, mode):
    cfgs = [
        AgentConfig(
            model=model,
            ctrl_mode=mode,
            max_speed=BIG,
            max_accel=BIG,
            max_ang_vel=BIG,
            max_ang_accel=BIG,
            max_steer=1.0,
            l_f=0.16,
            l_r=0.14,
        )
        for _ in range(2)
    ]
    stepper = make_stepper(cfgs)
    state = make_state(2, 2, requires_grad=True, seed=1)
    actions = (0.3 * torch.randn(2, 2, 2, dtype=torch.float64)).requires_grad_(True)

    def fn(pos, theta, vel, speed, ang_vel, act):
        out = warp_step(stepper, TorchState(pos, theta, vel, speed, ang_vel), act)
        return tuple(out)

    assert torch.autograd.gradcheck(fn, (*state, actions), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("name,model,mode", MODEL_CASES, ids=[c[0] for c in MODEL_CASES])
def test_gradcheck_single_step_rk4(name, model, mode):
    """The RK4 path (four _deriv evaluations) is differentiable end-to-end."""
    cfgs = [
        AgentConfig(
            model=model,
            ctrl_mode=mode,
            max_speed=BIG,
            max_accel=BIG,
            max_ang_vel=BIG,
            max_ang_accel=BIG,
            max_steer=1.0,
            l_f=0.16,
            l_r=0.14,
        )
        for _ in range(2)
    ]
    stepper = Stepper(
        configs=cfgs,
        dt=0.1,
        device="cpu",
        dtype=wp.float64,
        world=WorldConfig(collisions=False, integrator=Integrator.RK4),
    )
    state = make_state(2, 2, requires_grad=True, seed=1)
    actions = (0.3 * torch.randn(2, 2, 2, dtype=torch.float64)).requires_grad_(True)

    def fn(pos, theta, vel, speed, ang_vel, act):
        out = warp_step(stepper, TorchState(pos, theta, vel, speed, ang_vel), act)
        return tuple(out)

    assert torch.autograd.gradcheck(fn, (*state, actions), eps=1e-6, atol=1e-5)


def test_action_dim_gt_2_ignored_and_gradchecks():
    """Actions may carry more than 2 slots; models read only slots 0,1, so the
    extra slot has exactly zero gradient and the step matches the 2-slot case."""
    cfgs = [AgentConfig(model=DynamicsModel.HOLONOMIC, max_speed=BIG, max_accel=BIG)]
    stepper = make_stepper(cfgs)
    state = make_state(1, 1, requires_grad=True, seed=7)

    a2 = 0.3 * torch.randn(1, 1, 2, dtype=torch.float64)
    a3 = torch.cat([a2, torch.full((1, 1, 1), 5.0, dtype=torch.float64)], dim=-1)
    out2 = warp_step(stepper, state, a2)
    out3 = warp_step(stepper, state, a3)
    # the padded slot must not change the dynamics
    for x2, x3 in zip(out2, out3, strict=True):
        torch.testing.assert_close(x2, x3)

    a3 = a3.requires_grad_(True)
    warp_step(stepper, state, a3).pos.sum().backward()
    assert a3.grad is not None
    torch.testing.assert_close(a3.grad[..., 2], torch.zeros(1, 1, dtype=torch.float64))

    def fn(pos, theta, vel, speed, ang_vel, act):
        return tuple(warp_step(stepper, TorchState(pos, theta, vel, speed, ang_vel), act))

    acts = (0.3 * torch.randn(1, 1, 3, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(fn, (*state, acts), eps=1e-6, atol=1e-5)


def test_gradcheck_substeps():
    cfgs = [AgentConfig(model=DynamicsModel.DIFF_DRIVE, max_speed=BIG, max_ang_vel=BIG)]
    stepper = make_stepper(cfgs, substeps=4)
    state = make_state(1, 1, requires_grad=True, seed=2)
    actions = (0.3 * torch.randn(1, 1, 2, dtype=torch.float64)).requires_grad_(True)

    def fn(pos, theta, vel, speed, ang_vel, act):
        return tuple(warp_step(stepper, TorchState(pos, theta, vel, speed, ang_vel), act))

    assert torch.autograd.gradcheck(fn, (*state, actions), eps=1e-6, atol=1e-5)


def test_gradcheck_multistep_rollout():
    """BPTT over a 5-step rollout: grads w.r.t. the full action sequence and initial state."""
    cfgs = [
        AgentConfig(
            model=DynamicsModel.KINEMATIC_BICYCLE,
            max_speed=BIG,
            max_accel=BIG,
            max_steer=1.0,
            l_f=0.16,
            l_r=0.14,
        ),
        AgentConfig(model=DynamicsModel.DIFF_DRIVE, max_speed=BIG, max_ang_vel=BIG),
    ]
    stepper = make_stepper(cfgs)
    state = make_state(1, 2, requires_grad=True, seed=3)
    actions_seq = (0.3 * torch.randn(5, 1, 2, 2, dtype=torch.float64)).requires_grad_(True)

    def fn(pos, theta, vel, speed, ang_vel, acts):
        final, _ = rollout(stepper, TorchState(pos, theta, vel, speed, ang_vel), acts)
        return final.pos, final.theta

    assert torch.autograd.gradcheck(fn, (*state, actions_seq), eps=1e-6, atol=1e-5)


def test_bptt_analytic_holonomic():
    """Holonomic velocity mode, no clamping: d x_T / d a_t = dt exactly, for every t < T."""
    T, dt = 6, 0.1
    cfgs = [AgentConfig(model=DynamicsModel.HOLONOMIC, max_speed=BIG)]
    stepper = make_stepper(cfgs, dt=dt)
    state = make_state(1, 1, seed=4)
    actions_seq = (0.2 * torch.randn(T, 1, 1, 2, dtype=torch.float64)).requires_grad_(True)

    final, _ = rollout(stepper, state, actions_seq)
    final.pos[0, 0, 0].backward()
    grad = actions_seq.grad
    expected = torch.zeros_like(grad)
    expected[:, 0, 0, 0] = dt  # x-component of every action moves x_T by dt
    torch.testing.assert_close(grad, expected)


def test_bptt_analytic_holonomic_accel():
    """Acceleration mode, 2 steps from known state: d x_2 / d a_0x = 2*dt^2."""
    dt = 0.1
    cfgs = [
        AgentConfig(
            model=DynamicsModel.HOLONOMIC,
            ctrl_mode=ControlMode.ACCELERATION,
            max_speed=BIG,
            max_accel=BIG,
        )
    ]
    stepper = make_stepper(cfgs, dt=dt)
    state = make_state(1, 1, seed=5)
    actions_seq = (0.2 * torch.randn(2, 1, 1, 2, dtype=torch.float64)).requires_grad_(True)

    final, _ = rollout(stepper, state, actions_seq)
    final.pos[0, 0, 0].backward()
    torch.testing.assert_close(
        actions_seq.grad[0, 0, 0, 0], torch.tensor(2 * dt * dt, dtype=torch.float64)
    )


def test_grads_finite_when_clamped():
    """Saturated actions must still yield finite (zero) gradients, not NaNs."""
    cfgs = [AgentConfig(model=DynamicsModel.DIFF_DRIVE, max_speed=1.0, max_ang_vel=1.0)]
    stepper = make_stepper(cfgs)
    state = make_state(1, 1, seed=6)
    actions = torch.full((1, 1, 2), 50.0, dtype=torch.float64, requires_grad=True)
    out = warp_step(stepper, state, actions)
    out.pos.sum().backward()
    assert torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() == 0.0  # fully saturated -> zero grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_float32_matches_cpu_float64():
    cfgs = [
        AgentConfig(
            model=m,
            max_speed=BIG,
            max_accel=BIG,
            max_ang_vel=BIG,
            max_steer=1.0,
            l_f=0.16,
            l_r=0.14,
        )
        for m in (
            DynamicsModel.HOLONOMIC,
            DynamicsModel.DIFF_DRIVE,
            DynamicsModel.KINEMATIC_BICYCLE,
        )
    ]
    T = 4

    def run(device, tdtype, wdtype):
        stepper = make_stepper(cfgs, substeps=2, device=device, dtype=wdtype)
        state = TorchState(*(t.to(device=device, dtype=tdtype) for t in make_state(3, 3, seed=7)))
        gen = torch.Generator().manual_seed(8)
        acts = 0.3 * torch.randn(T, 3, 3, 2, dtype=torch.float64, generator=gen)
        acts = acts.to(device=device, dtype=tdtype).requires_grad_(True)
        final, _ = rollout(stepper, state, acts)
        (final.pos.square().sum() + final.theta.square().sum()).backward()
        return acts.grad.double().cpu()

    g_cpu = run("cpu", torch.float64, wp.float64)
    g_gpu = run("cuda:0", torch.float32, wp.float32)
    assert torch.isfinite(g_gpu).all()
    torch.testing.assert_close(g_gpu, g_cpu, rtol=1e-3, atol=1e-4)


def test_no_grad_path_gives_same_forward():
    """The tape-free fast path must produce the same next state as the autograd path."""
    cfgs = [AgentConfig(model=DynamicsModel.DIFF_DRIVE, max_speed=2.0, max_ang_vel=2.0)]
    stepper = make_stepper(cfgs, substeps=3)
    state = make_state(2, 1, seed=9)
    actions = 0.5 * torch.randn(2, 1, 2, dtype=torch.float64)

    grad_state = TorchState(*(t.clone().requires_grad_(True) for t in state))
    out_grad = warp_step(stepper, grad_state, actions.clone().requires_grad_(True))
    with torch.no_grad():
        out_fast = warp_step(stepper, state, actions)
    for a, b in zip(out_grad, out_fast, strict=True):
        torch.testing.assert_close(a.detach(), b)
    np.testing.assert_equal(out_fast.pos.requires_grad, False)
