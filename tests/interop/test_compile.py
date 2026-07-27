"""torch.compile-compatible custom-op step: parity, gradients, and a compiled run."""

import pytest
import torch
import warp as wp

from wmas.core.config import WorldConfig
from wmas.core.stepper import Stepper
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from wmas.interop.autograd import TorchState, warp_step
from wmas.interop.compile import CudaGraphStep, compiled_warp_step

BIG = 100.0


def _stepper(dtype=wp.float64, substeps=1, device="cpu", world=None):
    cfgs = [
        AgentConfig(
            model=DynamicsModel.DIFF_DRIVE,
            ctrl_mode=ControlMode.ACCELERATION,
            max_speed=BIG,
            max_accel=BIG,
            max_ang_vel=BIG,
            max_ang_accel=BIG,
        )
        for _ in range(2)
    ]
    return Stepper(cfgs, dt=0.1, substeps=substeps, device=device, dtype=dtype, world=world)


def _state(n_envs=2, n_agents=2, requires_grad=False):
    g = torch.Generator().manual_seed(0)

    def t(*s):
        return (0.4 * torch.randn(*s, generator=g, dtype=torch.float64)).requires_grad_(
            requires_grad
        )

    return TorchState(
        t(n_envs, n_agents, 2),
        t(n_envs, n_agents),
        t(n_envs, n_agents, 2),
        t(n_envs, n_agents),
        t(n_envs, n_agents),
    )


def test_compiled_matches_eager_forward():
    stepper = _stepper()
    state = _state()
    actions = 0.3 * torch.randn(2, 2, 2, dtype=torch.float64)
    with torch.no_grad():
        a = warp_step(stepper, state, actions)
        b = compiled_warp_step(stepper, state, actions)
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y)


def test_compiled_step_gradcheck():
    stepper = _stepper(substeps=2)
    state = _state(requires_grad=True)
    actions = (0.3 * torch.randn(2, 2, 2, dtype=torch.float64)).requires_grad_(True)

    def fn(pos, theta, vel, speed, ang_vel, act):
        return tuple(compiled_warp_step(stepper, TorchState(pos, theta, vel, speed, ang_vel), act))[
            :5
        ]

    core = (state.pos, state.theta, state.vel, state.speed, state.ang_vel)
    assert torch.autograd.gradcheck(fn, (*core, actions), eps=1e-6, atol=1e-5)


def test_torch_compile_runs_and_matches():
    """A short rollout compiled with torch.compile matches eager and stays finite."""
    stepper = _stepper()
    state = _state()
    actions = 0.2 * torch.randn(2, 2, 2, dtype=torch.float64)

    def rollout(state, actions):
        s = state
        total = torch.zeros((), dtype=torch.float64)
        for _ in range(3):
            s = compiled_warp_step(stepper, s, actions)
            total = total + s.pos.square().sum()
        return total

    eager = rollout(state, actions)
    compiled = torch.compile(rollout)(state, actions)
    assert torch.isfinite(compiled)
    torch.testing.assert_close(compiled, eager)


@pytest.mark.gpu
def test_cuda_graph_matches_eager():
    """A CUDA-graph-captured step reproduces the eager no-grad step over a rollout."""
    world = WorldConfig(collisions=True, collision_k=50.0)  # small fleet -> brute neighbors
    stepper = _stepper(dtype=wp.float32, substeps=2, device="cuda:0", world=world)
    n_envs = 16

    def mk_state():
        g = torch.Generator().manual_seed(1)
        f = lambda *s: (0.3 * torch.randn(*s, generator=g)).to("cuda:0")  # noqa: E731
        return TorchState(
            f(n_envs, 2, 2), f(n_envs, 2), f(n_envs, 2, 2), f(n_envs, 2), f(n_envs, 2)
        )

    graph = CudaGraphStep(stepper, n_envs, act_dim=2)
    s_eager = mk_state()
    s_graph = mk_state()
    gen = torch.Generator().manual_seed(9)
    for _ in range(6):
        a = (0.4 * torch.randn(n_envs, 2, 2, generator=gen)).to("cuda:0")
        with torch.no_grad():
            s_eager = warp_step(stepper, s_eager, a)
        s_graph = graph(s_graph, a)
    for x, y in zip(s_eager, s_graph, strict=True):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
