"""Command latency (``Stepper.enable_command_delay``): the first ``cmd_delay[e, a]``
substeps of a step apply the previous step's command, the rest the new one."""

import numpy as np
import pytest
import warp as wp
from conftest import DEVICES

from swarp.core.stepper import Stepper
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("enable_fused", [False, "auto"])
def test_delayed_command_moves_for_the_undelayed_substeps(device, enable_fused):
    """A unicycle at rest facing +x commanded 1 m/s for one step, then 0: it covers
    ``(substeps - d) sub_dt`` in the first step and ``d sub_dt`` in the second."""
    cfg = AgentConfig(model=DynamicsModel.DIFF_DRIVE, ctrl_mode=ControlMode.VELOCITY,
                      max_speed=2.0)
    substeps, dt = 8, 0.08
    stepper = Stepper([cfg, cfg], dt=dt, substeps=substeps, device=device, dtype=wp.float64)
    stepper.enable_fused_substeps = enable_fused
    delays = np.array([[0, 3], [5, 1]], dtype=np.int32)
    stepper.enable_command_delay(2, act_dim=2, max_delay=5)
    stepper.cmd_delay.assign(delays)
    state = stepper.alloc_state(2)
    out = stepper.alloc_state(2)
    buffers = stepper.cached_buffers(2)
    go = np.zeros((2, 2, 2))
    go[..., 0] = 1.0
    x = []
    for cmd in (go, np.zeros_like(go)):
        acts = wp.array(cmd, dtype=wp.float64, device=device)
        stepper.launch_substeps(state, acts, out, buffers)
        state, out = out, state
        x.append(state.pos.numpy()[..., 0].copy())
    sub_dt = dt / substeps
    np.testing.assert_allclose(x[0], (substeps - delays) * sub_dt, atol=1e-12)
    np.testing.assert_allclose(x[1] - x[0], delays * sub_dt, atol=1e-12)
    np.testing.assert_array_equal(stepper.prev_actions.numpy(), 0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_arena_stop_zeroes_only_outward_speed(device):
    """Beyond the edge, a command driving further out loses its speed but keeps its turn
    rate; driving or reversing back in passes, and inside the edge nothing is touched."""
    cfg = AgentConfig(model=DynamicsModel.DIFF_DRIVE, ctrl_mode=ControlMode.VELOCITY,
                      max_speed=2.0, max_ang_vel=2.0)
    stepper = Stepper([cfg] * 4, dt=0.08, substeps=8, device=device, dtype=wp.float64)
    stepper.enable_arena_stop(1, act_dim=2, edge=2.0)
    state = stepper.alloc_state(1)
    out = stepper.alloc_state(1)
    # beyond x driving out, beyond x reversing in, beyond -y driving out, inside driving out
    state.pos.assign(np.array([[[2.1, 0.0], [2.1, 0.0], [0.0, -2.1], [1.9, 0.0]]]))
    state.theta.assign(np.array([[0.0, 0.0, -np.pi / 2, 0.0]]))
    cmd = np.array([[[1.0, 0.5], [-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    acts = wp.array(cmd, dtype=wp.float64, device=device)
    stepper.launch_substeps(state, acts, out, stepper.cached_buffers(1))
    moved = np.linalg.norm(out.pos.numpy()[0] - state.pos.numpy()[0], axis=-1)
    np.testing.assert_allclose(moved, [0.0, 0.08, 0.0, 0.08], atol=1e-9)
    np.testing.assert_allclose(out.theta.numpy()[0, 0], 0.5 * 0.08, atol=1e-12)
