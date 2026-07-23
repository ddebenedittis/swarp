"""Dynamics correctness vs hand-computed numpy reference trajectories.

The references implement the *identical discrete recurrence* the Warp kernel is
specified to use (semi-implicit Euler: velocity-level state updated from the
action first, pose then integrated with the new velocities), in float64.
"""

import numpy as np
import pytest
import torch
import warp as wp

from wmas.core.state import WorldState
from wmas.dynamics.base import (
    AgentConfig,
    ControlMode,
    DynamicsModel,
    Integrator,
    build_agent_params,
)
from wmas.dynamics.kernels import launch_integrate

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def clamp_norm(v: np.ndarray, limit: float) -> np.ndarray:
    n = np.linalg.norm(v)
    return v * (limit / max(n, limit))


def ref_holonomic(pos, vel, actions, cfg: AgentConfig, dt: float):
    """Reference for one agent over a sequence of actions. Returns trajectory of pos."""
    traj = []
    for a in actions:
        if cfg.ctrl_mode == ControlMode.VELOCITY:
            v_new = clamp_norm(a, cfg.max_speed)
        else:
            acc = clamp_norm(a, cfg.max_accel)
            v_new = clamp_norm(vel + acc * dt, cfg.max_speed)
        pos = pos + v_new * dt
        vel = v_new
        traj.append((pos.copy(), vel.copy()))
    return traj


def ref_diff_drive(pos, theta, speed, ang_vel, actions, cfg: AgentConfig, dt: float):
    traj = []
    for a in actions:
        if cfg.ctrl_mode == ControlMode.VELOCITY:
            s_new = np.clip(a[0], -cfg.max_speed, cfg.max_speed)
            w_new = np.clip(a[1], -cfg.max_ang_vel, cfg.max_ang_vel)
        else:
            acc = np.clip(a[0], -cfg.max_accel, cfg.max_accel)
            alp = np.clip(a[1], -cfg.max_ang_accel, cfg.max_ang_accel)
            s_new = np.clip(speed + acc * dt, -cfg.max_speed, cfg.max_speed)
            w_new = np.clip(ang_vel + alp * dt, -cfg.max_ang_vel, cfg.max_ang_vel)
        v_trans = s_new * np.array([np.cos(theta), np.sin(theta)])
        pos = pos + v_trans * dt
        theta = theta + w_new * dt
        speed, ang_vel = s_new, w_new
        traj.append((pos.copy(), theta, v_trans.copy(), speed, ang_vel))
    return traj


def ref_bicycle(pos, theta, speed, actions, cfg: AgentConfig, dt: float):
    """Kinematic bicycle with slip angle beta (Polack et al. 2017, eq. 2)."""
    wheelbase = cfg.l_f + cfg.l_r
    traj = []
    for a in actions:
        acc = np.clip(a[0], -cfg.max_accel, cfg.max_accel)
        delta = np.clip(a[1], -cfg.max_steer, cfg.max_steer)
        s_new = np.clip(speed + acc * dt, -cfg.max_speed, cfg.max_speed)
        beta = np.arctan2(np.tan(delta) * cfg.l_r / wheelbase, 1.0)
        v_trans = s_new * np.array([np.cos(theta + beta), np.sin(theta + beta)])
        w_new = s_new / wheelbase * np.cos(beta) * np.tan(delta)
        pos = pos + v_trans * dt
        theta = theta + w_new * dt
        speed = s_new
        traj.append((pos.copy(), theta, v_trans.copy(), speed, w_new))
    return traj


def rollout_warp(
    cfgs, init, actions_seq, dt, device, dtype=wp.float64, integrator=Integrator.EULER
):
    """Step n_agents agents in a single env through actions_seq [T, n_agents, 2]."""
    n_agents = len(cfgs)
    params = build_agent_params(cfgs, device=device, dtype=dtype)
    state = WorldState.zeros(1, n_agents, dtype=dtype, device=device)
    npdt = np.float64 if dtype == wp.float64 else np.float32
    vec2 = wp.vec2d if dtype == wp.float64 else wp.vec2f

    state.pos.assign(init["pos"].astype(npdt).reshape(1, n_agents, 2))
    state.theta.assign(init["theta"].astype(npdt).reshape(1, n_agents))
    state.vel.assign(init["vel"].astype(npdt).reshape(1, n_agents, 2))
    state.speed.assign(init["speed"].astype(npdt).reshape(1, n_agents))
    state.ang_vel.assign(init["ang_vel"].astype(npdt).reshape(1, n_agents))

    forces = wp.zeros((1, n_agents), dtype=vec2, device=device)
    out = WorldState.zeros(1, n_agents, dtype=dtype, device=device)
    traj = []
    for actions in actions_seq:
        # actions are now a scalar [n_envs, n_agents, act_dim] array3d (not vec2)
        acts = wp.array(actions.astype(npdt).reshape(1, n_agents, 2), dtype=dtype, device=device)
        launch_integrate(state, out, acts, forces, params, dt, integrator=integrator)
        state, out = out, state
        # .numpy() is a zero-copy view on CPU -> copy before the buffer is reused
        traj.append(
            {
                "pos": state.pos.numpy().reshape(n_agents, 2).copy(),
                "theta": state.theta.numpy().reshape(n_agents).copy(),
                "vel": state.vel.numpy().reshape(n_agents, 2).copy(),
                "speed": state.speed.numpy().reshape(n_agents).copy(),
                "ang_vel": state.ang_vel.numpy().reshape(n_agents).copy(),
            }
        )
    return traj


def make_actions(rng, T, n_agents, scale=2.0):
    return scale * rng.standard_normal((T, n_agents, 2))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ctrl_mode", [ControlMode.VELOCITY, ControlMode.ACCELERATION])
def test_holonomic(device, ctrl_mode):
    cfg = AgentConfig(
        model=DynamicsModel.HOLONOMIC, ctrl_mode=ctrl_mode, max_speed=1.3, max_accel=2.0
    )
    rng = np.random.default_rng(0)
    T, dt = 100, 0.05
    actions = make_actions(rng, T, 1)
    init = {
        "pos": np.array([[0.3, -0.2]]),
        "theta": np.zeros(1),
        "vel": np.array([[0.1, 0.2]]),
        "speed": np.zeros(1),
        "ang_vel": np.zeros(1),
    }
    traj = rollout_warp([cfg], init, actions, dt, device)
    ref = ref_holonomic(init["pos"][0], init["vel"][0], actions[:, 0], cfg, dt)
    for t in range(T):
        np.testing.assert_allclose(traj[t]["pos"][0], ref[t][0], atol=1e-12)
        np.testing.assert_allclose(traj[t]["vel"][0], ref[t][1], atol=1e-12)
    # holonomic ignores heading
    assert traj[-1]["theta"][0] == 0.0
    np.testing.assert_allclose(traj[-1]["speed"][0], np.linalg.norm(traj[-1]["vel"][0]), atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ctrl_mode", [ControlMode.VELOCITY, ControlMode.ACCELERATION])
def test_diff_drive(device, ctrl_mode):
    cfg = AgentConfig(
        model=DynamicsModel.DIFF_DRIVE,
        ctrl_mode=ctrl_mode,
        max_speed=1.0,
        max_accel=1.5,
        max_ang_vel=2.0,
        max_ang_accel=4.0,
    )
    rng = np.random.default_rng(1)
    T, dt = 100, 0.05
    actions = make_actions(rng, T, 1)
    init = {
        "pos": np.array([[0.0, 0.0]]),
        "theta": np.array([0.7]),
        "vel": np.zeros((1, 2)),
        "speed": np.array([0.2]),
        "ang_vel": np.array([-0.3]),
    }
    traj = rollout_warp([cfg], init, actions, dt, device)
    ref = ref_diff_drive(
        init["pos"][0],
        init["theta"][0],
        init["speed"][0],
        init["ang_vel"][0],
        actions[:, 0],
        cfg,
        dt,
    )
    for t in range(T):
        np.testing.assert_allclose(traj[t]["pos"][0], ref[t][0], atol=1e-12)
        np.testing.assert_allclose(traj[t]["theta"][0], ref[t][1], atol=1e-12)
        np.testing.assert_allclose(traj[t]["vel"][0], ref[t][2], atol=1e-12)
        np.testing.assert_allclose(traj[t]["speed"][0], ref[t][3], atol=1e-12)
        np.testing.assert_allclose(traj[t]["ang_vel"][0], ref[t][4], atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_kinematic_bicycle(device):
    cfg = AgentConfig(
        model=DynamicsModel.KINEMATIC_BICYCLE,
        ctrl_mode=ControlMode.ACCELERATION,
        max_speed=2.0,
        max_accel=1.0,
        l_f=0.16,
        l_r=0.14,
        max_steer=0.6,
    )
    rng = np.random.default_rng(2)
    T, dt = 100, 0.05
    actions = make_actions(rng, T, 1)
    init = {
        "pos": np.array([[1.0, -1.0]]),
        "theta": np.array([-0.5]),
        "vel": np.zeros((1, 2)),
        "speed": np.array([0.5]),
        "ang_vel": np.zeros(1),
    }
    traj = rollout_warp([cfg], init, actions, dt, device)
    ref = ref_bicycle(init["pos"][0], init["theta"][0], init["speed"][0], actions[:, 0], cfg, dt)
    for t in range(T):
        np.testing.assert_allclose(traj[t]["pos"][0], ref[t][0], atol=1e-12)
        np.testing.assert_allclose(traj[t]["theta"][0], ref[t][1], atol=1e-12)
        np.testing.assert_allclose(traj[t]["speed"][0], ref[t][3], atol=1e-12)
        np.testing.assert_allclose(traj[t]["ang_vel"][0], ref[t][4], atol=1e-12)


def test_bicycle_straight_line_analytic():
    """Zero steering, constant accel from rest: discrete kinematics are summable by hand."""
    cfg = AgentConfig(
        model=DynamicsModel.KINEMATIC_BICYCLE,
        max_speed=10.0,
        max_accel=1.0,
        l_f=0.15,
        l_r=0.15,
        max_steer=0.6,
    )
    T, dt, acc = 20, 0.1, 0.8
    actions = np.tile(np.array([acc, 0.0]), (T, 1, 1))
    init = {
        "pos": np.zeros((1, 2)),
        "theta": np.zeros(1),
        "vel": np.zeros((1, 2)),
        "speed": np.zeros(1),
        "ang_vel": np.zeros(1),
    }
    traj = rollout_warp([cfg], init, actions, dt, "cpu")
    # semi-implicit Euler: x_T = sum_{k=1..T} (k * acc * dt) * dt
    expected_x = acc * dt * dt * T * (T + 1) / 2
    np.testing.assert_allclose(traj[-1]["pos"][0], [expected_x, 0.0], atol=1e-12)
    np.testing.assert_allclose(traj[-1]["speed"][0], acc * dt * T, atol=1e-12)
    assert traj[-1]["theta"][0] == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_clamping(device):
    """Huge commands saturate at the configured limits."""
    dt = 0.1
    big = np.array([[[100.0, 100.0]]])
    cfg_h = AgentConfig(model=DynamicsModel.HOLONOMIC, max_speed=1.0)
    init = {
        "pos": np.zeros((1, 2)),
        "theta": np.zeros(1),
        "vel": np.zeros((1, 2)),
        "speed": np.zeros(1),
        "ang_vel": np.zeros(1),
    }
    traj = rollout_warp([cfg_h], init, big, dt, device)
    np.testing.assert_allclose(np.linalg.norm(traj[0]["vel"][0]), 1.0, atol=1e-12)

    cfg_b = AgentConfig(
        model=DynamicsModel.KINEMATIC_BICYCLE,
        max_speed=1.0,
        max_accel=0.5,
        max_steer=0.3,
        l_f=0.1,
        l_r=0.1,
    )
    traj = rollout_warp([cfg_b], init, big, dt, device)
    np.testing.assert_allclose(traj[0]["speed"][0], 0.5 * dt, atol=1e-12)  # accel clamped
    ref = ref_bicycle(np.zeros(2), 0.0, 0.0, np.array([[100.0, 100.0]]), cfg_b, dt)
    np.testing.assert_allclose(traj[0]["theta"][0], ref[0][1], atol=1e-12)  # steer clamped


@pytest.mark.parametrize("device", DEVICES)
def test_heterogeneous_fleet(device):
    """Three agents with different models in one world match their individual references."""
    cfgs = [
        AgentConfig(model=DynamicsModel.HOLONOMIC, max_speed=1.3),
        AgentConfig(model=DynamicsModel.DIFF_DRIVE, max_speed=1.0, max_ang_vel=2.0),
        AgentConfig(
            model=DynamicsModel.KINEMATIC_BICYCLE,
            max_speed=2.0,
            max_accel=1.0,
            l_f=0.16,
            l_r=0.14,
            max_steer=0.6,
        ),
    ]
    rng = np.random.default_rng(3)
    T, dt = 50, 0.05
    actions = make_actions(rng, T, 3)
    init = {
        "pos": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "theta": np.array([0.0, 0.5, -0.5]),
        "vel": np.zeros((3, 2)),
        "speed": np.array([0.0, 0.1, 0.4]),
        "ang_vel": np.zeros(3),
    }
    traj = rollout_warp(cfgs, init, actions, dt, device)

    ref_h = ref_holonomic(init["pos"][0], init["vel"][0], actions[:, 0], cfgs[0], dt)
    ref_d = ref_diff_drive(
        init["pos"][1],
        init["theta"][1],
        init["speed"][1],
        init["ang_vel"][1],
        actions[:, 1],
        cfgs[1],
        dt,
    )
    ref_b = ref_bicycle(
        init["pos"][2], init["theta"][2], init["speed"][2], actions[:, 2], cfgs[2], dt
    )
    np.testing.assert_allclose(traj[-1]["pos"][0], ref_h[-1][0], atol=1e-12)
    np.testing.assert_allclose(traj[-1]["pos"][1], ref_d[-1][0], atol=1e-12)
    np.testing.assert_allclose(traj[-1]["pos"][2], ref_b[-1][0], atol=1e-12)


def test_float32_matches_float64_loosely():
    cfg = AgentConfig(model=DynamicsModel.DIFF_DRIVE, max_speed=1.0, max_ang_vel=2.0)
    rng = np.random.default_rng(4)
    T, dt = 100, 0.05
    actions = make_actions(rng, T, 1)
    init = {
        "pos": np.zeros((1, 2)),
        "theta": np.zeros(1),
        "vel": np.zeros((1, 2)),
        "speed": np.zeros(1),
        "ang_vel": np.zeros(1),
    }
    t64 = rollout_warp([cfg], init, actions, dt, "cpu", dtype=wp.float64)
    t32 = rollout_warp([cfg], init, actions, dt, "cpu", dtype=wp.float32)
    np.testing.assert_allclose(t32[-1]["pos"], t64[-1]["pos"], atol=1e-4)


def test_drone_placeholder_raises():
    from wmas.dynamics.drone import DronePlaceholder

    with pytest.raises(NotImplementedError):
        DronePlaceholder()
    with pytest.raises(NotImplementedError):
        AgentConfig(model=DynamicsModel.DRONE)


# --------------------------------------------------------------------- RK4


def _diff_drive_circle_analytic(x0, y0, th0, s, w, t):
    """Exact solution of dtheta/dt = w, dp/dt = s*[cos theta, sin theta]."""
    th = th0 + w * t
    x = x0 + (s / w) * (np.sin(th) - np.sin(th0))
    y = y0 - (s / w) * (np.cos(th) - np.cos(th0))
    return np.array([x, y]), th


def _diff_drive_final(dt, N, device, integrator):
    """Roll a constant (speed, ang_vel) command for N steps; return final pos."""
    cfg = AgentConfig(
        model=DynamicsModel.DIFF_DRIVE,
        ctrl_mode=ControlMode.VELOCITY,
        max_speed=5.0,
        max_ang_vel=5.0,
    )
    s, w = 1.3, 0.9
    actions = np.tile(np.array([s, w]), (N, 1, 1))
    init = {
        "pos": np.array([[0.2, -0.1]]),
        "theta": np.array([0.3]),
        "vel": np.zeros((1, 2)),
        "speed": np.zeros(1),
        "ang_vel": np.zeros(1),
    }
    traj = rollout_warp([cfg], init, actions, dt, device, integrator=integrator)
    return traj[-1]["pos"][0], (init, s, w)


@pytest.mark.parametrize("device", DEVICES)
def test_rk4_convergence_order(device):
    """On a curved (circular) trajectory, RK4 converges ~dt^4, Euler ~dt^1, and
    at a fixed step RK4 is far more accurate than semi-implicit Euler."""
    total_t = 2.0

    def errs(integrator, N):
        dt = total_t / N
        final, (init, s, w) = _diff_drive_final(dt, N, device, integrator)
        exact, _ = _diff_drive_circle_analytic(
            init["pos"][0, 0], init["pos"][0, 1], init["theta"][0], s, w, total_t
        )
        return np.linalg.norm(final - exact)

    # Two resolutions to estimate the observed convergence order.
    e_rk4_coarse = errs(Integrator.RK4, 100)
    e_rk4_fine = errs(Integrator.RK4, 200)
    e_eu_coarse = errs(Integrator.EULER, 100)
    e_eu_fine = errs(Integrator.EULER, 200)

    order_rk4 = np.log2(e_rk4_coarse / e_rk4_fine)
    order_eu = np.log2(e_eu_coarse / e_eu_fine)

    assert order_rk4 > 3.5, f"RK4 order {order_rk4:.2f} not ~4"
    assert 0.8 < order_eu < 1.3, f"Euler order {order_eu:.2f} not ~1"
    # RK4 is orders of magnitude tighter at the same step count.
    assert e_rk4_coarse < e_eu_coarse * 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_rk4_matches_euler_on_straight_line(device):
    """With no curvature (constant holonomic velocity), RK4 and semi-implicit
    Euler are both exact, so they agree to round-off."""
    cfg = AgentConfig(model=DynamicsModel.HOLONOMIC, ctrl_mode=ControlMode.VELOCITY, max_speed=2.0)
    actions = np.tile(np.array([0.7, -0.4]), (20, 1, 1))
    init = {
        "pos": np.zeros((1, 2)),
        "theta": np.zeros(1),
        "vel": np.zeros((1, 2)),
        "speed": np.zeros(1),
        "ang_vel": np.zeros(1),
    }
    t_eu = rollout_warp([cfg], init, actions, 0.05, device, integrator=Integrator.EULER)
    t_rk = rollout_warp([cfg], init, actions, 0.05, device, integrator=Integrator.RK4)
    np.testing.assert_allclose(t_rk[-1]["pos"], t_eu[-1]["pos"], atol=1e-12)
