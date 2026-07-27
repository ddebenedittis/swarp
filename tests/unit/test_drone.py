"""6-DOF quadrotor drone: numpy-reference trajectories, analytic sub-cases, gradcheck.

The numpy reference reproduces the exact semi-implicit-Euler recurrence the Warp
kernel runs (attitude as an (x, y, z, w) quaternion, body->world), so full
trajectories match to float64 round-off. Hover and vertical-climb are additionally
checked against closed-form expressions.
"""

import numpy as np
import pytest
import torch
import warp as wp
from conftest import DEVICES

from wmas.core.state import VEC2, WorldState
from wmas.dynamics.base import AgentConfig, DynamicsModel, build_agent_params
from wmas.dynamics.kernels import launch_integrate
from wmas.interop.autograd import TorchState, warp_step

PRM = dict(
    mass=1.0,
    arm=0.15,
    ixx=0.01,
    iyy=0.01,
    izz=0.02,
    kappa=0.02,
    gravity=9.81,
    thrust_max=20.0,
)


def drone_cfg():
    return AgentConfig(
        model=DynamicsModel.DRONE,
        mass=PRM["mass"],
        arm_length=PRM["arm"],
        inertia_xx=PRM["ixx"],
        inertia_yy=PRM["iyy"],
        inertia_zz=PRM["izz"],
        torque_coeff=PRM["kappa"],
        gravity=PRM["gravity"],
        thrust_max=PRM["thrust_max"],
    )


def _rotate(q, v):
    u = q[:3]
    t = 2.0 * np.cross(u, v)
    return v + q[3] * t + np.cross(u, t)


def _qderiv(q, w):
    x, y, z, ww = q
    wx, wy, wz = w
    return 0.5 * np.array(
        [
            ww * wx + y * wz - z * wy,
            ww * wy - x * wz + z * wx,
            ww * wz + x * wy - y * wx,
            -x * wx - y * wy - z * wz,
        ]
    )


def ref_drone(init, thrusts_seq, dt):
    """Semi-implicit Euler reference identical to the kernel's DRONE branch."""
    p = init["p"].astype(np.float64).copy()
    z = float(init["z"])
    v = init["v"].astype(np.float64).copy()
    vz = float(init["vz"])
    q = init["q"].astype(np.float64).copy()
    br = init["br"].astype(np.float64).copy()
    traj = []
    for a in thrusts_seq:
        f = np.clip(a.astype(np.float64), 0.0, PRM["thrust_max"])
        thrust = f.sum()
        f_world = _rotate(q, np.array([0.0, 0.0, thrust]))
        accel = f_world / PRM["mass"] - np.array([0.0, 0.0, PRM["gravity"]])
        tx = PRM["arm"] * (f[1] - f[3])
        ty = PRM["arm"] * (f[2] - f[0])
        tz = PRM["kappa"] * (f[0] - f[1] + f[2] - f[3])
        ax = (tx - (PRM["izz"] - PRM["iyy"]) * br[1] * br[2]) / PRM["ixx"]
        ay = (ty - (PRM["ixx"] - PRM["izz"]) * br[2] * br[0]) / PRM["iyy"]
        az = (tz - (PRM["iyy"] - PRM["ixx"]) * br[0] * br[1]) / PRM["izz"]
        br = br + np.array([ax, ay, az]) * dt
        vz = vz + accel[2] * dt
        v = v + accel[:2] * dt
        q = q + _qderiv(q, br) * dt
        q = q / np.linalg.norm(q)
        p = p + v * dt
        z = z + vz * dt
        traj.append(
            {"p": p.copy(), "z": z, "v": v.copy(), "vz": vz, "q": q.copy(), "br": br.copy()}
        )
    return traj


def rollout_drone(init, thrusts_seq, dt, device, dtype=wp.float64):
    params = build_agent_params([drone_cfg()], device=device, dtype=dtype)
    npdt = np.float64 if dtype == wp.float64 else np.float32
    state = WorldState.zeros(1, 1, dtype=dtype, device=device)
    state.pos.assign(init["p"].reshape(1, 1, 2).astype(npdt))
    state.z.assign(np.array([[init["z"]]], dtype=npdt))
    state.vel.assign(init["v"].reshape(1, 1, 2).astype(npdt))
    state.vz.assign(np.array([[init["vz"]]], dtype=npdt))
    state.attitude.assign(init["q"].reshape(1, 1, 4).astype(npdt))
    state.body_rates.assign(init["br"].reshape(1, 1, 3).astype(npdt))
    out = WorldState.zeros(1, 1, dtype=dtype, device=device)
    forces = wp.zeros((1, 1), dtype=VEC2[dtype], device=device)
    traj = []
    for a in thrusts_seq:
        acts = wp.array(a.reshape(1, 1, 4).astype(npdt), dtype=dtype, device=device)
        launch_integrate(state, out, acts, forces, params, dt)
        state, out = out, state
        traj.append(
            {
                "p": state.pos.numpy().reshape(2).copy(),
                "z": float(state.z.numpy().reshape(())),
                "v": state.vel.numpy().reshape(2).copy(),
                "vz": float(state.vz.numpy().reshape(())),
                "q": state.attitude.numpy().reshape(4).copy(),
                "br": state.body_rates.numpy().reshape(3).copy(),
            }
        )
    return traj


def _rest_init():
    return {
        "p": np.array([0.3, -0.2]),
        "z": 1.0,
        "v": np.zeros(2),
        "vz": 0.0,
        "q": np.array([0.0, 0.0, 0.0, 1.0]),  # identity
        "br": np.zeros(3),
    }


@pytest.mark.parametrize("device", DEVICES)
def test_drone_hover_equilibrium(device):
    """Balanced thrust at m*g/4 per rotor holds position and level attitude."""
    hover = PRM["mass"] * PRM["gravity"] / 4.0
    thrusts = np.tile(np.full(4, hover), (50, 1))
    traj = rollout_drone(_rest_init(), thrusts, 0.02, device)
    last = traj[-1]
    np.testing.assert_allclose(last["p"], [0.3, -0.2], atol=1e-10)
    np.testing.assert_allclose(last["z"], 1.0, atol=1e-10)
    np.testing.assert_allclose(last["q"], [0.0, 0.0, 0.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(last["br"], [0.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_drone_vertical_climb_analytic(device):
    """Symmetric thrust above hover: no torque, closed-form altitude."""
    dt, T = 0.02, 40
    f = PRM["mass"] * PRM["gravity"] / 4.0 + 0.5
    thrusts = np.tile(np.full(4, f), (T, 1))
    traj = rollout_drone(_rest_init(), thrusts, dt, device)
    az = 4.0 * f / PRM["mass"] - PRM["gravity"]
    # semi-implicit: z_T = z0 + sum_{k=1..T} (k*az*dt) * dt
    expected_z = 1.0 + az * dt * dt * T * (T + 1) / 2.0
    np.testing.assert_allclose(traj[-1]["z"], expected_z, atol=1e-9)
    np.testing.assert_allclose(traj[-1]["vz"], az * dt * T, atol=1e-10)
    np.testing.assert_allclose(traj[-1]["q"], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(traj[-1]["p"], [0.3, -0.2], atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_drone_yaw_torque(device):
    """Rotors 0/2 above, 1/3 below hover: pure yaw, net-zero vertical thrust."""
    dt, T = 0.01, 30
    hover = PRM["mass"] * PRM["gravity"] / 4.0
    d = 0.5
    thrusts = np.tile(np.array([hover + d, hover - d, hover + d, hover - d]), (T, 1))
    traj = rollout_drone(_rest_init(), thrusts, dt, device)
    # net vertical thrust unchanged -> no altitude change; only yaw rate grows
    np.testing.assert_allclose(traj[-1]["z"], 1.0, atol=1e-9)
    assert traj[-1]["br"][2] > 1e-3  # yaw rate accumulated
    assert abs(traj[-1]["q"][2]) > 1e-3  # rotation about z
    # matches the numpy reference bit-for-bit
    ref = ref_drone(_rest_init(), thrusts, dt)
    np.testing.assert_allclose(traj[-1]["q"], ref[-1]["q"], atol=1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_drone_matches_numpy_reference(device):
    """General asymmetric thrusts (roll+pitch+yaw coupling) match the reference."""
    rng = np.random.default_rng(0)
    dt, T = 0.01, 60
    hover = PRM["mass"] * PRM["gravity"] / 4.0
    thrusts = hover + 0.4 * rng.standard_normal((T, 4))
    init = {
        "p": np.array([0.0, 0.0]),
        "z": 0.5,
        "v": np.array([0.1, -0.05]),
        "vz": 0.2,
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "br": np.array([0.05, -0.03, 0.1]),
    }
    traj = rollout_drone(init, thrusts, dt, device)
    ref = ref_drone(init, thrusts, dt)
    for key in ("p", "z", "v", "vz", "q", "br"):
        np.testing.assert_allclose(traj[-1][key], ref[-1][key], atol=1e-9, err_msg=key)


def test_drone_gradcheck():
    """The full 6-DOF drone step is differentiable w.r.t. state and rotor commands."""
    stepper_cfg = drone_cfg()
    from wmas.core.stepper import Stepper

    stepper = Stepper([stepper_cfg], dt=0.02, device="cpu", dtype=wp.float64)
    hover = PRM["mass"] * PRM["gravity"] / 4.0

    pos = torch.zeros(1, 1, 2, dtype=torch.float64, requires_grad=True)
    theta = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    vel = (0.05 * torch.randn(1, 1, 2, dtype=torch.float64)).requires_grad_(True)
    speed = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    ang_vel = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    z = torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
    vz = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    att = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64, requires_grad=True)
    br = (0.02 * torch.randn(1, 1, 3, dtype=torch.float64)).requires_grad_(True)
    actions = (hover + 0.1 * torch.randn(1, 1, 4, dtype=torch.float64)).requires_grad_(True)

    def fn(pos, theta, vel, speed, ang_vel, z, vz, att, br, act):
        out = warp_step(stepper, TorchState(pos, theta, vel, speed, ang_vel, z, vz, att, br), act)
        return tuple(out)

    assert torch.autograd.gradcheck(
        fn,
        (pos, theta, vel, speed, ang_vel, z, vz, att, br, actions),
        eps=1e-6,
        atol=1e-5,
    )
