"""Dynamics model tags, control modes, and per-agent configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np
import warp as wp


class DynamicsModel(IntEnum):
    """Tag selecting the motion model inside the unified integrate kernel."""

    HOLONOMIC = 0
    DIFF_DRIVE = 1
    KINEMATIC_BICYCLE = 2
    DRONE = 3  # 6-DOF quadrotor, see swarp.dynamics.drone


#: Number of scalar action slots each model reads from the action vector. All
#: current 2D vehicle models use 2; the action array is padded to the env-level
#: max, and models simply ignore slots beyond their arity (see integrate_kernel).
MODEL_ACTION_DIM = {
    "HOLONOMIC": 2,
    "DIFF_DRIVE": 2,
    "KINEMATIC_BICYCLE": 2,
    "DRONE": 4,  # the 6-DOF drone's four per-rotor thrust commands
}


def action_dim(model: DynamicsModel) -> int:
    """Action arity for a dynamics model."""
    return MODEL_ACTION_DIM[model.name]


class ControlMode(IntEnum):
    """Interpretation of the 2D action vector.

    HOLONOMIC: VELOCITY -> (vx, vy), ACCELERATION -> (ax, ay).
    DIFF_DRIVE: VELOCITY -> (v, omega), ACCELERATION -> (a, alpha).
    KINEMATIC_BICYCLE: always (acceleration, steering angle); the flag is ignored.
    DRONE: always four per-rotor thrusts; the flag is ignored.
    """

    VELOCITY = 0
    ACCELERATION = 1


class Integrator(Enum):
    """Time integrator for the dynamics: semi-implicit Euler, or classical RK4."""

    EULER = "euler"
    RK4 = "rk4"


# Column indices into the float parameter matrix [n_agents, NUM_PARAMS].
P_RADIUS = 0
P_MASS = 1
P_MAX_SPEED = 2
P_MAX_ACCEL = 3
P_MAX_ANG_VEL = 4
P_MAX_ANG_ACCEL = 5
P_LF = 6
P_LR = 7
P_MAX_STEER = 8
# 6-DOF drone parameters (ignored by the 2D vehicle models).
P_THRUST_MAX = 9  # per-rotor max thrust (N)
P_ARM = 10  # rotor arm length from CoG (m)
P_IXX = 11  # body-frame diagonal inertia
P_IYY = 12
P_IZZ = 13
P_KAPPA = 14  # yaw reaction torque per unit rotor thrust
P_GRAVITY = 15  # gravitational acceleration (m/s^2)
NUM_PARAMS = 16


@dataclass
class AgentConfig:
    """Static per-agent definition: model, shape, and actuation limits."""

    model: DynamicsModel = DynamicsModel.HOLONOMIC
    ctrl_mode: ControlMode = ControlMode.VELOCITY
    radius: float = 0.05
    mass: float = 1.0
    max_speed: float = 1.0
    max_accel: float = 1.0
    max_ang_vel: float = math.pi
    max_ang_accel: float = 2.0 * math.pi
    l_f: float = 0.1  # front axle to center of gravity (bicycle only)
    l_r: float = 0.1  # rear axle to center of gravity (bicycle only)
    max_steer: float = math.pi / 4  # bicycle only
    # 6-DOF drone parameters (used only when model == DRONE).
    thrust_max: float = 10.0  # per-rotor max thrust (N)
    arm_length: float = 0.15  # rotor arm length from CoG (m)
    inertia_xx: float = 0.01
    inertia_yy: float = 0.01
    inertia_zz: float = 0.02
    torque_coeff: float = 0.02  # yaw reaction torque per unit rotor thrust
    gravity: float = 9.81

    def __post_init__(self) -> None:
        if self.l_f + self.l_r <= 0.0:
            raise ValueError("Bicycle wheelbase l_f + l_r must be positive.")
        for name in ("radius", "mass", "max_speed", "max_accel", "max_ang_vel", "max_ang_accel"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"AgentConfig.{name} must be positive.")
        if self.model == DynamicsModel.DRONE:
            for name in ("thrust_max", "arm_length", "inertia_xx", "inertia_yy", "inertia_zz"):
                if getattr(self, name) <= 0.0:
                    raise ValueError(f"AgentConfig.{name} must be positive for a drone.")

    def to_row(self) -> list[float]:
        return [
            self.radius,
            self.mass,
            self.max_speed,
            self.max_accel,
            self.max_ang_vel,
            self.max_ang_accel,
            self.l_f,
            self.l_r,
            self.max_steer,
            self.thrust_max,
            self.arm_length,
            self.inertia_xx,
            self.inertia_yy,
            self.inertia_zz,
            self.torque_coeff,
            self.gravity,
        ]


@dataclass
class AgentParams:
    """Device-resident per-agent parameters.

    ``floats`` is the shared ``[n_agents, NUM_PARAMS]`` layout used by all envs
    (the default fast path). ``floats_per_env`` is an opt-in
    ``[n_envs, n_agents, NUM_PARAMS]`` override for domain randomization; when it
    is not ``None`` the stepper launches the per-env kernel variants that index
    ``params[e, a, ...]`` instead of ``params[a, ...]``. ``model_tag``/
    ``ctrl_mode`` stay per-agent (structural, never randomized).
    """

    floats: wp.array  # [n_agents, NUM_PARAMS], dtype float32/float64
    model_tag: wp.array  # [n_agents], int32
    ctrl_mode: wp.array  # [n_agents], int32
    floats_per_env: wp.array | None = None  # [n_envs, n_agents, NUM_PARAMS] or None


def build_agent_params(
    configs: list[AgentConfig],
    device: str,
    dtype=wp.float32,
) -> AgentParams:
    """Pack a list of AgentConfig into device arrays consumed by the kernels."""
    npdt = np.float64 if dtype == wp.float64 else np.float32
    rows = np.array([c.to_row() for c in configs], dtype=npdt)
    tags = np.array([int(c.model) for c in configs], dtype=np.int32)
    modes = np.array([int(c.ctrl_mode) for c in configs], dtype=np.int32)
    return AgentParams(
        floats=wp.array(rows, dtype=dtype, device=device),
        model_tag=wp.array(tags, dtype=wp.int32, device=device),
        ctrl_mode=wp.array(modes, dtype=wp.int32, device=device),
    )


def per_env_float_template(configs: list[AgentConfig], n_envs: int) -> np.ndarray:
    """A ``[n_envs, n_agents, NUM_PARAMS]`` float64 array pre-filled from ``configs``.

    Every env starts as a copy of the shared per-agent rows; edit columns (e.g.
    ``[..., P_MASS]``) to randomize, then hand the result to
    :meth:`swarp.core.stepper.Stepper.set_agent_params_per_env`.
    """
    rows = np.array([c.to_row() for c in configs], dtype=np.float64)
    return np.broadcast_to(rows, (n_envs, *rows.shape)).copy()
