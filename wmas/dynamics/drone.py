"""6-DOF quadrotor drone dynamics helpers.

The drone is a first-class :class:`~wmas.dynamics.base.DynamicsModel`: its
integration lives in the unified kernel (the ``TAG_DRONE`` branch of
:mod:`wmas.dynamics.kernels`), its extra state (altitude ``z``, vertical
velocity ``vz``, attitude quaternion, body rates) lives in
:class:`~wmas.core.state.WorldState`, and its parameters (per-rotor thrust
limit, arm length, diagonal inertia, yaw reaction coefficient, gravity) extend
the parameter matrix in :mod:`wmas.dynamics.base`.

Model: a "+"-configuration quadrotor. The 4 action slots are per-rotor thrust
commands, clamped to ``[0, thrust_max]``. Total thrust acts along body +z;
roll/pitch torques come from opposing-rotor thrust differences over the arm,
yaw torque from the rotor reaction sum. Attitude follows quaternion kinematics
and the body rates follow Euler's rigid-body equation with diagonal inertia.
The whole step is differentiable and integrates with either Euler or RK4.
"""

from __future__ import annotations

from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel


def drone_config(
    *,
    mass: float = 1.0,
    radius: float = 0.15,
    thrust_max: float = 10.0,
    arm_length: float = 0.15,
    inertia_xx: float = 0.01,
    inertia_yy: float = 0.01,
    inertia_zz: float = 0.02,
    torque_coeff: float = 0.02,
    gravity: float = 9.81,
) -> AgentConfig:
    """Build an :class:`AgentConfig` for a 6-DOF quadrotor drone.

    ``radius`` is the collision footprint used by the shared 2D neighbor/contact
    machinery (the drone's horizontal projection).
    """
    return AgentConfig(
        model=DynamicsModel.DRONE,
        ctrl_mode=ControlMode.ACCELERATION,  # ignored by the drone branch
        radius=radius,
        mass=mass,
        thrust_max=thrust_max,
        arm_length=arm_length,
        inertia_xx=inertia_xx,
        inertia_yy=inertia_yy,
        inertia_zz=inertia_zz,
        torque_coeff=torque_coeff,
        gravity=gravity,
    )
