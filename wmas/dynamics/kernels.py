"""Unified differentiable integration kernel for all 2D vehicle models.

One thread per (env, agent). The kernel is generic over precision and
explicitly instantiated for float32 and float64 via ``wp.overload`` (the
documented pattern; implicit instantiation triggers slow module reloads).

Discrete-time convention (semi-implicit Euler): the action first updates the
velocity-level state (clamped to the agent's limits), the pose is then
integrated with the *new* velocities. ``vel`` output always holds the
translational velocity used in the pose update, so observations are uniform
across models. Collision/boundary forces enter as an extra translational
velocity contribution ``F/m * dt`` (an approximation for the nonholonomic
models, in the same spirit as VMAS's force-based response).
"""

from __future__ import annotations

from typing import Any

import warp as wp

from wmas.core.state import VEC2, WorldState
from wmas.dynamics.base import (
    P_LF,
    P_LR,
    P_MASS,
    P_MAX_ACCEL,
    P_MAX_ANG_ACCEL,
    P_MAX_ANG_VEL,
    P_MAX_SPEED,
    P_MAX_STEER,
    AgentParams,
    DynamicsModel,
)

TAG_HOLONOMIC = wp.constant(int(DynamicsModel.HOLONOMIC))
TAG_DIFF_DRIVE = wp.constant(int(DynamicsModel.DIFF_DRIVE))
TAG_BICYCLE = wp.constant(int(DynamicsModel.KINEMATIC_BICYCLE))
MODE_VELOCITY = wp.constant(0)


@wp.func
def clamp_norm(v: Any, limit: Any):
    """Scale ``v`` so its norm does not exceed ``limit`` (differentiable, branch-free)."""
    return v * (limit / wp.max(wp.length(v), limit))


@wp.kernel
def integrate_kernel(
    # state in
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    speed: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
    # inputs
    actions: wp.array2d(dtype=Any),
    forces: wp.array2d(dtype=Any),
    params: wp.array2d(dtype=Any),
    model_tag: wp.array(dtype=wp.int32),
    ctrl_mode: wp.array(dtype=wp.int32),
    dt: Any,
    # state out
    pos_out: wp.array2d(dtype=Any),
    theta_out: wp.array2d(dtype=Any),
    vel_out: wp.array2d(dtype=Any),
    speed_out: wp.array2d(dtype=Any),
    ang_vel_out: wp.array2d(dtype=Any),
):
    e, a = wp.tid()
    p = pos[e, a]
    th = theta[e, a]
    v = vel[e, a]
    s = speed[e, a]
    w = ang_vel[e, a]
    act = actions[e, a]
    tag = model_tag[a]
    mode = ctrl_mode[a]

    zero = type(dt)(0.0)
    dv_f = forces[e, a] * (dt / params[a, P_MASS])

    if tag == TAG_HOLONOMIC:
        if mode == MODE_VELOCITY:
            v_new = clamp_norm(act, params[a, P_MAX_SPEED])
        else:
            acc_vec = clamp_norm(act, params[a, P_MAX_ACCEL])
            v_new = clamp_norm(v + acc_vec * dt, params[a, P_MAX_SPEED])
        v_new = v_new + dv_f
        pos_out[e, a] = p + v_new * dt
        theta_out[e, a] = th
        vel_out[e, a] = v_new
        speed_out[e, a] = wp.length(v_new)
        ang_vel_out[e, a] = zero
    elif tag == TAG_DIFF_DRIVE:
        max_s = params[a, P_MAX_SPEED]
        max_w = params[a, P_MAX_ANG_VEL]
        if mode == MODE_VELOCITY:
            s_new = wp.clamp(act[0], -max_s, max_s)
            w_new = wp.clamp(act[1], -max_w, max_w)
        else:
            max_a = params[a, P_MAX_ACCEL]
            max_al = params[a, P_MAX_ANG_ACCEL]
            acc = wp.clamp(act[0], -max_a, max_a)
            alp = wp.clamp(act[1], -max_al, max_al)
            s_new = wp.clamp(s + acc * dt, -max_s, max_s)
            w_new = wp.clamp(w + alp * dt, -max_w, max_w)
        v_trans = type(p)(s_new * wp.cos(th), s_new * wp.sin(th)) + dv_f
        pos_out[e, a] = p + v_trans * dt
        theta_out[e, a] = th + w_new * dt
        vel_out[e, a] = v_trans
        speed_out[e, a] = s_new
        ang_vel_out[e, a] = w_new
    elif tag == TAG_BICYCLE:
        # Kinematic bicycle with slip angle beta (Polack et al. 2017, eq. 2).
        max_s = params[a, P_MAX_SPEED]
        max_a = params[a, P_MAX_ACCEL]
        max_d = params[a, P_MAX_STEER]
        acc = wp.clamp(act[0], -max_a, max_a)
        delta = wp.clamp(act[1], -max_d, max_d)
        s_new = wp.clamp(s + acc * dt, -max_s, max_s)
        wheelbase = params[a, P_LF] + params[a, P_LR]
        beta = wp.atan(wp.tan(delta) * params[a, P_LR] / wheelbase)
        v_trans = type(p)(s_new * wp.cos(th + beta), s_new * wp.sin(th + beta)) + dv_f
        w_new = s_new / wheelbase * wp.cos(beta) * wp.tan(delta)
        pos_out[e, a] = p + v_trans * dt
        theta_out[e, a] = th + w_new * dt
        vel_out[e, a] = v_trans
        speed_out[e, a] = s_new
        ang_vel_out[e, a] = w_new
    else:
        # Unknown tag (e.g. future models): hold state.
        pos_out[e, a] = p
        theta_out[e, a] = th
        vel_out[e, a] = v
        speed_out[e, a] = s
        ang_vel_out[e, a] = w


def _signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    a2s = wp.array2d(dtype=dtype)
    a1i = wp.array(dtype=wp.int32)
    return [
        a2v, a2s, a2v, a2s, a2s,  # state in
        a2v, a2v, a2s, a1i, a1i, dtype,  # actions, forces, params, tags, modes, dt
        a2v, a2s, a2v, a2s, a2s,  # state out
    ]


for _T in (wp.float32, wp.float64):
    wp.overload(integrate_kernel, _signature(_T))


def launch_integrate(
    state_in: WorldState,
    state_out: WorldState,
    actions: wp.array,
    forces: wp.array,
    params: AgentParams,
    dt: float,
) -> None:
    """Launch one integration sub-step. Functional: never writes to ``state_in``."""
    dtype = state_in.theta.dtype
    n_envs, n_agents = state_in.pos.shape
    wp.launch(
        integrate_kernel,
        dim=(n_envs, n_agents),
        inputs=[
            *state_in.arrays(),
            actions,
            forces,
            params.floats,
            params.model_tag,
            params.ctrl_mode,
            dtype(dt),
        ],
        outputs=state_out.arrays(),
        device=state_in.pos.device,
    )
