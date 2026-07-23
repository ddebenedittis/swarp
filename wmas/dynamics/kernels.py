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
    P_RADIUS,
    AgentParams,
    DynamicsModel,
    Integrator,
)

TAG_HOLONOMIC = wp.constant(int(DynamicsModel.HOLONOMIC))
TAG_DIFF_DRIVE = wp.constant(int(DynamicsModel.DIFF_DRIVE))
TAG_BICYCLE = wp.constant(int(DynamicsModel.KINEMATIC_BICYCLE))
MODE_VELOCITY = wp.constant(0)

# Integrator tags (must match Integrator enum order in wmas.dynamics.base).
INT_EULER = wp.constant(0)
INT_RK4 = wp.constant(1)


@wp.func
def clamp_norm(v: Any, limit: Any):
    """Scale ``v`` so its norm does not exceed ``limit`` (differentiable, branch-free)."""
    return v * (limit / wp.max(wp.length(v), limit))


@wp.func
def _integrate_agent(
    # cell + loaded state
    e: wp.int32,
    a: wp.int32,
    p: Any,
    th: Any,
    v: Any,
    s: Any,
    w: Any,
    # inputs
    a0: Any,
    a1: Any,
    tag: wp.int32,
    mode: wp.int32,
    force: Any,
    # per-agent parameters, pre-loaded from either the shared or per-env layout
    mass: Any,
    max_speed: Any,
    max_accel: Any,
    max_ang_vel: Any,
    max_ang_accel: Any,
    l_f: Any,
    l_r: Any,
    max_steer: Any,
    radius: Any,
    dt: Any,
    bounds_min: Any,
    bounds_max: Any,
    clamp_bounds: wp.int32,
    # state out
    pos_out: wp.array2d(dtype=Any),
    theta_out: wp.array2d(dtype=Any),
    vel_out: wp.array2d(dtype=Any),
    speed_out: wp.array2d(dtype=Any),
    ang_vel_out: wp.array2d(dtype=Any),
):
    """The one-agent semi-implicit-Euler recurrence, shared by the shared-param
    and per-env kernels. It takes the parameters as scalars so the *only*
    difference between the two kernels is how those scalars are indexed out of
    the ``[n_agents, P]`` vs ``[n_envs, n_agents, P]`` layout."""
    zero = type(dt)(0.0)
    dv_f = force * (dt / mass)
    act01 = type(p)(a0, a1)

    if tag == TAG_HOLONOMIC:
        if mode == MODE_VELOCITY:
            v_new = clamp_norm(act01, max_speed)
        else:
            acc_vec = clamp_norm(act01, max_accel)
            v_new = clamp_norm(v + acc_vec * dt, max_speed)
        v_new = v_new + dv_f
        p_new = p + v_new * dt
        th_new = th
        v_out = v_new
        s_out = wp.length(v_new)
        w_out = zero
    elif tag == TAG_DIFF_DRIVE:
        if mode == MODE_VELOCITY:
            s_new = wp.clamp(a0, -max_speed, max_speed)
            w_new = wp.clamp(a1, -max_ang_vel, max_ang_vel)
        else:
            acc = wp.clamp(a0, -max_accel, max_accel)
            alp = wp.clamp(a1, -max_ang_accel, max_ang_accel)
            s_new = wp.clamp(s + acc * dt, -max_speed, max_speed)
            w_new = wp.clamp(w + alp * dt, -max_ang_vel, max_ang_vel)
        v_trans = type(p)(s_new * wp.cos(th), s_new * wp.sin(th)) + dv_f
        p_new = p + v_trans * dt
        th_new = th + w_new * dt
        v_out = v_trans
        s_out = s_new
        w_out = w_new
    elif tag == TAG_BICYCLE:
        # Kinematic bicycle with slip angle beta (Polack et al. 2017, eq. 2).
        acc = wp.clamp(a0, -max_accel, max_accel)
        delta = wp.clamp(a1, -max_steer, max_steer)
        s_new = wp.clamp(s + acc * dt, -max_speed, max_speed)
        wheelbase = l_f + l_r
        beta = wp.atan(wp.tan(delta) * l_r / wheelbase)
        v_trans = type(p)(s_new * wp.cos(th + beta), s_new * wp.sin(th + beta)) + dv_f
        w_new = s_new / wheelbase * wp.cos(beta) * wp.tan(delta)
        p_new = p + v_trans * dt
        th_new = th + w_new * dt
        v_out = v_trans
        s_out = s_new
        w_out = w_new
    else:
        # Unknown tag (e.g. future models): hold state.
        p_new = p
        th_new = th
        v_out = v
        s_out = s
        w_out = w

    if clamp_bounds == 1:
        p_new = type(p)(
            wp.clamp(p_new[0], bounds_min[0] + radius, bounds_max[0] - radius),
            wp.clamp(p_new[1], bounds_min[1] + radius, bounds_max[1] - radius),
        )

    pos_out[e, a] = p_new
    theta_out[e, a] = th_new
    vel_out[e, a] = v_out
    speed_out[e, a] = s_out
    ang_vel_out[e, a] = w_out


@wp.func
def _deriv(
    # state (velocity-level fields hold the current instantaneous values)
    p: Any,
    th: Any,
    v: Any,
    s: Any,
    w: Any,
    # action + model
    a0: Any,
    a1: Any,
    tag: wp.int32,
    mode: wp.int32,
    max_accel: Any,
    max_ang_accel: Any,
    max_steer: Any,
    l_f: Any,
    l_r: Any,
):
    """Pure continuous-time state derivative ``(dpos, dtheta, dvel, dspeed, dang_vel)``.

    Shared by RK4; the semi-implicit Euler path keeps its own fused update. Only
    the *independent* integrable states per model carry a nonzero derivative:
    holonomic integrates ``(pos, vel)``; diff-drive ``(pos, theta, speed,
    ang_vel)``; bicycle ``(pos, theta, speed)``. Derived quantities (``vel`` for
    the nonholonomic models, ``speed`` for holonomic, ``ang_vel`` for the
    bicycle) never feed back into any derivative, so they are integrated with a
    zero rate here and reconstructed after the RK4 combine. In velocity-control
    mode the velocity-level state is a constant input (its derivative is zero);
    the caller sets it before integrating.
    """
    zero = type(s)(0.0)
    zvec = type(p)(zero, zero)
    if tag == TAG_HOLONOMIC:
        dp = v
        dth = zero
        if mode == MODE_VELOCITY:
            dv = zvec
        else:
            dv = clamp_norm(type(p)(a0, a1), max_accel)
        ds = zero
        dw = zero
    elif tag == TAG_DIFF_DRIVE:
        dp = type(p)(s * wp.cos(th), s * wp.sin(th))
        dth = w
        dv = zvec
        if mode == MODE_VELOCITY:
            ds = zero
            dw = zero
        else:
            ds = wp.clamp(a0, -max_accel, max_accel)
            dw = wp.clamp(a1, -max_ang_accel, max_ang_accel)
    elif tag == TAG_BICYCLE:
        wheelbase = l_f + l_r
        delta = wp.clamp(a1, -max_steer, max_steer)
        beta = wp.atan(wp.tan(delta) * l_r / wheelbase)
        dp = type(p)(s * wp.cos(th + beta), s * wp.sin(th + beta))
        dth = s / wheelbase * wp.cos(beta) * wp.tan(delta)
        dv = zvec
        ds = wp.clamp(a0, -max_accel, max_accel)
        dw = zero
    else:
        dp = zvec
        dth = zero
        dv = zvec
        ds = zero
        dw = zero
    return dp, dth, dv, ds, dw


@wp.func
def _integrate_rk4(
    e: wp.int32,
    a: wp.int32,
    p: Any,
    th: Any,
    v: Any,
    s: Any,
    w: Any,
    a0: Any,
    a1: Any,
    tag: wp.int32,
    mode: wp.int32,
    force: Any,
    mass: Any,
    max_speed: Any,
    max_accel: Any,
    max_ang_vel: Any,
    max_ang_accel: Any,
    l_f: Any,
    l_r: Any,
    max_steer: Any,
    radius: Any,
    dt: Any,
    bounds_min: Any,
    bounds_max: Any,
    clamp_bounds: wp.int32,
    pos_out: wp.array2d(dtype=Any),
    theta_out: wp.array2d(dtype=Any),
    vel_out: wp.array2d(dtype=Any),
    speed_out: wp.array2d(dtype=Any),
    ang_vel_out: wp.array2d(dtype=Any),
):
    """Classic RK4 over the model's independent state, four :func:`_deriv` calls.

    Signature matches :func:`_integrate_agent` so the two kernels select between
    them on the ``integrator`` tag. Actions are held constant across the four
    stages; the actuation limits are enforced on the combined result (staying in
    the unsaturated regime keeps the ~dt^4 convergence a clean test can measure).
    Collision/boundary ``force`` enters as the same ``F/m`` velocity contribution
    the Euler path uses.
    """
    zero = type(dt)(0.0)
    half = type(dt)(0.5)
    two = type(dt)(2.0)
    dt2 = dt * half
    sixth = dt / type(dt)(6.0)

    # Velocity-control commands set the velocity-level state directly (constant
    # input across the step); acceleration-control keeps the current state.
    if mode == MODE_VELOCITY:
        if tag == TAG_HOLONOMIC:
            v = clamp_norm(type(p)(a0, a1), max_speed)
        elif tag == TAG_DIFF_DRIVE:
            s = wp.clamp(a0, -max_speed, max_speed)
            w = wp.clamp(a1, -max_ang_vel, max_ang_vel)

    k1p, k1th, k1v, k1s, k1w = _deriv(
        p, th, v, s, w, a0, a1, tag, mode, max_accel, max_ang_accel, max_steer, l_f, l_r
    )
    k2p, k2th, k2v, k2s, k2w = _deriv(
        p + k1p * dt2,
        th + k1th * dt2,
        v + k1v * dt2,
        s + k1s * dt2,
        w + k1w * dt2,
        a0,
        a1,
        tag,
        mode,
        max_accel,
        max_ang_accel,
        max_steer,
        l_f,
        l_r,
    )
    k3p, k3th, k3v, k3s, k3w = _deriv(
        p + k2p * dt2,
        th + k2th * dt2,
        v + k2v * dt2,
        s + k2s * dt2,
        w + k2w * dt2,
        a0,
        a1,
        tag,
        mode,
        max_accel,
        max_ang_accel,
        max_steer,
        l_f,
        l_r,
    )
    k4p, k4th, k4v, k4s, k4w = _deriv(
        p + k3p * dt,
        th + k3th * dt,
        v + k3v * dt,
        s + k3s * dt,
        w + k3w * dt,
        a0,
        a1,
        tag,
        mode,
        max_accel,
        max_ang_accel,
        max_steer,
        l_f,
        l_r,
    )

    p_new = p + (k1p + k2p * two + k3p * two + k4p) * sixth
    th_i = th + (k1th + k2th * two + k3th * two + k4th) * sixth
    v_i = v + (k1v + k2v * two + k3v * two + k4v) * sixth
    s_i = s + (k1s + k2s * two + k3s * two + k4s) * sixth
    w_i = w + (k1w + k2w * two + k3w * two + k4w) * sixth

    dv_f = force * (dt / mass)

    if tag == TAG_HOLONOMIC:
        v_out = clamp_norm(v_i, max_speed) + dv_f
        p_new = p_new + dv_f * dt
        th_new = th
        s_out = wp.length(v_out)
        w_out = zero
    elif tag == TAG_DIFF_DRIVE:
        s_out = wp.clamp(s_i, -max_speed, max_speed)
        w_out = wp.clamp(w_i, -max_ang_vel, max_ang_vel)
        th_new = th_i
        v_out = type(p)(s_out * wp.cos(th_new), s_out * wp.sin(th_new)) + dv_f
        p_new = p_new + dv_f * dt
    elif tag == TAG_BICYCLE:
        s_out = wp.clamp(s_i, -max_speed, max_speed)
        th_new = th_i
        delta = wp.clamp(a1, -max_steer, max_steer)
        wheelbase = l_f + l_r
        beta = wp.atan(wp.tan(delta) * l_r / wheelbase)
        w_out = s_out / wheelbase * wp.cos(beta) * wp.tan(delta)
        v_out = type(p)(s_out * wp.cos(th_new + beta), s_out * wp.sin(th_new + beta)) + dv_f
        p_new = p_new + dv_f * dt
    else:
        p_new = p
        th_new = th
        v_out = v
        s_out = s
        w_out = w

    if clamp_bounds == 1:
        p_new = type(p)(
            wp.clamp(p_new[0], bounds_min[0] + radius, bounds_max[0] - radius),
            wp.clamp(p_new[1], bounds_min[1] + radius, bounds_max[1] - radius),
        )

    pos_out[e, a] = p_new
    theta_out[e, a] = th_new
    vel_out[e, a] = v_out
    speed_out[e, a] = s_out
    ang_vel_out[e, a] = w_out


@wp.kernel
def integrate_kernel(
    # state in
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    speed: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
    # inputs
    actions: wp.array3d(dtype=Any),  # [n_envs, n_agents, act_dim] scalar
    forces: wp.array2d(dtype=Any),
    params: wp.array2d(dtype=Any),  # [n_agents, NUM_PARAMS] shared across envs
    model_tag: wp.array(dtype=wp.int32),
    ctrl_mode: wp.array(dtype=wp.int32),
    dt: Any,
    bounds_min: Any,
    bounds_max: Any,
    clamp_bounds: wp.int32,
    integrator: wp.int32,
    # state out
    pos_out: wp.array2d(dtype=Any),
    theta_out: wp.array2d(dtype=Any),
    vel_out: wp.array2d(dtype=Any),
    speed_out: wp.array2d(dtype=Any),
    ang_vel_out: wp.array2d(dtype=Any),
):
    e, a = wp.tid()
    # Action arity is decoupled from geometry: read the scalar slots each model
    # needs (all current models use 2); a0/a1 are slots 0,1.
    if integrator == INT_RK4:
        _integrate_rk4(
            e,
            a,
            pos[e, a],
            theta[e, a],
            vel[e, a],
            speed[e, a],
            ang_vel[e, a],
            actions[e, a, 0],
            actions[e, a, 1],
            model_tag[a],
            ctrl_mode[a],
            forces[e, a],
            params[a, P_MASS],
            params[a, P_MAX_SPEED],
            params[a, P_MAX_ACCEL],
            params[a, P_MAX_ANG_VEL],
            params[a, P_MAX_ANG_ACCEL],
            params[a, P_LF],
            params[a, P_LR],
            params[a, P_MAX_STEER],
            params[a, P_RADIUS],
            dt,
            bounds_min,
            bounds_max,
            clamp_bounds,
            pos_out,
            theta_out,
            vel_out,
            speed_out,
            ang_vel_out,
        )
    else:
        _integrate_agent(
            e,
            a,
            pos[e, a],
            theta[e, a],
            vel[e, a],
            speed[e, a],
            ang_vel[e, a],
            actions[e, a, 0],
            actions[e, a, 1],
            model_tag[a],
            ctrl_mode[a],
            forces[e, a],
            params[a, P_MASS],
            params[a, P_MAX_SPEED],
            params[a, P_MAX_ACCEL],
            params[a, P_MAX_ANG_VEL],
            params[a, P_MAX_ANG_ACCEL],
            params[a, P_LF],
            params[a, P_LR],
            params[a, P_MAX_STEER],
            params[a, P_RADIUS],
            dt,
            bounds_min,
            bounds_max,
            clamp_bounds,
            pos_out,
            theta_out,
            vel_out,
            speed_out,
            ang_vel_out,
        )


@wp.kernel
def integrate_kernel_per_env(
    # state in
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    speed: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
    # inputs
    actions: wp.array3d(dtype=Any),  # [n_envs, n_agents, act_dim] scalar
    forces: wp.array2d(dtype=Any),
    params: wp.array3d(dtype=Any),  # [n_envs, n_agents, NUM_PARAMS] per-env
    model_tag: wp.array(dtype=wp.int32),
    ctrl_mode: wp.array(dtype=wp.int32),
    dt: Any,
    bounds_min: Any,
    bounds_max: Any,
    clamp_bounds: wp.int32,
    integrator: wp.int32,
    # state out
    pos_out: wp.array2d(dtype=Any),
    theta_out: wp.array2d(dtype=Any),
    vel_out: wp.array2d(dtype=Any),
    speed_out: wp.array2d(dtype=Any),
    ang_vel_out: wp.array2d(dtype=Any),
):
    e, a = wp.tid()
    if integrator == INT_RK4:
        _integrate_rk4(
            e,
            a,
            pos[e, a],
            theta[e, a],
            vel[e, a],
            speed[e, a],
            ang_vel[e, a],
            actions[e, a, 0],
            actions[e, a, 1],
            model_tag[a],
            ctrl_mode[a],
            forces[e, a],
            params[e, a, P_MASS],
            params[e, a, P_MAX_SPEED],
            params[e, a, P_MAX_ACCEL],
            params[e, a, P_MAX_ANG_VEL],
            params[e, a, P_MAX_ANG_ACCEL],
            params[e, a, P_LF],
            params[e, a, P_LR],
            params[e, a, P_MAX_STEER],
            params[e, a, P_RADIUS],
            dt,
            bounds_min,
            bounds_max,
            clamp_bounds,
            pos_out,
            theta_out,
            vel_out,
            speed_out,
            ang_vel_out,
        )
    else:
        _integrate_agent(
            e,
            a,
            pos[e, a],
            theta[e, a],
            vel[e, a],
            speed[e, a],
            ang_vel[e, a],
            actions[e, a, 0],
            actions[e, a, 1],
            model_tag[a],
            ctrl_mode[a],
            forces[e, a],
            params[e, a, P_MASS],
            params[e, a, P_MAX_SPEED],
            params[e, a, P_MAX_ACCEL],
            params[e, a, P_MAX_ANG_VEL],
            params[e, a, P_MAX_ANG_ACCEL],
            params[e, a, P_LF],
            params[e, a, P_LR],
            params[e, a, P_MAX_STEER],
            params[e, a, P_RADIUS],
            dt,
            bounds_min,
            bounds_max,
            clamp_bounds,
            pos_out,
            theta_out,
            vel_out,
            speed_out,
            ang_vel_out,
        )


def _signature(dtype, per_env: bool = False) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    a2s = wp.array2d(dtype=dtype)
    a3s = wp.array3d(dtype=dtype)  # actions: [n_envs, n_agents, act_dim] scalar
    a1i = wp.array(dtype=wp.int32)
    params = wp.array3d(dtype=dtype) if per_env else a2s
    return [
        a2v,
        a2s,
        a2v,
        a2s,
        a2s,  # state in
        a3s,
        a2v,
        params,
        a1i,
        a1i,
        dtype,  # actions, forces, params, tags, modes, dt
        vec2,
        vec2,
        wp.int32,
        wp.int32,  # bounds_min, bounds_max, clamp_bounds, integrator
        a2v,
        a2s,
        a2v,
        a2s,
        a2s,  # state out
    ]


for _T in (wp.float32, wp.float64):
    wp.overload(integrate_kernel, _signature(_T))
    wp.overload(integrate_kernel_per_env, _signature(_T, per_env=True))


#: Integrator enum value -> kernel tag (must match the INT_* constants above).
_INTEGRATOR_TAG = {Integrator.EULER: 0, Integrator.RK4: 1}


def launch_integrate(
    state_in: WorldState,
    state_out: WorldState,
    actions: wp.array,
    forces: wp.array,
    params: AgentParams,
    dt: float,
    clamp_bounds: tuple[float, float, float, float] | None = None,
    integrator: Integrator = Integrator.EULER,
) -> None:
    """Launch one integration sub-step. Functional: never writes to ``state_in``.

    ``clamp_bounds=(x_min, x_max, y_min, y_max)`` hard-clamps positions inside
    the rectangle (inset by each agent's radius). ``integrator`` selects the
    semi-implicit Euler recurrence (default) or classic RK4.
    """
    dtype = state_in.theta.dtype
    vec2 = VEC2[dtype]
    n_envs, n_agents = state_in.pos.shape
    if clamp_bounds is None:
        b_min, b_max, do_clamp = vec2(0.0, 0.0), vec2(0.0, 0.0), 0
    else:
        x_min, x_max, y_min, y_max = clamp_bounds
        b_min, b_max, do_clamp = vec2(x_min, y_min), vec2(x_max, y_max), 1
    if params.floats_per_env is None:
        kernel, floats = integrate_kernel, params.floats
    else:
        kernel, floats = integrate_kernel_per_env, params.floats_per_env
    wp.launch(
        kernel,
        dim=(n_envs, n_agents),
        inputs=[
            *state_in.arrays(),
            actions,
            forces,
            floats,
            params.model_tag,
            params.ctrl_mode,
            dtype(dt),
            b_min,
            b_max,
            wp.int32(do_clamp),
            wp.int32(_INTEGRATOR_TAG[integrator]),
        ],
        outputs=state_out.arrays(),
        device=state_in.pos.device,
    )
