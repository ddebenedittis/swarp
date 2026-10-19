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

from swarp._overloads import concrete, register
from swarp.core.state import QUAT, VEC2, VEC3, WorldState
from swarp.dynamics.base import (
    P_ARM,
    P_CMD_GAIN_V,
    P_CMD_GAIN_W,
    P_GRAVITY,
    P_HALF_X,
    P_HALF_Y,
    P_IXX,
    P_IYY,
    P_IZZ,
    P_KAPPA,
    P_LF,
    P_LR,
    P_MASS,
    P_MAX_ACCEL,
    P_MAX_ANG_ACCEL,
    P_MAX_ANG_VEL,
    P_MAX_SPEED,
    P_MAX_STEER,
    P_MIN_SPEED,
    P_RADIUS,
    P_SHAPE,
    P_THRUST_MAX,
    AgentParams,
    ControlMode,
    DynamicsModel,
    Integrator,
)

# Lean launches (no grid-stride loop); see docs/performance.md.
wp.set_module_options({"default_grid_stride": False})

TAG_HOLONOMIC = wp.constant(int(DynamicsModel.HOLONOMIC))
TAG_DIFF_DRIVE = wp.constant(int(DynamicsModel.DIFF_DRIVE))
TAG_BICYCLE = wp.constant(int(DynamicsModel.KINEMATIC_BICYCLE))
TAG_DRONE = wp.constant(int(DynamicsModel.DRONE))
MODE_VELOCITY = wp.constant(0)
MODE_SERVO = wp.constant(int(ControlMode.SERVO))

# Integrator tags (must match Integrator enum order in swarp.dynamics.base).
INT_EULER = wp.constant(0)
INT_RK4 = wp.constant(1)


@wp.func
def clamp_norm(v: Any, limit: Any):
    """Scale ``v`` so its norm does not exceed ``limit`` (differentiable, branch-free)."""
    return v * (limit / wp.max(wp.length(v), limit))


#: ``clamp_bounds`` tag of the footprint clamp, which a world holding box agents uses
#: (``0`` is off and ``1`` the disc clamp, inset by the radius).
CLAMP_FOOTPRINT = wp.constant(2)


@wp.func
def clamp_footprint(
    p: Any, th: Any, radius: Any, shape: Any, hx: Any, hy: Any, b_min: Any, b_max: Any
):
    """Hard-clamp a position so the agent's whole footprint stays inside the bounds.

    A disc (``shape`` 0) is inset by its radius, the same arithmetic as the disc clamp. A
    box at heading ``th`` is inset by its support extents ``ex = hx|cos| + hy|sin|`` and
    ``ey = hx|sin| + hy|cos|``, so a box turned 45 degrees is held off the wall by its
    corner rather than by its circumscribed radius."""
    ex = radius
    ey = radius
    if shape > type(radius)(0.5):
        ca = wp.abs(wp.cos(th))
        sa = wp.abs(wp.sin(th))
        ex = hx * ca + hy * sa
        ey = hx * sa + hy * ca
    return type(p)(
        wp.clamp(p[0], b_min[0] + ex, b_max[0] - ex),
        wp.clamp(p[1], b_min[1] + ey, b_max[1] - ey),
    )


# ------------------------------------------------------------------ drone (6-DOF)
# Quaternion convention: (x, y, z, w), body->world. The rotation and quaternion
# kinematics are written out explicitly (not via wp.quat_rotate) so a numpy
# reference can reproduce the exact arithmetic for bit-level trajectory tests.


@wp.func
def _quat_rotate_vec(q: Any, v: Any):
    """Rotate vec3 ``v`` by quaternion ``q`` = (x, y, z, w)."""
    u = type(v)(q[0], q[1], q[2])
    two = type(q[3])(2.0)
    t = two * wp.cross(u, v)
    return v + q[3] * t + wp.cross(u, t)


@wp.func
def _quat_deriv(q: Any, wx: Any, wy: Any, wz: Any):
    """Quaternion time-derivative ``0.5 * q (x) (wx, wy, wz, 0)`` (Hamilton)."""
    half = type(wx)(0.5)
    return type(q)(
        half * (q[3] * wx + q[1] * wz - q[2] * wy),
        half * (q[3] * wy - q[0] * wz + q[2] * wx),
        half * (q[3] * wz + q[0] * wy - q[1] * wx),
        half * (-q[0] * wx - q[1] * wy - q[2] * wz),
    )


@wp.func
def _drone_rates(
    # 4 clamped rotor thrusts + geometry/inertia
    f0: Any,
    f1: Any,
    f2: Any,
    f3: Any,
    q: Any,
    br: Any,
    mass: Any,
    arm: Any,
    ixx: Any,
    iyy: Any,
    izz: Any,
    kappa: Any,
    gravity: Any,
):
    """Continuous 6-DOF quadrotor rates: returns (accel_world vec3, ang_accel vec3).

    "+" rotor layout: motors 0/2 on the body x-axis, 1/3 on the y-axis. Total
    thrust acts along body +z; yaw torque is the rotor reaction sum.
    """
    thrust = f0 + f1 + f2 + f3
    tvec = type(br)(type(thrust)(0.0), type(thrust)(0.0), thrust)
    f_world = _quat_rotate_vec(q, tvec)
    accel = f_world / mass - type(br)(type(mass)(0.0), type(mass)(0.0), gravity)

    tx = arm * (f1 - f3)
    ty = arm * (f2 - f0)
    tz = kappa * (f0 - f1 + f2 - f3)
    # Euler's rigid-body equation with diagonal inertia (body frame).
    ax = (tx - (izz - iyy) * br[1] * br[2]) / ixx
    ay = (ty - (ixx - izz) * br[2] * br[0]) / iyy
    az = (tz - (iyy - ixx) * br[0] * br[1]) / izz
    return accel, type(br)(ax, ay, az)


@wp.func
def servo_command(
    a0: Any,
    a1: Any,
    s: Any,
    w: Any,
    max_speed: Any,
    max_ang_vel: Any,
    max_accel: Any,
    max_ang_accel: Any,
    gain_v: Any,
    gain_w: Any,
    dt: Any,
):
    """``ControlMode.SERVO`` as the ``VELOCITY`` command it amounts to this substep.

    The diff-drive's speed ``s`` and turn rate ``w`` (the drive's own state, which the
    contact kick never enters) move toward the gained command by at most
    ``max_accel * dt`` and ``max_ang_accel * dt``. The integrate kernels call this on a
    ``SERVO`` agent only and then run the ``VELOCITY`` branch on the result, so no other
    mode loads the gain or limit columns."""
    tv = wp.clamp(a0, -max_speed, max_speed) * gain_v
    tw = wp.clamp(a1, -max_ang_vel, max_ang_vel) * gain_w
    dv = max_accel * dt
    dw = max_ang_accel * dt
    return s + wp.clamp(tv - s, -dv, dv), w + wp.clamp(tw - w, -dw, dw)


@wp.func
def _step_2d(
    # loaded 2D state
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
    # per-agent parameters (scalars)
    max_speed: Any,
    min_speed: Any,
    max_accel: Any,
    max_ang_vel: Any,
    max_ang_accel: Any,
    l_f: Any,
    l_r: Any,
    max_steer: Any,
    mass: Any,
    dt: Any,
):
    """The 2D semi-implicit-Euler recurrence for the holonomic, diff-drive, and
    bicycle models. Returns ``(p_new, th_new, v_out, s_out, w_out)`` *before* any
    position clamp (the caller applies bounds). Extracted verbatim from
    :func:`_integrate_agent` so the slim 2D kernel and the full kernel produce
    bit-identical results; unknown tags hold state."""
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
        s_new = wp.clamp(s + acc * dt, min_speed, max_speed)
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
        p_new = p
        th_new = th
        v_out = v
        s_out = s
        w_out = w
    return p_new, th_new, v_out, s_out, w_out


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
    z: Any,
    vz: Any,
    q: Any,
    br: Any,
    # inputs (up to 4 action slots; drone uses all four rotor commands)
    a0: Any,
    a1: Any,
    a2: Any,
    a3: Any,
    tag: wp.int32,
    mode: wp.int32,
    force: Any,
    # per-agent parameters, pre-loaded from either the shared or per-env layout
    mass: Any,
    max_speed: Any,
    min_speed: Any,
    max_accel: Any,
    max_ang_vel: Any,
    max_ang_accel: Any,
    l_f: Any,
    l_r: Any,
    max_steer: Any,
    radius: Any,
    shape: Any,
    half_x: Any,
    half_y: Any,
    thrust_max: Any,
    arm: Any,
    ixx: Any,
    iyy: Any,
    izz: Any,
    kappa: Any,
    gravity: Any,
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
    z_out: wp.array2d(dtype=Any),
    vz_out: wp.array2d(dtype=Any),
    attitude_out: wp.array2d(dtype=Any),
    body_rates_out: wp.array2d(dtype=Any),
):
    """The one-agent semi-implicit-Euler recurrence, shared by the shared-param
    and per-env kernels. It takes the parameters as scalars so the *only*
    difference between the two kernels is how those scalars are indexed out of
    the ``[n_agents, P]`` vs ``[n_envs, n_agents, P]`` layout. The 2D branches
    live in :func:`_step_2d` (also used by the slim kernel); this kernel adds the
    6-DOF drone branch and passes the drone state through unchanged for 2D."""
    zero = type(dt)(0.0)
    # Drone fields default to passthrough; only the DRONE branch updates them.
    z_new = z
    vz_new = vz
    q_new = q
    br_new = br

    if tag == TAG_DRONE:
        dv_f = force * (dt / mass)
        # Semi-implicit Euler: update body rates + linear velocity, then advance
        # attitude (with the new rates) and position (with the new velocities).
        f0 = wp.clamp(a0, zero, thrust_max)
        f1 = wp.clamp(a1, zero, thrust_max)
        f2 = wp.clamp(a2, zero, thrust_max)
        f3 = wp.clamp(a3, zero, thrust_max)
        accel, angacc = _drone_rates(
            f0, f1, f2, f3, q, br, mass, arm, ixx, iyy, izz, kappa, gravity
        )
        br_new = br + angacc * dt
        vz_new = vz + accel[2] * dt
        v_xy = v + type(p)(accel[0], accel[1]) * dt + dv_f
        q_new = wp.normalize(q + _quat_deriv(q, br_new[0], br_new[1], br_new[2]) * dt)
        p_new = p + v_xy * dt
        z_new = z + vz_new * dt
        th_new = th
        v_out = v_xy
        s_out = s
        w_out = w
    else:
        # 2D models (and unknown tags -> hold state); drone fields pass through.
        p_new, th_new, v_out, s_out, w_out = _step_2d(
            p,
            th,
            v,
            s,
            w,
            a0,
            a1,
            tag,
            mode,
            force,
            max_speed,
            min_speed,
            max_accel,
            max_ang_vel,
            max_ang_accel,
            l_f,
            l_r,
            max_steer,
            mass,
            dt,
        )

    if clamp_bounds == 1:
        p_new = type(p)(
            wp.clamp(p_new[0], bounds_min[0] + radius, bounds_max[0] - radius),
            wp.clamp(p_new[1], bounds_min[1] + radius, bounds_max[1] - radius),
        )
    elif clamp_bounds == CLAMP_FOOTPRINT:
        p_new = clamp_footprint(
            p_new, th_new, radius, shape, half_x, half_y, bounds_min, bounds_max
        )

    pos_out[e, a] = p_new
    theta_out[e, a] = th_new
    vel_out[e, a] = v_out
    speed_out[e, a] = s_out
    ang_vel_out[e, a] = w_out
    z_out[e, a] = z_new
    vz_out[e, a] = vz_new
    attitude_out[e, a] = q_new
    body_rates_out[e, a] = br_new


@wp.func
def _deriv(
    # state (velocity-level fields hold the current instantaneous values)
    p: Any,
    th: Any,
    v: Any,
    s: Any,
    w: Any,
    z: Any,
    vz: Any,
    q: Any,
    br: Any,
    # action + model
    a0: Any,
    a1: Any,
    a2: Any,
    a3: Any,
    tag: wp.int32,
    mode: wp.int32,
    mass: Any,
    max_accel: Any,
    max_ang_accel: Any,
    max_steer: Any,
    l_f: Any,
    l_r: Any,
    thrust_max: Any,
    arm: Any,
    ixx: Any,
    iyy: Any,
    izz: Any,
    kappa: Any,
    gravity: Any,
):
    """Pure continuous-time state derivative of all nine state fields.

    Shared by RK4; the semi-implicit Euler path keeps its own fused update. For
    the 2D models only the *independent* integrable states carry a nonzero
    derivative: holonomic integrates ``(pos, vel)``; diff-drive ``(pos, theta,
    speed, ang_vel)``; bicycle ``(pos, theta, speed)`` — derived quantities are
    zero-rate here and reconstructed after the RK4 combine. The drone integrates
    ``(pos_xy, z, vel_xy, vz, attitude, body_rates)`` and leaves theta/speed/
    ang_vel at zero rate. In velocity-control mode the 2D velocity-level state is
    a constant input; the caller sets it before integrating.
    """
    zero = type(s)(0.0)
    zvec = type(p)(zero, zero)
    dz = zero
    dvz = zero
    dq = type(q)(zero, zero, zero, zero)
    dbr = type(br)(zero, zero, zero)
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
    elif tag == TAG_DRONE:
        f0 = wp.clamp(a0, zero, thrust_max)
        f1 = wp.clamp(a1, zero, thrust_max)
        f2 = wp.clamp(a2, zero, thrust_max)
        f3 = wp.clamp(a3, zero, thrust_max)
        accel, angacc = _drone_rates(
            f0, f1, f2, f3, q, br, mass, arm, ixx, iyy, izz, kappa, gravity
        )
        dp = v
        dth = zero
        dv = type(p)(accel[0], accel[1])
        ds = zero
        dw = zero
        dz = vz
        dvz = accel[2]
        dq = _quat_deriv(q, br[0], br[1], br[2])
        dbr = angacc
    else:
        dp = zvec
        dth = zero
        dv = zvec
        ds = zero
        dw = zero
    return dp, dth, dv, ds, dw, dz, dvz, dq, dbr


@wp.func
def _integrate_rk4(
    e: wp.int32,
    a: wp.int32,
    p: Any,
    th: Any,
    v: Any,
    s: Any,
    w: Any,
    z: Any,
    vz: Any,
    q: Any,
    br: Any,
    a0: Any,
    a1: Any,
    a2: Any,
    a3: Any,
    tag: wp.int32,
    mode: wp.int32,
    force: Any,
    mass: Any,
    max_speed: Any,
    min_speed: Any,
    max_accel: Any,
    max_ang_vel: Any,
    max_ang_accel: Any,
    l_f: Any,
    l_r: Any,
    max_steer: Any,
    radius: Any,
    shape: Any,
    half_x: Any,
    half_y: Any,
    thrust_max: Any,
    arm: Any,
    ixx: Any,
    iyy: Any,
    izz: Any,
    kappa: Any,
    gravity: Any,
    dt: Any,
    bounds_min: Any,
    bounds_max: Any,
    clamp_bounds: wp.int32,
    pos_out: wp.array2d(dtype=Any),
    theta_out: wp.array2d(dtype=Any),
    vel_out: wp.array2d(dtype=Any),
    speed_out: wp.array2d(dtype=Any),
    ang_vel_out: wp.array2d(dtype=Any),
    z_out: wp.array2d(dtype=Any),
    vz_out: wp.array2d(dtype=Any),
    attitude_out: wp.array2d(dtype=Any),
    body_rates_out: wp.array2d(dtype=Any),
):
    """Classic RK4 over the model's independent state, four :func:`_deriv` calls.

    Signature matches :func:`_integrate_agent` so the two kernels select between
    them on the ``integrator`` tag. Actions are held constant across the four
    stages; the actuation limits are enforced on the combined result (staying in
    the unsaturated regime keeps the ~dt^4 convergence a clean test can measure).
    Collision/boundary ``force`` enters as the same ``F/m`` velocity contribution
    the Euler path uses. The drone integrates its full 6-DOF state and the
    attitude quaternion is renormalized after the combine.
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

    k1p, k1th, k1v, k1s, k1w, k1z, k1vz, k1q, k1br = _deriv(
        p,
        th,
        v,
        s,
        w,
        z,
        vz,
        q,
        br,
        a0,
        a1,
        a2,
        a3,
        tag,
        mode,
        mass,
        max_accel,
        max_ang_accel,
        max_steer,
        l_f,
        l_r,
        thrust_max,
        arm,
        ixx,
        iyy,
        izz,
        kappa,
        gravity,
    )
    k2p, k2th, k2v, k2s, k2w, k2z, k2vz, k2q, k2br = _deriv(
        p + k1p * dt2,
        th + k1th * dt2,
        v + k1v * dt2,
        s + k1s * dt2,
        w + k1w * dt2,
        z + k1z * dt2,
        vz + k1vz * dt2,
        q + k1q * dt2,
        br + k1br * dt2,
        a0,
        a1,
        a2,
        a3,
        tag,
        mode,
        mass,
        max_accel,
        max_ang_accel,
        max_steer,
        l_f,
        l_r,
        thrust_max,
        arm,
        ixx,
        iyy,
        izz,
        kappa,
        gravity,
    )
    k3p, k3th, k3v, k3s, k3w, k3z, k3vz, k3q, k3br = _deriv(
        p + k2p * dt2,
        th + k2th * dt2,
        v + k2v * dt2,
        s + k2s * dt2,
        w + k2w * dt2,
        z + k2z * dt2,
        vz + k2vz * dt2,
        q + k2q * dt2,
        br + k2br * dt2,
        a0,
        a1,
        a2,
        a3,
        tag,
        mode,
        mass,
        max_accel,
        max_ang_accel,
        max_steer,
        l_f,
        l_r,
        thrust_max,
        arm,
        ixx,
        iyy,
        izz,
        kappa,
        gravity,
    )
    k4p, k4th, k4v, k4s, k4w, k4z, k4vz, k4q, k4br = _deriv(
        p + k3p * dt,
        th + k3th * dt,
        v + k3v * dt,
        s + k3s * dt,
        w + k3w * dt,
        z + k3z * dt,
        vz + k3vz * dt,
        q + k3q * dt,
        br + k3br * dt,
        a0,
        a1,
        a2,
        a3,
        tag,
        mode,
        mass,
        max_accel,
        max_ang_accel,
        max_steer,
        l_f,
        l_r,
        thrust_max,
        arm,
        ixx,
        iyy,
        izz,
        kappa,
        gravity,
    )

    p_new = p + (k1p + k2p * two + k3p * two + k4p) * sixth
    th_i = th + (k1th + k2th * two + k3th * two + k4th) * sixth
    v_i = v + (k1v + k2v * two + k3v * two + k4v) * sixth
    s_i = s + (k1s + k2s * two + k3s * two + k4s) * sixth
    w_i = w + (k1w + k2w * two + k3w * two + k4w) * sixth
    z_i = z + (k1z + k2z * two + k3z * two + k4z) * sixth
    vz_i = vz + (k1vz + k2vz * two + k3vz * two + k4vz) * sixth
    q_i = q + (k1q + k2q * two + k3q * two + k4q) * sixth
    br_i = br + (k1br + k2br * two + k3br * two + k4br) * sixth

    dv_f = force * (dt / mass)
    # Drone fields default to passthrough (2D models never touch them).
    z_new = z
    vz_new = vz
    q_new = q
    br_new = br

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
        s_out = wp.clamp(s_i, min_speed, max_speed)
        th_new = th_i
        delta = wp.clamp(a1, -max_steer, max_steer)
        wheelbase = l_f + l_r
        beta = wp.atan(wp.tan(delta) * l_r / wheelbase)
        w_out = s_out / wheelbase * wp.cos(beta) * wp.tan(delta)
        v_out = type(p)(s_out * wp.cos(th_new + beta), s_out * wp.sin(th_new + beta)) + dv_f
        p_new = p_new + dv_f * dt
    elif tag == TAG_DRONE:
        th_new = th
        s_out = s
        w_out = w
        v_out = v_i + dv_f
        p_new = p_new + dv_f * dt
        z_new = z_i
        vz_new = vz_i
        q_new = wp.normalize(q_i)
        br_new = br_i
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
    elif clamp_bounds == CLAMP_FOOTPRINT:
        p_new = clamp_footprint(
            p_new, th_new, radius, shape, half_x, half_y, bounds_min, bounds_max
        )

    pos_out[e, a] = p_new
    theta_out[e, a] = th_new
    vel_out[e, a] = v_out
    speed_out[e, a] = s_out
    ang_vel_out[e, a] = w_out
    z_out[e, a] = z_new
    vz_out[e, a] = vz_new
    attitude_out[e, a] = q_new
    body_rates_out[e, a] = br_new


@wp.func
def _load_actions(actions: wp.array3d(dtype=Any), e: wp.int32, a: wp.int32):
    """Load up to four action slots, zero-filling any the array does not carry
    (so 2D worlds with act_dim < 4 never read out of bounds)."""
    a0 = actions[e, a, 0]
    a1 = actions[e, a, 1]
    az = type(a0)(0.0)
    a2 = az
    a3 = az
    if actions.shape[2] > 2:
        a2 = actions[e, a, 2]
    if actions.shape[2] > 3:
        a3 = actions[e, a, 3]
    return a0, a1, a2, a3


@wp.kernel
def integrate_kernel(
    # state in
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    speed: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
    z: wp.array2d(dtype=Any),
    vz: wp.array2d(dtype=Any),
    attitude: wp.array2d(dtype=Any),
    body_rates: wp.array2d(dtype=Any),
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
    z_out: wp.array2d(dtype=Any),
    vz_out: wp.array2d(dtype=Any),
    attitude_out: wp.array2d(dtype=Any),
    body_rates_out: wp.array2d(dtype=Any),
):
    e, a = wp.tid()
    a0, a1, a2, a3 = _load_actions(actions, e, a)
    mode = ctrl_mode[a]
    if mode == MODE_SERVO:
        a0, a1 = servo_command(
            a0,
            a1,
            speed[e, a],
            ang_vel[e, a],
            params[a, P_MAX_SPEED],
            params[a, P_MAX_ANG_VEL],
            params[a, P_MAX_ACCEL],
            params[a, P_MAX_ANG_ACCEL],
            params[a, P_CMD_GAIN_V],
            params[a, P_CMD_GAIN_W],
            dt,
        )
        mode = MODE_VELOCITY
    if integrator == INT_RK4:
        _integrate_rk4(
            e,
            a,
            pos[e, a],
            theta[e, a],
            vel[e, a],
            speed[e, a],
            ang_vel[e, a],
            z[e, a],
            vz[e, a],
            attitude[e, a],
            body_rates[e, a],
            a0,
            a1,
            a2,
            a3,
            model_tag[a],
            mode,
            forces[e, a],
            params[a, P_MASS],
            params[a, P_MAX_SPEED],
            params[a, P_MIN_SPEED],
            params[a, P_MAX_ACCEL],
            params[a, P_MAX_ANG_VEL],
            params[a, P_MAX_ANG_ACCEL],
            params[a, P_LF],
            params[a, P_LR],
            params[a, P_MAX_STEER],
            params[a, P_RADIUS],
            params[a, P_SHAPE],
            params[a, P_HALF_X],
            params[a, P_HALF_Y],
            params[a, P_THRUST_MAX],
            params[a, P_ARM],
            params[a, P_IXX],
            params[a, P_IYY],
            params[a, P_IZZ],
            params[a, P_KAPPA],
            params[a, P_GRAVITY],
            dt,
            bounds_min,
            bounds_max,
            clamp_bounds,
            pos_out,
            theta_out,
            vel_out,
            speed_out,
            ang_vel_out,
            z_out,
            vz_out,
            attitude_out,
            body_rates_out,
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
            z[e, a],
            vz[e, a],
            attitude[e, a],
            body_rates[e, a],
            a0,
            a1,
            a2,
            a3,
            model_tag[a],
            mode,
            forces[e, a],
            params[a, P_MASS],
            params[a, P_MAX_SPEED],
            params[a, P_MIN_SPEED],
            params[a, P_MAX_ACCEL],
            params[a, P_MAX_ANG_VEL],
            params[a, P_MAX_ANG_ACCEL],
            params[a, P_LF],
            params[a, P_LR],
            params[a, P_MAX_STEER],
            params[a, P_RADIUS],
            params[a, P_SHAPE],
            params[a, P_HALF_X],
            params[a, P_HALF_Y],
            params[a, P_THRUST_MAX],
            params[a, P_ARM],
            params[a, P_IXX],
            params[a, P_IYY],
            params[a, P_IZZ],
            params[a, P_KAPPA],
            params[a, P_GRAVITY],
            dt,
            bounds_min,
            bounds_max,
            clamp_bounds,
            pos_out,
            theta_out,
            vel_out,
            speed_out,
            ang_vel_out,
            z_out,
            vz_out,
            attitude_out,
            body_rates_out,
        )


# The ``_per_env`` twins below duplicate their base kernel verbatim apart from the
# ``params`` rank and indexing (``params[a, X]`` vs ``params[e, a, X]``). That is
# deliberate, and was measured: collapsing each pair into one kernel taking a 3-D
# ``params`` plus a 0/1 env stride (the shared rows as a zero-copy ``[1, n_agents, P]``
# view, indexed ``params[e * p_stride, a, X]``) is correct and bit-identical, but costs
# ~4% of NavigationScenario throughput at 16k envs x 64 agents — an interleaved A/B on
# an RTX 3070 put the collapsed variant's median below the baseline's worst run. One
# integer multiply per thread is not free on the hottest kernel in the library, and 194
# lines of mechanical duplication are cheaper than that. Keep the twins in sync by hand;
# ``tests/unit/test_per_env_params.py`` exercises both paths against each other.
@wp.kernel
def integrate_kernel_per_env(
    # state in
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    speed: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
    z: wp.array2d(dtype=Any),
    vz: wp.array2d(dtype=Any),
    attitude: wp.array2d(dtype=Any),
    body_rates: wp.array2d(dtype=Any),
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
    z_out: wp.array2d(dtype=Any),
    vz_out: wp.array2d(dtype=Any),
    attitude_out: wp.array2d(dtype=Any),
    body_rates_out: wp.array2d(dtype=Any),
):
    e, a = wp.tid()
    a0, a1, a2, a3 = _load_actions(actions, e, a)
    mode = ctrl_mode[a]
    if mode == MODE_SERVO:
        a0, a1 = servo_command(
            a0,
            a1,
            speed[e, a],
            ang_vel[e, a],
            params[e, a, P_MAX_SPEED],
            params[e, a, P_MAX_ANG_VEL],
            params[e, a, P_MAX_ACCEL],
            params[e, a, P_MAX_ANG_ACCEL],
            params[e, a, P_CMD_GAIN_V],
            params[e, a, P_CMD_GAIN_W],
            dt,
        )
        mode = MODE_VELOCITY
    if integrator == INT_RK4:
        _integrate_rk4(
            e,
            a,
            pos[e, a],
            theta[e, a],
            vel[e, a],
            speed[e, a],
            ang_vel[e, a],
            z[e, a],
            vz[e, a],
            attitude[e, a],
            body_rates[e, a],
            a0,
            a1,
            a2,
            a3,
            model_tag[a],
            mode,
            forces[e, a],
            params[e, a, P_MASS],
            params[e, a, P_MAX_SPEED],
            params[e, a, P_MIN_SPEED],
            params[e, a, P_MAX_ACCEL],
            params[e, a, P_MAX_ANG_VEL],
            params[e, a, P_MAX_ANG_ACCEL],
            params[e, a, P_LF],
            params[e, a, P_LR],
            params[e, a, P_MAX_STEER],
            params[e, a, P_RADIUS],
            params[e, a, P_SHAPE],
            params[e, a, P_HALF_X],
            params[e, a, P_HALF_Y],
            params[e, a, P_THRUST_MAX],
            params[e, a, P_ARM],
            params[e, a, P_IXX],
            params[e, a, P_IYY],
            params[e, a, P_IZZ],
            params[e, a, P_KAPPA],
            params[e, a, P_GRAVITY],
            dt,
            bounds_min,
            bounds_max,
            clamp_bounds,
            pos_out,
            theta_out,
            vel_out,
            speed_out,
            ang_vel_out,
            z_out,
            vz_out,
            attitude_out,
            body_rates_out,
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
            z[e, a],
            vz[e, a],
            attitude[e, a],
            body_rates[e, a],
            a0,
            a1,
            a2,
            a3,
            model_tag[a],
            mode,
            forces[e, a],
            params[e, a, P_MASS],
            params[e, a, P_MAX_SPEED],
            params[e, a, P_MIN_SPEED],
            params[e, a, P_MAX_ACCEL],
            params[e, a, P_MAX_ANG_VEL],
            params[e, a, P_MAX_ANG_ACCEL],
            params[e, a, P_LF],
            params[e, a, P_LR],
            params[e, a, P_MAX_STEER],
            params[e, a, P_RADIUS],
            params[e, a, P_SHAPE],
            params[e, a, P_HALF_X],
            params[e, a, P_HALF_Y],
            params[e, a, P_THRUST_MAX],
            params[e, a, P_ARM],
            params[e, a, P_IXX],
            params[e, a, P_IYY],
            params[e, a, P_IZZ],
            params[e, a, P_KAPPA],
            params[e, a, P_GRAVITY],
            dt,
            bounds_min,
            bounds_max,
            clamp_bounds,
            pos_out,
            theta_out,
            vel_out,
            speed_out,
            ang_vel_out,
            z_out,
            vz_out,
            attitude_out,
            body_rates_out,
        )


@wp.kernel
def integrate2d_kernel(
    # 2D state in
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    speed: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
    # inputs
    actions: wp.array3d(dtype=Any),
    forces: wp.array2d(dtype=Any),
    params: wp.array2d(dtype=Any),  # [n_agents, NUM_PARAMS] shared across envs
    model_tag: wp.array(dtype=wp.int32),
    ctrl_mode: wp.array(dtype=wp.int32),
    dt: Any,
    bounds_min: Any,
    bounds_max: Any,
    clamp_bounds: wp.int32,
    # 2D state out
    pos_out: wp.array2d(dtype=Any),
    theta_out: wp.array2d(dtype=Any),
    vel_out: wp.array2d(dtype=Any),
    speed_out: wp.array2d(dtype=Any),
    ang_vel_out: wp.array2d(dtype=Any),
):
    """Slim Euler kernel: 2D fleets only (no drone fields), no RK4 branch. Reads
    and writes five state arrays instead of nine. Uses :func:`_step_2d`, so the
    live fields are bit-identical to the full kernel."""
    e, a = wp.tid()
    a0 = actions[e, a, 0]
    a1 = actions[e, a, 1]
    mode = ctrl_mode[a]
    if mode == MODE_SERVO:
        a0, a1 = servo_command(
            a0,
            a1,
            speed[e, a],
            ang_vel[e, a],
            params[a, P_MAX_SPEED],
            params[a, P_MAX_ANG_VEL],
            params[a, P_MAX_ACCEL],
            params[a, P_MAX_ANG_ACCEL],
            params[a, P_CMD_GAIN_V],
            params[a, P_CMD_GAIN_W],
            dt,
        )
        mode = MODE_VELOCITY
    p_new, th_new, v_out, s_out, w_out = _step_2d(
        pos[e, a],
        theta[e, a],
        vel[e, a],
        speed[e, a],
        ang_vel[e, a],
        a0,
        a1,
        model_tag[a],
        mode,
        forces[e, a],
        params[a, P_MAX_SPEED],
        params[a, P_MIN_SPEED],
        params[a, P_MAX_ACCEL],
        params[a, P_MAX_ANG_VEL],
        params[a, P_MAX_ANG_ACCEL],
        params[a, P_LF],
        params[a, P_LR],
        params[a, P_MAX_STEER],
        params[a, P_MASS],
        dt,
    )
    if clamp_bounds == 1:
        radius = params[a, P_RADIUS]
        p_new = type(p_new)(
            wp.clamp(p_new[0], bounds_min[0] + radius, bounds_max[0] - radius),
            wp.clamp(p_new[1], bounds_min[1] + radius, bounds_max[1] - radius),
        )
    elif clamp_bounds == CLAMP_FOOTPRINT:
        p_new = clamp_footprint(
            p_new,
            th_new,
            params[a, P_RADIUS],
            params[a, P_SHAPE],
            params[a, P_HALF_X],
            params[a, P_HALF_Y],
            bounds_min,
            bounds_max,
        )
    pos_out[e, a] = p_new
    theta_out[e, a] = th_new
    vel_out[e, a] = v_out
    speed_out[e, a] = s_out
    ang_vel_out[e, a] = w_out


@wp.kernel
def integrate2d_kernel_per_env(
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    speed: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
    actions: wp.array3d(dtype=Any),
    forces: wp.array2d(dtype=Any),
    params: wp.array3d(dtype=Any),  # [n_envs, n_agents, NUM_PARAMS] per-env
    model_tag: wp.array(dtype=wp.int32),
    ctrl_mode: wp.array(dtype=wp.int32),
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
    """Per-env-parameter variant of :func:`integrate2d_kernel`."""
    e, a = wp.tid()
    a0 = actions[e, a, 0]
    a1 = actions[e, a, 1]
    mode = ctrl_mode[a]
    if mode == MODE_SERVO:
        a0, a1 = servo_command(
            a0,
            a1,
            speed[e, a],
            ang_vel[e, a],
            params[e, a, P_MAX_SPEED],
            params[e, a, P_MAX_ANG_VEL],
            params[e, a, P_MAX_ACCEL],
            params[e, a, P_MAX_ANG_ACCEL],
            params[e, a, P_CMD_GAIN_V],
            params[e, a, P_CMD_GAIN_W],
            dt,
        )
        mode = MODE_VELOCITY
    p_new, th_new, v_out, s_out, w_out = _step_2d(
        pos[e, a],
        theta[e, a],
        vel[e, a],
        speed[e, a],
        ang_vel[e, a],
        a0,
        a1,
        model_tag[a],
        mode,
        forces[e, a],
        params[e, a, P_MAX_SPEED],
        params[e, a, P_MIN_SPEED],
        params[e, a, P_MAX_ACCEL],
        params[e, a, P_MAX_ANG_VEL],
        params[e, a, P_MAX_ANG_ACCEL],
        params[e, a, P_LF],
        params[e, a, P_LR],
        params[e, a, P_MAX_STEER],
        params[e, a, P_MASS],
        dt,
    )
    if clamp_bounds == 1:
        radius = params[e, a, P_RADIUS]
        p_new = type(p_new)(
            wp.clamp(p_new[0], bounds_min[0] + radius, bounds_max[0] - radius),
            wp.clamp(p_new[1], bounds_min[1] + radius, bounds_max[1] - radius),
        )
    elif clamp_bounds == CLAMP_FOOTPRINT:
        p_new = clamp_footprint(
            p_new,
            th_new,
            params[e, a, P_RADIUS],
            params[e, a, P_SHAPE],
            params[e, a, P_HALF_X],
            params[e, a, P_HALF_Y],
            bounds_min,
            bounds_max,
        )
    pos_out[e, a] = p_new
    theta_out[e, a] = th_new
    vel_out[e, a] = v_out
    speed_out[e, a] = s_out
    ang_vel_out[e, a] = w_out


_BOX_OPTS = {"fuse_fp": False}


@wp.func
def wheel_contact(
    p: Any,
    th: Any,
    p_new: Any,
    th_new: Any,
    v_out: Any,
    s_cmd: Any,
    w_cmd: Any,
    force: Any,
    torque: Any,
    tag: wp.int32,
    mass: Any,
    half_x: Any,
    half_y: Any,
    lat_grip: Any,
    turn_cap: Any,
    drive_cap: Any,
    dt: Any,
):
    """Rework a diff-drive agent's :func:`_step_2d` result for wheel-faithful contacts
    (``WorldConfig.wheel_contacts``); any other model passes through.

    The contact force acts along the heading as before (the drive gives way by
    ``F dt / m``) but sideways only by what exceeds the wheels' grip ``lat_grip * m``
    (``lat_grip`` is ``mu g``), and the contact torque turns the agent the same way, by
    what exceeds the torque ``turn_cap`` the drive holds its turn rate against:
    ``omega = w_cmd + (|tau| - turn_cap)+ sign(tau) dt / I``, ``I = m (hx^2 + hy^2) / 3``,
    and a torque beyond ``turn_cap`` that opposes the commanded turn also stalls the
    drive's turn for the substep. A box turning into a face that does not give way
    therefore stops once the corner it drives in pushes back harder than the drive turns,
    while an off-centre push that torques it less than ``turn_cap`` leaves its heading
    alone. A contact that pushes back
    along the heading harder than the drive's traction ``drive_cap`` (a force) stalls the
    drive's own speed ``s_cmd`` for the substep, so an agent pressing something that does
    not give way presses with about that force. Returns the new ``(pos, heading,
    velocity)``; the drive's own speed and turn rate stay the state."""
    zero = type(dt)(0.0)
    p_out = p_new
    th_out = th_new
    v_new = v_out
    if tag == TAG_DIFF_DRIVE:
        ca = wp.cos(th)
        sa = wp.sin(th)
        f_lat = -force[0] * sa + force[1] * ca
        slip = wp.max(wp.abs(f_lat) - lat_grip * mass, zero)
        if f_lat < zero:
            slip = -slip
        # The velocity _step_2d produced already holds the whole force; swap its sideways
        # part for the slip.
        dv = (slip - f_lat) * (dt / mass)
        corr = type(p)(-sa * dv, ca * dv)
        f_lon = force[0] * ca + force[1] * sa
        if f_lon * s_cmd < zero and wp.abs(f_lon) > drive_cap:
            corr = corr - type(p)(ca * s_cmd, sa * s_cmd)
        v_new = v_out + corr
        p_out = p_new + corr * dt
        inertia = mass * (half_x * half_x + half_y * half_y) / type(dt)(3.0)
        excess = wp.max(wp.abs(torque) - turn_cap, zero)
        if torque < zero:
            excess = -excess
        w_drive = w_cmd
        if torque * w_cmd < zero and wp.abs(torque) > turn_cap:
            w_drive = zero  # the contact resists the turn harder than the drive turns
        th_out = th + (w_drive + excess * (dt / inertia)) * dt
    return p_out, th_out, v_new


@wp.kernel(module="unique", module_options=_BOX_OPTS, grid_stride=False)
def integrate2d_wheel_kernel(
    # 2D state in
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    speed: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
    # inputs
    actions: wp.array3d(dtype=Any),
    forces: wp.array2d(dtype=Any),
    params: wp.array2d(dtype=Any),  # [n_agents, NUM_PARAMS] shared across envs
    model_tag: wp.array(dtype=wp.int32),
    ctrl_mode: wp.array(dtype=wp.int32),
    dt: Any,
    bounds_min: Any,
    bounds_max: Any,
    clamp_bounds: wp.int32,
    torques: wp.array2d(dtype=Any),
    lat_grip: Any,
    turn_cap: Any,
    drive_cap: Any,
    # 2D state out
    pos_out: wp.array2d(dtype=Any),
    theta_out: wp.array2d(dtype=Any),
    vel_out: wp.array2d(dtype=Any),
    speed_out: wp.array2d(dtype=Any),
    ang_vel_out: wp.array2d(dtype=Any),
):
    """:func:`integrate2d_kernel` with wheel-faithful contacts (see :func:`wheel_contact`)."""
    e, a = wp.tid()
    a0 = actions[e, a, 0]
    a1 = actions[e, a, 1]
    mode = ctrl_mode[a]
    if mode == MODE_SERVO:
        a0, a1 = servo_command(
            a0,
            a1,
            speed[e, a],
            ang_vel[e, a],
            params[a, P_MAX_SPEED],
            params[a, P_MAX_ANG_VEL],
            params[a, P_MAX_ACCEL],
            params[a, P_MAX_ANG_ACCEL],
            params[a, P_CMD_GAIN_V],
            params[a, P_CMD_GAIN_W],
            dt,
        )
        mode = MODE_VELOCITY
    p_new, th_new, v_out, s_out, w_out = _step_2d(
        pos[e, a],
        theta[e, a],
        vel[e, a],
        speed[e, a],
        ang_vel[e, a],
        a0,
        a1,
        model_tag[a],
        mode,
        forces[e, a],
        params[a, P_MAX_SPEED],
        params[a, P_MIN_SPEED],
        params[a, P_MAX_ACCEL],
        params[a, P_MAX_ANG_VEL],
        params[a, P_MAX_ANG_ACCEL],
        params[a, P_LF],
        params[a, P_LR],
        params[a, P_MAX_STEER],
        params[a, P_MASS],
        dt,
    )
    p_new, th_new, v_out = wheel_contact(
        pos[e, a],
        theta[e, a],
        p_new,
        th_new,
        v_out,
        s_out,
        w_out,
        forces[e, a],
        torques[e, a],
        model_tag[a],
        params[a, P_MASS],
        params[a, P_HALF_X],
        params[a, P_HALF_Y],
        lat_grip,
        turn_cap,
        drive_cap,
        dt,
    )
    if clamp_bounds == 1:
        radius = params[a, P_RADIUS]
        p_new = type(p_new)(
            wp.clamp(p_new[0], bounds_min[0] + radius, bounds_max[0] - radius),
            wp.clamp(p_new[1], bounds_min[1] + radius, bounds_max[1] - radius),
        )
    elif clamp_bounds == CLAMP_FOOTPRINT:
        p_new = clamp_footprint(
            p_new,
            th_new,
            params[a, P_RADIUS],
            params[a, P_SHAPE],
            params[a, P_HALF_X],
            params[a, P_HALF_Y],
            bounds_min,
            bounds_max,
        )
    pos_out[e, a] = p_new
    theta_out[e, a] = th_new
    vel_out[e, a] = v_out
    speed_out[e, a] = s_out
    ang_vel_out[e, a] = w_out

@wp.kernel(module="unique", module_options=_BOX_OPTS, grid_stride=False)
def integrate2d_wheel_kernel_per_env(
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    speed: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
    actions: wp.array3d(dtype=Any),
    forces: wp.array2d(dtype=Any),
    params: wp.array3d(dtype=Any),  # [n_envs, n_agents, NUM_PARAMS] per-env
    model_tag: wp.array(dtype=wp.int32),
    ctrl_mode: wp.array(dtype=wp.int32),
    dt: Any,
    bounds_min: Any,
    bounds_max: Any,
    clamp_bounds: wp.int32,
    torques: wp.array2d(dtype=Any),
    lat_grip: Any,
    turn_cap: Any,
    drive_cap: Any,
    pos_out: wp.array2d(dtype=Any),
    theta_out: wp.array2d(dtype=Any),
    vel_out: wp.array2d(dtype=Any),
    speed_out: wp.array2d(dtype=Any),
    ang_vel_out: wp.array2d(dtype=Any),
):
    """Per-env-parameter twin of :func:`integrate2d_wheel_kernel`."""
    e, a = wp.tid()
    a0 = actions[e, a, 0]
    a1 = actions[e, a, 1]
    mode = ctrl_mode[a]
    if mode == MODE_SERVO:
        a0, a1 = servo_command(
            a0,
            a1,
            speed[e, a],
            ang_vel[e, a],
            params[e, a, P_MAX_SPEED],
            params[e, a, P_MAX_ANG_VEL],
            params[e, a, P_MAX_ACCEL],
            params[e, a, P_MAX_ANG_ACCEL],
            params[e, a, P_CMD_GAIN_V],
            params[e, a, P_CMD_GAIN_W],
            dt,
        )
        mode = MODE_VELOCITY
    p_new, th_new, v_out, s_out, w_out = _step_2d(
        pos[e, a],
        theta[e, a],
        vel[e, a],
        speed[e, a],
        ang_vel[e, a],
        a0,
        a1,
        model_tag[a],
        mode,
        forces[e, a],
        params[e, a, P_MAX_SPEED],
        params[e, a, P_MIN_SPEED],
        params[e, a, P_MAX_ACCEL],
        params[e, a, P_MAX_ANG_VEL],
        params[e, a, P_MAX_ANG_ACCEL],
        params[e, a, P_LF],
        params[e, a, P_LR],
        params[e, a, P_MAX_STEER],
        params[e, a, P_MASS],
        dt,
    )
    p_new, th_new, v_out = wheel_contact(
        pos[e, a],
        theta[e, a],
        p_new,
        th_new,
        v_out,
        s_out,
        w_out,
        forces[e, a],
        torques[e, a],
        model_tag[a],
        params[e, a, P_MASS],
        params[e, a, P_HALF_X],
        params[e, a, P_HALF_Y],
        lat_grip,
        turn_cap,
        drive_cap,
        dt,
    )
    if clamp_bounds == 1:
        radius = params[e, a, P_RADIUS]
        p_new = type(p_new)(
            wp.clamp(p_new[0], bounds_min[0] + radius, bounds_max[0] - radius),
            wp.clamp(p_new[1], bounds_min[1] + radius, bounds_max[1] - radius),
        )
    elif clamp_bounds == CLAMP_FOOTPRINT:
        p_new = clamp_footprint(
            p_new,
            th_new,
            params[e, a, P_RADIUS],
            params[e, a, P_SHAPE],
            params[e, a, P_HALF_X],
            params[e, a, P_HALF_Y],
            bounds_min,
            bounds_max,
        )
    pos_out[e, a] = p_new
    theta_out[e, a] = th_new
    vel_out[e, a] = v_out
    speed_out[e, a] = s_out
    ang_vel_out[e, a] = w_out


#: Box-world copies of the slim kernels: the same functions compiled without FP
#: contraction (see ``swarp.core.collisions.BOX_MODULE_OPTIONS``), so a box world's
#: layered step rounds exactly like its fused one. Disc worlds launch the originals.
integrate2d_box_kernel = wp.kernel(
    integrate2d_kernel.func, module="unique", module_options=_BOX_OPTS, grid_stride=False
)
integrate2d_box_kernel_per_env = wp.kernel(
    integrate2d_kernel_per_env.func, module="unique", module_options=_BOX_OPTS, grid_stride=False
)


def _signature2d(dtype, per_env: bool = False) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    a2s = wp.array2d(dtype=dtype)
    a3s = wp.array3d(dtype=dtype)
    a1i = wp.array(dtype=wp.int32)
    params = wp.array3d(dtype=dtype) if per_env else a2s
    state = [a2v, a2s, a2v, a2s, a2s]  # pos, theta, vel, speed, ang_vel
    return [
        *state,
        a3s,
        a2v,
        params,
        a1i,
        a1i,
        dtype,
        vec2,
        vec2,
        wp.int32,
        *state,
    ]


def _signature2d_wheel(dtype, per_env: bool = False) -> list:
    """:func:`_signature2d` with ``torques``, ``lat_grip``, ``turn_cap`` and
    ``drive_cap`` after ``clamp_bounds``."""
    sig = _signature2d(dtype, per_env)
    return [*sig[:14], wp.array2d(dtype=dtype), dtype, dtype, dtype, *sig[14:]]


def _signature(dtype, per_env: bool = False) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    a2s = wp.array2d(dtype=dtype)
    a2v3 = wp.array2d(dtype=VEC3[dtype])
    a2q = wp.array2d(dtype=QUAT[dtype])
    a3s = wp.array3d(dtype=dtype)  # actions: [n_envs, n_agents, act_dim] scalar
    a1i = wp.array(dtype=wp.int32)
    params = wp.array3d(dtype=dtype) if per_env else a2s
    # State field order (in and out): pos, theta, vel, speed, ang_vel, z, vz,
    # attitude, body_rates — matches WorldState / STATE_FIELDS.
    state = [a2v, a2s, a2v, a2s, a2s, a2s, a2s, a2q, a2v3]
    return [
        *state,  # state in
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
        *state,  # state out
    ]


@wp.kernel
def delayed_actions_kernel(
    actions: wp.array3d(dtype=Any),
    prev_actions: wp.array3d(dtype=Any),
    delay: wp.array2d(dtype=wp.int32),
    out: wp.array4d(dtype=Any),
):
    """Thread per (env, agent): the command each of the first ``out.shape[0]`` substeps
    applies under a command latency, ``out[k] = prev_actions`` while ``k < delay`` and
    ``actions`` from then on (see :meth:`Stepper.enable_command_delay
    <swarp.core.stepper.Stepper.enable_command_delay>`)."""
    e, a = wp.tid()
    d = delay[e, a]
    for k in range(out.shape[0]):
        for i in range(actions.shape[2]):
            if k < d:
                out[k, e, a, i] = prev_actions[e, a, i]
            else:
                out[k, e, a, i] = actions[e, a, i]


for _T in (wp.float32, wp.float64):
    register(
        delayed_actions_kernel,
        _T,
        [wp.array3d(dtype=_T), wp.array3d(dtype=_T), wp.array2d(dtype=wp.int32),
         wp.array4d(dtype=_T)],
    )
    register(integrate_kernel, _T, _signature(_T))
    register(integrate_kernel_per_env, _T, _signature(_T, per_env=True))
    register(integrate2d_kernel, _T, _signature2d(_T))
    register(integrate2d_kernel_per_env, _T, _signature2d(_T, per_env=True))
    register(integrate2d_box_kernel, _T, _signature2d(_T))
    register(integrate2d_box_kernel_per_env, _T, _signature2d(_T, per_env=True))
    register(integrate2d_wheel_kernel, _T, _signature2d_wheel(_T))
    register(integrate2d_wheel_kernel_per_env, _T, _signature2d_wheel(_T, per_env=True))


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
    slim: bool = False,
    skip_drone: bool = False,
    footprint: bool = False,
    torques: wp.array | None = None,
    lat_grip: float = 0.0,
    turn_cap: float = 0.0,
    drive_cap: float = float("inf"),
) -> None:
    """Launch one integration sub-step. Functional: never writes to ``state_in``.

    ``clamp_bounds=(x_min, x_max, y_min, y_max)`` hard-clamps positions inside
    the rectangle (inset by each agent's radius). ``footprint=True`` insets each box
    agent by its support extents at its new heading instead (the stepper passes it exactly
    when the world holds a box agent; a disc is inset by its radius either way).
    ``integrator`` selects the semi-implicit Euler recurrence (default) or classic RK4.

    ``torques`` (the force pass's per-agent contact torques) selects the wheel-contact
    variant of the slim kernel (:func:`wheel_contact`, ``lat_grip`` is the wheels'
    sideways ``mu g``, ``turn_cap`` the torque the drive holds its turn rate against,
    ``drive_cap`` its traction along the heading); it needs ``slim``.

    ``slim=True`` launches the 2D-only Euler kernel that computes five state
    arrays instead of nine (no drone math, no RK4 branch). The caller must
    guarantee this is valid (no drone agents, Euler, not taped). The four drone
    fields are passed through unchanged so the result stays bit-identical to the
    full kernel for arbitrary inputs; pass ``skip_drone=True`` (graph/Environment
    context, where the drone state is known-zero and never read) to elide that
    passthrough entirely.
    """
    dtype = state_in.theta.dtype
    vec2 = VEC2[dtype]
    n_envs, n_agents = state_in.pos.shape
    if clamp_bounds is None:
        b_min, b_max, do_clamp = vec2(0.0, 0.0), vec2(0.0, 0.0), 0
    else:
        x_min, x_max, y_min, y_max = clamp_bounds
        b_min, b_max = vec2(x_min, y_min), vec2(x_max, y_max)
        do_clamp = 2 if footprint else 1
    per_env = params.floats_per_env is not None
    floats = params.floats_per_env if per_env else params.floats
    if torques is not None and not slim:
        raise ValueError("wheel contacts (torques) need the slim Euler 2D path")
    if slim:
        wheel = []
        if torques is not None:
            kernel2d = integrate2d_wheel_kernel_per_env if per_env else integrate2d_wheel_kernel
            wheel = [torques, dtype(lat_grip), dtype(turn_cap), dtype(drive_cap)]
        elif footprint:
            kernel2d = integrate2d_box_kernel_per_env if per_env else integrate2d_box_kernel
        else:
            kernel2d = integrate2d_kernel_per_env if per_env else integrate2d_kernel
        wp.launch(
            concrete(kernel2d, dtype),
            dim=(n_envs, n_agents),
            inputs=[
                state_in.pos,
                state_in.theta,
                state_in.vel,
                state_in.speed,
                state_in.ang_vel,
                actions,
                forces,
                floats,
                params.model_tag,
                params.ctrl_mode,
                dtype(dt),
                b_min,
                b_max,
                wp.int32(do_clamp),
                *wheel,
            ],
            outputs=[
                state_out.pos,
                state_out.theta,
                state_out.vel,
                state_out.speed,
                state_out.ang_vel,
            ],
            device=state_in.pos.device,
        )
        if not skip_drone:
            # The full kernel passes the 6-DOF drone state through unchanged for
            # 2D agents; mirror that so the slim result is bit-identical for any
            # input. Callers whose drone state is known-zero-and-unread (the
            # graph copy-back) pass skip_drone=True to elide these copies.
            wp.copy(state_out.z, state_in.z)
            wp.copy(state_out.vz, state_in.vz)
            wp.copy(state_out.attitude, state_in.attitude)
            wp.copy(state_out.body_rates, state_in.body_rates)
        return
    kernel = integrate_kernel_per_env if per_env else integrate_kernel
    wp.launch(
        concrete(kernel, dtype),
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
