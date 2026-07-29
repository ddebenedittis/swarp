"""Differentiable soft-collision (spring-damper) forces.

Gather-based: each thread sums the forces acting on its own agent from its
padded neighbor list, obstacles, and soft walls. Symmetric pairs are computed
twice — no atomics, so the result is deterministic and race-free. Gradients
flow through positions/velocities; the neighbor list itself is a fixed
discrete structure within a substep.

The normal damping is **linearly implicit** and clamped repulsive. With the contact
normal frozen over the substep, solving ``f = k*overlap - c*(v_n(f) - v_obs_n)`` for the
post-impulse normal velocity is a single divide::

    f = max(0, (k*overlap - c*v_rel_n) / (1 + c*sub_dt/m))

That is unconditionally stable in ``collision_c`` — the explicit form needed
``sub_dt < m/c``, which capped usable stiffness — and it cannot turn attractive, so a
contact never sticks to a separating agent. It stays a smooth rational function of the
state, which differentiates better than the explicit form (whose sign can flip).

Obstacles carry a velocity, so the damper always uses the **closing** velocity. An
obstacle that moves (Push-T installs its T as two boxes) would otherwise be damped
against the agent's absolute world velocity, applying drag that has nothing to do with
the contact; the error scales with ``collision_c``.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp.core.config import ObstacleShape
from swarp.core.state import VEC2
from swarp.dynamics.base import P_MASS, P_RADIUS, AgentParams

_EPS2 = 1.0e-10  # distance^2 floor: keeps sqrt adjoint finite for coincident points

# Kernel-side obstacle shape tags, derived from the enum so there is one source of truth.
SHAPE_CIRCLE = wp.constant(int(ObstacleShape.CIRCLE))
SHAPE_BOX = wp.constant(int(ObstacleShape.BOX))
SHAPE_SEGMENT = wp.constant(int(ObstacleShape.SEGMENT))


@wp.func
def _normal_coeff(overlap: Any, vn: Any, k: Any, c: Any, damp_denom: Any, max_overlap: Any):
    """Scalar normal force: linearly-implicit damping, clamped repulsive, depth-saturated.

    ``vn`` is the *closing* normal velocity (relative, positive when separating) and
    ``damp_denom = 1 + c*sub_dt/m`` is the implicit-solve denominator, precomputed once
    per agent by the caller (the Jacobi / per-agent-diagonal approximation).

    ``max_overlap > 0`` smoothly saturates the depth fed to the spring at that value
    (``tanh``, so the force stays differentiable in the depth everywhere, unlike a hard
    clamp). A velocity-mode agent has no contact memory, so it settles at an overlap of
    ``v*m/(k*sub_dt)`` — 20 mm at the Push-T defaults — and against a light body that depth
    is a violent impulse. Saturating bounds it without touching the shallow regime that
    legitimate pushing lives in. ``0`` disables it, which is the historical behaviour."""
    zero = type(k)(0.0)
    ov = overlap
    if max_overlap > zero:
        ov = max_overlap * wp.tanh(overlap / max_overlap)
    return wp.max((k * ov - c * vn) / damp_denom, zero)


@wp.func
def _pair_force(
    d: Any, rel_v: Any, min_dist: Any, k: Any, c: Any, damp_denom: Any, max_overlap: Any
):
    """Spring-damper repulsion along d (from other to self) within min_dist."""
    zero = type(k)(0.0)
    dist = wp.sqrt(wp.max(wp.dot(d, d), type(k)(_EPS2)))
    f = type(d)(zero, zero)
    overlap = min_dist - dist
    if overlap > zero:
        n = d / dist
        f = _normal_coeff(overlap, wp.dot(rel_v, n), k, c, damp_denom, max_overlap) * n
    return f


@wp.func
def _normal_force(
    n: Any, overlap: Any, rel_v: Any, k: Any, c: Any, damp_denom: Any, max_overlap: Any
):
    """Spring-damper along a precomputed unit normal ``n`` (n points from the
    surface toward the agent). Used when the normal is known analytically (box
    SDF) rather than derived from a separation vector."""
    return _normal_coeff(overlap, wp.dot(rel_v, n), k, c, damp_denom, max_overlap) * n


@wp.func
def _closest_on_segment(p: Any, center: Any, angle: Any, half_len: Any):
    """Closest point on a segment core (center, orientation ``angle``, half
    length ``half_len``) to ``p``. Differentiable (clamp subgradient)."""
    d = type(p)(wp.cos(angle), wp.sin(angle))
    t = wp.clamp(wp.dot(p - center, d), -half_len, half_len)
    return center + t * d


@wp.func
def _box_force(
    p: Any,
    v: Any,
    center: Any,
    angle: Any,
    half: Any,
    reach: Any,
    k: Any,
    c: Any,
    damp_denom: Any,
    max_overlap: Any,
    v_obs: Any,
    om_obs: Any,
):
    """Contact force from an oriented box using its signed distance field, so
    an agent whose center penetrates the box is still pushed out (a plain
    closest-point clamp has a zero-force interior dead-zone).

    ``reach = agent_radius + margin``; contact when the signed distance from the
    agent center to the box surface drops below ``reach``.
    """
    zero = type(k)(0.0)
    one = type(k)(1.0)
    ca = wp.cos(angle)
    sa = wp.sin(angle)
    d = p - center
    # rotate into the box frame (R^T d)
    lx = ca * d[0] + sa * d[1]
    ly = -sa * d[0] + ca * d[1]
    qx = wp.abs(lx) - half[0]
    qy = wp.abs(ly) - half[1]
    # closest boundary point in the box frame (exterior) via clamp
    cx = wp.clamp(lx, -half[0], half[0])
    cy = wp.clamp(ly, -half[1], half[1])
    ox = lx - cx
    oy = ly - cy
    out_d2 = ox * ox + oy * oy
    if out_d2 > type(k)(_EPS2):
        # exterior: signed distance is the positive outside distance
        s = wp.sqrt(out_d2)
        nlx = ox / s
        nly = oy / s
    else:
        # interior: nearest face determines the (negative) signed distance
        if qx > qy:
            s = qx
            if lx >= zero:
                nlx = one
            else:
                nlx = -one
            nly = zero
        else:
            s = qy
            nlx = zero
            if ly >= zero:
                nly = one
            else:
                nly = -one
    f = type(p)(zero, zero)
    overlap = reach - s
    if overlap > zero:
        # rotate the box-frame normal back to world (R n_local)
        n = type(p)(ca * nlx - sa * nly, sa * nlx + ca * nly)
        # Velocity of the box's material point in contact: v_obs + om x r, with r the
        # lever from the box centre to the closest surface point. Damping must see the
        # closing velocity, not the agent's absolute velocity.
        spx = lx - s * nlx
        spy = ly - s * nly
        rx = ca * spx - sa * spy
        ry = sa * spx + ca * spy
        v_surface = v_obs + type(p)(-om_obs * ry, om_obs * rx)
        f = _normal_force(n, overlap, v - v_surface, k, c, damp_denom, max_overlap)
    return f


@wp.func
def _static_forces(
    e: wp.int32,
    p: Any,
    v: Any,
    ra: Any,
    obs_pos: wp.array2d(dtype=Any),
    obs_radius: wp.array(dtype=Any),
    obs_type: wp.array(dtype=wp.int32),
    obs_angle: wp.array2d(dtype=Any),
    obs_half: wp.array(dtype=Any),
    obs_vel: wp.array2d(dtype=Any),
    obs_ang_vel: wp.array2d(dtype=Any),
    n_obstacles: wp.int32,
    k: Any,
    c: Any,
    margin: Any,
    damp_denom: Any,
    max_overlap: Any,
    soft_walls: wp.int32,
    bounds_min: Any,
    bounds_max: Any,
):
    """Env-static contact forces on an agent at ``p`` (radius ``ra``): obstacles
    and soft walls. Independent of the per-agent param layout, so both the
    shared and per-env kernels reuse it; only the pairwise-neighbor radius read
    differs between them.

    ``obs_vel``/``obs_ang_vel`` are the obstacles' own velocities (zero for a genuinely
    static obstacle), so the damper sees the closing velocity in every case."""
    zero = type(k)(0.0)
    f = type(p)(zero, zero)

    for o in range(n_obstacles):
        center = obs_pos[e, o]
        st = obs_type[o]
        vo = obs_vel[e, o]
        omo = obs_ang_vel[e, o]
        if st == SHAPE_BOX:
            # box surface is the boundary itself; agent inflated by ra + margin
            f += _box_force(
                p, v, center, obs_angle[e, o], obs_half[o], ra + margin, k, c, damp_denom,
                max_overlap, vo, omo,
            )
        elif st == SHAPE_SEGMENT:
            cp = _closest_on_segment(p, center, obs_angle[e, o], obs_half[o][0])
            # surface point velocity: v_obs + om x (cp - centre)
            r = cp - center
            vs = vo + type(p)(-omo * r[1], omo * r[0])
            f += _pair_force(
                p - cp, v - vs, ra + obs_radius[o] + margin, k, c, damp_denom, max_overlap
            )
        else:  # SHAPE_CIRCLE
            # a spinning disc has no normal-direction surface motion (frictionless)
            f += _pair_force(
                p - center, v - vo, ra + obs_radius[o] + margin, k, c, damp_denom, max_overlap
            )

    if soft_walls == 1:
        reach = ra + margin
        fx = zero
        fy = zero
        # Walls are static, and each contributes along its own inward normal, so the
        # closing velocity is +-v[i]; same implicit, clamped-repulsive coefficient.
        pen = reach - (p[0] - bounds_min[0])  # left wall, inward normal (+1, 0)
        if pen > zero:
            fx += _normal_coeff(pen, v[0], k, c, damp_denom, max_overlap)
        pen = reach - (bounds_max[0] - p[0])  # right wall, inward normal (-1, 0)
        if pen > zero:
            fx += -_normal_coeff(pen, -v[0], k, c, damp_denom, max_overlap)
        pen = reach - (p[1] - bounds_min[1])  # bottom wall
        if pen > zero:
            fy += _normal_coeff(pen, v[1], k, c, damp_denom, max_overlap)
        pen = reach - (bounds_max[1] - p[1])  # top wall
        if pen > zero:
            fy += -_normal_coeff(pen, -v[1], k, c, damp_denom, max_overlap)
        f += type(p)(fx, fy)

    return f


@wp.kernel
def collision_forces_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    params: wp.array2d(dtype=Any),  # [n_agents, NUM_PARAMS] shared across envs
    neighbor_idx: wp.array3d(dtype=wp.int32),
    neighbor_count: wp.array2d(dtype=wp.int32),
    obs_pos: wp.array2d(dtype=Any),
    obs_radius: wp.array(dtype=Any),
    obs_type: wp.array(dtype=wp.int32),
    obs_angle: wp.array2d(dtype=Any),
    obs_half: wp.array(dtype=Any),
    obs_vel: wp.array2d(dtype=Any),
    obs_ang_vel: wp.array2d(dtype=Any),
    n_obstacles: wp.int32,
    k: Any,
    c: Any,
    margin: Any,
    sub_dt: Any,
    max_overlap: Any,
    soft_walls: wp.int32,
    bounds_min: Any,
    bounds_max: Any,
    forces: wp.array2d(dtype=Any),
):
    e, a = wp.tid()
    p = pos[e, a]
    v = vel[e, a]
    ra = params[a, P_RADIUS]
    zero = type(k)(0.0)
    f = type(p)(zero, zero)
    # Implicit-damping denominator, Jacobi (per-agent) approximation of the effective mass.
    damp_denom = type(k)(1.0) + c * sub_dt / params[a, P_MASS]

    for n_i in range(neighbor_count[e, a]):
        b = neighbor_idx[e, a, n_i]
        f += _pair_force(
            p - pos[e, b], v - vel[e, b], ra + params[b, P_RADIUS] + margin, k, c, damp_denom,
            max_overlap,
        )

    f += _static_forces(
        e,
        p,
        v,
        ra,
        obs_pos,
        obs_radius,
        obs_type,
        obs_angle,
        obs_half,
        obs_vel,
        obs_ang_vel,
        n_obstacles,
        k,
        c,
        margin,
        damp_denom,
        max_overlap,
        soft_walls,
        bounds_min,
        bounds_max,
    )
    forces[e, a] = f


@wp.kernel
def collision_forces_kernel_per_env(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    params: wp.array3d(dtype=Any),  # [n_envs, n_agents, NUM_PARAMS] per-env
    neighbor_idx: wp.array3d(dtype=wp.int32),
    neighbor_count: wp.array2d(dtype=wp.int32),
    obs_pos: wp.array2d(dtype=Any),
    obs_radius: wp.array(dtype=Any),
    obs_type: wp.array(dtype=wp.int32),
    obs_angle: wp.array2d(dtype=Any),
    obs_half: wp.array(dtype=Any),
    obs_vel: wp.array2d(dtype=Any),
    obs_ang_vel: wp.array2d(dtype=Any),
    n_obstacles: wp.int32,
    k: Any,
    c: Any,
    margin: Any,
    sub_dt: Any,
    max_overlap: Any,
    soft_walls: wp.int32,
    bounds_min: Any,
    bounds_max: Any,
    forces: wp.array2d(dtype=Any),
):
    e, a = wp.tid()
    p = pos[e, a]
    v = vel[e, a]
    ra = params[e, a, P_RADIUS]
    zero = type(k)(0.0)
    f = type(p)(zero, zero)
    damp_denom = type(k)(1.0) + c * sub_dt / params[e, a, P_MASS]

    for n_i in range(neighbor_count[e, a]):
        b = neighbor_idx[e, a, n_i]
        # neighbor b's radius is env-specific in the per-env layout
        f += _pair_force(
            p - pos[e, b], v - vel[e, b], ra + params[e, b, P_RADIUS] + margin, k, c, damp_denom,
            max_overlap,
        )

    f += _static_forces(
        e,
        p,
        v,
        ra,
        obs_pos,
        obs_radius,
        obs_type,
        obs_angle,
        obs_half,
        obs_vel,
        obs_ang_vel,
        n_obstacles,
        k,
        c,
        margin,
        damp_denom,
        max_overlap,
        soft_walls,
        bounds_min,
        bounds_max,
    )
    forces[e, a] = f


def _signature(dtype, per_env: bool = False) -> list:
    vec2 = VEC2[dtype]
    params = wp.array3d(dtype=dtype) if per_env else wp.array2d(dtype=dtype)
    return [
        wp.array2d(dtype=vec2),
        wp.array2d(dtype=vec2),
        params,
        wp.array3d(dtype=wp.int32),
        wp.array2d(dtype=wp.int32),
        wp.array2d(dtype=vec2),  # obs_pos [n_envs, n_obs]
        wp.array(dtype=dtype),  # obs_radius [n_obs]
        wp.array(dtype=wp.int32),  # obs_type [n_obs]
        wp.array2d(dtype=dtype),  # obs_angle [n_envs, n_obs] (per-env: rotating bodies)
        wp.array(dtype=vec2),  # obs_half [n_obs]
        wp.array2d(dtype=vec2),  # obs_vel [n_envs, n_obs]
        wp.array2d(dtype=dtype),  # obs_ang_vel [n_envs, n_obs]
        wp.int32,
        dtype,  # k
        dtype,  # c
        dtype,  # margin
        dtype,  # sub_dt (implicit damping)
        dtype,  # max_overlap
        wp.int32,
        vec2,
        vec2,
        wp.array2d(dtype=vec2),
    ]


for _T in (wp.float32, wp.float64):
    wp.overload(collision_forces_kernel, _signature(_T))
    wp.overload(collision_forces_kernel_per_env, _signature(_T, per_env=True))


def launch_collision_forces(
    pos: wp.array,
    vel: wp.array,
    params: AgentParams,
    neighbor_idx: wp.array,
    neighbor_count: wp.array,
    obs_pos: wp.array,
    obs_radius: wp.array,
    obs_type: wp.array,
    obs_angle: wp.array,  # 2d [n_envs, n_obs]
    obs_half: wp.array,
    obs_vel: wp.array,  # 2d [n_envs, n_obs]
    obs_ang_vel: wp.array,  # 2d [n_envs, n_obs]
    forces: wp.array,  # output
    *,
    n_obstacles: int,
    k: float,
    c: float,
    margin: float,
    sub_dt: float,
    max_overlap: float,
    soft_walls: bool,
    bounds_min,
    bounds_max,
    dtype,
) -> None:
    """Launch the soft-contact force pass. Everything past the arrays is keyword-only:
    the scalar tail is three floats and a bool in a row, where a swapped argument would be
    invisible at the call site."""
    n_envs, n_agents = pos.shape
    if params.floats_per_env is None:
        kernel, floats = collision_forces_kernel, params.floats
    else:
        kernel, floats = collision_forces_kernel_per_env, params.floats_per_env
    wp.launch(
        kernel,
        dim=(n_envs, n_agents),
        inputs=[
            pos,
            vel,
            floats,
            neighbor_idx,
            neighbor_count,
            obs_pos,
            obs_radius,
            obs_type,
            obs_angle,
            obs_half,
            obs_vel,
            obs_ang_vel,
            wp.int32(n_obstacles),
            dtype(k),
            dtype(c),
            dtype(margin),
            dtype(sub_dt),
            dtype(max_overlap),
            wp.int32(1 if soft_walls else 0),
            bounds_min,
            bounds_max,
        ],
        outputs=[forces],
        device=pos.device,
    )
