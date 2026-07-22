"""Differentiable soft-collision (spring-damper) forces.

Gather-based: each thread sums the forces acting on its own agent from its
padded neighbor list, obstacles, and soft walls. Symmetric pairs are computed
twice — no atomics, so the result is deterministic and race-free. Gradients
flow through positions/velocities; the neighbor list itself is a fixed
discrete structure within a substep.

Note: the damping term can momentarily turn attractive when agents separate
fast (classic spring-damper artifact); keep ``collision_c`` moderate.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from wmas.core.state import VEC2
from wmas.dynamics.base import P_RADIUS, AgentParams

_EPS2 = 1.0e-10  # distance^2 floor: keeps sqrt adjoint finite for coincident points

# Obstacle shape tags (mirror wmas.core.config.ObstacleShape).
SHAPE_CIRCLE = wp.constant(0)
SHAPE_BOX = wp.constant(1)
SHAPE_SEGMENT = wp.constant(2)


@wp.func
def _pair_force(d: Any, rel_v: Any, min_dist: Any, k: Any, c: Any):
    """Spring-damper repulsion along d (from other to self) within min_dist."""
    zero = type(k)(0.0)
    dist = wp.sqrt(wp.max(wp.dot(d, d), type(k)(_EPS2)))
    f = type(d)(zero, zero)
    overlap = min_dist - dist
    if overlap > zero:
        n = d / dist
        f = (k * overlap) * n - (c * wp.dot(rel_v, n)) * n
    return f


@wp.func
def _normal_force(n: Any, overlap: Any, rel_v: Any, k: Any, c: Any):
    """Spring-damper along a precomputed unit normal ``n`` (n points from the
    surface toward the agent). Used when the normal is known analytically (box
    SDF) rather than derived from a separation vector."""
    return (k * overlap) * n - (c * wp.dot(rel_v, n)) * n


@wp.func
def _closest_on_segment(p: Any, center: Any, angle: Any, half_len: Any):
    """Closest point on a segment core (center, orientation ``angle``, half
    length ``half_len``) to ``p``. Differentiable (clamp subgradient)."""
    d = type(p)(wp.cos(angle), wp.sin(angle))
    t = wp.clamp(wp.dot(p - center, d), -half_len, half_len)
    return center + t * d


@wp.func
def _box_force(p: Any, v: Any, center: Any, angle: Any, half: Any, reach: Any, k: Any, c: Any):
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
        f = _normal_force(n, overlap, v, k, c)
    return f


@wp.kernel
def collision_forces_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    params: wp.array2d(dtype=Any),
    neighbor_idx: wp.array3d(dtype=wp.int32),
    neighbor_count: wp.array2d(dtype=wp.int32),
    obs_pos: wp.array2d(dtype=Any),
    obs_radius: wp.array(dtype=Any),
    obs_type: wp.array(dtype=wp.int32),
    obs_angle: wp.array(dtype=Any),
    obs_half: wp.array(dtype=Any),
    n_obstacles: wp.int32,
    k: Any,
    c: Any,
    margin: Any,
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

    for n_i in range(neighbor_count[e, a]):
        b = neighbor_idx[e, a, n_i]
        f += _pair_force(p - pos[e, b], v - vel[e, b], ra + params[b, P_RADIUS] + margin, k, c)

    for o in range(n_obstacles):
        center = obs_pos[e, o]
        st = obs_type[o]
        if st == SHAPE_BOX:
            # box surface is the boundary itself; agent inflated by ra + margin
            f += _box_force(p, v, center, obs_angle[o], obs_half[o], ra + margin, k, c)
        elif st == SHAPE_SEGMENT:
            cp = _closest_on_segment(p, center, obs_angle[o], obs_half[o][0])
            f += _pair_force(p - cp, v, ra + obs_radius[o] + margin, k, c)
        else:  # SHAPE_CIRCLE
            f += _pair_force(p - center, v, ra + obs_radius[o] + margin, k, c)

    if soft_walls == 1:
        reach = ra + margin
        fx = zero
        fy = zero
        pen = reach - (p[0] - bounds_min[0])  # left wall, inward normal (+1, 0)
        if pen > zero:
            fx += k * pen - c * v[0]
        pen = reach - (bounds_max[0] - p[0])  # right wall, inward normal (-1, 0)
        if pen > zero:
            fx += -(k * pen) - c * v[0]
        pen = reach - (p[1] - bounds_min[1])  # bottom wall
        if pen > zero:
            fy += k * pen - c * v[1]
        pen = reach - (bounds_max[1] - p[1])  # top wall
        if pen > zero:
            fy += -(k * pen) - c * v[1]
        f += type(p)(fx, fy)

    forces[e, a] = f


def _signature(dtype) -> list:
    vec2 = VEC2[dtype]
    return [
        wp.array2d(dtype=vec2),
        wp.array2d(dtype=vec2),
        wp.array2d(dtype=dtype),
        wp.array3d(dtype=wp.int32),
        wp.array2d(dtype=wp.int32),
        wp.array2d(dtype=vec2),
        wp.array(dtype=dtype),
        wp.array(dtype=wp.int32),
        wp.array(dtype=dtype),
        wp.array(dtype=vec2),
        wp.int32,
        dtype,
        dtype,
        dtype,
        wp.int32,
        vec2,
        vec2,
        wp.array2d(dtype=vec2),
    ]


for _T in (wp.float32, wp.float64):
    wp.overload(collision_forces_kernel, _signature(_T))


def launch_collision_forces(
    pos: wp.array,
    vel: wp.array,
    params: AgentParams,
    neighbor_idx: wp.array,
    neighbor_count: wp.array,
    obs_pos: wp.array,
    obs_radius: wp.array,
    obs_type: wp.array,
    obs_angle: wp.array,
    obs_half: wp.array,
    n_obstacles: int,
    k: float,
    c: float,
    margin: float,
    soft_walls: bool,
    bounds_min,
    bounds_max,
    forces: wp.array,
    dtype,
) -> None:
    n_envs, n_agents = pos.shape
    wp.launch(
        collision_forces_kernel,
        dim=(n_envs, n_agents),
        inputs=[
            pos,
            vel,
            params.floats,
            neighbor_idx,
            neighbor_count,
            obs_pos,
            obs_radius,
            obs_type,
            obs_angle,
            obs_half,
            wp.int32(n_obstacles),
            dtype(k),
            dtype(c),
            dtype(margin),
            wp.int32(1 if soft_walls else 0),
            bounds_min,
            bounds_max,
        ],
        outputs=[forces],
        device=pos.device,
    )
