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

**Box agents** (:attr:`~swarp.dynamics.base.AgentShape.BOX`) run their own kernel,
:func:`collision_forces_box_kernel`, routed by ``Stepper.has_box_agents``, so a disc-only
world keeps launching the disc kernels above unchanged. Every box pair goes through one
contact function, :func:`obb_contact`, and one frictionless spring-damper with the same
normal law, damped by the closing velocity of the two material points at the contact
point (``v + om x r`` on both sides). A box agent feels no contact *torque* yet: its
heading is its own (driven by its dynamics model), and the contact only pushes its centre.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import concrete, register
from swarp.core.config import ObstacleShape
from swarp.core.state import VEC2
from swarp.dynamics.base import P_HALF_X, P_HALF_Y, P_MASS, P_RADIUS, P_SHAPE, AgentParams

# Lean launches (no grid-stride loop); see docs/performance.md.
wp.set_module_options({"default_grid_stride": False})

_EPS2 = 1.0e-10  # distance^2 floor: keeps sqrt adjoint finite for coincident points

# Kernel-side obstacle shape tags, derived from the enum so there is one source of truth.
SHAPE_CIRCLE = wp.constant(int(ObstacleShape.CIRCLE))
SHAPE_BOX = wp.constant(int(ObstacleShape.BOX))
SHAPE_SEGMENT = wp.constant(int(ObstacleShape.SEGMENT))

#: Compile options of every box-world kernel: no floating-point contraction (FMA), so
#: the layered and fused paths, and the agent and body sides of a contact, round each
#: operation alike whatever nvcc inlines where. See docs/performance.md.
BOX_MODULE_OPTIONS = {"fuse_fp": False}


@wp.func
def shape_bound(st: wp.int32, r: Any, h: Any):
    """Radius of the disc about an obstacle's centre that holds the whole shape: a
    circle's radius, a box's half-diagonal, a capsule's half-length plus its radius."""
    b = r
    if st == SHAPE_BOX:
        b = wp.length(h)
    elif st == SHAPE_SEGMENT:
        b = h[0] + r
    return b


@wp.func
def bounds_apart(d: Any, reach: Any):
    """Whether two shapes whose bounding discs have centres ``d`` apart are too far to
    touch, ``reach`` being the two bounding radii plus the contact margin.

    The broad phase of every obstacle loop: each contact function starts from a zero force
    and adds one only while its overlap is positive, and the overlap is negative whenever
    the bounding discs are farther apart than ``reach``, so skipping such a pair changes
    no force. The slack (1e-4 relative, 1e-6 absolute) keeps a pair whose exact test would
    round to contact inside the loop."""
    lim = reach * type(reach)(1.0001) + type(reach)(1.0e-6)
    return wp.dot(d, d) > lim * lim


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
def pair_force(
    d: Any, rel_v: Any, min_dist: Any, k: Any, c: Any, damp_denom: Any, max_overlap: Any
):
    """Spring-damper repulsion along ``d`` (from other to self) within ``min_dist``.

    The circle-circle half of the contact law, and the one reusable piece of it: a
    custom contact model built on top of this engine calls it per pair instead of
    reimplementing the normal, the depth saturation and the closing-velocity damping.
    Callable from any Warp kernel (agent-agent, agent-obstacle, obstacle-obstacle all go
    through it here) and generic over ``float32``/``float64``.

    Args:
        d: separation vector from the other body's centre to this one's.
        rel_v: closing velocity, ``self - other``; only its normal component damps.
        min_dist: sum of the two radii plus the contact margin — contact below this.
        k: contact stiffness (N/m).
        c: contact damping.
        damp_denom: divisor applied to the whole normal coefficient, which is how the
            per-agent mass normalization enters.
        max_overlap: overlap saturation. Above zero the depth is passed through
            ``max_overlap * tanh(overlap / max_overlap)``, bounding the impulse a deep
            interpenetration can inject while leaving the shallow regime linear.

    Returns:
        The force on *self*, zero outside contact. Apply ``-f`` to the other body:
        the caller owns Newton's third law, this returns one side.
    """
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
def closest_on_segment(p: Any, center: Any, angle: Any, half_len: Any):
    """Closest point on a segment core to ``p``.

    The segment is its centre, its orientation ``angle`` (rad), and its half length —
    a capsule's core before the radius is added. Pair it with :func:`pair_force` to get
    a capsule contact: the closest point is the effective circle centre, so
    ``pair_force(p - closest_on_segment(...), ...)`` with ``min_dist = agent_radius +
    capsule_radius + margin`` is the whole law.

    Differentiable through the clamp (subgradient at the endpoints), so it is safe on a
    taped step.
    """
    d = type(p)(wp.cos(angle), wp.sin(angle))
    t = wp.clamp(wp.dot(p - center, d), -half_len, half_len)
    return center + t * d


@wp.func
def box_force(
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
    """Contact force from an oriented box, via its signed distance field.

    The box half of the contact law, and the counterpart to :func:`pair_force` for a
    custom contact model. The SDF (rather than a closest-point clamp) is what keeps an
    agent whose *centre* has penetrated the box pushed back out — a clamp has a
    zero-force interior dead-zone, so a deep hit sails straight through.

    Args:
        p: agent centre. ``v``: agent velocity.
        center, angle, half: the box's world pose and its half-extents.
        reach: ``agent_radius + margin``. Contact when the signed distance from ``p`` to
            the box surface drops below this.
        k, c, damp_denom, max_overlap: as in :func:`pair_force`.
        v_obs, om_obs: the box's own linear and angular velocity, so damping sees the
            *closing* velocity rather than the agent's absolute one.

    Returns:
        The force on the agent. The box's reaction is ``-f`` at the contact point, which
        is what :mod:`swarp.core.bodies` integrates for a movable box.
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


# ---------------------------------------------------------------- box agents


@wp.func
def _cross_w(w: Any, r: Any):
    """``om x r`` for a planar angular velocity ``w``: the velocity a lever ``r`` adds."""
    return type(r)(-w * r[1], w * r[0])


@wp.func
def _point_box(v: Any, center: Any, ca: Any, sa: Any, half: Any):
    """``(distance, closest point)`` from a point ``v`` *outside* an oriented box.

    The exterior half of :func:`box_force`'s SDF (same frame rotation, same clamp); the
    callers only evaluate it on a vertex of a box disjoint from this one, which is
    always outside. The distance is floored like :func:`pair_force`'s, so the adjoint of
    the square root stays finite."""
    d = v - center
    lx = ca * d[0] + sa * d[1]
    ly = -sa * d[0] + ca * d[1]
    cx = wp.clamp(lx, -half[0], half[0])
    cy = wp.clamp(ly, -half[1], half[1])
    ox = lx - cx
    oy = ly - cy
    dist = wp.sqrt(wp.max(ox * ox + oy * oy, type(ca)(_EPS2)))
    q = center + type(v)(ca * cx - sa * cy, sa * cx + ca * cy)
    return dist, q


@wp.func
def _vertex_vs_box(
    v: Any, center: Any, ca: Any, sa: Any, half: Any, flip: Any, best: Any, bn: Any, bx: Any
):
    """Keep ``(best, normal, point)`` or replace it by vertex ``v`` against a box.

    ``flip`` is ``+1`` when the vertex belongs to box A (the normal ``v - q`` already
    points from B to A) and ``-1`` when it belongs to B. The contact point is the middle
    of the closest pair, so it lies on neither surface but between them."""
    dist, q = _point_box(v, center, ca, sa, half)
    rb = best
    rn = bn
    rx = bx
    if dist < best:
        rb = dist
        rn = (v - q) * (flip / dist)
        rx = (v + q) * type(ca)(0.5)
    return rb, rn, rx


@wp.func
def obb_contact(cA: Any, thA: Any, hA: Any, cB: Any, thB: Any, hB: Any, reach: Any):
    """Signed distance, normal and contact point between two oriented boxes.

    Box A is centred at ``cA``, rotated by ``thA``, with half-extents ``hA`` along its
    own axes; likewise B. Returns ``(s, n, x)``:

    - ``s``, the signed distance: the exact Euclidean gap between the two rectangles
      when they are disjoint, ``-`` (the smallest overlap over the four separating axes)
      when they intersect. Both branches tend to ``0`` at touching, so ``s`` is
      continuous through contact.
    - ``n``, the unit contact normal, pointing **from B to A** (the direction that pushes
      A out).
    - ``x``, the world contact point, where both bodies' material-point velocities and
      the lever arm of a torque are read.

    Separating-axis test over the four face normals first. The largest projection gap is
    a lower bound on the distance (projection onto a unit axis is 1-Lipschitz), so when
    it exceeds ``reach`` the boxes cannot be in contact and the function returns early
    with that gap. Disjoint but within ``reach``, the closest pair of two disjoint convex
    polygons always involves a vertex, so the distance is the minimum over the eight
    vertex-versus-other-box distances (the exterior SDF of :func:`box_force`), and ``n``
    and ``x`` come from the winning vertex. Overlapping, the axis of least overlap is the
    normal (oriented B to A) and ``x`` sits on that axis' reference face, at the centre
    of the interval where the two boxes' projections onto the face overlap.

    A capsule is this with a degenerate box ``hB = (half_len, 0)`` and ``reach`` grown by
    its radius: the distance is then to the core segment, so the end caps come out
    exactly round. Straight-line code (no loops), so it differentiates on a taped step,
    through the headings too.
    """
    zero = type(reach)(0.0)
    one = type(reach)(1.0)
    half = type(reach)(0.5)
    caA = wp.cos(thA)
    saA = wp.sin(thA)
    caB = wp.cos(thB)
    saB = wp.sin(thB)
    uA0 = type(cA)(caA, saA)
    uA1 = type(cA)(-saA, caA)
    uB0 = type(cA)(caB, saB)
    uB1 = type(cA)(-saB, caB)
    d = cA - cB
    a00 = wp.abs(wp.dot(uA0, uB0))
    a01 = wp.abs(wp.dot(uA0, uB1))
    a10 = wp.abs(wp.dot(uA1, uB0))
    a11 = wp.abs(wp.dot(uA1, uB1))
    dA0 = wp.dot(d, uA0)
    dA1 = wp.dot(d, uA1)
    dB0 = wp.dot(d, uB0)
    dB1 = wp.dot(d, uB1)

    # Projection gap along each face normal; the largest one is the SAT separation.
    s = wp.abs(dA0) - hA[0] - (hB[0] * a00 + hB[1] * a01)
    axis = wp.int32(0)
    t = wp.abs(dA1) - hA[1] - (hB[0] * a10 + hB[1] * a11)
    if t > s:
        s = t
        axis = wp.int32(1)
    t = wp.abs(dB0) - hB[0] - (hA[0] * a00 + hA[1] * a10)
    if t > s:
        s = t
        axis = wp.int32(2)
    t = wp.abs(dB1) - hB[1] - (hA[0] * a01 + hA[1] * a11)
    if t > s:
        s = t
        axis = wp.int32(3)

    n = type(cA)(zero, zero)
    x = cA
    if s > reach:
        n = type(cA)(zero, zero)  # no contact; s is a lower bound on the gap
    elif s > zero:
        # Disjoint: exact distance, attained at a vertex of one box against the other.
        eA0 = uA0 * hA[0]
        eA1 = uA1 * hA[1]
        eB0 = uB0 * hB[0]
        eB1 = uB1 * hB[1]
        best = type(reach)(1.0e30)
        best, n, x = _vertex_vs_box(cA + eA0 + eA1, cB, caB, saB, hB, one, best, n, x)
        best, n, x = _vertex_vs_box(cA - eA0 + eA1, cB, caB, saB, hB, one, best, n, x)
        best, n, x = _vertex_vs_box(cA - eA0 - eA1, cB, caB, saB, hB, one, best, n, x)
        best, n, x = _vertex_vs_box(cA + eA0 - eA1, cB, caB, saB, hB, one, best, n, x)
        best, n, x = _vertex_vs_box(cB + eB0 + eB1, cA, caA, saA, hA, -one, best, n, x)
        best, n, x = _vertex_vs_box(cB - eB0 + eB1, cA, caA, saA, hA, -one, best, n, x)
        best, n, x = _vertex_vs_box(cB - eB0 - eB1, cA, caA, saA, hA, -one, best, n, x)
        best, n, x = _vertex_vs_box(cB + eB0 - eB1, cA, caA, saA, hA, -one, best, n, x)
        s = best
    else:
        # Overlapping: the axis of least overlap, oriented B -> A, and its reference face.
        u = uA0
        tdir = uA1
        dn = dA0
        h_n = hA[0]
        h_t = hA[1]
        r_t = hB[0] * a10 + hB[1] * a11  # the incident box's reach along tdir
        if axis == 1:
            u = uA1
            tdir = uA0
            dn = dA1
            h_n = hA[1]
            h_t = hA[0]
            r_t = hB[0] * a00 + hB[1] * a01
        elif axis == 2:
            u = uB0
            tdir = uB1
            dn = dB0
            h_n = hB[0]
            h_t = hB[1]
            r_t = hA[0] * a01 + hA[1] * a11
        elif axis == 3:
            u = uB1
            tdir = uB0
            dn = dB1
            h_n = hB[1]
            h_t = hB[0]
            r_t = hA[0] * a00 + hA[1] * a10
        sgn = one
        if dn < zero:
            sgn = -one
        n = u * sgn
        # Reference face: A's face toward B for an A axis, B's face toward A for a B axis.
        c_ref = cA
        c_inc = cB
        xn = wp.dot(cA, n) - h_n
        if axis >= 2:
            c_ref = cB
            c_inc = cA
            xn = wp.dot(cB, n) + h_n
        ct_ref = wp.dot(c_ref, tdir)
        ct_inc = wp.dot(c_inc, tdir)
        lo = wp.max(ct_ref - h_t, ct_inc - r_t)
        hi = wp.min(ct_ref + h_t, ct_inc + r_t)
        x = n * xn + tdir * (half * (lo + hi))
    return s, n, x


@wp.func
def obb_force(
    cA: Any,
    thA: Any,
    hA: Any,
    vA: Any,
    wA: Any,
    cB: Any,
    thB: Any,
    hB: Any,
    vB: Any,
    wB: Any,
    reach: Any,
    k: Any,
    c: Any,
    damp_denom: Any,
    max_overlap: Any,
):
    """Spring-damper force on box A from box B, and the contact point it acts at.

    The box-box counterpart of :func:`pair_force`: contact when :func:`obb_contact`'s
    signed distance drops below ``reach`` (the contact margin, or a capsule's radius
    plus the margin), with the same normal law. The damper reads the closing velocity of
    the two *material points* at the contact point, ``(vA + wA x rA) - (vB + wB x rB)``,
    so a spinning box is damped against how its surface actually moves.

    Returns ``(f, x)``: the force on A (apply ``-f`` to B; the caller owns the third
    law) and the world contact point, which a rigid body turns into a torque.
    """
    zero = type(k)(0.0)
    s, n, x = obb_contact(cA, thA, hA, cB, thB, hB, reach)
    f = type(cA)(zero, zero)
    overlap = reach - s
    if overlap > zero:
        vpa = vA + _cross_w(wA, x - cA)
        vpb = vB + _cross_w(wB, x - cB)
        f = _normal_force(n, overlap, vpa - vpb, k, c, damp_denom, max_overlap)
    return f, x


@wp.func
def _agent_pair_force(
    a: wp.int32,
    b: wp.int32,
    shape_a: Any,
    shape_b: Any,
    p: Any,
    th: Any,
    v: Any,
    w: Any,
    ra: Any,
    ha: Any,
    pb: Any,
    thb: Any,
    vb: Any,
    wb: Any,
    rb: Any,
    hb: Any,
    margin: Any,
    k: Any,
    c: Any,
    damp_denom: Any,
    max_overlap: Any,
):
    """Contact force on agent ``a`` from agent ``b``, either of which may be a box.

    disc-disc is :func:`pair_force` exactly as the disc kernel computes it; disc-box is
    :func:`box_force` with the disc as the point, negated when ``a`` is the box. Box-box
    evaluates :func:`obb_force` with the **lower index as A** on both threads and negates
    on the higher one, so the two threads compute one contact (same normal, same contact
    point, same closing velocity) and the pair's forces cancel exactly for equal masses.
    """
    half = type(k)(0.5)
    f = type(p)(type(k)(0.0), type(k)(0.0))
    if shape_a < half:
        if shape_b < half:
            f = pair_force(p - pb, v - vb, ra + rb + margin, k, c, damp_denom, max_overlap)
        else:
            f = box_force(p, v, pb, thb, hb, ra + margin, k, c, damp_denom, max_overlap, vb, wb)
    else:
        if shape_b < half:
            f = -box_force(pb, vb, p, th, ha, rb + margin, k, c, damp_denom, max_overlap, v, w)
        elif a < b:
            f, _x = obb_force(
                p, th, ha, v, w, pb, thb, hb, vb, wb, margin, k, c, damp_denom, max_overlap
            )
        else:
            fo, _x = obb_force(
                pb, thb, hb, vb, wb, p, th, ha, v, w, margin, k, c, damp_denom, max_overlap
            )
            f = -fo
    return f


@wp.func
def _box_obstacle_force(
    p: Any,
    th: Any,
    v: Any,
    w: Any,
    ha: Any,
    center: Any,
    st: wp.int32,
    angle: Any,
    obs_r: Any,
    obs_h: Any,
    vo: Any,
    omo: Any,
    k: Any,
    c: Any,
    margin: Any,
    damp_denom: Any,
    max_overlap: Any,
):
    """Force on a box agent from one obstacle, and the contact point it acts at.

    A ``BOX`` obstacle is :func:`obb_force` with ``reach = margin``; a ``SEGMENT`` is the
    degenerate box ``(half_len, 0)`` with ``reach = radius + margin``, i.e. the exact
    capsule; a ``CIRCLE`` is :func:`box_force` with the circle centre as the point,
    negated, acting at the circle centre (a frictionless normal through it). The movable
    body pass calls this very function and negates it, so action equals reaction."""
    zero = type(k)(0.0)
    f = type(p)(zero, zero)
    x = center
    if st == SHAPE_BOX:
        f, x = obb_force(
            p, th, ha, v, w, center, angle, obs_h, vo, omo, margin, k, c, damp_denom, max_overlap
        )
    elif st == SHAPE_SEGMENT:
        f, x = obb_force(
            p, th, ha, v, w, center, angle, type(p)(obs_h[0], zero), vo, omo, obs_r + margin,
            k, c, damp_denom, max_overlap,
        )
    else:  # SHAPE_CIRCLE
        f = -box_force(center, vo, p, th, ha, obs_r + margin, k, c, damp_denom, max_overlap, v, w)
    return f, x


@wp.func
def _obstacle_forces_box(
    e: wp.int32,
    p: Any,
    th: Any,
    v: Any,
    w: Any,
    ha: Any,
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
):
    """:func:`_obstacle_forces` for a box agent of half-extents ``ha`` at heading ``th``."""
    zero = type(k)(0.0)
    f = type(p)(zero, zero)
    ra = wp.length(ha)
    for o in range(n_obstacles):
        st = obs_type[o]
        if bounds_apart(
            p - obs_pos[e, o], ra + margin + shape_bound(st, obs_radius[o], obs_half[o])
        ):
            continue
        fo, _x = _box_obstacle_force(
            p, th, v, w, ha, obs_pos[e, o], obs_type[o], obs_angle[e, o], obs_radius[o],
            obs_half[o], obs_vel[e, o], obs_ang_vel[e, o], k, c, margin, damp_denom,
            max_overlap,
        )
        f += fo
    return f


@wp.func
def _sgn0(x: Any):
    """Sign of ``x``, and ``0`` at zero (``wp.sign`` returns ``+1`` there)."""
    zero = type(x)(0.0)
    r = zero
    if x > zero:
        r = type(x)(1.0)
    elif x < zero:
        r = type(x)(-1.0)
    return r


@wp.func
def _wall_forces_box(
    p: Any,
    th: Any,
    v: Any,
    w: Any,
    ha: Any,
    k: Any,
    c: Any,
    margin: Any,
    damp_denom: Any,
    max_overlap: Any,
    bounds_min: Any,
    bounds_max: Any,
):
    """:func:`_wall_forces` for a box agent: the walls see its support extents.

    A box at heading ``th`` reaches ``ex = hx|cos| + hy|sin|`` along x and
    ``ey = hx|sin| + hy|cos|`` along y from its centre, which replace the disc's radius.
    The damper reads the velocity of the support vertex touching each wall,
    ``v + w x r``; on a face flush with a wall (``sin`` or ``cos`` exactly zero) the
    vertex choice degenerates to the face centre, whose rotation adds nothing normal."""
    zero = type(k)(0.0)
    ca = wp.cos(th)
    sa = wp.sin(th)
    ex = ha[0] * wp.abs(ca) + ha[1] * wp.abs(sa)
    ey = ha[0] * wp.abs(sa) + ha[1] * wp.abs(ca)
    sc = _sgn0(ca)
    ss = _sgn0(sa)
    # Lever of the support vertex toward -x (its y offset) and toward -y (its x offset);
    # the +x and +y vertices are their reflections through the centre.
    rl_y = -sc * ha[0] * sa + ss * ha[1] * ca
    rb_x = -ss * ha[0] * ca + sc * ha[1] * sa
    fx = zero
    fy = zero
    pen = ex + margin - (p[0] - bounds_min[0])  # left wall, inward normal (+1, 0)
    if pen > zero:
        fx += _normal_coeff(pen, v[0] - w * rl_y, k, c, damp_denom, max_overlap)
    pen = ex + margin - (bounds_max[0] - p[0])  # right wall, inward normal (-1, 0)
    if pen > zero:
        fx += -_normal_coeff(pen, -(v[0] + w * rl_y), k, c, damp_denom, max_overlap)
    pen = ey + margin - (p[1] - bounds_min[1])  # bottom wall
    if pen > zero:
        fy += _normal_coeff(pen, v[1] + w * rb_x, k, c, damp_denom, max_overlap)
    pen = ey + margin - (bounds_max[1] - p[1])  # top wall
    if pen > zero:
        fy += -_normal_coeff(pen, -(v[1] - w * rb_x), k, c, damp_denom, max_overlap)
    return type(p)(fx, fy)


@wp.func
def _obstacle_forces(
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
):
    """Contact forces on an agent at ``p`` (radius ``ra``) from every obstacle.

    ``obs_vel``/``obs_ang_vel`` are the obstacles' own velocities (zero for a genuinely
    static obstacle), so the damper sees the closing velocity in every case."""
    zero = type(k)(0.0)
    f = type(p)(zero, zero)

    for o in range(n_obstacles):
        center = obs_pos[e, o]
        st = obs_type[o]
        if bounds_apart(p - center, ra + margin + shape_bound(st, obs_radius[o], obs_half[o])):
            continue
        vo = obs_vel[e, o]
        omo = obs_ang_vel[e, o]
        if st == SHAPE_BOX:
            # box surface is the boundary itself; agent inflated by ra + margin
            f += box_force(
                p, v, center, obs_angle[e, o], obs_half[o], ra + margin, k, c, damp_denom,
                max_overlap, vo, omo,
            )
        elif st == SHAPE_SEGMENT:
            cp = closest_on_segment(p, center, obs_angle[e, o], obs_half[o][0])
            # surface point velocity: v_obs + om x (cp - centre)
            r = cp - center
            vs = vo + type(p)(-omo * r[1], omo * r[0])
            f += pair_force(
                p - cp, v - vs, ra + obs_radius[o] + margin, k, c, damp_denom, max_overlap
            )
        else:  # SHAPE_CIRCLE
            # a spinning disc has no normal-direction surface motion (frictionless)
            f += pair_force(
                p - center, v - vo, ra + obs_radius[o] + margin, k, c, damp_denom, max_overlap
            )
    return f


@wp.func
def _wall_forces(
    p: Any,
    v: Any,
    ra: Any,
    k: Any,
    c: Any,
    margin: Any,
    damp_denom: Any,
    max_overlap: Any,
    bounds_min: Any,
    bounds_max: Any,
):
    """Contact force on an agent at ``p`` (radius ``ra``) from the four soft walls."""
    zero = type(k)(0.0)
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
    return type(p)(fx, fy)


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
    differs between them. The two halves are separate functions so the fused substep
    chain (swarp/core/fused_substeps.py) can compile out the one a world does not use."""
    f = _obstacle_forces(
        e, p, v, ra, obs_pos, obs_radius, obs_type, obs_angle, obs_half, obs_vel,
        obs_ang_vel, n_obstacles, k, c, margin, damp_denom, max_overlap,
    )
    if soft_walls == 1:
        f += _wall_forces(p, v, ra, k, c, margin, damp_denom, max_overlap, bounds_min, bounds_max)
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
        f += pair_force(
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
        f += pair_force(
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


@wp.func
def _static_forces_shaped(
    e: wp.int32,
    shape_a: Any,
    p: Any,
    th: Any,
    v: Any,
    w: Any,
    ra: Any,
    ha: Any,
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
    """:func:`_static_forces` for an agent of either shape: a disc takes the disc path
    verbatim, a box its oriented counterpart. Obstacles first, then walls, the order the
    disc kernel and the fused chain sum them in."""
    f = type(p)(type(k)(0.0), type(k)(0.0))
    if shape_a < type(k)(0.5):
        f = _static_forces(
            e, p, v, ra, obs_pos, obs_radius, obs_type, obs_angle, obs_half, obs_vel,
            obs_ang_vel, n_obstacles, k, c, margin, damp_denom, max_overlap, soft_walls,
            bounds_min, bounds_max,
        )
    else:
        f = _obstacle_forces_box(
            e, p, th, v, w, ha, obs_pos, obs_radius, obs_type, obs_angle, obs_half, obs_vel,
            obs_ang_vel, n_obstacles, k, c, margin, damp_denom, max_overlap,
        )
        if soft_walls == 1:
            f += _wall_forces_box(
                p, th, v, w, ha, k, c, margin, damp_denom, max_overlap, bounds_min, bounds_max
            )
    return f


@wp.kernel(module="unique", module_options=BOX_MODULE_OPTIONS, grid_stride=False)
def collision_forces_box_kernel(
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
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
    """:func:`collision_forces_kernel` for a world holding box agents.

    Same gather, same contact law, but every pair dispatches on the two agents'
    :data:`~swarp.dynamics.base.P_SHAPE` (see :func:`_agent_pair_force`), and it reads the
    headings and angular velocities a box contact needs. Only launched when the world
    has a box agent, so the disc kernel never pays for the extra loads. Compiled without
    FP contraction (:data:`BOX_MODULE_OPTIONS`), like every kernel a box world runs."""
    e, a = wp.tid()
    p = pos[e, a]
    th = theta[e, a]
    v = vel[e, a]
    w = ang_vel[e, a]
    ra = params[a, P_RADIUS]
    sha = params[a, P_SHAPE]
    ha = type(p)(params[a, P_HALF_X], params[a, P_HALF_Y])
    zero = type(k)(0.0)
    f = type(p)(zero, zero)
    damp_denom = type(k)(1.0) + c * sub_dt / params[a, P_MASS]

    for n_i in range(neighbor_count[e, a]):
        b = neighbor_idx[e, a, n_i]
        f += _agent_pair_force(
            a, b, sha, params[b, P_SHAPE], p, th, v, w, ra, ha, pos[e, b], theta[e, b],
            vel[e, b], ang_vel[e, b], params[b, P_RADIUS],
            type(p)(params[b, P_HALF_X], params[b, P_HALF_Y]), margin, k, c, damp_denom,
            max_overlap,
        )

    f += _static_forces_shaped(
        e, sha, p, th, v, w, ra, ha, obs_pos, obs_radius, obs_type, obs_angle, obs_half,
        obs_vel, obs_ang_vel, n_obstacles, k, c, margin, damp_denom, max_overlap, soft_walls,
        bounds_min, bounds_max,
    )
    forces[e, a] = f


@wp.kernel(module="unique", module_options=BOX_MODULE_OPTIONS, grid_stride=False)
def collision_forces_box_kernel_per_env(
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
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
    """Per-env-parameter twin of :func:`collision_forces_box_kernel`."""
    e, a = wp.tid()
    p = pos[e, a]
    th = theta[e, a]
    v = vel[e, a]
    w = ang_vel[e, a]
    ra = params[e, a, P_RADIUS]
    sha = params[e, a, P_SHAPE]
    ha = type(p)(params[e, a, P_HALF_X], params[e, a, P_HALF_Y])
    zero = type(k)(0.0)
    f = type(p)(zero, zero)
    damp_denom = type(k)(1.0) + c * sub_dt / params[e, a, P_MASS]

    for n_i in range(neighbor_count[e, a]):
        b = neighbor_idx[e, a, n_i]
        f += _agent_pair_force(
            a, b, sha, params[e, b, P_SHAPE], p, th, v, w, ra, ha, pos[e, b], theta[e, b],
            vel[e, b], ang_vel[e, b], params[e, b, P_RADIUS],
            type(p)(params[e, b, P_HALF_X], params[e, b, P_HALF_Y]), margin, k, c, damp_denom,
            max_overlap,
        )

    f += _static_forces_shaped(
        e, sha, p, th, v, w, ra, ha, obs_pos, obs_radius, obs_type, obs_angle, obs_half,
        obs_vel, obs_ang_vel, n_obstacles, k, c, margin, damp_denom, max_overlap, soft_walls,
        bounds_min, bounds_max,
    )
    forces[e, a] = f


# ------------------------------------------------- wheel-faithful contacts (torques)
#
# The torque twins of the box-agent force functions, for WorldConfig.wheel_contacts: the
# same forces, summed in the same order, plus each contact's torque about the agent
# centre. Only the wheel kernels below call them, so a box world without wheel contacts
# compiles none of it.


@wp.func
def _cross2(r: Any, f: Any):
    return r[0] * f[1] - r[1] * f[0]


@wp.func
def _box_sd(v: Any, center: Any, ca: Any, sa: Any, half: Any):
    """Signed distance from point ``v`` to an oriented box: the exterior distance, or
    minus the distance to the nearest face inside."""
    d = v - center
    lx = wp.abs(ca * d[0] + sa * d[1]) - half[0]
    ly = wp.abs(-sa * d[0] + ca * d[1]) - half[1]
    ox = wp.max(lx, type(ca)(0.0))
    oy = wp.max(ly, type(ca)(0.0))
    return wp.sqrt(ox * ox + oy * oy) + wp.min(wp.max(lx, ly), type(ca)(0.0))


@wp.func
def _manifold_point(
    cA: Any, thA: Any, hA: Any, cB: Any, thB: Any, hB: Any, reach: Any, x: Any
):
    """Where a box-box contact pushes, for its torque: the vertices of either box that
    reach into the other (within ``reach``), averaged with their depths as weights.

    :func:`obb_contact`'s point sits mid-way along the overlap, which is right for the
    force but carries no torque when a face lies nearly flush on another: there it is the
    deeper corner that the contact pushes on. Falls back to ``x`` when no vertex is in
    reach (an edge-on crossing)."""
    zero = type(reach)(0.0)
    caA = wp.cos(thA)
    saA = wp.sin(thA)
    caB = wp.cos(thB)
    saB = wp.sin(thB)
    eA0 = type(cA)(caA * hA[0], saA * hA[0])
    eA1 = type(cA)(-saA * hA[1], caA * hA[1])
    eB0 = type(cA)(caB * hB[0], saB * hB[0])
    eB1 = type(cA)(-saB * hB[1], caB * hB[1])
    acc = type(cA)(zero, zero)
    wsum = zero
    for i in range(4):
        sx = type(reach)(1.0)
        sy = type(reach)(1.0)
        if i == 1 or i == 2:
            sx = -sx
        if i >= 2:
            sy = -sy
        va = cA + eA0 * sx + eA1 * sy
        da = wp.max(reach - _box_sd(va, cB, caB, saB, hB), zero)
        acc += va * da
        wsum += da
        vb = cB + eB0 * sx + eB1 * sy
        db = wp.max(reach - _box_sd(vb, cA, caA, saA, hA), zero)
        acc += vb * db
        wsum += db
    out = x
    if wsum > zero:
        out = acc / wsum
    return out


@wp.func
def _agent_pair_force_torque(
    a: wp.int32,
    b: wp.int32,
    shape_a: Any,
    shape_b: Any,
    p: Any,
    th: Any,
    v: Any,
    w: Any,
    ra: Any,
    ha: Any,
    pb: Any,
    thb: Any,
    vb: Any,
    wb: Any,
    rb: Any,
    hb: Any,
    margin: Any,
    k: Any,
    c: Any,
    damp_denom: Any,
    max_overlap: Any,
):
    """:func:`_agent_pair_force` and its torque about agent ``a``'s centre: ``0`` for a
    disc ``a`` (its contacts pass through its centre), the lever to the contact point for
    a box."""
    half = type(k)(0.5)
    zero = type(k)(0.0)
    f = type(p)(zero, zero)
    tau = zero
    if shape_a < half:
        if shape_b < half:
            f = pair_force(p - pb, v - vb, ra + rb + margin, k, c, damp_denom, max_overlap)
        else:
            f = box_force(p, v, pb, thb, hb, ra + margin, k, c, damp_denom, max_overlap, vb, wb)
    else:
        if shape_b < half:
            f = -box_force(pb, vb, p, th, ha, rb + margin, k, c, damp_denom, max_overlap, v, w)
            fn = wp.length(f)
            if fn > zero:
                tau = _cross2(pb + f * (rb / fn) - p, f)  # the disc's point nearest to a
        else:
            x = p
            if a < b:
                f, x = obb_force(
                    p, th, ha, v, w, pb, thb, hb, vb, wb, margin, k, c, damp_denom, max_overlap
                )
            else:
                fo, x = obb_force(
                    pb, thb, hb, vb, wb, p, th, ha, v, w, margin, k, c, damp_denom, max_overlap
                )
                f = -fo
            if wp.length_sq(f) > zero:
                tau = _cross2(_manifold_point(p, th, ha, pb, thb, hb, margin, x) - p, f)
    return f, tau


@wp.func
def _static_forces_torque(
    e: wp.int32,
    shape_a: Any,
    p: Any,
    th: Any,
    v: Any,
    w: Any,
    ra: Any,
    ha: Any,
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
    """:func:`_static_forces_shaped` and the torque of a box agent's obstacle and wall
    contacts about its centre (a disc's is ``0``)."""
    zero = type(k)(0.0)
    f = type(p)(zero, zero)
    tau = zero
    if shape_a < type(k)(0.5):
        f = _static_forces(
            e, p, v, ra, obs_pos, obs_radius, obs_type, obs_angle, obs_half, obs_vel,
            obs_ang_vel, n_obstacles, k, c, margin, damp_denom, max_overlap, soft_walls,
            bounds_min, bounds_max,
        )
    else:
        rb = wp.length(ha)
        for o in range(n_obstacles):
            st = obs_type[o]
            if bounds_apart(
                p - obs_pos[e, o], rb + margin + shape_bound(st, obs_radius[o], obs_half[o])
            ):
                continue
            fo, x = _box_obstacle_force(
                p, th, v, w, ha, obs_pos[e, o], st, obs_angle[e, o], obs_radius[o],
                obs_half[o], obs_vel[e, o], obs_ang_vel[e, o], k, c, margin, damp_denom,
                max_overlap,
            )
            if wp.length_sq(fo) > zero:
                if st == SHAPE_CIRCLE:
                    x = x + fo * (obs_radius[o] / wp.length(fo))  # its point nearest the box
                elif st == SHAPE_BOX:
                    x = _manifold_point(p, th, ha, obs_pos[e, o], obs_angle[e, o], obs_half[o],
                                        margin, x)
                else:  # a segment: the degenerate box of its core, reached through its radius
                    x = _manifold_point(p, th, ha, obs_pos[e, o], obs_angle[e, o],
                                        type(p)(obs_half[o][0], zero), obs_radius[o] + margin, x)
            f += fo
            tau += _cross2(x - p, fo)
        if soft_walls == 1:
            # _wall_forces_box, wall by wall, each force at its support vertex.
            ca = wp.cos(th)
            sa = wp.sin(th)
            ex = ha[0] * wp.abs(ca) + ha[1] * wp.abs(sa)
            ey = ha[0] * wp.abs(sa) + ha[1] * wp.abs(ca)
            sc = _sgn0(ca)
            ss = _sgn0(sa)
            rl_y = -sc * ha[0] * sa + ss * ha[1] * ca
            rb_x = -ss * ha[0] * ca + sc * ha[1] * sa
            fx = zero
            fy = zero
            pen = ex + margin - (p[0] - bounds_min[0])
            if pen > zero:
                fl = _normal_coeff(pen, v[0] - w * rl_y, k, c, damp_denom, max_overlap)
                fx += fl
                tau += -rl_y * fl  # lever (-ex, rl_y)
            pen = ex + margin - (bounds_max[0] - p[0])
            if pen > zero:
                fr = -_normal_coeff(pen, -(v[0] + w * rl_y), k, c, damp_denom, max_overlap)
                fx += fr
                tau += rl_y * fr  # lever (ex, -rl_y)
            pen = ey + margin - (p[1] - bounds_min[1])
            if pen > zero:
                fb = _normal_coeff(pen, v[1] + w * rb_x, k, c, damp_denom, max_overlap)
                fy += fb
                tau += rb_x * fb  # lever (rb_x, -ey)
            pen = ey + margin - (bounds_max[1] - p[1])
            if pen > zero:
                ft = -_normal_coeff(pen, -(v[1] - w * rb_x), k, c, damp_denom, max_overlap)
                fy += ft
                tau += -rb_x * ft  # lever (-rb_x, ey)
            f += type(p)(fx, fy)
    return f, tau


@wp.kernel(module="unique", module_options=BOX_MODULE_OPTIONS, grid_stride=False)
def collision_forces_wheel_kernel(
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
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
    torques: wp.array2d(dtype=Any),
):
    """:func:`collision_forces_box_kernel` plus each agent's contact torque about its
    centre, for ``WorldConfig.wheel_contacts``."""
    e, a = wp.tid()
    p = pos[e, a]
    th = theta[e, a]
    v = vel[e, a]
    w = ang_vel[e, a]
    ra = params[a, P_RADIUS]
    sha = params[a, P_SHAPE]
    ha = type(p)(params[a, P_HALF_X], params[a, P_HALF_Y])
    zero = type(k)(0.0)
    f = type(p)(zero, zero)
    tau = zero
    damp_denom = type(k)(1.0) + c * sub_dt / params[a, P_MASS]
    for n_i in range(neighbor_count[e, a]):
        b = neighbor_idx[e, a, n_i]
        fb, tb = _agent_pair_force_torque(
            a, b, sha, params[b, P_SHAPE], p, th, v, w, ra, ha, pos[e, b], theta[e, b],
            vel[e, b], ang_vel[e, b], params[b, P_RADIUS],
            type(p)(params[b, P_HALF_X], params[b, P_HALF_Y]), margin, k, c, damp_denom,
            max_overlap,
        )
        f += fb
        tau += tb
    fs, ts = _static_forces_torque(
        e, sha, p, th, v, w, ra, ha, obs_pos, obs_radius, obs_type, obs_angle, obs_half,
        obs_vel, obs_ang_vel, n_obstacles, k, c, margin, damp_denom, max_overlap, soft_walls,
        bounds_min, bounds_max,
    )
    forces[e, a] = f + fs
    torques[e, a] = tau + ts


@wp.kernel(module="unique", module_options=BOX_MODULE_OPTIONS, grid_stride=False)
def collision_forces_wheel_kernel_per_env(
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
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
    torques: wp.array2d(dtype=Any),
):
    """Per-env-parameter twin of :func:`collision_forces_wheel_kernel`."""
    e, a = wp.tid()
    p = pos[e, a]
    th = theta[e, a]
    v = vel[e, a]
    w = ang_vel[e, a]
    ra = params[e, a, P_RADIUS]
    sha = params[e, a, P_SHAPE]
    ha = type(p)(params[e, a, P_HALF_X], params[e, a, P_HALF_Y])
    zero = type(k)(0.0)
    f = type(p)(zero, zero)
    tau = zero
    damp_denom = type(k)(1.0) + c * sub_dt / params[e, a, P_MASS]
    for n_i in range(neighbor_count[e, a]):
        b = neighbor_idx[e, a, n_i]
        fb, tb = _agent_pair_force_torque(
            a, b, sha, params[e, b, P_SHAPE], p, th, v, w, ra, ha, pos[e, b], theta[e, b],
            vel[e, b], ang_vel[e, b], params[e, b, P_RADIUS],
            type(p)(params[e, b, P_HALF_X], params[e, b, P_HALF_Y]), margin, k, c, damp_denom,
            max_overlap,
        )
        f += fb
        tau += tb
    fs, ts = _static_forces_torque(
        e, sha, p, th, v, w, ra, ha, obs_pos, obs_radius, obs_type, obs_angle, obs_half,
        obs_vel, obs_ang_vel, n_obstacles, k, c, margin, damp_denom, max_overlap, soft_walls,
        bounds_min, bounds_max,
    )
    forces[e, a] = f + fs
    torques[e, a] = tau + ts


def _box_signature(dtype, per_env: bool = False) -> list:
    """:func:`_signature` with ``theta`` and ``ang_vel`` after ``pos`` and ``vel``."""
    sig = _signature(dtype, per_env)
    a2s = wp.array2d(dtype=dtype)
    return [sig[0], a2s, sig[1], a2s, *sig[2:]]


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
    register(collision_forces_kernel, _T, _signature(_T))
    register(collision_forces_kernel_per_env, _T, _signature(_T, per_env=True))
    register(collision_forces_box_kernel, _T, _box_signature(_T))
    register(collision_forces_box_kernel_per_env, _T, _box_signature(_T, per_env=True))
    register(collision_forces_wheel_kernel, _T, [*_box_signature(_T), wp.array2d(dtype=_T)])
    register(
        collision_forces_wheel_kernel_per_env,
        _T,
        [*_box_signature(_T, per_env=True), wp.array2d(dtype=_T)],
    )


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
    theta: wp.array | None = None,
    ang_vel: wp.array | None = None,
    torques: wp.array | None = None,
) -> None:
    """Launch the soft-contact force pass. Everything past the arrays is keyword-only:
    the scalar tail is three floats and a bool in a row, where a swapped argument would be
    invisible at the call site.

    Passing the agents' ``theta`` and ``ang_vel`` selects the box-agent kernel
    (:func:`collision_forces_box_kernel`); the stepper does so exactly when the world
    holds a box agent, and a disc-only world launches the disc kernel as before.

    Passing ``torques`` (with ``theta``/``ang_vel``) selects the wheel-contact kernel
    (:func:`collision_forces_wheel_kernel`), which also writes each agent's contact torque
    about its centre."""
    n_envs, n_agents = pos.shape
    boxes = theta is not None
    per_env = params.floats_per_env is not None
    floats = params.floats_per_env if per_env else params.floats
    outputs = [forces]
    if torques is not None:
        kernel = collision_forces_wheel_kernel_per_env if per_env else collision_forces_wheel_kernel
        state = [pos, theta, vel, ang_vel]
        outputs = [forces, torques]
    elif boxes:
        kernel = collision_forces_box_kernel_per_env if per_env else collision_forces_box_kernel
        state = [pos, theta, vel, ang_vel]
    else:
        kernel = collision_forces_kernel_per_env if per_env else collision_forces_kernel
        state = [pos, vel]
    wp.launch(
        concrete(kernel, dtype),
        dim=(n_envs, n_agents),
        inputs=[
            *state,
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
        outputs=outputs,
        device=pos.device,
    )
