"""Randomised surroundings for the lidar Push-T: walls and clutter outside the arena.

The lidar sees them and nothing else does: they live in a
:class:`~swarp.sensors.lidar.Scenery`, which the stepper's contacts never read. That is
exact only because everything is placed outside the *free region*, the arena grown by a
clearance no robot (the arena stop plus its stopping distance) and no T (the out-of-bounds
limit plus its radius) can cross before its episode has failed. Inside the arena the
surroundings occlude nothing; what they change is what a robot near the edge sees past it,
which under the deployment's labels (``lidar_label_from_pose``) is a miss beyond the label
bound and the T inside it.

``scenery_reset_kernel`` (thread per env, masked) draws one layout per env on reset:

* ``SCENERY_NONE``: nothing;
* ``SCENERY_PERIMETER``: four walls parallel to the arena's sides, each its own gap past the
  free region;
* ``SCENERY_ROOM``: a room at a random yaw that contains the free region (its half-extents
  in its own frame are the free region's support extents there, plus a gap per side), so
  its walls are misaligned with the arena but never cross it;
* ``SCENERY_CLUTTER``: the walls of one of the two above, plus circles and boxes placed
  within lidar range of the free region and clear of it.

Each wall is a capsule (segment) whose inner surface sits at its gap, and each may be
missing (a doorway or an open side). Slots ``0..3`` hold the walls, the rest the clutter;
an unused slot has shape ``-1``.

``scenery_slots_reset_kernel`` draws the same layouts for surroundings the robots and the
T collide with, which the stepper holds as immovable obstacles. Their sizes are shared
across envs, so each slot has a fixed shape and size (:func:`slot_bank`) and a reset only
draws which slots an env uses and where; the clutter is circles and capsules (the T
collides with both, not with boxes) and the free region can be as small as the arena.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.config import ObstacleShape
from swarp.core.state import VEC2
from swarp.scenarios.reset_kernels import _as

SCENERY_NONE = 0
SCENERY_PERIMETER = 1
SCENERY_ROOM = 2
SCENERY_CLUTTER = 3
#: The layouts, in the order ``scenery_probs`` weights them.
SCENERY_MODES = ("none", "perimeter", "room", "clutter")
N_SCENERY_MODES = 4
#: Wall slots per env (slots ``0..3``); the clutter slots follow.
N_WALLS = 4

_CIRCLE = wp.constant(int(ObstacleShape.CIRCLE))
_BOX = wp.constant(int(ObstacleShape.BOX))
_SEGMENT = wp.constant(int(ObstacleShape.SEGMENT))
_PLACE_TRIES = wp.constant(8)


def slot_bank(
    n_clutter: int,
    wall_radius: tuple[float, float],
    wall_len: float,
    clutter_radius: tuple[float, float],
    clutter_half: tuple[float, float],
) -> tuple[list[int], list[float], list[tuple[float, float]]]:
    """``(shape, radius, half-extents)`` of every slot of :func:`scenery_slots_reset_kernel`.

    The four walls are capsules of half-length ``wall_len``, their radii spread evenly over
    ``wall_radius``. The clutter alternates circles and capsules: the circles' radii spread
    evenly over ``clutter_radius``; the capsules' half-lengths over ``clutter_half`` and
    their radii over ``clutter_radius`` in the opposite order, so the longest is the
    thinnest."""
    seg, circ = int(ObstacleShape.SEGMENT), int(ObstacleShape.CIRCLE)

    def spread(lo: float, hi: float, k: int, n: int) -> float:
        return lo + (hi - lo) * (k + 0.5) / max(n, 1)

    shape = [seg] * N_WALLS
    radius = [spread(*wall_radius, w, N_WALLS) for w in range(N_WALLS)]
    half = [(wall_len, 0.0)] * N_WALLS
    n_circ = (n_clutter + 1) // 2
    n_caps = n_clutter - n_circ
    for i in range(n_clutter):
        k = i // 2
        if i % 2 == 0:
            shape.append(circ)
            radius.append(spread(*clutter_radius, k, n_circ))
            half.append((0.0, 0.0))
        else:
            shape.append(seg)
            radius.append(spread(*clutter_radius, n_caps - 1 - k, n_caps))
            half.append((spread(*clutter_half, k, n_caps), 0.0))
    return shape, radius, half


@wp.kernel
def scenery_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    arena_half: wp.array(dtype=Any),
    clearance: Any,
    scenery_on: Any,
    p_perimeter: Any,
    p_room: Any,
    p_clutter: Any,
    gap_lo: Any,
    gap_hi: Any,
    wall_missing: Any,
    wall_r_lo: Any,
    wall_r_hi: Any,
    n_clutter_lo: wp.int32,
    n_clutter_hi: wp.int32,
    circle_lo: Any,
    circle_hi: Any,
    box_lo: Any,
    box_hi: Any,
    reach: Any,
    mode_out: wp.array(dtype=wp.int32),
    sc_pos: wp.array2d(dtype=Any),
    sc_angle: wp.array2d(dtype=Any),
    sc_half: wp.array2d(dtype=Any),
    sc_radius: wp.array2d(dtype=Any),
    sc_type: wp.array2d(dtype=wp.int32),
):
    """Thread per env: one layout (see the module docstring) for each env the mask
    selects. With probability ``1 - scenery_on`` the env has no surroundings; otherwise the
    layout is drawn with probabilities ``(1 - p_perimeter - p_room - p_clutter,
    p_perimeter, p_room, p_clutter)``. The free region is ``arena_half[e] + clearance``;
    gaps are uniform in ``[gap_lo, gap_hi]`` per side, wall radii in ``[wall_r_lo,
    wall_r_hi]``, the clutter count in ``[n_clutter_lo, n_clutter_hi]``, a circle's radius
    in ``[circle_lo, circle_hi]`` and a box's half-extents in ``[box_lo, box_hi]``. Clutter
    is drawn uniformly within ``reach`` of the free region and redrawn
    (``_PLACE_TRIES``) while its bounding disc comes closer to it than its radius."""
    e = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return
    rng = wp.rand_init(seed, e)
    zero = _as(0.0, clearance)
    one = _as(1.0, clearance)
    two = _as(2.0, clearance)
    half = _as(0.5, clearance)
    pi = _as(3.14159265358979, clearance)
    n_slots = sc_type.shape[1]
    for s in range(n_slots):
        sc_type[e, s] = wp.int32(-1)

    mode = wp.int32(SCENERY_NONE)
    if _as(wp.randf(rng), clearance) < scenery_on:
        u = _as(wp.randf(rng), clearance)
        p_none = one - p_perimeter - p_room - p_clutter
        if u >= p_none:
            mode = wp.int32(SCENERY_PERIMETER)
            if u >= p_none + p_perimeter:
                mode = wp.int32(SCENERY_ROOM)
                if u >= p_none + p_perimeter + p_room:
                    mode = wp.int32(SCENERY_CLUTTER)
    mode_out[e] = mode
    if mode == wp.int32(SCENERY_NONE):
        return

    ah = arena_half[e]
    fx = ah[0] + clearance
    fy = ah[1] + clearance

    # ---- walls: a perimeter (yaw 0) or a room at a random yaw, around the free region
    coin = _as(wp.randf(rng), clearance)
    turn = (_as(wp.randf(rng), clearance) * two - one) * pi
    yaw = zero
    if mode == wp.int32(SCENERY_ROOM) or (mode == wp.int32(SCENERY_CLUTTER) and coin < half):
        yaw = turn
    cy = wp.cos(yaw)
    sy = wp.sin(yaw)
    # The free region's support extents along the room's own axes.
    ex = fx * wp.abs(cy) + fy * wp.abs(sy)
    ey = fx * wp.abs(sy) + fy * wp.abs(cy)
    gr = gap_lo + _as(wp.randf(rng), clearance) * (gap_hi - gap_lo)
    gl = gap_lo + _as(wp.randf(rng), clearance) * (gap_hi - gap_lo)
    gt = gap_lo + _as(wp.randf(rng), clearance) * (gap_hi - gap_lo)
    gb = gap_lo + _as(wp.randf(rng), clearance) * (gap_hi - gap_lo)
    right = ex + gr
    left = -(ex + gl)
    top = ey + gt
    bottom = -(ey + gb)
    for w in range(N_WALLS):
        wr = wall_r_lo + _as(wp.randf(rng), clearance) * (wall_r_hi - wall_r_lo)
        keep = _as(wp.randf(rng), clearance) >= wall_missing
        # Centre and direction in the room frame; the core sits wr outside the surface.
        lx = zero
        ly = zero
        ang = zero
        hl = zero
        if w == 0:
            lx = right + wr
            ly = half * (top + bottom)
            ang = half * pi
            hl = half * (top - bottom) + wr
        elif w == 1:
            lx = left - wr
            ly = half * (top + bottom)
            ang = half * pi
            hl = half * (top - bottom) + wr
        elif w == 2:
            lx = half * (right + left)
            ly = top + wr
            hl = half * (right - left) + wr
        else:
            lx = half * (right + left)
            ly = bottom - wr
            hl = half * (right - left) + wr
        if keep:
            sc_pos[e, w] = wp.vector(cy * lx - sy * ly, sy * lx + cy * ly)
            sc_angle[e, w] = yaw + ang
            sc_half[e, w] = wp.vector(hl, zero)
            sc_radius[e, w] = wr
            sc_type[e, w] = _SEGMENT

    if mode != wp.int32(SCENERY_CLUTTER):
        return
    # ---- clutter, within reach of the free region and clear of it
    span = n_clutter_hi - n_clutter_lo + wp.int32(1)
    n = n_clutter_lo + wp.min(wp.int32(wp.randf(rng) * float(span)), span - wp.int32(1))
    for i in range(n_slots - N_WALLS):
        if i < n:
            s = N_WALLS + i
            is_box = wp.randf(rng) < 0.5
            r = circle_lo + _as(wp.randf(rng), clearance) * (circle_hi - circle_lo)
            hx = box_lo + _as(wp.randf(rng), clearance) * (box_hi - box_lo)
            hy = box_lo + _as(wp.randf(rng), clearance) * (box_hi - box_lo)
            rot = (_as(wp.randf(rng), clearance) * two - one) * pi
            rb = r
            if is_box:
                rb = wp.sqrt(hx * hx + hy * hy)
            placed = bool(False)  # noqa: UP018 - Warp: a loop-mutated flag must not be a constant
            for _k in range(_PLACE_TRIES):
                px = (_as(wp.randf(rng), clearance) * two - one) * (fx + reach)
                py = (_as(wp.randf(rng), clearance) * two - one) * (fy + reach)
                if not placed:
                    dx = wp.max(wp.abs(px) - fx, zero)
                    dy = wp.max(wp.abs(py) - fy, zero)
                    if dx * dx + dy * dy >= rb * rb:
                        placed = True
                        sc_pos[e, s] = wp.vector(px, py)
            if placed:
                sc_angle[e, s] = rot
                if is_box:
                    sc_half[e, s] = wp.vector(hx, hy)
                    sc_radius[e, s] = zero
                    sc_type[e, s] = _BOX
                else:
                    sc_half[e, s] = wp.vector(zero, zero)
                    sc_radius[e, s] = r
                    sc_type[e, s] = _CIRCLE


@wp.kernel
def scenery_slots_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    arena_half: wp.array(dtype=Any),
    clearance: Any,
    scenery_on: Any,
    p_perimeter: Any,
    p_room: Any,
    p_clutter: Any,
    gap_lo: Any,
    gap_hi: Any,
    wall_missing: Any,
    n_clutter_lo: wp.int32,
    n_clutter_hi: wp.int32,
    reach: Any,
    park: Any,
    slot_type: wp.array(dtype=wp.int32),
    slot_radius: wp.array(dtype=Any),
    slot_half: wp.array(dtype=Any),
    eng_offset: wp.int32,
    mode_out: wp.array(dtype=wp.int32),
    sc_pos: wp.array2d(dtype=Any),
    sc_angle: wp.array2d(dtype=Any),
    sc_half: wp.array2d(dtype=Any),
    sc_radius: wp.array2d(dtype=Any),
    sc_type: wp.array2d(dtype=wp.int32),
    eng_pos: wp.array2d(dtype=Any),
    eng_angle: wp.array2d(dtype=Any),
):
    """Thread per env: the layouts of :func:`scenery_reset_kernel` over a fixed bank of
    slots, for surroundings the stepper collides with (its obstacles share their sizes
    across envs). Slot ``s`` always has the shape ``slot_type[s]``, radius
    ``slot_radius[s]`` and half-extents ``slot_half[s]``; a reset only draws which slots
    an env uses and where. The walls (slots ``0..3``, capsules) are long enough to meet at
    every corner of the largest room, so they run on past it; the clutter is ``n`` slots
    of the bank (``n`` uniform in ``[n_clutter_lo, n_clutter_hi]``), chosen uniformly
    without replacement, each placed as in :func:`scenery_reset_kernel`.

    Besides the lidar's per-env arrays, every slot's pose goes to the stepper's obstacle
    arrays ``eng_pos``/``eng_angle`` at ``eng_offset + s``; an unused slot is parked at
    ``(park + s, park)``, out of everything's reach."""
    e = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return
    rng = wp.rand_init(seed, e)
    zero = _as(0.0, clearance)
    one = _as(1.0, clearance)
    two = _as(2.0, clearance)
    half = _as(0.5, clearance)
    pi = _as(3.14159265358979, clearance)
    n_slots = sc_type.shape[1]
    for s in range(n_slots):
        parked = wp.vector(park + _as(float(s), clearance), park)
        sc_type[e, s] = wp.int32(-1)
        sc_pos[e, s] = parked
        sc_angle[e, s] = zero
        sc_half[e, s] = slot_half[s]
        sc_radius[e, s] = slot_radius[s]
        eng_pos[e, eng_offset + s] = parked
        eng_angle[e, eng_offset + s] = zero

    mode = wp.int32(SCENERY_NONE)
    if _as(wp.randf(rng), clearance) < scenery_on:
        u = _as(wp.randf(rng), clearance)
        p_none = one - p_perimeter - p_room - p_clutter
        if u >= p_none:
            mode = wp.int32(SCENERY_PERIMETER)
            if u >= p_none + p_perimeter:
                mode = wp.int32(SCENERY_ROOM)
                if u >= p_none + p_perimeter + p_room:
                    mode = wp.int32(SCENERY_CLUTTER)
    mode_out[e] = mode
    if mode == wp.int32(SCENERY_NONE):
        return

    ah = arena_half[e]
    fx = ah[0] + clearance
    fy = ah[1] + clearance

    # ---- walls: a perimeter (yaw 0) or a room at a random yaw, around the free region
    coin = _as(wp.randf(rng), clearance)
    turn = (_as(wp.randf(rng), clearance) * two - one) * pi
    yaw = zero
    if mode == wp.int32(SCENERY_ROOM) or (mode == wp.int32(SCENERY_CLUTTER) and coin < half):
        yaw = turn
    cy = wp.cos(yaw)
    sy = wp.sin(yaw)
    ex = fx * wp.abs(cy) + fy * wp.abs(sy)
    ey = fx * wp.abs(sy) + fy * wp.abs(cy)
    right = ex + gap_lo + _as(wp.randf(rng), clearance) * (gap_hi - gap_lo)
    left = -(ex + gap_lo + _as(wp.randf(rng), clearance) * (gap_hi - gap_lo))
    top = ey + gap_lo + _as(wp.randf(rng), clearance) * (gap_hi - gap_lo)
    bottom = -(ey + gap_lo + _as(wp.randf(rng), clearance) * (gap_hi - gap_lo))
    for w in range(N_WALLS):
        wr = slot_radius[w]
        keep = _as(wp.randf(rng), clearance) >= wall_missing
        lx = zero
        ly = zero
        ang = zero
        if w == 0:
            lx = right + wr
            ly = half * (top + bottom)
            ang = half * pi
        elif w == 1:
            lx = left - wr
            ly = half * (top + bottom)
            ang = half * pi
        elif w == 2:
            lx = half * (right + left)
            ly = top + wr
        else:
            lx = half * (right + left)
            ly = bottom - wr
        if keep:
            p = wp.vector(cy * lx - sy * ly, sy * lx + cy * ly)
            sc_pos[e, w] = p
            sc_angle[e, w] = yaw + ang
            sc_type[e, w] = slot_type[w]
            eng_pos[e, eng_offset + w] = p
            eng_angle[e, eng_offset + w] = yaw + ang

    if mode != wp.int32(SCENERY_CLUTTER):
        return
    # ---- clutter: n slots of the bank, within reach of the free region and clear of it
    k = n_slots - N_WALLS
    span = n_clutter_hi - n_clutter_lo + wp.int32(1)
    need = n_clutter_lo + wp.min(wp.int32(wp.randf(rng) * float(span)), span - wp.int32(1))
    for i in range(k):
        # Selection sampling: slot i is taken with probability need / (slots left).
        if need > wp.int32(0) and wp.randf(rng) * float(k - i) < float(need):
            need -= wp.int32(1)
            s = N_WALLS + i
            st = slot_type[s]
            h = slot_half[s]
            rb = slot_radius[s]
            if st == _BOX:
                rb = wp.sqrt(h[0] * h[0] + h[1] * h[1])
            elif st == _SEGMENT:
                rb = h[0] + slot_radius[s]
            rot = (_as(wp.randf(rng), clearance) * two - one) * pi
            placed = bool(False)  # noqa: UP018 - Warp: a loop-mutated flag must not be a constant
            for _k in range(_PLACE_TRIES):
                px = (_as(wp.randf(rng), clearance) * two - one) * (fx + reach)
                py = (_as(wp.randf(rng), clearance) * two - one) * (fy + reach)
                if not placed:
                    dx = wp.max(wp.abs(px) - fx, zero)
                    dy = wp.max(wp.abs(py) - fy, zero)
                    if dx * dx + dy * dy >= rb * rb:
                        placed = True
                        sc_pos[e, s] = wp.vector(px, py)
                        eng_pos[e, eng_offset + s] = wp.vector(px, py)
            if placed:
                sc_angle[e, s] = rot
                sc_type[e, s] = st
                eng_angle[e, eng_offset + s] = rot


def _scenery_reset_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2s = wp.array2d(dtype=dtype)
    return [
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # use_mask
        wp.int32,  # seed
        wp.array(dtype=vec2),  # arena_half
        *([dtype] * 9),  # clearance, on, three probabilities, gap range, missing, wall_r_lo
        dtype,  # wall_r_hi
        wp.int32,  # n_clutter_lo
        wp.int32,  # n_clutter_hi
        *([dtype] * 5),  # circle range, box range, reach
        wp.array(dtype=wp.int32),  # mode_out
        wp.array2d(dtype=vec2),  # sc_pos
        a2s,  # sc_angle
        wp.array2d(dtype=vec2),  # sc_half
        a2s,  # sc_radius
        wp.array2d(dtype=wp.int32),  # sc_type
    ]


def _scenery_slots_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2s = wp.array2d(dtype=dtype)
    return [
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # use_mask
        wp.int32,  # seed
        wp.array(dtype=vec2),  # arena_half
        *([dtype] * 8),  # clearance, on, three probabilities, gap range, missing
        wp.int32,  # n_clutter_lo
        wp.int32,  # n_clutter_hi
        dtype,  # reach
        dtype,  # park
        wp.array(dtype=wp.int32),  # slot_type
        wp.array(dtype=dtype),  # slot_radius
        wp.array(dtype=vec2),  # slot_half
        wp.int32,  # eng_offset
        wp.array(dtype=wp.int32),  # mode_out
        wp.array2d(dtype=vec2),  # sc_pos
        a2s,  # sc_angle
        wp.array2d(dtype=vec2),  # sc_half
        a2s,  # sc_radius
        wp.array2d(dtype=wp.int32),  # sc_type
        wp.array2d(dtype=vec2),  # eng_pos
        a2s,  # eng_angle
    ]


for _T in (wp.float32, wp.float64):
    register(scenery_reset_kernel, _T, _scenery_reset_signature(_T))
    register(scenery_slots_reset_kernel, _T, _scenery_slots_signature(_T))
