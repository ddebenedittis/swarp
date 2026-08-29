"""Fused Warp kernels for the GiveWayScenario obs/reward/reset layer.

Replaces the eager-torch per-step cache (all-pairs geometry, contact ramps, corner-block
SDF and its gradient, position shaping, observation assembly, reward reduction) with two
kernels that share one all-pairs loop. The torch implementation in
:mod:`swarp.scenarios.giveway` stays the reference (and the differentiable path); these
kernels are the no-grad fast path and are validated against it.

**No neighbour grid.** Unlike the other fused scenarios, the observation here does not
read ``neighbor_idx``/``neighbor_count`` at all — sensing is all-pairs, because the grid's
radius *is* the engine's contact reach and a robot that cannot see an opponent until they
are almost touching cannot yield in time (see the ``giveway`` module docstring). With
``n_agents`` in the single digits an ``O(n^2)`` loop over a runtime ``n_agents`` is both
cheaper than the grid build it replaces and simpler; ``formation_kernels`` sets the
precedent for the same loop shape.

The wall test is the one piece of geometry worth reading twice. The four corner blocks
are mirror images of each other about both axes, so the distance from an agent to the
*nearest* block is a single box SDF evaluated at ``(|x|, |y|)`` against the first-quadrant
block alone — no obstacle-array read, no four-shape loop, and no dependence on the order
the obstacles happen to be installed in. :func:`_corner_sdf` is that evaluation and
:func:`_corner_sdf_grad` its analytic unit gradient; the torch reference recomputes both in
the same association order and with the **identical** ``>=`` tie-breaks, so the discrete
branch each one selects agrees bit-for-bit rather than merely closely.

Observation row (``obs_dim = 13 + 9 * k_obs``), matching the torch layout::

    own (0..12):  dot(p,u), clamp(dot(p,n)/r, -8, 8), dot(v,u), dot(v,n),
                  dot(g-p,u), |g-p|, clamp((sdf(p)-r)/r, -1, 8),
                  dot(grad sdf, u), dot(grad sdf, n), c/r, prio[a], u_x, u_y
    slot j:       rel_along, rel_lat, rvel_along, rvel_lat,
                  ndir_along, ndir_lat, n_goal_dist, dprio, valid

``(u, n)`` is the agent's per-episode travel frame, ``u = axis[e, a]`` (written by the
reset kernel, exactly a signed unit axis) and ``n = (-u_y, u_x)``. Indices 11/12 carry
``u`` itself, in **world** coordinates: everything else is rotation-invariant, and a
world-frame action makes a fully invariant row non-invertible (see the ``giveway`` module
docstring). The neighbour block is **grouped per neighbour**, i.e. nine consecutive floats
per slot — a deliberate departure from navigation's feature-grouped layout, which the
``13 + 9*j + f`` offsets below are the whole contract for.

Slot order is lexicographic in ``(squared distance, index)``: the squared distance is what
both paths compare (``dx*dx + dy*dy``, the same expression in the same order), so no
``sqrt`` can perturb a tie, and an exact tie goes to the lower agent index. The kernel
walks the order without a scratch array by asking, at each slot, for the smallest
``(d, b)`` strictly greater than the one it emitted last — Warp has no dynamically-sized
local array, and ``n_agents`` is a runtime value.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.state import VEC2
from swarp.scenarios.reset_kernels import _as


@wp.func
def _corner_sdf(px: Any, py: Any, cx: Any, cy: Any, hx: Any, hy: Any):
    """Signed distance from ``(px, py)`` to the nearest of the four corner blocks.

    ``(cx, cy)`` / ``(hx, hy)`` describe the **first-quadrant** block only. Folding the
    query point into that quadrant with ``wp.abs`` is exact (a sign flip, no rounding),
    and the four blocks are its mirror images about both axes, so the fold turns a
    four-shape minimum into one axis-aligned box SDF: positive outside, negative inside.

    Mirrors the clamp in :func:`swarp.core.bodies._closest_in_box`, minus the rotation —
    these blocks never rotate, so carrying an angle through would only cost two trig
    calls and a pair of ulps the torch reference would then have to reproduce.
    """
    zero = type(px)(0.0)
    qx = wp.abs(wp.abs(px) - cx) - hx
    qy = wp.abs(wp.abs(py) - cy) - hy
    ox = wp.max(qx, zero)
    oy = wp.max(qy, zero)
    return wp.sqrt(ox * ox + oy * oy) + wp.min(wp.max(qx, qy), zero)


@wp.func
def _corner_sdf_grad(px: Any, py: Any, cx: Any, cy: Any, hx: Any, hy: Any):
    """Analytic unit gradient of :func:`_corner_sdf` at ``(px, py)``.

    Three folds deep: with ``ax = |px|`` and ``qx = |ax - cx| - hx``,
    ``d qx / d px = sign(px) * sign(ax - cx)``, and likewise in ``y``. The outer
    derivative is ``(ox, oy) / |(ox, oy)|`` outside the block and the box-interior
    ``(1, 0)`` / ``(0, 1)`` inside it; the two cases are mutually exclusive, since
    ``min(max(qx, qy), 0)`` is only nonzero when both ``q`` are negative, which is exactly
    when ``ox = oy = 0``.

    Every tie-break is ``>=`` — at ``px == 0``, at ``ax == cx``, at ``qx == qy`` — and the
    torch oracle writes the identical ``>=``. These select *branches*, so a mismatch would
    be a discrete O(1) disagreement between the two paths rather than a rounding one. On a
    medial axis (``px == 0`` is the corridor centreline) the true gradient does not exist
    and the value returned is one of the two one-sided limits, chosen consistently.

    The norm is exactly 1 everywhere. No division by zero: the ``ox/L`` branch is only
    taken when ``L > 0``.
    """
    zero = type(px)(0.0)
    one = type(px)(1.0)
    sx = one
    if px < zero:
        sx = -one
    sy = one
    if py < zero:
        sy = -one
    ax = wp.abs(px)
    ay = wp.abs(py)
    tx = one
    if ax - cx < zero:
        tx = -one
    ty = one
    if ay - cy < zero:
        ty = -one
    qx = wp.abs(ax - cx) - hx
    qy = wp.abs(ay - cy) - hy
    ox = wp.max(qx, zero)
    oy = wp.max(qy, zero)
    sq = ox * ox + oy * oy
    gqx = zero
    gqy = zero
    if sq > zero:
        length = wp.sqrt(sq)
        gqx = ox / length
        gqy = oy / length
    elif qx >= qy:
        gqx = one
    else:
        gqy = one
    return wp.vector(gqx * sx * tx, gqy * sy * ty)


@wp.kernel
def giveway_obs_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    goals: wp.array2d(dtype=Any),
    axis: wp.array2d(dtype=Any),  # [n_envs, n_agents] per-episode unit travel axis
    prio: wp.array2d(dtype=Any),  # [n_envs, n_agents] priority token in [-1, 1]
    radius: wp.array(dtype=Any),  # [n_agents] static per-agent radius
    geom: wp.array(dtype=Any),  # [4] = (box_cx, box_cy, box_hx, box_hy)
    reset_mask: wp.array(dtype=wp.uint8),
    n_agents: wp.int32,
    k_obs: wp.int32,
    agent_radius: Any,
    wall_reach: Any,
    collision_reach: Any,
    contact_margin: Any,
    pos_shaping_factor: Any,
    goal_tolerance: Any,
    advance_prev: wp.int32,
    full_pass: wp.int32,
    obs: wp.array3d(dtype=Any),
    touching: wp.array2d(dtype=Any),
    proximity: wp.array2d(dtype=Any),
    wall_touch: wp.array2d(dtype=Any),
    wall_ramp: wp.array2d(dtype=Any),
    dist_to_goal: wp.array2d(dtype=Any),
    pos_shaping: wp.array2d(dtype=Any),
    on_goal: wp.array(dtype=wp.uint8, ndim=2),
    prev_dist: wp.array2d(dtype=Any),
):
    """Thread per (env, agent): observation row plus every reward input.

    ``geom`` is a **device buffer**, not four scalar arguments, precisely because
    :meth:`~swarp.scenarios.giveway.GiveWayScenario.set_corridor_scale` can move the
    corner blocks between training batches: a scalar is packed into the launch (and baked
    into the captured whole-step graph) by value, where the buffer is baked by pointer and
    its *contents* still track the geometry the engine has installed. Bake the corridor
    width in and the kernel's SDF silently disagrees with the obstacles after the first
    curriculum step. The corridor half-width the observation reports is read out of the
    same buffer as ``geom[0] - geom[2]`` for exactly that reason.

    ``axis``/``prio`` are written only by :func:`giveway_reset_kernel` and never
    reassigned, so their handles are pointer-stable across a capture without a ``watch``.
    ``prio`` is all-zero when the scenario is built with ``use_priority=False``, which is
    what keeps ``obs_dim`` flag-independent: no branch here, the token slots just carry
    zeros.

    ``advance_prev`` / ``full_pass`` / ``reset_mask`` carry the same meaning as in
    :mod:`swarp.scenarios.navigation_kernels` — a just-reset env rebases the shaping
    baseline instead of advancing it, and an obs-only auto-reset pass leaves the
    reward-input buffers alone because the reward that consumed them has already been
    returned for the transition just taken.
    """
    e, a = wp.tid()
    if full_pass == 0 and reset_mask[e] == wp.uint8(0):
        # Obs-only auto-reset pass: an env this mask didn't select has state identical to
        # what the STEP pass moments earlier already wrote into every output buffer below.
        # Reset envs (reset_mask[e] == 1) still fall through and get recomputed.
        return
    p = pos[e, a]
    v = vel[e, a]
    g = goals[e, a]
    grel = g - p
    zero = type(agent_radius)(0.0)
    one = type(agent_radius)(1.0)
    eight = type(agent_radius)(8.0)
    d = wp.sqrt(grel[0] * grel[0] + grel[1] * grel[1])
    ra = radius[a]
    pa = prio[e, a]

    # --- travel frame: u along the goal, n 90 degrees to its left --------------------
    u = axis[e, a]
    nx = -u[1]
    ny = u[0]
    inv_r = one / agent_radius

    # --- corner-block clearance and its analytic normal ------------------------------
    sdf = _corner_sdf(p[0], p[1], geom[0], geom[1], geom[2], geom[3])
    gr = _corner_sdf_grad(p[0], p[1], geom[0], geom[1], geom[2], geom[3])
    # geom[0] - geom[2] is the first-quadrant block's inner face: 0.5*(W+c) - 0.5*(W-c),
    # i.e. exactly the current corridor half-width. Without it in the row the curriculum
    # is a parameter the policy cannot see.
    half_width = geom[0] - geom[2]

    # --- own block, indices 0..10 -----------------------------------------------------
    obs[e, a, 0] = p[0] * u[0] + p[1] * u[1]
    obs[e, a, 1] = wp.clamp((p[0] * nx + p[1] * ny) * inv_r, -eight, eight)
    obs[e, a, 2] = v[0] * u[0] + v[1] * u[1]
    obs[e, a, 3] = v[0] * nx + v[1] * ny
    obs[e, a, 4] = grel[0] * u[0] + grel[1] * u[1]
    obs[e, a, 5] = d
    obs[e, a, 6] = wp.clamp((sdf - agent_radius) * inv_r, -one, eight)
    obs[e, a, 7] = gr[0] * u[0] + gr[1] * u[1]
    obs[e, a, 8] = gr[0] * nx + gr[1] * ny
    obs[e, a, 9] = half_width * inv_r
    obs[e, a, 10] = pa
    obs[e, a, 11] = u[0]
    obs[e, a, 12] = u[1]

    # --- all-pairs contact diagnostics -------------------------------------------------
    # One loop for both the discrete count (info) and the continuous ramp (reward). The
    # count is the *interpretable* metric and is structurally near-zero at this contact
    # stiffness; the ramp over the activation band is what the reward actually charges.
    touch = zero
    prox = zero
    for b in range(n_agents):
        if b != a:
            dx = pos[e, b][0] - p[0]
            dy = pos[e, b][1] - p[1]
            nd = wp.sqrt(dx * dx + dy * dy)
            if nd < ra + radius[b]:
                touch += one
            prox += wp.clamp((collision_reach - nd) / contact_margin, zero, one)

    # --- neighbour slots: lexicographic (squared distance, index) ----------------------
    # No scratch array (Warp has no dynamically-sized local one), so each slot asks for the
    # smallest (d, b) strictly greater than the one the previous slot emitted. Comparing
    # *squared* distances is what makes the tie-break bit-identical to the torch oracle's
    # stable sort: the same dx*dx + dy*dy in the same order, with no sqrt in between.
    prev_d = -one
    prev_b = wp.int32(-1)
    for j in range(k_obs):
        best_b = wp.int32(-1)
        best_d = zero
        for b in range(n_agents):
            if b != a:
                dx = pos[e, b][0] - p[0]
                dy = pos[e, b][1] - p[1]
                db = dx * dx + dy * dy
                # Strictly after the last emitted (d, b), and the smallest such — the two
                # halves of "next in lexicographic order", written out because Warp has no
                # dynamically-sized local array to mark consumed slots in.
                if (db > prev_d or (db == prev_d and b > prev_b)) and (
                    best_b < 0 or db < best_d or (db == best_d and b < best_b)
                ):
                    best_b = b
                    best_d = db
        base = wp.int32(13) + wp.int32(9) * j
        if best_b < 0:
            # Fewer than k_obs other agents: an all-zero, invalid slot. Unreachable at the
            # shipped k_obs = min(neighbor_obs, n_agents - 1), kept so the row stays
            # well-defined if a caller ever asks for more slots than there are robots.
            for f in range(9):
                obs[e, a, base + f] = zero
        else:
            rx = pos[e, best_b][0] - p[0]
            ry = pos[e, best_b][1] - p[1]
            rvx = vel[e, best_b][0] - v[0]
            rvy = vel[e, best_b][1] - v[1]
            nb = axis[e, best_b]
            gb = goals[e, best_b] - pos[e, best_b]
            obs[e, a, base + 0] = rx * u[0] + ry * u[1]
            obs[e, a, base + 1] = rx * nx + ry * ny
            obs[e, a, base + 2] = rvx * u[0] + rvy * u[1]
            obs[e, a, base + 3] = rvx * nx + rvy * ny
            obs[e, a, base + 4] = nb[0] * u[0] + nb[1] * u[1]
            obs[e, a, base + 5] = nb[0] * nx + nb[1] * ny
            obs[e, a, base + 6] = wp.sqrt(gb[0] * gb[0] + gb[1] * gb[1])
            obs[e, a, base + 7] = prio[e, best_b] - pa
            obs[e, a, base + 8] = one
            prev_d = best_d
            prev_b = best_b

    # --- position shaping baseline (navigation's rebase/advance/gate pattern) ----------
    prev = prev_dist[e, a]
    ps = (prev - d) * pos_shaping_factor
    reset_hit = wp.int32(reset_mask[e])
    if reset_hit == 1:
        prev_dist[e, a] = d
        if full_pass == 1:
            pos_shaping[e, a] = zero
    else:
        if advance_prev == 1:
            prev_dist[e, a] = d
        if full_pass == 1:
            pos_shaping[e, a] = ps
    if full_pass == 1:
        touching[e, a] = touch
        proximity[e, a] = prox
        wcontact = zero
        if sdf < wall_reach:
            wcontact = one
        wall_touch[e, a] = wcontact
        wall_ramp[e, a] = wp.clamp((wall_reach - sdf) / contact_margin, zero, one)
        dist_to_goal[e, a] = d
        if d < goal_tolerance:
            on_goal[e, a] = wp.uint8(1)
        else:
            on_goal[e, a] = wp.uint8(0)


@wp.kernel
def giveway_reward_kernel(
    pos_shaping: wp.array2d(dtype=Any),
    proximity: wp.array2d(dtype=Any),
    wall_ramp: wp.array2d(dtype=Any),
    on_goal: wp.array(dtype=wp.uint8, ndim=2),
    n_agents: wp.int32,
    inv_n_agents: Any,
    shaping_share: Any,
    collision_penalty: Any,
    wall_penalty: Any,
    time_penalty: Any,
    goal_hold_bonus: Any,
    final_reward: Any,
    reward: wp.array2d(dtype=Any),
    done: wp.array(dtype=wp.uint8),
    frac_on_goal: wp.array(dtype=Any),
):
    """Thread per env; sequential agent loops (deterministic, no atomics).

    ``shaping_share`` blends each agent's own shaping with the team **mean** — not the
    sum. A sum makes an agent's reward scale with ``n_agents`` and, at four robots, turns
    three quarters of it into teammate behaviour the agent did not cause; and the argument
    that sharing is what makes yielding pay does not survive the observation that shaping
    telescopes, so a detour into a passing bay is refunded on the way out regardless. The
    contact ramps, the clock and the goal-hold bonus are always the agent's own, so a
    robot is still told which of *its* actions cost it something.
    """
    e = wp.tid()
    zero = type(collision_penalty)(0.0)
    one = type(collision_penalty)(1.0)
    shaping_sum = zero
    n_on = zero
    all_og = wp.uint8(1)
    for a in range(n_agents):
        shaping_sum += pos_shaping[e, a]
        if on_goal[e, a] == wp.uint8(0):
            all_og = wp.uint8(0)
        else:
            n_on += one
    final = zero
    if all_og == wp.uint8(1):
        final = final_reward
    team = shaping_share * (shaping_sum * inv_n_agents)
    for a in range(n_agents):
        hold = zero
        if on_goal[e, a] == wp.uint8(1):
            hold = goal_hold_bonus
        r = (
            collision_penalty * proximity[e, a]
            + wall_penalty * wall_ramp[e, a]
            + time_penalty
            + hold
        )
        r = r + (one - shaping_share) * pos_shaping[e, a] + team
        reward[e, a] = r + final
    done[e] = all_og
    frac_on_goal[e] = n_on * inv_n_agents


@wp.kernel
def giveway_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    seed_prio: wp.int32,
    n_agents: wp.int32,
    use_priority: wp.int32,
    desync_random: wp.int32,
    s_max: Any,
    stagger: Any,
    desync_amp: Any,
    jitter_long: Any,
    jitter_lat: Any,
    pos: Any,
    theta: Any,
    vel: Any,
    speed: Any,
    ang_vel: Any,
    goals: Any,
    axis: Any,
    prio: Any,
):
    """Masked episode reset: one agent per arm end, goal at the opposite arm's end.

    Thread per **env**, because the one genuinely per-env draw is the arm rotation ``k``:
    every agent's arm is ``(a + k) % 4``, so it has to be drawn once and shared, exactly
    the reason :mod:`swarp.scenarios.reset_kernels` gives for the per-env threading.

    Agent ``a`` takes arm ``(a + k) % 4`` at depth rank ``a // 4``, so agents beyond the
    fourth stack up *inward* along their arm rather than on top of each other, and its
    goal is the same depth in the opposite arm (whose unit vector is exactly ``-d``, the
    arms being axis-aligned). The jitters are drawn inline — never through a ``@wp.func``,
    which would hand every call the same point (see ``reset_kernels``' warning).

    **RNG contract — two independent streams, and the first one is frozen.** The ``seed``
    stream draws exactly what it always has, in exactly this order: ``k`` once per env,
    then ``sj`` and ``lj`` per agent. Nothing added here may be drawn from it, because
    seeded spawn tests compare positions at ``desync_amp == 0`` against the layout this
    kernel had before the priority token existed. Everything new comes from ``seed_prio``,
    a *second* host seed drawn unconditionally by ``reset_world`` — unconditionally so the
    host-side seed layout does not depend on ``use_priority``.

    The token is a uniform random **permutation**, rank-mapped to ``[-1, 1]``, not
    ``n_agents`` i.i.d. uniforms: an i.i.d. near-tie occurs with probability ~2*eps and a
    near-tie *is* the deadlock case this token exists to break, so i.i.d. draws would put a
    hard ceiling on the solve rate. Warp has no dynamically-sized local array to sort in,
    so the rank is an O(n^2) count of how many agents drew below this one, with ties broken
    by index — which is a permutation by construction. Each agent's draw is re-derived from
    its own sub-stream (``rand_init(seed_prio, e*n_agents + b)``) rather than stored, which
    is what makes the count reproducible inside the loop.

    ``desync_amp`` staggers spawn depth so the robots reach the junction at different
    times. Under ``desync_random == 0`` the offset is the token's own rank, so "higher
    token goes first" is directly observable while the amplitude is large and has to be
    carried by the token alone once it anneals to zero; under ``desync_random == 1`` it is
    an independent draw (the second value of the agent's own sub-stream), which is the
    control that separates grounding from mere desynchronization.

    **The goal does not move with the spawn**, which is the one non-obvious thing here and
    was measured rather than reasoned. Mirroring the desync into the goal keeps each
    robot's travel distance constant — which sounds right — but it also drags the shallow
    robot's goal to within a few centimetres of the junction, so that robot arrives almost
    immediately and then sits in the junction mouth for the rest of the episode with
    nowhere to go. A scripted controller that solves 100% of episodes at ``desync_amp = 0``
    solves **0%** at any ``desync_amp > 0`` under the mirrored form, with ``frac_on_goal``
    pinned at exactly 1/4. The goal therefore stays at the un-desynced arm end; only the
    rank stagger for agents past the fourth (``a // 4``) is common to both.
    """
    e = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return
    rng = wp.rand_init(seed, e)
    zero = _as(0.0, s_max)
    one = _as(1.0, s_max)
    two = _as(2.0, s_max)

    # Whole-quarter-turn rotation of the arm assignment, so which arms are occupied
    # varies between episodes when n_agents is not a multiple of four.
    k = wp.int32(wp.randf(rng) * 4.0)
    if k > 3:
        k = 3

    inv_ranks = zero
    if n_agents > 1:
        inv_ranks = one / _as(wp.float32(n_agents - 1), s_max)

    for a in range(n_agents):
        arm = (a + k) % 4
        # --- priority rank, from the second stream only ------------------------------
        pa_state = wp.rand_init(seed_prio, e * n_agents + a)
        ua = wp.randf(pa_state)
        rank = wp.int32(0)
        for b in range(n_agents):
            pb_state = wp.rand_init(seed_prio, e * n_agents + b)
            ub = wp.randf(pb_state)
            if ub < ua or (ub == ua and b < a):
                rank += 1
        rank_frac = _as(wp.float32(rank), s_max) * inv_ranks
        # A second draw off the same sub-stream: independent of every agent's *first*
        # draw, so the "random" control is genuinely uncorrelated with the token.
        desync_frac = rank_frac
        if desync_random == 1:
            desync_frac = _as(wp.randf(pa_state), s_max)

        # The goal sits at the *un-desynced* arm end; only the spawn moves inward. See
        # the docstring: mirroring the desync into the goal parks a robot in the junction
        # mouth for the whole episode, which measures as a zero solve rate.
        goal_depth = s_max - _as(wp.float32(a / 4), s_max) * stagger
        depth = goal_depth - desync_frac * desync_amp
        dx = zero
        dy = zero
        if arm == 0:
            dx = one
        elif arm == 1:
            dy = one
        elif arm == 2:
            dx = -one
        else:
            dy = -one
        # Along-arm and lateral jitter. Both amplitudes are clamped host-side so a
        # jittered spawn can neither leave its arm, overlap the next rank, nor start
        # inside the corridor wall's contact band.
        sj = (_as(wp.randf(rng), s_max) * two - one) * jitter_long
        lj = (_as(wp.randf(rng), s_max) * two - one) * jitter_lat
        s = depth + sj
        # perpendicular is (-dy, dx)
        pos[e, a] = wp.vector(dx * s - dy * lj, dy * s + dx * lj)
        goals[e, a] = wp.vector(-dx * goal_depth, -dy * goal_depth)
        # The travel axis points at the goal: exactly -(arm direction), so it is a signed
        # unit axis with no sqrt and no rounding.
        axis[e, a] = wp.vector(-dx, -dy)
        pr = zero
        if use_priority == 1 and n_agents > 1:
            pr = two * rank_frac - one
        prio[e, a] = pr
        vel[e, a] = wp.vector(zero, zero)
        theta[e, a] = zero
        speed[e, a] = zero
        ang_vel[e, a] = zero


def _obs_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    a2s = wp.array2d(dtype=dtype)
    a3s = wp.array3d(dtype=dtype)
    u8_2 = wp.array(dtype=wp.uint8, ndim=2)
    return [
        a2v,  # pos
        a2v,  # vel
        a2v,  # goals
        a2v,  # axis
        a2s,  # prio
        wp.array(dtype=dtype),  # radius
        wp.array(dtype=dtype),  # geom
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # n_agents
        wp.int32,  # k_obs
        dtype,  # agent_radius
        dtype,  # wall_reach
        dtype,  # collision_reach
        dtype,  # contact_margin
        dtype,  # pos_shaping_factor
        dtype,  # goal_tolerance
        wp.int32,  # advance_prev
        wp.int32,  # full_pass
        a3s,  # obs
        a2s,  # touching
        a2s,  # proximity
        a2s,  # wall_touch
        a2s,  # wall_ramp
        a2s,  # dist_to_goal
        a2s,  # pos_shaping
        u8_2,  # on_goal
        a2s,  # prev_dist
    ]


def _reward_signature(dtype) -> list:
    a2s = wp.array2d(dtype=dtype)
    return [
        a2s,  # pos_shaping
        a2s,  # proximity
        a2s,  # wall_ramp
        wp.array(dtype=wp.uint8, ndim=2),  # on_goal
        wp.int32,  # n_agents
        dtype,  # inv_n_agents
        dtype,  # shaping_share
        dtype,  # collision_penalty
        dtype,  # wall_penalty
        dtype,  # time_penalty
        dtype,  # goal_hold_bonus
        dtype,  # final_reward
        a2s,  # reward
        wp.array(dtype=wp.uint8),  # done
        wp.array(dtype=dtype),  # frac_on_goal
    ]


def _reset_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    a2s = wp.array2d(dtype=dtype)
    return [
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # use_mask
        wp.int32,  # seed
        wp.int32,  # seed_prio
        wp.int32,  # n_agents
        wp.int32,  # use_priority
        wp.int32,  # desync_random
        dtype,  # s_max
        dtype,  # stagger
        dtype,  # desync_amp
        dtype,  # jitter_long
        dtype,  # jitter_lat
        a2v,  # pos
        a2s,  # theta
        a2v,  # vel
        a2s,  # speed
        a2s,  # ang_vel
        a2v,  # goals
        a2v,  # axis
        a2s,  # prio
    ]


for _T in (wp.float32, wp.float64):
    register(giveway_obs_kernel, _T, _obs_signature(_T))
    register(giveway_reward_kernel, _T, _reward_signature(_T))
    register(giveway_reset_kernel, _T, _reset_signature(_T))
