"""Fused Warp kernels for the GiveWayScenario obs/reward/reset layer.

Replaces the eager-torch per-step cache (neighbor gather, touching count, corner-block
contact test, position shaping, observation assembly, reward reduction) with two kernels
that share one neighbor loop. The torch implementation in
:mod:`swarp.scenarios.giveway` stays the reference (and the differentiable path); these
kernels are the no-grad fast path and are validated against it.

The wall test is the one piece of geometry worth reading twice. The four corner blocks
are mirror images of each other about both axes, so the distance from an agent to the
*nearest* block is a single box SDF evaluated at ``(|x|, |y|)`` against the first-quadrant
block alone — no obstacle-array read, no four-shape loop, and no dependence on the order
the obstacles happen to be installed in. :func:`_corner_sdf` is that evaluation, and the
torch reference recomputes the same expression in the same association order so the
resulting *discrete* contact flag agrees bit-for-bit rather than merely closely.

Observation row (``obs_dim = 6 + 5 * k_obs``), matching the torch ``cat`` order::

    [ pos(2), vel(2), goal-pos(2),
      rel_pos(2·k_obs), rel_vel(2·k_obs), valid(k_obs) ]

Grouped by feature rather than interleaved per neighbour, exactly as navigation lays it
out; the ``rp_base``/``rv_base``/``vd_base`` offsets below are the whole contract between
the two paths.
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


@wp.kernel
def giveway_obs_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    goals: wp.array2d(dtype=Any),
    neighbor_idx: wp.array3d(dtype=wp.int32),
    neighbor_count: wp.array2d(dtype=wp.int32),
    radius: wp.array(dtype=Any),  # [n_agents] static per-agent radius
    geom: wp.array(dtype=Any),  # [4] = (box_cx, box_cy, box_hx, box_hy)
    reset_mask: wp.array(dtype=wp.uint8),
    k_obs: wp.int32,
    wall_reach: Any,
    pos_shaping_factor: Any,
    goal_tolerance: Any,
    advance_prev: wp.int32,
    full_pass: wp.int32,
    obs: wp.array3d(dtype=Any),
    touching: wp.array2d(dtype=Any),
    wall_touch: wp.array2d(dtype=Any),
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
    curriculum step.

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
    d = wp.length(p - g)
    zero = type(d)(0.0)
    one = type(d)(1.0)
    cnt = neighbor_count[e, a]
    ra = radius[a]

    # --- observation row: own features, then the padded neighbour blocks -----------
    obs[e, a, 0] = p[0]
    obs[e, a, 1] = p[1]
    obs[e, a, 2] = v[0]
    obs[e, a, 3] = v[1]
    obs[e, a, 4] = grel[0]
    obs[e, a, 5] = grel[1]
    rp_base = wp.int32(6)
    rv_base = wp.int32(6) + wp.int32(2) * k_obs
    vd_base = wp.int32(6) + wp.int32(4) * k_obs
    for j in range(k_obs):
        rpx = zero
        rpy = zero
        rvx = zero
        rvy = zero
        valid = zero
        if j < cnt:
            b = neighbor_idx[e, a, j]
            rpx = pos[e, b][0] - p[0]
            rpy = pos[e, b][1] - p[1]
            rvx = vel[e, b][0] - v[0]
            rvy = vel[e, b][1] - v[1]
            valid = one
        obs[e, a, rp_base + wp.int32(2) * j] = rpx
        obs[e, a, rp_base + wp.int32(2) * j + 1] = rpy
        obs[e, a, rv_base + wp.int32(2) * j] = rvx
        obs[e, a, rv_base + wp.int32(2) * j + 1] = rvy
        obs[e, a, vd_base + j] = valid

    # --- agent-agent contacts, over this agent's own neighbour list ----------------
    touch = zero
    for ni in range(cnt):
        b = neighbor_idx[e, a, ni]
        nd = wp.length(pos[e, b] - p)
        if nd < ra + radius[b]:
            touch += one

    # --- corridor-wall contact: one folded box SDF, not a four-obstacle scan --------
    wcontact = zero
    if _corner_sdf(p[0], p[1], geom[0], geom[1], geom[2], geom[3]) < wall_reach:
        wcontact = one

    # --- position shaping baseline (navigation's rebase/advance/gate pattern) -------
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
        wall_touch[e, a] = wcontact
        dist_to_goal[e, a] = d
        if d < goal_tolerance:
            on_goal[e, a] = wp.uint8(1)
        else:
            on_goal[e, a] = wp.uint8(0)


@wp.kernel
def giveway_reward_kernel(
    pos_shaping: wp.array2d(dtype=Any),
    touching: wp.array2d(dtype=Any),
    wall_touch: wp.array2d(dtype=Any),
    on_goal: wp.array(dtype=wp.uint8, ndim=2),
    n_agents: wp.int32,
    inv_n_agents: Any,
    collision_penalty: Any,
    wall_penalty: Any,
    time_penalty: Any,
    final_reward: Any,
    shared_reward: wp.int32,
    reward: wp.array2d(dtype=Any),
    done: wp.array(dtype=wp.uint8),
    frac_on_goal: wp.array(dtype=Any),
):
    """Thread per env; sequential agent loops (deterministic, no atomics).

    ``shared_reward`` shares only the *shaping* sum, as in navigation: the contact and
    time terms stay with the agent that incurred them, so a robot is still told which of
    its own actions cost it something. Sharing the shaping is what makes yielding pay —
    the agent that backs into a passing bay eats a negative term the team's progress then
    more than covers.
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
    for a in range(n_agents):
        r = collision_penalty * touching[e, a] + wall_penalty * wall_touch[e, a] + time_penalty
        if shared_reward == 1:
            r = r + shaping_sum
        else:
            r = r + pos_shaping[e, a]
        reward[e, a] = r + final
    done[e] = all_og
    frac_on_goal[e] = n_on * inv_n_agents


@wp.kernel
def giveway_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    n_agents: wp.int32,
    s_max: Any,
    stagger: Any,
    jitter_long: Any,
    jitter_lat: Any,
    pos: Any,
    theta: Any,
    vel: Any,
    speed: Any,
    ang_vel: Any,
    goals: Any,
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

    for a in range(n_agents):
        arm = (a + k) % 4
        depth = s_max - _as(wp.float32(a / 4), s_max) * stagger
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
        goals[e, a] = wp.vector(-dx * depth, -dy * depth)
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
        wp.array3d(dtype=wp.int32),  # neighbor_idx
        wp.array2d(dtype=wp.int32),  # neighbor_count
        wp.array(dtype=dtype),  # radius
        wp.array(dtype=dtype),  # geom
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # k_obs
        dtype,  # wall_reach
        dtype,  # pos_shaping_factor
        dtype,  # goal_tolerance
        wp.int32,  # advance_prev
        wp.int32,  # full_pass
        a3s,  # obs
        a2s,  # touching
        a2s,  # wall_touch
        a2s,  # dist_to_goal
        a2s,  # pos_shaping
        u8_2,  # on_goal
        a2s,  # prev_dist
    ]


def _reward_signature(dtype) -> list:
    a2s = wp.array2d(dtype=dtype)
    return [
        a2s,  # pos_shaping
        a2s,  # touching
        a2s,  # wall_touch
        wp.array(dtype=wp.uint8, ndim=2),  # on_goal
        wp.int32,  # n_agents
        dtype,  # inv_n_agents
        dtype,  # collision_penalty
        dtype,  # wall_penalty
        dtype,  # time_penalty
        dtype,  # final_reward
        wp.int32,  # shared_reward
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
        wp.int32,  # n_agents
        dtype,  # s_max
        dtype,  # stagger
        dtype,  # jitter_long
        dtype,  # jitter_lat
        a2v,  # pos
        a2s,  # theta
        a2v,  # vel
        a2s,  # speed
        a2s,  # ang_vel
        a2v,  # goals
    ]


for _T in (wp.float32, wp.float64):
    register(giveway_obs_kernel, _T, _obs_signature(_T))
    register(giveway_reward_kernel, _T, _reward_signature(_T))
    register(giveway_reset_kernel, _T, _reset_signature(_T))
