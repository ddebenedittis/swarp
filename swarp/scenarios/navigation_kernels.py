"""Fused Warp kernels for the NavigationScenario obs/reward/done layer.

Replaces the eager-torch per-step cache (neighbor gather, touching count,
position shaping, observation assembly, reward reduction) with two kernels that
share a single neighbor loop. The torch implementation in
:mod:`swarp.scenarios.navigation` stays the reference (and the differentiable
path); these kernels are the no-grad fast path and are validated bit-close
against it (exact where the arithmetic is identical; ulp-close for the trig /
norm / reduction slots).

Layout follows :mod:`swarp.core.collisions`: generic over dtype via ``Any`` with
explicit float32/float64 overloads. The touching count uses the *static*
per-agent radius (``params.floats[a, P_RADIUS]``), matching the torch reference
(``World.agent_radius``) — heterogeneous static radii are honored per agent;
per-env radius randomization affects the physics forces but, like the reference,
not the reward's touching count.

Observation row (``obs_dim = 9 + 5 * k_obs``), matching the torch ``cat`` order::

    [ pos(2), vel(2), cosθ, sinθ, ang_vel, goal-pos(2),
      rel_pos(2·k_obs), rel_vel(2·k_obs), valid(k_obs) ]
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.rng import seed_from_state
from swarp.core.state import VEC2
from swarp.dynamics.base import P_RADIUS
from swarp.scenarios.reset_kernels import _as


@wp.func
def _obs_row(
    e: wp.int32,
    a: wp.int32,
    p: Any,
    v: Any,
    th: Any,
    wv: Any,
    grel: Any,
    k_obs: wp.int32,
    cnt: wp.int32,
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    neighbor_idx: wp.array3d(dtype=wp.int32),
    obs: wp.array3d(dtype=Any),
):
    """Write one agent's observation row (own features + up to ``k_obs`` neighbor
    features, invalid slots zeroed)."""
    zero = type(th)(0.0)
    obs[e, a, 0] = p[0]
    obs[e, a, 1] = p[1]
    obs[e, a, 2] = v[0]
    obs[e, a, 3] = v[1]
    obs[e, a, 4] = wp.cos(th)
    obs[e, a, 5] = wp.sin(th)
    obs[e, a, 6] = wv
    obs[e, a, 7] = grel[0]
    obs[e, a, 8] = grel[1]
    rp_base = wp.int32(9)
    rv_base = wp.int32(9) + wp.int32(2) * k_obs
    vd_base = wp.int32(9) + wp.int32(4) * k_obs
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
            valid = type(th)(1.0)
        obs[e, a, rp_base + wp.int32(2) * j] = rpx
        obs[e, a, rp_base + wp.int32(2) * j + 1] = rpy
        obs[e, a, rv_base + wp.int32(2) * j] = rvx
        obs[e, a, rv_base + wp.int32(2) * j + 1] = rvy
        obs[e, a, vd_base + j] = valid


@wp.func
def _shaping_and_flags(
    e: wp.int32,
    a: wp.int32,
    d: Any,
    touch: Any,
    neighbor_true: wp.int32,
    cnt: wp.int32,
    pos_shaping_factor: Any,
    goal_tolerance: Any,
    advance_prev: wp.int32,
    full_pass: wp.int32,
    reset_hit: wp.int32,
    prev_dist: wp.array2d(dtype=Any),
    touching: wp.array2d(dtype=Any),
    dist_to_goal: wp.array2d(dtype=Any),
    pos_shaping: wp.array2d(dtype=Any),
    on_goal: wp.array(dtype=wp.uint8, ndim=2),
    overflow: wp.array(dtype=wp.uint8, ndim=2),
):
    """Position-shaping baseline update + reward-input buffers.

    ``reset_hit`` marks a just-reset env (rebase the baseline, zero the shaping);
    otherwise ``advance_prev`` advances the baseline to the current distance.
    ``full_pass`` gates the reward/done input buffers so a post-reset obs-only
    pass never clobbers the values the reward already consumed this step."""
    zero = type(d)(0.0)
    prev = prev_dist[e, a]
    ps = (prev - d) * pos_shaping_factor
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
        dist_to_goal[e, a] = d
        if d < goal_tolerance:
            on_goal[e, a] = wp.uint8(1)
        else:
            on_goal[e, a] = wp.uint8(0)
        if neighbor_true > cnt:
            overflow[e, a] = wp.uint8(1)
        else:
            overflow[e, a] = wp.uint8(0)


@wp.kernel
def nav_obs_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
    goals: wp.array2d(dtype=Any),
    neighbor_idx: wp.array3d(dtype=wp.int32),
    neighbor_count: wp.array2d(dtype=wp.int32),
    neighbor_true: wp.array2d(dtype=wp.int32),
    params: wp.array2d(dtype=Any),  # [n_agents, NUM_PARAMS] shared
    reset_mask: wp.array(dtype=wp.uint8),
    k_obs: wp.int32,
    pos_shaping_factor: Any,
    goal_tolerance: Any,
    advance_prev: wp.int32,
    full_pass: wp.int32,
    obs: wp.array3d(dtype=Any),
    touching: wp.array2d(dtype=Any),
    dist_to_goal: wp.array2d(dtype=Any),
    pos_shaping: wp.array2d(dtype=Any),
    on_goal: wp.array(dtype=wp.uint8, ndim=2),
    overflow: wp.array(dtype=wp.uint8, ndim=2),
    prev_dist: wp.array2d(dtype=Any),
):
    e, a = wp.tid()
    if full_pass == 0 and reset_mask[e] == wp.uint8(0):
        # Obs-only auto-reset pass: an env this mask didn't select has state
        # identical to what the STEP pass moments earlier already wrote into
        # every output buffer below, so redoing the neighbor gather/shaping
        # math for it is pure waste. Reset envs (reset_mask[e] == 1) still
        # fall through and get recomputed.
        return
    p = pos[e, a]
    v = vel[e, a]
    g = goals[e, a]
    grel = g - p
    d = wp.length(p - g)
    cnt = neighbor_count[e, a]
    ra = params[a, P_RADIUS]
    touch = type(d)(0.0)
    for ni in range(cnt):
        b = neighbor_idx[e, a, ni]
        nd = wp.length(pos[e, b] - p)
        if nd < ra + params[b, P_RADIUS]:
            touch += type(d)(1.0)
    _obs_row(e, a, p, v, theta[e, a], ang_vel[e, a], grel, k_obs, cnt, pos, vel, neighbor_idx, obs)
    reset_hit = wp.int32(reset_mask[e])
    _shaping_and_flags(
        e,
        a,
        d,
        touch,
        neighbor_true[e, a],
        cnt,
        pos_shaping_factor,
        goal_tolerance,
        advance_prev,
        full_pass,
        reset_hit,
        prev_dist,
        touching,
        dist_to_goal,
        pos_shaping,
        on_goal,
        overflow,
    )


@wp.kernel
def nav_reward_kernel(
    touching: wp.array2d(dtype=Any),
    pos_shaping: wp.array2d(dtype=Any),
    on_goal: wp.array(dtype=wp.uint8, ndim=2),
    n_agents: wp.int32,
    collision_penalty: Any,
    final_reward: Any,
    shared_reward: wp.int32,
    reward: wp.array2d(dtype=Any),
    done: wp.array(dtype=wp.uint8),
):
    """Thread per env; sequential agent loops (deterministic, no atomics)."""
    e = wp.tid()
    zero = type(collision_penalty)(0.0)
    shaping_sum = zero
    all_og = wp.uint8(1)
    for a in range(n_agents):
        shaping_sum += pos_shaping[e, a]
        if on_goal[e, a] == wp.uint8(0):
            all_og = wp.uint8(0)
    final = zero
    if all_og == wp.uint8(1):
        final = final_reward
    for a in range(n_agents):
        r = collision_penalty * touching[e, a]
        if shared_reward == 1:
            r += shaping_sum
        else:
            r += pos_shaping[e, a]
        reward[e, a] = r + final
    done[e] = all_og


def _obs_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    a2s = wp.array2d(dtype=dtype)
    a3i = wp.array3d(dtype=wp.int32)
    a2i = wp.array2d(dtype=wp.int32)
    u8_1 = wp.array(dtype=wp.uint8)
    u8_2 = wp.array(dtype=wp.uint8, ndim=2)
    a3s = wp.array3d(dtype=dtype)
    params = a2s
    return [
        a2v,  # pos
        a2v,  # vel
        a2s,  # theta
        a2s,  # ang_vel
        a2v,  # goals
        a3i,  # neighbor_idx
        a2i,  # neighbor_count
        a2i,  # neighbor_true
        params,
        u8_1,  # reset_mask
        wp.int32,  # k_obs
        dtype,  # pos_shaping_factor
        dtype,  # goal_tolerance
        wp.int32,  # advance_prev
        wp.int32,  # full_pass
        a3s,  # obs
        a2s,  # touching
        a2s,  # dist_to_goal
        a2s,  # pos_shaping
        u8_2,  # on_goal
        u8_2,  # overflow
        a2s,  # prev_dist
    ]


@wp.kernel
def nav_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed_state: wp.array(dtype=wp.int32),
    lim: Any,
    cell: Any,
    jitter: Any,
    grid: wp.int32,
    n_cells: wp.int32,
    stratified: wp.int32,
    n_agents: wp.int32,
    perm: wp.array2d(dtype=wp.int32),  # [n_cells, n_envs] — env-major would not coalesce
    pos: Any,
    theta: Any,
    vel: Any,
    speed: Any,
    ang_vel: Any,
    goals: Any,
):
    """One masked episode reset per env: spawns, goals, headings, zeroed velocities.

    Thread per **env**, not per agent: the distinct-cell draw is a partial Fisher-Yates
    over this env's slice of ``perm``, which is inherently sequential. That is the whole
    reason this is a kernel rather than the torch chain it replaces — ``argsort`` is the
    only way to get a uniform random k-subset out of batched torch ops, and under
    ``auto_reset`` it costs a sort over ``[n_envs, n_cells]`` twice on *every* step.

    ``perm`` is **cell-major**, ``[n_cells, n_envs]``, and every access is ``perm[i, e]``:
    with one thread per env, the threads of a warp move through the Fisher-Yates in
    lockstep on ``i``, so a cell-major layout puts their 32 accesses in one contiguous
    line. The obvious ``[n_envs, n_cells]`` gives each thread a private contiguous row and
    strides adjacent threads ``n_cells * 4`` bytes apart — a separate memory transaction
    per lane, for a scratch buffer this kernel touches ``2 * (n_cells + n_agents)`` times
    per env. The permutation drawn is identical either way; only the addressing changes.

    ``stratified == 0`` is the packing-limit fallback: no grid both fits the points and
    leaves jitter room, so draw uniformly and give up the separation guarantee rather than
    pin every point to a cell centre. Mirrors the torch reference's fallback in
    :meth:`~swarp.scenarios.navigation.NavigationScenario._sample_separated`.

    ``_as`` widens the float32 ``wp.randf`` draw to the world's scalar type; everything
    downstream is generic arithmetic, so a float64 world gets float64 spawns.

    ``seed_state`` is the device-side ``[base, counter]`` pair from ``World.seed_state``
    (see ``swarp/core/rng.py``), not a plain scalar: a scalar argument gets baked into a
    captured launch by value and would replay the same seed forever, where this array is
    baked by pointer and its *contents* can still change between graph replays. The caller
    (``NavigationScenario._launch_reset``) launches ``advance_seed_kernel`` on this same
    array immediately before this kernel, every call — advance, then use — which is what
    reproduces ``World.next_kernel_seed``'s "increment first, return after" stream exactly.
    """
    e = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return

    rng = wp.rand_init(seed_from_state(seed_state), e)
    # Typed constants: Warp reads a bare float literal as float32, so every literal that
    # meets the world's scalar type has to be widened through ``_as`` first.
    zero = _as(0.0, lim)
    one = _as(1.0, lim)
    two = _as(2.0, lim)
    mid = _as(0.5, lim)
    pi = _as(3.14159265358979, lim)

    for a in range(n_agents):
        vel[e, a] = wp.vector(zero, zero)
        speed[e, a] = zero
        ang_vel[e, a] = zero
        theta[e, a] = (_as(wp.randf(rng), lim) * two - one) * pi

    if stratified == 0:
        for a in range(n_agents):
            x = (_as(wp.randf(rng), lim) * two - one) * lim
            y = (_as(wp.randf(rng), lim) * two - one) * lim
            pos[e, a] = wp.vector(x, y)
        for a in range(n_agents):
            x = (_as(wp.randf(rng), lim) * two - one) * lim
            y = (_as(wp.randf(rng), lim) * two - one) * lim
            goals[e, a] = wp.vector(x, y)
        return

    half = cell * mid - lim

    # Spawns, then goals: two independent uniform k-subsets of the cell grid.
    for i in range(n_cells):
        perm[i, e] = i
    for i in range(n_agents):
        j = i + wp.int32(wp.randf(rng) * wp.float32(n_cells - i))
        if j > n_cells - 1:
            j = n_cells - 1
        swap = perm[i, e]
        perm[i, e] = perm[j, e]
        perm[j, e] = swap
    for a in range(n_agents):
        c = perm[a, e]
        x = _as(wp.float32(c % grid), lim) * cell + half
        y = _as(wp.float32(c / grid), lim) * cell + half
        x += (_as(wp.randf(rng), lim) - mid) * jitter
        y += (_as(wp.randf(rng), lim) - mid) * jitter
        pos[e, a] = wp.vector(x, y)

    for i in range(n_cells):
        perm[i, e] = i
    for i in range(n_agents):
        j = i + wp.int32(wp.randf(rng) * wp.float32(n_cells - i))
        if j > n_cells - 1:
            j = n_cells - 1
        swap = perm[i, e]
        perm[i, e] = perm[j, e]
        perm[j, e] = swap
    for a in range(n_agents):
        c = perm[a, e]
        x = _as(wp.float32(c % grid), lim) * cell + half
        y = _as(wp.float32(c / grid), lim) * cell + half
        x += (_as(wp.randf(rng), lim) - mid) * jitter
        y += (_as(wp.randf(rng), lim) - mid) * jitter
        goals[e, a] = wp.vector(x, y)


def _reward_signature(dtype) -> list:
    a2s = wp.array2d(dtype=dtype)
    return [
        a2s,  # touching
        a2s,  # pos_shaping
        wp.array(dtype=wp.uint8, ndim=2),  # on_goal
        wp.int32,  # n_agents
        dtype,  # collision_penalty
        dtype,  # final_reward
        wp.int32,  # shared_reward
        a2s,  # reward
        wp.array(dtype=wp.uint8),  # done
    ]


def _reset_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    a2s = wp.array2d(dtype=dtype)
    return [
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # use_mask
        wp.array(dtype=wp.int32),  # seed_state
        dtype,  # lim
        dtype,  # cell
        dtype,  # jitter
        wp.int32,  # grid
        wp.int32,  # n_cells
        wp.int32,  # stratified
        wp.int32,  # n_agents
        wp.array2d(dtype=wp.int32),  # perm scratch
        a2v,  # pos
        a2s,  # theta
        a2v,  # vel
        a2s,  # speed
        a2s,  # ang_vel
        a2v,  # goals
    ]


for _T in (wp.float32, wp.float64):
    register(nav_obs_kernel, _T, _obs_signature(_T))
    register(nav_reward_kernel, _T, _reward_signature(_T))
    register(nav_reset_kernel, _T, _reset_signature(_T))
