"""Fused Warp kernels for the NavigationScenario obs/reward/done layer.

Replaces the eager-torch per-step cache (neighbor gather, touching count,
position shaping, observation assembly, reward reduction) with two kernels that
share a single neighbor loop. The torch implementation in
:mod:`wmas.scenarios.navigation` stays the reference (and the differentiable
path); these kernels are the no-grad fast path and are validated bit-close
against it (exact where the arithmetic is identical; ulp-close for the trig /
norm / reduction slots).

Layout follows :mod:`wmas.core.collisions`: generic over dtype via ``Any`` with
explicit float32/float64 overloads, and shared / per-env twins that read
``P_RADIUS`` from the ``[n_agents, P]`` or ``[n_envs, n_agents, P]`` layout so
heterogeneous fleets and per-env randomization both work.

Observation row (``obs_dim = 9 + 5 * k_obs``), matching the torch ``cat`` order::

    [ pos(2), vel(2), cosθ, sinθ, ang_vel, goal-pos(2),
      rel_pos(2·k_obs), rel_vel(2·k_obs), valid(k_obs) ]
"""

from __future__ import annotations

from typing import Any

import warp as wp

from wmas.core.state import VEC2
from wmas.dynamics.base import P_RADIUS


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
def nav_obs_kernel_per_env(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
    goals: wp.array2d(dtype=Any),
    neighbor_idx: wp.array3d(dtype=wp.int32),
    neighbor_count: wp.array2d(dtype=wp.int32),
    neighbor_true: wp.array2d(dtype=wp.int32),
    params: wp.array3d(dtype=Any),  # [n_envs, n_agents, NUM_PARAMS] per-env
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
    p = pos[e, a]
    v = vel[e, a]
    g = goals[e, a]
    grel = g - p
    d = wp.length(p - g)
    cnt = neighbor_count[e, a]
    ra = params[e, a, P_RADIUS]
    touch = type(d)(0.0)
    for ni in range(cnt):
        b = neighbor_idx[e, a, ni]
        nd = wp.length(pos[e, b] - p)
        if nd < ra + params[e, b, P_RADIUS]:
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


def _obs_signature(dtype, per_env: bool = False) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    a2s = wp.array2d(dtype=dtype)
    a3i = wp.array3d(dtype=wp.int32)
    a2i = wp.array2d(dtype=wp.int32)
    u8_1 = wp.array(dtype=wp.uint8)
    u8_2 = wp.array(dtype=wp.uint8, ndim=2)
    a3s = wp.array3d(dtype=dtype)
    params = wp.array3d(dtype=dtype) if per_env else a2s
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


for _T in (wp.float32, wp.float64):
    wp.overload(nav_obs_kernel, _obs_signature(_T))
    wp.overload(nav_obs_kernel_per_env, _obs_signature(_T, per_env=True))
    wp.overload(nav_reward_kernel, _reward_signature(_T))
