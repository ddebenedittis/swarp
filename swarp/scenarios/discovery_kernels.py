"""Fused Warp kernels for the DiscoveryScenario obs/reward layer.

Three launches, ordered ``cover -> {obs, reward}``:

* ``discovery_cover_kernel`` (thread per env,target) counts agents within the
  covering range, latches ``covered`` monotonically **in place**, and records
  ``newly`` from the *pre-update* latch. Each ``(env, target)`` entry is owned by
  exactly one thread, so the read-old/write-new is race-free without a snapshot.
* ``discovery_obs_kernel`` (thread per env,agent) builds the obs row and the
  all-pairs touching count, reading the *post-update* ``covered`` as the
  broadcast flag.
* ``discovery_reward_kernel`` (thread per env) folds the per-agent term
  (collision + time penalty) and the shared covering reward, and reduces
  ``done``.

The torch implementation in :mod:`swarp.scenarios.discovery` stays the reference
(and the differentiable path); its ``cdist`` distances are pinned to squared
broadcast diffs so the discrete coverage/touching flags match the kernels'
``dx*dx + dy*dy`` comparisons bit-for-bit. Observation cat order
(``obs_dim = 4 + 3 * n_targets``)::

    [ pos(2), vel(2), rel_targets(2*T), covered_flag(T) ]
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.state import VEC2


@wp.kernel
def discovery_cover_kernel(
    pos: wp.array2d(dtype=Any),
    targets: wp.array2d(dtype=Any),
    n_agents: wp.int32,
    cover_range_sq: Any,
    agents_per_target: wp.int32,
    covered: wp.array(dtype=wp.uint8, ndim=2),
    newly: wp.array(dtype=wp.uint8, ndim=2),
):
    """Thread per (env, target): count coverers, latch covered, record newly."""
    e, t = wp.tid()
    tg = targets[e, t]
    cnt = wp.int32(0)
    for a in range(n_agents):
        dx = pos[e, a][0] - tg[0]
        dy = pos[e, a][1] - tg[1]
        if dx * dx + dy * dy < cover_range_sq:
            cnt += 1
    covered_now = wp.uint8(0)
    if cnt >= agents_per_target:
        covered_now = wp.uint8(1)
    old = covered[e, t]
    if covered_now == wp.uint8(1) and old == wp.uint8(0):
        newly[e, t] = wp.uint8(1)
    else:
        newly[e, t] = wp.uint8(0)
    if covered_now == wp.uint8(1):
        covered[e, t] = wp.uint8(1)


@wp.kernel
def discovery_obs_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    targets: wp.array2d(dtype=Any),
    covered: wp.array(dtype=wp.uint8, ndim=2),
    n_agents: wp.int32,
    n_targets: wp.int32,
    col_dist_sq: Any,
    obs: wp.array3d(dtype=Any),
    touching: wp.array2d(dtype=Any),
):
    """Thread per (env, agent): obs row (rel targets + covered flag) + touching."""
    e, a = wp.tid()
    p = pos[e, a]
    v = vel[e, a]
    px = p[0]
    py = p[1]
    zero = type(px)(0.0)
    one = type(px)(1.0)
    obs[e, a, 0] = px
    obs[e, a, 1] = py
    obs[e, a, 2] = v[0]
    obs[e, a, 3] = v[1]
    for t in range(n_targets):
        tg = targets[e, t]
        obs[e, a, wp.int32(4) + wp.int32(2) * t] = tg[0] - px
        obs[e, a, wp.int32(4) + wp.int32(2) * t + 1] = tg[1] - py
    cflag_base = wp.int32(4) + wp.int32(2) * n_targets
    for t in range(n_targets):
        cv = zero
        if covered[e, t] == wp.uint8(1):
            cv = one
        obs[e, a, cflag_base + t] = cv

    # All-pairs touching (squared distance, minus self), matches navigation-style.
    cnt = zero
    for b in range(n_agents):
        dx = pos[e, b][0] - px
        dy = pos[e, b][1] - py
        if dx * dx + dy * dy < col_dist_sq:
            cnt += one
    touching[e, a] = cnt - one


@wp.kernel
def discovery_reward_kernel(
    touching: wp.array2d(dtype=Any),
    newly: wp.array(dtype=wp.uint8, ndim=2),
    covered: wp.array(dtype=wp.uint8, ndim=2),
    n_agents: wp.int32,
    n_targets: wp.int32,
    collision_penalty: Any,
    time_penalty: Any,
    covering_reward: Any,
    reward: wp.array2d(dtype=Any),
    done: wp.array(dtype=wp.uint8),
):
    """Thread per env; sequential loops (deterministic, no atomics)."""
    e = wp.tid()
    n_new = type(collision_penalty)(0.0)
    all_cov = wp.uint8(1)
    for t in range(n_targets):
        if newly[e, t] == wp.uint8(1):
            n_new += type(collision_penalty)(1.0)
        if covered[e, t] == wp.uint8(0):
            all_cov = wp.uint8(0)
    global_r = covering_reward * n_new
    for a in range(n_agents):
        reward[e, a] = collision_penalty * touching[e, a] + time_penalty + global_r
    done[e] = all_cov


def _cover_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    return [
        wp.array2d(dtype=vec2),  # pos
        wp.array2d(dtype=vec2),  # targets
        wp.int32,  # n_agents
        dtype,  # cover_range_sq
        wp.int32,  # agents_per_target
        wp.array(dtype=wp.uint8, ndim=2),  # covered
        wp.array(dtype=wp.uint8, ndim=2),  # newly
    ]


def _obs_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    return [
        wp.array2d(dtype=vec2),  # pos
        wp.array2d(dtype=vec2),  # vel
        wp.array2d(dtype=vec2),  # targets
        wp.array(dtype=wp.uint8, ndim=2),  # covered
        wp.int32,  # n_agents
        wp.int32,  # n_targets
        dtype,  # col_dist_sq
        wp.array3d(dtype=dtype),  # obs
        wp.array2d(dtype=dtype),  # touching
    ]


def _reward_signature(dtype) -> list:
    a2s = wp.array2d(dtype=dtype)
    return [
        a2s,  # touching
        wp.array(dtype=wp.uint8, ndim=2),  # newly
        wp.array(dtype=wp.uint8, ndim=2),  # covered
        wp.int32,  # n_agents
        wp.int32,  # n_targets
        dtype,  # collision_penalty
        dtype,  # time_penalty
        dtype,  # covering_reward
        a2s,  # reward
        wp.array(dtype=wp.uint8),  # done
    ]


for _T in (wp.float32, wp.float64):
    register(discovery_cover_kernel, _T, _cover_signature(_T))
    register(discovery_obs_kernel, _T, _obs_signature(_T))
    register(discovery_reward_kernel, _T, _reward_signature(_T))
