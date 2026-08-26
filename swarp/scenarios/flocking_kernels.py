"""Fused Warp kernel for the FlockingScenario obs/reward layer.

Replaces the eager-torch per-step cache (neighbor gather, centroid/velocity
means, crowding, observation assembly, reward) with a single kernel threaded per
(env, agent). The torch implementation in :mod:`swarp.scenarios.flocking` stays
the reference (and the differentiable path); this kernel is the no-grad fast
path, validated bit-close against it (ulp-close for the norm/reduction slots;
flocking has no discrete flags).

Layout follows :mod:`swarp.core.collisions`: generic over dtype via ``Any`` with
explicit float32/float64 overloads. The reward sums over the *full* within-radius
neighbor list (``neighbor_count`` slots), while the observation exposes only the
first ``k_obs`` neighbors — matching the torch reference's cat order::

    [ pos(2), vel(2), rel_pos(2·k_obs), rel_vel(2·k_obs), valid(k_obs) ]
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.state import VEC2


@wp.kernel
def flocking_obs_reward_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    neighbor_idx: wp.array3d(dtype=wp.int32),
    neighbor_count: wp.array2d(dtype=wp.int32),
    k_obs: wp.int32,
    cohesion: Any,
    alignment: Any,
    separation: Any,
    separation_dist: Any,
    full_pass: wp.int32,
    obs: wp.array3d(dtype=Any),
    reward: wp.array2d(dtype=Any),
    crowd_out: wp.array2d(dtype=Any),
):
    """Thread per (env, agent): boids reward + observation row."""
    e, a = wp.tid()
    p = pos[e, a]
    v = vel[e, a]
    px = p[0]
    py = p[1]
    vx = v[0]
    vy = v[1]
    zero = type(px)(0.0)
    one = type(px)(1.0)
    cnt = neighbor_count[e, a]

    # Reward aggregates over the full neighbor list (all `cnt` valid slots).
    srpx = zero
    srpy = zero
    srvx = zero
    srvy = zero
    crowd = zero
    nf = zero
    for ni in range(cnt):
        b = neighbor_idx[e, a, ni]
        dpx = pos[e, b][0] - px
        dpy = pos[e, b][1] - py
        dvx = vel[e, b][0] - vx
        dvy = vel[e, b][1] - vy
        srpx += dpx
        srpy += dpy
        srvx += dvx
        srvy += dvy
        nd = wp.sqrt(dpx * dpx + dpy * dpy)
        gap = separation_dist - nd
        if gap > zero:
            crowd += gap
        nf += one
    # mean neighbor offset / velocity difference (divisor clamped to >= 1)
    den = one
    if nf > one:
        den = nf
    cox = srpx / den
    coy = srpy / den
    vox = srvx / den
    voy = srvy / den
    co_norm = wp.sqrt(cox * cox + coy * coy)
    vo_norm = wp.sqrt(vox * vox + voy * voy)

    # Observation row: own pose then up to k_obs neighbor rel pos/vel + validity.
    obs[e, a, 0] = px
    obs[e, a, 1] = py
    obs[e, a, 2] = vx
    obs[e, a, 3] = vy
    rp_base = wp.int32(4)
    rv_base = wp.int32(4) + wp.int32(2) * k_obs
    vd_base = wp.int32(4) + wp.int32(4) * k_obs
    for j in range(k_obs):
        rpx = zero
        rpy = zero
        rvx = zero
        rvy = zero
        vld = zero
        if j < cnt:
            b = neighbor_idx[e, a, j]
            rpx = pos[e, b][0] - px
            rpy = pos[e, b][1] - py
            rvx = vel[e, b][0] - vx
            rvy = vel[e, b][1] - vy
            vld = one
        obs[e, a, rp_base + wp.int32(2) * j] = rpx
        obs[e, a, rp_base + wp.int32(2) * j + 1] = rpy
        obs[e, a, rv_base + wp.int32(2) * j] = rvx
        obs[e, a, rv_base + wp.int32(2) * j + 1] = rvy
        obs[e, a, vd_base + j] = vld

    # Reward / info gated behind full_pass so a post-auto-reset obs-only pass
    # never clobbers the reward already returned for the transition.
    if full_pass == 1:
        reward[e, a] = -cohesion * co_norm - alignment * vo_norm - separation * crowd
        crowd_out[e, a] = crowd


def _signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    a2s = wp.array2d(dtype=dtype)
    a3s = wp.array3d(dtype=dtype)
    return [
        a2v,  # pos
        a2v,  # vel
        wp.array3d(dtype=wp.int32),  # neighbor_idx
        wp.array2d(dtype=wp.int32),  # neighbor_count
        wp.int32,  # k_obs
        dtype,  # cohesion
        dtype,  # alignment
        dtype,  # separation
        dtype,  # separation_dist
        wp.int32,  # full_pass
        a3s,  # obs
        a2s,  # reward
        a2s,  # crowd_out
    ]


for _T in (wp.float32, wp.float64):
    register(flocking_obs_reward_kernel, _T, _signature(_T))
