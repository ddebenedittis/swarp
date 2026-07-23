"""Fused Warp kernels for the SamplingScenario obs/reward layer.

Two launches preserve the torch reference's read-before-write ordering on the
persistent ``consumed`` grid: every agent reads ``was_consumed`` (pass 1) before
any cell is marked (pass 2), so agents sharing a not-yet-consumed cell both earn
the field value — exactly as the reference's gather-before-scatter does. The
scatter writes the constant ``1``, so the race between co-located agents is
benign and needs no atomics.

The torch implementation in :mod:`wmas.scenarios.sampling` stays the reference
(and the differentiable path). Observation cat order (``obs_dim = 13``)::

    [ pos(2), vel(2), samples(9) ]   # samples = field on the 3x3 cell block

Layout follows :mod:`wmas.core.collisions`: generic over dtype via ``Any`` with
explicit float32/float64 overloads.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from wmas.core.state import VEC2


@wp.func
def _field_at(
    e: wp.int32,
    x: Any,
    y: Any,
    centers: wp.array2d(dtype=Any),
    n_gaussians: wp.int32,
    denom: Any,
):
    """Sum-of-Gaussians density at world point (x, y) for env ``e``."""
    acc = type(x)(0.0)
    for gi in range(n_gaussians):
        c = centers[e, gi]
        dx = x - c[0]
        dy = y - c[1]
        d2 = dx * dx + dy * dy
        acc += wp.exp(-d2 / denom)
    return acc


@wp.func
def _cell_xy(px: Any, py: Any, world_size: Any, grid_res: wp.int32):
    """Clamped integer cell coordinates (cx, cy) for a world position."""
    two_w = type(px)(2.0) * world_size
    resf = type(px)(grid_res)
    cxf = wp.floor((px + world_size) / two_w * resf)
    cyf = wp.floor((py + world_size) / two_w * resf)
    hi = grid_res - wp.int32(1)
    cx = wp.clamp(wp.int32(cxf), wp.int32(0), hi)
    cy = wp.clamp(wp.int32(cyf), wp.int32(0), hi)
    return cx, cy


@wp.kernel
def sampling_obs_reward_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    centers: wp.array2d(dtype=Any),
    consumed: wp.array(dtype=wp.uint8, ndim=2),
    world_size: Any,
    grid_res: wp.int32,
    n_gaussians: wp.int32,
    denom: Any,
    full_pass: wp.int32,
    obs: wp.array3d(dtype=Any),
    reward: wp.array2d(dtype=Any),
    field_out: wp.array2d(dtype=Any),
):
    """Thread per (env, agent): read the consumed grid, earn the field, build obs."""
    e, a = wp.tid()
    p = pos[e, a]
    px = p[0]
    py = p[1]
    v = vel[e, a]

    cx, cy = _cell_xy(px, py, world_size, grid_res)
    cell = cy * grid_res + cx
    was = consumed[e, cell]
    fval = _field_at(e, px, py, centers, n_gaussians, denom)

    # Observation: own pose + the field sampled on the 3x3 cell neighborhood.
    obs[e, a, 0] = px
    obs[e, a, 1] = py
    obs[e, a, 2] = v[0]
    obs[e, a, 3] = v[1]
    two_w = type(px)(2.0) * world_size
    resf = type(px)(grid_res)
    half = type(px)(0.5)
    hi = grid_res - wp.int32(1)
    cxc = wp.clamp(cx, wp.int32(1), grid_res - wp.int32(2))
    cyc = wp.clamp(cy, wp.int32(1), grid_res - wp.int32(2))
    for i in range(3):
        gx = wp.clamp(cxc + (wp.int32(i) - wp.int32(1)), wp.int32(0), hi)
        xco = (type(px)(gx) + half) / resf * two_w - world_size
        for jj in range(3):
            gy = wp.clamp(cyc + (wp.int32(jj) - wp.int32(1)), wp.int32(0), hi)
            yco = (type(px)(gy) + half) / resf * two_w - world_size
            obs[e, a, wp.int32(4) + wp.int32(i) * wp.int32(3) + wp.int32(jj)] = _field_at(
                e, xco, yco, centers, n_gaussians, denom
            )

    # Reward / field gated behind full_pass so a post-auto-reset obs-only pass
    # never clobbers the values already returned for the transition.
    if full_pass == 1:
        r = type(px)(0.0)
        if was == wp.uint8(0):
            r = fval
        reward[e, a] = r
        field_out[e, a] = fval


@wp.kernel
def sampling_scatter_kernel(
    pos: wp.array2d(dtype=Any),
    world_size: Any,
    grid_res: wp.int32,
    consumed: wp.array(dtype=wp.uint8, ndim=2),
):
    """Thread per (env, agent): mark the agent's cell consumed (idempotent)."""
    e, a = wp.tid()
    p = pos[e, a]
    cx, cy = _cell_xy(p[0], p[1], world_size, grid_res)
    consumed[e, cy * grid_res + cx] = wp.uint8(1)


def _obs_reward_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    a2s = wp.array2d(dtype=dtype)
    return [
        a2v,  # pos
        a2v,  # vel
        a2v,  # centers
        wp.array(dtype=wp.uint8, ndim=2),  # consumed
        dtype,  # world_size
        wp.int32,  # grid_res
        wp.int32,  # n_gaussians
        dtype,  # denom
        wp.int32,  # full_pass
        wp.array3d(dtype=dtype),  # obs
        a2s,  # reward
        a2s,  # field_out
    ]


def _scatter_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    return [
        wp.array2d(dtype=vec2),  # pos
        dtype,  # world_size
        wp.int32,  # grid_res
        wp.array(dtype=wp.uint8, ndim=2),  # consumed
    ]


for _T in (wp.float32, wp.float64):
    wp.overload(sampling_obs_reward_kernel, _obs_reward_signature(_T))
    wp.overload(sampling_scatter_kernel, _scatter_signature(_T))
