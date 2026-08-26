"""Warp-kernel lidar backend: ray-circle scan with no dense pairwise intermediate.

The torch backend in :mod:`swarp.sensors.lidar` materializes a family of dense
``[n_envs, n_agents, n_rays, n_targets]`` tensors (``proj``, ``perp2``, ``thc``,
``t``, ``valid``), so its memory scales as ``O(E*A*R*T)``. This kernel computes
the identical ranges (same analytic ray-circle test) but one thread owns an
``(env, agent)`` cell, loops over rays and targets, and keeps a thread-local
running-min — so only the ``[E, A, R]`` output is stored (``O(E*A*R)``, flat in
the target count). This is the trick Isaac Sim uses to keep high ray counts
affordable.

The kernel is inference-only (launched ``record_tape=False``); the differentiable
path stays the torch backend. Kept numerically equivalent to ``lidar_scan`` and
asserted so in ``tests/unit/test_lidar.py``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from swarp.core.state import TORCH_DTYPE_TO_WP, VEC2


@wp.func
def _ray_hit(oc: Any, d: Any, rad: Any, miss: Any):
    """Range along unit dir ``d`` to the circle at relative center ``oc`` (center -
    origin), or ``miss`` if the ray does not hit. Mirrors the torch ``cast`` at
    ``lidar.py``: ``proj``/``perp2`` projection, ``t = proj - sqrt(rad^2 - perp2)``,
    hit iff ``perp2 <= rad^2`` and ``proj > 0`` and ``t > 0``."""
    zero = type(rad)(0.0)
    proj = wp.dot(oc, d)
    perp2 = wp.dot(oc, oc) - proj * proj
    rad2 = rad * rad
    thc = wp.sqrt(wp.max(rad2 - perp2, zero))
    t = proj - thc
    out = miss
    if (perp2 <= rad2) and (proj > zero) and (t > zero):
        out = t
    return out


@wp.kernel
def lidar_scan_kernel(
    pos: wp.array2d(dtype=Any),  # [E, A] vec2
    theta: wp.array2d(dtype=Any),  # [E, A] scalar
    agent_radius: wp.array(dtype=Any),  # [A] scalar
    obs_pos: wp.array2d(dtype=Any),  # [E, T] vec2
    obs_radius: wp.array(dtype=Any),  # [T] scalar
    n_rays: wp.int32,
    max_range: Any,
    ray_step: Any,  # 2*pi / n_rays, precomputed to match torch exactly
    angle_start: Any,
    body_frame: wp.int32,
    include_agents: wp.int32,
    n_obstacles: wp.int32,
    range_out: wp.array3d(dtype=Any),  # [E, A, R] scalar
):
    e, a = wp.tid()
    p = pos[e, a]
    n_agents = pos.shape[1]

    base = type(max_range)(0.0)
    if body_frame != 0:
        base = theta[e, a]

    for r in range(n_rays):
        ang = base + angle_start + type(max_range)(r) * ray_step
        d = type(p)(wp.cos(ang), wp.sin(ang))
        best = max_range
        if include_agents != 0:
            for b in range(n_agents):
                if b != a:
                    best = wp.min(best, _ray_hit(pos[e, b] - p, d, agent_radius[b], max_range))
        for o in range(n_obstacles):
            best = wp.min(best, _ray_hit(obs_pos[e, o] - p, d, obs_radius[o], max_range))
        range_out[e, a, r] = best


def _signature(dtype) -> list:
    vec2 = VEC2[dtype]
    return [
        wp.array2d(dtype=vec2),  # pos
        wp.array2d(dtype=dtype),  # theta
        wp.array(dtype=dtype),  # agent_radius
        wp.array2d(dtype=vec2),  # obs_pos
        wp.array(dtype=dtype),  # obs_radius
        wp.int32,  # n_rays
        dtype,  # max_range
        dtype,  # ray_step
        dtype,  # angle_start
        wp.int32,  # body_frame
        wp.int32,  # include_agents
        wp.int32,  # n_obstacles
        wp.array3d(dtype=dtype),  # range_out
    ]


for _T in (wp.float32, wp.float64):
    wp.overload(lidar_scan_kernel, _signature(_T))


def lidar_scan_warp(
    pos: torch.Tensor,  # [n_envs, n_agents, 2]
    theta: torch.Tensor,  # [n_envs, n_agents]
    agent_radius: torch.Tensor,  # [n_agents]
    n_rays: int = 12,
    max_range: float = 1.0,
    *,
    body_frame: bool = True,
    angle_start: float = 0.0,
    include_agents: bool = True,
    obstacle_pos: torch.Tensor | None = None,  # [n_envs, n_obstacles, 2]
    obstacle_radius: torch.Tensor | None = None,  # [n_obstacles]
) -> torch.Tensor:
    """Warp-kernel equivalent of :func:`swarp.sensors.lidar.lidar_scan`.

    Same signature and identical ranges (to fp tolerance), but without the dense
    pairwise intermediate. Inference-only — not differentiable (launched with
    ``record_tape=False``). Callers needing gradients use the torch path.
    """
    wp.init()  # idempotent; ``from_torch`` needs the runtime, and callers may scan
    # before any other Warp API (Environment/World init it in normal use).
    n_envs, n_agents = pos.shape[0], pos.shape[1]
    wp_dtype = TORCH_DTYPE_TO_WP[str(pos.dtype)]
    vec2 = VEC2[wp_dtype]

    pos_wp = wp.from_torch(pos.contiguous(), dtype=vec2, requires_grad=False)
    theta_wp = wp.from_torch(theta.contiguous(), dtype=wp_dtype, requires_grad=False)
    radius_wp = wp.from_torch(agent_radius.contiguous(), dtype=wp_dtype, requires_grad=False)
    device = pos_wp.device

    has_obstacles = (
        obstacle_pos is not None and obstacle_radius is not None and obstacle_pos.shape[1] > 0
    )
    if has_obstacles:
        n_obstacles = obstacle_pos.shape[1]
        obs_pos_wp = wp.from_torch(obstacle_pos.contiguous(), dtype=vec2, requires_grad=False)
        obs_radius_wp = wp.from_torch(
            obstacle_radius.contiguous(), dtype=wp_dtype, requires_grad=False
        )
    else:
        n_obstacles = 0
        obs_pos_wp = wp.empty(shape=(n_envs, 0), dtype=vec2, device=device)
        obs_radius_wp = wp.empty(shape=0, dtype=wp_dtype, device=device)

    range_out = wp.empty(shape=(n_envs, n_agents, n_rays), dtype=wp_dtype, device=device)
    ray_step = 2.0 * math.pi / n_rays

    wp.launch(
        lidar_scan_kernel,
        dim=(n_envs, n_agents),
        inputs=[
            pos_wp,
            theta_wp,
            radius_wp,
            obs_pos_wp,
            obs_radius_wp,
            wp.int32(n_rays),
            wp_dtype(max_range),
            wp_dtype(ray_step),
            wp_dtype(angle_start),
            wp.int32(1 if body_frame else 0),
            wp.int32(1 if include_agents else 0),
            wp.int32(n_obstacles),
        ],
        outputs=[range_out],
        device=device,
        record_tape=False,
    )
    return wp.to_torch(range_out, requires_grad=False)
