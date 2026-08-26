"""Warp-kernel lidar backend: ray-circle scan with no dense pairwise intermediate.

The torch backend in :mod:`swarp.sensors.lidar` materializes a family of dense
``[n_envs, n_agents, n_rays, n_targets]`` tensors (``proj``, ``perp2``, ``thc``,
``t``, ``valid``), so its memory scales as ``O(E*A*R*T)``. This kernel computes
the identical ranges (same analytic ray-circle test) but one thread owns a single
``(env, agent, ray)`` and keeps a thread-local running-min over the targets — so
only the ``[E, A, R]`` output is stored (``O(E*A*R)``, flat in the target count).
This is the trick Isaac Sim uses to keep high ray counts affordable.

The ray is a *launch* dimension rather than a loop, which is what makes the 256-ray
case saturate a GPU: at ``E*A`` threads a 24-env, 4-agent scan launches 96 threads and
leaves the device idle, while ``E*A*R`` gives it 24576.

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
from swarp.interop.autograd import torch_stream_scope


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
    max_range: Any,
    ray_step: Any,  # 2*pi / n_rays, precomputed to match torch exactly
    angle_start: Any,
    body_frame: wp.int32,
    include_agents: wp.int32,
    n_obstacles: wp.int32,
    range_out: wp.array3d(dtype=Any),  # [E, A, R] scalar
):
    e, a, r = wp.tid()
    p = pos[e, a]
    n_agents = pos.shape[1]

    base = type(max_range)(0.0)
    if body_frame != 0:
        base = theta[e, a]

    # ``r`` comes from the launch grid, so the ray count is the third launch dimension
    # rather than a kernel argument. The arithmetic is unchanged from the loop it
    # replaces, which is what keeps ``test_warp_matches_torch`` exact.
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


# Single-slot wrap caches, one per kernel input, exactly as ``Stepper.wrap_actions``
# does for the action tensor: ``wp.from_torch`` is a handle allocation plus a few
# attribute reads, and a scan re-wraps the *same* live buffers on every step. Keyed on
# ``(data_ptr, shape, dtype)``, and the wrapped torch tensor is held alongside so its
# storage cannot be freed and recycled into a different tensor that would then hit the
# cache spuriously. Single-slot: two Lidars over different worlds just re-wrap, which is
# what the uncached path did anyway.
_WRAPS: dict[str, tuple[tuple, wp.array, torch.Tensor]] = {}
#: Reused output / empty-obstacle arrays, keyed by shape+dtype+device.
_BUFFERS: dict[tuple, wp.array] = {}


def _wrap(slot: str, t: torch.Tensor, dtype) -> wp.array:
    tc = t.contiguous()
    key = (tc.data_ptr(), tuple(tc.shape), dtype)
    cached = _WRAPS.get(slot)
    if cached is not None and cached[0] == key:
        return cached[1]
    arr = wp.from_torch(tc, dtype=dtype, requires_grad=False)
    _WRAPS[slot] = (key, arr, tc)
    return arr


def _buffer(shape: tuple[int, ...], dtype, device) -> wp.array:
    key = (shape, dtype, str(device))
    arr = _BUFFERS.get(key)
    if arr is None:
        arr = wp.empty(shape=shape, dtype=dtype, device=device)
        _BUFFERS[key] = arr
    return arr


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

    **The returned tensor is a view of a reused buffer**, unlike the torch backend's
    freshly-allocated one: the next scan of the same shape overwrites it. Consume it
    immediately (concatenating into an observation copies) or ``clone()`` to retain it.
    That is the same convention :meth:`swarp.core.environment.Environment.step` follows
    for its outputs, and it is what keeps a per-step scan allocation-free.

    Launches are scoped onto torch's current stream, so the scan is ordered against the
    torch ops that produced ``pos``/``theta`` under a user-created stream too.
    """
    wp.init()  # idempotent; ``from_torch`` needs the runtime, and callers may scan
    # before any other Warp API (Environment/World init it in normal use).
    n_envs, n_agents = pos.shape[0], pos.shape[1]
    wp_dtype = TORCH_DTYPE_TO_WP[str(pos.dtype)]
    vec2 = VEC2[wp_dtype]
    has_obstacles = (
        obstacle_pos is not None and obstacle_radius is not None and obstacle_pos.shape[1] > 0
    )
    n_obstacles = obstacle_pos.shape[1] if has_obstacles else 0
    ray_step = 2.0 * math.pi / n_rays

    # Everything Warp-side inside the scope: mempool allocations are stream-ordered too,
    # so the ``_buffer`` calls belong in here with the launch.
    with torch_stream_scope(str(pos.device)):
        pos_wp = _wrap("pos", pos, vec2)
        theta_wp = _wrap("theta", theta, wp_dtype)
        radius_wp = _wrap("radius", agent_radius, wp_dtype)
        device = pos_wp.device
        if has_obstacles:
            obs_pos_wp = _wrap("obs_pos", obstacle_pos, vec2)
            obs_radius_wp = _wrap("obs_radius", obstacle_radius, wp_dtype)
        else:
            obs_pos_wp = _buffer((n_envs, 0), vec2, device)
            obs_radius_wp = _buffer((0,), wp_dtype, device)
        range_out = _buffer((n_envs, n_agents, n_rays), wp_dtype, device)

        wp.launch(
            lidar_scan_kernel,
            dim=(n_envs, n_agents, n_rays),
            inputs=[
                pos_wp,
                theta_wp,
                radius_wp,
                obs_pos_wp,
                obs_radius_wp,
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
