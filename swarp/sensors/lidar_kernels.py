"""Warp-kernel lidar backend: a per-ray running minimum over every shape in the world.

The torch backend in :mod:`swarp.sensors.lidar` materializes dense
``[n_envs, n_agents, n_rays, n_targets]`` tensors, so its memory scales as
``O(E*A*R*T)``. Here one thread owns a single ``(env, agent, ray)`` and keeps a
thread-local running minimum over the targets, so only the ``[E, A, R]`` outputs are
stored (``O(E*A*R)``, flat in the target count). The ray is a *launch* dimension rather
than a loop, which is what lets a high ray count saturate the GPU.

Geometry, identical to the torch oracle (:func:`swarp.sensors.lidar.lidar_scan`):

* disc agent / circle obstacle — the closed-form ray-circle test (:func:`_ray_hit`);
* box agent / box obstacle — a slab test in the box frame (:func:`_ray_box`);
* segment obstacle — the exact capsule the collision step uses: the rectangle
  ``(half_len, radius)`` around the core plus a circle at each end (:func:`_ray_capsule`).

A ray whose origin lies inside a shape misses that shape. Targets are visited agents first
(by index), then obstacles (by shape index), then scenery slots, and a target replaces the
current best only when strictly nearer, so ties resolve to the first in that order.

*Scenery* (:class:`swarp.sensors.lidar.Scenery`) is geometry only the lidar sees: per-env
circles, boxes and segments in ``[n_envs, n_slots]`` arrays (pose *and* size per env, a
negative shape for an empty slot) that the stepper's contacts never read. A scenery hit is
:data:`HIT_IMMOVABLE` with hit index ``-1``.

Kernels are compiled per configuration (:func:`lidar_kernel`): agents on/off, obstacles
on/off, box agents on/off, which hit outputs are written, the rotating-scan sweep
(:func:`swarp.sensors.lidar.sweep_offsets`) and scenery on/off are ``wp.static`` flags, so
a disc-only, range-only scan runs exactly the arithmetic it always did. Inference-only
(``enable_backward=False``); gradients come from the torch path.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from swarp.core.config import ObstacleShape
from swarp.core.state import TORCH_DTYPE_TO_WP, VEC2
from swarp.dynamics.base import NUM_PARAMS, P_HALF_X, P_HALF_Y, P_RADIUS, P_SHAPE
from swarp.interop.autograd import torch_stream_scope

# Lean launches (no grid-stride loop); see docs/performance.md.
wp.set_module_options({"default_grid_stride": False})

_SHAPE_BOX = wp.constant(int(ObstacleShape.BOX))
_SHAPE_SEGMENT = wp.constant(int(ObstacleShape.SEGMENT))

#: Per-ray ``hit_class`` values.
HIT_NONE = 0
HIT_AGENT = 1
HIT_IMMOVABLE = 2  # an obstacle's class is HIT_IMMOVABLE + its ObstacleKind
HIT_MOVABLE = 3

#: Direction components smaller than this are replaced by ``±DIR_EPS`` in the slab test,
#: so ``1/d`` stays finite (and so do its gradients on the torch side).
DIR_EPS = 1.0e-12


def ray_step(n_rays: int, angle_start: float, angle_end: float | None) -> float:
    """Host-side angular spacing of the rays (the one place both backends take it from).

    A full circle (``angle_end`` unset, or a span of at least ``2*pi``) spaces ``n_rays``
    rays ``2*pi / n_rays`` apart, so the last ray stops one step short of the first. A
    partial field of view puts the first ray at ``angle_start`` and the last at
    ``angle_end``: ``(angle_end - angle_start) / (n_rays - 1)`` apart.
    """
    if angle_end is None:
        return 2.0 * math.pi / n_rays
    span = angle_end - angle_start
    if span < 0.0:
        raise ValueError(f"angle_end={angle_end} must not be below angle_start={angle_start}")
    if span >= 2.0 * math.pi:
        return 2.0 * math.pi / n_rays
    return span / (n_rays - 1) if n_rays > 1 else 0.0


@wp.func
def _ray_hit(oc: Any, d: Any, rad: Any, miss: Any):
    """Range along unit dir ``d`` to the circle at relative center ``oc`` (center -
    origin), or ``miss``. ``t = proj - sqrt(rad^2 - perp2)``, a hit iff
    ``perp2 <= rad^2`` and ``proj > 0`` and ``t > 0`` (an origin inside misses)."""
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


@wp.func
def _safe_dir(x: Any):
    """``x``, or ``±DIR_EPS`` (sign of ``x``, ``+`` at zero) when ``|x| < DIR_EPS``."""
    eps = type(x)(DIR_EPS)
    out = x
    if wp.abs(x) < eps:
        out = eps
        if x < type(x)(0.0):
            out = -eps
    return out


@wp.func
def _ray_box(rel: Any, d: Any, angle: Any, hx: Any, hy: Any, miss: Any):
    """Range along unit dir ``d`` from ``origin = center + rel`` to the box of
    half-extents ``(hx, hy)`` rotated by ``angle``, or ``miss``.

    Slab test in the box frame: the ray enters at ``tnear = max`` of the per-axis entry
    times and leaves at ``tfar = min`` of the exit times; a hit iff ``tnear <= tfar`` and
    ``tnear > 0``. An origin inside the box has ``tnear < 0`` and misses."""
    zero = type(hx)(0.0)
    ca = wp.cos(angle)
    sa = wp.sin(angle)
    ox = ca * rel[0] + sa * rel[1]
    oy = -sa * rel[0] + ca * rel[1]
    dx = _safe_dir(ca * d[0] + sa * d[1])
    dy = _safe_dir(-sa * d[0] + ca * d[1])
    t1x = (-hx - ox) / dx
    t2x = (hx - ox) / dx
    t1y = (-hy - oy) / dy
    t2y = (hy - oy) / dy
    tnear = wp.max(wp.min(t1x, t2x), wp.min(t1y, t2y))
    tfar = wp.min(wp.max(t1x, t2x), wp.max(t1y, t2y))
    out = miss
    if (tnear <= tfar) and (tnear > zero):
        out = tnear
    return out


@wp.func
def _ray_capsule(rel: Any, d: Any, angle: Any, half_len: Any, rad: Any, miss: Any):
    """Range along unit dir ``d`` from ``origin = center + rel`` to the capsule of core
    half-length ``half_len`` and radius ``rad`` oriented by ``angle``, or ``miss``.

    The capsule is the rectangle ``(half_len, rad)`` around its core united with a circle
    of radius ``rad`` at each end, which is exactly the set the collision step's
    ``closest_on_segment`` contact treats as solid. From outside, the first hit on a union
    is the nearest of the parts' first hits. An origin inside the capsule misses: tested
    directly, since an origin inside one end circle still sees the rectangle from
    outside."""
    u = type(rel)(wp.cos(angle), wp.sin(angle))
    s = wp.clamp(wp.dot(rel, u), -half_len, half_len)
    q = rel - s * u
    out = miss
    if wp.dot(q, q) >= rad * rad:
        out = _ray_box(rel, d, angle, half_len, rad, miss)
        out = wp.min(out, _ray_hit(half_len * u - rel, d, rad, miss))
        out = wp.min(out, _ray_hit(-half_len * u - rel, d, rad, miss))
    return out


def _make_kernel(
    agents: bool,
    obstacles: bool,
    box_agents: bool,
    emit_class: bool,
    emit_index: bool,
    sweep: bool = False,
    scenery: bool = False,
) -> wp.Kernel:
    """The scan kernel with the given features compiled in (see the module docstring)."""

    @wp.kernel(enable_backward=False, module="unique")
    def lidar_scan_kernel(
        pos: wp.array2d(dtype=Any),  # [E, A] vec2
        theta: wp.array2d(dtype=Any),  # [E, A]
        params: wp.array2d(dtype=Any),  # [A, NUM_PARAMS]: P_RADIUS, P_SHAPE, P_HALF_*
        obs_pos: wp.array2d(dtype=Any),  # [E, T] vec2, live pose
        obs_angle: wp.array2d(dtype=Any),  # [E, T], live pose
        obs_radius: wp.array(dtype=Any),  # [T]
        obs_half: wp.array(dtype=Any),  # [T] vec2
        obs_type: wp.array(dtype=wp.int32),  # [T] ObstacleShape
        obs_kind: wp.array(dtype=wp.int32),  # [T] ObstacleKind
        n_obstacles: wp.int32,
        max_range: Any,
        ray_step: Any,
        angle_start: Any,
        body_frame: wp.int32,
        vel: wp.array2d(dtype=Any),  # [E, A] vec2, read when sweep
        ang_vel: wp.array2d(dtype=Any),  # [E, A], read when sweep
        yaw_bias: wp.array2d(dtype=Any),  # [E, A], read when sweep
        stamp_phase: wp.array2d(dtype=Any),  # [E, A], read when sweep
        scan_period: Any,
        sweep_dir: Any,
        sc_pos: wp.array2d(dtype=Any),  # [E, S] vec2, read when scenery
        sc_angle: wp.array2d(dtype=Any),  # [E, S]
        sc_half: wp.array2d(dtype=Any),  # [E, S] vec2
        sc_radius: wp.array2d(dtype=Any),  # [E, S]
        sc_type: wp.array2d(dtype=wp.int32),  # [E, S] ObstacleShape, < 0 for an empty slot
        range_out: wp.array3d(dtype=Any),  # [E, A, R]
        class_out: wp.array3d(dtype=wp.uint8),  # [E, A, R] when emit_class
        index_out: wp.array3d(dtype=wp.int32),  # [E, A, R] when emit_index
    ):
        e, a, r = wp.tid()
        p = pos[e, a]
        n_agents = pos.shape[1]

        base = type(max_range)(0.0)
        if body_frame != 0:
            base = theta[e, a]
        # Same arithmetic as the torch offsets (angle_start + r * ray_step, added to the
        # heading), which is what keeps the two backends' ranges equal.
        ang = base + angle_start + type(max_range)(r) * ray_step
        if wp.static(sweep):
            # Cast from the pose at the time the rotating sensor took this ray (the model
            # of swarp.sensors.lidar.sweep_offsets), reported at the nominal angle.
            pi = type(max_range)(3.141592653589793)
            off = angle_start + type(max_range)(r) * ray_step
            x = sweep_dir * (off - pi) / (type(max_range)(2.0) * pi)
            tau = (stamp_phase[e, a] - (x - wp.floor(x))) * scan_period
            p = p - tau * vel[e, a]
            ang = ang + (yaw_bias[e, a] - tau * ang_vel[e, a])
        d = type(p)(wp.cos(ang), wp.sin(ang))
        best = max_range
        cls = wp.int32(0)
        idx = wp.int32(-1)
        if wp.static(agents):
            for b in range(n_agents):
                if b != a:
                    q = pos[e, b]
                    t = max_range
                    if wp.static(box_agents):
                        if params[b, P_SHAPE] > type(max_range)(0.5):
                            t = _ray_box(
                                p - q,
                                d,
                                theta[e, b],
                                params[b, P_HALF_X],
                                params[b, P_HALF_Y],
                                max_range,
                            )
                        else:
                            t = _ray_hit(q - p, d, params[b, P_RADIUS], max_range)
                    else:
                        t = _ray_hit(q - p, d, params[b, P_RADIUS], max_range)
                    if t < best:
                        best = t
                        cls = wp.int32(1)
                        idx = b
        if wp.static(obstacles):
            for o in range(n_obstacles):
                c = obs_pos[e, o]
                st = obs_type[o]
                t = max_range
                if st == _SHAPE_BOX:
                    h = obs_half[o]
                    t = _ray_box(p - c, d, obs_angle[e, o], h[0], h[1], max_range)
                elif st == _SHAPE_SEGMENT:
                    t = _ray_capsule(
                        p - c, d, obs_angle[e, o], obs_half[o][0], obs_radius[o], max_range
                    )
                else:
                    t = _ray_hit(c - p, d, obs_radius[o], max_range)
                if t < best:
                    best = t
                    cls = wp.int32(2) + obs_kind[o]
                    idx = o
        if wp.static(scenery):
            for s in range(sc_type.shape[1]):
                st = sc_type[e, s]
                if st >= 0:
                    c = sc_pos[e, s]
                    t = max_range
                    if st == _SHAPE_BOX:
                        h = sc_half[e, s]
                        t = _ray_box(p - c, d, sc_angle[e, s], h[0], h[1], max_range)
                    elif st == _SHAPE_SEGMENT:
                        t = _ray_capsule(
                            p - c, d, sc_angle[e, s], sc_half[e, s][0], sc_radius[e, s],
                            max_range,
                        )
                    else:
                        t = _ray_hit(c - p, d, sc_radius[e, s], max_range)
                    if t < best:
                        best = t
                        cls = wp.int32(2)
                        idx = wp.int32(-1)
        range_out[e, a, r] = best
        if wp.static(emit_class):
            class_out[e, a, r] = wp.uint8(cls)
        if wp.static(emit_index):
            index_out[e, a, r] = idx

    return lidar_scan_kernel


def _signature(dtype) -> list:
    vec2 = VEC2[dtype]
    return [
        wp.array2d(dtype=vec2),  # pos
        wp.array2d(dtype=dtype),  # theta
        wp.array2d(dtype=dtype),  # params
        wp.array2d(dtype=vec2),  # obs_pos
        wp.array2d(dtype=dtype),  # obs_angle
        wp.array(dtype=dtype),  # obs_radius
        wp.array(dtype=vec2),  # obs_half
        wp.array(dtype=wp.int32),  # obs_type
        wp.array(dtype=wp.int32),  # obs_kind
        wp.int32,  # n_obstacles
        dtype,  # max_range
        dtype,  # ray_step
        dtype,  # angle_start
        wp.int32,  # body_frame
        wp.array2d(dtype=vec2),  # vel
        wp.array2d(dtype=dtype),  # ang_vel
        wp.array2d(dtype=dtype),  # yaw_bias
        wp.array2d(dtype=dtype),  # stamp_phase
        dtype,  # scan_period
        dtype,  # sweep_dir
        wp.array2d(dtype=vec2),  # sc_pos
        wp.array2d(dtype=dtype),  # sc_angle
        wp.array2d(dtype=vec2),  # sc_half
        wp.array2d(dtype=dtype),  # sc_radius
        wp.array2d(dtype=wp.int32),  # sc_type
        wp.array3d(dtype=dtype),  # range_out
        wp.array3d(dtype=wp.uint8),  # class_out
        wp.array3d(dtype=wp.int32),  # index_out
    ]


#: (agents, obstacles, box_agents, emit_class, emit_index, sweep, scenery, dtype) ->
#: concrete kernel.
_KERNELS: dict[tuple, wp.Kernel] = {}


def lidar_kernel(
    dtype,
    agents: bool,
    obstacles: bool,
    box_agents: bool,
    emit_class: bool,
    emit_index: bool,
    sweep: bool = False,
    scenery: bool = False,
    *,
    device=None,
) -> wp.Kernel:
    """The concrete scan kernel for these flags and ``dtype``, built on first request.

    With ``device`` the kernel's module is also loaded there, together with its sibling
    that differs only in ``obstacles``: whether obstacles are scanned is decided per launch
    from the installed obstacle count, and a kernel first launched inside a graph capture
    cannot be loaded there. Without ``device`` the module loads lazily on first launch.
    """
    key = (agents, obstacles, box_agents, emit_class, emit_index, sweep, scenery, dtype)
    kernel = _KERNELS.get(key)
    if kernel is None or device is not None:
        for obs in (False, True):
            sib = (agents, obs, box_agents, emit_class, emit_index, sweep, scenery, dtype)
            k = _KERNELS.get(sib)
            if k is None:
                k = wp.overload(_make_kernel(*sib[:-1]), _signature(dtype))
                _KERNELS[sib] = k
            if device is not None:
                wp.load_module(k.module, device=device)
        kernel = _KERNELS[key]
    return kernel


def launch_lidar(
    *,
    pos: wp.array,
    theta: wp.array,
    params: wp.array,
    obs_pos: wp.array,
    obs_angle: wp.array,
    obs_radius: wp.array,
    obs_half: wp.array,
    obs_type: wp.array,
    obs_kind: wp.array,
    n_obstacles: int,
    n_rays: int,
    max_range: float,
    ray_step: float,
    angle_start: float,
    body_frame: bool,
    include_agents: bool,
    box_agents: bool,
    range_out: wp.array,
    class_out: wp.array,
    index_out: wp.array,
    emit_class: bool,
    emit_index: bool,
    sweep: tuple | None = None,
    scenery: tuple | None = None,
) -> None:
    """One scan launch: ``wp.launch`` and nothing else, so it is legal inside a capture.

    Every argument is a ready Warp array or a host scalar; the launch names its device and
    does not scope a stream (the caller decides that, see :meth:`Lidar.scan`). ``sweep``,
    when given, is ``(vel, ang_vel, yaw_bias, stamp_phase, scan_period, sweep_dir)`` (the
    model of :func:`swarp.sensors.lidar.sweep_offsets`); ``scenery``, when given, is the
    ``(pos, angle, half, radius, shape)`` ``[E, S]`` arrays of a
    :class:`~swarp.sensors.lidar.Scenery`. Without either the plain kernel runs, on
    stand-in arrays."""
    dtype = range_out.dtype
    n_envs, n_agents = pos.shape[0], pos.shape[1]
    kernel = lidar_kernel(
        dtype, include_agents, n_obstacles > 0, box_agents, emit_class, emit_index,
        sweep is not None, scenery is not None,
    )
    if sweep is None:
        sweep = (pos, theta, theta, theta, 0.0, 1.0)
    s_vel, s_w, s_yaw, s_phase, s_period, s_dir = sweep
    if scenery is None:
        scenery = (pos, theta, pos, theta, _no_scenery(range_out.device))
    wp.launch(
        kernel,
        dim=(n_envs, n_agents, n_rays),
        inputs=[
            pos,
            theta,
            params,
            obs_pos,
            obs_angle,
            obs_radius,
            obs_half,
            obs_type,
            obs_kind,
            wp.int32(n_obstacles),
            dtype(max_range),
            dtype(ray_step),
            dtype(angle_start),
            wp.int32(1 if body_frame else 0),
            s_vel,
            s_w,
            s_yaw,
            s_phase,
            dtype(s_period),
            dtype(s_dir),
            *scenery,
        ],
        outputs=[range_out, class_out, index_out],
        device=range_out.device,
        record_tape=False,
    )


_NO_SCENERY: dict[str, wp.array] = {}


def _no_scenery(device) -> wp.array:
    """A ``[1, 1]`` int32 stand-in for an unused scenery shape array, one per device."""
    key = str(device)
    arr = _NO_SCENERY.get(key)
    if arr is None:
        arr = wp.zeros((1, 1), dtype=wp.int32, device=device)
        _NO_SCENERY[key] = arr
    return arr


def _wp(t: torch.Tensor, dtype) -> wp.array:
    return wp.from_torch(t.detach().contiguous(), dtype=dtype, requires_grad=False)


def lidar_scan_warp(
    pos: torch.Tensor,  # [n_envs, n_agents, 2]
    theta: torch.Tensor,  # [n_envs, n_agents]
    agent_radius: torch.Tensor,  # [n_agents]
    n_rays: int = 12,
    max_range: float = 1.0,
    *,
    body_frame: bool = True,
    angle_start: float = 0.0,
    angle_end: float | None = None,
    include_agents: bool = True,
    agent_shape: torch.Tensor | None = None,  # [n_agents] AgentShape
    agent_half: torch.Tensor | None = None,  # [n_agents, 2]
    obstacle_pos: torch.Tensor | None = None,  # [n_envs, n_obstacles, 2]
    obstacle_radius: torch.Tensor | None = None,  # [n_obstacles]
    obstacle_shape: torch.Tensor | None = None,  # [n_obstacles] ObstacleShape
    obstacle_angle: torch.Tensor | None = None,  # [n_obstacles] or [n_envs, n_obstacles]
    obstacle_half: torch.Tensor | None = None,  # [n_obstacles, 2]
    obstacle_kind: torch.Tensor | None = None,  # [n_obstacles] ObstacleKind
    return_hits: bool = False,
    sweep: tuple | None = None,
    scenery: tuple | None = None,
):
    """Warp-kernel equivalent of :func:`swarp.sensors.lidar.lidar_scan`, from tensors.

    Same signature and the same results (to fp tolerance), freshly allocated. This is the
    standalone eager entry point (it wraps and packs its inputs on every call); a scenario
    scanning its own world every step uses :meth:`swarp.sensors.lidar.Lidar.bind` and
    :meth:`~swarp.sensors.lidar.Lidar.launch` instead, which do that once. Inference-only.
    """
    wp.init()  # idempotent; callers may scan before any other Warp API
    device = pos.device
    tdt = pos.dtype
    wdt = TORCH_DTYPE_TO_WP[str(tdt)]
    vec2 = VEC2[wdt]
    n_envs, n_agents = pos.shape[0], pos.shape[1]
    box_agents = agent_shape is not None
    params = torch.zeros(n_agents, NUM_PARAMS, device=device, dtype=tdt)
    params[:, P_RADIUS] = agent_radius
    if box_agents:
        params[:, P_SHAPE] = agent_shape.to(tdt)
        if agent_half is not None:
            params[:, P_HALF_X] = agent_half[:, 0]
            params[:, P_HALF_Y] = agent_half[:, 1]
    n_obs = 0
    if obstacle_pos is not None and obstacle_radius is not None:
        n_obs = obstacle_pos.shape[1]
    if n_obs > 0:

        def fill(t, shape, dt=tdt):
            return torch.zeros(shape, device=device, dtype=dt) if t is None else t.to(dt)

        o_angle = fill(obstacle_angle, (n_envs, n_obs))
        if o_angle.dim() == 1:
            o_angle = o_angle.unsqueeze(0).expand(n_envs, n_obs)
        obs = dict(
            obs_pos=_wp(obstacle_pos.to(tdt), vec2),
            obs_angle=_wp(o_angle, wdt),
            obs_radius=_wp(obstacle_radius.to(tdt), wdt),
            obs_half=_wp(fill(obstacle_half, (n_obs, 2)), vec2),
            obs_type=_wp(fill(obstacle_shape, (n_obs,), torch.int32), wp.int32),
            obs_kind=_wp(fill(obstacle_kind, (n_obs,), torch.int32), wp.int32),
        )
    else:
        obs = dict(
            obs_pos=wp.zeros((1, 1), dtype=vec2, device=str(device)),
            obs_angle=wp.zeros((1, 1), dtype=wdt, device=str(device)),
            obs_radius=wp.zeros(1, dtype=wdt, device=str(device)),
            obs_half=wp.zeros(1, dtype=vec2, device=str(device)),
            obs_type=wp.zeros(1, dtype=wp.int32, device=str(device)),
            obs_kind=wp.zeros(1, dtype=wp.int32, device=str(device)),
        )
    out_shape = (n_envs, n_agents, n_rays)
    ranges = torch.empty(out_shape, device=device, dtype=tdt)
    if return_hits:
        hit_class = torch.empty(out_shape, device=device, dtype=torch.uint8)
        hit_index = torch.empty(out_shape, device=device, dtype=torch.int32)
    else:
        hit_class = torch.empty((1, 1, 1), device=device, dtype=torch.uint8)
        hit_index = torch.empty((1, 1, 1), device=device, dtype=torch.int32)
    if sweep is not None:
        s_vel, s_w, s_yaw, s_phase, s_period, s_dir = sweep
        sweep = (
            _wp(s_vel.to(tdt), vec2), _wp(s_w.to(tdt), wdt), _wp(s_yaw.to(tdt), wdt),
            _wp(s_phase.to(tdt), wdt), float(s_period), float(s_dir),
        )
    if scenery is not None:
        sc_pos, sc_angle, sc_half, sc_radius, sc_shape = scenery
        scenery = (
            _wp(sc_pos.to(tdt), vec2), _wp(sc_angle.to(tdt), wdt), _wp(sc_half.to(tdt), vec2),
            _wp(sc_radius.to(tdt), wdt), _wp(sc_shape.to(torch.int32), wp.int32),
        )
    with torch_stream_scope(str(device)):
        launch_lidar(
            pos=_wp(pos, vec2),
            theta=_wp(theta, wdt),
            params=_wp(params, wdt),
            **obs,
            n_obstacles=n_obs,
            n_rays=n_rays,
            max_range=max_range,
            ray_step=ray_step(n_rays, angle_start, angle_end),
            angle_start=angle_start,
            body_frame=body_frame,
            include_agents=include_agents,
            box_agents=box_agents,
            range_out=_wp(ranges, wdt),
            class_out=_wp(hit_class, wp.uint8),
            index_out=_wp(hit_index, wp.int32),
            emit_class=return_hits,
            emit_index=return_hits,
            sweep=sweep,
            scenery=scenery,
        )
    if return_hits:
        return ranges, hit_class, hit_index
    return ranges
