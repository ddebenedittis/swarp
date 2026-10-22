"""Differentiable lidar: per-ray range (and what was hit) against every shape in the world.

The scan is a vectorized, fully differentiable torch computation over the world state
(the same layer scenarios build observations in), so gradients flow from a policy back
through the ranges to agent and obstacle poses. Every target is analytic:

* **disc agents and circle obstacles** — ray-circle in closed form: for a ray from ``o``
  in unit direction ``d`` and a circle centred at ``c`` with radius ``rad``::

      proj = (c - o) . d,   perp2 = |c - o|^2 - proj^2,
      hit  = proj > 0 and perp2 <= rad^2,   t = proj - sqrt(rad^2 - perp2)

* **box agents and box obstacles** — a slab test in the box frame (a box agent is
  oriented by its heading ``theta``, whatever ``body_frame`` says about the rays). Ray
  direction components below :data:`~swarp.sensors.lidar_kernels.DIR_EPS` are replaced by
  ``±DIR_EPS``, so the ``1/d`` terms and their gradients stay finite.
* **segment obstacles** — the exact capsule the collision step treats as solid: the
  rectangle ``(half_len, radius)`` around the core plus a circle of ``radius`` at each end.
  A segment of radius zero is a line, hit wherever a ray crosses it.

A ray whose origin lies inside a shape misses that shape (the circle rule ``t > 0``,
applied to every shape), an agent never sees itself, and a ray that hits nothing returns
``max_range``. With ``return_hits`` each ray also reports what it hit: a ``hit_class``
(:data:`~swarp.sensors.lidar_kernels.HIT_NONE` = 0, ``HIT_AGENT`` = 1, ``HIT_IMMOVABLE`` =
2, ``HIT_MOVABLE`` = 3) and a ``hit_index`` (the agent index, or the obstacle *shape*
index, ``-1`` for none; map a shape to its body through ``Obstacles.body``). Ties go to
agents before obstacles and to the lower index within each.

Rays start at ``angle_start``. Over the full circle they are ``2*pi / n_rays`` apart; with
``angle_end`` spanning less than ``2*pi`` the first ray is at ``angle_start``, the last at
``angle_end``, and they are ``fov / (n_rays - 1)`` apart. With ``body_frame=True`` they
rotate with each agent's heading.

A real lidar spins: its rays are taken one after the other over a scan period while the
robot moves, and all are reported against the one pose of the scan's stamp. ``sweep``
models that (and a mount yaw error) by casting each ray from the pose at its own time,
extrapolated at constant velocity; see :func:`sweep_offsets`.

``scenery`` (a :class:`Scenery`) adds geometry only the lidar sees: per-env circles, boxes
and segments that no contact ever touches, scanned after the obstacles and reported as
``HIT_IMMOVABLE`` with hit index ``-1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp

from swarp.core.config import ObstacleShape
from swarp.core.state import VEC2
from swarp.dynamics.base import AgentShape
from swarp.interop.autograd import torch_stream_scope
from swarp.sensors.lidar_kernels import (
    DIR_EPS,
    HIT_AGENT,
    HIT_IMMOVABLE,
    HIT_NONE,
    launch_lidar,
    lidar_kernel,
    lidar_scan_warp,
    ray_step,
)


def _ray_circle(oc, d, rad, miss: float):
    """Range to the circle at relative centre ``oc`` (centre - origin), ``miss`` if none.

    The square root takes a guarded argument, so a ray that grazes or misses gets a zero
    gradient instead of ``0 * inf``; the forward values are those of
    ``sqrt(clamp(rad^2 - perp2, 0))``."""
    proj = (oc * d).sum(-1)
    perp2 = (oc * oc).sum(-1) - proj * proj
    rad2 = rad * rad
    x = rad2 - perp2
    pos = x > 0.0
    thc = torch.where(pos, torch.sqrt(torch.where(pos, x, torch.ones_like(x))), 0.0)
    t = proj - thc
    valid = (perp2 <= rad2) & (proj > 0.0) & (t > 0.0)
    return torch.where(valid, t, miss)


def _safe_dir(x):
    eps = torch.full_like(x, DIR_EPS)
    return torch.where(x.abs() < DIR_EPS, torch.where(x < 0.0, -eps, eps), x)


def _ray_box(rel, d, ca, sa, hx, hy, miss: float):
    """Range from ``origin = centre + rel`` along ``d`` to the box ``(hx, hy)`` rotated by
    ``(cos, sin) = (ca, sa)``, ``miss`` if none. Same arithmetic as the Warp
    ``_ray_box``."""
    rx, ry = rel[..., 0], rel[..., 1]
    ox = ca * rx + sa * ry
    oy = -sa * rx + ca * ry
    dx = _safe_dir(ca * d[..., 0] + sa * d[..., 1])
    dy = _safe_dir(-sa * d[..., 0] + ca * d[..., 1])
    t1x = (-hx - ox) / dx
    t2x = (hx - ox) / dx
    t1y = (-hy - oy) / dy
    t2y = (hy - oy) / dy
    tnear = torch.maximum(torch.minimum(t1x, t2x), torch.minimum(t1y, t2y))
    tfar = torch.minimum(torch.maximum(t1x, t2x), torch.maximum(t1y, t2y))
    valid = (tnear <= tfar) & (tnear > 0.0)
    return torch.where(valid, tnear, miss)


def _ray_capsule(rel, d, ca, sa, half_len, rad, miss: float):
    """Range to the capsule (core half-length ``half_len``, radius ``rad``), ``miss`` if
    none or if the origin is inside it. Same construction as the Warp ``_ray_capsule``."""
    u = torch.stack([ca, sa], dim=-1)
    s = torch.minimum(torch.maximum((rel * u).sum(-1), -half_len), half_len)
    q = rel - s.unsqueeze(-1) * u
    outside = (q * q).sum(-1) >= rad * rad
    end = half_len.unsqueeze(-1) * u
    t = _ray_box(rel, d, ca, sa, half_len, rad, miss)
    t = torch.minimum(t, _ray_circle(end - rel, d, rad, miss))
    t = torch.minimum(t, _ray_circle(-end - rel, d, rad, miss))
    return torch.where(outside, t, miss)


@dataclass
class Scenery:
    """Per-env geometry that only the lidar sees: ``n_slots`` shapes per env, each with its
    own pose *and* size per env (the stepper's obstacles share their sizes across envs).

    The torch tensors ``pos [E, S, 2]``, ``angle [E, S]``, ``half [E, S, 2]`` (a box's
    half-extents; a segment's core half-length in ``half[..., 0]``), ``radius [E, S]`` (a
    circle's or a segment's radius) and ``shape [E, S]`` (int32
    :class:`~swarp.core.config.ObstacleShape`, negative for an empty slot) are zero-copy
    views of the Warp arrays ``wp_*``, which a capture-safe scan reads by pointer: write
    either side in place, never rebind them. No contact, collision or body pass reads
    them, so whatever is placed here must lie where nothing can reach it."""

    pos: torch.Tensor
    angle: torch.Tensor
    half: torch.Tensor
    radius: torch.Tensor
    shape: torch.Tensor
    wp_pos: wp.array
    wp_angle: wp.array
    wp_half: wp.array
    wp_radius: wp.array
    wp_shape: wp.array

    @classmethod
    def empty(cls, n_envs: int, n_slots: int, device, dtype=wp.float32) -> Scenery:
        """``n_slots`` empty slots per env (shape ``-1``), allocated once."""
        vec2 = VEC2[dtype]
        wa = {
            "pos": wp.zeros((n_envs, n_slots), dtype=vec2, device=device),
            "angle": wp.zeros((n_envs, n_slots), dtype=dtype, device=device),
            "half": wp.zeros((n_envs, n_slots), dtype=vec2, device=device),
            "radius": wp.zeros((n_envs, n_slots), dtype=dtype, device=device),
            "shape": wp.full((n_envs, n_slots), -1, dtype=wp.int32, device=device),
        }
        views = {k: wp.to_torch(v, requires_grad=False) for k, v in wa.items()}
        return cls(**views, **{f"wp_{k}": v for k, v in wa.items()})

    @property
    def n_slots(self) -> int:
        return self.shape.shape[1]

    def tensors(self) -> tuple:
        """``(pos, angle, half, radius, shape)``, the ``scenery`` argument of the scans."""
        return self.pos, self.angle, self.half, self.radius, self.shape

    def arrays(self) -> tuple:
        """The Warp arrays in the order :func:`~swarp.sensors.lidar_kernels.launch_lidar`
        takes them."""
        return self.wp_pos, self.wp_angle, self.wp_half, self.wp_radius, self.wp_shape


def sweep_offsets(
    offsets: torch.Tensor,  # [n_rays]
    vel: torch.Tensor,  # [n_envs, n_agents, 2]
    ang_vel: torch.Tensor,  # [n_envs, n_agents]
    yaw_bias: torch.Tensor,  # [n_envs, n_agents]
    stamp_phase: torch.Tensor,  # [n_envs, n_agents]
    scan_period: float,
    sweep_dir: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rotating scan's per-ray origin and angle shifts, ``([E, A, R, 2], [E, A, R])``.

    The sensor sweeps in ``sweep_dir`` (+1 counter-clockwise, -1 clockwise) once per
    ``scan_period``, starting and ending at the rear (``pi`` off the frame's axis), so ray
    ``r`` at offset ``o_r`` is taken at phase ``phi_r = frac(sweep_dir (o_r - pi) / 2 pi)``
    of the sweep. The scan is reported at ``stamp_phase`` of the sweep (1: its end, 0: its
    start), so ray ``r`` was taken ``tau_r = (stamp_phase - phi_r) scan_period`` before the
    reported pose. Cast from the pose then, at constant velocity: its origin is shifted by
    ``-tau_r vel`` and its angle by ``yaw_bias - tau_r ang_vel``, the mount's yaw error
    included. Other bodies are taken where they are now."""
    x = sweep_dir * (offsets - torch.pi) / (2.0 * torch.pi)
    tau = (stamp_phase.unsqueeze(-1) - (x - torch.floor(x))) * scan_period  # [E, A, R]
    shift = -tau.unsqueeze(-1) * vel.unsqueeze(2)
    return shift, yaw_bias.unsqueeze(-1) - tau * ang_vel.unsqueeze(-1)


def lidar_scan(
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
    """Ranges ``[n_envs, n_agents, n_rays]`` in ``[0, max_range]``; see the module docstring.

    ``agent_shape=None`` means every agent is a disc of ``agent_radius``;
    ``obstacle_shape=None`` means every obstacle is a circle of ``obstacle_radius``. Absent
    ``obstacle_angle``/``obstacle_half``/``obstacle_kind`` default to zero / zero /
    ``IMMOVABLE``. Differentiable with respect to ``pos``, ``theta`` (box agents, and the
    rays under ``body_frame``), ``obstacle_pos`` and ``obstacle_angle``.

    With ``return_hits=True`` returns ``(ranges, hit_class, hit_index)``: ``uint8`` and
    ``int32`` tensors of the same shape. No host synchronization either way.

    ``sweep = (vel, ang_vel, yaw_bias, stamp_phase, scan_period, sweep_dir)`` casts every
    ray from the pose at the time the rotating sensor took it (:func:`sweep_offsets`).
    ``scenery = (pos, angle, half, radius, shape)``, ``[n_envs, n_slots]`` each (see
    :class:`Scenery`), is scanned after the obstacles; an empty slot (negative shape) is
    never hit.
    """
    n_envs, n_agents = pos.shape[0], pos.shape[1]
    device, dtype = pos.device, pos.dtype

    base = theta if body_frame else torch.zeros_like(theta)
    offsets = angle_start + torch.arange(n_rays, device=device, dtype=dtype) * ray_step(
        n_rays, angle_start, angle_end
    )
    ray_ang = base.unsqueeze(-1) + offsets  # [E, A, R]
    origin = pos[:, :, None, None, :]  # [E, A, 1, 1, 2]
    if sweep is not None:
        shift, dang = sweep_offsets(offsets, *sweep)
        ray_ang = ray_ang + dang
        origin = (pos.unsqueeze(2) + shift).unsqueeze(3)  # [E, A, R, 1, 2]
    dirs = torch.stack([torch.cos(ray_ang), torch.sin(ray_ang)], dim=-1).unsqueeze(3)
    miss = float(max_range)

    rng = pos.new_full((n_envs, n_agents, n_rays), max_range)
    none_idx = torch.full((n_envs, n_agents, n_rays), -1, device=device, dtype=torch.long)
    best_ag, idx_ag = rng, none_idx

    if include_agents and n_agents > 1:
        centres = pos[:, None, None, :, :]  # [E, 1, 1, A, 2]
        rad = agent_radius.view(1, 1, 1, -1)
        t = _ray_circle(centres - origin, dirs, rad, miss)  # [E, A, R, A]
        if agent_shape is not None:
            ang = theta[:, None, None, :]
            half = (
                agent_half
                if agent_half is not None
                else torch.zeros(n_agents, 2, device=device, dtype=dtype)
            )
            t_box = _ray_box(
                origin - centres,
                dirs,
                torch.cos(ang),
                torch.sin(ang),
                half[:, 0].view(1, 1, 1, -1),
                half[:, 1].view(1, 1, 1, -1),
                miss,
            )
            is_box = (agent_shape == int(AgentShape.BOX)).view(1, 1, 1, -1)
            t = torch.where(is_box, t_box, t)
        eye = torch.eye(n_agents, device=device, dtype=torch.bool)  # [A, T=A]
        t = torch.where(eye.view(1, n_agents, 1, n_agents), miss, t)
        best_ag, idx_ag = t.min(dim=-1)
        rng = torch.minimum(rng, best_ag)

    has_obstacles = (
        obstacle_pos is not None and obstacle_radius is not None and obstacle_pos.shape[1] > 0
    )
    best_ob, idx_ob = None, None
    if has_obstacles:
        n_obs = obstacle_pos.shape[1]
        centres = obstacle_pos[:, None, None, :, :]  # [E, 1, 1, T, 2]
        rad = obstacle_radius.view(1, 1, 1, -1)
        t = _ray_circle(centres - origin, dirs, rad, miss)  # [E, A, R, T]
        if obstacle_shape is not None:
            ang = (
                torch.zeros(n_envs, n_obs, device=device, dtype=dtype)
                if obstacle_angle is None
                else obstacle_angle.expand(n_envs, n_obs)
            )[:, None, None, :]
            ca, sa = torch.cos(ang), torch.sin(ang)
            half = (
                obstacle_half
                if obstacle_half is not None
                else torch.zeros(n_obs, 2, device=device, dtype=dtype)
            )
            hx = half[:, 0].view(1, 1, 1, -1)
            hy = half[:, 1].view(1, 1, 1, -1)
            rel = origin - centres
            shape = obstacle_shape.view(1, 1, 1, -1)
            t_box = _ray_box(rel, dirs, ca, sa, hx, hy, miss)
            t_cap = _ray_capsule(rel, dirs, ca, sa, hx, rad, miss)
            t = torch.where(
                shape == int(ObstacleShape.BOX),
                t_box,
                torch.where(shape == int(ObstacleShape.SEGMENT), t_cap, t),
            )
        best_ob, idx_ob = t.min(dim=-1)
        rng = torch.minimum(rng, best_ob)

    sc_wins = None
    if scenery is not None and scenery[0].shape[1] > 0:
        sc_pos, sc_angle, sc_half, sc_radius, sc_shape = scenery
        centres = sc_pos[:, None, None, :, :]  # [E, 1, 1, S, 2]
        ang = sc_angle[:, None, None, :]
        ca, sa = torch.cos(ang), torch.sin(ang)
        rad = sc_radius[:, None, None, :]
        hx = sc_half[:, None, None, :, 0]
        hy = sc_half[:, None, None, :, 1]
        rel = origin - centres
        shape = sc_shape[:, None, None, :]
        t = torch.where(
            shape == int(ObstacleShape.BOX),
            _ray_box(rel, dirs, ca, sa, hx, hy, miss),
            torch.where(
                shape == int(ObstacleShape.SEGMENT),
                _ray_capsule(rel, dirs, ca, sa, hx, rad, miss),
                _ray_circle(centres - origin, dirs, rad, miss),
            ),
        )
        t = torch.where(shape >= 0, t, miss)
        best_sc = t.min(dim=-1).values
        sc_wins = best_sc < rng  # strict: a tie goes to an agent or an obstacle
        rng = torch.minimum(rng, best_sc)

    rng = rng.clamp(max=max_range)
    if not return_hits:
        return rng

    hit_class = torch.where(rng < max_range, HIT_AGENT, HIT_NONE)
    hit_index = torch.where(rng < max_range, idx_ag, none_idx)
    if best_ob is not None:
        obs_wins = best_ob < best_ag  # strict: a tie goes to the agent
        if obstacle_kind is None:
            cls_ob = torch.full_like(idx_ob, HIT_IMMOVABLE)
        else:
            kind = obstacle_kind.to(torch.long)
            cls_ob = HIT_IMMOVABLE + kind.index_select(0, idx_ob.reshape(-1)).view_as(idx_ob)
        hit_class = torch.where(obs_wins, cls_ob, hit_class)
        hit_index = torch.where(obs_wins, idx_ob, hit_index)
    if sc_wins is not None:
        hit_class = torch.where(sc_wins, HIT_IMMOVABLE, hit_class)
        hit_index = torch.where(sc_wins, -1, hit_index)
    return rng, hit_class.to(torch.uint8), hit_index.to(torch.int32)


class Lidar:
    """Opt-in lidar observation component.

    Construct once, then call :meth:`scan` (or the instance) with a ``World`` to get
    ``[n_envs, n_agents, n_rays]`` ranges to concatenate into observations::

        lidar = Lidar(n_rays=12, max_range=1.0)
        obs = torch.cat([base_obs, lidar.scan(self.world)], dim=-1)

    With ``return_hits=True`` a scan returns ``(ranges, hit_class, hit_index)`` instead
    (see :func:`lidar_scan`). ``angle_end`` narrows the field of view. ``scan_period > 0``
    turns on the rotating-scan sweep (:func:`sweep_offsets`, in ``sweep_dir``): each
    agent's mount yaw error and stamp phase are then the ``[n_envs, n_agents]`` buffers
    :attr:`yaw_bias` and :attr:`stamp_phase` (zero until written), allocated by
    :meth:`bind`, and the velocities are read from the world state.

    Two interchangeable backends, selected by ``backend``:

    * ``"torch"`` (default) — the pure-torch broadcast :func:`lidar_scan`.
      Differentiable, but materializes a dense ``[E, A, R, T]`` family per shape, so
      memory grows with the target count ``T``.
    * ``"warp"`` — the Warp kernel of :mod:`swarp.sensors.lidar_kernels`. The same results
      with no dense intermediate (flat memory in ``T``), which keeps high ray counts
      affordable. **Inference-only**: when gradients are required the scan falls back to
      the torch path, so ``backend="warp"`` means "warp for inference, torch for grad".
      It returns **views of buffers this Lidar owns**, overwritten by its next scan:
      concatenating into an observation copies, ``clone()`` to keep them.

    The Warp path is also the capture-safe one. :meth:`bind` does every allocation, wrap
    and module load once, outside any capture; :meth:`launch` is then a single
    ``wp.launch`` against the live engine arrays, legal inside ``wp.ScopedCapture``. See
    :meth:`launch` for how a fused scenario uses the pair.
    """

    def __init__(
        self,
        n_rays: int = 12,
        max_range: float = 1.0,
        *,
        body_frame: bool = True,
        angle_start: float = 0.0,
        angle_end: float | None = None,
        include_agents: bool = True,
        include_obstacles: bool = True,
        return_hits: bool = False,
        backend: str = "torch",
        scan_period: float = 0.0,
        sweep_dir: float = 1.0,
    ) -> None:
        if n_rays < 1:
            raise ValueError("n_rays must be >= 1")
        if scan_period < 0.0 or sweep_dir not in (1.0, -1.0):
            raise ValueError("scan_period must be >= 0 and sweep_dir +1 or -1")
        if max_range <= 0.0:
            raise ValueError("max_range must be positive")
        if backend not in ("torch", "warp"):
            raise ValueError('backend must be "torch" or "warp"')
        self.n_rays = n_rays
        self.max_range = max_range
        self.body_frame = body_frame
        self.angle_start = angle_start
        self.angle_end = angle_end
        #: Angular spacing of the rays, decided on the host once (validates ``angle_end``).
        self.ray_step = ray_step(n_rays, angle_start, angle_end)
        self.include_agents = include_agents
        self.include_obstacles = include_obstacles
        self.return_hits = return_hits
        self.backend = backend
        self.scan_period = float(scan_period)
        self.sweep_dir = float(sweep_dir)
        #: Whether the rays are cast from the pose at their own sweep time.
        self.sweep = self.scan_period > 0.0
        self.yaw_bias: torch.Tensor | None = None
        self.stamp_phase: torch.Tensor | None = None
        self.wp_yaw_bias = None
        self.wp_stamp_phase = None
        # Warp-side binding, see bind(). ``version`` is bumped whenever the output buffers
        # are reallocated, so a scenario can fold it into its recapture token.
        self._bound_key: tuple | None = None
        self.version = 0
        self.ranges: torch.Tensor | None = None
        self.hit_class: torch.Tensor | None = None
        self.hit_index: torch.Tensor | None = None
        self.wp_ranges = None
        self.wp_hit_class = None
        self.wp_hit_index = None
        self._box_agents = False
        #: Lidar-only geometry (:class:`Scenery`), set by :meth:`bind`.
        self.scenery: Scenery | None = None

    def ray_offsets(self, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """``[n_rays]`` ray angles relative to the frame (heading or world x axis)."""
        idx = torch.arange(self.n_rays, device=device, dtype=dtype)
        return self.angle_start + idx * self.ray_step

    # ------------------------------------------------------------------ eager

    def scan(self, world):
        """Scan ``world``'s *live* geometry; see the class docstring for the return value.

        Obstacle poses come from the engine's own arrays
        (:meth:`~swarp.core.world.World.obstacle_state_views` on the torch path, the
        stepper's ``obs_*`` arrays on the Warp path), not from ``world.obstacle_pos``: a
        movable obstacle is advanced in place inside those arrays, so the pose the
        scenario installed goes stale the moment the body is pushed. Every shape is
        scanned; nothing here reads the device back to the host.
        """
        st = world.state
        grad = torch.is_grad_enabled() and (st.pos.requires_grad or st.theta.requires_grad)
        if self.backend == "warp" and not grad:
            self.bind(world)
            with torch_stream_scope(world.device):
                self.launch(world)
            if self.return_hits:
                return self.ranges, self.hit_class, self.hit_index
            return self.ranges
        return self._scan_torch(world)

    def _scan_torch(self, world):
        sp = world.stepper
        kw: dict = {}
        if self.include_obstacles and sp.n_obstacles > 0:
            live_pos, live_angle = world.obstacle_state_views()[:2]
            kw = dict(
                obstacle_pos=live_pos,
                obstacle_radius=world.obstacle_radius,
                obstacle_shape=world.obstacle_shape,
                obstacle_angle=live_angle,
                obstacle_half=world.obstacle_half_extents,
                obstacle_kind=world.obstacle_kind,
            )
        if sp.has_box_agents:
            kw.update(agent_shape=world.agent_shape, agent_half=world.agent_half_extents)
        if self.sweep:
            st = world.state
            zero = torch.zeros_like(st.theta)
            yaw = zero if self.yaw_bias is None else self.yaw_bias
            phase = zero if self.stamp_phase is None else self.stamp_phase
            kw["sweep"] = (st.vel, st.ang_vel, yaw, phase, self.scan_period, self.sweep_dir)
        if self.scenery is not None:
            kw["scenery"] = self.scenery.tensors()
        return lidar_scan(
            world.state.pos,
            world.state.theta,
            world.agent_radius,
            n_rays=self.n_rays,
            max_range=self.max_range,
            body_frame=self.body_frame,
            angle_start=self.angle_start,
            angle_end=self.angle_end,
            include_agents=self.include_agents,
            return_hits=self.return_hits,
            **kw,
        )

    # --------------------------------------------------------- capture-safe

    def bind(self, world, scenery: Scenery | None = None) -> None:
        """Allocate the output buffers for ``world`` and load the scan kernels.

        Call it **outside** any capture (in ``make_world``, or before the first step): it
        allocates ``[n_envs, n_agents, n_rays]`` output arrays, wraps them as zero-copy
        torch views (:attr:`ranges`, :attr:`hit_class`, :attr:`hit_index`; the Warp handles
        are :attr:`wp_ranges`, :attr:`wp_hit_class`, :attr:`wp_hit_index`) and loads the
        kernel modules on the world's device, because a kernel first launched inside a
        capture cannot be loaded there. Idempotent: re-binding the same world with the same
        batch size keeps the buffers (and their pointers); anything else reallocates and
        bumps :attr:`version`.

        ``scenery`` (a :class:`Scenery` of the world's batch size) is scanned by every
        later scan on both backends, read by pointer.
        """
        if scenery is not None and scenery.shape.shape[0] != world.n_envs:
            raise ValueError(
                f"scenery holds {scenery.shape.shape[0]} envs, the world {world.n_envs}"
            )
        key = (id(world), world.n_envs, world.n_agents, world.dtype, world.device,
               None if scenery is None else id(scenery))
        if self._bound_key == key:
            return
        self.scenery = scenery
        shape = (world.n_envs, world.n_agents, self.n_rays)
        hit_shape = shape if self.return_hits else (1, 1, 1)
        dev = world.device
        self.wp_ranges = wp.zeros(shape, dtype=world.wp_dtype, device=dev)
        self.wp_hit_class = wp.zeros(hit_shape, dtype=wp.uint8, device=dev)
        self.wp_hit_index = wp.zeros(hit_shape, dtype=wp.int32, device=dev)
        if self.sweep:
            self.wp_yaw_bias = wp.zeros(shape[:2], dtype=world.wp_dtype, device=dev)
            self.wp_stamp_phase = wp.zeros(shape[:2], dtype=world.wp_dtype, device=dev)
            self.yaw_bias = wp.to_torch(self.wp_yaw_bias, requires_grad=False)
            self.stamp_phase = wp.to_torch(self.wp_stamp_phase, requires_grad=False)
        self.ranges = wp.to_torch(self.wp_ranges, requires_grad=False)
        if self.return_hits:
            self.hit_class = wp.to_torch(self.wp_hit_class, requires_grad=False)
            self.hit_index = wp.to_torch(self.wp_hit_index, requires_grad=False)
        self._box_agents = world.stepper.has_box_agents
        lidar_kernel(
            world.wp_dtype,
            self.include_agents,
            True,
            self._box_agents,
            self.return_hits,
            self.return_hits,
            self.sweep,
            scenery is not None,
            device=dev,
        )
        self._bound_key = key
        self.version += 1

    def launch(self, world, state=None) -> None:
        """Scan into the bound buffers: one ``wp.launch``, legal inside a graph capture.

        Reads the agent pose from ``state`` (anything with Warp ``pos``/``theta`` arrays,
        and ``vel``/``ang_vel`` with the sweep on) or, by default,
        :meth:`~swarp.core.world.World.state_wp`, which in persistent mode returns the
        runtime's own pointer-stable arrays. Obstacles are read from the
        stepper's live ``obs_*`` arrays at call time, and the agent footprint from its
        shared parameter matrix (a per-env parameter override is not seen). Nothing is
        allocated, wrapped or read back, and no stream is opened: inside a fused
        scenario's ``launch_fused`` the launch lands on the capture stream like every other
        kernel there, and an eager caller scopes it (:meth:`scan` does).

        A fused scenario binds in ``make_world`` (or before its first step) and then, in
        ``launch_fused``, calls ``self.lidar.launch(self.world)`` ahead of the observation
        kernel that reads ``self.lidar.wp_ranges`` (and ``wp_hit_class`` / ``wp_hit_index``).
        Those handles are pointer-stable for as long as :attr:`version` is unchanged.
        """
        if self._bound_key is None or self._bound_key[0] != id(world):
            raise RuntimeError("Lidar.launch needs Lidar.bind(world) first, outside capture")
        st = state if state is not None else world.state_wp()
        sp = world.stepper
        launch_lidar(
            pos=st.pos,
            theta=st.theta,
            params=sp.params.floats,
            obs_pos=sp.obs_pos,
            obs_angle=sp.obs_angle,
            obs_radius=sp.obs_radius,
            obs_half=sp.obs_half,
            obs_type=sp.obs_type,
            obs_kind=sp.obs_kind,
            n_obstacles=sp.n_obstacles if self.include_obstacles else 0,
            n_rays=self.n_rays,
            max_range=self.max_range,
            ray_step=self.ray_step,
            angle_start=self.angle_start,
            body_frame=self.body_frame,
            include_agents=self.include_agents,
            box_agents=self._box_agents,
            range_out=self.wp_ranges,
            class_out=self.wp_hit_class,
            index_out=self.wp_hit_index,
            emit_class=self.return_hits,
            emit_index=self.return_hits,
            sweep=(
                (st.vel, st.ang_vel, self.wp_yaw_bias, self.wp_stamp_phase, self.scan_period,
                 self.sweep_dir)
                if self.sweep
                else None
            ),
            scenery=None if self.scenery is None else self.scenery.arrays(),
        )

    __call__ = scan


__all__ = ["Lidar", "Scenery", "lidar_scan", "lidar_scan_warp", "sweep_offsets"]
