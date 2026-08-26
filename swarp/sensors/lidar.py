"""Differentiable lidar: per-ray range readings against circular agents/obstacles.

The scan is a vectorized, fully differentiable torch computation over the world
state (the same layer scenarios build observations in), so gradients flow from a
policy back through the ranges to agent/obstacle positions. Geometry is analytic
ray-circle intersection: for a ray from ``o`` in unit direction ``d`` and a
circle centred at ``c`` with radius ``rad``,

    proj = (c - o) . d,   perp2 = |c - o|^2 - proj^2,
    hit  = proj > 0 and perp2 <= rad^2,
    range = proj - sqrt(rad^2 - perp2)   (first intersection, else max_range).

Only circular targets are supported (agents by their radius, circle obstacles by
theirs), because that closed form is the whole geometry back-end: there is no
ray-box or ray-segment test here. :meth:`Lidar.scan` therefore **excludes**
non-circular obstacles from the scan — it filters on
:class:`~swarp.core.config.ObstacleShape` before casting. They are invisible to the
sensor but still act in the collision step. Feeding a box or a segment through the
circle test instead of dropping it would report a phantom hit on a disc of the
obstacle's ``radius`` centred at its origin, which for a segment is its midpoint.

Rays are spaced uniformly over ``2*pi``; with ``body_frame=True`` they rotate with
each agent's heading ``theta``.
"""

from __future__ import annotations

import math

import torch

from swarp.core.config import ObstacleShape
from swarp.sensors.lidar_kernels import lidar_scan_warp


def lidar_scan(
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
    """Ranges ``[n_envs, n_agents, n_rays]`` in ``[0, max_range]``.

    A ray that hits nothing returns ``max_range``. Differentiable w.r.t. ``pos``
    and ``obstacle_pos``.
    """
    n_envs, n_agents = pos.shape[0], pos.shape[1]
    device, dtype = pos.device, pos.dtype

    base = theta if body_frame else torch.zeros_like(theta)
    offsets = angle_start + torch.arange(n_rays, device=device, dtype=dtype) * (
        2.0 * math.pi / n_rays
    )
    ray_ang = base.unsqueeze(-1) + offsets  # [E, A, R]
    dirs = torch.stack([torch.cos(ray_ang), torch.sin(ray_ang)], dim=-1)  # [E, A, R, 2]
    origin = pos.unsqueeze(2)  # [E, A, 1, 2]

    rng = pos.new_full((n_envs, n_agents, n_rays), max_range)

    def cast(centers, radii, drop_self: bool):
        # centers: [E, T, 2], radii: [T]  ->  updated ranges via ray-circle
        oc = centers[:, None, None, :, :] - origin.unsqueeze(3)  # [E, A, R, T, 2]
        d = dirs.unsqueeze(3)  # [E, A, R, 1, 2]
        proj = (oc * d).sum(-1)  # [E, A, R, T]
        perp2 = (oc * oc).sum(-1) - proj * proj
        rad2 = (radii * radii).view(1, 1, 1, -1)
        thc = torch.sqrt(torch.clamp(rad2 - perp2, min=0.0))
        t = proj - thc
        valid = (perp2 <= rad2) & (proj > 0.0) & (t > 0.0)
        if drop_self:
            eye = torch.eye(n_agents, device=device, dtype=torch.bool)  # [A, T=A]
            valid = valid & ~eye.view(1, n_agents, 1, n_agents)
        t = torch.where(valid, t, t.new_full((), max_range))
        return t.amin(dim=-1)  # [E, A, R]

    if include_agents:
        rng = torch.minimum(rng, cast(pos, agent_radius, drop_self=True))
    if obstacle_pos is not None and obstacle_radius is not None and obstacle_pos.shape[1] > 0:
        rng = torch.minimum(rng, cast(obstacle_pos, obstacle_radius, drop_self=False))
    return rng.clamp(max=max_range)


class Lidar:
    """Opt-in lidar observation component.

    Construct once, then call :meth:`scan` (or the instance) with a ``World`` to
    get ``[n_envs, n_agents, n_rays]`` ranges to concatenate into observations::

        lidar = Lidar(n_rays=12, max_range=1.0)
        obs = torch.cat([base_obs, lidar.scan(self.world)], dim=-1)

    Two interchangeable backends, selected by ``backend``:

    * ``"torch"`` (default) — the pure-torch broadcast :func:`lidar_scan`.
      Differentiable, but materializes a dense ``[E, A, R, T]`` family (``proj``,
      ``perp2``, ``thc``, ``t``, ``valid``), so memory grows with the target count
      ``T``.
    * ``"warp"`` — the Warp kernel :func:`~swarp.sensors.lidar_kernels.lidar_scan_warp`.
      Numerically equivalent ranges with no dense intermediate (flat memory in
      ``T``), which keeps high ray counts affordable. **Inference-only**: when
      gradients are required the scan transparently falls back to the torch path,
      so ``backend="warp"`` means "warp for inference, torch for grad."
    """

    def __init__(
        self,
        n_rays: int = 12,
        max_range: float = 1.0,
        *,
        body_frame: bool = True,
        angle_start: float = 0.0,
        include_agents: bool = True,
        include_obstacles: bool = True,
        backend: str = "torch",
    ) -> None:
        if n_rays < 1:
            raise ValueError("n_rays must be >= 1")
        if max_range <= 0.0:
            raise ValueError("max_range must be positive")
        if backend not in ("torch", "warp"):
            raise ValueError('backend must be "torch" or "warp"')
        self.n_rays = n_rays
        self.max_range = max_range
        self.body_frame = body_frame
        self.angle_start = angle_start
        self.include_agents = include_agents
        self.include_obstacles = include_obstacles
        self.backend = backend
        # Memoized circle filter, see :meth:`_obstacle_targets`. Keyed on the identity of
        # the installed shape/radius tensors, so it survives every scan against one
        # obstacle set and is rebuilt when a different set is installed.
        self._filter_key: tuple[int, int, int] | None = None
        self._filter_keep: torch.Tensor | None = None
        self._filter_radius: torch.Tensor | None = None

    def scan(self, world) -> torch.Tensor:
        """Ranges ``[n_envs, n_agents, n_rays]`` against the world's *live* geometry.

        Two things this does that the free :func:`lidar_scan` cannot, because they need
        the ``World``:

        * It reads the obstacle pose from
          :meth:`~swarp.core.world.World.obstacle_state_views`, not from
          ``world.obstacle_pos``. A movable obstacle is advanced in place inside the
          stepper's own arrays, so the tensor the scenario *installed* is the spawn pose
          and goes stale the moment the body is pushed.
        * It drops non-circular obstacles, since the geometry back-end is ray-circle only
          (see the module docstring). The filter lives here, above the backend split, so
          the torch and Warp paths see identical targets and stay in parity.
        """
        obs_pos, obs_rad = self._obstacle_targets(world) if self.include_obstacles else (None, None)
        pos = world.state.pos
        # The Warp backend is inference-only; fall back to the differentiable
        # torch path whenever a gradient is actually being tracked.
        grad = torch.is_grad_enabled() and (
            pos.requires_grad or (obs_pos is not None and obs_pos.requires_grad)
        )
        scan_fn = lidar_scan_warp if (self.backend == "warp" and not grad) else lidar_scan
        return scan_fn(
            pos,
            world.state.theta,
            world.agent_radius,
            n_rays=self.n_rays,
            max_range=self.max_range,
            body_frame=self.body_frame,
            angle_start=self.angle_start,
            include_agents=self.include_agents,
            obstacle_pos=obs_pos,
            obstacle_radius=obs_rad,
        )

    def _obstacle_targets(self, world) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Live ``(pos [E, C, 2], radius [C])`` for the circle obstacles only.

        ``world.obstacle_pos is None`` means no obstacle set is installed — the stepper
        still holds a dummy ``(1, 1)`` array, which is why the guard is on the world's
        mirror of the spec rather than on the view.

        The filter itself is **memoized**, because deciding it costs two device→host
        syncs (``is_circle.all()`` and ``nonzero``) and the answer only changes when a new
        obstacle set is installed. Which shapes exist is static per install — obstacle
        *poses* move, their shape tags do not — so the cache is keyed on the identity of
        the shape/radius tensors and the obstacle count. Without it a lidar scan stalls the
        pipeline twice per step, which is the whole point of keeping the loop on device.
        """
        if world.obstacle_pos is None or world.obstacle_radius is None:
            return None, None
        live_pos = world.obstacle_state_views()[0]
        shape = world.obstacle_shape
        if shape is None:
            return live_pos, world.obstacle_radius  # documented "all circles"
        radius = world.obstacle_radius
        key = (shape.data_ptr(), radius.data_ptr(), int(shape.shape[0]))
        if key != self._filter_key:
            is_circle = shape == int(ObstacleShape.CIRCLE)
            if bool(is_circle.all()):  # sync #1, once per install
                keep, kept_radius = None, radius
            else:
                keep = is_circle.nonzero(as_tuple=False).squeeze(-1)  # sync #2
                kept_radius = radius.index_select(0, keep)
            self._filter_key, self._filter_keep, self._filter_radius = key, keep, kept_radius
        if self._filter_keep is None:
            return live_pos, self._filter_radius
        # The one remaining per-scan op: an allocation, not a sync.
        return live_pos.index_select(1, self._filter_keep), self._filter_radius

    __call__ = scan
