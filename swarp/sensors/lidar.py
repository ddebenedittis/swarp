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
theirs); box/segment obstacles are ignored by the lidar for now (they still act
in the collision step). Rays are spaced uniformly over ``2*pi``; with
``body_frame=True`` they rotate with each agent's heading ``theta``.
"""

from __future__ import annotations

import math

import torch

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
      Differentiable, but materializes an ``[E, A, R, T, 2]`` intermediate, so
      memory grows with the target count ``T``.
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

    def scan(self, world) -> torch.Tensor:
        obs_pos = world.obstacle_pos if self.include_obstacles else None
        obs_rad = world.obstacle_radius if self.include_obstacles else None
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

    __call__ = scan
