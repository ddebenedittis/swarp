"""Sampling: agents collect an unknown scalar field, consuming cells they visit.

Port of the VMAS ``sampling`` scenario (no new physics). Each env carries a
batched sum-of-Gaussians density on a ``grid_res x grid_res`` grid over the
world square. An agent earns the field value at its current cell the first time
any agent enters that cell; the cell is then marked consumed. Observation is the
agent's own pose plus the field sampled at the 3x3 cell neighbourhood around it
(a local gradient cue), all differentiable w.r.t. positions.
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from wmas.core.config import WorldConfig
from wmas.core.state import VEC2
from wmas.core.world import World
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from wmas.scenarios.base import Scenario
from wmas.scenarios.sampling_kernels import (
    sampling_obs_reward_kernel,
    sampling_scatter_kernel,
)


class SamplingScenario(Scenario):
    def __init__(
        self,
        n_agents: int = 4,
        agent_radius: float = 0.05,
        world_size: float = 1.0,
        n_gaussians: int = 3,
        grid_res: int = 12,
        field_std: float = 0.15,
        max_speed: float = 1.0,
        collision_penalty: float = -0.1,
    ) -> None:
        self.n_agents = n_agents
        self.agent_radius = agent_radius
        self.world_size = world_size
        self.n_gaussians = n_gaussians
        self.grid_res = grid_res
        self.field_std = field_std
        self.max_speed = max_speed
        self.collision_penalty = collision_penalty

    def make_world(self, n_envs, device, dt, substeps, dtype) -> World:
        cfgs = [
            AgentConfig(
                model=DynamicsModel.HOLONOMIC,
                ctrl_mode=ControlMode.VELOCITY,
                radius=self.agent_radius,
                max_speed=self.max_speed,
                max_accel=2.0 * self.max_speed,
            )
            for _ in range(self.n_agents)
        ]
        margin = 0.5 * self.agent_radius
        cfg = WorldConfig(
            collisions=True,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            max_neighbors=min(8, max(2, self.n_agents)),
        )
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        self.centers: torch.Tensor | None = None  # [n_envs, n_gaussians, 2]
        self.consumed: torch.Tensor | None = None  # [n_envs, grid_res*grid_res] bool
        self._cache: dict[str, torch.Tensor] | None = None
        self._fused_ready = False
        # Cell-centre coordinates in world units, per axis: (i+0.5)/res * 2W - W.
        i = torch.arange(self.grid_res, device=device, dtype=dtype)
        self._cell_coord = (i + 0.5) / self.grid_res * 2.0 * self.world_size - self.world_size
        return self.world

    #: Sampling ships fused Warp obs/reward kernels (2D holonomic).
    fused_available = True

    @property
    def obs_dim(self) -> int:
        return 4 + 9  # pos, vel, 3x3 field samples

    # ------------------------------------------------------------------ field

    def _field(self, pts: torch.Tensor) -> torch.Tensor:
        """Density at ``pts`` [..., 2] given the per-env Gaussian centres.

        ``pts`` broadcasts against ``centers`` on the env axis (dim 0).
        """
        c = self.centers  # [n_envs, G, 2]
        # pts: [n_envs, N, 2] -> [n_envs, N, 1, 2] ; centers -> [n_envs, 1, G, 2]
        d2 = (pts.unsqueeze(-2) - c.unsqueeze(1)).square().sum(-1)  # [n_envs, N, G]
        return torch.exp(-d2 / (2.0 * self.field_std**2)).sum(-1)  # [n_envs, N]

    def _cell_of(self, pos: torch.Tensor) -> torch.Tensor:
        """Linear cell id [n_envs, n_agents] for positions [n_envs, n_agents, 2]."""
        u = (pos + self.world_size) / (2.0 * self.world_size)  # -> [0, 1]
        c = (u * self.grid_res).floor().long().clamp(0, self.grid_res - 1)
        return c[..., 1] * self.grid_res + c[..., 0]

    # ------------------------------------------------------------------ reset

    def reset_world(
        self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False
    ) -> None:
        w = self.world
        n = w.n_envs
        lim = self.world_size - 2.0 * self.agent_radius
        spawn = w.sample_uniform((n, self.n_agents, 2), -lim, lim)
        centers = w.sample_uniform((n, self.n_gaussians, 2), -lim, lim)
        consumed = torch.zeros(n, self.grid_res * self.grid_res, dtype=torch.bool, device=w.device)

        if self.centers is None:
            self.centers = torch.zeros(n, self.n_gaussians, 2, device=w.device, dtype=w.dtype)
            self.consumed = torch.zeros(
                n, self.grid_res * self.grid_res, dtype=torch.bool, device=w.device
            )

        if env_mask is None:
            w.state.pos.data.copy_(spawn)
            w.state.vel.data.zero_()
            self.centers.copy_(centers)
            self.consumed.copy_(consumed)
        else:
            m3 = env_mask.view(-1, 1, 1)
            w.state.pos.data.copy_(torch.where(m3, spawn, w.state.pos.data))
            w.state.vel.data.copy_(
                torch.where(m3, torch.zeros_like(w.state.vel.data), w.state.vel.data)
            )
            self.centers.copy_(torch.where(m3, centers, self.centers))
            self.consumed.copy_(torch.where(env_mask.view(-1, 1), consumed, self.consumed))
        if self.fused_active:
            self._ensure_fused(w.n_envs)
            # full_pass=0 on the mid-step auto-reset obs-only pass (don't clobber
            # the reward/field already returned this step); 1 on a standalone
            # reset. The scatter always runs (mirrors the reference _refresh,
            # which marks the current cells consumed on every reset too).
            self._launch_obs_reward(full_pass=0 if obs_only else 1)
            self._launch_scatter()
        else:
            self._refresh()

    # -------------------------------------------------------- per-step caching

    def post_step(self) -> None:
        if self.fused_active:
            # Same sequence the whole-step graph runs (keeps non-graph fused mode
            # and CPU eager-persistent bit-identical).
            self._pre_graph_step()
            self._graph_post_physics()
        else:
            self._refresh()

    # ----------------------------------------------------- whole-step graph

    def graph_capturable(self) -> bool:
        return True

    def graph_recapture_token(self) -> int:
        # centers/consumed are allocated once and updated in place; state/grid
        # buffers are stable — nothing re-wrapped per launch, so no recapture.
        return 0

    def _graph_warmup_carries(self) -> list[torch.Tensor]:
        # The consumed-cell latch is advanced in place by _launch_scatter.
        return [self.consumed]

    def _pre_graph_step(self) -> None:
        self._ensure_fused(self.world.n_envs)

    def _graph_post_physics(self) -> None:
        self._launch_obs_reward(full_pass=1)
        self._launch_scatter()

    # --------------------------------------------------------- fused fast path

    def _ensure_fused(self, n_envs: int) -> None:
        """Allocate the persistent fused output buffers + cached wp handles.

        ``centers``/``consumed`` are allocated by ``reset_world`` before this runs
        and only ever updated in place (``copy_``), so their data_ptr is stable
        and can be wrapped once."""
        if self._fused_ready:
            return
        w = self.world
        na, dev, dt = self.n_agents, w.device, w.dtype

        def z(*shape):
            return torch.zeros(*shape, device=dev, dtype=dt)

        self._f_obs = z(n_envs, na, self.obs_dim)
        self._f_reward = z(n_envs, na)
        self._f_field = z(n_envs, na)
        scalar = w.wp_dtype
        self._wp = {
            "obs": wp.from_torch(self._f_obs, dtype=scalar),
            "reward": wp.from_torch(self._f_reward, dtype=scalar),
            "field": wp.from_torch(self._f_field, dtype=scalar),
            "centers": wp.from_torch(self.centers, dtype=VEC2[scalar]),
            "consumed": wp.from_torch(self.consumed.view(torch.uint8), dtype=wp.uint8),
        }
        self._fused_ready = True

    def _state_wp(self):
        """(pos, vel) as Warp arrays for the fused kernels."""
        w = self.world
        vec2 = VEC2[w.wp_dtype]
        if w._persistent and not w._detached:
            s = w.runtime.state
            return s.pos, s.vel
        st = w.state
        return (
            wp.from_torch(st.pos.contiguous(), dtype=vec2),
            wp.from_torch(st.vel.contiguous(), dtype=vec2),
        )

    def _launch_obs_reward(self, full_pass: int) -> None:
        w = self.world
        self._ensure_fused(w.n_envs)
        scalar = w.wp_dtype
        pos, vel = self._state_wp()
        wp.launch(
            sampling_obs_reward_kernel,
            dim=(w.n_envs, self.n_agents),
            inputs=[
                pos,
                vel,
                self._wp["centers"],
                self._wp["consumed"],
                scalar(self.world_size),
                wp.int32(self.grid_res),
                wp.int32(self.n_gaussians),
                scalar(2.0 * self.field_std**2),
                wp.int32(full_pass),
            ],
            outputs=[self._wp["obs"], self._wp["reward"], self._wp["field"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_scatter(self) -> None:
        w = self.world
        scalar = w.wp_dtype
        pos, _ = self._state_wp()
        wp.launch(
            sampling_scatter_kernel,
            dim=(w.n_envs, self.n_agents),
            inputs=[pos, scalar(self.world_size), wp.int32(self.grid_res)],
            outputs=[self._wp["consumed"]],
            device=w.device,
            record_tape=False,
        )

    def _refresh(self) -> None:
        w = self.world
        pos = w.state.pos
        cell = self._cell_of(pos)  # [n_envs, n_agents]
        was_consumed = torch.gather(self.consumed, 1, cell)  # [n_envs, n_agents]
        fval = self._field(pos)  # [n_envs, n_agents]
        reward = fval * (~was_consumed).to(w.dtype)
        # Mark the visited cells consumed (idempotent when agents share a cell).
        self.consumed.scatter_(1, cell, torch.ones_like(was_consumed))

        # Local 3x3 field samples around each agent's cell (observation cue).
        cx = (cell % self.grid_res).clamp(1, self.grid_res - 2)
        cy = (cell // self.grid_res).clamp(1, self.grid_res - 2)
        offs = torch.tensor([-1, 0, 1], device=w.device)
        gx = (cx.unsqueeze(-1) + offs).clamp(0, self.grid_res - 1)  # [n_envs, n_agents, 3]
        gy = (cy.unsqueeze(-1) + offs).clamp(0, self.grid_res - 1)
        # cell-centre coordinates for the 3x3 block: [n_envs, n_agents, 9, 2]
        xs = self._cell_coord[gx]  # [n_envs, n_agents, 3]
        ys = self._cell_coord[gy]
        grid_pts = torch.stack(
            [
                xs.unsqueeze(-1).expand(-1, -1, -1, 3),
                ys.unsqueeze(-2).expand(-1, -1, 3, -1),
            ],
            dim=-1,
        ).reshape(w.n_envs, w.n_agents * 9, 2)
        samples = self._field(grid_pts).reshape(w.n_envs, w.n_agents, 9)

        self._cache = {"reward": reward, "field": fval, "samples": samples}

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        if self.fused_active:
            return self._f_obs
        w = self.world
        s = w.state
        return torch.cat([s.pos, s.vel, self._cache["samples"]], dim=-1)

    def observation(self, agent_idx: int) -> torch.Tensor:
        return self.observations()[:, agent_idx]

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        return self._cache["reward"][:, agent_idx]

    def rewards(self) -> torch.Tensor:
        if self.fused_active:
            return self._f_reward
        return super().rewards()

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {"field": self._f_field, "consumed_frac": self.consumed.float().mean(-1)}
        return {"field": self._cache["field"], "consumed_frac": self.consumed.float().mean(-1)}
