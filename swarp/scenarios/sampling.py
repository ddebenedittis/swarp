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

from swarp.core.config import WorldConfig
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario
from swarp.scenarios.sampling_kernels import (
    sampling_obs_reward_kernel,
    sampling_scatter_kernel,
)


class SamplingScenario(FusedScenario):
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
        # Task state, allocated here (n_envs is known) and only ever written in place, so
        # the fused spec can adopt it with alloc="never". Allocating it on first reset
        # instead made the fused path silently depend on reset running first.
        self.centers = torch.zeros(  # [n_envs, n_gaussians, 2]
            n_envs, self.n_gaussians, 2, device=device, dtype=dtype
        )
        self.consumed = torch.zeros(  # [n_envs, grid_res*grid_res]
            n_envs, self.grid_res * self.grid_res, dtype=torch.bool, device=device
        )
        self._cache: dict[str, torch.Tensor] | None = None
        # Cell-centre coordinates in world units, per axis: (i+0.5)/res * 2W - W.
        i = torch.arange(self.grid_res, device=device, dtype=dtype)
        self._cell_coord = (i + 0.5) / self.grid_res * 2.0 * self.world_size - self.world_size
        return self.world

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

        w.write_state(env_mask, pos=spawn, vel=0.0)
        if env_mask is None:
            self.centers.copy_(centers)
            self.consumed.zero_()
        else:
            self.centers.copy_(torch.where(env_mask.view(-1, 1, 1), centers, self.centers))
            self.consumed.copy_(
                torch.where(env_mask.view(-1, 1), torch.zeros_like(self.consumed), self.consumed)
            )
        self.finish_reset(env_mask, obs_only=obs_only)

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh()

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        self._refresh()

    # --------------------------------------------------------- fused fast path

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        """Three framework outputs plus two pieces of *adopted* scenario state.

        ``centers`` and ``consumed`` are the scenario's own task state, allocated in
        ``make_world`` and only ever written in place, so the spec declares
        ``alloc="never"``: the framework wraps them (``consumed`` reinterpreted from torch
        ``bool`` to Warp ``uint8``) and treats a ``None`` as an error rather than quietly
        allocating a second copy.
        """
        ne, na = n_envs, self.n_agents
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("reward", (ne, na)),
            Buf("field", (ne, na)),
            Buf("centers", (ne, self.n_gaussians, 2), "vec2", attr="centers", alloc="never"),
            Buf(
                "consumed",
                (ne, self.grid_res * self.grid_res),
                "bool",
                attr="consumed",
                alloc="never",
                carry=True,
            ),
        )

    def launch_fused(self, pass_: FusedPass) -> None:
        """Obs+reward, then the consume scatter.

        The scatter runs on a reset too, mirroring the torch ``_refresh``, which marks the
        cells the fresh spawn sits on as consumed. Only ``full_pass`` differs: a mid-step
        auto-reset must not clobber the reward/field already returned for the transition.
        """
        self._launch_obs_reward(full_pass=pass_.full_pass)
        self._launch_scatter()

    def _launch_obs_reward(self, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        st = w.state_wp()
        wp.launch(
            sampling_obs_reward_kernel,
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
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
        wp.launch(
            sampling_scatter_kernel,
            dim=(w.n_envs, self.n_agents),
            inputs=[w.state_wp().pos, scalar(self.world_size), wp.int32(self.grid_res)],
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
            return self.fb["obs"]
        w = self.world
        s = w.state
        return torch.cat([s.pos, s.vel, self._cache["samples"]], dim=-1)

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        return self._cache["reward"][:, agent_idx]

    def rewards(self) -> torch.Tensor:
        return self.fb["reward"] if self.fused_active else super().rewards()

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {"field": self.fb["field"], "consumed_frac": self.consumed.float().mean(-1)}
        return {"field": self._cache["field"], "consumed_frac": self.consumed.float().mean(-1)}
