"""Flocking: a Reynolds-style boids reward over within-radius neighbours.

Each agent is rewarded for cohesion (staying near the local neighbour centroid)
and alignment (matching the local mean velocity), and penalized for crowding
(separation) when neighbours come closer than ``separation_dist``. All terms are
differentiable torch ops over the neighbour features the World already exposes.
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from swarp.core.config import WorldConfig
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.scenarios.flocking_kernels import flocking_obs_reward_kernel
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario


class FlockingScenario(FusedScenario):
    def __init__(
        self,
        n_agents: int = 8,
        agent_radius: float = 0.05,
        world_size: float = 1.0,
        neighbor_obs: int = 4,
        neighbor_radius: float = 0.5,
        max_speed: float = 1.0,
        cohesion: float = 1.0,
        alignment: float = 1.0,
        separation: float = 2.0,
        separation_dist: float = 0.15,
    ) -> None:
        self.n_agents = n_agents
        self.agent_radius = agent_radius
        self.world_size = world_size
        self.neighbor_obs = neighbor_obs
        self.neighbor_radius = neighbor_radius
        self.max_speed = max_speed
        self.cohesion = cohesion
        self.alignment = alignment
        self.separation = separation
        self.separation_dist = separation_dist
        # Provisional obs width so ``obs_dim`` answers before ``make_world``: it mirrors
        # this scenario's own ``max_neighbors`` default. ``make_world`` re-derives it
        # from the *resolved* config, so a ``world_config`` override still wins.
        self._k_obs = min(self.neighbor_obs, min(32, max(4, self.n_agents)))

    def make_world(self, n_envs, device, dt, substeps, dtype, world_config=None) -> World:
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
        reach = 2.0 * self.agent_radius + margin
        cfg = WorldConfig(
            collisions=True,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            neighbor_radius=max(self.neighbor_radius, reach),
            max_neighbors=min(32, max(4, self.n_agents)),
        ).override_with(world_config)
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        self._cache: dict[str, torch.Tensor] | None = None
        # Fused-kernel obs width (capped at max_neighbors).
        self._k_obs = min(self.neighbor_obs, cfg.max_neighbors)
        return self.world

    @property
    def obs_dim(self) -> int:
        return 4 + 5 * self._k_obs

    def reset_world(
        self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False
    ) -> None:
        w = self.world
        n = w.n_envs
        lim = self.world_size - 2.0 * self.agent_radius
        spawn = w.sample_uniform((n, self.n_agents, 2), -lim, lim)
        vel = w.sample_uniform((n, self.n_agents, 2), -0.3 * self.max_speed, 0.3 * self.max_speed)
        w.write_state(env_mask, pos=spawn, vel=vel)
        self.finish_reset(env_mask, obs_only=obs_only)

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh()

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        self._refresh()

    # --------------------------------------------------------- fused fast path

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        """The degenerate spec: three framework-owned outputs and nothing else.

        No adopted state, no carry, no watched handle and no reset mask — so
        :meth:`~swarp.scenarios.fused.FusedScenario.fused_token` stays pinned at 0
        *structurally* (there is nothing for the resync loop to iterate) rather than by a
        hand-written ``return 0`` that a later edit could quietly invalidate.
        """
        ne, na = n_envs, self.n_agents
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("reward", (ne, na)),
            Buf("crowd", (ne, na)),
        )

    def launch_fused(self, pass_: FusedPass) -> None:
        """One kernel, always. A reset differs only in ``full_pass``: it recomputes
        observations on the new state without clobbering the reward/info a mid-step
        auto-reset has already returned. The launch rebuilds neighbors itself (mirroring
        the torch ``_refresh``), so no ``mark_pos_dirty`` bookkeeping is needed either."""
        self._launch(full_pass=pass_.full_pass)

    def _launch(self, full_pass: int) -> None:
        w = self.world
        n_envs = w.n_envs
        w.neighbors()  # build the grid on the current state
        grid = w.stepper.grid(n_envs)
        scalar = w.wp_dtype
        st = w.state_wp()
        wp.launch(
            flocking_obs_reward_kernel,
            dim=(n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                grid.neighbor_idx,
                grid.neighbor_count,
                wp.int32(self._k_obs),
                scalar(self.cohesion),
                scalar(self.alignment),
                scalar(self.separation),
                scalar(self.separation_dist),
                wp.int32(full_pass),
            ],
            outputs=[self._wp["obs"], self._wp["reward"], self._wp["crowd"]],
            device=w.device,
            record_tape=False,
        )

    def _refresh(self) -> None:
        w = self.world
        pos, vel = w.state.pos, w.state.vel
        idx, cnt = w.neighbors()
        idx = idx.long()
        k = idx.shape[-1]
        valid = torch.arange(k, device=w.device).view(1, 1, -1) < cnt.long().unsqueeze(-1)
        vf = valid.unsqueeze(-1).to(w.dtype)  # [n_envs, n_agents, k, 1]
        flat = idx.reshape(w.n_envs, -1)
        npos = torch.gather(pos, 1, flat.unsqueeze(-1).expand(-1, -1, 2)).view(
            w.n_envs, w.n_agents, k, 2
        )
        nvel = torch.gather(vel, 1, flat.unsqueeze(-1).expand(-1, -1, 2)).view(
            w.n_envs, w.n_agents, k, 2
        )
        rel_pos = (npos - pos.unsqueeze(2)) * vf
        rel_vel = (nvel - vel.unsqueeze(2)) * vf
        cnt_f = cnt.to(w.dtype).clamp(min=1.0).unsqueeze(-1)

        centroid_off = rel_pos.sum(dim=2) / cnt_f  # mean neighbour offset
        vel_off = rel_vel.sum(dim=2) / cnt_f  # mean neighbour velocity difference
        ndist = rel_pos.norm(dim=-1)  # [n_envs, n_agents, k]
        crowd = ((self.separation_dist - ndist).clamp(min=0.0) * valid.to(w.dtype)).sum(dim=-1)

        reward = (
            -self.cohesion * centroid_off.norm(dim=-1)
            - self.alignment * vel_off.norm(dim=-1)
            - self.separation * crowd
        )
        self._cache = {
            "reward": reward,
            "rel_pos": rel_pos[:, :, : self.neighbor_obs],
            "rel_vel": rel_vel[:, :, : self.neighbor_obs],
            "valid": valid[:, :, : self.neighbor_obs].to(w.dtype),
            "crowd": crowd,
        }

    def observations(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["obs"]
        w = self.world
        s = w.state
        c = self._cache
        k = c["rel_pos"].shape[2]
        return torch.cat(
            [
                s.pos,
                s.vel,
                c["rel_pos"].reshape(w.n_envs, w.n_agents, k * 2),
                c["rel_vel"].reshape(w.n_envs, w.n_agents, k * 2),
                c["valid"],
            ],
            dim=-1,
        )

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        return self._cache["reward"][:, agent_idx]

    def rewards(self) -> torch.Tensor:
        return self.fb["reward"] if self.fused_active else super().rewards()

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {"crowding": self.fb["crowd"]}
        return {"crowding": self._cache["crowd"]}
