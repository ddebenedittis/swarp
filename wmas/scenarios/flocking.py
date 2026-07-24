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

from wmas.core.config import WorldConfig
from wmas.core.state import VEC2
from wmas.core.world import World
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from wmas.scenarios.base import Scenario
from wmas.scenarios.flocking_kernels import flocking_obs_reward_kernel


class FlockingScenario(Scenario):
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
        reach = 2.0 * self.agent_radius + margin
        cfg = WorldConfig(
            collisions=True,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            neighbor_radius=max(self.neighbor_radius, reach),
            max_neighbors=min(32, max(4, self.n_agents)),
        )
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        self._cache: dict[str, torch.Tensor] | None = None
        # Fused-kernel obs width (capped at max_neighbors) and lazy-alloc flag.
        self._k_obs = min(self.neighbor_obs, cfg.max_neighbors)
        self._fused_ready = False
        return self.world

    def fused_available(self) -> bool:
        """Flocking ships a fused Warp obs/reward kernel (2D, needs neighbors)."""
        return True

    @property
    def obs_dim(self) -> int:
        return 4 + 5 * self._k_obs

    def reset_world(self, env_mask: torch.Tensor | None = None) -> None:
        w = self.world
        n = w.n_envs
        lim = self.world_size - 2.0 * self.agent_radius
        spawn = w.sample_uniform((n, self.n_agents, 2), -lim, lim)
        vel = w.sample_uniform((n, self.n_agents, 2), -0.3 * self.max_speed, 0.3 * self.max_speed)
        if env_mask is None:
            w.state.pos.data.copy_(spawn)
            w.state.vel.data.copy_(vel)
        else:
            m3 = env_mask.view(-1, 1, 1)
            w.state.pos.data.copy_(torch.where(m3, spawn, w.state.pos.data))
            w.state.vel.data.copy_(torch.where(m3, vel, w.state.vel.data))
        if self._fused_active:
            # full_pass=0 on the mid-step auto-reset obs-only pass (don't clobber
            # the reward/info already returned this step); 1 on a standalone reset.
            # _launch rebuilds neighbors on the new state, mirroring the torch
            # _refresh (which also calls w.neighbors()); no mark_pos_dirty needed.
            self._launch(full_pass=0 if self._fused_obs_only else 1)
        else:
            self._refresh()

    def post_step(self) -> None:
        if self._fused_active:
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
        # No per-launch re-wrapped handles and no persistent carry — the single
        # kernel reads stable state + grid buffers, so the graph never recaptures.
        return 0

    def _pre_graph_step(self) -> None:
        self._ensure_fused(self.world.n_envs)

    def _graph_post_physics(self) -> None:
        self._launch(full_pass=1)

    # --------------------------------------------------------- fused fast path

    def _ensure_fused(self, n_envs: int) -> None:
        """Allocate the persistent fused output buffers + cached wp handles."""
        if self._fused_ready:
            return
        w = self.world
        na, dev, dt = self.n_agents, w.device, w.dtype
        od = self.obs_dim

        def z(*shape):
            return torch.zeros(*shape, device=dev, dtype=dt)

        self._f_obs = z(n_envs, na, od)
        self._f_reward = z(n_envs, na)
        self._f_crowd = z(n_envs, na)
        scalar = w.wp_dtype
        self._wp = {
            "obs": wp.from_torch(self._f_obs, dtype=scalar),
            "reward": wp.from_torch(self._f_reward, dtype=scalar),
            "crowd": wp.from_torch(self._f_crowd, dtype=scalar),
        }
        self._fused_ready = True

    def _state_wp(self):
        """(pos, vel) as Warp arrays for the fused kernel."""
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

    def _launch(self, full_pass: int) -> None:
        w = self.world
        n_envs = w.n_envs
        self._ensure_fused(n_envs)
        w.neighbors()  # build the grid on the current state
        grid = w.stepper.grid(n_envs)
        scalar = w.wp_dtype
        pos, vel = self._state_wp()
        wp.launch(
            flocking_obs_reward_kernel,
            dim=(n_envs, self.n_agents),
            inputs=[
                pos,
                vel,
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
        if self._fused_active:
            return self._f_obs
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

    def observation(self, agent_idx: int) -> torch.Tensor:
        return self.observations()[:, agent_idx]

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        return self._cache["reward"][:, agent_idx]

    def rewards(self) -> torch.Tensor:
        if self._fused_active:
            return self._f_reward
        return super().rewards()

    def info(self) -> dict[str, Any]:
        if self._fused_active:
            return {"crowding": self._f_crowd}
        return {"crowding": self._cache["crowd"]}
