"""Flocking: a Reynolds-style boids reward over within-radius neighbours.

Each agent is rewarded for cohesion (staying near the local neighbour centroid)
and alignment (matching the local mean velocity), and penalized for crowding
(separation) when neighbours come closer than ``separation_dist``. All terms are
differentiable torch ops over the neighbour features the World already exposes.
"""

from __future__ import annotations

from typing import Any

import torch

from wmas.core.config import WorldConfig
from wmas.core.world import World
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from wmas.scenarios.base import Scenario


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
        return self.world

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
        self._refresh()

    def post_step(self) -> None:
        self._refresh()

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

    def info(self) -> dict[str, Any]:
        return {"crowding": self._cache["crowd"]}
