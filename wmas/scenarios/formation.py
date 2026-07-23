"""Formation: agents hold assigned slots of a regular polygon around a centre.

Each agent ``i`` targets slot ``i`` of an ``n_agents``-vertex regular polygon of
radius ``formation_radius`` centred at a per-env point. The reward is
position-shaping toward the assigned slot (``(prev_dist - dist) * factor``, the
same shaping NavigationScenario uses) plus a soft collision penalty, so the team
converges onto — and holds — the shape.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from wmas.core.config import WorldConfig
from wmas.core.world import World
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from wmas.scenarios.base import Scenario


class FormationScenario(Scenario):
    def __init__(
        self,
        n_agents: int = 5,
        agent_radius: float = 0.05,
        world_size: float = 1.0,
        formation_radius: float = 0.5,
        max_speed: float = 1.0,
        pos_shaping_factor: float = 1.0,
        collision_penalty: float = -1.0,
        goal_tolerance: float | None = None,
    ) -> None:
        self.n_agents = n_agents
        self.agent_radius = agent_radius
        self.world_size = world_size
        self.formation_radius = formation_radius
        self.max_speed = max_speed
        self.pos_shaping_factor = pos_shaping_factor
        self.collision_penalty = collision_penalty
        self.goal_tolerance = goal_tolerance if goal_tolerance is not None else 2.0 * agent_radius

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
            neighbor_radius=reach,
            max_neighbors=min(32, max(4, self.n_agents)),
        )
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        # Fixed slot offsets: regular polygon vertices (relative to the centre).
        ang = torch.arange(self.n_agents, device=device, dtype=dtype) * (
            2.0 * math.pi / self.n_agents
        )
        self._slot_offsets = self.formation_radius * torch.stack(
            [torch.cos(ang), torch.sin(ang)], dim=-1
        )  # [n_agents, 2]
        self._prev_dist: torch.Tensor | None = None
        self._cache: dict[str, torch.Tensor] | None = None
        return self.world

    def reset_world(self, env_mask: torch.Tensor | None = None) -> None:
        w = self.world
        n = w.n_envs
        lim = self.world_size - 2.0 * self.agent_radius
        spawn = w.sample_uniform((n, self.n_agents, 2), -lim, lim)
        # Formation centre kept within bounds so all slots stay inside the world.
        clim = max(0.0, self.world_size - self.formation_radius - self.agent_radius)
        center = w.sample_uniform((n, 1, 2), -clim, clim)
        goals = center + self._slot_offsets.unsqueeze(0)  # [n_envs, n_agents, 2]
        if w.goals is None:
            w.goals = torch.zeros(n, self.n_agents, 2, device=w.device, dtype=w.dtype)
        if env_mask is None:
            w.state.pos.data.copy_(spawn)
            w.state.vel.data.zero_()
            w.goals.copy_(goals)
        else:
            m3 = env_mask.view(-1, 1, 1)
            w.state.pos.data.copy_(torch.where(m3, spawn, w.state.pos.data))
            w.state.vel.data.copy_(
                torch.where(m3, torch.zeros_like(w.state.vel.data), w.state.vel.data)
            )
            w.goals.copy_(torch.where(m3, goals, w.goals))
        self._refresh(reset_mask=env_mask)

    def post_step(self) -> None:
        self._refresh()

    def _refresh(self, reset_mask: torch.Tensor | None = None) -> None:
        w = self.world
        pos = w.state.pos
        dist = (pos - w.goals).norm(dim=-1)  # [n_envs, n_agents]
        if self._prev_dist is None:
            self._prev_dist = dist.detach().clone()
        shaping = (self._prev_dist - dist) * self.pos_shaping_factor
        if reset_mask is None:
            self._prev_dist = dist.detach().clone()
        else:
            shaping = torch.where(reset_mask.unsqueeze(-1), torch.zeros_like(shaping), shaping)
            self._prev_dist = torch.where(reset_mask.unsqueeze(-1), dist.detach(), self._prev_dist)

        dd = torch.cdist(pos, pos)
        touching = (dd < 2.0 * self.agent_radius).sum(dim=-1).to(w.dtype) - 1.0
        self._cache = {
            "dist": dist,
            "shaping": shaping,
            "touching": touching,
            "in_formation": (dist < self.goal_tolerance),
        }

    def observations(self) -> torch.Tensor:
        w = self.world
        s = w.state
        return torch.cat([s.pos, s.vel, w.goals - s.pos], dim=-1)

    def observation(self, agent_idx: int) -> torch.Tensor:
        return self.observations()[:, agent_idx]

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        c = self._cache
        return c["shaping"][:, agent_idx] + self.collision_penalty * c["touching"][:, agent_idx]

    def done(self) -> torch.Tensor:
        return self._cache["in_formation"].all(dim=-1)

    def info(self) -> dict[str, Any]:
        c = self._cache
        # Per-agent objective vector [n_envs, n_agents, n_obj]: the two reward terms
        # (shaping, collision) kept separate so a lexicographic/multi-objective loop can
        # consume them via ``("next", "info", "multiobj_reward")``. Their sum over the last
        # dim equals the scalar per-agent reward; the reward key itself stays scalar.
        multiobj_reward = torch.stack(
            [c["shaping"], self.collision_penalty * c["touching"]], dim=-1
        )
        return {
            "multiobj_reward": multiobj_reward,  # [n_envs, n_agents, 2]
            "formation_error": c["dist"].mean(-1),  # [n_envs] scalar diagnostic
        }
