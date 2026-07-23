"""Discovery: a team covers scattered targets, each needing several agents nearby.

Port of the VMAS ``discovery`` scenario. A target counts as *covered* once at
least ``agents_per_target`` agents are within ``covering_range`` of it; the team
earns a one-off shared reward the step a target is first covered. Agents also pay
a small per-step time penalty and a per-contact collision penalty. Coverage is
computed with ``torch.cdist`` over agent/target positions.
"""

from __future__ import annotations

from typing import Any

import torch

from wmas.core.config import WorldConfig
from wmas.core.world import World
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from wmas.scenarios.base import Scenario


class DiscoveryScenario(Scenario):
    def __init__(
        self,
        n_agents: int = 5,
        n_targets: int = 5,
        agent_radius: float = 0.05,
        world_size: float = 1.0,
        covering_range: float = 0.2,
        agents_per_target: int = 2,
        covering_reward: float = 1.0,
        time_penalty: float = -0.01,
        collision_penalty: float = -0.1,
        max_speed: float = 1.0,
    ) -> None:
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.agent_radius = agent_radius
        self.world_size = world_size
        self.covering_range = covering_range
        self.agents_per_target = agents_per_target
        self.covering_reward = covering_reward
        self.time_penalty = time_penalty
        self.collision_penalty = collision_penalty
        self.max_speed = max_speed

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
        self.targets: torch.Tensor | None = None  # [n_envs, n_targets, 2]
        self.covered: torch.Tensor | None = None  # [n_envs, n_targets] bool (ever covered)
        self._cache: dict[str, torch.Tensor] | None = None
        return self.world

    def reset_world(self, env_mask: torch.Tensor | None = None) -> None:
        w = self.world
        n = w.n_envs
        lim = self.world_size - 2.0 * self.agent_radius
        spawn = w.sample_uniform((n, self.n_agents, 2), -lim, lim)
        targets = w.sample_uniform((n, self.n_targets, 2), -lim, lim)
        covered = torch.zeros(n, self.n_targets, dtype=torch.bool, device=w.device)

        if self.targets is None:
            self.targets = torch.zeros(n, self.n_targets, 2, device=w.device, dtype=w.dtype)
            self.covered = torch.zeros(n, self.n_targets, dtype=torch.bool, device=w.device)

        if env_mask is None:
            w.state.pos.data.copy_(spawn)
            w.state.vel.data.zero_()
            self.targets.copy_(targets)
            self.covered.copy_(covered)
        else:
            m3 = env_mask.view(-1, 1, 1)
            w.state.pos.data.copy_(torch.where(m3, spawn, w.state.pos.data))
            w.state.vel.data.copy_(
                torch.where(m3, torch.zeros_like(w.state.vel.data), w.state.vel.data)
            )
            self.targets.copy_(torch.where(m3, targets, self.targets))
            self.covered.copy_(torch.where(env_mask.view(-1, 1), covered, self.covered))
        self._refresh()

    def post_step(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        w = self.world
        pos = w.state.pos  # [n_envs, n_agents, 2]
        d = torch.cdist(pos, self.targets)  # [n_envs, n_agents, n_targets]
        within = d < self.covering_range
        count = within.sum(dim=1)  # [n_envs, n_targets] agents near each target
        covered_now = count >= self.agents_per_target
        newly = covered_now & ~self.covered  # [n_envs, n_targets]
        self.covered = self.covered | covered_now

        # agent-agent contacts (undirected touching count per agent)
        dd = torch.cdist(pos, pos)
        touch = (dd < 2.0 * self.agent_radius).sum(dim=-1).to(w.dtype) - 1.0  # minus self

        self._cache = {
            "newly": newly,
            "count": count,
            "touch": touch,
            "rel_targets": (self.targets.unsqueeze(1) - pos.unsqueeze(2)).reshape(
                w.n_envs, w.n_agents, self.n_targets * 2
            ),
            "covered_frac": self.covered.float().mean(-1),
        }

    def observations(self) -> torch.Tensor:
        w = self.world
        s = w.state
        covered_flag = self.covered.to(w.dtype).unsqueeze(1).expand(-1, w.n_agents, -1)
        return torch.cat([s.pos, s.vel, self._cache["rel_targets"], covered_flag], dim=-1)

    def observation(self, agent_idx: int) -> torch.Tensor:
        return self.observations()[:, agent_idx]

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        c = self._cache
        return self.collision_penalty * c["touch"][
            :, agent_idx
        ] + self.time_penalty * torch.ones_like(c["touch"][:, agent_idx])

    def global_reward(self) -> torch.Tensor:
        return self.covering_reward * self._cache["newly"].sum(dim=-1).to(self.world.dtype)

    def done(self) -> torch.Tensor:
        return self.covered.all(dim=-1)

    def info(self) -> dict[str, Any]:
        return {"covered_frac": self._cache["covered_frac"]}
