"""Scenario abstraction, mirroring VMAS's BaseScenario.

A Scenario defines the world composition (agents, obstacles, goals), how it
resets, and per-agent observation/reward. Observations and rewards are written
as torch ops over ``world.state`` tensors, so they are differentiable
end-to-end together with the Warp dynamics step.

Reward terms are split VMAS-style: ``agent_reward(i)`` returns the per-agent
term, ``global_reward()`` the term shared by all agents of an env; the
Environment sums them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from wmas.core.world import World


class Scenario(ABC):
    """Base class for scenarios. Subclasses must set ``self.world`` in make_world."""

    world: World

    @abstractmethod
    def make_world(
        self,
        n_envs: int,
        device: str,
        dt: float,
        substeps: int,
        dtype: torch.dtype,
    ) -> World:
        """Build and return the World (agents, limits, interaction config)."""

    @abstractmethod
    def reset_world(self, env_indices: torch.Tensor | None = None) -> None:
        """(Re)randomize state, goals, and obstacles.

        ``env_indices`` selects a subset of envs (None = all). Called by the
        Environment under ``torch.no_grad()``; may sync (not on the hot path).
        """

    @abstractmethod
    def observation(self, agent_idx: int) -> torch.Tensor:
        """Observation for one agent across all envs: ``[n_envs, obs_dim]``."""

    def observations(self) -> torch.Tensor:
        """All observations stacked: ``[n_envs, n_agents, obs_dim]``.

        Default stacks :meth:`observation` per agent; override with a fully
        batched version for large fleets (the per-agent loop costs
        O(n_agents) small kernel launches).
        """
        return torch.stack(
            [self.observation(i) for i in range(self.world.n_agents)], dim=1
        )

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        """Per-agent reward term ``[n_envs]``."""
        return torch.zeros(self.world.n_envs, device=self.world.device, dtype=self.world.dtype)

    def global_reward(self) -> torch.Tensor:
        """Reward term shared by every agent of an env: ``[n_envs]``."""
        return torch.zeros(self.world.n_envs, device=self.world.device, dtype=self.world.dtype)

    def rewards(self) -> torch.Tensor:
        """Total rewards ``[n_envs, n_agents]`` (per-agent + shared terms)."""
        per_agent = torch.stack(
            [self.agent_reward(i) for i in range(self.world.n_agents)], dim=1
        )
        return per_agent + self.global_reward().unsqueeze(1)

    def done(self) -> torch.Tensor:
        """Termination flags ``[n_envs]`` (bool). Default: never."""
        return torch.zeros(self.world.n_envs, device=self.world.device, dtype=torch.bool)

    def info(self) -> dict[str, Any]:
        """Extra diagnostics (device tensors preferred)."""
        return {}

    def post_step(self) -> None:
        """Hook called right after the physics step, before obs/rewards."""
