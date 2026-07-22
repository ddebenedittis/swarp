"""VMAS-style vectorized Environment over a Scenario."""

from __future__ import annotations

from typing import Any

import torch

from wmas.scenarios.base import Scenario


class Environment:
    """Vectorized multi-agent environment: all tensors stay on ``device``.

    ``step`` is differentiable end-to-end (dynamics via the Warp adjoint,
    observations/rewards via torch autograd) when called with grads enabled;
    under ``torch.no_grad()`` it runs the tape-free hot path with no per-step
    host<->device transfers.

    There is no auto-reset (like raw VMAS): inspect ``done`` and call
    :meth:`reset` when you want fresh episodes.
    """

    def __init__(
        self,
        scenario: Scenario,
        n_envs: int,
        device: str = "cuda:0",
        dt: float = 0.1,
        substeps: int = 1,
        dtype: torch.dtype = torch.float32,
        max_steps: int | None = None,
        seed: int = 0,
    ) -> None:
        self.scenario = scenario
        self.n_envs = n_envs
        self.device = device
        self.dtype = dtype
        self.max_steps = max_steps
        self.world = scenario.make_world(
            n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        self.n_agents = self.world.n_agents
        self._seed(seed)
        self._step_count = torch.zeros(n_envs, device=device, dtype=torch.int32)

    def _seed(self, seed: int) -> None:
        self.world.generator = torch.Generator(device=self.device)
        self.world.generator.manual_seed(seed)

    # ------------------------------------------------------------------- API

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Reset all envs; returns stacked observations [n_envs, n_agents, obs_dim]."""
        if seed is not None:
            self._seed(seed)
        with torch.no_grad():
            self.world.state = self.world.zero_state()
            self.scenario.reset_world(None)
        self._step_count.zero_()
        return self.scenario.observations()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Advance every env by one step.

        Args:
            actions: ``[n_envs, n_agents, 2]`` tensor on the env device.

        Returns:
            ``(obs [n_envs, n_agents, obs_dim], reward [n_envs, n_agents],
            done [n_envs] bool, info dict)`` — all on the env device.
        """
        if actions.shape != (self.n_envs, self.n_agents, 2):
            raise ValueError(
                f"actions must have shape {(self.n_envs, self.n_agents, 2)}, "
                f"got {tuple(actions.shape)}"
            )
        if actions.dtype != self.dtype:
            raise TypeError(f"actions dtype {actions.dtype} != env dtype {self.dtype}")

        self.world.step(actions)
        self.scenario.post_step()

        obs = self.scenario.observations()
        reward = self.scenario.rewards()
        self._step_count += 1
        done = self.scenario.done()
        if self.max_steps is not None:
            done = done | (self._step_count >= self.max_steps)
        return obs, reward, done, self.scenario.info()

    def radius_graph(self) -> torch.Tensor:
        """COO edge index [2, E] of the current within-radius neighbor graph."""
        return self.world.edge_index()
