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

    By default there is no auto-reset (like raw VMAS): inspect ``done`` and call
    :meth:`reset` or :meth:`reset_at` when you want fresh episodes. Set
    ``auto_reset=True`` to have :meth:`step` reset done envs in-place via a
    host-sync-free masked path (no ``.any()``/``.nonzero()`` round-trip), so the
    whole loop can stay on device.
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
        auto_reset: bool = False,
    ) -> None:
        self.scenario = scenario
        self.n_envs = n_envs
        self.device = device
        self.dtype = dtype
        self.max_steps = max_steps
        self.auto_reset = auto_reset
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

    def reset_at(self, env_mask: torch.Tensor) -> torch.Tensor:
        """Reset the envs where ``env_mask`` (bool ``[n_envs]``) is True.

        Host-sync-free: the scenario samples the full batch and blends the
        selected envs with ``torch.where``; unselected envs are untouched.
        Returns stacked observations for all envs.
        """
        with torch.no_grad():
            self.scenario.reset_world(env_mask)
        self._step_count = torch.where(
            env_mask, torch.zeros_like(self._step_count), self._step_count
        )
        return self.scenario.observations()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Advance every env by one step.

        Args:
            actions: ``[n_envs, n_agents, act_dim]`` tensor on the env device
                (``act_dim`` = ``world.act_dim``, the max action arity over agent
                models; 2 for the current 2D vehicle models).

        Returns:
            ``(obs [n_envs, n_agents, obs_dim], reward [n_envs, n_agents],
            done [n_envs] bool, info dict)`` — all on the env device.
        """
        act_dim = self.world.act_dim
        if actions.shape != (self.n_envs, self.n_agents, act_dim):
            raise ValueError(
                f"actions must have shape {(self.n_envs, self.n_agents, act_dim)}, "
                f"got {tuple(actions.shape)}"
            )
        if actions.dtype != self.dtype:
            raise TypeError(f"actions dtype {actions.dtype} != env dtype {self.dtype}")

        self.world.step(actions)
        self.scenario.post_step()

        # Reward/done/info describe the transition just taken (terminal state).
        reward = self.scenario.rewards()
        self._step_count += 1
        done = self.scenario.done()
        if self.max_steps is not None:
            done = done | (self._step_count >= self.max_steps)
        info = self.scenario.info()

        if self.auto_reset:
            with torch.no_grad():
                self.scenario.reset_world(done)
            self._step_count = torch.where(
                done, torch.zeros_like(self._step_count), self._step_count
            )

        # Observations reflect the state after any auto-reset (next episode's
        # first obs for done envs), matching the gym/VMAS vec-env convention.
        obs = self.scenario.observations()
        return obs, reward, done, info

    def radius_graph(self) -> torch.Tensor:
        """COO edge index [2, E] of the current within-radius neighbor graph.

        Reuses the neighbor grid that ``step``/``reset`` already built on the
        current state (via the scenario's cache refresh), so no rebuild is
        needed — just the one sync to materialize E.
        """
        return self.world.edge_index(rebuild=False)
