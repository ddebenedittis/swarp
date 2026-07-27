"""Scenario abstraction, mirroring VMAS's BaseScenario.

A Scenario defines the world composition (agents, obstacles, goals), how it
resets, and per-agent observation/reward. Observations and rewards are written
as torch ops over ``world.state`` tensors, so they are differentiable
end-to-end together with the Warp dynamics step.

Reward terms are split VMAS-style: ``agent_reward(i)`` returns the per-agent
term, ``global_reward()`` the term shared by all agents of an env; the
Environment sums them.

A scenario that also ships fused Warp obs/reward kernels for the no-grad hot path
should subclass :class:`~wmas.scenarios.fused.FusedScenario` rather than implement
the fused bookkeeping itself — see ``docs/writing-a-scenario.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from wmas.core.hooks import WholeStepHook
from wmas.core.world import World


class Scenario(ABC):
    """Base class for scenarios. Subclasses must set ``self.world`` in make_world.

    Fused fast path
    ---------------
    :attr:`fused_available` is a **class-level** fact: True if this scenario ships
    fused Warp obs/reward kernels for the no-grad hot path. The Environment turns
    them on per step via :meth:`set_fused_active` — off in grad mode, where the torch
    reference path runs so autograd works. Fused implementations return zero-copy
    views of persistent buffers that the next step overwrites; callers that must
    retain them across steps clone (the Environment's ``clone_outputs`` flag and the
    TorchRL wrapper do this).
    """

    world: World

    #: Whether this scenario ships fused Warp obs/reward kernels. A class attribute
    #: precisely so it can be read off the class — see
    #: :func:`wmas.scenarios.fused_scenarios`.
    fused_available: bool = False

    #: Whether the fused kernels are the active obs/reward path *right now*. Owned by
    #: the Environment (:meth:`set_fused_active`); read by the scenario.
    fused_active: bool = False

    # How far this scenario's fused path may legitimately differ from its torch reference
    # path on the same seeded trajectory. It is the *scenario* that knows why its two
    # paths differ, so the number lives here and both readers take it from here: the
    # per-scenario benchmark's parity gate (``wmas.benchmark.ablation._parity_ok``, via
    # ``wmas.benchmark.scenarios.run_config``) and the shared test harness
    # (``tests/conftest.py``'s ``FusedSpec``). A second hardcoded table is what this
    # replaces — the CLI's flat 1e-5 reported push-t as a parity failure for a difference
    # its own test suite had documented as by design.
    #
    # The default covers the ordinary case: the two paths compute the same quantity, and
    # they differ only by ulp-scale reassociation (sqrt, reduction order).
    parity_rtol: float = 1e-5
    parity_atol: float = 1e-6

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Per-agent observation width. Forwarded as ``Environment.obs_dim``."""

    def set_fused_active(self, active: bool) -> None:
        """Select the fused (True) or torch-reference (False) obs/reward path.

        Called by the Environment once per ``reset``/``step``. Overriding is for
        wrappers that need to know; the flag itself must stay authoritative.
        """
        self.fused_active = active

    @abstractmethod
    def make_world(
        self,
        n_envs: int,
        device: str,
        dt: float,
        substeps: int,
        dtype: torch.dtype,
    ) -> World:
        """Build and return the World (agents, limits, interaction config).

        This is also where **all** one-time buffer allocation belongs — goals, target
        sets, coverage latches, body state. ``n_envs`` is known here, so nothing needs
        a lazy ``if self.x is None`` branch inside ``reset_world`` (which runs on the
        per-step auto-reset path).
        """

    @abstractmethod
    def reset_world(self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False) -> None:
        """(Re)randomize state, goals, and obstacles.

        ``env_mask`` is either ``None`` (reset every env) or a boolean
        ``[n_envs]`` tensor selecting which envs to reset; unselected envs are
        left untouched. Implementations should stay host-sync-free (sample the
        full batch width and blend with ``torch.where(env_mask, ...)`` rather
        than gathering a variable number of indices) so masked auto-reset can
        run inside the step loop without a device→host round-trip.
        :meth:`wmas.core.world.World.write_state` does that blend for the agent state.

        ``obs_only=True`` marks the mid-step auto-reset pass: reward/done/info for the
        transition just taken have **already been returned**, so this pass must
        recompute observations without clobbering their buffers. ``False`` (a
        standalone ``reset``/``reset_at``) recomputes everything so ``info()`` is
        populated.
        """

    @abstractmethod
    def observations(self) -> torch.Tensor:
        """All observations, batched: ``[n_envs, n_agents, obs_dim]``.

        Batched rather than per-agent because that is what the fused kernels write and
        what a large fleet needs (a per-agent loop costs O(n_agents) small launches).
        :meth:`observation` slices this.
        """

    def observation(self, agent_idx: int) -> torch.Tensor:
        """Observation for one agent across all envs: ``[n_envs, obs_dim]``."""
        return self.observations()[:, agent_idx]

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        """Per-agent reward term ``[n_envs]``."""
        return torch.zeros(self.world.n_envs, device=self.world.device, dtype=self.world.dtype)

    def global_reward(self) -> torch.Tensor:
        """Reward term shared by every agent of an env: ``[n_envs]``."""
        return torch.zeros(self.world.n_envs, device=self.world.device, dtype=self.world.dtype)

    def rewards(self) -> torch.Tensor:
        """Total rewards ``[n_envs, n_agents]`` (per-agent + shared terms)."""
        per_agent = torch.stack([self.agent_reward(i) for i in range(self.world.n_agents)], dim=1)
        return per_agent + self.global_reward().unsqueeze(1)

    def done(self) -> torch.Tensor:
        """Termination flags ``[n_envs]`` (bool). Default: never."""
        return torch.zeros(self.world.n_envs, device=self.world.device, dtype=torch.bool)

    def info(self) -> dict[str, Any]:
        """Extra diagnostics (device tensors preferred)."""
        return {}

    def post_step(self) -> None:  # noqa: B027 (optional hook, intentionally empty)
        """Hook called right after the physics step, before obs/rewards."""

    def graph_hook(self) -> WholeStepHook | None:
        """The whole-step hook to fold into the CUDA graph, or ``None``.

        Returning ``None`` *is* "not capturable" — there is no separate opt-in flag to
        keep in sync with it. :class:`~wmas.scenarios.fused.FusedScenario` builds the
        hook from a scenario's declared buffer spec, so a fused scenario gets this for
        free; see :class:`~wmas.core.hooks.WholeStepHook` for the contract each member
        has to honour.
        """
        return None

    def render_extras(self, env_idx: int) -> dict[str, Any]:
        """Extra geometry for the viewer to overlay for env ``env_idx`` (default none).

        Override to feed custom drawables into ``RenderGeometry.extras`` (e.g. lidar rays,
        target zones, communication links) without the core renderer needing to know about
        them. Keys are overlay names; values are whatever that overlay expects (CPU-friendly).
        """
        return {}
