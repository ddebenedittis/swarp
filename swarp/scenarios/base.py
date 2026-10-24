"""Scenario abstraction, mirroring VMAS's BaseScenario.

A Scenario defines the world composition (agents, obstacles, goals), how it
resets, and per-agent observation/reward. Observations and rewards are written
as torch ops over ``world.state`` tensors, so they are differentiable
end-to-end together with the Warp dynamics step.

Reward terms are split VMAS-style: ``agent_reward(i)`` returns the per-agent
term, ``global_reward()`` the term shared by all agents of an env; the
Environment sums them.

A scenario that also ships fused Warp obs/reward kernels for the no-grad hot path
should subclass :class:`~swarp.scenarios.fused.FusedScenario` rather than implement
the fused bookkeeping itself — see ``docs/writing-a-scenario.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from swarp.core.config import WorldConfig
from swarp.core.hooks import WholeStepHook
from swarp.core.world import World


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
    #: :func:`swarp.scenarios.fused_scenarios`.
    fused_available: bool = False

    #: Whether the fused kernels are the active obs/reward path *right now*. Owned by
    #: the Environment (:meth:`set_fused_active`); read by the scenario.
    fused_active: bool = False

    # How far this scenario's fused path may legitimately differ from its torch reference
    # path on the same seeded trajectory. It is the *scenario* that knows why its two
    # paths differ, so the number lives here and both readers take it from here: the
    # per-scenario benchmark's parity gate (``swarp.benchmark.ablation.parity_ok``, via
    # ``swarp.benchmark.scenarios.run_config``) and the shared test harness
    # (``tests/conftest.py``'s ``FusedSpec``). A second hardcoded table is what this
    # replaces — the CLI's flat 1e-5 reported push-t as a parity failure for a difference
    # its own test suite had documented as by design.
    #
    # The default covers the ordinary case: the two paths compute the same quantity, and
    # they differ only by ulp-scale reassociation (sqrt, reduction order).
    parity_rtol: float = 1e-5
    parity_atol: float = 1e-6

    # Training hooks: optional, and "not provided" by default. A trainer reads them to
    # build a privileged critic, role-specific policy heads or an auxiliary loss. Each
    # tensor they return must be a fixed-address buffer written by the same pass as
    # ``observations()``, including the obs-only pass after an auto-reset, so that at
    # every step it describes the same state the observation does (see
    # ``docs/writing-a-scenario.md``).

    #: Width of :meth:`critic_state`; 0 means the scenario provides none.
    state_dim: int = 0
    #: Number of agent roles. 1 means a single role, and :meth:`agent_roles` is ``None``.
    n_roles: int = 1
    #: Agents per role, summing to ``n_agents`` and fixed for the run; empty when the
    #: scenario has no roles. Which agent holds which role may change per episode.
    role_counts: tuple[int, ...] = ()
    #: Width of :meth:`aux_targets`; 0 means the scenario provides none.
    aux_dim: int = 0

    def critic_state(self) -> torch.Tensor | None:
        """Privileged per-env state for a centralised critic, ``[n_envs, state_dim]``, or
        ``None``. Never shown to the actor.

        With roles, any per-agent part of the state is laid out in role-sorted agent order
        (role 0's agents by ascending index, then role 1's, ...), not by agent index: the
        critic reads it as-is and its per-agent value slot ``k`` is the agent at
        role-sorted rank ``k``."""
        return None

    def agent_roles(self) -> torch.Tensor | None:
        """Each agent's role index, ``[n_envs, n_agents]`` integer in ``[0, n_roles)``,
        with exactly ``role_counts[r]`` agents in role ``r`` per env; or ``None``."""
        return None

    def aux_targets(self) -> torch.Tensor | None:
        """Per-agent auxiliary regression targets, ``[n_envs, n_agents, aux_dim]`` (e.g. a
        hidden pose in the agent's frame), or ``None``."""
        return None

    def comm_mask(self) -> torch.Tensor | None:
        """Who may hear whom, ``[n_envs, n_agents, n_agents]`` bool (``[e, i, j]``: agent
        ``i`` receives agent ``j``'s message, e.g. within a radio range), or ``None``,
        meaning every agent hears every teammate. Read by a communicating policy; the
        diagonal is ignored (an agent never messages itself)."""
        return None

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
        world_config: WorldConfig | None = None,
    ) -> World:
        """Build and return the World (agents, limits, interaction config).

        This is also where **all** one-time buffer allocation belongs — goals, target
        sets, coverage latches, body state. ``n_envs`` is known here, so nothing needs
        a lazy ``if self.x is None`` branch inside ``reset_world`` (which runs on the
        per-step auto-reset path).

        ``world_config`` is the caller's engine override, forwarded by
        :class:`~swarp.core.environment.Environment` and :func:`swarp.make`. An
        implementation computes its own :class:`~swarp.core.config.WorldConfig` from its
        scenario parameters as usual and then ends with
        ``.override_with(world_config)`` — that one call is the whole contract, and it
        is what makes engine settings a scenario does not expose (``integrator``,
        ``bounds_mode``, ``neighbor_reuse``, ``grid_dim``, ``uniform_bins``, the
        obstacle damping) reachable without subclassing. See
        :meth:`~swarp.core.config.WorldConfig.override_with` for the merge rule; a test
        pins that all twelve built-ins honour it.
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
        :meth:`swarp.core.world.World.write_state` does that blend for the agent state.

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
        keep in sync with it. :class:`~swarp.scenarios.fused.FusedScenario` builds the
        hook from a scenario's declared buffer spec, so a fused scenario gets this for
        free; see :class:`~swarp.core.hooks.WholeStepHook` for the contract each member
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
