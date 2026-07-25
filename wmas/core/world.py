"""Torch-facing world container: state tensors, stepper, goals, obstacles, RNG."""

from __future__ import annotations

import torch
import warp as wp

from wmas.core.config import WorldConfig
from wmas.core.state import VEC2
from wmas.core.stepper import Stepper
from wmas.dynamics.base import AgentConfig, action_dim
from wmas.interop.autograd import TorchState, warp_step

TORCH_TO_WP = {torch.float32: wp.float32, torch.float64: wp.float64}


class World:
    """One batched multi-agent world: [n_envs, n_agents] on a single device.

    Scenarios build a World in ``make_world`` and write into ``state`` /
    ``goals`` / obstacles during ``reset_world``. All tensors live on
    ``device``; the step itself runs as Warp kernels via the Stepper.
    """

    def __init__(
        self,
        agent_configs: list[AgentConfig],
        world_config: WorldConfig | None = None,
        n_envs: int = 1,
        device: str = "cuda:0",
        dt: float = 0.1,
        substeps: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_envs = n_envs
        self.n_agents = len(agent_configs)
        self.device = device
        self.dtype = dtype
        self.wp_dtype = TORCH_TO_WP[dtype]
        self.agent_configs = agent_configs
        # Env-level action width: max arity over agent models (models ignore
        # slots beyond their own). All current 2D vehicle models use 2.
        self.act_dim = max(action_dim(c.model) for c in agent_configs)
        self.config = world_config
        self.stepper = Stepper(
            agent_configs,
            dt=dt,
            substeps=substeps,
            device=device,
            dtype=self.wp_dtype,
            world=world_config,
        )
        self.state: TorchState = self.zero_state()
        # Persistent-buffer / CUDA-graph execution (opt-in via enable_persistent).
        self.runtime = None
        self._persistent = False
        self._detached = False
        # True after a persistent no-grad step whose runtime ran the whole-step
        # post-physics hook (obs/reward) — Environment then skips a redundant post_step.
        self.ran_post_physics = False
        # Last applied action, exposed for scenarios that shape reward on the
        # control input (e.g. action-smoothness). Set by Environment.step before
        # the post-physics reward path; cleared to None on reset.
        self.action: torch.Tensor | None = None  # [n_envs, n_agents, act_dim]
        self.goals: torch.Tensor | None = None  # [n_envs, n_agents, 2]
        self.obstacle_pos: torch.Tensor | None = None  # [n_envs, n_obstacles, 2]
        self.obstacle_radius: torch.Tensor | None = None  # [n_obstacles]
        self.obstacle_shape: torch.Tensor | None = None  # [n_obstacles]
        self.obstacle_angle: torch.Tensor | None = None  # [n_obstacles]
        self.obstacle_half_extents: torch.Tensor | None = None  # [n_obstacles, 2]
        self.generator: torch.Generator | None = None  # installed by Environment
        self.agent_radius = torch.tensor(
            [c.radius for c in agent_configs], device=device, dtype=dtype
        )

    # ------------------------------------------------------------------ state

    def zero_state(self) -> TorchState:
        e, a = self.n_envs, self.n_agents

        def z(*shape):
            return torch.zeros(*shape, device=self.device, dtype=self.dtype)

        return TorchState(
            pos=z(e, a, 2), theta=z(e, a), vel=z(e, a, 2), speed=z(e, a), ang_vel=z(e, a)
        )

    def step(self, actions: torch.Tensor) -> None:
        """Advance the world one step (differentiable when grads are enabled)."""
        grad_mode = torch.is_grad_enabled() and (
            actions.requires_grad or any(t is not None and t.requires_grad for t in self.state)
        )
        if self._persistent and not grad_mode:
            if self._detached:
                # Returning from a grad step: reload persistent buffers from the
                # current (fresh-tensor) state, then rebind to the stable views.
                self.runtime.load_state(self.state)
                self.state = self.runtime.state_views
                self._detached = False
            self.state = self.runtime.step(actions)
            self.ran_post_physics = self.runtime.ran_post_physics
        elif self._persistent:
            # Grad step: detach+clone so the tape references fresh arrays, never
            # the persistent buffers (which the no-grad path overwrites in place).
            detached = TorchState(*(t.detach().clone() for t in self.state))
            self._detached = True
            self.state = warp_step(self.stepper, detached, actions)
            self.ran_post_physics = False
        else:
            self.state = warp_step(self.stepper, self.state, actions)
            self.ran_post_physics = False

    def enable_persistent(self, use_graph: bool = True) -> None:
        """Switch to persistent-buffer execution (optionally CUDA-graph-backed).

        Builds a :class:`~wmas.interop.persistent.StepRuntime`, seeds it with the
        current state, and rebinds ``self.state`` to the runtime's stable
        zero-copy views. The no-grad :meth:`step` then routes through the runtime;
        grad steps transparently fall back to the functional path.
        """
        from wmas.interop.persistent import StepRuntime

        self.runtime = StepRuntime(self.stepper, self.n_envs, self.act_dim, use_graph=use_graph)
        self.runtime.load_state(self.state)
        self.state = self.runtime.state_views
        self._persistent = True
        self._detached = False

    def reset_state(self) -> None:
        """Reset the state to zeros. In persistent mode this zeroes the buffers
        in place (keeping the views/graph valid); otherwise it reallocates."""
        if self._persistent:
            self.runtime.reset_state()
            self.state = self.runtime.state_views
            self._detached = False
        else:
            self.state = self.zero_state()

    # ------------------------------------------------------------- randomness

    def sample_uniform(self, shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
        u = torch.rand(shape, generator=self.generator, device=self.device, dtype=self.dtype)
        return u * (high - low) + low

    # -------------------------------------------------------------- obstacles

    def set_obstacles(
        self,
        pos: torch.Tensor,
        radius: torch.Tensor,
        shape: torch.Tensor | None = None,
        angle: torch.Tensor | None = None,
        half_extents: torch.Tensor | None = None,
    ) -> None:
        """Install static obstacles (circle/box/segment). See
        :meth:`wmas.core.stepper.Stepper.set_obstacles` for shape semantics."""
        self.obstacle_pos = pos
        self.obstacle_radius = radius
        self.obstacle_shape = shape
        self.obstacle_angle = angle
        self.obstacle_half_extents = half_extents
        self.stepper.set_obstacles(pos, radius, shape=shape, angle=angle, half_extents=half_extents)

    # -------------------------------------------------------------- neighbors

    def neighbors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Padded within-radius neighbor lists on the *current* (post-step) state.

        Returns zero-copy ``(neighbor_idx [n_envs, n_agents, K] int32,
        neighbor_count [n_envs, n_agents] int32)`` views; they are overwritten
        by the next call, so gather from them within the same step.

        This builds neighbors on the final post-step state (what observations and
        rewards need). It also stamps ``grid.built_version`` with the stepper's
        current ``state_version``, so the next step can recognize that its input
        state already has a matching neighbor list and skip substep 0's rebuild
        (``WorldConfig.neighbor_reuse``) — turning the two builds per step (this
        one plus the force-time query) into one.
        """
        if not self.stepper.collisions:
            raise RuntimeError("neighbor lists require WorldConfig.collisions=True")
        grid = self.stepper.grid(self.n_envs)
        if self._persistent and not self._detached:
            # The persistent state's pos is already a Warp array — build directly
            # on it (no re-wrap). Runs on the default stream, ordered with the
            # graph replay that reads the grid's lists on the next step.
            grid.build(self.runtime.state.pos)
        else:
            pos_wp = wp.from_torch(
                self.state.pos.detach().contiguous(),
                dtype=VEC2[self.wp_dtype],
                requires_grad=False,
            )
            grid.build(pos_wp)
        grid.built_version = self.stepper.state_version
        return grid.torch_views()

    def mark_pos_dirty(self) -> None:
        """Invalidate any cached neighbor list after writing ``state.pos`` out of
        band (a reset or an interactive drag), so the next step rebuilds instead
        of reusing a list that no longer matches the positions."""
        if self.stepper.collisions:
            self.stepper.grid(self.n_envs).built_version = -1

    def neighbor_overflow(self) -> torch.Tensor:
        """Bool ``[n_envs, n_agents]`` flagging agents whose in-radius neighbor
        count exceeded ``max_neighbors`` on the last :meth:`neighbors` build (so
        their collision forces and counts are truncated). Zero-copy, no sync."""
        return self.stepper.grid(self.n_envs).overflow_view()

    def edge_index(self, rebuild: bool = True) -> torch.Tensor:
        """Radius graph on the current state as COO [2, E] (syncs once for E).

        With ``rebuild=False`` the grid is assumed already built on the current
        state (e.g. by the scenario's ``post_step``/reset, which both call
        :meth:`neighbors`) and the redundant rebuild is skipped.
        """
        if rebuild:
            self.neighbors()
        return self.stepper.grid(self.n_envs).edge_index()
