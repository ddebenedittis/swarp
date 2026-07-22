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
        self.goals: torch.Tensor | None = None  # [n_envs, n_agents, 2]
        self.obstacle_pos: torch.Tensor | None = None  # [n_envs, n_obstacles, 2]
        self.obstacle_radius: torch.Tensor | None = None  # [n_obstacles]
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
        self.state = warp_step(self.stepper, self.state, actions)

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
        self.stepper.set_obstacles(pos, radius, shape=shape, angle=angle, half_extents=half_extents)

    # -------------------------------------------------------------- neighbors

    def neighbors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Padded within-radius neighbor lists on the *current* (post-step) state.

        Returns zero-copy ``(neighbor_idx [n_envs, n_agents, K] int32,
        neighbor_count [n_envs, n_agents] int32)`` views; they are overwritten
        by the next call, so gather from them within the same step.

        This is a separate build from the stepper's per-substep neighbor lists:
        those are queried on each *intermediate* (pre-integration) substep state
        to compute collision forces and are discarded, whereas observations and
        rewards need neighbors on the final post-step state. The two are
        genuinely different states, so the build here is not redundant with the
        force-time queries (it is intentionally recomputed once per step).
        """
        if not self.stepper.collisions:
            raise RuntimeError("neighbor lists require WorldConfig.collisions=True")
        grid = self.stepper.grid(self.n_envs)
        pos_wp = wp.from_torch(
            self.state.pos.detach().contiguous(),
            dtype=VEC2[self.wp_dtype],
            requires_grad=False,
        )
        grid.build(pos_wp)
        return grid.torch_views()

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
