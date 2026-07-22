"""Low-level stepping: the substep pipeline (neighbors -> forces -> integrate).

The Stepper is the single place that knows the pipeline. The autograd bridge
(grad mode) and the Environment hot path (no-grad mode) both call
:meth:`Stepper.launch_substeps` with a :class:`StepBuffers`.

Tape-correctness rules encoded here:

* Every array written during a taped substep (intermediate states, force
  buffers, neighbor lists) must be distinct per substep and fresh per step —
  overwriting an array recorded on a tape silently corrupts its adjoint, and
  adjoint kernels re-read the neighbor lists as data.
* Neighbor construction itself is launched with ``record_tape=False`` (a
  discrete pass; gradients flow through positions read by the force kernel).

The no-grad hot path recycles one cached :class:`StepBuffers` per batch size,
so steady-state stepping performs no allocations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp

from wmas.core.collisions import launch_collision_forces
from wmas.core.config import WorldConfig
from wmas.core.neighbors import NeighborGrid
from wmas.core.state import VEC2, WorldState
from wmas.dynamics.base import AgentConfig, AgentParams, Integrator, build_agent_params
from wmas.dynamics.kernels import launch_integrate


@dataclass
class StepBuffers:
    """Per-step scratch: everything written during the substep chain."""

    inter_states: list[WorldState]  # substeps - 1 intermediate states
    forces: list[wp.array]  # one per substep (shared zero buffer if unused)
    nbr_idx: list[wp.array]  # one per substep (empty if collisions off)
    nbr_cnt: list[wp.array]


class Stepper:
    """Owns per-agent parameters, interaction config, and the substep pipeline."""

    def __init__(
        self,
        configs: list[AgentConfig],
        dt: float = 0.1,
        substeps: int = 1,
        device: str = "cuda:0",
        dtype=wp.float32,
        world: WorldConfig | None = None,
    ) -> None:
        world = world if world is not None else WorldConfig(collisions=False)
        if world.integrator is not Integrator.EULER:
            raise NotImplementedError(
                f"Integrator {world.integrator} is not implemented yet; the dynamics "
                "derivative is a pure @wp.func, so RK4 slots into wmas/dynamics/kernels.py."
            )
        if substeps < 1:
            raise ValueError("substeps must be >= 1")
        self.configs = configs
        self.n_agents = len(configs)
        self.dt = dt
        self.substeps = substeps
        self.sub_dt = dt / substeps
        self.device = device
        self.dtype = dtype
        self.world = world
        self.params: AgentParams = build_agent_params(configs, device=device, dtype=dtype)

        self.collisions = world.collisions
        self.max_agent_radius = max(c.radius for c in configs)
        self.neighbor_radius = 0.0
        if self.collisions:
            reach = 2.0 * self.max_agent_radius + world.collision_margin
            self.neighbor_radius = world.neighbor_radius or reach
            if self.neighbor_radius < reach:
                raise ValueError(
                    f"neighbor_radius={self.neighbor_radius} is smaller than the collision "
                    f"reach {reach} (= 2 * max radius + margin); collisions would be missed."
                )

        self._soft_walls = world.bounds is not None and world.bounds_mode == "soft"
        self._clamp_bounds = (
            world.bounds if world.bounds is not None and world.bounds_mode == "clamp" else None
        )
        vec2 = VEC2[dtype]
        if world.bounds is not None:
            x_min, x_max, y_min, y_max = world.bounds
            self._bounds_min = vec2(x_min, y_min)
            self._bounds_max = vec2(x_max, y_max)
        else:
            self._bounds_min = vec2(0.0, 0.0)
            self._bounds_max = vec2(0.0, 0.0)

        # Obstacles (static per episode); dummies keep kernel signatures fixed.
        self.n_obstacles = 0
        self._obs_pos = wp.zeros((1, 1), dtype=vec2, device=device)
        self._obs_radius = wp.zeros(1, dtype=dtype, device=device)

        self._zero_forces: dict[int, wp.array] = {}
        self._zero_nbr: dict[int, tuple[wp.array, wp.array]] = {}
        self._grids: dict[int, NeighborGrid] = {}
        self._cached_buffers: dict[int, StepBuffers] = {}

    # ------------------------------------------------------------------ setup

    def set_obstacles(self, pos: torch.Tensor, radius: torch.Tensor) -> None:
        """Install static circular obstacles.

        Args:
            pos: ``[n_envs, n_obstacles, 2]`` tensor of centers.
            radius: ``[n_obstacles]`` tensor of radii.
        """
        vec2 = VEC2[self.dtype]
        self.n_obstacles = pos.shape[1]
        self._obs_pos = wp.clone(wp.from_torch(pos.detach().contiguous(), dtype=vec2))
        self._obs_radius = wp.clone(wp.from_torch(radius.detach().contiguous(), dtype=self.dtype))
        # _needs_forces may have flipped: cached buffers could alias the shared
        # zero-force buffer, which the force pass would then overwrite.
        self._cached_buffers.clear()

    # ------------------------------------------------------------- allocation

    def alloc_state(self, n_envs: int, requires_grad: bool = False) -> WorldState:
        return WorldState.zeros(
            n_envs,
            self.n_agents,
            dtype=self.dtype,
            device=self.device,
            requires_grad=requires_grad,
        )

    @property
    def _needs_forces(self) -> bool:
        return self.collisions or self._soft_walls or self.n_obstacles > 0

    def zero_forces(self, n_envs: int) -> wp.array:
        """Shared all-zero force buffer (read-only, safe to reuse under a tape)."""
        buf = self._zero_forces.get(n_envs)
        if buf is None:
            buf = wp.zeros((n_envs, self.n_agents), dtype=VEC2[self.dtype], device=self.device)
            self._zero_forces[n_envs] = buf
        return buf

    def _zero_neighbors(self, n_envs: int) -> tuple[wp.array, wp.array]:
        """All-zero-count neighbor lists (for force passes without collisions)."""
        pair = self._zero_nbr.get(n_envs)
        if pair is None:
            pair = (
                wp.zeros((n_envs, self.n_agents, 1), dtype=wp.int32, device=self.device),
                wp.zeros((n_envs, self.n_agents), dtype=wp.int32, device=self.device),
            )
            self._zero_nbr[n_envs] = pair
        return pair

    def grid(self, n_envs: int) -> NeighborGrid:
        g = self._grids.get(n_envs)
        if g is None:
            g = NeighborGrid(
                n_envs,
                self.n_agents,
                radius=self.neighbor_radius,
                max_neighbors=self.world.max_neighbors,
                device=self.device,
                dtype=self.dtype,
                grid_dim=self.world.grid_dim,
                method=self.world.neighbor_method,
            )
            self._grids[n_envs] = g
        return g

    def make_buffers(self, n_envs: int, requires_grad: bool) -> StepBuffers:
        """Fresh per-step scratch buffers (grad mode allocates these per step)."""
        inter = [self.alloc_state(n_envs, requires_grad) for _ in range(self.substeps - 1)]
        if self._needs_forces:
            forces = [
                wp.zeros(
                    (n_envs, self.n_agents),
                    dtype=VEC2[self.dtype],
                    device=self.device,
                    requires_grad=requires_grad,
                )
                for _ in range(self.substeps)
            ]
        else:
            forces = [self.zero_forces(n_envs)] * self.substeps
        if self.collisions:
            nbr_idx = [
                wp.zeros(
                    (n_envs, self.n_agents, self.world.max_neighbors),
                    dtype=wp.int32,
                    device=self.device,
                )
                for _ in range(self.substeps)
            ]
            nbr_cnt = [
                wp.zeros((n_envs, self.n_agents), dtype=wp.int32, device=self.device)
                for _ in range(self.substeps)
            ]
        else:
            nbr_idx, nbr_cnt = [], []
        return StepBuffers(inter_states=inter, forces=forces, nbr_idx=nbr_idx, nbr_cnt=nbr_cnt)

    def cached_buffers(self, n_envs: int) -> StepBuffers:
        """Recycled scratch for the tape-free hot path (zero steady-state allocs)."""
        bufs = self._cached_buffers.get(n_envs)
        if bufs is None:
            bufs = self.make_buffers(n_envs, requires_grad=False)
            self._cached_buffers[n_envs] = bufs
        return bufs

    # ---------------------------------------------------------------- stepping

    def launch_substeps(
        self,
        state_in: WorldState,
        actions: wp.array,
        state_out: WorldState,
        buffers: StepBuffers,
    ) -> None:
        """Advance one full env step. Functional: ``state_in`` is never written."""
        n_envs = state_in.pos.shape[0]
        chain = [state_in, *buffers.inter_states, state_out]
        world = self.world
        for k in range(self.substeps):
            st = chain[k]
            forces = buffers.forces[k]
            if self._needs_forces:
                if self.collisions:
                    idx, cnt = buffers.nbr_idx[k], buffers.nbr_cnt[k]
                    self.grid(n_envs).query_into(st.pos, idx, cnt)
                else:
                    idx, cnt = self._zero_neighbors(n_envs)
                launch_collision_forces(
                    st.pos,
                    st.vel,
                    self.params,
                    idx,
                    cnt,
                    self._obs_pos,
                    self._obs_radius,
                    self.n_obstacles,
                    world.collision_k,
                    world.collision_c,
                    world.collision_margin,
                    self._soft_walls,
                    self._bounds_min,
                    self._bounds_max,
                    forces,
                    self.dtype,
                )
            launch_integrate(
                st,
                chain[k + 1],
                actions,
                forces,
                self.params,
                self.sub_dt,
                clamp_bounds=self._clamp_bounds,
            )
