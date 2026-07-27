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

import numpy as np
import torch
import warp as wp

from wmas.core.bodies import launch_obstacle_dynamics
from wmas.core.collisions import launch_collision_forces
from wmas.core.config import WorldConfig
from wmas.core.neighbors import NeighborGrid
from wmas.core.state import VEC2, WorldState
from wmas.dynamics.base import (
    NUM_PARAMS,
    P_RADIUS,
    AgentConfig,
    AgentParams,
    DynamicsModel,
    Integrator,
    build_agent_params,
)
from wmas.dynamics.kernels import launch_integrate


@dataclass
class StepBuffers:
    """Per-step scratch: everything written during the substep chain."""

    inter_states: list[WorldState]  # substeps - 1 intermediate states
    forces: list[wp.array]  # one per substep (shared zero buffer if unused)
    nbr_idx: list[wp.array]  # one per substep (empty if collisions off)
    nbr_cnt: list[wp.array]
    taped: bool = False  # True when allocated for a taped (grad) step


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
        if substeps < 1:
            raise ValueError("substeps must be >= 1")
        self.configs = configs
        self.n_agents = len(configs)
        self.dt = dt
        self.substeps = substeps
        self.sub_dt = dt / substeps
        self.device = device
        self.dtype = dtype
        self.torch_dtype = torch.float64 if dtype == wp.float64 else torch.float32
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
        # Pose is per-env — obs_pos [n_envs, n_obs] and obs_angle [n_envs, n_obs]
        # — so a body that moves and *rotates* independently in each env (a
        # scenario-layer movable body, e.g. PushTScenario's T) can be installed
        # as obstacles. Extent attributes (type/radius/half) are env-independent
        # [n_obs]. Default shape is a circle (type 0).
        self.n_obstacles = 0
        self._obs_pos = wp.zeros((1, 1), dtype=vec2, device=device)
        self._obs_radius = wp.zeros(1, dtype=dtype, device=device)
        self._obs_type = wp.zeros(1, dtype=wp.int32, device=device)
        self._obs_angle = wp.zeros((1, 1), dtype=dtype, device=device)
        self._obs_half = wp.zeros(1, dtype=vec2, device=device)
        # Obstacle velocities, so contact damping can use the closing velocity even for
        # a moving body installed as obstacles. Zero for a genuinely static obstacle.
        self._obs_vel = wp.zeros((1, 1), dtype=vec2, device=device)
        self._obs_ang_vel = wp.zeros((1, 1), dtype=dtype, device=device)
        # IMMOVABLE (0) vs MOVABLE (1) per obstacle; a movable body carries mass/inertia
        # and is integrated per substep from the reaction of the agent contacts.
        self._obs_kind = wp.zeros(1, dtype=wp.int32, device=device)
        self._obs_mass = wp.ones(1, dtype=dtype, device=device)
        self._obs_inertia = wp.ones(1, dtype=dtype, device=device)
        # Compound bodies: several shapes rigidly sharing one pose. _obs_body[s] names the
        # body a shape belongs to (its lowest-index shape) and _obs_body_off[s] places the
        # shape in the body frame; the body's own state lives in _body_* at the root index.
        self._obs_body = wp.zeros(1, dtype=wp.int32, device=device)
        self._obs_body_off = wp.zeros(1, dtype=vec2, device=device)
        self._body_pos = wp.zeros((1, 1), dtype=vec2, device=device)
        self._body_angle = wp.zeros((1, 1), dtype=dtype, device=device)
        self._body_vel = wp.zeros((1, 1), dtype=vec2, device=device)
        self._body_ang_vel = wp.zeros((1, 1), dtype=dtype, device=device)
        self._any_movable = False

        self._zero_forces: dict[int, wp.array] = {}
        self._zero_nbr: dict[int, tuple[wp.array, wp.array]] = {}
        self._grids: dict[int, NeighborGrid] = {}
        self._cached_buffers: dict[int, StepBuffers] = {}
        self._out_states: dict[int, list[WorldState]] = {}
        self._out_idx: dict[int, int] = {}
        # Cached zero-copy torch views of each ping-pong output slot (stable
        # tensor objects, so a returned state re-entering the step is recognized
        # by identity) and the reverse map id(view.pos) -> wrapped WorldState.
        self._slot_views: dict[int, list[torch.Tensor]] = {}
        self._wrapped_by_pos: dict[int, WorldState] = {}
        # Single-slot cache for the wrapped action array (the hot loop reuses one
        # action buffer); keyed on (data_ptr, shape, dtype) with the contiguous
        # source held alive so the pointer cannot be recycled under us.
        self._action_wrap: tuple | None = None
        # Bumped whenever a change requires a Stage-4 CUDA-graph recapture
        # (obstacle-count change, first per-env param install). Steady-state
        # per-reset updates hit the in-place paths and leave this untouched.
        self.mutation_version: int = 0
        # Monotonic counter identifying the current state generation; bumped at
        # the end of every full step. The neighbor grid stamps the version it was
        # built at so substep 0 can reuse a still-matching list (see
        # ``launch_substeps`` and ``WorldConfig.neighbor_reuse``).
        self.state_version: int = 0
        self.neighbor_reuse: bool = world.neighbor_reuse
        # The slim 2D Euler kernel is eligible only for an all-2D fleet under the
        # Euler integrator on the no-grad path (the grad path uses the full
        # kernel, whose drone passthrough carries the identity adjoints the slim
        # kernel would drop). ``enable_slim2d`` lets the ablation disable it.
        self.has_drone: bool = any(c.model == DynamicsModel.DRONE for c in configs)
        self.enable_slim2d: bool = True

    # ------------------------------------------------------------------ setup

    def set_obstacles(
        self,
        pos: torch.Tensor,
        radius: torch.Tensor,
        shape: torch.Tensor | None = None,
        angle: torch.Tensor | None = None,
        half_extents: torch.Tensor | None = None,
        vel: torch.Tensor | None = None,
        ang_vel: torch.Tensor | None = None,
        kind: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
        inertia: torch.Tensor | None = None,
        body: torch.Tensor | None = None,
        body_offset: torch.Tensor | None = None,
    ) -> None:
        """Install static obstacles.

        Args:
            pos: ``[n_envs, n_obstacles, 2]`` tensor of centers (per-env).
            radius: ``[n_obstacles]`` radii. Circle radius, capsule radius; a box
                ignores it (its surface is the box boundary).
            shape: ``[n_obstacles]`` int tensor of :class:`ObstacleShape` tags
                (0=circle, 1=box, 2=segment). Defaults to all circles.
            angle: orientation (rad) for box/segment, either ``[n_obstacles]``
                (shared by every env) or ``[n_envs, n_obstacles]`` (per-env, for
                a body that rotates independently in each env). Stored per-env
                either way. Default 0.
            half_extents: ``[n_obstacles, 2]`` box half-extents; for a segment
                the ``[:, 0]`` column is the half-length. Default 0.
            vel: ``[n_envs, n_obstacles, 2]`` obstacle linear velocities, and
                ``ang_vel`` ``[n_envs, n_obstacles]`` angular velocities. Contact
                damping uses the closing velocity, so a *moving* obstacle (a
                scenario-layer movable body installed as obstacles) must pass these or
                its contacts are damped against the agent's absolute velocity instead.
                Default 0, i.e. a static obstacle.
            kind: ``[n_obstacles]`` int tensor of :class:`ObstacleKind` tags
                (0=immovable, 1=movable). Default all immovable. A MOVABLE obstacle is
                integrated inside the substep loop from the reaction of the agent
                contacts, so ``pos``/``angle``/``vel``/``ang_vel`` are treated as its
                initial state rather than a fixed pose.
            mass: ``[n_obstacles]`` masses and ``inertia`` ``[n_obstacles]`` moments of
                inertia about each obstacle's centre; used only by movable obstacles.
                Default 1.
            body: ``[n_obstacles]`` int tensor grouping shapes into *compound* rigid
                bodies — several shapes that share one pose, e.g. Push-T's T as a crossbar
                plus a stem. Each entry names the body's root (its lowest-index shape);
                the default (``None``) gives every shape its own body. ``mass``/``inertia``
                are read from the root and are about the body origin.
            body_offset: ``[n_obstacles, 2]`` placement of each shape in its body frame,
                relative to the body origin. Default 0. ``pos``/``angle`` are still the
                shapes' *world* poses; the body origin is derived from the root.
        """
        vec2 = VEC2[self.dtype]
        n_envs, n_obs = pos.shape[0], pos.shape[1]
        dev = self.device

        def body_arrays():
            """(body ids, offsets, body pos, body angle) as contiguous torch tensors.

            The body origin is derived from its root shape: ``root_pos - R(angle) @ off``,
            so a caller only ever has to describe where the *shapes* are."""
            ids = (
                torch.arange(n_obs, device=dev, dtype=torch.int32)
                if body is None
                else body.detach().to(torch.int32).contiguous()
            )
            off = (
                torch.zeros(n_obs, 2, device=dev, dtype=self.torch_dtype)
                if body_offset is None
                else body_offset.detach().contiguous()
            )
            root = ids.long()
            r_pos = pos.detach()[:, root]  # [n_envs, n_obs, 2] root world position
            if angle is None:
                r_ang = torch.zeros(n_envs, n_obs, device=dev, dtype=self.torch_dtype)
            else:
                a = angle.detach()
                r_ang = (a.unsqueeze(0).expand(n_envs, -1) if a.dim() == 1 else a)[:, root]
            r_off = off[root]  # [n_obs, 2]
            ca, sa = torch.cos(r_ang), torch.sin(r_ang)
            wx = ca * r_off[:, 0] - sa * r_off[:, 1]
            wy = sa * r_off[:, 0] + ca * r_off[:, 1]
            b_pos = r_pos - torch.stack([wx, wy], dim=-1)
            return ids, off, b_pos.contiguous(), r_ang.contiguous()

        def angle_2d() -> torch.Tensor:
            """``angle`` as a contiguous per-env ``[n_envs, n_obs]`` tensor."""
            a = angle.detach()
            if a.dim() == 1:
                a = a.unsqueeze(0).expand(n_envs, -1)
            return a.contiguous()

        # In-place refresh when the obstacle *count* is unchanged: no realloc, no
        # buffer clear, no version bump — so a scenario that re-samples obstacle
        # positions every reset (NavigationScenario) does not force a graph
        # recapture in steady state. Only a count change reallocates.
        same_count = n_obs == self.n_obstacles and self._obs_pos.shape[1] == n_obs and n_obs > 0
        if same_count:
            wp.copy(self._obs_pos, wp.from_torch(pos.detach().contiguous(), dtype=vec2))
            wp.copy(self._obs_radius, wp.from_torch(radius.detach().contiguous(), dtype=self.dtype))
            if shape is None:
                self._obs_type.zero_()
            else:
                wp.copy(
                    self._obs_type,
                    wp.from_torch(shape.detach().to(torch.int32).contiguous(), dtype=wp.int32),
                )
            if angle is None:
                self._obs_angle.zero_()
            else:
                wp.copy(self._obs_angle, wp.from_torch(angle_2d(), dtype=self.dtype))
            if half_extents is None:
                self._obs_half.zero_()
            else:
                wp.copy(
                    self._obs_half, wp.from_torch(half_extents.detach().contiguous(), dtype=vec2)
                )
            if vel is None:
                self._obs_vel.zero_()
            else:
                wp.copy(self._obs_vel, wp.from_torch(vel.detach().contiguous(), dtype=vec2))
            if ang_vel is None:
                self._obs_ang_vel.zero_()
            else:
                wp.copy(
                    self._obs_ang_vel,
                    wp.from_torch(ang_vel.detach().contiguous(), dtype=self.dtype),
                )
            if kind is None:
                self._obs_kind.zero_()
                self._any_movable = False
            else:
                wp.copy(
                    self._obs_kind,
                    wp.from_torch(kind.detach().to(torch.int32).contiguous(), dtype=wp.int32),
                )
                self._any_movable = bool((kind != 0).any().item())
            if mass is not None:
                wp.copy(self._obs_mass, wp.from_torch(mass.detach().contiguous(), dtype=self.dtype))
            if inertia is not None:
                wp.copy(
                    self._obs_inertia,
                    wp.from_torch(inertia.detach().contiguous(), dtype=self.dtype),
                )
            if kind is None and not self._any_movable:
                # No bodies involved: skip the body bookkeeping entirely. It allocates
                # torch temporaries and reads a flag off the device, neither of which is
                # legal inside a CUDA graph capture — and transport re-installs its
                # obstacles from *inside* the captured whole-step graph.
                return
            ids, off, b_pos, b_ang = body_arrays()
            wp.copy(self._obs_body, wp.from_torch(ids, dtype=wp.int32))
            wp.copy(self._obs_body_off, wp.from_torch(off, dtype=vec2))
            # Re-seed the body state from what was just installed, so a caller can write a
            # body's pose (a reset) or its whole state (a scenario that integrated the body
            # itself on a grad step) and have the engine continue from there.
            wp.copy(self._body_pos, wp.from_torch(b_pos, dtype=vec2))
            wp.copy(self._body_angle, wp.from_torch(b_ang, dtype=self.dtype))
            if vel is None:
                self._body_vel.zero_()
            else:
                root = ids.long()
                wp.copy(
                    self._body_vel,
                    wp.from_torch(vel.detach()[:, root].contiguous(), dtype=vec2),
                )
            if ang_vel is None:
                self._body_ang_vel.zero_()
            else:
                root = ids.long()
                wp.copy(
                    self._body_ang_vel,
                    wp.from_torch(ang_vel.detach()[:, root].contiguous(), dtype=self.dtype),
                )
            return

        self.n_obstacles = n_obs
        self._obs_pos = wp.clone(wp.from_torch(pos.detach().contiguous(), dtype=vec2))
        self._obs_radius = wp.clone(wp.from_torch(radius.detach().contiguous(), dtype=self.dtype))

        if shape is None:
            self._obs_type = wp.zeros(n_obs, dtype=wp.int32, device=dev)
        else:
            self._obs_type = wp.clone(
                wp.from_torch(shape.detach().to(torch.int32).contiguous(), dtype=wp.int32)
            )
        if angle is None:
            self._obs_angle = wp.zeros((n_envs, n_obs), dtype=self.dtype, device=dev)
        else:
            self._obs_angle = wp.clone(wp.from_torch(angle_2d(), dtype=self.dtype))
        if half_extents is None:
            self._obs_half = wp.zeros(n_obs, dtype=vec2, device=dev)
        else:
            self._obs_half = wp.clone(wp.from_torch(half_extents.detach().contiguous(), dtype=vec2))
        if vel is None:
            self._obs_vel = wp.zeros((n_envs, n_obs), dtype=vec2, device=dev)
        else:
            self._obs_vel = wp.clone(wp.from_torch(vel.detach().contiguous(), dtype=vec2))
        if ang_vel is None:
            self._obs_ang_vel = wp.zeros((n_envs, n_obs), dtype=self.dtype, device=dev)
        else:
            self._obs_ang_vel = wp.clone(
                wp.from_torch(ang_vel.detach().contiguous(), dtype=self.dtype)
            )
        if kind is None:
            self._obs_kind = wp.zeros(n_obs, dtype=wp.int32, device=dev)
            self._any_movable = False
        else:
            self._obs_kind = wp.clone(
                wp.from_torch(kind.detach().to(torch.int32).contiguous(), dtype=wp.int32)
            )
            self._any_movable = bool((kind != 0).any().item())
        if mass is None:
            self._obs_mass = wp.full(n_obs, self.dtype(1.0), dtype=self.dtype, device=dev)
        else:
            self._obs_mass = wp.clone(wp.from_torch(mass.detach().contiguous(), dtype=self.dtype))
        if inertia is None:
            self._obs_inertia = wp.full(n_obs, self.dtype(1.0), dtype=self.dtype, device=dev)
        else:
            self._obs_inertia = wp.clone(
                wp.from_torch(inertia.detach().contiguous(), dtype=self.dtype)
            )
        ids, off, b_pos, b_ang = body_arrays()
        self._obs_body = wp.clone(wp.from_torch(ids, dtype=wp.int32))
        self._obs_body_off = wp.clone(wp.from_torch(off, dtype=vec2))
        self._body_pos = wp.clone(wp.from_torch(b_pos, dtype=vec2))
        self._body_angle = wp.clone(wp.from_torch(b_ang, dtype=self.dtype))
        self._body_vel = wp.zeros((n_envs, n_obs), dtype=vec2, device=dev)
        self._body_ang_vel = wp.zeros((n_envs, n_obs), dtype=self.dtype, device=dev)
        # _needs_forces may have flipped: cached buffers could alias the shared
        # zero-force buffer, which the force pass would then overwrite. A shape
        # change also invalidates any captured CUDA graph.
        self._cached_buffers.clear()
        self.mutation_version += 1

    def set_agent_params_per_env(self, floats: torch.Tensor | np.ndarray) -> None:
        """Install a per-env parameter override for domain randomization.

        Once set, every step launches the per-env kernel variants that index
        ``params[e, a, ...]``; pass ``None``-equivalent by never calling this to
        keep the shared fast path. Static per episode — call again at reset to
        re-randomize. Buffer shapes are unchanged, so no hot-path buffers are
        cleared.

        Args:
            floats: ``[n_envs, n_agents, NUM_PARAMS]`` array (torch.Tensor or
                np.ndarray); columns follow ``AgentConfig.to_row()`` order. See
                :func:`wmas.dynamics.base.per_env_float_template` for a template.
                ``n_envs`` must match the batch size passed to the step.
        """
        is_torch = hasattr(floats, "detach")
        shape = tuple(floats.shape)
        if len(shape) != 3 or shape[1:] != (self.n_agents, NUM_PARAMS):
            raise ValueError(
                f"per-env params must have shape [n_envs, n_agents={self.n_agents}, "
                f"NUM_PARAMS={NUM_PARAMS}]; got {shape}"
            )
        if self.collisions:
            if is_torch:
                max_r = float(floats[..., P_RADIUS].max())
            else:
                max_r = float(np.asarray(floats)[..., P_RADIUS].max())
            reach = 2.0 * max_r + self.world.collision_margin
            if self.neighbor_radius < reach:
                raise ValueError(
                    f"per-env radius up to {max_r} needs neighbor_radius >= {reach} "
                    f"(= 2 * max per-env radius + margin), but neighbor_radius="
                    f"{self.neighbor_radius}; set WorldConfig.neighbor_radius accordingly."
                )
        torch_dt = self.torch_dtype
        existing = self.params.floats_per_env
        # In-place refresh (no numpy round-trip, no realloc, no version bump) when
        # an on-device torch buffer of the matching shape/dtype is already
        # installed — the per-reset re-randomization path in graph mode.
        if (
            is_torch
            and existing is not None
            and tuple(existing.shape) == shape
            and floats.dtype == torch_dt
            and str(floats.device) == str(self.device)
        ):
            wp.copy(existing, wp.from_torch(floats.contiguous(), dtype=self.dtype))
            return
        # (Re)allocate. Build on-device from torch when possible to avoid a host
        # round-trip; numpy inputs go through torch once.
        if is_torch:
            t = floats.detach().to(device=self.device, dtype=torch_dt).contiguous()
        else:
            t = torch.as_tensor(np.asarray(floats), dtype=torch_dt, device=self.device)
        first_install = existing is None
        self.params.floats_per_env = wp.clone(wp.from_torch(t, dtype=self.dtype))
        if first_install:
            self.mutation_version += 1

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
                uniform_bins=self.world.uniform_bins,
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
        return StepBuffers(
            inter_states=inter,
            forces=forces,
            nbr_idx=nbr_idx,
            nbr_cnt=nbr_cnt,
            taped=requires_grad,
        )

    def cached_buffers(self, n_envs: int) -> StepBuffers:
        """Recycled scratch for the tape-free hot path (zero steady-state allocs)."""
        bufs = self._cached_buffers.get(n_envs)
        if bufs is None:
            bufs = self.make_buffers(n_envs, requires_grad=False)
            self._cached_buffers[n_envs] = bufs
        return bufs

    def output_state(self, n_envs: int) -> WorldState:
        """Ping-pong output buffer for the tape-free hot path (two states cycled
        per batch size), so steady-state stepping allocates no output arrays —
        also the fixed buffer a future CUDA-graph capture needs.

        The returned tensors stay valid until this method is called twice more
        for the same ``n_envs`` (the two-buffer cycle guarantees a step's input
        and output never alias). The grad path never uses this — it allocates a
        fresh output per step for tape correctness.
        """
        states = self._out_states.get(n_envs)
        if states is None:
            states = [self.alloc_state(n_envs), self.alloc_state(n_envs)]
            self._out_states[n_envs] = states
            self._out_idx[n_envs] = 0
            for st in states:
                views = [wp.to_torch(a, requires_grad=False) for a in st.arrays()]
                self._slot_views[id(st)] = views
                self._wrapped_by_pos[id(views[0])] = st  # views[0] is pos
        i = self._out_idx[n_envs]
        self._out_idx[n_envs] = 1 - i
        return states[i]

    def wrapped_views(self, state: WorldState) -> list[torch.Tensor]:
        """Cached, stable zero-copy torch views of an output slot's arrays."""
        return self._slot_views[id(state)]

    def lookup_wrapped(self, state) -> WorldState | None:
        """The wrapped :class:`WorldState` backing ``state`` if it is one of the
        cached output slots (O(1) identity check on ``state.pos``), else None."""
        return self._wrapped_by_pos.get(id(state.pos))

    def wrap_actions(self, actions: torch.Tensor, scalar) -> wp.array:
        """Zero-copy Warp view of the action tensor, reusing the wrap when the
        underlying buffer is unchanged (the hot loop re-fills one buffer)."""
        ac = actions.contiguous()
        key = (ac.data_ptr(), tuple(ac.shape), scalar)
        cached = self._action_wrap
        if cached is not None and cached[0] == key:
            return cached[1]
        arr = wp.from_torch(ac, dtype=scalar, requires_grad=False)
        # Hold ``ac`` alive so its storage (and thus data_ptr) cannot be recycled
        # into a different tensor that would then spuriously hit this cache.
        self._action_wrap = (key, arr, ac)
        return arr

    # ---------------------------------------------------------------- stepping

    def launch_substeps(
        self,
        state_in: WorldState,
        actions: wp.array,
        state_out: WorldState,
        buffers: StepBuffers,
        reuse_neighbors: bool = False,
        skip_drone: bool = False,
    ) -> None:
        """Advance one full env step. Functional: ``state_in`` is never written.

        With ``reuse_neighbors=True`` (the no-grad hot path) substep 0 skips its
        neighbor rebuild and reads the grid's existing lists when they were built
        on this exact input state (``grid.built_version == state_version``) — the
        list the previous step's post-step build already produced. The taped path
        always passes ``False``: it must rebuild into fresh per-substep buffers so
        the recorded neighbor lists survive until backward (the grid's own lists
        get overwritten by the next build before then).
        """
        n_envs = state_in.pos.shape[0]
        chain = [state_in, *buffers.inter_states, state_out]
        world = self.world
        slim = (
            self.enable_slim2d
            and not self.has_drone
            and not buffers.taped
            and world.integrator == Integrator.EULER
        )
        grid = self.grid(n_envs) if self.collisions else None
        can_reuse = (
            reuse_neighbors
            and self.neighbor_reuse
            and grid is not None
            and grid.built_version == self.state_version
        )
        for k in range(self.substeps):
            st = chain[k]
            forces = buffers.forces[k]
            if self._needs_forces:
                if self.collisions:
                    if k == 0 and can_reuse:
                        # Bit-identical to a fresh build on st.pos (same grid,
                        # radius, positions); read the grid's lists directly.
                        idx, cnt = grid.neighbor_idx, grid.neighbor_count
                    else:
                        idx, cnt = buffers.nbr_idx[k], buffers.nbr_cnt[k]
                        grid.query_into(st.pos, idx, cnt)
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
                    self._obs_type,
                    self._obs_angle,
                    self._obs_half,
                    self._obs_vel,
                    self._obs_ang_vel,
                    self.n_obstacles,
                    world.collision_k,
                    world.collision_c,
                    world.collision_margin,
                    self.sub_dt,
                    world.contact_max_overlap,
                    self._soft_walls,
                    self._bounds_min,
                    self._bounds_max,
                    forces,
                    self.dtype,
                )
                if self._any_movable and not buffers.taped:
                    # Same state the force pass just used, so action and reaction match;
                    # advancing here (not once per env step) keeps the pose an agent
                    # collides against at most one substep old.
                    #
                    # Skipped on a taped step: body state is advanced in place with
                    # record_tape=False, which would corrupt the adjoint of an array the
                    # tape recorded. A scenario that wants gradients through a movable
                    # body integrates it itself in torch on the grad path (PushTScenario).
                    launch_obstacle_dynamics(
                        st.pos,
                        st.vel,
                        self.params.floats,
                        self,
                        self.n_agents,
                        world.collision_k,
                        world.collision_c,
                        world.collision_margin,
                        self.sub_dt,
                        world.contact_max_overlap,
                        world.obstacle_linear_damping,
                        world.obstacle_angular_damping,
                        self._bounds_min,
                        self._bounds_max,
                        world.bounds is not None,
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
                integrator=world.integrator,
                slim=slim,
                skip_drone=skip_drone and slim,
            )
        # State has advanced; the grid's lists no longer match this generation.
        self.state_version += 1
