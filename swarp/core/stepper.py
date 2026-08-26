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

from swarp.core.bodies import launch_obstacle_dynamics
from swarp.core.collisions import launch_collision_forces
from swarp.core.config import Obstacles, WorldConfig
from swarp.core.neighbors import NeighborGrid
from swarp.core.state import VEC2, WorldState
from swarp.dynamics.base import (
    NUM_PARAMS,
    P_RADIUS,
    AgentConfig,
    AgentParams,
    DynamicsModel,
    Integrator,
    build_agent_params,
)
from swarp.dynamics.kernels import launch_integrate


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
        # Note the asymmetry with WorldConfig()'s own collisions=True default: a bare
        # Stepper is the dynamics-only integrator the dynamics tests drive, so it stays
        # collision-free unless a config asks otherwise. World always passes one through.
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

        # Obstacle arrays. Public: the kernel wrappers in swarp.core.bodies and the
        # scenario/render layers read them directly. Written only by
        # :meth:`set_obstacles`; dummies here keep the kernel signatures fixed at
        # zero obstacles.
        #
        # Pose is per-env — obs_pos [n_envs, n_obs] and obs_angle [n_envs, n_obs]
        # — so a body that moves and *rotates* independently in each env (a
        # scenario-layer movable body, e.g. PushTScenario's T) can be installed
        # as obstacles. Extent attributes (type/radius/half) are env-independent
        # [n_obs]. Default shape is a circle (type 0).
        self.n_obstacles = 0
        self.obs_pos = wp.zeros((1, 1), dtype=vec2, device=device)
        self.obs_radius = wp.zeros(1, dtype=dtype, device=device)
        self.obs_type = wp.zeros(1, dtype=wp.int32, device=device)
        self.obs_angle = wp.zeros((1, 1), dtype=dtype, device=device)
        self.obs_half = wp.zeros(1, dtype=vec2, device=device)
        # Obstacle velocities, so contact damping can use the closing velocity even for
        # a moving body installed as obstacles. Zero for a genuinely static obstacle.
        self.obs_vel = wp.zeros((1, 1), dtype=vec2, device=device)
        self.obs_ang_vel = wp.zeros((1, 1), dtype=dtype, device=device)
        # IMMOVABLE (0) vs MOVABLE (1) per obstacle; a movable body carries mass/inertia
        # and is integrated per substep from the reaction of the agent contacts.
        self.obs_kind = wp.zeros(1, dtype=wp.int32, device=device)
        # ---- the body group. Read only when ``any_movable``; see set_obstacles.
        self.obs_mass = wp.ones(1, dtype=dtype, device=device)
        self.obs_inertia = wp.ones(1, dtype=dtype, device=device)
        # Compound bodies: several shapes rigidly sharing one pose. obs_body[s] names the
        # body a shape belongs to (its lowest-index shape) and obs_body_off[s] places the
        # shape in the body frame; the body's own state lives in body_* at the root index.
        self.obs_body = wp.zeros(1, dtype=wp.int32, device=device)
        self.obs_body_off = wp.zeros(1, dtype=vec2, device=device)
        self.body_pos = wp.zeros((1, 1), dtype=vec2, device=device)
        self.body_angle = wp.zeros((1, 1), dtype=dtype, device=device)
        self.body_vel = wp.zeros((1, 1), dtype=vec2, device=device)
        self.body_ang_vel = wp.zeros((1, 1), dtype=dtype, device=device)
        self.any_movable = False

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

    def set_obstacles(self, obstacles: Obstacles) -> None:
        """Install (or refresh) the obstacle set described by ``obstacles``.

        The spec is normalized once (:meth:`~swarp.core.config.Obstacles.resolve`) and then
        written by a single install path in one of two regimes:

        * **in place** — the obstacle count *and* the batch size are unchanged. Every array
          is refreshed with a device-to-device copy or a fill: no allocation, no cached
          buffer invalidation and no ``mutation_version`` bump, so a scenario that
          re-samples obstacle poses every reset does not force a CUDA-graph recapture.
        * **reallocate** — first install, or the obstacle count or ``n_envs`` changed. Fresh
          arrays, cached step buffers dropped (``_needs_forces`` may have flipped, and a
          shape change invalidates any captured graph), ``mutation_version`` bumped.

        Both regimes give the *same* result: an absent field always means the documented
        default (see :class:`~swarp.core.config.Obstacles`), never "keep the previous
        value". To keep previous values, retain the :class:`~swarp.core.config.Obstacles`
        and re-install it — which is what :class:`~swarp.core.world.World` does.

        One asymmetry is deliberate: the **body group** (``obs_mass``/``obs_inertia``/
        ``obs_body``/``obs_body_off`` and the derived ``body_*`` state) is refreshed in
        place only when at least one obstacle is ``MOVABLE``. Nothing reads it otherwise,
        and deriving it costs torch temporaries and a host read.

        **Capture safety.** The in-place regime is allocation-free and host-read-free
        exactly when the spec carries no movable obstacles (``kind`` is ``None`` or all
        immovable) and its ``angle`` is already 2-D or absent — the regime
        :class:`~swarp.scenarios.transport.TransportScenario` re-installs from *inside* the
        captured whole-step graph. An install with movable bodies derives the body group in
        torch and reads ``any_movable`` back to the host, so it must stay outside a
        capture. Pinned by ``tests/unit/test_obstacles.py``.
        """
        obs = obstacles.resolve(self.device, self.torch_dtype)
        n_envs, n_obs = obs.n_envs, obs.n_obstacles
        # n_envs is part of the test: wp.copy sizes itself from the *source*, so a smaller
        # batch would silently copy into the head of the array and leave a stale tail.
        in_place = (
            n_obs > 0 and n_obs == self.n_obstacles and tuple(self.obs_pos.shape) == (n_envs, n_obs)
        )
        self._install(obs, in_place=in_place)

    def _install(self, obs: Obstacles, *, in_place: bool) -> None:
        """Write a resolved obstacle set into the Warp arrays (the single install path)."""
        vec2 = VEC2[self.dtype]
        dtype, dev = self.dtype, self.device
        n_envs, n_obs = obs.n_envs, obs.n_obstacles

        def write(name, src, wp_dtype, shape, fill=0.0) -> None:
            """Install one field: ``src`` if given, else ``fill`` everywhere.

            In place this is a ``wp.copy`` / memset / fill — no allocation, so it is legal
            inside a graph capture. On a reallocation it clones into a fresh array; ``src``
            is already on ``self.device``, which is what keeps a CPU input to a CUDA
            stepper from producing a mixed-device obstacle set.
            """
            if in_place:
                arr = getattr(self, name)
                if src is None:
                    if fill == 0.0:
                        arr.zero_()
                    else:
                        arr.fill_(fill)
                else:
                    wp.copy(arr, wp.from_torch(src, dtype=wp_dtype))
                return
            if src is not None:
                arr = wp.clone(wp.from_torch(src, dtype=wp_dtype))
            elif fill == 0.0:
                arr = wp.zeros(shape, dtype=wp_dtype, device=dev)
            else:
                arr = wp.full(shape, wp_dtype(fill), dtype=wp_dtype, device=dev)
            setattr(self, name, arr)

        write("obs_pos", obs.pos, vec2, (n_envs, n_obs))
        write("obs_radius", obs.radius, dtype, n_obs)
        write("obs_type", obs.shape, wp.int32, n_obs)
        write("obs_angle", obs.angle, dtype, (n_envs, n_obs))
        write("obs_half", obs.half_extents, vec2, n_obs)
        write("obs_vel", obs.vel, vec2, (n_envs, n_obs))
        write("obs_ang_vel", obs.ang_vel, dtype, (n_envs, n_obs))
        write("obs_kind", obs.kind, wp.int32, n_obs)
        self.any_movable = obs.any_movable
        if not in_place:
            self.n_obstacles = n_obs

        if self.any_movable or not in_place:
            write("obs_mass", obs.mass, dtype, n_obs, fill=1.0)
            write("obs_inertia", obs.inertia, dtype, n_obs, fill=1.0)
            ids, off, b_pos, b_ang, b_vel, b_ang_vel = self._body_seed(obs)
            write("obs_body", ids, wp.int32, n_obs)
            write("obs_body_off", off, vec2, n_obs)
            write("body_pos", b_pos, vec2, (n_envs, n_obs))
            write("body_angle", b_ang, dtype, (n_envs, n_obs))
            write("body_vel", b_vel, vec2, (n_envs, n_obs))
            write("body_ang_vel", b_ang_vel, dtype, (n_envs, n_obs))

        if not in_place:
            self._cached_buffers.clear()
            self.mutation_version += 1

    def _body_seed(self, obs: Obstacles) -> tuple[torch.Tensor, ...]:
        """``(ids, offsets, body pos, angle, vel, ang_vel)`` as contiguous torch tensors.

        The body origin is derived from its root shape (``root_pos - R(angle) @ off``) and
        its velocity is the root shape's, so a caller only ever has to describe where the
        *shapes* are — writing the shapes' world poses hands over the whole body state,
        which is how a reset (or a scenario that integrated the body itself on a grad step)
        makes the engine continue from there.

        All torch, so it allocates: never call this from inside a graph capture.
        """
        dev, td = self.device, self.torch_dtype
        n_envs, n_obs = obs.n_envs, obs.n_obstacles
        ids = (
            torch.arange(n_obs, device=dev, dtype=torch.int32) if obs.body is None else obs.body
        )
        off = (
            torch.zeros(n_obs, 2, device=dev, dtype=td)
            if obs.body_offset is None
            else obs.body_offset
        )
        root = ids.long()
        r_ang = (
            torch.zeros(n_envs, n_obs, device=dev, dtype=td)
            if obs.angle is None
            else obs.angle[:, root].contiguous()
        )
        r_off = off[root]  # [n_obs, 2]
        ca, sa = torch.cos(r_ang), torch.sin(r_ang)
        wx = ca * r_off[:, 0] - sa * r_off[:, 1]
        wy = sa * r_off[:, 0] + ca * r_off[:, 1]
        b_pos = (obs.pos[:, root] - torch.stack([wx, wy], dim=-1)).contiguous()
        b_vel = (
            torch.zeros(n_envs, n_obs, 2, device=dev, dtype=td)
            if obs.vel is None
            else obs.vel[:, root].contiguous()
        )
        b_ang_vel = (
            torch.zeros(n_envs, n_obs, device=dev, dtype=td)
            if obs.ang_vel is None
            else obs.ang_vel[:, root].contiguous()
        )
        return ids, off, b_pos, r_ang, b_vel, b_ang_vel

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
                :func:`swarp.dynamics.base.per_env_float_template` for a template.
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
        *,
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
                    self.obs_pos,
                    self.obs_radius,
                    self.obs_type,
                    self.obs_angle,
                    self.obs_half,
                    self.obs_vel,
                    self.obs_ang_vel,
                    forces,
                    n_obstacles=self.n_obstacles,
                    k=world.collision_k,
                    c=world.collision_c,
                    margin=world.collision_margin,
                    sub_dt=self.sub_dt,
                    max_overlap=world.contact_max_overlap,
                    soft_walls=self._soft_walls,
                    bounds_min=self._bounds_min,
                    bounds_max=self._bounds_max,
                    dtype=self.dtype,
                )
                if self.any_movable and not buffers.taped:
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
                        n_agents=self.n_agents,
                        k=world.collision_k,
                        c=world.collision_c,
                        margin=world.collision_margin,
                        sub_dt=self.sub_dt,
                        max_overlap=world.contact_max_overlap,
                        lin_damping=world.obstacle_linear_damping,
                        ang_damping=world.obstacle_angular_damping,
                        bounds_min=self._bounds_min,
                        bounds_max=self._bounds_max,
                        clamp_bounds=world.bounds is not None,
                        dtype=self.dtype,
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
