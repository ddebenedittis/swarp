"""Give way at a one-lane intersection: the repo's only scenario that can deadlock.

Four ``IMMOVABLE`` ``BOX`` obstacles fill the corner quadrants of the arena, leaving a
plus-shaped free space — two corridors crossing at a junction. One agent starts at the end
of each arm and must reach the opposite arm. The corridor is deliberately sized to fit
**one** agent and not two abreast (``agent_radius < half_w < 2 * agent_radius``), so two
agents meeting head-on in an arm cannot pass. The only feasible passing bay is the
*perpendicular* arm: one agent ducks into it while the other crosses the junction, then
resumes. That makes this the only built-in scenario whose solution requires moving **away**
from your own goal, and the only one where a symmetric joint policy is not merely
suboptimal but a hard deadlock.

Reward (per-agent, with the repo's ``prev - cur`` shaping convention):

  * position shaping ``(prev_dist - dist) * pos_shaping_factor``,
  * agent-contact penalty ``collision_penalty * touching_count``,
  * wall-contact penalty ``wall_penalty`` while within ``wall_margin`` of a block,
  * a per-step ``time_penalty``,
  * ``final_reward`` once **all** agents are on their goals.

``shared_reward`` defaults to **True** here, unlike navigation: the yielding agent's
manoeuvre costs it shaping and buys the *team* the crossing, so under per-agent shaping the
agent that solves the task is the one that gets punished for it. Both settings are kept
(and both are parity-tested) because the per-agent variant is a useful ablation, not
because it is a reasonable default.

Observation per agent: own pos, vel, heading (cos/sin), angular velocity, goal-relative
vector, **politeness**, and features of up to ``neighbor_obs`` within-radius neighbors
(relative position/velocity + validity mask, padded-list order).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from swarp._overloads import concrete
from swarp.core.cached_launch import CachedLaunch, ptr_key
from swarp.core.config import ObstacleKind, Obstacles, ObstacleShape, WorldConfig
from swarp.core.rng import advance_seed_kernel
from swarp.core.state import VEC2
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.interop.autograd import torch_stream_scope
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario
from swarp.scenarios.giveway_kernels import (
    N_OBJ,
    giveway_obs_kernel,
    giveway_reset_kernel,
    giveway_reward_kernel,
)


class GiveWayScenario(FusedScenario):
    def __init__(
        self,
        n_agents: int = 4,
        agent_radius: float = 0.05,
        max_speed: float = 1.0,
        world_size: float = 1.0,
        corridor_half_width: float | None = None,
        slot_spacing: float | None = None,
        long_jitter: float | None = None,
        lat_jitter: float | None = None,
        neighbor_obs: int = 3,
        neighbor_radius: float | None = None,
        shared_reward: bool = True,
        pos_shaping_factor: float = 1.0,
        collision_penalty: float = -1.0,
        wall_penalty: float = -0.02,
        wall_margin: float | None = None,
        time_penalty: float = -0.01,
        final_reward: float = 5.0,
        goal_tolerance: float | None = None,
        contact_k: float = 8000.0,
        contact_c: float = 80.0,
        neighbor_method: str = "auto",
    ) -> None:
        self.n_agents = n_agents
        self.agent_radius = agent_radius
        self.max_speed = max_speed
        self.world_size = world_size
        self.neighbor_obs = neighbor_obs
        self.neighbor_radius = neighbor_radius
        self.shared_reward = shared_reward
        self.pos_shaping_factor = pos_shaping_factor
        self.collision_penalty = collision_penalty
        self.wall_penalty = wall_penalty
        self.time_penalty = time_penalty
        self.final_reward = final_reward
        self.goal_tolerance = goal_tolerance if goal_tolerance is not None else agent_radius
        self.contact_k = contact_k
        self.contact_c = contact_c
        self.neighbor_method = neighbor_method

        # ---- corridor geometry, and the one constraint the whole task rests on
        self.half_w = (
            1.5 * agent_radius if corridor_half_width is None else float(corridor_half_width)
        )
        if not (agent_radius < self.half_w < 2.0 * agent_radius):
            raise ValueError(
                "give-way needs a corridor that fits exactly one agent: "
                f"agent_radius ({agent_radius}) < corridor_half_width ({self.half_w}) < "
                f"2 * agent_radius ({2.0 * agent_radius}). At or below the lower bound the "
                "agent does not fit at all; at or above the upper bound two agents pass "
                "abreast and there is no give-way problem left to solve."
            )
        # Corner blocks: each spans [half_w, world_size]^2 in its own quadrant, so the
        # free space is the plus shape {|x| <= half_w} u {|y| <= half_w}.
        self._block_half = 0.5 * (world_size - self.half_w)
        self._block_center = 0.5 * (self.half_w + world_size)
        # NOTE ON MAGNITUDES. ``wall_penalty`` is deliberately two orders below
        # ``collision_penalty``, which looks wrong until you measure it. In a corridor
        # 1.5 agent-diameters wide, wall proximity is not a mistake an agent can avoid --
        # a random policy has ``wall_contact`` firing on ~55% of agent-steps. At the -0.5
        # that mirrors ``collision_penalty`` the term averages -0.28/step against a
        # position-shaping term of +0.012, i.e. it outweighs the actual objective 22x and
        # totals ~-55 over a 200-step episode against a +5 arrival bonus. The optimum is
        # then to stand still in mid-corridor and never attempt the crossing. The wall term
        # is here to discourage *grinding along* a wall, not to price the corridor itself,
        # so it belongs at the scale of the per-step time penalty. Watch
        # ``info()["multiobj_reward"]`` if you retune it.
        #
        # Deliberately tight. The flag fires at ``gap < agent_radius + wall_margin``, and
        # in a corridor whose spare lateral room is only ``half_w - agent_radius`` (0.025
        # at the defaults) a generous margin makes the penalty fire almost every step for
        # almost every agent — a constant offset the policy cannot act on, rather than a
        # signal. A tenth of a radius keeps it meaning "scraping the wall".
        self.wall_margin = 0.1 * agent_radius if wall_margin is None else float(wall_margin)

        # ---- queue geometry along each arm
        self.slot_spacing = 3.0 * agent_radius if slot_spacing is None else float(slot_spacing)
        self.n_slots = max(1, math.ceil(n_agents / 4))
        # Outermost slot sits a full *diameter* inside the arena edge, so the agent's disc
        # clears the boundary by one radius rather than resting flush against it. One
        # radius of centre clearance would leave the disc exactly touching the wall, which
        # under this scenario's stiff contact and ``bounds_mode="clamp"`` is a spawn-time
        # boundary contact rather than a free start -- and it is what
        # ``tests/unit/test_reset_kernel.py::test_reset_spawns_inside_the_world`` pins for
        # every scenario.
        self._arm_len = world_size - 2.0 * agent_radius
        # Closest an agent may spawn to the origin: any nearer and it is standing *in* the
        # junction rather than queued in an arm.
        self._min_dist = self.half_w + agent_radius
        innermost = self._arm_len - (self.n_slots - 1) * self.slot_spacing
        if innermost <= self._min_dist:
            raise ValueError(
                f"{self.n_slots} slots of spacing {self.slot_spacing} do not fit in an arm of "
                f"length {self._arm_len} (innermost slot would land at {innermost}, inside the "
                f"junction at {self._min_dist}); raise world_size or lower slot_spacing/n_agents"
            )
        # Longitudinal jitter staggers arrival at the junction; it may not push the
        # innermost slot into the junction itself.
        cap = innermost - self._min_dist
        self.long_jitter = cap if long_jitter is None else min(float(long_jitter), cap)
        # Lateral jitter offsets the spawn across the corridor. Beyond
        # ``half_w - agent_radius`` the agent's disc overlaps a block at t=0, which under
        # this scenario's stiff contact is a spawn-time impulse, not a nudge.
        lat_cap = self.half_w - agent_radius
        self.lat_jitter = lat_cap if lat_jitter is None else float(lat_jitter)
        if not (0.0 <= self.lat_jitter <= lat_cap):
            raise ValueError(
                f"lat_jitter must lie in [0, half_w - agent_radius] = [0, {lat_cap}]; "
                f"got {self.lat_jitter} — a larger offset spawns an agent inside a block"
            )

        self._difficulty = 1.0
        # Set by ``make_world``; ``None`` until then so the property setter can be used
        # before there is a device to write to.
        self._long_jitter_t: torch.Tensor | None = None

        # Provisional obs width so ``obs_dim`` answers before ``make_world``: it mirrors
        # this scenario's own ``max_neighbors`` default. ``make_world`` re-derives it from
        # the *resolved* config, so a ``world_config`` override still wins.
        self._k_obs = min(self.neighbor_obs, self._max_neighbors())

    def _max_neighbors(self) -> int:
        return min(32, max(4, self.n_agents))

    # -------------------------------------------------------------- curriculum

    @property
    def difficulty(self) -> float:
        """Curriculum knob a trainer pokes between batches (Push-T's mechanism).

        Mapped onto the longitudinal stagger: ``0`` spreads the spawns down the arms so
        agents reach the junction one at a time and there is no conflict to resolve, ``1``
        pins every agent to its nominal slot for a simultaneous four-way conflict.
        Clamped to ``[0, 1]``.

        A property rather than a plain attribute for one reason, and it is not
        cosmetic: this scenario folds its reset into the captured whole-step graph
        (:meth:`reset_in_graph`), and a *scalar* kernel argument is baked into that
        capture by value. So the derived stagger lives in a one-element device tensor the
        reset kernel reads by pointer, and this setter is what keeps it in step with the
        Python value. Assigning the attribute therefore still "just works" — including
        mid-training, with a graph already captured — where a bare float would have
        silently stopped taking effect after the first capture.
        """
        return self._difficulty

    @difficulty.setter
    def difficulty(self, value: float) -> None:
        self._difficulty = min(max(float(value), 0.0), 1.0)
        if self._long_jitter_t is not None:
            # One tiny host->device write per curriculum change (not per step): a
            # ``fill_`` rather than an indexed assignment so nothing reads back.
            self._long_jitter_t.fill_((1.0 - self._difficulty) * self.long_jitter)

    # ------------------------------------------------------------------ world

    def _agent_configs(self) -> list[AgentConfig]:
        return [
            AgentConfig(
                model=DynamicsModel.HOLONOMIC,
                ctrl_mode=ControlMode.VELOCITY,
                radius=self.agent_radius,
                max_speed=self.max_speed,
                max_accel=2.0 * self.max_speed,
            )
            for _ in range(self.n_agents)
        ]

    def make_world(self, n_envs, device, dt, substeps, dtype, world_config=None) -> World:
        configs = self._agent_configs()
        # Small relative to the corridor's spare lateral room (``half_w - agent_radius``,
        # 0.025 at the defaults): the margin inflates the agent for contact purposes, so a
        # margin near that figure would leave an agent pushed against the centre line for
        # the whole episode.
        margin = 0.1 * self.agent_radius
        # Push-T's derivation. A velocity-mode agent has no contact memory: each substep
        # its velocity is overwritten by the command plus f*sub_dt/m, so holding it out of
        # a block needs f >= m*max_speed/sub_dt, i.e. an equilibrium penetration of
        # max_speed/(k*sub_dt). At the engine's default collision_k=100 with dt=0.05,
        # substeps=1 that depth is several agent radii and the corridor walls would be a
        # suggestion rather than a wall — hence contact_k=8000 by default, and hence
        # substeps >= 8 at dt=0.05 as the practical floor (sub_dt shrinks the depth
        # linearly). Saturating at twice the equilibrium depth bounds the impulse a deep
        # sweep can inject while leaving the holding force intact.
        sub_dt = dt / max(1, substeps)
        self.max_overlap = 2.0 * self.max_speed / (self.contact_k * sub_dt)
        cfg = WorldConfig(
            collisions=True,
            collision_k=self.contact_k,
            collision_c=self.contact_c,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            # "clamp", not the inherited "soft": the arms are open-ended at the arena
            # boundary, and a stiff block contact resolved against a soft wall can squirt
            # an agent clean out of the corridor. A hard clamp cannot.
            bounds_mode="clamp",
            # NOT navigation's contact-range default. Navigation sets neighbor_radius to
            # 2*agent_radius + margin, i.e. an agent only observes neighbours it is
            # *already touching* — fine for reactive avoidance, fatal here: give-way is a
            # negotiation that has to start while the agents are still a corridor apart.
            # With a contact-range radius the observation carries no opposing agent until
            # the moment passing is already impossible, and the task looks unlearnable for
            # a reason that has nothing to do with the reward.
            neighbor_radius=max(self.neighbor_radius or 0.0, 0.75 * self.world_size),
            max_neighbors=self._max_neighbors(),
            neighbor_method=self.neighbor_method,
            contact_max_overlap=self.max_overlap,
        ).override_with(world_config)
        self.world = World(
            configs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        self._k_obs = min(self.neighbor_obs, cfg.max_neighbors)
        self._nbr_cache: dict[str, torch.Tensor] | None = None
        # Position-shaping baseline; left None so the first refresh seeds it from the spawn
        # distance (shaping 0) rather than from zeros. The spec adopts it with
        # alloc="if_none", and watch=True catches the torch path reassigning it.
        self._prev_dist: torch.Tensor | None = None
        # Cached, repack-once launches for the eager paths, plus separate instances for the
        # in-graph reset tail — the same split navigation documents: a captured launch is
        # packed once at warm-up and never touched again, and sharing a cache with the
        # eager path that repacks every step entangles two independent lifecycles.
        self._seed_launch = CachedLaunch()
        self._reset_launch = CachedLaunch()
        self._obs_launch = CachedLaunch()
        self._seed_launch_g = CachedLaunch()
        self._reset_launch_g = CachedLaunch()
        self._obs_launch_g = CachedLaunch()

        tt = {"device": device, "dtype": dtype}
        # Goals, written in place by every reset; allocated here so the fused spec adopts
        # them rather than racing the first reset.
        self.world.goals = torch.zeros(n_envs, self.n_agents, 2, **tt)
        # ---- the single most important design decision in this scenario ----
        # A shared-weight actor fed a purely ego-centric/relative observation row emits
        # *identical* actions for two agents in mirror-image situations. In a corridor that
        # fits one agent, identical behaviour is not a tie to be broken later — it *is* the
        # deadlock: both advance, both stall, neither yields, forever. So the observation
        # carries a per-agent scalar in [0, 1) drawn fresh at every reset. It is the
        # coordinate on which the policy can condition a yield/go convention ("lower
        # politeness goes first") without any explicit communication channel, and it is
        # re-drawn per episode so the convention has to be a *function* of the scalar
        # rather than of an agent index the policy never sees. Remove it and the task is
        # not merely harder, it is unsolvable by a symmetric policy.
        self.politeness = torch.zeros(n_envs, self.n_agents, **tt)
        # The curriculum stagger, by pointer — see :attr:`difficulty`. Allocated once and
        # never reallocated, so the handle stays valid across captures; re-assigning
        # ``difficulty`` through the property is what fills it.
        self._long_jitter_t = torch.zeros(1, **tt)
        self._long_jitter_wp = wp.from_torch(self._long_jitter_t, dtype=self.world.wp_dtype)
        self.difficulty = self._difficulty  # sync the device scalar to the current value

        # The four corner blocks. Installed **once**, here, and never redrawn — which is
        # exactly what makes ``supports_graph_reset`` possible below, where navigation's
        # obstacle path cannot follow (it re-randomizes poses from a torch.Generator every
        # reset). ``radius`` is zeros: a BOX's surface is its boundary and ignores it.
        c = self._block_center
        centers = torch.tensor(
            [[c, c], [-c, c], [-c, -c], [c, -c]], **tt
        ).unsqueeze(0).expand(n_envs, 4, 2)
        self.world.set_obstacles(
            Obstacles(
                centers.contiguous(),
                torch.zeros(4, **tt),
                shape=torch.full((4,), int(ObstacleShape.BOX), device=device, dtype=torch.int32),
                angle=torch.zeros(4, **tt),
                half_extents=torch.full((4, 2), self._block_half, **tt),
                kind=torch.full(
                    (4,), int(ObstacleKind.IMMOVABLE), device=device, dtype=torch.int32
                ),
            )
        )
        return self.world

    @property
    def obs_dim(self) -> int:
        return 10 + 5 * self._k_obs

    # ------------------------------------------------------------------ reset

    def _launch_reset(self, env_mask: torch.Tensor | None, *, in_graph: bool = False) -> None:
        """Launch the masked device-side reset (see :meth:`reset_world`).

        ``in_graph=True`` is the tail :meth:`reset_in_graph` runs from inside the captured
        whole-step graph. It differs from the eager call in exactly the three ways
        :meth:`~swarp.scenarios.navigation.NavigationScenario._launch_reset` documents at
        length: the mask comes straight off the fused reset-mask buffer (the episode-end
        kernel earlier in the same ``run`` already stamped it), :meth:`sync_fused_handles`
        is skipped (host-side, and ``prepare_fused`` already ran it), and the launches are
        not wrapped in :func:`torch_stream_scope` (the capture region records on Warp's own
        stream). It also skips ``mark_pos_dirty``: that bumps a host-side version counter
        read once per *eager* step, and the grid the next replay reads is refreshed by
        :meth:`reset_in_graph`'s following masked ``_launch_obs``.
        """
        w = self.world
        if in_graph:
            mask, use_mask = self._wp[self._fused_mask], wp.int32(1)
        else:
            mask, use_mask = self.reset_mask_wp(env_mask)

        st = w.state_wp()
        if self.fused_active:
            # Reuse the spec's cached, pointer-resynced handles instead of re-wrapping
            # goals/politeness on every reset. ``sync_fused_handles`` must run first on
            # the eager path: it is what notices a grad-path reassignment and rebuilds the
            # handle before this kernel writes through it.
            self.ensure_fused()
            if not in_graph:
                self.sync_fused_handles()
            goals, polite = self._wp["goals"], self._wp["polite"]
        else:
            goals = wp.from_torch(w.goals.contiguous(), dtype=VEC2[w.wp_dtype])
            polite = wp.from_torch(self.politeness.contiguous(), dtype=w.wp_dtype)
        scalar = w.wp_dtype
        seed_cache = self._seed_launch_g if in_graph else self._seed_launch
        reset_cache = self._reset_launch_g if in_graph else self._reset_launch

        def _do_launches() -> None:
            # Advance, then use (see swarp/core/rng.py): bump World.seed_state's counter
            # first, so the reset kernel below reads the fresh value. Both launches share
            # the array by pointer only — ``seed_state`` never moves, which is what lets a
            # captured graph replay this pair with a different draw every time.
            seed_cache.get(
                advance_seed_kernel,
                dim=1,
                inputs=[w.seed_state],
                device=w.device,
                key=(ptr_key(w.seed_state),),
            ).launch()

            # ``use_mask`` is the only genuinely per-call *value*; everything else is
            # fixed geometry or a pointer-stable handle and lives in the cache key. The
            # curriculum stagger is among the handles on purpose (see :attr:`difficulty`):
            # its value can change at any time, so it is read out of device memory rather
            # than re-packed here, which is also what makes it survive graph capture.
            launch = reset_cache.get(
                concrete(giveway_reset_kernel, scalar),
                dim=w.n_envs,
                inputs=[
                    mask,
                    use_mask,
                    w.seed_state,
                    scalar(self._arm_len),
                    scalar(self.slot_spacing),
                    scalar(self._min_dist),
                    self._long_jitter_wp,
                    scalar(self.lat_jitter),
                    wp.int32(self.n_agents),
                    st.pos,
                    st.theta,
                    st.vel,
                    st.speed,
                    st.ang_vel,
                    goals,
                    polite,
                ],
                device=w.device,
                key=(
                    w.n_envs,
                    ptr_key(mask),
                    ptr_key(w.seed_state),
                    self._arm_len,
                    self.slot_spacing,
                    self._min_dist,
                    ptr_key(self._long_jitter_wp),
                    self.lat_jitter,
                    self.n_agents,
                    ptr_key(st.pos),
                    ptr_key(st.theta),
                    ptr_key(st.vel),
                    ptr_key(st.speed),
                    ptr_key(st.ang_vel),
                    ptr_key(goals),
                    ptr_key(polite),
                ),
            )
            launch.set_param_by_name("use_mask", use_mask)
            launch.launch()

        if in_graph:
            _do_launches()
        else:
            with torch_stream_scope(w.device):
                _do_launches()
            w.mark_pos_dirty()

    def reset_world(
        self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False
    ) -> None:
        """Reset all envs (``env_mask=None``) or the ``True`` entries of a boolean
        ``[n_envs]`` mask.

        Host-sync-free and entirely device-side: one masked Warp launch writes the queued
        spawns, the mirrored goals, the per-arm headings, the zeroed velocities and the
        fresh politeness draw. The obstacles are static, so unlike navigation there is
        nothing else to redraw — which is the whole reason :meth:`supports_graph_reset`
        can return ``True``.
        """
        self._launch_reset(env_mask)
        self._nbr_cache = None
        self.finish_reset(env_mask, obs_only=obs_only)

    # ------------------------------------------- capture-safe reset (auto_reset)

    def supports_graph_reset(self) -> bool:
        """Always ``True``: the whole reset is :meth:`_launch_reset`.

        Navigation returns ``False`` whenever it has obstacles, but only because it
        *re-randomizes* their poses on every reset — a ``torch.Generator`` draw plus a
        ``set_obstacles`` re-install, neither capture-safe. Give-way's four blocks are
        installed once in :meth:`make_world` and never move, so the reset reduces to pure
        Warp launches against pointer-stable buffers seeded from the device-resident
        ``World.seed_state``, which is exactly what the base class asks for.
        """
        return True

    def reset_in_graph(self) -> None:
        """The capture-safe reset tail: masked spawn/goal/politeness draw, then an
        obs-only refresh over the same mask.

        The mask was already stamped by the episode-end kernel earlier in the same ``run``
        (``Environment``'s composed :class:`~swarp.core.hooks.WholeStepHook`), so neither
        call needs — or is given — an explicit ``env_mask``.
        """
        self._launch_reset(env_mask=None, in_graph=True)
        self._launch_obs(advance_prev=0, full_pass=0, in_graph=True)

    # --------------------------------------------------------- fused fast path

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        ne, na = n_envs, self.n_agents
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("touch", (ne, na)),
            Buf("wall", (ne, na)),
            Buf("dist", (ne, na)),
            Buf("shaping", (ne, na)),
            Buf("reward", (ne, na)),
            Buf("multiobj", (ne, na, N_OBJ)),
            Buf("ongoal", (ne, na), "uint8", bool_view=True),
            Buf("done", (ne,), "uint8", bool_view=True),
            Buf("resetmask", (ne,), "uint8", reset_mask=True),
            Buf("goals", (ne, na, 2), "vec2", attr="world.goals", alloc="never", watch=True),
            # Scenario state, allocated in make_world and written by the reset kernel. Not
            # a carry: no fused *step* launch advances it, exactly like ``goals``.
            Buf("polite", (ne, na), attr="politeness", alloc="never"),
            Buf("prev", (ne, na), attr="_prev_dist", alloc="if_none", carry=True, watch=True),
        )

    def launch_fused(self, pass_: FusedPass) -> None:
        """Observations, then the reward — on every pass except an obs-only auto-reset.

        Gated on ``full_pass``, not ``is_step``: a mid-step auto-reset must leave
        ``reward``/``done``/``multiobj`` alone (they were already returned for the
        transition just taken), while a *standalone* ``reset``/``reset_at`` has to
        recompute them or the fused path reports the previous episode where the torch
        oracle reports the new one.
        """
        self._launch_obs(advance_prev=pass_.advance_prev, full_pass=pass_.full_pass)
        if pass_.full_pass:
            self._launch_reward()

    def _launch_obs(self, advance_prev: int, full_pass: int, *, in_graph: bool = False) -> None:
        w = self.world
        n_envs = w.n_envs
        resetmask = self._wp["resetmask"]
        # ``full_pass == 0`` happens only on the obs-only auto-reset pass, the one pass
        # where every env the mask does not select has positions unchanged since the step
        # pass built the lists moments earlier — so masking the rebuild there is safe and
        # can never be reached by a normal step or a full reset.
        w.build_neighbors(reset_mask=resetmask if full_pass == 0 else None)
        grid = w.stepper.grid(n_envs)
        scalar = w.wp_dtype
        st = w.state_wp()
        goals, polite, prev = self._wp["goals"], self._wp["polite"], self._wp["prev"]
        params = w.stepper.params.floats
        obs, touch, wall, dist, shaping = (
            self._wp["obs"],
            self._wp["touch"],
            self._wp["wall"],
            self._wp["dist"],
            self._wp["shaping"],
        )
        ongoal = self._wp["ongoal"]
        obs_cache = self._obs_launch_g if in_graph else self._obs_launch
        launch = obs_cache.get(
            concrete(giveway_obs_kernel, scalar),
            dim=(n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                st.theta,
                st.ang_vel,
                goals,
                polite,
                grid.neighbor_idx,
                grid.neighbor_count,
                params,
                resetmask,
                wp.int32(self._k_obs),
                scalar(self.half_w),
                scalar(self.world_size),
                scalar(self.agent_radius + self.wall_margin),
                scalar(self.pos_shaping_factor),
                scalar(self.goal_tolerance),
                wp.int32(advance_prev),
                wp.int32(full_pass),
            ],
            outputs=[obs, touch, wall, dist, shaping, ongoal, prev],
            device=w.device,
            key=(
                n_envs,
                self.n_agents,
                ptr_key(st.pos),
                ptr_key(st.vel),
                ptr_key(st.theta),
                ptr_key(st.ang_vel),
                ptr_key(goals),
                ptr_key(polite),
                ptr_key(grid.neighbor_idx),
                ptr_key(grid.neighbor_count),
                ptr_key(params),
                ptr_key(resetmask),
                self._k_obs,
                self.half_w,
                self.world_size,
                self.wall_margin,
                self.pos_shaping_factor,
                self.goal_tolerance,
                ptr_key(obs),
                ptr_key(touch),
                ptr_key(wall),
                ptr_key(dist),
                ptr_key(shaping),
                ptr_key(ongoal),
                ptr_key(prev),
            ),
        )
        launch.set_param_by_name("advance_prev", wp.int32(advance_prev))
        launch.set_param_by_name("full_pass", wp.int32(full_pass))
        launch.launch()

    def _launch_reward(self) -> None:
        w = self.world
        scalar = w.wp_dtype
        wp.launch(
            concrete(giveway_reward_kernel, scalar),
            dim=w.n_envs,
            inputs=[
                self._wp["touch"],
                self._wp["wall"],
                self._wp["shaping"],
                self._wp["ongoal"],
                wp.int32(self.n_agents),
                scalar(self.collision_penalty),
                scalar(self.wall_penalty),
                scalar(self.time_penalty),
                scalar(self.final_reward),
                wp.int32(1 if self.shared_reward else 0),
            ],
            outputs=[self._wp["reward"], self._wp["done"], self._wp["multiobj"]],
            device=w.device,
            record_tape=False,
        )

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh_step_cache()

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        self._refresh_step_cache(reset_mask=env_mask)

    def _wall_gap(self, pos: torch.Tensor) -> torch.Tensor:
        """Signed distance from each agent to the nearest corner block ``[n_envs, n_agents]``.

        The four blocks are mirror images about both axes, so folding the query into the
        first quadrant with ``abs`` reduces "nearest of four boxes" to one box SDF against
        the slab ``[half_w, world_size]`` per axis.

        This deliberately uses the *same* ``max(lo - a, a - hi)`` slab decomposition as
        :func:`~swarp.scenarios.giveway_kernels._wall_gap` rather than an equivalent
        ``|a - centre| - half`` form, for the reason formation's ``_refresh`` uses squared
        distances: ``wall_contact`` is a **discrete** flag compared *exactly* against the
        fused path, and two algebraically identical forms that round differently would
        flip it for an agent sitting on the threshold. The parity oracle's job is to be an
        independent implementation of the same *quantity*, not to reach it by a route
        chosen for looking different.
        """
        a = pos.abs()
        d = (self.half_w - a).maximum(a - self.world_size)  # per-axis slab distance
        return d.clamp(min=0.0).norm(dim=-1) + d.amax(dim=-1).clamp(max=0.0)

    def _refresh_step_cache(self, reset_mask: torch.Tensor | None = None) -> None:
        """Neighbor features, distances, contact flags and reward terms for the current state.

        ``reset_mask`` (a boolean ``[n_envs]`` or ``None``) marks envs that were just
        reset: their shaping baseline is rebased to the fresh spawn distance and their
        shaping term zeroed, while non-reset envs keep their carried-over baseline. That
        masked rebase is what keeps a partial/auto reset from clobbering other envs.
        """
        w = self.world
        pos, vel = w.state.pos, w.state.vel
        idx, cnt = w.neighbors()
        k_all = idx.shape[-1]
        k = min(self.neighbor_obs, k_all)
        idx_long = idx.long()
        arange_k = torch.arange(k_all, device=w.device)
        valid = arange_k.view(1, 1, -1) < cnt.long().unsqueeze(-1)  # [n_envs, n_agents, k_all]

        flat = idx_long.reshape(w.n_envs, -1)
        npos = torch.gather(pos, 1, flat.unsqueeze(-1).expand(-1, -1, 2))
        npos = npos.view(w.n_envs, w.n_agents, k_all, 2)
        rel_pos_all = npos - pos.unsqueeze(2)

        # Touching pairs -> per-agent contact counts (non-differentiable count).
        ndist = rel_pos_all.norm(dim=-1)
        r = w.agent_radius  # [n_agents]
        min_dist = r.view(1, -1, 1) + r[idx_long]  # r_i + r_j
        touching = ((ndist < min_dist) & valid).sum(dim=-1).to(w.dtype)

        # Observation features only need the first ``neighbor_obs`` slots.
        rel_pos = rel_pos_all[:, :, :k] * valid[:, :, :k].unsqueeze(-1)
        flat_k = idx_long[:, :, :k].reshape(w.n_envs, -1)
        nvel = torch.gather(vel, 1, flat_k.unsqueeze(-1).expand(-1, -1, 2))
        nvel = nvel.view(w.n_envs, w.n_agents, k, 2)
        rel_vel = (nvel - vel.unsqueeze(2)) * valid[:, :, :k].unsqueeze(-1)

        wall = (self._wall_gap(pos) < self.agent_radius + self.wall_margin).to(w.dtype)

        dist_to_goal = (pos - w.goals).norm(dim=-1)
        if self._prev_dist is None:
            self._prev_dist = dist_to_goal.detach().clone()
        pos_shaping = (self._prev_dist - dist_to_goal) * self.pos_shaping_factor
        if reset_mask is None:
            self._prev_dist = dist_to_goal.detach().clone()
        else:
            m = reset_mask.unsqueeze(-1)
            pos_shaping = torch.where(m, torch.zeros_like(pos_shaping), pos_shaping)
            self._prev_dist = torch.where(m, dist_to_goal.detach(), self._prev_dist)

        self._nbr_cache = {
            "rel_pos": rel_pos,
            "rel_vel": rel_vel,
            "valid": valid,
            "touching": touching,
            "wall": wall,
            "dist_to_goal": dist_to_goal,
            "on_goal": dist_to_goal < self.goal_tolerance,
            "pos_shaping": pos_shaping,
        }

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        """Fully batched observations ``[n_envs, n_agents, obs_dim]``."""
        if self.fused_active:
            return self.fb["obs"]
        w = self.world
        s = w.state
        cache = self._nbr_cache
        feats = [
            s.pos,
            s.vel,
            torch.cos(s.theta).unsqueeze(-1),
            torch.sin(s.theta).unsqueeze(-1),
            s.ang_vel.unsqueeze(-1),
            w.goals - s.pos,
            self.politeness.unsqueeze(-1),
        ]
        k = min(self.neighbor_obs, cache["rel_pos"].shape[2])
        if k > 0:
            ne, na = w.n_envs, w.n_agents
            feats += [
                cache["rel_pos"][:, :, :k].reshape(ne, na, -1),
                cache["rel_vel"][:, :, :k].reshape(ne, na, -1),
                cache["valid"][:, :, :k].to(w.dtype),
            ]
        return torch.cat(feats, dim=-1)

    def _terms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(shaping, collision, wall, final)`` per agent — the reward, factorized.

        ``rewards``, ``agent_reward``/``global_reward`` and ``multiobj_reward`` all read
        this one derivation, so the scalar reward and the objective vector cannot disagree
        about what a term is worth. ``time_penalty`` is a constant and stays a literal.
        """
        c = self._nbr_cache
        shaping = c["pos_shaping"]
        if self.shared_reward:
            shaping = shaping.sum(dim=-1, keepdim=True).expand_as(shaping)
        final = (
            self.final_reward
            * c["on_goal"].all(dim=-1, keepdim=True).to(self.world.dtype)
        ).expand_as(shaping)
        return (
            shaping,
            self.collision_penalty * c["touching"],
            self.wall_penalty * c["wall"],
            final,
        )

    def rewards(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["reward"]
        shaping, col, wall, final = self._terms()
        return shaping + col + wall + self.time_penalty + final

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        shaping, col, wall, _ = self._terms()
        rew = col[:, agent_idx] + wall[:, agent_idx] + self.time_penalty
        if not self.shared_reward:
            rew = rew + shaping[:, agent_idx]
        return rew

    def global_reward(self) -> torch.Tensor:
        shaping, _, _, final = self._terms()
        rew = final[:, 0]
        if self.shared_reward:
            rew = rew + shaping[:, 0]
        return rew

    def done(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["done_bool"]
        return self._nbr_cache["on_goal"].all(dim=-1)

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {
                "dist_to_goal": self.fb["dist"],
                "on_goal": self.fb["ongoal_bool"],
                "collisions": self.fb["touch"],
                "wall_contact": self.fb["wall"],
                "multiobj_reward": self.fb["multiobj"],
            }
        c = self._nbr_cache
        shaping, col, wall, final = self._terms()
        # Per-agent objective vector [n_envs, n_agents, N_OBJ]: the five reward terms kept
        # separate so a trainer logs one column each and a shaping imbalance shows up on
        # iteration 1 rather than hour 3. Its sum over the last dim equals the scalar
        # per-agent reward exactly; the ``rewards()`` key itself stays scalar.
        time = torch.full_like(col, self.time_penalty)
        return {
            "dist_to_goal": c["dist_to_goal"],
            "on_goal": c["on_goal"],
            "collisions": c["touching"],
            "wall_contact": c["wall"],
            "multiobj_reward": torch.stack([shaping, col, wall, time, final], dim=-1),
        }
