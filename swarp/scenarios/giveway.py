"""Give-way: two corridors crossing in a one-lane intersection.

Four immovable box obstacles fill the corner quadrants of the arena, leaving a
plus-shaped free space — two corridors that cross at a junction. Every agent starts at
the outer end of one arm and must reach the outer end of the **opposite** arm. The
corridor is sized so that one robot fits and two abreast do not
(``2*agent_radius < 2*corridor_half_width < 4*agent_radius``), so opposing robots have to
take turns: one ducks into a perpendicular arm — the passing bay — while the other
crosses the junction.

That constraint is the whole point, and it makes this the one **non-monotone** task in
the package. Every other scenario here can be solved by a policy that decreases its own
distance-to-goal monotonically; here the feasible joint solution *requires* an agent to
move away from its goal for several steps while the other passes. Position shaping alone
therefore points the yielding robot the wrong way, which is why ``shared_reward``
defaults to ``True`` (unlike navigation): with per-agent shaping the robot that gives way
pays the entire cost of a manoeuvre only the team benefits from.

Reward:
  * position shaping ``(prev_dist - dist) * pos_shaping_factor``, summed over the team
    when ``shared_reward=True`` and masked-rebased on reset,
  * ``collision_penalty`` per agent-agent contact,
  * ``wall_penalty`` while in contact with a corner block (scraping the corridor),
  * ``time_penalty`` every step, so standing still is never a solution,
  * ``final_reward`` for the whole team once every agent is on its goal.

Observation per agent: own pos, own vel, goal-relative vector, then relative
position/velocity and a validity flag for up to ``neighbor_obs`` in-radius neighbours.
There are deliberately **no wall features**: the policy is meant to infer the corridor
from contact and from where its neighbours are, which is also what keeps the observation
width independent of the (curriculum-variable) geometry.

**Contact stiffness — the parameter this scenario lives or dies by.**
``WorldConfig.collision_k``'s default of 100 is far too soft for a corridor: a
velocity-mode agent has no contact memory and settles at a penetration of
``max_speed / (collision_k * sub_dt)``, so the corridor walls would be a suggestion.
Push-T's ``contact_k=8000`` is *also* too soft here, which is not obvious and was worth
measuring: at 8000 a robot driven into a corner block sinks **0.033** into it, two thirds
of its own radius. Blocks that yield that far are not one-lane geometry, and the
consequence is not subtle — a greedy "drive straight at the goal" policy solves the task
**100%** of the time, because the robots simply squeeze through each other and the walls.
The whole premise of the scenario is gone.

Two *different* terms set how far a robot sinks, and they pull opposite ways in
``substeps`` — which is why stiffness alone cannot fix this:

* **settling depth**, ``max_speed / (k * sub_dt)`` — a velocity-mode agent has no contact
  memory, so holding it out of a wall needs ``f >= m * max_speed / sub_dt``. Falls with
  ``k``, *rises* with more substeps.
* **transient overshoot**, ``max_speed * sub_dt`` — an agent travels that far in the one
  substep before the contact can answer. Independent of ``k`` entirely, and falls with
  more substeps.

Raising ``k`` alone therefore hits a floor at the overshoot term. Measured, 8 robots on
random actions for 600 steps, worst overlap as a fraction of one radius:

======================  ==========  ==========  ==========
``contact_k``           ss=8        ss=16       ss=32
======================  ==========  ==========  ==========
200 000                 16.8%       **0%**      **0%**
500 000                 10.5%       **0%**      **0%**
2 000 000               16.7%       **0%**      **0%**
======================  ==========  ==========  ==========

At ``substeps=8`` the overshoot is ``1.0 * 0.05/8 = 0.00625``, half a radius's worth and
larger than the contact margin, so a fast robot is already inside the block before the
spring sees it — and no stiffness recovers that. At 16 the overshoot lands inside the
margin, contact catches it first, and penetration goes to *exactly* zero at every
stiffness tried. **Hence ``substeps >= 16``**, not the 8 this scenario originally
documented.

``contact_k`` defaults to 200 000 and the spring saturation (``contact_max_overlap``) is
switched **off**. Push-T needs that saturation to stop a stiff contact launching a light
movable body; give-way has no movable body at all — every obstacle is immovable scenery —
so a cap on the restoring force can only ever let robots sink deeper under a multi-robot
jam. The parity suites run at ``substeps=1``, where the settling depth is smallest.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from swarp._overloads import concrete
from swarp.core.cached_launch import CachedLaunch, ptr_key
from swarp.core.config import Obstacles, ObstacleShape, WorldConfig
from swarp.core.state import VEC2
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.interop.autograd import torch_stream_scope
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario
from swarp.scenarios.giveway_kernels import (
    giveway_obs_kernel,
    giveway_reset_kernel,
    giveway_reward_kernel,
)


class GiveWayScenario(FusedScenario):
    def __init__(
        self,
        n_agents: int = 4,
        agent_radius: float = 0.05,
        world_size: float = 1.0,
        corridor_half_width: float | None = None,
        max_speed: float = 1.0,
        neighbor_obs: int = 2,
        neighbor_radius: float | None = None,
        shared_reward: bool = True,
        pos_shaping_factor: float = 1.0,
        collision_penalty: float = -1.0,
        wall_penalty: float = -0.1,
        time_penalty: float = -0.01,
        final_reward: float = 5.0,
        goal_tolerance: float | None = None,
        spawn_jitter: float | None = None,
        contact_k: float = 200000.0,
        contact_c: float = 100.0,
        neighbor_method: str = "auto",
    ) -> None:
        # Four is the default because arm ``i % 4`` only produces a *head-on* pair from
        # three agents up (0 and 2 share the x corridor), and it is the head-on pair that
        # makes the task non-monotone — two agents alone meet at right angles, where
        # yielding costs nothing more than waiting.
        if n_agents < 1:
            raise ValueError(f"n_agents must be >= 1, got {n_agents}")
        self.n_agents = n_agents
        self.agent_radius = agent_radius
        self.world_size = world_size
        # One lane wide: 2r < 2c < 4r. Derived from the agent radius rather than
        # hardcoded, because the whole task definition is "one robot fits, two do not" —
        # a literal would silently stop meaning that the moment agent_radius moved.
        c = 1.5 * agent_radius if corridor_half_width is None else corridor_half_width
        if not agent_radius < c < 2.0 * agent_radius:
            raise ValueError(
                "corridor_half_width must satisfy agent_radius < c < 2*agent_radius so "
                "that exactly one robot fits abreast (2r < 2c < 4r); got "
                f"c={c} with agent_radius={agent_radius}"
            )
        self.corridor_half_width = c
        self.max_speed = max_speed
        self.neighbor_obs = neighbor_obs
        self.neighbor_radius = neighbor_radius
        self.shared_reward = shared_reward
        self.pos_shaping_factor = pos_shaping_factor
        self.collision_penalty = collision_penalty
        self.wall_penalty = wall_penalty
        self.time_penalty = time_penalty
        self.final_reward = final_reward
        self.goal_tolerance = goal_tolerance if goal_tolerance is not None else 2.0 * agent_radius
        self.spawn_jitter = spawn_jitter if spawn_jitter is not None else 0.25 * agent_radius
        self.contact_k = contact_k
        self.contact_c = contact_c
        self.neighbor_method = neighbor_method
        # A quarter of the radius, not the half navigation uses. The contact-activation
        # gap is what the reward's wall-contact flag is measured against, and at
        # margin = 0.5*r the activation band (r + margin = 1.5r) would land *exactly* on
        # the corridor's centreline distance (c = 1.5r) — every agent driving straight
        # down the middle would sit on the knife edge of a discrete flag. A quarter
        # radius puts the centreline a clear 0.25r outside the band.
        self.contact_margin = 0.25 * agent_radius
        # The reward's "scraping a corridor wall" test: the same reach the engine's box
        # contact uses (``_static_forces`` inflates the agent by ``ra + margin``), so the
        # flag means "the contact force is live", not some second, softer notion of near.
        self._wall_reach = agent_radius + self.contact_margin
        # Depth at which the contact spring saturates; filled in by make_world, which is
        # the first place ``sub_dt`` is known. Same derivation as pusht.
        self.max_overlap = 0.0
        #: Current multiple of ``corridor_half_width`` (see :meth:`set_corridor_scale`).
        self.corridor_scale = 1.0
        self._set_geometry(c)
        # Provisional obs width so ``obs_dim`` answers before ``make_world``; re-derived
        # there from the *resolved* config so a ``world_config`` override still wins.
        self._k_obs = min(self.neighbor_obs, min(32, max(4, self.n_agents)))

    # --------------------------------------------------------------- geometry

    def _set_geometry(self, c: float) -> None:
        """Recompute every corridor-width-dependent host constant for half-width ``c``.

        Split out from ``__init__`` because :meth:`set_corridor_scale` needs exactly this
        set recomputed and nothing else. The one-lane check is *not* repeated here: a
        curriculum deliberately runs wider-than-one-lane corridors, and only the
        constructor's base width defines the task.
        """
        r, w = self.agent_radius, self.world_size
        # The first-quadrant block spans [c, W] x [c, W]; its three mirror images fill the
        # other corners, leaving the plus-shaped free space.
        self._box_center = (0.5 * (w + c), 0.5 * (w + c))
        self._box_half = (0.5 * (w - c), 0.5 * (w - c))
        # The usable stretch of one arm: from ``s_min`` (any deeper and the agent is in
        # the junction rather than in its arm) out to ``outer``, the same
        # ``world_size - 2*agent_radius`` limit every other scenario spawns within.
        outer = w - 2.0 * r
        self._s_min = c + 2.0 * r
        if outer <= self._s_min:
            raise ValueError(
                f"world_size={w} leaves no arm at corridor_half_width={c} with "
                f"agent_radius={r}; need world_size > c + 4*agent_radius"
            )
        # Agents past the fourth stack up inward along their arm, one rank per four.
        per_arm = math.ceil(self.n_agents / 4)
        span = outer - self._s_min
        # The floor on the rank spacing is the *contact activation* distance, not merely
        # the touching distance: at this stiffness two agents that spawn already
        # inside each other's activation band start the episode being shoved apart, which
        # is a physics artefact rather than a task. Below it there is genuinely no room in
        # a one-lane arm for this many ranks, so it raises rather than packing them anyway.
        min_gap = 2.0 * r + self.contact_margin
        # The jitter is budgeted *before* the ranks are laid out, so the outermost agent
        # still lands inside ``outer`` and the innermost inside ``s_min`` — hence the
        # ``span - 2*cap`` the ranks are packed into and the ``outer - jitter`` below,
        # rather than a jitter bolted onto a layout that already filled the arm.
        cap = min(self.spawn_jitter, 0.5 * max(0.0, 3.0 * r - min_gap))
        inner_span = max(0.0, span - 2.0 * cap)
        if per_arm > 1:
            self._stagger = min(3.0 * r, inner_span / (per_arm - 1))
            if self._stagger < min_gap:
                raise ValueError(
                    f"{self.n_agents} agents need {per_arm} ranks per arm, which packs "
                    f"them {self._stagger:.4f} apart in an arm of length {span:.4f} — "
                    f"closer than the {min_gap:.4f} contact distance. Raise world_size "
                    "or lower n_agents."
                )
            long_room = 0.5 * (self._stagger - min_gap)
        else:
            self._stagger = 0.0
            long_room = 0.5 * span
        # Jitter amplitudes, clamped so a spawn can neither leave its arm, close on the
        # next rank, nor start inside the wall's contact band (which would hand the very
        # first step a wall-contact penalty it did nothing to earn).
        self._jitter_long = min(cap, max(0.0, long_room))
        self._jitter_lat = min(self.spawn_jitter, 0.5 * max(0.0, c - self._wall_reach))
        self._s_max = outer - self._jitter_long

    def set_corridor_scale(self, scale: float) -> None:
        """Widen (or restore) the corridor to ``scale`` times the constructor's width.

        The curriculum hook the MAPPO trainer drives: at ``scale=2`` two robots fit
        abreast and the task degenerates to ordinary navigation, so a novice policy
        actually reaches the terminal bonus; annealing back to ``scale=1`` restores the
        real one-lane intersection. ``set_corridor_scale(1.0)`` reproduces the
        constructor's geometry exactly (the derivation is re-run from the base width, not
        undone by inverse arithmetic).

        Host-side only — call it between batches, never inside a capture. It writes into
        the retained obstacle spec's own tensors and re-installs it, so the obstacle
        *count* never changes and ``Stepper.set_obstacles`` takes its in-place path: no
        reallocation, no graph recapture. The corner-block extents the fused SDF reads
        live in the ``geom`` device buffer this also rewrites, which is why the kernel
        takes them as an array rather than as four baked scalars — see
        :func:`~swarp.scenarios.giveway_kernels.giveway_obs_kernel`.
        """
        if scale <= 0.0:
            raise ValueError(f"corridor scale must be positive, got {scale}")
        self._set_geometry(self.corridor_half_width * scale)
        self.corridor_scale = scale
        if getattr(self, "world", None) is not None:
            self._install_geometry()

    def _install_geometry(self) -> None:
        """Push the current corridor geometry to the engine and to the fused SDF buffer."""
        (bx, by), (hx, hy) = self._box_center, self._box_half
        tt = {"device": self.world.device, "dtype": self.world.dtype}
        # Broadcast [4, 2] into the [n_envs, 4, 2] the spec holds: the blocks are the same
        # scenery in every env.
        self._box_pos.copy_(
            torch.tensor([[bx, by], [-bx, by], [-bx, -by], [bx, -by]], **tt)
        )
        self._box_half_t.copy_(torch.tensor([[hx, hy]] * 4, **tt))
        self._geom.copy_(torch.tensor([bx, by, hx, hy], **tt))
        # Re-install the *retained* resolved spec: already normalized (so ``resolve``
        # returns it untouched), ``kind`` absent (so ``any_movable`` never syncs), and an
        # unchanged count, which is the whole in-place contract.
        self.world.set_obstacles(self._obstacles)

    # ------------------------------------------------------------------ world

    def make_world(self, n_envs, device, dt, substeps, dtype, world_config=None) -> World:
        cfgs = [
            AgentConfig(
                model=DynamicsModel.HOLONOMIC,
                ctrl_mode=ControlMode.VELOCITY,
                radius=self.agent_radius,
                max_speed=self.max_speed,
                max_accel=2.0 * self.max_speed,
            )
            for _ in range(self.n_agents)
        ]
        margin = self.contact_margin
        reach = 2.0 * self.agent_radius + margin
        # Saturation OFF, unlike pusht. ``contact_max_overlap`` caps the restoring force
        # beyond a given depth, which pusht needs so a stiff contact cannot launch its
        # light movable T. Give-way has no movable body -- every obstacle is immovable
        # scenery -- so the cap has no upside here and one real downside: under a
        # four-robot junction jam the forces superpose, and a capped spring lets the pile
        # sink instead of pushing back harder. Kept as an attribute because it is part of
        # the contact story this class documents.
        self.max_overlap = 0.0
        cfg = WorldConfig(
            collisions=True,
            collision_k=self.contact_k,
            collision_c=self.contact_c,
            collision_margin=margin,
            contact_max_overlap=self.max_overlap,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            # The corner blocks are scanned linearly by ``_static_forces``, not looked up
            # in the neighbor grid, so their size does not have to inflate the
            # agent<->agent reach.
            neighbor_radius=max(self.neighbor_radius or 0.0, reach),
            max_neighbors=min(32, max(4, self.n_agents)),
            neighbor_method=self.neighbor_method,
        ).override_with(world_config)
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        tt = {"device": device, "dtype": dtype}
        self._k_obs = min(self.neighbor_obs, cfg.max_neighbors)
        # Goals: engine-independent per-agent targets, written in place by every reset;
        # allocated here (not in reset_world) so the fused spec can adopt them.
        self.world.goals = torch.zeros(n_envs, self.n_agents, 2, **tt)
        # The corner-block geometry the fused SDF reads, as a device buffer rather than
        # kernel scalars — see set_corridor_scale.
        self._geom = torch.zeros(4, **tt)
        # The obstacle spec, built and resolved once. Every field is already on the right
        # device/dtype and contiguous, so ``resolve`` hands back tensors that *share
        # storage* with these: writing them is writing the spec.
        self._box_pos = torch.zeros(n_envs, 4, 2, **tt)
        self._box_half_t = torch.zeros(4, 2, **tt)
        self._obstacles = Obstacles(
            self._box_pos,
            torch.zeros(4, **tt),  # a BOX ignores radius; its surface is its boundary
            shape=torch.full((4,), int(ObstacleShape.BOX), device=device, dtype=torch.int32),
            half_extents=self._box_half_t,
        ).resolve(device, dtype)
        # Installed **once**, here: the blocks are static scenery, so no reset and no
        # captured step ever touches the obstacle set again. Only the host-side curriculum
        # hook re-installs, and then with an unchanged count.
        self._install_geometry()
        # Static per-agent radii for the fused touching count (the torch reference reads
        # ``World.agent_radius``, which is this same tensor). Never reassigned, so the
        # handle stays pointer-stable.
        self._radius_wp = wp.from_torch(self.world.agent_radius, dtype=self.world.wp_dtype)
        self._geom_wp = wp.from_torch(self._geom, dtype=self.world.wp_dtype)
        # Left None so the first refresh seeds the shaping baseline from the spawn
        # distance (shaping 0) rather than from zeros; the fused spec adopts it with
        # alloc="if_none". The torch path reassigns it, which watch=True catches.
        self._prev_dist: torch.Tensor | None = None
        self._cache: dict[str, torch.Tensor] | None = None
        # Cached, repack-once launches for the eager reset path — see
        # swarp/core/cached_launch.py.
        self._reset_launch = CachedLaunch()
        self._obs_launch = CachedLaunch()
        return self.world

    @property
    def obs_dim(self) -> int:
        # own pos(2) + vel(2) + goal-relative(2), then rel_pos/rel_vel/valid per neighbour.
        return 6 + 5 * self._k_obs

    # ------------------------------------------------------------------ reset

    def reset_world(
        self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False
    ) -> None:
        """Masked reset in one Warp launch (see :mod:`swarp.scenarios.reset_kernels`).

        Host-sync-free: the mask is applied inside the kernel, so there is no host-side
        "is anything done?" gate. The corner blocks are *not* re-installed — they are
        static scenery installed once in :meth:`make_world`, which is what keeps the
        obstacle set entirely out of the reset path.
        """
        w = self.world
        mask, use_mask = self.reset_mask_wp(env_mask)
        st = w.state_wp()
        scalar = w.wp_dtype
        if self.fused_active:
            # Reuse the fused spec's cached, pointer-resynced handle instead of
            # re-wrapping ``world.goals`` on every reset (see navigation's
            # ``_launch_reset`` for the same fix and its rationale).
            self.ensure_fused()
            self.sync_fused_handles()
            goals = self._wp["goals"]
        else:
            goals = wp.from_torch(w.goals.contiguous(), dtype=VEC2[scalar])
        seed = wp.int32(w.next_kernel_seed())
        with torch_stream_scope(w.device):
            launch = self._reset_launch.get(
                concrete(giveway_reset_kernel, scalar),
                dim=w.n_envs,
                inputs=[
                    mask,
                    use_mask,
                    seed,
                    wp.int32(self.n_agents),
                    scalar(self._s_max),
                    scalar(self._stagger),
                    scalar(self._jitter_long),
                    scalar(self._jitter_lat),
                    st.pos,
                    st.theta,
                    st.vel,
                    st.speed,
                    st.ang_vel,
                    goals,
                ],
                device=w.device,
                key=(
                    w.n_envs,
                    ptr_key(mask),
                    self.n_agents,
                    # The four spawn scalars are in the key, not re-set per call: they are
                    # step-invariant, but ``set_corridor_scale`` does move three of them,
                    # and a key entry is what repacks the launch when it does.
                    self._s_max,
                    self._stagger,
                    self._jitter_long,
                    self._jitter_lat,
                    ptr_key(st.pos),
                    ptr_key(st.theta),
                    ptr_key(st.vel),
                    ptr_key(st.speed),
                    ptr_key(st.ang_vel),
                    ptr_key(goals),
                ),
            )
            launch.set_param_by_name("use_mask", use_mask)
            launch.set_param_by_name("seed", seed)
            launch.launch()
        w.mark_pos_dirty()
        self.finish_reset(env_mask, obs_only=obs_only)

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
            Buf("ongoal", (ne, na), "uint8", bool_view=True),
            Buf("frac", (ne,)),
            Buf("done", (ne,), "uint8", bool_view=True),
            Buf("resetmask", (ne,), "uint8", reset_mask=True),
            Buf("goals", (ne, na, 2), "vec2", attr="world.goals", alloc="never", watch=True),
            Buf("prev", (ne, na), attr="_prev_dist", alloc="if_none", carry=True, watch=True),
        )

    def launch_fused(self, pass_: FusedPass) -> None:
        """Obs, then reward — the reward on every pass but an obs-only auto-reset.

        Same rule as navigation's: ``full_pass=0`` (a mid-step auto-reset) keeps the
        reward/done already returned for that transition, while a standalone reset
        recomputes them so the fused path reports the *new* episode, as the torch oracle
        does. Nothing here touches the obstacle set, so the whole sequence is trivially
        capture-safe: two launches against pointer-stable handles.
        """
        self._launch_obs(advance_prev=pass_.advance_prev, full_pass=pass_.full_pass)
        if pass_.full_pass:
            self._launch_reward()

    def _launch_obs(self, advance_prev: int, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        resetmask = self._wp["resetmask"]
        # ``full_pass == 0`` is *only* the obs-only auto-reset pass, the one pass where
        # every env the mask does not select has positions unchanged since the STEP pass
        # built these lists — so the masked rebuild can never be reached by a normal step
        # or a full reset.
        w.build_neighbors(reset_mask=resetmask if full_pass == 0 else None)
        grid = w.stepper.grid(w.n_envs)
        st = w.state_wp()
        goals, prev = self._wp["goals"], self._wp["prev"]
        obs, touch, wall, dist, shaping, ongoal = (
            self._wp["obs"],
            self._wp["touch"],
            self._wp["wall"],
            self._wp["dist"],
            self._wp["shaping"],
            self._wp["ongoal"],
        )
        # ``advance_prev``/``full_pass`` are the only genuinely per-call arguments;
        # everything else is static config or a pointer-stable handle, so it lives in the
        # cache key instead. The corner geometry is *not* in the key — it is read out of
        # the ``geom`` buffer by the kernel, so a curriculum step needs no repack (and,
        # for the captured step, no recapture).
        launch = self._obs_launch.get(
            concrete(giveway_obs_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                goals,
                grid.neighbor_idx,
                grid.neighbor_count,
                self._radius_wp,
                self._geom_wp,
                resetmask,
                wp.int32(self._k_obs),
                scalar(self._wall_reach),
                scalar(self.pos_shaping_factor),
                scalar(self.goal_tolerance),
                wp.int32(advance_prev),
                wp.int32(full_pass),
            ],
            outputs=[obs, touch, wall, dist, shaping, ongoal, prev],
            device=w.device,
            key=(
                w.n_envs,
                self.n_agents,
                ptr_key(st.pos),
                ptr_key(st.vel),
                ptr_key(goals),
                ptr_key(grid.neighbor_idx),
                ptr_key(grid.neighbor_count),
                ptr_key(self._radius_wp),
                ptr_key(self._geom_wp),
                ptr_key(resetmask),
                self._k_obs,
                self._wall_reach,
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
                self._wp["shaping"],
                self._wp["touch"],
                self._wp["wall"],
                self._wp["ongoal"],
                wp.int32(self.n_agents),
                scalar(1.0 / self.n_agents),
                scalar(self.collision_penalty),
                scalar(self.wall_penalty),
                scalar(self.time_penalty),
                scalar(self.final_reward),
                wp.int32(1 if self.shared_reward else 0),
            ],
            outputs=[self._wp["reward"], self._wp["done"], self._wp["frac"]],
            device=w.device,
            record_tape=False,
        )

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh()

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        self._refresh(reset_mask=env_mask)

    def _corner_sdf(self, pos: torch.Tensor) -> torch.Tensor:
        """Signed distance from each agent to the nearest corner block.

        The torch half of :func:`~swarp.scenarios.giveway_kernels._corner_sdf`: the same
        fold into the first quadrant and the same association order (an explicit
        ``sqrt(ox*ox + oy*oy)`` rather than ``.norm()``), so the *discrete* contact flag
        the two paths derive from it agrees exactly rather than only closely. Written
        independently of the kernel — this is the parity oracle, not a shared helper.
        """
        (bx, by), (hx, hy) = self._box_center, self._box_half
        qx = (pos[..., 0].abs() - bx).abs() - hx
        qy = (pos[..., 1].abs() - by).abs() - hy
        ox = qx.clamp(min=0.0)
        oy = qy.clamp(min=0.0)
        return (ox * ox + oy * oy).sqrt() + torch.maximum(qx, qy).clamp(max=0.0)

    def _refresh(self, reset_mask: torch.Tensor | None = None) -> None:
        """Neighbor features, distances, contacts and reward terms for the current state.

        ``reset_mask`` (a boolean ``[n_envs]`` or ``None``) marks envs that were just
        reset: their shaping baseline is rebased to the fresh spawn distance and their
        shaping term zeroed, while every other env keeps its carried-over baseline.
        """
        w = self.world
        ne, na = w.n_envs, w.n_agents
        pos, vel = w.state.pos, w.state.vel
        idx, cnt = w.neighbors()
        k_all = idx.shape[-1]
        k = min(self.neighbor_obs, k_all)
        idx_long = idx.long()
        arange_k = torch.arange(k_all, device=w.device)
        valid = arange_k.view(1, 1, -1) < cnt.long().unsqueeze(-1)  # [ne, na, k_all]

        # Neighbor positions for every padded slot: the touching count needs them all.
        flat = idx_long.reshape(ne, -1).unsqueeze(-1).expand(-1, -1, 2)
        npos = torch.gather(pos, 1, flat).view(ne, na, k_all, 2)
        rel_pos_all = npos - pos.unsqueeze(2)
        ndist = rel_pos_all.norm(dim=-1)
        r = w.agent_radius  # [n_agents]
        touching = ((ndist < (r.view(1, -1, 1) + r[idx_long])) & valid).sum(dim=-1).to(w.dtype)

        # Observation features only need the first ``neighbor_obs`` slots.
        rel_pos = rel_pos_all[:, :, :k] * valid[:, :, :k].unsqueeze(-1)
        flat_k = idx_long[:, :, :k].reshape(ne, -1).unsqueeze(-1).expand(-1, -1, 2)
        nvel = torch.gather(vel, 1, flat_k).view(ne, na, k, 2)
        rel_vel = (nvel - vel.unsqueeze(2)) * valid[:, :, :k].unsqueeze(-1)

        wall_touch = (self._corner_sdf(pos) < self._wall_reach).to(w.dtype)
        dist_to_goal = (pos - w.goals).norm(dim=-1)
        if self._prev_dist is None:
            self._prev_dist = dist_to_goal.detach().clone()
        pos_shaping = (self._prev_dist - dist_to_goal) * self.pos_shaping_factor
        if reset_mask is None:
            self._prev_dist = dist_to_goal.detach().clone()
        else:
            # Rebase only reset envs; the others keep their continuous baseline.
            m = reset_mask.unsqueeze(-1)
            pos_shaping = torch.where(m, torch.zeros_like(pos_shaping), pos_shaping)
            self._prev_dist = torch.where(m, dist_to_goal.detach(), self._prev_dist)

        on_goal = dist_to_goal < self.goal_tolerance
        self._cache = {
            "rel_pos": rel_pos,
            "rel_vel": rel_vel,
            "valid": valid,
            "touching": touching,
            "wall_touch": wall_touch,
            "dist_to_goal": dist_to_goal,
            "on_goal": on_goal,
            "frac_on_goal": on_goal.to(w.dtype).sum(dim=-1) * (1.0 / self.n_agents),
            "pos_shaping": pos_shaping,
        }

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        """Fully batched observations ``[n_envs, n_agents, obs_dim]``."""
        if self.fused_active:
            return self.fb["obs"]
        w = self.world
        s = w.state
        c = self._cache
        feats = [s.pos, s.vel, w.goals - s.pos]
        k = min(self.neighbor_obs, c["rel_pos"].shape[2])
        if k > 0:
            feats += [
                c["rel_pos"][:, :, :k].reshape(w.n_envs, w.n_agents, -1),
                c["rel_vel"][:, :, :k].reshape(w.n_envs, w.n_agents, -1),
                c["valid"][:, :, :k].to(w.dtype),
            ]
        return torch.cat(feats, dim=-1)

    def _own_terms(self) -> torch.Tensor:
        """The per-agent, never-shared part of the reward: contacts plus the clock."""
        c = self._cache
        return (
            self.collision_penalty * c["touching"]
            + self.wall_penalty * c["wall_touch"]
            + self.time_penalty
        )

    def rewards(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["reward"]
        c = self._cache
        rew = self._own_terms()
        if self.shared_reward:
            rew = rew + c["pos_shaping"].sum(dim=-1, keepdim=True)
        else:
            rew = rew + c["pos_shaping"]
        final = self.final_reward * c["on_goal"].all(dim=-1).to(self.world.dtype)
        return rew + final.unsqueeze(-1)

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        c = self._cache
        rew = self._own_terms()[:, agent_idx]
        if not self.shared_reward:
            rew = rew + c["pos_shaping"][:, agent_idx]
        return rew

    def global_reward(self) -> torch.Tensor:
        c = self._cache
        rew = self.final_reward * c["on_goal"].all(dim=-1).to(self.world.dtype)
        if self.shared_reward:
            rew = rew + c["pos_shaping"].sum(dim=-1)
        return rew

    def done(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["done_bool"]
        return self._cache["on_goal"].all(dim=-1)

    def info(self) -> dict[str, Any]:
        """Diagnostics an RL trainer reads through ``("next", "info", <key>)``.

        ``frac_on_goal`` is the scalar success signal: a per-env fraction rather than the
        bare ``all_on_goal`` flag, because on a task where the *last* robot through the
        junction decides the episode, a 0/1 signal spends most of training flat.
        """
        if self.fused_active:
            return {
                "dist_to_goal": self.fb["dist"],
                "on_goal": self.fb["ongoal_bool"],
                "collisions": self.fb["touch"],
                "wall_contacts": self.fb["wall"],
                "frac_on_goal": self.fb["frac"],
                "all_on_goal": self.fb["done_bool"],
            }
        c = self._cache
        return {
            "dist_to_goal": c["dist_to_goal"],
            "on_goal": c["on_goal"],
            "collisions": c["touching"],
            "wall_contacts": c["wall_touch"],
            "frac_on_goal": c["frac_on_goal"],
            "all_on_goal": c["on_goal"].all(dim=-1),
        }
