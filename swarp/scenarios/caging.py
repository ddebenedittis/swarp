"""Caging: a team surrounds an evasive disc so it cannot escape.

The only built-in scenario whose success criterion is **topological** rather than metric.
Being close to the disc is worth nothing; what counts is that the agents' *bearings as
seen from the disc* leave no large angular gap, that they all sit at roughly the right
radius, and that the disc is not pinned against the arena wall. A team can be tightly
clustered on one side of the disc and be arbitrarily far from caging it.

The disc is not passive. Per env step it feels

  1. a **per-episode constant drift** — a wander direction drawn once at reset,
  2. a **flee** push away from every agent within ``flee_radius``, profile
     ``(1 - d / flee_radius)^2``,
  3. the **contact reaction** of the same spring-damper contact the engine applies to the
     agents,

and is integrated by the scenario with one semi-implicit Euler step plus linear damping.
Term 3 is what stops the task from degenerating into ``formation``: without it "caged"
would be a reward label with no dynamics behind it, and the ring would be a pattern to
hold rather than a wall to hold *against*.

Why the disc is not an engine ``MOVABLE`` obstacle
--------------------------------------------------
There is no external-force hook for one. :func:`swarp.core.bodies.obstacle_dynamics_kernel`
derives force purely from agent reaction and body-body contact; its signature has no slot
for an external force, and ``ObstacleKind`` is read in exactly one place, gating only
whether the engine integrates the obstacle at all. Stamping the body velocity between
steps is erased by the per-substep damping, and the contact law is identically zero
beyond ``r_a + r_obs + margin``, so it cannot express a flee potential that reaches
further than contact. So this scenario follows transport's pattern instead: it owns
``disc_pos``/``disc_vel``, installs a ``kind``-free circular obstacle aliasing them (an
immovable-tagged circle still pushes agents — :func:`swarp.core.collisions._static_forces`
dispatches on the *shape* tag and never reads ``obs_kind``), re-installs the new pose
from inside the captured whole-step graph, and integrates the disc itself. Unlike
transport it also installs ``vel``, so the agent-disc damper sees the **closing**
velocity rather than the agent's absolute one.

Reward (repo ``prev - cur`` shaping convention)::

    global     gap_shaping_factor  * (prev_gap_soft - gap_soft)
             + band_shaping_factor * (prev_band_err - band_err)
             + cage_reward * float(caged)          # dwell bonus, EVERY caged step
             + time_penalty
    per-agent  collision_penalty * touching_count

``cage_reward`` is a dwell bonus rather than a one-off terminal because ``hold_steps``
asks the policy to *maintain* the cage, and a terminal-only bonus gives zero signal
during the hold. The measured per-term magnitudes, and why ``band_shaping_factor``
defaults to 4 rather than 1, are in ``__init__``.

Observation per agent (``obs_dim = 13 + 2 * (n_agents - 1)``): own pos and vel, the
disc's relative position and its velocity, own radial error, ``cos``/``sin`` of own
angular gap, the team's ``gap_max``, the ``caged`` flag, and the other agents' relative
positions. The last two scalars are broadcast identically to every agent, and that is
what makes a topological objective observable to a decentralised policy: without them no
agent can tell whether the cage is closed.

**``substeps >= 8`` at ``dt = 0.05``.** A velocity-mode agent has no contact memory, so it
settles at a contact depth of ``max_speed / (contact_k * sub_dt)``; at ``substeps = 1``
that depth is several agent radii and the disc is a suggestion rather than an obstacle.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from swarp._overloads import concrete
from swarp.core.cached_launch import CachedLaunch, ptr_key
from swarp.core.config import Obstacles, WorldConfig
from swarp.core.state import VEC2
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.interop.autograd import torch_stream_scope
from swarp.scenarios.caging_kernels import (
    N_OBJ,
    caging_disc_kernel,
    caging_obs_kernel,
    caging_reset_kernel,
    caging_reward_kernel,
)
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario

_TWO_PI = 2.0 * math.pi


class CagingScenario(FusedScenario):
    """See the module docstring. Parity note on :attr:`parity_rtol`/:attr:`parity_atol`.

    Both paths run exactly one semi-implicit Euler step for the disc, at the same ``dt``,
    with the same force law — there is no substep mismatch here, so this is *not*
    push-t-loose. What is left is per-agent force-sum reassociation (a sequential kernel
    loop against torch's ``.sum(dim=1)``) plus ``sqrt``/``atan2``/``exp``/``log`` ulps,
    amplified by the ``prev - cur`` shaping differencing two nearly equal quantities.
    ``1e-4``/``1e-5`` covers that with room and is still four orders tighter than a
    genuine model difference would need.

    The residual risk no tolerance can cover: ``caged`` is a conjunction of ``n_agents +
    2`` threshold tests and ``done`` is compared **exactly** by the parity harness. Keep a
    test configuration whose spawn distribution does not sit near either
    ``band_half`` or ``gap_threshold``.
    """

    parity_rtol: float = 1e-4
    parity_atol: float = 1e-5

    def __init__(
        self,
        n_agents: int = 5,
        agent_radius: float = 0.05,
        max_speed: float = 1.0,
        world_size: float = 1.0,
        cage_radius: float = 0.35,
        band_half: float = 0.12,
        gap_threshold: float | None = None,
        hold_steps: int = 10,
        wall_free_radius: float | None = None,
        disc_radius: float = 0.1,
        disc_mass: float = 1.0,
        linear_damping: float = 0.5,
        flee_radius: float = 0.3,
        flee_speed: float | None = None,
        drift_speed: float | None = None,
        contact_k: float = 2000.0,
        contact_c: float = 40.0,
        contact_margin: float = 0.01,
        beta: float = 3.0,
        gap_shaping_factor: float = 1.0,
        band_shaping_factor: float = 4.0,
        cage_reward: float = 0.5,
        time_penalty: float = -0.01,
        collision_penalty: float = -1.0,
        neighbor_method: str = "auto",
    ) -> None:
        self.n_agents = n_agents
        self.agent_radius = agent_radius
        self.max_speed = max_speed
        self.world_size = world_size
        self.cage_radius = cage_radius
        self.band_half = band_half
        # Derived from the team size, not a constant. The best any ring of N agents can
        # do is N equal arcs of ``2*pi/N``, so a fixed threshold is a different task at
        # every N and simply unsatisfiable for N <= 3 (2*pi/3 = 2.09). What is meaningful
        # is the *margin* over that floor, and it should be roughly constant in absolute
        # angle: 0.75 rad of slack lets the ring be visibly irregular while still leaving
        # no escape corridor. At the default five agents this comes out at 2.007, i.e. the
        # 2.0 rad the task was specified with.
        self.gap_threshold = (
            _TWO_PI / n_agents + 0.75 if gap_threshold is None else float(gap_threshold)
        )
        self.disc_radius = disc_radius
        self.disc_mass = disc_mass
        self.linear_damping = linear_damping
        self.flee_radius = flee_radius
        self.contact_k = contact_k
        self.contact_c = contact_c
        self.contact_margin = contact_margin
        self.beta = beta
        self.gap_shaping_factor = gap_shaping_factor
        # NOTE ON MAGNITUDES, measured rather than guessed (100 steps, 512 envs, uniform
        # random actions, ``info()["multiobj_reward"]`` averaged per column):
        #
        #   term            mean|.| per agent-step, difficulty 0 / 0.5 / 1
        #   gap_shaping     0.065 / 0.060 / 0.054
        #   band_shaping    0.051 / 0.058 / 0.052   (at this factor of 4; 0.013 at 1.0)
        #   collision       0.018 / 0.034 / 0.016
        #   time            0.010 (constant)
        #   cage            0.044 / 0.0003 / 0.000  (0.5 when it fires; see below)
        #
        # 4.0, not 1.0, because the two shaping terms are not in the same units:
        # ``gap_soft`` is radians and spans ~4.5 over an episode, ``band_err`` is metres
        # and spans ~0.5. At factor 1 the band term is 5x smaller per step and ~9x smaller
        # cumulatively — the radial band would be a rounding error against the angular
        # objective, and a team would learn to spread out at any radius. 4.0 equalizes the
        # per-step magnitudes, which is what a policy-gradient estimator actually sees.
        #
        # ``cage_reward`` is deliberately ~8x the largest shaping term *when it fires*.
        # That is not an imbalance to fix: it is the objective, it is sparse (8.8% of
        # agent-steps under a random policy at difficulty 0, ~0 at difficulty 1), and a
        # dwell bonus that did not dominate the shaping it interrupts would not be worth
        # holding the cage for.
        self.band_shaping_factor = band_shaping_factor
        self.cage_reward = cage_reward
        self.time_penalty = time_penalty
        self.collision_penalty = collision_penalty
        self.neighbor_method = neighbor_method

        if not 1 <= int(hold_steps) <= 255:
            raise ValueError(
                f"hold_steps must lie in [1, 255]; got {hold_steps}. The hold counter is a "
                "saturating uint8 device buffer (one byte per env, read by both paths), so "
                "255 is the largest dwell the predicate can express."
            )
        self.hold_steps = int(hold_steps)
        if not 0.0 < self.gap_threshold < _TWO_PI:
            raise ValueError(
                f"gap_threshold must lie in (0, 2*pi); got {self.gap_threshold}. It is compared "
                "against the largest of the n_agents arcs of the circular bearing order, "
                "which sum to exactly 2*pi — at or above 2*pi every configuration is caged."
            )
        # A regular ring of N agents has every arc equal to 2*pi/N, so a threshold at or
        # below that is unreachable however well the team is spread.
        if self.gap_threshold <= _TWO_PI / n_agents:
            raise ValueError(
                f"gap_threshold ({self.gap_threshold}) is at or below 2*pi/n_agents "
                f"({_TWO_PI / n_agents}), the arc of a *perfectly* regular ring of "
                f"{n_agents} agents; no configuration can ever satisfy it"
            )

        # ---- the wall-free conjunct, and why it is a conjunct
        # Once the disc is clamped to the arena, the trivial optimum is to shove it into a
        # corner: two walls then close most escape directions for free and ``gap_max``
        # never has to shrink. Nothing in a parity suite can see that — it is a correct
        # optimizer finding the wrong optimum. So "the disc is away from the wall" is part
        # of the ``caged`` *predicate*, not a penalty: a penalty is one more weight to
        # tune and can be traded off against, a predicate cannot. It has to be in from the
        # start, because retrofitting it changes the reward function and invalidates every
        # trained checkpoint.
        self.wall_free_radius = (
            0.6 * world_size if wall_free_radius is None else float(wall_free_radius)
        )
        # The ring has to fit inside the wall-free zone, or a caged disc implies agents
        # outside the arena.
        reach = self.wall_free_radius + cage_radius + band_half + agent_radius
        if reach > world_size * math.sqrt(2.0):
            raise ValueError(
                f"cage geometry does not fit: wall_free_radius + cage_radius + band_half + "
                f"agent_radius = {reach} exceeds the arena's half-diagonal "
                f"{world_size * math.sqrt(2.0)}; lower wall_free_radius or cage_radius"
            )

        # ---- force magnitudes, expressed as terminal speeds
        # Under ``(1 - lambda*dt)`` decay a constant force ``f`` settles at
        # ``f / (m * lambda)``, so sizing these as speeds is the only way to state them
        # that survives a change of mass or damping. A raw stiffness would silently mean
        # something different the moment either moved.
        self.drift_speed = 0.4 * max_speed if drift_speed is None else float(drift_speed)
        self.flee_speed = 0.8 * max_speed if flee_speed is None else float(flee_speed)
        self.drift_mag = disc_mass * linear_damping * self.drift_speed
        self.flee_k = disc_mass * linear_damping * self.flee_speed

        # ---- spawn geometry
        # Angle jitter breaks the exact symmetry of the ring without letting two
        # neighbours' arcs approach ``gap_threshold`` (see the class docstring's note on
        # near-degenerate thresholds); radial jitter stays inside half the deadband so a
        # difficulty-0 spawn is genuinely in-band for every agent.
        self.ring_ang_jitter = 0.15
        self.ring_rad_jitter = 0.5 * band_half

        self._difficulty = 1.0
        # Set by ``make_world``; None until there is a device to write to.
        self._difficulty_t: torch.Tensor | None = None

    # -------------------------------------------------------------- curriculum

    @property
    def difficulty(self) -> float:
        """Curriculum knob a trainer pokes between batches (Push-T's mechanism).

        ``0`` gives an inert disc (no drift, no evade) at the origin with the agents
        spawned **on the cage ring**, i.e. already caged on step 1 — so the dwell bonus
        and the terminal are experienced immediately, which is the single reason Push-T's
        curriculum works. ``1`` gives a uniform spawn, full drift and full flee. Clamped
        to ``[0, 1]``.

        Deliberately **not** annealed: ``gap_threshold`` and ``cage_radius``. Moving the
        success criterion moves the reward under the value function.

        A property rather than a plain attribute because the derived value reaches the
        reset kernel through a one-element device tensor read *by pointer* (see
        :func:`~swarp.scenarios.caging_kernels.caging_reset_kernel`); this setter is what
        keeps that tensor in step with the Python value, so assigning the attribute still
        "just works" mid-training.
        """
        return self._difficulty

    @difficulty.setter
    def difficulty(self, value: float) -> None:
        self._difficulty = min(max(float(value), 0.0), 1.0)
        if self._difficulty_t is not None:
            # One tiny host->device write per curriculum change (not per step); a
            # ``fill_`` rather than an indexed assignment so nothing reads back.
            self._difficulty_t.fill_(self._difficulty)

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
        # Push-T's derivation. A velocity-mode agent has no contact memory: each substep
        # its velocity is overwritten by the command plus f*sub_dt/m, so holding it off
        # the disc needs f >= m*max_speed/sub_dt, i.e. an equilibrium penetration of
        # max_speed/(k*sub_dt). Saturating the spring *below* that depth lets agents walk
        # through the disc; twice it bounds the impulse a deep sweep injects while leaving
        # the holding force intact.
        sub_dt = dt / max(1, substeps)
        self.max_overlap = 2.0 * self.max_speed / (self.contact_k * sub_dt)
        # ``contact_margin`` is BOTH the engine's collision margin and this scenario's own
        # agent<->disc activation gap, unlike transport which keeps two numbers. The disc's
        # reaction has to be the third-law partner of the force the engine applies to the
        # agents, and two different margins would make the pair activate at two different
        # separations.
        cfg = WorldConfig(
            collisions=True,
            collision_k=self.contact_k,
            collision_c=self.contact_c,
            collision_margin=self.contact_margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            # No neighbor_radius inflation for the disc: obstacles are scanned linearly by
            # ``_static_forces``, not looked up in the neighbor grid, so the disc's size
            # does not have to widen the agent<->agent reach. (Transport inflates it; that
            # is dead cost.)
            neighbor_radius=2.0 * self.agent_radius + self.contact_margin,
            max_neighbors=min(32, max(4, self.n_agents)),
            neighbor_method=self.neighbor_method,
            contact_max_overlap=self.max_overlap,
        ).override_with(world_config)
        self.dt = dt
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        tt = {"device": device, "dtype": dtype}
        # ---- scenario-owned disc state and per-episode draws. All allocated here, where
        # n_envs is known, so the fused spec adopts them rather than racing the first
        # reset; all written in place by the reset kernel so their handles stay valid.
        self.disc_pos = torch.zeros(n_envs, 1, 2, **tt)
        self.disc_vel = torch.zeros(n_envs, 1, 2, **tt)
        self.drift = torch.zeros(n_envs, 1, 2, **tt)
        self.evade = torch.zeros(n_envs, **tt)
        # Saturating dwell counter. One byte per env: ``hold_steps <= 255`` is validated
        # in __init__ precisely so this can stay a byte the reward kernel advances in
        # place, rather than an int32 buffer for a number that never exceeds a few tens.
        self.hold = torch.zeros(n_envs, device=device, dtype=torch.uint8)
        # ONE retained obstacle spec aliasing the disc state, re-installed rather than
        # rebuilt: ``resolve`` and ``any_movable`` both memoize, so a re-install allocates
        # nothing and costs no device->host sync, which is what makes it legal inside the
        # capture. ``vel`` is installed (transport omits it) so the agent-disc damper sees
        # the closing velocity.
        self._disc_radius_t = torch.full((1,), self.disc_radius, **tt)
        self._obstacles = self._build_obstacles()
        # Shaping baselines, left None so the first refresh seeds them from the spawn
        # values (shaping 0) rather than from zeros; the spec adopts them with
        # alloc="if_none" and watch=True catches the torch path reassigning them.
        self._prev_gap: torch.Tensor | None = None
        self._prev_band: torch.Tensor | None = None
        self._cache: dict[str, torch.Tensor] | None = None
        # The curriculum value, by pointer — see :attr:`difficulty`. Allocated once and
        # never reallocated, so the handle stays valid for the life of the world.
        self._difficulty_t = torch.zeros(1, **tt)
        self._difficulty_wp = wp.from_torch(self._difficulty_t, dtype=self.world.wp_dtype)
        self.difficulty = self._difficulty  # sync the device value
        # Teammate index table for the torch observation: row ``a`` is every agent but
        # ``a`` in ascending order, which is the order the fused kernel's ``slot`` counter
        # produces. Fixed by ``n_agents``, so built once here.
        na = self.n_agents
        idx = torch.arange(na, device=device)
        others = idx.unsqueeze(0).expand(na, -1)[idx.unsqueeze(1) != idx.unsqueeze(0)]
        self._others = others.view(na, na - 1)  # [A, A-1]
        # Cached, repack-once launches for the eager paths — see swarp/core/cached_launch.py.
        self._reset_launch = CachedLaunch()
        self._disc_launch = CachedLaunch()
        self._obs_launch = CachedLaunch()
        self._reward_launch = CachedLaunch()
        return self.world

    @property
    def obs_dim(self) -> int:
        return 13 + 2 * (self.n_agents - 1)

    # ------------------------------------------------------------------ reset

    def reset_world(
        self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False
    ) -> None:
        """Masked reset in one Warp launch (see :mod:`swarp.scenarios.reset_kernels`).

        Host-sync-free: the disc pose, the per-episode drift force, the evade gain, the
        hold counter and the blended ring/uniform spawns are all written device-side by
        one masked kernel, so nothing here depends on *which* envs the mask selected.
        """
        w = self.world
        agent_lim = self.world_size - 2.0 * self.agent_radius
        # The disc starts inside the wall-free zone with room to spare, so an episode
        # never opens with the corner exploit already available.
        disc_lim = 0.6 * self.wall_free_radius
        mask, use_mask = self.reset_mask_wp(env_mask)
        st = w.state_wp()
        scalar = w.wp_dtype
        vec2 = VEC2[scalar]
        if self.fused_active:
            # Reuse the spec's cached, pointer-resynced handles instead of re-wrapping the
            # watched tensors on every reset. ``sync_fused_handles`` must run first: it is
            # what notices a grad-path reassignment and rebuilds the handle before this
            # kernel writes through it.
            self.ensure_fused()
            self.sync_fused_handles()
            disc_pos, disc_vel = self._wp["disc_pos"], self._wp["disc_vel"]
            drift, evade, hold = self._wp["drift"], self._wp["evade"], self._wp["hold"]
        else:
            disc_pos = wp.from_torch(self.disc_pos.contiguous(), dtype=vec2)
            disc_vel = wp.from_torch(self.disc_vel.contiguous(), dtype=vec2)
            drift = wp.from_torch(self.drift.contiguous(), dtype=vec2)
            evade = wp.from_torch(self.evade.contiguous(), dtype=scalar)
            hold = wp.from_torch(self.hold.contiguous(), dtype=wp.uint8)
        seed = wp.int32(w.next_kernel_seed())
        with torch_stream_scope(w.device):
            launch = self._reset_launch.get(
                concrete(caging_reset_kernel, scalar),
                dim=w.n_envs,
                inputs=[
                    mask,
                    use_mask,
                    seed,
                    self._difficulty_wp,
                    wp.int32(self.n_agents),
                    scalar(agent_lim),
                    scalar(disc_lim),
                    scalar(self.cage_radius),
                    scalar(self.ring_ang_jitter),
                    scalar(self.ring_rad_jitter),
                    scalar(self.drift_mag),
                    st.pos,
                    st.theta,
                    st.vel,
                    st.speed,
                    st.ang_vel,
                    disc_pos,
                    disc_vel,
                    drift,
                    evade,
                    hold,
                ],
                device=w.device,
                key=(
                    w.n_envs,
                    ptr_key(mask),
                    ptr_key(self._difficulty_wp),
                    self.n_agents,
                    agent_lim,
                    disc_lim,
                    self.cage_radius,
                    self.ring_ang_jitter,
                    self.ring_rad_jitter,
                    self.drift_mag,
                    ptr_key(st.pos),
                    ptr_key(st.theta),
                    ptr_key(st.vel),
                    ptr_key(st.speed),
                    ptr_key(st.ang_vel),
                    ptr_key(disc_pos),
                    ptr_key(disc_vel),
                    ptr_key(drift),
                    ptr_key(evade),
                    ptr_key(hold),
                ),
            )
            launch.set_param_by_name("use_mask", use_mask)
            launch.set_param_by_name("seed", seed)
            launch.launch()
        w.mark_pos_dirty()

        self._install_obstacles()
        if not self.fused_active:
            self._prev_gap = None
            self._prev_band = None
        self.finish_reset(env_mask, obs_only=obs_only)

    def _build_obstacles(self) -> Obstacles:
        return Obstacles(
            self.disc_pos.detach(), self._disc_radius_t, vel=self.disc_vel.detach()
        ).resolve(self.world.device, self.world.dtype)

    def _install_obstacles(self) -> None:
        """Hand the current disc pose *and velocity* to the engine as a circular obstacle.

        Called from inside the captured whole-step graph (see :meth:`launch_fused`), so it
        stays in the capture-safe regime of
        :meth:`swarp.core.stepper.Stepper.set_obstacles`: unchanged obstacle count, no
        movable obstacle in the spec, no 1-D ``angle`` to broadcast. The spec is built
        once and re-installed rather than rebuilt — it aliases ``disc_pos``/``disc_vel``,
        and both ``resolve`` and ``any_movable`` memoize, so the re-install allocates
        nothing.

        The rebuild branch fires only when one of those tensors is *reassigned*, which is
        the grad path integrating the disc in torch (it needs fresh tensors for the tape).
        That is a host-side pointer compare, and on the captured path the pointers never
        move, so it cannot fire inside a capture.

        The disc is tagged ``IMMOVABLE`` by omission, and that is the load-bearing part:
        :func:`swarp.core.collisions._static_forces` dispatches on the *shape* tag and
        never reads ``obs_kind``, so the agents feel the disc exactly as they would a
        movable one — while :meth:`_launch_disc` keeps sole ownership of its motion.
        """
        if (
            self._obstacles.pos.data_ptr() != self.disc_pos.data_ptr()
            or self._obstacles.vel.data_ptr() != self.disc_vel.data_ptr()
        ):
            self._obstacles = self._build_obstacles()
        self.world.set_obstacles(self._obstacles)

    # --------------------------------------------------------- fused fast path

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        ne, na = n_envs, self.n_agents
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("gap", (ne, na)),
            Buf("band", (ne, na)),
            Buf("touch", (ne, na)),
            Buf("inband", (ne, na), "uint8", bool_view=True),
            Buf("gapmax", (ne,)),
            Buf("gapsoft", (ne,)),
            Buf("banderr", (ne,)),
            Buf("reward", (ne, na)),
            Buf("multiobj", (ne, na, N_OBJ)),
            Buf("caged", (ne,), "uint8", bool_view=True),
            Buf("done", (ne,), "uint8", bool_view=True),
            Buf("resetmask", (ne,), "uint8", reset_mask=True),
            # Scenario state. The disc pose/velocity are advanced in place by the disc
            # kernel (carry) and reassigned by the grad path's torch integrator (watch).
            Buf("disc_pos", (ne, 1, 2), "vec2", attr="disc_pos", alloc="never", carry=True,
                watch=True),
            Buf("disc_vel", (ne, 1, 2), "vec2", attr="disc_vel", alloc="never", carry=True,
                watch=True),
            # Per-episode draws: written by the reset kernel only, so not carries.
            Buf("drift", (ne, 1, 2), "vec2", attr="drift", alloc="never"),
            Buf("evade", (ne,), attr="evade", alloc="never"),
            # Advanced in place by the reward kernel, zeroed by the reset kernel.
            Buf("hold", (ne,), "uint8", attr="hold", alloc="never", carry=True),
            Buf("prevgap", (ne,), attr="_prev_gap", alloc="if_none", carry=True, watch=True),
            Buf("prevband", (ne,), attr="_prev_band", alloc="if_none", carry=True, watch=True),
        )

    def engine_carries(self) -> list[torch.Tensor]:
        """The engine's obstacle position and velocity: :meth:`_install_obstacles`
        overwrites them from inside the hook, and the *next* step's physics reads them."""
        views = self.world.obstacle_state_views()
        return [views[0], views[2]]

    def launch_fused(self, pass_: FusedPass) -> None:
        """Disc, re-install, obs, reward.

        A reset skips the disc integration (there is nothing to advance: the disc was just
        placed) and passes ``advance_prev=0``, which both rebases the shaping baselines
        for the reset envs and stops the hold counter from crediting the spawn instant.

        The reward launch is **not** gated on ``full_pass``: it owns the two broadcast
        observation columns (``gap_max``, ``caged``), which an obs-only auto-reset still
        has to refresh. ``full_pass`` gates the reward/done/multiobj writes *inside* it.
        """
        st = self.world.state_wp()  # one wrap for every launch in this pass
        if pass_.is_step:
            self._launch_disc(st)
            self._install_obstacles()  # new disc pose AND velocity for the next step
        self._launch_obs(st, full_pass=pass_.full_pass)
        self._launch_reward(advance_prev=pass_.advance_prev, full_pass=pass_.full_pass)

    def _launch_disc(self, st) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        bound = self.world_size - self.disc_radius
        launch = self._disc_launch.get(
            concrete(caging_disc_kernel, scalar),
            dim=(w.n_envs, 1),
            inputs=[
                st.pos,
                st.vel,
                pk["drift"],
                pk["evade"],
                wp.int32(self.n_agents),
                scalar(self.agent_radius),
                scalar(self.disc_radius),
                scalar(self.contact_margin),
                scalar(self.flee_radius),
                scalar(self.flee_k),
                scalar(self.contact_k),
                scalar(self.contact_c),
                scalar(self.disc_mass),
                scalar(self.linear_damping),
                scalar(self.dt),
                scalar(bound),
            ],
            outputs=[pk["disc_pos"], pk["disc_vel"]],
            device=w.device,
            key=(
                w.n_envs,
                ptr_key(st.pos),
                ptr_key(st.vel),
                ptr_key(pk["drift"]),
                ptr_key(pk["evade"]),
                self.n_agents,
                self.flee_k,
                self.dt,
                bound,
                ptr_key(pk["disc_pos"]),
                ptr_key(pk["disc_vel"]),
            ),
        )
        launch.launch()

    def _launch_obs(self, st, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        launch = self._obs_launch.get(
            concrete(caging_obs_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                pk["disc_pos"],
                pk["disc_vel"],
                pk["resetmask"],
                wp.int32(self.n_agents),
                scalar((2.0 * self.agent_radius) ** 2),
                scalar(self.cage_radius),
                scalar(self.band_half),
                scalar(_TWO_PI),
                wp.int32(full_pass),
            ],
            outputs=[pk["obs"], pk["gap"], pk["band"], pk["touch"], pk["inband"]],
            device=w.device,
            key=(
                w.n_envs,
                self.n_agents,
                ptr_key(st.pos),
                ptr_key(st.vel),
                ptr_key(pk["disc_pos"]),
                ptr_key(pk["disc_vel"]),
                ptr_key(pk["resetmask"]),
                self.agent_radius,
                self.cage_radius,
                self.band_half,
                ptr_key(pk["obs"]),
                ptr_key(pk["gap"]),
                ptr_key(pk["band"]),
                ptr_key(pk["touch"]),
                ptr_key(pk["inband"]),
            ),
        )
        launch.set_param_by_name("full_pass", wp.int32(full_pass))
        launch.launch()

    def _launch_reward(self, advance_prev: int, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        launch = self._reward_launch.get(
            concrete(caging_reward_kernel, scalar),
            dim=w.n_envs,
            inputs=[
                pk["gap"],
                pk["band"],
                pk["touch"],
                pk["inband"],
                pk["disc_pos"],
                pk["resetmask"],
                wp.int32(self.n_agents),
                scalar(1.0 / self.n_agents),
                scalar(_TWO_PI),
                scalar(self.beta),
                scalar(1.0 / self.beta),
                scalar(self.gap_threshold),
                scalar(self.wall_free_radius),
                wp.int32(self.hold_steps),
                scalar(self.gap_shaping_factor),
                scalar(self.band_shaping_factor),
                scalar(self.cage_reward),
                scalar(self.time_penalty),
                scalar(self.collision_penalty),
                wp.int32(advance_prev),
                wp.int32(full_pass),
            ],
            outputs=[
                pk["prevgap"],
                pk["prevband"],
                pk["obs"],
                pk["gapmax"],
                pk["gapsoft"],
                pk["banderr"],
                pk["caged"],
                pk["hold"],
                pk["done"],
                pk["reward"],
                pk["multiobj"],
            ],
            device=w.device,
            key=(
                w.n_envs,
                ptr_key(pk["gap"]),
                ptr_key(pk["band"]),
                ptr_key(pk["touch"]),
                ptr_key(pk["inband"]),
                ptr_key(pk["disc_pos"]),
                ptr_key(pk["resetmask"]),
                self.n_agents,
                self.beta,
                self.gap_threshold,
                self.wall_free_radius,
                self.hold_steps,
                self.gap_shaping_factor,
                self.band_shaping_factor,
                self.cage_reward,
                self.time_penalty,
                self.collision_penalty,
                ptr_key(pk["prevgap"]),
                ptr_key(pk["prevband"]),
                ptr_key(pk["obs"]),
                ptr_key(pk["gapmax"]),
                ptr_key(pk["gapsoft"]),
                ptr_key(pk["banderr"]),
                ptr_key(pk["caged"]),
                ptr_key(pk["hold"]),
                ptr_key(pk["done"]),
                ptr_key(pk["reward"]),
                ptr_key(pk["multiobj"]),
            ),
        )
        launch.set_param_by_name("advance_prev", wp.int32(advance_prev))
        launch.set_param_by_name("full_pass", wp.int32(full_pass))
        launch.launch()

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh(integrate=True, advance=True)

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        self._refresh(reset_mask=env_mask, integrate=False, advance=False)

    def _integrate_disc(self) -> None:
        """One semi-implicit Euler step of the disc, in torch (the differentiable path).

        Deliberately an independent implementation of
        :func:`~swarp.scenarios.caging_kernels.caging_disc_kernel` rather than a shared
        helper: these two *are* the parity test. The one place they are pinned to the same
        form on purpose is the flee profile, written as ``clamp(1 - d/R, min=0)**2`` here
        against ``if d < R: (1 - d/R)**2`` there — algebraically the same function, and
        the same two roundings, because the disc's trajectory feeds a threshold predicate
        that ``done`` is compared on exactly.
        """
        w = self.world
        pos, vel = w.state.pos, w.state.vel  # [E, A, 2]
        q = self.disc_pos[:, 0]  # [E, 2]
        dv = self.disc_vel[:, 0]  # [E, 2]
        rel = q.unsqueeze(1) - pos  # agent -> disc [E, A, 2]
        d = torch.sqrt((rel * rel).sum(-1)).clamp(min=1e-9)  # [E, A]
        n_hat = rel / d.unsqueeze(-1)

        wgt = (1.0 - d / self.flee_radius).clamp(min=0.0)
        flee = self.evade.unsqueeze(-1) * self.flee_k * wgt * wgt  # [E, A]

        overlap = (self.agent_radius + self.disc_radius + self.contact_margin) - d
        vn = ((dv.unsqueeze(1) - vel) * n_hat).sum(-1)  # closing normal velocity
        contact = torch.where(
            overlap > 0.0,
            self.contact_k * overlap - self.contact_c * vn,
            torch.zeros_like(overlap),
        )
        f = self.drift[:, 0] + ((flee + contact).unsqueeze(-1) * n_hat).sum(dim=1)  # [E, 2]

        dt = self.dt
        new_v = (dv + f / self.disc_mass * dt) * (1.0 - self.linear_damping * dt)
        bound = self.world_size - self.disc_radius
        new_q = (q + new_v * dt).clamp(-bound, bound)
        # Fresh tensors, not in-place: the grad path needs them for the tape. ``watch=True``
        # on both spec entries is what rebuilds the fused handles afterwards.
        self.disc_vel = new_v.unsqueeze(1)
        self.disc_pos = new_q.unsqueeze(1)
        self._install_obstacles()  # for the next step

    def _refresh(
        self,
        reset_mask: torch.Tensor | None = None,
        integrate: bool = True,
        advance: bool = True,
    ) -> None:
        w = self.world
        if integrate:
            self._integrate_disc()
        pos = w.state.pos
        q = self.disc_pos[:, 0]  # [E, 2]
        rel = pos - q.unsqueeze(1)  # disc -> agent [E, A, 2]
        r = torch.sqrt((rel * rel).sum(-1))  # [E, A]
        th = torch.atan2(rel[..., 1], rel[..., 0])  # bearings in (-pi, pi]

        # Max angular gap as the minimum wrapped CCW difference, NOT a sort. See
        # ``caging_kernels``' module docstring: the sort form agrees almost everywhere but
        # differs exactly at bearing ties and the 2pi wrap, and the diagonal is filled
        # with 2*pi rather than inf so the ``n_agents == 1`` case (gap = 2*pi) falls out of
        # the same expression on both paths.
        dd = th.unsqueeze(1) - th.unsqueeze(2)  # dd[e, i, j] = th[j] - th[i]
        wrapped = torch.where(dd < 0.0, dd + _TWO_PI, dd)
        eye = torch.eye(self.n_agents, device=w.device, dtype=torch.bool)
        wrapped = wrapped.masked_fill(eye, _TWO_PI)
        gap = wrapped.min(-1).values  # [E, A]
        gap_max = gap.max(-1).values  # [E]
        # Every exponent is <= 0 because gap <= 2*pi, so no max-subtraction is needed and
        # the surrogate stays exp/log only — a hidden ``max`` here would reintroduce
        # exactly the sparse gradient ``gap_soft`` exists to avoid.
        gap_soft = _TWO_PI + torch.log(
            torch.exp(self.beta * (gap - _TWO_PI)).sum(-1)
        ) / self.beta

        err = (r - self.cage_radius).abs() - self.band_half
        band = err.clamp(min=0.0)
        in_band = err <= 0.0
        band_err = band.mean(-1)

        # Same all-pairs squared form as formation's oracle: the touching count is a
        # discrete quantity compared exactly against the kernel, and a ``cdist`` or a
        # neighbor-list walk would round (or truncate) differently at the boundary.
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dd2 = (diff * diff).sum(-1)
        touching = (dd2 < (2.0 * self.agent_radius) ** 2).sum(-1).to(w.dtype) - 1.0

        q_norm = torch.sqrt((q * q).sum(-1))
        caged = (
            (gap_max < self.gap_threshold)
            & (in_band.sum(-1) == self.n_agents)
            & (q_norm < self.wall_free_radius)
        )

        if self._prev_gap is None:
            self._prev_gap = gap_soft.detach().clone()
            self._prev_band = band_err.detach().clone()
        gap_shaping = (self._prev_gap - gap_soft) * self.gap_shaping_factor
        band_shaping = (self._prev_band - band_err) * self.band_shaping_factor
        if reset_mask is None:
            self._prev_gap = gap_soft.detach().clone()
            self._prev_band = band_err.detach().clone()
        else:
            zeros = torch.zeros_like(gap_shaping)
            gap_shaping = torch.where(reset_mask, zeros, gap_shaping)
            band_shaping = torch.where(reset_mask, zeros, band_shaping)
            self._prev_gap = torch.where(reset_mask, gap_soft.detach(), self._prev_gap)
            self._prev_band = torch.where(reset_mask, band_err.detach(), self._prev_band)

        # The hold counter mirrors the kernel exactly: advanced only on a step (a
        # standalone reset is a full pass but must not credit the spawn instant), zeroed
        # for reset envs by the reset kernel, saturating at 255. Written in place so the
        # adopted buffer's pointer never moves.
        if advance:
            h = self.hold.int()
            self.hold.copy_(torch.where(caged, (h + 1).clamp(max=255), torch.zeros_like(h)))
        elif reset_mask is None:
            self.hold.zero_()
        else:
            self.hold.masked_fill_(reset_mask, 0)

        self._cache = {
            "gap": gap,
            "gap_max": gap_max,
            "gap_soft": gap_soft,
            "band_err": band_err,
            "in_band": in_band,
            "touching": touching,
            "caged": caged,
            "gap_shaping": gap_shaping,
            "band_shaping": band_shaping,
            "radial_err": r - self.cage_radius,
            "disc_rel": (q.unsqueeze(1) - pos),
        }

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        """Fully batched observations ``[n_envs, n_agents, obs_dim]``."""
        if self.fused_active:
            return self.fb["obs"]
        w = self.world
        s = w.state
        c = self._cache
        ne, na = w.n_envs, self.n_agents
        dv = self.disc_vel[:, 0].unsqueeze(1).expand(ne, na, 2)
        others = (s.pos[:, self._others] - s.pos.unsqueeze(2)).flatten(2)  # [E, A, 2(A-1)]
        return torch.cat(
            [
                s.pos,
                s.vel,
                c["disc_rel"],
                dv,
                c["radial_err"].unsqueeze(-1),
                torch.cos(c["gap"]).unsqueeze(-1),
                torch.sin(c["gap"]).unsqueeze(-1),
                c["gap_max"].view(ne, 1, 1).expand(ne, na, 1),
                c["caged"].to(w.dtype).view(ne, 1, 1).expand(ne, na, 1),
                others,
            ],
            dim=-1,
        )

    def _terms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(gap_shaping, band_shaping, cage, collision)`` — the reward, factorized.

        ``rewards``, ``agent_reward``/``global_reward`` and ``multiobj_reward`` all read
        this one derivation, so the scalar reward and the objective vector cannot disagree
        about what a term is worth. ``time_penalty`` is a constant and stays a literal.
        """
        c = self._cache
        cage = self.cage_reward * c["caged"].to(self.world.dtype)
        return (
            c["gap_shaping"],
            c["band_shaping"],
            cage,
            self.collision_penalty * c["touching"],
        )

    def rewards(self) -> torch.Tensor:
        return self.fb["reward"] if self.fused_active else super().rewards()

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        return self._terms()[3][:, agent_idx]

    def global_reward(self) -> torch.Tensor:
        gap_sh, band_sh, cage, _ = self._terms()
        return gap_sh + band_sh + cage + self.time_penalty

    def done(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["done_bool"]
        return self.hold >= self.hold_steps

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {
                "gap_max": self.fb["gapmax"],
                "gap_soft": self.fb["gapsoft"],
                "band_error": self.fb["banderr"],
                "caged": self.fb["caged_bool"],
                "hold": self.fb["hold"],
                "disc_pos": self.disc_pos,
                "collisions": self.fb["touch"],
                "multiobj_reward": self.fb["multiobj"],
            }
        c = self._cache
        gap_sh, band_sh, cage, col = self._terms()
        # Per-agent objective vector [n_envs, n_agents, N_OBJ]: the five reward terms kept
        # separate so a trainer logs one column each and a shaping imbalance shows up on
        # iteration 1 rather than hour 3. Its sum over the last dim equals the scalar
        # per-agent reward exactly.
        shared = [gap_sh, band_sh, cage]
        cols = [t.unsqueeze(-1).expand_as(col) for t in shared]
        cols.append(torch.full_like(col, self.time_penalty))
        cols.append(col)
        return {
            "gap_max": c["gap_max"],
            "gap_soft": c["gap_soft"],
            "band_error": c["band_err"],
            "caged": c["caged"],
            "hold": self.hold,
            "disc_pos": self.disc_pos,
            "collisions": c["touching"],
            "multiobj_reward": torch.stack(cols, dim=-1),
        }
