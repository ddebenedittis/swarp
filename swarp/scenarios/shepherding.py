"""Shepherding: shepherds herd fleeing sheep into a pen.

The one **reactive** environment in the package. Every other scenario's world is either
inert (navigation's goals, formation's targets) or purely contact-driven (transport's
package, push-t's T): it only ever does what the agents' contacts make it do. Here the
sheep run their own scripted policy, so the world *pushes back* — a shepherd that charges
straight at the flock scatters it, and the reward penalizes exactly that. Learning the
task means learning to approach obliquely and hold a line.

The sheep are **not** agents. There is no action-override hook in this engine (``world.
action`` is written by ``Environment.step`` after the only pre-physics hook fires), so a
sheep cannot be an agent whose action a scenario intercepts. They are instead modelled the
way :mod:`swarp.scenarios.transport` models its packages, and every step:

  1. the sheep are installed as circular obstacles, so the Warp agent step pushes
     shepherds *off* them (the existing soft agent-obstacle contact);
  2. after the step, the force on each sheep is summed — the reaction of that same
     spring-damper contact (Newton's third law), plus the two terms the engine knows
     nothing about: **flee** repulsion from every shepherd inside ``flee_radius``
     (summed, then magnitude-capped at ``flee_gain``, so a sheep can never outrun the
     shepherds herding it), and **separation** from every other sheep inside
     ``sep_radius`` — and the sheep is integrated in torch (semi-implicit Euler +
     damping, then a ``max_sheep_speed`` cap that keeps the flock slower than the
     shepherds herding it);
  3. the updated sheep positions are installed for the next step.

Deliberately *not* ``ObstacleKind.MOVABLE``: the engine integrates those from contact
reaction only, with nowhere to add the flee force; the body integrator is
``record_tape=False`` so it vanishes on a taped step; and an install carrying
``kind=MOVABLE`` reads ``any_movable`` back to the host, which is not capture-safe. The
transport pattern above avoids all three. A sheep tracks **position only** — a circular
body under frictionless normal contact has zero lever arm
(:mod:`swarp.core.bodies`), so there is no rotation to carry and none is allocated.

Coupling is staggered by one step (shepherds see last step's sheep positions). The sheep
integration is plain torch, so gradients flow sheep->shepherd->action across a rollout
(BPTT); the intra-step shepherd-avoids-sheep force is not taped (obstacles are constants
inside a Warp step), the same documented limitation transport carries.

Spawn difficulty is a curriculum knob: :meth:`ShepherdingScenario.set_spawn_scale`
shrinks the ring the sheep are drawn on, so a novice policy starts with the sheep already
at the pen wall and actually experiences the pen bonus, and annealing the scale back to
``1.0`` restores the constructor's distribution exactly.

The three steps above describe the **torch reference path** — the parity oracle, and the
only path that carries gradients. The no-grad hot path does the same physics in
``shepherding_force_kernel`` + ``shepherding_body_kernel``
(:mod:`swarp.scenarios.shepherding_kernels`) instead, entirely on-device inside the fused
whole-step hook. The two are tested against each other in
``tests/scenarios/test_shepherding_fused.py``.
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
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario
from swarp.scenarios.shepherding_kernels import (
    shepherding_body_kernel,
    shepherding_force_kernel,
    shepherding_obs_kernel,
    shepherding_reset_kernel,
    shepherding_reward_kernel,
)

_TWO_PI = 2.0 * math.pi


class ShepherdingScenario(FusedScenario):
    """Drive ``n_sheep`` fleeing sheep into a disc-shaped pen with ``n_agents`` shepherds.

    ``n_agents`` counts **shepherds only**; the sheep are scenario-owned bodies, not
    agents, and ``n_sheep`` is their own knob.
    """

    # The fused and torch paths compute the same physics but reduce the shepherd and
    # sheep contributions in different orders, so they diverge at the ulp scale from the
    # first step. Transport lives with the default 1e-5/1e-6 because its only force is a
    # contact that damping pulls back together; here the *flee* term is active at a
    # distance, so a ulp of positional difference feeds straight back into a force the
    # sheep feels every step (and through the flock, into its neighbours) rather than
    # only when someone is touching it. Measured over the 20-step parity rollout the
    # worst field difference is ~1e-7, i.e. still comfortably inside the default; the
    # bound is loosened by one decade anyway because that number *grows with the
    # horizon* here rather than settling, and a 10x margin on an absolute 1e-6 is thin
    # enough that a longer rollout or a different GPU would trip it for no real reason.
    parity_rtol: float = 1e-4
    parity_atol: float = 1e-5

    def __init__(
        self,
        n_agents: int = 3,
        n_sheep: int = 5,
        agent_radius: float = 0.05,
        sheep_radius: float = 0.05,
        sheep_mass: float = 1.0,
        world_size: float = 1.0,
        max_speed: float = 1.0,
        pen_radius: float = 0.2,
        contact_k: float = 100.0,
        contact_c: float = 5.0,
        contact_margin: float = 0.01,
        flee_radius: float = 0.35,
        flee_gain: float = 1.0,
        sep_radius: float = 0.2,
        sep_gain: float = 1.0,
        cohesion_gain: float = 0.5,
        linear_damping: float = 2.0,
        max_sheep_speed: float | None = None,
        pos_shaping_factor: float = 1.0,
        pen_reward: float = 0.5,
        scatter_penalty: float = 0.1,
    ) -> None:
        self.n_agents = n_agents
        self.n_sheep = n_sheep
        self.agent_radius = agent_radius
        self.sheep_radius = sheep_radius
        self.sheep_mass = sheep_mass
        self.world_size = world_size
        self.max_speed = max_speed
        self.pen_radius = pen_radius
        # contact_k/contact_c ARE the engine's WorldConfig.collision_k/collision_c — the
        # same spring-damper law, named for the shepherd<->sheep contact that dominates
        # here. contact_margin is NOT the engine's collision_margin (derived from
        # agent_radius in make_world): it is this scenario's own shepherd<->sheep
        # activation gap, used by the torch reference path below.
        self.contact_k = contact_k
        self.contact_c = contact_c
        self.contact_margin = contact_margin
        # The flee law: every shepherd within flee_radius pushes the sheep directly away
        # with magnitude flee_gain * (1 - d / flee_radius) — full strength on top of the
        # sheep, tapering linearly to nothing at the radius, so there is no discontinuity
        # at the boundary for the parity paths to disagree about. Linear rather than
        # 1/d^2 on purpose: an inverse-square law is unbounded at contact, where the
        # spring is already applying a large force, and the sum of the two blows the
        # explicit integrator up at any usable dt.
        #
        # The **sum** over shepherds is then magnitude-capped at flee_gain (in both
        # paths), which is what makes the task solvable at all. A sustained force f gives
        # a terminal speed of f / (sheep_mass * linear_damping), so the cap fixes the
        # fastest a sheep can ever run at
        #
        #     flee_gain / (sheep_mass * linear_damping) = 1.0 / (1.0 * 2.0) = 0.5
        #
        # against a shepherd's max_speed of 1.0 — half, independent of how many
        # shepherds converge. Uncapped and at the old flee_gain=2.0, three shepherds
        # summed to a measured peak sheep speed of 2.3 (> 2x max_speed): a flock that
        # literally cannot be cornered, and a 6000-iteration MAPPO run that plateaued at
        # ~7% penned. Herding only works when the flock is slower than the dogs.
        # ``test_sheep_cannot_outrun_the_shepherds`` pins it. Contact stays *outside* the
        # cap: it is the direct push the shepherds actually herd with, it is
        # self-limiting (the spring only acts while someone overlaps the sheep), and
        # capping it would make a sheep block rather than yield.
        self.flee_radius = flee_radius
        self.flee_gain = flee_gain
        # Separation uses the same falloff shape between sheep, so the flock spreads to
        # roughly sep_radius rather than collapsing to a point under cohesion. Left
        # uncapped: unlike flee it is not sustained by anything a policy controls — it
        # vanishes as the sheep it acts on move apart — and at sep_gain=1.0 a single term
        # already sits at the flee cap. Equilibrium against cohesion is where
        # sep_gain * (1 - d/sep_radius) == cohesion_gain * d/2, i.e. d ~= 0.19 here.
        self.sep_radius = sep_radius
        self.sep_gain = sep_gain
        # Mild cohesion toward the flock centroid; it is what keeps a scattered flock
        # recoverable instead of turning one bad approach into a lost episode. Identically
        # zero for n_sheep=1 (the centroid is the sheep).
        self.cohesion_gain = cohesion_gain
        self.linear_damping = linear_damping
        # A sheep's top running speed, and the hard guarantee behind
        # ``test_sheep_cannot_outrun_the_shepherds``. The flee cap above already bounds
        # the *sustained* speed at 0.5, but it says nothing about a **contact transient**:
        # a shepherd arriving at full speed compresses a k=100 spring several centimetres
        # before the sheep responds, and the measured peak was 2.07 even with the flee sum
        # capped and a single shepherd chasing a single sheep. Softening the contact
        # instead was the wrong trade — contact_k is also the engine's agent<->agent
        # collision_k, so the k=15 that would have sufficed lets shepherds interpenetrate
        # — and even then it only bought a 3% margin. A speed cap is independent of dt, of
        # the stiffness and of how many shepherds pile on, and it is the standard
        # formulation (Strombom's sheep move at a fixed top speed). Default: 75% of the
        # shepherds' max_speed, so a shepherd can always close on a fleeing sheep, with
        # the flee-only terminal 0.5 left comfortably underneath — the cap catches
        # transients and leaves the flee law itself shaping the behaviour.
        self.max_sheep_speed = max_sheep_speed if max_sheep_speed is not None else 0.75 * max_speed
        self.pos_shaping_factor = pos_shaping_factor
        self.pen_reward = pen_reward
        self.scatter_penalty = scatter_penalty
        #: Current fraction of the constructor's sheep-spawn spread (see
        #: :meth:`set_spawn_scale`). ``1.0`` is the task as constructed.
        self.spawn_scale = 1.0
        self._set_spawn_geometry(1.0)

    # ------------------------------------------------------------ spawn geometry

    def _set_spawn_geometry(self, scale: float) -> None:
        """Recompute every spawn constant the reset kernel takes, at spread ``scale``.

        Split out of ``__init__`` because :meth:`set_spawn_scale` needs exactly this set
        recomputed and nothing else, and re-derives it from the constructor's parameters
        rather than undoing the previous scale by inverse arithmetic. Only
        ``_sheep_span`` depends on ``scale``; the rest are here so there is one place
        that owns the reset kernel's geometry.
        """
        r, ws = self.sheep_radius, self.world_size
        self._spawn_lim = ws - 2.0 * self.agent_radius  # shepherd spawn box
        self._pen_lim = max(0.0, ws - self.pen_radius)  # keeps the pen disc inside
        self._sheep_bound = ws - r  # sheep position clamp
        # Sheep spawn on a ring about the pen: never nearer than one sheep diameter
        # outside the pen wall (so an episode never starts already solved), and never
        # further than ``scale`` times the constructor's spread. ``scale=1.0`` multiplies
        # by exactly 1.0, which is exact in IEEE754 — the restored distribution is
        # bit-for-bit the constructor's, not merely close.
        self._sheep_min_r = self.pen_radius + 2.0 * r
        self._sheep_span = scale * max(0.0, ws - self._sheep_min_r)

    def set_spawn_scale(self, scale: float) -> None:
        """Shrink (or restore) the sheep spawn ring to ``scale`` times its spread.

        The curriculum hook the MAPPO trainer drives, mirroring
        :meth:`~swarp.scenarios.giveway.GiveWayScenario.set_corridor_scale`. At a small
        scale every sheep starts just outside the pen wall, so a novice policy actually
        *experiences* the pen bonus instead of only ever seeing shaping; annealing back
        to ``scale=1.0`` restores the real task. ``set_spawn_scale(1.0)`` reproduces the
        constructor's distribution exactly for a given seed — the derivation is re-run
        from the constructor's parameters, and the RNG stream is untouched (the scale
        rides on the *span* of an existing draw, it does not add or reorder one).

        ``scale=0`` is a legal, maximally easy stage: every sheep starts at exactly
        ``pen_radius + 2 * sheep_radius`` from the pen centre. Values above 1 are allowed
        (the world-bounds clamp in the reset kernel absorbs them) but pointless — the
        constructor's spread already reaches the wall.

        Host-side only — call it between batches, never inside a capture. Unlike
        give-way's corridor scale, none of this reaches a kernel that runs *inside* the
        whole-step graph: these scalars are read by the reset kernel alone, which is
        eager (``supports_graph_reset()`` is ``False``), so there is no device buffer to
        keep them in and no recapture to trigger. What it *does* have to survive is the
        reset launch's :class:`~swarp.core.cached_launch.CachedLaunch`, which packs its
        arguments once — that is why ``_sheep_span`` and ``_sheep_min_r`` are entries in
        that launch's key, so changing a scale mismatches the key and repacks instead of
        silently replaying the old spread. ``test_spawn_scale_shrinks_the_ring`` is what
        pins it.
        """
        if scale < 0.0:
            raise ValueError(f"spawn scale must be non-negative, got {scale}")
        self._set_spawn_geometry(scale)
        self.spawn_scale = scale

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
        margin = 0.5 * self.agent_radius
        # neighbor reach must cover shepherd<->sheep contact so the step sees it
        reach = 2.0 * max(self.agent_radius, self.sheep_radius) + margin
        cfg = WorldConfig(
            collisions=True,
            collision_k=self.contact_k,
            collision_c=self.contact_c,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            neighbor_radius=reach,
            max_neighbors=min(32, max(4, self.n_agents)),
        ).override_with(world_config)
        self.dt = dt
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        # Flock state and pen: allocated here (n_envs is known) so the fused spec can
        # adopt them, and written in place by every reset so their handles stay valid.
        tt = {"device": device, "dtype": dtype}
        self.sheep_pos = torch.zeros(n_envs, self.n_sheep, 2, **tt)
        self.sheep_vel = torch.zeros(n_envs, self.n_sheep, 2, **tt)
        self.pen_pos = torch.zeros(n_envs, 2, **tt)
        # ``world.goals`` is this scenario's **render-side mirror of the pen centre**,
        # one row per shepherd, written by the reset kernel. It is not an input to obs,
        # reward, done or the fused kernels — ``pen_pos`` is the authoritative copy those
        # read — but the viewer's existing goal overlay draws it, which is what puts a
        # pen marker in a rendered frame without any renderer change. It is also the
        # honest reading: the pen is the shepherds' one shared goal. Allocated here (not
        # in reset_world) so the fused spec can adopt it.
        self.world.goals = torch.zeros(n_envs, self.n_agents, 2, **tt)
        # ONE retained obstacle spec over sheep_pos, re-installed rather than rebuilt:
        # both ``resolve`` and ``any_movable`` memoize, so a re-install allocates nothing
        # and costs no device->host sync. Built here so the single count-changing install
        # happens before any graph capture. It aliases sheep_pos, so writing new sheep
        # positions in place is all a "move the flock" update needs. No ``kind``: the
        # engine must NOT integrate these (this scenario does), and a spec without a kind
        # field skips the ``any_movable`` reduction entirely.
        self._sheep_radius = torch.full((self.n_sheep,), self.sheep_radius, **tt)
        self._obstacles = Obstacles(self.sheep_pos.detach(), self._sheep_radius).resolve(
            device, dtype
        )
        self._prev_dist: torch.Tensor | None = None
        self._cache: dict[str, torch.Tensor] | None = None
        # The reset kernel's spawn constants (all host-side, so the reset stays
        # sync-free) are owned by ``_set_spawn_geometry``, which ``__init__`` has already
        # run and which ``set_spawn_scale`` re-runs — deriving them here too would give a
        # curriculum change one copy to miss.
        # Cached, repack-once launches for the eager reset path — see
        # swarp/core/cached_launch.py.
        self._reset_launch = CachedLaunch()
        self._obs_launch = CachedLaunch()
        self._reward_launch = CachedLaunch()
        return self.world

    @property
    def obs_dim(self) -> int:
        return 6 + 4 * self.n_sheep

    # ------------------------------------------------------------------ reset

    def reset_world(self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False) -> None:
        """Masked reset in one Warp launch (see :mod:`swarp.scenarios.reset_kernels`).

        The flock buffers are written **in place** by the kernel, so the fused path's
        cached Warp handles and the whole-step graph stay valid across resets; the grad
        path's ``_refresh`` still reassigns them (fresh tensors for the tape), which the
        framework's handle resync catches on the next no-grad step.
        """
        w = self.world
        mask, use_mask = self.reset_mask_wp(env_mask)
        st = w.state_wp()
        scalar = w.wp_dtype
        vec2 = VEC2[scalar]
        if self.fused_active:
            # Reuse the fused spec's cached, pointer-resynced handles instead of
            # re-wrapping these watched tensors on every reset (see navigation's
            # ``_launch_reset`` for the same fix and its rationale). ``sync_fused_
            # handles`` must run first: it is what notices a grad-path reassignment
            # and rebuilds the handle before this kernel writes through it.
            self.ensure_fused()
            self.sync_fused_handles()
            sheep_pos = self._wp["sheep_pos"]
            sheep_vel = self._wp["sheep_vel"]
            pen = self._wp["pen"]
            goals = self._wp["goals"]
        else:
            sheep_pos = wp.from_torch(self.sheep_pos.contiguous(), dtype=vec2)
            sheep_vel = wp.from_torch(self.sheep_vel.contiguous(), dtype=vec2)
            pen = wp.from_torch(self.pen_pos.contiguous(), dtype=vec2)
            goals = wp.from_torch(w.goals.contiguous(), dtype=vec2)
        seed = wp.int32(w.next_kernel_seed())
        with torch_stream_scope(w.device):
            launch = self._reset_launch.get(
                concrete(shepherding_reset_kernel, scalar),
                dim=w.n_envs,
                inputs=[
                    mask,
                    use_mask,
                    seed,
                    scalar(self._spawn_lim),
                    scalar(self._pen_lim),
                    scalar(self._sheep_bound),
                    scalar(self._sheep_min_r),
                    scalar(self._sheep_span),
                    scalar(_TWO_PI),
                    wp.int32(self.n_agents),
                    wp.int32(self.n_sheep),
                    st.pos,
                    st.vel,
                    sheep_pos,
                    sheep_vel,
                    pen,
                    goals,
                ],
                device=w.device,
                key=(
                    w.n_envs,
                    ptr_key(mask),
                    self._spawn_lim,
                    self._pen_lim,
                    self._sheep_bound,
                    self._sheep_min_r,
                    self._sheep_span,
                    self.n_agents,
                    self.n_sheep,
                    ptr_key(st.pos),
                    ptr_key(st.vel),
                    ptr_key(sheep_pos),
                    ptr_key(sheep_vel),
                    ptr_key(pen),
                    ptr_key(goals),
                ),
            )
            launch.set_param_by_name("use_mask", use_mask)
            launch.set_param_by_name("seed", seed)
            launch.launch()
        w.mark_pos_dirty()

        self._install_obstacles()
        if not self.fused_active:
            self._prev_dist = None
        self.finish_reset(env_mask, obs_only=obs_only)

    def _install_obstacles(self) -> None:
        """Hand the current sheep positions to the engine as circular obstacles.

        Called from inside the captured whole-step graph (see :meth:`launch_fused`), so it
        must stay in the capture-safe regime of :meth:`swarp.core.stepper.Stepper.
        set_obstacles`: an unchanged obstacle count, and a spec carrying no movable
        obstacles and no 1-D ``angle`` to broadcast. All three hold by construction — the
        flock size is fixed at ``make_world``, the spec declares no ``kind`` and no
        ``angle``.

        The spec is built once, in ``make_world``, and **re-installed** rather than
        rebuilt: it aliases ``sheep_pos``, and both ``Obstacles.resolve`` and
        ``Obstacles.any_movable`` memoize, so a re-install allocates nothing and costs no
        device->host sync.

        The retained spec is rebuilt only when ``sheep_pos`` is *reassigned*, which is the
        grad path integrating the flock in torch (it needs fresh tensors for the tape).
        That is a host-side pointer compare, and on the captured path the pointer never
        moves, so the rebuild branch cannot fire inside a capture.
        """
        if self._obstacles.pos.data_ptr() != self.sheep_pos.data_ptr():
            self._obstacles = Obstacles(self.sheep_pos.detach(), self._sheep_radius).resolve(
                self.world.device, self.world.dtype
            )
        self.world.set_obstacles(self._obstacles)

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh(integrate=True)

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        self._refresh(reset_mask=env_mask, integrate=False)

    # --------------------------------------------------------- fused fast path

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        ne, na, k = n_envs, self.n_agents, self.n_sheep
        flock = {"alloc": "never", "carry": True, "watch": True}
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("reward", (ne, na)),
            Buf("dist", (ne,)),
            Buf("penned", (ne,)),
            Buf("done", (ne,), "uint8", bool_view=True),
            Buf("resetmask", (ne,), "uint8", reset_mask=True),
            # Scratch between the force and body launches. Framework-owned and fully
            # rewritten every pass, so it is not a carry.
            Buf("force", (ne, k, 2), "vec2"),
            # Flock state and pen: the scenario's own tensors, advanced in place by the
            # body kernel and reassigned by the grad path's torch integrator.
            Buf("sheep_pos", (ne, k, 2), "vec2", attr="sheep_pos", **flock),
            Buf("sheep_vel", (ne, k, 2), "vec2", attr="sheep_vel", **flock),
            Buf("pen", (ne, 2), "vec2", attr="pen_pos", alloc="never", watch=True),
            # The render-side pen mirror. Adopted (never allocated here) so the reset
            # kernel writes through the same cached handle as everything else; no kernel
            # on the step path reads it, so it is neither a carry nor an obs input.
            Buf("goals", (ne, na, 2), "vec2", attr="world.goals", alloc="never", watch=True),
            Buf("prev", (ne, k), attr="_prev_dist", alloc="if_none", carry=True, watch=True),
        )

    def engine_carries(self) -> list[torch.Tensor]:
        """The engine's obstacle positions: ``_install_obstacles`` overwrites them from
        inside the hook, and the *next* step's physics reads them."""
        return [self.world.obstacle_state_views()[0]]

    def launch_fused(self, pass_: FusedPass) -> None:
        """Force, integrate, re-install the flock, then obs and reward.

        A reset skips the flock advance (there is nothing to integrate: the sheep were
        just placed) and passes ``advance_prev=0`` so the reward kernel *rebases* the
        shaping baseline for the reset envs instead of differencing against a stale one.
        The ``_install_obstacles`` in the middle is an ordinary line here, running inside
        ``wp.ScopedCapture`` on a step — see its docstring for why that is legal.
        """
        st = self.world.state_wp()  # one wrap for every launch in this pass
        if pass_.is_step:
            self._launch_force(st)
            self._launch_body()
            self._install_obstacles()  # updated sheep positions for the next step
        self._launch_obs(st)
        self._launch_reward(advance_prev=pass_.advance_prev, full_pass=pass_.full_pass)

    def _launch_force(self, st) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        wp.launch(
            concrete(shepherding_force_kernel, scalar),
            dim=(w.n_envs, self.n_sheep),
            inputs=[
                st.pos,
                st.vel,
                pk["sheep_pos"],
                pk["sheep_vel"],
                wp.int32(self.n_agents),
                wp.int32(self.n_sheep),
                scalar(self.agent_radius),
                scalar(self.sheep_radius),
                scalar(self.contact_margin),
                scalar(self.contact_k),
                scalar(self.contact_c),
                scalar(self.flee_radius),
                scalar(self.flee_gain),
                scalar(self.sep_radius),
                scalar(self.sep_gain),
                scalar(self.cohesion_gain),
            ],
            outputs=[pk["force"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_body(self) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        bound = self.world_size - self.sheep_radius
        wp.launch(
            concrete(shepherding_body_kernel, scalar),
            dim=(w.n_envs, self.n_sheep),
            inputs=[
                pk["force"],
                scalar(self.sheep_mass),
                scalar(self.linear_damping),
                scalar(self.max_sheep_speed),
                scalar(self.dt),
                scalar(bound),
            ],
            outputs=[pk["sheep_pos"], pk["sheep_vel"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_obs(self, st) -> None:
        w = self.world
        sheep_pos, pen = self._wp["sheep_pos"], self._wp["pen"]
        obs = self._wp["obs"]
        # No per-call-varying argument at all: pointer-stable persistent-mode handles.
        launch = self._obs_launch.get(
            concrete(shepherding_obs_kernel, w.wp_dtype),
            dim=(w.n_envs, self.n_agents),
            inputs=[st.pos, st.vel, sheep_pos, pen, wp.int32(self.n_sheep)],
            outputs=[obs],
            device=w.device,
            key=(
                w.n_envs,
                self.n_agents,
                ptr_key(st.pos),
                ptr_key(st.vel),
                ptr_key(sheep_pos),
                ptr_key(pen),
                self.n_sheep,
                ptr_key(obs),
            ),
        )
        launch.launch()

    def _launch_reward(self, advance_prev: int, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        sheep_pos, pen, resetmask = pk["sheep_pos"], pk["pen"], pk["resetmask"]
        prev, reward, done = pk["prev"], pk["reward"], pk["done"]
        dist, penned = pk["dist"], pk["penned"]
        launch = self._reward_launch.get(
            concrete(shepherding_reward_kernel, scalar),
            dim=w.n_envs,
            inputs=[
                sheep_pos,
                pen,
                resetmask,
                wp.int32(self.n_agents),
                wp.int32(self.n_sheep),
                scalar(self.pos_shaping_factor),
                scalar(self.pen_radius),
                scalar(self.pen_reward),
                scalar(self.scatter_penalty),
                wp.int32(advance_prev),
                wp.int32(full_pass),
            ],
            outputs=[prev, reward, done, dist, penned],
            device=w.device,
            key=(
                w.n_envs,
                ptr_key(sheep_pos),
                ptr_key(pen),
                ptr_key(resetmask),
                self.n_agents,
                self.n_sheep,
                self.pos_shaping_factor,
                self.pen_radius,
                self.pen_reward,
                self.scatter_penalty,
                ptr_key(prev),
                ptr_key(reward),
                ptr_key(done),
                ptr_key(dist),
                ptr_key(penned),
            ),
        )
        launch.set_param_by_name("advance_prev", wp.int32(advance_prev))
        launch.set_param_by_name("full_pass", wp.int32(full_pass))
        launch.launch()

    def _refresh(self, reset_mask: torch.Tensor | None = None, integrate: bool = True) -> None:
        w = self.world
        pos, vel = w.state.pos, w.state.vel  # [n_envs, n_agents, 2]

        if integrate:
            q = self.sheep_pos  # [E, K, 2]
            rel = q.unsqueeze(1) - pos.unsqueeze(2)  # shepherd->sheep [E, A, K, 2]
            dist = rel.norm(dim=-1).clamp(min=1e-9)  # [E, A, K]
            n_hat = rel / dist.unsqueeze(-1)
            # (1) contact: the reaction of the same spring-damper the agent step applies
            overlap = (self.agent_radius + self.sheep_radius + self.contact_margin) - dist
            active = (overlap > 0).to(w.dtype)
            fmag = self.contact_k * overlap.clamp(min=0.0)
            rel_vel = self.sheep_vel.unsqueeze(1) - vel.unsqueeze(2)  # [E, A, K, 2]
            vn = (rel_vel * n_hat).sum(-1)
            f_contact = ((fmag - self.contact_c * vn) * active).unsqueeze(-1) * n_hat
            # (2) flee: linear falloff repulsion from every shepherd inside flee_radius,
            # acting well beyond contact range. This is the reactive part of the world.
            fall = (1.0 - dist / self.flee_radius).clamp(min=0.0)
            near = (dist < self.flee_radius).to(w.dtype)
            f_flee = ((self.flee_gain * fall * near).unsqueeze(-1) * n_hat).sum(dim=1)
            # Cap the *summed* flee force at flee_gain — N converging shepherds must not
            # stack N times the panic (see the flee_gain comment in __init__ for why
            # that made the task unsolvable). The kernel does the same clamp at the same
            # point in the same order: below the cap the scale is exactly 1.0, so this is
            # bit-identical to the unclamped sum there, not merely close.
            flee_mag = f_flee.norm(dim=-1, keepdim=True)
            f_flee = f_flee * (self.flee_gain / flee_mag.clamp(min=1e-9)).clamp(max=1.0)
            # (3) separation: same falloff shape between sheep, so the flock does not
            # collapse to a point. The diagonal is dropped explicitly (an ``eye`` mask,
            # matching the kernel's ``j != k``) rather than by testing d > 0.
            sep_rel = q.unsqueeze(2) - q.unsqueeze(1)  # [E, K(i), K(j), 2], j -> i
            sd = sep_rel.norm(dim=-1)  # [E, K, K]
            sdc = sd.clamp(min=1e-9)
            eye = torch.eye(self.n_sheep, device=w.device, dtype=torch.bool)
            s_active = ((sd < self.sep_radius) & ~eye).to(w.dtype)
            s_fall = (1.0 - sd / self.sep_radius).clamp(min=0.0)
            s_hat = sep_rel / sdc.unsqueeze(-1)
            f_sep = ((self.sep_gain * s_fall * s_active).unsqueeze(-1) * s_hat).sum(dim=2)
            # (4) cohesion toward the flock centroid; identically zero for one sheep.
            f_coh = self.cohesion_gain * (q.mean(dim=1, keepdim=True) - q)

            f_total = (f_contact.sum(dim=1) + f_flee) + f_sep + f_coh
            dt = self.dt
            self.sheep_vel = (self.sheep_vel + f_total / self.sheep_mass * dt) * (
                1.0 - self.linear_damping * dt
            )
            # Speed cap, applied to the integrated velocity before it moves the position
            # — same place, same order as the kernel. Below the cap the scale is exactly
            # 1.0, so this is bit-identical to the uncapped velocity there.
            speed = self.sheep_vel.norm(dim=-1, keepdim=True)
            self.sheep_vel = self.sheep_vel * (
                self.max_sheep_speed / speed.clamp(min=1e-9)
            ).clamp(max=1.0)
            b = self.world_size - self.sheep_radius
            self.sheep_pos = (self.sheep_pos + self.sheep_vel * dt).clamp(-b, b)
            self._install_obstacles()  # for the next step

        pen = self.pen_pos.unsqueeze(1)  # [E, 1, 2]
        dist_to_pen = (self.sheep_pos - pen).norm(dim=-1)  # [E, K]
        if self._prev_dist is None:
            self._prev_dist = dist_to_pen.detach().clone()
        shaping = (self._prev_dist - dist_to_pen) * self.pos_shaping_factor
        if reset_mask is None:
            self._prev_dist = dist_to_pen.detach().clone()
        else:
            shaping = torch.where(reset_mask.unsqueeze(-1), torch.zeros_like(shaping), shaping)
            self._prev_dist = torch.where(
                reset_mask.unsqueeze(-1), dist_to_pen.detach(), self._prev_dist
            )

        penned = dist_to_pen < self.pen_radius  # [E, K]
        centroid = self.sheep_pos.mean(dim=1, keepdim=True)
        spread = (self.sheep_pos - centroid).norm(dim=-1).mean(dim=-1)  # [E]
        self._cache = {
            "dist_to_pen": dist_to_pen.mean(dim=-1),  # [E]
            "penned_frac": penned.to(w.dtype).mean(dim=-1),  # [E]
            "penned_count": penned.to(w.dtype).sum(dim=-1),  # [E]
            "all_penned": penned.all(dim=-1),  # [E]
            "shaping": shaping.sum(dim=-1),  # [E]
            "spread": spread,
            "sheep_rel": (self.sheep_pos.unsqueeze(1) - pos.unsqueeze(2)).reshape(
                w.n_envs, w.n_agents, self.n_sheep * 2
            ),
        }

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["obs"]
        w = self.world
        s = w.state
        pen_rel = self.pen_pos.unsqueeze(1) - s.pos  # [E, A, 2]
        sheep_to_pen = (
            (self.pen_pos.unsqueeze(1) - self.sheep_pos)
            .reshape(w.n_envs, 1, self.n_sheep * 2)
            .expand(-1, w.n_agents, -1)
        )
        return torch.cat([s.pos, s.vel, pen_rel, self._cache["sheep_rel"], sheep_to_pen], dim=-1)

    def global_reward(self) -> torch.Tensor:
        c = self._cache
        return (
            c["shaping"] + self.pen_reward * c["penned_count"] - self.scatter_penalty * c["spread"]
        )

    def rewards(self) -> torch.Tensor:
        return self.fb["reward"] if self.fused_active else super().rewards()

    def done(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["done_bool"]
        return self._cache["all_penned"]

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {
                "sheep_penned": self.fb["penned"],
                "sheep_dist_to_pen": self.fb["dist"],
            }
        return {
            "sheep_penned": self._cache["penned_frac"],
            "sheep_dist_to_pen": self._cache["dist_to_pen"],
        }
