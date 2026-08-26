"""Push-T: agents push a T-shaped rigid body to a target pose.

The T is a **movable compound obstacle** (:class:`~swarp.core.config.ObstacleKind`):
two oriented ``BOX`` shapes tagged with the same body id and placed by their offsets in
the body frame, so :mod:`swarp.core.bodies` integrates one rigid pose for the pair from
the reaction of the very same agent contacts the collision kernel applies. That happens
*inside* the substep loop, so the pose an agent collides against is at most one substep
old — a body advanced once per env step instead sweeps its surface across agents that
cannot react until the next step, which shows up as visible interpenetration.

The scenario therefore owns no physics on the no-grad path: it seeds the body on reset
(:meth:`_install_obstacles`), reads the engine's state back into ``tee_*``
(:meth:`_sync_from_engine`), and computes obs/reward from it. Mass splits by area and
inertia comes from the parallel-axis theorem, both about the area centroid, which is the
body origin — so a contact above it produces the torque that makes the orientation half
of the task solvable.

Contacts are **frictionless** — normal-only spring plus normal damping, as everywhere
else in the engine; see :mod:`swarp.core.collisions` for the contact law (linearly
implicit damping, clamped repulsive, closing-velocity based, depth-saturated).

Gradients: ``Stepper`` skips movable bodies on a taped step (body state is advanced with
``record_tape=False``), so on the grad path the scenario integrates the T itself in plain
torch (:meth:`_refresh`) and hands the result back. BPTT flows T->agent->action across a
rollout; the intra-step agent-avoids-T force is still not taped — the same documented
limitation as transport, not a full in-tape rigid body.

The stiff contact (``contact_k`` 8000) needs ``substeps >= 8`` at ``dt=0.05``; below that
it is unstable, and ``contact_max_overlap`` must stay above the depth a velocity-mode
agent settles at or agents walk straight through the T. Both are derived in
:meth:`make_world`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from swarp.core.bodies import body_state_gather_kernel
from swarp.core.config import ObstacleKind, Obstacles, ObstacleShape, WorldConfig
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario
from swarp.scenarios.pusht_kernels import (
    pusht_obs_kernel,
    pusht_reward_kernel,
)


class PushTScenario(FusedScenario):
    def __init__(
        self,
        n_agents: int = 4,
        agent_radius: float = 0.025,
        bar_len: float = 0.40,
        bar_width: float = 0.12,
        stem_len: float = 0.28,
        stem_width: float = 0.12,
        tee_mass: float = 1.0,
        world_size: float = 1.0,
        max_speed: float = 1.0,
        contact_k: float = 8000.0,
        contact_c: float = 100.0,
        contact_margin: float = 0.002,
        body_substeps: int = 16,
        linear_damping: float = 40.0,
        angular_damping: float = 40.0,
        pos_shaping_factor: float = 1.0,
        rot_shaping_factor: float = 0.5,
        agent_dist_shaping: float = 1.0,
        joint_shaping: float = 2.0,
        goal_reward: float = 5.0,
        goal_tolerance: float = 0.1,
        angle_tolerance: float = 0.25,
    ) -> None:
        self.n_agents = n_agents
        self.agent_radius = agent_radius
        self.tee_mass = tee_mass
        self.world_size = world_size
        self.max_speed = max_speed
        # contact_k/contact_c ARE the engine's WorldConfig.collision_k/collision_c —
        # the same spring-damper law, named for the agent<->T contact that dominates
        # here. contact_margin is NOT the engine's collision_margin (which is derived
        # from agent_radius in make_world): it is this scenario's own agent<->T
        # activation gap, used by the torch reference path below.
        self.contact_k = contact_k
        self.contact_c = contact_c
        self.contact_margin = contact_margin
        # Body substeps for the *grad path only*: on a taped step the engine leaves movable
        # bodies alone, so _refresh integrates the T in torch and needs its own substepping
        # to stay stable at this stiffness. The no-grad path uses Stepper.substeps instead.
        self.body_substeps = body_substeps
        # Depth at which the contact spring saturates (smoothly), bounding the impulse a
        # deep overlap can inject. Derived in make_world, where sub_dt is known: it has to
        # sit *above* the depth a velocity-controlled agent settles at, or the contact can
        # no longer hold the agent out and it walks straight through the T.
        self.max_overlap = 0.0
        # Viscous drag standing in for table friction, and the knob that sets how
        # responsive the T is: under a sustained push it settles at
        # ``sum(f) / (mass * linear_damping)``, so N agents pressing together can drive it
        # at N times the speed one can. At damping 10 four robots drove it to ~4 m/s off
        # 1.25 mm of compression, which reads as the T moving without being touched (and
        # then outrunning its pushers). At 40 it tops out near the robots' own 1 m/s and
        # needs a visible 5 mm of compression to do it.
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.pos_shaping_factor = pos_shaping_factor
        self.rot_shaping_factor = rot_shaping_factor
        self.agent_dist_shaping = agent_dist_shaping
        # Weight of the joint pose "crater": a potential-based Gaussian bump around
        # the full (position AND orientation) goal. The linear pos/rot terms reward
        # each error independently — this is the only term that rewards closing the
        # last stretch of both at once, which is what the solved condition needs.
        self.joint_shaping = joint_shaping
        self.goal_reward = goal_reward
        self.goal_tolerance = goal_tolerance
        self.angle_tolerance = angle_tolerance

        # --- T geometry, in the body frame, origin at the area centroid ---------
        # Crossbar sits on top at local y=0; the stem hangs below it. Shifting both
        # by the centroid puts the rotation origin and the inertia origin together.
        a_bar = bar_len * bar_width
        a_stem = stem_width * stem_len
        y_stem = -(bar_width + stem_len) / 2.0
        cy = a_stem * y_stem / (a_bar + a_stem)
        self.box_half = ((bar_len / 2.0, bar_width / 2.0), (stem_width / 2.0, stem_len / 2.0))
        self.box_off = ((0.0, -cy), (0.0, y_stem - cy))
        self.n_boxes = len(self.box_half)

        # Mass splits by area; inertia about the centroid by the parallel-axis theorem.
        inertia = 0.0
        for (hx, hy), (_, oy) in zip(self.box_half, self.box_off, strict=True):
            m_i = tee_mass * (4.0 * hx * hy) / (a_bar + a_stem)
            inertia += m_i * ((2.0 * hx) ** 2 + (2.0 * hy) ** 2) / 12.0 + m_i * oy * oy
        self.tee_inertia = inertia
        # Farthest corner from the centroid: the clamp radius that keeps the whole
        # T inside the world bounds.
        self.tee_radius = max(
            math.hypot(hx, oy + sy * hy)
            for (hx, hy), (_, oy) in zip(self.box_half, self.box_off, strict=True)
            for sy in (-1.0, 1.0)
        )
        # Staging point for the approach shaping, push_point_offset past the T centre
        # directly away from the goal. 0 = the T centre itself: plain approach shaping.
        # (>0 turned out noisier in practice — the point moves with the T.)
        self.push_point_offset = 0.0
        # Curriculum hooks: when set, the goal pose is sampled within this radius /
        # angle of the T's spawn pose instead of uniformly. Trainers anneal these.
        self.goal_spawn_radius: float | None = None
        self.goal_spawn_angle: float | None = None

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
        # A velocity-mode agent has no contact memory: each substep its velocity is
        # overwritten by the command plus f*sub_dt/m, so holding it out of the body needs
        # f >= m*max_speed/sub_dt, i.e. an equilibrium depth of
        # max_speed*m/(k*sub_dt). Saturating the spring *below* that lets agents walk
        # through the T; twice it bounds a deep sweep's impulse while leaving the holding
        # force intact (2x headroom, and legitimate pushing sits ~20x shallower).
        sub_dt = dt / max(1, substeps)
        equilibrium_depth = self.max_speed / (self.contact_k * sub_dt)
        self.max_overlap = 2.0 * equilibrium_depth
        # No neighbor_radius override: obstacles are scanned linearly by
        # _static_forces, not looked up in the neighbor grid, so the T's size does
        # not have to inflate the agent<->agent reach (the default 2*r + margin).
        cfg = WorldConfig(
            collisions=True,
            collision_k=self.contact_k,
            collision_c=self.contact_c,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            max_neighbors=min(32, max(4, self.n_agents)),
            # The T is a movable obstacle now, so its drag is engine config.
            obstacle_linear_damping=self.linear_damping,
            obstacle_angular_damping=self.angular_damping,
            contact_max_overlap=self.max_overlap,
        ).override_with(world_config)
        self.dt = dt
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        tt = {"device": device, "dtype": dtype}
        # Agent masses, for the oracle's contact denominator. The implicit damping solve
        # belongs to whoever the impulse is applied to, and on the engine side that is the
        # *agent*: :func:`swarp.core.bodies._reaction` reuses the agent-side force verbatim
        # and negates it, so its ``damp_denom`` is built from the agent's mass. ``[1, A, 1]``
        # so it broadcasts over the ``[E, A, B]`` contact grid.
        self._agent_mass = torch.tensor([c.mass for c in cfgs], **tt).view(1, -1, 1)
        # Teammate index for the torch observation: row ``a`` is every agent but ``a``, in
        # ascending order. Fixed by ``n_agents``, so it is built here rather than rebuilt
        # (four ops and two allocations) on every ``observations()`` call.
        na = self.n_agents
        idx = torch.arange(na, device=device)
        others = idx.unsqueeze(0).expand(na, -1)[idx.unsqueeze(1) != idx.unsqueeze(0)]
        self._others = others.view(na, na - 1)  # [A, A-1]
        # Env-independent obstacle attributes (both boxes, every env).
        self._obs_shape = torch.full((self.n_boxes,), int(ObstacleShape.BOX), device=device,
                                     dtype=torch.int32)
        self._obs_radius = torch.zeros(self.n_boxes, **tt)  # a box ignores radius
        self._obs_half = torch.tensor(self.box_half, **tt)
        self._box_off_t = torch.tensor(self.box_off, **tt)  # [n_boxes, 2]
        # The T is ONE movable compound body: both boxes share body 0 (the root) and sit
        # at their local offsets, so the engine integrates a single pose for the pair.
        # Mass/inertia are read from the root and are about the body origin (the centroid).
        self._obs_kind = torch.full(
            (self.n_boxes,), int(ObstacleKind.MOVABLE), device=device, dtype=torch.int32
        )
        self._obs_body = torch.zeros(self.n_boxes, device=device, dtype=torch.int32)
        self._obs_mass = torch.full((self.n_boxes,), self.tee_mass, **tt)
        self._obs_inertia = torch.full((self.n_boxes,), self.tee_inertia, **tt)
        # T pose and target pose: allocated here (n_envs is known) so the fused spec can
        # adopt them, and written in place by every reset so their handles stay valid.
        self.tee_pos = torch.zeros(n_envs, 2, **tt)
        self.tee_vel = torch.zeros(n_envs, 2, **tt)
        self.tee_theta = torch.zeros(n_envs, **tt)  # [n_envs]
        self.tee_ang_vel = torch.zeros(n_envs, **tt)
        self.goal_pos = torch.zeros(n_envs, 2, **tt)
        self.goal_theta = torch.zeros(n_envs, **tt)
        # The per-shape world poses handed to the engine. These are *derived* from
        # ``tee_*`` (unlike transport's package poses, which the obstacle spec can alias
        # directly), so they get buffers of their own that ``_install_obstacles`` writes
        # into. Persistent, because the ``Obstacles`` spec built just below aliases them:
        # that is what lets every later install re-use one already-resolved spec.
        self._box_centers = torch.zeros(n_envs, self.n_boxes, 2, **tt)
        self._box_angles = torch.zeros(n_envs, self.n_boxes, **tt)
        self._box_vel = torch.zeros(n_envs, self.n_boxes, 2, **tt)
        self._box_ang_vel = torch.zeros(n_envs, self.n_boxes, **tt)
        # Built and resolved once. ``resolve`` is a no-op for a field that is already on
        # the right device/dtype and contiguous, so the resolved spec's tensors *share
        # storage* with the four buffers above — writing them is writing the spec.
        self._obstacles = Obstacles(
            self._box_centers,
            self._obs_radius,
            shape=self._obs_shape,
            angle=self._box_angles,  # per-env: the T rotates independently in each env
            half_extents=self._obs_half,
            vel=self._box_vel,
            ang_vel=self._box_ang_vel,
            kind=self._obs_kind,
            mass=self._obs_mass,
            inertia=self._obs_inertia,
            body=self._obs_body,
            body_offset=self._box_off_t,
        ).resolve(self.world.device, self.world.dtype)
        # Shaping baselines: left None so the first refresh seeds them from the fresh pose
        # (shaping 0) rather than from zeros; the spec adopts them with alloc="if_none".
        self._prev_dist: torch.Tensor | None = None
        self._prev_ang: torch.Tensor | None = None
        self._prev_adist: torch.Tensor | None = None  # [n_envs, n_agents]
        self._cache: dict[str, torch.Tensor] | None = None
        return self.world

    # Push-T's two paths do not compute the same thing, by design: the fused path lets the
    # engine integrate the T at ``Stepper.substeps`` inside the substep loop, while the
    # torch reference integrates it itself at ``body_substeps`` (16) because the engine
    # skips movable bodies on a taped step. The trajectories therefore diverge by far more
    # than reassociation, and the stiff contact (``contact_k`` 8000) means an agent near the
    # contact threshold can be judged in contact by one path and out by the other — a
    # discrete flip worth ~1e-3 of shaping, which the reward (a *difference* of successive
    # pose errors) then amplifies. Two to three orders looser than the default, and the
    # single place that says so.
    parity_rtol: float = 1e-2
    parity_atol: float = 5e-3

    @property
    def obs_dim(self) -> int:
        # 12 own/task features + the other agents' relative positions (coordination).
        return 12 + 2 * (self.n_agents - 1)

    # ------------------------------------------------------------------ reset

    def reset_world(
        self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False
    ) -> None:
        w = self.world
        n = w.n_envs
        lim = self.world_size - 2.0 * self.agent_radius
        tlim = self.world_size - self.tee_radius
        spawn = w.sample_uniform((n, self.n_agents, 2), -lim, lim)
        tee = w.sample_uniform((n, 2), -tlim, tlim)
        # Push any agent that landed inside the T's bounding disk out to its rim. With a
        # stiff contact_k a spawn overlap is a violent ejection (k * depth * sub_dt is
        # metres per second), so the reset must not start interpenetrating.
        clear = self.tee_radius + 2.0 * self.agent_radius
        d = spawn - tee.unsqueeze(1)  # [n, A, 2]
        dn = d.norm(dim=-1, keepdim=True)
        # Straight up (arbitrary but deterministic) for an agent exactly on the centre.
        unit = torch.where(dn > 1.0e-9, d / dn.clamp(min=1.0e-9), torch.tensor(
            [0.0, 1.0], device=w.device, dtype=w.dtype).expand_as(d))
        spawn = torch.where(dn < clear, tee.unsqueeze(1) + unit * clear, spawn).clamp(-lim, lim)
        theta = w.sample_uniform((n,), -math.pi, math.pi)
        if self.goal_spawn_radius is None:
            goal = w.sample_uniform((n, 2), -tlim, tlim)
        else:
            # Uniform in a disk of goal_spawn_radius around the T spawn (curriculum).
            gdir = w.sample_uniform((n,), -math.pi, math.pi)
            grad = self.goal_spawn_radius * w.sample_uniform((n,), 0.0, 1.0).sqrt()
            goal = (
                tee + torch.stack([grad * gdir.cos(), grad * gdir.sin()], dim=-1)
            ).clamp(-tlim, tlim)
        if self.goal_spawn_angle is None:
            goal_th = w.sample_uniform((n,), -math.pi, math.pi)
        else:
            goal_th = theta + w.sample_uniform(
                (n,), -self.goal_spawn_angle, self.goal_spawn_angle
            )
        zeros_2 = torch.zeros_like(self.tee_pos)
        zeros_1 = torch.zeros_like(self.tee_theta)

        # In-place body updates (copy_) so the fused path's cached wp handles and the
        # whole-step graph stay valid across resets; the grad path's _refresh still
        # reassigns them (fresh tensors for the tape), which the framework's handle
        # resync catches on the next no-grad step.
        w.write_state(env_mask, pos=spawn, vel=0.0)
        if env_mask is None:
            self.tee_pos.copy_(tee)
            self.tee_vel.zero_()
            self.tee_theta.copy_(theta)
            self.tee_ang_vel.zero_()
            self.goal_pos.copy_(goal)
            self.goal_theta.copy_(goal_th)
        else:
            m2 = env_mask.view(-1, 1)
            self.tee_pos.copy_(torch.where(m2, tee, self.tee_pos))
            self.tee_vel.copy_(torch.where(m2, zeros_2, self.tee_vel))
            self.tee_theta.copy_(torch.where(env_mask, theta, self.tee_theta))
            self.tee_ang_vel.copy_(torch.where(env_mask, zeros_1, self.tee_ang_vel))
            self.goal_pos.copy_(torch.where(m2, goal, self.goal_pos))
            self.goal_theta.copy_(torch.where(env_mask, goal_th, self.goal_theta))

        self._install_obstacles()
        self.finish_reset(env_mask, obs_only=obs_only)

    def _box_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The two box centres ``[n_envs, n_boxes, 2]`` and angles ``[n_envs, n_boxes]``
        for the current T pose, written **into** the persistent pose buffers.

        Torch (the fused path does this inside the body kernel). Returns the buffers so the
        caller can go on using them as plain tensors.
        """
        th = self.tee_theta.detach()
        ca, sa = torch.cos(th), torch.sin(th)  # [E]
        ox, oy = self._box_off_t[:, 0], self._box_off_t[:, 1]  # [B]
        # rotate each local offset into world: R(theta) @ off
        wx = ca.unsqueeze(1) * ox - sa.unsqueeze(1) * oy  # [E, B]
        wy = sa.unsqueeze(1) * ox + ca.unsqueeze(1) * oy
        self._box_centers.copy_(self.tee_pos.detach().unsqueeze(1) + torch.stack([wx, wy], dim=-1))
        self._box_angles.copy_(th.unsqueeze(1).expand(-1, self.n_boxes))
        return self._box_centers, self._box_angles

    def _install_obstacles(self) -> None:
        """Seed the engine's movable body from ``tee_*``.

        Called on reset, and after a *grad* step (where the torch reference integrates the
        body instead of the engine). The engine derives the body's own pose from the root
        shape and takes its velocity from the root, so writing the shapes' world poses is
        enough to hand over the whole state.

        The spec is built once, in ``make_world``, and **re-installed** rather than
        rebuilt: it aliases the four ``_box_*`` buffers this method writes, and both
        ``Obstacles.resolve`` and ``Obstacles.any_movable`` memoize, so the re-install
        neither allocates a 12-field spec nor pays ``any_movable``'s reduction plus
        ``.item()`` device->host sync. Building a fresh spec per call — as this used to —
        put that sync on every reset, i.e. on every step under ``auto_reset=True``.

        Unlike transport's, this install is *never* inside a graph capture: the T is a
        movable body, so ``Stepper._install`` always derives the body group in torch. It
        therefore has to be sync-free, not capture-safe. ``_box_*`` are framework-side
        buffers that nothing outside this method reassigns — the grad path reassigns
        ``tee_*``, which these are *derived from* rather than aliases of — so the retained
        spec can never go stale and needs no pointer re-check.
        """
        centers, _ = self._box_poses()
        # Each box's own velocity: the body's linear velocity plus omega x r, where r is
        # the lever from the T centroid to that box centre. Contact damping needs it —
        # without it the agents are damped against their absolute velocity and the
        # moving T applies drag unrelated to the contact.
        r = centers - self.tee_pos.detach().unsqueeze(1)  # [E, B, 2]
        om = self.tee_ang_vel.detach().unsqueeze(1)  # [E, 1]
        self._box_vel.copy_(
            self.tee_vel.detach().unsqueeze(1)
            + torch.stack([-om * r[..., 1], om * r[..., 0]], dim=-1)
        )
        self._box_ang_vel.copy_(om.expand(-1, self.n_boxes))
        self.world.set_obstacles(self._obstacles)

    def _sync_from_engine(self) -> None:
        """Copy the engine's body state into ``tee_*`` (the obs/reward inputs).

        The engine owns the T on no-grad steps: it integrates the body inside the substep
        loop, so the pose the agents collided against is at most one substep old rather
        than a step behind. These are four ``[n_envs]``-sized strided copies of the root
        body's slot; the fused obs/reward kernels keep reading ``tee_*`` unchanged.
        """
        b_pos, b_angle, b_vel, b_ang_vel = self.world.obstacle_state_views(body=True)
        self.tee_pos.copy_(b_pos[:, 0])
        self.tee_theta.copy_(b_angle[:, 0])
        self.tee_vel.copy_(b_vel[:, 0])
        self.tee_ang_vel.copy_(b_ang_vel[:, 0])

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        """The T after a physics step, on the non-fused path — still two cases.

        A *taped* step: the engine skips movable bodies (it advances body state with
        ``record_tape=False``), so the T is integrated here in differentiable torch and
        handed back — that is what makes BPTT through the body work. An untaped step: the
        engine did integrate it inside the substep loop, so only read it back.

        Note the two run genuinely different physics — ``_refresh`` substeps the T
        ``body_substeps`` times, the engine ``Stepper.substeps`` times — and are not meant
        to be bit-comparable. :attr:`parity_rtol` / :attr:`parity_atol` say so numerically.
        """
        if self._grad_step():
            self._refresh(integrate=True)
        else:
            self._sync_from_engine()
            self._refresh(integrate=False)

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        # Drop the shaping baselines so _refresh re-seeds them from the fresh pose
        # (shaping 0) rather than differencing against the previous episode's.
        self._prev_dist = None
        self._prev_ang = None
        self._prev_adist = None
        self._refresh(reset_mask=env_mask, integrate=False)

    def _grad_step(self) -> bool:
        """True when the step just taken was taped, so the engine left the body alone."""
        s = self.world.state
        return torch.is_grad_enabled() and any(
            t is not None and t.requires_grad for t in (s.pos, s.vel)
        )

    # --------------------------------------------------------- fused fast path

    _BODY_ATTRS = ("tee_pos", "tee_vel", "tee_theta", "tee_ang_vel", "goal_pos", "goal_theta")

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        """The largest spec: five outputs, the reset mask, the body/goal pose, and three
        shaping carries.

        Every ``tee_*``/``goal_*``/``_prev_*`` entry is watched: the grad path integrates
        the T in torch and reassigns them all, and the fused path caches Warp handles over
        them, so a graph captured against the old pointers has to be thrown away. That used
        to be nine hand-written pointer compares across two dicts and three scalars.
        """
        ne, na = n_envs, self.n_agents
        body = {"alloc": "never", "carry": True, "watch": True}
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("reward", (ne, na)),
            Buf("dist", (ne,)),
            Buf("ang", (ne,)),
            Buf("done", (ne,), "uint8", bool_view=True),
            Buf("resetmask", (ne,), "uint8", reset_mask=True),
            Buf("tee_pos", (ne, 2), "vec2", attr="tee_pos", **body),
            Buf("tee_vel", (ne, 2), "vec2", attr="tee_vel", **body),
            Buf("tee_theta", (ne,), attr="tee_theta", **body),
            Buf("tee_ang_vel", (ne,), attr="tee_ang_vel", **body),
            # The goal pose is written on reset but never advanced by the hook.
            Buf("goal_pos", (ne, 2), "vec2", attr="goal_pos", alloc="never", watch=True),
            Buf("goal_theta", (ne,), attr="goal_theta", alloc="never", watch=True),
            Buf("prev_dist", (ne,), attr="_prev_dist", alloc="if_none", carry=True, watch=True),
            Buf("prev_ang", (ne,), attr="_prev_ang", alloc="if_none", carry=True, watch=True),
            Buf(
                "prev_adist", (ne, na), attr="_prev_adist", alloc="if_none", carry=True, watch=True
            ),
        )

    def engine_carries(self) -> list[torch.Tensor]:
        """The engine's obstacle *and* body arrays.

        ``swarp.core.bodies`` advances the T in place inside the substep loop and rewrites
        its box poses there, so graph warm-up — which runs the physics once to compile
        kernels — would otherwise leave the body one step ahead of the eager path.
        """
        w = self.world
        return [*w.obstacle_state_views(), *w.obstacle_state_views(body=True)]

    def launch_fused(self, pass_: FusedPass) -> None:
        """Lift the engine's body state into ``tee_*``, then obs and reward.

        There is no body *launch*: :mod:`swarp.core.bodies` already advanced the T inside
        the substep loop and re-installed its box poses there. Copying the root body's
        state into the cached ``tee_*`` arrays keeps the obs/reward kernels (which read
        ``tee_*``) unchanged and stays capture-safe — four fixed-size device-to-device
        copies, no allocation.

        A reset skips that copy (the T was just placed, so ``tee_*`` is authoritative and
        the engine's copy is what got seeded *from* it) and passes ``advance_prev=0`` so the
        reward kernel rebases the shaping baselines for the reset envs.
        """
        state = self.world.state_wp()  # one wrap for every launch in this pass
        if pass_.is_step:
            self._launch_body_sync()
        self._launch_obs(state)
        self._launch_reward(state, advance_prev=pass_.advance_prev, full_pass=pass_.full_pass)

    def _launch_body_sync(self) -> None:
        """Lift the engine's root-body state into the cached ``tee_*`` Warp arrays."""
        w = self.world
        st = w.stepper
        wp.launch(
            body_state_gather_kernel,
            dim=w.n_envs,
            inputs=[st.body_pos, st.body_angle, st.body_vel, st.body_ang_vel, wp.int32(0)],
            outputs=[
                self._wp["tee_pos"],
                self._wp["tee_theta"],
                self._wp["tee_vel"],
                self._wp["tee_ang_vel"],
            ],
            device=w.device,
            record_tape=False,
        )

    def _launch_obs(self, st) -> None:
        w = self.world
        wp.launch(
            pusht_obs_kernel,
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                self._wp["tee_pos"],
                self._wp["tee_theta"],
                self._wp["goal_pos"],
                self._wp["goal_theta"],
            ],
            outputs=[self._wp["obs"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_reward(self, st, advance_prev: int, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        wp.launch(
            pusht_reward_kernel,
            dim=w.n_envs,
            inputs=[
                st.pos,
                self._wp["tee_pos"],
                self._wp["tee_theta"],
                self._wp["goal_pos"],
                self._wp["goal_theta"],
                self._wp["resetmask"],
                wp.int32(self.n_agents),
                scalar(self.pos_shaping_factor),
                scalar(self.rot_shaping_factor),
                scalar(self.agent_dist_shaping),
                scalar(self.push_point_offset),
                scalar(self.joint_shaping),
                scalar(self.goal_tolerance),
                scalar(self.angle_tolerance),
                scalar(self.goal_reward),
                wp.int32(advance_prev),
                wp.int32(full_pass),
            ],
            outputs=[
                self._wp["prev_dist"],
                self._wp["prev_ang"],
                self._wp["prev_adist"],
                self._wp["reward"],
                self._wp["done"],
                self._wp["dist"],
                self._wp["ang"],
            ],
            device=w.device,
            record_tape=False,
        )

    def _box_contact(self, pos: torch.Tensor, vel: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Per (env, agent, box) oriented-box SDF contact, mirroring
        :func:`swarp.core.collisions.box_force` (exterior clamp + interior nearest
        face). Returns the force on the T ``[E, A, B, 2]`` and its torque ``[E, A, B]``
        about the body centroid."""
        th = self.tee_theta  # [E]
        ca, sa = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]  # [E,1,1]
        ox, oy = self._box_off_t[:, 0], self._box_off_t[:, 1]  # [B]
        # box centres in world = tee centre + R(theta) @ local offset
        owx = ca * ox - sa * oy  # [E, 1, B]
        owy = sa * ox + ca * oy
        cwx = self.tee_pos[:, None, None, 0] + owx  # [E, 1, B]
        cwy = self.tee_pos[:, None, None, 1] + owy
        hx, hy = self._obs_half[:, 0], self._obs_half[:, 1]  # [B]

        dx = pos[..., 0].unsqueeze(-1) - cwx  # [E, A, B]
        dy = pos[..., 1].unsqueeze(-1) - cwy
        lx = ca * dx + sa * dy  # into the box frame (R^T d)
        ly = -sa * dx + ca * dy
        cx, cy = lx.clamp(-hx, hx), ly.clamp(-hy, hy)
        ex, ey = lx - cx, ly - cy
        out_d2 = ex * ex + ey * ey
        exterior = out_d2 > 1.0e-10
        s_out = out_d2.clamp(min=1.0e-10).sqrt()
        nlx_out, nly_out = ex / s_out, ey / s_out
        # interior: the nearer face fixes the (negative) signed distance and normal
        gx, gy = lx.abs() - hx, ly.abs() - hy
        use_x = gx > gy
        s_in = torch.where(use_x, gx, gy)
        nlx_in = torch.where(use_x, torch.sign(lx) + (lx == 0).to(lx.dtype), torch.zeros_like(lx))
        nly_in = torch.where(use_x, torch.zeros_like(ly), torch.sign(ly) + (ly == 0).to(ly.dtype))
        s = torch.where(exterior, s_out, s_in)
        nlx = torch.where(exterior, nlx_out, nlx_in)
        nly = torch.where(exterior, nly_out, nly_in)

        overlap = (self.agent_radius + self.contact_margin) - s
        active = (overlap > 0).to(pos.dtype)
        # Smooth depth saturation, mirroring :func:`swarp.core.collisions._normal_coeff`:
        # bounds the impulse a frozen-pose sweep can inject without a gradient-killing
        # hard clamp.
        mo = self.max_overlap
        overlap = mo * torch.tanh(overlap.clamp(min=0.0) / mo)
        # closest surface point (box frame) -> lever arm about the tee centroid
        spx, spy = lx - s * nlx, ly - s * nly
        rx = owx + (ca * spx - sa * spy)
        ry = owy + (sa * spx + ca * spy)
        # m = -n: unit normal pointing from the agent into the T
        mx = -(ca * nlx - sa * nly)
        my = -(sa * nlx + ca * nly)
        # relative velocity of the T's contact point w.r.t. the agent
        om = self.tee_ang_vel[:, None, None]
        rvx = self.tee_vel[:, None, None, 0] - om * ry - vel[..., 0].unsqueeze(-1)
        rvy = self.tee_vel[:, None, None, 1] + om * rx - vel[..., 1].unsqueeze(-1)
        vn = rvx * mx + rvy * my
        # Linearly-implicit damping + repulsive clamp; see
        # :func:`swarp.core.collisions._normal_coeff` for the derivation. The mass in the
        # denominator is the **agent's**, not the T's: the engine solves the implicit step
        # for the body the impulse acts on, and reuses that one number as the reaction on
        # the T (:func:`swarp.core.bodies._reaction`). ``sub_dt`` *does* stay this loop's
        # own — the implicit term belongs to the scheme applying the impulse, and here that
        # scheme is the oracle's ``body_substeps`` loop rather than the engine's substep.
        sub_dt = self.dt / self.body_substeps
        coeff = (self.contact_k * overlap - self.contact_c * vn) / (
            1.0 + self.contact_c * sub_dt / self._agent_mass
        )
        coeff = coeff.clamp(min=0.0) * active
        fx, fy = coeff * mx, coeff * my
        return torch.stack([fx, fy], dim=-1), rx * fy - ry * fx

    def _refresh(self, reset_mask: torch.Tensor | None = None, integrate: bool = True) -> None:
        w = self.world
        pos, vel = w.state.pos, w.state.vel  # [n_envs, n_agents, 2]

        if integrate:
            # Substepped exactly like :func:`swarp.core.bodies.obstacle_dynamics_kernel`
            # (whose per-shape reaction is :func:`swarp.core.bodies._reaction`): the contact
            # force is recomputed from the frozen agent state each sub-step, which is what
            # keeps a stiff contact_k stable. Gradients flow through every sub-step.
            sub_dt = self.dt / self.body_substeps
            b = self.world_size - self.tee_radius
            for _ in range(self.body_substeps):
                force, torque = self._box_contact(pos, vel)  # [E,A,B,2], [E,A,B]
                f_total = force.sum(dim=(1, 2))  # [E, 2] net force on the T
                tau = torque.sum(dim=(1, 2))  # [E]
                self.tee_vel = (self.tee_vel + f_total / self.tee_mass * sub_dt) * (
                    1.0 - self.linear_damping * sub_dt
                )
                self.tee_ang_vel = (self.tee_ang_vel + tau / self.tee_inertia * sub_dt) * (
                    1.0 - self.angular_damping * sub_dt
                )
                self.tee_pos = (self.tee_pos + self.tee_vel * sub_dt).clamp(-b, b)
                self.tee_theta = self.tee_theta + self.tee_ang_vel * sub_dt
            self._install_obstacles()  # for the next step

        dist_to_goal = (self.tee_pos - self.goal_pos).norm(dim=-1)  # [E]
        raw = self.tee_theta - self.goal_theta
        angle_error = torch.atan2(torch.sin(raw), torch.cos(raw)).abs()  # [E]
        tee_rel = self.tee_pos.unsqueeze(1) - pos  # [E, A, 2]
        # Pushing point: push_point_offset past the T centre, directly away from the
        # goal — being there (and pressing in) is what moves the T goal-ward.
        away = (self.tee_pos - self.goal_pos) / dist_to_goal.clamp(min=1e-6).unsqueeze(-1)
        push_pt = self.tee_pos + away * self.push_point_offset  # [E, 2]
        agent_dist = (push_pt.unsqueeze(1) - pos).norm(dim=-1)  # [E, A]
        if self._prev_dist is None:
            self._prev_dist = dist_to_goal.detach().clone()
            self._prev_ang = angle_error.detach().clone()
        if self._prev_adist is None:
            self._prev_adist = agent_dist.detach().clone()
        shaping = (self._prev_dist - dist_to_goal) * self.pos_shaping_factor + (
            self._prev_ang - angle_error
        ) * self.rot_shaping_factor

        def crater(dd: torch.Tensor, aa: torch.Tensor) -> torch.Tensor:
            # 1.5x the tolerances so the bump's gradient reaches past the goal box.
            sd, sa = 1.5 * self.goal_tolerance, 1.5 * self.angle_tolerance
            return torch.exp(-(dd / sd) ** 2 - (aa / sa) ** 2)

        shaping = shaping + self.joint_shaping * (
            crater(dist_to_goal, angle_error) - crater(self._prev_dist, self._prev_ang)
        )
        agent_shaping = (self._prev_adist - agent_dist) * self.agent_dist_shaping  # [E, A]
        if reset_mask is None:
            self._prev_dist = dist_to_goal.detach().clone()
            self._prev_ang = angle_error.detach().clone()
            self._prev_adist = agent_dist.detach().clone()
        else:
            m2 = reset_mask.view(-1, 1)
            shaping = torch.where(reset_mask, torch.zeros_like(shaping), shaping)
            agent_shaping = torch.where(m2, torch.zeros_like(agent_shaping), agent_shaping)
            self._prev_dist = torch.where(reset_mask, dist_to_goal.detach(), self._prev_dist)
            self._prev_ang = torch.where(reset_mask, angle_error.detach(), self._prev_ang)
            self._prev_adist = torch.where(m2, agent_dist.detach(), self._prev_adist)

        on_goal = (dist_to_goal < self.goal_tolerance) & (angle_error < self.angle_tolerance)
        self._cache = {
            "dist_to_goal": dist_to_goal,
            "angle_error": angle_error,
            "shaping": shaping,  # [E]
            "agent_shaping": agent_shaping,  # [E, A]
            "on_goal": on_goal,
            "tee_rel": tee_rel,
        }

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["obs"]
        w = self.world
        s = w.state
        na = w.n_agents
        tee_to_goal = (self.goal_pos - self.tee_pos).unsqueeze(1).expand(-1, na, -1)
        raw = (self.tee_theta - self.goal_theta).unsqueeze(1).expand(-1, na)
        th = self.tee_theta.unsqueeze(1).expand(-1, na)
        angles = torch.stack([th.cos(), th.sin(), raw.cos(), raw.sin()], dim=-1)
        # Teammates' positions relative to each agent, ascending index, self skipped.
        rel = (s.pos[:, self._others] - s.pos.unsqueeze(2)).flatten(2)  # [E, A, 2(A-1)]
        return torch.cat(
            [s.pos, s.vel, self._cache["tee_rel"], tee_to_goal, angles, rel], dim=-1
        )

    def global_reward(self) -> torch.Tensor:
        c = self._cache
        return c["shaping"] + self.goal_reward * c["on_goal"].to(self.world.dtype)

    def rewards(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["reward"]
        return self._cache["agent_shaping"] + self.global_reward().unsqueeze(1)

    def done(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["done_bool"]
        return self._cache["on_goal"]

    def render_extras(self, env_idx: int) -> dict[str, Any]:
        """The target pose of the T, as oriented boxes for the ``goal_pose`` overlay.

        Same two boxes the body is built from, placed at ``(goal_pos, goal_theta)``:
        the outline shows exactly where the T has to end up.
        """
        th = float(self.goal_theta[env_idx])
        gx, gy = (float(v) for v in self.goal_pos[env_idx])
        ca, sa = math.cos(th), math.sin(th)
        rows = []
        for (hx, hy), (ox, oy) in zip(self.box_half, self.box_off, strict=True):
            rows.append(
                (gx + ca * ox - sa * oy, gy + sa * ox + ca * oy, th, hx, hy)
            )
        return {"goal_pose": rows}

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {"tee_dist_to_goal": self.fb["dist"], "tee_angle_error": self.fb["ang"]}
        return {
            "tee_dist_to_goal": self._cache["dist_to_goal"],
            "tee_angle_error": self._cache["angle_error"],
        }
