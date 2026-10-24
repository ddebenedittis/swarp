"""Push-T: agents push a T-shaped rigid body to a target pose.

The T is a **movable compound obstacle** (:class:`~swarp.core.config.ObstacleKind`):
two oriented ``BOX`` shapes tagged with the same body id and placed by their offsets in
the body frame (any other box compound from :mod:`swarp.scenarios.shapes` via ``shape=``),
so :mod:`swarp.core.bodies` integrates one rigid pose for the set from
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

from swarp._overloads import concrete
from swarp.core.bodies import body_state_gather_kernel
from swarp.core.cached_launch import CachedLaunch, ptr_key
from swarp.core.config import ObstacleKind, Obstacles, ObstacleShape, WorldConfig
from swarp.core.state import VEC2
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, AgentShape, ControlMode, DynamicsModel, _coerce
from swarp.interop.autograd import torch_stream_scope
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario
from swarp.scenarios.pusht_kernels import (
    pusht_actuation_reset_kernel,
    pusht_arena_reset_kernel,
    pusht_friction_reset_kernel,
    pusht_heading_reset_kernel,
    pusht_latency_reset_kernel,
    pusht_obs_kernel,
    pusht_reset_kernel,
    pusht_reward_kernel,
)
from swarp.scenarios.shapes import SHAPES, CompoundShape, tee


def _as_range(
    v: float | tuple[float, float] | None, default: float, name: str
) -> tuple[float, float]:
    """``(low, high)`` of an actuation knob: a value is its own range, ``None`` the
    default's."""
    if v is None:
        return (default, default)
    lo, hi = (float(v), float(v)) if isinstance(v, int | float) else (float(v[0]), float(v[1]))
    if not 0.0 < lo <= hi:
        raise ValueError(f"{name} must be positive with low <= high, got {v!r}")
    return (lo, hi)


def arena_ranges(v: float | tuple[float, ...]) -> tuple[float, float, float, float]:
    """``(w_lo, w_hi, h_lo, h_hi)`` of an ``arena_size``: one extent for both axes, a
    ``(lo, hi)`` range drawn for each axis independently, or a range per axis."""
    if isinstance(v, int | float):
        r = (float(v),) * 4
    elif len(v) == 2:
        r = (float(v[0]), float(v[1])) * 2
    elif len(v) == 4:
        r = tuple(float(x) for x in v)
    else:
        raise ValueError(
            f"arena_size takes a value, (lo, hi) or (w_lo, w_hi, h_lo, h_hi), got {v!r}"
        )
    if not (0.0 < r[0] <= r[1] and 0.0 < r[2] <= r[3]):
        raise ValueError(f"arena_size must be positive with low <= high, got {v!r}")
    return r


class PushTScenario(FusedScenario):
    #: The robot models the constructor's ``model`` accepts (see ``supports_model``).
    supported_models: tuple[DynamicsModel, ...] = (
        DynamicsModel.HOLONOMIC,
        DynamicsModel.DIFF_DRIVE,
    )

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
        rot_away_penalty: float = 0.0,
        goal_reward: float = 5.0,
        goal_tolerance: float = 0.1,
        angle_tolerance: float = 0.25,
        shape: str | CompoundShape | None = None,
        model: DynamicsModel | str = DynamicsModel.HOLONOMIC,
        agent_shape: AgentShape | str = AgentShape.DISC,
        agent_half_extents: tuple[float, float] | None = None,
        max_ang_vel: float = 2.0 * math.pi,
        servo_accel: float | tuple[float, float] | None = None,
        servo_ang_accel: float | tuple[float, float] | None = None,
        cmd_gain_v: float | tuple[float, float] | None = None,
        cmd_gain_w: float | tuple[float, float] | None = None,
        agent_mass: float = 1.0,
        floor_friction: float | tuple[float, float] | None = None,
        wheel_contacts: bool = False,
        wheel_lateral_friction: float = 0.2,
        wheel_turn_torque: float = 0.0,
        cmd_latency: float | tuple[float, float] = 0.0,
        open_arena: bool = False,
        arena_stop_margin: float = 0.25,
        arena_size: float | tuple[float, ...] | None = None,
    ) -> None:
        """``model="diff_drive"`` makes the robots unicycles commanded by ``(v, omega)``,
        with turn rates up to ``max_ang_vel``; ``agent_shape="box"`` makes them rectangles
        of ``agent_half_extents`` (default ``(agent_radius, agent_radius / 2)``, along and
        across the heading). Either one makes the heading matter, so it is drawn at random
        on reset and observed as its cos and sin (:attr:`heading_obs`). A box's radius is
        its circumscribed one and replaces ``agent_radius`` wherever the scenario reads it
        as a bound (spawn clearance, the lidar's ``contact_dist``); the contact margin
        keeps the nominal ``agent_radius``.

        ``servo_accel``, ``servo_ang_accel``, ``cmd_gain_v`` and ``cmd_gain_w`` give
        ``diff_drive`` robots a drive (:attr:`ControlMode.SERVO
        <swarp.dynamics.base.ControlMode.SERVO>`): the commanded speed and turn rate are
        scaled by the gains and reached at most at the accelerations, instead of at once.
        Each is a value, or a ``(low, high)`` range drawn uniformly per env and robot on
        every reset. Setting any one of them turns the drive on; the others then default
        to no limit and unit gain. All ``None`` keeps the instantaneous velocity command.

        ``agent_mass`` is the robots' mass. A velocity-commanded robot has no inertia of
        its own, so the mass only sets how far a contact deflects it: in a push it gives
        up ``tee_mass * linear_damping * sub_dt / agent_mass`` of its speed to the T.

        ``floor_friction`` puts the T on a Coulomb floor with that ``mu`` (see
        ``WorldConfig.obstacle_friction``), or a ``(low, high)`` range drawn per env on
        every reset; it adds to the viscous ``linear_damping``/``angular_damping``, which a
        Coulomb T usually sets near 0.

        ``wheel_contacts`` gives ``diff_drive`` robots wheel-faithful contacts
        (``WorldConfig.wheel_contacts``): a contact turns a box robot through its torque
        past the drive's ``wheel_turn_torque``, pushes it sideways only past the wheels'
        grip ``wheel_lateral_friction``.

        ``cmd_latency`` (seconds, a value or a ``(low, high)`` range drawn per env and
        robot on every reset) delays every command: for that long into each step, rounded
        to whole substeps, a robot still applies the previous step's command
        (:meth:`Stepper.enable_command_delay
        <swarp.core.stepper.Stepper.enable_command_delay>`); a respawned robot's previous
        command is zero. A latency past ``dt`` is held at ``dt``.

        ``open_arena`` removes the arena's walls, as on a real floor: the T is no longer
        clamped inside ``world_size`` and the robots meet no wall. A ``diff_drive`` robot
        instead gets a safety bridge's arena stop (:meth:`Stepper.enable_arena_stop
        <swarp.core.stepper.Stepper.enable_arena_stop>`) ``arena_stop_margin`` past the
        arena's edge. It needs ``arena_size``.

        ``arena_size`` (metres, full extents; needs ``open_arena``) gives every env its own
        rectangular arena, drawn on every reset: one value for a fixed square, ``(lo, hi)``
        for a width and a height each drawn uniformly from it, or ``(w_lo, w_hi, h_lo,
        h_hi)`` for a range per axis. The spawns, the arena stop (at the env's half-extents
        plus ``arena_stop_margin``) and, in subclasses, everything else that reads the arena
        follow the env's :attr:`arena_half`; ``world_size`` must cover the largest. It is
        read on every reset, so a trainer can move it between resets. ``None`` keeps the
        one walled square arena of ``world_size``."""
        self.n_agents = n_agents
        self.model = _coerce(DynamicsModel, model, "model")
        if self.model not in self.supported_models:
            raise ValueError(f"Push-T supports holonomic and diff_drive robots, not {model!r}")
        shape_tag = _coerce(AgentShape, agent_shape, "agent_shape")
        if shape_tag == AgentShape.BOX:
            if agent_half_extents is None:
                agent_half_extents = (agent_radius, 0.5 * agent_radius)
            footprint = AgentConfig(shape=shape_tag, half_extents=agent_half_extents)
        else:
            footprint = AgentConfig(radius=agent_radius)
        self.agent_shape = footprint.shape
        self.agent_half_extents = footprint.half_extents
        self.max_ang_vel = max_ang_vel
        if agent_mass <= 0.0:
            raise ValueError(f"agent_mass must be positive, got {agent_mass}")
        self.agent_mass = agent_mass
        if wheel_contacts and self.model != DynamicsModel.DIFF_DRIVE:
            raise ValueError("wheel_contacts needs model='diff_drive'")
        self.wheel_contacts = bool(wheel_contacts)
        self.wheel_lateral_friction = wheel_lateral_friction
        self.wheel_turn_torque = wheel_turn_torque
        lat = (cmd_latency, cmd_latency) if isinstance(cmd_latency, int | float) else cmd_latency
        if not 0.0 <= lat[0] <= lat[1]:
            raise ValueError(f"cmd_latency must be >= 0 with low <= high, got {cmd_latency!r}")
        #: ``(low, high)`` command latency in seconds; turned into substeps in make_world.
        self.cmd_latency = (float(lat[0]), float(lat[1]))
        if open_arena and self.model != DynamicsModel.DIFF_DRIVE:
            raise ValueError("open_arena needs model='diff_drive' (its arena stop)")
        self.open_arena = bool(open_arena)
        self.arena_stop_margin = float(arena_stop_margin)
        #: Whether every env draws its own arena (``arena_size`` set at construction).
        self.per_env_arena = arena_size is not None
        #: The per-env arena's extents (see :func:`arena_ranges`), read on every reset.
        self.arena_size = arena_size
        if self.per_env_arena != self.open_arena:
            raise ValueError(
                "open_arena and arena_size go together: walls and the T's clamp are global, "
                "and an open floor's arena is per env (a value fixes one square)"
            )
        self.floor_friction = (
            None if floor_friction is None else _as_range(floor_friction, 0.0, "floor_friction")
        )
        #: Whether ``mu`` is drawn per env on reset.
        self.friction_dr = self.floor_friction is not None and (
            self.floor_friction[0] != self.floor_friction[1]
        )
        self._actuation = (servo_accel, servo_ang_accel, cmd_gain_v, cmd_gain_w)
        #: Whether the robots drive through ``ControlMode.SERVO``.
        self.servo = any(v is not None for v in self._actuation)
        if self.servo and self.model != DynamicsModel.DIFF_DRIVE:
            raise ValueError("servo_accel/servo_ang_accel/cmd_gain_* need model='diff_drive'")
        defaults = (math.inf, math.inf, 1.0, 1.0)
        #: ``(low, high)`` per actuation column: accel, ang accel, speed gain, rate gain.
        self._actuation_ranges = tuple(
            _as_range(v, d, name)
            for v, d, name in zip(
                self._actuation, defaults,
                ("servo_accel", "servo_ang_accel", "cmd_gain_v", "cmd_gain_w"), strict=True,
            )
        )
        #: Whether any actuation value is drawn per env on reset.
        self.actuation_dr = any(lo != hi for lo, hi in self._actuation_ranges)
        # The contact margin stays a fraction of the nominal size, not of a box's
        # circumscribed radius, which would widen the box contact's reach.
        self._margin_radius = agent_radius
        self.agent_radius = footprint.radius
        #: Whether a robot's heading matters (a unicycle steers by it, a box collides by
        #: it): then it is randomised on reset and observed.
        self.heading_obs = (
            self.model != DynamicsModel.HOLONOMIC or self.agent_shape == AgentShape.BOX
        )
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
        # Extra cost per radian the heading error *grows*. The wrapped error refunds a
        # turn away from the goal once it passes pi, so without this an extra full turn
        # (or going the long way round) costs nothing. 0 keeps the default reward.
        self.rot_away_penalty = rot_away_penalty
        self.goal_reward = goal_reward
        self.goal_tolerance = goal_tolerance
        self.angle_tolerance = angle_tolerance

        # --- body geometry, in the body frame, origin at the area centroid -----
        # See swarp.scenarios.shapes: mass splits by area, inertia about the centroid by
        # the parallel-axis theorem, radius to the farthest corner (the clamp radius that
        # keeps the whole body inside the world bounds). The default is the T, crossbar
        # first, sized by ``bar_*``/``stem_*`` whether ``shape`` is ``None`` or ``"tee"``
        # (pusht_lidar's default, so its T can be resized too). ``shape`` swaps in any
        # other compound body: another ``SHAPES`` name (built at its defaults with
        # ``tee_mass``) or a ``CompoundShape``, which carries its own mass; either way the
        # ``bar_*``/``stem_*`` sizes are ignored.
        if shape is None or shape == "tee":
            shape = tee(bar_len, bar_width, stem_len, stem_width, mass=tee_mass)
        elif isinstance(shape, str):
            if shape not in SHAPES:
                raise ValueError(f"unknown shape {shape!r}; choose from {sorted(SHAPES)}")
            shape = SHAPES[shape](mass=tee_mass)
        self.shape = shape
        self.tee_mass = shape.mass
        self.box_half = shape.half
        self.box_off = shape.off
        self.n_boxes = shape.n_boxes
        self.tee_inertia = shape.inertia
        self.tee_radius = shape.radius
        if self.per_env_arena:
            w_lo, w_hi, h_lo, h_hi = arena_ranges(arena_size)
            if 0.5 * max(w_hi, h_hi) > world_size + 1e-9:
                raise ValueError(
                    f"world_size {world_size} must cover the largest arena half-extent "
                    f"{0.5 * max(w_hi, h_hi)}"
                )
            inset = max(self.tee_radius, 2.0 * self.agent_radius)
            if 0.5 * min(w_lo, h_lo) <= inset:
                raise ValueError(
                    f"the smallest arena half-extent {0.5 * min(w_lo, h_lo)} leaves no room "
                    f"for the T (radius {self.tee_radius}) and the robots"
                )
        # Staging point for the approach shaping, push_point_offset past the T centre
        # directly away from the goal. 0 = the T centre itself: plain approach shaping.
        # (>0 turned out noisier in practice — the point moves with the T.)
        self.push_point_offset = 0.0
        # Curriculum hooks: when set, the goal pose is sampled within this radius /
        # angle of the T's spawn pose instead of uniformly. Trainers anneal these.
        self.goal_spawn_radius: float | None = None
        self.goal_spawn_angle: float | None = None

    def _agent_configs(self) -> list[AgentConfig]:
        """One :class:`AgentConfig` per agent: velocity-controlled pushers of ``model``
        (holonomic, or a unicycle commanded by ``(v, omega)``) and ``agent_shape``.

        Split out, as in :meth:`NavigationScenario._agent_configs
        <swarp.scenarios.navigation.NavigationScenario._agent_configs>`, so a subclass can
        vary the fleet without reimplementing ``make_world``. The contact derivation in
        ``make_world`` assumes velocity control at ``self.max_speed``.
        """
        if self.servo:
            # The ranges' low ends; an actuation_dr world redraws them on every reset.
            (acc, _), (ang_acc, _), (gv, _), (gw, _) = self._actuation_ranges
            drive = {
                "ctrl_mode": ControlMode.SERVO, "max_accel": acc, "max_ang_accel": ang_acc,
                "cmd_gain_v": gv, "cmd_gain_w": gw,
            }
        else:
            drive = {"ctrl_mode": ControlMode.VELOCITY, "max_accel": 2.0 * self.max_speed}
        return [
            AgentConfig(
                model=self.model,
                radius=None if self.agent_shape == AgentShape.BOX else self.agent_radius,
                mass=self.agent_mass,
                max_speed=self.max_speed,
                max_ang_vel=self.max_ang_vel,
                shape=self.agent_shape,
                half_extents=self.agent_half_extents,
                **drive,
            )
            for _ in range(self.n_agents)
        ]

    def make_world(self, n_envs, device, dt, substeps, dtype, world_config=None) -> World:
        cfgs = self._agent_configs()
        margin = 0.5 * self._margin_radius
        # A velocity-mode agent has no contact memory: each substep its velocity is
        # overwritten by the command plus f*sub_dt/m, so holding it out of the body needs
        # f >= m*max_speed/sub_dt, i.e. an equilibrium depth of
        # max_speed*m/(k*sub_dt). Saturating the spring *below* that lets agents walk
        # through the T; twice it bounds a deep sweep's impulse while leaving the holding
        # force intact (2x headroom, and legitimate pushing sits ~20x shallower).
        sub_dt = dt / max(1, substeps)
        equilibrium_depth = self.max_speed * self.agent_mass / (self.contact_k * sub_dt)
        self.max_overlap = 2.0 * equilibrium_depth
        # No neighbor_radius override: obstacles are scanned linearly by
        # _static_forces, not looked up in the neighbor grid, so the T's size does
        # not have to inflate the agent<->agent reach (the default 2*r + margin).
        cfg = WorldConfig(
            collisions=True,
            collision_k=self.contact_k,
            collision_c=self.contact_c,
            collision_margin=margin,
            bounds=None
            if self.open_arena
            else (-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            max_neighbors=min(32, max(4, self.n_agents)),
            # The T is a movable obstacle now, so its drag is engine config.
            obstacle_linear_damping=self.linear_damping,
            obstacle_angular_damping=self.angular_damping,
            obstacle_friction=0.0 if self.floor_friction is None else self.floor_friction[0],
            wheel_contacts=self.wheel_contacts,
            wheel_lateral_friction=self.wheel_lateral_friction,
            wheel_turn_torque=self.wheel_turn_torque,
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
        # Env-independent obstacle attributes (every box, every env).
        self._obs_shape = torch.full((self.n_boxes,), int(ObstacleShape.BOX), device=device,
                                     dtype=torch.int32)
        self._obs_radius = torch.zeros(self.n_boxes, **tt)  # a box ignores radius
        self._obs_half = torch.tensor(self.box_half, **tt)
        self._box_off_t = torch.tensor(self.box_off, **tt)  # [n_boxes, 2]
        # The T is ONE movable compound body: all boxes share body 0 (the root) and sit
        # at their local offsets, so the engine integrates a single pose for the set.
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
        # Cached, repack-once launches for the eager reset path — see
        # swarp/core/cached_launch.py. ``_launch_body_sync`` is excluded: it only ever
        # runs on the STEP pass, i.e. inside capture, where it is paid exactly once.
        self._reset_launch = CachedLaunch()
        self._heading_launch = CachedLaunch()
        self._actuation_launch = CachedLaunch()
        self._friction_launch = CachedLaunch()
        self._latency_launch = CachedLaunch()
        #: The command delay's ``(low, high)`` in substeps; ``None`` without a latency.
        self.cmd_delay_substeps: tuple[int, int] | None = None
        if self.cmd_latency[1] > 0.0:
            lo, hi = (min(round(x / sub_dt), substeps) for x in self.cmd_latency)
            if hi > 0:
                self.cmd_delay_substeps = (lo, hi)
                self.world.stepper.enable_command_delay(n_envs, self.world.act_dim, hi)
        if self.per_env_arena:
            # Pointer-stable [n_envs, 2] buffers the arena reset kernel rewrites in place:
            # the half-extents every arena-aware kernel reads, and the arena stop's edges.
            self.arena_half = torch.full((n_envs, 2), float(self.world_size), **tt)
            self._arena_edges = self.arena_half + self.arena_stop_margin
            vec2 = VEC2[self.world.wp_dtype]
            self._arena_half_wp = wp.from_torch(self.arena_half, dtype=vec2)
            self._arena_edges_wp = wp.from_torch(self._arena_edges, dtype=vec2)
            self.world.stepper.enable_arena_stop(n_envs, self.world.act_dim, self._arena_edges_wp)
        if self.friction_dr:
            # A per-env mu * g, allocated here (outside any capture) and redrawn in place.
            g = self.world.config.gravity
            self.world.stepper.set_body_friction(
                torch.full((n_envs,), self.floor_friction[0] * g, **tt)
            )
        if self.actuation_dr:
            # Per-env parameter rows, installed once here (outside any capture) and then
            # redrawn in place by the reset kernel, so the step's kernels keep one pointer.
            stepper = self.world.stepper
            rows = wp.to_torch(stepper.params.floats)
            stepper.set_agent_params_per_env(rows.unsqueeze(0).expand(n_envs, -1, -1).clone())
        self._obs_launch = CachedLaunch()
        self._reward_launch = CachedLaunch()
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
        # 12 own/task features + the other agents' relative positions (coordination),
        # then the robot's own cos/sin heading when it matters.
        return 12 + 2 * (self.n_agents - 1) + (2 if self.heading_obs else 0)

    # ------------------------------------------------------------------ reset

    def reset_world(
        self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False
    ) -> None:
        """Masked reset in one Warp launch (see :mod:`swarp.scenarios.reset_kernels`).

        The body buffers are written **in place** by the kernel, so the fused path's
        cached Warp handles and the whole-step graph stay valid across resets; the grad
        path's ``_refresh`` still reassigns them (fresh tensors for the tape), which the
        framework's handle resync catches on the next no-grad step.
        """
        mask, use_mask = self.reset_mask_wp(env_mask)
        self._launch_spawn(mask, use_mask)
        self.world.mark_pos_dirty()

        self._install_obstacles()
        self.finish_reset(env_mask, obs_only=obs_only)

    def _launch_spawn(self, mask, use_mask) -> None:
        """Respawn agents, body and goal in the masked envs: one ``pusht_reset_kernel``.

        ``mask``/``use_mask`` are what :meth:`reset_mask_wp` returned. A subclass that
        samples extra per-env state on reset (roles, say) overrides this, calls
        ``super()._launch_spawn(mask, use_mask)`` and launches its own kernel with the
        same mask; ``reset_world`` still marks positions dirty, re-installs the body and
        finishes the reset afterwards.
        """
        w = self.world
        lim = self.world_size - 2.0 * self.agent_radius
        tlim = self.world_size - self.tee_radius
        # Any agent inside the T's bounding disk is pushed out to its rim by the kernel:
        # with a stiff contact_k a spawn overlap is a violent ejection.
        clear = self.tee_radius + 2.0 * self.agent_radius
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
            tee_pos = self._wp["tee_pos"]
            tee_vel = self._wp["tee_vel"]
            tee_theta = self._wp["tee_theta"]
            tee_ang_vel = self._wp["tee_ang_vel"]
            goal_pos = self._wp["goal_pos"]
            goal_theta = self._wp["goal_theta"]
        else:
            tee_pos = wp.from_torch(self.tee_pos.contiguous(), dtype=vec2)
            tee_vel = wp.from_torch(self.tee_vel.contiguous(), dtype=vec2)
            tee_theta = wp.from_torch(self.tee_theta.contiguous())
            tee_ang_vel = wp.from_torch(self.tee_ang_vel.contiguous())
            goal_pos = wp.from_torch(self.goal_pos.contiguous(), dtype=vec2)
            goal_theta = wp.from_torch(self.goal_theta.contiguous())
        seed = wp.int32(w.next_kernel_seed())
        goal_radius = 0.0 if self.goal_spawn_radius is None else self.goal_spawn_radius
        goal_angle = 0.0 if self.goal_spawn_angle is None else self.goal_spawn_angle
        use_goal_radius = 0 if self.goal_spawn_radius is None else 1
        use_goal_angle = 0 if self.goal_spawn_angle is None else 1
        with torch_stream_scope(w.device):
            if self.per_env_arena:
                self._launch_arena_spawn(
                    st, mask, use_mask, seed, clear, goal_radius, goal_angle,
                    use_goal_radius, use_goal_angle,
                    (tee_pos, tee_vel, tee_theta, tee_ang_vel, goal_pos, goal_theta),
                )
            else:
                self._launch_square_spawn(
                    st, mask, use_mask, seed, lim, tlim, clear, goal_radius, goal_angle,
                    use_goal_radius, use_goal_angle,
                    (tee_pos, tee_vel, tee_theta, tee_ang_vel, goal_pos, goal_theta),
                )
            if self.heading_obs:
                self._launch_heading_reset(st, mask, use_mask)
            if self.actuation_dr:
                self._launch_actuation_reset(mask, use_mask)
            if self.friction_dr:
                self._launch_friction_reset(mask, use_mask)
            if self.cmd_delay_substeps is not None:
                self._launch_latency_reset(mask, use_mask)

    def _launch_square_spawn(
        self, st, mask, use_mask, seed, lim, tlim, clear, goal_radius, goal_angle,
        use_goal_radius, use_goal_angle, body,
    ) -> None:
        """The one square arena's spawn: ``pusht_reset_kernel``."""
        w = self.world
        scalar = w.wp_dtype
        tee_pos, tee_vel, tee_theta, tee_ang_vel, goal_pos, goal_theta = body
        launch = self._reset_launch.get(
            concrete(pusht_reset_kernel, scalar),
            dim=w.n_envs,
            inputs=[
                mask,
                use_mask,
                seed,
                scalar(lim),
                scalar(tlim),
                scalar(clear),
                scalar(goal_radius),
                scalar(goal_angle),
                wp.int32(use_goal_radius),
                wp.int32(use_goal_angle),
                wp.int32(self.n_agents),
                st.pos,
                st.vel,
                tee_pos,
                tee_vel,
                tee_theta,
                tee_ang_vel,
                goal_pos,
                goal_theta,
            ],
            device=w.device,
            key=(
                w.n_envs,
                ptr_key(mask),
                lim,
                tlim,
                clear,
                goal_radius,
                goal_angle,
                use_goal_radius,
                use_goal_angle,
                self.n_agents,
                ptr_key(st.pos),
                ptr_key(st.vel),
                ptr_key(tee_pos),
                ptr_key(tee_vel),
                ptr_key(tee_theta),
                ptr_key(tee_ang_vel),
                ptr_key(goal_pos),
                ptr_key(goal_theta),
            ),
        )
        launch.set_param_by_name("use_mask", use_mask)
        launch.set_param_by_name("seed", seed)
        launch.launch()

    def _launch_arena_spawn(
        self, st, mask, use_mask, seed, clear, goal_radius, goal_angle, use_goal_radius,
        use_goal_angle, body,
    ) -> None:
        """A per-env arena's draw and spawn: ``pusht_arena_reset_kernel``, which also
        rewrites :attr:`arena_half` and the arena stop's edges in place."""
        w = self.world
        scalar = w.wp_dtype
        tee_pos, tee_vel, tee_theta, tee_ang_vel, goal_pos, goal_theta = body
        w_lo, w_hi, h_lo, h_hi = arena_ranges(self.arena_size)
        agent_inset = 2.0 * self.agent_radius
        agent_clear = 2.0 * self.agent_radius
        # The bare clearance a crowded arena falls back to: the T's bounding disc and a
        # robot's, 2 cm apart.
        clear_min = self.tee_radius + self.agent_radius + 0.02
        launch = self._reset_launch.get(
            concrete(pusht_arena_reset_kernel, scalar),
            dim=w.n_envs,
            inputs=[
                mask, use_mask, seed,
                scalar(w_lo), scalar(w_hi), scalar(h_lo), scalar(h_hi),
                scalar(agent_inset), scalar(self.tee_radius), scalar(clear), scalar(clear_min),
                scalar(agent_clear), scalar(self.arena_stop_margin),
                scalar(goal_radius), scalar(goal_angle),
                wp.int32(use_goal_radius), wp.int32(use_goal_angle), wp.int32(self.n_agents),
                self._arena_half_wp, self._arena_edges_wp,
                st.pos, st.vel, tee_pos, tee_vel, tee_theta, tee_ang_vel, goal_pos, goal_theta,
            ],
            device=w.device,
            key=(
                w.n_envs, ptr_key(mask), w_lo, w_hi, h_lo, h_hi, agent_inset,
                self.tee_radius, clear, clear_min, agent_clear, self.arena_stop_margin,
                goal_radius,
                goal_angle, use_goal_radius, use_goal_angle, self.n_agents,
                ptr_key(self._arena_half_wp), ptr_key(self._arena_edges_wp),
                ptr_key(st.pos), ptr_key(st.vel), ptr_key(tee_pos), ptr_key(tee_vel),
                ptr_key(tee_theta), ptr_key(tee_ang_vel), ptr_key(goal_pos),
                ptr_key(goal_theta),
            ),
        )
        launch.set_param_by_name("use_mask", use_mask)
        launch.set_param_by_name("seed", seed)
        launch.launch()

    def _launch_heading_reset(self, st, mask, use_mask) -> None:
        """Random headings, zero speed and spin for the respawned robots (only launched
        when the heading matters, so the default fleet's reset is unchanged)."""
        w = self.world
        seed = wp.int32(w.next_kernel_seed())
        launch = self._heading_launch.get(
            concrete(pusht_heading_reset_kernel, w.wp_dtype),
            dim=(w.n_envs, self.n_agents),
            inputs=[mask, use_mask, seed, st.theta, st.speed, st.ang_vel],
            device=w.device,
            key=(
                w.n_envs,
                ptr_key(mask),
                ptr_key(st.theta),
                ptr_key(st.speed),
                ptr_key(st.ang_vel),
            ),
        )
        launch.set_param_by_name("use_mask", use_mask)
        launch.set_param_by_name("seed", seed)
        launch.launch()

    def _launch_friction_reset(self, mask, use_mask) -> None:
        """Redraw the respawned envs' floor friction (``friction_dr``)."""
        w = self.world
        scalar = w.wp_dtype
        seed = wp.int32(w.next_kernel_seed())
        out = w.stepper.body_friction
        lo, hi = self.floor_friction
        launch = self._friction_launch.get(
            concrete(pusht_friction_reset_kernel, scalar),
            dim=w.n_envs,
            inputs=[mask, use_mask, seed, scalar(lo), scalar(hi),
                    scalar(w.config.gravity), out],
            device=w.device,
            key=(w.n_envs, ptr_key(mask), ptr_key(out)),
        )
        launch.set_param_by_name("use_mask", use_mask)
        launch.set_param_by_name("seed", seed)
        launch.launch()

    def _launch_latency_reset(self, mask, use_mask) -> None:
        """Redraw the respawned robots' command delays and zero their last command."""
        w = self.world
        st = w.stepper
        seed = wp.int32(w.next_kernel_seed())
        lo, hi = self.cmd_delay_substeps
        launch = self._latency_launch.get(
            concrete(pusht_latency_reset_kernel, w.wp_dtype),
            dim=(w.n_envs, self.n_agents),
            inputs=[mask, use_mask, seed, wp.int32(lo), wp.int32(hi), w.wp_dtype(0.0),
                    st.cmd_delay, st.prev_actions],
            device=w.device,
            key=(w.n_envs, ptr_key(mask), lo, hi, ptr_key(st.cmd_delay),
                 ptr_key(st.prev_actions)),
        )
        launch.set_param_by_name("use_mask", use_mask)
        launch.set_param_by_name("seed", seed)
        launch.launch()

    def _launch_actuation_reset(self, mask, use_mask) -> None:
        """Redraw the respawned envs' drive accelerations and gains (``actuation_dr``)."""
        w = self.world
        scalar = w.wp_dtype
        seed = wp.int32(w.next_kernel_seed())
        params = w.stepper.params.floats_per_env
        bounds = [scalar(x) for lo_hi in self._actuation_ranges for x in lo_hi]
        launch = self._actuation_launch.get(
            concrete(pusht_actuation_reset_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[mask, use_mask, seed, *bounds, params],
            device=w.device,
            key=(w.n_envs, ptr_key(mask), ptr_key(params)),
        )
        launch.set_param_by_name("use_mask", use_mask)
        launch.set_param_by_name("seed", seed)
        launch.launch()

    def _box_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The box centres ``[n_envs, n_boxes, 2]`` and angles ``[n_envs, n_boxes]``
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
            if self.agent_shape == AgentShape.BOX:
                raise NotImplementedError(
                    "a taped Push-T step integrates the T in torch against disc robots; "
                    "box robots push it only on the no-grad path"
                )
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
        out = [*w.obstacle_state_views(), *w.obstacle_state_views(body=True)]
        if w.stepper.prev_actions is not None:
            # The step copies its command into the latency's last-command buffer.
            out.append(wp.to_torch(w.stepper.prev_actions))
        return out

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
            concrete(body_state_gather_kernel, self.world.wp_dtype),
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
        tee_pos, tee_theta = self._wp["tee_pos"], self._wp["tee_theta"]
        goal_pos, goal_theta = self._wp["goal_pos"], self._wp["goal_theta"]
        obs = self._wp["obs"]
        # No per-call-varying argument: pointer-stable persistent-mode/watched handles.
        launch = self._obs_launch.get(
            concrete(pusht_obs_kernel, self.world.wp_dtype),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                st.theta,
                tee_pos,
                tee_theta,
                goal_pos,
                goal_theta,
                wp.int32(self.heading_obs),
            ],
            outputs=[obs],
            device=w.device,
            key=(
                w.n_envs,
                self.n_agents,
                ptr_key(st.pos),
                ptr_key(st.vel),
                ptr_key(st.theta),
                ptr_key(tee_pos),
                ptr_key(tee_theta),
                ptr_key(goal_pos),
                ptr_key(goal_theta),
                ptr_key(obs),
            ),
        )
        launch.launch()

    def _launch_reward(self, st, advance_prev: int, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        tee_pos, tee_theta = self._wp["tee_pos"], self._wp["tee_theta"]
        goal_pos, goal_theta = self._wp["goal_pos"], self._wp["goal_theta"]
        resetmask = self._wp["resetmask"]
        prev_dist, prev_ang, prev_adist = (
            self._wp["prev_dist"],
            self._wp["prev_ang"],
            self._wp["prev_adist"],
        )
        reward, done, dist, ang = (
            self._wp["reward"],
            self._wp["done"],
            self._wp["dist"],
            self._wp["ang"],
        )
        launch = self._reward_launch.get(
            concrete(pusht_reward_kernel, self.world.wp_dtype),
            dim=w.n_envs,
            inputs=[
                st.pos,
                tee_pos,
                tee_theta,
                goal_pos,
                goal_theta,
                resetmask,
                wp.int32(self.n_agents),
                scalar(self.pos_shaping_factor),
                scalar(self.rot_shaping_factor),
                scalar(self.agent_dist_shaping),
                scalar(self.push_point_offset),
                scalar(self.joint_shaping),
                scalar(self.rot_away_penalty),
                scalar(self.goal_tolerance),
                scalar(self.angle_tolerance),
                scalar(self.goal_reward),
                wp.int32(advance_prev),
                wp.int32(full_pass),
            ],
            outputs=[prev_dist, prev_ang, prev_adist, reward, done, dist, ang],
            device=w.device,
            key=(
                w.n_envs,
                ptr_key(st.pos),
                ptr_key(tee_pos),
                ptr_key(tee_theta),
                ptr_key(goal_pos),
                ptr_key(goal_theta),
                ptr_key(resetmask),
                self.n_agents,
                self.pos_shaping_factor,
                self.rot_shaping_factor,
                self.agent_dist_shaping,
                self.push_point_offset,
                self.joint_shaping,
                self.rot_away_penalty,
                self.goal_tolerance,
                self.angle_tolerance,
                self.goal_reward,
                ptr_key(prev_dist),
                ptr_key(prev_ang),
                ptr_key(prev_adist),
                ptr_key(reward),
                ptr_key(done),
                ptr_key(dist),
                ptr_key(ang),
            ),
        )
        launch.set_param_by_name("advance_prev", wp.int32(advance_prev))
        launch.set_param_by_name("full_pass", wp.int32(full_pass))
        launch.launch()

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
            b = math.inf if self.open_arena else self.world_size - self.tee_radius
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
                new_pos = self.tee_pos + self.tee_vel * sub_dt
                self.tee_pos = new_pos.clamp(-b, b)
                # A clamped axis loses its velocity, as in the engine kernel.
                self.tee_vel = torch.where(
                    self.tee_pos != new_pos, torch.zeros_like(self.tee_vel), self.tee_vel
                )
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
        shaping = shaping - self.rot_away_penalty * (angle_error - self._prev_ang).clamp(min=0)
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
        parts = [s.pos, s.vel, self._cache["tee_rel"], tee_to_goal, angles, rel]
        if self.heading_obs:
            parts.append(torch.stack([s.theta.cos(), s.theta.sin()], dim=-1))
        return torch.cat(parts, dim=-1)

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

        Same boxes the body is built from, placed at ``(goal_pos, goal_theta)``:
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
