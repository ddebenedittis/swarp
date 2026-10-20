"""Lidar Push-T: a team of robots pushes a T it cannot see except through its lidars.

See :class:`PushTLidarScenario` and ``docs/pusht-lidar.md``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from swarp._overloads import concrete
from swarp.core.cached_launch import CachedLaunch, ptr_key
from swarp.core.state import VEC2
from swarp.core.world import World
from swarp.dynamics.base import (
    P_CMD_GAIN_V,
    P_CMD_GAIN_W,
    P_MAX_ACCEL,
    P_MAX_ANG_ACCEL,
    AgentShape,
)
from swarp.interop.autograd import torch_stream_scope
from swarp.scenarios.fused import Buf, FusedPass
from swarp.scenarios.pusht import PushTScenario
from swarp.scenarios.pusht_lidar_artefacts import (
    ART_BIAS,
    ART_DIM,
    ART_DROPOUT,
    ART_SCALE,
    ART_SPURIOUS,
    lidar_artefact_kernel,
    lidar_artefact_reset_kernel,
    lidar_relabel_kernel,
    pusht_lidar_critic_dr_kernel,
)
from swarp.scenarios.pusht_lidar_kernels import (
    FOV_EPS,
    N_OBJ,
    N_SECTORS,
    OBJECTIVES,
    SELF_DIM,
    STATE_AGENT,
    STATE_HEAD,
    TEE_DIM,
    lidar_sensor_kernel,
    pose_obs_noise_kernel,
    pusht_lidar_dr_reset_kernel,
    pusht_lidar_obs_kernel,
    pusht_lidar_reward_kernel,
    pusht_lidar_roles_kernel,
)
from swarp.scenarios.shapes import CompoundShape
from swarp.sensors.lidar import Lidar, lidar_scan
from swarp.sensors.lidar_kernels import HIT_AGENT, HIT_MOVABLE


def _range(v: float | tuple[float, float], name: str) -> tuple[float, float]:
    """``(low, high)`` of a knob given as a value or a range, both ends ``>= 0``."""
    lo, hi = (float(v), float(v)) if isinstance(v, int | float) else (float(v[0]), float(v[1]))
    if not 0.0 <= lo <= hi:
        raise ValueError(f"{name} must be >= 0 with low <= high, got {v!r}")
    return (lo, hi)


class PushTLidarScenario(PushTScenario):
    """Multi-robot Push-T observed through lidar, with pusher and observer roles.

    ``n_agents`` identical holonomic robots, ``n_pushers`` of them pushers (role 0) and the
    rest observers (role 1). Which robot holds which role is drawn per env per episode by a
    masked Fisher-Yates kernel, with the counts fixed. Every robot can push the T; the
    roles differ only in their rewards and, in the trainer, in the policy head. A robot
    knows its own pose, the goal pose and its teammates' positions and roles; from
    ``obs_level=1`` on it does not observe the T's pose.

    Observation (``obs_dim`` wide, fixed; :meth:`obs_layout` returns the slices). ``A``
    agents, ``R = n_rays``, ``F = frame_stack``, ``K = n_comm_pts``:

    ========== ========================= ==============================================
    level      block                     width
    ========== ========================= ==============================================
    all        own pos, vel              4
    all        role sign                 1  (+1 observer, -1 pusher)
    all        goal pos - own pos        2
    all        cos, sin goal heading     2
    all        teammates                 3 (A - 1): relative pos, role sign; ascending
                                         index, self skipped
    0 only     the T (oracle)            8: Push-T's tee_rel, tee_to_goal, cos/sin theta,
                                         cos/sin(theta - goal theta)
    all        lidar, F frames           3 R F: per frame range / max_range (R), object
                                         mask (R), agent mask (R); oldest frame first
    2 only     pooled object hits        3 A K: per agent j (self included, ascending),
                                         K hit points relative to own pos, plus a valid
                                         flag
    ========== ========================= ==============================================

    With the defaults (``A = 5``, ``R = 32``, ``F = 1``, ``K = 6``) that is 125 at level 0,
    117 at level 1 and 207 at level 2. Rays are world-frame (``body_frame=False``) because
    a holonomic robot's heading never changes; frame stacking keeps a ring of the last
    ``F`` scans, zeroed at reset. Pooled slot ``k`` of agent ``j`` is ``j``'s object hit
    number ``floor((k + 0.5) n_hit / K)`` in ray order, in the world frame (every robot
    can compute it because it knows its own pose); it is invalid when ``j`` has no object
    hit or is farther than ``comm_range``.

    Rewards, one row per agent (``info()["multiobj_reward"]`` keeps the seven terms, see
    :data:`~swarp.scenarios.pusht_lidar_kernels.OBJECTIVES`, and sums to the reward):

    * ``team`` — every robot: Push-T's pose shaping plus its goal bonus;
    * ``approach`` — pushers: Push-T's approach-to-push-point term;
    * ``standoff`` — observers: a potential on ``-| |p - T| - standoff_dist |``;
    * ``visibility`` — observers: ``-w_v (1 - min(n_obj / (f* R), 1))``, ``n_obj`` the
      rays that hit the T;
    * ``coverage`` — observers: ``-w_c (1 - c)``, ``c`` the fraction of 8 sectors around
      the T centroid (T frame) that hold an observer's object-hit point;
    * ``occlusion`` — observers: ``-w_o`` times the fraction of the observer's rays
      inside the T's angular cone (its bounding disc) that hit a robot first;
    * ``contact`` — observers: ``-contact_penalty`` while an object ray is shorter than
      ``contact_dist`` (default ``0``).

    The observer terms are penalties or potential differences, so prolonging an episode
    never pays. Training hooks: :meth:`critic_state` (role-sorted), :meth:`agent_roles`,
    :meth:`aux_targets` (the T in each robot's frame), all written by the observation
    pass.

    Sensor model (all off by default). The rays are cast from the robot's centre; a real
    lidar sits ``lidar_mount_offset`` ahead of it and returns nothing closer than
    ``lidar_min_range`` to itself, so a hit within that distance of the mount reads as a
    miss (range ``lidar_max_range``, no class). ``lidar_fov`` below ``2 pi`` also blanks
    the rays more than half of it off the heading (a body blocking the rear), and
    ``lidar_noise`` adds Gaussian noise of that standard deviation to every remaining
    hit's range (clamped to ``[0, lidar_max_range]``). These edit the scan itself, so the
    observation, the auxiliary targets' visibility statistics and the reward inputs all
    see the sensor's scan.

    Scan artefacts (all off by default; :mod:`swarp.scenarios.pusht_lidar_artefacts`).
    ``lidar_span`` below ``2 pi`` spans the rays over that angle centred on the heading
    (first and last ray at its ends) instead of the full circle. ``lidar_scan_period``
    turns the scan into a rotating one, each ray cast from the pose at its own time in the
    sweep, with a per-robot mount yaw error of up to ``lidar_yaw_bias`` and a per-robot
    stamp phase drawn from ``lidar_stamp_phase`` (:func:`~swarp.sensors.lidar.
    sweep_offsets`). Before the sensor model, a T hit at incidence cosine ``c`` is lost
    with probability ``1 - c / lidar_grazing_cos``, any hit with a per-robot probability
    drawn from ``lidar_dropout``, any ray returns a spurious T hit with a per-robot
    probability drawn from ``lidar_spurious``, and the hits' ranges take a per-robot scale
    error and bias of up to ``lidar_range_scale`` and ``lidar_range_bias``. With
    ``lidar_label_from_pose`` the hit classes are the deployment's, from the reported
    poses: a hit inside a teammate's reported footprint grown by ``lidar_label_margin``
    is an agent, a hit beyond ``label_bound`` a miss, anything else the T.
    """

    #: Two roles: pushers (0) and observers (1).
    n_roles: int = 2

    # The fused path reads the engine's box poses, the torch oracle rebuilds them from the
    # body pose, and rays that graze a box corner or the occlusion cone can be classified
    # differently by the two. Continuous fields stay within Push-T's bound.
    parity_rtol: float = 1e-2
    parity_atol: float = 5e-3

    def __init__(
        self,
        n_agents: int = 5,
        n_pushers: int = 2,
        obs_level: int = 1,
        n_rays: int = 32,
        lidar_max_range: float | None = None,
        n_comm_pts: int = 6,
        comm_range: float = math.inf,
        frame_stack: int = 1,
        shape: str | CompoundShape | None = "tee",
        standoff_dist: float = 0.5,
        standoff_shaping: float = 1.0,
        visibility_target: float = 0.1,
        visibility_weight: float = 0.02,
        coverage_weight: float = 0.02,
        occlusion_weight: float = 0.1,
        contact_penalty: float = 0.0,
        contact_dist: float | None = None,
        lidar_body_frame: bool | None = None,
        lidar_mount_offset: float = 0.0,
        lidar_min_range: float = 0.0,
        lidar_fov: float | tuple[float, float] = 2.0 * math.pi,
        lidar_noise: float | tuple[float, float] = 0.0,
        lidar_fov_at_mount: bool = False,
        pose_noise: float | tuple[float, float] = 0.0,
        pose_bias_std: tuple[float, float] = (0.03, 0.03),
        pose_jitter_std: tuple[float, float] = (0.02, 0.02),
        vel_noise_std: float = 0.02,
        pose_drift_tau: float | tuple[float, float] = 0.0,
        pose_drift_std: tuple[float, float] = (0.0, 0.0),
        pose_slip_gain: tuple[float, float] = (1.0, 1.0),
        oob_margin: float | None = None,
        oob_penalty: float = 0.05,
        lidar_span: float = 2.0 * math.pi,
        lidar_dropout: float | tuple[float, float] = 0.0,
        lidar_spurious: float | tuple[float, float] = 0.0,
        lidar_grazing_cos: float = 0.0,
        lidar_range_scale: float = 0.0,
        lidar_range_bias: float = 0.0,
        lidar_scan_period: float = 0.0,
        lidar_sweep_dir: float = 1.0,
        lidar_stamp_phase: float | tuple[float, float] = 1.0,
        lidar_yaw_bias: float = 0.0,
        lidar_label_from_pose: bool = False,
        lidar_label_margin: float = 0.03,
        critic_dr: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_agents=n_agents, shape=shape, **kwargs)
        # A unicycle pushes along its heading, so its scan is mounted on the body (the T
        # straight ahead is always the same rays); a holonomic disc keeps world-axis rays.
        self.lidar_body_frame = (
            self.heading_obs if lidar_body_frame is None else bool(lidar_body_frame)
        )
        #: Widths of the self block and of a robot's critic-state block: the heading adds
        #: its cos and sin to both when it matters.
        self.self_dim = SELF_DIM + (2 if self.heading_obs else 0)
        self.state_agent_dim = STATE_AGENT + (2 if self.heading_obs else 0)
        if obs_level not in (0, 1, 2):
            raise ValueError(f"obs_level must be 0, 1 or 2, got {obs_level}")
        if n_rays < 1 or frame_stack < 1 or n_comm_pts < 1:
            raise ValueError("n_rays, frame_stack and n_comm_pts must all be >= 1")
        self.n_pushers = max(0, min(int(n_pushers), n_agents))
        self.obs_level = obs_level
        self.n_rays = n_rays
        # 0.8 world_size: 40% of the arena side. A robot sees the T from about three T
        # radii away but not from across the arena, and the T's narrowest silhouette
        # (~0.4 m for the default T, broadside or end-on) still spans two to three ray
        # spacings (2 pi / 32) at the edge of the range.
        self.lidar_max_range = (
            0.8 * self.world_size if lidar_max_range is None else float(lidar_max_range)
        )
        self.n_comm_pts = n_comm_pts
        self.comm_range = float(comm_range)
        self.frame_stack = frame_stack
        # An observer resting at the standoff must clear the body's farthest corner, or the
        # shaping parks it against the body and it jams the rotation the pushers need
        # (a 0.22 standoff against the T's 0.265 corner radius left the angle error flat).
        if self.n_agents > self.n_pushers and standoff_dist < self.tee_radius + self.agent_radius:
            raise ValueError(
                f"standoff_dist {standoff_dist} puts observers inside the body: it must be at "
                f"least tee_radius + agent_radius = {self.tee_radius + self.agent_radius:.3f}"
            )
        self.standoff_dist = standoff_dist
        self.standoff_shaping = standoff_shaping
        self.visibility_target = visibility_target
        self.visibility_weight = visibility_weight
        self.coverage_weight = coverage_weight
        self.occlusion_weight = occlusion_weight
        self.contact_penalty = contact_penalty
        # Default: an object ray shorter than two robot radii, i.e. the robot within one
        # radius of the T's surface along that ray.
        self.contact_dist = 2.0 * self.agent_radius if contact_dist is None else contact_dist
        # The auxiliary T offset is scaled by the lidar range, so an in-range T is in [-1, 1].
        self.aux_scale = 1.0 / self.lidar_max_range
        fov = _range(lidar_fov, "lidar_fov")
        noise = _range(lidar_noise, "lidar_noise")
        pose = _range(pose_noise, "pose_noise")
        if lidar_min_range < 0.0:
            raise ValueError("lidar_min_range must be >= 0")
        if not (fov[0] > 0.0 and fov[1] <= 2.0 * math.pi):
            raise ValueError(f"lidar_fov must lie in (0, 2 pi], got {lidar_fov}")
        self.lidar_mount_offset = float(lidar_mount_offset)
        self.lidar_min_range = float(lidar_min_range)
        #: ``(low, high)`` of the field of view and of the range noise's deviation, drawn
        #: per env and robot on every reset when ``low < high``.
        self.lidar_fov = fov
        self.lidar_noise = noise
        #: Whether the field of view is measured at the mount (the sensor's own crop) rather
        #: than per centre ray.
        self.lidar_fov_at_mount = bool(lidar_fov_at_mount)
        #: Whether the scan goes through :func:`lidar_sensor_kernel` (and its torch twin).
        self.sensor_model = (
            self.lidar_min_range > 0.0 or noise[1] > 0.0 or fov[0] < 2.0 * math.pi
        )
        #: ``(low, high)`` of the per-env pose-noise scale; ``pose_bias_std``,
        #: ``pose_jitter_std`` (position, heading) and ``vel_noise_std`` are at scale 1.
        self.pose_noise = pose
        self.pose_bias_std = (float(pose_bias_std[0]), float(pose_bias_std[1]))
        self.pose_jitter_std = (float(pose_jitter_std[0]), float(pose_jitter_std[1]))
        self.vel_noise_std = float(vel_noise_std)
        #: Whether the self and teammate blocks observe a noisy pose.
        self.pose_noise_on = pose[1] > 0.0
        tau = (
            (float(pose_drift_tau),) * 2
            if isinstance(pose_drift_tau, int | float)
            else (float(pose_drift_tau[0]), float(pose_drift_tau[1]))
        )
        if min(tau) < 0.0 or (max(tau) > 0.0 and min(tau) == 0.0):
            raise ValueError(f"pose_drift_tau must be 0 or positive, got {pose_drift_tau}")
        #: The localisation drift (see :func:`pose_obs_noise_kernel`): time constant in
        #: seconds (position, heading; 0: off), stationary deviation (position, heading)
        #: at scale 1, and the share of the wheel-odometry slip that enters it (position,
        #: heading).
        self.pose_drift_tau = tau
        self.pose_drift_std = (float(pose_drift_std[0]), float(pose_drift_std[1]))
        self.pose_slip_gain = (float(pose_slip_gain[0]), float(pose_slip_gain[1]))
        self.pose_drift_on = self.pose_noise_on and tau[0] > 0.0
        #: Out of bounds: once the T's centroid is more than ``world_size + oob_margin``
        #: from the centre on either axis, the episode has failed (see
        #: :func:`pusht_lidar_reward_kernel`); ``None`` never fails it.
        self.oob_limit = 0.0 if oob_margin is None else self.world_size + float(oob_margin)
        self.oob_penalty = float(oob_penalty)
        #: Whether a reset draws per-robot sensor or pose-noise values.
        self.sensor_dr = fov[0] < fov[1] or noise[0] < noise[1] or self.pose_noise_on
        if not 0.0 < lidar_span <= 2.0 * math.pi:
            raise ValueError(f"lidar_span must lie in (0, 2 pi], got {lidar_span}")
        #: The angle the rays span, centred on the heading (``2 pi``: the full circle).
        self.lidar_span = float(lidar_span)
        drop = _range(lidar_dropout, "lidar_dropout")
        spur = _range(lidar_spurious, "lidar_spurious")
        phase = _range(lidar_stamp_phase, "lidar_stamp_phase")
        if drop[1] > 1.0 or spur[1] > 1.0 or phase[1] > 1.0:
            raise ValueError("lidar_dropout, lidar_spurious and lidar_stamp_phase lie in [0, 1]")
        widths = (lidar_grazing_cos, lidar_range_scale, lidar_range_bias, lidar_scan_period,
                  lidar_yaw_bias, lidar_label_margin)
        if min(widths) < 0.0 or lidar_grazing_cos > 1.0:
            raise ValueError("the lidar artefact widths must be >= 0 (lidar_grazing_cos <= 1)")
        #: The scan artefacts (:func:`~swarp.scenarios.pusht_lidar_artefacts.
        #: lidar_artefact_kernel`): ``(low, high)`` of the per-robot dropout and
        #: spurious-return probabilities, the grazing threshold on the incidence cosine,
        #: and the half-widths of the per-robot range scale error and range bias.
        self.lidar_dropout = drop
        self.lidar_spurious = spur
        self.lidar_grazing_cos = float(lidar_grazing_cos)
        self.lidar_range_scale = float(lidar_range_scale)
        self.lidar_range_bias = float(lidar_range_bias)
        #: Whether the scan goes through the artefact kernel.
        self.lidar_artefacts = (
            drop[1] > 0.0 or spur[1] > 0.0 or self.lidar_grazing_cos > 0.0
            or self.lidar_range_scale > 0.0 or self.lidar_range_bias > 0.0
        )
        #: The rotating scan (:func:`~swarp.sensors.lidar.sweep_offsets`): its period (0:
        #: off), direction, the ``(low, high)`` of the per-robot stamp phase and the
        #: half-width of the per-robot mount yaw error.
        self.lidar_scan_period = float(lidar_scan_period)
        self.lidar_sweep_dir = float(lidar_sweep_dir)
        self.lidar_stamp_phase = phase
        self.lidar_yaw_bias = float(lidar_yaw_bias)
        self.lidar_sweep = self.lidar_scan_period > 0.0
        if self.lidar_yaw_bias > 0.0 and not self.lidar_sweep:
            raise ValueError("lidar_yaw_bias needs lidar_scan_period > 0 (the sweep kernel)")
        #: Whether a reset draws per-robot artefact values.
        self.artefact_dr = self.lidar_artefacts or self.lidar_sweep
        #: Whether the hit classes come from the deployment's rule on the reported poses
        #: (:func:`~swarp.scenarios.pusht_lidar_artefacts.lidar_relabel_kernel`), with the
        #: teammates' footprints grown by ``lidar_label_margin`` and hits beyond
        #: ``label_bound`` on either axis dropped: the out-of-bounds limit when there is
        #: one, else the arena (none in an open arena).
        self.lidar_label_from_pose = bool(lidar_label_from_pose)
        self.lidar_label_margin = float(lidar_label_margin)
        if self.oob_limit > 0.0:
            self.label_bound = self.oob_limit
        else:
            self.label_bound = math.inf if self.open_arena else float(self.world_size)
        #: Whether the critic state carries the randomised parameters (:meth:`critic_state`).
        self.critic_dr = bool(critic_dr)
        self.dr_env_dim, self.dr_agent_dim = self._critic_dr_layout() if critic_dr else (0, 0)

    # ------------------------------------------------------------- dimensions

    @property
    def n_observers(self) -> int:
        return self.n_agents - self.n_pushers

    @property
    def role_counts(self) -> tuple[int, ...]:  # type: ignore[override]
        return (self.n_pushers, self.n_observers)

    @property
    def state_dim(self) -> int:  # type: ignore[override]
        return self.state_base_dim + self.dr_env_dim + self.dr_agent_dim * self.n_agents

    @property
    def state_base_dim(self) -> int:
        """Width of the critic state without the randomised parameters."""
        return STATE_HEAD + self.state_agent_dim * self.n_agents

    def _critic_dr_layout(self) -> tuple[int, int]:
        """``(env, per-robot)`` widths of the randomised-parameter block: one column per
        parameter a reset draws (see :meth:`critic_state`)."""
        env = int(self.friction_dr)
        agent = sum(lo < hi for lo, hi in self._actuation_ranges)
        agent += int(self.cmd_latency[0] < self.cmd_latency[1])
        agent += int(self.lidar_noise[0] < self.lidar_noise[1])
        if self.pose_noise_on:
            agent += int(self.pose_noise[0] < self.pose_noise[1]) + 3
            agent += 3 if self.pose_drift_on else 0
        agent += len(self._art_columns())
        agent += 2 if self._sweep_dr else 0
        return env, agent

    def _art_columns(self) -> list[tuple[int, float, float]]:
        """``(art column, mid, half-width)`` of every drawn artefact parameter."""
        cols = []
        for col, (lo, hi) in ((ART_DROPOUT, self.lidar_dropout),
                              (ART_SPURIOUS, self.lidar_spurious)):
            if lo < hi:
                cols.append((col, 0.5 * (lo + hi), 0.5 * (hi - lo)))
        for col, w in ((ART_SCALE, self.lidar_range_scale), (ART_BIAS, self.lidar_range_bias)):
            if w > 0.0:
                cols.append((col, 0.0, w))
        return cols

    @property
    def _sweep_dr(self) -> bool:
        lo, hi = self.lidar_stamp_phase
        return self.lidar_sweep and (self.lidar_yaw_bias > 0.0 or lo < hi)

    @property
    def aux_dim(self) -> int:  # type: ignore[override]
        return 4

    @property
    def obs_dim(self) -> int:
        return self.obs_layout()["_end"].stop

    def obs_layout(self) -> dict[str, slice]:
        """Name -> slice of the observation row. ``"_end"`` is the empty slice at
        ``obs_dim``. ``"lidar"`` covers every stacked frame, ``"lidar_newest"`` the last."""
        na, w = self.n_agents, 3 * self.n_rays
        out = {
            "self": slice(0, self.self_dim),
            "pos": slice(0, 2),
            "vel": slice(2, 4),
            "role": slice(4, 5),
            "goal_rel": slice(5, 7),
            "goal_angle": slice(7, 9),
        }
        if self.heading_obs:
            out["heading"] = slice(9, 11)
        k = self.self_dim
        out["teammates"] = slice(k, k + 3 * (na - 1))
        k += 3 * (na - 1)
        if self.obs_level == 0:
            out["tee"] = slice(k, k + TEE_DIM)
            k += TEE_DIM
        out["lidar"] = slice(k, k + w * self.frame_stack)
        out["lidar_newest"] = slice(k + w * (self.frame_stack - 1), k + w * self.frame_stack)
        k += w * self.frame_stack
        if self.obs_level == 2:
            out["comm"] = slice(k, k + 3 * na * self.n_comm_pts)
            k += 3 * na * self.n_comm_pts
        out["_end"] = slice(k, k)
        return out

    # ------------------------------------------------------------------ world

    def make_world(self, n_envs, device, dt, substeps, dtype, world_config=None) -> World:
        world = super().make_world(n_envs, device, dt, substeps, dtype, world_config)
        na = self.n_agents
        tt = {"device": device, "dtype": dtype}
        # Roles: int32, written in place by the role kernel on every reset, so both the
        # tensor and its Warp handle are fixed for the run. Seeded with the sorted
        # multiset so even an un-reset world honours role_counts.
        base = torch.ones(na, device=device, dtype=torch.int32)
        base[: self.n_pushers] = 0
        self.roles = base.unsqueeze(0).repeat(n_envs, 1).contiguous()
        self._roles_wp = wp.from_torch(self.roles, dtype=wp.int32)
        self._roles_launch = CachedLaunch()
        # The lidar ring (see the class docstring). Adopted by the fused spec; the torch
        # path reassigns it, which the spec's watch catches.
        self._ring = torch.zeros(n_envs, na, 3 * self.n_rays * self.frame_stack, **tt)
        self._prev_standoff: torch.Tensor | None = None
        # The fused path's lidar: bound here, outside any capture, so launching it inside
        # the captured step is a bare wp.launch on pointer-stable buffers.
        half_span = 0.5 * self.lidar_span
        self.lidar = Lidar(
            self.n_rays,
            self.lidar_max_range,
            body_frame=self.lidar_body_frame,
            angle_start=-half_span if self.lidar_span < 2.0 * math.pi else 0.0,
            angle_end=half_span if self.lidar_span < 2.0 * math.pi else None,
            return_hits=True,
            backend="warp",
            scan_period=self.lidar_scan_period,
            sweep_dir=self.lidar_sweep_dir,
        )
        self.lidar.bind(world)
        if self.lidar_sweep:
            self.lidar.stamp_phase.fill_(self.lidar_stamp_phase[0])
        if self.sensor_model:
            # One RNG stream per robot for the range noise, advanced in place by the sensor
            # kernel (so it draws fresh noise on every replay of a captured step).
            self._lidar_rng = self._rng_streams(world)
            # Per-robot half field of view and noise deviation, at the ranges' low ends
            # until a sensor_dr reset redraws them in place.
            self._half_fov = torch.full((n_envs, na), 0.5 * self.lidar_fov[0], **tt)
            self._noise_std = torch.full((n_envs, na), self.lidar_noise[0], **tt)
            self._half_fov_wp = wp.from_torch(self._half_fov)
            self._noise_std_wp = wp.from_torch(self._noise_std)
        if self.sensor_dr:
            vec2 = VEC2[world.wp_dtype]
            self._pose_scale = torch.zeros(n_envs, na, **tt)
            self._bias_pos = torch.zeros(n_envs, na, 2, **tt)
            self._bias_theta = torch.zeros(n_envs, na, **tt)
            self._pose_dr_wp = [
                wp.from_torch(self._pose_scale),
                wp.from_torch(self._bias_pos, dtype=vec2),
                wp.from_torch(self._bias_theta),
            ]
            if not self.sensor_model:
                self._half_fov_wp = wp.zeros((n_envs, na), dtype=world.wp_dtype,
                                             device=world.device)
                self._noise_std_wp = wp.zeros_like(self._half_fov_wp)
            self._dr_launch = CachedLaunch()
        if self.pose_noise_on:
            # The pose the observation reports, written by the noise kernel before the
            # observation kernel on every pass, and its RNG streams.
            self._pose_rng = self._rng_streams(world)
            self._obs_pos = torch.zeros(n_envs, na, 2, **tt)
            self._obs_vel = torch.zeros(n_envs, na, 2, **tt)
            self._obs_theta = torch.zeros(n_envs, na, **tt)
            self._obs_pose_wp = [
                wp.from_torch(self._obs_pos, dtype=vec2),
                wp.from_torch(self._obs_vel, dtype=vec2),
                wp.from_torch(self._obs_theta),
            ]
        # The drift's carry (pose, speed and turn rate after the last pass, then the drift),
        # adopted by the fused spec; a one-element dummy fills the kernel slot without it.
        shape = (n_envs, na, 8) if self.pose_drift_on else (1, 1, 8)
        self._drift_state = torch.zeros(*shape, **tt)
        self._drift_wp = wp.from_torch(self._drift_state)
        # The out-of-bounds latch (a fused carry when the check is on).
        self._oob = torch.zeros(n_envs if self.oob_limit > 0.0 else 1, dtype=torch.uint8,
                                device=device)
        self._oob_wp = wp.from_torch(self._oob)
        if self.artefact_dr:
            # Per-robot artefact values (ART_* columns), at the ranges' low ends until a
            # reset redraws them in place, and their RNG streams.
            art = torch.zeros(n_envs, na, ART_DIM, **tt)
            art[..., ART_DROPOUT] = self.lidar_dropout[0]
            art[..., ART_SPURIOUS] = self.lidar_spurious[0]
            self._lidar_art = art
            self._lidar_art_wp = wp.from_torch(art)
            self._art_rng = self._rng_streams(world)
            self._art_launch = CachedLaunch()
        if self.critic_dr:
            self._setup_critic_dr(world)
        self._lidar_obs_launch = CachedLaunch()
        self._lidar_reward_launch = CachedLaunch()
        self._lcache: dict[str, torch.Tensor] | None = None
        return world

    def _rng_streams(self, world: World) -> wp.array:
        """``[n_envs, n_agents]`` uint32 RNG states, one stream per robot, seeded from the
        world's kernel seed."""
        gen = torch.Generator().manual_seed(world.next_kernel_seed())
        seeds = torch.randint(
            0, 2**31 - 1, (world.n_envs, self.n_agents), generator=gen, dtype=torch.int64
        )
        return wp.array(seeds.numpy().astype("uint32"), dtype=wp.uint32, device=world.device)

    # ------------------------------------------------------------------ reset

    def _launch_spawn(self, mask, use_mask) -> None:
        """Push-T's spawn, then a fresh role permutation for the same envs."""
        super()._launch_spawn(mask, use_mask)
        w = self.world
        seed = wp.int32(w.next_kernel_seed())
        with torch_stream_scope(w.device):
            launch = self._roles_launch.get(
                pusht_lidar_roles_kernel,
                dim=w.n_envs,
                inputs=[mask, use_mask, seed, wp.int32(self.n_pushers)],
                outputs=[self._roles_wp],
                device=w.device,
                key=(w.n_envs, ptr_key(mask), self.n_pushers, ptr_key(self._roles_wp)),
            )
            launch.set_param_by_name("use_mask", use_mask)
            launch.set_param_by_name("seed", seed)
            launch.launch()
            if self.sensor_dr:
                self._launch_sensor_dr(mask, use_mask)
            if self.artefact_dr:
                self._launch_artefact_dr(mask, use_mask)

    def _launch_sensor_dr(self, mask, use_mask) -> None:
        """Redraw the respawned robots' field of view, range noise, pose-noise scale and
        localisation bias (``sensor_dr``)."""
        w = self.world
        scalar = w.wp_dtype
        seed = wp.int32(w.next_kernel_seed())
        bounds = (
            0.5 * self.lidar_fov[0], 0.5 * self.lidar_fov[1], *self.lidar_noise,
            *self.pose_noise, *self.pose_bias_std,
        )
        outs = [self._half_fov_wp, self._noise_std_wp, *self._pose_dr_wp]
        launch = self._dr_launch.get(
            concrete(pusht_lidar_dr_reset_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[mask, use_mask, seed, *(scalar(x) for x in bounds)],
            outputs=outs,
            device=w.device,
            key=(w.n_envs, ptr_key(mask), *bounds, *(ptr_key(o) for o in outs)),
        )
        launch.set_param_by_name("use_mask", use_mask)
        launch.set_param_by_name("seed", seed)
        launch.launch()

    def _launch_artefact_dr(self, mask, use_mask) -> None:
        """Redraw the respawned robots' artefact values (``artefact_dr``)."""
        w = self.world
        scalar = w.wp_dtype
        seed = wp.int32(w.next_kernel_seed())
        bounds = (
            *self.lidar_dropout, *self.lidar_spurious, self.lidar_range_scale,
            self.lidar_range_bias, self.lidar_yaw_bias, *self.lidar_stamp_phase,
        )
        lid = self.lidar
        yaw = lid.wp_yaw_bias if self.lidar_sweep else self._dummy_2d()
        phase = lid.wp_stamp_phase if self.lidar_sweep else yaw
        outs = [self._lidar_art_wp, yaw, phase]
        launch = self._art_launch.get(
            concrete(lidar_artefact_reset_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[mask, use_mask, seed, *(scalar(x) for x in bounds),
                    wp.int32(int(self.lidar_sweep))],
            outputs=outs,
            device=w.device,
            key=(w.n_envs, ptr_key(mask), *bounds, *(ptr_key(o) for o in outs)),
        )
        launch.set_param_by_name("use_mask", use_mask)
        launch.set_param_by_name("seed", seed)
        launch.launch()

    def _dummy_3d(self) -> wp.array:
        """A ``[1, 1, 1]`` scalar array for a kernel slot a config leaves unused."""
        if getattr(self, "_dummy3", None) is None:
            self._dummy3 = wp.zeros((1, 1, 1), dtype=self.world.wp_dtype,
                                    device=self.world.device)
        return self._dummy3

    def _dummy_2d(self) -> wp.array:
        """A ``[1, 1]`` scalar array for a kernel slot a config leaves unused."""
        if getattr(self, "_dummy2", None) is None:
            self._dummy2 = wp.zeros((1, 1), dtype=self.world.wp_dtype, device=self.world.device)
        return self._dummy2

    def _setup_critic_dr(self, world: World) -> None:
        """The randomised-parameter block's column scaling and the critic kernel's fixed
        inputs (see :meth:`critic_state`), once, outside any capture."""
        st = world.stepper
        dev, wdt = world.device, world.wp_dtype
        env: list[tuple[float, float]] = []
        agent: list[tuple[float, float]] = []

        def mid_half(lo: float, hi: float) -> tuple[float, float]:
            return 0.5 * (lo + hi), 0.5 * (hi - lo)

        if self.friction_dr:
            env.append(mid_half(*self.floor_friction))
        act_cols = []
        for col, (lo, hi) in zip(
            (P_MAX_ACCEL, P_MAX_ANG_ACCEL, P_CMD_GAIN_V, P_CMD_GAIN_W),
            self._actuation_ranges, strict=True,
        ):
            if lo < hi:
                act_cols.append(col)
                agent.append(mid_half(lo, hi))
        self._dr_delay = self.cmd_latency[0] < self.cmd_latency[1]
        if self._dr_delay:
            agent.append(mid_half(*(self.cmd_delay_substeps or (0, 0))))
        delay = st.cmd_delay
        if delay is None:  # a latency range that rounds to no substep: always 0
            delay = wp.zeros((world.n_envs, self.n_agents), dtype=wp.int32, device=dev)
        self._dr_noise = self.lidar_noise[0] < self.lidar_noise[1]
        if self._dr_noise:
            agent.append(mid_half(*self.lidar_noise))
        self._dr_scale = self.pose_noise_on and self.pose_noise[0] < self.pose_noise[1]
        if self._dr_scale:
            agent.append(mid_half(*self.pose_noise))
        if self.pose_noise_on:
            # Zero-mean Gaussians: two deviations at the top of the scale span [-1, 1].
            u = self.pose_noise[1]
            bp, bt = self.pose_bias_std
            agent += [(0.0, 2.0 * u * bp)] * 2 + [(0.0, 2.0 * u * bt)]
            if self.pose_drift_on:
                dp, dt_ = self.pose_drift_std
                agent += [(0.0, 2.0 * u * dp)] * 2 + [(0.0, 2.0 * u * dt_)]
        art_cols = [c for c, _, _ in self._art_columns()]
        agent += [(m, h) for _, m, h in self._art_columns()]
        if self._sweep_dr:
            agent += [(0.0, self.lidar_yaw_bias), mid_half(*self.lidar_stamp_phase)]
        assert (len(env), len(agent)) == (self.dr_env_dim, self.dr_agent_dim)
        cols = env + agent
        mid = torch.tensor([m for m, _ in cols] or [0.0], dtype=torch.float64)
        half = torch.tensor([h for _, h in cols] or [0.0], dtype=torch.float64)
        inv = torch.where(half > 0.0, 1.0 / half.clamp(min=1e-30), torch.zeros_like(half))
        tt = {"device": world.device, "dtype": world.dtype}
        #: Per column of the block: ``(x - mid) * inv``.
        self._dr_mid, self._dr_inv = mid.to(**tt), inv.to(**tt)
        self._dr_mid_wp = wp.from_torch(self._dr_mid)
        self._dr_inv_wp = wp.from_torch(self._dr_inv)
        self._dr_act_cols = act_cols
        self._dr_art_cols = art_cols
        self._dr_act_wp = wp.array(act_cols or [-1], dtype=wp.int32, device=dev)
        self._dr_art_wp = wp.array(art_cols or [-1], dtype=wp.int32, device=dev)
        d2 = self._dummy_2d()
        self._dr_inputs = [
            wp.int32(int(self.friction_dr)),
            st.body_friction if self.friction_dr else wp.zeros(1, dtype=wdt, device=dev),
            wdt(1.0 / world.config.gravity) if self.friction_dr else wdt(0.0),
            self._dr_act_wp,
            st.params.floats_per_env if act_cols else wp.zeros((1, 1, 1), dtype=wdt, device=dev),
            wp.int32(int(self._dr_delay)),
            delay if self._dr_delay else wp.zeros((1, 1), dtype=wp.int32, device=dev),
            wp.int32(int(self._dr_noise)),
            self._noise_std_wp if self._dr_noise else d2,
            wp.int32(int(self.pose_noise_on)),
            wp.int32(int(self._dr_scale)),
            *(self._pose_dr_wp if self.pose_noise_on
              else [d2, wp.zeros((1, 1), dtype=VEC2[wdt], device=dev), d2]),
            wp.int32(int(self.pose_drift_on)),
        ]

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        """Push-T's body step on the torch path, then the lidar observation and reward."""
        super().post_step_torch()
        self._refresh_lidar(None, reset=False)

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        super().reset_torch(env_mask)
        self._prev_standoff = None
        self._refresh_lidar(env_mask, reset=True)

    def _sensor_torch(
        self, ranges: torch.Tensor, hit_class: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The torch twin of :func:`lidar_sensor_kernel`: the blind zone and field of view
        exactly, the range noise from torch's generator (so a noisy scan is not bit-equal
        across the two paths)."""
        th = self.world.state.theta.unsqueeze(-1)  # [E, A, 1]
        ang = self._ray_angles()
        hx = ranges * torch.cos(ang) - self.lidar_mount_offset * torch.cos(th)
        hy = ranges * torch.sin(ang) - self.lidar_mount_offset * torch.sin(th)
        off = ang - th
        if self.lidar_fov_at_mount:
            off = torch.atan2(hy, hx) - th
        elif self.lidar.body_frame:
            off = self.lidar.ray_offsets(device=ang.device, dtype=ang.dtype).expand_as(ang)
        rel = torch.atan2(torch.sin(off), torch.cos(off))
        half = self._half_fov.unsqueeze(-1) + FOV_EPS
        blind = (hit_class != 0) & (
            (hx * hx + hy * hy < self.lidar_min_range**2) | (rel.abs() > half)
        )
        mr = self.lidar_max_range
        ranges = torch.where(blind, torch.full_like(ranges, mr), ranges)
        hit_class = torch.where(blind, torch.zeros_like(hit_class), hit_class)
        if self.lidar_noise[1] > 0.0:
            sigma = self._noise_std.unsqueeze(-1)
            noisy = (ranges + sigma * torch.randn_like(ranges)).clamp(0.0, mr)
            ranges = torch.where(hit_class != 0, noisy, ranges)
        return ranges, hit_class

    def _artefacts_torch(
        self, ranges: torch.Tensor, hit_class: torch.Tensor, hit_index: torch.Tensor,
        centres: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The torch twin of :func:`~swarp.scenarios.pusht_lidar_artefacts.
        lidar_artefact_kernel`, its draws from torch's generator. ``centres`` ``[E, B, 2]``
        are the T's box centres (every box at the T's heading)."""
        mr = self.lidar_max_range
        art = self._lidar_art
        ang = self._ray_angles()
        dx, dy = torch.cos(ang), torch.sin(ang)
        p = self.world.state.pos.unsqueeze(2)  # [E, A, 1, 2]
        lost = torch.zeros_like(hit_class, dtype=torch.bool)
        if self.lidar_grazing_cos > 0.0:
            idx = hit_index.long().clamp(min=0)  # [E, A, R]
            e_idx = torch.arange(idx.shape[0], device=idx.device).view(-1, 1, 1)
            cen = centres[e_idx, idx]  # [E, A, R, 2]
            half = self._obs_half[idx]
            th = self.tee_theta.view(-1, 1, 1)
            co, so = torch.cos(th), torch.sin(th)
            rx = p[..., 0] + ranges * dx - cen[..., 0]
            ry = p[..., 1] + ranges * dy - cen[..., 1]
            lx = co * rx + so * ry
            ly = -so * rx + co * ry
            face_x = lx.abs() * half[..., 1] > ly.abs() * half[..., 0]
            cos_i = torch.where(face_x, (co * dx + so * dy).abs(), (-so * dx + co * dy).abs())
            p_lost = 1.0 - cos_i / self.lidar_grazing_cos
            lost = (hit_class == HIT_MOVABLE) & (torch.rand_like(ranges) < p_lost)
        pd = art[..., ART_DROPOUT].unsqueeze(-1)
        lost = lost | ((hit_class != 0) & (torch.rand_like(ranges) < pd))
        ranges = torch.where(lost, torch.full_like(ranges, mr), ranges)
        hit_class = torch.where(lost, torch.zeros_like(hit_class), hit_class)
        smin = self.agent_radius
        ps = art[..., ART_SPURIOUS].unsqueeze(-1)
        spur = (torch.rand_like(ranges) < ps) & (ranges > smin)
        ranges = torch.where(spur, smin + (ranges - smin) * torch.rand_like(ranges), ranges)
        hit_class = torch.where(spur, torch.full_like(hit_class, HIT_MOVABLE), hit_class)
        kappa = art[..., ART_SCALE].unsqueeze(-1)
        beta = art[..., ART_BIAS].unsqueeze(-1)
        scaled = (ranges * (1.0 + kappa) + beta).clamp(0.0, mr)
        return torch.where(hit_class != 0, scaled, ranges), hit_class

    def _relabel_torch(
        self, ranges: torch.Tensor, hit_class: torch.Tensor, opos: torch.Tensor,
        oth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The torch twin of :func:`~swarp.scenarios.pusht_lidar_artefacts.
        lidar_relabel_kernel`: hits classed from the reported poses ``opos``/``oth``."""
        w = self.world
        na = self.n_agents
        off = self.lidar.ray_offsets(device=ranges.device, dtype=ranges.dtype)
        base = oth if self.lidar.body_frame else torch.zeros_like(oth)
        ang = base.unsqueeze(-1) + off
        hit = opos.unsqueeze(2) + ranges.unsqueeze(-1) * torch.stack(
            [torch.cos(ang), torch.sin(ang)], dim=-1
        )  # [E, A, R, 2]
        out = (hit.abs() > self.label_bound).any(-1)
        rel = hit.unsqueeze(3) - opos.view(-1, 1, 1, na, 2)  # [E, A, R, J, 2]
        cj = torch.cos(oth).view(-1, 1, 1, na)
        sj = torch.sin(oth).view(-1, 1, 1, na)
        lx = cj * rel[..., 0] + sj * rel[..., 1]
        ly = -sj * rel[..., 0] + cj * rel[..., 1]
        m = self.lidar_label_margin
        hx, hy = w.agent_half_extents[:, 0], w.agent_half_extents[:, 1]
        in_box = (lx.abs() <= hx + m) & (ly.abs() <= hy + m)
        in_disc = rel.norm(dim=-1) <= w.agent_radius + m
        is_box = w.agent_shape == int(AgentShape.BOX)
        inside = torch.where(is_box, in_box, in_disc)
        eye = torch.eye(na, dtype=torch.bool, device=ranges.device).view(1, na, 1, na)
        mate = (inside & ~eye).any(-1)
        hit_any = hit_class != 0
        cls = torch.where(mate, HIT_AGENT, HIT_MOVABLE)
        cls = torch.where(hit_any & ~out, cls, 0).to(hit_class.dtype)
        ranges = torch.where(hit_any & out, torch.full_like(ranges, self.lidar_max_range), ranges)
        return ranges, cls

    def _critic_dr_torch(self, perm: torch.Tensor) -> torch.Tensor:
        """``[E, dr_env_dim + dr_agent_dim A]``: the randomised parameters, normalised
        and role-sorted (the torch twin of :func:`~swarp.scenarios.pusht_lidar_artefacts.
        pusht_lidar_critic_dr_kernel`)."""
        w = self.world
        st = w.stepper
        dt = w.state.pos.dtype
        env, agent = [], []
        if self.friction_dr:
            env.append(wp.to_torch(st.body_friction) / w.config.gravity)
        if self._dr_act_cols:
            params = wp.to_torch(st.params.floats_per_env)
            agent += [params[..., c] for c in self._dr_act_cols]
        if self._dr_delay:
            delay = st.cmd_delay
            agent.append(
                torch.zeros_like(self.roles, dtype=dt) if delay is None
                else wp.to_torch(delay).to(dt)
            )
        if self._dr_noise:
            agent.append(self._noise_std)
        if self._dr_scale:
            agent.append(self._pose_scale)
        if self.pose_noise_on:
            agent += [self._bias_pos[..., 0], self._bias_pos[..., 1], self._bias_theta]
            if self.pose_drift_on:
                agent += [self._drift_state[..., 5 + i] for i in range(3)]
        agent += [self._lidar_art[..., c] for c in self._dr_art_cols]
        if self._sweep_dr:
            agent += [self.lidar.yaw_bias, self.lidar.stamp_phase]
        ne = self.dr_env_dim
        out = []
        if env:
            out.append((torch.stack(env, -1) - self._dr_mid[:ne]) * self._dr_inv[:ne])
        if agent:
            block = (torch.stack(agent, -1) - self._dr_mid[ne:]) * self._dr_inv[ne:]
            block = block.gather(1, perm.unsqueeze(-1).expand(-1, -1, block.shape[-1]))
            out.append(block.flatten(1))
        return torch.cat(out, dim=-1)

    def _observed_pose(
        self, env_mask: torch.Tensor | None, *, reset: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The torch twin of :func:`pose_obs_noise_kernel`: the reported position, velocity
        and heading, with the jitter from torch's generator. A reset pass keeps the rows
        of the envs it did not respawn."""
        s = self.world.state
        u = self._pose_scale
        jp, jt = self.pose_jitter_std
        n = torch.randn(*u.shape, 5, device=u.device, dtype=u.dtype)
        dp = torch.zeros_like(s.pos)
        dtd = torch.zeros_like(u)
        if self.pose_drift_on:
            dp, dtd = self._drift_torch(env_mask, reset=reset)
        opos = s.pos + self._bias_pos + dp + (u * jp).unsqueeze(-1) * n[..., :2]
        dth = self._bias_theta + dtd + u * jt * n[..., 2]
        c, sn = torch.cos(dth), torch.sin(dth)
        v = s.vel
        ovel = torch.stack(
            [c * v[..., 0] - sn * v[..., 1], sn * v[..., 0] + c * v[..., 1]], dim=-1
        ) + (u * self.vel_noise_std).unsqueeze(-1) * n[..., 3:]
        oth = s.theta + dth
        if reset and env_mask is not None:
            m = env_mask.view(-1, 1)
            opos = torch.where(m.unsqueeze(-1), opos, self._obs_pos)
            ovel = torch.where(m.unsqueeze(-1), ovel, self._obs_vel)
            oth = torch.where(m, oth, self._obs_theta)
        self._obs_pos, self._obs_vel, self._obs_theta = opos, ovel, oth
        return opos, ovel, oth

    def _drift_torch(
        self, env_mask: torch.Tensor | None, *, reset: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The drift of :func:`pose_obs_noise_kernel` on the torch path: ``[E, A, 2]``
        position and ``[E, A]`` heading, advancing ``_drift_state``."""
        s = self.world.state
        u = self._pose_scale
        prev = self._drift_state
        p, th = s.pos.detach(), s.theta.detach()
        turn = torch.atan2(torch.sin(th - prev[..., 2]), torch.cos(th - prev[..., 2]))
        thm = prev[..., 2] + 0.5 * turn
        ds = 0.5 * (prev[..., 3] + s.speed) * self.dt
        slip = (p - prev[..., :2]) - ds.unsqueeze(-1) * torch.stack(
            [thm.cos(), thm.sin()], dim=-1
        )
        slip_th = turn - 0.5 * (prev[..., 4] + s.ang_vel) * self.dt
        a, at = (math.exp(-self.dt / t) for t in self.pose_drift_tau)
        kick, kick_t = math.sqrt(1.0 - a * a), math.sqrt(1.0 - at * at)
        g = u * self.pose_slip_gain[0]
        nd = torch.randn(*u.shape, 3, device=u.device, dtype=u.dtype)
        sp, st = self.pose_drift_std
        dp = a * prev[..., 5:7] - g.unsqueeze(-1) * slip + (u * sp * kick).unsqueeze(-1) * nd[
            ..., :2
        ]
        dtd = at * prev[..., 7] - u * self.pose_slip_gain[1] * slip_th + u * st * kick_t * nd[
            ..., 2
        ]
        fresh = torch.ones_like(u, dtype=torch.bool) if reset else torch.zeros_like(
            u, dtype=torch.bool
        )
        if reset and env_mask is not None:
            fresh = env_mask.view(-1, 1).expand_as(u)
        dp = torch.where(fresh.unsqueeze(-1), torch.zeros_like(dp), dp)
        dtd = torch.where(fresh, torch.zeros_like(dtd), dtd)
        new = torch.cat(
            [p, th.unsqueeze(-1), s.speed.unsqueeze(-1), s.ang_vel.unsqueeze(-1), dp,
             dtd.unsqueeze(-1)], dim=-1,
        )
        if reset and env_mask is not None:
            new = torch.where(env_mask.view(-1, 1, 1), new, prev)
            dp, dtd = new[..., 5:7], new[..., 7]
        self._drift_state = new.contiguous()
        return dp, dtd

    def _ray_angles(self) -> torch.Tensor:
        """``[E, A, R]`` world-frame ray angles, as :func:`lidar_scan` builds them."""
        s = self.world.state
        offsets = self.lidar.ray_offsets(device=s.pos.device, dtype=s.pos.dtype)
        base = s.theta if self.lidar.body_frame else torch.zeros_like(s.theta)
        return base.unsqueeze(-1) + offsets

    def _refresh_lidar(self, env_mask: torch.Tensor | None, *, reset: bool) -> None:
        """The torch oracle of the fused obs and reward kernels, after Push-T's refresh.

        Scans with :func:`lidar_scan` against the T's boxes rebuilt in torch from the body
        pose (differentiable in it), and recomputes every observation block, the trainer
        buffers and the seven reward terms from that scan.
        """
        w = self.world
        s = w.state
        pos, vel = s.pos, s.vel
        ne, na, nr = w.n_envs, self.n_agents, self.n_rays
        dt = pos.dtype
        mr = self.lidar_max_range
        roles = self.roles
        is_obs = roles == 1  # [E, A]
        sgn = is_obs.to(dt) * 2.0 - 1.0

        # --- the T's boxes from the body pose, and the scan
        th = self.tee_theta
        ca, sa = torch.cos(th), torch.sin(th)
        ox, oy = self._box_off_t[:, 0], self._box_off_t[:, 1]
        centres = self.tee_pos.unsqueeze(1) + torch.stack(
            [
                ca.unsqueeze(1) * ox - sa.unsqueeze(1) * oy,
                sa.unsqueeze(1) * ox + ca.unsqueeze(1) * oy,
            ],
            dim=-1,
        )  # [E, B, 2]
        lid = self.lidar
        sweep = None
        if lid.sweep:
            sweep = (s.vel, s.ang_vel, lid.yaw_bias, lid.stamp_phase, lid.scan_period,
                     lid.sweep_dir)
        ranges, hit_class, hit_index = lidar_scan(
            pos,
            s.theta,
            w.agent_radius,
            nr,
            mr,
            body_frame=self.lidar.body_frame,
            angle_start=self.lidar.angle_start,
            angle_end=self.lidar.angle_end,
            sweep=sweep,
            obstacle_pos=centres,
            obstacle_radius=self._obs_radius,
            obstacle_shape=self._obs_shape,
            obstacle_angle=th.unsqueeze(1).expand(-1, self.n_boxes),
            obstacle_half=self._obs_half,
            obstacle_kind=self._obs_kind,
            agent_shape=w.agent_shape if self.agent_shape == AgentShape.BOX else None,
            agent_half=w.agent_half_extents,
            return_hits=True,
        )
        if self.lidar_artefacts:
            ranges, hit_class = self._artefacts_torch(ranges, hit_class, hit_index, centres)
        if self.sensor_model:
            ranges, hit_class = self._sensor_torch(ranges, hit_class)
        opos, ovel, oth = pos, vel, s.theta
        if self.pose_noise_on:
            opos, ovel, oth = self._observed_pose(env_mask, reset=reset)
        if self.lidar_label_from_pose:
            ranges, hit_class = self._relabel_torch(ranges, hit_class, opos, oth)
        is_obj = hit_class == HIT_MOVABLE  # [E, A, R]
        is_ag = hit_class == HIT_AGENT
        ang = self._ray_angles()  # [E, A, R]
        dirs = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
        hits = pos.unsqueeze(2) + ranges.unsqueeze(-1) * dirs  # [E, A, R, 2] world

        # --- observation blocks, the self and teammate ones from the reported pose
        q = self.tee_pos
        g = self.goal_pos
        gth = self.goal_theta.unsqueeze(1).expand(-1, na)
        feats = [
            opos,
            ovel,
            sgn.unsqueeze(-1),
            g.unsqueeze(1) - opos,
            torch.stack([gth.cos(), gth.sin()], dim=-1),
        ]
        if self.heading_obs:
            feats.append(torch.stack([oth.cos(), oth.sin()], dim=-1))
        others = self._others  # [A, A-1]
        mates = torch.cat(
            [opos[:, others] - opos.unsqueeze(2), sgn[:, others].unsqueeze(-1)], dim=-1
        )  # [E, A, A-1, 3]
        feats.append(mates.flatten(2))
        if self.obs_level == 0:
            err = (self.tee_theta - self.goal_theta).unsqueeze(1).expand(-1, na)
            tth = self.tee_theta.unsqueeze(1).expand(-1, na)
            feats += [
                q.unsqueeze(1) - pos,
                (g - q).unsqueeze(1).expand(-1, na, -1),
                torch.stack([tth.cos(), tth.sin(), err.cos(), err.sin()], dim=-1),
            ]
        frame = torch.cat([ranges / mr, is_obj.to(dt), is_ag.to(dt)], dim=-1)  # [E, A, 3R]
        nf, width = self.frame_stack, 3 * nr
        ring = self._ring.view(ne, na, nf, width)
        fresh = frame.detach().unsqueeze(2)
        if reset:
            m = (
                torch.ones(ne, dtype=torch.bool, device=pos.device)
                if env_mask is None
                else env_mask
            ).view(-1, 1, 1, 1)
            new = torch.where(m, torch.cat([torch.zeros_like(ring[:, :, 1:]), fresh], 2), ring)
        else:
            new = torch.cat([ring[:, :, 1:], fresh], dim=2)
        self._ring = new.reshape(ne, na, nf * width).contiguous()
        # The newest frame from the live scan, so gradients reach it; older ones are data.
        feats.append(torch.cat([new[:, :, :-1].flatten(2), frame], dim=-1))
        if self.obs_level == 2:
            feats.append(self._pooled_hits(pos, hits, is_obj))
        obs = torch.cat(feats, dim=-1)
        if reset and env_mask is not None and self._lcache is not None:
            # An env the reset did not select is unchanged, so its row is the one the step
            # just computed. Keeping that tensor keeps its gradient: the auto-reset pass
            # runs under no_grad after every taped step, hence the local enable_grad.
            with torch.enable_grad():
                obs = torch.where(env_mask.view(-1, 1, 1), obs, self._lcache["obs"])

        # --- per-agent stats from the agent's own rays
        dq = q.unsqueeze(1) - pos  # [E, A, 2]
        dist = dq.norm(dim=-1)
        bearing = torch.atan2(dq[..., 1], dq[..., 0])
        r_t = self.tee_radius
        half = torch.where(
            dist > r_t,
            torch.asin((r_t / dist.clamp(min=1e-12)).clamp(max=1.0)),
            torch.full_like(dist, 4.0),
        )
        dang = ang - bearing.unsqueeze(-1)
        in_cone = torch.atan2(torch.sin(dang), torch.cos(dang)).abs() <= half.unsqueeze(-1)
        n_cone = in_cone.sum(-1)
        occl = (in_cone & is_ag).sum(-1).to(dt) / n_cone.clamp(min=1).to(dt)
        n_obj = is_obj.sum(-1).to(dt)
        rel = hits - q.view(ne, 1, 1, 2)
        cth, sth = ca.view(ne, 1, 1), sa.view(ne, 1, 1)
        lx = cth * rel[..., 0] + sth * rel[..., 1]
        ly = -sth * rel[..., 0] + cth * rel[..., 1]
        sec = torch.floor((torch.atan2(ly, lx) + math.pi) / (math.pi / 4.0))
        sec = sec.clamp(0, N_SECTORS - 1).long()
        onehot = torch.nn.functional.one_hot(sec, N_SECTORS).bool() & is_obj.unsqueeze(-1)
        sectors = onehot.any(dim=2)  # [E, A, 8]
        near = (is_obj & (ranges < self.contact_dist)).any(-1)

        # --- rewards
        c = self._cache  # Push-T's refresh, just run by super()
        on_goal = c["on_goal"]
        failed = torch.zeros_like(on_goal)
        if self.oob_limit > 0.0:
            out = (q.detach().abs() > self.oob_limit).any(-1)
            if reset:
                fresh = torch.ones_like(out) if env_mask is None else env_mask
                failed = torch.where(fresh, torch.zeros_like(out), self._oob.bool())
            else:
                failed = self._oob.bool() | out
            self._oob = failed.to(torch.uint8)
            on_goal = on_goal & ~failed
        team = (c["shaping"] + self.goal_reward * on_goal.to(dt)).unsqueeze(1).expand(-1, na)
        team = torch.where(failed.unsqueeze(1), torch.full_like(team, -self.oob_penalty), team)
        zero = torch.zeros_like(team)
        approach = torch.where(is_obs, zero, c["agent_shaping"])
        serr = (dist - self.standoff_dist).abs()
        if self._prev_standoff is None:
            self._prev_standoff = serr.detach().clone()
        ssh = (self._prev_standoff - serr) * self.standoff_shaping
        if reset:
            if env_mask is None:
                ssh = torch.zeros_like(ssh)
                self._prev_standoff = serr.detach().clone()
            else:
                m2 = env_mask.view(-1, 1)
                ssh = torch.where(m2, torch.zeros_like(ssh), ssh)
                self._prev_standoff = torch.where(m2, serr.detach(), self._prev_standoff)
        else:
            self._prev_standoff = serr.detach().clone()
        cov = (sectors & is_obs.unsqueeze(-1)).any(dim=1).sum(-1).to(dt) / N_SECTORS  # [E]
        vis = (n_obj / (self.visibility_target * nr)).clamp(max=1.0)
        standoff = torch.where(is_obs, ssh, zero)
        vis_t = torch.where(is_obs, -self.visibility_weight * (1.0 - vis), zero)
        cov_t = torch.where(
            is_obs, (-self.coverage_weight * (1.0 - cov)).unsqueeze(1).expand(-1, na), zero
        )
        occ_t = torch.where(is_obs, -self.occlusion_weight * occl, zero)
        con_t = torch.where(is_obs & near, torch.full_like(zero, -self.contact_penalty), zero)
        if self.oob_limit > 0.0:
            f = failed.unsqueeze(1)
            approach, standoff, vis_t, cov_t, occ_t, con_t = (
                torch.where(f, zero, t) for t in (approach, standoff, vis_t, cov_t, occ_t, con_t)
            )
        terms = (team, approach, standoff, vis_t, cov_t, occ_t, con_t)
        reward = team + approach + standoff + vis_t + cov_t + occ_t + con_t
        n_obs = is_obs.sum(-1).clamp(min=1).to(dt)

        # --- trainer buffers
        head = torch.cat(
            [
                q,
                g - q,
                torch.stack([ca, sa], dim=-1),
                torch.stack(
                    [
                        torch.cos(self.tee_theta - self.goal_theta),
                        torch.sin(self.tee_theta - self.goal_theta),
                    ],
                    dim=-1,
                ),
                self.tee_vel,
                self.tee_ang_vel.unsqueeze(-1),
                g,
                torch.stack([self.goal_theta.cos(), self.goal_theta.sin()], dim=-1),
            ],
            dim=-1,
        )  # [E, 15]
        blocks = [pos, vel, sgn.unsqueeze(-1)]
        if self.heading_obs:
            blocks.append(torch.stack([s.theta.cos(), s.theta.sin()], dim=-1))
        per_agent = torch.cat(blocks, dim=-1)  # [E, A, state_agent_dim]
        perm = torch.argsort(roles, dim=-1, stable=True)
        sorted_agents = per_agent.gather(
            1, perm.unsqueeze(-1).expand(-1, -1, self.state_agent_dim)
        )
        critic = torch.cat([head, sorted_agents.flatten(1)], dim=-1)
        if self.critic_dr:
            critic = torch.cat([critic, self._critic_dr_torch(perm)], dim=-1)
        aux = torch.cat(
            [dq * self.aux_scale, torch.stack([ca, sa], -1).unsqueeze(1).expand(-1, na, -1)],
            dim=-1,
        )

        self._lcache = {
            "done": on_goal,
            "out_of_bounds": failed,
            "obs": obs,
            "reward": reward,
            "multiobj": torch.stack(terms, dim=-1),
            "team_coverage": cov,
            "observer_visibility": (vis * is_obs).sum(-1) / n_obs,
            "occlusion": (occl * is_obs).sum(-1) / n_obs,
            "tee_seen": n_obj > 0,
            "critic": critic,
            "aux": aux,
            "ranges": ranges,
            "hit_class": hit_class,
        }

    def _pooled_hits(self, pos, hits, is_obj) -> torch.Tensor:
        """``[E, A, 3 A K]``: every agent's K evenly spread object hits, relative to each
        agent, with a valid flag (torch oracle of the obs kernel's level-2 block)."""
        na, kk = self.n_agents, self.n_comm_pts
        dt = pos.dtype
        nh = is_obj.sum(-1)  # [E, J]
        rank = is_obj.long().cumsum(-1) - 1  # [E, J, R]: hit number of each object ray
        slot = torch.arange(kk, device=pos.device)
        target = ((2 * slot + 1) * nh.unsqueeze(-1)) // (2 * kk)  # [E, J, K]
        match = is_obj.unsqueeze(2) & (rank.unsqueeze(2) == target.unsqueeze(-1))  # [E,J,K,R]
        pts = (match.to(dt).unsqueeze(-1) * hits.unsqueeze(2)).sum(-2)  # [E, J, K, 2]
        rel = pts.unsqueeze(1) - pos.view(pos.shape[0], na, 1, 1, 2)  # [E, A, J, K, 2]
        d_aj = (pos.unsqueeze(1) - pos.unsqueeze(2)).norm(dim=-1)  # [E, A, J]
        eye = torch.eye(na, dtype=torch.bool, device=pos.device)
        in_range = (d_aj <= self.comm_range) | eye
        valid = ((nh > 0).unsqueeze(1) & in_range).unsqueeze(-1).expand(-1, -1, -1, kk)
        vf = valid.to(dt).unsqueeze(-1)
        return torch.cat([rel * vf, vf], dim=-1).flatten(2)

    # --------------------------------------------------------- fused fast path

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        """Push-T's spec plus the lidar ring, the per-agent ray statistics, the trainer
        buffers and the observer diagnostics."""
        ne, na = n_envs, self.n_agents
        return (
            *super().fused_spec(n_envs),
            Buf(
                "ring",
                (ne, na, 3 * self.n_rays * self.frame_stack),
                attr="_ring",
                alloc="never",
                carry=True,
                watch=True,
            ),
            Buf(
                "prev_standoff",
                (ne, na),
                attr="_prev_standoff",
                alloc="if_none",
                carry=True,
                watch=True,
            ),
            Buf("critic", (ne, self.state_dim)),
            Buf("aux", (ne, na, self.aux_dim)),
            Buf("n_obj", (ne, na)),
            Buf("occl", (ne, na)),
            Buf("sectors", (ne, na), "uint8"),
            Buf("near", (ne, na), "uint8"),
            Buf("multiobj", (ne, na, N_OBJ)),
            Buf("coverage", (ne,)),
            Buf("visibility", (ne,)),
            Buf("occlusion", (ne,)),
            Buf("tee_seen", (ne, na), "uint8", bool_view=True),
            *(
                (Buf("oob", (ne,), "uint8", attr="_oob", alloc="never", carry=True,
                     watch=True),)
                if self.oob_limit > 0.0
                else ()
            ),
            *(
                (Buf("drift_state", (ne, na, 8), attr="_drift_state", alloc="never",
                     carry=True, watch=True),)
                if self.pose_drift_on
                else ()
            ),
        )

    def launch_fused(self, pass_: FusedPass) -> None:
        """Body sync (a step only), the lidar, the observation kernel, the reward kernel.

        The reward kernel runs on every pass, as Push-T's does: it owns the shaping
        carries, which a reset rebases, and writes its outputs only on a full pass.
        """
        st = self.world.state_wp()
        if pass_.is_step:
            self._launch_body_sync()
        self.lidar.launch(self.world, st)
        if self.lidar_artefacts:
            self._launch_artefacts(st)
        if self.sensor_model:
            self._launch_sensor(st)
        if self.pose_noise_on:
            self._launch_pose_noise(st, pass_.advance_prev)
        if self.lidar_label_from_pose:
            self._launch_relabel(st)
        self._launch_obs(st, pass_.advance_prev)
        if self.critic_dr:
            self._launch_critic_dr()
        self._launch_reward(st, pass_.advance_prev, pass_.full_pass)

    def _launch_artefacts(self, st) -> None:
        """The scan artefacts over the lidar's own buffers, in place."""
        w = self.world
        scalar = w.wp_dtype
        lid = self.lidar
        sp = w.stepper
        wp.launch(
            concrete(lidar_artefact_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.theta,
                scalar(lid.max_range),
                scalar(lid.ray_step),
                scalar(lid.angle_start),
                wp.int32(int(lid.body_frame)),
                sp.obs_pos,
                sp.obs_angle,
                sp.obs_half,
                scalar(self.lidar_grazing_cos),
                scalar(self.agent_radius),
                self._lidar_art_wp,
                self._art_rng,
                lid.wp_ranges,
                lid.wp_hit_class,
                lid.wp_hit_index,
            ],
            device=w.device,
            record_tape=False,
        )

    def _launch_relabel(self, st) -> None:
        """The deployment's hit classes from the reported poses, in place."""
        w = self.world
        scalar = w.wp_dtype
        lid = self.lidar
        seen = self._obs_pose_wp if self.pose_noise_on else [st.pos, st.vel, st.theta]
        wp.launch(
            concrete(lidar_relabel_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                seen[0],
                seen[2],
                w.stepper.params.floats,
                scalar(lid.max_range),
                scalar(lid.ray_step),
                scalar(lid.angle_start),
                wp.int32(int(lid.body_frame)),
                scalar(min(self.label_bound, 1e30)),
                scalar(self.lidar_label_margin),
                lid.wp_ranges,
                lid.wp_hit_class,
            ],
            device=w.device,
            record_tape=False,
        )

    def _launch_critic_dr(self) -> None:
        """The randomised parameters into the critic state's tail."""
        w = self.world
        scalar = w.wp_dtype
        d2 = self._dummy_2d()
        art = self._lidar_art_wp if self.artefact_dr else self._dummy_3d()
        lid = self.lidar
        wp.launch(
            concrete(pusht_lidar_critic_dr_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                self._roles_wp,
                wp.int32(self.n_pushers),
                wp.int32(self.state_base_dim),
                wp.int32(self.dr_env_dim),
                wp.int32(self.dr_agent_dim),
                *self._dr_inputs,
                self._wp["drift_state"] if self.pose_drift_on else self._drift_wp,
                self._dr_art_wp,
                art,
                wp.int32(int(self._sweep_dr)),
                lid.wp_yaw_bias if self._sweep_dr else d2,
                lid.wp_stamp_phase if self._sweep_dr else d2,
                self._dr_mid_wp,
                self._dr_inv_wp,
                self._wp["critic"],
            ],
            device=w.device,
            record_tape=False,
        )

    def _launch_sensor(self, st) -> None:
        """The sensor model over the lidar's own scan buffers, in place."""
        w = self.world
        scalar = w.wp_dtype
        lid = self.lidar
        wp.launch(
            concrete(lidar_sensor_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.theta,
                scalar(lid.max_range),
                scalar(lid.ray_step),
                scalar(lid.angle_start),
                wp.int32(int(lid.body_frame)),
                scalar(self.lidar_mount_offset),
                scalar(self.lidar_min_range),
                self._half_fov_wp,
                wp.int32(int(self.lidar_fov_at_mount)),
                self._noise_std_wp,
                self._lidar_rng,
                lid.wp_ranges,
                lid.wp_hit_class,
            ],
            device=w.device,
            record_tape=False,
        )

    def _launch_pose_noise(self, st, advance_prev: int) -> None:
        """The reported poses (:func:`pose_obs_noise_kernel`), before the observation."""
        w = self.world
        scalar = w.wp_dtype
        wp.launch(
            concrete(pose_obs_noise_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                st.theta,
                st.speed,
                st.ang_vel,
                self._wp["resetmask"],
                wp.int32(advance_prev),
                *self._pose_dr_wp,
                scalar(self.pose_jitter_std[0]),
                scalar(self.pose_jitter_std[1]),
                scalar(self.vel_noise_std),
                wp.int32(int(self.pose_drift_on)),
                scalar(self.dt),
                *(scalar(math.exp(-self.dt / t) if self.pose_drift_on else 0.0)
                  for t in self.pose_drift_tau),
                scalar(self.pose_drift_std[0]),
                scalar(self.pose_drift_std[1]),
                scalar(self.pose_slip_gain[0]),
                scalar(self.pose_slip_gain[1]),
                self._wp["drift_state"] if self.pose_drift_on else self._drift_wp,
                self._pose_rng,
            ],
            outputs=self._obs_pose_wp,
            device=w.device,
            record_tape=False,
        )

    def _launch_obs(self, st, advance_prev: int = 1) -> None:
        w = self.world
        scalar = w.wp_dtype
        lid = self.lidar
        names = ("tee_pos", "tee_theta", "tee_vel", "tee_ang_vel", "goal_pos", "goal_theta")
        body = [self._wp[n] for n in names]
        outs = [
            self._wp[n]
            for n in ("obs", "ring", "critic", "aux", "n_obj", "occl", "sectors", "near")
        ]
        resetmask = self._wp["resetmask"]
        seen = self._obs_pose_wp if self.pose_noise_on else [st.pos, st.vel, st.theta]
        cfg = (
            self.n_pushers,
            self.obs_level,
            self.frame_stack,
            self.n_comm_pts,
            self.comm_range,
            lid.max_range,
            lid.ray_step,
            lid.angle_start,
            int(lid.body_frame),
            self.tee_radius,
            self.aux_scale,
            self.contact_dist,
            int(self.heading_obs),
        )
        launch = self._lidar_obs_launch.get(
            concrete(pusht_lidar_obs_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                st.theta,
                *seen,
                *body,
                self._roles_wp,
                lid.wp_ranges,
                lid.wp_hit_class,
                resetmask,
                wp.int32(self.n_pushers),
                wp.int32(self.obs_level),
                wp.int32(self.frame_stack),
                wp.int32(self.n_comm_pts),
                scalar(self.comm_range),
                scalar(lid.max_range),
                scalar(lid.ray_step),
                scalar(lid.angle_start),
                wp.int32(int(lid.body_frame)),
                scalar(self.tee_radius),
                scalar(self.aux_scale),
                scalar(self.contact_dist),
                wp.int32(int(self.heading_obs)),
                wp.int32(advance_prev),
            ],
            outputs=outs,
            device=w.device,
            key=(
                w.n_envs,
                self.n_agents,
                ptr_key(st.pos),
                ptr_key(st.vel),
                ptr_key(st.theta),
                *(ptr_key(x) for x in seen),
                *(ptr_key(b) for b in body),
                ptr_key(self._roles_wp),
                ptr_key(lid.wp_ranges),
                ptr_key(lid.wp_hit_class),
                ptr_key(resetmask),
                *cfg,
                *(ptr_key(o) for o in outs),
            ),
        )
        launch.set_param_by_name("advance_prev", wp.int32(advance_prev))
        launch.launch()

    def _launch_reward(self, st, advance_prev: int, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        names = ("tee_pos", "tee_theta", "goal_pos", "goal_theta")
        body = [self._wp[n] for n in names]
        stats = [self._wp[n] for n in ("n_obj", "occl", "sectors", "near")]
        outs = [
            self._wp[n]
            for n in (
                "prev_dist",
                "prev_ang",
                "prev_adist",
                "prev_standoff",
                "reward",
                "done",
                "dist",
                "ang",
                "multiobj",
                "coverage",
                "visibility",
                "occlusion",
                "tee_seen",
            )
        ]
        resetmask = self._wp["resetmask"]
        weights = (
            self.pos_shaping_factor,
            self.rot_shaping_factor,
            self.agent_dist_shaping,
            self.push_point_offset,
            self.joint_shaping,
            self.rot_away_penalty,
            self.goal_tolerance,
            self.angle_tolerance,
            self.goal_reward,
            self.standoff_dist,
            self.standoff_shaping,
            self.visibility_target * self.n_rays,
            self.visibility_weight,
            self.coverage_weight,
            self.occlusion_weight,
            self.contact_penalty,
            self.oob_limit,
            self.oob_penalty,
        )
        oob = self._wp["oob"] if self.oob_limit > 0.0 else self._oob_wp
        launch = self._lidar_reward_launch.get(
            concrete(pusht_lidar_reward_kernel, scalar),
            dim=w.n_envs,
            inputs=[
                st.pos,
                *body,
                resetmask,
                self._roles_wp,
                *stats,
                wp.int32(self.n_agents),
                *(scalar(x) for x in weights),
                wp.int32(advance_prev),
                wp.int32(full_pass),
                oob,
            ],
            outputs=outs,
            device=w.device,
            key=(
                w.n_envs,
                self.n_agents,
                ptr_key(st.pos),
                *(ptr_key(b) for b in body),
                ptr_key(resetmask),
                ptr_key(self._roles_wp),
                *(ptr_key(x) for x in stats),
                *weights,
                ptr_key(oob),
                *(ptr_key(o) for o in outs),
            ),
        )
        launch.set_param_by_name("advance_prev", wp.int32(advance_prev))
        launch.set_param_by_name("full_pass", wp.int32(full_pass))
        launch.launch()

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["obs"]
        return self._lcache["obs"]

    def rewards(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["reward"]
        return self._lcache["reward"]

    def done(self) -> torch.Tensor:
        """The goal pose reached, in an episode not failed out of bounds."""
        if self.fused_active:
            return self.fb["done_bool"]
        return self._lcache["done"]

    def critic_state(self) -> torch.Tensor:
        """``[E, state_dim]``: the T's pose, goal offset, heading terms, velocity and
        angular velocity, the goal pose, then per agent in role-sorted order its pos, vel
        and role sign (and heading).

        With ``critic_dr`` the randomised parameters follow (``state_base_dim`` on), each
        scaled to about ``[-1, 1]`` over its range: the floor friction, then per robot in
        role-sorted order the drawn drive parameters, command delay, range-noise
        deviation, pose-noise scale, localisation bias and drift (x, y, heading each),
        artefact parameters, mount yaw error and stamp phase, each only when a reset draws
        it."""
        if self.fused_active:
            return self.fb["critic"]
        return self._lcache["critic"]

    def agent_roles(self) -> torch.Tensor:
        """``[E, A]`` int32: 0 pusher, 1 observer, exactly :attr:`role_counts` per env."""
        return self.roles

    def aux_targets(self) -> torch.Tensor:
        """``[E, A, 4]``: the T's position relative to each agent (world axes) divided by
        the lidar range, and the cos/sin of its heading."""
        if self.fused_active:
            return self.fb["aux"]
        return self._lcache["aux"]

    def render_extras(self, env_idx: int) -> dict[str, Any]:
        """Push-T's extras plus the observers' rays for the viewer's ``lidar`` overlay, from
        the newest scan in the ring (current on the fused and the torch path alike): rim to
        hit, or to the full range on a miss, with the end dot only where the ray hit
        something."""
        out = super().render_extras(env_idx)
        nr = self.n_rays
        with torch.no_grad():
            frame = self._ring[env_idx, :, -3 * nr :]
            ranges = frame[:, :nr] * self.lidar_max_range  # [A, R]
            pos = self.world.state.pos[env_idx]
            ang = self._ray_angles()[env_idx]
            dirs = torch.stack((torch.cos(ang), torch.sin(ang)), dim=-1)  # [A, R, 2]
            starts = pos.unsqueeze(1) + self.world.agent_radius.view(-1, 1, 1) * dirs
            ends = pos.unsqueeze(1) + ranges.unsqueeze(-1) * dirs
            observers = torch.nonzero(self.roles[env_idx] == 1).squeeze(-1)
            rays = torch.stack((starts, ends), dim=2)[observers]  # [O, R, 2, 2]
            hit = (frame[:, nr:] > 0.5).reshape(-1, 2, nr).any(dim=1)[observers]
        rays = rays.cpu().numpy()
        out.update(
            lidar=rays.reshape(-1, 2, 2),
            lidar_by_agent=rays,
            lidar_agents=observers.cpu().numpy(),
            lidar_hit=hit.cpu().numpy(),
        )
        return out

    def info(self) -> dict[str, Any]:
        out = super().info()
        if self.fused_active:
            out.update(
                multiobj_reward=self.fb["multiobj"],
                team_coverage=self.fb["coverage"],
                observer_visibility=self.fb["visibility"],
                occlusion=self.fb["occlusion"],
                tee_seen=self.fb["tee_seen_bool"],
            )
            if self.oob_limit > 0.0:
                out["out_of_bounds"] = self._oob.bool()
            return out
        c = self._lcache
        out.update(
            multiobj_reward=c["multiobj"],
            team_coverage=c["team_coverage"],
            observer_visibility=c["observer_visibility"],
            occlusion=c["occlusion"],
            tee_seen=c["tee_seen"],
        )
        if self.oob_limit > 0.0:
            out["out_of_bounds"] = c["out_of_bounds"]
        return out


__all__ = ["OBJECTIVES", "PushTLidarScenario"]
