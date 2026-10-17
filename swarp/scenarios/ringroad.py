"""Ring road: mixed-autonomy stop-and-go waves (the Flow / Sugiyama ring).

``n_vehicles`` cars share a single-lane ring. ``n_agents`` of them are RL cars, engine
``KINEMATIC_BICYCLE`` agents with actions ``(accel, steer)``. The rest are human drivers
following the Intelligent Driver Model (IDM). At the Sugiyama density (22 cars on 230 m)
the uniform flow of IDM drivers is string-unstable: any perturbation grows into a
stop-and-go wave. The RL cars have to damp it.

This is the only scenario here where the phenomenon itself needs **many** vehicles. The
instability is long-wavelength, and an ``N``-car ring only supports wavenumbers
``k = 2*pi*m / N``. Linearising the ring at this density shows the uniform flow stable for
``N <= 8`` and unstable for ``N >= 11``. A simulation sweep agrees: at ``N = 6`` the
humans alone damp a reset perturbation in 20-35 s, while at ``N = 22`` it grows into a
wave with a 2.9 m/s speed spread. So a policy trained on a small ring at the same density
never sees a wave to damp.

Three structural decisions:

**1. The humans are scenario-owned 1-D state, not agents and not obstacles.** As with
shepherding's sheep, ``Environment`` gives no way to inject internally generated actions.
A human is just an arc position and a speed (``idm_s``, ``idm_v``) integrated here. The
engine never sees them: ``collisions=False``, and a crash is read off the bumper-to-bumper
gap rather than produced by contact forces.

**2. Order around the ring is a per-step brute-force search, not an invariant.** Each
vehicle scans all ``N`` for its nearest neighbour ahead and behind by wrapped arc
distance. That is O(N^2) per env, trivial at N=22, and it means an RL car that overtakes
or falls behind (they share one lane, but nothing stops them) needs no re-sorting. The
search runs twice per step, before the humans move (their IDM input) and after (what
obs, reward and crash see).

**3. The geometry is set by density, not by a world size.** ``ring_length`` defaults to
``n_vehicles * 230/22`` and ``n_vehicles`` to ``max(22, n_agents + 1)``. Fixed density is
what the small-ring experiment needs. The registry-wide sweeps build every scenario with
``n_agents`` up to 64, which a fixed 22-car ring cannot hold. There is deliberately no
``world_size`` argument: it would be in metres here and in unit squares everywhere else.
``world_size`` is derived and exposed as an attribute.

Reward, per agent (Flow's ``WaveAttenuationEnv`` form, plus lane-keeping, crash and time)::

    speed  = speed_coef * mean_v                  # all N vehicles; Flow: 4/20
    accel  = -accel_coef * |speed - prev| / dt    # realized; Flow: -4 * mean |accel|
    lane   = -lane_coef * clamp((r - R) / lane_half, -3, 3)^2
    crash  = crash_penalty  if this car's gap ahead or behind is < 0
    time   = time_penalty
    done   = speed std < std_threshold and mean_v > speed_fraction * v_eq,
             held for hold_steps

The accel term uses the *realized* acceleration because the engine does not hand a
scenario its actions; the step-to-step change of the bicycle's own speed is that action,
clamped. Three coefficients differ from Flow on purpose, each from a measured failure.
The magnitudes are per-step means over 600 steps of 64 envs at N=22:

* ``accel_coef = 1``, not 4. With Flow's 4 the accel term of the IDM-driven arm is 1.98
  against a speed term of 0.56. Coasting (zero accel, crashing through the car ahead on
  23% of steps) then out-scores driving: -0.70 per step against -1.42.
* ``crash_penalty = -20``. That makes a crash cost more than any braking it saves.
* ``time_penalty = -1.5``. ``done`` ends the episode, so it must be the best thing that
  can happen. Without this term, a stabilised ring earns +0.69 per step, and terminating
  would forfeit a positive stream, the same trap caging's spec documents. Since
  ``speed_coef * mean_v`` stays below 1.5 for any ring up to 270 m (v_eq 5.2 m/s), every
  step is net negative.

``std_threshold = 0.3`` m/s, ``speed_fraction = 0.5`` and ``hold_steps = 100`` come
from the physics gate. A FollowerStopper AV stabilises the 22-car ring in about 64 s at
58% of the equilibrium speed and in 250 s at 87%. So 0.5 is reachable inside a 150 s
episode, while the humans-only wave (std about 2.9) never comes close.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from swarp._overloads import concrete
from swarp.core.cached_launch import CachedLaunch, ptr_key
from swarp.core.config import WorldConfig
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.interop.autograd import torch_stream_scope
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario
from swarp.scenarios.ringroad_kernels import (
    LANE_CAP,
    N_OBJ,
    OBS_DIM,
    ringroad_arc_kernel,
    ringroad_idm_kernel,
    ringroad_leader_kernel,
    ringroad_obs_kernel,
    ringroad_reset_kernel,
    ringroad_reward_kernel,
)

_TWO_PI = 2.0 * math.pi
#: Sugiyama's 22 cars on 230 m.
SPACING = 230.0 / 22.0
#: IDM gap floor and deceleration floor. With collisions off an RL car can drive through a
#: human, so the gap can reach zero or go negative; unfloored, ``(s*/s)^2`` divides by it.
_S_EPS = 0.1
_DECEL_FLOOR = -9.0


class RingRoadScenario(FusedScenario):
    # Both paths run the same IDM step at the same dt and differ only by ulp-scale
    # reassociation (``pow``, the mean/std reductions). The leader search is the same
    # wrapped subtraction and strict-``<`` scan in both, so neighbour identity agrees
    # exactly; ``done`` and the crash flag are threshold tests compared exactly.
    parity_rtol: float = 1e-4
    parity_atol: float = 1e-5

    def __init__(
        self,
        n_agents: int = 1,
        n_vehicles: int | None = None,
        ring_length: float | None = None,
        ring_length_min: float | None = None,
        ring_length_max: float | None = None,
        car_length: float = 5.0,
        lane_width: float = 3.7,
        v_max: float = 30.0,
        idm_v0: float = 30.0,
        idm_time_headway: float = 1.0,
        idm_accel: float = 1.0,
        idm_decel: float = 1.5,
        idm_delta: float = 4.0,
        idm_s0: float = 2.0,
        max_accel: float = 2.0,
        max_steer: float = 0.6,
        gap_amp: float = 3.0,
        v_amp: float = 3.0,
        speed_coef: float = 0.2,
        accel_coef: float = 1.0,
        lane_coef: float = 0.1,
        crash_penalty: float = -20.0,
        time_penalty: float = -1.5,
        std_threshold: float = 0.3,
        speed_fraction: float = 0.5,
        hold_steps: int = 100,
    ) -> None:
        if n_agents < 1:
            raise ValueError(f"ringroad needs at least one RL car; got n_agents={n_agents}")
        self.n_agents = n_agents
        self.n_vehicles = max(22, n_agents + 1) if n_vehicles is None else int(n_vehicles)
        if self.n_vehicles <= n_agents:
            raise ValueError(
                f"n_vehicles ({self.n_vehicles}) must exceed n_agents ({n_agents}): the ring "
                "needs at least one human driver, or there is no mixed traffic to stabilise"
            )
        self.n_humans = self.n_vehicles - n_agents
        if (ring_length_min is None) != (ring_length_max is None):
            raise ValueError("ring_length_min and ring_length_max must be given together")
        if ring_length_min is not None:
            if ring_length is not None:
                raise ValueError("give ring_length or ring_length_min/max, not both")
            lo, hi = float(ring_length_min), float(ring_length_max)
            if not 0.0 < lo <= hi:
                raise ValueError(f"need 0 < ring_length_min <= ring_length_max; got {lo}, {hi}")
        else:
            lo = hi = self.n_vehicles * SPACING if ring_length is None else float(ring_length)
        self.ring_length_min = lo
        self.ring_length_max = hi
        self.car_length = car_length
        self.lane_width = lane_width
        self.v_max = v_max
        self.idm_v0 = idm_v0
        self.idm_time_headway = idm_time_headway
        self.idm_accel = idm_accel
        self.idm_decel = idm_decel
        self.idm_delta = idm_delta
        self.idm_s0 = idm_s0
        # 2 m/s^2, not Flow's 1. Flow's AV never crashes because SUMO's safe-speed mode
        # overrides it; here nothing does, and an AV clamped at 1 m/s^2 driving the IDM
        # rear-ends its leader in the wave (the IDM wants to brake harder). At 2 m/s^2 the
        # IDM-driven car reproduces the humans-only ring exactly (speed std 2.925 vs 2.924
        # with no AV), which is what makes ``BASELINES["ringroad"]`` the humans-only floor.
        self.max_accel = max_accel
        self.max_steer = max_steer
        self.gap_amp = gap_amp
        self.v_amp = v_amp
        self.speed_coef = speed_coef
        self.accel_coef = accel_coef
        self.lane_coef = lane_coef
        self.crash_penalty = crash_penalty
        self.time_penalty = time_penalty
        self.std_threshold = std_threshold
        self.speed_fraction = speed_fraction
        if not 1 <= int(hold_steps) <= 255:
            raise ValueError(f"hold_steps must lie in [1, 255] (a uint8 counter); got {hold_steps}")
        self.hold_steps = int(hold_steps)

        # Bicycle geometry: l_f = l_r = car_length / 4 (a 2.5 m wheelbase for a 5 m car).
        self.wheelbase = 0.5 * car_length
        self.agent_radius = 0.5 * car_length
        gap_min = lo / self.n_vehicles - car_length
        if gap_min <= idm_s0:
            raise ValueError(
                f"the bumper gap at the shortest ring ({gap_min:.2f} m) must exceed the IDM "
                f"jam distance s0={idm_s0}: there is no positive equilibrium speed below it"
            )
        r_min = lo / _TWO_PI
        turn_r = self.wheelbase / math.tan(max_steer)
        if r_min < 2.0 * turn_r:
            raise ValueError(
                f"ring radius {r_min:.2f} m is under twice the bicycle's turning radius "
                f"({turn_r:.2f} m at max_steer={max_steer}); raise max_steer or the ring length"
            )
        # The box the engine clamps to: the outermost ring plus two car radii (what
        # ``tests/unit/test_reset_kernel.py`` requires of a spawn) and room to leave the lane.
        self.world_size = hi / _TWO_PI + 2.0 * self.agent_radius + 2.0 * lane_width
        # Observation scales, fixed rather than derived from N, so the same actor reads
        # the same units at every ring size. 15 m/s is Flow's speed normaliser.
        self.v_scale = 15.0
        self.gap_scale = 10.0

        self._difficulty = 1.0
        self._difficulty_t: torch.Tensor | None = None

    # -------------------------------------------------------------- curriculum

    @property
    def difficulty(self) -> float:
        """Amplitude of the reset wave, as a fraction of ``gap_amp`` / ``v_amp``.

        ``0`` starts every ring in exact uniform flow, ``1`` with a +/-3 m, +/-3 m/s
        ``m = 1`` wave. Read by the reset kernel through a one-element device tensor, for
        the reason :attr:`ShepherdingScenario.difficulty
        <swarp.scenarios.shepherding.ShepherdingScenario.difficulty>` documents. The
        success thresholds are deliberately not annealed.
        """
        return self._difficulty

    @difficulty.setter
    def difficulty(self, value: float) -> None:
        self._difficulty = min(max(float(value), 0.0), 1.0)
        if self._difficulty_t is not None:
            self._difficulty_t.fill_(self._difficulty)

    # ------------------------------------------------------------------ world

    def make_world(self, n_envs, device, dt, substeps, dtype, world_config=None) -> World:
        cfgs = [
            AgentConfig(
                model=DynamicsModel.KINEMATIC_BICYCLE,
                ctrl_mode=ControlMode.ACCELERATION,
                radius=self.agent_radius,
                max_speed=self.v_max,
                min_speed=0.0,
                max_accel=self.max_accel,
                l_f=0.5 * self.wheelbase,
                l_r=0.5 * self.wheelbase,
                max_steer=self.max_steer,
            )
            for _ in range(self.n_agents)
        ]
        b = self.world_size
        cfg = WorldConfig(
            # Crashes are read off the headway; contact forces would only fight the IDM.
            collisions=False,
            collision_margin=0.5 * self.agent_radius,
            bounds=(-b, b, -b, b),
            bounds_mode="clamp",
        ).override_with(world_config)
        self.dt = dt
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )

        tt = {"device": device, "dtype": dtype}
        ne, nv, k = n_envs, self.n_vehicles, self.n_humans
        self.idm_s = torch.zeros(ne, k, **tt)
        self.idm_v = torch.zeros(ne, k, **tt)
        self.ring_R = torch.zeros(ne, **tt)
        self.ring_L = torch.zeros(ne, **tt)
        self.v_eq = torch.zeros(ne, **tt)
        self.hold = torch.zeros(ne, device=device, dtype=torch.uint8)
        self._prev_speed: torch.Tensor | None = None
        self._cache: dict[str, torch.Tensor] | None = None
        self._difficulty_t = torch.zeros(1, **tt)
        self._difficulty_wp = wp.from_torch(self._difficulty_t, dtype=self.world.wp_dtype)
        self.difficulty = self._difficulty
        self._eye = torch.eye(nv, device=device, dtype=torch.bool)
        self._reset_launch = CachedLaunch()
        return self.world

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    # ------------------------------------------------------------------ reset

    def reset_world(self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False) -> None:
        """Masked reset in one Warp launch, writing every buffer in place."""
        w = self.world
        mask, use_mask = self.reset_mask_wp(env_mask)
        st = w.state_wp()
        scalar = w.wp_dtype
        if self.fused_active:
            self.ensure_fused()
            self.sync_fused_handles()
            pk = self._wp
            idm_s, idm_v, hold = pk["idm_s"], pk["idm_v"], pk["hold"]
            ring_R, ring_L, v_eq = pk["ring_R"], pk["ring_L"], pk["v_eq"]
        else:
            idm_s = wp.from_torch(self.idm_s.contiguous(), dtype=scalar)
            idm_v = wp.from_torch(self.idm_v.contiguous(), dtype=scalar)
            ring_R = wp.from_torch(self.ring_R, dtype=scalar)
            ring_L = wp.from_torch(self.ring_L, dtype=scalar)
            v_eq = wp.from_torch(self.v_eq, dtype=scalar)
            hold = wp.from_torch(self.hold, dtype=wp.uint8)
        seed = wp.int32(w.next_kernel_seed())
        with torch_stream_scope(w.device):
            launch = self._reset_launch.get(
                concrete(ringroad_reset_kernel, scalar),
                dim=w.n_envs,
                inputs=[
                    mask,
                    use_mask,
                    seed,
                    self._difficulty_wp,
                    scalar(self.ring_length_min),
                    scalar(self.ring_length_max),
                    scalar(self.car_length),
                    scalar(self.gap_amp),
                    scalar(self.v_amp),
                    scalar(self.idm_v0),
                    scalar(self.idm_time_headway),
                    scalar(self.idm_delta),
                    scalar(self.idm_s0),
                    scalar(self.v_max),
                    wp.int32(self.n_agents),
                    wp.int32(self.n_vehicles),
                    st.pos,
                    st.theta,
                    st.vel,
                    st.speed,
                    st.ang_vel,
                    idm_s,
                    idm_v,
                    ring_R,
                    ring_L,
                    v_eq,
                    hold,
                ],
                device=w.device,
                key=(
                    w.n_envs,
                    ptr_key(mask),
                    ptr_key(self._difficulty_wp),
                    self.ring_length_min,
                    self.ring_length_max,
                    self.car_length,
                    self.gap_amp,
                    self.v_amp,
                    self.idm_v0,
                    self.idm_time_headway,
                    self.idm_delta,
                    self.idm_s0,
                    self.v_max,
                    self.n_agents,
                    self.n_vehicles,
                    ptr_key(st.pos),
                    ptr_key(st.theta),
                    ptr_key(st.vel),
                    ptr_key(st.speed),
                    ptr_key(st.ang_vel),
                    ptr_key(idm_s),
                    ptr_key(idm_v),
                    ptr_key(ring_R),
                    ptr_key(ring_L),
                    ptr_key(v_eq),
                    ptr_key(hold),
                ),
            )
            launch.set_param_by_name("use_mask", use_mask)
            launch.set_param_by_name("seed", seed)
            launch.launch()
        w.mark_pos_dirty()
        if not self.fused_active:
            self._prev_speed = None
        self.finish_reset(env_mask, obs_only=obs_only)

    # --------------------------------------------------------- fused fast path

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        ne, na, nv, k = n_envs, self.n_agents, self.n_vehicles, self.n_humans
        return (
            Buf("obs", (ne, na, OBS_DIM)),
            Buf("reward", (ne, na)),
            Buf("multiobj", (ne, na, N_OBJ)),
            Buf("lane", (ne, na)),
            Buf("std", (ne,)),
            Buf("mean", (ne,)),
            Buf("crashed", (ne,)),
            Buf("done", (ne,), "uint8", bool_view=True),
            Buf("resetmask", (ne,), "uint8", reset_mask=True),
            # Scratch rewritten in full by every pass before it is read: not carries.
            Buf("arc_s", (ne, nv)),
            Buf("arc_v", (ne, nv)),
            Buf("lead_gap", (ne, nv)),
            Buf("lead_v", (ne, nv)),
            Buf("foll_gap", (ne, nv)),
            Buf("foll_v", (ne, nv)),
            # The humans, advanced in place by the IDM kernel and reassigned by the grad
            # path's torch integrator.
            Buf("idm_s", (ne, k), attr="idm_s", alloc="never", carry=True, watch=True),
            Buf("idm_v", (ne, k), attr="idm_v", alloc="never", carry=True, watch=True),
            # Per-episode geometry, written only by the reset kernel.
            Buf("ring_R", (ne,), attr="ring_R", alloc="never"),
            Buf("ring_L", (ne,), attr="ring_L", alloc="never"),
            Buf("v_eq", (ne,), attr="v_eq", alloc="never"),
            Buf("hold", (ne,), "uint8", attr="hold", alloc="never", carry=True),
            Buf("prevspeed", (ne, na), attr="_prev_speed", alloc="if_none", carry=True, watch=True),
        )

    def launch_fused(self, pass_: FusedPass) -> None:
        """``arc -> leader [-> idm -> leader] -> obs -> reward``; a reset moves no human."""
        st = self.world.state_wp()
        self._launch_arc(st)
        self._launch_leader()
        if pass_.is_step:
            self._launch_idm()
            self._launch_leader()
        self._launch_obs(st, full_pass=pass_.full_pass)
        self._launch_reward(st, advance_prev=pass_.advance_prev, full_pass=pass_.full_pass)

    def _launch_arc(self, st) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        wp.launch(
            concrete(ringroad_arc_kernel, scalar),
            dim=(w.n_envs, self.n_vehicles),
            inputs=[
                st.pos,
                st.vel,
                pk["idm_s"],
                pk["idm_v"],
                pk["ring_R"],
                wp.int32(self.n_agents),
                scalar(_TWO_PI),
            ],
            outputs=[pk["arc_s"], pk["arc_v"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_leader(self) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        wp.launch(
            concrete(ringroad_leader_kernel, scalar),
            dim=(w.n_envs, self.n_vehicles),
            inputs=[
                pk["arc_s"],
                pk["arc_v"],
                pk["ring_L"],
                wp.int32(self.n_vehicles),
                scalar(self.car_length),
            ],
            outputs=[pk["lead_gap"], pk["lead_v"], pk["foll_gap"], pk["foll_v"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_idm(self) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        wp.launch(
            concrete(ringroad_idm_kernel, scalar),
            dim=(w.n_envs, self.n_humans),
            inputs=[
                pk["lead_gap"],
                pk["lead_v"],
                pk["ring_L"],
                wp.int32(self.n_agents),
                scalar(self.idm_v0),
                scalar(self.idm_time_headway),
                scalar(self.idm_accel),
                scalar(self.idm_delta),
                scalar(self.idm_s0),
                scalar(self._inv_2sqrt_ab),
                scalar(_S_EPS),
                scalar(_DECEL_FLOOR),
                scalar(self.v_max),
                scalar(self.dt),
            ],
            outputs=[pk["idm_s"], pk["idm_v"], pk["arc_s"], pk["arc_v"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_obs(self, st, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        wp.launch(
            concrete(ringroad_obs_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.theta,
                pk["arc_v"],
                pk["lead_gap"],
                pk["lead_v"],
                pk["foll_gap"],
                pk["foll_v"],
                pk["ring_R"],
                pk["resetmask"],
                scalar(1.0 / self.v_scale),
                scalar(1.0 / self.gap_scale),
                scalar(2.0 / self.lane_width),
                scalar(self.wheelbase),
                scalar(1.0 / self.max_steer),
                wp.int32(full_pass),
            ],
            outputs=[pk["obs"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_reward(self, st, advance_prev: int, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        wp.launch(
            concrete(ringroad_reward_kernel, scalar),
            dim=w.n_envs,
            inputs=[
                st.pos,
                st.speed,
                pk["arc_v"],
                pk["lead_gap"],
                pk["foll_gap"],
                pk["ring_R"],
                pk["v_eq"],
                pk["resetmask"],
                wp.int32(self.n_agents),
                wp.int32(self.n_vehicles),
                scalar(1.0 / self.n_vehicles),
                scalar(self.speed_coef),
                scalar(self.accel_coef),
                scalar(1.0 / self.dt),
                scalar(self.lane_coef),
                scalar(2.0 / self.lane_width),
                scalar(LANE_CAP),
                scalar(self.crash_penalty),
                scalar(self.time_penalty),
                scalar(self.std_threshold),
                scalar(self.speed_fraction),
                wp.int32(self.hold_steps),
                wp.int32(advance_prev),
                wp.int32(full_pass),
            ],
            outputs=[
                pk["prevspeed"],
                pk["hold"],
                pk["reward"],
                pk["done"],
                pk["multiobj"],
                pk["std"],
                pk["mean"],
                pk["crashed"],
                pk["lane"],
            ],
            device=w.device,
            record_tape=False,
        )

    @property
    def _inv_2sqrt_ab(self) -> float:
        return 1.0 / (2.0 * math.sqrt(self.idm_accel * self.idm_decel))

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh(integrate=True)

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        self._refresh(reset_mask=env_mask, integrate=False)

    def _arcs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Arc position and tangential speed of every vehicle ``[n_envs, n_vehicles]``."""
        s = self.world.state
        x, y = s.pos[..., 0], s.pos[..., 1]
        phi = torch.atan2(y, x)
        phi = torch.where(phi < 0.0, phi + _TWO_PI, phi)
        r = torch.sqrt(x * x + y * y).clamp(min=1e-9)
        arc_rl = self.ring_R.unsqueeze(1) * phi
        v_rl = (s.vel[..., 1] * x - s.vel[..., 0] * y) / r
        return torch.cat([arc_rl, self.idm_s], dim=1), torch.cat([v_rl, self.idm_v], dim=1)

    def _neighbours(self, arc_s: torch.Tensor, arc_v: torch.Tensor):
        """``(lead_gap, lead_v, foll_gap, foll_v)``, each ``[n_envs, n_vehicles]``."""
        L = self.ring_L.view(-1, 1, 1)
        ahead = arc_s.unsqueeze(1) - arc_s.unsqueeze(2)  # [E, i, j] = s_j - s_i
        ahead = torch.where(ahead < 0.0, ahead + L, ahead).masked_fill(self._eye, math.inf)
        behind = arc_s.unsqueeze(2) - arc_s.unsqueeze(1)  # s_i - s_j
        behind = torch.where(behind < 0.0, behind + L, behind).masked_fill(self._eye, math.inf)
        ja = ahead.argmin(dim=-1, keepdim=True)
        jb = behind.argmin(dim=-1, keepdim=True)
        lead_gap = ahead.gather(-1, ja).squeeze(-1) - self.car_length
        foll_gap = behind.gather(-1, jb).squeeze(-1) - self.car_length
        lead_v = arc_v.gather(-1, ja.squeeze(-1))
        foll_v = arc_v.gather(-1, jb.squeeze(-1))
        return lead_gap, lead_v, foll_gap, foll_v

    def _refresh(self, reset_mask: torch.Tensor | None = None, integrate: bool = True) -> None:
        w = self.world
        s = w.state
        na = self.n_agents
        if integrate:
            arc_s, arc_v = self._arcs()
            lead_gap, lead_v, _, _ = self._neighbours(arc_s, arc_v)
            v = self.idm_v
            dyn = v * self.idm_time_headway + v * (v - lead_v[:, na:]) * self._inv_2sqrt_ab
            s_star = self.idm_s0 + dyn.clamp(min=0.0)
            ratio = s_star / lead_gap[:, na:].clamp(min=_S_EPS)
            free = torch.pow(v / self.idm_v0, self.idm_delta)
            acc = self.idm_accel * (1.0 - free - ratio * ratio)
            vn = (v + acc.clamp(min=_DECEL_FLOOR) * self.dt).clamp(0.0, self.v_max)
            sn = self.idm_s + vn * self.dt
            L = self.ring_L.unsqueeze(1)
            self.idm_v = vn
            self.idm_s = torch.where(sn >= L, sn - L, sn)

        arc_s, arc_v = self._arcs()
        lead_gap, lead_v, foll_gap, foll_v = self._neighbours(arc_s, arc_v)
        inv_n = 1.0 / self.n_vehicles
        mean_v = arc_v.sum(dim=-1) * inv_n
        std_v = torch.sqrt((arc_v - mean_v.unsqueeze(1)).pow(2).sum(dim=-1) * inv_n)
        stable = (std_v < self.std_threshold) & (mean_v > self.speed_fraction * self.v_eq)
        if integrate:
            # Mirrors the kernel: advanced only on a step, saturating at 255, zeroed by the
            # reset kernel. In place, so the adopted buffer's pointer never moves.
            h = self.hold.int()
            self.hold.copy_(torch.where(stable, (h + 1).clamp(max=255), torch.zeros_like(h)))

        speed = s.speed
        if self._prev_speed is None:
            self._prev_speed = speed.detach().clone()
        acc_term = -self.accel_coef * (speed - self._prev_speed).abs() / self.dt
        if reset_mask is None:
            self._prev_speed = speed.detach().clone()
        else:
            m = reset_mask.unsqueeze(1)
            acc_term = torch.where(m, torch.zeros_like(acc_term), acc_term)
            self._prev_speed = torch.where(m, speed.detach(), self._prev_speed)

        x, y = s.pos[..., 0], s.pos[..., 1]
        r = torch.sqrt(x * x + y * y)
        off = r - self.ring_R.unsqueeze(1)
        q = (off * (2.0 / self.lane_width)).clamp(-LANE_CAP, LANE_CAP)
        crash = (lead_gap[:, :na] < 0.0) | (foll_gap[:, :na] < 0.0)
        self._cache = {
            "arc_v": arc_v,
            "lead_gap": lead_gap,
            "lead_v": lead_v,
            "foll_gap": foll_gap,
            "foll_v": foll_v,
            "phi": torch.atan2(y, x),
            "off": off,
            "mean_v": mean_v,
            "std_v": std_v,
            "crashed": (lead_gap < 0.0).any(dim=-1),
            "speed_term": self.speed_coef * mean_v,
            "acc_term": acc_term,
            "lane_term": -self.lane_coef * q * q,
            "crash_term": crash.to(w.dtype) * self.crash_penalty,
        }

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["obs"]
        c = self._cache
        s = self.world.state
        na = self.n_agents
        v = c["arc_v"][:, :na]
        rel = s.theta - c["phi"]
        ff = torch.atan(self.wheelbase / self.ring_R) / self.max_steer
        return torch.stack(
            [
                v / self.v_scale,
                (c["lead_v"][:, :na] - v) / self.v_scale,
                c["lead_gap"][:, :na] / self.gap_scale,
                (c["foll_v"][:, :na] - v) / self.v_scale,
                c["foll_gap"][:, :na] / self.gap_scale,
                c["off"] * (2.0 / self.lane_width),
                -torch.cos(rel),
                torch.sin(rel),
                ff.unsqueeze(1).expand(-1, na),
            ],
            dim=-1,
        )

    def _terms(self) -> list[torch.Tensor]:
        """The five per-agent reward terms ``[n_envs, n_agents]``, in ``OBJ_*`` order."""
        c = self._cache
        speed = c["speed_term"].unsqueeze(1).expand_as(c["acc_term"])
        time = torch.full_like(c["acc_term"], self.time_penalty)
        return [speed, c["acc_term"], c["lane_term"], c["crash_term"], time]

    def rewards(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["reward"]
        speed, acc, lane, crash, time = self._terms()
        return speed + acc + lane + crash + time

    def done(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["done_bool"]
        return self.hold.int() >= self.hold_steps

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {
                "speed_std": self.fb["std"],
                "mean_speed": self.fb["mean"],
                "crashed": self.fb["crashed"],
                "lane_offset": self.fb["lane"],
                "multiobj_reward": self.fb["multiobj"],
            }
        c = self._cache
        return {
            "speed_std": c["std_v"],
            "mean_speed": c["mean_v"],
            "crashed": c["crashed"].to(self.world.dtype),
            "lane_offset": c["off"],
            "multiobj_reward": torch.stack(self._terms(), dim=-1),
        }

    def render_extras(self, env_idx: int) -> dict[str, Any]:
        """The human cars as boxes on the ring, and the lane as an annulus."""
        R = float(self.ring_R[env_idx])
        ang = (self.idm_s[env_idx] / R).detach().cpu().double()
        boxes = torch.stack(
            [
                R * torch.cos(ang),
                R * torch.sin(ang),
                ang + 0.5 * math.pi,
                torch.full_like(ang, 0.5 * self.car_length),
                torch.full_like(ang, 0.25 * self.lane_width),
            ],
            dim=-1,
        )
        half = 0.5 * self.lane_width
        return {"vehicles": boxes.numpy(), "road_ring": (R - half, R + half)}
