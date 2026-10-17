"""The ring-road task as a JaxMARL ``MultiAgentEnv``, for the cross-simulator benchmark.

The same task as :class:`swarp.scenarios.ringroad.RingRoadScenario` at its defaults,
written the way a JaxMARL environment is: one env as pure ``jnp`` functions over a
frozen state, which the benchmark ``vmap``s over envs and ``lax.scan``s over time inside
one ``jit``. Same recurrences as swarp's kernels: the kinematic bicycle with accel/steer
clamps and ``min_speed = 0`` (semi-implicit Euler), a brute-force leader search before
and after one IDM Euler step, the 9-wide observation and the five-term reward.

JaxMARL's ``step`` resets on ``done`` by evaluating a full reset every step and selecting
it, so that cost is part of the measurement, as it is for every JaxMARL environment. The
ring length is fixed here, so the IDM equilibrium speed is a host constant instead of the
per-env bisection swarp's reset kernel runs.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box

from swarp.benchmark._adapters.ringroad_constants import (
    ACCEL_COEF,
    CAR_LENGTH,
    CRASH_PENALTY,
    DECEL_FLOOR,
    GAP_AMP,
    GAP_SCALE,
    HOLD_STEPS,
    IDM,
    LANE_CAP,
    LANE_COEF,
    LANE_WIDTH,
    MAX_ACCEL,
    MAX_STEER,
    S_EPS,
    SPACING,
    SPEED_COEF,
    SPEED_FRACTION,
    STD_THRESHOLD,
    TIME_PENALTY,
    V_AMP,
    V_MAX,
    V_SCALE,
    WHEELBASE,
    idm_equilibrium_speed,
)

_TWO_PI = 2.0 * math.pi


@struct.dataclass
class RingState:
    pos: jnp.ndarray  # [na, 2]
    vel: jnp.ndarray  # [na, 2]
    theta: jnp.ndarray  # [na]
    speed: jnp.ndarray  # [na]
    prev_speed: jnp.ndarray  # [na]
    idm_s: jnp.ndarray  # [nh]
    idm_v: jnp.ndarray  # [nh]
    hold: jnp.ndarray  # int32 scalar
    step: jnp.ndarray  # int32 scalar


class RingRoadJax(MultiAgentEnv):
    def __init__(
        self,
        num_agents: int = 1,
        n_vehicles: int | None = None,
        dt: float = 0.1,
        max_steps: int = 1500,
    ) -> None:
        super().__init__(num_agents=num_agents)
        self.na = num_agents
        self.nv = max(22, num_agents + 1) if n_vehicles is None else n_vehicles
        self.nh = self.nv - self.na
        self.L = self.nv * SPACING
        self.R = self.L / _TWO_PI
        self.v_eq = idm_equilibrium_speed(SPACING - CAR_LENGTH)
        self.dt, self.max_steps = dt, max_steps
        self.agents = [f"car_{i}" for i in range(num_agents)]
        hi = jnp.array([MAX_ACCEL, MAX_STEER])
        self.action_spaces = {a: Box(-hi, hi, (2,)) for a in self.agents}
        self.observation_spaces = {a: Box(-jnp.inf, jnp.inf, (9,)) for a in self.agents}
        rl = [(a * self.nv) // self.na for a in range(self.na)]
        self._rl_slot = jnp.array(rl)
        self._human_slot = jnp.array([i for i in range(self.nv) if i not in rl])
        self._eye = jnp.eye(self.nv, dtype=bool)

    # ------------------------------------------------------------------ reset

    def reset(self, key):
        k1, k2 = jax.random.split(key)
        i = jnp.arange(self.nv)
        th = _TWO_PI * i / self.nv + jax.random.uniform(k1) * _TWO_PI
        sn = jnp.sin(th)
        gaps = SPACING + min(GAP_AMP, 0.8 * (SPACING - CAR_LENGTH)) * sn
        x = jax.random.uniform(k2) * self.L + jnp.cumsum(gaps) - gaps
        x = jnp.where(x >= self.L, x - self.L, x)
        v = jnp.clip(self.v_eq + V_AMP * sn, 0.0, V_MAX)
        xa, va = x[self._rl_slot], v[self._rl_slot]
        ang = xa / self.R
        hd = ang + 0.5 * math.pi
        state = RingState(
            pos=jnp.stack([self.R * jnp.cos(ang), self.R * jnp.sin(ang)], -1),
            vel=jnp.stack([va * jnp.cos(hd), va * jnp.sin(hd)], -1),
            theta=hd,
            speed=va,
            prev_speed=va,
            idm_s=x[self._human_slot],
            idm_v=v[self._human_slot],
            hold=jnp.int32(0),
            step=jnp.int32(0),
        )
        return self.get_obs(state), state

    # ------------------------------------------------------------------- step

    def _arcs(self, s: RingState):
        x, y = s.pos[:, 0], s.pos[:, 1]
        phi = jnp.arctan2(y, x)
        phi = jnp.where(phi < 0.0, phi + _TWO_PI, phi)
        r = jnp.maximum(jnp.sqrt(x * x + y * y), 1e-9)
        v_rl = (s.vel[:, 1] * x - s.vel[:, 0] * y) / r
        return (jnp.concatenate([self.R * phi, s.idm_s]), jnp.concatenate([v_rl, s.idm_v]), phi, r)

    def _neighbours(self, arc_s, arc_v):
        ahead = arc_s[None, :] - arc_s[:, None]
        ahead = jnp.where(self._eye, jnp.inf, jnp.where(ahead < 0.0, ahead + self.L, ahead))
        behind = arc_s[:, None] - arc_s[None, :]
        behind = jnp.where(self._eye, jnp.inf, jnp.where(behind < 0.0, behind + self.L, behind))
        ja, jb = jnp.argmin(ahead, -1), jnp.argmin(behind, -1)
        return (
            jnp.min(ahead, -1) - CAR_LENGTH,
            arc_v[ja],
            jnp.min(behind, -1) - CAR_LENGTH,
            arc_v[jb],
        )

    def step_env(self, key, state: RingState, actions):
        a = jnp.stack([actions[ag] for ag in self.agents])
        acc = jnp.clip(a[:, 0], -MAX_ACCEL, MAX_ACCEL)
        delta = jnp.clip(a[:, 1], -MAX_STEER, MAX_STEER)
        s_new = jnp.clip(state.speed + acc * self.dt, 0.0, V_MAX)
        beta = jnp.arctan(jnp.tan(delta) * 0.5)  # l_r / wheelbase = 1/2
        hd = state.theta + beta
        vel = jnp.stack([s_new * jnp.cos(hd), s_new * jnp.sin(hd)], -1)
        w = s_new / WHEELBASE * jnp.cos(beta) * jnp.tan(delta)
        state = state.replace(
            pos=state.pos + vel * self.dt, vel=vel, theta=state.theta + w * self.dt, speed=s_new
        )

        na = self.na
        arc_s, arc_v, _, _ = self._arcs(state)
        lead_gap, lead_v, _, _ = self._neighbours(arc_s, arc_v)
        v = state.idm_v
        inv_2sqrt_ab = 1.0 / (2.0 * math.sqrt(IDM["a"] * IDM["b"]))
        dyn = jnp.maximum(v * IDM["T"] + v * (v - lead_v[na:]) * inv_2sqrt_ab, 0.0)
        ratio = (IDM["s0"] + dyn) / jnp.maximum(lead_gap[na:], S_EPS)
        idm_acc = jnp.maximum(
            IDM["a"] * (1.0 - (v / IDM["v0"]) ** IDM["delta"] - ratio * ratio), DECEL_FLOOR
        )
        vn = jnp.clip(v + idm_acc * self.dt, 0.0, V_MAX)
        sn = state.idm_s + vn * self.dt
        state = state.replace(idm_v=vn, idm_s=jnp.where(sn >= self.L, sn - self.L, sn))

        obs, reward, stable = self._evaluate(state)
        hold = jnp.where(stable, jnp.minimum(state.hold + 1, 255), 0)
        state = state.replace(hold=hold, prev_speed=state.speed, step=state.step + 1)
        done = (hold >= HOLD_STEPS) | (state.step >= self.max_steps)
        rewards = {ag: reward[i] for i, ag in enumerate(self.agents)}
        dones = {ag: done for ag in self.agents} | {"__all__": done}
        return obs, state, rewards, dones, {}

    def _evaluate(self, state: RingState):
        na = self.na
        arc_s, arc_v, phi, r = self._arcs(state)
        lead_gap, lead_v, foll_gap, foll_v = self._neighbours(arc_s, arc_v)
        mean_v, std_v = jnp.mean(arc_v), jnp.std(arc_v)
        stable = (std_v < STD_THRESHOLD) & (mean_v > SPEED_FRACTION * self.v_eq)
        own = arc_v[:na]
        rel = state.theta - phi
        off = r - self.R
        obs_m = jnp.stack(
            [
                own / V_SCALE,
                (lead_v[:na] - own) / V_SCALE,
                lead_gap[:na] / GAP_SCALE,
                (foll_v[:na] - own) / V_SCALE,
                foll_gap[:na] / GAP_SCALE,
                off * (2.0 / LANE_WIDTH),
                -jnp.cos(rel),
                jnp.sin(rel),
                jnp.full_like(own, math.atan(WHEELBASE / self.R) / MAX_STEER),
            ],
            -1,
        )
        q = jnp.clip(off * (2.0 / LANE_WIDTH), -LANE_CAP, LANE_CAP)
        crash = (lead_gap[:na] < 0.0) | (foll_gap[:na] < 0.0)
        accel = -ACCEL_COEF * jnp.abs(state.speed - state.prev_speed) / self.dt
        reward = (
            SPEED_COEF * mean_v + accel - LANE_COEF * q * q + crash * CRASH_PENALTY + TIME_PENALTY
        )
        return {ag: obs_m[i] for i, ag in enumerate(self.agents)}, reward, stable

    def get_obs(self, state: RingState):
        return self._evaluate(state)[0]
