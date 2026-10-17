"""The ring-road task written as a VMAS scenario, for the cross-simulator benchmark.

The same task as :class:`swarp.scenarios.ringroad.RingRoadScenario` at its defaults, the
way an efficient VMAS scenario would be written:

* each RL car is a VMAS ``Agent`` with VMAS's own ``KinematicBicycle`` dynamics (Euler, to
  match swarp's semi-implicit Euler), drag and friction off, collisions off. swarp's
  action is ``(accel, steer)`` and VMAS's bicycle takes ``(speed, steer)``, so
  ``process_action`` integrates the commanded acceleration into the speed command,
  clamped to ``[0, v_max]``: the same recurrence swarp's kernel runs;
* the IDM humans are one batched ``[n_envs, n_humans]`` tensor pair advanced in
  ``post_step``, not one VMAS entity each. That is the same algorithm swarp runs (a
  brute-force leader search before and after the IDM step), vectorised over envs and
  vehicles, so the comparison measures the simulators rather than a per-entity loop a
  fair port would not write;
* observation, reward and done are computed once per step in ``post_step`` and sliced
  per agent, again the efficient form.

Not a parity oracle: VMAS integrates the bicycle through its force pipeline, so
trajectories match swarp's to float rounding, not bit for bit. It exists to be timed.
"""

from __future__ import annotations

import math

import torch
from vmas.simulator.core import Agent, Box, World
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario

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


class RingRoadVMAS(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.na = kwargs.get("n_agents", 1)
        self.nv = kwargs.get("n_vehicles", max(22, self.na + 1))
        self.nh = self.nv - self.na
        self.L = self.nv * SPACING
        self.R = self.L / _TWO_PI
        self.v_eq = idm_equilibrium_speed(SPACING - CAR_LENGTH)
        self.dt = 0.1
        world = World(
            batch_dim, device, dt=self.dt, drag=0.0, linear_friction=0.0, angular_friction=0.0
        )
        for i in range(self.na):
            world.add_agent(
                Agent(
                    name=f"car_{i}",
                    shape=Box(length=CAR_LENGTH, width=2.0),
                    collide=False,
                    u_range=[MAX_ACCEL, MAX_STEER],
                    dynamics=KinematicBicycle(
                        world,
                        width=2.0,
                        l_f=0.5 * WHEELBASE,
                        l_r=0.5 * WHEELBASE,
                        max_steering_angle=MAX_STEER,
                        integration="euler",
                    ),
                )
            )
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.idm_s, self.idm_v = z(batch_dim, self.nh), z(batch_dim, self.nh)
        self.speed, self.prev_speed = z(batch_dim, self.na), z(batch_dim, self.na)
        self.hold = torch.zeros(batch_dim, device=device, dtype=torch.int32)
        self._eye = torch.eye(self.nv, device=device, dtype=torch.bool)
        rl = [(a * self.nv) // self.na for a in range(self.na)]
        self._rl_slot = torch.tensor(rl, device=device)
        self._human_slot = torch.tensor([i for i in range(self.nv) if i not in rl], device=device)
        return world

    # ------------------------------------------------------------------ reset

    def reset_world_at(self, env_index: int | None = None) -> None:
        w = self.world
        B, dev = w.batch_dim, w.device
        rows = (
            torch.arange(B, device=dev)
            if env_index is None
            else torch.tensor([env_index], device=dev)
        )
        n = rows.numel()
        i = torch.arange(self.nv, device=dev)
        phase = torch.rand(n, 1, device=dev) * _TWO_PI
        x0 = torch.rand(n, 1, device=dev) * self.L
        th = _TWO_PI * i / self.nv + phase
        sn = torch.sin(th)
        ga = min(GAP_AMP, 0.8 * (SPACING - CAR_LENGTH))
        gaps = SPACING + ga * sn
        x = x0 + torch.cumsum(gaps, dim=1) - gaps
        x = torch.where(x >= self.L, x - self.L, x)
        v = (self.v_eq + V_AMP * sn).clamp(0.0, V_MAX)
        self.idm_s[rows] = x[:, self._human_slot]
        self.idm_v[rows] = v[:, self._human_slot]
        for a, agent in enumerate(w.agents):
            xa, va = x[:, self._rl_slot[a]], v[:, self._rl_slot[a]]
            ang = xa / self.R
            hd = ang + 0.5 * math.pi
            pos = torch.stack([self.R * torch.cos(ang), self.R * torch.sin(ang)], -1)
            vel = torch.stack([va * torch.cos(hd), va * torch.sin(hd)], -1)
            if env_index is None:
                agent.set_pos(pos, batch_index=None)
                agent.set_rot(hd.unsqueeze(-1), batch_index=None)
                agent.set_vel(vel, batch_index=None)
            else:
                agent.set_pos(pos[0], batch_index=env_index)
                agent.set_rot(hd[0].unsqueeze(-1), batch_index=env_index)
                agent.set_vel(vel[0], batch_index=env_index)
            self.speed[rows, a] = va
            self.prev_speed[rows, a] = va
        self.hold[rows] = 0
        self._refresh(advance=False)

    # ------------------------------------------------------------------- step

    def process_action(self, agent: Agent) -> None:
        a = self.world.agents.index(agent)
        v = (self.speed[:, a] + agent.action.u[:, 0] * self.dt).clamp(0.0, V_MAX)
        self.speed[:, a] = v
        agent.action.u[:, 0] = v  # VMAS's bicycle is speed-commanded

    def post_step(self) -> None:
        self._refresh(advance=True)

    def _arcs(self):
        pos = torch.stack([a.state.pos for a in self.world.agents], 1)
        vel = torch.stack([a.state.vel for a in self.world.agents], 1)
        x, y = pos[..., 0], pos[..., 1]
        phi = torch.atan2(y, x)
        phi = torch.where(phi < 0.0, phi + _TWO_PI, phi)
        r = torch.sqrt(x * x + y * y).clamp(min=1e-9)
        v_rl = (vel[..., 1] * x - vel[..., 0] * y) / r
        arc_s = torch.cat([self.R * phi, self.idm_s], 1)
        arc_v = torch.cat([v_rl, self.idm_v], 1)
        return arc_s, arc_v, pos, phi, r

    def _neighbours(self, arc_s, arc_v):
        ahead = arc_s.unsqueeze(1) - arc_s.unsqueeze(2)
        ahead = torch.where(ahead < 0.0, ahead + self.L, ahead).masked_fill(self._eye, math.inf)
        behind = arc_s.unsqueeze(2) - arc_s.unsqueeze(1)
        behind = torch.where(behind < 0.0, behind + self.L, behind).masked_fill(self._eye, math.inf)
        ja, jb = ahead.argmin(-1, keepdim=True), behind.argmin(-1, keepdim=True)
        return (
            ahead.gather(-1, ja).squeeze(-1) - CAR_LENGTH,
            arc_v.gather(-1, ja.squeeze(-1)),
            behind.gather(-1, jb).squeeze(-1) - CAR_LENGTH,
            arc_v.gather(-1, jb.squeeze(-1)),
        )

    def _refresh(self, advance: bool) -> None:
        na = self.na
        if advance:
            arc_s, arc_v, *_ = self._arcs()
            lead_gap, lead_v, _, _ = self._neighbours(arc_s, arc_v)
            v = self.idm_v
            dyn = v * IDM["T"] + v * (v - lead_v[:, na:]) / (2.0 * math.sqrt(IDM["a"] * IDM["b"]))
            ratio = (IDM["s0"] + dyn.clamp(min=0.0)) / lead_gap[:, na:].clamp(min=S_EPS)
            acc = IDM["a"] * (1.0 - (v / IDM["v0"]) ** IDM["delta"] - ratio * ratio)
            vn = (v + acc.clamp(min=DECEL_FLOOR) * self.dt).clamp(0.0, V_MAX)
            sn = self.idm_s + vn * self.dt
            self.idm_v, self.idm_s = vn, torch.where(sn >= self.L, sn - self.L, sn)
        arc_s, arc_v, pos, phi, r = self._arcs()
        lead_gap, lead_v, foll_gap, foll_v = self._neighbours(arc_s, arc_v)
        mean_v = arc_v.mean(-1)
        std_v = arc_v.std(-1, unbiased=False)
        if advance:
            stable = (std_v < STD_THRESHOLD) & (mean_v > SPEED_FRACTION * self.v_eq)
            self.hold = torch.where(stable, (self.hold + 1).clamp(max=255), 0)
        own = arc_v[:, :na]
        theta = torch.stack([a.state.rot[:, 0] for a in self.world.agents], 1)
        rel = theta - phi
        off = r - self.R
        self._obs = torch.stack(
            [
                own / V_SCALE,
                (lead_v[:, :na] - own) / V_SCALE,
                lead_gap[:, :na] / GAP_SCALE,
                (foll_v[:, :na] - own) / V_SCALE,
                foll_gap[:, :na] / GAP_SCALE,
                off * (2.0 / LANE_WIDTH),
                -torch.cos(rel),
                torch.sin(rel),
                torch.full_like(own, math.atan(WHEELBASE / self.R) / MAX_STEER),
            ],
            -1,
        )
        q = (off * (2.0 / LANE_WIDTH)).clamp(-LANE_CAP, LANE_CAP)
        crash = (lead_gap[:, :na] < 0.0) | (foll_gap[:, :na] < 0.0)
        accel = -ACCEL_COEF * (self.speed - self.prev_speed).abs() / self.dt
        self._rew = (
            SPEED_COEF * mean_v.unsqueeze(1)
            + accel
            - LANE_COEF * q * q
            + crash.float() * CRASH_PENALTY
            + TIME_PENALTY
        )
        self.prev_speed = self.speed.clone()

    # ---------------------------------------------------------- obs / reward

    def observation(self, agent: Agent) -> torch.Tensor:
        return self._obs[:, self.world.agents.index(agent)]

    def reward(self, agent: Agent) -> torch.Tensor:
        return self._rew[:, self.world.agents.index(agent)]

    def done(self) -> torch.Tensor:
        return self.hold >= HOLD_STEPS
