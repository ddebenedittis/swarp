"""Push-T: agents push a T-shaped rigid body to a target pose.

The movable-body model of :mod:`wmas.scenarios.transport`, extended from a disk to a
non-convex body and from a position goal to a full **pose** goal. Each step:

  1. the T is installed as two oriented ``BOX`` obstacles (crossbar + stem) sharing
     one pose, so the Warp agent step pushes agents *off* it through the existing
     soft agent-obstacle box-SDF contact;
  2. after the step, the reaction contact force the T receives from the agents
     (Newton's third law of that same spring-damper contact) is summed, gather-style,
     into a net force + torque about the body centroid, and the T is integrated
     (semi-implicit Euler);
  3. the updated pose is installed for the next step.

Contacts are **frictionless** — normal-only spring plus normal damping, as everywhere
else in the engine. The T still rotates: torque comes from normal forces applied off
the centroid, which is what makes the orientation half of the task solvable.

Coupling is staggered by one step (agents see last step's T pose), and the body
integration is plain torch on the reference path, so gradients flow
T->agent->action across a rollout (BPTT); the intra-step agent-avoids-T force is not
taped (obstacles are constants inside a Warp step) — the same documented limitation as
transport, not a full in-tape rigid body.

Installing the T requires **per-env obstacle angle** (``[n_envs, n_obstacles]``), since
the body rotates independently in every env; see
:meth:`wmas.core.stepper.Stepper.set_obstacles`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from wmas.core.config import ObstacleShape, WorldConfig
from wmas.core.state import VEC2
from wmas.core.world import World
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from wmas.scenarios.base import Scenario
from wmas.scenarios.pusht_kernels import (
    pusht_body_kernel,
    pusht_obs_kernel,
    pusht_reward_kernel,
)


class PushTScenario(Scenario):
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
        contact_k: float = 100.0,
        contact_c: float = 5.0,
        contact_margin: float = 0.01,
        linear_damping: float = 10.0,
        angular_damping: float = 10.0,
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
        self.contact_k = contact_k
        self.contact_c = contact_c
        self.contact_margin = contact_margin
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

    def make_world(self, n_envs, device, dt, substeps, dtype) -> World:
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
        )
        self.dt = dt
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        tt = {"device": device, "dtype": dtype}
        # Env-independent obstacle attributes (both boxes, every env).
        self._obs_shape = torch.full((self.n_boxes,), int(ObstacleShape.BOX), device=device,
                                     dtype=torch.int32)
        self._obs_radius = torch.zeros(self.n_boxes, **tt)  # a box ignores radius
        self._obs_half = torch.tensor(self.box_half, **tt)
        self._box_off_t = torch.tensor(self.box_off, **tt)  # [n_boxes, 2]
        self.tee_pos: torch.Tensor | None = None  # [n_envs, 2]
        self.tee_vel: torch.Tensor | None = None
        self.tee_theta: torch.Tensor | None = None  # [n_envs]
        self.tee_ang_vel: torch.Tensor | None = None
        self.goal_pos: torch.Tensor | None = None  # [n_envs, 2]
        self.goal_theta: torch.Tensor | None = None  # [n_envs]
        self._prev_dist: torch.Tensor | None = None
        self._prev_ang: torch.Tensor | None = None
        self._prev_adist: torch.Tensor | None = None  # [n_envs, n_agents]
        self._cache: dict[str, torch.Tensor] | None = None
        self._fused_ready = False
        # Bumped by _sync_fused_handles when a cached buffer handle is rebuilt.
        self._handle_version = 0
        return self.world

    def fused_available(self) -> bool:
        """Push-T ships fused Warp obs/reward + movable-body kernels."""
        return True

    @property
    def obs_dim(self) -> int:
        # 12 own/task features + the other agents' relative positions (coordination).
        return 12 + 2 * (self.n_agents - 1)

    # ------------------------------------------------------------------ reset

    def reset_world(self, env_mask: torch.Tensor | None = None) -> None:
        w = self.world
        n = w.n_envs
        lim = self.world_size - 2.0 * self.agent_radius
        tlim = self.world_size - self.tee_radius
        spawn = w.sample_uniform((n, self.n_agents, 2), -lim, lim)
        tee = w.sample_uniform((n, 2), -tlim, tlim)
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
        zeros_2 = torch.zeros(n, 2, device=w.device, dtype=w.dtype)
        zeros_1 = torch.zeros(n, device=w.device, dtype=w.dtype)

        if self.tee_pos is None:
            self.tee_pos = zeros_2.clone()
            self.tee_vel = zeros_2.clone()
            self.tee_theta = zeros_1.clone()
            self.tee_ang_vel = zeros_1.clone()
            self.goal_pos = zeros_2.clone()
            self.goal_theta = zeros_1.clone()

        # In-place body updates (copy_) so the fused path's cached wp handles and the
        # whole-step graph stay valid across resets; the grad path's _refresh still
        # reassigns them (fresh tensors for the tape), which _sync_fused_handles
        # catches on the next no-grad step.
        if env_mask is None:
            w.state.pos.data.copy_(spawn)
            w.state.vel.data.zero_()
            self.tee_pos.copy_(tee)
            self.tee_vel.zero_()
            self.tee_theta.copy_(theta)
            self.tee_ang_vel.zero_()
            self.goal_pos.copy_(goal)
            self.goal_theta.copy_(goal_th)
        else:
            m3 = env_mask.view(-1, 1, 1)
            m2 = env_mask.view(-1, 1)
            w.state.pos.data.copy_(torch.where(m3, spawn, w.state.pos.data))
            w.state.vel.data.copy_(
                torch.where(m3, torch.zeros_like(w.state.vel.data), w.state.vel.data)
            )
            self.tee_pos.copy_(torch.where(m2, tee, self.tee_pos))
            self.tee_vel.copy_(torch.where(m2, zeros_2, self.tee_vel))
            self.tee_theta.copy_(torch.where(env_mask, theta, self.tee_theta))
            self.tee_ang_vel.copy_(torch.where(env_mask, zeros_1, self.tee_ang_vel))
            self.goal_pos.copy_(torch.where(m2, goal, self.goal_pos))
            self.goal_theta.copy_(torch.where(env_mask, goal_th, self.goal_theta))

        self._install_obstacles()
        if self._fused_active:
            self._ensure_fused(w.n_envs)
            self._sync_fused_handles()  # this eager _launch_* uses cached handles
            if env_mask is None:
                self._f_resetmask.fill_(1)
            else:
                self._f_resetmask.copy_(env_mask)  # bool -> uint8
            full = 0 if self._fused_obs_only else 1
            # No body integration on reset: recompute obs and rebase the shaping
            # baselines only. reset_hit rebases prev_dist/prev_ang.
            self._launch_obs()
            self._launch_reward(advance_prev=0, full_pass=full)
        else:
            self._prev_dist = None
            self._prev_ang = None
            self._prev_adist = None
            self._refresh(reset_mask=env_mask, integrate=False)

    def _box_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The two box centres ``[n_envs, n_boxes, 2]`` and angles ``[n_envs, n_boxes]``
        for the current T pose (torch; the fused path does this inside the body kernel)."""
        th = self.tee_theta.detach()
        ca, sa = torch.cos(th), torch.sin(th)  # [E]
        ox, oy = self._box_off_t[:, 0], self._box_off_t[:, 1]  # [B]
        # rotate each local offset into world: R(theta) @ off
        wx = ca.unsqueeze(1) * ox - sa.unsqueeze(1) * oy  # [E, B]
        wy = sa.unsqueeze(1) * ox + ca.unsqueeze(1) * oy
        centers = self.tee_pos.detach().unsqueeze(1) + torch.stack([wx, wy], dim=-1)
        return centers, th.unsqueeze(1).expand(-1, self.n_boxes)

    def _install_obstacles(self) -> None:
        centers, angles = self._box_poses()
        self.world.set_obstacles(
            centers,
            self._obs_radius,
            shape=self._obs_shape,
            angle=angles,  # per-env: the T rotates independently in each env
            half_extents=self._obs_half,
        )

    # -------------------------------------------------------------- T physics

    def post_step(self) -> None:
        if self._fused_active:
            # Same sequence the whole-step graph runs (keeps non-graph fused mode
            # and CPU eager-persistent bit-identical).
            self._pre_graph_step()
            self._graph_post_physics()
        else:
            self._refresh(integrate=True)

    # --------------------------------------------------------- fused fast path

    _BODY_ATTRS = ("tee_pos", "tee_vel", "tee_theta", "tee_ang_vel", "goal_pos", "goal_theta")

    def _ensure_fused(self, n_envs: int) -> None:
        """Allocate the persistent fused OUTPUT buffers + cached wp handles.

        Body state (tee_*/goal_*) and the two shaping carries ARE cached here (reset
        updates them in place); the grad path reassigns them, and _sync_fused_handles
        rebuilds any handle whose data_ptr moves."""
        if self._fused_ready:
            return
        w = self.world
        na, dev, dt = self.n_agents, w.device, w.dtype

        def z(*shape, d=dt):
            return torch.zeros(*shape, device=dev, dtype=d)

        self._f_obs = z(n_envs, na, self.obs_dim)
        self._f_reward = z(n_envs, na)
        self._f_dist = z(n_envs)
        self._f_ang = z(n_envs)
        self._f_done = z(n_envs, d=torch.uint8)
        self._f_resetmask = z(n_envs, d=torch.uint8)
        if self._prev_dist is None:
            self._prev_dist = z(n_envs)
        if self._prev_ang is None:
            self._prev_ang = z(n_envs)
        if self._prev_adist is None:
            self._prev_adist = z(n_envs, na)
        scalar = w.wp_dtype
        vec2 = VEC2[scalar]
        self._wp = {
            "obs": wp.from_torch(self._f_obs, dtype=scalar),
            "reward": wp.from_torch(self._f_reward, dtype=scalar),
            "dist": wp.from_torch(self._f_dist, dtype=scalar),
            "ang": wp.from_torch(self._f_ang, dtype=scalar),
            "done": wp.from_torch(self._f_done, dtype=wp.uint8),
            "resetmask": wp.from_torch(self._f_resetmask, dtype=wp.uint8),
        }
        self._f_done_bool = self._f_done.view(torch.bool)
        # Static box geometry, uploaded once.
        self._wp_box_off = wp.from_torch(self._box_off_t.contiguous(), dtype=vec2)
        self._wp_box_half = wp.from_torch(self._obs_half.contiguous(), dtype=vec2)
        self._body_dtype = {
            "tee_pos": vec2, "tee_vel": vec2, "tee_theta": scalar,
            "tee_ang_vel": scalar, "goal_pos": vec2, "goal_theta": scalar,
        }
        self._wp_body = {
            k: wp.from_torch(getattr(self, k).contiguous(), dtype=self._body_dtype[k])
            for k in self._BODY_ATTRS
        }
        self._wp_prev = wp.from_torch(self._prev_dist.contiguous(), dtype=scalar)
        self._wp_prev_ang = wp.from_torch(self._prev_ang.contiguous(), dtype=scalar)
        self._wp_prev_adist = wp.from_torch(self._prev_adist.contiguous(), dtype=scalar)
        self._body_ptrs = {k: getattr(self, k).data_ptr() for k in self._BODY_ATTRS}
        self._prev_ptr = self._prev_dist.data_ptr()
        self._prev_ang_ptr = self._prev_ang.data_ptr()
        self._prev_adist_ptr = self._prev_adist.data_ptr()
        self._fused_ready = True

    def _sync_fused_handles(self) -> None:
        """Rebuild any cached body/carry handle whose backing tensor was reallocated
        (grad-path reassignment); bumps ``_handle_version`` to force recapture. Cheap
        pointer compare in steady state; runs outside capture."""
        scalar = self.world.wp_dtype
        changed = False
        for k in self._BODY_ATTRS:
            t = getattr(self, k)
            if t.data_ptr() != self._body_ptrs[k]:
                self._wp_body[k] = wp.from_torch(t.contiguous(), dtype=self._body_dtype[k])
                self._body_ptrs[k] = t.data_ptr()
                changed = True
        if self._prev_dist.data_ptr() != self._prev_ptr:
            self._wp_prev = wp.from_torch(self._prev_dist.contiguous(), dtype=scalar)
            self._prev_ptr = self._prev_dist.data_ptr()
            changed = True
        if self._prev_ang.data_ptr() != self._prev_ang_ptr:
            self._wp_prev_ang = wp.from_torch(self._prev_ang.contiguous(), dtype=scalar)
            self._prev_ang_ptr = self._prev_ang.data_ptr()
            changed = True
        if self._prev_adist.data_ptr() != self._prev_adist_ptr:
            self._wp_prev_adist = wp.from_torch(self._prev_adist.contiguous(), dtype=scalar)
            self._prev_adist_ptr = self._prev_adist.data_ptr()
            changed = True
        if changed:
            self._handle_version += 1

    # ----------------------------------------------------- whole-step graph

    def graph_capturable(self) -> bool:
        return True

    def graph_recapture_token(self) -> int:
        return self._handle_version

    def _graph_warmup_carries(self) -> list[torch.Tensor]:
        # _launch_body advances the body state in place and overwrites the stepper's
        # obstacle pose buffers (read by the next step's physics); _launch_reward
        # advances both shaping carries. Snapshot all of them so warm-up (which runs
        # the hook only to compile kernels) never advances them. Unlike transport we
        # must also snapshot _obs_angle — the T's orientation lives there.
        st = self.world.stepper
        return [
            self.tee_pos, self.tee_vel, self.tee_theta, self.tee_ang_vel,
            self._prev_dist, self._prev_ang, self._prev_adist,
            wp.to_torch(st._obs_pos), wp.to_torch(st._obs_angle),
        ]

    def _pre_graph_step(self) -> None:
        self._ensure_fused(self.world.n_envs)
        self._f_resetmask.zero_()  # a normal step resets no env
        self._sync_fused_handles()

    def _graph_post_physics(self) -> None:
        # _launch_body also re-installs the obstacle pose for the next step, so
        # there is no host-side set_obstacles call inside the graph.
        self._launch_body()
        self._launch_obs()
        self._launch_reward(advance_prev=1, full_pass=1)

    def _state_wp(self):
        """(pos, vel) agent state as Warp arrays for the fused kernels."""
        w = self.world
        vec2 = VEC2[w.wp_dtype]
        if w._persistent and not w._detached:
            s = w.runtime.state
            return s.pos, s.vel
        st = w.state
        return (
            wp.from_torch(st.pos.contiguous(), dtype=vec2),
            wp.from_torch(st.vel.contiguous(), dtype=vec2),
        )

    def _body_wp(self) -> dict:
        """Cached Warp handles over the body tensors (rebuilt by _sync_fused_handles
        when the grad path reassigns them)."""
        return self._wp_body

    def _launch_body(self) -> None:
        w = self.world
        scalar = w.wp_dtype
        pos, vel = self._state_wp()
        bd = self._body_wp()
        st = w.stepper
        wp.launch(
            pusht_body_kernel,
            dim=w.n_envs,
            inputs=[
                pos,
                vel,
                self._wp_box_off,
                self._wp_box_half,
                wp.int32(self.n_agents),
                wp.int32(self.n_boxes),
                scalar(self.agent_radius),
                scalar(self.contact_margin),
                scalar(self.contact_k),
                scalar(self.contact_c),
                scalar(self.tee_mass),
                scalar(self.tee_inertia),
                scalar(self.linear_damping),
                scalar(self.angular_damping),
                scalar(self.dt),
                scalar(self.world_size - self.tee_radius),
            ],
            outputs=[
                bd["tee_pos"],
                bd["tee_vel"],
                bd["tee_theta"],
                bd["tee_ang_vel"],
                st._obs_pos,
                st._obs_angle,
            ],
            device=w.device,
            record_tape=False,
        )

    def _launch_obs(self) -> None:
        w = self.world
        self._ensure_fused(w.n_envs)
        pos, vel = self._state_wp()
        bd = self._body_wp()
        wp.launch(
            pusht_obs_kernel,
            dim=(w.n_envs, self.n_agents),
            inputs=[
                pos,
                vel,
                bd["tee_pos"],
                bd["tee_theta"],
                bd["goal_pos"],
                bd["goal_theta"],
            ],
            outputs=[self._wp["obs"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_reward(self, advance_prev: int, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        bd = self._body_wp()
        pos, _ = self._state_wp()
        wp.launch(
            pusht_reward_kernel,
            dim=w.n_envs,
            inputs=[
                pos,
                bd["tee_pos"],
                bd["tee_theta"],
                bd["goal_pos"],
                bd["goal_theta"],
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
                self._wp_prev,
                self._wp_prev_ang,
                self._wp_prev_adist,
                self._wp["reward"],
                self._wp["done"],
                self._wp["dist"],
                self._wp["ang"],
            ],
            device=w.device,
            record_tape=False,
        )

    # ------------------------------------------- torch reference (differentiable)

    def _box_contact(self, pos: torch.Tensor, vel: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Per (env, agent, box) oriented-box SDF contact, mirroring
        :func:`wmas.core.collisions._box_force` (exterior clamp + interior nearest
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
        coeff = (self.contact_k * overlap.clamp(min=0.0) - self.contact_c * vn) * active
        fx, fy = coeff * mx, coeff * my
        return torch.stack([fx, fy], dim=-1), rx * fy - ry * fx

    def _refresh(self, reset_mask: torch.Tensor | None = None, integrate: bool = True) -> None:
        w = self.world
        pos, vel = w.state.pos, w.state.vel  # [n_envs, n_agents, 2]

        if integrate:
            force, torque = self._box_contact(pos, vel)  # [E,A,B,2], [E,A,B]
            f_total = force.sum(dim=(1, 2))  # [E, 2] net force on the T
            tau = torque.sum(dim=(1, 2))  # [E]

            dt = self.dt
            self.tee_vel = (self.tee_vel + f_total / self.tee_mass * dt) * (
                1.0 - self.linear_damping * dt
            )
            self.tee_ang_vel = (self.tee_ang_vel + tau / self.tee_inertia * dt) * (
                1.0 - self.angular_damping * dt
            )
            b = self.world_size - self.tee_radius
            self.tee_pos = (self.tee_pos + self.tee_vel * dt).clamp(-b, b)
            self.tee_theta = self.tee_theta + self.tee_ang_vel * dt
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
        if self._fused_active:
            return self._f_obs
        w = self.world
        s = w.state
        na = w.n_agents
        tee_to_goal = (self.goal_pos - self.tee_pos).unsqueeze(1).expand(-1, na, -1)
        raw = (self.tee_theta - self.goal_theta).unsqueeze(1).expand(-1, na)
        th = self.tee_theta.unsqueeze(1).expand(-1, na)
        angles = torch.stack([th.cos(), th.sin(), raw.cos(), raw.sin()], dim=-1)
        # Teammates' positions relative to each agent, ascending index, self skipped.
        idx = torch.arange(na, device=s.pos.device)
        others = idx.unsqueeze(0).expand(na, -1)[idx.unsqueeze(1) != idx.unsqueeze(0)]
        others = others.view(na, na - 1)  # [A, A-1]
        rel = (s.pos[:, others] - s.pos.unsqueeze(2)).flatten(2)  # [E, A, 2(A-1)]
        return torch.cat(
            [s.pos, s.vel, self._cache["tee_rel"], tee_to_goal, angles, rel], dim=-1
        )

    def observation(self, agent_idx: int) -> torch.Tensor:
        return self.observations()[:, agent_idx]

    def global_reward(self) -> torch.Tensor:
        c = self._cache
        return c["shaping"] + self.goal_reward * c["on_goal"].to(self.world.dtype)

    def rewards(self) -> torch.Tensor:
        if self._fused_active:
            return self._f_reward
        return self._cache["agent_shaping"] + self.global_reward().unsqueeze(1)

    def done(self) -> torch.Tensor:
        if self._fused_active:
            return self._f_done_bool
        return self._cache["on_goal"]

    def render_extras(self, env_idx: int) -> dict[str, Any]:
        """The target pose of the T, as oriented boxes for the ``goal_pose`` overlay.

        Same two boxes the body is built from, placed at ``(goal_pos, goal_theta)``:
        the outline shows exactly where the T has to end up.
        """
        if self.goal_pos is None:
            return {}
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
        if self._fused_active:
            return {"tee_dist_to_goal": self._f_dist, "tee_angle_error": self._f_ang}
        return {
            "tee_dist_to_goal": self._cache["dist_to_goal"],
            "tee_angle_error": self._cache["angle_error"],
        }
