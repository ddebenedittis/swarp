"""Transport: agents push movable package(s) to a goal.

A first movable-body model built at the scenario/torch layer (no core-kernel or
autograd-bridge changes). Each step:

  1. the packages are installed as circular obstacles, so the Warp agent step
     pushes agents *off* them (the existing soft agent-obstacle contact);
  2. after the step, the reaction contact force each package receives from the
     agents (Newton's third law of that same spring-damper contact) is summed,
     gather-style, into a net force + torque, and the package is integrated in
     torch (semi-implicit Euler);
  3. the updated package pose is installed for the next step.

Coupling is thus staggered by one step (agents see last step's package pose).
The package integration is plain torch, so gradients flow package->agent->action
across a rollout (BPTT); the intra-step agent-avoids-package force is not taped
(obstacles are constants inside a Warp step), a documented limitation of this
first pass vs. a full in-tape rigid body.
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from wmas.core.config import WorldConfig
from wmas.core.state import VEC2
from wmas.core.world import World
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from wmas.scenarios.base import Scenario
from wmas.scenarios.transport_kernels import (
    transport_body_kernel,
    transport_obs_kernel,
    transport_reward_kernel,
)


class TransportScenario(Scenario):
    def __init__(
        self,
        n_agents: int = 4,
        n_packages: int = 1,
        agent_radius: float = 0.05,
        package_radius: float = 0.15,
        package_mass: float = 1.0,
        world_size: float = 1.0,
        max_speed: float = 1.0,
        contact_k: float = 100.0,
        contact_c: float = 5.0,
        contact_margin: float = 0.01,
        linear_damping: float = 0.5,
        angular_damping: float = 0.5,
        pos_shaping_factor: float = 1.0,
        goal_reward: float = 5.0,
        goal_tolerance: float | None = None,
    ) -> None:
        self.n_agents = n_agents
        self.n_packages = n_packages
        self.agent_radius = agent_radius
        self.package_radius = package_radius
        self.package_mass = package_mass
        # Solid-disk moment of inertia 0.5*m*r^2.
        self.package_inertia = 0.5 * package_mass * package_radius**2
        self.world_size = world_size
        self.max_speed = max_speed
        self.contact_k = contact_k
        self.contact_c = contact_c
        self.contact_margin = contact_margin
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.pos_shaping_factor = pos_shaping_factor
        self.goal_reward = goal_reward
        self.goal_tolerance = goal_tolerance if goal_tolerance is not None else package_radius

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
        # neighbor reach must cover agent<->package contact so the step sees it
        reach = 2.0 * max(self.agent_radius, self.package_radius) + margin
        cfg = WorldConfig(
            collisions=True,
            collision_k=self.contact_k,
            collision_c=self.contact_c,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            neighbor_radius=reach,
            max_neighbors=min(32, max(4, self.n_agents)),
        )
        self.dt = dt
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        self._pkg_radius = torch.full(
            (self.n_packages,), self.package_radius, device=device, dtype=dtype
        )
        self.pkg_pos: torch.Tensor | None = None  # [n_envs, n_packages, 2]
        self.pkg_vel: torch.Tensor | None = None
        self.pkg_theta: torch.Tensor | None = None  # [n_envs, n_packages]
        self.pkg_ang_vel: torch.Tensor | None = None
        self.goal: torch.Tensor | None = None  # [n_envs, n_packages, 2]
        self._prev_dist: torch.Tensor | None = None
        self._cache: dict[str, torch.Tensor] | None = None
        self._fused_ready = False
        return self.world

    def fused_available(self) -> bool:
        """Transport ships fused Warp obs/reward + movable-body kernels."""
        return True

    @property
    def obs_dim(self) -> int:
        return 4 + 4 * self.n_packages

    # ------------------------------------------------------------------ reset

    def reset_world(self, env_mask: torch.Tensor | None = None) -> None:
        w = self.world
        n = w.n_envs
        lim = self.world_size - 2.0 * self.agent_radius
        plim = self.world_size - self.package_radius
        spawn = w.sample_uniform((n, self.n_agents, 2), -lim, lim)
        pkg = w.sample_uniform((n, self.n_packages, 2), -plim, plim)
        goal = w.sample_uniform((n, self.n_packages, 2), -plim, plim)
        zeros_p2 = torch.zeros(n, self.n_packages, 2, device=w.device, dtype=w.dtype)
        zeros_p = torch.zeros(n, self.n_packages, device=w.device, dtype=w.dtype)

        if self.pkg_pos is None:
            self.pkg_pos = zeros_p2.clone()
            self.pkg_vel = zeros_p2.clone()
            self.pkg_theta = zeros_p.clone()
            self.pkg_ang_vel = zeros_p.clone()
            self.goal = zeros_p2.clone()

        if env_mask is None:
            w.state.pos.data.copy_(spawn)
            w.state.vel.data.zero_()
            self.pkg_pos = pkg
            self.pkg_vel = zeros_p2.clone()
            self.pkg_theta = zeros_p.clone()
            self.pkg_ang_vel = zeros_p.clone()
            self.goal = goal
        else:
            m3 = env_mask.view(-1, 1, 1)
            m3p = env_mask.view(-1, 1, 1)
            w.state.pos.data.copy_(torch.where(m3, spawn, w.state.pos.data))
            w.state.vel.data.copy_(
                torch.where(m3, torch.zeros_like(w.state.vel.data), w.state.vel.data)
            )
            self.pkg_pos = torch.where(m3p, pkg, self.pkg_pos)
            self.pkg_vel = torch.where(m3p, zeros_p2, self.pkg_vel)
            self.pkg_theta = torch.where(env_mask.view(-1, 1), zeros_p, self.pkg_theta)
            self.pkg_ang_vel = torch.where(env_mask.view(-1, 1), zeros_p, self.pkg_ang_vel)
            self.goal = torch.where(m3p, goal, self.goal)

        self._install_obstacles()
        if self._fused_active:
            self._ensure_fused(w.n_envs)
            if env_mask is None:
                self._f_resetmask.fill_(1)
            else:
                self._f_resetmask.copy_(env_mask)  # bool -> uint8
            full = 0 if self._fused_obs_only else 1
            # No body integration on reset (integrate=False): recompute obs and
            # rebase the shaping baseline only. reset_hit rebases prev_dist.
            self._launch_obs()
            self._launch_reward(advance_prev=0, full_pass=full)
        else:
            self._prev_dist = None
            self._refresh(reset_mask=env_mask, integrate=False)

    def _install_obstacles(self) -> None:
        self.world.set_obstacles(self.pkg_pos.detach(), self._pkg_radius)

    # -------------------------------------------------------- package physics

    def post_step(self) -> None:
        if self._fused_active:
            self._ensure_fused(self.world.n_envs)
            self._f_resetmask.zero_()  # a normal step resets no env
            self._launch_body()
            self._install_obstacles()  # updated package pose for the next step
            self._launch_obs()
            self._launch_reward(advance_prev=1, full_pass=1)
        else:
            self._refresh(integrate=True)

    # --------------------------------------------------------- fused fast path

    def _ensure_fused(self, n_envs: int) -> None:
        """Allocate the persistent fused OUTPUT buffers + wp handles.

        Package state (pkg_pos/vel/theta/ang_vel/goal) and prev_dist are NOT
        cached here — they are re-wrapped fresh each launch (the torch/grad path
        may reassign them), mirroring how navigation re-wraps its state."""
        if self._fused_ready:
            return
        w = self.world
        na, dev, dt = self.n_agents, w.device, w.dtype

        def z(*shape, d=dt):
            return torch.zeros(*shape, device=dev, dtype=d)

        self._f_obs = z(n_envs, na, self.obs_dim)
        self._f_reward = z(n_envs, na)
        self._f_dist = z(n_envs, self.n_packages)
        self._f_done = z(n_envs, d=torch.uint8)
        self._f_resetmask = z(n_envs, d=torch.uint8)
        scalar = w.wp_dtype
        self._wp = {
            "obs": wp.from_torch(self._f_obs, dtype=scalar),
            "reward": wp.from_torch(self._f_reward, dtype=scalar),
            "dist": wp.from_torch(self._f_dist, dtype=scalar),
            "done": wp.from_torch(self._f_done, dtype=wp.uint8),
            "resetmask": wp.from_torch(self._f_resetmask, dtype=wp.uint8),
        }
        self._f_done_bool = self._f_done.view(torch.bool)
        self._fused_ready = True

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

    def _pkg_wp(self) -> dict:
        """Fresh Warp handles over the (possibly reassigned) package tensors.

        The body kernel writes pkg_pos/vel/theta/ang_vel in place; these tensors
        stay contiguous (torch.where / arithmetic / clone), so the fresh wrap
        aliases them and the writes propagate."""
        scalar = self.world.wp_dtype
        vec2 = VEC2[scalar]
        return {
            "pkg_pos": wp.from_torch(self.pkg_pos.contiguous(), dtype=vec2),
            "pkg_vel": wp.from_torch(self.pkg_vel.contiguous(), dtype=vec2),
            "pkg_theta": wp.from_torch(self.pkg_theta.contiguous(), dtype=scalar),
            "pkg_ang_vel": wp.from_torch(self.pkg_ang_vel.contiguous(), dtype=scalar),
            "goal": wp.from_torch(self.goal.contiguous(), dtype=vec2),
        }

    def _launch_body(self) -> None:
        w = self.world
        scalar = w.wp_dtype
        pos, vel = self._state_wp()
        pk = self._pkg_wp()
        bound = self.world_size - self.package_radius
        wp.launch(
            transport_body_kernel,
            dim=(w.n_envs, self.n_packages),
            inputs=[
                pos,
                vel,
                wp.int32(self.n_agents),
                scalar(self.agent_radius),
                scalar(self.package_radius),
                scalar(self.contact_margin),
                scalar(self.contact_k),
                scalar(self.contact_c),
                scalar(self.package_mass),
                scalar(self.package_inertia),
                scalar(self.linear_damping),
                scalar(self.angular_damping),
                scalar(self.dt),
                scalar(bound),
            ],
            outputs=[pk["pkg_pos"], pk["pkg_vel"], pk["pkg_theta"], pk["pkg_ang_vel"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_obs(self) -> None:
        w = self.world
        self._ensure_fused(w.n_envs)
        pos, vel = self._state_wp()
        pk = self._pkg_wp()
        wp.launch(
            transport_obs_kernel,
            dim=(w.n_envs, self.n_agents),
            inputs=[pos, vel, pk["pkg_pos"], pk["goal"], wp.int32(self.n_packages)],
            outputs=[self._wp["obs"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_reward(self, advance_prev: int, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        if self._prev_dist is None:
            self._prev_dist = torch.zeros(
                w.n_envs, self.n_packages, device=w.device, dtype=w.dtype
            )
        pk = self._pkg_wp()
        prev = wp.from_torch(self._prev_dist.contiguous(), dtype=scalar)
        wp.launch(
            transport_reward_kernel,
            dim=w.n_envs,
            inputs=[
                pk["pkg_pos"],
                pk["goal"],
                self._wp["resetmask"],
                wp.int32(self.n_agents),
                wp.int32(self.n_packages),
                scalar(self.pos_shaping_factor),
                scalar(self.goal_tolerance),
                scalar(self.goal_reward),
                wp.int32(advance_prev),
                wp.int32(full_pass),
            ],
            outputs=[prev, self._wp["reward"], self._wp["done"], self._wp["dist"]],
            device=w.device,
            record_tape=False,
        )

    def _refresh(self, reset_mask: torch.Tensor | None = None, integrate: bool = True) -> None:
        w = self.world
        pos, vel = w.state.pos, w.state.vel  # [n_envs, n_agents, 2]

        if integrate:
            q = self.pkg_pos  # [n_envs, K, 2]
            rel = q.unsqueeze(1) - pos.unsqueeze(2)  # agent->package [E, A, K, 2]
            dist = rel.norm(dim=-1).clamp(min=1e-9)  # [E, A, K]
            n_hat = rel / dist.unsqueeze(-1)
            overlap = (self.agent_radius + self.package_radius + self.contact_margin) - dist
            active = (overlap > 0).to(w.dtype)
            fmag = self.contact_k * overlap.clamp(min=0.0)
            # damping along the contact normal (package approaching agent)
            rel_vel = self.pkg_vel.unsqueeze(1) - vel.unsqueeze(2)  # [E, A, K, 2]
            vn = (rel_vel * n_hat).sum(-1)
            force = (fmag - self.contact_c * vn).unsqueeze(-1) * n_hat * active.unsqueeze(-1)
            f_total = force.sum(dim=1)  # [E, K, 2] net force on each package
            # torque about package centre: contact point ~ package surface toward agent
            r_contact = -self.package_radius * n_hat  # [E, A, K, 2]
            torque = r_contact[..., 0] * force[..., 1] - r_contact[..., 1] * force[..., 0]
            tau = (torque * active).sum(dim=1)  # [E, K]

            dt = self.dt
            self.pkg_vel = (self.pkg_vel + f_total / self.package_mass * dt) * (
                1.0 - self.linear_damping * dt
            )
            self.pkg_ang_vel = (self.pkg_ang_vel + tau / self.package_inertia * dt) * (
                1.0 - self.angular_damping * dt
            )
            self.pkg_pos = self.pkg_pos + self.pkg_vel * dt
            b = self.world_size - self.package_radius
            self.pkg_pos = self.pkg_pos.clamp(-b, b)
            self.pkg_theta = self.pkg_theta + self.pkg_ang_vel * dt
            self._install_obstacles()  # for the next step

        dist_to_goal = (self.pkg_pos - self.goal).norm(dim=-1)  # [E, K]
        if self._prev_dist is None:
            self._prev_dist = dist_to_goal.detach().clone()
        shaping = (self._prev_dist - dist_to_goal) * self.pos_shaping_factor
        if reset_mask is None:
            self._prev_dist = dist_to_goal.detach().clone()
        else:
            shaping = torch.where(reset_mask.unsqueeze(-1), torch.zeros_like(shaping), shaping)
            self._prev_dist = torch.where(
                reset_mask.unsqueeze(-1), dist_to_goal.detach(), self._prev_dist
            )

        on_goal = dist_to_goal < self.goal_tolerance
        self._cache = {
            "dist_to_goal": dist_to_goal,
            "shaping": shaping.sum(dim=-1),  # [E]
            "on_goal": on_goal,
            "pkg_rel": (self.pkg_pos.unsqueeze(1) - pos.unsqueeze(2)).reshape(
                w.n_envs, w.n_agents, self.n_packages * 2
            ),
        }

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        if self._fused_active:
            return self._f_obs
        w = self.world
        s = w.state
        pkg_to_goal = (
            (self.goal - self.pkg_pos)
            .reshape(w.n_envs, 1, self.n_packages * 2)
            .expand(-1, w.n_agents, -1)
        )
        return torch.cat([s.pos, s.vel, self._cache["pkg_rel"], pkg_to_goal], dim=-1)

    def observation(self, agent_idx: int) -> torch.Tensor:
        return self.observations()[:, agent_idx]

    def global_reward(self) -> torch.Tensor:
        c = self._cache
        return c["shaping"] + self.goal_reward * c["on_goal"].all(dim=-1).to(self.world.dtype)

    def rewards(self) -> torch.Tensor:
        if self._fused_active:
            return self._f_reward
        return super().rewards()

    def done(self) -> torch.Tensor:
        if self._fused_active:
            return self._f_done_bool
        return self._cache["on_goal"].all(dim=-1)

    def info(self) -> dict[str, Any]:
        if self._fused_active:
            return {"package_dist_to_goal": self._f_dist}
        return {"package_dist_to_goal": self._cache["dist_to_goal"]}
