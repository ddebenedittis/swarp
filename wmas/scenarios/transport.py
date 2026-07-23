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

from wmas.core.config import WorldConfig
from wmas.core.world import World
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from wmas.scenarios.base import Scenario


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
        return self.world

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
        self._prev_dist = None
        self._refresh(reset_mask=env_mask, integrate=False)

    def _install_obstacles(self) -> None:
        self.world.set_obstacles(self.pkg_pos.detach(), self._pkg_radius)

    # -------------------------------------------------------- package physics

    def post_step(self) -> None:
        self._refresh(integrate=True)

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

    def done(self) -> torch.Tensor:
        return self._cache["on_goal"].all(dim=-1)

    def info(self) -> dict[str, Any]:
        return {"package_dist_to_goal": self._cache["dist_to_goal"]}
