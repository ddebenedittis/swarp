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

The three steps above describe the **torch reference path** — the parity oracle, and the
only path that carries gradients. The no-grad hot path does the same physics in
``transport_body_kernel`` (:mod:`swarp.scenarios.transport_kernels`) instead: the force
gather, the integration and the pose write-back all happen on-device inside the fused
whole-step hook, so nothing crosses back into torch per step and the whole step stays
capturable as one CUDA graph. The two are tested against each other in
``tests/scenarios/test_transport_fused.py``.
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from swarp._overloads import concrete
from swarp.core.config import Obstacles, WorldConfig
from swarp.core.state import VEC2
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.interop.autograd import torch_stream_scope
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario
from swarp.scenarios.transport_kernels import (
    transport_body_kernel,
    transport_obs_kernel,
    transport_reset_kernel,
    transport_reward_kernel,
)


class TransportScenario(FusedScenario):
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
        # contact_k/contact_c ARE the engine's WorldConfig.collision_k/collision_c —
        # the same spring-damper law, named for the agent<->package contact that dominates
        # here. contact_margin is NOT the engine's collision_margin (which is derived
        # from agent_radius in make_world): it is this scenario's own agent<->package
        # activation gap, used by the torch reference path below.
        self.contact_k = contact_k
        self.contact_c = contact_c
        self.contact_margin = contact_margin
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.pos_shaping_factor = pos_shaping_factor
        self.goal_reward = goal_reward
        self.goal_tolerance = goal_tolerance if goal_tolerance is not None else package_radius

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
        ).override_with(world_config)
        self.dt = dt
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        # Package state and goal: allocated here (n_envs is known) so the fused spec can
        # adopt them, and written in place by every reset so their handles stay valid.
        tt = {"device": device, "dtype": dtype}
        self.pkg_pos = torch.zeros(n_envs, self.n_packages, 2, **tt)
        self.pkg_vel = torch.zeros(n_envs, self.n_packages, 2, **tt)
        self.pkg_theta = torch.zeros(n_envs, self.n_packages, **tt)  # [n_envs, n_packages]
        self.pkg_ang_vel = torch.zeros(n_envs, self.n_packages, **tt)
        self.goal = torch.zeros(n_envs, self.n_packages, 2, **tt)
        # ONE retained obstacle spec over pkg_pos, re-installed rather than rebuilt: both
        # ``resolve`` and ``any_movable`` memoize, so a re-install allocates nothing and
        # costs no device->host sync. Built here so the single count-changing install
        # happens before any graph capture. It aliases pkg_pos, so writing new package
        # poses in place is all a "move the packages" update needs.
        self._pkg_radius = torch.full((self.n_packages,), self.package_radius, **tt)
        self._obstacles = Obstacles(self.pkg_pos.detach(), self._pkg_radius).resolve(device, dtype)
        self._prev_dist: torch.Tensor | None = None
        self._cache: dict[str, torch.Tensor] | None = None
        return self.world

    @property
    def obs_dim(self) -> int:
        return 4 + 4 * self.n_packages

    # ------------------------------------------------------------------ reset

    def reset_world(
        self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False
    ) -> None:
        """Masked reset in one Warp launch (see :mod:`swarp.scenarios.reset_kernels`).

        The package buffers are written **in place** by the kernel, so the fused path's
        cached Warp handles and the whole-step graph stay valid across resets; the grad
        path's ``_refresh`` still reassigns them (fresh tensors for the tape), which the
        framework's handle resync catches on the next no-grad step.
        """
        w = self.world
        lim = self.world_size - 2.0 * self.agent_radius
        plim = self.world_size - self.package_radius
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
            pkg_pos = self._wp["pkg_pos"]
            pkg_vel = self._wp["pkg_vel"]
            pkg_theta = self._wp["pkg_theta"]
            pkg_ang_vel = self._wp["pkg_ang_vel"]
            goal = self._wp["goal"]
        else:
            pkg_pos = wp.from_torch(self.pkg_pos.contiguous(), dtype=vec2)
            pkg_vel = wp.from_torch(self.pkg_vel.contiguous(), dtype=vec2)
            pkg_theta = wp.from_torch(self.pkg_theta.contiguous())
            pkg_ang_vel = wp.from_torch(self.pkg_ang_vel.contiguous())
            goal = wp.from_torch(self.goal.contiguous(), dtype=vec2)
        with torch_stream_scope(w.device):
            wp.launch(
                concrete(transport_reset_kernel, scalar),
                dim=w.n_envs,
                inputs=[
                    mask,
                    use_mask,
                    wp.int32(w.next_kernel_seed()),
                    scalar(lim),
                    scalar(plim),
                    wp.int32(self.n_agents),
                    wp.int32(self.n_packages),
                    st.pos,
                    st.vel,
                    pkg_pos,
                    pkg_vel,
                    pkg_theta,
                    pkg_ang_vel,
                    goal,
                ],
                device=w.device,
                record_tape=False,
            )
        w.mark_pos_dirty()

        self._install_obstacles()
        if not self.fused_active:
            self._prev_dist = None
        self.finish_reset(env_mask, obs_only=obs_only)

    def _install_obstacles(self) -> None:
        """Hand the current package poses to the engine as circular obstacles.

        Called from inside the captured whole-step graph (see :meth:`launch_fused`), so it
        must stay in the capture-safe regime of :meth:`swarp.core.stepper.Stepper.
        set_obstacles`: an unchanged obstacle count, and a spec carrying no movable
        obstacles and no 1-D ``angle`` to broadcast.

        The spec is built once, in ``make_world``, and **re-installed** rather than
        rebuilt: it aliases ``pkg_pos``, and both ``Obstacles.resolve`` and
        ``Obstacles.any_movable`` memoize, so a re-install allocates nothing and costs no
        device->host sync. Building a fresh spec per call — as this used to — put
        ``any_movable``'s reduction plus ``.item()`` on every step that reset an env under
        ``auto_reset=True``.

        The retained spec is rebuilt only when ``pkg_pos`` is *reassigned*, which is the
        grad path integrating the package in torch (it needs fresh tensors for the tape).
        That is a host-side pointer compare, and on the captured path the pointer never
        moves, so the rebuild branch cannot fire inside a capture.
        """
        if self._obstacles.pos.data_ptr() != self.pkg_pos.data_ptr():
            self._obstacles = Obstacles(self.pkg_pos.detach(), self._pkg_radius).resolve(
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
        ne, na, k = n_envs, self.n_agents, self.n_packages
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("reward", (ne, na)),
            Buf("dist", (ne, k)),
            Buf("done", (ne,), "uint8", bool_view=True),
            Buf("resetmask", (ne,), "uint8", reset_mask=True),
            # Package state and goal: the scenario's own tensors, advanced in place by the
            # body kernel and reassigned by the grad path's torch integrator.
            Buf("pkg_pos", (ne, k, 2), "vec2", attr="pkg_pos", alloc="never", carry=True,
                watch=True),
            Buf("pkg_vel", (ne, k, 2), "vec2", attr="pkg_vel", alloc="never", carry=True,
                watch=True),
            Buf("pkg_theta", (ne, k), attr="pkg_theta", alloc="never", carry=True, watch=True),
            Buf("pkg_ang_vel", (ne, k), attr="pkg_ang_vel", alloc="never", carry=True,
                watch=True),
            Buf("goal", (ne, k, 2), "vec2", attr="goal", alloc="never", watch=True),
            Buf("prev", (ne, k), attr="_prev_dist", alloc="if_none", carry=True, watch=True),
        )

    def engine_carries(self) -> list[torch.Tensor]:
        """The engine's obstacle positions: ``_install_obstacles`` overwrites them from
        inside the hook, and the *next* step's physics reads them."""
        return [self.world.obstacle_state_views()[0]]

    def launch_fused(self, pass_: FusedPass) -> None:
        """Integrate the package, re-install its pose, then obs and reward.

        A reset skips the body integration (there is nothing to advance: the package was
        just placed) and passes ``advance_prev=0`` so the reward kernel *rebases* the
        shaping baseline for the reset envs instead of differencing against a stale one.
        The ``_install_obstacles`` in the middle is an ordinary line here, running inside
        ``wp.ScopedCapture`` on a step — see its docstring for why that is legal.
        """
        st = self.world.state_wp()  # one wrap for every launch in this pass
        if pass_.is_step:
            self._launch_body(st)
            self._install_obstacles()  # updated package pose for the next step
        self._launch_obs(st)
        self._launch_reward(advance_prev=pass_.advance_prev, full_pass=pass_.full_pass)

    def _launch_body(self, st) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        bound = self.world_size - self.package_radius
        wp.launch(
            concrete(transport_body_kernel, self.world.wp_dtype),
            dim=(w.n_envs, self.n_packages),
            inputs=[
                st.pos,
                st.vel,
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

    def _launch_obs(self, st) -> None:
        w = self.world
        wp.launch(
            concrete(transport_obs_kernel, self.world.wp_dtype),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                self._wp["pkg_pos"],
                self._wp["goal"],
                wp.int32(self.n_packages),
            ],
            outputs=[self._wp["obs"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_reward(self, advance_prev: int, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        wp.launch(
            concrete(transport_reward_kernel, self.world.wp_dtype),
            dim=w.n_envs,
            inputs=[
                self._wp["pkg_pos"],
                self._wp["goal"],
                self._wp["resetmask"],
                wp.int32(self.n_agents),
                wp.int32(self.n_packages),
                scalar(self.pos_shaping_factor),
                scalar(self.goal_tolerance),
                scalar(self.goal_reward),
                wp.int32(advance_prev),
                wp.int32(full_pass),
            ],
            outputs=[self._wp["prev"], self._wp["reward"], self._wp["done"], self._wp["dist"]],
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
        if self.fused_active:
            return self.fb["obs"]
        w = self.world
        s = w.state
        pkg_to_goal = (
            (self.goal - self.pkg_pos)
            .reshape(w.n_envs, 1, self.n_packages * 2)
            .expand(-1, w.n_agents, -1)
        )
        return torch.cat([s.pos, s.vel, self._cache["pkg_rel"], pkg_to_goal], dim=-1)

    def global_reward(self) -> torch.Tensor:
        c = self._cache
        return c["shaping"] + self.goal_reward * c["on_goal"].all(dim=-1).to(self.world.dtype)

    def rewards(self) -> torch.Tensor:
        return self.fb["reward"] if self.fused_active else super().rewards()

    def done(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["done_bool"]
        return self._cache["on_goal"].all(dim=-1)

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {"package_dist_to_goal": self.fb["dist"]}
        return {"package_dist_to_goal": self._cache["dist_to_goal"]}
