"""Discovery: a team covers scattered targets, each needing several agents nearby.

Port of the VMAS ``discovery`` scenario. A target counts as *covered* once at
least ``agents_per_target`` agents are within ``covering_range`` of it; the team
earns a one-off shared reward the step a target is first covered. Agents also pay
a small per-step time penalty and a per-contact collision penalty. Coverage is
computed with ``torch.cdist`` over agent/target positions.
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
from wmas.scenarios.discovery_kernels import (
    discovery_cover_kernel,
    discovery_obs_kernel,
    discovery_reward_kernel,
)


class DiscoveryScenario(Scenario):
    def __init__(
        self,
        n_agents: int = 5,
        n_targets: int = 5,
        agent_radius: float = 0.05,
        world_size: float = 1.0,
        covering_range: float = 0.2,
        agents_per_target: int = 2,
        covering_reward: float = 1.0,
        time_penalty: float = -0.01,
        collision_penalty: float = -0.1,
        max_speed: float = 1.0,
    ) -> None:
        self.n_agents = n_agents
        self.n_targets = n_targets
        self.agent_radius = agent_radius
        self.world_size = world_size
        self.covering_range = covering_range
        self.agents_per_target = agents_per_target
        self.covering_reward = covering_reward
        self.time_penalty = time_penalty
        self.collision_penalty = collision_penalty
        self.max_speed = max_speed

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
        cfg = WorldConfig(
            collisions=True,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            max_neighbors=min(8, max(2, self.n_agents)),
        )
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        self.targets: torch.Tensor | None = None  # [n_envs, n_targets, 2]
        self.covered: torch.Tensor | None = None  # [n_envs, n_targets] bool (ever covered)
        self._cache: dict[str, torch.Tensor] | None = None
        self._fused_ready = False
        return self.world

    def fused_available(self) -> bool:
        """Discovery ships fused Warp obs/reward kernels (2D holonomic)."""
        return True

    @property
    def obs_dim(self) -> int:
        return 4 + 3 * self.n_targets

    def reset_world(self, env_mask: torch.Tensor | None = None) -> None:
        w = self.world
        n = w.n_envs
        lim = self.world_size - 2.0 * self.agent_radius
        spawn = w.sample_uniform((n, self.n_agents, 2), -lim, lim)
        targets = w.sample_uniform((n, self.n_targets, 2), -lim, lim)
        covered = torch.zeros(n, self.n_targets, dtype=torch.bool, device=w.device)

        if self.targets is None:
            self.targets = torch.zeros(n, self.n_targets, 2, device=w.device, dtype=w.dtype)
            self.covered = torch.zeros(n, self.n_targets, dtype=torch.bool, device=w.device)

        if env_mask is None:
            w.state.pos.data.copy_(spawn)
            w.state.vel.data.zero_()
            self.targets.copy_(targets)
            self.covered.copy_(covered)
        else:
            m3 = env_mask.view(-1, 1, 1)
            w.state.pos.data.copy_(torch.where(m3, spawn, w.state.pos.data))
            w.state.vel.data.copy_(
                torch.where(m3, torch.zeros_like(w.state.vel.data), w.state.vel.data)
            )
            self.targets.copy_(torch.where(m3, targets, self.targets))
            self.covered.copy_(torch.where(env_mask.view(-1, 1), covered, self.covered))
        if self._fused_active:
            self._ensure_fused(w.n_envs)
            # A reset recomputes coverage/obs on the new state (updating the
            # covered latch, as the reference _refresh does) but never the reward
            # (the transition's reward buffer was already returned this step).
            self._launch_cover()
            self._launch_obs()
        else:
            self._refresh()

    def post_step(self) -> None:
        if self._fused_active:
            self._ensure_fused(self.world.n_envs)
            self._launch_cover()
            self._launch_obs()
            self._launch_reward()
        else:
            self._refresh()

    def _refresh(self) -> None:
        w = self.world
        pos = w.state.pos  # [n_envs, n_agents, 2]
        # Squared broadcast distances (no sqrt) so the reference's discrete
        # coverage/touching flags match the fused kernels' dx*dx+dy*dy bit-for-bit
        # at the threshold; torch.cdist's matmul path rounds differently.
        diff_t = pos.unsqueeze(2) - self.targets.unsqueeze(1)  # [n_envs, n_agents, n_targets, 2]
        dt2 = (diff_t * diff_t).sum(-1)  # [n_envs, n_agents, n_targets]
        within = dt2 < self.covering_range**2
        count = within.sum(dim=1)  # [n_envs, n_targets] agents near each target
        covered_now = count >= self.agents_per_target
        newly = covered_now & ~self.covered  # [n_envs, n_targets]
        self.covered |= covered_now  # in-place latch (storage stable for the fused path)

        # agent-agent contacts (undirected touching count per agent)
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dd2 = (diff * diff).sum(-1)
        touch = (dd2 < (2.0 * self.agent_radius) ** 2).sum(dim=-1).to(w.dtype) - 1.0  # minus self

        self._cache = {
            "newly": newly,
            "count": count,
            "touch": touch,
            "rel_targets": (self.targets.unsqueeze(1) - pos.unsqueeze(2)).reshape(
                w.n_envs, w.n_agents, self.n_targets * 2
            ),
            "covered_frac": self.covered.float().mean(-1),
        }

    # --------------------------------------------------------- fused fast path

    def _ensure_fused(self, n_envs: int) -> None:
        """Allocate the persistent fused output buffers + cached wp handles.

        ``targets``/``covered`` are allocated by ``reset_world`` before this runs
        and only ever updated in place, so their data_ptr is stable."""
        if self._fused_ready:
            return
        w = self.world
        na, dev, dt = self.n_agents, w.device, w.dtype

        def z(*shape, d=dt):
            return torch.zeros(*shape, device=dev, dtype=d)

        self._f_obs = z(n_envs, na, self.obs_dim)
        self._f_touch = z(n_envs, na)
        self._f_reward = z(n_envs, na)
        self._f_newly = z(n_envs, self.n_targets, d=torch.uint8)
        self._f_done = z(n_envs, d=torch.uint8)
        scalar = w.wp_dtype
        self._wp = {
            "obs": wp.from_torch(self._f_obs, dtype=scalar),
            "touch": wp.from_torch(self._f_touch, dtype=scalar),
            "reward": wp.from_torch(self._f_reward, dtype=scalar),
            "newly": wp.from_torch(self._f_newly, dtype=wp.uint8),
            "done": wp.from_torch(self._f_done, dtype=wp.uint8),
            "covered": wp.from_torch(self.covered.view(torch.uint8), dtype=wp.uint8),
            "targets": wp.from_torch(self.targets, dtype=VEC2[scalar]),
        }
        self._f_done_bool = self._f_done.view(torch.bool)
        self._fused_ready = True

    def _state_wp(self):
        """(pos, vel) as Warp arrays for the fused kernels."""
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

    def _launch_cover(self) -> None:
        w = self.world
        self._ensure_fused(w.n_envs)
        scalar = w.wp_dtype
        pos, _ = self._state_wp()
        wp.launch(
            discovery_cover_kernel,
            dim=(w.n_envs, self.n_targets),
            inputs=[
                pos,
                self._wp["targets"],
                wp.int32(self.n_agents),
                scalar(self.covering_range**2),
                wp.int32(self.agents_per_target),
            ],
            outputs=[self._wp["covered"], self._wp["newly"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_obs(self) -> None:
        w = self.world
        scalar = w.wp_dtype
        pos, vel = self._state_wp()
        wp.launch(
            discovery_obs_kernel,
            dim=(w.n_envs, self.n_agents),
            inputs=[
                pos,
                vel,
                self._wp["targets"],
                self._wp["covered"],
                wp.int32(self.n_agents),
                wp.int32(self.n_targets),
                scalar((2.0 * self.agent_radius) ** 2),
            ],
            outputs=[self._wp["obs"], self._wp["touch"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_reward(self) -> None:
        w = self.world
        scalar = w.wp_dtype
        wp.launch(
            discovery_reward_kernel,
            dim=w.n_envs,
            inputs=[
                self._wp["touch"],
                self._wp["newly"],
                self._wp["covered"],
                wp.int32(self.n_agents),
                wp.int32(self.n_targets),
                scalar(self.collision_penalty),
                scalar(self.time_penalty),
                scalar(self.covering_reward),
            ],
            outputs=[self._wp["reward"], self._wp["done"]],
            device=w.device,
            record_tape=False,
        )

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        if self._fused_active:
            return self._f_obs
        w = self.world
        s = w.state
        covered_flag = self.covered.to(w.dtype).unsqueeze(1).expand(-1, w.n_agents, -1)
        return torch.cat([s.pos, s.vel, self._cache["rel_targets"], covered_flag], dim=-1)

    def observation(self, agent_idx: int) -> torch.Tensor:
        return self.observations()[:, agent_idx]

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        c = self._cache
        return self.collision_penalty * c["touch"][
            :, agent_idx
        ] + self.time_penalty * torch.ones_like(c["touch"][:, agent_idx])

    def global_reward(self) -> torch.Tensor:
        return self.covering_reward * self._cache["newly"].sum(dim=-1).to(self.world.dtype)

    def rewards(self) -> torch.Tensor:
        if self._fused_active:
            return self._f_reward
        return super().rewards()

    def done(self) -> torch.Tensor:
        if self._fused_active:
            return self._f_done_bool
        return self.covered.all(dim=-1)

    def info(self) -> dict[str, Any]:
        if self._fused_active:
            return {"covered_frac": self.covered.float().mean(-1)}
        return {"covered_frac": self._cache["covered_frac"]}
