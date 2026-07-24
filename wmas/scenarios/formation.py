"""Formation: agents hold assigned slots of a regular polygon around a centre.

Each agent ``i`` targets slot ``i`` of an ``n_agents``-vertex regular polygon of
radius ``formation_radius`` centred at a per-env point. The reward is
position-shaping toward the assigned slot (``(prev_dist - dist) * factor``, the
same shaping NavigationScenario uses) plus a soft collision penalty, so the team
converges onto — and holds — the shape.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from wmas.core.config import WorldConfig
from wmas.core.state import VEC2
from wmas.core.world import World
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from wmas.scenarios.base import Scenario
from wmas.scenarios.formation_kernels import formation_obs_kernel, formation_reward_kernel


class FormationScenario(Scenario):
    def __init__(
        self,
        n_agents: int = 5,
        agent_radius: float = 0.05,
        world_size: float = 1.0,
        formation_radius: float = 0.5,
        max_speed: float = 1.0,
        pos_shaping_factor: float = 1.0,
        collision_penalty: float = -1.0,
        goal_tolerance: float | None = None,
    ) -> None:
        self.n_agents = n_agents
        self.agent_radius = agent_radius
        self.world_size = world_size
        self.formation_radius = formation_radius
        self.max_speed = max_speed
        self.pos_shaping_factor = pos_shaping_factor
        self.collision_penalty = collision_penalty
        self.goal_tolerance = goal_tolerance if goal_tolerance is not None else 2.0 * agent_radius

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
        reach = 2.0 * self.agent_radius + margin
        cfg = WorldConfig(
            collisions=True,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            neighbor_radius=reach,
            max_neighbors=min(32, max(4, self.n_agents)),
        )
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        # Fixed slot offsets: regular polygon vertices (relative to the centre).
        ang = torch.arange(self.n_agents, device=device, dtype=dtype) * (
            2.0 * math.pi / self.n_agents
        )
        self._slot_offsets = self.formation_radius * torch.stack(
            [torch.cos(ang), torch.sin(ang)], dim=-1
        )  # [n_agents, 2]
        self._prev_dist: torch.Tensor | None = None
        self._cache: dict[str, torch.Tensor] | None = None
        self._fused_ready = False
        # Bumped by _sync_fused_handles when a cached buffer handle is rebuilt.
        self._handle_version = 0
        return self.world

    def fused_available(self) -> bool:
        """Formation ships fused Warp obs/reward kernels (2D holonomic)."""
        return True

    @property
    def obs_dim(self) -> int:
        return 6

    def reset_world(self, env_mask: torch.Tensor | None = None) -> None:
        w = self.world
        n = w.n_envs
        lim = self.world_size - 2.0 * self.agent_radius
        spawn = w.sample_uniform((n, self.n_agents, 2), -lim, lim)
        # Formation centre kept within bounds so all slots stay inside the world.
        clim = max(0.0, self.world_size - self.formation_radius - self.agent_radius)
        center = w.sample_uniform((n, 1, 2), -clim, clim)
        goals = center + self._slot_offsets.unsqueeze(0)  # [n_envs, n_agents, 2]
        if w.goals is None:
            w.goals = torch.zeros(n, self.n_agents, 2, device=w.device, dtype=w.dtype)
        if env_mask is None:
            w.state.pos.data.copy_(spawn)
            w.state.vel.data.zero_()
            w.goals.copy_(goals)
        else:
            m3 = env_mask.view(-1, 1, 1)
            w.state.pos.data.copy_(torch.where(m3, spawn, w.state.pos.data))
            w.state.vel.data.copy_(
                torch.where(m3, torch.zeros_like(w.state.vel.data), w.state.vel.data)
            )
            w.goals.copy_(torch.where(m3, goals, w.goals))
        if self._fused_active:
            self._ensure_fused(w.n_envs)
            self._sync_fused_handles()  # this eager _launch_obs uses cached handles
            if env_mask is None:
                self._f_resetmask.fill_(1)
            else:
                self._f_resetmask.copy_(env_mask)  # bool -> uint8
            full = 0 if self._fused_obs_only else 1
            self._launch_obs(advance_prev=0, full_pass=full)
        else:
            self._refresh(reset_mask=env_mask)

    def post_step(self) -> None:
        if self._fused_active:
            # Same sequence the whole-step graph runs (keeps non-graph fused mode
            # and CPU eager-persistent bit-identical).
            self._pre_graph_step()
            self._graph_post_physics()
        else:
            self._refresh()

    # --------------------------------------------------------- fused fast path

    def _ensure_fused(self, n_envs: int) -> None:
        """Allocate the persistent fused output buffers + cached wp handles."""
        if self._fused_ready:
            return
        w = self.world
        na, dev, dt = self.n_agents, w.device, w.dtype

        def z(*shape, d=dt):
            return torch.zeros(*shape, device=dev, dtype=d)

        self._f_obs = z(n_envs, na, self.obs_dim)
        self._f_shaping = z(n_envs, na)
        self._f_touch = z(n_envs, na)
        self._f_dist = z(n_envs, na)
        self._f_reward = z(n_envs, na)
        self._f_multiobj = z(n_envs, na, 2)
        self._f_ferror = z(n_envs)
        self._f_inform = z(n_envs, na, d=torch.uint8)
        self._f_done = z(n_envs, d=torch.uint8)
        self._f_resetmask = z(n_envs, d=torch.uint8)
        if self._prev_dist is None:
            self._prev_dist = z(n_envs, na)
        scalar = w.wp_dtype
        self._wp = {
            "obs": wp.from_torch(self._f_obs, dtype=scalar),
            "shaping": wp.from_torch(self._f_shaping, dtype=scalar),
            "touch": wp.from_torch(self._f_touch, dtype=scalar),
            "dist": wp.from_torch(self._f_dist, dtype=scalar),
            "reward": wp.from_torch(self._f_reward, dtype=scalar),
            "multiobj": wp.from_torch(self._f_multiobj, dtype=scalar),
            "ferror": wp.from_torch(self._f_ferror, dtype=scalar),
            "inform": wp.from_torch(self._f_inform, dtype=wp.uint8),
            "done": wp.from_torch(self._f_done, dtype=wp.uint8),
            "resetmask": wp.from_torch(self._f_resetmask, dtype=wp.uint8),
        }
        self._f_inform_bool = self._f_inform.view(torch.bool)
        self._f_done_bool = self._f_done.view(torch.bool)
        # Stable wp handles for the two buffers otherwise re-wrapped per launch.
        # w.goals is updated in place (copy_) after its one-time alloc; _prev_dist
        # is written in place by the fused kernel but reassigned on the torch/grad
        # path — _sync_fused_handles rebuilds either if its data_ptr moves.
        vec2 = VEC2[scalar]
        self._wp_goals = wp.from_torch(w.goals.contiguous(), dtype=vec2)
        self._wp_prev = wp.from_torch(self._prev_dist.contiguous(), dtype=scalar)
        self._goals_ptr = w.goals.data_ptr()
        self._prev_ptr = self._prev_dist.data_ptr()
        self._fused_ready = True

    def _sync_fused_handles(self) -> None:
        """Rebuild any cached wp handle whose backing tensor was reallocated
        (grad-path ``_prev_dist`` reassignment; goals realloc); bumps
        ``_handle_version`` to force recapture. Runs outside capture."""
        w = self.world
        scalar = w.wp_dtype
        vec2 = VEC2[scalar]
        changed = False
        if w.goals.data_ptr() != self._goals_ptr:
            self._wp_goals = wp.from_torch(w.goals.contiguous(), dtype=vec2)
            self._goals_ptr = w.goals.data_ptr()
            changed = True
        if self._prev_dist.data_ptr() != self._prev_ptr:
            self._wp_prev = wp.from_torch(self._prev_dist.contiguous(), dtype=scalar)
            self._prev_ptr = self._prev_dist.data_ptr()
            changed = True
        if changed:
            self._handle_version += 1

    # ----------------------------------------------------- whole-step graph

    def graph_capturable(self) -> bool:
        return True

    def graph_recapture_token(self) -> int:
        return self._handle_version

    def _graph_warmup_carries(self) -> list[torch.Tensor]:
        return [self._prev_dist]

    def _pre_graph_step(self) -> None:
        self._ensure_fused(self.world.n_envs)
        self._f_resetmask.zero_()  # a normal step resets no env
        self._sync_fused_handles()

    def _graph_post_physics(self) -> None:
        self._launch_obs(advance_prev=1, full_pass=1)
        self._launch_reward()

    def _state_wp(self):
        """(pos, vel, goals) as Warp arrays for the fused kernels."""
        w = self.world
        vec2 = VEC2[w.wp_dtype]
        goals = self._wp_goals
        if w._persistent and not w._detached:
            s = w.runtime.state
            return s.pos, s.vel, goals
        st = w.state
        return (
            wp.from_torch(st.pos.contiguous(), dtype=vec2),
            wp.from_torch(st.vel.contiguous(), dtype=vec2),
            goals,
        )

    def _launch_obs(self, advance_prev: int, full_pass: int) -> None:
        w = self.world
        self._ensure_fused(w.n_envs)
        scalar = w.wp_dtype
        pos, vel, goals = self._state_wp()
        prev = self._wp_prev
        wp.launch(
            formation_obs_kernel,
            dim=(w.n_envs, self.n_agents),
            inputs=[
                pos,
                vel,
                goals,
                self._wp["resetmask"],
                scalar((2.0 * self.agent_radius) ** 2),
                scalar(self.goal_tolerance),
                scalar(self.pos_shaping_factor),
                wp.int32(self.n_agents),
                wp.int32(advance_prev),
                wp.int32(full_pass),
            ],
            outputs=[
                self._wp["obs"],
                self._wp["shaping"],
                self._wp["touch"],
                self._wp["dist"],
                self._wp["inform"],
                prev,
            ],
            device=w.device,
            record_tape=False,
        )

    def _launch_reward(self) -> None:
        w = self.world
        scalar = w.wp_dtype
        wp.launch(
            formation_reward_kernel,
            dim=w.n_envs,
            inputs=[
                self._wp["shaping"],
                self._wp["touch"],
                self._wp["dist"],
                self._wp["inform"],
                wp.int32(self.n_agents),
                scalar(self.collision_penalty),
                scalar(1.0 / self.n_agents),
            ],
            outputs=[
                self._wp["reward"],
                self._wp["done"],
                self._wp["multiobj"],
                self._wp["ferror"],
            ],
            device=w.device,
            record_tape=False,
        )

    def _refresh(self, reset_mask: torch.Tensor | None = None) -> None:
        w = self.world
        pos = w.state.pos
        dist = (pos - w.goals).norm(dim=-1)  # [n_envs, n_agents]
        if self._prev_dist is None:
            self._prev_dist = dist.detach().clone()
        shaping = (self._prev_dist - dist) * self.pos_shaping_factor
        if reset_mask is None:
            self._prev_dist = dist.detach().clone()
        else:
            shaping = torch.where(reset_mask.unsqueeze(-1), torch.zeros_like(shaping), shaping)
            self._prev_dist = torch.where(reset_mask.unsqueeze(-1), dist.detach(), self._prev_dist)

        # Squared broadcast distance (dx^2 + dy^2, no sqrt) so the reference's
        # touching count matches the fused kernel bit-for-bit at the < 2r
        # boundary; torch.cdist's matmul expansion rounds differently (and its
        # non-matmul mode is ~50x slower). Same O(n_agents^2) as the original.
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [n_envs, n_agents, n_agents, 2]
        dd2 = (diff * diff).sum(dim=-1)
        touching = (dd2 < (2.0 * self.agent_radius) ** 2).sum(dim=-1).to(w.dtype) - 1.0
        self._cache = {
            "dist": dist,
            "shaping": shaping,
            "touching": touching,
            "in_formation": (dist < self.goal_tolerance),
        }

    def observations(self) -> torch.Tensor:
        if self._fused_active:
            return self._f_obs
        w = self.world
        s = w.state
        return torch.cat([s.pos, s.vel, w.goals - s.pos], dim=-1)

    def observation(self, agent_idx: int) -> torch.Tensor:
        return self.observations()[:, agent_idx]

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        c = self._cache
        return c["shaping"][:, agent_idx] + self.collision_penalty * c["touching"][:, agent_idx]

    def rewards(self) -> torch.Tensor:
        if self._fused_active:
            return self._f_reward
        return super().rewards()

    def done(self) -> torch.Tensor:
        if self._fused_active:
            return self._f_done_bool
        return self._cache["in_formation"].all(dim=-1)

    def info(self) -> dict[str, Any]:
        if self._fused_active:
            return {
                "multiobj_reward": self._f_multiobj,
                "formation_error": self._f_ferror,
            }
        c = self._cache
        # Per-agent objective vector [n_envs, n_agents, n_obj]: the two reward terms
        # (shaping, collision) kept separate so a lexicographic/multi-objective loop can
        # consume them via ``("next", "info", "multiobj_reward")``. Their sum over the last
        # dim equals the scalar per-agent reward; the reward key itself stays scalar.
        multiobj_reward = torch.stack(
            [c["shaping"], self.collision_penalty * c["touching"]], dim=-1
        )
        return {
            "multiobj_reward": multiobj_reward,  # [n_envs, n_agents, 2]
            "formation_error": c["dist"].mean(-1),  # [n_envs] scalar diagnostic
        }
