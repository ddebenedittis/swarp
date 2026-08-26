"""Formation: agents hold assigned slots of a regular polygon around a centre.

Each agent ``i`` targets slot ``i`` of an ``n_agents``-vertex regular polygon of
radius ``formation_radius`` centred at a per-env point. The reward is
position-shaping toward the assigned slot (``(prev_dist - dist) * factor``, the
same shaping NavigationScenario uses) plus a soft collision penalty, so the team
converges onto — and holds — the shape.

**Scaling.** Both the torch path and the fused kernels do the inter-agent part of the
observation as **O(n_agents^2)** all-pairs work rather than walking the neighbor list.
That is deliberate: the fused path exists to match the torch parity oracle bit-for-bit,
and the neighbor list is truncated at ``max_neighbors``, so reading it would make the two
paths disagree by construction whenever the list overflowed. The quadratic term is the
price of that guarantee — fine at the tens-of-agents this scenario is written for, and
worth knowing about before pushing ``n_agents`` into the hundreds.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from swarp.core.config import WorldConfig
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.scenarios.formation_kernels import formation_obs_kernel, formation_reward_kernel
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario


class FormationScenario(FusedScenario):
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
        reach = 2.0 * self.agent_radius + margin
        cfg = WorldConfig(
            collisions=True,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            neighbor_radius=reach,
            max_neighbors=min(32, max(4, self.n_agents)),
        ).override_with(world_config)
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
        # Left None so the first refresh seeds the shaping baseline from the spawn
        # distance (shaping 0) rather than from zeros; the fused spec adopts it with
        # alloc="if_none". The torch path reassigns it, which watch=True catches.
        self._prev_dist: torch.Tensor | None = None
        self._cache: dict[str, torch.Tensor] | None = None
        # Allocated here, not in reset_world, so the fused spec can adopt the slots.
        self.world.goals = torch.zeros(n_envs, self.n_agents, 2, device=device, dtype=dtype)
        return self.world

    @property
    def obs_dim(self) -> int:
        return 6

    def reset_world(
        self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False
    ) -> None:
        w = self.world
        n = w.n_envs
        lim = self.world_size - 2.0 * self.agent_radius
        spawn = w.sample_uniform((n, self.n_agents, 2), -lim, lim)
        # Formation centre kept within bounds so all slots stay inside the world.
        clim = max(0.0, self.world_size - self.formation_radius - self.agent_radius)
        center = w.sample_uniform((n, 1, 2), -clim, clim)
        goals = center + self._slot_offsets.unsqueeze(0)  # [n_envs, n_agents, 2]
        w.write_state(env_mask, pos=spawn, vel=0.0)
        if env_mask is None:
            w.goals.copy_(goals)
        else:
            w.goals.copy_(torch.where(env_mask.view(-1, 1, 1), goals, w.goals))
        self.finish_reset(env_mask, obs_only=obs_only)

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh()

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        self._refresh(reset_mask=env_mask)

    # --------------------------------------------------------- fused fast path

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        ne, na = n_envs, self.n_agents
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("shaping", (ne, na)),
            Buf("touch", (ne, na)),
            Buf("dist", (ne, na)),
            Buf("reward", (ne, na)),
            Buf("multiobj", (ne, na, 2)),
            Buf("ferror", (ne,)),
            Buf("inform", (ne, na), "uint8", bool_view=True),
            Buf("done", (ne,), "uint8", bool_view=True),
            Buf("resetmask", (ne,), "uint8", reset_mask=True),
            Buf("goals", (ne, na, 2), "vec2", attr="world.goals", alloc="never", watch=True),
            Buf("prev", (ne, na), attr="_prev_dist", alloc="if_none", carry=True, watch=True),
        )

    def launch_fused(self, pass_: FusedPass) -> None:
        """Obs, then reward — the reward on every pass but an obs-only auto-reset.

        Same rule as :meth:`~swarp.scenarios.navigation.NavigationScenario.launch_fused`:
        ``full_pass=0`` (a mid-step auto-reset) keeps the reward/done already returned for
        that transition, and a standalone reset recomputes them to match the torch oracle.
        """
        self._launch_obs(advance_prev=pass_.advance_prev, full_pass=pass_.full_pass)
        if pass_.full_pass:
            self._launch_reward()

    def _launch_obs(self, advance_prev: int, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        st = w.state_wp()
        wp.launch(
            formation_obs_kernel,
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                self._wp["goals"],
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
                self._wp["prev"],
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
        if self.fused_active:
            return self.fb["obs"]
        w = self.world
        s = w.state
        return torch.cat([s.pos, s.vel, w.goals - s.pos], dim=-1)

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        c = self._cache
        return c["shaping"][:, agent_idx] + self.collision_penalty * c["touching"][:, agent_idx]

    def rewards(self) -> torch.Tensor:
        return self.fb["reward"] if self.fused_active else super().rewards()

    def done(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["done_bool"]
        return self._cache["in_formation"].all(dim=-1)

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {
                "multiobj_reward": self.fb["multiobj"],
                "formation_error": self.fb["ferror"],
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
