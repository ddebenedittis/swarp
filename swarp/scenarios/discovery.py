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

from swarp.core.config import WorldConfig
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.scenarios.discovery_kernels import (
    discovery_cover_kernel,
    discovery_obs_kernel,
    discovery_reward_kernel,
)
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario


class DiscoveryScenario(FusedScenario):
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
        # Task state, allocated here (n_envs is known) and only ever written in place, so
        # the fused spec can adopt it with alloc="never". Allocating it on first reset
        # instead made the fused path silently depend on reset running first.
        self.targets = torch.zeros(  # [n_envs, n_targets, 2]
            n_envs, self.n_targets, 2, device=device, dtype=dtype
        )
        self.covered = torch.zeros(  # [n_envs, n_targets] bool (ever covered)
            n_envs, self.n_targets, dtype=torch.bool, device=device
        )
        self._cache: dict[str, torch.Tensor] | None = None
        return self.world

    @property
    def obs_dim(self) -> int:
        return 4 + 3 * self.n_targets

    def reset_world(
        self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False
    ) -> None:
        w = self.world
        n = w.n_envs
        lim = self.world_size - 2.0 * self.agent_radius
        spawn = w.sample_uniform((n, self.n_agents, 2), -lim, lim)
        targets = w.sample_uniform((n, self.n_targets, 2), -lim, lim)

        w.write_state(env_mask, pos=spawn, vel=0.0)
        if env_mask is None:
            self.targets.copy_(targets)
            self.covered.zero_()
        else:
            self.targets.copy_(torch.where(env_mask.view(-1, 1, 1), targets, self.targets))
            self.covered.copy_(
                torch.where(env_mask.view(-1, 1), torch.zeros_like(self.covered), self.covered)
            )
        self.finish_reset(env_mask, obs_only=obs_only)

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh()

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
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

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        ne, na = n_envs, self.n_agents
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("touch", (ne, na)),
            Buf("reward", (ne, na)),
            Buf("newly", (ne, self.n_targets), "uint8"),
            Buf("done", (ne,), "uint8", bool_view=True),
            Buf("targets", (ne, self.n_targets, 2), "vec2", attr="targets", alloc="never"),
            Buf("covered", (ne, self.n_targets), "bool", attr="covered", alloc="never", carry=True),
        )

    def launch_fused(self, pass_: FusedPass) -> None:
        """Coverage, observations, and — on a step only — the reward.

        Discovery's kernels take *neither* ``advance_prev`` nor ``full_pass``: there is no
        shaping baseline to rebase, and "a reset must not clobber the reward already
        returned for this transition" is expressed by simply not launching the reward
        kernel. Coverage still runs on a reset, updating the ``covered`` latch exactly as
        the torch ``_refresh`` does.
        """
        self._launch_cover()
        self._launch_obs()
        if pass_.is_step:
            self._launch_reward()

    def _launch_cover(self) -> None:
        w = self.world
        scalar = w.wp_dtype
        wp.launch(
            discovery_cover_kernel,
            dim=(w.n_envs, self.n_targets),
            inputs=[
                w.state_wp().pos,
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
        st = w.state_wp()
        wp.launch(
            discovery_obs_kernel,
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
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
        if self.fused_active:
            return self.fb["obs"]
        w = self.world
        s = w.state
        covered_flag = self.covered.to(w.dtype).unsqueeze(1).expand(-1, w.n_agents, -1)
        return torch.cat([s.pos, s.vel, self._cache["rel_targets"], covered_flag], dim=-1)

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        c = self._cache
        return self.collision_penalty * c["touch"][
            :, agent_idx
        ] + self.time_penalty * torch.ones_like(c["touch"][:, agent_idx])

    def global_reward(self) -> torch.Tensor:
        return self.covering_reward * self._cache["newly"].sum(dim=-1).to(self.world.dtype)

    def rewards(self) -> torch.Tensor:
        return self.fb["reward"] if self.fused_active else super().rewards()

    def done(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["done_bool"]
        return self.covered.all(dim=-1)

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {"covered_frac": self.covered.float().mean(-1)}
        return {"covered_frac": self._cache["covered_frac"]}
