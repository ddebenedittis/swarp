"""Multi-agent navigation to per-agent goals with soft collision avoidance.

Reward (VMAS navigation pattern):
  * position shaping: ``(prev_dist - dist) * pos_shaping_factor`` per agent
    (shared across agents when ``shared_reward=True``),
  * collision penalty per touching pair member,
  * global bonus when all agents reach their goals.

Observation per agent: own pos, vel, heading (cos/sin), angular velocity,
goal-relative vector, and features of up to ``neighbor_obs`` within-radius
neighbors (relative position/velocity + validity mask, padded-list order).
"""

from __future__ import annotations

from typing import Any

import torch

from wmas.core.config import WorldConfig
from wmas.core.world import World
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from wmas.scenarios.base import Scenario


class NavigationScenario(Scenario):
    def __init__(
        self,
        n_agents: int = 4,
        agent_radius: float = 0.05,
        model: DynamicsModel = DynamicsModel.HOLONOMIC,
        ctrl_mode: ControlMode = ControlMode.VELOCITY,
        max_speed: float = 1.0,
        world_size: float = 1.0,
        n_obstacles: int = 0,
        obstacle_radius: float = 0.1,
        neighbor_obs: int = 2,
        neighbor_radius: float | None = None,
        shared_reward: bool = False,
        pos_shaping_factor: float = 1.0,
        collision_penalty: float = -1.0,
        final_reward: float = 5.0,
        goal_tolerance: float | None = None,
        min_spawn_separation: float | None = None,
    ) -> None:
        self.n_agents = n_agents
        self.agent_radius = agent_radius
        self.model = model
        self.ctrl_mode = ctrl_mode
        self.max_speed = max_speed
        self.world_size = world_size
        self.n_obstacles = n_obstacles
        self.obstacle_radius = obstacle_radius
        self.neighbor_obs = neighbor_obs
        self.neighbor_radius = neighbor_radius
        self.shared_reward = shared_reward
        self.pos_shaping_factor = pos_shaping_factor
        self.collision_penalty = collision_penalty
        self.final_reward = final_reward
        self.goal_tolerance = goal_tolerance if goal_tolerance is not None else agent_radius
        self.min_spawn_separation = (
            min_spawn_separation if min_spawn_separation is not None else 3.0 * agent_radius
        )

    # ------------------------------------------------------------------ world

    def make_world(self, n_envs, device, dt, substeps, dtype) -> World:
        configs = [
            AgentConfig(
                model=self.model,
                ctrl_mode=self.ctrl_mode,
                radius=self.agent_radius,
                max_speed=self.max_speed,
                max_accel=2.0 * self.max_speed,
            )
            for _ in range(self.n_agents)
        ]
        margin = 0.5 * self.agent_radius
        reach = 2.0 * self.agent_radius + margin
        world_config = WorldConfig(
            collisions=True,
            collision_k=100.0,
            collision_c=1.0,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            neighbor_radius=max(self.neighbor_radius or 0.0, reach),
            max_neighbors=min(32, max(4, self.n_agents)),
        )
        self.world = World(
            configs,
            world_config,
            n_envs=n_envs,
            device=device,
            dt=dt,
            substeps=substeps,
            dtype=dtype,
        )
        self._nbr_cache: dict[str, torch.Tensor] | None = None
        self._prev_dist: torch.Tensor | None = None
        return self.world

    def _sample_separated(self, n_envs: int, n_points: int, min_dist: float, tries: int = 16):
        """Uniform positions with pairwise separation via bounded resampling.

        Host-sync-free: runs a fixed ``tries`` iterations with no early-exit
        ``.any()`` check, so it never forces a device→host round-trip (safe to
        call inside a masked reset on the step loop).
        """
        w = self.world
        lim = self.world_size - 2.0 * self.agent_radius
        pos = w.sample_uniform((n_envs, n_points, 2), -lim, lim)
        if n_points == 1:
            return pos
        eye = torch.eye(n_points, device=w.device, dtype=w.dtype) * 1e9
        for _ in range(tries):
            d = torch.cdist(pos, pos) + eye
            conflict = (d.min(dim=-1).values < min_dist).unsqueeze(-1)  # [n_envs, n_points, 1]
            resampled = w.sample_uniform((n_envs, n_points, 2), -lim, lim)
            pos = torch.where(conflict, resampled, pos)
        return pos

    def reset_world(self, env_mask: torch.Tensor | None = None) -> None:
        """Reset all envs (``env_mask=None``) or the ``True`` entries of a
        boolean ``[n_envs]`` mask. Host-sync-free: the full batch is always
        sampled and blended with ``torch.where`` so no variable-length gather or
        ``.any()`` is needed."""
        w = self.world
        n = w.n_envs  # always sample full width; blend selected envs with where
        spawn = self._sample_separated(n, self.n_agents, self.min_spawn_separation)
        goals = self._sample_separated(n, self.n_agents, self.min_spawn_separation)
        theta = w.sample_uniform((n, self.n_agents), -torch.pi, torch.pi)

        if w.goals is None:
            w.goals = torch.zeros(w.n_envs, self.n_agents, 2, device=w.device, dtype=w.dtype)

        if env_mask is None:
            w.state.pos.data.copy_(spawn)
            w.state.theta.data.copy_(theta)
            w.state.vel.data.zero_()
            w.state.speed.data.zero_()
            w.state.ang_vel.data.zero_()
            w.goals.copy_(goals)
        else:
            m3 = env_mask.view(-1, 1, 1)
            m2 = env_mask.view(-1, 1)
            zeros_v = torch.zeros_like(w.state.vel.data)
            w.state.pos.data.copy_(torch.where(m3, spawn, w.state.pos.data))
            w.state.theta.data.copy_(torch.where(m2, theta, w.state.theta.data))
            w.state.vel.data.copy_(torch.where(m3, zeros_v, w.state.vel.data))
            w.state.speed.data.copy_(
                torch.where(m2, torch.zeros_like(w.state.speed.data), w.state.speed.data)
            )
            w.state.ang_vel.data.copy_(
                torch.where(m2, torch.zeros_like(w.state.ang_vel.data), w.state.ang_vel.data)
            )
            w.goals.copy_(torch.where(m3, goals, w.goals))

        if self.n_obstacles > 0:
            lim = self.world_size - self.obstacle_radius
            obs_pos = w.sample_uniform((n, self.n_obstacles, 2), -lim, lim)
            if w.obstacle_pos is None:
                w.set_obstacles(
                    torch.zeros(w.n_envs, self.n_obstacles, 2, device=w.device, dtype=w.dtype),
                    torch.full(
                        (self.n_obstacles,), self.obstacle_radius, device=w.device, dtype=w.dtype
                    ),
                )
            if env_mask is None:
                w.obstacle_pos.copy_(obs_pos)
            else:
                w.obstacle_pos.copy_(torch.where(env_mask.view(-1, 1, 1), obs_pos, w.obstacle_pos))
            w.set_obstacles(w.obstacle_pos, w.obstacle_radius)

        self._nbr_cache = None
        self._refresh_step_cache(reset_mask=env_mask)

    # ------------------------------------------------------- per-step caching

    def post_step(self) -> None:
        self._refresh_step_cache()

    def _refresh_step_cache(self, reset_mask: torch.Tensor | None = None) -> None:
        """Neighbor features, distances, and reward terms for the current state.

        ``reset_mask`` (a boolean ``[n_envs]`` or ``None``) marks envs that were
        just reset: their position-shaping baseline is rebased to the fresh
        spawn distance and their shaping term zeroed, while non-reset envs keep
        their carried-over baseline. On a normal step (``None``) every env's
        baseline advances to the current distance. This masked rebase is what
        keeps a partial/auto reset from clobbering other envs' shaping."""
        w = self.world
        pos, vel = w.state.pos, w.state.vel
        idx, cnt = w.neighbors()
        idx = idx.long()
        k_all = idx.shape[-1]
        valid = torch.arange(k_all, device=w.device).view(1, 1, -1) < cnt.long().unsqueeze(
            -1
        )  # [n_envs, n_agents, k_all]
        # neighbor positions (all slots: needed for touching counts)
        flat = idx.reshape(w.n_envs, -1)
        npos = torch.gather(pos, 1, flat.unsqueeze(-1).expand(-1, -1, 2))
        npos = npos.view(w.n_envs, w.n_agents, k_all, 2)
        rel_pos_all = npos - pos.unsqueeze(2)

        # touching pairs -> per-agent collision counts (non-differentiable count)
        ndist = rel_pos_all.norm(dim=-1)
        r = w.agent_radius  # [n_agents]
        min_dist = r.view(1, -1, 1) + r[idx]  # r_i + r_j
        touching = ((ndist < min_dist) & valid).sum(dim=-1).to(w.dtype)

        # observation features only need the first `neighbor_obs` slots
        k = min(self.neighbor_obs, k_all)
        rel_pos = rel_pos_all[:, :, :k] * valid[:, :, :k].unsqueeze(-1)
        flat_k = idx[:, :, :k].reshape(w.n_envs, -1)
        nvel = torch.gather(vel, 1, flat_k.unsqueeze(-1).expand(-1, -1, 2))
        nvel = nvel.view(w.n_envs, w.n_agents, k, 2)
        rel_vel = (nvel - vel.unsqueeze(2)) * valid[:, :, :k].unsqueeze(-1)

        dist_to_goal = (pos - w.goals).norm(dim=-1)
        if self._prev_dist is None:
            self._prev_dist = dist_to_goal.detach().clone()
        pos_shaping = (self._prev_dist - dist_to_goal) * self.pos_shaping_factor
        if reset_mask is None:
            self._prev_dist = dist_to_goal.detach().clone()
        else:
            # Rebase only reset envs; others keep their continuous baseline.
            pos_shaping = torch.where(
                reset_mask.unsqueeze(-1), torch.zeros_like(pos_shaping), pos_shaping
            )
            self._prev_dist = torch.where(
                reset_mask.unsqueeze(-1), dist_to_goal.detach(), self._prev_dist
            )

        self._nbr_cache = {
            "rel_pos": rel_pos,
            "rel_vel": rel_vel,
            "valid": valid,
            "touching": touching,
            "dist_to_goal": dist_to_goal,
            "on_goal": dist_to_goal < self.goal_tolerance,
            "pos_shaping": pos_shaping,
        }

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        """Fully batched observations [n_envs, n_agents, obs_dim]."""
        w = self.world
        s = w.state
        cache = self._nbr_cache
        feats = [
            s.pos,
            s.vel,
            torch.cos(s.theta).unsqueeze(-1),
            torch.sin(s.theta).unsqueeze(-1),
            s.ang_vel.unsqueeze(-1),
            w.goals - s.pos,
        ]
        k = min(self.neighbor_obs, cache["rel_pos"].shape[2])
        if k > 0:
            n_envs, n_agents = w.n_envs, w.n_agents
            feats += [
                cache["rel_pos"][:, :, :k].reshape(n_envs, n_agents, -1),
                cache["rel_vel"][:, :, :k].reshape(n_envs, n_agents, -1),
                cache["valid"][:, :, :k].to(w.dtype),
            ]
        return torch.cat(feats, dim=-1)

    def observation(self, agent_idx: int) -> torch.Tensor:
        return self.observations()[:, agent_idx]

    def rewards(self) -> torch.Tensor:
        cache = self._nbr_cache
        rew = self.collision_penalty * cache["touching"]
        if self.shared_reward:
            rew = rew + cache["pos_shaping"].sum(dim=-1, keepdim=True)
        else:
            rew = rew + cache["pos_shaping"]
        final = self.final_reward * cache["on_goal"].all(dim=-1).to(self.world.dtype)
        return rew + final.unsqueeze(-1)

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        cache = self._nbr_cache
        rew = self.collision_penalty * cache["touching"][:, agent_idx]
        if not self.shared_reward:
            rew = rew + cache["pos_shaping"][:, agent_idx]
        return rew

    def global_reward(self) -> torch.Tensor:
        cache = self._nbr_cache
        rew = self.final_reward * cache["on_goal"].all(dim=-1).to(self.world.dtype)
        if self.shared_reward:
            rew = rew + cache["pos_shaping"].sum(dim=-1)
        return rew

    def done(self) -> torch.Tensor:
        return self._nbr_cache["on_goal"].all(dim=-1)

    def info(self) -> dict[str, Any]:
        return {
            "dist_to_goal": self._nbr_cache["dist_to_goal"],
            "on_goal": self._nbr_cache["on_goal"],
            "collisions": self._nbr_cache["touching"],
        }
