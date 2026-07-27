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
import warp as wp

from wmas.core.config import Obstacles, WorldConfig
from wmas.core.state import VEC2
from wmas.core.world import World
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from wmas.scenarios.base import Scenario
from wmas.scenarios.navigation_kernels import nav_obs_kernel, nav_reward_kernel


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
        neighbor_method: str = "auto",
        eager_trims: bool = True,
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
        self.neighbor_method = neighbor_method
        # When True (default) the per-step obs/reward cache reuses persistent
        # scratch buffers (arange, int64 index, gathered neighbor pos/vel,
        # cos/sin) and writes the position-shaping baseline in place, cutting the
        # eager-torch layer's per-step allocations. Bit-exact vs the untrimmed
        # path; set False (used by the ablation baseline) to measure the churn.
        self.eager_trims = eager_trims

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
            neighbor_method=self.neighbor_method,
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
        # Persistent eager-trim scratch (allocated lazily on first refresh).
        self._eager_k_all: int = -1
        self._eager_k: int = -1
        # Fused-kernel obs width and lazy-allocation flag.
        self._k_obs = min(self.neighbor_obs, world_config.max_neighbors)
        self._fused_ready = False
        # Bumped by _sync_fused_handles when a cached buffer handle is rebuilt;
        # the whole-step graph recaptures when this token changes.
        self._handle_version = 0
        return self.world

    def fused_available(self) -> bool:
        """Navigation ships fused Warp obs/reward kernels (2D, needs neighbors)."""
        return True

    @property
    def obs_dim(self) -> int:
        return 9 + 5 * self._k_obs

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
            if w.obstacles is None:
                tt = {"device": w.device, "dtype": w.dtype}
                w.set_obstacles(
                    Obstacles(
                        torch.zeros(w.n_envs, self.n_obstacles, 2, **tt),
                        torch.full((self.n_obstacles,), self.obstacle_radius, **tt),
                    )
                )
            if env_mask is None:
                w.obstacle_pos.copy_(obs_pos)
            else:
                w.obstacle_pos.copy_(torch.where(env_mask.view(-1, 1, 1), obs_pos, w.obstacle_pos))
            # Re-install the retained spec: the poses were written into its own tensor, and
            # an unchanged count takes the in-place path (no graph recapture).
            w.set_obstacles(w.obstacles)

        w.mark_pos_dirty()  # positions written out of band; force a fresh build
        self._nbr_cache = None
        if self._fused_active:
            self._ensure_fused(w.n_envs)
            self._sync_fused_handles()  # this eager _launch_obs uses cached handles
            if env_mask is None:
                self._f_resetmask.fill_(1)
            else:
                self._f_resetmask.copy_(env_mask)  # bool -> uint8
            # An auto-reset (mid-step) must not clobber the reward/done/info
            # buffers already returned this step: obs-only pass (full_pass=0). A
            # standalone reset recomputes everything (full_pass=1) so info() is
            # populated. Either way the shaping baseline rebases only reset envs.
            full = 0 if self._fused_obs_only else 1
            self._launch_obs(advance_prev=0, full_pass=full)
        else:
            self._refresh_step_cache(reset_mask=env_mask)

    # --------------------------------------------------------- fused fast path

    def _ensure_fused(self, n_envs: int) -> None:
        """Allocate the persistent fused output buffers + cached wp handles."""
        if self._fused_ready:
            return
        w = self.world
        na, dev, dt = self.n_agents, w.device, w.dtype
        od = self.obs_dim

        def z(*shape, d=dt):
            return torch.zeros(*shape, device=dev, dtype=d)

        self._f_obs = z(n_envs, na, od)
        self._f_touch = z(n_envs, na)
        self._f_dist = z(n_envs, na)
        self._f_shaping = z(n_envs, na)
        self._f_reward = z(n_envs, na)
        self._f_ongoal = z(n_envs, na, d=torch.uint8)
        self._f_overflow = z(n_envs, na, d=torch.uint8)
        self._f_done = z(n_envs, d=torch.uint8)
        self._f_resetmask = z(n_envs, d=torch.uint8)
        if self._prev_dist is None:
            self._prev_dist = z(n_envs, na)
        scalar = w.wp_dtype
        self._wp = {
            "obs": wp.from_torch(self._f_obs, dtype=scalar),
            "touch": wp.from_torch(self._f_touch, dtype=scalar),
            "dist": wp.from_torch(self._f_dist, dtype=scalar),
            "shaping": wp.from_torch(self._f_shaping, dtype=scalar),
            "reward": wp.from_torch(self._f_reward, dtype=scalar),
            "ongoal": wp.from_torch(self._f_ongoal, dtype=wp.uint8),
            "overflow": wp.from_torch(self._f_overflow, dtype=wp.uint8),
            "done": wp.from_torch(self._f_done, dtype=wp.uint8),
            "resetmask": wp.from_torch(self._f_resetmask, dtype=wp.uint8),
        }
        # Zero-copy bool reinterpret views (uint8 0/1 -> bool).
        self._f_ongoal_bool = self._f_ongoal.view(torch.bool)
        self._f_overflow_bool = self._f_overflow.view(torch.bool)
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
        (grad-path ``_prev_dist`` reassignment; goals realloc). Cheap pointer
        compare in steady state; bumps ``_handle_version`` to force recapture.
        Runs outside capture (eager prep)."""
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
        """Capture-unsafe prep, run eagerly before the physics step: ensure the
        fused buffers, clear the reset mask (a normal step resets no env), and
        refresh any moved handle."""
        self._ensure_fused(self.world.n_envs)
        self._f_resetmask.zero_()
        self._sync_fused_handles()

    def _graph_post_physics(self) -> None:
        """Capture-safe obs+reward launch sequence (assumes ``resetmask==0`` and
        fresh handles from ``_pre_graph_step``). Pure ``wp.launch`` + grid build."""
        self._launch_obs(advance_prev=1, full_pass=1)
        self._launch_reward()

    def _state_wp(self):
        """(pos, vel, theta, ang_vel) as Warp arrays for the fused kernels."""
        w = self.world
        scalar = w.wp_dtype
        vec2 = VEC2[scalar]
        if w._persistent and not w._detached:
            s = w.runtime.state
            return s.pos, s.vel, s.theta, s.ang_vel
        st = w.state
        return (
            wp.from_torch(st.pos.contiguous(), dtype=vec2),
            wp.from_torch(st.vel.contiguous(), dtype=vec2),
            wp.from_torch(st.theta.contiguous(), dtype=scalar),
            wp.from_torch(st.ang_vel.contiguous(), dtype=scalar),
        )

    def _launch_obs(self, advance_prev: int, full_pass: int) -> None:
        w = self.world
        n_envs = w.n_envs
        self._ensure_fused(n_envs)
        w.neighbors()  # build the grid on the current state (fills grid buffers)
        grid = w.stepper.grid(n_envs)
        scalar = w.wp_dtype
        pos, vel, theta, ang_vel = self._state_wp()
        goals = self._wp_goals
        prev = self._wp_prev
        # Touching uses the static per-agent radius (matches the torch reference's
        # World.agent_radius); per-env randomization affects forces, not this count.
        params = w.stepper.params.floats
        wp.launch(
            nav_obs_kernel,
            dim=(n_envs, self.n_agents),
            inputs=[
                pos,
                vel,
                theta,
                ang_vel,
                goals,
                grid.neighbor_idx,
                grid.neighbor_count,
                grid.neighbor_true_count,
                params,
                self._wp["resetmask"],
                wp.int32(self._k_obs),
                scalar(self.pos_shaping_factor),
                scalar(self.goal_tolerance),
                wp.int32(advance_prev),
                wp.int32(full_pass),
            ],
            outputs=[
                self._wp["obs"],
                self._wp["touch"],
                self._wp["dist"],
                self._wp["shaping"],
                self._wp["ongoal"],
                self._wp["overflow"],
                prev,
            ],
            device=w.device,
            record_tape=False,
        )

    def _launch_reward(self) -> None:
        w = self.world
        scalar = w.wp_dtype
        wp.launch(
            nav_reward_kernel,
            dim=w.n_envs,
            inputs=[
                self._wp["touch"],
                self._wp["shaping"],
                self._wp["ongoal"],
                wp.int32(self.n_agents),
                scalar(self.collision_penalty),
                scalar(self.final_reward),
                wp.int32(1 if self.shared_reward else 0),
            ],
            outputs=[self._wp["reward"], self._wp["done"]],
            device=w.device,
            record_tape=False,
        )

    # ------------------------------------------------------- per-step caching

    def post_step(self) -> None:
        if self._fused_active:
            # Same sequence the whole-step graph runs; delegating keeps non-graph
            # fused mode and CPU eager-persistent bit-identical to graph mode.
            self._pre_graph_step()
            self._graph_post_physics()
        else:
            self._refresh_step_cache()

    def _ensure_eager_buffers(self, k_all: int, k: int) -> None:
        """Allocate the persistent eager-trim scratch (once per shape)."""
        if self._eager_k_all == k_all and self._eager_k == k:
            return
        w = self.world
        ne, na, dev, dt = w.n_envs, w.n_agents, w.device, w.dtype
        self._arange_k = torch.arange(k_all, device=dev)
        self._idx_long = torch.empty((ne, na, k_all), dtype=torch.int64, device=dev)
        self._npos_flat = torch.empty((ne, na * k_all, 2), dtype=dt, device=dev)
        self._cosbuf = torch.empty((ne, na), dtype=dt, device=dev)
        self._sinbuf = torch.empty((ne, na), dtype=dt, device=dev)
        self._nvel_flat = torch.empty((ne, na * k, 2), dtype=dt, device=dev) if k > 0 else None
        self._eager_k_all = k_all
        self._eager_k = k

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
        k_all = idx.shape[-1]
        k = min(self.neighbor_obs, k_all)
        eager = self.eager_trims
        # ``gather(out=)`` is not autograd-compatible, so the out= fast path is
        # only taken when no grad is being recorded; the persistent int64 index,
        # arange, and in-place shaping baseline are grad-safe and always used.
        fast = eager and not torch.is_grad_enabled()

        if eager:
            self._ensure_eager_buffers(k_all, k)
            idx_long = self._idx_long
            idx_long.copy_(idx)
            arange_k = self._arange_k
        else:
            idx_long = idx.long()
            arange_k = torch.arange(k_all, device=w.device)

        valid = arange_k.view(1, 1, -1) < cnt.long().unsqueeze(-1)  # [n_envs, n_agents, k_all]
        # neighbor positions (all slots: needed for touching counts)
        flat = idx_long.reshape(w.n_envs, -1)
        idx_exp = flat.unsqueeze(-1).expand(-1, -1, 2)
        npos = (
            torch.gather(pos, 1, idx_exp, out=self._npos_flat)
            if fast
            else torch.gather(pos, 1, idx_exp)
        )
        npos = npos.view(w.n_envs, w.n_agents, k_all, 2)
        rel_pos_all = npos - pos.unsqueeze(2)

        # touching pairs -> per-agent collision counts (non-differentiable count)
        ndist = rel_pos_all.norm(dim=-1)
        r = w.agent_radius  # [n_agents]
        min_dist = r.view(1, -1, 1) + r[idx_long]  # r_i + r_j
        touching = ((ndist < min_dist) & valid).sum(dim=-1).to(w.dtype)

        # observation features only need the first `neighbor_obs` slots
        rel_pos = rel_pos_all[:, :, :k] * valid[:, :, :k].unsqueeze(-1)
        flat_k = idx_long[:, :, :k].reshape(w.n_envs, -1)
        vidx_exp = flat_k.unsqueeze(-1).expand(-1, -1, 2)
        nvel = (
            torch.gather(vel, 1, vidx_exp, out=self._nvel_flat)
            if (fast and k > 0)
            else torch.gather(vel, 1, vidx_exp)
        )
        nvel = nvel.view(w.n_envs, w.n_agents, k, 2)
        rel_vel = (nvel - vel.unsqueeze(2)) * valid[:, :, :k].unsqueeze(-1)

        dist_to_goal = (pos - w.goals).norm(dim=-1)
        if self._prev_dist is None:
            self._prev_dist = dist_to_goal.detach().clone()
        pos_shaping = (self._prev_dist - dist_to_goal) * self.pos_shaping_factor
        if reset_mask is None:
            if eager:
                self._prev_dist.copy_(dist_to_goal.detach())
            else:
                self._prev_dist = dist_to_goal.detach().clone()
        else:
            # Rebase only reset envs; others keep their continuous baseline.
            m = reset_mask.unsqueeze(-1)
            pos_shaping = torch.where(m, torch.zeros_like(pos_shaping), pos_shaping)
            new_prev = torch.where(m, dist_to_goal.detach(), self._prev_dist)
            if eager:
                self._prev_dist.copy_(new_prev)
            else:
                self._prev_dist = new_prev

        self._nbr_cache = {
            "rel_pos": rel_pos,
            "rel_vel": rel_vel,
            "valid": valid,
            "touching": touching,
            "dist_to_goal": dist_to_goal,
            "on_goal": dist_to_goal < self.goal_tolerance,
            "pos_shaping": pos_shaping,
            # agents whose neighbor list truncated at max_neighbors (undercounted)
            "neighbor_overflow": w.neighbor_overflow(),
        }

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        """Fully batched observations [n_envs, n_agents, obs_dim]."""
        if self._fused_active:
            return self._f_obs
        w = self.world
        s = w.state
        cache = self._nbr_cache
        fast = self.eager_trims and not torch.is_grad_enabled()
        if fast:
            cos = torch.cos(s.theta, out=self._cosbuf).unsqueeze(-1)
            sin = torch.sin(s.theta, out=self._sinbuf).unsqueeze(-1)
        else:
            cos = torch.cos(s.theta).unsqueeze(-1)
            sin = torch.sin(s.theta).unsqueeze(-1)
        feats = [
            s.pos,
            s.vel,
            cos,
            sin,
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
        if self._fused_active:
            return self._f_reward
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
        if self._fused_active:
            return self._f_done_bool
        return self._nbr_cache["on_goal"].all(dim=-1)

    def info(self) -> dict[str, Any]:
        if self._fused_active:
            return {
                "dist_to_goal": self._f_dist,
                "on_goal": self._f_ongoal_bool,
                "collisions": self._f_touch,
                "neighbor_overflow": self._f_overflow_bool,
            }
        return {
            "dist_to_goal": self._nbr_cache["dist_to_goal"],
            "on_goal": self._nbr_cache["on_goal"],
            "collisions": self._nbr_cache["touching"],
            "neighbor_overflow": self._nbr_cache["neighbor_overflow"],
        }
