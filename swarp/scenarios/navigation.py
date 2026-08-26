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

import math
from typing import Any

import torch
import warp as wp

from swarp.core.config import Obstacles, WorldConfig
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario
from swarp.scenarios.navigation_kernels import nav_obs_kernel, nav_reward_kernel


class NavigationScenario(FusedScenario):
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
        # Provisional obs width so ``obs_dim`` answers before ``make_world``: it mirrors
        # this scenario's own ``max_neighbors`` default. ``make_world`` re-derives it
        # from the *resolved* config, so a ``world_config`` override still wins.
        self._k_obs = min(self.neighbor_obs, min(32, max(4, self.n_agents)))

    # ------------------------------------------------------------------ world

    def _agent_configs(self) -> list[AgentConfig]:
        """One :class:`AgentConfig` per agent — a homogeneous fleet of ``self.model``.

        Split out so a subclass can vary the fleet (``render/demo.py``'s mixed-dynamics
        demo does) without reimplementing ``make_world`` and, with it, the fused
        bookkeeping.
        """
        return [
            AgentConfig(
                model=self.model,
                ctrl_mode=self.ctrl_mode,
                radius=self.agent_radius,
                max_speed=self.max_speed,
                max_accel=2.0 * self.max_speed,
            )
            for _ in range(self.n_agents)
        ]

    def make_world(self, n_envs, device, dt, substeps, dtype, world_config=None) -> World:
        configs = self._agent_configs()
        margin = 0.5 * self.agent_radius
        reach = 2.0 * self.agent_radius + margin
        cfg = WorldConfig(
            collisions=True,
            collision_k=100.0,
            collision_c=1.0,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            neighbor_radius=max(self.neighbor_radius or 0.0, reach),
            max_neighbors=min(32, max(4, self.n_agents)),
            neighbor_method=self.neighbor_method,
        ).override_with(world_config)
        self.world = World(
            configs,
            cfg,
            n_envs=n_envs,
            device=device,
            dt=dt,
            substeps=substeps,
            dtype=dtype,
        )
        self._nbr_cache: dict[str, torch.Tensor] | None = None
        # The position-shaping baseline. Left None so the *first* refresh seeds it from
        # the spawn distance (shaping 0) rather than from zeros; the fused spec adopts it
        # with alloc="if_none". Reassigned by the torch path when eager_trims is off,
        # which is what watch=True catches.
        self._prev_dist: torch.Tensor | None = None
        # Goals are engine-independent per-agent targets, written in place on every
        # reset; allocated here (not in reset_world) so the fused spec can adopt them.
        self.world.goals = torch.zeros(n_envs, self.n_agents, 2, device=device, dtype=dtype)
        # Persistent eager-trim scratch (allocated lazily on first refresh).
        self._eager_k_all: int = -1
        self._eager_k: int = -1
        self._k_obs = min(self.neighbor_obs, cfg.max_neighbors)
        if self.n_obstacles > 0:
            # Install the (zeroed) obstacle set once, here: ``World`` then retains the
            # resolved spec, so every reset just writes new poses into its own tensor and
            # re-installs it — allocation-free, and with an unchanged count also free of a
            # graph recapture. Installing on first *reset* instead would put the one
            # count-changing install after capture.
            tt = {"device": device, "dtype": dtype}
            self.world.set_obstacles(
                Obstacles(
                    torch.zeros(n_envs, self.n_obstacles, 2, **tt),
                    torch.full((self.n_obstacles,), self.obstacle_radius, **tt),
                )
            )
        return self.world

    #: When True (the default) the torch obs/reward cache reuses persistent scratch
    #: (arange, int64 index, gathered neighbor pos/vel, cos/sin) and writes the shaping
    #: baseline in place, cutting that path's per-step allocations. Bit-exact against the
    #: untrimmed path; the ablation baseline sets it False to measure the churn. Not an
    #: ``__init__`` parameter: it is an ablation knob, not part of the task definition.
    eager_trims: bool = True

    @property
    def obs_dim(self) -> int:
        return 9 + 5 * self._k_obs

    #: Cells per spawn point the stratified sampler aims for. More cells keep the choice
    #: of cell closer to uniform; the cap that actually binds is jitter room per cell.
    _SPAWN_CELL_OVERSAMPLE = 8

    def _sample_separated(self, n_envs: int, n_points: int, min_dist: float):
        """Positions with a **guaranteed** pairwise separation of ``min_dist``.

        Stratified jittered-cell sampling. The spawn square is cut into a ``g x g`` grid of
        cells of side ``s``; each env draws ``n_points`` *distinct* cells, and each point is
        jittered uniformly inside a centred sub-square of side ``s - min_dist``. Two points
        in different cells then differ by at least ``min_dist`` along whichever axis
        separates their cells, so the separation holds *by construction*.

        That replaces a fixed 16-iteration ``torch.cdist`` rejection loop, which cost a
        batched distance matrix per iteration and could still return overlapping points.
        Under ``auto_reset`` this runs on every step, for the whole batch (the reset is
        masked device-side, so there is no host-side "is anything done?" gate to skip it) —
        making it cheap is the only lever, and it was worth ~300x on the step time at
        4096x8. See ``docs/performance.md``.

        Host-sync-free (``torch.rand`` / ``argsort`` / arithmetic on ``world.generator``),
        so it stays safe inside a masked reset on the step loop.

        Falls back to plain uniform sampling when no grid can both hold ``n_points`` cells
        and leave jitter room — i.e. when the requested separation is at the packing limit
        for this world. Better an honest uniform draw than a grid silently packed so tight
        that every point sits pinned at its cell centre.
        """
        w = self.world
        lim = self.world_size - 2.0 * self.agent_radius
        shape = (n_envs, n_points, 2)
        if n_points == 1 or min_dist <= 0.0:
            return w.sample_uniform(shape, -lim, lim)

        side = 2.0 * lim
        g = math.ceil(math.sqrt(n_points * self._SPAWN_CELL_OVERSAMPLE))
        g = min(g, int(side // (2.0 * min_dist)))  # keep at least min_dist of jitter room
        g = max(g, math.ceil(math.sqrt(n_points)))  # ...but the points have to fit
        if side / g <= min_dist:
            return w.sample_uniform(shape, -lim, lim)
        s = side / g
        jitter = s - min_dist

        # n_points distinct cells per env. argsort of random keys is a batched partial
        # permutation: distinctness is structural, where a rejection loop only ever
        # approaches it. Cheap because g*g stays O(n_points), not O(world / min_dist).
        cell = w.sample_uniform((n_envs, g * g), 0.0, 1.0).argsort(dim=-1)[:, :n_points]
        col = (cell % g).to(w.dtype)
        row = torch.div(cell, g, rounding_mode="floor").to(w.dtype)
        centers = torch.stack([col, row], dim=-1) * s + (0.5 * s - lim)
        return centers + (w.sample_uniform(shape, 0.0, 1.0) - 0.5) * jitter

    def reset_world(
        self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False
    ) -> None:
        """Reset all envs (``env_mask=None``) or the ``True`` entries of a
        boolean ``[n_envs]`` mask. Host-sync-free: the full batch is always
        sampled and blended with ``torch.where`` so no variable-length gather or
        ``.any()`` is needed."""
        w = self.world
        n = w.n_envs  # always sample full width; blend selected envs with where
        spawn = self._sample_separated(n, self.n_agents, self.min_spawn_separation)
        goals = self._sample_separated(n, self.n_agents, self.min_spawn_separation)
        theta = w.sample_uniform((n, self.n_agents), -torch.pi, torch.pi)

        w.write_state(env_mask, pos=spawn, theta=theta, vel=0.0, speed=0.0, ang_vel=0.0)
        if env_mask is None:
            w.goals.copy_(goals)
        else:
            w.goals.copy_(torch.where(env_mask.view(-1, 1, 1), goals, w.goals))

        if self.n_obstacles > 0:
            lim = self.world_size - self.obstacle_radius
            obs_pos = w.sample_uniform((n, self.n_obstacles, 2), -lim, lim)
            if env_mask is None:
                w.obstacle_pos.copy_(obs_pos)
            else:
                w.obstacle_pos.copy_(torch.where(env_mask.view(-1, 1, 1), obs_pos, w.obstacle_pos))
            # Re-install the retained spec: the poses were written into its own tensor, and
            # an unchanged count takes the in-place path (no graph recapture).
            w.set_obstacles(w.obstacles)

        self._nbr_cache = None
        self.finish_reset(env_mask, obs_only=obs_only)

    # --------------------------------------------------------- fused fast path

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        ne, na = n_envs, self.n_agents
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("touch", (ne, na)),
            Buf("dist", (ne, na)),
            Buf("shaping", (ne, na)),
            Buf("reward", (ne, na)),
            Buf("ongoal", (ne, na), "uint8", bool_view=True),
            Buf("overflow", (ne, na), "uint8", bool_view=True),
            Buf("done", (ne,), "uint8", bool_view=True),
            Buf("resetmask", (ne,), "uint8", reset_mask=True),
            Buf("goals", (ne, na, 2), "vec2", attr="world.goals", alloc="never", watch=True),
            Buf("prev", (ne, na), attr="_prev_dist", alloc="if_none", carry=True, watch=True),
        )

    def launch_fused(self, pass_: FusedPass) -> None:
        """Observations, then the reward — the latter only on a step.

        A reset must leave ``reward``/``done`` alone: an auto-reset's were already returned
        for the transition just taken. A standalone reset still passes ``full_pass=1``,
        which fills the ``info`` outputs; those all come off the obs kernel. Either way the
        shaping baseline rebases only the envs the reset mask selects.
        """
        self._launch_obs(advance_prev=pass_.advance_prev, full_pass=pass_.full_pass)
        if pass_.is_step:
            self._launch_reward()

    def _launch_obs(self, advance_prev: int, full_pass: int) -> None:
        w = self.world
        n_envs = w.n_envs
        w.neighbors()  # build the grid on the current state (fills grid buffers)
        grid = w.stepper.grid(n_envs)
        scalar = w.wp_dtype
        st = w.state_wp()
        goals = self._wp["goals"]
        prev = self._wp["prev"]
        # Touching uses the static per-agent radius (matches the torch reference's
        # World.agent_radius); per-env randomization affects forces, not this count.
        params = w.stepper.params.floats
        wp.launch(
            nav_obs_kernel,
            dim=(n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                st.theta,
                st.ang_vel,
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

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh_step_cache()

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        self._refresh_step_cache(reset_mask=env_mask)

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
        if self.fused_active:
            return self.fb["obs"]
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

    def rewards(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["reward"]
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
        if self.fused_active:
            return self.fb["done_bool"]
        return self._nbr_cache["on_goal"].all(dim=-1)

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {
                "dist_to_goal": self.fb["dist"],
                "on_goal": self.fb["ongoal_bool"],
                "collisions": self.fb["touch"],
                "neighbor_overflow": self.fb["overflow_bool"],
            }
        return {
            "dist_to_goal": self._nbr_cache["dist_to_goal"],
            "on_goal": self._nbr_cache["on_goal"],
            "collisions": self._nbr_cache["touching"],
            "neighbor_overflow": self._nbr_cache["neighbor_overflow"],
        }
