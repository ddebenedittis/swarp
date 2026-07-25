"""Runnable demo: ``python -m wmas.render.demo``.

With no ``--save`` it opens the interactive window (pan/zoom, toggle overlays with the
per-overlay keys, ``[`` / ``]`` to switch env, drag an agent, right-click to move its goal,
space to pause). With ``--save PATH`` it renders a rollout to a video (.mp4/.webm) headlessly.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from wmas import Environment, NavigationScenario
from wmas.core.config import WorldConfig
from wmas.core.world import World
from wmas.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from wmas.render.overlays import DEFAULT_ENABLED
from wmas.render.style import THEMES
from wmas.sensors import Lidar

MODEL_CHOICES = ("holonomic", "diff-drive", "bicycle", "mixed")
MODEL_BY_NAME = {
    "holonomic": DynamicsModel.HOLONOMIC,
    "diff-drive": DynamicsModel.DIFF_DRIVE,
    "bicycle": DynamicsModel.KINEMATIC_BICYCLE,
}


class VisualizationScenario(NavigationScenario):
    """Navigation demo scenario with viewer-only lidar rays and communication links."""

    def __init__(self, *args, lidar_rays: int = 16, lidar_range: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lidar = Lidar(n_rays=lidar_rays, max_range=lidar_range, backend="warp")
        self.comm_range = self.neighbor_radius or lidar_range

    def render_extras(self, env_idx: int) -> dict[str, np.ndarray]:
        with torch.no_grad():
            world = self.world
            ranges = self.lidar.scan(world)[env_idx]
            pos = world.state.pos[env_idx]
            theta = world.state.theta[env_idx]
            offsets = self.lidar.angle_start + torch.arange(
                self.lidar.n_rays, device=pos.device, dtype=pos.dtype
            ) * (2.0 * math.pi / self.lidar.n_rays)
            angles = theta.unsqueeze(-1) + offsets
            dirs = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
            starts = pos.unsqueeze(1).expand(-1, self.lidar.n_rays, -1)
            ends = starts + ranges.unsqueeze(-1) * dirs
            lidar = torch.stack((starts, ends), dim=2).reshape(-1, 2, 2).cpu().numpy()

            diff = pos[:, None, :] - pos[None, :, :]
            dist = torch.linalg.norm(diff, dim=-1)
            mask = torch.triu(dist <= self.comm_range, diagonal=1)
            pairs = torch.nonzero(mask, as_tuple=False)
            if pairs.numel() == 0:
                comm_lines = np.empty((0, 2, 2), dtype=np.float64)
            else:
                comm_lines = torch.stack((pos[pairs[:, 0]], pos[pairs[:, 1]]), dim=1).cpu().numpy()
        return {
            "lidar": lidar,
            "lidar_by_agent": lidar.reshape(self.n_agents, self.lidar.n_rays, 2, 2),
            "comm_lines": comm_lines,
        }


class MixedVisualizationScenario(VisualizationScenario):
    """Navigation demo with heterogeneous 2D dynamics models."""

    def make_world(self, n_envs, device, dt, substeps, dtype) -> World:
        cycle = (
            DynamicsModel.HOLONOMIC,
            DynamicsModel.DIFF_DRIVE,
            DynamicsModel.KINEMATIC_BICYCLE,
        )
        configs = [
            AgentConfig(
                model=cycle[i % len(cycle)],
                ctrl_mode=ControlMode.VELOCITY,
                radius=self.agent_radius,
                max_speed=self.max_speed,
                max_accel=2.0 * self.max_speed,
            )
            for i in range(self.n_agents)
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
        self._nbr_cache = None
        self._prev_dist = None
        self._eager_k_all = -1
        self._eager_k = -1
        self._k_obs = min(self.neighbor_obs, world_config.max_neighbors)
        self._fused_ready = False
        self._handle_version = 0
        return self.world


def build_env(
    n_envs: int,
    n_agents: int,
    n_obstacles: int,
    device: str,
    lidar_rays: int = 16,
    lidar_range: float = 1.0,
    model: str = "holonomic",
) -> Environment:
    scenario_cls = MixedVisualizationScenario if model == "mixed" else VisualizationScenario
    scenario = scenario_cls(
        n_agents=n_agents,
        n_obstacles=n_obstacles,
        world_size=1.0,
        neighbor_radius=0.6,
        lidar_rays=lidar_rays,
        lidar_range=lidar_range,
        model=MODEL_BY_NAME.get(model, DynamicsModel.HOLONOMIC),
    )
    env = Environment(scenario, n_envs=n_envs, device=device, dt=0.1, seed=0)
    env.reset()
    return env


def goal_seeking_policy(env: Environment):
    """A trivial proportional controller so the demo actually moves."""

    def policy(_obs):
        world = env.world
        to_goal = world.goals - world.state.pos
        dist = torch.linalg.norm(to_goal, dim=-1).clamp_min(1e-6)
        desired = torch.atan2(to_goal[..., 1], to_goal[..., 0])
        heading_err = torch.atan2(
            torch.sin(desired - world.state.theta), torch.cos(desired - world.state.theta)
        )
        actions = torch.zeros(
            env.n_envs, env.n_agents, world.act_dim, dtype=env.dtype, device=env.device
        )
        for i, cfg in enumerate(world.agent_configs):
            if cfg.model == DynamicsModel.HOLONOMIC:
                actions[:, i, :2] = torch.clamp(to_goal[:, i], -1.0, 1.0)
            elif cfg.model == DynamicsModel.DIFF_DRIVE:
                actions[:, i, 0] = torch.clamp(dist[:, i], -cfg.max_speed, cfg.max_speed)
                actions[:, i, 1] = torch.clamp(
                    2.5 * heading_err[:, i], -cfg.max_ang_vel, cfg.max_ang_vel
                )
            elif cfg.model == DynamicsModel.KINEMATIC_BICYCLE:
                actions[:, i, 0] = torch.clamp(
                    dist[:, i] - world.state.speed[:, i], -cfg.max_accel, cfg.max_accel
                )
                actions[:, i, 1] = torch.clamp(heading_err[:, i], -cfg.max_steer, cfg.max_steer)
        return actions

    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(description="wmas viewer demo")
    parser.add_argument(
        "--save", default=None, help="write a video (.mp4 or .webm) here instead of a window"
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--obstacles", type=int, default=2)
    parser.add_argument("--size", type=int, default=700)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mosaic", action="store_true", help="grid of all envs + focus pane")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lidar-rays", type=int, default=16)
    parser.add_argument("--lidar-range", type=float, default=1.0)
    parser.add_argument("--model", choices=MODEL_CHOICES, default="holonomic")
    parser.add_argument("--color-mode", choices=("agent", "model"), default="agent")
    parser.add_argument("--lidar-mode", choices=("none", "rays", "area", "both"), default="rays")
    parser.add_argument("--trajectory", choices=("none", "trail", "fade"), default="none")
    parser.add_argument("--trail-len", type=int, default=80)
    parser.add_argument("--theme", choices=tuple(THEMES), default="light")
    parser.add_argument(
        "--supersample", type=int, default=2, help="offscreen AA factor (1 disables)"
    )
    args = parser.parse_args(argv)

    env = build_env(
        args.envs,
        args.agents,
        args.obstacles,
        args.device,
        lidar_rays=args.lidar_rays,
        lidar_range=args.lidar_range,
        model=args.model,
    )
    policy = goal_seeking_policy(env)
    size = (args.size, args.size)
    overlays = set(DEFAULT_ENABLED) | {"comm_lines", "lidar", "trajectories"}
    style = THEMES[args.theme](
        color_mode=args.color_mode,
        lidar_mode=args.lidar_mode,
        trajectory_mode=args.trajectory,
        trajectory_len=args.trail_len,
        supersample=args.supersample,
    )

    if args.save:
        from wmas.render.video import save_video

        out = save_video(
            env,
            args.save,
            action_fn=policy,
            n_steps=args.steps,
            fps=args.fps,
            size=size,
            overlays=overlays,
            style=style,
        )
        print(f"saved {out}")
        return out

    from wmas.render.viewer import Viewer

    viewer = Viewer(
        env, size=size, mosaic=args.mosaic, fps=args.fps, overlays=overlays, style=style
    )
    viewer.run(action_fn=policy)
    return None


if __name__ == "__main__":
    main()
