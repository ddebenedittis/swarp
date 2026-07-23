"""Runnable demo: ``python -m wmas.render.demo``.

With no ``--save`` it opens the interactive window (pan/zoom, toggle overlays with the
per-overlay keys, ``[`` / ``]`` to switch env, drag an agent, right-click to move its goal,
space to pause). With ``--save PATH`` it renders a rollout to a video (.mp4/.webm) headlessly.
"""

from __future__ import annotations

import argparse

import torch

from wmas import Environment, NavigationScenario


def build_env(n_envs: int, n_agents: int, n_obstacles: int, device: str) -> Environment:
    scenario = NavigationScenario(
        n_agents=n_agents, n_obstacles=n_obstacles, world_size=1.0, neighbor_radius=0.6
    )
    env = Environment(scenario, n_envs=n_envs, device=device, dt=0.1, seed=0)
    env.reset()
    return env


def goal_seeking_policy(env: Environment):
    """A trivial proportional controller so the demo actually moves."""

    def policy(_obs):
        return torch.clamp(env.world.goals - env.world.state.pos, -1.0, 1.0).to(env.dtype)

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
    args = parser.parse_args(argv)

    env = build_env(args.envs, args.agents, args.obstacles, args.device)
    policy = goal_seeking_policy(env)
    size = (args.size, args.size)

    if args.save:
        from wmas.render.video import save_video

        out = save_video(
            env, args.save, action_fn=policy, n_steps=args.steps, fps=args.fps, size=size
        )
        print(f"saved {out}")
        return out

    from wmas.render.viewer import Viewer

    Viewer(env, size=size, mosaic=args.mosaic, fps=args.fps).run(action_fn=policy)
    return None


if __name__ == "__main__":
    main()
