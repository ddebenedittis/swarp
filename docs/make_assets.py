"""Regenerate the documentation screenshots and the demo clip.

    uv pip install -e '.[viz]'
    python docs/make_assets.py

Everything renders headlessly on the CPU (a few seconds per asset) and lands in
``docs/_static/``. The outputs are committed, so this only needs re-running when the
renderer or the scenarios change appearance.

The navigation shots reuse :mod:`swarp.render.demo`'s scenarios and controller, which
already attach the lidar and communication-link ``render_extras`` the viewer draws.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must precede the first pygame import, so no display is ever required.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402

from swarp import Environment, FlockingScenario, PushTScenario  # noqa: E402
from swarp.render.demo import (  # noqa: E402
    MixedVisualizationScenario,
    VisualizationScenario,
    goal_seeking_policy,
)
from swarp.render.style import Style  # noqa: E402
from swarp.render.video import iter_rollout_frames  # noqa: E402

DEVICE = "cpu"
SEED = 3

NAV_OVERLAYS = {"bounds", "obstacles", "goals", "agents", "heading", "lidar", "comm_lines"}


def _drive(env: Environment, steps: int, *, policy=None) -> None:
    """Advance the env so the shot shows a task in progress rather than a fresh reset."""
    torch.manual_seed(SEED)
    obs = env.scenario.observations()
    zero = torch.zeros(env.n_envs, env.n_agents, env.act_dim, dtype=env.dtype)
    with torch.no_grad():
        for _ in range(steps):
            obs, *_ = env.step(zero if policy is None else policy(obs))


def _shoot(env: Environment, path: Path, size: tuple[int, int], **kwargs) -> None:
    frame = env.render(mode="rgb_array", size=size, **kwargs)
    imageio.imwrite(path, frame)
    print(f"wrote {path}  {frame.shape[1]}x{frame.shape[0]}")


def hero(out: Path) -> None:
    env = Environment(
        VisualizationScenario(n_agents=8, n_obstacles=3, lidar_rays=16, lidar_range=0.3),
        n_envs=4,
        device=DEVICE,
        dt=0.05,
        seed=SEED,
    )
    env.reset()
    _drive(env, 18, policy=goal_seeking_policy(env))
    _shoot(env, out / "hero.png", (1100, 700), overlays=NAV_OVERLAYS | {"velocity"})


def mosaic(out: Path) -> None:
    env = Environment(
        VisualizationScenario(n_agents=6, n_obstacles=2, lidar_rays=12, lidar_range=0.3),
        n_envs=9,
        device=DEVICE,
        dt=0.05,
        seed=SEED,
    )
    env.reset()
    _drive(env, 30, policy=goal_seeking_policy(env))
    _shoot(env, out / "mosaic.png", (1100, 760), mosaic=True)


def mixed(out: Path) -> None:
    env = Environment(
        MixedVisualizationScenario(n_agents=6, n_obstacles=2, lidar_rays=12, lidar_range=0.3),
        n_envs=4,
        device=DEVICE,
        dt=0.05,
        seed=SEED,
    )
    env.reset()
    _drive(env, 50, policy=goal_seeking_policy(env))
    _shoot(
        env,
        out / "mixed.png",
        (960, 720),
        overlays=NAV_OVERLAYS | {"velocity"},
        style=Style.light(color_mode="model"),
    )


def flocking(out: Path) -> None:
    env = Environment(
        FlockingScenario(n_agents=24, neighbor_radius=0.45),
        n_envs=4,
        device=DEVICE,
        dt=0.05,
        seed=SEED,
    )
    env.reset()
    _drive(env, 40)
    _shoot(
        env,
        out / "flocking.png",
        (960, 720),
        overlays={"bounds", "agents", "heading", "velocity", "neighbor_graph"},
    )


def pusht(out: Path) -> None:
    env = Environment(
        PushTScenario(n_agents=4),
        n_envs=4,
        device=DEVICE,
        dt=0.05,
        substeps=16,
        seed=SEED,
    )
    env.reset()
    _drive(env, 40)
    _shoot(
        env,
        out / "pusht.png",
        (860, 760),
        overlays={"bounds", "obstacles", "goal_pose", "agents", "heading", "ids"},
    )


def clip(out: Path, steps: int) -> None:
    """Encode the rollout clip embedded on the landing and visualization pages.

    Lidar is deliberately off here: 8 agents x 16 rays in motion reads as noise at video
    size (the still shots keep it, where the eye can settle on one fan). Fading trails
    carry the motion instead, and the loop ends while the agents are still converging —
    a longer clip just holds on a parked fleet.

    ``save_video`` would do this in one call, but it does not expose an encoder quality
    knob, so the frames come from the same generator ``save_video`` uses and the writer is
    opened here with an explicit CRF.
    """
    env = Environment(
        VisualizationScenario(n_agents=10, n_obstacles=3, lidar_rays=16, lidar_range=0.3),
        n_envs=4,
        device=DEVICE,
        dt=0.05,
        seed=SEED,
    )
    path = out / "demo.webm"
    writer = imageio.get_writer(
        str(path),
        fps=25,
        codec="libvpx-vp9",
        output_params=[
            # Constant-quality VP9; the default CRF leaves the file several MB.
            *("-b:v", "0", "-crf", "30"),
            *("-row-mt", "1", "-deadline", "good", "-cpu-used", "2"),
        ],
    )
    try:
        for frame in iter_rollout_frames(
            env,
            goal_seeking_policy(env),
            steps,
            size=(800, 448),
            overlays=(NAV_OVERLAYS - {"lidar"}) | {"trajectories"},
            style=Style.light(trajectory_mode="fade", trajectory_len=45),
        ):
            writer.append_data(frame)
    finally:
        writer.close()
    print(f"wrote {path}  ({steps} frames)")


ASSETS = {
    "hero": hero,
    "mosaic": mosaic,
    "mixed": mixed,
    "flocking": flocking,
    "pusht": pusht,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the swarp docs assets.")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "_static")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[*ASSETS, "clip"],
        help="subset to regenerate (default: all)",
    )
    parser.add_argument("--clip-steps", type=int, default=100, help="frames in demo.webm")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else {*ASSETS, "clip"}
    for name, fn in ASSETS.items():
        if name in wanted:
            fn(args.out)
    if "clip" in wanted:
        clip(args.out, args.clip_steps)


if __name__ == "__main__":
    main()
