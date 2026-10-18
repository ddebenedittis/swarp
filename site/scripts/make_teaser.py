"""Render the project page's header teaser, three panels side by side.

Runs in the swarp repository's own environment (it needs the ``viz`` extra) and writes only
``site/src/assets/teaser.webm``:

    uv pip install -e '.[viz]'
    python site/scripts/make_teaser.py

Every agent is driven by the scripted goal-seeking controller, not a trained policy. The
panels are one navigation env with its communication graph, the mixed holonomic /
diff-drive / bicycle fleet, and a 4x4 grid of sixteen envs of one batch stepping together.
The header lays a 40% black scrim over the clip, so the panels use the dark style.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must precede the first pygame import, so no display is ever required.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from swarp import Environment  # noqa: E402
from swarp.render.demo import (  # noqa: E402
    MixedVisualizationScenario,
    VisualizationScenario,
    goal_seeking_policy,
)
from swarp.render.geometry import extract_geometry  # noqa: E402
from swarp.render.renderer import render_frame  # noqa: E402
from swarp.render.style import Style  # noqa: E402
from swarp.render.video import iter_rollout_frames  # noqa: E402

DEVICE = "cpu"
SEED = 3
TILE = 640  # px; three square panels give a 3:1 strip the header crops from the sides
GAP = 6  # px of background between panels and between grid cells
GRID = 4  # the batch panel is GRID x GRID envs
BASE = {"bounds", "obstacles", "goals", "agents", "heading"}


def _single(scenario, steps: int, overlays: set[str], style: Style):
    env = Environment(scenario, n_envs=1, device=DEVICE, dt=0.05, seed=SEED)
    return iter_rollout_frames(
        env, goal_seeking_policy(env), steps, size=(TILE, TILE), overlays=overlays, style=style
    )


def _batch(steps: int, style: Style):
    env = Environment(
        VisualizationScenario(n_agents=6, n_obstacles=2),
        n_envs=GRID * GRID,
        device=DEVICE,
        dt=0.05,
        seed=SEED + 1,
    )
    policy = goal_seeking_policy(env)
    cell = (TILE - (GRID - 1) * GAP) // GRID
    obs = env.reset()
    for _ in range(steps):
        with torch.no_grad():
            obs, *_ = env.step(policy(obs))
        cells = [
            render_frame(
                extract_geometry(env.world, i, scenario=env.scenario),
                size=(cell, cell),
                overlays=BASE,
                style=style,
            )
            for i in range(GRID * GRID)
        ]
        yield _tile(cells, GRID, TILE)


def _tile(cells: list[np.ndarray], cols: int, side: int) -> np.ndarray:
    """Lay ``cells`` out row-major on a ``side`` x ``side`` canvas of their background."""
    out = np.empty((side, side, 3), dtype=np.uint8)
    out[:] = cells[0][2, 2]
    cell = cells[0].shape[0]
    for k, img in enumerate(cells):
        r, c = divmod(k, cols)
        y, x = r * (cell + GAP), c * (cell + GAP)
        out[y : y + cell, x : x + cell] = img
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the project page header teaser.")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent.parent / "src/assets"
    )
    parser.add_argument("--steps", type=int, default=100, help="frames in the clip")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    style = Style.dark(trajectory_mode="fade", trajectory_len=45)
    panels = [
        _single(
            VisualizationScenario(n_agents=10, n_obstacles=3),
            args.steps,
            BASE | {"comm_lines", "trajectories"},
            style,
        ),
        _single(
            MixedVisualizationScenario(n_agents=8, n_obstacles=2),
            args.steps,
            BASE | {"trajectories"},
            style,
        ),
        _batch(args.steps, style),
    ]

    path = args.out / "teaser.webm"
    writer = imageio.get_writer(
        str(path),
        fps=25,
        codec="libvpx-vp9",
        output_params=[
            *("-b:v", "0", "-crf", "34"),
            *("-row-mt", "1", "-deadline", "good", "-cpu-used", "2"),
        ],
    )
    try:
        for frames in zip(*panels, strict=True):
            gap = np.broadcast_to(frames[0][2, 2], (TILE, GAP, 3))
            row = [frames[0]]
            for frame in frames[1:]:
                row += [gap, frame]
            writer.append_data(np.ascontiguousarray(np.concatenate(row, axis=1)))
    finally:
        writer.close()
    print(f"wrote {path}  ({args.steps} frames)")


if __name__ == "__main__":
    main()
