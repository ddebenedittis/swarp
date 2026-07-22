"""Frame sequences -> video/gif, and one-call rollout recording.

Uses imageio's streaming writer, so frames are encoded as they are produced (no need to
hold a whole episode in memory). The container is chosen from the file extension: ``.mp4``
(and friends) via the bundled ffmpeg, ``.gif`` via pillow.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from wmas.render.geometry import extract_geometry
from wmas.render.renderer import render_frame
from wmas.render.style import Style


def _open_writer(path, fps: int):
    import imageio.v2 as imageio

    if Path(path).suffix.lower() == ".gif":
        # pillow's gif writer wants per-frame duration in ms, not fps; loop=0 => forever.
        return imageio.get_writer(str(path), duration=1000.0 / fps, loop=0)
    return imageio.get_writer(str(path), fps=fps)


def frames_to_video(frames, path, fps: int = 30) -> str:
    """Write an iterable of ``(H, W, 3)`` uint8 frames to ``path`` (mp4/gif by extension)."""
    writer = _open_writer(path, fps)
    n = 0
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame))
            n += 1
    finally:
        writer.close()
    if n == 0:
        raise ValueError("no frames to write")
    return str(path)


def save_video(
    env,
    path,
    *,
    action_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    n_steps: int = 100,
    fps: int = 30,
    env_index: int = 0,
    size: tuple[int, int] = (600, 600),
    overlays: set[str] | None = None,
    style: Style | None = None,
    reset: bool = True,
) -> str:
    """Roll ``env`` forward ``n_steps`` and stream one rendered frame per step to ``path``.

    ``action_fn`` maps the latest observation to an action tensor of shape
    ``[n_envs, n_agents, world.act_dim]``; if ``None``, zero actions are used. Only env
    ``env_index`` of the batch is rendered. Runs under ``torch.no_grad()``.
    """
    obs = env.reset() if reset else env.scenario.observations()
    writer = _open_writer(path, fps)
    try:
        with torch.no_grad():
            for _ in range(n_steps):
                if action_fn is None:
                    actions = torch.zeros(
                        env.n_envs,
                        env.n_agents,
                        env.world.act_dim,
                        dtype=env.dtype,
                        device=env.device,
                    )
                else:
                    actions = action_fn(obs)
                obs, *_ = env.step(actions)
                geometry = extract_geometry(env.world, env_index, scenario=env.scenario)
                writer.append_data(
                    render_frame(geometry, size=size, overlays=overlays, style=style)
                )
    finally:
        writer.close()
    return str(Path(path))
