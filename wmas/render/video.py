"""Frame sequences -> video, and one-call rollout recording.

Uses imageio's streaming writer (bundled ffmpeg), so frames are encoded as they are produced
(no need to hold a whole episode in memory). Only real video containers are supported —
``.mp4`` (H.264) and ``.webm`` (VP9); ``.gif`` is intentionally rejected.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from wmas.render.geometry import extract_geometry
from wmas.render.renderer import render_frame
from wmas.render.style import Style

# Only real video containers — gif is intentionally unsupported. webm needs an explicit
# VP9 codec (ffmpeg's default libx264 is invalid in a webm container).
_VIDEO_CODECS = {".mp4": "libx264", ".webm": "libvpx-vp9"}


def _open_writer(path, fps: int):
    import imageio.v2 as imageio

    suffix = Path(path).suffix.lower()
    codec = _VIDEO_CODECS.get(suffix)
    if codec is None:
        raise ValueError(
            f"unsupported video extension {suffix or '(none)'!r}; "
            "wmas writes .mp4 (H.264) or .webm (VP9) only"
        )
    return imageio.get_writer(str(path), fps=fps, codec=codec)


def frames_to_video(frames, path, fps: int = 30) -> str:
    """Write an iterable of ``(H, W, 3)`` uint8 frames to ``path`` (.mp4 or .webm)."""
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


def _zero_actions(env):
    return torch.zeros(
        env.n_envs, env.n_agents, env.world.act_dim, dtype=env.dtype, device=env.device
    )


def iter_rollout_frames(
    env,
    action_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    n_steps: int = 100,
    *,
    env_index: int = 0,
    size: tuple[int, int] = (600, 600),
    overlays: set[str] | None = None,
    style: Style | None = None,
    reset: bool = True,
):
    """Yield one rendered frame per step of a rollout of env ``env_index`` (under no_grad).

    ``action_fn`` maps the latest observation to an action of shape
    ``[n_envs, n_agents, world.act_dim]``; ``None`` means zero actions.
    """
    obs = env.reset() if reset else env.scenario.observations()
    for _ in range(n_steps):
        with torch.no_grad():
            actions = _zero_actions(env) if action_fn is None else action_fn(obs)
            obs, *_ = env.step(actions)
            geometry = extract_geometry(env.world, env_index, scenario=env.scenario)
            frame = render_frame(geometry, size=size, overlays=overlays, style=style)
        yield frame


def record_frames(env, action_fn=None, n_steps: int = 100, **kwargs) -> list[np.ndarray]:
    """Collect a rollout into a list of frames in memory (for notebooks / custom encoding)."""
    return list(iter_rollout_frames(env, action_fn, n_steps, **kwargs))


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

    ``path`` must end in ``.mp4`` (H.264) or ``.webm`` (VP9). Only env ``env_index`` is
    rendered. See :func:`iter_rollout_frames` for the rollout semantics.
    """
    writer = _open_writer(path, fps)
    try:
        for frame in iter_rollout_frames(
            env,
            action_fn,
            n_steps,
            env_index=env_index,
            size=size,
            overlays=overlays,
            style=style,
            reset=reset,
        ):
            writer.append_data(frame)
    finally:
        writer.close()
    return str(Path(path))
