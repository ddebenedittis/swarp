"""Notebook helpers: embed a rollout as an inline HTML5 video.

``animate(env, ...)`` is the one-liner for a Jupyter/Colab cell; ``to_html5_video(frames)``
encodes an already-recorded frame list. Both return an ``IPython.display.HTML`` when IPython
is importable, else the raw HTML string (so this module has no hard IPython dependency).
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

from wmas.render.video import frames_to_video, record_frames


def to_html5_video(frames, fps: int = 30, *, loop: bool = True):
    """Encode ``frames`` to mp4 and return an inline ``<video>`` element."""
    handle, path = tempfile.mkstemp(suffix=".mp4")
    os.close(handle)
    try:
        frames_to_video(frames, path, fps=fps)
        data = Path(path).read_bytes()
    finally:
        os.remove(path)

    b64 = base64.b64encode(data).decode("ascii")
    loop_attr = " loop" if loop else ""
    html = (
        f'<video controls autoplay{loop_attr} style="max-width:100%" '
        f'src="data:video/mp4;base64,{b64}"></video>'
    )
    try:
        from IPython.display import HTML

        return HTML(html)
    except ModuleNotFoundError:
        return html


def animate(env, *, action_fn=None, n_steps: int = 100, fps: int = 30, **kwargs):
    """Record a rollout and return an inline HTML5 video (records then encodes)."""
    frames = record_frames(env, action_fn, n_steps, **kwargs)
    return to_html5_video(frames, fps=fps)
