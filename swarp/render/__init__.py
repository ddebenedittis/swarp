"""swarp rendering: a pygame-backed interactive viewer plus headless frame/video export.

The heavy pieces (``Renderer``, ``Viewer``) depend on the optional ``viz`` extra
(``pip install swarp[viz]``) and are imported lazily, so ``swarp.render.geometry`` — pure
numpy state extraction — stays importable without pygame installed.
"""

from __future__ import annotations

import os
from typing import Any

# Set before any submodule imports pygame, so the SDL banner never hits stdout.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

__all__ = [
    "Camera",
    "RenderGeometry",
    "Style",
    "Viewer",
    "animate",
    "extract_geometry",
    "record_frames",
    "render_frame",
    "save_video",
    "to_html5_video",
]

# Attribute -> (submodule, name-in-submodule). Lazy so importing the package never forces
# pygame to load (only swarp.render.geometry is pygame-free).
_LAZY = {
    "RenderGeometry": ("swarp.render.geometry", "RenderGeometry"),
    "extract_geometry": ("swarp.render.geometry", "extract_geometry"),
    "Camera": ("swarp.render.camera", "Camera"),
    "Style": ("swarp.render.style", "Style"),
    "render_frame": ("swarp.render.renderer", "render_frame"),
    "save_video": ("swarp.render.video", "save_video"),
    "record_frames": ("swarp.render.video", "record_frames"),
    "to_html5_video": ("swarp.render.notebook", "to_html5_video"),
    "animate": ("swarp.render.notebook", "animate"),
    "Viewer": ("swarp.render.viewer", "Viewer"),
}


def __getattr__(name: str) -> Any:
    import importlib

    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module_name), attr)
