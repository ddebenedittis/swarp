"""wmas rendering: a pygame-backed interactive viewer plus headless frame/video export.

The heavy pieces (``Renderer``, ``Viewer``) depend on the optional ``viz`` extra
(``pip install wmas[viz]``) and are imported lazily, so ``wmas.render.geometry`` — pure
numpy state extraction — stays importable without pygame installed.
"""

from __future__ import annotations

import os
from typing import Any

# Set before any submodule imports pygame, so the SDL banner never hits stdout.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

__all__ = ["RenderGeometry", "extract_geometry", "Camera", "Viewer"]


def __getattr__(name: str) -> Any:
    # Lazy re-exports so importing the package never forces pygame to load.
    if name in ("RenderGeometry", "extract_geometry"):
        from wmas.render import geometry

        return getattr(geometry, name)
    if name == "Camera":
        from wmas.render.camera import Camera

        return Camera
    if name == "Viewer":
        from wmas.render.viewer import Viewer

        return Viewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
