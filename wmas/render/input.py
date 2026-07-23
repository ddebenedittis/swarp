"""Map pygame input events to view state and camera changes.

Kept free of any window/display calls so it is unit-testable headlessly: feed synthetic
``pygame.event.Event`` objects to :meth:`InteractionController.handle_event` and assert on
:class:`ViewState` / the :class:`Camera`. The live window loop (in ``viewer.py``) is a thin
driver over this. Write-back (drag/place) is added in a later milestone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pygame

from wmas.render.camera import Camera
from wmas.render.geometry import RenderGeometry
from wmas.render.overlays import OVERLAYS

_ZOOM_STEP = 1.1
_PAN_BUTTON = 2  # middle mouse


def pick_agent(
    geometry: RenderGeometry, camera: Camera, screen_xy, extra_px: float = 4.0
) -> int | None:
    """Index of the agent whose drawn disk covers ``screen_xy`` (nearest center wins), else None."""
    if geometry.n_agents == 0:
        return None
    centers = camera.world_to_screen(geometry.pos)
    sx, sy = float(screen_xy[0]), float(screen_xy[1])
    dist = np.hypot(centers[:, 0] - sx, centers[:, 1] - sy)
    hit_radius = np.maximum(geometry.radius * camera.scale, 1.0) + extra_px
    hits = np.nonzero(dist <= hit_radius)[0]
    if hits.size == 0:
        return None
    return int(hits[np.argmin(dist[hits])])


@dataclass
class ViewState:
    n_envs: int
    enabled: set[str]
    focus_env: int = 0
    paused: bool = False
    hover_agent: int | None = None
    quit: bool = False


_OVERLAY_KEYS: dict[int, str] | None = None


def _overlay_keys() -> dict[int, str]:
    # getattr(pygame, "K_<letter>") avoids pygame.key.key_code(), which warns before init().
    global _OVERLAY_KEYS
    if _OVERLAY_KEYS is None:
        _OVERLAY_KEYS = {getattr(pygame, f"K_{o.key}"): o.name for o in OVERLAYS if o.key}
    return _OVERLAY_KEYS


class InteractionController:
    """Stateful translator from pygame events to :class:`ViewState` + :class:`Camera` updates."""

    def __init__(
        self,
        state: ViewState,
        camera: Camera,
        geometry_getter: Callable[[], RenderGeometry] | None = None,
        tile_resolver: Callable[[tuple], int | None] | None = None,
    ) -> None:
        self.state = state
        self.camera = camera
        self._geometry_getter = geometry_getter
        self._tile_resolver = tile_resolver
        self._panning = False
        self._last_mouse = (0.0, 0.0)

    def handle_event(self, event) -> None:
        et = event.type
        if et == pygame.KEYDOWN:
            self._on_keydown(event)
        elif et == pygame.MOUSEWHEEL:
            self.camera.zoom_at(_ZOOM_STEP**event.y, self._last_mouse)
        elif et == pygame.MOUSEMOTION:
            self._on_motion(event)
        elif et == pygame.MOUSEBUTTONDOWN:
            if event.button == _PAN_BUTTON:
                self._panning = True
            elif event.button == 1:
                self._on_left_down(event)
        elif et == pygame.MOUSEBUTTONUP and event.button == _PAN_BUTTON:
            self._panning = False
        elif et == pygame.QUIT:
            self.state.quit = True

    def _on_left_down(self, event) -> None:
        # A left click on a mosaic tile focuses that env. (Focus-pane left clicks are
        # reserved for write-back in a later milestone.)
        if self._tile_resolver is not None:
            env_idx = self._tile_resolver(event.pos)
            if env_idx is not None:
                self.state.focus_env = env_idx

    def _on_keydown(self, event) -> None:
        s = self.state
        key = event.key
        overlays = _overlay_keys()
        if key in overlays:
            name = overlays[key]
            s.enabled.discard(name) if name in s.enabled else s.enabled.add(name)
        elif key == pygame.K_LEFTBRACKET:
            s.focus_env = (s.focus_env - 1) % s.n_envs
        elif key == pygame.K_RIGHTBRACKET:
            s.focus_env = (s.focus_env + 1) % s.n_envs
        elif key == pygame.K_SPACE:
            s.paused = not s.paused
        elif key in (pygame.K_ESCAPE, pygame.K_q):
            s.quit = True

    def _on_motion(self, event) -> None:
        pos = event.pos
        if self._panning:
            rel = getattr(event, "rel", None)
            if rel is None:
                rel = (pos[0] - self._last_mouse[0], pos[1] - self._last_mouse[1])
            self.camera.pan(float(rel[0]), float(rel[1]))
        elif self._geometry_getter is not None:
            geometry = self._geometry_getter()
            if geometry is not None:
                self.state.hover_agent = pick_agent(geometry, self.camera, pos)
        self._last_mouse = (float(pos[0]), float(pos[1]))
