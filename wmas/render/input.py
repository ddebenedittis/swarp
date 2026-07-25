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


def _obstacle_extent(geometry: RenderGeometry) -> np.ndarray:
    """Per-obstacle pick radius in world units, honoring non-circular shapes.

    A box's collision radius is unused by the simulator and is legitimately 0, so picking on
    ``obstacle_radius`` alone would make boxes undraggable. Derived only from the geometry.
    """
    extent = np.asarray(geometry.obstacle_radius, dtype=np.float64).copy()
    shape, half = geometry.obstacle_shape, geometry.obstacle_half_extents
    if shape is None or half is None:
        return extent
    is_box = shape == 1  # ObstacleShape.BOX
    is_seg = shape == 2  # ObstacleShape.SEGMENT
    extent[is_box] = np.hypot(half[is_box, 0], half[is_box, 1])
    extent[is_seg] = half[is_seg, 0] + extent[is_seg]
    return extent


def pick_obstacle(
    geometry: RenderGeometry, camera: Camera, screen_xy, extra_px: float = 4.0
) -> int | None:
    """Index of the obstacle whose drawn shape covers ``screen_xy``, else None."""
    if geometry.obstacle_pos is None or geometry.obstacle_radius is None:
        return None
    centers = camera.world_to_screen(geometry.obstacle_pos)
    sx, sy = float(screen_xy[0]), float(screen_xy[1])
    dist = np.hypot(centers[:, 0] - sx, centers[:, 1] - sy)
    hit_radius = np.maximum(_obstacle_extent(geometry) * camera.scale, 1.0) + extra_px
    hits = np.nonzero(dist <= hit_radius)[0]
    if hits.size == 0:
        return None
    return int(hits[np.argmin(dist[hits])])


def _is_help_key(event) -> bool:
    if event.key == pygame.K_F1:
        return True
    if hasattr(pygame, "K_HELP") and event.key == pygame.K_HELP:
        return True
    if hasattr(pygame, "K_QUESTION") and event.key == pygame.K_QUESTION:
        return True
    if getattr(event, "unicode", "") == "?":
        return True
    if event.key == pygame.K_SLASH and (getattr(event, "mod", 0) & pygame.KMOD_SHIFT):
        return True
    try:
        return pygame.key.name(event.key).lower() in {"f1", "?"}
    except Exception:
        return False


@dataclass
class ViewState:
    n_envs: int
    enabled: set[str]
    focus_env: int = 0
    paused: bool = False
    step_once: bool = False  # one-shot: advance a single step while paused, then clear
    show_help: bool = False
    reset_requested: bool = False
    hover_agent: int | None = None
    selected_agent: int | None = None
    selected_obstacle: int | None = None
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
        on_drag_agent: Callable[[int, tuple[float, float]], None] | None = None,
        on_drag_obstacle: Callable[[int, tuple[float, float]], None] | None = None,
        on_place_goal: Callable[[int, tuple[float, float]], None] | None = None,
        on_cycle_trajectory: Callable[[], None] | None = None,
        on_cycle_lidar: Callable[[], None] | None = None,
        on_cycle_color: Callable[[], None] | None = None,
    ) -> None:
        self.state = state
        self.camera = camera
        self._geometry_getter = geometry_getter
        self._tile_resolver = tile_resolver
        self._on_drag_agent = on_drag_agent
        self._on_drag_obstacle = on_drag_obstacle
        self._on_place_goal = on_place_goal
        self._on_cycle_trajectory = on_cycle_trajectory
        self._on_cycle_lidar = on_cycle_lidar
        self._on_cycle_color = on_cycle_color
        self._panning = False
        self._dragging_agent: int | None = None
        self._dragging_obstacle: int | None = None
        self._last_mouse = (0.0, 0.0)

    def handle_event(self, event) -> None:
        et = event.type
        if et == pygame.KEYDOWN:
            self._on_keydown(event)
        elif et == pygame.TEXTINPUT and getattr(event, "text", "") == "?":
            self.state.show_help = not self.state.show_help
        elif et == pygame.MOUSEWHEEL:
            self.camera.zoom_at(_ZOOM_STEP**event.y, self._last_mouse)
        elif et == pygame.MOUSEMOTION:
            self._on_motion(event)
        elif et == pygame.MOUSEBUTTONDOWN:
            if event.button == _PAN_BUTTON:
                self._panning = True
            elif event.button == 1:
                self._on_left_down(event)
            elif event.button == 3:
                self._on_right_down(event)
        elif et == pygame.MOUSEBUTTONUP:
            if event.button == _PAN_BUTTON:
                self._panning = False
            elif event.button == 1:
                self._dragging_agent = None
                self._dragging_obstacle = None
        elif et == pygame.QUIT:
            self.state.quit = True

    def _world_at(self, pos) -> tuple[float, float]:
        wx, wy = self.camera.screen_to_world(pos)
        return float(wx), float(wy)

    def _on_left_down(self, event) -> None:
        # In mosaic mode a left click on a tile focuses that env.
        if self._tile_resolver is not None and self._tile_resolver(event.pos) is not None:
            self.state.focus_env = self._tile_resolver(event.pos)
            return
        # Otherwise (focus pane) select the item under the cursor and begin a drag.
        if self._geometry_getter is None:
            return
        geometry = self._geometry_getter()
        if geometry is None:
            return
        idx = pick_agent(geometry, self.camera, event.pos)
        if idx is not None:
            self.state.selected_agent = idx
            self.state.selected_obstacle = None
            if self._on_drag_agent is not None:
                self._dragging_agent = idx
            return
        obs_idx = pick_obstacle(geometry, self.camera, event.pos)
        if obs_idx is not None:
            self.state.selected_obstacle = obs_idx
            self.state.selected_agent = None
            if self._on_drag_obstacle is not None:
                self._dragging_obstacle = obs_idx

    def _on_right_down(self, event) -> None:
        # Right click moves the selected agent's goal to the clicked point.
        if self.state.selected_agent is not None and self._on_place_goal is not None:
            self._on_place_goal(self.state.selected_agent, self._world_at(event.pos))

    def _on_keydown(self, event) -> None:
        s = self.state
        key = event.key
        overlays = _overlay_keys()
        shift = bool(getattr(event, "mod", 0) & pygame.KMOD_SHIFT)
        if key == pygame.K_t and self._on_cycle_trajectory is not None:
            self._on_cycle_trajectory()
        elif key == pygame.K_l and shift and self._on_cycle_lidar is not None:
            # Shift-qualified: plain "l" must fall through to toggling the lidar *overlay*,
            # whose registry key is also "l" (it was unreachable while this branch took it).
            self._on_cycle_lidar()
        elif key == pygame.K_k and self._on_cycle_color is not None:
            self._on_cycle_color()
        elif key == pygame.K_PERIOD:
            # A period cannot collide with a future overlay (those auto-bind letters), and
            # frame-advance is the debugger convention.
            s.step_once = True
        elif key in overlays:
            name = overlays[key]
            s.enabled.discard(name) if name in s.enabled else s.enabled.add(name)
        elif key == pygame.K_LEFTBRACKET:
            s.focus_env = (s.focus_env - 1) % s.n_envs
        elif key == pygame.K_RIGHTBRACKET:
            s.focus_env = (s.focus_env + 1) % s.n_envs
        elif key == pygame.K_SPACE:
            s.paused = not s.paused
        elif key == pygame.K_r:
            s.reset_requested = True
        elif _is_help_key(event):
            s.show_help = not s.show_help
        elif key in (pygame.K_ESCAPE, pygame.K_q):
            s.quit = True

    def _on_motion(self, event) -> None:
        pos = event.pos
        if self._panning:
            rel = getattr(event, "rel", None)
            if rel is None:
                rel = (pos[0] - self._last_mouse[0], pos[1] - self._last_mouse[1])
            self.camera.pan(float(rel[0]), float(rel[1]))
        elif self._dragging_agent is not None and self._on_drag_agent is not None:
            self._on_drag_agent(self._dragging_agent, self._world_at(pos))
        elif self._dragging_obstacle is not None and self._on_drag_obstacle is not None:
            self._on_drag_obstacle(self._dragging_obstacle, self._world_at(pos))
        elif self._geometry_getter is not None:
            geometry = self._geometry_getter()
            if geometry is not None:
                self.state.hover_agent = pick_agent(geometry, self.camera, pos)
        self._last_mouse = (float(pos[0]), float(pos[1]))
