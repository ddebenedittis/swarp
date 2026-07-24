"""Toggleable draw layers, as an ordered registry.

Each :class:`Overlay` is a named draw function ``(surface, geometry, camera, style)`` with a
default keyboard toggle and a default-on flag. The renderer walks :data:`OVERLAYS` in order
and calls each one whose name is in the enabled set, so registry order == draw order.

Overlays are read-only w.r.t. the simulator; they only consume a :class:`RenderGeometry`.
The ``lidar`` overlay is a no-op until ``geometry.extras["lidar"]`` is populated by a future
lidar sensor — so this layer can ship before the sensor does.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pygame

from wmas.render.camera import Camera
from wmas.render.geometry import RenderGeometry
from wmas.render.style import Style

DrawFn = Callable[["pygame.Surface", RenderGeometry, Camera, Style], None]


@dataclass(frozen=True)
class Overlay:
    name: str
    draw: DrawFn
    key: str | None
    default_on: bool


def _p(camera: Camera, world_pt) -> tuple[int, int]:
    s = camera.world_to_screen(world_pt)
    return int(round(float(s[0]))), int(round(float(s[1])))


def _r_px(camera: Camera, r: float, floor: int = 1) -> int:
    return max(floor, int(round(float(r) * camera.scale)))


def _draw_bounds(surface, g, camera, style):
    if g.bounds is None:
        return
    x0, x1, y0, y1 = g.bounds
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    pygame.draw.lines(
        surface, style.bounds_color, True, [_p(camera, c) for c in corners], style.line_width
    )


def _draw_obstacles(surface, g, camera, style):
    if g.obstacle_pos is None:
        return
    for i in range(g.obstacle_pos.shape[0]):
        pygame.draw.circle(
            surface,
            style.obstacle_color,
            _p(camera, g.obstacle_pos[i]),
            _r_px(camera, g.obstacle_radius[i]),
        )


def _draw_neighbor_graph(surface, g, camera, style):
    for i, j in g.edges:
        pygame.draw.line(
            surface, style.edge_color, _p(camera, g.pos[i]), _p(camera, g.pos[j]), style.edge_width
        )


def _draw_goals(surface, g, camera, style):
    if g.goals is None:
        return
    for i in range(g.n_agents):
        center = _p(camera, g.goals[i])
        color = style.agent_color(i)
        r = _r_px(camera, g.radius[i], floor=style.goal_min_px)
        pygame.draw.circle(surface, color, center, r, style.goal_ring_width)
        pygame.draw.circle(surface, color, center, 2)  # center dot


def _draw_agents(surface, g, camera, style):
    for i in range(g.n_agents):
        center = _p(camera, g.pos[i])
        r = _r_px(camera, g.radius[i], floor=style.agent_min_px)
        pygame.draw.circle(surface, style.agent_color(i), center, r)
        pygame.draw.circle(surface, style.agent_outline, center, r, 1)


def _draw_heading(surface, g, camera, style):
    for i in range(g.n_agents):
        pos = g.pos[i]
        theta = float(g.theta[i])
        reach = float(g.radius[i]) * style.heading_len_factor
        tip = (pos[0] + math.cos(theta) * reach, pos[1] + math.sin(theta) * reach)
        pygame.draw.line(
            surface, style.heading_color, _p(camera, pos), _p(camera, tip), style.line_width
        )


def _draw_velocity(surface, g, camera, style):
    for i in range(g.n_agents):
        pos = g.pos[i]
        tip = (
            pos[0] + float(g.vel[i, 0]) * style.velocity_scale,
            pos[1] + float(g.vel[i, 1]) * style.velocity_scale,
        )
        pygame.draw.line(
            surface, style.velocity_color, _p(camera, pos), _p(camera, tip), style.line_width
        )


def _draw_ids(surface, g, camera, style):
    font = _get_font(style.font_size)
    for i in range(g.n_agents):
        label = font.render(str(i), True, style.text_color)
        cx, cy = _p(camera, g.pos[i])
        surface.blit(label, (cx + 4, cy - style.font_size))


def _draw_comm_lines(surface, g, camera, style):
    """Draw inter-agent communication lines from ``extras['comm_lines']``.

    Expected format: world-space segments with shape ``(n_pairs, 2, 2)`` —
    ``[endpoint_a_xy, endpoint_b_xy]`` per pair. Absent/empty -> no-op. Restores
    the VMAS comm-line visual (agent pairs within a communication range).
    """
    segs = g.extras.get("comm_lines")
    if segs is None:
        return
    segs = np.asarray(segs, dtype=np.float64)
    if segs.size == 0:
        return
    color = getattr(style, "comm_line_color", style.edge_color)
    for start, end in segs.reshape(-1, 2, 2):
        pygame.draw.line(surface, color, _p(camera, start), _p(camera, end), style.edge_width)


def _draw_lidar(surface, g, camera, style):
    """Draw lidar rays if a sensor supplied them via ``extras['lidar']``.

    Expected (tentative) format: an array of world-space segments with shape
    ``(n_rays, 2, 2)`` — ``[start_xy, end_xy]`` per ray. Absent/empty -> no-op.
    """
    rays = g.extras.get("lidar")
    if rays is None:
        return
    rays = np.asarray(rays, dtype=np.float64)
    if rays.size == 0:
        return
    for start, end in rays.reshape(-1, 2, 2):
        pygame.draw.line(surface, style.lidar_color, _p(camera, start), _p(camera, end), 1)


_FONT_CACHE: dict[int, pygame.font.Font] = {}


def _get_font(size: int) -> pygame.font.Font:
    if not pygame.font.get_init():
        pygame.font.init()
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.Font(None, size)
    return _FONT_CACHE[size]


# Registry order == draw order (back to front).
OVERLAYS: tuple[Overlay, ...] = (
    Overlay("bounds", _draw_bounds, "b", True),
    Overlay("obstacles", _draw_obstacles, "o", True),
    Overlay("neighbor_graph", _draw_neighbor_graph, "n", False),
    Overlay("goals", _draw_goals, "g", True),
    Overlay("agents", _draw_agents, None, True),
    Overlay("heading", _draw_heading, "h", True),
    Overlay("velocity", _draw_velocity, "v", False),
    Overlay("ids", _draw_ids, "i", False),
    Overlay("comm_lines", _draw_comm_lines, "c", False),
    Overlay("lidar", _draw_lidar, "l", True),
)

DEFAULT_ENABLED: frozenset[str] = frozenset(o.name for o in OVERLAYS if o.default_on)
