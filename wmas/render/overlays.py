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
        color = style.agent_color(i, g.model[i])
        r = _r_px(camera, g.radius[i], floor=style.goal_min_px)
        pygame.draw.circle(surface, color, center, r, style.goal_ring_width)
        pygame.draw.circle(surface, color, center, 2)  # center dot


def _agent_points(pos, theta: float, radius: float, factors):
    c = math.cos(theta)
    s = math.sin(theta)
    pts = []
    for fx, fy in factors:
        pts.append((pos[0] + radius * (fx * c - fy * s), pos[1] + radius * (fx * s + fy * c)))
    return pts


def _draw_agents(surface, g, camera, style):
    for i in range(g.n_agents):
        center = _p(camera, g.pos[i])
        r = _r_px(camera, g.radius[i], floor=style.agent_min_px)
        wheel_w = max(2, r // 2)
        color = style.agent_color(i, g.model[i])
        model = int(g.model[i])
        if model == 1:  # diff-drive: compact body with two side tracks.
            body = _agent_points(
                g.pos[i],
                float(g.theta[i]),
                float(g.radius[i]),
                [
                    (0.95, 0.55),
                    (0.55, 0.9),
                    (-0.65, 0.9),
                    (-0.95, 0.55),
                    (-0.95, -0.55),
                    (-0.65, -0.9),
                    (0.55, -0.9),
                    (0.95, -0.55),
                ],
            )
            pygame.draw.polygon(surface, color, [_p(camera, p) for p in body])
            for y in (-1.05, 1.05):
                track = _agent_points(
                    g.pos[i], float(g.theta[i]), float(g.radius[i]), [(0.65, y), (-0.65, y)]
                )
                pygame.draw.line(
                    surface,
                    style.agent_outline,
                    _p(camera, track[0]),
                    _p(camera, track[1]),
                    wheel_w,
                )
            nose = _agent_points(g.pos[i], float(g.theta[i]), float(g.radius[i]), [(0.55, 0.0)])
            pygame.draw.circle(surface, style.agent_outline, _p(camera, nose[0]), max(1, r // 5))
        elif model == 2:  # kinematic bicycle: stylized car with wheels and cabin.
            body = _agent_points(
                g.pos[i],
                float(g.theta[i]),
                float(g.radius[i]),
                [
                    (1.75, 0.0),
                    (1.25, 0.65),
                    (0.25, 0.8),
                    (-1.25, 0.65),
                    (-1.55, 0.4),
                    (-1.55, -0.4),
                    (-1.25, -0.65),
                    (0.25, -0.8),
                    (1.25, -0.65),
                ],
            )
            pygame.draw.polygon(surface, color, [_p(camera, p) for p in body])
            cabin = _agent_points(
                g.pos[i],
                float(g.theta[i]),
                float(g.radius[i]),
                [(0.55, 0.38), (-0.35, 0.42), (-0.6, -0.42), (0.55, -0.38)],
            )
            pygame.draw.polygon(surface, style.background, [_p(camera, p) for p in cabin])
            pygame.draw.polygon(surface, style.agent_outline, [_p(camera, p) for p in cabin], 1)
            for x in (-0.95, 0.95):
                for y in (-0.82, 0.82):
                    wheel = _agent_points(
                        g.pos[i],
                        float(g.theta[i]),
                        float(g.radius[i]),
                        [(x + 0.25, y), (x - 0.25, y)],
                    )
                    pygame.draw.line(
                        surface,
                        style.agent_outline,
                        _p(camera, wheel[0]),
                        _p(camera, wheel[1]),
                        wheel_w,
                    )
            pygame.draw.polygon(surface, style.agent_outline, [_p(camera, p) for p in body], 1)
            front = _agent_points(g.pos[i], float(g.theta[i]), float(g.radius[i]), [(1.2, 0.0)])
            pygame.draw.circle(surface, style.background, _p(camera, front[0]), max(1, r // 5))
        elif model == 3:  # drone placeholder: quadrotor cross.
            pygame.draw.circle(surface, color, center, max(2, r // 2))
            for a in (0.0, math.pi / 2):
                arm = _agent_points(
                    g.pos[i],
                    float(g.theta[i]) + a,
                    float(g.radius[i]),
                    [(1.5, 0.0), (-1.5, 0.0)],
                )
                pygame.draw.line(
                    surface, color, _p(camera, arm[0]), _p(camera, arm[1]), style.line_width
                )
                pygame.draw.circle(surface, color, _p(camera, arm[0]), max(2, r // 3), 1)
                pygame.draw.circle(surface, color, _p(camera, arm[1]), max(2, r // 3), 1)
        else:
            pygame.draw.circle(surface, color, center, r)
        pygame.draw.circle(surface, style.agent_outline, center, r, 1)


def _draw_heading(surface, g, camera, style):
    for i in range(g.n_agents):
        if int(g.model[i]) == 0:  # holonomic agents have no meaningful body heading.
            continue
        pos = g.pos[i]
        theta = float(g.theta[i])
        reach = float(g.radius[i]) * style.heading_len_factor
        tip = (pos[0] + math.cos(theta) * reach, pos[1] + math.sin(theta) * reach)
        pygame.draw.line(
            surface, style.heading_color, _p(camera, pos), _p(camera, tip), style.line_width
        )


def _draw_trajectories(surface, g, camera, style):
    if style.trajectory_mode == "none":
        return
    trails = g.extras.get("trajectories")
    if trails is None:
        return
    trails = np.asarray(trails, dtype=np.float64)
    if trails.ndim != 3 or trails.shape[1] < 2:
        return
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for i in range(min(g.n_agents, trails.shape[0])):
        pts = [_p(camera, p) for p in trails[i]]
        color = style.agent_color(i, g.model[i])
        if style.trajectory_mode == "fade":
            n = max(1, len(pts) - 1)
            for k in range(n):
                alpha = int(
                    style.trajectory_fade_min_alpha
                    + (style.trajectory_alpha - style.trajectory_fade_min_alpha) * (k + 1) / n
                )
                pygame.draw.line(overlay, (*color, alpha), pts[k], pts[k + 1], style.line_width)
        else:
            pygame.draw.lines(
                overlay, (*color, style.trajectory_alpha), False, pts, style.line_width
            )
    surface.blit(overlay, (0, 0))


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
    if style.lidar_mode == "none":
        return
    by_agent = g.extras.get("lidar_by_agent")
    if by_agent is not None and style.lidar_mode in {"area", "both"}:
        grouped = np.asarray(by_agent, dtype=np.float64)
        if grouped.size:
            overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            for i in range(min(g.n_agents, grouped.shape[0])):
                endpoints = grouped[i, :, 1, :]
                if endpoints.shape[0] < 2:
                    continue
                color = style.agent_color(i, g.model[i])
                poly = [_p(camera, g.pos[i]), *[_p(camera, p) for p in endpoints]]
                pygame.draw.polygon(overlay, (*color, style.lidar_area_alpha), poly)
            surface.blit(overlay, (0, 0))

    if style.lidar_mode == "area":
        return

    rays = g.extras.get("lidar")
    if rays is None:
        return
    rays = np.asarray(rays, dtype=np.float64)
    if rays.size == 0:
        return
    grouped = np.asarray(by_agent, dtype=np.float64) if by_agent is not None else None
    if grouped is not None and grouped.size:
        for i in range(min(g.n_agents, grouped.shape[0])):
            color = style.agent_color(i, g.model[i])
            for start, end in grouped[i].reshape(-1, 2, 2):
                pygame.draw.line(surface, color, _p(camera, start), _p(camera, end), 1)
                pygame.draw.circle(surface, color, _p(camera, end), style.lidar_hit_px)
        return
    for start, end in rays.reshape(-1, 2, 2):
        pygame.draw.line(surface, style.lidar_color, _p(camera, start), _p(camera, end), 1)
        pygame.draw.circle(surface, style.lidar_color, _p(camera, end), style.lidar_hit_px)


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
    Overlay("trajectories", _draw_trajectories, None, True),
    Overlay("goals", _draw_goals, "g", True),
    Overlay("agents", _draw_agents, None, True),
    Overlay("heading", _draw_heading, "h", True),
    Overlay("velocity", _draw_velocity, "v", False),
    Overlay("ids", _draw_ids, "i", False),
    Overlay("comm_lines", _draw_comm_lines, "c", False),
    Overlay("lidar", _draw_lidar, "l", True),
)

DEFAULT_ENABLED: frozenset[str] = frozenset(o.name for o in OVERLAYS if o.default_on)
