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

from wmas.dynamics.base import P_MAX_STEER, P_THRUST_MAX
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


def _pts(camera: Camera, world_pts) -> list[list[int]]:
    """Batch world->screen: one vectorized transform for a whole point array.

    ``np.rint`` is round-half-to-even, matching ``int(round(float(x)))`` on floats, so this is
    pixel-identical to a loop over :func:`_p` — just without the per-point Python call, which
    dominates overlays like lidar (n_agents x n_rays x 2 points per frame).
    """
    s = camera.world_to_screen(np.asarray(world_pts, dtype=np.float64))
    return np.rint(s).astype(np.int64).tolist()


def _r_px(camera: Camera, r: float, floor: int = 1) -> int:
    return max(floor, int(round(float(r) * camera.scale)))


_ALPHA_CACHE: dict[tuple[str, int, int], pygame.Surface] = {}


def _alpha_layer(tag: str, size: tuple[int, int]) -> pygame.Surface:
    """A transparent scratch layer of ``size``, reused across frames and cleared on each call.

    Clearing is a memset; the alternative (a fresh ``SRCALPHA`` surface per frame) allocates,
    zeroes and frees several MB every frame. Keyed by ``tag`` so two overlays never share a
    live layer. A layer costs ``4 * w * h`` bytes, which supersampling multiplies by S^2.
    """
    key = (tag, int(size[0]), int(size[1]))
    layer = _ALPHA_CACHE.get(key)
    if layer is None:
        if len(_ALPHA_CACHE) > 8:  # window resizes / supersample changes
            _ALPHA_CACHE.clear()
        layer = pygame.Surface((key[1], key[2]), pygame.SRCALPHA)
        _ALPHA_CACHE[key] = layer
    else:
        layer.fill((0, 0, 0, 0))
    return layer


def _draw_bounds(surface, g, camera, style):
    if g.bounds is None:
        return
    x0, x1, y0, y1 = g.bounds
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    pygame.draw.lines(
        surface, style.bounds_color, True, [_p(camera, c) for c in corners], style.line_width
    )


SHAPE_CIRCLE, SHAPE_BOX, SHAPE_SEGMENT = 0, 1, 2  # mirrors core.config.ObstacleShape


def _rotated_rect(center, angle: float, half_x: float, half_y: float) -> np.ndarray:
    """World-space corners of a box centered at ``center``, rotated by ``angle``."""
    corners = np.array([(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]) * (half_x, half_y)
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return np.asarray(center, dtype=np.float64) + corners @ rot.T


def _draw_obstacles(surface, g, camera, style):
    """Obstacles as the simulator actually collides with them: circle, box or capsule.

    Mirrors the shape dispatch in ``core.collisions._static_forces`` — a box's boundary *is*
    its collision surface (the radius is unused), and a segment is a capsule of
    ``obstacle_radius`` around a spine of half-length ``half_extents[:, 0]``.
    """
    if g.obstacle_pos is None:
        return
    for i in range(g.obstacle_pos.shape[0]):
        center = g.obstacle_pos[i]
        shape = SHAPE_CIRCLE if g.obstacle_shape is None else int(g.obstacle_shape[i])
        angle = 0.0 if g.obstacle_angle is None else float(g.obstacle_angle[i])
        radius = float(g.obstacle_radius[i])

        if shape == SHAPE_BOX and g.obstacle_half_extents is not None:
            hx, hy = (float(v) for v in g.obstacle_half_extents[i])
            _filled_polygon(surface, style, _pts(camera, _rotated_rect(center, angle, hx, hy)))
        elif shape == SHAPE_SEGMENT and g.obstacle_half_extents is not None:
            half_len = float(g.obstacle_half_extents[i, 0])
            spine = np.array([(half_len, 0.0), (-half_len, 0.0)])
            c, s = math.cos(angle), math.sin(angle)
            ends = center + spine @ np.array([[c, -s], [s, c]]).T
            r_px = _r_px(camera, radius, floor=0)
            if r_px < 1:  # a zero-radius spine still has to be visible
                pts = _pts(camera, ends)
                pygame.draw.line(surface, style.obstacle_color, pts[0], pts[1], style.line_width)
                continue
            _filled_polygon(
                surface, style, _pts(camera, _rotated_rect(center, angle, half_len, radius))
            )
            for end in _pts(camera, ends):
                pygame.draw.circle(surface, style.obstacle_color, end, r_px)
        else:
            center_px = _p(camera, center)
            r_px = _r_px(camera, radius)
            pygame.draw.circle(surface, style.obstacle_color, center_px, r_px)
            pygame.draw.circle(
                surface, style.obstacle_outline, center_px, r_px, style.obstacle_outline_width
            )


def _filled_polygon(surface, style, pts) -> None:
    pygame.draw.polygon(surface, style.obstacle_color, pts)
    pygame.draw.polygon(surface, style.obstacle_outline, pts, style.obstacle_outline_width)


def _draw_neighbor_graph(surface, g, camera, style):
    if g.edges.size == 0:
        return
    pts = _pts(camera, g.pos)
    for i, j in g.edges:
        pygame.draw.line(surface, style.edge_color, pts[i], pts[j], style.edge_width)


def _dashed_segments(p0, p1, dash_px: int, gap_px: int) -> np.ndarray:
    """Screen-space on-segments of a dashed line from ``p0`` to ``p1`` as (N, 2, 2) int pixels."""
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    span = b - a
    length = float(np.hypot(span[0], span[1]))
    if length < 1.0:
        return np.empty((0, 2, 2), dtype=np.int64)
    unit = span / length
    period = max(1.0, float(dash_px) + float(gap_px))
    starts = np.arange(0.0, length, period)
    ends = np.minimum(starts + float(dash_px), length)
    on = np.stack((a + starts[:, None] * unit, a + ends[:, None] * unit), axis=1)
    return np.rint(on).astype(np.int64)


def _draw_goals(surface, g, camera, style):
    """Goal rings, plus an agent->goal connector and a filled ring once the goal is reached."""
    if g.goals is None:
        return
    centers = _pts(camera, g.goals)
    agents = _pts(camera, g.pos)
    reached = np.linalg.norm(g.pos - g.goals, axis=1) <= g.radius * style.goal_reached_factor

    if style.goal_connector != "none":
        # Drawn on an alpha layer first, so connectors sit under the rings and the bodies.
        layer = _alpha_layer("goals", surface.get_size())
        for i in range(g.n_agents):
            if reached[i]:
                continue  # a reached goal needs no leader line
            color = (*style.agent_color(i, g.model[i]), style.goal_connector_alpha)
            if style.goal_connector == "dashed":
                for start, end in _dashed_segments(
                    agents[i], centers[i], style.goal_dash_px, style.goal_gap_px
                ):
                    pygame.draw.line(layer, color, start, end, style.edge_width)
            else:
                pygame.draw.line(layer, color, agents[i], centers[i], style.edge_width)
        surface.blit(layer, (0, 0))

    for i in range(g.n_agents):
        color = style.agent_color(i, g.model[i])
        r = _r_px(camera, g.radius[i], floor=style.goal_min_px)
        if reached[i]:
            # Fill, then keep the agent-colored ring on top: both "reached" and "whose" readable.
            pygame.draw.circle(surface, style.goal_reached_color, centers[i], r)
        pygame.draw.circle(surface, color, centers[i], r, style.goal_ring_width)
        pygame.draw.circle(surface, color, centers[i], style.goal_dot_px)  # center dot


def _agent_points(pos, theta: float, radius: float, factors) -> np.ndarray:
    """Body-frame ``(fx, fy)`` factors (in units of radius) as world-space points."""
    f = np.asarray(factors, dtype=np.float64).reshape(-1, 2)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(pos, dtype=np.float64) + radius * (f @ rot.T)


# Body-frame silhouettes, in units of agent radius. Keyed by DynamicsModel tag; a model with no
# entry falls back to a plain circle of the agent radius.
_BODY_SHAPES: dict[int, np.ndarray] = {
    1: np.array(  # diff-drive: compact body with two side tracks
        [
            (0.95, 0.55),
            (0.55, 0.9),
            (-0.65, 0.9),
            (-0.95, 0.55),
            (-0.95, -0.55),
            (-0.65, -0.9),
            (0.55, -0.9),
            (0.95, -0.55),
        ]
    ),
    2: np.array(  # kinematic bicycle: stylized car
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
        ]
    ),
}


def _body_polygon(g, i: int) -> np.ndarray | None:
    """World-space silhouette of agent ``i``, or None when it is drawn as a plain circle."""
    shape = _BODY_SHAPES.get(int(g.model[i]))
    if shape is None:
        return None
    return _agent_points(g.pos[i], float(g.theta[i]), float(g.radius[i]), shape)


def _draw_agent_decorations(surface, g, camera, style, i: int, r: int, color) -> None:
    """Per-model detail drawn on top of an agent's filled body (tracks, cabin, wheels, arms)."""
    pos, theta, radius = g.pos[i], float(g.theta[i]), float(g.radius[i])
    model = int(g.model[i])
    wheel_w = max(2, r // 2)
    if model == 1:  # diff-drive: two side tracks and a nose dot
        for y in (-1.05, 1.05):
            track = _pts(camera, _agent_points(pos, theta, radius, [(0.65, y), (-0.65, y)]))
            pygame.draw.line(surface, style.agent_outline, track[0], track[1], wheel_w)
        nose = _pts(camera, _agent_points(pos, theta, radius, [(0.55, 0.0)]))
        pygame.draw.circle(surface, style.agent_outline, nose[0], max(1, r // 5))
    elif model == 2:  # kinematic bicycle: cabin, four wheels, headlight
        cabin = _pts(
            camera,
            _agent_points(
                pos, theta, radius, [(0.55, 0.38), (-0.35, 0.42), (-0.6, -0.42), (0.55, -0.38)]
            ),
        )
        pygame.draw.polygon(surface, style.background, cabin)
        pygame.draw.polygon(surface, style.agent_outline, cabin, style.agent_outline_width)
        for x in (-0.95, 0.95):
            for y in (-0.82, 0.82):
                wheel = _pts(
                    camera, _agent_points(pos, theta, radius, [(x + 0.25, y), (x - 0.25, y)])
                )
                pygame.draw.line(surface, style.agent_outline, wheel[0], wheel[1], wheel_w)
        front = _pts(camera, _agent_points(pos, theta, radius, [(1.2, 0.0)]))
        pygame.draw.circle(surface, style.background, front[0], max(1, r // 5))
    elif model == 3:  # drone placeholder: quadrotor cross
        for a in (0.0, math.pi / 2):
            arm = _pts(camera, _agent_points(pos, theta + a, radius, [(1.5, 0.0), (-1.5, 0.0)]))
            pygame.draw.line(surface, color, arm[0], arm[1], style.line_width)
            for tip in arm:
                pygame.draw.circle(surface, color, tip, max(2, r // 3), style.agent_outline_width)


def _contact_mask(g, style) -> np.ndarray:
    """Bool ``(n_agents,)``: agents whose bodies overlap a neighbor's.

    Derived purely from ``edges`` + ``pos`` + ``radius``, so the renderer stays read-only
    w.r.t. the simulator. Neighbor lists exclude self, so an agent cannot flag itself; empty
    ``edges`` (collisions disabled) yields all-False.
    """
    mask = np.zeros(g.n_agents, dtype=bool)
    if g.edges.size == 0:
        return mask
    i, j = g.edges[:, 0], g.edges[:, 1]
    gap = np.linalg.norm(g.pos[i] - g.pos[j], axis=1)
    touching = gap < (g.radius[i] + g.radius[j]) * (1.0 + style.contact_tol)
    mask[i[touching]] = True
    mask[j[touching]] = True
    return mask


def _draw_agent_shadows(surface, g, camera, style) -> None:
    """Offset silhouettes under every body, drawn in one pass so no shadow lands on a body."""
    off = style.agent_shadow_offset_px
    layer = _alpha_layer("agents", surface.get_size())
    centers = _pts(camera, g.pos)
    for i in range(g.n_agents):
        body = _body_polygon(g, i)
        if body is None:
            r = _r_px(camera, g.radius[i], floor=style.agent_min_px)
            pygame.draw.circle(layer, style.agent_shadow, _offset(centers[i], off), r)
        else:
            pygame.draw.polygon(
                layer, style.agent_shadow, [_offset(p, off) for p in _pts(camera, body)]
            )
    surface.blit(layer, (0, 0))


def _offset(pt, d: int) -> tuple[int, int]:
    return (pt[0] + d, pt[1] + d)


def _draw_agents(surface, g, camera, style):
    if style.depth_cue == "shadow":
        _draw_agent_shadows(surface, g, camera, style)
    contact = (
        _contact_mask(g, style) if style.contact_highlight else np.zeros(g.n_agents, dtype=bool)
    )
    halo = style.depth_cue == "halo"
    centers = _pts(camera, g.pos)
    for i in range(g.n_agents):
        center = centers[i]
        r = _r_px(camera, g.radius[i], floor=style.agent_min_px)
        color = style.agent_color(i, g.model[i])
        body = _body_polygon(g, i)
        pts = None if body is None else _pts(camera, body)

        if halo:
            # Stroke the silhouette before filling: half the width straddles outside the body
            # and the fill covers the inner half, so the halo hugs even a 1.75r car sprite.
            # Painter order does the rest — a later agent gets a clean gap over an earlier one.
            w = 2 * style.agent_halo_px
            if pts is None:
                pygame.draw.circle(surface, style.agent_halo, center, r + style.agent_halo_px, w)
            else:
                pygame.draw.polygon(surface, style.agent_halo, pts, w)

        if int(g.model[i]) == 3:  # drone: a hub, the arms are decorations
            pygame.draw.circle(surface, color, center, max(2, r // 2))
        elif pts is None:
            pygame.draw.circle(surface, color, center, r)
        else:
            pygame.draw.polygon(surface, color, pts)

        _draw_agent_decorations(surface, g, camera, style, i, r, color)

        outline = style.contact_color if contact[i] else style.agent_outline
        width = style.contact_outline_width if contact[i] else style.agent_outline_width
        if pts is not None:
            pygame.draw.polygon(surface, outline, pts, width)
        pygame.draw.circle(surface, outline, center, r, width)


def _draw_heading(surface, g, camera, style):
    reach = (g.radius * style.heading_len_factor)[:, None]
    dirs = np.stack((np.cos(g.theta), np.sin(g.theta)), axis=-1)
    starts = _pts(camera, g.pos)
    tips = _pts(camera, g.pos + reach * dirs)
    for i in range(g.n_agents):
        if int(g.model[i]) == 0:  # holonomic agents have no meaningful body heading.
            continue
        pygame.draw.line(surface, style.heading_color, starts[i], tips[i], style.line_width)


def _arrow(surface, style, p0, p1, color, width: int, *, filled: bool = True) -> None:
    """A line from ``p0`` to ``p1`` with an arrowhead at ``p1``, in screen pixels."""
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    span = b - a
    length = float(np.hypot(span[0], span[1]))
    pygame.draw.line(surface, color, a.astype(int).tolist(), b.astype(int).tolist(), width)
    head = float(style.action_arrow_px)
    if length < head:
        return
    unit = span / length
    normal = np.array([-unit[1], unit[0]])
    tri = np.rint(
        np.stack((b, b - head * unit + 0.5 * head * normal, b - head * unit - 0.5 * head * normal))
    ).astype(np.int64)
    pygame.draw.polygon(surface, color, tri.tolist(), 0 if filled else max(1, width))


def _draw_action(surface, g, camera, style):
    """Draw the last applied action per agent, interpreted by its dynamics model.

    Reads ``geometry.action`` (``World.action``, set by ``Environment.step``), so this shows
    the action that produced the *drawn* state — i.e. one step older than the state itself.
    A no-op before the first step, or for an agent whose action vector is too short.
    """
    if g.action is None or g.ctrl_mode is None:
        return
    act = np.asarray(g.action, dtype=np.float64)
    if act.ndim != 2 or act.shape[1] < 2:
        return
    params = g.agent_params
    centers = _pts(camera, g.pos)

    for i in range(min(g.n_agents, act.shape[0])):
        model = int(g.model[i])
        radius = float(g.radius[i])
        theta = float(g.theta[i])
        r_px = _r_px(camera, radius, floor=style.agent_min_px)

        if model == 0:  # holonomic: (vx, vy) or (ax, ay) — an arrow from the body center
            gain = style.action_scale if int(g.ctrl_mode[i]) == 0 else style.action_accel_scale
            tip = _p(camera, g.pos[i] + act[i, :2] * gain)
            # An open head distinguishes an acceleration command from a velocity one.
            _arrow(
                surface,
                style,
                centers[i],
                tip,
                style.action_color,
                style.line_width,
                filled=int(g.ctrl_mode[i]) == 0,
            )
        elif model == 1:  # diff-drive: per-track bars, at the offsets the sprite's tracks use
            v, omega = float(act[i, 0]), float(act[i, 1])
            for y, wheel_v in ((-1.05, v - omega * radius), (1.05, v + omega * radius)):
                reach = wheel_v * style.action_scale
                bar = _pts(camera, _agent_points(g.pos[i], theta, radius, [(0.0, y)]))
                tip = _p(
                    camera,
                    _agent_points(g.pos[i], theta, radius, [(0.0, y)])[0]
                    + reach * np.array([math.cos(theta), math.sin(theta)]),
                )
                color = style.action_color if wheel_v >= 0.0 else style.action_brake_color
                pygame.draw.line(surface, color, bar[0], tip, style.action_bar_px)
        elif model == 2:  # bicycle: steered front wheels + a longitudinal accel arrow
            accel, steer = float(act[i, 0]), float(act[i, 1])
            if params is not None:
                max_steer = float(params[i, P_MAX_STEER])
                if max_steer > 0.0:  # a bad policy must not draw a 90-degree wheel
                    steer = max(-max_steer, min(max_steer, steer))
            for y in (-0.82, 0.82):
                hub = _agent_points(g.pos[i], theta, radius, [(0.95, y)])[0]
                span = 0.25 * radius * np.array([math.cos(theta + steer), math.sin(theta + steer)])
                wheel = _pts(camera, np.stack((hub + span, hub - span)))
                pygame.draw.line(surface, style.action_color, wheel[0], wheel[1], max(2, r_px // 2))
            reach = accel * style.action_accel_scale
            tip = _p(camera, g.pos[i] + reach * np.array([math.cos(theta), math.sin(theta)]))
            color = style.action_color if accel >= 0.0 else style.action_brake_color
            _arrow(surface, style, centers[i], tip, color, style.line_width)
        elif model == 3 and act.shape[1] >= 4:  # drone: per-rotor thrust dots at the arm tips
            thrust_max = 1.0
            if params is not None and params[i, P_THRUST_MAX] > 0.0:
                thrust_max = float(params[i, P_THRUST_MAX])
            tips = []
            for a in (0.0, math.pi / 2):
                tips.extend(_agent_points(g.pos[i], theta + a, radius, [(1.5, 0.0), (-1.5, 0.0)]))
            for k, tip in enumerate(_pts(camera, np.asarray(tips))):
                frac = min(1.0, abs(float(act[i, k])) / thrust_max)
                pygame.draw.circle(
                    surface, style.action_color, tip, max(1, int(round(frac * r_px)))
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
    overlay = _alpha_layer("trails", surface.get_size())
    for i in range(min(g.n_agents, trails.shape[0])):
        pts = _pts(camera, trails[i])
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
    starts = _pts(camera, g.pos)
    tips = _pts(camera, g.pos + g.vel * style.velocity_scale)
    for i in range(g.n_agents):
        pygame.draw.line(surface, style.velocity_color, starts[i], tips[i], style.line_width)


def _draw_ids(surface, g, camera, style):
    size = style.font_px(camera.vh)
    font = _get_font(size)
    centers = _pts(camera, g.pos)
    for i in range(g.n_agents):
        label = font.render(str(i), True, style.text_color)
        cx, cy = centers[i]
        surface.blit(label, (cx + style.id_offset_px, cy - size))


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
    pts = _pts(camera, segs.reshape(-1, 2))
    for k in range(0, len(pts), 2):
        pygame.draw.line(surface, color, pts[k], pts[k + 1], style.edge_width)


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
            overlay = _alpha_layer("lidar", surface.get_size())
            for i in range(min(g.n_agents, grouped.shape[0])):
                endpoints = grouped[i, :, 1, :]
                if endpoints.shape[0] < 2:
                    continue
                color = style.agent_color(i, g.model[i])
                poly = _pts(camera, np.vstack((g.pos[i][None, :], endpoints)))
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
            _draw_rays(surface, camera, style, grouped[i].reshape(-1, 2, 2), color)
        return
    _draw_rays(surface, camera, style, rays.reshape(-1, 2, 2), style.lidar_color)


def _draw_rays(surface, camera, style, segs, color) -> None:
    """One batch transform for a whole ray bundle, then a line + hit dot per ray."""
    pts = _pts(camera, segs.reshape(-1, 2))
    for k in range(0, len(pts), 2):
        pygame.draw.line(surface, color, pts[k], pts[k + 1], style.lidar_ray_width)
        pygame.draw.circle(surface, color, pts[k + 1], style.lidar_hit_px)


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
    Overlay("action", _draw_action, "a", False),
    Overlay("velocity", _draw_velocity, "v", False),
    Overlay("ids", _draw_ids, "i", False),
    Overlay("comm_lines", _draw_comm_lines, "c", False),
    Overlay("lidar", _draw_lidar, "l", True),
)

DEFAULT_ENABLED: frozenset[str] = frozenset(o.name for o in OVERLAYS if o.default_on)
