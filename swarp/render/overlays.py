"""Toggleable draw layers, as an ordered registry.

Each :class:`Overlay` is a named draw function ``(surface, geometry, camera, style)`` with a
default keyboard toggle and a default-on flag. The renderer walks :data:`OVERLAYS` in order
and calls each one whose name is in the enabled set, so registry order == draw order.

Overlays are read-only w.r.t. the simulator; they only consume a :class:`RenderGeometry`.
The ``lidar`` overlay is a no-op until ``geometry.extras["lidar"]`` is populated, which
:mod:`swarp.render.demo` does from :mod:`swarp.sensors.lidar`; a scenario without a lidar
simply draws no rays.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pygame

from swarp.core.config import ObstacleKind, ObstacleShape
from swarp.dynamics.base import P_LF, P_LR, P_MAX_STEER, P_THRUST_MAX
from swarp.render.camera import Camera
from swarp.render.geometry import RenderGeometry
from swarp.render.style import Style

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


# Derived, not mirrored: RenderGeometry carries the tags as plain int arrays, and
# ObstacleShape's docstring promises the kernel-side copies "cannot drift".
SHAPE_CIRCLE = int(ObstacleShape.CIRCLE)
SHAPE_BOX = int(ObstacleShape.BOX)
SHAPE_SEGMENT = int(ObstacleShape.SEGMENT)
KIND_MOVABLE = int(ObstacleKind.MOVABLE)


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

    Colour carries the *kind*: immovable scenery is black, a movable (pushable) body grey,
    so which obstacles the agents can shift is readable at a glance.
    """
    if g.obstacle_pos is None:
        return
    for i in range(g.obstacle_pos.shape[0]):
        center = g.obstacle_pos[i]
        shape = SHAPE_CIRCLE if g.obstacle_shape is None else int(g.obstacle_shape[i])
        angle = 0.0 if g.obstacle_angle is None else float(g.obstacle_angle[i])
        radius = float(g.obstacle_radius[i])
        movable = g.obstacle_kind is not None and int(g.obstacle_kind[i]) == KIND_MOVABLE
        fill = style.obstacle_color if movable else style.obstacle_immovable_color
        line = style.obstacle_outline if movable else style.obstacle_immovable_outline

        if shape == SHAPE_BOX and g.obstacle_half_extents is not None:
            hx, hy = (float(v) for v in g.obstacle_half_extents[i])
            _filled_polygon(
                surface, style, _pts(camera, _rotated_rect(center, angle, hx, hy)), fill, line
            )
        elif shape == SHAPE_SEGMENT and g.obstacle_half_extents is not None:
            half_len = float(g.obstacle_half_extents[i, 0])
            spine = np.array([(half_len, 0.0), (-half_len, 0.0)])
            c, s = math.cos(angle), math.sin(angle)
            ends = center + spine @ np.array([[c, -s], [s, c]]).T
            r_px = _r_px(camera, radius, floor=0)
            if r_px < 1:  # a zero-radius spine still has to be visible
                pts = _pts(camera, ends)
                pygame.draw.line(surface, fill, pts[0], pts[1], style.line_width)
                continue
            _filled_polygon(
                surface,
                style,
                _pts(camera, _rotated_rect(center, angle, half_len, radius)),
                fill,
                line,
            )
            for end in _pts(camera, ends):
                pygame.draw.circle(surface, fill, end, r_px)
        else:
            center_px = _p(camera, center)
            r_px = _r_px(camera, radius)
            pygame.draw.circle(surface, fill, center_px, r_px)
            pygame.draw.circle(surface, line, center_px, r_px, style.obstacle_outline_width)


def _filled_polygon(surface, style, pts, fill=None, outline=None) -> None:
    pygame.draw.polygon(surface, fill if fill is not None else style.obstacle_color, pts)
    pygame.draw.polygon(
        surface,
        outline if outline is not None else style.obstacle_outline,
        pts,
        style.obstacle_outline_width,
    )


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


def _rounded_rect_factors(hx: float, hy: float, corner: float, arc: int = 4) -> np.ndarray:
    """Body-frame polygon of a rounded box, in units of agent radius, CCW from the +x face.

    ``corner`` is clamped to fit the box, so an oversized value degrades to a stadium instead
    of self-intersecting. Four samples per quarter turn is plenty: the viewer has no
    anti-aliased primitives, so smooth corners come from the supersample downscale in
    :func:`renderer.draw_supersampled`, not from vertex count.
    """
    c = max(0.0, min(corner, hx, hy))
    ax, ay = hx - c, hy - c
    if c == 0.0:
        return np.array([(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)], dtype=np.float64)
    sweep = np.linspace(0.0, 0.5 * math.pi, arc + 2)
    quarters = ((ax, ay, 0.0), (-ax, ay, 0.5), (-ax, -ay, 1.0), (ax, -ay, 1.5))
    return np.concatenate(
        [
            np.stack(
                (
                    cx + c * np.cos(phase * math.pi + sweep),
                    cy + c * np.sin(phase * math.pi + sweep),
                ),
                axis=-1,
            )
            for cx, cy, phase in quarters
        ]
    )


# Body boxes as (half_length, half_width, corner_radius) in units of agent radius, keyed by
# DynamicsModel tag. Holonomic (0) has no entry: an omnidirectional agent *is* its circle.
_BODY_BOX: dict[int, tuple[float, float, float]] = {
    1: (1.05, 0.72, 0.26),  # diff-drive: rounded rectangle, wheels at the rear
    2: (1.70, 0.70, 0.34),  # kinematic bicycle: longer chassis, wheels front and rear
    3: (0.85, 0.85, 0.30),  # drone: rounded square frame carrying four rotors
}

# Body-frame silhouettes, in units of agent radius. A model with no entry falls back to a plain
# circle of the agent radius.
_BODY_SHAPES: dict[int, np.ndarray] = {
    tag: _rounded_rect_factors(*box) for tag, box in _BODY_BOX.items()
}

# Wheels straddle the chassis edge — half the tread's width falls outside the body, so they read
# as wheels rather than as stripes painted on it. All four numbers are in units of agent radius.
_DIFF_WHEEL = (-0.58, 0.86, 0.36, 0.17)  # (hub x, |y|, half length, half width) of a rear wheel
_BICYCLE_WHEEL_Y = 0.84
_BICYCLE_WHEEL_HALF_LEN = 0.42
_BICYCLE_WHEEL_HALF_W = 0.16
_BICYCLE_AXLE_SPAN = 2.5  # drawn front-to-rear axle distance, in units of agent radius
_DRONE_ARM = 1.5  # rotor distance from the hub, in units of agent radius


def _chevron_factors(hx: float) -> np.ndarray:
    """Forward-pointing triangle in the front third of a body of half-length ``hx``."""
    return np.array([(hx - 0.22, 0.0), (hx - 0.78, 0.40), (hx - 0.78, -0.40)], dtype=np.float64)


# The front cue: which end of the chassis is the nose, for the models that have one.
_CHEVRONS: dict[int, np.ndarray] = {tag: _chevron_factors(box[0]) for tag, box in _BODY_BOX.items()}


def _body_polygon(g, i: int) -> np.ndarray | None:
    """World-space silhouette of agent ``i``, or None when it is drawn as a plain circle."""
    shape = _BODY_SHAPES.get(int(g.model[i]))
    if shape is None:
        return None
    return _agent_points(g.pos[i], float(g.theta[i]), float(g.radius[i]), shape)


def _body_pts(camera: Camera, g, i: int, factors) -> list[list[int]]:
    """Screen points of body-frame ``factors`` on agent ``i`` — the decoration workhorse."""
    return _pts(camera, _agent_points(g.pos[i], float(g.theta[i]), float(g.radius[i]), factors))


def _steer_angle(g, i: int) -> float:
    """Commanded steering angle of bicycle agent ``i``, clamped to its ``max_steer``.

    Zero before the first step or when the action vector is too short: a straight wheel is the
    honest default when no command has been applied yet.
    """
    if g.action is None:
        return 0.0
    act = np.asarray(g.action, dtype=np.float64)
    if act.ndim != 2 or act.shape[1] < 2 or i >= act.shape[0]:
        return 0.0
    steer = float(act[i, 1])
    if g.agent_params is not None:
        max_steer = float(g.agent_params[i, P_MAX_STEER])
        if max_steer > 0.0:  # a bad policy must not draw a 90-degree wheel
            steer = max(-max_steer, min(max_steer, steer))
    return steer


def _bicycle_axles(g, i: int) -> tuple[float, float]:
    """Body-frame ``(front_x, rear_x)`` of the two axles, from the agent's own ``l_f``/``l_r``.

    The CoG sits at the body origin exactly as in the kinematic bicycle kernel, so the axles
    are placed at ``+l_f`` / ``-l_r`` rescaled to :data:`_BICYCLE_AXLE_SPAN` — the *ratio* is
    faithful, the absolute size stays tied to the collision radius. Both are clamped inside the
    chassis so a lopsided CoG cannot push an axle off the body.
    """
    hx = _BODY_BOX[2][0]
    l_f = l_r = 1.0
    if g.agent_params is not None:
        l_f, l_r = float(g.agent_params[i, P_LF]), float(g.agent_params[i, P_LR])
    total = l_f + l_r
    if total <= 0.0:  # AgentConfig forbids this, but the renderer must never divide by zero
        l_f = l_r = 1.0
        total = 2.0
    scale = _BICYCLE_AXLE_SPAN / total
    limit = hx - 0.20
    return min(limit, scale * l_f), -min(limit, scale * l_r)


def _draw_wheel(surface, camera, g, i: int, hub, angle, half_len, half_w, color) -> None:
    """A wheel as a *rectangle* centered on body-frame ``hub``, ``angle`` off the body heading.

    A polygon, not a thick line: ``pygame.draw.line`` thickens a line by extending it along one
    screen axis, so a rotated bar rasterizes as a parallelogram with axis-aligned end caps —
    visibly trapezoidal on a turned wheel. Lengths are in units of agent radius, so a wheel
    scales with zoom and supersampling like the rest of the sprite. Below a pixel of tread width
    it falls back to a hairline, where the shape can no longer resolve anyway.
    """
    theta, radius = float(g.theta[i]), float(g.radius[i])
    center = _agent_points(g.pos[i], theta, radius, [hub])[0]
    if _r_px(camera, half_w * radius, floor=0) < 1:
        span = half_len * radius * np.array([math.cos(theta + angle), math.sin(theta + angle)])
        pts = _pts(camera, np.stack((center + span, center - span)))
        pygame.draw.line(surface, color, pts[0], pts[1], 1)
        return
    corners = _rotated_rect(center, theta + angle, half_len * radius, half_w * radius)
    pygame.draw.polygon(surface, color, _pts(camera, corners))


def _draw_agent_decorations(surface, g, camera, style, i: int, r: int) -> None:
    """Per-model detail on top of an agent's filled body (wheels, front chevron, rotors)."""
    model = int(g.model[i])
    if model == 1:  # diff-drive: two rear wheels and a forward chevron
        hub_x, wheel_y, half_len, half_w = _DIFF_WHEEL
        for y in (-wheel_y, wheel_y):
            _draw_wheel(
                surface, camera, g, i, (hub_x, y), 0.0, half_len, half_w, style.agent_outline
            )
        pygame.draw.polygon(surface, style.agent_outline, _body_pts(camera, g, i, _CHEVRONS[1]))
    elif model == 2:  # bicycle: rear axle, steered front axle, forward chevron
        front_x, rear_x = _bicycle_axles(g, i)
        steer = _steer_angle(g, i)
        for x, angle in ((rear_x, 0.0), (front_x, steer)):
            for y in (-_BICYCLE_WHEEL_Y, _BICYCLE_WHEEL_Y):
                _draw_wheel(
                    surface,
                    camera,
                    g,
                    i,
                    (x, y),
                    angle,
                    _BICYCLE_WHEEL_HALF_LEN,
                    _BICYCLE_WHEEL_HALF_W,
                    style.agent_outline,
                )
        pygame.draw.polygon(surface, style.agent_outline, _body_pts(camera, g, i, _CHEVRONS[2]))
    elif model == 3:  # drone: quadrotor cross, front rotor filled
        # No chevron: the drone branch holds `theta` fixed (yaw lives in the attitude quaternion,
        # which RenderGeometry does not carry), so a rotating front cue would be misleading. The
        # filled +x rotor is a body-frame fact and stays honest.
        rotor = max(style.agent_min_px, r // 3)
        a = _DRONE_ARM
        tips = _body_pts(camera, g, i, [(a, 0.0), (0.0, a), (-a, 0.0), (0.0, -a)])
        center = _p(camera, g.pos[i])
        for tip in tips:
            pygame.draw.line(surface, style.agent_outline, center, tip, style.line_width)
        pygame.draw.circle(surface, style.agent_outline, tips[0], rotor)  # +x rotor: the front
        for tip in tips[1:]:
            pygame.draw.circle(surface, style.agent_outline, tip, rotor, style.agent_outline_width)


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

        if pts is None:
            pygame.draw.circle(surface, color, center, r)
        else:
            pygame.draw.polygon(surface, color, pts)

        _draw_agent_decorations(surface, g, camera, style, i, r)

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
        elif model == 1:  # diff-drive: per-wheel bars, at the offsets the sprite's wheels use
            v, omega = float(act[i, 0]), float(act[i, 1])
            wheel_y = _DIFF_WHEEL[1]
            for y, wheel_v in ((-wheel_y, v - omega * radius), (wheel_y, v + omega * radius)):
                reach = wheel_v * style.action_scale
                bar = _pts(camera, _agent_points(g.pos[i], theta, radius, [(0.0, y)]))
                tip = _p(
                    camera,
                    _agent_points(g.pos[i], theta, radius, [(0.0, y)])[0]
                    + reach * np.array([math.cos(theta), math.sin(theta)]),
                )
                color = style.action_color if wheel_v >= 0.0 else style.action_brake_color
                pygame.draw.line(surface, color, bar[0], tip, style.action_bar_px)
        elif model == 2:  # bicycle: the steered front wheels, highlighted, + an accel arrow
            accel, steer = float(act[i, 0]), _steer_angle(g, i)
            front_x, _ = _bicycle_axles(g, i)
            for y in (-_BICYCLE_WHEEL_Y, _BICYCLE_WHEEL_Y):
                # Repaint the sprite's own front wheels in the action color: same geometry, so
                # the command reads as a highlight rather than as a second pair of wheels.
                _draw_wheel(
                    surface,
                    camera,
                    g,
                    i,
                    (front_x, y),
                    steer,
                    _BICYCLE_WHEEL_HALF_LEN,
                    _BICYCLE_WHEEL_HALF_W,
                    style.action_color,
                )
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


def _draw_goal_pose(surface, g, camera, style):
    """Outline where a scenario's *movable body* should end up, from ``extras['goal_pose']``.

    Expected format: an array of oriented boxes, one row per box,
    ``(cx, cy, angle, half_x, half_y)`` in world space — the same primitive
    :func:`_draw_obstacles` uses for a ``BOX``, so a body built from boxes (Push-T's
    crossbar + stem) draws its target pose with the identical geometry it collides with.
    Drawn as a translucent outline so the body itself stays readable on top of it.
    """
    boxes = g.extras.get("goal_pose")
    if boxes is None:
        return
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 5)
    if boxes.size == 0:
        return
    color = (*getattr(style, "goal_pose_color", style.bounds_color),
             getattr(style, "goal_pose_alpha", 110))
    layer = _alpha_layer("goal_pose", surface.get_size())
    for cx, cy, angle, hx, hy in boxes:
        pts = _pts(camera, _rotated_rect((cx, cy), angle, hx, hy))
        pygame.draw.polygon(layer, color, pts, style.goal_ring_width)
    surface.blit(layer, (0, 0))


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
    Overlay("goal_pose", _draw_goal_pose, "p", True),
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
