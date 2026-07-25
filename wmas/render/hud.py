"""Screen-space HUD and hover-inspect panel, drawn on top of the world scene.

Unlike overlays (world-space, per-agent), these render in pixel space and read the interactive
:class:`ViewState`, so they live here rather than in the overlay registry.
"""

from __future__ import annotations

import numpy as np
import pygame

from wmas.dynamics.base import DynamicsModel
from wmas.render.geometry import RenderGeometry
from wmas.render.input import ViewState
from wmas.render.overlays import OVERLAYS, _get_font
from wmas.render.style import Style


def _model_name(tag: int) -> str:
    """``DynamicsModel`` tag as a readable name, falling back to the raw int if unknown."""
    try:
        return DynamicsModel(int(tag)).name.lower()
    except ValueError:
        return str(int(tag))


def _blit_lines(surface, font, lines, color, origin) -> None:
    x, y = origin
    line_h = font.get_height()
    for i, line in enumerate(lines):
        surface.blit(font.render(line, True, color), (x, y + i * line_h))


def _blit_panel(pygame, surface, rect, bg, style: Style) -> None:
    """Translucent framed panel at ``rect`` — the shared chrome of every HUD box."""
    x, y, width, height = rect
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    panel.fill(bg)
    pygame.draw.rect(panel, style.panel_border, panel.get_rect(), style.tile_border_width)
    surface.blit(panel, (x, y))


def draw_hud(
    surface,
    state: ViewState,
    style: Style,
    *,
    step: int | None = None,
    fps: float | None = None,
) -> None:
    """Top-left status: focus env, step, fps, pause flag, and active overlays."""
    clip = surface.get_clip()
    font = _get_font(style.font_px(clip.height))
    header = f"env {state.focus_env}/{state.n_envs - 1}"
    if step is not None:
        header += f"   step {step}"
    if fps:
        header += f"   {fps:4.0f} fps"
    lines = [header]
    if state.paused:
        lines.append("PAUSED (space)")
    lines.append("F1/? help")
    lines.append(
        f"color {style.color_mode}   lidar {style.lidar_mode}   trail {style.trajectory_mode}"
    )
    lines.append("on: " + ",".join(sorted(state.enabled)))
    origin = (clip.left + style.hud_margin, clip.top + max(1, style.hud_margin - 2))
    _blit_lines(surface, font, lines, style.text_color, origin)


def draw_help(surface, state: ViewState, style: Style) -> None:
    """Top-right controls legend for the interactive viewer."""
    if not state.show_help:
        return
    clip = surface.get_clip()
    font = _get_font(style.font_px(clip.height))
    overlay_lines = [f"{o.key}: {o.name}" for o in OVERLAYS if o.key]
    lines = [
        "Controls",
        "space: pause/resume",
        "r: reset simulation",
        "left drag agent/obstacle: move it",
        "right click: move selected goal",
        ".: step once while paused",
        "t: cycle trajectory mode",
        "shift+l: cycle lidar mode",
        "k: cycle color mode",
        "mouse wheel: zoom",
        "middle drag: pan",
        "[ / ]: focus env",
        "F1 or ?: toggle help",
        "q/esc: quit",
        "",
        "Overlays",
        *overlay_lines,
    ]
    pad = style.panel_pad
    line_h = font.get_height()
    width = max(font.size(line)[0] for line in lines) + 2 * pad
    height = len(lines) * line_h + 2 * pad
    x = clip.right - width - style.hud_margin
    y = clip.top + style.hud_margin

    _blit_panel(pygame, surface, (x, y, width, height), style.help_panel_bg, style)
    _blit_lines(surface, font, lines, style.text_color, (x + pad, y + pad))


def draw_reward_hud(surface, rewards, geometry: RenderGeometry, style: Style) -> None:
    """Bottom-right per-agent reward sparkline over a ``(T, n_agents)`` window.

    One shared y-scale across agents, so the lines are directly comparable. No-op below two
    samples — a single point has no slope to show.
    """
    if rewards is None:
        return
    data = np.asarray(rewards, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] == 0:
        return

    pad = style.hover_pad
    width, height = style.reward_hud_size
    clip = surface.get_clip()
    font = _get_font(style.font_px(clip.height))
    x = clip.right - width - style.hud_margin
    y = clip.bottom - height - style.hud_margin
    _blit_panel(pygame, surface, (x, y, width, height), style.hover_panel_bg, style)

    label = f"reward  mean {float(data[-1].mean()):+.3f}"
    surface.blit(font.render(label, True, style.text_color), (x + pad, y + pad))

    plot_top = y + pad + font.get_height()
    plot_h = max(1, (y + height - pad) - plot_top)
    plot_w = max(1, width - 2 * pad)
    lo, hi = float(data.min()), float(data.max())
    span = hi - lo
    if span < 1e-12:  # flat window: center the line rather than divide by ~0
        lo, span = lo - 0.5, 1.0

    def to_px(col: np.ndarray) -> np.ndarray:
        xs = x + pad + np.linspace(0.0, plot_w, num=col.shape[0])
        ys = plot_top + plot_h * (1.0 - (col - lo) / span)
        return np.rint(np.stack((xs, ys), axis=-1)).astype(np.int64)

    if lo < 0.0 < hi:  # zero baseline, for sign-readability
        zero_y = int(round(plot_top + plot_h * (1.0 - (0.0 - lo) / span)))
        pygame.draw.line(
            surface,
            style.bounds_color,
            (x + pad, zero_y),
            (x + width - pad, zero_y),
            style.tile_border_width,
        )
    for i in range(data.shape[1]):
        color = style.agent_color(i, geometry.model[i] if i < geometry.n_agents else None)
        pygame.draw.lines(surface, color, False, to_px(data[:, i]).tolist(), style.edge_width)


def draw_hover_panel(
    surface, geometry: RenderGeometry, agent_idx: int | None, style: Style
) -> None:
    """Bottom-left inspector for the hovered agent (read-only state)."""
    if agent_idx is None or agent_idx >= geometry.n_agents:
        return
    font = _get_font(style.font_px(surface.get_height()))
    p = geometry.pos[agent_idx]
    v = geometry.vel[agent_idx]
    speed = float((v[0] ** 2 + v[1] ** 2) ** 0.5)
    lines = [
        f"agent {agent_idx}",
        f"pos ({p[0]:+.2f}, {p[1]:+.2f})",
        f"vel ({v[0]:+.2f}, {v[1]:+.2f})",
        f"spd {speed:.2f}",
        f"model {_model_name(geometry.model[agent_idx])}",
    ]
    pad = style.hover_pad
    line_h = font.get_height()
    width = max(font.size(line)[0] for line in lines) + 2 * pad
    height = len(lines) * line_h + 2 * pad
    x = style.hud_margin
    y = surface.get_height() - height - style.hud_margin

    _blit_panel(pygame, surface, (x, y, width, height), style.hover_panel_bg, style)
    _blit_lines(surface, font, lines, style.text_color, (x + pad, y + pad))
