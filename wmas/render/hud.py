"""Screen-space HUD and hover-inspect panel, drawn on top of the world scene.

Unlike overlays (world-space, per-agent), these render in pixel space and read the interactive
:class:`ViewState`, so they live here rather than in the overlay registry.
"""

from __future__ import annotations

import pygame

from wmas.render.geometry import RenderGeometry
from wmas.render.input import ViewState
from wmas.render.overlays import OVERLAYS, _get_font
from wmas.render.style import Style


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
        "t: cycle trajectory mode",
        "l: cycle lidar mode",
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
        f"model {int(geometry.model[agent_idx])}",
    ]
    pad = style.hover_pad
    line_h = font.get_height()
    width = max(font.size(line)[0] for line in lines) + 2 * pad
    height = len(lines) * line_h + 2 * pad
    x = style.hud_margin
    y = surface.get_height() - height - style.hud_margin

    _blit_panel(pygame, surface, (x, y, width, height), style.hover_panel_bg, style)
    _blit_lines(surface, font, lines, style.text_color, (x + pad, y + pad))
