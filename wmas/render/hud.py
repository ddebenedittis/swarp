"""Screen-space HUD and hover-inspect panel, drawn on top of the world scene.

Unlike overlays (world-space, per-agent), these render in pixel space and read the interactive
:class:`ViewState`, so they live here rather than in the overlay registry.
"""

from __future__ import annotations

import pygame

from wmas.render.geometry import RenderGeometry
from wmas.render.input import ViewState
from wmas.render.overlays import _get_font
from wmas.render.style import Style


def _blit_lines(surface, font, lines, color, origin) -> None:
    x, y = origin
    line_h = font.get_height()
    for i, line in enumerate(lines):
        surface.blit(font.render(line, True, color), (x, y + i * line_h))


def draw_hud(
    surface,
    state: ViewState,
    style: Style,
    *,
    step: int | None = None,
    fps: float | None = None,
) -> None:
    """Top-left status: focus env, step, fps, pause flag, and active overlays."""
    font = _get_font(style.font_size)
    header = f"env {state.focus_env}/{state.n_envs - 1}"
    if step is not None:
        header += f"   step {step}"
    if fps:
        header += f"   {fps:4.0f} fps"
    lines = [header]
    if state.paused:
        lines.append("PAUSED (space)")
    lines.append("on: " + ",".join(sorted(state.enabled)))
    _blit_lines(surface, font, lines, style.text_color, (8, 6))


def draw_hover_panel(
    surface, geometry: RenderGeometry, agent_idx: int | None, style: Style
) -> None:
    """Bottom-left inspector for the hovered agent (read-only state)."""
    if agent_idx is None or agent_idx >= geometry.n_agents:
        return
    font = _get_font(style.font_size)
    p = geometry.pos[agent_idx]
    v = geometry.vel[agent_idx]
    speed = float((v[0] ** 2 + v[1] ** 2) ** 0.5)
    lines = [
        f"agent {agent_idx}",
        f"pos ({p[0]:+.2f}, {p[1]:+.2f})",
        f"vel ({v[0]:+.2f}, {v[1]:+.2f})",
        f"spd {speed:.2f}",
    ]
    pad = 6
    line_h = font.get_height()
    width = max(font.size(line)[0] for line in lines) + 2 * pad
    height = len(lines) * line_h + 2 * pad
    x = 8
    y = surface.get_height() - height - 8

    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    panel.fill((255, 255, 255, 210))
    pygame.draw.rect(panel, style.text_color, panel.get_rect(), 1)
    surface.blit(panel, (x, y))
    _blit_lines(surface, font, lines, style.text_color, (x + pad, y + pad))
