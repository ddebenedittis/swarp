"""The Viewer: one class serving offscreen frames and a live interactive window.

- ``render_array(hud=...)`` composes scene + optional HUD/hover offscreen (headless-safe) —
  the primitive behind notebook display and screenshots.
- ``run(...)`` opens a pygame window and drives an event loop over the same drawing code and
  the :class:`InteractionController`. It needs a display, so it is exercised manually rather
  than in the headless test suite.

Rendering always runs under ``torch.no_grad()`` and never on the differentiable hot path.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from wmas.render.camera import Camera
from wmas.render.geometry import extract_geometry
from wmas.render.hud import draw_hover_panel, draw_hud
from wmas.render.input import InteractionController, ViewState
from wmas.render.layout import MosaicLayout, compute_mosaic_layout, tile_at
from wmas.render.overlays import DEFAULT_ENABLED, _p, _r_px
from wmas.render.renderer import _bounds_from_geometry, _ensure_pygame, draw_scene
from wmas.render.style import Style


class Viewer:
    def __init__(
        self,
        env,
        *,
        size: tuple[int, int] = (700, 700),
        overlays: set[str] | None = None,
        env_index: int = 0,
        style: Style | None = None,
        scenario=None,
        fps: int = 30,
        mosaic: bool = False,
        max_tiles: int = 16,
        focus_frac: float = 0.68,
        allow_write_back: bool = True,
    ) -> None:
        self.env = env
        self.scenario = scenario if scenario is not None else getattr(env, "scenario", None)
        self.size = (int(size[0]), int(size[1]))
        self.style = style or Style()
        self.fps = fps
        self.mosaic = mosaic
        self.max_tiles = max_tiles
        self.focus_frac = focus_frac
        self.allow_write_back = allow_write_back
        enabled = set(overlays) if overlays is not None else set(DEFAULT_ENABLED)
        self.state = ViewState(n_envs=env.n_envs, enabled=enabled, focus_env=env_index)
        self._camera: Camera | None = None
        self._layout: MosaicLayout | None = None
        self._window = None
        self._step_count = 0

    # ------------------------------------------------------------------ draw

    def _geometry(self, env_idx: int | None = None):
        idx = self.state.focus_env if env_idx is None else env_idx
        return extract_geometry(self.env.world, idx, scenario=self.scenario)

    def _ensure_layout(self) -> MosaicLayout:
        if self._layout is None:
            self._layout = compute_mosaic_layout(
                self.size, self.env.n_envs, focus_frac=self.focus_frac, max_tiles=self.max_tiles
            )
        return self._layout

    def _focus_viewport(self):
        return (
            self._ensure_layout().focus_rect if self.mosaic else (0, 0, self.size[0], self.size[1])
        )

    def _camera_for(self, geometry) -> Camera:
        if self._camera is None:
            bounds = geometry.bounds or _bounds_from_geometry(geometry)
            self._camera = Camera(bounds, viewport=self._focus_viewport())
        return self._camera

    def _tile_env_at(self, screen_xy) -> int | None:
        return tile_at(self._ensure_layout(), screen_xy) if self.mosaic else None

    # ------------------------------------------------------------ write-back

    def _write_agent_pos(self, agent_idx: int, world_xy) -> None:
        """Move an agent in the focus env and zero its velocity (so it stays put).

        Writes the *current* world.state, which the next step re-wraps for Warp; must not
        cache a state reference across steps (World.step reassigns it).
        """
        with torch.no_grad():
            e = self.state.focus_env
            s = self.env.world.state
            s.pos[e, agent_idx] = torch.tensor(
                world_xy, dtype=self.env.dtype, device=self.env.device
            )
            s.vel[e, agent_idx] = 0
            s.speed[e, agent_idx] = 0
            s.ang_vel[e, agent_idx] = 0
            # Positions changed out of band; drop any reusable neighbor list.
            self.env.world.mark_pos_dirty()

    def _write_goal(self, agent_idx: int, world_xy) -> None:
        """Move an agent's goal in the focus env (no-op if the scenario has no goals)."""
        world = self.env.world
        if world.goals is None:
            return
        with torch.no_grad():
            world.goals[self.state.focus_env, agent_idx] = torch.tensor(
                world_xy, dtype=self.env.dtype, device=self.env.device
            )

    def _draw(self, surface, geometry, camera, *, hud: bool, fps=None, clear: bool = True) -> None:
        draw_scene(surface, geometry, camera, self.state.enabled, self.style, clear=clear)
        if hud:
            draw_hud(surface, self.state, self.style, step=self._step_count, fps=fps)
            draw_hover_panel(surface, geometry, self.state.hover_agent, self.style)

    def _draw_tile(self, pygame, surface, geometry, camera, rect, *, focused: bool) -> None:
        r = pygame.Rect(*rect)
        surface.fill((236, 236, 240), r)
        prev = surface.get_clip()
        surface.set_clip(r)
        if geometry.bounds is not None:
            x0, x1, y0, y1 = geometry.bounds
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            pygame.draw.lines(
                surface, self.style.bounds_color, True, [_p(camera, c) for c in corners], 1
            )
        for i in range(geometry.n_agents):
            pygame.draw.circle(
                surface,
                self.style.agent_color(i),
                _p(camera, geometry.pos[i]),
                _r_px(camera, geometry.radius[i], floor=2),
            )
        surface.set_clip(prev)
        border = self.style.velocity_color if focused else (185, 185, 190)
        pygame.draw.rect(surface, border, r, 2 if focused else 1)

    def _draw_mosaic(self, pygame, surface, *, hud: bool, fps=None) -> None:
        layout = self._ensure_layout()
        surface.fill(self.style.background)
        for tile_rect, env_idx in zip(layout.tiles, layout.tile_envs, strict=True):
            g = self._geometry(env_idx)
            cam = Camera(g.bounds or _bounds_from_geometry(g), viewport=tile_rect)
            self._draw_tile(
                pygame, surface, g, cam, tile_rect, focused=env_idx == self.state.focus_env
            )
        focus = pygame.Rect(*layout.focus_rect)
        surface.fill(self.style.background, focus)
        prev = surface.get_clip()
        surface.set_clip(focus)
        gf = self._geometry()
        self._draw(surface, gf, self._camera_for(gf), hud=hud, fps=fps, clear=False)
        surface.set_clip(prev)
        pygame.draw.rect(surface, self.style.bounds_color, focus, 1)

    def _render_onto(self, pygame, surface, *, hud: bool, fps=None) -> None:
        if self.mosaic:
            self._draw_mosaic(pygame, surface, hud=hud, fps=fps)
        else:
            g = self._geometry()
            self._draw(surface, g, self._camera_for(g), hud=hud, fps=fps)

    def render_array(self, *, hud: bool = False) -> np.ndarray:
        """Render the current view (single env or mosaic) to an ``(H, W, 3)`` uint8 array."""
        pygame = _ensure_pygame()
        surface = pygame.Surface(self.size)
        self._render_onto(pygame, surface, hud=hud)
        arr = pygame.surfarray.array3d(surface)
        return np.ascontiguousarray(np.transpose(arr, (1, 0, 2)))

    def render_human_frame(self) -> None:
        """Update a persistent window with one frame (VMAS ``render(mode='human')`` style)."""
        pygame = _ensure_pygame()
        if self._window is None:
            pygame.display.init()
            self._window = pygame.display.set_mode(self.size)
            pygame.display.set_caption("wmas viewer")
        for event in pygame.event.get():  # keep the window responsive / closeable
            if event.type == pygame.QUIT:
                self.state.quit = True
        self._render_onto(pygame, self._window, hud=True)
        pygame.display.flip()

    def close(self) -> None:
        """Close the window if one is open."""
        if self._window is not None:
            _ensure_pygame().display.quit()
            self._window = None

    # ------------------------------------------------------------------- run

    def _actions(self, action_fn, obs):
        if action_fn is None:
            return torch.zeros(
                self.env.n_envs,
                self.env.n_agents,
                self.env.world.act_dim,
                dtype=self.env.dtype,
                device=self.env.device,
            )
        return action_fn(obs)

    def run(
        self,
        action_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        max_steps: int | None = None,
        close_when_done: bool = False,
    ) -> None:
        """Open a window and run the interactive loop until quit (Esc/Q or window close).

        With ``max_steps`` set, stepping stops at that many steps; ``close_when_done`` then
        also closes the window (useful for scripted/headless-dummy runs).
        """
        pygame = _ensure_pygame()
        pygame.display.init()
        window = pygame.display.set_mode(self.size)
        pygame.display.set_caption("wmas viewer")
        clock = pygame.time.Clock()
        camera = self._camera_for(self._geometry())
        controller = InteractionController(
            self.state,
            camera,
            geometry_getter=self._geometry,
            tile_resolver=self._tile_env_at,
            on_drag_agent=self._write_agent_pos if self.allow_write_back else None,
            on_place_goal=self._write_goal if self.allow_write_back else None,
        )
        obs = self.scenario.observations() if self.scenario is not None else None
        try:
            while not self.state.quit:
                for event in pygame.event.get():
                    controller.handle_event(event)
                done = max_steps is not None and self._step_count >= max_steps
                if not self.state.paused and not done:
                    with torch.no_grad():
                        obs, *_ = self.env.step(self._actions(action_fn, obs))
                    self._step_count += 1
                self._render_onto(pygame, window, hud=True, fps=clock.get_fps())
                pygame.display.flip()
                clock.tick(self.fps)
                if done and close_when_done:
                    self.state.quit = True
        finally:
            pygame.display.quit()
