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
from wmas.render.overlays import DEFAULT_ENABLED
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
    ) -> None:
        self.env = env
        self.scenario = scenario if scenario is not None else getattr(env, "scenario", None)
        self.size = (int(size[0]), int(size[1]))
        self.style = style or Style()
        self.fps = fps
        enabled = set(overlays) if overlays is not None else set(DEFAULT_ENABLED)
        self.state = ViewState(n_envs=env.n_envs, enabled=enabled, focus_env=env_index)
        self._camera: Camera | None = None
        self._step_count = 0

    # ------------------------------------------------------------------ draw

    def _geometry(self):
        return extract_geometry(self.env.world, self.state.focus_env, scenario=self.scenario)

    def _camera_for(self, geometry) -> Camera:
        if self._camera is None:
            bounds = geometry.bounds or _bounds_from_geometry(geometry)
            self._camera = Camera(bounds, viewport=(0, 0, self.size[0], self.size[1]))
        return self._camera

    def _draw(self, surface, geometry, camera, *, hud: bool, fps: float | None = None) -> None:
        draw_scene(surface, geometry, camera, self.state.enabled, self.style)
        if hud:
            draw_hud(surface, self.state, self.style, step=self._step_count, fps=fps)
            draw_hover_panel(surface, geometry, self.state.hover_agent, self.style)

    def render_array(self, *, hud: bool = False) -> np.ndarray:
        """Render the focus env to an ``(H, W, 3)`` uint8 array, headless."""
        pygame = _ensure_pygame()
        geometry = self._geometry()
        camera = self._camera_for(geometry)
        surface = pygame.Surface(self.size)
        self._draw(surface, geometry, camera, hud=hud)
        arr = pygame.surfarray.array3d(surface)
        return np.ascontiguousarray(np.transpose(arr, (1, 0, 2)))

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
        controller = InteractionController(self.state, camera, geometry_getter=self._geometry)
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
                self._draw(window, self._geometry(), camera, hud=True, fps=clock.get_fps())
                pygame.display.flip()
                clock.tick(self.fps)
                if done and close_when_done:
                    self.state.quit = True
        finally:
            pygame.display.quit()
