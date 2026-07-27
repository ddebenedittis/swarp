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
from wmas.render.geometry import extract_geometry, extract_geometry_batch
from wmas.render.hud import (
    draw_help,
    draw_hover_panel,
    draw_hud,
    draw_reward_hud,
    draw_speed_badge,
)
from wmas.render.input import InteractionController, ViewState
from wmas.render.layout import MosaicLayout, compute_mosaic_layout, tile_at
from wmas.render.overlays import DEFAULT_ENABLED, _p, _r_px
from wmas.render.renderer import (
    _bounds_from_geometry,
    _ensure_pygame,
    draw_scene,
    draw_supersampled,
)
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
        fps: int = 60,
        steps_per_frame: float = 1.0,
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
        # Base pacing, in sim steps per rendered frame; the up/down ladder multiplies it. A
        # caller that knows its dt sets this to (real-time steps per second) / fps, which makes
        # the interactive multiplier read as a multiple of *real* time.
        self.steps_per_frame = float(steps_per_frame)
        self.mosaic = mosaic
        self.max_tiles = max_tiles
        self.focus_frac = focus_frac
        self.allow_write_back = allow_write_back
        enabled = set(overlays) if overlays is not None else set(DEFAULT_ENABLED)
        self.state = ViewState(n_envs=env.n_envs, enabled=enabled, focus_env=env_index)
        self._camera: Camera | None = None
        self._layout: MosaicLayout | None = None
        self._window = None
        self._controller: InteractionController | None = None
        self._step_count = 0
        self._step_accum = 0.0  # fractional-speed carry, see _steps_this_frame
        self._trail_history: list[np.ndarray] = []
        self._reward_history: list[np.ndarray] = []
        self._buffer_env = env_index  # which focus_env the two rolling buffers belong to

    # ------------------------------------------------------------------ draw

    def _geometry(self, env_idx: int | None = None):
        idx = self.state.focus_env if env_idx is None else env_idx
        g = extract_geometry(self.env.world, idx, scenario=self.scenario)
        if env_idx is None and self.style.trajectory_mode != "none" and self._trail_history:
            g.extras["trajectories"] = np.stack(self._trail_history, axis=1)
        return g

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

    def _interaction_controller(self, camera: Camera) -> InteractionController:
        if self._controller is None:
            self._controller = InteractionController(
                self.state,
                camera,
                geometry_getter=self._geometry,
                tile_resolver=self._tile_env_at,
                on_drag_agent=self._write_agent_pos if self.allow_write_back else None,
                on_drag_obstacle=self._write_obstacle_pos if self.allow_write_back else None,
                on_place_goal=self._write_goal if self.allow_write_back else None,
                on_cycle_trajectory=self._cycle_trajectory,
                on_cycle_lidar=self._cycle_lidar,
                on_cycle_color=self._cycle_color,
            )
        return self._controller

    def _cycle_trajectory(self) -> None:
        modes = ("none", "trail", "fade")
        self.style.trajectory_mode = modes[
            (modes.index(self.style.trajectory_mode) + 1) % len(modes)
        ]
        if self.style.trajectory_mode == "none":
            self._trail_history.clear()

    def _cycle_lidar(self) -> None:
        modes = ("none", "rays", "area", "both")
        self.style.lidar_mode = modes[(modes.index(self.style.lidar_mode) + 1) % len(modes)]

    def _cycle_color(self) -> None:
        modes = ("agent", "model")
        self.style.color_mode = modes[(modes.index(self.style.color_mode) + 1) % len(modes)]

    def _append_trail_sample(self) -> None:
        if self.style.trajectory_mode == "none":
            return
        pos = self.env.world.state.pos[self.state.focus_env].detach().to("cpu").numpy().copy()
        self._trail_history.append(pos)
        if len(self._trail_history) > self.style.trajectory_len:
            del self._trail_history[: len(self._trail_history) - self.style.trajectory_len]

    def _append_reward_sample(self, reward) -> None:
        """Buffer one ``[n_agents]`` reward row for the HUD sparkline.

        Costs exactly one small device->host copy per *rendered* step, and only while
        ``style.reward_hud`` is on. The viewer never runs on the differentiable hot path.
        """
        if not self.style.reward_hud or reward is None:
            return
        row = reward[self.state.focus_env].detach().to("cpu").numpy().astype(np.float32, copy=True)
        self._reward_history.append(row)
        if len(self._reward_history) > self.style.reward_hud_len:
            del self._reward_history[: len(self._reward_history) - self.style.reward_hud_len]

    def _invalidate_buffers_on_focus_change(self) -> None:
        """Drop the trail and reward buffers when the focus env changes.

        Both buffers hold one env's history; without this, ``[``/``]`` smears one env's trail
        and rewards onto the next. Cleared together so the two cannot drift apart.
        """
        if self._buffer_env != self.state.focus_env:
            self._trail_history.clear()
            self._reward_history.clear()
            self._buffer_env = self.state.focus_env

    def _steps_this_frame(self, done: bool) -> int:
        """How many sim steps this rendered frame advances; consumes a ``step_once`` request.

        The frame rate stays at ``self.fps`` at every playback speed and
        ``steps_per_frame * state.speed`` decides how many steps ride on each frame — so slow
        motion keeps the window as responsive to input as 1x, instead of ticking the clock
        (and the event pump) at 3 fps. Fractional rates carry across frames: 0.25 steps per
        frame advances on every fourth frame.
        """
        once = self.state.step_once
        self.state.step_once = False
        if done:
            self._step_accum = 0.0
            return 0
        if self.state.paused:
            self._step_accum = 0.0
            return 1 if once else 0
        self._step_accum += self.steps_per_frame * self.state.speed
        n = int(self._step_accum)
        self._step_accum -= n
        return n

    def _apply_reset_request(self):
        if not self.state.reset_requested:
            return None
        obs = self.env.reset()
        self._step_count = 0
        self._trail_history.clear()
        self._reward_history.clear()
        self.state.reset_requested = False
        return obs

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
            if self._trail_history:
                self._trail_history[-1][agent_idx] = np.asarray(world_xy, dtype=np.float64)
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

    def _write_obstacle_pos(self, obstacle_idx: int, world_xy) -> None:
        """Move an obstacle in the focus env and refresh the stepper obstacle buffers."""
        world = self.env.world
        if world.obstacle_pos is None or world.obstacle_radius is None:
            return
        with torch.no_grad():
            world.obstacle_pos[self.state.focus_env, obstacle_idx] = torch.tensor(
                world_xy, dtype=self.env.dtype, device=self.env.device
            )
            world.set_obstacles(
                world.obstacle_pos,
                world.obstacle_radius,
                shape=world.obstacle_shape,
                angle=world.obstacle_angle,
                half_extents=world.obstacle_half_extents,
            )

    def _draw(
        self, surface, geometry, camera, style, *, hud: bool, fps=None, clear: bool = True
    ) -> None:
        draw_scene(surface, geometry, camera, self.state.enabled, style, clear=clear)
        if hud:
            draw_hud(surface, self.state, style, step=self._step_count, fps=fps)
            draw_hover_panel(surface, geometry, self.state.hover_agent, style)
            rewards = np.stack(self._reward_history) if len(self._reward_history) > 1 else None
            draw_reward_hud(surface, rewards, geometry, style)
            badge_h = draw_speed_badge(surface, self.state, style)
            draw_help(surface, self.state, style, top_offset=badge_h)

    def _draw_tile(self, pygame, surface, geometry, camera, rect, style, *, focused: bool) -> None:
        r = pygame.Rect(*rect)
        surface.fill(style.tile_background, r)
        prev = surface.get_clip()
        surface.set_clip(r)
        if geometry.bounds is not None:
            x0, x1, y0, y1 = geometry.bounds
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            pygame.draw.lines(
                surface,
                style.bounds_color,
                True,
                [_p(camera, c) for c in corners],
                style.tile_border_width,
            )
        for i in range(geometry.n_agents):
            pygame.draw.circle(
                surface,
                style.agent_color(i, geometry.model[i]),
                _p(camera, geometry.pos[i]),
                _r_px(camera, geometry.radius[i], floor=style.tile_agent_min_px),
            )
        surface.set_clip(prev)
        border = style.tile_focus_border if focused else style.tile_border
        width = style.tile_focus_border_width if focused else style.tile_border_width
        pygame.draw.rect(surface, border, r, width)

    def _draw_mosaic(self, pygame, surface, scale: int, style, *, hud: bool, fps=None) -> None:
        # Rects come from the *scaled* layout; self._layout stays window-space for tile_at().
        layout = self._ensure_layout().scaled(scale)
        surface.fill(style.background)
        # Tiles draw only bounds/pos/radius, so skip the neighbor rebuild and the sensor hook.
        tiles = extract_geometry_batch(
            self.env.world, layout.tile_envs, with_edges=False, with_extras=False
        )
        for tile_rect, env_idx, g in zip(layout.tiles, layout.tile_envs, tiles, strict=True):
            cam = Camera(g.bounds or _bounds_from_geometry(g), viewport=tile_rect)
            self._draw_tile(
                pygame, surface, g, cam, tile_rect, style, focused=env_idx == self.state.focus_env
            )
        focus = pygame.Rect(*layout.focus_rect)
        surface.fill(style.background, focus)
        prev = surface.get_clip()
        surface.set_clip(focus)
        gf = self._geometry()
        self._draw(
            surface, gf, self._camera_for(gf).scaled(scale), style, hud=hud, fps=fps, clear=False
        )
        surface.set_clip(prev)
        pygame.draw.rect(surface, style.bounds_color, focus, style.tile_border_width)

    def _render_onto(self, pygame, surface, *, hud: bool, fps=None) -> None:
        """Render the current view onto ``surface``, supersampling when the style asks for it."""
        self._invalidate_buffers_on_focus_change()
        draw_supersampled(
            pygame,
            surface,
            self.style.supersample,
            lambda surf, s: self._render_onto_scaled(pygame, surf, s, hud=hud, fps=fps),
        )

    def _render_onto_scaled(self, pygame, surface, scale: int, *, hud: bool, fps=None) -> None:
        style = self.style.scaled(scale)
        if self.mosaic:
            self._draw_mosaic(pygame, surface, scale, style, hud=hud, fps=fps)
        else:
            g = self._geometry()
            self._draw(surface, g, self._camera_for(g).scaled(scale), style, hud=hud, fps=fps)

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
        camera = self._camera_for(self._geometry())
        controller = self._interaction_controller(camera)
        for event in pygame.event.get():  # keep the window responsive / closeable
            controller.handle_event(event)
        self._apply_reset_request()
        self._append_trail_sample()
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
        also closes the window (useful for scripted/headless-dummy runs). Up/down change the
        playback speed around the nominal ``fps`` (see :meth:`_steps_this_frame`).
        """
        pygame = _ensure_pygame()
        pygame.display.init()
        window = pygame.display.set_mode(self.size)
        pygame.display.set_caption("wmas viewer")
        clock = pygame.time.Clock()
        camera = self._camera_for(self._geometry())
        controller = self._interaction_controller(camera)
        obs = self.scenario.observations() if self.scenario is not None else None
        try:
            while not self.state.quit:
                for event in pygame.event.get():
                    controller.handle_event(event)
                reset_obs = self._apply_reset_request()
                if reset_obs is not None:
                    obs = reset_obs
                done = max_steps is not None and self._step_count >= max_steps
                for _ in range(self._steps_this_frame(done)):
                    with torch.no_grad():
                        obs, reward, *_ = self.env.step(self._actions(action_fn, obs))
                    self._step_count += 1
                    self._append_trail_sample()
                    self._append_reward_sample(reward)
                    if max_steps is not None and self._step_count >= max_steps:
                        done = True
                        break
                self._render_onto(pygame, window, hud=True, fps=clock.get_fps())
                pygame.display.flip()
                clock.tick(self.fps)
                if done and close_when_done:
                    self.state.quit = True
        finally:
            pygame.display.quit()
