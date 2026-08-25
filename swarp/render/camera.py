"""World<->screen affine for a single viewport.

Fits a world-space rectangle into a pixel viewport with equal aspect and a margin, then
maps points both ways. Screen space is top-left origin with y growing downward, so world
y is flipped. ``world_to_screen`` / ``screen_to_world`` are vectorized and exact inverses.
"""

from __future__ import annotations

import numpy as np


class Camera:
    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        viewport: tuple[float, float, float, float],
        zoom: float = 1.0,
        margin_frac: float = 0.05,
    ) -> None:
        x_min, x_max, y_min, y_max = bounds
        self.bounds = bounds
        self.vx, self.vy, self.vw, self.vh = viewport
        self.margin_frac = margin_frac
        self.zoom = zoom
        self.pan_x = 0.0
        self.pan_y = 0.0

        self.wcx = 0.5 * (x_min + x_max)
        self.wcy = 0.5 * (y_min + y_max)
        self._fit_viewport()

    def _fit_viewport(self) -> None:
        x_min, x_max, y_min, y_max = self.bounds
        world_w = max(x_max - x_min, 1e-12)
        world_h = max(y_max - y_min, 1e-12)
        avail_w = self.vw * (1.0 - 2.0 * self.margin_frac)
        avail_h = self.vh * (1.0 - 2.0 * self.margin_frac)
        self._base_scale = min(avail_w / world_w, avail_h / world_h)
        self.vcx = self.vx + 0.5 * self.vw
        self.vcy = self.vy + 0.5 * self.vh

    def resize_viewport(self, viewport: tuple[float, float, float, float]) -> None:
        """Re-fit scale/center to a new pixel viewport in place, resetting pan.

        Used when the window is resized: an existing pixel-space pan offset would otherwise
        misalign once the viewport it was computed against no longer exists. The camera
        object's identity is preserved, since :class:`InteractionController` holds a
        reference to it directly rather than re-fetching it each frame.
        """
        self.vx, self.vy, self.vw, self.vh = viewport
        self._fit_viewport()
        self.pan_x = 0.0
        self.pan_y = 0.0

    @property
    def scale(self) -> float:
        return self._base_scale * self.zoom

    def world_to_screen(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        x = self.vcx + self.pan_x + self.scale * (p[..., 0] - self.wcx)
        y = self.vcy + self.pan_y - self.scale * (p[..., 1] - self.wcy)
        return np.stack([x, y], axis=-1)

    def screen_to_world(self, s) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        wx = self.wcx + (s[..., 0] - self.vcx - self.pan_x) / self.scale
        wy = self.wcy - (s[..., 1] - self.vcy - self.pan_y) / self.scale
        return np.stack([wx, wy], axis=-1)

    def scaled(self, factor: float) -> Camera:
        """A copy whose screen space is ``factor`` x this one's: ``w2s'(p) == factor * w2s(p)``.

        Used for supersampled rendering. It must be a *copy*: the interaction controller holds
        the original and drives it with raw window-pixel coordinates, so scaling in place would
        silently break picking, panning and zoom-under-cursor.
        """
        f = float(factor)
        if f == 1.0:
            return self
        out = Camera(
            self.bounds,
            (self.vx * f, self.vy * f, self.vw * f, self.vh * f),
            zoom=self.zoom,
            margin_frac=self.margin_frac,
        )
        # _base_scale, vcx and vcy all scale by f from the viewport alone; pan is in pixels.
        out.pan_x = self.pan_x * f
        out.pan_y = self.pan_y * f
        return out

    def pan(self, dx: float, dy: float) -> Camera:
        """Shift the view by a screen-pixel delta (e.g. a mouse drag)."""
        self.pan_x += dx
        self.pan_y += dy
        return self

    def zoom_at(self, factor: float, screen_point) -> Camera:
        """Multiply zoom by ``factor``, keeping the world point under ``screen_point`` fixed."""
        wx, wy = self.screen_to_world(screen_point)
        self.zoom *= factor
        sx, sy = float(screen_point[0]), float(screen_point[1])
        # Solve world_to_screen(w) == screen_point for the new pan at the new scale.
        self.pan_x = sx - self.vcx - self.scale * (wx - self.wcx)
        self.pan_y = sy - self.vcy + self.scale * (wy - self.wcy)
        return self
