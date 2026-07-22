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
        world_w = max(x_max - x_min, 1e-12)
        world_h = max(y_max - y_min, 1e-12)
        avail_w = self.vw * (1.0 - 2.0 * margin_frac)
        avail_h = self.vh * (1.0 - 2.0 * margin_frac)
        self._base_scale = min(avail_w / world_w, avail_h / world_h)

        self.vcx = self.vx + 0.5 * self.vw
        self.vcy = self.vy + 0.5 * self.vh

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
