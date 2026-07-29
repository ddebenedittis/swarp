"""Mosaic + focus layout: viewport rectangles for a batch of envs.

A large focus pane (full overlays, interactive) on the left; a grid of small read-only tiles
on the right, one per env. When there are more envs than ``max_tiles`` the extra ones are
hidden and the count is logged (never silently dropped).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger("swarp.render")

Rect = tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class MosaicLayout:
    focus_rect: Rect
    tiles: list[Rect]
    tile_envs: list[int]
    n_hidden: int = 0

    def scaled(self, factor: int) -> MosaicLayout:
        """A copy with every rect multiplied by ``factor`` (for supersampled rendering).

        The unscaled layout stays authoritative for :func:`tile_at`, which resolves clicks in
        window pixels.
        """
        f = max(1, int(factor))
        if f == 1:
            return self
        return MosaicLayout(
            focus_rect=_scale_rect(self.focus_rect, f),
            tiles=[_scale_rect(t, f) for t in self.tiles],
            tile_envs=list(self.tile_envs),
            n_hidden=self.n_hidden,
        )


def _scale_rect(rect: Rect, factor: int) -> Rect:
    x, y, w, h = rect
    return (x * factor, y * factor, w * factor, h * factor)


def compute_mosaic_layout(
    size: tuple[int, int],
    n_envs: int,
    *,
    focus_frac: float = 0.68,
    max_tiles: int = 16,
    gap: int = 4,
) -> MosaicLayout:
    width, height = int(size[0]), int(size[1])
    focus_w = int(width * focus_frac)
    focus_rect = (0, 0, focus_w, height)

    region_x = focus_w + gap
    region_w = max(1, width - region_x)

    n_shown = max(1, min(n_envs, max_tiles))
    n_hidden = max(0, n_envs - n_shown)
    if n_hidden:
        logger.warning(
            "mosaic: showing %d of %d envs (%d hidden); raise max_tiles to see more",
            n_shown,
            n_envs,
            n_hidden,
        )

    # Prefer a column count matched to the (usually tall, narrow) mosaic strip's aspect.
    cols = max(1, min(n_shown, math.ceil(math.sqrt(n_shown * region_w / max(height, 1)))))
    rows = math.ceil(n_shown / cols)
    tile_w = max(1, (region_w - (cols + 1) * gap) // cols)
    tile_h = max(1, (height - (rows + 1) * gap) // rows)

    tiles: list[Rect] = []
    for k in range(n_shown):
        row, col = divmod(k, cols)
        x = region_x + gap + col * (tile_w + gap)
        y = gap + row * (tile_h + gap)
        tiles.append((x, y, tile_w, tile_h))

    return MosaicLayout(focus_rect, tiles, list(range(n_shown)), n_hidden)


def tile_at(layout: MosaicLayout, screen_xy) -> int | None:
    """Env index of the tile containing ``screen_xy`` (None if in the focus pane / gaps)."""
    x, y = float(screen_xy[0]), float(screen_xy[1])
    for k, (tx, ty, tw, th) in enumerate(layout.tiles):
        if tx <= x < tx + tw and ty <= y < ty + th:
            return layout.tile_envs[k]
    return None
