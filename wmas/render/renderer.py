"""Turn a :class:`RenderGeometry` into pixels with pygame.

``draw_scene`` paints onto any pygame ``Surface`` (used by both the live window and the
offscreen path). ``render_frame`` is the headless primitive: it needs no display — pygame
draws to a software ``Surface`` and we read it back as an ``(H, W, 3)`` uint8 array, which
feeds video export and notebook embedding alike.
"""

from __future__ import annotations

import numpy as np

from wmas.render.camera import Camera
from wmas.render.geometry import RenderGeometry
from wmas.render.overlays import DEFAULT_ENABLED, OVERLAYS
from wmas.render.style import Style


def _ensure_pygame():
    try:
        import pygame
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError(
            "wmas visualization requires the optional 'viz' extra: "
            "pip install 'wmas[viz]'  (or  uv pip install -e '.[viz]')"
        ) from exc
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame


_HIRES_CACHE: dict[tuple[int, int], object] = {}


def _hires_surface(pygame, size: tuple[int, int]):
    """A reusable offscreen surface of ``size`` — the run loop must not allocate per frame."""
    key = (int(size[0]), int(size[1]))
    surf = _HIRES_CACHE.get(key)
    if surf is None:
        if len(_HIRES_CACHE) > 4:  # window resizes / supersample changes; keep the cache small
            _HIRES_CACHE.clear()
        surf = pygame.Surface(key)
        _HIRES_CACHE[key] = surf
    return surf


def draw_supersampled(pygame, target, factor: int, draw_fn) -> None:
    """Run ``draw_fn(surface, scale)`` at ``factor``x offscreen, then smoothscale onto ``target``.

    ``factor <= 1`` draws straight onto ``target`` with ``scale=1``, at zero overhead.
    ``draw_fn`` owns its coordinate spaces — it must scale the camera (:meth:`Camera.scaled`),
    the style (:meth:`Style.scaled`) and any layout rects itself. Scaling the camera alone
    would leave 1px features 1px wide in the hi-res buffer, so they would downscale to a
    washed-out partial pixel: thinner, not smoother.
    """
    f = max(1, int(factor))
    if f == 1:
        draw_fn(target, 1)
        return
    w, h = target.get_size()
    hi = _hires_surface(pygame, (w * f, h * f))
    hi.set_clip(None)
    draw_fn(hi, f)
    try:
        pygame.transform.smoothscale(hi, (w, h), target)
    except (ValueError, TypeError):  # target format rejected as a destination surface
        target.blit(pygame.transform.smoothscale(hi, (w, h)), (0, 0))


def _bounds_from_geometry(g: RenderGeometry) -> tuple[float, float, float, float]:
    """Fallback view box when the world has no bounds: fit all drawable points + a margin."""
    pts = [g.pos]
    if g.goals is not None:
        pts.append(g.goals)
    if g.obstacle_pos is not None:
        pts.append(g.obstacle_pos)
    allp = np.concatenate(pts, axis=0)
    x0, y0 = allp.min(axis=0)
    x1, y1 = allp.max(axis=0)
    pad = 2.0 * float(g.radius.max()) + 0.1 * max(x1 - x0, y1 - y0, 1e-3)
    return (float(x0 - pad), float(x1 + pad), float(y0 - pad), float(y1 + pad))


def draw_scene(
    surface, geometry: RenderGeometry, camera: Camera, enabled, style: Style, *, clear: bool = True
) -> None:
    """Draw every enabled overlay in registry (=draw) order.

    ``clear`` fills the whole surface with the background first; pass ``clear=False`` when the
    caller has already prepared the region (e.g. a clipped focus pane inside a mosaic).
    """
    if clear:
        surface.fill(style.background)
    for overlay in OVERLAYS:
        if overlay.name in enabled:
            overlay.draw(surface, geometry, camera, style)


def render_frame(
    geometry: RenderGeometry,
    size: tuple[int, int] = (600, 600),
    overlays: set[str] | None = None,
    style: Style | None = None,
    camera: Camera | None = None,
) -> np.ndarray:
    """Render one env to an ``(H, W, 3)`` uint8 RGB array, headless (no window)."""
    pygame = _ensure_pygame()
    width, height = size
    style = style or Style()
    enabled = DEFAULT_ENABLED if overlays is None else set(overlays)
    if camera is None:
        bounds = geometry.bounds if geometry.bounds is not None else _bounds_from_geometry(geometry)
        camera = Camera(bounds, viewport=(0, 0, width, height))

    surface = pygame.Surface((width, height))
    draw_supersampled(
        pygame,
        surface,
        style.supersample,
        lambda surf, s: draw_scene(surf, geometry, camera.scaled(s), enabled, style.scaled(s)),
    )
    arr = pygame.surfarray.array3d(surface)  # (W, H, 3)
    return np.ascontiguousarray(np.transpose(arr, (1, 0, 2)))  # (H, W, 3)
