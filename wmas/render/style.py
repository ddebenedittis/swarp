"""Colors and sizes for the renderer. One place to restyle the whole viewer.

Every color and every pixel-valued size in the renderer lives here — no module outside this
one may hardcode either, because two mechanisms depend on total coverage:

- :meth:`Style.font_px` derives text size from the viewport height, so a small window or a
  mosaic tile gets proportionally smaller text instead of a fixed 15px.
- :meth:`Style.scaled` multiplies every pixel field for supersampled rendering (see
  ``renderer.draw_supersampled``). A width left as a literal would stay 1px in the hi-res
  buffer and downscale to a washed-out half-covered pixel — i.e. thinner, not smoother.

:data:`_PX_FIELDS` and :data:`_UNSCALED_FIELDS` partition the dataclass exactly, so adding a
field without classifying it fails ``test_style_field_partition_is_exhaustive``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]

# Okabe-Ito colorblind-safe qualitative palette (per-agent cycling).
_PALETTE: tuple[RGB, ...] = (
    (0, 114, 178),
    (230, 159, 0),
    (0, 158, 115),
    (204, 121, 167),
    (86, 180, 233),
    (213, 94, 0),
    (150, 150, 30),
    (100, 60, 160),
)


@dataclass
class Style:
    # ----------------------------------------------------------------- colors
    background: RGB = (245, 245, 247)
    bounds_color: RGB = (120, 120, 130)
    obstacle_color: RGB = (150, 150, 155)
    edge_color: RGB = (175, 175, 185)
    agent_outline: RGB = (30, 30, 35)
    heading_color: RGB = (30, 30, 35)
    velocity_color: RGB = (200, 60, 60)
    lidar_color: RGB = (60, 170, 200)
    comm_line_color: RGB = (120, 190, 140)
    text_color: RGB = (40, 40, 45)
    tile_background: RGB = (236, 236, 240)
    tile_border: RGB = (185, 185, 190)
    tile_focus_border: RGB = (200, 60, 60)
    help_panel_bg: RGBA = (255, 255, 255, 225)
    hover_panel_bg: RGBA = (255, 255, 255, 210)
    panel_border: RGB = (40, 40, 45)
    agent_halo: RGB = (245, 245, 247)  # depth_cue="halo"; defaults to the light background
    agent_shadow: RGBA = (0, 0, 0, 70)  # depth_cue="shadow"
    contact_color: RGB = (220, 50, 40)
    goal_reached_color: RGB = (0, 158, 115)
    palette: tuple[RGB, ...] = field(default_factory=lambda: _PALETTE)
    model_palette: dict[int, RGB] = field(
        default_factory=lambda: {
            0: (0, 114, 178),
            1: (213, 94, 0),
            2: (0, 158, 115),
            3: (204, 121, 167),
        }
    )

    # ------------------------------------------------------------------ modes
    color_mode: str = "agent"  # agent | model
    lidar_mode: str = "rays"  # none | rays | area | both
    trajectory_mode: str = "none"  # none | trail | fade
    depth_cue: str = "halo"  # none | halo | shadow
    goal_connector: str = "dashed"  # none | solid | dashed
    contact_highlight: bool = True

    # ------------------------------------------------------------ pixel sizes
    line_width: int = 2
    edge_width: int = 1
    lidar_hit_px: int = 2
    lidar_ray_width: int = 1
    goal_ring_width: int = 3
    goal_dot_px: int = 2
    agent_min_px: int = 3
    goal_min_px: int = 5
    agent_outline_width: int = 1
    agent_halo_px: int = 3
    agent_shadow_offset_px: int = 3
    contact_outline_width: int = 3
    goal_dash_px: int = 6
    goal_gap_px: int = 4
    tile_border_width: int = 1
    tile_focus_border_width: int = 2
    tile_agent_min_px: int = 2
    panel_pad: int = 8
    hover_pad: int = 6
    hud_margin: int = 8
    id_offset_px: int = 4

    # ------------------------------------------------------------------ fonts
    font_size: int = 15  # fallback when no viewport height is known
    font_scale: float = 0.0214  # fraction of viewport height; 0.0214 * 700 -> 15
    font_min: int = 11
    font_max: int = 28

    # --------------------------------------------------------- scalars, ratios
    heading_len_factor: float = 1.6  # relative to agent radius
    velocity_scale: float = 0.3  # world units per (m/s)
    lidar_area_alpha: int = 34
    trajectory_alpha: int = 125
    trajectory_fade_min_alpha: int = 20
    trajectory_len: int = 80
    supersample: int = 2  # 1 disables; >1 renders NxN offscreen and downscales
    goal_connector_alpha: int = 90
    goal_reached_factor: float = 1.0  # goal counts as reached within this many agent radii
    contact_tol: float = 0.0  # slack on (r_i + r_j) before an overlap counts as contact

    # ------------------------------------------------------------------ colors
    def agent_color(self, i: int, model: int | None = None) -> RGB:
        if self.color_mode == "model" and model is not None:
            return self.model_palette.get(int(model), self.palette[i % len(self.palette)])
        return self.palette[i % len(self.palette)]

    # ------------------------------------------------------------------- fonts
    def font_px(self, viewport_h: float | None = None) -> int:
        """Text size for a viewport ``viewport_h`` pixels tall, clamped to [min, max].

        Falls back to :attr:`font_size` when the height is unknown or scaling is disabled
        (``font_scale <= 0``). The default ratio reproduces ``font_size`` exactly at the
        default 700px viewport, so scaling changes nothing at the default window size.
        """
        if viewport_h is None or self.font_scale <= 0.0:
            return self.font_size
        return int(min(self.font_max, max(self.font_min, round(self.font_scale * viewport_h))))

    # ------------------------------------------------------------------ themes
    @classmethod
    def light(cls, **overrides) -> Style:
        """The default light theme (identical to ``Style(**overrides)``)."""
        return cls(**overrides)

    @classmethod
    def dark(cls, **overrides) -> Style:
        """Dark theme. The bicycle cabin fill uses :attr:`background`, so it follows along."""
        return cls(**{**_DARK, **overrides})

    # ------------------------------------------------------------- supersample
    def scaled(self, factor: int) -> Style:
        """Copy with every pixel-valued field multiplied by ``factor`` (clamped to >= 1px).

        ``font_scale`` is deliberately *not* scaled: the viewport height handed to
        :meth:`font_px` is already the hi-res one, so the ratio stays correct; only the
        clamps and the fixed fallback need scaling.
        """
        f = max(1, int(factor))
        if f == 1:
            return self
        return replace(self, **{name: _scale_px(getattr(self, name), f) for name in _PX_FIELDS})


def _scale_px(value, factor: int):
    if isinstance(value, tuple):
        return tuple(max(1, int(v) * factor) for v in value)
    return max(1, int(value) * factor)


# Fields measured in pixels: multiplied by Style.scaled() for supersampled rendering.
_PX_FIELDS: frozenset[str] = frozenset(
    {
        "line_width",
        "edge_width",
        "lidar_hit_px",
        "lidar_ray_width",
        "goal_ring_width",
        "goal_dot_px",
        "agent_min_px",
        "goal_min_px",
        "agent_outline_width",
        "agent_halo_px",
        "agent_shadow_offset_px",
        "contact_outline_width",
        "goal_dash_px",
        "goal_gap_px",
        "tile_border_width",
        "tile_focus_border_width",
        "tile_agent_min_px",
        "panel_pad",
        "hover_pad",
        "hud_margin",
        "id_offset_px",
        "font_size",
        "font_min",
        "font_max",
    }
)

# Everything else: colors, mode strings, alphas, ratios and counts — resolution-independent.
_UNSCALED_FIELDS: frozenset[str] = frozenset(
    {
        "background",
        "bounds_color",
        "obstacle_color",
        "edge_color",
        "agent_outline",
        "heading_color",
        "velocity_color",
        "lidar_color",
        "comm_line_color",
        "text_color",
        "tile_background",
        "tile_border",
        "tile_focus_border",
        "help_panel_bg",
        "hover_panel_bg",
        "panel_border",
        "agent_halo",
        "agent_shadow",
        "contact_color",
        "goal_reached_color",
        "palette",
        "model_palette",
        "color_mode",
        "lidar_mode",
        "trajectory_mode",
        "depth_cue",
        "goal_connector",
        "contact_highlight",
        "goal_connector_alpha",
        "goal_reached_factor",
        "contact_tol",
        "font_scale",
        "heading_len_factor",
        "velocity_scale",
        "lidar_area_alpha",
        "trajectory_alpha",
        "trajectory_fade_min_alpha",
        "trajectory_len",
        "supersample",
    }
)

_DARK: dict[str, object] = {
    "background": (18, 18, 22),
    "bounds_color": (90, 90, 100),
    "obstacle_color": (78, 78, 86),
    "edge_color": (70, 70, 80),
    "agent_outline": (235, 235, 240),
    "heading_color": (235, 235, 240),
    "velocity_color": (235, 105, 95),
    "text_color": (228, 228, 234),
    "tile_background": (28, 28, 34),
    "tile_border": (60, 60, 70),
    "tile_focus_border": (235, 105, 95),
    "help_panel_bg": (26, 26, 32, 230),
    "hover_panel_bg": (26, 26, 32, 215),
    "panel_border": (120, 120, 130),
    "agent_halo": (18, 18, 22),
}

THEMES: dict[str, object] = {"light": Style.light, "dark": Style.dark}


def style_field_names() -> frozenset[str]:
    """Every :class:`Style` field name — the partition invariant's left-hand side."""
    return frozenset(f.name for f in fields(Style))
