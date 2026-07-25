"""Colors and sizes for the renderer. One place to restyle the whole viewer."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    palette: tuple[RGB, ...] = field(default_factory=lambda: _PALETTE)
    model_palette: dict[int, RGB] = field(
        default_factory=lambda: {
            0: (0, 114, 178),
            1: (213, 94, 0),
            2: (0, 158, 115),
            3: (204, 121, 167),
        }
    )

    color_mode: str = "agent"  # agent | model
    lidar_mode: str = "rays"  # none | rays | area | both
    trajectory_mode: str = "none"  # none | trail | fade

    line_width: int = 2
    edge_width: int = 1
    lidar_hit_px: int = 2
    goal_ring_width: int = 3
    agent_min_px: int = 3
    goal_min_px: int = 5
    heading_len_factor: float = 1.6  # relative to agent radius
    velocity_scale: float = 0.3  # world units per (m/s)
    font_size: int = 15
    lidar_area_alpha: int = 34
    trajectory_alpha: int = 125
    trajectory_fade_min_alpha: int = 20
    trajectory_len: int = 80

    def agent_color(self, i: int, model: int | None = None) -> RGB:
        if self.color_mode == "model" and model is not None:
            return self.model_palette.get(int(model), self.palette[i % len(self.palette)])
        return self.palette[i % len(self.palette)]
