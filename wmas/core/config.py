"""World-level configuration."""

from __future__ import annotations

from dataclasses import dataclass

from wmas.dynamics.base import Integrator


@dataclass
class WorldConfig:
    """Interaction and boundary settings shared by all envs.

    Attributes:
        collisions: enable agent-agent soft collision forces.
        collision_k: spring stiffness of the soft penalty.
        collision_c: damping coefficient (normal direction).
        collision_margin: forces activate within this gap around touching radii.
        bounds: world rectangle ``(x_min, x_max, y_min, y_max)``, or ``None``.
        bounds_mode: ``"soft"`` (spring-damper walls) or ``"clamp"``
            (positions hard-clamped inside; differentiable subgradient).
        neighbor_radius: radius for the neighbor lists; defaults to the
            interaction reach ``2 * max_agent_radius + collision_margin``.
            Must not be smaller than that reach when collisions are on.
        max_neighbors: padded neighbor-list width (truncates beyond).
        neighbor_method: ``"auto"`` | ``"grid"`` | ``"brute"`` — see
            :class:`wmas.core.neighbors.NeighborGrid`.
        grid_dim: hash-grid bucket dimension per axis.
    """

    collisions: bool = True
    collision_k: float = 100.0
    collision_c: float = 1.0
    collision_margin: float = 0.02
    bounds: tuple[float, float, float, float] | None = None
    bounds_mode: str = "soft"
    neighbor_radius: float | None = None
    max_neighbors: int = 32
    neighbor_method: str = "auto"
    grid_dim: int = 128
    integrator: Integrator = Integrator.EULER

    def __post_init__(self) -> None:
        if self.bounds_mode not in ("soft", "clamp"):
            raise ValueError('bounds_mode must be "soft" or "clamp"')
        if self.bounds is not None:
            x_min, x_max, y_min, y_max = self.bounds
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("bounds must satisfy x_min < x_max and y_min < y_max")
        if self.collision_k < 0.0 or self.collision_c < 0.0 or self.collision_margin < 0.0:
            raise ValueError("collision constants must be non-negative")
