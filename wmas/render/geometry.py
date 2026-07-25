"""Device->host geometry extraction: pull ONE env's renderable state to CPU numpy.

This is the single synchronization boundary between the GPU-resident simulator and the
(CPU, pygame) renderer. It is deliberately backend-agnostic — it knows nothing about
pygame — so any renderer can consume a :class:`RenderGeometry`. Extraction is read-only
and never runs on the differentiable hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class RenderGeometry:
    """One env's renderable state, as plain CPU numpy arrays.

    Agent-index conventions are per-env local (0..n_agents-1); ``edges`` holds
    within-radius neighbor pairs ``(i, j)`` in those local indices.
    """

    n_agents: int
    pos: np.ndarray  # (n_agents, 2)
    theta: np.ndarray  # (n_agents,)
    vel: np.ndarray  # (n_agents, 2)
    radius: np.ndarray  # (n_agents,)
    model: np.ndarray  # (n_agents,) int DynamicsModel tags
    goals: np.ndarray | None  # (n_agents, 2) or None
    obstacle_pos: np.ndarray | None  # (n_obstacles, 2) or None
    obstacle_radius: np.ndarray | None  # (n_obstacles,) or None
    bounds: tuple[float, float, float, float] | None  # (x_min, x_max, y_min, y_max)
    edges: np.ndarray  # (E, 2) int64, local (i, j) neighbor pairs
    extras: dict[str, Any] = field(default_factory=dict)


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").numpy()


def extract_geometry(world, env_idx: int, scenario=None) -> RenderGeometry:
    """Extract env ``env_idx`` of ``world`` as a :class:`RenderGeometry`.

    ``scenario`` is optional; if it defines ``render_extras(env_idx) -> dict`` the
    result is merged into :attr:`RenderGeometry.extras` for custom overlays.
    """
    s = world.state
    goals = _to_np(world.goals[env_idx]) if world.goals is not None else None
    if world.obstacle_pos is not None:
        obstacle_pos = _to_np(world.obstacle_pos[env_idx])
        obstacle_radius = _to_np(world.obstacle_radius)
    else:
        obstacle_pos = None
        obstacle_radius = None

    bounds = world.config.bounds if world.config is not None else None

    extras: dict[str, Any] = {}
    if scenario is not None and hasattr(scenario, "render_extras"):
        extras = scenario.render_extras(env_idx) or {}

    return RenderGeometry(
        n_agents=world.n_agents,
        pos=_to_np(s.pos[env_idx]),
        theta=_to_np(s.theta[env_idx]),
        vel=_to_np(s.vel[env_idx]),
        radius=_to_np(world.agent_radius),
        model=np.array([int(c.model) for c in world.agent_configs], dtype=np.int32),
        goals=goals,
        obstacle_pos=obstacle_pos,
        obstacle_radius=obstacle_radius,
        bounds=bounds,
        edges=_extract_edges(world, env_idx),
        extras=extras,
    )


def _extract_edges(world, env_idx: int) -> np.ndarray:
    """Within-radius neighbor pairs of one env as (E, 2) local (i, j) indices.

    Uses ``World.neighbors()`` (per-env local indices; see neighbors.py) rather than
    the batch-global ``edge_index()``. Returns an empty (0, 2) array when collisions
    (hence neighbor lists) are disabled.
    """
    if not getattr(world.stepper, "collisions", False):
        return np.empty((0, 2), dtype=np.int64)

    neighbor_idx, neighbor_count = world.neighbors()
    idx = neighbor_idx[env_idx].to("cpu").numpy()  # (n_agents, K)
    cnt = neighbor_count[env_idx].to("cpu").numpy()  # (n_agents,)

    pairs = [(i, int(idx[i, k])) for i in range(idx.shape[0]) for k in range(int(cnt[i]))]
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(pairs, dtype=np.int64)
