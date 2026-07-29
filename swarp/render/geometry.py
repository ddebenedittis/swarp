"""Device->host geometry extraction: pull renderable state to CPU numpy.

This is the single synchronization boundary between the GPU-resident simulator and the
(CPU, pygame) renderer. It is deliberately backend-agnostic — it knows nothing about
pygame — so any renderer can consume a :class:`RenderGeometry`. Extraction is read-only
and never runs on the differentiable hot path.

:func:`extract_geometry_batch` is the primitive; :func:`extract_geometry` is the single-env
case, so there is exactly one place that touches device memory. Batching matters for the
mosaic view, which would otherwise pay a neighbor-grid rebuild and a full set of transfers
per tile.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(kw_only=True)
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
    # Obstacle geometry beyond a radius. Per-obstacle (not per-env), mirroring the simulator;
    # None means "every obstacle is a circle", which is how Stepper.set_obstacles zero-fills.
    obstacle_shape: np.ndarray | None = None  # (n_obstacles,) int ObstacleShape tags
    obstacle_angle: np.ndarray | None = None  # (n_obstacles,) rad
    obstacle_half_extents: np.ndarray | None = None  # (n_obstacles, 2); [:, 0] = SEGMENT half-len
    # (n_obstacles,) ObstacleKind tags; None means "all immovable". Renderers draw
    # immovable obstacles black and movable (pushable) ones grey.
    obstacle_kind: np.ndarray | None = None
    # Last action applied to each agent — one step older than the state drawn alongside it.
    action: np.ndarray | None = None  # (n_agents, act_dim) or None before the first step
    ctrl_mode: np.ndarray | None = None  # (n_agents,) int ControlMode tags
    agent_params: np.ndarray | None = None  # (n_agents, NUM_PARAMS) AgentConfig.to_row() rows
    extras: dict[str, Any] = field(default_factory=dict)


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").numpy()


def extract_geometry(world, env_idx: int, scenario=None) -> RenderGeometry:
    """Extract env ``env_idx`` of ``world`` as a :class:`RenderGeometry`.

    ``scenario`` is optional; if it defines ``render_extras(env_idx) -> dict`` the
    result is merged into :attr:`RenderGeometry.extras` for custom overlays.
    """
    return extract_geometry_batch(world, (env_idx,), scenario)[0]


def extract_geometry_batch(
    world,
    env_indices: Sequence[int],
    scenario=None,
    *,
    with_edges: bool = True,
    with_extras: bool = True,
) -> list[RenderGeometry]:
    """Extract several envs with ONE device->host transfer per tensor field.

    Env-independent data (radii, dynamics tags, obstacle radii, bounds) is read once and
    shared by every returned geometry. ``with_edges=False`` skips ``world.neighbors()``
    entirely — that call rebuilds the neighbor grid, so a caller that does not draw edges
    (e.g. mosaic tiles) should switch it off. ``with_extras=False`` likewise skips the
    scenario hook, which may run real sensors.
    """
    idx = list(int(i) for i in env_indices)
    if not idx:
        return []

    s = world.state
    sel = torch.as_tensor(idx, dtype=torch.long, device=s.pos.device)
    pos = _to_np(s.pos.index_select(0, sel))
    theta = _to_np(s.theta.index_select(0, sel))
    vel = _to_np(s.vel.index_select(0, sel))
    goals = _to_np(world.goals.index_select(0, sel)) if world.goals is not None else None

    action = _to_np(world.action.index_select(0, sel)) if world.action is not None else None

    if world.obstacle_pos is not None:
        # Movable obstacles are advanced in place inside the stepper's own buffers, so
        # the live pose lives there, not in the tensor the scenario installed.
        live_pos, live_angle, _, _ = world.obstacle_state_views()
        obstacle_pos = _to_np(live_pos.index_select(0, sel))
        obstacle_radius = _to_np(world.obstacle_radius)
        # Same for orientation, but only where orientation is meaningful: a circle-only
        # scenario keeps obstacle_angle None (the documented "no angle" contract), while a
        # movable box needs the live value because it rotates as it is pushed.
        movable = getattr(world, "obstacle_kind", None) is not None and bool(
            (world.obstacle_kind != 0).any()
        )
        live_obs_angle = (
            _to_np(live_angle.index_select(0, sel))
            if (world.obstacle_angle is not None or movable)
            else None
        )
    else:
        obstacle_pos = None
        obstacle_radius = None
        live_obs_angle = None

    # Env-independent: extracted once, shared by every geometry below.
    radius = _to_np(world.agent_radius)
    model = np.array([int(c.model) for c in world.agent_configs], dtype=np.int32)
    ctrl_mode = np.array([int(c.ctrl_mode) for c in world.agent_configs], dtype=np.int32)
    agent_params = np.array([c.to_row() for c in world.agent_configs], dtype=np.float64)
    bounds = world.config.bounds if world.config is not None else None
    edges = _extract_edges_batch(world, idx) if with_edges else [_no_edges()] * len(idx)
    # Obstacle shape/half-extents are per-obstacle, so no env indexing here.
    obs_shape = _shape_tags(world)
    # Angle may be per-obstacle [n_obs] (shared) or per-env [n_envs, n_obs] — a
    # body that rotates independently per env. Env-index the latter so each
    # RenderGeometry still carries a flat (n_obstacles,) angle array.
    obs_angle = live_obs_angle if live_obs_angle is not None else _obstacle_angles(world, sel)
    obs_kind = (
        _to_np(world.obstacle_kind) if getattr(world, "obstacle_kind", None) is not None else None
    )
    obs_half = (
        _to_np(world.obstacle_half_extents) if world.obstacle_half_extents is not None else None
    )

    out: list[RenderGeometry] = []
    for k, env_idx in enumerate(idx):
        extras: dict[str, Any] = {}
        if with_extras and scenario is not None and hasattr(scenario, "render_extras"):
            extras = scenario.render_extras(env_idx) or {}
        out.append(
            RenderGeometry(
                n_agents=world.n_agents,
                pos=pos[k],
                theta=theta[k],
                vel=vel[k],
                radius=radius,
                model=model,
                goals=None if goals is None else goals[k],
                obstacle_pos=None if obstacle_pos is None else obstacle_pos[k],
                obstacle_radius=obstacle_radius,
                bounds=bounds,
                edges=edges[k],
                obstacle_shape=obs_shape,
                obstacle_angle=None if obs_angle is None else obs_angle[k],
                obstacle_half_extents=obs_half,
                obstacle_kind=obs_kind,
                action=None if action is None else action[k],
                ctrl_mode=ctrl_mode,
                agent_params=agent_params,
                extras=extras,
            )
        )
    return out


def _obstacle_angles(world, sel) -> np.ndarray | None:
    """Obstacle angles as ``(len(sel), n_obstacles)``, broadcasting the shared
    ``[n_obstacles]`` layout so callers index one row per selected env."""
    if world.obstacle_angle is None:
        return None
    a = world.obstacle_angle
    if a.dim() == 1:
        a = a.unsqueeze(0).expand(len(sel), -1)
    else:
        a = a.index_select(0, sel)
    return _to_np(a)


def _shape_tags(world) -> np.ndarray | None:
    if world.obstacle_shape is None:
        return None
    return _to_np(world.obstacle_shape).astype(np.int32)


def _no_edges() -> np.ndarray:
    return np.empty((0, 2), dtype=np.int64)


def _extract_edges_batch(world, env_indices: Sequence[int]) -> list[np.ndarray]:
    """Within-radius neighbor pairs per env, as (E, 2) local (i, j) indices.

    Uses ``World.neighbors()`` (per-env local indices; see neighbors.py) rather than the
    batch-global ``edge_index()``, and calls it **once** for the whole batch: it rebuilds the
    grid and returns views that a later call overwrites. Empty when collisions (hence
    neighbor lists) are disabled.
    """
    if not getattr(world.stepper, "collisions", False):
        return [_no_edges() for _ in env_indices]

    neighbor_idx, neighbor_count = world.neighbors()
    sel = torch.as_tensor(list(env_indices), dtype=torch.long, device=neighbor_idx.device)
    idx = _to_np(neighbor_idx.index_select(0, sel))  # (T, n_agents, K)
    cnt = _to_np(neighbor_count.index_select(0, sel))  # (T, n_agents)
    return [_pairs_from_lists(idx[k], cnt[k]) for k in range(len(env_indices))]


def _pairs_from_lists(idx: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """One env's (n_agents, K) neighbor lists + counts as an (E, 2) pair array."""
    pairs = [(i, int(idx[i, k])) for i in range(idx.shape[0]) for k in range(int(cnt[i]))]
    if not pairs:
        return _no_edges()
    return np.array(pairs, dtype=np.int64)
