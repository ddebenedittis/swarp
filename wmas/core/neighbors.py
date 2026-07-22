"""Within-radius neighbor search over all envs via a single Warp hash grid.

All envs share one ``wp.HashGrid``: 2D positions are lifted to 3D with
``z = env_idx * z_spacing`` where ``z_spacing`` exceeds the query radius, so
cross-env pairs are impossible by construction. Query results are a superset
(hash collisions, cell granularity) and are filtered by exact distance in the
kernel — the filter uses the same arithmetic as the brute-force reference, so
the two paths agree bit-exactly on membership.

The output is a padded neighbor list (``neighbor_idx``/``neighbor_count``),
i.e. a discrete structure carrying no gradients; downstream force kernels read
positions through these integer indices, which keeps the simulation step
differentiable w.r.t. positions while the neighbor *set* stays fixed within a
substep (also required for tape replay correctness — the grid is rebuilt every
substep, so adjoint kernels must not re-query it).
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from wmas.core.state import VEC2

VEC3 = {wp.float32: wp.vec3f, wp.float64: wp.vec3d}


@wp.kernel
def _fill_points(
    pos: wp.array2d(dtype=Any),
    z_spacing: Any,
    points: wp.array(dtype=Any),
):
    e, a = wp.tid()
    p = pos[e, a]
    points[e * pos.shape[1] + a] = type(points[0])(p[0], p[1], type(z_spacing)(e) * z_spacing)


@wp.kernel
def _query_grid(
    grid_id: wp.uint64,
    points: wp.array(dtype=Any),
    radius: Any,
    neighbor_idx: wp.array3d(dtype=wp.int32),
    neighbor_count: wp.array2d(dtype=wp.int32),
):
    e, a = wp.tid()
    n_agents = neighbor_idx.shape[1]
    max_neighbors = neighbor_idx.shape[2]
    i = e * n_agents + a
    p = points[i]
    count = wp.int32(0)
    query = wp.hash_grid_query(grid_id, p, radius * type(radius)(1.001))
    j = wp.int32(0)
    while wp.hash_grid_query_next(query, j):
        if j != i and wp.length(points[j] - p) <= radius:
            if count < max_neighbors:
                neighbor_idx[e, a, count] = j - e * n_agents
                count += wp.int32(1)
    neighbor_count[e, a] = count


@wp.kernel
def _brute_force(
    pos: wp.array2d(dtype=Any),
    radius: Any,
    neighbor_idx: wp.array3d(dtype=wp.int32),
    neighbor_count: wp.array2d(dtype=wp.int32),
):
    e, a = wp.tid()
    n_agents = pos.shape[1]
    max_neighbors = neighbor_idx.shape[2]
    p = pos[e, a]
    count = wp.int32(0)
    for b in range(n_agents):
        if b != a and wp.length(pos[e, b] - p) <= radius:
            if count < max_neighbors:
                neighbor_idx[e, a, count] = b
                count += wp.int32(1)
    neighbor_count[e, a] = count


for _T in (wp.float32, wp.float64):
    wp.overload(_fill_points, [wp.array2d(dtype=VEC2[_T]), _T, wp.array(dtype=VEC3[_T])])
    wp.overload(
        _query_grid,
        [wp.uint64, wp.array(dtype=VEC3[_T]), _T,
         wp.array3d(dtype=wp.int32), wp.array2d(dtype=wp.int32)],
    )
    wp.overload(
        _brute_force,
        [wp.array2d(dtype=VEC2[_T]), _T,
         wp.array3d(dtype=wp.int32), wp.array2d(dtype=wp.int32)],
    )


class NeighborGrid:
    """Padded within-radius neighbor lists for [n_envs, n_agents] worlds.

    ``method`` selects the search strategy:

    * ``"brute"`` — per-env O(n_agents²) kernel. Linear in ``n_envs`` with a
      tiny constant; the fastest choice for the usual multi-agent regime
      (measured ~300x faster than the grid at 16k envs x 64 agents).
    * ``"grid"`` — one ``wp.HashGrid`` over all envs via a z-lift. Warp's hash
      grid wraps *cell coordinates* modulo its dims, so many envs alias into
      the same cells and query cost grows with ``n_envs``; only worthwhile for
      very large per-env populations and few envs.
    * ``"auto"`` (default) — grid only when ``n_agents > 512`` and all env
      z-slabs fit inside ``grid_dim`` without wrapping; brute otherwise.
    """

    def __init__(
        self,
        n_envs: int,
        n_agents: int,
        radius: float,
        max_neighbors: int = 32,
        device: str = "cuda:0",
        dtype=wp.float32,
        grid_dim: int = 128,
        method: str = "auto",
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if method not in ("auto", "grid", "brute"):
            raise ValueError('method must be "auto", "grid", or "brute"')
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.device = device
        self.dtype = dtype
        # Strictly larger than the (padded) query radius -> no cross-env pairs.
        self.z_spacing = 2.0 * radius
        if method == "auto":
            method = "grid" if (n_agents > 512 and 2 * n_envs <= grid_dim) else "brute"
        self.method = method
        self._points: wp.array | None = None
        self._grid: wp.HashGrid | None = None
        if method == "grid":
            self._points = wp.zeros(n_envs * n_agents, dtype=VEC3[dtype], device=device)
            self._grid = wp.HashGrid(grid_dim, grid_dim, grid_dim, device=device, dtype=dtype)
        self.neighbor_idx = wp.zeros(
            (n_envs, n_agents, max_neighbors), dtype=wp.int32, device=device
        )
        self.neighbor_count = wp.zeros((n_envs, n_agents), dtype=wp.int32, device=device)

    def build(self, pos: wp.array) -> None:
        """Refresh the padded lists from positions [n_envs, n_agents] (vec2)."""
        self.query_into(pos, self.neighbor_idx, self.neighbor_count)

    def query_into(self, pos: wp.array, neighbor_idx: wp.array, neighbor_count: wp.array) -> None:
        """Write padded lists into caller-owned buffers using the selected method.

        Launches use ``record_tape=False``: neighbor construction is a discrete,
        non-differentiable pass and must not be replayed by tape adjoints.
        """
        if self.method == "grid":
            self._query_grid_into(pos, neighbor_idx, neighbor_count)
        else:
            self._query_brute_into(pos, neighbor_idx, neighbor_count)

    def _ensure_grid(self) -> None:
        if self._grid is None:
            self._points = wp.zeros(
                self.n_envs * self.n_agents, dtype=VEC3[self.dtype], device=self.device
            )
            self._grid = wp.HashGrid(128, 128, 128, device=self.device, dtype=self.dtype)

    def _query_grid_into(self, pos, neighbor_idx, neighbor_count) -> None:
        self._ensure_grid()
        dim = (self.n_envs, self.n_agents)
        wp.launch(
            _fill_points,
            dim=dim,
            inputs=[pos, self.dtype(self.z_spacing)],
            outputs=[self._points],
            device=self.device,
            record_tape=False,
        )
        self._grid.build(self._points, self.radius)
        wp.launch(
            _query_grid,
            dim=dim,
            inputs=[wp.uint64(self._grid.id), self._points, self.dtype(self.radius)],
            outputs=[neighbor_idx, neighbor_count],
            device=self.device,
            record_tape=False,
        )

    def _query_brute_into(self, pos, neighbor_idx, neighbor_count) -> None:
        wp.launch(
            _brute_force,
            dim=(self.n_envs, self.n_agents),
            inputs=[pos, self.dtype(self.radius)],
            outputs=[neighbor_idx, neighbor_count],
            device=self.device,
            record_tape=False,
        )

    def build_grid(self, pos: wp.array) -> None:
        """Force the hash-grid path into the internal buffers (used by tests)."""
        self._query_grid_into(pos, self.neighbor_idx, self.neighbor_count)

    def build_brute_force(self, pos: wp.array) -> None:
        """Force the O(n_agents^2) reference path into the internal buffers."""
        self._query_brute_into(pos, self.neighbor_idx, self.neighbor_count)

    def torch_views(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-copy (neighbor_idx, neighbor_count) torch tensors (no sync)."""
        return wp.to_torch(self.neighbor_idx), wp.to_torch(self.neighbor_count)

    def edge_index(self) -> torch.Tensor:
        """Radius graph as COO ``[2, E]`` (sender, receiver) with global node ids
        ``env_idx * n_agents + agent_idx``, for GNN libraries.

        Note: extracting E forces one device->host sync; use :meth:`torch_views`
        on the hot path instead.
        """
        idx, cnt = self.torch_views()
        idx = idx.long()
        device = idx.device
        ar = torch.arange(self.max_neighbors, device=device)
        mask = ar.view(1, 1, -1) < cnt.long().unsqueeze(-1)  # [E?, valid slots]
        env_offset = (
            torch.arange(self.n_envs, device=device).view(-1, 1, 1) * self.n_agents
        )
        senders = (idx + env_offset)[mask]
        receivers = (
            (torch.arange(self.n_agents, device=device).view(1, -1, 1) + env_offset)
            .expand(-1, -1, self.max_neighbors)[mask]
        )
        return torch.stack([senders, receivers], dim=0)
