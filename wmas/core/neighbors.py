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

import math
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
    neighbor_true: wp.array2d(dtype=wp.int32),
):
    e, a = wp.tid()
    n_agents = neighbor_idx.shape[1]
    max_neighbors = neighbor_idx.shape[2]
    i = e * n_agents + a
    p = points[i]
    count = wp.int32(0)  # written (capped at max_neighbors)
    total = wp.int32(0)  # true in-radius count (uncapped)
    query = wp.hash_grid_query(grid_id, p, radius * type(radius)(1.001))
    j = wp.int32(0)
    while wp.hash_grid_query_next(query, j):
        if j != i and wp.length(points[j] - p) <= radius:
            total += wp.int32(1)
            if count < max_neighbors:
                neighbor_idx[e, a, count] = j - e * n_agents
                count += wp.int32(1)
    neighbor_count[e, a] = count
    neighbor_true[e, a] = total


@wp.kernel
def _brute_force(
    pos: wp.array2d(dtype=Any),
    radius: Any,
    neighbor_idx: wp.array3d(dtype=wp.int32),
    neighbor_count: wp.array2d(dtype=wp.int32),
    neighbor_true: wp.array2d(dtype=wp.int32),
):
    e, a = wp.tid()
    n_agents = pos.shape[1]
    max_neighbors = neighbor_idx.shape[2]
    p = pos[e, a]
    count = wp.int32(0)  # written (capped at max_neighbors)
    total = wp.int32(0)  # true in-radius count (uncapped)
    for b in range(n_agents):
        if b != a and wp.length(pos[e, b] - p) <= radius:
            total += wp.int32(1)
            if count < max_neighbors:
                neighbor_idx[e, a, count] = b
                count += wp.int32(1)
    neighbor_count[e, a] = count
    neighbor_true[e, a] = total


# --------------------------------------------------------------------------
# Batched uniform-grid backend (radix-sort). Unlike ``wp.HashGrid`` (which wraps
# cell coordinates modulo its dims and so aliases envs into shared cells), this
# gives every env a disjoint block of ``bins*bins`` cells in one linear key
# ``e*bins*bins + cy*bins + cx``, so it stays linear in ``n_envs`` with no
# cross-env aliasing. Positions are placed against a shared origin (the batch's
# min corner) with an adaptive cell size ``max(radius, extent/bins)``; agents
# past ``bins`` cells clamp to the edge cell (still correct — the distance
# filter rejects false candidates, and a cell size >= radius guarantees every
# true within-radius neighbor lands in the queried 3x3 block).


@wp.kernel
def _bounds_init(big: Any, bounds: wp.array(dtype=Any)):
    bounds[0] = big  # min x
    bounds[1] = big  # min y
    bounds[2] = -big  # max x
    bounds[3] = -big  # max y


@wp.kernel
def _bounds_reduce(pos: wp.array2d(dtype=Any), bounds: wp.array(dtype=Any)):
    e, a = wp.tid()
    p = pos[e, a]
    wp.atomic_min(bounds, 0, p[0])
    wp.atomic_min(bounds, 1, p[1])
    wp.atomic_max(bounds, 2, p[0])
    wp.atomic_max(bounds, 3, p[1])


@wp.kernel
def _finalize_grid(
    bounds: wp.array(dtype=Any),
    radius: Any,
    bins: wp.int32,
    origin: wp.array(dtype=Any),
    cell_size: wp.array(dtype=Any),
):
    ext = wp.max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    origin[0] = bounds[0]
    origin[1] = bounds[1]
    cell_size[0] = wp.max(radius, ext / type(radius)(bins))


@wp.kernel
def _compute_keys(
    pos: wp.array2d(dtype=Any),
    origin: wp.array(dtype=Any),
    cell_size: wp.array(dtype=Any),
    bins: wp.int32,
    n_agents: wp.int32,
    keys: wp.array(dtype=wp.int32),
    vals: wp.array(dtype=wp.int32),
):
    e, a = wp.tid()
    p = pos[e, a]
    cs = cell_size[0]
    cx = wp.clamp(wp.int32((p[0] - origin[0]) / cs), 0, bins - 1)
    cy = wp.clamp(wp.int32((p[1] - origin[1]) / cs), 0, bins - 1)
    i = e * n_agents + a
    keys[i] = e * bins * bins + cy * bins + cx
    vals[i] = i


@wp.kernel
def _build_offsets(
    sorted_keys: wp.array(dtype=wp.int32),
    n: wp.int32,
    cell_start: wp.array(dtype=wp.int32),
    cell_end: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    k = sorted_keys[i]
    if i == 0 or sorted_keys[i - 1] != k:
        cell_start[k] = i
    if i == n - 1 or sorted_keys[i + 1] != k:
        cell_end[k] = i + 1


@wp.kernel
def _query_uniform(
    pos: wp.array2d(dtype=Any),
    origin: wp.array(dtype=Any),
    cell_size: wp.array(dtype=Any),
    bins: wp.int32,
    n_agents: wp.int32,
    radius: Any,
    sorted_vals: wp.array(dtype=wp.int32),
    cell_start: wp.array(dtype=wp.int32),
    cell_end: wp.array(dtype=wp.int32),
    neighbor_idx: wp.array3d(dtype=wp.int32),
    neighbor_count: wp.array2d(dtype=wp.int32),
    neighbor_true: wp.array2d(dtype=wp.int32),
):
    e, a = wp.tid()
    max_neighbors = neighbor_idx.shape[2]
    p = pos[e, a]
    cs = cell_size[0]
    cx = wp.clamp(wp.int32((p[0] - origin[0]) / cs), 0, bins - 1)
    cy = wp.clamp(wp.int32((p[1] - origin[1]) / cs), 0, bins - 1)
    count = wp.int32(0)  # written (capped at max_neighbors)
    total = wp.int32(0)  # true in-radius count (uncapped)
    for dy in range(-1, 2):
        ncy = cy + dy
        if ncy >= 0 and ncy < bins:
            for dx in range(-1, 2):
                ncx = cx + dx
                if ncx >= 0 and ncx < bins:
                    ncell = e * bins * bins + ncy * bins + ncx
                    for si in range(cell_start[ncell], cell_end[ncell]):
                        b = sorted_vals[si] - e * n_agents
                        if b != a and wp.length(pos[e, b] - p) <= radius:
                            total += wp.int32(1)
                            if count < max_neighbors:
                                neighbor_idx[e, a, count] = b
                                count += wp.int32(1)
    neighbor_count[e, a] = count
    neighbor_true[e, a] = total


for _T in (wp.float32, wp.float64):
    wp.overload(_fill_points, [wp.array2d(dtype=VEC2[_T]), _T, wp.array(dtype=VEC3[_T])])
    wp.overload(
        _query_grid,
        [
            wp.uint64,
            wp.array(dtype=VEC3[_T]),
            _T,
            wp.array3d(dtype=wp.int32),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=wp.int32),
        ],
    )
    wp.overload(
        _brute_force,
        [
            wp.array2d(dtype=VEC2[_T]),
            _T,
            wp.array3d(dtype=wp.int32),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=wp.int32),
        ],
    )
    wp.overload(_bounds_init, [_T, wp.array(dtype=_T)])
    wp.overload(_bounds_reduce, [wp.array2d(dtype=VEC2[_T]), wp.array(dtype=_T)])
    wp.overload(
        _finalize_grid,
        [wp.array(dtype=_T), _T, wp.int32, wp.array(dtype=_T), wp.array(dtype=_T)],
    )
    wp.overload(
        _compute_keys,
        [
            wp.array2d(dtype=VEC2[_T]),
            wp.array(dtype=_T),
            wp.array(dtype=_T),
            wp.int32,
            wp.int32,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
        ],
    )
    wp.overload(
        _query_uniform,
        [
            wp.array2d(dtype=VEC2[_T]),
            wp.array(dtype=_T),
            wp.array(dtype=_T),
            wp.int32,
            wp.int32,
            _T,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array3d(dtype=wp.int32),
            wp.array2d(dtype=wp.int32),
            wp.array2d(dtype=wp.int32),
        ],
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
    * ``"uniform_grid"`` — a batched radix-sort uniform grid: each env owns a
      disjoint block of ``bins*bins`` cells (no cross-env aliasing), agents are
      sorted by cell key with ``wp.utils.radix_sort_pairs``, and each agent
      queries its 3x3 cell neighborhood. Linear in ``n_envs`` and in agents;
      the intended choice for large per-env populations across many envs.
    * ``"auto"`` (default) — ``uniform_grid`` when ``n_agents > 512``; brute
      otherwise.
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
        uniform_bins: int | None = None,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if method not in ("auto", "grid", "brute", "uniform_grid"):
            raise ValueError('method must be "auto", "grid", "brute", or "uniform_grid"')
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.device = device
        self.dtype = dtype
        # Strictly larger than the (padded) query radius -> no cross-env pairs.
        self.z_spacing = 2.0 * radius
        if method == "auto":
            method = "uniform_grid" if n_agents > 512 else "brute"
        self.method = method
        # Cells per axis for the uniform grid: ~sqrt(n_agents) keeps occupancy
        # near one agent per cell (memory is n_envs * bins^2 int32 offsets).
        self.uniform_bins = uniform_bins or max(4, min(128, int(math.ceil(math.sqrt(n_agents)))))
        self._points: wp.array | None = None
        self._grid: wp.HashGrid | None = None
        self._u_alloc = False  # uniform-grid buffers allocated lazily
        if method == "grid":
            self._points = wp.zeros(n_envs * n_agents, dtype=VEC3[dtype], device=device)
            self._grid = wp.HashGrid(grid_dim, grid_dim, grid_dim, device=device, dtype=dtype)
        elif method == "uniform_grid":
            self._ensure_uniform()
        self.neighbor_idx = wp.zeros(
            (n_envs, n_agents, max_neighbors), dtype=wp.int32, device=device
        )
        self.neighbor_count = wp.zeros((n_envs, n_agents), dtype=wp.int32, device=device)
        # True (uncapped) in-radius count per agent; > max_neighbors means the
        # padded list truncated (collision forces + counts undercount). One
        # grid-owned buffer shared by every query_into call — it is not taped
        # and only read after build()/neighbors(), so sharing is safe.
        self.neighbor_true_count = wp.zeros((n_envs, n_agents), dtype=wp.int32, device=device)
        # Dedupe instrumentation: ``build_count`` is the number of actual neighbor
        # queries launched (any backend, any target buffer) — the ablation reads
        # its per-step delta. ``built_version`` is the ``Stepper.state_version``
        # the internal lists were last built at (stamped by ``World.neighbors``);
        # ``launch_substeps`` reuses them for substep 0 when it still matches.
        self.build_count: int = 0
        self.built_version: int = -1

    def build(self, pos: wp.array) -> None:
        """Refresh the padded lists from positions [n_envs, n_agents] (vec2)."""
        self.query_into(pos, self.neighbor_idx, self.neighbor_count)

    def query_into(self, pos: wp.array, neighbor_idx: wp.array, neighbor_count: wp.array) -> None:
        """Write padded lists into caller-owned buffers using the selected method.

        Launches use ``record_tape=False``: neighbor construction is a discrete,
        non-differentiable pass and must not be replayed by tape adjoints.
        """
        self.build_count += 1
        if self.method == "grid":
            self._query_grid_into(pos, neighbor_idx, neighbor_count)
        elif self.method == "uniform_grid":
            self._query_uniform_into(pos, neighbor_idx, neighbor_count)
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
            outputs=[neighbor_idx, neighbor_count, self.neighbor_true_count],
            device=self.device,
            record_tape=False,
        )

    def _query_brute_into(self, pos, neighbor_idx, neighbor_count) -> None:
        wp.launch(
            _brute_force,
            dim=(self.n_envs, self.n_agents),
            inputs=[pos, self.dtype(self.radius)],
            outputs=[neighbor_idx, neighbor_count, self.neighbor_true_count],
            device=self.device,
            record_tape=False,
        )

    def _ensure_uniform(self) -> None:
        if self._u_alloc:
            return
        n = self.n_envs * self.n_agents
        n_cells = self.n_envs * self.uniform_bins * self.uniform_bins

        def z(size, dt):
            return wp.zeros(size, dtype=dt, device=self.device)

        self._u_bounds = z(4, self.dtype)
        self._u_origin = z(2, self.dtype)
        self._u_cell_size = z(1, self.dtype)
        # radix_sort_pairs needs 2*count-length key/value buffers (ping-pong scratch).
        self._u_keys = z(2 * n, wp.int32)
        self._u_vals = z(2 * n, wp.int32)
        self._u_cell_start = z(n_cells, wp.int32)
        self._u_cell_end = z(n_cells, wp.int32)
        self._u_alloc = True

    def _query_uniform_into(self, pos, neighbor_idx, neighbor_count) -> None:
        self._ensure_uniform()
        n_envs, n_agents = self.n_envs, self.n_agents
        n = n_envs * n_agents
        bins = self.uniform_bins
        dim = (n_envs, n_agents)
        common = dict(device=self.device, record_tape=False)
        # Origin (batch min corner) + adaptive cell size, all device-side (no sync).
        wp.launch(
            _bounds_init, dim=1, inputs=[self.dtype(1e30)], outputs=[self._u_bounds], **common
        )
        wp.launch(_bounds_reduce, dim=dim, inputs=[pos], outputs=[self._u_bounds], **common)
        wp.launch(
            _finalize_grid,
            dim=1,
            inputs=[self._u_bounds, self.dtype(self.radius), bins],
            outputs=[self._u_origin, self._u_cell_size],
            **common,
        )
        wp.launch(
            _compute_keys,
            dim=dim,
            inputs=[pos, self._u_origin, self._u_cell_size, bins, n_agents],
            outputs=[self._u_keys, self._u_vals],
            **common,
        )
        # Offsets are rebuilt from scratch each query; clear stale ranges first.
        self._u_cell_start.zero_()
        self._u_cell_end.zero_()
        wp.utils.radix_sort_pairs(self._u_keys, self._u_vals, n)
        wp.launch(
            _build_offsets,
            dim=n,
            inputs=[self._u_keys, n],
            outputs=[self._u_cell_start, self._u_cell_end],
            **common,
        )
        wp.launch(
            _query_uniform,
            dim=dim,
            inputs=[
                pos,
                self._u_origin,
                self._u_cell_size,
                bins,
                n_agents,
                self.dtype(self.radius),
                self._u_vals,
                self._u_cell_start,
                self._u_cell_end,
            ],
            outputs=[neighbor_idx, neighbor_count, self.neighbor_true_count],
            **common,
        )

    def build_grid(self, pos: wp.array) -> None:
        """Force the hash-grid path into the internal buffers (used by tests)."""
        self._query_grid_into(pos, self.neighbor_idx, self.neighbor_count)

    def build_brute_force(self, pos: wp.array) -> None:
        """Force the O(n_agents^2) reference path into the internal buffers."""
        self._query_brute_into(pos, self.neighbor_idx, self.neighbor_count)

    def build_uniform(self, pos: wp.array) -> None:
        """Force the radix-sort uniform-grid path into the internal buffers."""
        self._query_uniform_into(pos, self.neighbor_idx, self.neighbor_count)

    def torch_views(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-copy (neighbor_idx, neighbor_count) torch tensors (no sync)."""
        return wp.to_torch(self.neighbor_idx), wp.to_torch(self.neighbor_count)

    def true_count_view(self) -> torch.Tensor:
        """Zero-copy true (uncapped) in-radius count ``[n_envs, n_agents]``."""
        return wp.to_torch(self.neighbor_true_count)

    def overflow_view(self) -> torch.Tensor:
        """Zero-copy bool ``[n_envs, n_agents]``: agents whose in-radius count
        exceeded ``max_neighbors`` (their padded list — and thus collision
        forces/counts — is truncated). Valid after :meth:`build`; no host sync.
        """
        return wp.to_torch(self.neighbor_true_count) > wp.to_torch(self.neighbor_count)

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
        env_offset = torch.arange(self.n_envs, device=device).view(-1, 1, 1) * self.n_agents
        senders = (idx + env_offset)[mask]
        receivers = (torch.arange(self.n_agents, device=device).view(1, -1, 1) + env_offset).expand(
            -1, -1, self.max_neighbors
        )[mask]
        return torch.stack([senders, receivers], dim=0)
