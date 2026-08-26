"""Hash-grid neighbor search must exactly match the brute-force O(n^2) reference."""

import numpy as np
import pytest
import warp as wp
from conftest import DEVICES

from swarp.core.neighbors import NeighborGrid


def neighbor_sets(idx: np.ndarray, cnt: np.ndarray) -> list[list[set]]:
    """[n_envs][n_agents] -> set of neighbor indices."""
    n_envs, n_agents, _ = idx.shape
    return [[set(idx[e, a, : cnt[e, a]].tolist()) for a in range(n_agents)] for e in range(n_envs)]


def make_positions(rng, n_envs, n_agents, extent=2.0):
    return (rng.random((n_envs, n_agents, 2)) * 2.0 - 1.0) * extent


def build_both(pos_np, radius, max_neighbors, device, dtype=wp.float32):
    n_envs, n_agents, _ = pos_np.shape
    npdt = np.float64 if dtype == wp.float64 else np.float32
    vec2 = wp.vec2d if dtype == wp.float64 else wp.vec2f
    pos = wp.array(pos_np.astype(npdt), dtype=vec2, device=device)

    grid = NeighborGrid(
        n_envs,
        n_agents,
        radius=radius,
        max_neighbors=max_neighbors,
        device=device,
        dtype=dtype,
    )
    grid.build_grid(pos)  # force the hash-grid path (build() may auto-pick brute)
    g_idx, g_cnt = grid.neighbor_idx.numpy().copy(), grid.neighbor_count.numpy().copy()

    grid.build_brute_force(pos)
    b_idx, b_cnt = grid.neighbor_idx.numpy().copy(), grid.neighbor_count.numpy().copy()
    return (g_idx, g_cnt), (b_idx, b_cnt)


def build_uniform_and_brute(pos_np, radius, max_neighbors, device, dtype=wp.float32):
    """Build the radix-sort uniform grid and the brute-force reference on one grid."""
    n_envs, n_agents, _ = pos_np.shape
    npdt = np.float64 if dtype == wp.float64 else np.float32
    vec2 = wp.vec2d if dtype == wp.float64 else wp.vec2f
    pos = wp.array(pos_np.astype(npdt), dtype=vec2, device=device)

    grid = NeighborGrid(
        n_envs,
        n_agents,
        radius=radius,
        max_neighbors=max_neighbors,
        device=device,
        dtype=dtype,
        method="uniform_grid",
    )
    grid.build_uniform(pos)
    u_idx, u_cnt = grid.neighbor_idx.numpy().copy(), grid.neighbor_count.numpy().copy()
    u_true = grid.neighbor_true_count.numpy().copy()
    grid.build_brute_force(pos)
    b_idx, b_cnt = grid.neighbor_idx.numpy().copy(), grid.neighbor_count.numpy().copy()
    b_true = grid.neighbor_true_count.numpy().copy()
    return (u_idx, u_cnt, u_true), (b_idx, b_cnt, b_true)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_envs,n_agents", [(1, 8), (4, 32), (2, 128)])
@pytest.mark.parametrize("radius", [0.15, 0.5])
def test_grid_matches_brute_force(device, n_envs, n_agents, radius):
    rng = np.random.default_rng(42)
    pos_np = make_positions(rng, n_envs, n_agents)
    (g_idx, g_cnt), (b_idx, b_cnt) = build_both(pos_np, radius, 64, device)
    np.testing.assert_array_equal(g_cnt, b_cnt)
    assert neighbor_sets(g_idx, g_cnt) == neighbor_sets(b_idx, b_cnt)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_envs,n_agents", [(1, 8), (4, 32), (2, 128), (2, 600)])
@pytest.mark.parametrize("radius", [0.15, 0.5])
def test_uniform_grid_matches_brute_force(device, n_envs, n_agents, radius):
    """The radix-sort uniform grid agrees with brute force: byte-identical counts
    and true counts, identical neighbor sets (sparse config, no truncation)."""
    rng = np.random.default_rng(42)
    pos_np = make_positions(rng, n_envs, n_agents)
    (u_idx, u_cnt, u_true), (b_idx, b_cnt, b_true) = build_uniform_and_brute(
        pos_np, radius, 128, device
    )
    np.testing.assert_array_equal(u_cnt, b_cnt)
    np.testing.assert_array_equal(u_true, b_true)
    assert not (b_true > 128).any(), "config truncated; widen max_neighbors"
    assert neighbor_sets(u_idx, u_cnt) == neighbor_sets(b_idx, b_cnt)


@pytest.mark.parametrize("device", DEVICES)
def test_uniform_grid_no_cross_env_leaks(device):
    """Identical dense layouts per env give identical, env-local neighbor sets."""
    rng = np.random.default_rng(7)
    single = make_positions(rng, 1, 16, extent=0.3)
    pos_np = np.repeat(single, 8, axis=0)
    (u_idx, u_cnt, u_true), (b_idx, b_cnt, _) = build_uniform_and_brute(pos_np, 0.4, 32, device)
    np.testing.assert_array_equal(u_cnt, b_cnt)
    sets = neighbor_sets(u_idx, u_cnt)
    for e in range(1, 8):
        assert sets[e] == sets[0]
    assert u_idx.min() >= 0 and u_idx.max() < 16


@pytest.mark.parametrize("device", DEVICES)
def test_uniform_grid_float64(device):
    rng = np.random.default_rng(3)
    pos_np = make_positions(rng, 2, 24)
    (u_idx, u_cnt, _), (b_idx, b_cnt, _) = build_uniform_and_brute(
        pos_np, 0.3, 32, device, dtype=wp.float64
    )
    np.testing.assert_array_equal(u_cnt, b_cnt)
    assert neighbor_sets(u_idx, u_cnt) == neighbor_sets(b_idx, b_cnt)


@pytest.mark.parametrize("device", DEVICES)
def test_uniform_grid_rebuild_deterministic(device):
    rng = np.random.default_rng(11)
    pos_np = make_positions(rng, 4, 64, extent=0.5)
    a = build_uniform_and_brute(pos_np, 0.25, 32, device)[0]
    b = build_uniform_and_brute(pos_np, 0.25, 32, device)[0]
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])


def test_auto_selects_uniform_grid():
    """auto picks the uniform grid for large per-env populations, brute otherwise.

    ``device="cpu"`` explicitly: the heuristic is arithmetic on ``n_agents`` and has
    nothing to do with the device, but NeighborGrid defaults to ``"cuda:0"`` and
    allocates its buffers in ``__init__``, so leaving it out makes a pure-logic test
    fail on a GPU-less machine.
    """
    assert NeighborGrid(4, 600, radius=0.1, method="auto", device="cpu").method == "uniform_grid"
    assert NeighborGrid(4, 512, radius=0.1, method="auto", device="cpu").method == "brute"


@pytest.mark.parametrize("device", DEVICES)
def test_no_cross_env_leaks(device):
    """Identical layouts in every env: neighbor sets must be identical per env."""
    rng = np.random.default_rng(7)
    single = make_positions(rng, 1, 16, extent=0.3)  # dense -> many neighbors
    pos_np = np.repeat(single, 8, axis=0)
    (g_idx, g_cnt), (b_idx, b_cnt) = build_both(pos_np, 0.4, 32, device)
    np.testing.assert_array_equal(g_cnt, b_cnt)
    sets = neighbor_sets(g_idx, g_cnt)
    for e in range(1, 8):
        assert sets[e] == sets[0]
    # all neighbor indices are valid agent indices (env-local)
    assert g_idx.min() >= 0 and g_idx.max() < 16


@pytest.mark.parametrize("device", DEVICES)
def test_max_neighbors_overflow(device):
    """A 5-agent cluster with max_neighbors=2: capped count, valid true neighbors."""
    pos_np = np.zeros((1, 5, 2))
    pos_np[0, :, 0] = np.linspace(0.0, 0.04, 5)  # all within radius of each other
    (g_idx, g_cnt), _ = build_both(pos_np, 0.5, 2, device)
    np.testing.assert_array_equal(g_cnt, np.full((1, 5), 2))
    for a in range(5):
        picked = set(g_idx[0, a, :2].tolist())
        assert len(picked) == 2 and a not in picked and picked <= set(range(5))


_BUILD_METHOD = {
    "brute": "build_brute_force",
    "grid": "build_grid",
    "uniform_grid": "build_uniform",
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["brute", "grid", "uniform_grid"])
def test_overflow_flag(device, method):
    """The overflow flag fires exactly where the true count exceeds the cap."""
    pos_np = np.zeros((1, 5, 2))
    pos_np[0, :, 0] = np.linspace(0.0, 0.04, 5)  # 5-agent cluster, 4 neighbors each
    vec2 = wp.vec2f
    pos = wp.array(pos_np.astype(np.float32), dtype=vec2, device=device)

    grid = NeighborGrid(1, 5, radius=0.5, max_neighbors=2, device=device)
    getattr(grid, _BUILD_METHOD[method])(pos)
    # every agent truly sees 4 neighbors, capped to 2 -> overflow everywhere
    np.testing.assert_array_equal(grid.true_count_view().cpu().numpy(), np.full((1, 5), 4))
    assert grid.overflow_view().all()

    grid_ok = NeighborGrid(1, 5, radius=0.5, max_neighbors=8, device=device)
    getattr(grid_ok, _BUILD_METHOD[method])(pos)
    assert not grid_ok.overflow_view().any()  # room for all 4 -> no overflow


def test_float64_grid():
    rng = np.random.default_rng(3)
    pos_np = make_positions(rng, 2, 24)
    (g_idx, g_cnt), (b_idx, b_cnt) = build_both(pos_np, 0.3, 32, "cpu", dtype=wp.float64)
    np.testing.assert_array_equal(g_cnt, b_cnt)
    assert neighbor_sets(g_idx, g_cnt) == neighbor_sets(b_idx, b_cnt)


@pytest.mark.parametrize("device", DEVICES)
def test_grid_rebuild_deterministic(device):
    rng = np.random.default_rng(11)
    pos_np = make_positions(rng, 4, 64, extent=0.5)
    a = build_both(pos_np, 0.25, 32, device)[0]
    b = build_both(pos_np, 0.25, 32, device)[0]
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize("device", DEVICES)
def test_edge_index(device):
    """COO radius graph agrees with the padded representation."""
    rng = np.random.default_rng(5)
    n_envs, n_agents = 3, 16
    pos_np = make_positions(rng, n_envs, n_agents, extent=0.4)
    vec2 = wp.vec2f
    pos = wp.array(pos_np.astype(np.float32), dtype=vec2, device=device)
    grid = NeighborGrid(n_envs, n_agents, radius=0.3, max_neighbors=32, device=device)
    grid.build(pos)
    edges = grid.edge_index()  # [2, E], global node ids: e * n_agents + a

    idx, cnt = grid.neighbor_idx.numpy(), grid.neighbor_count.numpy()
    expected = set()
    for e in range(n_envs):
        for a in range(n_agents):
            for k in range(cnt[e, a]):
                expected.add((e * n_agents + idx[e, a, k], e * n_agents + a))
    got = set(map(tuple, edges.T.cpu().numpy().tolist()))
    assert got == expected
    assert edges.device.type == ("cuda" if device.startswith("cuda") else "cpu")


@pytest.mark.parametrize("device", DEVICES)
def test_lazy_hash_grid_honours_grid_dim(device, monkeypatch):
    """``build_grid`` on a non-``"grid"`` backend must not fall back to a hardcoded 128.

    The lazy allocation in ``_ensure_grid`` is the only way ``WorldConfig.grid_dim``
    can be silently discarded, because it is reached from ``build_grid`` regardless of
    the configured method.
    """
    dims = []
    real = wp.HashGrid

    def spy(dim_x, dim_y, dim_z, **kw):  # wp.HashGrid keeps no dim attributes
        dims.append((dim_x, dim_y, dim_z))
        return real(dim_x, dim_y, dim_z, **kw)

    monkeypatch.setattr(wp, "HashGrid", spy)
    rng = np.random.default_rng(0)
    pos_np = make_positions(rng, 2, 6)
    grid = NeighborGrid(2, 6, radius=0.5, device=device, method="brute", grid_dim=64)
    assert grid._grid is None and not dims  # nothing allocated for the brute backend
    pos = wp.array(pos_np.astype(np.float32), dtype=wp.vec2f, device=device)
    grid.build_grid(pos)
    assert dims == [(64, 64, 64)]


@pytest.mark.parametrize("device", DEVICES)
def test_lazy_hash_grid_agrees_with_brute_force_at_a_small_grid_dim(device):
    """...and the smaller grid still produces the same neighbor sets."""
    rng = np.random.default_rng(1)
    pos_np = make_positions(rng, 3, 12)
    pos = wp.array(pos_np.astype(np.float32), dtype=wp.vec2f, device=device)
    ref = NeighborGrid(3, 12, radius=0.6, device=device, method="brute")
    ref.build_brute_force(pos)
    small = NeighborGrid(3, 12, radius=0.6, device=device, method="brute", grid_dim=32)
    small.build_grid(pos)
    a = neighbor_sets(ref.neighbor_idx.numpy(), ref.neighbor_count.numpy())
    b = neighbor_sets(small.neighbor_idx.numpy(), small.neighbor_count.numpy())
    assert a == b


def test_over_fine_uniform_grid_warns():
    """``bins**2`` far above ``n_agents`` means each query memsets more cells than it searches."""
    with pytest.warns(RuntimeWarning, match="cells per env"):
        NeighborGrid(2, 8, radius=0.5, device="cpu", method="uniform_grid", uniform_bins=128)


def test_default_uniform_bins_is_quiet():
    """The ~sqrt(n_agents) heuristic is the shape the warning is calibrated against."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NeighborGrid(2, 8, radius=0.5, device="cpu", method="uniform_grid")


def test_uniform_bins_must_be_positive():
    with pytest.raises(ValueError, match="uniform_bins"):
        NeighborGrid(2, 8, radius=0.5, device="cpu", method="uniform_grid", uniform_bins=0)

