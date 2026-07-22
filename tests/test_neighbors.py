"""Hash-grid neighbor search must exactly match the brute-force O(n^2) reference."""

import numpy as np
import pytest
import torch
import warp as wp

from wmas.core.neighbors import NeighborGrid

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def neighbor_sets(idx: np.ndarray, cnt: np.ndarray) -> list[list[set]]:
    """[n_envs][n_agents] -> set of neighbor indices."""
    n_envs, n_agents, _ = idx.shape
    return [
        [set(idx[e, a, : cnt[e, a]].tolist()) for a in range(n_agents)] for e in range(n_envs)
    ]


def make_positions(rng, n_envs, n_agents, extent=2.0):
    return (rng.random((n_envs, n_agents, 2)) * 2.0 - 1.0) * extent


def build_both(pos_np, radius, max_neighbors, device, dtype=wp.float32):
    n_envs, n_agents, _ = pos_np.shape
    npdt = np.float64 if dtype == wp.float64 else np.float32
    vec2 = wp.vec2d if dtype == wp.float64 else wp.vec2f
    pos = wp.array(pos_np.astype(npdt), dtype=vec2, device=device)

    grid = NeighborGrid(
        n_envs, n_agents, radius=radius, max_neighbors=max_neighbors,
        device=device, dtype=dtype,
    )
    grid.build(pos)
    g_idx, g_cnt = grid.neighbor_idx.numpy().copy(), grid.neighbor_count.numpy().copy()

    grid.build_brute_force(pos)
    b_idx, b_cnt = grid.neighbor_idx.numpy().copy(), grid.neighbor_count.numpy().copy()
    return (g_idx, g_cnt), (b_idx, b_cnt)


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
