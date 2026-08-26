"""Navigation's device-side reset kernel: the invariants, not the values.

``reset_world`` no longer samples in torch. It launches one masked Warp kernel that
writes spawns, goals, headings and zeroed velocities, because under ``auto_reset`` the
reset runs on *every* step for the whole batch -- there is no host-side "is anything
done?" gate that could skip it -- and the torch chain it replaced (two batched
``argsort``s, four ``sample_uniform``s, the ``torch.where`` blends) was 86% of the step
time at 16384x16.

``NavigationScenario._sample_separated`` stays the torch reference for the same draw and
keeps its own tests. The two deliberately share no code and their RNG streams are
independent, so the kernel is pinned against the reference's *invariants* rather than its
numbers -- exactly how the fused obs/reward kernels are treated.
"""

import contextlib
import math
from unittest import mock

import pytest
import torch
from conftest import DEVICES

from swarp import Environment, NavigationScenario


def _raise(*args, **kwargs):
    raise AssertionError("device->host transfer in the reset")


@contextlib.contextmanager
def forbid_host_transfers():
    with (
        mock.patch.object(torch.Tensor, "cpu", _raise),
        mock.patch.object(torch.Tensor, "item", _raise),
        mock.patch.object(torch.Tensor, "numpy", _raise),
        mock.patch.object(torch.Tensor, "tolist", _raise),
        mock.patch.object(torch.Tensor, "__bool__", _raise),
    ):
        yield


def _env(n_agents, *, n_envs=256, agent_radius=0.05, world_size=1.0, device="cpu", seed=0):
    scen = NavigationScenario(
        n_agents=n_agents, agent_radius=agent_radius, world_size=world_size
    )
    return Environment(scen, n_envs=n_envs, device=device, seed=seed)


def _min_pairwise(pos):
    n = pos.shape[1]
    d = torch.cdist(pos, pos) + torch.eye(n, device=pos.device, dtype=pos.dtype) * 1e9
    return d.min().item()


@pytest.mark.parametrize("n_agents", [2, 4, 8, 16, 32, 64])
def test_kernel_separation_is_guaranteed(n_agents):
    """The same "not usually, always" guarantee the torch reference gives."""
    env = _env(n_agents)
    try:
        env.reset(seed=0)
        assert _min_pairwise(env.world.state.pos) >= env.scenario.min_spawn_separation - 1e-6
    finally:
        env.close()


@pytest.mark.parametrize("n_agents", [2, 8, 64])
def test_kernel_spawns_and_goals_stay_inside_the_square(n_agents):
    env = _env(n_agents)
    try:
        env.reset(seed=0)
        scen = env.scenario
        lim = scen.world_size - 2.0 * scen.agent_radius
        assert env.world.state.pos.abs().max().item() <= lim + 1e-6
        assert env.world.goals.abs().max().item() <= lim + 1e-6
    finally:
        env.close()


def test_kernel_spawns_are_not_degenerate():
    """Jitter is real, and spawns and goals are independent draws."""
    env = _env(8)
    try:
        env.reset(seed=0)
        pos, goals = env.world.state.pos, env.world.goals
        assert not torch.allclose(pos, goals)
        assert pos.std().item() > 0.1
        assert env.world.state.vel.abs().max().item() == 0.0
    finally:
        env.close()


def test_kernel_draws_many_distinct_cell_subsets():
    """The cell subset must be a *uniform* k-subset, not a structured family.

    A cheaper sampler that picks cells by an arithmetic progression (random base and
    stride) still separates, still fits the square, and still jitters -- every other test
    here passes -- while collapsing the reachable spawn layouts from C(25,16) to a few
    hundred. That is invisible in a reward curve and shows up much later as a
    generalization failure, so it gets its own test.
    """
    n_envs, n_agents = 4096, 16
    env = _env(n_agents, n_envs=n_envs)
    try:
        env.reset(seed=0)
        scen = env.scenario
        lim, cell, _, grid, n_cells, stratified = scen._reset_grid()
        assert stratified, "this test only means something on the stratified path"
        idx = ((env.world.state.pos + lim) / cell).floor().clamp(0, grid - 1).long()
        cells = (idx[..., 0] * grid + idx[..., 1]).sort(dim=-1).values
        distinct = len(torch.unique(cells, dim=0))
        # Uniform k-subsets of 25 cells: essentially every env draws its own.
        assert distinct > n_envs // 2, (
            f"only {distinct} distinct cell subsets across {n_envs} envs "
            f"(C({n_cells},{n_agents}) = {math.comb(n_cells, n_agents)}) — "
            "the sampler has collapsed onto a structured family"
        )
    finally:
        env.close()


def test_masked_reset_leaves_other_envs_untouched():
    env = _env(8, n_envs=64)
    try:
        env.reset(seed=0)
        before = env.world.state.pos.clone()
        mask = torch.zeros(64, dtype=torch.bool, device=env.device)
        mask[::2] = True
        env.reset_at(mask)
        after = env.world.state.pos
        assert torch.equal(after[~mask], before[~mask])
        assert not torch.equal(after[mask], before[mask])
    finally:
        env.close()


def test_same_seed_reproduces_the_reset():
    def spawns(seed):
        env = _env(8, seed=seed)
        try:
            env.reset(seed=seed)
            return env.world.state.pos.clone(), env.world.goals.clone()
        finally:
            env.close()

    a_pos, a_goal = spawns(7)
    b_pos, b_goal = spawns(7)
    c_pos, _ = spawns(8)
    assert torch.equal(a_pos, b_pos)
    assert torch.equal(a_goal, b_goal)
    assert not torch.equal(a_pos, c_pos)


def test_repeated_resets_keep_drawing_fresh_layouts():
    """The kernel seed stream advances: two resets in a row must not coincide."""
    env = _env(8)
    try:
        env.reset(seed=0)
        first = env.world.state.pos.clone()
        env.reset()
        assert not torch.equal(first, env.world.state.pos)
    finally:
        env.close()


def test_infeasible_separation_falls_back_to_uniform():
    """At the packing limit the kernel draws uniformly rather than pinning to centres."""
    env = _env(8, agent_radius=0.2)
    try:
        env.reset(seed=0)
        scen = env.scenario
        assert not scen._reset_grid()[-1]  # fell back, as documented
        pos = env.world.state.pos
        assert torch.isfinite(pos).all()
        assert pos.abs().max().item() <= scen.world_size - 2.0 * scen.agent_radius + 1e-6
    finally:
        env.close()


@pytest.mark.parametrize("device", DEVICES)
def test_reset_is_host_sync_free(device):
    """It runs inside a masked reset on the step loop, where a round-trip stalls the batch."""
    env = _env(8, n_envs=64, device=device)
    try:
        env.reset(seed=0)  # warm up (buffer allocation, kernel load)
        mask = torch.zeros(64, dtype=torch.bool, device=device)
        mask[::2] = True
        with torch.no_grad(), forbid_host_transfers():
            env.scenario.reset_world(mask)
    finally:
        env.close()


@pytest.mark.parametrize("device", DEVICES)
def test_float64_world_resets_in_float64(device):
    """``_as`` widens the float32 draw, so a float64 world gets float64 spawns."""
    scen = NavigationScenario(n_agents=8)
    env = Environment(scen, n_envs=64, device=device, seed=0, dtype=torch.float64)
    try:
        env.reset(seed=0)
        assert env.world.state.pos.dtype == torch.float64
        assert _min_pairwise(env.world.state.pos) >= scen.min_spawn_separation - 1e-6
    finally:
        env.close()
