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

# --------------------------------------------------------------- every scenario

# What each scenario's reset must freshly draw, beyond the agent spawns: the attribute
# holding per-env state, and whether it is a flag buffer that must come back cleared.
RESET_STATE = {
    "navigation": [("world.goals", "draw")],
    "flocking": [],
    "formation": [("world.goals", "draw")],
    # "reclaim": episode progress that accumulates and must not survive a reset. It is
    # not asserted zero straight after one, because the obs pass that closes a reset
    # immediately re-marks whatever already sits within range of a fresh spawn.
    "discovery": [("targets", "draw"), ("covered", "reclaim")],
    "sampling": [("centers", "draw"), ("consumed", "reclaim")],
    "transport": [
        ("pkg_pos", "draw"),
        ("goal", "draw"),
        ("pkg_vel", "clear"),
        ("pkg_theta", "clear"),
        ("pkg_ang_vel", "clear"),
    ],
    "pusht": [
        ("tee_pos", "draw"),
        ("goal_pos", "draw"),
        ("tee_vel", "clear"),
        ("tee_ang_vel", "clear"),
    ],
}


def _resolve(env, path):
    obj = env.scenario
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _scenario_env(name, *, n_envs=256, n_agents=6, device="cpu", seed=0):
    from swarp.scenarios import SCENARIOS

    scen = SCENARIOS[name](n_agents=n_agents)
    return Environment(scen, n_envs=n_envs, device=device, seed=seed, max_steps=200)


@pytest.mark.parametrize("name", sorted(RESET_STATE))
def test_reset_draws_fresh_state(name):
    """A reset redraws the episode's per-env state and clears its flags."""
    env = _scenario_env(name)
    try:
        env.reset(seed=0)
        before = {p: _resolve(env, p).clone() for p, _ in RESET_STATE[name]}
        env.reset()
        for path, kind in RESET_STATE[name]:
            now = _resolve(env, path)
            if kind == "clear":
                assert not now.any(), f"{name}.{path} not cleared by reset"
            elif kind == "draw":
                assert not torch.equal(before[path], now), f"{name}.{path} not redrawn"
    finally:
        env.close()


@pytest.mark.parametrize("name", sorted(RESET_STATE))
def test_reset_reclaims_accumulated_progress(name):
    """Episode progress must not carry into the next episode.

    Asserted against an episode that actually accumulated some, rather than against zero
    right after a reset: the fused obs pass that closes a reset re-marks whatever is
    already within range of the new spawn, so a couple of flags are legitimately set
    before the first step.
    """
    fields = [p for p, k in RESET_STATE[name] if k == "reclaim"]
    if not fields:
        pytest.skip(f"{name} has no accumulating episode state")
    env = _scenario_env(name, n_envs=64)
    try:
        env.reset(seed=0)
        actions = torch.zeros(64, env.n_agents, env.world.act_dim, device=env.device)
        gen = torch.Generator(device=env.device).manual_seed(0)
        with torch.no_grad():
            for _ in range(40):
                actions.uniform_(-1.0, 1.0, generator=gen)
                env.step(actions)
        earned = {p: _resolve(env, p).sum().item() for p in fields}
        assert any(v > 0 for v in earned.values()), "nothing accumulated — test is vacuous"
        env.reset()
        for path in fields:
            after = _resolve(env, path).sum().item()
            assert after < earned[path], f"{name}.{path} survived the reset"
    finally:
        env.close()


@pytest.mark.parametrize("name", sorted(RESET_STATE))
def test_masked_reset_touches_only_selected_envs(name):
    """The mask is applied in the kernel: unselected envs keep every field they had."""
    env = _scenario_env(name, n_envs=64)
    try:
        env.reset(seed=0)
        paths = ["world.state.pos"] + [p for p, _ in RESET_STATE[name]]
        before = {p: _resolve(env, p).clone() for p in paths}
        mask = torch.zeros(64, dtype=torch.bool, device=env.device)
        mask[::2] = True
        env.reset_at(mask)
        for path in paths:
            now = _resolve(env, path)
            assert torch.equal(now[~mask], before[path][~mask]), (
                f"{name}.{path} changed in an env the mask excluded"
            )
    finally:
        env.close()


@pytest.mark.parametrize("name", sorted(RESET_STATE))
def test_reset_spawns_inside_the_world(name):
    env = _scenario_env(name)
    try:
        env.reset(seed=0)
        scen = env.scenario
        lim = scen.world_size - 2.0 * scen.agent_radius
        assert env.world.state.pos.abs().max().item() <= lim + 1e-5
        assert torch.isfinite(env.world.state.pos).all()
    finally:
        env.close()


@pytest.mark.parametrize("name", sorted(RESET_STATE))
def test_reset_spawns_are_spread_not_collapsed(name):
    """Guards the ``@wp.func`` RNG trap: a helper that takes the state by value returns
    the *same* draw every call, which puts every agent of an env on one point while still
    landing inside the world (see :mod:`swarp.scenarios.reset_kernels`)."""
    env = _scenario_env(name, n_envs=128, n_agents=6)
    try:
        env.reset(seed=0)
        pos = env.world.state.pos
        assert pos.std().item() > 0.05, f"{name} spawns are degenerate"
        spread = (pos - pos.mean(dim=1, keepdim=True)).norm(dim=-1).max().item()
        assert spread > 1e-3, f"{name} collapsed every agent in an env onto one point"
    finally:
        env.close()


@pytest.mark.parametrize("name", sorted(RESET_STATE))
def test_reset_same_seed_reproduces(name):
    def draw(seed):
        env = _scenario_env(name, seed=seed)
        try:
            env.reset(seed=seed)
            return env.world.state.pos.clone()
        finally:
            env.close()

    assert torch.equal(draw(3), draw(3))
    assert not torch.equal(draw(3), draw(4))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(RESET_STATE))
def test_reset_is_host_sync_free_for_every_scenario(device, name):
    env = _scenario_env(name, n_envs=64, device=device)
    try:
        env.reset(seed=0)  # warm up (buffers, kernel load)
        mask = torch.zeros(64, dtype=torch.bool, device=device)
        mask[::2] = True
        with torch.no_grad(), forbid_host_transfers():
            env.scenario.reset_world(mask)
    finally:
        env.close()


def test_pusht_reset_clears_the_tee():
    """No agent may start inside the T's bounding disk: with a stiff contact_k a spawn
    overlap is a violent ejection, so the kernel pushes them out to the rim."""
    env = _scenario_env("pusht", n_envs=512, n_agents=4)
    try:
        env.reset(seed=0)
        scen = env.scenario
        d = (env.world.state.pos - scen.tee_pos.unsqueeze(1)).norm(dim=-1)
        clear = scen.tee_radius + 2.0 * scen.agent_radius
        lim = scen.world_size - 2.0 * scen.agent_radius
        inside = d < clear - 1e-4
        # The push-out is followed by a clamp back into the square (as in the torch reset
        # this replaces), so the rim is missed only for a T sitting near the wall — and
        # then the agent must be *at* that wall. Anything else is a push-out that failed.
        at_wall = (env.world.state.pos.abs() >= lim - 1e-4).any(dim=-1)
        assert bool((~inside | at_wall).all()), "an agent spawned inside the T, off-wall"
        assert inside.float().mean().item() < 0.05, "the clamp should be the rare case"
    finally:
        env.close()
