"""Give-way scenario: torch-path unit suite.

Every test here builds its :class:`~swarp.core.environment.Environment` with
``fused=False`` on purpose: this suite exercises the *torch reference* path (``_refresh``,
``_build_obs``, ``_corner_sdf``), poking scenario internals directly and reading the cache
back. The fused Warp kernels are being iterated on independently, and their parity against
this torch oracle is what ``test_giveway_fused.py`` is for.
"""

import pytest
import torch
from conftest import DEVICES

from swarp import Environment
from swarp.scenarios.giveway import GiveWayScenario


def _env(device, n_envs=4, n_agents=4, **kw):
    return Environment(
        GiveWayScenario(n_agents=n_agents, **kw),
        n_envs=n_envs,
        device=device,
        dt=0.05,
        seed=1,
        fused=False,
    )


# --------------------------------------------------------------------------------------
# 1. The flagship symmetry test.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_head_on_pair_sees_identical_obs_except_the_token(device):
    """A head-on pair's observations are exact 180-degree rotations of each other.

    Agent 1 is agent 0 reflected through the origin -- position, velocity, travel axis
    and goal all negated -- which is exactly what "meeting head-on, mirrored" means. Own
    frame ``(u, n)`` is negated too, so every rotation-invariant own feature (indices
    0..9, everything but the axis and the token) comes out bit-for-bit equal, and the same
    holds for the one neighbour slot's geometry (indices 13..19 and the ``valid`` flag at
    21) because the relative vectors and the neighbour's own axis are *also* mirrored.

    The only features that must differ are the ones that carry the asymmetry on purpose:
    the priority token (10), the world-frame axis (11, 12) -- which must be exact
    negations of one another, since a head-on pair's axes are exact opposites -- and the
    per-slot priority difference ``dprio`` (13 + 9*j + 7). This is the geometric argument
    the module docstring makes for why the token has to exist: without it a shared-weight
    policy cannot tell these two robots apart.

    The probe point is deliberately off both medial axes (``px == 0``, ``py == 0``,
    ``|qx| == |qy|``): the folded corner SDF has a genuine gradient ridge there (see
    ``test_sdf_normal_points_to_free_space``), and exactly on it the two one-sided limits
    for the gradient's tie-broken sign do *not* mirror even though the SDF value itself
    does -- landing an agent on the axis would make this test fail on the gradient
    features for a reason that has nothing to do with the property being pinned here.
    """
    scen = GiveWayScenario(n_agents=2, use_priority=True)
    env = Environment(scen, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset(seed=0)

    dt = {"device": device, "dtype": env.dtype}
    p0 = torch.tensor([-0.3, 0.02], **dt)
    v0 = torch.tensor([0.15, 0.05], **dt)
    u0 = torch.tensor([1.0, 0.0], **dt)
    g0 = torch.tensor([0.3, -0.1], **dt)

    pos = torch.stack([p0, -p0]).unsqueeze(0)
    vel = torch.stack([v0, -v0]).unsqueeze(0)
    axis = torch.stack([u0, -u0]).unsqueeze(0)
    goals = torch.stack([g0, -g0]).unsqueeze(0)
    prio = torch.tensor([[-1.0, 1.0]], **dt)

    env.world.write_state(None, pos=pos, vel=vel)
    scen._axis.copy_(axis)  # documented internal write: forcing an exact mirror config
    env.world.goals.copy_(goals)
    scen._prio.copy_(prio)
    scen.post_step()

    obs = scen.observations()
    assert obs.shape == (1, 2, scen.obs_dim)
    own0, own1 = obs[0, 0], obs[0, 1]

    excluded = {10, 11, 12}
    k = scen._k_obs
    assert k == 1
    for j in range(k):
        excluded.add(13 + 9 * j + 7)

    shared = [i for i in range(obs.shape[-1]) if i not in excluded]
    torch.testing.assert_close(own0[shared], own1[shared], rtol=1e-5, atol=1e-5)

    # The axis is the mirror's defining feature: exact negation. Index 12 (u_y) happens to
    # be exactly 0 for both agents with this particular axis choice (u0 = +e_x), so it is
    # excluded from the "must actually differ" check below but still covered by the
    # negation check.
    torch.testing.assert_close(own0[11:13], -own1[11:13], rtol=1e-6, atol=1e-6)
    for i in excluded - {12}:
        assert (own0[i] - own1[i]).abs().item() > 1e-3, f"index {i} should differ"


# --------------------------------------------------------------------------------------
# 2. The scripted priority policy -- the gate on the observation being sufficient.
# --------------------------------------------------------------------------------------


def _rank_scheduled_policy(obs: torch.Tensor, t: int, r: float, n_agents: int, k: int):
    """A hand-written controller that reads only ``obs`` (plus a step counter).

    Give-way with four agents occupies all four arms at once, so *two* head-on pairs must
    each duck-and-cross -- a purely reactive "yield when something is coming" rule
    deadlocks or bounces the yielding robot backwards indefinitely (measured while
    developing this test: solve rate 0.0, with the yielding robot's distance-to-goal
    diverging as the non-yielding robot simply plows through it, because a lower-priority
    robot that only *reacts* to a conflict never gets off the shared line before the
    higher-priority one arrives -- the escape route only exists within ``|along| < c`` of
    the junction, and reaching it takes as long as reaching the collision itself when both
    robots start equidistant and desynced by zero).

    So this is a coarse **schedule** instead: each robot reads its own priority token
    (index 10) and its head-on partner's relative token (``dprio`` on the slot whose
    ``ndir_along`` is closest to -1) to figure out (a) which of the two global head-on
    pairs it belongs to and (b) whether it is the one that goes first or the one that
    ducks into the perpendicular bay, then runs a small fixed sequence of frame-relative
    manoeuvres keyed off the step count. It still reads only the observation for every
    *decision* (which neighbour is my partner, am I first, where am I along my own arm) --
    the step counter is controller-internal state, not privileged access to the world.

    Frame-relative actions are converted to world-frame velocity via the axis carried at
    obs indices 11/12 (``u``) and its perpendicular ``n`` -- this is exactly the
    "rotate out of the frame" step the module docstring says the axis feature exists for.
    """
    own = obs[..., :13]
    along = own[..., 0]
    lat_r = own[..., 1]
    u = own[..., 11:13]
    n = torch.stack([-u[..., 1], u[..., 0]], dim=-1)
    own_tok = own[..., 10]

    ne, na = obs.shape[0], obs.shape[1]
    nb = obs[..., 13:].reshape(ne, na, k, 9)
    ndir_along, dprio = nb[..., 4], nb[..., 7]
    partner_idx = ndir_along.argmin(dim=-1)
    partner_dprio = torch.gather(dprio, -1, partner_idx.unsqueeze(-1)).squeeze(-1)
    i_go_first = partner_dprio < 0
    group0 = (own_tok >= 1 - 1e-3) | (own_tok + partner_dprio >= 1 - 1e-3)

    def drive_to_goal():
        return torch.ones_like(along), -0.5 * lat_r.clamp(-1.0, 1.0)

    def drive_to_bay():
        not_there = along < 0.0
        a_along = torch.where(not_there, torch.ones_like(along), torch.zeros_like(along))
        reached = (lat_r * r) >= 0.20
        a_lat = torch.where(reached, torch.zeros_like(lat_r), torch.ones_like(lat_r))
        return a_along, a_lat

    def hold_at_mouth():
        not_there = along < -0.16
        a_along = torch.where(not_there, torch.ones_like(along), torch.zeros_like(along))
        return a_along, torch.zeros_like(along)

    def hold():
        return torch.zeros_like(along), torch.zeros_like(along)

    a_along = torch.zeros_like(along)
    a_lat = torch.zeros_like(along)

    m = group0 & ~i_go_first
    ta, tl = drive_to_bay() if t < 30 else (hold() if t < 80 else drive_to_goal())
    a_along, a_lat = torch.where(m, ta, a_along), torch.where(m, tl, a_lat)

    m = group0 & i_go_first
    ta, tl = hold_at_mouth() if t < 30 else (drive_to_goal() if t < 80 else hold())
    a_along, a_lat = torch.where(m, ta, a_along), torch.where(m, tl, a_lat)

    tt = t - 130
    m1 = ~group0 & ~i_go_first
    m2 = ~group0 & i_go_first
    if tt < 0:
        ta1 = tl1 = ta2 = tl2 = torch.zeros_like(along)
    else:
        ta1, tl1 = drive_to_bay() if tt < 30 else (hold() if tt < 80 else drive_to_goal())
        ta2, tl2 = hold_at_mouth() if tt < 30 else (drive_to_goal() if tt < 80 else hold())
    a_along, a_lat = torch.where(m1, ta1, a_along), torch.where(m1, tl1, a_lat)
    a_along, a_lat = torch.where(m2, ta2, a_along), torch.where(m2, tl2, a_lat)

    return (a_along.unsqueeze(-1) * u + a_lat.unsqueeze(-1) * n).clamp(-1.0, 1.0)


def test_scripted_priority_policy_solves_it():
    """Solvability gate: a hand-written controller reading only the observation must
    clear the four-way junction from the hardest (fully synchronized) spawn pattern.

    ``set_spawn_desync(0.0)`` is deliberately the worst case named in the brief: every
    robot starts equidistant from the junction and moving at the same speed, so all four
    arrive at the crossing at once and both head-on pairs must duck-and-cross
    simultaneously. If the observation set did not carry enough information to solve
    this (no priority token, no world-frame axis, no per-neighbour conflict type), no
    policy -- scripted or learned -- could do better than chance here.

    Measured: 1.0 solve rate at 500 envs / seed 1 and at 256 envs / seed 3, and 1.0 at 64
    envs / seed 0 during development. The threshold below is left at the 0.9 the brief
    specifies even though the measured rate is exactly 1.0, since the point of the test is
    the gate, not a tight bound on this particular scripted policy.
    """
    scen = GiveWayScenario(n_agents=4, use_priority=True)
    scen.set_spawn_desync(0.0)
    n_envs = 64
    env = Environment(
        scen,
        n_envs=n_envs,
        device="cpu",
        dt=0.05,
        substeps=16,
        seed=0,
        max_steps=None,
        auto_reset=False,
        fused=False,
    )
    obs = env.reset(seed=0)
    k = scen._k_obs
    r = scen.agent_radius
    ever_solved = torch.zeros(n_envs, dtype=torch.bool)
    with torch.no_grad():
        for t in range(300):
            action = _rank_scheduled_policy(obs, t, r, scen.n_agents, k)
            obs, _, _, _, info = env.step(action)
            ever_solved |= info["all_on_goal"]
    solve_rate = ever_solved.float().mean().item()
    assert solve_rate >= 0.9, f"scripted policy only solved {solve_rate:.1%} of envs"


# --------------------------------------------------------------------------------------
# 3. The priority token is a genuine rank-mapped permutation.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_priority_token_is_a_permutation(device):
    """``_prio`` sorts to ``linspace(-1, 1, n)`` in every env, is roughly uniform across
    agent slots, is constant within an episode, is redrawn on reset, and is all-zero when
    ``use_priority=False`` -- the controlled ablation the module docstring promises.
    """
    n_envs, n_agents = 4096, 4
    env = _env(device, n_envs=n_envs, n_agents=n_agents, use_priority=True)
    env.reset(seed=0)
    scen = env.scenario

    expected = torch.linspace(-1.0, 1.0, n_agents, device=device, dtype=env.dtype)
    sorted_prio, _ = torch.sort(scen._prio, dim=-1)
    torch.testing.assert_close(
        sorted_prio, expected.unsqueeze(0).expand(n_envs, -1), rtol=1e-5, atol=1e-5
    )

    # Each *slot* (agent index), averaged over many envs, should see roughly a uniform
    # spread of ranks -- not always the same rank -- so the per-slot mean should sit near
    # the population mean of 0.
    per_slot_mean = scen._prio.mean(dim=0)
    assert (per_slot_mean.abs() < 0.1).all(), per_slot_mean

    before = scen._prio.clone()
    actions = torch.zeros(n_envs, n_agents, env.act_dim, device=device, dtype=env.dtype)
    for _ in range(5):
        env.step(actions)
    assert torch.equal(scen._prio, before), "priority token must be constant within an episode"

    env.reset(seed=1)
    assert not torch.equal(scen._prio, before), "priority token must be redrawn on reset"

    env2 = _env(device, n_envs=64, n_agents=n_agents, use_priority=False)
    env2.reset(seed=0)
    assert torch.equal(env2.scenario._prio, torch.zeros_like(env2.scenario._prio))


# --------------------------------------------------------------------------------------
# 4. The corner SDF gradient is a genuine unit normal pointing into free space.
# --------------------------------------------------------------------------------------


def test_sdf_normal_points_to_free_space():
    """``_corner_sdf_grad`` is a unit vector everywhere, and stepping along it increases
    the (folded) distance -- i.e. it points away from the nearest block, toward free space.

    Probes deliberately avoid the three medial axes (``px == 0``, ``py == 0``,
    ``|qx| == |qy|``): the folded SDF has a genuine gradient ridge there (two branches of
    the fold meet with equal distance but opposite normals), and the one-sided limit the
    ``>=`` tie-break picks does not have to increase the SDF along itself -- that is a
    property of the ridge, not a bug in the gradient. Off-axis, in the corner region, the
    x-face region, the y-face region and the block interior, the gradient must be a
    genuine ascent direction.

    Determinism is checked too: two calls with the same input must agree exactly, since
    every branch is a plain ``>=``/``clamp`` with no randomness, and this is the only
    thing this test *can* check against without a Warp path (the fused kernel is being
    rewritten concurrently, so it is not a parity oracle here).
    """
    scen = GiveWayScenario(n_agents=1, world_size=1.0, agent_radius=0.05)
    bx, hx = scen._box_center[0], scen._box_half[0]
    # Corner region (outside the block, off both face axes and the |qx|==|qy| diagonal),
    # x-face region, y-face region, and block interior -- each off every medial axis.
    probes = torch.tensor(
        [
            [bx + hx + 0.05, bx + hx + 0.13],  # corner region, off the diagonal
            [bx + hx + 0.03, 0.30],  # x-face region (only the x side is "outside")
            [0.35, bx + hx + 0.04],  # y-face region (only the y side is "outside")
            [bx, bx],  # deep inside the block, off the diagonal by construction...
        ],
        dtype=torch.float64,
    )
    # ...except the last probe *is* on the diagonal (bx, bx) -- replace it with an
    # off-diagonal interior point instead.
    probes[3] = torch.tensor([bx - 0.01, bx + 0.05])

    for i in range(probes.shape[0]):
        p = probes[i]
        px, py = p[0].item(), p[1].item()
        assert abs(px) > 1e-6 and abs(py) > 1e-6, "probe must be off the px==0/py==0 axes"
        assert abs(abs(px) - abs(py)) > 1e-6, "probe must be off the |qx|==|qy| diagonal"

        grad = scen._corner_sdf_grad(p)
        norm = grad.norm().item()
        assert norm == pytest.approx(1.0, abs=1e-6), f"probe {i}: |grad| = {norm}"

        sdf0 = scen._corner_sdf(p).item()
        h = 1e-4
        sdf1 = scen._corner_sdf(p + h * grad).item()
        assert sdf1 > sdf0, f"probe {i}: stepping along grad did not increase sdf"

        grad_again = scen._corner_sdf_grad(p)
        assert torch.equal(grad, grad_again), f"probe {i}: gradient is not deterministic"


# --------------------------------------------------------------------------------------
# 5. Wall and contact ramps are continuous over their activation bands.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_wall_and_contact_terms_are_continuous(device):
    """The reward's wall ramp and proximity ramp are continuous ramps, not step counts.

    A single agent is swept laterally across the corridor (no physics -- ``write_state``
    plus ``post_step()``), well inside one arm so only the wall ramp can be nonzero. It
    must lie in [0, 1], hit exactly 0 at the centreline, rise monotonically with
    ``|lateral|`` on each side of the centreline (it ramps up as the wall is approached),
    and never jump by more than a small multiple of the sweep step measured against the
    ramp's own width, ``contact_margin``.

    A second sweep drives two agents together along the corridor axis and checks the
    proximity ramp: 0 at or beyond ``collision_reach``, 1 at or inside touching distance
    (``2r``), and continuous in between.
    """
    scen = GiveWayScenario(n_agents=1, world_size=1.0, agent_radius=0.05)
    env = Environment(scen, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset(seed=0)
    c = scen.c
    dt = {"device": device, "dtype": env.dtype}

    n_steps = 200
    lats = torch.linspace(-(c - 1e-4), c - 1e-4, n_steps, **dt)
    ramps = torch.empty(n_steps, **dt)
    for i, lat in enumerate(lats):
        pos = torch.tensor([[[0.4, lat.item()]]], **dt)  # deep inside the +x arm
        env.world.write_state(None, pos=pos, vel=torch.zeros_like(pos))
        scen.post_step()
        ramps[i] = scen._cache["wallramp"][0, 0]

    assert ((ramps >= 0.0) & (ramps <= 1.0)).all()
    centre_idx = n_steps // 2
    # The two lattice points straddling the true centreline (lat=0) should both be ~0.
    assert ramps[centre_idx - 1].item() < 1e-2
    assert ramps[centre_idx].item() < 1e-2

    half = n_steps // 2
    left, right = ramps[: half + 1].flip(0), ramps[half:]
    assert (left.diff() >= -1e-6).all(), "wall ramp not monotone on the -lat side"
    assert (right.diff() >= -1e-6).all(), "wall ramp not monotone on the +lat side"

    step_lat = (2.0 * (c - 1e-4)) / (n_steps - 1)
    max_jump = ramps.diff().abs().max().item()
    assert max_jump <= 6.0 * step_lat / scen.contact_margin

    # --- proximity ramp: two agents approaching head-on along the corridor axis -------
    scen2 = GiveWayScenario(n_agents=2, world_size=1.0, agent_radius=0.05)
    env2 = Environment(scen2, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env2.reset(seed=0)
    r = scen2.agent_radius
    reach = scen2._collision_reach
    n_steps2 = 200
    gaps = torch.linspace(reach + 0.05, 2.0 * r - 0.05, n_steps2, **dt)
    prox = torch.empty(n_steps2, **dt)
    for i, gap in enumerate(gaps):
        half_gap = gap.item() / 2.0
        pos = torch.tensor([[[-half_gap, 0.0], [half_gap, 0.0]]], **dt)
        env2.world.write_state(None, pos=pos, vel=torch.zeros_like(pos))
        scen2.post_step()
        prox[i] = scen2._cache["proximity"][0, 0]

    assert ((prox >= 0.0) & (prox <= 1.0)).all()
    assert prox[0].item() < 1e-6, "proximity must be 0 at/beyond collision_reach"
    assert prox[-1].item() == pytest.approx(1.0, abs=1e-3), "proximity must saturate at 2r"
    assert (prox.diff() >= -1e-6).all(), "proximity must be monotone as the gap closes"


# --------------------------------------------------------------------------------------
# 6. Neighbour slot tie-break: exact distance ties go to the lower agent index.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_neighbour_ordering_tiebreak(device):
    """Two neighbours at bit-identical squared distance: the lower index wins slot 0.

    Agent 1 and agent 2 are placed as exact mirror images around agent 0 along agent 0's
    lateral axis, so ``dsq`` is bit-identical for both -- the sort is stable, so agent 1
    (the lower index) must land in slot 0.
    """
    scen = GiveWayScenario(n_agents=3, world_size=1.0, use_priority=False)
    env = Environment(scen, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset(seed=0)
    dt = {"device": device, "dtype": env.dtype}
    u0 = torch.tensor([1.0, 0.0], **dt)

    pos = torch.tensor([[0.4, 0.0], [0.4, 0.03], [0.4, -0.03]], **dt).unsqueeze(0)
    axis = u0.unsqueeze(0).expand(3, -1).unsqueeze(0).clone()
    goals = torch.tensor([[0.6, 0.0], [0.9, 0.03], [-0.9, -0.03]], **dt).unsqueeze(0)

    env.world.write_state(None, pos=pos, vel=torch.zeros_like(pos))
    scen._axis.copy_(axis)
    env.world.goals.copy_(goals)
    scen.post_step()

    obs = scen.observations()
    # n_goal_dist at slot 0 must be agent 1's distance-to-goal (0.5), not agent 2's (1.3).
    n_goal_dist_slot0 = obs[0, 0, 13 + 6]
    assert n_goal_dist_slot0.item() == pytest.approx(0.5, abs=1e-4)


# --------------------------------------------------------------------------------------
# 7. Neighbour axis direction separates traffic type.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_neighbour_goal_direction_separates_traffic(device):
    """``(ndir_along, ndir_lat)`` is ``(-1, 0)`` head-on, ``(0, +-1)`` crossing, ``(1, 0)``
    same-way -- the one feature that tells a robot what *kind* of conflict it is in.
    """
    scen = GiveWayScenario(n_agents=2, world_size=1.0, use_priority=False)
    env = Environment(scen, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset(seed=0)
    dt = {"device": device, "dtype": env.dtype}

    def ndir_for(axis0, axis1):
        pos = torch.tensor([[0.0, 0.0], [0.1, 0.1]], **dt).unsqueeze(0)
        axis = torch.stack([torch.tensor(axis0, **dt), torch.tensor(axis1, **dt)]).unsqueeze(0)
        goals = torch.zeros(1, 2, 2, **dt)
        env.world.write_state(None, pos=pos, vel=torch.zeros_like(pos))
        scen._axis.copy_(axis)
        env.world.goals.copy_(goals)
        scen.post_step()
        obs = scen.observations()
        base = 13
        return obs[0, 0, base + 4].item(), obs[0, 0, base + 5].item()

    head_on = ndir_for([1.0, 0.0], [-1.0, 0.0])
    assert head_on == pytest.approx((-1.0, 0.0), abs=1e-5)

    crossing = ndir_for([1.0, 0.0], [0.0, 1.0])
    assert crossing[0] == pytest.approx(0.0, abs=1e-5)
    assert abs(crossing[1]) == pytest.approx(1.0, abs=1e-5)

    same_way = ndir_for([1.0, 0.0], [1.0, 0.0])
    assert same_way == pytest.approx((1.0, 0.0), abs=1e-5)


# --------------------------------------------------------------------------------------
# 8. Spawn desync orders arrivals by the priority token.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_spawn_desync_orders_arrivals(device):
    """``set_spawn_desync`` staggers spawn depth by rank; ``desync_mode="random"`` is the
    uncorrelated control; and at ``frac=1`` no agent spawns inside the wall band or the
    junction.
    """
    n_envs = 2048
    scen = GiveWayScenario(n_agents=4, use_priority=True, desync_mode="ranked")
    env = Environment(scen, n_envs=n_envs, device=device, dt=0.05, seed=0, fused=False)

    def depths_and_prio(seed):
        env.reset(seed=seed)
        pos = env.world.state.pos
        depth = (pos * scen._axis).sum(dim=-1).abs()  # |dot(pos, arm_dir)|
        return depth, scen._prio.clone(), pos.clone()

    depth0, _, _ = depths_and_prio(0)
    # At frac=0 every agent in its own env spawns near the same depth, up to jitter.
    assert depth0.std(dim=-1).mean().item() < 0.05

    scen.set_spawn_desync(1.0)
    depth1, prio1, pos1 = depths_and_prio(1)
    flat_depth = depth1.reshape(-1)
    flat_prio = prio1.reshape(-1)
    corr = torch.corrcoef(torch.stack([flat_depth, flat_prio]))[0, 1].item()
    assert corr < -0.5, f"ranked desync should anti-correlate depth with token, got {corr}"

    sdf = scen._corner_sdf(pos1)
    assert (sdf >= scen._wall_reach - 1e-5).all(), "an agent spawned inside the wall band"
    assert (depth1 >= scen._s_min - 1e-5).all(), "an agent spawned inside the junction"

    scen_random = GiveWayScenario(n_agents=4, use_priority=True, desync_mode="random")
    env_random = Environment(
        scen_random, n_envs=n_envs, device=device, dt=0.05, seed=0, fused=False
    )
    scen_random.set_spawn_desync(1.0)
    env_random.reset(seed=1)
    pos_r = env_random.world.state.pos
    depth_r = (pos_r * scen_random._axis).sum(dim=-1).abs()
    corr_r = torch.corrcoef(torch.stack([depth_r.reshape(-1), scen_random._prio.reshape(-1)]))[
        0, 1
    ].item()
    assert abs(corr_r) < 0.15, f"random desync should not correlate with the token, got {corr_r}"


# --------------------------------------------------------------------------------------
# 9. Goal-hold bonus and termination arithmetic.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_hold_bonus_and_termination(device):
    """``done()`` requires every agent on goal simultaneously; on-goal agents collect the
    hold bonus whether or not the episode has actually terminated, and the reward
    arithmetic is checked against the documented coefficients rather than a golden number.
    """
    scen = GiveWayScenario(n_agents=2, world_size=1.0)
    env = Environment(scen, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset(seed=0)
    dt = {"device": device, "dtype": env.dtype}

    goals = torch.tensor([[0.5, 0.0], [0.0, -0.5]], **dt).unsqueeze(0)
    env.world.goals.copy_(goals)

    # Both agents exactly on their (well-separated, wall-clear) goals.
    pos = goals.clone()
    env.world.write_state(None, pos=pos, vel=torch.zeros_like(pos))
    scen.post_step()  # settles dist_to_goal, but bakes in a large one-off shaping jump
    scen.post_step()  # second call: prev_dist now equals dist_to_goal, so shaping is 0

    assert scen.done()[0].item() is True
    assert scen._cache["proximity"][0].abs().sum().item() < 1e-9
    assert scen._cache["wallramp"][0].abs().sum().item() < 1e-9
    assert scen._cache["pos_shaping"][0].abs().sum().item() < 1e-9

    expected_all_on = scen.time_penalty + scen.goal_hold_bonus + scen.final_reward
    reward_all_on = scen.rewards()[0]
    torch.testing.assert_close(
        reward_all_on, torch.full_like(reward_all_on, expected_all_on), rtol=1e-5, atol=1e-5
    )

    # Move agent 1 off its goal; agent 0 stays exactly on its goal.
    pos_off = pos.clone()
    pos_off[0, 1] = torch.tensor([0.0, -0.3], **dt)  # 0.2 short of goal, > goal_tolerance
    env.world.write_state(None, pos=pos_off, vel=torch.zeros_like(pos_off))
    scen.post_step()
    scen.post_step()

    assert scen.done()[0].item() is False
    assert scen._cache["on_goal"][0, 0].item() is True
    assert scen._cache["on_goal"][0, 1].item() is False

    expected_on = scen.time_penalty + scen.goal_hold_bonus  # no final_reward: not all on
    expected_off = scen.time_penalty  # no hold bonus: agent 1 is not on goal
    reward_split = scen.rewards()[0]
    assert reward_split[0].item() == pytest.approx(expected_on, abs=1e-5)
    assert reward_split[1].item() == pytest.approx(expected_off, abs=1e-5)


# --------------------------------------------------------------------------------------
# 10. The curriculum knobs are two-sided.
# --------------------------------------------------------------------------------------


def _geometry_snapshot(scen):
    return (
        scen._box_half,
        scen._box_center,
        scen._s_min,
        scen._s_max,
        scen._stagger,
        scen._jitter_long,
        scen._jitter_lat,
        scen._desync_amp,
    )


def test_curriculum_is_two_sided():
    """``set_corridor_scale``/``set_spawn_desync`` up-and-back-down reproduces the exact
    starting geometry, and a corridor-scale change re-clamps a previously set desync
    amplitude rather than leaving it stale.
    """
    scen = GiveWayScenario(n_agents=4, world_size=1.0)
    base = _geometry_snapshot(scen)

    scen.set_corridor_scale(1.5)
    assert _geometry_snapshot(scen) != base
    scen.set_corridor_scale(1.0)
    assert _geometry_snapshot(scen) == base, "corridor_scale(1.0) must restore exactly"

    scen.set_spawn_desync(1.0)
    widened = _geometry_snapshot(scen)
    assert widened != base
    scen.set_spawn_desync(0.0)
    assert _geometry_snapshot(scen) == base, "spawn_desync(0.0) must restore exactly"

    # A corridor-scale change after a nonzero desync must re-clamp the amplitude rather
    # than replay a stale one computed for the old (wider) geometry.
    scen.set_spawn_desync(1.0)
    amp_wide = scen._desync_amp
    assert amp_wide <= scen._desync_max
    scen.set_corridor_scale(0.6)  # a narrower corridor: less room left for the arm
    assert scen._desync_amp <= scen._desync_max
    assert scen._desync_amp != amp_wide, "corridor_scale must recompute the desync amplitude"
