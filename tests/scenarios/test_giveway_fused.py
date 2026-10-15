"""Fused Warp obs/reward kernels must match the torch reference path (bit-close).

Give-way's own additions on top of the navigation-shaped parity suite: the wall-contact
term (a discrete flag derived from the mirrored box SDF, so it has to match *exactly*),
the ``multiobj_reward`` sum identity, and the corridor-width constraint that the whole
task rests on.
"""

import math

import pytest
import torch
from conftest import (
    DEVICES,
    FusedSpec,
    assert_fused_determinism,
    assert_grad_falls_back_to_torch,
    assert_reset_parity,
    fused_env,
    fused_rollout,
)

from swarp.scenarios.giveway import GiveWayScenario

SPEC = FusedSpec(
    scenario=GiveWayScenario,
    fields=(
        "obs",
        "rew",
        "done",
        "info:collisions",
        "info:dist_to_goal",
        "info:on_goal",
        "info:wall_contact",
    ),
    grad_steps=6,
    grad_index=3,
    grad_backprop="rew",  # give-way's reward is differentiable shaping
    # Load-bearing: this seed is what makes ``test_feature_guards_actually_fire`` see both
    # a wall contact and an agent-agent contact. Changing it can make that guard pass
    # vacuously, which no failure reports.
    action_seed=2,
    # A velocity-mode agent settles at a contact depth of ``max_speed / (k * sub_dt)``, so
    # the corridor is only a wall at a small ``sub_dt``. 8 is the documented floor at
    # dt=0.05, and the parity rollouts run at the floor deliberately: the stiff regime is
    # where the two paths' contact geometry is most likely to disagree.
    substeps=8,
)

#: Two slots per arm (``n_agents // 4``), which is what makes the agent-agent contact
#: guard fire inside the harness's 5-step episodes: agents queued 3 radii apart in the
#: same arm can close that gap in a couple of steps, where agents in *different* arms
#: cannot reach the junction in time. ``WORLD`` is then the smallest arena those two slots
#: fit in with room to spare. Corridor width is set by ``agent_radius``, not by either.
N_AGENTS = 8
WORLD = 0.4


def _scen(
    *, n_agents=N_AGENTS, shared_reward=True, neighbor_obs=3, world_size=WORLD, difficulty=1.0
):
    scen = GiveWayScenario(
        n_agents=n_agents,
        shared_reward=shared_reward,
        neighbor_obs=neighbor_obs,
        world_size=world_size,
        neighbor_method="brute",  # pin the backend so both paths see identical lists
    )
    scen.difficulty = difficulty
    return scen


def _env(device, fused, *, dtype=torch.float32, **kw):
    return fused_env(_scen(**kw), device, fused, spec=SPEC, dtype=dtype)


def _run(env, n_steps, device, n_agents, dtype=torch.float32):
    return fused_rollout(env, n_steps, device, n_agents, SPEC, dtype=dtype)


# ------------------------------------------------------------------- parity


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shared_reward", [False, True])
def test_fused_matches_torch(device, shared_reward):
    kw = dict(shared_reward=shared_reward)
    fused = _run(_env(device, fused=True, **kw), 20, device, N_AGENTS)
    torchp = _run(_env(device, fused=False, **kw), 20, device, N_AGENTS)
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, df, cf, gf, ogf, wf = f
        ot, rt, dt_, ct, gt, ogt, wt = r
        torch.testing.assert_close(of, ot, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"reward@{t}")
        assert torch.equal(df, dt_), f"done@{t}"
        # touching / on_goal / wall_contact are discrete: exact, or the reward built on
        # them is wrong by a whole penalty rather than by an ulp.
        assert torch.equal(cf, ct), f"collisions@{t}"
        assert torch.equal(ogf, ogt), f"on_goal@{t}"
        assert torch.equal(wf, wt), f"wall_contact@{t}"
        torch.testing.assert_close(gf, gt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"dist@{t}")


@pytest.mark.parametrize("device", DEVICES)
def test_feature_guards_actually_fire(device):
    """The two contact terms must genuinely occur under ``SPEC.action_seed``.

    A parity assertion over a rollout where nothing ever touches a wall or another agent
    compares zeros against zeros. Both flags are checked on the *torch* rollout, i.e. on
    the oracle, so a fused bug cannot make this guard pass.
    """
    torchp = _run(_env(device, fused=False), 20, device, N_AGENTS)
    saw_wall = any(step[6].sum().item() > 0 for step in torchp)
    saw_touch = any(step[3].sum().item() > 0 for step in torchp)
    assert saw_wall, "test config induced no wall contact"
    assert saw_touch, "test config induced no agent-agent contact"


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device, N_AGENTS)
    b = _run(_env(device, fused=True), 10, device, N_AGENTS)
    assert_fused_determinism(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    env = _env(device, fused=True)
    assert_grad_falls_back_to_torch(env, device, N_AGENTS, SPEC)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_parity(device):
    """A standalone ``reset()``/``reset_at()`` must leave the same outputs as torch."""
    assert_reset_parity(lambda fused: _env(device, fused), device, N_AGENTS, SPEC)


# ------------------------------------------------------------------ geometry


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("difficulty", [0.0, 1.0])
def test_spawns_lie_in_the_free_plus_and_goals_mirror_nominal_slots(device, difficulty):
    env = _env(device, fused=True, n_agents=6, world_size=0.5, difficulty=difficulty)
    env.reset(seed=0)
    scen = env.scenario
    pos = env.world.state.pos
    hw, arena, r = scen.half_w, scen.world_size, scen.agent_radius

    # Free space is the plus {|x| <= hw} u {|y| <= hw}: at least one coordinate is inside
    # the corridor half-width.
    a = pos.abs()
    assert (a.amin(dim=-1) <= hw + 1e-6).all(), "a spawn landed inside a corner block"
    assert (a <= arena + 1e-6).all(), "a spawn landed outside the arena"
    # ...and the *disc* clears the blocks too, which is the stronger statement the
    # lat_jitter bound exists to guarantee.
    assert (scen._wall_gap(pos) >= r - 1e-6).all(), "a spawned disc overlaps a corner block"

    # Goals are the point reflection of the NOMINAL (un-jittered) slot, so they are exact
    # and independent of difficulty.
    want = torch.empty_like(env.world.goals[0])
    for i in range(scen.n_agents):
        ux, uy = _ARM_DIRS[i % 4]
        base = scen._arm_len - (i // 4) * scen.slot_spacing
        want[i, 0] = -ux * base
        want[i, 1] = -uy * base
    for e in range(env.n_envs):
        torch.testing.assert_close(env.world.goals[e], want, rtol=0, atol=1e-6)

    # Headings point down the arm toward the junction.
    theta = env.world.state.theta
    for i in range(scen.n_agents):
        assert math.isclose(theta[0, i].item(), _ARM_THETA[i % 4], abs_tol=1e-6)


_ARM_DIRS = ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0))
_ARM_THETA = (-0.5 * math.pi, math.pi, 0.5 * math.pi, 0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_difficulty_controls_the_longitudinal_stagger(device):
    """``difficulty`` must still bite after the reset kernel has been launched once.

    The stagger reaches the kernel through a one-element device tensor precisely so a
    trainer can move it mid-training with a whole-step graph already captured; a scalar
    argument would have been baked into that capture. This pins the *observable*
    consequence — the spread of spawn radii — on both the eager and captured paths, rather
    than the mechanism.
    """
    env = _env(device, fused=True, n_agents=4, difficulty=1.0)
    env.reset(seed=0)
    scen = env.scenario

    def spread():
        r = env.world.state.pos.norm(dim=-1)
        return (r.amax() - r.amin()).item()

    # difficulty 1: every agent pinned to its nominal slot, so every radius is the arm
    # length up to the lateral jitter's contribution.
    assert spread() < 2.0 * scen.lat_jitter
    scen.difficulty = 0.0
    env.reset(seed=1)
    # difficulty 0: spawns spread down the arm, which is what staggers arrival.
    assert spread() > 0.5 * scen.long_jitter
    # Clamped, and readable back.
    scen.difficulty = 5.0
    assert scen.difficulty == 1.0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [False, True])
def test_politeness_is_observed_and_redrawn_every_reset(device, fused):
    """The symmetry-breaking scalar has to reach the policy, and has to move per episode.

    Without it a shared-weight actor emits identical actions for mirror-image agents, and
    in a corridor that fits one agent identical behaviour *is* the deadlock — so this
    pins both halves: the value lands in observation column 9 (the slot both the torch
    ``cat`` and the kernel write), and a reset re-draws it.
    """
    env = _env(device, fused=fused, n_agents=4)
    env.reset(seed=0)
    scen = env.scenario
    first = scen.politeness.clone()
    assert ((first >= 0.0) & (first < 1.0)).all()
    torch.testing.assert_close(env.scenario.observations()[..., 9], scen.politeness)
    # No ``seed=`` on the second reset: re-seeding would (correctly) reproduce the first
    # draw, and what is being pinned here is that the *stream* advances per episode.
    env.reset()
    assert not torch.equal(first, scen.politeness), "politeness was not redrawn on reset"


def test_corridor_width_constraint_is_enforced():
    # Too narrow: the agent does not fit in the corridor at all.
    with pytest.raises(ValueError, match="fits exactly one agent"):
        GiveWayScenario(agent_radius=0.05, corridor_half_width=0.04)
    # Too wide: two agents pass abreast and there is no give-way problem left.
    with pytest.raises(ValueError, match="fits exactly one agent"):
        GiveWayScenario(agent_radius=0.05, corridor_half_width=0.2)
    # The defaults sit strictly inside the band.
    scen = GiveWayScenario()
    assert scen.agent_radius < scen.half_w < 2.0 * scen.agent_radius


def test_lat_jitter_beyond_the_corridor_is_rejected():
    with pytest.raises(ValueError, match="lat_jitter"):
        GiveWayScenario(agent_radius=0.05, lat_jitter=0.5)


def test_constructs_on_cpu_with_three_agents():
    # What the registry-wide scenario test will do once this is registered.
    scen = GiveWayScenario(n_agents=3)
    assert scen.obs_dim == 10 + 5 * min(scen.neighbor_obs, 4)


# ------------------------------------------------------------- multiobj reward


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("shared_reward", [False, True])
def test_multiobj_reward_sums_to_the_scalar_reward(device, fused, shared_reward):
    env = _env(device, fused=fused, shared_reward=shared_reward)
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(SPEC.action_seed)
    with torch.no_grad():
        for _ in range(12):
            act = torch.empty(env.n_envs, N_AGENTS, 2, device=device).uniform_(-1, 1, generator=gen)
            _, rew, _, _, info = env.step(act)
            mo = info["multiobj_reward"]
            assert mo.shape == (env.n_envs, N_AGENTS, 5)
            torch.testing.assert_close(mo.sum(-1), rew, rtol=SPEC.rtol, atol=SPEC.atol)
