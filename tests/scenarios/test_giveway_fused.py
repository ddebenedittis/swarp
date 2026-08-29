"""Fused Warp obs/reward kernels must match the torch reference path (bit-close).

The give-way suite's own "did the feature fire" guard is **wall contact**: the whole
scenario is four corner blocks forming a one-lane corridor, so a rollout in which no agent
ever touched a block would leave the corridor physics — and the folded box SDF both paths
evaluate for it — completely untested. ``action_seed=17`` is what makes that guard, and
the agent-contact one next to it, pass; see ``FusedSpec`` in ``tests/conftest.py``.
"""

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

from swarp import Environment
from swarp.scenarios.giveway import GiveWayScenario

SPEC = FusedSpec(
    scenario=GiveWayScenario,
    fields=(
        "obs",
        "rew",
        "done",
        "info:collisions",
        "info:wall_contacts",
        "info:dist_to_goal",
        "info:on_goal",
        "info:frac_on_goal",
    ),
    grad_steps=6,
    grad_index=3,  # shaping baseline exercised on both sides of the grad step
    grad_backprop="rew",  # the reward is differentiable shaping
    action_seed=17,
)

# Five agents in a small arena: two of them share an arm (ranks 0 and 1), which is what
# puts agents within contact range of each other inside a five-step episode. A larger
# world simply cannot be crossed at ``max_speed`` before the harness's auto-reset fires,
# and the agent-contact half of the parity check would go vacuous.
N_AGENTS = 5
WORLD = 0.6


def _env(device, fused, *, n_agents=N_AGENTS, shared_reward=True, neighbor_obs=2,
         world_size=WORLD):
    scen = GiveWayScenario(
        n_agents=n_agents,
        world_size=world_size,
        shared_reward=shared_reward,
        neighbor_obs=neighbor_obs,
        neighbor_method="brute",  # pin the backend so both paths see identical lists
    )
    return fused_env(scen, device, fused, spec=SPEC)


def _run(env, n_steps, device, n_agents=N_AGENTS):
    return fused_rollout(env, n_steps, device, n_agents, SPEC)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shared_reward", [True, False])
@pytest.mark.parametrize("neighbor_obs", [1, 3])
def test_fused_matches_torch(device, shared_reward, neighbor_obs):
    kw = dict(device=device, shared_reward=shared_reward, neighbor_obs=neighbor_obs)
    fused = _run(_env(fused=True, **kw), 20, device)
    torchp = _run(_env(fused=False, **kw), 20, device)
    # The corridor must actually be felt, and the agents must actually meet — otherwise
    # the box SDF and the neighbor loop are both untested by this rollout.
    assert any(step[4].sum().item() > 0 for step in torchp), "no agent ever touched a block"
    assert any(step[3].sum().item() > 0 for step in torchp), "no agent-agent contact"
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, df, cf, wf, gf, ogf, fgf = f
        ot, rt, dt_, ct, wt, gt, ogt, fgt = r
        torch.testing.assert_close(of, ot, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"reward@{t}")
        assert torch.equal(df, dt_), f"done@{t}"
        # touching / wall contact / on_goal are discrete: exact, not close.
        assert torch.equal(cf, ct), f"collisions@{t}"
        assert torch.equal(wf, wt), f"wall_contacts@{t}"
        assert torch.equal(ogf, ogt), f"on_goal@{t}"
        torch.testing.assert_close(gf, gt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"dist@{t}")
        torch.testing.assert_close(fgf, fgt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"frac@{t}")


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device)
    b = _run(_env(device, fused=True), 10, device)
    assert_fused_determinism(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    """A grad step must take the differentiable torch path and leave the shaping
    baseline continuous across the switch back to fused."""
    env = _env(device, fused=True)
    assert_grad_falls_back_to_torch(env, device, N_AGENTS, SPEC)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_parity(device):
    """A standalone ``reset()``/``reset_at()`` must leave the same outputs as torch."""
    assert_reset_parity(lambda fused: _env(device, fused), device, N_AGENTS, SPEC)


@pytest.mark.parametrize("device", DEVICES)
def test_corridor_scale_widens_the_free_lane(device):
    """The curriculum knob must move the *engine's* blocks and the fused SDF together.

    A point one lane-width off the centreline is buried inside a corner block at
    ``scale=1`` and out in the open at ``scale=2``; both the torch oracle's SDF and the
    kernel's ``geom``-driven contact flag have to say so, or the policy would be trained
    against geometry the physics no longer has.
    """
    env = _env(device, fused=True, n_agents=1)
    scen = env.scenario
    env.reset(seed=0)
    base_half = env.world.obstacle_half_extents.clone()
    c = scen.corridor_half_width
    # A lateral offset that is *past* the base corridor's wall (|y| > c, i.e. inside a
    # block) yet still clear of the widened corridor's contact band (|y| < 2c - reach).
    # Midway between the two, so neither assertion below sits near a threshold.
    lat = 0.5 * (c + (2.0 * c - scen._wall_reach))
    probe = torch.tensor([[[0.5 * WORLD, lat]]], device=device, dtype=env.dtype)

    def contact_flag():
        env.world.write_state(None, pos=probe, vel=torch.zeros_like(probe))
        scen.post_step()  # recompute obs/reward in place, without stepping the physics
        return scen.info()["wall_contacts"][0, 0].item()

    assert scen._corner_sdf(probe).item() < 0.0  # inside a block
    assert contact_flag() == 1.0

    scen.set_corridor_scale(2.0)
    assert scen._corner_sdf(probe).item() > 0.0  # now free corridor
    assert contact_flag() == 0.0
    assert env.world.obstacle_half_extents.shape == base_half.shape  # count unchanged
    assert not torch.equal(env.world.obstacle_half_extents, base_half)

    scen.set_corridor_scale(1.0)  # restores the constructor's geometry exactly
    assert torch.equal(env.world.obstacle_half_extents, base_half)
    assert contact_flag() == 1.0


@pytest.mark.parametrize("device", DEVICES)
def test_the_corridor_is_solid_and_one_lane(device):
    """Robots and walls must not yield — the scenario's premise depends on it.

    This is a physics-parameter regression test, not a kernel one. ``contact_k`` was
    originally inherited from Push-T at 8000, and at that stiffness a velocity-mode robot
    settles *0.033 deep* into a corner block — two thirds of its own radius. Blocks that
    soft are not one-lane geometry, and the failure is silent: nothing errors, the task
    simply stops being the task. The observable symptom was that a greedy "drive straight
    at the goal" policy solved 100% of episodes by squeezing through robots and walls.

    So this pins the two things that make give-way give-way, at the real defaults:

    1. two robots driven head-on in a straight arm never interpenetrate and never swap
       ends (the arm is genuinely one lane, away from the junction's passing bay);
    2. a robot driven into a block does not end up inside it.

    A ``contact_k`` regression trips both, and so does dropping ``substeps`` below the
    documented 16 — at 8 the one-substep overshoot (``max_speed * sub_dt`` = 0.00625) is
    wider than the contact margin, so a fast robot is inside a block before the spring
    sees it and no stiffness recovers that. Both bounds are asserted at *zero* tolerance
    because that is what the shipped defaults actually achieve.
    """
    scen = GiveWayScenario(n_agents=2)
    env = Environment(
        scen, n_envs=1, device=device, dt=0.05, substeps=16, seed=0, max_steps=None
    )
    env.reset(seed=0)
    r, c = scen.agent_radius, scen.corridor_half_width

    # Both robots inside the +x arm, clear of the junction, driving through each other.
    start = torch.tensor([[[0.40, 0.0], [0.70, 0.0]]], device=device, dtype=env.dtype)
    goal = torch.tensor([[[0.95, 0.0], [0.15, 0.0]]], device=device, dtype=env.dtype)
    env.world.write_state(None, pos=start, vel=0.0)
    closest = float("inf")
    with torch.no_grad():
        for _ in range(400):
            d = goal - env.world.state.pos
            env.step(d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9))
            p = env.world.state.pos[0]
            closest = min(closest, (p[0] - p[1]).norm().item())

    p = env.world.state.pos[0]
    assert torch.isfinite(p).all(), "stiff contact went unstable"
    assert closest >= 2.0 * r, (
        f"robots interpenetrated: closest centre distance {closest:.4f} vs touching "
        f"{2 * r:.4f} — contact_k={scen.contact_k:g} / substeps too soft"
    )
    assert p[0, 0] < p[1, 0], "robots swapped ends: the arm is not one lane"

    # ...and a robot cannot bury itself in a corner block.
    env.world.write_state(
        None,
        pos=torch.tensor([[[0.5, 0.0], [-0.5, 0.0]]], device=device, dtype=env.dtype),
        vel=0.0,
    )
    push = torch.tensor([[[0.0, 1.0], [0.0, -1.0]]], device=device, dtype=env.dtype)
    with torch.no_grad():
        for _ in range(300):
            env.step(push)
    y = env.world.state.pos[0, :, 1].abs()
    assert torch.isfinite(y).all()
    assert (y <= c - r + 1e-3).all(), (
        f"robot sank into a corner block: |y|={y.tolist()} exceeds the wall at "
        f"c - r = {c - r:.4f}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_no_interpenetration_in_a_crowded_junction(device):
    """Forces superpose in a jam, so two robots head-on is not a sufficient test.

    Eight robots on random actions pile into the junction and grind along the blocks. This
    is the configuration that exposed the shipped defaults as too soft: at ``substeps=8``
    the worst overlap reached 16.8% of a robot radius even at ``contact_k`` 2e5, because
    the floor there is the one-substep overshoot rather than the spring.
    """
    scen = GiveWayScenario(n_agents=8)
    env = Environment(
        scen, n_envs=16, device=device, dt=0.05, substeps=16, seed=0, max_steps=None
    )
    env.reset(seed=0)
    r = scen.agent_radius
    gen = torch.Generator(device=device).manual_seed(3)
    worst_pair = worst_block = 0.0
    with torch.no_grad():
        for _ in range(200):
            act = torch.empty(
                16, 8, env.act_dim, device=device, dtype=env.dtype
            ).uniform_(-1, 1, generator=gen)
            env.step(act)
            p = env.world.state.pos
            for i in range(8):
                for j in range(i + 1, 8):
                    gap = (p[:, i] - p[:, j]).norm(dim=-1)
                    worst_pair = max(worst_pair, (2 * r - gap).max().item())
            worst_block = max(worst_block, (r - scen._corner_sdf(p)).max().item())

    assert torch.isfinite(env.world.state.pos).all(), "stiff contact went unstable"
    assert worst_pair <= 0.0, (
        f"robots overlapped by {worst_pair:.5f} ({worst_pair / r:.1%} of a radius)"
    )
    assert worst_block <= 0.0, f"robot cut {worst_block:.5f} into a block"
