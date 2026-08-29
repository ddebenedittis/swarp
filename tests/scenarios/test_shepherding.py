"""Shepherding scenario: shepherds herd fleeing sheep into a pen."""

import pytest
import torch
from conftest import DEVICES

from swarp import Environment
from swarp.scenarios.shepherding import ShepherdingScenario


def _env(device, n_envs=4, **kw):
    # fused=False on purpose: this suite exercises the *torch reference* flock
    # integration (``_refresh``), poking ``sheep_*`` directly and reading it back. The
    # fused parity of that path is what test_shepherding_fused.py is for.
    kw.setdefault("n_agents", 3)
    kw.setdefault("n_sheep", 2)
    return Environment(
        ShepherdingScenario(**kw),
        n_envs=n_envs,
        device=device,
        dt=0.05,
        seed=1,
        fused=False,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_shepherding_api_and_finiteness(device):
    env = _env(device, n_envs=8)
    obs = env.reset()
    assert obs.shape == (8, 3, 6 + 4 * 2) and torch.isfinite(obs).all()
    gen = torch.Generator(device=device).manual_seed(0)
    for _ in range(10):
        a = torch.rand(8, 3, env.world.act_dim, generator=gen, device=device) * 2 - 1
        obs, rew, term, trunc, info = env.step(a)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
        assert rew.shape == (8, 3) and term.shape == (8,) and trunc.shape == (8,)
        assert info["sheep_penned"].shape == (8,) and info["sheep_dist_to_pen"].shape == (8,)
        assert ((info["sheep_penned"] >= 0.0) & (info["sheep_penned"] <= 1.0)).all()


@pytest.mark.parametrize("device", DEVICES)
def test_shepherding_determinism(device):
    def run():
        env = _env(device, n_envs=4)
        env.reset(seed=4)
        gen = torch.Generator(device=device).manual_seed(2)
        out = []
        for _ in range(8):
            a = torch.rand(4, 3, env.world.act_dim, generator=gen, device=device) * 2 - 1
            obs, rew, *_ = env.step(a)
            out += [obs, rew, env.scenario.sheep_pos.clone()]
        return out

    for x, y in zip(run(), run(), strict=True):
        assert torch.equal(x, y)


@pytest.mark.parametrize("device", DEVICES)
def test_shepherd_repels_sheep_without_touching_it(device):
    """The defining property: a shepherd *near* a sheep pushes it away, contact or not.

    One stationary shepherd is parked 0.2 away from a lone sheep — well outside the
    contact reach (``agent_radius + sheep_radius + contact_margin`` = 0.11) and well
    inside ``flee_radius`` = 0.35 — and the gap is asserted to never close. Everything
    the sheep does here is the flee term; separation and cohesion are identically zero
    for a flock of one.
    """
    scenario = ShepherdingScenario(n_agents=1, n_sheep=1, world_size=1.0)
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset()
    scenario.sheep_pos = torch.tensor([[[0.0, 0.0]]], device=device, dtype=torch.float32)
    scenario.sheep_vel = torch.zeros_like(scenario.sheep_vel)
    scenario.pen_pos = torch.tensor([[0.7, 0.0]], device=device, dtype=torch.float32)
    shepherd = torch.tensor([[[-0.2, 0.0]]], device=device, dtype=torch.float32)
    scenario.world.state = scenario.world.state._replace(pos=shepherd)
    scenario._install_obstacles()
    scenario._prev_dist = None
    scenario._refresh(integrate=False)

    reach = scenario.agent_radius + scenario.sheep_radius + scenario.contact_margin
    actions = torch.zeros(1, 1, env.world.act_dim, device=device)
    min_gap = float("inf")
    for _ in range(20):
        env.step(actions)
        gap = (scenario.sheep_pos[0, 0] - scenario.world.state.pos[0, 0]).norm().item()
        min_gap = min(min_gap, gap)
    assert min_gap > reach, "the sheep was touched; this no longer isolates the flee force"
    assert scenario.sheep_pos[0, 0, 0].item() > 0.01  # driven directly away, in +x
    assert abs(scenario.sheep_pos[0, 0, 1].item()) < 1e-6  # and only in +x


@pytest.mark.parametrize("device", DEVICES)
def test_sheep_separate_rather_than_stacking(device):
    """Two sheep dropped on top of each other spread out instead of merging."""
    scenario = ShepherdingScenario(n_agents=1, n_sheep=2, world_size=1.0)
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset()
    scenario.sheep_pos = torch.tensor(
        [[[0.0, 0.01], [0.0, -0.01]]], device=device, dtype=torch.float32
    )
    scenario.sheep_vel = torch.zeros_like(scenario.sheep_vel)
    scenario.pen_pos = torch.zeros(1, 2, device=device, dtype=torch.float32)
    # the shepherd is parked in a far corner, > flee_radius from either sheep
    far = torch.tensor([[[0.9, 0.9]]], device=device, dtype=torch.float32)
    scenario.world.state = scenario.world.state._replace(pos=far)
    scenario._install_obstacles()
    scenario._prev_dist = None
    scenario._refresh(integrate=False)

    d0 = (scenario.sheep_pos[0, 0] - scenario.sheep_pos[0, 1]).norm().item()
    actions = torch.zeros(1, 1, env.world.act_dim, device=device)
    for _ in range(120):
        env.step(actions)
    d1 = (scenario.sheep_pos[0, 0] - scenario.sheep_pos[0, 1]).norm().item()
    assert d1 > d0 + 0.02, f"sheep stacked instead of separating ({d0} -> {d1})"
    # Long enough to overshoot and settle: separation and cohesion balance where
    # sep_gain * (1 - d/sep_radius) == cohesion_gain * d/2, i.e. d ~= 0.194 here, so the
    # upper bound also pins that cohesion is what stops them drifting apart forever.
    assert d1 < scenario.sep_radius * 1.5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_agents", [1, 5])
def test_sheep_cannot_outrun_the_shepherds(device, n_agents):
    """A sheep must never move faster than a shepherd, however many of them converge.

    This is the property that decides whether the task is *learnable* rather than
    whether any single number is right, and violating it fails nothing else: a flock
    that outruns its dogs can never be cornered, so the pen bonus is unreachable and a
    run just plateaus. It was real — an uncapped flee sum at ``flee_gain=2.0`` peaked at
    2.3x ``max_speed`` under this policy and stalled a 6000-iteration MAPPO run at ~7%
    penned.

    The policy is the adversarial one: every shepherd drives at full speed straight at
    its nearest sheep, which maximizes both the number of superposed flee terms and the
    contact-transient spike. Parametrized over the flock-size extremes because the two
    bounds have different causes — the summed-flee cap for many shepherds, the
    ``max_sheep_speed`` cap for the single-shepherd ram.
    """
    scenario = ShepherdingScenario(n_agents=n_agents, n_sheep=2)
    env = Environment(scenario, n_envs=8, device=device, dt=0.05, seed=0, fused=False)
    env.reset(seed=0)
    peak = 0.0
    for _ in range(200):
        rel = scenario.sheep_pos.unsqueeze(1) - env.world.state.pos.unsqueeze(2)  # [E,A,K,2]
        nearest = rel.norm(dim=-1).argmin(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, -1, 2)
        toward = rel.gather(2, nearest).squeeze(2)
        env.step(toward / toward.norm(dim=-1, keepdim=True).clamp(min=1e-9))
        peak = max(peak, scenario.sheep_vel.norm(dim=-1).max().item())
    assert peak <= scenario.max_sheep_speed + 1e-6, f"speed cap leaked: {peak}"
    assert peak < scenario.max_speed, f"a sheep outran the shepherds: {peak}"


@pytest.mark.parametrize("device", DEVICES)
def test_shepherding_shaping_reward_positive_when_closer(device):
    """Global reward is positive on a step that moves a sheep toward the pen."""
    scenario = ShepherdingScenario(n_agents=3, n_sheep=1)
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset()
    scenario.pen_pos = torch.tensor([[0.8, 0.0]], device=device, dtype=torch.float32)
    scenario.sheep_pos = torch.tensor([[[0.0, 0.0]]], device=device, dtype=torch.float32)
    scenario._prev_dist = None
    scenario._refresh(integrate=False)
    # manually advance the sheep toward the pen and refresh (no integrate)
    scenario._prev_dist = scenario._cache["dist_to_pen"].unsqueeze(-1).detach().clone()
    scenario.sheep_pos = torch.tensor([[[0.1, 0.0]]], device=device, dtype=torch.float32)
    scenario._refresh(integrate=False)
    assert scenario.global_reward()[0].item() > 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_spawn_scale_shrinks_the_ring(device):
    """A small spawn scale must actually draw the sheep nearer the pen.

    Also the regression test for the cache this could quietly break: the reset launch is
    a ``CachedLaunch`` that packs its arguments once, so a scale change is only visible
    because ``_sheep_span``/``_sheep_min_r`` are entries in its key. Drop them from the
    key and this test fails while nothing else does — the curriculum would silently
    replay the constructor's spread forever.
    """
    scenario = ShepherdingScenario(n_agents=3, n_sheep=3)
    env = Environment(scenario, n_envs=64, device=device, dt=0.05, seed=1, fused=False)

    def pen_dist():
        return (scenario.sheep_pos - scenario.pen_pos.unsqueeze(1)).norm(dim=-1)

    env.reset(seed=5)
    wide = pen_dist().mean().item()
    scenario.set_spawn_scale(0.1)
    env.reset(seed=5)
    near = pen_dist().mean().item()
    assert near < wide, f"spawn scale had no effect ({wide} -> {near})"

    # scale=0 is the degenerate easiest stage: every sheep on the ring itself. The only
    # thing that can move it off is the world-bounds clamp, and that can only pull a
    # sheep *toward* an interior pen, never push it out.
    scenario.set_spawn_scale(0.0)
    env.reset(seed=5)
    ring = scenario.pen_radius + 2.0 * scenario.sheep_radius
    assert (pen_dist() <= ring + 1e-5).all()
    assert pen_dist().mean().item() < near

    with pytest.raises(ValueError, match="non-negative"):
        scenario.set_spawn_scale(-0.1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [False, True])
def test_spawn_scale_one_restores_the_constructor_draw(device, fused):
    """Annealing back to 1.0 must reproduce the original draw bit-for-bit.

    Bit-for-bit rather than merely close, because the scale rides on the *span* of an
    existing draw (``r = min_r + u * span``, ``span *= scale``) rather than adding or
    reordering an RNG call — so ``scale=1.0`` is a multiplication by exactly 1.0 and the
    whole stream is untouched. Run on the fused path too: there the reset kernel writes
    through the spec's cached Warp handles rather than a fresh ``wp.from_torch``.
    """
    scenario = ShepherdingScenario(n_agents=3, n_sheep=2)
    env = Environment(scenario, n_envs=32, device=device, dt=0.05, seed=1, fused=fused)
    env.reset(seed=7)
    ref = (scenario.sheep_pos.clone(), scenario.pen_pos.clone(), env.world.state.pos.clone())

    scenario.set_spawn_scale(0.2)
    env.reset(seed=7)
    assert not torch.equal(scenario.sheep_pos, ref[0])  # the easy stage really is different
    assert torch.equal(scenario.pen_pos, ref[1])  # ...but only the sheep ring moved
    assert torch.equal(env.world.state.pos, ref[2])

    scenario.set_spawn_scale(1.0)
    env.reset(seed=7)
    assert scenario.spawn_scale == 1.0
    assert torch.equal(scenario.sheep_pos, ref[0])
    assert torch.equal(scenario.pen_pos, ref[1])
    assert torch.equal(env.world.state.pos, ref[2])


def test_shepherding_differentiable_rollout():
    """BPTT: the sheep-to-pen loss backprops to the action sequence, through the flee force.

    The shepherds are placed 0.15 from the sheep — outside the 0.11 contact reach, inside
    the 0.35 flee radius — rather than left where the reset drew them. That is on purpose
    twice over: the loss only depends on the actions through *some* shepherd->sheep force,
    so a purely random spawn would make this a test of whether the seed happened to put
    someone in range; and at this offset the only such force is the flee term, so a
    gradient here is a gradient through the reactive part of the world.
    """
    scenario = ShepherdingScenario(n_agents=3, n_sheep=1)
    env = Environment(scenario, n_envs=2, device="cpu", dt=0.05, seed=0, fused=False)
    env.reset()
    near = scenario.sheep_pos[:, 0].unsqueeze(1) + torch.tensor([[0.15, 0.0]])
    env.world.write_state(None, pos=near.expand(-1, scenario.n_agents, -1).clone())
    actions = torch.zeros(2, 3, env.world.act_dim, requires_grad=True)
    loss = torch.zeros((), dtype=torch.float32)
    for _ in range(4):
        env.step(actions)
        loss = scenario._cache["dist_to_pen"].sum()
    loss.backward()
    assert actions.grad is not None and torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() > 0.0
