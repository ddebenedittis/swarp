"""Caging scenario: agents surround a drifting disc so it cannot escape."""

import math

import pytest
import torch
from conftest import DEVICES

from swarp import Environment
from swarp.scenarios.caging import CagingScenario

TWO_PI = 2.0 * math.pi


def _env(device, n_envs=4, n_agents=3, **kw):
    # fused=False on purpose: this suite exercises the *torch reference* disc integration
    # and gap reduction (``_refresh``), poking ``disc_pos`` directly and reading the
    # cache back. The fused parity of that path is what test_caging_fused.py is for.
    return Environment(
        CagingScenario(n_agents=n_agents, **kw),
        n_envs=n_envs,
        device=device,
        dt=0.05,
        seed=1,
        fused=False,
    )


def _ring(scenario, bearings_deg, radius=None, device="cpu"):
    """Place the agents at known bearings around the disc, disc parked at the origin."""
    radius = scenario.cage_radius if radius is None else radius
    scenario.disc_pos = torch.zeros_like(scenario.disc_pos)
    scenario.disc_vel = torch.zeros_like(scenario.disc_vel)
    pos = torch.tensor(
        [[[radius * math.cos(math.radians(b)), radius * math.sin(math.radians(b))]
          for b in bearings_deg]],
        device=device,
        dtype=torch.float32,
    )
    scenario.world.write_state(None, pos=pos)
    scenario._install_obstacles()
    scenario._refresh(integrate=False)


@pytest.mark.parametrize("device", DEVICES)
def test_caging_api_and_finiteness(device):
    env = _env(device, n_envs=8)
    obs = env.reset()
    assert obs.shape == (8, 3, 10) and torch.isfinite(obs).all()
    gen = torch.Generator(device=device).manual_seed(0)
    for _ in range(10):
        a = torch.rand(8, 3, env.world.act_dim, generator=gen, device=device) * 2 - 1
        obs, rew, term, trunc, info = env.step(a)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
        assert rew.shape == (8, 3) and term.shape == (8,) and trunc.shape == (8,)
        assert info["max_gap"].shape == (8,) and info["caged"].dtype == torch.bool
        # A gap is an angle on the circle: nothing may ever leave [0, 2*pi].
        assert (info["max_gap"] >= 0).all() and (info["max_gap"] <= TWO_PI + 1e-5).all()


@pytest.mark.parametrize("device", DEVICES)
def test_caging_determinism(device):
    def run():
        env = _env(device, n_envs=4)
        env.reset(seed=4)
        gen = torch.Generator(device=device).manual_seed(2)
        out = []
        for _ in range(8):
            a = torch.rand(4, 3, env.world.act_dim, generator=gen, device=device) * 2 - 1
            obs, rew, *_ = env.step(a)
            out += [obs, rew, env.scenario.disc_pos.clone()]
        return out

    for x, y in zip(run(), run(), strict=True):
        assert torch.equal(x, y)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("bearings", "expected_deg"),
    [
        ([0.0, 120.0, 240.0], 120.0),  # evenly spaced: the tightest a trio can cage
        ([0.0, 10.0, 20.0], 340.0),  # all bunched: the answer *is* the wrap-around gap
        ([350.0, 5.0, 170.0], 180.0),  # a pair straddling +/-pi, so the sort order wraps
    ],
)
def test_max_gap_closes_the_circle(device, bearings, expected_deg):
    """The torch oracle's sort + circular diff, checked against pen-and-paper.

    The wrap-around term — the step from the *last* sorted bearing back round to the
    first — is the one that is easy to leave out, and two of these three cases are about
    nothing else: leave it out and the bunched trio reports 10 degrees instead of 340.
    """
    scenario = CagingScenario(n_agents=len(bearings), world_size=1.0)
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset()
    _ring(scenario, bearings, device=device)
    got = math.degrees(scenario._cache["max_gap"][0].item())
    assert got == pytest.approx(expected_deg, abs=1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_surrounding_the_disc_closes_the_gap_and_raises_reward(device):
    """A closed ring beats the same agents bunched on one side — the whole point.

    Both configurations sit at exactly ``cage_radius``, so the ring shaping term is
    identical between them and the *only* thing that differs is the topology.
    """
    scenario = CagingScenario(n_agents=5, world_size=1.0)
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset()

    _ring(scenario, [0.0, 72.0, 144.0, 216.0, 288.0], device=device)
    caged_gap = scenario._cache["max_gap"][0].item()
    caged_reward = scenario.global_reward()[0].item()
    assert scenario.done()[0].item()  # gap below threshold, all within capture radius

    _ring(scenario, [0.0, 20.0, 40.0, 60.0, 80.0], device=device)
    bunched_gap = scenario._cache["max_gap"][0].item()
    bunched_reward = scenario.global_reward()[0].item()
    assert not scenario.done()[0].item()

    assert caged_gap == pytest.approx(math.radians(72.0), abs=1e-4)
    assert bunched_gap == pytest.approx(math.radians(280.0), abs=1e-4)
    assert caged_reward > bunched_reward


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_agents", [1, 8])
def test_the_disc_cannot_outrun_its_captors(device, n_agents):
    """The disc must never move faster than an agent, however many of them press.

    This decides whether the task is *learnable* rather than whether any single number is
    right, and violating it fails nothing else: a disc that outruns its captors can never
    be surrounded, so ``caged`` is unreachable and a run just plateaus. It was real — with
    ``escape_accel=0.5`` the sustained escape speed equalled ``max_speed`` exactly and
    contact transients peaked at 2.9x it, and a 4000-iteration MAPPO run drove ``max_gap``
    from 3.21 to 5.83 with ``caged`` at ~0.0002 throughout.

    The policy is the adversarial one: every agent drives at full speed straight at the
    disc, which both maximizes the escape drift (they all pile onto one side, so the mean
    unit bearing has no cancellation left) and maximizes the contact-transient spike.
    Parametrized over the extremes because the two bounds have different causes — the
    mean-of-unit-vectors drift for the pack, the ``max_disc_speed`` cap for the
    single-agent ram, which no drift bound can catch.
    """
    scenario = CagingScenario(n_agents=n_agents)
    env = Environment(scenario, n_envs=8, device=device, dt=0.05, seed=0, fused=False)
    env.reset(seed=0)
    peak = 0.0
    for _ in range(300):
        rel = scenario.disc_pos[:, 0].unsqueeze(1) - env.world.state.pos  # agent -> disc
        env.step(rel / rel.norm(dim=-1, keepdim=True).clamp(min=1e-9))
        peak = max(peak, scenario.disc_vel.norm(dim=-1).max().item())
    assert peak <= scenario.max_disc_speed + 1e-6, f"speed cap leaked: {peak}"
    assert peak < scenario.max_speed, f"the disc outran its captors: {peak}"


def test_a_scripted_even_spacing_policy_actually_cages_the_disc():
    """Solvability: a trivial hand-written policy must be able to close the cage.

    Every agent is driven at its own even-spacing slot on the ``cage_radius`` ring around
    the *live* disc position — no learning, one P-controller, privileged access to the
    disc pose. If that cannot get ``max_gap`` under ``gap_threshold`` then no policy can,
    and the parameters are wrong however good the reward looks. This is the check that
    catches an escape law the agents cannot outrun and a ``gap_threshold`` set below what
    the agent count can geometrically reach.

    ``n_agents=3`` is the tight case and the reason it is parametrized here: three agents
    can do no better than ``2*pi/3`` (2.094 rad) even played perfectly, so it has the
    least margin against the 2.4 threshold of any configuration.
    """
    for n_agents, floor in ((3, 2 * math.pi / 3), (5, 2 * math.pi / 5)):
        scenario = CagingScenario(n_agents=n_agents)
        env = Environment(scenario, n_envs=64, device="cpu", dt=0.05, seed=0)
        env.reset(seed=0)
        phase = torch.arange(n_agents, dtype=torch.float32) * (2 * math.pi / n_agents)
        slot = torch.stack([phase.cos(), phase.sin()], -1) * scenario.cage_radius  # [A, 2]
        with torch.no_grad():
            for _ in range(200):
                target = scenario.disc_pos[:, 0].unsqueeze(1) + slot
                err = target - env.world.state.pos
                *_, info = env.step((6.0 * err).clamp(-1.0, 1.0))
        caged = info["caged"].float().mean().item()
        gap = info["max_gap"].mean().item()
        assert caged > 0.9, f"n_agents={n_agents}: only {caged:.1%} of envs caged the disc"
        assert gap < scenario.gap_threshold, f"n_agents={n_agents}: max_gap {gap}"
        # ...and it settles near the geometric optimum, not merely under the threshold.
        # One-sided: ``floor`` is a hard lower bound no arrangement of n agents can beat,
        # and the slack above it is the P-controller's lag on a still-drifting disc.
        assert floor <= gap < floor + 0.3, f"n_agents={n_agents}: max_gap {gap}"


@pytest.mark.parametrize("device", DEVICES)
def test_even_spacing_beats_clumping_for_every_agent_at_the_same_radius(device):
    """The dense shaping term's reason to exist, pinned as a property.

    ``gap_reward``/``caged_reward`` are zeroed so the shared term is a constant across
    both configurations, and both rings sit at exactly ``cage_radius`` so the radial and
    contact terms are identical too. What is left is the spacing term alone — and it has
    to favour the even ring for **every** agent, not just in the sum: the whole point is
    that a max-based angular signal reaches only the two agents bordering the widest gap,
    so the replacement has to be dense.
    """
    scenario = CagingScenario(n_agents=5, world_size=1.0, gap_reward=0.0, caged_reward=0.0)
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset()

    _ring(scenario, [0.0, 72.0, 144.0, 216.0, 288.0], device=device)
    even = scenario.rewards()[0].clone()
    _ring(scenario, [0.0, 20.0, 40.0, 60.0, 80.0], device=device)
    clumped = scenario.rewards()[0].clone()

    assert (even > clumped).all(), "clumping is not penalized for every agent"
    # The even ring is the term's exact optimum, so it costs nothing at all there.
    assert even.abs().max().item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_spacing_shaping_is_off_when_its_weight_is_zero(device):
    """``spacing_factor=0`` must recover the unshaped reward exactly, so the term is
    auditable in isolation (and a training run can ablate it without a code change)."""
    rewards = []
    for factor in (0.0, CagingScenario().spacing_factor):
        scenario = CagingScenario(n_agents=5, world_size=1.0, spacing_factor=factor)
        env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
        env.reset()
        _ring(scenario, [0.0, 20.0, 40.0, 60.0, 80.0], device=device)
        rewards.append(scenario.rewards()[0].clone())
    off, on = rewards
    assert (on < off).all()  # the clumped ring is strictly penalized once it is on


@pytest.mark.parametrize("device", DEVICES)
def test_a_gap_in_the_ring_lets_the_disc_escape_through_it(device):
    """The escape drift is what makes the objective topological rather than metric.

    Five agents spread over the upper half only leave a hole pointing at -y, and the
    disc has to come out through it; the same five spread evenly cancel the drift and it
    stays put. Both rings sit at ``cage_radius``, which is outside contact range, so this
    is the escape law on its own with no contact force muddying it.
    """
    def drift(bearings):
        scenario = CagingScenario(n_agents=5, world_size=1.0)
        env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
        env.reset()
        _ring(scenario, bearings, device=device)
        hold = torch.zeros(1, 5, env.world.act_dim, device=device)
        # 25 steps rather than 10: ``escape_accel`` is deliberately only half of what a
        # sustained chase can close at (see ``max_disc_speed``), so the drift needs a
        # slightly longer horizon to be unambiguous. It reaches -0.086 here.
        for _ in range(25):
            env.step(hold)
        return scenario.disc_pos[0, 0].detach().clone()

    open_ring = drift([0.0, 45.0, 90.0, 135.0, 180.0])  # hole centred on -y
    closed_ring = drift([0.0, 72.0, 144.0, 216.0, 288.0])
    assert open_ring[1].item() < -0.05, "disc did not escape through the hole"
    assert abs(open_ring[0].item()) < abs(open_ring[1].item())  # and it went *down*
    assert closed_ring.norm().item() < 0.2 * open_ring.norm().item()


def test_caging_differentiable_rollout():
    """BPTT: the largest-circular-gap loss backprops to the action sequence.

    ``atan2`` -> ``sort`` -> ``max`` is differentiable, so the topological objective
    itself carries gradient — no contact with the disc is needed for this one, unlike
    transport's package loss.
    """
    scenario = CagingScenario(n_agents=3)
    env = Environment(scenario, n_envs=2, device="cpu", dt=0.05, seed=0, fused=False)
    env.reset()
    actions = torch.zeros(2, 3, env.world.act_dim, requires_grad=True)
    loss = torch.zeros((), dtype=torch.float32)
    for _ in range(4):
        env.step(actions)
        loss = scenario._cache["max_gap"].sum()
    loss.backward()
    assert actions.grad is not None and torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() > 0.0
