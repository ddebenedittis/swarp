"""Shepherding scenario: shepherds herd fleeing sheep into a pen.

The redesign this suite tracks fixed a task that was **geometrically infeasible**: the
shipped ``pen_radius``/``sep_radius``/spawn combination made ``all_penned`` require the
flock centroid within 0.003 of the pen centre, a scripted Strombom controller solved
~0% of episodes, and a 3000-iteration MAPPO run learned nothing -- while this suite's
predecessor stayed green throughout, because nothing in it checked that the task was
*solvable* at all. ``test_the_flock_fits_in_the_pen`` and
``test_a_scripted_shepherd_solves_most_episodes`` below are what closes that hole; see
``swarp/scenarios/shepherding.py``'s module docstring for the full measurement.
"""

import math

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
    assert obs.shape == (8, 3, env.scenario.obs_dim) and torch.isfinite(obs).all()
    gen = torch.Generator(device=device).manual_seed(0)
    for _ in range(10):
        a = torch.rand(8, 3, env.world.act_dim, generator=gen, device=device) * 2 - 1
        obs, rew, term, trunc, info = env.step(a)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
        assert rew.shape == (8, 3) and term.shape == (8,) and trunc.shape == (8,)
        assert info["sheep_penned"].shape == (8,) and info["sheep_dist_to_pen"].shape == (8,)
        assert ((info["sheep_penned"] >= 0.0) & (info["sheep_penned"] <= 1.0)).all()
        # New info keys: the trainer auto-discovers info entries, so a NaN or an
        # unshaped tensor here silently corrupts logged metrics rather than raising.
        assert info["flock_radius"].shape == (8,) and torch.isfinite(info["flock_radius"]).all()
        assert info["all_penned"].shape == (8,) and torch.isfinite(info["all_penned"]).all()


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
    # sep_gain * (1 - d/sep_radius) == cohesion_gain * d/2, i.e. d ~= 0.145 at the
    # redesign's sep_radius=0.15 (measured: the two sheep settle at radius 0.0723 each
    # about their centroid). The upper bound pins that cohesion is what stops them
    # drifting apart forever, not that the number moved.
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
    """Global reward is positive on a step that moves a sheep toward the pen.

    The reward now sums three potentials (position, gather, penned), each seeded from a
    ``None`` baseline on its first read (see ``_refresh``) -- seeding only ``_prev_dist``
    and leaving the other two ``None`` across the manual "advance the sheep" step would
    make the *second* ``_refresh`` call re-seed them against the moved sheep instead of
    the original one, silently zeroing terms this test isn't about rather than actually
    isolating the position term it is.
    """
    scenario = ShepherdingScenario(n_agents=3, n_sheep=1)
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset()
    scenario.pen_pos = torch.tensor([[0.8, 0.0]], device=device, dtype=torch.float32)
    scenario.sheep_pos = torch.tensor([[[0.0, 0.0]]], device=device, dtype=torch.float32)
    scenario._prev_dist = None
    scenario._prev_radius = None
    scenario._prev_penned = None
    scenario._refresh(integrate=False)
    # manually advance the sheep toward the pen and refresh (no integrate), seeding all
    # three baselines from the state just measured so the second refresh isolates the
    # shaping delta rather than also re-triggering the gather/penned potentials.
    scenario._prev_dist = scenario._cache["dist_to_pen"].unsqueeze(-1).detach().clone()
    scenario._prev_radius = scenario._cache["mean_radius"].detach().clone()
    scenario._prev_penned = scenario._cache["penned_count"].detach().clone()
    scenario.sheep_pos = torch.tensor([[[0.1, 0.0]]], device=device, dtype=torch.float32)
    scenario._refresh(integrate=False)
    assert scenario.global_reward()[0].item() > 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_flock_distance_moves_the_flock(device):
    """A smaller flock-distance scale must actually draw the flock nearer the pen.

    Also the regression test for the cache this could quietly break: the reset launch is
    a ``CachedLaunch`` that packs its arguments once, so a scale change is only visible
    because ``_flock_dmin``/``_flock_span`` are entries in its key. Drop them from the
    key and this test fails while nothing else does — a curriculum change would silently
    replay the constructor's distance forever. Compared as **batch means**, not per-env:
    each ``reset()`` draws a fresh RNG stream, so a scale change is only guaranteed to
    move the distribution, not every individual env's draw.
    """
    scenario = ShepherdingScenario(n_agents=3, n_sheep=3)
    env = Environment(scenario, n_envs=64, device=device, dt=0.05, seed=1, fused=False)

    def pen_dist():
        return (scenario.sheep_pos - scenario.pen_pos.unsqueeze(1)).norm(dim=-1).mean().item()

    env.reset(seed=5)
    far = pen_dist()
    scenario.set_flock_distance(0.1)
    env.reset(seed=5)
    near = pen_dist()
    assert near < far, f"flock distance scale had no effect ({far} -> {near})"

    with pytest.raises(ValueError, match="non-negative"):
        scenario.set_flock_distance(-0.1)


@pytest.mark.parametrize("device", DEVICES)
def test_flock_spread_scales_the_cluster(device):
    """A small spread scale must tighten the flock's own dispersion without moving it.

    Distance and spread are supposed to be independent curriculum axes (module
    docstring: "close and already collected" vs. "close but scattered" need to vary
    separately) — this is what would catch ``_set_spawn_geometry`` coupling them, e.g.
    by letting the ``reach`` clamp on one leak into the other.
    """
    scenario = ShepherdingScenario(n_agents=3, n_sheep=5)
    env = Environment(scenario, n_envs=64, device=device, dt=0.05, seed=1, fused=False)

    def flock_radius():
        centroid = scenario.sheep_pos.mean(dim=1, keepdim=True)
        return (scenario.sheep_pos - centroid).norm(dim=-1).mean().item()

    def pen_dist():
        return (scenario.sheep_pos - scenario.pen_pos.unsqueeze(1)).norm(dim=-1).mean().item()

    env.reset(seed=5)
    wide_radius, wide_dist = flock_radius(), pen_dist()
    scenario.set_flock_spread(0.1)
    env.reset(seed=5)
    tight_radius, tight_dist = flock_radius(), pen_dist()

    assert tight_radius < wide_radius, (
        f"spread scale had no effect ({wide_radius} -> {tight_radius})"
    )
    # "Nearly irrelevant" (module docstring), not exactly zero: measured at these knobs,
    # the radius collapses ~90% while the pen distance moves ~18% (the reach clamp in
    # ``_set_spawn_geometry`` couples them a little at the extremes) -- 0.3 is loose
    # enough to pass that coupling and still catch spread actually driving the distance.
    assert abs(tight_dist - wide_dist) < 0.3 * wide_dist, (
        f"spread scale moved the flock's distance to the pen too much ({wide_dist} -> {tight_dist})"
    )

    with pytest.raises(ValueError, match="non-negative"):
        scenario.set_flock_spread(-0.1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [False, True])
def test_flock_scales_of_one_restore_the_constructor_draw(device, fused):
    """Annealing both curriculum knobs back to 1.0 must reproduce the original draw
    bit-for-bit.

    Bit-for-bit rather than merely close, because each scale rides on the *span* of an
    existing draw (``dmin + u * span``) rather than adding or reordering an RNG call —
    so ``scale=1.0`` is a multiplication by exactly 1.0 and the whole stream is
    untouched. Run on the fused path too: there the reset kernel writes through the
    spec's cached Warp handles rather than a fresh ``wp.from_torch``. Also pins that
    neither knob touches the shepherd or pen draws — the RNG ordering was designed so
    only the flock-centre and per-sheep draws move.
    """
    scenario = ShepherdingScenario(n_agents=3, n_sheep=2)
    env = Environment(scenario, n_envs=32, device=device, dt=0.05, seed=1, fused=fused)
    env.reset(seed=7)
    ref = (scenario.sheep_pos.clone(), scenario.pen_pos.clone(), env.world.state.pos.clone())

    scenario.set_flock_distance(0.2)
    env.reset(seed=7)
    assert not torch.equal(scenario.sheep_pos, ref[0])  # the easy stage really is different
    assert torch.equal(scenario.pen_pos, ref[1])  # ...but only the flock moved
    assert torch.equal(env.world.state.pos, ref[2])

    scenario.set_flock_distance(1.0)
    env.reset(seed=7)
    assert scenario.flock_dist_scale == 1.0
    assert torch.equal(scenario.sheep_pos, ref[0])
    assert torch.equal(scenario.pen_pos, ref[1])
    assert torch.equal(env.world.state.pos, ref[2])

    scenario.set_flock_spread(0.2)
    env.reset(seed=7)
    assert not torch.equal(scenario.sheep_pos, ref[0])
    assert torch.equal(scenario.pen_pos, ref[1])
    assert torch.equal(env.world.state.pos, ref[2])

    scenario.set_flock_spread(1.0)
    env.reset(seed=7)
    assert scenario.flock_spread_scale == 1.0
    assert torch.equal(scenario.sheep_pos, ref[0])
    assert torch.equal(scenario.pen_pos, ref[1])
    assert torch.equal(env.world.state.pos, ref[2])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_sheep", [3, 5, 7])
def test_the_flock_fits_in_the_pen(device, n_sheep):
    """Three lines of physics that would have caught the original bug outright.

    The shipped task sized ``pen_radius`` below the free flock's own relaxed size, so
    ``all_penned`` was unreachable regardless of what the shepherds did (module
    docstring §0: measured ``terminated`` of 0.0016). This drops the flock near the
    origin with every shepherd held past ``flee_radius`` (rewritten every step, so an
    action cannot drag one back in) and checks the free-relaxation radius directly
    against ``pen_radius``, with no shepherd or reward involved at all.

    Measured equilibria at ``sep_radius=0.15``: 0.085 / 0.150 / 0.145 for
    ``n_sheep in (3, 5, 7)``, against the 0.2 bound (``pen_radius - 0.1``). 800 steps
    matters: at 400 one random start in 64 was still at 0.209 and had not converged —
    do not lower it without re-measuring. ``n_envs`` stays small; this is the slowest
    test in the file.
    """
    scenario = ShepherdingScenario(n_agents=3, n_sheep=n_sheep, world_size=2.0)
    n_envs = 4
    env = Environment(scenario, n_envs=n_envs, device=device, dt=0.05, seed=0, fused=False)
    env.reset()

    angles = torch.linspace(0, 2 * math.pi, n_sheep + 1, device=device)[:-1]
    pos0 = 0.05 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # [K, 2]
    scenario.sheep_pos = pos0.unsqueeze(0).expand(n_envs, -1, -1).clone().to(torch.float32)
    scenario.sheep_vel = torch.zeros_like(scenario.sheep_pos)
    far = torch.full((n_envs, 3, 2), 5.0, device=device, dtype=torch.float32)  # >> flee_radius
    env.world.write_state(None, pos=far, vel=torch.zeros_like(far))
    scenario._install_obstacles()
    scenario._prev_dist = None
    scenario._prev_radius = None
    scenario._prev_penned = None

    zero_actions = torch.zeros(n_envs, 3, env.world.act_dim, device=device)
    for _ in range(800):
        env.step(zero_actions)
        env.world.write_state(None, pos=far, vel=torch.zeros_like(far))

    gcm = scenario.sheep_pos.mean(dim=1, keepdim=True)
    max_r = (scenario.sheep_pos - gcm).norm(dim=-1).max().item()
    assert max_r < scenario.pen_radius - 0.1, (
        f"free flock (n_sheep={n_sheep}) settles wider than the pen can cover: {max_r}"
    )


def _unit(v, floor=1e-6):
    n = v.norm(dim=-1, keepdim=True)
    return torch.where(n >= floor, v / n.clamp(min=floor), torch.zeros_like(v))


def _scripted(obs, sep_radius, n_sheep):
    """Strombom collect/drive/retreat, straight out of the observation row."""
    own = obs[..., 0:2]  # [E, A, 2]
    gcm = obs[..., 6:8] + own
    pen = obs[..., 8:10] + gcm
    max_r = obs[..., 13]
    drive = obs[..., 16:18] + own
    collect = obs[..., 18:20] + own
    away = _unit(gcm - pen)  # pen -> flock
    lateral = torch.stack([-away[..., 1], away[..., 0]], -1)  # abreast, not stacked
    a = torch.arange(own.shape[1], device=obs.device, dtype=obs.dtype)
    off = ((a - (own.shape[1] - 1) / 2) * 0.25).view(1, -1, 1)
    f_n = sep_radius * n_sheep ** (2.0 / 3.0)  # Strombom's collect/drive switch
    goal = torch.where(
        (max_r > f_n).unsqueeze(-1),
        collect + off * lateral * 0.5,
        drive + off * lateral,
    )
    # Retreat past flee_radius once the flock is on the pen, so it can relax inside it.
    # Without this the shepherds park at the drive point and hold the flock inflated
    # past the pen diameter: solve rate drops from 0.84 to 0.37.
    near = ((gcm - pen).norm(dim=-1) < 0.13).unsqueeze(-1)
    goal = torch.where(near, gcm + 0.55 * _unit(own - gcm), goal)
    return _unit(goal - own)


@pytest.mark.parametrize("device", DEVICES)
def test_a_scripted_shepherd_solves_most_episodes(device):
    """The feasibility gate, as a CI-checked property rather than a claim in a doc.

    The shipped task was geometrically infeasible (module docstring §0): a scripted
    Strombom policy solved ~0% of episodes, and a 3000-iteration MAPPO training run
    learned nothing, while the test suite that shipped alongside it stayed green — it
    never asked whether the task was solvable at all. This closes that hole by driving
    the observation through a fixed scripted controller (it reads *only* the
    observation, proving the Strombom hint slots -- ``drive_pt``/``collect_pt`` -- are
    actually sufficient for the task, not merely present) and requiring it to clear a
    solve-rate floor a "do nothing" policy cannot.

    Measured over seeds 0/1/2 on both cpu and cuda: scripted solves 0.766 / 0.875 /
    0.766 of episodes, do-nothing solves 0.0 everywhere (the cluster spawn is not
    degenerate the way the old ring spawn was), and a uniform-random policy solves
    <= 0.016. The 0.6 floor leaves comfortable margin under the worst measured seed.
    """
    n_envs, n_steps, seed = 64, 200, 0

    def solve_rate(policy):
        scenario = ShepherdingScenario()  # full difficulty, default 3 shepherds / 5 sheep
        env = Environment(scenario, n_envs=n_envs, device=device, dt=0.05, seed=seed, fused=True)
        obs = env.reset(seed=seed)
        ever_done = torch.zeros(n_envs, dtype=torch.bool, device=device)
        with torch.no_grad():
            for _ in range(n_steps):
                obs, rew, term, trunc, info = env.step(policy(obs, scenario))
                ever_done |= term
        return ever_done.float().mean().item()

    scripted_rate = solve_rate(
        lambda obs, scenario: _scripted(obs, scenario.sep_radius, scenario.n_sheep)
    )
    idle_rate = solve_rate(
        lambda obs, scenario: torch.zeros(n_envs, scenario.n_agents, 2, device=device)
    )
    assert scripted_rate >= 0.6, f"scripted solve rate collapsed: {scripted_rate}"
    assert idle_rate == 0.0, f"doing nothing solves episodes: {idle_rate}"


@pytest.mark.parametrize("device", DEVICES)
def test_a_parked_flock_stays_put(device):
    """``done`` is a stable state, not a knife edge that decays the instant it's reached.

    Cohesion sums to zero over the flock (it pulls every sheep toward the mean, which
    is a no-op on the mean itself) and separation is pairwise antisymmetric, so with no
    shepherd within ``flee_radius`` the flock centroid is provably conserved. Measured
    drift is ~1e-7 over 400 steps, so the 1e-3 bound here has enormous margin — a
    genuine regression (a stray asymmetric term, an off-by-one in the pairwise sum)
    would blow well past it, not creep up to it.
    """
    scenario = ShepherdingScenario(n_agents=3, n_sheep=5, world_size=2.0)
    n_envs = 4
    env = Environment(scenario, n_envs=n_envs, device=device, dt=0.05, seed=0, fused=False)
    env.reset()
    far = torch.full((n_envs, 3, 2), 5.0, device=device, dtype=torch.float32)
    env.world.write_state(None, pos=far, vel=torch.zeros_like(far))
    scenario._install_obstacles()
    start = scenario.sheep_pos.mean(dim=1).clone()

    zero_actions = torch.zeros(n_envs, 3, env.world.act_dim, device=device)
    for _ in range(200):
        env.step(zero_actions)
        env.world.write_state(None, pos=far, vel=torch.zeros_like(far))

    drift = (scenario.sheep_pos.mean(dim=1) - start).norm(dim=-1).max().item()
    assert drift < 1e-3, f"a parked flock's centroid drifted: {drift}"


@pytest.mark.parametrize("device", DEVICES)
def test_penning_a_sheep_pays_once(device):
    """The direct regression test for the diagnosed local optimum.

    The old level reward paid ``pen_reward`` every step a sheep sat in the pen, worth
    up to 375 over an episode for "park two sheep and stop" against a shaping budget of
    ~2.5 — a policy that finds that is never going to bother finishing. As a potential
    it should pay exactly once, on the step a sheep *enters* the pen, and nothing on
    every step after. Every other reward coefficient is zeroed so the isolated
    ``pen_reward * (penned - prev_penned)`` term is the only thing that can move the
    total off zero.
    """
    scenario = ShepherdingScenario(
        n_agents=1,
        n_sheep=1,
        pos_shaping_factor=0.0,
        gather_factor=0.0,
        done_reward=0.0,
        side_factor=0.0,
        crowd_factor=0.0,
        intrude_penalty=0.0,
    )
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset()
    scenario.pen_pos = torch.zeros(1, 2, device=device, dtype=torch.float32)
    scenario.sheep_pos = torch.tensor([[[0.9, 0.0]]], device=device, dtype=torch.float32)  # outside
    scenario.sheep_vel = torch.zeros_like(scenario.sheep_pos)
    far = torch.tensor([[[5.0, 5.0]]], device=device, dtype=torch.float32)  # outside flee_radius
    env.world.write_state(None, pos=far, vel=torch.zeros_like(far))
    scenario._install_obstacles()
    scenario._prev_dist = None
    scenario._prev_radius = None
    scenario._prev_penned = None
    scenario._refresh(integrate=False)  # baseline: not penned

    scenario.sheep_pos = torch.zeros(1, 1, 2, device=device, dtype=torch.float32)  # now in the pen
    scenario._install_obstacles()
    zero_actions = torch.zeros(1, 1, env.world.act_dim, device=device)
    totals = []
    for _ in range(10):
        env.step(zero_actions)
        env.world.write_state(None, pos=far, vel=torch.zeros_like(far))
        totals.append(scenario.global_reward()[0].item())

    assert totals[0] > 0.0, "entering the pen should pay the potential once"
    assert sum(totals[1:]) == pytest.approx(0.0, abs=1e-9), (
        "the pen reward kept paying every step it stayed penned; the level bug is back"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_obs_flock_features_match_definitions(device):
    """Recompute every flock feature in plain torch and compare to the obs slots.

    This is what keeps ``test_fused_matches_torch`` in ``test_shepherding_fused.py``
    meaningful: that test only proves the fused and torch paths *agree*. Without this
    one, both could agree on a wrong ``gcm``, ``stray`` or ``collect_pt`` and nothing
    would notice.
    """
    n_agents, n_sheep = 3, 5
    scenario = ShepherdingScenario(n_agents=n_agents, n_sheep=n_sheep)
    env = Environment(scenario, n_envs=4, device=device, dt=0.05, seed=3, fused=False)
    env.reset(seed=3)
    gen = torch.Generator(device=device).manual_seed(1)
    obs = None
    for _ in range(5):
        a = torch.rand(4, n_agents, env.world.act_dim, generator=gen, device=device) * 2 - 1
        obs, *_ = env.step(a)

    own_pos = env.world.state.pos
    q, v, pen = scenario.sheep_pos, scenario.sheep_vel, scenario.pen_pos

    gcm = q.mean(dim=1)
    gcm_vel = v.mean(dim=1)
    r_k = (q - gcm.unsqueeze(1)).norm(dim=-1)
    mean_r = r_k.mean(dim=-1)
    max_r, stray_idx = r_k.max(dim=-1)
    stray = q.gather(1, stray_idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)

    def safe_unit(vec, floor=scenario.dir_floor):
        n = vec.norm(dim=-1, keepdim=True)
        return torch.where(n >= floor, vec / n.clamp(min=floor), torch.zeros_like(vec))

    drive_pt = gcm + scenario._drive_offset * safe_unit(gcm - pen)
    collect_pt = stray + scenario._collect_offset * safe_unit(stray - gcm)

    gcm_rel = gcm.unsqueeze(1) - own_pos
    pen_gcm = (pen - gcm).unsqueeze(1).expand(-1, n_agents, -1)
    gcm_vel_b = gcm_vel.unsqueeze(1).expand(-1, n_agents, -1)
    mean_r_b = mean_r.view(-1, 1).expand(-1, n_agents)
    max_r_b = max_r.view(-1, 1).expand(-1, n_agents)
    stray_rel = stray.unsqueeze(1).expand(-1, n_agents, -1) - own_pos
    drive_rel = drive_pt.unsqueeze(1).expand(-1, n_agents, -1) - own_pos
    collect_rel = collect_pt.unsqueeze(1).expand(-1, n_agents, -1) - own_pos

    torch.testing.assert_close(obs[..., 6:8], gcm_rel)
    torch.testing.assert_close(obs[..., 8:10], pen_gcm)
    torch.testing.assert_close(obs[..., 10:12], gcm_vel_b)
    torch.testing.assert_close(obs[..., 12], mean_r_b)
    torch.testing.assert_close(obs[..., 13], max_r_b)
    torch.testing.assert_close(obs[..., 14:16], stray_rel)
    torch.testing.assert_close(obs[..., 16:18], drive_rel)
    torch.testing.assert_close(obs[..., 18:20], collect_rel)


@pytest.mark.parametrize("device", DEVICES)
def test_gather_shaping_rewards_tightening(device):
    """The collect half of the reward: contracting the flock about its own centroid
    must pay, and pay nothing on the reset step that seeds the baseline.

    The shipped reward never priced this in at all — only driving the centroid toward
    the pen was rewarded — so a policy had no signal to actually collect a scattered
    flock before driving it. Every other coefficient is zeroed so the isolated
    ``gather_factor * (prev_mean_R - mean_R)`` term is the only thing that can move the
    total off zero.
    """
    scenario = ShepherdingScenario(
        n_agents=1,
        n_sheep=3,
        pos_shaping_factor=0.0,
        pen_reward=0.0,
        done_reward=0.0,
        side_factor=0.0,
        crowd_factor=0.0,
        intrude_penalty=0.0,
    )
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset()
    # a pen far away: the sheep is never accidentally penned by this test
    scenario.pen_pos = torch.tensor([[5.0, 5.0]], device=device, dtype=torch.float32)
    scenario.sheep_pos = torch.tensor(
        [[[0.3, 0.0], [-0.15, 0.26], [-0.15, -0.26]]], device=device, dtype=torch.float32
    )
    scenario.sheep_vel = torch.zeros_like(scenario.sheep_pos)
    far = torch.tensor([[[5.0, -5.0]]], device=device, dtype=torch.float32)
    env.world.write_state(None, pos=far, vel=torch.zeros_like(far))
    scenario._install_obstacles()
    scenario._prev_dist = None
    scenario._prev_radius = None
    scenario._prev_penned = None
    reset_mask = torch.ones(1, dtype=torch.bool, device=device)
    scenario._refresh(reset_mask=reset_mask, integrate=False)
    assert scenario.global_reward()[0].item() == pytest.approx(0.0)  # a reset step earns nothing

    scenario.sheep_pos = scenario.sheep_pos * 0.5  # contract the flock about its centroid
    scenario._refresh(integrate=False)
    assert scenario.global_reward()[0].item() > 0.0, (
        "tightening the flock did not pay the gather term"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_done_requires_the_flock_to_hold(device):
    """``done`` needs ``pen_hold`` *consecutive* fully-penned steps, and the terminal
    bonus is paid exactly once when that latch trips.

    A single instantaneous ``all_penned`` used to be enough; requiring a hold makes a
    lucky pass-through insufficient and rewards actually cornering the flock. Every
    sheep starts already in the pen (frozen, shepherd parked past ``flee_radius``) so
    the hold counter's only input is the step count itself.
    """
    scenario = ShepherdingScenario(n_agents=1, n_sheep=2, pen_hold=5)
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset()
    scenario.pen_pos = torch.zeros(1, 2, device=device, dtype=torch.float32)
    scenario.sheep_pos = torch.zeros(1, 2, 2, device=device, dtype=torch.float32)
    scenario.sheep_vel = torch.zeros_like(scenario.sheep_pos)
    far = torch.tensor([[[5.0, 5.0]]], device=device, dtype=torch.float32)
    env.world.write_state(None, pos=far, vel=torch.zeros_like(far))
    scenario._install_obstacles()
    scenario._prev_dist = None
    scenario._prev_radius = None
    scenario._prev_penned = None
    scenario._hold = None

    zero_actions = torch.zeros(1, 1, env.world.act_dim, device=device)
    dones, rewards = [], []
    for _ in range(scenario.pen_hold + 2):
        env.step(zero_actions)
        env.world.write_state(None, pos=far, vel=torch.zeros_like(far))
        dones.append(bool(scenario.done()[0].item()))
        rewards.append(scenario.global_reward()[0].item())

    assert dones[: scenario.pen_hold - 1] == [False] * (scenario.pen_hold - 1)
    assert dones[scenario.pen_hold - 1], "done did not trip on the pen_hold-th held step"
    spikes = sum(1 for r in rewards if r >= scenario.done_reward - 1e-6)
    assert spikes == 1, f"done_reward paid {spikes} times instead of exactly once"


@pytest.mark.parametrize("device", DEVICES)
def test_shepherds_observe_each_other(device):
    """Catches an off-by-one in the ascending-excluding-self teammate block.

    Each shepherd's row must see the *other* two, in ascending index order, as relative
    position and relative velocity -- easy to get backwards (including self, or
    dropping the wrong index) since the gather trick indexes past the diagonal.
    """
    n_agents, n_sheep = 3, 2
    scenario = ShepherdingScenario(n_agents=n_agents, n_sheep=n_sheep)
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset()
    pos = torch.tensor([[[0.0, 0.0], [0.3, 0.0], [0.0, 0.3]]], device=device, dtype=torch.float32)
    vel = torch.tensor([[[0.1, 0.0], [0.0, 0.1], [-0.1, 0.0]]], device=device, dtype=torch.float32)
    env.world.write_state(None, pos=pos, vel=vel)
    scenario._install_obstacles()
    scenario._prev_dist = None
    scenario._prev_radius = None
    scenario._prev_penned = None
    scenario._refresh(integrate=False)
    obs = scenario.observations()

    off, width = 20, 4  # rel_pos(2) + rel_vel(2) per other shepherd
    for a in range(n_agents):
        for slot, j in enumerate(j for j in range(n_agents) if j != a):
            rel_pos = obs[0, a, off + slot * width : off + slot * width + 2]
            rel_vel = obs[0, a, off + slot * width + 2 : off + slot * width + 4]
            torch.testing.assert_close(rel_pos, pos[0, j] - pos[0, a])
            torch.testing.assert_close(rel_vel, vel[0, j] - vel[0, a])


@pytest.mark.parametrize("device", DEVICES)
def test_a_shepherd_in_the_pen_is_penalized(device):
    """A real failure mode, not hygiene: every other term pulls a shepherd toward the
    pen, and one parked *in* it flees the sheep straight back out, making ``done``
    unreachable for as long as it sits there.

    Two envs share an identical sheep/pen configuration (so the shared reward term is
    identical) and differ only in whether their one shepherd sits inside or outside the
    pen disc; ``side_factor`` is zeroed so the per-agent difference is exactly the
    ``intrude_penalty`` term this test is about.
    """
    scenario = ShepherdingScenario(n_agents=1, n_sheep=1, side_factor=0.0)
    env = Environment(scenario, n_envs=2, device=device, dt=0.05, seed=0, fused=False)
    env.reset()
    scenario.pen_pos = torch.zeros(2, 2, device=device, dtype=torch.float32)
    scenario.sheep_pos = torch.tensor(
        [[[0.6, 0.0]], [[0.6, 0.0]]], device=device, dtype=torch.float32
    )
    scenario.sheep_vel = torch.zeros_like(scenario.sheep_pos)
    # env 0: shepherd inside the pen (pen_radius=0.3); env 1: otherwise identical, outside it
    pos = torch.tensor([[[0.1, 0.0]], [[0.5, 0.0]]], device=device, dtype=torch.float32)
    env.world.write_state(None, pos=pos, vel=torch.zeros_like(pos))
    scenario._install_obstacles()
    scenario._prev_dist = None
    scenario._prev_radius = None
    scenario._prev_penned = None
    scenario._refresh(integrate=False)
    rew = scenario.rewards()
    assert rew[0, 0].item() < rew[1, 0].item(), "a shepherd standing in the pen was not penalized"


@pytest.mark.parametrize("n_agents,n_sheep", [(1, 1), (3, 5), (2, 7)])
def test_obs_dim_formula(n_agents, n_sheep):
    scenario = ShepherdingScenario(n_agents=n_agents, n_sheep=n_sheep)
    assert scenario.obs_dim == 16 + 4 * n_agents + 2 * n_sheep


@pytest.mark.parametrize("device", DEVICES)
def test_n_sheep_one_is_the_degenerate_case_the_dir_floor_exists_for(device):
    """``n_sheep=1`` collapses ``mean_R``/``max_R`` to zero and ``stray`` to ``gcm`` (the
    lone sheep is its own centroid), so ``safe_unit(stray - gcm)`` must resolve the
    exact-zero vector through the hard floor rather than blowing up on a ``0/0``
    normalize -- exactly the case ``_safe_unit``'s floor exists for (see its docstring).
    """
    scenario = ShepherdingScenario(n_agents=1, n_sheep=1)
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset()
    scenario.pen_pos = torch.tensor([[0.5, 0.0]], device=device, dtype=torch.float32)
    scenario.sheep_pos = torch.tensor([[[0.2, 0.1]]], device=device, dtype=torch.float32)
    scenario.sheep_vel = torch.zeros_like(scenario.sheep_pos)
    zero_pos = torch.zeros(1, 1, 2, device=device, dtype=torch.float32)
    env.world.write_state(None, pos=zero_pos, vel=torch.zeros_like(zero_pos))
    scenario._install_obstacles()
    scenario._prev_dist = None
    scenario._prev_radius = None
    scenario._prev_penned = None
    scenario._refresh(integrate=False)
    obs = scenario.observations()

    assert obs[0, 0, 12].item() == 0.0  # mean_R
    assert obs[0, 0, 13].item() == 0.0  # max_R
    gcm_rel, stray_rel = obs[0, 0, 6:8], obs[0, 0, 14:16]
    torch.testing.assert_close(stray_rel, gcm_rel)  # stray == gcm
    # collect_pt = stray + collect_offset * safe_unit(stray - gcm) = stray, since
    # stray - gcm is exactly zero and the floor sends it to (0, 0) rather than NaN.
    torch.testing.assert_close(obs[0, 0, 18:20], stray_rel)


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
