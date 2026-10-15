"""Fused Warp obs/reward kernels must match the torch reference path (bit-close).

Caging's own additions on top of the navigation-shaped parity suite:

* an **independent** ``torch.sort``-based check of the max-gap *value*, which is what
  proves the circular-min identity the kernels and the oracle both rely on (both of those
  use the min-of-wrapped-differences form deliberately, so a parity pass alone would not
  catch the two of them being wrong together);
* a guard that the disc genuinely moves, that ``gap_max`` genuinely varies, and that
  ``caged`` genuinely fires — without the third the ``cage_reward`` branch and the
  ``hold``/``done`` path are never executed and parity passes vacuously;
* the ``wall_free_radius`` conjunct: a corner-pinned disc must not count as caged, which
  is the degenerate optimum the predicate exists to design out;
* the arena clamp on the scenario-owned disc, which no part of the engine performs.
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

from swarp.scenarios.caging import CagingScenario

SPEC = FusedSpec(
    scenario=CagingScenario,
    fields=("obs", "rew", "done", "info:gap_max", "info:band_error", "info:caged"),
    grad_steps=6,
    grad_index=3,
    grad_backprop="rew",  # caging's reward is differentiable shaping
    # Load-bearing: this seed is what makes ``test_feature_guards_actually_fire`` see the
    # disc move and ``gap_max`` vary. Changing it can make that guard pass vacuously,
    # which no failure reports.
    action_seed=2,
    # A velocity-mode agent settles at a contact depth of ``max_speed / (k * sub_dt)``, so
    # the disc is only solid at a small ``sub_dt``. 8 is the documented floor at dt=0.05,
    # and the parity rollouts run at the floor deliberately: the stiff regime is where the
    # two paths' contact geometry is most likely to disagree.
    substeps=8,
)

#: Five agents, spawned on the ring (``difficulty=0``). Both halves are load-bearing.
#: ``caged`` has to fire inside the harness's 5-step episodes for the dwell bonus and the
#: hold counter to be exercised at all, and only a ring spawn does that; and 2*pi/5 =
#: 1.257 rad sits 0.44 rad clear of ``GAP_THRESHOLD`` even with both neighbours' angular
#: jitter pushing the same way, which keeps the ``caged`` conjunction — compared
#: *exactly*, via ``done`` — away from its thresholds.
N_AGENTS = 5
GAP_THRESHOLD = 2.0
#: Two, not the default ten: the harness's episodes are 5 steps long, so a ten-step dwell
#: could never reach ``done`` and the terminal path would go untested.
HOLD_STEPS = 2


def _scen(
    *, n_agents=N_AGENTS, difficulty=0.0, hold_steps=HOLD_STEPS, gap_threshold=GAP_THRESHOLD, **kw
):
    scen = CagingScenario(
        n_agents=n_agents,
        hold_steps=hold_steps,
        gap_threshold=gap_threshold,
        neighbor_method="brute",  # pin the backend so both paths see identical lists
        **kw,
    )
    scen.difficulty = difficulty
    return scen


def _env(device, fused, *, dtype=torch.float32, **kw):
    return fused_env(_scen(**kw), device, fused, spec=SPEC, dtype=dtype)


def _run(env, n_steps, device, n_agents=N_AGENTS, dtype=torch.float32):
    return fused_rollout(env, n_steps, device, n_agents, SPEC, dtype=dtype)


# ------------------------------------------------------------------- parity


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("difficulty", [0.0, 1.0])
def test_fused_matches_torch(device, difficulty):
    """Both curriculum ends: a ring spawn with an inert disc, and a uniform spawn with
    full drift and flee. The two exercise different force terms entirely."""
    kw = dict(difficulty=difficulty)
    fused = _run(_env(device, fused=True, **kw), 20, device)
    torchp = _run(_env(device, fused=False, **kw), 20, device)
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, df, gf, bf, cf = f
        ot, rt, dt_, gt, bt, ct = r
        torch.testing.assert_close(of, ot, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"reward@{t}")
        assert torch.equal(df, dt_), f"done@{t}"
        # ``caged`` is discrete and ``done`` is built on the hold counter it drives, so it
        # has to match exactly, not to within an ulp.
        assert torch.equal(cf, ct), f"caged@{t}"
        torch.testing.assert_close(gf, gt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"gap_max@{t}")
        torch.testing.assert_close(bf, bt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"band@{t}")


def test_fused_matches_torch_in_float64_on_cpu():
    """float64 on CPU: the other precision the kernels are instantiated for.

    Worth its own case rather than a ``dtype`` parametrization of the whole suite: it is
    where a scalar literal that was silently read as float32 inside a kernel shows up (the
    reason ``reset_kernels._as`` exists), and where the ``atan2``/``exp``/``log`` chain has
    no float32 slack left to hide a genuine formula difference behind.
    """
    fused = _run(
        _env("cpu", fused=True, dtype=torch.float64), 12, "cpu", dtype=torch.float64
    )
    torchp = _run(
        _env("cpu", fused=False, dtype=torch.float64), 12, "cpu", dtype=torch.float64
    )
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        torch.testing.assert_close(f[0], r[0], rtol=SPEC.rtol, atol=SPEC.atol, msg=f"obs@{t}")
        torch.testing.assert_close(f[1], r[1], rtol=SPEC.rtol, atol=SPEC.atol, msg=f"rew@{t}")
        assert torch.equal(f[2], r[2]), f"done@{t}"
        assert torch.equal(f[5], r[5]), f"caged@{t}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_agents", [3, 6])
def test_fused_matches_torch_over_team_size(device, n_agents):
    """``n_agents`` changes the obs width and the arc a regular ring achieves, so this also
    covers the team-size-relative ``gap_threshold`` default (left unset here)."""
    kw = dict(n_agents=n_agents, gap_threshold=None)
    fused = _run(_env(device, fused=True, **kw), 12, device, n_agents)
    torchp = _run(_env(device, fused=False, **kw), 12, device, n_agents)
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        torch.testing.assert_close(f[0], r[0], rtol=SPEC.rtol, atol=SPEC.atol, msg=f"obs@{t}")
        torch.testing.assert_close(f[1], r[1], rtol=SPEC.rtol, atol=SPEC.atol, msg=f"rew@{t}")
        assert torch.equal(f[2], r[2]), f"done@{t}"
        assert torch.equal(f[5], r[5]), f"caged@{t}"


@pytest.mark.parametrize("device", DEVICES)
def test_feature_guards_actually_fire(device):
    """The three features parity would otherwise compare vacuously.

    All checked on the **torch** rollout, i.e. on the oracle, so a fused bug cannot make
    this guard pass. ``caged`` is the important one: without it the ``cage_reward`` branch
    and the whole ``hold``/``done`` path are dead code in every other test here.
    """
    env = _env(device, fused=False)
    torchp = _run(env, 20, device)
    scen = env.scenario

    gap_values = torch.stack([step[3] for step in torchp])  # [T, E]
    assert gap_values.std().item() > 1e-3, "gap_max never varied across the rollout"

    saw_caged = any(step[5].any().item() for step in torchp)
    assert saw_caged, "test config never produced a caged step (cage_reward path untested)"

    saw_done = any(step[2].any().item() for step in torchp)
    assert saw_done, "the hold counter never reached hold_steps (done path untested)"

    # The disc has to be a moving body, not decoration: at difficulty 0 it is inert until
    # an agent touches it, so this is a statement about the contact reaction specifically.
    env2 = _env(device, fused=False, difficulty=1.0)
    env2.reset(seed=0)
    start = env2.scenario.disc_pos.clone()
    _run(env2, 20, device)
    moved = (env2.scenario.disc_pos - start).norm(dim=-1).max().item()
    assert moved > 1e-3, "the disc never moved"
    assert scen.obs_dim == 13 + 2 * (N_AGENTS - 1)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device)
    b = _run(_env(device, fused=True), 10, device)
    assert_fused_determinism(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    env = _env(device, fused=True)
    assert_grad_falls_back_to_torch(env, device, N_AGENTS, SPEC)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_parity(device):
    """A standalone ``reset()``/``reset_at()`` must leave the same outputs as torch."""
    assert_reset_parity(lambda fused: _env(device, fused), device, N_AGENTS, SPEC)


# ------------------------------------------------------------- the gap identity


def _sort_gap_max(th: torch.Tensor) -> torch.Tensor:
    """Max angular gap of bearings ``[..., N]``, by sorting — the independent oracle.

    Sort the bearings, difference consecutive ones, and close the circle with the wrap
    term ``first + 2*pi - last``. This is the formulation the *kernels* deliberately do
    not use (it disagrees with the min-of-wrapped-differences form exactly at bearing ties
    and at the 2*pi wrap, which would look like a kernel bug and is not one), which is
    precisely why it belongs here: it is an argument-free check of the **value**, so the
    two paths agreeing cannot mean they are both wrong in the same way.
    """
    s, _ = th.sort(dim=-1)
    if s.shape[-1] == 1:
        return torch.full(s.shape[:-1], 2.0 * math.pi, dtype=s.dtype, device=s.device)
    arcs = torch.cat([s.diff(dim=-1), (s[..., :1] + 2.0 * math.pi - s[..., -1:])], dim=-1)
    return arcs.max(dim=-1).values


def _min_wrapped_gap_max(th: torch.Tensor) -> torch.Tensor:
    """The form both the kernel and the torch oracle use, lifted out for the comparison."""
    dd = th.unsqueeze(-2) - th.unsqueeze(-1)
    wrapped = torch.where(dd < 0.0, dd + 2.0 * math.pi, dd)
    eye = torch.eye(th.shape[-1], device=th.device, dtype=torch.bool)
    return wrapped.masked_fill(eye, 2.0 * math.pi).min(-1).values.max(-1).values


def test_gap_matches_sort_oracle():
    # Hand-built cases, each one a thing the identity has to get right.
    cases = {
        # Regular quadrilateral (bearings in (-pi, pi]): every arc is exactly pi/2.
        "regular": ([-0.5 * math.pi, 0.0, 0.5 * math.pi, math.pi], 0.5 * math.pi),
        # All four bearings in one narrow cluster: the complementary arc is the gap.
        "cluster": ([0.0, 0.05, 0.1, 0.15], 2.0 * math.pi - 0.15),
        # Straddling the +-pi branch cut of atan2, which is where a naive wrap breaks. The
        # two bearings either side of the cut are 0.2 apart, so the largest arc is the
        # -3.04 -> 0 one, i.e. pi - 0.1 -- not the 0.2 sliver a broken wrap would report.
        "wrap": ([math.pi - 0.1, -math.pi + 0.1, 0.0, 1.0], math.pi - 0.1),
    }
    for name, (angles, want) in cases.items():
        th = torch.tensor([angles], dtype=torch.float64)
        got = _min_wrapped_gap_max(th)
        torch.testing.assert_close(
            got, torch.tensor([want], dtype=torch.float64), rtol=0, atol=1e-12, msg=name
        )
        torch.testing.assert_close(got, _sort_gap_max(th), rtol=0, atol=1e-12, msg=name)

    # ...and the one place the two forms genuinely part company, pinned rather than hidden.
    # At an EXACT bearing tie the min-of-wrapped-differences form gives each duplicate a
    # zero arc, so the arcs no longer sum to 2*pi and it under-reports; the sort form
    # collapses the duplicates and reports pi. Exact ties have measure zero in floating
    # point, and what parity needs is that the kernel and the torch oracle agree with each
    # OTHER — which they do, because both use the min form. This assertion exists so that a
    # future reader who "fixes" one side to the sort form finds out here rather than in a
    # parity failure that looks like a kernel bug.
    tied = torch.tensor([[0.0, 0.0, math.pi, math.pi]], dtype=torch.float64)
    assert _min_wrapped_gap_max(tied).item() == 0.0
    assert _sort_gap_max(tied).item() == pytest.approx(math.pi)

    # Random batch, several team sizes, against the sort oracle.
    gen = torch.Generator().manual_seed(0)
    for n in (1, 2, 3, 5, 8):
        th = torch.empty(256, n, dtype=torch.float64).uniform_(
            -math.pi, math.pi, generator=gen
        )
        torch.testing.assert_close(
            _min_wrapped_gap_max(th), _sort_gap_max(th), rtol=0, atol=1e-9, msg=f"n={n}"
        )
        # The N arcs sum to exactly 2*pi, which is the property that makes the minimum of
        # wrapped differences *be* the circular order rather than merely resemble it.
        if n > 1:
            dd = th.unsqueeze(-2) - th.unsqueeze(-1)
            wrapped = torch.where(dd < 0.0, dd + 2.0 * math.pi, dd)
            eye = torch.eye(n, dtype=torch.bool)
            arcs = wrapped.masked_fill(eye, 2.0 * math.pi).min(-1).values
            torch.testing.assert_close(
                arcs.sum(-1),
                torch.full((256,), 2.0 * math.pi, dtype=torch.float64),
                rtol=0,
                atol=1e-9,
            )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [False, True])
def test_reported_gap_max_matches_the_sort_oracle(device, fused):
    """And the *scenario's* ``gap_max`` agrees with the sort oracle on real states."""
    env = _env(device, fused=fused, difficulty=1.0)
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(SPEC.action_seed)
    with torch.no_grad():
        for _ in range(8):
            act = torch.empty(env.n_envs, N_AGENTS, 2, device=device).uniform_(
                -1, 1, generator=gen
            )
            _, _, term, trunc, info = env.step(act)
            # An env that ended on this step has already been auto-reset, so the state read
            # below is the *next* episode's spawn while ``info`` still describes the
            # transition that was returned. Compare only the envs that ran on.
            live = ~(term | trunc)
            if not bool(live.any()):
                continue
            rel = env.world.state.pos - env.scenario.disc_pos[:, 0].unsqueeze(1)
            th = torch.atan2(rel[..., 1], rel[..., 0]).double()
            torch.testing.assert_close(
                info["gap_max"].double()[live],
                _sort_gap_max(th)[live],
                rtol=1e-5,
                atol=1e-5,
            )


# --------------------------------------------------------------- the predicate


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [False, True])
def test_corner_pinned_disc_is_not_caged(device, fused):
    """The ``wall_free_radius`` conjunct, which nothing else in the suite can see.

    Once the disc is clamped to the arena the trivial optimum is to shove it into a
    corner: two walls close most escape directions for free and ``gap_max`` never has to
    shrink. A ring of agents placed perfectly around a corner-pinned disc satisfies the
    *other* two conjuncts exactly, so this is the whole content of the third one.
    """
    env = _env(device, fused=fused, n_agents=N_AGENTS)
    env.reset(seed=0)
    scen = env.scenario
    ang = torch.arange(N_AGENTS, device=device, dtype=env.world.dtype) * (
        2.0 * math.pi / N_AGENTS
    )
    ring = scen.cage_radius * torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)

    def place(qx, qy):
        q = torch.tensor([qx, qy], device=device, dtype=env.world.dtype)
        scen.disc_pos.copy_(q.view(1, 1, 2).expand(env.n_envs, 1, 2))
        scen.disc_vel.zero_()
        env.world.write_state(None, pos=(q + ring).expand(env.n_envs, N_AGENTS, 2), vel=0.0)
        scen._install_obstacles()
        # ``finish_reset`` is the one entry point that recomputes the predicate for the
        # state just written *without* integrating the disc or advancing the hold counter,
        # and it branches on ``fused_active`` internally — so this reads the same on both
        # paths rather than reaching into either one's launches.
        scen.finish_reset(None, obs_only=False)
        return env.scenario.info()["caged"].clone()

    # Centred: a perfect ring around a disc well away from the walls is caged.
    assert place(0.0, 0.0).all(), "a perfect central ring should be caged"
    # Corner: identical ring geometry, identical gaps, identical radii — only |q| moved.
    far = 0.95 * scen.wall_free_radius
    assert not place(far, far).any(), "a corner-pinned disc must not count as caged"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [False, True])
def test_disc_stays_inside_the_arena(device, fused):
    """The scenario clamps its own disc: nothing in the engine bounds an obstacle it does
    not integrate (``Stepper`` feeds the clamp to a kernel that early-returns for a
    non-``MOVABLE`` obstacle). Full drift and full flee, long enough for the drift alone
    to carry the disc past the wall if it were unclamped."""
    env = _env(device, fused=fused, difficulty=1.0)
    _run(env, 60, device)
    bound = env.scenario.world_size - env.scenario.disc_radius
    assert env.scenario.disc_pos.abs().max().item() <= bound + 1e-6


# ------------------------------------------------------------- multiobj reward


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [False, True])
def test_multiobj_reward_sums_to_the_scalar_reward(device, fused):
    env = _env(device, fused=fused)
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(SPEC.action_seed)
    with torch.no_grad():
        for _ in range(12):
            act = torch.empty(env.n_envs, N_AGENTS, 2, device=device).uniform_(
                -1, 1, generator=gen
            )
            _, rew, _, _, info = env.step(act)
            mo = info["multiobj_reward"]
            assert mo.shape == (env.n_envs, N_AGENTS, 5)
            torch.testing.assert_close(mo.sum(-1), rew, rtol=SPEC.rtol, atol=SPEC.atol)


# ------------------------------------------------------------------- geometry


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("difficulty", [0.0, 1.0])
def test_spawns_lie_inside_the_arena(device, difficulty):
    env = _env(device, fused=True, difficulty=difficulty)
    env.reset(seed=0)
    scen = env.scenario
    lim = scen.world_size - scen.agent_radius
    assert env.world.state.pos.abs().max().item() <= lim + 1e-6
    assert scen.disc_pos.abs().max().item() <= scen.world_size - scen.disc_radius + 1e-6
    # At difficulty 0 the disc is inert and at the origin, and the agents sit on the ring:
    # the single reason the curriculum works is that the dwell bonus is reachable on step 1.
    if difficulty == 0.0:
        assert scen.disc_pos.abs().max().item() == 0.0
        assert scen.drift.abs().max().item() == 0.0
        assert scen.evade.abs().max().item() == 0.0
        r = (env.world.state.pos - scen.disc_pos[:, 0].unsqueeze(1)).norm(dim=-1)
        assert ((r - scen.cage_radius).abs() <= scen.band_half).all()


@pytest.mark.parametrize("device", DEVICES)
def test_difficulty_scales_the_disc_forces(device):
    """``difficulty`` must still bite after the reset kernel has been launched once.

    It reaches the kernel through a one-element device tensor precisely so a trainer can
    move it mid-training; this pins the observable consequence (the per-episode drift and
    evade draws) rather than the mechanism.
    """
    env = _env(device, fused=True, difficulty=1.0)
    env.reset(seed=0)
    scen = env.scenario
    assert scen.drift.norm(dim=-1).max().item() > 0.5 * scen.drift_mag
    torch.testing.assert_close(scen.evade, torch.ones_like(scen.evade))
    scen.difficulty = 0.0
    env.reset(seed=1)
    assert scen.drift.abs().max().item() == 0.0
    assert scen.evade.abs().max().item() == 0.0
    scen.difficulty = 5.0
    assert scen.difficulty == 1.0


def test_unreachable_gap_threshold_is_rejected():
    # 2*pi/5 = 1.257 is the arc of a perfectly regular ring of five, so nothing at or
    # below it is ever satisfiable.
    with pytest.raises(ValueError, match="2\\*pi/n_agents"):
        CagingScenario(n_agents=5, gap_threshold=1.2)
    with pytest.raises(ValueError, match="gap_threshold"):
        CagingScenario(gap_threshold=7.0)


def test_hold_steps_must_fit_in_a_byte():
    with pytest.raises(ValueError, match="hold_steps"):
        CagingScenario(hold_steps=0)
    with pytest.raises(ValueError, match="hold_steps"):
        CagingScenario(hold_steps=256)


def test_constructs_on_cpu_with_three_agents():
    # Exactly what the registry-wide scenario tests do once this is registered — bare
    # ``cls(n_agents=3)``, no other argument. It is the reason ``gap_threshold`` defaults
    # to a value derived from ``n_agents``: at three agents a regular ring's arc is already
    # 2.09 rad, so the 2.0 the task was specified with would be unsatisfiable and the
    # constructor would reject its own default.
    scen = CagingScenario(n_agents=3)
    assert scen.obs_dim == 13 + 2 * 2
    assert scen.gap_threshold > 2.0 * math.pi / 3


def test_default_gap_threshold_tracks_the_team_size():
    # Constant absolute slack over the regular-ring arc, so the task means the same thing
    # at every team size; 2.0 rad at the default five agents.
    for n in (3, 5, 8):
        scen = CagingScenario(n_agents=n)
        assert scen.gap_threshold == pytest.approx(2.0 * math.pi / n + 0.75)
    assert CagingScenario().gap_threshold == pytest.approx(2.0, abs=0.01)
