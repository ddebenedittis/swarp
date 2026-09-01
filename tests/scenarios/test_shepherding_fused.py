"""Fused Warp obs/reward + flock kernels for ShepherdingScenario.

Shepherding fuses the sheep force/Euler physics too, so the fused and torch flock
trajectories are only allclose (different shepherd- and sheep-reduction order). Unlike
transport that difference is amplified rather than damped — the flee force acts at a
distance, so a ulp of positional drift changes a force the sheep feels every step — which
is why the scenario declares a looser ``parity_rtol``/``parity_atol``. Discrete flags
(done) still match.
"""

import pytest
import torch
from conftest import (
    DEVICES,
    FusedSpec,
    _pick,
    assert_fused_determinism,
    assert_grad_falls_back_to_torch,
    assert_reset_parity,
    fused_env,
    fused_rollout,
)

from swarp.scenarios.shepherding import ShepherdingScenario

N_AGENTS = 3

SPEC = FusedSpec(
    scenario=ShepherdingScenario,
    fields=(
        "obs",
        "rew",
        "done",
        "info:sheep_penned",
        "info:sheep_dist_to_pen",
        "info:flock_radius",
    ),
    grad_steps=4,
    grad_index=2,
    grad_backprop="rew",  # the flee force makes the shaping term differentiable in actions
    action_seed=13,
)


def _env(device, fused, *, n_agents=N_AGENTS, n_sheep=2, world_size=0.6, pen_radius=0.15):
    scen = ShepherdingScenario(
        n_agents=n_agents,
        n_sheep=n_sheep,
        world_size=world_size,
        pen_radius=pen_radius,
    )
    return fused_env(scen, device, fused, spec=SPEC)


def _run(env, n_steps, device, n_agents=N_AGENTS):
    return fused_rollout(env, n_steps, device, n_agents, SPEC)


def _probe(device, n_steps=4):
    """Did the flock physics actually fire under ``action_seed``?

    Returns ``(max sheep displacement, a sheep moved while out of contact, max flock
    radius change)``. The second is load-bearing for the *flee* force — the whole point
    of this scenario — rather than the contact reaction transport already covers; the
    third is load-bearing for the *gather* reward term added in the redesign — without
    it, a probe that only checks the sheep moved says nothing about whether the flock's
    own radius (what ``gather_factor`` shapes) ever actually changed. ``n_steps`` stays
    under the harness's ``max_steps=5`` so no auto-reset teleports a sheep and fakes a
    move.
    """
    env = _env(device, fused=True)
    scen = env.scenario
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(SPEC.action_seed)
    reach = scen.agent_radius + scen.sheep_radius + scen.contact_margin
    start = scen.sheep_pos.clone()
    start_radius = scen.info()["flock_radius"].clone()
    max_radius_change = 0.0
    free_move = False
    with torch.no_grad():
        for _ in range(n_steps):
            a = torch.empty(env.n_envs, N_AGENTS, 2, device=device).uniform_(-1, 1, generator=gen)
            before = scen.sheep_pos.clone()
            gap = (
                (before.unsqueeze(1) - env.world.state.pos.unsqueeze(2))
                .norm(dim=-1)
                .min(dim=1)
                .values
            )
            env.step(a)
            moved = (scen.sheep_pos - before).norm(dim=-1)
            free_move |= bool(((moved > 1e-5) & (gap > reach)).any())
            radius_change = (scen.info()["flock_radius"] - start_radius).abs().max().item()
            max_radius_change = max(max_radius_change, radius_change)
    return (scen.sheep_pos - start).norm(dim=-1).max().item(), free_move, max_radius_change


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_sheep", [1, 2])
def test_fused_matches_torch(device, n_sheep):
    kw = dict(device=device, n_sheep=n_sheep)
    fused = _run(_env(fused=True, **kw), 20, device)
    torchp = _run(_env(fused=False, **kw), 20, device)
    # The sheep must actually move, and at least one of them while nobody is touching
    # it — otherwise the flee force is untested and this is transport with new names.
    # The flock radius must move too, or the gather reward term is unexercised.
    disp, free_move, radius_change = _probe(device)
    assert disp > 1e-3, "sheep never moved; flock physics not exercised"
    assert free_move, "no sheep moved out of contact; the flee force never fired"
    assert radius_change > 1e-3, "flock radius never changed; the gather term is unexercised"
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, df, pf, gf, radf = f
        ot, rt, dt_, pt, gt, radt = r
        torch.testing.assert_close(of, ot, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"reward@{t}")
        assert torch.equal(df, dt_), f"done@{t}"
        torch.testing.assert_close(pf, pt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"penned@{t}")
        torch.testing.assert_close(gf, gt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"pen_dist@{t}")
        torch.testing.assert_close(
            radf, radt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"flock_radius@{t}"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device)
    b = _run(_env(device, fused=True), 10, device)
    assert_fused_determinism(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    """The torch reference (differentiable flock) path must be taken under enable_grad."""
    env = _env(device, fused=True, world_size=1.0, pen_radius=0.2)
    assert_grad_falls_back_to_torch(env, device, N_AGENTS, SPEC)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_parity(device):
    """A standalone ``reset()``/``reset_at()`` must leave the same outputs as torch."""
    assert_reset_parity(lambda fused: _env(device, fused), device, N_AGENTS, SPEC)


@pytest.mark.parametrize("device", DEVICES)
def test_an_already_penned_flock_keeps_the_paths_in_parity(device):
    """The three new carries (``prev_radius``, ``prev_penned``, ``hold``), the ``done``
    latch and the 10.0 terminal bonus are not exercised at all by the 20-step random
    rollout above -- no random policy pens five sheep in 20 steps, so none of that new
    reward machinery has ever been compared between the two paths.

    Both envs are reset (same seed, so their initial draws are bit-identical), then the
    flock is seeded already inside the pen and held there (shepherds parked past
    ``flee_radius``) for a further 20 zero-action steps, long enough for the hold
    counter to reach ``pen_hold`` and the ``done_reward`` spike to actually fire on both
    paths.
    """
    fe, te = _env(device, fused=True), _env(device, fused=False)
    for env in (fe, te):
        env.reset(seed=0)

    n_envs, n_sheep = fe.n_envs, fe.scenario.n_sheep
    pen = torch.zeros(n_envs, 2, device=device, dtype=torch.float32)
    sheep = torch.zeros(n_envs, n_sheep, 2, device=device, dtype=torch.float32)
    far = torch.full((n_envs, N_AGENTS, 2), 5.0, device=device, dtype=torch.float32)
    for env in (fe, te):
        env.scenario.pen_pos = pen.clone()
        env.scenario.sheep_pos = sheep.clone()
        env.scenario.sheep_vel = torch.zeros_like(sheep)
        env.world.write_state(None, pos=far.clone(), vel=torch.zeros_like(far))
        env.scenario._install_obstacles()

    zero = torch.zeros(n_envs, N_AGENTS, 2, device=device)
    for t in range(20):
        fo, fr, ft, ftr, fi = fe.step(zero)
        to, tr, tt, ttr, ti = te.step(zero)
        fdone, tdone = ft | ftr, tt | ttr
        for name in SPEC.fields:
            f = _pick(name, fo, fr, fdone, fi)
            r = _pick(name, to, tr, tdone, ti)
            if name == "done":
                assert torch.equal(f, r), f"done@{t}"
            else:
                torch.testing.assert_close(f, r, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"{name}@{t}")
