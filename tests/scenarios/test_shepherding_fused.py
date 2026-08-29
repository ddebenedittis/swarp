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
    fields=("obs", "rew", "done", "info:sheep_penned", "info:sheep_dist_to_pen"),
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

    Returns ``(max sheep displacement, a sheep moved while out of contact)``. The second
    is the load-bearing one: it is the *flee* force — the whole point of this scenario —
    rather than the contact reaction transport already covers. ``n_steps`` stays under
    the harness's ``max_steps=5`` so no auto-reset teleports a sheep and fakes a move.
    """
    env = _env(device, fused=True)
    scen = env.scenario
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(SPEC.action_seed)
    reach = scen.agent_radius + scen.sheep_radius + scen.contact_margin
    start = scen.sheep_pos.clone()
    free_move = False
    with torch.no_grad():
        for _ in range(n_steps):
            a = torch.empty(env.n_envs, N_AGENTS, 2, device=device).uniform_(
                -1, 1, generator=gen
            )
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
    return (scen.sheep_pos - start).norm(dim=-1).max().item(), free_move


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_sheep", [1, 2])
def test_fused_matches_torch(device, n_sheep):
    kw = dict(device=device, n_sheep=n_sheep)
    fused = _run(_env(fused=True, **kw), 20, device)
    torchp = _run(_env(fused=False, **kw), 20, device)
    # The sheep must actually move, and at least one of them while nobody is touching
    # it — otherwise the flee force is untested and this is transport with new names.
    disp, free_move = _probe(device)
    assert disp > 1e-3, "sheep never moved; flock physics not exercised"
    assert free_move, "no sheep moved out of contact; the flee force never fired"
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, df, pf, gf = f
        ot, rt, dt_, pt, gt = r
        torch.testing.assert_close(of, ot, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"reward@{t}")
        assert torch.equal(df, dt_), f"done@{t}"
        torch.testing.assert_close(pf, pt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"penned@{t}")
        torch.testing.assert_close(gf, gt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"pen_dist@{t}")


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
