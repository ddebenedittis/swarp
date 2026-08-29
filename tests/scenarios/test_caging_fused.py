"""Fused Warp obs/reward + escaping-disc kernels for CagingScenario.

Caging fuses the disc's spring-damper/Euler physics *and* the escape drift, so the two
paths' disc trajectories are only allclose (different agent-reduction order), kept
bounded by the linear damping. The largest circular gap is deliberately computed two
different ways — a sort plus circular diff in torch, a storage-free O(A^2)
"nearest bearing ahead" reduction in the kernel (Warp has no dynamically sized register
array to sort into) — so this suite is what pins that they are the same function.
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

from swarp import Environment
from swarp.scenarios.caging import CagingScenario

SPEC = FusedSpec(
    scenario=CagingScenario,
    fields=("obs", "rew", "done", "info:max_gap", "info:disc_pos"),
    grad_steps=4,
    grad_index=2,
    grad_backprop="rew",  # the reward is atan2/sort/norm all the way down: differentiable
    # Unique to this suite (see ``FusedSpec.action_seed``): the guards in
    # ``test_fused_matches_torch`` -- that the largest circular gap actually *moves*
    # over the rollout, and that the disc actually escapes somewhere -- pass because of
    # this seed. Share it with another suite and they can go vacuous unnoticed.
    action_seed=11,
)


def _env(device, fused, *, n_agents=3, world_size=1.0):
    scen = CagingScenario(n_agents=n_agents, world_size=world_size)
    return fused_env(scen, device, fused, spec=SPEC)


def _run(env, n_steps, device, n_agents):
    return fused_rollout(env, n_steps, device, n_agents, SPEC)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_agents", [3, 5])
def test_fused_matches_torch(device, n_agents):
    kw = dict(device=device, n_agents=n_agents)
    fused = _run(_env(fused=True, **kw), 20, device, n_agents)
    torchp = _run(_env(fused=False, **kw), 20, device, n_agents)
    # The caging geometry must actually be exercised: if the ring never changes shape,
    # every gap comparison below is comparing the same constant to itself.
    gaps = torch.stack([step[3] for step in fused])  # [T, n_envs]
    assert (gaps.amax(dim=0) - gaps.amin(dim=0)).max().item() > 0.1, "max gap never moved"
    # ...and the disc must actually drift/get pushed, or the body kernel is untested.
    disc = torch.stack([step[4] for step in fused])  # [T, n_envs, 1, 2]
    assert (disc - disc[0]).abs().max().item() > 1e-3, "disc never moved"
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, df, gf, pf = f
        ot, rt, dt_, gt, pt = r
        torch.testing.assert_close(of, ot, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"reward@{t}")
        assert torch.equal(df, dt_), f"done@{t}"
        torch.testing.assert_close(gf, gt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"max_gap@{t}")
        torch.testing.assert_close(pf, pt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"disc_pos@{t}")


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device, 3)
    b = _run(_env(device, fused=True), 10, device, 3)
    assert_fused_determinism(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    """The torch reference (differentiable disc + gap) path must run under enable_grad."""
    env = _env(device, fused=True, n_agents=3)
    assert_grad_falls_back_to_torch(env, device, 3, SPEC)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_parity(device):
    """A standalone ``reset()``/``reset_at()`` must leave the same outputs as torch."""
    assert_reset_parity(lambda fused: _env(device, fused), device, 3, SPEC)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    ("bearings", "expected_deg"),
    [
        ([0.0, 120.0, 240.0], 120.0),  # evenly spaced: the tightest a trio can cage
        ([0.0, 10.0, 20.0], 340.0),  # all bunched: the answer *is* the wrap-around gap
        ([350.0, 5.0, 170.0], 180.0),  # a pair straddling +/-pi, so the sort order wraps
        ([0.0, 180.0], 180.0),  # degenerate n_agents=2
        ([0.0], 360.0),  # degenerate n_agents=1: a lone agent cages nothing
    ],
)
def test_kernel_max_gap_matches_the_analytic_answer(device, bearings, expected_deg):
    """Plant known bearings and check the fused reduction against pen-and-paper.

    The wrap-around gap between the last and first bearing is the one an implementation
    gets wrong, so three of these five cases are about nothing else. The kernel never
    sorts — it takes the smallest counter-clockwise step out of each agent — and the
    modulo in that step is what has to carry a bearing past ``+pi`` round to ``-pi``.
    """
    scen = CagingScenario(n_agents=len(bearings), world_size=1.0)
    env = Environment(scen, n_envs=1, device=device, dt=0.05, seed=0, fused=True)
    env.reset(seed=0)
    scen.disc_pos.zero_()
    scen.disc_vel.zero_()
    radius = scen.cage_radius
    pos = torch.tensor(
        [[[radius * math.cos(math.radians(b)), radius * math.sin(math.radians(b))]
          for b in bearings]],
        device=device,
        dtype=torch.float32,
    )
    env.world.write_state(None, pos=pos)
    scen._install_obstacles()
    scen.finish_reset(None, obs_only=False)  # a full fused pass over the planted state
    got = math.degrees(scen.info()["max_gap"][0].item())
    assert got == pytest.approx(expected_deg, abs=1e-3)
