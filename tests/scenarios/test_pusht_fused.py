"""Fused Warp obs/reward + movable-T-body kernels for PushTScenario.

Push-T fuses the T's box-SDF contact physics and substepped Euler integration too, so
the fused and torch body trajectories are only allclose (different agent/box reduction
order), kept bounded by the damping. Discrete flags (done/on_goal) still match.

The reward is checked loosely. It is built from *differences* of successive pose errors,
which amplifies the ulp-level trajectory gap, and the stiff contact (contact_k 8000)
means an agent sitting near the contact threshold can be judged in contact by one path
and out by the other — a discrete flip worth ~1e-3 of shaping. The pose itself stays
within ~1e-4 over 20 steps, which is what the tighter checks below assert.
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

from swarp import PushTScenario

SPEC = FusedSpec(
    scenario=PushTScenario,
    fields=("obs", "rew", "done", "info:tee_dist_to_goal", "info:tee_angle_error"),
    grad_steps=4,
    grad_index=2,
    grad_backprop="obs",  # the reward is a difference of pose errors; obs is smooth
    substeps=8,  # the scenario's operating point; contact_k 8000 needs it
)


def _env(device, fused, *, n_agents=4, world_size=0.5, **fleet):
    # rot_away_penalty on, so parity covers the turn-away cost (it defaults to 0).
    scen = PushTScenario(
        n_agents=n_agents, world_size=world_size, rot_away_penalty=1.0, **fleet
    )
    return fused_env(scen, device, fused, spec=SPEC)


def _run(env, n_steps, device, n_agents):
    return fused_rollout(env, n_steps, device, n_agents, SPEC)


@pytest.mark.parametrize("device", DEVICES)
# n_agents=1 is the single-robot Push-T: obs_dim collapses to 12 (the obs kernel's
# teammate loop writes nothing) and the body kernel reduces over one contact.
@pytest.mark.parametrize("n_agents", [1, 4])
def test_fused_matches_torch(device, n_agents):
    fused = _run(_env(device, fused=True, n_agents=n_agents), 20, device, n_agents)
    torchp = _run(_env(device, fused=False, n_agents=n_agents), 20, device, n_agents)
    assert fused[0][0].shape[-1] == 12 + 2 * (n_agents - 1)
    # The T must actually translate *and* rotate (contact physics exercised).
    moved = any((fused[t][3] - fused[0][3]).abs().sum().item() > 0 for t in range(1, 20))
    spun = any((fused[t][4] - fused[0][4]).abs().sum().item() > 0 for t in range(1, 20))
    assert moved, "T never moved; contact physics not exercised"
    assert spun, "T never rotated; contact torque not exercised"
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, df, gf, af = f
        ot, rt, dt_, gt, at = r
        # obs and the pose diagnostics are held *tighter* than PushTScenario's declared
        # parity bound (SPEC.rtol/atol, which the reward below uses and which the benchmark
        # parity gate reads): the pose itself stays within ~1e-4 over 20 steps, and only the
        # reward — a difference of successive pose errors — needs the full bound.
        torch.testing.assert_close(of, ot, rtol=1e-3, atol=1e-3, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"reward@{t}")
        assert torch.equal(df, dt_), f"done@{t}"
        torch.testing.assert_close(gf, gt, rtol=1e-3, atol=1e-3, msg=f"tee_dist@{t}")
        torch.testing.assert_close(af, at, rtol=1e-3, atol=1e-3, msg=f"tee_angle@{t}")


FLEETS = pytest.mark.parametrize(
    "fleet",
    [
        {"model": "diff_drive"},
        {"agent_shape": "box"},
        {"model": "diff_drive", "agent_shape": "box"},
    ],
    ids=["unicycle", "box", "unicycle-box"],
)


@pytest.mark.parametrize("device", DEVICES)
@FLEETS
def test_fleet_fused_matches_torch(device, fleet):
    """Unicycle and box robots: random spawn headings, the heading appended to the obs, and
    box contacts pushing the T, the same on both paths."""
    fused = _run(_env(device, fused=True, **fleet), 20, device, 4)
    torchp = _run(_env(device, fused=False, **fleet), 20, device, 4)
    assert fused[0][0].shape[-1] == 12 + 2 * 3 + 2
    heading = fused[0][0][..., -2:]
    assert heading[..., 0].std() > 0.3, "spawn headings are not randomised"
    moved = any((fused[t][3] - fused[0][3]).abs().sum().item() > 0 for t in range(1, 20))
    assert moved, "T never moved; contact physics not exercised"
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        torch.testing.assert_close(f[0], r[0], rtol=1e-3, atol=1e-3, msg=f"obs@{t}")
        torch.testing.assert_close(f[1], r[1], rtol=SPEC.rtol, atol=SPEC.atol, msg=f"rew@{t}")
        assert torch.equal(f[2], r[2]), f"done@{t}"


@pytest.mark.parametrize("device", DEVICES)
def test_taped_step_refuses_box_robots(device):
    """The grad path integrates the T in torch against disc robots, so it refuses boxes
    rather than push them as discs."""
    env = _env(device, fused=True, agent_shape="box")
    env.reset(seed=0)
    act = torch.zeros(env.n_envs, 4, 2, device=device, requires_grad=True)
    with torch.enable_grad(), pytest.raises(NotImplementedError, match="box robots"):
        env.step(act)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device, 4)
    b = _run(_env(device, fused=True), 10, device, 4)
    assert_fused_determinism(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    """The torch reference (differentiable body) path must be taken under enable_grad."""
    env = _env(device, fused=True, n_agents=4, world_size=1.0)
    assert_grad_falls_back_to_torch(env, device, 4, SPEC)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_parity(device):
    """A standalone ``reset()``/``reset_at()`` must leave the same outputs as torch."""
    assert_reset_parity(lambda fused: _env(device, fused), device, 4, SPEC)
