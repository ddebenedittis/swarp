"""Fused Warp obs/reward + movable-body kernels for TransportScenario.

Transport fuses the package spring-damper/Euler physics too, so the fused and
torch package trajectories are only allclose (different agent-reduction order),
kept bounded by the damping. Discrete flags (done/on_goal) still match.
"""

import pytest
import torch
from conftest import (
    DEVICES,
    FusedSpec,
    assert_fused_determinism,
    assert_grad_falls_back_to_torch,
    fused_env,
    fused_rollout,
)

from wmas import TransportScenario

SPEC = FusedSpec(
    scenario=TransportScenario,
    fields=("obs", "rew", "done", "info:package_dist_to_goal"),
    grad_steps=4,
    grad_index=2,
    grad_backprop="obs",  # the reward is built from discrete on_goal flags; obs is not
)


def _env(device, fused, *, n_agents=4, n_packages=1, world_size=0.5):
    scen = TransportScenario(
        n_agents=n_agents,
        n_packages=n_packages,
        world_size=world_size,
    )
    return fused_env(scen, device, fused, spec=SPEC)


def _run(env, n_steps, device, n_agents):
    return fused_rollout(env, n_steps, device, n_agents, SPEC)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_packages", [1, 2])
def test_fused_matches_torch(device, n_packages):
    kw = dict(device=device, n_packages=n_packages)
    fused = _run(_env(fused=True, **kw), 20, device, 4)
    torchp = _run(_env(fused=False, **kw), 20, device, 4)
    # The package must actually move (contact physics exercised).
    moved = any((fused[t][3] - fused[0][3]).abs().sum().item() > 0 for t in range(1, 20))
    assert moved, "package never moved; contact physics not exercised"
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, df, gf = f
        ot, rt, dt_, gt = r
        torch.testing.assert_close(of, ot, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"reward@{t}")
        assert torch.equal(df, dt_), f"done@{t}"
        torch.testing.assert_close(gf, gt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"pkg_dist@{t}")


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
