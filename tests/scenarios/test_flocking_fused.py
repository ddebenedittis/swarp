"""Fused Warp obs/reward kernel for FlockingScenario must match the torch path."""

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

from swarp import FlockingScenario

SPEC = FusedSpec(
    scenario=FlockingScenario,
    fields=("obs", "rew", "info:crowding"),
    grad_steps=4,
    grad_index=2,
    grad_backprop="rew",  # flocking's reward is differentiable cohesion/alignment
)


def _env(device, fused, *, n_agents=6, neighbor_obs=4, world_size=0.6):
    scen = FlockingScenario(
        n_agents=n_agents,
        neighbor_obs=neighbor_obs,
        world_size=world_size,
        neighbor_radius=0.4,
    )
    return fused_env(scen, device, fused, spec=SPEC)


def _run(env, n_steps, device, n_agents):
    return fused_rollout(env, n_steps, device, n_agents, SPEC)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("neighbor_obs", [1, 4])
def test_fused_matches_torch(device, neighbor_obs):
    kw = dict(device=device, neighbor_obs=neighbor_obs)
    fused = _run(_env(fused=True, **kw), 20, device, 6)
    torchp = _run(_env(fused=False, **kw), 20, device, 6)
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, cf = f
        ot, rt, ct = r
        torch.testing.assert_close(of, ot, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"reward@{t}")
        torch.testing.assert_close(cf, ct, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"crowding@{t}")


@pytest.mark.parametrize("device", DEVICES)
def test_fused_dense_neighbors_parity(device):
    # Many agents in a small world -> dense neighbor lists exercising the reward
    # aggregation and the crowding (separation) term.
    kw = dict(device=device, n_agents=10, world_size=0.25, neighbor_obs=3)
    f = _run(_env(fused=True, **kw), 12, device, 10)
    t = _run(_env(fused=False, **kw), 12, device, 10)
    saw_crowd = any(step[2].abs().sum().item() > 0 for step in t)
    assert saw_crowd, "test config did not induce any crowding"
    for f_i, t_i in zip(f, t, strict=True):
        torch.testing.assert_close(f_i[0], t_i[0], rtol=SPEC.rtol, atol=SPEC.atol)
        torch.testing.assert_close(f_i[1], t_i[1], rtol=SPEC.rtol, atol=SPEC.atol)
        torch.testing.assert_close(f_i[2], t_i[2], rtol=SPEC.rtol, atol=SPEC.atol)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device, 6)
    b = _run(_env(device, fused=True), 10, device, 6)
    assert_fused_determinism(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    """A grad-enabled step must take the differentiable torch path (fused off)."""
    env = _env(device, fused=True, n_agents=4, world_size=1.0)
    assert_grad_falls_back_to_torch(env, device, 4, SPEC)
