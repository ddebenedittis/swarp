"""Fused Warp obs/reward kernels for SamplingScenario must match the torch path.

Sampling carries a persistent ``consumed`` grid whose evolution must stay
bit-identical between the fused (2-launch read-then-scatter) and torch
(gather-then-scatter) paths — the discrete part of the reward and the
``consumed_frac`` diagnostic both depend on it.
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

from wmas import SamplingScenario

SPEC = FusedSpec(
    fields=("obs", "rew", "info:field", "info:consumed_frac"),
    grad_steps=4,
    grad_index=2,
    grad_backprop="rew",  # the field reward is a differentiable gaussian mixture read
)


def _env(device, fused, *, n_agents=4, grid_res=12, n_gaussians=3, world_size=1.0):
    scen = SamplingScenario(
        n_agents=n_agents,
        grid_res=grid_res,
        n_gaussians=n_gaussians,
        world_size=world_size,
    )
    return fused_env(scen, device, fused, spec=SPEC)


def _run(env, n_steps, device, n_agents):
    return fused_rollout(env, n_steps, device, n_agents, SPEC)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("grid_res", [8, 12])
def test_fused_matches_torch(device, grid_res):
    kw = dict(device=device, grid_res=grid_res)
    fused = _run(_env(fused=True, **kw), 20, device, 4)
    torchp = _run(_env(fused=False, **kw), 20, device, 4)
    saw_reward = any(step[1].abs().sum().item() > 0 for step in torchp)
    assert saw_reward, "test config earned no field reward"
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, ff, cff = f
        ot, rt, ft, cft = r
        torch.testing.assert_close(of, ot, rtol=1e-5, atol=1e-6, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=1e-5, atol=1e-6, msg=f"reward@{t}")
        torch.testing.assert_close(ff, ft, rtol=1e-5, atol=1e-6, msg=f"field@{t}")
        # consumed grid is discrete -> its coverage fraction must match exactly
        assert torch.equal(cff, cft), f"consumed_frac@{t}"


@pytest.mark.parametrize("device", DEVICES)
def test_fused_shared_cell_double_reward(device):
    # Two agents starting in the same not-yet-consumed cell must both earn the
    # field value (read-before-write); the fused 2-launch path must reproduce it.
    kw = dict(device=device, n_agents=6, world_size=0.4, grid_res=6)
    f = _run(_env(fused=True, **kw), 12, device, 6)
    t = _run(_env(fused=False, **kw), 12, device, 6)
    for f_i, t_i in zip(f, t, strict=True):
        torch.testing.assert_close(f_i[1], t_i[1], rtol=1e-5, atol=1e-6)  # reward
        assert torch.equal(f_i[3], t_i[3])  # consumed_frac


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device, 4)
    b = _run(_env(device, fused=True), 10, device, 4)
    assert_fused_determinism(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    env = _env(device, fused=True, n_agents=4, world_size=1.0)
    assert_grad_falls_back_to_torch(env, device, 4, SPEC)
