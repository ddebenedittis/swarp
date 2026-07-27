"""Fused Warp obs/reward kernels for FormationScenario must match the torch path."""

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

from wmas import FormationScenario

SPEC = FusedSpec(
    fields=("obs", "rew", "done", "info:multiobj_reward", "info:formation_error"),
    grad_steps=6,
    grad_index=3,
    grad_backprop="rew",  # formation's reward is differentiable shaping
)


def _env(device, fused, *, n_agents=5, world_size=0.6, formation_radius=0.3):
    scen = FormationScenario(
        n_agents=n_agents,
        world_size=world_size,
        formation_radius=formation_radius,
    )
    return fused_env(scen, device, fused, spec=SPEC)


def _run(env, n_steps, device, n_agents):
    return fused_rollout(env, n_steps, device, n_agents, SPEC)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_matches_torch(device):
    fused = _run(_env(device, fused=True), 20, device, 5)
    torchp = _run(_env(device, fused=False), 20, device, 5)
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, df, mf, ef = f
        ot, rt, dt_, mt, et = r
        torch.testing.assert_close(of, ot, rtol=1e-5, atol=1e-6, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=1e-5, atol=1e-6, msg=f"reward@{t}")
        assert torch.equal(df, dt_), f"done@{t}"
        # multiobj_reward[..., 1] = collision_penalty * touching -> discrete, exact
        assert torch.equal(mf[..., 1], mt[..., 1]), f"collision term@{t}"
        torch.testing.assert_close(mf[..., 0], mt[..., 0], rtol=1e-5, atol=1e-6, msg=f"shaping@{t}")
        torch.testing.assert_close(ef, et, rtol=1e-5, atol=1e-6, msg=f"formation_error@{t}")


@pytest.mark.parametrize("device", DEVICES)
def test_fused_dense_touching_parity(device):
    # Many agents in a small world -> lots of touching pairs; the collision count
    # (derived from multiobj_reward) must match the torch reference exactly.
    kw = dict(device=device, n_agents=7, world_size=0.25, formation_radius=0.12)
    f = _run(_env(fused=True, **kw), 12, device, 7)
    t = _run(_env(fused=False, **kw), 12, device, 7)
    saw_touch = any(step[3][..., 1].abs().sum().item() > 0 for step in t)
    assert saw_touch, "test config did not induce any touching pairs"
    for f_i, t_i in zip(f, t, strict=True):
        assert torch.equal(f_i[3][..., 1], t_i[3][..., 1])  # collision term (touching)
        assert torch.equal(f_i[2], t_i[2])  # done
        torch.testing.assert_close(f_i[1], t_i[1], rtol=1e-5, atol=1e-6)  # reward


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device, 5)
    b = _run(_env(device, fused=True), 10, device, 5)
    assert_fused_determinism(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_interleave_shaping_continuous(device):
    """A grad step (torch path) interleaved with fused no-grad steps must keep the
    shaping baseline continuous (shared _prev_dist) and not crash."""
    env = _env(device, fused=True, n_agents=4, world_size=1.0)
    assert_grad_falls_back_to_torch(env, device, 4, SPEC)
