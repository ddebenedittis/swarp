"""Fused Warp obs/reward kernels for DiscoveryScenario must match the torch path.

Discovery carries a persistent monotonic ``covered`` latch and a per-target
coverage reduction; the discrete coverage/touching flags (and the one-off
``newly`` covering reward) must stay bit-identical between the fused and torch
paths across auto-resets.
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

from swarp import DiscoveryScenario

SPEC = FusedSpec(
    scenario=DiscoveryScenario,
    fields=("obs", "rew", "done", "info:covered_frac"),
    grad_steps=4,
    grad_index=2,
    # Discovery's reward is pure coverage/touching counts (non-diff), so backprop
    # through obs, which depends on positions (hence actions).
    grad_backprop="obs",
)


def _env(device, fused, *, n_agents=6, n_targets=3, world_size=0.5, covering_range=0.25):
    scen = DiscoveryScenario(
        n_agents=n_agents,
        n_targets=n_targets,
        world_size=world_size,
        covering_range=covering_range,
        agents_per_target=2,
    )
    return fused_env(scen, device, fused, spec=SPEC)


def _run(env, n_steps, device, n_agents):
    return fused_rollout(env, n_steps, device, n_agents, SPEC)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_targets", [3, 5])
def test_fused_matches_torch(device, n_targets):
    kw = dict(device=device, n_targets=n_targets)
    fused = _run(_env(fused=True, **kw), 20, device, 6)
    torchp = _run(_env(fused=False, **kw), 20, device, 6)
    saw_cover = any(step[3].sum().item() > 0 for step in torchp)
    assert saw_cover, "test config covered no targets"
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, df, cff = f
        ot, rt, dt_, cft = r
        torch.testing.assert_close(of, ot, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"reward@{t}")
        assert torch.equal(df, dt_), f"done@{t}"
        assert torch.equal(cff, cft), f"covered_frac@{t}"  # discrete latch, exact


@pytest.mark.parametrize("device", DEVICES)
def test_fused_dense_touching_parity(device):
    # Small world -> touching pairs and coverage events; discrete quantities exact.
    kw = dict(device=device, n_agents=8, n_targets=4, world_size=0.25, covering_range=0.2)
    f = _run(_env(fused=True, **kw), 12, device, 8)
    t = _run(_env(fused=False, **kw), 12, device, 8)
    for f_i, t_i in zip(f, t, strict=True):
        assert torch.equal(f_i[2], t_i[2])  # done
        assert torch.equal(f_i[3], t_i[3])  # covered_frac
        torch.testing.assert_close(f_i[1], t_i[1], rtol=SPEC.rtol, atol=SPEC.atol)  # reward


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device, 6)
    b = _run(_env(device, fused=True), 10, device, 6)
    assert_fused_determinism(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    env = _env(device, fused=True, n_agents=4, n_targets=3, world_size=1.0)
    assert_grad_falls_back_to_torch(env, device, 4, SPEC)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_parity(device):
    """A standalone ``reset()``/``reset_at()`` must leave the same outputs as torch."""
    assert_reset_parity(lambda fused: _env(device, fused), device, 6, SPEC)
