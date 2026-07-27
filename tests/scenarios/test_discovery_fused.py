"""Fused Warp obs/reward kernels for DiscoveryScenario must match the torch path.

Discovery carries a persistent monotonic ``covered`` latch and a per-target
coverage reduction; the discrete coverage/touching flags (and the one-off
``newly`` covering reward) must stay bit-identical between the fused and torch
paths across auto-resets.
"""

import pytest
import torch
from conftest import DEVICES

from wmas import DiscoveryScenario, Environment


def _env(device, fused, *, n_agents=6, n_targets=3, n_envs=24, world_size=0.5, covering_range=0.25):
    scen = DiscoveryScenario(
        n_agents=n_agents,
        n_targets=n_targets,
        world_size=world_size,
        covering_range=covering_range,
        agents_per_target=2,
    )
    return Environment(
        scen,
        n_envs=n_envs,
        device=device,
        dt=0.05,
        substeps=1,
        seed=0,
        auto_reset=True,
        max_steps=5,
        fused=fused,
    )


def _run(env, n_steps, device, n_agents):
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(2)
    out = []
    with torch.no_grad():
        for _ in range(n_steps):
            a = torch.empty(env.n_envs, n_agents, 2, device=device).uniform_(-1, 1, generator=gen)
            obs, rew, done, info = env.step(a)
            out.append((obs.clone(), rew.clone(), done.clone(), info["covered_frac"].clone()))
    return out


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
        torch.testing.assert_close(of, ot, rtol=1e-5, atol=1e-6, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=1e-5, atol=1e-6, msg=f"reward@{t}")
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
        torch.testing.assert_close(f_i[1], t_i[1], rtol=1e-5, atol=1e-6)  # reward


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device, 6)
    b = _run(_env(device, fused=True), 10, device, 6)
    for (o1, r1, d1, c1), (o2, r2, d2, c2) in zip(a, b, strict=True):
        assert torch.equal(o1, o2) and torch.equal(r1, r2)
        assert torch.equal(d1, d2) and torch.equal(c1, c2)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    env = _env(device, fused=True, n_agents=4, n_targets=3, world_size=1.0)
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(5)
    for i in range(4):
        act = torch.empty(env.n_envs, 4, 2, device=device).uniform_(-1, 1, generator=gen)
        if i == 2:
            act = act.clone().requires_grad_(True)
            with torch.enable_grad():
                obs, rew, done, _ = env.step(act)
            # Discovery's reward is pure coverage/touching counts (non-diff), so
            # backprop through obs, which depends on positions (hence actions).
            assert obs.requires_grad  # torch reference path was taken
            obs.pow(2).sum().backward()
            assert act.grad is not None
        else:
            with torch.no_grad():
                obs, rew, done, _ = env.step(act)
            assert torch.isfinite(rew).all()
