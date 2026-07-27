"""Fused Warp obs/reward + movable-body kernels for TransportScenario.

Transport fuses the package spring-damper/Euler physics too, so the fused and
torch package trajectories are only allclose (different agent-reduction order),
kept bounded by the damping. Discrete flags (done/on_goal) still match.
"""

import pytest
import torch
from conftest import DEVICES

from wmas import Environment, TransportScenario


def _env(device, fused, *, n_agents=4, n_packages=1, n_envs=24, world_size=0.5):
    scen = TransportScenario(
        n_agents=n_agents,
        n_packages=n_packages,
        world_size=world_size,
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
            out.append(
                (obs.clone(), rew.clone(), done.clone(), info["package_dist_to_goal"].clone())
            )
    return out


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
        torch.testing.assert_close(of, ot, rtol=1e-5, atol=1e-5, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=1e-5, atol=1e-5, msg=f"reward@{t}")
        assert torch.equal(df, dt_), f"done@{t}"
        torch.testing.assert_close(gf, gt, rtol=1e-5, atol=1e-5, msg=f"pkg_dist@{t}")


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device, 4)
    b = _run(_env(device, fused=True), 10, device, 4)
    for (o1, r1, d1, g1), (o2, r2, d2, g2) in zip(a, b, strict=True):
        assert torch.equal(o1, o2) and torch.equal(r1, r2)
        assert torch.equal(d1, d2) and torch.equal(g1, g2)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    env = _env(device, fused=True, n_agents=4, world_size=1.0)
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(5)
    for i in range(4):
        act = torch.empty(env.n_envs, 4, 2, device=device).uniform_(-1, 1, generator=gen)
        if i == 2:
            act = act.clone().requires_grad_(True)
            with torch.enable_grad():
                obs, rew, done, _ = env.step(act)
            assert obs.requires_grad  # torch reference (differentiable body) path
            obs.pow(2).sum().backward()
            assert act.grad is not None
        else:
            with torch.no_grad():
                obs, rew, done, _ = env.step(act)
            assert torch.isfinite(rew).all()
