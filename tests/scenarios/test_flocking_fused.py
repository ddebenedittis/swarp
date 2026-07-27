"""Fused Warp obs/reward kernel for FlockingScenario must match the torch path."""

import pytest
import torch
from conftest import DEVICES

from wmas import Environment, FlockingScenario


def _env(device, fused, *, n_agents=6, n_envs=24, neighbor_obs=4, world_size=0.6):
    scen = FlockingScenario(
        n_agents=n_agents,
        neighbor_obs=neighbor_obs,
        world_size=world_size,
        neighbor_radius=0.4,
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
            out.append((obs.clone(), rew.clone(), info["crowding"].clone()))
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("neighbor_obs", [1, 4])
def test_fused_matches_torch(device, neighbor_obs):
    kw = dict(device=device, neighbor_obs=neighbor_obs)
    fused = _run(_env(fused=True, **kw), 20, device, 6)
    torchp = _run(_env(fused=False, **kw), 20, device, 6)
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, cf = f
        ot, rt, ct = r
        torch.testing.assert_close(of, ot, rtol=1e-5, atol=1e-6, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=1e-5, atol=1e-6, msg=f"reward@{t}")
        torch.testing.assert_close(cf, ct, rtol=1e-5, atol=1e-6, msg=f"crowding@{t}")


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
        torch.testing.assert_close(f_i[0], t_i[0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(f_i[1], t_i[1], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(f_i[2], t_i[2], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device, 6)
    b = _run(_env(device, fused=True), 10, device, 6)
    for (o1, r1, c1), (o2, r2, c2) in zip(a, b, strict=True):
        assert torch.equal(o1, o2) and torch.equal(r1, r2) and torch.equal(c1, c2)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    """A grad-enabled step must take the differentiable torch path (fused off)."""
    env = _env(device, fused=True, n_agents=4, world_size=1.0)
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(5)
    for i in range(4):
        act = torch.empty(env.n_envs, 4, 2, device=device).uniform_(-1, 1, generator=gen)
        if i == 2:
            act = act.clone().requires_grad_(True)
            with torch.enable_grad():
                obs, rew, done, _ = env.step(act)
            assert obs.requires_grad  # torch reference path was taken
            rew.pow(2).sum().backward()
            assert act.grad is not None
        else:
            with torch.no_grad():
                obs, rew, done, _ = env.step(act)
            assert torch.isfinite(rew).all()
