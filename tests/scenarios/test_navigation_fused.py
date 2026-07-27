"""Fused Warp obs/reward kernels must match the torch reference path (bit-close)."""

import numpy as np
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

from wmas import NavigationScenario
from wmas.dynamics.base import P_RADIUS, per_env_float_template

SPEC = FusedSpec(
    fields=(
        "obs",
        "rew",
        "done",
        "info:collisions",
        "info:dist_to_goal",
        "info:on_goal",
        "info:neighbor_overflow",
    ),
    grad_steps=6,
    grad_index=3,
    grad_backprop="rew",  # navigation's reward is differentiable shaping
)


def _env(
    device,
    fused,
    *,
    n_agents=5,
    shared_reward=False,
    neighbor_obs=2,
    world_size=0.6,
    max_neighbors=None,
    n_obstacles=0,
    dtype=torch.float32,
):
    scen = NavigationScenario(
        n_agents=n_agents,
        shared_reward=shared_reward,
        neighbor_obs=neighbor_obs,
        world_size=world_size,
        n_obstacles=n_obstacles,
        neighbor_method="brute",  # pin the backend so both paths see identical lists
    )
    if max_neighbors is not None:
        scen.neighbor_obs = neighbor_obs
    return fused_env(scen, device, fused, spec=SPEC, dtype=dtype)


def _run(env, n_steps, device, n_agents, dtype=torch.float32):
    return fused_rollout(env, n_steps, device, n_agents, SPEC, dtype=dtype)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shared_reward", [False, True])
@pytest.mark.parametrize("neighbor_obs", [1, 3])
def test_fused_matches_torch(device, shared_reward, neighbor_obs):
    kw = dict(device=device, shared_reward=shared_reward, neighbor_obs=neighbor_obs)
    fused = _run(_env(fused=True, **kw), 20, device, 5)
    torchp = _run(_env(fused=False, **kw), 20, device, 5)
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, df, cf, gf, ogf, ovf = f
        ot, rt, dt_, ct, gt, ogt, ovt = r
        torch.testing.assert_close(of, ot, rtol=1e-5, atol=1e-6, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=1e-5, atol=1e-6, msg=f"reward@{t}")
        assert torch.equal(df, dt_), f"done@{t}"
        # touching / on_goal are discrete; must match exactly on non-degenerate configs
        assert torch.equal(cf, ct), f"collisions@{t}"
        assert torch.equal(ogf, ogt), f"on_goal@{t}"
        assert torch.equal(ovf, ovt), f"overflow@{t}"
        torch.testing.assert_close(gf, gt, rtol=1e-5, atol=1e-6, msg=f"dist@{t}")


@pytest.mark.parametrize("device", DEVICES)
def test_fused_dense_touching_parity(device):
    # Many agents in a small world -> lots of touching pairs; touching/reward
    # must still match the torch reference exactly (overflow stays zero because
    # navigation sizes max_neighbors to n_agents).
    scen_kw = dict(device=device, n_agents=8, world_size=0.25, neighbor_obs=2)
    f = _run(_env(fused=True, **scen_kw), 12, device, 8)
    t = _run(_env(fused=False, **scen_kw), 12, device, 8)
    saw_touch = any(step[3].sum().item() > 0 for step in t)
    assert saw_touch, "test config did not induce any touching pairs"
    for f_i, t_i in zip(f, t, strict=True):
        assert torch.equal(f_i[3], t_i[3])  # collisions/touching
        assert torch.equal(f_i[6], t_i[6])  # overflow
        torch.testing.assert_close(f_i[1], t_i[1], rtol=1e-5, atol=1e-6)  # reward


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device, 5)
    b = _run(_env(device, fused=True), 10, device, 5)
    assert_fused_determinism(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_per_env_params(device):
    # Randomize per-env radius; fused per-env kernel must match torch per-env path.
    def build(fused):
        env = _env(device, fused=fused, n_agents=5, world_size=0.5)
        env.reset(seed=0)
        configs = env.world.agent_configs
        floats = per_env_float_template(configs, env.n_envs)
        rng = np.random.default_rng(0)
        floats[..., P_RADIUS] = 0.03 + 0.02 * rng.random(floats[..., P_RADIUS].shape)
        env.world.stepper.set_agent_params_per_env(
            torch.tensor(floats, dtype=torch.float32, device=device)
        )
        env.reset(seed=0)  # re-seed spawn after installing params
        return env

    fe, te = build(True), build(False)
    gen_seed = 3
    outs = []
    for env in (fe, te):
        env.reset(seed=0)
        gen = torch.Generator(device=device).manual_seed(gen_seed)
        steps = []
        with torch.no_grad():
            for _ in range(10):
                act = torch.empty(env.n_envs, 5, 2, device=device).uniform_(-1, 1, generator=gen)
                o, r, d, info = env.step(act)
                steps.append((o.clone(), r.clone(), info["collisions"].clone()))
        outs.append(steps)
    for (of, rf, cf), (ot, rt, ct) in zip(outs[0], outs[1], strict=True):
        assert torch.equal(cf, ct)
        torch.testing.assert_close(of, ot, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(rf, rt, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_interleave_shaping_continuous(device):
    """Interleaving a grad step (torch path, _fused_active off) with fused no-grad
    steps must not crash and must keep the shaping baseline continuous (shared
    _prev_dist). Grad off entirely -> fused; a grad step transparently uses the
    differentiable reference. The fused-only run and the interleaved run agree on
    the shared-across-both steps (they share _prev_dist storage)."""
    env = _env(device, fused=True, n_agents=4, world_size=1.0)
    assert_grad_falls_back_to_torch(env, device, 4, SPEC)
