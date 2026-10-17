"""Fused Warp ring-road kernels must match the torch reference path (bit-close).

Ring road's own additions on top of the shared parity suite: the IDM humans are a second
dynamical system integrated here, so the suite pins the premise the scenario exists for
(``test_wave_persists_at_22_and_dies_at_6``), that a crash is detected and survived, and
that the leader search is entered with RL cars and humans interleaved rather than
compared on a trivially ordered ring.
"""

import math

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

from swarp.core.environment import Environment
from swarp.rl.recipes import ringroad_idm_driver
from swarp.scenarios.ringroad import RingRoadScenario

SPEC = FusedSpec(
    scenario=RingRoadScenario,
    fields=("obs", "rew", "done", "info:speed_std", "info:mean_speed", "info:crashed"),
    grad_steps=4,
    grad_index=2,
    # ``obs`` rather than ``rew``: the speed term is a mean over the humans, which the RL
    # action reaches only through the IDM's leader input, and the crash term is a step
    # function. ``obs`` carries the car's own tangential speed and lane offset.
    grad_backprop="obs",
    action_seed=2,
)


def _scen(*, n_agents=1, difficulty=1.0, **kw):
    scen = RingRoadScenario(n_agents=n_agents, **kw)
    scen.difficulty = difficulty
    return scen


def _env(device, fused, *, n_agents=1, dtype=torch.float32, **kw):
    return fused_env(_scen(n_agents=n_agents, **kw), device, fused, spec=SPEC, dtype=dtype)


def _make(device, *, n_envs=8, dt=0.1, fused=True, **kw):
    return Environment(
        _scen(**kw),
        n_envs=n_envs,
        device=device,
        dt=dt,
        seed=0,
        auto_reset=False,
        dtype=torch.float32,
        fused=fused,
    )


# ------------------------------------------------------------------- parity


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_agents", [1, 3])
def test_fused_matches_torch(device, n_agents):
    """``n_agents=3`` interleaves RL cars with humans, so the leader search mixes the
    engine-state rows with the scenario rows on both paths."""
    fused = fused_rollout(_env(device, True, n_agents=n_agents), 20, device, n_agents, SPEC)
    torchp = fused_rollout(_env(device, False, n_agents=n_agents), 20, device, n_agents, SPEC)
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, df, stdf, meanf, crf = f
        ot, rt, dt_, stdt, meant, crt = r
        torch.testing.assert_close(of, ot, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"reward@{t}")
        assert torch.equal(df, dt_), f"done@{t}"
        assert torch.equal(crf, crt), f"crashed@{t}"
        torch.testing.assert_close(stdf, stdt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"std@{t}")
        torch.testing.assert_close(meanf, meant, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"mean@{t}")


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = fused_rollout(_env(device, True), 10, device, 1, SPEC)
    b = fused_rollout(_env(device, True), 10, device, 1, SPEC)
    assert_fused_determinism(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    assert_grad_falls_back_to_torch(_env(device, True), device, 1, SPEC)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_parity(device):
    assert_reset_parity(lambda fused: _env(device, fused), device, 1, SPEC)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [False, True])
def test_multiobj_reward_sums_to_the_scalar_reward(device, fused):
    env = _env(device, fused, n_agents=2)
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(SPEC.action_seed)
    with torch.no_grad():
        for _ in range(12):
            act = torch.empty(env.n_envs, 2, 2, device=device).uniform_(-1, 1, generator=gen)
            _, rew, _, _, info = env.step(act)
            mo = info["multiobj_reward"]
            assert mo.shape == (env.n_envs, 2, 5)
            torch.testing.assert_close(mo.sum(-1), rew, rtol=SPEC.rtol, atol=SPEC.atol)


# --------------------------------------------------------------- the traffic model


def _drive(env, steps, action_fn=None):
    """Roll ``env`` under ``action_fn`` (default: the IDM driver); return the info trace."""
    fn = (action_fn or ringroad_idm_driver())(env)
    obs = env.reset(seed=0)
    trace = []
    with torch.no_grad():
        for _ in range(steps):
            obs, _, _, _, info = env.step(fn(obs))
            trace.append({k: v.clone() for k, v in info.items()})
    return trace


@pytest.mark.parametrize("device", DEVICES)
def test_wave_persists_at_22_and_dies_at_6(device):
    """The premise. At the Sugiyama density the uniform IDM flow is string-unstable on a
    22-car ring and stable on a 6-car one, so the same reset wave grows at 22 and decays
    at 6. If this fails, a policy trained on the small ring is not missing anything."""
    std = {}
    for n in (6, 22):
        env = _make(device, n_vehicles=n)
        trace = _drive(env, 600)
        std[n] = trace[-1]["speed_std"]
        assert not any(t["crashed"].any() for t in trace), f"IDM ring of {n} crashed"
    assert (std[22] > 1.0).all(), f"no wave at N=22: {std[22]}"
    assert (std[6] < 0.05).all(), f"the wave did not die at N=6: {std[6]}"


@pytest.mark.parametrize("device", DEVICES)
def test_humans_only_ring_terminates_only_where_stable(device):
    """``done`` is reachable (the stable ring reaches it) and not free (the unstable one
    never does) under the humans-only arm."""
    for n, expect in ((6, True), (22, False)):
        env = _make(device, n_vehicles=n)
        fn = ringroad_idm_driver()(env)
        obs = env.reset(seed=0)
        hit = torch.zeros(env.n_envs, dtype=torch.bool, device=device)
        with torch.no_grad():
            for _ in range(400):
                obs, _, term, _, _ = env.step(fn(obs))
                hit |= term
        if expect:
            assert hit.all(), f"the stable {n}-car ring never terminated: {hit}"
        else:
            assert not hit.any(), f"the unstable {n}-car ring terminated: {hit}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [False, True])
def test_crash_is_detected_and_survived(device, fused):
    """An RL car that floors it on the ring rear-ends its leader. The crash term fires,
    nothing goes non-finite through the overlap (the IDM gap floor), and no human
    reverses or exceeds ``v_max``."""
    env = _make(device, difficulty=0.0, fused=fused)
    scen = env.scenario

    def floor_it(env):
        steer = math.atan(scen.wheelbase / float(scen.ring_length_min / (2 * math.pi)))
        a = torch.tensor([[[1.0, steer]]], device=device).expand(env.n_envs, 1, 2)
        return lambda obs: a

    trace = _drive(env, 150, floor_it)
    assert any(t["crashed"].all() for t in trace), "no crash was ever detected"
    assert any((t["multiobj_reward"][..., 3] < 0).all() for t in trace)
    for t in trace:
        assert all(torch.isfinite(v).all() for v in t.values())
    assert scen.idm_v.min().item() >= 0.0 and scen.idm_v.max().item() <= scen.v_max


@pytest.mark.parametrize("device", DEVICES)
def test_rl_car_cannot_reverse(device):
    """``min_speed=0``: full braking parks the car, it never rolls backwards."""
    env = _make(device)
    brake = torch.tensor([[[-1.0, 0.0]]], device=device).expand(env.n_envs, 1, 2)
    _drive(env, 80, lambda env: lambda obs: brake)
    assert env.world.state.speed.min().item() == 0.0


# ------------------------------------------------------------------ contracts


def test_needs_at_least_one_human():
    with pytest.raises(ValueError, match="at least one human"):
        RingRoadScenario(n_agents=4, n_vehicles=4)


def test_default_ring_holds_any_fleet_at_sugiyama_density():
    """The benchmark sweep builds every scenario with up to 64 agents."""
    scen = RingRoadScenario(n_agents=64)
    assert scen.n_vehicles == 65
    assert scen.ring_length_min == pytest.approx(65 * 230 / 22)


def test_ring_too_tight_to_steer_raises():
    with pytest.raises(ValueError, match="turning radius"):
        RingRoadScenario(n_vehicles=6, max_steer=0.2)


def test_ring_length_and_range_are_exclusive():
    with pytest.raises(ValueError, match="not both"):
        RingRoadScenario(ring_length=230.0, ring_length_min=220.0, ring_length_max=270.0)
    with pytest.raises(ValueError, match="together"):
        RingRoadScenario(ring_length_min=220.0)


def test_obs_dim_does_not_depend_on_ring_size():
    """What lets one actor be evaluated at every N."""
    dims = {RingRoadScenario(n_vehicles=n, n_agents=a).obs_dim for n in (6, 22) for a in (1, 4)}
    assert dims == {9}


@pytest.mark.parametrize("device", DEVICES)
def test_ring_length_is_drawn_per_env(device):
    env = _make(device, n_envs=64, ring_length_min=220.0, ring_length_max=270.0)
    env.reset(seed=0)
    L = env.scenario.ring_L
    assert L.min().item() >= 220.0 and L.max().item() <= 270.0 and L.std().item() > 1.0
    torch.testing.assert_close(env.scenario.ring_R * 2 * math.pi, L)
