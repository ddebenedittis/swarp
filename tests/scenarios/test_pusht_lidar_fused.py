"""Fused Warp kernels vs the torch oracle for PushTLidarScenario.

The fused path scans with the capture-safe Warp lidar against the engine's own box poses;
the torch oracle scans with ``lidar_scan`` against boxes it rebuilds from the body pose.
The two agree to float rounding, so a ray that grazes a box corner, the occlusion cone or
a coverage-sector boundary can be classified differently by the two. The comparisons
below therefore allow a small share of elements outside the declared parity bound
(:data:`MISMATCH_SHARE`) rather than none; in practice the share is zero.
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
)

from swarp import PushTLidarScenario

FIELDS = (
    "obs",
    "rew",
    "done",
    "info:tee_dist_to_goal",
    "info:tee_angle_error",
    "info:multiobj_reward",
    "info:team_coverage",
    "info:observer_visibility",
    "info:occlusion",
)

SPEC = FusedSpec(
    scenario=PushTLidarScenario,
    fields=FIELDS,
    grad_steps=4,
    grad_index=2,
    grad_backprop="obs",  # the reward is a difference of pose errors; obs is smooth
    substeps=8,  # Push-T's contact law needs it at dt=0.05
)

#: Largest share of elements allowed outside the parity bound (grazing-ray flips).
MISMATCH_SHARE = 2e-3


def _scen(n_agents=5, obs_level=1, **kw):
    # A small arena so the robots reach the T within a few steps; frame_stack 2 and a
    # finite comm_range exercise the ring and the communication gate.
    kw.setdefault("world_size", 0.5)
    kw.setdefault("frame_stack", 2)
    kw.setdefault("comm_range", 0.5)
    kw.setdefault("rot_away_penalty", 1.0)
    return PushTLidarScenario(n_agents=n_agents, obs_level=obs_level, **kw)


def _env(device, fused, **kw):
    return fused_env(_scen(**kw), device, fused, spec=SPEC)


def _rollout(env, n_steps, device):
    """Seeded no-grad rollout: every field plus the three trainer buffers, per step."""
    scen = env.scenario
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(SPEC.action_seed)
    out = []
    with torch.no_grad():
        for _ in range(n_steps):
            a = torch.empty(env.n_envs, env.n_agents, 2, device=device).uniform_(
                -1, 1, generator=gen
            )
            obs, rew, term, trunc, info = env.step(a)
            out.append(
                {
                    "obs": obs.clone(),
                    "rew": rew.clone(),
                    "done": (term | trunc).clone(),
                    **{k: v.clone() for k, v in info.items()},
                    "critic": scen.critic_state().clone(),
                    "roles": scen.agent_roles().clone(),
                    "aux": scen.aux_targets().clone(),
                }
            )
    return out


def _mismatch(a, b):
    """Share of elements outside the scenario's parity bound (exact for discrete ones)."""
    if a.dtype in (torch.bool, torch.uint8, torch.int32, torch.int64):
        return (a != b).float().mean().item()
    tol = SPEC.atol + SPEC.rtol * b.abs()
    return ((a - b).abs() > tol).float().mean().item()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("obs_level", [0, 1, 2])
@pytest.mark.parametrize("n_agents", [3, 5])
def test_fused_matches_torch(device, n_agents, obs_level):
    _assert_parity(device, n_agents=n_agents, obs_level=obs_level)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("obs_level", [1, 2])
@pytest.mark.parametrize(
    "fleet",
    [
        {"model": "diff_drive"},
        {"agent_shape": "box"},
        {"model": "diff_drive", "agent_shape": "box"},
        # A drive drawn per env and robot, a lidar blind zone ahead and a rear blind
        # sector: the per-env reset draws and the sensor model match the torch twin. The
        # accelerations are stiff enough for the robots to reach the T in four steps.
        {
            "model": "diff_drive", "agent_shape": "box", "servo_accel": (8.0, 20.0),
            "servo_ang_accel": (20.0, 60.0), "cmd_gain_w": (0.7, 1.0),
            "lidar_mount_offset": 0.03, "lidar_min_range": 0.06, "lidar_fov": 4.5,
        },
        # The real-robot knobs, with every draw the two paths share: per-robot fields of
        # view, a localisation bias (jitter and velocity noise off, as their streams
        # differ by path) and a command latency.
        {
            "model": "diff_drive", "agent_shape": "box", "servo_accel": (8.0, 20.0),
            "servo_ang_accel": (20.0, 60.0), "lidar_mount_offset": 0.03,
            "lidar_min_range": 0.02, "lidar_fov": (4.2, 2.0 * math.pi),
            "pose_noise": (0.5, 1.0), "pose_jitter_std": (0.0, 0.0), "vel_noise_std": 0.0,
            "cmd_latency": (0.0, 0.05),
        },
    ],
    ids=["unicycle", "box", "unicycle-box", "drive-sensor", "real"],
)
def test_fleet_fused_matches_torch(device, fleet, obs_level):
    """Unicycles and boxes: heading in the obs and critic, body-frame rays, box hits."""
    _assert_parity(device, obs_level=obs_level, **fleet)


def _assert_parity(device, **kw):
    fused = _rollout(_env(device, True, **kw), 20, device)
    torchp = _rollout(_env(device, False, **kw), 20, device)
    scen = _scen(**kw)
    assert fused[0]["obs"].shape[-1] == scen.obs_dim
    assert fused[0]["critic"].shape[-1] == scen.state_dim
    # The features have to fire, or parity on them proves nothing.
    assert any(s["tee_seen"].any() for s in fused), "no robot ever saw the T"
    assert any((s["occlusion"] > 0).any() for s in fused), "no occlusion ever measured"
    assert any((s["team_coverage"] > 0).any() for s in fused), "no coverage ever measured"
    moved = any(
        (fused[t]["tee_dist_to_goal"] - fused[0]["tee_dist_to_goal"]).abs().sum() > 0
        for t in range(1, 5)
    )
    assert moved, "T never moved; contact physics not exercised"
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        assert torch.equal(f["done"], r["done"]), f"done@{t}"
        assert torch.equal(f["roles"], r["roles"]), f"roles@{t}"
        for key in (
            "obs",
            "rew",
            "multiobj_reward",
            "critic",
            "aux",
            "tee_dist_to_goal",
            "tee_angle_error",
            "team_coverage",
            "observer_visibility",
            "occlusion",
            "tee_seen",
        ):
            share = _mismatch(f[key], r[key])
            assert share <= MISMATCH_SHARE, f"{key}@{t}: {share:.4f} of elements differ"


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    def run():
        env = _env(device, True, obs_level=2)
        return [tuple(s.values()) for s in _rollout(env, 10, device)]

    assert_fused_determinism(run(), run())


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    """A taped step takes the differentiable torch path, lidar oracle included."""
    env = _env(device, True, obs_level=2, world_size=1.0)
    assert_grad_falls_back_to_torch(env, device, env.n_agents, SPEC)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("obs_level", [0, 2])
def test_reset_parity(device, obs_level):
    """A standalone ``reset()``/``reset_at()`` leaves the same outputs as torch."""
    assert_reset_parity(lambda fused: _env(device, fused, obs_level=obs_level), device, 5, SPEC)
