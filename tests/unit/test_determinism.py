"""Same seed + same actions -> bit-identical trajectories."""

import pytest
import torch
from conftest import DEVICES

from swarp import Environment, NavigationScenario


def run_trajectory(device, seed, T=100):
    scenario = NavigationScenario(n_agents=8, n_obstacles=2, world_size=0.8)  # dense: contacts
    env = Environment(scenario, n_envs=16, device=device, dt=0.05, substeps=2, seed=seed)
    obs = env.reset(seed=seed)
    gen = torch.Generator().manual_seed(seed)
    obs_trace, rew_trace, pos_trace = [obs], [], []
    with torch.no_grad():
        for _ in range(T):
            actions = torch.randn(16, 8, 2, generator=gen).to(device)
            obs, rew, *_ = env.step(actions)
            obs_trace.append(obs)
            rew_trace.append(rew)
            pos_trace.append(env.world.state.pos.clone())
    return obs_trace, rew_trace, pos_trace


@pytest.mark.parametrize("device", DEVICES)
def test_bit_identical_trajectories(device):
    a = run_trajectory(device, seed=1234)
    b = run_trajectory(device, seed=1234)
    for ta, tb in zip(a, b, strict=True):
        for xa, xb in zip(ta, tb, strict=True):
            assert torch.equal(xa, xb)  # bit-exact, not allclose


@pytest.mark.parametrize("device", DEVICES)
def test_different_seeds_differ(device):
    a = run_trajectory(device, seed=1, T=5)
    b = run_trajectory(device, seed=2, T=5)
    assert not torch.equal(a[2][-1], b[2][-1])
