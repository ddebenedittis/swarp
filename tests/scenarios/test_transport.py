"""Transport scenario: agents push a movable package to a goal."""

import pytest
import torch
from conftest import DEVICES

from wmas import Environment
from wmas.scenarios.transport import TransportScenario


def _env(device, n_envs=4, **kw):
    # fused=False on purpose: this suite exercises the *torch reference* package
    # integration (``_refresh``), poking ``pkg_*`` directly and reading it back. The
    # fused parity of that path is what test_transport_fused.py is for.
    return Environment(
        TransportScenario(n_agents=3, n_packages=1, **kw),
        n_envs=n_envs,
        device=device,
        dt=0.05,
        seed=1,
        fused=False,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_transport_api_and_finiteness(device):
    env = _env(device, n_envs=8)
    obs = env.reset()
    assert obs.shape[:2] == (8, 3) and torch.isfinite(obs).all()
    gen = torch.Generator(device=device).manual_seed(0)
    for _ in range(10):
        a = torch.rand(8, 3, env.world.act_dim, generator=gen, device=device) * 2 - 1
        obs, rew, done, info = env.step(a)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
        assert rew.shape == (8, 3) and done.shape == (8,)
        assert info["package_dist_to_goal"].shape == (8, 1)


@pytest.mark.parametrize("device", DEVICES)
def test_transport_determinism(device):
    def run():
        env = _env(device, n_envs=4)
        env.reset(seed=4)
        gen = torch.Generator(device=device).manual_seed(2)
        out = []
        for _ in range(8):
            a = torch.rand(4, 3, env.world.act_dim, generator=gen, device=device) * 2 - 1
            obs, rew, _, _ = env.step(a)
            out += [obs, rew, env.scenario.pkg_pos.clone()]
        return out

    for x, y in zip(run(), run(), strict=True):
        assert torch.equal(x, y)


@pytest.mark.parametrize("device", DEVICES)
def test_agents_push_package_toward_goal(device):
    """Agents behind the package, driving toward the goal, move it closer."""
    scenario = TransportScenario(
        n_agents=4, n_packages=1, package_radius=0.12, agent_radius=0.05, world_size=1.0
    )
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset()
    # place package at origin, goal to the +x, agents just behind it (-x side)
    scenario.pkg_pos = torch.tensor([[[0.0, 0.0]]], device=device, dtype=torch.float32)
    scenario.pkg_vel = torch.zeros_like(scenario.pkg_vel)
    scenario.goal = torch.tensor([[[0.7, 0.0]]], device=device, dtype=torch.float32)
    ax = torch.tensor(
        [[[-0.22, -0.08], [-0.22, 0.0], [-0.22, 0.08], [-0.30, 0.0]]],
        device=device,
        dtype=torch.float32,
    )
    scenario.world.state = scenario.world.state._replace(pos=ax)
    scenario._install_obstacles()
    scenario._prev_dist = None
    scenario._refresh(integrate=False)
    d0 = scenario._cache["dist_to_goal"].item()

    actions = torch.tensor([[[1.0, 0.0]] * 4], device=device, dtype=torch.float32)
    for _ in range(40):
        env.step(actions)
    d1 = scenario._cache["dist_to_goal"].item()
    assert scenario.pkg_pos[0, 0, 0].item() > 0.05  # package moved in +x
    assert d1 < d0 - 0.05  # and got closer to the goal


@pytest.mark.parametrize("device", DEVICES)
def test_transport_shaping_reward_positive_when_closer(device):
    """Global reward is positive on a step that moves the package toward goal."""
    scenario = TransportScenario(n_agents=3, n_packages=1)
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, seed=0, fused=False)
    env.reset()
    scenario.goal = torch.tensor([[[0.8, 0.0]]], device=device, dtype=torch.float32)
    scenario.pkg_pos = torch.tensor([[[0.0, 0.0]]], device=device, dtype=torch.float32)
    scenario._prev_dist = None
    scenario._refresh(integrate=False)
    # manually advance the package toward the goal and refresh (no integrate)
    scenario._prev_dist = scenario._cache["dist_to_goal"].detach().clone()
    scenario.pkg_pos = torch.tensor([[[0.1, 0.0]]], device=device, dtype=torch.float32)
    scenario._refresh(integrate=False)
    assert scenario.global_reward()[0].item() > 0.0


def test_transport_differentiable_rollout():
    """BPTT: the package-to-goal loss backprops to the action sequence."""
    scenario = TransportScenario(n_agents=3, n_packages=1)
    env = Environment(scenario, n_envs=2, device="cpu", dt=0.05, seed=0, fused=False)
    env.reset()
    actions = torch.zeros(2, 3, env.world.act_dim, requires_grad=True)
    loss = torch.zeros((), dtype=torch.float32)
    for _ in range(4):
        env.step(actions)
        loss = scenario._cache["dist_to_goal"].sum()
    loss.backward()
    assert actions.grad is not None and torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() > 0.0
