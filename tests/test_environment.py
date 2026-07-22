"""Environment + NavigationScenario API behavior."""

import pytest
import torch

from wmas import DynamicsModel, Environment, NavigationScenario

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def make_env(device, n_envs=8, n_agents=3, **kw):
    scenario_kw = {
        k: kw.pop(k)
        for k in list(kw)
        if k
        in (
            "model",
            "n_obstacles",
            "neighbor_obs",
            "shared_reward",
            "world_size",
            "max_speed",
        )
    }
    scenario = NavigationScenario(n_agents=n_agents, **scenario_kw)
    return Environment(scenario, n_envs=n_envs, device=device, dt=0.1, seed=7, **kw)


@pytest.mark.parametrize("device", DEVICES)
def test_api_shapes_and_devices(device):
    env = make_env(device, n_envs=8, n_agents=3, n_obstacles=2)
    obs = env.reset()
    expected_dev = torch.device(device).type
    assert obs.shape[0] == 8 and obs.shape[1] == 3 and obs.device.type == expected_dev
    assert obs.dtype == torch.float32

    actions = torch.zeros(8, 3, 2, device=device)
    obs, rew, done, info = env.step(actions)
    assert obs.shape[:2] == (8, 3)
    assert rew.shape == (8, 3) and rew.device.type == expected_dev
    assert done.shape == (8,) and done.dtype == torch.bool and done.device.type == expected_dev
    assert info["dist_to_goal"].shape == (8, 3)
    # obs layout: pos(2) vel(2) cos/sin/angvel(3) goal_rel(2) + 5*neighbor_obs
    assert obs.shape[2] == 9 + 5 * env.scenario.neighbor_obs

    with pytest.raises(ValueError):
        env.step(torch.zeros(8, 4, 2, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_reset_determinism(device):
    env1 = make_env(device)
    env2 = make_env(device)
    torch.testing.assert_close(env1.reset(seed=123), env2.reset(seed=123))
    assert not torch.equal(env1.reset(seed=1), env2.reset(seed=2))


def test_reward_sign_toward_goal():
    """An isolated agent commanded straight at its goal earns positive shaping."""
    env = make_env("cpu", n_envs=4, n_agents=2, world_size=5.0)
    env.reset(seed=0)
    pos = env.world.state.pos
    goal_dir = env.world.goals - pos
    actions = torch.nn.functional.normalize(goal_dir, dim=-1) * 0.5
    _, rew, _, info = env.step(actions)
    # spawn separation is 3 radii, one 0.05-step can't create contact,
    # so reward = shaping only, and it must be positive for every agent
    assert (info["collisions"] == 0).all()
    assert (rew > 0).all()


def test_collision_penalty_applied():
    env = make_env("cpu", n_envs=1, n_agents=2, world_size=5.0)
    env.reset(seed=0)
    with torch.no_grad():  # start just outside contact reach, driving head-on
        env.world.state.pos[0, 0] = torch.tensor([-0.08, 0.0])
        env.world.state.pos[0, 1] = torch.tensor([0.08, 0.0])
        # teleporting invalidates the shaping baseline; rebase it
        env.scenario._prev_dist = (env.world.state.pos - env.world.goals).norm(dim=-1)
    actions = torch.tensor([[[0.5, 0.0], [-0.5, 0.0]]])
    # post-step distance = 0.16 - 2*0.05 = 0.06 < 2*radius = 0.1 -> touching
    _, rew, _, info = env.step(actions)
    assert (info["collisions"][0] >= 1).all()
    # collision penalty (-1 each) dominates the small shaping term
    assert (rew[0] < -0.5).all()


def test_done_on_goals_and_truncation():
    env = make_env("cpu", n_envs=4, n_agents=2, world_size=1.0, max_steps=200)
    env.reset(seed=3)
    done = None
    for _ in range(200):
        goal_dir = env.world.goals - env.world.state.pos
        actions = goal_dir.clamp(-1.0, 1.0) * 5.0  # P-controller, saturates at max_speed
        _, _, done, _ = env.step(actions)
        if bool(done.all()):
            break
    assert bool(done.all())
    assert bool((env._step_count < 200).all())  # reached goals, not truncated

    env2 = make_env("cpu", n_envs=2, n_agents=2, max_steps=5)
    env2.reset(seed=1)
    for _ in range(5):
        _, _, done, _ = env2.step(torch.zeros(2, 2, 2))
    assert bool(done.all())


def test_differentiable_through_env():
    env = make_env("cpu", n_envs=2, n_agents=2)
    env.reset(seed=0)
    actions = torch.zeros(2, 2, 2, requires_grad=True)
    obs, rew, done, _ = env.step(actions * 1.0)
    loss = obs.square().sum() + rew.sum()
    loss.backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() > 0


@pytest.mark.parametrize("model", [DynamicsModel.DIFF_DRIVE, DynamicsModel.KINEMATIC_BICYCLE])
def test_nonholonomic_navigation_smoke(model):
    env = make_env("cpu", n_envs=4, n_agents=3, model=model)
    env.reset(seed=0)
    for _ in range(10):
        obs, rew, done, _ = env.step(0.3 * torch.randn(4, 3, 2))
    assert torch.isfinite(obs).all() and torch.isfinite(rew).all()


@pytest.mark.parametrize("device", DEVICES)
def test_reset_at_masked(device):
    """reset_at zeroes step_count and randomizes only the masked envs."""
    env = make_env(device, n_envs=4, n_agents=3, world_size=5.0)
    env.reset(seed=0)
    for _ in range(3):
        env.step(0.2 * torch.ones(4, 3, 2, device=device))
    pos_before = env.world.state.pos.clone()
    count_before = env._step_count.clone()
    assert (count_before == 3).all()

    mask = torch.tensor([True, False, True, False], device=device)
    env.reset_at(mask)

    torch.testing.assert_close(
        env._step_count, torch.tensor([0, 3, 0, 3], dtype=torch.int32, device=device)
    )
    # unreset envs untouched; reset envs moved
    torch.testing.assert_close(env.world.state.pos[1], pos_before[1])
    torch.testing.assert_close(env.world.state.pos[3], pos_before[3])
    assert not torch.equal(env.world.state.pos[0], pos_before[0])
    assert not torch.equal(env.world.state.pos[2], pos_before[2])


@pytest.mark.parametrize("device", DEVICES)
def test_auto_reset_on_truncation(device):
    """auto_reset=True resets done envs in-place; step_count returns to zero."""
    env = make_env(device, n_envs=3, n_agents=2, max_steps=3, auto_reset=True)
    env.reset(seed=0)
    done = None
    for _ in range(3):
        _, _, done, _ = env.step(torch.zeros(3, 2, 2, device=device))
    assert bool(done.all())  # truncated at max_steps
    assert (env._step_count == 0).all()  # and auto-reset back to zero
    # keeps running afterwards
    _, _, done2, _ = env.step(torch.zeros(3, 2, 2, device=device))
    assert (env._step_count == 1).all()


def test_partial_reset_preserves_other_envs_shaping_baseline():
    """Regression: reset_at must not rebase a non-reset env's _prev_dist."""
    env = make_env("cpu", n_envs=2, n_agents=2, world_size=5.0)
    env.reset(seed=0)
    prev1_before = env.scenario._prev_dist[1].clone()
    # Move env 1 without refreshing the cache, so its baseline != current dist.
    with torch.no_grad():
        env.world.state.pos[1] += 1.0
    env.reset_at(torch.tensor([True, False]))
    # env 1 was not reset: its shaping baseline must be exactly preserved
    # (the old code clobbered _prev_dist for all envs during the reset refresh).
    torch.testing.assert_close(env.scenario._prev_dist[1], prev1_before)


def test_radius_graph_api():
    env = make_env("cpu", n_envs=3, n_agents=4)
    env.reset(seed=0)
    edges = env.radius_graph()
    assert edges.shape[0] == 2 and edges.dtype == torch.int64
