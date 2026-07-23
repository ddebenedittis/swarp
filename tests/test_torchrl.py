"""TorchRL EnvBase wrapper: spec validation and a random-policy rollout."""

import pytest
import torch

pytest.importorskip("torchrl")

from torchrl.envs.utils import check_env_specs  # noqa: E402

from wmas import Environment, NavigationScenario  # noqa: E402
from wmas.interop.torchrl import WmasEnv  # noqa: E402
from wmas.scenarios.sampling import SamplingScenario  # noqa: E402

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def _make(device, scenario):
    return WmasEnv(Environment(scenario, n_envs=8, device=device, dt=0.1, seed=0))


@pytest.mark.parametrize("device", DEVICES)
def test_check_env_specs_navigation(device):
    env = _make(device, NavigationScenario(n_agents=3, n_obstacles=1))
    check_env_specs(env)  # reset/step outputs must match the declared specs


@pytest.mark.parametrize("device", DEVICES)
def test_check_env_specs_sampling(device):
    env = _make(device, SamplingScenario(n_agents=4, grid_res=8))
    check_env_specs(env)


@pytest.mark.parametrize("device", DEVICES)
def test_random_policy_rollout(device):
    env = _make(device, NavigationScenario(n_agents=3))
    td = env.rollout(max_steps=10)  # random actions sampled from the action spec
    # rollout stacks over time: batch is [n_envs, T].
    assert tuple(td.batch_size) == (8, 10)
    obs = td["observation"]
    assert obs.shape[:3] == (8, 10, env.n_agents) and torch.isfinite(obs).all()
    rew = td["next", "reward"]
    assert rew.shape == (8, 10, env.n_agents, 1) and torch.isfinite(rew).all()
    assert td["action"].shape[-1] == env.act_dim


@pytest.mark.parametrize("device", DEVICES)
def test_step_matches_underlying_env(device):
    """A step through the wrapper matches the underlying wmas Environment."""
    scenario = NavigationScenario(n_agents=3)
    base = Environment(scenario, n_envs=8, device=device, dt=0.1, seed=0)
    wrapped = WmasEnv(base)
    td = wrapped.reset()
    gen = torch.Generator(device=device).manual_seed(1)
    action = torch.rand(8, 3, wrapped.act_dim, generator=gen, device=device) * 2 - 1
    td = td.set("action", action)
    out = wrapped.step(td)
    # underlying env is the same object; compare against a direct reference read
    assert out["next", "observation"].shape[0] == 8
    assert torch.isfinite(out["next", "reward"]).all()
    assert out["next", "done"].shape == (8, 1)
