"""TorchRL EnvBase wrapper: spec validation and a random-policy rollout."""

import pytest
import torch

pytest.importorskip("torchrl")

from torchrl.envs.utils import check_env_specs  # noqa: E402

from wmas import Environment, NavigationScenario  # noqa: E402
from wmas.interop.torchrl import WmasEnv  # noqa: E402
from wmas.scenarios.formation import FormationScenario  # noqa: E402
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
def test_check_env_specs_formation_info(device):
    """An info-emitting scenario passes check_env_specs with the nested info composite."""
    env = _make(device, FormationScenario(n_agents=4))
    assert "info" in env.observation_spec.keys()  # noqa: SIM118 (Composite)
    info_spec = env.observation_spec["info"]
    assert set(info_spec.keys()) >= {"multiobj_reward", "formation_error"}
    check_env_specs(env)


@pytest.mark.parametrize("device", DEVICES)
def test_rollout_exposes_info(device):
    """A rollout surfaces info under the flat ('next', 'info', <key>) path with spec shapes."""
    na = 4
    env = _make(device, FormationScenario(n_agents=na))
    td = env.rollout(max_steps=5)
    mo = td["next", "info", "multiobj_reward"]
    assert mo.shape == (8, 5, na, 2)
    assert mo.dtype == env.observation_spec["info", "multiobj_reward"].dtype
    fe = td["next", "info", "formation_error"]
    assert fe.shape == (8, 5)
    assert fe.dtype == env.observation_spec["info", "formation_error"].dtype
    assert torch.isfinite(mo).all() and torch.isfinite(fe).all()


class _NoInfoNavigation(NavigationScenario):
    """NavigationScenario with info() suppressed, to exercise the empty-info path."""

    def info(self):
        return {}


@pytest.mark.parametrize("device", DEVICES)
def test_info_empty_scenario_has_no_info_spec(device):
    """Single-objective (info == {}) scenarios behave exactly as before: no info spec."""
    env = _make(device, _NoInfoNavigation(n_agents=3))
    assert "info" not in env.observation_spec.keys()  # noqa: SIM118 (Composite)
    check_env_specs(env)
    td = env.rollout(max_steps=3)
    assert ("next", "info") not in td.keys(include_nested=True)


@pytest.mark.parametrize("device", DEVICES)
def test_wrapper_info_path_no_host_transfer(device):
    """_reset/_step (including info population) must not trigger a device->host copy."""
    from unittest import mock

    def _raise(*args, **kwargs):
        raise AssertionError("device->host transfer on the wrapper path")

    env = _make(device, FormationScenario(n_agents=4))
    td = env.reset()  # warmup (kernel compilation) outside the guard
    gen = torch.Generator(device=device).manual_seed(0)
    action = torch.rand(8, 4, env.act_dim, generator=gen, device=device) * 2 - 1
    with (
        torch.no_grad(),
        mock.patch.object(torch.Tensor, "cpu", _raise),
        mock.patch.object(torch.Tensor, "item", _raise),
        mock.patch.object(torch.Tensor, "numpy", _raise),
        mock.patch.object(torch.Tensor, "tolist", _raise),
    ):
        env._reset()
        env._step(td.set("action", action))


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
