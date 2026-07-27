"""The hot loop must perform no device->host transfers.

Two independent checks:
1. An API-level guard (all devices): host-transfer entry points
   (``Tensor.cpu/.item/.numpy/.tolist``, ``wp.array.numpy``, ``bool(Tensor)``)
   are patched to raise while the env steps.
2. A CUDA profiler check: zero Memcpy DtoH events during steady-state stepping.
"""

import contextlib
from unittest import mock

import pytest
import torch
import warp as wp
from conftest import DEVICES

from wmas import Environment, NavigationScenario


def make_env(device, n_envs=32):
    scenario = NavigationScenario(n_agents=8, n_obstacles=2)
    return Environment(scenario, n_envs=n_envs, device=device, dt=0.05, substeps=2, seed=0)


def _raise(*args, **kwargs):
    raise AssertionError("device->host transfer on the hot path")


@contextlib.contextmanager
def forbid_host_transfers():
    with (
        mock.patch.object(torch.Tensor, "cpu", _raise),
        mock.patch.object(torch.Tensor, "item", _raise),
        mock.patch.object(torch.Tensor, "numpy", _raise),
        mock.patch.object(torch.Tensor, "tolist", _raise),
        mock.patch.object(torch.Tensor, "__bool__", _raise),
        mock.patch.object(wp.array, "numpy", _raise),
    ):
        yield


@pytest.mark.parametrize("device", DEVICES)
def test_hot_loop_api_guard(device):
    env = make_env(device)
    env.reset(seed=0)
    actions = torch.zeros(32, 8, 2, device=device)
    env.step(actions)  # warmup outside the guard (kernel compilation etc.)
    with torch.no_grad(), forbid_host_transfers():
        for _ in range(20):
            obs, rew, done, info = env.step(actions)
    assert obs.device.type == torch.device(device).type


@pytest.mark.parametrize("device", DEVICES)
def test_reset_no_host_transfer(device):
    """reset / reset_at must also stay host-sync-free (no .any() in spawn)."""
    env = make_env(device)
    env.reset(seed=0)  # warmup (kernel compilation) outside the guard
    with torch.no_grad(), forbid_host_transfers():
        env.reset(seed=1)
        env.reset_at(torch.ones(env.n_envs, dtype=torch.bool, device=device))
        env.reset_at(torch.zeros(env.n_envs, dtype=torch.bool, device=device))


@pytest.mark.gpu
def test_no_memcpy_dtoh_in_profile():
    from torch.profiler import ProfilerActivity, profile

    env = make_env("cuda:0")
    env.reset(seed=0)
    actions = torch.zeros(32, 8, 2, device="cuda:0")
    with torch.no_grad():
        for _ in range(5):  # warmup: module load, grid reserve, mempool growth
            env.step(actions)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                env.step(actions)
            torch.cuda.synchronize()
    dtoh = [
        e
        for e in prof.events()
        if "memcpy" in e.name.lower() and "dtoh" in e.name.lower().replace(" ", "")
    ]
    assert not dtoh, f"found device->host copies: {[e.name for e in dtoh]}"
