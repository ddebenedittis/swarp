"""Navigation's spawn sampler: separation by construction, and cheaply.

``_sample_separated`` runs on every step under ``auto_reset`` (the reset is masked
device-side, so there is no host-side gate that could skip it for a batch with nothing
done). It used to be a fixed 16-iteration ``torch.cdist`` rejection loop, which was both
the dominant cost of an auto-reset step and only *probabilistically* separated. The
stratified jittered-cell sampler that replaced it guarantees the separation, which is what
these pin.
"""

import contextlib
from unittest import mock

import pytest
import torch
from conftest import DEVICES

from swarp import Environment, NavigationScenario


def _raise(*args, **kwargs):
    raise AssertionError("device->host transfer in the spawn sampler")


@contextlib.contextmanager
def forbid_host_transfers():
    """Local copy of ``tests/interop/test_no_host_copies.py``'s guard (rootdir-relative
    test dirs are not importable across each other)."""
    with (
        mock.patch.object(torch.Tensor, "cpu", _raise),
        mock.patch.object(torch.Tensor, "item", _raise),
        mock.patch.object(torch.Tensor, "numpy", _raise),
        mock.patch.object(torch.Tensor, "tolist", _raise),
        mock.patch.object(torch.Tensor, "__bool__", _raise),
    ):
        yield


def _sample(n_agents, agent_radius=0.05, world_size=1.0, n_envs=128, device="cpu", seed=0):
    scen = NavigationScenario(
        n_agents=n_agents, agent_radius=agent_radius, world_size=world_size
    )
    env = Environment(scen, n_envs=n_envs, device=device, seed=seed)
    try:
        pos = scen._sample_separated(n_envs, n_agents, scen.min_spawn_separation)
        return scen, pos.clone()
    finally:
        env.close()


def _min_pairwise(pos):
    n = pos.shape[1]
    d = torch.cdist(pos, pos) + torch.eye(n, device=pos.device, dtype=pos.dtype) * 1e9
    return d.min().item()


@pytest.mark.parametrize("n_agents", [2, 4, 8, 16, 32, 64, 100])
def test_separation_is_guaranteed(n_agents):
    """Every pair is at least ``min_spawn_separation`` apart -- not "usually", always.

    The rejection loop this replaced could return overlapping points whenever 16 tries
    ran out, and did so silently.
    """
    scen, pos = _sample(n_agents)
    assert _min_pairwise(pos) >= scen.min_spawn_separation - 1e-6


@pytest.mark.parametrize("n_agents", [2, 8, 64])
def test_spawns_stay_inside_the_spawn_square(n_agents):
    """The jitter box is inset by half a cell, so no point can escape the square."""
    scen, pos = _sample(n_agents)
    lim = scen.world_size - 2.0 * scen.agent_radius
    assert pos.abs().max().item() <= lim + 1e-6


def test_infeasible_separation_falls_back_to_uniform():
    """At the packing limit there is no grid that both fits the points and leaves jitter
    room. The sampler then draws uniformly instead of pinning every point to a cell
    centre -- it gives up the guarantee rather than the honesty of the distribution."""
    # 8 agents, radius 0.2 => spawn square side 1.2, min separation 0.6: a 2x2 grid is
    # all that fits, i.e. 4 cells for 8 points. Genuinely impossible.
    scen, pos = _sample(8, agent_radius=0.2)
    assert _min_pairwise(pos) < scen.min_spawn_separation  # fell back, as documented
    assert torch.isfinite(pos).all()


def test_spawns_are_not_degenerate():
    """Jitter is real: points must not collapse onto their cell centres, and the two
    draws in one reset (spawns and goals) must differ."""
    scen, spawn = _sample(8)
    goals = scen._sample_separated(128, 8, scen.min_spawn_separation)
    assert not torch.allclose(spawn, goals)
    # Distinct across envs, and spread within a cell rather than quantized to one value.
    assert spawn.std().item() > 0.1


@pytest.mark.parametrize("device", DEVICES)
def test_sampler_is_host_sync_free(device):
    """No ``.item()``/``.cpu()`` anywhere in the draw -- it runs inside a masked reset on
    the step loop, where a device->host round-trip would stall the whole batch."""
    scen = NavigationScenario(n_agents=8)
    env = Environment(scen, n_envs=64, device=device, seed=0)
    try:
        scen._sample_separated(64, 8, scen.min_spawn_separation)  # warmup
        with torch.no_grad(), forbid_host_transfers():
            scen._sample_separated(64, 8, scen.min_spawn_separation)
    finally:
        env.close()


def test_same_seed_reproduces_spawns():
    _, a = _sample(8, seed=7)
    _, b = _sample(8, seed=7)
    _, c = _sample(8, seed=8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
