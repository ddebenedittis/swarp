"""Neighbor-list reuse (WorldConfig.neighbor_reuse) must be exact and dedupe builds."""

import pytest
import torch
from conftest import DEVICES

from wmas import Environment, NavigationScenario


def _make_env(device, reuse, n_agents=4, n_envs=16, substeps=1, n_obstacles=0, auto_reset=False):
    scen = NavigationScenario(n_agents=n_agents, n_obstacles=n_obstacles, world_size=1.0)
    env = Environment(
        scen,
        n_envs=n_envs,
        device=device,
        dt=0.05,
        substeps=substeps,
        seed=0,
        auto_reset=auto_reset,
        max_steps=4,
        fused=False,  # exercise the torch path so reuse is the only variable
    )
    env.world.stepper.neighbor_reuse = reuse
    return env


def _traj(env, n_steps, device, n_agents):
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(1)
    out = []
    with torch.no_grad():
        for _ in range(n_steps):
            a = torch.empty(env.n_envs, n_agents, 2, device=device).uniform_(-1, 1, generator=gen)
            obs, rew, done, _ = env.step(a)
            out.append((obs.clone(), rew.clone(), done.clone()))
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("substeps", [1, 2])
@pytest.mark.parametrize("auto_reset", [False, True])
@pytest.mark.parametrize("n_obstacles", [0, 2])
def test_reuse_bit_identical(device, substeps, auto_reset, n_obstacles):
    kw = dict(device=device, substeps=substeps, auto_reset=auto_reset, n_obstacles=n_obstacles)
    on = _traj(_make_env(reuse=True, **kw), 8, device, 4)
    off = _traj(_make_env(reuse=False, **kw), 8, device, 4)
    for (o1, r1, d1), (o0, r0, d0) in zip(on, off, strict=True):
        assert torch.equal(o1, o0)
        assert torch.equal(r1, r0)
        assert torch.equal(d1, d0)


@pytest.mark.parametrize("device", DEVICES)
def test_build_count_drops_by_one_per_step(device):
    for reuse, expected in [(True, 1), (False, 2)]:
        env = _make_env(device, reuse=reuse)
        env.reset(seed=0)
        grid = env.world.stepper.grid(env.n_envs)
        a = torch.zeros(env.n_envs, 4, 2, device=device)
        with torch.no_grad():
            for _ in range(3):  # warm
                env.step(a)
            c0 = grid.build_count
            for _ in range(10):
                env.step(a)
            builds = (grid.build_count - c0) / 10
        assert builds == expected, f"reuse={reuse}: {builds} builds/step (want {expected})"


@pytest.mark.parametrize("device", DEVICES)
def test_mark_pos_dirty_forces_rebuild(device):
    env = _make_env(device, reuse=True)
    env.reset(seed=0)
    stepper = env.world.stepper
    grid = stepper.grid(env.n_envs)
    a = torch.zeros(env.n_envs, 4, 2, device=device)
    with torch.no_grad():
        env.step(a)
    # After a step + post-step build, the grid matches the current generation.
    assert grid.built_version == stepper.state_version
    env.world.mark_pos_dirty()
    assert grid.built_version == -1
    # The next step cannot reuse -> it rebuilds substep 0 itself.
    c0 = grid.build_count
    with torch.no_grad():
        env.step(a)
    assert grid.build_count - c0 == 2  # substep-0 build + post-step build


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_never_reuses(device):
    from wmas.interop.autograd import warp_step

    env = _make_env(device, reuse=True)
    env.reset(seed=0)
    stepper = env.world.stepper
    grid = stepper.grid(env.n_envs)
    st = env.world.state
    s = st._replace(pos=st.pos.detach().clone().requires_grad_(True))
    a = torch.zeros(env.n_envs, 4, 2, device=device, requires_grad=True)
    # Even though the grid still matches the generation, a taped step must
    # rebuild substep-0 neighbors into fresh buffers (grid lists would be
    # overwritten before backward), so build_count advances.
    grid.built_version = stepper.state_version
    c0 = grid.build_count
    with torch.enable_grad():
        warp_step(stepper, s, a)
    assert grid.build_count - c0 == 1
