"""The single-launch substep chain must be bit-identical to the layered one."""

import pytest
import torch

from swarp import Environment, NavigationScenario
from swarp.core.config import WorldConfig
from swarp.core.fused_substeps import block_layout
from swarp.scenarios import SCENARIOS, make_scenario

pytestmark = pytest.mark.gpu


def _rollout(env, fused, n_steps=12, seed=1):
    # True, not "auto": the parity has to hold wherever the chain is eligible, including
    # the shapes "auto" would route to the layered pipeline.
    env.world.stepper.enable_fused_substeps = fused
    obs = env.reset(seed=0)
    gen = torch.Generator(device=env.device).manual_seed(seed)
    out = [obs.clone()]
    lo, hi = env.action_bounds
    with torch.no_grad():
        for _ in range(n_steps):
            u = torch.rand(env.n_envs, env.n_agents, env.act_dim, device=env.device, generator=gen)
            obs, rew, term, trunc, _ = env.step(lo + (hi - lo) * u)
            st = env.world.state
            out.append(torch.cat([obs.flatten(), rew.flatten(), (term | trunc).float()]))
            out.append(torch.cat([st.pos.flatten(), st.vel.flatten(), st.theta.flatten()]))
    return out


def _assert_same(env_fn):
    on = _rollout(env_fn(), fused=True)
    off = _rollout(env_fn(), fused=False)
    for a, b in zip(on, off, strict=True):
        assert torch.equal(a, b)


def _nav_env(n_agents, n_envs, substeps, use_graph, auto_reset, n_obstacles=0, **world):
    scen = NavigationScenario(
        n_agents=n_agents, n_obstacles=n_obstacles, world_size=max(0.5, n_agents**0.5 / 8)
    )
    return Environment(
        scen,
        n_envs=n_envs,
        device="cuda:0",
        dt=0.05,
        substeps=substeps,
        seed=0,
        use_graph=use_graph,
        auto_reset=auto_reset,
        max_steps=5,
        world_config=WorldConfig(**world) if world else None,
    )


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("substeps", [1, 4])
@pytest.mark.parametrize("auto_reset", [False, True])
@pytest.mark.parametrize("n_obstacles", [0, 2])
def test_navigation_bit_identical(use_graph, substeps, auto_reset, n_obstacles):
    _assert_same(lambda: _nav_env(4, 37, substeps, use_graph, auto_reset, n_obstacles))


# Block-layout edges: one agent, a block's tail left idle (5, 33), an env that fills a
# block exactly (64), and one bigger than the target block (130 -> one env per block).
@pytest.mark.parametrize("n_agents", [1, 5, 33, 64, 130])
def test_layouts_bit_identical(n_agents):
    _assert_same(lambda: _nav_env(n_agents, 19, 2, True, False))


def test_truncated_lists_bit_identical():
    """A crowded world overflows ``max_neighbors``; the fused force keeps the same cap."""
    _assert_same(lambda: _nav_env(24, 9, 2, True, False, max_neighbors=4))


def test_clamp_bounds_bit_identical():
    _assert_same(lambda: _nav_env(8, 9, 3, True, False, bounds_mode="clamp"))


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_scenario_bit_identical(name):
    """Scenarios the fused chain cannot run (movable bodies) must fall back cleanly."""

    def env_fn():
        return Environment(
            make_scenario(name),
            n_envs=13,
            device="cuda:0",
            dt=0.05,
            substeps=3,
            seed=0,
            auto_reset=True,
            max_steps=5,
        )

    _assert_same(env_fn)


def test_post_step_list_replaces_the_rebuild():
    """With a list consumer, the fused step emits it: one build per step, not two."""
    env = _nav_env(4, 16, 1, False, False)
    env.world.stepper.enable_fused_substeps = True
    env.reset(seed=0)
    grid = env.world.stepper.grid(env.n_envs)
    a = torch.zeros(env.n_envs, 4, 2, device="cuda:0")
    with torch.no_grad():
        for _ in range(3):
            env.step(a)
        c0 = grid.build_count
        for _ in range(10):
            env.step(a)
    assert (grid.build_count - c0) / 10 == 1
    assert grid.post_step_version == env.world.stepper.state_version  # the fused chain ran
    env.world.mark_pos_dirty()
    assert grid.post_step_version == -1


@pytest.mark.parametrize("n_agents", [1, 3, 4, 5, 33, 64, 128, 129, 1024])
def test_block_layout(n_agents):
    block_dim, envs_per_block = block_layout(n_agents)
    assert block_dim % 32 == 0 and block_dim <= 1024
    assert envs_per_block * n_agents <= block_dim


@pytest.mark.parametrize(
    ("n_envs", "n_agents", "substeps", "fused"),
    [(4, 4, 1, True), (4, 8, 1, False), (8192, 4, 1, False), (8192, 4, 2, True),
     (4, 16, 4, True), (4, 32, 4, False)],
)
def test_auto_routes_by_agents_per_substep(n_envs, n_agents, substeps, fused):
    env = _nav_env(n_agents, n_envs, substeps, True, False)
    stepper = env.world.stepper
    assert stepper.enable_fused_substeps == "auto"
    state = stepper.alloc_state(n_envs)
    assert stepper._can_fuse_substeps(state, stepper.cached_buffers(n_envs)) is fused
