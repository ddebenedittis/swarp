"""Fused Warp obs/reward/sheep kernels must match the torch reference path (bit-close).

Shepherding's own additions on top of the navigation-shaped parity suite: the flock is a
second dynamical system this scenario integrates itself, so the suite pins that it moves
(``test_sheep_flee_from_dogs`` is the one test that proves the task's premise), that it
stays inside the arena over a long rollout (nothing in the engine bounds a scenario-owned
obstacle), and that the pen bonus / terminal branch is actually *entered* during the parity
rollout rather than compared as zeros against zeros.
"""

import pytest
import torch
from conftest import (
    DEVICES,
    FusedSpec,
    assert_fused_determinism,
    assert_grad_falls_back_to_torch,
    assert_reset_parity,
    fused_env,
    fused_rollout,
)

from swarp.core.environment import Environment
from swarp.scenarios.shepherding import ShepherdingScenario

SPEC = FusedSpec(
    scenario=ShepherdingScenario,
    fields=("obs", "rew", "done", "info:sheep_dist_to_pen", "info:pen_fraction"),
    grad_steps=4,
    grad_index=2,
    # Deliberately ``obs``, matching transport rather than give-way. The reward's
    # action-dependence is flee/contact-gated: an env where no dog is within
    # ``flee_radius`` of any sheep contributes exactly zero gradient, so a reward-backprop
    # test would be asserting on whichever envs happened to make contact. ``obs`` carries
    # the dog-relative sheep positions and is unconditionally differentiable.
    grad_backprop="obs",
    # Load-bearing: this seed is what makes ``test_feature_guards_actually_fire`` see the
    # flock move, the penned fraction change, and at least one sheep inside the pen.
    # Changing it can make that guard pass vacuously, which no failure reports.
    action_seed=2,
    # A velocity-mode dog settles at a contact depth of ``max_speed / (k * sub_dt)``, so
    # the sheep are only solid at a small ``sub_dt``. 8 is the documented floor at dt=0.05,
    # and the parity rollouts run at the floor deliberately: the stiff regime is where the
    # two paths' contact geometry is most likely to disagree.
    substeps=8,
)

#: A small arena so three dogs and a handful of sheep actually meet inside the harness's
#: 5-step episodes; ``pen_radius`` scaled to match so the pen is still a target rather
#: than most of the world.
N_AGENTS = 3
N_SHEEP = 4
WORLD = 0.6
PEN_R = 0.2


def _scen(*, n_agents=N_AGENTS, n_sheep=N_SHEEP, difficulty=1.0, **kw):
    scen = ShepherdingScenario(
        n_agents=n_agents,
        n_sheep=n_sheep,
        world_size=WORLD,
        pen_radius=PEN_R,
        **kw,
    )
    scen.difficulty = difficulty
    return scen


def _env(device, fused, *, dtype=torch.float32, **kw):
    return fused_env(_scen(**kw), device, fused, spec=SPEC, dtype=dtype)


def _run(env, n_steps, device, n_agents=N_AGENTS, dtype=torch.float32):
    return fused_rollout(env, n_steps, device, n_agents, SPEC, dtype=dtype)


# ------------------------------------------------------------------- parity


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_sheep", [1, N_SHEEP])
def test_fused_matches_torch(device, n_sheep):
    """``n_sheep=1`` is not decoration: it is the only case that exercises the
    ``n_sheep > 1`` guard around the cohesion term on both sides."""
    kw = dict(n_sheep=n_sheep, difficulty=0.4)
    fused = _run(_env(device, fused=True, **kw), 20, device)
    torchp = _run(_env(device, fused=False, **kw), 20, device)
    for t, (f, r) in enumerate(zip(fused, torchp, strict=True)):
        of, rf, df, distf, fracf = f
        ot, rt, dt_, distt, fract = r
        torch.testing.assert_close(of, ot, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"obs@{t}")
        torch.testing.assert_close(rf, rt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"reward@{t}")
        # ``done`` is the AND of K threshold tests, and ``pen_fraction`` their count: both
        # are discrete, so a near-miss there is a whole ``pen_reward`` of error.
        assert torch.equal(df, dt_), f"done@{t}"
        torch.testing.assert_close(
            fracf, fract, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"pen_fraction@{t}"
        )
        torch.testing.assert_close(distf, distt, rtol=SPEC.rtol, atol=SPEC.atol, msg=f"dist@{t}")


@pytest.mark.parametrize("device", DEVICES)
def test_feature_guards_actually_fire(device):
    """The flock must genuinely move and the pen branch must genuinely be entered.

    A parity assertion over a rollout where the sheep never move and no sheep is ever
    penned compares zeros against zeros: the ``pen_reward``/``done`` branch of the reward
    kernel would never execute and parity would pass vacuously. All three checks run on the
    *torch* rollout, i.e. on the oracle, so no fused bug can make this guard pass.
    """
    env = _env(device, fused=False, difficulty=0.4)
    start = None
    fracs = []
    gen = torch.Generator(device=device).manual_seed(SPEC.action_seed)
    env.reset(seed=0)
    scen = env.scenario
    start = scen.sheep_pos.clone()
    moved = 0.0
    with torch.no_grad():
        for _ in range(20):
            a = torch.empty(env.n_envs, N_AGENTS, 2, device=device).uniform_(-1, 1, generator=gen)
            _, _, _, _, info = env.step(a)
            fracs.append(info["pen_fraction"].clone())
            moved = max(moved, (scen.sheep_pos - start).abs().max().item())
    frac = torch.stack(fracs)
    assert moved > 1e-3, "the sheep never moved"
    assert frac.amax().item() > 0.0, "no sheep was ever inside the pen"
    assert frac.unique().numel() > 1, "the penned fraction never changed"


@pytest.mark.parametrize("device", DEVICES)
def test_fused_seeded_determinism(device):
    a = _run(_env(device, fused=True), 10, device)
    b = _run(_env(device, fused=True), 10, device)
    assert_fused_determinism(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_falls_back_to_torch(device):
    env = _env(device, fused=True)
    assert_grad_falls_back_to_torch(env, device, N_AGENTS, SPEC)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_parity(device):
    """A standalone ``reset()``/``reset_at()`` must leave the same outputs as torch."""
    assert_reset_parity(lambda fused: _env(device, fused), device, N_AGENTS, SPEC)


# -------------------------------------------------------------- the sheep model


def _place(env, dog, sheep, *, evade):
    """Pin one dog and one sheep at given positions, in place (handles stay valid)."""
    scen = env.scenario
    w = env.world
    w.state.pos.copy_(torch.tensor([[dog]], device=w.device, dtype=w.dtype))
    w.state.vel.zero_()
    w.mark_pos_dirty()
    scen.sheep_pos.copy_(torch.tensor([[sheep]], device=w.device, dtype=w.dtype))
    scen.sheep_vel.zero_()
    scen.drift.zero_()  # isolate the flee force from the per-episode wander
    scen.evade.fill_(evade)
    scen._install_obstacles()


@pytest.mark.parametrize("device", DEVICES)
def test_sheep_flee_from_dogs(device):
    """A dog driven straight at a sheep must push it *further* than contact alone would.

    This is the test that proves the task's premise: without a flee force shepherding
    degenerates into transport with a rolling package, and the dogs would have no reason to
    approach from the far side. The assertion is differential — same actions, same
    geometry, ``evade`` 1 vs 0 — because the dog is faster than the sheep by design
    (``sheep_max_speed = 0.6 * max_speed``), so the raw separation *shrinks* in both cases
    and only the comparison isolates the force under test.
    """
    seps = {}
    disp = {}
    for evade in (0.0, 1.0):
        env = Environment(
            _scen(n_agents=1, n_sheep=1),
            n_envs=1,
            device=device,
            dt=0.05,
            substeps=8,
            seed=0,
            auto_reset=False,
            dtype=torch.float32,
            fused=False,
        )
        env.reset(seed=0)
        # 0.25 apart: inside ``flee_radius`` (0.3) and well outside the contact reach
        # (agent_radius + sheep_radius + margin = 0.1), so the first steps are pure flee.
        _place(env, (-0.25, 0.0), (0.0, 0.0), evade=evade)
        with torch.no_grad():
            for _ in range(6):
                env.step(torch.tensor([[[1.0, 0.0]]], device=device))
        q = env.scenario.sheep_pos[0, 0]
        p = env.world.state.pos[0, 0]
        seps[evade] = (q - p).norm().item()
        disp[evade] = q[0].item()

    assert disp[1.0] > 1e-3, "the fleeing sheep did not move away from the dog"
    assert seps[1.0] > seps[0.0] + 1e-4, (
        f"evade made no difference to the separation: {seps[1.0]:.5f} vs {seps[0.0]:.5f}"
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [False, True])
def test_sheep_stay_in_bounds(device, fused):
    """Over a long rollout the flock must stay in the arena and the obs must stay finite.

    Nothing in the engine bounds a scenario-owned obstacle, so the arena clamp is written
    into the integrate kernel and its torch twin. Forgetting it does not fail fast: the
    sheep drift away and the observation goes non-finite a few hundred steps in, which is
    exactly the horizon this test covers.
    """
    env = Environment(
        _scen(),
        n_envs=8,
        device=device,
        dt=0.05,
        substeps=8,
        seed=0,
        auto_reset=False,
        dtype=torch.float32,
        fused=fused,
    )
    env.reset(seed=0)
    scen = env.scenario
    bound = scen.world_size - scen.sheep_radius
    gen = torch.Generator(device=device).manual_seed(7)
    with torch.no_grad():
        for _ in range(300):
            a = torch.empty(8, N_AGENTS, 2, device=device).uniform_(-1, 1, generator=gen)
            obs, rew, *_ = env.step(a)
            assert torch.isfinite(obs).all()
            assert torch.isfinite(rew).all()
    assert scen.sheep_pos.abs().amax().item() <= bound + 1e-5
    assert scen.sheep_vel.norm(dim=-1).amax().item() <= scen.sheep_max_speed + 1e-5


# ------------------------------------------------------------- multiobj reward


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [False, True])
def test_multiobj_reward_sums_to_the_scalar_reward(device, fused):
    env = _env(device, fused=fused, difficulty=0.4)
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(SPEC.action_seed)
    with torch.no_grad():
        for _ in range(12):
            act = torch.empty(env.n_envs, N_AGENTS, 2, device=device).uniform_(
                -1, 1, generator=gen
            )
            _, rew, _, _, info = env.step(act)
            mo = info["multiobj_reward"]
            assert mo.shape == (env.n_envs, N_AGENTS, 5)
            torch.testing.assert_close(mo.sum(-1), rew, rtol=SPEC.rtol, atol=SPEC.atol)


# ------------------------------------------------------------------ contracts


def test_pen_walls_is_not_implemented():
    with pytest.raises(NotImplementedError, match="pen_walls"):
        ShepherdingScenario(pen_walls=True)


def test_at_least_one_sheep_is_required():
    with pytest.raises(ValueError, match="at least one sheep"):
        ShepherdingScenario(n_sheep=0)


def test_constructs_on_cpu_with_three_agents():
    # What the registry-wide scenario test will do once this is registered.
    scen = ShepherdingScenario(n_agents=3)
    assert scen.obs_dim == 10 + 4 * scen.n_sheep + 2 * (3 - 1)


@pytest.mark.parametrize("device", DEVICES)
def test_difficulty_zero_spawns_the_flock_inside_the_pen(device):
    """The curriculum's whole point: at ``difficulty=0`` the terminal is reached on step 1.

    That is what gives the value function a non-zero ``pen_reward`` to bootstrap from
    before the dogs can herd anything — the mechanism Push-T's curriculum rests on. Also
    pins the two per-episode buffers the flee force reads, since ``difficulty`` reaches the
    reset kernel through a device tensor rather than a baked scalar.
    """
    env = _env(device, fused=True, difficulty=0.0)
    env.reset(seed=0)
    scen = env.scenario
    d = (scen.sheep_pos - scen.pen_pos.unsqueeze(1)).norm(dim=-1)
    assert (d < scen.pen_radius).all(), "difficulty=0 spawned a sheep outside the pen"
    assert torch.equal(scen.evade, torch.zeros_like(scen.evade))
    assert torch.equal(scen.drift, torch.zeros_like(scen.drift))
    assert env.scenario.done().all(), "difficulty=0 must be solved at t=0"

    scen.difficulty = 1.0
    env.reset(seed=1)
    assert (scen.evade > 0).all()
    assert scen.drift.norm(dim=-1).amax().item() > 0.0
    # Clamped, and readable back.
    scen.difficulty = 5.0
    assert scen.difficulty == 1.0
