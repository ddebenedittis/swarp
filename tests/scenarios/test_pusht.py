"""Push-T scenario: agents push a T-shaped body to a target pose."""

import math

import pytest
import torch
from conftest import DEVICES

from wmas import Environment
from wmas.scenarios.pusht import PushTScenario


def _env(device, n_envs=4, **kw):
    # substeps=8 is the scenario's documented operating point: contact_k 8000 with
    # dt=0.05 needs it (k*sub_dt^2/m < 2 and the velocity-mode holding force), and the
    # examples use the same. At substeps=1 this stiffness is simply unstable.
    return Environment(
        PushTScenario(n_agents=4, **kw),
        n_envs=n_envs,
        device=device,
        dt=0.05,
        substeps=8,
        seed=1,
    )


def _park(scenario, device, n_envs=1, goal=(0.7, 0.0), goal_theta=0.0):
    """Park the T at the origin, upright, with a goal to the +x. Returns the world-frame
    x of the crossbar's left face — agents must be within ``r + margin`` of it to touch."""
    z2 = torch.zeros(n_envs, 2, device=device, dtype=torch.float32)
    z1 = torch.zeros(n_envs, device=device, dtype=torch.float32)
    scenario.tee_pos = z2.clone()
    scenario.tee_vel = z2.clone()
    scenario.tee_theta = z1.clone()
    scenario.tee_ang_vel = z1.clone()
    scenario.goal_pos = torch.tensor([goal], device=device, dtype=torch.float32).expand(n_envs, 2)
    scenario.goal_theta = torch.full((n_envs,), goal_theta, device=device, dtype=torch.float32)
    return -scenario.box_half[0][0]  # crossbar half-length, centred on x=0


@pytest.mark.parametrize("device", DEVICES)
def test_pusht_api_and_finiteness(device):
    env = _env(device, n_envs=8)
    obs = env.reset()
    assert obs.shape == (8, 4, 18) and torch.isfinite(obs).all()
    gen = torch.Generator(device=device).manual_seed(0)
    for _ in range(10):
        a = torch.rand(8, 4, env.world.act_dim, generator=gen, device=device) * 2 - 1
        obs, rew, done, info = env.step(a)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
        assert rew.shape == (8, 4) and done.shape == (8,)
        assert info["tee_dist_to_goal"].shape == (8,)
        assert info["tee_angle_error"].shape == (8,)
        # the wrapped heading error always lands in [0, pi]
        assert (info["tee_angle_error"] >= 0).all()
        assert (info["tee_angle_error"] <= math.pi + 1e-5).all()


@pytest.mark.parametrize("device", DEVICES)
def test_pusht_determinism(device):
    def run():
        env = _env(device, n_envs=4)
        env.reset(seed=4)
        gen = torch.Generator(device=device).manual_seed(2)
        out = []
        for _ in range(8):
            a = torch.rand(4, 4, env.world.act_dim, generator=gen, device=device) * 2 - 1
            obs, rew, _, _ = env.step(a)
            out += [obs, rew, env.scenario.tee_pos.clone(), env.scenario.tee_theta.clone()]
        return out

    for x, y in zip(run(), run(), strict=True):
        assert torch.equal(x, y)


@pytest.mark.parametrize("device", DEVICES)
def test_agents_push_tee_toward_goal(device):
    """Agents behind the T, driving toward the goal, move it closer.

    They line up *on the centroid's own height* (local y=0, against the stem's left face)
    and stack behind one another. A normal force there passes through the body origin, so
    this is a clean translation test; pushing the crossbar instead is off-centre for the
    centroid and legitimately spins the T, which
    :func:`test_offset_push_rotates_tee` covers.
    """
    scenario = PushTScenario(n_agents=4, world_size=1.0)
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, substeps=8, seed=0)
    env.reset()
    _park(scenario, device)
    # agents on the -x side, in a line through the centroid, driving into the stem
    ax = torch.tensor(
        [[[-0.10, 0.0], [-0.16, 0.0], [-0.22, 0.0], [-0.28, 0.0]]],
        device=device,
        dtype=torch.float32,
    )
    scenario.world.state = scenario.world.state._replace(pos=ax)
    scenario._install_obstacles()
    scenario._prev_dist = None
    scenario._prev_ang = None
    scenario._refresh(integrate=False)
    d0 = scenario._cache["dist_to_goal"].item()

    actions = torch.tensor([[[1.0, 0.0]] * 4], device=device, dtype=torch.float32)
    for _ in range(40):
        env.step(actions)
    d1 = scenario._cache["dist_to_goal"].item()
    assert scenario.tee_pos[0, 0].item() > 0.05  # the T moved in +x
    assert d1 < d0 - 0.05  # and got closer to the goal


@pytest.mark.parametrize("device", DEVICES)
def test_offset_push_rotates_tee(device):
    """Frictionless normal contact still spins the T: a single agent pressing the
    crossbar off-centre applies a torque about the centroid."""
    scenario = PushTScenario(n_agents=1, world_size=1.0)
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, substeps=8, seed=0)
    env.reset()
    _park(scenario, device)
    # push +x against the far right end of the crossbar -> negative torque
    bar_hx, bar_hy = scenario.box_half[0]
    bar_y = scenario.box_off[0][1]
    ax = torch.tensor([[[-(bar_hx + 0.10), bar_y]]], device=device, dtype=torch.float32)
    scenario.world.state = scenario.world.state._replace(pos=ax)
    scenario._install_obstacles()
    scenario._prev_dist = None
    scenario._prev_ang = None
    scenario._refresh(integrate=False)

    actions = torch.tensor([[[1.0, 0.0]]], device=device, dtype=torch.float32)
    for _ in range(30):
        env.step(actions)
    # the contact sits above the centroid (bar_y > 0), so pushing +x rotates the T
    assert bar_y > 0.0 and bar_hy > 0.0
    assert abs(scenario.tee_theta[0].item()) > 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_pusht_shaping_reward_positive_when_closer(device):
    """Global reward is positive on a step that reduces position *or* angle error."""
    scenario = PushTScenario(n_agents=4)
    env = Environment(scenario, n_envs=1, device=device, dt=0.05, substeps=8, seed=0)
    env.reset()
    _park(scenario, device, goal=(0.8, 0.0), goal_theta=0.0)
    scenario.tee_theta = torch.tensor([0.5], device=device, dtype=torch.float32)
    scenario._prev_dist = None
    scenario._prev_ang = None
    scenario._refresh(integrate=False)

    # position only: advance the T toward the goal, leave the heading alone
    scenario._prev_dist = scenario._cache["dist_to_goal"].detach().clone()
    scenario._prev_ang = scenario._cache["angle_error"].detach().clone()
    scenario.tee_pos = torch.tensor([[0.1, 0.0]], device=device, dtype=torch.float32)
    scenario._refresh(integrate=False)
    assert scenario.global_reward()[0].item() > 0.0

    # orientation only: hold the position, shrink the heading error
    scenario._prev_dist = scenario._cache["dist_to_goal"].detach().clone()
    scenario._prev_ang = scenario._cache["angle_error"].detach().clone()
    scenario.tee_theta = torch.tensor([0.2], device=device, dtype=torch.float32)
    scenario._refresh(integrate=False)
    assert scenario.global_reward()[0].item() > 0.0


def test_pusht_differentiable_rollout():
    """BPTT: the T-pose loss backprops to the action sequence.

    The agents are seeded against the crossbar and *commanded into it* — gradient
    reaches the body only through an active contact, so a random spawn would
    (correctly) give zeros. The command matters: agents are velocity-controlled, so
    with a zero action the stiff contact simply expels them to zero overlap within the
    step and the body then sees no contact at all. Substeps match the scenario's
    operating point (the stiff spring-damper needs them to stay stable).
    """
    device = "cpu"
    scenario = PushTScenario(n_agents=4)
    env = Environment(scenario, n_envs=2, device=device, dt=0.05, substeps=8, seed=0)
    env.reset()
    _park(scenario, device, n_envs=2)
    bar_hx = scenario.box_half[0][0]
    bar_y = scenario.box_off[0][1]
    # reach is agent_radius + contact_margin past the crossbar's left face
    x = -(bar_hx + scenario.agent_radius)
    ax = torch.tensor(
        [[[x, bar_y], [x, bar_y + 0.04], [x, bar_y - 0.04], [x - 0.02, bar_y]]],
        dtype=torch.float32,
    ).expand(2, 4, 2).contiguous()
    scenario.world.state = scenario.world.state._replace(pos=ax)
    scenario._install_obstacles()
    scenario._prev_dist = None
    scenario._prev_ang = None
    scenario._refresh(integrate=False)

    # Push +x, straight into the crossbar's left face, so the contact stays live.
    base = torch.zeros(2, 4, env.world.act_dim)
    base[..., 0] = 1.0
    actions = base.clone().requires_grad_(True)
    loss = torch.zeros((), dtype=torch.float32)
    for _ in range(4):
        env.step(actions)
        loss = scenario._cache["dist_to_goal"].sum() + scenario._cache["angle_error"].sum()
    loss.backward()
    assert actions.grad is not None and torch.isfinite(actions.grad).all()
    assert actions.grad.abs().sum() > 0.0
