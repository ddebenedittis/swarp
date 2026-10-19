"""PushTLidarScenario's own contract: roles, trainer buffers, reward terms, layout."""

import math

import pytest
import torch
import warp as wp
from conftest import DEVICES

from swarp import Environment, PushTLidarScenario, PushTScenario
from swarp._overloads import concrete
from swarp.scenarios.pusht_lidar_kernels import (
    OBJECTIVES,
    STATE_AGENT,
    STATE_HEAD,
    lidar_sensor_kernel,
)


def _env(device, *, n_envs=32, fused=True, max_steps=None, auto_reset=False, **kw):
    scen = PushTLidarScenario(**kw)
    return Environment(
        scen,
        n_envs=n_envs,
        device=device,
        dt=0.05,
        substeps=8,
        seed=0,
        auto_reset=auto_reset,
        max_steps=max_steps,
        fused=fused,
    )


def _actions(env, gen):
    return torch.empty(env.n_envs, env.n_agents, 2, device=env.device).uniform_(
        -1, 1, generator=gen
    )


def _assert_counts(roles, counts):
    for r, c in enumerate(counts):
        assert ((roles == r).sum(-1) == c).all(), f"role {r} count != {c}"


# ------------------------------------------------------------------------- roles


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("n_agents", "n_pushers"), [(5, 2), (4, 3), (3, 7)])
def test_roles_have_exact_counts_and_vary(device, n_agents, n_pushers):
    env = _env(device, n_agents=n_agents, n_pushers=n_pushers, auto_reset=True, max_steps=3)
    scen = env.scenario
    counts = scen.role_counts
    assert counts == (min(n_pushers, n_agents), n_agents - min(n_pushers, n_agents))
    assert env.role_counts == counts and scen.n_roles == 2
    env.reset(seed=0)
    first = scen.agent_roles().clone()
    assert first.dtype == torch.int32 and tuple(first.shape) == (env.n_envs, n_agents)
    _assert_counts(first, counts)
    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for _ in range(7):  # crosses two auto-resets
            env.step(_actions(env, gen))
            _assert_counts(scen.agent_roles(), counts)
    if 0 < counts[0] < n_agents:
        assert (first != first[0]).any(), "every env drew the same roles"
        assert not torch.equal(scen.agent_roles(), first), "auto-resets drew no new roles"
        env.reset()
        assert not torch.equal(scen.agent_roles(), first), "reset() drew the same roles"


@pytest.mark.parametrize("device", DEVICES)
def test_masked_reset_redraws_only_selected_roles(device):
    env = _env(device, n_envs=64)
    scen = env.scenario
    env.reset(seed=0)
    before = scen.agent_roles().clone()
    ptr = scen.agent_roles().data_ptr()
    mask = torch.zeros(64, dtype=torch.bool, device=device)
    mask[::2] = True
    env.reset_at(mask)
    after = scen.agent_roles()
    assert after.data_ptr() == ptr, "the roles tensor must keep its address"
    assert torch.equal(after[~mask], before[~mask])
    assert not torch.equal(after[mask], before[mask])
    _assert_counts(after, scen.role_counts)


# --------------------------------------------------------------- trainer buffers


def _truth_state(scen):
    """The critic state rebuilt from the scenario's own tensors, role-sorted."""
    s = scen.world.state
    th, gth = scen.tee_theta, scen.goal_theta
    head = torch.cat(
        [
            scen.tee_pos,
            scen.goal_pos - scen.tee_pos,
            torch.stack([th.cos(), th.sin(), (th - gth).cos(), (th - gth).sin()], -1),
            scen.tee_vel,
            scen.tee_ang_vel.unsqueeze(-1),
            scen.goal_pos,
            torch.stack([gth.cos(), gth.sin()], -1),
        ],
        dim=-1,
    )
    roles = scen.agent_roles()
    sign = roles.to(s.pos.dtype) * 2 - 1
    rows = []
    for e in range(roles.shape[0]):
        order = [a for a in range(roles.shape[1]) if roles[e, a] == 0]
        order += [a for a in range(roles.shape[1]) if roles[e, a] == 1]
        rows.append(
            torch.cat([torch.cat([s.pos[e, a], s.vel[e, a], sign[e, a : a + 1]]) for a in order])
        )
    return torch.cat([head, torch.stack(rows)], dim=-1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False])
def test_critic_state_is_role_sorted_truth_after_auto_reset(device, fused):
    env = _env(device, fused=fused, n_envs=16, auto_reset=True, max_steps=3)
    scen = env.scenario
    env.reset(seed=0)
    assert scen.state_dim == STATE_HEAD + STATE_AGENT * env.n_agents == env.state_dim
    torch.testing.assert_close(scen.critic_state(), _truth_state(scen))
    gen = torch.Generator(device=device).manual_seed(1)
    ptrs = set()
    with torch.no_grad():
        for t in range(4):
            env.step(_actions(env, gen))
            if fused:
                ptrs.add((scen.critic_state().data_ptr(), scen.aux_targets().data_ptr()))
            # Step 3 ends every episode: the state must already describe the new one.
            torch.testing.assert_close(scen.critic_state(), _truth_state(scen), msg=f"@{t}")
            rel = (scen.tee_pos.unsqueeze(1) - scen.world.state.pos) * scen.aux_scale
            torch.testing.assert_close(scen.aux_targets()[..., :2], rel)
            torch.testing.assert_close(
                scen.aux_targets()[..., 2],
                scen.tee_theta.cos().unsqueeze(1).expand(-1, env.n_agents),
            )
    if fused:
        assert len(ptrs) == 1, "critic_state()/aux_targets() moved between steps"


# ------------------------------------------------------------------------ rewards


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False])
def test_multiobj_sums_to_reward_and_observer_terms_are_penalties(device, fused):
    env = _env(
        device,
        fused=fused,
        n_envs=32,
        world_size=0.5,
        contact_penalty=0.1,
        auto_reset=True,
        max_steps=6,
    )
    scen = env.scenario
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(3)
    idx = {o: i for i, o in enumerate(OBJECTIVES)}
    with torch.no_grad():
        for _ in range(12):
            roles = scen.agent_roles().clone()  # the roles this transition ran with
            _, rew, _, _, info = env.step(_actions(env, gen))
            m = info["multiobj_reward"]
            assert m.shape == (env.n_envs, env.n_agents, len(OBJECTIVES))
            torch.testing.assert_close(m.sum(-1), rew, rtol=1e-5, atol=1e-6)
            obs_role, push_role = roles == 1, roles == 0
            for o in ("visibility", "coverage", "occlusion", "contact"):
                assert (m[..., idx[o]] <= 0).all(), o
                assert (m[..., idx[o]][push_role] == 0).all(), f"{o} reached a pusher"
            assert (m[..., idx["standoff"]][push_role] == 0).all()
            assert (m[..., idx["approach"]][obs_role] == 0).all()
            for k in ("team_coverage", "observer_visibility", "occlusion"):
                assert ((info[k] >= 0) & (info[k] <= 1)).all(), k
            assert info["tee_seen"].dtype == torch.bool


# ------------------------------------------------------------------- observation


def test_obs_dims_per_level():
    dims = [PushTLidarScenario(obs_level=lvl).obs_dim for lvl in (0, 1, 2)]
    assert dims == [125, 117, 207]
    lay = PushTLidarScenario(obs_level=2, frame_stack=3).obs_layout()
    assert lay["lidar"].stop - lay["lidar"].start == 3 * 3 * 32
    assert lay["comm"].stop == lay["_end"].stop
    assert "tee" not in lay and "tee" in PushTLidarScenario(obs_level=0).obs_layout()


@pytest.mark.parametrize("device", DEVICES)
def test_level0_tee_block_equals_pusht(device):
    """The oracle block is Push-T's observation of the T, on Push-T's own spawn stream."""
    kw = dict(n_agents=5, world_size=0.5)
    lid = Environment(
        PushTLidarScenario(obs_level=0, **kw), n_envs=16, device=device, dt=0.05, substeps=8, seed=0
    )
    ref = Environment(PushTScenario(**kw), n_envs=16, device=device, dt=0.05, substeps=8, seed=0)
    tee = lid.scenario.obs_layout()["tee"]
    o_l, o_r = lid.reset(seed=0), ref.reset(seed=0)
    torch.testing.assert_close(o_l[..., tee], o_r[..., 4:12])
    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for _ in range(5):
            a = _actions(lid, gen)
            o_l, o_r = lid.step(a)[0], ref.step(a)[0]
            torch.testing.assert_close(o_l[..., tee], o_r[..., 4:12])
            torch.testing.assert_close(o_l[..., :4], o_r[..., :4])  # pos, vel


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False])
def test_frame_stack_ring_is_oldest_to_newest_and_zeroed_at_reset(device, fused):
    nf = 3
    env = _env(device, fused=fused, n_envs=8, frame_stack=nf, auto_reset=True, max_steps=4)
    scen = env.scenario
    lay = scen.obs_layout()
    width = 3 * scen.n_rays

    def frames(obs):
        return obs[..., lay["lidar"]].unflatten(-1, (nf, width))

    def newest(obs):
        return obs[..., lay["lidar_newest"]]

    obs = env.reset(seed=0).clone()
    assert (frames(obs)[:, :, : nf - 1] == 0).all()
    history = [newest(obs)]
    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for t in range(1, 4):
            obs = env.step(_actions(env, gen))[0].clone()
            history.append(newest(obs))
            f = frames(obs)
            for i in range(nf):
                past = t - (nf - 1 - i)
                want = history[past] if past >= 0 else torch.zeros_like(history[0])
                torch.testing.assert_close(f[:, :, i], want, msg=f"frame {i} @ step {t}")
        # Step 4 ends every episode: the next episode starts from an empty history.
        obs = env.step(_actions(env, gen))[0]
        assert (frames(obs)[:, :, : nf - 1] == 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_pooled_hits_respect_comm_range(device):
    env = _env(device, n_envs=16, obs_level=2, comm_range=0.0, world_size=0.5)
    scen = env.scenario
    obs = env.reset(seed=0)
    comm = obs[..., scen.obs_layout()["comm"]].unflatten(-1, (env.n_agents, scen.n_comm_pts, 3))
    valid = comm[..., 2]
    eye = torch.eye(env.n_agents, dtype=torch.bool, device=device)
    assert (valid[:, ~eye] == 0).all(), "a teammate beyond comm_range was shared"
    seen = scen.info()["tee_seen"]
    # Own slots are valid exactly when the robot sees the T.
    own = valid[:, eye].view(env.n_envs, env.n_agents, scen.n_comm_pts)
    assert torch.equal(own.amax(-1) > 0, seen)
    assert seen.any()


def test_standoff_inside_the_body_is_rejected():
    s = PushTLidarScenario()
    floor = s.tee_radius + s.agent_radius
    with pytest.raises(ValueError, match="inside the body"):
        PushTLidarScenario(standoff_dist=0.9 * floor)
    # Without observers the standoff is never used.
    PushTLidarScenario(n_agents=2, n_pushers=2, standoff_dist=0.9 * floor)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False])
def test_lidar_blind_zone_hides_a_touching_face(device, fused):
    """A face 5 mm off the bumper is within ``lidar_min_range`` of the mount: the forward
    ray reads a miss with the sensor model and the face without it. The rear ray, outside
    ``lidar_fov``, misses the second robot parked behind."""
    seen = {}
    for sensor in (False, True):
        kw = {}
        if sensor:
            kw = {"lidar_mount_offset": 0.04, "lidar_min_range": 0.05, "lidar_fov": 5.0}
        env = _env(device, n_envs=1, fused=fused, n_agents=2, n_pushers=2, model="diff_drive",
                   agent_shape="box", agent_half_extents=(0.05, 0.03), **kw)
        env.reset(seed=0)
        scen = env.scenario
        with torch.no_grad():
            scen.tee_pos.zero_()
            scen.tee_theta.zero_()
            scen.tee_vel.zero_()
            scen.tee_ang_vel.zero_()
            scen._install_obstacles()
            top = scen.box_off[0][1] + scen.box_half[0][1]  # the crossbar's top face
            pos = torch.tensor([[[0.0, top + 0.055], [0.0, top + 0.25]]], device=device)
            theta = torch.full((1, 2), -torch.pi / 2, device=device)
            env.world.write_state(pos=pos, theta=theta, speed=0.0, ang_vel=0.0)
            obs, *_ = env.step(torch.zeros(1, 2, 2, device=device))
        newest = obs[0, 0, scen.obs_layout()["lidar_newest"]]
        r = scen.n_rays
        seen[sensor] = (newest[r].item(), newest[2 * r + r // 2].item())  # T ray 0, robot ray 16
    assert seen[False] == (1.0, 1.0)
    assert seen[True] == (0.0, 0.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("fov_deg", "kept"), [(240.0, 10), (270.0, 12), (360.0, 16)])
def test_field_of_view_keeps_the_rays_within_half_of_it(device, fov_deg, kept):
    """Body-frame ray ``r`` sits ``r 2 pi / 32`` off the heading, wrapped to (-pi, pi]; it
    is kept iff that offset is within half the field of view (inclusive), whatever the
    heading: 240 deg keeps rays -10..10, 270 deg -12..12 (the edge rays at +-135 deg
    included), 360 deg all."""
    nr, n_agents = 32, 3
    theta = torch.tensor([[0.0, 1.234, -2.9]], dtype=torch.float32)
    wp_th = wp.from_torch(theta.to(device))
    ranges = wp.from_torch(torch.full((1, n_agents, nr), 0.5, device=device))
    hit = wp.from_torch(torch.full((1, n_agents, nr), 3, dtype=torch.uint8, device=device))
    half = wp.from_torch(torch.full((1, n_agents), math.radians(fov_deg) / 2, device=device))
    sigma = wp.from_torch(torch.zeros(1, n_agents, device=device))
    rng = wp.zeros((1, n_agents), dtype=wp.uint32, device=device)
    wp.launch(
        concrete(lidar_sensor_kernel, wp.float32),
        dim=(1, n_agents),
        inputs=[wp_th, 1.5, 2.0 * math.pi / nr, 0.0, 1, 0.1, 0.02, half, sigma, rng,
                ranges, hit],
        device=device,
    )
    k = torch.arange(nr)
    offset = torch.where(k > nr // 2, k - nr, k)  # ray index off the heading, (-16, 16]
    expect = (offset.abs() <= kept).expand(n_agents, -1)
    assert torch.equal(wp.to_torch(hit).cpu()[0] == 3, expect)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False])
def test_pose_noise_is_observed_only(device, fused):
    """With jitter off, a robot observes its true pose plus its episode bias, a teammate
    at the teammate's own biased position, and its velocity turned by its heading bias;
    the critic state and the reward see the true state."""
    kw = {"n_agents": 2, "n_pushers": 2, "model": "diff_drive", "agent_shape": "box",
          "agent_half_extents": (0.05, 0.03)}
    noise = {"pose_noise": (1.0, 1.0), "pose_jitter_std": (0.0, 0.0), "vel_noise_std": 0.0,
             "pose_bias_std": (0.05, 0.1)}
    runs = {}
    for on in (False, True):
        env = _env(device, n_envs=16, fused=fused, **kw, **(noise if on else {}))
        env.reset(seed=0)
        gen = torch.Generator(device=device).manual_seed(1)
        with torch.no_grad():
            for _ in range(3):
                obs, rew, *_ = env.step(_actions(env, gen))
        scen = env.scenario
        runs[on] = (obs.clone(), rew.clone(), scen.critic_state().clone(), scen)
    (obs0, rew0, crit0, _), (obs1, rew1, crit1, scen) = runs[False], runs[True]
    assert torch.equal(rew0, rew1)
    assert torch.equal(crit0, crit1)
    lay = scen.obs_layout()
    s = scen.world.state
    bias, bth = scen._bias_pos, scen._bias_theta
    assert bias.abs().max() > 0.0
    tol = {"atol": 1e-5, "rtol": 0}
    torch.testing.assert_close(obs1[..., lay["pos"]], s.pos + bias, **tol)
    torch.testing.assert_close(obs1[..., lay["goal_rel"]],
                               scen.goal_pos.unsqueeze(1) - s.pos - bias, **tol)
    th = s.theta + bth
    torch.testing.assert_close(obs1[..., lay["heading"]],
                               torch.stack([th.cos(), th.sin()], -1), **tol)
    c, sn = bth.cos(), bth.sin()
    v = s.vel
    turned = torch.stack([c * v[..., 0] - sn * v[..., 1], sn * v[..., 0] + c * v[..., 1]], -1)
    torch.testing.assert_close(obs1[..., lay["vel"]], turned, **tol)
    mate = obs1[:, 0, lay["teammates"]][:, :2]
    torch.testing.assert_close(mate, (s.pos[:, 1] + bias[:, 1]) - (s.pos[:, 0] + bias[:, 0]),
                               **tol)
    torch.testing.assert_close(obs1[..., lay["lidar"]], obs0[..., lay["lidar"]])
