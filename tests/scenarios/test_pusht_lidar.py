"""PushTLidarScenario's own contract: roles, trainer buffers, reward terms, layout."""

import math

import pytest
import torch
import warp as wp
from conftest import DEVICES

from swarp import Environment, PushTLidarScenario, PushTScenario
from swarp._overloads import concrete
from swarp.dynamics.base import NUM_PARAMS, P_HALF_X, P_HALF_Y, P_RADIUS, P_SHAPE
from swarp.scenarios.pusht_lidar_artefacts import (
    ART_DIM,
    ART_DROPOUT,
    ART_SPURIOUS,
    lidar_artefact_kernel,
    lidar_relabel_kernel,
)
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
    wp.init()
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
        inputs=[wp_th, 1.5, 2.0 * math.pi / nr, 0.0, 1, 0.1, 0.02, half, 0, sigma, rng,
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False])
def test_pose_drift_misses_a_shove_then_decays(device, fused):
    """A robot moved 0.1 m while its wheels stand still reports its old position (its
    odometry missed the move), and the error then decays by ``exp(-dt / tau)`` a step."""
    tau = 0.2
    env = _env(device, n_envs=1, fused=fused, n_agents=2, n_pushers=2, model="diff_drive",
               agent_shape="box", agent_half_extents=(0.05, 0.03), pose_noise=(1.0, 1.0),
               pose_bias_std=(0.0, 0.0), pose_jitter_std=(0.0, 0.0), vel_noise_std=0.0,
               pose_drift_tau=tau)
    env.reset(seed=0)
    scen = env.scenario
    pos = torch.tensor([[[-0.3, -0.3], [0.3, 0.3]]], device=device)
    zero = torch.zeros(1, 2, 2, device=device)
    with torch.no_grad():
        env.world.write_state(pos=pos, theta=torch.zeros(1, 2, device=device), speed=0.0,
                              ang_vel=0.0)
        obs, *_ = env.step(zero)  # the placement itself is a first missed move
        lay = scen.obs_layout()
        e0 = obs[0, :, lay["pos"]] - env.world.state.pos[0]
        before = env.world.state.pos.clone()
        env.world.write_state(pos=before + torch.tensor([0.1, 0.0], device=device))
        errs = []
        for _ in range(3):
            obs, *_ = env.step(zero)
            errs.append(obs[0, :, lay["pos"]] - env.world.state.pos[0])
    a = math.exp(-scen.dt / tau)
    shove = torch.tensor([0.1, 0.0], device=device)
    for k, err in enumerate(errs):
        torch.testing.assert_close(err, e0 * a ** (k + 1) - shove * a**k, atol=1e-5, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_field_of_view_at_the_mount_hides_close_side_hits(device):
    """With ``fov_at_mount`` the 240 deg field of view is the sensor's, centred on the mount
    0.1 m ahead: a hit 0.2 m away on ray 10 (112.5 deg off the heading from the centre)
    is 134 deg off it from the mount and reads as a miss, while a hit 1.4 m away on the
    same ray (118 deg from the mount) stays."""
    wp.init()
    nr = 32
    th = wp.from_torch(torch.zeros(1, 2, device=device))
    r = torch.full((1, 2, nr), 0.2, device=device)
    r[0, 1] = 1.4
    ranges = wp.from_torch(r)
    hit = wp.from_torch(torch.full((1, 2, nr), 3, dtype=torch.uint8, device=device))
    half = wp.from_torch(torch.full((1, 2), math.radians(120.0), device=device))
    sigma = wp.from_torch(torch.zeros(1, 2, device=device))
    rng = wp.zeros((1, 2), dtype=wp.uint32, device=device)
    wp.launch(
        concrete(lidar_sensor_kernel, wp.float32),
        dim=(1, 2),
        inputs=[th, 1.5, 2.0 * math.pi / nr, 0.0, 1, 0.1, 0.02, half, 1, sigma, rng,
                ranges, hit],
        device=device,
    )
    seen = wp.to_torch(hit).cpu()[0] == 3
    assert not seen[0, 10] and not seen[0, 22]
    assert seen[1, 10] and seen[1, 22]
    assert seen[0, 8] and seen[0, 0]  # 90 deg: 117 deg from the mount


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False])
def test_out_of_bounds_fails_the_episode_until_respawn(device, fused):
    """In an open arena a T whose centroid passes ``world_size + oob_margin`` fails its
    episode: from that step on every robot earns ``-oob_penalty`` and the goal no longer
    counts, even with the T brought back onto it; a respawn clears it."""
    env = _env(device, n_envs=2, fused=fused, n_agents=2, n_pushers=2, model="diff_drive",
               agent_shape="box", agent_half_extents=(0.05, 0.03), open_arena=True,
               oob_margin=0.5, oob_penalty=0.3)
    env.reset(seed=0)
    scen = env.scenario
    zero = torch.zeros(2, 2, 2, device=device)

    def place(x):
        with torch.no_grad():
            scen.tee_pos[0] = torch.tensor([x, 0.0], device=device)
            scen.tee_pos[1] = scen.goal_pos[1]
            scen.tee_theta[1] = scen.goal_theta[1]
            scen.tee_vel.zero_()
            scen.tee_ang_vel.zero_()
            scen._install_obstacles()

    place(scen.world_size + 0.6)
    _, rew, term, *_ = env.step(zero)
    assert torch.equal(rew[0], torch.full((2,), -0.3, device=device))
    assert not term[0] and term[1]
    with torch.no_grad():  # back onto its goal: still failed
        scen.tee_pos[0] = scen.goal_pos[0]
        scen.tee_theta[0] = scen.goal_theta[0]
        scen._install_obstacles()
    _, rew, term, *_ = env.step(zero)
    assert torch.equal(rew[0], torch.full((2,), -0.3, device=device)) and not term[0]
    env.reset(seed=1)
    _, rew, term, *_ = env.step(zero)
    assert (rew[0] != -0.3).any()


# ------------------------------------------------------------------- lidar artefacts


def _artefacts(device, pos, theta, ranges, cls, idx, art, *, grazing=0.0, smin=0.1,
               box=((0.0, 0.0), 0.0, (0.5, 0.02))):
    """One launch of the artefact kernel over a synthetic scan (one ray per agent, along
    the heading) against a single box obstacle; returns the edited (ranges, classes)."""
    wp.init()
    n_e, n_a = theta.shape
    v2 = wp.vec2f
    rng = wp.array((torch.arange(n_e * n_a) * 7919 + 1).view(n_e, n_a).numpy().astype("uint32"),
                   dtype=wp.uint32, device=device)
    r, c, i = ranges.clone().to(device), cls.clone().to(device), idx.clone().to(device)
    wp.launch(
        concrete(lidar_artefact_kernel, wp.float32),
        dim=(n_e, n_a),
        inputs=[
            wp.from_torch(pos.to(device), dtype=v2), wp.from_torch(theta.to(device)), 2.0,
            0.0, 0.0, 1,
            wp.array([[box[0]]] * n_e, dtype=v2, device=device),
            wp.array([[box[1]]] * n_e, dtype=wp.float32, device=device),
            wp.array([box[2]], dtype=v2, device=device),
            grazing, smin, wp.from_torch(art.to(device)), rng, wp.from_torch(r),
            wp.from_torch(c), wp.from_torch(i),
        ],
        device=device,
    )
    return r.cpu(), c.cpu()


def _binomial_ok(k, n, p):
    return abs(k / n - p) <= 5.0 * math.sqrt(p * (1.0 - p) / n) + 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_dropout_and_spurious_returns_at_their_rates(device):
    """Every hit is lost with the robot's dropout probability; every ray, a miss
    included, returns a spurious hit (the T's class) with its spurious probability, at a
    range between ``spurious_min`` and where the ray ended. A range scale and bias then
    apply to every remaining hit."""
    n = 20000
    pos = torch.zeros(n, 1, 2)
    theta = torch.zeros(n, 1)
    idx = torch.zeros(n, 1, 1, dtype=torch.int32)
    art = torch.zeros(n, 1, ART_DIM)
    art[..., ART_DROPOUT] = 0.2
    hits = torch.full((n, 1, 1), 3, dtype=torch.uint8)
    r, c = _artefacts(device, pos, theta, torch.full((n, 1, 1), 1.0), hits, idx, art)
    lost = (c == 0).sum().item()
    assert _binomial_ok(lost, n, 0.2)
    assert (r[c == 0] == 2.0).all() and (r[c == 3] == 1.0).all()

    art.zero_()
    art[..., ART_SPURIOUS] = 0.1
    misses = torch.zeros(n, 1, 1, dtype=torch.uint8)
    r, c = _artefacts(device, pos, theta, torch.full((n, 1, 1), 2.0), misses, idx, art)
    spur = c == 3
    assert _binomial_ok(spur.sum().item(), n, 0.1)
    assert (r[spur] >= 0.1).all() and (r[spur] <= 2.0).all() and (r[~spur] == 2.0).all()

    art.zero_()
    art[..., 2], art[..., 3] = 0.1, 0.01  # scale, bias
    r, c = _artefacts(device, pos[:4], theta[:4], torch.full((4, 1, 1), 1.0), hits[:4],
                      idx[:4], art[:4])
    torch.testing.assert_close(r, torch.full((4, 1, 1), 1.11))


@pytest.mark.parametrize("device", DEVICES)
def test_grazing_hits_on_the_t_are_lost(device):
    """A ray square onto the bar's end face always returns; one meeting its long face at
    incidence cosine 0.18 is lost with probability ``1 - 0.18 / grazing_cos``."""
    n = 20000
    head = torch.tensor([-1.5, 0.0])
    graze = torch.tensor([-1.5, 0.3])
    d = torch.tensor([0.0, 0.02]) - graze
    cos_i = (d[1].abs() / d.norm()).item()
    pos = torch.stack([head, graze]).repeat(n // 2, 1).view(n, 1, 2)
    theta = torch.tensor([0.0, math.atan2(d[1], d[0])]).repeat(n // 2).view(n, 1)
    t = torch.tensor([1.0, d.norm().item()]).repeat(n // 2).view(n, 1, 1)
    cls = torch.full((n, 1, 1), 3, dtype=torch.uint8)
    idx = torch.zeros(n, 1, 1, dtype=torch.int32)
    _, c = _artefacts(device, pos, theta, t, cls, idx, torch.zeros(n, 1, ART_DIM),
                      grazing=0.5)
    c = c.view(n // 2, 2)
    assert (c[:, 0] == 3).all()
    assert _binomial_ok((c[:, 1] == 0).sum().item(), n // 2, 1.0 - cos_i / 0.5)


@pytest.mark.parametrize("device", DEVICES)
def test_labels_come_from_the_reported_poses(device):
    """The deployment's rule (swarp_gz_test ``label_rays``): a hit inside a teammate's
    *reported* box grown by the margin is an agent hit, anything else in bounds is the T,
    and a hit beyond the bound is a miss. A teammate hit 0.84 m ahead reads as an agent at
    the teammate's true pose and as the T once its reported pose is 0.1 m further off."""
    assert _label(device, 1.0, 0.84) == (1, pytest.approx(0.84), 0)
    assert _label(device, 1.1, 0.84) == (3, pytest.approx(0.84), 0)
    assert _label(device, 1.0, 0.84, bound=0.5) == (0, 3.0, 0)


@pytest.mark.parametrize("device", DEVICES)
def test_per_env_label_bound_is_the_arena_plus_margin_per_axis(device):
    """With a per-env arena the bound is ``arena_half + margin`` on each axis: a scenery
    hit 0.84 m ahead along x is a miss in a 1.2 m wide arena (0.6 + 0.2 < 0.84) and the T,
    a leaked one, in a 2 m wide one; turned along y the narrow arena's 2 m height keeps
    it."""
    narrow, wide = (0.6, 1.0), (1.0, 0.6)
    assert _label(device, 5.0, 0.84, bound=0.2, arena=narrow, cls_in=2) == (0, 3.0, 0)
    assert _label(device, 5.0, 0.84, bound=0.2, arena=wide, cls_in=2) == (
        3, pytest.approx(0.84), 1)
    assert _label(device, 5.0, 0.84, bound=0.2, arena=narrow, cls_in=2, theta=math.pi / 2) == (
        3, pytest.approx(0.84), 1)


def _label(device, mate_x, t, bound=2.0, arena=None, cls_in=1, theta=0.0):
    """One relabel launch: robot 0 at the origin heading ``theta`` with one ray ahead at
    range ``t`` and class ``cls_in``, a 0.32 x 0.22 m teammate at ``(mate_x, 0)``; with
    ``arena`` the bound is per env and per axis. Returns robot 0's class, range and
    scenery leak count."""
    wp.init()
    v2 = wp.vec2f
    opos = torch.tensor([[[0.0, 0.0], [mate_x, 0.0]]])
    params = torch.zeros(2, NUM_PARAMS)
    params[:, P_SHAPE], params[:, P_HALF_X], params[:, P_HALF_Y] = 1.0, 0.16, 0.11
    params[:, P_RADIUS] = 0.195
    r = torch.tensor([[[t], [3.0]]], device=device)
    c = torch.tensor([[[cls_in], [0]]], dtype=torch.uint8, device=device)
    leak = torch.zeros(1, 2, device=device)
    half = torch.tensor([arena or (0.0, 0.0)], device=device)
    wp.launch(
        concrete(lidar_relabel_kernel, wp.float32),
        dim=(1, 2),
        inputs=[wp.from_torch(opos.to(device), dtype=v2),
                wp.from_torch(torch.tensor([[theta, 0.0]], device=device)),
                wp.from_torch(params.to(device)), 3.0, 0.0, 0.0, 1, bound, 0.03,
                int(arena is not None), wp.from_torch(half, dtype=v2), 1,
                wp.from_torch(leak), 0, 0.0, wp.from_torch(r), wp.from_torch(c),
                wp.zeros((1, 1, 1), dtype=wp.uint8, device=device)],
        device=device,
    )
    return c[0, 0, 0].item(), r[0, 0, 0].item(), int(leak[0, 0].item())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False])
def test_critic_carries_the_randomised_parameters_role_sorted(device, fused):
    """``critic_dr`` appends every drawn parameter, normalised over its range, per robot
    in role-sorted order: here the speed gain and the range-noise deviation."""
    env = _env(device, n_envs=16, fused=fused, n_agents=3, n_pushers=1, model="diff_drive",
               agent_shape="box", agent_half_extents=(0.05, 0.03),
               cmd_gain_v=(0.8, 1.0), lidar_noise=(0.0, 0.02), critic_dr=True)
    scen = env.scenario
    assert (scen.dr_env_dim, scen.dr_agent_dim) == (0, 2)
    assert scen.state_dim == scen.state_base_dim + 2 * 3
    env.reset(seed=0)
    with torch.no_grad():
        env.step(torch.zeros(16, 3, 2, device=device))
    from swarp.dynamics.base import P_CMD_GAIN_V

    gain = wp.to_torch(env.world.stepper.params.floats_per_env)[..., P_CMD_GAIN_V]
    want = torch.stack([(gain - 0.9) / 0.1, (scen._noise_std - 0.01) / 0.01], dim=-1)
    perm = torch.argsort(scen.agent_roles(), dim=-1, stable=True)
    want = want.gather(1, perm.unsqueeze(-1).expand(-1, -1, 2)).flatten(1)
    got = scen.critic_state()[:, scen.state_base_dim:]
    torch.testing.assert_close(got, want)
    assert got.std() > 0.3


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False])
def test_span_scans_only_its_sector(device, fused):
    """A 240-deg span puts its first and last rays at +-120 deg and never sees the T
    straight behind, which the full circle does."""
    seen = {}
    for span in (2.0 * math.pi, 4.0 * math.pi / 3.0):
        env = _env(device, n_envs=1, fused=fused, n_agents=1, n_pushers=1, n_rays=43,
                   model="diff_drive", agent_shape="box", agent_half_extents=(0.05, 0.03),
                   lidar_span=span)
        env.reset(seed=0)
        scen = env.scenario
        with torch.no_grad():
            scen.tee_pos.zero_()
            scen.tee_vel.zero_()
            scen.tee_ang_vel.zero_()
            scen._install_obstacles()
            env.world.write_state(pos=torch.tensor([[[0.6, 0.0]]], device=device),
                                  theta=torch.zeros(1, 1, device=device), speed=0.0,
                                  ang_vel=0.0)
            obs, *_ = env.step(torch.zeros(1, 1, 2, device=device))
        lidar = obs[0, 0, scen.obs_layout()["lidar_newest"]]
        seen[span < 6.0] = bool((lidar[43:86] > 0.5).any())
        if span < 6.0:
            off = scen.lidar.ray_offsets()
            assert off[0].item() == pytest.approx(-2.0 * math.pi / 3.0)
            assert off[-1].item() == pytest.approx(2.0 * math.pi / 3.0)
    assert seen == {False: True, True: False}


# ------------------------------------------------------------ per-env arena, scenery

#: The Limo fleet and slim T at real scale, in an open arena with the out-of-bounds check.
_LIMO = dict(n_agents=2, n_pushers=2, model="diff_drive", agent_shape="box",
             agent_half_extents=(0.161, 0.110), max_speed=0.4, world_size=2.5,
             bar_len=0.5, bar_width=0.04, stem_len=0.5, stem_width=0.04, open_arena=True,
             oob_margin=0.5, lidar_max_range=1.5)
_LAYOUTS = (0.25, 0.25, 0.25, 0.25)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("arena", [(2.0, 2.0), (2.0, 5.0)])
def test_per_env_arena_spawns_inside_and_clear(device, arena):
    """Each env's width and height are drawn from the range independently; the T, its
    goal and both robots spawn inside that env's arena, the robots clear of the T and of
    each other even in a 2 x 2 m arena, and the arena stop sits at the env's own edges."""
    env = _env(device, n_envs=1024, arena_size=arena, **_LIMO)
    env.reset(seed=0)
    scen = env.scenario
    half = scen.arena_half
    assert (half >= 0.5 * arena[0] - 1e-6).all() and (half <= 0.5 * arena[1] + 1e-6).all()
    if arena[0] < arena[1]:
        assert (half[:, 0] - half[:, 1]).abs().max() > 0.5
    r_t, r_a = scen.tee_radius, scen.agent_radius
    pos = env.world.state.pos
    assert (scen.tee_pos.abs() <= half - r_t + 1e-5).all()
    assert (scen.goal_pos.abs() <= half - r_t + 1e-5).all()
    assert (pos.abs() <= (half - 2.0 * r_a).unsqueeze(1) + 1e-5).all()
    assert ((pos - scen.tee_pos.unsqueeze(1)).norm(dim=-1) >= r_t + r_a).all()
    assert ((pos[:, 0] - pos[:, 1]).norm(dim=-1) >= 2.0 * r_a - 1e-5).all()
    torch.testing.assert_close(scen._arena_edges, half + scen.arena_stop_margin)


@pytest.mark.parametrize("device", DEVICES)
def test_arena_and_scenery_redraw_only_where_reset(device):
    """A masked reset redraws the arena and the surroundings of the envs it selects only,
    and what a trainer's ``Ramp`` writes (``arena_size``, ``scenery_on``) reaches the next
    reset."""
    env = _env(device, n_envs=64, arena_size=(2.0, 5.0), scenery_probs=_LAYOUTS, **_LIMO)
    env.reset(seed=0)
    scen = env.scenario

    def snap():
        return [t.clone() for t in (scen.arena_half, scen.scenery_mode, *scen.scenery.tensors())]

    before = snap()
    mask = torch.zeros(64, dtype=torch.bool, device=device)
    mask[:8] = True
    env.reset_at(mask)
    after = snap()
    for b, a in zip(before, after, strict=True):
        assert torch.equal(b[8:], a[8:])
    assert not torch.equal(before[0][:8], after[0][:8])
    scen.arena_size = (3.0, 3.0)
    scen.scenery_on = 0.0
    env.reset(seed=1)
    torch.testing.assert_close(scen.arena_half, torch.full_like(scen.arena_half, 1.5))
    assert (scen.scenery_mode == 0).all() and (scen.scenery.shape < 0).all()


def _point_rect(p, fx, fy):
    """Distance from the points ``p [N, 2]`` to the centred rectangles ``|x| <= fx, |y| <=
    fy`` (zero inside)."""
    return torch.hypot((p[:, 0].abs() - fx).clamp(min=0.0), (p[:, 1].abs() - fy).clamp(min=0.0))


def _segment_rect(a, b, fx, fy):
    """Exact distance from the segments ``a -> b`` to the centred rectangles: zero when
    they cross (Liang-Barsky clip), else the nearest of the endpoints to the rectangle and
    of the corners to the segment."""
    d = b - a
    t0 = torch.zeros_like(fx)
    t1 = torch.ones_like(fx)
    hit = torch.ones_like(fx, dtype=torch.bool)
    for k, f in ((0, fx), (1, fy)):
        dk, ak = d[:, k], a[:, k]
        flat = dk.abs() < 1e-12
        safe = torch.where(flat, torch.ones_like(dk), dk)
        lo = torch.minimum((-f - ak) / safe, (f - ak) / safe)
        hi = torch.maximum((-f - ak) / safe, (f - ak) / safe)
        t0 = torch.where(flat, t0, torch.maximum(t0, lo))
        t1 = torch.where(flat, t1, torch.minimum(t1, hi))
        hit &= ~(flat & (ak.abs() > f))
    hit &= t0 <= t1
    best = torch.minimum(_point_rect(a, fx, fy), _point_rect(b, fx, fy))
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            c = torch.stack([sx * fx, sy * fy], dim=-1)
            u = (((c - a) * d).sum(-1) / (d * d).sum(-1).clamp(min=1e-12)).clamp(0.0, 1.0)
            best = torch.minimum(best, (a + u.unsqueeze(-1) * d - c).norm(dim=-1))
    return torch.where(hit, torch.zeros_like(best), best)


@pytest.mark.parametrize("device", DEVICES)
def test_scenery_stays_outside_the_free_region(device):
    """Over every layout, no wall (a capsule) and no clutter shape comes within the free
    region: the arena grown by the clearance, which defaults to the farther of a robot's
    and the T's reach past the arena. No layout leaves slots where it has none to fill;
    perimeter walls are parallel to the arena's sides and a room's are not."""
    from swarp.core.config import ObstacleShape
    from swarp.scenarios.pusht_lidar_scenery import N_WALLS

    env = _env(device, n_envs=4096, arena_size=(2.0, 5.0), scenery_probs=_LAYOUTS, **_LIMO)
    env.reset(seed=0)
    scen = env.scenario
    sc = scen.scenery
    clear = scen.scenery_clearance
    assert clear == pytest.approx(max(scen.robot_reach(), scen.tee_reach()))
    assert clear >= scen.oob_margin + scen.tee_radius
    fx = scen.arena_half[:, 0] + clear
    fy = scen.arena_half[:, 1] + clear
    mode = scen.scenery_mode
    assert set(mode.tolist()) == {0, 1, 2, 3}
    used = sc.shape >= 0
    assert not used[mode == 0].any()
    assert not used[mode != 3][:, N_WALLS:].any()
    assert used[mode == 3][:, N_WALLS:].any(-1).float().mean() > 0.95
    walls = used[:, :N_WALLS]
    tilt = torch.sin(2.0 * sc.angle[:, :N_WALLS]).abs()
    assert (tilt[walls & (mode == 1).unsqueeze(-1)] < 1e-5).all()
    assert tilt[walls & (mode == 2).unsqueeze(-1)].mean() > 0.3
    for s in range(sc.n_slots):
        u = used[:, s]
        if s < N_WALLS:
            ang = sc.angle[:, s]
            tip = torch.stack([ang.cos(), ang.sin()], -1) * sc.half[:, s, :1]
            dist = _segment_rect(sc.pos[:, s] - tip, sc.pos[:, s] + tip, fx, fy)
            need = sc.radius[:, s]
        else:
            dist = _point_rect(sc.pos[:, s], fx, fy)
            is_box = sc.shape[:, s] == int(ObstacleShape.BOX)
            need = torch.where(is_box, sc.half[:, s].norm(dim=-1), sc.radius[:, s])
        assert (dist[u] >= need[u] - 1e-5).all(), f"slot {s} enters the free region"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("solid", [True, False])
def test_solid_scenery_stops_robots_and_is_penalised(device, solid):
    """Robots driving straight out of a 2 x 2 m arena with no arena stop, into closed
    perimeter walls at its edge: solid walls hold every robot centre inside them and flag
    the contact (paid as ``collision_penalty``), lidar-only walls let them through. Solid
    scenery is also stepper obstacles after the T's boxes, and its clutter keeps clear of
    the arena grown by the clearance (0 by default)."""
    kw = {"scenery_collide": True, "collision_penalty": 0.05} if solid else {
        "scenery_clearance": 0.0}
    env = _env(device, n_envs=64, arena_size=(2.0, 2.0), scenery_probs=(0.0, 1.0, 0.0, 0.0),
               wall_missing=0.0, scenery_gap=(0.0, 0.1), wall_radius=(0.05, 0.1),
               **{**_LIMO, "arena_stop_margin": 5.0}, **kw)
    env.reset(seed=0)
    scen = env.scenario
    sc = scen.scenery
    flagged = torch.zeros(64, 2, dtype=torch.bool, device=device)
    paid = False
    with torch.no_grad():
        for _ in range(60):
            a = torch.zeros(64, 2, 2, device=device)
            a[..., 0] = 1.0
            env.step(a)
            if solid:
                info = scen.info()
                touch = info["scenery_collision"] | info["robot_collision"]
                flagged |= info["scenery_collision"]
                col = info["multiobj_reward"][..., OBJECTIVES.index("collision")]
                torch.testing.assert_close(col, torch.where(touch, -0.05, 0.0) * (col != 0))
                paid |= bool((col == -0.05).any())
    p = env.world.state.pos
    inner_x = (sc.pos[:, 0, 0] - sc.radius[:, 0]).unsqueeze(1)
    inner_y = (sc.pos[:, 2, 1] - sc.radius[:, 2]).unsqueeze(1)
    inside = (p[..., 0] < inner_x + 1e-3) & (p[..., 1] < inner_y + 1e-3)
    if solid:
        assert inside.all(), "a robot crossed a solid wall"
        assert flagged.float().mean() > 0.5 and paid
        assert env.world.stepper.n_obstacles == scen.n_boxes + sc.n_slots
    else:
        assert not inside.all(), "no robot crossed a lidar-only wall"


@pytest.mark.parametrize("device", DEVICES)
def test_tee_size_bank_draws_per_env_and_parks_the_rest(device):
    """``tee_scale`` (0.8, 1.2) over three sizes: every size is drawn, a masked reset
    redraws only the envs it selects, the stepper holds each size's boxes scaled (inertia
    with the square), and after a step the env's own size is the T while the others stay
    parked at rest."""
    env = _env(device, n_envs=256, tee_scale=(0.8, 1.2), n_tee_sizes=3, **_LIMO)
    env.reset(seed=0)
    scen = env.scenario
    nb = scen.n_boxes
    assert scen.tee_scales == pytest.approx((0.8, 1.0, 1.2))
    assert scen.tee_radius == pytest.approx(1.2 * scen.shape.radius)
    assert set(scen.tee_size.tolist()) == {0, 1, 2}
    sp = env.world.stepper
    assert sp.n_obstacles == 3 * nb
    half = wp.to_torch(sp.obs_half).view(3, nb, 2)
    base = torch.tensor(scen.box_half, device=device, dtype=half.dtype)
    for k, f in enumerate((0.8, 1.0, 1.2)):
        torch.testing.assert_close(half[k], f * base)
    inertia = wp.to_torch(sp.obs_inertia).view(3, nb)[:, 0]
    torch.testing.assert_close(inertia, scen.tee_inertia * torch.tensor(
        [0.64, 1.0, 1.44], device=device, dtype=inertia.dtype))
    before = scen.tee_size.clone()
    mask = torch.zeros(256, dtype=torch.bool, device=device)
    mask[:64] = True
    env.reset_at(mask)
    assert torch.equal(scen.tee_size[64:], before[64:])
    assert not torch.equal(scen.tee_size[:64], before[:64])
    gen = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for _ in range(5):
            env.step(_actions(env, gen))
    bpos = wp.to_torch(sp.body_pos)
    e = torch.arange(256, device=device)
    torch.testing.assert_close(bpos[e, scen._tee_root.long()], scen.tee_pos)
    other = (scen.tee_size.long() + 1) % 3
    assert (bpos[e, other * nb].abs() > 900.0).all()
    assert (wp.to_torch(sp.body_vel)[e, other * nb] == 0.0).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False])
def test_arena_flag_marks_the_t_hits_past_the_arena(device, fused):
    """``lidar_arena_flag`` adds a fourth mask per frame: exactly the T-labelled rays whose
    hit, placed from the reported pose, lies past the env's arena on either axis. With
    surroundings at the arena's edge it fires, and never on a ray the T mask leaves out."""
    kw = dict(arena_size=(2.0, 3.0), scenery_probs=(0.0, 0.5, 0.0, 0.5), scenery_collide=True,
              lidar_label_from_pose=True, lidar_arena_flag=True, **_LIMO)
    env = _env(device, n_envs=128, fused=fused, **kw)
    scen = env.scenario
    nr = scen.n_rays
    plain = PushTLidarScenario(**{**kw, "lidar_arena_flag": False})
    assert scen.obs_dim == plain.obs_dim + nr
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(0)
    fired = False
    with torch.no_grad():
        for _ in range(10):
            obs, *_ = env.step(_actions(env, gen))
            frame = obs[..., scen.obs_layout()["lidar_newest"]]
            rng, t_mask, out = frame[..., :nr], frame[..., nr : 2 * nr], frame[..., 3 * nr :]
            assert not ((out > 0.5) & (t_mask < 0.5)).any()
            ang = scen._ray_angles()
            hit = env.world.state.pos.unsqueeze(2) + (rng * scen.lidar_max_range).unsqueeze(
                -1) * torch.stack([ang.cos(), ang.sin()], -1)
            past_by = (hit.abs() - scen.arena_half.view(-1, 1, 1, 2)).amax(-1)
            past = (past_by > 0.0) & (t_mask > 0.5)
            # Rebuilt from the observed range / max_range: exact but at the edge itself.
            edge = past_by.abs() < 1e-4
            assert torch.equal(past[~edge], (out > 0.5)[~edge])
            fired |= bool(past.any())
    assert fired, "no T-labelled hit past the arena"


@pytest.mark.parametrize("device", DEVICES)
def test_solid_clutter_stays_outside_the_arena(device):
    from swarp.core.config import ObstacleShape
    from swarp.scenarios.pusht_lidar_scenery import N_WALLS

    env = _env(device, n_envs=2048, arena_size=(2.0, 5.0), scenery_probs=(0.0, 0.0, 0.0, 1.0),
               scenery_collide=True, n_clutter=12, clutter_count=(2, 6), **_LIMO)
    env.reset(seed=0)
    scen = env.scenario
    sc = scen.scenery
    assert scen.scenery_clearance == 0.0
    used = sc.shape[:, N_WALLS:] >= 0
    n = used.sum(-1)
    assert n.max() <= 6 and (n >= 1).float().mean() > 0.95
    assert used.float().mean(0).min() > 0.1, "a clutter slot is never drawn"
    assert not (sc.shape[:, N_WALLS:] == int(ObstacleShape.BOX)).any()
    for s in range(N_WALLS, sc.n_slots):
        u = sc.shape[:, s] >= 0
        need = sc.half[:, s, 0] + sc.radius[:, s]
        dist = _point_rect(sc.pos[:, s], scen.arena_half[:, 0], scen.arena_half[:, 1])
        assert (dist[u] >= need[u] - 1e-5).all(), f"slot {s} enters the arena"
    eng = wp.to_torch(env.world.stepper.obs_pos)[:, scen.n_boxes:]
    torch.testing.assert_close(eng[sc.shape >= 0], sc.pos[sc.shape >= 0])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False])
def test_per_env_out_of_bounds_is_per_axis(device, fused):
    """In a 2 x 5 m arena with ``oob_margin`` 0.5 the T fails 1.6 m out along x (past
    1.5) but not 1.6 m out along y (inside 3.0)."""
    env = _env(device, n_envs=2, fused=fused, arena_size=(2.0, 2.0, 5.0, 5.0),
               oob_penalty=0.3, **_LIMO)
    env.reset(seed=0)
    scen = env.scenario
    with torch.no_grad():
        scen.tee_pos[0] = torch.tensor([1.6, 0.0], device=device)
        scen.tee_pos[1] = torch.tensor([0.0, 1.6], device=device)
        scen.tee_vel.zero_()
        scen.tee_ang_vel.zero_()
        scen._install_obstacles()
        _, rew, *_ = env.step(torch.zeros(2, 2, 2, device=device))
    assert scen.info()["out_of_bounds"].tolist() == [True, False]
    assert torch.equal(rew[0], torch.full((2,), -0.3, device=device))
    assert (rew[1] != -0.3).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fused", [True, False])
def test_arena_is_observed_and_in_the_critic_state(device, fused):
    """The self block ends with the env's arena half-extents, and ``critic_dr`` adds them
    (normalised over the range) and the layout's one-hot (as -1 / +1) as env columns."""
    env = _env(device, n_envs=64, fused=fused, arena_size=(2.0, 5.0), scenery_probs=_LAYOUTS,
               critic_dr=True, **_LIMO)
    scen = env.scenario
    assert scen.obs_dim == PushTLidarScenario(**_LIMO).obs_dim + 2
    lay = scen.obs_layout()
    assert lay["arena"] == slice(scen.self_dim - 2, scen.self_dim)
    assert lay["teammates"].start == scen.self_dim
    env.reset(seed=0)
    with torch.no_grad():
        obs, *_ = env.step(torch.zeros(64, 2, 2, device=device))
    half = scen.arena_half
    torch.testing.assert_close(obs[:, :, lay["arena"]], half.unsqueeze(1).expand(-1, 2, -1))
    assert (scen.dr_env_dim, scen.dr_agent_dim) == (6, 0)
    hot = torch.nn.functional.one_hot(scen.scenery_mode.long(), 4).to(half.dtype) * 2.0 - 1.0
    want = torch.cat([(half - 1.75) / 0.75, hot], dim=-1)
    torch.testing.assert_close(scen.critic_state()[:, scen.state_base_dim:], want)


def test_arena_and_scenery_validation():
    with pytest.raises(ValueError, match="open_arena"):
        PushTLidarScenario(**{**_LIMO, "open_arena": False}, arena_size=(2.0, 5.0))
    with pytest.raises(ValueError, match="world_size"):
        PushTLidarScenario(**{**_LIMO, "world_size": 2.0}, arena_size=(2.0, 5.0))
    with pytest.raises(ValueError, match="no room"):
        PushTLidarScenario(**_LIMO, arena_size=(0.7, 5.0))
    with pytest.raises(ValueError, match="arena_size and oob_margin"):
        PushTLidarScenario(**_LIMO, scenery_probs=_LAYOUTS)
    with pytest.raises(ValueError, match="sum to 1"):
        PushTLidarScenario(**_LIMO, arena_size=(2.0, 5.0), scenery_probs=(0.5, 0.5, 0.5, 0.0))
