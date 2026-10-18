"""PushTLidarScenario's own contract: roles, trainer buffers, reward terms, layout."""

import pytest
import torch
from conftest import DEVICES

from swarp import Environment, PushTLidarScenario, PushTScenario
from swarp.scenarios.pusht_lidar_kernels import OBJECTIVES, STATE_AGENT, STATE_HEAD


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
