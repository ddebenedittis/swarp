"""The role-aware networks of ``swarp.rl``: role routing, the state critic, the aux head."""

from __future__ import annotations

import pytest
import torch

from swarp.rl.networks import Actor, PolicyNet, StateCritic, role_permutation, role_slots

COUNTS = (2, 3, 1)  # three roles over six agents


def _random_roles(batch: int, counts=COUNTS, seed: int = 0) -> torch.Tensor:
    """``[batch, A]`` roles with exactly ``counts[r]`` agents in role ``r`` per row."""
    gen = torch.Generator().manual_seed(seed)
    base = torch.cat([torch.full((c,), r) for r, c in enumerate(counts)])
    return torch.stack([base[torch.randperm(len(base), generator=gen)] for _ in range(batch)])


def test_role_permutation_is_a_stable_argsort():
    roles = _random_roles(64)
    perm = role_permutation(roles, COUNTS)
    assert perm.dtype == torch.long
    assert torch.equal(perm, torch.argsort(roles, dim=-1, stable=True))
    # role_slots is its inverse: the agent at slot role_slots[i] is agent i.
    slots = role_slots(roles, len(COUNTS))
    assert torch.equal(perm.gather(-1, slots), torch.arange(6).expand(64, 6))


def test_role_permutation_handles_extra_batch_dims():
    roles = _random_roles(12).view(3, 4, 6)
    assert torch.equal(role_permutation(roles, COUNTS), torch.argsort(roles, dim=-1, stable=True))


def test_state_critic_per_agent_values_follow_their_role_slot():
    """Slot ``j`` of the per-agent head belongs to the ``j``-th agent in role-sorted order,
    whichever agent index holds it this episode."""
    torch.manual_seed(0)
    critic = StateCritic(5, 3, 6, 16, role_counts=COUNTS, per_agent=True)
    state = torch.randn(8, 5)
    roles = _random_roles(8)
    v = critic(torch.zeros(8, 6, 3), state, roles)
    v_sorted = critic.net(state)  # the head's own slot order
    perm = torch.argsort(roles, dim=-1, stable=True)
    assert torch.allclose(v.gather(-1, perm), v_sorted)


def test_state_critic_reads_obs_in_role_order():
    """With ``use_obs`` the critic is invariant to a joint permutation of agents and roles
    that keeps the role-sorted layout, and its per-agent values permute with the agents."""
    torch.manual_seed(1)
    critic = StateCritic(4, 3, 6, 16, use_obs=True, role_counts=COUNTS, per_agent=True)
    obs, state, roles = torch.randn(8, 6, 3), torch.randn(8, 4), _random_roles(8)
    v = critic(obs, state, roles)
    # Re-order agents (and their roles) the same way per row, keeping within-role order.
    sigma = torch.argsort(roles, dim=-1, stable=True)
    v2 = critic(obs.gather(1, sigma[..., None].expand(-1, -1, 3)), state, roles.gather(1, sigma))
    assert torch.allclose(v2, v.gather(1, sigma), atol=1e-6)


def test_state_critic_broadcasts_one_value_by_default():
    critic = StateCritic(5, 3, 4, 16)
    v = critic(torch.zeros(7, 4, 3), torch.randn(7, 5), None)
    assert v.shape == (7, 4) and torch.equal(v, v[:, :1].expand(7, 4))
    assert not critic.uses_roles


def test_state_critic_rejects_an_empty_input():
    with pytest.raises(ValueError, match="state_dim > 0 or use_obs"):
        StateCritic(0, 3, 4)


# ------------------------------------------------------------------------------ PolicyNet


def _joint_permutation(obs, roles, seed: int = 3):
    """A random per-row permutation of the agents, applied to obs and roles together."""
    gen = torch.Generator().manual_seed(seed)
    sigma = torch.stack([torch.randperm(roles.shape[-1], generator=gen) for _ in roles])
    return sigma, _take_rows(obs, sigma), roles.gather(-1, sigma)


def _take_rows(x, index):
    return x.gather(1, index[..., None].expand(*index.shape, x.shape[-1]))


@pytest.mark.parametrize("role_params", ["shared", "heads", "separate"])
def test_policy_routing_is_equivariant_under_a_joint_permutation(role_params):
    """An agent's action depends on its own obs and role only, not on its index."""
    torch.manual_seed(0)
    net = PolicyNet(4, 2, 16, role_counts=COUNTS, role_params=role_params, aux_dim=3)
    obs, roles = torch.randn(10, 6, 4), _random_roles(10)
    loc, scale, aux, _ = net.step(obs, roles, with_aux=True)
    sigma, obs_p, roles_p = _joint_permutation(obs, roles)
    loc_p, scale_p, aux_p, _ = net.step(obs_p, roles_p, with_aux=True)
    assert torch.allclose(loc_p, _take_rows(loc, sigma), atol=1e-6)
    assert torch.allclose(scale_p, _take_rows(scale, sigma), atol=1e-6)
    assert torch.allclose(aux_p, _take_rows(aux, sigma), atol=1e-6)


@pytest.mark.parametrize("role_params", ["heads", "separate"])
def test_role_specific_parameters_get_gradient_only_from_their_role(role_params):
    torch.manual_seed(0)
    net = PolicyNet(4, 2, 16, role_counts=COUNTS, role_params=role_params)
    obs, roles = torch.randn(10, 6, 4), _random_roles(10)
    loc, scale = net(obs, roles)
    ((loc.square() + scale) * (roles == 0).unsqueeze(-1)).sum().backward()
    if role_params == "heads":
        g = net.head.weight.grad.view(len(COUNTS), 4, 16)  # [R, 2 * act_dim, F]
        assert g[0].abs().sum() > 0
        assert torch.equal(g[1:], torch.zeros_like(g[1:]))
    else:
        for r, (trunk, head) in enumerate(zip(net.trunks, net.heads, strict=True)):
            grads = [p.grad for p in (*trunk.parameters(), *head.parameters())]
            if r == 0:
                assert all(g is not None and g.abs().sum() > 0 for g in grads)
            else:
                assert all(g is None or torch.equal(g, torch.zeros_like(g)) for g in grads)


def test_policy_without_roles_is_an_actor_with_an_aux_head():
    """No roles: the same layer shapes and the same tanh-Gaussian maths as ``Actor``."""
    torch.manual_seed(0)
    net = PolicyNet(3, 2, 8, aux_dim=2, low=[-2.0, -0.5], high=[2.0, 0.5])
    actor = Actor(3, 2, 8, low=torch.tensor([-2.0, -0.5]), high=torch.tensor([2.0, 0.5]))
    for dst, src in zip((*net.trunk, net.head), actor.net, strict=True):
        dst.load_state_dict(src.state_dict())
    obs = torch.randn(5, 4, 3)
    for a, b in zip(net(obs), actor(obs), strict=True):
        assert torch.equal(a, b)
    assert torch.equal(net.greedy(obs), actor.greedy(obs))
    u = torch.randn(5, 4, 2) * 5
    assert torch.equal(net.squash(u), actor.squash(u))
    assert "act_mid" not in net.state_dict()


def test_aux_loss_trains_the_head_and_coef_zero_leaves_the_policy_gradient_alone():
    from swarp.rl.ppo import ppo_terms

    torch.manual_seed(0)
    net = PolicyNet(4, 2, 16, role_counts=COUNTS, aux_dim=3)
    obs, roles = torch.randn(10, 6, 4), _random_roles(10)
    u, noise = torch.randn(10, 6, 2), torch.randn(10, 6, 2)
    logp_old, adv, target = torch.randn(10, 6), torch.randn(10, 6), torch.randn(10, 6, 3)

    def grads(coef: float | None) -> list[torch.Tensor]:
        net.zero_grad(set_to_none=False)
        loc, scale, aux, _ = net.step(obs, roles, with_aux=coef is not None)
        obj, ent = ppo_terms(loc, scale, u, logp_old, adv, noise)
        loss = obj - 1e-2 * ent
        if coef is not None:
            loss = loss + coef * torch.nn.functional.mse_loss(aux, target)
        loss.backward()
        return [torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in net.parameters()]

    plain, zero, on = grads(None), grads(0.0), grads(1.0)
    for a, b in zip(plain, zero, strict=True):
        assert torch.equal(a, b)
    assert net.aux_head.weight.grad.abs().sum() > 0  # the last call, coef 1.0
    trunk_w = net.trunk[0].weight
    i = [p is trunk_w for p in net.parameters()].index(True)
    assert not torch.equal(on[i], plain[i])  # the aux loss shapes the shared features


def test_policy_needs_roles_when_it_has_them():
    net = PolicyNet(4, 2, 8, role_counts=(1, 1))
    with pytest.raises(ValueError, match="pass roles"):
        net(torch.zeros(2, 2, 4))
    with pytest.raises(ValueError, match="role_params"):
        PolicyNet(4, 2, 8, role_counts=(1, 1), role_params="mixture")


# ---------------------------------------------------------------- memory and communication


@pytest.mark.parametrize("role_params", ["heads", "separate"])
def test_unroll_equals_stepping_with_the_hidden_state_reset_at_dones(role_params):
    """unroll runs the trunk once over the sequence and only the GRU step by step; it must
    equal L calls of step, with h zeroed after each done, including roles that change at
    a done (separate GRUs then swap agents)."""
    torch.manual_seed(0)
    net = PolicyNet(4, 2, 16, role_counts=COUNTS, role_params=role_params, aux_dim=3,
                    recurrent=True, hidden_dim=8)
    L, S = 5, 7
    obs = torch.randn(L, S, 6, 4)
    roles = torch.stack([_random_roles(S, seed=s) for s in range(L)])
    done = (torch.rand(L, S) < 0.3).float()
    h0 = torch.randn(S, 6, 8)
    loc, scale, aux, h_last = net.unroll(obs, roles, h0, done, with_aux=True)
    h, ref = h0, []
    for i in range(L):
        if i:
            h = h * (1 - done[i - 1])[:, None, None]
        out = net.step(obs[i], roles[i], h, with_aux=True)
        h = out[3]
        ref.append(out[:3])
    for got, want in zip((loc, scale, aux), zip(*ref, strict=True), strict=True):
        assert torch.allclose(got, torch.stack(want), atol=1e-6)
    assert torch.allclose(h_last, h, atol=1e-6)
    assert loc.shape == (L, S, 6, 2) and h_last.shape == (S, 6, 8)


def test_recurrent_policy_carries_its_hidden_state():
    torch.manual_seed(0)
    net = PolicyNet(3, 2, 8, recurrent=True, hidden_dim=5)
    obs = torch.randn(4, 2, 3)
    loc0, _, _, h1 = net.step(obs)  # from zeros
    assert torch.equal(loc0, net.step(obs, h=torch.zeros(4, 2, 5))[0])
    loc1, _, _, h2 = net.step(obs, h=h1)
    assert h1.shape == (4, 2, 5) and not torch.allclose(loc0, loc1)
    assert torch.equal(net.greedy(obs, h=h1), net.squash(loc1))
    assert net.stateful and net.config["recurrent"] and net.config["hidden_dim"] == 5
    assert "recurrent" not in PolicyNet(3, 2, 8).config  # absent when off


def test_attention_never_attends_to_self_and_respects_the_mask():
    torch.manual_seed(0)
    net = PolicyNet(4, 2, 16, role_counts=COUNTS, comm="attn", comm_dim=8)
    B, A = 9, 6
    obs, roles = torch.randn(B, A, 4), _random_roles(B)
    mask = torch.rand(B, A, A) < 0.5
    mask[0, 1] = torch.tensor([True, True, False, False, True, False])  # 1 hears 0 and 4
    mask[0, 2] = False  # agent 2 of row 0 hears nobody
    eye = torch.eye(A, dtype=torch.bool)
    for m in (None, mask):
        w = net.comm_weights(obs, roles, m)
        allowed = (~eye if m is None else m & ~eye).expand(B, A, A)
        assert torch.equal(w.diagonal(dim1=-2, dim2=-1), torch.zeros(B, A))
        assert torch.equal(w[~allowed], torch.zeros(int((~allowed).sum())))
        heard = allowed.any(-1)
        assert torch.allclose(w.sum(-1)[heard], torch.ones(int(heard.sum())), atol=1e-6)
        assert torch.equal(w.sum(-1)[~heard], torch.zeros(int((~heard).sum())))
    # What a masked-off teammate sees cannot reach the listener; what a heard one sees can.
    loc = net.step(obs, roles, comm_mask=mask)[0]
    for j, reaches in ((3, False), (4, True)):
        obs2 = obs.clone()
        obs2[0, j] += 1.0
        loc2 = net.step(obs2, roles, comm_mask=mask)[0]
        assert torch.equal(loc2[0, 1], loc[0, 1]) is not reaches
    alone = PolicyNet(4, 2, 16, comm="attn", comm_dim=8)
    assert torch.isfinite(alone.step(obs[:, :1])[0]).all()  # one agent: a zero message


def test_attention_gradients_are_finite_for_an_isolated_agent():
    torch.manual_seed(0)
    net = PolicyNet(4, 2, 16, comm="attn", comm_dim=8, recurrent=True, hidden_dim=8)
    obs = torch.randn(3, 4, 4)
    mask = torch.zeros(3, 4, 4, dtype=torch.bool)  # nobody hears anybody
    loc, scale, _, _ = net.step(obs, comm_mask=mask)
    (loc.sum() + scale.sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_communication_needs_a_shared_trunk():
    with pytest.raises(ValueError, match="not 'separate'"):
        PolicyNet(4, 2, 8, role_counts=(1, 1), role_params="separate", comm="attn")
    with pytest.raises(ValueError, match="comm must be one of"):
        PolicyNet(4, 2, 8, comm="radio")
    PolicyNet(4, 2, 8, role_params="separate", comm="attn")  # no roles: one trunk anyway
