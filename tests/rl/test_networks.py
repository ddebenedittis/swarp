"""The role-aware networks of ``swarp.rl``: role routing, the state critic, the aux head."""

from __future__ import annotations

import pytest
import torch

from swarp.rl.networks import StateCritic, role_permutation, role_slots

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
