"""The MAPPO networks and their tanh-squashed Gaussian, in plain torch.

Numerically the same modules as the torchrl reference trainer (``examples/mappo.py``):
a shared-weight per-agent actor (torchrl ``MultiAgentMLP(centralised=False,
share_params=True)`` followed by ``NormalParamExtractor``) and a centralised critic
(``MultiAgentMLP(centralised=True, share_params=True)``). Both are depth-2 Tanh MLPs with
torch's default ``nn.Linear`` init. The state-dict keys differ, so checkpoints of the two
trainers are not interchangeable; :func:`swarp.rl.ppo.load_checkpoint` reads this one's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

# tensordict's ``biased_softplus_1.0``: softplus(x + b) + min_val, with b chosen so a zero
# input maps to a std of exactly 1.0. ``NormalParamExtractor`` then clamps at 1e-4.
_SCALE_MIN = 0.01
_SCALE_BIAS = math.log(math.expm1(1.0 - _SCALE_MIN))
_SCALE_LB = 1e-4
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_LOG2 = math.log(2.0)


def mlp(n_in: int, n_out: int, num_cells: int, depth: int = 2) -> nn.Sequential:
    """``depth`` hidden Tanh layers of ``num_cells``, then a linear head."""
    layers: list[nn.Module] = []
    width = n_in
    for _ in range(depth):
        layers += [nn.Linear(width, num_cells), nn.Tanh()]
        width = num_cells
    layers.append(nn.Linear(width, n_out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Decentralised actor with shared weights: ``[*, n_agents, obs_dim]`` -> (loc, scale).

    The action is ``tanh(u)`` with ``u ~ N(loc, scale)``, i.e. torchrl's ``TanhNormal`` on
    ``[-1, 1]``, mapped affinely onto ``[low, high]`` per action slot when a box is given
    (ring road's accel and steering angle are not unit-ranged). The trainer stores ``u``
    rather than the action, so the log-prob never goes through ``atanh``, and the affine
    map's constant Jacobian cancels in PPO's ratio: the loss is the unit-box loss.

    The box is held in non-persistent buffers, so it moves with ``.to()`` but stays out of
    the state dict and checkpoints from before it existed still load.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_cells: int = 256,
        low: torch.Tensor | None = None,
        high: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim, self.act_dim, self.num_cells = obs_dim, act_dim, num_cells
        self.net = mlp(obs_dim, 2 * act_dim, num_cells)
        low = -torch.ones(act_dim) if low is None else torch.as_tensor(low, dtype=torch.float32)
        high = torch.ones(act_dim) if high is None else torch.as_tensor(high, dtype=torch.float32)
        # A Python flag, read when a graph is captured: the unit box costs nothing extra.
        self.unit_box = bool((low == -1.0).all() and (high == 1.0).all())
        self.register_buffer("act_mid", 0.5 * (high + low), persistent=False)
        self.register_buffer("act_half", 0.5 * (high - low), persistent=False)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc, raw = self.net(obs).chunk(2, dim=-1)
        scale = (F.softplus(raw + _SCALE_BIAS) + _SCALE_MIN).clamp_min(_SCALE_LB)
        return loc, scale

    def squash(self, u: torch.Tensor) -> torch.Tensor:
        """The env action for a pre-tanh sample ``u``."""
        t = torch.tanh(u)
        return t if self.unit_box else torch.addcmul(self.act_mid, self.act_half, t)

    @torch.no_grad()
    def greedy(self, obs: torch.Tensor) -> torch.Tensor:
        """The deterministic action ``squash(loc)``: consumes no RNG."""
        return self.squash(self(obs)[0])


class Critic(nn.Module):
    """Centralised critic: every agent's obs in, one value, broadcast to each agent.

    ``[*, n_agents, obs_dim]`` -> ``[*, n_agents]``. Its input width is
    ``n_agents * obs_dim``, so a critic does not transfer across agent counts.
    """

    def __init__(self, obs_dim: int, n_agents: int, num_cells: int = 256) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.net = mlp(n_agents * obs_dim, 1, num_cells)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        v = self.net(obs.flatten(-2))
        return v.expand(*v.shape[:-1], self.n_agents)


# ------------------------------------------------------------------------------- roles


def role_slots(roles: torch.Tensor, n_roles: int) -> torch.Tensor:
    """Each agent's slot in role-sorted order, ``[*, A]`` long.

    Role-sorted order lists role 0's agents first, then role 1's, and so on, ties by agent
    index (a stable sort). Built from a one-hot cumulative sum, with no sort, no host read
    and no data-dependent shape, so it is CUDA-graph capturable. The role offsets come
    from the data, so a batch need not honour ``role_counts`` for this to be a valid
    permutation; the fixed-size slicing in :class:`PolicyNet` does need it.
    """
    roles = roles.long()
    onehot = (roles.unsqueeze(-1) == torch.arange(n_roles, device=roles.device)).long()
    rank = ((onehot.cumsum(-2) - 1) * onehot).sum(-1)  # index within the agent's role
    count = onehot.sum(-2)  # [*, R]
    offset = (count.cumsum(-1) - count).gather(-1, roles)  # first slot of the agent's role
    return offset + rank


def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    """The inverse of a batch of permutations over the last dim (a scatter, no sort)."""
    ar = torch.arange(perm.shape[-1], device=perm.device).expand_as(perm)
    return torch.empty_like(perm).scatter_(-1, perm, ar)


def role_permutation(roles: torch.Tensor, counts: Sequence[int]) -> torch.Tensor:
    """The agent at each role-sorted slot, ``[*, A]`` long: ``x.gather(-1, perm)`` lists
    ``x`` in role-sorted order. Equal to ``torch.argsort(roles, stable=True)``, without the
    sort (see :func:`role_slots`)."""
    return invert_permutation(role_slots(roles, len(counts)))


def _take(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``x[..., index[k], :]`` along the agent dim (-2) of ``x`` ``[*, A, F]``."""
    return x.gather(-2, index.unsqueeze(-1).expand(*index.shape, x.shape[-1]))


# ------------------------------------------------------------------------------ actors

ROLE_PARAMS = ("shared", "heads", "separate")


def _trunk(n_in: int, num_cells: int, depth: int = 2) -> nn.Sequential:
    """:func:`mlp` without its linear head: ``depth`` Tanh layers, ``num_cells`` out."""
    layers: list[nn.Module] = []
    for i in range(depth):
        layers += [nn.Linear(n_in if i == 0 else num_cells, num_cells), nn.Tanh()]
    return nn.Sequential(*layers)


class PolicyNet(nn.Module):
    """Decentralised shared-weight actor with per-role parameters and an auxiliary head.

    The same tanh-Gaussian as :class:`Actor` (``squash``, ``greedy`` and the scale law are
    identical), over ``[*, A, obs_dim]`` observations plus each agent's role ``[*, A]``.
    ``role_params`` says where roles enter when ``role_counts`` is non-empty:

    * ``"shared"``: one trunk and one head over ``cat(obs, onehot(role))``.
    * ``"heads"``: the same trunk; the final layer outputs ``R * 2 * act_dim`` values,
      viewed ``[*, A, R, 2 * act_dim]`` and gathered by role, so a role's head rows get
      gradient only from that role's agents.
    * ``"separate"``: one trunk and head per role, each applied to its role's fixed-size
      slice of the agents in role-sorted order (:func:`role_permutation`): ``R`` plain
      GEMMs rather than a per-sample weight gather.

    Without roles it is :class:`Actor`'s network plus the optional ``aux_head``: a linear
    map from the trunk's last hidden layer to ``aux_dim`` targets (e.g. a hidden pose),
    trained by the trainer's auxiliary loss.

    :meth:`step` is the general entry point, ``(obs, roles, h) -> (loc, scale, aux, h)``;
    the hidden state ``h`` is passed through untouched, the slot a recurrent core uses.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_cells: int = 256,
        *,
        role_counts: Sequence[int] = (),
        role_params: str = "heads",
        aux_dim: int = 0,
        low: torch.Tensor | Sequence[float] | None = None,
        high: torch.Tensor | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if role_params not in ROLE_PARAMS:
            raise ValueError(f"role_params must be one of {ROLE_PARAMS}, got {role_params!r}")
        self.obs_dim, self.act_dim, self.num_cells = obs_dim, act_dim, num_cells
        self.role_counts = tuple(int(c) for c in role_counts)
        self.role_params, self.aux_dim = role_params, aux_dim
        R = len(self.role_counts)
        self.n_roles = max(R, 1)
        # Python flags, read when a graph is captured.
        self.separate = R > 0 and role_params == "separate"
        self.role_heads = R > 0 and role_params == "heads"
        self.role_input = R > 0 and role_params in ("shared", "heads")
        if self.separate:
            self.trunks = nn.ModuleList(_trunk(obs_dim, num_cells) for _ in range(R))
            self.heads = nn.ModuleList(nn.Linear(num_cells, 2 * act_dim) for _ in range(R))
            offsets = [sum(self.role_counts[:r]) for r in range(R)]
            self._spans = list(zip(offsets, self.role_counts, strict=True))
        else:
            self.trunk = _trunk(obs_dim + (R if self.role_input else 0), num_cells)
            self.head = nn.Linear(num_cells, (R if self.role_heads else 1) * 2 * act_dim)
        self.aux_head = nn.Linear(num_cells, aux_dim) if aux_dim else None
        self.register_buffer("role_ids", torch.arange(self.n_roles), persistent=False)
        lo = -torch.ones(act_dim) if low is None else torch.as_tensor(low, dtype=torch.float32)
        hi = torch.ones(act_dim) if high is None else torch.as_tensor(high, dtype=torch.float32)
        self.unit_box = bool((lo == -1.0).all() and (hi == 1.0).all())
        self.register_buffer("act_mid", 0.5 * (hi + lo), persistent=False)
        self.register_buffer("act_half", 0.5 * (hi - lo), persistent=False)
        #: The constructor's arguments as primitives, which is what a checkpoint stores.
        self.config = {
            "obs_dim": obs_dim, "act_dim": act_dim, "num_cells": num_cells,
            "role_counts": list(self.role_counts), "role_params": role_params,
            "aux_dim": aux_dim,
            "low": None if low is None else lo.tolist(),
            "high": None if high is None else hi.tolist(),
        }

    @property
    def uses_roles(self) -> bool:
        """Whether :meth:`step` reads ``roles``."""
        return bool(self.role_counts)

    def step(
        self,
        obs: torch.Tensor,
        roles: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
        *,
        with_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """``(loc, scale, aux, h)``: the Gaussian over the pre-tanh action, ``[*, A,
        act_dim]`` each, the auxiliary prediction ``[*, A, aux_dim]`` when ``with_aux``
        (else ``None``), and the hidden state, passed through."""
        if self.role_counts and roles is None:
            raise ValueError("this PolicyNet has roles; pass roles [*, n_agents]")
        if self.separate:
            slots = role_slots(roles, self.n_roles)
            o = _take(obs, invert_permutation(slots))
            feats = [
                trunk(o[..., off : off + n, :])
                for trunk, (off, n) in zip(self.trunks, self._spans, strict=True)
            ]
            outs = [head(f) for head, f in zip(self.heads, feats, strict=True)]
            out = _take(torch.cat(outs, -2), slots)
            feat = _take(torch.cat(feats, -2), slots) if with_aux else None
        else:
            if self.role_input:
                roles = roles.long()
                onehot = (roles.unsqueeze(-1) == self.role_ids).to(obs.dtype)
                obs = torch.cat([obs, onehot], -1)
            feat = self.trunk(obs)
            out = self.head(feat)
            if self.role_heads:
                two_d = 2 * self.act_dim
                out = out.view(*out.shape[:-1], self.n_roles, two_d)
                index = roles[..., None, None].expand(*roles.shape, 1, two_d)
                out = out.gather(-2, index).squeeze(-2)
        loc, raw = out.chunk(2, dim=-1)
        scale = (F.softplus(raw + _SCALE_BIAS) + _SCALE_MIN).clamp_min(_SCALE_LB)
        aux = None
        if with_aux:
            if self.aux_head is None:
                raise ValueError("this PolicyNet has no aux head (aux_dim=0)")
            aux = self.aux_head(feat)
        return loc, scale, aux, h

    def forward(
        self, obs: torch.Tensor, roles: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loc, scale, _, _ = self.step(obs, roles)
        return loc, scale

    def squash(self, u: torch.Tensor) -> torch.Tensor:
        """The env action for a pre-tanh sample ``u`` (as :meth:`Actor.squash`)."""
        t = torch.tanh(u)
        return t if self.unit_box else torch.addcmul(self.act_mid, self.act_half, t)

    @torch.no_grad()
    def greedy(self, obs: torch.Tensor, roles: torch.Tensor | None = None) -> torch.Tensor:
        """The deterministic action ``squash(loc)``: consumes no RNG."""
        return self.squash(self(obs, roles)[0])


# ----------------------------------------------------------------------------- critics


class StateCritic(nn.Module):
    """Centralised critic over a privileged state, the agents' observations, or both.

    ``forward(obs [*, A, obs_dim], state [*, state_dim], roles [*, A]) -> [*, A]``. The
    input is ``state`` (if ``state_dim > 0``) followed, with ``use_obs``, by every agent's
    observation flattened in role-sorted order when the scenario has roles (agent order
    otherwise), so the critic sees the team in a fixed layout however roles were drawn.

    One value broadcast to every agent by default. With ``per_agent`` it outputs one value
    per role-sorted slot (per agent index without roles), mapped back to agent order: the
    per-role rewards then get a value each instead of sharing the team's.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        n_agents: int,
        num_cells: int = 256,
        *,
        use_obs: bool = False,
        role_counts: Sequence[int] = (),
        per_agent: bool = False,
    ) -> None:
        super().__init__()
        n_in = state_dim + (n_agents * obs_dim if use_obs else 0)
        if n_in == 0:
            raise ValueError("StateCritic needs state_dim > 0 or use_obs=True")
        if role_counts and sum(role_counts) != n_agents:
            raise ValueError(f"role_counts {tuple(role_counts)} do not sum to {n_agents} agents")
        self.state_dim, self.obs_dim, self.n_agents = state_dim, obs_dim, n_agents
        self.use_obs, self.per_agent = use_obs, per_agent
        self.role_counts = tuple(int(c) for c in role_counts)
        #: Whether ``forward`` reads ``roles``: only to order the obs or the per-slot values.
        self.uses_roles = bool(self.role_counts) and (use_obs or per_agent)
        self.net = mlp(n_in, n_agents if per_agent else 1, num_cells)
        #: The constructor's arguments as primitives, which is what a checkpoint stores.
        self.config = {
            "state_dim": state_dim, "obs_dim": obs_dim, "n_agents": n_agents,
            "num_cells": num_cells, "use_obs": use_obs,
            "role_counts": list(self.role_counts), "per_agent": per_agent,
        }

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | None = None,
        roles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        slots = role_slots(roles, len(self.role_counts)) if self.uses_roles else None
        parts = [state] if self.state_dim else []
        if self.use_obs:
            o = obs if slots is None else _take(obs, invert_permutation(slots))
            parts.append(o.flatten(-2))
        v = self.net(parts[0] if len(parts) == 1 else torch.cat(parts, -1))
        if not self.per_agent:
            return v.expand(*v.shape[:-1], self.n_agents)
        return v if slots is None else v.gather(-1, slots)


def tanh_normal_log_prob(
    u: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Log-density of ``tanh(u)`` for ``u ~ N(loc, scale)``, summed over the action dim.

    The tanh correction ``log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u))`` is
    ``TanhTransform.log_abs_det_jacobian``, the stable form.
    """
    z = (u - loc) / scale
    normal = -0.5 * z * z - scale.log() - _LOG_SQRT_2PI
    jac = 2.0 * (_LOG2 - u - F.softplus(-2.0 * u))
    return (normal - jac).sum(-1)
