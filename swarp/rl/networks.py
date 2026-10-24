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
#: ``PolicyNet.comm`` values: no communication, one round of attention over teammates'
#: current inputs, or a message each agent encodes from its own observation and its
#: teammates hear one step later (:data:`DELAYED_COMMS`).
COMMS = ("none", "attn", "latent", "vae", "pose")
#: The delayed messages: ``"latent"`` a learned ``tanh`` code trained end to end through
#: the listeners' losses, ``"vae"`` a Gaussian code sampled with stored noise, bottlenecked
#: by a KL to the unit Gaussian and grounded by decoding the sender's aux targets, and
#: ``"pose"`` the sender's own estimate of its aux targets with their log-variances,
#: trained by a Gaussian likelihood and detached from the listeners.
DELAYED_COMMS = ("latent", "vae", "pose")
#: Width of the message encoder's (and the VAE decoder's) hidden layer.
MSG_HIDDEN = 128
_LOGVAR_RANGE = (-8.0, 4.0)


def _trunk(n_in: int, num_cells: int, depth: int = 2) -> nn.Sequential:
    """:func:`mlp` without its linear head: ``depth`` Tanh layers, ``num_cells`` out."""
    layers: list[nn.Module] = []
    for i in range(depth):
        layers += [nn.Linear(n_in if i == 0 else num_cells, num_cells), nn.Tanh()]
    return nn.Sequential(*layers)


class PolicyNet(nn.Module):
    """Decentralised shared-weight actor with per-role parameters, memory, communication
    and an auxiliary head.

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
    map from the features the heads read to ``aux_dim`` targets (e.g. a hidden pose),
    trained by the trainer's auxiliary loss.

    ``recurrent`` adds a ``GRUCell(num_cells, hidden_dim)`` after the trunk (one per role
    with ``"separate"``); the heads and the aux head then read its output, which is also
    the new hidden state ``h`` ``[*, A, hidden_dim]``. ``comm="attn"`` adds one round of
    learned communication before the trunk: each agent's input (obs and role one-hot) is
    encoded to ``e`` ``[*, A, comm_dim]``, attends over its teammates with single-head
    scaled dot-product attention (never over itself, and only over the teammates
    ``comm_mask`` ``[*, A, A]`` allows: ``comm_mask[..., i, j]`` means ``i`` hears ``j``),
    and the trunk reads ``cat(e, message)``. Communication needs one shared trunk, so it
    is refused with ``"separate"`` roles.

    A delayed message (``comm`` in :data:`DELAYED_COMMS`) is what a deployment can send
    over a network at the policy rate: agent ``j`` encodes its own observation into
    ``msg_dim`` numbers with ``msg_enc`` (a one-hidden-layer MLP over the observation and
    role one-hot, never its recurrent state), broadcasts them, and its teammates read them
    on their next step. :meth:`step` therefore takes ``msg_obs`` ``[*, A, obs_dim]``, each
    agent's observation of the previous step, and ``msg_mask`` ``[*, A, A]``
    (``msg_mask[..., i, j]``: ``i`` received ``j``'s message; false after an episode start
    or a dropped packet), and agent ``i``'s trunk reads ``cat(obs, role, mean of the
    received messages, any received)``. Those columns come last, so a checkpoint without
    communication widens into this network with zero weights on them
    (:func:`swarp.rl.ppo.load_widened`). ``"vae"`` samples its code with ``msg_noise``
    ``[*, A, comm_dim]`` (the trainer stores the rollout's draw, so the update recomputes
    the same log-probabilities; ``None`` sends the mean, the greedy message), and
    :meth:`message_loss` is the encoder's own objective.

    :meth:`step` is the one-step entry point, ``(obs, roles, h) -> (loc, scale, aux, h)``;
    without ``recurrent`` the hidden state is passed through untouched. :meth:`unroll` runs
    a ``[L, ...]`` sequence from a starting hidden state, resetting it at episode ends.
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
        recurrent: bool = False,
        hidden_dim: int = 128,
        comm: str = "none",
        comm_dim: int = 64,
        low: torch.Tensor | Sequence[float] | None = None,
        high: torch.Tensor | Sequence[float] | None = None,
        scale_min: float = _SCALE_MIN,
    ) -> None:
        super().__init__()
        if role_params not in ROLE_PARAMS:
            raise ValueError(f"role_params must be one of {ROLE_PARAMS}, got {role_params!r}")
        if comm not in COMMS:
            raise ValueError(f"comm must be one of {COMMS}, got {comm!r}")
        self.obs_dim, self.act_dim, self.num_cells = obs_dim, act_dim, num_cells
        self.role_counts = tuple(int(c) for c in role_counts)
        self.role_params, self.aux_dim = role_params, aux_dim
        R = len(self.role_counts)
        self.n_roles = max(R, 1)
        # Python flags, read when a graph is captured.
        self.separate = R > 0 and role_params == "separate"
        self.role_heads = R > 0 and role_params == "heads"
        self.role_input = R > 0 and role_params in ("shared", "heads")
        self.recurrent = bool(recurrent)
        self.hidden_dim = hidden_dim
        self.comm, self.comm_dim = comm, comm_dim
        self.communicates = comm != "none"
        if self.communicates and self.separate:
            raise ValueError(
                f"comm={comm!r} needs one shared trunk: use role_params 'heads' or 'shared' "
                "with roles, not 'separate'"
            )
        self.delayed = comm in DELAYED_COMMS
        if comm == "pose" and not aux_dim:
            raise ValueError("comm='pose' sends the aux estimate: it needs aux_dim > 0")
        #: Width of one delayed message (0 without).
        self.msg_dim = {"latent": comm_dim, "vae": comm_dim, "pose": 2 * aux_dim}.get(comm, 0)
        feat_dim = hidden_dim if self.recurrent else num_cells
        if self.separate:
            self.trunks = nn.ModuleList(_trunk(obs_dim, num_cells) for _ in range(R))
            if self.recurrent:
                self.grus = nn.ModuleList(nn.GRUCell(num_cells, hidden_dim) for _ in range(R))
            self.heads = nn.ModuleList(nn.Linear(feat_dim, 2 * act_dim) for _ in range(R))
            offsets = [sum(self.role_counts[:r]) for r in range(R)]
            self._spans = list(zip(offsets, self.role_counts, strict=True))
        else:
            n_in = obs_dim + (R if self.role_input else 0)
            if comm == "attn":
                self.comm_enc = nn.Linear(n_in, comm_dim)
                self.comm_qkv = nn.Linear(comm_dim, 3 * comm_dim)
                n_in = 2 * comm_dim
            elif self.delayed:
                self.msg_dec = None
                width = 2 * comm_dim if comm == "vae" else self.msg_dim
                self.msg_enc = nn.Sequential(
                    nn.Linear(n_in, MSG_HIDDEN), nn.Tanh(), nn.Linear(MSG_HIDDEN, width)
                )
                if comm == "vae" and aux_dim:
                    self.msg_dec = nn.Sequential(
                        nn.Linear(comm_dim, MSG_HIDDEN), nn.Tanh(), nn.Linear(MSG_HIDDEN, aux_dim)
                    )
                n_in += self.msg_dim + 1
            self.trunk = _trunk(n_in, num_cells)
            if self.recurrent:
                self.gru = nn.GRUCell(num_cells, hidden_dim)
            self.head = nn.Linear(feat_dim, (R if self.role_heads else 1) * 2 * act_dim)
        self.aux_head = nn.Linear(feat_dim, aux_dim) if aux_dim else None
        self.register_buffer("role_ids", torch.arange(self.n_roles), persistent=False)
        lo = -torch.ones(act_dim) if low is None else torch.as_tensor(low, dtype=torch.float32)
        hi = torch.ones(act_dim) if high is None else torch.as_tensor(high, dtype=torch.float32)
        self.unit_box = bool((lo == -1.0).all() and (hi == 1.0).all())
        self.register_buffer("act_mid", 0.5 * (hi + lo), persistent=False)
        self.register_buffer("act_half", 0.5 * (hi - lo), persistent=False)
        #: The constructor's arguments as primitives, which is what a checkpoint stores.
        #: Memory and communication appear only when on, so a checkpoint written before
        #: they existed describes the same network as the flags that build it today.
        # The floor under the Gaussian's std, with the bias that keeps a zero input at 1.0.
        if not 0.0 < scale_min < 1.0:
            raise ValueError(f"scale_min must be in (0, 1), got {scale_min}")
        self.scale_min = float(scale_min)
        self._scale_bias = math.log(math.expm1(1.0 - self.scale_min))
        self.config = {
            "obs_dim": obs_dim, "act_dim": act_dim, "num_cells": num_cells,
            "role_counts": list(self.role_counts), "role_params": role_params,
            "aux_dim": aux_dim,
            "low": None if low is None else lo.tolist(),
            "high": None if high is None else hi.tolist(),
        }
        if self.recurrent:
            self.config.update(recurrent=True, hidden_dim=hidden_dim)
        if self.communicates:
            self.config.update(comm=comm, comm_dim=comm_dim)
        if self.scale_min != _SCALE_MIN:
            self.config.update(scale_min=self.scale_min)

    @property
    def uses_roles(self) -> bool:
        """Whether :meth:`step` reads ``roles``."""
        return bool(self.role_counts)

    @property
    def stateful(self) -> bool:
        """Whether acting needs more than ``(obs, roles)``: a hidden state to carry, a
        communication mask to read from the env, or the previous step's messages."""
        return self.recurrent or self.communicates

    # ------------------------------------------------------------------ the pieces
    # A role layout ``route`` is ``(slots, perm)`` for "separate" (role_slots and its
    # inverse), else None. With "separate" the features are a list of per-role tensors in
    # role-sorted order, otherwise one tensor in agent order.

    def _route(self, roles: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.separate:
            return None
        slots = role_slots(roles, self.n_roles)
        return slots, invert_permutation(slots)

    def _with_role(self, obs: torch.Tensor, roles: torch.Tensor | None) -> torch.Tensor:
        if not self.role_input:
            return obs
        onehot = (roles.unsqueeze(-1) == self.role_ids).to(obs.dtype)
        return torch.cat([obs, onehot], -1)

    def _communicate(
        self, x: torch.Tensor, comm_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cat(e, message), weights)``: one round of attention over teammates."""
        e = torch.tanh(self.comm_enc(x))
        q, k, v = self.comm_qkv(e).chunk(3, -1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (self.comm_dim**-0.5)
        n = x.shape[-2]
        allowed = ~torch.eye(n, dtype=torch.bool, device=x.device)
        if comm_mask is not None:
            allowed = allowed & comm_mask.bool()
        scores = scores.masked_fill(~allowed, float("-inf"))
        # An agent with nobody to hear (one agent, or masked off) gets a zero message, not
        # the NaN of a softmax over nothing, in the forward and the backward pass.
        scores = torch.where(allowed.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        w = torch.softmax(scores, -1) * allowed
        return torch.cat([e, torch.matmul(w, v)], -1), w

    def encode(
        self, msg_obs: torch.Tensor, roles: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The encoder's ``(mean, log-variance)`` ``[*, A, *]`` for each agent's own
        observation (``"latent"``: the pre-``tanh`` code and ``None``)."""
        out = self.msg_enc(self._with_role(msg_obs, roles))
        if self.comm == "latent":
            return out, None
        mean, logvar = out.chunk(2, -1)
        return mean, logvar.clamp(*_LOGVAR_RANGE)

    def messages(
        self,
        msg_obs: torch.Tensor,
        roles: torch.Tensor | None,
        msg_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[*, A, msg_dim]``: what each agent sends for the observation ``msg_obs``."""
        mean, logvar = self.encode(msg_obs, roles)
        if self.comm == "latent":
            return torch.tanh(mean)
        if self.comm == "vae":
            return mean if msg_noise is None else mean + torch.exp(0.5 * logvar) * msg_noise
        return torch.cat([mean, logvar], -1).detach()

    def _listen(self, obs, roles, msg_obs, msg_mask, msg_noise):
        """``[*, A, msg_dim + 1]``: the mean of the messages each agent received, and
        whether it received any (zeros when ``msg_obs`` is ``None``)."""
        if msg_obs is None or msg_mask is None:
            return obs.new_zeros(*obs.shape[:-1], self.msg_dim + 1)
        m = self.messages(msg_obs, roles, msg_noise)
        n = obs.shape[-2]
        heard = (msg_mask.bool() & ~torch.eye(n, dtype=torch.bool, device=obs.device)).to(m.dtype)
        count = heard.sum(-1, keepdim=True)
        pooled = torch.matmul(heard, m) / count.clamp(min=1.0)
        return torch.cat([pooled, (count > 0).to(m.dtype)], -1)

    def message_loss(
        self,
        msg_obs: torch.Tensor,
        roles: torch.Tensor | None,
        target: torch.Tensor,
        valid: torch.Tensor,
        beta: float = 0.0,
        msg_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The encoder's own objective over the senders ``valid`` ``[*, A]`` marks:
        ``"pose"`` the Gaussian negative log-likelihood of ``target`` ``[*, A, aux_dim]``
        (the senders' aux targets for ``msg_obs``) under its estimate, ``"vae"`` the
        decoder's squared error on it from the code sampled with ``msg_noise`` (the one the
        listeners heard; the mean when ``None``) plus ``beta`` times the KL to the unit
        Gaussian, ``"latent"`` zero (it learns only through the listeners)."""
        if self.comm == "latent":
            return msg_obs.new_zeros(())
        mean, logvar = self.encode(msg_obs, roles)
        w = valid.to(mean.dtype).unsqueeze(-1)
        denom = w.sum().clamp(min=1.0)
        if self.comm == "pose":
            nll = 0.5 * ((target - mean).square() * torch.exp(-logvar) + logvar)
            return (nll * w).sum() / (denom * nll.shape[-1])
        z = mean if msg_noise is None else mean + torch.exp(0.5 * logvar) * msg_noise
        kl = 0.5 * (mean.square() + logvar.exp() - 1.0 - logvar).sum(-1, keepdim=True)
        loss = beta * (kl * w).sum() / denom
        if self.msg_dec is not None:
            err = (self.msg_dec(z) - target).square()
            loss = loss + (err * w).sum() / (denom * err.shape[-1])
        return loss

    def _pre(self, obs, roles, route, comm_mask, msg=(None, None, None)):
        """Everything before the recurrent core: it never reads ``h``."""
        if self.separate:
            o = _take(obs, route[1])
            return [
                trunk(o[..., off : off + n, :])
                for trunk, (off, n) in zip(self.trunks, self._spans, strict=True)
            ]
        x = self._with_role(obs, roles)
        if self.comm == "attn":
            x = self._communicate(x, comm_mask)[0]
        elif self.delayed:
            x = torch.cat([x, self._listen(obs, roles, *msg)], -1)
        return self.trunk(x)

    def _input_gates(self, feat):
        """The GRU's input projection ``W_ih x + b_ih``, the part that never reads ``h``:
        :meth:`unroll` runs it once over the whole sequence."""
        if self.separate:
            return [
                F.linear(f, gru.weight_ih, gru.bias_ih)
                for gru, f in zip(self.grus, feat, strict=True)
            ]
        return F.linear(feat, self.gru.weight_ih, self.gru.bias_ih)

    @staticmethod
    def _gru(cell: nn.GRUCell, gi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """``cell(x, h)`` from the input gates ``gi`` of ``x``: torch's GRU cell, the fused
        kernel ``GRUCell`` itself runs on CUDA, the same equations elsewhere."""
        gi2, h2 = gi.reshape(-1, gi.shape[-1]), h.reshape(-1, h.shape[-1])
        gh = F.linear(h2, cell.weight_hh, cell.bias_hh)
        if gi2.is_cuda:
            out = torch.ops.aten._thnn_fused_gru_cell(gi2, gh, h2)[0]
        else:
            i_r, i_z, i_n = gi2.chunk(3, 1)
            h_r, h_z, h_n = gh.chunk(3, 1)
            r, z = torch.sigmoid(i_r + h_r), torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            out = n + z * (h2 - n)
        return out.view(*gi.shape[:-1], out.shape[-1])

    def _core(self, gi, h, route):
        """One GRU step from the input gates: ``(features for the heads, new h in agent
        order)``."""
        if self.separate:
            hs = _take(h, route[1])
            outs = [
                self._gru(gru, g, hs[..., off : off + n, :])
                for gru, g, (off, n) in zip(self.grus, gi, self._spans, strict=True)
            ]
            return outs, _take(torch.cat(outs, -2), route[0])
        out = self._gru(self.gru, gi, h)
        return out, out

    def _post(self, feat, roles, route, with_aux):
        """The heads: ``(loc, scale, aux)`` in agent order."""
        if self.separate:
            outs = [head(f) for head, f in zip(self.heads, feat, strict=True)]
            out = _take(torch.cat(outs, -2), route[0])
            feat = _take(torch.cat(feat, -2), route[0]) if with_aux else None
        else:
            out = self.head(feat)
            if self.role_heads:
                two_d = 2 * self.act_dim
                out = out.view(*out.shape[:-1], self.n_roles, two_d)
                index = roles[..., None, None].expand(*roles.shape, 1, two_d)
                out = out.gather(-2, index).squeeze(-2)
        loc, raw = out.chunk(2, dim=-1)
        scale = (F.softplus(raw + self._scale_bias) + self.scale_min).clamp_min(_SCALE_LB)
        return loc, scale, self.aux_head(feat) if with_aux else None

    def _check(self, roles: torch.Tensor | None, with_aux: bool) -> torch.Tensor | None:
        if self.role_counts and roles is None:
            raise ValueError("this PolicyNet has roles; pass roles [*, n_agents]")
        if with_aux and self.aux_head is None:
            raise ValueError("this PolicyNet has no aux head (aux_dim=0)")
        return roles.long() if self.role_counts else roles

    # ------------------------------------------------------------------ the entry points

    def step(
        self,
        obs: torch.Tensor,
        roles: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
        *,
        with_aux: bool = False,
        comm_mask: torch.Tensor | None = None,
        msg_obs: torch.Tensor | None = None,
        msg_mask: torch.Tensor | None = None,
        msg_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """``(loc, scale, aux, h)``: the Gaussian over the pre-tanh action, ``[*, A,
        act_dim]`` each, the auxiliary prediction ``[*, A, aux_dim]`` when ``with_aux``
        (else ``None``), and the hidden state: the GRU's output when ``recurrent`` (from
        zeros when ``h`` is ``None``), else ``h`` passed through. ``msg_*`` are the
        delayed messages' inputs (see the class docstring); without them nothing is
        heard."""
        roles = self._check(roles, with_aux)
        route = self._route(roles)
        feat = self._pre(obs, roles, route, comm_mask, (msg_obs, msg_mask, msg_noise))
        if self.recurrent:
            if h is None:
                h = obs.new_zeros(*obs.shape[:-1], self.hidden_dim)
            feat, h = self._core(self._input_gates(feat), h, route)
        loc, scale, aux = self._post(feat, roles, route, with_aux)
        return loc, scale, aux, h

    def unroll(
        self,
        obs: torch.Tensor,
        roles: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
        done: torch.Tensor | None = None,
        *,
        with_aux: bool = False,
        comm_mask: torch.Tensor | None = None,
        msg_obs: torch.Tensor | None = None,
        msg_mask: torch.Tensor | None = None,
        msg_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """:meth:`step` over a sequence: ``obs`` ``[L, *, A, obs_dim]`` (``roles`` and
        ``comm_mask`` with the same leading ``[L, *]``), from the hidden state ``h``
        ``[*, A, hidden_dim]`` entering step 0.

        ``done[l]`` ``[*]`` (``{0, 1}``) zeroes the hidden state after step ``l``, so step
        ``l + 1`` starts a new episode from zeros, as the rollout does. Everything that
        does not read ``h``, the GRU's input projection included, runs once over the whole
        sequence; only the hidden-to-hidden half of the GRU steps one at a time. Returns
        ``[L, ...]`` outputs and the hidden state after step ``L - 1`` (not reset).
        """
        msg = (msg_obs, msg_mask, msg_noise)
        if not self.recurrent:
            return self.step(obs, roles, h, with_aux=with_aux, comm_mask=comm_mask,
                             msg_obs=msg_obs, msg_mask=msg_mask, msg_noise=msg_noise)
        roles = self._check(roles, with_aux)
        route = self._route(roles)
        gi = self._input_gates(self._pre(obs, roles, route, comm_mask, msg))
        if h is None:
            h = obs.new_zeros(*obs.shape[1:-1], self.hidden_dim)
        keep = None if done is None else (1.0 - done.to(obs.dtype))[..., None, None].unbind(0)
        # unbind, not gi[i]: the backward of L selects would build L full-size gradients.
        if self.separate:
            steps = list(zip(*(g.unbind(0) for g in gi), strict=True))
            routes = list(zip(route[0].unbind(0), route[1].unbind(0), strict=True))
        else:
            steps, routes = gi.unbind(0), [None] * obs.shape[0]
        outs = []
        for i, (step_feat, step_route) in enumerate(zip(steps, routes, strict=True)):
            if i and keep is not None:
                h = h * keep[i - 1]
            f, h = self._core(list(step_feat) if self.separate else step_feat, h, step_route)
            outs.append(f)
        if self.separate:
            core = [torch.stack([o[r] for o in outs]) for r in range(len(self._spans))]
        else:
            core = torch.stack(outs)
        loc, scale, aux = self._post(core, roles, route, with_aux)
        return loc, scale, aux, h

    def comm_weights(
        self,
        obs: torch.Tensor,
        roles: torch.Tensor | None = None,
        comm_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The attention weights ``[*, A, A]`` (row ``i``: what agent ``i`` hears from each
        teammate), for inspection. Zero on the diagonal and wherever ``comm_mask`` is
        ``False``."""
        if self.comm != "attn":
            raise ValueError("only comm='attn' has attention weights")
        roles = self._check(roles, False)
        return self._communicate(self._with_role(obs, roles), comm_mask)[1]

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
    def greedy(
        self,
        obs: torch.Tensor,
        roles: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
        comm_mask: torch.Tensor | None = None,
        msg_obs: torch.Tensor | None = None,
        msg_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The deterministic action ``squash(loc)``: consumes no RNG (a ``"vae"`` message
        is its mean). A recurrent net acts from ``h`` (zeros when ``None``) and the new
        hidden state is dropped; carry it, and the previous observation a delayed message
        is encoded from, with :meth:`step`, as :func:`swarp.rl.ppo.greedy_env_arm` does."""
        out = self.step(obs, roles, h, comm_mask=comm_mask, msg_obs=msg_obs, msg_mask=msg_mask)
        return self.squash(out[0])


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

    ``state`` is read as it comes, never reordered: with roles, slot ``k`` of the output is
    the agent holding role-sorted rank ``k``, so a per-agent part of ``critic_state()``
    must already be in role-sorted agent order (role 0's agents by ascending index, then
    role 1's, ...), the contract in ``docs/writing-a-scenario.md``. No role one-hot is
    added; the order is the role information.
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
