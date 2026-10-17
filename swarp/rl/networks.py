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
