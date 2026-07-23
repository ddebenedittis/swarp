"""TorchRL ``EnvBase`` wrapper around a wmas :class:`~wmas.core.environment.Environment`.

Exposes a batched (``batch_size=[n_envs]``), GPU-resident environment with
``TensorDict`` specs for observation/action/reward/done, so wmas scenarios can be
driven by TorchRL collectors and policies. Everything stays on-device; the wmas
step already returns stacked device tensors, so ``_step``/``_reset`` are thin.

Requires the optional ``torchrl`` dependency group (``uv pip install -e .
--group torchrl``). Import this module only when you need the wrapper — the core
package does not depend on torchrl.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

from wmas.core.environment import Environment


class WmasEnv(EnvBase):
    """A wmas ``Environment`` as a batched TorchRL ``EnvBase``.

    Observations are keyed ``"observation"`` ``[n_envs, n_agents, obs_dim]``;
    actions ``"action"`` ``[n_envs, n_agents, act_dim]`` (bounded to the
    normalized ``[-1, 1]`` range the wmas kernels clamp against); reward
    ``[n_envs, n_agents, 1]``; a shared per-env ``done`` ``[n_envs, 1]``.
    """

    def __init__(
        self, env: Environment, action_low: float = -1.0, action_high: float = 1.0
    ) -> None:
        super().__init__(device=env.device, batch_size=torch.Size([env.n_envs]))
        self._env = env
        self.n_agents = env.n_agents
        self.act_dim = env.world.act_dim
        obs = env.reset()  # one reset to discover obs_dim
        self.obs_dim = obs.shape[-1]
        self._make_specs(action_low, action_high, obs.dtype)

    def _make_specs(self, low: float, high: float, dtype: torch.dtype) -> None:
        ne, na = self._env.n_envs, self.n_agents
        self.observation_spec = Composite(
            observation=Unbounded(shape=(ne, na, self.obs_dim), dtype=dtype, device=self.device),
            shape=(ne,),
            device=self.device,
        )
        self.action_spec = Composite(
            action=Bounded(
                low=low, high=high, shape=(ne, na, self.act_dim), dtype=dtype, device=self.device
            ),
            shape=(ne,),
            device=self.device,
        )
        self.reward_spec = Composite(
            reward=Unbounded(shape=(ne, na, 1), dtype=dtype, device=self.device),
            shape=(ne,),
            device=self.device,
        )
        # A single shared done/terminated per env.
        self.done_spec = Composite(
            done=Unbounded(shape=(ne, 1), dtype=torch.bool, device=self.device),
            terminated=Unbounded(shape=(ne, 1), dtype=torch.bool, device=self.device),
            shape=(ne,),
            device=self.device,
        )

    # ------------------------------------------------------------------ EnvBase

    def _reset(self, tensordict: TensorDict | None = None, **kwargs) -> TensorDict:
        if tensordict is not None and "_reset" in tensordict.keys():  # noqa: SIM118 (TensorDict)
            mask = tensordict.get("_reset").reshape(self._env.n_envs)
            obs = self._env.reset_at(mask)
        else:
            obs = self._env.reset()
        ne = self._env.n_envs
        return TensorDict(
            {
                "observation": obs,
                "done": torch.zeros(ne, 1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(ne, 1, dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict.get("action")
        obs, reward, done, _info = self._env.step(action)
        done = done.reshape(-1, 1)
        return TensorDict(
            {
                "observation": obs,
                "reward": reward.unsqueeze(-1),
                "done": done,
                "terminated": done,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            self._env.reset(seed=seed)
