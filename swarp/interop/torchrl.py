"""TorchRL ``EnvBase`` wrapper around a swarp :class:`~swarp.core.environment.Environment`.

Exposes a batched (``batch_size=[n_envs]``), GPU-resident environment with
``TensorDict`` specs for observation/action/reward/done, so swarp scenarios can be
driven by TorchRL collectors and policies. Everything stays on-device; the swarp
step already returns stacked device tensors, so ``_step``/``_reset`` are thin.

Requires the optional ``torchrl`` extra (``uv pip install -e '.[torchrl]'``).
Import this module only when you need the wrapper — the core package does not
depend on torchrl.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

from swarp.core.environment import Environment


class SwarpEnv(EnvBase):
    """A swarp ``Environment`` as a batched TorchRL ``EnvBase``.

    Observations are keyed ``"observation"`` ``[n_envs, n_agents, obs_dim]``;
    actions ``"action"`` ``[n_envs, n_agents, act_dim]``; reward
    ``[n_envs, n_agents, 1]``; shared per-env ``done``/``terminated``/``truncated``
    ``[n_envs, 1]``.

    Action bounds
    -------------
    swarp actions are **physical, not normalized**: the kernels clamp them to the
    per-agent limits in :class:`~swarp.dynamics.base.AgentConfig`, and nothing rescales
    them on the way in. The ``[-1, 1]`` default here is therefore correct **only** for a
    holonomic fleet with ``max_speed == 1.0`` under
    :attr:`~swarp.dynamics.base.ControlMode.VELOCITY` — which is what every built-in
    scenario except ``navigation(model=...)`` happens to use. Outside that case the
    default silently mis-specifies the action space:

    - ``max_speed=3.0`` throttles the policy to a third of the achievable velocity;
    - ``ControlMode.ACCELERATION`` bounds against ``max_accel``, which the built-ins set
      to ``2 * max_speed``;
    - the kinematic bicycle's second slot is a steering *angle* (``±max_steer``,
      default ``pi/4``), not a speed;
    - a quadrotor's four slots are per-rotor thrusts in ``[0, thrust_max]``, so under
      ``[-1, 1]`` **the drone cannot reach hover** (``mass * gravity`` needs ~2.45 N per
      rotor against a declared cap of 1.0) and the whole negative half of the box maps
      to identical, zero-gradient dynamics.

    Pass :attr:`~swarp.core.environment.Environment.action_bounds` to get the box the
    kernels actually enforce::

        low, high = env.action_bounds          # [n_agents, act_dim] each
        tenv = SwarpEnv(env, action_low=low, action_high=high)

    The bounds accept a scalar (broadcast over every agent and slot) or any tensor
    broadcastable to ``[n_envs, n_agents, act_dim]`` — a single scalar pair cannot
    express the bicycle's or the drone's box, which is why per-slot tensors are allowed.
    The default stays ``[-1, 1]`` for backward compatibility; it is not the physically
    correct answer, only the historical one.

    Termination
    -----------
    ``terminated`` is the scenario's own terminal condition; ``truncated`` is the
    ``max_steps`` time limit; ``done`` is their OR, which is what TorchRL's resetters
    key off. Because the time limit no longer masquerades as a terminal state, GAE and
    the other value estimators bootstrap correctly through a timeout instead of
    truncating the return at every episode boundary — a scenario with a ``max_steps``
    no longer needs to reconstruct its real terminal condition from ``info``.

    Scenario ``info()``
    -------------------
    If the scenario emits a non-empty :meth:`~swarp.scenarios.base.Scenario.info`
    dict, each key is spec'd and forwarded as a nested ``info`` ``Composite``
    inside ``observation_spec``, so it appears in both the reset output and the
    step output under the **flat** path ``("next", "info", <key>)``. This is how
    a multi-objective reward vector (e.g. ``multiobj_reward``
    ``[n_envs, n_agents, n_obj]``) reaches a downstream lexicographic-MAPPO loop
    while the ``reward`` key itself stays scalar (``[n_envs, n_agents, 1]``).

    Note the layout: ``SwarpEnv`` is flat (no ``("agents", …)`` group), so info is
    at ``("next", "info", <key>)`` — **not** group-nested like TorchRL's
    ``VmasEnv`` (``("next", "<group>", "info", <key>)``). Downstream code that
    expects the grouped path must use the flat path here.

    The info schema (per-key shape/dtype) is discovered at construction from
    ``scenario.info()`` read **after the probe reset**, without advancing the
    dynamics — so a scenario must populate its ``info()`` by reset time to have
    it spec'd. Every info tensor must have leading dim ``n_envs``. Scenarios with
    empty info (``{}``) get no info spec and behave exactly as before.
    """

    def __init__(
        self,
        env: Environment,
        action_low: float | torch.Tensor = -1.0,
        action_high: float | torch.Tensor = 1.0,
    ) -> None:
        super().__init__(device=env.device, batch_size=torch.Size([env.n_envs]))
        self._env = env
        self.n_agents = env.n_agents
        self.act_dim = env.world.act_dim
        obs = env.reset()  # one reset to discover obs_dim
        self.obs_dim = obs.shape[-1]
        # Sample the info schema without advancing dynamics (the scenario populates
        # info() at reset time via reset_world -> post_step's cache refresh).
        info = self._env.scenario.info()
        self._info_keys = list(info.keys())
        self._make_specs(action_low, action_high, obs.dtype, info)

    def _action_bound(self, val: float | torch.Tensor, dtype: torch.dtype):
        """Normalize one action bound to what ``Bounded`` wants.

        A scalar is passed straight through (so the ``[-1, 1]`` default produces exactly
        the spec it always did); anything else is broadcast to the full
        ``[n_envs, n_agents, act_dim]`` action shape, which is the only form that can
        carry a per-slot box (bicycle: accel + steering angle; drone: four one-sided
        rotor thrusts).
        """
        shape = (self._env.n_envs, self.n_agents, self.act_dim)
        if not isinstance(val, torch.Tensor):
            return float(val)
        t = val.to(device=self.device, dtype=dtype)
        if t.ndim == 0:
            return float(t)
        try:
            return t.expand(shape).contiguous()
        except RuntimeError as exc:
            raise ValueError(
                f"action bound of shape {tuple(t.shape)} is not broadcastable to the "
                f"action shape {shape}. Environment.action_bounds returns "
                f"[n_agents, act_dim] tensors, which are."
            ) from exc

    def _make_specs(
        self,
        low: float | torch.Tensor,
        high: float | torch.Tensor,
        dtype: torch.dtype,
        info: dict[str, torch.Tensor],
    ) -> None:
        ne, na = self._env.n_envs, self.n_agents
        low, high = self._action_bound(low, dtype), self._action_bound(high, dtype)
        self.observation_spec = Composite(
            observation=Unbounded(shape=(ne, na, self.obs_dim), dtype=dtype, device=self.device),
            shape=(ne,),
            device=self.device,
        )
        if info:
            info_spec = Composite(shape=(ne,), device=self.device)
            for key, val in info.items():
                if val.shape[0] != ne:
                    raise ValueError(
                        f"info[{key!r}] must have leading dim n_envs={ne}, "
                        f"got shape {tuple(val.shape)}"
                    )
                info_spec[key] = Unbounded(
                    shape=tuple(val.shape), dtype=val.dtype, device=self.device
                )
            # Nest under observation_spec so it surfaces in both reset and ("next", ...).
            self.observation_spec["info"] = info_spec
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
        # A single shared done/terminated/truncated per env.
        self.done_spec = Composite(
            done=Unbounded(shape=(ne, 1), dtype=torch.bool, device=self.device),
            terminated=Unbounded(shape=(ne, 1), dtype=torch.bool, device=self.device),
            truncated=Unbounded(shape=(ne, 1), dtype=torch.bool, device=self.device),
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
        obs = obs.clone()  # fused/graph obs is a view into a reused buffer
        ne = self._env.n_envs
        out = TensorDict(
            {
                "observation": obs,
                "done": torch.zeros(ne, 1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(ne, 1, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(ne, 1, dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        if self._info_keys:
            # The scenario populated info() during reset_world's cache refresh; forward it
            # so reset and step satisfy the same info spec. Tensors stay on-device.
            info = self._env.scenario.info()
            out.set("info", self._info_td(info))
        return out

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict.get("action")
        obs, reward, terminated, truncated, info = self._env.step(action)
        # Fused / graph-mode outputs are zero-copy views into persistent buffers
        # overwritten next step; collectors hold refs across steps, so clone.
        obs, reward = obs.clone(), reward.clone()
        terminated = terminated.clone().reshape(-1, 1)
        truncated = truncated.clone().reshape(-1, 1)
        out = TensorDict(
            {
                "observation": obs,
                "reward": reward.unsqueeze(-1),
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        if self._info_keys:
            out.set("info", self._info_td(info))
        return out

    def _info_td(self, info: dict[str, torch.Tensor]) -> TensorDict:
        """Pack the scenario info dict into a nested TensorDict (tensors kept on-device)."""
        return TensorDict(
            {key: info[key].clone() for key in self._info_keys},
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        if seed is not None:
            self._env.reset(seed=seed)
