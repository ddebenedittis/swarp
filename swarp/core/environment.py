"""VMAS-style vectorized Environment over a Scenario."""

from __future__ import annotations

from typing import Any

import torch

from swarp.core.config import WorldConfig
from swarp.dynamics.base import action_bounds
from swarp.scenarios.base import Scenario


class Environment:
    """Vectorized multi-agent environment: all tensors stay on ``device``.

    ``step`` is differentiable end-to-end (dynamics via the Warp adjoint,
    observations/rewards via torch autograd) when called with grads enabled;
    under ``torch.no_grad()`` it runs the tape-free hot path with no per-step
    host<->device transfers.

    :meth:`step` returns the Gymnasium 5-tuple
    ``(obs, reward, terminated, truncated, info)``: ``terminated`` is the scenario's
    own terminal condition, ``truncated`` the ``max_steps`` time limit. Keeping them
    apart is what lets a value estimator bootstrap through a timeout instead of
    treating it as a real terminal state.

    By default there is no auto-reset (like raw VMAS): inspect
    ``terminated | truncated`` and call :meth:`reset` or :meth:`reset_at` when you want
    fresh episodes. Set ``auto_reset=True`` to have :meth:`step` reset those envs
    in-place via a host-sync-free masked path (no ``.any()``/``.nonzero()``
    round-trip), so the whole loop can stay on device.

    Two consequences of ``auto_reset=True`` worth knowing before you rely on it:

    - **There is no terminal observation.** Gym's ``final_observation`` has no
      counterpart here: the ``obs`` returned alongside a ``True`` ``terminated`` /
      ``truncated`` is already the **new** episode's first observation, not the state the
      episode ended in. An algorithm that needs the terminal state (n-step returns
      bootstrapping off it, for instance) must run with ``auto_reset=False`` and reset
      itself, or capture the state before the step that ends the episode.
    - **The reset runs every step.** Because the mask is applied device-side with no
      host-side "is anything done?" gate, the scenario's ``reset_world`` executes on every
      step over the whole batch, whether or not any env is done — see
      "The cost of ``auto_reset``" in ``docs/performance.md``.

    Args:
        use_graph: persistent-buffer execution backed by a whole-step CUDA graph.
            ``"auto"`` (the default) enables it whenever it can actually pay off — a
            CUDA device and a scenario that provides a capturable whole-step hook —
            which is the configuration ``docs/benchmarks.md`` measures at 2.5-5x.
            Pass ``False`` to force the plain functional step, ``True`` to demand
            persistent execution even where capture is unavailable (it then falls back
            to eager persistent execution **with a warning**, since you asked for
            something you did not get). ``"auto"`` never warns: capture also needs the
            ``brute``/``uniform_grid`` neighbor backend, which ``"auto"`` does not check,
            and eager persistent execution is still faster than the functional step — so
            the fallback there is the intended outcome, not a degradation.
            :attr:`graph_mode` reports what is actually in play.
        clone_outputs: clone ``obs``/``reward``/``terminated``/``truncated`` and the
            ``info`` values before returning them. Off by default, so :meth:`step` hands
            back **zero-copy views of buffers the next step overwrites** — fine for a
            policy that consumes them immediately, wrong for anything that retains them
            (a replay buffer, a trajectory list). Turn this on, or clone at the call site.
        fused: use the scenario's fused Warp obs/reward kernels on the no-grad path.
            ``"auto"`` follows :attr:`~swarp.scenarios.base.Scenario.fused_available`;
            grad mode always falls back to the differentiable torch path.
        world_config: engine-level overrides applied on top of whatever the scenario
            computes, so settings no scenario exposes as a constructor argument
            (``integrator``, ``bounds_mode``, ``neighbor_reuse``, ``neighbor_method``,
            ``grid_dim``, ``uniform_bins``, the obstacle damping, ``contact_max_overlap``)
            are reachable without subclassing. Only the fields set away from the
            ``WorldConfig()`` defaults are taken — the scenario keeps its computed
            ``bounds`` and ``neighbor_radius``. See
            :meth:`~swarp.core.config.WorldConfig.override_with`.
    """

    def __init__(
        self,
        scenario: Scenario,
        n_envs: int,
        device: str = "cuda:0",
        dt: float = 0.1,
        substeps: int = 1,
        dtype: torch.dtype = torch.float32,
        max_steps: int | None = None,
        seed: int = 0,
        auto_reset: bool = False,
        use_graph: bool | str = "auto",
        clone_outputs: bool = False,
        fused: bool | str = "auto",
        world_config: WorldConfig | None = None,
    ) -> None:
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        self.scenario = scenario
        self.n_envs = n_envs
        # Normalized to a string: it is consumed as one (``.startswith("cuda")`` below,
        # Warp device names downstream), so a ``torch.device`` has to be accepted here.
        self.device = str(device)
        device = self.device
        self.dtype = dtype
        self.max_steps = max_steps
        self.auto_reset = auto_reset
        self.clone_outputs = clone_outputs
        self._viewer: Any = None
        self.world = scenario.make_world(
            n_envs=n_envs,
            device=device,
            dt=dt,
            substeps=substeps,
            dtype=dtype,
            world_config=world_config,
        )
        self.n_agents = self.world.n_agents
        self._seed(seed)
        self._step_count = torch.zeros(n_envs, device=device, dtype=torch.int32)
        # Returned as ``truncated`` when there is no time limit. Allocated once: the hot
        # path must not allocate a fresh all-false tensor on every step.
        self._never_truncated = torch.zeros(n_envs, device=device, dtype=torch.bool)
        # Fused Warp obs/reward/done kernels for the no-grad hot path. "auto"
        # follows the scenario; grad mode always falls back to the torch path.
        self._fused = scenario.fused_available if fused == "auto" else bool(fused)
        # The whole-step hook (fused obs/reward folded into the graph), or None when
        # the scenario has none / fused is off. It also decides the "auto" default for
        # use_graph: persistent execution is only the *advertised* fast path when the
        # obs/reward launches can ride along inside the same graph.
        self._hook = scenario.graph_hook() if self._fused else None
        # Whether the *caller* demanded capture, as opposed to us inferring it. Only an
        # explicit request warns when capture turns out to be unavailable: "auto" also
        # gates on a whole-step hook and a CUDA device, but not on the neighbor backend,
        # so warning here would scold a user for a default they never chose — and the
        # eager persistent fallback is still faster than the functional step.
        graph_requested = use_graph is True
        if use_graph == "auto":
            use_graph = self._hook is not None and device.startswith("cuda")
        if use_graph:
            self.world.enable_persistent(use_graph=True, graph_requested=graph_requested)
        runtime = self.world.runtime
        self._whole_step = self._hook is not None and runtime is not None
        if self._whole_step:
            runtime.set_post_physics(self._hook)
        else:
            self._hook = None

    def _taped_step(self, actions: torch.Tensor | None = None) -> bool:
        """Whether the coming step will be recorded on the Warp tape.

        This is the same predicate :meth:`swarp.core.world.World.step` uses to choose
        between the taped functional path and the no-grad hot path, and the two have to
        agree: the fused obs/reward launches ride on the no-grad path, and once they are
        baked into a whole-step CUDA graph that graph cannot be replayed "half" — the
        physics and the fused kernels are one capture. Merely being inside
        ``enable_grad`` is not enough to tape a step; something has to require grad.
        """
        if not torch.is_grad_enabled():
            return False
        if actions is not None and actions.requires_grad:
            return True
        return any(t is not None and t.requires_grad for t in self.world.state)

    def _set_fused_active(self, actions: torch.Tensor | None = None) -> None:
        """Fused kernels run only on the untaped path (a taped step uses the torch ref)."""
        self.scenario.set_fused_active(self._fused and not self._taped_step(actions))

    def seed(self, seed: int) -> None:
        """Reseed the world RNG **without** touching episode state.

        ``reset(seed=...)`` also reseeds, but restarts every episode as a side effect.
        This is the one to call when only the RNG stream should move — TorchRL's
        ``_set_seed`` contract, which collectors invoke during worker setup.
        """
        self._seed(seed)

    def _seed(self, seed: int) -> None:
        # Called from ``__init__`` before ``_step_count`` exists: must stay
        # RNG-only, never reach for episode state.
        self.world.generator = torch.Generator(device=self.device)
        self.world.generator.manual_seed(seed)

    @property
    def graph_mode(self) -> bool:
        """True when a CUDA graph is actively backing the no-grad step."""
        return self.world.runtime is not None and self.world.runtime.graph_active

    @property
    def obs_dim(self) -> int:
        """Per-agent observation width, forwarded from the scenario."""
        return self.scenario.obs_dim

    @property
    def act_dim(self) -> int:
        """Env-level action width (max arity over the agent models)."""
        return self.world.act_dim

    @property
    def action_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The action box the dynamics kernels clamp against: ``(low, high)``, each
        ``[n_agents, act_dim]`` on this env's device and dtype.

        Actions are **physical**, never normalized — the kernels clamp to the per-agent
        limits in :class:`~swarp.dynamics.base.AgentConfig`, so a fleet with
        ``max_speed=3.0`` accepts ``[-3, 3]`` and a quadrotor accepts ``[0, thrust_max]``
        per rotor. Feed this to an RL wrapper rather than assuming ``[-1, 1]``; see
        :class:`swarp.interop.torchrl.SwarpEnv`.

        Slots past an agent's model arity are reported as ``[0, 0]``: they exist only
        because ``act_dim`` is the max over a mixed fleet, and the agent's branch ignores
        them. A zero-width slot is what tells a policy not to spend capacity there.
        Rows are per *agent*, not per env: per-env randomized limits
        (:meth:`~swarp.core.stepper.Stepper.set_agent_params_per_env`) are not reflected.
        """
        na, ad = self.n_agents, self.world.act_dim
        low = torch.zeros(na, ad, device=self.device, dtype=self.dtype)
        high = torch.zeros_like(low)
        for i, cfg in enumerate(self.world.agent_configs):
            lo, hi = action_bounds(cfg)
            low[i, : len(lo)] = torch.tensor(lo, device=self.device, dtype=self.dtype)
            high[i, : len(hi)] = torch.tensor(hi, device=self.device, dtype=self.dtype)
        return low, high

    # ------------------------------------------------------------------- API

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Reset all envs; returns stacked observations [n_envs, n_agents, obs_dim]."""
        if seed is not None:
            self._seed(seed)
        self._set_fused_active()
        self.world.action = None  # no action applied yet this episode
        with torch.no_grad():
            self.world.reset_state()
            self.scenario.reset_world(None)
        self._step_count.zero_()
        return self.scenario.observations()

    def reset_at(self, env_mask: torch.Tensor) -> torch.Tensor:
        """Reset the envs where ``env_mask`` (bool ``[n_envs]``) is True.

        Host-sync-free: the scenario samples the full batch and blends the
        selected envs with ``torch.where``; unselected envs are untouched.
        Returns stacked observations for all envs.

        Unlike :meth:`reset` this does **not** clear ``world.action``, and cannot: the
        action tensor is ``[n_envs, n_agents, act_dim]`` but it is one *shared* handle
        rather than per-env state, so there is nothing to mask. It does not matter —
        :meth:`step` overwrites ``world.action`` before any reward path reads it, so a
        reset env never sees the pre-reset action.
        """
        self._set_fused_active()
        with torch.no_grad():
            self.scenario.reset_world(env_mask)
        self._step_count = torch.where(
            env_mask, torch.zeros_like(self._step_count), self._step_count
        )
        return self.scenario.observations()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Advance every env by one step.

        Args:
            actions: ``[n_envs, n_agents, act_dim]`` tensor on the env device
                (``act_dim`` = ``world.act_dim``, the max action arity over agent
                models; 2 for the current 2D vehicle models).

        Returns:
            The Gymnasium 5-tuple ``(obs [n_envs, n_agents, obs_dim],
            reward [n_envs, n_agents], terminated [n_envs] bool,
            truncated [n_envs] bool, info dict)`` — all on the env device.

            ``terminated`` is the scenario's own terminal condition
            (:meth:`~swarp.scenarios.base.Scenario.done`); ``truncated`` is the
            ``max_steps`` time limit, and is an all-false view when ``max_steps is
            None``. Episode end — for auto-reset and for the step counter — is
            ``terminated | truncated``.
        """
        act_dim = self.world.act_dim
        if actions.shape != (self.n_envs, self.n_agents, act_dim):
            raise ValueError(
                f"actions must have shape {(self.n_envs, self.n_agents, act_dim)}, "
                f"got {tuple(actions.shape)}"
            )
        if actions.dtype != self.dtype:
            raise TypeError(f"actions dtype {actions.dtype} != env dtype {self.dtype}")

        self._set_fused_active(actions)
        # Whole-step graph: run the hook's capture-unsafe prep (buffer ensure, reset-mask
        # zero, handle sync) eagerly on the default stream before the replay.
        if self._whole_step and self.scenario.fused_active:
            self._hook.prepare()
        # Expose the applied action to the scenario reward path (post_step reads
        # world.action for control-input shaping, e.g. action-smoothness).
        self.world.action = actions
        self.world.step(actions)
        # The graph (or the CPU eager hook) already filled the obs/reward buffers;
        # skip the redundant torch post_step. Grad steps still take the torch path.
        if not self.world.ran_post_physics:
            self.scenario.post_step()

        # Reward/done/info describe the transition just taken (terminal state).
        reward = self.scenario.rewards()
        self._step_count += 1
        terminated = self.scenario.done()
        truncated = (
            self._step_count >= self.max_steps
            if self.max_steps is not None
            else self._never_truncated
        )
        info = self.scenario.info()

        if self.auto_reset:
            episode_end = terminated | truncated
            # Obs-only reset pass: reward/done/info were already returned for this
            # transition and their (fused) buffers must not be clobbered.
            with torch.no_grad():
                self.scenario.reset_world(episode_end, obs_only=True)
            self._step_count = torch.where(
                episode_end, torch.zeros_like(self._step_count), self._step_count
            )

        # Observations reflect the state after any auto-reset (next episode's
        # first obs for done envs), matching the gym/VMAS vec-env convention.
        obs = self.scenario.observations()
        if self.clone_outputs:
            obs, reward = obs.clone(), reward.clone()
            terminated, truncated = terminated.clone(), truncated.clone()
            # ``info`` values are views into the scenario's fused buffers, so they alias
            # just like obs/reward do — the flag has to cover them to mean anything.
            info = {k: v.clone() if torch.is_tensor(v) else v for k, v in info.items()}
        return obs, reward, terminated, truncated, info

    def radius_graph(self) -> torch.Tensor:
        """COO edge index [2, E] of the current within-radius neighbor graph.

        Reuses the neighbor grid that ``step``/``reset`` already built on the
        current state (via the scenario's cache refresh), so no rebuild is
        needed — just the one sync to materialize E.
        """
        return self.world.edge_index(rebuild=False)

    # ------------------------------------------------------------- rendering

    def render(
        self,
        mode: str = "human",
        env_index: int = 0,
        agent_index_focus: int | None = None,
        visualize_when_rgb: bool = False,  # accepted for VMAS-signature compatibility
        **viewer_kwargs: Any,
    ) -> Any:
        """VMAS-style convenience over :class:`swarp.render.Viewer` (needs the ``viz`` extra).

        ``mode="rgb_array"`` returns an ``(H, W, 3)`` uint8 frame of env ``env_index``;
        ``mode="human"`` updates a persistent window and returns ``None``. Extra keyword
        args (``size``, ``overlays``, ``mosaic``, ...) are forwarded to the Viewer, which is
        created once and reused. Import is lazy so the core has no hard pygame dependency.

        This only *draws*: the caller owns the stepping, so the viewer's interactive
        pause and single-step controls cannot take effect (they gate
        :meth:`swarp.render.viewer.Viewer.run`'s own loop). A caller-driven loop that
        keeps calling ``step`` will keep moving while the HUD reads "paused". For an
        interactive window, hand the loop over instead::

            Viewer(env, fps=20).run(action_fn=policy_fn)
        """
        if mode not in ("human", "rgb_array"):
            raise ValueError(f"render mode must be 'human' or 'rgb_array', got {mode!r}")
        from swarp.render.viewer import Viewer

        if self._viewer is None:
            self._viewer = Viewer(self, env_index=env_index, **viewer_kwargs)
        viewer = self._viewer
        viewer.state.focus_env = env_index
        if agent_index_focus is not None:
            viewer.state.hover_agent = agent_index_focus
        if mode == "rgb_array":
            return viewer.render_array(hud=False)
        return viewer.render_human_frame()

    def close_viewer(self) -> None:
        """Close the persistent render window, if one was opened by ``render``."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ---------------------------------------------------------------- teardown

    def close(self) -> None:
        """Release what this env holds outside its own tensors, and stay usable.

        Two things: the render window (as :meth:`close_viewer`) and the persistent
        runtime's captured CUDA graph, whose device-side resources are otherwise pinned
        for as long as the env is alive. Call it when a script keeps many environments
        around, or when handing the GPU to something else.

        Idempotent, and **not** a destructor: the persistent buffers, the state and the
        scenario are untouched, so stepping after ``close()`` just recaptures the graph on
        the next step — same as any recapture the token already triggers. Nothing needs
        reconstructing.
        """
        self.close_viewer()
        if self.world.runtime is not None:
            self.world.runtime.release()
