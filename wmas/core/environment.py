"""VMAS-style vectorized Environment over a Scenario."""

from __future__ import annotations

from typing import Any

import torch

from wmas.scenarios.base import Scenario


class Environment:
    """Vectorized multi-agent environment: all tensors stay on ``device``.

    ``step`` is differentiable end-to-end (dynamics via the Warp adjoint,
    observations/rewards via torch autograd) when called with grads enabled;
    under ``torch.no_grad()`` it runs the tape-free hot path with no per-step
    host<->device transfers.

    By default there is no auto-reset (like raw VMAS): inspect ``done`` and call
    :meth:`reset` or :meth:`reset_at` when you want fresh episodes. Set
    ``auto_reset=True`` to have :meth:`step` reset done envs in-place via a
    host-sync-free masked path (no ``.any()``/``.nonzero()`` round-trip), so the
    whole loop can stay on device.
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
        use_graph: bool = False,
        copy_outputs: bool = False,
        fused: bool | str = "auto",
    ) -> None:
        self.scenario = scenario
        self.n_envs = n_envs
        self.device = device
        self.dtype = dtype
        self.max_steps = max_steps
        self.auto_reset = auto_reset
        self.copy_outputs = copy_outputs
        self.world = scenario.make_world(
            n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        self.n_agents = self.world.n_agents
        self._seed(seed)
        self._step_count = torch.zeros(n_envs, device=device, dtype=torch.int32)
        # Fused Warp obs/reward/done kernels for the no-grad hot path. "auto"
        # follows the scenario; grad mode always falls back to the torch path.
        self._fused = scenario.fused_available() if fused == "auto" else bool(fused)
        # Opt-in persistent-buffer + CUDA-graph execution for the no-grad hot
        # path. Zero-copy views are returned by default; copy_outputs clones them.
        if use_graph:
            self.world.enable_persistent(use_graph=True)
        # Fold the fused obs/reward launches into the whole-step CUDA graph when
        # the scenario is fused + capture-safe and a persistent runtime exists.
        self._whole_step = False
        self._maybe_wire_whole_step_graph()

    def _maybe_wire_whole_step_graph(self) -> None:
        runtime = self.world.runtime
        if self._fused and runtime is not None and self.scenario.graph_capturable():
            runtime.set_post_physics(
                self.scenario._graph_post_physics,
                self.scenario.graph_recapture_token,
                self.scenario._graph_warmup_carries,
            )
            self._whole_step = True

    def _set_fused_active(self) -> None:
        """Fused kernels run only on the no-grad path (grad uses the torch ref)."""
        self.scenario._fused_active = self._fused and not torch.is_grad_enabled()

    def _seed(self, seed: int) -> None:
        self.world.generator = torch.Generator(device=self.device)
        self.world.generator.manual_seed(seed)

    @property
    def graph_mode(self) -> bool:
        """True when a CUDA graph is actively backing the no-grad step."""
        return self.world.runtime is not None and self.world.runtime.graph_active

    # ------------------------------------------------------------------- API

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Reset all envs; returns stacked observations [n_envs, n_agents, obs_dim]."""
        if seed is not None:
            self._seed(seed)
        self._set_fused_active()
        self.scenario._fused_obs_only = False
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
        """
        self._set_fused_active()
        self.scenario._fused_obs_only = False
        with torch.no_grad():
            self.scenario.reset_world(env_mask)
        self._step_count = torch.where(
            env_mask, torch.zeros_like(self._step_count), self._step_count
        )
        return self.scenario.observations()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Advance every env by one step.

        Args:
            actions: ``[n_envs, n_agents, act_dim]`` tensor on the env device
                (``act_dim`` = ``world.act_dim``, the max action arity over agent
                models; 2 for the current 2D vehicle models).

        Returns:
            ``(obs [n_envs, n_agents, obs_dim], reward [n_envs, n_agents],
            done [n_envs] bool, info dict)`` — all on the env device.
        """
        act_dim = self.world.act_dim
        if actions.shape != (self.n_envs, self.n_agents, act_dim):
            raise ValueError(
                f"actions must have shape {(self.n_envs, self.n_agents, act_dim)}, "
                f"got {tuple(actions.shape)}"
            )
        if actions.dtype != self.dtype:
            raise TypeError(f"actions dtype {actions.dtype} != env dtype {self.dtype}")

        self._set_fused_active()
        self.scenario._fused_obs_only = False
        # Whole-step graph: run capture-unsafe prep (buffer ensure, resetmask
        # zero, handle sync) eagerly on the default stream before the replay.
        if self._whole_step and self.scenario._fused_active:
            self.scenario._pre_graph_step()
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
        done = self.scenario.done()
        if self.max_steps is not None:
            done = done | (self._step_count >= self.max_steps)
        info = self.scenario.info()

        if self.auto_reset:
            # Obs-only reset pass: reward/done/info were already returned for this
            # transition and their (fused) buffers must not be clobbered.
            self.scenario._fused_obs_only = True
            with torch.no_grad():
                self.scenario.reset_world(done)
            self.scenario._fused_obs_only = False
            self._step_count = torch.where(
                done, torch.zeros_like(self._step_count), self._step_count
            )

        # Observations reflect the state after any auto-reset (next episode's
        # first obs for done envs), matching the gym/VMAS vec-env convention.
        obs = self.scenario.observations()
        if self.copy_outputs:
            obs, reward = obs.clone(), reward.clone()
            done = done.clone()
        return obs, reward, done, info

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
        """VMAS-style convenience over :class:`wmas.render.Viewer` (needs the ``viz`` extra).

        ``mode="rgb_array"`` returns an ``(H, W, 3)`` uint8 frame of env ``env_index``;
        ``mode="human"`` updates a persistent window and returns ``None``. Extra keyword
        args (``size``, ``overlays``, ``mosaic``, ...) are forwarded to the Viewer, which is
        created once and reused. Import is lazy so the core has no hard pygame dependency.

        This only *draws*: the caller owns the stepping, so the viewer's interactive
        pause and single-step controls cannot take effect (they gate
        :meth:`wmas.render.viewer.Viewer.run`'s own loop). A caller-driven loop that
        keeps calling ``step`` will keep moving while the HUD reads "paused". For an
        interactive window, hand the loop over instead::

            Viewer(env, fps=20).run(action_fn=policy_fn)
        """
        if mode not in ("human", "rgb_array"):
            raise ValueError(f"render mode must be 'human' or 'rgb_array', got {mode!r}")
        from wmas.render.viewer import Viewer

        if getattr(self, "_viewer", None) is None:
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
        viewer = getattr(self, "_viewer", None)
        if viewer is not None:
            viewer.close()
            self._viewer = None
