"""VMAS-style vectorized Environment over a Scenario."""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from swarp.core.cached_launch import CachedLaunch, ptr_key
from swarp.core.config import WorldConfig
from swarp.core.episode_kernels import episode_end_kernel, step_count_kernel
from swarp.core.hooks import WholeStepHook
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

    For an eligible scenario (currently ``NavigationScenario`` with ``n_obstacles == 0``
    — see :meth:`~swarp.scenarios.fused.FusedScenario.supports_graph_reset`) that "every
    step" reset is not eager host work: the episode-end bookkeeping (``step_count``,
    ``truncated``, the reset mask) and the reset itself are folded into the same
    whole-step CUDA graph as the physics and fused obs/reward, so a step under
    ``auto_reset=True`` is one graph replay with **no** host-side tail at all — the
    increment/compare/mask-fill/``reset_world`` call this docstring describes above
    still happen, just as ``wp.launch``\\ es inside the capture instead of Python calling
    them each step. A scenario that samples with a ``torch.Generator`` during reset (or
    allocates) is not eligible and keeps paying the eager ``reset_world`` every step, same
    as before this was added — but not the bookkeeping around it: the step count and
    ``truncated`` ride inside the graph for *every* whole-step-hook configuration,
    ``auto_reset=False`` included.

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
            (a replay buffer, a trajectory list). ``truncated`` is a persistent buffer
            owned by the ``Environment`` itself (not the scenario's fused buffers) but is
            overwritten in place every step the same way, so it needs the same treatment.
            Turn this on, or clone at the call site.
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
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"device {device!r} requested but CUDA is not available; pass device='cpu' "
                "(everything runs on CPU) or install a CUDA-enabled torch"
            )
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
        # Persistent ``truncated``/``episode_end`` buffers: the hot path must not
        # allocate a fresh bool tensor every step. ``truncated`` is handed back to the
        # caller as a zero-copy view (like the fused obs/reward buffers) unless
        # ``clone_outputs=True``; ``episode_end`` never escapes ``step``.
        self._truncated_buf = torch.zeros(n_envs, device=device, dtype=torch.bool)
        self._episode_end_buf = torch.zeros(n_envs, device=device, dtype=torch.bool)
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
        # Whether the auto-reset tail (episode-end bookkeeping + the scenario's own
        # reset) is folded into the same hook as the physics/obs/reward pass, so that
        # under ``auto_reset=True`` a step is one graph replay with no eager host work
        # left over. Gated on ``self._whole_step`` — the condition for the hook running
        # at all, captured or eager-persistent-fallback alike — rather than on CUDA or
        # on capture having actually succeeded: the composed hook's ``run`` is correct
        # either way (see ``swarp/interop/persistent.py``'s module docstring), and
        # gating on capture would leave the eager-persistent-fallback case, which is
        # exactly as capture-safe, paying the old host tail for no reason.
        self._reset_in_graph = self._whole_step and auto_reset and scenario.supports_graph_reset()
        if self._whole_step:
            # Either composition advances ``_step_count``/``_truncated_buf`` inside the
            # hook, so ``step`` skips that host arithmetic whenever the hook ran; the
            # reset composition additionally runs the scenario's own reset.
            if self._reset_in_graph:
                self._hook = self._compose_graph_reset_hook(scenario, self._hook)
            else:
                self._hook = self._compose_step_count_hook(self._hook)
            runtime.set_post_physics(self._hook)
        else:
            self._hook = None
            self._reset_in_graph = False

    def _compose_graph_reset_hook(
        self, scenario: Scenario, base_hook: WholeStepHook
    ) -> WholeStepHook:
        """Fold the auto-reset tail into ``base_hook`` (the scenario's own step hook).

        Builds the composed :class:`~swarp.core.hooks.WholeStepHook` whose ``run`` is
        ``base_hook.run`` (physics-adjacent obs/reward) followed by the episode-end
        kernel and ``scenario.reset_in_graph()``. The launch sequence the capture then
        holds is: physics substeps -> fused step obs/reward -> episode-end kernel ->
        masked reset kernel -> masked neighbor rebuild -> masked obs-only pass.

        The episode-end kernel's own cached Warp handles (``done``/``step_count``/
        ``truncated``/``reset_mask``) are built lazily, in the composed ``prepare``, not
        here: building them needs ``scenario.ensure_fused()`` to have already allocated
        the fused buffers at the real batch size, and ``ensure_fused`` is itself lazy —
        calling it from ``__init__`` would allocate before anything the caller has done
        justifies it (a plain ``Environment(...)`` construction with no ``reset()``/
        ``step()`` yet).
        """
        w = self.world
        episode_launch = CachedLaunch()
        handles: dict[str, wp.array] = {}

        def _ensure_episode_handles() -> None:
            if handles:
                return
            scenario.ensure_fused()
            mask_name = scenario._fused_mask
            if mask_name is None:
                raise RuntimeError(
                    f"{type(scenario).__name__}.supports_graph_reset() returned True "
                    "but its fused_spec declares no reset_mask buffer -- the episode-"
                    "end kernel needs one to stamp which envs just ended"
                )
            # All four are pointer-stable for the runtime's lifetime: ``done`` and
            # ``resetmask`` are framework-owned fused buffers that ``sync_fused_handles``
            # never rebuilds (only ``watch=True`` adopted buffers move), and
            # ``_step_count``/``_truncated_buf`` are allocated once in
            # ``Environment.__init__`` and never reassigned.
            handles["done"] = scenario._wp["done"]
            handles["step_count"] = wp.from_torch(self._step_count)
            handles["truncated"] = wp.from_torch(
                self._truncated_buf.view(torch.uint8), dtype=wp.uint8
            )
            handles["reset_mask"] = scenario._wp[mask_name]

        def _prepare() -> None:
            # ``_run`` (below) is Python that only executes at warm-up and at capture
            # time, never on a graph replay -- so marking the mask dirty from inside it
            # would fire once and then never again. ``_prepare`` (this function) runs
            # eagerly before every replay instead, so it marks the flag unconditionally,
            # before delegating to ``base_hook.prepare()`` (``FusedScenario.prepare_fused``,
            # which reads the flag): a replay always restamps the mask via
            # ``episode_end_kernel`` inside ``_run``, so the buffer must be treated as
            # dirty on every step regardless of what last step's contents were. This is
            # also correct for the eager-persistent fallback, where ``_run`` does run
            # per step and would otherwise mark it redundantly.
            scenario.mark_reset_mask_dirty()
            base_hook.prepare()
            _ensure_episode_handles()

        def _run() -> None:
            base_hook.run()
            launch = episode_launch.get(
                episode_end_kernel,
                dim=self.n_envs,
                inputs=[
                    handles["done"],
                    wp.int32(self.max_steps if self.max_steps is not None else 0),
                    handles["step_count"],
                    handles["truncated"],
                    handles["reset_mask"],
                ],
                device=self.device,
                # ``max_steps`` is genuinely re-set every call (a caller may reassign
                # ``env.max_steps``): baking it into the key would repack the whole
                # launch on every step that happens to leave it unchanged, so instead
                # it lives in ``token`` below, which forces a full *recapture* -- the
                # right response, since a captured graph has already baked the old
                # limit into this launch's params by value and repacking the cached
                # ``Launch`` object does nothing to a graph node that was recorded
                # before the repack happened.
                key=(
                    self.n_envs,
                    ptr_key(handles["done"]),
                    ptr_key(handles["step_count"]),
                    ptr_key(handles["truncated"]),
                    ptr_key(handles["reset_mask"]),
                ),
            )
            launch.set_param_by_name(
                "max_steps", wp.int32(self.max_steps if self.max_steps is not None else 0)
            )
            launch.launch()
            scenario.reset_in_graph()

        def _token() -> object:
            return (base_hook.token(), self.max_steps)

        def _carries() -> list[torch.Tensor]:
            out = list(base_hook.carries())
            # Everything the tail advances *in place* that ``base_hook.carries()`` does
            # not already cover. The physics warm-up is out-of-place, but the *reset*
            # kernel writes the persistent state and the goals directly; the episode-end
            # kernel writes the step count and ``truncated``; and the reset mask is the
            # subtle one — ``prepare`` zeroes it before the step, warm-up's episode-end
            # kernel stamps it, and the capture step's replay would then run its own
            # *step* obs pass against a dirty mask, rebasing shaping baselines for envs
            # that never reset. The seed counter is carried so warm-up does not consume
            # a draw the eager path would not have.
            out.extend(w.runtime.state_views)  # pos/theta/vel/speed/ang_vel
            out.append(w.goals)
            out.append(self._step_count)
            out.append(self._truncated_buf)
            out.append(scenario.fb[scenario._fused_mask])
            out.append(w._seed_state_t)
            return out

        def _after_warmup() -> None:
            base_hook.after_warmup()
            # The tail's *masked* neighbor rebuild (inside ``reset_in_graph`` ->
            # ``_launch_obs``) only touches the envs the reset mask selects; after the
            # carries above are restored to their pre-warm-up values, the grid still
            # holds neighbor lists built from warm-up's now-discarded positions for
            # every env the mask did not select. A full, unmasked rebuild here
            # re-derives the grid from the restored state. See
            # ``WholeStepHook.after_warmup``'s docstring for the long version.
            w.build_neighbors(reset_mask=None)

        return WholeStepHook(
            run=_run,
            prepare=_prepare,
            token=_token,
            carries=_carries,
            after_warmup=_after_warmup,
        )

    def _compose_step_count_hook(self, base_hook: WholeStepHook) -> WholeStepHook:
        """Fold just the step-count bookkeeping into ``base_hook``.

        The counterpart to :meth:`_compose_graph_reset_hook` for every step that has no
        in-graph reset to run: ``auto_reset=False``, and ``auto_reset=True`` on a scenario
        whose reset is not capture-safe. Both still owe the same two lines —
        ``step_count += 1`` and ``truncated = step_count >= max_steps`` — and paying them
        host-side is two torch kernel launches per step over ``[n_envs]``, which on a
        4-agent config is about half the wall clock (the device finishes the replay long
        before the host finishes issuing the step). Inside the capture they cost nothing.

        Auto-reset that stays host-side keeps working unchanged: this kernel only advances
        the counter, and ``step``'s ``masked_fill_`` still zeroes it for the envs that just
        reset, after the reset itself.
        """
        launch = CachedLaunch()
        handles: dict[str, wp.array] = {}

        def _ensure_handles() -> None:
            # Both buffers are allocated once in ``__init__`` and never reassigned, so a
            # handle built here stays valid; it is built lazily anyway, to keep
            # constructing an ``Environment`` free of Warp work the caller may never use.
            if handles:
                return
            handles["step_count"] = wp.from_torch(self._step_count)
            handles["truncated"] = wp.from_torch(
                self._truncated_buf.view(torch.uint8), dtype=wp.uint8
            )

        def _prepare() -> None:
            base_hook.prepare()
            _ensure_handles()

        def _run() -> None:
            base_hook.run()
            max_steps = wp.int32(self.max_steps if self.max_steps is not None else 0)
            packed = launch.get(
                step_count_kernel,
                dim=self.n_envs,
                inputs=[max_steps, handles["step_count"], handles["truncated"]],
                device=self.device,
                # ``max_steps`` is out of the key and re-set below for the same reason as
                # in ``_compose_graph_reset_hook``: a caller may reassign
                # ``env.max_steps``, and the response that actually works under capture is
                # the recapture ``_token`` forces, not a repack of the cached launch.
                key=(self.n_envs, ptr_key(handles["step_count"]), ptr_key(handles["truncated"])),
            )
            packed.set_param_by_name("max_steps", max_steps)
            packed.launch()

        def _token() -> object:
            return (base_hook.token(), self.max_steps)

        def _carries() -> list[torch.Tensor]:
            # The two buffers the launch above advances in place. Without them, graph
            # warm-up would leave the step count one ahead and ``truncated`` describing a
            # step that never happened.
            out = list(base_hook.carries())
            out.append(self._step_count)
            out.append(self._truncated_buf)
            return out

        return WholeStepHook(
            run=_run,
            prepare=_prepare,
            token=_token,
            carries=_carries,
            after_warmup=base_hook.after_warmup,
        )

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
        self.world.set_kernel_seed(int(seed))

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
        self._step_count.masked_fill_(env_mask, 0)
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

            ``truncated`` is a **persistent buffer** written in place every step (like
            the fused obs/reward buffers), not a fresh allocation — the same
            ``clone_outputs`` contract as the rest of the tuple applies to it.

            Wherever a whole-step hook is in play, the step-count increment and
            ``truncated`` already happened *inside* ``self.world.step(actions)`` above, as
            part of the same graph replay (``step_count_kernel``); when ``auto_reset=True``
            and the scenario supports it, so did the mask-fill and ``reset_world``
            (``episode_end_kernel`` plus the scenario's own reset) — see
            ``self._reset_in_graph`` / ``World.ran_post_physics`` and the class docstring's
            "For an eligible scenario" paragraph. The code below still runs the equivalent
            host-side arithmetic, but only for the parts that did **not** run in the hook
            (all of it on a taped/grad step or with no hook at all), so this docstring's
            description of what ``step`` does is accurate either way — only *where* the
            work happens changes.
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
        # ``tail_ran`` is True iff the composed hook's episode-end kernel + masked
        # reset already ran as *part of* ``self.world.step`` above (a captured replay,
        # or the eager-persistent fallback running the identical hook eagerly) --
        # ``ran_post_physics`` is exactly that gate, and is False on a taped/grad step
        # or when ``self._reset_in_graph`` is False, both of which must keep the old
        # host-side tail below.
        tail_ran = self._reset_in_graph and self.world.ran_post_physics
        # Both hook compositions advance the step count and write ``truncated``
        # (``episode_end_kernel`` / ``step_count_kernel``), so the host arithmetic is owed
        # only when neither ran: a taped/grad step, or no whole-step hook at all.
        counted = self._whole_step and self.world.ran_post_physics
        if not counted:
            self._step_count += 1
        # ``terminated`` reads the fused ``done`` buffer regardless of ``tail_ran``: the
        # in-graph tail's reset pass is obs-only (``full_pass=0``), so it never touches
        # ``done`` -- this still describes the pre-reset transition exactly as it always
        # has, whether or not that transition's episode also just ended and got reset.
        terminated = self.scenario.done()
        if self.max_steps is not None:
            if counted:
                # The hook's kernel already wrote this in place (see
                # swarp/core/episode_kernels.py); redoing the host ``torch.ge`` would
                # just recompute the same value one host round-trip later.
                truncated = self._truncated_buf
            else:
                truncated = torch.ge(self._step_count, self.max_steps, out=self._truncated_buf)
        else:
            truncated = self._never_truncated
        info = self.scenario.info()

        if self.auto_reset and not tail_ran:
            episode_end = torch.logical_or(terminated, truncated, out=self._episode_end_buf)
            # Obs-only reset pass: reward/done/info were already returned for this
            # transition and their (fused) buffers must not be clobbered.
            with torch.no_grad():
                self.scenario.reset_world(episode_end, obs_only=True)
            self._step_count.masked_fill_(episode_end, 0)

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
