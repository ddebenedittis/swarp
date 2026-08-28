"""Persistent-buffer step execution with optional whole-step CUDA-graph replay.

Isaac-Gym-style hot path: instead of allocating (or even re-wrapping) state each
step, one :class:`StepRuntime` owns a fixed set of on-device buffers — the state,
a scratch output (used only by the non-slim drone/RK4 fallback and by the
pre-capture warm-up), the substep scratch, and the action buffer — plus
zero-copy torch views created once. Each step copies the incoming actions into
the fixed buffer and either replays a captured CUDA graph or, when capture is
unavailable, runs the same launches eagerly. For the common slim 2D Euler case
the substep chain integrates straight into ``self.state`` in place (see
``Stepper.launch_substeps``'s ``write_in_place``), so a real step issues no
copy-back at all; drone fleets / RK4 fall back to the old scratch-output-then-
copy-back sequence, which ``launch_substeps`` cannot yet do safely in place.
Either way ``step`` returns the same stable torch views, so the scenario's
obs/reward layer reads persistent memory with no per-step wrapping.

Streams. Warp launches go to Warp's **own** created stream, not the legacy default
stream — that is what makes capture legal at all. That stream carries the *blocking* flag,
so the driver serializes it against stream 0 in both directions, which is why torch code
on the default stream stays ordered against these launches with no event of ours involved.
Under a **user-created** ``torch.cuda.Stream`` that guarantee does not apply, so every
eager entry point here opens :func:`~swarp.interop.autograd.torch_stream_scope`: the
replay, the eager fallback, the pre-capture warm-up, and the two state-loading paths
(whose ``.contiguous()`` temporaries are torch-allocator memory freed at return — a
use-after-free under a user stream). On the default stream that scope deliberately
degrades to a **no-op** behind a 0.4 us stream check: re-deriving with events what the
blocking flag already guarantees cost 10-36% of a graph-mode step, most of it in
``torch.cuda.current_stream()`` itself.

The one thing that cannot be scoped at all is the capture itself: ``wp.ScopedCapture``
captures the *current* stream and capturing torch's legacy stream is a hard CUDA error, so
``_ensure_graph``'s capture block stays on Warp's own stream. Replaying a graph on the
legacy stream is fine; only capturing it is not.

With that, masked auto-resets (writes) and observations (reads) stay ordered with the
replay without a device sync, under a user-created stream as much as under the default one.

Auto-reset used to stay *outside* the graph unconditionally, on the theory that it uses a
``torch.Generator`` whose philox offset is not capture-safe. That was half the story: the
philox offset is exactly as capture-unsafe as it sounds, but the *other* blocker was that
``World.next_kernel_seed`` handed a reset kernel a scalar seed, and a scalar argument gets
baked into a captured launch by value — a graph seeded that way would replay the same draw
forever. ``World.seed_state`` (``swarp/core/rng.py``) fixes that half: a device-resident
seed pair, advanced by its own one-thread kernel immediately before the reset kernel reads
it, baked into the graph by *pointer* rather than by value, so its contents can keep
changing between replays. The real, narrower rule is: a scenario whose whole reset is Warp
launches against pointer-stable buffers with a device-side seed (``supports_graph_reset()``)
can fold its reset into the graph; one that still needs a ``torch.Generator`` draw or an
allocation (navigation with obstacles samples obstacle poses that way) cannot, and keeps
running its reset on the eager tail outside the graph exactly as before. See
``Environment.__init__``'s ``_reset_in_graph`` and
``FusedScenario.supports_graph_reset``/``reset_in_graph``.

:class:`CudaGraphStep` at the bottom is the same idea stripped to its core — one
capture over fixed buffers, no persistent state, no recapture logic — kept as the
standalone reference the graph-vs-eager parity test compares against.
"""

from __future__ import annotations

import warnings

import torch
import warp as wp

from swarp.core.hooks import WholeStepHook
from swarp.core.stepper import Stepper
from swarp.dynamics.base import Integrator
from swarp.interop.autograd import TorchState, fill_state_defaults, torch_stream_scope


class StepRuntime:
    """Owns the persistent step buffers and (optionally) a captured CUDA graph.

    Args:
        stepper: the configured :class:`Stepper`.
        n_envs: batch size (fixed for the runtime's lifetime).
        act_dim: env-level action width.
        use_graph: attempt CUDA-graph capture. Falls back to eager persistent
            execution when the device is CPU or the neighbor backend allocates
            during a build (the ``wp.HashGrid`` ``"grid"`` method).
        warn_on_fallback: whether that fallback warns. True when the caller *asked*
            for capture — they demanded something they did not get, and should hear
            about it. False when capture was inferred (``Environment``'s
            ``use_graph="auto"``), where the fallback is the intended behaviour and a
            warning would only tell the user off for a default they never chose.
    """

    def __init__(
        self,
        stepper: Stepper,
        n_envs: int,
        act_dim: int,
        use_graph: bool = True,
        *,
        warn_on_fallback: bool = True,
    ) -> None:
        self.stepper = stepper
        self.n_envs = n_envs
        self.act_dim = act_dim
        self.device = str(stepper.device)

        self.state = stepper.alloc_state(n_envs)
        self._state_out = stepper.alloc_state(n_envs)
        self._buffers = stepper.make_buffers(n_envs, requires_grad=False)
        self._actions = wp.zeros(
            (n_envs, stepper.n_agents, act_dim), dtype=stepper.dtype, device=stepper.device
        )
        # Zero-copy torch views, created once (stable objects across steps).
        self.state_views = TorchState(
            *(wp.to_torch(a, requires_grad=False) for a in self.state.arrays())
        )
        self.actions_view = wp.to_torch(self._actions, requires_grad=False)

        # Capture eligibility: CUDA + an allocation-free neighbor backend.
        self._can_graph = use_graph and self.device.startswith("cuda")
        if self._can_graph and stepper.collisions:
            method = stepper.grid(n_envs).method
            if method not in ("brute", "uniform_grid"):
                self._can_graph = False
        if use_graph and not self._can_graph and warn_on_fallback:
            warnings.warn(
                "CUDA-graph capture unavailable (needs a CUDA device and the "
                "brute/uniform_grid neighbor backend); using eager persistent "
                "execution instead.",
                stacklevel=2,
            )
        self._graph = None
        self._graph_version: tuple[int, int] | None = None

        # Optional whole-step hook (neighbor query + fused obs/reward into the
        # persistent buffers), run right after the physics copy-back inside both the
        # captured graph and the eager fallback. Wired by ``Environment`` from
        # ``Scenario.graph_hook()``; see :class:`~swarp.core.hooks.WholeStepHook`.
        self._hook: WholeStepHook | None = None
        self._ran_post = False

    # ------------------------------------------------------------------ state

    def load_state(self, state: TorchState) -> None:
        """Copy an external torch state into the persistent buffers in place.

        Scoped onto torch's stream: a ``.contiguous()`` here can be a fresh
        torch-allocator tensor that is freed the moment this returns, so a copy issued on
        Warp's own stream would be reading memory torch has already handed out again.
        """
        state = fill_state_defaults(state)
        with torch_stream_scope(self.device):
            for dst, src in zip(self.state.arrays(), state, strict=True):
                wp.copy(dst, wp.from_torch(src.detach().contiguous(), dtype=dst.dtype))

    def reset_state(self) -> None:
        """Zero the persistent state in place (keeps the views/graph valid)."""
        with torch_stream_scope(self.device):
            self.state.zero_()

    @property
    def graph_active(self) -> bool:
        return self._graph is not None

    def set_post_physics(self, hook: WholeStepHook | None) -> None:
        """Register (or clear) the whole-step hook. Invalidates any existing graph.

        The runtime uses four of the hook's five members: :attr:`~swarp.core.hooks.
        WholeStepHook.run` is captured / invoked eagerly after the physics step,
        :attr:`~swarp.core.hooks.WholeStepHook.token` gates recapture,
        :attr:`~swarp.core.hooks.WholeStepHook.carries` is snapshotted around warm-up, and
        :attr:`~swarp.core.hooks.WholeStepHook.after_warmup` runs once the restore is
        done. ``prepare`` is the caller's to run, eagerly, before the step — see
        :class:`~swarp.core.hooks.WholeStepHook` for why it cannot live in here.
        """
        self._hook = hook
        self._graph = None
        self._graph_version = None

    def release(self) -> None:
        """Drop the captured CUDA graph, freeing its device-side resources.

        Idempotent, and non-destructive: the persistent buffers, the stable torch views
        and the registered hook all survive, so the next :meth:`step` simply recaptures
        (exactly what happens after any recapture-token bump). This is what
        :meth:`~swarp.core.environment.Environment.close` calls to hand the graph's memory
        back without tearing the runtime down.
        """
        self._graph = None
        self._graph_version = None

    @property
    def ran_post_physics(self) -> bool:
        return self._ran_post

    # ------------------------------------------------------------------ step

    def _is_slim(self) -> bool:
        """Whether the fleet currently qualifies for the in-place slim 2D Euler
        path (``Stepper.launch_substeps(write_in_place=True)``).

        Mirrors the ``slim`` predicate ``launch_substeps`` computes internally
        (its ``not buffers.taped`` conjunct is always True here since these
        buffers are allocated with ``requires_grad=False``). Computed fresh on
        every call rather than cached: ``stepper.enable_slim2d`` is a mutable
        test/parity knob (see ``test_slim_matches_full``) that can be toggled
        after this runtime is constructed, and a cached value would drift out
        of sync with what ``launch_substeps`` actually does. Drone fleets / RK4
        fall back to the scratch-output + copy-back sequence below."""
        stepper = self.stepper
        return (
            stepper.enable_slim2d
            and not stepper.has_drone
            and stepper.world.integrator == Integrator.EULER
        )

    def _copy_back(self) -> None:
        """Copy the freshly computed output back into the persistent state.

        Only used on the non-slim fallback (drone fleets / RK4), where
        ``launch_substeps`` cannot write in place. Slim 2D fleets integrate
        straight into ``self.state`` and never hit this."""
        n = 9
        for dst, src in zip(self.state.arrays()[:n], self._state_out.arrays()[:n], strict=True):
            wp.copy(dst, src)

    def _run_eager(self) -> None:
        if self._is_slim():
            self.stepper.launch_substeps(
                self.state,
                self._actions,
                self.state,
                self._buffers,
                reuse_neighbors=True,
                skip_drone=True,
                write_in_place=True,
            )
        else:
            self.stepper.launch_substeps(
                self.state,
                self._actions,
                self._state_out,
                self._buffers,
                reuse_neighbors=True,
                skip_drone=True,
            )
            self._copy_back()
        if self._hook is not None:
            self._hook.run()

    def _graph_key(self) -> tuple[int, object]:
        """Recapture key: physics mutation version + the hook's handle token.

        The hook's ``token`` is typed ``Callable[[], object]`` (see
        :class:`~swarp.core.hooks.WholeStepHook`), so this tuple's second element can be
        anything comparable — an ``int`` for the common case, or a composite tuple when
        the hook was assembled from more than one thing that can force a recapture.
        """
        post_ver = 0 if self._hook is None else self._hook.token()
        return (self.stepper.mutation_version, post_ver)

    def _ensure_graph(self) -> None:
        """(Re)capture the whole-step graph when missing or stale.

        Recapture fires on a physics mutation (``stepper.mutation_version`` — e.g.
        an obstacle-count change) or when the post-physics hook's buffer handles
        change (its ``token`` — e.g. a shaping-baseline reallocation)."""
        key = self._graph_key()
        if self._graph is not None and self._graph_version == key:
            return
        stepper = self.stepper
        # Whether the grid currently holds neighbors for ``state`` (built by the
        # scenario's reset/post-step). Only then may the graph bake in substep-0
        # reuse — a scenario that never refreshes the grid must instead have the
        # graph build neighbors itself, or it would reuse an unpopulated list.
        grid_valid = (
            stepper.collisions and stepper.grid(self.n_envs).built_version == stepper.state_version
        )
        # Warm-up runs the whole step once *before* ScopedCapture (which forbids
        # allocation), purely to compile kernels and allocate scratch: the physics into
        # the scratch output with no copy-back, then the hook, which compiles the
        # obs/reward kernels. This stays out-of-place (``write_in_place=False``) even
        # for a slim fleet that will capture in place below: advancing
        # ``self.state`` here — deliberately or by aliasing accident — would leave the
        # graph's *first* replay one step ahead of the eager path. Out-of-place warm-up
        # still compiles the identical kernels (``write_in_place`` only changes which
        # array pointers ``launch_integrate`` binds, not the kernel or its module) and
        # allocates the identical scratch, so it satisfies capture's no-compile/
        # no-allocation requirement for the in-place capture that follows just as well.
        #
        # Neither launch is side-effect-free, so snapshot everything they advance **in
        # place** up front and restore it afterwards. That is the hook's declared carries
        # (shaping baselines, coverage latches, movable-body state) *and* the engine's own
        # movable-body/obstacle arrays, which ``launch_substeps`` advances in place inside
        # the substep loop even though it writes agent state only to the scratch output.
        # Snapshotting after the physics warm-up — as this used to — could not undo that:
        # a scenario with a movable body entered its first replay one extra body-step
        # ahead of the eager path.
        #
        # A snapshot cannot fix everything, though: the hook's ``after_warmup`` runs right
        # after the restore for state that is a *function* of the carries rather than a
        # copy of them — see ``WholeStepHook.after_warmup``'s docstring for why the
        # neighbor grid needs exactly this.
        #
        # The warm-up is scoped onto torch's stream (it reads buffers torch has just
        # written, and the carry restore below is a torch write racing the hook's launches
        # otherwise); the ``wp.synchronize_device`` after it is what bridges to the capture
        # stream, which must stay Warp's own — see the module docstring.
        carries = self._hook.carries() if self._hook is not None else []
        saved = [c.clone() for c in carries]
        with torch_stream_scope(self.device):
            stepper.launch_substeps(
                self.state,
                self._actions,
                self._state_out,
                self._buffers,
                reuse_neighbors=True,
                skip_drone=True,
            )
            if self._hook is not None:
                self._hook.run()
            for c, s in zip(carries, saved, strict=True):
                c.copy_(s)
            if self._hook is not None:
                self._hook.after_warmup()
        wp.synchronize_device(stepper.device)
        # Re-sync built_version after warm-up (which bumped state_version without
        # touching the grid's contents) so the capture takes the reuse branch iff
        # the grid was genuinely valid for ``state``.
        if grid_valid:
            stepper.grid(self.n_envs).built_version = stepper.state_version
        with wp.ScopedCapture(device=stepper.device) as capture:
            if self._is_slim():
                # In-place: the integrate kernel reads its own (e, a) slot into
                # locals before writing, and the force pass that reads neighbours
                # cross-thread is a separate, already-completed launch — see
                # ``Stepper.launch_substeps``'s ``write_in_place`` docstring. Every
                # array the graph touches (state, actions, force/neighbor scratch,
                # obstacle arrays) was already allocated by the warm-up above, so
                # binding the integrate output to ``self.state`` instead of
                # ``self._state_out`` triggers neither a fresh kernel compile (same
                # ``concrete(...)`` overload either way) nor a capture-time
                # allocation.
                stepper.launch_substeps(
                    self.state,
                    self._actions,
                    self.state,
                    self._buffers,
                    reuse_neighbors=True,
                    skip_drone=True,
                    write_in_place=True,
                )
            else:
                stepper.launch_substeps(
                    self.state,
                    self._actions,
                    self._state_out,
                    self._buffers,
                    reuse_neighbors=True,
                    skip_drone=True,
                )
                self._copy_back()
            if self._hook is not None:
                self._hook.run()
        self._graph = capture.graph
        self._graph_version = key

    def step(self, actions: torch.Tensor) -> TorchState:
        """Advance the persistent state one step; returns the stable views.

        The returned :class:`TorchState` aliases the persistent buffers and is
        overwritten by the next call — clone if a copy must outlive the step.
        """
        self.actions_view.copy_(actions)
        # ``_ensure_graph`` deliberately stays *outside* the scope: it captures, and
        # capture has to happen on Warp's own stream (module docstring). The replay and the
        # eager fallback go on torch's stream, so they are ordered against the action copy
        # just above and the obs read just after.
        if self._can_graph:
            self._ensure_graph()
            with torch_stream_scope(self.device):
                wp.capture_launch(self._graph)
        else:
            with torch_stream_scope(self.device):
                self._run_eager()
        # Both paths run the hook when one is registered (captured into the graph
        # or invoked eagerly); Environment reads this to skip a redundant post_step.
        self._ran_post = self._hook is not None
        return self.state_views


class CudaGraphStep:
    """CUDA-graph capture of the no-grad hot path to cut per-step launch latency.

    Standalone building block kept for the tests. For end-to-end use prefer
    :class:`StepRuntime` above (via ``Environment(..., use_graph=True)``), which owns
    persistent state, replays on the default stream so eager torch resets/observations
    stay ordered, and recaptures automatically on obstacle/param changes.

    Captures a single ``launch_substeps`` on fixed input/output/scratch buffers
    with ``wp.ScopedCapture``; each call copies the incoming state+actions into
    the fixed inputs, replays the graph, and returns cloned outputs. The step
    must be allocation-free during capture, so it requires a CUDA device and a
    neighbor backend that does not allocate *during a build*: ``brute`` and
    ``uniform_grid`` both qualify — the latter allocates its sort scratch lazily on the
    first build and reuses it thereafter, which the warm-up above forces before capture.
    Only the ``wp.HashGrid`` ``"grid"`` backend is ineligible, because Warp reserves
    inside every ``build``. This is the same eligibility rule :class:`StepRuntime`
    applies, and where it gets its ``brute``/``uniform_grid`` allow-list.
    Determinism and no-host-copy behaviour are unchanged (same kernels).
    """

    def __init__(self, stepper: Stepper, n_envs: int, act_dim: int) -> None:
        if not str(stepper.device).startswith("cuda"):
            raise ValueError("CUDA-graph capture requires a CUDA device")
        self.stepper = stepper
        self.n_envs = n_envs
        self._in = stepper.alloc_state(n_envs)
        self._out = stepper.alloc_state(n_envs)
        self._actions = wp.zeros(
            (n_envs, stepper.n_agents, act_dim), dtype=stepper.dtype, device=stepper.device
        )
        self._buffers = stepper.make_buffers(n_envs, requires_grad=False)
        # Warm up so kernels/grids are compiled and allocated before capture.
        stepper.launch_substeps(self._in, self._actions, self._out, self._buffers)
        wp.synchronize_device(stepper.device)
        with wp.ScopedCapture(device=stepper.device) as capture:
            stepper.launch_substeps(self._in, self._actions, self._out, self._buffers)
        self._graph = capture.graph

    def __call__(self, state: TorchState, actions: torch.Tensor) -> TorchState:
        state = fill_state_defaults(state)
        # Same rule as ``StepRuntime.step``: the in-copies read torch-allocator
        # temporaries and the replay must be ordered against them, so both go on torch's
        # current stream. (The capture in ``__init__`` stays on Warp's own.)
        with torch_stream_scope(str(self.stepper.device)):
            wp.copy(self._actions, wp.from_torch(actions.contiguous(), dtype=self._actions.dtype))
            for dst, src in zip(self._in.arrays(), state, strict=True):
                wp.copy(dst, wp.from_torch(src.contiguous(), dtype=dst.dtype))
            wp.capture_launch(self._graph)
        return TorchState(
            *(wp.to_torch(a, requires_grad=False).clone() for a in self._out.arrays())
        )
