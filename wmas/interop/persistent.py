"""Persistent-buffer step execution with optional whole-step CUDA-graph replay.

Isaac-Gym-style hot path: instead of allocating (or even re-wrapping) state each
step, one :class:`StepRuntime` owns a fixed set of on-device buffers — the state,
a scratch output, the substep scratch, and the action buffer — plus zero-copy
torch views created once. Each step copies the incoming actions into the fixed
buffer and either replays a captured CUDA graph (substeps + copy-back) or, when
capture is unavailable, runs the same launches eagerly. Either way it returns the
same stable torch views, so the scenario's obs/reward layer reads persistent
memory with no per-step wrapping.

Capture runs on a side stream (CUDA forbids capturing the legacy default
stream); the graph is *replayed* on the current stream, which is the same
default stream PyTorch's eager ops use, so masked auto-resets (writes) and
observations (reads) stay ordered with the replay without a device sync. Auto-
reset stays *outside* the graph: it uses a ``torch.Generator`` whose philox
offset is not capture-safe. (This mirrors :class:`wmas.interop.compile.CudaGraphStep`.)
"""

from __future__ import annotations

import warnings

import torch
import warp as wp

from wmas.core.hooks import WholeStepHook
from wmas.core.stepper import Stepper
from wmas.dynamics.base import Integrator
from wmas.interop.autograd import TorchState, fill_state_defaults


class StepRuntime:
    """Owns the persistent step buffers and (optionally) a captured CUDA graph.

    Args:
        stepper: the configured :class:`Stepper`.
        n_envs: batch size (fixed for the runtime's lifetime).
        act_dim: env-level action width.
        use_graph: attempt CUDA-graph capture. Falls back to eager persistent
            execution (with a one-time warning) when the device is CPU or the
            neighbor backend allocates during a build (the ``wp.HashGrid``
            ``"grid"`` method).
    """

    def __init__(self, stepper: Stepper, n_envs: int, act_dim: int, use_graph: bool = True) -> None:
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

        # The slim 2D copy-back touches five fields; the drone fields of a 2D
        # fleet stay zero and are never read, so they are never copied.
        self._slim = (
            stepper.enable_slim2d
            and not stepper.has_drone
            and stepper.world.integrator == Integrator.EULER
        )

        # Capture eligibility: CUDA + an allocation-free neighbor backend.
        self._want_graph = use_graph
        self._can_graph = use_graph and self.device.startswith("cuda")
        if self._can_graph and stepper.collisions:
            method = stepper.grid(n_envs).method
            if method not in ("brute", "uniform_grid"):
                self._can_graph = False
        if use_graph and not self._can_graph:
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
        # ``Scenario.graph_hook()``; see :class:`~wmas.core.hooks.WholeStepHook`.
        self._hook: WholeStepHook | None = None
        self._ran_post = False

    # ------------------------------------------------------------------ state

    def load_state(self, state: TorchState) -> None:
        """Copy an external torch state into the persistent buffers in place."""
        state = fill_state_defaults(state)
        for dst, src in zip(self.state.arrays(), state, strict=True):
            wp.copy(dst, wp.from_torch(src.detach().contiguous(), dtype=dst.dtype))

    def reset_state(self) -> None:
        """Zero the persistent state in place (keeps the views/graph valid)."""
        self.state.zero_()

    @property
    def graph_active(self) -> bool:
        return self._graph is not None

    def set_post_physics(self, hook: WholeStepHook | None) -> None:
        """Register (or clear) the whole-step hook. Invalidates any existing graph.

        The runtime uses three of the hook's four members: :attr:`~wmas.core.hooks.
        WholeStepHook.run` is captured / invoked eagerly after the physics copy-back,
        :attr:`~wmas.core.hooks.WholeStepHook.token` gates recapture, and
        :attr:`~wmas.core.hooks.WholeStepHook.carries` is snapshotted around warm-up.
        ``prepare`` is the caller's to run, eagerly, before the step — see
        :class:`~wmas.core.hooks.WholeStepHook` for why it cannot live in here.
        """
        self._hook = hook
        self._graph = None
        self._graph_version = None

    @property
    def ran_post_physics(self) -> bool:
        return self._ran_post

    # ------------------------------------------------------------------ step

    def _copy_back(self) -> None:
        """Copy the freshly computed output back into the persistent state.

        Slim 2D fleets copy only the five live fields; the drone fields stay at
        their (zero) allocation and are never read."""
        n = 5 if self._slim else 9
        for dst, src in zip(self.state.arrays()[:n], self._state_out.arrays()[:n], strict=True):
            wp.copy(dst, src)

    def _run_eager(self) -> None:
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

    def _graph_key(self) -> tuple[int, int]:
        """Recapture key: physics mutation version + the hook's handle token."""
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
        # obs/reward kernels.
        #
        # Neither launch is side-effect-free, so snapshot everything they advance **in
        # place** up front and restore it afterwards. That is the hook's declared carries
        # (shaping baselines, coverage latches, movable-body state) *and* the engine's own
        # movable-body/obstacle arrays, which ``launch_substeps`` advances in place inside
        # the substep loop even though it writes agent state only to the scratch output.
        # Snapshotting after the physics warm-up — as this used to — could not undo that:
        # a scenario with a movable body entered its first replay one extra body-step
        # ahead of the eager path.
        carries = self._hook.carries() if self._hook is not None else []
        saved = [c.clone() for c in carries]
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
        wp.synchronize_device(stepper.device)
        # Re-sync built_version after warm-up (which bumped state_version without
        # touching the grid's contents) so the capture takes the reuse branch iff
        # the grid was genuinely valid for ``state``.
        if grid_valid:
            stepper.grid(self.n_envs).built_version = stepper.state_version
        with wp.ScopedCapture(device=stepper.device) as capture:
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
        if self._can_graph:
            self._ensure_graph()
            wp.capture_launch(self._graph)
        else:
            self._run_eager()
        # Both paths run the hook when one is registered (captured into the graph
        # or invoked eagerly); Environment reads this to skip a redundant post_step.
        self._ran_post = self._hook is not None
        return self.state_views
