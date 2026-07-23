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
        self._graph_version: int | None = None

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

    def _ensure_graph(self) -> None:
        """(Re)capture the physics graph when missing or stale (a mutation bumped
        ``stepper.mutation_version`` — e.g. an obstacle-count change)."""
        if self._graph is not None and self._graph_version == self.stepper.mutation_version:
            return
        stepper = self.stepper
        # Warm-up compiles kernels/allocates without mutating state (launch into
        # the scratch output only, no copy-back).
        stepper.launch_substeps(
            self.state,
            self._actions,
            self._state_out,
            self._buffers,
            reuse_neighbors=True,
            skip_drone=True,
        )
        wp.synchronize_device(stepper.device)
        # Force the neighbor-reuse branch at capture time (its correctness at
        # replay is maintained by the eager post-step build each step).
        if stepper.collisions:
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
        self._graph = capture.graph
        self._graph_version = stepper.mutation_version

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
        return self.state_views
