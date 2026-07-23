"""Zero-copy PyTorch autograd bridge over Warp's tape/adjoint machinery.

``warp_step`` advances the simulation one step as a differentiable torch op;
``rollout`` chains steps for backprop-through-time. Forward wraps the input
tensors as Warp arrays (zero-copy), records all substep launches on a
``wp.Tape``, and returns zero-copy views of freshly allocated output arrays.
Backward seeds the output adjoints and replays the tape in reverse.

Grad-mode steps are strictly functional (fresh buffers per substep) because
overwriting an array recorded on a tape silently corrupts its adjoint. Under
``torch.no_grad()`` the tape is skipped and intermediate buffers are recycled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import warp as wp

from wmas.core.state import WorldState, field_wp_dtype
from wmas.core.stepper import StepBuffers, Stepper

_WP_SCALAR = {torch.float32: wp.float32, torch.float64: wp.float64}


class TorchState(NamedTuple):
    """Torch-side view of the world state (see wmas.core.state.WorldState).

    The last four fields are the 6-DOF drone state; they default to ``None`` and
    are auto-filled (zeros, identity attitude) by :func:`warp_step` so the 2D
    models keep their five-field construction.
    """

    pos: torch.Tensor  # [n_envs, n_agents, 2]
    theta: torch.Tensor  # [n_envs, n_agents]
    vel: torch.Tensor  # [n_envs, n_agents, 2]
    speed: torch.Tensor  # [n_envs, n_agents]
    ang_vel: torch.Tensor  # [n_envs, n_agents]
    z: torch.Tensor | None = None  # [n_envs, n_agents]
    vz: torch.Tensor | None = None  # [n_envs, n_agents]
    attitude: torch.Tensor | None = None  # [n_envs, n_agents, 4] (x, y, z, w)
    body_rates: torch.Tensor | None = None  # [n_envs, n_agents, 3]


def fill_state_defaults(state: TorchState) -> TorchState:
    """Materialize any ``None`` drone fields as zeros (identity attitude),
    matching the batch shape/device/dtype of ``pos``."""
    if all(t is not None for t in state):
        return state
    ref = state.pos
    n_envs, n_agents = ref.shape[0], ref.shape[1]
    opts = {"device": ref.device, "dtype": ref.dtype}
    z = state.z if state.z is not None else torch.zeros(n_envs, n_agents, **opts)
    vz = state.vz if state.vz is not None else torch.zeros(n_envs, n_agents, **opts)
    if state.attitude is not None:
        att = state.attitude
    else:
        att = torch.zeros(n_envs, n_agents, 4, **opts)
        att[..., 3] = 1.0  # identity quaternion (x, y, z, w)
    br = (
        state.body_rates
        if state.body_rates is not None
        else torch.zeros(n_envs, n_agents, 3, **opts)
    )
    return state._replace(z=z, vz=vz, attitude=att, body_rates=br)


def _wrap_input_state(tensors: TorchState, scalar, with_grad: bool):
    """Wrap torch tensors as Warp arrays; optionally attach fresh grad buffers."""
    n = len(TorchState._fields)
    grads = TorchState(*(torch.zeros_like(t) for t in tensors)) if with_grad else None
    arrays = {}
    for name, t, g in zip(
        TorchState._fields, tensors, grads if with_grad else (None,) * n, strict=True
    ):
        dt = field_wp_dtype(name, scalar)
        if with_grad:
            arrays[name] = wp.from_torch(t.contiguous(), dtype=dt, grad=g)
        else:
            arrays[name] = wp.from_torch(t.contiguous(), dtype=dt, requires_grad=False)
    return WorldState(**arrays), grads


def _wrap_actions(actions: torch.Tensor, scalar, with_grad: bool):
    """Wrap ``[n_envs, n_agents, act_dim]`` actions as a scalar ``array3d``.

    Action arity is decoupled from geometry (``vec2``): the integrate kernel
    reads the scalar slots each model needs, so ``act_dim`` may exceed 2.
    """
    grad = torch.zeros_like(actions) if with_grad else None
    arr = wp.from_torch(
        actions.contiguous(),
        dtype=scalar,
        **({"grad": grad} if with_grad else {"requires_grad": False}),
    )
    return arr, grad


# Wrapping a torch stream into a Warp stream is not free (a handle alloc + a few
# attribute reads); the torch current stream is stable across steps, so cache the
# wrapped ``wp.Stream`` keyed by (device, cudaStream_t pointer).
_STREAM_CACHE: dict[tuple[str, int], object] = {}


def _torch_stream_scope(device: str):
    if device.startswith("cuda"):
        ts = torch.cuda.current_stream()
        key = (device, ts.cuda_stream)
        wp_stream = _STREAM_CACHE.get(key)
        if wp_stream is None:
            wp_stream = wp.stream_from_torch(ts)
            _STREAM_CACHE[key] = wp_stream
        return wp.ScopedStream(wp_stream)
    return wp.ScopedDevice(device)


@dataclass
class StepGradSlot:
    """Pre-allocated per-step scratch for a taped step: output state, substep
    buffers, and the input/action gradient buffers the adjoint accumulates into.
    See :class:`GradRing`."""

    out_wp: WorldState
    buffers: StepBuffers
    in_grads: TorchState  # gradient buffers, one per input state field
    act_grad: torch.Tensor


class GradRing:
    """Reusable ring of :class:`StepGradSlot` for BPTT rollouts.

    A ``T``-step rollout draws ``T`` distinct slots (one per step — they must not
    alias while the tapes are live), so ``capacity`` must be ``>= T``. Across
    optimizer iterations the ring is :meth:`reset` and the same slots are reused
    with no new allocations (safe because each backward clones the grads it
    returns, so a slot's buffers are free once its step's backward has run).
    Passing no ring (the default) keeps the fresh-allocation path that
    ``torch.autograd.gradcheck`` relies on.
    """

    def __init__(self, stepper: Stepper, n_envs: int, act_dim: int, capacity: int) -> None:
        self.stepper = stepper
        self.capacity = capacity
        self._idx = 0
        act_dtype = torch.float64 if stepper.dtype == wp.float64 else torch.float32
        self.slots: list[StepGradSlot] = []
        for _ in range(capacity):
            out_wp = stepper.alloc_state(n_envs, requires_grad=True)
            buffers = stepper.make_buffers(n_envs, requires_grad=True)
            in_grads = TorchState(*(torch.zeros_like(wp.to_torch(a)) for a in out_wp.arrays()))
            act_grad = torch.zeros(
                n_envs, stepper.n_agents, act_dim, device=stepper.device, dtype=act_dtype
            )
            self.slots.append(StepGradSlot(out_wp, buffers, in_grads, act_grad))

    def reset(self) -> None:
        """Rewind to the first slot (call once at the start of each rollout)."""
        self._idx = 0

    def acquire(self) -> StepGradSlot:
        """Return the next slot with its gradient buffers re-zeroed."""
        if self._idx >= self.capacity:
            raise RuntimeError(
                f"GradRing capacity {self.capacity} exhausted; size it to the rollout length T"
            )
        slot = self.slots[self._idx]
        self._idx += 1
        for g in slot.in_grads:
            g.zero_()
        slot.act_grad.zero_()
        return slot


class _WarpStepFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        stepper: Stepper,
        slot: StepGradSlot | None,
        actions: torch.Tensor,
        *state_tensors: torch.Tensor,
    ):
        scalar = _WP_SCALAR[actions.dtype]
        state = TorchState(*state_tensors)
        n_envs = actions.shape[0]

        with _torch_stream_scope(stepper.device):
            if slot is None:
                state_wp, in_grads = _wrap_input_state(state, scalar, with_grad=True)
                actions_wp, act_grad = _wrap_actions(actions, scalar, with_grad=True)
                out_wp = stepper.alloc_state(n_envs, requires_grad=True)
                buffers = stepper.make_buffers(n_envs, requires_grad=True)
            else:
                in_grads = slot.in_grads
                act_grad = slot.act_grad
                arrays = {}
                for name, t, g in zip(TorchState._fields, state, in_grads, strict=True):
                    arrays[name] = wp.from_torch(
                        t.contiguous(), dtype=field_wp_dtype(name, scalar), grad=g
                    )
                state_wp = WorldState(**arrays)
                actions_wp = wp.from_torch(actions.contiguous(), dtype=scalar, grad=act_grad)
                out_wp = slot.out_wp
                buffers = slot.buffers

            tape = wp.Tape()
            with tape:
                stepper.launch_substeps(state_wp, actions_wp, out_wp, buffers)
            out_tensors = tuple(wp.to_torch(a) for a in out_wp.arrays())

        ctx.stepper = stepper
        ctx.tape = tape
        ctx.buffers = buffers
        ctx.scalar = scalar
        ctx.state_wp = state_wp
        ctx.actions_wp = actions_wp
        ctx.out_wp = out_wp
        ctx.in_grads = in_grads
        ctx.act_grad = act_grad
        return out_tensors

    @staticmethod
    def backward(ctx, *adj_out: torch.Tensor):
        stepper: Stepper = ctx.stepper
        with _torch_stream_scope(stepper.device):
            # Adjoint kernels accumulate (+=); zero all tape grads so repeated
            # backward calls on the same graph (retain_graph) stay correct.
            ctx.tape.zero()
            seeds = {}
            for name, arr, adj in zip(
                TorchState._fields, ctx.out_wp.arrays(), adj_out, strict=True
            ):
                seeds[arr] = wp.from_torch(
                    adj.contiguous(),
                    dtype=field_wp_dtype(name, ctx.scalar),
                    requires_grad=False,
                )
            ctx.tape.backward(grads=seeds)
            # Clone: the buffers are zeroed/rewritten by later backward calls.
            # (None for stepper and slot, which are not differentiable inputs.)
            return (None, None, ctx.act_grad.clone(), *(g.clone() for g in ctx.in_grads))


def warp_step(
    stepper: Stepper,
    state: TorchState,
    actions: torch.Tensor,
    slot: StepGradSlot | None = None,
) -> TorchState:
    """Differentiable env step: ``state, actions -> next state`` (torch tensors).

    Under ``torch.no_grad()`` (or when no input requires grad) a tape-free fast
    path is used with recycled intermediate buffers. ``slot`` (a :class:`GradRing`
    slot) supplies pre-allocated taped scratch on the grad path; ``None`` keeps
    the fresh-allocation behaviour.
    """
    state = fill_state_defaults(state)
    grad_mode = torch.is_grad_enabled() and (
        actions.requires_grad or any(t.requires_grad for t in state)
    )
    if grad_mode:
        return TorchState(*_WarpStepFn.apply(stepper, slot, actions, *state))

    scalar = _WP_SCALAR[actions.dtype]
    n_envs = actions.shape[0]
    with _torch_stream_scope(stepper.device):
        # When ``state`` is the previous step's cached output (the common hot-loop
        # case), its tensors already back a wrapped WorldState — reuse it instead
        # of re-running ``wp.from_torch`` on all nine fields.
        cached_in = stepper.lookup_wrapped(state)
        if cached_in is not None:
            state_wp = cached_in
        else:
            state_wp, _ = _wrap_input_state(state, scalar, with_grad=False)
        actions_wp = stepper.wrap_actions(actions, scalar)
        # Recycled ping-pong output (zero steady-state allocation); input and
        # output never alias. Returned tensors are valid until this batch size
        # is stepped twice more (documented on Stepper.output_state).
        out_wp = stepper.output_state(n_envs)
        stepper.launch_substeps(
            state_wp, actions_wp, out_wp, stepper.cached_buffers(n_envs), reuse_neighbors=True
        )
        return TorchState(*stepper.wrapped_views(out_wp))


def rollout(
    stepper: Stepper,
    state: TorchState,
    actions_seq: torch.Tensor,
    ring: GradRing | None = None,
) -> tuple[TorchState, list[TorchState]]:
    """Backprop-through-time rollout.

    Args:
        state: initial state.
        actions_seq: ``[T, n_envs, n_agents, 2]`` action sequence.
        ring: optional :class:`GradRing` supplying reusable taped scratch; its
            ``capacity`` must be ``>= T``. Reused across iterations with no new
            allocations. ``None`` allocates fresh scratch each step.

    Returns:
        ``(final_state, trajectory)`` where trajectory holds the state after
        each of the T steps; gradients flow to ``state`` and ``actions_seq``.
    """
    if ring is not None:
        ring.reset()
    trajectory: list[TorchState] = []
    for t in range(actions_seq.shape[0]):
        slot = ring.acquire() if ring is not None else None
        state = warp_step(stepper, state, actions_seq[t], slot=slot)
        trajectory.append(state)
    return state, trajectory
