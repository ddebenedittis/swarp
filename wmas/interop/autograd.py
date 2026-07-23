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

from typing import NamedTuple

import torch
import warp as wp

from wmas.core.state import WorldState, field_wp_dtype
from wmas.core.stepper import Stepper

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


def _torch_stream_scope(device: str):
    if device.startswith("cuda"):
        return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream()))
    return wp.ScopedDevice(device)


class _WarpStepFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, stepper: Stepper, actions: torch.Tensor, *state_tensors: torch.Tensor):
        scalar = _WP_SCALAR[actions.dtype]
        state = TorchState(*state_tensors)
        n_envs = actions.shape[0]

        with _torch_stream_scope(stepper.device):
            state_wp, in_grads = _wrap_input_state(state, scalar, with_grad=True)
            actions_wp, act_grad = _wrap_actions(actions, scalar, with_grad=True)
            out_wp = stepper.alloc_state(n_envs, requires_grad=True)
            buffers = stepper.make_buffers(n_envs, requires_grad=True)

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
            return (None, ctx.act_grad.clone(), *(g.clone() for g in ctx.in_grads))


def warp_step(stepper: Stepper, state: TorchState, actions: torch.Tensor) -> TorchState:
    """Differentiable env step: ``state, actions -> next state`` (torch tensors).

    Under ``torch.no_grad()`` (or when no input requires grad) a tape-free fast
    path is used with recycled intermediate buffers.
    """
    state = fill_state_defaults(state)
    grad_mode = torch.is_grad_enabled() and (
        actions.requires_grad or any(t.requires_grad for t in state)
    )
    if grad_mode:
        return TorchState(*_WarpStepFn.apply(stepper, actions, *state))

    scalar = _WP_SCALAR[actions.dtype]
    n_envs = actions.shape[0]
    with _torch_stream_scope(stepper.device):
        state_wp, _ = _wrap_input_state(state, scalar, with_grad=False)
        actions_wp, _ = _wrap_actions(actions, scalar, with_grad=False)
        # Recycled ping-pong output (zero steady-state allocation); input and
        # output never alias. Returned tensors are valid until this batch size
        # is stepped twice more (documented on Stepper.output_state).
        out_wp = stepper.output_state(n_envs)
        stepper.launch_substeps(state_wp, actions_wp, out_wp, stepper.cached_buffers(n_envs))
        return TorchState(*(wp.to_torch(a, requires_grad=False) for a in out_wp.arrays()))


def rollout(
    stepper: Stepper, state: TorchState, actions_seq: torch.Tensor
) -> tuple[TorchState, list[TorchState]]:
    """Backprop-through-time rollout.

    Args:
        state: initial state.
        actions_seq: ``[T, n_envs, n_agents, 2]`` action sequence.

    Returns:
        ``(final_state, trajectory)`` where trajectory holds the state after
        each of the T steps; gradients flow to ``state`` and ``actions_seq``.
    """
    trajectory: list[TorchState] = []
    for t in range(actions_seq.shape[0]):
        state = warp_step(stepper, state, actions_seq[t])
        trajectory.append(state)
    return state, trajectory
