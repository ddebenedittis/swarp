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

from wmas.core.state import STATE_FIELDS, VEC2, WorldState
from wmas.core.stepper import Stepper

_WP_SCALAR = {torch.float32: wp.float32, torch.float64: wp.float64}


class TorchState(NamedTuple):
    """Torch-side view of the world state (see wmas.core.state.WorldState)."""

    pos: torch.Tensor  # [n_envs, n_agents, 2]
    theta: torch.Tensor  # [n_envs, n_agents]
    vel: torch.Tensor  # [n_envs, n_agents, 2]
    speed: torch.Tensor  # [n_envs, n_agents]
    ang_vel: torch.Tensor  # [n_envs, n_agents]


def _field_wp_dtype(name: str, scalar):
    return VEC2[scalar] if STATE_FIELDS[name] else scalar


def _wrap_input_state(tensors: TorchState, scalar, with_grad: bool):
    """Wrap torch tensors as Warp arrays; optionally attach fresh grad buffers."""
    grads = TorchState(*(torch.zeros_like(t) for t in tensors)) if with_grad else None
    arrays = {}
    for name, t, g in zip(
        TorchState._fields, tensors, grads if with_grad else (None,) * 5, strict=True
    ):
        dt = _field_wp_dtype(name, scalar)
        if with_grad:
            arrays[name] = wp.from_torch(t.contiguous(), dtype=dt, grad=g)
        else:
            arrays[name] = wp.from_torch(t.contiguous(), dtype=dt, requires_grad=False)
    return WorldState(**arrays), grads


def _wrap_actions(actions: torch.Tensor, scalar, with_grad: bool):
    grad = torch.zeros_like(actions) if with_grad else None
    arr = wp.from_torch(
        actions.contiguous(), dtype=VEC2[scalar],
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
                    adj.contiguous(), dtype=_field_wp_dtype(name, ctx.scalar),
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
        out_wp = stepper.alloc_state(n_envs)
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
