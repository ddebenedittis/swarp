"""``torch.compile``-compatible step via ``torch.library.custom_op``.

The default :func:`swarp.warp_step` wraps a ``torch.autograd.Function`` around raw
Warp launches, which ``torch.compile`` cannot trace through (it graph-breaks on
the opaque Function and the Warp C calls). This module registers the step as a
functional custom op ``swarp::step`` with a fake (meta) implementation and an
autograd rule, so a compiled policy/env loop treats it as a single opaque,
differentiable operator instead of breaking the graph.

The op is *functional*: the forward allocates fresh **outputs** and mutates no
input, and the backward re-records a Warp tape from the saved inputs and replays
it (recompute-in-backward), so no Warp state has to survive between the forward
and backward calls — exactly what Dynamo/AOTAutograd expect. The substep
*scratch* is recycled from the stepper's cache: it is consumed inside the launch
and never escapes, so it is invisible to the op's contract.

Use :func:`compiled_warp_step`, which mirrors :func:`warp_step`'s signature but
routes through the registered op.

Two consequences of that contract worth knowing:

* **The per-step clone of every output is required, not defensive.** A custom op may not
  return a tensor that aliases storage it does not own, and ``wp.to_torch`` returns a view
  of the Warp array. Dropping the clone is a use-after-free once the stepper recycles that
  array, and AOTAutograd's aliasing checks reject it outright.
* **``torch.compile(mode="reduce-overhead")`` is not supported.** It wants to CUDA-graph
  the compiled region, and the forward allocates Warp arrays every call — capture-illegal.
  For graph execution use :class:`swarp.interop.persistent.StepRuntime` instead, which
  captures the whole step (physics plus the fused obs/reward launches) against persistent
  buffers; that is the fast path this module is *not* trying to be.
"""

from __future__ import annotations

import weakref

import torch

from swarp.core.state import field_wp_dtype
from swarp.core.stepper import Stepper
from swarp.interop.autograd import (
    TorchState,
    fill_state_defaults,
    torch_stream_scope,
    warp_step,
    wrap_actions,
    wrap_input_state,
)

# Steppers are plain Python objects and cannot cross the custom-op boundary, so they are
# referenced by a small integer handle registered here. The registry holds them **weakly**:
# a strong dict made every stepper ever compiled immortal, pinning its device buffers (and
# in an env's case the whole world) for the life of the process. ``_HANDLES`` is the
# reverse map, so re-registering is an O(1) lookup rather than a scan of the registry on
# every ``compiled_warp_step`` call; a finalizer drops its entry when the stepper dies,
# which also runs before the id can be reused by a new object.
_STEPPERS: weakref.WeakValueDictionary[int, Stepper] = weakref.WeakValueDictionary()
_HANDLES: dict[int, int] = {}  # id(stepper) -> handle
_NEXT_ID = 0


def register_stepper(stepper: Stepper) -> int:
    """Return a stable int handle for ``stepper`` usable with the custom op.

    The handle does not keep ``stepper`` alive: the caller (normally the ``Environment``
    or ``World`` that owns it) must outlive the compiled step.
    """
    global _NEXT_ID
    handle = _HANDLES.get(id(stepper))
    if handle is not None:
        return handle
    handle = _NEXT_ID
    _NEXT_ID += 1
    _STEPPERS[handle] = stepper
    _HANDLES[id(stepper)] = handle
    weakref.finalize(stepper, _HANDLES.pop, id(stepper), None)
    return handle


def _stepper(handle: int) -> Stepper:
    """The registered stepper, or a legible error if it has been collected."""
    try:
        return _STEPPERS[handle]
    except KeyError:
        raise RuntimeError(
            f"swarp::step was called with handle {handle}, whose Stepper has been "
            "garbage-collected. Keep the Environment/World (or the Stepper itself) alive "
            "for as long as the compiled step is used."
        ) from None


@torch.library.custom_op("swarp::step", mutates_args=())
def _step_op(
    stepper_id: int,
    actions: torch.Tensor,
    pos: torch.Tensor,
    theta: torch.Tensor,
    vel: torch.Tensor,
    speed: torch.Tensor,
    ang_vel: torch.Tensor,
    z: torch.Tensor,
    vz: torch.Tensor,
    attitude: torch.Tensor,
    body_rates: torch.Tensor,
) -> list[torch.Tensor]:
    import warp as wp

    stepper = _stepper(stepper_id)
    scalar = {torch.float32: wp.float32, torch.float64: wp.float64}[actions.dtype]
    state = TorchState(pos, theta, vel, speed, ang_vel, z, vz, attitude, body_rates)
    n_envs = actions.shape[0]
    with torch_stream_scope(stepper.device):
        state_wp, _ = wrap_input_state(state, scalar, with_grad=False)
        actions_wp, _ = wrap_actions(actions, scalar, with_grad=False)
        out_wp = stepper.alloc_state(n_envs, requires_grad=False)
        # The scratch is consumed inside the launch and never handed back, so recycling
        # it does not break the op's functional contract — the *outputs* are still
        # freshly allocated, which is what AOTAutograd needs.
        stepper.launch_substeps(state_wp, actions_wp, out_wp, stepper.cached_buffers(n_envs))
        return [wp.to_torch(a, requires_grad=False).clone() for a in out_wp.arrays()]


@_step_op.register_fake
def _step_op_fake(
    stepper_id, actions, pos, theta, vel, speed, ang_vel, z, vz, attitude, body_rates
):
    # Outputs have the same shapes/dtypes/device as the corresponding inputs — but not
    # necessarily the same *strides*: the real op returns clones of Warp arrays, which are
    # always contiguous, while an input may be a non-contiguous view. ``empty_like``
    # inherits the input's layout and would let the traced graph plan against strides the
    # eager op never produces.
    return [
        torch.empty(t.shape, dtype=t.dtype, device=t.device)
        for t in (pos, theta, vel, speed, ang_vel, z, vz, attitude, body_rates)
    ]


def _step_setup_context(ctx, inputs, output):
    stepper_id, actions, *state = inputs
    ctx.stepper_id = stepper_id
    ctx.save_for_backward(actions, *state)


def _step_backward(ctx, *grad_outputs):
    import warp as wp

    # A list-valued op output arrives as a single list of per-output grads.
    grads = grad_outputs[0] if len(grad_outputs) == 1 else grad_outputs
    stepper = _stepper(ctx.stepper_id)
    actions, *state_tensors = ctx.saved_tensors
    scalar = {torch.float32: wp.float32, torch.float64: wp.float64}[actions.dtype]
    state = TorchState(*state_tensors)
    n_envs = actions.shape[0]
    with torch_stream_scope(stepper.device):
        state_wp, in_grads = wrap_input_state(state, scalar, with_grad=True)
        actions_wp, act_grad = wrap_actions(actions, scalar, with_grad=True)
        out_wp = stepper.alloc_state(n_envs, requires_grad=True)
        buffers = stepper.make_buffers(n_envs, requires_grad=True)
        tape = wp.Tape()
        with tape:
            stepper.launch_substeps(state_wp, actions_wp, out_wp, buffers)
        seeds = {}
        for name, arr, adj in zip(TorchState._fields, out_wp.arrays(), grads, strict=True):
            seeds[arr] = wp.from_torch(
                adj.contiguous(), dtype=field_wp_dtype(name, scalar), requires_grad=False
            )
        tape.backward(grads=seeds)
    return (None, act_grad.clone(), *(g.clone() for g in in_grads))


_step_op.register_autograd(_step_backward, setup_context=_step_setup_context)


def compiled_warp_step(stepper: Stepper, state: TorchState, actions: torch.Tensor) -> TorchState:
    """``warp_step`` routed through the ``swarp::step`` custom op (torch.compile-safe)."""
    state = fill_state_defaults(state)
    handle = register_stepper(stepper)
    outs = torch.ops.swarp.step(handle, actions, *state)
    return TorchState(*outs)


# Re-export the eager reference so callers can compare paths.
__all__ = [
    "compiled_warp_step",
    "register_stepper",
    "warp_step",
]
