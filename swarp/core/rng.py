"""Device-side kernel-seed state: the capture-safe replacement for a scalar seed.

A ``wp.launch`` argument that is a plain scalar (``wp.int32(seed)``) gets baked into the
launch by *value* the moment ``record_cmd=True`` packs it — and a whole-step CUDA graph
bakes the launch it captures once and replays it unchanged forever. A reset kernel seeded
that way would draw the exact same spawn/goal points on every single replay, which is
useless for training. An array argument, in contrast, is baked by *pointer*: the graph
replays whatever the pointer holds *at replay time*, so a seed that lives in device memory
can keep changing between replays as long as nothing ever reallocates it (see
``World.seed_state`` in ``swarp/core/world.py`` — allocated once, pointer stable for the
life of the ``World``).

That moves "advance the seed" from Python (``World.next_kernel_seed``, host-side, one
``int`` per call) to a one-thread kernel a scenario launches immediately before the kernel
that consumes the seed. **Advance, then use** — always in that order, matching
``next_kernel_seed``'s own "increment first, return after" so the two derivations agree
call for call (see ``NavigationScenario._launch_reset``, the only consumer for now).

Only :class:`~swarp.scenarios.navigation.NavigationScenario` uses this today (it is the
scenario en route to a captured auto-reset); the other six still call
``World.next_kernel_seed`` host-side, which is untouched and unaffected by any of this.

**Why ``int32``, not ``uint32``.** ``World.seed_state`` is now backed by a torch tensor
(``World._seed_state_t``) so it can be snapshotted and restored around graph warm-up like
any other carry — see ``StepRuntime._ensure_graph``'s ``carries`` list — and torch has no
unsigned 32-bit integer dtype. So the array is declared ``wp.array(dtype=wp.int32)`` and
every function below does its actual mixing in ``wp.uint32`` after an explicit cast:
``wp.uint32(x)`` where ``x: wp.int32`` reinterprets the same 32 bits unsigned (Warp follows
C's cast semantics here, not a value-preserving conversion), so the unsigned arithmetic is
bit-for-bit what it always was — only the storage type changed. The alternative, doing the
mixing in ``wp.int32`` directly, would be wrong for a different reason: ``kernel_seed *
0x9E3779B1`` overflows a signed 32-bit product for most seeds, and signed overflow is
undefined behaviour in the C++ Warp generates, where unsigned overflow is defined to wrap.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def advance_seed_kernel(seed_state: wp.array(dtype=wp.int32)):
    """Single-thread launch: bump the monotonic counter in ``seed_state[1]``.

    Dim-1, no dtype overload needed (it touches no float — the arithmetic is entirely
    32-bit integer), so unlike the physics kernels this lives outside the
    ``_signature(dtype)`` / ``wp.overload`` machinery in ``kernels.py``/``collisions.py``.
    """
    seed_state[1] = wp.int32(wp.uint32(seed_state[1]) + wp.uint32(1))


@wp.func
def seed_from_state(seed_state: wp.array(dtype=wp.int32)) -> wp.int32:
    """The kernel-RNG seed for the *current* ``seed_state``, bit-identical to
    ``World.next_kernel_seed``'s ``(kernel_seed * 0x9E3779B1 + _kernel_step) & 0x7FFFFFFF``.

    Deliberately **uint32** arithmetic, not int32: ``kernel_seed * 0x9E3779B1`` overflows a
    signed 32-bit product for most seeds, and signed overflow is UB in the C++ Warp
    generates. Unsigned arithmetic wraps by definition, and its low 32 bits equal the low
    32 bits of Python's arbitrary-precision product regardless of ``kernel_seed``'s
    magnitude or sign (two's-complement truncation), so masking with ``0x7FFFFFFF``
    afterwards reproduces Python's result exactly. ``wp.uint32(seed_state[i])`` — a cast
    from the ``int32`` storage — is that same two's-complement reinterpretation, so this is
    unchanged from when ``seed_state`` was itself a ``uint32`` array.
    """
    return wp.int32(
        (wp.uint32(seed_state[0]) * wp.uint32(0x9E3779B1) + wp.uint32(seed_state[1]))
        & wp.uint32(0x7FFFFFFF)
    )
