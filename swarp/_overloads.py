"""Concrete kernel instantiation: keep what :func:`warp.overload` returns.

Every generic Warp kernel in this package is explicitly instantiated per dtype at import
time (the ``_signature(dtype)`` helpers next to each kernel). ``wp.overload`` *returns* the
concrete kernel it registers, but the registration loops used to discard it and launch the
**generic** kernel instead — and :func:`warp.launch` re-resolves a generic kernel on every
call::

    if kernel.is_generic:
        fwd_types = kernel.infer_argument_types(fwd_args)   # walks every argument
        kernel = kernel.add_overload(fwd_types)             # builds a signature string

That is pure per-launch Python overhead recomputing a constant: the same kernel, with the
same argument types, every step. Measured on the navigation hot path it was ~65% of the
cost of a launch (492 -> 173 us for the 2D integrator at 4096x16).

So: :func:`register` stores what ``wp.overload`` hands back, and :func:`concrete` looks it
up at launch time, which costs one dict hit. ``concrete`` falls back to the generic kernel
when a signature was never registered, so an unregistered dtype still runs (slowly) rather
than raising.

The ``key`` is whatever distinguishes one instantiation of a kernel from another — a dtype
for most, a ``(dtype, per_env)`` tuple where the parameter matrix has two layouts.
"""

from __future__ import annotations

import warp as wp

#: (kernel, key) -> concrete (non-generic) kernel.
_CONCRETE: dict[tuple, wp.Kernel] = {}


def register(kernel: wp.Kernel, key, signature: list) -> wp.Kernel:
    """Instantiate ``kernel`` for ``signature`` and retain it under ``key``."""
    instance = wp.overload(kernel, signature)
    _CONCRETE[(kernel, key)] = instance
    return instance


def concrete(kernel: wp.Kernel, key) -> wp.Kernel:
    """The concrete instantiation of ``kernel`` for ``key``, or ``kernel`` itself."""
    return _CONCRETE.get((kernel, key), kernel)
