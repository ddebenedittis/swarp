"""Shared piece of the scenarios' device-side masked resets.

Under ``auto_reset`` a reset runs on **every** step, for the **whole** batch: ``done`` is
a device tensor and the step promises no device->host round-trip, so there is no
host-side "is anything done?" gate that could skip it (see ``docs/performance.md``). The
per-env mask decides what lands, not whether the work happens.

That makes a reset hot-path work, and what it is bound by is the *number* of launches
rather than the arithmetic — a chain of masked ``sample_uniform``/``torch.where`` ops
costs far more in launch overhead than the same draw does inside one kernel. Every
scenario therefore does its whole reset in a single masked launch, one thread per env.
Thread per env, rather than per agent, because the aux state a reset writes is per-env —
a formation centre, a package pose, the T's goal — and drawing it once per env is what
keeps every agent in that env consistent with it.

.. warning::

    **Draw randomness inline, never through a** ``@wp.func``. Warp passes a ``uint32``
    RNG state to a user function *by value*, so ``wp.randf`` advances a local copy and the
    caller's state does not move: a helper like ``rand_point(rng, lim)`` returns the
    **same point every call**. That spawns every agent on top of its neighbours while
    still landing inside the world, so bounds and dtype tests all pass and only a
    separation or spread assertion catches it. Only stateless helpers (``_as`` below)
    belong here; keep ``wp.rand_init``/``wp.randf`` in the kernel body.
"""

from __future__ import annotations

import warp as wp


@wp.func
def _as(x: wp.float32, ref: wp.float32) -> wp.float32:
    return x


@wp.func
def _as(x: wp.float32, ref: wp.float64) -> wp.float64:
    return wp.float64(x)
