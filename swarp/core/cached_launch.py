"""``CachedLaunch``: reuse a packed ``wp.launch`` across calls instead of re-packing.

``wp.launch``'s host cost is dominated by re-marshalling every kernel argument into a
ctypes struct on every call (see ``docs/performance.md`` / the cProfile CLAUDE.md's
"Speed Is a First-Class Citizen" points at). Warp 1.15 exposes exactly the fix:
``wp.launch(..., record_cmd=True)`` returns a :class:`wp.Launch` that packed its
arguments once; ``Launch.set_param_at_index``/``set_param_by_name`` re-pack a single
argument, and ``Launch.launch()`` dispatches with none of the rest re-marshalled.

This is for the **eager** reset path only — the one CLAUDE.md's profile shows paying the
packing cost every step (900 launches / 300 steps, none of them the captured step).
Nothing here is used inside a captured region: the whole-step CUDA graph already
amortizes ``wp.launch``'s cost to once (capture time), so caching a ``Launch`` there
would add complexity for a launch that already only happens once.

**Pointer staleness is the hazard** a raw ``Launch`` cache creates: it holds raw device
pointers, so a buffer move (a watched fused handle rebuilt by ``sync_fused_handles``, a
config change) makes a stale cache launch into memory that may no longer hold what the
kernel expects — silent corruption, not a crash. :class:`CachedLaunch` is keyed instead
of separately invalidated: every call recomputes a small tuple — the ``.ptr`` of every
array-typed argument that is supposed to be step-invariant, plus the value of every
scalar config argument that is supposed to be step-invariant — and a key mismatch throws
the old ``Launch`` away and repacks from scratch. That makes a stale hit structurally
impossible, at the cost of a `key comparison` every call; the comparison is a handful of
integer/float equality checks, orders of magnitude cheaper than the repack it may still
trigger. Truly per-call arguments (an RNG seed, a reset-pass flag) must **not** be part
of the key — that would rebuild the whole launch every call and defeat the point — the
caller instead updates them explicitly, after :meth:`CachedLaunch.get`, with
``Launch.set_param_by_name``/``set_param_at_index``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import warp as wp


def ptr_key(x: Any) -> Any:
    """A key entry for one launch argument: a ``wp.array``'s device pointer, or the
    value itself for anything that is not an array (a python/``wp`` scalar, a bool)."""
    p = getattr(x, "ptr", None)
    return p if p is not None else x


class CachedLaunch:
    """A ``wp.Launch`` kept across calls, rebuilt (full repack) only when ``key`` changes.

    Usage::

        _cache = CachedLaunch()
        ...
        launch = _cache.get(
            concrete(some_kernel, scalar),
            dim=w.n_envs,
            inputs=[...],
            outputs=[...],
            device=w.device,
            key=(w.n_envs, ptr_key(arr_a), ptr_key(arr_b), static_scalar),
        )
        launch.set_param_by_name("seed", wp.int32(seed))  # the one per-call argument
        launch.launch()

    ``key`` must include an entry for every argument that can move or change value
    between calls but is *not* explicitly re-set afterwards — anything left out of
    ``key`` is assumed to never need to invalidate the cache. Conversely, an argument the
    caller updates explicitly every call (via ``set_param_by_name``/``set_param_at_index``
    on the returned ``Launch``) should be left **out** of ``key``: baking a value that
    changes every call into the key would give a fresh key every call and rebuild the
    whole launch, exactly defeating the cache.
    """

    __slots__ = ("_launch", "_key")

    def __init__(self) -> None:
        self._launch: wp.Launch | None = None
        self._key: tuple[Any, ...] | None = None

    def get(
        self,
        kernel: wp.Kernel,
        *,
        dim: int | Sequence[int],
        inputs: Sequence[Any],
        outputs: Sequence[Any] = (),
        device: Any,
        key: tuple[Any, ...],
    ) -> wp.Launch:
        """The cached :class:`wp.Launch` for ``key``, rebuilding it first if ``key``
        (or nothing having been built yet) requires it.

        The rebuild is a plain ``wp.launch(..., record_cmd=True)`` with the given
        ``inputs``/``outputs`` — the same full repack the uncached call would have paid,
        so a cache that never hits costs one extra key comparison and nothing else.
        """
        if self._launch is None or self._key != key:
            self._launch = wp.launch(
                kernel,
                dim=dim,
                inputs=inputs,
                outputs=outputs,
                device=device,
                record_tape=False,
                record_cmd=True,
            )
            self._key = key
        return self._launch
