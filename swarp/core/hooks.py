"""The whole-step hook: what a scenario hands the persistent step runtime.

:class:`WholeStepHook` is the declared form of a contract that used to be carried on
four undeclared private members of :class:`~swarp.scenarios.base.Scenario`
(``_pre_graph_step`` / ``_graph_post_physics`` / ``graph_recapture_token`` /
``_graph_warmup_carries``), passed to the runtime as three loose callables. Bundling
them makes the contract greppable, lets a scenario be checked against it, and means
adding a fifth concern later is one field rather than another positional argument on
:meth:`~swarp.interop.persistent.StepRuntime.set_post_physics`.

This is deliberately a **leaf module** — it imports only ``torch`` and the standard
library — so the core (``interop.persistent``, ``core.environment``) and the scenario
layer can both depend on it without an import cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


def _noop() -> None:
    """Nothing to prepare."""


def _zero() -> int:
    """A hook whose buffer handles can never move needs no recapture."""
    return 0


def _no_carries() -> list[torch.Tensor]:
    """A hook that advances no persistent state needs nothing snapshotted."""
    return []


@dataclass(frozen=True)
class WholeStepHook:
    """Everything :class:`~swarp.interop.persistent.StepRuntime` needs to fold a
    scenario's fused obs/reward launches into the whole-step CUDA graph.

    Attributes:
        run: the capture-safe launch sequence, run right after the physics copy-back
            inside both the captured graph and the eager fallback. **Only**
            ``wp.launch`` / neighbor-grid builds against pointer-stable buffers: no
            allocation, no host reads, no torch ops that allocate. Anything that
            cannot honour that belongs in :attr:`prepare`.
        prepare: capture-*un*safe preparation, run eagerly on the default stream
            before the physics step — lazy buffer allocation, clearing the reset
            mask, re-wrapping a handle whose tensor moved. Defaults to a no-op.
        token: a token that changes whenever :attr:`run`'s cached buffer handles do,
            so the runtime knows to recapture. Defaults to a constant ``0``, which is
            correct exactly when no handle can ever move. The return type is ``object``
            rather than ``int``: the runtime only ever compares two tokens for equality
            (``__eq__``, via the recapture-key tuple in ``StepRuntime._graph_key``), so
            any comparable value works — an ``int``, or a tuple when more than one thing
            can independently force a recapture (``Environment`` composes one from the
            scenario's own token *and* ``max_steps``, since the episode-end launch bakes
            ``max_steps`` in by value).
        carries: the persistent buffers :attr:`run` advances **in place** (a shaping
            baseline, a coverage latch, movable-body state). Graph warm-up invokes
            ``run`` on the input state purely to compile kernels, so these are
            snapshotted before and restored after it; anything omitted here is
            silently advanced one extra time at capture. Defaults to empty.
        after_warmup: run once, after graph warm-up's carries have been restored, to
            re-derive state that is a *function* of the carries rather than a snapshot
            of them. Defaults to a no-op.

            The motivating case is the neighbor grid. ``carries`` fixes every buffer
            warm-up wrote in place, but a *masked* reset's neighbor rebuild
            (``World.build_neighbors(reset_mask=...)``) only touches the envs the reset
            mask selects — by design, see that method's docstring — so after the
            restore the grid still holds neighbor lists built from warm-up's (now
            discarded) positions for exactly those envs, while ``built_version`` claims
            it matches the restored state. A snapshot cannot fix this because the grid
            is not itself in the carry list (rebuilding it is cheap and pointer-stable
            handles don't need snapshotting) — only its *contents* are wrong, and only
            for a subset of envs a masked rebuild will not touch again until they next
            reset. A full, unmasked rebuild after the restore re-derives the grid from
            the now-correct state and is the one thing that fixes it.
    """

    run: Callable[[], None]
    prepare: Callable[[], None] = _noop
    token: Callable[[], object] = _zero
    carries: Callable[[], list[torch.Tensor]] = _no_carries
    after_warmup: Callable[[], None] = _noop
