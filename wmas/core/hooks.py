"""The whole-step hook: what a scenario hands the persistent step runtime.

:class:`WholeStepHook` is the declared form of a contract that used to be carried on
four undeclared private members of :class:`~wmas.scenarios.base.Scenario`
(``_pre_graph_step`` / ``_graph_post_physics`` / ``graph_recapture_token`` /
``_graph_warmup_carries``), passed to the runtime as three loose callables. Bundling
them makes the contract greppable, lets a scenario be checked against it, and means
adding a fifth concern later is one field rather than another positional argument on
:meth:`~wmas.interop.persistent.StepRuntime.set_post_physics`.

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
    """Everything :class:`~wmas.interop.persistent.StepRuntime` needs to fold a
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
            correct exactly when no handle can ever move.
        carries: the persistent buffers :attr:`run` advances **in place** (a shaping
            baseline, a coverage latch, movable-body state). Graph warm-up invokes
            ``run`` on the input state purely to compile kernels, so these are
            snapshotted before and restored after it; anything omitted here is
            silently advanced one extra time at capture. Defaults to empty.
    """

    run: Callable[[], None]
    prepare: Callable[[], None] = _noop
    token: Callable[[], int] = _zero
    carries: Callable[[], list[torch.Tensor]] = _no_carries
