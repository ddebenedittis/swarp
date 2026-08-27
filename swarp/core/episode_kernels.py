"""The episode-end kernels: ``Environment.step``'s host tail, folded into a launch.

``Environment`` (not any scenario) owns ``max_steps``/``step_count``/``truncated`` — they
are engine-level bookkeeping, not part of a scenario's own terminal condition — so this is
an *engine*-level kernel and lives in ``swarp/core`` rather than in ``swarp/scenarios``,
where ``reset_kernels.py`` holds the shared piece of a scenario's own masked reset (see
that module's docstring).

:func:`episode_end_kernel` reproduces, in one masked launch, exactly the host arithmetic
``Environment.step`` used to run every step under ``auto_reset=True``:

    step_count += 1
    truncated = step_count >= max_steps          # all-False when max_steps is None
    episode_end = terminated | truncated
    step_count.masked_fill_(episode_end, 0)

plus stamping the scenario's fused reset-mask buffer with ``episode_end`` — the same
buffer :meth:`~swarp.scenarios.fused.FusedScenario.reset_mask_wp` fills from Python on the
eager path, filled here instead so the whole tail can live inside the captured whole-step
graph. No dtype overload needed: every argument is integer/byte, so this sits outside the
``_signature(dtype)`` / ``wp.overload`` machinery the physics kernels use.

:func:`step_count_kernel` is the same arithmetic with the reset dropped — the first two
lines only. It covers every *other* step: ``auto_reset=False``, and ``auto_reset=True``
on a scenario whose reset cannot be captured. Those steps still have to advance the
counter and decide ``truncated``, and doing it host-side costs two torch launches (a
``+= 1`` and a ``torch.ge``) that measure ~44 us together on a 4-agent config — half the
step, spent on two elementwise passes over ``[n_envs]``. In the graph they are free.
It deliberately does *not* stamp the reset mask: with no reset to run there is nothing to
stamp, and a stamped mask would force
:meth:`~swarp.scenarios.fused.FusedScenario.prepare_fused` back into zeroing it every step.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def episode_end_kernel(
    done: wp.array(dtype=wp.uint8),
    max_steps: wp.int32,
    step_count: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.uint8),
    reset_mask: wp.array(dtype=wp.uint8),
):
    """Thread per env. ``max_steps <= 0`` means "no time limit" (``Environment.max_steps
    is None``), matching the ``_never_truncated`` all-False buffer the host path returns
    in that case.

    Order matters and mirrors the host code above exactly: increment first, decide
    ``truncated`` off the *incremented* count (so a step landing exactly on the limit
    truncates the same step it reaches it, not one step later), OR it with ``done`` for
    the reset mask, and only then zero the counter for the envs the mask selects —
    zeroing before computing ``truncated``/the mask would make both wrong.
    """
    e = wp.tid()
    sc = step_count[e] + wp.int32(1)
    tr = wp.uint8(0)
    if max_steps > wp.int32(0) and sc >= max_steps:
        tr = wp.uint8(1)
    truncated[e] = tr
    m = wp.uint8(0)
    if done[e] != wp.uint8(0) or tr != wp.uint8(0):
        m = wp.uint8(1)
    reset_mask[e] = m
    if m != wp.uint8(0):
        sc = wp.int32(0)
    step_count[e] = sc


@wp.kernel
def step_count_kernel(
    max_steps: wp.int32,
    step_count: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.uint8),
):
    """Thread per env: advance the step count and decide ``truncated``, nothing else.

    ``max_steps <= 0`` means "no time limit" and writes all-False, exactly as in
    :func:`episode_end_kernel`. The counter is *not* zeroed on truncation here: without
    auto-reset the count keeps running past the limit (and ``truncated`` stays True) until
    the caller resets, which is what the host ``+= 1`` / ``torch.ge`` pair this replaces
    did. When auto-reset is on but running host-side, ``Environment.step``'s own
    ``masked_fill_`` still does the zeroing, after the reset it belongs to.
    """
    e = wp.tid()
    sc = step_count[e] + wp.int32(1)
    tr = wp.uint8(0)
    if max_steps > wp.int32(0) and sc >= max_steps:
        tr = wp.uint8(1)
    truncated[e] = tr
    step_count[e] = sc
