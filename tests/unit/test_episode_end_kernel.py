"""``episode_end_kernel``: the captured replacement for ``Environment.step``'s host tail.

Pins the kernel against the exact host arithmetic it replaces --

    step_count += 1
    truncated = step_count >= max_steps      # all-False when max_steps is None
    episode_end = terminated | truncated
    step_count.masked_fill_(episode_end, 0)

-- rather than against ``Environment`` end to end, so a regression here points straight
at the kernel instead of somewhere in the graph-composition plumbing that also has its
own tests (``tests/interop/test_persistent.py``).
"""

import torch
import warp as wp

from swarp.core.episode_kernels import episode_end_kernel

wp.init()


def _run_kernel(done: torch.Tensor, max_steps: int | None, step_count: torch.Tensor):
    """Launch the kernel on CPU torch buffers, returning (step_count, truncated, mask)."""
    n = done.shape[0]
    step_count = step_count.clone()
    truncated = torch.zeros(n, dtype=torch.bool)
    reset_mask = torch.zeros(n, dtype=torch.uint8)
    wp.launch(
        episode_end_kernel,
        dim=n,
        inputs=[
            wp.from_torch(done.to(torch.uint8).contiguous()),
            wp.int32(max_steps if max_steps is not None else 0),
            wp.from_torch(step_count),
            wp.from_torch(truncated.view(torch.uint8)),
            wp.from_torch(reset_mask),
        ],
        device="cpu",
    )
    return step_count, truncated, reset_mask.bool()


def _host_reference(done: torch.Tensor, max_steps: int | None, step_count: torch.Tensor):
    """The exact host arithmetic ``Environment.step`` used to run every step."""
    step_count = step_count.clone() + 1
    if max_steps is not None:
        truncated = torch.ge(step_count, max_steps)
    else:
        truncated = torch.zeros_like(done)
    episode_end = torch.logical_or(done, truncated)
    step_count.masked_fill_(episode_end, 0)
    return step_count, truncated, episode_end


def _check(done, max_steps, step_count):
    got_sc, got_tr, got_mask = _run_kernel(done, max_steps, step_count)
    want_sc, want_tr, want_mask = _host_reference(done, max_steps, step_count)
    assert torch.equal(got_sc, want_sc)
    assert torch.equal(got_tr, want_tr)
    assert torch.equal(got_mask, want_mask)


def test_matches_host_arithmetic_time_limit_only():
    n = 32
    step_count = torch.arange(n, dtype=torch.int32) % 5  # some at/near the limit
    done = torch.zeros(n, dtype=torch.bool)
    _check(done, 5, step_count)


def test_matches_host_arithmetic_terminated_only():
    """``done`` alone must trigger a reset even when nothing has hit the time limit."""
    n = 16
    step_count = torch.zeros(n, dtype=torch.int32)
    done = torch.arange(n) % 3 == 0
    _check(done, None, step_count)
    assert not _run_kernel(done, None, step_count)[1].any()  # never truncates


def test_matches_host_arithmetic_staggered_and_mixed():
    """A realistic mix: different envs at different counts, some done, some at the limit."""
    torch.manual_seed(0)
    n = 200
    max_steps = 17
    step_count = torch.randint(0, max_steps, (n,), dtype=torch.int32)
    done = torch.rand(n) < 0.3
    _check(done, max_steps, step_count)


def test_reaching_the_limit_truncates_the_same_step():
    """A step that lands exactly on ``max_steps`` truncates immediately, not one step
    later -- the kernel derives ``truncated`` from the *incremented* count."""
    step_count = torch.tensor([4], dtype=torch.int32)
    done = torch.tensor([False])
    got_sc, got_tr, got_mask = _run_kernel(done, 5, step_count)
    assert bool(got_tr[0])
    assert bool(got_mask[0])
    assert int(got_sc[0]) == 0


def test_no_reset_leaves_the_counter_advancing():
    step_count = torch.tensor([1, 2], dtype=torch.int32)
    done = torch.tensor([False, False])
    got_sc, got_tr, got_mask = _run_kernel(done, 10, step_count)
    assert torch.equal(got_sc, torch.tensor([2, 3], dtype=torch.int32))
    assert not got_tr.any()
    assert not got_mask.any()
