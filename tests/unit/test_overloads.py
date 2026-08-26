"""Every kernel on the step path is launched as a *concrete* overload.

``wp.launch`` re-resolves a generic kernel on every call -- it re-infers the argument
types and rebuilds a signature string to look the overload up (``Kernel.add_overload``).
That is a constant being recomputed per step, and on the navigation hot path it measured
~65% of the cost of a launch.

:mod:`swarp._overloads` fixes it by keeping what ``wp.overload`` returns and dispatching
it at launch time. The invariant that keeps it fixed is the one pinned here: nothing
generic reaches ``wp.launch``. A new kernel that forgets ``register``/``concrete`` still
*works* -- it just silently pays the resolution again -- so a test is the only thing that
notices.
"""

from typing import Any

import pytest
import torch
import warp as wp
from conftest import DEVICES

from swarp import Environment, NavigationScenario
from swarp._overloads import concrete, register


@pytest.fixture
def record_launches(monkeypatch):
    """Collect ``(kernel_key, is_generic)`` for every kernel launched while active."""
    seen: list[tuple[str, bool]] = []
    original = wp.launch

    def recording(kernel, *args, **kwargs):
        seen.append((kernel.key, bool(getattr(kernel, "is_generic", False))))
        return original(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", recording)
    return seen


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_graph", [False, True])
def test_step_launches_nothing_generic(device, use_graph, record_launches):
    """The whole step -- physics, fused obs/reward, and the auto-reset pass."""
    scenario = NavigationScenario(n_agents=8)
    env = Environment(
        scenario,
        n_envs=64,
        device=device,
        seed=0,
        use_graph=use_graph,
        auto_reset=True,
        max_steps=50,
    )
    try:
        env.reset(seed=0)
        actions = torch.zeros(64, 8, 2, device=device, dtype=env.dtype)
        record_launches.clear()
        with torch.no_grad():
            for _ in range(3):
                env.step(actions)
    finally:
        env.close()

    assert record_launches, "no kernels launched -- the probe is not wired up"
    generic = sorted({key for key, is_generic in record_launches if is_generic})
    assert not generic, f"generic kernels still resolved per launch: {generic}"


def test_concrete_falls_back_to_the_generic_kernel():
    """An unregistered key returns the kernel itself, so a miss is slow, never broken."""

    @wp.kernel
    def _unregistered(x: wp.array(dtype=wp.float32)):
        i = wp.tid()
        x[i] = x[i]

    assert concrete(_unregistered, wp.float32) is _unregistered


def test_register_returns_a_concrete_instantiation():
    """``register`` hands back the non-generic kernel and ``concrete`` finds it again."""

    @wp.kernel
    def _generic(x: wp.array(dtype=Any), y: Any):  # noqa: F821
        i = wp.tid()
        x[i] = y

    instance = register(_generic, wp.float32, [wp.array(dtype=wp.float32), wp.float32])
    assert _generic.is_generic
    assert not instance.is_generic
    assert concrete(_generic, wp.float32) is instance
