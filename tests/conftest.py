"""Shared test fixtures, helpers and marker gating.

Import the helpers explicitly, e.g. ``from conftest import DEVICES``. ``tests/`` is on
``sys.path`` because pytest's prepend import mode inserts a conftest's directory when
it loads it, and this conftest is always loaded before any test module.

Markers
-------
``gpu``   skipped automatically when CUDA is unavailable. Pass a reason if the generic
          one is not informative: ``@pytest.mark.gpu(reason="...")``.
``viz``   skipped automatically when pygame is unavailable. ``tests/render/`` is also
          dropped from collection entirely in that case, because three of its modules
          import pygame at module scope.

Note that the ``DEVICES``-parametrized tests are a *different* pattern: they always run
on CPU and add a CUDA case opportunistically. They are not ``gpu``-marked.

Deliberately NOT shared
-----------------------
Several helpers look duplicated across test modules but diverge in load-bearing ways.
They stay local on purpose; unifying them would silently change what is exercised.

``make_state``
    Four definitions, all four different. ``unit/test_collisions`` takes
    ``(pos, vel=None, dtype)`` and always builds ``n_envs=1``; ``scenarios/
    test_per_env_params`` adds ``n_envs`` and passes 3-D ``pos`` through;
    ``unit/test_geometry`` has no ``vel`` parameter at all; and ``unit/test_gradients``
    is an unrelated function -- ``(n_envs, n_agents, device, dtype, requires_grad,
    seed)`` building a *random seeded* state.

``_env`` / ``make_env``
    ``scenarios/test_transport`` and ``scenarios/test_transport_fused`` share the name
    but differ in arity and semantics (``n_agents`` 3 vs 4, ``seed`` 1 vs 0,
    ``auto_reset`` False vs True, ``max_steps`` None vs 5); same for the pusht pair. The
    ``auto_reset=True, max_steps=5`` of the fused suites is deliberate -- those parity
    rollouts cross auto-resets on purpose. Of the seven ``make_env``s, the two non-render
    ones return ``env`` alone and do *not* call ``reset()`` (``test_environment`` asserts
    pre-reset state), and the render defaults are load-bearing: ``render/
    test_render_mosaic`` needs ``n_envs >= 6`` and ``render/test_render_interaction``
    needs ``n_envs >= 3``.

``_run`` / ``_traj`` / ``run_trajectory``
    Three different action seeds -- 2 in the fused suites (see ``FusedSpec``), 7 in
    ``interop/test_persistent``, 1 in ``interop/test_neighbor_reuse`` -- and
    ``unit/test_determinism`` builds its generator on CPU and moves actions to the
    device on purpose, with ``torch.randn`` rather than ``.uniform_``.

``_controller``
    ``render/test_render_writeback`` takes ``(geometry, camera, **callbacks)`` and
    returns a 2-tuple; the ``**callbacks`` pass-through is what the write-back tests
    assert on. ``render/test_render_interaction`` takes ``(geometry=None, camera=None,
    n_envs=4)``, builds the Camera itself, and returns a 3-tuple.

``holo`` (``unit/test_geometry``)
    Same body as the shared ``holo_cfgs`` but no ``mode`` parameter and ``n`` defaults
    to 1, so it is not substitutable at its call sites.

``_raise``
    ``interop/test_no_host_copies`` raises on host transfer; ``interop/test_torchrl``
    raises on ``wp.capture_begin``. Different messages, different things asserted.
"""

import importlib.util
from dataclasses import dataclass

import pytest
import torch

from swarp import Environment
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.interop.autograd import TorchState

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda:0"] if CUDA else [])

_PYGAME = importlib.util.find_spec("pygame") is not None


# ------------------------------------------------------------------ marker gating


def pytest_ignore_collect(collection_path, config):
    """Drop ``tests/render/`` when pygame is missing.

    ``test_render_interaction``/``_mosaic``/``_writeback`` import pygame at module
    scope, so a ``viz`` skip marker would never get a chance to run.
    """
    if _PYGAME:
        return None
    if collection_path.name == "render" or collection_path.parent.name == "render":
        return True
    return None


_GATES = (
    ("gpu", CUDA, "needs CUDA"),
    ("viz", _PYGAME, "pygame not installed (install the `viz` extra)"),
)


def pytest_collection_modifyitems(config, items):
    for name, available, default_reason in _GATES:
        if available:
            continue
        for item in items:
            marker = item.get_closest_marker(name)
            if marker is not None:
                reason = marker.kwargs.get("reason", default_reason)
                item.add_marker(pytest.mark.skip(reason=reason))


# ------------------------------------------------------------------ state helpers


def _core(state):
    """The five 2D state tensors (drops the optional drone fields)."""
    return (state.pos, state.theta, state.vel, state.speed, state.ang_vel)


def _map5(state, f):
    """Apply ``f`` to the five 2D fields, leaving the drone fields at default."""
    return TorchState(*(f(t) for t in _core(state)))


def holo_cfgs(n, radius=0.1, mode=ControlMode.VELOCITY):
    return [
        AgentConfig(
            model=DynamicsModel.HOLONOMIC,
            ctrl_mode=mode,
            radius=radius,
            max_speed=100.0,
            max_accel=100.0,
        )
        for _ in range(n)
    ]


# ------------------------------------------------------------------ optional deps


def _imageio_available() -> bool:
    try:
        import imageio  # noqa: F401

        return True
    except Exception:
        return False


def _ffmpeg_available() -> bool:
    try:
        import imageio_ffmpeg  # noqa: F401

        return True
    except Exception:
        return False


# -------------------------------------------------------- fused-vs-torch harness


@dataclass(frozen=True)
class FusedSpec:
    """Per-scenario knobs for the shared ``*_fused`` parity harness.

    Every field here differs between scenarios *by design*. None of them may be
    unified into a single default:

    ``fields``
        What each suite's rollout collects and compares. Widths run from 3 to 7, and
        flocking/sampling deliberately do not compare ``done``. Entries are ``"obs"``,
        ``"rew"``, ``"done"``, or ``"info:<key>"``.
    ``grad_steps`` / ``grad_index`` / ``grad_backprop``
        The grad-fallback test backprops through ``rew`` where the reward is
        differentiable, and through ``obs`` where it is not -- discovery, transport and
        pusht rewards are built from coverage counts / discrete latches. Navigation and
        formation additionally run 6 steps with the grad step at index 3 (rather than
        4 / 2) so the shaping baseline is exercised on both sides of it.
    ``action_seed``
        **Load-bearing.** Each suite has a "did the feature actually fire" guard
        (``saw_touch`` / ``saw_crowd`` / ``saw_cover`` / ``saw_reward`` / ``moved`` /
        ``spun``) that passes *because of* its seed. Unify the seeds and those guards
        can pass vacuously -- a silent loss of coverage that no failure reports.
    ``scenario``
        The scenario class, read only for :attr:`rtol` / :attr:`atol`. The fused-vs-torch
        tolerance is declared on the scenario (``Scenario.parity_rtol`` / ``parity_atol``)
        because it is the scenario that knows why its two paths differ; both this harness
        and ``swarp.benchmark``'s parity gate read it from there, so there is exactly one
        number per scenario. A suite may still assert something *tighter* than the declared
        bound for a field it knows more about -- push-t's pose checks do -- but nothing
        re-declares a looser one.
    """

    fields: tuple[str, ...]
    grad_steps: int
    grad_index: int
    grad_backprop: str  # "rew" or "obs"
    scenario: type | None = None
    action_seed: int = 2
    grad_action_seed: int = 5
    substeps: int = 1
    n_envs: int = 24

    @property
    def rtol(self) -> float:
        return self.scenario.parity_rtol

    @property
    def atol(self) -> float:
        return self.scenario.parity_atol


def fused_env(scenario, device, fused, *, spec, dtype=torch.float32):
    """The ``Environment`` settings the seven ``*_fused`` parity suites share.

    ``auto_reset=True`` with ``max_steps=5`` is deliberate and must stay: the parity
    rollouts are 10-20 steps long, so they cross several auto-resets and thereby pin
    the fused kernels' reset path too. Do not "tidy" these towards the settings the
    non-fused ``test_transport``/``test_pusht`` suites use.
    """
    return Environment(
        scenario,
        n_envs=spec.n_envs,
        device=device,
        dt=0.05,
        substeps=spec.substeps,
        seed=0,
        auto_reset=True,
        max_steps=5,
        dtype=dtype,
        fused=fused,
    )


def _pick(name, obs, rew, done, info):
    if name == "obs":
        return obs
    if name == "rew":
        return rew
    if name == "done":
        return done
    return info[name.removeprefix("info:")]


def fused_rollout(env, n_steps, device, n_agents, spec, *, dtype=torch.float32):
    """Seeded no-grad rollout; returns one ``spec.fields``-shaped tuple of clones per step."""
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(spec.action_seed)
    out = []
    with torch.no_grad():
        for _ in range(n_steps):
            a = torch.empty(env.n_envs, n_agents, 2, device=device, dtype=dtype).uniform_(
                -1, 1, generator=gen
            )
            obs, rew, done, info = env.step(a)
            out.append(tuple(_pick(f, obs, rew, done, info).clone() for f in spec.fields))
    return out


def assert_fused_determinism(rollout_a, rollout_b):
    """Two identically seeded fused rollouts must agree bit-for-bit, field by field."""
    for step_a, step_b in zip(rollout_a, rollout_b, strict=True):
        for u, v in zip(step_a, step_b, strict=True):
            assert torch.equal(u, v)


def assert_grad_falls_back_to_torch(env, device, n_agents, spec):
    """A grad-enabled step must take the differentiable torch path (fused off).

    Interleaved with fused no-grad steps: for the shaping scenarios this also pins that
    the shaping baseline (shared ``_prev_dist``) stays continuous across the switch.
    """
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(spec.grad_action_seed)
    for i in range(spec.grad_steps):
        act = torch.empty(env.n_envs, n_agents, 2, device=device).uniform_(-1, 1, generator=gen)
        if i == spec.grad_index:
            act = act.clone().requires_grad_(True)
            with torch.enable_grad():
                obs, rew, done, _ = env.step(act)
            assert obs.requires_grad  # torch reference path was taken
            target = rew if spec.grad_backprop == "rew" else obs
            target.pow(2).sum().backward()
            assert act.grad is not None
        else:
            with torch.no_grad():
                obs, rew, done, _ = env.step(act)
            assert torch.isfinite(rew).all()
