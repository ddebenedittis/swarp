"""Per-simulator adapters for the cross-sim throughput benchmark.

Each adapter module exposes ``build(scenario, n_envs, n_agents, device, seed=0)
-> Runner``. A ``Runner`` hides every backend difference behind two callables:

* ``rollout(n)`` — advance the batched simulation ``n`` steps, leaving nothing
  pending on the device.
* ``sync()`` — block until the device is idle (``cuda.synchronize`` for torch,
  ``jax.block_until_ready`` for JAX).

The interface is ``rollout(n)`` rather than a per-step ``step()`` because JAX
must run its time loop inside a single jitted ``lax.scan`` to be timed fairly —
a per-step Python loop would pay trace/dispatch overhead every step and grossly
understate JAX throughput. Torch backends just wrap the existing Python step
loop, so their numbers stay identical to ``compare_vmas.py``.

Heavy imports (torch, jax, vmas, jaxmarl, camar) live *inside* the adapter
modules, never here, so the orchestrator can import this package without pulling
in any conflicting numpy/jax/torch stack.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


class CpuOnlyError(RuntimeError):
    """Raised by a JAX adapter when no GPU device is available.

    JAX silently falls back to CPU when a CUDA-enabled jaxlib is missing, which
    would make a "throughput" number meaningless; the orchestrator maps this to
    a ``CPU-only`` cell rather than reporting a bogus figure.
    """


@dataclass
class Runner:
    rollout: Callable[[int], None]  # advance n steps; leave nothing pending on device
    sync: Callable[[], None]  # block until device idle
    n_envs: int
    n_agents: int  # ACTUAL agents realized by the sim (may differ from requested)
    backend: str  # "torch" | "jax"
    close: Callable[[], None] = field(default=lambda: None)


def make_torch_runner(env, step, n_envs: int, n_agents: int, device: str) -> Runner:
    """Wrap a torch ``(env, step)`` pair (as returned by ``compare_vmas`` builders)."""
    import torch

    def rollout(n: int) -> None:
        with torch.no_grad():
            for _ in range(n):
                step()

    def sync() -> None:
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    def close() -> None:
        nonlocal env
        del env
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    return Runner(rollout, sync, n_envs, n_agents, "torch", close)


def assert_jax_gpu() -> None:
    """Raise :class:`CpuOnlyError` unless JAX sees a GPU device."""
    import jax

    if jax.devices()[0].platform != "gpu":
        raise CpuOnlyError("JAX fell back to CPU (no CUDA-enabled jaxlib)")


def make_jax_runner(
    *,
    reset_one: Callable,
    step_one: Callable,
    sample_actions: Callable,
    n_envs: int,
    n_agents: int,
    seed: int = 0,
) -> Runner:
    """Build a JAX ``Runner`` that vmaps over envs and scans over time.

    ``reset_one(key)`` / ``step_one(key, state, actions)`` are the sim's
    single-env functions (they get ``vmap``-ed over the leading env axis).
    ``sample_actions(key, n_envs)`` returns a batched-actions pytree with a
    leading ``n_envs`` axis. The scan length is baked into a per-``n`` jitted
    function, so the first call for a given ``n`` compiles; the benchmark warms
    up with the timed length so compilation never lands inside the timed region.
    """
    import jax
    from jax import lax, random, vmap

    reset_v = vmap(reset_one)
    step_v = vmap(step_one)

    key = random.PRNGKey(seed)
    key, rk = random.split(key)
    _, state0 = reset_v(random.split(rk, n_envs))

    def one_step(carry, _):
        state, k, acc = carry
        k, ak, sk = random.split(k, 3)
        actions = sample_actions(ak, n_envs)
        obs, state, reward, _, _ = step_v(random.split(sk, n_envs), state, actions)
        # Fold a reduction of obs+reward into the carry. Without this, XLA's
        # dead-code elimination deletes the observation/reward computation
        # entirely (they don't feed the next state), so the scan would time
        # physics only — unfair vs wmas/VMAS, which always compute obs+reward.
        acc = acc + sum(jax.numpy.sum(x) for x in jax.tree_util.tree_leaves((obs, reward)))
        return (state, k, acc), None

    compiled: dict[int, Callable] = {}

    def _scan_fn(n: int) -> Callable:
        fn = compiled.get(n)
        if fn is None:
            fn = jax.jit(lambda st, k, a: lax.scan(one_step, (st, k, a), None, length=n)[0])
            compiled[n] = fn
        return fn

    box = {"state": state0, "key": key, "acc": jax.numpy.float32(0.0)}

    def rollout(n: int) -> None:
        box["state"], box["key"], box["acc"] = _scan_fn(n)(
            box["state"], box["key"], box["acc"]
        )

    def sync() -> None:
        jax.block_until_ready(box["acc"])

    return Runner(rollout, sync, n_envs, n_agents, "jax")
