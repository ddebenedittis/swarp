"""wmas adapter: reuses the ``_make_wmas`` navigation builder from ``compare_vmas``."""

from __future__ import annotations

from wmas.benchmark._adapters import Runner, make_torch_runner
from wmas.benchmark.compare_vmas import _make_wmas


def build(scenario: str, n_envs: int, n_agents: int, device: str, seed: int = 0) -> Runner:
    if scenario != "navigation":
        raise ValueError(f"wmas adapter only implements 'navigation', got {scenario!r}")
    env, step = _make_wmas(n_envs, n_agents, device)
    return make_torch_runner(env, step, n_envs, env.n_agents, device)
