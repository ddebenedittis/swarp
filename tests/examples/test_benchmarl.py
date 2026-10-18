"""Tests for the BenchMARL baseline (``examples/train_benchmarl.py``).

CPU only and seconds long: the env layout BenchMARL reads, the task enum against the recipe
table, and that a saved checkpoint, loaded the way ``eval_mappo.py`` loads it, acts exactly
like BenchMARL's own deterministic policy. Training itself is not run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("benchmarl", reason="needs the optional [benchmarl] extra (own venv)")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import torch  # noqa: E402
from torchrl.envs.utils import ExplorationType, check_env_specs, set_exploration_type  # noqa: E402
from train_benchmarl import (  # noqa: E402
    GROUP,
    GroupedSwarpEnv,
    SwarpTask,
    build_experiment,
    parse_args,
)

import swarp  # noqa: E402
from swarp.rl.recipes import SPECS  # noqa: E402


def test_task_enum_covers_specs():
    assert sorted(t.name.lower() for t in SwarpTask) == sorted(SPECS)


@pytest.mark.parametrize("scenario", ["navigation", "caging"])
def test_grouped_env_specs(scenario):
    """Caging's info mixes per-env ``[E]``, per-agent ``[E, A]`` and ``[E, 1, 2]`` tensors."""
    spec = SPECS[scenario]
    env = GroupedSwarpEnv(
        swarp.make(
            scenario, n_envs=4, device="cpu", dt=spec.dt, substeps=1, max_steps=5,
            n_agents=spec.n_agents, **spec.scen_kwargs,
        )
    )
    check_env_specs(env)
    # Crosses the max_steps=5 boundary, so the collector-style partial reset runs too.
    td = env.rollout(8, break_when_any_done=False)
    assert td.get(GROUP).batch_size == (4, 8, spec.n_agents)
    assert td.get(("next", "done")).shape == (4, 8, 1)


def test_checkpoint_acts_like_benchmarl(tmp_path):
    """``eval_mappo``'s greedy arm on a saved checkpoint is BenchMARL's deterministic action."""
    import eval_mappo

    cfg = parse_args([
        "--scenario", "navigation", "--device", "cpu", "--n-envs", "4",
        "--steps-per-batch", "4", "--num-cells", "8", "--checkpoint-dir", str(tmp_path),
    ])
    experiment, callback = build_experiment(cfg)
    try:
        path = callback.save("test")
        env = callback.env
        td = env.reset()
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            expected = experiment.policy(td.clone()).get((GROUP, "action"))
        arm, ckpt = eval_mappo._load_arm(str(path), "cpu")
        lo, hi = env._env.action_bounds
        got = arm["policy"](4, "cpu", lo, hi)(td.get((GROUP, "observation")))
    finally:
        experiment.close()
    assert ckpt["format"] == "benchmarl/1"
    torch.testing.assert_close(got, expected)
