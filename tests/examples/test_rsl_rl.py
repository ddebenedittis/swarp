"""Tests for the rsl_rl baseline trainer's agents-as-envs adapter and checkpoint format.

Not tested: convergence or speed (``swarp.benchmark.train_throughput`` is the speed
measurement). What is pinned is the mapping rsl_rl cannot check for us: which row is which
agent, what the critic sees, how per-env flags reach per-agent rows, and that the adapter's
observation buffers survive the step after the one that returned them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("rsl_rl", reason="examples/train_rsl_rl.py needs the optional [rsl-rl] extra")

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import torch  # noqa: E402
from train_rsl_rl import SwarpVecEnv, load_checkpoint, parse_args, train  # noqa: E402

import swarp  # noqa: E402

E, A = 3, 4


def _venv() -> tuple[swarp.Environment, SwarpVecEnv]:
    env = swarp.make(
        "navigation", n_envs=E, n_agents=A, device="cpu", max_steps=50, seed=0, auto_reset=True
    )
    return env, SwarpVecEnv(env)


def test_observation_layout_and_double_buffering():
    env, venv = _venv()
    td0 = venv.reset()
    obs = env.scenario.observations().clone()
    Ob = env.obs_dim
    assert venv.num_envs == E * A
    assert torch.equal(td0["policy"], obs.reshape(E * A, Ob))
    critic = td0["critic"].view(E, A, A * Ob)
    for e in range(E):
        for i in range(A):
            assert torch.equal(critic[e, i], obs[e].flatten())  # Critic's obs.flatten(-2)
    assert venv.get_observations() is td0

    # rsl_rl stores the obs it acted on only after the next step returns, so that step
    # must not overwrite it.
    before = td0.clone()
    td1, *_ = venv.step(torch.zeros(E * A, env.act_dim))
    assert td1 is not td0
    assert torch.equal(td0["policy"], before["policy"])
    assert torch.equal(td0["critic"], before["critic"])
    assert torch.equal(td1["policy"], env.scenario.observations().reshape(E * A, Ob))


def test_flags_broadcast_per_env_and_actions_are_clamped(monkeypatch):
    env, venv = _venv()
    venv.reset()
    real_step = env.step
    seen = {}
    term = torch.tensor([True, False, False])
    trunc = torch.tensor([True, True, False])

    def step(actions):
        seen["actions"] = actions.clone()
        obs, rew, _, _, info = real_step(actions)
        return obs, rew, term, trunc, info

    monkeypatch.setattr(env, "step", step)
    _, rew, dones, extras = venv.step(torch.full((E * A, env.act_dim), 10.0))

    _, hi = env.action_bounds
    assert torch.equal(seen["actions"], hi.expand(E, A, -1))
    assert rew.shape == (E * A,)
    per_agent = lambda x: x.repeat_interleave(A)  # noqa: E731
    assert torch.equal(dones, per_agent(term | trunc))
    assert torch.equal(extras["time_outs"], per_agent(trunc & ~term))
    assert torch.equal(venv.episode_end, term | trunc)


def test_checkpoint_round_trips_through_eval_mappo(tmp_path):
    pytest.importorskip("torchrl", reason="eval_mappo imports the torchrl example helpers")
    import eval_mappo

    cfg = parse_args([
        "--scenario", "navigation", "--device", "cpu", "--iters", "2", "--n-envs", "4",
        "--steps-per-batch", "4", "--epochs", "1", "--minibatches", "2", "--num-cells", "8",
        "--max-steps", "5", "--eval-every", "0", "--checkpoint-every", "0",
        "--checkpoint-dir", str(tmp_path),
    ])
    path = train(cfg)
    assert path.name == "navigation_final.pt"
    actor, meta = load_checkpoint(path)
    assert (meta["scenario"], meta["n_agents"], meta["max_steps"]) == ("navigation", 4, 5)

    arm, ckpt = eval_mappo._load_arm(str(path), "cpu")
    assert ckpt["format"] == "rsl_rl/1"
    lo, hi = -0.1 * torch.ones(4, ckpt["act_dim"]), 0.1 * torch.ones(4, ckpt["act_dim"])
    obs = torch.randn(3, 4, ckpt["obs_dim"])
    act = arm(3, "cpu", lo, hi)(obs)
    mean = actor.distribution.deterministic_output(actor.mlp(obs))
    assert torch.equal(act, mean.clamp(lo, hi).detach())
    sys.modules.pop("eval_mappo", None)
