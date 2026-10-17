"""The lean MAPPO trainer in ``swarp.rl``: its maths, its graphs, and its plumbing."""

from __future__ import annotations

import csv
import math
from dataclasses import replace

import pytest
import torch

from swarp.rl import MAPPO, Actor, Critic, load_checkpoint, parse_args, train
from swarp.rl.networks import tanh_normal_log_prob
from swarp.rl.ppo import gae
from swarp.rl.recipes import SPECS, build_env

SMOKE = [
    "--device", "cpu", "--iters", "2", "--n-envs", "8", "--steps-per-batch", "4",
    "--max-steps", "5", "--epochs", "1", "--minibatches", "1", "--num-cells", "8",
    "--substeps", "1", "--eval-every", "0", "--checkpoint-every", "0",
]


def test_gae_matches_hand_computation():
    """One env, one agent, three steps: a plain step, a timeout, then a termination."""
    g, lam = 0.9, 0.8
    rew = torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1)
    v = torch.tensor([0.5, 0.25, 1.5, 4.0]).view(4, 1, 1)
    term = torch.tensor([0.0, 0.0, 1.0]).view(3, 1)
    trunc = torch.tensor([0.0, 1.0, 0.0]).view(3, 1)
    adv = gae(rew, v, term, trunc, g, lam).flatten().tolist()
    # t=2 terminated: no bootstrap, trace cut.
    a2 = 3.0 - 1.5
    # t=1 truncated: bootstraps from its own value V(s_1), trace cut (s_2 is a new episode).
    a1 = (2.0 + g * 0.25) - 0.25
    # t=0 ordinary: bootstraps from V(s_1) and inherits t=1's advantage.
    a0 = (1.0 + g * 0.25 - 0.5) + g * lam * a1
    assert adv == pytest.approx([a0, a1, a2])


def test_tanh_normal_log_prob_matches_torch_distributions():
    torch.manual_seed(0)
    loc, scale = torch.randn(5, 3, 2), torch.rand(5, 3, 2) + 0.1
    u = loc + scale * torch.randn(5, 3, 2)
    dist = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(loc, scale), [torch.distributions.TanhTransform()]
    )
    ref = dist.log_prob(torch.tanh(u)).sum(-1)
    assert torch.allclose(tanh_normal_log_prob(u, loc, scale), ref, atol=1e-4)


def test_actor_scale_is_one_at_zero_output():
    """tensordict's ``biased_softplus_1.0``: a zero pre-activation means std 1."""
    actor = Actor(obs_dim=3, act_dim=2, num_cells=4)
    torch.nn.init.zeros_(actor.net[-1].weight)
    torch.nn.init.zeros_(actor.net[-1].bias)
    _, scale = actor(torch.randn(6, 2, 3))
    assert torch.allclose(scale, torch.ones_like(scale))


def test_actor_maps_onto_a_non_unit_action_box():
    """Ring road's box is accel +/-2, steer +/-0.6: the greedy action spans it exactly, and
    the unit box stays the plain tanh (no extra op, same numbers as before boxes existed)."""
    low, high = torch.tensor([-2.0, -0.6]), torch.tensor([2.0, 0.6])
    actor = Actor(obs_dim=3, act_dim=2, num_cells=4, low=low, high=high)
    big = torch.tensor([[50.0, -50.0], [-50.0, 50.0]]).view(2, 1, 2)
    expect = torch.tensor([[2.0, -0.6], [-2.0, 0.6]]).view(2, 1, 2)
    assert torch.allclose(actor.squash(big), expect, atol=1e-6)
    assert torch.equal(actor.squash(torch.zeros(1, 1, 2)), torch.zeros(1, 1, 2))
    unit = Actor(obs_dim=3, act_dim=2, num_cells=4)
    assert unit.unit_box
    u = torch.randn(4, 2, 2)
    assert torch.equal(unit.squash(u), torch.tanh(u))
    assert "act_mid" not in actor.state_dict()  # checkpoints from before boxes still load


def test_train_smoke_on_a_non_unit_box_stores_the_box(tmp_path):
    cfg = parse_args(["--scenario", "ringroad", *SMOKE, "--checkpoint-dir", str(tmp_path)])
    actor, _, meta = load_checkpoint(train(cfg))
    assert meta["act_low"] == pytest.approx([-2.0, -0.6])
    assert meta["act_high"] == pytest.approx([2.0, 0.6])
    act = actor.greedy(torch.randn(5, 1, actor.obs_dim))
    assert (act.abs() <= torch.tensor([2.0, 0.6]) + 1e-6).all()


def test_train_smoke_writes_metrics_and_a_loadable_checkpoint(tmp_path):
    cfg = parse_args(["--scenario", "navigation", *SMOKE, "--checkpoint-dir", str(tmp_path)])
    path = train(cfg)
    assert path.name == "navigation_final.pt"
    with (tmp_path / "metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert {"env_sps", "ep_solve_rate", "entropy", "grad_skips", "dist_to_goal"} <= set(rows[0])
    assert not math.isnan(float(rows[0]["reward_per_step"]))

    actor, _, meta = load_checkpoint(path)
    assert (meta["scenario"], meta["max_steps"], meta["n_agents"]) == ("navigation", 5, 4)
    obs = torch.randn(3, 4, actor.obs_dim)
    assert torch.equal(actor.greedy(obs), actor.greedy(obs))


def test_eval_mappo_reads_a_swarp_rl_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("torchrl")
    import sys
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    import eval_mappo

    cfg = parse_args(["--scenario", "navigation", *SMOKE, "--checkpoint-dir", str(tmp_path)])
    arm, ckpt = eval_mappo._load_arm(str(train(cfg)), "cpu")
    assert ckpt["scenario"] == "navigation"
    act = arm(3, "cpu", None, None)(torch.zeros(3, 4, ckpt["obs_dim"]))
    assert act.shape == (3, 4, ckpt["act_dim"]) and act.abs().max() <= 1.0
    sys.modules.pop("eval_mappo", None)


# ---------------------------------------------------------------------------- CUDA graphs


def _algo(graphs: bool, seed: int = 0, **spec_kw) -> MAPPO:
    spec = replace(
        SPECS["navigation"], n_envs=64, steps_per_batch=8, epochs=2, minibatches=4,
        num_cells=32, **spec_kw,
    )
    env = build_env(
        "navigation", n_envs=spec.n_envs, n_agents=spec.n_agents, device="cuda:0",
        dt=spec.dt, substeps=spec.substeps, max_steps=spec.max_steps, seed=seed,
        env_kwargs={"auto_reset": True},
    )
    torch.manual_seed(seed)
    actor = Actor(env.obs_dim, env.act_dim, spec.num_cells).cuda()
    critic = Critic(env.obs_dim, env.n_agents, spec.num_cells).cuda()
    algo = MAPPO(env, actor, critic, spec, graphs=graphs)
    algo.reset()
    return algo


@pytest.mark.gpu
def test_captured_rollout_fills_consistent_slots():
    """Every slot's stored log-prob is the actor's log-prob of that slot's (obs, u)."""
    algo = _algo(graphs=True)
    for _ in range(2):  # the second rollout replays graphs captured in the first
        algo.rollout()
    assert algo.t.item() == algo.T
    with torch.no_grad():
        loc, scale = algo.actor(algo.obs_buf[: algo.T])
        ref = tanh_normal_log_prob(algo.u_buf, loc, scale)
    assert torch.allclose(algo.logp_buf, ref, atol=1e-4)
    assert torch.equal(algo.obs_buf[algo.T], algo._out[0])


def _copy_rollout(dst: MAPPO, src: MAPPO) -> None:
    for name in ("obs_buf", "u_buf", "logp_buf", "rew_buf", "term_buf", "trunc_buf"):
        getattr(dst, name).copy_(getattr(src, name))


@pytest.mark.gpu
def test_captured_update_matches_eager_update():
    """Same weights, same rollout, same permutations: the graphed update lands where the
    eager one does. Entropy is off, since its sample is the one draw the two cannot share."""
    eager, graphed = _algo(graphs=False), _algo(graphs=True)
    graphed.actor.load_state_dict(eager.actor.state_dict())
    graphed.critic.load_state_dict(eager.critic.state_dict())
    eager.rollout()
    _copy_rollout(graphed, eager)
    for algo in (eager, graphed):
        algo.ent_coef.fill_(0.0)
        algo._prepare_body()  # eager on both, so the permutations can be shared
    graphed.perms.copy_(eager.perms)
    for _ in range(eager.spec.epochs * eager.M):
        eager._minibatch()
        graphed._minibatch()
    for a, b in zip(eager.params, graphed.params, strict=True):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
    assert graphed.mb_stats[3].item() == 0.0


@pytest.mark.gpu
def test_non_finite_minibatch_is_skipped_on_device():
    """A NaN advantage poisons every minibatch; the weights and Adam state must not move."""
    algo = _algo(graphs=True)
    algo.rollout()
    algo.update(algo.spec.entropy_coeff)  # one healthy update: Adam state exists
    before = [p.detach().clone() for p in algo.params]
    state = [s["exp_avg"].clone() for s in algo.optim.state.values()]
    steps = [s["step"].clone() for s in algo.optim.state.values()]
    algo.rollout()
    algo._prepare()
    algo.adv_buf.fill_(float("nan"))
    n_mb = algo.spec.epochs * algo.M
    for _ in range(n_mb):
        algo._minibatch()
    assert algo.mb_stats[3].item() == n_mb
    for p, q in zip(algo.params, before, strict=True):
        assert torch.equal(p, q)
    for s, a, n in zip(algo.optim.state.values(), state, steps, strict=True):
        assert torch.equal(s["exp_avg"], a) and torch.equal(s["step"], n)
