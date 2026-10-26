"""The lean MAPPO trainer in ``swarp.rl``: its maths, its graphs, and its plumbing."""

from __future__ import annotations

import copy
import csv
import gc
import math
from dataclasses import replace
from functools import partial

import pytest
import torch

from swarp.rl import MAPPO, Actor, Critic, load_checkpoint, parse_args, train
from swarp.rl.networks import PolicyNet, StateCritic, role_permutation, tanh_normal_log_prob
from swarp.rl.ppo import (
    LEGACY_STAT_NAMES,
    check_resume,
    freeze_actor,
    gae,
    greedy_env_arm,
    greedy_score_kwargs,
    load_widened,
    network_archs,
    policy_outputs,
    ppo_loss,
    standardize_advantage,
    tie_learned_roles,
)
from swarp.rl.recipes import SPECS, Ramp, SolveRateWindow, build_env, scenario_name, score

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


def test_advantages_are_standardized_per_role_slot():
    """Roles move between agents per row: each role-sorted slot gets its own statistics,
    not each agent index, and the result comes back in agent order."""
    torch.manual_seed(0)
    B, counts = 64, (1, 3)
    base = torch.tensor([0, 1, 1, 1])
    roles = torch.stack([base[torch.randperm(4)] for _ in range(B)])
    # Role 0 is worth 10x the others: per-agent-index statistics would mix the two.
    adv = torch.randn(B, 4) * torch.where(roles == 0, 10.0, 1.0) + 3.0 * roles
    out = standardize_advantage(adv, roles, counts)
    perm = torch.argsort(roles, dim=-1, stable=True)
    ranked = adv.gather(-1, perm)
    ref = (ranked - ranked.mean(0)) / ranked.std(0)
    assert torch.allclose(out.gather(-1, perm), ref, atol=1e-5)
    assert torch.equal(perm, role_permutation(roles, counts))
    # Without roles (or counts) it is the per-agent-index formula, unchanged.
    plain = (adv - adv.mean(0)) / adv.std(0)
    assert torch.equal(standardize_advantage(adv), plain)
    assert torch.equal(standardize_advantage(adv, roles, ()), plain)


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
    assert meta["format"] == "swarp.rl/2"
    assert (meta["actor_cls"], meta["critic_cls"]) == ("Actor", "Critic")
    assert (meta["state_dim"], meta["role_counts"], meta["aux_dim"]) == (0, [], 0)
    obs = torch.randn(3, 4, actor.obs_dim)
    assert torch.equal(actor.greedy(obs), actor.greedy(obs))


def test_ramp_quantizes_rounds_and_holds():
    r = Ramp("x", (0.0, 0), (1.0, 20), start=10, end=20, steps=4)
    assert r.value(0) == r.value(12) == (0.0, 0)
    assert r.value(13) == (0.25, 5)
    assert r.value(20) == r.value(99) == (1.0, 20)
    assert isinstance(r.value(13)[1], int)


def test_train_applies_ramps_and_refuses_a_typo(tmp_path):
    cfg = parse_args(["--scenario", "pusht", *SMOKE, "--iters", "4", "--lr-final", "1e-5",
                      "--checkpoint-dir", str(tmp_path)])
    cfg.spec = replace(cfg.spec, ramps=(Ramp("rot_away_penalty", 0.0, 1.0, 0, 2, steps=2),))
    train(cfg)
    with (tmp_path / "metrics.csv").open() as fh:
        assert [r["ramp"] for r in csv.DictReader(fh)] == ["0.000", "0.500", "1.000", "1.000"]

    cfg.spec = replace(cfg.spec, ramps=(Ramp("no_such_attr", 0.0, 1.0, 0, 2),))
    with pytest.raises(ValueError, match="no_such_attr"):
        train(cfg)


def test_train_with_a_policynet_and_a_per_agent_critic_round_trips(tmp_path):
    """Navigation has no roles, state or aux targets, so this is PolicyNet's plain path and
    an obs-reading per-agent StateCritic, through train(), its checkpoint and the resume."""
    flags = ["--scenario", "navigation", *SMOKE, "--checkpoint-dir", str(tmp_path),
             "--actor", "policynet", "--critic-input", "obs", "--per-agent-value"]
    path = train(parse_args(flags))
    actor, critic, meta = load_checkpoint(path)
    assert isinstance(actor, PolicyNet) and isinstance(critic, StateCritic)
    assert (meta["actor_cls"], meta["critic_cls"]) == ("PolicyNet", "StateCritic")
    assert critic.per_agent and critic.use_obs and critic.state_dim == 0
    obs = torch.randn(3, 4, actor.obs_dim)
    assert critic(obs).shape == (3, 4) and actor.greedy(obs).shape == (3, 4, 2)
    resumed = parse_args([*flags, "--resume", str(path), "--resume-critic-only"])
    assert resumed.resume_critic_only
    train(resumed)


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        (["--critic-input", "state"], "no critic_state"),
        (["--aux-coef", "0.1"], "needs actor='policynet'"),
        (["--actor", "policynet", "--aux-coef", "0.1"], "no aux_targets"),
        (["--recurrent"], "recurrent=True needs actor='policynet'"),
        (["--comm", "attn"], "needs actor='policynet'"),
        (["--actor", "policynet", "--recurrent", "--seq-len", "3"], "must divide"),
        (["--actor", "policynet", "--seq-len", "2"], "needs recurrent=True"),
        (["--actor", "policynet", "--privileged-actor"], "privileged_actor reads the aux"),
    ],
)
def test_train_refuses_options_the_scenario_cannot_feed(tmp_path, flags, match, capsys):
    argv = ["--scenario", "navigation", *SMOKE, "--checkpoint-dir", str(tmp_path), *flags]
    try:
        cfg = parse_args(argv)
    except SystemExit:  # rejected by the spec itself, as a CLI error
        assert match in capsys.readouterr().err
        return
    with pytest.raises(ValueError, match=match):
        train(cfg)


def test_resume_refuses_a_checkpoint_of_other_networks(tmp_path):
    """A policynet checkpoint resumed under the default flags: the error names the
    mismatch and the flags that rebuild it, rather than a state-dict key error."""
    flags = ["--scenario", "navigation", *SMOKE, "--checkpoint-dir", str(tmp_path)]
    path = train(parse_args([*flags, "--actor", "policynet", "--per-agent-value"]))
    with pytest.raises(ValueError, match=r"actor \(checkpoint vs flags: class PolicyNet") as e:
        train(parse_args([*flags, "--resume", str(path)]))
    msg = str(e.value)
    assert "--actor policynet --role-params heads" in msg
    assert "--critic-input obs --per-agent-value" in msg
    # Critic only: the actor may differ, the critic may not.
    with pytest.raises(ValueError, match="critic") as e:
        train(parse_args([*flags, "--resume", str(path), "--resume-critic-only"]))
    assert "actor (" not in str(e.value)
    critic_only = ["--resume", str(path), "--resume-critic-only", "--per-agent-value"]
    train(parse_args([*flags, *critic_only]))
    with pytest.raises(ValueError, match="num_cells 8 vs 16"):
        train(parse_args([*flags, "--resume", str(path), "--actor", "policynet",
                          "--per-agent-value", "--num-cells", "16"]))


def test_spec_network_defaults_reproduce_the_plain_trainer():
    from swarp.rl.recipes import TrainSpec

    spec = TrainSpec()
    assert (spec.actor, spec.critic_input, spec.per_agent_value, spec.aux_coef) == (
        "mlp", "obs", False, 0.0,
    )
    # Every recipe that predates the training hooks keeps the plain networks; the lidar
    # Push-T recipes are the ones written for them.
    assert all(
        (s.actor, s.critic_input, s.per_agent_value, s.aux_coef) == ("mlp", "obs", False, 0.0)
        for name, s in SPECS.items()
        if scenario_name(name) != "pusht_lidar"
    )
    s = SPECS["pusht_lidar"]
    assert (s.actor, s.role_params, s.critic_input, s.per_agent_value) == (
        "policynet", "heads", "state", True,
    )
    assert all((s.recurrent, s.seq_len, s.comm) == (False, 0, "none") for s in SPECS.values())


def test_a_v2_checkpoint_round_trips_a_role_aware_architecture(tmp_path):
    torch.manual_seed(0)
    counts = (1, 2)
    actor = PolicyNet(
        5, 2, 8, role_counts=counts, role_params="separate", aux_dim=2,
        low=[-2.0, -1.0], high=[2.0, 1.0],
    )
    critic = StateCritic(4, 5, 3, 8, use_obs=True, role_counts=counts, per_agent=True)
    path = tmp_path / "v2.pt"
    torch.save(
        {
            "format": "swarp.rl/2", "actor_cls": "PolicyNet", "actor_kwargs": actor.config,
            "critic_cls": "StateCritic", "critic_kwargs": critic.config,
            "actor": actor.state_dict(), "critic": critic.state_dict(),
            "state_dim": 4, "role_counts": list(counts), "aux_dim": 2,
        },
        path,
    )
    a, c, meta = load_checkpoint(path)
    assert isinstance(a, PolicyNet) and isinstance(c, StateCritic)
    assert a.role_counts == counts and a.role_params == "separate" and not a.unit_box
    obs, state, roles = torch.randn(6, 3, 5), torch.randn(6, 4), torch.tensor([[1, 0, 1]] * 6)
    assert torch.equal(a.greedy(obs, roles), actor.greedy(obs, roles))
    aux = [net.step(obs, roles, with_aux=True)[2] for net in (a, actor)]
    assert torch.equal(*aux)
    assert torch.equal(c(obs, state, roles), critic(obs, state, roles))


def test_a_hand_built_v1_checkpoint_still_loads(tmp_path):
    """The exact key set ``swarp.rl/1`` wrote: no class names, rebuilt from its dims."""
    torch.manual_seed(0)
    actor, critic = Actor(5, 2, 8, low=torch.tensor([-2.0, -1.0])), Critic(5, 3, 8)
    path = tmp_path / "v1.pt"
    torch.save(
        {
            "format": "swarp.rl/1", "actor": actor.state_dict(), "critic": critic.state_dict(),
            "obs_dim": 5, "act_dim": 2, "n_agents": 3, "num_cells": 8,
            "act_low": [-2.0, -1.0], "act_high": [1.0, 1.0], "scenario": "navigation",
            "scen_kwargs": {}, "dt": 0.05, "substeps": 4, "max_steps": 200, "difficulty": 1.0,
        },
        path,
    )
    a, c, meta = load_checkpoint(path)
    assert type(a) is Actor and type(c) is Critic and meta["format"] == "swarp.rl/1"
    obs = torch.randn(6, 3, 5)
    assert torch.equal(a.greedy(obs), actor.greedy(obs))
    assert torch.equal(c(obs), critic(obs))


def test_eval_mappo_reads_a_swarp_rl_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("torchrl")
    import sys
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    import eval_mappo

    cfg = parse_args(["--scenario", "navigation", *SMOKE, "--checkpoint-dir", str(tmp_path)])
    arm, ckpt = eval_mappo._load_arm(str(train(cfg)), "cpu")
    assert ckpt["scenario"] == "navigation"
    act = arm["policy"](3, "cpu", None, None)(torch.zeros(3, 4, ckpt["obs_dim"]))
    assert act.shape == (3, 4, ckpt["act_dim"]) and act.abs().max() <= 1.0
    sys.modules.pop("eval_mappo", None)


# ---------------------------------------------------------------------------- CUDA graphs


ROLE_COUNTS = (1, 3)
AUX_DIM = 3


def _patch_training_hooks(env) -> None:
    """Give navigation the three training hooks, all fixed-address: the state is the obs
    buffer flattened, the roles a constant per-env buffer (shuffled, so the role sort is
    not the identity), the aux targets a slice of the obs buffer."""
    scen, E, A = env.scenario, env.n_envs, env.n_agents
    gen = torch.Generator().manual_seed(7)
    base = torch.tensor([r for r, c in enumerate(ROLE_COUNTS) for _ in range(c)])
    roles = torch.stack([base[torch.randperm(A, generator=gen)] for _ in range(E)])
    roles = roles.to(env.device, torch.int32)
    scen.state_dim = A * env.obs_dim
    scen.critic_state = lambda: scen.observations().flatten(-2)
    scen.n_roles, scen.role_counts = len(ROLE_COUNTS), ROLE_COUNTS
    scen.agent_roles = lambda: roles
    scen.aux_dim = AUX_DIM
    scen.aux_targets = lambda: scen.observations()[..., :AUX_DIM]


def _patch_single_role(env) -> None:
    """:func:`_patch_training_hooks` with one role filled, what learned roles need:
    ``role_counts`` ``(A, 0)`` and every agent in role 0."""
    _patch_training_hooks(env)
    scen = env.scenario
    roles = torch.zeros(env.n_envs, env.n_agents, dtype=torch.int32, device=env.device)
    scen.role_counts = (env.n_agents, 0)
    scen.agent_roles = lambda: roles


POSE_SCALE = 0.5


def _patch_pose_aux(env) -> None:
    """:func:`_patch_training_hooks` with ``pusht_lidar``'s flag-free aux layout as far as
    ``priv_noise`` reads it: four columns (a position over ``aux_scale``, then a heading's
    cos/sin) and an ``aux_scale``."""
    _patch_training_hooks(env)
    scen = env.scenario
    scen.aux_dim, scen.aux_scale = 4, POSE_SCALE
    scen.aux_targets = lambda: scen.observations()[..., :4]


def _patch_reset_roles(env) -> None:
    """:func:`_patch_training_hooks`, with roles that change at every reset: a
    fixed-address buffer that ``reset_world`` re-permutes for the masked envs with
    capturable ops (a random argsort into the base role vector, written through a
    ``where``), so an auto-reset hands the next episode new roles in place."""
    _patch_training_hooks(env)
    scen, E, A = env.scenario, env.n_envs, env.n_agents
    base = torch.tensor(
        [r for r, c in enumerate(ROLE_COUNTS) for _ in range(c)], device=env.device
    )
    roles = torch.zeros(E, A, dtype=torch.int32, device=env.device)
    reset_world = scen.reset_world

    def draw(mask: torch.Tensor | None) -> None:
        new = base[torch.rand(E, A, device=env.device).argsort(-1)].to(roles.dtype)
        roles.copy_(new if mask is None else torch.where(mask.unsqueeze(-1), new, roles))

    def reset_with_roles(env_mask=None, *, obs_only=False):
        draw(env_mask)
        reset_world(env_mask, obs_only=obs_only)

    scen.reset_world = reset_with_roles
    scen.agent_roles = lambda: roles


def _algo(
    graphs: bool,
    seed: int = 0,
    device: str = "cuda:0",
    features: bool = False,
    patch=_patch_training_hooks,
    scen_kwargs=(),
    train_only=(),
    teacher=None,
    metrics=(),
    **spec_kw,
) -> MAPPO:
    """Navigation at test size; ``features`` turns on the state critic (state + obs, per
    agent), role heads and the aux loss over ``patch`` (:func:`_patch_training_hooks`).
    ``teacher(actor, env)`` builds a teacher to distil into the actor."""
    # A previous test's MAPPO holds captured torch graphs in reference cycles (bound
    # methods); collected during this env's Warp capture, their teardown would invalidate
    # it. ``torch.cuda.graph`` collects before its own captures for the same reason.
    gc.collect()
    if features:
        spec_kw = {"actor": "policynet", "aux_coef": 0.5, **spec_kw}
    spec = replace(
        SPECS["navigation"], n_envs=64, steps_per_batch=8, epochs=2, minibatches=4,
        num_cells=32, **spec_kw,
    )
    env = build_env(
        "navigation", n_envs=spec.n_envs, n_agents=spec.n_agents, device=device,
        dt=spec.dt, substeps=spec.substeps, max_steps=spec.max_steps, seed=seed,
        scen_kwargs=scen_kwargs, env_kwargs={"auto_reset": True},
    )
    torch.manual_seed(seed)
    Ob, A = env.obs_dim, env.n_agents
    memory = {
        "recurrent": spec.recurrent, "hidden_dim": spec.hidden_dim, "comm": spec.comm,
        "comm_dim": spec.comm_dim,
    }
    if features:
        patch(env)
        counts = env.role_counts
        actor = PolicyNet(
            Ob, env.act_dim, spec.num_cells, role_counts=counts,
            role_params=spec.role_params, aux_dim=env.aux_dim,
            priv_dim=env.aux_dim if spec.privileged_actor else 0,
            learned_roles=spec.learned_roles, role_dwell=spec.role_dwell, **memory,
        )
        role_state = A * (len(counts) + 1) if spec.learned_roles else 0
        critic = StateCritic(
            env.state_dim + role_state, Ob, A, spec.num_cells, use_obs=True,
            role_counts=counts, per_agent=True,
        )
    elif spec.actor == "policynet":
        if patch is not _patch_training_hooks:
            patch(env)
        actor = PolicyNet(Ob, env.act_dim, spec.num_cells, **memory)
        critic = Critic(Ob, A, spec.num_cells)
    else:
        actor = Actor(Ob, env.act_dim, spec.num_cells)
        critic = Critic(Ob, A, spec.num_cells)
    if train_only:
        freeze_actor(actor, train_only)
    module = None if teacher is None else teacher(actor, env).to(device)
    algo = MAPPO(env, actor.to(device), critic.to(device), spec, metrics, graphs=graphs,
                 teacher=module)
    algo.reset()
    return algo


def _fresh_teacher(actor, env, **kw) -> PolicyNet:
    """A privileged teacher over the env's roles, with weights of its own."""
    return PolicyNet(env.obs_dim, env.act_dim, 32, role_counts=env.role_counts,
                     aux_dim=AUX_DIM, priv_dim=AUX_DIM, **kw)


def test_defaults_allocate_no_new_buffers_and_keep_the_legacy_stats():
    algo = _algo(graphs=False, device="cpu")
    assert algo.state_buf is None and algo.role_buf is None and algo.aux_buf is None
    assert algo.mb_stats.numel() == 4
    assert algo.stat_names == list(LEGACY_STAT_NAMES) == [
        "mean_reward", "n_terminated", "n_done", "loss_objective", "loss_critic", "entropy",
        "grad_skips",
    ]
    algo.rollout()
    assert algo._act_ptrs == (algo._out[0].data_ptr(),)  # only the obs is watched
    stats = algo.update(algo.spec.entropy_coeff)
    assert list(stats) == algo.stat_names


def test_a_scenario_with_roles_gets_a_role_ordered_obs_critic():
    """The default obs critic flattens obs in agent order, blind to roles; with roles it is
    a StateCritic that orders them by role."""
    env = build_env(
        "navigation", n_envs=4, n_agents=4, device="cpu", dt=0.05, substeps=1,
        max_steps=5, seed=0,
    )
    box = (-torch.ones(env.act_dim), torch.ones(env.act_dim))
    spec = SPECS["navigation"]
    assert network_archs(spec, env, *box)[2] == "Critic"
    _patch_training_hooks(env)
    _, _, cls, kw = network_archs(spec, env, *box)
    assert cls == "StateCritic"
    assert (kw["use_obs"], kw["state_dim"], kw["per_agent"]) == (True, 0, False)
    assert kw["role_counts"] == list(ROLE_COUNTS)
    assert StateCritic(**kw).uses_roles


def test_reset_rejects_roles_that_break_role_counts():
    def bad_row(env):
        _patch_training_hooks(env)
        roles = env.scenario.agent_roles()
        roles[2] = 0  # every agent in role 0
        env.scenario.agent_roles = lambda: roles

    with pytest.raises(ValueError, match=r"breaks role_counts \(1, 3\) in 1/64 envs; env 2"):
        _algo(graphs=False, device="cpu", features=True, patch=bad_row)


def test_features_add_their_buffers_and_the_aux_stat():
    algo = _algo(graphs=False, device="cpu", features=True)
    T, E, A, S = algo.T, algo.E, algo.A, algo.env.state_dim
    assert algo.state_buf.shape == (T + 1, E, S)
    assert algo.role_buf.shape == (T + 1, E, A) and algo.role_buf.dtype == torch.long
    assert algo.aux_buf.shape == (T, E, A, AUX_DIM)
    assert algo.stat_names == [*LEGACY_STAT_NAMES[:6], "loss_aux", "grad_skips"]
    algo.rollout()
    stats = algo.update(algo.spec.entropy_coeff)
    assert stats["loss_aux"] > 0 and stats["grad_skips"] == 0


@pytest.mark.gpu
def test_captured_rollout_fills_consistent_slots():
    """Every slot's stored log-prob is the actor's log-prob of that slot's (obs, u)."""
    algo = _algo(graphs=True)
    algo.rollout()
    assert 1 <= algo.act_recaptures <= 2
    algo.rollout()  # replays the graphs captured in the first
    assert algo.act_recaptures == 0
    assert algo.t.item() == algo.T
    with torch.no_grad():
        loc, scale = algo.actor(algo.obs_buf[: algo.T])
        ref = tanh_normal_log_prob(algo.u_buf, loc, scale)
    assert torch.allclose(algo.logp_buf, ref, atol=1e-4)
    assert torch.equal(algo.obs_buf[algo.T], algo._out[0])


@pytest.mark.gpu
def test_captured_rollout_records_state_roles_and_aux_with_their_obs():
    """Each slot's state, roles and aux targets are the ones written with that slot's obs
    (the patched hooks are views of the obs buffer, so the rows must agree exactly)."""
    algo = _algo(graphs=True, features=True)
    for _ in range(2):
        algo.rollout()
    T = algo.T
    assert torch.equal(algo.state_buf, algo.obs_buf.flatten(-2))
    assert torch.equal(algo.aux_buf, algo.obs_buf[:T, ..., :AUX_DIM])
    roles = algo.env.agent_roles().long()
    assert torch.equal(algo.role_buf, roles.expand_as(algo.role_buf))
    with torch.no_grad():
        loc, scale = algo.actor(algo.obs_buf[:T], algo.role_buf[:T])
        ref = tanh_normal_log_prob(algo.u_buf, loc, scale)
    assert torch.allclose(algo.logp_buf, ref, atol=1e-4)


@pytest.mark.gpu
def test_a_hook_returning_fresh_tensors_is_refused():
    """A fresh agent_roles() tensor per call would recapture the act graph every step."""

    def fresh_roles(env):
        _patch_training_hooks(env)
        roles = env.scenario.agent_roles()
        env.scenario.agent_roles = lambda: roles.clone()

    algo = _algo(graphs=True, features=True, patch=fresh_roles)
    with pytest.raises(RuntimeError, match="fixed-address buffer"):
        algo.rollout()
    assert algo.act_recaptures == 3


# Navigation with an obstacle resets on the host (its reset is not capture-safe), so the
# patched reset_world runs at every auto-reset; three-step episodes end inside a rollout.
_RESET_ROLES = {"patch": _patch_reset_roles, "scen_kwargs": {"n_obstacles": 1}, "max_steps": 3}


@pytest.mark.gpu
def test_roles_that_change_at_auto_reset_are_recorded_with_their_obs():
    """Each role_buf row is the roles live when its obs was written, including the row
    right after a done, when the auto-reset has already drawn the next episode's roles;
    and the act graph sampled under those roles."""
    algo = _algo(graphs=True, features=True, role_params="separate", **_RESET_ROLES)
    env, live = algo.env, []
    step = env.step

    def recording_step(actions):
        out = step(actions)
        live.append(env.agent_roles().long().clone())
        return out

    env.step = recording_step
    for _ in range(2):  # the second rollout replays the graphs
        live.clear()
        start = env.agent_roles().long().clone()
        algo.rollout()
        expected = torch.stack([start, *live])
        assert torch.equal(algo.role_buf, expected)
    done = torch.maximum(algo.term_buf, algo.trunc_buf) > 0  # [T, E]
    changed = (expected[1:] != expected[:-1]).any(-1)
    assert changed.any() and not (changed & ~done).any()  # new roles, only at a done
    T = algo.T
    with torch.no_grad():
        loc, scale = algo.actor(algo.obs_buf[:T], algo.role_buf[:T])
        ref = tanh_normal_log_prob(algo.u_buf, loc, scale)
    assert torch.allclose(algo.logp_buf, ref, atol=1e-4)


HIDDEN = 16
# Three-step episodes from staggered starts, so at a given step some envs end their
# episode and others do not.
_MEMORY = {"actor": "policynet", "recurrent": True, "hidden_dim": HIDDEN, "max_steps": 3,
           "stagger_starts": True}


def _patch_comm_mask(env) -> None:
    """A fixed-address ``[E, A, A]`` comm mask: each agent hears a random half of its
    teammates."""
    gen = torch.Generator().manual_seed(11)
    mask = (torch.rand(env.n_envs, env.n_agents, env.n_agents, generator=gen) < 0.5)
    mask = mask.to(env.device)
    env.scenario.comm_mask = lambda: mask


_COMM = {"actor": "policynet", "comm": "attn", "comm_dim": 8, "patch": _patch_comm_mask}


_ROLLOUT = ("obs_buf", "u_buf", "logp_buf", "rew_buf", "term_buf", "trunc_buf",
            "state_buf", "role_buf", "aux_buf", "h_buf", "mask_buf", "msg_obs_buf",
            "msg_mask_buf", "msg_noise_buf", "msg_aux_buf", "msg_valid_buf", "lrole_in_buf",
            "lrole_buf", "decide_buf", "teacher_loc_buf", "priv_buf")
# An estimator's errors on the privileged pose, outliers included.
_NOISE = (0.03, 0.03, 0.05, 0.05, 0.05)
# Learned roles over one filled role, decided every 3 steps in 5-step episodes.
_LEARNED = {"learned_roles": True, "role_dwell": 3, "normalize_advantage": False,
            "critic_input": "state+obs", "max_steps": 5, "patch": _patch_single_role}


def _copy_rollout(dst: MAPPO, src: MAPPO) -> None:
    for name in _ROLLOUT:
        if getattr(src, name) is not None:
            getattr(dst, name).copy_(getattr(src, name))


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("features", "kw"),
    [
        (False, {}), (True, {}), (True, {"role_params": "separate", **_RESET_ROLES}),
        (False, {**_MEMORY, "seq_len": 4}), (False, _COMM),
        (True, {**_MEMORY, "seq_len": 2, "comm": "attn", "comm_dim": 8}),
        (True, {**_MEMORY, "role_params": "separate"}),
        (True, {"comm": "vae", "comm_dim": 4, "msg_drop": 0.3, "max_steps": 3}),
        (True, {"privileged_actor": True}),
        (True, _LEARNED),
        (True, {**_LEARNED, "comm": "pose", "comm_dim": 4, "msg_drop": 0.3}),
        (True, {**_LEARNED, "recurrent": True, "hidden_dim": HIDDEN, "seq_len": 4}),
        (True, {**_MEMORY, "seq_len": 4, "teacher": _fresh_teacher}),
        (True, {"privileged_actor": True, "priv_noise": _NOISE, "patch": _patch_pose_aux}),
    ],
    ids=["default", "features", "reset-roles", "recurrent", "comm", "recurrent-comm-features",
         "recurrent-separate", "vae-message", "privileged", "learned-roles",
         "learned-roles-pose", "learned-roles-recurrent", "teacher", "priv-noise"],
)
def test_captured_update_matches_eager_update(features, kw):
    """Same weights, same rollout, same permutations: the graphed update lands where the
    eager one does. Entropy is off, since its sample is the one draw the two cannot share.
    ``features`` runs it with the state critic, role heads and aux loss all on;
    ``reset-roles`` with per-role networks and roles redrawn at every auto-reset, so the
    rollout's roles (and the per-slot advantage statistics) move between agents.
    The recurrent cases also force a done in the middle of a BPTT chunk; ``comm`` runs
    attention under a scenario comm mask."""
    eager = _algo(graphs=False, features=features, **kw)
    graphed = _algo(graphs=True, features=features, **kw)
    graphed.actor.load_state_dict(eager.actor.state_dict())
    graphed.critic.load_state_dict(eager.critic.state_dict())
    eager.rollout()
    if eager.recurrent:
        eager.term_buf[eager.L // 2 - 1, ::3] = 1.0  # mid-chunk, in a third of the envs
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
    assert graphed.mb_stats[-1].item() == 0.0


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


# ---------------------------------------------------------------- memory and communication


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_hidden_state_resets_only_in_envs_whose_episode_ended(device):
    """``h_buf[t]`` is the hidden state entering step ``t``: the GRU output of step
    ``t - 1``, zeroed exactly in the envs whose episode ended there and carried in the
    rest, including across the rollout boundary and under the captured graphs."""
    algo = _algo(graphs=device != "cpu", device=device, **_MEMORY)
    assert algo.h.shape == (algo.E, algo.A, HIDDEN)
    algo.rollout()
    carried = algo.h.clone()
    algo.rollout()  # replays the graphs captured in the first
    assert torch.equal(algo.h_buf[0], carried)
    T = algo.T
    with torch.no_grad():
        h_next = algo.actor.step(algo.obs_buf[:T], None, algo.h_buf)[3]
    ended = torch.maximum(algo.term_buf, algo.trunc_buf) > 0  # [T, E]
    handed = torch.cat([algo.h_buf[1:], algo.h.unsqueeze(0)])  # what step t handed on
    assert (ended.any(-1) & (~ended).any(-1)).any()  # some envs end while others go on
    assert torch.equal(handed[ended], torch.zeros_like(handed[ended]))
    assert torch.allclose(handed[~ended], h_next[~ended], atol=1e-5)
    assert (handed[~ended].abs().sum(-1) > 0).all()


@pytest.mark.parametrize("seq_len", [0, 4])
@pytest.mark.parametrize("comm", [False, True], ids=["gru", "gru+comm"])
def test_bptt_from_stored_chunk_starts_recomputes_the_rollout_log_probs(seq_len, comm):
    """Unrolling every chunk from its stored ``h_buf`` row, with the hidden state zeroed
    after each done, gives back each step's stored log-prob; the chunks tile the rollout
    and a minibatch holds as many samples as without memory."""
    kw = {**_MEMORY, "seq_len": seq_len}
    if comm:
        kw.update(comm="attn", comm_dim=8, patch=_patch_comm_mask)
    algo = _algo(graphs=False, device="cpu", **kw)
    algo.rollout()
    algo._prepare_body()
    T, N, A = algo.T, algo.N, algo.A
    chunk, per_minibatch = seq_len or T, N // algo.M
    assert chunk == algo.L and per_minibatch == algo.B
    assert (algo.mask_buf is not None) is comm
    if comm:
        assert torch.equal(algo.mask_buf, algo.env.comm_mask().expand_as(algo.mask_buf))
    rows = algo._sequence_rows(torch.arange(algo.n_units))
    assert torch.equal(rows.sort().values, torch.arange(N))
    done = algo.done_buf.reshape(N)[rows].view(algo.L, -1)
    assert done[:-1].any()  # an episode ends inside a chunk
    mask = None if algo.mask_buf is None else algo.mask_buf.reshape(N, A, A)[rows]
    with torch.no_grad():
        loc, scale, _ = policy_outputs(
            algo.actor, algo.obs_buf[:T].reshape(N, A, -1)[rows], None,
            h0=algo.h_buf.reshape(N, A, -1)[rows[: algo.n_units]], done=done, comm_mask=mask,
        )
    logp = tanh_normal_log_prob(algo.u_buf.reshape(N, A, -1)[rows], loc, scale)
    assert torch.allclose(logp, algo.logp_buf.reshape(N, A)[rows], atol=1e-5)
    stats = algo.update(algo.spec.entropy_coeff)
    assert stats["grad_skips"] == 0 and math.isfinite(stats["loss_objective"])


def test_comm_with_separate_role_networks_is_refused():
    env = build_env(
        "navigation", n_envs=4, n_agents=4, device="cpu", dt=0.05, substeps=1,
        max_steps=5, seed=0,
    )
    _patch_training_hooks(env)
    box = (-torch.ones(env.act_dim), torch.ones(env.act_dim))
    spec = replace(SPECS["navigation"], actor="policynet", role_params="separate", comm="attn")
    with pytest.raises(ValueError, match="not 'separate'"):
        network_archs(spec, env, *box)


def test_train_round_trips_a_recurrent_communicating_policy_and_scores_it(tmp_path):
    """Through train(), its checkpoint, a resume, and the greedy eval, which carries the
    hidden state over score()'s one un-reset episode per env."""
    memory = ["--actor", "policynet", "--recurrent", "--hidden-dim", "6", "--seq-len", "2",
              "--comm", "attn", "--comm-dim", "4"]
    flags = ["--scenario", "navigation", *SMOKE, "--checkpoint-dir", str(tmp_path)]
    path = train(parse_args([*flags, *memory]))
    actor, _, meta = load_checkpoint(path)
    assert isinstance(actor, PolicyNet) and actor.recurrent and actor.communicates
    assert (actor.hidden_dim, actor.comm_dim) == (6, 4)
    kw = meta["actor_kwargs"]
    assert (kw["recurrent"], kw["hidden_dim"], kw["comm"], kw["comm_dim"]) == (True, 6, "attn", 4)
    train(parse_args([*flags, *memory, "--resume", str(path)]))
    with pytest.raises(ValueError, match="--recurrent --hidden-dim 6 --comm attn --comm-dim 4"):
        train(parse_args([*flags, "--actor", "policynet", "--resume", str(path)]))

    seen = []
    make = greedy_score_kwargs(actor)["policy_from_env"]

    def spy(env):
        seen.append((env.auto_reset, env.max_steps))
        return make(env)

    res = score("navigation", policy_from_env=spy, n_agents=4, n_envs=8, steps=6,
                device="cpu", seed=1, dt=0.1, substeps=1, arms=("policy",))["policy"]
    assert seen == [(False, None)] and 0.0 <= res.solve_rate <= 1.0  # one episode, no resets
    env = build_env("navigation", n_envs=3, n_agents=4, device="cpu", dt=0.1, substeps=1,
                    max_steps=None, seed=0)
    obs = env.reset()
    act = greedy_env_arm(actor)(env)
    first, second = act(obs), act(obs)
    with torch.no_grad():
        h1 = actor.step(obs)[3]
    assert torch.equal(first, actor.greedy(obs))  # from a zero hidden state
    assert torch.equal(second, actor.greedy(obs, h=h1)) and not torch.equal(first, second)
    assert torch.equal(greedy_env_arm(actor)(env)(obs), first)  # a fresh arm starts at zero


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_action_rate_penalty_charges_each_step_s_change(device):
    """With ``action_rate_coef`` every agent's stored reward is the env's minus the coef
    times the mean squared change of its action since the last step, from rest (zero) on
    an episode's first step, across the rollout boundary too; the actions taken are the
    unpenalised run's, and the mean change is logged as ``action_rate``."""
    coef = 0.3
    runs = []
    for c in (0.0, coef):  # one after the other: the policies draw from torch's RNG
        a = _algo(graphs=device != "cpu", device=device, action_rate_coef=c, max_steps=6)
        if c > 0:
            a.metrics = {"action_rate": 1}
            a.metric_acc = torch.zeros(1, device=device)
        a.rollout()
        last = a.actor.squash(a.u_buf[-1]).masked_fill(
            (torch.maximum(a.term_buf[-1], a.trunc_buf[-1]) > 0).view(-1, 1, 1), 0.0
        )
        a.metric_acc.zero_()
        a.rollout()
        runs.append(a)
    plain, algo = runs
    assert torch.equal(plain.u_buf, algo.u_buf)
    acts = algo.actor.squash(algo.u_buf)  # [T, E, A, D]
    ended = (torch.maximum(algo.term_buf, algo.trunc_buf) > 0).view(algo.T, -1, 1, 1)
    prev = torch.cat([last.unsqueeze(0), acts[:-1].masked_fill(ended[:-1], 0.0)])
    rate = (acts - prev).square().mean(-1)
    assert ended.any()  # an episode ended inside the rollout
    torch.testing.assert_close(algo.rew_buf, plain.rew_buf - coef * rate)
    torch.testing.assert_close(algo.metric_acc[0], rate.mean(dim=(1, 2)).sum())


# ------------------------------------------------------------------- single-run guards


def _healthy(algo: MAPPO) -> None:
    """One ordinary rollout and update, so Adam has state and the graphs are captured."""
    algo.rollout()
    algo.update(algo.spec.entropy_coeff)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_kl_brake_freezes_the_rest_of_the_iteration(device):
    """At a vanishing threshold the first minibatch (KL ~0 against its own rollout) steps,
    the second trips the brake, and no later step of the iteration moves the weights."""
    algo = _algo(graphs=device != "cpu", device=device, target_kl=1e-9)
    _healthy(algo)
    algo.rollout()
    algo._prepare()
    algo._minibatch()
    after_first = [p.detach().clone() for p in algo.params]
    n_mb = algo.spec.epochs * algo.M
    for _ in range(n_mb - 1):
        algo._minibatch()
    for p, q in zip(algo.params, after_first, strict=True):
        assert torch.equal(p, q)
    n = algo.n_losses
    assert algo.mb_stats[n].item() == 0  # no non-finite skip: the NaN abort never sees it
    assert algo.mb_stats[n + 2].item() == n_mb - 1
    assert algo.mb_stats[n + 1].item() > 0
    algo.rollout()  # a fresh rollout: its first minibatch steps again, as prepare clears
    stats = algo.update(algo.spec.entropy_coeff)  # the brake each iteration
    assert {"approx_kl", "kl_stops"} <= set(stats) and stats["kl_stops"] == n_mb - 1


def test_kl_brake_off_leaves_the_update_bit_identical():
    """The brake's ops are detached: an untripped brake lands on exactly the weights of a
    run without one, from the same rollout and permutations (CPU, so deterministic)."""
    plain = _algo(graphs=False, device="cpu")
    braked = _algo(graphs=False, device="cpu", target_kl=1e9)
    assert plain.stat_names == list(LEGACY_STAT_NAMES)
    braked.actor.load_state_dict(plain.actor.state_dict())
    braked.critic.load_state_dict(plain.critic.state_dict())
    plain.rollout()
    _copy_rollout(braked, plain)
    for algo in (plain, braked):
        algo.ent_coef.fill_(0.0)
        algo._prepare_body()
    braked.perms.copy_(plain.perms)
    for _ in range(plain.spec.epochs * plain.M):
        plain._minibatch()
        braked._minibatch()
    for a, b in zip(plain.params, braked.params, strict=True):
        assert torch.equal(a, b)
    assert braked.mb_stats[braked.n_losses + 2].item() == 0


@pytest.mark.gpu
def test_rollback_restores_weights_and_adam_in_place_and_the_graphs_replay():
    algo = _algo(graphs=True, rollback=True)
    _healthy(algo)
    snap = algo.snapshot()
    live = [p.detach() for p in algo.params]
    live += [v for p in algo.params for v in algo.optim.state[p].values()]
    ptrs = [t.data_ptr() for t in live]
    graph = algo._minibatch.graph
    _healthy(algo)  # moves the weights and the moments
    assert not all(torch.equal(a, b) for a, b in zip(live, snap, strict=True))
    algo.restore(snap)
    assert all(torch.equal(a, b) for a, b in zip(live, snap, strict=True))
    assert [t.data_ptr() for t in live] == ptrs
    _healthy(algo)
    assert algo._minibatch.graph is graph  # replayed, not recaptured
    assert not all(torch.equal(a, b) for a, b in zip(live, snap, strict=True))
    assert all(torch.isfinite(p).all() for p in algo.params)


def _staged_collapse_trace(dip: bool = True) -> list[float]:
    """A windowed solve rate shaped like a staged slim-T Limo run's (logged every 25
    iterations): learned by 3000, a 0.95-0.97 plateau, the 5650-5700 dip and its recovery,
    then the collapse from 0.83 at 5900 to 0.22 at 5960. A small deterministic wobble stands
    in for the noise."""
    knots = [(0, 0.0), (3000, 0.9), (3200, 0.95), (5550, 0.97)]
    if dip:
        knots += [(5575, 0.92), (5600, 0.89), (5625, 0.88), (5650, 0.625), (5675, 0.537),
                  (5700, 0.553), (5725, 0.794), (5750, 0.579), (5775, 0.553), (5800, 0.807),
                  (5850, 0.90)]
    knots += [(5900, 0.83), (5960, 0.22), (6000, 0.04)]
    out = []
    for it in range(6000):
        pairs = zip(knots[:-1], knots[1:], strict=True)
        (a, va), (b, vb) = next((k, n) for k, n in pairs if n[0] > it)
        out.append(va + (vb - va) * (it - a) / (b - a) + 0.015 * math.sin(it / 3.0))
    return out


def _first_rollback(trace: list[float], guard) -> int | None:
    for it, solve in enumerate(trace):
        if guard.step(it, solve, 0) == "rollback":
            return it
    return None


def test_rollback_guard_fires_on_the_l8b_s1_dip_and_restores_from_before_it():
    """The dip to 0.53 (44 points under the plateau for ~75 iterations) was the onset of
    the instability that ended the run at 0.04: the guard rolls back there, to a snapshot
    taken before the rate started falling."""
    from swarp.rl.ppo import RollbackGuard

    guard = RollbackGuard()
    it = _first_rollback(_staged_collapse_trace(), guard)
    assert it is not None and 5650 <= it <= 5725
    # Before the fall from 0.88 at 5625, while still within keep * best of the plateau.
    assert guard.restore_it < 5625 and guard.snaps[-1] == guard.restore_it
    assert len(guard.snaps) <= guard.ring


def test_rollback_guard_fires_on_the_final_collapse_and_ignores_wobble():
    from swarp.rl.ppo import RollbackGuard

    guard = RollbackGuard()
    it = _first_rollback(_staged_collapse_trace(dip=False), guard)
    assert it is not None and 5900 < it <= 5960  # never on the plateau's wobble
    assert guard.restore_it < 5900
    # A short excursion (under the patience) is not a collapse.
    short = [0.95] * 400 + [0.4] * 20 + [0.95] * 200
    assert _first_rollback(short, RollbackGuard()) is None


def test_rollback_guard_ignores_a_policy_still_learning_around_the_floor():
    """A Limo run's trace: rising through 0.2-0.51 with windowed noise of +-0.05 is not a
    collapse, so nothing is judged until the rate has passed floor + drop."""
    from swarp.rl.ppo import RollbackGuard

    rising = [0.2 + 0.3 * i / 300 + (0.05 if i % 40 < 20 else -0.05) for i in range(300)]
    hover = [0.5 + (0.01 if i % 60 < 10 else -0.1) for i in range(600)]
    guard = RollbackGuard()
    assert _first_rollback(rising + hover, guard) is None
    assert not guard.armed


def test_rollback_guard_judges_each_ramp_position_on_its_own_best_and_stops_at_the_limit():
    from swarp.rl.ppo import RollbackGuard

    # A harder condition (a new bucket) holding 0.7 is not a collapse from the old 0.97.
    guard = RollbackGuard()
    for it in range(1000):
        assert guard.step(it, 0.97 if it < 500 else 0.7, 0 if it < 500 else 1) != "rollback"
    # Every collapse rolls back until the budget is spent, then the run stops.
    guard = RollbackGuard(max_rollbacks=2)
    actions, it = [], 0
    for _ in range(3):
        for solve in [0.95] * 300 + [0.2] * 30:
            a = guard.step(it, solve, 0)
            if a in ("rollback", "stop"):
                actions.append(a)
            it += 1
    assert actions == ["rollback", "rollback", "stop"]


class _SolvedWindow(SolveRateWindow):
    """A window that always reports a solved policy, so a smoke run can open the gate."""

    @property
    def solve_rate(self) -> float:
        return 1.0


@pytest.mark.parametrize("gate_open", [False, True])
def test_gated_ramps_follow_the_clock_and_end_the_run(tmp_path, monkeypatch, gate_open):
    """Closed, the clock runs to the first ramp's start and holds there for the whole cap;
    open, the ramp runs on it and the run ends at the last end plus hold_iters."""
    if gate_open:
        monkeypatch.setattr("swarp.rl.ppo.SolveRateWindow", _SolvedWindow)
    cfg = parse_args(["--scenario", "navigation", *SMOKE, "--iters", "10", "--lr-final",
                      "1e-5", "--checkpoint-dir", str(tmp_path)])
    cfg.spec = replace(
        cfg.spec, ramps=(Ramp("pos_shaping_factor", 0.0, 1.0, 2, 4, steps=2),),
        ramp_gate=0.5, hold_iters=2,
    )
    train(cfg)
    with (tmp_path / "metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    clock = [int(r["ramp_clock"]) for r in rows]
    ramp = [r["ramp"] for r in rows]
    if gate_open:
        assert clock == [0, 1, 2, 3, 4, 5]
        assert ramp == ["0.000", "0.000", "0.000", "0.500", "1.000", "1.000"]
    else:
        assert clock == [0, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        assert set(ramp) == {"0.000"}


def test_guard_fields_are_validated():
    from swarp.rl.recipes import TrainSpec

    with pytest.raises(ValueError, match="target_kl"):
        TrainSpec(target_kl=0.0)
    with pytest.raises(ValueError, match="adaptive_kl"):
        TrainSpec(adaptive_kl=-1.0)
    with pytest.raises(ValueError, match="needs ramps"):
        TrainSpec(ramp_gate=0.5)
    with pytest.raises(ValueError, match="needs ramp_gate"):
        TrainSpec(ramps=(Ramp("x", 0.0, 1.0, 0, 1),), hold_iters=10)


def test_train_rolls_back_halves_the_lr_and_stops_at_the_limit(tmp_path, monkeypatch):
    """train()'s side of the guard: a scripted solve rate collapses twice; the first
    collapse restores a snapshot (logged, lr halved), the second exceeds a budget of one
    and stops the run with its state saved."""
    from swarp.rl.ppo import RollbackGuard, RollbackLimit

    script = [0.9] * 6 + [0.1] * 20
    calls = [0]

    class Scripted(SolveRateWindow):
        def update_counts(self, *args) -> None:
            super().update_counts(*args)
            calls[0] += 1

        @property
        def solve_rate(self) -> float:
            return script[calls[0] - 1]

    monkeypatch.setattr("swarp.rl.ppo.SolveRateWindow", Scripted)
    monkeypatch.setattr(
        "swarp.rl.ppo.RollbackGuard",
        lambda lag: RollbackGuard(every=1, patience=2, lag=1, max_rollbacks=1),
    )
    cfg = parse_args(["--scenario", "navigation", *SMOKE, "--iters", "20",
                      "--checkpoint-dir", str(tmp_path)])
    cfg.spec = replace(cfg.spec, rollback=True)
    with pytest.raises(RollbackLimit, match="after 1 rollbacks"):
        train(cfg)
    with (tmp_path / "metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["rollbacks"]) for r in rows] == [0] * 7 + [1, 1]
    assert [r["lr_mult"] for r in rows] == ["1"] * 7 + ["0.5", "0.5"]
    assert (tmp_path / "navigation_collapsed.pt").exists()


def test_adaptive_kl_lr_steps_within_its_band_and_bounds():
    from swarp.rl.ppo import KL_LR_FLOOR, KL_LR_STEP, adapt_kl_lr

    assert adapt_kl_lr(1.0, 0.02, 0.02) == 1.0  # inside [target / 2, 2 target]
    assert adapt_kl_lr(1.0, 0.05, 0.02) == pytest.approx(1.0 / KL_LR_STEP)
    assert adapt_kl_lr(0.5, 0.005, 0.02) == pytest.approx(0.5 * KL_LR_STEP)
    assert adapt_kl_lr(1.0, 0.001, 0.02) == 1.0  # never above the schedule
    assert adapt_kl_lr(KL_LR_FLOOR, 1.0, 0.02) == KL_LR_FLOOR


def test_train_adapts_the_lr_to_the_kl_without_braking(tmp_path):
    """A target far below any real KL cuts the lr every iteration and logs it; with no
    target_kl the KL is measured but never stops a minibatch."""
    from swarp.rl.ppo import KL_LR_STEP

    # Two epochs: a minibatch's KL is measured before its own step, so it needs one before.
    cfg = parse_args(["--scenario", "navigation", *SMOKE, "--iters", "4", "--epochs", "2",
                      "--checkpoint-dir", str(tmp_path)])
    cfg.spec = replace(cfg.spec, adaptive_kl=1e-12)
    train(cfg)
    with (tmp_path / "metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [float(r["kl_lr_mult"]) for r in rows] == pytest.approx(
        [KL_LR_STEP**-i for i in range(4)], rel=1e-3
    )
    assert {r["kl_stop_frac"] for r in rows} == {"0.000"}


def test_actor_scale_min_floors_the_std_and_round_trips_in_the_config():
    from swarp.rl.networks import PolicyNet
    from swarp.rl.recipes import TrainSpec

    net = PolicyNet(6, 2, 8, scale_min=0.1)
    with torch.no_grad():
        net.head.bias.fill_(-100.0)  # drive the raw scale to the floor
    _, scale = net(torch.zeros(3, 2, 6))
    assert torch.allclose(scale, torch.full_like(scale, 0.1))
    with torch.no_grad():
        net.head.weight.zero_()
        net.head.bias.zero_()
    _, scale = net(torch.zeros(3, 2, 6))
    assert torch.allclose(scale, torch.ones_like(scale))  # a zero input still maps to 1
    assert PolicyNet(**net.config).scale_min == 0.1
    assert "scale_min" not in PolicyNet(6, 2, 8).config  # default checkpoints unchanged
    with pytest.raises(ValueError, match="actor_scale_min"):
        TrainSpec(actor_scale_min=0.1)  # the default mlp actor has no such option


def test_load_widened_keeps_the_function_of_the_saved_network():
    """A critic whose state gained a column computes what the saved one did, whatever the
    new column holds; a shape that cannot widen is refused."""
    torch.manual_seed(0)
    old = StateCritic(3, 5, 2, 16, per_agent=True)
    new = StateCritic(4, 5, 2, 16, per_agent=True)
    load_widened(new, old.state_dict())
    state = torch.randn(7, 4)
    obs = torch.zeros(7, 2, 5)
    torch.testing.assert_close(new(obs, state), old(obs, state[:, :3]))
    with pytest.raises(ValueError, match="cannot widen"):
        load_widened(StateCritic(2, 5, 2, 16, per_agent=True), old.state_dict())


@pytest.mark.parametrize("comm", ["latent", "vae", "pose"])
@pytest.mark.parametrize("recurrent", [False, True], ids=["mlp", "gru"])
def test_delayed_messages_come_from_the_last_step_and_recompute_the_log_probs(comm, recurrent):
    """Each step hears what the teammates encoded from the previous step's observation of
    the same episode (nothing after a done, a dropped packet now and then), and the update
    recomputes the rollout's log-probabilities from the stored inputs and noise."""
    kw = {"comm": comm, "comm_dim": 4, "msg_drop": 0.3, "max_steps": 3}
    if recurrent:
        kw.update(recurrent=True, hidden_dim=HIDDEN, seq_len=4)
    algo = _algo(graphs=False, device="cpu", features=True, **kw)
    algo.rollout()
    T, A, N = algo.T, algo.A, algo.N
    ended = torch.maximum(algo.term_buf, algo.trunc_buf) > 0  # [T, E]
    heard = algo.msg_mask_buf & ~torch.eye(A, dtype=torch.bool)  # [T, E, A, A]
    assert not heard[1:][ended[:-1]].any()  # nothing from the last episode
    assert torch.equal(algo.msg_obs_buf[1:][~ended[:-1]], algo.obs_buf[: T - 1][~ended[:-1]])
    live = heard[1:][~ended[:-1]]
    pair = ~torch.eye(A, dtype=torch.bool).expand_as(live)
    assert 0.5 < live[pair].float().mean() < 0.9  # about 1 - msg_drop got through
    assert (algo.msg_noise_buf is not None) is (comm == "vae")
    assert (algo.msg_aux_buf is not None) is (comm != "latent")
    algo._prepare_body()
    rows = algo._sequence_rows(torch.arange(algo.n_units)) if recurrent else torch.arange(N)
    msg = {
        "msg_obs": algo.msg_obs_buf.reshape(N, A, -1)[rows],
        "msg_mask": algo.msg_mask_buf.reshape(N, A, A)[rows],
        "msg_noise": None if algo.msg_noise_buf is None
        else algo.msg_noise_buf.reshape(N, A, -1)[rows],
    }
    mem = {}
    if recurrent:
        mem = {"h0": algo.h_buf.reshape(N, A, -1)[rows[: algo.n_units]],
               "done": algo.done_buf.reshape(N)[rows].view(algo.L, -1)}
    roles = algo.role_buf[:T].reshape(N, A)[rows]
    with torch.no_grad():
        loc, scale, _ = policy_outputs(algo.actor, algo.obs_buf[:T].reshape(N, A, -1)[rows],
                                       roles, msg=msg, **mem)
    logp = tanh_normal_log_prob(algo.u_buf.reshape(N, A, -1)[rows], loc, scale)
    assert torch.allclose(logp, algo.logp_buf.reshape(N, A)[rows], atol=1e-5)
    stats = algo.update(algo.spec.entropy_coeff)
    assert stats["grad_skips"] == 0 and math.isfinite(stats["loss_objective"])
    assert ("loss_msg" in stats) is (comm != "latent")


def test_a_heard_message_changes_the_listener_and_a_policy_widens_into_messages():
    """Only a received message moves the listener's action, and a policy without messages
    widened into one acts as before until its message weights move."""
    torch.manual_seed(0)
    plain = PolicyNet(6, 2, 16, aux_dim=3)
    talk = PolicyNet(6, 2, 16, aux_dim=3, comm="latent", comm_dim=4)
    load_widened(talk, plain.state_dict())
    obs, prev = torch.randn(5, 3, 6), torch.randn(5, 3, 6)
    mask = torch.zeros(5, 3, 3, dtype=torch.bool)
    mask[:, 0, 1] = True  # agent 0 hears agent 1
    torch.testing.assert_close(talk.step(obs, msg_obs=prev, msg_mask=mask)[0], plain.step(obs)[0])
    with torch.no_grad():
        talk.trunk[0].weight.normal_()
    a = talk.step(obs, msg_obs=prev, msg_mask=mask)[0]
    b = talk.step(obs, msg_obs=prev + 1.0, msg_mask=mask)[0]
    assert not torch.allclose(a[:, 0], b[:, 0]) and torch.equal(a[:, 1:], b[:, 1:])


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphed"])
def test_train_only_moves_the_named_actor_parameters_and_the_critic(graphs):
    """With the trunk frozen, an update moves the action head and the critic, nothing else."""
    if graphs and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    algo = _algo(graphs=graphs, device="cuda:0" if graphs else "cpu", features=True,
                 train_only=("head.",))
    before = {n: p.detach().clone() for n, p in algo.actor.named_parameters()}
    critic = [p.detach().clone() for p in algo.critic.parameters()]
    algo.rollout()
    algo.update(algo.spec.entropy_coeff)
    for n, p in algo.actor.named_parameters():
        assert torch.equal(p, before[n]) is not n.startswith("head."), n
    assert any(not torch.equal(p, q) for p, q in zip(algo.critic.parameters(), critic, strict=True))
    with pytest.raises(ValueError, match="matches none"):
        freeze_actor(algo.actor, ("nope",))


def test_a_scen_kwarg_pins_its_attribute_and_holds_the_ramp_on_it():
    """A ramp would overwrite a --scen-kwarg from the first iteration on; the flag wins, and
    the ramp stays (held at the flag's value) so the gated clock keeps its length."""
    spec = SPECS["pusht_lidar_limo"]
    assert "rot_away_penalty" in {r.attr for r in spec.ramps}
    cfg = parse_args(["--scenario", "pusht_lidar_limo", "--scen-kwarg", "rot_away_penalty=0"])
    assert cfg.scen_kwargs["rot_away_penalty"] == 0
    for old, new in zip(spec.ramps, cfg.spec.ramps, strict=True):
        assert (old.attr, old.start, old.end) == (new.attr, new.start, new.end)
        if new.attr == "rot_away_penalty":
            assert new.value(new.start) == new.value(new.end) == 0
        else:
            assert new == old
    # A scalar on a tuple-valued ramp holds it in every entry, as the constructor reads it.
    cfg = parse_args(["--scenario", "pusht_lidar_limo", "--scen-kwarg", "pose_noise=0.5"])
    (ramp,) = (r for r in cfg.spec.ramps if r.attr == "pose_noise")
    assert ramp.value(ramp.start) == ramp.value(ramp.end) == (0.5, 0.5)


def test_resume_widen_refuses_columns_it_would_misplace_and_inputs_it_would_reinterpret():
    """A width that grows ahead of other columns of its layer, or attention or memory added,
    is refused; new columns that come last, a fresh message encoder and separate role
    trunks still widen."""
    critic = {"state_dim": 3, "obs_dim": 6, "n_agents": 3, "num_cells": 16, "use_obs": False,
              "role_counts": [], "per_agent": False}

    def resume(saved, built, saved_critic=critic, built_critic=critic):
        ckpt = {"format": "swarp.rl/2", "actor_cls": "PolicyNet", "actor_kwargs": saved,
                "critic_cls": "StateCritic", "critic_kwargs": saved_critic}
        built = ("PolicyNet", built, "StateCritic", built_critic)
        check_resume("old.pt", ckpt, built, widen=True)

    def config(obs_dim=6, aux_dim=4, **kw):
        return PolicyNet(obs_dim, 2, 16, aux_dim=aux_dim, **kw).config

    roles, sep = {"role_counts": (2, 0)}, {"role_counts": (1, 2), "role_params": "separate"}
    pose = {"comm": "pose", "comm_dim": 4}
    resume(config(**roles), config(aux_dim=5, **roles, **pose))  # a fresh pose encoder
    resume(config(**sep), config(8, **sep))
    resume(config(**roles), config(**roles, priv_dim=4))
    resume(config(), config(comm="latent", comm_dim=4))
    resume(config(), config(), built_critic={**critic, "state_dim": 5})
    for saved, built, why in (
        (config(**roles), config(8, **roles), "obs_dim grew ahead of the role"),
        (config(**pose), config(8, **pose), "obs_dim grew ahead of the message"),
        (config(priv_dim=4), config(8, priv_dim=4), "obs_dim grew ahead of the privileged"),
        (config(**pose), config(aux_dim=5, **pose), "aux_dim grew inside"),
        (config(), config(comm="attn", comm_dim=4), "comm 'attn' added"),
        (config(), config(recurrent=True), "recurrent added"),
    ):
        with pytest.raises(ValueError, match=why):
            resume(saved, built)
    both = {**critic, "use_obs": True}
    with pytest.raises(ValueError, match="state_dim grew ahead of the obs"):
        resume(config(), config(), both, {**both, "state_dim": 5})
    with pytest.raises(ValueError, match="obs_dim grew inside every agent"):
        resume(config(), config(), both, {**both, "obs_dim": 8})


def test_a_plain_policy_widens_into_a_privileged_one_and_acts_as_before():
    """The privileged columns come last in the trunk, after the role one-hot and the
    delayed messages: a checkpoint without them widens (and passes the resume check) into
    a network that acts as before whatever they hold. Messages read only the observation."""
    torch.manual_seed(0)
    kw = {"role_counts": (1, 2), "aux_dim": 3, "comm": "pose", "comm_dim": 4}
    plain = PolicyNet(6, 2, 16, **kw)
    priv = PolicyNet(6, 2, 16, priv_dim=3, **kw)
    assert "priv_dim" not in plain.config and priv.config["priv_dim"] == 3
    load_widened(priv, plain.state_dict())
    critic = ("Critic", {"obs_dim": 6, "n_agents": 3, "num_cells": 16})
    ckpt = {"format": "swarp.rl/2", "actor_cls": "PolicyNet", "actor_kwargs": plain.config,
            "critic_cls": critic[0], "critic_kwargs": critic[1]}
    built = ("PolicyNet", priv.config, *critic)
    check_resume("plain.pt", ckpt, built, widen=True)
    with pytest.raises(ValueError, match="priv_dim None vs 3"):
        check_resume("plain.pt", ckpt, built)
    obs, prev = torch.randn(5, 3, 6), torch.randn(5, 3, 6)
    roles = torch.tensor([0, 1, 1])[torch.stack([torch.randperm(3) for _ in range(5)])]
    mask = torch.rand(5, 3, 3) < 0.7
    ref = plain.step(obs, roles, msg_obs=prev, msg_mask=mask)
    for extra in (torch.zeros(5, 3, 3), torch.randn(5, 3, 3)):
        wide = torch.cat([obs, extra], -1)
        for msg_obs in (prev, torch.cat([prev, torch.randn(5, 3, 3)], -1)):
            out = priv.step(wide, roles, msg_obs=msg_obs, msg_mask=mask)
            torch.testing.assert_close(out[0], ref[0])
            torch.testing.assert_close(out[1], ref[1])
    with pytest.raises(ValueError, match="obs_dim \\+ priv_dim = 6 \\+ 3"):
        priv.step(obs, roles)
    with pytest.raises(ValueError, match="not 'separate'"):
        PolicyNet(6, 2, 16, role_counts=(1, 2), role_params="separate", priv_dim=3)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_a_privileged_actor_acts_and_updates_on_the_obs_and_aux_targets(device):
    """The rollout acts on ``cat(obs, aux targets)`` (captured on CUDA) while ``obs_buf``,
    the critic and the messages keep the plain observation; the stored log-probs are the
    actor's on ``cat(obs_buf, aux_buf)``, which is what the update reads."""
    algo = _algo(graphs=device != "cpu", device=device, features=True, privileged_actor=True,
                 comm="pose", comm_dim=4, msg_drop=0.3, max_steps=3)
    for _ in range(2):
        algo.rollout()
    T, A, N, Ob = algo.T, algo.A, algo.N, algo.env.obs_dim
    assert algo.obs_buf.shape[-1] == Ob and algo.msg_obs_buf.shape[-1] == Ob
    obs = torch.cat([algo.obs_buf[:T], algo.aux_buf], -1).reshape(N, A, -1)
    msg = {"msg_obs": algo.msg_obs_buf.reshape(N, A, -1),
           "msg_mask": algo.msg_mask_buf.reshape(N, A, A)}
    with torch.no_grad():
        loc, scale, _ = policy_outputs(algo.actor, obs, algo.role_buf[:T].reshape(N, A),
                                       msg=msg)
    logp = tanh_normal_log_prob(algo.u_buf.reshape(N, A, -1), loc, scale)
    assert torch.allclose(logp, algo.logp_buf.reshape(N, A), atol=1e-4)
    stats = algo.update(algo.spec.entropy_coeff)
    assert stats["grad_skips"] == 0 and math.isfinite(stats["loss_objective"])


def test_greedy_env_arm_feeds_a_privileged_actor_the_aux_targets():
    env = build_env("navigation", n_envs=3, n_agents=4, device="cpu", dt=0.1, substeps=1,
                    max_steps=None, seed=0)
    _patch_training_hooks(env)
    torch.manual_seed(0)
    actor = PolicyNet(env.obs_dim, env.act_dim, 16, role_counts=ROLE_COUNTS, aux_dim=AUX_DIM,
                      priv_dim=AUX_DIM)
    obs = env.reset()
    roles = env.agent_roles()
    act = greedy_score_kwargs(actor)["policy_from_env"](env)
    first = act(obs)
    assert torch.equal(first, actor.greedy(torch.cat([obs, env.aux_targets()], -1), roles))
    moved = env.aux_targets() + 1.0
    env.scenario.aux_targets = lambda: moved
    assert not torch.allclose(act(obs), first)
    # Without roles or memory it still reads the env.
    alone = PolicyNet(env.obs_dim, env.act_dim, 16, aux_dim=AUX_DIM, priv_dim=AUX_DIM)
    act = greedy_score_kwargs(alone)["policy_from_env"](env)
    assert torch.equal(act(obs), alone.greedy(torch.cat([obs, moved], -1)))
    plain = PolicyNet(env.obs_dim, env.act_dim, 16)  # the arm acts, not returns the method
    assert torch.equal(greedy_env_arm(plain)(env)(obs), plain.greedy(obs))


def test_privileged_actor_is_refused_without_the_aux_targets():
    with pytest.raises(ValueError, match="needs actor='policynet' and aux_coef > 0"):
        replace(SPECS["navigation"], actor="policynet", privileged_actor=True)
    spec = replace(SPECS["navigation"], actor="policynet", aux_coef=0.5, privileged_actor=True)
    env = build_env("navigation", n_envs=4, n_agents=4, device="cpu", dt=0.05, substeps=1,
                    max_steps=5, seed=0)
    _patch_training_hooks(env)
    box = (-torch.ones(env.act_dim), torch.ones(env.act_dim))
    assert network_archs(spec, env, *box)[1]["priv_dim"] == AUX_DIM
    with pytest.raises(ValueError, match="not 'separate'"):
        network_archs(replace(spec, role_params="separate"), env, *box)


def _priv_algo(noise, device="cpu", patch=_patch_pose_aux, **kw) -> MAPPO:
    return _algo(graphs=device != "cpu", device=device, features=True, privileged_actor=True,
                 priv_noise=noise, patch=patch, **kw)


def test_zero_priv_noise_acts_on_the_truth_and_jitter_has_the_requested_std():
    """All-zero noise hands the actor the exact targets: the stored log-probs are the
    noise-free policy's on ``cat(obs, aux)``, and the update reads ``priv_buf``, equal to
    ``aux_buf``. A jitter alone moves the position columns by its std in metres and leaves
    the heading as it is."""
    algo = _priv_algo((0.0,) * 5)
    algo.rollout()
    T, A, N = algo.T, algo.A, algo.N
    assert torch.equal(algo.priv_buf, algo.aux_buf)
    obs = torch.cat([algo.obs_buf[:T], algo.aux_buf], -1).reshape(N, A, -1)
    with torch.no_grad():
        loc, scale, _ = policy_outputs(algo.actor, obs, algo.role_buf[:T].reshape(N, A))
    logp = tanh_normal_log_prob(algo.u_buf.reshape(N, A, -1), loc, scale)
    torch.testing.assert_close(logp, algo.logp_buf.reshape(N, A))
    algo = _priv_algo((0.0, 0.05, 0.0, 0.0, 0.0))
    algo.rollout()
    d = (algo.priv_buf - algo.aux_buf)[..., :2] / POSE_SCALE
    assert abs(d.std().item() - 0.05) < 0.005 and abs(d.mean().item()) < 0.005
    assert torch.equal(algo.priv_buf[..., 2:], algo.aux_buf[..., 2:])
    stats = algo.update(algo.spec.entropy_coeff)
    assert stats["grad_skips"] == 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_the_priv_noise_bias_holds_within_an_episode_and_is_redrawn_after_a_done(device):
    algo = _priv_algo((0.1, 0.0, 0.0, 0.0, 0.0), device=device, max_steps=3)
    for _ in range(2):  # the second one replays the captured graphs
        algo.rollout()
    d = (algo.priv_buf - algo.aux_buf)[..., :2] / POSE_SCALE  # [T, E, A, 2]
    assert d[0].abs().amax() > 0.05
    done = torch.logical_or(algo.term_buf > 0, algo.trunc_buf > 0)[:-1]  # [T - 1, E]
    step = (d[1:] - d[:-1]).abs().amax((-1, -2))
    assert done.any() and (~done).any()
    assert (step[~done] < 1e-4).all() and (step[done] > 1e-3).all()


def test_greedy_env_arm_holds_a_noisy_pose_s_bias_and_zero_noise_is_the_truth():
    env = build_env("navigation", n_envs=3, n_agents=4, device="cpu", dt=0.1, substeps=1,
                    max_steps=None, seed=0)
    _patch_pose_aux(env)
    actor = PolicyNet(env.obs_dim, env.act_dim, 16, aux_dim=4, priv_dim=4)
    obs = env.reset()
    truth = greedy_score_kwargs(actor)["policy_from_env"](env)(obs)
    exact = greedy_score_kwargs(actor, (0.0,) * 5)["policy_from_env"](env)
    assert torch.equal(exact(obs), truth)
    biased = greedy_env_arm(actor, (0.2, 0.0, 0.3, 0.0, 0.0))(env)
    first = biased(obs)
    assert torch.equal(biased(obs), first) and not torch.allclose(first, truth)


def test_priv_noise_is_refused_where_it_cannot_work():
    spec = replace(SPECS["navigation"], actor="policynet", aux_coef=0.5)
    with pytest.raises(ValueError, match="needs privileged_actor"):
        replace(spec, priv_noise=(0.0,) * 5)
    spec = replace(spec, privileged_actor=True)
    for bad in [(0.1, 0.1), (0.1, -0.1, 0.0, 0.0, 0.0)]:
        with pytest.raises(ValueError, match="5 values >= 0"):
            replace(spec, priv_noise=bad)
    with pytest.raises(ValueError, match=r"outlier_prob must lie in \[0, 1\)"):
        replace(spec, priv_noise=(0.0, 0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="aux_scale and aux_dim >= 4"):
        _priv_algo(_NOISE, patch=_patch_training_hooks)

    def with_flag(env):
        _patch_pose_aux(env)
        env.scenario.aux_goal = True

    with pytest.raises(ValueError, match="aux_goal=False"):
        _priv_algo(_NOISE, patch=with_flag)
    with pytest.raises(ValueError, match="the actor has none"):
        greedy_env_arm(PolicyNet(4, 2, 8), _NOISE)
    assert parse_args(["--scenario", "pusht_lidar", "--privileged-actor", "--aux-coef", "0.5",
                       "--actor", "policynet", "--priv-noise", *map(str, _NOISE)]
                      ).spec.priv_noise == _NOISE


# ------------------------------------------------------------------------- learned roles


@pytest.mark.parametrize("comm", ["none", "pose"])
def test_learned_roles_decide_every_dwell_and_recompute_the_rollout_log_probs(comm):
    """A role is chosen when its age reaches the dwell or after a done and held otherwise;
    the next step reads it as the previous role (0 after a done). The update recomputes the
    stored log-probabilities, messages encoded with the previous roles, and the role head
    moves them only on decision rows, where the role entropy is averaged."""
    kw = {"comm": comm, "comm_dim": 4, "msg_drop": 0.3} if comm != "none" else {}
    algo = _algo(graphs=False, device="cpu", features=True,
                 metrics={"role1_frac": 1, "switch_rate": 1}, **_LEARNED, **kw)
    age = algo.role_age.clone()
    algo.rollout()
    T, A, N, dwell = algo.T, algo.A, algo.N, algo.dwell
    ended = (torch.maximum(algo.term_buf, algo.trunc_buf) > 0).unsqueeze(-1)  # [T, E, 1]
    decide = algo.decide_buf > 0
    for t in range(T):
        assert torch.equal(decide[t], age >= dwell)
        age = (age + 1).masked_fill(decide[t], 1).masked_fill(ended[t], dwell)
    role, role_in = algo.lrole_buf, algo.lrole_in_buf
    assert torch.equal(role[~decide], role_in[:T][~decide])
    assert torch.equal(role_in[1:], role.masked_fill(ended, 0))
    assert ended.any() and (role != role_in[:T])[decide].any()
    assert algo.state_buf.shape[-1] == algo.env.state_dim + A * 3
    msg = {}
    if comm != "none":
        msg = {"msg_obs": algo.msg_obs_buf.reshape(N, A, -1),
               "msg_mask": algo.msg_mask_buf.reshape(N, A, A)}
    rows = {"role_in": role_in[:T].reshape(N, A), "role_out": role.reshape(N, A),
            "decide": algo.decide_buf.reshape(N, A)}
    obs, u = algo.obs_buf[:T].reshape(N, A, -1), algo.u_buf.reshape(N, A, -1)

    def logp() -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            loc, scale, _, logits = policy_outputs(
                algo.actor, obs, rows["role_in"], msg=msg, head_roles=rows["role_out"]
            )
        log_pr = logits.log_softmax(-1)
        pick = log_pr.gather(-1, rows["role_out"].unsqueeze(-1)).squeeze(-1)
        ent = -(log_pr.exp() * log_pr).sum(-1)
        return tanh_normal_log_prob(u, loc, scale) + rows["decide"] * pick, ent

    stored = algo.logp_buf.reshape(N, A)
    assert torch.allclose(logp()[0], stored, atol=1e-5)
    with torch.no_grad():
        algo.actor.role_head.weight.normal_()
    new, ent = logp()
    moved = (new - stored).abs() > 1e-6
    rows_d = rows["decide"] > 0
    assert moved[rows_d].any() and not moved[~rows_d].any()
    algo._prepare_body()
    with torch.no_grad():
        out = ppo_loss(
            algo.actor, algo.critic, obs, u, stored, algo.adv_buf.reshape(N, A),
            algo.ret_buf.reshape(N, A), torch.zeros_like(u), False,
            state=algo.state_buf[:T].reshape(N, -1), roles=algo.role_buf[:T].reshape(N, A),
            msg=msg or None, **rows,
        )
    torch.testing.assert_close(out[-1], ent[rows_d].mean())
    assert not torch.isclose(out[-1], ent.mean())
    stats = algo.update(algo.spec.entropy_coeff)
    assert stats["grad_skips"] == 0 and math.isfinite(stats["role_entropy"])
    # The logged switch rate is the rollout's switches over its decisions.
    assert stats["switch_rate"] == pytest.approx((role != role_in[:T])[decide].float().mean())


def test_teammate_role_columns_and_a_tied_warm_start_that_acts_as_before():
    """The teammates' mean previous-role one-hot follows the agent's own; a checkpoint
    without learned roles (plain, or sending pose messages) widens into one that acts in
    any previous and acting role as the checkpoint did in role 0, from uniform role
    logits."""
    torch.manual_seed(0)
    kw = {"role_counts": (3, 0), "aux_dim": 3}
    net = PolicyNet(6, 2, 16, learned_roles=True, **kw)
    x = net._with_role(torch.zeros(1, 3, 6), torch.tensor([[0, 1, 1]]))
    assert torch.equal(x[0, :, 8:], torch.tensor([[0.0, 1.0], [0.5, 0.5], [0.5, 0.5]]))
    for comm in ({}, {"comm": "pose", "comm_dim": 4}):
        plain = PolicyNet(6, 2, 16, **kw, **comm)
        learned = PolicyNet(6, 2, 16, learned_roles=True, role_dwell=2, **kw, **comm)
        assert "learned_roles" not in plain.config and learned.config["role_dwell"] == 2
        critic = ("Critic", {"obs_dim": 6, "n_agents": 3, "num_cells": 16})
        ckpt = {"format": "swarp.rl/2", "actor_cls": "PolicyNet", "actor_kwargs": plain.config,
                "critic_cls": critic[0], "critic_kwargs": critic[1]}
        check_resume("plain.pt", ckpt, ("PolicyNet", learned.config, *critic), widen=True)
        fresh = load_widened(learned, tie_learned_roles(learned, plain.state_dict()))
        assert fresh == ["role_head.weight", "role_head.bias"]
        obs = torch.randn(5, 3, 6)
        msg = {"msg_obs": torch.randn(5, 3, 6), "msg_mask": torch.rand(5, 3, 3) < 0.7}
        msg = msg if comm else {}
        ref = plain.step(obs, torch.zeros(5, 3, dtype=torch.long), **msg)
        for _ in range(3):
            prev, role = torch.randint(0, 2, (2, 5, 3))
            out = learned.step(obs, prev, head_roles=role, **msg)
            torch.testing.assert_close(out[0], ref[0])
            torch.testing.assert_close(out[1], ref[1])
            assert torch.equal(out[4], torch.zeros(5, 3, 2))


def test_greedy_arm_holds_a_learned_role_for_the_dwell():
    env = build_env("navigation", n_envs=3, n_agents=4, device="cpu", dt=0.1, substeps=1,
                    max_steps=None, seed=0)
    _patch_single_role(env)
    torch.manual_seed(0)
    actor = PolicyNet(env.obs_dim, env.act_dim, 16, role_counts=(4, 0), learned_roles=True,
                      role_dwell=3)
    obs = env.reset()
    with pytest.raises(ValueError, match="learns its roles"):
        actor.greedy(obs, env.agent_roles())

    def acting(prev: int, role: int) -> torch.Tensor:
        full = torch.full((3, 4), prev, dtype=torch.long)
        return actor.greedy(obs, full, head_roles=torch.full_like(full, role))

    act = greedy_score_kwargs(actor)["policy_from_env"](env)
    bias = actor.role_head.bias
    with torch.no_grad():
        bias.copy_(torch.tensor([0.0, 5.0]))
    assert torch.equal(act(obs), acting(0, 1))  # the first call decides
    with torch.no_grad():
        bias.copy_(torch.tensor([5.0, 0.0]))
    assert torch.equal(act(obs), acting(1, 1)) and torch.equal(act(obs), acting(1, 1))
    assert torch.equal(act(obs), acting(1, 0))  # held for the dwell, then decided again
    assert not torch.equal(acting(1, 1), acting(1, 0))


def test_learned_roles_are_refused_where_they_cannot_work():
    base = replace(SPECS["navigation"], actor="policynet", critic_input="state",
                   normalize_advantage=False)
    for bad in ({"actor": "mlp"}, {"role_params": "shared"}, {"critic_input": "obs"},
                {"normalize_advantage": True}):
        with pytest.raises(ValueError, match="learned_roles needs"):
            replace(base, learned_roles=True, **bad)
    with pytest.raises(ValueError, match="role_dwell must be >= 1"):
        replace(base, learned_roles=True, role_dwell=0)
    for counts in ((), (1, 3), (4,)):
        with pytest.raises(ValueError, match="roles past the first are empty"):
            PolicyNet(6, 2, 16, role_counts=counts, learned_roles=True)


# --------------------------------------------------------------------------- distillation


def test_a_teacher_is_recorded_on_the_rollout_and_distilled_into_the_actor_only():
    """The rollout records the frozen teacher's mean on ``cat(obs, aux targets)`` under the
    env's roles; ``loss_bc`` is positive against another network, zero for a student that
    is the teacher, and the update leaves the teacher as it was."""
    algo = _algo(graphs=False, device="cpu", features=True, teacher=_fresh_teacher)
    assert algo.stat_names == [*LEGACY_STAT_NAMES[:6], "loss_aux", "loss_bc", "grad_skips"]
    frozen = [p.clone() for p in algo.teacher.parameters()]
    assert not any(p.requires_grad for p in algo.teacher.parameters())
    algo.rollout()
    T = algo.T
    with torch.no_grad():
        obs = torch.cat([algo.obs_buf[:T], algo.aux_buf], -1)
        ref = algo.teacher.step(obs, algo.role_buf[:T])[0]
    torch.testing.assert_close(algo.teacher_loc_buf, ref)
    stats = algo.update(algo.spec.entropy_coeff)
    assert stats["loss_bc"] > 0 and stats["grad_skips"] == 0
    assert all(torch.equal(a, b) for a, b in zip(frozen, algo.teacher.parameters(), strict=True))
    same = _algo(graphs=False, device="cpu", features=True, privileged_actor=True,
                 teacher=lambda actor, env: copy.deepcopy(actor))
    same.rollout()
    same._prepare_body()
    same._minibatch_body()  # the first minibatch, before the actor has moved
    assert same.mb_stats[same.stat_names.index("loss_bc") - 3].item() < 1e-10


def test_a_teacher_is_refused_without_the_aux_targets_or_with_other_networks():
    with pytest.raises(ValueError, match="a teacher reads the aux targets"):
        replace(SPECS["navigation"], actor="policynet", teacher="teacher.pt")
    with pytest.raises(ValueError, match="needs aux_coef > 0"):
        _algo(graphs=False, device="cpu", actor="policynet", teacher=_fresh_teacher)
    with pytest.raises(ValueError, match="without memory"):
        _algo(graphs=False, device="cpu", features=True,
              teacher=partial(_fresh_teacher, recurrent=True))
    for more_obs, priv in ((0, AUX_DIM + 1), (1, AUX_DIM)):
        with pytest.raises(ValueError, match=r"the teacher's \(obs_dim, priv_dim"):
            _algo(graphs=False, device="cpu", features=True,
                  teacher=lambda actor, env, o=more_obs, d=priv: PolicyNet(
                      env.obs_dim + o, env.act_dim, 32, role_counts=env.role_counts,
                      aux_dim=AUX_DIM, priv_dim=d))
