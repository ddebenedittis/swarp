"""The lean MAPPO trainer in ``swarp.rl``: its maths, its graphs, and its plumbing."""

from __future__ import annotations

import csv
import gc
import math
from dataclasses import replace

import pytest
import torch

from swarp.rl import MAPPO, Actor, Critic, load_checkpoint, parse_args, train
from swarp.rl.networks import PolicyNet, StateCritic, role_permutation, tanh_normal_log_prob
from swarp.rl.ppo import (
    LEGACY_STAT_NAMES,
    gae,
    greedy_env_arm,
    greedy_score_kwargs,
    network_archs,
    policy_outputs,
    standardize_advantage,
)
from swarp.rl.recipes import SPECS, build_env, scenario_name, score

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
    **spec_kw,
) -> MAPPO:
    """Navigation at test size; ``features`` turns on the state critic (state + obs, per
    agent), role heads and the aux loss over ``patch`` (:func:`_patch_training_hooks`)."""
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
        actor = PolicyNet(
            Ob, env.act_dim, spec.num_cells, role_counts=ROLE_COUNTS,
            role_params=spec.role_params, aux_dim=AUX_DIM, **memory,
        )
        critic = StateCritic(
            env.state_dim, Ob, A, spec.num_cells, use_obs=True, role_counts=ROLE_COUNTS,
            per_agent=True,
        )
    elif spec.actor == "policynet":
        if patch is not _patch_training_hooks:
            patch(env)
        actor = PolicyNet(Ob, env.act_dim, spec.num_cells, **memory)
        critic = Critic(Ob, A, spec.num_cells)
    else:
        actor = Actor(Ob, env.act_dim, spec.num_cells)
        critic = Critic(Ob, A, spec.num_cells)
    algo = MAPPO(env, actor.to(device), critic.to(device), spec, graphs=graphs)
    algo.reset()
    return algo


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
            "state_buf", "role_buf", "aux_buf", "h_buf", "mask_buf")


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
    ],
    ids=["default", "features", "reset-roles", "recurrent", "comm", "recurrent-comm-features",
         "recurrent-separate"],
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
