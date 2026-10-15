"""Tests for the MAPPO example trainer/evaluator.

Deliberately *not* tested here: convergence, any reward value, GPU throughput (that is
``swarp.benchmark`` and CLAUDE.md's speed protocol), the render path (``tests/render/``
covers it), and anything slower than a few seconds. Nor is any CLI driven by subprocess --
``parse_args`` + ``train(cfg)`` is testable in-process, and subprocess tests serialize
badly against everything else on one GPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("torchrl", reason="examples/ needs the optional [torchrl] extra")

# The example scripts rely on Python putting the script's own directory on sys.path (that
# is how eval_mappo imports mappo), so a test importing them has to reproduce that.
#
# This lives here rather than in a tests/examples/conftest.py on purpose: the other suites
# do `from conftest import DEVICES`, i.e. they import `conftest` as a *top-level* module,
# and with no __init__.py in the test dirs the first conftest.py imported claims that name
# for the whole session. A tests/examples/conftest.py sorts before tests/interop/ and would
# shadow tests/conftest.py, breaking unrelated suites.
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

# Imports below the sys.path setup above, hence the noqa: E402 markers.
import torch  # noqa: E402
from mappo import (  # noqa: E402
    SPECS,
    UNTRAINABLE,
    CurriculumKnob,
    SolveRateWindow,
    SuccessSpec,
    build_critic,
    build_policy,
    coerce_scen_kwargs,
    mcnemar_interval,
    requires_success_spec,
    score,
    wilson_ci,
)
from train_mappo import parse_args, train  # noqa: E402

import swarp  # noqa: E402
from swarp.scenarios import SCENARIOS  # noqa: E402

# The saved state_dict keys encode the module nesting, and they are what makes an old
# checkpoint loadable. Read off the real runs/pusht_1a/pusht_final.pt.
POLICY_KEYS = {
    f"module.0.module.0.params.{i}.{s}" for i in (0, 2, 4) for s in ("weight", "bias")
}
CRITIC_KEYS = {f"module.params.{i}.{s}" for i in (0, 2, 4) for s in ("weight", "bias")}

SMOKE = dict(
    device="cpu", iters=2, n_envs=8, steps_per_batch=4, max_steps=5, epochs=1,
    minibatches=1, num_cells=8, substeps=1,
)


def _smoke_argv(scenario: str, ckpt_dir, **over) -> list[str]:
    args = {**SMOKE, **over}
    argv = ["--scenario", scenario, "--checkpoint-dir", str(ckpt_dir),
            "--eval-every", "0", "--checkpoint-every", "0"]
    for k, v in args.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


# ------------------------------------------------------------------- the spec table


def test_specs_cover_registry():
    """Every registered scenario is either trainable (has a spec) or explicitly not.

    This is the guard that adding a scenario does not silently omit it from
    ``--scenario``.
    """
    unaccounted = set(SCENARIOS) - set(SPECS) - set(UNTRAINABLE)
    assert not unaccounted, (
        f"scenario(s) {sorted(unaccounted)} have neither a TrainSpec nor an UNTRAINABLE "
        f"entry explaining why they cannot be trained"
    )
    assert not (set(SPECS) & set(UNTRAINABLE)), "a scenario cannot be both"
    assert set(SPECS) <= set(SCENARIOS)
    assert set(UNTRAINABLE) <= set(SCENARIOS)


def test_untrainable_scenarios_really_lack_a_terminal_condition():
    """The UNTRAINABLE list is a claim about the code; check it still holds."""
    for name in UNTRAINABLE:
        env = swarp.make(name, n_envs=2, device="cpu", n_agents=3)
        assert requires_success_spec(env.scenario), (
            f"{name} is listed UNTRAINABLE for lacking done(), but it now overrides it"
        )


def test_every_spec_metric_is_a_real_info_key():
    """A metric key not in info() would log a blank column forever."""
    for name, spec in SPECS.items():
        env = swarp.make(name, n_envs=2, device="cpu", n_agents=spec.n_agents,
                         **dict(spec.scen_kwargs))
        env.reset(seed=0)
        keys = set(env.scenario.info())
        assert set(spec.metrics) <= keys, (
            f"{name}: spec.metrics {sorted(set(spec.metrics) - keys)} not in info()"
        )


def test_every_curriculum_knob_names_a_real_attribute():
    for name, spec in SPECS.items():
        if not spec.curriculum:
            continue
        scen = swarp.scenarios.make_scenario(name, n_agents=spec.n_agents,
                                             **dict(spec.scen_kwargs))
        for knob in spec.curriculum:
            assert hasattr(scen, knob.attr), f"{name}: no attribute {knob.attr!r}"


def test_pusht_spec_reproduces_published_recipe():
    """``--scenario pusht`` must be a reproduction, not a re-derivation.

    Pins every number the original hardcoded script used, including that the two
    curriculum knobs evaluate to its exact schedules.
    """
    s = SPECS["pusht"]
    assert (s.dt, s.substeps, s.max_steps, s.n_agents) == (0.05, 8, 400, 4)
    assert (s.n_envs, s.steps_per_batch, s.epochs, s.minibatches) == (512, 32, 8, 4)
    assert (s.lr, s.gamma, s.lmbda) == (5e-4, 0.99, 0.95)
    assert (s.entropy_coeff, s.num_cells, s.normalize_advantage) == (3e-3, 256, False)
    assert dict(s.scen_kwargs) == {
        "pos_shaping_factor": 5.0, "rot_shaping_factor": 0.5, "rot_away_penalty": 1.0,
    }
    assert (s.curriculum_iters, s.curriculum_gate, s.iters) == (250, 0.0, 4000)
    assert s.metrics == ("tee_dist_to_goal", "tee_angle_error")

    radius, angle = s.curriculum
    assert (radius.attr, angle.attr) == ("goal_spawn_radius", "goal_spawn_angle")
    for f in (0.0, 0.1, 0.5, 0.9, 0.999):
        assert radius.value(f) == pytest.approx(0.25 + 1.9 * f)
        assert angle.value(f) == pytest.approx(0.4 + (math.pi - 0.4) * f)
    # None, not the numeric maximum: the scenario reads it as "unrestricted".
    assert radius.value(1.0) is None and angle.value(1.0) is None


# --------------------------------------------------------------- checkpoint compatibility


def test_build_policy_state_dict_keys():
    """The checkpoint-compat guard: these key strings are the saved-file contract."""
    policy = build_policy(9, 2, 3, "cpu", num_cells=8)
    critic = build_critic(9, 3, "cpu", num_cells=8)
    assert set(policy.state_dict()) == POLICY_KEYS
    assert set(critic.state_dict()) == CRITIC_KEYS


def test_build_policy_num_cells_default_is_load_bearing():
    """``eval_mappo._load`` falls back to 128 for checkpoints predating ``num_cells``."""
    import inspect

    assert inspect.signature(build_policy).parameters["num_cells"].default == 128


def test_loads_legacy_pusht_checkpoint():
    """A checkpoint written by the old Push-T script still loads."""
    path = REPO_ROOT / "runs" / "pusht_1a" / "pusht_final.pt"
    if not path.exists():
        pytest.skip("runs/pusht_1a/pusht_final.pt not present")
    from eval_mappo import _load

    policy, ckpt = _load(str(path), "cpu")
    assert (ckpt["obs_dim"], ckpt["act_dim"], ckpt["n_agents"]) == (12, 2, 1)
    assert "scenario" not in ckpt, "this file is the pre-scenario-key format under test"
    assert set(ckpt["policy"]) == POLICY_KEYS


def test_checkpoint_round_trip(tmp_path):
    """Save -> load -> identical greedy action for a fixed observation."""
    from eval_mappo import _load
    from mappo import greedy_actions

    cfg = parse_args(_smoke_argv("navigation", tmp_path))
    final = train(cfg)
    assert final.exists() and final.name == "navigation_final.pt"

    ckpt = torch.load(final, map_location="cpu", weights_only=True)
    assert ckpt["scenario"] == "navigation"
    assert ckpt["max_steps"] == 5 and ckpt["substeps"] == 1

    reloaded, _ = _load(str(final), "cpu")
    obs = torch.zeros(1, ckpt["n_agents"], ckpt["obs_dim"])
    lo = torch.full((ckpt["n_agents"], ckpt["act_dim"]), -1.0)
    hi = torch.full((ckpt["n_agents"], ckpt["act_dim"]), 1.0)
    a = greedy_actions(reloaded, 1, "cpu", lo, hi)(obs)
    b = greedy_actions(reloaded, 1, "cpu", lo, hi)(obs)
    torch.testing.assert_close(a, b)


def test_metrics_csv_header_and_rows(tmp_path):
    import csv

    cfg = parse_args(_smoke_argv("navigation", tmp_path))
    train(cfg)
    with (tmp_path / "metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    for col in ("iter", "frames", "wall_s", "env_sps", "reward_per_step",
                "ep_solve_rate", "ep_ends", "ep_len_mean", "difficulty", "entropy",
                "grad_skips"):
        assert col in rows[0], col
    # The old, misleading column must be gone rather than renamed.
    assert "solved" not in rows[0]
    for col in SPECS["navigation"].metrics:
        assert col in rows[0]
    for r in rows:
        assert not math.isnan(float(r["reward_per_step"]))


def test_multiobj_reward_expands_to_one_column_per_term(tmp_path):
    """A shaping imbalance must be visible per term, not summed away."""
    import csv

    cfg = parse_args(_smoke_argv("formation", tmp_path))
    train(cfg)
    with (tmp_path / "metrics.csv").open() as fh:
        cols = next(csv.reader(fh))
    terms = [c for c in cols if c.startswith("rew_term")]
    assert len(terms) >= 2, f"expected per-term columns, got {cols}"
    assert "multiobj_reward" not in cols


# ------------------------------------------------------------------------- CLI plumbing


def test_scen_kwarg_coercion():
    got = coerce_scen_kwargs(
        "navigation",
        ["n_obstacles=3", "agent_radius=0.07", "shared_reward=true",
         "goal_tolerance=none", "neighbor_method=brute"],
    )
    assert got == {
        "n_obstacles": 3, "agent_radius": 0.07, "shared_reward": True,
        "goal_tolerance": None, "neighbor_method": "brute",
    }
    assert isinstance(got["n_obstacles"], int)
    assert isinstance(got["agent_radius"], float)


def test_scen_kwarg_rejects_environment_keywords():
    """Routing is by name, so ``dt=`` would silently override the spec's dt."""
    for pair in ("dt=0.02", "substeps=4", "max_steps=10", "device=cpu"):
        with pytest.raises(ValueError, match="Environment keyword"):
            coerce_scen_kwargs("navigation", [pair])


def test_scen_kwarg_requires_k_equals_v():
    with pytest.raises(ValueError, match="k=v"):
        coerce_scen_kwargs("navigation", ["n_obstacles"])


def test_cli_overrides_beat_the_spec(tmp_path):
    cfg = parse_args(_smoke_argv("pusht", tmp_path, max_steps=7, substeps=2, iters=3))
    assert cfg.spec.max_steps == 7 and cfg.spec.substeps == 2 and cfg.spec.iters == 3
    # The smoke argv also sets num_cells, so check fields it does *not* touch.
    assert cfg.spec.dt == 0.05 and cfg.spec.lr == 5e-4
    assert cfg.spec.curriculum_iters == 250 and cfg.spec.n_agents == 4


def test_scen_kwarg_overrides_spec_scen_kwargs(tmp_path):
    cfg = parse_args(
        _smoke_argv("pusht", tmp_path) + ["--scen-kwarg", "pos_shaping_factor=2.5"]
    )
    assert cfg.scen_kwargs["pos_shaping_factor"] == 2.5
    assert cfg.scen_kwargs["rot_shaping_factor"] == 0.5  # spec value survives


def test_curriculum_knob_validates_attr(tmp_path):
    """A typo'd knob must fail at startup, not silently ramp nothing."""
    from dataclasses import replace

    cfg = parse_args(_smoke_argv("navigation", tmp_path))
    cfg.spec = replace(
        cfg.spec, curriculum=(CurriculumKnob("no_such_attr", 0.0, 1.0),),
        curriculum_iters=10,
    )
    with pytest.raises(ValueError, match="no_such_attr"):
        train(cfg)


def test_bad_metric_key_fails_at_startup(tmp_path):
    from dataclasses import replace

    cfg = parse_args(_smoke_argv("navigation", tmp_path))
    cfg.spec = replace(cfg.spec, metrics=("not_a_key",))
    with pytest.raises(ValueError, match="not_a_key"):
        train(cfg)


# ------------------------------------------------------------------ honest accounting


def test_episode_solve_rate_accounting():
    """The estimator is sum(terminated)/sum(done), and blank until enough ends."""
    w = SolveRateWindow(window=10, min_ends=4)
    # 8 env-steps per update, 2 ends of which 1 is a success.
    term = torch.zeros(8, dtype=torch.bool)
    done = torch.zeros(8, dtype=torch.bool)
    term[0] = True
    done[0] = True
    done[1] = True
    w.update(term, done)
    assert w.ends == 2
    assert w.solve_rate is None, "must stay blank below min_ends"
    assert w.ep_len_mean is None, "ep_len_mean is garbage before steady state"
    w.update(term, done)
    assert w.ends == 4
    assert w.solve_rate == pytest.approx(0.5)
    assert w.ep_len_mean == pytest.approx(16 / 4)


def test_episode_solve_rate_differs_from_the_old_per_step_proxy():
    """The old ``solved`` column was smaller by ~the episode length.

    ``runs/pusht_1a`` ended at solved=0.012 with max_steps=400, and the curriculum gate
    compared 0.012*400 = 4.8 against 0.35 -- open from the first iterations.
    """
    max_steps, n_envs = 400, 512
    term = torch.zeros(n_envs, dtype=torch.bool)
    done = torch.zeros(n_envs, dtype=torch.bool)
    # Every env ends this batch; half of them solved.
    done[:] = True
    term[: n_envs // 2] = True
    w = SolveRateWindow(window=1, min_ends=1)
    w.update(term, done)
    assert w.solve_rate == pytest.approx(0.5)
    old_proxy = term.float().mean().item() / max_steps  # per-step termination fraction
    assert old_proxy < w.solve_rate / 100, "the two scales must differ by ~max_steps"


def test_wilson_ci_stays_inside_the_unit_interval():
    for k, n in ((0, 100), (100, 100), (95, 100), (7, 2048)):
        lo, hi = wilson_ci(k, n)
        assert 0.0 <= lo <= hi <= 1.0
        # The interval brackets the point estimate; at k == n the upper bound is 1 only up
        # to rounding, so compare with a tolerance rather than exactly.
        assert lo - 1e-12 <= k / n <= hi + 1e-12
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_width_matches_the_sample_size_claim():
    """2048 envs is chosen for a roughly +/-0.02 half-width at p = 0.7."""
    lo, hi = wilson_ci(int(0.7 * 2048), 2048)
    assert (hi - lo) / 2 == pytest.approx(0.02, abs=0.003)


def test_mcnemar_interval_is_paired():
    a = torch.tensor([True, True, False, False])
    b = torch.tensor([True, False, True, False])
    diff, (lo, hi) = mcnemar_interval(a, b)
    assert diff == pytest.approx(0.0)  # one win each way
    assert lo <= 0.0 <= hi
    # Strictly dominating a -> positive difference.
    diff, _ = mcnemar_interval(torch.ones(4, dtype=torch.bool), b)
    assert diff == pytest.approx(0.5)


# --------------------------------------------------------------------- the solve predicate


@pytest.mark.parametrize("name", sorted(SPECS))
def test_trainable_scenarios_have_a_terminal_condition(name):
    """Every scenario with a spec must have something to train *toward*."""
    spec = SPECS[name]
    env = swarp.make(name, n_envs=2, device="cpu", n_agents=spec.n_agents,
                     **dict(spec.scen_kwargs))
    assert not requires_success_spec(env.scenario)


def test_solved_predicate_matches_pusht_hand_written():
    """The generic ``terminated`` predicate is the same tensor the old script recomputed.

    That identity is what makes the generic evaluator *reproduce* the published Push-T
    number rather than replace it with a similar-looking one.
    """
    env = swarp.make("pusht", n_envs=8, device="cpu", dt=0.05, substeps=8, n_agents=2,
                     seed=0)
    env.reset(seed=0)
    scen = env.scenario
    gen = torch.Generator(device="cpu").manual_seed(3)
    for _ in range(20):
        a = torch.empty(8, 2, env.act_dim).uniform_(-1, 1, generator=gen)
        _obs, _rew, terminated, _trunc, _info = env.step(a)
        d = (scen.tee_pos - scen.goal_pos).norm(dim=-1)
        raw = scen.tee_theta - scen.goal_theta
        ang = torch.atan2(raw.sin(), raw.cos()).abs()
        hand = (d < scen.goal_tolerance) & (ang < scen.angle_tolerance)
        assert torch.equal(terminated, hand)


def test_done_less_scenario_is_rejected_by_score():
    with pytest.raises(ValueError, match="does not override done"):
        score("flocking", n_agents=3, n_envs=4, steps=2, device="cpu", dt=0.05,
              substeps=1, arms=("random",))


def test_success_spec_fallback_works_for_a_done_less_scenario():
    res = score("sampling", n_agents=3, n_envs=8, steps=12, device="cpu", dt=0.05,
                substeps=1, arms=("random",),
                success=SuccessSpec("consumed_frac", "gt", 0.02))
    assert 0.0 <= res["random"].solve_rate <= 1.0


def test_random_and_policy_arms_are_paired():
    """Both arms must see bit-identical initial conditions.

    The whole policy-vs-baseline comparison, and the McNemar interval in particular, rests
    on this. Nothing else pins it.
    """
    policy = build_policy(19, 2, 4, "cpu", num_cells=8)
    res = score("navigation", policy=policy, n_agents=4, n_envs=16, steps=4,
                device="cpu", dt=0.05, substeps=1,
                metrics=("dist_to_goal",), arms=("policy", "random", "zero"))
    firsts = [r.metrics_first["dist_to_goal"] for r in res.values()]
    assert firsts[0] == firsts[1] == firsts[2], (
        f"arms started from different states: {firsts}"
    )


def test_scripted_baseline_beats_random_on_navigation():
    """Sanity check on the contrast arm itself: driving at the goal should work."""
    res = score("navigation", n_agents=4, n_envs=64, steps=40, device="cpu", dt=0.05,
                substeps=1, arms=("random", "scripted"),
                baseline=__import__("mappo").BASELINES["navigation"])
    assert res["scripted"].solve_rate > res["random"].solve_rate


# ------------------------------------------------------------------------- smoke tests


@pytest.mark.parametrize(
    "name",
    ["navigation", "formation", "discovery", "transport", "giveway", "shepherding",
     "caging"],
)
def test_train_smoke(name, tmp_path):
    """Two iterations end to end.

    This catches the shape/key/spec errors that are most of what breaks a training
    script, which is why it is parametrized over scenarios rather than run once.
    """
    cfg = parse_args(_smoke_argv(name, tmp_path))
    final = train(cfg)
    assert final.exists()
    ckpt = torch.load(final, map_location="cpu", weights_only=True)
    assert ckpt["scenario"] == name
    assert set(ckpt["policy"]) == POLICY_KEYS
    assert set(ckpt["critic"]) == CRITIC_KEYS
