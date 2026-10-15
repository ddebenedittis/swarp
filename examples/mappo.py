"""Shared MAPPO machinery for swarp's RL examples.

Both :mod:`train_mappo` and :mod:`eval_mappo` import from here, so a checkpoint written
by the trainer loads into a byte-identical module in the evaluator. Three groups of
things live here:

1. **The networks** — :func:`build_policy` / :func:`build_critic`. Decentralised actor
   with shared weights, centralised critic: the usual MAPPO split (which is what this is;
   IPPO would be the same code with ``centralised=False`` on the critic).
2. **The per-scenario recipe table** — :data:`SPECS`, a :class:`TrainSpec` per trainable
   scenario. See the note on that class for why this lives with the trainer rather than
   on the ``Scenario`` classes.
3. **Honest accounting** — :class:`SolveRateWindow`, :func:`wilson_ci`,
   :func:`mcnemar_interval` and :func:`score`, i.e. everything that turns a training run
   into a number you can quote.

Needs the optional torchrl extra::

    uv pip install -e '.[torchrl]'
"""

from __future__ import annotations

import ast
import csv
import inspect
import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import MultiAgentMLP, NormalParamExtractor, ProbabilisticActor, TanhNormal

import swarp
from swarp import Environment
from swarp.scenarios import scenario_class
from swarp.scenarios.base import Scenario

# ---------------------------------------------------------------- done/reward broadcast

# SwarpEnv is flat (no ("agents", ...) group) and emits one shared done per env,
# [n_envs, 1], while reward is per-agent [n_envs, n_agents, 1]. GAE needs the two
# broadcastable, so we expand done/terminated into these dedicated keys each batch
# and point the value estimator at them.
DONE_NAME, TERM_NAME = "agents_done", "agents_terminated"
DONE_KEY, TERM_KEY = ("next", DONE_NAME), ("next", TERM_NAME)


def expand_flag(t: torch.Tensor, n_agents: int) -> torch.Tensor:
    """``[*B, 1]`` shared flag -> ``[*B, n_agents, 1]``, matching the reward shape."""
    return t.unsqueeze(-2).expand(*t.shape[:-1], n_agents, 1)


# ------------------------------------------------------------------------- the networks


def build_policy(obs_dim: int, act_dim: int, n_agents: int, device: str, num_cells: int = 128):
    """Decentralised actor with shared weights (the MAPPO actor half).

    Shared with ``eval_mappo.py`` so a checkpoint loads into an identical module.

    **Do not restructure this function.** The saved ``state_dict`` keys encode the module
    nesting -- ``module.0.module.0.params.{0,2,4}.{weight,bias}`` -- so inserting a layer,
    changing ``depth``, or wrapping the actor differently silently invalidates every
    existing checkpoint. The ``num_cells=128`` default is likewise load-bearing:
    ``eval_mappo``'s loader falls back to it for checkpoints written before ``num_cells``
    was recorded.
    """
    net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * act_dim,  # loc + scale, split below
            n_agents=n_agents,
            centralised=False,
            share_params=True,
            device=device,
            depth=2,
            num_cells=num_cells,
        ),
        NormalParamExtractor(),
    )
    return ProbabilisticActor(
        module=TensorDictModule(net, in_keys=["observation"], out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True,
    )


def build_critic(obs_dim: int, n_agents: int, device: str, num_cells: int = 128):
    """Centralised critic (the MAPPO critic half): every agent's obs, one value.

    Its input width is ``n_agents * obs_dim``, which is why a critic cannot be transferred
    across an agent-count curriculum stage -- see the note in ``TrainSpec``.
    """
    return TensorDictModule(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=1,
            n_agents=n_agents,
            centralised=True,
            share_params=True,
            device=device,
            depth=2,
            num_cells=num_cells,
        ),
        in_keys=["observation"],
        out_keys=["state_value"],
    )


# ------------------------------------------------------------------ the per-scenario spec


@dataclass(frozen=True)
class CurriculumKnob:
    """One scenario attribute the trainer ramps as difficulty goes 0 -> 1.

    Declared as *data* rather than as a bare ``(scen, f) -> None`` callable so the trainer
    can validate the attribute exists at startup (a typo'd name would otherwise set a new
    attribute and the curriculum would silently do nothing, forever), log the resolved
    value, and serialize it into the checkpoint.
    """

    attr: str
    lo: float
    hi: float
    none_at_full: bool = True
    """Set the attribute to ``None`` at ``f >= 1``. Push-T's spawn knobs read ``None`` as
    "unrestricted", which is not the same as their numeric maximum."""
    exponent: float = 1.0
    """Shapes the ramp: the knob follows ``f ** exponent``, so ``1.0`` is linear and
    ``> 1.0`` spends most of the ramp near ``lo``. For a task whose difficulty is itself
    non-linear in this attribute, a linear knob is one threshold against a cliff -- see
    caging, where holding the cage goes from 86% of steps at 0.073 to 7% at 0.192."""

    def value(self, f: float) -> float | None:
        if f >= 1.0 and self.none_at_full:
            return None
        return self.lo + (self.hi - self.lo) * f ** self.exponent


@dataclass(frozen=True)
class TrainSpec:
    """The recipe that produced a checkpoint, for one scenario.

    **Why this table lives here and not on the ``Scenario`` classes.** The repo has
    precedent both ways and the line is clean: ``Scenario.parity_rtol`` lives on the
    scenario because two independent readers need it and only the scenario knows why its
    two paths differ. PPO hyperparameters have exactly one reader, and putting ``lr`` or
    ``minibatches`` on a ``Scenario`` would grow the simulator's public API a PPO
    vocabulary that the next consumer (a SAC example, a BPTT example) would immediately
    want to differ on.

    ``dt`` and ``substeps`` are the arguable cases -- Push-T's ``substeps=8`` is a
    stability floor of its contact law, i.e. a physics fact. They are duplicated here
    anyway, deliberately: this table is **provenance**, the recipe that reproduces a
    specific run, and that is what makes ``--scenario pusht`` a reproduction rather than a
    re-derivation.
    """

    # --- simulation ---------------------------------------------------------------
    dt: float = 0.1
    substeps: int = 1
    max_steps: int = 200
    n_agents: int = 4
    scen_kwargs: Mapping[str, Any] = field(default_factory=dict)
    # --- PPO ----------------------------------------------------------------------
    iters: int = 2000
    n_envs: int = 512
    steps_per_batch: int = 32
    epochs: int = 8
    minibatches: int = 4
    lr: float = 1e-3
    gamma: float = 0.99
    lmbda: float = 0.95
    entropy_coeff: float = 3e-3
    entropy_coeff_final: float | None = None
    """When set, ``entropy_coeff`` decays linearly to this over the run. Apt for
    deadlock-prone tasks that want broad early exploration and committed late behaviour."""
    num_cells: int = 256
    normalize_advantage: bool = True
    """torchrl's ``ClipPPOLoss`` defaults this to ``False``. With a large sparse terminal
    bonus on top of small dense shaping -- which is every scenario here -- unnormalized
    advantages swing the effective step size by orders of magnitude between iterations."""
    # --- curriculum ---------------------------------------------------------------
    curriculum: tuple[CurriculumKnob, ...] = ()
    curriculum_iters: int = 0
    """Iterations of *gated* progress to reach full difficulty; 0 disables. Gated, so this
    is a floor on the ramp rather than a schedule: a stalled run holds its difficulty and
    shows a flat ``difficulty`` column, which is the diagnostic you want."""
    curriculum_gate: float = 0.35
    curriculum_fn: Callable[[Scenario, float], None] | None = None
    """Escape hatch for anything the affine knobs cannot express. Applied after them."""
    # --- reporting ----------------------------------------------------------------
    metrics: tuple[str, ...] = ()
    """``scenario.info()`` keys to log, mean-reduced. A ``multiobj_reward`` key is
    expanded into one column per reward term, which is how a shaping imbalance becomes
    visible at iteration 1 instead of hour 3."""
    target_solve_rate: float = 0.7


# Scenarios with no success criterion: both inherit the all-False ``Scenario.done``, so an
# episode solve rate is identically 0 and there is nothing to train *toward*. Listed
# explicitly (with the reason) rather than silently omitted, and pinned by a test.
UNTRAINABLE: dict[str, str] = {
    "flocking": "no done() override: a flock has no terminal state to reach",
    "sampling": "no done() override: coverage is scored continuously, never 'achieved'",
}


SPECS: dict[str, TrainSpec] = {
    "navigation": TrainSpec(
        # 600 iters overtrained: greedy solve was 1.000 at iters 100-300, then the run
        # collapsed at ~330 (stochastic solve 1.00 -> 0.26) and the final greedy policy
        # fell to 0.69, with unsolved envs stuck at ~2 of 4 agents on goal and the rest
        # ~1.3 away (a standoff only exploration noise breaks, not a near miss).
        dt=0.05, substeps=4, max_steps=200, n_agents=4,
        iters=250, metrics=("dist_to_goal", "on_goal", "collisions"),
    ),
    "formation": TrainSpec(
        dt=0.05, substeps=4, max_steps=200, n_agents=4,
        iters=800, metrics=("formation_error", "multiobj_reward"),
    ),
    # The covering reward alone is too sparse: a random policy covers 0.45 of 5 targets
    # per episode against a -3 time-penalty return, and spec-only runs plateaued at ~0.27
    # solve. Progress shaping toward the nearest uncovered target supplies the gradient;
    # the lighter collision penalty (-0.02, from run r1) is what those runs used.
    "discovery": TrainSpec(
        dt=0.05, substeps=4, max_steps=300, n_agents=4,
        scen_kwargs={"pos_shaping_factor": 1.0, "collision_penalty": -0.02},
        iters=1000, n_envs=1024, metrics=("covered_frac",),
    ),
    # A single agent can move the package (a scripted "get behind it and push" controller
    # solves 93% with one agent), so this is not a physics problem. At the default
    # shaping factor 1.0 a whole episode's shaping sums to ~0.9 (the mean spawn distance)
    # against a 5.0 terminal bonus; 10.0 makes package progress the dominant signal.
    "transport": TrainSpec(
        dt=0.05, substeps=8, max_steps=300, n_agents=4,
        scen_kwargs={"pos_shaping_factor": 10.0},
        iters=1500, metrics=("package_dist_to_goal",),
    ),
    # The published Push-T recipe, with the changes below; pinned by
    # tests/examples/test_mappo.py::test_pusht_spec_reproduces_published_recipe.
    # The two shaping weights are not the scenario's defaults: per-step |dt angle| runs
    # ~8.7x |dt distance| under a random policy, so the raw 1.0/0.5 weights make the
    # (easier) rotation term dominate and the policy ignores position entirely.
    #
    # normalize_advantage=False is what the published script ran (torchrl's default). With
    # the now-real curriculum gate, normalizing held difficulty at 0.000 for 1400+ iters
    # (solve ~0.13-0.33 against a 0.35 gate); unnormalized, the gate opened and difficulty
    # reached 1.0 by iter ~560. lr 5e-4, not 1e-3: at 1e-3 the run reached 0.64 greedy at
    # full difficulty and then one update at iter 1079 blew the policy up (entropy
    # -2.6 -> -0.4, solve 0.59 -> 0.06) with no recovery. At 5e-4 the ramp takes ~1200
    # iters and the run ends at 0.71-0.75 greedy.
    #
    # curriculum_gate=0.0 (a fixed ramp to full difficulty by iter 250) and iters=4000.
    # Difficulty 0 is not an easy version of the task: it is only the final precise
    # placement, which a finished policy solves at 0.85 against 0.73 at full difficulty.
    # Its shaping is tiny (goals within 0.25), so seeds settled at a stochastic solve of
    # 0.13-0.30, under the 0.35 gate, and held difficulty at 0 for all 2100 iters. At full
    # difficulty the position shaping carries the signal and both seeds climb, to a final
    # greedy 0.875 (seed 0) and 0.981 (seed 1). 2100 iters is too short for that: at iter
    # 2000 the two seeds were at 0.62 and 0.90 greedy.
    #
    # rot_away_penalty=1.0: the team learned to spin the T and catch the goal heading on
    # a later lap (all agents torquing it the same way), since the wrapped heading error
    # refunds every full turn. Solved episodes turned it 8-9 rad against 1.6 needed. With
    # the cost, both seeds solve 0.99+ greedy at median step 44 and turn it ~2.6 rad.
    "pusht": TrainSpec(
        dt=0.05, substeps=8, max_steps=400, n_agents=4,
        scen_kwargs={
            "pos_shaping_factor": 5.0, "rot_shaping_factor": 0.5, "rot_away_penalty": 1.0,
        },
        iters=4000, n_envs=512, steps_per_batch=32, epochs=8, minibatches=4,
        lr=5e-4, entropy_coeff=3e-3, num_cells=256, normalize_advantage=False,
        curriculum=(
            CurriculumKnob("goal_spawn_radius", 0.25, 0.25 + 1.9),
            CurriculumKnob("goal_spawn_angle", 0.4, math.pi),
        ),
        curriculum_iters=250, curriculum_gate=0.0,
        metrics=("tee_dist_to_goal", "tee_angle_error"),
    ),
    # Give-way. Four agents, one per arm, crossing to the opposite arm through a junction
    # that fits one of them. shared_reward defaults True in the scenario: under per-agent
    # shaping the yielding agent pays the whole cost of a manoeuvre only the team benefits
    # from. substeps=8 is the contact law's floor at dt=0.05, not a preference.
    #
    # The curriculum knob is the arrival stagger: at difficulty 0 the agents reach the
    # junction at random, spread-out times, so conflicts are fewer and milder (the opposing
    # pair on each axis still has to pass, so it is not conflict-free: at penalty -1.0 the
    # scripted arm pays -6.4 collision per agent there, -24.0 at difficulty 1); at 1 they
    # arrive together. Ramping the *conflict density* rather than the geometry keeps the
    # success criterion fixed.
    "giveway": TrainSpec(
        dt=0.05, substeps=8, max_steps=200, n_agents=4,
        # Measured under a random policy: raw shaping averages +0.012/step against a
        # -0.017 collision term and a -0.010 time penalty, so at factor 1.0 the objective
        # is the smallest term in its own reward. 5.0 puts it clearly on top, the same
        # rebalance and the same reason as Push-T's pos_shaping_factor.
        #
        # collision_penalty -1.0 -> -0.25. At -1.0 the crossing is barely worth making:
        # measured per agent over one episode at difficulty 1.0, the scripted P controller
        # (which solves 100%) nets +15.5 -- shaping +35.1, collisions -24.0 -- against +12.0
        # for stopping at the junction mouth and never touching anyone. A novice policy's
        # first crossings collide more than the scripted one's, so stalling won: the 5e-3
        # run sat at difficulty 0, dist_to_goal ~1.08, on_goal 0 for all 3000 iterations.
        # At -0.25 the scripted crossing nets +33.5 against the same +12.0, and training
        # solves difficulty 0 within 100 iterations and full difficulty by ~450 -- with
        # collisions falling to ~4e-4/agent-step, so the policy still learns to avoid them.
        scen_kwargs={"pos_shaping_factor": 5.0, "collision_penalty": -0.25},
        iters=3000, n_envs=1024, steps_per_batch=32, minibatches=8,
        # Floored at 5e-3 rather than 1e-3. The 1e-3 schedule solved the task by iter 1250
        # and then destroyed it: entropy fell -4.48 -> -8.31 and `ep_solve_rate` went
        # 0.9999 (iter 1750) -> 0.443 (2250) -> 0.000 (2750), so the run ended on a dead
        # policy. `NormalParamExtractor` floors the std at its default 1e-4, which is no
        # floor at all, so the entropy bonus is the only thing holding the policy
        # stochastic. 5e-3 is the coefficient the run last held at iter 1750, where it was
        # both solving and stable; below ~4e-3 the collapse started.
        entropy_coeff=1e-2, entropy_coeff_final=5e-3,
        curriculum=(CurriculumKnob("difficulty", 0.0, 1.0, none_at_full=False),),
        curriculum_iters=400, curriculum_gate=0.35,
        metrics=("dist_to_goal", "on_goal", "collisions", "wall_contact",
                 "multiobj_reward"),
    ),
    # Shepherding. Dogs are the only agents; the sheep are scenario-owned obstacles with a
    # flee force, so the action space stays dogs-only. Longest horizon of the three -- the
    # dogs have to get *behind* the flock before any progress happens, which no amount of
    # shaping shortens.
    #
    # Measured under a random policy at difficulty 1.0: position shaping +0.0029/step,
    # spread shaping +0.0030, collision -0.0027 -- i.e. cohesion was pulling as hard as the
    # actual objective. 3.0 puts reaching the pen clearly on top; keeping the flock together
    # is instrumental, not the goal. (The -0.01 time penalty looks dominant but is a
    # per-step constant: it shifts values without distorting the policy gradient, and earns
    # its keep only through episode length.)
    "shepherding": TrainSpec(
        dt=0.05, substeps=8, max_steps=400, n_agents=3,
        scen_kwargs={"pos_shaping_factor": 3.0},
        iters=4000, n_envs=512, steps_per_batch=32, minibatches=4,
        entropy_coeff=1e-2, entropy_coeff_final=1e-3,
        curriculum=(CurriculumKnob("difficulty", 0.0, 1.0, none_at_full=False),),
        curriculum_iters=500, curriculum_gate=0.35,
        metrics=("pen_fraction", "sheep_spread", "multiobj_reward"),
    ),
    # Caging. The only scenario here whose objective is topological rather than metric: the
    # team must close the angular gaps around an evasive disc, not just get near it.
    #
    # Measured under a random policy at difficulty 1.0: gap shaping -0.018/step, band
    # shaping -0.022, collision -0.014 -- i.e. the radial band was pulling slightly harder
    # than the angular closure that *is* the task. 2.0 puts the topological term on top.
    #
    # The curriculum is verified reachable at its easy end: at difficulty 0 the agents spawn
    # on the cage ring with an inert disc, and a zero-action policy holds the cage for 100%
    # of steps and terminates in 100% of envs -- so the dwell bonus and the terminal are
    # both experienced from iteration 1. The hard end is verified reachable too: a scripted
    # ring-follower (P control to evenly spaced slots on the cage radius, plus the observed
    # disc velocity as feedforward) solves 0.95 of envs at difficulty 1, median step 22.
    "caging": TrainSpec(
        dt=0.05, substeps=8, max_steps=300, n_agents=5,
        # cage_reward/time_penalty: the scenario's 0.5/-0.01 make a caged step worth +0.49,
        # so terminating forfeits a positive stream and the policy learns to *avoid* done.
        # Measured on the old iter-500 checkpoint at difficulty 0: caged on 87.5% of steps,
        # never 10 in a row, ~37 deliberate cage breaks per episode, solve rate 0.000.
        # Net -0.05 while caged (vs -0.15 uncaged) keeps the dwell signal but makes the
        # terminal the best thing that can happen.
        scen_kwargs={"gap_shaping_factor": 2.0, "cage_reward": 0.1, "time_penalty": -0.15},
        iters=1500, n_envs=1024, steps_per_batch=32, minibatches=8,
        # Both knobs moved after the 0.35/1e-2 run died at iter 2454 with non-finite
        # gradients. The gate never opened -- `difficulty` held 0.053 for 2400 iterations
        # against an `ep_solve_rate` of ~0.07 -- so `loss_objective` stayed at ~0.003 while
        # the entropy bonus grew to ~0.05 (coeff 4.6e-3 x entropy 11.3), i.e. ~20x the
        # objective. With no upper bound on `scale`, maximising entropy was free reward:
        # the std ran away, TanhNormal log-probs went non-finite, and 47/64 minibatches
        # produced NaN gradients. 3e-3 stops the bonus out-weighing a weak objective.
        # (giveway collapsed the *opposite* way, for the same missing-std-bound reason --
        # see its spec above.) The gate itself was never the problem: solve sat at ~0.07
        # because the policy was farming the dwell bonus (see cage_reward above), and with
        # that fixed ep_solve_rate is 1.0 through difficulty 0.5, so it stays at 0.35.
        # Constant, not decayed to 1e-3: with the decay the run solved (greedy 0.978 at
        # iter ~1270) and then collapsed as the coefficient fell past ~2.3e-3 -- entropy
        # -1.3 -> -4.4, ep_solve_rate 0.96 -> 0.51 by iter 1750. Held at 3e-3 for 1500
        # iterations it ends solved (greedy 0.944) with entropy steady around -1.6.
        entropy_coeff=3e-3,
        # exponent 3.0, not linear. The measured cliff sits between difficulty 0.073
        # (`caged` 0.861) and 0.192 (`caged` 0.070), so a linear ramp spends ~19% of its
        # iterations below the cliff and the rest at a difficulty the policy cannot touch.
        # Cubed, half the ramp lands under 0.125 and 80% of it under 0.512, which puts the
        # iterations where the learning has to happen.
        curriculum=(
            CurriculumKnob("difficulty", 0.0, 1.0, none_at_full=False, exponent=3.0),
        ),
        curriculum_iters=600, curriculum_gate=0.35,
        metrics=("gap_max", "band_error", "caged", "multiobj_reward"),
    ),
}


# ------------------------------------------------------------- episode solve accounting


class SolveRateWindow:
    """Windowed, unbiased episode solve rate from the flags a batch already carries.

    ``SwarpEnv`` splits ``terminated`` (the scenario's ``done()``) from ``truncated``
    (the ``max_steps`` timeout), and the collector resets an env when ``done`` fires. So
    every episode ends exactly once, marked ``terminated`` iff the task condition fired,
    and over a window

        ep_solve_rate = sum(terminated) / sum(done)

    estimates the fraction of episode *starts* that get solved. It is unbiased: each env
    is a renewal process contributing one end per cycle, and a difference in length
    between solved and truncated episodes changes how *many* ends you observe, not the
    proportion of them that are marked. (The biased quantity would be the fraction of
    *steps* spent in solved episodes, which is a different statistic.)

    This replaces ``terminated.float().mean()``, a per-*step* termination fraction whose
    scale is ``1/episode_length``: at Push-T's settings it read 0.012 while the episode
    solve rate was of order 1, and the curriculum gate that compared
    ``0.012 * 400 = 4.8`` against 0.35 was therefore open from the first iterations.
    """

    def __init__(self, window: int = 25, min_ends: int = 200) -> None:
        self.window = window
        self.min_ends = min_ends
        self._term: deque[int] = deque(maxlen=window)
        self._done: deque[int] = deque(maxlen=window)
        self._steps: deque[int] = deque(maxlen=window)
        self._seen_end = False

    def update(self, terminated: torch.Tensor, done: torch.Tensor) -> None:
        n_done = int(done.sum().item())
        self._term.append(int(terminated.sum().item()))
        self._done.append(n_done)
        # Only count env-steps from the first episode end onward, so the ep_len_mean ratio
        # covers the same period in numerator and denominator. Without this, the whole
        # warm-up's steps sit over a denominator of one or two stragglers and the reported
        # length exceeds max_steps, which is impossible and reads as a bug.
        if n_done:
            self._seen_end = True
        self._steps.append(int(done.numel()) if self._seen_end else 0)

    @property
    def ends(self) -> int:
        return sum(self._done)

    @property
    def solve_rate(self) -> float | None:
        """``None`` until enough episodes have ended to mean anything.

        All envs start in phase, so the first ``ceil(max_steps / steps_per_batch)``
        iterations see zero ends; reporting 0.0 there would read as failure and, worse,
        would be fed to the curriculum gate.
        """
        n = self.ends
        if n < self.min_ends:
            return None
        return sum(self._term) / n

    @property
    def ep_len_mean(self) -> float | None:
        """Window mean episode length -- the truncation diagnostic.

        If this hugs ``max_steps`` while the solve rate is still climbing, successes are
        being truncated away and ``--max-steps`` is the thing to raise.

        Gated on ``min_ends`` for the same reason as :attr:`solve_rate`, and it matters
        more here: env-steps divided by ends is the mean episode length only once the
        window is in steady state. During warm-up the numerator counts a full window of
        steps while the denominator is one or two stragglers, which reads as an episode
        length of tens of thousands. Blank is the honest answer.

        Even after that, it is a steady-state estimator: episode ends arrive in waves of
        period ``max_steps / steps_per_batch`` because every env starts in phase, so the
        value oscillates for roughly the first window before settling. Read it as a coarse
        "are successes being truncated" signal, not a precise statistic.
        """
        n = self.ends
        if n < self.min_ends:
            return None
        return sum(self._steps) / n


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials.

    Not the normal approximation: near p = 0.95, where a good policy lands, the normal
    interval overshoots 1.0 and looks unserious.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_interval(
    a: torch.Tensor, b: torch.Tensor, z: float = 1.96
) -> tuple[float, tuple[float, float]]:
    """Paired difference ``P(a) - P(b)`` and its CI, for two masks over the *same* envs.

    Both arms of :func:`score` see bit-identical initial conditions, so this is a paired
    comparison and is strictly tighter than differencing two independent intervals. It is
    the statistic that supports "the policy solves cases the scripted controller cannot",
    as opposed to merely "the two numbers differ".
    """
    n = a.numel()
    if n == 0:
        return (0.0, (0.0, 0.0))
    b01 = int((a & ~b).sum().item())  # a solved, b did not
    b10 = int((~a & b).sum().item())
    diff = (b01 - b10) / n
    # Agresti-Min style variance for the paired difference of proportions.
    var = (b01 + b10 - (b01 - b10) ** 2 / n) / (n * n)
    half = z * math.sqrt(max(var, 0.0))
    return (diff, (diff - half, diff + half))


# ------------------------------------------------------------------------ CLI kwarg glue


def _coerce_one(raw: str, annotation: Any) -> Any:
    low = raw.strip().lower()
    text = str(annotation)
    if low in ("none", "null") and ("None" in text or annotation is inspect.Parameter.empty):
        return None
    if annotation is bool or text == "bool":
        if low in ("true", "1", "yes", "y"):
            return True
        if low in ("false", "0", "no", "n"):
            return False
        raise ValueError(f"{raw!r} is not a boolean")
    if annotation is int or text == "int":
        return int(raw)
    if annotation is float or "float" in text:
        return float(raw)
    if annotation is str or text == "str":
        return raw
    # Enums (DynamicsModel, ControlMode, ...) by member name.
    if isinstance(annotation, type) and issubclass(annotation, __import__("enum").Enum):
        return annotation[raw.upper()]
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def coerce_scen_kwargs(name: str, pairs: Sequence[str]) -> dict[str, Any]:
    """Parse repeated ``--scen-kwarg k=v`` into typed scenario keywords.

    Types come from the scenario's own ``__init__`` annotations, so no per-scenario code
    is needed. *Validation* of unknown names is deliberately left to :func:`swarp.make`,
    which already raises a legible ``TypeError`` listing the scenario's keywords plus a
    ``difflib`` suggestion -- one error message, one place.

    The one check ``swarp.make`` cannot do for us: reject a name that belongs to
    ``Environment``. Because ``make`` routes by name, ``--scen-kwarg dt=0.02`` would be
    silently accepted as an ``Environment`` keyword and quietly override the spec's ``dt``.
    ``tests/scenarios/test_scenarios.py`` pins that the two parameter-name sets are
    disjoint for every registered scenario, so this can never false-positive.
    """
    params = inspect.signature(scenario_class(name).__init__).parameters
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--scen-kwarg expects k=v, got {pair!r}")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if key in swarp._ENV_KWARGS:
            raise ValueError(
                f"{key!r} is an Environment keyword, not a scenario one; use the "
                f"dedicated --{key.replace('_', '-')} flag so the spec is not silently "
                f"overridden."
            )
        ann = params[key].annotation if key in params else inspect.Parameter.empty
        out[key] = _coerce_one(raw, ann)
    return out


# ----------------------------------------------------------------------- the action arms

ActionFn = Callable[[torch.Tensor], torch.Tensor]


def greedy_actions(
    policy, n_envs: int, device: str, lo: torch.Tensor, hi: torch.Tensor
) -> ActionFn:
    """``obs -> action`` from the distribution mean: no exploration noise, no RNG.

    Consuming no randomness is what keeps this arm *paired* with the random arm -- both
    see bit-identical initial conditions.
    """

    def action_fn(obs: torch.Tensor) -> torch.Tensor:
        td = TensorDict({"observation": obs}, batch_size=[n_envs], device=device)
        with torch.no_grad():
            policy(td)
            return td.get("loc").clamp(lo, hi)

    return action_fn


def random_actions(
    n_envs: int, n_agents: int, act_dim: int, device: str, seed: int,
    lo: torch.Tensor, hi: torch.Tensor,
) -> ActionFn:
    """Uniform over the *real* action box.

    Not a hardcoded ``[-1, 1]^2``: swarp actions are physical, and the box differs by
    dynamics model (a bicycle's second slot is a steering angle; a drone under ``[-1, 1]``
    cannot even reach hover). ``swarp/interop/torchrl.py`` documents this at length.

    Drawn from a dedicated generator rather than the env RNG, so the env's initial
    conditions are untouched and this arm stays paired with the others.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    def action_fn(obs: torch.Tensor) -> torch.Tensor:
        u = torch.rand(n_envs, n_agents, act_dim, device=device, generator=gen)
        return lo + (hi - lo) * u

    return action_fn


def zero_actions(n_envs: int, n_agents: int, act_dim: int, device: str) -> ActionFn:
    """Do nothing. Informative wherever inaction differs from thrashing."""
    z = torch.zeros(n_envs, n_agents, act_dim, device=device)
    return lambda obs: z


def goal_p_controller(gain: float = 4.0) -> Callable[[Environment], ActionFn]:
    """Scripted contrast arm: drive straight at ``world.goals`` with a P controller.

    This is what makes a solve rate a *claim* rather than a number. On navigation it
    nearly solves the task, which is a useful honesty check on the whole pipeline -- if
    MAPPO cannot beat ten lines of proportional control there, the trainer is broken. On a
    scenario whose solution requires yielding, it should deadlock, and the gap between the
    two arms is the evidence that something non-greedy was learnt.
    """

    def make(env: Environment) -> ActionFn:
        world = env.world
        lo, hi = env.action_bounds

        def action_fn(obs: torch.Tensor) -> torch.Tensor:
            err = world.goals - world.state.pos
            # act_dim can exceed 2 for a mixed fleet; only the first two slots are the
            # holonomic velocity command this controller knows how to drive.
            cmd = torch.zeros(*err.shape[:-1], world.act_dim, device=err.device, dtype=err.dtype)
            cmd[..., :2] = gain * err
            return cmd.clamp(lo, hi)

        return action_fn

    return make


# Per-scenario scripted baseline, keyed like SPECS. Only for scenarios where "drive at the
# goal" is a meaningful controller; a scenario with no per-agent goal has no entry.
BASELINES: dict[str, Callable[[Environment], ActionFn]] = {
    "navigation": goal_p_controller(),
    # Measured, this does *not* deadlock: under the stiff contact the head-on pairs shove
    # each other into the perpendicular arms and it solves 100% at difficulty 1.0 (median
    # step 48), paying heavily in collisions. So on giveway this arm is a ceiling on the
    # solve rate, not a floor; compare collisions and solve step instead.
    "giveway": goal_p_controller(),
}


# --------------------------------------------------------------------------- the scorer


@dataclass
class ArmResult:
    solve_rate: float
    ci: tuple[float, float]
    n: int
    solve_step_median: float | None
    metrics_first: dict[str, float]
    metrics_last: dict[str, float]
    solved_mask: torch.Tensor


@dataclass(frozen=True)
class SuccessSpec:
    """Fallback success test for a scenario with no ``done()`` override."""

    key: str
    cmp: str  # "lt" or "gt"
    threshold: float

    def evaluate(self, info: Mapping[str, torch.Tensor]) -> torch.Tensor:
        v = info[self.key]
        hit = v < self.threshold if self.cmp == "lt" else v > self.threshold
        while hit.ndim > 1:
            hit = hit.all(dim=-1)
        return hit


def requires_success_spec(scen: Scenario) -> bool:
    """``True`` iff this scenario inherits the never-terminate default ``done()``.

    Exact and free. Without this check the evaluator prints ``solved 0.000``, which reads
    as "the policy failed" when the truth is "this scenario has no success criterion".
    """
    return type(scen).done is Scenario.done


def build_env(
    scenario: str, *, n_envs: int, n_agents: int, device: str, dt: float, substeps: int,
    max_steps: int | None, seed: int, scen_kwargs: Mapping[str, Any] = (),
) -> Environment:
    """One place that turns a spec into an ``Environment``.

    Goes through :func:`swarp.make`, which already routes keywords by name and raises a
    legible ``TypeError`` (with a ``difflib`` suggestion) for a misspelling -- there is no
    reason for this file to reimplement any of that. ``swarp.make`` does not pass
    ``auto_reset``, so it stays ``False``, which is what ``SwarpEnv`` requires.
    """
    return swarp.make(
        scenario, n_envs=n_envs, device=device, dt=dt, substeps=substeps,
        seed=seed, max_steps=max_steps, n_agents=n_agents, **dict(scen_kwargs),
    )


def score(
    scenario: str,
    *,
    policy=None,
    n_agents: int,
    n_envs: int = 2048,
    steps: int = 400,
    device: str = "cpu",
    seed: int = 123,
    dt: float = 0.05,
    substeps: int = 8,
    scen_kwargs: Mapping[str, Any] = (),
    metrics: Sequence[str] = (),
    baseline: Callable[[Environment], ActionFn] | None = None,
    arms: Sequence[str] = ("policy", "random", "zero", "scripted"),
    success: SuccessSpec | None = None,
) -> dict[str, ArmResult]:
    """Episode solve rate per arm, over one un-reset episode of ``steps`` steps.

    An env counts as solved if its terminal condition fires at **any** point, because that
    is where the episode would have ended in training. The rollout deliberately keeps
    simulating a terminated env, which may then drift back out of tolerance -- it still
    counts. (Looks like a bug otherwise, hence this note.)

    ``max_steps=None`` and ``auto_reset=False``, so the episode is genuinely single and
    ``truncated`` never fires. Each arm gets a **fresh env on the same seed**, and the
    policy arm consumes no RNG, so the arms are paired: differences are attributable to
    the controller, not to the initial conditions. ``tests/examples`` pins that.
    """
    out: dict[str, ArmResult] = {}
    for name in arms:
        if name == "policy" and policy is None:
            continue
        if name == "scripted" and baseline is None:
            continue
        env = build_env(
            scenario, n_envs=n_envs, n_agents=n_agents, device=device, dt=dt,
            substeps=substeps, max_steps=None, seed=seed, scen_kwargs=scen_kwargs,
        )
        scen = env.scenario
        if success is None and requires_success_spec(scen):
            raise ValueError(
                f"scenario {scenario!r} does not override done(), so its episode solve "
                f"rate is identically 0 and would be reported as policy failure. Pass a "
                f"success spec (--success-key/--success-cmp/--success-threshold) naming an "
                f"info() key to threshold instead. Reason: "
                f"{UNTRAINABLE.get(scenario, 'inherits Scenario.done')}"
            )
        obs = env.reset(seed=seed)
        lo, hi = env.action_bounds
        act_dim = env.world.act_dim
        if name == "policy":
            act = greedy_actions(policy, n_envs, device, lo, hi)
        elif name == "random":
            act = random_actions(n_envs, n_agents, act_dim, device, seed, lo, hi)
        elif name == "zero":
            act = zero_actions(n_envs, n_agents, act_dim, device)
        else:
            act = baseline(env)

        solved = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first = torch.full((n_envs,), -1, dtype=torch.long, device=device)
        m_first: dict[str, float] = {}
        m_last: dict[str, float] = {}
        with torch.no_grad():
            info0 = scen.info()
            for k in metrics:
                if k in info0:
                    m_first[k] = info0[k].float().mean().item()
            for t in range(steps):
                a = act(obs)
                obs, _rew, terminated, _trunc, info = env.step(a)
                hit = success.evaluate(info) if success is not None else terminated
                newly = hit & ~solved
                first = torch.where(newly, torch.full_like(first, t), first)
                solved |= hit
            for k in metrics:
                if k in info:
                    m_last[k] = info[k].float().mean().item()

        k_solved = int(solved.sum().item())
        steps_of_solved = first[solved]
        out[name] = ArmResult(
            solve_rate=k_solved / n_envs,
            ci=wilson_ci(k_solved, n_envs),
            n=n_envs,
            solve_step_median=(
                float(steps_of_solved.float().median().item()) if k_solved else None
            ),
            metrics_first=m_first,
            metrics_last=m_last,
            solved_mask=solved,
        )
    return out


def plot_curve(path: str) -> None:
    """ASCII learning curves from a trainer ``metrics.csv`` (no plotting deps).

    Header-driven rather than hardcoded to a fixed column triple, so both the current
    schema and the older Push-T one plot, each showing whatever numeric columns it has.
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"{path} is empty")
        return
    skip = {"iter", "frames", "wall_s", "ep_ends"}
    for col in rows[0]:
        if col in skip:
            continue
        vals = []
        for r in rows:
            try:
                vals.append(float(r[col]))
            except (TypeError, ValueError):
                continue  # blank cells (warm-up, non-eval iterations)
        if len(vals) < 2:
            continue
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        step = max(1, len(vals) // 60)
        pts = [sum(vals[i : i + step]) / len(vals[i : i + step]) for i in range(0, len(vals), step)]
        bars = "".join(" ▁▂▃▄▅▆▇█"[int((v - lo) / span * 8)] for v in pts)
        print(f"  {col:24s} [{lo:+.4f} .. {hi:+.4f}]\n    {bars}")
