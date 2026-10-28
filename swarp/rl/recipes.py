"""Per-scenario training recipes and the honest-accounting helpers around them.

Torch-only (no torchrl), so both trainers share it: the lean CUDA-graph trainer in
:mod:`swarp.rl.ppo` and the TorchRL reference in ``examples/train_mappo.py``, which
re-exports everything here through ``examples/mappo.py``.

1. **The recipe table** -- :data:`SPECS`, a :class:`TrainSpec` per trainable scenario,
   plus :data:`UNTRAINABLE` for the two that have no success criterion.
2. **Honest accounting** -- :class:`SolveRateWindow`, :func:`wilson_ci`,
   :func:`mcnemar_interval` and :func:`score`, i.e. everything that turns a training run
   into a number you can quote.
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

import swarp
from swarp import Environment
from swarp.rl.networks import COMMS, ROLE_PARAMS
from swarp.scenarios import scenario_class
from swarp.scenarios.base import Scenario

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
class Ramp:
    """One scenario attribute the trainer ramps from ``lo`` to ``hi`` over iterations
    ``[start, end)``: what a staged recipe does by resuming under new keywords, in one run.

    Unlike a :class:`CurriculumKnob` it follows the iteration count, not the solve-gated
    difficulty, so a cost or a noise can come in only once the policy has learned without
    it. A value is a float or a tuple of floats; an ``int`` in ``hi`` rounds its entry. The
    ramp moves in ``steps`` increments, and the trainer recaptures the env's graph on each:
    a captured step bakes in every scalar it reads by value, so a changed reward weight
    would otherwise never reach it.
    """

    attr: str
    lo: float | tuple[float, ...]
    hi: float | tuple[float, ...]
    start: int
    end: int
    steps: int = 20

    def progress(self, it: int) -> float:
        """The quantized fraction of the ramp done at iteration ``it``, in ``[0, 1]``."""
        if it >= self.end:
            return 1.0
        f = max(0.0, (it - self.start) / max(self.end - self.start, 1))
        return math.floor(f * self.steps) / self.steps

    def value(self, it: int) -> float | tuple[float, ...]:
        f = self.progress(it)

        def lerp(lo: float, hi: float) -> float:
            v = lo + (hi - lo) * f
            return round(v) if isinstance(hi, int) else v

        if isinstance(self.hi, tuple):
            return tuple(lerp(lo, hi) for lo, hi in zip(self.lo, self.hi, strict=True))
        return lerp(self.lo, self.hi)


#: ``TrainSpec.actor`` values: the obs-only Actor or the role-aware PolicyNet.
ACTORS = ("mlp", "policynet")
#: ``TrainSpec.critic_input`` values.
CRITIC_INPUTS = ("obs", "state", "state+obs")


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
    scenario: str | None = None
    """The registered scenario this recipe builds, for a recipe keyed by another name (a
    variant of a scenario, such as its physical-scale configuration). ``None`` means the
    ``SPECS`` key itself; :func:`scenario_name` resolves either."""
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
    lr_final: float | None = None
    """When set, ``lr`` decays linearly to this over the run."""
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
    target_kl: float | None = None
    """When set, an iteration's remaining minibatch steps are skipped on device once one
    minibatch's approximate KL to the rollout policy, ``mean((r - 1) - log r)``, exceeds
    ``1.5 * target_kl`` (that minibatch's own step included). A brake on destructive
    updates: a staged slim-T Limo run fell 0.83 -> 0.22 in 60 iterations of 128 steps
    each, no step non-finite."""
    actor_scale_min: float | None = None
    """When set, the floor under the actor's Gaussian std (``PolicyNet``'s ``scale_min``,
    0.01 by default). A higher floor keeps a policy from sharpening faster than it learns;
    the greedy action is the mean, so the deployed policy is unaffected."""
    adaptive_kl: float | None = None
    """When set, the learning rate follows the KL (rsl_rl's adaptive schedule, per
    iteration): its multiplier is divided by 1.5 after an iteration whose mean approximate
    KL exceeds ``2 * adaptive_kl`` and multiplied by 1.5 after one below ``adaptive_kl / 2``,
    within ``[1/64, 1]`` of the scheduled rate. A policy that sharpens (its entropy falling)
    moves further per step; this slows it there instead of freezing the iteration as
    ``target_kl`` does."""
    rollback: bool = False
    """Roll the actor, critic and Adam back to an earlier snapshot when the windowed solve
    rate collapses, halving the learning rate, instead of training on a dead policy (see
    :class:`~swarp.rl.ppo.RollbackGuard`). The run stops, its best checkpoint saved, after
    ``RollbackGuard.max_rollbacks`` of them."""
    # --- curriculum ---------------------------------------------------------------
    curriculum: tuple[CurriculumKnob, ...] = ()
    curriculum_iters: int = 0
    """Iterations of *gated* progress to reach full difficulty; 0 disables. Gated, so this
    is a floor on the ramp rather than a schedule: a stalled run holds its difficulty and
    shows a flat ``difficulty`` column, which is the diagnostic you want."""
    curriculum_gate: float = 0.35
    curriculum_fn: Callable[[Scenario, float], None] | None = None
    """Escape hatch for anything the affine knobs cannot express. Applied after them."""
    ramps: tuple[Ramp, ...] = ()
    """Scenario attributes on an iteration schedule (:class:`Ramp`). With any, the best
    checkpoint is ranked on how far they have run before the solve rate, so a policy that
    solved before the noise came in does not stay "best"."""
    ramp_gate: float | None = None
    """When set, the ramps follow a gated clock instead of the iteration: it runs freely up
    to the first ramp's start, and from there advances only on iterations whose windowed
    solve rate is at least this, so no condition comes in before the policy has mastered
    the last. The learning rate then decays over that clock (to ``lr_final`` at the last
    ramp's end plus ``hold_iters``), and the run ends there, ``iters`` being a cap. The
    entropy schedule stays on the iteration count."""
    hold_iters: int = 0
    """With ``ramp_gate``: clock ticks to train at the final conditions after the last ramp
    ends."""
    # --- reporting ----------------------------------------------------------------
    metrics: tuple[str, ...] = ()
    """``scenario.info()`` keys to log, mean-reduced. A ``multiobj_reward`` key is
    expanded into one column per reward term, which is how a shaping imbalance becomes
    visible at iteration 1 instead of hour 3."""
    target_solve_rate: float = 0.7
    # --- rollout ------------------------------------------------------------------
    stagger_starts: bool = False
    """Desynchronise episode starts before the first rollout (see ``MAPPO.reset``). For a
    scenario whose episodes mostly run to ``max_steps``: without it every env truncates on
    the same step and each batch is one slice of the episode, in phase across all envs."""
    # --- networks -----------------------------------------------------------------
    # The defaults are the plain shared-weight Actor and obs-only Critic. The rest needs
    # the scenario's training hooks (``Scenario.critic_state`` / ``agent_roles`` /
    # ``aux_targets``); see docs/training.md.
    actor: str = "mlp"
    """``"mlp"``: :class:`~swarp.rl.networks.Actor`, obs only. ``"policynet"``:
    :class:`~swarp.rl.networks.PolicyNet`, which reads the scenario's roles (if it has
    any) through ``role_params`` and carries the aux head when ``aux_coef > 0``."""
    role_params: str = "heads"
    """How a ``"policynet"`` actor's parameters depend on the role: ``"shared"`` (role as an
    input only), ``"heads"`` (one output head per role) or ``"separate"`` (one network per
    role). Ignored for ``"mlp"`` and for a scenario without roles."""
    critic_input: str = "obs"
    """``"obs"`` (every agent's observation), ``"state"`` (the scenario's privileged
    ``critic_state()``) or ``"state+obs"``."""
    per_agent_value: bool = False
    """One value per agent (per role-sorted slot when the scenario has roles) instead of
    one team value broadcast to every agent."""
    aux_coef: float = 0.0
    """Weight of the auxiliary loss ``mse(aux_head(features), scenario.aux_targets())``;
    0 disables it. Above 0 it needs ``actor="policynet"`` and a scenario with
    ``aux_dim > 0``."""
    privileged_actor: bool = False
    """Append ``scenario.aux_targets()`` to the actor's input, as columns after everything
    else (``PolicyNet``'s ``priv_dim``): the truth an aux head would estimate, handed over.
    An experiment, not a deployable policy: it tells whether a task fails on perception.
    Needs ``actor="policynet"`` and ``aux_coef > 0``."""
    priv_noise: tuple[float, ...] = ()
    """Train the privileged actor under a pose estimator's errors: ``(pos_bias_m,
    pos_jitter_m, heading_bias_rad, heading_jitter_rad, outlier_prob)``. Per robot, a
    Gaussian bias drawn once per episode plus a Gaussian jitter per step perturb the
    T-pose columns of ``pusht_lidar``'s aux layout (relative position over
    ``aux_scale``, heading as cos/sin), and with ``outlier_prob`` per robot-step the
    estimate is a gross error (position uniform in a 1 m box around the truth, heading
    uniform). Only the actor's input is noisy: the aux loss and a teacher keep the truth.
    Empty: off. Needs ``privileged_actor`` and a flag-free aux layout (``aux_goal=False``),
    since a noisy pose next to a true on-goal flag leaks the truth."""
    actor_drop: tuple[str, ...] = ()
    """``obs_layout()`` blocks the actor does not read (``PolicyNet``'s ``obs_keep``), e.g.
    ``("lidar",)`` for a privileged actor that acts on the poses alone. Needs
    ``actor="policynet"``."""
    learned_roles: bool = False
    """Each agent chooses its role (``PolicyNet``'s ``learned_roles``): every
    ``role_dwell`` steps a role head samples it, the choice enters the PPO ratio, and the
    role is held in between. Needs ``actor="policynet"``, ``role_params="heads"``, a
    critic that reads the state (the roles and their ages are appended to it),
    ``normalize_advantage=False`` (per-role statistics assume fixed counts), and a
    scenario whose roles past the first are empty, like ``role_counts`` ``(A, 0)``."""
    role_dwell: int = 5
    """Steps a learned role is held between decisions; an episode's first step decides."""
    role_entropy_coeff: float = 0.01
    """Weight of the learned roles' entropy bonus: the exact categorical entropy averaged
    over the decision steps."""
    teacher: str = ""
    """Path to a :func:`~swarp.rl.ppo.train` checkpoint of a privileged ``PolicyNet``
    (``priv_dim`` = the scenario's ``aux_dim``, not recurrent, not communicating) to distil
    into the actor: on the actor's own rollouts the teacher's pre-tanh mean is recorded, and
    the update adds ``bc_coef * mse(actor mean, teacher mean)``. Empty: no teacher. Needs
    ``actor="policynet"`` and ``aux_coef > 0`` (the teacher reads the aux targets)."""
    bc_coef: float = 1.0
    """Weight of the distillation loss. Ignored without ``teacher``."""
    bc_decay_iters: int = 0
    """Iterations over which ``bc_coef`` decays linearly to 0; 0 keeps it constant. Counted
    from the run's first iteration, a resumed run included."""
    action_rate_coef: float = 0.0
    """Weight of the action-rate penalty: every agent's reward loses
    ``action_rate_coef * mean_i(((a_t - a_{t-1}) / half_i)^2)``, ``half_i`` the action
    box's half-width, with ``a_{t-1} = 0`` on an episode's first step (a robot starts at
    rest). 0 disables it; above 0 the trainer logs the mean squared change as
    ``action_rate``."""
    action_rate_ramp_iters: int = 0
    """Iterations over which the action-rate weight ramps linearly from 0 to
    ``action_rate_coef``; 0 starts at the full weight. Counted from the run's first
    iteration, a resumed run included."""
    # --- memory and communication (a ``"policynet"`` actor only) ----------------------
    recurrent: bool = False
    """A GRU after the policy's trunk, its hidden state carried across steps and reset at
    episode ends; the update backpropagates through time over ``seq_len`` chunks."""
    hidden_dim: int = 128
    """Width of the GRU's hidden state. Ignored unless ``recurrent``."""
    seq_len: int = 0
    """Length of the update's BPTT chunks; 0 means ``steps_per_batch`` (one chunk per env
    and rollout). Must divide ``steps_per_batch``. Ignored unless ``recurrent``."""
    comm: str = "none"
    """``"none"``; ``"attn"``: one round of attention over teammates' encoded inputs
    before the trunk, restricted by the scenario's ``comm_mask()`` when it has one; or a
    delayed message (``"latent"``, ``"vae"``, ``"pose"``; see
    :class:`~swarp.rl.networks.PolicyNet`), encoded by each agent from its own observation
    and heard by its teammates on the next step."""
    comm_dim: int = 64
    """Width of the encoded input, the keys, the values and the message (``"attn"``), or
    of the code (``"latent"``, ``"vae"``)."""
    msg_coef: float = 1.0
    """Weight of a delayed message's own loss (``PolicyNet.message_loss``): the ``"pose"``
    likelihood or the ``"vae"`` reconstruction and KL."""
    msg_beta: float = 1e-3
    """``"vae"``: the KL's weight inside the message loss."""
    msg_drop: float = 0.1
    """Delayed messages: the probability that a packet from one teammate to another is lost
    on a step, drawn independently per pair and step in the rollout."""

    def __post_init__(self) -> None:
        for name, allowed in (
            ("actor", ACTORS), ("role_params", ROLE_PARAMS), ("critic_input", CRITIC_INPUTS),
            ("comm", COMMS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.aux_coef < 0:
            raise ValueError(f"aux_coef must be >= 0, got {self.aux_coef}")
        if self.action_rate_coef < 0 or self.action_rate_ramp_iters < 0:
            raise ValueError("action_rate_coef and action_rate_ramp_iters must be >= 0")
        if self.aux_coef > 0 and self.actor != "policynet":
            raise ValueError("aux_coef > 0 needs actor='policynet' (the aux head lives there)")
        if self.privileged_actor and not (self.actor == "policynet" and self.aux_coef > 0):
            raise ValueError("privileged_actor reads the aux targets as inputs: it needs "
                             "actor='policynet' and aux_coef > 0")
        if self.actor_drop and self.actor != "policynet":
            raise ValueError("actor_drop needs actor='policynet'")
        if self.priv_noise:
            if not self.privileged_actor:
                raise ValueError("priv_noise perturbs the privileged input: it needs "
                                 "privileged_actor")
            if len(self.priv_noise) != 5 or any(x < 0 for x in self.priv_noise):
                raise ValueError("priv_noise must be 5 values >= 0 (pos_bias, pos_jitter, "
                                 f"heading_bias, heading_jitter, outlier_prob), got "
                                 f"{self.priv_noise}")
            if self.priv_noise[4] >= 1:
                raise ValueError(f"priv_noise's outlier_prob must lie in [0, 1), got "
                                 f"{self.priv_noise[4]}")
        if self.learned_roles and not (
            self.actor == "policynet" and self.role_params == "heads"
            and self.critic_input in ("state", "state+obs") and not self.normalize_advantage
        ):
            raise ValueError("learned_roles needs actor='policynet', role_params='heads', "
                             "critic_input 'state' or 'state+obs' and normalize_advantage=False")
        if self.role_dwell < 1 or self.role_entropy_coeff < 0:
            raise ValueError("role_dwell must be >= 1 and role_entropy_coeff >= 0")
        if self.teacher and not (self.actor == "policynet" and self.aux_coef > 0):
            raise ValueError("a teacher reads the aux targets: it needs actor='policynet' and "
                             "aux_coef > 0")
        if self.bc_coef < 0 or self.bc_decay_iters < 0:
            raise ValueError("bc_coef and bc_decay_iters must be >= 0")
        if self.recurrent and self.actor != "policynet":
            raise ValueError("recurrent=True needs actor='policynet' (the GRU lives there)")
        if self.comm != "none" and self.actor != "policynet":
            raise ValueError(f"comm={self.comm!r} needs actor='policynet' (the attention "
                             "lives there)")
        if self.comm in ("vae", "pose") and self.aux_coef <= 0:
            raise ValueError(f"comm={self.comm!r} grounds the message in the aux targets: "
                             "it needs aux_coef > 0")
        if not 0.0 <= self.msg_drop < 1.0 or self.msg_coef < 0 or self.msg_beta < 0:
            raise ValueError("msg_drop must lie in [0, 1); msg_coef and msg_beta must be >= 0")
        for name in ("hidden_dim", "comm_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.seq_len < 0:
            raise ValueError(f"seq_len must be >= 0 (0 means steps_per_batch), got {self.seq_len}")
        if self.seq_len and not self.recurrent:
            raise ValueError("seq_len is the BPTT chunk length: it needs recurrent=True")
        if self.seq_len and self.steps_per_batch % self.seq_len:
            raise ValueError(
                f"seq_len {self.seq_len} must divide steps_per_batch {self.steps_per_batch}"
            )
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError(f"target_kl must be > 0, got {self.target_kl}")
        if self.actor_scale_min is not None and self.actor == "mlp":
            raise ValueError("actor_scale_min is a PolicyNet option: it needs actor != 'mlp'")
        if self.adaptive_kl is not None and self.adaptive_kl <= 0:
            raise ValueError(f"adaptive_kl must be > 0, got {self.adaptive_kl}")
        if self.ramp_gate is not None and not self.ramps:
            raise ValueError("ramp_gate gates the ramps' clock: it needs ramps")
        if self.hold_iters < 0:
            raise ValueError(f"hold_iters must be >= 0, got {self.hold_iters}")
        if self.hold_iters and self.ramp_gate is None:
            raise ValueError("hold_iters counts ticks of the gated ramp clock: it needs "
                             "ramp_gate")


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
    # Lidar Push-T, stage T0: Push-T's physics and shaping with five robots (two pushers,
    # three observers) and the oracle T block still in the observation (obs_level 0), so
    # this stage isolates the roles, the observer rewards and the privileged critic from
    # the perception problem. The critic reads the role-sorted critic_state() with one
    # value per role-sorted slot; the actor is one trunk with a head per role. The aux
    # loss is off until the stages that drop the T pose (docs/pusht-lidar.md).
    "pusht_lidar": TrainSpec(
        dt=0.05, substeps=8, max_steps=400, n_agents=5,
        scen_kwargs={
            "pos_shaping_factor": 5.0, "rot_shaping_factor": 0.5, "rot_away_penalty": 1.0,
            "obs_level": 0,
        },
        iters=4000, n_envs=512, steps_per_batch=32, epochs=8, minibatches=4,
        lr=5e-4, entropy_coeff=3e-3, num_cells=256, normalize_advantage=False,
        curriculum=(
            CurriculumKnob("goal_spawn_radius", 0.25, 0.25 + 1.9),
            CurriculumKnob("goal_spawn_angle", 0.4, math.pi),
        ),
        curriculum_iters=250, curriculum_gate=0.0,
        metrics=("tee_dist_to_goal", "tee_angle_error", "team_coverage",
                 "observer_visibility", "occlusion", "tee_seen", "multiobj_reward"),
        actor="policynet", role_params="heads", critic_input="state",
        per_agent_value=True, aux_coef=0.0,
    ),
    # Lidar Push-T on two real AgileX Limos (EAI T-mini Pro lidar, localised by mocap or nav2
    # AMCL), in metres and seconds; notes/pusht-lidar-limo.md records how each value below
    # was arrived at, and docs/pusht-lidar.md what the recipe trains to.
    #
    # Scale and timing. 0.322 x 0.220 m unicycle boxes at up to 0.4 m/s and 1.5 rad/s, the
    # real slim T (two 0.50 x 0.04 m bars), a 5 Hz policy on the latest 10 Hz scan: at
    # dt 0.1 one action moves a robot 4% of the T and PPO kept a near-random policy for
    # thousands of iterations. 32 substeps keep Push-T's validated 6.25 ms contact sub-step;
    # 450 steps (90 s) are Push-T's episode scaled by the task's ~4.5x duration; gamma 0.99
    # and lmbda 0.95 are 20 s / 4 s credit horizons. The length-based shaping is halved
    # (pos 2.5, approach 0.5) so an episode's shaping keeps Push-T's ratio to the goal bonus.
    #
    # Drive and contacts, from a Gazebo sysid (swarp_gz_test out/sysid). ControlMode.SERVO
    # with accelerations and command gains drawn per env and robot around Gazebo's Limo
    # (1 m/s^2, 3 rad/s^2, gains 1.0 / 0.79). The T on a Coulomb floor (mu 0.2-0.5, no
    # viscous drag), which turns an off-centre push as the quasi-static reference does
    # (60 / 142 deg/m at 0.2 / 0.4 m, reference 64 / 145); robots of mass 2, so a pusher
    # keeps 0.95-0.98 of its command in contact. Wheel-faithful contacts: a contact torque
    # turns a robot only past the drive's 3.5 N m, and its wheels grip sideways with mu 0.2,
    # so two robots squeezing the T from opposite faces cannot spin in place as a Limo
    # cannot. A command delay of 0-125 ms per robot stands in for WiFi and processing.
    #
    # Sensor. The lidar sits 0.103 m ahead of the centre, returns from 0.02 m with 5-20 mm
    # of range noise, and sees 220 deg cut at the mount (the lab's drivers deliver
    # -116..+108 deg on both robots), 40 rays at a 5.6 deg spacing, out to 5 m: at 1.5 m a
    # robot without the T in range in a 4-5 m arena had nothing to go on and training
    # stalled. What the real scan and the deployment add (pusht_lidar_artefacts.py): labels
    # from the reported poses (a teammate hit off its reported footprint reads as the T),
    # grazing-incidence losses on the T, 0-3% dropouts, 0-0.5% spurious short returns, a
    # 1% scale and 1 cm bias error, the 10 Hz rotating sweep with its stamp anywhere in it
    # and a 2 deg mount yaw error. lidar_arena_flag adds a fourth mask per frame, the
    # T-labelled hits past the arena's edge, which the deployment builds from the reported
    # pose as it builds the labels.
    #
    # Localisation. pose_noise on the observation only, sized from nav2 AMCL in Gazebo:
    # per-episode bias, jitter and a slip-driven, decaying drift (position decorrelates in
    # 3.6 s, heading in 1.6 s; the slip gain is 2 because AMCL amplifies a shove), at a
    # scale drawn per env from 0 (mocap) to 1.25 (worse than tuned AMCL). Reward, success,
    # the critic and the auxiliary targets read the true state; critic_dr appends every
    # randomised parameter to the critic state.
    #
    # Arena and surroundings. An open floor: each env's arena is W x H with W, H in 2-5 m,
    # observed as its half-extents; the robots meet the safety bridge's stop 0.25 m past
    # it, and a T centroid 0.5 m past it fails the episode (-0.05 per robot and step, no
    # solve). Outside it, per env, nothing, perimeter walls, a rotated room or walls plus
    # 2-6 circle and capsule clutter shapes within 1.5 m, all solid: the robots and the T
    # collide with them, and every robot pays 0.03 per step within 1 cm of the other robot
    # or a surroundings shape. The T's size is drawn per env from three copies scaled 0.9,
    # 1.0 and 1.1.
    #
    # Training. From scratch in one run of 2048 envs and 16 minibatches, the world made
    # hard in ramps on a gated clock (a tick needs a windowed solve of 0.75, so no condition
    # comes in before the last is mastered): the arena widens from 4 x 4 m over ticks
    # 1000-3000; the pose noise, the command delay, the surroundings and the collision cost
    # come in over 3000-5000; the turn-away cost over 8000-10000, since from scratch it made
    # a clumsy push of the slim T net-negative and the policies stopped pushing. The run
    # ends 4000 ticks after the last ramp (24000 iterations is a cap). Guards against the
    # late collapses unguarded runs met: the learning rate decays 5e-4 -> 1e-4 over the
    # clock and follows the KL (adaptive_kl), a KL brake skips the rest of a destructive
    # iteration, the actor's std is floored at 0.1 (the greedy action is the mean, so the
    # floor does not reach the deployment), and rollback restores a pre-collapse snapshot at
    # half the learning rate. An action-rate penalty of 0.03 ramps in over the first 2000
    # iterations. Two pushers on their own lidar, frame_stack 4 and aux loss 0.1, which
    # solved 0.99 at the default scale; standoff_dist 1.0 only clears the scenario's check,
    # since pushers never read it.
    "pusht_lidar_limo": TrainSpec(
        scenario="pusht_lidar",
        dt=0.2, substeps=32, max_steps=450, n_agents=2,
        scen_kwargs={
            # Robots and task.
            "n_pushers": 2, "model": "diff_drive", "agent_shape": "box",
            "agent_half_extents": (0.161, 0.110), "max_speed": 0.4, "max_ang_vel": 1.5,
            "bar_len": 0.5, "bar_width": 0.04, "stem_len": 0.5, "stem_width": 0.04,
            "tee_scale": (0.9, 1.1), "n_tee_sizes": 3,
            "goal_tolerance": 0.15, "angle_tolerance": 0.25, "obs_level": 1,
            "frame_stack": 4, "standoff_dist": 1.0,
            "pos_shaping_factor": 2.5, "rot_shaping_factor": 0.5, "rot_away_penalty": 1.0,
            "agent_dist_shaping": 0.5,
            # Drive and contacts.
            "agent_mass": 2.0, "servo_accel": (0.6, 2.0), "servo_ang_accel": (2.0, 6.0),
            "cmd_gain_v": (0.9, 1.05), "cmd_gain_w": (0.7, 1.0),
            "floor_friction": (0.2, 0.5), "linear_damping": 0.0, "angular_damping": 0.0,
            "wheel_contacts": True, "wheel_lateral_friction": 0.2, "wheel_turn_torque": 3.5,
            "cmd_latency": (0.0, 0.125),
            # Sensor.
            "n_rays": 40, "lidar_span": math.radians(220.0), "lidar_fov": math.radians(220.0),
            "lidar_max_range": 5.0, "lidar_mount_offset": 0.103, "lidar_min_range": 0.02,
            "lidar_noise": (0.005, 0.02),
            "lidar_label_from_pose": True, "lidar_grazing_cos": math.cos(math.radians(80.0)),
            "lidar_dropout": (0.0, 0.03), "lidar_spurious": (0.0, 0.005),
            "lidar_range_scale": 0.01, "lidar_range_bias": 0.01, "lidar_scan_period": 0.1,
            "lidar_stamp_phase": (0.0, 1.0), "lidar_yaw_bias": math.radians(2.0),
            "lidar_arena_flag": True,
            # Localisation.
            "pose_noise": (0.0, 1.25), "pose_bias_std": (0.04, 0.015),
            "pose_jitter_std": (0.01, 0.005), "vel_noise_std": 0.02,
            "pose_drift_tau": (3.6, 1.6), "pose_drift_std": (0.05, 0.015),
            "pose_slip_gain": (2.0, 0.1), "critic_dr": True,
            # Arena and surroundings.
            "world_size": 2.5, "open_arena": True, "arena_size": (2.0, 5.0),
            "arena_stop_margin": 0.25, "oob_margin": 0.5, "oob_penalty": 0.05,
            "scenery_probs": (0.25, 0.25, 0.25, 0.25), "n_clutter": 12,
            "clutter_count": (2, 6), "clutter_reach": 1.5,
            "collision_penalty": 0.03, "collision_margin": 0.01,
        },
        iters=24000, n_envs=2048, steps_per_batch=32, epochs=8, minibatches=16,
        lr=5e-4, lr_final=1e-4, gamma=0.99, lmbda=0.95, entropy_coeff=3e-3, num_cells=256,
        normalize_advantage=False,
        curriculum=(
            CurriculumKnob("goal_spawn_radius", 0.4, 3.8),
            CurriculumKnob("goal_spawn_angle", 0.4, math.pi),
        ),
        curriculum_iters=500, curriculum_gate=0.0,
        ramps=(
            Ramp("arena_size", (4.0, 4.0), (2.0, 5.0), 1000, 3000),
            Ramp("pose_noise", (0.0, 0.0), (0.0, 1.25), 3000, 5000),
            Ramp("cmd_delay_substeps", (0, 0), (0, 20), 3000, 5000),
            Ramp("scenery_on", 0.0, 1.0, 3000, 5000),
            Ramp("rot_away_penalty", 0.0, 1.0, 8000, 10000),
            Ramp("collision_penalty", 0.0, 0.03, 3000, 5000),
        ),
        ramp_gate=0.75, hold_iters=4000, target_kl=0.3, adaptive_kl=0.02, rollback=True,
        actor_scale_min=0.1, action_rate_coef=0.03, action_rate_ramp_iters=2000,
        metrics=("tee_dist_to_goal", "tee_angle_error", "team_coverage",
                 "observer_visibility", "occlusion", "tee_seen", "multiobj_reward",
                 "out_of_bounds", "scenery_leak", "robot_collision", "scenery_collision"),
        stagger_starts=True,
        actor="policynet", role_params="heads", critic_input="state",
        per_agent_value=True, aux_coef=0.1,
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
    # Flow's ring: 1 AV among 21 IDM humans on the Sugiyama 230 m ring. The multi-AV variant
    # is `--n-agents 4` (n_vehicles stays 22, so 18 humans): SPECS keys it by its registry
    # name, and the obs width does not depend on either count. dt=0.1 is Flow's (SUMO's)
    # step. max_steps=1500 (150 s), because a FollowerStopper AV needs 64 s to stabilise
    # this ring at 58% of v_eq. 60 s, the first guess, left no margin for a learner.
    # gamma 0.995: the 10 s credit horizon of 0.99 is shorter than the time a wave takes
    # to travel once around the ring (about 30 s at the wave's backward speed), so the
    # policy could not see the payoff of an early brake. Measured per-step terms at
    # difficulty 1, 64 envs x 600 steps: the IDM-driven arm nets -1.44 (speed +0.56,
    # accel -0.49, time -1.50, no crashes), coasting -5.69, random -6.53.
    #
    # The greedy policy is bistable across checkpoints: on the 22-car ring it flips between
    # solving all 512 eval envs and solving none from one 250-iteration checkpoint to the
    # next, and its failure mode is parking (the AV stops and the whole ring jams, which
    # never crashes and never terminates). Measured greedy solve per 250-iteration
    # checkpoint: lr 1e-3 solved at iters 1500 and 2750 only and parked from 3500 on; lr
    # 3e-4 solved at 1000 (0.90), 2000 and 2500. Strengthening the speed term (speed_coef
    # 0.5, time_penalty -3) did *not* help (0.004 best over 3000 iters). So: lr 3e-4,
    # `--checkpoint-every 250`, and pick the checkpoint by greedy eval
    # (`examples/eval_mappo.py`), not `_best.pt`, which is keyed on the stochastic window
    # and here held a policy whose greedy action solves nothing.
    "ringroad": TrainSpec(
        dt=0.1, substeps=1, max_steps=1500, n_agents=1,
        scen_kwargs={"n_vehicles": 22},
        iters=3000, n_envs=1024, steps_per_batch=32, minibatches=8, lr=3e-4,
        gamma=0.995, lmbda=0.97,
        # The reset wave's amplitude. Even at 0.2 the 22-car ring grows a full wave within
        # a minute, so the low end is not trivially solved; it only gives the policy a
        # wave it can catch before it is fully developed.
        curriculum=(CurriculumKnob("difficulty", 0.2, 1.0, none_at_full=False),),
        curriculum_iters=300, curriculum_gate=0.35,
        metrics=("speed_std", "mean_speed", "crashed", "multiobj_reward"),
        # Nearly every early episode runs the full 1500 steps, so all 1024 envs would
        # truncate on the same step, forever: each batch would be the same 32-step slice of
        # the episode everywhere (measured: reward/step swinging -2 to -6 with episode time).
        stagger_starts=True,
    ),
}


def scenario_name(name: str) -> str:
    """The registered scenario behind ``name``: a ``SPECS`` key's ``scenario`` when it
    sets one, else ``name`` itself (a registry name, or a key that is one)."""
    spec = SPECS.get(name)
    return name if spec is None or spec.scenario is None else spec.scenario


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
        self.update_counts(int(terminated.sum().item()), int(done.sum().item()), done.numel())

    def update_counts(self, n_term: int, n_done: int, n_steps: int) -> None:
        """:meth:`update` from host counts, for a caller that already read them in one sync."""
        self._term.append(n_term)
        self._done.append(n_done)
        # Only count env-steps from the first episode end onward, so the ep_len_mean ratio
        # covers the same period in numerator and denominator. Without this, the whole
        # warm-up's steps sit over a denominator of one or two stragglers and the reported
        # length exceeds max_steps, which is impossible and reads as a bug.
        if n_done:
            self._seen_end = True
        self._steps.append(n_steps if self._seen_end else 0)

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
    if annotation is float or ("float" in text and "tuple" not in text):
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
    params = inspect.signature(scenario_class(scenario_name(name)).__init__).parameters
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


# Builds the policy arm of :func:`score`: ``(n_envs, device, lo, hi) -> ActionFn``. The
# action fn must consume no RNG (a greedy action), which keeps the arms paired.
PolicyArm = Callable[[int, str, torch.Tensor, torch.Tensor], ActionFn]


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


def ringroad_idm_driver(
    k_off: float = 0.3, k_head: float = 1.0
) -> Callable[[Environment], ActionFn]:
    """Scripted ring-road arm: drive the RL car as one more IDM human.

    This is the "humans only" reference. With it, the ring is the uncontrolled IDM ring
    (up to the RL car's ``max_accel`` clamp), so its solve rate is the floor a policy has
    to beat: at the Sugiyama density it is zero at N=22 and one at N <= 8.

    Longitudinal: the IDM acceleration from the observation's own speed, leader speed and
    gap, with the scenario's own IDM parameters. Lateral: the curvature feedforward
    ``atan(L/R)`` plus a P law on the lateral offset and the heading error. Linearised,
    the offset obeys ``o'' + (v k_head / L) o' + (v^2 k_off / L) o = 0``, which is damped
    for any positive gains (zeta ~ 0.6 at the ring's 3.5 m/s equilibrium speed).
    """

    def make(env: Environment) -> ActionFn:
        scen = env.scenario
        lo, hi = env.action_bounds
        half_lane = 0.5 * scen.lane_width

        def action_fn(obs: torch.Tensor) -> torch.Tensor:
            v = obs[..., 0] * scen.v_scale
            dv = -obs[..., 1] * scen.v_scale  # own speed minus the leader's
            gap = (obs[..., 2] * scen.gap_scale).clamp(min=0.1)
            dyn = (v * scen.idm_time_headway + v * dv * scen._inv_2sqrt_ab).clamp(min=0.0)
            ratio = (scen.idm_s0 + dyn) / gap
            acc = scen.idm_accel * (1.0 - (v / scen.idm_v0) ** scen.idm_delta - ratio**2)
            steer = (
                obs[..., 8] * scen.max_steer
                + k_off * obs[..., 5] * half_lane
                - k_head * obs[..., 6]
            )
            return torch.stack([acc, steer], dim=-1).clamp(lo, hi)

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
    # Not a goal controller: the RL car driven as one more IDM human, i.e. the humans-only
    # ring. Its solve rate is the floor, zero wherever the ring is string-unstable.
    "ringroad": ringroad_idm_driver(),
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
    env_kwargs: Mapping[str, Any] = (),
) -> Environment:
    """One place that turns a spec into an ``Environment``.

    Goes through :func:`swarp.make`, which already routes keywords by name and raises a
    legible ``TypeError`` (with a ``difflib`` suggestion) for a misspelling -- there is no
    reason for this file to reimplement any of that. ``auto_reset`` defaults to ``False``,
    which is what ``SwarpEnv`` requires; the lean trainer passes ``True``. ``scenario`` is
    a registry name or a ``SPECS`` key (see :func:`scenario_name`).
    """
    return swarp.make(
        scenario_name(scenario), n_envs=n_envs, device=device, dt=dt, substeps=substeps,
        seed=seed, max_steps=max_steps, n_agents=n_agents, **dict(env_kwargs),
        **dict(scen_kwargs),
    )


def score(
    scenario: str,
    *,
    policy: PolicyArm | None = None,
    policy_from_env: Callable[[Environment], ActionFn] | None = None,
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
    ``truncated`` never fires. ``policy`` builds the policy arm (see :data:`PolicyArm`);
    ``policy_from_env`` builds it from the arm's env instead, for a policy that reads more
    than the observation each call (the roles in ``env.agent_roles()``, a hidden state).
    Pass one of the two.
    Each arm gets a **fresh env on the same seed**, and the policy arm consumes no RNG, so
    the arms are paired: differences are attributable to the controller, not to the
    initial conditions. ``tests/examples`` pins that.
    """
    if policy is not None and policy_from_env is not None:
        raise ValueError("pass policy or policy_from_env, not both")
    out: dict[str, ArmResult] = {}
    for name in arms:
        if name == "policy" and policy is None and policy_from_env is None:
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
            act = policy_from_env(env) if policy is None else policy(n_envs, device, lo, hi)
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
