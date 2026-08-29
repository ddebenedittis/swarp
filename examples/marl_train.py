"""MAPPO on any swarp scenario, via TorchRL: ``--scenario <name>`` off the registry.

Wraps a scenario in the batched TorchRL ``EnvBase`` from :mod:`swarp.interop.torchrl` and
runs an on-policy PPO loop with a shared decentralised actor and a centralised critic.
Everything stays on-device: the collector drives the vectorized swarp step directly, no
per-env Python loop.

This is the scenario-generic sibling of ``examples/pusht_torchrl.py``. That script stays as
the worked, heavily annotated Push-T recipe; this one exists so a new scenario gets a
training run without copying 350 lines of PPO plumbing. Everything task-specific lives in
one :class:`Task` record in :data:`TASKS` below — episode length, substeps, the scenario
constructor defaults, which ``info()`` key is the headline success metric, and an optional
curriculum.

Needs the optional torchrl group::

    uv pip install -e '.[torchrl]'

Run with::

    python examples/marl_train.py --scenario giveway --iters 600
    python examples/marl_train.py --scenario caging --smoke      # 5 iters, plumbing check

Metrics land in ``<checkpoint-dir>/metrics.csv``; every scalar the scenario's ``info()``
exposes is logged as a batch mean, alongside ``reward_per_step`` and ``solved``. Evaluate a
checkpoint against a random baseline on the same seed with::

    python examples/marl_eval.py --scenario giveway runs/giveway/policy_final.pt \
        --curve runs/giveway/metrics.csv

Lessons baked in here rather than left to be rediscovered. The first three came from the
Push-T run; the rest were paid for by the give-way and shepherding runs in ``runs/``:

* **Reward-term scale matters more than the algorithm.** Push-T's rotation term ran ~4x its
  position term and was the easier of the two to influence, so the policy optimized rotation
  and ignored position entirely. If a run plateaus with one metric flat, suspect the shaping
  weights before the hyperparameters.
* **An unreachable terminal bonus teaches nothing.** If the task-completion bonus is never
  experienced by a novice policy, it may as well not exist. That is what ``Task.curriculum``
  is for: start easy, widen only while the batch keeps solving.
* **A single non-finite gradient kills a run silently.** ``clip_grad_norm_`` rescales but does
  not filter, so one bad gradient turns every Adam moment NaN and the run keeps going,
  reporting nan reward, for as long as you let it. The optimizer step is skipped instead.
* **Skipping is not fixing, so the skip rate is a metric.** Give-way discarded **50852 of its
  64000 updates** this way and still printed a plausible reward curve for 2000 iterations,
  which read as a reward-design problem and was really an unbounded policy head. ``skip_frac``
  is a column in ``metrics.csv`` and anything over 5% prints a warning.
* **Bound both ends of the policy head.** ``loc`` unclamped saturates the ``TanhNormal``'s
  squash, where the log-prob of a boundary sample diverges and its gradient vanishes; an
  unclamped ``scale`` puts the mass there and does the same. See :class:`_ClampScale`.
* **Normalize.** Observations mix metres, radians and unit flags in one row, and per-step
  rewards run from ~0.06 to ~7.5 across the registry, under one shared ``lr``. See
  :class:`ObsNorm` and :class:`ReturnScaler`; MAPPO's own ablation calls value normalization
  the most influential of the five factors it isolates.
* **A curriculum that cannot reverse is a trap.** The monotone ratchet stranded give-way at
  difficulty 0.472 for 1700 iterations and shepherding at 0.404, both reporting zero success
  the whole time. :func:`set_curriculum` now lowers difficulty as well as raising it.
* **Decay the step size.** PPO here collapses late and does not recover -- a give-way run held
  a ~1.0 episode solve rate from iteration 400 to 2400, then fell to zero by 2800 and stayed
  there. ``--anneal-lr`` plus the ``--target-kl`` epoch cut-off are the two guards.
* **Do not measure a feasibility question with a training run.** Both reworked scenarios now
  carry a scripted-policy test that must pass before training is worth starting. Shepherding
  was geometrically unsolvable for its whole 3000-iteration run and the test suite was green
  throughout; a 30-second scripted rollout would have said so on day one.
"""

from __future__ import annotations

import argparse
import ast
import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from swarp import Environment
from swarp.interop.torchrl import SwarpEnv
from swarp.scenarios import make_scenario

# SwarpEnv is flat (no ("agents", ...) group) and emits one shared done per env,
# [n_envs, 1], while reward is per-agent [n_envs, n_agents, 1]. GAE needs the two
# broadcastable, so we expand done/terminated into these dedicated keys each batch
# and point the value estimator at them.
DONE_NAME, TERM_NAME = "agents_done", "agents_terminated"
DONE_KEY, TERM_KEY = ("next", DONE_NAME), ("next", TERM_NAME)


class ObsNorm(torch.nn.Module):
    """Running mean/std over observations, as the first layer of the actor and the critic.

    Two things here make normalization structural rather than a nicety. Observations are
    raw physical quantities and a scenario is free to mix scales in one row -- caging puts
    positions in metres next to bearings in radians, six times larger. And per-step rewards
    across the registry span two orders of magnitude. One shared ``lr`` cannot serve both.

    It lives *inside* the network rather than as an env transform for three reasons: the
    statistics land in ``state_dict`` so a checkpoint carries them and ``marl_eval.py``
    cannot accidentally evaluate with a different transform than training used; collection
    and the epoch loop see the same layer, so PPO's ratio is between two policies that
    differ only in weights; and it stays out of the scenario, whose fused hot path returns a
    persistent buffer a captured graph writes into -- a scenario normalizing in place would
    have to declare the statistics as a carry and snapshot them around graph warm-up.

    :meth:`update` is called once per iteration, *after* that iteration's epochs, from the
    collected batch only. Updating from a minibatch would be a feedback loop (the minibatch
    is drawn from a buffer these statistics already shaped), and updating mid-iteration
    would change the observation under the ratio the policy loss is still using.
    """

    def __init__(self, dim: int, device: str, enabled: bool = True, eps: float = 1e-4):
        super().__init__()
        self.enabled = enabled
        self.register_buffer("mean", torch.zeros(dim, device=device))
        self.register_buffer("var", torch.ones(dim, device=device))
        self.register_buffer("count", torch.tensor(eps, device=device))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if not self.enabled:
            return
        x = x.reshape(-1, self.mean.shape[0]).to(self.mean.dtype)
        n = x.shape[0]
        if n == 0:
            return
        m, v = x.mean(0), x.var(0, unbiased=False)
        delta = m - self.mean
        tot = self.count + n
        # Chan et al.'s parallel variance. The naive pooled form drops the cross term and
        # underestimates the variance whenever consecutive batches differ in mean, which is
        # every batch of an on-policy run.
        cross = delta.pow(2) * (self.count * n / tot)
        self.var.copy_((self.var * self.count + v * n + cross) / tot)
        self.mean.copy_(self.mean + delta * (n / tot))
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return (x - self.mean) / (self.var.sqrt() + 1e-8)


class ReturnScaler:
    """Divide rewards by the running std of the discounted return. Value normalization.

    MAPPO's own ablation calls value normalization the most influential of the five factors
    it isolates, and the reason is visible in this registry: per-step rewards run from
    ~0.06 (shepherding) to ~7.5 (caging), so the critic regresses onto targets two orders of
    magnitude apart under one shared ``lr``. Both reworked scenarios now also carry a large
    terminal bonus against a small dense stream, which widens the gap further.

    Scaling the reward by the return's std, rather than standardizing the value *target*,
    is the cheaper of the two equivalent forms and the only one that does not require
    reaching inside TorchRL's value estimator. It is scale-only on purpose: subtracting a
    mean from a reward is not a policy-preserving transform under discounting, since the
    shift accumulates differently over episodes of different length -- and give-way's whole
    difficulty is that its episodes end at very different times.
    """

    def __init__(self, gamma: float, device: str, enabled: bool = True):
        self.gamma, self.enabled = gamma, enabled
        self.ret = None  # running discounted return, carried across batches per env/agent
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = 1e-4

    @torch.no_grad()
    def __call__(self, reward: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return reward
        if self.ret is None or self.ret.shape != reward[:, 0].shape:
            self.ret = torch.zeros_like(reward[:, 0])
        # Walk the collector's time axis, resetting the accumulator where an episode ended.
        # The return is built from RAW rewards -- scaling its own input would compound.
        for t in range(reward.shape[1]):
            self.ret = self.ret * self.gamma + reward[:, t]
            flat = self.ret.reshape(-1)
            n = flat.numel()
            m, v = flat.mean(), flat.var(unbiased=False)
            delta = m - self.mean
            tot = self.count + n
            # Chan et al.'s parallel variance again. The mean has to be tracked even though
            # it is never subtracted from a reward: the cross term needs it, and without it
            # this silently measures the second moment about zero instead of the variance.
            cross = delta.pow(2) * (self.count * n / tot)
            self.var = (self.var * self.count + v * n + cross) / tot
            self.mean = self.mean + delta * (n / tot)
            self.count = tot
            # done is one shared flag per env, [B, T, 1]; the return is per agent,
            # [B, n_agents, 1]. Insert the agent axis so it broadcasts rather than
            # expanding a mismatched shape.
            ended = done[:, t].unsqueeze(-2)
            self.ret = torch.where(ended, torch.zeros_like(self.ret), self.ret)
        return reward / (self.var.sqrt() + 1e-8)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}


@dataclass(frozen=True)
class Task:
    """Everything that differs between scenarios, in one record.

    ``success`` names the ``info()`` key whose batch mean is the headline "is it working"
    number printed each iteration and used to gate the curriculum. It is deliberately a
    *scenario* quantity rather than the reward: reward scale is arbitrary and gets retuned,
    while "fraction of sheep penned" means the same thing across every run.

    ``curriculum`` receives ``(scenario, difficulty)`` with ``difficulty`` in ``[0, 1]`` and
    mutates the scenario in place. ``None`` means the task is learnable from a uniform reset
    and needs no ramp.

    A curriculum callback calls its scenario's setter **unguarded**, on purpose. There is no
    hook for this on the ``Scenario`` ABC -- the trainer duck-types it -- so these used to be
    wrapped in ``hasattr``, and that turns a renamed or deleted setter into a silent no-op:
    the run trains at fixed difficulty, reports a plausible curve, and says nothing. That has
    already happened once here. An ``AttributeError`` on the first batch is the better
    failure.
    """

    n_agents: int
    max_steps: int
    substeps: int = 1
    dt: float = 0.05
    scenario_kwargs: dict = field(default_factory=dict)
    success: str | None = None
    success_desc: str = ""
    curriculum: Callable[[object, float], None] | None = None


def _shepherding_curriculum(scen, f: float) -> None:
    """Drive first, collect second, on two independent axes.

    Measured: flock *distance* is the dominant axis and dispersion is nearly irrelevant --
    a scripted Strombom controller solves 0.97 at ``f=0`` and 0.85 at ``f=1``, while a
    random policy solves 0.03 and 0.01. Both knobs re-derive from the constructor
    parameters rather than undoing the previous scale, so ``f=1.0`` restores the
    constructor's draw bit-for-bit.

    This replaces ``set_spawn_scale``, whose single axis conflated "how far must the flock
    be driven" with "how scattered is it", and whose ring geometry was degenerate -- see
    :class:`~swarp.scenarios.shepherding.ShepherdingScenario`.
    """
    scen.set_flock_distance(max(0.05, f))
    scen.set_flock_spread(max(0.05, f))


def _giveway_curriculum(scen, f: float) -> None:
    """Widen the corridor early on, so a novice policy can pass without a perfect yield.

    Give-way's whole difficulty is that two robots do not fit abreast. The span is
    ``1.0 + 0.75*(1 - f)``, and the 0.75 is **measured, not derived**. Two robots fit
    abreast from ``scale = 2r/c = 1.333`` up, but that is a *geometric* threshold: at scale
    1.5 the whole slack is ``2c - 4r = 0.025``, a quarter of a robot diameter, so passing
    abreast needs near-perfect straight-line driving. A greedy controller solves 96% of
    episodes there; the same controller with modest Gaussian action noise solves 25-46%.
    The *behavioural* threshold is scale ~1.75, where the noisy controller is back to
    93-97%. So ``f=0`` lands on the learnability threshold, ``f~0.56`` crosses two-abreast,
    and ``f=1`` is the real one-lane width.

    The scenario also has a ``set_spawn_desync`` knob that staggers arrival at the junction.
    It is deliberately **not** wired in here: measured, any desync at the wide end makes the
    task *harder*, because a moving robot jams a parked one into a dead-end arm -- noisy
    greedy solve drops from 0.96 to 0.28.
    """
    scen.set_corridor_scale(1.0 + 0.75 * (1.0 - f))


TASKS: dict[str, Task] = {
    # --- the three new scenarios ------------------------------------------------
    # Give-way runs a stiff contact (the corridor walls have to be walls), and that
    # needs substeps >= 8 at dt=0.05 or the explicit spring-damper diverges.
    "giveway": Task(
        n_agents=4,
        max_steps=300,
        substeps=16,
        success="all_on_goal",
        success_desc="episodes with every robot on its goal",
        curriculum=_giveway_curriculum,
    ),
    "caging": Task(
        n_agents=5,
        max_steps=250,
        success="caged",
        success_desc="steps with the cage closed",
    ),
    # 200, not 150: the same scripted "drive in, then back off" policy solves 0.61 of
    # episodes at 150 steps, 0.84 at 200 and 0.91 at 250. The drive phase is the slow part
    # -- only the rear sheep feels a shepherd, and cohesion drags the rest.
    #
    # `all_penned`, not `sheep_penned`. The obvious metric is actively misleading here: it
    # is a batch mean over every step under auto-reset, so a policy that solves quickly
    # immediately respawns into a fresh unpenned episode and scores WORSE. Measured, the
    # scripted expert scores 0.174 on `sheep_penned` against a trained policy's 0.285, and
    # over one 3000-iteration run `all_penned` rose 0.031 -> 0.067 while `sheep_penned`
    # fell 0.366 -> 0.285. Since `solved` also drives `policy_best.pt` selection, the
    # anti-correlated metric was choosing the checkpoint. `all_penned` additionally cannot
    # be gamed by parking a subset of the flock, which was the local optimum that sank the
    # earlier run.
    "shepherding": Task(
        n_agents=3,
        max_steps=200,
        success="all_penned",
        success_desc="steps with the whole flock penned",
        curriculum=_shepherding_curriculum,
    ),
    # --- existing scenarios, so the trainer itself is testable ------------------
    "navigation": Task(
        n_agents=4,
        max_steps=200,
        success="on_goal",
        success_desc="fraction of robots on goal",
    ),
    "transport": Task(
        n_agents=4,
        max_steps=300,
        success=None,
        success_desc="",
    ),
}


class _ClampScale(torch.nn.Module):
    """Split ``loc``/``scale`` and bound both. Sits where ``NormalParamExtractor`` did.

    ``NormalParamExtractor`` floors the scale at ``scale_lb`` but leaves it unbounded
    above, and nothing at all bounds ``loc``. Both ends bite with a ``TanhNormal``: a large
    ``loc`` saturates the tanh, where the log-prob of a sample at the boundary diverges and
    the gradient through the squash vanishes; a large ``scale`` puts most of the mass at the
    boundary and does the same. That is the recipe for the non-finite gradients the
    give-way run was discarding 79% of its updates to.

    ``tanh_loc=True`` on the distribution would squash ``loc`` too, but it also reshapes the
    distribution's mode in a way ``marl_eval`` then has to mirror exactly. Clamping here is
    the same guarantee with a mode that stays plainly ``tanh(loc)``.
    """

    def __init__(self, loc_clip: float = 5.0, scale_min: float = 1e-3, scale_max: float = 2.0):
        super().__init__()
        self.loc_clip, self.scale_min, self.scale_max = loc_clip, scale_min, scale_max

    def forward(self, x: torch.Tensor):
        loc, raw = x.chunk(2, dim=-1)
        scale = torch.nn.functional.softplus(raw + 0.5413) + self.scale_min
        return loc.clamp(-self.loc_clip, self.loc_clip), scale.clamp(max=self.scale_max)


def build_policy(
    obs_dim: int,
    act_dim: int,
    n_agents: int,
    device: str,
    num_cells: int = 256,
    action_low: float = -1.0,
    action_high: float = 1.0,
    obs_norm: torch.nn.Module | None = None,
):
    """Decentralised actor with shared weights (the MAPPO actor half).

    Shared with ``marl_eval.py`` so a checkpoint loads into an identical module.

    ``action_low``/``action_high`` come from ``Environment.action_bounds`` rather than being
    hardcoded. They happen to be exactly +/-1 for every scenario in :data:`TASKS`, because
    all of them are holonomic velocity-mode agents at ``max_speed=1.0`` -- but that is a
    coincidence of the current registry, and ``--set max_speed=3.0`` would silently drive
    the fleet at a third of its authority with the old hardcoded bounds.
    """
    net = torch.nn.Sequential(
        obs_norm if obs_norm is not None else torch.nn.Identity(),
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
        _ClampScale(),
    )
    return ProbabilisticActor(
        module=TensorDictModule(net, in_keys=["observation"], out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": action_low, "high": action_high},
        return_log_prob=True,
    )


def build_env(name: str, task: Task, *, n_agents: int, n_envs: int, device: str, seed: int = 0):
    """The swarp ``Environment`` for ``name``, with the task's engine settings applied."""
    scen = make_scenario(name, n_agents=n_agents, **task.scenario_kwargs)
    return Environment(
        scen,
        n_envs=n_envs,
        device=device,
        dt=task.dt,
        substeps=task.substeps,
        seed=seed,
        max_steps=task.max_steps,
    )


def info_keys(sim: Environment) -> list[str]:
    """Scalar ``info()`` keys, discovered rather than hardcoded per scenario.

    A scenario owns its own diagnostics, so the trainer logs whatever it exposes instead of
    carrying a per-scenario key list that silently goes stale when a scenario is edited.
    """
    return sorted(k for k, v in sim.scenario.info().items() if torch.is_tensor(v))


def _expand(t: torch.Tensor, n_agents: int) -> torch.Tensor:
    """``[*B, 1]`` shared flag -> ``[*B, n_agents, 1]``, matching the reward shape."""
    return t.unsqueeze(-2).expand(*t.shape[:-1], n_agents, 1)


def main() -> None:
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=sorted(TASKS))
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--n-agents", type=int, help="default: the task's own team size")
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--n-envs", type=int, default=512)
    parser.add_argument("--steps-per-batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--entropy-coeff", type=float, default=3e-3)
    parser.add_argument("--num-cells", type=int, default=256, help="MLP width")
    parser.add_argument("--max-steps", type=int, help="override the task's episode length")
    parser.add_argument("--substeps", type=int, help="override the task's physics substeps")
    parser.add_argument("--success-key", help="override which info() key is the headline metric")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override a scenario constructor kwarg, e.g. --set collision_penalty=-0.1; "
             "repeatable. Values are parsed as Python literals.",
    )
    parser.add_argument(
        "--curriculum-iters", type=int, default=250,
        help="iters of gated progress to reach full difficulty; 0 disables",
    )
    parser.add_argument(
        "--curriculum-gate", type=float, default=0.5,
        help="fraction of finished episodes ending in success above which difficulty rises",
    )
    parser.add_argument(
        "--curriculum-gate-lo", type=float, default=0.15,
        help="fraction below which difficulty FALLS again; between the two it holds",
    )
    parser.add_argument(
        "--anneal-lr", action="store_true", default=True,
        help="linearly decay lr to --lr-final-frac over the run (default on)",
    )
    parser.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    parser.add_argument("--lr-final-frac", type=float, default=0.1)
    parser.add_argument(
        "--target-kl", type=float, default=0.02,
        help="stop the epoch loop once approximate KL exceeds this; 0 disables",
    )
    parser.add_argument(
        "--norm-obs", action="store_true", default=True,
        help="running mean/std on observations, saved with the checkpoint (default on)",
    )
    parser.add_argument("--no-norm-obs", dest="norm_obs", action="store_false")
    parser.add_argument(
        "--norm-value", action="store_true", default=True,
        help="running mean/std on value targets (default on)",
    )
    parser.add_argument("--no-norm-value", dest="norm_value", action="store_false")
    parser.add_argument(
        "--start-difficulty", type=float, default=0.0,
        help="initial curriculum difficulty in [0, 1]; use 1.0 when resuming a finished run. "
             "Has no effect with --curriculum-iters 0, which skips the curriculum callback "
             "entirely and so leaves the scenario at its constructor geometry",
    )
    parser.add_argument("--checkpoint-dir", help="default: runs/<scenario>")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="0 disables")
    parser.add_argument("--resume", help="policy_*.pt to continue from (actor + critic)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke", action="store_true",
        help="5 iters on 64 envs: checks the plumbing, teaches nothing",
    )
    args = parser.parse_args()

    task = TASKS[args.scenario]
    # Reward-term scale is the first thing to suspect when a scenario will not learn, so
    # it has to be sweepable without editing this file.
    overrides = dict(task.scenario_kwargs)
    for item in args.set:
        key, _, raw = item.partition("=")
        if not _:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = ast.literal_eval(raw)
    if args.max_steps or args.substeps or overrides != task.scenario_kwargs:
        task = Task(
            n_agents=task.n_agents,
            max_steps=args.max_steps or task.max_steps,
            substeps=args.substeps or task.substeps,
            dt=task.dt,
            scenario_kwargs=overrides,
            success=task.success,
            success_desc=task.success_desc,
            curriculum=task.curriculum,
        )
    n_agents = args.n_agents or task.n_agents
    iters, n_envs = (5, 64) if args.smoke else (args.iters, args.n_envs)
    device = args.device

    sim = build_env(
        args.scenario, task, n_agents=n_agents, n_envs=n_envs, device=device, seed=args.seed
    )
    # swarp actions are physical, and SwarpEnv specs sim.action_bounds by default -- the
    # limits the kernels actually clamp against. Nothing to pass.
    env = SwarpEnv(sim)
    obs_dim, act_dim = env.obs_dim, env.act_dim
    sim.reset(seed=args.seed)
    keys = info_keys(sim)
    success = args.success_key or task.success
    if success is not None and success not in keys:
        parser.error(
            f"--success-key {success!r} is not an info() key of {args.scenario}; "
            f"available: {keys}"
        )

    # Decentralised actor (shared weights), centralised critic -- the usual MAPPO split.
    # The action bounds come off the env, not a literal: swarp actions are physical and the
    # kernels clamp them to the per-agent limits in AgentConfig. They are +/-1 for every
    # scenario in TASKS today only because all of them are holonomic velocity agents at
    # max_speed=1.0, and a single --set max_speed=... would break that silently.
    a_low, a_high = sim.action_bounds
    act_low, act_high = float(a_low.min()), float(a_high.max())
    # One ObsNorm instance shared by both heads. Sharing rather than fitting two copies is
    # not just economy: the centralised critic flattens all agents into one row, so a
    # separate critic-side normalizer would be fitting per-agent-slot statistics that the
    # actor does not see, and the two would disagree about what an observation means.
    obs_norm = ObsNorm(obs_dim, device, enabled=args.norm_obs)
    policy = build_policy(
        obs_dim, act_dim, n_agents, device, num_cells=args.num_cells,
        action_low=act_low, action_high=act_high, obs_norm=obs_norm,
    )
    critic = TensorDictModule(
        torch.nn.Sequential(obs_norm, MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=1,
            n_agents=n_agents,
            centralised=True,
            share_params=True,
            device=device,
            depth=2,
            num_cells=args.num_cells,
        )),
        in_keys=["observation"],
        out_keys=["state_value"],
    )

    frames_per_batch = n_envs * args.steps_per_batch
    collector = Collector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * iters,
        device=device,
        auto_register_policy_transforms=True,
    )
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=frames_per_batch // args.minibatches,
    )
    # Advantage normalization is not optional across a registry this varied: per-step
    # rewards run from ~0.06 (shepherding) to ~7.5 (caging), a two-order-of-magnitude
    # spread, and the unnormalized caging advantages drove ~85% of minibatches to a
    # non-finite gradient — the run skipped almost every update and learned nothing.
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        entropy_coeff=args.entropy_coeff,
        normalize_advantage=True,
        # TorchRL warns here every iteration and suggests `normalize_advantage_exclude_dims`
        # for multi-agent settings. Ignore it deliberately: excluding the agent dimension
        # computes *per-agent-slot* statistics, and these agents are interchangeable -- one
        # shared actor, and give-way re-rolls which arm each index starts in every episode,
        # so an agent index carries no persistent meaning to fit separate statistics to.
        # Pooling across agents is the correct estimator for a parameter-shared team.
        # PPO's value clip. The critic sees a return that now contains a large terminal
        # spike against a small dense stream, and an unclipped regression onto that moves
        # the value head far enough in one epoch to invalidate the ratios the policy loss
        # is still using.
        clip_value=0.2,
    )
    # Leaf names: the value estimator looks them up under ("next", ...) itself.
    loss_module.set_keys(
        reward="reward", done=DONE_NAME, terminated=TERM_NAME, value="state_value"
    )
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=args.gamma, lmbda=args.lmbda)
    optim = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt["policy"])
        critic.load_state_dict(ckpt["critic"])
        print(f"resumed actor + critic from {args.resume}")

    scen = sim.scenario
    difficulty = min(1.0, max(0.0, args.start_difficulty))

    def set_curriculum(rate: float) -> float:
        """Two-sided: raise above the high gate, LOWER below the low one, hold between.

        ``rate`` is the fraction of *finished episodes* that ended in success, not the
        per-step terminal rate — see where it is computed for why that distinction sank an
        earlier give-way run (the gate was never cleared, so the corridor stayed at its
        easiest setting for all 4000 iterations).

        The monotone ratchet this replaces has a second failure, and it is the one that
        sank both runs in ``runs/``. Difficulty could only rise, so a policy that regressed
        after a lucky early batch was stranded at a setting it could no longer solve with
        no way back: give-way froze at 0.472 by iteration 299 and sat there for the next
        1700, shepherding at 0.404. Both then reported a success rate of zero for the rest
        of the run, which reads as "the reward is wrong" and was really "the schedule
        cannot reverse". Lowering runs at half the raising rate, so noise around the gate
        drifts upward rather than oscillating.
        """
        nonlocal difficulty
        if task.curriculum is None or not args.curriculum_iters:
            return 1.0
        step = 1.0 / args.curriculum_iters
        if rate > args.curriculum_gate:
            difficulty = min(1.0, difficulty + step)
        elif rate < args.curriculum_gate_lo:
            difficulty = max(0.0, difficulty - 0.5 * step)
        task.curriculum(scen, difficulty)
        return difficulty

    set_curriculum(0.0)
    ckpt_dir = Path(args.checkpoint_dir or f"runs/{args.scenario}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ckpt_dir / "metrics.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["iter", "frames", "reward_per_step", "terminated", "episode_solve", "solved",
         "difficulty", "skip_frac", *keys]
    )

    def save(tag: str) -> None:
        torch.save(
            {
                "policy": policy.state_dict(),
                "critic": critic.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "n_agents": n_agents,
                "num_cells": args.num_cells,
                "scenario": args.scenario,
                # Everything marl_eval needs to rebuild a byte-identical module. The
                # observation statistics ride inside `policy` (ObsNorm registers them as
                # buffers), but the eval side still has to know to *build* an ObsNorm, and
                # the action bounds decide what the distribution's mode actually is.
                "norm_obs": args.norm_obs,
                "action_low": act_low,
                "action_high": act_high,
            },
            ckpt_dir / f"policy_{tag}.pt",
        )

    print(
        f"{args.scenario}: {n_agents} agents x {n_envs} envs, {task.max_steps}-step episodes, "
        f"{task.substeps} substeps, obs {obs_dim} act {act_dim}\n"
        f"  logging {keys}"
        + (f"\n  success = mean({success})  [{task.success_desc}]" if success else "")
    )

    # PPO can collapse late and never recover: a give-way run climbed to a ~1.0 episode
    # solve rate by iteration 1200, held it to 2400, then fell to zero by 2800 and stayed
    # there. Checkpointing only on a fixed interval captured the wreckage, and the saved
    # policy evaluated *worse than random* despite the run having genuinely solved the task
    # for a third of its length. So track the best policy seen, smoothed so one lucky batch
    # cannot claim it, and always leave `policy_best.pt` behind next to `policy_final.pt`.
    best_score, smoothed = float("-inf"), None
    best_cur_score, best_cur_diff = float("-inf"), -1.0
    nonfinite_skips, kl_stops = 0, 0
    scaler = ReturnScaler(args.gamma, device, enabled=args.norm_value)
    for it, batch in enumerate(collector):
        # Linear LR decay. PPO here collapses late and does not recover -- a give-way run
        # held a ~1.0 episode solve rate from iteration 400 to 2400, then fell to zero by
        # 2800 and stayed there. A step size that is still 1e-3 after the policy has become
        # deterministic is how a converged run walks off its own optimum.
        if args.anneal_lr and iters > 1:
            frac = it / (iters - 1)
            lr_now = args.lr * (1.0 - frac * (1.0 - args.lr_final_frac))
            for g in optim.param_groups:
                g["lr"] = lr_now
        stats = {k: batch.get(("next", "info", k)).float().mean().item() for k in keys}
        # The wrapper splits the two: `terminated` is the scenario's own success condition,
        # `done` is that OR the max_steps timeout, so GAE bootstraps through a timeout.
        terminated = batch.get(("next", "terminated"))
        done_flat = batch.get(("next", "done"))
        batch.set(DONE_KEY, _expand(done_flat, n_agents))
        batch.set(TERM_KEY, _expand(terminated, n_agents))
        raw_reward = batch.get(("next", "reward"))
        mean_reward = raw_reward.mean().item()
        # Scale before GAE, not after: the value target is what the critic regresses onto.
        batch.set(("next", "reward"), scaler(raw_reward, done_flat))
        with torch.no_grad():
            loss_module.value_estimator(
                batch,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        flat = batch.reshape(-1)
        buffer.extend(flat)
        it_steps, it_skips, kl_last = 0, 0, 0.0
        for _ in range(args.epochs):
            for _ in range(args.minibatches):
                sub = buffer.sample()
                losses = loss_module(sub)
                kl_last = losses["kl_approx"].item()
                total = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
                optim.zero_grad()
                total.backward()
                # clip_grad_norm_ rescales but does not filter: one non-finite gradient goes
                # straight into Adam, whose moments then turn *every* parameter NaN on the
                # next step. That is unrecoverable, and nothing stops the run -- it keeps
                # collecting and reporting nan reward for as long as you let it. Skipping the
                # minibatch costs one update; on a healthy gradient this branch never runs.
                #
                # It is also not a fix, and treating it as one cost a whole run: give-way
                # discarded 50852 of 64000 updates this way and still reported a plausible
                # reward curve, so the run read as a reward-design problem for 2000
                # iterations. The skip ratio is a first-class metric below for that reason.
                gnorm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                it_steps += 1
                if not torch.isfinite(gnorm):
                    optim.zero_grad(set_to_none=True)
                    nonfinite_skips += 1
                    it_skips += 1
                    continue
                optim.step()
            # Clipping bounds the per-sample ratio but not how far eight epochs over the
            # same 16384 samples can walk in aggregate, and that aggregate drift is what a
            # late collapse looks like from the inside. `kl_approx` is the loss's own
            # estimate on the last minibatch of the epoch, so the gate costs no extra
            # forward pass.
            if args.target_kl and kl_last > args.target_kl:
                kl_stops += 1
                break
        buffer.empty()

        # `mean_reward` was captured from the RAW reward before scaling: the scale is an
        # optimizer detail, and a curve that moves because the scaler warmed up is not a
        # curve about the policy.
        #
        # Fit the observation statistics only now, after this iteration's epochs. Updating
        # before them would change the observation under the very ratio the policy loss is
        # comparing against, and updating from a minibatch would feed the buffer's own
        # statistics back into themselves.
        obs_norm.update(batch.get("observation"))
        term_rate = terminated.float().mean().item()
        # The fraction of episodes that ENDED IN SUCCESS, which is the number a human means
        # by "is it solving it". `terminated` is a per-step flag, so its mean is a per-step
        # hazard rate: a task that always solves on step 40 of a 300-step budget reports
        # 0.025, which reads like a 2.5% success rate and is really 100%. Dividing the
        # terminal flags by the episodes that actually finished removes the episode-length
        # scaling, and unlike Push-T's `rate * max_steps` it stays correct when episodes end
        # early. This is what gates the curriculum; a state-fraction metric like caging's
        # `caged` would be meaningless there.
        ended = batch.get(("next", "done")).sum().item()
        episode_solve = terminated.sum().item() / ended if ended else 0.0
        solved = stats.get(success, term_rate) if success else term_rate
        f = set_curriculum(episode_solve)  # difficulty for the next batch's resets
        skip_frac = it_skips / it_steps if it_steps else 0.0
        if skip_frac > 0.05:
            print(
                f"  WARNING iter {it}: {it_skips}/{it_steps} updates ({skip_frac:.0%}) "
                f"discarded as non-finite. The run is not training on most of its data; "
                f"suspect the policy head or the reward scale, not the reward design."
            )
        csv_writer.writerow(
            [it, (it + 1) * frames_per_batch, f"{mean_reward:.6f}", f"{term_rate:.6f}",
             f"{episode_solve:.6f}", f"{solved:.6f}", f"{f:.3f}", f"{skip_frac:.6f}",
             *(f"{stats[k]:.6f}" for k in keys)]
        )
        csv_file.flush()
        print(
            f"iter {it:4d}  reward/step {mean_reward:+.4f}  ep_solve {episode_solve:.3f}  "
            + "  ".join(f"{k} {stats[k]:.4f}" for k in keys)
            + (f"  diff {f:.2f}" if task.curriculum else "")
            + (f"  skipped {nonfinite_skips}" if nonfinite_skips else "")
        )
        # EMA over ~40 batches: responsive enough to catch a real improvement, slow enough
        # that a single favourable batch does not overwrite a genuinely better policy.
        smoothed = solved if smoothed is None else 0.95 * smoothed + 0.05 * solved
        # Only ever crown a policy that is being scored on the FULL task. Mid-curriculum
        # scores are not comparable with final ones -- give-way's success rate peaks while
        # the corridor is still widened, so an ungated best-tracker reliably saved an
        # easy-stage policy and then evaluated it on the real one-lane corridor, where it
        # solved nothing.
        if f >= 1.0 and smoothed > best_score:
            best_score = smoothed
            save("best")
        # ...and a second tracker with no difficulty gate, reset whenever the curriculum
        # moves. `policy_best.pt` only ever arms at full difficulty, which is right -- a
        # mid-curriculum score is not comparable with a final one -- but both runs in
        # `runs/` stalled below 1.0 and so left `best = -inf` and nothing loadable but a
        # `policy_final.pt` that happened to be whatever the last batch produced. This one
        # always leaves the best policy seen at the difficulty it was actually scored on.
        if f != best_cur_diff:
            best_cur_diff, best_cur_score = f, float("-inf")
        if smoothed > best_cur_score:
            best_cur_score = smoothed
            save("bestcur")
        if args.checkpoint_every and (it + 1) % args.checkpoint_every == 0:
            save(f"iter{it + 1:05d}")

    save("final")
    total_updates = iters * args.epochs * args.minibatches
    print(f"best smoothed {success or 'terminated'} = {best_score:.4f} -> policy_best.pt")
    print(
        f"best at final difficulty {best_cur_diff:.2f} = {best_cur_score:.4f}"
        f" -> policy_bestcur.pt"
    )
    if kl_stops:
        print(f"epoch loop cut short by the KL gate in {kl_stops}/{iters} iterations")
    if nonfinite_skips:
        print(
            f"non-finite updates discarded: {nonfinite_skips} "
            f"({nonfinite_skips / max(1, total_updates):.1%} of ~{total_updates})"
        )
    csv_file.close()
    collector.shutdown()
    print(f"\ncheckpoints + metrics.csv in {ckpt_dir}/")


if __name__ == "__main__":
    main()
