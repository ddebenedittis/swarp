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

Three lessons from the Push-T run are baked in here rather than left to be rediscovered:

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
from torchrl.modules import MultiAgentMLP, NormalParamExtractor, ProbabilisticActor, TanhNormal
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
    """Spawn the flock near the pen early on, widening to the full ring.

    Measured to be a *weak* lever on its own: a random policy already pens ~7% of sheep at
    full difficulty, so the pen bonus is experienced from the start and there is no
    never-seen-terminal-reward problem for a curriculum to solve. It is wired up because it
    costs nothing and composes with the rest; the episode length and exploration pressure
    are what actually move this task (a scripted policy solves it in ~72 steps, so a
    300-step budget dilutes the penning signal across a long tail of already-solved steps).
    """
    if hasattr(scen, "set_spawn_scale"):
        scen.set_spawn_scale(max(0.05, f))


def _giveway_curriculum(scen, f: float) -> None:
    """Widen the corridor early on, so a novice policy can pass without a perfect yield.

    Give-way's whole difficulty is that two robots do not fit abreast. At ``f=0`` the
    corridor is wide enough that they do, which makes the task ordinary navigation and gets
    the terminal bonus experienced at all; by ``f=1`` it is back to the real one-lane width.
    """
    if hasattr(scen, "set_corridor_scale"):
        scen.set_corridor_scale(2.0 - f)


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
    # 150 rather than 300: a scripted "drive the furthest sheep in, then back off" policy
    # solves this in a median of ~72 steps, and every step after the solve dilutes the
    # penned-fraction signal the policy is scored on.
    "shepherding": Task(
        n_agents=3,
        max_steps=150,
        success="sheep_penned",
        success_desc="fraction of sheep penned",
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


def build_policy(obs_dim: int, act_dim: int, n_agents: int, device: str, num_cells: int = 256):
    """Decentralised actor with shared weights (the MAPPO actor half).

    Shared with ``marl_eval.py`` so a checkpoint loads into an identical module.
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
    parser.add_argument("--lr", type=float, default=1e-3)
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
        "--curriculum-gate", type=float, default=0.35,
        help="min fraction of finished episodes ending in success to raise difficulty",
    )
    parser.add_argument(
        "--start-difficulty", type=float, default=0.0,
        help="initial curriculum difficulty in [0, 1]; use 1.0 when resuming a finished run",
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
    policy = build_policy(obs_dim, act_dim, n_agents, device, num_cells=args.num_cells)
    critic = TensorDictModule(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=1,
            n_agents=n_agents,
            centralised=True,
            share_params=True,
            device=device,
            depth=2,
            num_cells=args.num_cells,
        ),
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
        """Raise difficulty while the policy keeps clearing the gate.

        ``rate`` is the fraction of *finished episodes* that ended in success, not the
        per-step terminal rate — see where it is computed for why that distinction sank an
        earlier give-way run (the gate was never cleared, so the corridor stayed at its
        easiest setting for all 4000 iterations).
        """
        nonlocal difficulty
        if task.curriculum is None or not args.curriculum_iters:
            return 1.0
        if rate > args.curriculum_gate:
            difficulty = min(1.0, difficulty + 1.0 / args.curriculum_iters)
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
         "difficulty", *keys]
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
    nonfinite_skips = 0
    for it, batch in enumerate(collector):
        stats = {k: batch.get(("next", "info", k)).float().mean().item() for k in keys}
        # The wrapper splits the two: `terminated` is the scenario's own success condition,
        # `done` is that OR the max_steps timeout, so GAE bootstraps through a timeout.
        terminated = batch.get(("next", "terminated"))
        batch.set(DONE_KEY, _expand(batch.get(("next", "done")), n_agents))
        batch.set(TERM_KEY, _expand(terminated, n_agents))
        with torch.no_grad():
            loss_module.value_estimator(
                batch,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        flat = batch.reshape(-1)
        buffer.extend(flat)
        for _ in range(args.epochs):
            for _ in range(args.minibatches):
                sub = buffer.sample()
                losses = loss_module(sub)
                total = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
                optim.zero_grad()
                total.backward()
                # clip_grad_norm_ rescales but does not filter: one non-finite gradient goes
                # straight into Adam, whose moments then turn *every* parameter NaN on the
                # next step. That is unrecoverable, and nothing stops the run -- it keeps
                # collecting and reporting nan reward for as long as you let it. Skipping the
                # minibatch costs one update; on a healthy gradient this branch never runs.
                gnorm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                if not torch.isfinite(gnorm):
                    optim.zero_grad(set_to_none=True)
                    nonfinite_skips += 1
                    continue
                optim.step()
        buffer.empty()

        mean_reward = batch.get(("next", "reward")).mean().item()
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
        csv_writer.writerow(
            [it, (it + 1) * frames_per_batch, f"{mean_reward:.6f}", f"{term_rate:.6f}",
             f"{episode_solve:.6f}", f"{solved:.6f}", f"{f:.3f}",
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
        if args.checkpoint_every and (it + 1) % args.checkpoint_every == 0:
            save(f"iter{it + 1:05d}")

    save("final")
    print(f"best smoothed {success or 'terminated'} = {best_score:.4f} -> policy_best.pt")
    csv_file.close()
    collector.shutdown()
    print(f"\ncheckpoints + metrics.csv in {ckpt_dir}/")


if __name__ == "__main__":
    main()
