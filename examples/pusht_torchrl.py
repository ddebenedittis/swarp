"""MAPPO on Push-T: four robots learn to push a T to a target pose, via TorchRL.

Wraps :class:`~wmas.scenarios.pusht.PushTScenario` in the batched TorchRL ``EnvBase``
from :mod:`wmas.interop.torchrl` and runs a small on-policy PPO loop with a shared
actor and a centralised critic. Everything stays on-device: the collector drives the
vectorized wmas step directly, no per-env Python loop.

Needs the optional torchrl group::

    uv pip install -e . --group torchrl

Run with::

    python examples/pusht_torchrl.py [--device cuda:0] [--iters 600] [--n-envs 512]

The defaults take ~2 min on a modern GPU (~10M frames). Over that budget the policy
beats a random baseline on both halves of the pose — roughly ``tee-dist`` -0.11 and
``tee-angle`` -0.36 over a 200-step episode, where random *worsens* both. The
``solved`` column (inside **both** tolerances at once) stays near zero: that is a
genuinely tight target and needs far longer than this demo budget.

Note the shaping weights. The scenario's own defaults (1.0 position / 0.5 rotation)
are mis-scaled for learning: per step ``|dt angle|`` runs ~8.7x ``|dt distance|``, so
the rotation term ends up ~4x the position term *and* rotation is the easier of the
two to influence. A policy trained on the raw weights optimizes orientation and
leaves position no better than random. ``--pos-shaping 5.0`` rebalances them, and is
the default here.

Verify the trained policy against random, and plot the curves, with::

    python examples/pusht_eval.py runs/pusht/pusht_final.pt --curve runs/pusht/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.modules import MultiAgentMLP, NormalParamExtractor, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from wmas import Environment, PushTScenario
from wmas.interop.torchrl import WmasEnv

N_AGENTS = 4

# WmasEnv is flat (no ("agents", ...) group) and emits one shared done per env,
# [n_envs, 1], while reward is per-agent [n_envs, n_agents, 1]. GAE needs the two
# broadcastable, so we expand done/terminated into these dedicated keys each batch
# and point the value estimator at them.
DONE_NAME, TERM_NAME = "agents_done", "agents_terminated"
DONE_KEY, TERM_KEY = ("next", DONE_NAME), ("next", TERM_NAME)


def _expand(t: torch.Tensor, n_agents: int) -> torch.Tensor:
    """``[*B, 1]`` shared flag -> ``[*B, n_agents, 1]``, matching the reward shape."""
    return t.unsqueeze(-2).expand(*t.shape[:-1], n_agents, 1)


def build_policy(obs_dim: int, act_dim: int, n_agents: int, device: str):
    """Decentralised actor with shared weights (the MAPPO actor half).

    Shared with ``pusht_eval.py`` so a checkpoint loads into an identical module.
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
            num_cells=128,
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


def main() -> None:
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--n-envs", type=int, default=512)
    parser.add_argument("--steps-per-batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.95)
    # Per-step |dt angle| runs ~8.7x |dt distance| under a random policy, so the
    # scenario's raw 1.0/0.5 weights make the rotation term ~4x the position term
    # AND rotation is the easier of the two to influence — the policy then ignores
    # position entirely. These defaults rebalance the two terms.
    parser.add_argument("--pos-shaping", type=float, default=5.0)
    parser.add_argument("--rot-shaping", type=float, default=0.5)
    parser.add_argument("--checkpoint-dir", default="runs/pusht")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="0 disables")
    args = parser.parse_args()

    device = args.device
    env = WmasEnv(
        Environment(
            PushTScenario(
                n_agents=N_AGENTS,
                pos_shaping_factor=args.pos_shaping,
                rot_shaping_factor=args.rot_shaping,
            ),
            n_envs=args.n_envs,
            device=device,
            dt=0.05,
            seed=0,
            max_steps=100,
        )
    )
    obs_dim, act_dim = env.obs_dim, env.act_dim

    # Decentralised actor (shared weights), centralised critic — the usual MAPPO split.
    policy = build_policy(obs_dim, act_dim, N_AGENTS, device)
    critic = TensorDictModule(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=1,
            n_agents=N_AGENTS,
            centralised=True,
            share_params=True,
            device=device,
            depth=2,
            num_cells=128,
        ),
        in_keys=["observation"],
        out_keys=["state_value"],
    )

    frames_per_batch = args.n_envs * args.steps_per_batch
    collector = Collector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * args.iters,
        device=device,
        auto_register_policy_transforms=True,
    )
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=frames_per_batch // args.minibatches,
    )
    loss_module = ClipPPOLoss(actor_network=policy, critic_network=critic, entropy_coeff=1e-3)
    # Leaf names: the value estimator looks them up under ("next", ...) itself.
    loss_module.set_keys(
        reward="reward", done=DONE_NAME, terminated=TERM_NAME, value="state_value"
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=args.gamma, lmbda=args.lmbda
    )
    optim = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    scen = env._env.scenario
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ckpt_dir / "metrics.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["iter", "frames", "reward_per_step", "tee_dist", "tee_angle", "solved"])

    def save(tag: str) -> None:
        torch.save(
            {
                "policy": policy.state_dict(),
                "critic": critic.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "n_agents": N_AGENTS,
            },
            ckpt_dir / f"pusht_{tag}.pt",
        )

    for it, batch in enumerate(collector):
        dist = batch.get(("next", "info", "tee_dist_to_goal"))
        ang = batch.get(("next", "info", "tee_angle_error"))
        # `WmasEnv` reports the max_steps time limit as `terminated`, which would make
        # GAE cut the value bootstrap at every truncation. Recover the *task*
        # termination (the on-goal condition) from info so only real terminals cut,
        # and keep the wrapper's flag as `done` (terminated | truncated).
        terminated = ((dist < scen.goal_tolerance) & (ang < scen.angle_tolerance)).unsqueeze(-1)
        batch.set(DONE_KEY, _expand(batch.get(("next", "done")), N_AGENTS))
        batch.set(TERM_KEY, _expand(terminated, N_AGENTS))
        with torch.no_grad():
            loss_module.value_estimator(
                batch, params=loss_module.critic_network_params,
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
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                optim.step()
        buffer.empty()

        mean_reward = batch.get(("next", "reward")).mean().item()
        d, a = dist.mean().item(), ang.mean().item()
        solved = terminated.float().mean().item()
        csv_writer.writerow(
            [it, (it + 1) * frames_per_batch, f"{mean_reward:.6f}", f"{d:.6f}",
             f"{a:.6f}", f"{solved:.6f}"]
        )
        csv_file.flush()
        print(
            f"iter {it:3d}  reward/step {mean_reward:+.4f}  "
            f"tee-dist {d:.4f}  tee-angle {a:.4f}  solved {solved:.3f}"
        )
        if args.checkpoint_every and (it + 1) % args.checkpoint_every == 0:
            save(f"iter{it + 1:05d}")

    save("final")
    csv_file.close()
    collector.shutdown()
    print(f"\ncheckpoints + metrics.csv in {ckpt_dir}/")


if __name__ == "__main__":
    main()
