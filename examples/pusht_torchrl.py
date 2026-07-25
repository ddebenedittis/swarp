"""MAPPO on Push-T: four robots learn to push a T to a target pose, via TorchRL.

Wraps :class:`~wmas.scenarios.pusht.PushTScenario` in the batched TorchRL ``EnvBase``
from :mod:`wmas.interop.torchrl` and runs a small on-policy PPO loop with a shared
actor and a centralised critic. Everything stays on-device: the collector drives the
vectorized wmas step directly, no per-env Python loop.

Needs the optional torchrl group::

    uv pip install -e . --group torchrl

Run with::

    python examples/pusht_torchrl.py [--device cuda:0] [--iters 600] [--n-envs 512]

The defaults take ~1.5 min on a modern GPU (~10M frames) and show a clear trend:
``reward/step`` rises and ``tee-angle`` falls as the agents learn to spin the T into
the target orientation. Position (``tee-dist``) is the slower half of the task and
barely moves at this budget — it needs a substantially longer run.
"""

from __future__ import annotations

import argparse

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


def _expand_done(batch: torch.Tensor, n_agents: int) -> torch.Tensor:
    """``[*B, 1]`` shared done -> ``[*B, n_agents, 1]``, matching the reward shape."""
    return batch.unsqueeze(-2).expand(*batch.shape[:-1], n_agents, 1)


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
    args = parser.parse_args()

    device = args.device
    env = WmasEnv(
        Environment(
            PushTScenario(n_agents=N_AGENTS),
            n_envs=args.n_envs,
            device=device,
            dt=0.05,
            seed=0,
            max_steps=100,
        )
    )
    obs_dim, act_dim = env.obs_dim, env.act_dim

    # Decentralised actor (shared weights), centralised critic — the usual MAPPO split.
    actor_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * act_dim,  # loc + scale, split below
            n_agents=N_AGENTS,
            centralised=False,
            share_params=True,
            device=device,
            depth=2,
            num_cells=128,
        ),
        NormalParamExtractor(),
    )
    policy = ProbabilisticActor(
        module=TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True,
    )
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

    for it, batch in enumerate(collector):
        # Broadcast the shared per-env done onto the per-agent reward shape.
        batch.set(DONE_KEY, _expand_done(batch.get(("next", "done")), N_AGENTS))
        batch.set(TERM_KEY, _expand_done(batch.get(("next", "terminated")), N_AGENTS))
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
        dist = batch.get(("next", "info", "tee_dist_to_goal")).mean().item()
        ang = batch.get(("next", "info", "tee_angle_error")).mean().item()
        print(
            f"iter {it:3d}  reward/step {mean_reward:+.4f}  "
            f"tee-dist {dist:.4f}  tee-angle {ang:.4f}"
        )

    collector.shutdown()


if __name__ == "__main__":
    main()
