"""Differentiability demo: gradient-descend an action sequence to reach goals.

Backprop-through-time through the Warp dynamics (and soft collisions) tunes a
30-step plan for four diff-drive agents so each parks on its goal.

Run:  python examples/optimize_actions.py [--device cpu]
"""

from __future__ import annotations

import argparse

import torch
import warp as wp

from wmas import AgentConfig, ControlMode, DynamicsModel, Stepper, TorchState, WorldConfig, rollout

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--iters", type=int, default=200)
args = parser.parse_args()

device, T, n_agents = args.device, 30, 4
cfgs = [
    AgentConfig(
        model=DynamicsModel.DIFF_DRIVE, ctrl_mode=ControlMode.VELOCITY,
        radius=0.05, max_speed=1.0, max_ang_vel=3.0,
    )
    for _ in range(n_agents)
]
stepper = Stepper(cfgs, dt=0.1, device=device, dtype=wp.float32,
                  world=WorldConfig(collision_k=50.0, collision_margin=0.02))

zeros = torch.zeros(1, n_agents, device=device)
state0 = TorchState(
    pos=torch.tensor([[[-1.0, y] for y in (-0.3, -0.1, 0.1, 0.3)]], device=device),
    theta=zeros.clone(), vel=torch.zeros(1, n_agents, 2, device=device),
    speed=zeros.clone(), ang_vel=zeros.clone(),
)
goals = torch.tensor([[[1.0, y] for y in (0.3, 0.1, -0.1, -0.3)]], device=device)  # crossing paths

actions = torch.zeros(T, 1, n_agents, 2, device=device, requires_grad=True)
opt = torch.optim.Adam([actions], lr=0.05)

for it in range(args.iters):
    opt.zero_grad()
    final, traj = rollout(stepper, state0, actions)
    goal_loss = (final.pos - goals).square().sum()
    effort = 1e-3 * actions.square().sum()
    loss = goal_loss + effort
    loss.backward()
    opt.step()
    if it % 40 == 0 or it == args.iters - 1:
        print(f"iter {it:4d}  loss {loss.item():.6f}  final-dist "
              f"{(final.pos - goals).norm(dim=-1).mean().item():.4f}")

print("final positions:", final.pos.detach().cpu().numpy().round(3).tolist())
print("goals:          ", goals.cpu().numpy().round(3).tolist())
