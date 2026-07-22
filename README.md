# wmas — Warp Multi-Agent Simulator

A fast, GPU-resident, **differentiable**, vectorized multi-agent simulator for 2D robotic
vehicles, built on [NVIDIA Warp](https://github.com/NVIDIA/warp) with zero-copy PyTorch
interop. Conceptually: [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator),
but compiled as Warp kernels instead of PyTorch tensor ops.

*(The name `wmas` is provisional.)*

## What you get

- **Batched worlds** — all state lives on-device as `[n_envs, n_agents]` Warp arrays;
  the hot loop performs no host↔device transfers (tested with an API guard and a CUDA
  profiler check).
- **Three vehicle models**, mixable per-agent in one world (heterogeneous fleets):
  holonomic point (velocity or acceleration control), differential drive
  (velocity or acceleration control), and kinematic bicycle (slip-angle β formulation).
  A `DronePlaceholder` marks where 6-DOF models plug in.
- **Differentiable end-to-end** — the full step (dynamics + soft collisions + walls) runs
  under Warp's adjoint tape and is exposed to PyTorch autograd through a custom
  `torch.autograd.Function` with zero-copy `wp.from_torch`/`wp.to_torch`.
  Backprop-through-time over multi-step rollouts works out of the box, verified with
  `torch.autograd.gradcheck` (float64 on CPU) plus analytic gradient tests.
- **Soft interactions** — spring-damper repulsion between agents, against circular
  obstacles, and against world bounds (`soft` walls or hard position `clamp`).
- **Neighbor search** — padded within-radius neighbor lists (no sync) and a COO
  `edge_index()` radius graph for GNN policies. Two interchangeable backends (per-env
  brute force and a `wp.HashGrid` over all envs) that are tested to agree exactly.
- **VMAS-style scenarios** — a `Scenario` ABC (`make_world`, `reset_world`,
  `observation(agent)`, `reward(agent)` with per-agent vs global terms separated) and a
  concrete multi-agent goal `NavigationScenario`.

## Install

Requires Python ≥ 3.12. A CUDA GPU is optional — everything also runs on CPU.

```bash
git clone <this-repo> wmas && cd wmas
uv venv
uv pip install -e . --group dev
uv run pytest          # dynamics, gradients, neighbors, collisions, determinism, ...
```

## Quickstart

```python
import torch
from wmas import Environment, NavigationScenario

scenario = NavigationScenario(n_agents=8, n_obstacles=2)
env = Environment(scenario, n_envs=4096, device="cuda:0", dt=0.05, seed=0)

obs = env.reset()                                  # [n_envs, n_agents, obs_dim], on GPU
for _ in range(100):
    actions = torch.rand(4096, 8, 2, device="cuda:0") * 2 - 1
    obs, reward, done, info = env.step(actions)    # all tensors stay on the GPU
```

`step` returns `(obs [n_envs, n_agents, obs_dim], reward [n_envs, n_agents],
done [n_envs] bool, info dict)`. There is no auto-reset: check `done` and call
`env.reset()` (as in raw VMAS). For GNN policies, `env.radius_graph()` returns a COO
`[2, E]` edge index of the current within-radius graph.

## Differentiability

Everything between actions and rewards is differentiable. Each step wraps your torch
tensors as Warp arrays (zero-copy), records the substep kernel launches on a `wp.Tape`,
and replays the adjoints in `backward()`. Under `torch.no_grad()` a tape-free fast path
with recycled buffers runs instead — same kernels, zero steady-state allocations.

Optimize an action sequence by gradient descent through the physics
(full script: [`examples/optimize_actions.py`](examples/optimize_actions.py)):

```python
import torch, warp as wp
from wmas import AgentConfig, DynamicsModel, Stepper, TorchState, WorldConfig, rollout

cfgs = [AgentConfig(model=DynamicsModel.DIFF_DRIVE, max_speed=1.0) for _ in range(4)]
stepper = Stepper(cfgs, dt=0.1, device="cuda:0", world=WorldConfig(collision_k=50.0))

actions = torch.zeros(30, 1, 4, 2, device="cuda:0", requires_grad=True)  # [T, envs, agents, 2]
opt = torch.optim.Adam([actions], lr=0.05)
for _ in range(200):
    opt.zero_grad()
    final, traj = rollout(stepper, state0, actions)   # BPTT through Warp adjoints
    loss = (final.pos - goals).square().sum()
    loss.backward()
    opt.step()
```

Gradient notes: actions are clamped to the agent limits inside the kernel (saturated
actions get zero gradient — verified finite); the neighbor *set* is a discrete
structure, so gradients flow through contact geometry, not through neighbor membership.

## Writing a scenario

Subclass `wmas.Scenario` and implement four methods (mirroring VMAS `BaseScenario`):

```python
import torch
from wmas import AgentConfig, DynamicsModel, Scenario, World, WorldConfig

class MyScenario(Scenario):
    def make_world(self, n_envs, device, dt, substeps, dtype) -> World:
        configs = [AgentConfig(model=DynamicsModel.HOLONOMIC, radius=0.05)
                   for _ in range(4)]
        world_config = WorldConfig(bounds=(-1, 1, -1, 1), bounds_mode="soft")
        self.world = World(configs, world_config, n_envs=n_envs, device=device,
                           dt=dt, substeps=substeps, dtype=dtype)
        return self.world

    def reset_world(self, env_indices=None):
        w = self.world           # write into w.state.* / w.goals; use w.sample_uniform
        w.state.pos.data[:] = w.sample_uniform((w.n_envs, w.n_agents, 2), -1.0, 1.0)

    def observation(self, agent_idx):          # [n_envs, obs_dim], torch ops on w.state
        s = self.world.state
        return torch.cat([s.pos[:, agent_idx], s.vel[:, agent_idx]], dim=-1)

    def agent_reward(self, agent_idx):         # per-agent term [n_envs]
        return -self.world.state.pos[:, agent_idx].norm(dim=-1)

    # optional: global_reward() (shared term), done(), info(), post_step(),
    # and batched observations()/rewards() overrides for large fleets.
```

Observations and rewards are plain torch ops over `world.state`, so they are
differentiable together with the Warp step and stay on-device.

## Throughput

`python -m wmas.benchmark.throughput` steps the full NavigationScenario hot path
(dynamics, neighbor lists, soft collisions, obs/reward). On an RTX 3070 Laptop GPU
(8 GB), 100 timed steps per config:

```
  n_envs  n_agents   ms/step    env-steps/s   agent-steps/s
-----------------------------------------------------------
    1000         4      0.87      1,145,235       4,580,941
    1000        16      0.88      1,132,956      18,127,289
    1000        64      1.06        939,880      60,152,292
    4000         4      0.88      4,525,406      18,101,625
    4000        16      0.88      4,543,839      72,701,417
    4000        64      3.74      1,069,207      68,429,221
    8000         4      0.88      9,105,610      36,422,441
    8000        16      1.19      6,728,371     107,653,938
    8000        64      7.41      1,079,907      69,114,040
   16000         4      0.90     17,871,014      71,484,057
   16000        16      2.29      6,983,890     111,742,243
   16000        64     14.84      1,077,920      68,986,909
```

## Layout

```
wmas/core        state, stepper (substep pipeline), neighbors, collisions, world, environment
wmas/dynamics    model tags/configs, unified integrate kernel, drone placeholder
wmas/interop     torch.autograd.Function bridge + BPTT rollout
wmas/scenarios   Scenario ABC + NavigationScenario
wmas/benchmark   throughput script
```

Design invariants worth knowing before extending:

- The step is **functional** (`state_in -> state_out`); anything written during a taped
  step (intermediate states, force buffers, neighbor lists) is allocated fresh per step —
  overwriting a taped array silently corrupts Warp adjoints.
- Kernels are generic over precision and explicitly instantiated for float32/float64
  (`wp.overload`); float64 on CPU is what makes strict `gradcheck` possible.
- Collision forces are gather-based (each agent sums over its own neighbor list): no
  atomics, deterministic, race-free.
- `wp.HashGrid` wraps cell coordinates modulo its dims, which penalizes many-env
  batching; the neighbor backend is selectable (`WorldConfig.neighbor_method`) and
  defaults to the per-env brute-force kernel, which is much faster for ≤ a few hundred
  agents per env.

## Roadmap

TorchRL wrapper and VMAS scenario ports; 6-DOF drone dynamics; lidar-style sensors;
RK4 integrator; a batched uniform-grid neighbor backend (radix-sort based) for huge
per-env populations; richer scenarios (formation, flocking, transport).
