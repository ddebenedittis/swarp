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
- **Four dynamics models**, mixable per-agent in one world (heterogeneous fleets):
  holonomic point (velocity or acceleration control), differential drive
  (velocity or acceleration control), kinematic bicycle (slip-angle β formulation),
  and a **6-DOF quadrotor drone** (quaternion attitude + body-rate dynamics, four
  rotor-thrust commands). All are differentiable and integrate with Euler or RK4.
- **Differentiable end-to-end** — the full step (dynamics + soft collisions + walls) runs
  under Warp's adjoint tape and is exposed to PyTorch autograd through a custom
  `torch.autograd.Function` with zero-copy `wp.from_torch`/`wp.to_torch`.
  Backprop-through-time over multi-step rollouts works out of the box, verified with
  `torch.autograd.gradcheck` (float64 on CPU) plus analytic gradient tests.
- **Soft interactions** — spring-damper repulsion between agents, against static
  obstacles (**circle, oriented box, or segment/capsule**; boxes use an analytic signed
  distance field so a penetrating agent is still pushed out), and against world bounds
  (`soft` walls or hard position `clamp`).
- **Neighbor search** — padded within-radius neighbor lists (no sync), an overflow flag
  when a list exceeds `max_neighbors` (so truncation is never silent), and a COO
  `edge_index()` radius graph for GNN policies. Two interchangeable backends (per-env
  brute force and a `wp.HashGrid` over all envs) that are tested to agree exactly.
- **VMAS-style scenarios** — a `Scenario` ABC (`make_world`, `reset_world`,
  `observation(agent)`, `reward(agent)` with per-agent vs global terms separated) and a
  concrete multi-agent goal `NavigationScenario`. Host-sync-free masked reset
  (`env.reset_at(mask)`) and opt-in `auto_reset` keep the whole RL loop on-device.
- **Arbitrary action arity** — the action tensor is `[n_envs, n_agents, act_dim]` where
  `act_dim` is the max over agent models (2 for the current 2D vehicles); models read
  only the slots they use, so a wider action space (e.g. a future drone) drops in without
  touching the geometry.
- **Per-env parameter randomization** — agent params (mass, radius, speed/accel limits,
  wheelbase, …) are shared across envs by default (`[n_agents, P]`), the zero-overhead
  fast path. Opt in to domain randomization by handing
  `Stepper.set_agent_params_per_env` a `[n_envs, n_agents, P]` tensor (start from
  `per_env_float_template`); dedicated per-env kernel variants then index `params[e, a]`.
  The dynamics recurrence is shared with the default kernel, so both stay in lock-step;
  measured overhead on the hot path is ~2% in the mid band and within noise elsewhere.

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
done [n_envs] bool, info dict)`. By default there is no auto-reset (as in raw VMAS):
check `done` and call `env.reset()`, or `env.reset_at(done)` to reset just the finished
envs, or construct with `auto_reset=True` to have `step` do it in-place — all three stay
host-sync-free (no `.any()`/`.nonzero()` round-trip). For GNN policies,
`env.radius_graph()` returns a COO `[2, E]` edge index of the current within-radius graph.

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

    def reset_world(self, env_mask=None):    # None = all; else bool [n_envs] mask
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
    1000         4      0.76      1,315,991       5,263,966
    1000        16      0.76      1,309,226      20,947,618
    1000        64      1.04        964,776      61,745,656
    4000         4      0.76      5,288,951      21,155,804
    4000        16      0.76      5,270,723      84,331,570
    4000        64      3.76      1,062,932      68,027,651
    8000         4      0.76     10,523,470      42,093,881
    8000        16      1.19      6,739,194     107,827,109
    8000        64      7.46      1,072,564      68,644,106
   16000         4      0.77     20,875,875      83,503,501
   16000        16      2.29      6,998,961     111,983,371
   16000        64     14.90      1,073,575      68,708,813
```

The tape-free hot path is allocation-free at steady state (recycled scratch plus a
ping-pong output buffer), which is what makes the small-per-env configs latency-bound at
~0.76 ms/step; large-per-env configs are bound by the neighbor/force kernels instead.

### vs. VMAS

`python -m wmas.benchmark.compare_vmas` pits wmas against
[VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) on the navigation
scenario (the one both implement), across a `(n_envs, n_agents)` grid. Install the
comparison deps first: `uv pip install -e . --group bench` (pulls in `vmas`; the script
also falls back to a `./VectorizedMultiAgentSimulator` checkout if present). Use
`--metric memory` for peak device memory and `--metric both` for both.

env-steps/s on the same RTX 3070 Laptop GPU (float32, 60 timed steps). `vmas-lidar` is
VMAS's default navigation (collisions + 12-ray lidar); `vmas-simple` disables both as a
lidar-free lower bound:

```
 n_envs n_agents |          wmas |    vmas-lidar |   vmas-simple | wmas/lidar  wmas/simple
   16384        4 |    18,751,408 |     2,094,070 |     8,163,160 |      8.95x        2.30x
   16384       16 |     6,970,898 |       143,174 |     2,275,756 |     48.69x        3.06x
    1024        4 |     1,232,014 |       186,642 |       544,501 |      6.60x        2.26x
    1024       16 |     1,247,845 |        20,672 |       154,476 |     60.36x        8.08x
```

wmas is ~6–9× faster than VMAS's default navigation at 4 agents and ~40–75× at 16 agents:
its whole step is ~3 fused Warp kernels regardless of agent count, whereas VMAS dispatches
per-entity (and O(entities²) pairwise) PyTorch ops in Python, and its lidar raycasts every
agent against every entity. Peak device memory is comparable to VMAS-with-lidar and small in
absolute terms (≤ ~300 MiB across the grid), so throughput — not memory — is the constraint.

**Parity caveat.** wmas is *API-compatible in spirit, not trajectory-compatible* with
VMAS. The dynamics (semi-implicit Euler, no drag, force-as-velocity-contribution for the
nonholonomic models), collision constants, and observation model (padded neighbor lists,
not lidar) differ by design, so identical actions do **not** reproduce VMAS trajectories.
The comparison above measures throughput/memory on the shared navigation task, nothing
more.

## Layout

```
wmas/core        state, stepper (substep pipeline), neighbors, collisions, world, environment
wmas/dynamics    model tags/configs, unified integrate kernel (2D vehicles + drone)
wmas/interop     torch.autograd.Function bridge + BPTT rollout
wmas/scenarios   Scenario ABC + NavigationScenario
wmas/sensors     opt-in differentiable observation sensors (lidar)
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

## Non-goals (current scope)

wmas deliberately does **not** aim to be a drop-in physics clone of VMAS. Out of scope
for now:

- **Trajectory parity with VMAS** — see the parity caveat above.
- **Movable non-circular rigid bodies** — obstacles (box/segment) are *static*. Pushable
  box payloads with rotational rigid-body dynamics and **joints** (VMAS `transport`,
  `balance`) need a rigid-body model that isn't built yet.
- **Rendering** — no viewer; inspect state tensors / plot yourself.
- **Discrete or communication action spaces** — actions are continuous real vectors.

## Roadmap

Done recently: box/segment static collision geometry; arbitrary action arity; host-sync-
free masked/auto reset; allocation-free no-grad hot path; per-env parameter randomization;
RK4 integrator (`Integrator.RK4`, four evaluations of a pure derivative `@wp.func`); a
batched uniform-grid neighbor backend (`neighbor_method="uniform_grid"`, radix-sort based)
that stays linear in `n_envs` and beats brute force past ~512 agents/env (~4x at 1k, ~10x
at 4k on an RTX 3070); a **6-DOF quadrotor drone** model (quaternion attitude in the unified
differentiable step; the state SoA grew to carry altitude/vertical-velocity/attitude/body-
rate fields that the 2D models pass through, ~9% latency cost at tiny per-env batches).

a **lidar sensor** (`wmas.Lidar`): a differentiable, vectorized ray-cast returning per-ray
ranges against circular agents/obstacles, opt-in as an observation component a scenario
concatenates; and four circle-compatible **VMAS-style scenario ports** —
`SamplingScenario` (consume a batched sum-of-Gaussians field), `DiscoveryScenario`
(cover targets that each need several agents), `FlockingScenario` (Reynolds boids reward),
and `FormationScenario` (hold polygon slots).

Next: TorchRL wrapper; then movable rigid-body payloads for transport/balance.
