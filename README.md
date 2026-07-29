# swarp — Swarm simulation on NVIDIA Warp

A fast, GPU-resident, **differentiable**, vectorized multi-agent simulator for 2D robotic
vehicles, built on [NVIDIA Warp](https://github.com/NVIDIA/warp) with zero-copy PyTorch
interop. Conceptually: [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator),
but compiled as Warp kernels instead of PyTorch tensor ops.

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
- **Movable rigid bodies** — an obstacle tagged `ObstacleKind.MOVABLE` carries a mass and
  inertia and is integrated *inside the substep loop* from the reaction of the very same
  agent contacts (Newton's third law of one shared force), so the pose an agent collides
  against is at most one substep old. Several shapes can share one body id to form a
  compound body with rotation — that is what makes `PushTScenario`'s T-shape work.
- **Neighbor search** — padded within-radius neighbor lists (no sync), an overflow flag
  when a list exceeds `max_neighbors` (so truncation is never silent), and a COO
  `edge_index()` radius graph for GNN policies. Two interchangeable backends (per-env
  brute force and a `wp.HashGrid` over all envs) that are tested to agree exactly.
- **VMAS-style scenarios** — a `Scenario` ABC (`make_world`, `reset_world`,
  `observation(agent)`, `reward(agent)` with per-agent vs global terms separated) and
  seven concrete scenarios: navigation, flocking, formation, discovery, sampling,
  transport, and Push-T. Each ships fused Warp obs/reward kernels alongside the torch
  reference implementation the fused path is parity-tested against. Host-sync-free masked
  reset (`env.reset_at(mask)`) and opt-in `auto_reset` keep the whole RL loop on-device.
- **Arbitrary action arity** — the action tensor is `[n_envs, n_agents, act_dim]` where
  `act_dim` is the max over agent models (2 for the 2D vehicles, 4 for the quadrotor);
  models read only the slots they use, so a wider action space drops in without touching
  the geometry.
- **Per-env parameter randomization** — agent params (mass, radius, speed/accel limits,
  wheelbase, …) are shared across envs by default (`[n_agents, P]`), the zero-overhead
  fast path. Opt in to domain randomization by handing
  `Stepper.set_agent_params_per_env` a `[n_envs, n_agents, P]` tensor (start from
  `per_env_float_template`); dedicated per-env kernel variants then index `params[e, a]`.
  The dynamics recurrence is shared with the default kernel, so both stay in lock-step;
  measured overhead on the hot path is ~2% in the mid band and within noise elsewhere.
- **Interactive viewer** (optional `viz` extra) — a pygame renderer with headless frames,
  mp4/webm export, notebook embedding, a batch mosaic, live overlay toggles, and light
  write-back (drag an agent, right-click to move its goal). See [Visualization](#visualization).

## Install

Requires Python ≥ 3.12. A CUDA GPU is optional — everything also runs on CPU.

```bash
git clone <this-repo> swarp && cd swarp
uv venv
uv pip install -e . --group dev
uv run pytest          # dynamics, gradients, neighbors, collisions, determinism, ...
uv pip install -e '.[viz]'   # optional: interactive viewer + video export
```

## Quickstart

```python
import torch
from swarp import Environment, NavigationScenario

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
from swarp import AgentConfig, DynamicsModel, Stepper, TorchState, WorldConfig, rollout

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

Subclass `swarp.Scenario` and implement four members (mirroring VMAS `BaseScenario`):

```python
import torch
from swarp import AgentConfig, DynamicsModel, Scenario, World, WorldConfig

class MyScenario(Scenario):
    obs_dim = 4                                # per-agent observation width

    def make_world(self, n_envs, device, dt, substeps, dtype) -> World:
        configs = [AgentConfig(model=DynamicsModel.HOLONOMIC, radius=0.05)
                   for _ in range(4)]
        world_config = WorldConfig(bounds=(-1, 1, -1, 1), bounds_mode="soft")
        self.world = World(configs, world_config, n_envs=n_envs, device=device,
                           dt=dt, substeps=substeps, dtype=dtype)
        return self.world                      # allocate all persistent state here too

    def reset_world(self, env_mask=None, *, obs_only=False):  # None = all envs
        w = self.world                         # write_state does the masked blend
        w.write_state(env_mask,
                      pos=w.sample_uniform((w.n_envs, w.n_agents, 2), -1.0, 1.0),
                      vel=0.0)

    def observations(self):                    # [n_envs, n_agents, obs_dim]
        s = self.world.state
        return torch.cat([s.pos, s.vel], dim=-1)

    def agent_reward(self, agent_idx):         # per-agent term [n_envs]
        return -self.world.state.pos[:, agent_idx].norm(dim=-1)

    # optional: global_reward() (shared term), done(), info(), post_step(),
    # render_extras(); observation(i) and rewards() are derived from the above.
```

Observations and rewards are plain torch ops over `world.state`, so they are
differentiable together with the Warp step and stay on-device. A scenario can additionally
subclass `FusedScenario` to supply fused Warp obs/reward kernels for the no-grad hot path,
declaring its buffers rather than hand-rolling the CUDA-graph bookkeeping — see
[docs/writing-a-scenario.md](docs/writing-a-scenario.md), which covers both tiers.

## Visualization

An optional pygame-based viewer (install the `viz` extra: `uv pip install -e '.[viz]'`)
renders one env of the batch — or a mosaic of the whole batch — either headless or in an
interactive window.

```python
from swarp import Environment, NavigationScenario
from swarp.render import Viewer, save_video

env = Environment(NavigationScenario(n_agents=5, n_obstacles=2), n_envs=16, device="cpu")
env.reset()

# Headless: an (H, W, 3) uint8 frame, or a rollout to mp4/webm (by extension).
frame = env.render(mode="rgb_array", env_index=0)   # VMAS-compatible signature
save_video(env, "nav.mp4", n_steps=200)             # pass action_fn=policy to drive it

# Interactive window: pan/zoom, hover to inspect, toggle overlays live, drag agents.
Viewer(env, mosaic=True).run()                      # pass action_fn=policy to drive it
```

Rendering is opt-in, read-only, and off the differentiable hot path — one device→host copy
per frame. Try it straight away:

```bash
python -m swarp.render.demo               # interactive window (goal-seeking demo policy)
python -m swarp.render.demo --mosaic      # grid of all envs + a focus pane
python -m swarp.render.demo --save nav.webm --steps 200
```

**Controls** — wheel zoom, middle-drag pan, hover an agent to inspect it, `[` / `]` to step
through envs (or click a mosaic tile to focus it), space to pause. Overlays toggle by key:
`g` goals, `n` neighbor graph, `h` heading, `v` velocity, `i` ids, `o` obstacles, `b`
bounds, `l` lidar (drawn once a sensor supplies rays). Left-drag an agent to reposition it;
right-click to move its goal (writes into the shown env only).

In a notebook, embed a rollout inline:

```python
from swarp.render import animate
animate(env, n_steps=200)   # returns an HTML5 <video>
```

Scenarios feed custom drawables to the viewer via `Scenario.render_extras(env_idx) -> dict`.

## Benchmarks

On a single RTX 3070 Laptop GPU (8 GB), the full NavigationScenario hot path — dynamics,
neighbor lists, soft collisions, and obs/reward — sustains **24.1 M env-steps/s** at 16,000
envs × 16 agents (385 M agent-steps/s), peaking at **491 M agent-steps/s** at 16,000 × 64.
Almost every batch size lands on the same ~0.65–0.8 ms/step host floor, so raising
`n_agents` is close to free until the neighbor/force kernels saturate the GPU.

```bash
python -m swarp.benchmark.throughput      # the full (n_envs, n_agents) grid
python -m swarp.benchmark.compare_vmas    # vs VMAS: --metric {throughput,memory,both}
python -m swarp.benchmark.compare_sims    # vs VMAS, JaxMARL, and CAMAR
```

Full tables, the head-to-heads against VMAS / JaxMARL / CAMAR, the optimization ablation,
and the multi-venv setup those comparisons need are in
**[docs/benchmarks.md](docs/benchmarks.md)**.

swarp is *API-compatible in spirit, not trajectory-compatible* with VMAS: the dynamics,
collision constants, and observation model differ by design, so identical actions do not
reproduce VMAS trajectories.

## Layout

```
swarp/core       state (SoA), stepper (THE substep pipeline), neighbors, collisions,
                 movable rigid bodies, world, environment, config
swarp/dynamics   model tags/configs, unified integrate kernel (2D vehicles + drone)
swarp/interop    torch.autograd bridge + BPTT rollout, torch.compile custom op,
                 CUDA-graph capture, TorchRL EnvBase wrapper
swarp/scenarios  Scenario ABC + 7 scenarios (navigation, flocking, formation, discovery,
                 sampling, transport, pusht), each with its fused *_kernels.py
swarp/sensors    opt-in differentiable observation sensors (lidar: torch + Warp backends)
swarp/render     optional pygame viewer: renderer, overlays, camera, HUD, input handling,
                 mosaic layout, video export, notebook embed, demo entry point
swarp/benchmark  throughput, optimization ablation, per-scenario sweep, and two
                 cross-simulator comparisons (+ _adapters/ for swarp/vmas/jaxmarl/camar)
examples/        standalone scripts: action optimization, Push-T eval, Push-T + TorchRL
docs/            benchmarks, scenario-authoring notes
tests/           pytest suite
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

swarp deliberately does **not** aim to be a drop-in physics clone of VMAS. Out of scope
for now:

- **Trajectory parity with VMAS** — see the parity caveat above.
- **Movable bodies on the adjoint tape.** Movable rigid bodies themselves are *not* out of
  scope any more: `ObstacleKind.MOVABLE` bodies are integrated by `swarp/core/bodies.py`
  inside the substep loop, with rotation and with compound multi-shape bodies
  (`PushTScenario`). What is missing is differentiability *through* them — that
  integration runs with `record_tape=False`, so no gradient flows through a body's motion.
  A scenario that needs one keeps its own torch-side copy and integrates it itself
  (`PushTScenario` on the grad path; `TransportScenario` always, staggered by one step),
  which gives BPTT body→agent→action across a rollout but not through the intra-step
  agent-avoids-body force.
- **Joints** (VMAS `balance`) and **box-box obstacle-obstacle contacts** — obstacles
  collide with each other only when at least one of the pair is round; two boxes pass
  through one another, since that needs a polygon contact manifold rather than an SDF
  evaluated against a disc. Agent-vs-box is exact for every shape.
- **Discrete or communication action spaces** — actions are continuous real vectors.

## Roadmap

Next: putting movable bodies on the adjoint tape, and **joints** (VMAS `balance` parity).

Everything already landed is listed in [CHANGELOG.md](CHANGELOG.md).
