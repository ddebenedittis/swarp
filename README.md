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
- **Interactive viewer** (optional `viz` extra) — a pygame renderer with headless frames,
  mp4/webm export, notebook embedding, a batch mosaic, live overlay toggles, and light
  write-back (drag an agent, right-click to move its goal). See [Visualization](#visualization).

## Install

Requires Python ≥ 3.12. A CUDA GPU is optional — everything also runs on CPU.

```bash
git clone <this-repo> wmas && cd wmas
uv venv
uv pip install -e . --group dev
uv run pytest          # dynamics, gradients, neighbors, collisions, determinism, ...
uv pip install -e '.[viz]'   # optional: interactive viewer + video export
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

## Visualization

An optional pygame-based viewer (install the `viz` extra: `uv pip install -e '.[viz]'`)
renders one env of the batch — or a mosaic of the whole batch — either headless or in an
interactive window.

```python
from wmas import Environment, NavigationScenario
from wmas.render import Viewer, save_video

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
python -m wmas.render.demo               # interactive window (goal-seeking demo policy)
python -m wmas.render.demo --mosaic      # grid of all envs + a focus pane
python -m wmas.render.demo --save nav.webm --steps 200
```

**Controls** — wheel zoom, middle-drag pan, hover an agent to inspect it, `[` / `]` to step
through envs (or click a mosaic tile to focus it), space to pause. Overlays toggle by key:
`g` goals, `n` neighbor graph, `h` heading, `v` velocity, `i` ids, `o` obstacles, `b`
bounds, `l` lidar (drawn once a sensor supplies rays). Left-drag an agent to reposition it;
right-click to move its goal (writes into the shown env only).

In a notebook, embed a rollout inline:

```python
from wmas.render.notebook import animate
animate(env, n_steps=200)   # returns an HTML5 <video>
```

Scenarios feed custom drawables to the viewer via `Scenario.render_extras(env_idx) -> dict`.

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

### vs. JaxMARL & CAMAR

`python -m wmas.benchmark.compare_sims` broadens the head-to-head to the JAX-based field:
wmas vs [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator),
[JaxMARL](https://github.com/flairox/jaxmarl) (`MPE_simple_spread_v3`, continuous
cooperative navigation), and [CAMAR](https://github.com/AIRI-Institute/CAMAR)
(`random_grid` + `HolonomicDynamic` continuous navigation). All four run the same
env-steps/s measurement on the one scenario they share — continuous 2D
navigation-to-goal with collision avoidance — anchored on wmas.

VMAS pins `numpy < 2` while JaxMARL/CAMAR are JAX (recent numpy), so the four cannot
share one venv; JaxMARL and CAMAR also pin different jax versions. Each simulator is
therefore installed into its own Python 3.12 venv and benchmarked in its own
subprocess (JAX and torch never share a process, so they don't fight over VRAM):

```bash
for v in .venv .venv-jaxmarl .venv-camar; do uv venv --python 3.12 $v; done
VIRTUAL_ENV=.venv         uv pip install -e . --group dev --group bench     # wmas + vmas
VIRTUAL_ENV=.venv-jaxmarl uv pip install -e . --group bench-jaxmarl         # jaxmarl
VIRTUAL_ENV=.venv-camar   uv pip install -e . --group bench-camar           # camar
```

#### Getting JaxMARL onto the GPU (do this — it silently runs on CPU otherwise)

JaxMARL pins an older `jax`, and its dependency resolution pulls the **CPU-only**
`jaxlib`. If you skip this step JAX falls back to CPU and the JaxMARL column is a
CPU number that is meaningless next to the GPU sims — so `compare_sims.py` **refuses
to report it**: the adapter asserts a GPU device and the cell prints `CPU-only`
instead of a bogus figure.

The fix is to install the CUDA build of `jaxlib` *at the exact `jax` version JaxMARL
already pinned* (a different version would break JaxMARL). Find that version, then
install the matching CUDA wheels into the JaxMARL venv only:

```bash
# 1. discover the jax version JaxMARL installed (e.g. 0.4.38)
.venv-jaxmarl/bin/python -c "import jax; print(jax.__version__)"

# 2. install the CUDA-enabled build at that same version (adjust cuda12 -> your CUDA)
VIRTUAL_ENV=.venv-jaxmarl uv pip install "jax[cuda12]==0.4.38"

# 3. verify JAX now sees the GPU (must print a CudaDevice, not CpuDevice)
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv-jaxmarl/bin/python -c "import jax; print(jax.devices())"
```

`compare_sims.py` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` in each JAX child so JAX
grows VRAM on demand instead of grabbing ~75% up front; CAMAR (`bench-camar`) ships a
CUDA `jaxlib` already and needs no such fix.

One combined command drives all three subprocesses via a per-sim interpreter map and
prints a single table (the numpy split lives entirely at the subprocess boundary):

```bash
.venv/bin/python -m wmas.benchmark.compare_sims --device cuda:0 \
    --envs 1024 4096 16384 --agents 3 16 \
    --python jaxmarl=.venv-jaxmarl/bin/python \
    --python camar=.venv-camar/bin/python
```

The launching interpreter (`.venv/bin/python` here) runs `wmas` and `vmas` in-process
by default; `--python SIM=PATH` overrides the interpreter for a given simulator, and
any sim without an override uses the launcher (or `--python-default PATH`). To compare
**just wmas vs JaxMARL**, restrict the sims and point JaxMARL at its venv:

```bash
.venv/bin/python -m wmas.benchmark.compare_sims --device cuda:0 \
    --sims wmas jaxmarl --agents 3 16 \
    --python jaxmarl=.venv-jaxmarl/bin/python
```

`--sims` picks which simulators to run (`wmas` is the anchor and should stay in);
`--agents`/`--envs` set the sweep. JaxMARL's `MPE_simple_spread_v3` honors any agent
count (`num_agents` is configurable), so no cell is skipped for an agent mismatch; if a
simulator ever cannot match the requested count, its number is flagged with `*` and the
realized count is listed under the table. You can drop the JaxMARL comparison entirely
by omitting it from `--sims`.

#### wmas configurations

The three `wmas` entries are hot-path *configurations* of the same simulator, so the
benchmark doubles as an optimization ablation: `wmas-eager` (torch obs/reward, no
CUDA graph — the baseline), `wmas-fused` (fused Warp obs/reward kernels), and `wmas`
(fused **+** CUDA-graph capture — the shipped default). env-steps/s on the RTX 3070
Laptop GPU (float32, 60 timed steps):

```
  n_envs  n_agents |  wmas-eager  wmas-fused        wmas |  fused/eager  opt/eager
    4096         3 |   5,838,664   7,877,854  15,945,297 |       1.35x      2.73x
   16384         3 |  21,772,740  30,814,526  61,718,330 |       1.42x      2.83x
    4096        16 |   5,998,275   6,783,331  15,094,932 |       1.13x      2.52x
   16384        16 |   6,858,433  28,191,165  35,637,271 |       4.11x      5.20x
```

Fusing the obs/reward layer into Warp kernels is worth ~1.1–1.4× on its own; the
CUDA graph (which elides per-step kernel-launch overhead) roughly doubles it again,
for ~2.5–5× end-to-end over the eager baseline. The graph's win grows with batch size,
where launch overhead is the binding constraint.

#### wmas (optimized) vs the field — matched task & observations

To make it as apples-to-apples as a four-engine comparison can be, the **default**
`--sims` set controls for the two biggest asymmetries: all four run **obstacle-free**
open-field navigation with **raycasting-free, relative-position observations**:

| sim | config for the matched comparison |
|-----|-----------------------------------|
| `wmas` | optimized (fused Warp kernels + CUDA graph); neighbor-list obs, no obstacles |
| `vmas-nolidar` | VMAS navigation, `collisions=False` → no 12-ray lidar, relative-position obs |
| `jaxmarl` | `MPE_simple_spread_v3` continuous — relative-position obs (already raycast-free) |
| `camar` | **open arena** (`string_grid`, no obstacles, `frameskip=1`) instead of the default ~800-obstacle `random_grid` |

The JAX sims fold obs+reward into the scan carry so XLA cannot dead-code-eliminate
them, and JIT/XLA compile is excluded via warmup at the timed length. Both
`env-steps/s` and `agent-steps/s` (= env-steps/s × agents) are reported:

> **What this measures.** These figures are the *simulator* step only — physics +
> obs + reward — not a full RL loop. wmas folds that whole step (physics, neighbor
> query, and fused obs/reward) into a single CUDA graph, so a step is one graph
> replay with no per-launch host floor. In policy-in-the-loop RL the JAX sims
> `jit` the policy *and* env together into one XLA program, whereas wmas replays
> the env graph and runs the policy as separate launches; the end-to-end training
> throughput of each therefore depends on how the policy is compiled alongside the
> simulator, which this benchmark does not capture.

```
metric: env-steps/s
  n_envs  n_agents |         wmas  vmas-nolidar      jaxmarl        camar | vnl/w   jax/w  camar/w
    1024         3 |    4,717,415       633,305   13,731,386   11,644,687 | 0.13x   2.91x    2.47x
    4096         3 |   16,942,170     2,489,270   38,760,087   33,327,778 | 0.15x   2.29x    1.97x
   16384         3 |   63,640,378     9,946,801   65,213,783   67,508,356 | 0.16x   1.02x    1.06x
    1024        16 |    3,868,680       141,767    2,218,927    2,615,686 | 0.04x   0.57x    0.68x
    4096        16 |   15,595,567       549,767    2,231,675    3,353,155 | 0.04x   0.14x    0.22x
   16384        16 |   35,647,177     2,122,258    1,423,262    3,682,337 | 0.06x   0.04x    0.10x

metric: agent-steps/s
  n_envs  n_agents |         wmas  vmas-nolidar      jaxmarl        camar | vnl/w   jax/w  camar/w
    1024         3 |   14,152,244     1,899,916   41,194,157   34,934,061 | 0.13x   2.91x    2.47x
    4096         3 |   50,826,510     7,467,809  116,280,262   99,983,333 | 0.15x   2.29x    1.97x
   16384         3 |  190,921,134    29,840,404  195,641,349  202,525,067 | 0.16x   1.02x    1.06x
    1024        16 |   61,898,884     2,268,278   35,502,837   41,850,981 | 0.04x   0.57x    0.68x
    4096        16 |  249,529,074     8,796,274   35,706,801   53,650,474 | 0.04x   0.14x    0.22x
   16384        16 |  570,354,838    33,956,129   22,772,199   58,917,397 | 0.06x   0.04x    0.10x
```
(Ratios are identical in both tables — with agent counts matched across sims, the
`×agents` factor cancels. `agent-steps/s` is there for the absolute magnitudes.)

Reading it, once the task and obs are matched:

- **The two JAX sims (JaxMARL, CAMAR) win at few agents + low envs** (~2–3× at 3 agents,
  1024–4096 envs), because they run the whole rollout as one jitted `lax.scan` with no
  per-step launch floor. **wmas is latency-bound there** — its step floors at ~0.24 ms
  regardless of `n_envs` (1024→16384) *or* agent count (3→16), a Python-per-step-loop cost
  the JAX fused rollout doesn't pay. This is the real reason JAX leads at low occupancy.
- **They converge by 16384 envs** at 3 agents (all within ~6% — wmas 63.6 M, JaxMARL 65.2 M,
  CAMAR 67.5 M) as wmas's launch floor amortizes across more work.
- **wmas dominates at 16 agents** (JAX sims fall to 0.04–0.68×). wmas's step is a fixed
  handful of fused Warp kernels, nearly insensitive to agent count — its `agent-steps/s`
  climbs from ~14 M (3 agents) to ~570 M (16 agents), ~40×, doing far more work in about
  the same wall time — whereas MPE/CAMAR per-step cost grows with agents.
- **`vmas-nolidar` is consistently slowest** (0.04–0.16×): a Python per-entity step loop
  that neither the JAX scan nor Warp kernels have to pay.

**The default set is now genuinely comparable; the earlier "CAMAR is ~200× slower"
result was almost entirely CAMAR's ~800-obstacle default map** — the same open-arena
CAMAR here is 20–120× faster than that (`--sims camar-grid` reproduces the heavy native
task). Residual, deliberate differences: exact obs dimensionality still differs per sim;
`vmas-nolidar` also loses agent-agent collision physics (VMAS gates collisions and lidar
on one flag); dynamics/collision constants differ so trajectories won't match. Use
`--sims wmas vmas jaxmarl camar-grid` to compare each engine in its *native* setup instead.

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
- **In-tape rigid-body payloads / joints** — `TransportScenario` provides a *first* movable
  circular package (agents push it to a goal) via staggered coupling at the torch layer
  (differentiable across a rollout), but full in-step rigid-body payloads with rotation
  fully on the Warp adjoint tape, non-circular bodies, and **joints** (VMAS `balance`) are
  still out of scope.
- **Discrete or communication action spaces** — actions are continuous real vectors.

## Roadmap

Done recently: interactive pygame viewer (headless frames, mp4/webm export, mosaic view,
live overlay toggles, drag/goal write-back); box/segment static collision geometry;
arbitrary action arity; host-sync-free masked/auto reset; allocation-free no-grad hot path;
per-env parameter randomization;
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
and `FormationScenario` (hold polygon slots); a first **movable-package**
`TransportScenario` (agents push a circular payload to a goal; staggered torch-layer
coupling, differentiable across a rollout); and a **torch.compile-compatible step**
(`wmas.interop.compile.compiled_warp_step`, a `torch.library.custom_op` with fake +
autograd rules) plus an optional **CUDA-graph capture** of the no-grad hot path
(`CudaGraphStep`); and a **TorchRL `EnvBase` wrapper** (`wmas.interop.torchrl.WmasEnv`,
batched `TensorDict` specs, `--group torchrl`) that passes TorchRL's `check_env_specs`.

Next: in-tape rigid-body payloads and joints (VMAS `transport`/`balance` parity).
