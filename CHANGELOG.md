# Changelog

All notable changes to `wmas`. Newest first. Nothing has been released yet — version is
`0.1.0` and the API is pre-1.0, so anything here may still change.

## Unreleased

### Scenarios

- **Push-T** (`PushTScenario`): agents push a T-shaped **movable compound rigid body** to a
  target pose. Two oriented `BOX` shapes share one body id and are placed by body-frame
  offsets, so `wmas.core.bodies` integrates a single rigid pose (position *and* rotation)
  for the pair from the reaction of the very same agent contacts — inside the substep loop,
  so the pose an agent collides against is at most one substep old. Mass splits by area and
  inertia comes from the parallel-axis theorem about the area centroid, which makes the
  orientation half of the task solvable.
- **Movable obstacles** (`ObstacleKind.MOVABLE`): obstacles that carry mass and inertia and
  are advanced from agent-contact reaction forces (Newton's third law of one shared force)
  by `wmas/core/bodies.py`, gather-style with a thread per `(env, obstacle)` and no atomics.
  Obstacle-vs-obstacle contacts are modelled for every pair in which at least one body is
  round; box-box is not (that needs a polygon manifold, not an SDF against a disc).
- **Transport** (`TransportScenario`): a first movable circular package agents push to a
  goal, coupled at the torch layer (staggered by one step) so gradients flow
  package→agent→action across a rollout.
- Four circle-compatible **VMAS-style scenario ports**: `SamplingScenario` (consume a
  batched sum-of-Gaussians field), `DiscoveryScenario` (cover targets that each need
  several agents), `FlockingScenario` (Reynolds boids reward), and `FormationScenario`
  (hold polygon slots).

### Performance

- **Whole-step CUDA-graph capture**: physics plus fused obs/reward captured into one graph,
  so a step is a single graph replay with no per-launch host floor.
- **Fused Warp obs/reward/done kernels** for every scenario, with the torch
  implementations kept as the parity oracle the fused kernels are tested against.
- Hot-path overhaul: neighbor-build dedupe (reuse the previous step's list for substep 0 on
  the no-grad path, bit-identical to a fresh build), slim-2D state paths, eager trims, and
  an allocation-free no-grad step at steady state.
- **CUDA-graph capture of the no-grad hot path** as a standalone wrapper
  (`wmas.interop.persistent.CudaGraphStep`).
- A **batched uniform-grid neighbor backend** (`neighbor_method="uniform_grid"`,
  radix-sort based) that stays linear in `n_envs` and beats brute force past ~512
  agents/env (~4× at 1k, ~10× at 4k on an RTX 3070).

### Visualization

- Visualization overhaul: correct obstacle shapes (box/segment drawn as their true
  geometry), anti-aliased primitives, an action overlay, faster overlay rendering, and a
  comm-line overlay driven by the applied action now exposed on `World`.
- **Interactive pygame viewer** (`viz` extra): headless `(H, W, 3)` frames, mp4/webm
  export, notebook `<video>` embedding, a batch mosaic with a focus pane, live overlay
  toggles, and light write-back (drag an agent, right-click to move its goal).

### Sensors & dynamics

- **Warp-kernel lidar backend** for flat-memory high-ray-count scans, alongside the
  original torch implementation.
- **Lidar sensor** (`wmas.Lidar`): a differentiable, vectorized ray-cast returning per-ray
  ranges against circular agents and obstacles, opt-in as an observation component a
  scenario concatenates.
- **6-DOF quadrotor drone** model: quaternion attitude and body-rate dynamics with four
  rotor-thrust commands, inside the same unified differentiable step. The state SoA grew
  to carry altitude / vertical-velocity / attitude / body-rate fields that the 2D models
  pass through (~9% latency cost at tiny per-env batches).
- **RK4 integrator** (`Integrator.RK4`): four evaluations of a pure derivative `@wp.func`,
  sharing the recurrence with the Euler path.
- **Per-env parameter randomization**: hand `Stepper.set_agent_params_per_env` a
  `[n_envs, n_agents, P]` tensor (start from `per_env_float_template`) and dedicated
  per-env kernel variants index `params[e, a]`. The shared-params fast path
  (`[n_agents, P]`) stays the default and zero-overhead; measured per-env overhead on the
  hot path is ~2% in the mid band and within noise elsewhere.

### Interop

- **TorchRL `EnvBase` wrapper** (`wmas.interop.torchrl.WmasEnv`, `--group torchrl`):
  batched `TensorDict` specs, passes TorchRL's `check_env_specs`.
- **`torch.compile`-compatible step** (`wmas.interop.compile.compiled_warp_step`): a
  `torch.library.custom_op` with fake-tensor and autograd rules.

### Core

- **Box and segment/capsule static collision geometry**, boxes via an analytic signed
  distance field so a penetrating agent is still pushed out.
- **Host-sync-free masked and automatic reset** (`env.reset_at(mask)`, `auto_reset=True`) —
  no `.any()`/`.nonzero()` round-trip.
- **Arbitrary action arity**: the action tensor is `[n_envs, n_agents, act_dim]` with
  `act_dim` the max over agent models, decoupling the action space from the geometry.
- **Neighbor-list overflow is surfaced**, not silently truncated.
- Cross-simulator throughput benchmark (`wmas.benchmark.compare_sims`: wmas vs VMAS vs
  JaxMARL vs CAMAR, one subprocess per simulator) and a VMAS head-to-head
  (`wmas.benchmark.compare_vmas`). See [docs/benchmarks.md](docs/benchmarks.md).

### Repo

- MIT `LICENSE`, GitHub Actions CI (ruff + the CPU test suite on Python 3.12), a tracked
  `uv.lock`, and `docs/` split out of the README.

What is planned next lives in the README's [Roadmap](README.md#roadmap) section.
