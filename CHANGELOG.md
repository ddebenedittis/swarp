# Changelog

All notable changes to `swarp`. Newest first. Nothing has been released yet — version is
`0.1.0` and the API is pre-1.0, so anything here may still change.

## Unreleased

### Breaking changes

- **`Environment.step` returns the Gymnasium 5-tuple** `(obs, reward, terminated,
  truncated, info)` instead of `(obs, reward, done, info)`. `terminated` is the scenario's
  own terminal condition; `truncated` is the `max_steps` time limit, previously OR-ed into
  the single `done` flag. Auto-reset and the step counter key off `terminated | truncated`,
  so episode boundaries are unchanged — only the reporting is. Migration is mechanical:
  `obs, rew, done, info = env.step(a)` becomes
  `obs, rew, term, trunc, info = env.step(a)`, with `done = term | trunc` where a single
  flag is what you want.

  `swarp.interop.torchrl.SwarpEnv` now emits `terminated`, `truncated`, and `done` (their
  OR), so TorchRL's value estimators bootstrap through a timeout instead of cutting the
  return at every truncation. `examples/pusht_torchrl.py` no longer has to reconstruct the
  task terminal from `info` to work around the old behaviour.

### Fixed

- **`Lidar.scan` read the stale installed obstacle pose.** A movable obstacle is
  integrated in place inside the stepper's own arrays, so `world.obstacle_pos` is its
  *spawn* pose; the scan now takes the live pose from `World.obstacle_state_views()`, as
  the renderer already did. Measured divergence before the fix: 1.41 world units after 20
  Push-T steps.
- **Segment and box obstacles were mis-modelled by the lidar, not ignored.** Passed
  through the ray-circle test they reported a phantom hit on a disc of the obstacle's
  `radius` centred at its origin — for a segment, its midpoint. `Lidar.scan` now filters
  on `ObstacleShape`, so they are genuinely invisible to the sensor (and still act in the
  collision step), which is what the docs always claimed.
- `swarp/render/geometry.py` no longer syncs the device to the host every frame to decide
  whether any obstacle is movable; it reads the memoized `Obstacles.any_movable`.

### Scenarios

- **Push-T** (`PushTScenario`): agents push a T-shaped **movable compound rigid body** to a
  target pose. Two oriented `BOX` shapes share one body id and are placed by body-frame
  offsets, so `swarp.core.bodies` integrates a single rigid pose (position *and* rotation)
  for the pair from the reaction of the very same agent contacts — inside the substep loop,
  so the pose an agent collides against is at most one substep old. Mass splits by area and
  inertia comes from the parallel-axis theorem about the area centroid, which makes the
  orientation half of the task solvable.
- **Movable obstacles** (`ObstacleKind.MOVABLE`): obstacles that carry mass and inertia and
  are advanced from agent-contact reaction forces (Newton's third law of one shared force)
  by `swarp/core/bodies.py`, gather-style with a thread per `(env, obstacle)` and no atomics.
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
  (`swarp.interop.persistent.CudaGraphStep`).
- A **batched uniform-grid neighbor backend** (`neighbor_method="uniform_grid"`,
  radix-sort based) that stays linear in `n_envs` and beats brute force past ~512
  agents/env (~4× at 1k, ~10× at 4k on an RTX 3070).
- **Static-origin uniform grid**: `NeighborGrid(..., bounds=(x_min, x_max, y_min, y_max))`
  pins the grid frame once instead of re-deriving it from the batch on every build,
  dropping the three-launch bounds pass (a global-atomic reduction over every position).
  `Stepper.grid()` passes `World.bounds`, which all seven built-in scenarios set, so they
  get it for free; `bounds=None` keeps the adaptive path. Measured build time -33% to -35%
  across `(E, A)` in `{(256, 1024), (64, 4096), (1024, 512)}` on an RTX 3070 Laptop. Exact,
  not approximate: the edge clamp is safe for any origin once `cell_size >= radius`, and
  same-cell false candidates were always rejected by the distance filter.

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
- **Lidar sensor** (`swarp.Lidar`): a differentiable, vectorized ray-cast returning per-ray
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

- **TorchRL `EnvBase` wrapper** (`swarp.interop.torchrl.SwarpEnv`, `--group torchrl`):
  batched `TensorDict` specs, passes TorchRL's `check_env_specs`.
- **`torch.compile`-compatible step** (`swarp.interop.compile.compiled_warp_step`): a
  `torch.library.custom_op` with fake-tensor and autograd rules.
- `CudaGraphStep` moved from `swarp.interop.compile` to `swarp.interop.persistent`, where
  this changelog already documented it and where it belongs — `compile.py` is otherwise
  entirely about the `torch.library.custom_op` path.

### Core

- **Box and segment/capsule static collision geometry**, boxes via an analytic signed
  distance field so a penetrating agent is still pushed out.
- **Host-sync-free masked and automatic reset** (`env.reset_at(mask)`, `auto_reset=True`) —
  no `.any()`/`.nonzero()` round-trip.
- **Arbitrary action arity**: the action tensor is `[n_envs, n_agents, act_dim]` with
  `act_dim` the max over agent models, decoupling the action space from the geometry.
- **Neighbor-list overflow is surfaced**, not silently truncated.
- **`Environment.close()`**: idempotent teardown that closes the render window and
  releases the persistent runtime's captured CUDA graph (via a new `StepRuntime.release()`)
  without destroying the env — stepping afterwards simply recaptures.
- **`swarp.scenarios.register_scenario(name, cls)`**: register an out-of-tree scenario so
  it reaches `swarp.make`, `make_scenario`, `fused_scenarios()` and the benchmark CLIs
  without editing the installed package. Re-exported as `swarp.register_scenario`.
- Four names promoted to the top level: `Obstacles` (needed by any custom-obstacle
  scenario, and the class the already-exported `ObstacleKind`/`ObstacleShape` annotate),
  `GradRing`, `drone_config`, `register_scenario`.
- **14 cross-module private names promoted** to the public surface they already were, with
  docstrings to match: the three contact primitives `collisions.pair_force` / `box_force` /
  `closest_on_segment` (what a custom contact model wants), the torch<->Warp bridge
  `autograd.torch_stream_scope` / `wrap_actions` / `wrap_input_state`, the render helpers
  `overlays.to_px` / `to_px_batch` / `radius_px` / `get_font` and
  `renderer.bounds_from_geometry` / `ensure_pygame`, and the benchmark helpers
  `ablation.parity_ok` / `sync_device` and `compare_vmas.make_vmas`.
- Cross-simulator throughput benchmark (`swarp.benchmark.compare_sims`: swarp vs VMAS vs
  JaxMARL vs CAMAR, one subprocess per simulator) and a VMAS head-to-head
  (`swarp.benchmark.compare_vmas`). See the [benchmarks](https://ddebenedittis.github.io/swarp/benchmarks.html) page.

### Repo

- MIT `LICENSE`, GitHub Actions CI (ruff + the CPU test suite on Python 3.12), a tracked
  `uv.lock`, and `docs/` split out of the README.

What is planned next lives in the docs' [Roadmap](https://ddebenedittis.github.io/swarp/architecture.html#roadmap) section.
