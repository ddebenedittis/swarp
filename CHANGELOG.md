# Changelog

All notable changes to `swarp`. Newest first. Nothing has been released yet — version is
`0.1.0` and the API is pre-1.0, so anything here may still change.

## Unreleased

### Performance

- **Kernel overloads are resolved once, not per launch.** Every generic kernel was already
  instantiated per dtype at import, but the registration loops discarded what
  `wp.overload` returns and launched the *generic* kernel, so `wp.launch` re-inferred the
  argument types and rebuilt a signature string on every call. `swarp._overloads` keeps the
  concrete kernel and dispatches it for the cost of a dict lookup. Isolated: 492 -> 173 us
  for the 2D integrator at 4096x16. End to end (`swarp.benchmark.throughput`,
  `use_graph=False`): ~1.7-1.8x, e.g. 16000x16 from 5.1M to 8.9M env-steps/s. Outputs are
  bit-identical — only which kernel object reaches `wp.launch` changes.

- **The obstacle install caches the Warp view of each source tensor.** `Stepper._install`
  rebuilt a `wp.from_torch` wrapper per field per call, which a scenario re-sampling
  obstacle poses pays on every step under `auto_reset` — 22 wraps per reset for Push-T.
  The sources are updated in place (that is what keeps the install allocation-free and
  capture-legal), so the wrapper stays valid; it is now kept, keyed by field and validated
  against the tensor's data pointer, shape and dtype so the grad path's `_refresh`
  rebuilds instead of writing through a stale view. At 8192x8, graph on, `auto_reset=True`:
  Push-T 4.477 -> 4.138 ms (-7.6%), transport 1.586 -> 1.422 ms (-10.3%). Navigation, which
  installs no obstacles, is unchanged (-0.2%, run-to-run noise), and
  `swarp.benchmark.throughput` is flat.

  What remains of Push-T's reset is the 16 `wp.copy`s themselves plus `_body_seed` and its
  box-pose derivation. Cutting those means installing only the fields that changed, which
  contradicts `set_obstacles`' documented contract that an absent field always means the
  default rather than "keep the previous value" — so it is left alone.

- **Every scenario's masked reset is now a single Warp launch.** Under `auto_reset` the
  reset runs on every step for the whole batch, and it was 83-96% of the step time for all
  seven scenarios. Each now does its whole draw in one masked kernel, one thread per env
  (`swarp/scenarios/reset_kernels.py` holds the shared piece and documents the RNG trap
  that shape depends on). At 8192 envs x 8 agents, graph on, `auto_reset=True`:

      scenario     before      after
      flocking    1.190 ms   0.849 ms   1.40x
      formation   1.321 ms   0.831 ms   1.59x
      discovery   1.404 ms   1.017 ms   1.38x
      sampling    1.427 ms   1.079 ms   1.32x
      transport   2.437 ms   1.512 ms   1.61x
      pusht       6.053 ms   4.482 ms   1.35x

  Pusht gains least because its reset is no longer dominated by the draw: 61% of what
  remains is `_install_obstacles` re-installing the retained obstacle spec (16 `write`
  calls, each a fresh `wp.from_torch` plus a `wp.copy`), which is untouched here.

  `FusedScenario.reset_mask_wp` owns the uint8 mask buffer and its pointer-stable Warp
  handle, so that plumbing exists once rather than seven times.

- **Navigation's masked reset is one Warp launch instead of ~25 torch ops.** Under
  `auto_reset` the reset runs on every step for the whole batch (there is no host-side "is
  anything done?" gate, by design), and it was **86%** of the step time at 16,384x16 —
  two batched `argsort`s, four `sample_uniform`s and the `torch.where` blends, on a path
  bound by op count rather than arithmetic. `nav_reset_kernel` writes spawns, goals,
  headings and zeroed velocities in one masked launch, one thread per env, drawing its
  distinct cells with a partial Fisher-Yates over a scratch permutation. The RL
  configuration (graph on, `auto_reset=True`) went from 4.2M to 13.8M env-steps/s at
  16,384x16; the two changes together are ~3.3x there.

  The draw is still a *uniform* random k-subset of cells, matching the torch reference's
  distribution. That is deliberate and tested: a cheaper structured draw (cells by a
  random base and coprime stride) satisfies every separation and bounds check while
  collapsing the reachable spawn layouts from C(25,16) = 2,042,975 to 250 — a loss that
  shows up as a generalization failure long after it would show up in a test.

  `NavigationScenario._sample_separated` is retained as the torch reference and keeps its
  own tests; it and the kernel share no code, like the fused obs/reward kernels and their
  oracles.

### Changed

- **Seeded trajectories differ from previous versions.** No scenario's reset draws from
  `world.generator` any more, so every downstream torch draw sits at a different point in
  that stream. Reproducibility is unchanged going forward — same `seed` in, same
  trajectory out — and `World.next_kernel_seed` gives the kernel RNG its own deterministic
  host-side stream, reset alongside the generator, with no device->host round-trip.

### Breaking changes

- **`SamplingScenario(collision_penalty=...)` is gone.** The parameter was stored and never
  read — the reward never had a collision term. Agents still collide physically
  (`WorldConfig(collisions=True)`); nothing about the dynamics or the reward changes, only
  the constructor signature.
- **`SwarpEnv` rejects `Environment(auto_reset=True)`.** TorchRL resets from the done flags
  itself, and swarp's auto-reset returns the *next* episode's first observation alongside a
  `True` done — so every boundary transition a collector stored paired a reward with an
  observation from a different episode. Build the env with `auto_reset=False` (the default).
- **The Warp lidar backend returns a view of a reused buffer**, where the torch backend
  still allocates fresh output. Concatenating into an observation copies; `clone()` to keep
  it.
- **`swarp.benchmark.scenarios.SCENARIO_FACTORIES` is gone.** It was an alias for
  `swarp.scenarios.SCENARIOS`, which is the registry's home.
- **`swarp.make` now raises `TypeError` for a keyword neither `Environment` nor the chosen
  scenario accepts**, listing the scenario's keywords and suggesting the `Environment` one
  it looks like a typo of. Previously a misspelled `Environment` keyword was routed silently
  to the scenario and reported (much later) against the wrong constructor.
- **`Environment` raises for a `cuda` device on a CPU-only install** instead of failing
  deep inside Warp device resolution.
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

- **A body-body contact read the other obstacle's *angle* where it wanted its angular
  velocity** (`swarp/core/bodies.py`). The damper's closing velocity was therefore computed
  against a spurious surface motion, and the same physical wall pushed a movable body
  differently depending on which of the two equivalent segment angles (`+pi/2` vs `-pi/2`) a
  scenario happened to install — 1e-2 world units and 3e-2 rad apart after 120 steps. Both
  bodies' spins are now parameters, and the segment branches fold the other body's spin into
  the closest-point surface velocity the way `collisions._static_forces` already did on the
  agent side.
- **navigation, formation and discovery skipped their fused reward launch on every reset**,
  not just the obs-only auto-reset it was meant for. Their reward kernel is the only writer
  of the fused `done`, so the first `done()`/`rewards()`/`info()` after a standalone
  `reset()` or `reset_at()` still described the *previous* episode, while the torch
  reference recomputed them. Now gated on `full_pass`. No rollout-only parity test could
  see this: those cross auto-resets only, which is the one case the old gate got right.
- **Push-T's torch oracle divided the contact coefficient by `tee_mass`.** The implicit
  damping solve belongs to the body the impulse is applied to — the agent — and the engine
  reuses that one number as the reaction on the T (`bodies._reaction`). The oracle now
  carries the agent masses, so the parity oracle and the whole grad path agree with the
  engine for any T that is not unit mass. Bit-identical at the defaults, where both are 1.0.
- **`Lidar.scan` re-decided its circle filter with two device→host syncs on every scan**
  (`is_circle.all()` and `nonzero()`). Shape tags are static per obstacle install — poses
  move, shapes do not — so the filter is memoized and the per-scan work is at most one
  `index_select`.
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
- **`discovery`/`sampling` `info()` reported float32 in a float64 world** — the fraction was
  built with `.float()` rather than the world dtype.

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

- **Ray-parallel lidar kernel**: the Warp backend takes the ray as a third launch dimension
  instead of looping it, and caches its input wraps (as `Stepper.wrap_actions` does) and its
  output buffer. 683 → 455 us/scan at 24×4×256 and 1307 → 791 at 1024×16×128 on an RTX 3070
  Laptop.
- **Eager launch sites are scoped onto torch's current stream** (the graph replay, the eager
  persistent step, the pre-capture warm-up, the state-loading paths, `World.neighbors`, the
  fused obs/reward launches, the lidar scan), so a caller running inside its own
  `torch.cuda.Stream` is ordered correctly rather than relying on Warp's stream carrying the
  blocking flag. On the default stream — where the driver does guarantee that ordering — the
  scope degrades to a `ScopedDevice`, because opening a real one cost 10% of the graph replay
  (0.166 → 0.185 ms/step at 4000×16).
- Per-step allocations removed from the fused scenarios: one `state_wp()` wrap per pass
  instead of one per launch (discovery, sampling, transport, pusht), and sampling's 3×3
  stencil and pusht's teammate index built once in `make_world` instead of per call. Worth
  ~4-6% of the eager (capture-off) fused step, which is host-launch-bound; invisible under
  CUDA-graph capture, where those launches are inside the graph.
- `set_agent_params_per_env` no longer reads the max radius back to the host on the in-place
  refresh path, so per-reset domain randomization inside a graph-mode loop does not stall.
- `swarp.interop.compile`'s stepper registry is weak, so a compiled stepper (and through it
  a whole world) is no longer immortal for the life of the process, and the handle lookup is
  O(1) rather than a scan.
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

- A second CI job installs the `torchrl` extra and runs `tests/interop/test_torchrl.py`,
  which `importorskip`ed on every runner until now — the wrapper was effectively untested
  in CI.
- **Benchmark attribution corrected.** The 24.1 M env-steps/s headline is the fused kernels
  with capture *off* (`use_graph=False`, which `throughput.py` pins so the table stays
  comparable across commits), not "fused kernels plus capture" as the README, `docs/index`
  and `docs/performance` claimed; the graph-on figure (35.6 M at 16,384 × 16) is now quoted
  separately. The CAMAR comparison discloses that its 1.2×1.2 arena is not swarp's 2×2–4×4
  rather than claiming they match, and its `frameskip` is documented as the 0 the adapter
  actually passes.
- `pyproject.toml` grew `[project.urls]` and trove classifiers, and dropped the `slow`
  pytest marker nothing used.
- MIT `LICENSE`, GitHub Actions CI (ruff + the CPU test suite on Python 3.12), a tracked
  `uv.lock`, and `docs/` split out of the README.

What is planned next lives in the docs' [Roadmap](https://ddebenedittis.github.io/swarp/architecture.html#roadmap) section.
