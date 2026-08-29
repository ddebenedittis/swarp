# Changelog

All notable changes to `swarp`. Newest first. Nothing has been released yet — version is
`0.1.0` and the API is pre-1.0, so anything here may still change.

## Unreleased

### Fixed

- **Give-way and shepherding now train. Both were unlearnable for reasons a green test suite
  could not see, and one of the two was not a tuning problem at all.**

  Before: give-way's `all_on_goal` was **0.0000 for an entire 2000-iteration run** and
  shepherding's `sheep_penned` crawled to 0.29 with `terminated` at 0.0016.

  After, on `marl_eval`'s held-out seed 123, both against a random baseline on the same seed:

  | scenario | metric | policy | random |
  |---|---|---|---|
  | `giveway` | episodes solved (all four on goal) | **1.000** | 0.000 |
  | | distance to goal, start -> end | 1.775 -> **0.256** | 1.775 -> 1.450 |
  | `shepherding` | episodes penning the whole flock | **1.000** | 0.002 |
  | | sheep penned, start -> end | 0.071 -> **0.990** | 0.071 -> 0.057 |
  | | whole flock penned, end | **0.955** | 0.002 |
  | | flock radius, start -> end | 0.189 -> **0.107** | 0.189 -> 0.127 |
  | | sheep distance to pen, start -> end | 0.539 -> **0.130** | 0.539 -> 0.547 |

  Both halves of shepherding work rather than one: the flock radius falls (collect) *and*
  the distance to the pen falls (drive). During training, once the curriculum reaches full
  difficulty, give-way holds a mean `episode_solve` of **0.9989** over 1745 iterations and
  shepherding **0.9929** over 2586, neither with a late collapse. These are two independent
  runs at different seeds from the ones the design work was done on.

  **Shepherding was geometrically impossible, not badly shaped.** The sheep's own
  separation/cohesion equilibrium settles at a maximum radius of **0.198** — against a
  `pen_radius` of **0.2**. Satisfying `all_penned` therefore meant placing the flock centroid
  within **3 mm** in a world of half-extent 1.0, which is exactly the measured `terminated` of
  0.0016. It is worse than that: the flee force is capped at 1.0 while cohesion is only
  `0.5 * R ≈ 0.1`, so *any* shepherd inside `flee_radius` inflates the flock past the pen
  **diameter** (three shepherds at radius 0.25 blow it out to 0.405). A scripted Strömbom
  collect/drive controller solves **0%** of episodes at the shipped parameters. No reward
  function can fix that. The geometry is now `pen_radius` 0.3, `sep_radius` 0.15 (which drops
  the free-flock equilibrium to 0.150), 200-step episodes, and `done` requires the flock to
  hold for 5 consecutive steps rather than pass through.

  The spawn law was separately degenerate. Drawing each sheep at an independent angle on a
  ring about the pen puts the flock *centroid on the pen* by symmetry, so once the pen was
  enlarged **doing nothing solved 23%** — shepherding had accidentally become a gathering
  task rather than a herding one. It now draws a flock centre at a distance from the pen and
  scatters the sheep about that; do-nothing and random both drop to 0.00.

  **Give-way's collision penalty was multiplying a structurally zero quantity.** `touching`
  counts pairs closer than `2 * agent_radius`, but the contact spring activates at
  `2r + margin` and `test_no_interpenetration_in_a_crowded_junction` asserts *exactly zero*
  overlap at the shipped stiffness — so the `-1.0` coefficient could never fire, and
  `collisions` logged 0.000000 for the whole run. What the policy actually felt was the
  **wall** penalty: `-0.1 x 0.64 x 300 = -19.2` per agent per episode against a total
  available shaping of 1.774, as a step function with no gradient, through a wall-free band
  one sixth of the corridor wide. The policy was being taxed for existing in a corridor, and
  it responded by learning to avoid the junction. Both penalties are now continuous ramps
  over the contact-activation band; the discrete count and the binary flag survive as
  `info()` metrics, where they are diagnostics rather than gradients.

  Give-way's observation is rebuilt around the fact that **a head-on pair's observations are
  an exact 180-degree rotation of each other**, so a shared-weight policy maps them to
  rotated actions and both robots accelerate or both retreat. Observations are now in each
  robot's own travel frame (which collapses the plus-shape's 4-fold symmetry the shared actor
  was learning four times over, and makes distance-to-junction and lateral offset fall out
  for free), plus the world-frame travel axis, corner-block clearance and analytic SDF normal,
  the current corridor half-width, each neighbour's travel direction, and a per-episode
  **priority token** drawn as a uniform random permutation. A permutation rather than i.i.d.
  draws because a near-tie *is* the deadlock case, so i.i.d. tokens put a ceiling on the solve
  rate that no amount of training removes. `obs_dim` 16 -> 40, and sensing is all-pairs: the
  physics neighbour grid has left the observation path, since its radius (0.1125) meant a
  robot could not see an opponent until they were almost touching.

  Shepherding's observation gains what the Strömbom heuristic actually needs — flock centroid
  and velocity, mean and max flock radius, the stray furthest from the centroid, and the
  drive and collect points — plus, for the first time, the other shepherds. Three shepherds
  that cannot see each other cannot form the arc the multi-shepherd literature is unanimous
  about. `obs_dim` 26 -> 38.

  Two reward-shape corrections worth recording. Shepherding paid `0.5 x penned_count` **every
  step**, worth up to 375 per episode against a telescoped shaping total of 2.5 — a factor of
  150, whose optimum is "park two sheep near the pen and stop", which is precisely the
  observed signature of `sheep_penned` rising while `sheep_dist_to_pen` stayed flat. It is now
  a potential on the count, bounded at 5 per episode, alongside a gather potential on the mean
  flock radius (the "collect" half of Strömbom, previously absent) and a genuine terminal
  bonus, which `done` had never carried. Give-way's `shared_reward: bool` becomes
  `shaping_share: float` defaulting to 0.0: the docstring's argument for sharing was that
  shaping points the yielder the wrong way, but shaping telescopes to `d_spawn - d_final`, so
  a detour into a bay is fully refunded and the real cost of yielding is ~0.017 of discounting.

- **The MAPPO harness was discarding 79% of its gradient updates.** `runs/giveway.log` ends
  with `skipped 50852` out of 64000, and still printed a plausible reward curve for 2000
  iterations — so the run read as a reward-design problem for its entire length. The cause was
  an unbounded policy head: nothing clamped `loc`, so it saturated the `TanhNormal`, where a
  boundary sample's log-prob diverges and its gradient vanishes. `_ClampScale` now bounds both
  `loc` and `scale`, and the skip ratio is a first-class `metrics.csv` column that prints a
  warning above 5%. Measured after: **0.000000 every iteration** on all four scenarios tried.

  Also in the trainer, each paid for by one of the two dead runs:

  - **Observation and value normalization** (`ObsNorm`, `ReturnScaler`). Observations mix
    metres, radians and unit flags in one row and per-step rewards span two orders of
    magnitude across the registry, under one shared `lr`. `ObsNorm` is the first layer of both
    actor and critic, so its statistics ride in `state_dict` and `marl_eval` cannot evaluate
    through a different transform than training used.
  - **A curriculum that can go down.** The monotone ratchet stranded give-way at difficulty
    0.472 for 1700 iterations and shepherding at 0.404, both reporting zero success
    throughout, because a policy that regressed after one lucky early batch had no way back.
  - **LR annealing and a KL cut-off on the epoch loop**, against the documented late collapse
    (a give-way run held ~1.0 from iteration 400 to 2400, then fell to zero by 2800). `lr`
    drops 1e-3 -> 3e-4.
  - **Action bounds read from `Environment.action_bounds`** rather than a hardcoded +/-1,
    which was correct only by coincidence of every task using `max_speed=1.0`.
  - `marl_eval` was **executing a different policy than it trained**: `_greedy` returned
    `loc.clamp(-1, 1)`, but the mode of a `TanhNormal` is `tanh(loc)`. Every eval number in
    this file below predates that fix.

  On the two scenarios that already worked, the new trainer is not a regression but a large
  improvement: caging reaches `caged` **0.266 and holds** after 400 iterations on 256 envs,
  against **0.0013** after 3000 iterations on 512 envs before.

- **A scripted-expert feasibility test now runs in CI for both scenarios**, and this is the
  process lesson rather than a code change. `test_a_scripted_shepherd_solves_most_episodes`
  asserts a Strömbom controller clears 60% *and* that a do-nothing policy solves exactly 0;
  give-way has the equivalent for a token-scheduled controller. Either would have reported, in
  thirty seconds, what two multi-thousand-iteration runs took hours to fail to say — and in
  shepherding's case would have caught a task that was impossible at any reward.

### Added

- **Three scenarios that are not solved by monotone progress toward a goal.** All seven
  existing scenarios reward closing a distance, so a greedy policy does well on every one of
  them and none of them tests credit assignment, deadlock, or a non-metric objective. The
  registry is now ten:

  - `giveway` — four robots cross a one-lane intersection formed by four immovable corner
    blocks. The corridor fits one robot, not two abreast, so the only feasible solution has a
    robot back into a perpendicular arm and *give way* — it has to move away from its goal.
    This is the repo's first negative-shaping task, and the first where deadlock is a real
    outcome rather than something the geometry rules out. `shared_reward` defaults to `True`
    here (navigation defaults it `False`): under per-agent shaping the yielding robot pays the
    whole cost of a manoeuvre only the team benefits from. Needs the stiff contact
    (`contact_k` 8000, `substeps >= 8`) — at the engine's default `collision_k=100` a
    velocity-mode agent settles at a penetration several times its own radius and the corridor
    walls stop being walls.
  - `caging` — surround a disc that drifts away from the agents' mean bearing, so a cage that
    is merely nearby does not hold it. The reward is topological rather than metric: the
    largest angular gap in the ring of agents seen from the disc. Warp has no dynamically
    sized register array, so the kernel computes the gap in the storage-free O(A^2) form
    (`max_i min_j (b_j - b_i) mod 2*pi`) rather than sorting; wrap-around is then structural,
    and one and two agents need no special case.
  - `shepherding` — drive sheep into a pen. The sheep are not agents and not passive cargo:
    they run their own flee policy, so this is the first scenario whose environment pushes
    back, and a shepherd that charges straight in scatters the flock.

  All three needed a physics or reward correction that only *training* surfaced, which is
  worth recording because in each case the test suite was perfectly green either way:

  - `giveway` inherited Push-T's `contact_k=8000`, at which a velocity-mode robot settles
    **0.033 deep** into a corner block — two thirds of its own radius. Blocks that soft are
    not one-lane geometry, and the symptom was that a greedy "drive straight at the goal"
    policy solved **100%** of episodes by squeezing through robots and walls. At
    `contact_k=50000` a robot stops short of the wall and the same greedy policy solves
    **0%**. Pinned by `test_the_corridor_is_solid_and_one_lane`. Note penetration gets
    *worse* with more substeps (the settling depth is `max_speed / (k * sub_dt)`), so
    holding force has to come from `k`.
  - `shepherding`'s per-shepherd flee forces summed without a bound, so three converging
    shepherds accelerated a sheep to a measured **2.30** against their own top speed of
    1.0 — the flock could never be cornered, and MAPPO plateaued at 7% penned. The summed
    flee force is now capped, and sheep carry a top speed of `0.75 * max_speed`. Pinned by
    `test_sheep_cannot_outrun_the_shepherds`.
  - `caging`'s reward was `max_gap` alone, which is a *max*: only the two agents bordering
    the widest gap get any gradient, so the policy optimized the easy radial term and
    clumped. Training moved steadily *away* from a solve. A dense per-agent gap-variance
    term now shapes toward the evenly spaced ring, which is the same configuration that
    minimizes `max_gap`; `spacing_factor=0.0` recovers the old reward exactly.

  Caging and shepherding own and integrate their bodies the way `transport` does, rather than
  tagging them `ObstacleKind.MOVABLE`: the engine integrates a movable obstacle from contact
  reaction alone, with no hook for an escape or flee force, it is skipped entirely on a taped
  step, and an install carrying `kind=MOVABLE` reads `any_movable` back to the host, which is
  not capture-safe. Sheep in particular are scenario-owned bodies rather than agents with
  overridden actions, because nothing in the engine can intercept an action — `world.action`
  is written after the only pre-physics hook fires.

  Each is verified learnable by an actual MAPPO run, scored against a random policy on the
  **same seed** (512 envs, `examples/marl_eval.py`). "solved" is the fraction of episodes that
  ever reach the scenario's own termination condition.

  :::{note}
  These numbers were measured *before* give-way's contact was hardened
  (`contact_k` 50k -> 200k, `substeps` 8 -> 16) and before `shepherding`'s default flock grew
  from 2 sheep to 5. Both change the task, so both numbers will move — give-way's walls are
  now genuinely solid, and shepherding with five sheep is harder than with two. They are kept
  as the record of the runs that found the physics defects below; re-measure before quoting.
  :::

  | scenario | metric | policy | random |
  |---|---|---|---|
  | `caging` | episodes that close the cage | **0.994** | 0.125 |
  | | max angular gap, start -> end | 2.81 -> **1.92** | 2.81 -> 5.30 |
  | `shepherding` | sheep penned at episode end | **0.508** | 0.105 |
  | | episodes penning the whole flock | **0.133** | 0.033 |
  | `giveway` | distance to goal, start -> end | 1.77 -> **1.06** | 1.77 -> 1.45 |

  Caging is solved and shepherding looks learned. Give-way is only *partially* learned: the
  policy closes distance more than twice as fast as random but rarely completes the crossing.

  **All three readings above are now known to be wrong**, and the "Fixed" section at the top
  of this file has the corrected ones. The give-way conclusion blamed the task ("the one task
  whose solution requires moving away from the goal") for what was an unbounded policy head
  discarding 79% of its updates; shepherding's 0.508 was measured on a task whose terminal
  condition was geometrically unreachable; and every number in this table was produced by an
  eval that executed `loc.clamp(-1, 1)` where the policy's actual mode is `tanh(loc)`. The
  table is left in place as the record of what the runs reported at the time.

- **`examples/marl_train.py` / `examples/marl_eval.py` — MAPPO for any scenario.**
  `--scenario <name>` off the registry; everything task-specific is one `Task` record
  (episode length, substeps, scenario kwargs, which `info()` key is the headline success
  metric, an optional gated curriculum). Logged metrics are discovered from the scenario's
  own `info()` rather than hardcoded per scenario, so they cannot go stale. The eval script
  scores a checkpoint against a random baseline **on the same seed** and can render the
  rollout to a window or a video file. `examples/pusht_torchrl.py` and `pusht_eval.py` are
  unchanged — they remain the worked, heavily annotated Push-T recipe.

  Three defaults in it are load-bearing and were each paid for by a wasted run:

  - **`normalize_advantage=True`.** Per-step rewards across the registry span two orders of
    magnitude (shepherding ~0.06, caging ~7.5). Unnormalized, caging drove **~85% of
    minibatches to a non-finite gradient** — the run skipped almost every update and looked
    like a reward-design problem for a long time. With normalization its max angular gap
    went 3.06 -> 1.50 in 300 iterations.
  - **`policy_best.pt`, gated on full curriculum difficulty.** PPO collapses late and does
    not recover: give-way held an episode solve rate near 1.0 from iteration 400 to 2400,
    then fell to zero by 2800 and stayed there. Interval checkpointing saved the wreckage,
    and that policy evaluated *worse than random*. The gate matters too — success peaks
    while the curriculum is still easy, so an ungated tracker crowns an easy-stage policy
    that solves nothing on the real task.
  - **The curriculum gates on the fraction of finished episodes that ended in success**, not
    on `terminated.mean()`. The latter is a per-*step* hazard rate: a task that always solves
    on step 40 of a 300-step budget reports 0.025, which reads as a 2.5% success rate and is
    really 100%. Give-way's curriculum consequently never left its easiest setting for an
    entire 4000-iteration run.

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
