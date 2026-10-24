# Scenarios

A scenario defines the task: what the world is made of, how it resets, and what the agents observe and are rewarded for.
The engine knows nothing about tasks — every built-in scenario is written against the same public `Scenario` ABC you would use for your own.

## The twelve built-ins

| name | task | agents | `obs_dim` | notable |
|---|---|---|---|---|
| `navigation` | reach a per-agent goal while avoiding collisions | 4 | 19 | optional obstacles; `shared_reward` switches per-agent shaping to a team term |
| `flocking` | Reynolds-style cohesion + alignment, penalized for crowding | 8 | 24 | pure neighbor-feature reward, no goals |
| `formation` | hold assigned slots of a regular polygon | 5 | 6 | slot `i` is assigned to agent `i`, so the shape is ordered |
| `discovery` | cover scattered targets, each needing several agents nearby | 5 | 19 | one-off shared reward the step a target is first covered, plus optional per-agent closing speed on the nearest uncovered target (off by default) |
| `sampling` | collect an unknown scalar field, consuming cells | 4 | 13 | batched sum-of-Gaussians density on a grid; obs is the 3×3 cell neighborhood |
| `transport` | push a movable circular package to a goal | 4 | 8 | package integrated in torch, staggered by one step, so BPTT flows package→agent→action |
| `pusht` | push a T-shaped rigid body to a target **pose** | 4 | 18 | movable compound body with rotation; needs `substeps >= 8`; optional cost on heading-error growth, so extra turns are not free (off by default); `shape=` swaps the T for another compound from `swarp.scenarios.shapes` (`ell`, `plus`, `bar`) |
| `giveway` | cross a one-lane intersection without deadlocking | 4 | 25 | the only task whose solution requires moving *away* from your goal; a per-episode `politeness` scalar breaks the mirror symmetry that would otherwise force deadlock |
| `shepherding` | drive non-cooperative sheep into a pen | 3 dogs + 5 sheep | 34 | the environment pushes back: sheep flee the dogs and cohere, so a dog must herd rather than chase |
| `caging` | surround an evasive disc so it cannot escape | 5 | 23 | a *topological* objective — the maximum angular gap between agents, not any distance |
| `ringroad` | damp stop-and-go waves on a single-lane ring of IDM drivers | 1 car + 21 IDM humans | 9 | kinematic-bicycle RL cars among scenario-owned IDM traffic; the wave only exists on a ring of about 11 or more cars at this density |
| `pusht_lidar` | push the T to a target pose, observed only through each robot's lidar | 2 pushers + 3 observers | 117 | per-episode random roles with fixed counts; `obs_level` 0 adds the oracle T pose, 2 adds teammates' pooled object hits; the only scenario with the trainer's `critic_state`/`agent_roles`/`aux_targets` hooks ([Lidar Push-T](pusht-lidar.md)) |

Defaults shown; every scenario takes constructor keywords (`n_agents`, `world_size`, shaping factors, penalties, tolerances).
`navigation`, `flocking` and `giveway` size their observation from `neighbor_obs`, so their `obs_dim` moves with it;
`caging` and `shepherding` size theirs from the team (and, for `shepherding`, the flock), so theirs moves with `n_agents`/`n_sheep`.
`ringroad`'s is fixed at 9 for any number of cars, so one actor can be evaluated on any ring size; it takes `n_vehicles` and `ring_length` in metres instead of `world_size`.

The last three are newer and break assumptions the first seven share.
Every one of the original seven rewards monotone progress toward a goal, so a greedy "reduce the distance" controller does well on all of them;
`giveway` cannot be solved that way (the greedy policy deadlocks nose-to-nose in the corridor), `caging` scores an angular property rather than a distance, and `shepherding` is the only one whose environment reacts to what the agents do.
`ringroad` has no goal at all, and its difficulty comes from the number of vehicles, as the next section explains.
`pusht_lidar` is Push-T without the object pose: the team must infer the T from its lidars, and the observers are rewarded for keeping it in view rather than for moving it.

## Ring road: a task that needs many vehicles

`ringroad` is the Flow / Sugiyama ring: 22 cars on a 230 m single-lane ring, one of them an RL car (a kinematic bicycle with actions `(accel, steer)`) and the rest human drivers following the Intelligent Driver Model with Flow's parameters (v0 = 30 m/s, T = 1 s, a = 1, b = 1.5, δ = 4, s0 = 2 m, 5 m cars).
At this density the uniform flow of IDM drivers is string-unstable, and any perturbation grows into a stop-and-go wave.
The RL car's job is to damp it.
`done` fires once the speed spread of all cars stays below 0.3 m/s, with the mean speed above half the equilibrium speed, for 10 s.

The instability is long-wavelength, and an `N`-car ring only supports wavenumbers `2πm/N`, so whether a wave exists depends on `N` even at a fixed density.
Linearising the ring about uniform flow at 230/22 m per car, and simulating it from a ±3 m, ±3 m/s reset perturbation:

| cars `N` | ring radius (m) | linear growth rate of the slowest mode (1/s) | humans-only steady speed std (m/s) |
|---|---|---|---|
| 4 | 6.7 | −0.306 | 0.000 |
| 6 | 10.0 | −0.103 | 0.000 (wave gone in 34 s) |
| 8 | 13.3 | −0.028 | 0.000 (wave gone in 112 s) |
| 11 | 18.3 | +0.011 | 1.92 |
| 14 | 23.3 | +0.022 | 2.28 |
| 22 | 36.6 | +0.024 | 2.92 |
| 30 | 49.9 | +0.024 | 3.24 |

So a policy trained on a 6-car ring at the same density never sees a wave to damp.
`ring_length` defaults to `n_vehicles × 230/22`, so `--scen-kwarg n_vehicles=6` is that experiment.
The humans-only reference is `BASELINES["ringroad"]`, the RL car driven as one more IDM human: it solves the 6-car ring and never the 22-car one.

Two departures from Flow are deliberate.
The RL car may accelerate and brake at 2 m/s², not 1: Flow's car never crashes because SUMO's safe-speed mode overrides it, and here nothing does, so a car limited to 1 m/s² that drives the IDM rear-ends its leader in the wave.
The reward keeps Flow's form (mean speed of all cars minus the RL car's |acceleration|) with a smaller acceleration weight, plus a crash penalty, a lane-keeping term and a constant per-step cost that makes `done` the best outcome; the scenario's module docstring has the measurements behind each coefficient.

### Results: does a small ring teach wave damping?

<video src="_static/ringroad_transfer.webm" autoplay loop muted playsinline width="100%"></video>

The same 22-car ring from the same start, 150 s at 5×: the humans alone (left) keep the wave, the RL car trained on 22 cars (middle) spreads the traffic out evenly, and the one trained on 6 cars (right) leaves the wave in place and drives into the car ahead.

Two `python -m swarp.rl` runs, identical except for the ring.
Run a trained on the 22-car ring (`SPECS["ringroad"]`).
Run c trained on a 6-car ring at the same density (`--scen-kwarg n_vehicles=6`).
Both ran 4000 iterations at `lr 1e-3` (1024 envs × 32 steps per iteration, 94 ms each on an RTX 3070 Laptop GPU).
Each policy below is that run's iteration-1500 checkpoint, 2.5 minutes in; both greedy-solve their own training ring there.
Every cell is the greedy policy over one 1500-step episode of 1024 envs at full difficulty, paired by seed with the humans-only arm (the RL car driven as one more IDM human):

| ring size `N` | humans only | trained at N=22 | trained at N=6 | N=6 policy, crashed fraction at episode end |
|---|---|---|---|---|
| 6 | 1.000 | 1.000 | 1.000 | 0.000 |
| 8 | 1.000 | 0.953 | 1.000 | 0.000 |
| 11 | 0.000 | 1.000 | 1.000 | 0.000 |
| 14 | 0.000 | 1.000 | 0.000 | 0.000 |
| 18 | 0.000 | 1.000 | 0.000 | 0.307 |
| 22 | 0.000 | 1.000 | 0.000 | 0.328 |
| 30 | 0.000 | 1.000 | 0.000 | 0.430 |

The policy trained on the 22-car ring dissipates the wave on every ring from 6 to 30 cars, leaving a speed spread of 0.03 m/s at N=22 where the humans alone sustain 2.9.
The one trained on 6 cars handles every ring where the humans manage too, plus N=11 where the wave is barely unstable.
From N=14 up it fails, and from N=18 up it rear-ends its leader in the wave, braking it has never needed.

Steering is a possible confound: the small ring's curvature is 3.7× the large one's.
At N=22 the 6-car policy's mean lateral offset is 0.67 m against 0.15 m for the 22-car policy, though it stays inside the 1.85 m half-lane.
To separate the two, the table was re-run with each policy's acceleration and the scripted lane controller's steering.
The 6-car policy's longitudinal behaviour still fails at every N ≥ 14 and still crashes at N ≥ 18 (27–47% of envs), while the 22-car one still solves up to N=22.
The failure is in handling the wave, not in steering.

Training is not stable: the greedy policy flips between solving every env and solving none across consecutive checkpoints, with parking (the RL car stops and the ring jams) as the failure mode.
That is why the recipe checkpoints every 250 iterations and selects by greedy eval; `TrainSpec` documents the measurements.
The multi-AV variant (`--n-agents 4`, four RL cars among 22) does not train yet with this recipe.
In two runs (`lr 1e-3` for 4000 iterations, `lr 3e-4` for 3000) no checkpoint's greedy policy solved a single eval env.
The stochastic policy reached 0.99 once, at iteration 2000 of the first run, and both runs drift into the parked, jammed ring.

```{image} _static/flocking.png
:alt: Flocking — 24 agents with the within-radius neighbor graph drawn
:width: 75%
:align: center
```

## The registry

```python
import swarp

swarp.SCENARIOS                       # {'navigation': NavigationScenario, ...}
env = swarp.make("flocking", n_envs=1024, n_agents=16, device="cuda:0")

scenario = swarp.make_scenario("pusht", n_agents=6)   # scenario only, no Environment
cls = swarp.scenario_class("pusht")                   # the class itself
```

`SCENARIOS` is the one registry; `make` raises a `ValueError` listing the valid names for anything else.
`make_scenario` drops a `model=` keyword for holonomic-only scenarios, so a sweep over dynamics models can pass it uniformly.
`navigation` takes every planar model; `pusht` and `pusht_lidar` take holonomic and diff-drive robots (their `supported_models`), plus `agent_shape="box"`.

## Two execution paths

Every built-in scenario implements its observations and rewards **twice**:

1. **torch** — plain tensor ops over `world.state`, differentiable together with the Warp step and used whenever grads are enabled.
2. **fused Warp kernels** — a `*_kernels.py` module per scenario, used on the no-grad hot path and foldable into the whole-step CUDA graph.

The fused path is worth 4–11× (see [Benchmarks](benchmarks.md)), and `Environment(fused="auto")` picks it automatically when grads are off.

The torch implementation is the **parity oracle** the fused kernels are tested against, which is why the two are kept independent rather than sharing code — shared helpers would let a bug cancel itself out on both sides.

## Reward structure

`Scenario` separates the two reward terms:

- `agent_reward(i)` — the per-agent term, `[n_envs]`.
- `global_reward()` — a shared team term, `[n_envs]`, added to every agent.

`rewards()` combines them into the `[n_envs, n_agents]` tensor `Environment.step` returns.
Keeping them apart is what makes a scenario like navigation switchable between individual and shared credit with one flag, and it documents which part of the signal is a team objective.

## Writing your own

Subclass `Scenario`, implement four members, and you have a task that is differentiable end-to-end and runs on the batch:

```python
class MyScenario(Scenario):
    obs_dim = 4

    def make_world(self, n_envs, device, dt, substeps, dtype, world_config=None) -> World: ...
    def reset_world(self, env_mask=None, *, obs_only=False): ...
    def observations(self): ...
    def agent_reward(self, agent_idx): ...
```

The full contract — including the optional members, the `FusedScenario` tier, and the capture-safety rules a fused kernel must respect — is in [Writing a scenario](writing-a-scenario.md).
