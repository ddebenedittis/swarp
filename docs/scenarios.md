# Scenarios

A scenario defines the task: what the world is made of, how it resets, and what the agents observe and are rewarded for.
The engine knows nothing about tasks — every built-in scenario is written against the same public `Scenario` ABC you would use for your own.

## The ten built-ins

| name | task | agents | `obs_dim` | notable |
|---|---|---|---|---|
| `navigation` | reach a per-agent goal while avoiding collisions | 4 | 19 | optional obstacles; `shared_reward` switches per-agent shaping to a team term |
| `flocking` | Reynolds-style cohesion + alignment, penalized for crowding | 8 | 24 | pure neighbor-feature reward, no goals |
| `formation` | hold assigned slots of a regular polygon | 5 | 6 | slot `i` is assigned to agent `i`, so the shape is ordered |
| `discovery` | cover scattered targets, each needing several agents nearby | 5 | 19 | one-off shared reward the step a target is first covered |
| `sampling` | collect an unknown scalar field, consuming cells | 4 | 13 | batched sum-of-Gaussians density on a grid; obs is the 3×3 cell neighborhood |
| `transport` | push a movable circular package to a goal | 4 | 8 | package integrated in torch, staggered by one step, so BPTT flows package→agent→action |
| `pusht` | push a T-shaped rigid body to a target **pose** | 4 | 18 | movable compound body with rotation; needs `substeps >= 8` |
| `giveway` | cross a one-lane intersection without deadlocking | 4 | 16 | the only **non-monotone** task: solving it needs an agent to move *away* from its goal. Rigid contact (`contact_k=200000`, **`substeps >= 16`**) — softer and the walls stop being walls |
| `caging` | surround a drifting disc so it cannot escape | 4 | 10 | reward is the largest **angular gap** around the disc, not a distance, plus a dense per-agent even-spacing term |
| `shepherding` | drive fleeing sheep into a pen | 3 | 26 | 5 sheep, each running its own flee policy, so the environment pushes back |

Defaults shown; every scenario takes constructor keywords (`n_agents`, `world_size`, shaping factors, penalties, tolerances).
`navigation`, `flocking` and `giveway` size their observation from `neighbor_obs`, so their `obs_dim` moves with it;
`shepherding`'s moves with `n_sheep`.

```{image} _static/flocking.png
:alt: Flocking — 24 agents with the within-radius neighbor graph drawn
:width: 75%
:align: center
```

## The three that are not monotone

The first seven scenarios are all solved by closing a distance, which means a greedy policy that
always reduces its own distance-to-goal does well on every one of them.
`giveway`, `caging` and `shepherding` each break that in a different way, and each one needed a
physics correction that only *training* exposed — the test suite was green either way.

<video src="_static/giveway.webm" autoplay loop muted playsinline width="100%"
       title="Give-way — four robots deadlocked at the junction of a plus-shaped one-lane corridor"></video>

Give-way is the picture of the problem: four robots, each headed for the arm opposite its own,
jammed at a junction one robot wide.
Getting out requires one of them to *reverse into a side arm* and let another past, which is the
one thing distance shaping actively punishes.
The contact has to be rigid for any of that to be true — at Push-T's `contact_k=8000` a robot
sinks a third of the way into a corner block, the corridor stops being one lane, and a greedy
policy solves 100% of episodes by squeezing through walls.
Stiffness alone is not enough either: a velocity-mode agent covers `max_speed * sub_dt` in the
substep before contact can answer, which at `substeps=8` is half a radius and is why this
scenario asks for **16**. At 16 the overlap measured over a crowded jam is exactly zero.

<video src="_static/caging.webm" autoplay loop muted playsinline width="100%"
       title="Caging — six agents closing a ring around a drifting disc, one gap still open"></video>

Caging scores the *largest angular gap* in the ring of agents seen from the disc, so the objective
is a shape rather than a distance, and the disc drifts out through whatever gap is left open.
Note the gap still open at the upper right of the shot: that is the whole reward.

<video src="_static/shepherding.webm" autoplay loop muted playsinline width="100%"
       title="Shepherding — three shepherds driving fleeing sheep toward the pen marker"></video>

Shepherding is the only scenario whose environment pushes back.
The sheep are not agents and not cargo — they run their own flee policy, so a shepherd that
charges straight in scatters the flock, and the sheep have to be slower than the shepherds or the
task is quietly impossible.

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

## Two execution paths

Every built-in scenario implements its observations and rewards **twice**:

1. **torch** — plain tensor ops over `world.state`, differentiable together with the Warp step and used whenever grads are enabled.
2. **fused Warp kernels** — a `*_kernels.py` module per scenario, used on the no-grad hot path and foldable into the whole-step CUDA graph.

The fused path is worth 2.5–5× (see [Benchmarks](benchmarks.md)), and `Environment(fused="auto")` picks it automatically when grads are off.

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
