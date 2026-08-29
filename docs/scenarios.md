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
| `giveway` | cross a one-lane intersection without deadlocking | 4 | 40 | the only **non-monotone** task: solving it needs an agent to move *away* from its goal. Observations are in each robot's own travel frame and carry a per-episode priority token. Rigid contact (`contact_k=200000`, **`substeps >= 16`**) — softer and the walls stop being walls |
| `caging` | surround a drifting disc so it cannot escape | 4 | 10 | reward is the largest **angular gap** around the disc, not a distance, plus a dense per-agent even-spacing term |
| `shepherding` | drive fleeing sheep into a pen | 3 | 38 | 5 sheep, each running its own flee policy, so the environment pushes back; obs carries the Strömbom collect/drive points |

Defaults shown; every scenario takes constructor keywords (`n_agents`, `world_size`, shaping factors, penalties, tolerances).
`navigation`, `flocking` and `giveway` size their observation from `neighbor_obs`, so their `obs_dim` moves with it (give-way defaults it to `n_agents - 1`, since it senses all-pairs rather than off the physics neighbor grid);
`shepherding`'s moves with both `n_sheep` and `n_agents` (`16 + 4*n_agents + 2*n_sheep`).

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
       title="Give-way — four robots taking turns through the junction of a plus-shaped one-lane corridor"></video>

Give-way is the picture of the problem: four robots, each headed for the arm opposite its own,
jammed at a junction one robot wide.
Getting out requires one of them to *pull aside into a side arm* and let another past, which is the
one thing distance shaping actively punishes.
(The trained policy does exactly that, and does it forward: 57% of robots swing more than two radii
off their lane centreline — physically into a perpendicular arm — while only 3% ever reverse.)
The contact has to be rigid for any of that to be true — at Push-T's `contact_k=8000` a robot
sinks a third of the way into a corner block, the corridor stops being one lane, and a greedy
policy solves 100% of episodes by squeezing through walls.
Stiffness alone is not enough either: a velocity-mode agent covers `max_speed * sub_dt` in the
substep before contact can answer, which at `substeps=8` is half a radius and is why this
scenario asks for **16**. At 16 the overlap measured over a crowded jam is exactly zero.

Rigid contact is also what made the *first* version of this task unlearnable, in a way worth
recording because the failure was invisible in the metrics.
Zero interpenetration means the old collision penalty — a count of pairs closer than `2r` —
multiplied a structurally-zero quantity, so `collisions` logged `0.000000` for a whole 2000-iteration
run; what the policy actually felt was a binary wall penalty worth 2.7x the entire shaping budget
over an episode, for occupying a corridor whose wall-free band is one sixth of its width.
It learned to stay in its own arm, which is exactly what it was being paid for.
The reward now charges continuous **ramps** over the contact-activation band instead, and the
binary flag survives only as the `wall_contacts` metric.
The observation was rebuilt at the same time: every vector feature is expressed in the robot's own
travel frame `(u, n)` — plus the axis `u` itself, without which a rotation-invariant row could not
be turned back into a world-frame velocity — which collapses the plus-shape's four-fold symmetry, and it carries the
corner-block clearance and its analytic SDF normal, the current corridor half-width (otherwise the
curriculum is a hidden parameter), each neighbour's own travel direction — the feature that
separates head-on from crossing traffic — and a per-episode **priority token**.
The token is not a nicety: two robots meeting head-on have observations that are exact 180°
rotations of each other, so a shared-weight policy maps them to 180°-rotated actions and they both
accelerate or both retreat. There is a unit test for that symmetry.

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

It was quietly impossible for a different reason, and the failure is worth recording because it
was invisible to a green test suite.
The sheep's own separation-and-cohesion law has an equilibrium, and at the original
`sep_radius = 0.2` five relaxed sheep settle at radii `[0.07, 0.14, 0.14, 0.20, 0.20]` — a flock
whose *own* maximum radius is 0.198 against a pen of radius 0.2.
"All sheep inside the pen" therefore demanded the flock centroid be placed within **3 mm** of the
pen centre in a world of half-extent 1.0, and shepherds make it worse: the flee force is capped at
1.0 while cohesion is only `0.5 R ≈ 0.08`, so any shepherd inside `flee_radius` inflates the flock
faster than cohesion can pull it back.
A scripted Strömbom controller solved 5% of episodes; 3000 MAPPO iterations learned nothing, and
every unit test passed throughout.
The fix is geometry, not reward: `pen_radius` 0.3, `sep_radius` 0.15, 200-step episodes, and a
`done` that requires the flock to *hold* for five steps rather than pass through.

The spawn had to change too, and that part is not obvious.
Drawing each sheep at an independent bearing on a ring about the pen puts the flock centroid on
the pen by symmetry, so at the larger pen radius **doing nothing solved 23% of episodes** — the
ring spawn is almost all collecting and no driving.
The sheep now spawn as a *cluster*: a flock centre at a distance and bearing from the pen, then
each sheep in a disc about it. Under that spawn a scripted controller solves 84% and both a random
and a do-nothing policy solve exactly zero, which is the property
`test_a_scripted_shepherd_solves_most_episodes` now pins in CI.
The observation carries the Strömbom collect and drive points directly (dropping obs slots 16–19
recovers the un-hinted task), and `set_flock_distance` / `set_flock_spread` are the two curriculum
axes — distance is the one that matters; dispersion is nearly free.

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
