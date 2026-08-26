# Scenarios

A scenario defines the task: what the world is made of, how it resets, and what the agents observe and are rewarded for.
The engine knows nothing about tasks — every built-in scenario is written against the same public `Scenario` ABC you would use for your own.

## The seven built-ins

| name | task | agents | `obs_dim` | notable |
|---|---|---|---|---|
| `navigation` | reach a per-agent goal while avoiding collisions | 4 | 19 | optional obstacles; `shared_reward` switches per-agent shaping to a team term |
| `flocking` | Reynolds-style cohesion + alignment, penalized for crowding | 8 | 24 | pure neighbor-feature reward, no goals |
| `formation` | hold assigned slots of a regular polygon | 5 | 6 | slot `i` is assigned to agent `i`, so the shape is ordered |
| `discovery` | cover scattered targets, each needing several agents nearby | 5 | 19 | one-off shared reward the step a target is first covered |
| `sampling` | collect an unknown scalar field, consuming cells | 4 | 13 | batched sum-of-Gaussians density on a grid; obs is the 3×3 cell neighborhood |
| `transport` | push a movable circular package to a goal | 4 | 8 | package integrated in torch, staggered by one step, so BPTT flows package→agent→action |
| `pusht` | push a T-shaped rigid body to a target **pose** | 4 | 18 | movable compound body with rotation; needs `substeps >= 8` |

Defaults shown; every scenario takes constructor keywords (`n_agents`, `world_size`, shaping factors, penalties, tolerances).
`navigation` and `flocking` size their observation from `neighbor_obs`, so their `obs_dim` moves with it.

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
