# Quickstart

## The shortest loop

`swarp.make` builds a scenario and wraps it in an [`Environment`](environment.md) in one call.

```python
import torch
import swarp

env = swarp.make("navigation", n_envs=4096, n_agents=8, device="cuda:0", dt=0.05, seed=0)

obs = env.reset()                                   # [n_envs, n_agents, obs_dim], on the GPU
for _ in range(100):
    actions = torch.rand(4096, 8, env.act_dim, device="cuda:0") * 2 - 1
    obs, reward, term, trunc, info = env.step(actions)  # every tensor stays on the GPU
```

`make` routes its keyword arguments by name: those the `Environment` constructor accepts (`device`, `dt`, `substeps`, `dtype`, `max_steps`, `seed`, `auto_reset`, `use_graph`, `clone_outputs`, `fused`) go to it, and everything else (`n_agents`, `world_size`, `model`, …) goes to the scenario constructor.
The two parameter sets are disjoint, so the split is unambiguous.
Valid names are the keys of `swarp.SCENARIOS`: `navigation`, `flocking`, `formation`, `discovery`, `sampling`, `transport`, `pusht`.

## Constructing it explicitly

Build the scenario yourself when you want to configure it beyond what `make`'s keyword routing exposes, or when the scenario is your own.

```python
from swarp import Environment, NavigationScenario

scenario = NavigationScenario(n_agents=8, n_obstacles=2, shared_reward=True)
env = Environment(scenario, n_envs=4096, device="cuda:0", dt=0.05, substeps=1, seed=0)
```

The constructor arguments worth knowing:

`n_envs`
: batch size. Required — quietly simulating a single env in a vectorized simulator is a trap.

`device`, `dtype`
: `"cuda:0"` (default) or `"cpu"`; `torch.float32` (default) or `torch.float64`. Kernels are compiled for both precisions, and float64-on-CPU is what makes strict `gradcheck` possible.

`dt`, `substeps`
: outer timestep and how many physics substeps it is split into. Stiff contacts need more: Push-T derives `substeps >= 8` at `dt=0.05` in its own `make_world`.

`max_steps`
: episode length. `None` (default) means the scenario's own `done()` is the only terminal condition.

`auto_reset`
: when `True`, `step` resets finished envs in place through a host-sync-free masked path.

`use_graph`
: `"auto"` (default) captures the whole step into a CUDA graph when the device is CUDA and the scenario provides a capturable hook. See [Performance](performance.md).

`fused`
: `"auto"` (default) uses the scenario's fused Warp obs/reward kernels on the no-grad path. Grad mode always falls back to the differentiable torch path.

`clone_outputs`
: off by default, so `step` returns **zero-copy views of buffers the next step overwrites**. Fine for a policy that consumes them immediately, wrong for anything that retains them — turn it on, or clone at the call site.

## What `step` returns

```python
obs, reward, terminated, truncated, info = env.step(actions)
```

| value | shape | notes |
|---|---|---|
| `actions` (in) | `[n_envs, n_agents, act_dim]` | `env.act_dim` is the max action arity over the agent models: 2 for the 2D vehicles, 4 for the quadrotor |
| `obs` | `[n_envs, n_agents, obs_dim]` | `env.obs_dim` comes from the scenario |
| `reward` | `[n_envs, n_agents]` | per-agent term plus the scenario's shared global term |
| `terminated` | `[n_envs]`, bool | the scenario's terminal condition; one flag per env, shared by its agents |
| `truncated` | `[n_envs]`, bool | the `max_steps` time limit; all-false when `max_steps is None` |
| `info` | `dict[str, Tensor]` | whatever the scenario emits; navigation reports `dist_to_goal` |

Actions are clamped to each agent's configured limits inside the kernel, so a policy that emits values in `[-1, 1]` is always safe.

## Resetting

There is no auto-reset by default, exactly as in raw VMAS.
Three options, all host-sync-free — none of them needs an `.any()` or `.nonzero()` round-trip to the host:

```python
obs = env.reset()                 # every env
obs = env.reset_at(done)          # only the envs whose bool mask entry is True
env = swarp.make(..., auto_reset=True)   # step resets finished envs in place
```

With `auto_reset=True` the returned `obs` already reflects the reset — for a finished env it is the first observation of the next episode, matching the gym/VMAS vec-env convention — while `reward`, `done` and `info` still describe the transition that just terminated.

## Graph observations

For GNN policies, the current within-radius neighbour graph is available as a COO edge index:

```python
edge_index = env.radius_graph()   # [2, E] int, on the env device
```

It reuses the neighbour lists `step` already built, so there is no rebuild — just the single sync needed to materialize `E`.

## Where to go next

- [Environment](environment.md) — the loop in detail: reset semantics, determinism, action layout.
- [Scenarios](scenarios.md) — the seven built-in tasks and how to pick between them.
- [Writing a scenario](writing-a-scenario.md) — your own task, in torch and (optionally) fused Warp kernels.
- [Differentiability](differentiability.md) — gradients through the physics.
- [Visualization](visualization.md) — see what your agents are doing.
