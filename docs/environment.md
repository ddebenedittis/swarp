# The environment loop

`Environment` (`swarp/core/environment.py`) is the VMAS-style front door: it owns a [`Scenario`](scenarios.md), which owns a `World`, which owns the `Stepper` that launches the kernels.
Everything below stays on the environment's device.

```python
import torch
import swarp

env = swarp.make("navigation", n_envs=4096, n_agents=8, device="cuda:0", auto_reset=True)

obs = env.reset()
with torch.no_grad():
    for _ in range(1000):
        actions = policy(obs)
        obs, reward, done, info = env.step(actions)
```

`step` is differentiable end-to-end when called with grads enabled, and takes a tape-free hot path under `torch.no_grad()` — same kernels, no per-step allocations.
See [Differentiability](differentiability.md) for the first and [Performance](performance.md) for the second.

## Actions

The action tensor is always `[n_envs, n_agents, act_dim]` where `act_dim = env.act_dim` is the **maximum** arity over the agent models in the world: 2 for the three 2D vehicle models, 4 for the quadrotor.
A model reads only the slots it uses and ignores the rest, so a heterogeneous fleet of holonomic agents and drones takes one rectangular tensor with the holonomic agents ignoring slots 2 and 3.

`step` validates the shape and the dtype and raises rather than broadcasting silently.
Inside the kernel each action is clamped to that agent's configured limits, so out-of-range values are safe (they just saturate — and a saturated action has zero gradient, see [Differentiability](differentiability.md)).

What the two or four slots *mean* is set by the agent's `ControlMode`; see [Dynamics models](dynamics.md).

## Returns

```python
obs, reward, done, info = env.step(actions)
```

- `obs` — `[n_envs, n_agents, obs_dim]`, the scenario's observation after any auto-reset.
- `reward` — `[n_envs, n_agents]`, the scenario's per-agent term plus its shared global term.
- `done` — `[n_envs]` bool, one flag per env (agents in an env terminate together), OR-ed with the `max_steps` timeout when one is set.
- `info` — a dict of whatever the scenario chooses to expose; navigation reports `dist_to_goal`, Push-T reports the pose error, and so on.

Reward, done and info describe the transition that was just taken — the terminal state — while `obs` is already the next episode's first observation for any env that auto-reset.

:::{warning}
`clone_outputs` is `False` by default, so these are **zero-copy views of buffers the next step overwrites**.
That is what makes the hot path allocation-free, and it is wrong for anything that retains them: a replay buffer, a trajectory list, a plot at the end of the rollout.
Either construct with `clone_outputs=True` or clone at the call site.
:::

## Resetting without a host sync

```python
obs = env.reset()             # all envs; optionally reseed with reset(seed=...)
obs = env.reset_at(mask)      # only where the [n_envs] bool mask is True
```

`reset_at` writes with a masked `torch.where` blend, so it never needs `.any()` or `.nonzero()` — no device→host round-trip, and the whole RL loop can stay on-device.
`auto_reset=True` makes `step` do exactly that with the `done` mask it just computed.

The reset primitive scenarios build on is `World.write_state(mask, pos=..., vel=..., ...)`, which does the blend in place and invalidates the cached neighbour list when positions move.
Writing `state.pos` directly and forgetting `World.mark_pos_dirty()` is a silent stale-neighbour-list bug — that is the reason to route resets through `write_state`.

## Reading the state

`env.world.state` is a structure of arrays, one field per quantity, each `[n_envs, n_agents, …]`:

| field | shape | meaning |
|---|---|---|
| `pos` | `[E, A, 2]` | planar position |
| `vel` | `[E, A, 2]` | translational velocity used in the last pose update (all models) |
| `theta` | `[E, A]` | heading — ignored by holonomic agents |
| `speed` | `[E, A]` | scalar forward speed, integrated by diff-drive (acceleration mode) and bicycle |
| `ang_vel` | `[E, A]` | yaw rate — ignored by holonomic agents |
| `z`, `vz`, `attitude`, `body_rates` | `[E, A]`, `[E, A]`, `[E, A, 4]`, `[E, A, 3]` | the extra 6-DOF drone state; zero (identity attitude) for the 2D models |

One unified state serves every model, which is what keeps observations uniform across a heterogeneous fleet.

## Neighbours and the radius graph

```python
idx, count = env.world.neighbors()      # [E, A, K] int32, [E, A] int32 — zero-copy views
overflow = env.world.neighbor_overflow()  # [E, A] bool
edges = env.radius_graph()              # [2, E_edges] COO, for a GNN policy
```

The neighbour lists are padded to `WorldConfig.max_neighbors`, and `neighbor_overflow()` flags any agent whose true in-radius count exceeded that width — truncation is never silent.
`radius_graph()` reuses the grid `step` already built, so it costs one sync to materialize the edge count and no rebuild.
Both views are overwritten by the next call: gather from them within the step.

## Determinism and seeding

`Environment(..., seed=0)` seeds the world RNG, and `reset(seed=...)` reseeds it.
Given the same seed, device, dtype and action sequence, rollouts are bit-reproducible: collision forces are gather-based (each agent sums over its own neighbour list) so there are no atomics and no nondeterministic reduction order.

Determinism does **not** carry across devices or precisions — float32-on-CUDA and float64-on-CPU are different arithmetic — and it does not make swarp trajectory-compatible with VMAS. The dynamics, collision constants and observation model differ by design.

## Rendering

`env.render(mode="rgb_array", env_index=0)` returns an `(H, W, 3)` uint8 frame and `mode="human"` drives a persistent window; both need the `viz` extra.
[Visualization](visualization.md) covers the viewer, the overlays and video export.
