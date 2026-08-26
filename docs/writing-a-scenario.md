# Writing a scenario

A scenario defines a task: what the world is made of, how it resets, and what the agents
observe and are rewarded for. There are two tiers, and you only pay for the second if you
want it.

1. **`Scenario`** (`swarp/scenarios/base.py`) — plain torch. Everything is differentiable
   end-to-end with the Warp dynamics step, and there is nothing else to implement.
2. **`FusedScenario`** (`swarp/scenarios/fused.py`) — adds fused Warp obs/reward kernels for
   the no-grad hot path, foldable into the whole-step CUDA graph. Worth 2.5–5× (see
   [benchmarks](benchmarks.md)); costs you a kernel module and a buffer spec.

All seven built-in scenarios are `FusedScenario`s, and they implement the *same* member
set — there are no opt-outs to copy from.

---

## Tier 1: the plain `Scenario` ABC

Four abstract members:

| member | what it does |
|---|---|
| `make_world(n_envs, device, dt, substeps, dtype, world_config=None) -> World` | build the `World`, **and allocate all persistent state** |
| `reset_world(env_mask=None, *, obs_only=False)` | (re)randomize state, goals, obstacles |
| `observations() -> [n_envs, n_agents, obs_dim]` | batched observations |
| `obs_dim -> int` | per-agent observation width |

plus the optional `agent_reward(i)` / `global_reward()` / `done()` / `info()` /
`post_step()` / `render_extras(env_idx)`. `observation(i)` and `rewards()` are derived and
rarely worth overriding.

### `make_world` owns *all* allocation

`n_envs` is known here, so nothing needs a lazy `if self.x is None:` in `reset_world`.
That matters more than it looks: `reset_world` runs on the per-step auto-reset path, and a
lazy allocation there is an undeclared ordering precondition — the fused path breaks if it
runs before the first reset. `FusedScenario` turns that into a hard error rather than a
subtle one.

### `make_world` must honour `world_config`

Compute your `WorldConfig` from your own scenario parameters as usual, then end with one
call:

```python
cfg = WorldConfig(
    collisions=True,
    collision_margin=margin,
    bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
    neighbor_radius=reach,
).override_with(world_config)
```

That is the whole contract. It is what lets a caller reach engine settings your
constructor does not expose (`integrator`, `bounds_mode`, `neighbor_reuse`, `grid_dim`,
`uniform_bins`, the obstacle damping) via `swarp.make(..., world_config=...)` without
subclassing, while your computed `bounds` and `neighbor_radius` survive — only the fields
set away from the `WorldConfig()` defaults are taken. `tests/scenarios/test_scenarios.py`
pins that all seven built-ins do this.

### `reset_world` must be host-sync-free

Auto-reset runs *inside* the step loop, so a reset that syncs the device to the host costs
a round-trip every episode boundary. Sample the full batch width and blend, rather than
gathering a variable number of indices:

```python
def reset_world(self, env_mask=None, *, obs_only=False):
    w = self.world
    spawn = w.sample_uniform((w.n_envs, self.n_agents, 2), -lim, lim)   # full width
    w.write_state(env_mask, pos=spawn, vel=0.0)                          # masked blend
```

`World.write_state` does the `torch.where` blend plus an in-place `copy_` (so persistent
buffers, their zero-copy views and any captured graph stay valid) and calls
`mark_pos_dirty()` exactly once if `pos` was among the fields. Forgetting that call is a
silent stale-neighbor-list bug, which is the main reason to route resets through it.

No `.item()`, no `.any()`, no `if tensor:` — `tests/interop/test_no_host_copies.py` patches
those to raise and steps the env.

### `obs_only`

`True` marks the mid-step auto-reset pass. Reward/done/info for the transition just taken
have **already been returned to the caller**, so this pass must recompute observations
without clobbering their buffers. `False` (a standalone `reset`/`reset_at`) recomputes
everything, so `info()` is populated.

---

## Tier 2: `FusedScenario`

Write your kernels in `swarp/scenarios/<name>_kernels.py`, then implement four members:

```python
class MyScenario(FusedScenario):
    def fused_spec(self, n_envs) -> tuple[Buf, ...]: ...   # the buffers, declaratively
    def launch_fused(self, pass_: FusedPass) -> None: ...  # the wp.launch sequence
    def post_step_torch(self) -> None: ...                 # torch reference, after a step
    def reset_torch(self, env_mask) -> None: ...           # torch reference, after a reset
```

and end `reset_world` with `self.finish_reset(env_mask, obs_only=obs_only)`.

That is the whole contract. The framework then owns lazy allocation, `uint8 -> bool`
reinterpret views, Warp handle caching, pointer-move resync, the recapture token, the
warm-up carry list, the reset-mask stamp, `post_step`, and `graph_hook`.

### The buffer spec

`Buf(name, shape, dtype, attr, alloc, carry, watch, bool_view, reset_mask)`. Flocking's is
the minimum:

```python
def fused_spec(self, n_envs):
    ne, na = n_envs, self.n_agents
    return (
        Buf("obs", (ne, na, self.obs_dim)),
        Buf("reward", (ne, na)),
        Buf("crowd", (ne, na)),
    )
```

Buffers land in `self.fb[name]` (torch) and `self._wp[name]` (the cached `wp.array` your
launches pass to the kernels).

**`dtype`** says how the buffer is seen from torch *and* from Warp, in one field:

| `dtype` | torch | Warp | used for |
|---|---|---|---|
| `"float"` (default) | world dtype | scalar | obs, rewards, distances |
| `"vec2"` | world dtype, trailing `2` | `vec2` | positions, goals |
| `"uint8"` | `torch.uint8` | `uint8` | kernel-written flags |
| `"bool"` | `torch.bool` | `uint8` | an adopted torch bool latch a kernel also writes |

**`bool_view=True`** additionally exposes `fb[name + "_bool"]`, a zero-copy reinterpret, so
`done()` can return a real bool tensor from bytes a kernel wrote.

**`attr` + `alloc`** cover state the framework does *not* own:

- `alloc="always"` (default) — framework-owned, allocated here.
- `alloc="never"` — adopt whatever `attr` holds; `None` is an error. This is your task
  state (`self.targets`, `self.consumed`, `self.tee_pos`) or read-only world state
  (a dotted `attr="world.goals"`). Allocate it in `make_world`.
- `alloc="if_none"` — allocate and assign only if `attr` is `None`. For a carry whose
  *first* value must be derived from the state rather than be zero, so the torch reference
  path owns the seeding.

**`carry=True`** for anything your launches advance **in place**: a shaping baseline, a
coverage latch, movable-body state. Graph warm-up runs your hook once on the input state
purely to compile kernels, so carries are snapshotted before and restored after it.
Omitting one does not fail loudly — it advances the simulation one extra step at capture.
Declare it.

**`watch=True`** for anything the grad path *reassigns*. The torch reference builds fresh
tensors for the tape, and a graph captured against the old pointers holds dead memory. The
framework compares `data_ptr()` each `prepare`, rebuilds the handle, and bumps the
recapture token. Only meaningful with `attr`.

**`reset_mask=True`** on the one per-env "just reset" mask, if your kernels take one. The
framework zeroes it before a normal step and stamps it on a reset.

Navigation exercises every axis at once and is the reference example:

```python
def fused_spec(self, n_envs):
    ne, na = n_envs, self.n_agents
    return (
        Buf("obs", (ne, na, self.obs_dim)),
        Buf("touch", (ne, na)),
        Buf("dist", (ne, na)),
        Buf("shaping", (ne, na)),
        Buf("reward", (ne, na)),
        Buf("ongoal", (ne, na), "uint8", bool_view=True),
        Buf("overflow", (ne, na), "uint8", bool_view=True),
        Buf("done", (ne,), "uint8", bool_view=True),
        Buf("resetmask", (ne,), "uint8", reset_mask=True),
        Buf("goals", (ne, na, 2), "vec2", attr="world.goals", alloc="never", watch=True),
        Buf("prev", (ne, na), attr="_prev_dist", alloc="if_none", carry=True, watch=True),
    )
```

A buffer must be **contiguous at the source**. The framework raises rather than calling
`.contiguous()` for you: that copy is a temporary whose device pointer would be baked into
the captured graph and freed immediately afterwards.

### `launch_fused` and `FusedPass`

One method serves both the post-physics step and the two flavours of reset. `FusedPass`
describes the pass; you map it onto your own sequence:

| property | meaning |
|---|---|
| `is_step` | post-physics (`True`) vs reset (`False`) |
| `advance_prev` | `1` on a step (advance a shaping baseline), `0` on a reset (rebase it) |
| `full_pass` | `1` to write reward/info too, `0` for observations only |
| `env_mask` | which envs a reset touched (`None` = all) |

`advance_prev` and `full_pass` are named after the kernel arguments they feed, so the
mapping reads as a rename. Where each scenario puts them differs, and all of it is
expressible without touching a kernel signature:

```python
# navigation / formation — flags on the obs kernel; no reward on a reset
def launch_fused(self, pass_):
    self._launch_obs(advance_prev=pass_.advance_prev, full_pass=pass_.full_pass)
    if pass_.is_step:
        self._launch_reward()

# transport / pusht — flags on the reward kernel; no body advance on a reset
def launch_fused(self, pass_):
    if pass_.is_step:
        self._launch_body()
        self._install_obstacles()
    self._launch_obs()
    self._launch_reward(advance_prev=pass_.advance_prev, full_pass=pass_.full_pass)

# flocking / sampling — only full_pass
def launch_fused(self, pass_):
    self._launch(full_pass=pass_.full_pass)

# discovery — neither flag: "a reset must not clobber the reward" by omitting the launch
def launch_fused(self, pass_):
    self._launch_cover()
    self._launch_obs()
    if pass_.is_step:
        self._launch_reward()
```

### Capture safety — the one rule that bites

On a `is_step` pass, `launch_fused` runs **inside `wp.ScopedCapture`**. Every line must be
`wp.launch` or a neighbor-grid build against the pointer-stable handles in `self._wp`, with
**no device allocation and no host read**. A `torch.zeros`, an `.item()`, a `.contiguous()`
on something that is not already contiguous — any of those either fails the capture or, far
worse, bakes a pointer that is freed on return.

Torch calls are legal only where they provably allocate and sync nothing. Transport's
in-capture `world.set_obstacles(self._obstacles)` qualifies, and it is worth understanding
why, because it is the shape of every such exception:

- the spec is **retained**, not rebuilt, so `Obstacles.resolve()` returns `self` and
  allocates nothing;
- `Obstacles.any_movable` is memoized on that instance, so the reduction + `.item()` it
  costs on a *fresh* spec never happens;
- the obstacle count is unchanged, so `Stepper.set_obstacles` takes its in-place path.

`Stepper.set_obstacles`'s docstring states that contract and `tests/unit/test_obstacles.py`
pins it, including a `wp.ScopedCapture` case. Anything that cannot meet a bar like that
belongs in the spec instead, where the framework allocates it once, up front.

### The torch reference path is the parity oracle

`post_step_torch` / `reset_torch` and whatever they call (`_refresh`, `_box_contact`, …) are
what the fused kernels are *tested against*. Keep them independent — sharing code with the
fused path would weaken the exact property the seven `tests/scenarios/test_*_fused.py`
suites exist to test. For the same reason the `_launch_*` wrappers are deliberately not
unified: their `wp.launch` signatures are bespoke, and a launch DSL would be a worse
`wp.launch` with none of Warp's type checking.

If your two paths legitimately differ by more than reassociation — push-t integrates its
body in torch at `body_substeps` and in the engine at `Stepper.substeps` — say so once, on
the class:

```python
parity_rtol: float = 1e-2
parity_atol: float = 5e-3
```

Both the benchmark parity gate (`swarp.benchmark.scenarios`) and the test harness
(`tests/conftest.py`'s `FusedSpec`) read those, so there is one number per scenario rather
than two tables that drift.

---

## Registering it

`SCENARIOS` in `swarp/scenarios/__init__.py` is the single registry: `swarp.make(name, ...)`,
`make_scenario`, `scenario_class`, the benchmark CLIs' `--scenario` and the tests all read it,
and `fused_scenarios()` derives fused capability from `cls.fused_available` — which
`FusedScenario` sets — so there is no second list to keep in step.

Your scenario almost certainly lives outside this package, so add it with
`register_scenario` at import time of the module that defines it. No fork, no edit to the
installed package:

```python
import swarp
from swarp.scenarios import register_scenario

register_scenario("my_task", MyTaskScenario)

env = swarp.make("my_task", n_envs=4096, n_agents=8, device="cuda:0")
```

A name collision raises rather than silently swapping — pass `overwrite=True` if replacing a
built-in is what you meant.

For a scenario contributed *to* the package, add it to the `SCENARIOS` dict literal directly
instead; the registry is the same object either way.
