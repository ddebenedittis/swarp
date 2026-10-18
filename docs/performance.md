# Performance and interop

The defaults are already the fast configuration: `Environment(fused="auto", use_graph="auto")` uses the scenario's fused Warp obs/reward kernels and captures the whole step into a CUDA graph whenever both are available.
This page explains what those two switches do, what they cost, and how the step plugs into `torch.compile` and TorchRL.

Headline: the full navigation hot path sustains **33.9 M env-steps/s** at 16,000 envs × 16 agents on one RTX 3070 Laptop GPU (543 M agent-steps/s).
That figure is the fused kernels *without* capture — `use_graph=False`, the configuration `swarp.benchmark.throughput --no-graph` pins — so it is the floor the graph builds on rather than the number the defaults produce; capture on top takes the same shape to 38.8 M env-steps/s and 16,000 × 4 to 146 M ([Benchmarks](benchmarks.md#throughput)).
Full tables and the head-to-heads against VMAS, JaxMARL and CAMAR are in [Benchmarks](benchmarks.md).

## The no-grad hot path

Under `torch.no_grad()` the step skips the tape entirely and recycles one cached `StepBuffers` per batch size — same kernels, zero steady-state allocations.
That is the path every RL rollout takes, and the one the numbers above measure.

Two things stack on top of it.

### Fused obs/reward kernels

Each built-in scenario ships a `*_kernels.py` module that computes observations and rewards in Warp instead of torch.
It removes a per-step chain of small torch launches, and — more importantly — makes the *whole* step capturable as one graph.

Worth about 1.8–4.5× on its own in the open-field benchmark below.
`fused="auto"` follows the scenario's `fused_available`; grad mode always falls back to the differentiable torch path.

### Whole-step CUDA graph

`use_graph="auto"` (the default) enables persistent-buffer execution backed by a captured CUDA graph when the device is CUDA and the scenario supplies a capturable whole-step hook.
`StepRuntime` (`swarp/interop/persistent.py`) then owns a fixed set of on-device buffers — the state, a scratch output, the substep scratch, the action buffer — plus zero-copy torch views created once. Each step copies the incoming actions into the fixed buffer and replays the graph.

The graph adds another 1.1–5.4× on top of the fused kernels, for ~4.4–11× end to end over an eager baseline (`compare_sims --sims swarp swarp-fused swarp-eager`, obstacle-free navigation, 300 timed steps):

```text
  n_envs  n_agents | swarp-eager swarp-fused       swarp |  fused/eager  opt/eager
    4096         3 |   7,649,544  15,980,008  86,003,167 |       2.09x     11.24x
   16384         3 |  25,962,701  47,203,892 145,339,064 |       1.82x      5.60x
    4096        16 |   6,020,546  14,022,673  26,341,168 |       2.33x      4.38x
   16384        16 |   6,538,976  29,355,880  32,409,682 |       4.49x      4.96x
```

Details that matter in practice:

- Warp launches go to Warp's *own* created stream, which is what makes capture legal in the first place. That stream is created with the **blocking** flag, so the driver serializes it against the legacy default stream both ways — which is what keeps masked auto-resets and observation reads ordered with the replay, with no device sync and nothing for swarp to do. Under a **user-created** `torch.cuda.Stream` that guarantee lapses, so every eager entry point (the replay, the eager fallback, the pre-capture warm-up, `World.neighbors`, the fused obs/reward launches, the lidar scan) opens a scope onto torch's current stream; on the default stream that scope is deliberately a no-op behind a 0.4 us stream check, since paying for it there cost 10-36% of a graph-mode step (most of it inside `torch.cuda.current_stream()`, which builds a Python object and measured 17 us against 0.4 us for the raw accessor). The capture itself can never be scoped: `wp.ScopedCapture` captures the current stream, and capturing torch's legacy stream is a hard CUDA error, so capture stays on Warp's own stream. Replaying a graph on the legacy stream is fine; only capturing it is not.
- Auto-reset stays **outside** the graph: it uses a `torch.Generator` whose philox offset is not capture-safe.
- Passing `use_graph=True` demands persistent execution even where capture is unavailable — it then falls back to eager persistent execution with a one-time warning. `use_graph=False` forces the plain functional step.
- Capture is unavailable on CPU, and with the `wp.HashGrid` `"grid"` neighbor backend, which allocates during a build.

`env.graph_mode` reports whether a graph is actually in play.

### Fused substep chain

Inside the graph, each launch is a node, and at the sizes RL trains at (a few agents per env, a few thousand envs, 4–8 substeps) a node costs more than the work in it.
The layered pipeline issues a neighbor build, a force pass and an integrate pass per substep, so at 1024 envs × 4 agents × 8 substeps the physics alone was ~23 nodes.
`swarp/core/fused_substeps.py` runs the whole chain as one kernel instead: one CUDA block owns a group of whole envs, one thread per agent, and a block barrier separates the force phase from the in-place integrate phase of every substep.
The force phase reads the previous step's neighbor list on substep 0 and scans the env directly on later substeps, with the brute-force builder's membership test in its order, so it sums the same terms in the same order and the step is bit-identical to the layered pipeline.
When something reads neighbor lists after the step (navigation's observations do), the kernel also writes the post-step list, and the hook's rebuild becomes a no-op.

The trade is launches against occupancy: a kernel that holds the whole chain needs ~100 registers per thread where the layered kernels need 35–80, and its neighbor scans are latency-bound.
`Stepper.enable_fused_substeps="auto"` (the default) therefore takes it only where it measured faster — at most 4 agents per env per substep, and with a single substep at most 16,384 agents in all — and falls back to the layered pipeline elsewhere.
It also needs the CUDA device, the slim 2D Euler path, shared agent parameters, no movable obstacles and the `brute` neighbor backend.
`True` forces it wherever it is eligible, `False` turns it off.

At 1024 envs × 4 agents × 8 substeps with `auto_reset=True` the step drops from 84 to 43 µs; the full table is in [Benchmarks](benchmarks.md#fused-substep-chain).

### Lean launches

Every swarp kernel module sets the Warp module option `default_grid_stride=False` (Warp ≥ 1.15), and the fused chain passes `grid_stride=False` itself.
A Warp kernel otherwise wraps its body in a grid-stride loop so that a capped block count can still cover the launch; swarp never caps it, so the loop is pure per-thread overhead.
Dropping it measured 2–6% faster on the launch-bound shapes (small batches, and the 3–5-agent, 4–8-substep training shapes) and neutral elsewhere.
It is set per module rather than through `wp.config`, so it does not change how a user's own kernels compile.

## `torch.compile`

The default `warp_step` wraps a `torch.autograd.Function` around raw Warp launches, which Dynamo cannot trace through — it graph-breaks on the opaque Function and the Warp C calls.
`swarp/interop/compile.py` registers the step as a functional custom op `swarp::step` with a fake (meta) implementation and an autograd rule, so a compiled loop treats it as one opaque differentiable operator instead:

```python
from swarp.interop.compile import compiled_warp_step, register_stepper

register_stepper(stepper)
state = compiled_warp_step(stepper, state, actions)   # same signature as warp_step
```

The op is functional: the forward allocates fresh outputs (no buffer recycling, no input mutation) and the backward re-records a tape from the saved inputs and replays it, so no Warp state has to survive between the two calls — which is exactly what AOTAutograd expects.

## TorchRL

`swarp/interop/torchrl.py` exposes an `Environment` as a batched `EnvBase` (needs the `torchrl` extra):

```python
from swarp.interop.torchrl import SwarpEnv

sim = swarp.make("navigation", n_envs=4096, device="cuda:0")
env = SwarpEnv(sim)                       # action_spec == sim.action_bounds
```

- `batch_size=[n_envs]`, everything on-device.
- `"observation"` `[n_envs, n_agents, obs_dim]`; `"action"` `[n_envs, n_agents, act_dim]`; `reward` `[n_envs, n_agents, 1]`; a shared per-env `done` `[n_envs, 1]`.
- A non-empty scenario `info()` is spec'd and forwarded as a nested `info` composite, reachable at the **flat** path `("next", "info", <key>)`.

:::{note}
swarp actions are physical, so the action spec defaults to `env.action_bounds` — the box the kernels actually clamp to.
Pass an explicit scalar pair (`action_low=-1.0, action_high=1.0`) if you want a normalized box instead — see [Action bounds are physical](environment.md#action-bounds-are-physical).
:::

:::{warning}
`SwarpEnv` is flat — there is no `("agents", …)` group — so info is at `("next", "info", <key>)`, **not** the group-nested path TorchRL's `VmasEnv` uses.
Downstream code written against `VmasEnv` needs the flat path here.
:::

[`examples/train_mappo.py`](https://github.com/ddebenedittis/swarp/blob/main/examples/train_mappo.py) is a full MAPPO training loop on top of it.
It is kept as the reference implementation; [`swarp.rl`](training.md) trains the same recipes 2–3× faster without TorchRL; `examples/train_mappo_tuned.py` is the same trainer with TorchRL's own speed features, 1.3–1.5× behind `swarp.rl` at the recipes' sizes.

## Choosing a batch shape

Every batch size up to 16,000 envs × 16 agents lands on the same ~0.3–0.5 ms/step host floor eager, so raising `n_agents` is close to free until the neighbor and force kernels saturate the GPU.
If you have a throughput target, raise `n_agents` before `n_envs`.

Two knobs with real cost:

- `substeps` multiplies the physics work linearly. Only raise it for stiff contacts (Push-T derives `substeps >= 8` at `dt=0.05` in its own `make_world`). With few agents per env the [fused substep chain](#fused-substep-chain) makes extra substeps much cheaper than linear (8 substeps cost 1.5x one substep at 1024 envs × 4 agents, against 2.4x for the layered pipeline), because the cost there was launches rather than arithmetic.
- `max_neighbors` sets the padded list width. Too small silently truncates — except that it does not, because `World.neighbor_overflow()` flags it; check that flag once when tuning rather than guessing.

## The cost of `auto_reset`

`auto_reset=True` is **not** free, and the reason is structural rather than incidental. `done` is a device tensor and the step promises no device→host round-trip, so there is no host-side "is anything done?" gate: the scenario's `reset_world` runs on **every** step, over the **whole** batch, and the per-env `reset_mask` selects what actually lands. Resetting one env out of 4096 therefore costs the same as resetting all of them.

That makes the reset path's own cost the thing to watch. It used to be a fixed *host-side* tail appended after the captured step — first a 16-iteration `torch.cdist` rejection loop in navigation's spawn sampler, run twice per reset (spawns, then goals), for 40.8 ms/step at 4096x8 (315x the plain step); stratified jittered-cell sampling replaced it, getting the pairwise separation by construction in a single draw. That left a torch draw still running every step, and at 16,384x16 it was **86%** of the step: two batched `argsort`s (the only way to get a uniform random k-subset out of batched torch ops), four `sample_uniform`s and the `torch.where` blends — around 25 ops, on a path that is bound by op count rather than by arithmetic. Consolidating that into one masked Warp launch per scenario (`nav_reset_kernel` and its siblings, one thread per env, drawing distinct cells by a partial Fisher–Yates over a scratch permutation; shared pieces in `swarp/scenarios/reset_kernels.py`) got the reset from 3.9 to 1.20 ms/step at 4096x8, and 1.3–1.6x on every other scenario's `auto_reset` steps.

That 1.20 ms/step was still an eager tail after the graph replay, though — the masked reset kernel, the masked neighbor rebuild and the masked obs pass all ran as separate host-launched calls every step, on a config where the tail cost more than the physics it followed. The remaining fix was to stop treating the reset as something that happens *after* the graph: the reset RNG seed moves onto the device (a one-thread kernel advances it in place before each draw, reproducing the old host-side "increment then use" arithmetic exactly, so the drawn sequence is unchanged) and the whole masked sequence — episode-end test, seed advance, reset, neighbor rebuild, obs pass — gets folded into the same `wp.ScopedCapture` as the physics. A step under `auto_reset` is then one graph replay end to end, the same shape as a step without it. This is opt-in per scenario (`FusedScenario.supports_graph_reset()`; navigation qualifies when it has no obstacles, since sampling obstacle poses needs `world.generator` and a spec re-install, neither of which belongs inside a capture) — scenarios that don't qualify keep the eager tail above.

Same guarantee, same distribution throughout — a *uniform* k-subset, which is the part worth protecting: a cheaper structured draw (cells by a random base and stride) passes every separation and bounds check while collapsing the reachable spawn layouts from C(25,16) ≈ 2.0M to 250, which surfaces much later as a generalization failure rather than as a test failure. `tests/unit/test_reset_kernel.py` pins it.

At 4096x8 on an RTX 3070 Laptop, graph on, obstacle-free navigation:

| | ms/step | vs `auto_reset=False` |
|---|---|---|
| `auto_reset=False` | 0.08 | — |
| `auto_reset=True`, realistic (a slice of the batch resets each step) | 0.10 | 1.24x |
| `auto_reset=True`, worst case (every env resets every step) | 0.13 | 1.60x |

At 16,000x16 the gap narrows further in relative terms as the physics work grows to dominate the step: 0.40 ms/step without reset, 0.43 ms realistic, 0.87 ms worst case (2.2x) — against the 3.9 ms a *single* reset used to cost at the smaller 4096x8 config before any of this work. Resetting one env out of the batch now costs close to nothing extra; resetting the whole batch on the same step is the regime that still shows up, because the masked kernels still touch the same amount of data whether one env changed or all of them did.

Two things follow for anyone writing a scenario:

- Whatever `reset_world` does, it does on every step under `auto_reset`. Budget it as hot-path work, not as setup — and prefer one masked kernel over a chain of masked torch ops, because what costs you there is the number of launches, not the arithmetic. If the reset can be made capture-safe (no host-side RNG draws, no allocation, no branching on tensor values), `supports_graph_reset()` folds it into the graph and removes the launch cost entirely.
- If you only need resets at episode boundaries you control, leave `auto_reset=False` and call `env.reset_at` yourself with a mask — you then pay the reset cost only on the steps that need it.

## Running the benchmarks

```bash
python -m swarp.benchmark.throughput      # the full (n_envs, n_agents) grid
python -m swarp.benchmark.ablation        # cumulative optimization ablation, parity-gated
python -m swarp.benchmark.scenarios       # all 11 scenarios x model x lidar rays
python -m swarp.benchmark.compare_vmas    # vs VMAS: --metric {throughput,memory,both}
python -m swarp.benchmark.compare_sims    # vs VMAS, JaxMARL and CAMAR
```

[Benchmarks](benchmarks.md) has the numbers, the measurement conditions and the multi-venv setup the cross-simulator comparison needs.
