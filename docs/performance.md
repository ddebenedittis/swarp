# Performance and interop

The defaults are already the fast configuration: `Environment(fused="auto", use_graph="auto")` uses the scenario's fused Warp obs/reward kernels and captures the whole step into a CUDA graph whenever both are available.
This page explains what those two switches do, what they cost, and how the step plugs into `torch.compile` and TorchRL.

Headline: the full navigation hot path sustains **24.1 M env-steps/s** at 16,000 envs × 16 agents on one RTX 3070 Laptop GPU, peaking at 491 M agent-steps/s at 16,000 × 64.
That figure is the fused kernels *without* capture — `use_graph=False`, the configuration `swarp.benchmark.throughput` pins — so it is the floor the graph builds on rather than the number the defaults produce; capture on top reaches 35.6 M env-steps/s at 16,384 × 16 ([table below](#whole-step-cuda-graph)).
Full tables and the head-to-heads against VMAS, JaxMARL and CAMAR are in [Benchmarks](benchmarks.md).

## The no-grad hot path

Under `torch.no_grad()` the step skips the tape entirely and recycles one cached `StepBuffers` per batch size — same kernels, zero steady-state allocations.
That is the path every RL rollout takes, and the one the numbers above measure.

Two things stack on top of it.

### Fused obs/reward kernels

Each built-in scenario ships a `*_kernels.py` module that computes observations and rewards in Warp instead of torch.
It removes a per-step chain of small torch launches, and — more importantly — makes the *whole* step capturable as one graph.

Worth about 1.1–1.4× on its own, more at large batch sizes where launch overhead dominates.
`fused="auto"` follows the scenario's `fused_available`; grad mode always falls back to the differentiable torch path.

### Whole-step CUDA graph

`use_graph="auto"` (the default) enables persistent-buffer execution backed by a captured CUDA graph when the device is CUDA and the scenario supplies a capturable whole-step hook.
`StepRuntime` (`swarp/interop/persistent.py`) then owns a fixed set of on-device buffers — the state, a scratch output, the substep scratch, the action buffer — plus zero-copy torch views created once. Each step copies the incoming actions into the fixed buffer and replays the graph.

The graph roughly doubles the fused throughput again, for ~2.5–5× end-to-end over an eager baseline:

```text
  n_envs  n_agents | swarp-eager swarp-fused       swarp |  fused/eager  opt/eager
    4096         3 |   5,838,664   7,877,854  15,945,297 |       1.35x      2.73x
   16384         3 |  21,772,740  30,814,526  61,718,330 |       1.42x      2.83x
    4096        16 |   5,998,275   6,783,331  15,094,932 |       1.13x      2.52x
   16384        16 |   6,858,433  28,191,165  35,637,271 |       4.11x      5.20x
```

Details that matter in practice:

- Warp launches go to Warp's *own* created stream, which is what makes capture legal in the first place. That stream is created with the **blocking** flag, so the driver serializes it against the legacy default stream both ways — which is what keeps masked auto-resets and observation reads ordered with the replay, with no device sync and nothing for swarp to do. Under a **user-created** `torch.cuda.Stream` that guarantee lapses, so every eager entry point (the replay, the eager fallback, the pre-capture warm-up, `World.neighbors`, the fused obs/reward launches, the lidar scan) opens a scope onto torch's current stream; on the default stream that scope is deliberately a no-op behind a 0.4 us stream check, since paying for it there cost 10-36% of a graph-mode step (most of it inside `torch.cuda.current_stream()`, which builds a Python object and measured 17 us against 0.4 us for the raw accessor). The capture itself can never be scoped: `wp.ScopedCapture` captures the current stream, and capturing torch's legacy stream is a hard CUDA error, so capture stays on Warp's own stream. Replaying a graph on the legacy stream is fine; only capturing it is not.
- Auto-reset stays **outside** the graph: it uses a `torch.Generator` whose philox offset is not capture-safe.
- Passing `use_graph=True` demands persistent execution even where capture is unavailable — it then falls back to eager persistent execution with a one-time warning. `use_graph=False` forces the plain functional step.
- Capture is unavailable on CPU, and with the `wp.HashGrid` `"grid"` neighbor backend, which allocates during a build.

`env.graph_mode` reports whether a graph is actually in play.

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

[`examples/pusht_torchrl.py`](https://github.com/ddebenedittis/swarp/blob/main/examples/pusht_torchrl.py) is a full MAPPO training loop on top of it.

## Choosing a batch shape

Almost every batch size lands on the same ~0.65–0.8 ms/step host floor, so raising `n_agents` is close to free until the neighbor and force kernels saturate the GPU.
If you have a throughput target, raise `n_agents` before `n_envs`.

Two knobs with real cost:

- `substeps` multiplies the physics work linearly. Only raise it for stiff contacts (Push-T derives `substeps >= 8` at `dt=0.05` in its own `make_world`).
- `max_neighbors` sets the padded list width. Too small silently truncates — except that it does not, because `World.neighbor_overflow()` flags it; check that flag once when tuning rather than guessing.

## The cost of `auto_reset`

`auto_reset=True` is **not** free, and the reason is structural rather than incidental. `done` is a device tensor and the step promises no device→host round-trip, so there is no host-side "is anything done?" gate: the scenario's `reset_world` runs on **every** step, over the **whole** batch, and the per-env `reset_mask` selects what actually lands. Resetting one env out of 4096 therefore costs the same as resetting all of them.

That makes the reset path's own cost the thing to watch. At 4096x8 on an RTX 3070 Laptop:

| | ms/step |
|---|---|
| `auto_reset=False` | 0.14 |
| `auto_reset=True` | 1.20 |

Navigation's spawn sampler used to dominate that second row, twice over. First it was a fixed 16-iteration `torch.cdist` rejection loop, run twice per reset (spawns, then goals), for 40.8 ms/step — 315x the plain step; stratified jittered-cell sampling replaced it, getting the pairwise separation by construction in a single draw.

That left a torch draw still running every step, and at 16,384x16 it was **86%** of the step: two batched `argsort`s (the only way to get a uniform random k-subset out of batched torch ops), four `sample_uniform`s and the `torch.where` blends — around 25 ops, on a path that is bound by op count rather than by arithmetic. The whole masked reset is now a single Warp launch (`nav_reset_kernel`), one thread per env, drawing its distinct cells by a partial Fisher–Yates over a scratch permutation. Every other scenario does the same — one masked kernel per reset, with the shared pieces in `swarp/scenarios/reset_kernels.py` — for 1.3-1.6x on their `auto_reset` steps. Same guarantee, same distribution — a *uniform* k-subset, which is the part worth protecting: a cheaper structured draw (cells by a random base and stride) passes every separation and bounds check while collapsing the reachable spawn layouts from C(25,16) ≈ 2.0M to 250, which surfaces much later as a generalization failure rather than as a test failure. `tests/unit/test_reset_kernel.py` pins it.

The reset went from 3.9 to 1.20 ms/step at 4096x8; at 16,384x16 the RL configuration (graph on, `auto_reset=True`) went from 4.2M to 13.8M env-steps/s.

Two things follow for anyone writing a scenario:

- Whatever `reset_world` does, it does on every step under `auto_reset`. Budget it as hot-path work, not as setup — and prefer one masked kernel over a chain of masked torch ops, because what costs you there is the number of launches, not the arithmetic.
- If you only need resets at episode boundaries you control, leave `auto_reset=False` and call `env.reset_at` yourself with a mask — you then pay the reset cost only on the steps that need it.

## Running the benchmarks

```bash
python -m swarp.benchmark.throughput      # the full (n_envs, n_agents) grid
python -m swarp.benchmark.ablation        # cumulative optimization ablation, parity-gated
python -m swarp.benchmark.scenarios       # all 7 scenarios x model x lidar rays
python -m swarp.benchmark.compare_vmas    # vs VMAS: --metric {throughput,memory,both}
python -m swarp.benchmark.compare_sims    # vs VMAS, JaxMARL and CAMAR
```

[Benchmarks](benchmarks.md) has the numbers, the measurement conditions and the multi-venv setup the cross-simulator comparison needs.
