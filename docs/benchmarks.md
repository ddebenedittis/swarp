# Benchmarks

All figures below were measured on a single **NVIDIA GeForce RTX 3070 Laptop GPU** (8 GB,
sm_86), float32. They are only comparable to each other on that machine — treat them as
ratios and orders of magnitude, not absolutes.

The benchmark entry points (`swarp/benchmark/`):

| command | what it measures |
|---------|------------------|
| `python -m swarp.benchmark.throughput` | swarp alone: NavigationScenario hot path across a `(n_envs, n_agents)` grid |
| `python -m swarp.benchmark.compare_vmas` | swarp vs VMAS (throughput and/or peak device memory) |
| `python -m swarp.benchmark.compare_sims` | swarp vs VMAS vs JaxMARL vs CAMAR, one subprocess per simulator |
| `python -m swarp.benchmark.ablation` | cumulative optimization ablation of the same hot path, one feature per row, parity-gated |
| `python -m swarp.benchmark.scenarios` | all 10 scenarios × robot model × lidar rays: baseline vs optimized, parity-gated |
| `python -m swarp.benchmark.train_throughput` | end-to-end MAPPO training: [`swarp.rl`](training.md) vs TorchRL (as usually written and tuned), BenchMARL and rsl_rl |

## Throughput

`python -m swarp.benchmark.throughput` steps the full NavigationScenario hot path
(dynamics, neighbor lists, soft collisions, obs/reward) under `torch.no_grad()` with
random actions kept on-device. Measurement conditions: **100 timed steps per config**
(after 10 warm-up steps), float32, `dt=0.05`, `substeps=1`, fused Warp obs/reward kernels
and no CUDA-graph capture — run it as `python -m swarp.benchmark.throughput --no-graph`,
which pins capture off so this table stays comparable across commits. The plain command
leaves capture *on* (`throughput.py`'s default, matching `Environment`'s `use_graph="auto"`
for a fused scenario on a CUDA device) and reports the faster numbers. On the
RTX 3070 Laptop GPU:

```
  n_envs  n_agents   ms/step    env-steps/s   agent-steps/s
-----------------------------------------------------------
    1000         4      1.67        597,883       2,391,531
    1000        16      1.68        594,768       9,516,284
    1000        64      1.67        599,083      38,341,308
    4000         4      1.62      2,461,680       9,846,719
    4000        16      1.65      2,429,189      38,867,024
    4000        64      1.63      2,455,776     157,169,634
    8000         4      1.61      4,983,550      19,934,199
    8000        16      1.62      4,923,460      78,775,364
    8000        64      1.61      4,970,447     318,108,588
   16000         4      1.58     10,127,806      40,511,225
   16000        16      1.59     10,035,537     160,568,595
   16000        64      2.06      7,781,203     497,997,017
```

> **Reproducing these requires a cold GPU.** On this laptop the eager (graph-off) path is
> dominated by per-step launch overhead, which makes it acutely sensitive to SM boost
> state — a machine that has been running sustained GPU work for a while settles onto a
> lower clock than one measured fresh, and the eager path (which never keeps the SM busy
> enough on its own to re-boost) inherits whatever clock state it finds. The table above
> was reproducible to within ~3% across three separate runs, cold and after other GPU work,
> so it should be taken as this machine's current settled-clock number rather than a
> best-case one; treat any large deviation (a former revision of this table showed rows
> above 6 M env-steps/s at `4000×16` on a freshly-idle machine) as a clock-state difference,
> not a code regression, and re-measure both trees **interleaved** in the same session
> before concluding otherwise. The graph-on path does not have this problem: a step is one
> graph replay, so it stays close to its own number (`4000×16` was ~0.15 ms/step here)
> regardless of thermal/boost state.

The tape-free hot path is allocation-free at steady state (recycled scratch plus a
ping-pong output buffer), which is what makes almost every config land on the same
host-launch-bound floor on this run (~1.6–1.7 ms/step eager): up to 16,000 envs × 16
agents the step is latency-bound, not throughput-bound, and env-steps/s scales
essentially linearly with `n_envs`. Only the two largest configs (8,000–16,000 envs ×
64 agents, i.e. ≥ 512k agents) become genuinely GPU-bound in the neighbor/force kernels
— and those are where `agent-steps/s` peaks at ~500 M. The captured-graph path removes
the launch floor entirely (see below): the same grid at `--graph` runs `1000×4` at
0.08 ms/step versus 1.67 ms eager, ~21× faster, with the gap narrowing toward the
GPU-bound configs where launch overhead was never the constraint.

Raising `n_agents` at fixed `n_envs` is close to free until that point, which is the
structural difference from per-entity simulators: the step is a fixed handful of Warp
kernels whose thread count grows with agents, not a Python loop whose *length* does.

## Training throughput

`python -m swarp.benchmark.train_throughput` measures what a training run pays per iteration: rollout plus PPO update, logging and eval excluded.
Each configuration runs its trainer's real CLI in a subprocess, and the number is the median `wall_s` from the `metrics.csv` it writes, after the warm-up iterations.
`sim` is the simulator alone over the same rollout, and *sim share* is that time over the trainer's.
RTX 3070 Laptop GPU, float32, nothing else on the GPU.

Every trainer runs the same `TrainSpec` recipe (batch, epochs, minibatches, network width, learning rate, curriculum) through the library's own loop:

| config | trainer | what it is |
|---|---|---|
| `torchrl` | `examples/train_mappo.py` | MAPPO written with TorchRL the usual way: `Collector`, `ClipPPOLoss`, `ReplayBuffer`, eager update |
| `torchrl-tuned` | `examples/train_mappo_tuned.py` | the same MAPPO with TorchRL's own speed features: `cudagraph_policy` in the collector, the update step in `CudaGraphModule`, fused capturable Adam with an on-device skip, one host read per iteration |
| `torchrl-tuned-fast` | the same, `--compile --tf32` | plus `torch.compile` of the update and policy, and TF32 matmuls |
| `benchmarl` | `examples/train_benchmarl.py` | BenchMARL 1.5.2's `Experiment` with `MappoConfig`, swarp registered as a task like VMAS (own venv: BenchMARL pins `torchrl < 0.12`) |
| `rsl_rl` | `examples/train_rsl_rl.py` | rsl_rl 5.5's `OnPolicyRunner`, one rsl_rl env per agent, the joint observation as the critic's observation group |
| `eager` | `python -m swarp.rl --no-graphs` | the lean loop without CUDA graphs |
| `graphs` | `python -m swarp.rl` | rollout and update CUDA-graph captured |
| `fast` | `python -m swarp.rl --compile --tf32` | plus `torch.compile` of the loss and TF32 matmuls |

The baselines are not identical algorithms, because each library fixes some choices; every script lists its deviations from `swarp.rl` in its module docstring.
The ones that matter for speed: rsl_rl evaluates the critic once per agent on identical joint observations (`n_agents` times the critic work), and BenchMARL's MAPPO cannot normalise advantages.

```bash
uv pip install -e '.[torchrl,rsl-rl]' --group dev
UV_PROJECT_ENVIRONMENT=.venv-benchmarl uv sync --extra benchmarl --group dev
.venv/bin/python -m swarp.benchmark.train_throughput \
    --configs torchrl torchrl-tuned torchrl-tuned-fast benchmarl rsl_rl eager graphs fast \
    --python benchmarl=.venv-benchmarl/bin/python
```

At the recipes' own batch sizes:

```
scenario      n_envs   mb             config   ms/iter   env-steps/s vs torchrl sim share
-----------------------------------------------------------------------------------------
navigation       512    4            torchrl     232.0        70,621      1.00x      0.8%
navigation       512    4      torchrl-tuned     120.5       135,967      1.93x      1.6%
navigation       512    4 torchrl-tuned-fast     101.0       162,218      2.30x      1.9%
navigation       512    4          benchmarl     251.5        65,145      0.92x      0.8%
navigation       512    4             rsl_rl     162.0       101,136      1.43x      1.2%
navigation       512    4              eager     105.5       155,299      2.20x      1.8%
navigation       512    4             graphs      97.0       168,907      2.39x      2.0%
navigation       512    4               fast      72.0       227,556      3.22x      2.7%
navigation       512                     sim       1.9     8,502,622
giveway         1024    8            torchrl     463.5        70,697      1.00x      0.8%
giveway         1024    8      torchrl-tuned     242.0       135,405      1.92x      1.6%
giveway         1024    8 torchrl-tuned-fast     189.5       172,918      2.45x      2.0%
giveway         1024    8          benchmarl     479.5        68,338      0.97x      0.8%
giveway         1024    8             rsl_rl     315.5       103,861      1.47x      1.2%
giveway         1024    8              eager     204.0       160,627      2.27x      1.9%
giveway         1024    8             graphs     189.0       173,376      2.45x      2.0%
giveway         1024    8               fast     145.0       225,986      3.20x      2.6%
giveway         1024                     sim       3.8     8,601,597
```

Scaling `n_envs` on navigation with the minibatch held at 4096 env-steps (`--minibatch-samples 4096`, so the minibatch count grows with the batch):

```
scenario      n_envs   mb             config   ms/iter   env-steps/s vs torchrl sim share
-----------------------------------------------------------------------------------------
navigation      1024    8            torchrl     430.5        76,116      1.00x      0.5%
navigation      1024    8      torchrl-tuned     230.0       142,470      1.87x      1.0%
navigation      1024    8 torchrl-tuned-fast     186.0       176,172      2.31x      1.2%
navigation      1024    8          benchmarl     478.5        68,481      0.90x      0.5%
navigation      1024    8             rsl_rl     316.0       103,696      1.36x      0.7%
navigation      1024    8             graphs     192.0       170,667      2.24x      1.2%
navigation      1024    8               fast     141.0       232,397      3.05x      1.6%
navigation      1024                     sim       2.3    14,309,770
navigation      4096   32            torchrl    1648.0        79,534      1.00x      0.1%
navigation      4096   32      torchrl-tuned     842.5       155,575      1.96x      0.3%
navigation      4096   32 torchrl-tuned-fast     638.5       205,281      2.58x      0.4%
navigation      4096   32          benchmarl    1876.0        69,868      0.88x      0.1%
navigation      4096   32             rsl_rl    1299.5       100,863      1.27x      0.2%
navigation      4096   32             graphs     775.0       169,125      2.13x      0.3%
navigation      4096   32               fast     561.0       233,640      2.94x      0.4%
navigation      4096                     sim       2.4    53,930,683
navigation     16384  128            torchrl    7499.5        69,910      1.00x      0.1%
navigation     16384  128      torchrl-tuned    3606.5       145,373      2.08x      0.2%
navigation     16384  128 torchrl-tuned-fast    2496.5       210,009      3.00x      0.2%
navigation     16384  128          benchmarl    7049.5        74,372      1.06x      0.1%
navigation     16384  128             rsl_rl    5010.0       104,648      1.50x      0.1%
navigation     16384  128             graphs    3008.5       174,269      2.49x      0.2%
navigation     16384  128               fast    2192.0       239,182      3.42x      0.3%
navigation     16384                     sim       5.6    92,870,753
```

Reading it:

- **The physics is not the cost.** The simulator is at most 2.7% of any trainer's iteration, so every trainer over swarp is bounded by its own loop and its PPO update. With the overhead gone that update is GPU-bound: 8 epochs through two 256-wide MLPs, ~170 k env-steps/s in fp32 and ~235 k with TF32 matmuls and Inductor-fused elementwise ops, flat in `n_envs`.
- **Against TorchRL as usually written, `swarp.rl` is 2.1–2.5× faster (2.9–3.4× with `--compile --tf32`); against TorchRL with its own speed features on, 1.1–1.3× (1.1–1.4×).** Most of the first gap is overhead that TorchRL's APIs can remove: capturing the update alone takes the TorchRL trainer from ~240 to ~140 ms on navigation. What the tuned trainer still pays sits in TorchRL's env-step path, which its API does not let a user own: `EnvBase.step_and_maybe_reset` builds TensorDicts and reads `done.any()` on the host every step, and `CudaGraphModule` syncs the device before every replay, so collecting a 32-step rollout takes ~18 ms around ~2 ms of physics. Per minibatch it adds a sync, host-side `ReplayBuffer` sampling and a copy into the graph's static inputs. The gap narrows as the batch grows, because the GPU-bound update dominates both.
- **Tuning TorchRL took two non-obvious workarounds.** In TorchRL 0.13.3 with tensordict 0.13.0, `cudagraph_policy` with a `ProbabilisticActor` hands the collector the graph's static output buffers uncloned, so every step of a batch carries the last step's action and log-probability: training runs without an error and never learns. `TanhNormal` with float bounds issues a host-to-device copy that a CUDA graph cannot capture. `examples/train_mappo_tuned.py` documents both fixes.
- **BenchMARL runs at the speed of the untuned TorchRL trainer (0.9–1.1×)**, which it is built on; as shipped it uses none of TorchRL's graph or compile options.
- **rsl_rl sits in between (1.3–1.5×).** Its loop is lean but fully eager, with `.item()` reads per minibatch, and the agents-as-envs mapping makes it evaluate the critic `n_agents` times per env.
- **TorchRL is the one that suffers under CPU contention.** With two Isaac Sim processes busy on the same laptop, the TorchRL trainer ran 1.3–1.4× slower (navigation at 512 envs: 328 ms per iteration instead of 244 ms) and ran out of GPU memory at 16,384 envs, while `swarp.rl` moved by under 5%. Its loop is host-bound; the captured one is not.

Learning is unchanged by the speedup.
On `giveway` (1024 envs, 3000 iterations, seed 0) `swarp.rl` and the TorchRL trainer both reach full curriculum difficulty with a ≥ 0.7 episode solve rate at the same iteration (414 vs 415), and both final checkpoints score a greedy solve rate of 1.000 [0.998, 1.000] over 2048 envs with a median solve step of 39 (the scripted controller: 1.000 at step 48).
`swarp.rl` got there in 83 s of training wall-clock against 208 s, and finished the run in 604 s against 1845 s.
Every baseline learns `navigation` at the recipe: after 250 iterations, the greedy policy over 512 evaluation envs solves 1.000 [0.993, 1.000] for rsl_rl and for BenchMARL, and 0.982 [0.967, 0.991] for the tuned TorchRL trainer against 0.984 [0.969, 0.992] for the untuned one, from the same seed.
The tuned trainer's update also matches the untuned one numerically on fixed batches (`tests/examples/test_mappo_tuned.py`).

## vs. VMAS

`python -m swarp.benchmark.compare_vmas` pits swarp against
[VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) on the navigation
scenario (the one both implement), across a `(n_envs, n_agents)` grid. Install the
comparison deps first: `uv pip install -e '.[bench]'` (pulls in `vmas`; the script
also falls back to a `./VectorizedMultiAgentSimulator` clone if present). Use
`--metric memory` for peak device memory and `--metric both` for both.

env-steps/s on the RTX 3070 Laptop GPU (float32, 60 timed steps). `vmas-lidar` is
VMAS's default navigation (collisions + 12-ray lidar); `vmas-simple` disables both as a
lidar-free lower bound:

```
 n_envs n_agents |         swarp |    vmas-lidar |   vmas-simple | swarp/lidar  swarp/simple
   16384        4 |    18,751,408 |     2,094,070 |     8,163,160 |       8.95x         2.30x
   16384       16 |     6,970,898 |       143,174 |     2,275,756 |      48.69x         3.06x
    1024        4 |     1,232,014 |       186,642 |       544,501 |       6.60x         2.26x
    1024       16 |     1,247,845 |        20,672 |       154,476 |      60.36x         8.08x
```

> **Note.** The `swarp` column here predates the current hot path (it was measured before
> whole-step CUDA-graph capture and the fused obs/reward work), so the ratios are a
> *lower bound* on today's margin. The VMAS columns are unaffected. Re-run the script for
> current figures.

swarp is ~6–9× faster than VMAS's default navigation at 4 agents and ~40–75× at 16 agents:
its whole step is ~3 fused Warp kernels regardless of agent count, whereas VMAS dispatches
per-entity (and O(entities²) pairwise) PyTorch ops in Python, and its lidar raycasts every
agent against every entity. Peak device memory is comparable to VMAS-with-lidar and small in
absolute terms (≤ ~300 MiB across the grid), so throughput — not memory — is the constraint.

**Parity caveat.** swarp is *API-compatible in spirit, not trajectory-compatible* with
VMAS. The dynamics (semi-implicit Euler, no drag, force-as-velocity-contribution for the
nonholonomic models), collision constants, and observation model (padded neighbor lists,
not lidar) differ by design, so identical actions do **not** reproduce VMAS trajectories.
The comparison above measures throughput/memory on the shared navigation task, nothing
more.

## vs. JaxMARL & CAMAR

`python -m swarp.benchmark.compare_sims` broadens the head-to-head to the JAX-based field:
swarp vs [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator),
[JaxMARL](https://github.com/flairox/jaxmarl) (`MPE_simple_spread_v3`, continuous
cooperative navigation), and [CAMAR](https://github.com/AIRI-Institute/CAMAR)
(`random_grid` + `HolonomicDynamic` continuous navigation). All four run the same
env-steps/s measurement on the one scenario they share — continuous 2D
navigation-to-goal with collision avoidance — anchored on swarp.

### Setting up the venvs

VMAS pins `numpy < 2` while JaxMARL/CAMAR are JAX (recent numpy), so the four cannot
share one venv; JaxMARL and CAMAR also pin different jax versions. Each simulator is
therefore installed into its own Python 3.12 venv and benchmarked in its own
subprocess (JAX and torch never share a process, so they don't fight over VRAM):

```bash
for v in .venv .venv-jaxmarl .venv-camar; do uv venv --python 3.12 $v; done
VIRTUAL_ENV=.venv         uv pip install -e '.[bench]' --group dev            # swarp + vmas
VIRTUAL_ENV=.venv-jaxmarl uv pip install -e '.[bench-jaxmarl]'         # jaxmarl
VIRTUAL_ENV=.venv-camar   uv pip install -e '.[bench-camar]'           # camar
```

### Getting JaxMARL onto the GPU (do this — it silently runs on CPU otherwise)

JaxMARL pins an older `jax`, and its dependency resolution pulls the **CPU-only**
`jaxlib`. If you skip this step JAX falls back to CPU and the JaxMARL column is a
CPU number that is meaningless next to the GPU sims — so `compare_sims.py` **refuses
to report it**: the adapter asserts a GPU device and the cell prints `CPU-only`
instead of a bogus figure.

The fix is to install the CUDA build of `jaxlib` *at the exact `jax` version JaxMARL
already pinned* (a different version would break JaxMARL). Find that version, then
install the matching CUDA wheels into the JaxMARL venv only:

```bash
# 1. discover the jax version JaxMARL installed (e.g. 0.4.38)
.venv-jaxmarl/bin/python -c "import jax; print(jax.__version__)"

# 2. install the CUDA-enabled build at that same version (adjust cuda12 -> your CUDA)
VIRTUAL_ENV=.venv-jaxmarl uv pip install "jax[cuda12]==0.4.38"

# 3. verify JAX now sees the GPU (must print a CudaDevice, not CpuDevice)
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv-jaxmarl/bin/python -c "import jax; print(jax.devices())"
```

`compare_sims.py` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` in each JAX child so JAX
grows VRAM on demand instead of grabbing ~75% up front; CAMAR (`bench-camar`) ships a
CUDA `jaxlib` already and needs no such fix.

### Running it

One combined command drives all three subprocesses via a per-sim interpreter map and
prints a single table (the numpy split lives entirely at the subprocess boundary):

```bash
.venv/bin/python -m swarp.benchmark.compare_sims --device cuda:0 \
    --envs 1024 4096 16384 --agents 3 16 \
    --python jaxmarl=.venv-jaxmarl/bin/python \
    --python camar=.venv-camar/bin/python
```

The launching interpreter (`.venv/bin/python` here) runs `swarp` and `vmas` in-process
by default; `--python SIM=PATH` overrides the interpreter for a given simulator, and
any sim without an override uses the launcher (or `--python-default PATH`). To compare
**just swarp vs JaxMARL**, restrict the sims and point JaxMARL at its venv:

```bash
.venv/bin/python -m swarp.benchmark.compare_sims --device cuda:0 \
    --sims swarp jaxmarl --agents 3 16 \
    --python jaxmarl=.venv-jaxmarl/bin/python
```

`--sims` picks which simulators to run (`swarp` is the anchor and should stay in);
`--agents`/`--envs` set the sweep. JaxMARL's `MPE_simple_spread_v3` honors any agent
count (`num_agents` is configurable), so no cell is skipped for an agent mismatch; if a
simulator ever cannot match the requested count, its number is flagged with `*` and the
realized count is listed under the table. You can drop the JaxMARL comparison entirely
by omitting it from `--sims`.

### swarp configurations

The three `swarp` entries are hot-path *configurations* of the same simulator, so the
benchmark doubles as an optimization ablation: `swarp-eager` (torch obs/reward, no
CUDA graph — the baseline), `swarp-fused` (fused Warp obs/reward kernels), and `swarp`
(fused **+** CUDA-graph capture). env-steps/s on the RTX 3070 Laptop GPU (float32,
60 timed steps):

```
  n_envs  n_agents | swarp-eager swarp-fused       swarp |  fused/eager  opt/eager
    4096         3 |   5,838,664   7,877,854  15,945,297 |       1.35x      2.73x
   16384         3 |  21,772,740  30,814,526  61,718,330 |       1.42x      2.83x
    4096        16 |   5,998,275   6,783,331  15,094,932 |       1.13x      2.52x
   16384        16 |   6,858,433  28,191,165  35,637,271 |       4.11x      5.20x
```

Fusing the obs/reward layer into Warp kernels is worth ~1.1–1.4× on its own; the
CUDA graph (which elides per-step kernel-launch overhead) roughly doubles it again,
for ~2.5–5× end-to-end over the eager baseline. The graph's win grows with batch size,
where launch overhead is the binding constraint.

These numbers are higher than the [Throughput](#throughput) table's because this task is
obstacle-free open-field navigation (matched to what the other simulators do), and
because the top configuration adds CUDA-graph capture. That is now `Environment`'s
default (`use_graph="auto"` — on for a fused scenario on a CUDA device), and
`throughput.py`'s; the [Throughput](#throughput) table above deliberately pins it off
with `--no-graph`.

### swarp (optimized) vs the field — matched task & observations

To make it as apples-to-apples as a four-engine comparison can be, the **default**
`--sims` set controls for the two biggest asymmetries: all four run **obstacle-free**
open-field navigation with **raycasting-free, relative-position observations**:

| sim | config for the matched comparison |
|-----|-----------------------------------|
| `swarp` | optimized (fused Warp kernels + CUDA graph); neighbor-list obs, no obstacles |
| `vmas-nolidar` | VMAS navigation, `collisions=False` → no 12-ray lidar, relative-position obs |
| `jaxmarl` | `MPE_simple_spread_v3` continuous — relative-position obs (already raycast-free) |
| `camar` | **open arena** (`string_grid`, no obstacles, `frameskip=0` — one world step per step, like `substeps=1`) instead of the default ~800-obstacle `random_grid` |

The arenas are *not* the same size: CAMAR's 12×12 all-free `string_grid` is a ~1.2×1.2
world fixed by its map generator, while the swarp adapter uses `world_size = max(1,
sqrt(n_agents)/4)` with bounds at ±`world_size` — 2×2 up to 16 agents, 4×4 at 64. Agent
density therefore differs, which matters for the neighbor/contact half of the step; there
is no CAMAR knob that would match it exactly.

The JAX sims fold obs+reward into the scan carry so XLA cannot dead-code-eliminate
them, and JIT/XLA compile is excluded via warmup at the timed length. Both
`env-steps/s` and `agent-steps/s` (= env-steps/s × agents) are reported:

> **What this measures.** These figures are the *simulator* step only — physics +
> obs + reward — not a full RL loop. swarp folds that whole step (physics, neighbor
> query, and fused obs/reward) into a single CUDA graph, so a step is one graph
> replay with no per-launch host floor. In policy-in-the-loop RL the JAX sims
> `jit` the policy *and* env together into one XLA program, whereas swarp replays
> the env graph and runs the policy as separate launches; the end-to-end training
> throughput of each therefore depends on how the policy is compiled alongside the
> simulator, which this benchmark does not capture.

```
metric: env-steps/s
  n_envs  n_agents |        swarp  vmas-nolidar      jaxmarl        camar | vnl/s   jax/s  camar/s
    1024         3 |    4,717,415       633,305   13,731,386   11,644,687 | 0.13x   2.91x    2.47x
    4096         3 |   16,942,170     2,489,270   38,760,087   33,327,778 | 0.15x   2.29x    1.97x
   16384         3 |   63,640,378     9,946,801   65,213,783   67,508,356 | 0.16x   1.02x    1.06x
    1024        16 |    3,868,680       141,767    2,218,927    2,615,686 | 0.04x   0.57x    0.68x
    4096        16 |   15,595,567       549,767    2,231,675    3,353,155 | 0.04x   0.14x    0.22x
   16384        16 |   35,647,177     2,122,258    1,423,262    3,682,337 | 0.06x   0.04x    0.10x

metric: agent-steps/s
  n_envs  n_agents |        swarp  vmas-nolidar      jaxmarl        camar | vnl/s   jax/s  camar/s
    1024         3 |   14,152,244     1,899,916   41,194,157   34,934,061 | 0.13x   2.91x    2.47x
    4096         3 |   50,826,510     7,467,809  116,280,262   99,983,333 | 0.15x   2.29x    1.97x
   16384         3 |  190,921,134    29,840,404  195,641,349  202,525,067 | 0.16x   1.02x    1.06x
    1024        16 |   61,898,884     2,268,278   35,502,837   41,850,981 | 0.04x   0.57x    0.68x
    4096        16 |  249,529,074     8,796,274   35,706,801   53,650,474 | 0.04x   0.14x    0.22x
   16384        16 |  570,354,838    33,956,129   22,772,199   58,917,397 | 0.06x   0.04x    0.10x
```
(Ratios are identical in both tables — with agent counts matched across sims, the
`×agents` factor cancels. `agent-steps/s` is there for the absolute magnitudes.)

Reading it, once the task and obs are matched:

- **The two JAX sims (JaxMARL, CAMAR) win at few agents + low envs** (~2–3× at 3 agents,
  1024–4096 envs), because they run the whole rollout as one jitted `lax.scan` with no
  per-step launch floor. **swarp is latency-bound there** — its step floors at ~0.24 ms
  regardless of `n_envs` (1024→16384) *or* agent count (3→16), a Python-per-step-loop cost
  the JAX fused rollout doesn't pay. This is the real reason JAX leads at low occupancy.
- **They converge by 16384 envs** at 3 agents (all within ~6% — swarp 63.6 M, JaxMARL 65.2 M,
  CAMAR 67.5 M) as swarp's launch floor amortizes across more work.
- **swarp dominates at 16 agents** (JAX sims fall to 0.04–0.68×). swarp's step is a fixed
  handful of fused Warp kernels, nearly insensitive to agent count — its `agent-steps/s`
  climbs from ~14 M (3 agents) to ~570 M (16 agents), ~40×, doing far more work in about
  the same wall time — whereas MPE/CAMAR per-step cost grows with agents.
- **`vmas-nolidar` is consistently slowest** (0.04–0.16×): a Python per-entity step loop
  that neither the JAX scan nor Warp kernels have to pay.

**The default set is now genuinely comparable; the earlier "CAMAR is ~200× slower"
result was almost entirely CAMAR's ~800-obstacle default map** — the same open-arena
CAMAR here is 20–120× faster than that (`--sims camar-grid` reproduces the heavy native
task). Residual, deliberate differences: exact obs dimensionality still differs per sim;
`vmas-nolidar` also loses agent-agent collision physics (VMAS gates collisions and lidar
on one flag); dynamics/collision constants differ so trajectories won't match. Use
`--sims swarp vmas jaxmarl camar-grid` to compare each engine in its *native* setup instead.
