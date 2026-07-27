# Benchmarks

All figures below were measured on a single **NVIDIA GeForce RTX 3070 Laptop GPU** (8 GB,
sm_86), float32. They are only comparable to each other on that machine — treat them as
ratios and orders of magnitude, not absolutes.

The benchmark entry points (`wmas/benchmark/`):

| command | what it measures |
|---------|------------------|
| `python -m wmas.benchmark.throughput` | wmas alone: NavigationScenario hot path across a `(n_envs, n_agents)` grid |
| `python -m wmas.benchmark.compare_vmas` | wmas vs VMAS (throughput and/or peak device memory) |
| `python -m wmas.benchmark.compare_sims` | wmas vs VMAS vs JaxMARL vs CAMAR, one subprocess per simulator |
| `python -m wmas.benchmark.ablation` | cumulative optimization ablation of the same hot path, one feature per row, parity-gated |
| `python -m wmas.benchmark.scenarios` | all 7 scenarios × robot model × lidar rays: baseline vs optimized, parity-gated |

## Throughput

`python -m wmas.benchmark.throughput` steps the full NavigationScenario hot path
(dynamics, neighbor lists, soft collisions, obs/reward) under `torch.no_grad()` with
random actions kept on-device. Measurement conditions: **100 timed steps per config**
(after 10 warm-up steps), float32, `dt=0.05`, `substeps=1`, fused Warp obs/reward kernels
and no CUDA-graph capture (`use_graph=False`, which `throughput.py` pins so this table
stays comparable across commits — `Environment`'s own default is now `use_graph="auto"`,
i.e. capture *on* for a fused scenario on a CUDA device; `--graph` measures that), on the
RTX 3070 Laptop GPU:

```
  n_envs  n_agents   ms/step    env-steps/s   agent-steps/s
-----------------------------------------------------------
    1000         4      0.73      1,361,898       5,447,591
    1000        16      0.64      1,566,530      25,064,480
    1000        64      0.71      1,417,162      90,698,363
    4000         4      0.65      6,181,194      24,724,776
    4000        16      0.66      6,083,722      97,339,558
    4000        64      0.63      6,333,313     405,332,016
    8000         4      0.69     11,633,037      46,532,147
    8000        16      0.71     11,290,530     180,648,480
    8000        64      1.08      7,437,094     475,974,017
   16000         4      0.79     20,140,930      80,563,719
   16000        16      0.66     24,088,449     385,415,179
   16000        64      2.09      7,671,187     490,955,980
```

The tape-free hot path is allocation-free at steady state (recycled scratch plus a
ping-pong output buffer), which is what makes almost every config land on the same
~0.65–0.8 ms/step host floor: up to 16,000 envs × 16 agents the step is latency-bound,
not throughput-bound, and env-steps/s scales essentially linearly with `n_envs`. Only the
two largest configs (8,000–16,000 envs × 64 agents, i.e. ≥ 512k agents) become genuinely
GPU-bound in the neighbor/force kernels — and those are where `agent-steps/s` peaks at
~490 M.

Raising `n_agents` at fixed `n_envs` is close to free until that point, which is the
structural difference from per-entity simulators: the step is a fixed handful of Warp
kernels whose thread count grows with agents, not a Python loop whose *length* does.

## vs. VMAS

`python -m wmas.benchmark.compare_vmas` pits wmas against
[VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) on the navigation
scenario (the one both implement), across a `(n_envs, n_agents)` grid. Install the
comparison deps first: `uv pip install -e . --group bench` (pulls in `vmas`; the script
also falls back to a `./VectorizedMultiAgentSimulator` clone if present). Use
`--metric memory` for peak device memory and `--metric both` for both.

env-steps/s on the RTX 3070 Laptop GPU (float32, 60 timed steps). `vmas-lidar` is
VMAS's default navigation (collisions + 12-ray lidar); `vmas-simple` disables both as a
lidar-free lower bound:

```
 n_envs n_agents |          wmas |    vmas-lidar |   vmas-simple | wmas/lidar  wmas/simple
   16384        4 |    18,751,408 |     2,094,070 |     8,163,160 |      8.95x        2.30x
   16384       16 |     6,970,898 |       143,174 |     2,275,756 |     48.69x        3.06x
    1024        4 |     1,232,014 |       186,642 |       544,501 |      6.60x        2.26x
    1024       16 |     1,247,845 |        20,672 |       154,476 |     60.36x        8.08x
```

> **Note.** The `wmas` column here predates the current hot path (it was measured before
> whole-step CUDA-graph capture and the fused obs/reward work), so the ratios are a
> *lower bound* on today's margin. The VMAS columns are unaffected. Re-run the script for
> current figures.

wmas is ~6–9× faster than VMAS's default navigation at 4 agents and ~40–75× at 16 agents:
its whole step is ~3 fused Warp kernels regardless of agent count, whereas VMAS dispatches
per-entity (and O(entities²) pairwise) PyTorch ops in Python, and its lidar raycasts every
agent against every entity. Peak device memory is comparable to VMAS-with-lidar and small in
absolute terms (≤ ~300 MiB across the grid), so throughput — not memory — is the constraint.

**Parity caveat.** wmas is *API-compatible in spirit, not trajectory-compatible* with
VMAS. The dynamics (semi-implicit Euler, no drag, force-as-velocity-contribution for the
nonholonomic models), collision constants, and observation model (padded neighbor lists,
not lidar) differ by design, so identical actions do **not** reproduce VMAS trajectories.
The comparison above measures throughput/memory on the shared navigation task, nothing
more.

## vs. JaxMARL & CAMAR

`python -m wmas.benchmark.compare_sims` broadens the head-to-head to the JAX-based field:
wmas vs [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator),
[JaxMARL](https://github.com/flairox/jaxmarl) (`MPE_simple_spread_v3`, continuous
cooperative navigation), and [CAMAR](https://github.com/AIRI-Institute/CAMAR)
(`random_grid` + `HolonomicDynamic` continuous navigation). All four run the same
env-steps/s measurement on the one scenario they share — continuous 2D
navigation-to-goal with collision avoidance — anchored on wmas.

### Setting up the venvs

VMAS pins `numpy < 2` while JaxMARL/CAMAR are JAX (recent numpy), so the four cannot
share one venv; JaxMARL and CAMAR also pin different jax versions. Each simulator is
therefore installed into its own Python 3.12 venv and benchmarked in its own
subprocess (JAX and torch never share a process, so they don't fight over VRAM):

```bash
for v in .venv .venv-jaxmarl .venv-camar; do uv venv --python 3.12 $v; done
VIRTUAL_ENV=.venv         uv pip install -e . --group dev --group bench     # wmas + vmas
VIRTUAL_ENV=.venv-jaxmarl uv pip install -e . --group bench-jaxmarl         # jaxmarl
VIRTUAL_ENV=.venv-camar   uv pip install -e . --group bench-camar           # camar
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
.venv/bin/python -m wmas.benchmark.compare_sims --device cuda:0 \
    --envs 1024 4096 16384 --agents 3 16 \
    --python jaxmarl=.venv-jaxmarl/bin/python \
    --python camar=.venv-camar/bin/python
```

The launching interpreter (`.venv/bin/python` here) runs `wmas` and `vmas` in-process
by default; `--python SIM=PATH` overrides the interpreter for a given simulator, and
any sim without an override uses the launcher (or `--python-default PATH`). To compare
**just wmas vs JaxMARL**, restrict the sims and point JaxMARL at its venv:

```bash
.venv/bin/python -m wmas.benchmark.compare_sims --device cuda:0 \
    --sims wmas jaxmarl --agents 3 16 \
    --python jaxmarl=.venv-jaxmarl/bin/python
```

`--sims` picks which simulators to run (`wmas` is the anchor and should stay in);
`--agents`/`--envs` set the sweep. JaxMARL's `MPE_simple_spread_v3` honors any agent
count (`num_agents` is configurable), so no cell is skipped for an agent mismatch; if a
simulator ever cannot match the requested count, its number is flagged with `*` and the
realized count is listed under the table. You can drop the JaxMARL comparison entirely
by omitting it from `--sims`.

### wmas configurations

The three `wmas` entries are hot-path *configurations* of the same simulator, so the
benchmark doubles as an optimization ablation: `wmas-eager` (torch obs/reward, no
CUDA graph — the baseline), `wmas-fused` (fused Warp obs/reward kernels), and `wmas`
(fused **+** CUDA-graph capture). env-steps/s on the RTX 3070 Laptop GPU (float32,
60 timed steps):

```
  n_envs  n_agents |  wmas-eager  wmas-fused        wmas |  fused/eager  opt/eager
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
default (`use_graph="auto"` — on for a fused scenario on a CUDA device); the
[Throughput](#throughput) table above deliberately pins it off.

### wmas (optimized) vs the field — matched task & observations

To make it as apples-to-apples as a four-engine comparison can be, the **default**
`--sims` set controls for the two biggest asymmetries: all four run **obstacle-free**
open-field navigation with **raycasting-free, relative-position observations**:

| sim | config for the matched comparison |
|-----|-----------------------------------|
| `wmas` | optimized (fused Warp kernels + CUDA graph); neighbor-list obs, no obstacles |
| `vmas-nolidar` | VMAS navigation, `collisions=False` → no 12-ray lidar, relative-position obs |
| `jaxmarl` | `MPE_simple_spread_v3` continuous — relative-position obs (already raycast-free) |
| `camar` | **open arena** (`string_grid`, no obstacles, `frameskip=1`) instead of the default ~800-obstacle `random_grid` |

The JAX sims fold obs+reward into the scan carry so XLA cannot dead-code-eliminate
them, and JIT/XLA compile is excluded via warmup at the timed length. Both
`env-steps/s` and `agent-steps/s` (= env-steps/s × agents) are reported:

> **What this measures.** These figures are the *simulator* step only — physics +
> obs + reward — not a full RL loop. wmas folds that whole step (physics, neighbor
> query, and fused obs/reward) into a single CUDA graph, so a step is one graph
> replay with no per-launch host floor. In policy-in-the-loop RL the JAX sims
> `jit` the policy *and* env together into one XLA program, whereas wmas replays
> the env graph and runs the policy as separate launches; the end-to-end training
> throughput of each therefore depends on how the policy is compiled alongside the
> simulator, which this benchmark does not capture.

```
metric: env-steps/s
  n_envs  n_agents |         wmas  vmas-nolidar      jaxmarl        camar | vnl/w   jax/w  camar/w
    1024         3 |    4,717,415       633,305   13,731,386   11,644,687 | 0.13x   2.91x    2.47x
    4096         3 |   16,942,170     2,489,270   38,760,087   33,327,778 | 0.15x   2.29x    1.97x
   16384         3 |   63,640,378     9,946,801   65,213,783   67,508,356 | 0.16x   1.02x    1.06x
    1024        16 |    3,868,680       141,767    2,218,927    2,615,686 | 0.04x   0.57x    0.68x
    4096        16 |   15,595,567       549,767    2,231,675    3,353,155 | 0.04x   0.14x    0.22x
   16384        16 |   35,647,177     2,122,258    1,423,262    3,682,337 | 0.06x   0.04x    0.10x

metric: agent-steps/s
  n_envs  n_agents |         wmas  vmas-nolidar      jaxmarl        camar | vnl/w   jax/w  camar/w
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
  per-step launch floor. **wmas is latency-bound there** — its step floors at ~0.24 ms
  regardless of `n_envs` (1024→16384) *or* agent count (3→16), a Python-per-step-loop cost
  the JAX fused rollout doesn't pay. This is the real reason JAX leads at low occupancy.
- **They converge by 16384 envs** at 3 agents (all within ~6% — wmas 63.6 M, JaxMARL 65.2 M,
  CAMAR 67.5 M) as wmas's launch floor amortizes across more work.
- **wmas dominates at 16 agents** (JAX sims fall to 0.04–0.68×). wmas's step is a fixed
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
`--sims wmas vmas jaxmarl camar-grid` to compare each engine in its *native* setup instead.
