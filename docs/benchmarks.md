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
| `python -m swarp.benchmark.scenarios` | all 11 scenarios × robot model × lidar rays: baseline vs optimized, parity-gated |
| `python -m swarp.benchmark.train_throughput` | end-to-end MAPPO training: [`swarp.rl`](training.md) vs TorchRL (as usually written and tuned), BenchMARL and rsl_rl |

## Throughput

`python -m swarp.benchmark.throughput` steps the full NavigationScenario hot path (dynamics, neighbor lists, soft collisions, obs/reward) under `torch.no_grad()` with random actions kept on-device.
Measurement conditions: **100 timed steps per config** (after 10 warm-up steps), float32, `dt=0.05`, `substeps=1`, fused Warp obs/reward kernels and no CUDA-graph capture.
Run it as `python -m swarp.benchmark.throughput --no-graph`, which pins capture off so this table stays comparable across commits.
The plain command leaves capture *on* (`throughput.py`'s default, matching `Environment`'s `use_graph="auto"` for a fused scenario on a CUDA device) and reports the faster numbers.
On the RTX 3070 Laptop GPU:

```
  n_envs  n_agents   ms/step    env-steps/s   agent-steps/s
-----------------------------------------------------------
    1000         4      0.29      3,398,312      13,593,246
    1000        16      0.36      2,753,071      44,049,134
    1000        64      0.36      2,803,282     179,410,070
    4000         4      0.29     13,747,433      54,989,730
    4000        16      0.35     11,334,742     181,355,869
    4000        64      0.58      6,863,382     439,256,422
    8000         4      0.37     21,737,790      86,951,160
    8000        16      0.36     22,375,841     358,013,457
    8000        64      1.06      7,541,604     482,662,649
   16000         4      0.35     45,393,422     181,573,687
   16000        16      0.47     33,916,430     542,662,884
   16000        64      2.14      7,490,570     479,396,449
```

With capture on (the plain command) the same grid runs:

```
  n_envs  n_agents   ms/step    env-steps/s   agent-steps/s
-----------------------------------------------------------
    1000         4      0.03     36,852,108     147,408,430
    1000        16      0.06     16,372,076     261,953,211
    1000        64      0.24      4,150,284     265,618,172
    4000         4      0.05     82,862,931     331,451,723
    4000        16      0.15     27,297,834     436,765,351
    4000        64      0.62      6,475,123     414,407,887
    8000         4      0.08    100,176,473     400,705,894
    8000        16      0.26     30,806,414     492,902,626
    8000        64      1.08      7,382,113     472,455,250
   16000         4      0.11    145,785,598     583,142,392
   16000        16      0.41     38,818,634     621,098,151
   16000        64      1.94      8,227,839     526,581,706
```

> **The eager table depends on the GPU's clock state.**
> The eager (graph-off) path is dominated by per-step launch overhead, which makes it sensitive to SM boost state: the first run of the session above, on a cold GPU, read 0.22–0.27 ms/step on the rows that read 0.29–0.37 ms here, after 40 minutes of other GPU work.
> The tables above are the settled-clock numbers; re-measure both trees **interleaved** in the same session before reading a large deviation as a code regression.
> The graph-on path does not have this problem: a step is one graph replay, so it stays close to its own number regardless of thermal and boost state.

The tape-free hot path is allocation-free at steady state (recycled scratch plus a ping-pong output buffer), which is why every config up to 16,000 envs × 16 agents lands on the same ~0.3–0.5 ms/step host-launch floor eager: the step is latency-bound, not throughput-bound, and env-steps/s scales almost linearly with `n_envs`.
Only the 64-agent configs from 4,000 envs up become GPU-bound in the neighbor and force kernels, which is where `agent-steps/s` peaks at ~480 M.
Capture removes the launch floor: `1000×4` runs at 0.03 ms/step against 0.29 ms eager, ~10× faster, and the 4-agent rows reach 146 M env-steps/s at 16,000 envs, with the gap closing toward the GPU-bound configs where launch overhead was never the constraint.

Raising `n_agents` at fixed `n_envs` is close to free until that point, which is the structural difference from per-entity simulators: the step is a fixed handful of Warp kernels whose thread count grows with agents, not a Python loop whose *length* does.

### Fused substep chain

The [fused substep chain](performance.md#fused-substep-chain) and [lean launches](performance.md#lean-launches), against the commit before them.
NavigationScenario, graph on, float32, `dt=0.05`, random actions, RTX 3070 Laptop GPU.
Each arm ran in its own process, best of 3 timed windows of ~0.3 s, in 6 alternating pairs; the table gives the median µs/step and the range of the per-pair ratios.
The `throughput.py` grid (one substep, `auto_reset=False`) comes first, then the shapes the `swarp.rl` recipes train at (3–5 agents, 4–8 substeps, `auto_reset=True`).

| n_envs × agents × substeps | before | after | change | pair ratios |
|---|---|---|---|---|
| 1000 × 4 × 1 | 31.2 | 24.0 | −23% | 0.67–1.14 |
| 1000 × 16 × 1 | 48.0 | 46.4 | −3% | 0.96–0.98 |
| 1000 × 64 × 1 | 184.8 | 181.0 | −2% | 0.97–0.99 |
| 4000 × 4 × 1 | 40.3 | 36.3 | −10% | 0.89–0.92 |
| 4000 × 16 × 1 | 120.3 | 118.5 | −1% | 0.98–0.99 |
| 4000 × 64 × 1 | 520.9 | 518.0 | −1% | 0.99–1.01 |
| 8000 × 4 × 1 | 65.5 | 63.3 | −3% | 0.96–0.97 |
| 8000 × 16 × 1 | 219.6 | 219.5 | 0% | 0.99–1.00 |
| 8000 × 64 × 1 | 982.7 | 979.8 | 0% | 0.98–1.00 |
| 16000 × 4 × 1 | 107.2 | 106.6 | −1% | 0.99–1.00 |
| 16000 × 16 × 1 | 419.6 | 417.1 | −1% | 0.99–1.01 |
| 16000 × 64 × 1 | 1996.9 | 1984.7 | −1% | 0.99–1.00 |
| 512 × 4 × 4, auto-reset | 54.0 | 32.2 | −40% | 0.59–0.61 |
| 1024 × 3 × 8, auto-reset | 82.7 | 38.3 | −54% | 0.46–0.47 |
| 1024 × 4 × 8, auto-reset | 84.9 | 40.4 | −52% | 0.47–0.48 |
| 1024 × 5 × 8, auto-reset | 89.0 | 44.8 | −50% | 0.50–0.51 |
| 4096 × 4 × 4, auto-reset | 77.7 | 50.6 | −35% | 0.64–0.66 |
| 4096 × 8 × 4, auto-reset | 122.4 | 96.5 | −21% | 0.79–0.79 |
| 16384 × 4 × 4, auto-reset | 178.1 | 147.5 | −17% | 0.83–0.84 |

The one-substep rows with 16 or more agents run the layered pipeline under `"auto"`, so they move only by the lean launches; forcing the fused chain there (`enable_fused_substeps=True`) was up to 29% *slower* in the sweep that set the routing rule.

End to end the gain is small, because the simulator is a small part of a training iteration.
`python -m swarp.rl` for 60 iterations at the recipes' sizes went from 99.5 to 98.5 ms/iteration on navigation, 193 to 192.5 on giveway and 224 to 223 on caging; the PPO update is ~90% of an iteration.
The runs are bit-identical: every learning column of `metrics.csv` (reward, solve rate, losses, entropy) matched the previous commit's at every iteration, since the chain sums the same terms in the same order.

### Ring road

`ringroad` carries most of its work outside the engine: per step, its fused pass projects every vehicle onto the ring, runs an O(N²) leader search twice and integrates the IDM humans, for 22 vehicles per env at the default size.
`python -m swarp.benchmark.scenarios --scenarios ringroad --envs 1024 4096 16000 --agents 1 16 64 --lidar-rays 0 --steps 300`, RTX 3070 Laptop GPU, ms/step.
The ring holds `max(22, n_agents + 1)` vehicles, so the 64-agent rows simulate 65.
Every row ran captured (`graph`):

| n_envs | RL cars (vehicles) | baseline (torch obs/reward, eager) | optimized (fused + graph) | speedup |
|---|---|---|---|---|
| 1024 | 1 (22) | 1.138 | 0.034 | 33.6× |
| 1024 | 16 (22) | 1.223 | 0.082 | 14.9× |
| 1024 | 64 (65) | 3.378 | 0.341 | 9.9× |
| 4096 | 1 (22) | 1.816 | 0.058 | 31.2× |
| 4096 | 16 (22) | 1.914 | 0.138 | 13.9× |
| 4096 | 64 (65) | 12.632 | 0.677 | 18.7× |
| 16000 | 1 (22) | 6.131 | 0.208 | 29.5× |
| 16000 | 16 (22) | 6.708 | 0.605 | 11.1× |
| 16000 | 64 (65) | 48.524 | 3.330 | 14.6× |

At the training recipe's size (1024 envs, 1 RL car among 22) the step costs 0.034 ms, about 30 M env-steps/s.
A whole `swarp.rl` iteration (1024 envs × 32 steps, rollout plus PPO update) takes about 94 ms, so the update, not the ring, is the cost.

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
navigation       512    4            torchrl     223.5        73,306      1.00x      0.5%
navigation       512    4      torchrl-tuned     129.0       127,008      1.73x      0.8%
navigation       512    4 torchrl-tuned-fast     107.0       153,121      2.09x      1.0%
navigation       512    4          benchmarl     273.5        59,905      0.82x      0.4%
navigation       512    4             rsl_rl     164.0        99,902      1.36x      0.6%
navigation       512    4              eager     107.0       153,121      2.09x      1.0%
navigation       512    4             graphs      97.0       168,907      2.30x      1.1%
navigation       512    4               fast      72.0       227,556      3.10x      1.4%
navigation       512                     sim       1.0    15,807,180
giveway         1024    8            torchrl     451.5        72,576      1.00x      0.5%
giveway         1024    8      torchrl-tuned     242.0       135,405      1.87x      0.9%
giveway         1024    8 torchrl-tuned-fast     190.0       172,463      2.38x      1.2%
giveway         1024    8          benchmarl     491.5        66,669      0.92x      0.5%
giveway         1024    8             rsl_rl     315.0       104,025      1.43x      0.7%
giveway         1024    8              eager     200.5       163,431      2.25x      1.1%
giveway         1024    8             graphs     188.0       174,298      2.40x      1.2%
giveway         1024    8               fast     144.0       227,556      3.14x      1.6%
giveway         1024                     sim       2.3    14,312,670
```

Scaling `n_envs` on navigation with the minibatch held at 4096 env-steps (`--minibatch-samples 4096`, so the minibatch count grows with the batch):

```
scenario      n_envs   mb             config   ms/iter   env-steps/s vs torchrl sim share
-----------------------------------------------------------------------------------------
navigation      1024    8            torchrl     428.0        76,561      1.00x      0.3%
navigation      1024    8      torchrl-tuned     233.0       140,635      1.84x      0.6%
navigation      1024    8 torchrl-tuned-fast     189.0       173,376      2.26x      0.7%
navigation      1024    8          benchmarl     503.5        65,080      0.85x      0.3%
navigation      1024    8             rsl_rl     320.0       102,400      1.34x      0.4%
navigation      1024    8             graphs     192.0       170,667      2.23x      0.7%
navigation      1024    8               fast     140.0       234,057      3.06x      1.0%
navigation      1024                     sim       1.4    23,262,798
navigation      4096   32            torchrl    1611.5        81,335      1.00x      0.1%
navigation      4096   32      torchrl-tuned     833.5       157,255      1.93x      0.2%
navigation      4096   32 torchrl-tuned-fast     641.0       204,480      2.51x      0.3%
navigation      4096   32          benchmarl    1837.5        71,332      0.88x      0.1%
navigation      4096   32             rsl_rl    1298.5       100,941      1.24x      0.1%
navigation      4096   32             graphs     777.5       168,581      2.07x      0.2%
navigation      4096   32               fast     559.0       234,476      2.88x      0.3%
navigation      4096                     sim       1.6    81,093,956
navigation     16384  128            torchrl    6251.5        83,866      1.00x      0.1%
navigation     16384  128      torchrl-tuned    3287.5       159,479      1.90x      0.2%
navigation     16384  128 torchrl-tuned-fast    2508.0       209,046      2.49x      0.2%
navigation     16384  128          benchmarl    7785.5        67,342      0.80x      0.1%
navigation     16384  128             rsl_rl    5644.0        92,893      1.11x      0.1%
navigation     16384  128             graphs    3326.0       157,633      1.88x      0.2%
navigation     16384  128               fast    2221.0       236,059      2.81x      0.2%
navigation     16384                     sim       5.1   101,878,489
```

The 16384-env row is the one the GPU's temperature moves: run on its own after a two-minute cooldown, `graphs` takes 3000–3004 ms and `fast` 2192–2195 ms, against 3326 and 2221 ms at the end of the row, after half an hour of the other trainers; the previous commit times the same (2999–3023 and 2193–2200 ms).

Reading it:

- **The physics is not the cost.** The simulator is at most 1.6% of any trainer's iteration, so every trainer over swarp is bounded by its own loop and its PPO update. With the overhead gone that update is GPU-bound: 8 epochs through two 256-wide MLPs, ~170 k env-steps/s in fp32 and ~230 k with TF32 matmuls and Inductor-fused elementwise ops, flat in `n_envs`.
- **Against TorchRL as usually written, `swarp.rl` is 1.9–2.4× faster (2.8–3.1× with `--compile --tf32`); against TorchRL with its own speed features on, 1.0–1.3× (1.1–1.5×).** Most of the first gap is overhead that TorchRL's APIs can remove: capturing the update alone takes the TorchRL trainer from ~225 to ~130 ms on navigation. What the tuned trainer still pays sits in TorchRL's env-step path, which its API does not let a user own: `EnvBase.step_and_maybe_reset` builds TensorDicts and reads `done.any()` on the host every step, and `CudaGraphModule` syncs the device before every replay, so collecting a 32-step rollout takes ~18 ms around ~1 ms of physics. Per minibatch it adds a sync, host-side `ReplayBuffer` sampling and a copy into the graph's static inputs. The gap narrows as the batch grows, because the GPU-bound update dominates both.
- **Tuning TorchRL took two non-obvious workarounds.** In TorchRL 0.13.3 with tensordict 0.13.0, `cudagraph_policy` with a `ProbabilisticActor` hands the collector the graph's static output buffers uncloned, so every step of a batch carries the last step's action and log-probability: training runs without an error and never learns. TorchRL 0.14.0 fixed this ([pytorch/rl#3002](https://github.com/pytorch/rl/issues/3002), [#4143](https://github.com/pytorch/rl/pull/4143)), as did tensordict 0.15.0 for `CudaGraphModule` itself ([pytorch/tensordict#1999](https://github.com/pytorch/tensordict/pull/1999)). `TanhNormal` with float bounds issues a host-to-device copy that a CUDA graph cannot capture, still so in TorchRL 0.14.1. `examples/train_mappo_tuned.py` documents both fixes. The tables above were measured on TorchRL 0.14.1 with tensordict 0.14.3 (BenchMARL on its pinned TorchRL 0.11.1); an earlier run on TorchRL 0.13.3 timed both TorchRL trainers within 4% of 0.14.1.
- **BenchMARL runs a little behind the untuned TorchRL trainer (0.80–0.92×)**, which it is built on; as shipped it uses none of TorchRL's graph or compile options.
- **rsl_rl sits in between (1.1–1.4×).** Its loop is lean but fully eager, with `.item()` reads per minibatch, and the agents-as-envs mapping makes it evaluate the critic `n_agents` times per env.
- **TorchRL is the one that suffers under CPU contention.** With two Isaac Sim processes busy on the same laptop, the TorchRL trainer ran 1.3–1.4× slower (navigation at 512 envs: 328 ms per iteration instead of 244 ms) and ran out of GPU memory at 16,384 envs, while `swarp.rl` moved by under 5%. Its loop is host-bound; the captured one is not.

Learning is unchanged by the speedup.
On `giveway` (1024 envs, the full 3000-iteration recipe, seeds 0–2) every trainer learns the task, so the time to solve is the iteration time:

| trainer | full difficulty at iteration | wall-clock to get there | whole run | greedy solve rate (2048 envs) | median solve step |
|---|---|---|---|---|---|
| `swarp.rl` | 413–414 | **78–80 s** | 587 s | 1.000 [0.998, 1.000] | 38–41 |
| TorchRL, tuned | 414–416 | 101–115 s | 751 s | 1.000 | 39–41 |
| rsl_rl | 413–414 | 132–133 s | 957 s | 1.000 | 38–39 |
| TorchRL, as usually written | 414–415 | 174–176 s | 1280 s | 1.000 | 37–39 |
| BenchMARL | 416–417 | 206–209 s | 1462 s | 1.000 | 42 |

Every run reaches full difficulty with a windowed episode solve rate above 0.95 at that same iteration, and every final checkpoint solves all 2048 evaluation envs (the scripted controller: 1.000 at step 48); wall-clock excludes evaluation and is the range over the three seeds, and whole-run times are means.
This measures speed, not sample efficiency: the recipe's curriculum raises difficulty by at most 1/400 per iteration, so no run can reach full difficulty before iteration 400, and every trainer clears the gate from about iteration 15 on.
A comparison of sample efficiency needs a task where learning, not the curriculum, sets the pace.
Every baseline learns `navigation` at the recipe: after 250 iterations, the greedy policy over 512 evaluation envs solves 1.000 [0.993, 1.000] for rsl_rl, BenchMARL and both TorchRL trainers, from the same seed.
The tuned trainer's update also matches the untuned one numerically on fixed batches (`tests/examples/test_mappo_tuned.py`).

## vs. VMAS

`python -m swarp.benchmark.compare_vmas` pits swarp against
[VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) on the navigation
scenario (the one both implement), across a `(n_envs, n_agents)` grid. Install the
comparison deps first: `uv pip install -e '.[bench]'` (pulls in `vmas`; the script
also falls back to a `./VectorizedMultiAgentSimulator` clone if present). Use
`--metric memory` for peak device memory and `--metric both` for both.

env-steps/s on the RTX 3070 Laptop GPU (float32, `--steps 300 --warmup 20`, swarp with its defaults: fused kernels and whole-step capture).
`vmas-lidar` is VMAS's default navigation (collisions + 12-ray lidar); `vmas-simple` disables both as a lidar-free lower bound:

```
 n_envs n_agents |         swarp |    vmas-lidar |   vmas-simple | swarp/lidar swarp/simple
  16384        4 |   141,544,015 |     1,924,356 |     6,805,183 |      73.55x       20.80x
   1024        4 |    51,060,511 |       157,802 |       447,798 |     323.57x      114.03x
  16384       16 |    33,083,440 |       134,244 |     1,787,329 |     246.44x       18.51x
   1024       16 |    17,857,796 |        17,523 |       117,271 |    1019.08x      152.28x
```

swarp is ~70–320× faster than VMAS's default navigation at 4 agents and ~250–1000× at 16 agents, and ~20–150× faster than VMAS with collisions and lidar off.
Its whole step is one CUDA-graph replay of a handful of Warp kernels regardless of agent count, whereas VMAS dispatches per-entity (and O(entities²) pairwise) PyTorch ops in Python, and its lidar raycasts every agent against every entity.
The ratio is largest at small batches, where VMAS's per-step Python cost is spread over the fewest envs.
Peak device memory (`--metric memory`, same grid) is small for both and no larger for swarp where it matters: 56 MiB against VMAS-with-lidar's 78 MiB at 16384 × 4 and 142 MiB against 334 MiB at 16384 × 16, with 32–52 MiB at 1024 envs, mostly fixed overhead.
Throughput, not memory, is the constraint.

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

The three `swarp` entries are hot-path *configurations* of the same simulator, so the benchmark doubles as an optimization ablation: `swarp-eager` (torch obs/reward, no CUDA graph — the baseline), `swarp-fused` (fused Warp obs/reward kernels), and `swarp` (fused **+** CUDA-graph capture).
env-steps/s on the RTX 3070 Laptop GPU (float32, `--steps 300 --warmup 20`):

```
  n_envs  n_agents | swarp-eager swarp-fused       swarp |  fused/eager  opt/eager
    4096         3 |   7,649,544  15,980,008  86,003,167 |       2.09x     11.24x
   16384         3 |  25,962,701  47,203,892 145,339,064 |       1.82x      5.60x
    4096        16 |   6,020,546  14,022,673  26,341,168 |       2.33x      4.38x
   16384        16 |   6,538,976  29,355,880  32,409,682 |       4.49x      4.96x
```

Fusing the obs/reward layer into Warp kernels is worth ~1.8–4.5× on its own, and the CUDA graph adds another 1.1–5.4× on top, for ~4.4–11× end to end over the eager baseline.
The graph's share is largest where the step is shortest (5.4× at 4096 × 3, where it also folds the [fused substep chain](performance.md#fused-substep-chain)'s single launch) and smallest at 16384 × 16, where the kernels themselves keep the GPU busy.

These numbers are higher than the [Throughput](#throughput) table's because this task is obstacle-free open-field navigation (matched to what the other simulators do), and because the top configuration adds CUDA-graph capture.
That is `Environment`'s default (`use_graph="auto"` — on for a fused scenario on a CUDA device), and `throughput.py`'s; the first [Throughput](#throughput) table deliberately pins it off with `--no-graph`.

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
    1024         3 |   40,049,629       604,766   16,304,942   18,960,345 | 0.02x   0.41x    0.47x
    4096         3 |   86,018,717     2,329,720   47,420,062   53,368,329 | 0.03x   0.55x    0.62x
   16384         3 |  129,699,246     9,286,969   61,406,922   98,370,008 | 0.07x   0.47x    0.76x
    1024        16 |   14,426,112       129,181    1,853,129    4,068,782 | 0.01x   0.13x    0.28x
    4096        16 |   27,344,400       521,612    1,077,086    5,580,428 | 0.02x   0.04x    0.20x
   16384        16 |   32,597,571     2,172,560      694,972    5,526,954 | 0.07x   0.02x    0.17x

metric: agent-steps/s
  n_envs  n_agents |        swarp  vmas-nolidar      jaxmarl        camar | vnl/s   jax/s  camar/s
    1024         3 |  120,148,888     1,814,299   48,914,827   56,881,034 | 0.02x   0.41x    0.47x
    4096         3 |  258,056,152     6,989,161  142,260,186  160,104,986 | 0.03x   0.55x    0.62x
   16384         3 |  389,097,738    27,860,908  184,220,765  295,110,025 | 0.07x   0.47x    0.76x
    1024        16 |  230,817,791     2,066,903   29,650,070   65,100,512 | 0.01x   0.13x    0.28x
    4096        16 |  437,510,402     8,345,790   17,233,372   89,286,842 | 0.02x   0.04x    0.20x
   16384        16 |  521,561,136    34,760,961   11,119,554   88,431,259 | 0.07x   0.02x    0.17x
```
(Ratios are identical in both tables — with agent counts matched across sims, the `×agents` factor cancels. `agent-steps/s` is there for the absolute magnitudes.)
`--steps 300 --warmup 20`, VMAS from `.venv-vmas`, JaxMARL from `.venv-jaxmarl`, CAMAR from `.venv-camar`.

Reading it, once the task and obs are matched:

- **swarp leads at every size.** At 3 agents it is 1.3–2.1× ahead of CAMAR and 1.8–2.5× ahead of JaxMARL; at 16 agents 3.5–6× ahead of CAMAR and 8–47× ahead of JaxMARL.
- **Few agents and few envs no longer favour the JAX sims.** They run the whole rollout as one jitted `lax.scan`, which used to beat swarp's per-step floor of ~0.24 ms there; with the whole step one graph replay that floor is now ~0.026 ms (1024 × 3), and at 4096 × 3 the [fused substep chain](performance.md#fused-substep-chain) makes the physics a single launch as well.
- **The gap is narrowest at 16384 × 3** (CAMAR at 0.76×), where every simulator has enough work per step to fill the GPU.
- **The gap grows with the agent count.** swarp's step is a fixed handful of Warp kernels, so its `agent-steps/s` climbs from ~120–390 M at 3 agents to ~230–520 M at 16, while MPE/CAMAR per-step cost grows with agents and their `agent-steps/s` falls.
- **`vmas-nolidar` is consistently slowest** (0.01–0.07×): a Python per-entity step loop that neither the JAX scan nor Warp kernels have to pay.

**The default set is now genuinely comparable; the earlier "CAMAR is ~200× slower"
result was almost entirely CAMAR's ~800-obstacle default map** — the same open-arena
CAMAR here is 20–120× faster than that (`--sims camar-grid` reproduces the heavy native
task). Residual, deliberate differences: exact obs dimensionality still differs per sim;
`vmas-nolidar` also loses agent-agent collision physics (VMAS gates collisions and lidar
on one flag); dynamics/collision constants differ so trajectories won't match. Use
`--sims swarp vmas jaxmarl camar-grid` to compare each engine in its *native* setup instead.

### Ring road in VMAS and JaxMARL

Neither VMAS nor JaxMARL ships the ring road, so `compare_sims --scenarios ringroad` times two ports written for the comparison.
`swarp/benchmark/_adapters/ringroad_vmas.py` is a VMAS scenario: each RL car is a VMAS agent with VMAS's own `KinematicBicycle` (Euler, drag and collisions off), and `process_action` turns swarp's `(accel, steer)` action into the bicycle's speed command.
`swarp/benchmark/_adapters/ringroad_jax.py` is a JaxMARL `MultiAgentEnv`.
Both run the same algorithm as `RingRoadScenario`: 22 vehicles, the IDM humans as one batched array (not one simulator entity each), a brute-force leader search before and after the IDM step, the same 9-wide observation and five-term reward.
Each port is written the efficient way for its simulator: per-step tensor ops vectorised over envs for VMAS, and one jitted `vmap` + `lax.scan` over the rollout for JAX.
Driven by the same IDM controller from their observations, all three reproduce the same physics: the wave dies on 6 cars (speed std 0.045 m/s, mean 3.512 m/s in all three) and persists on 22 (std 2.917 in VMAS, 2.918 in JAX, 2.925 in swarp), with no crashes.
JaxMARL's `step` evaluates a reset every step and selects it on `done`, as every JaxMARL environment does; that cost is included.

VMAS lives in its own venv (`uv venv --python 3.12 .venv-vmas && VIRTUAL_ENV=.venv-vmas uv pip install -e '.[bench]'`), JaxMARL in `.venv-jaxmarl` as above (with `jax[cuda13]==0.11.2` for JaxMARL's jax 0.11.2):

```bash
.venv/bin/python -m swarp.benchmark.compare_sims --device cuda:0 --scenarios ringroad \
    --sims swarp swarp-fused swarp-eager vmas jaxmarl --envs 1024 4096 16384 --agents 1 4 \
    --steps 300 --warmup 20 \
    --python vmas=.venv-vmas/bin/python --python jaxmarl=.venv-jaxmarl/bin/python
```

env-steps/s on the RTX 3070 Laptop GPU, random actions over the action box, `n_agents` RL cars among 22 vehicles:

| n_envs | RL cars | swarp | swarp-fused | swarp-eager | VMAS | JaxMARL port | swarp / VMAS | swarp / JAX |
|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | 18.6 M | 2.65 M | 0.76 M | 0.58 M | 12.8 M | 32× | 1.5× |
| 4096 | 1 | 53.6 M | 11.2 M | 2.27 M | 1.75 M | 31.1 M | 31× | 1.7× |
| 16384 | 1 | 74.8 M | 39.6 M | 2.66 M | 2.45 M | 41.2 M | 30× | 1.8× |
| 1024 | 4 | 16.7 M | 2.75 M | 0.73 M | 0.30 M | 8.9 M | 56× | 1.9× |
| 4096 | 4 | 48.8 M | 10.3 M | 2.24 M | 1.05 M | 24.9 M | 47× | 2.0× |
| 16384 | 4 | 52.1 M | 43.0 M | 2.62 M | 2.00 M | 25.9 M | 26× | 2.0× |

swarp leads the JAX port at every size here, by 1.5–2.0×, as it does on navigation.
A likely reason, not measured here: the ring road's step is a long chain of small dependent passes (arc projection, two leader searches, IDM, obs, reward), which the whole-step CUDA graph replays as one launch of hand-written kernels while XLA has to fuse the equivalent `jnp` ops.
VMAS pays a Python-level step per env step; its eager torch path is close to `swarp-eager`, which runs the same task in plain torch ops.

#### More envs

Past 16384 envs the GPU is full, and swarp and the JAX port both plateau at the same ratio (1 RL car among 22 vehicles, `--steps 200`, one run):

| n_envs | swarp | VMAS | JaxMARL port | swarp / VMAS | swarp / JAX |
|---|---|---|---|---|---|
| 16384 | 67.1 M | 2.44 M | 41.0 M | 27× | 1.64× |
| 65536 | 77.3 M | 2.69 M | 51.6 M | 29× | 1.50× |
| 262144 | 78.7 M | 2.76 M | 52.7 M | 29× | 1.49× |
| 1048576 | 78.5 M | OOM | 52.4 M | – | 1.50× |

More envs add the same work to both at the same per-env efficiency, so the ratio settles at swarp's per-env advantage, about 1.5×.
Batch size is not where swarp's lead over JAX grows; work per env is, as in navigation, where it grew with the agent count.

#### More vehicles

`--vehicles` sets the vehicles per ring (RL cars included) at the fixed Sugiyama density, so the ring grows with them:

```bash
.venv/bin/python -m swarp.benchmark.compare_sims --device cuda:0 --scenarios ringroad \
    --sims swarp vmas jaxmarl --envs 16384 --agents 1 --vehicles 22 64 128 256 \
    --steps 300 --warmup 20 \
    --python vmas=.venv-vmas/bin/python --python jaxmarl=.venv-jaxmarl/bin/python
```

env-steps/s at 16384 envs, 1 RL car (22–128 at `--steps 300`, 256 at `--steps 100`, one run; the finer sweep below repeats the swarp and JAX columns, which put the 22-vehicle row within 15% run to run):

| vehicles | swarp | VMAS | JaxMARL port | swarp / VMAS | swarp / JAX |
|---|---|---|---|---|---|
| 22 | 67.1 M | 2.48 M | 47.3 M | 27× | 1.4× |
| 64 | 16.5 M | 0.37 M | 5.15 M | 45× | 3.2× |
| 128 | 4.94 M | 0.10 M | 2.16 M | 48× | 2.3× |
| 256 | 1.34 M | OOM | OOM | – | – |

Every simulator's step grows with the square of the vehicle count, because the leader search compares every pair of vehicles in a ring.
The ratio to JAX does not grow monotonically, so a finer sweep of swarp and the JAX port through the same adapters (best of three 100-step rollouts per size, 16384 envs unless noted) counts the pair comparisons per second as well:

| vehicles | swarp | JaxMARL port | swarp / JAX | swarp pairs/s | JAX pairs/s |
|---|---|---|---|---|---|
| 22 | 78.7 M | 40.7 M | 1.9× | 38 G | 20 G |
| 32 | 39.8 M | 10.8 M | 3.7× | 41 G | 11 G |
| 48 | 25.9 M | 6.77 M | 3.8× | 60 G | 16 G |
| 64 | 16.7 M | 5.27 M | 3.2× | 68 G | 22 G |
| 96 | 8.35 M | 2.90 M | 2.9× | 77 G | 27 G |
| 128 | 5.09 M | 2.21 M | 2.3× | 83 G | 36 G |
| 192 | 2.33 M | 1.06 M | 2.2× | 86 G | 39 G |
| 256 (4096 envs) | 1.38 M | 0.78 M | 1.8× | 90 G | 51 G |

The compiled XLA program explains the shape.
At 22 vehicles XLA emits each argmin of the leader search as a loop fusion, one thread per vehicle scanning the ring, which is the algorithm swarp's kernel runs.
From 32 vehicles on it switches to its row-reduction emitter, where threads cooperate on each row; on rows this short a `(value, index)` reduction runs badly, so 2× the work costs 4× the time.
That emitter improves as rows lengthen (JAX's pair rate rises 4.6× from 32 to 256 vehicles, swarp's 2.2×, since swarp's per-thread loop is already efficient and only its fixed per-step cost amortises), so the gap narrows from about 3.8× toward 1.8×.
XLA never writes the N×N distance arrays to memory up to 192 vehicles: they stay fused into the reductions, and its scratch memory grows linearly (97 MB at 128 vehicles, where one 16384×128×128 float32 array would be 1 GiB).
At 256 vehicles and 16384 envs it fails allocating 4 GiB, exactly one 16384×256×256 float32 array; at 4096 envs it runs.

Simulation is only part of training time.
At the recipe's size (1024 envs × 32 steps per iteration) swarp simulates an iteration in about 1.7 ms of a 94 ms `swarp.rl` iteration; the PPO update is the rest.
The same rollout costs about 57 ms in VMAS and 2.6 ms in the JAX port, so under the same PPO trainer VMAS would make an iteration about 1.6× slower, not 32×.
The JAX port's training speed would depend on a JAX trainer, which this comparison does not include.
