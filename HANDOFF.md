# Handoff: lidar Push-T with observer and pusher robots

Branch `pusht-lidar` (42 commits on top of `main` at ce3c713, head 9e95078), not pushed to GitHub and no PR opened.
The approved plan is `/home/davide/.claude-unipi/plans/however-i-do-not-hashed-octopus.md`; the design page with the related-work survey is `docs/pusht-lidar.md`.
This file is untracked: delete it or move its content into `docs/pusht-lidar.md` before merging.

## Goal

m robots push a T (or another compound shape) to a target pose end to end from lidar.
Robots know their own pose but not the T's.
n robots are observers and m−n are pushers, roles drawn at random per episode with fixed counts; all robots share a body and can push, and only rewards and policy heads differ.

## What is on the branch

- **Lidar** (`swarp/sensors/lidar*.py`): rays hit circles, oriented boxes, segments as capsules and box agents, with a per-ray hit class (0 none, 1 agent, 2 static, 3 movable) and hit index; `Lidar.bind(world)` outside capture, `Lidar.launch(world)` inside the whole-step graph.
- **Box agents** (`AgentShape.BOX`, `AgentConfig.half_extents`): oriented-box contacts with agents, obstacles, walls and movable bodies (`obb_contact` in `swarp/core/collisions.py`), layered and fused paths bit-identical (box kernels compiled without FP contraction); `"auto"` keeps box worlds on the layered path because the fused box variant uses 213–255 registers.
- **Shapes** (`swarp/scenarios/shapes.py`): `tee`, `ell`, `plus`, `bar`; `PushTScenario(shape=...)`; Push-T numbers bit-identical to `main`.
- **Scenario `pusht_lidar`** (`swarp/scenarios/pusht_lidar.py`): `obs_level` 0 (T-pose oracle + lidar), 1 (own lidar), 2 (+ teammates' object hits pooled in the world frame); `frame_stack`; observer rewards (standoff potential, capped visibility, sector coverage, occlusion, optional contact), all penalties or potentials; `critic_state()` role-sorted, `agent_roles()`, `aux_targets()`; `SPECS["pusht_lidar"]` is stage T0.
- **Trainer** (`swarp/rl`): `StateCritic` (privileged, per-agent values), `PolicyNet` (role heads or per-role networks, aux head, GRU via `--recurrent`, attention communication via `--comm attn`), per-role advantage normalization, checkpoint format `swarp.rl/2`, guards for bad roles, moving hook buffers and mismatched `--resume`.

## Verification done

- Full suite on the merged branch: 1494 passed, 11 skipped, 2 deselected (`test_matches_under_a_user_created_stream`, which fails the same way on the base commit when the GPU is shared).
- Default training is byte-identical to `main` (20-iteration navigation `metrics.csv` A/B after every trainer commit).
- Trainer throughput on davide-cf (`train_throughput`, navigation and pusht, 1024/4096 envs, graphs and fast): identical ms/iter before and after.
- Step throughput after the lidar/box-parameter merge (`swarp.benchmark.throughput`, 2000 steps, 6 alternating pairs, best of 6, davide-cf): all 24 configurations within −0.8% to +1.6%, graph and no-graph.

- Step throughput, d5af49e vs 0012dbf (covers the box-physics merge), davide-cf RTX 4070 Ti SUPER, 6 alternating reps of the full sweep: 22 of 24 cells within ±1.5%.
  The two outliers (graph 16000×16 −2.6%, no-graph 8000×16 −2%) were rechecked in isolation (`~/bench2/cell.py` + `run3.sh`: 5000 steps, base/new rotated, 9 reps): base vs new −0.1% and +0.4%, so no regression.
  A compile-time removal of the dead `CLAMP_FOOTPRINT` branch in `integrate2d_kernel` was tried and reverted: identical in isolation.

- Box worlds fused vs layered (`~/bench2/out_box.log`): same crossover as discs, so commit 0549e98 drops the box exclusion from `"auto"`; table in `docs/performance.md`.
- Push-T/transport `swarp.benchmark.scenarios` A/B, d5af49e vs 0012dbf, 4 interleaved reps (`~/bench2/scen_*.csv`): graph path within ±1% in all 8 cells. One eager-reference cell (transport 1024×2) read +25% there; 6 interleaved reps on the quiet laptop GPU put every transport eager cell at −1.8% to −0.5%.
- Clips with the observers' lidar rays (commit 3e7a577): `runs/clips/H{2b,3b,5b}-best-lidar.mp4`, also on davidez in `~/Videos/swarp-pusht-lidar/`.

- Full suite at 87e14d3: 1517 passed, 11 skipped; `test_matches_under_a_user_created_stream` passes on a quiet GPU.
- Unicycle/box robots (87e14d3): default fleets bit-identical (30-step digests of obs, reward and critic, graph and eager); `pusht`/`pusht_lidar` graph-path step within noise (scenario benchmark, 6 interleaved reps, laptop).
- `swarp.benchmark.scenarios`' parity gate fails `pusht_lidar` at 5 agents on the base commit as well as now (probably its single substep where Push-T needs 8, plus grazing-ray flips the scenario's own tests tolerate); not investigated.

## Training results

One seed each, greedy solve rate at full curriculum difficulty, 512 eval envs.

Default arena (lidar range 0.8, 2 pushers + 3 observers unless noted):

| run | iter 1000 | final (iter) |
|---|---|---|
| T0 oracle (L0) | 0.47 | 0.99 (4000) |
| T1 own lidar, frame_stack 4, aux 0.1 | 0.81 | 0.99 (4000) |
| T2 pooled hits, frame_stack 4, aux 0.1 | 0.90 | 1.00 (4000) |
| T1 with lidar range 0.05 (leak control) | 0.004 | stopped |
| T1 frame_stack 1 | 0.81 | 0.98 (2500) |
| T1 no aux loss | 0.86 | 0.96 (2000, stopped) |
| 2 pushers only | 0.96 | 0.97 (1500, stopped) |
| 5 pushers, no observers | 0.24 | 0.95 (2000, stopped) |

The default arena is too easy to show the value of observers or communication: the T is almost always in view, and memory and the aux loss barely matter.
The short-range regime below is the informative one.

Short range (lidar range 0.3, observer standoff 0.4):

| run | iter 1000 | iter 1500 | latest |
|---|---|---|---|
| H1 2 pushers only, seed 0 | 0.62 | 0.83 | 0.95 (4000) |
| H1 seed 1 | 0.83 | 0.91 | 0.99 (4000) |
| H1 seed 2 | 0.92 | 0.96 | 0.98 (2500), running |
| H2b + 3 observers, own lidar | 0.77 | 0.80 | 0.99 (4000) |
| H3b + 3 observers, pooled hits | 0.69 | 0.91 | 0.98 (4000) |
| H4b + 3 observers, pooled hits, GRU | 0.80 | 0.94 | 0.98 (3500), running |
| H5b + 3 observers, own lidar, attention comm | 0.87 | 0.95 | 0.98 (4000) |

At 4000 iterations every short-range variant ends at 0.975–0.99 with overlapping intervals; the differences are only in early learning speed, and the two extra pushers-only seeds learn as fast as the best observer run, so the seed-0 ranking does not hold so far.
Seeds 1 and 2 of H2b, H3b, H5b launch on davide-cf at the end of `~/bench2/chain.sh` (`swarp-next/runs/<run>-s{1,2}`; that checkout is now at 0012dbf on branch `pusht-lidar-head`).
Clips of the seed-0 best checkpoints: `runs/clips/H{2b,3b,5b}-best.mp4` (greedy 0.994, 0.977, 0.980; robots spread around the T at the standoff; the renderer does not colour roles). Seed-0 run artifacts are pulled into `runs/remote-davide-cf/`.

The first short-range observer runs (H2–H5, standoff 0.22) were invalid: the standoff sat inside the T's 0.265 corner radius, observers jammed the T and the angle error stayed near 1.6 rad.
The scenario now raises on such a standoff (commit 0012dbf).

## Runs in progress

- davide-cf, unicycle box robots (`--scen-kwarg model=diff_drive --scen-kwarg agent_shape=box`, level 1, `frame_stack` 4, aux 0.1, seed 0), from `swarp-next` at 87e14d3 (branch `pusht-lidar-87e14d3`): `runs/U-H2b-r03-L1` (short range, 2 pushers + 3 observers), `runs/U-H1-r03-2push` (short range, 2 pushers), `runs/U-T1-L1` (default range). Each `cmd` file has the flags.
- All holonomic seed repeats are finished and in `docs/pusht-lidar.md` (Results); their run directories are pulled into `runs/remote-davide-cf/` and `~/swarp-runs/`.
- Finished runs: laptop `~/swarp-runs/{T0-L0-s0,C-blind-L1-r005,A-L1-fs1-aux,H1-r03-2push}`; davide-cf `~/Documents/lp/swarp/runs/{T1-L1-fs4-aux,T2-L2-fs4-aux,A-L1-noaux,A-L1-2push-noobs,A-L1-5push-noobs,pusht-n2-state,pusht-n3-state}` and the stale `swarp-next/runs/H{2,3}-r03-*`, `H5-r03-L1-comm`.
- Each run directory holds `commit`, `cmd` (remote), `stdout.log`, `metrics.csv` and checkpoints; check with `grep "eval @" <run>/stdout.log | tail -1`.
- Pull with `rsync -az --include='metrics.csv' --include='*.log' --include='commit' --include='cmd' --include='*_best.pt' --include='*_final.pt' --exclude='*' davide-cf@davide-cf:Documents/lp/swarp-next/runs/<run>/ runs/remote-davide-cf/<run>/` (only the two full-state baselines are pulled so far).

## Launch commands

All runs are `python -m swarp.rl --scenario pusht_lidar --seed <s> --checkpoint-dir <dir>` plus the flags below; everything else comes from `SPECS["pusht_lidar"]` (4000 iterations, 512 envs, Push-T curriculum, role heads, state critic).

| run | extra flags |
|---|---|
| T0 | (none: the recipe is obs_level 0) |
| T1 | `--scen-kwarg obs_level=1 --scen-kwarg frame_stack=4 --aux-coef 0.1` |
| T2 | `--scen-kwarg obs_level=2 --scen-kwarg frame_stack=4 --aux-coef 0.1` |
| leak control | T1 flags plus `--scen-kwarg lidar_max_range=0.05 --iters 1500` |
| T1 frame_stack 1 | `--scen-kwarg obs_level=1 --scen-kwarg frame_stack=1 --aux-coef 0.1 --iters 2500` |
| T1 no aux | `--scen-kwarg obs_level=1 --scen-kwarg frame_stack=4 --aux-coef 0` |
| 2 pushers only | T1 flags plus `--n-agents 2 --scen-kwarg n_pushers=2` |
| 5 pushers | T1 flags plus `--scen-kwarg n_pushers=5` |
| H1 | `--n-agents 2 --scen-kwarg n_pushers=2 --scen-kwarg obs_level=1 --scen-kwarg frame_stack=4 --scen-kwarg lidar_max_range=0.3 --aux-coef 0.1` |
| H2b | `--scen-kwarg obs_level=1 --scen-kwarg frame_stack=4 --scen-kwarg lidar_max_range=0.3 --scen-kwarg standoff_dist=0.4 --aux-coef 0.1` |
| H3b | H2b with `obs_level=2` |
| H4b | `--scen-kwarg obs_level=2 --scen-kwarg lidar_max_range=0.3 --scen-kwarg standoff_dist=0.4 --aux-coef 0.1 --recurrent` |
| H5b | H2b plus `--comm attn` |

Laptop runs go in `~/swarp-runs/<run>` (outside Dropbox); write `git rev-parse HEAD > <dir>/commit` before launching.
On davide-cf, run from `~/Documents/lp/swarp-next` with `PYTHONPATH=$PWD ~/Documents/lp/swarp/.venv/bin/python …`, after updating that checkout to the current branch head (it is at 87e14d3, branch `pusht-lidar-87e14d3`; push the head to a new branch name there, since a push to its checked-out branch is rejected).

Refresh the results from every log:

```bash
for f in ~/swarp-runs/*/stdout.log; do echo "$(basename $(dirname $f)): $(grep 'eval @' $f | tail -1)"; done
ssh davide-cf@davide-cf 'for f in ~/Documents/lp/swarp*/runs/*/stdout.log; do echo "$(basename $(dirname $f)): $(grep "eval @" $f | tail -1)"; done'
```

## Gotchas

- davide-cf: `ssh davide-cf@davide-cf` (Tailscale), venv `~/Documents/lp/swarp/.venv`; a push to the checked-out branch is rejected, so detach there first or push to another branch name.
- Launch with `setsid nohup … < /dev/null & disown`.
- Never `pkill -f <pattern>` with the pattern spelled literally in the same command line, locally or over ssh: it kills the calling shell (and any Monitor whose script contains it); use a bracket such as `H2b-r03-L[1]`.
- Training cost: `pusht_lidar` is about 30% slower per iteration than Push-T with 5 robots; the GRU is about 2.3× and attention about 1.6× the plain `PolicyNet` per iteration.
- `swarp.rl` names the multiobjective metric columns `rew_term0..6`; their order is `team, approach, standoff, visibility, coverage, occlusion, contact` (`OBJECTIVES` in `swarp/scenarios/pusht_lidar_kernels.py`).

## Next steps

1. Unicycle box runs done (`runs/remote-davide-cf/U-*`, Results in `docs/pusht-lidar.md`, clips on davidez `~/Videos/swarp-pusht-lidar/`). Open question: with observers they learn ~1000 iterations later and end at 0.94–0.95; if that matters, try a seed repeat, `contact_penalty`, or an observer term for staying out of the T-to-goal corridor.
2. Variable team per episode (1–2 pushers, 1–3 observers, one policy): the user wants it later. Audit findings: no inactive-agent concept in the engine (soft walls pull a parked robot back, so park by re-pinning or by an inactive role), the roles kernel, teammates/critic blocks and observer means assume fixed counts, `PolicyNet(arch="separate")` and `_check_role_counts` assume fixed counts, and PPO has no per-agent validity mask (GAE, losses and logging average over all agents); `comm_mask()` already reaches the attention policy.
3. Decide on a regime where observers matter (shorter range, occluding clutter, larger arena): at range 0.3 two pushers solve it alone.
4. Remove this file, then open a PR from `pusht-lidar` to `main`.

Known limitation: a contact pushes a unicycle sideways (the integrator adds the whole contact force to its velocity), so box/unicycle robots slide along faces; wheel-faithful contacts would be an engine change shared with every diff-drive scenario.

Planned but not implemented:
- Training on `ell`/`plus`/`bar` (one shape per world; a per-env shape mask would need engine work).
- The binned SGS reset sampler (`swarp/scenarios/sgs.py`) as an ablation of the hand curriculum.
- Distillation from a restricted privileged teacher (`swarp/rl/distill.py`).
