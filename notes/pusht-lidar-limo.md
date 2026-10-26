# Lidar Push-T on the Limos: how the recipe got here

This note records the twelve recipe variants (L1 to L12) that led to `SPECS["pusht_lidar_limo"]`, what each one changed, why, and what it did in swarp, in Gazebo and on the real robots.
It is a working document for the follow-up branch, not part of the docs site, and can be deleted once nobody needs the history.
The recipe itself is documented in `docs/pusht-lidar.md` (section "Limo recipe") and derived value by value in its comment in `swarp/rl/recipes.py`.

Every variant, its code and its commit history are kept under the tag `archive/pusht-lidar-limo` (a07e788, the last commit of the `pusht-lidar-limo` branch before the consolidation).
`git show archive/pusht-lidar-limo:swarp/rl/recipes.py` has every `SPECS["pusht_lidar_limo_*"]` entry with its original comment, and `git show archive/pusht-lidar-limo:docs/pusht-lidar.md` has the full results log this note condenses.
A run directory's `commit` file names the commit it was trained at; all of them resolve through the tag.

## Where things are

- Training runs: `runs/remote-davide-cf/<run>/` locally (`metrics.csv`, `cmd`, `commit`, best checkpoint), trained on davide-cf (RTX 4070 Ti SUPER) under `~/Documents/lp/swarp/runs/`.
- The kept policy: `runs/remote-davide-cf/L12-flag-s0/pusht_lidar_limo_slim240_solid_tsize_r5_flag_best.pt`, with `pusht_lidar_limo_best.pt` next to it, a copy whose `scenario` key names the consolidated recipe and whose stored keywords drop the two pruned flags (`lidar_fov_at_mount`, `scenery_collide`), so `examples/eval_mappo.py` loads it on the current code.
- The deployment and Gazebo evaluation: the sibling repository `swarp_gz_test` (`docs/results.md`, `docs/real_robot.md`, Gazebo outputs in `out/<run>/shard*/episodes.csv`).
- Real-robot sessions: `~/Documents/swarp_exp/<date>/summary.md`, one entry per episode.

## Timeline

Greedy solve rate at full difficulty unless noted; "Gazebo" is solved episodes over greedy episodes in `swarp_gz_test`.

| id | recipe at the tag | change | swarp | Gazebo / real |
|---|---|---|---|---|
| L1 | `pusht_lidar_limo` at `dt` 0.1 | U-H1 at the Limo's scale, 10 Hz policy | 0.54 after 12000 | not run |
| L2 | `pusht_lidar_limo` | 5 Hz policy (`dt` 0.2) | 0.857 | 0.567 |
| L3 | `pusht_lidar_limo_dr`, T drag 6/s | sysid: servo drive, mount blind zone, lighter T drag | 0.08 at 2000 | not run |
| L3b | `pusht_lidar_limo_dr` | robot mass 6.7 instead of the drag | 0.986 | 0.350 |
| L4 | `pusht_lidar_limo_dr_coulomb` | Coulomb floor, robot mass 2 | 0.973 | 0.883 |
| L5 / L5r / L5s | `pusht_lidar_limo_dr_wheel` | wheel-faithful contacts, then the turn stall | 0.959 / 0.992 / 0.986 | 0.800 / 0.867 / 0.917 |
| L6 | `pusht_lidar_limo_real` | real sensor, pose noise, latency, FOV 240–360° per centre ray | 0.949 | 1/20 at 240° |
| L7 | `pusht_lidar_limo_amcl` | AMCL-sized drift, FOV cut at the mount | fine-tune of L6 | 5/40 GT, 3/40 AMCL at 240° |
| (L7b) | `pusht_lidar_limo_open` | open floor, out-of-bounds failure | not trained alone | |
| (L7c) | `pusht_lidar_limo_slim` | the real slim T, 64 rays | not trained alone | |
| L8 | `pusht_lidar_limo_slim240` | from scratch at 240°, artefacts, labels, action rate | 0.18 peak | not run |
| L8b | `pusht_lidar_limo_slim240_s1`, `_s2` | two stages, no turn-away cost | 0.992 (stage 2) | 34/40 AMCL at 240° |
| L8c | `pusht_lidar_limo_slim240_1run` | the stages as one run with ramps | not trained to the end | |
| L9c | `pusht_lidar_limo_slim240_arena` | 2–5 m arenas, lidar-only surroundings, guards | 0.987 | 38/40 AMCL; 0/30 landmarks at the edge; real 0/4 |
| L10 / L10b | `pusht_lidar_limo_slim240_solid`, `_tsize` | solid surroundings at the edge, collision cost, T sizes | stalled (0.06 training solve) | not run |
| L11 | `pusht_lidar_limo_slim240_solid_tsize_r5` | 5 m lidar, three T sizes | 0.318–0.993 by surroundings | not run |
| L12 | `pusht_lidar_limo_slim240_solid_tsize_r5_flag` | outside-the-arena lidar mask | 0.972 | 39/40 GT, 34/40 AMCL; at the edge 31/40 GT, 26/40 AMCL |

L12 is the consolidated `pusht_lidar_limo`.

## L1–L2: the Limo's scale, and why the policy runs at 5 Hz

L1 took the U-H1 run (two pushers, own lidar, frame stack 4, aux loss 0.1, 0.99 greedy at the default scale) to the Limo's scale: 0.322 × 0.220 m unicycle boxes at up to 0.4 m/s and 1.5 rad/s, a 1 m T in a 4 × 4 m walled arena, a 1.5 m lidar, and a 10 Hz policy (`dt` 0.1, 16 substeps, 900 steps, gamma 0.995).
It learned slowly: greedy 0.04 at iteration 1000, 0.36 at 6000 and 0.54 after 12000, with the greedy robots standing still in 63% of robot-steps while the sampled policy solved 0.93 within 1800 steps.
At 10 Hz one action moves a robot 0.04 m, 4% of the T, against 12.5% in U-H1, and the credit per action was too small.
Lowering the entropy coefficient to 1e-3, normalising advantages, a per-step gamma of 0.99 and a 0.8 m/s robot each stayed between 0.06 and 0.15 at iteration 2000.

L2 runs the policy at 5 Hz on the latest 10 Hz scan (`dt` 0.2, 32 substeps to keep Push-T's 6.25 ms contact sub-step, 450 steps = 90 s): 0.56 greedy at iteration 1000 and 0.85 at 2000.
At 12.5 ms sub-steps a straight push chatters (the robot's speed swings 0.1–0.3 m/s and the T crawls at half speed), which is why the substep count doubled with `dt`.
The length-based shaping was halved (pos 2.5, approach 0.5) so a whole episode's shaping keeps Push-T's ratio to the goal bonus.
In Gazebo L2 solved 0.567 against 0.857 in swarp.

## L3–L5: matching Gazebo's drive and contacts

A system identification against Gazebo's Limo (`swarp_gz_test` `out/sysid/report.md`) found what swarp's robots lacked.

- **Drive.** Gazebo's Limo accelerates at 1 m/s² and 3 rad/s² and turns at 0.79 of the commanded rate. `ControlMode.SERVO` with accelerations and gains drawn per env and robot around those values reproduces every robot-alone row of the sysid to within one 0.1 s sample.
- **Blind zone.** The lidar sits 0.103 m ahead of the centre and Gazebo's sensor returns nothing within 0.10 m of it, so a face touching the bumper is invisible to the forward rays.
- **Push speed.** A velocity-mode pusher gave a quarter of its speed to the contact (T moving at 0.75 of the command, Gazebo 1.00). L3 lowered the T's drag to 6/s, which matched the push speed but let the T coast and spin on, and reached 0.08 greedy at iteration 2000. L3b raised the robots' mass to 6.7 instead: 0.98 at iteration 2000, 0.986 final, and the T stopped where Gazebo's does.

L3b solved only 0.350 in Gazebo.
In 32 of its 60 episodes its two robots squeezed the T head-on from opposite faces and spun in place, which swarp allowed (contacts only translated a robot) and a Limo cannot do: its flush front corner would have to swing through a face the other robot holds.

L4 moved the T to a Coulomb floor (μ drawn per env from 0.2–0.5, no viscous drag), because a viscous drag under-rotates an off-centre push: the quasi-static pushing reference turns the bar 64°/m at a 0.2 m offset and 145°/m at 0.4 m, a drag of 40/s 39 and 113°/m, the Coulomb floor 60 and 142°/m.
A Coulomb T needs only about 3 N to slide, so the robots came down to mass 2 and still keep 0.95–0.98 of their command in contact.
L4 solved 0.883 in Gazebo.

L5 added wheel-faithful contacts (`wheel_contacts`): a contact torque turns a robot only past the 3.5 N·m its drive holds, and its wheels grip sideways with μ 0.2.
Two robots pressing the T from opposite faces then hold their yaw (swarp −1° and −2° of a commanded +258°, Gazebo 0°), and one turning against the free T still turns (+199°, Gazebo +198°).
From scratch it reached 0.96 greedy at 16000 iterations (L5, Gazebo 0.800); resumed from L4 it reached 0.99 in 8000 (L5r, Gazebo 0.867).
L5s added the turn stall (a contact torque beyond the drive's that opposes the commanded turn stalls the turn for the substep) and resumed from L5r: 0.986 in swarp, 0.917 in Gazebo, four of its five sampled failures being the T leaving the unwalled ±2 m Gazebo arena that swarp clamped.
L5s was the deployment checkpoint for Gazebo.

## L6–L7: the real sensor and localisation

L6 (`pusht_lidar_limo_real`) prepared L5s for the real Limos: the T-mini Pro's 0.02 m minimum range, range noise drawn per robot from 5–20 mm, a field of view drawn per robot from 240–360° (blanking body-frame rays per centre ray), localisation noise on the observation only (per-episode bias, per-step jitter, velocity noise, at a scale drawn per env from 0 for mocap to 1 for AMCL), and a 0–125 ms command latency.
Fine-tuned from L5s for 8000 iterations it held 0.92–0.95; L5s itself ignored the pose noise and the latency but broke at 240° (0.094 greedy), so the rear blind sector was what the real robots needed training for.

In Gazebo at 240° L6 solved 1/20: a real lidar, cropped at its mount, loses a hit close beside the robot that is within 112.5° of the heading from the centre but 125–135° from the mount.
L7 (`pusht_lidar_limo_amcl`) cut the field of view at the mount, as the sensor does, and resized the localisation error from nav2 AMCL in Gazebo (median 0.09 m, p95 0.58 m, max 2.8 m, decorrelating in 3.6 s and 1.6 s, 89% of the excursions above 0.3 m starting within 5 s of a shove the odometry missed): a slip-driven drift with gain 2, since AMCL amplifies a shove beyond what the shove accounts for, at a scale up to 1.25.
L7 was a fine-tune of L6 and still solved only 5/40 (ground truth) and 3/40 (AMCL) at 240° in Gazebo, against 18/40 and 11/36 at 360°.

Two more recipes followed without being trained alone: `_open` (no walls, the arena stop 0.25 m past the arena, the episode failed once the T is 0.5 m past it; L6 had its T at the clamp before solving in 61% of episodes at 240°) and `_slim` (the real T, two 0.50 × 0.04 m bars, with 64 rays so the thin T keeps the hits per view the old one had at 32).

## L8: the slim T from scratch, and what blocked it

L8 (`pusht_lidar_limo_slim240`) trained from scratch for the setup the robots meet: the slim T, 43 rays over ±120° (the 5.6° spacing of 64 rays), the field of view fixed at 240° and cut at the mount, plus what the real scan and the deployment add (labels from the reported poses, grazing losses, dropouts, spurious returns, range scale and bias, the rotating sweep and a mount yaw error), every randomised parameter in the critic (`critic_dr`), and an action-rate penalty of 0.03.
It learned slowly and unstably: 0.18 greedy at its peak, 0.148 at 16000 iterations.

Pilots at 512 envs (`S240-*` runs) separated the causes:

| run | change from L8 | greedy (iter) |
|---|---|---|
| D | the old 1.0 / 0.7 × 0.3 m T | 0.037 (1000), then 0.000 (3000) |
| N2 | old T, no pose noise or latency | 0.281 (3500), then 0.018 (4000) |
| N1 | no pose noise or latency | 0.002 (1500) |
| O3 | N1 with the T's pose observed | 0.006 (1500) |
| G1, G3 | curriculum gated at 0.5 | 0.000 (1500, stuck at difficulty 0) |
| R1 | N1 with `rot_away_penalty` 0.25 | 0.021 (3000) |
| R3 | N1 with `rot_away_penalty` 0 | 0.244 (4000) |
| R4 | `rot_away_penalty` 0, full noise | 0.000 (2000) |
| R5 | R3 with 2048 envs | 0.209 (1000) |

- The slim T is a manipulation problem, not a perception one: the oracle (O3) fails as the lidar policy does. It turns 3–6× more per metre pushed than the old T, so under `rot_away_penalty` 1.0 a clumsy push is net-negative, and policies settled on approaching the T without pushing it.
- AMCL-sized pose noise and latency block learning from scratch with either T; L6 and L7 only ever met them as a fine-tune.
- More envs help: 2048 envs reach in 1000 iterations what 512 do in 4000.

L8b trained in stages at 2048 envs, each resumed from the last: stage 1 without noise, latency or turn-away cost (0.982 at 3500, then collapsed to 0.004 by 6000), stage 2 with the full noise at half the learning rate (0.988), stages 3–4 with the turn-away cost back (0.97–0.99; at 2.0 unstable, 0.78–0.97).
Stage 2's checkpoint scored 0.992 on its recipe and 0.980 at the worst AMCL, and solved 34/40 in Gazebo with AMCL at 240°.
Both stage 2 and stage 4 spin the T 1.2–1.6 times round before stopping it on the goal heading: `angle_error` is wrapped to [0, π], so heading progress pays back on the way down what it charged on the way up, and a whole turn costs only `rot_away_penalty · π`.
A penalty on the T's angular speed, which a whole turn cannot repay, is still the open lead.

L8c (`_1run`) folded the stages into one run with `TrainSpec.ramps` and a decaying learning rate; it was superseded by L9 before it finished.

## L9: arenas, surroundings and guards in one run

L9 (`pusht_lidar_limo_slim240_arena`) trained across per-env arenas of 2–5 m per side and four layouts of lidar-only surroundings (nothing, perimeter walls, a rotated room, walls plus clutter) placed beyond the free region, 0.89 m past the arena, where no robot or T could touch them before its episode failed.
The field of view became 220° with 40 rays, the largest symmetric sector inside what the lab's ydlidar configuration delivers on both Limos (−116…+108°); lm2591 also has no returns from +8 to +92°, which no recipe models.
The ramps follow a gated clock (`ramp_gate` 0.75), and four guards answer the late collapses L8b met (stage 1 held 0.95–0.97 for 2500 iterations, then fell 0.83 → 0.22 in 60 iterations with nothing non-finite):

- an adaptive learning rate on the KL (`adaptive_kl` 0.02): a skip brake alone at 0.1 froze 71% of iterations 1000–1500;
- a floor of 0.1 under the actor's std (`actor_scale_min`): with the adaptive rate alone the entropy fell from −1 to −4.2 and the training solve stalled at 0.16 (the first L9 run);
- a KL brake (`target_kl` 0.3) for a single destructive iteration;
- automatic rollback, armed once the solve passes 0.75: L9b armed it at 0.5 and took its rise through 0.38–0.51 for a collapse, rolling back four times and stopping at iteration 1656.

L9c trained in 15323 iterations (58 min) with no rollback and scored 0.987 on its recipe, 0.999 at 2 × 2 m and 0.922 at 5 × 5 m, the weak case.
`scenery_leak` stayed below 1e-4 of the rays: the labels drop everything past the arena plus 0.5 m, so the policy was robust to the surroundings mostly because it never saw them.

In Gazebo L9c solved 38/40 with AMCL and spread-out landmarks but 0/30 with landmarks at the arena's edge: it drove out of the arena and pushed the landmarks, whose hits inside the label bound read as the T.
On the real robots (2026-10-23, four episodes on a 2 × 2 and a 2.5 × 2.5 m arena) no episode solved, and most failures were the laptop-side AMCL of one robot stalling after its seed, so that robot ran on odometry alone; the T was never tracked, so no episode isolates the policy.

## L10–L12: surroundings the robots can touch

L10 (`_solid`) made the surroundings stepper obstacles (`scenery_collide`) that the robots and the T collide with, up to the arena's edge, with a fixed slot bank (four walls, six circles, six capsules; capsules because the T collides with circles and capsules but not boxes) and a collision cost of 0.03 per step within 1 cm of the other robot or a surroundings shape.
Driving both robots straight into closed walls for 200 steps with the arena stop off, 0 of 2048 robots crossed a wall of radius 0.02–0.1 m.
L10b (`_tsize`) added a ±10% T size, as a bank of five scaled copies of the T, one drawn per env.
Both stalled in the clean world (training solve 0.06 at iteration 2000 against L9c's 0.75), with `tee_seen` falling from 0.69 to 0.27: with a 1.5 m lidar in a 4–5 m arena a robot that lost the T had nothing to go on.

L11 (`_r5`) raised the range to 5 m (the T-mini Pro reaches 12 m, and the labels drop what is past the bound whatever the range), kept the clutter within 1.5 m of the arena (`clutter_reach`) and cut the bank to three sizes (each parked size still runs its contact and friction loops).
It learned the clean world (0.99 by iteration 3000) but not the surroundings: its iteration-5000 checkpoint solved 0.993 without them, 0.995 with them 0.89 m out, 0.668 at 0.3 m and 0.318 at the edge.
Its rollback guard also held it back: it kept the best solve per ramp position, but the position was the least progress over all ramps, which stays 0 until the last ramp starts, so every condition was held to the clean world's 0.99 and the run rolled back three times; the guard now keys on every ramp's progress.

L12 (`_flag`) added `lidar_arena_flag`, a fourth mask per lidar frame flagging the T-labelled hits past the arena's edge, placed from the reported pose as the deployment's labels are, so the policy no longer has to work out each hit's position to tell a wall just outside the arena from the T inside it.
It trained in 14156 iterations (365 ms each, about 86 min) with no rollback, and scored 0.972 on its recipe, 0.953 with walls and clutter at the edge, 0.934 with perimeter walls at the edge and 0.870 at the worst AMCL with walls and clutter in a 5 × 5 m arena.
In Gazebo it solved 39/40 (ground truth) and 34/40 (AMCL) with spread-out landmarks, and 31/40 and 26/40 with them at the arena's edge, where L9c solved none.
It has not run on the real robots yet.

## L13: settled hold from L12 (failed)

Branch `pusht-lidar-stop-comm` (kept as is; its useful commits are ported to `pusht-lidar-roles`) asked L12 to hold the T on its goal: on goal and settled for 1 s (`hold_steps` 5, T below 3 cm/s and 0.1 rad/s, robots below 0.1 m/s and 0.4 rad/s).
L12 solves 0.96 when passing through the goal counts.
Every fine-tune onto the full hold peaked at about 0.15 greedy hold-solve and then decayed.
The runs are under `~/Documents/lp/swarp/runs/` on davide-cf; H2 (`runs/ckpt/H2_4000.pt`) is L12 fine-tuned on the hold ramp, the base of the R runs.

| runs | change | greedy hold-solve |
|---|---|---|
| F0–F3 | full hold, 20% dead lidars, aux on-goal target | → ~0 |
| G1 / G2 | no entropy / normalised advantage | decline / 0.09 → 0.03 |
| R0 (none), R2 (vae), R5 (vae), R6 (latent) | delayed messages from H2, gate 0.5 | training solve 0.25 at iteration 540 (R0), 0.26 at 369 (R2), 0.54–0.56 at 85 (R5, R6, before the hold ramp had come in); all stopped early |
| R3, R4 | gate 0.5 ramp | stall at clock ~11700, 0.14 → 0.03 |
| B0, B1 | learned brake output | 0.06 → 0.01–0.02 |
| P0, P1 | on-goal reward 3 | 0.16 at iteration 500 → 0.02–0.04 at 1500 |
| P2 | on-goal reward 3 + ramp | ~0.13, stalls at clock 11704 |
| K0–K2 | frozen drive, brake only | 0.002 → 0.021 (K0), others ≤ 0.02 |

K0's last word: best training solve 0.070, final windowed 0.020 (stochastic policy).
Decayed policies lose reach (F0: reach 73% → 0.3%) and slow down everywhere (P0 at iteration 1500 drives at 0.13 m/s against L12's 0.24 m/s).
The learned brake is dropped from the code.

The hold is feasible: with H2's T within tolerance, every settle check passes on 21–27% of steps.
The likely cause is perception.
Each robot's own estimate of the T pose (its aux head) has a median error of 0.18 m / 17° at contact range and 0.24 m / 39° at 0.4–1 m, against a 0.15 m / 14° goal tolerance, so the robots cannot tell when the T is on goal.
AMCL-sized pose noise adds a median 0.12 m to every goal-frame estimate.

## L14: what blocks the settled hold, and a lidar student that learns it

Branch `pusht-lidar-roles` set out to let the two robots learn pusher and observer roles, so that one robot could watch the T and tell the other when it is on goal.
Every run fine-tunes L12 (`runs/ckpt/L12_best.pt`) for 2000 iterations at `--start-clock 12000`, one seed, under mocap (`pose_noise=(0, 1e-6)`, which keeps the critic's randomisation columns; 0 removes them and the resume fails), with the full hold (`hold_steps` 5, settle thresholds 0.03 / 0.1 / 0.1 / 0.4, `hold_reward` 1, `on_goal_reward` 3), `aux_goal` and no approach shaping (`agent_dist_shaping` 0).
The numbers are the periodic greedy hold-solve over 512 envs (`eval_solve_rate`); every run is under `~/Documents/lp/swarp/runs/<id>` on davide-cf or `runs/local/<id>` on the laptop.

### Two gates that did not need roles

- **G0, can the lidar see the T well enough?** A supervised study on teleported viewpoints (scratchpad report, not kept): a template fit of the T outline to the T-labelled points reaches a median 0.04–0.07 m and 2–7° from one scan within 0.8 m of the stem side or a bar end, and fusing both robots' scans through their relative pose reaches 0.05 m / 4° everywhere out to 1.8 m. From behind the bar, where the stem is hidden, one scan's heading error is about 90°. Learned MLPs on the scans reach that accuracy only within 0.5 m. The sensor is enough; the policy's learned estimate (0.18 m / 17°) is the weak part.
- **G1, does the true T pose fix the hold?** `--privileged-actor` appends the aux targets (the T's position relative to each robot, its heading and the on-goal flag) to the actor's input. From L12 it scored no better than the plain fine-tune.

| id | change | greedy hold-solve (best / final) |
|---|---|---|
| N0-plain | plain fine-tune | 0.043 / 0.014 |
| NP-pose | `--comm pose` | 0.029 / 0.008 |
| G1-priv | `--privileged-actor` | 0.025 / 0.006 |
| NG-near | goals within 0.4 m first (`--start-difficulty 0 --curriculum-iters 3000`) | 0.025 / 0.008 |
| N0H-plain, G1H-priv | the same from H2 (`runs/ckpt/H2_4000.pt`) | 0.176 / 0.100, 0.365 / 0.160 |

### The first block: the turn-away penalty

Greedy rollouts of these checkpoints (scratchpad study D1) showed what happens.
L12 reaches the goal in 98% of episodes, but crosses it at 0.28 m/s while spinning the T at 0.57 rad/s, about 5.6 times per episode.
The fine-tunes did not learn to stop: they stopped pushing, with reach falling to 1–8% and the robots' mean speed from 0.24 to 0.06–0.11 m/s.
Stopping is physically easy: L12 plus a scripted zero command while the T is on goal holds 0.98.
The cause is `rot_away_penalty` (1.0 per radian the heading error grows): once passing through no longer ends the episode with the goal bonus, L12 pays 0.03–0.06 per step to it against 0.025 of pose shaping, so not pushing is the cheapest policy.
That also explains the learned-brake runs of L13.

With `rot_away_penalty=0` the privileged actor learns the hold and the lidar actor does not:

| id | change | greedy hold-solve by iteration (250 … 2000) |
|---|---|---|
| RA0-priv | penalty 0, `--privileged-actor` | 0, 0, 0, 0.25, 0.62, 0.86, 0.89, **0.92** |
| RA0-plain | penalty 0 | 0 throughout |
| NP-RA0, AUX-RA0, LAT-RA0 | penalty 0 with pose messages, aux loss 1.0, latent messages | ≤ 0.002 (stopped early) |
| RS2 | penalty 0, `rot_shaping_factor` 2 | 0 |
| RA50 | penalty 0.5 | 0.020 |
| RA25, RA25b | penalty 0.25, then 3000 more iterations | 0.068, then **0.334** |

Without the penalty the lidar policies hold the T near the goal (0.45 m mean) at a random heading: the penalty was their dense heading signal, and only the privileged actor, which reads the heading, does without it.

### The privileged teacher stops on the on-goal flag

Evaluating RA0-priv with noise on its privileged input (scratchpad study E1): it holds 0.93 clean, 0.72 under 0.05 m / 5° jitter, 0.29 under 0.07 m / 7° and 0.00 under 0.10 m / 10°; a per-episode bias of 0.05 / 0.10 m gives 0.77 / 0.37, and 20% gross outliers 0.22.
With the on-goal flag zeroed it scores 0.000, and with the true flag kept it shrugs off 0.10 m / 10° of jitter: all the loss comes through the flag recomputed from the noisy pose.
A teacher without the flag (TF-noflag, `aux_goal=False`) did not learn the hold in 2000 iterations (0.004); one trained under estimator-like noise on its input (`--priv-noise`, flag-free) reached 0.205 at the higher noise (TN-hi, 0.03 / 0.05 m bias / jitter, 3° / 5°, 10% outliers, scored under that noise) and 0.045 at the lower (TN-mid).

### Distilling the teacher into the lidar policy

`--teacher runs/ckpt/RA0priv_best.pt` adds a squared error between the student's and the teacher's pre-tanh action means on the student's own rollouts (DAgger-style, weight `bc_coef` 1) to the PPO loss.
The student reads only its lidar; the teacher reads the truth.

| id | student | greedy hold-solve |
|---|---|---|
| DS-plain, -b, -c | L12, penalty 0 | 0 until 1500, 0.100 at 2000; then 0.17–0.39 over 250–750 and 0.465 at 4000; 0.47–0.61 over 4250–6000 (best 0.605 at 5500) |
| DS-roles-pose, -b, -c | L12 with learned roles and pose messages | 0 at 1000 (stopped, resumed from iteration 1000); 0.27 at 2500; 0.31–0.42 over 2750–4000 |
| DS-RA25 | RA25b with the teacher, penalty 0.25 | 0.24 (below RA25b's own 0.33) |
| DS-decay | DS-plain-c with `--bc-decay-iters 1500` (the imitation weight falls 1 → 0, then PPO alone) | 0.54, 0.63, 0.63, 0.69, 0.67, **0.74** (1500, weight 0), 0.71, 0.70 |
| DS-amcl, -b | DS-plain-b under the recipe's full pose noise (0–1.25 AMCL scale) | 0.32–0.42 over 2000 iterations, then 0.40–0.50 over 2000 more |

Final greedy evaluation of the candidates (`examples/eval_mappo.py`, 1024 envs, seed 7, the fine-tune's hold settings, Wilson 95%), under mocap and under the recipe's AMCL-sized pose noise (scale 0–1.25):

| checkpoint | mocap | AMCL |
|---|---|---|
| L12 | 0.000 | 0.000 |
| RA25b final | 0.319 [0.292, 0.349] | 0.291 [0.264, 0.320] |
| DS-plain-c at 1500 | 0.592 [0.561, 0.622] | 0.543 [0.512, 0.573] |
| DS-decay at 1500 | **0.738 [0.711, 0.764]** | **0.705 [0.676, 0.732]** |
| DS-decay final | 0.710 [0.681, 0.737] | 0.692 [0.663, 0.720] |
| DS-amcl final | 0.471 [0.440, 0.501] | 0.426 [0.396, 0.456] |
| DS-roles-pose-c final | 0.392 [0.362, 0.422] | 0.346 [0.317, 0.375] |
| RA0-priv (reads the true pose) | 0.921 [0.903, 0.936] | 0.903 [0.884, 0.920] |

The pose noise costs the distilled student about five points, and fine-tuning it under the noise (DS-amcl, the same number of iterations as DS-plain-c) did not help, on one seed.
Releasing the imitation term is what lifted the student past its plateau: the teacher's mean is a good start but caps the student at what it can copy without the truth, and PPO alone then improves on it.
Training on past the release does not lift it further: 2000 more PPO iterations from DS-decay's final checkpoint held 0.64–0.71 under mocap (PPO-after) and fell from 0.66 to 0.59 under AMCL-sized noise (PPO-after-amcl), on the 512-env periodic eval, so DS-decay at iteration 1500 stays the best lidar policy (`runs/ckpt/L14_ds_decay_1500.pt`, gitignored).

### Learned roles did not emerge

`--learned-roles` (a role head sampled every 5 steps, tied warm start so both roles act as L12) kept its role choice a coin flip in every run: role entropy stayed at ln 2 and half the decisions switched (RP-roles-pose, no teacher: 0.006 best; DS-roles-pose: see above).
With both roles acting identically, the choice of role does not change the return, so its policy gradient is zero in expectation: the tied start is a saddle that only a role-dependent incentive breaks (the information reward or an observer view bonus, not built).
The pose messages, not the roles, are the plausible source of any difference between the two distilled students.

### Gazebo

`../swarp_gz_test` (branch `hold-scoring`), edge landmark layout, 220° field of view, 4 shards × 10 episodes, scored with the hold (`hold_steps:=5 hold_settle:=0.03,0.1,0.1,0.4` for checkpoints trained without it), Wilson 95%:

| policy | robot poses | hold-solve | reached goal |
|---|---|---|---|
| L14 student (DS-decay at 1500) | ground truth | 9/40 = 0.23 [0.12, 0.38] | 24/40 |
| L14 student | AMCL | 2/40 = 0.05 | 6/40 |
| L12 | ground truth | 0/40 | 32/40 |
| L12 | AMCL | 0/20 (two shards hung) | 7/20 |
| RA0-priv (reads the true T pose from `/tee/pose`) | ground truth | 31/40 = 0.78 [0.62, 0.88] | 31/40 |

The privileged policy holds every episode in which it reaches the goal; its nine failures never get there.
Of the student's 15 reached-but-not-held episodes, 13 ended with the T outside tolerance.
The videos (`out/vid_L14_student`, `out/vid_L12`, `out/vid_priv`, seed 0, episodes 0–5) show why: the student brings the T onto the goal and both robots stay in contact and keep pushing, so the T turns through the goal and out again (episode 3 is on goal at 32 s and again at 68–72 s); it does not stop and then see the T slide.
The student loses 0.74 − 0.23 in the transfer, where the privileged policy loses 0.92 − 0.78: the on-goal decision from lidar alone is what does not transfer.

### What to take from it

- The hold needs a crisp on-goal decision. A lidar policy learns one from a teacher that has it, distilled on its own rollouts and then released to PPO: 0.74 under mocap and 0.71 under AMCL-sized noise, against 0.00 for L12 and 0.33 for the best reward-only run (penalty 0.25), all on one seed.
- A classical T fit (G0) is accurate enough for the teacher's continuous inputs but not, as is, for its flag: a deployed teacher would need a flag-free or noise-trained policy, which learns far slower.
- Roles need symmetry breaking before they can be evaluated.

## L15: push and observe with fixed roles (failed)

The plan: fine-tune the L14 student with one pusher and one observer per episode (Phase A), and only if that team beats two pushers, resume it as learned roles so the switch has two distinct heads to choose between (Phase B).
No teacher in either phase, since the privileged teacher never needs to look.
`--resume-widen` now resumes a `(2, 0)` checkpoint into `(1, 1)` roles, starting the never-trained observer head and role input from the pusher's (commit f455fdf).

Both runs: the L14 fine-tune flags (`--seed 0 --resume-widen --start-clock 12000 --start-difficulty 1.0 --action-rate-ramp-iters 0 --iters 2000`, exact robot poses `pose_noise=0,1e-6`, the hold `hold_steps=5 hold_reward=1.0`, settle 0.03/0.1/0.1/0.4, `on_goal_reward=3.0 aux_goal=True agent_dist_shaping=0 rot_away_penalty=0`) plus `--comm pose`.

| id | start | roles | machine, commit |
|---|---|---|---|
| A1 | DS-decay at 1500 | `n_pushers=1 standoff_dist=0.8`: a pusher and an observer, reshuffled every reset, with the observer terms (standoff, visibility, coverage, occlusion) | davide-cf, f455fdf, 0.46 s/it |
| A0 | DS-decay at 1500 | `(2, 0)`, the control | laptop, 9759916 |

At iteration 1 the largest observer term (coverage, −0.012 per observer step) was 0.56× the team term (+0.021), so no term needed rescaling.
The periodic greedy evals (512 envs) wandered within 0.60–0.74 in both runs and the stochastic solve rate fell from 0.66 to 0.53–0.55 by iteration 2000.

Final greedy evaluation (`examples/eval_mappo.py`, 1024 envs, seed 7, the hold, Wilson 95%; each checkpoint under its own roles):

| checkpoint | mocap | AMCL |
|---|---|---|
| A1 best | 0.741 [0.714, 0.767] | 0.729 [0.701, 0.756] |
| A1 at 250 | 0.729 [0.701, 0.756] | 0.723 [0.694, 0.749] |
| A1 final | 0.673 [0.644, 0.701] | 0.637 [0.607, 0.666] |
| A0 best | 0.727 [0.698, 0.753] | 0.696 [0.667, 0.724] |
| A0 final | 0.590 [0.559, 0.620] | 0.599 [0.568, 0.628] |
| DS-decay at 1500 (the start) | 0.738 [0.711, 0.764] | 0.705 [0.676, 0.732] |

The observer never observed.
In 512 greedy episodes of A1's best checkpoint its median distance to the T centroid was 0.39 m (the pusher's 0.38 m), it spent 74% of the steps within touching range (0.45 m) and 6% near the 0.8 m standoff, and it was in touching range for 79% of the steps the T moved.
Coverage rose only from 0.41 to 0.45 over the run, and visibility was 0.97 from the start because a pusher already sees the T.
The team term goes to every robot and outweighs the observer terms, so the observer keeps the pusher's behaviour it was started from.

Gate A failed: the best checkpoints' intervals overlap each other and DS-decay's, and the observer does not take a viewpoint.
Only the final checkpoints separate under mocap (0.67 against 0.59), one comparison of four and dependent on where the run stopped.
Phase B (learned roles from A1) was not run, and neither were the Gazebo evaluations of A1 and A0.
The Gazebo node can now exchange `--comm pose` messages (`swarp_gz_test` e2ba5a4, matched by stamp), so a messaging policy can be deployed when one is worth it.

## What the consolidation removed

Besides the recipe variants, the scenario knobs that only they used were pruned (all are in the tag):

- `lidar_fov` as a `(lo, hi)` range drawn per robot (L6): the field of view is one value.
- The field of view per centre ray (`lidar_fov_at_mount=False`, L6): the sensor model always crops at the mount, which is identical at 360° or with the lidar at the centre.
- Lidar-only surroundings (`scenery_collide=False`, L9) with their reach-based clearance (`robot_reach`, `tee_reach`): surroundings are always stepper obstacles, `scenery_clearance` defaults to 0, and `Lidar.bind` no longer takes a `Scenery` (the standalone `lidar_scan` and `lidar_scan_warp` still do).
- An open floor of one fixed size (`open_arena` without `arena_size`, L7b–L8c): `open_arena` and `arena_size` go together (a value fixes a square), and `oob_margin` needs `open_arena`.
- `wheel_drive_force`, which no recipe ever set.

The consolidation changed only which per-robot values L12's resets draw (one fewer random draw in the sensor-randomisation reset kernel), not their distributions; every default world, and L12 before that last change, steps bit-identically to the tag.

## Deployment notes for `swarp_gz_test`

- `obs.py` reads `lidar_fov_at_mount` from a checkpoint's keywords and defaults it to False (per centre ray). Checkpoints trained after the consolidation no longer store it and are always cropped at the mount, so the deployment's default must become True for them (or key on the checkpoint's commit). The original L12 checkpoint still stores `lidar_fov_at_mount=True`, so it deploys as before.
- `swarp_reference.py` and `sysid_swarp.py` look recipes up by name in `SPECS`: use `pusht_lidar_limo`, or check out the tag for an old name.
- `scripts/real/config.bash` and `localize.py` still point at L9c; L12 needs `lidar_fov:=3.8397243543875255` (220°) and the arena's half side, as L9c did.

## Open leads

- Run L12 on the real robots once both laptop-side AMCLs stay alive for a whole episode (the 2026-10-23 sessions lost one robot's localisation after its seed).
- The T spin: a penalty on the T's angular speed instead of `rot_away_penalty`, which a whole turn repays.
- lm2591's blind sector (+8…+92°), which no recipe models.
- The 5 × 5 m arena and perimeter walls at the edge are the weakest conditions (0.92 and 0.93).
- Earlier ideas not pursued: a variable team per episode (needs an inactive-agent concept and a per-agent validity mask in PPO), training on other shapes (`ell`, `plus`, `bar`), Success Guided Sampling instead of the hand curriculum.
- A second seed for DS-decay and RA0-priv before the hold goes into `SPECS["pusht_lidar_limo"]`; every L14 number is one seed.
- Break the tie between learned roles with a role-dependent incentive (the information reward or a view bonus of the original plan, neither built), then judge roles against DS-decay. L15 showed the scenario's observer terms are too weak against the team term to pull a pusher-initialised observer off the T.
- A deployable on-goal decision: the teacher's flag from a T estimator, or a teacher trained flag-free under noise for longer than TN-hi's 2000 iterations.
- The video renderer frames the whole world, so on the Limo arena the robots are a few pixels; a camera that follows the T is needed before a video is worth recording.

## Working on davide-cf

- `ssh davide-cf@davide-cf` (Tailscale); venv `~/Documents/lp/swarp/.venv`; runs under `~/Documents/lp/swarp/runs/<run>`, each with `cmd`, `commit`, `metrics.csv` and checkpoints.
- Launch with `setsid nohup … < /dev/null & disown`, and write `git rev-parse HEAD > <dir>/commit` first.
- Never `pkill -f <pattern>` with the pattern spelled literally in the same command line, locally or over ssh: it kills the calling shell; use a bracket such as `L12-flag-s[0]`.
- Pull a run with `rsync -az --include='metrics.csv' --include='*.log' --include='commit' --include='cmd' --include='*_best.pt' --exclude='*' davide-cf@davide-cf:Documents/lp/swarp/runs/<run>/ runs/remote-davide-cf/<run>/`.
