# Lidar Push-T

`pusht_lidar` is Push-T without the object pose.
A team of robots must bring a T-shaped body (or another compound shape) to a target pose while each robot observes only its own pose, the goal, its teammates and its own lidar.
The team is split into **pushers**, which move the body, and **observers**, which position themselves so that the body is well observed and their scans are not blocked by teammates.
Roles are drawn at random per episode with fixed counts, all robots share the same body, and every robot can push: the roles differ only in their rewards and in the policy head they act through.

This page records the motivation and the related work behind the task, and the training stages and their results as they land.

## Related work

### Multi-robot pushing and transport

Learned multi-robot pushing and transport almost always observes the object pose directly.

- DeReCo ([arXiv 2603.08111](https://arxiv.org/abs/2603.08111)) trains MAPPO in three stages that first give the policy the object's shape, mass and friction as privileged information, then reconstruct it from local observations, then drop it; the object pose relative to the end effector is still observed (motion capture on hardware).
- Feng et al. ([arXiv 2411.07104](https://arxiv.org/abs/2411.07104)) push a T with two quadrupeds through a three-level hierarchy (RRT plus a centralised subgoal policy, a decentralised MAPPO mid-level, a locomotion controller), with robot and object state from 24 motion-capture cameras.
- decPLM ([arXiv 2509.14342](https://arxiv.org/abs/2509.14342)) transports objects with 2 to 10 quadrupeds through a decentralised, communication-free hierarchical policy with a pose-tracking "constellation" reward.
- Multi-Agent Behavior Retrieval ([arXiv 2312.02008](https://arxiv.org/abs/2312.02008)) learns cooperative pushing for two to four wheeled robots from a few demonstrations plus retrieved ones.
- Bloom et al. ([arXiv 2306.12926](https://arxiv.org/abs/2306.12926)) compare implicit coordination through the object with an auxiliary network that predicts the global state, for swarm transport; the prediction head is the closest precedent for the auxiliary pose loss used here.
- Sayed et al. ([arXiv 2606.09610](https://arxiv.org/abs/2606.09610)) form shapes around arbitrary objects for transport with MAPPO and list onboard estimation of object geometry as future work.
- PushingBots ([arXiv 2511.15995](https://arxiv.org/abs/2511.15995)) and MultiPush ([arXiv 2609.27005](https://arxiv.org/abs/2609.27005)) plan or schedule pushes for many robots and objects at the task level rather than learning control from sensing.

The sensing-only line is hand-designed.
Occlusion-based cooperative transport (Chen, Gauci, Li, Kolling, Groß, IEEE T-RO 31(2), 2015, [pdf](https://eprints.whiterose.ac.uk/106369/1/07055285.pdf)) has each robot push wherever the object hides the goal from it, which is provably correct for convex objects and needs no object pose, but it controls position only.
Its follow-ups extend it to concave objects ([arXiv 2409.02436](https://arxiv.org/abs/2409.02436)) and to chains of sub-goals around obstacles ([arXiv 2605.13006](https://arxiv.org/abs/2605.13006)).

### Pushing from range or depth

EgoPush ([arXiv 2602.18071](https://arxiv.org/abs/2602.18071)) distils a privileged RL teacher online into a depth-only student on a mobile robot.
Its central result shapes the later stages here: three teachers with identical rewards and architecture all exceed 98% success, yet their students reach 0%, 54.8% and 87.3%, and the only difference is how much the teacher is allowed to see.
A teacher whose privileged information the student cannot infer cannot be imitated.
Single-robot full-pose pushing with PPO and an LSTM ([arXiv 2308.02459](https://arxiv.org/abs/2308.02459)) still observes the box pose, and the Push-T benchmark of Diffusion Policy offers only state or pixel observations ([gym-pusht](https://pypi.org/project/gym-pusht/)).

### Active perception, roles and communication

- Observer-Actor ([arXiv 2511.18140](https://arxiv.org/abs/2511.18140)) is the closest conceptual match to the observer and pusher split: on a dual-arm robot, one arm moves its camera to an unoccluded view while the other acts, but it is a single robot trained by imitation.
- Cooperative informative sensing with MARL ([arXiv 2604.23179](https://arxiv.org/abs/2604.23179)) rewards robots for monitoring accuracy under occlusion and limited field of view, which is the template for the observer reward.
- HetGPPO ([arXiv 2301.07137](https://arxiv.org/abs/2301.07137)) shows that heterogeneous policies with GNN communication succeed where shared-parameter policies fail on strongly heterogeneous tasks, and System Neural Diversity ([arXiv 2305.02128](https://arxiv.org/abs/2305.02128)) measures that heterogeneity.
- Role-based MADDPG with scout UAVs ([arXiv 2303.01799](https://arxiv.org/abs/2303.01799)) is a precedent for a sensing role with its own reward.

### Training under partial observability

- Asymmetric actor-critic ([arXiv 1710.06542](https://arxiv.org/abs/1710.06542)) gives the critic the full state while the actor sees only its sensors; MAPPO ([arXiv 2103.01955](https://arxiv.org/abs/2103.01955)) uses an agent-specific global state for its critic.
- Learning by Cheating ([arXiv 1912.12294](https://arxiv.org/abs/1912.12294)) and CTDS ([arXiv 2203.08412](https://arxiv.org/abs/2203.08412)) distil a privileged teacher into a sensor-only student; MAGPO ([arXiv 2507.18059](https://arxiv.org/abs/2507.18059)) argues, as EgoPush finds, that a student cannot copy a teacher whose privileged information is not inferable.
- Success Guided Sampling ([arXiv 2610.12465](https://arxiv.org/abs/2610.12465)) reweights reset configurations toward intermediate success; it is single-agent, tuned for very large batches, and is tested here only as a replacement for the hand-written curriculum.

### Lidar in batched simulators

VMAS ([arXiv 2207.03530](https://arxiv.org/abs/2207.03530)) ray-casts against boxes, spheres and lines and returns distances only, so telling object hits from robot hits needs a second, filtered lidar.
swarp's lidar returns a per-ray hit class next to the range, which is what the observer reward and the object mask in the observation read.

### The gap

Every learned multi-robot pushing result above gets the object pose from motion capture or simulator state, and the sensing-only methods are hand-designed and do not control orientation.
No work treats observer robots as an active-perception role inside a pushing team, and none estimates the pose of a non-convex body such as the T from several lidars.

## Design

The levers the related work points to, and which the training stages add one at a time:

- a critic that sees the true object pose while the actors see only their sensors;
- an auxiliary loss that predicts the object pose in each robot's frame;
- engineered communication first (teammates' object hits pooled in the world frame, which every robot can compute because it knows its own pose), learned attention-based messages after;
- memory, first as a stack of recent scans and then as a recurrent actor;
- one output head per role on a shared trunk;
- observer rewards that are all penalties or potential-based, so that an observer cannot earn reward by prolonging the episode;
- distillation from a teacher whose privileged view is restricted, if the stages above plateau.

## Scenario

`swarp.make("pusht_lidar", n_envs=..., device=...)` builds the task on top of `pusht`: the same T, contact law and pose shaping (it needs `substeps >= 8` at `dt = 0.05`), with five robots by default.
Every `PushTScenario` keyword passes through, and `shape=` takes any compound from `swarp.scenarios.shapes`.

### Robots and roles

The robots are identical, and any of them can push the T.
By default they are holonomic discs commanded by a velocity `(vx, vy)`.
`model="diff_drive"` makes them unicycles commanded by `(v, omega)`, with turn rates up to `max_ang_vel` (default 2π rad/s), and `agent_shape="box"` makes them rectangles of `agent_half_extents` (default `(agent_radius, agent_radius / 2)`, along and across the heading); the two combine, and both pass through to Push-T, which supports them too.
Box robots collide with the T, each other and the walls as oriented rectangles, and the lidar sees them as such.
Either option makes the heading matter: it is drawn uniformly on every reset (by a separate kernel with its own seed, so the default fleet's spawn is unchanged), observed as its cos and sin, and added to each robot's critic-state block.
The taped (gradient) step integrates the T in torch against disc robots, so it refuses box robots; they push it on the no-grad path.
`n_pushers` of them (default 2) are pushers, role 0, and the rest are observers, role 1.
Each reset draws a fresh assignment per env with a masked Fisher-Yates kernel on its own seed, launched after Push-T's spawn kernel, so the counts are exact in every env and an env the reset does not select keeps its roles.
`agent_roles()` is that `[n_envs, n_agents]` int32 tensor, at a fixed address.

### Lidar

Every robot carries a 32-ray lidar over the full circle.
For a holonomic disc it is in the world frame, because such a robot's heading never changes; when the heading matters it is mounted on the body (ray 0 along the heading), so a unicycle sees the T it pushes on the same rays whatever its pose. `lidar_body_frame` overrides either default.
Its range defaults to `0.8 * world_size`: a robot sees the T from about three T radii away but not from across the arena, and the T's narrowest silhouette (about 0.4 m) still spans two to three ray spacings at the edge of the range.
Each ray reports its range and what it hit, so the observation carries an object mask and a robot mask next to the ranges.
The fused path launches the capture-safe Warp lidar inside the whole-step CUDA graph; the torch path scans with the `lidar_scan` oracle against boxes it rebuilds from the body pose, which keeps it differentiable in the T pose.

### Observation levels

`obs_level` sets what a robot observes about the T, and `obs_layout()` returns the slice of every block.
The widths below are for the defaults (`n_agents = 5`, `n_rays = 32`, `frame_stack = 1`, `n_comm_pts = 6`).

| block | levels | width | content |
|---|---|---|---|
| self | all | 9 (11) | own position and velocity, role sign (+1 observer, −1 pusher), goal position relative to the robot, cos and sin of the goal heading; then cos and sin of the robot's heading when it matters (unicycles, boxes) |
| teammates | all | 12 | per teammate in ascending index, self skipped: relative position and role sign |
| T | 0 | 8 | Push-T's eight numbers: T relative to the robot, goal relative to the T, cos and sin of the T heading and of its heading error |
| lidar | all | 96 per frame | range / max range, object mask and robot mask per ray; `frame_stack` keeps a ring of the last scans, oldest first, zeroed at reset |
| pooled hits | 2 | 90 | per robot (self included, ascending index), `n_comm_pts` of its object-hit points relative to the observer, each with a valid flag |

That makes 125 at level 0, 117 at level 1 and 207 at level 2, two more each when the heading is observed.
Level 0 is the oracle stage; level 1 is a robot's own sensing; level 2 adds the engineered communication of the design list above.
A pooled point is `j`'s object hit number `floor((k + 0.5) n_hit / K)` in ray order, placed in the world frame, which every robot can do because it knows its own pose; it is invalid when `j` sees no part of the T or is farther than `comm_range` (default unlimited).

### Rewards

Every term is reported per robot in `info()["multiobj_reward"]` (`[n_envs, n_agents, 7]`, zero where a role does not apply), and the columns sum to the reward.

| term | who | definition | default |
|---|---|---|---|
| `team` | everyone | Push-T's pose shaping plus its goal bonus | Push-T's weights |
| `approach` | pushers | Push-T's per-robot approach to the push point | `agent_dist_shaping = 1.0` |
| `standoff` | observers | potential difference of `−abs(‖p − T‖ − d*)` | `standoff_shaping = 1.0`, `standoff_dist = 0.5` |
| `visibility` | observers | `−w_v (1 − min(n_obj / (f* R), 1))`, `n_obj` the rays that hit the T | `w_v = 0.02`, `f* = 0.1` |
| `coverage` | observers | `−w_c (1 − c)`, `c` the share of 8 sectors around the T centroid (T frame) holding an observer's object hit | `w_c = 0.02` |
| `occlusion` | observers | `−w_o` times the share of the observer's rays inside the T's angular cone (its bounding disc) that hit a robot first | `w_o = 0.1` |
| `contact` | observers | `−contact_penalty` while an object ray is shorter than two robot radii | `0` |

The observer terms are penalties or potential differences, so an observer never earns reward by prolonging an episode.
`info()` also carries `team_coverage`, `observer_visibility` and `occlusion` per env and `tee_seen` per robot, next to Push-T's `tee_dist_to_goal` and `tee_angle_error`.

The weights were set from a random-policy rollout (512 envs, 400 steps, the recipe's Push-T shaping), per robot-step:

| term | role | mean | mean of the magnitude |
|---|---|---|---|
| `team` | both | −0.0015 | 0.0031 |
| `approach` | pushers | −0.0001 | 0.0232 |
| `standoff` | observers | −0.0001 | 0.0228 |
| `visibility` | observers | −0.0106 | 0.0106 |
| `coverage` | observers | −0.0119 | 0.0119 |
| `occlusion` | observers | −0.0035 | 0.0035 |

An observer's terms total about −0.028 per step against −0.0016 for a pusher's, and each is within a factor of seven of the pushers' approach term.
Occlusion is rare under a random policy (11% of observer steps, covering a third of the cone when it happens), so its weight is five times the others': a blocked view then costs about 0.033 per step, the size of the approach term's 0.023.

### Training hooks

The observation kernel writes the three trainer buffers on every pass, the obs-only pass after an auto-reset included, so they always describe the state the observation does.

- `critic_state()`, `[n_envs, 15 + 5 n_agents]` (`15 + 7 n_agents` when the heading is observed): the T's position, the goal relative to it, cos and sin of its heading and heading error, its velocity and angular velocity, the goal pose, then per robot in role-sorted order its position, velocity, role sign and, when it matters, cos and sin of its heading.
- `agent_roles()` with `role_counts = (n_pushers, n_agents − n_pushers)` and `n_roles = 2`.
- `aux_targets()`, `[n_envs, n_agents, 4]`: the T's position relative to each robot divided by the lidar range, and cos and sin of its heading.

`SPECS["pusht_lidar"]` is stage T0: Push-T's recipe with five robots at `obs_level=0`, a `PolicyNet` with one head per role, and a per-slot `StateCritic` on `critic_state()`, with the auxiliary loss off.

`SPECS["pusht_lidar_limo"]` (`python -m swarp.rl --scenario pusht_lidar_limo`) builds the same scenario at the physical scale of two AgileX Limo robots, in metres and seconds, for deployment in Gazebo: 0.322 × 0.220 m unicycle boxes at up to 0.4 m/s and 1.5 rad/s, a 1 m T in a 4 × 4 m arena, a 1.5 m lidar, a 5 Hz policy acting on the latest 10 Hz scan (`dt = 0.2`, 32 substeps to keep Push-T's contact sub-step) and 450-step (90 s) episodes.
The policy runs at 5 Hz rather than at the lidar's 10 Hz because a 10 Hz action moves a Limo only 0.04 m (4% of the T), against 12.5% of the T per action in U-H1, and the credit per action was too small to learn from.
At `dt = 0.1` (16 substeps, 900 steps, gamma 0.995) seed 0 reached a greedy solve of 0.04 at iteration 1000, 0.36 at 6000 and 0.54 after 12000, and its greedy robots stood still in 63% of robot-steps while the sampled policy solved 0.93 within 1800 steps.
Lowering the entropy coefficient to 1e-3, normalizing advantages, a per-step gamma of 0.99 and a 0.8 m/s robot each stayed between 0.06 and 0.15 greedy at iteration 2000, while `dt = 0.2` reached 0.56 at iteration 1000 and 0.85 at 2000.
A deployment holds each command for 0.2 s and advances the frame stack once per policy step, so four frames span 0.8 s.

`SPECS["pusht_lidar_limo_dr"]` adds what a sysid against Gazebo (standing in for the real Limo) found missing.
The robots weigh 6.7 instead of 1 (`agent_mass`), so a velocity-mode pusher keeps 95% of its commanded speed in contact (Gazebo's servoed robot keeps all of it, swarp's kept 75%) and the T stops where Gazebo's does.
Lowering the T's drag to 6/s matched the push speed as well, but the T then coasted and spun on, and training reached 0.08 greedy at iteration 2000 against 0.98 with the heavier robots.
The robots drive through `ControlMode.SERVO` with accelerations and command gains drawn per env and robot on every reset (`servo_accel` 0.6–2.0 m/s², `servo_ang_accel` 2–6 rad/s², `cmd_gain_v` 0.9–1.05, `cmd_gain_w` 0.7–1.0); Gazebo's Limo sits inside every range (1 m/s², 3 rad/s², 1.0, 0.79).
The lidar takes a sensor model: rays are still cast from the robot's centre, but a hit within `lidar_min_range` (0.10 m) of a mount `lidar_mount_offset` (0.103 m) ahead of the centre reads as a miss, and the remaining hits carry Gaussian range noise of `lidar_noise` (0.01 m).
`lidar_fov` below 2π would also blank the rays behind the robot; the recipe leaves it at 2π.
A deployment that bins the real scan into the 32 centre rays needs no change for any of this, because the real scan already lacks the returns inside the blind zone and already carries its noise.

`SPECS["pusht_lidar_limo_dr_coulomb"]` puts the T on a Coulomb floor (`floor_friction`, μ drawn per env from 0.2–0.5) with no viscous drag, and robots of mass 2.
A viscous drag under-rotates an off-centre push: the quasi-static pushing reference turns the bar 64°/m at a 0.2 m offset and 145°/m at 0.4 m, and 39 and 113°/m on a viscous floor.
On the Coulomb floor swarp gives 60, 142 and 48°/m (glancing push; reference 48), and the T stops where Gazebo's does.

`SPECS["pusht_lidar_limo_dr_wheel"]` adds wheel-faithful contacts (`wheel_contacts`, see [World](world.md#box-agents)): a robot's contact torque turns it only past the 3.5 N·m its drive holds, and its wheels grip sideways with μ 0.2.
A policy trained without them (L3b, 0.99 greedy in swarp) solved 0.35 in Gazebo: its two robots squeezed the T head-on from opposite faces and spun in place, which a Limo cannot do, because its flush front corner would have to swing through a face the other robot holds.
With wheel contacts, two robots pressing the T from opposite faces hold their yaw as in Gazebo (−1° and −2° of a commanded +258°; Gazebo 0°), and a robot turning against the free T still turns (+199°; Gazebo +198°).
Trained from scratch the recipe reaches 0.96 greedy at 16000 iterations; resumed from the Coulomb policy it reaches 0.99 in 8000.
Its comment in `swarp/rl/recipes.py` derives every value that differs from U-H1's.
A recipe keyed by a name other than its scenario's sets `TrainSpec.scenario`, which `scenario_name` resolves wherever a scenario is built from a `SPECS` key.

## Results

Every number below is the greedy solve rate at full curriculum difficulty over 512 evaluation envs (the Wilson interval is about ±0.01 near 0.98 and ±0.04 near 0.6), from `python -m swarp.rl --scenario pusht_lidar` with the recipe's 4000 iterations and 512 training envs.
The robots are holonomic discs except in the last table.

### Default arena

Lidar range 0.8, two pushers and three observers unless noted, one seed each.

| run | iter 1000 | final (iter) |
|---|---|---|
| oracle T pose (level 0) | 0.47 | 0.99 (4000) |
| own lidar (level 1), `frame_stack` 4, aux loss 0.1 | 0.81 | 0.99 (4000) |
| pooled hits (level 2), `frame_stack` 4, aux loss 0.1 | 0.90 | 1.00 (4000) |
| level 1 with lidar range 0.05 (leak control) | 0.004 | stopped at 1500 |
| level 1, `frame_stack` 1 | 0.81 | 0.98 (2500) |
| level 1, no aux loss | 0.86 | 0.96 (2000) |
| two pushers, no observers | 0.96 | 0.97 (1500) |
| five pushers, no observers | 0.24 | 0.95 (2000) |

The leak control is the check that matters most: with a lidar too short to see the T the team never learns, so the policies above do not read the T pose from anywhere but their scans.
Otherwise this arena is too easy to separate the variants: the T is almost always in some robot's view, and memory, the aux loss and observers barely change the outcome.

### Short range

Lidar range 0.3 and observer standoff 0.4, level 1 or 2 with `frame_stack` 4 and aux loss 0.1, three seeds each (0, 1, 2).

| run | iter 1000 | iter 1500 | iter 4000 |
|---|---|---|---|
| two pushers only | 0.62 / 0.83 / 0.92 (mean 0.79) | 0.83 / 0.91 / 0.96 (0.90) | 0.95 / 0.99 / 0.99 (0.98) |
| + three observers, own lidar | 0.77 / 0.81 / 0.46 (0.68) | 0.80 / 0.96 / 0.88 (0.88) | 0.99 / 1.00 / 0.98 (0.99) |
| + three observers, pooled hits | 0.69 / 0.61 / 0.63 (0.64) | 0.91 / 0.90 / 0.91 (0.91) | 0.98 / 0.98 / 0.98 (0.98) |
| + three observers, own lidar, attention communication | 0.87 / 0.14 / 0.74 (0.58) | 0.95 / 0.91 / 0.93 (0.93) | 0.98 / 0.99 / 0.98 (0.99) |
| + three observers, pooled hits, GRU (seed 0 only) | 0.80 | 0.94 | 0.97 |

Every variant ends between 0.975 and 1.000 on every seed, so at 4000 iterations observers, pooled hits and learned communication neither help nor hurt.
Early on the seed spread is wider than any difference between variants: seed 0 alone suggested attention communication > pooled hits > own lidar ≈ pushers only, but at iteration 1000 the three-seed means run the other way round (pushers only 0.79, own lidar 0.68, pooled hits 0.64, attention 0.58, whose seed-1 run sat at 0.14), and by iteration 1500 all four are within 0.05 of each other.
At this range two pushers still find and turn the T on their own, so a regime that needs the observers has to make the T harder to keep in view: a shorter range, clutter that occludes it, or a larger arena.

An earlier short-range batch with observer standoff 0.22 is excluded: the standoff sat inside the T's 0.265 corner radius, so the shaping parked observers against the T and they jammed its rotation (the angle error stayed near 1.6 rad).
The scenario now refuses a standoff below `tee_radius + agent_radius`.

### Unicycle box robots

The same recipe with `model="diff_drive"` and `agent_shape="box"` (5 × 2.5 cm rectangles commanded by `(v, omega)`, body-frame lidar), level 1, `frame_stack` 4, aux loss 0.1, seed 0.

| run | iter 1000 | iter 2000 | iter 4000 |
|---|---|---|---|
| short range, two pushers only | 0.80 | 0.98 | 0.99 |
| short range, + three observers | 0.24 | 0.52 | 0.94 |
| default range, + three observers | 0.08 | 0.47 | 0.95 |

Two unicycle pushers learn the task as fast as holonomic ones.
With observers the team gets there about 1000 iterations later than the holonomic team and ends four to five points lower, still rising.
The observers keep the T in view better than holonomic ones do (visibility 0.85–0.96 against 0.67–0.75 at iterations 500–1500), while the team reward lags at a third of the holonomic run's: in rollouts the observers bunch on one side of the T, often between it and its goal, and a unicycle cannot sidestep out of a pusher's way.

### Gazebo transfer

Each Limo-scale policy was also run greedy in Gazebo Harmonic on simulated Limo robots, with the observation rebuilt from the Gazebo lidar scan and robot poses (`swarp_gz_test`, results and method in its `docs/results.md`).

| policy | recipe | Gazebo | swarp |
|---|---|---|---|
| L2 | `pusht_lidar_limo` | 0.567 | 0.857 |
| L3b | `pusht_lidar_limo_dr` (robot mass 6.7) | 0.350 | 0.986 |
| L4 | `pusht_lidar_limo_dr_coulomb` | 0.883 | 0.973 |
| L5 | `pusht_lidar_limo_dr_wheel`, from scratch | 0.800 | 0.959 |
| L5r | `pusht_lidar_limo_dr_wheel`, resumed from L4 | 0.867 | 0.992 |
| L5s | `pusht_lidar_limo_dr_wheel` with the turn stall, resumed from L5r | 0.917 | 0.986 |

L3b fails mostly by the two-robot squeeze: in 32 of its 60 Gazebo episodes the robots lock each other's yaw against the T, which swarp let them spin out of before wheel-faithful contacts.
L5s sampled solves 0.917 in Gazebo and 0.998 in swarp; four of its five sampled failures are the T leaving the ±2 m arena, which swarp clamps inside and Gazebo does not wall.
The deployment recipe is `pusht_lidar_limo_dr_wheel`, and the deployment checkpoint is L5s (`runs/L5s-wheelstall-limo-s0-fromL5r/pusht_lidar_limo_dr_wheel_best.pt` on davide-cf, pulled to `runs/remote-davide-cf/`).

### Real Limo recipe

`SPECS["pusht_lidar_limo_real"]` prepares `pusht_lidar_limo_dr_wheel` for the real AgileX Limos (EAI T-mini Pro lidar, localised by Qualisys mocap or by nav2 AMCL); every knob is opt-in and leaves the default worlds bit-identical.
The lidar returns from 0.02 m of its mount (`lidar_min_range`), with range noise drawn per env and robot from 5–20 mm (`lidar_noise` as a range) and a field of view drawn per env and robot from 240–360° (`lidar_fov` as a range).
The field of view keeps body-frame ray `r` iff its offset `r · 2π/32`, wrapped to (−π, π], is within half the field of view plus 1e-4 rad, whatever the heading: 240° keeps rays −10…10 (21 rays), 270° keeps −12…12, and a deployment blanks the same rays for its measured field of view.
`pose_noise` adds a localisation error to the observation only: a per-episode bias per robot (3 cm per axis and 0.03 rad at scale 1, `pose_bias_std`), per-step jitter (2 cm and 0.02 rad, `pose_jitter_std`) and the velocity turned by the heading error plus 2 cm/s (`vel_noise_std`), all times a scale drawn per env from 0–1, so mocap (scale 0) and AMCL (scale 1) are both in the distribution.
The self block, the goal offset and the teammate offsets read each robot's own reported pose, so a teammate offset carries both robots' errors, while the lidar stays in the true body frame (the sensor is bolted to the body, so a heading error shows up as a mismatch between the scan and the goal direction, not as a rotated scan); reward, success, the critic and the auxiliary targets read the true state.
`cmd_latency` (0–125 ms, drawn per env and robot) applies the previous command for that many of the step's 32 substeps (`Stepper.enable_command_delay`), standing in for WiFi and processing delay.
Fine-tuned from L5s for 8000 iterations (L6, fresh Adam, full difficulty), the greedy solve rate climbed from 0.73 at iteration 500 to 0.91 at 3500 and stayed at 0.92 to 8000.

| condition | L5s greedy | L5s sampled | L6 greedy | L6 sampled |
|---|---|---|---|---|
| training distribution | 0.506 | 0.568 | 0.949 | 0.980 |
| mocap-like (no pose noise, 360°, σ 1 cm, no latency) | 0.961 | 1.000 | 0.926 | 0.990 |
| AMCL-like (pose scale 1, 240°, σ 2 cm, 100 ms) | 0.119 | 0.113 | 0.896 | 0.916 |
| AMCL pose noise only | 0.969 | 1.000 | 0.924 | 0.984 |
| 240° field of view only | 0.094 | 0.113 | 0.826 | 0.936 |
| 100 ms latency only | 0.980 | 0.992 | 0.967 | 0.988 |

All rows are 512 envs at full difficulty with L6's `_best.pt`; the remaining knobs are the recipe's (`lidar_min_range` 0.02, the rest at the values listed).
L5s ignores AMCL-scale pose noise and 100 ms of latency, but a 240° field of view breaks it, so the rear blind sector is what the real robots needed training for.
L6 pays about 0.035 greedy in the mocap-like condition for it, while its sampled policy holds 0.99.
The checkpoint is `runs/L6-real-limo-s0-fromL5s/pusht_lidar_limo_real_best.pt` on davide-cf, pulled to `runs/remote-davide-cf/`.

### Slim T at 240°, from scratch

`SPECS["pusht_lidar_limo_slim240"]` (L8) trains from scratch for the setup the real robots meet, after L6 and L7 solved only 0.05–0.13 in Gazebo at 240° (swarp_gz_test `docs/results.md`).
It builds on `pusht_lidar_limo_amcl` (L7: the field of view cut at the mount, AMCL-sized localisation drift):

- an open floor (`open_arena`): no walls and no clamp on the T, the robots stopped by the safety bridge's arena stop 0.25 m past the arena, and an episode failed once the T's centroid is 0.5 m past it (`oob_margin`, −0.05 per robot and step from then on);
- the real T: two 0.50 × 0.04 m bars, the stem hanging from the bar's middle;
- 43 rays over ±120° (`lidar_span`), the 5.6° spacing of 64 rays over the full circle, with the field of view fixed at 240° and cut at the mount.

The scan gains what a real T-mini Pro and the deployment add (`swarp/scenarios/pusht_lidar_artefacts.py`):

- **Labels from the reported poses** (`lidar_label_from_pose`). The deployment (`label_rays`) can tell a teammate from the T only by the teammate's reported footprint, grown by 3 cm. Under the recipe's pose noise a teammate hit off that footprint reads as the T: in swarp that moves 28% of the teammate hits into the T mask, which training on the true classes never saw.
- **Lost and spurious returns.** A T hit at incidence cosine `c` is lost with probability `1 − c / cos 80°` (`lidar_grazing_cos`), any hit with a per-robot probability drawn from 0–3% (`lidar_dropout`), and any ray returns a spurious short hit, labelled T as the deployment would, with a per-robot probability drawn from 0–0.5% (`lidar_spurious`).
- **Range errors.** A per-robot scale error of up to 1% and bias of up to 1 cm (`lidar_range_scale`, `lidar_range_bias`), the datasheet's accuracy.
- **The rotating scan** (`lidar_scan_period` 0.1 s, the driver's 10 Hz). Each ray is cast from the pose at the time the sensor took it, extrapolated at constant velocity (`swarp.sensors.lidar.sweep_offsets`): at 1.5 rad/s a scan turns by 0.15 rad, nearly three ray spacings. The stamp's place in the sweep is drawn per robot over the whole sweep (`lidar_stamp_phase`), since neither the driver's stamp convention nor the pose's latency is measured, and so is a mount yaw error of up to 2° (`lidar_yaw_bias`). The sweep is counter-clockwise from the rear; the real sensor's direction in time order is still to be measured.

`critic_dr` appends every randomised parameter to the critic state, scaled to about [−1, 1] and in role-sorted order: the floor friction, and per robot the drive's accelerations and gains, the command delay, the range-noise deviation, the pose-noise scale, the localisation bias and drift, and the artefact values.
The value then no longer has to average over drives, delays and sensors it cannot see.

`action_rate_coef` 0.03 takes `0.03 · mean_i((a_t − a_{t−1}) / half_i)²` off every robot's reward, with `a_{t−1} = 0` on an episode's first step.
That is 16% of the team term at L7's sampled jitter (0.30 per step against a team term of 0.056 at full difficulty) and 4% at its greedy one (0.055 against 0.043).
It ramps in over the first 2000 iterations: the initial policy's change is 0.73, a penalty of 0.022 per step against team and approach terms of about 0.001.

From scratch the recipe learns slowly and unstably: run S240-e512-s0 (seed 0, 512 envs, 16000 iterations) peaked at 0.18 greedy and ended at 0.148 (0.123 with the artefacts off, 0.143 with mocap-like poses), where L5 had 0.98 at the same length.
Its greedy policy is 3.6× smoother than L7's (mean squared normalised action change 0.015 against 0.055; sampled 0.090 against 0.30).
Pilots at 512 envs unless noted, seed 0, greedy solve at full difficulty (S240-* on davide-cf), separate the causes:

| run | change from the recipe | greedy (iter) |
|---|---|---|
| D | the 1.0 / 0.7 × 0.3 m T | 0.037 (1000), then collapses to 0.000 (3000) |
| N2 | old T, no pose noise or latency | 0.281 (3500), then 0.018 (4000) |
| N1 | no pose noise or latency | 0.002 (1500) |
| O3 | N1 with the T's pose observed (obs level 0) | 0.006 (1500) |
| G1, G3 | curriculum gated at 0.5 | 0.000 (1500, stuck at difficulty 0, two seeds) |
| R1 | N1 with `rot_away_penalty` 0.25 | 0.021 (3000) |
| R3 | N1 with `rot_away_penalty` 0 | 0.244 (4000) |
| R4 | `rot_away_penalty` 0, full noise | 0.000 (2000) |
| R5 | R3 with 2048 envs (16 minibatches) | 0.209 (1000) |

- **The slim T is a manipulation problem, not a perception one**: the oracle fails as the lidar policy does (O3, N1). It turns 3–6× more per metre pushed than the old T at the same offset (48 against 18°/m at 0.1 m, 266 against 50 at 0.2 m), so with `rot_away_penalty` 1.0 a clumsy push costs 1.5 per radian it turns the T the wrong way and earns 0.5 per radian back. Policies settle on approaching the T and not pushing it: under D's collapsed checkpoint the T travels 0.10 m per episode against 1.55 m at iteration 1000. The out-of-bounds failure is not it (0.2% of episodes).
- **AMCL-sized pose noise and latency block learning from scratch** with either T (D against N2, R4 against R3). L6 and L7 only ever met them as a fine-tune.
- More envs help: R5 reaches in 1000 iterations what R3 does in 4000.

The recipe is trained in stages instead, each resumed from the last one's checkpoint (`--resume <checkpoint> --start-difficulty 1` from stage 2 on), 2048 envs and 16 minibatches throughout (L8b-* on davide-cf, seed 0):

| stage | recipe and change | iterations | greedy solve |
|---|---|---|---|
| 1 | `pusht_lidar_limo_slim240_s1`: no pose noise or latency, `rot_away_penalty` 0 | 6000 | 0.982 (3500), collapsed to 0.004 by 6000 |
| 2 | `pusht_lidar_limo_slim240_s2`: the full recipe, `rot_away_penalty` 0, lr 2.5e-4 | 8000 | 0.988 |
| 3 | stage 2 with `--scen-kwarg rot_away_penalty=1.0` | 4000 | 0.97–0.99 |
| 4 | stage 3's last checkpoint, `rot_away_penalty` 1.0 again | 4000 | 0.97–0.98 |
| 4′ | the same with `rot_away_penalty` 2.0 | 4000 | 0.78–0.97, unstable |

Stage 1's noise ranges are `(0, 1e-9)` rather than off, so its critic has the same randomisation columns as stage 2's and the checkpoint resumes as it is.
With `rot_away_penalty` 0.25 instead of 1.0 stage 3 drifted down to 0.904.

`SPECS["pusht_lidar_limo_slim240_1run"]` runs the same stages as one training (`python -m swarp.rl --scenario pusht_lidar_limo_slim240_1run`, 14000 iterations, about 55 minutes at 230 ms an iteration), with `TrainSpec.ramps` in place of the resumes.
Stage 1's world holds until iteration 3000, the pose-noise scale and the command delay ramp in by 5000, `rot_away_penalty` by 10000, and the learning rate decays linearly from 5e-4 to 1e-4 over the run.
Each `Ramp` moves in 20 increments and the trainer recaptures the env's graph on each, because a captured step keeps the reward weights it was captured with.
The best checkpoint is ranked on ramp progress before the solve rate, so it comes from the final conditions.
It has not yet been trained to completion.

`examples/eval_mappo.py`, 512 envs at full difficulty, greedy, on stage 2's best checkpoint and on stage 4's at iteration 3000:

| condition | stage 2 | stage 4 (3000) |
|---|---|---|
| the recipe | 0.992 | 0.973 |
| artefacts off | 0.996 | 0.980 |
| mocap-like poses (`pose_noise` 0, `cmd_latency` 0) | 0.996 | 0.973 |
| the worst AMCL (`pose_noise` 1.25, `cmd_latency` 0.1) | 0.980 | 0.959 |
| mean squared normalised action change, greedy (L7: 0.055) | 0.033 | 0.034 |
| net turn of the T before the solve, median | 10.2 rad | 7.7 rad |

Both policies spin the T: the median episode needs 1.57 rad of heading correction and the team turns the T 1.2–1.6 times round in one direction before stopping it on the goal heading (the net signed turn equals the largest excursion, so it is not wobble).
The reward allows it: `angle_error` is wrapped to [0, π], so heading progress pays back on the way down what it charged on the way up, and a whole turn costs only `rot_away_penalty · π`, about 0.6 m of position progress at 1.0.
Raising the turn-away cost trades solve rate for little less spin (stage 4′ at 2.0: 6.2 rad at 0.95 greedy, then falling to 0.86), so a penalty on the T's angular speed, which a whole turn cannot repay, is the open lead.
Stage 2's best checkpoint is the one to deploy until then; stage 4's spins a quarter less at two points of solve rate.

### Variable arena and surroundings, in one run

`SPECS["pusht_lidar_limo_slim240_arena"]` (L9) trains the slim T across arena sizes and whatever stands around the arena, from scratch and as one run.

**Arena.**
`arena_size` gives every env its own rectangular arena, redrawn on every reset: `(2.0, 5.0)` draws the width and the height from 2–5 m independently (a 4-tuple gives each axis its own range, a value fixes a square).
The self block ends with the env's half-extents `(W/2, H/2)` in metres, the units of the observed position, so `half − |x|` is the distance to the edge.
The spawns, the arena stop (`Stepper.enable_arena_stop` with a per-env edge array), the out-of-bounds limit and the label bound are each env's own, per axis: the half-extents plus `arena_stop_margin`, plus `oob_margin`.
Robots spawn clear of the T and of each other even in a 2 × 2 m arena (rejection over uniform draws, then rings around the T), and a goal outside the arena is redrawn rather than clamped onto its edge.

**Surroundings.**
`scenery_probs` draws per env one of four layouts outside the arena: nothing, four walls parallel to its sides, a room at a random yaw that contains it, or walls plus up to `n_clutter` circles and boxes (`swarp/scenarios/pusht_lidar_scenery.py`).
Each wall stands its own gap (`scenery_gap`, 0–1 m) past the free region and may be missing (`wall_missing`, 0.2).
Only the lidar sees the surroundings (a `swarp.sensors.Scenery`, per-env shapes the stepper's contacts never read), so they stand outside the free region: the arena grown by the farther of a robot's reach past the arena stop and the T's past the out-of-bounds limit, 0.89 m for the recipe (`scenery_clearance`).
No robot or T can touch them before its episode fails, which is what makes lidar-only geometry exact; a real floor must keep the same 0.89 m clear at lidar height, or the deployment meets what training never showed.
`scenery_on` scales the share of envs that have any, so a trainer can ramp them in.

What the surroundings can teach is bounded by the deployment's labels.
`label_rays` turns every hit past the label bound (the arena plus `oob_margin`, from the robot's reported pose) into a miss, and swarp trains on the same rule, so under mocap-grade poses nothing outside the arena reaches the observation.
Under AMCL-sized localisation error a wall near the bound lands inside it and is labelled as the T; `info()["scenery_leak"]` is the share of rays where that happens, and that leak is what training with surroundings makes the policy robust to.
The arena sizes change the task itself: the policy has to keep the T inside an edge that is now its own per env.

**Field of view.**
Both Limos' ydlidar drivers crop the scan to ±110° (`angle_min`/`angle_max`), read off the running robots, with returns at every bearing inside it.
The recipe sees 220° cut at the mount with 40 rays (`lidar_span`), the 5.6° spacing of the 240° recipes, so its observation is narrower than theirs and its checkpoint does not load into them.
The two robots' mount transforms disagree with each other and with the 0.103 m `lidar_mount_offset` from the Gazebo identification (one publishes 0.18 m ahead of `base_footprint`, the other two different `laser_link` poses); the recipe keeps 0.103 m.

**One run.**
The recipe is `pusht_lidar_limo_slim240_1run`'s, with five guards against the collapses the staged runs met (`swarp.rl`, all off by default):

- the ramps follow a gated clock (`ramp_gate` 0.75): past the first ramp's start a tick needs a windowed solve of at least 0.75, so no condition comes in before the last is mastered; the learning rate decays over that clock, and the run ends `hold_iters` (4000) past the last ramp, `iters` (24000) being a cap;
- an adaptive learning rate (`adaptive_kl` 0.02, rsl_rl's schedule once per iteration): the factor on the scheduled rate is divided by 1.5 after an iteration whose mean approximate KL to the rollout policy exceeds 0.04 and multiplied by 1.5 (up to 1) after one below 0.01; a first run with only the skip brake at 0.1 froze 71% of iterations 1000–1500, where the entropy fell to −1.6 and two or three Adam steps already pushed the KL past 0.15;
- a floor of 0.1 under the actor's std (`actor_scale_min`, 0.01 by default): with the adaptive rate alone the policy still sharpened, its entropy falling from −1 to −4.2 over iterations 500–3000 while the rate fell to 6% of its schedule and the training solve stalled at 0.16 in the clean 4 × 4 m world; the greedy action is the mean, so the floor does not reach the deployment;
- a KL brake (`target_kl` 0.3) for a single destructive iteration: once a minibatch's approximate KL exceeds 1.5 times it, the iteration's remaining minibatch steps are skipped on device;
- automatic rollback (`rollback`), armed once the windowed solve has passed 0.75: when it then stays below `max(0.5, best − 0.25)` for 25 iterations, the actor, critic and Adam state return to a snapshot from before the drop, the learning rate halves, and the ramps return to that snapshot's clock; a fifth collapse stops the run on its best checkpoint.

L8b-s1 is the case they answer: it held 0.95–0.97 for 2500 iterations, then fell 0.83 → 0.22 in 60, with no step non-finite and its entropy rising only after the fall.
The arena widens from 4 × 4 m over ticks 1000–3000, in stage 1's clean world, and the surroundings come in with the pose noise and the latency (3000–5000), since only the pose error lets them leak.

**Result (L9c).**
`python -m swarp.rl --scenario pusht_lidar_limo_slim240_arena --eval-every 500` (seed 0, davide-cf) trained in one run of 15323 iterations, about 58 minutes at 227 ms an iteration, with no rollback, no braked minibatch and the learning rate at its schedule for the last 1000 iterations.
The training solve reached the 0.75 gate at iteration 2000, the arena had widened by 4300, the noise, latency and surroundings were in by 6400 and the turn-away cost by 11400; the in-training greedy eval at the final conditions passed 0.70 at 3500 and held 0.984–0.994 from 8000 on.
Two earlier attempts on the same recipe did not get there: with the actor's std floored at 0.01 the policy sharpened to an entropy of −4.2 and stalled at 0.16, and L9b, with the floor at 0.1, learned as L9c did until the rollback guard, then armed at 0.5, took its rise through 0.38–0.51 for a collapse and stopped it at iteration 1656 (the guard now arms at 0.75).

`examples/eval_mappo.py`, 1024 envs, greedy, on the best checkpoint (95% Wilson intervals):

| condition | solved | median solve step |
|---|---|---|
| the recipe (2–5 m, every layout, the full noise) | 0.987 [0.978, 0.993] | 116 |
| 2 × 2 m | 0.999 [0.994, 1.000] | 80 |
| 4 × 4 m | 0.990 [0.982, 0.995] | 130 |
| 5 × 5 m | 0.922 [0.904, 0.937] | 157 |
| 2 × 5 m | 0.964 [0.951, 0.974] | 122 |
| no surroundings | 0.990 [0.982, 0.995] | 116 |
| walls plus clutter | 0.989 [0.981, 0.994] | 115 |
| mocap-like (`pose_noise` 0, `cmd_latency` 0) | 0.995 [0.989, 0.998] | 110 |
| the worst AMCL (`pose_noise` 1.25, `cmd_latency` 0.1) | 0.983 [0.974, 0.990] | 126 |
| the worst AMCL, walls plus clutter, 5 × 5 m | 0.915 [0.896, 0.931] | 173 |

At 4 × 4 m it matches L8b stage 2 (0.992 on its own fixed arena), at 220° instead of 240°.
The surroundings cost nothing (0.990 without, 0.989 with walls and clutter), and `scenery_leak` stayed below 1e-4 of the rays over the last 1000 iterations: at the trained pose noise (centimetres) the label bound drops almost every surroundings hit, so the policy is robust to them because the deployment's labels are, not because it learnt to read them.
A localisation error of tens of centimetres near the bound, as AMCL reaches at its worst, is outside what this training saw.
The 5 × 5 m arena is the weak case (0.92): its goals are the farthest, the median solve takes 157 of the 450 steps, and the failures there are not broken down here.

The deployment side (`swarp_gz_test`: the observation's arena block, per-axis bounds in `label_rays`, the safety bridge and the localisation mask) is not updated yet; until it is, this checkpoint cannot run on the robots.
