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
