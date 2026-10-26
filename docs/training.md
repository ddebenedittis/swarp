# Training

`swarp.rl` is a MAPPO trainer whose rollout and update are both CUDA-graph captured.
It needs only torch: no torchrl, no tensordict.
It trains every built-in scenario that has a success criterion from the recipe table that produced the published policies.

```bash
python -m swarp.rl --scenario navigation
python -m swarp.rl --scenario giveway --compile --tf32     # the fastest configuration
```

Checkpoints and a `metrics.csv` land in `runs/<scenario>/`, as with the TorchRL trainer.
Evaluate a checkpoint against random, zero-action and scripted baselines with `examples/eval_mappo.py`, which reads both trainers' checkpoints:

```bash
python examples/eval_mappo.py runs/navigation/navigation_final.pt --n-envs 2048
```

## Why another trainer

The simulator is a small fraction of what a training run costs.
At the recipes' batch sizes the physics of one rollout takes 2–4 ms, while an iteration of the TorchRL trainer (`examples/train_mappo.py`) takes 240–440 ms.
The rest is per-step TensorDict construction and cloning, host syncs in the collector's reset check and in every `TanhNormal` constructor, and an eager update with an `.item()` per minibatch.

`swarp.rl` keeps the algorithm and removes the overhead:

- **Rollout.** Each env step is one replay of an *act* graph (actor forward, sample, write the action and the rollout slot), one `Environment.step` (its own Warp graph), and one replay of a *store* graph (reward, flags and logged `info()` means into the slot).
- **Update.** One replay of a *prepare* graph (critic over the rollout, GAE, the epochs' minibatch permutations), then one replay of a *minibatch* graph per minibatch (gather, forward, loss, backward, gradient clipping, fused Adam step).
- **One host sync per iteration**, reading every logged statistic at once.

The environment runs with `auto_reset=True`, in-graph where the scenario supports it, so nothing on the host looks at a done flag.

## Speed

`python -m swarp.benchmark.train_throughput` runs each trainer through its CLI and reads the per-iteration wall time it writes to `metrics.csv` (collection plus update; logging, eval and checkpoints excluded).
At the recipes' batch sizes, on an RTX 3070 Laptop GPU, in ms per iteration:

| trainer | `navigation`, 512 envs × 32 steps | `giveway`, 1024 envs × 32 steps |
|---|---|---|
| TorchRL as usually written (`examples/train_mappo.py`) | 224 | 452 |
| BenchMARL MAPPO | 274 | 492 |
| rsl_rl, one env per agent | 164 | 315 |
| TorchRL with its speed features (`examples/train_mappo_tuned.py`) | 129 | 242 |
| the same, `--compile --tf32` | 107 | 190 |
| `swarp.rl` (graphs) | **97** (169 k env-steps/s) | **188** (174 k) |
| `swarp.rl --compile --tf32` | **72** (228 k) | **144** (228 k) |

With the overhead gone the iteration is GPU-bound in the PPO update: 8 epochs over the batch through two 256-wide MLPs.
The physics is 1–2% of an iteration even in the fastest configuration, which is why `--tf32` (faster matmuls) and `--compile` (Inductor fuses the elementwise work between them) are the knobs that still move the number.
TorchRL with its own CUDA-graph and compile options gets within 1.3–1.5× of `swarp.rl` at these sizes; the rest is its per-step env machinery, which a TorchRL user cannot take over.
The speedup carries through to time-to-solve: on `giveway`, over three seeds, every trainer reaches full curriculum difficulty at iteration 413–417, `swarp.rl` in a median 78 s of wall-clock against 102 s for tuned TorchRL, 132 s for rsl_rl, 175 s for TorchRL as usually written and 207 s for BenchMARL, and every final policy solves all 2048 evaluation envs.
[Benchmarks](benchmarks.md#training-throughput) has the scaling sweep over `n_envs`, what each baseline differs in, and the learning comparison.

## Flags

`python -m swarp.rl` takes the same flags as `examples/train_mappo.py`: every `TrainSpec` field (`--iters`, `--n-envs`, `--lr`, `--entropy-coeff`, ...), `--scen-kwarg k=v` for any scenario keyword, and the curriculum, checkpoint and eval controls.
A `--scen-kwarg` on an attribute the spec ramps pins it: the ramp on it holds the flag's value, since it would otherwise overwrite it from the first iteration on, and the ramp clock keeps its length.
Three are its own:

`--no-graphs`
: run the same rollout and update eagerly. On a non-CUDA device this is automatic.

`--tf32`
: TF32 matmuls on Ampere and later. Off by default, so the numerics match the TorchRL trainer.

`--compile`
: `torch.compile` the minibatch loss (needs triton). Adds tens of seconds of startup; the compiled code is still captured by the minibatch graph.

Four more serve fine-tunes from a checkpoint (`--resume`):

`--resume-widen`
: load a checkpoint whose networks are narrower than the new ones (more observation, state or aux columns, a message input); the new weights start at zero, so the policy acts as it did.

`--train-only PREFIX ...`
: train only the actor parameters whose names start with one of the prefixes; the rest of the actor is frozen and the critic trains.

`--ramp-gate`
: override the solve rate the gated ramp clock needs to advance.

`--start-clock`
: with a ramp gate, start the ramps' clock at this tick, e.g. where the checkpoint's run ended.

The network options (`--actor`, `--role-params`, `--critic-input`, `--per-agent-value`, `--aux-coef`, `--privileged-actor`, `--learned-roles`, `--resume-critic-only`) are described in [Roles, privileged critics and auxiliary losses](#roles-privileged-critics-and-auxiliary-losses), and the actor's memory and communication (`--recurrent`, `--hidden-dim`, `--seq-len`, `--comm`, `--comm-dim`) in [Memory and communication](#memory-and-communication).

## Using it from Python

`swarp.rl.train(swarp.rl.parse_args([...]))` is the CLI as a function.
For your own environment or loop, `MAPPO` is the core without logging, checkpoints or curriculum:

```python
import torch, swarp
from swarp.rl import MAPPO, Actor, Critic, SPECS

spec = SPECS["navigation"]
env = swarp.make("navigation", n_envs=spec.n_envs, device="cuda:0", dt=spec.dt,
                 substeps=spec.substeps, max_steps=spec.max_steps, auto_reset=True)
actor = Actor(env.obs_dim, env.act_dim, spec.num_cells).cuda()
critic = Critic(env.obs_dim, env.n_agents, spec.num_cells).cuda()
algo = MAPPO(env, actor, critic, spec)

algo.reset()
for it in range(spec.iters):
    algo.rollout()
    stats = algo.update(spec.entropy_coeff)   # {name: float}, keys in algo.stat_names
    print(it, stats["mean_reward"], stats["n_done"])
```

The environment must be built with `auto_reset=True` and `clone_outputs=False` (the default): the graphs read its output buffers by address.
`Actor` samples in a `[-1, 1]` tanh box and maps it affinely onto the env's per-slot action box, which `swarp.rl.check_action_box(env, name)` returns (it raises if agents' boxes disagree, since the weights are shared); `train` runs it for you and stores the box in the checkpoint.
The map's Jacobian is constant, so it cancels in the PPO ratio and the loss is the unit-box loss.

## Roles, privileged critics and auxiliary losses

Seven `TrainSpec` fields, each with a CLI flag, change the networks.
Their defaults are the plain trainer: with all seven at their default the same kernels run in the same order with the same random draws, and no buffer is added.

| field | flag | values (default first) | needs from the scenario |
|---|---|---|---|
| `actor` | `--actor` | `"mlp"`, `"policynet"` | nothing |
| `role_params` | `--role-params` | `"heads"`, `"shared"`, `"separate"` | `agent_roles()` |
| `critic_input` | `--critic-input` | `"obs"`, `"state"`, `"state+obs"` | `critic_state()` for `state` |
| `per_agent_value` | `--per-agent-value` | `False`, `True` | nothing |
| `aux_coef` | `--aux-coef` | `0.0`, any positive weight | `aux_targets()` |
| `privileged_actor` | `--privileged-actor` | `False`, `True` | `aux_targets()` |
| `learned_roles` | `--learned-roles` | `False`, `True` | `agent_roles()` with `role_counts` `(A, 0)` |

The scenario side is three optional hooks, `critic_state()`, `agent_roles()` and `aux_targets()`, described in [Writing a scenario](writing-a-scenario.md#training-hooks-optional).
`train` refuses an option whose hook the scenario does not provide, rather than training without it.

**Actor.**
`"mlp"` is `Actor`, the obs-only shared-weight network.
`"policynet"` is `PolicyNet`, the same tanh-Gaussian (same squash, same greedy action), which also reads each agent's role when the scenario has roles.
`role_params` sets where the role enters:

- `"shared"`: one trunk and one head over the observation concatenated with a one-hot of the role.
- `"heads"`: the same trunk, and a final layer with one head per role, gathered by role, so a role's head only learns from that role's agents.
- `"separate"`: one network per role. Agents are put in role-sorted order with `role_permutation` (a one-hot cumulative sum, no sort), and each role's network runs on its fixed-size slice, so the cost is one plain GEMM per role rather than a per-sample weight gather.

Roles may move between agents every episode; the role counts are fixed for the run, which is what keeps the slices fixed-size.

**Critic.**
`"obs"` with one value per team is the default `Critic` on a scenario without roles.
Anything else, and the obs critic of a scenario with roles, is a `StateCritic`, which reads the scenario's privileged `critic_state()` (`"state"`), every agent's observation (`"obs"`), or both (`"state+obs"`).
With roles, the observations enter in role-sorted order, so the critic sees the team in a fixed layout however the roles were drawn.
`per_agent_value` gives each role-sorted slot its own value instead of one team value, which matters once the roles are rewarded differently.
The privileged state is not reordered: a scenario with roles writes it in role-sorted agent order (see [Writing a scenario](writing-a-scenario.md#training-hooks-optional)).
The critic only shapes the advantages, so a privileged state helps training and never reaches the deployed policy.

**Auxiliary loss.**
With `aux_coef > 0` the `PolicyNet` carries a linear `aux_head` on its last hidden layer, and the minibatch loss gains `aux_coef * mse(aux_head(features), aux_targets)`.
Its use is to make the policy's features encode something the observation only implies, such as a hidden object pose behind a lidar scan.
The mean auxiliary MSE is logged as a `loss_aux` column, which `metrics.csv` has only when the loss is on.

**Privileged actor.**
`privileged_actor` (with `"policynet"` and `aux_coef > 0`) appends the auxiliary targets to the actor's input, so the policy reads the truth its aux head would estimate.
It is a diagnostic, not a deployable policy: if a fine-tune with it learns what the plain policy cannot, the task fails on perception rather than on control.
The targets enter the trunk as its last columns, after the role one-hot and any message, so `--resume --resume-widen --privileged-actor` starts from a plain checkpoint that acts exactly as it did.
The rollout buffer, the critic and the messages keep the plain observation; the update appends the recorded targets, and the greedy evaluation reads them from the env.

**Learned roles.**
`learned_roles` lets each agent choose its role, on a scenario whose roles past the first are empty (`role_counts` `(A, 0)`, every agent in role 0), with `"policynet"`, `"heads"`, a critic that reads the state and `normalize_advantage=False`.
A zero-initialised role head on the last hidden layer gives role logits, so every role starts equally likely.
Every `role_dwell` steps (`--role-dwell`, 5), and on an episode's first step, the rollout samples the role by Gumbel-argmax and its log-probability joins the action's in the PPO ratio; in between the role is held and adds nothing to the ratio.
The actor reads its previous role as the role one-hot and the teammates' mean previous-role one-hot right after it, and acts with the head of the role it holds now; its messages are encoded with the previous role.
The critic keeps the scenario's roles and reads every agent's previous role and dwell age appended to its state.
The loss gains `role_entropy_coeff` (`--role-entropy-coeff`, 0.01) times the exact categorical entropy averaged over the decision steps, logged as `role_entropy`; `role1_frac` and `switch_rate` (the decisions that changed the role) are logged with the metrics.
`--resume --resume-widen --learned-roles` starts from a checkpoint without learned roles with every role's head and role input tied to role 0's, so the policy acts as it did whichever role it holds.

The rollout records the critic state, the roles and the auxiliary targets next to the observation, inside the same captured act graph, and reads them back by address, as it reads the observation.
That is why the scenario has to write them in the observation pass.

**Advantages.**
With roles and `normalize_advantage`, the advantages are standardized per role-sorted slot rather than per agent index, so each column of statistics holds one role however the roles were drawn.

`--resume-critic-only` with `--resume` loads only the critic from a checkpoint and starts a fresh actor.
`--resume` refuses a checkpoint whose networks differ from the ones the flags build, and names the flags that rebuild them.
A checkpoint of a role-aware policy evaluates through `examples/eval_mappo.py` as any other: the evaluator reads the roles from the env at every step.

```bash
python -m swarp.rl --scenario <name> --actor policynet --role-params heads \
    --critic-input state+obs --per-agent-value --aux-coef 0.1
```

## Memory and communication

A `"policynet"` actor can also remember and talk.
Both are off by default, and with both off the networks, buffers and graphs are the ones above.

| field | flag | values (default first) | needs from the scenario |
|---|---|---|---|
| `recurrent` | `--recurrent` | `False`, `True` | nothing |
| `hidden_dim` | `--hidden-dim` | `128`, any width | nothing |
| `seq_len` | `--seq-len` | `0` (= `steps_per_batch`), a divisor of `steps_per_batch` | nothing |
| `comm` | `--comm` | `"none"`, `"attn"`, `"latent"`, `"vae"`, `"pose"` | `comm_mask()`, optionally; `aux_targets()` for `"vae"` and `"pose"` |
| `comm_dim` | `--comm-dim` | `64`, any width | nothing |
| `msg_coef` | `--msg-coef` | `1.0` | nothing |
| `msg_beta` | `--msg-beta` | `1e-3` | nothing |
| `msg_drop` | `--msg-drop` | `0.1`, in `[0, 1)` | nothing |

**Memory.**
`recurrent` puts a `GRUCell` after the trunk (one per role with `"separate"`), and the action heads and the auxiliary head read its output.
Its hidden state is the policy's memory of the episode: what a lidar-only agent saw a few steps ago, or where an object went when it left the field of view.
The rollout carries the hidden state `[n_envs, n_agents, hidden_dim]` from step to step and from one rollout to the next, and zeroes it in each env whose episode just ended, since with `auto_reset=True` the next observation already belongs to the next episode.
The act graph also stores the hidden state *entering* each step, so the update never has to recompute it.

The update trains it by backpropagation through time.
Each env's `steps_per_batch` steps are cut into chunks of `seq_len` consecutive steps, minibatches are drawn over whole chunks, and each chunk is unrolled from its stored starting hidden state, with the hidden state zeroed after every episode end inside it.
The recomputed log-probs are then the ones the rollout sampled with, which is what makes the PPO ratio start at one.
A minibatch holds as many env steps as without memory; only the GRU steps one at a time, while the trunk, the heads and the critic run over the whole minibatch at once.
A shorter `seq_len` gives more, shorter chunks per minibatch: cheaper gradients that cannot reach as far back.
`python -m swarp.benchmark.train_throughput --configs policynet recurrent comm recurrent-comm` measures what each option costs per iteration.

**Communication.**
`comm="attn"` adds one round of learned communication before the trunk.
Each agent's input (its observation and role one-hot) is encoded to a `comm_dim`-wide vector and projected to a query, a key and a value.
Each agent then attends over its teammates' keys with single-head scaled dot-product attention, and the trunk reads its encoding concatenated with the weighted sum of their values, the message.
An agent never attends to itself.
A scenario that limits who hears whom (a radio range, line of sight) provides `comm_mask()`, a `[n_envs, n_agents, n_agents]` bool tensor where entry `[e, i, j]` lets agent `i` hear agent `j`; without one every agent hears every teammate.
An agent that hears nobody gets a zero message.
The message is computed from the current step's inputs, so communication works with and without memory, and with memory the GRU is what lets a message's content persist.
Communication needs one shared trunk, so it combines with `role_params` `"heads"` or `"shared"`; with roles and `"separate"`, `train` refuses it.

**Delayed messages.**
`"attn"` reads every teammate's *current* input, which a deployment can only reproduce by exchanging observations and waiting for them within the step.
`"latent"`, `"vae"` and `"pose"` are messages a deployment can send over a network at the policy rate instead.
Each agent encodes its own observation and role one-hot into a message with a one-hidden-layer MLP (never its recurrent state), broadcasts it, and its teammates read it on their *next* step.
The listener's trunk reads its own observation and role, then the mean of the messages it received and a flag for whether it received any; those columns come last, so a policy trained without messages widens into one with zero weights on them (`--resume-widen`).
The rollout stores the previous step's observation next to each step, and draws which packets arrive: a message is never heard across an episode start, and each sender-receiver pair loses a packet with probability `msg_drop` per step, so the policy learns to act without one.
The three differ in what shapes the message:

- `"latent"`: a `tanh` code of width `comm_dim`, trained only through the listeners' losses (the PPO objective and their aux loss), so it carries whatever helps them.
- `"vae"`: a Gaussian code of width `comm_dim`, sampled with noise the rollout stores, so the update recomputes the log-probabilities the rollout acted with. Its own loss (weight `msg_coef`) is a decoder's squared error on the *sender's* aux targets plus `msg_beta` times the KL to the unit Gaussian, which bottlenecks the code and keeps it grounded; the listeners' gradients reach it too. The greedy message is the mean.
- `"pose"`: the sender's own estimate of its aux targets and their log-variances, trained by a Gaussian likelihood (weight `msg_coef`) and detached from the listeners: an explicit, inspectable message whose variance says how much to trust it.

`metrics.csv` logs the message's own loss as `loss_msg`.

The mask is a training hook with the same fixed-address contract as the others ([Writing a scenario](writing-a-scenario.md#training-hooks-optional)); the rollout records it next to the observation and the update reads it back with each sample.
Whether a scenario has one is decided once, when `MAPPO` is built, from whether `comm_mask()` returns `None`.

**Evaluation.**
`swarp.rl.greedy_env_arm(actor)` (and `examples/eval_mappo.py`, which goes through it) gives a recurrent policy a stateful action function: it starts from a zero hidden state and carries it from call to call, which is right for the evaluator's single un-reset episode per env.
A loop whose envs reset mid-run builds a fresh action function per episode.
`PolicyNet.greedy(obs, roles, h)` acts from a given hidden state and drops the new one; `PolicyNet.step` returns it.
For a delayed-message policy the action function also keeps the previous observation and hears every teammate from the second step on, with no packet loss.

```bash
python -m swarp.rl --scenario <name> --actor policynet --recurrent --hidden-dim 128 \
    --seq-len 16 --comm attn --comm-dim 64
```

## What differs from the TorchRL trainer

Same algorithm and the same recipes: clipped surrogate with ε = 0.2, smooth-L1 value loss, a one-sample Monte Carlo entropy bonus, per-agent advantage normalization, gradient clipping at 1.0 over actor and critic together, one Adam, GAE with the recipe's γ and λ, the curriculum gate, and the same `metrics.csv` columns.
Three things differ on purpose:

- **Timeouts bootstrap from the current state.** With `auto_reset=True` the observation after a truncation already belongs to the next episode, and there is no terminal observation. A truncated, non-terminated step therefore gets `r + γ·V(s_t)` and cuts the trace (the rsl_rl and Isaac Lab convention), where the TorchRL trainer uses `V(s_{t+1})` of the true terminal state.
- **Curriculum knobs are applied before the first reset**, so the first episodes already run at the starting difficulty.
- **The greedy action is `tanh(loc)`**, the mode of the squashed Gaussian. The TorchRL trainer's evaluator clamps the pre-tanh mean to the action box instead.

A minibatch with a non-finite gradient is skipped on the device, through Adam's `found_inf` (the `GradScaler` mechanism): weights, Adam moments and step count stay exactly as they were, which is the skip the TorchRL trainer does on the host.
The checkpoint format differs (`"format": "swarp.rl/2"`), so the two trainers' checkpoints cannot resume each other.
A `swarp.rl/2` checkpoint keeps every `swarp.rl/1` key and adds the network classes and their constructor arguments (`actor_cls`, `actor_kwargs`, `critic_cls`, `critic_kwargs`, resolved through `swarp.rl.ARCHS`), so it still loads with `torch.load(weights_only=True)`; `swarp.rl/1` checkpoints load as before.

## Not done yet

The environment step is still launched from the host once per step: a whole rollout is `steps_per_batch` env-graph launches plus two small torch-graph replays each.
Embedding the Warp step graph into one torch graph for the whole rollout is possible (as a child-graph node added to torch's live capture), but at the recipes' sizes the update dominates and it would save about 1–2% of an iteration.
It would matter for a small batch with few epochs, where the rollout dominates.
