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
Three are its own:

`--no-graphs`
: run the same rollout and update eagerly. On a non-CUDA device this is automatic.

`--tf32`
: TF32 matmuls on Ampere and later. Off by default, so the numerics match the TorchRL trainer.

`--compile`
: `torch.compile` the minibatch loss (needs triton). Adds tens of seconds of startup; the compiled code is still captured by the minibatch graph.

The network options (`--actor`, `--role-params`, `--critic-input`, `--per-agent-value`, `--aux-coef`, `--resume-critic-only`) are described in [Roles, privileged critics and auxiliary losses](#roles-privileged-critics-and-auxiliary-losses).

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

Five `TrainSpec` fields, each with a CLI flag, change the networks.
Their defaults are the plain trainer: with all five at their default the same kernels run in the same order with the same random draws, and no buffer is added.

| field | flag | values (default first) | needs from the scenario |
|---|---|---|---|
| `actor` | `--actor` | `"mlp"`, `"policynet"` | nothing |
| `role_params` | `--role-params` | `"heads"`, `"shared"`, `"separate"` | `agent_roles()` |
| `critic_input` | `--critic-input` | `"obs"`, `"state"`, `"state+obs"` | `critic_state()` for `state` |
| `per_agent_value` | `--per-agent-value` | `False`, `True` | nothing |
| `aux_coef` | `--aux-coef` | `0.0`, any positive weight | `aux_targets()` |

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

The rollout records the critic state, the roles and the auxiliary targets next to the observation, inside the same captured act graph, and reads them back by address, as it reads the observation.
That is why the scenario has to write them in the observation pass.

`--resume-critic-only` with `--resume` loads only the critic from a checkpoint and starts a fresh actor.
A checkpoint of a role-aware policy evaluates through `examples/eval_mappo.py` as any other: the evaluator reads the roles from the env at every step.

```bash
python -m swarp.rl --scenario <name> --actor policynet --role-params heads \
    --critic-input state+obs --per-agent-value --aux-coef 0.1
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
