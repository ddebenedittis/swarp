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

`python -m swarp.benchmark.train_throughput` runs both trainers through their CLIs and reads the per-iteration wall time each writes to `metrics.csv` (collection plus update; logging, eval and checkpoints excluded).
At the recipes' batch sizes, on an RTX 3070 Laptop GPU:

| scenario | batch | TorchRL | `swarp.rl` (graphs) | `swarp.rl --compile --tf32` |
|---|---|---|---|---|
| `navigation` | 512 envs × 32 steps | 244 ms (67 k env-steps/s) | 97 ms (169 k, 2.5×) | 73 ms (226 k, 3.4×) |
| `giveway` | 1024 envs × 32 steps | 443 ms (74 k env-steps/s) | 188 ms (174 k, 2.4×) | 145 ms (226 k, 3.1×) |

With the overhead gone the iteration is GPU-bound in the PPO update: 8 epochs over the batch through two 256-wide MLPs.
The physics is 2–3% of an iteration even in the fastest configuration, which is why `--tf32` (faster matmuls) and `--compile` (Inductor fuses the elementwise work between them) are the knobs that still move the number.
The speedup carries through to time-to-solve: on `giveway` both trainers reach full curriculum difficulty at the same iteration, `swarp.rl` in 83 s of wall-clock and TorchRL in 208 s, and both final policies solve 1.000 of 2048 evaluation envs.
[Benchmarks](benchmarks.md#training-throughput) has the scaling sweep over `n_envs` and the learning comparison.

## Flags

`python -m swarp.rl` takes the same flags as `examples/train_mappo.py`: every `TrainSpec` field (`--iters`, `--n-envs`, `--lr`, `--entropy-coeff`, ...), `--scen-kwarg k=v` for any scenario keyword, and the curriculum, checkpoint and eval controls.
Three are its own:

`--no-graphs`
: run the same rollout and update eagerly. On a non-CUDA device this is automatic.

`--tf32`
: TF32 matmuls on Ampere and later. Off by default, so the numerics match the TorchRL trainer.

`--compile`
: `torch.compile` the minibatch loss (needs triton). Adds tens of seconds of startup; the compiled code is still captured by the minibatch graph.

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
    mean_reward, n_terminated, n_done, *_ = algo.update(spec.entropy_coeff)
```

The environment must be built with `auto_reset=True` and `clone_outputs=False` (the default): the graphs read its output buffers by address.

## What differs from the TorchRL trainer

Same algorithm and the same recipes: clipped surrogate with ε = 0.2, smooth-L1 value loss, a one-sample Monte Carlo entropy bonus, per-agent advantage normalization, gradient clipping at 1.0 over actor and critic together, one Adam, GAE with the recipe's γ and λ, the curriculum gate, and the same `metrics.csv` columns.
Three things differ on purpose:

- **Timeouts bootstrap from the current state.** With `auto_reset=True` the observation after a truncation already belongs to the next episode, and there is no terminal observation. A truncated, non-terminated step therefore gets `r + γ·V(s_t)` and cuts the trace (the rsl_rl and Isaac Lab convention), where the TorchRL trainer uses `V(s_{t+1})` of the true terminal state.
- **Curriculum knobs are applied before the first reset**, so the first episodes already run at the starting difficulty.
- **The greedy action is `tanh(loc)`**, the mode of the squashed Gaussian. The TorchRL trainer's evaluator clamps the pre-tanh mean to the action box instead.

A minibatch with a non-finite gradient is skipped on the device, through Adam's `found_inf` (the `GradScaler` mechanism): weights, Adam moments and step count stay exactly as they were, which is the skip the TorchRL trainer does on the host.
The checkpoint format differs (`"format": "swarp.rl/1"`), so the two trainers' checkpoints cannot resume each other.

## Not done yet

The environment step is still launched from the host once per step: a whole rollout is `steps_per_batch` env-graph launches plus two small torch-graph replays each.
Embedding the Warp step graph into one torch graph for the whole rollout is possible (as a child-graph node added to torch's live capture), but at the recipes' sizes the update dominates and it would save about 1–2% of an iteration.
It would matter for a small batch with few epochs, where the rollout dominates.
