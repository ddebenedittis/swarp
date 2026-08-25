# Differentiability

Everything between actions and rewards is differentiable.
The dynamics, the soft collisions and the walls run under Warp's adjoint tape; observations and rewards are plain torch ops over the world state. The two meet in a `torch.autograd.Function`, so `loss.backward()` reaches all the way to the actions.

## How the bridge works

`swarp/interop/autograd.py` holds three pieces:

`TorchState`
: a `NamedTuple` view of the world state — `pos`, `theta`, `vel`, `speed`, `ang_vel`, plus the four 6-DOF drone fields, which default to `None` and are auto-filled with zeros and an identity attitude.

`warp_step(stepper, state, actions)`
: one differentiable step. The forward wraps the input tensors as Warp arrays **zero-copy**, records every substep launch on a `wp.Tape`, and returns zero-copy views of freshly allocated outputs. The backward seeds the output adjoints and replays the tape in reverse.

`rollout(stepper, state, actions_seq)`
: chains `T` steps for backprop-through-time and returns `(final_state, trajectory)`.

Under `torch.no_grad()` the tape is skipped entirely and intermediate buffers are recycled — same kernels, no steady-state allocations.

## Optimizing through the physics

Gradient-descend a plan directly, with no policy and no learning algorithm in the way ([`examples/optimize_actions.py`](https://github.com/ddebenedittis/swarp/blob/main/examples/optimize_actions.py) is the runnable version):

```python
import torch, warp as wp
from swarp import AgentConfig, ControlMode, DynamicsModel, Stepper, TorchState, WorldConfig, rollout

T, n_agents, device = 30, 4, "cuda:0"
cfgs = [AgentConfig(model=DynamicsModel.DIFF_DRIVE, ctrl_mode=ControlMode.VELOCITY,
                    max_speed=1.0, max_ang_vel=3.0) for _ in range(n_agents)]
stepper = Stepper(cfgs, dt=0.1, device=device, world=WorldConfig(collision_k=50.0))

actions = torch.zeros(T, 1, n_agents, 2, device=device, requires_grad=True)
opt = torch.optim.Adam([actions], lr=0.05)

for _ in range(200):
    opt.zero_grad()
    final, traj = rollout(stepper, state0, actions)     # BPTT through the Warp adjoints
    loss = (final.pos - goals).square().sum() + 1e-3 * actions.square().sum()
    loss.backward()
    opt.step()
```

With crossing start/goal pairs the collision term is what makes this interesting: the agents have to learn to yield to each other, and that signal arrives entirely through the contact gradients.

For an RL-style loop the same machinery is behind `Environment.step` — just call it with grads enabled.

### Reusing the taped scratch

A `T`-step rollout needs `T` distinct sets of taped buffers, since they must not alias while the tapes are live.
`GradRing` pre-allocates them once and hands them out across optimizer iterations:

```python
from swarp.interop.autograd import GradRing

ring = GradRing(stepper, n_envs=1, act_dim=2, capacity=T)
for _ in range(iters):
    final, traj = rollout(stepper, state0, actions, ring=ring)
```

Passing no ring keeps the fresh-allocation path, which is the one `torch.autograd.gradcheck` relies on.

## Correctness

Gradients are checked two ways in the test suite:

- **Strict `torch.autograd.gradcheck`** in float64 on the CPU. Kernels are generic over dtype and explicitly instantiated for float32 and float64 via `wp.overload`, and float64-on-CPU is exactly what makes a strict check possible.
- **Analytic gradient tests** for cases with a closed form, so a systematically wrong adjoint cannot pass by matching a finite-difference approximation of itself.

## What is *not* differentiable

Knowing where the gradient stops matters more than the headline.

**Saturated actions.** Limits are clamped inside the kernel, so an action past its bound gets zero gradient. That is correct — and verified finite, not NaN — but a policy initialized far outside the action range will see no signal until it comes back in.

**Neighbour membership.** The neighbour set is a discrete structure and is built with `record_tape=False`. Gradients flow through contact *geometry* (how deep the overlap is, how fast the closing velocity), not through *which* pairs are in contact.

**Movable rigid bodies.** `swarp/core/bodies.py` advances body state with `record_tape=False`, so no gradient flows through a body's motion. A scenario that needs one keeps its own torch-side copy and integrates it there:

- `TransportScenario` always does, staggered by one step.
- `PushTScenario` does it on the grad path only; the no-grad path uses the engine body.

Both give BPTT body→agent→action across a rollout, but not through the intra-step agent-avoids-body force. Putting movable bodies on the tape is on the [roadmap](architecture.md#roadmap).

## The functional-step invariant

Every array written during a taped step — intermediate states, force buffers, neighbour lists — must be allocated **fresh per step**.
Overwriting an array that is recorded on a `wp.Tape` silently corrupts its adjoint: no error, just wrong gradients.

`Stepper` is the single place that knows this. Grad-mode steps are strictly functional; the no-grad hot path recycles one cached `StepBuffers` per batch size.
If you extend the step, read [Architecture](architecture.md) first.
