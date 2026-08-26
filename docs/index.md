# swarp

**Swarm simulation on NVIDIA Warp** — a GPU-resident, differentiable, vectorized simulator for 2D multi-robot tasks.

swarp keeps every environment in the batch on-device as `[n_envs, n_agents]` Warp arrays and advances all of them with compiled kernels, so the hot loop performs no host↔device transfers.
Conceptually it is [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator), with the step compiled as [NVIDIA Warp](https://github.com/NVIDIA/warp) kernels instead of per-entity PyTorch ops — and with the whole step exposed to PyTorch autograd through Warp's adjoint tape.

```{image} _static/hero.png
:alt: Eight agents navigating to goals with lidar rays and communication links drawn
```

<video src="_static/demo.webm" autoplay loop muted playsinline width="100%"></video>

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Batched and on-device
:link: environment
:link-type: doc

One `step` call advances thousands of environments. Observations, rewards, resets and the radius graph never leave the GPU.
:::

:::{grid-item-card} Differentiable end-to-end
:link: differentiability
:link-type: doc

Dynamics, collisions and walls run under a `wp.Tape`; BPTT through multi-step rollouts is verified with `gradcheck`.
:::

:::{grid-item-card} Four dynamics models
:link: dynamics
:link-type: doc

Holonomic, differential drive, kinematic bicycle and a 6-DOF quadrotor — mixable per agent in one world.
:::

:::{grid-item-card} Contacts and rigid bodies
:link: world
:link-type: doc

Soft agent-agent contacts, circle/box/segment obstacles, and movable compound rigid bodies pushed by agent reaction forces.
:::

:::{grid-item-card} Seven scenarios
:link: scenarios
:link-type: doc

Navigation, flocking, formation, discovery, sampling, transport and Push-T — each with fused Warp obs/reward kernels.
:::

:::{grid-item-card} 24 M env-steps/s
:link: benchmarks
:link-type: doc

Full navigation hot path at 16,000 envs × 16 agents on one RTX 3070 Laptop GPU.
:::

::::

## Get going

```bash
uv pip install -e .
```

```python
import torch, swarp

env = swarp.make("navigation", n_envs=4096, n_agents=8, device="cuda:0")
obs = env.reset()
for _ in range(100):
    actions = torch.rand(4096, 8, 2, device="cuda:0") * 2 - 1
    obs, reward, term, trunc, info = env.step(actions)
```

[Installation](installation.md) covers the extras (viewer, TorchRL, benchmarks); [Quickstart](quickstart.md) walks the loop above line by line.

## Status

Pre-1.0 and unreleased — the API may still change.
Everything that has landed is in the [changelog](changelog.md); the source lives at [github.com/ddebenedittis/swarp](https://github.com/ddebenedittis/swarp).

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
```

```{toctree}
:hidden:
:caption: User guide

environment
dynamics
world
scenarios
differentiability
visualization
performance
```

```{toctree}
:hidden:
:caption: Extending

writing-a-scenario
architecture
```

```{toctree}
:hidden:
:caption: Reference

benchmarks
changelog
```
