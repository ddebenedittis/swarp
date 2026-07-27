# CLAUDE.md

## Project Overview

`wmas` (Warp Multi-Agent Simulator) is a GPU-resident, differentiable, vectorized 2D
multi-agent simulator built on NVIDIA Warp with zero-copy PyTorch interop. Conceptually
VMAS, but the step is compiled Warp kernels instead of per-entity PyTorch ops. All state
lives on-device as `[n_envs, n_agents]` Warp arrays; the hot loop does no host↔device copies.

`VectorizedMultiAgentSimulator/` is **not** a submodule and not part of this package: it is
an optional, gitignored local VMAS clone that `benchmark/compare_vmas.py` falls back to when
`vmas` is not importable. It is ruff-excluded. Do not edit it.

## Build & Run

```bash
uv venv
uv pip install -e . --group dev
uv pip install -e '.[viz]'                 # optional: viewer/video tests stop skipping
uv run pytest                              # dynamics, gradients, neighbors, collisions, determinism
uv run pytest -m "not gpu"                 # what CI runs (no CUDA device on the runners)
python -m wmas.benchmark.throughput        # NavigationScenario hot-path env-steps/s
python -m wmas.benchmark.compare_vmas      # vs VMAS; needs `--group bench` (pulls in vmas, numpy<2)
```

Markers declared in `pyproject.toml`: `gpu` (needs CUDA), `slow`, `viz` (needs the `viz`
extra). Benchmark numbers and the multi-venv cross-simulator setup live in
`docs/benchmarks.md` — keep them out of the README.

`compare_vmas` takes `--metric {throughput,memory,both}`; falls back to the
`./VectorizedMultiAgentSimulator` checkout if `vmas` is not importable.

## Architecture

Layering (torch-facing at the top, Warp kernels at the bottom):

```
Environment          wmas/core/environment.py   VMAS-style API: reset/step/radius_graph over a Scenario
    ↓ drives
Scenario (ABC)       wmas/scenarios/base.py     make_world / reset_world / observation / *_reward
    ↓ builds
World                wmas/core/world.py         batched state tensors, goals, obstacles, RNG
    ↓ owns
Stepper              wmas/core/stepper.py       THE substep pipeline: neighbors → forces → bodies → integrate
    ↓ launches
kernels              wmas/dynamics/kernels.py, wmas/core/collisions.py, wmas/core/neighbors.py,
                     wmas/core/bodies.py, wmas/sensors/lidar_kernels.py, wmas/scenarios/*_kernels.py
```

Off to the side of that spine (all optional, none on the hot path unless asked for):
`wmas/render/` (~2.1k LOC pygame viewer: renderer/overlays/camera/hud/input/layout/video/
notebook/demo), `wmas/sensors/lidar.py` (opt-in differentiable ray-cast, torch + Warp
backends), and `wmas/interop/{compile,persistent,torchrl}.py`
(`torch.library.custom_op` for `torch.compile`, CUDA-graph capture of the no-grad step,
TorchRL `EnvBase` wrapper).

- `wmas/core/state.py` — `WorldState`: structure-of-arrays Warp storage. One unified state
  for all models (`pos/theta/vel/speed/ang_vel`); holonomic agents ignore `theta/ang_vel`.
- `wmas/core/bodies.py` — movable/compound rigid bodies. An `ObstacleKind.MOVABLE` obstacle
  is integrated **inside the substep loop** from the reaction of the same agent contacts
  `collisions.py` applies (gather-based, thread per `(env, obstacle)`, no atomics). Shapes
  sharing a body id form one compound rigid pose (Push-T's T). **Not taped**
  (`record_tape=False`), so a scenario needing gradients through a body integrates it in
  torch itself. Box-box obstacle-obstacle contacts are not modelled.
- `wmas/dynamics/base.py` — `DynamicsModel`/`ControlMode`/`Integrator` enums, `AgentConfig`
  (per-agent, mixable in one world), and `build_agent_params`. Four models: holonomic point,
  diff-drive, kinematic bicycle (all 2D), and a 6-DOF `+`-config quadrotor whose `AgentConfig`
  factory lives in `wmas/dynamics/drone.py`.
- `wmas/interop/autograd.py` — the torch↔Warp bridge: `_WarpStepFn` (a `torch.autograd.Function`
  replaying Warp adjoints in `backward`), `warp_step`, and `rollout` for BPTT over multi-step rollouts.
- `wmas/core/neighbors.py` — `NeighborGrid` with two backends (per-env brute force and one
  `wp.HashGrid` over all envs), selected via `WorldConfig.neighbor_method`; tested to agree exactly.
- `wmas/scenarios/` — 7 scenarios: navigation, flocking, formation, discovery, sampling,
  transport, pusht. Each pairs a torch implementation of obs/reward/done with fused Warp
  `*_kernels.py` for the no-grad hot path. The torch path is the **parity oracle** the fused
  kernels are tested against — keep them independent rather than sharing code.

## Design Invariants (read before extending the step)

- **The step is functional** (`state_in → state_out`). Every array written during a taped step
  (intermediate states, force buffers, neighbor lists) must be allocated **fresh per step** —
  overwriting an array recorded on a `wp.Tape` silently corrupts its adjoint. `Stepper` is the
  single place that knows this; the no-grad hot path recycles one cached `StepBuffers` per batch size.
- **Precision is explicit.** Kernels are generic over dtype and instantiated for float32/float64
  via `wp.overload` (see the `_signature(dtype)` helpers in `kernels.py`/`collisions.py`).
  float64-on-CPU is what makes strict `torch.autograd.gradcheck` possible.
- **Collision forces are gather-based** — each agent sums over its own neighbor list. No atomics:
  deterministic and race-free.
- **Neighbor construction is not taped** (`record_tape=False`); the neighbor *set* is discrete, so
  gradients flow through contact geometry, not through membership.
- `wp.HashGrid` wraps cell coordinates modulo its dims, which aliases cells across the z-lifted
  env batching — brute force wins up to a few hundred agents/env.

## Code Style

- Formatter/linter: ruff (config in `pyproject.toml`), line length 100, `target-version = py312`.
- Lint set: `E,W,F,I,UP,B,SIM`; `SIM108` ignored (ternaries aren't always clearer inside kernels).
- `VectorizedMultiAgentSimulator/` is `extend-exclude`d.
