# Architecture

Read this before extending the step.

## Layering

Torch-facing at the top, Warp kernels at the bottom:

```text
Environment          swarp/core/environment.py  VMAS-style API: reset/step/radius_graph over a Scenario
    ↓ drives
Scenario (ABC)       swarp/scenarios/base.py    make_world / reset_world / observation / *_reward
    ↓ builds
World                swarp/core/world.py        batched state tensors, goals, obstacles, RNG
    ↓ owns
Stepper              swarp/core/stepper.py      THE substep pipeline: neighbors → forces → bodies → integrate
    ↓ launches
kernels              swarp/dynamics/kernels.py, swarp/core/collisions.py, swarp/core/neighbors.py,
                     swarp/core/bodies.py, swarp/sensors/lidar_kernels.py, swarp/scenarios/*_kernels.py
```

Off to the side of that spine, all optional and none on the hot path unless asked for: `swarp/render/` (the pygame viewer), `swarp/sensors/lidar.py`, and `swarp/interop/{compile,persistent,torchrl}.py`.

## Package layout

```text
swarp/core       state (SoA), stepper (THE substep pipeline), neighbors, collisions,
                 movable rigid bodies, world, environment, config
swarp/dynamics   model tags/configs, unified integrate kernel (2D vehicles + drone)
swarp/interop    torch.autograd bridge + BPTT rollout, torch.compile custom op,
                 CUDA-graph capture, TorchRL EnvBase wrapper
swarp/scenarios  Scenario ABC + 7 scenarios, each with its fused *_kernels.py
swarp/sensors    opt-in differentiable observation sensors (lidar: torch + Warp backends)
swarp/render     optional pygame viewer: renderer, overlays, camera, HUD, input handling,
                 mosaic layout, video export, notebook embed, demo entry point
swarp/benchmark  throughput, optimization ablation, per-scenario sweep, and two
                 cross-simulator comparisons (+ _adapters/ for swarp/vmas/jaxmarl/camar)
examples/        standalone scripts: action optimization, Push-T eval, Push-T + TorchRL
docs/            this site
tests/           pytest suite
```

Key modules:

`swarp/core/state.py`
: `WorldState`, the structure-of-arrays Warp storage. One unified state serves every model (`pos`/`theta`/`vel`/`speed`/`ang_vel`, plus the drone's `z`/`vz`/`attitude`/`body_rates`); holonomic agents ignore `theta`/`ang_vel`.

`swarp/core/bodies.py`
: movable and compound rigid bodies, integrated inside the substep loop from the reaction of the same agent contacts `collisions.py` applies. Gather-based, one thread per `(env, obstacle)`, no atomics. Not taped.

`swarp/dynamics/base.py`
: the `DynamicsModel`/`ControlMode`/`Integrator` enums, `AgentConfig` (per-agent, mixable in one world), and `build_agent_params`.

`swarp/interop/autograd.py`
: the torch↔Warp bridge — `_WarpStepFn` (a `torch.autograd.Function` replaying Warp adjoints in `backward`), `warp_step`, and `rollout`.

`swarp/scenarios/fused.py`
: `FusedScenario`, which all seven built-ins subclass. A scenario declares its persistent buffers and its launch sequence; the framework owns lazy allocation, uint8→bool reinterpret views, Warp handle caching, pointer-move resync, the recapture token, the warm-up carry list and the reset-mask stamp. There are **zero opt-outs** — a test asserts all seven implement the same four members and override none of the framework's.

## Design invariants

**The step is functional** (`state_in → state_out`).
Every array written during a taped step — intermediate states, force buffers, neighbor lists — must be allocated **fresh per step**: overwriting an array recorded on a `wp.Tape` silently corrupts its adjoint.
`Stepper` is the single place that knows this; the no-grad hot path recycles one cached `StepBuffers` per batch size.

**Precision is explicit.**
Kernels are generic over dtype and instantiated for float32 and float64 via `wp.overload` (see the `_signature(dtype)` helpers in `kernels.py` and `collisions.py`).
float64-on-CPU is what makes strict `torch.autograd.gradcheck` possible.

**Collision forces are gather-based.**
Each agent sums over its own neighbor list. No atomics: deterministic and race-free.

**Neighbor construction is not taped.**
The neighbor *set* is discrete, so gradients flow through contact geometry, not through membership.

**`wp.HashGrid` wraps cell coordinates modulo its dims**, which aliases cells across the z-lifted env batching — brute force wins up to a few hundred agents per env, and `uniform_grid` takes over above that.

## Non-goals

swarp deliberately does not aim to be a drop-in physics clone of VMAS. Currently out of scope:

**Trajectory parity with VMAS.**
swarp is API-compatible in spirit, not trajectory-compatible: the dynamics, collision constants and observation model differ by design, so identical actions do not reproduce VMAS trajectories.

**Movable bodies on the adjoint tape.**
The bodies themselves are supported — `ObstacleKind.MOVABLE` bodies are integrated inside the substep loop, with rotation and with compound multi-shape bodies. What is missing is differentiability *through* them; a scenario that needs it keeps a torch-side copy and integrates it itself.

**Joints** (VMAS `balance`) **and box-box obstacle-obstacle contacts.**
Obstacles collide with each other only when at least one of the pair is round; two boxes pass through one another, since that needs a polygon contact manifold rather than an SDF evaluated against a disc. Agent-vs-box is exact for every shape.

**Discrete or communication action spaces.**
Actions are continuous real vectors.

## Roadmap

Next: putting movable bodies on the adjoint tape, and **joints** (VMAS `balance` parity).

Everything already landed is in the [changelog](changelog.md).

## Contributing

- Formatter and linter: ruff, line length 100, `target-version = py312`. Lint set `E,W,F,I,UP,B,SIM`, with `SIM108` ignored (ternaries are not always clearer inside kernels).
- `uv run ruff check .` and `uv run pytest -m "not gpu"` are what CI runs.
- A new scenario is expected to ship both tiers and a parity test against its torch oracle — [Writing a scenario](writing-a-scenario.md) covers the contract.
