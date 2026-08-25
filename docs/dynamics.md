# Agents and dynamics models

An agent is described by one `AgentConfig` (`swarp/dynamics/base.py`): its motion model, how its action vector is interpreted, its collision footprint, and its actuation limits.
A world takes a *list* of them, so a fleet can be heterogeneous — the models are branches of one unified integrate kernel, not separate code paths.

```python
from swarp import AgentConfig, ControlMode, DynamicsModel

configs = [
    AgentConfig(model=DynamicsModel.HOLONOMIC, radius=0.05, max_speed=1.0),
    AgentConfig(model=DynamicsModel.DIFF_DRIVE, ctrl_mode=ControlMode.ACCELERATION),
    AgentConfig(model=DynamicsModel.KINEMATIC_BICYCLE, l_f=0.12, l_r=0.12),
]
```

```{image} _static/mixed.png
:alt: Six agents of three different dynamics models, coloured by model
```

## The four models

| `DynamicsModel` | state it moves | action slots | notes |
|---|---|---|---|
| `HOLONOMIC` | `pos`, `vel` | 2 | a point that can move in any direction; `theta`/`ang_vel` are unused |
| `DIFF_DRIVE` | `pos`, `theta`, `speed`, `ang_vel` | 2 | unicycle: forward speed and yaw rate |
| `KINEMATIC_BICYCLE` | `pos`, `theta`, `speed` | 2 | slip-angle β formulation with front/rear axle offsets |
| `DRONE` | full 6-DOF: `pos`, `z`, `vel`, `vz`, `attitude`, `body_rates` | 4 | `+`-configuration quadrotor, four per-rotor thrusts |

All four are differentiable and integrate with either semi-implicit Euler or classical RK4 (`WorldConfig.integrator`).

### Control modes

`ControlMode` decides how the action vector is read:

| model | `VELOCITY` | `ACCELERATION` |
|---|---|---|
| `HOLONOMIC` | `(vx, vy)` | `(ax, ay)` |
| `DIFF_DRIVE` | `(v, ω)` | `(a, α)` |
| `KINEMATIC_BICYCLE` | always `(acceleration, steering angle)` — the flag is ignored | |
| `DRONE` | always four per-rotor thrusts — the flag is ignored | |

Every slot is clamped inside the kernel against that agent's limits (`max_speed`, `max_accel`, `max_ang_vel`, `max_ang_accel`, `max_steer`, `thrust_max`).

### The quadrotor

The drone is a first-class model, not a bolt-on: its integration is the `TAG_DRONE` branch of the same kernel, and its extra state lives in the same `WorldState`.
Total thrust acts along body +z, roll and pitch torques come from opposing-rotor thrust differences over the arm, yaw torque from the rotor reaction sum; attitude follows quaternion kinematics and body rates follow Euler's rigid-body equation with diagonal inertia.

A convenience factory fills the rotor parameters:

```python
from swarp.dynamics.drone import drone_config

cfg = drone_config(mass=1.0, thrust_max=10.0, arm_length=0.15, inertia_zz=0.02)
```

`radius` is the horizontal footprint the shared 2D neighbour and contact machinery uses.

:::{note}
The attitude loop is much stiffer than the 2D models'.
With inertia ~1e-2 kg·m² a 10 Hz control step diverges — substep the physics (the viewer demo integrates the quadrotor at 100 Hz behind a `dt=0.1` control step with `substeps=10`).
:::

## Configuration fields

`AgentConfig` validates in `__post_init__`, so a non-positive mass or a zero wheelbase fails at construction rather than in a kernel.

| field | default | used by |
|---|---|---|
| `model`, `ctrl_mode` | `HOLONOMIC`, `VELOCITY` | all |
| `radius`, `mass` | 0.05, 1.0 | all (contacts, and the drone's rigid body) |
| `max_speed`, `max_accel` | 1.0, 1.0 | all 2D models |
| `max_ang_vel`, `max_ang_accel` | π, 2π | diff-drive |
| `l_f`, `l_r`, `max_steer` | 0.1, 0.1, π/4 | kinematic bicycle |
| `thrust_max`, `arm_length`, `inertia_xx/yy/zz`, `torque_coeff`, `gravity` | see `drone_config` | drone |

## Heterogeneous fleets

Because `act_dim` is the max arity over the models present, a world mixing 2D vehicles and drones takes one rectangular `[n_envs, n_agents, 4]` action tensor and the 2D agents simply ignore slots 2 and 3.
Nothing else in the stack changes: observations come from the same unified state, and contacts use each agent's own `radius`.

## Per-env parameter randomization

By default the parameter matrix is shared across the batch — `[n_agents, NUM_PARAMS]`, the zero-overhead fast path.
Domain randomization is opt-in: hand the stepper a full `[n_envs, n_agents, NUM_PARAMS]` tensor and it switches to the kernel variants that index `params[e, a]`.

```python
import numpy as np
from swarp import P_MASS, P_MAX_SPEED, per_env_float_template

floats = per_env_float_template(world.agent_configs, n_envs)      # [E, A, NUM_PARAMS]
rng = np.random.default_rng(0)
floats[..., P_MASS] *= rng.uniform(0.8, 1.2, size=floats.shape[:2])
floats[..., P_MAX_SPEED] *= rng.uniform(0.9, 1.1, size=floats.shape[:2])

world.stepper.set_agent_params_per_env(floats)
```

Columns follow `AgentConfig.to_row()` order and are addressed by the exported `P_*` index constants (`P_RADIUS`, `P_MASS`, `P_MAX_SPEED`, `P_MAX_ACCEL`, `P_MAX_ANG_VEL`, `P_MAX_ANG_ACCEL`, `P_LF`, `P_LR`, `P_MAX_STEER`, `P_THRUST_MAX`, `P_ARM`, `P_IXX`, `P_IYY`, `P_IZZ`, `P_KAPPA`, `P_GRAVITY`).

Parameters are static per episode, so call `set_agent_params_per_env` again at reset to re-randomize.
Randomizing `P_RADIUS` upward is checked against `neighbor_radius`: if the widened contact reach would exceed it, the call raises rather than silently missing contacts.
Model tags and control modes stay per-agent — they are structural and are never randomized.

The recurrence itself is shared between the shared and per-env kernels, so the two cannot drift; the measured hot-path overhead is about 2% in the mid band and within noise elsewhere.
