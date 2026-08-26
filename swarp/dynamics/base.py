"""Dynamics model tags, control modes, and per-agent configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np
import warp as wp


class DynamicsModel(IntEnum):
    """Tag selecting the motion model inside the unified integrate kernel."""

    HOLONOMIC = 0
    DIFF_DRIVE = 1
    KINEMATIC_BICYCLE = 2
    DRONE = 3  # 6-DOF quadrotor, see swarp.dynamics.drone


#: Number of scalar action slots each model reads from the action vector. All
#: current 2D vehicle models use 2; the action array is padded to the env-level
#: max, and models simply ignore slots beyond their arity (see integrate_kernel).
MODEL_ACTION_DIM = {
    "HOLONOMIC": 2,
    "DIFF_DRIVE": 2,
    "KINEMATIC_BICYCLE": 2,
    "DRONE": 4,  # the 6-DOF drone's four per-rotor thrust commands
}


def _coerce(enum_cls, val, argname: str):
    """``val`` as a member of ``enum_cls``, accepting the member, its name or its value.

    Every other string-ish option in the API takes a string — ``neighbor_method``,
    ``bounds_mode``, ``use_graph``, the scenario name — so ``model="drone"`` has to work
    too. Names are matched case-insensitively with ``-`` and ``_`` interchangeable, which
    is what lets a CLI ``--model diff-drive`` reach the enum unaltered.
    """
    if isinstance(val, enum_cls):
        return val
    if isinstance(val, str):
        key = val.strip().upper().replace("-", "_")
        try:
            return enum_cls[key]
        except KeyError:
            valid = ", ".join(m.name.lower() for m in enum_cls)
            raise ValueError(
                f"unknown {argname} {val!r}; valid names are {valid} "
                f"(or a {enum_cls.__name__} member)"
            ) from None
    return enum_cls(val)  # an int, or anything else the enum accepts by value


def action_dim(model: DynamicsModel | str) -> int:
    """Action arity for a dynamics model. See :func:`action_bounds` for the limits."""
    return MODEL_ACTION_DIM[_coerce(DynamicsModel, model, "model").name]


class ControlMode(IntEnum):
    """Interpretation of the 2D action vector.

    HOLONOMIC: VELOCITY -> (vx, vy), ACCELERATION -> (ax, ay).
    DIFF_DRIVE: VELOCITY -> (v, omega), ACCELERATION -> (a, alpha).
    KINEMATIC_BICYCLE: always (acceleration, steering angle); the flag is ignored.
    DRONE: always four per-rotor thrusts; the flag is ignored.
    """

    VELOCITY = 0
    ACCELERATION = 1


class Integrator(Enum):
    """Time integrator for the dynamics: semi-implicit Euler, or classical RK4."""

    EULER = "euler"
    RK4 = "rk4"


#: Layout of the float parameter matrix ``[n_agents, NUM_PARAMS]``, in column order:
#: each entry is the :class:`AgentConfig` field packed into that column.
#:
#: This tuple is the **single source of truth** for the layout. The ``P_*`` column
#: indices below, :data:`NUM_PARAMS` and :meth:`AgentConfig.to_row` are all derived
#: from it, so inserting or moving a parameter cannot leave the kernels reading
#: ``mass`` out of the column ``max_speed`` was packed into — the failure mode of
#: three hand-maintained positional lists, which is silent (right shape, wrong
#: physics). Same principle as :class:`~swarp.core.config.ObstacleShape` and the
#: ``TAG_*`` constants in :mod:`swarp.dynamics.kernels`: derive, never duplicate.
PARAM_FIELDS: tuple[str, ...] = (
    "radius",
    "mass",
    "max_speed",
    "max_accel",
    "max_ang_vel",
    "max_ang_accel",
    "l_f",
    "l_r",
    "max_steer",
    # 6-DOF drone parameters (ignored by the 2D vehicle models).
    "thrust_max",
    "arm_length",
    "inertia_xx",
    "inertia_yy",
    "inertia_zz",
    "torque_coeff",
    "gravity",
)
NUM_PARAMS = len(PARAM_FIELDS)
_COL = {name: i for i, name in enumerate(PARAM_FIELDS)}

# Column indices into the float parameter matrix, spelled out (rather than injected
# into globals()) so the kernels' `from ... import P_MASS` stays statically resolvable.
# A mistyped field name is an import-time KeyError, not a wrong column.
P_RADIUS = _COL["radius"]
P_MASS = _COL["mass"]
P_MAX_SPEED = _COL["max_speed"]
P_MAX_ACCEL = _COL["max_accel"]
P_MAX_ANG_VEL = _COL["max_ang_vel"]
P_MAX_ANG_ACCEL = _COL["max_ang_accel"]
P_LF = _COL["l_f"]
P_LR = _COL["l_r"]
P_MAX_STEER = _COL["max_steer"]
P_THRUST_MAX = _COL["thrust_max"]  # per-rotor max thrust (N)
P_ARM = _COL["arm_length"]  # rotor arm length from CoG (m)
P_IXX = _COL["inertia_xx"]  # body-frame diagonal inertia
P_IYY = _COL["inertia_yy"]
P_IZZ = _COL["inertia_zz"]
P_KAPPA = _COL["torque_coeff"]  # yaw reaction torque per unit rotor thrust
P_GRAVITY = _COL["gravity"]  # gravitational acceleration (m/s^2)


@dataclass
class AgentConfig:
    """Static per-agent definition: model, shape, and actuation limits."""

    #: Accepts the enum member, its name as a string (``"drone"``, ``"diff-drive"``), or
    #: its int value; ``__post_init__`` normalizes it to the member. Same for
    #: ``ctrl_mode``. This is the one boundary every scenario's fleet passes through, so
    #: coercing here is what makes ``swarp.make("navigation", model="drone")`` work.
    model: DynamicsModel | str | int = DynamicsModel.HOLONOMIC
    ctrl_mode: ControlMode | str | int = ControlMode.VELOCITY
    radius: float = 0.05
    mass: float = 1.0
    max_speed: float = 1.0
    max_accel: float = 1.0
    max_ang_vel: float = math.pi
    max_ang_accel: float = 2.0 * math.pi
    l_f: float = 0.1  # front axle to center of gravity (bicycle only)
    l_r: float = 0.1  # rear axle to center of gravity (bicycle only)
    max_steer: float = math.pi / 4  # bicycle only
    # 6-DOF drone parameters (used only when model == DRONE).
    thrust_max: float = 10.0  # per-rotor max thrust (N)
    arm_length: float = 0.15  # rotor arm length from CoG (m)
    inertia_xx: float = 0.01
    inertia_yy: float = 0.01
    inertia_zz: float = 0.02
    torque_coeff: float = 0.02  # yaw reaction torque per unit rotor thrust
    gravity: float = 9.81

    def __post_init__(self) -> None:
        self.model = _coerce(DynamicsModel, self.model, "model")
        self.ctrl_mode = _coerce(ControlMode, self.ctrl_mode, "ctrl_mode")
        if self.l_f + self.l_r <= 0.0:
            raise ValueError("Bicycle wheelbase l_f + l_r must be positive.")
        for name in ("radius", "mass", "max_speed", "max_accel", "max_ang_vel", "max_ang_accel"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"AgentConfig.{name} must be positive.")
        if self.model == DynamicsModel.DRONE:
            for name in ("thrust_max", "arm_length", "inertia_xx", "inertia_yy", "inertia_zz"):
                if getattr(self, name) <= 0.0:
                    raise ValueError(f"AgentConfig.{name} must be positive for a drone.")

    def to_row(self) -> list[float]:
        """This config as one ``[NUM_PARAMS]`` kernel row, in :data:`PARAM_FIELDS` order."""
        return [float(getattr(self, name)) for name in PARAM_FIELDS]


#: Which ``AgentConfig`` limit each model's integrate branch clamps each action slot
#: against, as ``(low, high)`` field names per slot. ``None`` low means "the negation of
#: the high field"; a literal float is used as-is. This is the table
#: :func:`~swarp.dynamics.kernels._step_2d` and
#: :func:`~swarp.dynamics.kernels._integrate_agent` implement — keep the two in step.
_ACTION_LIMITS: dict[tuple[str, str], tuple[tuple[float | None, str], ...]] = {
    # holonomic clamps the action *vector*, so both slots share one limit
    ("HOLONOMIC", "VELOCITY"): ((None, "max_speed"), (None, "max_speed")),
    ("HOLONOMIC", "ACCELERATION"): ((None, "max_accel"), (None, "max_accel")),
    ("DIFF_DRIVE", "VELOCITY"): ((None, "max_speed"), (None, "max_ang_vel")),
    ("DIFF_DRIVE", "ACCELERATION"): ((None, "max_accel"), (None, "max_ang_accel")),
    # the bicycle and drone ignore ctrl_mode, so both modes map to one row
    ("KINEMATIC_BICYCLE", "VELOCITY"): ((None, "max_accel"), (None, "max_steer")),
    ("KINEMATIC_BICYCLE", "ACCELERATION"): ((None, "max_accel"), (None, "max_steer")),
    # per-rotor thrust is one-sided: a rotor cannot pull
    ("DRONE", "VELOCITY"): ((0.0, "thrust_max"),) * 4,
    ("DRONE", "ACCELERATION"): ((0.0, "thrust_max"),) * 4,
}


def action_bounds(cfg: AgentConfig) -> tuple[list[float], list[float]]:
    """The per-slot action box the integrate kernels clamp ``cfg``'s action to.

    Returns ``(low, high)``, each ``action_dim(cfg.model)`` long. These are
    **physical** limits, not a normalized range: a holonomic velocity-mode agent with
    ``max_speed=3.0`` is clamped to ``[-3, 3]``, and a drone's four rotor commands to
    ``[0, thrust_max]`` — actions are never rescaled on the way in.

    Two caveats on the box:

    - For the holonomic model the kernel clamps the action's *norm*
      (``clamp_norm(a, max_speed)``), so the true feasible set is the disc inscribed in
      this box. The box edges are still exact along each axis, which is what a
      per-slot bound can express.
    - Slots past a model's arity are ignored by its branch; the padded-out width is a
      caller concern (see :meth:`swarp.core.environment.Environment.action_bounds`).
    """
    key = (DynamicsModel(cfg.model).name, ControlMode(cfg.ctrl_mode).name)
    slots = _ACTION_LIMITS[key]
    high = [getattr(cfg, field) for _, field in slots]
    low = [-h if lo is None else lo for (lo, _), h in zip(slots, high, strict=True)]
    return low, high


@dataclass
class AgentParams:
    """Device-resident per-agent parameters.

    ``floats`` is the shared ``[n_agents, NUM_PARAMS]`` layout used by all envs
    (the default fast path). ``floats_per_env`` is an opt-in
    ``[n_envs, n_agents, NUM_PARAMS]`` override for domain randomization; when it
    is not ``None`` the stepper launches the per-env kernel variants that index
    ``params[e, a, ...]`` instead of ``params[a, ...]``. ``model_tag``/
    ``ctrl_mode`` stay per-agent (structural, never randomized).
    """

    floats: wp.array  # [n_agents, NUM_PARAMS], dtype float32/float64
    model_tag: wp.array  # [n_agents], int32
    ctrl_mode: wp.array  # [n_agents], int32
    floats_per_env: wp.array | None = None  # [n_envs, n_agents, NUM_PARAMS] or None


def build_agent_params(
    configs: list[AgentConfig],
    device: str,
    dtype=wp.float32,
) -> AgentParams:
    """Pack a list of AgentConfig into device arrays consumed by the kernels."""
    npdt = np.float64 if dtype == wp.float64 else np.float32
    rows = np.array([c.to_row() for c in configs], dtype=npdt)
    tags = np.array([int(c.model) for c in configs], dtype=np.int32)
    modes = np.array([int(c.ctrl_mode) for c in configs], dtype=np.int32)
    return AgentParams(
        floats=wp.array(rows, dtype=dtype, device=device),
        model_tag=wp.array(tags, dtype=wp.int32, device=device),
        ctrl_mode=wp.array(modes, dtype=wp.int32, device=device),
    )


def per_env_float_template(configs: list[AgentConfig], n_envs: int) -> np.ndarray:
    """A ``[n_envs, n_agents, NUM_PARAMS]`` float64 array pre-filled from ``configs``.

    Every env starts as a copy of the shared per-agent rows; edit columns (e.g.
    ``[..., P_MASS]``) to randomize, then hand the result to
    :meth:`swarp.core.stepper.Stepper.set_agent_params_per_env`.
    """
    rows = np.array([c.to_row() for c in configs], dtype=np.float64)
    return np.broadcast_to(rows, (n_envs, *rows.shape)).copy()
