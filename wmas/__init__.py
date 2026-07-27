"""wmas — Warp Multi-Agent Simulator.

GPU-resident, differentiable, vectorized 2D multi-agent simulation on NVIDIA
Warp with zero-copy PyTorch interop.

The quickest way in is :func:`make`::

    import wmas
    env = wmas.make("navigation", n_envs=4096, n_agents=8, device="cuda:0")
    obs = env.reset()

The names it accepts are the keys of :data:`wmas.scenarios.SCENARIOS`, the one
scenario registry.
"""

from __future__ import annotations

import inspect

from wmas.core.config import ObstacleKind, ObstacleShape, WorldConfig
from wmas.core.environment import Environment
from wmas.core.state import WorldState
from wmas.core.stepper import Stepper
from wmas.core.world import World
from wmas.dynamics.base import (
    NUM_PARAMS,
    P_ARM,
    P_GRAVITY,
    P_IXX,
    P_IYY,
    P_IZZ,
    P_KAPPA,
    P_LF,
    P_LR,
    P_MASS,
    P_MAX_ACCEL,
    P_MAX_ANG_ACCEL,
    P_MAX_ANG_VEL,
    P_MAX_SPEED,
    P_MAX_STEER,
    P_RADIUS,
    P_THRUST_MAX,
    AgentConfig,
    ControlMode,
    DynamicsModel,
    Integrator,
    per_env_float_template,
)
from wmas.interop.autograd import TorchState, rollout, warp_step
from wmas.scenarios import (
    SCENARIOS,
    DiscoveryScenario,
    FlockingScenario,
    FormationScenario,
    NavigationScenario,
    PushTScenario,
    SamplingScenario,
    Scenario,
    TransportScenario,
    fused_scenarios,
    make_scenario,
    scenario_class,
)
from wmas.sensors.lidar import Lidar, lidar_scan

__version__ = "0.1.0"

# Keywords :func:`make` forwards to Environment; everything else goes to the
# scenario constructor. Derived from the signature so the two never drift.
_ENV_KWARGS = frozenset(inspect.signature(Environment.__init__).parameters) - {
    "self",
    "scenario",
    "n_envs",
}


def make(name: str, n_envs: int, **kwargs) -> Environment:
    """Build an :class:`~wmas.core.environment.Environment` for a registered scenario.

    ``name`` is a key of :data:`wmas.scenarios.SCENARIOS` (``ValueError`` listing
    the valid names otherwise); ``n_envs`` is required rather than defaulted,
    because quietly simulating a single env in a vectorized simulator is a trap.

    Remaining keywords are routed by name: those the ``Environment`` constructor
    accepts (``device``, ``dt``, ``substeps``, ``dtype``, ``max_steps``, ``seed``,
    ``auto_reset``, ``use_graph``, ``copy_outputs``, ``fused``) go to it, and every
    other keyword goes to the scenario constructor (``n_agents``, ``world_size``,
    ``model``, ...). The two parameter sets are disjoint — a test pins that — so
    the split is unambiguous; construct the scenario yourself to bypass it.

    ``model=`` is dropped for holonomic-only scenarios (see
    :func:`wmas.scenarios.make_scenario`).
    """
    env_kwargs = {k: v for k, v in kwargs.items() if k in _ENV_KWARGS}
    scen_kwargs = {k: v for k, v in kwargs.items() if k not in _ENV_KWARGS}
    scenario = make_scenario(name, **scen_kwargs)
    return Environment(scenario, n_envs=n_envs, **env_kwargs)


__all__ = [
    "AgentConfig",
    "ControlMode",
    "DiscoveryScenario",
    "DynamicsModel",
    "Environment",
    "FlockingScenario",
    "FormationScenario",
    "Integrator",
    "Lidar",
    "NUM_PARAMS",
    "NavigationScenario",
    "ObstacleKind",
    "ObstacleShape",
    "P_ARM",
    "P_GRAVITY",
    "P_IXX",
    "P_IYY",
    "P_IZZ",
    "P_KAPPA",
    "P_LF",
    "P_LR",
    "P_MASS",
    "P_MAX_ACCEL",
    "P_MAX_ANG_ACCEL",
    "P_MAX_ANG_VEL",
    "P_MAX_SPEED",
    "P_MAX_STEER",
    "P_RADIUS",
    "P_THRUST_MAX",
    "PushTScenario",
    "SCENARIOS",
    "SamplingScenario",
    "Scenario",
    "Stepper",
    "TorchState",
    "TransportScenario",
    "World",
    "WorldConfig",
    "WorldState",
    "fused_scenarios",
    "lidar_scan",
    "make",
    "make_scenario",
    "per_env_float_template",
    "rollout",
    "scenario_class",
    "warp_step",
    "__version__",
]
