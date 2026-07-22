"""wmas — Warp Multi-Agent Simulator.

GPU-resident, differentiable, vectorized 2D multi-agent simulation on NVIDIA
Warp with zero-copy PyTorch interop.
"""

from wmas.core.config import ObstacleShape, WorldConfig
from wmas.core.environment import Environment
from wmas.core.stepper import Stepper
from wmas.core.world import World
from wmas.dynamics.base import (
    AgentConfig,
    ControlMode,
    DynamicsModel,
    Integrator,
    per_env_float_template,
)
from wmas.interop.autograd import TorchState, rollout, warp_step
from wmas.scenarios.base import Scenario
from wmas.scenarios.navigation import NavigationScenario

__version__ = "0.1.0"

__all__ = [
    "AgentConfig",
    "ControlMode",
    "DynamicsModel",
    "Environment",
    "Integrator",
    "NavigationScenario",
    "ObstacleShape",
    "Scenario",
    "Stepper",
    "TorchState",
    "World",
    "WorldConfig",
    "per_env_float_template",
    "rollout",
    "warp_step",
    "__version__",
]
