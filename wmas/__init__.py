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
from wmas.scenarios.discovery import DiscoveryScenario
from wmas.scenarios.flocking import FlockingScenario
from wmas.scenarios.formation import FormationScenario
from wmas.scenarios.navigation import NavigationScenario
from wmas.scenarios.sampling import SamplingScenario
from wmas.scenarios.transport import TransportScenario
from wmas.sensors.lidar import Lidar, lidar_scan

__version__ = "0.1.0"

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
    "NavigationScenario",
    "ObstacleShape",
    "SamplingScenario",
    "Scenario",
    "Stepper",
    "TorchState",
    "TransportScenario",
    "World",
    "WorldConfig",
    "lidar_scan",
    "per_env_float_template",
    "rollout",
    "warp_step",
    "__version__",
]
