"""swarp — swarm simulation on NVIDIA Warp.

GPU-resident, differentiable, vectorized 2D multi-agent simulation on NVIDIA
Warp with zero-copy PyTorch interop.

The quickest way in is :func:`make`::

    import swarp
    env = swarp.make("navigation", n_envs=4096, n_agents=8, device="cuda:0")
    obs = env.reset()

The names it accepts are the keys of :data:`swarp.scenarios.SCENARIOS`, the one
scenario registry.
"""

from __future__ import annotations

import difflib
import inspect

from swarp.core.config import ObstacleKind, Obstacles, ObstacleShape, WorldConfig
from swarp.core.environment import Environment
from swarp.core.state import WorldState
from swarp.core.stepper import Stepper
from swarp.core.world import World
from swarp.dynamics.base import (
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
    PARAM_FIELDS,
    AgentConfig,
    ControlMode,
    DynamicsModel,
    Integrator,
    action_bounds,
    per_env_float_template,
)
from swarp.dynamics.drone import drone_config
from swarp.interop.autograd import GradRing, TorchState, rollout, warp_step
from swarp.scenarios import (
    SCENARIOS,
    Buf,
    DiscoveryScenario,
    FlockingScenario,
    FormationScenario,
    FusedPass,
    FusedScenario,
    NavigationScenario,
    PushTScenario,
    SamplingScenario,
    Scenario,
    TransportScenario,
    fused_scenarios,
    make_scenario,
    register_scenario,
    scenario_class,
)
from swarp.sensors.lidar import Lidar, lidar_scan

__version__ = "0.1.0"

# Keywords :func:`make` forwards to Environment; everything else goes to the
# scenario constructor. Derived from the signature so the two never drift.
_ENV_KWARGS = frozenset(inspect.signature(Environment.__init__).parameters) - {
    "self",
    "scenario",
    "n_envs",
}


def make(name: str, n_envs: int, **kwargs) -> Environment:
    """Build an :class:`~swarp.core.environment.Environment` for a registered scenario.

    ``name`` is a key of :data:`swarp.scenarios.SCENARIOS` (``ValueError`` listing
    the valid names otherwise); ``n_envs`` is required rather than defaulted,
    because quietly simulating a single env in a vectorized simulator is a trap.

    Remaining keywords are routed by name: those the ``Environment`` constructor
    accepts (``device``, ``dt``, ``substeps``, ``dtype``, ``max_steps``, ``seed``,
    ``auto_reset``, ``use_graph``, ``clone_outputs``, ``fused``, ``world_config``) go to
    it, and every other keyword goes to the scenario constructor (``n_agents``,
    ``world_size``, ``model``, ...). The two parameter sets are disjoint — a test pins
    that — so the split is unambiguous; construct the scenario yourself to bypass it.

    ``model=`` is dropped, with a warning, for holonomic-only scenarios (see
    :func:`swarp.scenarios.make_scenario`).

    Because the routing is by name, a typo in an ``Environment`` keyword silently becomes
    a scenario keyword. So an unknown keyword is checked against the chosen scenario
    *before* anything is constructed, and the ``TypeError`` lists that scenario's
    keywords — plus the ``Environment`` one it looks like a misspelling of.
    """
    env_kwargs = {k: v for k, v in kwargs.items() if k in _ENV_KWARGS}
    scen_kwargs = {k: v for k, v in kwargs.items() if k not in _ENV_KWARGS}
    _check_scenario_kwargs(name, scen_kwargs)
    scenario = make_scenario(name, **scen_kwargs)
    return Environment(scenario, n_envs=n_envs, **env_kwargs)


def _check_scenario_kwargs(name: str, scen_kwargs: dict) -> None:
    """Raise a legible ``TypeError`` for keywords the scenario cannot take.

    Without this the failure is either a bare ``__init__() got an unexpected keyword
    argument`` from deep inside construction, or — for a misspelled ``Environment``
    keyword — nothing at all until the same message arrives from the scenario, naming the
    wrong constructor.
    """
    params = inspect.signature(scenario_class(name).__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return  # takes **kwargs: it decides for itself what is valid
    valid = sorted(set(params) - {"self"})
    for key in scen_kwargs:
        # ``model`` is legitimately accepted-or-dropped-with-a-warning by
        # :func:`swarp.scenarios.make_scenario`; that decision stays there.
        if key in valid or key == "model":
            continue
        msg = (
            f"{key!r} is not a keyword of scenario {name!r} or of Environment. "
            f"Scenario keywords: {', '.join(valid)}."
        )
        close = difflib.get_close_matches(key, _ENV_KWARGS, n=1)
        if close:
            msg += f" Did you mean Environment's {close[0]!r}?"
        raise TypeError(msg)


__all__ = [
    "AgentConfig",
    "Buf",
    "ControlMode",
    "DiscoveryScenario",
    "DynamicsModel",
    "Environment",
    "FlockingScenario",
    "FormationScenario",
    "FusedPass",
    "FusedScenario",
    "GradRing",
    "Integrator",
    "Lidar",
    "NUM_PARAMS",
    "NavigationScenario",
    "ObstacleKind",
    "ObstacleShape",
    "Obstacles",
    "PARAM_FIELDS",
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
    "action_bounds",
    "drone_config",
    "fused_scenarios",
    "lidar_scan",
    "make",
    "make_scenario",
    "per_env_float_template",
    "register_scenario",
    "rollout",
    "scenario_class",
    "warp_step",
    "__version__",
]
