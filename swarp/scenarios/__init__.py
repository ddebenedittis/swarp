"""Built-in scenarios and **the** scenario registry.

:data:`SCENARIOS` is the single source of truth mapping a short name to a scenario
class. Everything that needs to enumerate scenarios — the cross-scenario benchmark
CLI (:mod:`swarp.benchmark.scenarios`), :func:`swarp.make`, the tests — reads it from
here, so adding a scenario (or lighting up its fused kernels) surfaces it everywhere
without a second hand-maintained list going stale.

Fused capability is *derived* from the classes (:func:`fused_scenarios`), never
listed by hand.
"""

from __future__ import annotations

import inspect
import warnings

from swarp.scenarios.base import Scenario
from swarp.scenarios.caging import CagingScenario
from swarp.scenarios.discovery import DiscoveryScenario
from swarp.scenarios.flocking import FlockingScenario
from swarp.scenarios.formation import FormationScenario
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario
from swarp.scenarios.giveway import GiveWayScenario
from swarp.scenarios.navigation import NavigationScenario
from swarp.scenarios.pusht import PushTScenario
from swarp.scenarios.sampling import SamplingScenario
from swarp.scenarios.shepherding import ShepherdingScenario
from swarp.scenarios.transport import TransportScenario

__all__ = [
    "SCENARIOS",
    # The two base classes a scenario author subclasses, and the fused vocabulary.
    "Buf",
    "CagingScenario",
    "DiscoveryScenario",
    "FlockingScenario",
    "FormationScenario",
    "FusedPass",
    "FusedScenario",
    "GiveWayScenario",
    "NavigationScenario",
    "PushTScenario",
    "SamplingScenario",
    "Scenario",
    "ShepherdingScenario",
    "TransportScenario",
    "fused_scenarios",
    "make_scenario",
    "register_scenario",
    "resolve_scenarios",
    "scenario_class",
    "supports_model",
]

#: name -> scenario class. The registry; insertion order is the canonical order.
SCENARIOS: dict[str, type[Scenario]] = {
    "navigation": NavigationScenario,
    "flocking": FlockingScenario,
    "formation": FormationScenario,
    "discovery": DiscoveryScenario,
    "sampling": SamplingScenario,
    "transport": TransportScenario,
    "pusht": PushTScenario,
    "giveway": GiveWayScenario,
    "shepherding": ShepherdingScenario,
    "caging": CagingScenario,
}


# -------------------------------------------------------------------- registration


def register_scenario(name: str, cls: type[Scenario], *, overwrite: bool = False) -> None:
    """Register an out-of-tree scenario under ``name``.

    This is how a scenario that lives outside the package reaches everything that reads
    the registry — :func:`swarp.make`, :func:`make_scenario`, :func:`scenario_class`,
    :func:`fused_scenarios`, :func:`resolve_scenarios`, and the benchmark CLIs that take
    ``--scenario`` — without editing (or forking) the installed package::

        from swarp.scenarios import register_scenario

        register_scenario("my_task", MyTaskScenario)
        env = swarp.make("my_task", n_envs=4096, n_agents=8)

    Built-in scenarios are entries in the :data:`SCENARIOS` literal instead; this is the
    door for everyone else. The registry is process-global, so call it at import time of
    the module defining the scenario.

    Args:
        name: the short name callers will pass.
        cls: a :class:`~swarp.scenarios.base.Scenario` subclass (the class, not an
            instance — the registry constructs it per environment).
        overwrite: allow replacing an existing entry. Off by default so a name
            collision — including shadowing a built-in — is an error rather than a
            silent swap.

    Raises:
        TypeError: ``cls`` is not a ``Scenario`` subclass.
        ValueError: ``name`` is empty, or already registered and ``overwrite`` is False.
    """
    if not name:
        raise ValueError("scenario name must be a non-empty string")
    if not (isinstance(cls, type) and issubclass(cls, Scenario)):
        raise TypeError(f"cls must be a Scenario subclass, got {cls!r}")
    if name in SCENARIOS and not overwrite:
        raise ValueError(
            f"scenario {name!r} is already registered as {SCENARIOS[name].__name__}; "
            "pass overwrite=True to replace it"
        )
    SCENARIOS[name] = cls


# ------------------------------------------------------------------ introspection


def scenario_class(name: str) -> type[Scenario]:
    """Look up a registered scenario class; ``ValueError`` lists the valid names."""
    try:
        return SCENARIOS[name]
    except KeyError:
        raise ValueError(
            f"unknown scenario {name!r}; valid names are {list(SCENARIOS)}"
        ) from None


def supports_model(cls: type[Scenario]) -> bool:
    """Whether ``cls.__init__`` accepts a ``model`` kwarg (i.e. is not holonomic-only)."""
    return "model" in inspect.signature(cls.__init__).parameters


def fused_scenarios() -> list[str]:
    """Registered scenarios that ship fused Warp obs/reward kernels.

    Read straight off :attr:`~swarp.scenarios.base.Scenario.fused_available`, which is a
    class attribute (``True`` for every :class:`~swarp.scenarios.fused.FusedScenario`)
    precisely so this needs no instance and no hand-maintained list.
    """
    return [name for name, cls in SCENARIOS.items() if cls.fused_available]


def resolve_scenarios(names: list[str]) -> list[str]:
    """Expand a scenario selection (names, or the literal ``"all"``).

    De-duplicates while preserving the caller's order. Raises ``ValueError``
    (listing the valid names) on any unknown name.
    """
    valid = list(SCENARIOS)
    if names == ["all"] or "all" in names:
        return valid
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenario(s) {unknown}; valid names are {valid} (or 'all')")
    seen: dict[str, None] = {}
    for n in names:
        seen.setdefault(n, None)
    return list(seen)


# ------------------------------------------------------------------- construction


def make_scenario(name: str, **kwargs) -> Scenario:
    """Construct a registered scenario by name.

    ``model=`` is dropped for scenarios whose ``__init__`` does not accept it (the
    holonomic-only ones), so a caller sweeping the robot-model axis over the whole
    registry needs no special-casing. Dropping it **warns**, though: the alternative is
    ``swarp.make("flocking", model=DynamicsModel.DRONE)`` handing back a holonomic env
    with no signal at all, which reads as a silent no-op rather than a design choice.

    A sweep that expects the drop should filter on :func:`supports_model` first — the
    benchmark's scenario sweep does — and then the warning never fires.

    All other kwargs are forwarded verbatim.
    """
    cls = scenario_class(name)
    if "model" in kwargs and not supports_model(cls):
        dropped = kwargs["model"]
        kwargs = {k: v for k, v in kwargs.items() if k != "model"}
        warnings.warn(
            f"scenario {name!r} is holonomic-only: its __init__ takes no 'model', so "
            f"model={dropped!r} was dropped and the fleet stays holonomic. Filter with "
            "swarp.scenarios.supports_model() to sweep this axis without the warning.",
            stacklevel=2,
        )
    return cls(**kwargs)
