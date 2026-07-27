"""Built-in scenarios and **the** scenario registry.

:data:`SCENARIOS` is the single source of truth mapping a short name to a scenario
class. Everything that needs to enumerate scenarios — the cross-scenario benchmark
CLI (:mod:`wmas.benchmark.scenarios`), :func:`wmas.make`, the tests — reads it from
here, so adding a scenario (or lighting up its fused kernels) surfaces it everywhere
without a second hand-maintained list going stale.

Fused capability is *derived* from the classes (:func:`fused_scenarios`), never
listed by hand.
"""

from __future__ import annotations

import inspect

from wmas.scenarios.base import Scenario
from wmas.scenarios.discovery import DiscoveryScenario
from wmas.scenarios.flocking import FlockingScenario
from wmas.scenarios.formation import FormationScenario
from wmas.scenarios.navigation import NavigationScenario
from wmas.scenarios.pusht import PushTScenario
from wmas.scenarios.sampling import SamplingScenario
from wmas.scenarios.transport import TransportScenario

__all__ = [
    "DiscoveryScenario",
    "FlockingScenario",
    "FormationScenario",
    "NavigationScenario",
    "PushTScenario",
    "SCENARIOS",
    "SamplingScenario",
    "Scenario",
    "TransportScenario",
    "fused_scenarios",
    "make_scenario",
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
}


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


def _class_fused_available(cls: type[Scenario]) -> bool:
    """Query ``fused_available`` without paying for a constructed scenario.

    ``Scenario.fused_available`` is an *instance* method (a class-level declaration
    would be cleaner, but changing that contract belongs to the fused-scenario
    refactor). None of the implementations read ``self`` — they return a literal —
    so an uninitialized instance is enough. The ``cls()`` fallback covers a future
    implementation that does look at instance state; nothing is swallowed, so a
    scenario that cannot answer fails loudly instead of silently reporting False.
    """
    try:
        return bool(cls.fused_available(cls.__new__(cls)))
    except (AttributeError, TypeError):
        return bool(cls().fused_available())


def fused_scenarios() -> list[str]:
    """Registered scenarios that ship fused Warp obs/reward kernels.

    Derived from ``fused_available()`` on every registered class, in registry order.
    """
    return [name for name, cls in SCENARIOS.items() if _class_fused_available(cls)]


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

    ``model=`` is dropped for scenarios whose ``__init__`` does not accept it
    (the holonomic-only ones), so a caller sweeping the robot-model axis over the
    whole registry — the benchmark does exactly this — needs no special-casing.
    All other kwargs are forwarded verbatim.
    """
    cls = scenario_class(name)
    if "model" in kwargs and not supports_model(cls):
        kwargs = {k: v for k, v in kwargs.items() if k != "model"}
    return cls(**kwargs)
