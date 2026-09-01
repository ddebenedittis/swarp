"""The cross-scenario benchmark registry + parity gate."""

import pytest

from swarp.benchmark.scenarios import (
    MODELS,
    build_scenario,
    resolve_scenarios,
    run_config,
)
from swarp.scenarios import SCENARIOS, fused_scenarios, supports_model

# Derived from the registry, never hand-listed: a scenario that ships fused
# kernels is parity-gated the moment it is registered.
FUSED_SCENARIOS = fused_scenarios()


def test_registry_has_all_scenarios():
    assert set(SCENARIOS) == {
        "navigation",
        "flocking",
        "formation",
        "discovery",
        "sampling",
        "transport",
        "pusht",
        "giveway",
        "caging",
        "shepherding",
    }


def test_fused_scenarios_is_derived_and_covers_every_scenario():
    """Today every registered scenario is fused; the list must come from the classes."""
    assert list(SCENARIOS) == FUSED_SCENARIOS


def test_resolve_scenarios_single_and_all():
    assert resolve_scenarios(["navigation"]) == ["navigation"]
    assert resolve_scenarios(["all"]) == list(SCENARIOS)
    # de-dup while preserving order
    assert resolve_scenarios(["flocking", "flocking", "navigation"]) == [
        "flocking",
        "navigation",
    ]


def test_resolve_scenarios_unknown_lists_valid():
    with pytest.raises(ValueError) as exc:
        resolve_scenarios(["bogus"])
    msg = str(exc.value)
    assert "bogus" in msg
    for name in SCENARIOS:
        assert name in msg


def test_only_navigation_supports_model():
    assert supports_model(SCENARIOS["navigation"])
    for name in ("flocking", "formation", "discovery", "sampling", "transport", "pusht"):
        assert not supports_model(SCENARIOS[name])


def test_build_scenario_applies_model_only_where_supported():
    nav = build_scenario("navigation", 6, "bicycle")
    assert nav.model == MODELS["bicycle"]
    # A model_name is silently ignored for holonomic-only scenarios.
    flock = build_scenario("flocking", 6, "bicycle")
    assert flock.n_agents == 6


@pytest.mark.gpu(reason="use_graph needs CUDA")
@pytest.mark.parametrize("name", FUSED_SCENARIOS)
def test_parity_gate_passes_for_fused_scenarios(name):
    """The optimized (fused + CUDA graph) path must match the baseline path."""
    row = run_config(
        name,
        model_name=None,
        n_envs=64,
        n_agents=8,
        n_rays=0,
        device="cuda:0",
        steps=5,
        warmup=3,
    )
    assert row["status"] == "ok", row.get("detail", row["status"])
    assert row["graph_mode"] == "graph"
    assert row["speedup"] is not None
