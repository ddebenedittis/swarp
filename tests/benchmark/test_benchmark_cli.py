"""The benchmark front doors: the swarp adapter and the two comparison CLIs.

These generate the numbers in ``docs/benchmarks.md`` and had no test at all. The
cross-simulator adapters genuinely cannot be covered here (they need vmas / jaxmarl /
camar, which CI cannot install), but ``swarp_adapter`` needs no optional dependency and
the argument parsers are pure Python — so an import error, a bad ``choices`` list or a
broken ``--python SIM=PATH`` parser stops being something only a manual benchmark run
would find.
"""

import argparse

import pytest
import torch
from conftest import DEVICES

from swarp.benchmark import compare_sims, compare_vmas
from swarp.benchmark._adapters import swarp_adapter

# ------------------------------------------------------------------- the adapter


@pytest.mark.parametrize("device", DEVICES)
def test_swarp_adapter_builds_and_rolls_out(device):
    runner = swarp_adapter.build("navigation", n_envs=4, n_agents=3, device=device, seed=0)
    assert runner.n_envs == 4 and runner.n_agents == 3 and runner.backend == "torch"
    with torch.no_grad():
        runner.rollout(3)
    runner.sync()
    runner.close()


def test_swarp_adapter_rejects_unimplemented_scenarios():
    """The adapter is navigation-only; a typo must not silently benchmark something else."""
    with pytest.raises(ValueError, match="navigation"):
        swarp_adapter.build("flocking", n_envs=2, n_agents=2, device="cpu")


@pytest.mark.parametrize("fused,use_graph", [(False, False), ("auto", False)])
def test_swarp_adapter_honours_its_configuration_knobs(fused, use_graph):
    """``fused``/``use_graph`` are the whole point of this adapter existing."""
    runner = swarp_adapter.build(
        "navigation", n_envs=2, n_agents=2, device="cpu", fused=fused, use_graph=use_graph
    )
    with torch.no_grad():
        runner.rollout(2)
    runner.close()


# ---------------------------------------------------------------- the arg parsers


def test_compare_sims_parser_defaults():
    args = compare_sims.build_parser().parse_args([])
    assert set(args.sims) <= set(compare_sims.ALL_SIMS)
    assert "swarp" in args.sims  # swarp is the anchor every table is normalized against
    assert set(args.scenarios) <= set(compare_sims.SCENARIOS)
    assert args.steps > 0 and args.warmup >= 0 and args.envs and args.agents


def test_compare_sims_parser_maps_interpreters():
    """``--python SIM=PATH`` is how the multi-venv setup in docs/benchmarks.md is driven."""
    args = compare_sims.build_parser().parse_args(["--python", "jaxmarl=/venvs/jax/bin/python"])
    assert args.python == [("jaxmarl", "/venvs/jax/bin/python")]


@pytest.mark.parametrize("bad", ["jaxmarl", "bogus=/x/python"])
def test_compare_sims_parser_rejects_bad_interpreter_maps(bad):
    with pytest.raises(SystemExit):
        compare_sims.build_parser().parse_args(["--python", bad])


def test_compare_sims_sim_path_error_messages():
    """Exercised directly too: argparse swallows the message into a SystemExit above."""
    with pytest.raises(argparse.ArgumentTypeError, match="SIM=PATH"):
        compare_sims._sim_path("jaxmarl")
    with pytest.raises(argparse.ArgumentTypeError, match="unknown sim"):
        compare_sims._sim_path("bogus=/x/python")


def test_compare_vmas_parser_metric_choices():
    """``--metric memory`` is the path compare_sims does not have; keep it reachable."""
    parser = compare_vmas.build_parser()
    assert parser.parse_args([]).metric == "throughput"
    for metric in ("throughput", "memory", "both"):
        assert parser.parse_args(["--metric", metric]).metric == metric
    with pytest.raises(SystemExit):
        parser.parse_args(["--metric", "bogus"])


def test_compare_vmas_hidden_child_flag_still_parses():
    """The memory probe re-execs itself with this; a rename would break it silently."""
    args = compare_vmas.build_parser().parse_args(["--_mem_child", "swarp", "8", "2", "5"])
    assert args._mem_child == ["swarp", "8", "2", "5"]
