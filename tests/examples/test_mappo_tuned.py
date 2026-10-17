"""The tuned TorchRL trainer's update is the reference trainer's update, captured.

``train_mappo_tuned.TunedUpdate`` replaces ``train_mappo.EagerUpdate``'s per-minibatch host
logic (``.item()`` reads, a host-side non-finite check) with a CUDA-graphed step and an
on-device skip. Both are fed the same collected batches from identical weights; the
losses and the weights must agree to float32 round-off through warm-up, capture and
replay, and a non-finite batch must be skipped by both without touching the weights.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import pytest

pytest.importorskip("torchrl", reason="examples/ needs the optional [torchrl] extra")

# See test_mappo.py: the example scripts import each other as top-level modules.
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import torch  # noqa: E402
from mappo import (  # noqa: E402
    DONE_KEY,
    TERM_KEY,
    build_critic,
    build_env,
    build_policy,
    expand_flag,
)
from torchrl.collectors import Collector  # noqa: E402
from train_mappo import EagerUpdate, build_loss, parse_args  # noqa: E402
from train_mappo_tuned import TunedConfig, TunedUpdate  # noqa: E402

from swarp.interop.torchrl import SwarpEnv  # noqa: E402

DEVICE = "cuda:0"
# CudaGraphModule(warmup=5): four eager calls, a capture, then replays.
CALLS = 7


def _setup(tmp_path):
    # One minibatch over the whole batch, so the sampler's permutation only reorders a
    # mean, and no entropy bonus, so the one-sample Monte Carlo entropy (whose noise the
    # graph draws from a different Philox offset) does not reach the gradient.
    cfg = parse_args([
        "--scenario", "navigation", "--device", DEVICE, "--n-envs", "64",
        "--steps-per-batch", "8", "--epochs", "1", "--minibatches", "1", "--num-cells", "32",
        "--entropy-coeff", "0", "--checkpoint-dir", str(tmp_path),
    ])
    spec = cfg.spec
    sim = build_env(
        "navigation", n_envs=spec.n_envs, n_agents=spec.n_agents, device=DEVICE, dt=spec.dt,
        substeps=spec.substeps, max_steps=spec.max_steps, seed=0, scen_kwargs=spec.scen_kwargs,
    )
    env = SwarpEnv(sim)
    nets = []
    for _ in range(2):
        torch.manual_seed(0)
        nets.append((
            build_policy(env.obs_dim, env.act_dim, spec.n_agents, DEVICE, num_cells=32),
            build_critic(env.obs_dim, spec.n_agents, DEVICE, num_cells=32),
        ))
    fpb = spec.n_envs * spec.steps_per_batch
    # A fresh batch per call: a replay that read stale inputs would not match.
    collector = Collector(
        env, nets[0][0], frames_per_batch=fpb, total_frames=fpb * CALLS, device=DEVICE,
        no_cuda_sync=True, auto_register_policy_transforms=False,
    )
    batches = []
    for batch in collector:
        batch.set(DONE_KEY, expand_flag(batch.get(("next", "done")), spec.n_agents))
        batch.set(TERM_KEY, expand_flag(batch.get(("next", "terminated")), spec.n_agents))
        batches.append(batch)
    collector.shutdown()

    tuned_cfg = TunedConfig(**{f.name: getattr(cfg, f.name) for f in fields(cfg)})
    eager = EagerUpdate(build_loss(*nets[0], spec), cfg, fpb)
    tuned = TunedUpdate(build_loss(*nets[1], tuned_cfg.spec), tuned_cfg, fpb)
    TunedUpdate.collector_kwargs(nets[1][0], tuned_cfg)  # the device-tensor TanhNormal bounds
    return batches, eager, tuned


def _params(update):
    return [p.detach().clone() for p in update.loss_module.parameters()]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CudaGraphModule needs CUDA")
def test_tuned_update_matches_eager(tmp_path):
    batches, eager, tuned = _setup(tmp_path)
    for call, batch in enumerate(batches):
        se, st = eager(batch.clone(), 0.0), tuned(batch.clone(), 0.0)
        assert (se.skips, se.updates) == (st.skips, st.updates) == (0, 1), call
        assert st.obj_sum == pytest.approx(se.obj_sum, rel=1e-4, abs=1e-6), call
        assert st.crit_sum == pytest.approx(se.crit_sum, rel=1e-4, abs=1e-6), call
        for pe, pt in zip(_params(eager), _params(tuned), strict=True):
            torch.testing.assert_close(pt, pe, rtol=1e-4, atol=1e-5, msg=f"call {call}")

    # A non-finite observation poisons the loss: both skip, and in the captured graph the
    # skip leaves weights (and Adam state) exactly as they were.
    bad = batches[0].clone()
    bad["observation"][0, 0, 0, 0] = float("nan")
    before = _params(tuned)
    se, st = eager(bad.clone(), 0.0), tuned(bad.clone(), 0.0)
    assert (se.skips, st.skips) == (1, 1)
    assert st.updates == 0
    for p0, p1 in zip(before, _params(tuned), strict=True):
        assert torch.equal(p0, p1)
