"""MAPPO on any swarp scenario via TorchRL, with TorchRL's own speed features on.

The same algorithm as :mod:`train_mappo` -- the same networks, ``ClipPPOLoss`` settings,
GAE, curriculum, ``metrics.csv`` and checkpoint format, all imported from it rather than
copied -- with what a TorchRL user would reach for to make it fast, following TorchRL's
``sota-implementations/ppo`` scripts. Indicative ms/iteration on navigation (512 envs,
RTX 3070 Laptop) next to the reference trainer's ~240:

* **Update** (~240 -> ~140 alone): loss forward, backward, gradient clip and optimizer
  step as one function wrapped in ``tensordict.nn.CudaGraphModule``. Adam is ``fused`` and
  ``capturable``, and a non-finite gradient norm skips the step on device through Adam's
  ``found_inf`` (the ``GradScaler`` mechanism) instead of the reference trainer's host
  check; the losses come back as device tensors, read once per iteration rather than with
  ``.item()`` per minibatch. Those last two without the graph are worth ~10%.
* **Collector** (~140 -> ~120): ``cudagraph_policy``, the actor forward and its sampling
  replayed as one graph per env step. On TorchRL 0.13 that needs two fixes, see
  :class:`FreshOutputs` and :meth:`TunedUpdate.collector_kwargs`; on 0.14 only the second.
  ``auto_register_policy_transforms=False`` (~7 ms): the policy is not recurrent, and the
  empty ``InitTracker`` the collector otherwise adds
  wraps the env in a ``TransformedEnv`` with unlocked specs, so ``step_mdp`` rebuilt its
  key plan every env step. ``track_traj_ids=False`` and ``return_same_td=True``: nothing
  reads trajectory ids or keeps a batch past its iteration (no measurable gain).
  ``no_cuda_sync=True`` stays, for the stream race described in ``train_mappo``.
* ``--compile`` (~13% on top of ``--tf32``): ``torch.compile`` the update step inside the
  graph (compiling only the loss forward gained half as much) and ``compile_policy`` in
  the collector. ``--tf32``: TF32 matmuls (~5%).

Tried and dropped: ``Distribution.set_default_validate_args(False)`` (no measurable change;
captured code runs no validation anyway) and capturing GAE: TorchRL 0.13-0.14's value
estimators branch on ``nan_mask.any()`` (a sanitizer for ``compact_obs`` collectors'
NaN next-observations) before calling the critic, which a graph cannot capture. GAE stays
eager, ~3 ms.

What remains: the update is GPU-bound (~2.9 ms of kernels per minibatch, where
``swarp.rl``'s is ~2.8 ms including GAE). Around it, TorchRL's API leaves the env
step inside ``EnvBase.step_and_maybe_reset``, which builds TensorDicts and reads
``done.any()`` on the host every step (~18 ms per 32-step rollout whose physics is
~2 ms); a full device sync before every ``CudaGraphModule`` replay; and host-driven
``ReplayBuffer`` sampling plus a copy into the graph's static inputs per minibatch.
``swarp.rl`` avoids these by owning the loop.

Run with::

    python examples/train_mappo_tuned.py --scenario navigation
    python examples/train_mappo_tuned.py --scenario navigation --compile --tf32

Checkpoints are :mod:`train_mappo`'s, so ``examples/eval_mappo.py`` reads them as is.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import CudaGraphModule, TensorDictModuleBase
from torchrl.objectives import ClipPPOLoss
from train_mappo import (
    EagerUpdate,
    TrainConfig,
    UpdateStats,
    build_parser,
    config_from_args,
    train,
)

_MAX_GRAD_NORM = 1.0  # train_mappo's clip


@dataclass
class TunedConfig(TrainConfig):
    """:class:`train_mappo.TrainConfig` plus the speed switches."""

    cudagraphs: bool = True
    compile: bool = False
    tf32: bool = False


def parse_args(argv: list[str] | None = None) -> TunedConfig:
    p = build_parser(__doc__)
    p.add_argument("--cudagraphs", action="store_true", default=None,
                   help="CUDA-graph the collector's policy and the update (default on CUDA)")
    p.add_argument("--no-cudagraphs", dest="cudagraphs", action="store_false")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the collector's policy and the update step")
    p.add_argument("--tf32", action="store_true", help="TF32 matmuls (Ampere and later)")
    a = p.parse_args(argv)
    base = config_from_args(p, a, run_suffix="_torchrl-tuned")
    cuda = torch.device(base.device).type == "cuda"
    if a.cudagraphs and not cuda:
        p.error("--cudagraphs needs a CUDA device")
    return TunedConfig(
        **{f.name: getattr(base, f.name) for f in fields(base)},
        cudagraphs=cuda if a.cudagraphs is None else a.cudagraphs,
        compile=a.compile,
        tf32=a.tf32,
    )


class FreshOutputs(TensorDictModuleBase):
    """``policy`` run on a new TensorDict holding only its inputs.

    ``CudaGraphModule`` returns its static output buffers, uncloned, when the wrapped
    module returns the TensorDict it was given, and the collector keeps a reference per
    step: under ``cudagraph_policy`` every step of a batch then carried the *last* step's
    action and log-prob, and nothing learnt (TorchRL 0.13.3, tensordict 0.13.0). For a
    module that returns a new TensorDict, ``CudaGraphModule`` clones the outputs.

    Fixed upstream in TorchRL 0.14.0, whose collector clones a cudagraphed policy's
    outputs (pytorch/rl#4143, the report is pytorch/rl#3002), and in tensordict 0.15.0
    (pytorch/tensordict#1999). Kept so the script stays correct on 0.13; on 0.14 it learns
    with or without it, at the same ms/iteration.
    """

    def __init__(self, policy: TensorDictModuleBase) -> None:
        super().__init__()
        self.policy = policy
        self.in_keys, self.out_keys = policy.in_keys, policy.out_keys

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.policy(tensordict.select(*self.in_keys))


class TunedUpdate(EagerUpdate):
    """:class:`train_mappo.EagerUpdate` with the minibatch step captured and no host syncs
    until the iteration's statistics are read."""

    def __init__(self, loss_module: ClipPPOLoss, cfg: TunedConfig, frames_per_batch: int):
        super().__init__(loss_module, cfg, frames_per_batch)
        self.params = list(loss_module.parameters())
        # fused + capturable: a step that can be captured, and a kernel that honours
        # ``found_inf`` by leaving the weights, the moments and the step count untouched.
        cuda = torch.device(cfg.device).type == "cuda"
        self.optim = torch.optim.Adam(self.params, lr=cfg.spec.lr, fused=True, capturable=cuda)
        self.found_inf = torch.zeros((), device=cfg.device)
        self.optim.found_inf = self.found_inf
        self.optim.grad_scale = None
        # As TorchRL's sota PPO: compile the whole step, then capture the compiled code.
        self.step = torch.compile(self._step) if cfg.compile else self._step
        if cfg.cudagraphs:
            self.step = CudaGraphModule(self.step, in_keys=[], out_keys=[], warmup=5)

    @staticmethod
    def collector_kwargs(policy: torch.nn.Module, cfg: TunedConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            auto_register_policy_transforms=False, track_traj_ids=False, return_same_td=True
        )
        if cfg.cudagraphs:
            # TanhNormal turns float bounds into device tensors on every construction, an
            # H2D copy a CUDA graph cannot capture. Same values, so the same distribution.
            policy[-1].distribution_kwargs.update(
                low=torch.tensor(-1.0, device=cfg.device),
                high=torch.tensor(1.0, device=cfg.device),
            )
            kwargs["policy"] = FreshOutputs(policy)
            kwargs["cudagraph_policy"] = {"warmup": 10}
        if cfg.compile:
            kwargs["compile_policy"] = {"warmup": 1}
        return kwargs

    def _step(self, sub: TensorDictBase) -> TensorDict:
        """One minibatch. Returns ``[objective, critic, entropy, skipped]``, losses zeroed
        on a skipped step so the sums cover the applied ones only."""
        losses = self.loss_module(sub)
        total = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
        self.optim.zero_grad()
        total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.params, _MAX_GRAD_NORM, foreach=True)
        found = (~torch.isfinite(gnorm)).float()
        self.found_inf.copy_(found)
        self.optim.step()
        kept = torch.stack(
            [losses["loss_objective"], losses["loss_critic"], losses["entropy"]]
        ).detach()
        return TensorDict(
            stats=torch.cat([torch.where(found > 0, torch.zeros_like(kept), kept), found[None]])
        )

    def __call__(self, batch: TensorDictBase, coeff: float) -> UpdateStats:
        loss_module, buffer, spec = self.loss_module, self.buffer, self.spec
        with torch.no_grad():
            loss_module.value_estimator(
                batch,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
        # The loss reads none of ("next", ...): leaving it out halves what each minibatch
        # copies into the graph's static inputs.
        buffer.extend(batch.exclude("next").reshape(-1))
        n_mb = spec.epochs * spec.minibatches
        out = [self.step(buffer.sample())["stats"] for _ in range(n_mb)]
        buffer.empty()
        obj, crit, ent, skips = torch.stack(out).sum(0).tolist()  # the one host read
        skips = int(skips)
        return UpdateStats(skips, n_mb - skips, ent, obj, crit)


def main() -> None:
    cfg = parse_args()
    if torch.device(cfg.device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    print(
        f"cudagraphs {'on' if cfg.cudagraphs else 'off'}, "
        f"compile {'on' if cfg.compile else 'off'}, tf32 {'on' if cfg.tf32 else 'off'}"
    )
    train(cfg, learner=TunedUpdate)


if __name__ == "__main__":
    main()
