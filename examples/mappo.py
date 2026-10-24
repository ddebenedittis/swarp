"""Shared MAPPO machinery for swarp's TorchRL reference trainer.

Both :mod:`train_mappo` and :mod:`eval_mappo` import from here, so a checkpoint written
by the trainer loads into a byte-identical module in the evaluator. Two groups of
things live here:

1. **The networks** — :func:`build_policy` / :func:`build_critic`. Decentralised actor
   with shared weights, centralised critic: the usual MAPPO split (which is what this is;
   IPPO would be the same code with ``centralised=False`` on the critic).
2. **Re-exports from** :mod:`swarp.rl.recipes` — the per-scenario recipe table
   (:data:`SPECS`) and the honest-accounting helpers (:class:`SolveRateWindow`,
   :func:`wilson_ci`, :func:`mcnemar_interval`, :func:`score`). They are torchrl-free, so
   the lean trainer in :mod:`swarp.rl` shares them. :func:`score` here takes a torchrl
   policy module and wraps it in :func:`greedy_actions`.

Needs the optional torchrl extra::

    uv pip install -e '.[torchrl]'
"""

from __future__ import annotations

from functools import partial

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import MultiAgentMLP, NormalParamExtractor, ProbabilisticActor, TanhNormal

from swarp.rl import recipes
from swarp.rl.recipes import (  # noqa: F401  (re-exported for train_mappo / eval_mappo / tests)
    BASELINES,
    SPECS,
    UNTRAINABLE,
    ActionFn,
    ArmResult,
    CurriculumKnob,
    SolveRateWindow,
    SuccessSpec,
    TrainSpec,
    build_env,
    coerce_scen_kwargs,
    goal_p_controller,
    mcnemar_interval,
    plot_curve,
    random_actions,
    requires_success_spec,
    scenario_name,
    wilson_ci,
    zero_actions,
)

# ---------------------------------------------------------------- done/reward broadcast

# SwarpEnv is flat (no ("agents", ...) group) and emits one shared done per env,
# [n_envs, 1], while reward is per-agent [n_envs, n_agents, 1]. GAE needs the two
# broadcastable, so we expand done/terminated into these dedicated keys each batch
# and point the value estimator at them.
DONE_NAME, TERM_NAME = "agents_done", "agents_terminated"
DONE_KEY, TERM_KEY = ("next", DONE_NAME), ("next", TERM_NAME)


def expand_flag(t: torch.Tensor, n_agents: int) -> torch.Tensor:
    """``[*B, 1]`` shared flag -> ``[*B, n_agents, 1]``, matching the reward shape."""
    return t.unsqueeze(-2).expand(*t.shape[:-1], n_agents, 1)


# ------------------------------------------------------------------------- the networks


def build_policy(obs_dim: int, act_dim: int, n_agents: int, device: str, num_cells: int = 128):
    """Decentralised actor with shared weights (the MAPPO actor half).

    Shared with ``eval_mappo.py`` so a checkpoint loads into an identical module.

    **Do not restructure this function.** The saved ``state_dict`` keys encode the module
    nesting -- ``module.0.module.0.params.{0,2,4}.{weight,bias}`` -- so inserting a layer,
    changing ``depth``, or wrapping the actor differently silently invalidates every
    existing checkpoint. The ``num_cells=128`` default is likewise load-bearing:
    ``eval_mappo``'s loader falls back to it for checkpoints written before ``num_cells``
    was recorded.
    """
    net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * act_dim,  # loc + scale, split below
            n_agents=n_agents,
            centralised=False,
            share_params=True,
            device=device,
            depth=2,
            num_cells=num_cells,
        ),
        NormalParamExtractor(),
    )
    return ProbabilisticActor(
        module=TensorDictModule(net, in_keys=["observation"], out_keys=["loc", "scale"]),
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": -1.0, "high": 1.0},
        return_log_prob=True,
    )


def build_critic(obs_dim: int, n_agents: int, device: str, num_cells: int = 128):
    """Centralised critic (the MAPPO critic half): every agent's obs, one value.

    Its input width is ``n_agents * obs_dim``, which is why a critic cannot be transferred
    across an agent-count curriculum stage -- see the note in ``TrainSpec``.
    """
    return TensorDictModule(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=1,
            n_agents=n_agents,
            centralised=True,
            share_params=True,
            device=device,
            depth=2,
            num_cells=num_cells,
        ),
        in_keys=["observation"],
        out_keys=["state_value"],
    )


def greedy_actions(
    policy, n_envs: int, device: str, lo: torch.Tensor, hi: torch.Tensor
) -> ActionFn:
    """``obs -> action`` from the distribution mean: no exploration noise, no RNG.

    Consuming no randomness is what keeps this arm *paired* with the random arm -- both
    see bit-identical initial conditions.
    """

    def action_fn(obs: torch.Tensor) -> torch.Tensor:
        td = TensorDict({"observation": obs}, batch_size=[n_envs], device=device)
        with torch.no_grad():
            policy(td)
            return td.get("loc").clamp(lo, hi)

    return action_fn




def score(scenario: str, *, policy=None, **kwargs) -> dict[str, ArmResult]:
    """:func:`swarp.rl.recipes.score` with a torchrl ``policy`` module as the policy arm."""
    arm = None if policy is None else partial(greedy_actions, policy)
    return recipes.score(scenario, policy=arm, **kwargs)
