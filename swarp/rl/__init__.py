"""Training on swarp: a lean MAPPO whose rollout and update are CUDA-graph captured.

``python -m swarp.rl --scenario navigation`` trains a built-in scenario from its recipe in
:data:`swarp.rl.recipes.SPECS`; :func:`train` is the same loop as a function.
"""

from swarp.rl.networks import (
    Actor,
    Critic,
    DenseGATv2,
    PolicyNet,
    StateCritic,
    role_permutation,
)
from swarp.rl.ppo import (
    ARCHS,
    MAPPO,
    Minibatch,
    NonFiniteGradients,
    PPORule,
    TrainConfig,
    UpdateRule,
    check_action_box,
    greedy_arm,
    greedy_env_arm,
    greedy_score_kwargs,
    load_checkpoint,
    load_policy,
    parse_args,
    train,
    write_grads,
)
from swarp.rl.recipes import SPECS, TrainSpec

__all__ = [
    "ARCHS",
    "MAPPO",
    "SPECS",
    "Actor",
    "Critic",
    "DenseGATv2",
    "Minibatch",
    "NonFiniteGradients",
    "PPORule",
    "PolicyNet",
    "StateCritic",
    "TrainConfig",
    "TrainSpec",
    "UpdateRule",
    "check_action_box",
    "greedy_arm",
    "greedy_env_arm",
    "greedy_score_kwargs",
    "load_checkpoint",
    "load_policy",
    "parse_args",
    "role_permutation",
    "train",
    "write_grads",
]
