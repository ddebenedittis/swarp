"""``python -m swarp.rl --scenario <name>``: train a built-in scenario from its recipe."""

from swarp.rl.ppo import parse_args, train

if __name__ == "__main__":
    train(parse_args())
