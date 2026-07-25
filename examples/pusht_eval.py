"""Evaluate / visualize a Push-T checkpoint saved by ``pusht_torchrl.py``.

Three views of "performance", any combination:

* ``--video out.mp4`` — render one env's rollout to a file (needs the ``viz`` extra).
* ``--window``        — same rollout in a live pygame window.
* always              — a numeric score over ``--n-envs`` parallel envs, next to a
  random-action baseline on the *same* seed, so the number means something.

Also plots ``metrics.csv`` written by the trainer::

    python examples/pusht_eval.py --curve runs/pusht/metrics.csv

Run with::

    python examples/pusht_eval.py runs/pusht/pusht_final.pt [--video pusht.mp4]
    python examples/pusht_eval.py runs/pusht/pusht_iter00100.pt --window
"""

from __future__ import annotations

import argparse
import csv

import torch

# Same directory: Python puts the script's dir on sys.path, so this resolves when
# invoked as `python examples/pusht_eval.py` (the convention used across examples/).
from pusht_torchrl import build_policy
from tensordict import TensorDict

from wmas import Environment, PushTScenario


def _pose(scen) -> tuple[torch.Tensor, torch.Tensor]:
    """Current ``(distance, |wrapped angle error|)`` of the T from its goal pose."""
    d = (scen.tee_pos - scen.goal_pos).norm(dim=-1)
    raw = scen.tee_theta - scen.goal_theta
    return d, torch.atan2(raw.sin(), raw.cos()).abs()


def _load(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    policy = build_policy(ckpt["obs_dim"], ckpt["act_dim"], ckpt["n_agents"], device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, ckpt["n_agents"]


def _greedy(policy, n_envs: int, device: str):
    """``action_fn(obs) -> action`` using the distribution mean (no exploration noise)."""

    def action_fn(obs: torch.Tensor) -> torch.Tensor:
        td = TensorDict({"observation": obs}, batch_size=[n_envs], device=device)
        with torch.no_grad():
            policy(td)
            return td.get("loc").clamp(-1.0, 1.0)

    return action_fn


def score(policy, n_agents: int, n_envs: int, steps: int, device: str, seed: int) -> dict:
    """Mean pose error before/after a fresh episode, for the policy and for random."""
    out = {}
    for name in ("policy", "random"):
        scen = PushTScenario(n_agents=n_agents)
        env = Environment(scen, n_envs=n_envs, device=device, dt=0.05, seed=seed)
        env.reset(seed=seed)
        act = _greedy(policy, n_envs, device)
        gen = torch.Generator(device=device).manual_seed(seed)

        with torch.no_grad():
            d0, a0 = (t.clone() for t in _pose(scen))
            for _ in range(steps):
                if name == "policy":
                    a = act(scen.observations())
                else:
                    a = torch.empty(n_envs, n_agents, 2, device=device).uniform_(
                        -1, 1, generator=gen
                    )
                env.step(a)
            d1, a1 = _pose(scen)
        out[name] = {
            "dist": (d0.mean().item(), d1.mean().item()),
            "angle": (a0.mean().item(), a1.mean().item()),
            "solved": ((d1 < scen.goal_tolerance) & (a1 < scen.angle_tolerance))
            .float()
            .mean()
            .item(),
        }
    return out


def plot_curve(path: str) -> None:
    """ASCII learning curves from the trainer's metrics.csv (no plotting deps)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"{path} is empty")
        return
    for col, label in (
        ("reward_per_step", "reward/step"),
        ("tee_angle", "tee-angle (lower better)"),
        ("tee_dist", "tee-dist  (lower better)"),
    ):
        vals = [float(r[col]) for r in rows]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        # downsample to 60 columns, 9 rows of block glyphs
        step = max(1, len(vals) // 60)
        pts = [sum(vals[i : i + step]) / len(vals[i : i + step]) for i in range(0, len(vals), step)]
        bars = "".join(" ▁▂▃▄▅▆▇█"[int((v - lo) / span * 8)] for v in pts)
        print(f"  {label:26s} [{lo:+.4f} .. {hi:+.4f}]\n    {bars}")


def main() -> None:
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", help="pusht_*.pt from pusht_torchrl.py")
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--n-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--video", help="write an mp4/webm of one env's rollout")
    parser.add_argument("--window", action="store_true", help="live pygame window")
    parser.add_argument("--curve", help="metrics.csv to plot as ASCII curves")
    args = parser.parse_args()

    if args.curve:
        print(f"learning curves from {args.curve}")
        plot_curve(args.curve)
        print()

    if not args.checkpoint:
        return

    policy, n_agents = _load(args.checkpoint, args.device)
    res = score(policy, n_agents, args.n_envs, args.steps, args.device, args.seed)
    print(f"{args.checkpoint}: {args.steps} steps x {args.n_envs} envs (seed {args.seed})")
    for name, m in res.items():
        d0, d1 = m["dist"]
        a0, a1 = m["angle"]
        print(
            f"  {name:7s} dist {d0:.4f} -> {d1:.4f} ({d1 - d0:+.4f})   "
            f"angle {a0:.4f} -> {a1:.4f} ({a1 - a0:+.4f})   solved {m['solved']:.3f}"
        )

    if args.video or args.window:
        scen = PushTScenario(n_agents=n_agents)
        env = Environment(scen, n_envs=1, device=args.device, dt=0.05, seed=args.seed)
        env.reset(seed=args.seed)
        act = _greedy(policy, 1, args.device)
        if args.video:
            from wmas.render.video import save_video

            path = save_video(env, args.video, action_fn=act, n_steps=args.steps, fps=30)
            print(f"  wrote {path}")
        if args.window:
            env.reset(seed=args.seed)
            for _ in range(args.steps):
                env.step(act(scen.observations()))
                env.render(mode="human")
            env.close_viewer()


if __name__ == "__main__":
    main()
