"""Evaluate / visualize a Push-T checkpoint saved by ``pusht_torchrl.py``.

Three views of "performance", any combination:

* ``--video out.mp4`` — render one env's rollout to a file (needs the ``viz`` extra).
* ``--window``        — same rollout in a live pygame window.
* always              — a numeric score over ``--n-envs`` parallel envs, next to a
  random-action baseline on the *same* seed, so the number means something.

The rendered rollout runs ``--render-steps`` steps and respawns the T and its goal
every ``--episode-steps``, so it plays as a continuous run of episodes rather than
one clip that freezes once the T is parked. The *scored* rollout is separate and
stays a single un-reset episode (``--steps``), since it measures start-to-end
displacement (and counts an env as solved if it is *ever* inside both tolerances,
which is where an episode would terminate in training).

The T's **target pose** is drawn as a green outline (the ``goal_pose`` overlay, fed by
``PushTScenario.render_extras``; toggle it with ``p`` in the live window). Playback is
half real time by default — ``--speed 1`` matches the simulated clock, ``--speed 0.25``
crawls; ``--fps`` overrides the implied frame rate outright.

Also plots ``metrics.csv`` written by the trainer::

    python examples/pusht_eval.py --curve runs/pusht_v11/metrics.csv

Run with::

    python examples/pusht_eval.py runs/pusht_v11/pusht_final.pt [--video pusht.mp4]
    python examples/pusht_eval.py runs/pusht_v11/pusht_final.pt --window --speed 0.25
    python examples/pusht_eval.py runs/pusht_v11/pusht_final.pt --video long.mp4 \
        --render-steps 2400 --episode-steps 400        # 6 episodes back to back
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
    policy = build_policy(
        ckpt["obs_dim"], ckpt["act_dim"], ckpt["n_agents"], device,
        num_cells=ckpt.get("num_cells", 128),
    )
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


def score(
    policy, n_agents: int, n_envs: int, steps: int, device: str, seed: int, substeps: int = 8
) -> dict:
    """Mean pose error before/after a fresh episode, for the policy and for random."""
    out = {}
    for name in ("policy", "random"):
        scen = PushTScenario(n_agents=n_agents)
        env = Environment(
            scen, n_envs=n_envs, device=device, dt=0.05, seed=seed, substeps=substeps
        )
        env.reset(seed=seed)
        act = _greedy(policy, n_envs, device)
        gen = torch.Generator(device=device).manual_seed(seed)

        solved = torch.zeros(n_envs, dtype=torch.bool, device=device)
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
                # Termination semantics: the episode would end at the first step the
                # pose is inside both tolerances, so count envs that ever get there.
                solved |= (d1 < scen.goal_tolerance) & (a1 < scen.angle_tolerance)
        out[name] = {
            "dist": (d0.mean().item(), d1.mean().item()),
            "angle": (a0.mean().item(), a1.mean().item()),
            "solved": solved.float().mean().item(),
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
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--substeps", type=int, default=8,
        help="physics substeps; keep equal to training (the stiff contact needs them)",
    )
    parser.add_argument("--video", help="write an mp4/webm of one env's rollout")
    parser.add_argument("--window", action="store_true", help="live pygame window")
    parser.add_argument(
        "--render-steps", type=int, default=1200, help="total sim steps to render"
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=400,
        help="steps before the T and goal respawn; the rollout keeps going",
    )
    # One sim step is dt=0.05 s, so 20 frames = 1 s of simulated time. --speed is a
    # multiple of that real-time rate: 1.0 plays as fast as the T actually moves,
    # 0.5 (the default) plays it at half speed, which is much easier to follow.
    parser.add_argument("--speed", type=float, default=0.5, help="playback speed vs real time")
    parser.add_argument("--fps", type=int, help="override the frame rate implied by --speed")
    parser.add_argument("--curve", help="metrics.csv to plot as ASCII curves")
    args = parser.parse_args()

    if args.curve:
        print(f"learning curves from {args.curve}")
        plot_curve(args.curve)
        print()

    if not args.checkpoint:
        return

    policy, n_agents = _load(args.checkpoint, args.device)
    res = score(
        policy, n_agents, args.n_envs, args.steps, args.device, args.seed,
        substeps=args.substeps,
    )
    print(f"{args.checkpoint}: {args.steps} steps x {args.n_envs} envs (seed {args.seed})")
    for name, m in res.items():
        d0, d1 = m["dist"]
        a0, a1 = m["angle"]
        print(
            f"  {name:7s} dist {d0:.4f} -> {d1:.4f} ({d1 - d0:+.4f})   "
            f"angle {a0:.4f} -> {a1:.4f} ({a1 - a0:+.4f})   solved {m['solved']:.3f}"
        )

    if args.video or args.window:
        # auto_reset + max_steps makes the episode respawn in place when the T reaches
        # the goal or the time limit hits, so a long rollout plays as a continuous
        # sequence of episodes instead of one clip that freezes at the end.
        dt = 0.05
        scen = PushTScenario(n_agents=n_agents)
        env = Environment(
            scen,
            n_envs=1,
            device=args.device,
            dt=dt,
            substeps=args.substeps,
            seed=args.seed,
            max_steps=args.episode_steps,
            auto_reset=True,
        )
        env.reset(seed=args.seed)
        act = _greedy(policy, 1, args.device)
        n = args.render_steps
        fps = args.fps if args.fps else max(1, round(args.speed / dt))
        print(
            f"  rendering {n} steps at {fps} fps ({args.speed:g}x real time), "
            f"respawning every {args.episode_steps}"
        )
        if args.video:
            from wmas.render.video import save_video

            path = save_video(env, args.video, action_fn=act, n_steps=n, fps=fps)
            print(f"  wrote {path} ({n / fps:.0f}s)")
        if args.window:
            import pygame

            env.reset(seed=args.seed)
            clock = pygame.time.Clock()
            for _ in range(n):
                env.step(act(scen.observations()))
                env.render(mode="human")
                clock.tick(fps)  # the window loop is otherwise GPU-speed, i.e. a blur
            env.close_viewer()


if __name__ == "__main__":
    main()
