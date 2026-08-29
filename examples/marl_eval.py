"""Evaluate / visualize a checkpoint saved by ``marl_train.py``, for any scenario.

Three views of "performance", any combination:

* always              — a numeric score over ``--n-envs`` parallel envs, next to a
  random-action baseline on the **same seed**, so the number means something. A learned
  policy that merely looks busy scores like random; this is what separates the two.
* ``--video out.webm`` — render one env's rollout to a file (needs the ``viz`` extra).
* ``--window``        — the same rollout in a live pygame window.

The scenario is read from the checkpoint, so the only required argument is the file::

    python examples/marl_eval.py runs/giveway/policy_final.pt
    python examples/marl_eval.py runs/caging/policy_final.pt --video caging.webm
    python examples/marl_eval.py --curve runs/shepherding/metrics.csv    # curves only

Scores are reported per ``info()`` key as ``start -> end``, plus ``solved``: the fraction of
envs that were *ever* inside the scenario's own termination condition during the rollout.
"Ever" rather than "at the end" is deliberate — that is the step an episode would actually
have terminated on in training, so it is the quantity the training curve is measuring.

The rendered rollout runs ``--render-steps`` steps with ``auto_reset``, so it plays as a
continuous run of episodes rather than one clip that freezes once the task is solved.
Playback runs at real time by default; ``--speed 0.25`` crawls. ``--window`` hands the loop to
:class:`~swarp.render.viewer.Viewer`, so the usual controls work: space pauses, ``.`` steps
once while paused, ``r`` resets, ``?`` lists the rest.
"""

from __future__ import annotations

import argparse
import csv

import torch

# Same directory: Python puts the script's dir on sys.path, so this resolves when invoked as
# `python examples/marl_eval.py` (the convention used across examples/).
from marl_train import TASKS, Task, build_env, build_policy, info_keys
from tensordict import TensorDict

from swarp import Environment


def _load(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    policy = build_policy(
        ckpt["obs_dim"], ckpt["act_dim"], ckpt["n_agents"], device,
        num_cells=ckpt.get("num_cells", 256),
    )
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, ckpt


def _greedy(policy, n_envs: int, device: str):
    """``action_fn(obs) -> action`` using the distribution mean (no exploration noise)."""

    def action_fn(obs: torch.Tensor) -> torch.Tensor:
        td = TensorDict({"observation": obs}, batch_size=[n_envs], device=device)
        with torch.no_grad():
            policy(td)
            return td.get("loc").clamp(-1.0, 1.0)

    return action_fn


def score(policy, name: str, task: Task, n_agents: int, n_envs: int, steps: int,
          device: str, seed: int) -> dict:
    """Per-info-key start/end means and a solved rate, for the policy and for random."""
    out = {}
    for who in ("policy", "random"):
        sim = build_env(name, task, n_agents=n_agents, n_envs=n_envs, device=device, seed=seed)
        sim.reset(seed=seed)
        keys = info_keys(sim)
        act = _greedy(policy, n_envs, device)
        gen = torch.Generator(device=device).manual_seed(seed)

        solved = torch.zeros(n_envs, dtype=torch.bool, device=device)
        with torch.no_grad():
            first = {k: v.float().mean().item() for k, v in sim.scenario.info().items()
                     if torch.is_tensor(v)}
            last = dict(first)
            for _ in range(steps):
                if who == "policy":
                    a = act(sim.scenario.observations())
                else:
                    a = torch.empty(
                        n_envs, n_agents, sim.act_dim, device=device
                    ).uniform_(-1, 1, generator=gen)
                _, _, term, _, info = sim.step(a)
                # Termination semantics: the episode would end at the first step the
                # scenario reports success, so count envs that ever get there.
                solved |= term
                last = {k: v.float().mean().item() for k, v in info.items()
                        if torch.is_tensor(v)}
        out[who] = {"keys": keys, "first": first, "last": last,
                    "solved": solved.float().mean().item()}
    return out


def plot_curve(path: str) -> None:
    """ASCII learning curves from the trainer's metrics.csv (no plotting deps)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"{path} is empty")
        return
    skip = {"iter", "frames", "difficulty"}
    cols = [c for c in rows[0] if c not in skip]
    for col in cols:
        try:
            vals = [float(r[col]) for r in rows]
        except (TypeError, ValueError):
            continue
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        # downsample to 60 columns, 9 rows of block glyphs
        step = max(1, len(vals) // 60)
        pts = [sum(vals[i : i + step]) / len(vals[i : i + step]) for i in range(0, len(vals), step)]
        bars = "".join(" ▁▂▃▄▅▆▇█"[int((v - lo) / span * 8)] for v in pts)
        print(f"  {col:26s} [{lo:+.4f} .. {hi:+.4f}]\n    {bars}")


def main() -> None:
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", help="policy_*.pt from marl_train.py")
    parser.add_argument("--scenario", choices=sorted(TASKS), help="default: from the checkpoint")
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--n-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, help="default: the task's episode length")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--video", help="write an mp4/webm of one env's rollout")
    parser.add_argument("--window", action="store_true", help="live pygame window")
    parser.add_argument("--render-steps", type=int, default=1200, help="total sim steps to render")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed vs real time")
    parser.add_argument("--fps", type=int, default=60, help="frame rate of the window/video")
    parser.add_argument("--curve", help="metrics.csv to plot as ASCII curves")
    args = parser.parse_args()

    if args.curve:
        print(f"learning curves from {args.curve}")
        plot_curve(args.curve)
        print()

    if not args.checkpoint:
        return

    policy, ckpt = _load(args.checkpoint, args.device)
    name = args.scenario or ckpt.get("scenario")
    if name is None:
        parser.error("checkpoint has no 'scenario' field; pass --scenario explicitly")
    task = TASKS[name]
    n_agents = ckpt["n_agents"]
    steps = args.steps or task.max_steps

    res = score(policy, name, task, n_agents, args.n_envs, steps, args.device, args.seed)
    print(f"{args.checkpoint}: {name}, {steps} steps x {args.n_envs} envs (seed {args.seed})")
    for who, m in res.items():
        print(f"  {who:7s} solved {m['solved']:.3f}")
        for k in m["keys"]:
            a, b = m["first"].get(k, float("nan")), m["last"].get(k, float("nan"))
            print(f"    {k:24s} {a:+.4f} -> {b:+.4f} ({b - a:+.4f})")

    if args.video or args.window:
        # auto_reset + max_steps makes the episode respawn in place when the task is solved or
        # the time limit hits, so a long rollout plays as a continuous sequence of episodes
        # instead of one clip that freezes at the end.
        scen = build_env(
            name, task, n_agents=n_agents, n_envs=1, device=args.device, seed=args.seed
        ).scenario
        env = Environment(
            scen, n_envs=1, device=args.device, dt=task.dt, substeps=task.substeps,
            seed=args.seed, max_steps=task.max_steps, auto_reset=True,
        )
        env.reset(seed=args.seed)
        act = _greedy(policy, 1, args.device)
        n = args.render_steps
        step_rate = args.speed / task.dt  # sim steps per second of wall clock
        print(
            f"  rendering {n} steps at {args.speed:g}x real time ({step_rate:.0f} steps/s), "
            f"respawning every {task.max_steps}"
        )
        if args.video:
            from swarp.render.video import save_video

            # save_video steps once per frame, so the file's fps *is* its playback rate.
            video_fps = max(1, round(step_rate))
            path = save_video(env, args.video, action_fn=act, n_steps=n, fps=video_fps)
            print(f"  wrote {path} ({n / video_fps:.0f}s at {video_fps} fps)")
        if args.window:
            # Hand the loop to the Viewer rather than stepping ourselves: it owns the
            # interactive controls and its own fps clock. A caller-driven loop steps
            # unconditionally, so pause would have nothing to act on.
            from swarp.render.viewer import Viewer

            env.reset(seed=args.seed)
            Viewer(env, fps=args.fps, steps_per_frame=step_rate / args.fps).run(
                action_fn=act, max_steps=n, close_when_done=True
            )


if __name__ == "__main__":
    main()
