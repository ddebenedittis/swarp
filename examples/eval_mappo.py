"""Evaluate / visualize a checkpoint saved by ``train_mappo.py``.

Three views of "performance", any combination:

* always -- an **episode solve rate** per arm, over ``--n-envs`` parallel envs, with a
  Wilson 95% interval, next to random / zero-action / scripted baselines on the *same*
  seed, so the number means something.
* ``--video out.mp4`` -- render one env's rollout to a file (needs the ``viz`` extra).
* ``--window`` -- the same rollout in a live pygame window.

An env counts as solved if its terminal condition fires at **any** point during one
un-reset episode, because that is where the episode would have ended in training. For
Push-T this is *literally the same tensor* as the hand-written
``(dist < goal_tolerance) & (angle < angle_tolerance)`` the previous version recomputed, so
the generic predicate reproduces the published number rather than replacing it with a
similar-looking one.

A scenario that does not override ``done()`` (flocking, sampling) has an identically-zero
solve rate; rather than print ``solved 0.000`` -- which reads as policy failure when the
truth is "this task has no success criterion" -- the run refuses and asks for
``--success-key``/``--success-cmp``/``--success-threshold`` naming an ``info()`` key to
threshold instead.

The **scripted** arm is what turns the number into a claim: a ten-line P controller that
drives at the goal. Where the task is solvable greedily it nearly solves it (a useful
honesty check on the whole pipeline); where the solution needs yielding or enclosure it
should fail, and the paired ``policy - scripted`` difference is the evidence that
something non-greedy was learnt. Being paired -- both arms see bit-identical initial
conditions -- that difference gets a McNemar interval, which is strictly tighter than
differencing two independent ones.

The reported interval covers sampling of initial conditions, **not** training-seed
variance: a single training run's solve rate is one draw, and MAPPO's across-seed spread
on hard cooperative tasks is routinely +/- 0.15.

The rendered rollout runs ``--render-steps`` steps and respawns every ``--episode-steps``,
so it plays as a continuous run of episodes rather than one clip that freezes once the task
is done. Playback is real time by default -- ``--speed 0.25`` crawls, and in the window
up/down change it live. ``--fps`` sets how often the window redraws, independently of the
playback rate. ``--window`` hands the loop to :class:`~swarp.render.viewer.Viewer`, so the
usual controls work: space pauses, ``.`` steps once while paused, ``r`` resets, ``?`` lists
the rest.

Also plots a trainer ``metrics.csv``::

    python examples/eval_mappo.py --curve runs/navigation/metrics.csv

Run with::

    python examples/eval_mappo.py runs/navigation/navigation_final.pt --n-envs 2048
    python examples/eval_mappo.py runs/pusht/pusht_final.pt --window --speed 0.25
"""

from __future__ import annotations

import argparse
import sys

import torch
from mappo import (
    BASELINES,
    SPECS,
    SuccessSpec,
    build_policy,
    greedy_actions,
    mcnemar_interval,
    plot_curve,
    score,
)

import swarp


def _load(path: str, device: str):
    """Load a checkpoint into a byte-identical module.

    New keys are read with ``.get`` so a checkpoint written by the older Push-T script --
    which stored only ``policy``/``critic``/``obs_dim``/``act_dim``/``n_agents``/
    ``num_cells`` -- still loads, with ``--scenario`` supplying what it lacks.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    policy = build_policy(
        ckpt["obs_dim"], ckpt["act_dim"], ckpt["n_agents"], device,
        num_cells=ckpt.get("num_cells", 128),
    )
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, ckpt


def main() -> int:
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", nargs="?", help="{scenario}_*.pt from train_mappo.py")
    p.add_argument("--scenario", choices=sorted(SPECS),
                   help="only needed for a checkpoint that does not record it")
    p.add_argument("--device", default=default_device)
    p.add_argument("--n-envs", type=int, default=2048,
                   help="the sample size behind the CI; 2048 gives about +/-0.02 at p=0.7")
    p.add_argument("--steps", type=int, help="episode length; default = the spec's")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--substeps", type=int, help="keep equal to training")
    p.add_argument("--arms", default="policy,random,zero,scripted")
    p.add_argument("--target", type=float, help="exit nonzero below this solve rate")
    p.add_argument("--success-key", help="info() key to threshold, for a done()-less scenario")
    p.add_argument("--success-cmp", choices=("lt", "gt"), default="lt")
    p.add_argument("--success-threshold", type=float)
    p.add_argument("--video", help="write an mp4/webm of one env's rollout")
    p.add_argument("--window", action="store_true", help="live pygame window")
    p.add_argument("--render-steps", type=int, default=1200)
    p.add_argument("--episode-steps", type=int, help="steps before respawn; default = --steps")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed vs real time")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--curve", help="metrics.csv to plot as ASCII curves")
    args = p.parse_args()

    if args.curve:
        print(f"learning curves from {args.curve}")
        plot_curve(args.curve)
        print()

    if not args.checkpoint:
        return 0

    policy, ckpt = _load(args.checkpoint, args.device)
    scenario = args.scenario or ckpt.get("scenario")
    if scenario is None:
        p.error(
            f"{args.checkpoint} does not record its scenario (it predates that key); "
            f"pass --scenario"
        )
    spec = SPECS[scenario]
    n_agents = ckpt["n_agents"]
    steps = args.steps or ckpt.get("max_steps") or spec.max_steps
    substeps = args.substeps or ckpt.get("substeps") or spec.substeps
    dt = ckpt.get("dt") or spec.dt
    target = args.target if args.target is not None else spec.target_solve_rate

    success = None
    if args.success_key:
        if args.success_threshold is None:
            p.error("--success-key needs --success-threshold")
        success = SuccessSpec(args.success_key, args.success_cmp, args.success_threshold)

    res = score(
        scenario,
        policy=policy,
        n_agents=n_agents,
        n_envs=args.n_envs,
        steps=steps,
        device=args.device,
        seed=args.seed,
        dt=dt,
        substeps=substeps,
        scen_kwargs=spec.scen_kwargs,
        metrics=spec.metrics,
        baseline=BASELINES.get(scenario),
        arms=tuple(a.strip() for a in args.arms.split(",") if a.strip()),
        success=success,
    )

    print(
        f"{args.checkpoint}: {scenario}, {n_agents} agents, {steps} steps x "
        f"{args.n_envs} envs (seed {args.seed})"
    )
    for name, m in res.items():
        med = "n/a" if m.solve_step_median is None else f"{m.solve_step_median:.0f}"
        print(
            f"  {name:9s} solved {m.solve_rate:.3f} [{m.ci[0]:.3f}, {m.ci[1]:.3f}]"
            f"   median solve step {med}"
        )
    if "scripted" not in res:
        print(
            f"  (no scripted baseline registered for {scenario!r}; the solve rate has no "
            f"difficulty denominator without one)"
        )
    for other in ("scripted", "random"):
        if "policy" in res and other in res:
            diff, (dlo, dhi) = mcnemar_interval(
                res["policy"].solved_mask, res[other].solved_mask
            )
            print(f"  paired policy - {other}: {diff:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
    print(
        "  the interval covers initial-condition sampling only, not training-seed "
        "variance (single seed)"
    )

    if "policy" in res:
        met = res["policy"].solve_rate >= target
        print(f"  target {target:.2f}: {'MET' if met else 'NOT met'}")

    if args.video or args.window:
        # auto_reset + max_steps makes the episode respawn in place when the task is
        # solved or the time limit hits, so a long rollout plays as a continuous sequence
        # of episodes instead of one clip that freezes at the end.
        episode_steps = args.episode_steps or steps
        env = swarp.make(
            scenario, n_envs=1, device=args.device, dt=dt, substeps=substeps,
            seed=args.seed, max_steps=episode_steps, auto_reset=True,
            n_agents=n_agents, **dict(spec.scen_kwargs),
        )
        env.reset(seed=args.seed)
        lo, hi = env.action_bounds
        act = greedy_actions(policy, 1, args.device, lo, hi)
        n = args.render_steps
        step_rate = args.speed / dt  # sim steps per second of wall clock
        print(
            f"  rendering {n} steps at {args.speed:g}x real time ({step_rate:.0f} steps/s), "
            f"respawning every {episode_steps}"
        )
        if args.video:
            from swarp.render.video import save_video

            # save_video steps once per frame, so the file's fps *is* its playback rate.
            video_fps = max(1, round(step_rate))
            path = save_video(env, args.video, action_fn=act, n_steps=n, fps=video_fps)
            print(f"  wrote {path} ({n / video_fps:.0f}s at {video_fps} fps)")
        if args.window:
            # Hand the loop to the Viewer rather than stepping ourselves: it owns the
            # interactive controls (space to pause, "." to step once while paused, "r" to
            # reset, up/down for speed, overlay toggles, dragging) and its own fps clock.
            # A caller-driven loop steps unconditionally, so pause has nothing to act on.
            from swarp.render.viewer import Viewer

            env.reset(seed=args.seed)
            Viewer(env, fps=args.fps, steps_per_frame=step_rate / args.fps).run(
                action_fn=act, max_steps=n, close_when_done=True
            )

    if "policy" in res and res["policy"].solve_rate < target:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
