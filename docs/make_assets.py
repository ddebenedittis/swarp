"""Regenerate the documentation screenshots and the demo clip.

    uv pip install -e '.[viz]'
    python docs/make_assets.py

Everything renders headlessly on the CPU (a few seconds per asset) and lands in
``docs/_static/``. The outputs are committed, so this only needs re-running when the
renderer or the scenarios change appearance.

The navigation shots reuse :mod:`swarp.render.demo`'s scenarios and controller, which
already attach the lidar and communication-link ``render_extras`` the viewer draws.

**Not everything in ``_static`` comes from here.** ``giveway.webm``, ``caging.webm`` and
``shepherding.webm`` show *trained* policies, so they need a checkpoint and the ``torchrl``
extra — neither of which this script should depend on. Regenerate them with::

    python examples/marl_eval.py runs/caging/policy_best.pt \
        --video docs/_static/caging.webm --render-steps 600

The matching ``.png`` stills for those three scenarios *are* generated here, and are what the
README and link previews use.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must precede the first pygame import, so no display is ever required.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402

from swarp import (  # noqa: E402
    CagingScenario,
    Environment,
    FlockingScenario,
    GiveWayScenario,
    PushTScenario,
    ShepherdingScenario,
)
from swarp.render.demo import (  # noqa: E402
    MixedVisualizationScenario,
    VisualizationScenario,
    goal_seeking_policy,
)
from swarp.render.style import Style  # noqa: E402
from swarp.render.video import iter_rollout_frames  # noqa: E402

DEVICE = "cpu"
SEED = 3

NAV_OVERLAYS = {"bounds", "obstacles", "goals", "agents", "heading", "lidar", "comm_lines"}


def _drive(env: Environment, steps: int, *, policy=None) -> None:
    """Advance the env so the shot shows a task in progress rather than a fresh reset."""
    torch.manual_seed(SEED)
    obs = env.scenario.observations()
    zero = torch.zeros(env.n_envs, env.n_agents, env.act_dim, dtype=env.dtype)
    with torch.no_grad():
        for _ in range(steps):
            obs, *_ = env.step(zero if policy is None else policy(obs))


def _shoot(env: Environment, path: Path, size: tuple[int, int], **kwargs) -> None:
    frame = env.render(mode="rgb_array", size=size, **kwargs)
    imageio.imwrite(path, frame)
    print(f"wrote {path}  {frame.shape[1]}x{frame.shape[0]}")


def hero(out: Path) -> None:
    env = Environment(
        VisualizationScenario(n_agents=8, n_obstacles=3, lidar_rays=16, lidar_range=0.3),
        n_envs=4,
        device=DEVICE,
        dt=0.05,
        seed=SEED,
    )
    env.reset()
    _drive(env, 18, policy=goal_seeking_policy(env))
    _shoot(env, out / "hero.png", (1100, 700), overlays=NAV_OVERLAYS | {"velocity"})


def mosaic(out: Path) -> None:
    env = Environment(
        VisualizationScenario(n_agents=6, n_obstacles=2, lidar_rays=12, lidar_range=0.3),
        n_envs=9,
        device=DEVICE,
        dt=0.05,
        seed=SEED,
    )
    env.reset()
    _drive(env, 30, policy=goal_seeking_policy(env))
    _shoot(env, out / "mosaic.png", (1100, 760), mosaic=True)


def mixed(out: Path) -> None:
    env = Environment(
        MixedVisualizationScenario(n_agents=6, n_obstacles=2, lidar_rays=12, lidar_range=0.3),
        n_envs=4,
        device=DEVICE,
        dt=0.05,
        seed=SEED,
    )
    env.reset()
    _drive(env, 50, policy=goal_seeking_policy(env))
    _shoot(
        env,
        out / "mixed.png",
        (960, 720),
        overlays=NAV_OVERLAYS | {"velocity"},
        style=Style.light(color_mode="model"),
    )


def flocking(out: Path) -> None:
    env = Environment(
        FlockingScenario(n_agents=24, neighbor_radius=0.45),
        n_envs=4,
        device=DEVICE,
        dt=0.05,
        seed=SEED,
    )
    env.reset()
    _drive(env, 40)
    _shoot(
        env,
        out / "flocking.png",
        (960, 720),
        overlays={"bounds", "agents", "heading", "velocity", "neighbor_graph"},
    )


def pusht(out: Path) -> None:
    env = Environment(
        PushTScenario(n_agents=4),
        n_envs=4,
        device=DEVICE,
        dt=0.05,
        substeps=16,
        seed=SEED,
    )
    env.reset()
    _drive(env, 40)
    _shoot(
        env,
        out / "pusht.png",
        (860, 760),
        overlays={"bounds", "obstacles", "goal_pose", "agents", "heading", "ids"},
    )


def clip(out: Path, steps: int) -> None:
    """Encode the rollout clip embedded on the landing and visualization pages.

    Lidar is deliberately off here: 8 agents x 16 rays in motion reads as noise at video
    size (the still shots keep it, where the eye can settle on one fan). Fading trails
    carry the motion instead, and the loop ends while the agents are still converging —
    a longer clip just holds on a parked fleet.

    ``save_video`` would do this in one call, but it does not expose an encoder quality
    knob, so the frames come from the same generator ``save_video`` uses and the writer is
    opened here with an explicit CRF.
    """
    env = Environment(
        VisualizationScenario(n_agents=10, n_obstacles=3, lidar_rays=16, lidar_range=0.3),
        n_envs=4,
        device=DEVICE,
        dt=0.05,
        seed=SEED,
    )
    path = out / "demo.webm"
    writer = imageio.get_writer(
        str(path),
        fps=25,
        codec="libvpx-vp9",
        output_params=[
            # Constant-quality VP9; the default CRF leaves the file several MB.
            *("-b:v", "0", "-crf", "30"),
            *("-row-mt", "1", "-deadline", "good", "-cpu-used", "2"),
        ],
    )
    try:
        for frame in iter_rollout_frames(
            env,
            goal_seeking_policy(env),
            steps,
            size=(800, 448),
            overlays=(NAV_OVERLAYS - {"lidar"}) | {"trajectories"},
            style=Style.light(trajectory_mode="fade", trajectory_len=45),
        ):
            writer.append_data(frame)
    finally:
        writer.close()
    print(f"wrote {path}  ({steps} frames)")


def _seek(env: Environment, target_fn, *, brake: float | None = None):
    """Drive every agent toward ``target_fn(env)`` (a ``[E, A, 2]`` point).

    The zero-action ``_drive`` leaves these three scenarios sitting at their spawn ring,
    which shows the arena but not the task. A crude chase is enough to get agents into the
    corridor / around the disc / behind the flock for the shot.

    ``brake`` is the distance within which the command tapers to zero. Without it an agent
    arrives at full speed and keeps pushing, which for caging shoves the disc into a corner
    instead of ringing it. ``None`` keeps full speed all the way in, which is what give-way
    wants — the robots are *supposed* to pile into the junction.
    """
    d = target_fn(env) - env.world.state.pos
    n = d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    if brake is None:
        return d / n
    return d / n * (n / brake).clamp(max=1.0)


def giveway(out: Path) -> None:
    """Four robots converging on the junction — the moment one of them has to yield."""
    env = Environment(
        GiveWayScenario(n_agents=4),
        n_envs=4,
        device=DEVICE,
        dt=0.05,
        substeps=16,
        seed=SEED,
    )
    env.reset()
    _drive(env, 26, policy=lambda _obs: _seek(env, lambda e: e.world.goals))
    _shoot(
        env,
        out / "giveway.png",
        (860, 860),
        overlays={"bounds", "obstacles", "goals", "agents", "heading", "ids"},
    )


def caging(out: Path) -> None:
    """Agents closing a ring around the drifting disc."""
    scen = CagingScenario(n_agents=6)
    # 8 envs so there is a choice of disc spawn; env 2 is the one whose disc lands nearest
    # the arena centre at SEED, which keeps the whole ring inside the frame.
    env = Environment(scen, n_envs=8, device=DEVICE, dt=0.05, seed=SEED)
    env.reset()

    def ring(e: Environment) -> torch.Tensor:
        # Each agent's slot on the cage circle, so the shot shows a closed cage rather
        # than a clump: agent i sits at bearing 2*pi*i/n around the disc.
        n = e.n_agents
        ang = torch.arange(n, dtype=e.dtype) * (2.0 * torch.pi / n)
        offs = torch.stack([ang.cos(), ang.sin()], dim=-1) * scen.cage_radius
        return scen.disc_pos + offs  # disc_pos is [E, 1, 2], so this broadcasts to [E, A, 2]

    # 18 steps is the sweet spot: the ring has closed enough to read as a cage, and the
    # disc has not yet coasted to a wall. Its escape drift only cancels once the ring is
    # shut, so a longer chase just walks the disc out of frame.
    _drive(env, 18, policy=lambda _obs: _seek(env, ring, brake=0.12))
    _shoot(
        env,
        out / "caging.png",
        (860, 860),
        env_index=2,
        overlays={"bounds", "obstacles", "agents", "heading", "ids"},
    )


def shepherding(out: Path) -> None:
    """Shepherds working a flock of fleeing sheep toward the pen."""
    scen = ShepherdingScenario(n_agents=3)
    # 8 envs for a choice of layout; env 6 is the one at SEED whose pen and flock both
    # sit near the arena centre, so the shot is not a huddle in one corner.
    env = Environment(scen, n_envs=8, device=DEVICE, dt=0.05, seed=SEED)
    env.reset()

    def behind(e: Environment) -> torch.Tensor:
        # Aim for the far side of the nearest sheep, which is where a shepherd has to be
        # for its flee force to push the sheep penward.
        pos = e.world.state.pos                                   # [E, A, 2]
        rel = scen.sheep_pos.unsqueeze(1) - pos.unsqueeze(2)      # [E, A, K, 2]
        near = rel.norm(dim=-1).argmin(dim=-1)                    # [E, A]
        tgt = torch.gather(
            scen.sheep_pos.unsqueeze(1).expand(-1, e.n_agents, -1, -1),
            2,
            near[..., None, None].expand(-1, -1, 1, 2),
        ).squeeze(2)
        away = tgt - scen.pen_pos.unsqueeze(1)  # pen is per-env: [E, 1, 2]
        return tgt + 0.18 * away / away.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    _drive(env, 32, policy=lambda _obs: _seek(env, behind, brake=0.15))
    _shoot(
        env,
        out / "shepherding.png",
        (860, 860),
        env_index=6,
        overlays={"bounds", "obstacles", "goals", "agents", "heading", "ids"},
    )


def _encode(env, action_fn, path: Path, steps: int, overlays, size=(720, 720)) -> None:
    """Constant-quality VP9 of a scripted rollout (same writer settings as ``clip``)."""
    writer = imageio.get_writer(
        str(path),
        fps=20,
        codec="libvpx-vp9",
        output_params=[
            *("-b:v", "0", "-crf", "32"),
            *("-row-mt", "1", "-deadline", "good", "-cpu-used", "2"),
        ],
    )
    try:
        for frame in iter_rollout_frames(
            env, action_fn, steps, size=size, overlays=overlays,
            style=Style.light(trajectory_mode="fade", trajectory_len=30),
        ):
            writer.append_data(frame)
    finally:
        writer.close()
    print(f"wrote {path}  ({steps} frames)")


def clips(out: Path, steps: int) -> None:
    """The three scenario clips embedded on the Scenarios page.

    Deliberately **scripted**, not a trained policy. A checkpoint would tie a committed
    docs asset to a training run and to the ``torchrl`` extra, and it goes stale the moment
    an observation width changes -- which is exactly what happened when ``shepherding``'s
    default flock grew and its ``obs_dim`` went 14 -> 26, leaving the previous clip
    unregenerable. These use the same crude controllers as the stills above, so
    ``python docs/make_assets.py`` reproduces them from the shipped code alone.

    For a clip of an actually-trained policy, use the eval script instead::

        python examples/marl_eval.py runs/caging/policy_best.pt --video caging.webm
    """
    gw = Environment(
        GiveWayScenario(n_agents=4), n_envs=4, device=DEVICE, dt=0.05, substeps=16, seed=SEED
    )
    gw.reset()
    _encode(
        gw, lambda _o: _seek(gw, lambda e: e.world.goals), out / "giveway.webm", steps,
        {"bounds", "obstacles", "goals", "agents", "heading", "ids", "trajectories"},
    )

    cs = CagingScenario(n_agents=6)
    cg = Environment(cs, n_envs=8, device=DEVICE, dt=0.05, seed=SEED)
    cg.reset()

    def ring(e: Environment) -> torch.Tensor:
        n = e.n_agents
        ang = torch.arange(n, dtype=e.dtype) * (2.0 * torch.pi / n)
        offs = torch.stack([ang.cos(), ang.sin()], dim=-1) * cs.cage_radius
        return cs.disc_pos + offs

    _encode(
        cg, lambda _o: _seek(cg, ring, brake=0.12), out / "caging.webm", steps,
        {"bounds", "obstacles", "agents", "heading", "ids", "trajectories"},
    )

    ss = ShepherdingScenario(n_agents=3)
    sh = Environment(ss, n_envs=8, device=DEVICE, dt=0.05, seed=SEED)
    sh.reset()

    def behind(e: Environment) -> torch.Tensor:
        pos = e.world.state.pos
        rel = ss.sheep_pos.unsqueeze(1) - pos.unsqueeze(2)
        near = rel.norm(dim=-1).argmin(dim=-1)
        tgt = torch.gather(
            ss.sheep_pos.unsqueeze(1).expand(-1, e.n_agents, -1, -1),
            2, near[..., None, None].expand(-1, -1, 1, 2),
        ).squeeze(2)
        away = tgt - ss.pen_pos.unsqueeze(1)
        return tgt + 0.18 * away / away.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    _encode(
        sh, lambda _o: _seek(sh, behind, brake=0.15), out / "shepherding.webm", steps,
        {"bounds", "obstacles", "goals", "agents", "heading", "ids", "trajectories"},
    )


ASSETS = {
    "hero": hero,
    "mosaic": mosaic,
    "mixed": mixed,
    "flocking": flocking,
    "pusht": pusht,
    "giveway": giveway,
    "caging": caging,
    "shepherding": shepherding,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the swarp docs assets.")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "_static")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[*ASSETS, "clip", "clips"],
        help="subset to regenerate (default: all)",
    )
    parser.add_argument("--clip-steps", type=int, default=100, help="frames in demo.webm")
    parser.add_argument("--scenario-clip-steps", type=int, default=220,
                        help="frames in the three scenario clips")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else {*ASSETS, "clip", "clips"}
    for name, fn in ASSETS.items():
        if name in wanted:
            fn(args.out)
    if "clip" in wanted:
        clip(args.out, args.clip_steps)
    if "clips" in wanted:
        clips(args.out, args.scenario_clip_steps)


if __name__ == "__main__":
    main()
