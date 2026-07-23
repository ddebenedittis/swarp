"""Ablation benchmark for the hot-path performance overhaul.

Steps the ``NavigationScenario`` hot path under ``torch.no_grad()`` (like
``throughput.py``) across a cumulative stack of optimizations, one feature added
per row::

    baseline -> +eager-trims -> +nbr-dedupe -> +slim-2d -> +fused-obs-rew -> +cuda-graph

Each variant is defined by the feature toggles it enables on top of the previous
one. Toggles that the current checkout does not yet expose make the row report
``n/a`` (genuine feature-detection), so this script is useful from the very
first commit while the stages land incrementally.

Two guarantees are enforced before a row is timed:

* **Parity** — every variant replays the same seeded 5-step trajectory from
  ``reset(seed=0)`` and is compared against the ``baseline`` variant
  (``allclose`` on obs/reward at f32 rtol 1e-5, exact match on ``done``). A
  mismatch prints a loud ``PARITY FAIL`` and skips timing for that row.
* **Neighbor builds/step** — the ``NeighborGrid.build_count`` delta across the
  timed window is reported (``builds`` column), so the neighbor-dedupe stage can
  be seen to drop it from 2 to 1 per step at ``substeps=1``.

Run with::

    python -m wmas.benchmark.ablation --envs 256 4096 16384 --agents 4 64 256 \
        [--csv out.csv] [--grad] [--device cuda:0] [--steps 100]
"""

from __future__ import annotations

import argparse
import csv as csvmod
import inspect
import time
from dataclasses import dataclass

import torch

from wmas import Environment, NavigationScenario

DEF_ENVS = (256, 4096, 16384)
DEF_AGENTS = (4, 64, 256)


@dataclass(frozen=True)
class Variant:
    """A cumulative point in the optimization stack (a set of feature toggles)."""

    name: str
    eager_trims: bool = False
    neighbor_reuse: bool = False
    slim2d: bool = False
    fused: bool = False
    use_graph: bool = False

    def features(self) -> list[str]:
        """Toggle keys this variant turns on (used for availability checks)."""
        keys = []
        if self.eager_trims:
            keys.append("eager_trims")
        if self.neighbor_reuse:
            keys.append("neighbor_reuse")
        if self.slim2d:
            keys.append("slim2d")
        if self.fused:
            keys.append("fused")
        if self.use_graph:
            keys.append("use_graph")
        return keys


VARIANTS: list[Variant] = [
    Variant("baseline"),
    Variant("+eager-trims", eager_trims=True),
    Variant("+nbr-dedupe", eager_trims=True, neighbor_reuse=True),
    Variant("+slim-2d", eager_trims=True, neighbor_reuse=True, slim2d=True),
    Variant("+fused-obs-rew", eager_trims=True, neighbor_reuse=True, slim2d=True, fused=True),
    Variant(
        "+cuda-graph",
        eager_trims=True,
        neighbor_reuse=True,
        slim2d=True,
        fused=True,
        use_graph=True,
    ),
]


# --------------------------------------------------------------- feature probing


def _scenario_supports(kw: str) -> bool:
    return kw in inspect.signature(NavigationScenario.__init__).parameters


def _env_supports(kw: str) -> bool:
    return kw in inspect.signature(Environment.__init__).parameters


def _stepper_supports(attr: str) -> bool:
    # Stepper-level flags added by later stages; probe the class (attrs are set
    # in __init__, so a class-level default or an instance check both work — we
    # build a throwaway 1x1 env lazily only if the class check is inconclusive).
    from wmas.core.stepper import Stepper

    return hasattr(Stepper, attr) or attr in _stepper_init_names()


def _stepper_init_names() -> set[str]:
    from wmas.core.stepper import Stepper

    return set(inspect.signature(Stepper.__init__).parameters)


def _feature_available(feature: str) -> bool:
    if feature == "eager_trims":
        return _scenario_supports("eager_trims")
    if feature == "neighbor_reuse":
        # WorldConfig field or a Stepper attribute; either is enough to toggle.
        from wmas.core.config import WorldConfig

        return "neighbor_reuse" in inspect.signature(WorldConfig.__init__).parameters or (
            _stepper_supports("neighbor_reuse")
        )
    if feature == "slim2d":
        return _stepper_supports("enable_slim2d")
    if feature == "fused":
        return _env_supports("fused")
    if feature == "use_graph":
        return _env_supports("use_graph")
    return False


def _variant_available(v: Variant) -> bool:
    return all(_feature_available(f) for f in v.features())


# --------------------------------------------------------------- env construction


def _build_env(v: Variant, n_envs: int, n_agents: int, device: str) -> Environment:
    """Construct an Environment for a variant, applying its feature toggles.

    Assumes :func:`_variant_available` already returned True for ``v``.
    """
    scen_kwargs: dict = {"n_agents": n_agents, "world_size": max(1.0, n_agents**0.5 / 4)}
    if v.eager_trims and _scenario_supports("eager_trims"):
        scen_kwargs["eager_trims"] = True
    elif _scenario_supports("eager_trims"):
        # Baseline / pre-eager variants explicitly disable the trims so the
        # un-optimized eager path is what gets measured.
        scen_kwargs["eager_trims"] = False
    scenario = NavigationScenario(**scen_kwargs)

    env_kwargs: dict = {
        "n_envs": n_envs,
        "device": device,
        "dt": 0.05,
        "substeps": 1,
        "seed": 0,
    }
    if _env_supports("fused"):
        env_kwargs["fused"] = v.fused
    if _env_supports("use_graph"):
        env_kwargs["use_graph"] = v.use_graph
    env = Environment(scenario, **env_kwargs)

    stepper = env.world.stepper
    if _stepper_supports("neighbor_reuse"):
        stepper.neighbor_reuse = v.neighbor_reuse
    if _stepper_supports("enable_slim2d"):
        stepper.enable_slim2d = v.slim2d
    return env


# ------------------------------------------------------------------- parity


def _make_action_seq(n_envs: int, n_agents: int, device: str, n_steps: int) -> list[torch.Tensor]:
    """A fixed seeded action sequence reused by every variant (identical inputs)."""
    gen = torch.Generator(device=device).manual_seed(12345)
    seq = []
    for _ in range(n_steps):
        a = torch.empty(n_envs, n_agents, 2, device=device)
        a.uniform_(-1.0, 1.0, generator=gen)
        seq.append(a)
    return seq


def _trajectory(env: Environment, action_seq: list[torch.Tensor]) -> dict:
    env.reset(seed=0)
    obs_l, rew_l, done_l = [], [], []
    with torch.no_grad():
        for a in action_seq:
            obs, rew, done, _ = env.step(a)
            obs_l.append(obs.detach().clone())
            rew_l.append(rew.detach().clone())
            done_l.append(done.detach().clone())
    return {"obs": obs_l, "reward": rew_l, "done": done_l}


def _parity_ok(ref: dict, cur: dict, rtol: float = 1e-5, atol: float = 1e-6) -> tuple[bool, str]:
    for key in ("obs", "reward"):
        for t, (r, c) in enumerate(zip(ref[key], cur[key], strict=True)):
            if r.shape != c.shape:
                return False, f"{key}[{t}] shape {tuple(r.shape)} != {tuple(c.shape)}"
            if not torch.allclose(r, c, rtol=rtol, atol=atol):
                md = (r - c).abs().max().item()
                return False, f"{key}[{t}] max|Δ|={md:.3e}"
    for t, (r, c) in enumerate(zip(ref["done"], cur["done"], strict=True)):
        if not torch.equal(r, c):
            return False, f"done[{t}] mismatch ({int((r != c).sum())} envs)"
    return True, ""


# ------------------------------------------------------------------- timing


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _grid_build_count(env: Environment) -> int | None:
    stepper = env.world.stepper
    if not getattr(stepper, "collisions", False):
        return None
    try:
        grid = stepper.grid(env.n_envs)
    except Exception:
        return None
    return getattr(grid, "build_count", None)


def _time_env(env: Environment, n_envs: int, n_agents: int, device: str, steps: int, warmup: int):
    gen = torch.Generator(device=device).manual_seed(0)
    actions = torch.empty(n_envs, n_agents, 2, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            actions.uniform_(-1.0, 1.0, generator=gen)
            env.step(actions)
        _sync(device)
        build0 = _grid_build_count(env)
        t0 = time.perf_counter()
        for _ in range(steps):
            actions.uniform_(-1.0, 1.0, generator=gen)
            env.step(actions)
        _sync(device)
        elapsed = time.perf_counter() - t0
        build1 = _grid_build_count(env)
    builds_per_step = None if build0 is None or build1 is None else (build1 - build0) / steps
    return {
        "ms_per_step": 1e3 * elapsed / steps,
        "env_steps_s": n_envs * steps / elapsed,
        "builds_per_step": builds_per_step,
    }


def bench_one(
    variant: Variant,
    n_envs: int,
    n_agents: int,
    device: str,
    steps: int,
    ref_traj: dict | None,
    action_seq: list[torch.Tensor],
    warmup: int = 10,
) -> dict:
    """Benchmark one variant at one (n_envs, n_agents); parity-checked vs baseline.

    Returns a row dict with ``status`` in {ok, n/a, parity-fail, OOM}.
    """
    row = {
        "variant": variant.name,
        "n_envs": n_envs,
        "n_agents": n_agents,
        "ms_per_step": None,
        "env_steps_s": None,
        "builds_per_step": None,
        "status": "ok",
    }
    if not _variant_available(variant):
        row["status"] = "n/a"
        return row
    try:
        env = _build_env(variant, n_envs, n_agents, device)
        # Parity first (cheap, catches correctness regressions before timing).
        cur_traj = _trajectory(env, action_seq)
        if ref_traj is not None:
            ok, why = _parity_ok(ref_traj, cur_traj)
            if not ok:
                row["status"] = "parity-fail"
                row["detail"] = why
                return row
        stats = _time_env(env, n_envs, n_agents, device, steps, warmup)
        row.update(stats)
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
        torch.cuda.empty_cache()
    return row


# --------------------------------------------------------------------- grad


def _bench_grad_one(
    n_envs: int, n_agents: int, device: str, use_ring: bool, T: int = 8, iters: int = 5
) -> dict:
    """Time a T-step BPTT rollout + backward, optionally reusing a GradRing."""
    from wmas.interop.autograd import rollout

    ring = None
    if use_ring:
        try:
            from wmas.interop.autograd import GradRing
        except ImportError:
            return {"status": "n/a"}
        if "ring" not in inspect.signature(rollout).parameters:
            return {"status": "n/a"}

    scenario = NavigationScenario(n_agents=n_agents, world_size=max(1.0, n_agents**0.5 / 4))
    env = Environment(scenario, n_envs=n_envs, device=device, dt=0.05, substeps=1, seed=0)
    env.reset(seed=0)
    stepper = env.world.stepper
    if use_ring:
        from wmas.interop.autograd import GradRing

        ring = GradRing(stepper, n_envs, env.world.act_dim, capacity=T)

    def run_once() -> None:
        state = env.world.state
        state = state._replace(
            pos=state.pos.detach().clone().requires_grad_(True),
            vel=state.vel.detach().clone().requires_grad_(True),
        )
        actions = torch.zeros(
            T, n_envs, n_agents, 2, device=device, dtype=env.dtype, requires_grad=True
        )
        with torch.enable_grad():
            if use_ring:
                ring.reset()
                final, _ = rollout(stepper, state, actions, ring=ring)
            else:
                final, _ = rollout(stepper, state, actions)
            loss = final.pos.pow(2).sum()
        loss.backward()

    try:
        for _ in range(2):  # warmup
            run_once()
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            run_once()
        _sync(device)
        elapsed = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"status": "OOM"}
    return {"status": "ok", "ms_per_rollout": 1e3 * elapsed / iters}


# ----------------------------------------------------------------------- CLI


def _fmt(row: dict) -> str:
    status = row["status"]
    if status != "ok":
        base = f"{row['variant']:>16} {row['n_envs']:>8} {row['n_agents']:>9} {status:>10}"
        if "detail" in row:
            base += f"  {row['detail']}"
        return base
    b = row["builds_per_step"]
    builds = "-" if b is None else f"{b:.2f}"
    return (
        f"{row['variant']:>16} {row['n_envs']:>8} {row['n_agents']:>9} "
        f"{row['ms_per_step']:>9.3f} {row['env_steps_s']:>14,.0f} {builds:>7}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--envs", type=int, nargs="+", default=list(DEF_ENVS))
    parser.add_argument("--agents", type=int, nargs="+", default=list(DEF_AGENTS))
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--grad", action="store_true", help="also run the BPTT grad section")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        print(f"device: {args.device} ({torch.cuda.get_device_name(args.device)})")
    else:
        print("device: cpu (throughput will be far below GPU numbers)")
    print(f"timed steps per config: {args.steps}\n")

    header = (
        f"{'variant':>16} {'n_envs':>8} {'n_agents':>9} "
        f"{'ms/step':>9} {'env-steps/s':>14} {'builds':>7}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    for n_envs in args.envs:
        for n_agents in args.agents:
            action_seq = _make_action_seq(n_envs, n_agents, args.device, n_steps=5)
            # Reference trajectory from the baseline variant for this config.
            ref_traj = None
            try:
                ref_env = _build_env(VARIANTS[0], n_envs, n_agents, args.device)
                ref_traj = _trajectory(ref_env, action_seq)
                del ref_env
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
            for v in VARIANTS:
                row = bench_one(v, n_envs, n_agents, args.device, args.steps, ref_traj, action_seq)
                rows.append(row)
                print(_fmt(row))
            print()

    if args.grad:
        print("\n== BPTT (T=8) rollout + backward ==")
        gh = f"{'ring':>6} {'n_envs':>8} {'n_agents':>9} {'ms/rollout':>12} {'status':>8}"
        print(gh)
        print("-" * len(gh))
        for n_envs in args.envs:
            for n_agents in args.agents:
                for use_ring in (False, True):
                    r = _bench_grad_one(n_envs, n_agents, args.device, use_ring)
                    ms = f"{r['ms_per_rollout']:.3f}" if r.get("ms_per_rollout") else "-"
                    print(f"{str(use_ring):>6} {n_envs:>8} {n_agents:>9} {ms:>12} {r['status']:>8}")
                    rows.append(
                        {
                            "variant": f"grad(ring={use_ring})",
                            "n_envs": n_envs,
                            "n_agents": n_agents,
                            "ms_per_step": r.get("ms_per_rollout"),
                            "env_steps_s": None,
                            "builds_per_step": None,
                            "status": r["status"],
                        }
                    )

    if args.csv:
        fieldnames = [
            "variant",
            "n_envs",
            "n_agents",
            "ms_per_step",
            "env_steps_s",
            "builds_per_step",
            "status",
        ]
        with open(args.csv, "w", newline="") as f:
            writer = csvmod.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
