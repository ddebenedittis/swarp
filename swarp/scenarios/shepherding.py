"""Shepherding: shepherds herd fleeing sheep into a pen.

The one **reactive** environment in the package. Every other scenario's world is either
inert (navigation's goals, formation's targets) or purely contact-driven (transport's
package, push-t's T): it only ever does what the agents' contacts make it do. Here the
sheep run their own scripted policy, so the world *pushes back* — a shepherd that charges
straight at the flock scatters it, and (now that the reward carries a gather potential on
the flock's own radius, see below) the reward penalizes exactly that for the first time.
Learning the task means learning to approach obliquely and hold a line.

Why the task needed redesigning
--------------------------------
The shipped task was **geometrically infeasible**. Relaxing 5 free sheep under this
scenario's own force law converges to a non-symmetric attractor with radii
``[0.053, 0.104, 0.107, 0.150, 0.151]`` at ``sep_radius=0.15`` and
``[0.071, 0.138, 0.141, 0.196, 0.198]`` at ``sep_radius=0.2``. At the shipped
``pen_radius=0.2`` the free flock's own **max radius is 0.198**, so ``all_penned``
demanded the flock centroid land within 0.003 of the pen centre in a world of half-extent
1.0 — a measured ``terminated`` rate of 0.0016. Shepherds made it worse: three shepherds
at radius ``Rs`` about the flock relax it to ``Rs=0.25 -> 0.350, 0.30 -> 0.268,
0.35 -> 0.195, 0.40 -> 0.154, 0.50 -> 0.152`` (at ``sep 0.15``), because the capped flee
force (1.0) beats cohesion (``0.5*R ~= 0.08``) by 10x.

Measured scripted-Strombom solve rates (256 envs, 5 sheep, 3 shepherds, hold 5):

::

    config                                      scripted  random  do-nothing
    shipped: pen 0.2, sep 0.2, 150, ring spawn      0.00    0.00        0.00
    pen 0.3 + sep 0.15, 200 steps, ring spawn       0.81    0.24        0.23
    pen 0.3 + sep 0.15, 200 steps, CLUSTER spawn    0.84    0.00        0.00
      the same, at 150 steps                        0.61    0.00        0.00
      the same, at 250 steps                        0.91    0.00        0.00

The ring spawn was itself degenerate: drawing each sheep at an independent bearing about
the pen puts the flock centroid *on the pen* by symmetry (``E|c - pen| ~ rbar/sqrt(K)``),
so at ``pen 0.3`` doing nothing already solved 23% of episodes. Shepherding is not a
gathering task; the spawn is now a **flock cluster** at a distance (see
:meth:`ShepherdingScenario._set_spawn_geometry`) rather than a ring, ``pen_radius`` moved
to 0.3 and ``sep_radius`` to 0.15, and the reward gained the collect-side potential
described below.

Ablation: removing observation slots 16-19 (the two Strombom hint points, ``drive_pt``
and ``collect_pt``) recovers the un-hinted task — worth knowing what the hints cost before
assuming they are free.

The sheep are **not** agents. There is no action-override hook in this engine (``world.
action`` is written by ``Environment.step`` after the only pre-physics hook fires), so a
sheep cannot be an agent whose action a scenario intercepts. They are instead modelled the
way :mod:`swarp.scenarios.transport` models its packages, and every step:

  1. the sheep are installed as circular obstacles, so the Warp agent step pushes
     shepherds *off* them (the existing soft agent-obstacle contact);
  2. after the step, the force on each sheep is summed — the reaction of that same
     spring-damper contact (Newton's third law), plus the two terms the engine knows
     nothing about: **flee** repulsion from every shepherd inside ``flee_radius``
     (summed, then magnitude-capped at ``flee_gain``, so a sheep can never outrun the
     shepherds herding it), and **separation** from every other sheep inside
     ``sep_radius`` — and the sheep is integrated in torch (semi-implicit Euler +
     damping, then a ``max_sheep_speed`` cap that keeps the flock slower than the
     shepherds herding it);
  3. the updated sheep positions are installed for the next step.

Deliberately *not* ``ObstacleKind.MOVABLE``: the engine integrates those from contact
reaction only, with nowhere to add the flee force; the body integrator is
``record_tape=False`` so it vanishes on a taped step; and an install carrying
``kind=MOVABLE`` reads ``any_movable`` back to the host, which is not capture-safe. The
transport pattern above avoids all three. A sheep tracks **position only** — a circular
body under frictionless normal contact has zero lever arm
(:mod:`swarp.core.bodies`), so there is no rotation to carry and none is allocated.

Coupling is staggered by one step (shepherds see last step's sheep positions). The sheep
integration is plain torch, so gradients flow sheep->shepherd->action across a rollout
(BPTT); the intra-step shepherd-avoids-sheep force is not taped (obstacles are constants
inside a Warp step), the same documented limitation transport carries.

Spawn difficulty is a curriculum knob along two independent axes:
:meth:`ShepherdingScenario.set_flock_distance` (how far the flock centre is driven from
the pen — measured the dominant axis) and :meth:`ShepherdingScenario.set_flock_spread`
(how dispersed the flock starts — measured nearly irrelevant on its own). Annealing both
back to ``1.0`` restores the constructor's distribution bit-for-bit.

The three steps above describe the **torch reference path** — the parity oracle, and the
only path that carries gradients. The no-grad hot path does the same physics in
``shepherding_force_kernel`` + ``shepherding_body_kernel``
(:mod:`swarp.scenarios.shepherding_kernels`) instead, entirely on-device inside the fused
whole-step hook. The two are tested against each other in
``tests/scenarios/test_shepherding_fused.py``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from swarp._overloads import concrete
from swarp.core.cached_launch import CachedLaunch, ptr_key
from swarp.core.config import Obstacles, WorldConfig
from swarp.core.state import VEC2
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.interop.autograd import torch_stream_scope
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario
from swarp.scenarios.shepherding_kernels import (
    shepherding_body_kernel,
    shepherding_force_kernel,
    shepherding_obs_kernel,
    shepherding_reset_kernel,
    shepherding_reward_kernel,
)

_TWO_PI = 2.0 * math.pi


class ShepherdingScenario(FusedScenario):
    """Drive ``n_sheep`` fleeing sheep into a disc-shaped pen with ``n_agents`` shepherds.

    ``n_agents`` counts **shepherds only**; the sheep are scenario-owned bodies, not
    agents, and ``n_sheep`` is their own knob.
    """

    # The fused and torch paths compute the same physics but reduce the shepherd and
    # sheep contributions in different orders, so they diverge at the ulp scale from the
    # first step. Transport lives with the default 1e-5/1e-6 because its only force is a
    # contact that damping pulls back together; here the *flee* term is active at a
    # distance, so a ulp of positional difference feeds straight back into a force the
    # sheep feels every step (and through the flock, into its neighbours) rather than
    # only when someone is touching it. Measured over the 20-step parity rollout the
    # worst field difference is ~1e-7, i.e. still comfortably inside the default; the
    # bound is loosened by one decade anyway because that number *grows with the
    # horizon* here rather than settling, and a 10x margin on an absolute 1e-6 is thin
    # enough that a longer rollout or a different GPU would trip it for no real reason.
    #
    # Loosened a further decade for the redesign: the observation now carries two
    # *normalizations* (``safe_unit`` in ``drive_pt``/``collect_pt``), and a relative
    # error there is ``|delta| / |v|``. ``dir_floor = 1e-3`` bounds the denominator, so a
    # ~1e-7 positional divergence near the floor can show up as a ``1e-7 / 1e-3 = 1e-4``
    # relative error in the normalized direction — exactly where the old ``rtol=1e-4``
    # sat, with no margin left. Moving to ``1e-3`` restores the same margin the old bound
    # had before these two terms existed.
    parity_rtol: float = 1e-3
    parity_atol: float = 1e-5

    def __init__(
        self,
        n_agents: int = 3,
        n_sheep: int = 5,
        agent_radius: float = 0.05,
        sheep_radius: float = 0.05,
        sheep_mass: float = 1.0,
        world_size: float = 1.0,
        max_speed: float = 1.0,
        pen_radius: float = 0.3,
        contact_k: float = 100.0,
        contact_c: float = 5.0,
        contact_margin: float = 0.01,
        flee_radius: float = 0.35,
        flee_gain: float = 1.0,
        sep_radius: float = 0.15,
        sep_gain: float = 1.0,
        cohesion_gain: float = 0.5,
        linear_damping: float = 2.0,
        max_sheep_speed: float | None = None,
        pos_shaping_factor: float = 10.0,
        pen_reward: float = 1.0,
        scatter_penalty: float = 0.0,
        gather_factor: float = 2.0,
        pen_level_reward: float = 0.0,
        done_reward: float = 10.0,
        side_factor: float = 0.01,
        crowd_factor: float = 0.3,
        crowd_radius: float = 0.25,
        intrude_penalty: float = 1.0,
        drive_factor: float = 0.8,
        dir_floor: float = 1e-3,
        pen_hold: int = 5,
        flock_dist_min: float = 0.5,
        flock_dist_max: float = 1.2,
        flock_spread: float = 0.35,
    ) -> None:
        self.n_agents = n_agents
        self.n_sheep = n_sheep
        self.agent_radius = agent_radius
        self.sheep_radius = sheep_radius
        self.sheep_mass = sheep_mass
        self.world_size = world_size
        self.max_speed = max_speed
        # pen_radius=0.3 (was 0.2): the free flock's own relaxed radius measures 0.198 at
        # sep_radius=0.15 (see the module docstring's §0 measurement), so a pen sized to
        # the old value demanded the centroid within 0.003 of the pen centre — infeasible
        # for anything but a scripted controller. 0.3 comfortably covers the settled
        # flock and is what the measured solve rates above are for.
        self.pen_radius = pen_radius
        # contact_k/contact_c ARE the engine's WorldConfig.collision_k/collision_c — the
        # same spring-damper law, named for the shepherd<->sheep contact that dominates
        # here. contact_margin is NOT the engine's collision_margin (derived from
        # agent_radius in make_world): it is this scenario's own shepherd<->sheep
        # activation gap, used by the torch reference path below.
        self.contact_k = contact_k
        self.contact_c = contact_c
        self.contact_margin = contact_margin
        # The flee law: every shepherd within flee_radius pushes the sheep directly away
        # with magnitude flee_gain * (1 - d / flee_radius) — full strength on top of the
        # sheep, tapering linearly to nothing at the radius, so there is no discontinuity
        # at the boundary for the parity paths to disagree about. Linear rather than
        # 1/d^2 on purpose: an inverse-square law is unbounded at contact, where the
        # spring is already applying a large force, and the sum of the two blows the
        # explicit integrator up at any usable dt.
        #
        # The **sum** over shepherds is then magnitude-capped at flee_gain (in both
        # paths), which is what makes the task solvable at all. A sustained force f gives
        # a terminal speed of f / (sheep_mass * linear_damping), so the cap fixes the
        # fastest a sheep can ever run at
        #
        #     flee_gain / (sheep_mass * linear_damping) = 1.0 / (1.0 * 2.0) = 0.5
        #
        # against a shepherd's max_speed of 1.0 — half, independent of how many
        # shepherds converge. Uncapped and at the old flee_gain=2.0, three shepherds
        # summed to a measured peak sheep speed of 2.3 (> 2x max_speed): a flock that
        # literally cannot be cornered, and a 6000-iteration MAPPO run that plateaued at
        # ~7% penned. Herding only works when the flock is slower than the dogs.
        # ``test_sheep_cannot_outrun_the_shepherds`` pins it. Contact stays *outside* the
        # cap: it is the direct push the shepherds actually herd with, it is
        # self-limiting (the spring only acts while someone overlaps the sheep), and
        # capping it would make a sheep block rather than yield.
        self.flee_radius = flee_radius
        self.flee_gain = flee_gain
        # Separation uses the same falloff shape between sheep, so the flock spreads to
        # roughly sep_radius rather than collapsing to a point under cohesion.
        # sep_radius=0.15 (was 0.2): the wider radius pushed the free flock's own
        # equilibrium size past what a pen could contain at any reasonable pen_radius
        # (see the module docstring's §0 measurement); 0.15 shrinks the settled flock so
        # ``pen_radius=0.3`` covers it with margin. Left uncapped: unlike flee it is not
        # sustained by anything a policy controls — it vanishes as the sheep it acts on
        # move apart — and at sep_gain=1.0 a single term already sits at the flee cap.
        # Equilibrium against cohesion is where sep_gain * (1 - d/sep_radius) ==
        # cohesion_gain * d/2, i.e. d ~= 0.14 here.
        self.sep_radius = sep_radius
        self.sep_gain = sep_gain
        # Mild cohesion toward the flock centroid; it is what keeps a scattered flock
        # recoverable instead of turning one bad approach into a lost episode. Identically
        # zero for n_sheep=1 (the centroid is the sheep).
        self.cohesion_gain = cohesion_gain
        self.linear_damping = linear_damping
        # A sheep's top running speed, and the hard guarantee behind
        # ``test_sheep_cannot_outrun_the_shepherds``. The flee cap above already bounds
        # the *sustained* speed at 0.5, but it says nothing about a **contact transient**:
        # a shepherd arriving at full speed compresses a k=100 spring several centimetres
        # before the sheep responds, and the measured peak was 2.07 even with the flee sum
        # capped and a single shepherd chasing a single sheep. Softening the contact
        # instead was the wrong trade — contact_k is also the engine's agent<->agent
        # collision_k, so the k=15 that would have sufficed lets shepherds interpenetrate
        # — and even then it only bought a 3% margin. A speed cap is independent of dt, of
        # the stiffness and of how many shepherds pile on, and it is the standard
        # formulation (Strombom's sheep move at a fixed top speed). Default: 75% of the
        # shepherds' max_speed, so a shepherd can always close on a fleeing sheep, with
        # the flee-only terminal 0.5 left comfortably underneath — the cap catches
        # transients and leaves the flee law itself shaping the behaviour.
        self.max_sheep_speed = max_sheep_speed if max_sheep_speed is not None else 0.75 * max_speed
        self.pos_shaping_factor = pos_shaping_factor
        # pen_reward is now a **potential** on the penned count (see the reward section
        # below), not a per-step level bonus: the old level form bounded an episode's
        # total at 0.5 * 5 sheep * 150 steps = 375 for "park two sheep and stop", the
        # exact local optimum the diagnosed failed run found. As a potential its total is
        # bounded at pen_reward * n_sheep = 5, a 75x cut that removes the incentive to
        # stall instead of finish. pen_reward=1.0 (was 0.5) since it is paid once now
        # rather than every step.
        self.pen_reward = pen_reward
        # scatter_penalty defaults to 0.0: the old flat penalty on flock spread is
        # superseded by gather_factor's potential on the same quantity (mean_R), which
        # does not have the "penalize being scattered forever" failure mode a level term
        # does. Kept reachable for a scenario config that wants both.
        self.scatter_penalty = scatter_penalty
        # gather_factor: potential-shaping reward on shrinking mean_R (the flock's own
        # mean radius about its centroid) — the "collect" half of Strombom's
        # collect-then-drive strategy, which the shipped reward never priced in at all
        # (only driving the centroid toward the pen was rewarded). Shaped on the mean,
        # not the max: caging's lesson (see caging.py) is that a max hands gradient to
        # exactly one member and the policy learns to service only that one.
        self.gather_factor = gather_factor
        # pen_level_reward: an optional per-step *level* bonus on the penned count, off
        # by default because a level term is exactly what caused the original failure
        # (see pen_reward above); kept as a knob for a config that wants a small
        # continuous pull in addition to the potential.
        self.pen_level_reward = pen_level_reward
        # done_reward: paid once, the step the hold counter reaches pen_hold, so success
        # itself carries a term at least as large as the whole per-step shaping budget
        # rather than being visible only through the potentials' endpoints.
        self.done_reward = done_reward
        # side_factor: a per-agent "get behind the flock" bonus (dot of the agent's own
        # offset from the centroid against the centroid's offset from the pen — positive
        # when the agent is on the far side, driving the flock in). Kept small (its
        # episode total at 200 steps is at most 2.0, versus done_reward's 10) precisely
        # because it is a *level* term, and level terms are what caused the failure this
        # redesign fixes.
        self.side_factor = side_factor
        # crowd_factor/crowd_radius: a per-agent penalty for shepherds bunching up on top
        # of each other, squared rather than absolute for caging.py's reason (the L1
        # gradient vanishes whenever an agent's two relevant terms fall on the same side
        # of the target — measured there as a third to half of agents getting zero
        # gradient; the squared form starves only the single stationary point).
        self.crowd_factor = crowd_factor
        self.crowd_radius = crowd_radius
        # intrude_penalty: a per-agent penalty for standing inside the pen. A real
        # failure mode, not hygiene — every other term pulls a shepherd toward the pen,
        # and a shepherd parked *in* the pen flees the sheep straight back out, making
        # done unreachable for as long as it sits there.
        self.intrude_penalty = intrude_penalty
        # drive_factor sets how far past the flock (away from the pen) the "drive point"
        # Strombom hint sits, in units of sep_radius * sqrt(n_sheep) (Strombom's own
        # scaling for the standoff a shepherd needs to push a flock of that size without
        # scattering it).
        self.drive_factor = drive_factor
        # dir_floor: the hard floor for safe_unit (see _safe_unit) — not a mere epsilon.
        self.dir_floor = dir_floor
        self.pen_hold = int(pen_hold)
        self.flock_dist_min = flock_dist_min
        self.flock_dist_max = flock_dist_max
        self.flock_spread = flock_spread
        # Derived host constants, computed once here so both paths (torch and the fused
        # kernels) use bit-identical values — recomputing any of these independently in
        # each path is a parity bug waiting to happen.
        self._inv_sheep = 1.0 / n_sheep
        self._drive_offset = drive_factor * sep_radius * math.sqrt(n_sheep)
        self._collect_offset = sep_radius
        self._dir_floor = dir_floor
        # The stray argmax's tie-break margin. See ``observations`` for why an argmax
        # needs one at all; the value has to live here so the kernel and the oracle scan
        # with the identical constant.
        self._tie_eps = 1.0e-5
        #: Current multipliers on the constructor's flock-spawn distance/spread (see
        #: :meth:`set_flock_distance` / :meth:`set_flock_spread`). ``1.0`` is the task as
        #: constructed.
        self.flock_dist_scale = 1.0
        self.flock_spread_scale = 1.0
        self._set_spawn_geometry(1.0, 1.0)

    # ------------------------------------------------------------ helpers

    def _safe_unit(self, v: torch.Tensor) -> torch.Tensor:
        """Unit vector, or exactly zero below a **hard floor** — not an epsilon.

        Everywhere else in this scenario, normalizing with ``clamp(min=1e-9)`` is fine
        because those normals are always multiplied by a magnitude that vanishes with
        them (e.g. the contact/flee/separation normals above). ``drive_pt`` and
        ``collect_pt`` are not: ``gcm - pen -> 0`` is *exactly* the solved configuration
        the policy will live in, so an ill-conditioned direction there is an O(1)
        disagreement between the torch and fused paths, not a ulp — hence a real floor
        (``dir_floor = 1e-3``) below which the direction is defined to be zero on both
        paths, rather than a tiny epsilon that still blows up as ``v -> 0``.
        """
        n = v.norm(dim=-1, keepdim=True)
        return torch.where(
            n >= self._dir_floor, v / n.clamp(min=self._dir_floor), torch.zeros_like(v)
        )

    # ------------------------------------------------------------ spawn geometry

    def _set_spawn_geometry(self, dist_scale: float, spread_scale: float) -> None:
        """Recompute every spawn constant the reset kernel takes, at the given scales.

        Split out of ``__init__`` because :meth:`set_flock_distance` /
        :meth:`set_flock_spread` need exactly this set recomputed and nothing else, and
        re-derives it from the constructor's parameters rather than undoing a previous
        scale by inverse arithmetic — an argument worth keeping precisely because a
        second scale axis would otherwise have to compose with whatever the first left
        behind. ``dist_scale=spread_scale=1.0`` multiplies by exactly 1.0, which is exact
        in IEEE754 — the restored distribution is bit-for-bit the constructor's, not
        merely close.
        """
        r, ws = self.sheep_radius, self.world_size
        self._spawn_lim = ws - 2.0 * self.agent_radius  # shepherd spawn box
        self._pen_lim = max(0.0, ws - self.pen_radius)  # keeps the pen disc inside
        self._sheep_bound = ws - r  # sheep position clamp
        spread = spread_scale * self.flock_spread
        # Never start already solved: the flock centre stands off past the pen wall by at
        # least one sheep diameter beyond it.
        stand_off = self.pen_radius + 2.0 * r
        dmin = max(stand_off, dist_scale * self.flock_dist_min)
        dmax = max(dmin, dist_scale * self.flock_dist_max)
        # And it must fit in the world, or a small world puts every flock centre on the
        # wall regardless of what dist_scale asked for. (This clamp is load-bearing:
        # ``test_shepherding_fused.py::_env`` uses ``world_size=0.6``.)
        reach = ws - spread - r
        dmin = min(dmin, reach)
        dmax = min(dmax, max(reach, dmin))
        self._flock_spread = spread
        self._flock_dmin = dmin
        self._flock_span = dmax - dmin

    def set_flock_distance(self, scale: float) -> None:
        """Scale how far the flock centre is drawn from the pen at reset.

        The dominant curriculum axis (measured: easy end ``D ~ [0.30, 0.40]`` scripted
        solve 0.97 vs. full difficulty ``D ~ [0.5, 1.2]`` scripted 0.85 — see the module
        docstring). Mirrors :meth:`~swarp.scenarios.giveway.GiveWayScenario.
        set_corridor_scale`'s curriculum-hook shape. ``set_flock_distance(1.0)``
        reproduces the constructor's distribution exactly for a given seed — the
        derivation is re-run from the constructor's parameters, and the RNG stream is
        untouched (the scale rides on the *span* of an existing draw, it does not add or
        reorder one).

        Host-side only — call it between batches, never inside a capture. None of this
        reaches a kernel that runs *inside* the whole-step graph: these scalars are read
        by the reset kernel alone, which is eager (``supports_graph_reset()`` is
        ``False``), so there is no device buffer to keep them in and no recapture to
        trigger. What it *does* have to survive is the reset launch's
        :class:`~swarp.core.cached_launch.CachedLaunch`, which packs its arguments once
        — that is why ``_flock_dmin``/``_flock_span`` (and ``_flock_spread``, shared with
        :meth:`set_flock_spread`) are entries in that launch's key, so changing a scale
        mismatches the key and repacks instead of silently replaying the old geometry
        forever. ``test_spawn_scale_shrinks_the_ring`` is what pins it.
        """
        if scale < 0.0:
            raise ValueError(f"flock distance scale must be non-negative, got {scale}")
        self.flock_dist_scale = scale
        self._set_spawn_geometry(self.flock_dist_scale, self.flock_spread_scale)

    def set_flock_spread(self, scale: float) -> None:
        """Scale how dispersed the flock starts at reset (see :meth:`set_flock_distance`
        for the shared discipline this and that method both follow).

        Measured nearly irrelevant next to distance on its own (module docstring), but
        kept as an independent axis rather than folded into distance: a curriculum that
        wants "close and already collected" (test the drive phase) versus "close but
        scattered" (test the collect phase) needs the two to vary separately.
        """
        if scale < 0.0:
            raise ValueError(f"flock spread scale must be non-negative, got {scale}")
        self.flock_spread_scale = scale
        self._set_spawn_geometry(self.flock_dist_scale, self.flock_spread_scale)

    def make_world(self, n_envs, device, dt, substeps, dtype, world_config=None) -> World:
        cfgs = [
            AgentConfig(
                model=DynamicsModel.HOLONOMIC,
                ctrl_mode=ControlMode.VELOCITY,
                radius=self.agent_radius,
                max_speed=self.max_speed,
                max_accel=2.0 * self.max_speed,
            )
            for _ in range(self.n_agents)
        ]
        margin = 0.5 * self.agent_radius
        # neighbor reach must cover shepherd<->sheep contact so the step sees it
        reach = 2.0 * max(self.agent_radius, self.sheep_radius) + margin
        cfg = WorldConfig(
            collisions=True,
            collision_k=self.contact_k,
            collision_c=self.contact_c,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            neighbor_radius=reach,
            max_neighbors=min(32, max(4, self.n_agents)),
        ).override_with(world_config)
        self.dt = dt
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        # Flock state and pen: allocated here (n_envs is known) so the fused spec can
        # adopt them, and written in place by every reset so their handles stay valid.
        tt = {"device": device, "dtype": dtype}
        self.sheep_pos = torch.zeros(n_envs, self.n_sheep, 2, **tt)
        self.sheep_vel = torch.zeros(n_envs, self.n_sheep, 2, **tt)
        self.pen_pos = torch.zeros(n_envs, 2, **tt)
        # ``world.goals`` is this scenario's **render-side mirror of the pen centre**,
        # one row per shepherd, written by the reset kernel. It is not an input to obs,
        # reward, done or the fused kernels — ``pen_pos`` is the authoritative copy those
        # read — but the viewer's existing goal overlay draws it, which is what puts a
        # pen marker in a rendered frame without any renderer change. It is also the
        # honest reading: the pen is the shepherds' one shared goal. Allocated here (not
        # in reset_world) so the fused spec can adopt it.
        self.world.goals = torch.zeros(n_envs, self.n_agents, 2, **tt)
        # ONE retained obstacle spec over sheep_pos, re-installed rather than rebuilt:
        # both ``resolve`` and ``any_movable`` memoize, so a re-install allocates nothing
        # and costs no device->host sync. Built here so the single count-changing install
        # happens before any graph capture. It aliases sheep_pos, so writing new sheep
        # positions in place is all a "move the flock" update needs. No ``kind``: the
        # engine must NOT integrate these (this scenario does), and a spec without a kind
        # field skips the ``any_movable`` reduction entirely.
        self._sheep_radius = torch.full((self.n_sheep,), self.sheep_radius, **tt)
        self._obstacles = Obstacles(self.sheep_pos.detach(), self._sheep_radius).resolve(
            device, dtype
        )
        self._prev_dist: torch.Tensor | None = None
        self._prev_radius: torch.Tensor | None = None
        self._prev_penned: torch.Tensor | None = None
        self._hold: torch.Tensor | None = None
        self._cache: dict[str, torch.Tensor] | None = None
        # Teammate index for the torch observation: row ``a`` is every other shepherd, in
        # ascending order — the same gather trick ``pusht.py`` uses for its own
        # teammate block. Fixed by ``n_agents``, so it is built here rather than rebuilt
        # on every ``observations()`` call.
        na = self.n_agents
        idx = torch.arange(na, device=device)
        others = idx.unsqueeze(0).expand(na, -1)[idx.unsqueeze(1) != idx.unsqueeze(0)]
        self._others = others.view(na, max(na - 1, 0))  # [A, A-1]
        # The reset kernel's spawn constants (all host-side, so the reset stays
        # sync-free) are owned by ``_set_spawn_geometry``, which ``__init__`` has already
        # run and which ``set_flock_distance``/``set_flock_spread`` re-run — deriving
        # them here too would give a curriculum change one copy to miss.
        # Cached, repack-once launches for the eager reset path — see
        # swarp/core/cached_launch.py.
        self._reset_launch = CachedLaunch()
        self._obs_launch = CachedLaunch()
        self._reward_launch = CachedLaunch()
        return self.world

    @property
    def obs_dim(self) -> int:
        return 16 + 4 * self.n_agents + 2 * self.n_sheep

    # ------------------------------------------------------------------ reset

    def reset_world(self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False) -> None:
        """Masked reset in one Warp launch (see :mod:`swarp.scenarios.reset_kernels`).

        The flock buffers are written **in place** by the kernel, so the fused path's
        cached Warp handles and the whole-step graph stay valid across resets; the grad
        path's ``_refresh`` still reassigns them (fresh tensors for the tape), which the
        framework's handle resync catches on the next no-grad step.
        """
        w = self.world
        mask, use_mask = self.reset_mask_wp(env_mask)
        st = w.state_wp()
        scalar = w.wp_dtype
        vec2 = VEC2[scalar]
        if self.fused_active:
            # Reuse the fused spec's cached, pointer-resynced handles instead of
            # re-wrapping these watched tensors on every reset (see navigation's
            # ``_launch_reset`` for the same fix and its rationale). ``sync_fused_
            # handles`` must run first: it is what notices a grad-path reassignment
            # and rebuilds the handle before this kernel writes through it.
            self.ensure_fused()
            self.sync_fused_handles()
            sheep_pos = self._wp["sheep_pos"]
            sheep_vel = self._wp["sheep_vel"]
            pen = self._wp["pen"]
            goals = self._wp["goals"]
        else:
            sheep_pos = wp.from_torch(self.sheep_pos.contiguous(), dtype=vec2)
            sheep_vel = wp.from_torch(self.sheep_vel.contiguous(), dtype=vec2)
            pen = wp.from_torch(self.pen_pos.contiguous(), dtype=vec2)
            goals = wp.from_torch(w.goals.contiguous(), dtype=vec2)
        seed = wp.int32(w.next_kernel_seed())
        with torch_stream_scope(w.device):
            launch = self._reset_launch.get(
                concrete(shepherding_reset_kernel, scalar),
                dim=w.n_envs,
                inputs=[
                    mask,
                    use_mask,
                    seed,
                    scalar(self._spawn_lim),
                    scalar(self._pen_lim),
                    scalar(self._sheep_bound),
                    scalar(self._flock_dmin),
                    scalar(self._flock_span),
                    scalar(self._flock_spread),
                    scalar(_TWO_PI),
                    wp.int32(self.n_agents),
                    wp.int32(self.n_sheep),
                    st.pos,
                    st.vel,
                    sheep_pos,
                    sheep_vel,
                    pen,
                    goals,
                ],
                device=w.device,
                key=(
                    w.n_envs,
                    ptr_key(mask),
                    self._spawn_lim,
                    self._pen_lim,
                    self._sheep_bound,
                    self._flock_dmin,
                    self._flock_span,
                    self._flock_spread,
                    self.n_agents,
                    self.n_sheep,
                    ptr_key(st.pos),
                    ptr_key(st.vel),
                    ptr_key(sheep_pos),
                    ptr_key(sheep_vel),
                    ptr_key(pen),
                    ptr_key(goals),
                ),
            )
            launch.set_param_by_name("use_mask", use_mask)
            launch.set_param_by_name("seed", seed)
            launch.launch()
        w.mark_pos_dirty()

        self._install_obstacles()
        if not self.fused_active:
            # The three *baselines* are dropped and re-derived from the state below;
            # ``_hold`` deliberately is not. It is a **counter**, not a baseline: there
            # is nothing in the current state to re-derive it from, and
            # ``Environment.step`` runs an obs-only ``reset_world`` on **every** step
            # (with an all-false mask when nothing actually ended), so nulling it here
            # would silently reset the hold to zero once per step and make ``done``
            # unreachable. Dropping the baselines is safe for exactly the reason it is
            # unsafe for the counter — a baseline re-derived on a pass that did not move
            # the flock lands back on the value the step already stored. ``_refresh``'s
            # reset branch zeroes the counter for the masked envs instead, which is what
            # the fused kernel's ``reset_hit`` does.
            self._prev_dist = None
            self._prev_radius = None
            self._prev_penned = None
        self.finish_reset(env_mask, obs_only=obs_only)

    def _install_obstacles(self) -> None:
        """Hand the current sheep positions to the engine as circular obstacles.

        Called from inside the captured whole-step graph (see :meth:`launch_fused`), so it
        must stay in the capture-safe regime of :meth:`swarp.core.stepper.Stepper.
        set_obstacles`: an unchanged obstacle count, and a spec carrying no movable
        obstacles and no 1-D ``angle`` to broadcast. All three hold by construction — the
        flock size is fixed at ``make_world``, the spec declares no ``kind`` and no
        ``angle``.

        The spec is built once, in ``make_world``, and **re-installed** rather than
        rebuilt: it aliases ``sheep_pos``, and both ``Obstacles.resolve`` and
        ``Obstacles.any_movable`` memoize, so a re-install allocates nothing and costs no
        device->host sync.

        The retained spec is rebuilt only when ``sheep_pos`` is *reassigned*, which is the
        grad path integrating the flock in torch (it needs fresh tensors for the tape).
        That is a host-side pointer compare, and on the captured path the pointer never
        moves, so the rebuild branch cannot fire inside a capture.
        """
        if self._obstacles.pos.data_ptr() != self.sheep_pos.data_ptr():
            self._obstacles = Obstacles(self.sheep_pos.detach(), self._sheep_radius).resolve(
                self.world.device, self.world.dtype
            )
        self.world.set_obstacles(self._obstacles)

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh(integrate=True)

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        """Materialize ``env_mask=None`` as an all-true mask, then refresh.

        ``_refresh`` distinguishes a step from a reset by ``reset_mask is None``, so
        handing it a bare ``None`` for "reset everything" would take the *step* branch
        and advance the hold counter on a reset. The fused path has exactly this shape
        and resolves it the same way: ``reset_mask_wp`` returns ``use_mask=0`` for a full
        reset and :meth:`~swarp.scenarios.fused.FusedScenario.finish_reset` then fills
        the shared mask buffer with ones, precisely so the obs/reward kernels can read it
        unconditionally. This is that fill.
        """
        if env_mask is None:
            env_mask = torch.ones(self.world.n_envs, dtype=torch.bool, device=self.world.device)
        self._refresh(reset_mask=env_mask, integrate=False)

    # --------------------------------------------------------- fused fast path

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        ne, na, k = n_envs, self.n_agents, self.n_sheep
        flock = {"alloc": "never", "carry": True, "watch": True}
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("reward", (ne, na)),
            Buf("dist", (ne,)),
            Buf("penned", (ne,)),
            Buf("done", (ne,), "uint8", bool_view=True),
            Buf("resetmask", (ne,), "uint8", reset_mask=True),
            # Scratch between the force and body launches. Framework-owned and fully
            # rewritten every pass, so it is not a carry.
            Buf("force", (ne, k, 2), "vec2"),
            # Flock state and pen: the scenario's own tensors, advanced in place by the
            # body kernel and reassigned by the grad path's torch integrator.
            Buf("sheep_pos", (ne, k, 2), "vec2", attr="sheep_pos", **flock),
            Buf("sheep_vel", (ne, k, 2), "vec2", attr="sheep_vel", **flock),
            Buf("pen", (ne, 2), "vec2", attr="pen_pos", alloc="never", watch=True),
            # The render-side pen mirror. Adopted (never allocated here) so the reset
            # kernel writes through the same cached handle as everything else; no kernel
            # on the step path reads it, so it is neither a carry nor an obs input.
            Buf("goals", (ne, na, 2), "vec2", attr="world.goals", alloc="never", watch=True),
            Buf("prev", (ne, k), attr="_prev_dist", alloc="if_none", carry=True, watch=True),
            # The three new reward baselines. carry=True because the reward kernel
            # advances them **in place** and graph warm-up must snapshot them —
            # fused.py's carry docstring is explicit that omitting this fails silently
            # (it just advances the simulation one extra step at capture). watch=True
            # because the torch oracle reassigns them via torch.where exactly as it does
            # ``_prev_dist``. ``hold`` is a float in the world dtype (not an int buffer)
            # so the torch oracle's rebase stays a plain ``torch.where``.
            Buf("prev_radius", (ne,), attr="_prev_radius", alloc="if_none", carry=True, watch=True),
            Buf("prev_penned", (ne,), attr="_prev_penned", alloc="if_none", carry=True, watch=True),
            Buf("hold", (ne,), attr="_hold", alloc="if_none", carry=True, watch=True),
            Buf("radius", (ne,)),  # info: flock_radius
            Buf("allpen", (ne,), "uint8", bool_view=True),  # info: all_penned
        )

    def engine_carries(self) -> list[torch.Tensor]:
        """The engine's obstacle positions: ``_install_obstacles`` overwrites them from
        inside the hook, and the *next* step's physics reads them."""
        return [self.world.obstacle_state_views()[0]]

    def launch_fused(self, pass_: FusedPass) -> None:
        """Force, integrate, re-install the flock, then obs and reward.

        A reset skips the flock advance (there is nothing to integrate: the sheep were
        just placed) and passes ``advance_prev=0`` so the reward kernel *rebases* the
        shaping baselines for the reset envs instead of differencing against a stale one.
        The ``_install_obstacles`` in the middle is an ordinary line here, running inside
        ``wp.ScopedCapture`` on a step — see its docstring for why that is legal.
        """
        st = self.world.state_wp()  # one wrap for every launch in this pass
        if pass_.is_step:
            self._launch_force(st)
            self._launch_body()
            self._install_obstacles()  # updated sheep positions for the next step
        self._launch_obs(st)
        self._launch_reward(st, advance_prev=pass_.advance_prev, full_pass=pass_.full_pass)

    def _launch_force(self, st) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        wp.launch(
            concrete(shepherding_force_kernel, scalar),
            dim=(w.n_envs, self.n_sheep),
            inputs=[
                st.pos,
                st.vel,
                pk["sheep_pos"],
                pk["sheep_vel"],
                wp.int32(self.n_agents),
                wp.int32(self.n_sheep),
                scalar(self.agent_radius),
                scalar(self.sheep_radius),
                scalar(self.contact_margin),
                scalar(self.contact_k),
                scalar(self.contact_c),
                scalar(self.flee_radius),
                scalar(self.flee_gain),
                scalar(self.sep_radius),
                scalar(self.sep_gain),
                scalar(self.cohesion_gain),
            ],
            outputs=[pk["force"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_body(self) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        bound = self.world_size - self.sheep_radius
        wp.launch(
            concrete(shepherding_body_kernel, scalar),
            dim=(w.n_envs, self.n_sheep),
            inputs=[
                pk["force"],
                scalar(self.sheep_mass),
                scalar(self.linear_damping),
                scalar(self.max_sheep_speed),
                scalar(self.dt),
                scalar(bound),
            ],
            outputs=[pk["sheep_pos"], pk["sheep_vel"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_obs(self, st) -> None:
        w = self.world
        scalar = w.wp_dtype
        sheep_pos, sheep_vel, pen = self._wp["sheep_pos"], self._wp["sheep_vel"], self._wp["pen"]
        obs = self._wp["obs"]
        # No per-call-varying argument at all: pointer-stable persistent-mode handles
        # plus host scalars fixed for the scenario's lifetime.
        launch = self._obs_launch.get(
            concrete(shepherding_obs_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                sheep_pos,
                sheep_vel,
                pen,
                wp.int32(self.n_agents),
                wp.int32(self.n_sheep),
                scalar(self._inv_sheep),
                scalar(self._drive_offset),
                scalar(self._collect_offset),
                scalar(self._dir_floor),
                scalar(self._tie_eps),
            ],
            outputs=[obs],
            device=w.device,
            key=(
                w.n_envs,
                self.n_agents,
                ptr_key(st.pos),
                ptr_key(st.vel),
                ptr_key(sheep_pos),
                ptr_key(sheep_vel),
                ptr_key(pen),
                self.n_sheep,
                self._inv_sheep,
                self._drive_offset,
                self._collect_offset,
                self._dir_floor,
                self._tie_eps,
                ptr_key(obs),
            ),
        )
        launch.launch()

    def _launch_reward(self, st, advance_prev: int, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        sheep_pos, pen, resetmask = pk["sheep_pos"], pk["pen"], pk["resetmask"]
        agent_pos = st.pos
        prev, reward, done = pk["prev"], pk["reward"], pk["done"]
        prev_radius, prev_penned, hold = pk["prev_radius"], pk["prev_penned"], pk["hold"]
        dist, penned, radius, allpen = pk["dist"], pk["penned"], pk["radius"], pk["allpen"]
        launch = self._reward_launch.get(
            concrete(shepherding_reward_kernel, scalar),
            dim=w.n_envs,
            inputs=[
                sheep_pos,
                pen,
                agent_pos,
                resetmask,
                wp.int32(self.n_agents),
                wp.int32(self.n_sheep),
                scalar(self._inv_sheep),
                scalar(self.pos_shaping_factor),
                scalar(self.gather_factor),
                scalar(self.pen_radius),
                scalar(self.pen_reward),
                scalar(self.pen_level_reward),
                scalar(self.done_reward),
                scalar(self.scatter_penalty),
                scalar(self.side_factor),
                scalar(self.crowd_factor),
                scalar(self.crowd_radius),
                scalar(self.intrude_penalty),
                scalar(self._dir_floor),
                wp.int32(self.pen_hold),
                wp.int32(advance_prev),
                wp.int32(full_pass),
            ],
            outputs=[
                prev,
                prev_radius,
                prev_penned,
                hold,
                reward,
                done,
                dist,
                penned,
                radius,
                allpen,
            ],
            device=w.device,
            key=(
                w.n_envs,
                ptr_key(sheep_pos),
                ptr_key(pen),
                ptr_key(agent_pos),
                ptr_key(resetmask),
                self.n_agents,
                self.n_sheep,
                self._inv_sheep,
                self.pos_shaping_factor,
                self.gather_factor,
                self.pen_radius,
                self.pen_reward,
                self.pen_level_reward,
                self.done_reward,
                self.scatter_penalty,
                self.side_factor,
                self.crowd_factor,
                self.crowd_radius,
                self.intrude_penalty,
                self._dir_floor,
                self.pen_hold,
                ptr_key(prev),
                ptr_key(prev_radius),
                ptr_key(prev_penned),
                ptr_key(hold),
                ptr_key(reward),
                ptr_key(done),
                ptr_key(dist),
                ptr_key(penned),
                ptr_key(radius),
                ptr_key(allpen),
            ),
        )
        launch.set_param_by_name("advance_prev", wp.int32(advance_prev))
        launch.set_param_by_name("full_pass", wp.int32(full_pass))
        launch.launch()

    def _refresh(self, reset_mask: torch.Tensor | None = None, integrate: bool = True) -> None:
        w = self.world
        pos, vel = w.state.pos, w.state.vel  # [n_envs, n_agents, 2]

        if integrate:
            q = self.sheep_pos  # [E, K, 2]
            rel = q.unsqueeze(1) - pos.unsqueeze(2)  # shepherd->sheep [E, A, K, 2]
            dist = rel.norm(dim=-1).clamp(min=1e-9)  # [E, A, K]
            n_hat = rel / dist.unsqueeze(-1)
            # (1) contact: the reaction of the same spring-damper the agent step applies
            overlap = (self.agent_radius + self.sheep_radius + self.contact_margin) - dist
            active = (overlap > 0).to(w.dtype)
            fmag = self.contact_k * overlap.clamp(min=0.0)
            rel_vel = self.sheep_vel.unsqueeze(1) - vel.unsqueeze(2)  # [E, A, K, 2]
            vn = (rel_vel * n_hat).sum(-1)
            f_contact = ((fmag - self.contact_c * vn) * active).unsqueeze(-1) * n_hat
            # (2) flee: linear falloff repulsion from every shepherd inside flee_radius,
            # acting well beyond contact range. This is the reactive part of the world.
            fall = (1.0 - dist / self.flee_radius).clamp(min=0.0)
            near = (dist < self.flee_radius).to(w.dtype)
            f_flee = ((self.flee_gain * fall * near).unsqueeze(-1) * n_hat).sum(dim=1)
            # Cap the *summed* flee force at flee_gain — N converging shepherds must not
            # stack N times the panic (see the flee_gain comment in __init__ for why
            # that made the task unsolvable). The kernel does the same clamp at the same
            # point in the same order: below the cap the scale is exactly 1.0, so this is
            # bit-identical to the unclamped sum there, not merely close.
            flee_mag = f_flee.norm(dim=-1, keepdim=True)
            f_flee = f_flee * (self.flee_gain / flee_mag.clamp(min=1e-9)).clamp(max=1.0)
            # (3) separation: same falloff shape between sheep, so the flock does not
            # collapse to a point. The diagonal is dropped explicitly (an ``eye`` mask,
            # matching the kernel's ``j != k``) rather than by testing d > 0.
            sep_rel = q.unsqueeze(2) - q.unsqueeze(1)  # [E, K(i), K(j), 2], j -> i
            sd = sep_rel.norm(dim=-1)  # [E, K, K]
            sdc = sd.clamp(min=1e-9)
            eye = torch.eye(self.n_sheep, device=w.device, dtype=torch.bool)
            s_active = ((sd < self.sep_radius) & ~eye).to(w.dtype)
            s_fall = (1.0 - sd / self.sep_radius).clamp(min=0.0)
            s_hat = sep_rel / sdc.unsqueeze(-1)
            f_sep = ((self.sep_gain * s_fall * s_active).unsqueeze(-1) * s_hat).sum(dim=2)
            # (4) cohesion toward the flock centroid; identically zero for one sheep.
            f_coh = self.cohesion_gain * (q.mean(dim=1, keepdim=True) - q)

            f_total = (f_contact.sum(dim=1) + f_flee) + f_sep + f_coh
            dt = self.dt
            self.sheep_vel = (self.sheep_vel + f_total / self.sheep_mass * dt) * (
                1.0 - self.linear_damping * dt
            )
            # Speed cap, applied to the integrated velocity before it moves the position
            # — same place, same order as the kernel. Below the cap the scale is exactly
            # 1.0, so this is bit-identical to the uncapped velocity there.
            speed = self.sheep_vel.norm(dim=-1, keepdim=True)
            self.sheep_vel = self.sheep_vel * (self.max_sheep_speed / speed.clamp(min=1e-9)).clamp(
                max=1.0
            )
            b = self.world_size - self.sheep_radius
            self.sheep_pos = (self.sheep_pos + self.sheep_vel * dt).clamp(-b, b)
            self._install_obstacles()  # for the next step

        pen = self.pen_pos.unsqueeze(1)  # [E, 1, 2]
        dist_to_pen = (self.sheep_pos - pen).norm(dim=-1)  # [E, K]
        if self._prev_dist is None:
            self._prev_dist = dist_to_pen.detach().clone()
        shaping = (self._prev_dist - dist_to_pen) * self.pos_shaping_factor

        centroid = self.sheep_pos.mean(dim=1, keepdim=True)  # gcm, [E, 1, 2]
        r_k = (self.sheep_pos - centroid).norm(dim=-1)  # [E, K]
        mean_r = r_k.mean(dim=-1)  # [E]
        if self._prev_radius is None:
            self._prev_radius = mean_r.detach().clone()
        gather = (self._prev_radius - mean_r) * self.gather_factor  # [E]

        penned_mask = dist_to_pen < self.pen_radius  # [E, K]
        penned_count = penned_mask.to(w.dtype).sum(dim=-1)  # [E]
        if self._prev_penned is None:
            self._prev_penned = penned_count.detach().clone()
        pen_delta = (penned_count - self._prev_penned) * self.pen_reward  # [E]
        all_penned = penned_mask.all(dim=-1)  # [E]

        if reset_mask is None:
            # A step: rebase every baseline to the value it just measured, and advance
            # the hold counter under "a step happened" (the fused kernel's
            # ``advance_prev``/``full_pass`` gates collapse to this on the eager path,
            # which never runs an obs-only pass).
            self._prev_dist = dist_to_pen.detach().clone()
            self._prev_radius = mean_r.detach().clone()
            self._prev_penned = penned_count.detach().clone()
            if self._hold is None:
                self._hold = torch.zeros_like(mean_r)
            self._hold = torch.where(all_penned, self._hold + 1.0, torch.zeros_like(self._hold))
            first_done = self._hold == float(self.pen_hold)  # STRICT ==, once per episode
        else:
            # A reset: rebase all four baselines for the reset envs (no reward earned on
            # the transition into a fresh episode) and leave the others untouched —
            # exactly the discipline ``_prev_dist`` already followed before this redesign.
            shaping = torch.where(reset_mask.unsqueeze(-1), torch.zeros_like(shaping), shaping)
            self._prev_dist = torch.where(
                reset_mask.unsqueeze(-1), dist_to_pen.detach(), self._prev_dist
            )
            gather = torch.where(reset_mask, torch.zeros_like(gather), gather)
            self._prev_radius = torch.where(reset_mask, mean_r.detach(), self._prev_radius)
            pen_delta = torch.where(reset_mask, torch.zeros_like(pen_delta), pen_delta)
            self._prev_penned = torch.where(reset_mask, penned_count.detach(), self._prev_penned)
            if self._hold is None:
                self._hold = torch.zeros_like(mean_r)
            self._hold = torch.where(reset_mask, torch.zeros_like(self._hold), self._hold)
            first_done = torch.zeros_like(all_penned)

        global_r = (
            shaping.sum(dim=-1)
            + gather
            + pen_delta
            + self.pen_level_reward * penned_count
            + self.done_reward * first_done.to(w.dtype)
            - self.scatter_penalty * mean_r
        )
        done = self._hold >= float(self.pen_hold)

        # Per-agent terms (side/crowd/intrude), computed here rather than deferred to
        # ``rewards()`` so ``_cache`` holds the finished per-agent tensor exactly the way
        # ``global_r`` above is finished — one place per step that touches this state.
        gcm_to_pen = self._safe_unit(centroid.squeeze(1) - self.pen_pos)  # [E, 2]
        off = pos - centroid  # gcm -> own_pos is own_pos - gcm; side wants p_a - gcm
        side_dir = self._safe_unit(off)  # [E, A, 2]
        side = self.side_factor * (side_dir * gcm_to_pen.unsqueeze(1)).sum(dim=-1)  # [E, A]

        d_ag = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)  # [E, A, A]
        eye_a = torch.eye(self.n_agents, device=w.device, dtype=torch.bool)
        crowd_gap = (self.crowd_radius - d_ag).clamp(min=0.0)
        crowd_gap = torch.where(eye_a, torch.zeros_like(crowd_gap), crowd_gap)
        crowd = -self.crowd_factor * (crowd_gap**2).sum(dim=-1)  # [E, A]

        d_pen = (pos - self.pen_pos.unsqueeze(1)).norm(dim=-1)  # [E, A]
        intrude = -self.intrude_penalty * (self.pen_radius - d_pen).clamp(min=0.0)  # [E, A]

        per_agent = side + crowd + intrude  # [E, A]

        self._cache = {
            "dist_to_pen": dist_to_pen.mean(dim=-1),  # [E]
            "penned_frac": penned_count * self._inv_sheep,  # [E]
            "penned_count": penned_count,  # [E]
            "all_penned": all_penned,  # [E]
            "mean_radius": mean_r,  # [E]
            "global_reward": global_r,  # [E]
            "per_agent_reward": per_agent,  # [E, A]
            "done": done,  # [E]
            "sheep_rel": (self.sheep_pos.unsqueeze(1) - pos.unsqueeze(2)).reshape(
                w.n_envs, w.n_agents, self.n_sheep * 2
            ),
            "gcm": centroid.squeeze(1),  # [E, 2]
            "gcm_vel": self.sheep_vel.mean(dim=1),  # [E, 2]
            "r_k": r_k,  # [E, K]
        }

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["obs"]
        w = self.world
        s = w.state
        c = self._cache
        na, k = self.n_agents, self.n_sheep

        own_pos, own_vel = s.pos, s.vel  # [E, A, 2]
        pen_rel = self.pen_pos.unsqueeze(1) - own_pos  # [E, A, 2]
        gcm = c["gcm"].unsqueeze(1)  # [E, 1, 2]
        gcm_rel = gcm - own_pos  # [E, A, 2]
        pen_gcm = (self.pen_pos - c["gcm"]).unsqueeze(1).expand(-1, na, -1)  # [E, A, 2]
        gcm_vel = c["gcm_vel"].unsqueeze(1).expand(-1, na, -1)  # [E, A, 2]
        mean_r = c["mean_radius"].view(-1, 1, 1).expand(-1, na, 1)  # [E, A, 1]

        # max_R / stray: a running scan that only takes a new maximum when it clears the
        # incumbent by ``_tie_eps``, so a near-tie always resolves to the *lower* index.
        #
        # A plain ``argmax`` is wrong here, and not subtly: the parity oracle and the
        # fused kernel integrate the flock in different reduction orders, so their sheep
        # positions differ by ~1e-9 immediately and by ~1e-7 over the 20-step parity
        # rollout (see the class comment). ``stray`` is an *identity*, so any comparison
        # decided within that drift flips it and moves this slot -- and ``collect_pt``
        # with it -- by O(1), not by a ulp. That is not hypothetical: at ``n_sheep=2``
        # the centroid is the midpoint, so the two radii are **exactly** equal every
        # step and the argmax is a pure coin flip between the two paths. It failed the
        # fused obs parity on the very first rollout. The same argument is why the
        # per-sheep block is not sorted.
        #
        # ``_tie_eps = 1e-5`` sits two decades above the worst observed path divergence
        # and more than an order of magnitude below the smallest genuine radius gap the
        # flock's equilibrium produces (2e-4 at ``n_sheep=3``, 1.5e-3 at 5), so it
        # resolves the degenerate ties without ever masking a real stray. ``max_r`` is
        # carried alongside rather than recomputed, so slot 13 and slots 14-15 always
        # describe the *same* sheep.
        max_r = c["r_k"][:, 0]  # [E]
        stray_idx = torch.zeros(w.n_envs, dtype=torch.long, device=w.device)
        for j in range(1, k):
            take = c["r_k"][:, j] > max_r + self._tie_eps
            max_r = torch.where(take, c["r_k"][:, j], max_r)
            stray_idx = torch.where(take, j, stray_idx)
        max_r = max_r.view(-1, 1, 1).expand(-1, na, 1)  # [E, A, 1]
        stray = self.sheep_pos.gather(1, stray_idx.view(-1, 1, 1).expand(-1, 1, 2))  # [E, 1, 2]
        stray_full = stray.expand(-1, na, -1)  # [E, A, 2], same stray for every agent's row
        stray_rel = stray_full - own_pos  # [E, A, 2]

        drive_pt = gcm + self._drive_offset * self._safe_unit(gcm - self.pen_pos.unsqueeze(1))
        drive_rel = drive_pt.expand(-1, na, -1) - own_pos  # [E, A, 2]
        collect_pt = stray_full + self._collect_offset * self._safe_unit(stray_full - gcm)
        collect_rel = collect_pt - own_pos  # [E, A, 2]

        if na > 1:
            other_pos = own_pos[:, self._others]  # [E, A, A-1, 2]
            other_vel = own_vel[:, self._others]  # [E, A, A-1, 2]
            other_rel_pos = other_pos - own_pos.unsqueeze(2)
            other_rel_vel = other_vel - own_vel.unsqueeze(2)
            others_block = torch.cat([other_rel_pos, other_rel_vel], dim=-1).flatten(2)
        else:
            others_block = own_pos.new_zeros(w.n_envs, na, 0)

        sheep_rel = self.sheep_pos.unsqueeze(1) - own_pos.unsqueeze(2)  # [E, A, K, 2]
        sheep_block = sheep_rel.reshape(w.n_envs, na, k * 2)

        return torch.cat(
            [
                own_pos,
                own_vel,
                pen_rel,
                gcm_rel,
                pen_gcm,
                gcm_vel,
                mean_r,
                max_r,
                stray_rel,
                drive_rel,
                collect_rel,
                others_block,
                sheep_block,
            ],
            dim=-1,
        )

    def global_reward(self) -> torch.Tensor:
        return self._cache["global_reward"]

    def rewards(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["reward"]
        return self.global_reward().unsqueeze(1) + self._cache["per_agent_reward"]

    def done(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["done_bool"]
        return self._cache["done"]

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {
                "sheep_penned": self.fb["penned"],
                "sheep_dist_to_pen": self.fb["dist"],
                "flock_radius": self.fb["radius"],
                "all_penned": self.fb["allpen_bool"].to(self.world.dtype),
            }
        return {
            "sheep_penned": self._cache["penned_frac"],
            "sheep_dist_to_pen": self._cache["dist_to_pen"],
            "flock_radius": self._cache["mean_radius"],
            "all_penned": self._cache["all_penned"].to(self.world.dtype),
        }
