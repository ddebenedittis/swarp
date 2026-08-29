"""Give-way: two corridors crossing in a one-lane intersection.

Four immovable box obstacles fill the corner quadrants of the arena, leaving a
plus-shaped free space — two corridors that cross at a junction. Every agent starts at
the outer end of one arm and must reach the outer end of the **opposite** arm. The
corridor is sized so that one robot fits and two abreast do not
(``2*agent_radius < 2*corridor_half_width < 4*agent_radius``), so opposing robots have to
take turns: one ducks into a perpendicular arm — the passing bay — while the other
crosses the junction.

That constraint is the whole point, and it makes this the one **non-monotone** task in
the package. Every other scenario here can be solved by a policy that decreases its own
distance-to-goal monotonically; here the feasible joint solution *requires* an agent to
move away from its goal for several steps while the other passes.

**Why the shaping is *not* shared by default.** This scenario used to default to
``shared_reward=True`` on the argument that per-agent shaping punishes the robot that
yields. That argument is wrong, and it is worth writing down why, because it is the kind
of mistake that survives a whole training run. Position shaping is a *potential*: summed
over an episode it telescopes to ``d_spawn - d_final``, so a detour into a passing bay is
fully refunded the moment the robot comes back out. The real cost of yielding is
**discounting** — at ``gamma=0.99`` a six-step detour costs about 1.7% of the terminal
bonus — not shaping. What sharing *does* buy is noise: summed over four agents, roughly
three quarters of each agent's reward is then teammate behaviour it did not cause. So the
knob is now the continuous :attr:`shaping_share` (a **mean**, not a sum), defaulting to
``0.0``; ``shared_reward=True`` is kept as a deprecated alias for ``1.0``.

Reward:
  * position shaping ``(prev_dist - dist) * pos_shaping_factor``, blended with the team
    mean by ``shaping_share`` and masked-rebased on reset,
  * ``collision_penalty`` times a **proximity ramp** ``prox`` (see below),
  * ``wall_penalty`` times a **wall ramp** (see below),
  * ``time_penalty`` every step, so standing still is never a solution,
  * ``goal_hold_bonus`` per agent per step while inside ``goal_tolerance``, so "arrive and
    stay" beats "arrive and drift" — which matters because ``done`` needs every robot on
    its goal *simultaneously*,
  * ``final_reward`` for the whole team once every agent is on its goal.

**Ramps, not counts — the reward bug this replaces.** The original contact term was
``collision_penalty`` times a count of pairs closer than ``r_a + r_b``. That count is
**structurally zero**: the contact spring activates at ``2r + contact_margin`` and, at the
shipped ``contact_k``/``substeps``, ``tests/scenarios/test_giveway_fused.py``'s crowded-
junction case measures *exactly zero* overlap — a pair can never reach ``d < 2r``. A
penetration-depth-proportional penalty would be just as void, for the same reason. So the
term is now a ramp over the **activation band**::

    prox[a]  = sum_{b != a} clamp((collision_reach - d_ab) / contact_margin, 0, 1)
    wallramp = clamp((wall_reach - sdf(p)) / contact_margin, 0, 1)

with ``collision_reach = 2r + contact_margin`` and ``wall_reach = r + contact_margin``.
Both are continuous, both actually fire, and both are deliberately **narrow**: the
anticipation this task needs has to come from the observation, not from a wide reward moat
that punishes exactly the close pass the corridor requires. The binary wall flag survives
as ``info()["wall_contacts"]`` so that metric stays interpretable, and the pair count
survives as ``info()["collisions"]`` — now a live diagnostic rather than a constant zero.

The wall term is also why the old balance failed. At ``wall_penalty=-0.1`` against a
measured 0.64 contact rate over a 300-step episode it was worth ``-19.2`` per agent
against a total available shaping of ``1.774`` — 2.7x the entire shaping budget, as a step
function with no gradient, and nearly unavoidable: the wall-free band is
``c - (r + margin) = 0.0125``, one sixth of the corridor width. The policy was being taxed
for existing in a corridor, and it learned to sit in its own arm.

**Observation: the travel frame.** Every vector feature is expressed in the per-episode
orthonormal frame ``(u, n)`` where ``u`` is the agent's own unit travel axis (exactly
``±e_x`` or ``±e_y``, written into ``_axis`` at reset) and ``n = (-u_y, u_x)``. The
plus-shape is 90°-symmetric, so nothing is lost, and a 4-fold symmetry the shared actor
would otherwise have to learn four times collapses to one. Two features come for free:
``dot(p, u)`` *is* signed distance along the arm (zero at the junction centre, negative
before it) and ``dot(p, n)`` *is* the lateral offset within the lane.

Row layout, ``obs_dim = 13 + 9 * k_obs`` (40 at the default four agents)::

    own (0..12):  dot(p,u), clamp(dot(p,n)/r, -8, 8), dot(v,u), dot(v,n),
                  dot(g-p,u), |g-p|, clamp((sdf(p)-r)/r, -1, 8),
                  dot(grad sdf, u), dot(grad sdf, n), c/r, own priority token,
                  u_x, u_y            <- world frame, see below
    per slot j:   rel_along, rel_lat, rvel_along, rvel_lat,
                  ndir_along, ndir_lat, n_goal_dist, dprio, valid

**The axis is in the row because the action is not in the frame.** Indices 11 and 12 are
the only two features that are *not* rotation-invariant, and they are load-bearing: a
holonomic velocity-mode agent is commanded in **world** coordinates, so a fully
frame-invariant row would not be invertible to an action. A robot in the ``+x`` arm
heading ``-x`` and a robot in the ``+y`` arm heading ``-y`` have identical own blocks and
need opposite world velocities; with ``u`` in the row the policy can compute its decision
in the frame and rotate it out, and the four-fold case analysis it is left with is a gate
on two inputs rather than the whole task learned four times over. It does not weaken the
head-on argument below — a head-on pair's axes are exact negations, so the two robots are
still identical *up to the frame*, and a policy that correctly rotates still hands them
the same frame-relative intent.

The neighbour block is **grouped per neighbour**, a deliberate departure from navigation's
feature-grouped layout: nine heterogeneous features per slot read far better grouped, and
this scenario has no shared-layout obligation to navigation. ``ndir`` is the neighbour's
own travel axis projected into this agent's frame, so it is exactly ``(-1, 0)`` for
head-on traffic, ``(0, ±1)`` for crossing traffic and ``(1, 0)`` for same-way traffic —
the one feature that tells a robot *what kind* of conflict it is in. ``c/r`` is the current
corridor half-width: without it the curriculum is a hidden parameter and the policy is
being trained on a task whose geometry it cannot see.

**Sensing is decoupled from contact.** Neighbour slots are chosen **all-pairs** by
distance (lexicographic ``(d, b)``, so an exact tie goes to the lower index), not off the
physics neighbour grid. The grid's radius is the engine's *contact* reach, 0.1125 — a
robot could not see an opponent until they were almost touching, so every yield decision
was made after the collision was already paid for. Raising ``neighbor_radius`` is not the
fix either: it is the same number the contact search uses, so it would inflate the physics
neighbour lists on all 16 substeps. ``neighbor_obs`` defaults to ``n_agents - 1`` (see
each other robot); ``neighbor_radius`` remains a *physics-only* knob.

**The priority token.** Two robots meeting head-on have observations that are exact 180°
rotations of each other, so a shared-weight policy maps them to 180°-rotated actions:
both accelerate, or both retreat. The symmetry is exact, not approximate — there is a test
for it — so it has to be broken by the observation. Each episode draws a uniform random
**permutation** of the agents and rank-maps it to ``prio[a] = 2*rank/(n-1) - 1``. A
permutation rather than i.i.d. uniforms because an i.i.d. near-tie happens with
probability ~2*eps and a near-tie *is* the deadlock case, so i.i.d. draws put a hard
ceiling on the solve rate. Each agent sees its own token and the per-neighbour difference
``prio[b] - prio[a]``. ``use_priority=False`` zeroes the token everywhere while keeping
``obs_dim`` unchanged, so the ablation is a controlled A/B.

**Contact stiffness — the parameter this scenario lives or dies by.**
``WorldConfig.collision_k``'s default of 100 is far too soft for a corridor: a
velocity-mode agent has no contact memory and settles at a penetration of
``max_speed / (collision_k * sub_dt)``, so the corridor walls would be a suggestion.
Push-T's ``contact_k=8000`` is *also* too soft here, which is not obvious and was worth
measuring: at 8000 a robot driven into a corner block sinks **0.033** into it, two thirds
of its own radius. Blocks that yield that far are not one-lane geometry, and the
consequence is not subtle — a greedy "drive straight at the goal" policy solves the task
**100%** of the time, because the robots simply squeeze through each other and the walls.
The whole premise of the scenario is gone.

Two *different* terms set how far a robot sinks, and they pull opposite ways in
``substeps`` — which is why stiffness alone cannot fix this:

* **settling depth**, ``max_speed / (k * sub_dt)`` — a velocity-mode agent has no contact
  memory, so holding it out of a wall needs ``f >= m * max_speed / sub_dt``. Falls with
  ``k``, *rises* with more substeps.
* **transient overshoot**, ``max_speed * sub_dt`` — an agent travels that far in the one
  substep before the contact can answer. Independent of ``k`` entirely, and falls with
  more substeps.

Raising ``k`` alone therefore hits a floor at the overshoot term. Measured, 8 robots on
random actions for 600 steps, worst overlap as a fraction of one radius:

======================  ==========  ==========  ==========
``contact_k``           ss=8        ss=16       ss=32
======================  ==========  ==========  ==========
200 000                 16.8%       **0%**      **0%**
500 000                 10.5%       **0%**      **0%**
2 000 000               16.7%       **0%**      **0%**
======================  ==========  ==========  ==========

At ``substeps=8`` the overshoot is ``1.0 * 0.05/8 = 0.00625``, half a radius's worth and
larger than the contact margin, so a fast robot is already inside the block before the
spring sees it — and no stiffness recovers that. At 16 the overshoot lands inside the
margin, contact catches it first, and penetration goes to *exactly* zero at every
stiffness tried. **Hence ``substeps >= 16``**, not the 8 this scenario originally
documented.

``contact_k`` defaults to 200 000 and the spring saturation (``contact_max_overlap``) is
switched **off**. Push-T needs that saturation to stop a stiff contact launching a light
movable body; give-way has no movable body at all — every obstacle is immovable scenery —
so a cap on the restoring force can only ever let robots sink deeper under a multi-robot
jam. The parity suites run at ``substeps=1``, where the settling depth is smallest.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal

import torch
import warp as wp

from swarp._overloads import concrete
from swarp.core.cached_launch import CachedLaunch, ptr_key
from swarp.core.config import Obstacles, ObstacleShape, WorldConfig
from swarp.core.state import VEC2
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.interop.autograd import torch_stream_scope
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario
from swarp.scenarios.giveway_kernels import (
    giveway_obs_kernel,
    giveway_reset_kernel,
    giveway_reward_kernel,
)


def _safe_sqrt(sq: torch.Tensor) -> torch.Tensor:
    """``sqrt`` that is finite *and* has a finite gradient at exactly zero.

    ``sqrt'(0) = inf``, and a single ``torch.where`` around the ``sqrt`` does **not**
    help: autograd still evaluates the taken-and-not-taken branch, so the ``inf`` lands
    on the tape and every downstream gradient becomes ``nan``. The two-``where`` idiom
    below feeds ``sqrt`` a strictly positive number in *both* branches and selects the
    zero afterwards.

    This matters here in two places that are exactly zero over most of the world:
    :meth:`GiveWayScenario._corner_sdf`'s ``sqrt(ox*ox + oy*oy)`` is zero everywhere
    inside a block *and* in either face region — i.e. down the whole corridor — and the
    pairwise distance matrix is zero on its own diagonal. Both feed observation features
    now, where they used to feed only a ``<``. Bit-identical to ``wp.sqrt(0.0) == 0.0``,
    so parity is unaffected.
    """
    safe = torch.where(sq > 0, sq, torch.ones_like(sq))
    return torch.where(sq > 0, safe.sqrt(), torch.zeros_like(sq))


class GiveWayScenario(FusedScenario):
    def __init__(
        self,
        n_agents: int = 4,
        agent_radius: float = 0.05,
        world_size: float = 1.0,
        corridor_half_width: float | None = None,
        max_speed: float = 1.0,
        neighbor_obs: int | None = None,
        neighbor_radius: float | None = None,
        use_priority: bool = True,
        shaping_share: float = 0.0,
        shared_reward: bool | None = None,
        pos_shaping_factor: float = 2.0,
        collision_penalty: float = -0.1,
        wall_penalty: float = -0.02,
        time_penalty: float = -0.02,
        goal_hold_bonus: float = 0.02,
        final_reward: float = 15.0,
        goal_tolerance: float | None = None,
        spawn_jitter: float | None = None,
        desync_mode: Literal["ranked", "random"] = "ranked",
        contact_k: float = 200000.0,
        contact_c: float = 100.0,
        neighbor_method: str = "auto",
    ) -> None:
        # Four is the default because arm ``i % 4`` only produces a *head-on* pair from
        # three agents up (0 and 2 share the x corridor), and it is the head-on pair that
        # makes the task non-monotone — two agents alone meet at right angles, where
        # yielding costs nothing more than waiting.
        if n_agents < 1:
            raise ValueError(f"n_agents must be >= 1, got {n_agents}")
        self.n_agents = n_agents
        self.agent_radius = agent_radius
        self.world_size = world_size
        # One lane wide: 2r < 2c < 4r. Derived from the agent radius rather than
        # hardcoded, because the whole task definition is "one robot fits, two do not" —
        # a literal would silently stop meaning that the moment agent_radius moved.
        c = 1.5 * agent_radius if corridor_half_width is None else corridor_half_width
        if not agent_radius < c < 2.0 * agent_radius:
            raise ValueError(
                "corridor_half_width must satisfy agent_radius < c < 2*agent_radius so "
                "that exactly one robot fits abreast (2r < 2c < 4r); got "
                f"c={c} with agent_radius={agent_radius}"
            )
        self.corridor_half_width = c
        self.max_speed = max_speed
        # Every other robot, by default: sensing is all-pairs and deliberately decoupled
        # from the contact-sized physics grid (see the module docstring).
        self.neighbor_obs = n_agents - 1 if neighbor_obs is None else neighbor_obs
        #: Physics only — the engine's agent<->agent contact search radius. It is **not**
        #: what the observation sees; raising it would inflate the neighbour lists the
        #: collision kernel walks on every one of the 16 substeps and buy the policy
        #: nothing.
        self.neighbor_radius = neighbor_radius
        self.use_priority = use_priority
        if shared_reward is not None:
            warnings.warn(
                "GiveWayScenario(shared_reward=...) is deprecated; use the continuous "
                "shaping_share (a mean, not a sum) instead. True maps to 1.0, False to "
                "0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            shaping_share = 1.0 if shared_reward else 0.0
        if not 0.0 <= shaping_share <= 1.0:
            raise ValueError(f"shaping_share must lie in [0, 1], got {shaping_share}")
        #: How much of each agent's shaping is the **team mean** rather than its own.
        self.shaping_share = shaping_share
        self.pos_shaping_factor = pos_shaping_factor
        self.collision_penalty = collision_penalty
        self.wall_penalty = wall_penalty
        self.time_penalty = time_penalty
        self.goal_hold_bonus = goal_hold_bonus
        self.final_reward = final_reward
        self.goal_tolerance = goal_tolerance if goal_tolerance is not None else 2.0 * agent_radius
        self.spawn_jitter = spawn_jitter if spawn_jitter is not None else 0.25 * agent_radius
        if desync_mode not in ("ranked", "random"):
            raise ValueError(f"desync_mode must be 'ranked' or 'random', got {desync_mode!r}")
        #: ``"ranked"`` staggers spawn depth *by the priority token* (which is what grounds
        #: the token: at low difficulty "higher token goes first" is trivially observable,
        #: and as the amplitude anneals to zero the correlation vanishes and the token has
        #: to carry the convention alone). ``"random"`` is the uncorrelated control.
        self.desync_mode = desync_mode
        self.contact_k = contact_k
        self.contact_c = contact_c
        self.neighbor_method = neighbor_method
        # A quarter of the radius, not the half navigation uses. The contact-activation
        # gap is what the reward's wall ramp is measured against, and at margin = 0.5*r
        # the activation band (r + margin = 1.5r) would land *exactly* on the corridor's
        # centreline distance (c = 1.5r) — every agent driving straight down the middle
        # would sit on the knife edge of the ramp. A quarter radius puts the centreline a
        # clear 0.25r outside the band.
        self.contact_margin = 0.25 * agent_radius
        # The two activation reaches the reward ramps run over: the same reaches the
        # engine's contacts use (``_static_forces`` inflates the agent by ``ra + margin``,
        # the pair search by ``ra + rb + margin``), so a ramp reaching 1 means "the contact
        # force is live", not some second, softer notion of near.
        self._wall_reach = agent_radius + self.contact_margin
        self._collision_reach = 2.0 * agent_radius + self.contact_margin
        # Depth at which the contact spring saturates; filled in by make_world, which is
        # the first place ``sub_dt`` is known. Same derivation as pusht.
        self.max_overlap = 0.0
        #: Current multiple of ``corridor_half_width`` (see :meth:`set_corridor_scale`).
        self.corridor_scale = 1.0
        # Read by _set_geometry, so it has to exist before the first call.
        self._desync_frac = 0.0
        self._set_geometry(c)
        self._k_obs = min(self.neighbor_obs, max(0, self.n_agents - 1))

    # --------------------------------------------------------------- geometry

    def _set_geometry(self, c: float) -> None:
        """Recompute every corridor-width-dependent host constant for half-width ``c``.

        Split out from ``__init__`` because :meth:`set_corridor_scale` and
        :meth:`set_spawn_desync` each need exactly this set recomputed and nothing else.
        The one-lane check is *not* repeated here: a curriculum deliberately runs
        wider-than-one-lane corridors, and only the constructor's base width defines the
        task.
        """
        r, w = self.agent_radius, self.world_size
        #: The half-width currently installed — the base width times ``corridor_scale``.
        #: Also what the observation reports as ``c/r``, and what
        #: :meth:`set_spawn_desync` re-runs this method with.
        self.c = c
        # The first-quadrant block spans [c, W] x [c, W]; its three mirror images fill the
        # other corners, leaving the plus-shaped free space.
        self._box_center = (0.5 * (w + c), 0.5 * (w + c))
        self._box_half = (0.5 * (w - c), 0.5 * (w - c))
        # The usable stretch of one arm: from ``s_min`` (any deeper and the agent is in
        # the junction rather than in its arm) out to ``outer``, the same
        # ``world_size - 2*agent_radius`` limit every other scenario spawns within.
        outer = w - 2.0 * r
        self._s_min = c + 2.0 * r
        if outer <= self._s_min:
            raise ValueError(
                f"world_size={w} leaves no arm at corridor_half_width={c} with "
                f"agent_radius={r}; need world_size > c + 4*agent_radius"
            )
        # Agents past the fourth stack up inward along their arm, one rank per four.
        per_arm = math.ceil(self.n_agents / 4)
        span = outer - self._s_min
        # The floor on the rank spacing is the *contact activation* distance, not merely
        # the touching distance: at this stiffness two agents that spawn already
        # inside each other's activation band start the episode being shoved apart, which
        # is a physics artefact rather than a task. Below it there is genuinely no room in
        # a one-lane arm for this many ranks, so it raises rather than packing them anyway.
        min_gap = 2.0 * r + self.contact_margin
        # The jitter is budgeted *before* the ranks are laid out, so the outermost agent
        # still lands inside ``outer`` and the innermost inside ``s_min`` — hence the
        # ``span - 2*cap`` the ranks are packed into and the ``outer - jitter`` below,
        # rather than a jitter bolted onto a layout that already filled the arm.
        cap = min(self.spawn_jitter, 0.5 * max(0.0, 3.0 * r - min_gap))
        inner_span = max(0.0, span - 2.0 * cap)
        if per_arm > 1:
            self._stagger = min(3.0 * r, inner_span / (per_arm - 1))
            if self._stagger < min_gap:
                raise ValueError(
                    f"{self.n_agents} agents need {per_arm} ranks per arm, which packs "
                    f"them {self._stagger:.4f} apart in an arm of length {span:.4f} — "
                    f"closer than the {min_gap:.4f} contact distance. Raise world_size "
                    "or lower n_agents."
                )
            long_room = 0.5 * (self._stagger - min_gap)
        else:
            self._stagger = 0.0
            long_room = 0.5 * span
        # Jitter amplitudes, clamped so a spawn can neither leave its arm, close on the
        # next rank, nor start inside the wall's contact band (which would hand the very
        # first step a wall-contact penalty it did nothing to earn).
        self._jitter_long = min(cap, max(0.0, long_room))
        self._jitter_lat = min(self.spawn_jitter, 0.5 * max(0.0, c - self._wall_reach))
        self._s_max = outer - self._jitter_long
        # Spawn-desynchronization amplitude, clamped by exactly the same two budgets the
        # jitter is: a staggered *and* jittered spawn may neither cross ``_s_min`` into the
        # junction nor close on the next rank in its own arm. Recomputed here rather than
        # in ``set_spawn_desync`` because ``set_corridor_scale`` moves ``_s_min``, and a
        # stale amplitude would then spawn a robot inside the junction.
        room = self._s_max - self._jitter_long - (per_arm - 1) * self._stagger - self._s_min
        desync_max = max(0.0, room)
        if per_arm > 1:
            desync_max = min(
                desync_max, max(0.0, self._stagger - min_gap - 2.0 * self._jitter_long)
            )
        #: Largest depth offset the geometry can absorb; ``set_spawn_desync(1.0)`` uses it.
        self._desync_max = desync_max
        self._desync_amp = self._desync_frac * desync_max

    def set_corridor_scale(self, scale: float) -> None:
        """Widen (or restore) the corridor to ``scale`` times the constructor's width.

        The curriculum hook the MAPPO trainer drives, and its useful range is **not** the
        obvious one. Two robots fit abreast as soon as ``2*(scale*c) >= 4r``, i.e. from
        ``scale = 2r/c = 1.333`` up — but that is a *geometric* threshold, and geometry is
        not what a stochastic policy experiences. At ``scale = 1.5`` the whole slack is
        ``2c - 4r = 0.025``, a quarter of a robot diameter, so passing abreast needs
        near-perfect straight-line driving: a greedy "drive at the goal" controller solves
        96% of episodes there, and the *same* controller with modest Gaussian action noise
        solves 25–46%. The behavioural threshold is measured at ``scale ~ 1.75``, where the
        noisy controller is back to 93–97%.

        So the curriculum span is ``1.0 + 0.75*(1 - f)``: ``f = 0`` lands on the measured
        learnability threshold rather than the geometric one, ``f ~ 0.56`` crosses
        two-abreast, and ``f = 1`` is the real one-lane width. This is worth spelling out
        because both obvious alternatives fail — ``2.0 - f`` spends a quarter of its range
        above 1.75 on indistinguishable geometry, and ``1.0 + 0.5*(1 - f)`` never reaches a
        setting a noisy policy can actually solve, which measures as a run whose difficulty
        *falls* to zero and still plateaus at ``episode_solve ~ 0.23``.

        ``set_corridor_scale(1.0)`` reproduces the constructor's geometry exactly (the
        derivation is re-run from the base width, not undone by inverse arithmetic).

        Host-side only — call it between batches, never inside a capture. It writes into
        the retained obstacle spec's own tensors and re-installs it, so the obstacle
        *count* never changes and ``Stepper.set_obstacles`` takes its in-place path: no
        reallocation, no graph recapture. The corner-block extents the fused SDF reads
        live in the ``geom`` device buffer this also rewrites, which is why the kernel
        takes them as an array rather than as four baked scalars — see
        :func:`~swarp.scenarios.giveway_kernels.giveway_obs_kernel`.
        """
        if scale <= 0.0:
            raise ValueError(f"corridor scale must be positive, got {scale}")
        self._set_geometry(self.corridor_half_width * scale)
        self.corridor_scale = scale
        if getattr(self, "world", None) is not None:
            self._install_geometry()

    def set_spawn_desync(self, frac: float) -> None:
        """Stagger how deep in its arm each robot spawns, as a fraction of the room there is.

        The second curriculum axis, and the one that **grounds the priority token**. At
        ``frac = 1`` the robots start at very different depths and therefore reach the
        junction one at a time, so "higher token goes first" is trivially observable —
        under the default ``desync_mode="ranked"`` the depth offset *is* the token's rank.
        As the trainer anneals ``frac`` to zero the correlation vanishes and the token has
        to carry the convention on its own. ``desync_mode="random"`` staggers by an
        independent draw instead, which is the control that says how much of the benefit
        was the grounding rather than the mere desynchronization.

        Only the **spawn** moves; the goal stays at the un-desynced arm end. Mirroring the
        offset into the goal (which would hold each robot's travel distance constant) puts
        the shallowest robot's goal a few centimetres from the junction, so it arrives at
        once and then blocks the crossing for the rest of the episode: a scripted
        controller that solves 100% of episodes at ``frac=0`` solves **0%** at any
        ``frac>0`` under that form, with ``frac_on_goal`` pinned at exactly 1/4. Measured,
        not argued.

        Host-side only, like :meth:`set_corridor_scale`; the amplitude is clamped by
        :meth:`_set_geometry` so a staggered, jittered spawn can neither cross into the
        junction nor close on the next rank in its own arm.
        """
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"spawn desync must lie in [0, 1], got {frac}")
        self._desync_frac = frac
        self._set_geometry(self.c)

    @property
    def spawn_desync(self) -> float:
        """The fraction last passed to :meth:`set_spawn_desync` (0 = synchronized)."""
        return self._desync_frac

    def _install_geometry(self) -> None:
        """Push the current corridor geometry to the engine and to the fused SDF buffer."""
        (bx, by), (hx, hy) = self._box_center, self._box_half
        tt = {"device": self.world.device, "dtype": self.world.dtype}
        # Broadcast [4, 2] into the [n_envs, 4, 2] the spec holds: the blocks are the same
        # scenery in every env.
        self._box_pos.copy_(torch.tensor([[bx, by], [-bx, by], [-bx, -by], [bx, -by]], **tt))
        self._box_half_t.copy_(torch.tensor([[hx, hy]] * 4, **tt))
        self._geom.copy_(torch.tensor([bx, by, hx, hy], **tt))
        # Re-install the *retained* resolved spec: already normalized (so ``resolve``
        # returns it untouched), ``kind`` absent (so ``any_movable`` never syncs), and an
        # unchanged count, which is the whole in-place contract.
        self.world.set_obstacles(self._obstacles)

    # ------------------------------------------------------------------ world

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
        margin = self.contact_margin
        reach = 2.0 * self.agent_radius + margin
        # Saturation OFF, unlike pusht. ``contact_max_overlap`` caps the restoring force
        # beyond a given depth, which pusht needs so a stiff contact cannot launch its
        # light movable T. Give-way has no movable body -- every obstacle is immovable
        # scenery -- so the cap has no upside here and one real downside: under a
        # four-robot junction jam the forces superpose, and a capped spring lets the pile
        # sink instead of pushing back harder. Kept as an attribute because it is part of
        # the contact story this class documents.
        self.max_overlap = 0.0
        cfg = WorldConfig(
            collisions=True,
            collision_k=self.contact_k,
            collision_c=self.contact_c,
            collision_margin=margin,
            contact_max_overlap=self.max_overlap,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            # The corner blocks are scanned linearly by ``_static_forces``, not looked up
            # in the neighbor grid, so their size does not have to inflate the
            # agent<->agent reach. Nothing in the observation reads this grid either — see
            # the module docstring on decoupling sensing from contact.
            neighbor_radius=max(self.neighbor_radius or 0.0, reach),
            max_neighbors=min(32, max(4, self.n_agents)),
            neighbor_method=self.neighbor_method,
        ).override_with(world_config)
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        tt = {"device": device, "dtype": dtype}
        # Goals: engine-independent per-agent targets, written in place by every reset;
        # allocated here (not in reset_world) so the fused spec can adopt them.
        self.world.goals = torch.zeros(n_envs, self.n_agents, 2, **tt)
        # The per-episode travel frame and priority token. Both are written **only** by
        # the reset kernel and read by the obs path, and neither is ever *reassigned* —
        # which is precisely what makes ``watch=False`` in the fused spec correct: a
        # watched buffer exists to catch a pointer move, and these cannot move. (Nor are
        # they carries: no fused pass advances them in place, and
        # ``supports_graph_reset()`` stays False, so warm-up never runs a reset.)
        self._axis = torch.zeros(n_envs, self.n_agents, 2, **tt)
        self._prio = torch.zeros(n_envs, self.n_agents, **tt)
        # The corner-block geometry the fused SDF reads, as a device buffer rather than
        # kernel scalars — see set_corridor_scale.
        self._geom = torch.zeros(4, **tt)
        # The obstacle spec, built and resolved once. Every field is already on the right
        # device/dtype and contiguous, so ``resolve`` hands back tensors that *share
        # storage* with these: writing them is writing the spec.
        self._box_pos = torch.zeros(n_envs, 4, 2, **tt)
        self._box_half_t = torch.zeros(4, 2, **tt)
        self._obstacles = Obstacles(
            self._box_pos,
            torch.zeros(4, **tt),  # a BOX ignores radius; its surface is its boundary
            shape=torch.full((4,), int(ObstacleShape.BOX), device=device, dtype=torch.int32),
            half_extents=self._box_half_t,
        ).resolve(device, dtype)
        # Installed **once**, here: the blocks are static scenery, so no reset and no
        # captured step ever touches the obstacle set again. Only the host-side curriculum
        # hook re-installs, and then with an unchanged count.
        self._install_geometry()
        # Static per-agent radii for the fused proximity ramp (the torch reference reads
        # ``World.agent_radius``, which is this same tensor). Never reassigned, so the
        # handle stays pointer-stable.
        self._radius_wp = wp.from_torch(self.world.agent_radius, dtype=self.world.wp_dtype)
        self._geom_wp = wp.from_torch(self._geom, dtype=self.world.wp_dtype)
        self._axis_wp = wp.from_torch(self._axis, dtype=VEC2[self.world.wp_dtype])
        self._prio_wp = wp.from_torch(self._prio, dtype=self.world.wp_dtype)
        # Left None so the first refresh seeds the shaping baseline from the spawn
        # distance (shaping 0) rather than from zeros; the fused spec adopts it with
        # alloc="if_none". The torch path reassigns it, which watch=True catches.
        self._prev_dist: torch.Tensor | None = None
        self._cache: dict[str, torch.Tensor] | None = None
        # Cached, repack-once launches for the eager reset path — see
        # swarp/core/cached_launch.py.
        self._reset_launch = CachedLaunch()
        self._obs_launch = CachedLaunch()
        return self.world

    @property
    def obs_dim(self) -> int:
        # 13 own features, then 9 per neighbour slot — see the module docstring's layout.
        return 13 + 9 * self._k_obs

    # ------------------------------------------------------------------ reset

    def reset_world(self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False) -> None:
        """Masked reset in one Warp launch (see :mod:`swarp.scenarios.reset_kernels`).

        Host-sync-free: the mask is applied inside the kernel, so there is no host-side
        "is anything done?" gate. The corner blocks are *not* re-installed — they are
        static scenery installed once in :meth:`make_world`, which is what keeps the
        obstacle set entirely out of the reset path.

        **Two** kernel seeds are drawn, always, in this order: the spawn stream and the
        priority stream. Unconditionally, even with ``use_priority=False``, so the host
        seed layout does not depend on a flag — see the reset kernel's docstring for the
        matching device-side contract.
        """
        w = self.world
        mask, use_mask = self.reset_mask_wp(env_mask)
        st = w.state_wp()
        scalar = w.wp_dtype
        if self.fused_active:
            # Reuse the fused spec's cached, pointer-resynced handle instead of
            # re-wrapping ``world.goals`` on every reset (see navigation's
            # ``_launch_reset`` for the same fix and its rationale).
            self.ensure_fused()
            self.sync_fused_handles()
            goals = self._wp["goals"]
            axis, prio = self._wp["axis"], self._wp["prio"]
        else:
            goals = wp.from_torch(w.goals.contiguous(), dtype=VEC2[scalar])
            axis, prio = self._axis_wp, self._prio_wp
        seed = wp.int32(w.next_kernel_seed())
        seed_prio = wp.int32(w.next_kernel_seed())
        with torch_stream_scope(w.device):
            launch = self._reset_launch.get(
                concrete(giveway_reset_kernel, scalar),
                dim=w.n_envs,
                inputs=[
                    mask,
                    use_mask,
                    seed,
                    seed_prio,
                    wp.int32(self.n_agents),
                    wp.int32(1 if self.use_priority else 0),
                    wp.int32(1 if self.desync_mode == "random" else 0),
                    scalar(self._s_max),
                    scalar(self._stagger),
                    scalar(self._desync_amp),
                    scalar(self._jitter_long),
                    scalar(self._jitter_lat),
                    st.pos,
                    st.theta,
                    st.vel,
                    st.speed,
                    st.ang_vel,
                    goals,
                    axis,
                    prio,
                ],
                device=w.device,
                key=(
                    w.n_envs,
                    ptr_key(mask),
                    self.n_agents,
                    self.use_priority,
                    self.desync_mode,
                    # The spawn scalars are in the key, not re-set per call: they are
                    # step-invariant, but ``set_corridor_scale``/``set_spawn_desync`` do
                    # move them, and a key entry is what repacks the launch when they do.
                    # ``_desync_amp`` in particular — leave it out and a curriculum step
                    # silently replays the previous amplitude.
                    self._s_max,
                    self._stagger,
                    self._desync_amp,
                    self._jitter_long,
                    self._jitter_lat,
                    ptr_key(st.pos),
                    ptr_key(st.theta),
                    ptr_key(st.vel),
                    ptr_key(st.speed),
                    ptr_key(st.ang_vel),
                    ptr_key(goals),
                    ptr_key(axis),
                    ptr_key(prio),
                ),
            )
            launch.set_param_by_name("use_mask", use_mask)
            launch.set_param_by_name("seed", seed)
            launch.set_param_by_name("seed_prio", seed_prio)
            launch.launch()
        w.mark_pos_dirty()
        self.finish_reset(env_mask, obs_only=obs_only)

    # --------------------------------------------------------- fused fast path

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        ne, na = n_envs, self.n_agents
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("touch", (ne, na)),
            Buf("prox", (ne, na)),
            Buf("wall", (ne, na)),
            Buf("wallramp", (ne, na)),
            Buf("dist", (ne, na)),
            Buf("shaping", (ne, na)),
            Buf("reward", (ne, na)),
            Buf("ongoal", (ne, na), "uint8", bool_view=True),
            Buf("frac", (ne,)),
            Buf("done", (ne,), "uint8", bool_view=True),
            Buf("resetmask", (ne,), "uint8", reset_mask=True),
            Buf("goals", (ne, na, 2), "vec2", attr="world.goals", alloc="never", watch=True),
            # Written only by the reset kernel, never reassigned (make_world owns them),
            # so: adopt rather than allocate, no ``watch`` (nothing can move), and no
            # ``carry`` (no fused pass advances them, and supports_graph_reset() is False
            # so warm-up never runs a reset).
            Buf("axis", (ne, na, 2), "vec2", attr="_axis", alloc="never"),
            Buf("prio", (ne, na), attr="_prio", alloc="never"),
            Buf("prev", (ne, na), attr="_prev_dist", alloc="if_none", carry=True, watch=True),
        )

    def launch_fused(self, pass_: FusedPass) -> None:
        """Obs, then reward — the reward on every pass but an obs-only auto-reset.

        Same rule as navigation's: ``full_pass=0`` (a mid-step auto-reset) keeps the
        reward/done already returned for that transition, while a standalone reset
        recomputes them so the fused path reports the *new* episode, as the torch oracle
        does. Nothing here touches the obstacle set — and, since the neighbour grid left
        the observation path, nothing here builds one either — so the whole sequence is
        trivially capture-safe: two launches against pointer-stable handles.
        """
        self._launch_obs(advance_prev=pass_.advance_prev, full_pass=pass_.full_pass)
        if pass_.full_pass:
            self._launch_reward()

    def _launch_obs(self, advance_prev: int, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        resetmask = self._wp["resetmask"]
        st = w.state_wp()
        goals, prev = self._wp["goals"], self._wp["prev"]
        axis, prio = self._wp["axis"], self._wp["prio"]
        obs, touch, prox, wall, wallramp, dist, shaping, ongoal = (
            self._wp["obs"],
            self._wp["touch"],
            self._wp["prox"],
            self._wp["wall"],
            self._wp["wallramp"],
            self._wp["dist"],
            self._wp["shaping"],
            self._wp["ongoal"],
        )
        # ``advance_prev``/``full_pass`` are the only genuinely per-call arguments;
        # everything else is static config or a pointer-stable handle, so it lives in the
        # cache key instead. The corner geometry is *not* in the key — it is read out of
        # the ``geom`` buffer by the kernel, so a curriculum step needs no repack (and,
        # for the captured step, no recapture).
        launch = self._obs_launch.get(
            concrete(giveway_obs_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                goals,
                axis,
                prio,
                self._radius_wp,
                self._geom_wp,
                resetmask,
                wp.int32(self.n_agents),
                wp.int32(self._k_obs),
                scalar(self.agent_radius),
                scalar(self._wall_reach),
                scalar(self._collision_reach),
                scalar(self.contact_margin),
                scalar(self.pos_shaping_factor),
                scalar(self.goal_tolerance),
                wp.int32(advance_prev),
                wp.int32(full_pass),
            ],
            outputs=[obs, touch, prox, wall, wallramp, dist, shaping, ongoal, prev],
            device=w.device,
            key=(
                w.n_envs,
                self.n_agents,
                ptr_key(st.pos),
                ptr_key(st.vel),
                ptr_key(goals),
                ptr_key(axis),
                ptr_key(prio),
                ptr_key(self._radius_wp),
                ptr_key(self._geom_wp),
                ptr_key(resetmask),
                self._k_obs,
                self.agent_radius,
                self._wall_reach,
                self._collision_reach,
                self.contact_margin,
                self.pos_shaping_factor,
                self.goal_tolerance,
                ptr_key(obs),
                ptr_key(touch),
                ptr_key(prox),
                ptr_key(wall),
                ptr_key(wallramp),
                ptr_key(dist),
                ptr_key(shaping),
                ptr_key(ongoal),
                ptr_key(prev),
            ),
        )
        launch.set_param_by_name("advance_prev", wp.int32(advance_prev))
        launch.set_param_by_name("full_pass", wp.int32(full_pass))
        launch.launch()

    def _launch_reward(self) -> None:
        w = self.world
        scalar = w.wp_dtype
        wp.launch(
            concrete(giveway_reward_kernel, scalar),
            dim=w.n_envs,
            inputs=[
                self._wp["shaping"],
                self._wp["prox"],
                self._wp["wallramp"],
                self._wp["ongoal"],
                wp.int32(self.n_agents),
                scalar(1.0 / self.n_agents),
                scalar(self.shaping_share),
                scalar(self.collision_penalty),
                scalar(self.wall_penalty),
                scalar(self.time_penalty),
                scalar(self.goal_hold_bonus),
                scalar(self.final_reward),
            ],
            outputs=[self._wp["reward"], self._wp["done"], self._wp["frac"]],
            device=w.device,
            record_tape=False,
        )

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh()

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        self._refresh(reset_mask=env_mask)

    def _corner_sdf(self, pos: torch.Tensor) -> torch.Tensor:
        """Signed distance from each agent to the nearest corner block.

        The torch half of :func:`~swarp.scenarios.giveway_kernels._corner_sdf`: the same
        fold into the first quadrant and the same association order (an explicit
        ``sqrt(ox*ox + oy*oy)`` rather than ``.norm()``), so the *discrete* contact flag
        the two paths derive from it agrees exactly rather than only closely. Written
        independently of the kernel — this is the parity oracle, not a shared helper.

        The ``sqrt`` goes through :func:`_safe_sqrt`: ``ox*ox + oy*oy`` is exactly zero
        inside a block *and* in either face region, which is most of the corridor, and
        this now feeds an observation feature rather than only a ``<``.
        """
        (bx, by), (hx, hy) = self._box_center, self._box_half
        qx = (pos[..., 0].abs() - bx).abs() - hx
        qy = (pos[..., 1].abs() - by).abs() - hy
        ox = qx.clamp(min=0.0)
        oy = qy.clamp(min=0.0)
        return _safe_sqrt(ox * ox + oy * oy) + torch.maximum(qx, qy).clamp(max=0.0)

    def _corner_sdf_grad(self, pos: torch.Tensor) -> torch.Tensor:
        """Analytic unit gradient of :meth:`_corner_sdf`, ``[..., 2]``.

        Analytic rather than a finite difference: it is about fifteen flops of sign
        selects sharing the ``sqrt`` the SDF already computes, and a finite difference
        would be a *different* function that the fused kernel would then have to
        reproduce to the last ulp.

        The chain is three folds deep. With ``ax = |px|`` and ``qx = |ax - cx| - hx``,
        ``d qx/d px = sign(px) * sign(ax - cx)``, and likewise in ``y``; the outer
        derivative is ``(ox, oy)/|(ox, oy)|`` where the point is outside the block and the
        box-interior ``(1, 0)`` or ``(0, 1)`` where it is inside. The two cases are
        mutually exclusive — ``min(max(qx, qy), 0)`` is only nonzero when both ``q`` are
        negative, which is exactly when ``ox = oy = 0``.

        Every tie-break is written ``>=`` (at ``px == 0``, at ``ax == cx``, at
        ``qx == qy``) and the Warp kernel writes the identical ``>=``: these select
        *branches*, so a mismatch between the two paths would be a discrete O(1)
        disagreement, not a rounding one. The norm is exactly 1 everywhere, including on
        the medial axes where the sign choice is arbitrary (``px == 0`` is the corridor
        centreline, where the SDF has a genuine ridge and no gradient — the value there is
        one of the two one-sided limits, chosen consistently by the ``>=``).
        """
        (bx, by), (hx, hy) = self._box_center, self._box_half
        px, py = pos[..., 0], pos[..., 1]
        one = torch.ones_like(px)
        sx = torch.where(px >= 0, one, -one)
        sy = torch.where(py >= 0, one, -one)
        ax, ay = px.abs(), py.abs()
        tx = torch.where(ax - bx >= 0, one, -one)
        ty = torch.where(ay - by >= 0, one, -one)
        qx = (ax - bx).abs() - hx
        qy = (ay - by).abs() - hy
        ox = qx.clamp(min=0.0)
        oy = qy.clamp(min=0.0)
        sq = ox * ox + oy * oy
        length = torch.where(sq > 0, _safe_sqrt(sq), one)  # 1 where unused: no 0/0
        inside_x = torch.where(qx >= qy, one, torch.zeros_like(one))
        gqx = torch.where(sq > 0, ox / length, inside_x)
        gqy = torch.where(sq > 0, oy / length, one - inside_x)
        return torch.stack([gqx * sx * tx, gqy * sy * ty], dim=-1)

    def _frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The per-episode travel frame ``(u, n)``, ``n`` being ``u`` turned 90° left."""
        u = self._axis  # [ne, na, 2], exactly a signed unit axis
        return u, torch.stack([-u[..., 1], u[..., 0]], dim=-1)

    def _pair_dsq(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(rel, dsq)`` for every ordered pair, ``rel[e, a, b] = pos[e, b] - pos[e, a]``.

        Squared, deliberately: the *ordering* the neighbour slots are chosen by is then a
        bit-identical expression on both paths (the kernel compares ``dx*dx + dy*dy`` in
        the same order), so no ``sqrt`` can perturb a tie.
        """
        rel = pos.unsqueeze(1) - pos.unsqueeze(2)
        return rel, rel[..., 0] * rel[..., 0] + rel[..., 1] * rel[..., 1]

    def _slot_order(self, dsq: torch.Tensor) -> torch.Tensor:
        """Neighbour slot indices ``[n_envs, n_agents, k_obs]``, lexicographic in ``(d, b)``.

        A **stable** sort on the squared distance with the diagonal pushed to ``+inf``
        reproduces exactly that: an exact tie keeps index order, and an agent can never
        win a slot from itself. The kernel walks the same order with a scratch-free
        "smallest ``(d, b)`` strictly greater than the last" scan.
        """
        na = dsq.shape[-1]
        eye = torch.eye(na, device=dsq.device, dtype=torch.bool).view(1, na, na)
        masked = torch.where(eye, torch.full_like(dsq, float("inf")), dsq)
        return torch.sort(masked, dim=-1, stable=True).indices[..., : self._k_obs]

    def _build_obs(self) -> torch.Tensor:
        """Assemble the observation row from the **live** state, ``[ne, na, obs_dim]``.

        Deliberately *not* cached by :meth:`_refresh`. A mid-step auto-reset refreshes the
        cache from inside ``torch.no_grad()``, so an observation computed there would come
        back detached and the differentiable path would silently stop being
        differentiable — a failure that shows up only as a missing gradient, which is
        exactly what ``test_grad_step_falls_back_to_torch`` exists to catch. The reward
        *inputs* have the opposite requirement (they must describe the transition that was
        already scored, not the fresh spawn), which is why they stay in the cache.
        """
        w = self.world
        ne, na = w.n_envs, w.n_agents
        pos, vel = w.state.pos, w.state.vel
        u, n = self._frame()

        def along(v):  # noqa: ANN001 - local
            return v[..., 0] * u[..., 0] + v[..., 1] * u[..., 1]

        def lat(v):  # noqa: ANN001 - local
            return v[..., 0] * n[..., 0] + v[..., 1] * n[..., 1]

        goal_rel = w.goals - pos
        dist_to_goal = _safe_sqrt(
            goal_rel[..., 0] * goal_rel[..., 0] + goal_rel[..., 1] * goal_rel[..., 1]
        )
        sdf = self._corner_sdf(pos)
        grad = self._corner_sdf_grad(pos)
        inv_r = 1.0 / self.agent_radius

        own = torch.stack(
            [
                along(pos),
                (lat(pos) * inv_r).clamp(-8.0, 8.0),
                along(vel),
                lat(vel),
                along(goal_rel),
                dist_to_goal,
                ((sdf - self.agent_radius) * inv_r).clamp(-1.0, 8.0),
                along(grad),
                lat(grad),
                # From the *device* geom buffer, not the host float, so both paths round
                # the same way: geom[0] - geom[2] = 0.5*(W+c) - 0.5*(W-c) is the current
                # corridor half-width, and the kernel reads it out of the same two floats.
                torch.zeros_like(dist_to_goal) + (self._geom[0] - self._geom[2]) * inv_r,
                self._prio,
                # The travel axis itself, in **world** frame — the one feature that is not
                # rotation-invariant, and the reason the rest can afford to be. See the
                # module docstring: without it the row is not invertible to an action.
                u[..., 0],
                u[..., 1],
            ],
            dim=-1,
        )
        k = self._k_obs
        if k == 0:
            return own

        rel, dsq = self._pair_dsq(pos)
        relv = vel.unsqueeze(1) - vel.unsqueeze(2)
        sel = self._slot_order(dsq)  # [ne, na, k]
        pick = sel.unsqueeze(-1).expand(ne, na, k, 2)
        nrel, nrelv = torch.gather(rel, 2, pick), torch.gather(relv, 2, pick)
        uk, nk = u.unsqueeze(2), n.unsqueeze(2)

        def kalong(v):  # noqa: ANN001 - local
            return v[..., 0] * uk[..., 0] + v[..., 1] * uk[..., 1]

        def klat(v):  # noqa: ANN001 - local
            return v[..., 0] * nk[..., 0] + v[..., 1] * nk[..., 1]

        # Per-agent quantities of the *selected* neighbours, gathered by their index.
        flat = sel.reshape(ne, -1)
        nb_axis = torch.gather(u, 1, flat.unsqueeze(-1).expand(-1, -1, 2)).view(ne, na, k, 2)
        nb_prio = torch.gather(self._prio, 1, flat).view(ne, na, k)
        nb_goal_dist = torch.gather(dist_to_goal, 1, flat).view(ne, na, k)
        block = torch.stack(
            [
                kalong(nrel),
                klat(nrel),
                kalong(nrelv),
                klat(nrelv),
                kalong(nb_axis),
                klat(nb_axis),
                nb_goal_dist,
                nb_prio - self._prio.unsqueeze(-1),
                # Always 1: slots are all-pairs and ``k_obs <= n_agents - 1``, so none can
                # go unfilled. Kept in the row so the layout survives a radius-gated
                # variant, and pinned as a known-constant column by the parity suite.
                torch.ones_like(nb_goal_dist),
            ],
            dim=-1,
        ).reshape(ne, na, 9 * k)
        return torch.cat([own, block], dim=-1)

    def _refresh(self, reset_mask: torch.Tensor | None = None) -> None:
        """Contact ramps, distances and reward terms for the current state.

        ``reset_mask`` (a boolean ``[n_envs]`` or ``None``) marks envs that were just
        reset: their shaping baseline is rebased to the fresh spawn distance and their
        shaping term zeroed, while every other env keeps its carried-over baseline.

        This is the **parity oracle**: it is written independently of
        :mod:`swarp.scenarios.giveway_kernels` and is what the fused path is tested
        against, so it deliberately shares no code with it.
        """
        w = self.world
        pos = w.state.pos
        na = w.n_agents
        r = w.agent_radius  # [n_agents]

        _, dsq = self._pair_dsq(pos)
        pair_d = _safe_sqrt(dsq)
        eye = torch.eye(na, device=w.device, dtype=torch.bool).view(1, na, na)

        # Contact diagnostics over every ordered pair but the diagonal. The discrete count
        # is the interpretable metric; the ramp over the activation band is what the
        # reward charges, because the count is structurally near-zero at this stiffness.
        rr = r.view(1, -1, 1) + r.view(1, 1, -1)
        touching = ((pair_d < rr) & ~eye).sum(dim=-1).to(w.dtype)
        ramp = ((self._collision_reach - pair_d) / self.contact_margin).clamp(0.0, 1.0)
        prox = torch.where(eye, torch.zeros_like(ramp), ramp).sum(dim=-1)

        goal_rel = w.goals - pos
        dist_to_goal = _safe_sqrt(
            goal_rel[..., 0] * goal_rel[..., 0] + goal_rel[..., 1] * goal_rel[..., 1]
        )
        sdf = self._corner_sdf(pos)
        wall_touch = (sdf < self._wall_reach).to(w.dtype)
        wallramp = ((self._wall_reach - sdf) / self.contact_margin).clamp(0.0, 1.0)

        # --- shaping ----------------------------------------------------------------
        if self._prev_dist is None:
            self._prev_dist = dist_to_goal.detach().clone()
        pos_shaping = (self._prev_dist - dist_to_goal) * self.pos_shaping_factor
        if reset_mask is None:
            self._prev_dist = dist_to_goal.detach().clone()
        else:
            # Rebase only reset envs; the others keep their continuous baseline.
            m = reset_mask.unsqueeze(-1)
            pos_shaping = torch.where(m, torch.zeros_like(pos_shaping), pos_shaping)
            self._prev_dist = torch.where(m, dist_to_goal.detach(), self._prev_dist)

        on_goal = dist_to_goal < self.goal_tolerance
        self._cache = {
            "touching": touching,
            "proximity": prox,
            "wall_touch": wall_touch,
            "wallramp": wallramp,
            "dist_to_goal": dist_to_goal,
            "on_goal": on_goal,
            "frac_on_goal": on_goal.to(w.dtype).sum(dim=-1) * (1.0 / self.n_agents),
            "pos_shaping": pos_shaping,
        }

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        """Fully batched observations ``[n_envs, n_agents, obs_dim]``."""
        if self.fused_active:
            return self.fb["obs"]
        return self._build_obs()

    def _own_terms(self) -> torch.Tensor:
        """The per-agent, never-shared part of the reward.

        Contact ramps, the clock, and the goal-hold bonus. The hold bonus is what makes
        "arrive and stay" beat "arrive and drift" on a task whose ``done`` needs every
        robot on its goal at the *same* step. Its loiter hazard is bounded rather than
        argued away: sitting on a goal forever forgoes ``final_reward`` for a discounted
        hold stream worth ``goal_hold_bonus / (1 - gamma) ~ 0.02 * 86 = 1.7`` against
        ``15``, a 9x margin. Do not raise it past ~0.05 without raising ``final_reward``.
        """
        c = self._cache
        return (
            self.collision_penalty * c["proximity"]
            + self.wall_penalty * c["wallramp"]
            + self.time_penalty
            + self.goal_hold_bonus * c["on_goal"].to(self.world.dtype)
        )

    def _shaping_split(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(own share, team share)`` of the position shaping, per :attr:`shaping_share`.

        The team share is the **mean**, not the sum: a sum would make each agent's reward
        scale with ``n_agents`` and turn three quarters of a four-robot reward into
        teammate noise.
        """
        s = self._cache["pos_shaping"]
        return (1.0 - self.shaping_share) * s, self.shaping_share * s.mean(dim=-1)

    def rewards(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["reward"]
        own_share, team_share = self._shaping_split()
        final = self.final_reward * self._cache["on_goal"].all(dim=-1).to(self.world.dtype)
        return self._own_terms() + own_share + (team_share + final).unsqueeze(-1)

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        own_share, _ = self._shaping_split()
        return self._own_terms()[:, agent_idx] + own_share[:, agent_idx]

    def global_reward(self) -> torch.Tensor:
        _, team_share = self._shaping_split()
        final = self.final_reward * self._cache["on_goal"].all(dim=-1).to(self.world.dtype)
        return team_share + final

    def done(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["done_bool"]
        return self._cache["on_goal"].all(dim=-1)

    def info(self) -> dict[str, Any]:
        """Diagnostics an RL trainer reads through ``("next", "info", <key>)``.

        ``frac_on_goal`` is the scalar success signal: a per-env fraction rather than the
        bare ``all_on_goal`` flag, because on a task where the *last* robot through the
        junction decides the episode, a 0/1 signal spends most of training flat.

        ``collisions`` and ``wall_contacts`` stay the **discrete** counts/flags even
        though the reward now runs off continuous ramps: they are what a training curve is
        read in, and a fraction-of-a-contact is not. ``proximity`` is the ramp the reward
        actually charges, exposed alongside so the two can be compared — ``collisions``
        was a constant zero for an entire 2000-iteration run before the ramp replaced it,
        and nothing in the metrics said so.
        """
        if self.fused_active:
            return {
                "dist_to_goal": self.fb["dist"],
                "on_goal": self.fb["ongoal_bool"],
                "collisions": self.fb["touch"],
                "proximity": self.fb["prox"],
                "wall_contacts": self.fb["wall"],
                "frac_on_goal": self.fb["frac"],
                "all_on_goal": self.fb["done_bool"],
            }
        c = self._cache
        return {
            "dist_to_goal": c["dist_to_goal"],
            "on_goal": c["on_goal"],
            "collisions": c["touching"],
            "proximity": c["proximity"],
            "wall_contacts": c["wall_touch"],
            "frac_on_goal": c["frac_on_goal"],
            "all_on_goal": c["on_goal"].all(dim=-1),
        }
