"""Fused Warp kernels for the CagingScenario obs/reward layer and its evasive disc.

Four launches per step, in this order (see
:meth:`~swarp.scenarios.caging.CagingScenario.launch_fused`):

* ``caging_disc_kernel`` (thread per ``(env, 1)``) — the scenario-owned disc. Drift +
  flee + agent contact reaction, one semi-implicit Euler step, clamped to the arena.
  The disc is **not** an engine ``MOVABLE`` body: it is installed as an ordinary
  ``IMMOVABLE`` circle so the agents feel it through ``_static_forces``, and integrated
  here, because the engine's obstacle integrator derives force only from agent reaction
  and has no slot for an external one (a flee potential that reaches beyond contact
  cannot be expressed as a contact law).
* ``caging_obs_kernel`` (thread per ``(env, agent)``) — the O(N^2) bearing work. Every
  per-agent reward input (``gap``, ``band``, ``touch``, ``inband``) and every per-agent
  observation column.
* ``caging_reward_kernel`` (thread per env) — O(N) reductions only: ``gap_max``, the
  soft-max surrogate, the band error, the ``caged`` predicate, the hold counter, the
  reward and the objective vector. It also writes the two **broadcast** observation
  columns (``gap_max``, ``caged``), which is why it runs on every pass rather than only
  on a full one: those columns are per-env reductions that the per-agent obs kernel
  cannot see, and folding them in here keeps the scenario at four kernels instead of
  splitting the observation across two per-agent launches.
* ``caging_reset_kernel`` (thread per env) — masked episode reset, including the
  **per-episode** drift force and evade gain (see the scenario docstring for why those
  cannot be per-step draws).

Max angular gap, without a sort
-------------------------------
With ``theta_i = atan2(p_i - q)`` in ``(-pi, pi]``::

    gap_i   = min over j != i of wrap(theta_j - theta_i),  wrap(x) = x + 2pi if x < 0
    gap_max = max over i of gap_i

The CCW successor of ``i`` is exactly the ``j`` that minimizes the wrapped difference, so
``{gap_i}`` are precisely the N arcs of the circular order and they sum to ``2*pi``. The
alternative — sort the bearings and difference them — is avoided for three reasons: there
is no precedent in this repo for a per-thread local array and Warp has no good one (a
``wp.types.vector(length=N)`` needs a module-scope declaration per ``(N, dtype)`` pair,
i.e. a compile-time cap on ``n_agents``); O(N^2) at the tens of agents this scenario is
written for is nothing, and it is the same trade
:mod:`swarp.scenarios.formation` already makes *for parity*; and a sort-based torch
oracle agrees with this form almost everywhere but **differs exactly at bearing ties and
the 2pi wrap**, which would surface as a parity failure that looks like a kernel bug and
is not one. The torch reference computes the identical min-of-wrapped-differences
expression.

Because ``theta`` is in ``(-pi, pi]`` the raw difference lies in ``(-2pi, 2pi)``, so the
**single conditional add** below is exact and bit-identical to ``torch.where(d < 0, d +
2pi, d)``. Do not "simplify" it to ``fmod``/``remainder``/``floor`` — those round
differently and ``gap_max`` feeds a threshold that ``done`` is compared on exactly.

``best`` is seeded at ``2*pi`` rather than at ``+inf``, which is also how the
``n_agents == 1`` case is defined (an empty loop leaves ``gap = 2*pi``) — the torch
oracle reproduces it by masking the diagonal with ``2*pi`` instead of ``inf``.

The "arcs sum to 2*pi" identity holds for *distinct* bearings. At an exact tie both
duplicates get a zero arc and this form under-reports (a sort would collapse them and
report the real arc). Exact ties have measure zero in floating point, both paths use the
same form so parity is unaffected, and ``tests/scenarios/test_caging_fused.py::
test_gap_matches_sort_oracle`` pins the divergence explicitly so that a future "fix" of
one side to the sort form is caught there rather than as a mystery parity failure.

Observation row (``obs_dim = 13 + 2 * (n_agents - 1)``)::

    [ pos(2), vel(2), disc_pos - pos(2), disc_vel(2), r - cage_radius,
      cos(gap_a), sin(gap_a), gap_max, caged, other agents' rel pos(2*(N-1)) ]

``gap_max`` and ``caged`` are broadcast to every agent on purpose: "caged" is a
*topological* property of the whole team, and without those two columns no agent in a
decentralised policy can tell whether the cage is closed. ``cos/sin(gap_a)`` rather than
"my CCW successor's relative position" because the successor needs an **argmin**, and
torch does not guarantee a lowest-index tie-break on CUDA, so oracle and kernel could
pick different successors at a bearing tie.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.state import VEC2
from swarp.scenarios.reset_kernels import _as

#: Objective columns of ``info()["multiobj_reward"]``, in order. Kept next to the kernel
#: that writes them so the two cannot drift apart.
OBJ_GAP = 0
OBJ_BAND = 1
OBJ_CAGE = 2
OBJ_TIME = 3
OBJ_COLLISION = 4
N_OBJ = 5

#: The two observation columns the *reward* kernel fills (per-env reductions broadcast to
#: every agent), and where the other-agent block starts. The per-agent obs kernel owns
#: columns 0..10; keeping the split as named constants is what stops the two kernels from
#: silently writing the same slot.
C_GAP_MAX = 11
C_CAGED = 12
C_OTHERS = 13


@wp.kernel
def caging_disc_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    drift: wp.array2d(dtype=Any),
    evade: wp.array(dtype=Any),
    n_agents: wp.int32,
    agent_radius: Any,
    disc_radius: Any,
    contact_margin: Any,
    flee_radius: Any,
    flee_k: Any,
    contact_k: Any,
    contact_c: Any,
    disc_mass: Any,
    linear_damping: Any,
    dt: Any,
    bound: Any,
    disc_pos: wp.array2d(dtype=Any),
    disc_vel: wp.array2d(dtype=Any),
):
    """Thread per ``(env, 1)``: accumulate the disc's force, integrate, clamp.

    Force terms, in the order the torch oracle sums them:

    1. ``drift[e, 0]`` — a per-episode constant wander, written by the reset kernel.
    2. a **flee** push away from every agent within ``flee_radius``, profile
       ``(1 - d / flee_radius)^2``. Quadratic, not exponential: it is C^1 at the cutoff,
       so an agent crossing the flee boundary does not step the disc's acceleration.
    3. the agent **contact reaction**, the same spring-damper the engine applies to the
       agents (Newton's third law of it), which is what makes "caged" a dynamical fact
       rather than a reward label — without it the disc slides through the ring and the
       task degenerates into ``formation``.

    Term 3 is the plain explicit spring-damper, whereas the agent side of the same
    contact goes through :func:`swarp.core.collisions.pair_force`'s linearly-implicit,
    depth-saturated law. So the pair is not third-law-exact; the same one-sided
    simplification :mod:`swarp.scenarios.transport` makes, and it is bounded by the same
    damping. ``agent_radius`` is a scenario scalar rather than ``params[a, P_RADIUS]``:
    the cage geometry is derived from the scenario's single agent radius, so a per-agent
    radius would describe a ring that does not exist.

    The clamp is the scenario's own responsibility: nothing in the engine bounds a
    scenario-owned obstacle (``Stepper`` feeds the arena clamp to
    :mod:`swarp.core.bodies`, inside a kernel that early-returns for a non-``MOVABLE``
    obstacle).
    """
    e, k = wp.tid()
    q = disc_pos[e, k]
    dv = disc_vel[e, k]
    zero = type(q[0])(0.0)
    one = type(q[0])(1.0)
    eps = type(q[0])(1.0e-9)
    reach = agent_radius + disc_radius + contact_margin
    ev = evade[e]

    d0 = drift[e, k]
    fx = d0[0]
    fy = d0[1]
    for a in range(n_agents):
        relx = q[0] - pos[e, a][0]
        rely = q[1] - pos[e, a][1]
        dist = wp.sqrt(relx * relx + rely * rely)
        if dist < eps:
            dist = eps
        nhx = relx / dist
        nhy = rely / dist

        if dist < flee_radius:
            w = one - dist / flee_radius
            coeff = ev * flee_k * w * w
            fx += coeff * nhx
            fy += coeff * nhy

        overlap = reach - dist
        if overlap > zero:
            vnx = dv[0] - vel[e, a][0]
            vny = dv[1] - vel[e, a][1]
            vn = vnx * nhx + vny * nhy
            cc = contact_k * overlap - contact_c * vn
            fx += cc * nhx
            fy += cc * nhy

    decay = one - linear_damping * dt
    nvx = (dv[0] + fx / disc_mass * dt) * decay
    nvy = (dv[1] + fy / disc_mass * dt) * decay
    disc_vel[e, k] = type(dv)(nvx, nvy)
    disc_pos[e, k] = type(q)(
        wp.clamp(q[0] + nvx * dt, -bound, bound),
        wp.clamp(q[1] + nvy * dt, -bound, bound),
    )


@wp.kernel
def caging_obs_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    disc_pos: wp.array2d(dtype=Any),
    disc_vel: wp.array2d(dtype=Any),
    reset_mask: wp.array(dtype=wp.uint8),
    n_agents: wp.int32,
    col_dist_sq: Any,
    cage_radius: Any,
    band_half: Any,
    two_pi: Any,
    full_pass: wp.int32,
    obs: wp.array3d(dtype=Any),
    gap: wp.array2d(dtype=Any),
    band: wp.array2d(dtype=Any),
    touching: wp.array2d(dtype=Any),
    inband: wp.array(dtype=wp.uint8, ndim=2),
):
    """Thread per ``(env, agent)``: bearings, angular gap, radial band, contacts, obs row.

    One pass over the other agents does all three O(N) jobs — the wrapped-difference
    minimum, the all-pairs contact count, and the other-agent observation block — so the
    quadratic term is walked once rather than three times.

    The contact count is an all-pairs scan against the static ``2 * agent_radius``
    distance rather than a walk of the neighbor list, for formation's reason: the list is
    truncated at ``max_neighbors``, so reading it would make the two paths disagree by
    construction whenever it overflowed. Squared comparison (no ``sqrt``) so it matches
    the torch oracle's squared broadcast distance bit-for-bit at the boundary.
    """
    e, a = wp.tid()
    if full_pass == 0 and reset_mask[e] == wp.uint8(0):
        # Obs-only auto-reset pass: an env this mask did not select has state identical to
        # what the step pass wrote into every buffer below moments earlier.
        return
    p = pos[e, a]
    v = vel[e, a]
    q = disc_pos[e, 0]
    dv = disc_vel[e, 0]
    zero = type(q[0])(0.0)
    one = type(q[0])(1.0)

    dx = p[0] - q[0]
    dy = p[1] - q[1]
    r = wp.sqrt(dx * dx + dy * dy)
    th = wp.atan2(dy, dx)

    obs[e, a, 0] = p[0]
    obs[e, a, 1] = p[1]
    obs[e, a, 2] = v[0]
    obs[e, a, 3] = v[1]
    obs[e, a, 4] = -dx  # disc_pos - own pos
    obs[e, a, 5] = -dy
    obs[e, a, 6] = dv[0]
    obs[e, a, 7] = dv[1]
    obs[e, a, 8] = r - cage_radius

    best = two_pi
    cnt = zero
    slot = wp.int32(0)
    for b in range(n_agents):
        bx = pos[e, b][0] - p[0]
        by = pos[e, b][1] - p[1]
        if bx * bx + by * by < col_dist_sq:
            cnt += one
        if b != a:
            dd = wp.atan2(pos[e, b][1] - q[1], pos[e, b][0] - q[0]) - th
            # Exact: th, and every other bearing, lie in (-pi, pi], so dd is in
            # (-2pi, 2pi) and one conditional add is the whole wrap.
            if dd < zero:
                dd += two_pi
            if dd < best:
                best = dd
            obs[e, a, C_OTHERS + wp.int32(2) * slot] = bx
            obs[e, a, C_OTHERS + wp.int32(2) * slot + 1] = by
            slot += wp.int32(1)

    obs[e, a, 9] = wp.cos(best)
    obs[e, a, 10] = wp.sin(best)

    gap[e, a] = best
    err = wp.abs(r - cage_radius) - band_half
    band[e, a] = wp.max(err, zero)
    if err <= zero:
        inband[e, a] = wp.uint8(1)
    else:
        inband[e, a] = wp.uint8(0)
    touching[e, a] = cnt - one


@wp.kernel
def caging_reward_kernel(
    gap: wp.array2d(dtype=Any),
    band: wp.array2d(dtype=Any),
    touching: wp.array2d(dtype=Any),
    inband: wp.array(dtype=wp.uint8, ndim=2),
    disc_pos: wp.array2d(dtype=Any),
    reset_mask: wp.array(dtype=wp.uint8),
    n_agents: wp.int32,
    inv_n_agents: Any,
    two_pi: Any,
    beta: Any,
    inv_beta: Any,
    gap_threshold: Any,
    wall_free_radius: Any,
    hold_steps: wp.int32,
    gap_shaping_factor: Any,
    band_shaping_factor: Any,
    cage_reward: Any,
    time_penalty: Any,
    collision_penalty: Any,
    advance_prev: wp.int32,
    full_pass: wp.int32,
    prev_gap: wp.array(dtype=Any),
    prev_band: wp.array(dtype=Any),
    obs: wp.array3d(dtype=Any),
    gap_max_out: wp.array(dtype=Any),
    gap_soft_out: wp.array(dtype=Any),
    band_err_out: wp.array(dtype=Any),
    caged: wp.array(dtype=wp.uint8),
    hold: wp.array(dtype=wp.uint8),
    done: wp.array(dtype=wp.uint8),
    reward: wp.array2d(dtype=Any),
    multiobj: wp.array3d(dtype=Any),
):
    """Thread per env; sequential agent loops (deterministic, no atomics).

    ``gap_soft = 2pi + (1/beta) * log(sum_i exp(beta * (gap_i - 2pi)))`` is the *shaped*
    signal, and ``gap_max`` only the success criterion. That split is the point:
    ``gap_max``'s gradient is supported on one pair of agents at a time, so a team of six
    with one big gap would give signal to two agents and zero to four. Every exponent
    here is ``<= 0`` (because ``gap_i <= 2pi``), so the sum needs no max-subtraction for
    stability and stays exp/log only — no hidden ``max`` reintroducing the sparse
    gradient.

    ``caged`` is a **conjunction**, and ``|q| < wall_free_radius`` is one of its
    conjuncts rather than a penalty term, because once the disc is clamped to the arena
    the trivial optimum is to shove it into a corner where two walls close most escape
    directions for free and ``gap_max`` never has to shrink. A penalty is one more weight
    to tune and can be traded off against; a predicate cannot.

    ``cage_reward`` is paid on **every** caged step, not once at ``done``: ``hold_steps``
    asks the policy to *maintain* the cage, and a terminal-only bonus gives zero signal
    during the hold.

    The hold counter advances on ``advance_prev`` (i.e. on a step) rather than on
    ``full_pass``: a standalone ``reset()`` is a full pass, and advancing there would
    count the spawn instant as a held step, which the torch oracle — whose ``reset_torch``
    does not advance — would not.
    """
    e = wp.tid()
    if full_pass == 0 and reset_mask[e] == wp.uint8(0):
        return
    zero = type(two_pi)(0.0)
    one = type(two_pi)(1.0)

    gmax = zero
    esum = zero
    bsum = zero
    n_in = wp.int32(0)
    for a in range(n_agents):
        g = gap[e, a]
        if g > gmax:
            gmax = g
        esum += wp.exp(beta * (g - two_pi))
        bsum += band[e, a]
        if inband[e, a] == wp.uint8(1):
            n_in += wp.int32(1)
    gsoft = two_pi + inv_beta * wp.log(esum)
    berr = bsum * inv_n_agents

    q = disc_pos[e, 0]
    qn = wp.sqrt(q[0] * q[0] + q[1] * q[1])
    cg = wp.int32(0)
    if gmax < gap_threshold and n_in == n_agents and qn < wall_free_radius:
        cg = wp.int32(1)
    cgf = zero
    if cg == 1:
        cgf = one

    if full_pass == 1:
        # Gated, like every other reward/info buffer: the obs-only auto-reset pass must
        # leave the diagnostics already returned for the transition just taken alone. The
        # two *observation* columns below are the exception — an obs-only pass exists
        # precisely to refresh the observation — and they read ``gmax``/``cgf`` from this
        # thread's locals, not from the buffers.
        caged[e] = wp.uint8(cg)
        gap_max_out[e] = gmax
        gap_soft_out[e] = gsoft
        band_err_out[e] = berr
    for a in range(n_agents):
        obs[e, a, C_GAP_MAX] = gmax
        obs[e, a, C_CAGED] = cgf

    sg = (prev_gap[e] - gsoft) * gap_shaping_factor
    sb = (prev_band[e] - berr) * band_shaping_factor
    if reset_mask[e] == wp.uint8(1):
        prev_gap[e] = gsoft
        prev_band[e] = berr
        sg = zero
        sb = zero
    elif advance_prev == 1:
        prev_gap[e] = gsoft
        prev_band[e] = berr

    h = wp.int32(hold[e])
    if advance_prev == 1:
        if cg == 1:
            if h < 255:
                h += wp.int32(1)
        else:
            h = wp.int32(0)
        hold[e] = wp.uint8(h)

    if full_pass == 1:
        if h >= hold_steps:
            done[e] = wp.uint8(1)
        else:
            done[e] = wp.uint8(0)
        cage_term = cage_reward * cgf
        glob = sg + sb + cage_term + time_penalty
        for a in range(n_agents):
            col = collision_penalty * touching[e, a]
            multiobj[e, a, OBJ_GAP] = sg
            multiobj[e, a, OBJ_BAND] = sb
            multiobj[e, a, OBJ_CAGE] = cage_term
            multiobj[e, a, OBJ_TIME] = time_penalty
            multiobj[e, a, OBJ_COLLISION] = col
            reward[e, a] = glob + col


@wp.kernel
def caging_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    difficulty: wp.array(dtype=Any),
    n_agents: wp.int32,
    agent_lim: Any,
    disc_lim: Any,
    cage_radius: Any,
    ring_ang_jitter: Any,
    ring_rad_jitter: Any,
    drift_mag: Any,
    pos: Any,
    theta: Any,
    vel: Any,
    speed: Any,
    ang_vel: Any,
    disc_pos: Any,
    disc_vel: Any,
    drift: Any,
    evade: wp.array(dtype=Any),
    hold: wp.array(dtype=wp.uint8),
):
    """Masked episode reset: the disc, the per-episode drift/evade draw, and the spawns.

    ``difficulty`` is a one-element **array**, read by pointer, not a scalar. This
    scenario's reset is not folded into the whole-step graph (its seed is drawn
    host-side), but the same discipline giveway documents applies the moment it were: a
    scalar kernel argument is baked into a capture *by value*, so a trainer moving the
    curriculum between batches would silently keep getting the value captured first.
    Reading it out of device memory costs nothing and cannot go stale.

    At ``f = 0`` the disc is inert (no drift, no evade) at the origin and the agents
    spawn **on the cage ring** with small jitter, i.e. already caged on step 1 — so the
    dwell bonus and the terminal are experienced immediately, which is the single reason
    Push-T's curriculum works. At ``f = 1`` the spawn is uniform and the disc drifts and
    flees at full strength. Spawn positions are the linear blend of the two, so the
    curriculum is continuous in ``f``.

    ``drift`` is a per-episode **force vector**, not a per-step draw, and that is
    load-bearing rather than a simplification. A per-step draw inside the fused step
    kernel would have to come either from a scalar seed argument — which a CUDA graph
    bakes in by value, so the "noise" freezes at replay, silently — or from
    ``World.seed_state``, which the torch oracle cannot reproduce, destroying parity by
    construction. Written once here, both paths read the identical numbers.

    ``hold`` is zeroed here rather than in the reward kernel, so it is cleared on every
    flavour of reset including the obs-only auto-reset pass (which does not write the
    reward outputs at all).
    """
    e = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return
    rng = wp.rand_init(seed, e)
    zero = _as(0.0, agent_lim)
    one = _as(1.0, agent_lim)
    two = _as(2.0, agent_lim)
    two_pi = _as(6.28318530717959, agent_lim)

    f = difficulty[0]

    # ---- the disc: at the origin when inert, uniform inside the wall-free zone at f = 1
    qx = (_as(wp.randf(rng), agent_lim) * two - one) * disc_lim * f
    qy = (_as(wp.randf(rng), agent_lim) * two - one) * disc_lim * f
    disc_pos[e, 0] = wp.vector(qx, qy)
    disc_vel[e, 0] = wp.vector(zero, zero)

    # ---- per-episode wander: a constant force, so under (1 - lambda*dt) decay the disc
    # settles at drift_mag / (mass * lambda) — which is how the caller sizes drift_mag.
    phi = _as(wp.randf(rng), agent_lim) * two_pi
    mag = drift_mag * f
    drift[e, 0] = wp.vector(mag * wp.cos(phi), mag * wp.sin(phi))
    evade[e] = f
    hold[e] = wp.uint8(0)

    inv_n = one / _as(wp.float32(n_agents), agent_lim)
    for a in range(n_agents):
        # Draw inline, never through a @wp.func: Warp passes the RNG state by value, so a
        # helper would advance a local copy and hand back the same number every call.
        ang = _as(wp.float32(a), agent_lim) * two_pi * inv_n + (
            _as(wp.randf(rng), agent_lim) * two - one
        ) * ring_ang_jitter
        rad = cage_radius + (_as(wp.randf(rng), agent_lim) * two - one) * ring_rad_jitter
        rx = qx + rad * wp.cos(ang)
        ry = qy + rad * wp.sin(ang)
        ux = (_as(wp.randf(rng), agent_lim) * two - one) * agent_lim
        uy = (_as(wp.randf(rng), agent_lim) * two - one) * agent_lim
        px = (one - f) * rx + f * ux
        py = (one - f) * ry + f * uy
        pos[e, a] = wp.vector(
            wp.clamp(px, -agent_lim, agent_lim), wp.clamp(py, -agent_lim, agent_lim)
        )
        theta[e, a] = zero
        vel[e, a] = wp.vector(zero, zero)
        speed[e, a] = zero
        ang_vel[e, a] = zero


def _disc_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    return [
        a2v,  # pos
        a2v,  # vel
        a2v,  # drift
        wp.array(dtype=dtype),  # evade
        wp.int32,  # n_agents
        dtype,  # agent_radius
        dtype,  # disc_radius
        dtype,  # contact_margin
        dtype,  # flee_radius
        dtype,  # flee_k
        dtype,  # contact_k
        dtype,  # contact_c
        dtype,  # disc_mass
        dtype,  # linear_damping
        dtype,  # dt
        dtype,  # bound
        a2v,  # disc_pos
        a2v,  # disc_vel
    ]


def _obs_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    a2s = wp.array2d(dtype=dtype)
    return [
        a2v,  # pos
        a2v,  # vel
        a2v,  # disc_pos
        a2v,  # disc_vel
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # n_agents
        dtype,  # col_dist_sq
        dtype,  # cage_radius
        dtype,  # band_half
        dtype,  # two_pi
        wp.int32,  # full_pass
        wp.array3d(dtype=dtype),  # obs
        a2s,  # gap
        a2s,  # band
        a2s,  # touching
        wp.array(dtype=wp.uint8, ndim=2),  # inband
    ]


def _reward_signature(dtype) -> list:
    a1s = wp.array(dtype=dtype)
    a2s = wp.array2d(dtype=dtype)
    return [
        a2s,  # gap
        a2s,  # band
        a2s,  # touching
        wp.array(dtype=wp.uint8, ndim=2),  # inband
        wp.array2d(dtype=VEC2[dtype]),  # disc_pos
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # n_agents
        dtype,  # inv_n_agents
        dtype,  # two_pi
        dtype,  # beta
        dtype,  # inv_beta
        dtype,  # gap_threshold
        dtype,  # wall_free_radius
        wp.int32,  # hold_steps
        dtype,  # gap_shaping_factor
        dtype,  # band_shaping_factor
        dtype,  # cage_reward
        dtype,  # time_penalty
        dtype,  # collision_penalty
        wp.int32,  # advance_prev
        wp.int32,  # full_pass
        a1s,  # prev_gap
        a1s,  # prev_band
        wp.array3d(dtype=dtype),  # obs
        a1s,  # gap_max_out
        a1s,  # gap_soft_out
        a1s,  # band_err_out
        wp.array(dtype=wp.uint8),  # caged
        wp.array(dtype=wp.uint8),  # hold
        wp.array(dtype=wp.uint8),  # done
        a2s,  # reward
        wp.array3d(dtype=dtype),  # multiobj
    ]


def _reset_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    a2s = wp.array2d(dtype=dtype)
    return [
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # use_mask
        wp.int32,  # seed
        wp.array(dtype=dtype),  # difficulty (by pointer, never a baked scalar)
        wp.int32,  # n_agents
        dtype,  # agent_lim
        dtype,  # disc_lim
        dtype,  # cage_radius
        dtype,  # ring_ang_jitter
        dtype,  # ring_rad_jitter
        dtype,  # drift_mag
        a2v,  # pos
        a2s,  # theta
        a2v,  # vel
        a2s,  # speed
        a2s,  # ang_vel
        a2v,  # disc_pos
        a2v,  # disc_vel
        a2v,  # drift
        wp.array(dtype=dtype),  # evade
        wp.array(dtype=wp.uint8),  # hold
    ]


for _T in (wp.float32, wp.float64):
    register(caging_disc_kernel, _T, _disc_signature(_T))
    register(caging_obs_kernel, _T, _obs_signature(_T))
    register(caging_reward_kernel, _T, _reward_signature(_T))
    register(caging_reset_kernel, _T, _reset_signature(_T))
