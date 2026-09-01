"""Fused Warp kernels for the CagingScenario obs/reward + escaping-disc body layer.

Four launches, ordered ``body -> {obs, reward}`` (with ``set_obstacles`` between ``body``
and the next physics step, on the host), plus the masked reset:

* ``caging_body_kernel`` (thread per env) sums the spring-damper reaction force the
  disc receives from every agent (Newton's third law of the same soft contact the agent
  step applies) *and* the escape acceleration — ``-escape_accel`` times the mean of the
  unit vectors pointing from the disc to each agent — then integrates the disc position
  in place (semi-implicit Euler + damping + a ``max_disc_speed`` cap). One disc per env,
  one thread per disc: no cross-thread races, and the per-agent reduction loops
  ``0..A-1``.
* ``caging_obs_kernel`` (thread per (env, agent)) builds the obs row, including the
  agent's own two *local* angular gaps — the bearing distance to the next agent
  counter-clockwise and clockwise around the disc. Those are what make the topological
  objective observable from a single agent's row.
* ``caging_reward_kernel`` (thread per env) computes the largest circular gap between
  agent bearings, the capture latch, and the per-agent reward.
* ``caging_reset_kernel`` (thread per env) draws the disc pose and an annulus of agents
  around it in one masked launch.

**The largest circular gap, two ways.** The torch reference
(:mod:`swarp.scenarios.caging`) sorts the bearings and differences them, wrap-around
included. The kernel cannot: Warp has no dynamically sized register array, so an
insertion sort over a runtime ``n_agents`` has nowhere to live. It uses the equivalent
"nearest neighbour ahead" formulation instead — for every agent ``i``, the gap that
*starts* at ``i`` is ``min_j (b_j - b_i) mod 2*pi`` over the other agents, and the
largest circular gap is the max of those over ``i``. That is O(A^2) with zero storage,
and it makes the wrap-around structural rather than a special case: the modulo is what
carries a bearing past ``+pi`` back round to ``-pi``. The two paths being independent
implementations of the same quantity is the point — that is what
``tests/scenarios/test_caging_fused.py`` tests.

Degenerate agent counts fall out of the same formula. With ``n_agents == 1`` the inner
loop never runs and the gap stays at its ``2*pi`` initializer; with ``n_agents == 2`` the
two gaps sum to ``2*pi`` and the larger one wins. Nothing special-cases them.

Observation cat order (``obs_dim = 10``)::

    [ pos(2), vel(2), disc_rel(2), disc_vel(2), gap_ccw(1), gap_cw(1) ]
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.state import VEC2
from swarp.scenarios.reset_kernels import _as

#: 2*pi, as a Python float; cast to the kernel's dtype at use (``type(x)(TWO_PI)``).
TWO_PI = 6.283185307179586


@wp.kernel
def caging_body_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    n_agents: wp.int32,
    inv_agents: Any,
    agent_radius: Any,
    disc_radius: Any,
    contact_margin: Any,
    contact_k: Any,
    contact_c: Any,
    disc_mass: Any,
    linear_damping: Any,
    escape_accel: Any,
    max_disc_speed: Any,
    dt: Any,
    bound: Any,
    disc_pos: wp.array2d(dtype=Any),
    disc_vel: wp.array2d(dtype=Any),
):
    """Thread per env: contact reaction + escape drift, then integrate the disc.

    Position only. A circle under frictionless normal contact has zero lever arm about
    its own centre (``swarp/core/bodies.py``), so there is no orientation to track and
    none is allocated.
    """
    e = wp.tid()
    q = disc_pos[e, 0]
    pv = disc_vel[e, 0]
    qx = q[0]
    qy = q[1]
    zero = type(qx)(0.0)
    one = type(qx)(1.0)
    eps = type(qx)(1.0e-9)
    reach = agent_radius + disc_radius + contact_margin

    fx = zero
    fy = zero
    # Sum of the unit vectors disc -> agent. Its mean is the "where the crowd is"
    # direction; the disc accelerates the other way (see the escape law below).
    sx = zero
    sy = zero
    for a in range(n_agents):
        relx = qx - pos[e, a][0]  # agent -> disc
        rely = qy - pos[e, a][1]
        dist = wp.sqrt(relx * relx + rely * rely)
        if dist < eps:
            dist = eps
        nhx = relx / dist
        nhy = rely / dist
        overlap = reach - dist
        active = zero
        if overlap > zero:
            active = one
        ov = overlap
        if ov < zero:
            ov = zero
        fmag = contact_k * ov
        rvx = pv[0] - vel[e, a][0]
        rvy = pv[1] - vel[e, a][1]
        vn = rvx * nhx + rvy * nhy
        coeff = (fmag - contact_c * vn) * active
        fx += coeff * nhx
        fy += coeff * nhy
        sx -= nhx  # disc -> agent is -n_hat
        sy -= nhy

    # Escape law: a = -escape_accel * mean_a(unit(disc -> agent)). A ring that closes
    # around the disc cancels the sum and the drift vanishes; a ring with a hole leaves
    # a residual pointing *into* the hole, at up to ``escape_accel``. Smooth in the
    # agent positions, so no argmax and no discontinuity when the widest gap changes
    # hands.
    ex = -escape_accel * sx * inv_agents
    ey = -escape_accel * sy * inv_agents

    decay = one - linear_damping * dt
    new_vx = (pv[0] + (fx / disc_mass + ex) * dt) * decay
    new_vy = (pv[1] + (fy / disc_mass + ey) * dt) * decay
    # Speed cap, applied to the integrated velocity *before* it moves the position, so
    # the clamped velocity is what the state carries and what the next step's damping
    # acts on — the same order the torch reference uses, and the same formulation as
    # shepherding's ``max_sheep_speed``.
    speed = wp.sqrt(new_vx * new_vx + new_vy * new_vy)
    if speed > max_disc_speed:
        vscale = max_disc_speed / speed
        new_vx = new_vx * vscale
        new_vy = new_vy * vscale
    new_px = wp.clamp(qx + new_vx * dt, -bound, bound)
    new_py = wp.clamp(qy + new_vy * dt, -bound, bound)
    disc_vel[e, 0] = type(pv)(new_vx, new_vy)
    disc_pos[e, 0] = type(q)(new_px, new_py)


@wp.kernel
def caging_obs_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    disc_pos: wp.array2d(dtype=Any),
    disc_vel: wp.array2d(dtype=Any),
    n_agents: wp.int32,
    obs: wp.array3d(dtype=Any),
):
    """Thread per (env, agent): pose, disc-relative pose, and the agent's two local gaps.

    ``gap_ccw`` is the bearing distance from this agent to the next one counter-clockwise
    around the disc, ``gap_cw`` the same clockwise. Both are computed as their own
    modulo, rather than one as ``2*pi - other``: the latter cancels catastrophically for
    a bearing pair that wrapped, and the torch oracle takes two independent remainders.
    """
    e, a = wp.tid()
    p = pos[e, a]
    v = vel[e, a]
    q = disc_pos[e, 0]
    dv = disc_vel[e, 0]
    px = p[0]
    py = p[1]
    zero = type(px)(0.0)
    two_pi = type(px)(TWO_PI)
    obs[e, a, 0] = px
    obs[e, a, 1] = py
    obs[e, a, 2] = v[0]
    obs[e, a, 3] = v[1]
    obs[e, a, 4] = q[0] - px
    obs[e, a, 5] = q[1] - py
    obs[e, a, 6] = dv[0]
    obs[e, a, 7] = dv[1]

    bi = wp.atan2(py - q[1], px - q[0])
    ccw = two_pi
    cw = two_pi
    for j in range(n_agents):
        if j != a:
            bj = wp.atan2(pos[e, j][1] - q[1], pos[e, j][0] - q[0])
            d = bj - bi
            if d < zero:
                d += two_pi
            if d < ccw:
                ccw = d
            dc = bi - bj
            if dc < zero:
                dc += two_pi
            if dc < cw:
                cw = dc
    obs[e, a, 8] = ccw
    obs[e, a, 9] = cw


@wp.kernel
def caging_reward_kernel(
    pos: wp.array2d(dtype=Any),
    disc_pos: wp.array2d(dtype=Any),
    n_agents: wp.int32,
    agent_radius: Any,
    disc_radius: Any,
    cage_radius: Any,
    capture_radius: Any,
    gap_threshold: Any,
    gap_reward: Any,
    radius_factor: Any,
    contact_penalty: Any,
    caged_reward: Any,
    spacing_factor: Any,
    even_gap: Any,
    gap_out: wp.array(dtype=Any),
    reward: wp.array2d(dtype=Any),
    caged: wp.array(dtype=wp.uint8),
):
    """Thread per env: largest circular gap, the capture latch, and the reward row.

    Three ``0..A-1`` sweeps, deterministic and atomic-free: the O(A^2) gap reduction
    (see the module docstring for why it is not a sort), the capture check — which needs
    every agent's radius before the bonus is known — and then the per-agent write.

    The dense per-agent spacing shaping rides along in the *first* sweep for free: the
    inner loop's ``gi`` is already this agent's counter-clockwise local gap, so tracking
    the clockwise one alongside it costs two more comparisons and no extra pass. It is
    parked straight into ``reward[e, i]`` because a thread has nowhere else to keep one
    value per agent — Warp has no dynamically sized register array — and the third sweep
    then folds the shared and radial terms on top.
    """
    e = wp.tid()
    q = disc_pos[e, 0]
    qx = q[0]
    qy = q[1]
    zero = type(qx)(0.0)
    two_pi = type(qx)(TWO_PI)

    # Largest circular gap = max over i of the smallest bearing step counter-clockwise
    # from i to any other agent. ``gi`` starts at a full turn, which is exactly the right
    # answer when there is no other agent (n_agents == 1).
    gmax = zero
    for i in range(n_agents):
        bi = wp.atan2(pos[e, i][1] - qy, pos[e, i][0] - qx)
        gi = two_pi
        ci = two_pi
        for j in range(n_agents):
            if j != i:
                bj = wp.atan2(pos[e, j][1] - qy, pos[e, j][0] - qx)
                d = bj - bi
                if d < zero:
                    d += two_pi
                if d < gi:
                    gi = d
                dc = bi - bj
                if dc < zero:
                    dc += two_pi
                if dc < ci:
                    ci = dc
        if gi > gmax:
            gmax = gi
        # Dense per-agent spacing shaping: pull each agent's own two local gaps towards
        # an even share of the circle. Its optimum *is* the evenly spaced ring, i.e. the
        # configuration that minimizes ``gmax`` — so it agrees with the topological
        # objective rather than competing with it, and unlike a max it hands every agent
        # a gradient every step. Squared, not absolute: see the scenario module docstring
        # for why an L1 deviation goes flat for a third of the agents.
        d_ccw = gi - even_gap
        d_cw = ci - even_gap
        reward[e, i] = -spacing_factor * (d_ccw * d_ccw + d_cw * d_cw)
    gap_out[e] = gmax

    # Caged: the ring is closed *and* it is a ring around this disc rather than a
    # coincidental bearing spread from far away.
    flag = wp.uint8(0)
    if gmax < gap_threshold:
        flag = wp.uint8(1)
    for a in range(n_agents):
        dx = pos[e, a][0] - qx
        dy = pos[e, a][1] - qy
        if wp.sqrt(dx * dx + dy * dy) > capture_radius:
            flag = wp.uint8(0)
    bonus = zero
    if flag == wp.uint8(1):
        bonus = caged_reward
    shared = gap_reward * (gap_threshold - gmax) + bonus

    contact_reach = agent_radius + disc_radius
    for a in range(n_agents):
        dx = pos[e, a][0] - qx
        dy = pos[e, a][1] - qy
        d = wp.sqrt(dx * dx + dy * dy)
        ring = wp.abs(d - cage_radius)
        pen = contact_reach - d
        if pen < zero:
            pen = zero
        # ``reward[e, a]`` already holds the spacing term from the first sweep. The
        # parenthesisation mirrors the torch oracle's ``agent_reward + global_reward``
        # split exactly, so the two paths associate these four terms the same way.
        reward[e, a] = (reward[e, a] - radius_factor * ring - contact_penalty * pen) + shared
    caged[e] = flag


@wp.kernel
def caging_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    alim: Any,
    dlim: Any,
    spawn_min: Any,
    spawn_max: Any,
    n_agents: wp.int32,
    pos: Any,
    vel: Any,
    disc_pos: Any,
    disc_vel: Any,
):
    """Masked episode reset: a disc pose, then an annulus of agents around it.

    Agents spawn on a random bearing at a random radius in
    ``[spawn_min, spawn_max]`` about the disc rather than uniformly over the world:
    caging is about *where on the ring* they sit, and a uniform spawn spends most of an
    episode just closing the distance. ``dlim`` is shrunk so the annulus fits inside the
    world, and the final clamp only bites for a world too small to hold it.

    Every ``wp.randf`` is drawn inline — a ``@wp.func`` helper would take the RNG state
    by value and hand back the same point every call (see
    :mod:`swarp.scenarios.reset_kernels`).
    """
    e = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return
    rng = wp.rand_init(seed, e)
    zero = _as(0.0, alim)
    one = _as(1.0, alim)
    two = _as(2.0, alim)
    two_pi = _as(TWO_PI, alim)

    qx = (_as(wp.randf(rng), dlim) * two - one) * dlim
    qy = (_as(wp.randf(rng), dlim) * two - one) * dlim
    disc_pos[e, 0] = wp.vector(qx, qy)
    disc_vel[e, 0] = wp.vector(zero, zero)

    for a in range(n_agents):
        phi = _as(wp.randf(rng), alim) * two_pi
        r = spawn_min + _as(wp.randf(rng), alim) * (spawn_max - spawn_min)
        px = wp.clamp(qx + r * wp.cos(phi), -alim, alim)
        py = wp.clamp(qy + r * wp.sin(phi), -alim, alim)
        pos[e, a] = wp.vector(px, py)
        vel[e, a] = wp.vector(zero, zero)


def _body_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    return [
        a2v,        # pos
        a2v,        # vel
        wp.int32,   # n_agents
        dtype,      # inv_agents
        dtype,      # agent_radius
        dtype,      # disc_radius
        dtype,      # contact_margin
        dtype,      # contact_k
        dtype,      # contact_c
        dtype,      # disc_mass
        dtype,      # linear_damping
        dtype,      # escape_accel
        dtype,      # max_disc_speed
        dtype,      # dt
        dtype,      # bound
        a2v,        # disc_pos
        a2v,        # disc_vel
    ]


def _obs_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    return [
        a2v,                      # pos
        a2v,                      # vel
        a2v,                      # disc_pos
        a2v,                      # disc_vel
        wp.int32,                 # n_agents
        wp.array3d(dtype=dtype),  # obs
    ]


def _reward_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    return [
        a2v,                         # pos
        a2v,                         # disc_pos
        wp.int32,                    # n_agents
        dtype,                       # agent_radius
        dtype,                       # disc_radius
        dtype,                       # cage_radius
        dtype,                       # capture_radius
        dtype,                       # gap_threshold
        dtype,                       # gap_reward
        dtype,                       # radius_factor
        dtype,                       # contact_penalty
        dtype,                       # caged_reward
        dtype,                       # spacing_factor
        dtype,                       # even_gap
        wp.array(dtype=dtype),       # gap_out
        wp.array2d(dtype=dtype),     # reward
        wp.array(dtype=wp.uint8),    # caged
    ]


def _reset_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    return [
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,                  # use_mask
        wp.int32,                  # seed
        dtype,                     # alim
        dtype,                     # dlim
        dtype,                     # spawn_min
        dtype,                     # spawn_max
        wp.int32,                  # n_agents
        a2v,                       # pos
        a2v,                       # vel
        a2v,                       # disc_pos
        a2v,                       # disc_vel
    ]


for _T in (wp.float32, wp.float64):
    register(caging_body_kernel, _T, _body_signature(_T))
    register(caging_obs_kernel, _T, _obs_signature(_T))
    register(caging_reward_kernel, _T, _reward_signature(_T))
    register(caging_reset_kernel, _T, _reset_signature(_T))
