"""Fused Warp kernels for the ShepherdingScenario obs/reward + the sheep flock layer.

Four launches per step, ordered ``force -> body -> {obs, reward}`` (with
``set_obstacles`` between ``body`` and the next physics step, on the host):

* ``shepherding_force_kernel`` (thread per env,sheep) sums the three force terms that
  act on a sheep — shepherd contact reaction, shepherd *flee* repulsion (whose **sum** is
  magnitude-capped at ``flee_gain``, so a converging pack cannot stack panic), sheep-sheep
  separation, plus the mild flock cohesion — into ``force``. It is a **separate launch
  from the integration** for one reason: separation and cohesion read the *other* sheep's
  positions, so a kernel that both read neighbours and wrote its own pose in place would
  race (sheep ``i``'s thread reading ``sheep_pos[e, j]`` while sheep ``j``'s thread writes
  it). Splitting force from integration makes the read set frozen for the whole pass —
  no atomics, no double buffer, no pointer swap a captured graph would have to notice.
* ``shepherding_body_kernel`` (thread per env,sheep) integrates the sheep from that force
  (semi-implicit Euler + linear damping, then a ``max_sheep_speed`` cap), mutating
  ``sheep_pos``/``sheep_vel`` in place.
* ``shepherding_obs_kernel`` (thread per env,agent) builds the obs row.
* ``shepherding_reward_kernel`` (thread per env) reduces the per-sheep pen shaping, the
  penned bonus and the flock-spread penalty, using the navigation ``prev_dist`` /
  ``reset_hit`` / ``advance_prev`` / ``full_pass`` machinery.

The torch implementation in :mod:`swarp.scenarios.shepherding` stays the reference (and
the differentiable path — sheep gradients flow through the plain-torch flock physics).
The fused kernels are the no-grad fast path; because they sum the per-shepherd and
per-sheep contributions in a different order than torch's ``.sum(dim=...)``, the two
sheep trajectories can differ at the ulp scale. Unlike transport that difference is
*amplified* rather than only damped: the flee term is active at a distance, so a tiny
positional difference feeds back through a force the sheep would not otherwise feel,
which is why the scenario declares a slightly looser ``parity_rtol``/``parity_atol``
(validated allclose, not bit-exact).

Observation cat order (``obs_dim = 6 + 4 * n_sheep``)::

    [ pos(2), vel(2), pen_rel(2), sheep_rel(2*K), sheep_to_pen(2*K) ]
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.state import VEC2
from swarp.scenarios.reset_kernels import _as


@wp.kernel
def shepherding_force_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    sheep_pos: wp.array2d(dtype=Any),
    sheep_vel: wp.array2d(dtype=Any),
    n_agents: wp.int32,
    n_sheep: wp.int32,
    agent_radius: Any,
    sheep_radius: Any,
    contact_margin: Any,
    contact_k: Any,
    contact_c: Any,
    flee_radius: Any,
    flee_gain: Any,
    sep_radius: Any,
    sep_gain: Any,
    cohesion_gain: Any,
    force: wp.array2d(dtype=Any),
):
    """Thread per (env, sheep): the net force on one sheep, read-only in the flock.

    The four terms are accumulated into *separate* running sums and added once at the
    end, in the same grouping the torch reference uses
    (``contact.sum(A) + cap(flee.sum(A)) + sep.sum(K) + cohesion``). Interleaving them
    into a single accumulator would be a gratuitous reassociation against the parity
    oracle — and the flee sum has to stand alone anyway, since it is the sum that gets
    magnitude-capped.
    """
    e, k = wp.tid()
    q = sheep_pos[e, k]
    pv = sheep_vel[e, k]
    qx = q[0]
    qy = q[1]
    zero = type(qx)(0.0)
    one = type(qx)(1.0)
    eps = type(qx)(1.0e-9)
    reach = agent_radius + sheep_radius + contact_margin

    # --- shepherd terms: contact (spring-damper) and flee (soft, acts at a distance)
    fcx = zero  # contact
    fcy = zero
    ffx = zero  # flee
    ffy = zero
    for a in range(n_agents):
        relx = qx - pos[e, a][0]
        rely = qy - pos[e, a][1]
        dist = wp.sqrt(relx * relx + rely * rely)
        if dist < eps:
            dist = eps
        nhx = relx / dist
        nhy = rely / dist
        overlap = reach - dist
        if overlap > zero:
            fmag = contact_k * overlap
            rvx = pv[0] - vel[e, a][0]
            rvy = pv[1] - vel[e, a][1]
            vn = rvx * nhx + rvy * nhy
            coeff = fmag - contact_c * vn
            fcx += coeff * nhx
            fcy += coeff * nhy
        if dist < flee_radius:
            # linear falloff: full ``flee_gain`` on top of the shepherd, zero at the
            # flee radius. This is the whole reactive-environment story — it is active
            # far outside contact range, so charging the flock scatters it.
            fmag = flee_gain * (one - dist / flee_radius)
            ffx += fmag * nhx
            ffy += fmag * nhy
    # Cap the *summed* flee force at flee_gain, so N converging shepherds cannot stack
    # N times the panic into one sheep. Capping the sum rather than each term is what
    # keeps the direction meaningful when the flock is surrounded: the terms still
    # compose into "away from where the shepherds are", they just cannot outrun the
    # shepherds doing the herding. See the scenario's flee_gain comment for the
    # terminal-speed arithmetic this pins.
    flee_mag = wp.sqrt(ffx * ffx + ffy * ffy)
    if flee_mag > flee_gain:
        scale = flee_gain / flee_mag
        ffx = ffx * scale
        ffy = ffy * scale

    # --- flock terms: separation from every other sheep, cohesion to the flock centroid
    fsx = zero
    fsy = zero
    cx = zero
    cy = zero
    for j in range(n_sheep):
        ox = sheep_pos[e, j][0]
        oy = sheep_pos[e, j][1]
        cx += ox
        cy += oy
        if j != k:
            relx = qx - ox
            rely = qy - oy
            d = wp.sqrt(relx * relx + rely * rely)
            if d < sep_radius:
                dc = d
                if dc < eps:
                    dc = eps
                fmag = sep_gain * (one - d / sep_radius)
                fsx += fmag * relx / dc
                fsy += fmag * rely / dc
    inv_k = one / type(qx)(n_sheep)
    fkx = cohesion_gain * (cx * inv_k - qx)
    fky = cohesion_gain * (cy * inv_k - qy)

    force[e, k] = type(q)((fcx + ffx) + fsx + fkx, (fcy + ffy) + fsy + fky)


@wp.kernel
def shepherding_body_kernel(
    force: wp.array2d(dtype=Any),
    sheep_mass: Any,
    linear_damping: Any,
    max_sheep_speed: Any,
    dt: Any,
    bound: Any,
    sheep_pos: wp.array2d(dtype=Any),
    sheep_vel: wp.array2d(dtype=Any),
):
    """Thread per (env, sheep): semi-implicit Euler + damping + speed cap, in place.

    Position only — a frictionless normal contact on a disc has zero lever arm, so a
    circular sheep can never acquire angular velocity (``swarp/core/bodies.py``). There
    is no ``theta``/``ang_vel`` to carry, and none is allocated.

    The speed cap is applied to the integrated velocity *before* it moves the position,
    so the clamped velocity is what the state carries and what the next step's damping
    acts on — the same order the torch reference uses.
    """
    e, k = wp.tid()
    q = sheep_pos[e, k]
    pv = sheep_vel[e, k]
    f = force[e, k]
    one = type(q[0])(1.0)
    decay = one - linear_damping * dt
    new_vx = (pv[0] + f[0] / sheep_mass * dt) * decay
    new_vy = (pv[1] + f[1] / sheep_mass * dt) * decay
    speed = wp.sqrt(new_vx * new_vx + new_vy * new_vy)
    if speed > max_sheep_speed:
        vscale = max_sheep_speed / speed
        new_vx = new_vx * vscale
        new_vy = new_vy * vscale
    new_px = wp.clamp(q[0] + new_vx * dt, -bound, bound)
    new_py = wp.clamp(q[1] + new_vy * dt, -bound, bound)
    sheep_vel[e, k] = type(pv)(new_vx, new_vy)
    sheep_pos[e, k] = type(q)(new_px, new_py)


@wp.kernel
def shepherding_obs_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    sheep_pos: wp.array2d(dtype=Any),
    pen: wp.array(dtype=Any),
    n_sheep: wp.int32,
    obs: wp.array3d(dtype=Any),
):
    """Thread per (env, agent): obs row (agent pose, pen-relative, sheep, sheep->pen)."""
    e, a = wp.tid()
    p = pos[e, a]
    v = vel[e, a]
    px = p[0]
    py = p[1]
    c = pen[e]
    obs[e, a, 0] = px
    obs[e, a, 1] = py
    obs[e, a, 2] = v[0]
    obs[e, a, 3] = v[1]
    obs[e, a, 4] = c[0] - px
    obs[e, a, 5] = c[1] - py
    p_base = wp.int32(6) + wp.int32(2) * n_sheep
    for k in range(n_sheep):
        q = sheep_pos[e, k]
        obs[e, a, wp.int32(6) + wp.int32(2) * k] = q[0] - px
        obs[e, a, wp.int32(6) + wp.int32(2) * k + 1] = q[1] - py
        obs[e, a, p_base + wp.int32(2) * k] = c[0] - q[0]
        obs[e, a, p_base + wp.int32(2) * k + 1] = c[1] - q[1]


@wp.kernel
def shepherding_reward_kernel(
    sheep_pos: wp.array2d(dtype=Any),
    pen: wp.array(dtype=Any),
    reset_mask: wp.array(dtype=wp.uint8),
    n_agents: wp.int32,
    n_sheep: wp.int32,
    pos_shaping_factor: Any,
    pen_radius: Any,
    pen_reward: Any,
    scatter_penalty: Any,
    advance_prev: wp.int32,
    full_pass: wp.int32,
    prev_dist: wp.array2d(dtype=Any),
    reward: wp.array2d(dtype=Any),
    done: wp.array(dtype=wp.uint8),
    dist_out: wp.array(dtype=Any),
    penned_out: wp.array(dtype=Any),
):
    """Thread per env; per-sheep shaping + penned bonus + spread penalty (no atomics)."""
    e = wp.tid()
    zero = type(pos_shaping_factor)(0.0)
    one = type(pos_shaping_factor)(1.0)
    inv_k = one / type(pos_shaping_factor)(n_sheep)
    c = pen[e]
    shaping_sum = zero
    dist_sum = zero
    penned = zero
    cx = zero
    cy = zero
    all_penned = wp.uint8(1)
    reset_hit = wp.int32(reset_mask[e])
    for k in range(n_sheep):
        qx = sheep_pos[e, k][0]
        qy = sheep_pos[e, k][1]
        cx += qx
        cy += qy
        dx = qx - c[0]
        dy = qy - c[1]
        d = wp.sqrt(dx * dx + dy * dy)
        prev = prev_dist[e, k]
        ps = (prev - d) * pos_shaping_factor
        if reset_hit == 1:
            prev_dist[e, k] = d
        else:
            if advance_prev == 1:
                prev_dist[e, k] = d
            shaping_sum += ps
        dist_sum += d
        if d < pen_radius:
            penned += one
        else:
            all_penned = wp.uint8(0)
    if full_pass == 1:
        # flock spread: mean distance to the flock centroid. A second pass, because the
        # centroid is only known once the first one has finished.
        cx = cx * inv_k
        cy = cy * inv_k
        spread = zero
        for k in range(n_sheep):
            dx = sheep_pos[e, k][0] - cx
            dy = sheep_pos[e, k][1] - cy
            spread += wp.sqrt(dx * dx + dy * dy)
        spread = spread * inv_k
        global_r = shaping_sum + pen_reward * penned - scatter_penalty * spread
        for a in range(n_agents):
            reward[e, a] = global_r
        done[e] = all_penned
        dist_out[e] = dist_sum * inv_k
        penned_out[e] = penned * inv_k


@wp.kernel
def shepherding_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    lim: Any,
    plim: Any,
    sbound: Any,
    min_r: Any,
    span: Any,
    two_pi: Any,
    n_agents: wp.int32,
    n_sheep: wp.int32,
    pos: Any,
    vel: Any,
    sheep_pos: Any,
    sheep_vel: Any,
    pen: Any,
    goals: Any,
):
    """Masked episode reset: uniform shepherd spawns, a fresh pen, sheep on a ring.

    The sheep are drawn in polar coordinates *about the pen* rather than uniformly in the
    box, so an episode does not start already solved: every sheep begins at least
    ``min_r`` (> the pen radius) away from the pen centre. Randomness is drawn inline —
    a ``@wp.func`` helper would hand back the same point every call (see
    :mod:`swarp.scenarios.reset_kernels`).
    """
    e = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return
    rng = wp.rand_init(seed, e)
    zero = _as(0.0, lim)
    two = _as(2.0, lim)
    one = _as(1.0, lim)

    for a in range(n_agents):
        px = (_as(wp.randf(rng), lim) * two - one) * lim
        py = (_as(wp.randf(rng), lim) * two - one) * lim
        pos[e, a] = wp.vector(px, py)
        vel[e, a] = wp.vector(zero, zero)

    cx = (_as(wp.randf(rng), plim) * two - one) * plim
    cy = (_as(wp.randf(rng), plim) * two - one) * plim
    pen[e] = wp.vector(cx, cy)
    # ``world.goals`` mirrors the pen centre into every shepherd's row. Render-side
    # only — nothing in this scenario's obs/reward reads it, ``pen`` stays the
    # authoritative copy the kernels take — but it is what makes the viewer's goal
    # overlay mark the pen without a renderer change, and it is semantically honest:
    # the pen IS the shepherds' one shared goal. Written in its own loop *after* the
    # pen draw rather than folded into the spawn loop above, so the RNG stream (and
    # therefore every spawn this kernel produces) is bit-for-bit what it was before.
    for a in range(n_agents):
        goals[e, a] = wp.vector(cx, cy)

    for k in range(n_sheep):
        ang = _as(wp.randf(rng), lim) * two_pi
        r = min_r + _as(wp.randf(rng), lim) * span
        sx = wp.clamp(cx + r * wp.cos(ang), -sbound, sbound)
        sy = wp.clamp(cy + r * wp.sin(ang), -sbound, sbound)
        sheep_pos[e, k] = wp.vector(sx, sy)
        sheep_vel[e, k] = wp.vector(zero, zero)


def _force_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    return [
        a2v,  # pos
        a2v,  # vel
        a2v,  # sheep_pos
        a2v,  # sheep_vel
        wp.int32,  # n_agents
        wp.int32,  # n_sheep
        dtype,  # agent_radius
        dtype,  # sheep_radius
        dtype,  # contact_margin
        dtype,  # contact_k
        dtype,  # contact_c
        dtype,  # flee_radius
        dtype,  # flee_gain
        dtype,  # sep_radius
        dtype,  # sep_gain
        dtype,  # cohesion_gain
        a2v,  # force
    ]


def _body_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    return [
        a2v,  # force
        dtype,  # sheep_mass
        dtype,  # linear_damping
        dtype,  # max_sheep_speed
        dtype,  # dt
        dtype,  # bound
        a2v,  # sheep_pos
        a2v,  # sheep_vel
    ]


def _obs_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    return [
        wp.array2d(dtype=vec2),  # pos
        wp.array2d(dtype=vec2),  # vel
        wp.array2d(dtype=vec2),  # sheep_pos
        wp.array(dtype=vec2),  # pen
        wp.int32,  # n_sheep
        wp.array3d(dtype=dtype),  # obs
    ]


def _reward_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    return [
        wp.array2d(dtype=vec2),  # sheep_pos
        wp.array(dtype=vec2),  # pen
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # n_agents
        wp.int32,  # n_sheep
        dtype,  # pos_shaping_factor
        dtype,  # pen_radius
        dtype,  # pen_reward
        dtype,  # scatter_penalty
        wp.int32,  # advance_prev
        wp.int32,  # full_pass
        wp.array2d(dtype=dtype),  # prev_dist
        wp.array2d(dtype=dtype),  # reward
        wp.array(dtype=wp.uint8),  # done
        wp.array(dtype=dtype),  # dist_out
        wp.array(dtype=dtype),  # penned_out
    ]


def _reset_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a1v = wp.array(dtype=vec2)
    a2v = wp.array2d(dtype=vec2)
    return [
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # use_mask
        wp.int32,  # seed
        dtype,  # lim
        dtype,  # plim
        dtype,  # sbound
        dtype,  # min_r
        dtype,  # span
        dtype,  # two_pi
        wp.int32,  # n_agents
        wp.int32,  # n_sheep
        a2v,  # pos
        a2v,  # vel
        a2v,  # sheep_pos
        a2v,  # sheep_vel
        a1v,  # pen
        a2v,  # goals
    ]


for _T in (wp.float32, wp.float64):
    register(shepherding_force_kernel, _T, _force_signature(_T))
    register(shepherding_body_kernel, _T, _body_signature(_T))
    register(shepherding_obs_kernel, _T, _obs_signature(_T))
    register(shepherding_reward_kernel, _T, _reward_signature(_T))
    register(shepherding_reset_kernel, _T, _reset_signature(_T))
