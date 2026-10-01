"""Fused Warp kernels for the ShepherdingScenario: sheep dynamics + obs/reward/reset.

Five launches, ordered ``force -> integrate -> {obs, reward}`` (with ``set_obstacles``
between ``integrate`` and the next physics step, on the host):

* ``shepherding_force_kernel`` (thread per ``(env, sheep)``) sums the per-episode drift,
  the flee force from every dog, the spring-damper contact reaction from those same dogs,
  the separation force from the other sheep and the cohesion pull toward the centroid of
  the *other* sheep, and writes the net force into a scratch buffer.
* ``shepherding_integrate_kernel`` (thread per ``(env, sheep)``) applies one semi-implicit
  Euler step with linear damping, the speed clamp and the arena clamp.
* ``shepherding_obs_kernel`` (thread per ``(env, agent)``) builds the obs row and the
  per-agent dog-dog contact count the reward reduces.
* ``shepherding_reward_kernel`` (thread per env) does the O(``n_sheep``) reductions —
  centroid, mean distance to the pen, RMS spread, in-pen count — and the shaping,
  bonus and penalty assembly.
* ``shepherding_reset_kernel`` (thread per env) draws a whole episode: dog spawns, pen
  centre, sheep spawns, the per-episode drift vectors and the evade gain.

**Why the sheep step is two kernels and not one.** Every sheep reads the positions of
every *other* sheep (separation, cohesion), so a single kernel that also wrote
``sheep_pos[e, s]`` would have threads reading a buffer their siblings are concurrently
writing — a race whose outcome depends on scheduling, which would break both determinism
and parity. Splitting force accumulation (read-only over ``sheep_pos``) from integration
(each thread touches only its own slot) removes the race without a ping-pong buffer,
which would have meant swapping pointers the cached Warp handles and the captured graph
are keyed on. ``transport_body_kernel`` gets away with one kernel only because packages
do not interact with each other.

The contact term is written out **explicitly** here rather than calling the engine's
``pair_force`` ``@wp.func``: that carries the implicit ``damp_denom`` and the
``max_overlap`` ``tanh`` saturation, and the torch oracle in
:mod:`swarp.scenarios.shepherding` has to be an *independent* implementation of the same
quantity — reproducing the implicit denominator in torch is parity surface bought for no
behavioural gain. ``transport_kernels`` made exactly the same call.

Observation row (``obs_dim = 10 + 4 * n_sheep + 2 * (n_agents - 1)``)::

    [ pos(2), vel(2), pen - pos(2), sheep_centroid - pos(2), spread, pen_fraction,
      per sheep: (q_s - pos)(2), sheep_vel(2),
      other dogs' rel pos(2 * (n_agents - 1)) ]
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.state import VEC2
from swarp.scenarios.reset_kernels import _as

#: Objective columns of ``info()["multiobj_reward"]``, in order. Kept next to the kernel
#: that writes them so the two cannot drift apart.
OBJ_POS = 0
OBJ_SPREAD = 1
OBJ_PEN = 2
OBJ_TIME = 3
OBJ_COLLISION = 4
N_OBJ = 5


@wp.kernel
def shepherding_force_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    sheep_pos: wp.array2d(dtype=Any),
    sheep_vel: wp.array2d(dtype=Any),
    drift: wp.array2d(dtype=Any),
    evade: wp.array(dtype=Any),
    n_agents: wp.int32,
    n_sheep: wp.int32,
    reach: Any,
    contact_k: Any,
    contact_c: Any,
    flee_radius: Any,
    flee_k: Any,
    sep_radius: Any,
    sep_k: Any,
    coh_k: Any,
    inv_others: Any,
    force: wp.array2d(dtype=Any),
):
    """Thread per ``(env, sheep)``: the whole flocking + flee + contact force law.

    Read-only over ``sheep_pos``/``sheep_vel`` — see the module docstring for why that
    matters. ``evade`` and ``drift`` are per-episode *buffers*, not scalars: a scalar
    kernel argument is baked into the whole-step CUDA graph **by value**, so a curriculum
    that moved the flee gain between batches would silently keep getting the value the
    graph was captured with (and per-step in-kernel RNG would be unreproducible by the
    torch oracle at any tolerance).
    """
    e, s = wp.tid()
    q = sheep_pos[e, s]
    vs = sheep_vel[e, s]
    qx = q[0]
    qy = q[1]
    zero = type(qx)(0.0)
    one = type(qx)(1.0)
    eps = type(qx)(1.0e-9)

    d0 = drift[e, s]
    fx = d0[0]
    fy = d0[1]
    ev = evade[e]

    for a in range(n_agents):
        pa = pos[e, a]
        relx = qx - pa[0]
        rely = qy - pa[1]
        dist = wp.sqrt(relx * relx + rely * rely)
        if dist < eps:
            dist = eps
        nhx = relx / dist
        nhy = rely / dist
        coef = zero
        if dist < flee_radius:
            t = one - dist / flee_radius
            coef += ev * flee_k * t * t
        overlap = reach - dist
        if overlap > zero:
            va = vel[e, a]
            vn = (vs[0] - va[0]) * nhx + (vs[1] - va[1]) * nhy
            coef += contact_k * overlap - contact_c * vn
        fx += coef * nhx
        fy += coef * nhy

    # Separation from, and cohesion toward, the other sheep — one pass over the flock.
    cx = zero
    cy = zero
    for t2 in range(n_sheep):
        if t2 != s:
            qt = sheep_pos[e, t2]
            relx = qx - qt[0]
            rely = qy - qt[1]
            dist = wp.sqrt(relx * relx + rely * rely)
            if dist < eps:
                dist = eps
            if dist < sep_radius:
                u = one - dist / sep_radius
                c2 = sep_k * u * u
                fx += c2 * relx / dist
                fy += c2 * rely / dist
            cx += qt[0]
            cy += qt[1]
    if n_sheep > 1:
        fx += coh_k * (cx * inv_others - qx)
        fy += coh_k * (cy * inv_others - qy)

    force[e, s] = type(q)(fx, fy)


@wp.kernel
def shepherding_integrate_kernel(
    force: wp.array2d(dtype=Any),
    sheep_mass: Any,
    linear_damping: Any,
    dt: Any,
    max_speed: Any,
    bound: Any,
    sheep_pos: wp.array2d(dtype=Any),
    sheep_vel: wp.array2d(dtype=Any),
):
    """Thread per ``(env, sheep)``: semi-implicit Euler, speed clamp, arena clamp.

    The arena clamp is the scenario's own job: nothing in the engine bounds a
    scenario-owned obstacle (``bodies.py``'s bounds handling early-returns for anything
    that is not ``MOVABLE``), so without it the sheep drift out of the world and the
    observation goes non-finite a few hundred steps in.

    The speed clamp is what makes the sheep *catchable*, and it is written here exactly as
    the torch oracle writes it (``scale = min(1, v_max / |v|)`` with ``|v|`` floored at
    eps) — two algebraically equal forms that round differently are a parity trap.
    """
    e, s = wp.tid()
    f = force[e, s]
    v = sheep_vel[e, s]
    q = sheep_pos[e, s]
    one = type(bound)(1.0)
    eps = type(bound)(1.0e-9)
    decay = one - linear_damping * dt
    nvx = (v[0] + f[0] / sheep_mass * dt) * decay
    nvy = (v[1] + f[1] / sheep_mass * dt) * decay
    sp = wp.sqrt(nvx * nvx + nvy * nvy)
    if sp < eps:
        sp = eps
    scale = max_speed / sp
    if scale > one:
        scale = one
    nvx *= scale
    nvy *= scale
    npx = wp.clamp(q[0] + nvx * dt, -bound, bound)
    npy = wp.clamp(q[1] + nvy * dt, -bound, bound)
    sheep_vel[e, s] = type(v)(nvx, nvy)
    sheep_pos[e, s] = type(q)(npx, npy)


@wp.kernel
def shepherding_obs_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    sheep_pos: wp.array2d(dtype=Any),
    sheep_vel: wp.array2d(dtype=Any),
    pen: wp.array(dtype=Any),
    reset_mask: wp.array(dtype=wp.uint8),
    n_agents: wp.int32,
    n_sheep: wp.int32,
    inv_n_sheep: Any,
    pen_radius: Any,
    col_dist_sq: Any,
    full_pass: wp.int32,
    obs: wp.array3d(dtype=Any),
    touching: wp.array2d(dtype=Any),
):
    """Thread per ``(env, agent)``: the obs row plus the dog-dog contact count.

    The flock aggregates (centroid, RMS spread, in-pen fraction) are recomputed here
    rather than read from the reward kernel's outputs: the reward kernel does not run on
    an obs-only auto-reset pass, so depending on its buffers would leave the observation
    one step stale exactly on the episode boundary. ``n_sheep`` is small and the loops are
    O(K), so the duplication costs a handful of FLOPs per thread.

    ``full_pass == 0`` is the obs-only auto-reset pass: an env the mask did not select has
    state identical to what the step pass wrote moments earlier (the sheep kernels do not
    run on a reset pass either), so it is skipped outright, and ``touching`` — already
    returned for the transition just taken — is left alone.
    """
    e, a = wp.tid()
    if full_pass == 0 and reset_mask[e] == wp.uint8(0):
        return
    p = pos[e, a]
    v = vel[e, a]
    px = p[0]
    py = p[1]
    zero = type(px)(0.0)
    one = type(px)(1.0)
    obs[e, a, 0] = px
    obs[e, a, 1] = py
    obs[e, a, 2] = v[0]
    obs[e, a, 3] = v[1]
    g = pen[e]
    obs[e, a, 4] = g[0] - px
    obs[e, a, 5] = g[1] - py

    base = wp.int32(10)
    sx = zero
    sy = zero
    n_in = zero
    for s in range(n_sheep):
        q = sheep_pos[e, s]
        vq = sheep_vel[e, s]
        sx += q[0]
        sy += q[1]
        dx = q[0] - g[0]
        dy = q[1] - g[1]
        if wp.sqrt(dx * dx + dy * dy) < pen_radius:
            n_in += one
        obs[e, a, base + wp.int32(4) * s] = q[0] - px
        obs[e, a, base + wp.int32(4) * s + 1] = q[1] - py
        obs[e, a, base + wp.int32(4) * s + 2] = vq[0]
        obs[e, a, base + wp.int32(4) * s + 3] = vq[1]
    cx = sx * inv_n_sheep
    cy = sy * inv_n_sheep
    dev = zero
    for s in range(n_sheep):
        q = sheep_pos[e, s]
        dx = q[0] - cx
        dy = q[1] - cy
        dev += dx * dx + dy * dy
    obs[e, a, 6] = cx - px
    obs[e, a, 7] = cy - py
    obs[e, a, 8] = wp.sqrt(dev * inv_n_sheep)
    obs[e, a, 9] = n_in * inv_n_sheep

    # Other dogs, ascending index, self skipped — the same order the torch oracle's
    # precomputed index table produces.
    ob = base + wp.int32(4) * n_sheep
    j = wp.int32(0)
    cnt = zero
    for b in range(n_agents):
        pb = pos[e, b]
        rx = pb[0] - px
        ry = pb[1] - py
        # Squared comparison (no sqrt) so it matches the oracle's squared broadcast
        # distance bit-for-bit at the 2r boundary — ``touching`` is discrete and the
        # harness compares it exactly.
        if rx * rx + ry * ry < col_dist_sq:
            cnt += one
        if b != a:
            obs[e, a, ob + wp.int32(2) * j] = rx
            obs[e, a, ob + wp.int32(2) * j + 1] = ry
            j += 1
    if full_pass == 1:
        touching[e, a] = cnt - one  # drop the self hit


@wp.kernel
def shepherding_reward_kernel(
    sheep_pos: wp.array2d(dtype=Any),
    pen: wp.array(dtype=Any),
    touching: wp.array2d(dtype=Any),
    reset_mask: wp.array(dtype=wp.uint8),
    n_agents: wp.int32,
    n_sheep: wp.int32,
    inv_n_sheep: Any,
    pen_radius: Any,
    pos_shaping_factor: Any,
    spread_shaping_factor: Any,
    pen_reward: Any,
    time_penalty: Any,
    collision_penalty: Any,
    advance_prev: wp.int32,
    full_pass: wp.int32,
    prev_dist: wp.array(dtype=Any),
    prev_spread: wp.array(dtype=Any),
    reward: wp.array2d(dtype=Any),
    done: wp.array(dtype=wp.uint8),
    multiobj: wp.array3d(dtype=Any),
    dist_out: wp.array(dtype=Any),
    spread_out: wp.array(dtype=Any),
    penfrac_out: wp.array(dtype=Any),
):
    """Thread per env; two sequential O(``n_sheep``) passes (deterministic, no atomics).

    ``multiobj``'s last-dim sum is *exactly* ``reward``, written as one addition of the
    five columns rather than accumulated twice, so the identity cannot drift. That is what
    makes it useful: a trainer logs one column per term and sees a shaping imbalance on
    iteration 1 rather than hour 3.

    ``reset_hit`` / ``advance_prev`` / ``full_pass`` follow the navigation contract
    verbatim, applied to *two* baselines (distance and spread) instead of one.
    """
    e = wp.tid()
    zero = type(pos_shaping_factor)(0.0)
    one = type(pos_shaping_factor)(1.0)
    g = pen[e]
    sx = zero
    sy = zero
    sd = zero
    n_in = zero
    all_in = wp.uint8(1)
    for s in range(n_sheep):
        q = sheep_pos[e, s]
        sx += q[0]
        sy += q[1]
        dx = q[0] - g[0]
        dy = q[1] - g[1]
        d = wp.sqrt(dx * dx + dy * dy)
        sd += d
        if d < pen_radius:
            n_in += one
        else:
            all_in = wp.uint8(0)
    mean_dist = sd * inv_n_sheep
    cx = sx * inv_n_sheep
    cy = sy * inv_n_sheep
    dev = zero
    for s in range(n_sheep):
        q = sheep_pos[e, s]
        dx = q[0] - cx
        dy = q[1] - cy
        dev += dx * dx + dy * dy
    # RMS distance from the centroid, not the max: a max hands the gradient to one sheep
    # at a time, so the dogs only ever get told about the current worst straggler.
    spread = wp.sqrt(dev * inv_n_sheep)

    pos_term = (prev_dist[e] - mean_dist) * pos_shaping_factor
    spr_term = (prev_spread[e] - spread) * spread_shaping_factor
    if reset_mask[e] == wp.uint8(1):
        prev_dist[e] = mean_dist
        prev_spread[e] = spread
        pos_term = zero
        spr_term = zero
    else:
        if advance_prev == 1:
            prev_dist[e] = mean_dist
            prev_spread[e] = spread

    if full_pass == 1:
        bonus = zero
        if all_in == wp.uint8(1):
            bonus = pen_reward
        for a in range(n_agents):
            col = collision_penalty * touching[e, a]
            multiobj[e, a, OBJ_POS] = pos_term
            multiobj[e, a, OBJ_SPREAD] = spr_term
            multiobj[e, a, OBJ_PEN] = bonus
            multiobj[e, a, OBJ_TIME] = time_penalty
            multiobj[e, a, OBJ_COLLISION] = col
            reward[e, a] = pos_term + spr_term + bonus + time_penalty + col
        done[e] = all_in
        dist_out[e] = mean_dist
        spread_out[e] = spread
        penfrac_out[e] = n_in * inv_n_sheep


@wp.kernel
def shepherding_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    difficulty: wp.array(dtype=Any),
    lim: Any,
    slim: Any,
    plim: Any,
    cluster_r: Any,
    min_pen_dist: Any,
    drift_max: Any,
    n_agents: wp.int32,
    n_sheep: wp.int32,
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    speed: wp.array2d(dtype=Any),
    sheep_pos: wp.array2d(dtype=Any),
    sheep_vel: wp.array2d(dtype=Any),
    pen: wp.array(dtype=Any),
    drift: wp.array2d(dtype=Any),
    evade: wp.array(dtype=Any),
):
    """One masked episode reset per env: dog spawns, pen centre, sheep spawns, the
    per-episode drift vectors and the evade gain.

    Thread per **env** (see ``swarp/scenarios/reset_kernels.py``): the draws are per-agent
    and per-sheep but they share one RNG stream, and one launch per env is what keeps
    ``auto_reset``'s every-step reset off the launch-count budget.

    The curriculum lives entirely here. ``difficulty`` arrives as a one-element **array**
    read by pointer rather than as a scalar, for the reason
    :attr:`~swarp.scenarios.shepherding.ShepherdingScenario.difficulty` documents, and it
    interpolates three things at once:

    * the sheep spawn, from a tight cluster centred on the pen (``f = 0``) to a uniform
      draw pushed at least ``min_pen_dist`` away from it (``f = 1``);
    * ``drift_mag = f * drift_max``, the per-episode wander;
    * ``evade = f``, the gain on the flee force.

    At ``f = 0`` the sheep therefore start *inside* the pen and are inert, so ``all_in``
    is true on step 1 and the terminal bonus is experienced immediately — which is the
    single reason Push-T's curriculum works. The task at ``f = 0`` is essentially
    transport; at ``f = 1`` it is shepherding.

    ``pen_radius`` is deliberately **not** annealed: moving the success criterion moves
    the reward under the value function.
    """
    e = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return
    rng = wp.rand_init(seed, e)
    zero = _as(0.0, lim)
    one = _as(1.0, lim)
    two = _as(2.0, lim)
    eps = _as(1.0e-9, lim)
    tau = _as(6.28318530717959, lim)
    f = difficulty[0]

    for a in range(n_agents):
        # Draw inline, never through a @wp.func: Warp passes the RNG state by value, so a
        # helper would advance a local copy and hand back the same number every call.
        px = (_as(wp.randf(rng), lim) * two - one) * lim
        py = (_as(wp.randf(rng), lim) * two - one) * lim
        pos[e, a] = wp.vector(px, py)
        vel[e, a] = wp.vector(zero, zero)
        speed[e, a] = zero

    gx = (_as(wp.randf(rng), lim) * two - one) * plim
    gy = (_as(wp.randf(rng), lim) * two - one) * plim
    pen[e] = wp.vector(gx, gy)
    evade[e] = f
    dm = f * drift_max

    for s in range(n_sheep):
        ux = (_as(wp.randf(rng), lim) * two - one) * slim
        uy = (_as(wp.randf(rng), lim) * two - one) * slim
        kx = gx + (_as(wp.randf(rng), lim) * two - one) * cluster_r
        ky = gy + (_as(wp.randf(rng), lim) * two - one) * cluster_r
        qx = (one - f) * kx + f * ux
        qy = (one - f) * ky + f * uy
        relx = qx - gx
        rely = qy - gy
        d = wp.sqrt(relx * relx + rely * rely)
        if d < eps:
            d = eps
        mind = f * min_pen_dist
        if d < mind:
            qx = gx + relx / d * mind
            qy = gy + rely / d * mind
        sheep_pos[e, s] = wp.vector(
            wp.clamp(qx, -slim, slim), wp.clamp(qy, -slim, slim)
        )
        sheep_vel[e, s] = wp.vector(zero, zero)
        ang = _as(wp.randf(rng), lim) * tau
        drift[e, s] = wp.vector(dm * wp.cos(ang), dm * wp.sin(ang))


def _force_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    return [
        a2v,  # pos
        a2v,  # vel
        a2v,  # sheep_pos
        a2v,  # sheep_vel
        a2v,  # drift
        wp.array(dtype=dtype),  # evade
        wp.int32,  # n_agents
        wp.int32,  # n_sheep
        dtype,  # reach
        dtype,  # contact_k
        dtype,  # contact_c
        dtype,  # flee_radius
        dtype,  # flee_k
        dtype,  # sep_radius
        dtype,  # sep_k
        dtype,  # coh_k
        dtype,  # inv_others
        a2v,  # force
    ]


def _integrate_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    return [
        a2v,  # force
        dtype,  # sheep_mass
        dtype,  # linear_damping
        dtype,  # dt
        dtype,  # max_speed
        dtype,  # bound
        a2v,  # sheep_pos
        a2v,  # sheep_vel
    ]


def _obs_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    return [
        a2v,  # pos
        a2v,  # vel
        a2v,  # sheep_pos
        a2v,  # sheep_vel
        wp.array(dtype=vec2),  # pen
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # n_agents
        wp.int32,  # n_sheep
        dtype,  # inv_n_sheep
        dtype,  # pen_radius
        dtype,  # col_dist_sq
        wp.int32,  # full_pass
        wp.array3d(dtype=dtype),  # obs
        wp.array2d(dtype=dtype),  # touching
    ]


def _reward_signature(dtype) -> list:
    a1s = wp.array(dtype=dtype)
    return [
        wp.array2d(dtype=VEC2[dtype]),  # sheep_pos
        wp.array(dtype=VEC2[dtype]),  # pen
        wp.array2d(dtype=dtype),  # touching
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # n_agents
        wp.int32,  # n_sheep
        dtype,  # inv_n_sheep
        dtype,  # pen_radius
        dtype,  # pos_shaping_factor
        dtype,  # spread_shaping_factor
        dtype,  # pen_reward
        dtype,  # time_penalty
        dtype,  # collision_penalty
        wp.int32,  # advance_prev
        wp.int32,  # full_pass
        a1s,  # prev_dist
        a1s,  # prev_spread
        wp.array2d(dtype=dtype),  # reward
        wp.array(dtype=wp.uint8),  # done
        wp.array3d(dtype=dtype),  # multiobj
        a1s,  # dist_out
        a1s,  # spread_out
        a1s,  # penfrac_out
    ]


def _reset_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    return [
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # use_mask
        wp.int32,  # seed
        wp.array(dtype=dtype),  # difficulty (by pointer, see the kernel docstring)
        dtype,  # lim
        dtype,  # slim
        dtype,  # plim
        dtype,  # cluster_r
        dtype,  # min_pen_dist
        dtype,  # drift_max
        wp.int32,  # n_agents
        wp.int32,  # n_sheep
        a2v,  # pos
        a2v,  # vel
        wp.array2d(dtype=dtype),  # speed
        a2v,  # sheep_pos
        a2v,  # sheep_vel
        wp.array(dtype=vec2),  # pen
        a2v,  # drift
        wp.array(dtype=dtype),  # evade
    ]


for _T in (wp.float32, wp.float64):
    register(shepherding_force_kernel, _T, _force_signature(_T))
    register(shepherding_integrate_kernel, _T, _integrate_signature(_T))
    register(shepherding_obs_kernel, _T, _obs_signature(_T))
    register(shepherding_reward_kernel, _T, _reward_signature(_T))
    register(shepherding_reset_kernel, _T, _reset_signature(_T))
