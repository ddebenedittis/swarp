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
* ``shepherding_obs_kernel`` (thread per env,agent) builds the obs row, including the two
  Strombom-style drive/collect hint points.
* ``shepherding_reward_kernel`` (thread per env) reduces the per-sheep pen shaping, the
  gather-to-centroid shaping, the penned potential and the per-agent side/crowd/intrude
  terms, using the navigation ``prev_dist`` / ``reset_hit`` / ``advance_prev`` /
  ``full_pass`` machinery plus three new carried baselines.

The torch implementation in :mod:`swarp.scenarios.shepherding` stays the reference (and
the differentiable path — sheep gradients flow through the plain-torch flock physics).
The fused kernels are the no-grad fast path; because they sum the per-shepherd and
per-sheep contributions in a different order than torch's ``.sum(dim=...)``, the two
sheep trajectories can differ at the ulp scale. Unlike transport that difference is
*amplified* rather than only damped: the flee term is active at a distance, so a tiny
positional difference feeds back through a force the sheep would not otherwise feel,
which is why the scenario declares a slightly looser ``parity_rtol``/``parity_atol``
(validated allclose, not bit-exact).

**Why the shipped task was geometrically infeasible, and what changed.** Relaxing 5 free
sheep under this scenario's own force law converges to a non-symmetric attractor with
radii ``[0.053, 0.104, 0.107, 0.150, 0.151]`` at ``sep_radius=0.15`` and
``[0.071, 0.138, 0.141, 0.196, 0.198]`` at ``sep_radius=0.2``. At the shipped
``pen_radius=0.2`` the free flock's own max radius is 0.198, so ``all_penned`` demanded
the flock centroid land within 0.003 of the pen centre in a world of half-extent 1.0 — a
measured ``terminated`` rate of 0.0016. Shepherds make it worse: three shepherds at
radius ``Rs`` about the flock relax it to ``Rs=0.25 -> 0.350, 0.30 -> 0.268,
0.35 -> 0.195, 0.40 -> 0.154, 0.50 -> 0.152`` (at ``sep 0.15``), because the capped flee
force (1.0) beats cohesion (``0.5*R ~= 0.08``) by 10x. Fixing the geometry
(``pen_radius=0.3``, ``sep_radius=0.15``) alone was not enough either: drawing every
sheep at an independent bearing about the pen (the old ring spawn) puts the flock
centroid *on the pen* by symmetry (``E|c - pen| ~ rbar/sqrt(K)``), so at ``pen 0.3``
doing nothing already solved 23% of episodes — shepherding is not a gathering task, so the
spawn is now a **flock cluster** at a distance (:func:`shepherding_reset_kernel`) and the
reward gained an explicit collect-to-centroid shaping term. Measured scripted-Strombom
solve rate on the new geometry + cluster spawn (256 envs, 5 sheep, 3 shepherds, hold 5,
200 steps): 0.84 scripted vs. 0.00 random and 0.00 do-nothing — both degenerate baselines
that solved the old task collapse to zero on the new one.

Observation layout (thread per (env, agent), ``obs_dim = 16 + 4*n_agents + 2*n_sheep``,
``A = n_agents``, ``K = n_sheep``)::

    0  own pos                              (2)
    2  own vel                              (2)
    4  pen - own_pos                        (2)
    6  gcm - own_pos                        (2)
    8  pen - gcm                            (2)
    10 gcm_vel                              (2)
    12 mean_R                               (1)
    13 max_R                                (1)
    14 stray - own_pos                      (2)
    16 drive_pt - own_pos                   (2)
    18 collect_pt - own_pos                 (2)
    20 other shepherds j != a, ascending:    (4*(A-1))
       rel_pos(2), rel_vel(2)
    20+4*(A-1)  sheep k, index order:        (2*K)
       q_k - own_pos

``gcm``/``gcm_vel`` are the flock centroid and its velocity, ``mean_R``/``max_R`` the
mean/max sheep distance to the centroid, ``stray`` the argmax sheep, and
``drive_pt``/``collect_pt`` the two Strombom hint points computed from them (see
:func:`shepherding_obs_kernel`). Removing slots 16-19 recovers the un-hinted task — a
one-line ablation, and honest about what the Strombom hints cost. ``(pen - sheep_k)`` is
not a slot: it is exactly ``(pen - own_pos) - (sheep_k - own_pos)``, a linear combination
of two blocks already in the row with the same weights for every ``k``.
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
    sheep_vel: wp.array2d(dtype=Any),
    pen: wp.array(dtype=Any),
    n_agents: wp.int32,
    n_sheep: wp.int32,
    inv_sheep: Any,
    drive_offset: Any,
    collect_offset: Any,
    dir_floor: Any,
    tie_eps: Any,
    obs: wp.array3d(dtype=Any),
):
    """Thread per (env, agent): the flock aggregates, the two Strombom hint points, then
    the row (see the module docstring for the exact layout).

    **No separate flock-aggregate launch, no GCM buffer.** The force/body split above
    exists because separation and cohesion *write* ``sheep_pos`` while reading every
    other sheep's position — a genuine read/write hazard across threads. Nothing here
    writes anything the flock reads: every agent's thread only *reads* ``sheep_pos`` and
    reduces it into its own row, so recomputing ``gcm``/``mean_R``/``max_R``/``stray``
    independently in each of the ``A`` threads is hazard-free, merely redundant — 15
    extra ``K``-iterations at the default (3 shepherds, 5 sheep) against standing up a
    fifth launch, a per-env scratch buffer and a carry declaration for something that is
    cheap to recompute. If ``n_sheep`` ever gets large enough that the redundant K-work
    dominates, the right move is a per-env ``Buf("flock", (ne, 8))`` kernel that computes
    the aggregates once and this kernel reads them — not before.

    Four passes, in order:

    1. sum ``q``, sum ``v`` over ``k`` -> ``gcm``, ``gcm_vel``.
    2. per-``k`` distance to ``gcm`` -> ``mean_R``, and the strict-``>`` running max ->
       ``max_R``/``stray``; then the two Strombom points via ``safe_unit``.
    3. the other-shepherds block, ascending index, skipping ``a``.
    4. per-``k`` ``q_k - own_pos``.

    **The stray scan needs a tie *margin*, not just a strict ``>``.** ``stray`` is an
    *identity*, and the two paths' flock trajectories differ by ~1e-9 from the first step
    and by ~1e-7 over the 20-step parity rollout (see the module docstring). Any
    comparison decided inside that drift flips the identity and moves obs slots 14-15 and
    18-19 by O(1), not by a ulp. This is measured, not hypothetical: at ``n_sheep=2`` the
    centroid *is* the midpoint, so ``r_0`` and ``r_1`` are exactly equal every step and a
    bare ``argmax`` is a coin flip -- it failed the fused obs parity on the first
    rollout. So the scan only displaces the incumbent when it clears it by ``tie_eps``,
    which resolves every near-tie to the lower index on both paths.

    ``tie_eps = 1e-5`` (a host constant on the scenario, so both paths scan with the
    identical value) sits two decades above the worst observed path divergence and more
    than an order of magnitude below the smallest genuine gap the flock's equilibrium
    produces (2e-4 at ``n_sheep=3``, 1.5e-3 at 5) -- it cannot mask a real stray.
    ``max_R`` is carried by the same scan rather than reduced separately, so slot 13 and
    slots 14-15 always describe the same sheep. ``k == 0`` seeds the incumbent outright
    (rather than starting from zero) so that a flock whose every radius is below
    ``tie_eps`` -- ``n_sheep == 1``, where ``r_0 == 0`` -- still reports ``max_R = r_0``.

    **``safe_unit`` uses a hard floor (`dir_floor`), not ``clamp(min=1e-9)``.** Every
    other normalization in this file multiplies its unit vector back by a magnitude that
    vanishes with the same denominator, so an ill-conditioned direction there costs a
    ulp. These two do not: ``gcm - pen -> 0`` is exactly the solved configuration the
    policy lives in near the end of an episode, so an unfloored normalize would blow a
    near-zero vector up to a unit vector pointing in an arbitrary (rounding-noise)
    direction — an O(1) disagreement between the two paths, not a ulp. Below the floor
    the direction is defined to be exactly zero instead, on both paths.
    """
    e, a = wp.tid()
    p = pos[e, a]
    v = vel[e, a]
    px = p[0]
    py = p[1]
    c = pen[e]
    zero = type(px)(0.0)

    # pass 1: gcm, gcm_vel
    sx = zero
    sy = zero
    svx = zero
    svy = zero
    for k in range(n_sheep):
        q = sheep_pos[e, k]
        sx += q[0]
        sy += q[1]
        qv = sheep_vel[e, k]
        svx += qv[0]
        svy += qv[1]
    gcx = sx * inv_sheep
    gcy = sy * inv_sheep
    gvx = svx * inv_sheep
    gvy = svy * inv_sheep

    # pass 2: r_k -> mean_R, strict-`>` running max -> max_R/stray (see docstring: this
    # is the parity-critical tie-break, not a style choice)
    rsum = zero
    max_r = zero
    stray_idx = wp.int32(0)
    for k in range(n_sheep):
        q = sheep_pos[e, k]
        dx = q[0] - gcx
        dy = q[1] - gcy
        r = wp.sqrt(dx * dx + dy * dy)
        rsum += r
        if k == 0:
            max_r = r
        elif r > max_r + tie_eps:
            max_r = r
            stray_idx = k
    mean_r = rsum * inv_sheep
    stray = sheep_pos[e, stray_idx]
    strayx = stray[0]
    strayy = stray[1]

    # drive_pt = gcm + drive_offset * safe_unit(gcm - pen)
    dvx = gcx - c[0]
    dvy = gcy - c[1]
    dd = wp.sqrt(dvx * dvx + dvy * dvy)
    if dd >= dir_floor:
        dux = dvx / dd
        duy = dvy / dd
    else:
        dux = zero
        duy = zero
    drive_x = gcx + drive_offset * dux
    drive_y = gcy + drive_offset * duy

    # collect_pt = stray + collect_offset * safe_unit(stray - gcm); the n_sheep == 1
    # degenerate case (stray == gcm) lands here at the floor, giving collect_pt = stray.
    cvx = strayx - gcx
    cvy = strayy - gcy
    cd = wp.sqrt(cvx * cvx + cvy * cvy)
    if cd >= dir_floor:
        cux = cvx / cd
        cuy = cvy / cd
    else:
        cux = zero
        cuy = zero
    collect_x = strayx + collect_offset * cux
    collect_y = strayy + collect_offset * cuy

    obs[e, a, 0] = px
    obs[e, a, 1] = py
    obs[e, a, 2] = v[0]
    obs[e, a, 3] = v[1]
    obs[e, a, 4] = c[0] - px
    obs[e, a, 5] = c[1] - py
    obs[e, a, 6] = gcx - px
    obs[e, a, 7] = gcy - py
    obs[e, a, 8] = c[0] - gcx
    obs[e, a, 9] = c[1] - gcy
    obs[e, a, 10] = gvx
    obs[e, a, 11] = gvy
    obs[e, a, 12] = mean_r
    obs[e, a, 13] = max_r
    obs[e, a, 14] = strayx - px
    obs[e, a, 15] = strayy - py
    obs[e, a, 16] = drive_x - px
    obs[e, a, 17] = drive_y - py
    obs[e, a, 18] = collect_x - px
    obs[e, a, 19] = collect_y - py

    # pass 3: other shepherds, ascending index, skipping self
    base = wp.int32(20)
    idx = wp.int32(0)
    for j in range(n_agents):
        if j != a:
            op = pos[e, j]
            ov = vel[e, j]
            o = base + wp.int32(4) * idx
            obs[e, a, o] = op[0] - px
            obs[e, a, o + 1] = op[1] - py
            obs[e, a, o + 2] = ov[0] - v[0]
            obs[e, a, o + 3] = ov[1] - v[1]
            idx += 1

    # pass 4: sheep block, index order
    sbase = base + wp.int32(4) * (n_agents - wp.int32(1))
    for k in range(n_sheep):
        q = sheep_pos[e, k]
        obs[e, a, sbase + wp.int32(2) * k] = q[0] - px
        obs[e, a, sbase + wp.int32(2) * k + 1] = q[1] - py


@wp.kernel
def shepherding_reward_kernel(
    sheep_pos: wp.array2d(dtype=Any),
    pen: wp.array(dtype=Any),
    agent_pos: wp.array2d(dtype=Any),
    reset_mask: wp.array(dtype=wp.uint8),
    n_agents: wp.int32,
    n_sheep: wp.int32,
    inv_sheep: Any,
    pos_shaping_factor: Any,
    gather_factor: Any,
    pen_radius: Any,
    pen_reward: Any,
    pen_level_reward: Any,
    done_reward: Any,
    scatter_penalty: Any,
    side_factor: Any,
    crowd_factor: Any,
    crowd_radius: Any,
    intrude_penalty: Any,
    dir_floor: Any,
    pen_hold: wp.int32,
    advance_prev: wp.int32,
    full_pass: wp.int32,
    prev_dist: wp.array2d(dtype=Any),
    prev_radius: wp.array(dtype=Any),
    prev_penned: wp.array(dtype=Any),
    hold: wp.array(dtype=Any),
    reward: wp.array2d(dtype=Any),
    done: wp.array(dtype=wp.uint8),
    dist_out: wp.array(dtype=Any),
    penned_out: wp.array(dtype=Any),
    radius_out: wp.array(dtype=Any),
    allpen_out: wp.array(dtype=wp.uint8),
):
    """Thread per env: pen shaping, gather shaping, penned potential, per-agent terms.

    Three ``0..K-1``-ish sweeps plus a per-agent loop, no atomics:

    1. accumulate ``cx, cy`` over the sheep -> ``gcm`` (needed before the next pass can
       shape against it).
    2. per-sheep ``d_k`` to the pen (the existing ``prev_dist`` rebase/advance/shaping
       discipline, unchanged) *and* ``r_k`` to ``gcm``, accumulated into ``dist_sum``,
       ``penned`` and ``rad_sum`` in the same pass.
    3. the shared reward: fold ``mean_R`` against the ``prev_radius`` carry (gather
       shaping) and ``penned`` against the ``prev_penned`` carry (the pen potential).
    4. per agent: ``side``/``crowd``/``intrude``, added onto the shared term.

    **Ordering discipline, preserved from the pre-redesign kernel.** The ``reset_hit``
    rebase of every shaping baseline (``prev_dist`` here, plus the three new carries
    ``prev_radius``/``prev_penned``/``hold``) happens *outside* the ``full_pass`` gate —
    a masked reset restamps its baseline the instant it fires, on every pass, so a reset
    env never pays one step of shaping computed against a stale pre-reset value. The
    reward *reduction* (everything that writes ``reward``/``done``/the info outputs)
    happens *inside* ``full_pass`` — it is meaningless, and the ``*_out`` buffers are
    stale, on a sub-pass.
    """
    e = wp.tid()
    zero = type(pos_shaping_factor)(0.0)
    one = type(pos_shaping_factor)(1.0)
    c = pen[e]
    reset_hit = wp.int32(reset_mask[e])

    # pass 1: flock centroid
    cx = zero
    cy = zero
    for k in range(n_sheep):
        q = sheep_pos[e, k]
        cx += q[0]
        cy += q[1]
    gcx = cx * inv_sheep
    gcy = cy * inv_sheep

    # pass 2: per-sheep pen distance (prev_dist discipline unchanged) + flock radius
    shaping_sum = zero
    dist_sum = zero
    penned = zero
    rad_sum = zero
    all_penned = wp.uint8(1)
    for k in range(n_sheep):
        q = sheep_pos[e, k]
        qx = q[0]
        qy = q[1]
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
        rx = qx - gcx
        ry = qy - gcy
        rad_sum += wp.sqrt(rx * rx + ry * ry)

    mean_r = rad_sum * inv_sheep
    gather = (prev_radius[e] - mean_r) * gather_factor
    pen_delta = (penned - prev_penned[e]) * pen_reward
    if reset_hit == 1:
        # Same rule as prev_dist above: rebase every carried baseline the instant a
        # reset fires (not only on a full_pass), so the first shaped step of a fresh
        # episode never differences against a value from before the reset.
        prev_radius[e] = mean_r
        prev_penned[e] = penned
        hold[e] = zero
        gather = zero
        pen_delta = zero
    else:
        if advance_prev == 1:
            prev_radius[e] = mean_r
            prev_penned[e] = penned

    if full_pass == 1:
        first_done = zero
        if reset_hit == 0 and advance_prev == 1:
            h = hold[e]
            if all_penned == wp.uint8(1):
                h = h + one
            else:
                h = zero
            hold[e] = h
            # Strict `==`: this fires once, on the step the hold counter *reaches*
            # pen_hold, not on every step it stays there — see the scenario docstring.
            if h == type(h)(pen_hold):
                first_done = one

        global_r = (
            shaping_sum
            + gather
            + pen_delta
            + pen_level_reward * penned
            + done_reward * first_done
            - scatter_penalty * mean_r
        )

        for a in range(n_agents):
            p = agent_pos[e, a]
            px = p[0]
            py = p[1]

            # side_factor: "get behind the flock" — dot of the two safe-unit directions,
            # both floored the same way obs's drive/collect points are (see
            # shepherding_obs_kernel's docstring for why a hard floor and not an eps).
            svx = px - gcx
            svy = py - gcy
            sd = wp.sqrt(svx * svx + svy * svy)
            if sd >= dir_floor:
                sux = svx / sd
                suy = svy / sd
            else:
                sux = zero
                suy = zero
            gvx = gcx - c[0]
            gvy = gcy - c[1]
            gd = wp.sqrt(gvx * gvx + gvy * gvy)
            if gd >= dir_floor:
                gux = gvx / gd
                guy = gvy / gd
            else:
                gux = zero
                guy = zero
            side = side_factor * (sux * gux + suy * guy)

            # crowd_factor: squared hinge against every other shepherd, for caging's
            # reason (caging.py: an L1 gradient goes flat for a large fraction of
            # agents; squared keeps every pair informative).
            crowd = zero
            for j in range(n_agents):
                if j != a:
                    op = agent_pos[e, j]
                    ox = px - op[0]
                    oy = py - op[1]
                    od = wp.sqrt(ox * ox + oy * oy)
                    gap = crowd_radius - od
                    if gap > zero:
                        crowd += gap * gap

            # intrude_penalty: a real failure mode, not hygiene — every other term pulls
            # a shepherd toward the pen, and one parked *in* the pen flees the sheep
            # straight back out, making `done` unreachable for as long as it sits there.
            ipx = px - c[0]
            ipy = py - c[1]
            ipd = wp.sqrt(ipx * ipx + ipy * ipy)
            intrude = pen_radius - ipd
            if intrude < zero:
                intrude = zero

            reward[e, a] = global_r + side - crowd_factor * crowd - intrude_penalty * intrude

        done_flag = wp.uint8(0)
        if hold[e] >= type(hold[e])(pen_hold):
            done_flag = wp.uint8(1)
        done[e] = done_flag
        dist_out[e] = dist_sum * inv_sheep
        penned_out[e] = penned * inv_sheep
        radius_out[e] = mean_r
        allpen_out[e] = all_penned


@wp.kernel
def shepherding_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    lim: Any,
    plim: Any,
    sbound: Any,
    dmin: Any,
    dspan: Any,
    spread: Any,
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
    """Masked episode reset: shepherds, a pen, then a flock **cluster** at a distance.

    Sheep no longer spawn on an independent-bearing ring about the pen: drawing each
    sheep at its own bearing puts the flock centroid *on the pen* by symmetry
    (``E|centroid - pen| ~ rbar/sqrt(K)``), which made "do nothing" solve a real fraction
    of episodes on the old geometry (see the module docstring). Instead, one flock centre
    is drawn at a bearing and a distance in ``[dmin, dmin+dspan]`` from the pen, and every
    sheep is then drawn as its own bearing and a ``spread * sqrt(u)`` radius about *that*
    centre (a uniform disc, not a ring, so the cluster fills rather than rims).

    Draw order is binding for parity: shepherd coordinates, then the pen, then the
    (draw-free) goals mirror loop, then exactly 2 new draws for the flock centre, then
    the same *count* of per-sheep draws as before (bearing, then the disc radius's ``u``).
    Keeping the shepherd/pen draws first and unchanged in count means those two spawns
    stay bit-identical to the ring-spawn kernel this replaces — only what is drawn after
    the (still draw-free) goals loop is new.

    Every ``wp.randf`` is drawn inline — a ``@wp.func`` helper would take the RNG state
    by value and hand back the same point every call (see
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

    # Flock centre: one bearing, one distance, drawn *after* the (draw-free) goals loop
    # so the shepherd/pen RNG stream above is untouched. dmin/dspan already bake in the
    # "never start already solved" stand-off (the pen radius plus a sheep diameter) —
    # see ShepherdingScenario._set_spawn_geometry — so nothing here re-derives it.
    bearing = _as(wp.randf(rng), lim) * two_pi
    d = dmin + _as(wp.randf(rng), lim) * dspan
    fx = cx + d * wp.cos(bearing)
    fy = cy + d * wp.sin(bearing)

    for k in range(n_sheep):
        ang = _as(wp.randf(rng), lim) * two_pi
        u = _as(wp.randf(rng), lim)
        rr = spread * wp.sqrt(u)  # uniform disc, not a ring: sqrt(u) is the area measure
        sx = wp.clamp(fx + rr * wp.cos(ang), -sbound, sbound)
        sy = wp.clamp(fy + rr * wp.sin(ang), -sbound, sbound)
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
    a2v = wp.array2d(dtype=vec2)
    return [
        a2v,  # pos
        a2v,  # vel
        a2v,  # sheep_pos
        a2v,  # sheep_vel
        wp.array(dtype=vec2),  # pen
        wp.int32,  # n_agents
        wp.int32,  # n_sheep
        dtype,  # inv_sheep
        dtype,  # drive_offset
        dtype,  # collect_offset
        dtype,  # dir_floor
        dtype,  # tie_eps
        wp.array3d(dtype=dtype),  # obs
    ]


def _reward_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    a2s = wp.array2d(dtype=dtype)
    a1s = wp.array(dtype=dtype)
    return [
        a2v,  # sheep_pos
        wp.array(dtype=vec2),  # pen
        a2v,  # agent_pos
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # n_agents
        wp.int32,  # n_sheep
        dtype,  # inv_sheep
        dtype,  # pos_shaping_factor
        dtype,  # gather_factor
        dtype,  # pen_radius
        dtype,  # pen_reward
        dtype,  # pen_level_reward
        dtype,  # done_reward
        dtype,  # scatter_penalty
        dtype,  # side_factor
        dtype,  # crowd_factor
        dtype,  # crowd_radius
        dtype,  # intrude_penalty
        dtype,  # dir_floor
        wp.int32,  # pen_hold
        wp.int32,  # advance_prev
        wp.int32,  # full_pass
        a2s,  # prev_dist
        a1s,  # prev_radius
        a1s,  # prev_penned
        a1s,  # hold
        a2s,  # reward
        wp.array(dtype=wp.uint8),  # done
        a1s,  # dist_out
        a1s,  # penned_out
        a1s,  # radius_out
        wp.array(dtype=wp.uint8),  # allpen_out
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
        dtype,  # dmin
        dtype,  # dspan
        dtype,  # spread
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
