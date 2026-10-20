"""Fused Warp kernels for :class:`~swarp.scenarios.pusht_lidar.PushTLidarScenario`.

Four launches per pass, in this order (see
:meth:`~swarp.scenarios.pusht_lidar.PushTLidarScenario.launch_fused`), after Push-T's own
body sync on a step:

* :meth:`swarp.sensors.lidar.Lidar.launch` — the capture-safe all-shape lidar, one thread
  per ``(env, agent, ray)``, into the lidar's own ``[E, A, R]`` range / hit-class arrays.
* ``pusht_lidar_obs_kernel`` (thread per ``(env, agent)``) — the observation row, the
  frame-stack ring, the role-sorted critic state, the auxiliary targets, and the
  per-agent reward inputs read off the agent's own rays (object-ray count, occlusion
  fraction, coverage sectors, near-contact flag).
* ``pusht_lidar_reward_kernel`` (thread per env) — Push-T's team shaping and per-agent
  approach term (copied below as ``wp.func`` so ``pusht_kernels`` stays untouched), the
  observer terms, the objective vector and the per-env diagnostics. Sequential agent
  loops, no atomics.
* ``pusht_lidar_roles_kernel`` (thread per env, reset only) — a masked Fisher-Yates
  shuffle of the fixed role multiset, on its own kernel seed.

``lidar_sensor_kernel`` (thread per ``(env, agent)``) runs between the lidar and the
observation kernel only when the scenario enables a sensor effect (a blind zone around
the mount, a field of view, range noise): it edits the scan in place, so every consumer
downstream sees the scan a real sensor would return. Without one it is never launched.

The torch reference in :mod:`swarp.scenarios.pusht_lidar` recomputes all of it from the
torch lidar oracle (:func:`swarp.sensors.lidar.lidar_scan`) and shares no code with this
module.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.state import VEC2

# Lean launches (no grid-stride loop); see docs/performance.md.
wp.set_module_options({"default_grid_stride": False})

#: Objective columns of ``info()["multiobj_reward"]``, in order. Kept next to the kernel
#: that writes them so the two cannot drift apart.
OBJ_TEAM = 0
OBJ_APPROACH = 1
OBJ_STANDOFF = 2
OBJ_VISIBILITY = 3
OBJ_COVERAGE = 4
OBJ_OCCLUSION = 5
OBJ_CONTACT = 6
N_OBJ = 7
OBJECTIVES = ("team", "approach", "standoff", "visibility", "coverage", "occlusion", "contact")

#: Width of the per-env head of the critic state (T pose, goal, T velocity).
STATE_HEAD = 15
#: Per-agent width of the critic state: pos, vel, role sign (plus cos/sin heading, when
#: the heading matters).
STATE_AGENT = 5
#: Width of the self block of the observation (plus cos/sin heading, when it matters).
SELF_DIM = 9
#: Width of the oracle T block (obs level 0), the same eight numbers Push-T observes.
TEE_DIM = 8
#: Coverage sectors around the T centroid, in the T frame.
N_SECTORS = 8

#: Slack on the field-of-view test, so a ray exactly at half the field of view (270
#: degrees keeps the rays at +-135) is kept despite rounding.
FOV_EPS = wp.constant(1e-4)

_HIT_AGENT = wp.constant(wp.uint8(1))
_HIT_MOVABLE = wp.constant(wp.uint8(3))


@wp.func
def _team_shaping(
    d: Any,
    ang: Any,
    pd: Any,
    pa: Any,
    pos_shaping_factor: Any,
    rot_shaping_factor: Any,
    joint_shaping: Any,
    rot_away_penalty: Any,
    goal_tolerance: Any,
    angle_tolerance: Any,
):
    """Push-T's pose shaping, verbatim from ``pusht_reward_kernel``: linear position and
    heading progress, the joint Gaussian crater and the turn-away cost."""
    zero = type(d)(0.0)
    ps = (pd - d) * pos_shaping_factor + (pa - ang) * rot_shaping_factor
    sd = type(d)(1.5) * goal_tolerance
    sa = type(d)(1.5) * angle_tolerance
    ps += joint_shaping * (
        wp.exp(-(d / sd) * (d / sd) - (ang / sa) * (ang / sa))
        - wp.exp(-(pd / sd) * (pd / sd) - (pa / sa) * (pa / sa))
    )
    ps -= rot_away_penalty * wp.max(ang - pa, zero)
    return ps


@wp.func
def _wrap(x: Any):
    return wp.atan2(wp.sin(x), wp.cos(x))


@wp.kernel
def pusht_lidar_obs_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    obs_pos: wp.array2d(dtype=Any),
    obs_vel: wp.array2d(dtype=Any),
    obs_theta: wp.array2d(dtype=Any),
    tee_pos: wp.array(dtype=Any),
    tee_theta: wp.array(dtype=Any),
    tee_vel: wp.array(dtype=Any),
    tee_ang_vel: wp.array(dtype=Any),
    goal_pos: wp.array(dtype=Any),
    goal_theta: wp.array(dtype=Any),
    roles: wp.array2d(dtype=wp.int32),
    ranges: wp.array3d(dtype=Any),
    hit_class: wp.array3d(dtype=wp.uint8),
    reset_mask: wp.array(dtype=wp.uint8),
    n_pushers: wp.int32,
    obs_level: wp.int32,
    frame_stack: wp.int32,
    n_comm: wp.int32,
    comm_range: Any,
    max_range: Any,
    ray_step: Any,
    angle_start: Any,
    body_frame: wp.int32,
    tee_radius: Any,
    aux_scale: Any,
    contact_dist: Any,
    heading: wp.int32,
    advance_prev: wp.int32,
    obs: wp.array3d(dtype=Any),
    ring: wp.array3d(dtype=Any),
    critic: wp.array2d(dtype=Any),
    aux: wp.array3d(dtype=Any),
    n_obj_out: wp.array2d(dtype=Any),
    occl_out: wp.array2d(dtype=Any),
    sectors_out: wp.array2d(dtype=wp.uint8),
    near_out: wp.array2d(dtype=wp.uint8),
):
    """Thread per (env, agent). Layout: see ``PushTLidarScenario``'s docstring. The self
    and teammate blocks read the observed poses ``obs_pos``/``obs_vel``/``obs_theta``
    (the true ones without pose noise); everything else reads the true state."""
    e, a = wp.tid()
    n_agents = pos.shape[1]
    n_rays = ranges.shape[2]
    zero = type(max_range)(0.0)
    one = type(max_range)(1.0)
    pi = type(max_range)(3.141592653589793)
    p = pos[e, a]
    v = vel[e, a]
    q = tee_pos[e]
    th = tee_theta[e]
    g = goal_pos[e]
    gth = goal_theta[e]
    role = roles[e, a]
    sgn = -one
    if role == 1:
        sgn = one

    # ---- self block, from the observed pose
    po = obs_pos[e, a]
    vo = obs_vel[e, a]
    obs[e, a, 0] = po[0]
    obs[e, a, 1] = po[1]
    obs[e, a, 2] = vo[0]
    obs[e, a, 3] = vo[1]
    obs[e, a, 4] = sgn
    obs[e, a, 5] = g[0] - po[0]
    obs[e, a, 6] = g[1] - po[1]
    obs[e, a, 7] = wp.cos(gth)
    obs[e, a, 8] = wp.sin(gth)
    k = wp.int32(9)
    if heading == 1:
        obs[e, a, 9] = wp.cos(obs_theta[e, a])
        obs[e, a, 10] = wp.sin(obs_theta[e, a])
        k = wp.int32(11)
    # ---- teammates: relative observed position + role sign, ascending index, self skipped
    for j in range(n_agents):
        if j != a:
            pj = obs_pos[e, j]
            obs[e, a, k] = pj[0] - po[0]
            obs[e, a, k + 1] = pj[1] - po[1]
            sj = -one
            if roles[e, j] == 1:
                sj = one
            obs[e, a, k + 2] = sj
            k += wp.int32(3)
    # ---- oracle T block (level 0): Push-T's eight numbers
    if obs_level == 0:
        err = th - gth
        obs[e, a, k] = q[0] - p[0]
        obs[e, a, k + 1] = q[1] - p[1]
        obs[e, a, k + 2] = g[0] - q[0]
        obs[e, a, k + 3] = g[1] - q[1]
        obs[e, a, k + 4] = wp.cos(th)
        obs[e, a, k + 5] = wp.sin(th)
        obs[e, a, k + 6] = wp.cos(err)
        obs[e, a, k + 7] = wp.sin(err)
        k += wp.int32(8)

    # ---- lidar frame ring: [range / max_range | object mask | agent mask] per frame,
    # oldest first. A step shifts it by one frame; a reset zeroes it; the reset pass
    # leaves an env it did not reset alone (its newest frame is already this scan).
    width = wp.int32(3) * n_rays
    newest = (frame_stack - wp.int32(1)) * width
    write_now = wp.int32(0)
    if reset_mask[e] == wp.uint8(1):
        for i in range(newest):
            ring[e, a, i] = zero
        write_now = wp.int32(1)
    elif advance_prev == 1:
        for i in range(newest):
            ring[e, a, i] = ring[e, a, i + width]
        write_now = wp.int32(1)
    if write_now == 1:
        for r in range(n_rays):
            c = hit_class[e, a, r]
            ring[e, a, newest + r] = ranges[e, a, r] / max_range
            m_obj = zero
            if c == _HIT_MOVABLE:
                m_obj = one
            m_ag = zero
            if c == _HIT_AGENT:
                m_ag = one
            ring[e, a, newest + n_rays + r] = m_obj
            ring[e, a, newest + wp.int32(2) * n_rays + r] = m_ag
    for i in range(frame_stack * width):
        obs[e, a, k + i] = ring[e, a, i]
    k += frame_stack * width

    # ---- pooled object hits (level 2): K points per agent j, relative to own position
    if obs_level == 2:
        for j in range(n_agents):
            pj = pos[e, j]
            base_j = zero
            if body_frame != 0:
                base_j = theta[e, j]
            dx = pj[0] - p[0]
            dy = pj[1] - p[1]
            in_range = wp.int32(0)
            if j == a or wp.sqrt(dx * dx + dy * dy) <= comm_range:
                in_range = wp.int32(1)
            nh = wp.int32(0)
            for r in range(n_rays):
                if hit_class[e, j, r] == _HIT_MOVABLE:
                    nh += wp.int32(1)
            rr = wp.int32(0)  # next ray to inspect
            h = wp.int32(-1)  # hit number of the last hit found
            last = wp.int32(0)  # ray index of that hit
            for s in range(n_comm):
                if in_range == 1 and nh > 0:
                    target = ((wp.int32(2) * s + wp.int32(1)) * nh) // (wp.int32(2) * n_comm)
                    while h < target:
                        if hit_class[e, j, rr] == _HIT_MOVABLE:
                            h += wp.int32(1)
                            last = rr
                        rr += wp.int32(1)
                    ang = base_j + angle_start + type(max_range)(last) * ray_step
                    t = ranges[e, j, last]
                    hx = pj[0] + t * wp.cos(ang)
                    hy = pj[1] + t * wp.sin(ang)
                    obs[e, a, k] = hx - p[0]
                    obs[e, a, k + 1] = hy - p[1]
                    obs[e, a, k + 2] = one
                else:
                    obs[e, a, k] = zero
                    obs[e, a, k + 1] = zero
                    obs[e, a, k + 2] = zero
                k += wp.int32(3)

    # ---- per-agent reward inputs from the agent's own rays
    base = zero
    if body_frame != 0:
        base = theta[e, a]
    qx = q[0] - p[0]
    qy = q[1] - p[1]
    dist = wp.sqrt(qx * qx + qy * qy)
    bearing = wp.atan2(qy, qx)
    # The T's angular cone, from its bounding disc; every ray when inside the disc.
    half = type(max_range)(4.0)
    if dist > tee_radius:
        half = wp.asin(tee_radius / dist)
    ca = wp.cos(th)
    sa = wp.sin(th)
    quarter = pi / type(max_range)(4.0)
    n_obj = wp.int32(0)
    n_cone = wp.int32(0)
    n_occ = wp.int32(0)
    bits = wp.int32(0)
    near = wp.uint8(0)
    for r in range(n_rays):
        c = hit_class[e, a, r]
        ang = base + angle_start + type(max_range)(r) * ray_step
        if wp.abs(_wrap(ang - bearing)) <= half:
            n_cone += wp.int32(1)
            if c == _HIT_AGENT:
                n_occ += wp.int32(1)
        if c == _HIT_MOVABLE:
            n_obj += wp.int32(1)
            t = ranges[e, a, r]
            if t < contact_dist:
                near = wp.uint8(1)
            rx = p[0] + t * wp.cos(ang) - q[0]
            ry = p[1] + t * wp.sin(ang) - q[1]
            lx = ca * rx + sa * ry
            ly = -sa * rx + ca * ry
            sec = wp.int32(wp.floor((wp.atan2(ly, lx) + pi) / quarter))
            sec = wp.clamp(sec, wp.int32(0), wp.int32(7))
            bits = bits | (wp.int32(1) << sec)
    n_obj_out[e, a] = type(max_range)(n_obj)
    occl_out[e, a] = type(max_range)(n_occ) / type(max_range)(wp.max(n_cone, wp.int32(1)))
    sectors_out[e, a] = wp.uint8(bits)
    near_out[e, a] = near

    # ---- auxiliary targets: the T in this agent's (world-axis) frame, and its heading
    aux[e, a, 0] = qx * aux_scale
    aux[e, a, 1] = qy * aux_scale
    aux[e, a, 2] = ca
    aux[e, a, 3] = sa

    # ---- critic state, role-sorted: this agent's block at its role-sorted rank
    rank = wp.int32(0)
    for j in range(a):
        if roles[e, j] == role:
            rank += wp.int32(1)
    if role == 1:
        rank += n_pushers
    agent_w = wp.int32(STATE_AGENT) + wp.int32(2) * heading
    off = wp.int32(STATE_HEAD) + agent_w * rank
    critic[e, off] = p[0]
    critic[e, off + 1] = p[1]
    critic[e, off + 2] = v[0]
    critic[e, off + 3] = v[1]
    critic[e, off + 4] = sgn
    if heading == 1:
        critic[e, off + 5] = wp.cos(theta[e, a])
        critic[e, off + 6] = wp.sin(theta[e, a])
    if a == 0:
        tv = tee_vel[e]
        err = th - gth
        critic[e, 0] = q[0]
        critic[e, 1] = q[1]
        critic[e, 2] = g[0] - q[0]
        critic[e, 3] = g[1] - q[1]
        critic[e, 4] = ca
        critic[e, 5] = sa
        critic[e, 6] = wp.cos(err)
        critic[e, 7] = wp.sin(err)
        critic[e, 8] = tv[0]
        critic[e, 9] = tv[1]
        critic[e, 10] = tee_ang_vel[e]
        critic[e, 11] = g[0]
        critic[e, 12] = g[1]
        critic[e, 13] = wp.cos(gth)
        critic[e, 14] = wp.sin(gth)


@wp.kernel
def pusht_lidar_reward_kernel(
    pos: wp.array2d(dtype=Any),
    tee_pos: wp.array(dtype=Any),
    tee_theta: wp.array(dtype=Any),
    goal_pos: wp.array(dtype=Any),
    goal_theta: wp.array(dtype=Any),
    reset_mask: wp.array(dtype=wp.uint8),
    roles: wp.array2d(dtype=wp.int32),
    n_obj: wp.array2d(dtype=Any),
    occl: wp.array2d(dtype=Any),
    sectors: wp.array2d(dtype=wp.uint8),
    near: wp.array2d(dtype=wp.uint8),
    n_agents: wp.int32,
    pos_shaping_factor: Any,
    rot_shaping_factor: Any,
    agent_dist_shaping: Any,
    push_point_offset: Any,
    joint_shaping: Any,
    rot_away_penalty: Any,
    goal_tolerance: Any,
    angle_tolerance: Any,
    goal_reward: Any,
    standoff_dist: Any,
    standoff_shaping: Any,
    visibility_rays: Any,
    visibility_weight: Any,
    coverage_weight: Any,
    occlusion_weight: Any,
    contact_penalty: Any,
    advance_prev: wp.int32,
    full_pass: wp.int32,
    prev_dist: wp.array(dtype=Any),
    prev_ang: wp.array(dtype=Any),
    prev_adist: wp.array2d(dtype=Any),
    prev_standoff: wp.array2d(dtype=Any),
    reward: wp.array2d(dtype=Any),
    done: wp.array(dtype=wp.uint8),
    dist_out: wp.array(dtype=Any),
    ang_out: wp.array(dtype=Any),
    multiobj: wp.array3d(dtype=Any),
    coverage_out: wp.array(dtype=Any),
    visibility_out: wp.array(dtype=Any),
    occlusion_out: wp.array(dtype=Any),
    tee_seen: wp.array2d(dtype=wp.uint8),
):
    """Thread per env; sequential agent loops, no atomics.

    The carries (Push-T's three plus the standoff error) advance on a step and rebase on
    a reset exactly as in ``pusht_reward_kernel``; the outputs are written on a full pass
    only, so an obs-only auto-reset leaves the transition's reward and info intact.
    ``reward`` is written as one addition of the seven ``multiobj`` columns.
    """
    e = wp.tid()
    zero = type(pos_shaping_factor)(0.0)
    one = type(pos_shaping_factor)(1.0)
    tp = tee_pos[e]
    gx = goal_pos[e][0] - tp[0]
    gy = goal_pos[e][1] - tp[1]
    d = wp.sqrt(gx * gx + gy * gy)
    raw = tee_theta[e] - goal_theta[e]
    ang = wp.abs(wp.atan2(wp.sin(raw), wp.cos(raw)))
    reset_hit = wp.int32(reset_mask[e])
    ps = _team_shaping(
        d,
        ang,
        prev_dist[e],
        prev_ang[e],
        pos_shaping_factor,
        rot_shaping_factor,
        joint_shaping,
        rot_away_penalty,
        goal_tolerance,
        angle_tolerance,
    )
    shaping = zero
    if reset_hit == 1:
        prev_dist[e] = d
        prev_ang[e] = ang
    else:
        if advance_prev == 1:
            prev_dist[e] = d
            prev_ang[e] = ang
        shaping = ps
    on_goal = wp.uint8(0)
    if d < goal_tolerance and ang < angle_tolerance:
        on_goal = wp.uint8(1)
    bonus = zero
    if on_goal == wp.uint8(1):
        bonus = goal_reward
    team = shaping + bonus

    inv_d = one / wp.max(d, type(zero)(1.0e-6))
    bx = tp[0] - gx * inv_d * push_point_offset
    by = tp[1] - gy * inv_d * push_point_offset

    # Team coverage: sectors holding an observer's object hit, OR-ed over observers.
    bits = wp.int32(0)
    for a in range(n_agents):
        if roles[e, a] == 1:
            bits = bits | wp.int32(sectors[e, a])
    n_sec = wp.int32(0)
    for s in range(8):
        if (bits >> s) & wp.int32(1) == wp.int32(1):
            n_sec += wp.int32(1)
    cov = type(zero)(n_sec) / type(zero)(8.0)

    vis_sum = zero
    occ_sum = zero
    n_obs = wp.int32(0)
    for a in range(n_agents):
        p = pos[e, a]
        adx = p[0] - bx
        ady = p[1] - by
        ad = wp.sqrt(adx * adx + ady * ady)
        ash = (prev_adist[e, a] - ad) * agent_dist_shaping
        qx = p[0] - tp[0]
        qy = p[1] - tp[1]
        serr = wp.abs(wp.sqrt(qx * qx + qy * qy) - standoff_dist)
        ssh = (prev_standoff[e, a] - serr) * standoff_shaping
        if reset_hit == 1:
            prev_adist[e, a] = ad
            prev_standoff[e, a] = serr
            ash = zero
            ssh = zero
        else:
            if advance_prev == 1:
                prev_adist[e, a] = ad
                prev_standoff[e, a] = serr
        approach = zero
        standoff = zero
        vis_t = zero
        cov_t = zero
        occ_t = zero
        con_t = zero
        if roles[e, a] == 0:
            approach = ash
        else:
            vis = wp.min(n_obj[e, a] / visibility_rays, one)
            standoff = ssh
            vis_t = -visibility_weight * (one - vis)
            cov_t = -coverage_weight * (one - cov)
            occ_t = -occlusion_weight * occl[e, a]
            if near[e, a] == wp.uint8(1):
                con_t = -contact_penalty
            vis_sum += vis
            occ_sum += occl[e, a]
            n_obs += wp.int32(1)
        if full_pass == 1:
            multiobj[e, a, 0] = team
            multiobj[e, a, 1] = approach
            multiobj[e, a, 2] = standoff
            multiobj[e, a, 3] = vis_t
            multiobj[e, a, 4] = cov_t
            multiobj[e, a, 5] = occ_t
            multiobj[e, a, 6] = con_t
            reward[e, a] = team + approach + standoff + vis_t + cov_t + occ_t + con_t
            seen = wp.uint8(0)
            if n_obj[e, a] > zero:
                seen = wp.uint8(1)
            tee_seen[e, a] = seen

    if full_pass == 1:
        dist_out[e] = d
        ang_out[e] = ang
        done[e] = on_goal
        coverage_out[e] = cov
        denom = type(zero)(wp.max(n_obs, wp.int32(1)))
        visibility_out[e] = vis_sum / denom
        occlusion_out[e] = occ_sum / denom


@wp.kernel
def pusht_lidar_roles_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    n_pushers: wp.int32,
    roles: wp.array2d(dtype=wp.int32),
):
    """Masked per-env Fisher-Yates over ``n_pushers`` zeros and the rest ones.

    The row is refilled with the sorted multiset before the shuffle, so each reset draws
    a uniform permutation whatever the previous episode held, and the per-role counts are
    exact by construction.
    """
    e = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return
    n_agents = roles.shape[1]
    for a in range(n_agents):
        r = wp.int32(1)
        if a < n_pushers:
            r = wp.int32(0)
        roles[e, a] = r
    rng = wp.rand_init(seed, e)
    i = n_agents - 1
    while i > 0:
        j = wp.int32(wp.randi(rng, 0, i + 1))
        tmp = roles[e, i]
        roles[e, i] = roles[e, j]
        roles[e, j] = tmp
        i -= 1


def _obs_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a1s = wp.array(dtype=dtype)
    a2s = wp.array2d(dtype=dtype)
    return [
        wp.array2d(dtype=vec2),  # pos
        wp.array2d(dtype=vec2),  # vel
        a2s,  # theta
        wp.array2d(dtype=vec2),  # obs_pos
        wp.array2d(dtype=vec2),  # obs_vel
        a2s,  # obs_theta
        wp.array(dtype=vec2),  # tee_pos
        a1s,  # tee_theta
        wp.array(dtype=vec2),  # tee_vel
        a1s,  # tee_ang_vel
        wp.array(dtype=vec2),  # goal_pos
        a1s,  # goal_theta
        wp.array2d(dtype=wp.int32),  # roles
        wp.array3d(dtype=dtype),  # ranges
        wp.array3d(dtype=wp.uint8),  # hit_class
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # n_pushers
        wp.int32,  # obs_level
        wp.int32,  # frame_stack
        wp.int32,  # n_comm
        dtype,  # comm_range
        dtype,  # max_range
        dtype,  # ray_step
        dtype,  # angle_start
        wp.int32,  # body_frame
        dtype,  # tee_radius
        dtype,  # aux_scale
        dtype,  # contact_dist
        wp.int32,  # heading
        wp.int32,  # advance_prev
        wp.array3d(dtype=dtype),  # obs
        wp.array3d(dtype=dtype),  # ring
        a2s,  # critic
        wp.array3d(dtype=dtype),  # aux
        a2s,  # n_obj_out
        a2s,  # occl_out
        wp.array2d(dtype=wp.uint8),  # sectors_out
        wp.array2d(dtype=wp.uint8),  # near_out
    ]


def _reward_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a1s = wp.array(dtype=dtype)
    a2s = wp.array2d(dtype=dtype)
    return [
        wp.array2d(dtype=vec2),  # pos
        wp.array(dtype=vec2),  # tee_pos
        a1s,  # tee_theta
        wp.array(dtype=vec2),  # goal_pos
        a1s,  # goal_theta
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.array2d(dtype=wp.int32),  # roles
        a2s,  # n_obj
        a2s,  # occl
        wp.array2d(dtype=wp.uint8),  # sectors
        wp.array2d(dtype=wp.uint8),  # near
        wp.int32,  # n_agents
        *([dtype] * 16),  # the pusht weights, then the observer weights
        wp.int32,  # advance_prev
        wp.int32,  # full_pass
        a1s,  # prev_dist
        a1s,  # prev_ang
        a2s,  # prev_adist
        a2s,  # prev_standoff
        a2s,  # reward
        wp.array(dtype=wp.uint8),  # done
        a1s,  # dist_out
        a1s,  # ang_out
        wp.array3d(dtype=dtype),  # multiobj
        a1s,  # coverage_out
        a1s,  # visibility_out
        a1s,  # occlusion_out
        wp.array2d(dtype=wp.uint8),  # tee_seen
    ]


@wp.kernel
def lidar_sensor_kernel(
    theta: wp.array2d(dtype=Any),
    max_range: Any,
    ray_step: Any,
    angle_start: Any,
    body_frame: wp.int32,
    mount_offset: Any,
    min_range: Any,
    half_fov: wp.array2d(dtype=Any),
    fov_at_mount: wp.int32,
    noise_std: wp.array2d(dtype=Any),
    rng_state: wp.array2d(dtype=wp.uint32),
    ranges: wp.array3d(dtype=Any),
    hit_class: wp.array3d(dtype=wp.uint8),
):
    """Thread per (env, agent): turn the centre-ray scan into what the mounted sensor
    returns, in place. A hit within ``min_range`` of the mount (``mount_offset`` ahead of
    the centre along the heading) or on a ray more than this robot's ``half_fov`` (plus
    ``FOV_EPS``) off the heading reads as a miss (``max_range``, class 0); the remaining
    hits get Gaussian range noise of this robot's ``noise_std``, clamped to
    ``[0, max_range]``, from this thread's own RNG stream. A body-frame ray's angle off
    the heading is its fixed offset ``angle_start + r ray_step`` wrapped to
    ``(-pi, pi]``, so which rays a field of view keeps does not depend on the heading.
    With ``fov_at_mount`` the field of view is the sensor's instead: a hit is blanked when
    its bearing from the mount, off the heading, exceeds ``half_fov``."""
    e, a = wp.tid()
    n_rays = ranges.shape[2]
    zero = type(max_range)(0.0)
    hf = half_fov[e, a] + type(max_range)(FOV_EPS)
    sigma = noise_std[e, a]
    th = theta[e, a]
    base = zero
    if body_frame != 0:
        base = th
    mx = mount_offset * wp.cos(th)
    my = mount_offset * wp.sin(th)
    rng = rng_state[e, a]
    for r in range(n_rays):
        if hit_class[e, a, r] != wp.uint8(0):
            ang = base + angle_start + type(max_range)(r) * ray_step
            t = ranges[e, a, r]
            hx = t * wp.cos(ang) - mx
            hy = t * wp.sin(ang) - my
            blind = hx * hx + hy * hy < min_range * min_range
            off = ang - th
            if fov_at_mount != 0:
                off = wp.atan2(hy, hx) - th
            elif body_frame != 0:
                off = angle_start + type(max_range)(r) * ray_step
            if wp.abs(_wrap(off)) > hf:
                blind = True
            if blind:
                ranges[e, a, r] = max_range
                hit_class[e, a, r] = wp.uint8(0)
            elif sigma > zero:
                n = type(max_range)(wp.randn(rng))
                ranges[e, a, r] = wp.clamp(t + sigma * n, zero, max_range)
    rng_state[e, a] = rng


def _sensor_signature(dtype) -> list:
    return [
        wp.array2d(dtype=dtype),  # theta
        *([dtype] * 3),  # max_range, ray_step, angle_start
        wp.int32,  # body_frame
        *([dtype] * 2),  # mount_offset, min_range
        wp.array2d(dtype=dtype),  # half_fov
        wp.int32,  # fov_at_mount
        wp.array2d(dtype=dtype),  # noise_std
        wp.array2d(dtype=wp.uint32),  # rng_state
        wp.array3d(dtype=dtype),  # ranges
        wp.array3d(dtype=wp.uint8),  # hit_class
    ]


@wp.func
def _draw_range(u: float, lo: Any, hi: Any, old: Any):
    """``u`` in ``[0, 1)`` onto ``[lo, hi]``, or ``old`` for a fixed knob (``lo == hi``)."""
    out = old
    if hi > lo:
        out = lo + (hi - lo) * type(lo)(u)
    return out


@wp.kernel
def pusht_lidar_dr_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    fov_lo: Any,
    fov_hi: Any,
    noise_lo: Any,
    noise_hi: Any,
    scale_lo: Any,
    scale_hi: Any,
    bias_pos_std: Any,
    bias_theta_std: Any,
    half_fov: wp.array2d(dtype=Any),
    noise_std: wp.array2d(dtype=Any),
    pose_scale: wp.array2d(dtype=Any),
    bias_pos: wp.array2d(dtype=Any),
    bias_theta: wp.array2d(dtype=Any),
):
    """Thread per (env, robot), respawned envs only: this robot's half field of view and
    range-noise deviation, each uniform in its range (left alone when fixed); the env's
    pose-noise scale (one draw per env, so every robot of the env reads the same), and
    this robot's localisation bias for the episode, Gaussian with the scale times
    ``bias_pos_std`` per axis and ``bias_theta_std``."""
    e, a = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return
    na = half_fov.shape[1]
    rng = wp.rand_init(seed, e * (na + 1) + 1 + a)
    half_fov[e, a] = _draw_range(wp.randf(rng), fov_lo, fov_hi, half_fov[e, a])
    noise_std[e, a] = _draw_range(wp.randf(rng), noise_lo, noise_hi, noise_std[e, a])
    env_rng = wp.rand_init(seed, e * (na + 1))
    u = _draw_range(wp.randf(env_rng), scale_lo, scale_hi, scale_lo)
    pose_scale[e, a] = u
    bx = type(u)(wp.randn(rng))
    by = type(u)(wp.randn(rng))
    bt = type(u)(wp.randn(rng))
    bias_pos[e, a] = type(bias_pos[e, a])(u * bias_pos_std * bx, u * bias_pos_std * by)
    bias_theta[e, a] = u * bias_theta_std * bt


@wp.kernel
def pose_obs_noise_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    speed: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
    reset_mask: wp.array(dtype=wp.uint8),
    advance_prev: wp.int32,
    pose_scale: wp.array2d(dtype=Any),
    bias_pos: wp.array2d(dtype=Any),
    bias_theta: wp.array2d(dtype=Any),
    jitter_pos_std: Any,
    jitter_theta_std: Any,
    vel_std: Any,
    drift: wp.int32,
    dt: Any,
    drift_decay: Any,
    drift_decay_theta: Any,
    drift_pos_std: Any,
    drift_theta_std: Any,
    slip_gain_pos: Any,
    slip_gain_theta: Any,
    drift_state: wp.array3d(dtype=Any),
    rng_state: wp.array2d(dtype=wp.uint32),
    obs_pos: wp.array2d(dtype=Any),
    obs_vel: wp.array2d(dtype=Any),
    obs_theta: wp.array2d(dtype=Any),
):
    """Thread per (env, robot): the pose the robot's localisation reports, on a step and
    for the envs a reset respawned (a reset pass leaves the others' alone, as their
    state did not change). Position and heading are the true ones plus the episode's
    bias plus fresh Gaussian jitter of the env's scale times the jitter deviations; the
    world-frame velocity is the true one turned by the heading error (a body-frame
    odometry twist rotated by the reported heading) plus Gaussian noise of the scale
    times ``vel_std`` per axis.

    With ``drift`` on, a slowly decaying error adds to the bias: ``drift_state[e, a]``
    holds the robot's pose, speed and turn rate after the last pass (0-4) and the drift
    itself (5-7: x, y, heading). Each step the drift decays by ``drift_decay``
    (``exp(-dt / tau)``; ``drift_decay_theta`` for the heading), takes minus the scale
    times ``slip_gain_pos`` (``slip_gain_theta`` for the heading) times the step's slip
    (the motion its wheel odometry missed: the true displacement and turn minus the speed and
    turn rate averaged over the step, along the mid-step heading), and an
    Ornstein-Uhlenbeck kick whose stationary deviation is the scale times
    ``drift_pos_std`` per axis and ``drift_theta_std``. A respawn zeroes it."""
    e, a = wp.tid()
    if advance_prev == 0 and reset_mask[e] == wp.uint8(0):
        return
    rng = rng_state[e, a]
    u = pose_scale[e, a]
    jx = type(u)(wp.randn(rng))
    jy = type(u)(wp.randn(rng))
    jt = type(u)(wp.randn(rng))
    vx = type(u)(wp.randn(rng))
    vy = type(u)(wp.randn(rng))
    p = pos[e, a]
    th = theta[e, a]
    zero = type(u)(0.0)
    dp = type(p)(zero, zero)
    dtd = zero
    if drift == 1:
        nx = type(u)(wp.randn(rng))
        ny = type(u)(wp.randn(rng))
        nt = type(u)(wp.randn(rng))
        s1 = speed[e, a]
        w1 = ang_vel[e, a]
        if reset_mask[e] == wp.uint8(0):
            p0 = type(p)(drift_state[e, a, 0], drift_state[e, a, 1])
            th0 = drift_state[e, a, 2]
            half = type(u)(0.5)
            turn = _wrap(th - th0)
            thm = th0 + half * turn
            ds = half * (drift_state[e, a, 3] + s1) * dt
            slip = (p - p0) - type(p)(ds * wp.cos(thm), ds * wp.sin(thm))
            slip_th = turn - half * (drift_state[e, a, 4] + w1) * dt
            kick = wp.sqrt(type(u)(1.0) - drift_decay * drift_decay)
            kick_t = wp.sqrt(type(u)(1.0) - drift_decay_theta * drift_decay_theta)
            g = u * slip_gain_pos
            dp = drift_decay * type(p)(drift_state[e, a, 5], drift_state[e, a, 6]) - g * slip
            dp = dp + u * drift_pos_std * kick * type(p)(nx, ny)
            dtd = drift_decay_theta * drift_state[e, a, 7] - u * slip_gain_theta * slip_th
            dtd = dtd + u * drift_theta_std * kick_t * nt
        drift_state[e, a, 0] = p[0]
        drift_state[e, a, 1] = p[1]
        drift_state[e, a, 2] = th
        drift_state[e, a, 3] = s1
        drift_state[e, a, 4] = w1
        drift_state[e, a, 5] = dp[0]
        drift_state[e, a, 6] = dp[1]
        drift_state[e, a, 7] = dtd
    rng_state[e, a] = rng
    jp = u * jitter_pos_std
    obs_pos[e, a] = p + bias_pos[e, a] + dp + type(p)(jp * jx, jp * jy)
    dth = bias_theta[e, a] + dtd + u * jitter_theta_std * jt
    obs_theta[e, a] = th + dth
    v = vel[e, a]
    c = wp.cos(dth)
    s = wp.sin(dth)
    vs = u * vel_std
    obs_vel[e, a] = type(v)(c * v[0] - s * v[1] + vs * vx, s * v[0] + c * v[1] + vs * vy)


def _dr_reset_signature(dtype) -> list:
    a2s = wp.array2d(dtype=dtype)
    return [
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # use_mask
        wp.int32,  # seed
        *([dtype] * 8),  # fov, noise, scale ranges; bias deviations
        a2s,  # half_fov
        a2s,  # noise_std
        a2s,  # pose_scale
        wp.array2d(dtype=VEC2[dtype]),  # bias_pos
        a2s,  # bias_theta
    ]


def _pose_noise_signature(dtype) -> list:
    a2s = wp.array2d(dtype=dtype)
    a2v = wp.array2d(dtype=VEC2[dtype])
    return [
        a2v,  # pos
        a2v,  # vel
        a2s,  # theta
        a2s,  # speed
        a2s,  # ang_vel
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # advance_prev
        a2s,  # pose_scale
        a2v,  # bias_pos
        a2s,  # bias_theta
        *([dtype] * 3),  # jitter_pos_std, jitter_theta_std, vel_std
        wp.int32,  # drift
        *([dtype] * 7),  # dt, drift decay, std and slip gain (pos, theta each)
        wp.array3d(dtype=dtype),  # drift_state
        wp.array2d(dtype=wp.uint32),  # rng_state
        a2v,  # obs_pos
        a2v,  # obs_vel
        a2s,  # obs_theta
    ]


for _T in (wp.float32, wp.float64):
    register(pusht_lidar_dr_reset_kernel, _T, _dr_reset_signature(_T))
    register(pose_obs_noise_kernel, _T, _pose_noise_signature(_T))
    register(lidar_sensor_kernel, _T, _sensor_signature(_T))
    register(pusht_lidar_obs_kernel, _T, _obs_signature(_T))
    register(pusht_lidar_reward_kernel, _T, _reward_signature(_T))
