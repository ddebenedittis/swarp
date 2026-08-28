"""Fused Warp obs/reward kernels for PushTScenario.

Two launches, ``{obs, reward}``. The T's own rigid-body physics is **not** here: it is a
movable compound obstacle integrated by :mod:`swarp.core.bodies` inside the substep loop
(see :mod:`swarp.scenarios.pusht`), which is what keeps the pose an agent collides against
at most one substep old. These kernels only read the body state the engine produced.

* ``pusht_obs_kernel`` (thread per (env, agent)) builds the obs row.
* ``pusht_reward_kernel`` (thread per env) reduces position **and** orientation shaping,
  the joint pose "crater", the per-agent approach term and the pose bonus, using the
  navigation ``prev_dist`` / ``reset_hit`` / ``advance_prev`` / ``full_pass`` machinery
  extended to the angular and per-agent carries. No atomics.

The torch implementation in :mod:`swarp.scenarios.pusht` remains the differentiable
reference: on a taped step the engine leaves movable bodies alone and the scenario
integrates the T in plain torch instead. On the no-grad path both the fused and the torch
paths read the *same* engine body state, so they agree exactly rather than approximately.

Observation cat order (``obs_dim = 12 + 2*(n_agents-1)``)::

    [ pos(2), vel(2), tee_rel(2), tee_to_goal(2), cos/sin(theta)(2), cos/sin(ang_err)(2),
      teammate_rel(2 each, ascending agent index, self skipped) ]
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.state import VEC2
from swarp.scenarios.reset_kernels import _as


@wp.kernel
def pusht_obs_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    tee_pos: wp.array(dtype=Any),
    tee_theta: wp.array(dtype=Any),
    goal_pos: wp.array(dtype=Any),
    goal_theta: wp.array(dtype=Any),
    obs: wp.array3d(dtype=Any),
):
    """Thread per (env, agent): obs row (agent pose, T-relative, T->goal, angles)."""
    e, a = wp.tid()
    p = pos[e, a]
    v = vel[e, a]
    q = tee_pos[e]
    g = goal_pos[e]
    th = tee_theta[e]
    err = th - goal_theta[e]
    obs[e, a, 0] = p[0]
    obs[e, a, 1] = p[1]
    obs[e, a, 2] = v[0]
    obs[e, a, 3] = v[1]
    obs[e, a, 4] = q[0] - p[0]
    obs[e, a, 5] = q[1] - p[1]
    obs[e, a, 6] = g[0] - q[0]
    obs[e, a, 7] = g[1] - q[1]
    obs[e, a, 8] = wp.cos(th)
    obs[e, a, 9] = wp.sin(th)
    obs[e, a, 10] = wp.cos(err)
    obs[e, a, 11] = wp.sin(err)
    # Teammates' positions relative to this agent, ascending index, self skipped.
    n_agents = pos.shape[1]
    k = wp.int32(12)
    for j in range(n_agents):
        if j != a:
            q2 = pos[e, j]
            obs[e, a, k] = q2[0] - p[0]
            obs[e, a, k + 1] = q2[1] - p[1]
            k += wp.int32(2)


@wp.kernel
def pusht_reward_kernel(
    pos: wp.array2d(dtype=Any),
    tee_pos: wp.array(dtype=Any),
    tee_theta: wp.array(dtype=Any),
    goal_pos: wp.array(dtype=Any),
    goal_theta: wp.array(dtype=Any),
    reset_mask: wp.array(dtype=wp.uint8),
    n_agents: wp.int32,
    pos_shaping_factor: Any,
    rot_shaping_factor: Any,
    agent_dist_shaping: Any,
    push_point_offset: Any,
    joint_shaping: Any,
    goal_tolerance: Any,
    angle_tolerance: Any,
    goal_reward: Any,
    advance_prev: wp.int32,
    full_pass: wp.int32,
    prev_dist: wp.array(dtype=Any),
    prev_ang: wp.array(dtype=Any),
    prev_adist: wp.array2d(dtype=Any),
    reward: wp.array2d(dtype=Any),
    done: wp.array(dtype=wp.uint8),
    dist_out: wp.array(dtype=Any),
    ang_out: wp.array(dtype=Any),
):
    """Thread per env; pose (position + orientation) shaping plus a per-agent
    approach-shaping term (decrease of the agent's distance to the T centre),
    no atomics."""
    e = wp.tid()
    zero = type(pos_shaping_factor)(0.0)
    tp = tee_pos[e]
    gx = goal_pos[e][0] - tp[0]
    gy = goal_pos[e][1] - tp[1]
    d = wp.sqrt(gx * gx + gy * gy)
    # wrap the heading error to (-pi, pi]; the T has no rotational symmetry, so
    # the absolute wrapped error is the whole story.
    raw = tee_theta[e] - goal_theta[e]
    ang = wp.abs(wp.atan2(wp.sin(raw), wp.cos(raw)))

    shaping = zero
    reset_hit = wp.int32(reset_mask[e])
    ps = (prev_dist[e] - d) * pos_shaping_factor + (prev_ang[e] - ang) * rot_shaping_factor
    # Joint pose "crater": potential-based Gaussian bump around the full pose goal
    # (1.5x the tolerances), the only term coupling position and orientation.
    sd = type(zero)(1.5) * goal_tolerance
    sa = type(zero)(1.5) * angle_tolerance
    pd = prev_dist[e]
    pa = prev_ang[e]
    ps += joint_shaping * (
        wp.exp(-(d / sd) * (d / sd) - (ang / sa) * (ang / sa))
        - wp.exp(-(pd / sd) * (pd / sd) - (pa / sa) * (pa / sa))
    )
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
    global_r = shaping + bonus

    # Pushing point: push_point_offset past the T centre, directly away from the
    # goal — approaching it puts the agent in pushing position, not merely in contact.
    inv_d = type(pos_shaping_factor)(1.0) / wp.max(d, type(pos_shaping_factor)(1.0e-6))
    bx = tp[0] - gx * inv_d * push_point_offset
    by = tp[1] - gy * inv_d * push_point_offset

    # Per-agent approach shaping shares the reset/advance carry machinery above.
    for a in range(n_agents):
        p = pos[e, a]
        adx = p[0] - bx
        ady = p[1] - by
        ad = wp.sqrt(adx * adx + ady * ady)
        ash = (prev_adist[e, a] - ad) * agent_dist_shaping
        if reset_hit == 1:
            prev_adist[e, a] = ad
            ash = zero
        else:
            if advance_prev == 1:
                prev_adist[e, a] = ad
        if full_pass == 1:
            reward[e, a] = global_r + ash

    if full_pass == 1:
        dist_out[e] = d
        ang_out[e] = ang
        done[e] = on_goal


def _obs_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    return [
        wp.array2d(dtype=vec2),  # pos
        wp.array2d(dtype=vec2),  # vel
        wp.array(dtype=vec2),  # tee_pos
        wp.array(dtype=dtype),  # tee_theta
        wp.array(dtype=vec2),  # goal_pos
        wp.array(dtype=dtype),  # goal_theta
        wp.array3d(dtype=dtype),  # obs
    ]


def _reward_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a1s = wp.array(dtype=dtype)
    return [
        wp.array2d(dtype=vec2),  # pos
        wp.array(dtype=vec2),  # tee_pos
        a1s,  # tee_theta
        wp.array(dtype=vec2),  # goal_pos
        a1s,  # goal_theta
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # n_agents
        dtype,  # pos_shaping_factor
        dtype,  # rot_shaping_factor
        dtype,  # agent_dist_shaping
        dtype,  # push_point_offset
        dtype,  # joint_shaping
        dtype,  # goal_tolerance
        dtype,  # angle_tolerance
        dtype,  # goal_reward
        wp.int32,  # advance_prev
        wp.int32,  # full_pass
        a1s,  # prev_dist
        a1s,  # prev_ang
        wp.array2d(dtype=dtype),  # prev_adist
        wp.array2d(dtype=dtype),  # reward
        wp.array(dtype=wp.uint8),  # done
        a1s,  # dist_out
        a1s,  # ang_out
    ]



@wp.kernel
def pusht_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    lim: Any,
    tlim: Any,
    clear: Any,
    goal_radius: Any,
    goal_angle: Any,
    use_goal_radius: wp.int32,
    use_goal_angle: wp.int32,
    n_agents: wp.int32,
    pos: Any,
    vel: Any,
    tee_pos: Any,
    tee_vel: Any,
    tee_theta: Any,
    tee_ang_vel: Any,
    goal_pos: Any,
    goal_theta: Any,
):
    """Masked episode reset: T pose, goal pose, and spawns cleared of the T.

    Any agent inside the T's bounding disk is pushed out to its rim before the episode
    starts: with a stiff ``contact_k`` a spawn overlap is a violent ejection
    (``k * depth * sub_dt`` is metres per second), so the reset must not start
    interpenetrating. An agent exactly on the centre goes straight up — arbitrary, but
    deterministic, matching the torch reset this replaces.

    ``use_goal_radius``/``use_goal_angle`` select the curriculum draws: a goal uniform in
    the square, or uniform in a disk of ``goal_radius`` around the T; a goal heading
    uniform, or within ``goal_angle`` of the T's own.
    """
    e = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return
    rng = wp.rand_init(seed, e)
    zero = _as(0.0, lim)
    two = _as(2.0, lim)
    one = _as(1.0, lim)
    pi = _as(3.14159265358979, lim)

    tx = (_as(wp.randf(rng), tlim) * two - one) * tlim
    ty = (_as(wp.randf(rng), tlim) * two - one) * tlim
    tee = wp.vector(tx, ty)
    theta = (_as(wp.randf(rng), lim) * two - one) * pi

    tee_pos[e] = tee
    tee_vel[e] = wp.vector(zero, zero)
    tee_theta[e] = theta
    tee_ang_vel[e] = zero

    for a in range(n_agents):
        px = (_as(wp.randf(rng), lim) * two - one) * lim
        py = (_as(wp.randf(rng), lim) * two - one) * lim
        dx = px - tx
        dy = py - ty
        dn = wp.sqrt(dx * dx + dy * dy)
        if dn < clear:
            if dn > _as(1.0e-9, lim):
                px = tx + dx / dn * clear
                py = ty + dy / dn * clear
            else:
                px = tx
                py = ty + clear
        px = wp.clamp(px, -lim, lim)
        py = wp.clamp(py, -lim, lim)
        pos[e, a] = wp.vector(px, py)
        vel[e, a] = wp.vector(zero, zero)

    if use_goal_radius == 1:
        gdir = (_as(wp.randf(rng), lim) * two - one) * pi
        grad = goal_radius * wp.sqrt(_as(wp.randf(rng), lim))
        gx = wp.clamp(tx + grad * wp.cos(gdir), -tlim, tlim)
        gy = wp.clamp(ty + grad * wp.sin(gdir), -tlim, tlim)
    else:
        gx = (_as(wp.randf(rng), tlim) * two - one) * tlim
        gy = (_as(wp.randf(rng), tlim) * two - one) * tlim
    goal_pos[e] = wp.vector(gx, gy)

    if use_goal_angle == 1:
        goal_theta[e] = theta + (_as(wp.randf(rng), lim) * two - one) * goal_angle
    else:
        goal_theta[e] = (_as(wp.randf(rng), lim) * two - one) * pi


def _reset_signature(dtype) -> list:
    a1v = wp.array(dtype=VEC2[dtype])
    a1s = wp.array(dtype=dtype)
    a2v = wp.array2d(dtype=VEC2[dtype])
    return [
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # use_mask
        wp.int32,  # seed
        dtype,  # lim
        dtype,  # tlim
        dtype,  # clear
        dtype,  # goal_radius
        dtype,  # goal_angle
        wp.int32,  # use_goal_radius
        wp.int32,  # use_goal_angle
        wp.int32,  # n_agents
        a2v,  # pos
        a2v,  # vel
        a1v,  # tee_pos
        a1v,  # tee_vel
        a1s,  # tee_theta
        a1s,  # tee_ang_vel
        a1v,  # goal_pos
        a1s,  # goal_theta
    ]

for _T in (wp.float32, wp.float64):
    register(pusht_obs_kernel, _T, _obs_signature(_T))
    register(pusht_reward_kernel, _T, _reward_signature(_T))
    register(pusht_reset_kernel, _T, _reset_signature(_T))
