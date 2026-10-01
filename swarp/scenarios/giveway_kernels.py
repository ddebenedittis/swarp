"""Fused Warp kernels for the GiveWayScenario obs/reward/done layer.

Same three-kernel shape as :mod:`swarp.scenarios.navigation_kernels` — a per-``(env,
agent)`` obs kernel, a per-env reward reduction, and a masked per-env reset — and the
torch implementation in :mod:`swarp.scenarios.giveway` stays the reference (and the
differentiable path) these are validated against.

The one piece of geometry worth reading before touching anything: the four corner blocks
that shape the plus-shaped corridor are **mirror images of each other about both axes**,
so the distance from an agent to the *nearest* block is a single box SDF evaluated at
``(|x|, |y|)`` against the first-quadrant block. :func:`_wall_gap` does exactly that —
one SDF, no obstacle-array read, no four-shape loop — which is why the reward's
wall-contact term costs the same whether the arena has four blocks or four hundred. The
real obstacles are still installed in ``make_world``; they are what actually blocks the
agents, and this SDF only feeds the reward flag.

Observation row (``obs_dim = 10 + 5 * k_obs``), matching the torch ``cat`` order::

    [ pos(2), vel(2), cosθ, sinθ, ang_vel, goal-pos(2), politeness,
      rel_pos(2·k_obs), rel_vel(2·k_obs), valid(k_obs) ]

``politeness`` is not decoration: see :class:`~swarp.scenarios.giveway.GiveWayScenario`
for why a symmetry-breaking scalar is load-bearing in a one-lane corridor.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.rng import seed_from_state
from swarp.core.state import VEC2
from swarp.dynamics.base import P_RADIUS
from swarp.scenarios.reset_kernels import _as

#: Objective columns of ``info()["multiobj_reward"]``, in order. Kept here next to the
#: kernel that writes them so the two cannot drift apart.
OBJ_SHAPING = 0
OBJ_COLLISION = 1
OBJ_WALL = 2
OBJ_TIME = 3
OBJ_FINAL = 4
N_OBJ = 5


@wp.func
def _wall_gap(px: Any, py: Any, half_w: Any, arena: Any):
    """Signed distance from ``(px, py)`` to the nearest corner block.

    The four blocks each span ``[half_w, arena] x [half_w, arena]`` in their own quadrant,
    so they are related by the two axis reflections. Folding the query point into the
    first quadrant with ``abs`` therefore turns "nearest of four boxes" into "this one
    box", and the whole term is one box SDF:

    * ``dx``/``dy`` are the per-axis signed distances to the slab ``[half_w, arena]``,
      written as ``max(lo - a, a - hi)`` rather than ``|a - centre| - half`` because the
      slab bounds are the numbers this scenario actually parametrizes;
    * the positive parts give the Euclidean distance outside the box (correct around the
      corners, which is precisely where an agent cuts across the junction);
    * ``min(max(dx, dy), 0)`` adds the negative interior distance, which stays 0 outside.

    Positive = free space, negative = inside a block. Because it is a *fold*, not a
    distance to one specific block, it also reads correctly for an agent sitting in the
    junction: it returns the distance to the nearest of the four inner corners.
    """
    zero = type(px)(0.0)
    ax = wp.abs(px)
    ay = wp.abs(py)
    dx = wp.max(half_w - ax, ax - arena)
    dy = wp.max(half_w - ay, ay - arena)
    ox = wp.max(dx, zero)
    oy = wp.max(dy, zero)
    return wp.sqrt(ox * ox + oy * oy) + wp.min(wp.max(dx, dy), zero)


@wp.func
def _obs_row(
    e: wp.int32,
    a: wp.int32,
    p: Any,
    v: Any,
    th: Any,
    wv: Any,
    grel: Any,
    polite: Any,
    k_obs: wp.int32,
    cnt: wp.int32,
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    neighbor_idx: wp.array3d(dtype=wp.int32),
    obs: wp.array3d(dtype=Any),
):
    """Write one agent's observation row (own features + politeness + up to ``k_obs``
    neighbor features, invalid slots zeroed)."""
    zero = type(th)(0.0)
    obs[e, a, 0] = p[0]
    obs[e, a, 1] = p[1]
    obs[e, a, 2] = v[0]
    obs[e, a, 3] = v[1]
    obs[e, a, 4] = wp.cos(th)
    obs[e, a, 5] = wp.sin(th)
    obs[e, a, 6] = wv
    obs[e, a, 7] = grel[0]
    obs[e, a, 8] = grel[1]
    obs[e, a, 9] = polite
    rp_base = wp.int32(10)
    rv_base = wp.int32(10) + wp.int32(2) * k_obs
    vd_base = wp.int32(10) + wp.int32(4) * k_obs
    for j in range(k_obs):
        rpx = zero
        rpy = zero
        rvx = zero
        rvy = zero
        valid = zero
        if j < cnt:
            b = neighbor_idx[e, a, j]
            rpx = pos[e, b][0] - p[0]
            rpy = pos[e, b][1] - p[1]
            rvx = vel[e, b][0] - v[0]
            rvy = vel[e, b][1] - v[1]
            valid = type(th)(1.0)
        obs[e, a, rp_base + wp.int32(2) * j] = rpx
        obs[e, a, rp_base + wp.int32(2) * j + 1] = rpy
        obs[e, a, rv_base + wp.int32(2) * j] = rvx
        obs[e, a, rv_base + wp.int32(2) * j + 1] = rvy
        obs[e, a, vd_base + j] = valid


@wp.kernel
def giveway_obs_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
    goals: wp.array2d(dtype=Any),
    politeness: wp.array2d(dtype=Any),
    neighbor_idx: wp.array3d(dtype=wp.int32),
    neighbor_count: wp.array2d(dtype=wp.int32),
    params: wp.array2d(dtype=Any),  # [n_agents, NUM_PARAMS] shared
    reset_mask: wp.array(dtype=wp.uint8),
    k_obs: wp.int32,
    half_w: Any,
    arena: Any,
    wall_thresh: Any,
    pos_shaping_factor: Any,
    goal_tolerance: Any,
    advance_prev: wp.int32,
    full_pass: wp.int32,
    obs: wp.array3d(dtype=Any),
    touching: wp.array2d(dtype=Any),
    wall_contact: wp.array2d(dtype=Any),
    dist_to_goal: wp.array2d(dtype=Any),
    pos_shaping: wp.array2d(dtype=Any),
    on_goal: wp.array(dtype=wp.uint8, ndim=2),
    prev_dist: wp.array2d(dtype=Any),
):
    """Observation row plus every per-agent reward input, one thread per ``(env, agent)``.

    ``reset_hit``/``advance_prev``/``full_pass`` follow the navigation kernel's contract
    verbatim: a just-reset env rebases the shaping baseline and zeroes the shaping term,
    an obs-only auto-reset pass leaves the reward-input buffers alone, and every other env
    on such a pass is skipped outright because its state has not moved since the step pass
    wrote those buffers moments earlier.
    """
    e, a = wp.tid()
    if full_pass == 0 and reset_mask[e] == wp.uint8(0):
        return
    p = pos[e, a]
    v = vel[e, a]
    g = goals[e, a]
    grel = g - p
    d = wp.length(p - g)
    cnt = neighbor_count[e, a]
    ra = params[a, P_RADIUS]
    zero = type(d)(0.0)
    one = type(d)(1.0)

    touch = zero
    for ni in range(cnt):
        b = neighbor_idx[e, a, ni]
        nd = wp.length(pos[e, b] - p)
        if nd < ra + params[b, P_RADIUS]:
            touch += one

    # ``wall_thresh`` is ``agent_radius + wall_margin`` as a *scalar*, not
    # ``params[a, P_RADIUS] + margin``: the corridor width is derived from the scenario's
    # single ``agent_radius`` (the ``agent_radius < half_w < 2 * agent_radius``
    # constraint), so a per-agent radius would describe a corridor that does not exist.
    wallf = zero
    if _wall_gap(p[0], p[1], half_w, arena) < wall_thresh:
        wallf = one

    _obs_row(
        e, a, p, v, theta[e, a], ang_vel[e, a], grel, politeness[e, a],
        k_obs, cnt, pos, vel, neighbor_idx, obs,
    )

    prev = prev_dist[e, a]
    ps = (prev - d) * pos_shaping_factor
    if reset_mask[e] == wp.uint8(1):
        prev_dist[e, a] = d
        if full_pass == 1:
            pos_shaping[e, a] = zero
    else:
        if advance_prev == 1:
            prev_dist[e, a] = d
        if full_pass == 1:
            pos_shaping[e, a] = ps
    if full_pass == 1:
        touching[e, a] = touch
        wall_contact[e, a] = wallf
        dist_to_goal[e, a] = d
        if d < goal_tolerance:
            on_goal[e, a] = wp.uint8(1)
        else:
            on_goal[e, a] = wp.uint8(0)


@wp.kernel
def giveway_reward_kernel(
    touching: wp.array2d(dtype=Any),
    wall_contact: wp.array2d(dtype=Any),
    pos_shaping: wp.array2d(dtype=Any),
    on_goal: wp.array(dtype=wp.uint8, ndim=2),
    n_agents: wp.int32,
    collision_penalty: Any,
    wall_penalty: Any,
    time_penalty: Any,
    final_reward: Any,
    shared_reward: wp.int32,
    reward: wp.array2d(dtype=Any),
    done: wp.array(dtype=wp.uint8),
    multiobj: wp.array3d(dtype=Any),
):
    """Thread per env; sequential agent loops (deterministic, no atomics).

    ``multiobj`` keeps the five terms separate and its last-dim sum is *exactly*
    ``reward``, because that is what makes it useful: a trainer logs one column per term
    and sees a shaping imbalance on iteration 1 instead of hour 3. The identity is written
    out as one addition of the five columns rather than accumulated separately, so it
    cannot drift.
    """
    e = wp.tid()
    zero = type(collision_penalty)(0.0)
    shaping_sum = zero
    all_og = wp.uint8(1)
    for a in range(n_agents):
        shaping_sum += pos_shaping[e, a]
        if on_goal[e, a] == wp.uint8(0):
            all_og = wp.uint8(0)
    final = zero
    if all_og == wp.uint8(1):
        final = final_reward
    for a in range(n_agents):
        sh = shaping_sum
        if shared_reward == 0:
            sh = pos_shaping[e, a]
        col = collision_penalty * touching[e, a]
        wal = wall_penalty * wall_contact[e, a]
        multiobj[e, a, OBJ_SHAPING] = sh
        multiobj[e, a, OBJ_COLLISION] = col
        multiobj[e, a, OBJ_WALL] = wal
        multiobj[e, a, OBJ_TIME] = time_penalty
        multiobj[e, a, OBJ_FINAL] = final
        reward[e, a] = sh + col + wal + time_penalty + final
    done[e] = all_og


@wp.kernel
def giveway_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed_state: wp.array(dtype=wp.int32),
    arm_len: Any,
    spacing: Any,
    min_dist: Any,
    long_jitter: wp.array(dtype=Any),
    lat_jitter: Any,
    n_agents: wp.int32,
    pos: Any,
    theta: Any,
    vel: Any,
    speed: Any,
    ang_vel: Any,
    goals: Any,
    politeness: Any,
):
    """One masked episode reset per env: queued spawns, mirrored goals, arm headings,
    zeroed velocities, and a fresh politeness draw.

    Thread per **env** (see ``swarp/scenarios/reset_kernels.py``): the draws are per-agent
    but they share one RNG stream, and one launch per env is what keeps ``auto_reset``'s
    every-step reset off the launch-count budget.

    Agent ``a`` is queued at slot ``a / 4`` of arm ``a % 4`` (N, E, S, W) and its goal is
    the point reflection of its **nominal** (un-jittered) slot through the origin — i.e.
    the matching slot of the opposite arm. The arm assignment is deliberately *not*
    randomized: the agents are homogeneous, so permuting who gets which arm relabels the
    problem without changing it, and a policy cannot learn anything from the relabelling.
    What *is* randomized is the two jitters:

    * ``long_jitter`` pulls a spawn toward the junction, staggering who arrives first.
      This is the curriculum knob, derived by the caller from ``difficulty``, and it is a
      one-element **array** rather than a scalar for the same reason ``seed_state`` is (see
      ``swarp/core/rng.py``): this kernel is captured into the whole-step graph by
      ``reset_in_graph``, a scalar argument is baked into the capture *by value*, and a
      trainer that moved ``difficulty`` between batches would silently keep getting the
      stagger the graph was captured with. An array is baked by pointer, so the replay
      reads whatever the pointer holds at replay time.
    * ``lat_jitter`` offsets the spawn across the corridor. The caller bounds it by
      ``half_w - agent_radius``; anything larger spawns an agent inside a block.

    The goal is the mirror of the nominal slot, so neither jitter moves it: the task is
    "reach the opposite arm", not "reach wherever you happened to start, reflected".

    Headings are the exact per-arm values (``-π/2``, ``π``, ``π/2``, ``0``) rather than an
    ``atan2`` of the inward direction — four constants beat a transcendental, and they are
    exactly representable arguments for the ``cos``/``sin`` the observation takes.

    ``seed_state`` is the device-resident ``[base, counter]`` pair (see
    ``swarp/core/rng.py``); the caller launches ``advance_seed_kernel`` on it immediately
    before this kernel, every call — advance, then use — which is what lets a captured
    graph replay this launch with a different draw each time.
    """
    e = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return

    rng = wp.rand_init(seed_from_state(seed_state), e)
    # Typed constants: Warp reads a bare float literal as float32, so every literal that
    # meets the world's scalar type has to be widened through ``_as`` first.
    zero = _as(0.0, arm_len)
    one = _as(1.0, arm_len)
    two = _as(2.0, arm_len)
    pi = _as(3.14159265358979, arm_len)
    half_pi = _as(1.57079632679490, arm_len)

    for a in range(n_agents):
        arm = a % 4
        slot = a / 4

        # Outward unit vector of this arm, and the heading that points back down it
        # toward the junction.
        ux = zero
        uy = zero
        th = zero
        if arm == 0:  # north arm, faces south
            uy = one
            th = -half_pi
        elif arm == 1:  # east arm, faces west
            ux = one
            th = pi
        elif arm == 2:  # south arm, faces north
            uy = -one
            th = half_pi
        else:  # west arm, faces east
            ux = -one

        base = arm_len - _as(wp.float32(slot), arm_len) * spacing
        # Draw inline, never through a @wp.func: Warp passes the RNG state by value, so a
        # helper would advance a local copy and hand back the same number every call.
        d = base - _as(wp.randf(rng), arm_len) * long_jitter[0]
        if d < min_dist:
            d = min_dist
        lat = (_as(wp.randf(rng), arm_len) * two - one) * lat_jitter

        # Lateral axis is the arm's left normal (-uy, ux).
        pos[e, a] = wp.vector(ux * d - uy * lat, uy * d + ux * lat)
        theta[e, a] = th
        vel[e, a] = wp.vector(zero, zero)
        speed[e, a] = zero
        ang_vel[e, a] = zero
        goals[e, a] = wp.vector(-ux * base, -uy * base)

    # Politeness after the geometry, in its own loop: it is per-agent scenario state, not
    # part of the spawn, and keeping the draws grouped makes the stream easy to reason
    # about when the geometry changes.
    for a in range(n_agents):
        politeness[e, a] = _as(wp.randf(rng), arm_len)


def _obs_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    a2s = wp.array2d(dtype=dtype)
    a3i = wp.array3d(dtype=wp.int32)
    a2i = wp.array2d(dtype=wp.int32)
    a3s = wp.array3d(dtype=dtype)
    return [
        a2v,  # pos
        a2v,  # vel
        a2s,  # theta
        a2s,  # ang_vel
        a2v,  # goals
        a2s,  # politeness
        a3i,  # neighbor_idx
        a2i,  # neighbor_count
        a2s,  # params
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # k_obs
        dtype,  # half_w
        dtype,  # arena
        dtype,  # wall_thresh
        dtype,  # pos_shaping_factor
        dtype,  # goal_tolerance
        wp.int32,  # advance_prev
        wp.int32,  # full_pass
        a3s,  # obs
        a2s,  # touching
        a2s,  # wall_contact
        a2s,  # dist_to_goal
        a2s,  # pos_shaping
        wp.array(dtype=wp.uint8, ndim=2),  # on_goal
        a2s,  # prev_dist
    ]


def _reward_signature(dtype) -> list:
    a2s = wp.array2d(dtype=dtype)
    return [
        a2s,  # touching
        a2s,  # wall_contact
        a2s,  # pos_shaping
        wp.array(dtype=wp.uint8, ndim=2),  # on_goal
        wp.int32,  # n_agents
        dtype,  # collision_penalty
        dtype,  # wall_penalty
        dtype,  # time_penalty
        dtype,  # final_reward
        wp.int32,  # shared_reward
        a2s,  # reward
        wp.array(dtype=wp.uint8),  # done
        wp.array3d(dtype=dtype),  # multiobj
    ]


def _reset_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    a2s = wp.array2d(dtype=dtype)
    return [
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # use_mask
        wp.array(dtype=wp.int32),  # seed_state
        dtype,  # arm_len
        dtype,  # spacing
        dtype,  # min_dist
        wp.array(dtype=dtype),  # long_jitter (by pointer: read at graph *replay* time)
        dtype,  # lat_jitter
        wp.int32,  # n_agents
        a2v,  # pos
        a2s,  # theta
        a2v,  # vel
        a2s,  # speed
        a2s,  # ang_vel
        a2v,  # goals
        a2s,  # politeness
    ]


for _T in (wp.float32, wp.float64):
    register(giveway_obs_kernel, _T, _obs_signature(_T))
    register(giveway_reward_kernel, _T, _reward_signature(_T))
    register(giveway_reset_kernel, _T, _reset_signature(_T))
