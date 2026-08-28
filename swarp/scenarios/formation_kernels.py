"""Fused Warp kernels for the FormationScenario obs/reward layer.

Replaces the eager-torch per-step cache (goal-relative obs, position shaping,
all-pairs touching count, in-formation flag, reward reduction) with an
observation kernel threaded per (env, agent) and a reward kernel threaded per
env (sequential agent loop; deterministic, no atomics). The torch implementation
in :mod:`swarp.scenarios.formation` stays the reference (and the differentiable
path).

The position shaping reuses the NavigationScenario machinery verbatim
(``prev_dist`` baseline, ``reset_hit`` rebase, ``advance_prev`` advance,
``full_pass`` gating of the reward/done/info buffers). Touching uses the static
``2 * agent_radius`` contact distance (matching the torch reference, whose
``cdist`` is pinned to the direct diff-norm path so ``wp.length`` matches it
bit-for-bit). Observation cat order::

    [ pos(2), vel(2), goal - pos(2) ]   # obs_dim = 6
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.state import VEC2
from swarp.scenarios.reset_kernels import _as


@wp.kernel
def formation_obs_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    goals: wp.array2d(dtype=Any),
    reset_mask: wp.array(dtype=wp.uint8),
    col_dist_sq: Any,
    goal_tolerance: Any,
    pos_shaping_factor: Any,
    n_agents: wp.int32,
    advance_prev: wp.int32,
    full_pass: wp.int32,
    obs: wp.array3d(dtype=Any),
    shaping: wp.array2d(dtype=Any),
    touching: wp.array2d(dtype=Any),
    dist_out: wp.array2d(dtype=Any),
    in_formation: wp.array(dtype=wp.uint8, ndim=2),
    prev_dist: wp.array2d(dtype=Any),
):
    """Thread per (env, agent): obs row + shaping/touching/in-formation buffers."""
    e, a = wp.tid()
    if full_pass == 0 and reset_mask[e] == wp.uint8(0):
        # Obs-only auto-reset pass: an env this mask didn't select has state
        # identical to what the STEP pass moments earlier already wrote into
        # every output buffer below, so redoing the all-pairs touch count and
        # shaping math for it is pure waste. Reset envs (reset_mask[e] == 1)
        # still fall through and get recomputed.
        return
    p = pos[e, a]
    v = vel[e, a]
    g = goals[e, a]
    px = p[0]
    py = p[1]
    grelx = g[0] - px
    grely = g[1] - py
    d = wp.sqrt(grelx * grelx + grely * grely)
    zero = type(px)(0.0)
    one = type(px)(1.0)

    # Observation row: own pos, vel, goal-relative vector.
    obs[e, a, 0] = px
    obs[e, a, 1] = py
    obs[e, a, 2] = v[0]
    obs[e, a, 3] = v[1]
    obs[e, a, 4] = grelx
    obs[e, a, 5] = grely

    # Touching: all-pairs count within the contact distance, minus self.
    # Squared comparison (no sqrt) so it matches the torch reference's
    # squared broadcast distance bit-for-bit at the < 2r boundary.
    cnt = zero
    for b in range(n_agents):
        dx = pos[e, b][0] - px
        dy = pos[e, b][1] - py
        if dx * dx + dy * dy < col_dist_sq:
            cnt += one
    touch = cnt - one

    # Position shaping (navigation pattern): baseline rebase / advance / gating.
    prev = prev_dist[e, a]
    ps = (prev - d) * pos_shaping_factor
    reset_hit = wp.int32(reset_mask[e])
    if reset_hit == 1:
        prev_dist[e, a] = d
        if full_pass == 1:
            shaping[e, a] = zero
    else:
        if advance_prev == 1:
            prev_dist[e, a] = d
        if full_pass == 1:
            shaping[e, a] = ps
    if full_pass == 1:
        touching[e, a] = touch
        dist_out[e, a] = d
        if d < goal_tolerance:
            in_formation[e, a] = wp.uint8(1)
        else:
            in_formation[e, a] = wp.uint8(0)


@wp.kernel
def formation_reward_kernel(
    shaping: wp.array2d(dtype=Any),
    touching: wp.array2d(dtype=Any),
    dist_in: wp.array2d(dtype=Any),
    in_formation: wp.array(dtype=wp.uint8, ndim=2),
    n_agents: wp.int32,
    collision_penalty: Any,
    inv_n_agents: Any,
    reward: wp.array2d(dtype=Any),
    done: wp.array(dtype=wp.uint8),
    multiobj: wp.array3d(dtype=Any),
    formation_error: wp.array(dtype=Any),
):
    """Thread per env; sequential agent loop (deterministic, no atomics)."""
    e = wp.tid()
    all_if = wp.uint8(1)
    dsum = type(collision_penalty)(0.0)
    for a in range(n_agents):
        sh = shaping[e, a]
        col = collision_penalty * touching[e, a]
        reward[e, a] = sh + col
        multiobj[e, a, 0] = sh
        multiobj[e, a, 1] = col
        dsum += dist_in[e, a]
        if in_formation[e, a] == wp.uint8(0):
            all_if = wp.uint8(0)
    done[e] = all_if
    formation_error[e] = dsum * inv_n_agents


def _obs_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    a2s = wp.array2d(dtype=dtype)
    a3s = wp.array3d(dtype=dtype)
    u8_2 = wp.array(dtype=wp.uint8, ndim=2)
    return [
        a2v,  # pos
        a2v,  # vel
        a2v,  # goals
        wp.array(dtype=wp.uint8),  # reset_mask
        dtype,  # col_dist_sq
        dtype,  # goal_tolerance
        dtype,  # pos_shaping_factor
        wp.int32,  # n_agents
        wp.int32,  # advance_prev
        wp.int32,  # full_pass
        a3s,  # obs
        a2s,  # shaping
        a2s,  # touching
        a2s,  # dist_out
        u8_2,  # in_formation
        a2s,  # prev_dist
    ]


def _reward_signature(dtype) -> list:
    a2s = wp.array2d(dtype=dtype)
    return [
        a2s,  # shaping
        a2s,  # touching
        a2s,  # dist_in
        wp.array(dtype=wp.uint8, ndim=2),  # in_formation
        wp.int32,  # n_agents
        dtype,  # collision_penalty
        dtype,  # inv_n_agents
        a2s,  # reward
        wp.array(dtype=wp.uint8),  # done
        wp.array3d(dtype=dtype),  # multiobj
        wp.array(dtype=dtype),  # formation_error
    ]



@wp.kernel
def formation_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    lim: Any,
    clim: Any,
    n_agents: wp.int32,
    slot_offsets: Any,
    pos: Any,
    vel: Any,
    goals: Any,
):
    """Masked episode reset: uniform spawns, zero velocity, formation slots.

    The formation centre is drawn **once per env** and every slot is that centre plus its
    fixed offset — which is why this is a thread per env rather than per agent.
    """
    e = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return
    rng = wp.rand_init(seed, e)
    zero = _as(0.0, lim)
    two = _as(2.0, lim)
    one = _as(1.0, lim)

    cx = (_as(wp.randf(rng), clim) * two - one) * clim
    cy = (_as(wp.randf(rng), clim) * two - one) * clim
    for a in range(n_agents):
        px = (_as(wp.randf(rng), lim) * two - one) * lim
        py = (_as(wp.randf(rng), lim) * two - one) * lim
        pos[e, a] = wp.vector(px, py)
        vel[e, a] = wp.vector(zero, zero)
        off = slot_offsets[a]
        goals[e, a] = wp.vector(cx + off[0], cy + off[1])


def _reset_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    return [
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # use_mask
        wp.int32,  # seed
        dtype,  # lim
        dtype,  # clim
        wp.int32,  # n_agents
        wp.array(dtype=VEC2[dtype]),  # slot_offsets
        a2v,  # pos
        a2v,  # vel
        a2v,  # goals
    ]

for _T in (wp.float32, wp.float64):
    register(formation_obs_kernel, _T, _obs_signature(_T))
    register(formation_reward_kernel, _T, _reward_signature(_T))
    register(formation_reset_kernel, _T, _reset_signature(_T))
