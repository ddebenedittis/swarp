"""Fused Warp kernels for the TransportScenario obs/reward + movable-body layer.

Three launches, ordered ``body -> {obs, reward}`` (with ``set_obstacles`` between
``body`` and the next physics step, on the host):

* ``transport_body_kernel`` (thread per env,package) sums the spring-damper
  reaction force + torque the package receives from every agent (Newton's third
  law of the same soft contact the agent step applies) and integrates the
  package pose in place (semi-implicit Euler + damping), mutating
  ``pkg_pos/vel/theta/ang_vel``. Each package is owned by one thread — no
  cross-thread races; the per-agent reduction loops ``0..A-1``.
* ``transport_obs_kernel`` (thread per env,agent) builds the obs row.
* ``transport_reward_kernel`` (thread per env) reduces the per-package position
  shaping + goal bonus, using the navigation ``prev_dist`` / ``reset_hit`` /
  ``advance_prev`` / ``full_pass`` machinery.

The torch implementation in :mod:`swarp.scenarios.transport` stays the reference
(and the differentiable path — package gradients flow through the plain-torch
body physics). The fused body kernel is the no-grad fast path; because it sums
the agent reactions in a different order than torch's ``.sum(dim=1)``, the fused
and torch package trajectories can differ at the ulp scale, kept bounded by the
linear/angular damping (validated allclose, not bit-exact).

Observation cat order (``obs_dim = 4 + 4 * n_packages``)::

    [ pos(2), vel(2), pkg_rel(2*K), pkg_to_goal(2*K) ]
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp.core.state import VEC2


@wp.kernel
def transport_body_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    n_agents: wp.int32,
    agent_radius: Any,
    package_radius: Any,
    contact_margin: Any,
    contact_k: Any,
    contact_c: Any,
    package_mass: Any,
    package_inertia: Any,
    linear_damping: Any,
    angular_damping: Any,
    dt: Any,
    bound: Any,
    pkg_pos: wp.array2d(dtype=Any),
    pkg_vel: wp.array2d(dtype=Any),
    pkg_theta: wp.array2d(dtype=Any),
    pkg_ang_vel: wp.array2d(dtype=Any),
):
    """Thread per (env, package): accumulate contact force/torque, integrate."""
    e, k = wp.tid()
    q = pkg_pos[e, k]
    pv = pkg_vel[e, k]
    qx = q[0]
    qy = q[1]
    zero = type(qx)(0.0)
    one = type(qx)(1.0)
    reach = agent_radius + package_radius + contact_margin
    eps = type(qx)(1.0e-9)

    fx = zero
    fy = zero
    tau = zero
    for a in range(n_agents):
        relx = qx - pos[e, a][0]
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
        forcex = coeff * nhx
        forcey = coeff * nhy
        fx += forcex
        fy += forcey
        # torque about the package centre; contact point ~ -package_radius * n_hat
        rcx = -package_radius * nhx
        rcy = -package_radius * nhy
        tau += (rcx * forcey - rcy * forcex) * active

    lin_decay = one - linear_damping * dt
    ang_decay = one - angular_damping * dt
    new_vx = (pv[0] + fx / package_mass * dt) * lin_decay
    new_vy = (pv[1] + fy / package_mass * dt) * lin_decay
    new_av = (pkg_ang_vel[e, k] + tau / package_inertia * dt) * ang_decay
    new_px = qx + new_vx * dt
    new_py = qy + new_vy * dt
    new_px = wp.clamp(new_px, -bound, bound)
    new_py = wp.clamp(new_py, -bound, bound)
    pkg_vel[e, k] = type(pv)(new_vx, new_vy)
    pkg_ang_vel[e, k] = new_av
    pkg_pos[e, k] = type(q)(new_px, new_py)
    pkg_theta[e, k] = pkg_theta[e, k] + new_av * dt


@wp.kernel
def transport_obs_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    pkg_pos: wp.array2d(dtype=Any),
    goal: wp.array2d(dtype=Any),
    n_packages: wp.int32,
    obs: wp.array3d(dtype=Any),
):
    """Thread per (env, agent): obs row (agent pose, pkg-relative, pkg->goal)."""
    e, a = wp.tid()
    p = pos[e, a]
    v = vel[e, a]
    px = p[0]
    py = p[1]
    obs[e, a, 0] = px
    obs[e, a, 1] = py
    obs[e, a, 2] = v[0]
    obs[e, a, 3] = v[1]
    g_base = wp.int32(4) + wp.int32(2) * n_packages
    for k in range(n_packages):
        q = pkg_pos[e, k]
        g = goal[e, k]
        obs[e, a, wp.int32(4) + wp.int32(2) * k] = q[0] - px
        obs[e, a, wp.int32(4) + wp.int32(2) * k + 1] = q[1] - py
        obs[e, a, g_base + wp.int32(2) * k] = g[0] - q[0]
        obs[e, a, g_base + wp.int32(2) * k + 1] = g[1] - q[1]


@wp.kernel
def transport_reward_kernel(
    pkg_pos: wp.array2d(dtype=Any),
    goal: wp.array2d(dtype=Any),
    reset_mask: wp.array(dtype=wp.uint8),
    n_agents: wp.int32,
    n_packages: wp.int32,
    pos_shaping_factor: Any,
    goal_tolerance: Any,
    goal_reward: Any,
    advance_prev: wp.int32,
    full_pass: wp.int32,
    prev_dist: wp.array2d(dtype=Any),
    reward: wp.array2d(dtype=Any),
    done: wp.array(dtype=wp.uint8),
    dist_out: wp.array2d(dtype=Any),
):
    """Thread per env; per-package shaping reduction (deterministic, no atomics)."""
    e = wp.tid()
    zero = type(pos_shaping_factor)(0.0)
    shaping_sum = zero
    all_og = wp.uint8(1)
    reset_hit = wp.int32(reset_mask[e])
    for k in range(n_packages):
        gx = goal[e, k][0] - pkg_pos[e, k][0]
        gy = goal[e, k][1] - pkg_pos[e, k][1]
        d = wp.sqrt(gx * gx + gy * gy)
        prev = prev_dist[e, k]
        ps = (prev - d) * pos_shaping_factor
        if reset_hit == 1:
            prev_dist[e, k] = d
        else:
            if advance_prev == 1:
                prev_dist[e, k] = d
            shaping_sum += ps
        if full_pass == 1:
            dist_out[e, k] = d
            if d < goal_tolerance:
                pass
            else:
                all_og = wp.uint8(0)
    if full_pass == 1:
        bonus = zero
        if all_og == wp.uint8(1):
            bonus = goal_reward
        global_r = shaping_sum + bonus
        for a in range(n_agents):
            reward[e, a] = global_r
        done[e] = all_og


def _body_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    a2s = wp.array2d(dtype=dtype)
    return [
        a2v,  # pos
        a2v,  # vel
        wp.int32,  # n_agents
        dtype,  # agent_radius
        dtype,  # package_radius
        dtype,  # contact_margin
        dtype,  # contact_k
        dtype,  # contact_c
        dtype,  # package_mass
        dtype,  # package_inertia
        dtype,  # linear_damping
        dtype,  # angular_damping
        dtype,  # dt
        dtype,  # bound
        a2v,  # pkg_pos
        a2v,  # pkg_vel
        a2s,  # pkg_theta
        a2s,  # pkg_ang_vel
    ]


def _obs_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    return [
        wp.array2d(dtype=vec2),  # pos
        wp.array2d(dtype=vec2),  # vel
        wp.array2d(dtype=vec2),  # pkg_pos
        wp.array2d(dtype=vec2),  # goal
        wp.int32,  # n_packages
        wp.array3d(dtype=dtype),  # obs
    ]


def _reward_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2s = wp.array2d(dtype=dtype)
    return [
        wp.array2d(dtype=vec2),  # pkg_pos
        wp.array2d(dtype=vec2),  # goal
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # n_agents
        wp.int32,  # n_packages
        dtype,  # pos_shaping_factor
        dtype,  # goal_tolerance
        dtype,  # goal_reward
        wp.int32,  # advance_prev
        wp.int32,  # full_pass
        a2s,  # prev_dist
        a2s,  # reward
        wp.array(dtype=wp.uint8),  # done
        a2s,  # dist_out
    ]


for _T in (wp.float32, wp.float64):
    wp.overload(transport_body_kernel, _body_signature(_T))
    wp.overload(transport_obs_kernel, _obs_signature(_T))
    wp.overload(transport_reward_kernel, _reward_signature(_T))
