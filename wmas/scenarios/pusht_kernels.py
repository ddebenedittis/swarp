"""Fused Warp kernels for the PushTScenario obs/reward + movable-T-body layer.

Three launches, ordered ``body -> {obs, reward}``:

* ``pusht_body_kernel`` (thread per env) sums the spring-damper reaction force +
  torque the T receives from every agent across both of its boxes (Newton's third
  law of the same soft agent-obstacle contact the agent step applies), integrates
  the body pose in place (semi-implicit Euler + damping), **and writes the updated
  box poses straight into the stepper's obstacle buffers**. Each env is owned by
  one thread — no cross-thread races; the reduction loops ``0..A-1`` x ``0..B-1``.

  Folding the obstacle write into the kernel is the one structural difference from
  :mod:`wmas.scenarios.transport_kernels`. Transport re-installs its obstacles from
  the host inside the whole-step graph, which is capture-safe only because it is a
  bare ``wp.copy``; the T's install needs a rotation (``cos``/``sin`` + offsets),
  and doing that in torch under CUDA-graph capture would allocate. Writing
  ``obs_pos``/``obs_angle`` here keeps ``_graph_post_physics`` allocation-free.
* ``pusht_obs_kernel`` (thread per env,agent) builds the obs row.
* ``pusht_reward_kernel`` (thread per env) reduces position **and** orientation
  shaping plus the pose bonus, using the navigation ``prev_dist`` / ``reset_hit`` /
  ``advance_prev`` / ``full_pass`` machinery, extended to a second (angular) carry.

The torch implementation in :mod:`wmas.scenarios.pusht` stays the reference (and the
differentiable path — body gradients flow through the plain-torch physics). The fused
body kernel is the no-grad fast path; because it sums the agent reactions in a
different order than torch's ``.sum(dim=1)``, the fused and torch body trajectories
can differ at the ulp scale, kept bounded by the linear/angular damping (validated
allclose, not bit-exact).

Contacts are frictionless: the force is normal-only (spring + normal damping), exactly
like :func:`wmas.core.collisions._box_force`, whose SDF branches are reproduced here so
the reaction matches the action.

Observation cat order (``obs_dim = 12 + 2*(n_agents-1)``)::

    [ pos(2), vel(2), tee_rel(2), tee_to_goal(2), cos/sin(theta)(2), cos/sin(ang_err)(2),
      teammate_rel(2 each, ascending agent index, self skipped) ]
"""

from __future__ import annotations

from typing import Any

import warp as wp

from wmas.core.state import VEC2

_EPS2 = 1.0e-10  # distance^2 floor, matching wmas.core.collisions


@wp.kernel
def pusht_body_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    box_off: wp.array(dtype=Any),
    box_half: wp.array(dtype=Any),
    n_agents: wp.int32,
    n_boxes: wp.int32,
    agent_radius: Any,
    contact_margin: Any,
    contact_k: Any,
    contact_c: Any,
    tee_mass: Any,
    tee_inertia: Any,
    linear_damping: Any,
    angular_damping: Any,
    dt: Any,
    body_substeps: wp.int32,
    max_overlap: Any,
    bound: Any,
    tee_pos: wp.array(dtype=Any),
    tee_vel: wp.array(dtype=Any),
    tee_theta: wp.array(dtype=Any),
    tee_ang_vel: wp.array(dtype=Any),
    obs_pos: wp.array2d(dtype=Any),
    obs_angle: wp.array2d(dtype=Any),
    obs_vel: wp.array2d(dtype=Any),
    obs_ang_vel: wp.array2d(dtype=Any),
):
    """Thread per env: accumulate box-SDF contact force/torque, integrate, re-install.

    The integration is substepped ``body_substeps`` times at ``dt / body_substeps``,
    recomputing the contact force from the (frozen) agent state each time. An explicit
    spring-damper is stable only while ``sub_dt < 2 sqrt(m / k)``, so substepping here
    is what lets ``contact_k`` be stiff enough to behave like a real contact instead of
    a soft sponge. It is the body analogue of ``Stepper.substeps``, which does the same
    for the agent half of the step but does not reach this kernel.
    """
    e = wp.tid()
    q = tee_pos[e]
    tv = tee_vel[e]
    th = tee_theta[e]
    om = tee_ang_vel[e]
    qx = q[0]
    qy = q[1]
    tvx = tv[0]
    tvy = tv[1]
    zero = type(qx)(0.0)
    one = type(qx)(1.0)
    eps2 = type(qx)(_EPS2)
    reach = agent_radius + contact_margin
    sub_dt = dt / type(qx)(body_substeps)
    lin_decay = one - linear_damping * sub_dt
    ang_decay = one - angular_damping * sub_dt

    for _sub in range(body_substeps):
        ca = wp.cos(th)
        sa = wp.sin(th)
        fx = zero
        fy = zero
        tau = zero
        for b in range(n_boxes):
            off = box_off[b]
            half = box_half[b]
            # box centre in world = tee centre + R(theta) @ local offset
            owx = ca * off[0] - sa * off[1]
            owy = sa * off[0] + ca * off[1]
            cwx = qx + owx
            cwy = qy + owy
            hx = half[0]
            hy = half[1]
            for a in range(n_agents):
                p = pos[e, a]
                dx = p[0] - cwx
                dy = p[1] - cwy
                # into the box frame (R^T d)
                lx = ca * dx + sa * dy
                ly = -sa * dx + ca * dy
                cx = wp.clamp(lx, -hx, hx)
                cy = wp.clamp(ly, -hy, hy)
                ox = lx - cx
                oy = ly - cy
                out_d2 = ox * ox + oy * oy
                if out_d2 > eps2:
                    # exterior: signed distance is the positive outside distance
                    s = wp.sqrt(out_d2)
                    nlx = ox / s
                    nly = oy / s
                else:
                    # interior: nearest face gives the (negative) signed distance
                    gx = wp.abs(lx) - hx
                    gy = wp.abs(ly) - hy
                    if gx > gy:
                        s = gx
                        if lx >= zero:
                            nlx = one
                        else:
                            nlx = -one
                        nly = zero
                    else:
                        s = gy
                        nlx = zero
                        if ly >= zero:
                            nly = one
                        else:
                            nly = -one
                overlap = reach - s
                if overlap <= zero:
                    continue
                # Obstacle poses are frozen within an env step, so a moving T can sweep
                # its surface over an agent that was outside it; the agent cannot react
                # until the next step. Saturate the depth that feeds the spring so that
                # artifact cannot become a k*depth impulse (which fed back into a faster
                # sweep). tanh, not min(), to keep the force differentiable in the depth
                # everywhere. Legitimate pushing compresses ~0.05 agent radii, far below
                # max_overlap, where tanh(x) ~ x leaves the intended regime untouched.
                overlap = max_overlap * wp.tanh(overlap / max_overlap)
                # closest surface point, box frame -> lever arm about the tee centre
                spx = lx - s * nlx
                spy = ly - s * nly
                rx = owx + (ca * spx - sa * spy)
                ry = owy + (sa * spx + ca * spy)
                # m = -n: unit normal pointing from the agent into the T
                mx = -(ca * nlx - sa * nly)
                my = -(sa * nlx + ca * nly)
                # relative velocity of the T's contact point w.r.t. the agent
                av = vel[e, a]
                rvx = tvx - om * ry - av[0]
                rvy = tvy + om * rx - av[1]
                vn = rvx * mx + rvy * my
                # Linearly-implicit (semi-implicit) damping: with the normal frozen over
                # the sub-step, solving f = k*ov - c*vn(f) for the post-impulse normal
                # velocity is one divide, and is unconditionally stable in contact_c
                # instead of needing sub_dt < m/c. tee_mass is the Jacobi (per-contact
                # diagonal) approximation of the effective mass. Smooth in every input,
                # and better conditioned for adjoints than the explicit form, which can
                # flip sign.
                coeff = (contact_k * overlap - contact_c * vn) / (
                    one + contact_c * sub_dt / tee_mass
                )
                # Keep the contact repulsive: an explicit damper can turn a contact
                # attractive and yank the T toward a separating agent.
                coeff = wp.max(coeff, zero)
                forcex = coeff * mx
                forcey = coeff * my
                fx += forcex
                fy += forcey
                tau += rx * forcey - ry * forcex

        tvx = (tvx + fx / tee_mass * sub_dt) * lin_decay
        tvy = (tvy + fy / tee_mass * sub_dt) * lin_decay
        om = (om + tau / tee_inertia * sub_dt) * ang_decay
        qx = wp.clamp(qx + tvx * sub_dt, -bound, bound)
        qy = wp.clamp(qy + tvy * sub_dt, -bound, bound)
        th = th + om * sub_dt

    new_th = th
    tee_vel[e] = type(tv)(tvx, tvy)
    tee_ang_vel[e] = om
    tee_pos[e] = type(q)(qx, qy)
    tee_theta[e] = new_th

    # Re-install the two boxes at the new pose for the next physics step.
    nca = wp.cos(new_th)
    nsa = wp.sin(new_th)
    for b in range(n_boxes):
        off = box_off[b]
        owx = nca * off[0] - nsa * off[1]
        owy = nsa * off[0] + nca * off[1]
        obs_pos[e, b] = type(q)(qx + owx, qy + owy)
        obs_angle[e, b] = new_th
        # Each box's velocity, v + om x r, so the agent-side contact damper can use the
        # closing velocity next step (see wmas.core.collisions).
        obs_vel[e, b] = type(q)(tvx - om * owy, tvy + om * owx)
        obs_ang_vel[e, b] = om


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


def _body_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a1v = wp.array(dtype=vec2)
    a1s = wp.array(dtype=dtype)
    return [
        wp.array2d(dtype=vec2),  # pos
        wp.array2d(dtype=vec2),  # vel
        a1v,  # box_off
        a1v,  # box_half
        wp.int32,  # n_agents
        wp.int32,  # n_boxes
        dtype,  # agent_radius
        dtype,  # contact_margin
        dtype,  # contact_k
        dtype,  # contact_c
        dtype,  # tee_mass
        dtype,  # tee_inertia
        dtype,  # linear_damping
        dtype,  # angular_damping
        dtype,  # dt
        wp.int32,  # body_substeps
        dtype,  # max_overlap
        dtype,  # bound
        a1v,  # tee_pos
        a1v,  # tee_vel
        a1s,  # tee_theta
        a1s,  # tee_ang_vel
        wp.array2d(dtype=vec2),  # obs_pos [n_envs, n_boxes]
        wp.array2d(dtype=dtype),  # obs_angle [n_envs, n_boxes]
        wp.array2d(dtype=vec2),  # obs_vel [n_envs, n_boxes]
        wp.array2d(dtype=dtype),  # obs_ang_vel [n_envs, n_boxes]
    ]


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


for _T in (wp.float32, wp.float64):
    wp.overload(pusht_body_kernel, _body_signature(_T))
    wp.overload(pusht_obs_kernel, _obs_signature(_T))
    wp.overload(pusht_reward_kernel, _reward_signature(_T))
