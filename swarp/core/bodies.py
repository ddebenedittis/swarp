"""Movable-obstacle dynamics: rigid bodies pushed around by agent contacts.

An obstacle tagged :attr:`~swarp.core.config.ObstacleKind.MOVABLE` carries a mass and a
moment of inertia and is integrated from the *reaction* of the very same agent-obstacle
contacts that :mod:`swarp.core.collisions` applies to the agents — Newton's third law of
one shared force, so momentum is not invented at the interface. Immovable obstacles are
infinite-mass scenery and are skipped.

Two structural points:

* **Gather-based, thread per (env, obstacle).** Each body thread loops over all agents
  and sums its own force and torque, exactly like the agent side loops over its own
  neighbors. No atomics, so the reduction order is fixed and the result deterministic.
* **Launched inside the substep loop** (:meth:`swarp.core.stepper.Stepper.launch_substeps`),
  right after the force pass and before the agents integrate, so both sides of every
  contact see the same state and the pose an agent collides against is at most one
  *substep* old. A body integrated once per env step instead sweeps its surface over
  agents that cannot react until the next step — the artifact that produces visible
  interpenetration and lurching.

Obstacles also collide with *each other*, so a pushed body stops at scenery rather than
sliding through it — for every pair in which at least one body is round (circle or
capsule). **Box-box obstacle contacts are not modelled**: that needs a polygon contact
manifold (SAT), not an SDF evaluated against a disc, so two boxes pass through one
another. Agent-vs-box is exact for every shape; this gap is only obstacle-vs-obstacle.

Not taped: like the neighbor lists, body state is advanced with ``record_tape=False``.
The discrete question of which contacts exist is not differentiated, and a scenario that
needs gradients through a movable body keeps its own torch-side reference (see
:class:`swarp.scenarios.pusht.PushTScenario`).
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp.core.collisions import (
    SHAPE_BOX,
    SHAPE_SEGMENT,
    box_force,
    closest_on_segment,
    pair_force,
)
from swarp.core.config import ObstacleKind
from swarp.core.state import VEC2
from swarp.dynamics.base import P_MASS, P_RADIUS

# Derived from the enum so the kernel tag and the public tag cannot drift.
KIND_MOVABLE = wp.constant(int(ObstacleKind.MOVABLE))


@wp.func
def _reaction(
    p: Any,
    v: Any,
    ra: Any,
    am: Any,
    center: Any,
    angle: Any,
    st: wp.int32,
    obs_r: Any,
    half: Any,
    v_obs: Any,
    om_obs: Any,
    k: Any,
    c: Any,
    margin: Any,
    sub_dt: Any,
    max_overlap: Any,
):
    """Force *on the body* from one agent, plus the lever arm it acts through.

    Reuses the agent-side force functions verbatim and negates, so action and reaction
    are the same number. Returns ``(force_on_body, lever_from_body_centre)``.
    """
    zero = type(k)(0.0)
    damp_denom = type(k)(1.0) + c * sub_dt / am
    f_agent = type(p)(zero, zero)
    r = type(p)(zero, zero)
    if st == SHAPE_BOX:
        f_agent = box_force(
            p, v, center, angle, half, ra + margin, k, c, damp_denom, max_overlap, v_obs, om_obs
        )
        # Lever: from the body centre to the agent's closest point on the box. Using the
        # agent centre projected onto the surface normal would double-count the radius.
        r = p - center
    elif st == SHAPE_SEGMENT:
        cp = closest_on_segment(p, center, angle, half[0])
        rr = cp - center
        vs = v_obs + type(p)(-om_obs * rr[1], om_obs * rr[0])
        f_agent = pair_force(
            p - cp, v - vs, ra + obs_r + margin, k, c, damp_denom, max_overlap
        )
        r = rr
    else:  # SHAPE_CIRCLE: a frictionless normal passes through the centre -> no torque
        f_agent = pair_force(
            p - center, v - v_obs, ra + obs_r + margin, k, c, damp_denom, max_overlap
        )
        r = type(p)(zero, zero)
    return -f_agent, r


@wp.func
def _closest_in_box(d: Any, angle: Any, half: Any):
    """``d`` (a vector from the box centre) clamped onto the box, in world axes.

    The same clamp the box SDF uses, so the lever arm a body-body contact acts through is
    the box's own closest point rather than the line between the two centres."""
    ca = wp.cos(angle)
    sa = wp.sin(angle)
    lx = ca * d[0] + sa * d[1]
    ly = -sa * d[0] + ca * d[1]
    cx = wp.clamp(lx, -half[0], half[0])
    cy = wp.clamp(ly, -half[1], half[1])
    return type(d)(ca * cx - sa * cy, sa * cx + ca * cy)


@wp.func
def _body_body(
    q: Any,
    tv: Any,
    th: Any,
    om_self: Any,
    st: wp.int32,
    r_self: Any,
    half_self: Any,
    q2: Any,
    tv2: Any,
    th2: Any,
    om_other: Any,
    st2: wp.int32,
    r2: Any,
    half2: Any,
    k: Any,
    c: Any,
    margin: Any,
    damp_denom: Any,
    max_overlap: Any,
):
    """Contact force on body ``self`` from another obstacle, plus its lever arm.

    Covers every pair in which at least one body is a CIRCLE or SEGMENT, by reusing the
    agent-side functions with the round body playing the part of the agent. **Box-box
    obstacle contacts are not modelled** — that needs a polygon manifold (SAT), not an
    SDF against a disc — so two boxes pass through each other. Agents still collide
    correctly with every shape; this is only about obstacles hitting each other.

    ``om_self``/``om_other`` are the two bodies' angular velocities. They only enter the
    damper, through the velocity of the material point in contact (``v + om x r``) — the
    same closing-velocity correction :func:`swarp.core.collisions._static_forces` applies
    on the agent side. Getting them wrong damps a spinning body against an absolute
    velocity that has nothing to do with the contact.
    """
    zero = type(k)(0.0)
    f = type(q)(zero, zero)
    r = type(q)(zero, zero)
    if st == SHAPE_BOX:
        if st2 != SHAPE_BOX:
            # Round body vs this box: force on the round one, negated onto the box. For a
            # capsule that round body sits at the closest point of its spine, not at its
            # centre — otherwise a long wall would act like a small disc in its middle.
            c2 = q2
            v2 = tv2
            if st2 == SHAPE_SEGMENT:
                c2 = closest_on_segment(q, q2, th2, half2[0])
                rr = c2 - q2
                v2 = tv2 + type(q)(-om_other * rr[1], om_other * rr[0])
            f_other = box_force(
                c2, v2, q, th, half_self, r2 + margin, k, c, damp_denom, max_overlap, tv, om_self
            )
            f = -f_other
            r = _closest_in_box(c2 - q, th, half_self)
    elif st2 == SHAPE_BOX:
        # This (round) body vs a box.
        f = box_force(
            q, tv, q2, th2, half2, r_self + margin, k, c, damp_denom, max_overlap, tv2, om_other
        )
    elif st2 == SHAPE_SEGMENT:
        cp = closest_on_segment(q, q2, th2, half2[0])
        # Surface point velocity of a spinning capsule: v + om x (cp - centre), exactly as
        # :func:`swarp.core.collisions._static_forces` does it for the agent side.
        rr = cp - q2
        vs2 = tv2 + type(q)(-om_other * rr[1], om_other * rr[0])
        f = pair_force(q - cp, tv - vs2, r_self + r2 + margin, k, c, damp_denom, max_overlap)
    else:
        f = pair_force(q - q2, tv - tv2, r_self + r2 + margin, k, c, damp_denom, max_overlap)
    # A frictionless normal on a round body passes through its centre: no torque, so the
    # lever stays zero except in the box branch above.
    return f, r


@wp.kernel
def obstacle_dynamics_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    params: wp.array2d(dtype=Any),  # [n_agents, NUM_PARAMS]
    obs_kind: wp.array(dtype=wp.int32),
    obs_type: wp.array(dtype=wp.int32),
    obs_radius: wp.array(dtype=Any),
    obs_half: wp.array(dtype=Any),
    obs_mass: wp.array(dtype=Any),
    obs_inertia: wp.array(dtype=Any),
    obs_body: wp.array(dtype=wp.int32),
    obs_body_off: wp.array(dtype=Any),
    n_obstacles: wp.int32,
    n_agents: wp.int32,
    k: Any,
    c: Any,
    margin: Any,
    sub_dt: Any,
    max_overlap: Any,
    lin_damping: Any,
    ang_damping: Any,
    bounds_min: Any,
    bounds_max: Any,
    clamp_bounds: wp.int32,
    body_pos: wp.array2d(dtype=Any),
    body_angle: wp.array2d(dtype=Any),
    body_vel: wp.array2d(dtype=Any),
    body_ang_vel: wp.array2d(dtype=Any),
    obs_pos: wp.array2d(dtype=Any),
    obs_angle: wp.array2d(dtype=Any),
    obs_vel: wp.array2d(dtype=Any),
    obs_ang_vel: wp.array2d(dtype=Any),
):
    """Thread per (env, body root): sum reactions over the body's shapes, integrate once.

    A *compound* body is several obstacle shapes rigidly sharing one pose: ``obs_body[s]``
    names the body a shape belongs to (its lowest-index shape, the "root") and
    ``obs_body_off[s]`` places the shape in the body frame. Only the root thread does
    work; it accumulates force and torque about the body origin over every shape it owns,
    integrates the single body state, then writes each shape's world pose and velocity
    back for the agent-side force pass. A one-shape body is the ordinary case with a zero
    offset, so nothing special-cases it.
    """
    e, o = wp.tid()
    if obs_body[o] != o:
        return  # not a body root: its pose is written by the root below
    if obs_kind[o] != KIND_MOVABLE:
        return  # immovable scenery: infinite mass, never integrated

    q = body_pos[e, o]
    tv = body_vel[e, o]
    th = body_angle[e, o]
    om = body_ang_vel[e, o]
    zero = type(th)(0.0)
    one = type(th)(1.0)
    ca = wp.cos(th)
    sa = wp.sin(th)

    fx = zero
    fy = zero
    tau = zero
    denom_self = one + c * sub_dt / obs_mass[o]
    for s in range(n_obstacles):
        if obs_body[s] != o:
            continue
        off = obs_body_off[s]
        # this shape's world pose and the velocity of its own centre (v + om x r)
        owx = ca * off[0] - sa * off[1]
        owy = sa * off[0] + ca * off[1]
        cs = type(q)(q[0] + owx, q[1] + owy)
        vs = type(tv)(tv[0] - om * owy, tv[1] + om * owx)
        st = obs_type[s]
        obs_r = obs_radius[s]
        half = obs_half[s]

        for a in range(n_agents):
            f, r = _reaction(
                pos[e, a], vel[e, a], params[a, P_RADIUS], params[a, P_MASS],
                cs, th, st, obs_r, half, vs, om, k, c, margin, sub_dt, max_overlap,
            )
            # Lever about the *body* origin: out to the shape, then within the shape.
            lx = owx + r[0]
            ly = owy + r[1]
            fx += f[0]
            fy += f[1]
            tau += lx * f[1] - ly * f[0]

        # Contacts with shapes of *other* bodies, so a pushed body stops at scenery
        # instead of sliding through it. Shapes of the same body never self-collide.
        for o2 in range(n_obstacles):
            if obs_body[o2] != o:
                f2, r2v = _body_body(
                    cs, vs, th, om, st, obs_r, half,
                    obs_pos[e, o2], obs_vel[e, o2], obs_angle[e, o2], obs_ang_vel[e, o2],
                    obs_type[o2], obs_radius[o2], obs_half[o2],
                    k, c, margin, denom_self, max_overlap,
                )
                lx = owx + r2v[0]
                ly = owy + r2v[1]
                fx += f2[0]
                fy += f2[1]
                tau += lx * f2[1] - ly * f2[0]

    lin_decay = one - lin_damping * sub_dt
    ang_decay = one - ang_damping * sub_dt
    new_vx = (tv[0] + fx / obs_mass[o] * sub_dt) * lin_decay
    new_vy = (tv[1] + fy / obs_mass[o] * sub_dt) * lin_decay
    new_om = (om + tau / obs_inertia[o] * sub_dt) * ang_decay
    new_x = q[0] + new_vx * sub_dt
    new_y = q[1] + new_vy * sub_dt
    if clamp_bounds == 1:
        new_x = wp.clamp(new_x, bounds_min[0], bounds_max[0])
        new_y = wp.clamp(new_y, bounds_min[1], bounds_max[1])
    new_th = th + new_om * sub_dt
    body_vel[e, o] = type(tv)(new_vx, new_vy)
    body_ang_vel[e, o] = new_om
    body_pos[e, o] = type(q)(new_x, new_y)
    body_angle[e, o] = new_th

    # Re-install every shape of this body at the new pose, for the next substep's agent
    # force pass (and for rendering, which reads these arrays).
    nca = wp.cos(new_th)
    nsa = wp.sin(new_th)
    for s in range(n_obstacles):
        if obs_body[s] == o:
            off = obs_body_off[s]
            owx = nca * off[0] - nsa * off[1]
            owy = nsa * off[0] + nca * off[1]
            obs_pos[e, s] = type(q)(new_x + owx, new_y + owy)
            obs_angle[e, s] = new_th
            obs_vel[e, s] = type(tv)(new_vx - new_om * owy, new_vy + new_om * owx)
            obs_ang_vel[e, s] = new_om


@wp.kernel
def body_state_gather_kernel(
    body_pos: wp.array2d(dtype=Any),
    body_angle: wp.array2d(dtype=Any),
    body_vel: wp.array2d(dtype=Any),
    body_ang_vel: wp.array2d(dtype=Any),
    root: wp.int32,
    out_pos: wp.array(dtype=Any),
    out_angle: wp.array(dtype=Any),
    out_vel: wp.array(dtype=Any),
    out_ang_vel: wp.array(dtype=Any),
):
    """Thread per env: lift one body's ``[n_envs, n_obstacles]`` column into flat arrays.

    A scenario that keeps its own view of a movable body (Push-T's ``tee_*``, which its
    fused obs/reward kernels read) uses this to pick up the engine's state. A strided
    column of a 2-D array is not contiguous, so this cannot be a ``wp.copy``. Allocation
    free, hence safe inside a captured whole-step graph.
    """
    e = wp.tid()
    out_pos[e] = body_pos[e, root]
    out_angle[e] = body_angle[e, root]
    out_vel[e] = body_vel[e, root]
    out_ang_vel[e] = body_ang_vel[e, root]


def _gather_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    return [
        wp.array2d(dtype=vec2),  # body_pos
        wp.array2d(dtype=dtype),  # body_angle
        wp.array2d(dtype=vec2),  # body_vel
        wp.array2d(dtype=dtype),  # body_ang_vel
        wp.int32,  # root
        wp.array(dtype=vec2),  # out_pos
        wp.array(dtype=dtype),  # out_angle
        wp.array(dtype=vec2),  # out_vel
        wp.array(dtype=dtype),  # out_ang_vel
    ]


def _signature(dtype) -> list:
    vec2 = VEC2[dtype]
    return [
        wp.array2d(dtype=vec2),  # pos
        wp.array2d(dtype=vec2),  # vel
        wp.array2d(dtype=dtype),  # params
        wp.array(dtype=wp.int32),  # obs_kind
        wp.array(dtype=wp.int32),  # obs_type
        wp.array(dtype=dtype),  # obs_radius
        wp.array(dtype=vec2),  # obs_half
        wp.array(dtype=dtype),  # obs_mass
        wp.array(dtype=dtype),  # obs_inertia
        wp.array(dtype=wp.int32),  # obs_body
        wp.array(dtype=vec2),  # obs_body_off
        wp.int32,  # n_obstacles
        wp.int32,  # n_agents
        dtype,  # k
        dtype,  # c
        dtype,  # margin
        dtype,  # sub_dt
        dtype,  # max_overlap
        dtype,  # lin_damping
        dtype,  # ang_damping
        vec2,  # bounds_min
        vec2,  # bounds_max
        wp.int32,  # clamp_bounds
        wp.array2d(dtype=vec2),  # body_pos
        wp.array2d(dtype=dtype),  # body_angle
        wp.array2d(dtype=vec2),  # body_vel
        wp.array2d(dtype=dtype),  # body_ang_vel
        wp.array2d(dtype=vec2),  # obs_pos
        wp.array2d(dtype=dtype),  # obs_angle
        wp.array2d(dtype=vec2),  # obs_vel
        wp.array2d(dtype=dtype),  # obs_ang_vel
    ]


for _T in (wp.float32, wp.float64):
    wp.overload(obstacle_dynamics_kernel, _signature(_T))
    wp.overload(body_state_gather_kernel, _gather_signature(_T))


def launch_obstacle_dynamics(
    pos,
    vel,
    params,
    stepper,
    *,
    n_agents: int,
    k: float,
    c: float,
    margin: float,
    sub_dt: float,
    max_overlap: float,
    lin_damping: float,
    ang_damping: float,
    bounds_min,
    bounds_max,
    clamp_bounds: bool,
    dtype,
) -> None:
    """Advance every movable obstacle by one substep (no-op when there are none).

    Everything past ``stepper`` is keyword-only: the tail is seven interchangeable floats
    and a bool, where a swapped argument would be invisible at the call site.
    """
    n_envs = pos.shape[0]
    wp.launch(
        obstacle_dynamics_kernel,
        dim=(n_envs, stepper.n_obstacles),
        inputs=[
            pos,
            vel,
            params,
            stepper.obs_kind,
            stepper.obs_type,
            stepper.obs_radius,
            stepper.obs_half,
            stepper.obs_mass,
            stepper.obs_inertia,
            stepper.obs_body,
            stepper.obs_body_off,
            wp.int32(stepper.n_obstacles),
            wp.int32(n_agents),
            dtype(k),
            dtype(c),
            dtype(margin),
            dtype(sub_dt),
            dtype(max_overlap),
            dtype(lin_damping),
            dtype(ang_damping),
            bounds_min,
            bounds_max,
            wp.int32(1 if clamp_bounds else 0),
        ],
        outputs=[
            stepper.body_pos,
            stepper.body_angle,
            stepper.body_vel,
            stepper.body_ang_vel,
            stepper.obs_pos,
            stepper.obs_angle,
            stepper.obs_vel,
            stepper.obs_ang_vel,
        ],
        device=pos.device,
        record_tape=False,
    )
