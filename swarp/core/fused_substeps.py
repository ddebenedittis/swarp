"""The whole substep chain as one kernel: forces, integration and every substep.

The layered pipeline in :meth:`~swarp.core.stepper.Stepper.launch_substeps` issues a
neighbor build, a force pass and an integrate pass *per substep*, each its own launch.
Inside a captured graph a launch is a node, and at the sizes RL trains at (a few agents,
a few thousand envs) a node costs more than the work in it: at 1024 envs x 4 agents x 8
substeps the step was ~27 nodes of ~3 us each. This module folds all of it into a single
launch.

**Mapping.** One CUDA block owns ``envs_per_block`` whole envs, one thread per agent, so
every thread that reads an env's state lives in the same block as every thread that
writes it. A substep is then a force phase (each thread gathers the contact force on its
own agent from every other agent in its env), a block barrier, an integrate phase (each
thread advances its own agent in place), and another barrier. ``__syncthreads`` makes
the global writes before it visible to the whole block, so the barrier is the entire
synchronization story: no shared memory, no atomics, and the result does not depend on
scheduling.

**Neighbors.** Substep 0 reads the grid's list when it was built on the input state
(the previous step's post-step list, the layered pipeline's reuse rule). Every later
substep scans all ``n_agents`` agents of the env with the brute-force builder's
membership test (``b != a``, ``length <= radius``, capped at ``max_neighbors`` in index
order) into a per-thread list, then sums the contacts over it. Either way the force sums
the same terms in the same order as the layered pass, so the step is bit-identical to
it. Collecting before summing matters: a contact computed inside the scan stalls the
whole warp once per contact any of its lanes has, which measured ~30% slower at 64
agents per env.

**Post-step list.** Observations want a neighbor list on the *final* state. With
``emit_list`` the kernel builds it after the last substep, into the grid's own buffers,
with the brute builder's exact code; the caller stamps the grid so the scenario's
post-step ``World.build_neighbors()`` becomes a no-op.

Eligibility (decided by the stepper): CUDA only (the barrier has no CPU meaning, where a
launch runs its threads one after another), untaped, the slim 2D Euler path, no movable
obstacles (their reaction pass is a separate gather over agents), shared agent params,
and the brute-force neighbor method.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp.core.collisions import (
    BOX_MODULE_OPTIONS,
    _agent_pair_force,
    _obstacle_forces,
    _obstacle_forces_box,
    _wall_forces,
    _wall_forces_box,
    pair_force,
)
from swarp.core.state import VEC2
from swarp.dynamics.base import (
    P_CMD_GAIN_V,
    P_CMD_GAIN_W,
    P_HALF_X,
    P_HALF_Y,
    P_LF,
    P_LR,
    P_MASS,
    P_MAX_ACCEL,
    P_MAX_ANG_ACCEL,
    P_MAX_ANG_VEL,
    P_MAX_SPEED,
    P_MAX_STEER,
    P_MIN_SPEED,
    P_RADIUS,
    P_SHAPE,
)
from swarp.dynamics.kernels import (
    MODE_SERVO,
    MODE_VELOCITY,
    _step_2d,
    clamp_footprint,
    servo_command,
)

#: Thread budget a block aims for; an env never straddles two blocks.
TARGET_BLOCK = 128
#: Largest env one block can hold (CUDA's per-block thread limit).
MAX_AGENTS = 1024


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
"""
)
def block_sync(): ...


def block_layout(n_agents: int) -> tuple[int, int]:
    """``(block_dim, envs_per_block)`` for ``n_agents`` agents per env.

    As many whole envs as fit in :data:`TARGET_BLOCK` threads, rounded up to a whole
    warp; an env larger than that gets a block of its own.
    """
    envs_per_block = max(1, TARGET_BLOCK // n_agents)
    block_dim = -(-(envs_per_block * n_agents) // 32) * 32
    return block_dim, envs_per_block


def _make_kernel(
    collisions: bool,
    obstacles: bool,
    walls: bool,
    clamp: bool,
    emit: bool,
    reuse: bool,
    max_neighbors: int,
    boxes: bool = False,
):
    """The chain specialized at compile time to what a world uses.

    Every disabled feature is compiled out rather than skipped at run time: code that
    never runs still claims registers, and the neighbor scans here are latency-bound, so
    occupancy is what they run on. The flags only remove terms that would add zero (an
    obstacle loop over no obstacles, walls that are off), so every variant sums the same
    nonzero terms in the same order as the layered pipeline.

    ``reuse`` reads substep 0's neighbors from the grid's list, which the caller vouches
    was built on the input state (the previous step's emitted list); every other substep
    scans. ``max_neighbors`` sizes the per-thread list a scan collects into.

    ``boxes`` compiles in the box-agent contacts of
    :func:`~swarp.core.collisions.collision_forces_box_kernel` (every pair dispatched on
    the two agents' ``P_SHAPE``, box obstacles and walls, the footprint clamp), summed in
    the layered kernel's order. The box variant compiles without FP contraction, as every
    kernel of a box world does (the layered force pass and the box-world integrate
    included): inlined into this much larger kernel, nvcc contracted the box contact
    differently than in the layered one, a 1-ulp drift. With ``boxes=False`` every box
    branch is a ``wp.static`` that compiles out and the options are the default, so the
    disc variants generate the code they always did.
    """
    nbr_vec = wp.types.vector(length=max(1, max_neighbors), dtype=wp.int32)

    @wp.kernel(
        enable_backward=False,
        module="unique",
        grid_stride=False,
        module_options=BOX_MODULE_OPTIONS if boxes else None,
    )
    def fused_substeps_kernel(
        # 2D state in
        pos: wp.array2d(dtype=Any),
        theta: wp.array2d(dtype=Any),
        vel: wp.array2d(dtype=Any),
        speed: wp.array2d(dtype=Any),
        ang_vel: wp.array2d(dtype=Any),
        actions: wp.array3d(dtype=Any),
        params: wp.array2d(dtype=Any),  # [n_agents, NUM_PARAMS] shared across envs
        model_tag: wp.array(dtype=wp.int32),
        ctrl_mode: wp.array(dtype=wp.int32),
        n_envs: wp.int32,
        block_dim: wp.int32,
        envs_per_block: wp.int32,
        substeps: wp.int32,
        sub_dt: Any,
        # agent-agent contacts
        radius: Any,
        # obstacles and soft walls
        obs_pos: wp.array2d(dtype=Any),
        obs_radius: wp.array(dtype=Any),
        obs_type: wp.array(dtype=wp.int32),
        obs_angle: wp.array2d(dtype=Any),
        obs_half: wp.array(dtype=Any),
        obs_vel: wp.array2d(dtype=Any),
        obs_ang_vel: wp.array2d(dtype=Any),
        n_obstacles: wp.int32,
        k: Any,
        c: Any,
        margin: Any,
        max_overlap: Any,
        bounds_min: Any,
        bounds_max: Any,
        # hard position clamp
        clamp_min: Any,
        clamp_max: Any,
        # post-step neighbor list
        neighbor_idx: wp.array3d(dtype=wp.int32),
        neighbor_count: wp.array2d(dtype=wp.int32),
        neighbor_true: wp.array2d(dtype=wp.int32),
        # 2D state out (may alias the input)
        pos_out: wp.array2d(dtype=Any),
        theta_out: wp.array2d(dtype=Any),
        vel_out: wp.array2d(dtype=Any),
        speed_out: wp.array2d(dtype=Any),
        ang_vel_out: wp.array2d(dtype=Any),
    ):
        tid = wp.tid()
        n_agents = pos.shape[1]
        lane = tid % block_dim
        slot = lane / n_agents
        a = lane - slot * n_agents
        e = (tid / block_dim) * envs_per_block + slot
        # Idle lanes (the block's tail, or past the last env) still reach every barrier.
        active = slot < envs_per_block and e < n_envs
        zero = type(sub_dt)(0.0)

        # Seed the output with the input, so every substep below runs in place on it.
        if active:
            pos_out[e, a] = pos[e, a]
            theta_out[e, a] = theta[e, a]
            vel_out[e, a] = vel[e, a]
            speed_out[e, a] = speed[e, a]
            ang_vel_out[e, a] = ang_vel[e, a]
        block_sync()

        for _k in range(substeps):
            p = pos_out[0, 0]
            v = vel_out[0, 0]
            f = type(p)(zero, zero)
            if active:
                p = pos_out[e, a]
                v = vel_out[e, a]
                if wp.static(collisions or obstacles or walls):
                    ra = params[a, P_RADIUS]
                    damp_denom = type(sub_dt)(1.0) + c * sub_dt / params[a, P_MASS]
                    if wp.static(boxes):
                        th = theta_out[e, a]
                        w = ang_vel_out[e, a]
                        sha = params[a, P_SHAPE]
                        ha = type(p)(params[a, P_HALF_X], params[a, P_HALF_Y])
                if wp.static(collisions):
                    # The same neighbors in the same order as the list the layered force
                    # pass reads, so the same sum. Collected first, then summed: a contact
                    # inside the scan would stall the whole warp once per contact.
                    nbr = nbr_vec()
                    count = wp.int32(0)
                    if wp.static(reuse) and _k == 0:
                        count = neighbor_count[e, a]
                        for i in range(count):
                            nbr[i] = neighbor_idx[e, a, i]
                    else:
                        for b in range(n_agents):
                            if (
                                b != a
                                and count < wp.static(max_neighbors)
                                and wp.length(pos_out[e, b] - p) <= radius
                            ):
                                nbr[count] = b
                                count += wp.int32(1)
                    for i in range(count):
                        b = nbr[i]
                        if wp.static(boxes):
                            f += _agent_pair_force(
                                a,
                                b,
                                sha,
                                params[b, P_SHAPE],
                                p,
                                th,
                                v,
                                w,
                                ra,
                                ha,
                                pos_out[e, b],
                                theta_out[e, b],
                                vel_out[e, b],
                                ang_vel_out[e, b],
                                params[b, P_RADIUS],
                                type(p)(params[b, P_HALF_X], params[b, P_HALF_Y]),
                                margin,
                                k,
                                c,
                                damp_denom,
                                max_overlap,
                            )
                        else:
                            f += pair_force(
                                p - pos_out[e, b],
                                v - vel_out[e, b],
                                ra + params[b, P_RADIUS] + margin,
                                k,
                                c,
                                damp_denom,
                                max_overlap,
                            )
                if wp.static(obstacles or walls):
                    fs = type(p)(zero, zero)
                    if wp.static(obstacles):
                        if wp.static(boxes):
                            if sha < type(sub_dt)(0.5):
                                fs = _obstacle_forces(
                                    e,
                                    p,
                                    v,
                                    ra,
                                    obs_pos,
                                    obs_radius,
                                    obs_type,
                                    obs_angle,
                                    obs_half,
                                    obs_vel,
                                    obs_ang_vel,
                                    n_obstacles,
                                    k,
                                    c,
                                    margin,
                                    damp_denom,
                                    max_overlap,
                                )
                            else:
                                fs = _obstacle_forces_box(
                                    e,
                                    p,
                                    th,
                                    v,
                                    w,
                                    ha,
                                    obs_pos,
                                    obs_radius,
                                    obs_type,
                                    obs_angle,
                                    obs_half,
                                    obs_vel,
                                    obs_ang_vel,
                                    n_obstacles,
                                    k,
                                    c,
                                    margin,
                                    damp_denom,
                                    max_overlap,
                                )
                        else:
                            fs = _obstacle_forces(
                                e,
                                p,
                                v,
                                ra,
                                obs_pos,
                                obs_radius,
                                obs_type,
                                obs_angle,
                                obs_half,
                                obs_vel,
                                obs_ang_vel,
                                n_obstacles,
                                k,
                                c,
                                margin,
                                damp_denom,
                                max_overlap,
                            )
                    if wp.static(walls):
                        if wp.static(boxes):
                            if sha < type(sub_dt)(0.5):
                                fs += _wall_forces(
                                    p,
                                    v,
                                    ra,
                                    k,
                                    c,
                                    margin,
                                    damp_denom,
                                    max_overlap,
                                    bounds_min,
                                    bounds_max,
                                )
                            else:
                                fs += _wall_forces_box(
                                    p,
                                    th,
                                    v,
                                    w,
                                    ha,
                                    k,
                                    c,
                                    margin,
                                    damp_denom,
                                    max_overlap,
                                    bounds_min,
                                    bounds_max,
                                )
                        else:
                            fs += _wall_forces(
                                p,
                                v,
                                ra,
                                k,
                                c,
                                margin,
                                damp_denom,
                                max_overlap,
                                bounds_min,
                                bounds_max,
                            )
                    f += fs
            block_sync()
            if active:
                s_cur = speed_out[e, a]
                w_cur = ang_vel_out[e, a]
                c0 = actions[e, a, 0]
                c1 = actions[e, a, 1]
                mode = ctrl_mode[a]
                if mode == MODE_SERVO:
                    c0, c1 = servo_command(
                        c0,
                        c1,
                        s_cur,
                        w_cur,
                        params[a, P_MAX_SPEED],
                        params[a, P_MAX_ANG_VEL],
                        params[a, P_MAX_ACCEL],
                        params[a, P_MAX_ANG_ACCEL],
                        params[a, P_CMD_GAIN_V],
                        params[a, P_CMD_GAIN_W],
                        sub_dt,
                    )
                    mode = MODE_VELOCITY
                p_new, th_new, v_new, s_new, w_new = _step_2d(
                    p,
                    theta_out[e, a],
                    v,
                    s_cur,
                    w_cur,
                    c0,
                    c1,
                    model_tag[a],
                    mode,
                    f,
                    params[a, P_MAX_SPEED],
                    params[a, P_MIN_SPEED],
                    params[a, P_MAX_ACCEL],
                    params[a, P_MAX_ANG_VEL],
                    params[a, P_MAX_ANG_ACCEL],
                    params[a, P_LF],
                    params[a, P_LR],
                    params[a, P_MAX_STEER],
                    params[a, P_MASS],
                    sub_dt,
                )
                if wp.static(clamp):
                    if wp.static(boxes):
                        p_new = clamp_footprint(
                            p_new,
                            th_new,
                            params[a, P_RADIUS],
                            params[a, P_SHAPE],
                            params[a, P_HALF_X],
                            params[a, P_HALF_Y],
                            clamp_min,
                            clamp_max,
                        )
                    else:
                        r = params[a, P_RADIUS]
                        p_new = type(p_new)(
                            wp.clamp(p_new[0], clamp_min[0] + r, clamp_max[0] - r),
                            wp.clamp(p_new[1], clamp_min[1] + r, clamp_max[1] - r),
                        )
                pos_out[e, a] = p_new
                theta_out[e, a] = th_new
                vel_out[e, a] = v_new
                speed_out[e, a] = s_new
                ang_vel_out[e, a] = w_new
            block_sync()

        # Nested on purpose: a static test folded into a runtime ``and`` would compile the
        # body in every variant.
        if wp.static(emit):  # noqa: SIM102
            if active:
                # Verbatim the brute-force builder (swarp.core.neighbors._brute_force).
                p = pos_out[e, a]
                count = wp.int32(0)
                total = wp.int32(0)
                for b in range(n_agents):
                    if b != a and wp.length(pos_out[e, b] - p) <= radius:
                        total += wp.int32(1)
                        if count < wp.static(max_neighbors):
                            neighbor_idx[e, a, count] = b
                            count += wp.int32(1)
                neighbor_count[e, a] = count
                neighbor_true[e, a] = total

    return fused_substeps_kernel


def _signature(dtype) -> list:
    vec2 = VEC2[dtype]
    a2v = wp.array2d(dtype=vec2)
    a2s = wp.array2d(dtype=dtype)
    a1i = wp.array(dtype=wp.int32)
    state = [a2v, a2s, a2v, a2s, a2s]  # pos, theta, vel, speed, ang_vel
    return [
        *state,
        wp.array3d(dtype=dtype),  # actions
        a2s,  # params
        a1i,  # model_tag
        a1i,  # ctrl_mode
        wp.int32,  # n_envs
        wp.int32,  # block_dim
        wp.int32,  # envs_per_block
        wp.int32,  # substeps
        dtype,  # sub_dt
        dtype,  # radius
        a2v,  # obs_pos
        wp.array(dtype=dtype),  # obs_radius
        a1i,  # obs_type
        a2s,  # obs_angle
        wp.array(dtype=vec2),  # obs_half
        a2v,  # obs_vel
        a2s,  # obs_ang_vel
        wp.int32,  # n_obstacles
        dtype,  # k
        dtype,  # c
        dtype,  # margin
        dtype,  # max_overlap
        vec2,  # bounds_min
        vec2,  # bounds_max
        vec2,  # clamp_min
        vec2,  # clamp_max
        wp.array3d(dtype=wp.int32),  # neighbor_idx
        wp.array2d(dtype=wp.int32),  # neighbor_count
        wp.array2d(dtype=wp.int32),  # neighbor_true
        *state,
    ]


#: (collisions, obstacles, walls, clamp, emit, reuse, max_neighbors, boxes, dtype) -> kernel.
_KERNELS: dict[tuple, wp.Kernel] = {}


def fused_kernel(dtype, *flags, device=None, block_dim: int | None = None) -> wp.Kernel:
    """The concrete kernel for ``flags`` (see :func:`_make_kernel`) and ``dtype``.

    ``emit`` and ``reuse`` are decided per launch from host state that a whole-step graph
    capture does not see coming: its warm-up step may run without them and the captured
    step with them. A kernel first launched inside a capture cannot be loaded there, so
    the first request for a variant builds and loads all four ``(emit, reuse)`` siblings
    on ``device`` up front.
    """
    key = (*flags, dtype)
    kernel = _KERNELS.get(key)
    if kernel is None:
        collisions, obstacles, walls, clamp, _, _, max_neighbors, boxes = flags
        for emit in (False, True):
            for reuse in (False, True):
                sib = (collisions, obstacles, walls, clamp, emit, reuse, max_neighbors, boxes)
                k = wp.overload(_make_kernel(*sib), _signature(dtype))
                if device is not None:
                    wp.load_module(k.module, device=device, block_dim=block_dim)
                _KERNELS[(*sib, dtype)] = k
        kernel = _KERNELS[key]
    return kernel


def launch_fused_substeps(
    stepper, state_in, actions, state_out, *, emit_list: bool, reuse: bool
) -> None:
    """Advance ``state_in`` one full env step into ``state_out`` (which may alias it).

    ``stepper`` supplies everything else: params, contact constants, obstacles, bounds and
    the neighbor grid. ``reuse`` reads substep 0's neighbors from the grid's lists (the
    caller guarantees they were built on ``state_in``); ``emit_list`` writes the
    post-step lists into them.
    """
    dtype = stepper.dtype
    vec2 = VEC2[dtype]
    world = stepper.world
    n_envs, n_agents = state_in.pos.shape
    block_dim, envs_per_block = block_layout(n_agents)
    n_blocks = -(-n_envs // envs_per_block)
    collisions = stepper.collisions
    if collisions:
        grid = stepper.grid(n_envs)
        idx, cnt, true = grid.neighbor_idx, grid.neighbor_count, grid.neighbor_true_count
        max_neighbors = grid.max_neighbors
    else:
        idx, cnt = stepper._zero_neighbors(n_envs)
        true, max_neighbors = cnt, 0
    clamp = stepper._clamp_bounds
    if clamp is None:
        c_min, c_max = vec2(0.0, 0.0), vec2(0.0, 0.0)
    else:
        x_min, x_max, y_min, y_max = clamp
        c_min, c_max = vec2(x_min, y_min), vec2(x_max, y_max)
    params = stepper.params
    flags = (
        collisions,
        stepper.n_obstacles > 0,
        stepper._soft_walls,
        clamp is not None,
        emit_list and collisions,
        reuse and collisions,
        max_neighbors,
        stepper.has_box_agents,
    )
    wp.launch(
        fused_kernel(dtype, *flags, device=state_in.pos.device, block_dim=block_dim),
        dim=n_blocks * block_dim,
        block_dim=block_dim,
        inputs=[
            state_in.pos,
            state_in.theta,
            state_in.vel,
            state_in.speed,
            state_in.ang_vel,
            actions,
            params.floats,
            params.model_tag,
            params.ctrl_mode,
            wp.int32(n_envs),
            wp.int32(block_dim),
            wp.int32(envs_per_block),
            wp.int32(stepper.substeps),
            dtype(stepper.sub_dt),
            dtype(stepper.neighbor_radius),
            stepper.obs_pos,
            stepper.obs_radius,
            stepper.obs_type,
            stepper.obs_angle,
            stepper.obs_half,
            stepper.obs_vel,
            stepper.obs_ang_vel,
            wp.int32(stepper.n_obstacles),
            dtype(world.collision_k),
            dtype(world.collision_c),
            dtype(world.collision_margin),
            dtype(world.contact_max_overlap),
            stepper._bounds_min,
            stepper._bounds_max,
            c_min,
            c_max,
            idx,
            cnt,
            true,
        ],
        outputs=[
            state_out.pos,
            state_out.theta,
            state_out.vel,
            state_out.speed,
            state_out.ang_vel,
        ],
        device=state_in.pos.device,
        record_tape=False,
    )
