"""Warp kernels for the lidar artefacts and the randomisation-aware critic of
:class:`~swarp.scenarios.pusht_lidar.PushTLidarScenario`.

All opt-in; a scenario that enables none of them never launches any of these. In a pass
they slot into :meth:`~swarp.scenarios.pusht_lidar.PushTLidarScenario.launch_fused` as

* ``lidar_artefact_kernel`` (thread per ``(env, agent)``) right after the lidar, before
  ``lidar_sensor_kernel``: what the scan of a real sensor loses or gains on top of the
  thin-ray geometry (grazing returns lost on the T's faces, random dropouts, spurious
  short returns, a per-robot range scale and bias);
* ``lidar_relabel_kernel`` (thread per ``(env, agent)``) after the pose noise, before the
  observation: the deployment's labelling rule (``swarp_gz_test``'s ``label_rays``),
  which knows only the *reported* poses, replaces the scan's true hit classes;
* ``pusht_lidar_critic_dr_kernel`` (thread per ``(env, agent)``) after the observation
  kernel: the randomised parameters, appended to the role-sorted critic state;
* ``lidar_artefact_reset_kernel`` (thread per ``(env, agent)``, reset only): the
  per-robot artefact parameters of the respawned envs.

The torch twins live in :mod:`swarp.scenarios.pusht_lidar` and share no code with this
module. Random draws are not bit-equal across the two paths; everything else is.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.state import VEC2
from swarp.dynamics.base import P_HALF_X, P_HALF_Y, P_RADIUS, P_SHAPE

# Lean launches (no grid-stride loop); see docs/performance.md.
wp.set_module_options({"default_grid_stride": False})

_HIT_AGENT = wp.constant(wp.uint8(1))
_HIT_IMMOVABLE = wp.constant(wp.uint8(2))
_HIT_MOVABLE = wp.constant(wp.uint8(3))

#: Columns of the per-robot artefact buffer ``[E, A, ART_DIM]``: dropout probability,
#: spurious-return probability, range scale error, range bias.
ART_DROPOUT = 0
ART_SPURIOUS = 1
ART_SCALE = 2
ART_BIAS = 3
ART_DIM = 4


@wp.func
def _uniform(u: float, lo: Any, hi: Any, old: Any):
    """``u`` in ``[0, 1)`` onto ``[lo, hi]``, or ``old`` for a fixed knob (``lo == hi``)."""
    out = old
    if hi > lo:
        out = lo + (hi - lo) * type(lo)(u)
    return out


@wp.kernel
def lidar_artefact_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    dropout_lo: Any,
    dropout_hi: Any,
    spurious_lo: Any,
    spurious_hi: Any,
    scale_w: Any,
    bias_w: Any,
    yaw_w: Any,
    phase_lo: Any,
    phase_hi: Any,
    sweep: wp.int32,
    art: wp.array3d(dtype=Any),
    yaw_bias: wp.array2d(dtype=Any),
    stamp_phase: wp.array2d(dtype=Any),
):
    """Thread per (env, robot), respawned envs only: this robot's dropout and spurious
    probabilities (uniform in their ranges, left alone when fixed), its range scale error
    and bias (uniform in ``+-scale_w`` and ``+-bias_w``) and, with ``sweep``, its mount yaw
    error (uniform in ``+-yaw_w``) and scan stamp phase (uniform in its range). Its own
    seed, so the other reset draws are the same with or without it."""
    e, a = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return
    rng = wp.rand_init(seed, e * art.shape[1] + a)
    one = type(scale_w)(1.0)
    two = type(scale_w)(2.0)
    art[e, a, ART_DROPOUT] = _uniform(wp.randf(rng), dropout_lo, dropout_hi, art[e, a, ART_DROPOUT])
    art[e, a, ART_SPURIOUS] = _uniform(
        wp.randf(rng), spurious_lo, spurious_hi, art[e, a, ART_SPURIOUS]
    )
    art[e, a, ART_SCALE] = scale_w * (two * type(scale_w)(wp.randf(rng)) - one)
    art[e, a, ART_BIAS] = bias_w * (two * type(scale_w)(wp.randf(rng)) - one)
    if sweep == 1:
        yaw_bias[e, a] = yaw_w * (two * type(scale_w)(wp.randf(rng)) - one)
        stamp_phase[e, a] = _uniform(wp.randf(rng), phase_lo, phase_hi, stamp_phase[e, a])


@wp.kernel
def lidar_artefact_kernel(
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    max_range: Any,
    ray_step: Any,
    angle_start: Any,
    body_frame: wp.int32,
    obs_pos: wp.array2d(dtype=Any),
    obs_angle: wp.array2d(dtype=Any),
    obs_half: wp.array(dtype=Any),
    grazing_cos: Any,
    spurious_min: Any,
    art: wp.array3d(dtype=Any),
    rng_state: wp.array2d(dtype=wp.uint32),
    ranges: wp.array3d(dtype=Any),
    hit_class: wp.array3d(dtype=wp.uint8),
    hit_index: wp.array3d(dtype=wp.int32),
):
    """Thread per (env, agent): the scan's artefacts, in place, ray by ray in this order.

    * A hit on the movable body (the T) is lost with probability ``1 - cos_i /
      grazing_cos`` (when positive), ``cos_i`` the incidence cosine on the face the ray
      hit: the hit point in the box's frame, the face whose normalised coordinate is
      largest. ``grazing_cos = 0`` keeps every hit.
    * Any hit is lost with this robot's dropout probability.
    * Any ray, hit or not, returns a spurious hit with this robot's spurious probability,
      at a range uniform between ``spurious_min`` and its current range, classed as the
      movable body (what the deployment's labelling makes of clutter in the arena).
    * Every remaining hit's range becomes ``t (1 + scale) + bias``, clamped to
      ``[0, max_range]``.

    A lost hit reads as a miss (``max_range``, class 0, index -1). The geometry is the
    nominal ray's (``angle_start + r ray_step`` off the heading, from the centre)."""
    e, a = wp.tid()
    n_rays = ranges.shape[2]
    zero = type(max_range)(0.0)
    one = type(max_range)(1.0)
    p = pos[e, a]
    base = zero
    if body_frame != 0:
        base = theta[e, a]
    pd = art[e, a, ART_DROPOUT]
    ps = art[e, a, ART_SPURIOUS]
    kappa = art[e, a, ART_SCALE]
    beta = art[e, a, ART_BIAS]
    rng = rng_state[e, a]
    for r in range(n_rays):
        c = hit_class[e, a, r]
        t = ranges[e, a, r]
        lost = False
        if c == _HIT_MOVABLE and grazing_cos > zero:
            ang = base + angle_start + type(max_range)(r) * ray_step
            dx = wp.cos(ang)
            dy = wp.sin(ang)
            o = hit_index[e, a, r]
            cen = obs_pos[e, o]
            oa = obs_angle[e, o]
            h = obs_half[o]
            co = wp.cos(oa)
            so = wp.sin(oa)
            rx = p[0] + t * dx - cen[0]
            ry = p[1] + t * dy - cen[1]
            lx = co * rx + so * ry
            ly = -so * rx + co * ry
            cos_i = wp.abs(-so * dx + co * dy)
            if wp.abs(lx) * h[1] > wp.abs(ly) * h[0]:
                cos_i = wp.abs(co * dx + so * dy)
            p_lost = one - cos_i / grazing_cos
            if p_lost > zero and type(max_range)(wp.randf(rng)) < p_lost:
                lost = True
        if c != wp.uint8(0) and not lost and pd > zero and type(max_range)(wp.randf(rng)) < pd:
            lost = True
        if lost:
            c = wp.uint8(0)
            t = max_range
            hit_index[e, a, r] = wp.int32(-1)
        if ps > zero and type(max_range)(wp.randf(rng)) < ps and t > spurious_min:
            t = spurious_min + (t - spurious_min) * type(max_range)(wp.randf(rng))
            c = _HIT_MOVABLE
            hit_index[e, a, r] = wp.int32(-1)
        if c != wp.uint8(0):
            t = wp.clamp(t * (one + kappa) + beta, zero, max_range)
        ranges[e, a, r] = t
        hit_class[e, a, r] = c
    rng_state[e, a] = rng


@wp.kernel
def lidar_relabel_kernel(
    obs_pos: wp.array2d(dtype=Any),
    obs_theta: wp.array2d(dtype=Any),
    params: wp.array2d(dtype=Any),
    max_range: Any,
    ray_step: Any,
    angle_start: Any,
    body_frame: wp.int32,
    bound: Any,
    margin: Any,
    per_env_bound: wp.int32,
    arena_half: wp.array(dtype=Any),
    count_leak: wp.int32,
    leak: wp.array2d(dtype=Any),
    flag_outside: wp.int32,
    flag_half: Any,
    ranges: wp.array3d(dtype=Any),
    hit_class: wp.array3d(dtype=wp.uint8),
    outside: wp.array3d(dtype=wp.uint8),
):
    """Thread per (env, agent): classify every hit as the deployment does, from the
    reported poses alone, in place.

    The hit point is placed from this robot's reported pose along the nominal ray. Beyond
    ``bound`` on either axis (with ``per_env_bound``, beyond ``arena_half[e] + bound`` on
    its axis) it is a miss (``max_range``, class 0); inside a teammate's reported footprint
    (its box, or its disc, grown by ``margin``) it is an agent; anything else is the
    movable body. With ``count_leak``, ``leak[e, a]`` gets the share of the rays that hit
    scenery (class 2) and leave labelled as the body. With ``flag_outside``,
    ``outside[e, a, r]`` is 1 where a hit labelled as the body lies past the arena on
    either axis (``arena_half[e]`` with ``per_env_bound``, else ``flag_half``), between the
    arena's edge and the bound, and 0 everywhere else."""
    e, a = wp.tid()
    n_rays = ranges.shape[2]
    n_agents = obs_pos.shape[1]
    zero = type(max_range)(0.0)
    po = obs_pos[e, a]
    base = zero
    if body_frame != 0:
        base = obs_theta[e, a]
    bx = bound
    by = bound
    if per_env_bound == 1:
        bx = arena_half[e][0] + bound
        by = arena_half[e][1] + bound
    ax = flag_half
    ay = flag_half
    if per_env_bound == 1:
        ax = arena_half[e][0]
        ay = arena_half[e][1]
    n_leak = wp.int32(0)
    for r in range(n_rays):
        if flag_outside == 1:
            outside[e, a, r] = wp.uint8(0)
        c_in = hit_class[e, a, r]
        if c_in != wp.uint8(0):
            t = ranges[e, a, r]
            ang = base + angle_start + type(max_range)(r) * ray_step
            hx = po[0] + t * wp.cos(ang)
            hy = po[1] + t * wp.sin(ang)
            c = _HIT_MOVABLE
            if wp.abs(hx) > bx or wp.abs(hy) > by:
                c = wp.uint8(0)
                ranges[e, a, r] = max_range
            else:
                for j in range(n_agents):
                    if j != a and c == _HIT_MOVABLE:
                        pj = obs_pos[e, j]
                        dx = hx - pj[0]
                        dy = hy - pj[1]
                        inside = False
                        if params[j, P_SHAPE] > type(max_range)(0.5):
                            cj = wp.cos(obs_theta[e, j])
                            sj = wp.sin(obs_theta[e, j])
                            lx = cj * dx + sj * dy
                            ly = -sj * dx + cj * dy
                            inside = (
                                wp.abs(lx) <= params[j, P_HALF_X] + margin
                                and wp.abs(ly) <= params[j, P_HALF_Y] + margin
                            )
                        else:
                            inside = wp.sqrt(dx * dx + dy * dy) <= params[j, P_RADIUS] + margin
                        if inside:
                            c = _HIT_AGENT
            if c_in == _HIT_IMMOVABLE and c == _HIT_MOVABLE:
                n_leak += wp.int32(1)
            if flag_outside == 1 and c == _HIT_MOVABLE and (wp.abs(hx) > ax or wp.abs(hy) > ay):
                outside[e, a, r] = wp.uint8(1)
            hit_class[e, a, r] = c
    if count_leak == 1:
        leak[e, a] = type(max_range)(n_leak) / type(max_range)(n_rays)


@wp.kernel
def pusht_lidar_critic_dr_kernel(
    roles: wp.array2d(dtype=wp.int32),
    n_pushers: wp.int32,
    base_col: wp.int32,
    env_cols: wp.int32,
    agent_cols: wp.int32,
    use_mu: wp.int32,
    mu_g: wp.array(dtype=Any),
    inv_gravity: Any,
    use_arena: wp.int32,
    arena_half: wp.array(dtype=Any),
    n_modes: wp.int32,
    scenery_mode: wp.array(dtype=wp.int32),
    act_cols: wp.array(dtype=wp.int32),
    agent_params: wp.array3d(dtype=Any),
    use_delay: wp.int32,
    delay: wp.array2d(dtype=wp.int32),
    use_noise: wp.int32,
    noise_std: wp.array2d(dtype=Any),
    use_pose: wp.int32,
    use_pose_scale: wp.int32,
    pose_scale: wp.array2d(dtype=Any),
    bias_pos: wp.array2d(dtype=Any),
    bias_theta: wp.array2d(dtype=Any),
    use_drift: wp.int32,
    drift_state: wp.array3d(dtype=Any),
    art_cols: wp.array(dtype=wp.int32),
    art: wp.array3d(dtype=Any),
    use_sweep: wp.int32,
    yaw_bias: wp.array2d(dtype=Any),
    stamp_phase: wp.array2d(dtype=Any),
    mid: wp.array(dtype=Any),
    inv: wp.array(dtype=Any),
    critic: wp.array2d(dtype=Any),
):
    """Thread per (env, agent): the randomised parameters, normalised as ``(x - mid) inv``
    per column, from column ``base_col`` of the critic state on. First the env's floor
    friction (``use_mu``; ``mu_g / g``), arena half-extents (``use_arena``) and the one-hot
    of its surroundings' layout (``n_modes`` wide), then per robot in role-sorted order (the
    rank rule
    of ``pusht_lidar_obs_kernel``), ``agent_cols`` wide: the drive parameters in
    ``act_cols`` (a parameter column, or -1 to skip), the command delay in substeps, the
    range-noise deviation, the pose-noise scale (``use_pose_scale``), the localisation bias
    (x, y, heading; ``use_pose``), the drift (x, y, heading; ``use_drift``), the artefact
    columns in ``art_cols`` (an ``art`` column, or -1) and the mount yaw error and stamp
    phase (``use_sweep``). The host decides which columns exist and their ``mid``/``inv``."""
    e, a = wp.tid()
    role = roles[e, a]
    if a == 0:
        c = wp.int32(0)
        if use_mu == 1:
            critic[e, base_col] = (mu_g[e] * inv_gravity - mid[0]) * inv[0]
            c = wp.int32(1)
        if use_arena == 1:
            ah = arena_half[e]
            critic[e, base_col + c] = (ah[0] - mid[c]) * inv[c]
            critic[e, base_col + c + 1] = (ah[1] - mid[c + 1]) * inv[c + 1]
            c += wp.int32(2)
        for i in range(n_modes):
            hot = type(inv_gravity)(0.0)
            if scenery_mode[e] == i:
                hot = type(inv_gravity)(1.0)
            critic[e, base_col + c + i] = (hot - mid[c + i]) * inv[c + i]
    rank = wp.int32(0)
    for j in range(a):
        if roles[e, j] == role:
            rank += wp.int32(1)
    if role == 1:
        rank += n_pushers
    off = base_col + env_cols + agent_cols * rank
    k = wp.int32(0)
    for i in range(act_cols.shape[0]):
        col = act_cols[i]
        if col >= 0:
            m = env_cols + k
            critic[e, off + k] = (agent_params[e, a, col] - mid[m]) * inv[m]
            k += wp.int32(1)
    if use_delay == 1:
        m = env_cols + k
        critic[e, off + k] = (type(inv_gravity)(delay[e, a]) - mid[m]) * inv[m]
        k += wp.int32(1)
    if use_noise == 1:
        m = env_cols + k
        critic[e, off + k] = (noise_std[e, a] - mid[m]) * inv[m]
        k += wp.int32(1)
    if use_pose_scale == 1:
        m = env_cols + k
        critic[e, off + k] = (pose_scale[e, a] - mid[m]) * inv[m]
        k += wp.int32(1)
    if use_pose == 1:
        b = bias_pos[e, a]
        m = env_cols + k
        critic[e, off + k] = (b[0] - mid[m]) * inv[m]
        critic[e, off + k + 1] = (b[1] - mid[m + 1]) * inv[m + 1]
        critic[e, off + k + 2] = (bias_theta[e, a] - mid[m + 2]) * inv[m + 2]
        k += wp.int32(3)
    if use_drift == 1:
        m = env_cols + k
        for i in range(3):
            critic[e, off + k + i] = (drift_state[e, a, 5 + i] - mid[m + i]) * inv[m + i]
        k += wp.int32(3)
    for i in range(art_cols.shape[0]):
        col = art_cols[i]
        if col >= 0:
            m = env_cols + k
            critic[e, off + k] = (art[e, a, col] - mid[m]) * inv[m]
            k += wp.int32(1)
    if use_sweep == 1:
        m = env_cols + k
        critic[e, off + k] = (yaw_bias[e, a] - mid[m]) * inv[m]
        critic[e, off + k + 1] = (stamp_phase[e, a] - mid[m + 1]) * inv[m + 1]


def _artefact_reset_signature(dtype) -> list:
    a2s = wp.array2d(dtype=dtype)
    return [
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # use_mask
        wp.int32,  # seed
        *([dtype] * 9),  # dropout, spurious ranges; scale, bias, yaw widths; phase range
        wp.int32,  # sweep
        wp.array3d(dtype=dtype),  # art
        a2s,  # yaw_bias
        a2s,  # stamp_phase
    ]


def _artefact_signature(dtype) -> list:
    vec2 = VEC2[dtype]
    return [
        wp.array2d(dtype=vec2),  # pos
        wp.array2d(dtype=dtype),  # theta
        *([dtype] * 3),  # max_range, ray_step, angle_start
        wp.int32,  # body_frame
        wp.array2d(dtype=vec2),  # obs_pos
        wp.array2d(dtype=dtype),  # obs_angle
        wp.array(dtype=vec2),  # obs_half
        dtype,  # grazing_cos
        dtype,  # spurious_min
        wp.array3d(dtype=dtype),  # art
        wp.array2d(dtype=wp.uint32),  # rng_state
        wp.array3d(dtype=dtype),  # ranges
        wp.array3d(dtype=wp.uint8),  # hit_class
        wp.array3d(dtype=wp.int32),  # hit_index
    ]


def _relabel_signature(dtype) -> list:
    return [
        wp.array2d(dtype=VEC2[dtype]),  # obs_pos
        wp.array2d(dtype=dtype),  # obs_theta
        wp.array2d(dtype=dtype),  # params
        *([dtype] * 3),  # max_range, ray_step, angle_start
        wp.int32,  # body_frame
        dtype,  # bound
        dtype,  # margin
        wp.int32,  # per_env_bound
        wp.array(dtype=VEC2[dtype]),  # arena_half
        wp.int32,  # count_leak
        wp.array2d(dtype=dtype),  # leak
        wp.int32,  # flag_outside
        dtype,  # flag_half
        wp.array3d(dtype=dtype),  # ranges
        wp.array3d(dtype=wp.uint8),  # hit_class
        wp.array3d(dtype=wp.uint8),  # outside
    ]


def _critic_dr_signature(dtype) -> list:
    a2s = wp.array2d(dtype=dtype)
    i32 = wp.int32
    return [
        wp.array2d(dtype=wp.int32),  # roles
        *([i32] * 4),  # n_pushers, base_col, env_cols, agent_cols
        i32,  # use_mu
        wp.array(dtype=dtype),  # mu_g
        dtype,  # inv_gravity
        i32,  # use_arena
        wp.array(dtype=VEC2[dtype]),  # arena_half
        i32,  # n_modes
        wp.array(dtype=wp.int32),  # scenery_mode
        wp.array(dtype=wp.int32),  # act_cols
        wp.array3d(dtype=dtype),  # agent_params
        i32,  # use_delay
        wp.array2d(dtype=wp.int32),  # delay
        i32,  # use_noise
        a2s,  # noise_std
        *([i32] * 2),  # use_pose, use_pose_scale
        a2s,  # pose_scale
        wp.array2d(dtype=VEC2[dtype]),  # bias_pos
        a2s,  # bias_theta
        i32,  # use_drift
        wp.array3d(dtype=dtype),  # drift_state
        wp.array(dtype=wp.int32),  # art_cols
        wp.array3d(dtype=dtype),  # art
        i32,  # use_sweep
        a2s,  # yaw_bias
        a2s,  # stamp_phase
        wp.array(dtype=dtype),  # mid
        wp.array(dtype=dtype),  # inv
        a2s,  # critic
    ]


for _T in (wp.float32, wp.float64):
    register(lidar_artefact_reset_kernel, _T, _artefact_reset_signature(_T))
    register(lidar_artefact_kernel, _T, _artefact_signature(_T))
    register(lidar_relabel_kernel, _T, _relabel_signature(_T))
    register(pusht_lidar_critic_dr_kernel, _T, _critic_dr_signature(_T))
