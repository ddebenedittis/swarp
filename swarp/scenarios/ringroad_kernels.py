"""Fused Warp kernels for the RingRoadScenario: IDM traffic + obs/reward/reset.

Vehicles share one index space ``v in [0, N)``: the RL cars first (``v < n_agents``, read
from the engine's agent state), then the IDM humans (``v = n_agents + k``, scenario-owned
1-D state ``idm_s``/``idm_v``). Placement around the ring is independent of the index, so
the leader search below finds neighbours by arc position, never by index.

A step pass is ``arc -> leader -> idm -> leader -> obs -> reward``:

* ``ringroad_arc_kernel`` (thread per ``(env, vehicle)``) projects every vehicle onto the
  centreline: arc position ``R * phi`` and the tangential speed. The IDM rows are a copy.
* ``ringroad_leader_kernel`` (thread per ``(env, vehicle)``) is an O(N) scan for the
  nearest vehicle ahead and behind by wrapped arc distance, writing bumper-to-bumper
  gaps and the neighbours' speeds. Brute force, so an RL car that overtakes or falls
  behind needs no sort and no ordering invariant.
* ``ringroad_idm_kernel`` (thread per ``(env, human)``) integrates the IDM with one
  semi-implicit Euler step and refreshes its own arc row. It reads the leader buffers,
  never another vehicle's state, so it cannot race the threads writing those.
* The second leader scan is what lets obs, reward and the crash flag see the state
  *after* the humans moved; reusing the first would hand the policy gaps one step stale.
* ``ringroad_obs_kernel`` (thread per ``(env, agent)``) and ``ringroad_reward_kernel``
  (thread per env).
* ``ringroad_reset_kernel`` (thread per env) draws a whole episode.

The IDM gap is floored at ``s_eps`` and its deceleration at ``decel_floor``. Engine
collisions are off, so an RL car can drive *through* a human; the gap then goes negative
and the raw ``(s*/s)^2`` term would divide by zero. A crash is penalised, never
terminal: ``done`` means "the wave was dissipated" and is counted as solved.

Observation row (``obs_dim = 9``, independent of ``N`` so one actor runs at any ring size)::

    [ v/v_scale, (v_lead - v)/v_scale, gap_lead/gap_scale,
      (v_foll - v)/v_scale, gap_foll/gap_scale,
      (r - R)/lane_half_width, sin(heading error), cos(heading error),
      atan(wheelbase / R) / max_steer ]
"""

from __future__ import annotations

from typing import Any

import warp as wp

from swarp._overloads import register
from swarp.core.state import VEC2
from swarp.scenarios.reset_kernels import _as

# Lean launches (no grid-stride loop); see docs/performance.md.
wp.set_module_options({"default_grid_stride": False})

#: Objective columns of ``info()["multiobj_reward"]``, in order.
OBJ_SPEED = 0
OBJ_ACCEL = 1
OBJ_LANE = 2
OBJ_CRASH = 3
OBJ_TIME = 4
N_OBJ = 5

#: The lane term saturates at this many lane half-widths, so a car that has left the road
#: costs a bounded amount per step rather than a value target that grows without limit.
LANE_CAP = 3.0

OBS_DIM = 9


@wp.kernel
def ringroad_arc_kernel(
    pos: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    idm_s: wp.array2d(dtype=Any),
    idm_v: wp.array2d(dtype=Any),
    ring_R: wp.array(dtype=Any),
    n_agents: wp.int32,
    two_pi: Any,
    arc_s: wp.array2d(dtype=Any),
    arc_v: wp.array2d(dtype=Any),
):
    """Thread per ``(env, vehicle)``: centreline arc position and tangential speed.

    ``phi`` is wrapped into ``[0, 2*pi)`` with caging's conditional add rather than a
    modulo, so the torch oracle reproduces it bit for bit.
    """
    e, v = wp.tid()
    if v >= n_agents:
        k = v - n_agents
        arc_s[e, v] = idm_s[e, k]
        arc_v[e, v] = idm_v[e, k]
        return
    p = pos[e, v]
    x = p[0]
    y = p[1]
    phi = wp.atan2(y, x)
    if phi < type(x)(0.0):
        phi += two_pi
    r = wp.sqrt(x * x + y * y)
    if r < type(x)(1.0e-9):
        r = type(x)(1.0e-9)
    u = vel[e, v]
    arc_s[e, v] = ring_R[e] * phi
    arc_v[e, v] = (u[1] * x - u[0] * y) / r


@wp.kernel
def ringroad_leader_kernel(
    arc_s: wp.array2d(dtype=Any),
    arc_v: wp.array2d(dtype=Any),
    ring_L: wp.array(dtype=Any),
    n_vehicles: wp.int32,
    car_length: Any,
    lead_gap: wp.array2d(dtype=Any),
    lead_v: wp.array2d(dtype=Any),
    foll_gap: wp.array2d(dtype=Any),
    foll_v: wp.array2d(dtype=Any),
):
    """Thread per ``(env, vehicle)``: nearest vehicle ahead and behind, by arc distance.

    Distances are wrapped into ``[0, L)`` with a conditional add. Ties go to the lowest
    index (strict ``<``), which is also what ``torch.argmin`` returns.
    """
    e, i = wp.tid()
    L = ring_L[e]
    si = arc_s[e, i]
    zero = type(L)(0.0)
    best_a = L + L
    best_b = L + L
    ja = i
    jb = i
    for j in range(n_vehicles):
        if j != i:
            sj = arc_s[e, j]
            da = sj - si
            if da < zero:
                da += L
            db = si - sj
            if db < zero:
                db += L
            if da < best_a:
                best_a = da
                ja = j
            if db < best_b:
                best_b = db
                jb = j
    lead_gap[e, i] = best_a - car_length
    lead_v[e, i] = arc_v[e, ja]
    foll_gap[e, i] = best_b - car_length
    foll_v[e, i] = arc_v[e, jb]


@wp.kernel
def ringroad_idm_kernel(
    lead_gap: wp.array2d(dtype=Any),
    lead_v: wp.array2d(dtype=Any),
    ring_L: wp.array(dtype=Any),
    n_agents: wp.int32,
    v0: Any,
    time_headway: Any,
    a_max: Any,
    delta: Any,
    s0: Any,
    inv_2sqrt_ab: Any,
    s_eps: Any,
    decel_floor: Any,
    v_max: Any,
    dt: Any,
    idm_s: wp.array2d(dtype=Any),
    idm_v: wp.array2d(dtype=Any),
    arc_s: wp.array2d(dtype=Any),
    arc_v: wp.array2d(dtype=Any),
):
    """Thread per ``(env, human)``: IDM acceleration, then one semi-implicit Euler step."""
    e, k = wp.tid()
    i = n_agents + k
    v = idm_v[e, k]
    zero = type(v)(0.0)
    one = type(v)(1.0)
    dv = v - lead_v[e, i]
    dyn = v * time_headway + v * dv * inv_2sqrt_ab
    if dyn < zero:
        dyn = zero
    s_star = s0 + dyn
    gap = lead_gap[e, i]
    if gap < s_eps:
        gap = s_eps
    ratio = s_star / gap
    acc = a_max * (one - wp.pow(v / v0, delta) - ratio * ratio)
    if acc < decel_floor:
        acc = decel_floor
    vn = wp.clamp(v + acc * dt, zero, v_max)
    sn = idm_s[e, k] + vn * dt
    L = ring_L[e]
    if sn >= L:
        sn -= L
    idm_v[e, k] = vn
    idm_s[e, k] = sn
    arc_s[e, i] = sn
    arc_v[e, i] = vn


@wp.kernel
def ringroad_obs_kernel(
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    arc_v: wp.array2d(dtype=Any),
    lead_gap: wp.array2d(dtype=Any),
    lead_v: wp.array2d(dtype=Any),
    foll_gap: wp.array2d(dtype=Any),
    foll_v: wp.array2d(dtype=Any),
    ring_R: wp.array(dtype=Any),
    reset_mask: wp.array(dtype=wp.uint8),
    inv_v_scale: Any,
    inv_gap_scale: Any,
    inv_lane_half: Any,
    wheelbase: Any,
    inv_max_steer: Any,
    full_pass: wp.int32,
    obs: wp.array3d(dtype=Any),
):
    """Thread per ``(env, agent)``. Skips envs an obs-only auto-reset did not select."""
    e, a = wp.tid()
    if full_pass == 0 and reset_mask[e] == wp.uint8(0):
        return
    p = pos[e, a]
    x = p[0]
    y = p[1]
    R = ring_R[e]
    phi = wp.atan2(y, x)
    r = wp.sqrt(x * x + y * y)
    rel = theta[e, a] - phi
    v = arc_v[e, a]
    obs[e, a, 0] = v * inv_v_scale
    obs[e, a, 1] = (lead_v[e, a] - v) * inv_v_scale
    obs[e, a, 2] = lead_gap[e, a] * inv_gap_scale
    obs[e, a, 3] = (foll_v[e, a] - v) * inv_v_scale
    obs[e, a, 4] = foll_gap[e, a] * inv_gap_scale
    obs[e, a, 5] = (r - R) * inv_lane_half
    # heading error = theta - (phi + pi/2): sin(err) = -cos(rel), cos(err) = sin(rel)
    obs[e, a, 6] = -wp.cos(rel)
    obs[e, a, 7] = wp.sin(rel)
    obs[e, a, 8] = wp.atan(wheelbase / R) * inv_max_steer


@wp.kernel
def ringroad_reward_kernel(
    pos: wp.array2d(dtype=Any),
    speed: wp.array2d(dtype=Any),
    arc_v: wp.array2d(dtype=Any),
    lead_gap: wp.array2d(dtype=Any),
    foll_gap: wp.array2d(dtype=Any),
    ring_R: wp.array(dtype=Any),
    v_eq: wp.array(dtype=Any),
    reset_mask: wp.array(dtype=wp.uint8),
    n_agents: wp.int32,
    n_vehicles: wp.int32,
    inv_n: Any,
    speed_coef: Any,
    accel_coef: Any,
    inv_dt: Any,
    lane_coef: Any,
    inv_lane_half: Any,
    lane_cap: Any,
    crash_penalty: Any,
    time_penalty: Any,
    std_threshold: Any,
    speed_fraction: Any,
    hold_steps: wp.int32,
    advance_prev: wp.int32,
    full_pass: wp.int32,
    prev_speed: wp.array2d(dtype=Any),
    hold: wp.array(dtype=wp.uint8),
    reward: wp.array2d(dtype=Any),
    done: wp.array(dtype=wp.uint8),
    multiobj: wp.array3d(dtype=Any),
    std_out: wp.array(dtype=Any),
    mean_out: wp.array(dtype=Any),
    crashed_out: wp.array(dtype=Any),
    lane_out: wp.array2d(dtype=Any),
):
    """Thread per env: two O(N) passes for the speed mean and spread, then per agent.

    The speed term is Flow's ``WaveAttenuationEnv`` reward (``4/20 * mean speed`` over all
    vehicles) and the accel term its ``-4 * mean |accel|``, applied per agent to the
    *realized* acceleration ``(speed - prev_speed) / dt``: the engine does not hand the
    scenario its actions. ``multiobj``'s last-dim sum is exactly ``reward``.
    """
    e = wp.tid()
    zero = type(speed_coef)(0.0)
    one = type(speed_coef)(1.0)
    sv = zero
    crashed = zero
    for i in range(n_vehicles):
        sv += arc_v[e, i]
        if lead_gap[e, i] < zero:
            crashed = one
    mean_v = sv * inv_n
    dev = zero
    for i in range(n_vehicles):
        d = arc_v[e, i] - mean_v
        dev += d * d
    std_v = wp.sqrt(dev * inv_n)

    stable = 0
    if std_v < std_threshold and mean_v > speed_fraction * v_eq[e]:
        stable = 1
    h = wp.int32(hold[e])
    if advance_prev == 1:
        if stable == 1:
            if h < 255:
                h += wp.int32(1)
        else:
            h = wp.int32(0)
        hold[e] = wp.uint8(h)

    rst = reset_mask[e]
    R = ring_R[e]
    speed_term = speed_coef * mean_v
    for a in range(n_agents):
        s = speed[e, a]
        acc_term = -accel_coef * wp.abs(s - prev_speed[e, a]) * inv_dt
        if rst == wp.uint8(1):
            acc_term = zero
            prev_speed[e, a] = s
        elif advance_prev == 1:
            prev_speed[e, a] = s
        if full_pass == 1:
            p = pos[e, a]
            off = wp.sqrt(p[0] * p[0] + p[1] * p[1]) - R
            q = wp.clamp(off * inv_lane_half, -lane_cap, lane_cap)
            lane_term = -lane_coef * q * q
            crash_term = zero
            if lead_gap[e, a] < zero or foll_gap[e, a] < zero:
                crash_term = crash_penalty
            multiobj[e, a, OBJ_SPEED] = speed_term
            multiobj[e, a, OBJ_ACCEL] = acc_term
            multiobj[e, a, OBJ_LANE] = lane_term
            multiobj[e, a, OBJ_CRASH] = crash_term
            multiobj[e, a, OBJ_TIME] = time_penalty
            reward[e, a] = speed_term + acc_term + lane_term + crash_term + time_penalty
            lane_out[e, a] = off

    if full_pass == 1:
        if h >= hold_steps:
            done[e] = wp.uint8(1)
        else:
            done[e] = wp.uint8(0)
        std_out[e] = std_v
        mean_out[e] = mean_v
        crashed_out[e] = crashed


@wp.kernel
def ringroad_reset_kernel(
    reset_mask: wp.array(dtype=wp.uint8),
    use_mask: wp.int32,
    seed: wp.int32,
    difficulty: wp.array(dtype=Any),
    ring_len_lo: Any,
    ring_len_hi: Any,
    car_length: Any,
    gap_amp: Any,
    v_amp: Any,
    v0: Any,
    time_headway: Any,
    delta: Any,
    s0: Any,
    v_max: Any,
    n_agents: wp.int32,
    n_vehicles: wp.int32,
    pos: wp.array2d(dtype=Any),
    theta: wp.array2d(dtype=Any),
    vel: wp.array2d(dtype=Any),
    speed: wp.array2d(dtype=Any),
    ang_vel: wp.array2d(dtype=Any),
    idm_s: wp.array2d(dtype=Any),
    idm_v: wp.array2d(dtype=Any),
    ring_R: wp.array(dtype=Any),
    ring_L: wp.array(dtype=Any),
    v_eq: wp.array(dtype=Any),
    hold: wp.array(dtype=wp.uint8),
):
    """One masked episode per env: ring length, IDM equilibrium, and an m=1 wave.

    Vehicles sit at uniform spacing ``L/N`` plus an ``m = 1`` perturbation of the gap
    (``f * gap_amp``, capped at 80% of the equilibrium bumper gap so nobody starts
    overlapped) and of the speed (``f * v_amp``), in phase: the car with the larger gap
    ahead is the faster one, which is the shape of a travelling wave. At this density a
    +/-3 m, +/-3 m/s start already has a speed spread above 2 m/s at N=22, so no warm-up
    is needed for a wave to exist, while at N <= 8 the IDM alone damps it.

    The equilibrium speed solves ``1 - (v/v0)^delta = ((s0 + v*T)/gap)^2`` by 60-step
    bisection, per env because the ring length can be drawn per env. RL car ``a`` takes
    slot ``(a * N) // n_agents``, so several RL cars are spread evenly, and starts on the
    centreline heading along the tangent.
    """
    e = wp.tid()
    if use_mask == 1 and reset_mask[e] == wp.uint8(0):
        return
    rng = wp.rand_init(seed, e)
    zero = _as(0.0, v0)
    one = _as(1.0, v0)
    half = _as(0.5, v0)
    tau = _as(6.28318530717959, v0)
    f = difficulty[0]

    L = ring_len_lo + (ring_len_hi - ring_len_lo) * _as(wp.randf(rng), v0)
    R = L / tau
    nf = _as(wp.float32(n_vehicles), v0)
    spacing = L / nf
    gap0 = spacing - car_length

    lo = zero
    hi = v0
    for _ in range(60):
        mid = half * (lo + hi)
        q = (s0 + mid * time_headway) / gap0
        if one - wp.pow(mid / v0, delta) - q * q > zero:
            lo = mid
        else:
            hi = mid
    ve = wp.min(half * (lo + hi), v_max)

    ga = f * gap_amp
    cap = _as(0.8, v0) * gap0
    if ga > cap:
        ga = cap
    va = f * v_amp
    phase = _as(wp.randf(rng), v0) * tau
    x = _as(wp.randf(rng), v0) * L

    rl = wp.int32(0)
    k = wp.int32(0)
    for i in range(n_vehicles):
        th = tau * _as(wp.float32(i), v0) / nf + phase
        sn = wp.sin(th)
        xi = x
        if xi >= L:
            xi -= L
        vi = wp.clamp(ve + va * sn, zero, v_max)
        if rl < n_agents and i == (rl * n_vehicles) // n_agents:
            ang = xi / R
            hd = ang + half * _as(3.14159265358979, v0)
            pos[e, rl] = wp.vector(R * wp.cos(ang), R * wp.sin(ang))
            theta[e, rl] = hd
            vel[e, rl] = wp.vector(vi * wp.cos(hd), vi * wp.sin(hd))
            speed[e, rl] = vi
            ang_vel[e, rl] = zero
            rl += wp.int32(1)
        else:
            idm_s[e, k] = xi
            idm_v[e, k] = vi
            k += wp.int32(1)
        x = x + spacing + ga * sn

    ring_R[e] = R
    ring_L[e] = L
    v_eq[e] = ve
    hold[e] = wp.uint8(0)


def _arc_signature(dtype) -> list:
    a2v = wp.array2d(dtype=VEC2[dtype])
    a2s = wp.array2d(dtype=dtype)
    return [
        a2v,  # pos
        a2v,  # vel
        a2s,  # idm_s
        a2s,  # idm_v
        wp.array(dtype=dtype),  # ring_R
        wp.int32,  # n_agents
        dtype,  # two_pi
        a2s,  # arc_s
        a2s,  # arc_v
    ]


def _leader_signature(dtype) -> list:
    a2s = wp.array2d(dtype=dtype)
    return [
        a2s,  # arc_s
        a2s,  # arc_v
        wp.array(dtype=dtype),  # ring_L
        wp.int32,  # n_vehicles
        dtype,  # car_length
        a2s,  # lead_gap
        a2s,  # lead_v
        a2s,  # foll_gap
        a2s,  # foll_v
    ]


def _idm_signature(dtype) -> list:
    a2s = wp.array2d(dtype=dtype)
    return [
        a2s,  # lead_gap
        a2s,  # lead_v
        wp.array(dtype=dtype),  # ring_L
        wp.int32,  # n_agents
        dtype,  # v0
        dtype,  # time_headway
        dtype,  # a_max
        dtype,  # delta
        dtype,  # s0
        dtype,  # inv_2sqrt_ab
        dtype,  # s_eps
        dtype,  # decel_floor
        dtype,  # v_max
        dtype,  # dt
        a2s,  # idm_s
        a2s,  # idm_v
        a2s,  # arc_s
        a2s,  # arc_v
    ]


def _obs_signature(dtype) -> list:
    a2s = wp.array2d(dtype=dtype)
    return [
        wp.array2d(dtype=VEC2[dtype]),  # pos
        a2s,  # theta
        a2s,  # arc_v
        a2s,  # lead_gap
        a2s,  # lead_v
        a2s,  # foll_gap
        a2s,  # foll_v
        wp.array(dtype=dtype),  # ring_R
        wp.array(dtype=wp.uint8),  # reset_mask
        dtype,  # inv_v_scale
        dtype,  # inv_gap_scale
        dtype,  # inv_lane_half
        dtype,  # wheelbase
        dtype,  # inv_max_steer
        wp.int32,  # full_pass
        wp.array3d(dtype=dtype),  # obs
    ]


def _reward_signature(dtype) -> list:
    a1s = wp.array(dtype=dtype)
    a2s = wp.array2d(dtype=dtype)
    return [
        wp.array2d(dtype=VEC2[dtype]),  # pos
        a2s,  # speed
        a2s,  # arc_v
        a2s,  # lead_gap
        a2s,  # foll_gap
        a1s,  # ring_R
        a1s,  # v_eq
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # n_agents
        wp.int32,  # n_vehicles
        dtype,  # inv_n
        dtype,  # speed_coef
        dtype,  # accel_coef
        dtype,  # inv_dt
        dtype,  # lane_coef
        dtype,  # inv_lane_half
        dtype,  # lane_cap
        dtype,  # crash_penalty
        dtype,  # time_penalty
        dtype,  # std_threshold
        dtype,  # speed_fraction
        wp.int32,  # hold_steps
        wp.int32,  # advance_prev
        wp.int32,  # full_pass
        a2s,  # prev_speed
        wp.array(dtype=wp.uint8),  # hold
        a2s,  # reward
        wp.array(dtype=wp.uint8),  # done
        wp.array3d(dtype=dtype),  # multiobj
        a1s,  # std_out
        a1s,  # mean_out
        a1s,  # crashed_out
        a2s,  # lane_out
    ]


def _reset_signature(dtype) -> list:
    a1s = wp.array(dtype=dtype)
    a2s = wp.array2d(dtype=dtype)
    a2v = wp.array2d(dtype=VEC2[dtype])
    return [
        wp.array(dtype=wp.uint8),  # reset_mask
        wp.int32,  # use_mask
        wp.int32,  # seed
        a1s,  # difficulty (by pointer: a scalar would be baked into a capture)
        dtype,  # ring_len_lo
        dtype,  # ring_len_hi
        dtype,  # car_length
        dtype,  # gap_amp
        dtype,  # v_amp
        dtype,  # v0
        dtype,  # time_headway
        dtype,  # delta
        dtype,  # s0
        dtype,  # v_max
        wp.int32,  # n_agents
        wp.int32,  # n_vehicles
        a2v,  # pos
        a2s,  # theta
        a2v,  # vel
        a2s,  # speed
        a2s,  # ang_vel
        a2s,  # idm_s
        a2s,  # idm_v
        a1s,  # ring_R
        a1s,  # ring_L
        a1s,  # v_eq
        wp.array(dtype=wp.uint8),  # hold
    ]


for _T in (wp.float32, wp.float64):
    register(ringroad_arc_kernel, _T, _arc_signature(_T))
    register(ringroad_leader_kernel, _T, _leader_signature(_T))
    register(ringroad_idm_kernel, _T, _idm_signature(_T))
    register(ringroad_obs_kernel, _T, _obs_signature(_T))
    register(ringroad_reward_kernel, _T, _reward_signature(_T))
    register(ringroad_reset_kernel, _T, _reset_signature(_T))
