"""The obstacle-install contract: ``Obstacles`` normalization and ``set_obstacles``.

Two things are pinned here.

**Capture safety** (``test_in_place_install_*``). ``TransportScenario`` re-installs its
package obstacles from *inside* the captured whole-step graph — torch code inside
``wp.ScopedCapture``, see ``swarp/interop/persistent.py``'s ``set_post_physics`` contract.
That is legal only because the steady-state in-place install allocates nothing, launches
nothing on torch's stream, and never reads the device back to the host. Nothing used to
test it, so an innocuous edit to ``set_obstacles`` could silently corrupt transport's graph.
Three independent checks cover it: ``torch.cuda.memory_allocated()`` across the call, the
host-transfer entry points patched to raise, and a real graph capture + replay.

**Uniform install semantics.** An absent field means *the documented default*, on both the
in-place and the reallocating path — the two used to disagree, and the disagreements were
real bugs (a stale array tail on a batch-size change, mass/inertia stale in place but reset
on realloc, body velocity dropped on realloc, the input tensor's device adopted on realloc).
"""

import contextlib
from unittest import mock

import pytest
import torch
import warp as wp
from conftest import CUDA, DEVICES, holo_cfgs

from swarp.core.config import ObstacleKind, Obstacles, ObstacleShape, WorldConfig
from swarp.core.stepper import Stepper
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel

# --------------------------------------------------------------------------- helpers


def _stepper(device, dtype=wp.float32, **cfg_kw):
    world = WorldConfig(collision_k=100.0, collision_c=0.0, collision_margin=0.02, **cfg_kw)
    return Stepper(holo_cfgs(1), dt=0.1, device=device, dtype=dtype, world=world)


def _pos(n_envs, n_obs, device, dtype=torch.float32, base=0.0):
    """Distinct per-(env, obstacle) centres, so a stale tail or a partial copy shows up."""
    return (
        base
        + torch.arange(n_envs * n_obs * 2, device=device, dtype=dtype).view(n_envs, n_obs, 2) * 0.1
    )


def _radius(n_obs, device, dtype=torch.float32):
    return torch.full((n_obs,), 0.1, device=device, dtype=dtype)


def _movable(n_envs, n_obs, device, **kw):
    """A spec whose obstacles are all MOVABLE (so the body group is live)."""
    return Obstacles(
        _pos(n_envs, n_obs, device),
        _radius(n_obs, device),
        kind=torch.full((n_obs,), int(ObstacleKind.MOVABLE), device=device, dtype=torch.int32),
        **kw,
    )


def _t(arr):
    return wp.to_torch(arr)


def _raise(*args, **kwargs):
    raise AssertionError("device->host transfer during an in-place obstacle install")


@contextlib.contextmanager
def forbid_host_transfers():
    """Same guard as ``tests/interop/test_no_host_copies.py``, for the install path."""
    with (
        mock.patch.object(torch.Tensor, "cpu", _raise),
        mock.patch.object(torch.Tensor, "item", _raise),
        mock.patch.object(torch.Tensor, "numpy", _raise),
        mock.patch.object(torch.Tensor, "tolist", _raise),
        mock.patch.object(torch.Tensor, "__bool__", _raise),
        mock.patch.object(wp.array, "numpy", _raise),
    ):
        yield


# ------------------------------------------------------- Obstacles normalization


def test_resolve_fills_nothing_and_is_memoized():
    """Resolution normalizes *present* fields only; absent ones stay None so the installer
    can fill their default without allocating. Resolving twice returns the same object."""
    spec = Obstacles(_pos(2, 3, "cpu"), _radius(3, "cpu"))
    res = spec.resolve("cpu", torch.float32)
    assert res.shape is None and res.angle is None and res.mass is None
    assert res.resolve("cpu", torch.float32) is res  # memoized: re-installing is free
    # ...but a different target is resolved afresh.
    assert res.resolve("cpu", torch.float64) is not res


def test_resolve_broadcasts_a_shared_angle_and_casts_ints():
    spec = Obstacles(
        _pos(4, 2, "cpu", dtype=torch.float64),
        _radius(2, "cpu", dtype=torch.float64),
        angle=torch.tensor([0.5, 1.5], dtype=torch.float64),
        shape=torch.tensor([1.0, 2.0]),  # deliberately float: resolve casts to int32
        kind=torch.tensor([0.0, 1.0]),
    )
    res = spec.resolve("cpu", torch.float64)
    assert res.angle.shape == (4, 2) and res.angle.is_contiguous()
    torch.testing.assert_close(res.angle[0], res.angle[3])
    assert res.shape.dtype == torch.int32 and res.kind.dtype == torch.int32
    # A per-env angle is passed through as-is (the other half of the two-form contract).
    per_env = Obstacles(
        _pos(4, 2, "cpu"), _radius(2, "cpu"), angle=torch.zeros(4, 2)
    ).resolve("cpu", torch.float32)
    assert per_env.angle.shape == (4, 2)


def test_any_movable_is_free_without_kind():
    """``kind is None`` answers False without touching the device — the property that keeps
    a kind-free install legal inside a graph capture."""
    spec = Obstacles(_pos(2, 3, "cpu"), _radius(3, "cpu"))
    with forbid_host_transfers():
        assert spec.any_movable is False


def test_shape_validation():
    with pytest.raises(ValueError, match=r"pos must be \[n_envs"):
        Obstacles(torch.zeros(3, 2), torch.zeros(2))
    with pytest.raises(ValueError, match="radius must be"):
        Obstacles(torch.zeros(2, 3, 2), torch.zeros(2))
    with pytest.raises(ValueError, match="angle must be"):
        Obstacles(torch.zeros(2, 3, 2), torch.zeros(3), angle=torch.zeros(2, 3, 1))


# ------------------------------------------------- capture safety (the Wave 3 dependency)


@pytest.mark.parametrize("device", DEVICES)
def test_in_place_install_makes_no_host_read(device):
    """The steady-state in-place install must not read the device back to the host.

    ``tests/interop/test_no_host_copies.py`` patches the same entry points; this pins the
    obstacle refresh specifically, because it is the one piece of torch code that runs
    inside transport's captured whole-step graph.
    """
    st = _stepper(device)
    pos, radius = _pos(4, 3, device), _radius(3, device)
    st.set_obstacles(Obstacles(pos, radius))  # first install: reallocates
    st.set_obstacles(Obstacles(pos, radius))  # warm up the in-place path
    ver = st.mutation_version
    with forbid_host_transfers():
        pos.add_(0.01)
        st.set_obstacles(Obstacles(pos, radius))
    assert st.mutation_version == ver, "an in-place refresh must not force a graph recapture"
    torch.testing.assert_close(_t(st.obs_pos), pos)


@pytest.mark.parametrize("device", DEVICES)
def test_in_place_install_follows_a_reassigned_source_tensor(device):
    """A fresh source tensor must be picked up, not written through a stale Warp view.

    ``Stepper._src_handle`` caches the zero-copy wrap of each install source, because
    rebuilding it per field per step was most of the cost of a reset that re-samples
    obstacle poses. The cache is what this pins: the grad path's ``_refresh`` reassigns
    these tensors (fresh buffers for the tape) rather than updating them in place, and a
    handle still pointing at the old address would silently install stale poses.
    """
    st = _stepper(device)
    radius = _radius(3, device)
    pos = _pos(4, 3, device)
    st.set_obstacles(Obstacles(pos, radius))  # reallocates
    st.set_obstacles(Obstacles(pos, radius))  # warm the in-place path and the handle cache

    replacement = _pos(4, 3, device) + 0.5  # a different allocation, different values
    assert replacement.data_ptr() != pos.data_ptr()
    st.set_obstacles(Obstacles(replacement, radius))
    torch.testing.assert_close(_t(st.obs_pos), replacement)

    # ...and an in-place edit of the *new* tensor still lands, i.e. the rebuilt handle
    # views it rather than a copy of it.
    replacement.add_(0.25)
    st.set_obstacles(Obstacles(replacement, radius))
    torch.testing.assert_close(_t(st.obs_pos), replacement)


@pytest.mark.parametrize("device", DEVICES)
def test_in_place_install_reuses_one_source_handle(device):
    """The steady state builds no new wrap: same tensor in, same cached view out."""
    st = _stepper(device)
    pos, radius = _pos(4, 3, device), _radius(3, device)
    st.set_obstacles(Obstacles(pos, radius))
    st.set_obstacles(Obstacles(pos, radius))
    first = st._src_handles["obs_pos"][1]
    pos.add_(0.01)
    st.set_obstacles(Obstacles(pos, radius))
    assert st._src_handles["obs_pos"][1] is first, "the source wrap was rebuilt needlessly"


@pytest.mark.gpu(reason="torch.cuda.memory_allocated is the allocation oracle")
def test_in_place_install_allocates_nothing():
    """Zero bytes from torch's allocator across the call, spec construction included.

    ``body_arrays``/``angle_2d`` used to build torch temporaries here (arange, gathers,
    cos/sin, stack); allocation is illegal under ``wp.ScopedCapture``, and a freed
    temporary's pointer baked into a captured graph is worse than an error.
    """
    dev = "cuda:0"
    st = _stepper(dev)
    pos, radius = _pos(8, 3, dev), _radius(3, dev)
    for _ in range(2):  # first install reallocates; second warms the in-place path
        st.set_obstacles(Obstacles(pos, radius))
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(5):
        st.set_obstacles(Obstacles(pos, radius))
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


@pytest.mark.gpu(reason="CUDA-graph capture needs a GPU")
def test_in_place_install_is_graph_capturable():
    """The install replays correctly from inside a captured graph.

    This is the regime ``TransportScenario._graph_post_physics`` runs in. Mutating the
    source tensor between replays and seeing the change land proves the copies really are
    recorded in the graph (rather than the capture having silently no-oped).
    """
    dev = "cuda:0"
    st = _stepper(dev)
    pos, radius = _pos(8, 3, dev), _radius(3, dev)
    for _ in range(2):
        st.set_obstacles(Obstacles(pos, radius))
    wp.synchronize_device(dev)
    with wp.ScopedCapture(device=dev) as capture:
        # Built inside the capture, exactly as transport's _install_obstacles does it.
        st.set_obstacles(Obstacles(pos.detach(), radius))
    for scale in (2.0, 3.0):
        pos.mul_(scale)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(dev)
        torch.testing.assert_close(_t(st.obs_pos), pos)


@pytest.mark.parametrize("device", DEVICES)
def test_movable_install_writes_the_body_group(device):
    """The negative side of the contract: a spec with movable obstacles *does* derive the
    body group (torch temporaries + a host read of ``any_movable``), so such an install
    must stay outside a capture. Both halves are asserted so the split stays deliberate."""
    st = _stepper(device)
    spec = _movable(2, 2, device)
    st.set_obstacles(spec)
    st.set_obstacles(spec)  # in place
    assert st.any_movable
    # An immovable refresh of the same count leaves the body group alone (nothing reads it)
    # and is host-read-free.
    with forbid_host_transfers():
        st.set_obstacles(Obstacles(_pos(2, 2, device), _radius(2, device)))
    assert st.any_movable is False


# -------------------------------------------------------------------- defect 1: n_envs


@pytest.mark.parametrize("device", DEVICES)
def test_smaller_batch_reinstalls_instead_of_leaving_a_stale_tail(device):
    """``wp.copy`` sizes itself from the *source* and only raises on overflow, so a batch
    shrink used to copy into the head of the array and leave the tail at the old values."""
    st = _stepper(device)
    st.set_obstacles(Obstacles(_pos(4, 2, device, base=100.0), _radius(2, device)))
    small = _pos(2, 2, device)
    st.set_obstacles(Obstacles(small, _radius(2, device)))
    assert tuple(st.obs_pos.shape) == (2, 2)
    torch.testing.assert_close(_t(st.obs_pos), small)
    assert tuple(st.obs_angle.shape) == (2, 2)
    assert tuple(st.obs_vel.shape) == (2, 2)


@pytest.mark.parametrize("device", DEVICES)
def test_larger_batch_reinstalls_instead_of_raising(device):
    st = _stepper(device)
    st.set_obstacles(Obstacles(_pos(2, 2, device), _radius(2, device)))
    big = _pos(8, 2, device)
    st.set_obstacles(Obstacles(big, _radius(2, device)))  # used to raise on overflow
    assert tuple(st.obs_pos.shape) == (8, 2)
    torch.testing.assert_close(_t(st.obs_pos), big)


# ------------------------------------------------------------- defect 3: mass / inertia


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("regime", ["in_place", "realloc"])
def test_absent_mass_resets_to_the_default_on_both_paths(device, regime):
    """An omitted field means the documented default (1.0), never a stale value. In place
    the old code kept the previous mass; a reallocation reset it — the same call gave two
    different answers depending on whether the obstacle count happened to match."""
    st = _stepper(device)
    kinds = torch.full((3,), int(ObstacleKind.MOVABLE), device=device, dtype=torch.int32)
    st.set_obstacles(
        Obstacles(
            _pos(2, 3, device),
            _radius(3, device),
            kind=kinds,
            mass=torch.full((3,), 5.0, device=device),
            inertia=torch.full((3,), 7.0, device=device),
        )
    )
    torch.testing.assert_close(_t(st.obs_mass), torch.full((3,), 5.0, device=device))
    n_envs = 2 if regime == "in_place" else 5
    st.set_obstacles(Obstacles(_pos(n_envs, 3, device), _radius(3, device), kind=kinds))
    torch.testing.assert_close(_t(st.obs_mass), torch.ones(3, device=device))
    torch.testing.assert_close(_t(st.obs_inertia), torch.ones(3, device=device))


# ------------------------------------------------------ defect 4: body velocity seeding


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("regime", ["first_install", "in_place"])
def test_body_velocity_is_seeded_from_vel_on_both_paths(device, regime):
    """Handing over a *moving* body's whole state must work on the first install too. The
    reallocating path used to zero ``body_vel``/``body_ang_vel`` unconditionally, so a body
    silently started at rest unless the obstacle count already happened to match."""
    st = _stepper(device)
    vel = torch.full((2, 2, 2), 0.75, device=device)
    ang_vel = torch.full((2, 2), -1.25, device=device)
    spec = _movable(2, 2, device, vel=vel, ang_vel=ang_vel)
    if regime == "in_place":
        st.set_obstacles(_movable(2, 2, device))  # allocate first, then refresh in place
    st.set_obstacles(spec)
    torch.testing.assert_close(_t(st.body_vel), vel)
    torch.testing.assert_close(_t(st.body_ang_vel), ang_vel)


@pytest.mark.parametrize("device", DEVICES)
def test_compound_body_origin_is_derived_from_the_root_shape(device):
    """``body_pos`` is ``root_pos - R(angle) @ offset``: a caller only ever describes where
    the *shapes* are, and the engine derives the body origin back out."""
    st = _stepper(device)
    off = torch.tensor([[0.0, 0.3], [0.0, -0.3]], device=device)
    centers = torch.tensor([[[1.0, 0.3], [1.0, -0.3]]], device=device)
    st.set_obstacles(
        Obstacles(
            centers,
            torch.zeros(2, device=device),
            shape=torch.full((2,), int(ObstacleShape.BOX), device=device, dtype=torch.int32),
            angle=torch.zeros(1, 2, device=device),
            half_extents=torch.full((2, 2), 0.1, device=device),
            kind=torch.full((2,), int(ObstacleKind.MOVABLE), device=device, dtype=torch.int32),
            body=torch.zeros(2, device=device, dtype=torch.int32),  # one compound body
            body_offset=off,
        )
    )
    # Both shapes belong to body 0, whose origin is the T centroid at (1.0, 0.0).
    torch.testing.assert_close(_t(st.body_pos)[0, 0], torch.tensor([1.0, 0.0], device=device))


# ------------------------------------------------------------------- defect 5: device


@pytest.mark.gpu(reason="a device mismatch needs two devices")
@pytest.mark.parametrize("regime", ["realloc", "in_place"])
def test_cpu_input_lands_on_the_stepper_device(regime):
    """A CPU spec installed into a CUDA stepper must not produce a mixed-device obstacle
    set. The reallocating path used ``wp.clone(wp.from_torch(...))`` with no ``device=``,
    adopting the *input's* device while the zero-fill fallbacks stayed on the stepper's."""
    dev = "cuda:0"
    st = _stepper(dev)
    if regime == "in_place":
        st.set_obstacles(Obstacles(_pos(2, 3, dev), _radius(3, dev)))
    spec = Obstacles(
        _pos(2, 3, "cpu"),
        _radius(3, "cpu"),
        kind=torch.full((3,), int(ObstacleKind.MOVABLE), dtype=torch.int32),
        mass=torch.full((3,), 2.0),
    )
    st.set_obstacles(spec)
    arrays = (
        st.obs_pos, st.obs_radius, st.obs_type, st.obs_angle, st.obs_half, st.obs_vel,
        st.obs_ang_vel, st.obs_kind, st.obs_mass, st.obs_inertia, st.obs_body,
        st.obs_body_off, st.body_pos, st.body_angle, st.body_vel, st.body_ang_vel,
    )
    assert all(str(a.device) == dev for a in arrays)
    torch.testing.assert_close(_t(st.obs_pos), _pos(2, 3, dev))


# ------------------------------------- defect 2: a pose-only re-install keeps bodies alive


def _movable_world(device, n_envs=2):
    # max_speed=1.0 (not conftest's 100.0): the push tests need an agent that shoves the
    # body rather than teleporting through it.
    cfgs = [
        AgentConfig(
            model=DynamicsModel.HOLONOMIC, ctrl_mode=ControlMode.VELOCITY, radius=0.05,
            max_speed=1.0,
        )
    ]
    w = World(
        cfgs,
        WorldConfig(
            collisions=True,
            collision_k=1000.0,
            collision_c=20.0,
            collision_margin=0.01,
            bounds=(-2.0, 2.0, -2.0, 2.0),
        ),
        n_envs=n_envs,
        device=device,
        dt=0.05,
        substeps=4,
    )
    n = 1
    w.set_obstacles(
        Obstacles(
            torch.zeros(n_envs, n, 2, device=device),
            torch.full((n,), 0.1, device=device),
            kind=torch.full((n,), int(ObstacleKind.MOVABLE), device=device, dtype=torch.int32),
            mass=torch.full((n,), 1.0, device=device),
            inertia=torch.full((n,), 0.01, device=device),
        )
    )
    return w


def _push(w, steps=30, start=(-0.4, 0.0)):
    w.state.pos.data.copy_(torch.tensor([[list(start)]], device=w.device).expand(w.n_envs, 1, 2))
    w.state.vel.data.zero_()
    act = torch.tensor([[[1.0, 0.0]]], device=w.device).expand(w.n_envs, 1, 2).contiguous()
    for _ in range(steps):
        w.step(act)
    return w.obstacle_state_views()


@pytest.mark.parametrize("device", DEVICES)
def test_pose_only_reinstall_keeps_a_movable_body_movable(device):
    """The interactive obstacle drag: write a pose into the retained spec and re-install it.

    The viewer used to rebuild a partial spec naming only the pose fields, which reset
    ``kind`` to IMMOVABLE and turned every movable body into permanent scenery on the first
    drag (and, because the in-place path bumps no version, without even recapturing the
    graph). Retaining the whole spec is what makes this safe.
    """
    w = _movable_world(device)
    assert w.stepper.any_movable
    # Drag: move the obstacle in one env, then re-install the retained spec.
    w.obstacles.pos[0, 0] = torch.tensor([0.2, 0.1], device=device)
    w.set_obstacles(w.obstacles)
    assert w.stepper.any_movable, "a pose-only re-install froze the movable body"
    assert w.obstacle_kind is not None and int(w.obstacle_kind[0]) == int(ObstacleKind.MOVABLE)
    torch.testing.assert_close(_t(w.stepper.obs_mass), torch.ones(1, device=device))
    torch.testing.assert_close(
        _t(w.stepper.obs_inertia), torch.full((1,), 0.01, device=device)
    )
    pos, _, vel, _ = _push(w, start=(-0.4, 0.1))
    assert pos[1, 0, 0].item() > 0.05, "the body stopped being pushed after a re-install"
    assert vel[1, 0, 0].item() > 0.0


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_pose_only_reinstall_does_not_recapture_the_graph():
    """The drag path must stay on the no-bump branch: an unchanged obstacle count is an
    in-place refresh, so no CUDA-graph recapture and no cached-buffer churn."""
    w = _movable_world("cuda:0")
    ver, buffers = w.stepper.mutation_version, dict(w.stepper._cached_buffers)
    w.set_obstacles(w.obstacles)
    assert w.stepper.mutation_version == ver
    assert dict(w.stepper._cached_buffers) == buffers
