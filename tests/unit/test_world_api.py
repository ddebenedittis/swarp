"""``World`` helpers the scenarios build resets and fused launches on.

``write_state`` and ``state_wp`` exist to stop every scenario hand-rolling the same two
things: the masked ``torch.where`` reset blend (whose easy-to-forget ``mark_pos_dirty()``
is a silent stale-neighbor-list bug) and the ``_persistent``/``_detached`` dispatch that
decides whether the agent state is already a Warp array or has to be wrapped.
"""

import pytest
import torch
import warp as wp
from conftest import DEVICES, holo_cfgs

from swarp.core.config import ObstacleKind, Obstacles, WorldConfig
from swarp.core.world import AgentStateWp, World

FIELDS = ("pos", "theta", "vel", "speed", "ang_vel")


def _world(device, n_envs=4, n_agents=3, collisions=True):
    cfg = WorldConfig(collisions=collisions, collision_margin=0.02)
    return World(holo_cfgs(n_agents), cfg, n_envs=n_envs, device=device, dt=0.1)


# ------------------------------------------------------------------- write_state


@pytest.mark.parametrize("device", DEVICES)
def test_write_state_writes_every_env_without_a_mask(device):
    w = _world(device)
    pos = torch.full((4, 3, 2), 0.25, device=device)
    w.write_state(pos=pos, speed=torch.full((4, 3), 2.0, device=device))
    torch.testing.assert_close(w.state.pos, pos)
    torch.testing.assert_close(w.state.speed, torch.full((4, 3), 2.0, device=device))
    # untouched fields keep their values
    torch.testing.assert_close(w.state.vel, torch.zeros(4, 3, 2, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_write_state_blends_only_the_masked_envs(device):
    w = _world(device)
    w.write_state(pos=torch.full((4, 3, 2), -1.0, device=device))
    mask = torch.tensor([True, False, True, False], device=device)
    new = torch.full((4, 3, 2), 5.0, device=device)
    w.write_state(mask, pos=new, ang_vel=torch.full((4, 3), 3.0, device=device))
    got = w.state.pos
    torch.testing.assert_close(got[0], new[0])
    torch.testing.assert_close(got[2], new[2])
    torch.testing.assert_close(got[1], torch.full((3, 2), -1.0, device=device))
    torch.testing.assert_close(got[3], torch.full((3, 2), -1.0, device=device))
    # the mask is broadcast per field rank, not just for [n_envs, n_agents, 2]
    torch.testing.assert_close(w.state.ang_vel[:, 0], torch.tensor([3.0, 0.0, 3.0, 0.0],
                                                                  device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_write_state_writes_in_place(device):
    """Persistent buffers, their zero-copy views and any captured graph all alias the state
    tensors, so a reset has to ``copy_`` rather than rebind."""
    w = _world(device)
    objs = [getattr(w.state, f) for f in FIELDS]
    w.write_state(pos=torch.ones(4, 3, 2, device=device), vel=torch.ones(4, 3, 2, device=device))
    assert [getattr(w.state, f) for f in FIELDS] == objs


@pytest.mark.parametrize("device", DEVICES)
def test_write_state_broadcasts_scalars(device):
    w = _world(device)
    w.write_state(vel=torch.zeros(4, 3, 2, device=device) + 1.0, speed=0.0)
    torch.testing.assert_close(w.state.speed, torch.zeros(4, 3, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_write_state_marks_pos_dirty_only_when_pos_moved(device):
    """The whole point of routing resets through here: positions written out of band must
    invalidate the reusable neighbor list, and nothing else should pay for that."""
    w = _world(device)
    grid = w.stepper.grid(w.n_envs)

    w.neighbors()  # stamps grid.built_version with the current state_version
    assert grid.built_version == w.stepper.state_version
    w.write_state(vel=torch.ones(4, 3, 2, device=device))
    assert grid.built_version == w.stepper.state_version, "a velocity write invalidated the list"

    w.write_state(pos=torch.zeros(4, 3, 2, device=device))
    assert grid.built_version == -1


def test_write_state_without_collisions_is_a_noop_for_the_grid():
    """``mark_pos_dirty`` has no grid to invalidate when collisions are off."""
    w = _world("cpu", collisions=False)
    w.write_state(pos=torch.zeros(4, 3, 2))  # must not raise


# ---------------------------------------------------------------------- state_wp


@pytest.mark.parametrize("device", DEVICES)
def test_state_wp_wraps_the_torch_state_zero_copy(device):
    w = _world(device)
    w.write_state(pos=torch.full((4, 3, 2), 0.5, device=device))
    s = w.state_wp()
    assert isinstance(s, AgentStateWp) and len(s) == 5
    assert tuple(s._fields) == FIELDS
    for name, arr in zip(FIELDS, s, strict=True):
        assert arr.ptr == getattr(w.state, name).data_ptr(), name
    assert s.pos.shape == (4, 3)  # vec2 element type, not a trailing 2


@pytest.mark.parametrize("device", DEVICES)
def test_state_wp_returns_the_runtime_arrays_in_persistent_mode(device):
    """No re-wrap: the handles must be the runtime's own, pointer-stable arrays, or a
    fused kernel could not cache them and a CUDA graph could not bake them in."""
    w = _world(device)
    w.enable_persistent(use_graph=False)
    s = w.state_wp()
    rt = w.runtime.state
    for name in FIELDS:
        assert getattr(s, name) is getattr(rt, name), name
    # ...and stable across steps.
    with torch.no_grad():
        w.step(torch.zeros(4, 3, 2, device=device))
    assert w.state_wp().pos is rt.pos


@pytest.mark.parametrize("device", DEVICES)
def test_state_wp_falls_back_to_wrapping_after_a_grad_step(device):
    """A grad step in persistent mode detaches onto fresh tensors, so the runtime's arrays
    are no longer the live state and must not be handed out."""
    w = _world(device)
    w.enable_persistent(use_graph=False)
    actions = torch.zeros(4, 3, 2, device=device, requires_grad=True)
    w.step(actions)
    assert w._detached
    s = w.state_wp()
    assert s.pos is not w.runtime.state.pos
    assert s.pos.ptr == w.state.pos.data_ptr()


# ------------------------------------------------------------ obstacle_state_views


@pytest.mark.parametrize("device", DEVICES)
def test_obstacle_state_views_alias_the_engine_arrays_without_stable_identity(device):
    """Renamed from ``movable_obstacle_state``: it is not movable-only, and every call
    re-wraps, so a cached handle must key on ``data_ptr()`` rather than on ``is``."""
    w = _world(device, n_envs=2)
    n = 2
    w.set_obstacles(
        Obstacles(
            torch.zeros(2, n, 2, device=device),
            torch.full((n,), 0.1, device=device),
            kind=torch.full((n,), int(ObstacleKind.MOVABLE), device=device, dtype=torch.int32),
        )
    )
    a = w.obstacle_state_views()
    b = w.obstacle_state_views()
    assert all(x is not y for x, y in zip(a, b, strict=True))  # no stable identity
    assert all(x.data_ptr() == y.data_ptr() for x, y in zip(a, b, strict=True))
    assert a[0].data_ptr() == wp.to_torch(w.stepper.obs_pos).data_ptr()
    # body=True indexes the per-body state instead of the per-shape state.
    body = w.obstacle_state_views(body=True)
    assert body[0].data_ptr() == wp.to_torch(w.stepper.body_pos).data_ptr()
    assert body[0].data_ptr() != a[0].data_ptr()
