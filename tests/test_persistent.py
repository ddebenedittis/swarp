"""Persistent-buffer / CUDA-graph execution, slim 2D kernel, and GradRing."""

import pytest
import torch

from wmas import Environment, NavigationScenario
from wmas.dynamics.base import ControlMode, DynamicsModel

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda:0"] if CUDA else [])


def _mk(
    device,
    use_graph,
    *,
    n_agents=4,
    n_envs=16,
    auto_reset=False,
    n_obstacles=0,
    dtype=torch.float32,
):
    scen = NavigationScenario(
        n_agents=n_agents, world_size=1.0, n_obstacles=n_obstacles, neighbor_method="brute"
    )
    return Environment(
        scen,
        n_envs=n_envs,
        device=device,
        dt=0.05,
        substeps=1,
        seed=0,
        auto_reset=auto_reset,
        max_steps=5,
        use_graph=use_graph,
        dtype=dtype,
    )


def _run(env, n_steps, device, n_agents, dtype=torch.float32):
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(7)
    out = []
    with torch.no_grad():
        for _ in range(n_steps):
            a = torch.empty(env.n_envs, n_agents, 2, device=device, dtype=dtype).uniform_(
                -1, 1, generator=gen
            )
            o, r, d, _ = env.step(a)
            out.append((o.clone(), r.clone(), d.clone()))
    return out


# --------------------------------------------------------------- graph parity


@pytest.mark.skipif(not CUDA, reason="CUDA graph capture needs a GPU")
@pytest.mark.parametrize("auto_reset", [False, True])
@pytest.mark.parametrize("n_obstacles", [0, 3])
def test_graph_matches_eager(auto_reset, n_obstacles):
    dev = "cuda:0"
    g = _mk(dev, True, auto_reset=auto_reset, n_obstacles=n_obstacles)
    e = _mk(dev, False, auto_reset=auto_reset, n_obstacles=n_obstacles)
    gr, er = _run(g, 12, dev, 4), _run(e, 12, dev, 4)
    assert g.graph_mode  # graph captured lazily on the first step
    for (og, rg, dg), (oe, re, de) in zip(gr, er, strict=True):
        assert torch.equal(og, oe)
        assert torch.equal(rg, re)
        assert torch.equal(dg, de)


@pytest.mark.parametrize("device", DEVICES)
def test_persistent_cpu_matches_and_graph_flag(device):
    # On CPU use_graph falls back to eager persistent (graph_mode False) but still
    # produces the same trajectory as the non-persistent path.
    p = _mk(device, True)
    e = _mk(device, False)
    pr, er = _run(p, 8, device, 4), _run(e, 8, device, 4)
    if device == "cpu":
        assert not p.graph_mode
    for (op, rp, dp), (oe, re, de) in zip(pr, er, strict=True):
        assert torch.equal(op, oe) and torch.equal(rp, re) and torch.equal(dp, de)


@pytest.mark.parametrize("device", DEVICES)
def test_state_view_identity_stable(device):
    env = _mk(device, True)
    env.reset(seed=0)
    pos_obj = env.world.state.pos
    a = torch.zeros(env.n_envs, 4, 2, device=device)
    with torch.no_grad():
        for _ in range(3):
            env.step(a)
    # Persistent views are the same tensor objects across steps (no re-wrap).
    assert env.world.state.pos is pos_obj


@pytest.mark.skipif(not CUDA, reason="graph recapture counter is a CUDA-graph concept")
def test_obstacle_reset_no_recapture():
    dev = "cuda:0"
    env = _mk(dev, True, n_obstacles=3, auto_reset=True)
    env.reset(seed=0)
    a = torch.zeros(env.n_envs, 4, 2, device=dev)
    with torch.no_grad():
        env.step(a)  # triggers capture
    rt = env.world.runtime
    graph0, ver0 = rt._graph, rt._graph_version
    assert graph0 is not None
    with torch.no_grad():
        for _ in range(6):  # steps + auto-resets re-sample obstacles in place
            env.step(a)
        env.reset(seed=1)  # full reset re-samples obstacles in place
        env.step(a)
    # In-place obstacle updates keep the same obstacle count -> no recapture.
    assert env.world.runtime._graph is graph0
    assert env.world.runtime._graph_version == ver0


# --------------------------------------------------------------------- grad


@pytest.mark.parametrize("device", DEVICES)
def test_grad_step_in_persistent_env(device):
    # A grad step through a use_graph env must match the same step on a plain env
    # (persistent mode detaches+clones so the tape uses fresh arrays).
    from wmas.interop.autograd import warp_step

    def grads(use_graph):
        env = _mk(device, use_graph)
        env.reset(seed=0)
        st = env.world.state
        s = st._replace(
            pos=st.pos.detach().clone().requires_grad_(True),
            vel=st.vel.detach().clone().requires_grad_(True),
        )
        a = torch.zeros(env.n_envs, 4, 2, device=device, requires_grad=True)
        with torch.enable_grad():
            out = warp_step(env.world.stepper, s, a)
            out.pos.pow(2).sum().backward()
        return s.pos.grad.clone(), a.grad.clone()

    gp, ap = grads(True)
    ge, ae = grads(False)
    torch.testing.assert_close(gp, ge)
    torch.testing.assert_close(ap, ae)


# --------------------------------------------------------------------- slim


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "model,ctrl",
    [
        (DynamicsModel.HOLONOMIC, ControlMode.VELOCITY),
        (DynamicsModel.HOLONOMIC, ControlMode.ACCELERATION),
        (DynamicsModel.DIFF_DRIVE, ControlMode.VELOCITY),
        (DynamicsModel.DIFF_DRIVE, ControlMode.ACCELERATION),
        (DynamicsModel.KINEMATIC_BICYCLE, ControlMode.ACCELERATION),
    ],
)
def test_slim_matches_full(device, dtype, model, ctrl):
    def traj(slim):
        scen = NavigationScenario(
            n_agents=4, model=model, ctrl_mode=ctrl, world_size=1.0, neighbor_method="brute"
        )
        env = Environment(scen, n_envs=12, device=device, dt=0.05, seed=0, dtype=dtype)
        env.world.stepper.enable_slim2d = slim
        return _run(env, 8, device, 4, dtype=dtype)

    for (o1, r1, d1), (o0, r0, d0) in zip(traj(True), traj(False), strict=True):
        assert torch.equal(o1, o0)  # live fields (obs) bit-identical
        assert torch.equal(r1, r0)
        assert torch.equal(d1, d0)


def test_slim_leaves_drone_fields_zero():
    # A 2D fleet's drone state stays zero through slim no-grad steps.
    dev = DEVICES[-1]
    env = _mk(dev, False)
    env.reset(seed=0)
    a = torch.zeros(env.n_envs, 4, 2, device=dev)
    with torch.no_grad():
        for _ in range(4):
            env.step(a)
    st = env.world.state
    assert torch.count_nonzero(st.z) == 0
    assert torch.count_nonzero(st.vz) == 0
    assert torch.count_nonzero(st.body_rates) == 0


# ------------------------------------------------------------------ GradRing


@pytest.mark.parametrize("device", DEVICES)
def test_gradring_matches_no_ring_and_reuses(device):
    from wmas.interop.autograd import GradRing, rollout

    env = _mk(device, False)
    env.reset(seed=0)
    stepper = env.world.stepper
    st = env.world.state
    T = 6

    def run(ring):
        s = st._replace(
            pos=st.pos.detach().clone().requires_grad_(True),
            vel=st.vel.detach().clone().requires_grad_(True),
        )
        a = torch.zeros(T, env.n_envs, 4, 2, device=device, requires_grad=True)
        with torch.enable_grad():
            final, _ = rollout(stepper, s, a, ring=ring)
            final.pos.pow(2).sum().backward()
        return s.pos.grad.clone(), a.grad.clone()

    g0 = run(None)
    ring = GradRing(stepper, env.n_envs, env.world.act_dim, capacity=T)
    g1 = run(ring)
    g2 = run(ring)  # reuse across a second iteration, no new allocations
    for a, b in zip(g0, g1, strict=True):
        torch.testing.assert_close(a, b)
    for a, b in zip(g1, g2, strict=True):
        torch.testing.assert_close(a, b)


def test_gradring_capacity_guard():
    from wmas.interop.autograd import GradRing

    dev = DEVICES[-1]
    env = _mk(dev, False)
    ring = GradRing(env.world.stepper, env.n_envs, env.world.act_dim, capacity=2)
    ring.acquire()
    ring.acquire()
    with pytest.raises(RuntimeError, match="capacity"):
        ring.acquire()
