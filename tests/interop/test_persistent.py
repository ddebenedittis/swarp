"""Persistent-buffer / CUDA-graph execution, slim 2D kernel, and GradRing."""

import pytest
import torch
from conftest import DEVICES

from wmas import (
    DiscoveryScenario,
    Environment,
    FlockingScenario,
    FormationScenario,
    NavigationScenario,
    PushTScenario,
    SamplingScenario,
    TransportScenario,
)
from wmas.dynamics.base import ControlMode, DynamicsModel


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


@pytest.mark.gpu(reason="CUDA graph capture needs a GPU")
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


@pytest.mark.gpu(reason="CUDA graph capture needs a GPU")
@pytest.mark.parametrize(
    "scen_factory",
    [
        lambda n: NavigationScenario(n_agents=n),
        lambda n: FlockingScenario(n_agents=n),
        lambda n: FormationScenario(n_agents=n),
        lambda n: DiscoveryScenario(n_agents=n),
        lambda n: SamplingScenario(n_agents=n),
        lambda n: TransportScenario(n_agents=n),
        lambda n: PushTScenario(n_agents=n),
    ],
    ids=["navigation", "flocking", "formation", "discovery", "sampling", "transport", "pusht"],
)
def test_graph_matches_eager_all_scenarios(scen_factory):
    # Regression: the physics graph may only bake in neighbor reuse when the
    # scenario actually refreshes the grid each step; scenarios that don't
    # (sampling/formation) must have the graph build neighbors itself. All
    # scenarios must be bit-identical to the eager path.
    dev = "cuda:0"

    def traj(use_graph):
        env = Environment(
            scen_factory(8), n_envs=64, device=dev, dt=0.05, seed=0, use_graph=use_graph
        )
        env.reset(seed=0)
        gen = torch.Generator(device=dev).manual_seed(3)
        out = []
        with torch.no_grad():
            for _ in range(8):
                a = torch.empty(64, 8, 2, device=dev).uniform_(-1, 1, generator=gen)
                o, *_ = env.step(a)
                out.append(o.clone())
        return out, env

    eager, _ = traj(False)
    graphed, genv = traj(True)
    assert genv.graph_mode
    # Every fused scenario opts into the whole-step graph (obs/reward folded in).
    assert genv._whole_step
    assert genv.world.ran_post_physics
    for oe, og in zip(eager, graphed, strict=True):
        assert torch.equal(oe, og)


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


@pytest.mark.gpu(reason="graph recapture counter is a CUDA-graph concept")
def test_obstacle_reset_no_recapture():
    from wmas.dynamics.base import P_MASS, P_RADIUS, per_env_float_template

    dev = "cuda:0"
    env = _mk(dev, True, n_obstacles=3, auto_reset=True)
    assert env._whole_step  # whole-step wiring active for this regression
    env.reset(seed=0)
    a = torch.zeros(env.n_envs, 4, 2, device=dev)
    stepper = env.world.stepper
    # Install per-env DR once (this first install bumps mutation_version) BEFORE
    # capture, as a float32 on-device buffer so later updates take the in-place path.
    tmpl = torch.as_tensor(
        per_env_float_template(env.world.agent_configs, env.n_envs), dtype=torch.float32, device=dev
    )
    # Keep the per-env radius comfortably below neighbor_radius (float32 rounding
    # of 0.05 lands exactly on the reach bound); DR here only exercises recapture.
    tmpl[..., P_RADIUS] = 0.04
    stepper.set_agent_params_per_env(tmpl)
    with torch.no_grad():
        env.step(a)  # triggers capture (with per-env DR active)
    rt = env.world.runtime
    graph0, ver0 = rt._graph, rt._graph_version
    assert graph0 is not None
    with torch.no_grad():
        for i in range(2):  # in-place per-env DR re-randomization
            tmpl[..., P_MASS] = 1.0 + 0.1 * i
            stepper.set_agent_params_per_env(tmpl)
            env.step(a)
        for _ in range(6):  # steps + auto-resets re-sample obstacles in place
            env.step(a)
        env.reset(seed=1)  # full reset re-samples obstacles in place
        env.step(a)
    # In-place obstacle/DR updates keep the same obstacle count and buffer shapes
    # and don't touch the fused obs handles -> no recapture.
    assert env.world.runtime._graph is graph0
    assert env.world.runtime._graph_version == ver0


# --------------------------------------------------- whole-step graph (obs/reward)


def _run_info(env, n_steps, device, n_agents):
    """Trajectory including the info dict (for whole-step parity)."""
    env.reset(seed=0)
    gen = torch.Generator(device=device).manual_seed(7)
    out = []
    with torch.no_grad():
        for _ in range(n_steps):
            a = torch.empty(env.n_envs, n_agents, 2, device=device).uniform_(
                -1, 1, generator=gen
            )
            o, r, d, info = env.step(a)
            info_c = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in info.items()}
            out.append((o.clone(), r.clone(), d.clone(), info_c))
    return out


@pytest.mark.gpu(reason="CUDA graph capture needs a GPU")
@pytest.mark.parametrize("auto_reset", [False, True])
@pytest.mark.parametrize("n_obstacles", [0, 3])
def test_whole_step_matches_fused_eager(auto_reset, n_obstacles):
    # Whole-step graph (use_graph=True) vs fused-eager (use_graph=False): obs,
    # reward, done, and every info tensor bit-identical; the graph must really
    # fold obs/reward in (catch a silent regression to a physics-only graph).
    dev = "cuda:0"
    g = _mk(dev, True, auto_reset=auto_reset, n_obstacles=n_obstacles)
    e = _mk(dev, False, auto_reset=auto_reset, n_obstacles=n_obstacles)
    gr, er = _run_info(g, 12, dev, 4), _run_info(e, 12, dev, 4)
    assert g.graph_mode
    assert g._whole_step
    assert g.world.ran_post_physics
    for (og, rg, dg, ig), (oe, re, de, ie) in zip(gr, er, strict=True):
        assert torch.equal(og, oe)
        assert torch.equal(rg, re)
        assert torch.equal(dg, de)
        assert ig.keys() == ie.keys()
        for k in ig:
            if torch.is_tensor(ig[k]):
                assert torch.equal(ig[k], ie[k]), k


@pytest.mark.gpu(reason="CUDA graph capture needs a GPU")
def test_no_double_post_step():
    # The whole-step graph fills the obs/reward buffers, so Environment must skip
    # the redundant torch post_step on a no-grad step; a grad step (torch path)
    # still runs it.
    dev = "cuda:0"
    env = _mk(dev, True)
    env.reset(seed=0)
    calls = {"n": 0}
    orig = env.scenario.post_step

    def counting():
        calls["n"] += 1
        return orig()

    env.scenario.post_step = counting

    a = torch.zeros(env.n_envs, 4, 2, device=dev)
    with torch.no_grad():
        env.step(a)
    assert calls["n"] == 0  # graph ran obs/reward; post_step skipped

    ag = torch.zeros(env.n_envs, 4, 2, device=dev, requires_grad=True)
    with torch.enable_grad():
        o, r, *_ = env.step(ag)
        r.sum().backward()
    assert calls["n"] == 1  # grad step took the torch path -> post_step ran


@pytest.mark.gpu(reason="CUDA graph capture needs a GPU")
@pytest.mark.parametrize("eager_trims", [True, False])
def test_prev_dist_reassignment_recapture(eager_trims):
    # A grad step refreshes the shaping baseline: eager_trims=True writes it in
    # place (handle stable -> no recapture); eager_trims=False reassigns it
    # (handle moves -> recapture). Either way the resumed no-grad trajectory stays
    # bit-identical to the fused-eager reference.
    dev = "cuda:0"

    def mk(use_graph):
        scen = NavigationScenario(n_agents=4, neighbor_method="brute")
        scen.eager_trims = eager_trims  # an ablation knob, set directly (not a ctor kwarg)
        return Environment(scen, n_envs=16, device=dev, dt=0.05, seed=0, max_steps=50,
                           use_graph=use_graph)

    g, e = mk(True), mk(False)
    for env in (g, e):
        env.reset(seed=0)

    def one_grad_step(env):
        ag = torch.zeros(env.n_envs, 4, 2, device=dev, requires_grad=True)
        with torch.enable_grad():
            _, r, *_ = env.step(ag)
            r.sum().backward()

    a0 = torch.zeros(16, 4, 2, device=dev)
    gen = torch.Generator(device=dev).manual_seed(7)
    with torch.no_grad():
        for _ in range(3):  # capture
            g.step(a0)
            e.step(a0)
    rt = g.world.runtime
    graph0, tok0 = rt._graph, g.scenario.fused_token()
    one_grad_step(g)  # reassigns/refreshes _prev_dist on the torch path
    one_grad_step(e)
    with torch.no_grad():
        for _ in range(4):  # resume no-grad; recapture happens lazily on the first
            a = torch.empty(16, 4, 2, device=dev).uniform_(-1, 1, generator=gen)
            og, rg, dg, _ = g.step(a)
            oe, re, de, _ = e.step(a)
            assert torch.equal(og, oe)
            assert torch.equal(rg, re)
            assert torch.equal(dg, de)
    if eager_trims:
        assert g.scenario.fused_token() == tok0
        assert g.world.runtime._graph is graph0
    else:
        assert g.scenario.fused_token() > tok0
        assert g.world.runtime._graph is not graph0


@pytest.mark.parametrize("device", DEVICES)
def test_cpu_whole_step_eager(device):
    # On CPU (or any graph-ineligible backend) the whole-step hook runs eagerly in
    # _run_eager: graph_mode False, but ran_post_physics True and bit-identical.
    g = _mk(device, True)
    e = _mk(device, False)
    gr, er = _run(g, 8, device, 4), _run(e, 8, device, 4)
    assert g._whole_step
    assert g.world.ran_post_physics
    if device == "cpu":
        assert not g.graph_mode
    for (og, rg, dg), (oe, re, de) in zip(gr, er, strict=True):
        assert torch.equal(og, oe) and torch.equal(rg, re) and torch.equal(dg, de)


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
