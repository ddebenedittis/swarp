"""The ``FusedScenario`` framework's own guarantees.

The per-scenario ``tests/scenarios/test_*_fused.py`` suites check that each scenario's fused
kernels agree with its torch reference. This file checks the layer *underneath* them: that
the declarative :class:`~swarp.scenarios.fused.Buf` spec does what it says, and that the
mistakes it exists to prevent are now errors rather than silent corruption.
"""

import pytest
import torch
import warp as wp
from conftest import DEVICES

from swarp.core.hooks import WholeStepHook
from swarp.scenarios import SCENARIOS
from swarp.scenarios.fused import STEP, Buf, FusedPass, FusedScenario

FUSED = list(SCENARIOS.values())
IDS = list(SCENARIOS)


# ------------------------------------------------------------------- FusedPass


def test_fused_pass_properties_are_named_after_the_kernel_arguments():
    step = FusedPass("step")
    assert step.is_step and step.advance_prev == 1 and step.full_pass == 1
    full_reset = FusedPass("reset")
    assert not full_reset.is_step and full_reset.advance_prev == 0
    assert full_reset.full_pass == 1  # a standalone reset still fills info()
    obs_only = FusedPass("reset", full=False)
    assert obs_only.advance_prev == 0 and obs_only.full_pass == 0


def test_step_pass_is_a_shared_constant():
    """The hot path must not allocate even a descriptor object per step."""
    assert STEP.kind == "step" and STEP.env_mask is None


# -------------------------------------------- the spec, across every scenario


@pytest.mark.parametrize("cls", FUSED, ids=IDS)
def test_every_registered_scenario_implements_the_same_member_set(cls):
    """Zero opt-outs: the point of the abstraction is that every scenario uses all of it."""
    assert issubclass(cls, FusedScenario)
    assert cls.fused_available is True
    for name in ("fused_spec", "launch_fused", "post_step_torch", "reset_torch"):
        assert name in vars(cls), f"{cls.__name__} does not implement {name}"
    # ...and none of them override what the framework owns.
    for name in ("post_step", "graph_hook", "ensure_fused", "prepare_fused", "finish_reset"):
        assert name not in vars(cls), f"{cls.__name__} overrides the framework's {name}"
    # ``supports_graph_reset``/``reset_in_graph`` are declared *extension points*
    # (default ``False`` / ``NotImplementedError``), not framework machinery: a
    # scenario opts its own reset into the captured whole-step graph by overriding
    # both together, so this only pins that they can't be split — overriding
    # ``reset_in_graph`` alone would leave it permanently unreachable dead code
    # (``supports_graph_reset`` still says ``False``), and overriding
    # ``supports_graph_reset`` alone would flip the flag onto the base class's
    # ``NotImplementedError`` body.
    assert ("supports_graph_reset" in vars(cls)) == ("reset_in_graph" in vars(cls)), (
        f"{cls.__name__} overrides only one of supports_graph_reset/reset_in_graph"
    )


@pytest.mark.parametrize("cls", FUSED, ids=IDS)
def test_spec_shapes_and_dtypes_match_the_allocated_buffers(cls):
    from swarp import Environment

    env = Environment(cls(n_agents=3), n_envs=5, device="cpu", dt=0.05, seed=0)
    scen = env.scenario
    env.reset(seed=0)  # triggers ensure_fused
    for b in scen.fused_spec(5):
        t = scen.fb[b.name]
        assert tuple(t.shape) == tuple(b.shape), f"{cls.__name__}.{b.name}"
        assert t.is_contiguous()
        if b.dtype == "uint8":
            assert t.dtype == torch.uint8
        elif b.dtype == "bool":
            assert t.dtype == torch.bool
        else:
            assert t.dtype == env.dtype
        if b.bool_view:
            view = scen.fb[b.name + "_bool"]
            assert view.dtype == torch.bool and view.data_ptr() == t.data_ptr()
        # Every declared buffer has a cached Warp handle over that exact memory.
        assert scen._wp[b.name].ptr == t.data_ptr()


@pytest.mark.parametrize("cls", FUSED, ids=IDS)
def test_adopted_state_is_the_scenario_s_own_tensor_not_a_copy(cls):
    """``alloc="never"`` must adopt, never allocate a shadow copy."""
    from swarp import Environment

    env = Environment(cls(n_agents=3), n_envs=5, device="cpu", dt=0.05, seed=0)
    scen = env.scenario
    env.reset(seed=0)
    adopted = [b for b in scen.fused_spec(5) if b.alloc != "always"]
    for b in adopted:
        owner, attr = scen._fused_owner(b)
        assert scen.fb[b.name] is getattr(owner, attr), f"{cls.__name__}.{b.name}"


@pytest.mark.parametrize("cls", FUSED, ids=IDS)
def test_graph_hook_is_wired_and_the_token_starts_at_zero(cls):
    from swarp import Environment

    env = Environment(cls(n_agents=3), n_envs=5, device="cpu", dt=0.05, seed=0)
    hook = env.scenario.graph_hook()
    assert isinstance(hook, WholeStepHook)
    env.reset(seed=0)
    assert env.scenario.fused_token() == 0
    # Carries are the *live* tensors, so a reassignment cannot leave a stale snapshot.
    for c in hook.carries():
        assert torch.is_tensor(c)


def test_scenarios_without_a_watched_buffer_pin_the_token_structurally():
    """Flocking/sampling/discovery have no watched buffer, so no edit can move their
    token — as opposed to a hand-written ``return 0`` that a later edit invalidates."""
    from swarp import Environment
    from swarp.scenarios import DiscoveryScenario, FlockingScenario, SamplingScenario

    for cls in (FlockingScenario, SamplingScenario, DiscoveryScenario):
        env = Environment(cls(n_agents=3), n_envs=4, device="cpu", dt=0.05, seed=0)
        env.reset(seed=0)
        assert env.scenario._fused_watch == ()
        with torch.no_grad():
            for _ in range(3):
                env.step(torch.zeros(4, 3, env.act_dim))
        assert env.scenario.fused_token() == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scenario_name", ["NavigationScenario", "FormationScenario"])
def test_masked_obs_only_reset_matches_a_full_recompute(device, scenario_name):
    """The obs-only auto-reset pass skips recomputing ``obs``/``prev`` for envs the
    reset mask does not select (see the ``full_pass == 0 and reset_mask[e] == 0``
    early-out in ``navigation_kernels.nav_obs_kernel`` / ``formation_kernels.
    formation_obs_kernel``, and ``NeighborGrid``'s masked brute-force build). That is
    an optimization, not an approximation: what it leaves behind for a non-reset env
    must be bit-identical to what a full, unmasked recompute over the same state would
    have written. This forces exactly that comparison — a real (partial) masked pass,
    then an unmasked one over the identical post-reset state — for both scenarios whose
    obs kernel carries both ``full_pass`` and ``reset_mask``.
    """
    from swarp import Environment
    from swarp.interop.autograd import torch_stream_scope
    from swarp.scenarios import FormationScenario, NavigationScenario

    n_envs, n_agents = 24, 5
    cls = {"NavigationScenario": NavigationScenario, "FormationScenario": FormationScenario}[
        scenario_name
    ]
    scen = cls(n_agents=n_agents)
    env = Environment(
        scen, n_envs=n_envs, device=device, dt=0.05, seed=0, max_steps=5, auto_reset=True
    )
    env.reset(seed=0)
    # Force exactly every other env to truncate on this step (step_count == max_steps
    # after the increment) so the auto_reset mask is guaranteed partial — neither all
    # envs nor none, which is the only shape that exercises the masked path at all.
    step_count = torch.zeros(n_envs, device=device, dtype=torch.int32)
    step_count[::2] = env.max_steps - 1
    env._step_count.copy_(step_count)
    act = torch.zeros(n_envs, n_agents, env.act_dim, device=device)
    with torch.no_grad():
        env.step(act)

    mask = scen.fb["resetmask"].clone()
    assert bool(mask.any()) and not bool(mask.all()), "test needs a partial reset"
    masked = {k: scen.fb[k].clone() for k in ("obs", "prev")}

    # Force a full, unmasked recompute of the same obs-only pass over the identical
    # current state (no further physics): mark every env "reset" for the kernels'
    # purposes and relaunch. This is the parity oracle for the masked pass above.
    scen.fb["resetmask"].fill_(1)
    with torch_stream_scope(env.world.device):
        scen.launch_fused(FusedPass("reset", full=False))

    for name, before in masked.items():
        assert torch.equal(before, scen.fb[name]), (
            f"{scenario_name}.fb[{name!r}] differs between the masked auto-reset pass "
            "and an unmasked recompute over the same state"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_reset_mask_is_zeroed_before_a_step_and_stamped_on_a_reset(device):
    from swarp import Environment, NavigationScenario

    env = Environment(NavigationScenario(n_agents=3), n_envs=4, device=device, dt=0.05, seed=0)
    scen = env.scenario
    env.reset(seed=0)
    assert (scen.fb["resetmask"] == 1).all()  # a full reset marks every env
    mask = torch.tensor([True, False, True, False], device=device)
    env.reset_at(mask)
    assert torch.equal(scen.fb["resetmask"].bool(), mask)
    with torch.no_grad():
        env.step(torch.zeros(4, 3, env.act_dim, device=device))
    assert (scen.fb["resetmask"] == 0).all()  # a normal step resets no env
    # ``prepare_fused`` stops zeroing once the buffer is *known* clean, so the mask has
    # to stay zero over the steps that follow, not only over the first one after a reset.
    assert not scen._fused_mask_dirty
    with torch.no_grad():
        for _ in range(3):
            env.step(torch.zeros(4, 3, env.act_dim, device=device))
            assert (scen.fb["resetmask"] == 0).all()


@pytest.mark.parametrize("device", DEVICES)
def test_every_reset_mask_write_marks_the_buffer_dirty(device):
    """The invariant the skipped memset in ``prepare_fused`` rests on.

    ``prepare_fused`` zeroes the reset mask only when ``_fused_mask_dirty`` says a writer
    touched it, so a writer that forgets to mark leaves its stamp standing for every
    later step. This pins the two host-side writers; the third is the in-graph
    ``episode_end_kernel``, which ``Environment._compose_graph_reset_hook`` covers by
    marking unconditionally before every replay.
    """
    from swarp import Environment, NavigationScenario

    env = Environment(NavigationScenario(n_agents=3), n_envs=4, device=device, dt=0.05, seed=0)
    scen = env.scenario
    mask = torch.tensor([True, False, True, False], device=device)
    for write in (
        lambda: env.reset(seed=0),  # finish_reset's fill_(1)
        lambda: env.reset_at(mask),  # reset_mask_wp's copy_
    ):
        scen._fused_mask_dirty = False
        write()
        assert scen._fused_mask_dirty, "a reset-mask write did not mark the buffer dirty"


# ------------------------------------------------------- the errors it prevents


class _Broken(FusedScenario):
    """A minimal harness: only ``ensure_fused`` is exercised, via a faked world."""

    def __init__(self, spec, world):
        self._spec, self.world = spec, world

    obs_dim = 0

    def fused_spec(self, n_envs):
        return self._spec

    def make_world(self, *a):
        raise NotImplementedError

    def reset_world(self, env_mask=None, *, obs_only=False):
        raise NotImplementedError

    def observations(self):
        raise NotImplementedError

    def launch_fused(self, pass_):
        raise NotImplementedError

    def post_step_torch(self):
        raise NotImplementedError

    def reset_torch(self, env_mask):
        raise NotImplementedError


class _FakeWorld:
    n_envs = 4
    device = "cpu"
    dtype = torch.float32
    wp_dtype = wp.float32


def _build(*spec, **attrs):
    scen = _Broken(spec, _FakeWorld())
    for k, v in attrs.items():
        setattr(scen, k, v)
    return scen


def test_non_contiguous_buffer_raises_instead_of_being_copied():
    """The copy would be a temporary whose freed device pointer gets captured."""
    strided = torch.zeros(4, 6)[:, ::2]  # a real non-contiguous view
    assert not strided.is_contiguous()
    scen = _build(Buf("x", (4, 3), attr="x", alloc="never"), x=strided)
    with pytest.raises(RuntimeError, match="not contiguous"):
        scen.ensure_fused()


def test_unallocated_adopted_state_raises_rather_than_allocating_a_second_copy():
    scen = _build(Buf("x", (4, 3), attr="x", alloc="never"), x=None)
    with pytest.raises(RuntimeError, match="allocated in make_world"):
        scen.ensure_fused()


def test_alloc_always_with_an_attr_is_rejected():
    scen = _build(Buf("x", (4, 3), attr="x"), x=torch.ones(4, 3))
    with pytest.raises(ValueError, match="alloc='always'"):
        scen.ensure_fused()


def test_watch_without_an_attr_is_rejected():
    scen = _build(Buf("x", (4, 3), watch=True))
    with pytest.raises(ValueError, match="watch=True without an attr"):
        scen.ensure_fused()


def test_two_reset_masks_are_rejected():
    scen = _build(
        Buf("a", (4,), "uint8", reset_mask=True), Buf("b", (4,), "uint8", reset_mask=True)
    )
    with pytest.raises(ValueError, match="reset_mask"):
        scen.ensure_fused()


def test_if_none_allocates_and_assigns_the_attribute():
    scen = _build(Buf("x", (4, 3), attr="x", alloc="if_none"), x=None)
    scen.ensure_fused()
    assert scen.x is scen.fb["x"] and scen.x.shape == (4, 3)
    assert torch.count_nonzero(scen.x) == 0


def test_a_moved_watched_handle_is_rebuilt_and_bumps_the_token():
    scen = _build(Buf("x", (4, 3), attr="x", alloc="if_none", watch=True), x=None)
    scen.ensure_fused()
    assert scen.fused_token() == 0
    scen.x = torch.ones(4, 3)  # the grad path reassigning a carry
    scen.sync_fused_handles()
    assert scen.fused_token() == 1
    assert scen._wp["x"].ptr == scen.x.data_ptr()
    scen.sync_fused_handles()  # steady state: a pointer compare, no bump
    assert scen.fused_token() == 1
