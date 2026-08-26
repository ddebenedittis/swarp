"""``FusedScenario``: the fused / CUDA-graph bookkeeping, owned once.

A scenario with fused Warp obs/reward kernels needs the same six things, and before
this module each of the seven built-ins hand-wrote all six (~200 lines apiece, plus a
seventh hand-copied out of tree into ``render/demo.py``):

1. lazy allocation of the persistent output buffers, once, at the real ``n_envs``;
2. zero-copy ``uint8 -> bool`` reinterpret views, because a Warp kernel writes ``0/1``
   bytes but ``done`` has to come back as a torch ``bool``;
3. cached ``wp.array`` handles, so no launch re-wraps a tensor per step;
4. a pointer-move resync, because the *grad* path reassigns tensors the fused path had
   already wrapped, plus a monotonic token so the captured graph is recaptured when it
   happens;
5. the warm-up carry list — every buffer the hook advances **in place**, snapshotted
   around graph warm-up so warm-up does not advance the simulation;
6. the reset-mask stamp that tells the kernels which envs just respawned.

None of that is scenario-specific. What *is* scenario-specific is the list of buffers
and the launch sequence, so those are what a subclass declares: a tuple of :class:`Buf`
from :meth:`FusedScenario.fused_spec`, and :meth:`FusedScenario.launch_fused`.

Deliberately **not** absorbed here: the ``_launch_*`` wrappers (bespoke ``wp.launch``
signatures; a launch DSL would be a worse ``wp.launch`` with none of Warp's type
checking), :meth:`~swarp.scenarios.base.Scenario.info` (the keys differ per scenario and
half the entries are not buffers), and the torch reference paths — those are the
**parity oracle** the fused kernels are tested against, so sharing code with them would
weaken the exact property the seven parity suites exist to test.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import warp as wp

from swarp.core.hooks import WholeStepHook
from swarp.core.state import VEC2
from swarp.scenarios.base import Scenario

#: How a :class:`Buf` is seen from torch and from Warp, in one field.
#:
#: * ``"float"`` — the world's float dtype, a scalar Warp array;
#: * ``"vec2"``  — the world's float dtype with a trailing ``2``, a Warp ``vec2`` array;
#: * ``"uint8"`` — ``torch.uint8`` bytes, a Warp ``uint8`` array (kernel-written flags);
#: * ``"bool"``  — ``torch.bool`` storage reinterpreted as Warp ``uint8`` (an adopted
#:   torch bool latch, e.g. a coverage mask, that a kernel also writes).
BufDtype = Literal["float", "vec2", "uint8", "bool"]

#: Whether the framework allocates a buffer or adopts one that already exists.
#:
#: * ``"always"``  — framework-owned; allocated here, reachable as ``self.fb[name]``.
#: * ``"if_none"`` — ``attr`` may still be ``None``; allocate and assign if so. For
#:   carries whose *first* value has to be derived from the state rather than be zero,
#:   so the torch reference path owns the initialization.
#: * ``"never"``   — adopt what ``attr`` already holds; a ``None`` is a hard error.
#:   This is the only mode a dotted ``attr`` accepts, i.e. state the scenario reads but
#:   does not own (``"world.goals"``).
BufAlloc = Literal["always", "if_none", "never"]


@dataclass(frozen=True)
class Buf:
    """One buffer in a scenario's fused spec.

    Attributes:
        name: key under which the torch tensor lands in ``self.fb`` and the Warp handle
            in ``self._wp``. Kernel launches read ``self._wp[name]``.
        shape: full torch shape, ``n_envs`` included (``fused_spec`` is handed it).
        dtype: see :data:`BufDtype`.
        attr: attribute holding the tensor, for adopted state — plain
            (``"_prev_dist"``) or dotted and read-only (``"world.goals"``). ``None``
            means the framework owns the buffer outright.
        alloc: see :data:`BufAlloc`.
        carry: the fused launches advance this **in place**, so graph warm-up must
            snapshot and restore it. Getting this wrong does not fail loudly — it
            advances the simulation one extra step at capture — which is exactly why
            it is declared rather than remembered.
        watch: the tensor may be *reassigned* (the grad path builds fresh tensors for
            the tape), so its ``data_ptr`` is compared each ``prepare`` and the handle
            rebuilt + the recapture token bumped when it moves. Only meaningful with
            ``attr``: a framework-owned buffer cannot move.
        bool_view: also expose ``fb[name + "_bool"]``, a zero-copy ``bool`` reinterpret
            of ``uint8`` bytes. What lets ``done()`` return a real bool tensor.
        reset_mask: this is *the* per-env "just reset" mask — zeroed by
            :meth:`FusedScenario.prepare_fused` before a normal step and stamped by
            :meth:`FusedScenario.finish_reset`. At most one per spec.
    """

    name: str
    shape: tuple[int, ...]
    dtype: BufDtype = "float"
    attr: str | None = None
    alloc: BufAlloc = "always"
    carry: bool = False
    watch: bool = False
    bool_view: bool = False
    reset_mask: bool = False


@dataclass(frozen=True, eq=False)
class FusedPass:
    """Which fused pass to launch — a step, or one of the two flavours of reset.

    The three kernel flags the seven scenarios actually need are all derivable from
    ``(kind, full)``, but *where* each puts them differs: navigation and formation carry
    them on the obs kernel, transport and pusht on the reward kernel, flocking and
    sampling only take ``full_pass``, and discovery takes neither — it reads ``full_pass``
    in ``launch_fused`` itself to decide whether the reward kernel runs. So this describes
    the pass and :meth:`FusedScenario.launch_fused` maps it onto the scenario's own
    sequence. The properties are named after the kernel arguments they feed, so that
    mapping reads as a rename rather than a translation.

    Attributes:
        kind: ``"step"`` (post-physics) or ``"reset"``.
        env_mask: which envs the reset touched (``None`` = all). Informational for a
            ``launch_fused`` that needs it; the mask *buffer* is stamped by the
            framework.
        full: recompute everything (``True``) or observations only (``False``). A
            mid-step auto-reset is obs-only: reward/done/info for the transition just
            taken have already been returned and their buffers must survive.
    """

    kind: Literal["step", "reset"]
    env_mask: torch.Tensor | None = None
    full: bool = True

    @property
    def is_step(self) -> bool:
        return self.kind == "step"

    @property
    def advance_prev(self) -> int:
        """``advance_prev``: a step advances a shaping baseline, a reset rebases it."""
        return 1 if self.kind == "step" else 0

    @property
    def full_pass(self) -> int:
        """``full_pass``: write the reward/info outputs too, or observations only."""
        return 1 if self.full else 0


#: The one pass on the hot path. A module constant so a step allocates nothing at all.
STEP = FusedPass("step")


class FusedScenario(Scenario):
    """A :class:`~swarp.scenarios.base.Scenario` with fused Warp obs/reward kernels.

    A concrete subclass rather than a mixin, so the MRO of every built-in scenario stays
    ``Scenario -> FusedScenario -> Concrete`` and there is no cooperative-``super()``
    ordering to reason about.

    Subclasses implement four members and inherit everything else:

    * :meth:`fused_spec` — the buffers, declaratively;
    * :meth:`launch_fused` — the ``wp.launch`` sequence for a :class:`FusedPass`;
    * :meth:`post_step_torch` / :meth:`reset_torch` — the torch reference path, which
      is the parity oracle and stays entirely the scenario's own.

    and optionally :meth:`engine_carries` when the fused launches also advance state
    that lives in the engine rather than in the spec.
    """

    fused_available = True

    # ------------------------------------------------------- the four declarations

    @abstractmethod
    def fused_spec(self, n_envs: int) -> Sequence[Buf]:
        """The persistent buffers this scenario's fused kernels read and write.

        Called once, lazily, on the first fused pass — so ``n_envs`` and every
        ``__init__`` parameter are known and shapes can be written out literally.
        """

    @abstractmethod
    def launch_fused(self, pass_: FusedPass) -> None:
        """Launch this scenario's fused kernels for ``pass_``.

        **Capture-safety contract.** On a ``kind="step"`` pass this runs inside
        ``wp.ScopedCapture``, so every line must be ``wp.launch`` / a neighbor-grid
        build against the pointer-stable handles in ``self._wp``, with **no device
        allocation and no host read**. Torch calls are legal only where they provably
        allocate nothing and sync nothing — transport's in-capture
        ``world.set_obstacles`` of a retained spec qualifies (see
        :meth:`swarp.core.stepper.Stepper.set_obstacles`, and the
        ``tests/unit/test_obstacles.py`` case that pins it), an ``.item()`` or a
        ``torch.zeros`` does not. Anything that cannot honour that belongs in
        :meth:`prepare_fused`'s territory, i.e. in the spec.
        """

    @abstractmethod
    def post_step_torch(self) -> None:
        """Refresh the torch reference cache after a physics step (non-fused path)."""

    @abstractmethod
    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        """Refresh the torch reference cache after a reset (non-fused path)."""

    def engine_carries(self) -> list[torch.Tensor]:
        """Extra warm-up carries that live in the engine rather than in the spec.

        Scenarios whose fused pass advances *engine* state — a movable obstacle's pose,
        a compound body's state — list it here, via
        :meth:`swarp.core.world.World.obstacle_state_views`. Empty by default.
        """
        return []

    # ------------------------------------------------------------- framework state

    #: name -> torch tensor for every buffer in the spec (plus ``<name>_bool`` views).
    fb: dict[str, torch.Tensor]
    #: name -> the cached ``wp.array`` handle the launches pass to the kernels.
    _wp: dict[str, wp.array]

    _fused_ready: bool = False

    # ------------------------------------------------------------------ allocation

    def ensure_fused(self, n_envs: int | None = None) -> None:
        """Allocate/adopt the spec's buffers and cache their Warp handles. Idempotent."""
        if self._fused_ready:
            return
        w = self.world
        n_envs = w.n_envs if n_envs is None else n_envs
        spec = tuple(self.fused_spec(n_envs))

        masks = [b for b in spec if b.reset_mask]
        if len(masks) > 1:
            raise ValueError(
                f"{type(self).__name__}.fused_spec declares {len(masks)} reset_mask "
                f"buffers ({[b.name for b in masks]}); there can be at most one"
            )
        self.fb = {}
        self._wp = {}
        self._fused_ptrs: dict[str, int] = {}
        for b in spec:
            if b.watch and b.attr is None:
                raise ValueError(
                    f"fused buffer {b.name!r} sets watch=True without an attr; a "
                    "framework-owned buffer is never reassigned, so nothing can move"
                )
            self._fused_bind(b, self._fused_acquire(b, n_envs))
        self._fused_carries = tuple(b for b in spec if b.carry)
        self._fused_watch = tuple(b for b in spec if b.watch)
        self._fused_mask = masks[0].name if masks else None
        self._fused_token_value = 0
        self._fused_ready = True

    def _fused_owner(self, b: Buf) -> tuple[object, str]:
        """Resolve ``b.attr`` to ``(owner, attribute name)``; ``"world.goals"`` is dotted."""
        obj: object = self
        parts = b.attr.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        return obj, parts[-1]

    def _fused_acquire(self, b: Buf, n_envs: int) -> torch.Tensor:
        """The tensor backing ``b``: freshly zeroed, or the one ``b.attr`` already holds."""
        w = self.world
        dtype = {"float": w.dtype, "vec2": w.dtype, "uint8": torch.uint8, "bool": torch.bool}[
            b.dtype
        ]
        if b.alloc == "always":
            if b.attr is not None:
                raise ValueError(
                    f"fused buffer {b.name!r} names attr {b.attr!r} but alloc='always' "
                    "would overwrite it; use 'never' to adopt or 'if_none' to fill in"
                )
            return torch.zeros(b.shape, device=w.device, dtype=dtype)

        if b.attr is None:
            raise ValueError(f"fused buffer {b.name!r} needs an attr for alloc={b.alloc!r}")
        owner, attr = self._fused_owner(b)
        current = getattr(owner, attr, None)
        if current is not None:
            return current
        if b.alloc == "never":
            raise RuntimeError(
                f"fused buffer {b.name!r} adopts {b.attr!r}, which is None. Adopted state "
                "must be allocated in make_world (n_envs is known there) — allocating it "
                "lazily in reset_world makes the fused path depend on reset order."
            )
        if "." in b.attr:
            raise ValueError(
                f"fused buffer {b.name!r} uses a dotted attr {b.attr!r}, which is read-only; "
                "alloc='if_none' would have to assign to it. Allocate it in make_world."
            )
        current = torch.zeros(b.shape, device=w.device, dtype=dtype)
        setattr(owner, attr, current)
        return current

    def _fused_bind(self, b: Buf, t: torch.Tensor) -> None:
        """Record ``t`` as ``b``'s backing tensor and (re)build its Warp handle."""
        if not t.is_contiguous():
            # Never silently ``.contiguous()``: that copy is a temporary whose device
            # pointer would be wrapped, baked into the captured graph, and freed the
            # moment this function returns — a use-after-free that only shows up as
            # wrong numbers on replay. Fix the producer instead.
            raise RuntimeError(
                f"fused buffer {b.name!r} is not contiguous. A fused buffer must be "
                "contiguous at the source: the copy that would fix it here is a "
                "temporary whose freed pointer gets captured into the CUDA graph."
            )
        w = self.world
        self.fb[b.name] = t
        self._fused_ptrs[b.name] = t.data_ptr()
        if b.dtype == "vec2":
            self._wp[b.name] = wp.from_torch(t, dtype=VEC2[w.wp_dtype])
        elif b.dtype == "uint8":
            self._wp[b.name] = wp.from_torch(t, dtype=wp.uint8)
        elif b.dtype == "bool":
            self._wp[b.name] = wp.from_torch(t.view(torch.uint8), dtype=wp.uint8)
        else:
            self._wp[b.name] = wp.from_torch(t, dtype=w.wp_dtype)
        if b.bool_view:
            self.fb[b.name + "_bool"] = t.view(torch.bool)

    # ------------------------------------------------------------- handle tracking

    def sync_fused_handles(self) -> None:
        """Rebuild any watched handle whose tensor was reassigned; bump the token if so.

        A pointer compare per watched buffer in steady state. Runs outside capture (from
        :meth:`prepare_fused`), because a rebuilt handle means the captured graph holds a
        dead pointer and has to be thrown away — which is what the token tells the runtime.
        """
        changed = False
        for b in self._fused_watch:
            owner, attr = self._fused_owner(b)
            t = getattr(owner, attr)
            if t.data_ptr() != self._fused_ptrs[b.name]:
                self._fused_bind(b, t)
                changed = True
        if changed:
            self._fused_token_value += 1

    def fused_token(self) -> int:
        """The recapture token: bumped whenever a cached handle is rebuilt.

        A scenario with no watched buffers keeps this at ``0`` *structurally* — the loop
        in :meth:`sync_fused_handles` has nothing to iterate — rather than by returning a
        hand-written constant that a later edit could invalidate.
        """
        return self._fused_token_value if self._fused_ready else 0

    def fused_carries(self) -> list[torch.Tensor]:
        """Every buffer the fused pass advances in place (spec carries + engine state)."""
        self.ensure_fused()
        out = []
        for b in self._fused_carries:
            if b.attr is None:
                out.append(self.fb[b.name])
            else:
                owner, attr = self._fused_owner(b)
                out.append(getattr(owner, attr))
        out.extend(self.engine_carries())
        return out

    # -------------------------------------------------------------- the two passes

    def prepare_fused(self) -> None:
        """Capture-unsafe prep for a fused step: ensure, clear the reset mask, resync.

        Run eagerly on the default stream before the physics step — by the Environment
        when a graph backs the step, otherwise by :meth:`post_step` itself.
        """
        self.ensure_fused()
        if self._fused_mask is not None:
            self.fb[self._fused_mask].zero_()  # a normal step resets no env
        self.sync_fused_handles()

    def post_step(self) -> None:
        """Fused launches (or the torch reference refresh) right after the physics step.

        The fused arm runs the *same* sequence the whole-step graph captures, which is
        what keeps non-graph fused mode and CPU eager-persistent mode bit-identical to
        graph mode.
        """
        if self.fused_active:
            self.prepare_fused()
            self.launch_fused(STEP)
        else:
            self.post_step_torch()

    def finish_reset(self, env_mask: torch.Tensor | None, *, obs_only: bool) -> None:
        """Close out ``reset_world``: the fused reset pass, or the torch refresh.

        Every ``reset_world`` ends with this call, having already written the new state.
        """
        if self.fused_active:
            self.ensure_fused()
            self.sync_fused_handles()  # this pass launches with the cached handles
            if self._fused_mask is not None:
                mask = self.fb[self._fused_mask]
                if env_mask is None:
                    mask.fill_(1)
                else:
                    mask.copy_(env_mask)  # bool -> uint8
            self.launch_fused(FusedPass("reset", env_mask=env_mask, full=not obs_only))
        else:
            self.reset_torch(env_mask)

    # --------------------------------------------------------------- graph wiring

    def graph_hook(self) -> WholeStepHook:
        """The whole-step hook, assembled from the spec. Nothing to override."""
        return WholeStepHook(
            run=self._launch_fused_step,
            prepare=self.prepare_fused,
            token=self.fused_token,
            carries=self.fused_carries,
        )

    def _launch_fused_step(self) -> None:
        self.launch_fused(STEP)
