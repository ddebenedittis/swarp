"""World-level configuration and the obstacle-set description."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from enum import IntEnum

import torch

from swarp.dynamics.base import Integrator


class ObstacleShape(IntEnum):
    """Obstacle geometry tags.

    :mod:`swarp.core.collisions` derives its kernel-side ``SHAPE_*`` constants from these
    members, so the two cannot drift.
    """

    CIRCLE = 0
    BOX = 1
    SEGMENT = 2


class ObstacleKind(IntEnum):
    """Whether an obstacle can be pushed around.

    ``IMMOVABLE`` is infinite-mass scenery: agents bounce off it and it never moves.
    ``MOVABLE`` obstacles carry a mass and inertia and are integrated from the reaction
    of the very same agent contacts (Newton's third law), inside the substep loop — so
    agents see the body's up-to-date pose rather than last step's. Renderers draw
    immovable obstacles black and movable ones grey.
    """

    IMMOVABLE = 0
    MOVABLE = 1


@dataclass
class Obstacles:
    """One batched obstacle set — the complete input to ``set_obstacles``.

    ``pos`` and ``radius`` are required and come first; every other field is optional and
    ``None`` means **the documented default**, never "keep whatever was installed before".
    A field is resolved exactly once, by :meth:`resolve`: detached, made contiguous, moved
    to the target device/precision, and (for ``angle``) broadcast to its per-env layout.

    Fields (``E`` = ``n_envs``, ``N`` = ``n_obstacles``):

    ``pos`` ``[E, N, 2]``
        Shape centres, per env.
    ``radius`` ``[N]``
        Circle radius / capsule radius. A ``BOX`` **ignores** it: its surface is its
        boundary, so ``torch.zeros(N)`` is the idiomatic value there.
    ``shape`` ``[N]`` int, default ``CIRCLE``
        :class:`ObstacleShape` tag per obstacle.
    ``angle`` ``[N]`` or ``[E, N]``, default 0
        Orientation (rad) of a box/segment. A 1-D tensor is shared by every env; a 2-D one
        lets a body rotate independently per env. Stored per-env either way.
    ``half_extents`` ``[N, 2]``, default 0
        Box half-extents; for a segment ``[:, 0]`` is the half-length.
    ``vel`` ``[E, N, 2]`` / ``ang_vel`` ``[E, N]``, default 0
        The obstacles' own velocities. Contact damping uses the *closing* velocity, so a
        moving obstacle that omits these is damped against the agent's absolute velocity.
        For a movable obstacle they are its initial velocity.
    ``kind`` ``[N]`` int, default ``IMMOVABLE``
        :class:`ObstacleKind` tag. A ``MOVABLE`` obstacle is integrated inside the substep
        loop from the reaction of the agent contacts, so ``pos``/``angle``/``vel``/
        ``ang_vel`` are its *initial state* rather than a fixed pose.
    ``mass`` / ``inertia`` ``[N]``, default 1.0
        Read only for movable obstacles; taken from the body root and about the body origin.
    ``body`` ``[N]`` int, default "each shape its own body"
        Groups shapes into *compound* rigid bodies sharing one pose (Push-T's T is a
        crossbar plus a stem). Each entry names the body's root, its lowest-index shape.
    ``body_offset`` ``[N, 2]``, default 0
        Placement of each shape in its body frame. ``pos``/``angle`` are still the shapes'
        *world* poses; the body origin is derived back out from the root.

    Treat an instance as immutable apart from writing *into* its tensors: :meth:`resolve`
    and :attr:`any_movable` memoize, and a reassigned field would leave them stale.
    Mutating the tensors in place and re-installing is the supported partial update, and
    what :class:`~swarp.core.world.World` does for the interactive obstacle drag.
    """

    pos: torch.Tensor
    radius: torch.Tensor
    shape: torch.Tensor | None = None
    angle: torch.Tensor | None = None
    half_extents: torch.Tensor | None = None
    vel: torch.Tensor | None = None
    ang_vel: torch.Tensor | None = None
    kind: torch.Tensor | None = None
    mass: torch.Tensor | None = None
    inertia: torch.Tensor | None = None
    body: torch.Tensor | None = None
    body_offset: torch.Tensor | None = None
    # (device, dtype) this instance is already normalized for; None until resolved.
    _resolved_for: tuple | None = field(default=None, init=False, repr=False, compare=False)
    # Memoized ``any_movable``: reading it costs a device->host sync, so it is computed at
    # most once per instance and skipped entirely when ``kind`` is None.
    _any_movable: bool | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.pos.dim() != 3 or self.pos.shape[-1] != 2:
            raise ValueError(f"obstacle pos must be [n_envs, n_obstacles, 2]; got {self.pos.shape}")
        n = self.pos.shape[1]
        if self.radius.dim() != 1 or self.radius.shape[0] != n:
            raise ValueError(
                f"obstacle radius must be [n_obstacles={n}]; got {tuple(self.radius.shape)}"
            )
        if self.angle is not None and self.angle.dim() not in (1, 2):
            raise ValueError(
                "obstacle angle must be [n_obstacles] or [n_envs, n_obstacles]; got "
                f"{tuple(self.angle.shape)}"
            )

    @property
    def n_envs(self) -> int:
        return self.pos.shape[0]

    @property
    def n_obstacles(self) -> int:
        return self.pos.shape[1]

    @property
    def any_movable(self) -> bool:
        """Whether any obstacle is ``MOVABLE`` (i.e. whether the engine integrates a body).

        Reading this **syncs the device to the host** the first time (a reduction over
        ``kind`` plus an ``.item()``), so it is memoized, and returns ``False`` without
        touching the device when ``kind`` is ``None``. That is what keeps a ``kind``-free
        install legal inside a CUDA-graph capture.
        """
        if self._any_movable is None:
            self._any_movable = self.kind is not None and bool((self.kind != 0).any().item())
        return self._any_movable

    def resolve(self, device, dtype: torch.dtype) -> Obstacles:
        """This set, normalized for ``device``/``dtype``. Idempotent and memoized.

        Every present field is detached, cast, moved to ``device`` and made contiguous, and
        ``angle`` is broadcast to its per-env ``[n_envs, n_obstacles]`` layout; absent
        fields stay ``None`` so the installer can fill their defaults without allocating.
        Returns ``self`` when it is already normalized for this target, so re-installing a
        retained set costs nothing and allocates nothing.
        """
        key = (str(device), dtype)
        if self._resolved_for == key:
            return self

        def f(t):  # float field
            return None if t is None else t.detach().to(device=device, dtype=dtype).contiguous()

        def i(t):  # int32 field
            if t is None:
                return None
            return t.detach().to(device=device, dtype=torch.int32).contiguous()

        angle = f(self.angle)
        if angle is not None and angle.dim() == 1:
            angle = angle.unsqueeze(0).expand(self.n_envs, -1).contiguous()
        out = Obstacles(
            pos=f(self.pos),
            radius=f(self.radius),
            shape=i(self.shape),
            angle=angle,
            half_extents=f(self.half_extents),
            vel=f(self.vel),
            ang_vel=f(self.ang_vel),
            kind=i(self.kind),
            mass=f(self.mass),
            inertia=f(self.inertia),
            body=i(self.body),
            body_offset=f(self.body_offset),
        )
        out._resolved_for = key
        out._any_movable = self._any_movable
        return out


@dataclass
class WorldConfig:
    """Interaction and boundary settings shared by all envs.

    Attributes:
        collisions: enable agent-agent soft collision forces.
        collision_k: spring stiffness of the soft penalty.
        collision_c: damping coefficient (normal direction).
        collision_margin: forces activate within this gap around touching radii.
        bounds: world rectangle ``(x_min, x_max, y_min, y_max)``, or ``None``.
        bounds_mode: ``"soft"`` (spring-damper walls) or ``"clamp"``
            (positions hard-clamped inside; differentiable subgradient).
        neighbor_radius: radius for the neighbor lists; defaults to the
            interaction reach ``2 * max_agent_radius + collision_margin``.
            Must not be smaller than that reach when collisions are on.
        max_neighbors: padded neighbor-list width (truncates beyond).
        neighbor_method: ``"auto"`` | ``"grid"`` | ``"brute"`` |
            ``"uniform_grid"`` — see :class:`swarp.core.neighbors.NeighborGrid`.
        grid_dim: hash-grid bucket dimension per axis.
        uniform_bins: cells per axis for the ``"uniform_grid"`` backend
            (``None`` -> a ~sqrt(n_agents) heuristic).
    """

    collisions: bool = True
    collision_k: float = 100.0
    collision_c: float = 1.0
    collision_margin: float = 0.02
    # Viscous drag on MOVABLE obstacles, standing in for table friction: a pushed body
    # settles at sum(f) / (mass * damping) rather than accelerating without limit. Set 0
    # for a frictionless puck that coasts.
    obstacle_linear_damping: float = 10.0
    obstacle_angular_damping: float = 10.0
    # Depth at which the contact spring saturates (smoothly; 0 disables). A
    # velocity-controlled agent has no contact memory and settles at an overlap of
    # v*mass/(k*sub_dt), which at high collision_k is a violent impulse against a light
    # movable body. Saturating bounds it and leaves the shallow regime untouched.
    contact_max_overlap: float = 0.0
    bounds: tuple[float, float, float, float] | None = None
    bounds_mode: str = "soft"
    neighbor_radius: float | None = None
    max_neighbors: int = 32
    neighbor_method: str = "auto"
    grid_dim: int = 128
    uniform_bins: int | None = None
    integrator: Integrator = Integrator.EULER
    # Reuse the neighbor list the previous step's post-step build already
    # produced (on the state that is now the step input) for substep 0, instead
    # of rebuilding it — one build/step instead of two at substeps=1. Exact: the
    # reused list is bit-identical to a fresh build on the same positions. Only
    # engaged on the no-grad path (the taped path rebuilds so adjoints stay
    # correct). See :meth:`swarp.core.stepper.Stepper.launch_substeps`.
    neighbor_reuse: bool = True

    def override_with(self, other: WorldConfig | None) -> WorldConfig:
        """This config with ``other``'s **non-default** fields applied on top.

        Scenarios compute most of a ``WorldConfig`` from their own parameters — bounds
        from ``world_size``, ``neighbor_radius`` from the contact reach — so a caller's
        override cannot simply replace it wholesale without destroying those. Only the
        fields ``other`` sets away from the ``WorldConfig()`` defaults are taken, which
        is what makes ``world_config=WorldConfig(collision_k=50.0)`` mean "everything the
        scenario decided, but with that stiffness".

        The one thing this cannot express is forcing a field *back* to its
        ``WorldConfig()`` default against a scenario that changed it — that is
        indistinguishable from not asking. Build the scenario's ``World`` yourself when
        you need that.

        Returns ``self`` unchanged when there is nothing to apply.
        """
        if other is None:
            return self
        default = WorldConfig()
        changed = {
            f.name: getattr(other, f.name)
            for f in fields(self)
            if getattr(other, f.name) != getattr(default, f.name)
        }
        return replace(self, **changed) if changed else self

    def __post_init__(self) -> None:
        if self.bounds_mode not in ("soft", "clamp"):
            raise ValueError('bounds_mode must be "soft" or "clamp"')
        if self.bounds is not None:
            x_min, x_max, y_min, y_max = self.bounds
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("bounds must satisfy x_min < x_max and y_min < y_max")
        if self.collision_k < 0.0 or self.collision_c < 0.0 or self.collision_margin < 0.0:
            raise ValueError("collision constants must be non-negative")
        if self.obstacle_linear_damping < 0.0 or self.obstacle_angular_damping < 0.0:
            raise ValueError("obstacle damping must be non-negative")
        if self.contact_max_overlap < 0.0:
            raise ValueError("contact_max_overlap must be non-negative (0 disables)")
