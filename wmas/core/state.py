"""Batched world state: structure-of-arrays Warp storage, [n_envs, n_agents]."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum

import warp as wp

VEC2 = {wp.float32: wp.vec2f, wp.float64: wp.vec2d}
VEC3 = {wp.float32: wp.vec3f, wp.float64: wp.vec3d}
QUAT = {wp.float32: wp.quatf, wp.float64: wp.quatd}
TORCH_DTYPE_TO_WP = {"torch.float32": wp.float32, "torch.float64": wp.float64}


class FieldType(Enum):
    """Per-agent state field kind, mapped to a Warp dtype at allocation time."""

    SCALAR = 0
    VEC2 = 1
    VEC3 = 2
    QUAT = 3


#: Warp element type for a (FieldType, precision) pair.
_FIELD_WARP_TYPE = {
    FieldType.SCALAR: lambda dt: dt,
    FieldType.VEC2: lambda dt: VEC2[dt],
    FieldType.VEC3: lambda dt: VEC3[dt],
    FieldType.QUAT: lambda dt: QUAT[dt],
}


def field_wp_dtype(name: str, scalar):
    """Warp element dtype of a state field at the given scalar precision."""
    return _FIELD_WARP_TYPE[STATE_FIELDS[name]](scalar)


#: State field name -> FieldType. The first five are the 2D-vehicle state; the
#: last four are the extra 6-DOF drone state (altitude, vertical velocity,
#: attitude quaternion, body angular rates). 2D models carry the drone fields as
#: zeros and pass them through unchanged; the drone ignores theta/speed/ang_vel.
STATE_FIELDS = {
    "pos": FieldType.VEC2,
    "theta": FieldType.SCALAR,
    "vel": FieldType.VEC2,
    "speed": FieldType.SCALAR,
    "ang_vel": FieldType.SCALAR,
    "z": FieldType.SCALAR,
    "vz": FieldType.SCALAR,
    "attitude": FieldType.QUAT,
    "body_rates": FieldType.VEC3,
}


@dataclass
class WorldState:
    """Unified per-agent state for all dynamics models.

    ``vel`` always holds the (horizontal) translational velocity used in the
    last pose update (consistent across models, for uniform observations);
    ``speed`` is the scalar forward speed integrated by the diff-drive
    (acceleration mode) and bicycle models; ``theta``/``ang_vel`` are ignored by
    holonomic agents. ``z``/``vz``/``attitude``/``body_rates`` are the extra
    6-DOF drone state and are zero (identity for ``attitude`` once reset) for the
    2D vehicle models.
    """

    pos: wp.array  # [n_envs, n_agents] vec2 (x, y)
    theta: wp.array  # [n_envs, n_agents] float (heading)
    vel: wp.array  # [n_envs, n_agents] vec2 (horizontal translational velocity)
    speed: wp.array  # [n_envs, n_agents] float
    ang_vel: wp.array  # [n_envs, n_agents] float
    z: wp.array  # [n_envs, n_agents] float (altitude; drone)
    vz: wp.array  # [n_envs, n_agents] float (vertical velocity; drone)
    attitude: wp.array  # [n_envs, n_agents] quat (drone attitude, body->world)
    body_rates: wp.array  # [n_envs, n_agents] vec3 (drone body angular rates)

    @classmethod
    def zeros(
        cls,
        n_envs: int,
        n_agents: int,
        dtype=wp.float32,
        device: str = "cuda:0",
        requires_grad: bool = False,
    ) -> WorldState:
        shape = (n_envs, n_agents)

        def alloc(ftype: FieldType) -> wp.array:
            dt = _FIELD_WARP_TYPE[ftype](dtype)
            return wp.zeros(shape, dtype=dt, device=device, requires_grad=requires_grad)

        return cls(**{name: alloc(ftype) for name, ftype in STATE_FIELDS.items()})

    def arrays(self) -> list[wp.array]:
        return [getattr(self, f.name) for f in fields(self)]

    def assign(self, other: WorldState) -> None:
        for dst, src in zip(self.arrays(), other.arrays(), strict=True):
            wp.copy(dst, src)

    def zero_(self) -> None:
        for arr in self.arrays():
            arr.zero_()
