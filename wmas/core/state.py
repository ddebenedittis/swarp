"""Batched world state: structure-of-arrays Warp storage, [n_envs, n_agents]."""

from __future__ import annotations

from dataclasses import dataclass, fields

import warp as wp

VEC2 = {wp.float32: wp.vec2f, wp.float64: wp.vec2d}
TORCH_DTYPE_TO_WP = {"torch.float32": wp.float32, "torch.float64": wp.float64}

#: State field name -> True if vec2-valued (else scalar).
STATE_FIELDS = {
    "pos": True,
    "theta": False,
    "vel": True,
    "speed": False,
    "ang_vel": False,
}


@dataclass
class WorldState:
    """Unified per-agent state for all dynamics models.

    ``vel`` always holds the translational velocity actually used in the last
    pose update (consistent across models, for uniform observations); ``speed``
    is the scalar forward speed integrated by the diff-drive (acceleration
    mode) and bicycle models; ``theta``/``ang_vel`` are ignored by holonomic
    agents.
    """

    pos: wp.array  # [n_envs, n_agents] vec2
    theta: wp.array  # [n_envs, n_agents] float
    vel: wp.array  # [n_envs, n_agents] vec2
    speed: wp.array  # [n_envs, n_agents] float
    ang_vel: wp.array  # [n_envs, n_agents] float

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

        def alloc(is_vec: bool) -> wp.array:
            dt = VEC2[dtype] if is_vec else dtype
            return wp.zeros(shape, dtype=dt, device=device, requires_grad=requires_grad)

        return cls(**{name: alloc(is_vec) for name, is_vec in STATE_FIELDS.items()})

    def arrays(self) -> list[wp.array]:
        return [getattr(self, f.name) for f in fields(self)]

    def assign(self, other: WorldState) -> None:
        for dst, src in zip(self.arrays(), other.arrays(), strict=True):
            wp.copy(dst, src)

    def zero_(self) -> None:
        for arr in self.arrays():
            arr.zero_()
