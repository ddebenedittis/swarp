"""Low-level stepping: launches the dynamics substep chain on Warp arrays.

The Stepper is the single place that knows the substep pipeline. The autograd
bridge (grad mode) and the Environment hot path (no-grad mode) both call
:meth:`Stepper.launch_substeps`; collision-force passes plug in here without
touching either caller.
"""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

from wmas.core.state import VEC2, WorldState
from wmas.dynamics.base import AgentConfig, AgentParams, Integrator, build_agent_params
from wmas.dynamics.kernels import launch_integrate


class Stepper:
    """Owns per-agent parameters and the substep pipeline for one world layout."""

    def __init__(
        self,
        configs: list[AgentConfig],
        dt: float = 0.1,
        substeps: int = 1,
        device: str = "cuda:0",
        dtype=wp.float32,
        integrator: Integrator = Integrator.EULER,
    ) -> None:
        if integrator is not Integrator.EULER:
            raise NotImplementedError(
                f"Integrator {integrator} is not implemented yet; the dynamics derivative "
                "is a pure @wp.func, so RK4 slots into wmas/dynamics/kernels.py later."
            )
        if substeps < 1:
            raise ValueError("substeps must be >= 1")
        self.configs = configs
        self.n_agents = len(configs)
        self.dt = dt
        self.substeps = substeps
        self.sub_dt = dt / substeps
        self.device = device
        self.dtype = dtype
        self.integrator = integrator
        self.params: AgentParams = build_agent_params(configs, device=device, dtype=dtype)
        self._zero_forces: dict[int, wp.array] = {}

    def alloc_state(self, n_envs: int, requires_grad: bool = False) -> WorldState:
        return WorldState.zeros(
            n_envs, self.n_agents, dtype=self.dtype, device=self.device,
            requires_grad=requires_grad,
        )

    def zero_forces(self, n_envs: int) -> wp.array:
        """Shared all-zero force buffer (read-only, so safe to reuse under a tape)."""
        buf = self._zero_forces.get(n_envs)
        if buf is None:
            buf = wp.zeros(
                (n_envs, self.n_agents), dtype=VEC2[self.dtype], device=self.device
            )
            self._zero_forces[n_envs] = buf
        return buf

    def launch_substeps(
        self,
        state_in: WorldState,
        actions: wp.array,
        state_out: WorldState,
        make_buffer: Callable[[], WorldState] | None = None,
    ) -> None:
        """Advance one full env step (``substeps`` integrator substeps).

        Functional: ``state_in`` is never written. Intermediate states for
        substeps > 1 come from ``make_buffer``; under a tape the caller must
        supply *fresh* buffers (overwriting a taped array breaks the adjoint),
        while the no-grad hot path can recycle scratch buffers.
        """
        n_envs = state_in.pos.shape[0]
        forces = self.zero_forces(n_envs)
        chain = [state_in]
        for _ in range(self.substeps - 1):
            if make_buffer is None:
                raise ValueError("substeps > 1 requires make_buffer for intermediate states")
            chain.append(make_buffer())
        chain.append(state_out)
        for k in range(self.substeps):
            launch_integrate(chain[k], chain[k + 1], actions, forces, self.params, self.sub_dt)
