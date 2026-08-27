"""Sampling: agents collect an unknown scalar field, consuming cells they visit.

Port of the VMAS ``sampling`` scenario (no new physics). Each env carries a
batched sum-of-Gaussians density on a ``grid_res x grid_res`` grid over the
world square. An agent earns the field value at its current cell the first time
any agent enters that cell; the cell is then marked consumed. Observation is the
agent's own pose plus the field sampled at the 3x3 cell neighborhood around it
(a local gradient cue), all differentiable w.r.t. positions.
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from swarp._overloads import concrete
from swarp.core.cached_launch import CachedLaunch, ptr_key
from swarp.core.config import WorldConfig
from swarp.core.state import VEC2
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.interop.autograd import torch_stream_scope
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario
from swarp.scenarios.sampling_kernels import (
    sampling_obs_reward_kernel,
    sampling_reset_kernel,
    sampling_scatter_kernel,
)


class SamplingScenario(FusedScenario):
    def __init__(
        self,
        n_agents: int = 4,
        agent_radius: float = 0.05,
        world_size: float = 1.0,
        n_gaussians: int = 3,
        grid_res: int = 12,
        field_std: float = 0.15,
        max_speed: float = 1.0,
    ) -> None:
        self.n_agents = n_agents
        self.agent_radius = agent_radius
        self.world_size = world_size
        self.n_gaussians = n_gaussians
        self.grid_res = grid_res
        self.field_std = field_std
        self.max_speed = max_speed

    def make_world(self, n_envs, device, dt, substeps, dtype, world_config=None) -> World:
        cfgs = [
            AgentConfig(
                model=DynamicsModel.HOLONOMIC,
                ctrl_mode=ControlMode.VELOCITY,
                radius=self.agent_radius,
                max_speed=self.max_speed,
                max_accel=2.0 * self.max_speed,
            )
            for _ in range(self.n_agents)
        ]
        margin = 0.5 * self.agent_radius
        cfg = WorldConfig(
            collisions=True,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            max_neighbors=min(32, max(4, self.n_agents)),
        ).override_with(world_config)
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        # Task state, allocated here (n_envs is known) and only ever written in place, so
        # the fused spec can adopt it with alloc="never". Allocating it on first reset
        # instead made the fused path silently depend on reset running first.
        self.centers = torch.zeros(  # [n_envs, n_gaussians, 2]
            n_envs, self.n_gaussians, 2, device=device, dtype=dtype
        )
        self.consumed = torch.zeros(  # [n_envs, grid_res*grid_res]
            n_envs, self.grid_res * self.grid_res, dtype=torch.bool, device=device
        )
        self._cache: dict[str, torch.Tensor] | None = None
        # Cell-centre coordinates in world units, per axis: (i+0.5)/res * 2W - W.
        i = torch.arange(self.grid_res, device=device, dtype=dtype)
        self._cell_coord = (i + 0.5) / self.grid_res * 2.0 * self.world_size - self.world_size
        # The 3x3 neighbourhood stencil, built here rather than per refresh: a
        # ``torch.tensor([...])`` literal is a host allocation plus a H2D copy, and on the
        # torch path ``_refresh`` runs every step.
        self._offs = torch.tensor([-1, 0, 1], device=device)
        # Cached, repack-once launches for the eager reset path — see
        # swarp/core/cached_launch.py.
        self._reset_launch = CachedLaunch()
        self._obs_reward_launch = CachedLaunch()
        self._scatter_launch = CachedLaunch()
        return self.world

    @property
    def obs_dim(self) -> int:
        return 4 + 9  # pos, vel, 3x3 field samples

    # ------------------------------------------------------------------ field

    def _field(self, pts: torch.Tensor) -> torch.Tensor:
        """Density at ``pts`` [..., 2] given the per-env Gaussian centres.

        ``pts`` broadcasts against ``centers`` on the env axis (dim 0).
        """
        c = self.centers  # [n_envs, G, 2]
        # pts: [n_envs, N, 2] -> [n_envs, N, 1, 2] ; centers -> [n_envs, 1, G, 2]
        d2 = (pts.unsqueeze(-2) - c.unsqueeze(1)).square().sum(-1)  # [n_envs, N, G]
        return torch.exp(-d2 / (2.0 * self.field_std**2)).sum(-1)  # [n_envs, N]

    def _cell_of(self, pos: torch.Tensor) -> torch.Tensor:
        """Linear cell id [n_envs, n_agents] for positions [n_envs, n_agents, 2]."""
        u = (pos + self.world_size) / (2.0 * self.world_size)  # -> [0, 1]
        c = (u * self.grid_res).floor().long().clamp(0, self.grid_res - 1)
        return c[..., 1] * self.grid_res + c[..., 0]

    # ------------------------------------------------------------------ reset

    def reset_world(
        self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False
    ) -> None:
        """Masked reset in one Warp launch (see :mod:`swarp.scenarios.reset_kernels`)."""
        w = self.world
        lim = self.world_size - 2.0 * self.agent_radius
        mask, use_mask = self.reset_mask_wp(env_mask)
        st = w.state_wp()
        scalar = w.wp_dtype
        if self.fused_active:
            # ``centers``/``consumed`` are adopted (``alloc="never"``) and never
            # reassigned, so the fused spec's cached handles are always current —
            # reuse them instead of re-wrapping on every reset.
            self.ensure_fused()
            aux = self._wp["centers"]
            flags = self._wp["consumed"]
        else:
            aux = wp.from_torch(self.centers.contiguous(), dtype=VEC2[scalar])
            flags = wp.from_torch(self.consumed.view(torch.uint8))
        seed = wp.int32(w.next_kernel_seed())
        with torch_stream_scope(w.device):
            launch = self._reset_launch.get(
                concrete(sampling_reset_kernel, scalar),
                dim=w.n_envs,
                inputs=[
                    mask,
                    use_mask,
                    seed,
                    scalar(lim),
                    scalar(lim),
                    wp.int32(self.n_agents),
                    wp.int32(self.n_gaussians),
                    wp.int32(self.consumed.shape[1]),
                    st.pos,
                    st.vel,
                    aux,
                    flags,
                ],
                device=w.device,
                key=(
                    w.n_envs,
                    ptr_key(mask),
                    lim,
                    self.n_agents,
                    self.n_gaussians,
                    self.consumed.shape[1],
                    ptr_key(st.pos),
                    ptr_key(st.vel),
                    ptr_key(aux),
                    ptr_key(flags),
                ),
            )
            launch.set_param_by_name("use_mask", use_mask)
            launch.set_param_by_name("seed", seed)
            launch.launch()
        w.mark_pos_dirty()
        self.finish_reset(env_mask, obs_only=obs_only)

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh()

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        self._refresh()

    # --------------------------------------------------------- fused fast path

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        """Three framework outputs plus two pieces of *adopted* scenario state.

        ``centers`` and ``consumed`` are the scenario's own task state, allocated in
        ``make_world`` and only ever written in place, so the spec declares
        ``alloc="never"``: the framework wraps them (``consumed`` reinterpreted from torch
        ``bool`` to Warp ``uint8``) and treats a ``None`` as an error rather than quietly
        allocating a second copy.
        """
        ne, na = n_envs, self.n_agents
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("reward", (ne, na)),
            Buf("field", (ne, na)),
            Buf("centers", (ne, self.n_gaussians, 2), "vec2", attr="centers", alloc="never"),
            Buf(
                "consumed",
                (ne, self.grid_res * self.grid_res),
                "bool",
                attr="consumed",
                alloc="never",
                carry=True,
            ),
        )

    def launch_fused(self, pass_: FusedPass) -> None:
        """Obs+reward, then the consume scatter.

        The scatter runs on a reset too, mirroring the torch ``_refresh``, which marks the
        cells the fresh spawn sits on as consumed. Only ``full_pass`` differs: a mid-step
        auto-reset must not clobber the reward/field already returned for the transition.
        """
        st = self.world.state_wp()  # one wrap for both launches
        self._launch_obs_reward(st, full_pass=pass_.full_pass)
        self._launch_scatter(st)

    def _launch_obs_reward(self, st, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        centers, consumed = self._wp["centers"], self._wp["consumed"]
        obs, reward, field = self._wp["obs"], self._wp["reward"], self._wp["field"]
        launch = self._obs_reward_launch.get(
            concrete(sampling_obs_reward_kernel, self.world.wp_dtype),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                centers,
                consumed,
                scalar(self.world_size),
                wp.int32(self.grid_res),
                wp.int32(self.n_gaussians),
                scalar(2.0 * self.field_std**2),
                wp.int32(full_pass),
            ],
            outputs=[obs, reward, field],
            device=w.device,
            key=(
                w.n_envs,
                self.n_agents,
                ptr_key(st.pos),
                ptr_key(st.vel),
                ptr_key(centers),
                ptr_key(consumed),
                self.world_size,
                self.grid_res,
                self.n_gaussians,
                self.field_std,
                ptr_key(obs),
                ptr_key(reward),
                ptr_key(field),
            ),
        )
        launch.set_param_by_name("full_pass", wp.int32(full_pass))
        launch.launch()

    def _launch_scatter(self, st) -> None:
        w = self.world
        scalar = w.wp_dtype
        consumed = self._wp["consumed"]
        # No per-call-varying argument: idempotent write of the agent's own cell.
        launch = self._scatter_launch.get(
            concrete(sampling_scatter_kernel, self.world.wp_dtype),
            dim=(w.n_envs, self.n_agents),
            inputs=[st.pos, scalar(self.world_size), wp.int32(self.grid_res)],
            outputs=[consumed],
            device=w.device,
            key=(
                w.n_envs,
                self.n_agents,
                ptr_key(st.pos),
                self.world_size,
                self.grid_res,
                ptr_key(consumed),
            ),
        )
        launch.launch()

    def _refresh(self) -> None:
        w = self.world
        pos = w.state.pos
        cell = self._cell_of(pos)  # [n_envs, n_agents]
        was_consumed = torch.gather(self.consumed, 1, cell)  # [n_envs, n_agents]
        fval = self._field(pos)  # [n_envs, n_agents]
        reward = fval * (~was_consumed).to(w.dtype)
        # Mark the visited cells consumed (idempotent when agents share a cell).
        self.consumed.scatter_(1, cell, torch.ones_like(was_consumed))

        # Local 3x3 field samples around each agent's cell (observation cue).
        cx = (cell % self.grid_res).clamp(1, self.grid_res - 2)
        cy = (cell // self.grid_res).clamp(1, self.grid_res - 2)
        gx = (cx.unsqueeze(-1) + self._offs).clamp(0, self.grid_res - 1)  # [E, A, 3]
        gy = (cy.unsqueeze(-1) + self._offs).clamp(0, self.grid_res - 1)
        # cell-centre coordinates for the 3x3 block: [n_envs, n_agents, 9, 2]
        xs = self._cell_coord[gx]  # [n_envs, n_agents, 3]
        ys = self._cell_coord[gy]
        grid_pts = torch.stack(
            [
                xs.unsqueeze(-1).expand(-1, -1, -1, 3),
                ys.unsqueeze(-2).expand(-1, -1, 3, -1),
            ],
            dim=-1,
        ).reshape(w.n_envs, w.n_agents * 9, 2)
        samples = self._field(grid_pts).reshape(w.n_envs, w.n_agents, 9)

        self._cache = {"reward": reward, "field": fval, "samples": samples}

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["obs"]
        w = self.world
        s = w.state
        return torch.cat([s.pos, s.vel, self._cache["samples"]], dim=-1)

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        return self._cache["reward"][:, agent_idx]

    def rewards(self) -> torch.Tensor:
        return self.fb["reward"] if self.fused_active else super().rewards()

    def info(self) -> dict[str, Any]:
        field = self.fb["field"] if self.fused_active else self._cache["field"]
        return {"field": field, "consumed_frac": self._consumed_frac_now()}

    def _consumed_frac_now(self) -> torch.Tensor:
        """Fraction of cells consumed.

        ``.to(dtype)`` rather than ``.float()``: the old form reported float32 in a float64
        world. Reducing into a preallocated buffer instead looks like the obvious win and is
        not one — see :meth:`~swarp.scenarios.discovery.DiscoveryScenario.info` for the
        measurement (``sum(out=)`` + ``mul_`` is ~30% slower than the allocating mean).
        """
        return self.consumed.to(self.world.dtype).mean(-1)
