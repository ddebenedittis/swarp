"""Caging: agents surround a drifting disc so it cannot escape through any gap.

The objective is **topological, not metric**. What is rewarded is not "be near the disc"
but "leave no hole in the ring around it": the agents' bearings as seen from the disc
centre are sorted around the circle and the *largest circular gap* between consecutive
bearings is the score. A tight cluster of agents sitting on top of the disc has a gap of
nearly a full turn and scores terribly; the same agents spread evenly around it at the
same radius score best. That is the one thing none of the other scenarios reward.

Shaping, and why it is needed
-----------------------------
``max_gap`` is a **max**, so on its own it hands gradient to exactly the two agents
bordering the single widest gap and none to anybody else — while the radial
``|distance - cage_radius|`` term is dense and easy. A policy trained on that combination
learns the radius and ignores the bearing: the agents settle onto the right ring, clumped,
and ``max_gap`` *rises*. So the per-agent reward also carries a dense even-spacing term,

    -spacing_factor * ( (gap_ccw - 2*pi/n)**2 + (gap_cw - 2*pi/n)**2 )

over the agent's own two local gaps — the ones it already observes, obs slots 8 and 9.
Its optimum is the evenly spaced ring, which is exactly the configuration that minimizes
``max_gap``, so it agrees with the topological objective rather than competing with it,
the way navigation's per-agent shaping agrees with its team bonus.

**Squared, not absolute.** The obvious ``|gap_ccw - t| + |gap_cw - t|`` does not actually
deliver on "every agent gets a gradient". Differentiate the whole team's L1 sum by one
agent's bearing and the four terms it appears in collapse to
``2*(sign(gap_cw - t) - sign(gap_ccw - t))``, which is **exactly zero** whenever that
agent's two adjacent gaps fall on the same side of the even share — measured, that is a
third of the agents at ``n_agents=3`` and near half at ``n_agents=6``. The squared form
gives ``4*(gap_cw - gap_ccw)`` instead: zero only when the agent sits at the midpoint of
its two neighbours, which is the correct stationary point, and it starves ~1% of agents
rather than ~40%.

``max_gap``, ``caged``, :meth:`~CagingScenario.done` and the ``info`` keys are untouched
by any of this: the topological quantity stays the headline metric and the termination
condition, and ``spacing_factor=0.0`` recovers the unshaped reward exactly.

Escape law
----------
The disc is not passive. Every step it accelerates by

    a_escape = -escape_accel * mean_a( unit vector from the disc to agent a )

i.e. **away from where the agents are**, at up to ``escape_accel``. A ring that closes
around the disc cancels that mean and the drift vanishes; a ring with a hole leaves a
residual pointing straight *into* the hole, and the disc squeezes out through it. So a
cage that is merely *near* the disc does not hold — the agents have to actually close the
ring, which is exactly what the reward asks for. The law is deliberately smooth in the
agent positions (no ``argmax`` over gaps, no normalization at zero), so the two paths
below cannot disagree about which gap is widest and the disc trajectory has no
discontinuity when the widest gap changes hands.

Two bounds keep the disc catchable, and both matter. ``escape_accel`` fixes the
*sustained* escape speed at ``escape_accel / (disc_mass * linear_damping)`` = 0.5, half
the agents' ``max_speed``, and because the drift is a *mean* of unit vectors rather than a
sum it cannot stack however many agents press. On top of that ``max_disc_speed`` (default
``0.75 * max_speed``) caps the integrated velocity outright, which is what survives a
contact transient — an agent arriving at full speed compresses the contact spring before
the disc responds, and that measured a peak of 2.9x ``max_speed`` before the cap existed.
A disc that can outrun its captors makes the cage unreachable and the task unlearnable, so
``test_the_disc_cannot_outrun_its_captors`` pins both. Shepherding's sheep are bounded the
same way, deliberately.

The disc, and why it is not an ``ObstacleKind.MOVABLE``
------------------------------------------------------
The disc is **scenario-owned**, transport-style: this class holds ``disc_pos``/
``disc_vel``, integrates them itself, and re-installs them each step as a retained,
``kind``-free :class:`~swarp.core.config.Obstacles` spec so that agent-disc contact comes
free from the engine's existing soft obstacle contact. A ``MOVABLE`` obstacle would not
do: the engine integrates those from contact reaction *only*, with no hook for the escape
drift; that integration is ``record_tape=False``, so it is skipped outright on a taped
step; and an install carrying ``kind`` reads ``Obstacles.any_movable`` back to the host,
which is not CUDA-graph-capture-safe. Position only, no orientation: a circle under
frictionless normal contact has zero lever arm about its own centre
(``swarp/core/bodies.py``), so an angle would be state that never changes.

Coupling is staggered by one step, as in transport — the agents see the previous step's
disc pose. The torch integration in :meth:`CagingScenario._refresh` is the **parity
oracle** and the only path that carries gradients; the no-grad hot path does the same
physics in ``caging_body_kernel`` (:mod:`swarp.scenarios.caging_kernels`) so the whole
step stays capturable as one CUDA graph. The two are tested against each other in
``tests/scenarios/test_caging_fused.py``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp

from swarp._overloads import concrete
from swarp.core.cached_launch import CachedLaunch, ptr_key
from swarp.core.config import Obstacles, WorldConfig
from swarp.core.state import VEC2
from swarp.core.world import World
from swarp.dynamics.base import AgentConfig, ControlMode, DynamicsModel
from swarp.interop.autograd import torch_stream_scope
from swarp.scenarios.caging_kernels import (
    caging_body_kernel,
    caging_obs_kernel,
    caging_reset_kernel,
    caging_reward_kernel,
)
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario

TWO_PI = 2.0 * math.pi


class CagingScenario(FusedScenario):
    """Surround a drifting disc so tightly that no angular gap lets it out.

    Args:
        n_agents: agents per env. Holonomic, velocity-controlled.
        agent_radius: agent disc radius.
        disc_radius: radius of the caged disc.
        disc_mass: mass used to turn the summed contact force into an acceleration.
        world_size: half-width of the square arena.
        max_speed: agent speed limit.
        contact_k / contact_c: the engine's ``collision_k``/``collision_c``, named for
            the agent-disc contact that dominates here.
        contact_margin: this scenario's *own* agent-disc activation gap for the torch
            reference body physics. Not the engine's ``collision_margin``, which is
            derived from ``agent_radius`` in :meth:`make_world`.
        linear_damping: viscous damping on the disc velocity.
        escape_accel: magnitude cap of the outward escape drift (see the module
            docstring for the law). With the default damping this fixes the disc's
            sustained escape speed at ``escape_accel / (disc_mass * linear_damping)``.
        max_disc_speed: hard cap on the disc's speed, applied to the integrated velocity
            before it moves the position. Defaults to ``0.75 * max_speed`` so the disc can
            always be run down, whatever a contact transient does.
        gap_threshold: the largest circular gap, in radians, at or below which the cage
            counts as closed. Defaults to ``2.4`` rad (~137 deg), which leaves ~17 deg of
            margin over the ``2*pi/3`` (120 deg) floor three evenly spaced agents can
            reach — enough that a trio tracking a *moving* disc can actually hold it.
            ``2.2`` was the earlier value and it was a trap: a scripted policy parked on
            the exact even-spacing slots caged only 37% of envs at ``n_agents=3``, because
            controller lag alone eats 0.1 rad. The threshold only ever binds for three or
            four agents (five evenly spaced already sit at 1.26 rad), and because the
            reward term is ``gap_threshold - max_gap`` it shifts the reward by a constant
            and changes no gradient.
        gap_reward: scale on the ``(gap_threshold - max_gap)`` term. Positive once the
            ring closes, negative while it is open.
        cage_radius: the ring radius agents are shaped towards. Defaults to
            ``disc_radius + 2 * agent_radius``.
        radius_factor: per-agent penalty on ``|distance - cage_radius|``. This is what
            makes the agents cage *close* instead of forming a huge, easy ring at
            infinity.
        spacing_factor: per-agent penalty on the squared deviation of the agent's own two
            local gaps from an even share of the circle — the dense angular shaping (see
            the module docstring). ``0.0`` turns it off and leaves the sparse ``max_gap``
            term as the only bearing signal.
        contact_penalty: per-agent penalty on penetration into the disc, so "cage" does
            not degenerate into "sit on it".
        capture_radius: every agent must be within this of the disc centre for the cage
            to count as closed. Defaults to ``2 * cage_radius``.
        caged_reward: one-off bonus while the cage is closed.
    """

    def __init__(
        self,
        n_agents: int = 4,
        agent_radius: float = 0.05,
        disc_radius: float = 0.12,
        disc_mass: float = 1.0,
        world_size: float = 1.0,
        max_speed: float = 1.0,
        contact_k: float = 100.0,
        contact_c: float = 5.0,
        contact_margin: float = 0.01,
        linear_damping: float = 0.5,
        escape_accel: float = 0.25,
        max_disc_speed: float | None = None,
        gap_threshold: float = 2.4,
        gap_reward: float = 1.0,
        cage_radius: float | None = None,
        radius_factor: float = 0.5,
        spacing_factor: float = 0.1,
        contact_penalty: float = 1.0,
        capture_radius: float | None = None,
        caged_reward: float = 5.0,
    ) -> None:
        self.n_agents = n_agents
        self.agent_radius = agent_radius
        self.disc_radius = disc_radius
        self.disc_mass = disc_mass
        self.world_size = world_size
        self.max_speed = max_speed
        self.contact_k = contact_k
        self.contact_c = contact_c
        self.contact_margin = contact_margin
        self.linear_damping = linear_damping
        # A sustained acceleration ``a`` gives a terminal speed of
        # ``a / (disc_mass * linear_damping)``, so ``escape_accel`` alone fixes how fast
        # the disc can ever run away from a one-sided cage:
        #
        #     escape_accel / (disc_mass * linear_damping) = 0.25 / (1.0 * 0.5) = 0.5
        #
        # i.e. half the agents' ``max_speed``, independent of how many of them press —
        # the drift is a *mean* of unit vectors, so it cannot stack. It used to be 0.5,
        # giving a terminal 1.0 that exactly matched ``max_speed``: the disc could not be
        # cornered, and a 4000-iteration MAPPO run drove ``max_gap`` from 3.21 to 5.83
        # (every agent stranded on one side) with ``caged`` at ~0.0002 throughout.
        self.escape_accel = escape_accel
        # ...and the hard guarantee on top, because the terminal speed above says nothing
        # about a **contact transient**: an agent arriving at full speed compresses a
        # k=100 spring before the disc responds, which measured a peak of 2.9x
        # ``max_speed`` even with the drift capped. Softening the contact is the wrong
        # trade — ``contact_k`` is also the engine's agent<->agent ``collision_k``, so
        # lowering it lets agents interpenetrate. A speed cap is independent of dt, of the
        # stiffness and of the agent count, which is why shepherding chose the same
        # formulation for its sheep; caging mirrors it. Default 75% of ``max_speed``, so
        # an agent can always close on the fleeing disc, with the drift-only terminal 0.5
        # left comfortably underneath. ``test_the_disc_cannot_outrun_its_captors`` pins it.
        self.max_disc_speed = max_disc_speed if max_disc_speed is not None else 0.75 * max_speed
        self.gap_threshold = gap_threshold
        self.gap_reward = gap_reward
        self.cage_radius = (
            cage_radius if cage_radius is not None else disc_radius + 2.0 * agent_radius
        )
        self.radius_factor = radius_factor
        self.spacing_factor = spacing_factor
        # An even share of the circle: the per-agent spacing shaping's target, and a host
        # constant so both paths use the bit-identical value.
        self._even_gap = TWO_PI / float(n_agents)
        self.contact_penalty = contact_penalty
        self.capture_radius = (
            capture_radius if capture_radius is not None else 2.0 * self.cage_radius
        )
        self.caged_reward = caged_reward

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
        # neighbor reach must cover agent<->disc contact so the step sees it
        reach = 2.0 * max(self.agent_radius, self.disc_radius) + margin
        cfg = WorldConfig(
            collisions=True,
            collision_k=self.contact_k,
            collision_c=self.contact_c,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            neighbor_radius=reach,
            max_neighbors=min(32, max(4, self.n_agents)),
        ).override_with(world_config)
        self.dt = dt
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )
        # Reset geometry, derived once. Agents spawn on an annulus about the disc rather
        # than uniformly over the arena: caging is about *where on the ring* they sit, so
        # a uniform spawn would spend most of an episode just closing the distance. The
        # disc's own spawn box is shrunk by the annulus so the ring fits inside the world.
        self._agent_lim = self.world_size - 2.0 * self.agent_radius
        self._spawn_min = self.cage_radius + 2.0 * self.agent_radius
        # ...and never ask for a ring the arena cannot hold: a world too small for the
        # annulus shrinks it rather than clamping every agent onto the wall.
        self._spawn_max = min(
            max(1.5 * self._spawn_min, 0.5 * self.world_size),
            max(self._spawn_min, self._agent_lim),
        )
        self._disc_lim = max(0.0, self._agent_lim - self._spawn_max)
        self._disc_bound = self.world_size - self.disc_radius
        # 1/n_agents as a stored constant so the escape drift's reduction is a multiply
        # on both paths — a ``.mean()`` on one side and a reciprocal multiply on the
        # other would differ by an ulp *inside the dynamics*, which compounds.
        self._inv_agents = 1.0 / float(self.n_agents)
        # Disc state: allocated here (n_envs is known) so the fused spec can adopt it, and
        # written in place by every reset so its handles stay valid. Shaped [n_envs, 1, 2]
        # because that is the layout ``Obstacles`` wants; the kernels index ``[e, 0]``.
        tt = {"device": device, "dtype": dtype}
        self.disc_pos = torch.zeros(n_envs, 1, 2, **tt)
        self.disc_vel = torch.zeros(n_envs, 1, 2, **tt)
        # ONE retained obstacle spec over disc_pos, re-installed rather than rebuilt —
        # see ``_install_obstacles`` for why that is what makes an in-capture install
        # legal. Built here so the single count-changing install happens before any graph
        # capture, and it carries no ``kind``, so ``any_movable`` never touches the device.
        self._disc_radius_t = torch.full((1,), self.disc_radius, **tt)
        self._obstacles = Obstacles(self.disc_pos.detach(), self._disc_radius_t).resolve(
            device, dtype
        )
        self._cache: dict[str, torch.Tensor] | None = None
        # Cached, repack-once launches for the eager reset path — see
        # swarp/core/cached_launch.py.
        self._reset_launch = CachedLaunch()
        self._obs_launch = CachedLaunch()
        self._reward_launch = CachedLaunch()
        return self.world

    @property
    def obs_dim(self) -> int:
        # pos(2) + vel(2) + disc_rel(2) + disc_vel(2) + gap_ccw(1) + gap_cw(1)
        return 10

    # ------------------------------------------------------------------ reset

    def reset_world(
        self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False
    ) -> None:
        """Masked reset in one Warp launch (see :mod:`swarp.scenarios.reset_kernels`).

        The disc buffers are written **in place** by the kernel, so the fused path's
        cached Warp handles and the whole-step graph stay valid across resets; the grad
        path's ``_refresh`` still reassigns them (fresh tensors for the tape), which the
        framework's handle resync catches on the next no-grad step.
        """
        w = self.world
        mask, use_mask = self.reset_mask_wp(env_mask)
        st = w.state_wp()
        scalar = w.wp_dtype
        vec2 = VEC2[scalar]
        if self.fused_active:
            # Reuse the fused spec's cached, pointer-resynced handles instead of
            # re-wrapping these watched tensors on every reset. ``sync_fused_handles``
            # must run first: it is what notices a grad-path reassignment and rebuilds
            # the handle before this kernel writes through it.
            self.ensure_fused()
            self.sync_fused_handles()
            disc_pos = self._wp["disc_pos"]
            disc_vel = self._wp["disc_vel"]
        else:
            disc_pos = wp.from_torch(self.disc_pos.contiguous(), dtype=vec2)
            disc_vel = wp.from_torch(self.disc_vel.contiguous(), dtype=vec2)
        seed = wp.int32(w.next_kernel_seed())
        with torch_stream_scope(w.device):
            launch = self._reset_launch.get(
                concrete(caging_reset_kernel, scalar),
                dim=w.n_envs,
                inputs=[
                    mask,
                    use_mask,
                    seed,
                    scalar(self._agent_lim),
                    scalar(self._disc_lim),
                    scalar(self._spawn_min),
                    scalar(self._spawn_max),
                    wp.int32(self.n_agents),
                    st.pos,
                    st.vel,
                    disc_pos,
                    disc_vel,
                ],
                device=w.device,
                key=(
                    w.n_envs,
                    ptr_key(mask),
                    self._agent_lim,
                    self._disc_lim,
                    self._spawn_min,
                    self._spawn_max,
                    self.n_agents,
                    ptr_key(st.pos),
                    ptr_key(st.vel),
                    ptr_key(disc_pos),
                    ptr_key(disc_vel),
                ),
            )
            launch.set_param_by_name("use_mask", use_mask)
            launch.set_param_by_name("seed", seed)
            launch.launch()
        w.mark_pos_dirty()

        self._install_obstacles()
        self.finish_reset(env_mask, obs_only=obs_only)

    def _install_obstacles(self) -> None:
        """Hand the current disc pose to the engine as a circular obstacle.

        Called from inside the captured whole-step graph (see :meth:`launch_fused`), so it
        must stay in the capture-safe regime of :meth:`swarp.core.stepper.Stepper.
        set_obstacles`. All three of its preconditions hold here, by construction:

        * the spec is **retained** — built once in ``make_world`` and re-installed, never
          rebuilt — so ``Obstacles.resolve`` returns ``self`` and allocates nothing;
        * it carries no ``kind``, so ``Obstacles.any_movable`` short-circuits to ``False``
          without the reduction plus ``.item()`` that would sync the device to the host;
        * the obstacle count is fixed at one for the life of the world, so
          ``Stepper.set_obstacles`` takes its in-place path and no recapture fires.

        The spec aliases ``disc_pos``, so writing a new disc position in place is the
        whole of a "move the disc" update. It is rebuilt only when ``disc_pos`` is
        *reassigned*, which is the grad path integrating the disc in torch (it needs fresh
        tensors for the tape). That is a host-side pointer compare, and on the captured
        path the pointer never moves, so the rebuild branch cannot fire inside a capture.
        """
        if self._obstacles.pos.data_ptr() != self.disc_pos.data_ptr():
            self._obstacles = Obstacles(self.disc_pos.detach(), self._disc_radius_t).resolve(
                self.world.device, self.world.dtype
            )
        self.world.set_obstacles(self._obstacles)

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh(integrate=True)

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        self._refresh(integrate=False)

    # --------------------------------------------------------- fused fast path

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        ne, na = n_envs, self.n_agents
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("reward", (ne, na)),
            # ``gap``/``caged`` are written by the reward kernel only, i.e. only on a
            # full pass — an obs-only auto-reset must leave the info/done it already
            # returned for the transition just taken alone.
            Buf("gap", (ne,)),
            Buf("caged", (ne,), "uint8", bool_view=True),
            Buf("resetmask", (ne,), "uint8", reset_mask=True),
            # Disc state: the scenario's own tensors, advanced in place by the body
            # kernel and reassigned by the grad path's torch integrator.
            Buf("disc_pos", (ne, 1, 2), "vec2", attr="disc_pos", alloc="never", carry=True,
                watch=True),
            Buf("disc_vel", (ne, 1, 2), "vec2", attr="disc_vel", alloc="never", carry=True,
                watch=True),
        )

    def engine_carries(self) -> list[torch.Tensor]:
        """The engine's obstacle positions: ``_install_obstacles`` overwrites them from
        inside the hook, and the *next* step's physics reads them."""
        return [self.world.obstacle_state_views()[0]]

    def launch_fused(self, pass_: FusedPass) -> None:
        """Integrate the disc, re-install its pose, then obs and reward.

        A reset skips the body integration (there is nothing to advance: the disc was
        just placed). The reward launch is gated on ``full_pass`` rather than ``is_step``
        — a standalone ``reset()``/``reset_at()`` is a full pass and has to recompute
        ``gap``/``caged``/``reward``, while the mid-step obs-only auto-reset must not
        touch them. The gating is host-side and constant inside a capture (a captured
        pass is always ``STEP``, i.e. ``full_pass == 1``).

        The ``_install_obstacles`` in the middle is an ordinary line here, running inside
        ``wp.ScopedCapture`` on a step — see its docstring for why that is legal.
        """
        st = self.world.state_wp()  # one wrap for every launch in this pass
        if pass_.is_step:
            self._launch_body(st)
            self._install_obstacles()  # updated disc pose for the next step
        self._launch_obs(st)
        if pass_.full_pass:
            self._launch_reward(st)

    def _launch_body(self, st) -> None:
        w = self.world
        scalar = w.wp_dtype
        wpb = self._wp
        wp.launch(
            concrete(caging_body_kernel, scalar),
            dim=w.n_envs,
            inputs=[
                st.pos,
                st.vel,
                wp.int32(self.n_agents),
                scalar(self._inv_agents),
                scalar(self.agent_radius),
                scalar(self.disc_radius),
                scalar(self.contact_margin),
                scalar(self.contact_k),
                scalar(self.contact_c),
                scalar(self.disc_mass),
                scalar(self.linear_damping),
                scalar(self.escape_accel),
                scalar(self.max_disc_speed),
                scalar(self.dt),
                scalar(self._disc_bound),
            ],
            outputs=[wpb["disc_pos"], wpb["disc_vel"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_obs(self, st) -> None:
        w = self.world
        disc_pos, disc_vel = self._wp["disc_pos"], self._wp["disc_vel"]
        obs = self._wp["obs"]
        # No per-call-varying argument at all: pointer-stable persistent-mode handles.
        launch = self._obs_launch.get(
            concrete(caging_obs_kernel, w.wp_dtype),
            dim=(w.n_envs, self.n_agents),
            inputs=[st.pos, st.vel, disc_pos, disc_vel, wp.int32(self.n_agents)],
            outputs=[obs],
            device=w.device,
            key=(
                w.n_envs,
                self.n_agents,
                ptr_key(st.pos),
                ptr_key(st.vel),
                ptr_key(disc_pos),
                ptr_key(disc_vel),
                ptr_key(obs),
            ),
        )
        launch.launch()

    def _launch_reward(self, st) -> None:
        w = self.world
        scalar = w.wp_dtype
        disc_pos = self._wp["disc_pos"]
        gap, reward, caged = self._wp["gap"], self._wp["reward"], self._wp["caged"]
        launch = self._reward_launch.get(
            concrete(caging_reward_kernel, scalar),
            dim=w.n_envs,
            inputs=[
                st.pos,
                disc_pos,
                wp.int32(self.n_agents),
                scalar(self.agent_radius),
                scalar(self.disc_radius),
                scalar(self.cage_radius),
                scalar(self.capture_radius),
                scalar(self.gap_threshold),
                scalar(self.gap_reward),
                scalar(self.radius_factor),
                scalar(self.contact_penalty),
                scalar(self.caged_reward),
                scalar(self.spacing_factor),
                scalar(self._even_gap),
            ],
            outputs=[gap, reward, caged],
            device=w.device,
            key=(
                w.n_envs,
                ptr_key(st.pos),
                ptr_key(disc_pos),
                self.n_agents,
                self.cage_radius,
                self.capture_radius,
                self.gap_threshold,
                self.gap_reward,
                self.radius_factor,
                self.contact_penalty,
                self.caged_reward,
                self.spacing_factor,
                self._even_gap,
                ptr_key(gap),
                ptr_key(reward),
                ptr_key(caged),
            ),
        )
        launch.launch()

    # -------------------------------------------------- the torch oracle itself

    def _refresh(self, integrate: bool = True) -> None:
        w = self.world
        pos, vel = w.state.pos, w.state.vel  # [n_envs, n_agents, 2]

        if integrate:
            q = self.disc_pos[:, 0]  # [E, 2]
            dv = self.disc_vel[:, 0]
            rel = q.unsqueeze(1) - pos  # agent -> disc [E, A, 2]
            dist = rel.norm(dim=-1).clamp(min=1e-9)  # [E, A]
            n_hat = rel / dist.unsqueeze(-1)
            overlap = (self.agent_radius + self.disc_radius + self.contact_margin) - dist
            active = (overlap > 0).to(w.dtype)
            fmag = self.contact_k * overlap.clamp(min=0.0)
            # damping along the contact normal (disc approaching agent)
            rel_vel = dv.unsqueeze(1) - vel  # [E, A, 2]
            vn = (rel_vel * n_hat).sum(-1)
            force = (fmag - self.contact_c * vn).unsqueeze(-1) * n_hat * active.unsqueeze(-1)
            f_total = force.sum(dim=1)  # [E, 2] net contact force on the disc
            # Escape drift: away from the mean unit bearing to the agents. ``n_hat``
            # points agent -> disc, so the mean of the disc -> agent units is ``-n_hat``
            # and the acceleration is ``-escape * (-mean) = +escape * mean(n_hat)``.
            escape = self.escape_accel * (n_hat.sum(dim=1) * self._inv_agents)

            dt = self.dt
            new_vel = (dv + (f_total / self.disc_mass + escape) * dt) * (
                1.0 - self.linear_damping * dt
            )
            # Speed cap, applied to the integrated velocity before it moves the position
            # — same place, same order as the kernel. Below the cap the scale is exactly
            # 1.0, so this is bit-identical to the uncapped velocity there.
            speed = new_vel.norm(dim=-1, keepdim=True)
            new_vel = new_vel * (self.max_disc_speed / speed.clamp(min=1e-9)).clamp(max=1.0)
            b = self._disc_bound
            new_pos = (q + new_vel * dt).clamp(-b, b)
            self.disc_vel = new_vel.unsqueeze(1)
            self.disc_pos = new_pos.unsqueeze(1)
            self._install_obstacles()  # for the next step

        centre = self.disc_pos[:, 0]  # [E, 2]
        rel_a = pos - centre.unsqueeze(1)  # disc -> agent [E, A, 2]
        dist = rel_a.norm(dim=-1)  # [E, A]
        bearing = torch.atan2(rel_a[..., 1], rel_a[..., 0])  # [E, A] in (-pi, pi]

        # Largest circular gap: sort the bearings, difference consecutive pairs, and
        # close the circle with the wrap-around gap from the last bearing back round to
        # the first. That last term is the one that is easy to forget, and it is also the
        # only gap that exists at all when there is a single agent — with n_agents == 1
        # the ``diff`` below is empty and the wrap term is exactly 2*pi, which is the
        # right answer (a lone agent cages nothing).
        ordered, _ = torch.sort(bearing, dim=-1)
        steps = ordered[:, 1:] - ordered[:, :-1]  # [E, A-1]
        wrap = (ordered[:, :1] - ordered[:, -1:]) + TWO_PI  # [E, 1]
        max_gap = torch.cat([steps, wrap], dim=-1).max(dim=-1).values  # [E]

        # Per-agent local gaps, for the observation: the bearing distance to the nearest
        # other agent counter-clockwise (``delta[e, i, j] = (b_j - b_i) mod 2*pi``) and
        # clockwise (its transpose). The diagonal is masked to a full turn so an agent
        # never reports a zero gap to itself.
        delta = torch.remainder(bearing.unsqueeze(1) - bearing.unsqueeze(2), TWO_PI)
        eye = torch.eye(self.n_agents, device=w.device, dtype=torch.bool)
        delta = delta.masked_fill(eye, TWO_PI)
        gap_ccw = delta.min(dim=-1).values  # [E, A]
        gap_cw = delta.min(dim=-2).values  # [E, A]

        caged = (max_gap < self.gap_threshold) & (dist.max(dim=-1).values <= self.capture_radius)
        bonus = self.caged_reward * caged.to(w.dtype)
        shared = self.gap_reward * (self.gap_threshold - max_gap) + bonus  # [E]
        ring = (dist - self.cage_radius).abs()
        pen = (self.agent_radius + self.disc_radius - dist).clamp(min=0.0)
        # Dense angular shaping, per agent, towards an even share of the circle. Written
        # in this order (spacing first, then the two subtractions) because the fused
        # reward kernel folds the same four terms in the same order.
        d_ccw = gap_ccw - self._even_gap
        d_cw = gap_cw - self._even_gap
        spacing = -self.spacing_factor * (d_ccw * d_ccw + d_cw * d_cw)
        per_agent = spacing - self.radius_factor * ring - self.contact_penalty * pen  # [E, A]

        self._cache = {
            "max_gap": max_gap,
            "caged": caged,
            "shared": shared,
            "per_agent": per_agent,
            "disc_rel": centre.unsqueeze(1) - pos,
            "disc_vel": self.disc_vel[:, 0].unsqueeze(1).expand(-1, w.n_agents, -1),
            "gap_ccw": gap_ccw,
            "gap_cw": gap_cw,
        }

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["obs"]
        c = self._cache
        s = self.world.state
        return torch.cat(
            [
                s.pos,
                s.vel,
                c["disc_rel"],
                c["disc_vel"],
                c["gap_ccw"].unsqueeze(-1),
                c["gap_cw"].unsqueeze(-1),
            ],
            dim=-1,
        )

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        return self._cache["per_agent"][:, agent_idx]

    def global_reward(self) -> torch.Tensor:
        return self._cache["shared"]

    def rewards(self) -> torch.Tensor:
        return self.fb["reward"] if self.fused_active else super().rewards()

    def done(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["caged_bool"]
        return self._cache["caged"]

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {
                "max_gap": self.fb["gap"],
                "caged": self.fb["caged_bool"],
                "disc_pos": self.disc_pos,
            }
        return {
            "max_gap": self._cache["max_gap"],
            "caged": self._cache["caged"],
            "disc_pos": self.disc_pos,
        }
