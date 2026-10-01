"""Shepherding: dog agents drive non-cooperative sheep into a pen.

``n_agents`` "dogs" (the only things with an action) must herd ``n_sheep`` sheep into a
scored disc. The sheep flee from the dogs and cohere with each other, so a dog that simply
chases scatters the flock; the dogs have to approach from the far side and *push*. That
makes this the only scenario in the repo where the environment is **reactive** — it pushes
back — and the only one whose difficulty comes from the other bodies having a policy of
their own rather than from geometry or from a partner's ambiguity.

Three structural decisions, each of which looks like an omission until you try the
alternative:

**1. The sheep are not agents.** :class:`~swarp.core.environment.Environment` validates
``actions.shape == (n_envs, n_agents, act_dim)`` and passes the caller's tensor straight
through, so there is no injection point for internally-generated actions: sheep-as-agents
would have to be *emitted* by the policy, would appear in the action spec, and would get
their own observation and reward rows that every PPO loss then averages over (``ClipPPOLoss``
has no per-agent masking). And it would not help, because of decision 2.

**2. The sheep are scenario-owned, ``kind``-free ``CIRCLE`` obstacles, integrated here.**
There is no external-force hook for a ``MOVABLE`` obstacle: ``bodies.py``'s
``obstacle_dynamics_kernel`` derives force purely from agent reaction and body-body
contact, and its signature has no external-force slot — so a flee force could not be
applied to an engine-integrated body at all. What *is* load-bearing is that an
immovable-tagged circle still **pushes agents**: ``collisions.py``'s ``_static_forces``
dispatches on the obstacle's *shape* and never reads ``kind``. So the dogs feel the sheep
through the engine's own soft contact while this scenario integrates the sheep itself,
exactly as :mod:`swarp.scenarios.transport` does for its packages.

Two corrections to that template are made here deliberately. The obstacle spec installs
``vel`` (transport omits it), so the dog-sheep damper sees the *closing* velocity rather
than the dog's absolute velocity — the error ``collisions.py`` warns about, which scales
with ``contact_c`` and is large at the stiffness this scenario needs. And
``neighbor_radius`` is **not** inflated for the sheep: obstacles are scanned linearly by
``_static_forces``, never looked up in the neighbor grid, so inflating it is pure cost.

**3. The pen is a scored region, not walls.** Learnability is the smaller half of the
reason. The bigger half: ``bodies.py``'s ``_body_body`` only runs for ``MOVABLE`` bodies,
so sheep-vs-wall contact would have to be reimplemented **twice** — once in the fused
kernel, once in the torch oracle, which must stay an independent implementation — for
three box SDFs. That is the largest new parity surface in the whole design, bought for a
much harder task. ``pen_walls=True`` is reserved for that future extension and raises;
please do not "simplify" it into existence. With a region pen the obstacle set is
sheep-only, ``[E, K, 2]``, contiguous, and directly aliasable by a retained spec.

Two deliberate deviations from the textbook Strömbom sheep model, stated here so nobody
"fixes" them back:

* **Cohesion targets the centroid of all the other sheep**, not Strömbom's *k*-nearest
  local centre of mass. A *k*-nearest centre needs an in-kernel partial selection, and
  there is no precedent in this repo for a per-thread local array — Warp's only option is
  a ``wp.types.vector(length=N)`` needing a compile-time ``MAX_SHEEP`` cap. The all-others
  centroid is O(K), branch-free, deterministic, and trivially identical in torch.
* **No per-step noise; a per-episode per-sheep** ``drift`` **vector instead.** Per-step RNG
  inside the fused kernel would have to come either from a scalar seed argument — which a
  CUDA graph bakes in *by value*, so the noise freezes at replay, silently — or from
  ``World.seed_state``, which the torch oracle cannot reproduce, destroying parity by
  construction. A ``[E, K, 2]`` buffer written once per episode by the reset kernel and
  read identically by both paths has neither problem. There is deliberately no
  ``noise_k`` parameter.

Reward, per env (global) plus one per-agent term::

    c          = mean_s q_s                       # sheep centroid
    mean_dist  = mean_s |q_s - pen|
    spread     = sqrt(mean_s |q_s - c|^2)         # RMS, not max
    all_in     = all_s (|q_s - pen| < pen_radius)

    global:    pos_shaping_factor    * (prev_dist   - mean_dist)
             + spread_shaping_factor * (prev_spread - spread)
             + pen_reward * float(all_in)
             + time_penalty
    per-agent: collision_penalty * dog_dog_touching_count
    done     = all_in

Observation per dog (``obs_dim = 10 + 4 * n_sheep + 2 * (n_agents - 1)``): own pos and
vel, the pen and the sheep centroid relative to it, the flock's RMS spread, the fraction
of sheep already penned, then every sheep's relative position and velocity, then the other
dogs' relative positions. The per-sheep block is O(K) in the observation width; past
``n_sheep > 8`` it should give way to an aggregate-only form (centroid / spread / nearest
sheep), and that is a **documented cutover for a caller to make**, not a runtime branch —
a width that changes shape under a parameter is a policy that cannot be reloaded.
"""

from __future__ import annotations

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
from swarp.scenarios.fused import Buf, FusedPass, FusedScenario
from swarp.scenarios.shepherding_kernels import (
    N_OBJ,
    shepherding_force_kernel,
    shepherding_integrate_kernel,
    shepherding_obs_kernel,
    shepherding_reset_kernel,
    shepherding_reward_kernel,
)


class ShepherdingScenario(FusedScenario):
    # Both paths integrate the sheep with **one** semi-implicit Euler step at the same
    # ``dt`` and the same force law, so they differ only by the reduction order of the
    # per-dog and per-sheep sums (and by ``sqrt`` reassociation) — not by a substep
    # mismatch, which is what makes push-t's 1e-2 necessary there and unnecessary here.
    # This bound is *conditional on the per-episode-drift design*: with per-step in-kernel
    # RNG the two paths would not be comparable at any tolerance. The residual risk that
    # no tolerance covers is discrete: ``in_pen`` is K threshold tests and ``done`` is
    # their AND, and the harness compares both exactly.
    parity_rtol: float = 1e-4
    parity_atol: float = 1e-5

    def __init__(
        self,
        n_agents: int = 3,
        n_sheep: int = 5,
        agent_radius: float = 0.05,
        sheep_radius: float = 0.04,
        sheep_mass: float = 1.0,
        world_size: float = 1.0,
        max_speed: float = 1.0,
        sheep_max_speed: float | None = None,
        pen_radius: float = 0.25,
        pen_walls: bool = False,
        flee_radius: float = 0.3,
        flee_k: float = 8.0,
        sep_radius: float | None = None,
        sep_k: float = 12.0,
        coh_k: float = 6.0,
        drift_max: float = 1.0,
        linear_damping: float = 8.0,
        contact_k: float = 2000.0,
        contact_c: float = 40.0,
        contact_margin: float = 0.01,
        pos_shaping_factor: float = 1.0,
        spread_shaping_factor: float = 0.5,
        pen_reward: float = 5.0,
        time_penalty: float = -0.01,
        collision_penalty: float = -1.0,
    ) -> None:
        if n_sheep < 1:
            raise ValueError(f"shepherding needs at least one sheep; got n_sheep={n_sheep}")
        if pen_walls:
            raise NotImplementedError(
                "pen_walls=True is reserved for a future extension and is not implemented. "
                "A walled pen means sheep-vs-wall contact, and bodies.py's _body_body only "
                "runs for MOVABLE bodies — so it would have to be written twice, once in "
                "the fused kernel and once independently in the torch parity oracle, for "
                "three box SDFs. That is the largest new parity surface in this design, "
                "bought for a much harder task. The pen is a scored region instead."
            )
        self.n_agents = n_agents
        self.n_sheep = n_sheep
        self.agent_radius = agent_radius
        self.sheep_radius = sheep_radius
        self.sheep_mass = sheep_mass
        self.world_size = world_size
        self.max_speed = max_speed
        # The sheep must be **catchable**: a flock that outruns the dogs turns herding
        # into an unwinnable pursuit and the shaping term never moves.
        self.sheep_max_speed = (
            0.6 * max_speed if sheep_max_speed is None else float(sheep_max_speed)
        )
        self.pen_radius = pen_radius
        self.pen_walls = False
        self.flee_radius = flee_radius
        self.flee_k = flee_k
        self.sep_radius = 3.0 * sheep_radius if sep_radius is None else float(sep_radius)
        self.sep_k = sep_k
        self.coh_k = coh_k
        self.drift_max = drift_max
        # Note this is 8.0, not transport's 0.5. The sheep force law is a *steering* law,
        # not a momentum model: the steady-state speed under a constant force F is
        # F * (1 - lam*dt) / lam, so at lam = 0.5 every gain above ~0.3 saturates the speed
        # clamp and the whole force law collapses to "flee at max speed / don't". A heavy
        # damping puts the interesting range of every gain (flee, cohesion, separation,
        # drift) inside the clamp, where the sheep behaviour is actually a function of
        # distance. Retune the four gains together if you change it.
        self.linear_damping = linear_damping
        # contact_k/contact_c ARE the engine's WorldConfig.collision_k/collision_c — the
        # same spring-damper law, named for the dog<->sheep contact that dominates here.
        # contact_margin is NOT the engine's collision_margin (derived from agent_radius
        # in make_world): it is this scenario's own dog<->sheep activation gap.
        self.contact_k = contact_k
        self.contact_c = contact_c
        self.contact_margin = contact_margin
        # NOTE ON MAGNITUDES. Measured with ``info()["multiobj_reward"]`` over 100 steps of
        # a uniform-random policy, 256 envs, the defaults below (per-agent per-step means
        # of |term|), because a scenario whose terms are guessed rather than measured wastes
        # GPU hours before anyone notices:
        #
        #   difficulty=1.0   pos +0.0035  spread +0.0030  pen +0.0014  time 0.0100  col 0.0038
        #   difficulty=0.4   pos +0.0017  spread +0.0014  pen +0.0205  time 0.0100  col 0.0039
        #   difficulty=0.0   pos +0.0013  spread +0.0008  pen +5.0000  time 0.0100  col 0.0096
        #
        # Nothing dominates the objective: the largest ratio is the constant ``time_penalty``
        # at 2.9x a single shaping column, and 1.5x the two of them together — the same
        # ballpark give-way measured (+0.012 shaping against -0.01 time), and far from the
        # 22x imbalance that made give-way's first-cut wall penalty unlearnable. The shaping
        # numbers are small in absolute terms because they are *rates*: the flock moves at
        # most ``sheep_max_speed * dt`` per step, so a solved episode integrates to roughly
        # +0.7 of shaping against +5 of ``pen_reward`` and -2 of accumulated time — the same
        # sparse-bonus-dominant shape as navigation and give-way.
        #
        # If you raise ``time_penalty`` in magnitude, raise the two shaping factors with it:
        # a per-step constant the policy can only escape by *finishing* is fine while the
        # bonus is reachable and a pure discouragement while it is not.
        self.pos_shaping_factor = pos_shaping_factor
        self.spread_shaping_factor = spread_shaping_factor
        self.pen_reward = pen_reward
        self.time_penalty = time_penalty
        self.collision_penalty = collision_penalty

        self._difficulty = 1.0
        # Set by ``make_world``; None until then so the property setter is usable before
        # there is a device to write to.
        self._difficulty_t: torch.Tensor | None = None

    # -------------------------------------------------------------- curriculum

    @property
    def difficulty(self) -> float:
        """Curriculum knob a trainer pokes between batches (Push-T's mechanism).

        ``0`` spawns the sheep in a tight cluster **centred on the pen** and leaves them
        **inert** (zero drift, zero flee gain), so ``all_in`` is true on step 1 and both
        ``pen_reward`` and the terminal are experienced immediately — which is the single
        reason Push-T's curriculum works, and it makes the task at ``f = 0`` essentially
        transport. ``1`` spawns them uniformly, pushed clear of the pen, with full drift
        and full flee. Clamped to ``[0, 1]``.

        A property rather than a plain attribute, and the value lives in a one-element
        device tensor the reset kernel reads **by pointer**: a scalar kernel argument
        would be baked into a capture by value, and (independently) baked into the
        :class:`~swarp.core.cached_launch.CachedLaunch` key, so a mid-training change
        would either stop taking effect or repack the launch. Assigning the attribute
        therefore "just works"; a bare float would not have.

        ``pen_radius`` is deliberately **not** annealed: moving the success criterion moves
        the reward under the value function.
        """
        return self._difficulty

    @difficulty.setter
    def difficulty(self, value: float) -> None:
        self._difficulty = min(max(float(value), 0.0), 1.0)
        if self._difficulty_t is not None:
            # One tiny host->device write per curriculum change (not per step); a
            # ``fill_`` rather than an indexed assignment so nothing reads back.
            self._difficulty_t.fill_(self._difficulty)

    # ------------------------------------------------------------------ world

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
        # Push-T's derivation. A velocity-mode agent has no contact memory: each substep
        # its velocity is overwritten by the command plus f*sub_dt/m, so holding it off a
        # sheep needs f >= m*max_speed/sub_dt, i.e. an equilibrium penetration of
        # max_speed/(k*sub_dt). Saturating at twice that bounds the impulse a deep sweep
        # can inject while leaving the holding force intact. sub_dt shrinks the depth
        # linearly, which is why **substeps >= 8 at dt=0.05** is the practical floor: any
        # coarser and a dog walks through the flock instead of pushing it.
        sub_dt = dt / max(1, substeps)
        self.max_overlap = 2.0 * self.max_speed / (self.contact_k * sub_dt)
        # No neighbor_radius override: the sheep are obstacles, scanned linearly by
        # _static_forces rather than looked up in the neighbor grid, so their size must not
        # inflate the dog<->dog reach (the default 2*r + collision_margin). Transport
        # inflates it and pays for nothing.
        cfg = WorldConfig(
            collisions=True,
            collision_k=self.contact_k,
            collision_c=self.contact_c,
            collision_margin=margin,
            bounds=(-self.world_size, self.world_size, -self.world_size, self.world_size),
            bounds_mode="soft",
            max_neighbors=min(32, max(4, self.n_agents)),
            contact_max_overlap=self.max_overlap,
        ).override_with(world_config)
        self.dt = dt
        self.world = World(
            cfgs, cfg, n_envs=n_envs, device=device, dt=dt, substeps=substeps, dtype=dtype
        )

        tt = {"device": device, "dtype": dtype}
        # All persistent state is allocated here (n_envs is known) so the fused spec can
        # adopt it, and written in place by every reset so the handles stay valid.
        self.sheep_pos = torch.zeros(n_envs, self.n_sheep, 2, **tt)
        self.sheep_vel = torch.zeros(n_envs, self.n_sheep, 2, **tt)
        self.pen_pos = torch.zeros(n_envs, 2, **tt)
        self.drift = torch.zeros(n_envs, self.n_sheep, 2, **tt)
        self.evade = torch.zeros(n_envs, **tt)
        self._prev_dist: torch.Tensor | None = None
        self._prev_spread: torch.Tensor | None = None
        self._cache: dict[str, torch.Tensor] | None = None
        # The curriculum value, by pointer — see :attr:`difficulty`. Allocated once and
        # never reallocated, so the handle stays valid for the world's lifetime.
        self._difficulty_t = torch.zeros(1, **tt)
        self._difficulty_wp = wp.from_torch(self._difficulty_t, dtype=self.world.wp_dtype)
        self.difficulty = self._difficulty  # sync the device value

        # ONE retained obstacle spec over sheep_pos/sheep_vel, re-installed rather than
        # rebuilt: both ``resolve`` and ``any_movable`` memoize, so a re-install allocates
        # nothing and costs no device->host sync, which is what makes the mid-capture
        # install in :meth:`launch_fused` legal. ``kind`` is left None on purpose — the
        # engine must NOT integrate these — and ``vel`` is installed so the dog<->sheep
        # damper sees the closing velocity.
        self._sheep_radius_t = torch.full((self.n_sheep,), self.sheep_radius, **tt)
        self._obstacles = Obstacles(
            self.sheep_pos.detach(), self._sheep_radius_t, vel=self.sheep_vel.detach()
        ).resolve(device, dtype)

        # Teammate index table for the torch observation: row ``a`` is every dog but ``a``,
        # ascending. Fixed by n_agents, so it is built once here rather than rebuilt on
        # every ``observations()`` call (pusht.py does the same).
        idx = torch.arange(self.n_agents, device=device)
        others = idx.unsqueeze(0).expand(self.n_agents, -1)[idx.unsqueeze(1) != idx.unsqueeze(0)]
        self._others = others.view(self.n_agents, self.n_agents - 1)
        self._eye = torch.eye(self.n_sheep, device=device, dtype=torch.bool)

        # Cached, repack-once launches for the eager paths — see swarp/core/cached_launch.py.
        self._reset_launch = CachedLaunch()
        self._obs_launch = CachedLaunch()
        self._reward_launch = CachedLaunch()
        return self.world

    @property
    def obs_dim(self) -> int:
        return 10 + 4 * self.n_sheep + 2 * (self.n_agents - 1)

    # ------------------------------------------------------------------ reset

    def reset_world(self, env_mask: torch.Tensor | None = None, *, obs_only: bool = False) -> None:
        """Masked reset in one Warp launch (see :mod:`swarp.scenarios.shepherding_kernels`).

        The sheep/pen/drift/evade buffers are written **in place** by the kernel, so the
        fused path's cached Warp handles and the whole-step graph stay valid across resets;
        the grad path's ``_refresh`` still reassigns ``sheep_pos``/``sheep_vel`` (fresh
        tensors for the tape), which the framework's handle resync catches on the next
        no-grad step.
        """
        w = self.world
        lim = self.world_size - 2.0 * self.agent_radius
        slim = self.world_size - self.sheep_radius
        plim = self.world_size - self.pen_radius
        mask, use_mask = self.reset_mask_wp(env_mask)
        st = w.state_wp()
        scalar = w.wp_dtype
        vec2 = VEC2[scalar]
        if self.fused_active:
            # Reuse the fused spec's cached, pointer-resynced handles instead of
            # re-wrapping these watched tensors on every reset. ``sync_fused_handles``
            # must run first: it is what notices a grad-path reassignment and rebuilds the
            # handle before this kernel writes through it.
            self.ensure_fused()
            self.sync_fused_handles()
            sheep_pos = self._wp["sheep_pos"]
            sheep_vel = self._wp["sheep_vel"]
            pen = self._wp["pen"]
            drift = self._wp["drift"]
            evade = self._wp["evade"]
        else:
            sheep_pos = wp.from_torch(self.sheep_pos.contiguous(), dtype=vec2)
            sheep_vel = wp.from_torch(self.sheep_vel.contiguous(), dtype=vec2)
            pen = wp.from_torch(self.pen_pos.contiguous(), dtype=vec2)
            drift = wp.from_torch(self.drift.contiguous(), dtype=vec2)
            evade = wp.from_torch(self.evade.contiguous(), dtype=scalar)
        seed = wp.int32(w.next_kernel_seed())
        with torch_stream_scope(w.device):
            launch = self._reset_launch.get(
                concrete(shepherding_reset_kernel, scalar),
                dim=w.n_envs,
                inputs=[
                    mask,
                    use_mask,
                    seed,
                    self._difficulty_wp,
                    scalar(lim),
                    scalar(slim),
                    scalar(plim),
                    # Small enough that the whole cluster lands inside the pen at f=0:
                    # the corner of the box is 0.5*sqrt(2) = 0.71 pen radii out.
                    scalar(0.5 * self.pen_radius),
                    scalar(2.0 * self.pen_radius),
                    scalar(self.drift_max),
                    wp.int32(self.n_agents),
                    wp.int32(self.n_sheep),
                    st.pos,
                    st.vel,
                    st.speed,
                    sheep_pos,
                    sheep_vel,
                    pen,
                    drift,
                    evade,
                ],
                device=w.device,
                key=(
                    w.n_envs,
                    ptr_key(mask),
                    ptr_key(self._difficulty_wp),
                    lim,
                    slim,
                    plim,
                    self.pen_radius,
                    self.drift_max,
                    self.n_agents,
                    self.n_sheep,
                    ptr_key(st.pos),
                    ptr_key(st.vel),
                    ptr_key(st.speed),
                    ptr_key(sheep_pos),
                    ptr_key(sheep_vel),
                    ptr_key(pen),
                    ptr_key(drift),
                    ptr_key(evade),
                ),
            )
            launch.set_param_by_name("use_mask", use_mask)
            launch.set_param_by_name("seed", seed)
            launch.launch()
        w.mark_pos_dirty()

        self._install_obstacles()
        if not self.fused_active:
            self._prev_dist = None
            self._prev_spread = None
        self.finish_reset(env_mask, obs_only=obs_only)

    def _install_obstacles(self) -> None:
        """Hand the current sheep poses **and velocities** to the engine as circles.

        Called from inside the captured whole-step graph (see :meth:`launch_fused`), so it
        must stay in the capture-safe regime of
        :meth:`swarp.core.stepper.Stepper.set_obstacles`: an unchanged obstacle count, and
        a spec carrying no movable obstacles and no 1-D ``angle`` to broadcast. Both hold —
        ``kind`` and ``angle`` are ``None`` — and the spec is *re-installed*, not rebuilt,
        so ``resolve`` returns ``self`` and ``any_movable`` never reaches the device.

        The retained spec is rebuilt only when ``sheep_pos``/``sheep_vel`` is *reassigned*,
        which is the grad path integrating the flock in torch (it needs fresh tensors for
        the tape). That is a host-side pointer compare, and on the captured path the
        pointers never move, so the rebuild branch cannot fire inside a capture.
        """
        if (
            self._obstacles.pos.data_ptr() != self.sheep_pos.data_ptr()
            or self._obstacles.vel.data_ptr() != self.sheep_vel.data_ptr()
        ):
            self._obstacles = Obstacles(
                self.sheep_pos.detach(), self._sheep_radius_t, vel=self.sheep_vel.detach()
            ).resolve(self.world.device, self.world.dtype)
        self.world.set_obstacles(self._obstacles)

    # --------------------------------------------------------- fused fast path

    def fused_spec(self, n_envs: int) -> tuple[Buf, ...]:
        ne, na, k = n_envs, self.n_agents, self.n_sheep
        return (
            Buf("obs", (ne, na, self.obs_dim)),
            Buf("reward", (ne, na)),
            Buf("multiobj", (ne, na, N_OBJ)),
            Buf("touch", (ne, na)),
            Buf("dist", (ne,)),
            Buf("spread", (ne,)),
            Buf("penfrac", (ne,)),
            Buf("done", (ne,), "uint8", bool_view=True),
            Buf("resetmask", (ne,), "uint8", reset_mask=True),
            # Net sheep force: framework-owned scratch handed from the force kernel to the
            # integrate kernel. Not a carry — it is fully rewritten before it is read.
            Buf("force", (ne, k, 2), "vec2"),
            # The flock, advanced in place by the integrate kernel and reassigned by the
            # grad path's torch integrator.
            Buf("sheep_pos", (ne, k, 2), "vec2", attr="sheep_pos", alloc="never", carry=True,
                watch=True),
            Buf("sheep_vel", (ne, k, 2), "vec2", attr="sheep_vel", alloc="never", carry=True,
                watch=True),
            # Per-episode scenario state written by the reset kernel. Not carries: no fused
            # *step* launch advances them.
            Buf("pen", (ne, 2), "vec2", attr="pen_pos", alloc="never"),
            Buf("drift", (ne, k, 2), "vec2", attr="drift", alloc="never"),
            Buf("evade", (ne,), attr="evade", alloc="never"),
            Buf("prev", (ne,), attr="_prev_dist", alloc="if_none", carry=True, watch=True),
            Buf("prevspread", (ne,), attr="_prev_spread", alloc="if_none", carry=True,
                watch=True),
        )

    def engine_carries(self) -> list[torch.Tensor]:
        """The engine's obstacle positions and velocities: ``_install_obstacles``
        overwrites them from inside the hook, and the *next* step's physics reads them."""
        views = self.world.obstacle_state_views()
        return [views[0], views[2]]

    def launch_fused(self, pass_: FusedPass) -> None:
        """Advance the flock, re-install it, then obs and reward.

        A reset skips both sheep kernels (there is nothing to advance: the flock was just
        placed) and passes ``advance_prev=0`` so the reward kernel *rebases* the two
        shaping baselines for the reset envs instead of differencing against a stale one.
        The ``_install_obstacles`` in the middle is an ordinary line here, running inside
        ``wp.ScopedCapture`` on a step — see its docstring for why that is legal.
        """
        st = self.world.state_wp()  # one wrap for every launch in this pass
        if pass_.is_step:
            self._launch_sheep(st)
            self._install_obstacles()  # new poses AND velocities for the next step
        self._launch_obs(st, full_pass=pass_.full_pass)
        self._launch_reward(advance_prev=pass_.advance_prev, full_pass=pass_.full_pass)

    def _launch_sheep(self, st) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        k = self.n_sheep
        inv_others = 1.0 / (k - 1) if k > 1 else 0.0
        wp.launch(
            concrete(shepherding_force_kernel, scalar),
            dim=(w.n_envs, k),
            inputs=[
                st.pos,
                st.vel,
                pk["sheep_pos"],
                pk["sheep_vel"],
                pk["drift"],
                pk["evade"],
                wp.int32(self.n_agents),
                wp.int32(k),
                scalar(self.agent_radius + self.sheep_radius + self.contact_margin),
                scalar(self.contact_k),
                scalar(self.contact_c),
                scalar(self.flee_radius),
                scalar(self.flee_k),
                scalar(self.sep_radius),
                scalar(self.sep_k),
                scalar(self.coh_k),
                scalar(inv_others),
            ],
            outputs=[pk["force"]],
            device=w.device,
            record_tape=False,
        )
        wp.launch(
            concrete(shepherding_integrate_kernel, scalar),
            dim=(w.n_envs, k),
            inputs=[
                pk["force"],
                scalar(self.sheep_mass),
                scalar(self.linear_damping),
                scalar(self.dt),
                scalar(self.sheep_max_speed),
                scalar(self.world_size - self.sheep_radius),
            ],
            outputs=[pk["sheep_pos"], pk["sheep_vel"]],
            device=w.device,
            record_tape=False,
        )

    def _launch_obs(self, st, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        sheep_pos, sheep_vel, pen = pk["sheep_pos"], pk["sheep_vel"], pk["pen"]
        resetmask, obs, touch = pk["resetmask"], pk["obs"], pk["touch"]
        launch = self._obs_launch.get(
            concrete(shepherding_obs_kernel, scalar),
            dim=(w.n_envs, self.n_agents),
            inputs=[
                st.pos,
                st.vel,
                sheep_pos,
                sheep_vel,
                pen,
                resetmask,
                wp.int32(self.n_agents),
                wp.int32(self.n_sheep),
                scalar(1.0 / self.n_sheep),
                scalar(self.pen_radius),
                scalar((2.0 * self.agent_radius) ** 2),
                wp.int32(full_pass),
            ],
            outputs=[obs, touch],
            device=w.device,
            key=(
                w.n_envs,
                self.n_agents,
                self.n_sheep,
                ptr_key(st.pos),
                ptr_key(st.vel),
                ptr_key(sheep_pos),
                ptr_key(sheep_vel),
                ptr_key(pen),
                ptr_key(resetmask),
                self.pen_radius,
                self.agent_radius,
                ptr_key(obs),
                ptr_key(touch),
            ),
        )
        launch.set_param_by_name("full_pass", wp.int32(full_pass))
        launch.launch()

    def _launch_reward(self, advance_prev: int, full_pass: int) -> None:
        w = self.world
        scalar = w.wp_dtype
        pk = self._wp
        launch = self._reward_launch.get(
            concrete(shepherding_reward_kernel, scalar),
            dim=w.n_envs,
            inputs=[
                pk["sheep_pos"],
                pk["pen"],
                pk["touch"],
                pk["resetmask"],
                wp.int32(self.n_agents),
                wp.int32(self.n_sheep),
                scalar(1.0 / self.n_sheep),
                scalar(self.pen_radius),
                scalar(self.pos_shaping_factor),
                scalar(self.spread_shaping_factor),
                scalar(self.pen_reward),
                scalar(self.time_penalty),
                scalar(self.collision_penalty),
                wp.int32(advance_prev),
                wp.int32(full_pass),
            ],
            outputs=[
                pk["prev"],
                pk["prevspread"],
                pk["reward"],
                pk["done"],
                pk["multiobj"],
                pk["dist"],
                pk["spread"],
                pk["penfrac"],
            ],
            device=w.device,
            key=(
                w.n_envs,
                self.n_agents,
                self.n_sheep,
                ptr_key(pk["sheep_pos"]),
                ptr_key(pk["pen"]),
                ptr_key(pk["touch"]),
                ptr_key(pk["resetmask"]),
                self.pen_radius,
                self.pos_shaping_factor,
                self.spread_shaping_factor,
                self.pen_reward,
                self.time_penalty,
                self.collision_penalty,
                ptr_key(pk["prev"]),
                ptr_key(pk["prevspread"]),
                ptr_key(pk["reward"]),
                ptr_key(pk["done"]),
                ptr_key(pk["multiobj"]),
                ptr_key(pk["dist"]),
                ptr_key(pk["spread"]),
                ptr_key(pk["penfrac"]),
            ),
        )
        launch.set_param_by_name("advance_prev", wp.int32(advance_prev))
        launch.set_param_by_name("full_pass", wp.int32(full_pass))
        launch.launch()

    # ---------------------------------------- torch reference path (parity oracle)

    def post_step_torch(self) -> None:
        self._refresh(integrate=True)

    def reset_torch(self, env_mask: torch.Tensor | None) -> None:
        self._refresh(reset_mask=env_mask, integrate=False)

    def _sheep_force(self) -> torch.Tensor:
        """Net force on every sheep ``[n_envs, n_sheep, 2]`` — the torch oracle.

        Written independently of :func:`~swarp.scenarios.shepherding_kernels.
        shepherding_force_kernel` (that is the whole point of an oracle), but term for term
        the same *quantity*: per-episode drift, flee, explicit spring-damper contact,
        separation, and cohesion toward the all-others centroid. The contact term is spelt
        out rather than routed through the engine's ``pair_force``, whose implicit damping
        denominator and ``tanh`` saturation would be extra surface to reproduce here for no
        behavioural gain.
        """
        w = self.world
        eps = 1e-9
        q, vs = self.sheep_pos, self.sheep_vel  # [E, K, 2]
        pos, vel = w.state.pos, w.state.vel  # [E, A, 2]

        # ---- dogs: flee + contact reaction, summed over agents
        rel = q.unsqueeze(1) - pos.unsqueeze(2)  # dog->sheep [E, A, K, 2]
        d = rel.norm(dim=-1).clamp(min=eps)  # [E, A, K]
        n_hat = rel / d.unsqueeze(-1)
        flee = (
            self.evade.view(-1, 1, 1)
            * self.flee_k
            * (1.0 - d / self.flee_radius).clamp(min=0.0) ** 2
        )
        reach = self.agent_radius + self.sheep_radius + self.contact_margin
        overlap = reach - d
        rel_vel = vs.unsqueeze(1) - vel.unsqueeze(2)
        vn = (rel_vel * n_hat).sum(-1)
        contact = torch.where(
            overlap > 0.0,
            self.contact_k * overlap - self.contact_c * vn,
            torch.zeros_like(overlap),
        )
        f = ((flee + contact).unsqueeze(-1) * n_hat).sum(dim=1) + self.drift  # [E, K, 2]

        # ---- flock: separation from, cohesion toward, the other sheep
        rel_ss = q.unsqueeze(2) - q.unsqueeze(1)  # q_s - q_t, s on dim 1 [E, K, K, 2]
        d_ss = rel_ss.norm(dim=-1).clamp(min=eps)
        off = (~self._eye).to(w.dtype)  # zero the self term
        sep = self.sep_k * (1.0 - d_ss / self.sep_radius).clamp(min=0.0) ** 2 * off
        f = f + ((sep / d_ss).unsqueeze(-1) * rel_ss).sum(dim=2)
        if self.n_sheep > 1:
            centre = (q.sum(dim=1, keepdim=True) - q) / (self.n_sheep - 1)
            f = f + self.coh_k * (centre - q)
        return f

    def _refresh(self, reset_mask: torch.Tensor | None = None, integrate: bool = True) -> None:
        w = self.world
        if integrate:
            f = self._sheep_force()
            dt = self.dt
            v = (self.sheep_vel + f / self.sheep_mass * dt) * (1.0 - self.linear_damping * dt)
            sp = v.norm(dim=-1).clamp(min=1e-9)
            v = v * (self.sheep_max_speed / sp).clamp(max=1.0).unsqueeze(-1)
            b = self.world_size - self.sheep_radius
            self.sheep_vel = v
            # The arena clamp is the scenario's job — nothing in the engine bounds a
            # scenario-owned obstacle, and without it the flock walks out of the world.
            self.sheep_pos = (self.sheep_pos + v * dt).clamp(-b, b)
            self._install_obstacles()  # for the next step

        q, pen = self.sheep_pos, self.pen_pos.unsqueeze(1)  # [E, K, 2], [E, 1, 2]
        dists = (q - pen).norm(dim=-1)  # [E, K]
        mean_dist = dists.sum(dim=-1) / self.n_sheep
        centroid = q.sum(dim=1) / self.n_sheep  # [E, 2]
        spread = (
            (q - centroid.unsqueeze(1)).pow(2).sum(-1).sum(-1) / self.n_sheep
        ).sqrt()  # RMS
        in_pen = dists < self.pen_radius
        all_in = in_pen.all(dim=-1)
        pen_frac = in_pen.to(w.dtype).sum(dim=-1) / self.n_sheep

        if self._prev_dist is None:
            self._prev_dist = mean_dist.detach().clone()
        if self._prev_spread is None:
            self._prev_spread = spread.detach().clone()
        pos_term = (self._prev_dist - mean_dist) * self.pos_shaping_factor
        spr_term = (self._prev_spread - spread) * self.spread_shaping_factor
        if reset_mask is None:
            self._prev_dist = mean_dist.detach().clone()
            self._prev_spread = spread.detach().clone()
        else:
            zero = torch.zeros_like(pos_term)
            pos_term = torch.where(reset_mask, zero, pos_term)
            spr_term = torch.where(reset_mask, zero, spr_term)
            self._prev_dist = torch.where(reset_mask, mean_dist.detach(), self._prev_dist)
            self._prev_spread = torch.where(reset_mask, spread.detach(), self._prev_spread)

        # Dog-dog contact count, squared-distance compared so it matches the kernel's
        # discrete flag at the 2r boundary (the harness compares ``collisions`` exactly).
        pos = w.state.pos
        d2 = (pos.unsqueeze(2) - pos.unsqueeze(1)).pow(2).sum(-1)  # [E, A, A]
        touching = (d2 < (2.0 * self.agent_radius) ** 2).sum(dim=-1).to(w.dtype) - 1.0

        self._cache = {
            "mean_dist": mean_dist,
            "spread": spread,
            "centroid": centroid,
            "pen_frac": pen_frac,
            "all_in": all_in,
            "pos_term": pos_term,
            "spread_term": spr_term,
            "touching": touching,
        }

    # ------------------------------------------------------------ obs/rewards

    def observations(self) -> torch.Tensor:
        """Fully batched observations ``[n_envs, n_agents, obs_dim]``."""
        if self.fused_active:
            return self.fb["obs"]
        w = self.world
        s = w.state
        c = self._cache
        ne, na, k = w.n_envs, self.n_agents, self.n_sheep
        rel = self.sheep_pos.unsqueeze(1) - s.pos.unsqueeze(2)  # [E, A, K, 2]
        svel = self.sheep_vel.unsqueeze(1).expand(ne, na, k, 2)
        sheep_block = torch.cat([rel, svel], dim=-1).reshape(ne, na, 4 * k)
        others = (s.pos[:, self._others] - s.pos.unsqueeze(2)).flatten(2)  # [E, A, 2(A-1)]
        return torch.cat(
            [
                s.pos,
                s.vel,
                self.pen_pos.unsqueeze(1) - s.pos,
                c["centroid"].unsqueeze(1) - s.pos,
                c["spread"].view(ne, 1, 1).expand(ne, na, 1),
                c["pen_frac"].view(ne, 1, 1).expand(ne, na, 1),
                sheep_block,
                others,
            ],
            dim=-1,
        )

    def _global_terms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(pos_term, spread_term, pen_bonus)`` per env — the shared reward, factorized.

        ``rewards``, ``global_reward`` and ``multiobj_reward`` all read this one
        derivation, so the scalar reward and the objective vector cannot disagree about
        what a term is worth. ``time_penalty`` is a constant and stays a literal.
        """
        c = self._cache
        bonus = self.pen_reward * c["all_in"].to(self.world.dtype)
        return c["pos_term"], c["spread_term"], bonus

    def rewards(self) -> torch.Tensor:
        return self.fb["reward"] if self.fused_active else super().rewards()

    def agent_reward(self, agent_idx: int) -> torch.Tensor:
        return self.collision_penalty * self._cache["touching"][:, agent_idx]

    def global_reward(self) -> torch.Tensor:
        pos_term, spr_term, bonus = self._global_terms()
        return pos_term + spr_term + bonus + self.time_penalty

    def done(self) -> torch.Tensor:
        if self.fused_active:
            return self.fb["done_bool"]
        return self._cache["all_in"]

    def info(self) -> dict[str, Any]:
        if self.fused_active:
            return {
                "sheep_dist_to_pen": self.fb["dist"],
                "sheep_spread": self.fb["spread"],
                "pen_fraction": self.fb["penfrac"],
                "collisions": self.fb["touch"],
                "multiobj_reward": self.fb["multiobj"],
            }
        c = self._cache
        pos_term, spr_term, bonus = self._global_terms()
        # Per-agent objective vector [n_envs, n_agents, N_OBJ]: the five reward terms kept
        # separate so a trainer logs one column each and a shaping imbalance shows up on
        # iteration 1 rather than hour 3. Its sum over the last dim equals the scalar
        # per-agent reward exactly.
        col = self.collision_penalty * c["touching"]
        gl = [t.unsqueeze(-1).expand_as(col) for t in (pos_term, spr_term, bonus)]
        return {
            "sheep_dist_to_pen": c["mean_dist"],
            "sheep_spread": c["spread"],
            "pen_fraction": c["pen_frac"],
            "collisions": c["touching"],
            "multiobj_reward": torch.stack(
                [*gl, torch.full_like(col, self.time_penalty), col], dim=-1
            ),
        }
