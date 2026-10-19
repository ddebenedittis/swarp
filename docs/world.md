# World, contacts and sensing

`WorldConfig` (`swarp/core/config.py`) holds everything about the world that is not an agent: the contact law, the boundary, the neighbor search, and the integrator.
A scenario builds one in its `make_world`.

```python
from swarp import WorldConfig, World, Integrator

world_config = WorldConfig(
    bounds=(-1.0, 1.0, -1.0, 1.0),
    bounds_mode="soft",
    collision_k=100.0,
    collision_c=1.0,
    collision_margin=0.02,
    max_neighbors=32,
    integrator=Integrator.EULER,
)
world = World(agent_configs, world_config, n_envs=n_envs, device=device, dt=dt, substeps=1, dtype=dtype)
```

## Overriding a scenario's engine settings

The built-in scenarios compute their own `WorldConfig` — `bounds` from `world_size`, `neighbor_radius` from the contact reach — and most of its fields are not scenario constructor arguments.
Pass `world_config=` to `swarp.make` or `Environment` to override them without subclassing:

```python
import swarp
from swarp import WorldConfig, Integrator

env = swarp.make(
    "navigation",
    n_envs=4096,
    n_agents=8,
    world_config=WorldConfig(integrator=Integrator.RK4, bounds_mode="clamp", collision_k=50.0),
)
```

Only the fields you set **away from the `WorldConfig()` defaults** are applied, so the scenario keeps everything it computed (`WorldConfig.override_with` is the merge).
The one case this cannot express is forcing a field *back* to its default against a scenario that changed it — build the scenario's `World` yourself for that.

`Scenario.make_world` takes the same argument, and all twelve built-ins honour it (a test pins that).

## Soft contacts

Interactions are spring-damper penalties, not impulses: forces are computed from overlap and closing velocity and fed to the integrator.

- `collision_k` — spring stiffness.
- `collision_c` — normal damping.
- `collision_margin` — forces switch on within this gap around touching radii.
- `contact_max_overlap` — depth at which the spring saturates smoothly (`0` disables). A velocity-controlled agent has no contact memory and settles at an overlap of `v·m/(k·sub_dt)`, which at high stiffness is a violent impulse against a light movable body; saturating bounds it and leaves the shallow regime untouched.

The normal damping is **linearly implicit** and clamped repulsive. With the contact normal frozen over the substep, solving for the post-impulse normal velocity is one divide:

```text
f = max(0, (k*overlap - c*v_rel_n) / (1 + c*sub_dt/m))
```

That is unconditionally stable in `collision_c` — the explicit form required `sub_dt < m/c`, which capped usable stiffness — and it can never turn attractive, so a contact does not stick to a separating agent.
It is also a smooth rational function of the state, which differentiates better than the explicit form.

Contacts are **frictionless** everywhere in the engine: normal spring plus normal damping, no tangential term.

Forces are **gather-based** — each thread sums the forces on its own agent from its own neighbor list, obstacles and walls. Symmetric pairs are computed twice, which buys a deterministic, race-free result with no atomics.

### Box agents

A box agent (`AgentShape.BOX`, see [Agent shape](dynamics.md#agent-shape)) collides as the oriented rectangle it is, with the same normal law and the same constants as a disc.
Every contact it takes part in is a signed distance `s`, a unit normal `n` and a contact point `x`, from `obb_contact` in `swarp/core/collisions.py`:

- A separating-axis test over the four face normals runs first. The largest projection gap is a lower bound on the distance, so when it exceeds the contact reach the pair is skipped.
- Disjoint but within reach, `s` is the exact Euclidean gap. The closest pair of two disjoint rectangles always involves a vertex, so `s` is the minimum of the eight vertex-versus-other-box distances, and `n` and `x` come from the winning vertex (`x` midway across the gap).
- Overlapping, `s` is minus the smallest overlap over the four axes, `n` is that axis, and `x` sits on its reference face at the centre of the interval where the two projections onto the face overlap. A centred face-on push therefore has its contact point on the line through both centres.

Both branches reach zero at touching, so `s` is continuous through contact, and a `+` crossing of two thin boxes, where no corner lies inside the other box, is still caught by the axis test.
The force is `f = max(0, (k*overlap - c*v_n) / (1 + c*sub_dt/m)) * n` with `overlap = reach - s`, where `v_n` is the closing velocity of the two *material points* at `x`, `v + ω × r` on both sides.

| pair | geometry | reach |
|---|---|---|
| box agent – box agent | `obb_contact` | `collision_margin` |
| disc agent – box agent | `box_force` with the disc as the point | disc radius + margin |
| box agent – `BOX` obstacle | `obb_contact` | margin |
| box agent – `SEGMENT` obstacle | `obb_contact` against the degenerate box `(half_len, 0)` | capsule radius + margin, so the end caps are exactly round |
| box agent – `CIRCLE` obstacle | `box_force` with the circle centre as the point | circle radius + margin |

A box-box pair is evaluated with the lower agent index as box A on both threads, so the two threads compute one contact and the pair's forces cancel exactly for equal masses.
Contacts push a box agent's centre only: a box agent feels **no contact torque** yet.
Its heading is its own, driven by its dynamics model, so a diff-drive or bicycle box turns as it steers and a holonomic box keeps the heading it was reset with.

Every kernel a box world runs compiles without floating-point contraction (`fuse_fp=False`), which keeps the layered and fused paths bit-identical (see [Performance](performance.md#fused-substep-chain)).
A world without a box agent never launches them, so its step is the disc step unchanged.

## Bounds

`bounds=(x_min, x_max, y_min, y_max)`, or `None` for an unbounded world.

`bounds_mode="soft"`
: spring-damper walls, using the same contact constants. Agents decelerate into the boundary.

`bounds_mode="clamp"`
: positions are hard-clamped inside. Differentiable as a subgradient.

A box agent at heading `θ` reaches `ex = hx|cos θ| + hy|sin θ|` along x and `ey = hx|sin θ| + hy|cos θ|` along y from its centre, and both modes use those support extents in place of a disc's radius.
A soft wall damps the velocity of the support vertex touching it (on a face flush with the wall, the face centre), and the clamp insets each box by its extents at its *new* heading, so a box turned 45 degrees is held off the wall by its corner rather than by its circumscribed radius.

## Obstacles

An obstacle set is one batched `Obstacles` dataclass installed with `world.set_obstacles(...)`. Only `pos` `[E, N, 2]` and `radius` `[N]` are required; every other field defaults to a documented value rather than to whatever was installed before.

Three shapes (`ObstacleShape`):

| shape | geometry | notes |
|---|---|---|
| `CIRCLE` | centre + radius | the cheapest case |
| `BOX` | centre + `angle` + `half_extents` | analytic signed distance field, so a penetrating agent is still pushed out; `radius` is ignored |
| `SEGMENT` | centre + `angle` + half-length + `radius` | a capsule |

Two kinds (`ObstacleKind`):

`IMMOVABLE`
: infinite-mass scenery. Agents bounce off; it never moves. Drawn black.

`MOVABLE`
: carries `mass` and `inertia` and is integrated from the reaction of the very same agent contacts. `pos`/`angle`/`vel`/`ang_vel` are then its *initial* state. Drawn grey.

Obstacles also carry their own `vel`/`ang_vel`, and the damper uses the **closing** velocity — a moving obstacle that omitted them would be damped against the agent's absolute world velocity, applying drag unrelated to the contact.

## Movable and compound rigid bodies

```{image} _static/pusht.png
:alt: Push-T — a grey T-shaped compound body, its green target pose, and four agents
:width: 70%
:align: center
```

A movable body is integrated **inside the substep loop** (`swarp/core/bodies.py`), right after the force pass and before the agents integrate, from the reaction of the contacts the collision kernel just applied — Newton's third law of one shared force, so momentum is not invented at the interface.
Both sides therefore see the same state, and the pose an agent collides against is at most one *substep* old. A body advanced once per env step instead sweeps its surface across agents that cannot react until the next step, which shows up as visible interpenetration and lurching.

Like the agent side, this is gather-based — one thread per `(env, obstacle)`, looping over all agents, no atomics.

Several shapes that share a `body` id form one **compound** rigid body with a single pose, placed by their `body_offset` in the body frame. That is what makes Push-T's T work: a crossbar and a stem, one rigid pose, mass split by area and inertia from the parallel-axis theorem about the area centroid — which is what gives a contact above the centroid the torque that makes the orientation half of the task solvable.

`obstacle_linear_damping` and `obstacle_angular_damping` stand in for table friction: a pushed body settles at `Σf / (m·damping)` instead of accelerating without limit. Set them to 0 for a frictionless puck that coasts.
`obstacle_friction` (a Coulomb coefficient μ, 0 by default) instead puts movable bodies on a Coulomb floor: the weight `m·obstacle_gravity` is spread uniformly over each shape's footprint on an `obstacle_friction_grid` × `obstacle_friction_grid` lattice, and every lattice point resists its own sliding velocity with its share of μ·m·g, regularised below `obstacle_friction_speed` (5 mm/s).
Summed, that is the footprint's limit surface, so a body pushed off-centre turns about the pivot a real one would, and a sliding body stops within `v² / (2μg)`.
An impulse that would reverse a body's motion or spin sticks it instead (static friction), which keeps the stiff law stable at any substep.
`Stepper.set_body_friction` overrides μ·g per env.

A **box agent** pushes a body through the same oriented-box contact it feels itself: the body pass calls the very function the agent side summed and negates it, so action equals reaction to the bit, and the reaction acts at the contact point, so a box agent pushing a box body off-centre spins it.

Two limits worth knowing:

- **Box-box obstacle-obstacle contacts are not modelled.** Obstacles collide with each other for every pair in which at least one body is round; two boxes pass through one another, since the body-body pass evaluates an SDF against a round body. Agent-vs-box is exact for every shape, box agents included.
- **Body integration is not taped** (`record_tape=False`), so no gradient flows through a body's motion. A scenario that needs one keeps its own torch-side copy — see [Differentiability](differentiability.md).

## Neighbor search

Neighbor lists are padded to `max_neighbors` and built without a host sync. `neighbor_radius` defaults to the interaction reach `2·max_agent_radius + collision_margin`, and may not be set below it while collisions are on.

`neighbor_method` picks the backend:

| method | strategy | when |
|---|---|---|
| `"brute"` | per-env O(n_agents²) kernel | the usual multi-agent regime; linear in `n_envs` with a tiny constant |
| `"uniform_grid"` | batched radix-sort grid, each env owning a disjoint block of cells, 3×3 neighborhood query | large per-env populations across many envs |
| `"grid"` | one `wp.HashGrid` over all envs via a z-lift | very large per-env populations and few envs only |
| `"auto"` (default) | `uniform_grid` above 512 agents per env, `brute` otherwise | |

`wp.HashGrid` wraps cell *coordinates* modulo its dims, so the z-lifted envs alias into shared cells and query cost grows with `n_envs` — which is why `"grid"` is rarely the right answer here and is measured ~300× slower than brute force at 16k envs × 64 agents.
`grid_dim` (default 128) is that backend's bucket dimension per axis.

`uniform_bins` sets the cells per axis of the `"uniform_grid"` backend; `None` (the default) picks `~sqrt(n_agents)`, which keeps occupancy near one agent per cell.
Raising it is usually a mistake: every query zeroes `n_envs · bins²` `int32` cell offsets before the sort, so an over-fine grid spends more clearing empty cells than searching occupied ones — 128 bins with 8 agents is 16384 cells per env for 8 points.
A value that far above the heuristic warns.

The backends are tested to agree exactly. `World.neighbor_overflow()` flags any agent whose true in-radius count exceeded `max_neighbors`, so truncation is never silent, and `World.edge_index()` turns the same lists into a COO radius graph.

Neighbor construction is **not taped**: the neighbor *set* is a discrete structure, so gradients flow through contact geometry rather than through membership.
`neighbor_reuse` (on by default) lets the no-grad path reuse the list the previous step's post-step build already produced, turning two builds per step into one; the reused list is bit-identical to a fresh build on the same positions.

## Lidar

An opt-in differentiable range sensor lives in `swarp/sensors/lidar.py`. It is not part of the step — a scenario calls it in `observations()`.

```python
from swarp import Lidar

lidar = Lidar(n_rays=16, max_range=1.0, backend="warp")
ranges = lidar.scan(world)                 # [n_envs, n_agents, n_rays]
obs = torch.cat([base_obs, ranges], dim=-1)
```

Every shape in the world is scanned, with the geometry the collision step uses:

| target | geometry |
|---|---|
| disc agent, `CIRCLE` obstacle | closed-form ray-circle |
| box agent (`AgentShape.BOX`) | slab test against its `half_extents`, oriented by the agent's `theta` |
| `BOX` obstacle | slab test against its `half_extents` at its live angle |
| `SEGMENT` obstacle | the exact capsule: the rectangle `(half_len, radius)` around the core plus a circle of `radius` at each end |

A ray that hits nothing returns `max_range`, an agent never sees itself, and a ray whose origin lies inside a shape misses that shape.
A segment of radius zero is a line, hit wherever a ray crosses it.
Ray direction components below `1e-12` are replaced by `±1e-12` in the slab test, so the ranges and their gradients stay finite for a ray parallel to a face.
`Lidar.scan` reads obstacle poses from the engine's live arrays (`world.obstacle_state_views()` on the torch path, `stepper.obs_*` on the Warp path), so a *movable* obstacle is scanned where it currently is, not where it spawned.
Nothing in a scan reads the device back to the host.

### What each ray hit

`Lidar(..., return_hits=True)` makes a scan return `(ranges, hit_class, hit_index)`, all `[n_envs, n_agents, n_rays]`:

| `hit_class` (`uint8`) | meaning | `hit_index` (`int32`) |
|---|---|---|
| 0 | nothing within `max_range` | -1 |
| 1 | an agent | the agent index |
| 2 | an `IMMOVABLE` obstacle | the obstacle *shape* index |
| 3 | a `MOVABLE` obstacle | the obstacle *shape* index |

A compound body is several shapes, so a scenario that wants the body maps the shape index through its `Obstacles.body` tags.
Ties go to agents before obstacles, and to the lower index within each.

### Field of view

Rays start at `angle_start` and rotate with each agent's heading when `body_frame=True`.
Without `angle_end` they cover the full circle, `2π / n_rays` apart.
With `angle_end` spanning less than 2π the first ray points at `angle_start` and the last at `angle_end`, `(angle_end - angle_start) / (n_rays - 1)` apart; a span of 2π or more is the full circle again.
`Lidar.ray_offsets()` returns the ray angles, so a renderer draws exactly the rays that were cast.

### Backends

`"torch"` (default)
: the pure-torch broadcast implementation. Differentiable with respect to agent poses (including box agents' headings) and obstacle poses, but materializes a dense `[E, A, R, T]` family per shape, so memory grows with the target count.

`"warp"`
: the Warp kernel, one thread per `(env, agent, ray)` keeping a running minimum, so there is no dense intermediate and high ray counts stay affordable. It returns the same ranges and the same hits (tested on randomized mixed scenes in float32 and float64). Inference-only: when gradients are required the scan transparently falls back to the torch path. Its outputs are views of buffers the `Lidar` owns, overwritten by the next scan.

### Inside a captured step

`Lidar.scan` is the eager convenience wrapper.
A fused scenario that wants the scan inside the whole-step CUDA graph splits it in two:

- `Lidar.bind(world)`, called once **outside** any capture (in `make_world`, or before the first step): allocates the output buffers (`lidar.ranges`, `lidar.hit_class`, `lidar.hit_index` as torch views; `lidar.wp_ranges`, `lidar.wp_hit_class`, `lidar.wp_hit_index` as Warp handles) and loads the kernel modules, because a kernel first launched inside a capture cannot be loaded there. Re-binding the same world keeps the buffers; anything else reallocates them and bumps `lidar.version`, which a scenario can fold into its recapture token.
- `Lidar.launch(world)`, called in `launch_fused` ahead of the observation kernel that reads `lidar.wp_ranges`: a single `wp.launch` against `world.state_wp()` and the stepper's live `obs_*` arrays, with no allocation, wrap, host read or stream scope, so it follows the capture-safety rule of [Writing a scenario](writing-a-scenario.md).

The Warp path reads the agent footprint from the shared parameter matrix, so a per-env parameter override of the radius or half-extents is not seen by the lidar.
