# World, contacts and sensing

`WorldConfig` (`swarp/core/config.py`) holds everything about the world that is not an agent: the contact law, the boundary, the neighbour search, and the integrator.
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

`Scenario.make_world` takes the same argument, and all seven built-ins honour it (a test pins that).

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

Forces are **gather-based** — each thread sums the forces on its own agent from its own neighbour list, obstacles and walls. Symmetric pairs are computed twice, which buys a deterministic, race-free result with no atomics.

## Bounds

`bounds=(x_min, x_max, y_min, y_max)`, or `None` for an unbounded world.

`bounds_mode="soft"`
: spring-damper walls, using the same contact constants. Agents decelerate into the boundary.

`bounds_mode="clamp"`
: positions are hard-clamped inside. Differentiable as a subgradient.

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

Two limits worth knowing:

- **Box-box obstacle-obstacle contacts are not modelled.** Obstacles collide with each other for every pair in which at least one body is round; two boxes pass through one another, since that needs a polygon contact manifold rather than an SDF evaluated against a disc. Agent-vs-box is exact for every shape.
- **Body integration is not taped** (`record_tape=False`), so no gradient flows through a body's motion. A scenario that needs one keeps its own torch-side copy — see [Differentiability](differentiability.md).

## Neighbour search

Neighbour lists are padded to `max_neighbors` and built without a host sync. `neighbor_radius` defaults to the interaction reach `2·max_agent_radius + collision_margin`, and may not be set below it while collisions are on.

`neighbor_method` picks the backend:

| method | strategy | when |
|---|---|---|
| `"brute"` | per-env O(n_agents²) kernel | the usual multi-agent regime; linear in `n_envs` with a tiny constant |
| `"uniform_grid"` | batched radix-sort grid, each env owning a disjoint block of cells, 3×3 neighbourhood query | large per-env populations across many envs |
| `"grid"` | one `wp.HashGrid` over all envs via a z-lift | very large per-env populations and few envs only |
| `"auto"` (default) | `uniform_grid` above 512 agents per env, `brute` otherwise | |

`wp.HashGrid` wraps cell *coordinates* modulo its dims, so the z-lifted envs alias into shared cells and query cost grows with `n_envs` — which is why `"grid"` is rarely the right answer here and is measured ~300× slower than brute force at 16k envs × 64 agents.
`grid_dim` (default 128) is that backend's bucket dimension per axis.

`uniform_bins` sets the cells per axis of the `"uniform_grid"` backend; `None` (the default) picks `~sqrt(n_agents)`, which keeps occupancy near one agent per cell.
Raising it is usually a mistake: every query zeroes `n_envs · bins²` `int32` cell offsets before the sort, so an over-fine grid spends more clearing empty cells than searching occupied ones — 128 bins with 8 agents is 16384 cells per env for 8 points.
A value that far above the heuristic warns.

The backends are tested to agree exactly. `World.neighbor_overflow()` flags any agent whose true in-radius count exceeded `max_neighbors`, so truncation is never silent, and `World.edge_index()` turns the same lists into a COO radius graph.

Neighbour construction is **not taped**: the neighbour *set* is a discrete structure, so gradients flow through contact geometry rather than through membership.
`neighbor_reuse` (on by default) lets the no-grad path reuse the list the previous step's post-step build already produced, turning two builds per step into one; the reused list is bit-identical to a fresh build on the same positions.

## Lidar

An opt-in differentiable range sensor lives in `swarp/sensors/lidar.py`. It is not part of the step — a scenario calls it in `observations()`.

```python
from swarp import Lidar

lidar = Lidar(n_rays=16, max_range=1.0, backend="warp")
ranges = lidar.scan(world)                 # [n_envs, n_agents, n_rays]
obs = torch.cat([base_obs, ranges], dim=-1)
```

Geometry is analytic ray-circle intersection against other agents and circular obstacles; a ray that hits nothing returns `max_range`. Rays are spaced uniformly over 2π and rotate with each agent's heading when `body_frame=True`.
Box and segment obstacles are **excluded** from the scan — `Lidar.scan` filters on `ObstacleShape` before casting, because ray-circle is the only test implemented — so they are invisible to the sensor while still acting in the collision step.
`Lidar.scan` reads the obstacle pose from `world.obstacle_state_views()`, so a *movable* obstacle is scanned where it currently is, not where it spawned.

Two interchangeable backends:

`"torch"` (default)
: the pure-torch broadcast implementation. Differentiable, but materializes a dense `[E, A, R, T]` family, so memory grows with the target count.

`"warp"`
: the Warp kernel — numerically equivalent with no dense intermediate, which keeps high ray counts affordable. Inference-only: when gradients are required the scan transparently falls back to the torch path.
