"""Visualization: geometry extraction, camera, renderer, video (headless)."""

import numpy as np
import pytest
import torch
import warp as wp
from conftest import _ffmpeg_available

from wmas import Environment, NavigationScenario
from wmas.render.camera import Camera
from wmas.render.geometry import RenderGeometry, extract_geometry
from wmas.render.renderer import render_frame
from wmas.render.style import Style
from wmas.render.video import frames_to_video, save_video

pytestmark = pytest.mark.viz


def make_env(n_envs=4, n_agents=3, n_obstacles=2, world_size=1.0, device="cpu"):
    scenario = NavigationScenario(n_agents=n_agents, n_obstacles=n_obstacles, world_size=world_size)
    env = Environment(scenario, n_envs=n_envs, device=device, dt=0.1, seed=0)
    env.reset()
    return env, scenario


# --------------------------------------------------------------- geometry


def test_extract_geometry_shapes_and_is_cpu_numpy():
    env, scenario = make_env(n_agents=3, n_obstacles=2, world_size=1.0)
    g = extract_geometry(env.world, env_idx=1, scenario=scenario)

    assert isinstance(g, RenderGeometry)
    assert g.n_agents == 3
    for arr, shape in [
        (g.pos, (3, 2)),
        (g.theta, (3,)),
        (g.vel, (3, 2)),
        (g.radius, (3,)),
        (g.model, (3,)),
        (g.goals, (3, 2)),
        (g.obstacle_pos, (2, 2)),
        (g.obstacle_radius, (2,)),
    ]:
        assert isinstance(arr, np.ndarray), f"{arr!r} is not an ndarray"
        assert arr.shape == shape
    # world bounds come straight from the WorldConfig
    assert g.bounds == (-1.0, 1.0, -1.0, 1.0)
    # values match the source env for the requested env index
    np.testing.assert_allclose(g.pos, env.world.state.pos[1].cpu().numpy())
    np.testing.assert_allclose(g.goals, env.world.goals[1].cpu().numpy())
    np.testing.assert_array_equal(g.model, np.zeros(3, dtype=np.int32))


def test_extract_geometry_handles_absent_obstacles():
    env, scenario = make_env(n_obstacles=0)
    g = extract_geometry(env.world, env_idx=0, scenario=scenario)
    assert g.obstacle_pos is None
    assert g.obstacle_radius is None


def test_extract_geometry_edges_are_local_index_pairs():
    env, scenario = make_env(n_agents=4)
    g = extract_geometry(env.world, env_idx=0, scenario=scenario)
    assert g.edges.ndim == 2 and g.edges.shape[1] == 2
    assert g.edges.dtype == np.int64
    if g.edges.size:
        assert g.edges.min() >= 0 and g.edges.max() < g.n_agents


# ----------------------------------------------------------------- camera


def test_camera_centers_world_origin_in_viewport():
    cam = Camera(bounds=(-1.0, 1.0, -1.0, 1.0), viewport=(0, 0, 400, 400))
    cx, cy = cam.world_to_screen((0.0, 0.0))
    assert abs(cx - 200.0) < 1e-6
    assert abs(cy - 200.0) < 1e-6


def test_camera_flips_y_axis():
    cam = Camera(bounds=(-1.0, 1.0, -1.0, 1.0), viewport=(0, 0, 400, 400))
    top = cam.world_to_screen((0.0, 1.0))
    bottom = cam.world_to_screen((0.0, -1.0))
    assert top[1] < bottom[1]  # larger world-y is higher on screen (smaller pixel y)


def test_camera_screen_world_roundtrip():
    cam = Camera(bounds=(-2.0, 1.0, -1.0, 3.0), viewport=(10, 20, 640, 480))
    for p in [(0.5, -0.3), (-1.9, 2.9), (0.0, 0.0), (1.0, -1.0)]:
        s = cam.world_to_screen(p)
        w = cam.screen_to_world(s)
        np.testing.assert_allclose(w, p, atol=1e-6)


def test_camera_world_to_screen_is_vectorized():
    cam = Camera(bounds=(-1.0, 1.0, -1.0, 1.0), viewport=(0, 0, 400, 400))
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    out = cam.world_to_screen(pts)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out[0], [200.0, 200.0], atol=1e-6)


def test_camera_scaled_is_an_exact_magnification():
    cam = Camera(bounds=(-2.0, 1.0, -1.0, 3.0), viewport=(10, 20, 640, 480), zoom=1.7)
    cam.pan(37.0, -19.0)
    pts = np.array([[0.5, -0.3], [-1.9, 2.9], [0.0, 0.0]])
    for f in (2, 3):
        np.testing.assert_allclose(cam.scaled(f).world_to_screen(pts), f * cam.world_to_screen(pts))
    assert cam.scaled(1) is cam
    # the original must be untouched: the interaction controller holds this exact object
    assert (cam.pan_x, cam.pan_y, cam.zoom) == (37.0, -19.0, 1.7)


def test_batch_extraction_matches_per_env_extraction():
    from wmas.render.geometry import extract_geometry_batch

    env, scenario = make_env(n_envs=5, n_agents=4)
    idx = [0, 2, 4]
    batch = extract_geometry_batch(env.world, idx, scenario=scenario)
    assert len(batch) == 3
    for k, env_idx in enumerate(idx):
        one = extract_geometry(env.world, env_idx, scenario=scenario)
        for name in ("pos", "theta", "vel", "radius", "model", "goals", "obstacle_pos", "edges"):
            np.testing.assert_array_equal(
                getattr(batch[k], name), getattr(one, name), err_msg=f"{name} @ env {env_idx}"
            )
        assert batch[k].bounds == one.bounds


def test_batch_extraction_can_skip_edges_and_extras():
    from wmas.render.geometry import extract_geometry_batch

    env, scenario = make_env(n_envs=4, n_agents=4)
    calls = 0
    real = env.world.neighbors

    def counting(*a, **kw):
        nonlocal calls
        calls += 1
        return real(*a, **kw)

    env.world.neighbors = counting
    out = extract_geometry_batch(
        env.world, [0, 1, 2], scenario=scenario, with_edges=False, with_extras=False
    )
    assert calls == 0  # no neighbor-grid rebuild
    assert all(g.edges.shape == (0, 2) and g.extras == {} for g in out)


def test_batch_extraction_of_empty_index_list():
    env, scenario = make_env()
    from wmas.render.geometry import extract_geometry_batch

    assert extract_geometry_batch(env.world, [], scenario=scenario) == []


# --------------------------------------------------------------- renderer


def _geometry(n_agents=4, n_obstacles=2):
    env, scenario = make_env(n_agents=n_agents, n_obstacles=n_obstacles)
    return extract_geometry(env.world, env_idx=0, scenario=scenario)


# ------------------------------------------------------------------ style


def test_style_font_px_scales_with_viewport_and_clamps():
    s = Style()
    assert s.font_px(700) == s.font_size  # default ratio reproduces the fixed size exactly
    assert s.font_px(150) == s.font_min
    assert s.font_px(5000) == s.font_max
    assert s.font_px(None) == s.font_size
    assert s.font_px(350) < s.font_px(700) < s.font_px(1400)
    assert Style(font_scale=0.0).font_px(1400) == s.font_size  # scaling opt-out


def test_style_field_partition_is_exhaustive():
    """Every Style field must be classified as pixel-valued or not.

    Guards Style.scaled(): an unclassified size would stay 1px in a supersampled buffer
    and downscale to a washed-out pixel, so adding a field must force the decision.
    """
    from wmas.render.style import _PX_FIELDS, _UNSCALED_FIELDS, style_field_names

    assert _PX_FIELDS.isdisjoint(_UNSCALED_FIELDS)
    assert style_field_names() == _PX_FIELDS | _UNSCALED_FIELDS


def test_style_scaled_multiplies_pixel_fields_only():
    from wmas.render.style import _PX_FIELDS, _UNSCALED_FIELDS

    base = Style()
    big = base.scaled(3)
    for name in _PX_FIELDS:
        was, now = getattr(base, name), getattr(big, name)
        expected = tuple(3 * v for v in was) if isinstance(was, tuple) else 3 * was
        assert now == expected, name
    for name in _UNSCALED_FIELDS:
        assert getattr(big, name) == getattr(base, name), name
    assert base.scaled(1) is base
    # font_scale is unscaled, so a 3x viewport yields ~3x text via the ratio alone
    assert big.font_px(3 * 700) == 3 * base.font_px(700)


def test_style_themes_differ_and_accept_overrides():
    from wmas.render.style import THEMES

    light, dark = Style.light(), Style.dark()
    assert light == Style()
    assert dark.background != light.background
    assert dark.text_color != light.text_color
    assert Style.dark(color_mode="model").color_mode == "model"
    assert set(THEMES) == {"light", "dark"}


def test_dark_theme_changes_rendered_pixels():
    g = _geometry()
    light = render_frame(g, size=(200, 200), style=Style.light())
    dark = render_frame(g, size=(200, 200), style=Style.dark())
    assert not np.array_equal(light, dark)


def test_render_frame_shape_and_dtype():
    frame = render_frame(_geometry(), size=(320, 240))
    assert frame.shape == (240, 320, 3)  # (H, W, 3)
    assert frame.dtype == np.uint8


def test_render_frame_is_deterministic():
    g = _geometry()
    a = render_frame(g, size=(200, 200))
    b = render_frame(g, size=(200, 200))
    assert np.array_equal(a, b)


def test_render_frame_draws_agents_over_background():
    frame = render_frame(_geometry(), size=(200, 200), overlays={"agents"})
    assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 1


def test_render_frame_overlay_toggle_changes_pixels():
    g = _geometry()
    without = render_frame(g, size=(200, 200), overlays={"agents"})
    with_goals = render_frame(g, size=(200, 200), overlays={"agents", "goals"})
    assert not np.array_equal(without, with_goals)


def test_render_frame_lidar_overlay_is_noop_without_data():
    # lidar sensor does not exist yet; the overlay must be harmless when no data present.
    g = _geometry()
    assert "lidar" not in g.extras
    base = render_frame(g, size=(200, 200), overlays={"agents"})
    with_lidar = render_frame(g, size=(200, 200), overlays={"agents", "lidar"})
    assert np.array_equal(base, with_lidar)


def test_render_frame_lidar_endpoint_dot_changes_pixels():
    g = _geometry()
    g.extras["lidar"] = np.asarray([[[0.0, 0.0], [0.4, 0.0]]])
    without = render_frame(g, size=(200, 200), overlays={"agents"})
    with_lidar = render_frame(g, size=(200, 200), overlays={"agents", "lidar"})
    assert not np.array_equal(without, with_lidar)


def test_render_frame_lidar_area_mode_changes_pixels():
    from wmas.render.style import Style

    g = _geometry(n_agents=1)
    g.extras["lidar"] = np.asarray([[[0.0, 0.0], [0.4, 0.0]], [[0.0, 0.0], [0.0, 0.4]]])
    g.extras["lidar_by_agent"] = g.extras["lidar"].reshape(1, 2, 2, 2)
    without = render_frame(g, size=(200, 200), overlays={"agents"})
    with_area = render_frame(
        g, size=(200, 200), overlays={"agents", "lidar"}, style=Style(lidar_mode="area")
    )
    assert not np.array_equal(without, with_area)


def test_render_frame_trajectory_overlay_changes_pixels():
    from wmas.render.style import Style

    g = _geometry(n_agents=1)
    g.extras["trajectories"] = np.asarray([[[0.0, 0.0], [0.2, 0.0], [0.3, 0.2]]])
    without = render_frame(g, size=(200, 200), overlays={"agents"})
    with_trail = render_frame(
        g,
        size=(200, 200),
        overlays={"agents", "trajectories"},
        style=Style(trajectory_mode="fade"),
    )
    assert not np.array_equal(without, with_trail)


# ------------------------------------------------------- obstacle shapes


def _shaped_obstacle_env(shape, angle=0.4, half=(0.25, 0.12), radius=0.05):
    """An env whose single obstacle is a BOX/SEGMENT rather than a circle."""
    from wmas.core.config import Obstacles, ObstacleShape

    env, scenario = make_env(n_agents=2, n_obstacles=1)
    world = env.world
    n_obs = 1
    env.world.set_obstacles(
        Obstacles(
            torch.zeros(env.n_envs, n_obs, 2, dtype=env.dtype, device=env.device),
            torch.full((n_obs,), radius, dtype=env.dtype, device=env.device),
            shape=torch.full((n_obs,), int(shape), dtype=torch.int32, device=env.device),
            angle=torch.full((n_obs,), angle, dtype=env.dtype, device=env.device),
            half_extents=torch.tensor([half], dtype=env.dtype, device=env.device),
        )
    )
    assert world.obstacle_shape is not None and ObstacleShape.CIRCLE == 0
    return env, scenario


def test_extract_geometry_carries_obstacle_shape_angle_and_extents():
    from wmas.core.config import ObstacleShape

    env, scenario = _shaped_obstacle_env(ObstacleShape.BOX)
    g = extract_geometry(env.world, 0, scenario=scenario)
    assert g.obstacle_shape.shape == (1,) and int(g.obstacle_shape[0]) == int(ObstacleShape.BOX)
    assert g.obstacle_angle.shape == (1,)
    assert g.obstacle_half_extents.shape == (1, 2)
    np.testing.assert_allclose(g.obstacle_half_extents[0], [0.25, 0.12])


def test_extract_geometry_leaves_shape_fields_none_for_circle_only_scenarios():
    env, scenario = make_env(n_obstacles=2)
    g = extract_geometry(env.world, 0, scenario=scenario)
    assert g.obstacle_shape is None
    assert g.obstacle_angle is None
    assert g.obstacle_half_extents is None


def test_box_and_segment_obstacles_render_differently_from_circles():
    from wmas.core.config import ObstacleShape

    circles = render_frame(_geometry(n_obstacles=1), size=(240, 240), overlays={"obstacles"})
    for shape in (ObstacleShape.BOX, ObstacleShape.SEGMENT):
        env, scenario = _shaped_obstacle_env(shape)
        g = extract_geometry(env.world, 0, scenario=scenario)
        frame = render_frame(g, size=(240, 240), overlays={"obstacles"})
        assert not np.array_equal(frame, circles), shape
        assert len(np.unique(frame.reshape(-1, 3), axis=0)) > 1  # something was actually drawn


def test_obstacle_drag_preserves_kind_mass_and_keeps_the_body_movable():
    """Dragging an obstacle must not turn every movable body into permanent scenery.

    ``Viewer._write_obstacle_pos`` used to rebuild a *partial* obstacle spec naming only
    the pose fields, because ``World`` retained only six of the twelve. An absent ``kind``
    means IMMOVABLE, so one mouse drag zeroed ``obs_kind``, cleared ``any_movable`` and
    stopped the engine integrating any body — and since an in-place install bumps no
    version, not even a CUDA-graph recapture would have surfaced it. The fix is that
    ``World`` retains the whole resolved spec and the drag re-installs *that*.
    """
    from wmas.core.config import ObstacleKind
    from wmas.render.viewer import Viewer
    from wmas.scenarios.pusht import PushTScenario

    scenario = PushTScenario(n_agents=1, agent_radius=0.05)
    env = Environment(scenario, n_envs=2, device="cpu", dt=0.05, substeps=8, seed=0)
    env.reset(seed=0)
    world = env.world
    viewer = Viewer(env, scenario=scenario)
    assert world.stepper.any_movable

    # Put the T at the origin, unrotated, so the push below is deterministic.
    scenario.tee_pos.zero_()
    scenario.tee_theta.zero_()
    scenario.tee_vel.zero_()
    scenario.tee_ang_vel.zero_()
    scenario._install_obstacles()

    # Drag the crossbar (the body root) slightly right: the body origin follows it.
    crossbar_y = scenario.box_off[0][1]
    viewer._write_obstacle_pos(0, (0.02, crossbar_y))

    assert world.stepper.any_movable, "the drag froze every movable body"
    assert int(world.obstacle_kind[0]) == int(ObstacleKind.MOVABLE)
    assert world.obstacle_kind.shape == (scenario.n_boxes,)
    np.testing.assert_allclose(
        wp.to_torch(world.stepper.obs_mass).numpy(), scenario._obs_mass.numpy()
    )
    np.testing.assert_allclose(
        wp.to_torch(world.stepper.obs_inertia).numpy(), scenario._obs_inertia.numpy()
    )
    # ...and the engine still pushes it: one agent driving +x into the stem moves the T.
    world.state.pos.data[:, 0] = torch.tensor([-0.35, 0.0])
    world.state.vel.data.zero_()
    world.mark_pos_dirty()
    act = torch.zeros(env.n_envs, scenario.n_agents, 2)
    act[..., 0] = 1.0
    with torch.no_grad():
        for _ in range(40):
            env.step(act)
    assert scenario.tee_pos[0, 0].item() > 0.03, "a dragged movable body stopped being pushed"


def test_circle_only_obstacle_rendering_ignores_absent_shape_arrays():
    """A geometry with obstacle_shape=None must render exactly like the circle branch."""
    g = _geometry(n_obstacles=2)
    with_none = render_frame(g, size=(200, 200), overlays={"obstacles"})
    g.obstacle_shape = np.zeros(g.obstacle_pos.shape[0], dtype=np.int32)  # explicit CIRCLE
    g.obstacle_angle = np.zeros(g.obstacle_pos.shape[0])
    g.obstacle_half_extents = np.zeros((g.obstacle_pos.shape[0], 2))
    assert np.array_equal(with_none, render_frame(g, size=(200, 200), overlays={"obstacles"}))


# -------------------------------------------------------------- action overlay


def test_extract_geometry_exposes_the_applied_action_and_agent_params():
    from wmas.dynamics.base import NUM_PARAMS

    env, scenario = make_env(n_agents=3)
    g = extract_geometry(env.world, 0, scenario=scenario)
    assert g.action is None  # nothing applied yet
    assert g.ctrl_mode.shape == (3,)
    assert g.agent_params.shape == (3, NUM_PARAMS)

    actions = torch.randn(env.n_envs, 3, env.world.act_dim, dtype=env.dtype)
    env.step(actions)
    g = extract_geometry(env.world, 1, scenario=scenario)
    np.testing.assert_allclose(g.action, actions[1].numpy())


def test_action_overlay_is_noop_before_the_first_step():
    g = _geometry()
    assert g.action is None
    base = render_frame(g, size=(200, 200), overlays={"agents"})
    with_action = render_frame(g, size=(200, 200), overlays={"agents", "action"})
    assert np.array_equal(base, with_action)


def test_action_overlay_draws_for_every_dynamics_model():
    from wmas.render.demo import build_env, goal_seeking_policy

    for model in ("holonomic", "diff-drive", "bicycle", "mixed"):
        env = build_env(2, 3, 1, "cpu", model=model)
        policy = goal_seeking_policy(env)
        env.step(policy(None))
        g = extract_geometry(env.world, 0, scenario=env.scenario)
        assert g.action is not None
        without = render_frame(g, size=(240, 240), overlays={"agents"})
        with_action = render_frame(g, size=(240, 240), overlays={"agents", "action"})
        assert not np.array_equal(without, with_action), model


# ------------------------------------------------------------ agent visuals


def test_depth_cue_modes_change_pixels_for_overlapping_agents():
    g = _geometry(n_agents=2)
    g.pos[1] = g.pos[0] + [0.6 * float(g.radius[0]), 0.0]  # deliberate overlap
    frames = {
        mode: render_frame(
            g, size=(240, 240), overlays={"agents"}, style=Style(depth_cue=mode, supersample=1)
        )
        for mode in ("none", "halo", "shadow")
    }
    assert not np.array_equal(frames["none"], frames["halo"])
    assert not np.array_equal(frames["none"], frames["shadow"])
    assert not np.array_equal(frames["halo"], frames["shadow"])


def test_contact_mask_flags_both_endpoints_of_an_overlapping_pair():
    from wmas.render.overlays import _contact_mask

    g = _geometry(n_agents=3)
    style = Style()
    g.edges = np.array([[0, 1]], dtype=np.int64)
    g.pos[1] = g.pos[0] + [0.5 * (g.radius[0] + g.radius[1]), 0.0]  # overlapping
    mask = _contact_mask(g, style)
    assert mask.tolist() == [True, True, False]

    g.pos[1] = g.pos[0] + [5.0 * (g.radius[0] + g.radius[1]), 0.0]  # far apart
    assert not _contact_mask(g, style).any()

    g.edges = np.empty((0, 2), dtype=np.int64)  # collisions disabled
    assert not _contact_mask(g, style).any()


def test_contact_highlight_changes_pixels():
    g = _geometry(n_agents=2)
    g.edges = np.array([[0, 1]], dtype=np.int64)
    g.pos[1] = g.pos[0] + [0.5 * (g.radius[0] + g.radius[1]), 0.0]
    off = render_frame(
        g, size=(240, 240), overlays={"agents"}, style=Style(contact_highlight=False)
    )
    on = render_frame(g, size=(240, 240), overlays={"agents"}, style=Style(contact_highlight=True))
    assert not np.array_equal(off, on)


# ------------------------------------------------------- per-model agent sprites


def _single_agent_geometry(model: int):
    """One centered agent of ``model``, isolated from goals/obstacles/neighbors."""
    g = _geometry(n_agents=1, n_obstacles=0)
    g.pos[0] = (0.0, 0.0)
    g.theta[0] = 0.0
    g.model[0] = model
    g.goals = None
    g.obstacle_pos = None
    g.edges = np.empty((0, 2), dtype=np.int64)
    return g


_SPRITE_STYLE = Style(depth_cue="none", supersample=1)


def _sprite_frame(g):
    return render_frame(g, size=(240, 240), overlays={"agents"}, style=_SPRITE_STYLE)


def test_holonomic_body_is_orientation_free():
    """An omnidirectional agent is a circle: rotating it must not change a single pixel."""
    g = _single_agent_geometry(0)
    base = _sprite_frame(g)
    for theta in (0.3, 1.0, -2.2, np.pi):
        g.theta[0] = theta
        assert np.array_equal(base, _sprite_frame(g)), theta


@pytest.mark.parametrize("model", [1, 2, 3])
def test_oriented_bodies_rotate_with_theta(model):
    g = _single_agent_geometry(model)
    base = _sprite_frame(g)
    for theta in (0.4, 1.2, -2.0):
        g.theta[0] = theta
        assert not np.array_equal(base, _sprite_frame(g)), theta


@pytest.mark.parametrize("model", [1, 2, 3])
def test_non_holonomic_bodies_differ_from_a_plain_circle(model):
    """Each non-holonomic model draws its own silhouette, not the holonomic circle fallback."""
    circle = _sprite_frame(_single_agent_geometry(0))
    g = _single_agent_geometry(model)
    assert not np.array_equal(circle, _sprite_frame(g))


def test_bicycle_front_wheel_follows_the_commanded_steering():
    from wmas.dynamics.base import P_MAX_STEER

    g = _single_agent_geometry(2)
    max_steer = float(g.agent_params[0, P_MAX_STEER])
    assert max_steer > 0.0
    frames = []
    for steer in (-max_steer, 0.0, max_steer):
        g.action = np.array([[0.0, steer]])
        frames.append(_sprite_frame(g))
    assert not np.array_equal(frames[0], frames[1])
    assert not np.array_equal(frames[1], frames[2])
    assert not np.array_equal(frames[0], frames[2])


def test_bicycle_axles_follow_l_f_over_l_r():
    from wmas.dynamics.base import P_LF, P_LR
    from wmas.render.overlays import _BICYCLE_AXLE_SPAN, _bicycle_axles

    g = _single_agent_geometry(2)
    g.agent_params[0, P_LF] = g.agent_params[0, P_LR] = 0.1
    front, rear = _bicycle_axles(g, 0)
    assert front == pytest.approx(-rear)  # symmetric CoG -> symmetric axles
    assert front - rear == pytest.approx(_BICYCLE_AXLE_SPAN)

    g.agent_params[0, P_LF] = 0.3  # CoG pushed toward the rear axle
    front_biased, rear_biased = _bicycle_axles(g, 0)
    assert front_biased > front and rear_biased > rear

    g.agent_params[0, P_LF] = g.agent_params[0, P_LR] = 0.0  # never divide by zero
    assert _bicycle_axles(g, 0) == pytest.approx((front, rear))


def test_steer_angle_is_clamped_and_zero_without_an_action():
    from wmas.dynamics.base import P_MAX_STEER
    from wmas.render.overlays import _steer_angle

    g = _single_agent_geometry(2)
    max_steer = float(g.agent_params[0, P_MAX_STEER])
    assert _steer_angle(g, 0) == 0.0  # no action applied yet
    g.action = np.array([[0.0, 10.0 * max_steer]])
    assert _steer_angle(g, 0) == pytest.approx(max_steer)
    g.action = np.array([[0.0]])  # too narrow to carry a steering column
    assert _steer_angle(g, 0) == 0.0


def test_rounded_rect_factors_stay_inside_the_box_and_smooth_the_corners():
    from wmas.render.overlays import _rounded_rect_factors

    pts = _rounded_rect_factors(1.6, 0.75, 0.45)
    assert len(pts) > 4  # corners are sampled, not cut
    assert np.abs(pts[:, 0]).max() == pytest.approx(1.6)
    assert np.abs(pts[:, 1]).max() == pytest.approx(0.75)
    assert not np.any((np.abs(pts[:, 0]) > 1.6 + 1e-9) | (np.abs(pts[:, 1]) > 0.75 + 1e-9))
    # No vertex lands on a sharp corner of the enclosing box.
    assert np.min(np.hypot(np.abs(pts[:, 0]) - 1.6, np.abs(pts[:, 1]) - 0.75)) > 1e-3
    # An oversized corner radius degrades to a stadium instead of self-intersecting.
    stadium = _rounded_rect_factors(1.0, 0.5, 10.0)
    assert np.abs(stadium[:, 1]).max() == pytest.approx(0.5)


def test_hover_panel_names_the_dynamics_model():
    from wmas.render.hud import _model_name

    assert _model_name(0) == "holonomic"
    assert _model_name(2) == "kinematic_bicycle"
    assert _model_name(3) == "drone"
    assert _model_name(99) == "99"  # an unknown tag degrades to the raw int, never raises


def test_dashed_segments_cover_the_line_in_periodic_pieces():
    from wmas.render.overlays import _dashed_segments

    segs = _dashed_segments((0, 0), (100, 0), dash_px=6, gap_px=4)
    assert len(segs) == 10  # period 10 over a 100px span
    assert segs[0].tolist() == [[0, 0], [6, 0]]
    assert segs[-1][1][0] <= 100  # the final dash is clipped, never overshoots
    assert _dashed_segments((5, 5), (5, 5), 6, 4).shape == (0, 2, 2)  # degenerate


def test_goal_connector_modes_change_pixels():
    g = _geometry(n_agents=2)
    frames = {
        mode: render_frame(g, size=(240, 240), overlays={"goals"}, style=Style(goal_connector=mode))
        for mode in ("none", "solid", "dashed")
    }
    assert not np.array_equal(frames["none"], frames["solid"])
    assert not np.array_equal(frames["none"], frames["dashed"])
    assert not np.array_equal(frames["solid"], frames["dashed"])


def test_reached_goal_renders_differently_from_a_distant_one():
    g = _geometry(n_agents=2)
    far = render_frame(g, size=(240, 240), overlays={"goals"})
    g.goals = g.pos.copy()  # every agent sitting on its goal
    reached = render_frame(g, size=(240, 240), overlays={"goals"})
    assert not np.array_equal(far, reached)


# ----------------------------------------------------------- supersampling


def test_pts_is_pixel_identical_to_the_scalar_transform():
    from wmas.render.overlays import _p, _pts

    cam = Camera(bounds=(-1.0, 2.0, -1.0, 1.0), viewport=(5, 7, 333, 211), zoom=1.3)
    cam.pan(11.0, -23.0)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.5, 1.5, size=(64, 2))
    assert _pts(cam, pts) == [list(_p(cam, p)) for p in pts]


def test_supersampled_frame_keeps_shape_and_antialiases():
    g = _geometry()
    plain = render_frame(g, size=(200, 200), style=Style(supersample=1))
    aa = render_frame(g, size=(200, 200), style=Style(supersample=2))
    assert aa.shape == plain.shape == (200, 200, 3)
    assert not np.array_equal(aa, plain)
    # AA blends edge pixels, so the palette grows well beyond the flat-fill colour count
    assert len(np.unique(aa.reshape(-1, 3), axis=0)) > len(np.unique(plain.reshape(-1, 3), axis=0))


def test_supersampled_frame_is_deterministic():
    g = _geometry()
    style = Style(supersample=2)
    assert np.array_equal(
        render_frame(g, size=(160, 160), style=style),
        render_frame(g, size=(160, 160), style=style),
    )


def test_alpha_layer_is_reused_and_cleared():
    import pygame

    from wmas.render.overlays import _alpha_layer

    first = _alpha_layer("t", (16, 16))
    pygame.draw.rect(first, (255, 0, 0, 255), pygame.Rect(0, 0, 8, 8))
    again = _alpha_layer("t", (16, 16))
    assert again is first  # reused, not reallocated
    assert again.get_at((1, 1)) == (0, 0, 0, 0)  # and cleared on handout
    assert _alpha_layer("other", (16, 16)) is not first  # tags do not share a live layer


def test_cached_alpha_layer_keeps_trajectory_rendering_deterministic():
    g = _geometry(n_agents=1)
    g.extras["trajectories"] = np.asarray([[[0.0, 0.0], [0.2, 0.0], [0.3, 0.2]]])
    style = Style(trajectory_mode="fade")
    a = render_frame(g, size=(200, 200), overlays={"agents", "trajectories"}, style=style)
    b = render_frame(g, size=(200, 200), overlays={"agents", "trajectories"}, style=style)
    assert np.array_equal(a, b)  # a stale, uncleared layer would leak into the second frame


# ------------------------------------------------------------------ video


@pytest.mark.skipif(not _ffmpeg_available(), reason="imageio-ffmpeg not installed")
def test_frames_to_video_mp4_roundtrips_frame_count(tmp_path):
    import imageio.v2 as imageio

    frames = [np.full((32, 48, 3), fill, dtype=np.uint8) for fill in (10, 90, 170, 250)]
    out = frames_to_video(frames, tmp_path / "clip.mp4", fps=10)
    read = imageio.mimread(out)
    assert len(read) == 4
    assert read[0].shape[:2] == (32, 48)


@pytest.mark.skipif(not _ffmpeg_available(), reason="imageio-ffmpeg not installed")
def test_save_video_webm_has_one_frame_per_step(tmp_path):
    import imageio.v2 as imageio

    env, _ = make_env(n_agents=3)

    def policy(obs):
        return 0.6 * torch.ones(env.n_envs, env.n_agents, env.world.act_dim, dtype=env.dtype)

    out = save_video(
        env, tmp_path / "roll.webm", action_fn=policy, n_steps=6, size=(120, 120), fps=10
    )
    assert len(imageio.mimread(out)) == 6


def test_save_video_rejects_gif(tmp_path):
    """Extension validation must not require the optional imageio dependency."""
    env, _ = make_env(n_agents=2)
    with pytest.raises(ValueError, match="mp4|webm"):
        save_video(env, tmp_path / "roll.gif", n_steps=1, size=(80, 80))
