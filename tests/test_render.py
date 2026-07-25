"""Visualization: geometry extraction, camera, renderer, video (headless)."""

import numpy as np
import pytest
import torch

from wmas import Environment, NavigationScenario
from wmas.render.camera import Camera
from wmas.render.geometry import RenderGeometry, extract_geometry
from wmas.render.renderer import render_frame
from wmas.render.style import Style
from wmas.render.video import frames_to_video, save_video


def _ffmpeg_available() -> bool:
    try:
        import imageio_ffmpeg  # noqa: F401

        return True
    except Exception:
        return False


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
        assert getattr(big, name) == 3 * getattr(base, name), name
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
    env, _ = make_env(n_agents=2)
    with pytest.raises(ValueError, match="mp4|webm"):
        save_video(env, tmp_path / "roll.gif", n_steps=1, size=(80, 80))
