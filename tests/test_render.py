"""Visualization: geometry extraction, camera, renderer, video (headless)."""

import numpy as np
import pytest
import torch

from wmas import Environment, NavigationScenario
from wmas.render.camera import Camera
from wmas.render.geometry import RenderGeometry, extract_geometry
from wmas.render.renderer import render_frame
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


# --------------------------------------------------------------- renderer


def _geometry(n_agents=4, n_obstacles=2):
    env, scenario = make_env(n_agents=n_agents, n_obstacles=n_obstacles)
    return extract_geometry(env.world, env_idx=0, scenario=scenario)


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
