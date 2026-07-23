"""M5 integration: render_extras hook, Environment.render, notebook video, demo."""

import os

import numpy as np
import pytest

from wmas import Environment, NavigationScenario
from wmas.render.geometry import extract_geometry


def make_env(n_envs=3, n_agents=3, n_obstacles=1, device="cpu"):
    scenario = NavigationScenario(n_agents=n_agents, n_obstacles=n_obstacles)
    env = Environment(scenario, n_envs=n_envs, device=device, dt=0.1, seed=0)
    env.reset()
    return env, scenario


# ------------------------------------------------------- Scenario.render_extras


def test_scenario_base_render_extras_defaults_empty():
    _, scenario = make_env()
    assert scenario.render_extras(0) == {}


def test_render_extras_override_flows_into_geometry():
    class ExtrasScenario(NavigationScenario):
        def render_extras(self, env_idx):
            return {"lidar": [[[0.0, 0.0], [0.5, 0.5]]]}

    scenario = ExtrasScenario(n_agents=2)
    env = Environment(scenario, n_envs=2, device="cpu", dt=0.1, seed=0)
    env.reset()
    g = extract_geometry(env.world, 0, scenario=scenario)
    assert "lidar" in g.extras and np.asarray(g.extras["lidar"]).shape == (1, 2, 2)


# ---------------------------------------------------- Environment.render (VMAS)


def test_environment_render_rgb_array_returns_frame():
    env, _ = make_env(n_envs=3, n_agents=3)
    frame = env.render(mode="rgb_array", env_index=1, size=(240, 180))
    assert frame.shape == (180, 240, 3)
    assert frame.dtype == np.uint8


def test_environment_render_rejects_unknown_mode():
    env, _ = make_env()
    with pytest.raises(ValueError):
        env.render(mode="nope")


def test_environment_render_human_smoke(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env, _ = make_env()
    assert env.render(mode="human", size=(160, 120)) is None
    env.close_viewer()


# ------------------------------------------------------------- notebook video


def test_record_frames_returns_one_frame_per_step():
    from wmas.render.video import record_frames

    env, _ = make_env(n_agents=3)
    frames = record_frames(env, n_steps=4, size=(80, 60))
    assert len(frames) == 4
    assert frames[0].shape == (60, 80, 3)


def test_to_html5_video_embeds_mp4():
    import base64

    from wmas.render.notebook import to_html5_video

    frames = [np.full((32, 48, 3), fill, dtype=np.uint8) for fill in (20, 120, 220, 60)]
    html = to_html5_video(frames, fps=8)
    text = getattr(html, "data", html)  # IPython.display.HTML has .data; else a str
    assert "data:video/mp4;base64," in text
    data = base64.b64decode(text.split("base64,")[1].split('"')[0])
    assert len(data) > 0 and b"ftyp" in data[:64]  # a real mp4 container


# -------------------------------------------------------------------- demo


def _ffmpeg_available() -> bool:
    try:
        import imageio_ffmpeg  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ffmpeg_available(), reason="imageio-ffmpeg not installed")
def test_demo_save_path_writes_file(tmp_path):
    from wmas.render.demo import main

    out = main(
        [
            "--save",
            str(tmp_path / "demo.mp4"),
            "--steps",
            "4",
            "--size",
            "120",
            "--envs",
            "2",
            "--agents",
            "3",
        ]
    )
    assert os.path.exists(out)


# -------------------------------------------------------------- public API


def test_package_exposes_public_api():
    import wmas.render as render

    for name in (
        "Viewer",
        "render_frame",
        "save_video",
        "record_frames",
        "extract_geometry",
        "RenderGeometry",
        "Camera",
        "Style",
        "animate",
        "to_html5_video",
    ):
        assert getattr(render, name) is not None
