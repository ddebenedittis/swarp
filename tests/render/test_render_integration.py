"""M5 integration: render_extras hook, Environment.render, notebook video, demo."""

import os

import numpy as np
import pytest
from conftest import _ffmpeg_available

from swarp import Environment, NavigationScenario
from swarp.render.geometry import extract_geometry

pytestmark = pytest.mark.viz


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


def test_demo_visualization_scenario_emits_lidar_and_comm_lines():
    from swarp.render.demo import build_env

    env = build_env(1, 3, 1, "cpu", lidar_rays=4, lidar_range=1.0)
    extras = env.scenario.render_extras(0)
    assert np.asarray(extras["lidar"]).shape == (12, 2, 2)
    assert np.asarray(extras["lidar_by_agent"]).shape == (3, 4, 2, 2)
    assert np.asarray(extras["comm_lines"]).ndim == 3
    assert np.asarray(extras["comm_lines"]).shape[1:] == (2, 2)


def test_demo_mixed_model_env_exposes_distinct_agent_models():
    from swarp.dynamics.base import DynamicsModel
    from swarp.render.demo import build_env, goal_seeking_policy
    from swarp.render.geometry import extract_geometry

    env = build_env(1, 4, 0, "cpu", model="mixed")
    g = extract_geometry(env.world, 0, scenario=env.scenario)
    assert set(g.model.tolist()) == {
        int(DynamicsModel.HOLONOMIC),
        int(DynamicsModel.DIFF_DRIVE),
        int(DynamicsModel.KINEMATIC_BICYCLE),
    }
    actions = goal_seeking_policy(env)(None)
    assert actions.shape == (1, 4, env.world.act_dim)


def test_demo_drone_env_flies_to_its_goal_and_holds_altitude():
    """The demo's drone controller must be stable: `--model drone` has to be watchable."""
    import torch

    from swarp.dynamics.base import DynamicsModel
    from swarp.render.demo import _DRONE_HOVER_Z, build_env, goal_seeking_policy
    from swarp.render.geometry import extract_geometry
    from swarp.render.renderer import render_frame

    env = build_env(2, 3, 1, "cpu", model="drone")
    assert env.world.act_dim == 4  # four per-rotor thrusts
    policy = goal_seeking_policy(env)
    start = torch.linalg.norm(env.world.goals - env.world.state.pos, dim=-1).mean().item()
    for _ in range(120):
        env.step(policy(None))

    state = env.world.state
    assert torch.isfinite(state.pos).all() and torch.isfinite(state.attitude).all()
    assert torch.linalg.norm(env.world.goals - state.pos, dim=-1).mean().item() < 0.1 * start
    assert abs(state.z.mean().item() - _DRONE_HOVER_Z) < 0.05

    g = extract_geometry(env.world, 0, scenario=env.scenario)
    assert set(g.model.tolist()) == {int(DynamicsModel.DRONE)}
    assert render_frame(g, size=(160, 160), overlays={"agents"}).shape == (160, 160, 3)


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
    from swarp.render.video import record_frames

    env, _ = make_env(n_agents=3)
    frames = record_frames(env, n_steps=4, size=(80, 60))
    assert len(frames) == 4
    assert frames[0].shape == (60, 80, 3)


@pytest.mark.skipif(not _ffmpeg_available(), reason="imageio-ffmpeg not installed")
def test_to_html5_video_embeds_mp4():
    import base64

    from swarp.render.notebook import to_html5_video

    frames = [np.full((32, 48, 3), fill, dtype=np.uint8) for fill in (20, 120, 220, 60)]
    html = to_html5_video(frames, fps=8)
    text = getattr(html, "data", html)  # IPython.display.HTML has .data; else a str
    assert "data:video/mp4;base64," in text
    data = base64.b64decode(text.split("base64,")[1].split('"')[0])
    assert len(data) > 0 and b"ftyp" in data[:64]  # a real mp4 container


# -------------------------------------------------------------------- demo


@pytest.mark.skipif(not _ffmpeg_available(), reason="imageio-ffmpeg not installed")
def test_demo_save_path_writes_file(tmp_path):
    from swarp.render.demo import main

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
    import swarp.render as render

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
