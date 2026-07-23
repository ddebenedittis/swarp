"""M3 mosaic: layout math, tile picking, mosaic rendering, click-to-focus."""

import logging

import numpy as np
import pygame

from wmas import Environment, NavigationScenario
from wmas.render.layout import compute_mosaic_layout, tile_at
from wmas.render.viewer import Viewer


def make_env(n_envs=6, n_agents=3, n_obstacles=1, device="cpu"):
    scenario = NavigationScenario(n_agents=n_agents, n_obstacles=n_obstacles)
    env = Environment(scenario, n_envs=n_envs, device=device, dt=0.1, seed=0)
    env.reset()
    return env, scenario


# ------------------------------------------------------------------- layout


def test_mosaic_layout_tile_count_and_within_window():
    lay = compute_mosaic_layout((800, 600), n_envs=5)
    assert len(lay.tiles) == 5
    assert lay.tile_envs == [0, 1, 2, 3, 4]
    assert lay.n_hidden == 0
    fx, fy, fw, fh = lay.focus_rect
    assert (fx, fy) == (0, 0) and fw < 800 and fh == 600
    for x, y, w, h in lay.tiles:
        assert w > 0 and h > 0
        assert x >= fw  # mosaic strip sits to the right of the focus pane
        assert x + w <= 800 and y >= 0 and y + h <= 600


def test_mosaic_layout_truncates_and_logs_when_over_max(caplog):
    with caplog.at_level(logging.WARNING, logger="wmas.render"):
        lay = compute_mosaic_layout((800, 600), n_envs=100, max_tiles=16)
    assert len(lay.tiles) == 16
    assert lay.tile_envs == list(range(16))
    assert lay.n_hidden == 84
    assert any("hidden" in rec.message for rec in caplog.records)


def test_tile_at_maps_point_to_env_and_focus_pane_to_none():
    lay = compute_mosaic_layout((800, 600), n_envs=4)
    x, y, w, h = lay.tiles[2]
    assert tile_at(lay, (x + w / 2, y + h / 2)) == lay.tile_envs[2] == 2
    assert tile_at(lay, (5, 5)) is None  # inside the focus pane, not a tile


# ------------------------------------------------------- mosaic rendering


def test_viewer_mosaic_render_shape_and_focus_change_differs():
    env, _ = make_env(n_envs=6, n_agents=3)
    viewer = Viewer(env, size=(480, 360), mosaic=True, max_tiles=6)
    frame = viewer.render_array()
    assert frame.shape == (360, 480, 3)
    assert frame.dtype == np.uint8

    viewer.state.focus_env = 0
    f0 = viewer.render_array()
    viewer.state.focus_env = 4
    f4 = viewer.render_array()
    assert not np.array_equal(f0, f4)  # focus pane + highlighted tile both move


def test_click_on_tile_sets_focus_env():
    from wmas.render.input import InteractionController

    env, _ = make_env(n_envs=6, n_agents=3)
    viewer = Viewer(env, size=(480, 360), mosaic=True, max_tiles=6)
    viewer.render_array()  # builds the layout
    camera = viewer._camera_for(viewer._geometry())
    ctrl = InteractionController(
        viewer.state, camera, geometry_getter=viewer._geometry, tile_resolver=viewer._tile_env_at
    )
    tx, ty, tw, th = viewer._layout.tiles[3]
    ctrl.handle_event(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(tx + tw / 2, ty + th / 2))
    )
    assert viewer.state.focus_env == viewer._layout.tile_envs[3] == 3
