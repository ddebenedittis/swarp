"""M3 mosaic: layout math, tile picking, mosaic rendering, click-to-focus."""

import logging

import numpy as np
import pygame
import pytest

from swarp import Environment, NavigationScenario
from swarp.render.layout import compute_mosaic_layout, tile_at
from swarp.render.viewer import Viewer

pytestmark = pytest.mark.viz


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
    with caplog.at_level(logging.WARNING, logger="swarp.render"):
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


def test_mosaic_layout_scaled_multiplies_every_rect():
    lay = compute_mosaic_layout((800, 600), n_envs=5)
    big = lay.scaled(3)
    assert big.focus_rect == tuple(3 * v for v in lay.focus_rect)
    assert big.tiles == [tuple(3 * v for v in t) for t in lay.tiles]
    assert big.tile_envs == lay.tile_envs and big.n_hidden == lay.n_hidden
    assert lay.scaled(1) is lay


def test_mosaic_render_rebuilds_neighbor_grid_at_most_once_per_frame():
    """Tiles must not each trigger world.neighbors() — that rebuilds the grid per tile."""
    env, scenario = make_env(n_envs=6, n_agents=3)
    viewer = Viewer(env, size=(480, 360), mosaic=True, max_tiles=6, scenario=scenario)
    calls = 0
    real = env.world.neighbors

    def counting_neighbors(*a, **kw):
        nonlocal calls
        calls += 1
        return real(*a, **kw)

    env.world.neighbors = counting_neighbors
    viewer.render_array()
    assert calls <= 1, f"{calls} neighbor rebuilds for 6 tiles + 1 focus pane"


def test_mosaic_supersampled_render_keeps_shape_and_changes_pixels():
    from swarp.render.style import Style

    env, _ = make_env(n_envs=6, n_agents=3)
    plain = Viewer(env, size=(240, 180), mosaic=True, max_tiles=6, style=Style(supersample=1))
    aa = Viewer(env, size=(240, 180), mosaic=True, max_tiles=6, style=Style(supersample=2))
    f1, f2 = plain.render_array(hud=True), aa.render_array(hud=True)
    assert f1.shape == f2.shape == (180, 240, 3)
    assert not np.array_equal(f1, f2)


def test_click_on_tile_sets_focus_env():
    from swarp.render.input import InteractionController

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
