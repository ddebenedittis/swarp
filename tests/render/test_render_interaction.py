"""M2 interactivity: camera pan/zoom, agent picking, event->state controller, HUD."""

import numpy as np
import pygame
import pytest
import torch

from wmas import Environment, NavigationScenario
from wmas.render.camera import Camera
from wmas.render.geometry import extract_geometry
from wmas.render.input import (
    SPEED_LADDER,
    InteractionController,
    ViewState,
    pick_agent,
    pick_obstacle,
)
from wmas.render.overlays import DEFAULT_ENABLED
from wmas.render.viewer import Viewer

pytestmark = pytest.mark.viz


def make_env(n_envs=4, n_agents=3, n_obstacles=1, device="cpu"):
    scenario = NavigationScenario(n_agents=n_agents, n_obstacles=n_obstacles)
    env = Environment(scenario, n_envs=n_envs, device=device, dt=0.1, seed=0)
    env.reset()
    return env, scenario


# ------------------------------------------------------------- camera pan/zoom


def test_camera_pan_shifts_screen_coords_by_pixels():
    cam = Camera((-1.0, 1.0, -1.0, 1.0), (0, 0, 400, 400))
    s0 = cam.world_to_screen((0.3, -0.2))
    cam.pan(15.0, -25.0)
    s1 = cam.world_to_screen((0.3, -0.2))
    np.testing.assert_allclose(s1 - s0, [15.0, -25.0], atol=1e-6)


def test_camera_zoom_at_keeps_world_point_under_cursor_fixed():
    cam = Camera((-1.0, 1.0, -1.0, 1.0), (0, 0, 400, 400))
    cursor = (300.0, 120.0)
    w_before = cam.screen_to_world(cursor)
    cam.zoom_at(2.0, cursor)
    assert abs(cam.zoom - 2.0) < 1e-9
    w_after = cam.screen_to_world(cursor)
    np.testing.assert_allclose(w_after, w_before, atol=1e-6)


# ---------------------------------------------------------------- picking


def test_pick_agent_hits_center_and_misses_empty_space():
    env, scenario = make_env(n_agents=3)
    g = extract_geometry(env.world, 0, scenario=scenario)
    cam = Camera(g.bounds, (0, 0, 400, 400))
    center = cam.world_to_screen(g.pos[1])
    assert pick_agent(g, cam, center) == 1
    assert pick_agent(g, cam, (-100.0, -100.0)) is None


def test_pick_obstacle_hits_center_and_misses_empty_space():
    env, scenario = make_env(n_obstacles=1)
    g = extract_geometry(env.world, 0, scenario=scenario)
    cam = Camera(g.bounds, (0, 0, 400, 400))
    center = cam.world_to_screen(g.obstacle_pos[0])
    assert pick_obstacle(g, cam, center) == 0
    assert pick_obstacle(g, cam, (-100.0, -100.0)) is None


def test_pick_obstacle_hits_a_zero_radius_box_by_its_extents():
    """A box's collision radius is legitimately 0, so radius-only picking made boxes undraggable."""
    env, scenario = make_env(n_obstacles=1)
    g = extract_geometry(env.world, 0, scenario=scenario)
    g.obstacle_radius = np.zeros(1)
    g.obstacle_shape = np.array([1], dtype=np.int32)  # ObstacleShape.BOX
    g.obstacle_angle = np.zeros(1)
    g.obstacle_half_extents = np.array([[0.3, 0.2]])
    cam = Camera(g.bounds, (0, 0, 400, 400))
    assert pick_obstacle(g, cam, cam.world_to_screen(g.obstacle_pos[0])) == 0
    assert pick_obstacle(g, cam, (-100.0, -100.0)) is None


def test_supersampling_does_not_disturb_picking_or_the_viewers_camera():
    """The Viewer's camera must stay window-space; only per-frame copies are scaled.

    The interaction controller holds that exact object and feeds it raw window pixels, so a
    scaled camera leaking back would break picking, panning and zoom-under-cursor.
    """
    from wmas.render.style import Style

    env, scenario = make_env(n_agents=3)
    viewer = Viewer(env, size=(300, 300), scenario=scenario, style=Style(supersample=3))
    g = viewer._geometry()
    cam = viewer._camera_for(g)
    before = cam.world_to_screen(g.pos)

    viewer.render_array(hud=True)  # renders at 3x internally

    assert viewer._camera_for(g) is cam
    np.testing.assert_allclose(cam.world_to_screen(g.pos), before)
    for k in range(g.n_agents):
        assert pick_agent(g, cam, cam.world_to_screen(g.pos[k])) == k


# --------------------------------------------------------- event controller


def _controller(geometry=None, camera=None, n_envs=4):
    cam = camera or Camera((-1.0, 1.0, -1.0, 1.0), (0, 0, 400, 400))
    state = ViewState(n_envs=n_envs, enabled=set(DEFAULT_ENABLED))
    getter = (lambda: geometry) if geometry is not None else None
    ctrl = InteractionController(state, cam, geometry_getter=getter)
    return state, ctrl, cam


def _key(name_or_code):
    code = getattr(pygame, f"K_{name_or_code}") if isinstance(name_or_code, str) else name_or_code
    return pygame.event.Event(pygame.KEYDOWN, key=code)


def test_letter_key_toggles_matching_overlay():
    state, ctrl, _ = _controller()
    assert "goals" in state.enabled
    ctrl.handle_event(_key("g"))
    assert "goals" not in state.enabled
    ctrl.handle_event(_key("g"))
    assert "goals" in state.enabled


def test_bracket_keys_switch_focus_env_with_wraparound():
    state, ctrl, _ = _controller(n_envs=4)
    ctrl.handle_event(_key(pygame.K_RIGHTBRACKET))
    assert state.focus_env == 1
    ctrl.handle_event(_key(pygame.K_LEFTBRACKET))
    ctrl.handle_event(_key(pygame.K_LEFTBRACKET))
    assert state.focus_env == 3  # wrapped past 0


def test_space_pauses_and_escape_quits():
    state, ctrl, _ = _controller()
    ctrl.handle_event(_key(pygame.K_SPACE))
    assert state.paused is True
    ctrl.handle_event(_key(pygame.K_ESCAPE))
    assert state.quit is True


def test_reset_and_help_keys_update_view_state():
    state, ctrl, _ = _controller()
    ctrl.handle_event(_key("r"))
    assert state.reset_requested is True
    ctrl.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1))
    assert state.show_help is True
    ctrl.handle_event(pygame.event.Event(pygame.TEXTINPUT, text="?"))
    assert state.show_help is False
    ctrl.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SLASH, mod=pygame.KMOD_SHIFT))
    assert state.show_help is True


def test_mouse_wheel_zooms_in():
    state, ctrl, cam = _controller()
    ctrl.handle_event(
        pygame.event.Event(pygame.MOUSEMOTION, pos=(200, 200), rel=(0, 0), buttons=(0, 0, 0))
    )
    z0 = cam.zoom
    ctrl.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, y=1))
    assert cam.zoom > z0


def test_hover_over_agent_sets_hover_agent():
    env, scenario = make_env(n_agents=3)
    g = extract_geometry(env.world, 0, scenario=scenario)
    cam = Camera(g.bounds, (0, 0, 400, 400))
    state, ctrl, _ = _controller(geometry=g, camera=cam)
    cx, cy = cam.world_to_screen(g.pos[2])
    ctrl.handle_event(
        pygame.event.Event(
            pygame.MOUSEMOTION, pos=(float(cx), float(cy)), rel=(0, 0), buttons=(0, 0, 0)
        )
    )
    assert state.hover_agent == 2


# ----------------------------------------------------------- viewer (offscreen)


def test_viewer_render_array_shape_and_hud_overlay():
    env, _ = make_env(n_agents=3)
    viewer = Viewer(env, size=(300, 220))
    plain = viewer.render_array(hud=False)
    assert plain.shape == (220, 300, 3)
    assert plain.dtype == np.uint8
    with_hud = viewer.render_array(hud=True)
    assert not np.array_equal(plain, with_hud)  # HUD text drawn on top


def test_viewer_hover_panel_changes_pixels():
    env, _ = make_env(n_agents=3)
    viewer = Viewer(env, size=(320, 320))
    base = viewer.render_array(hud=True)
    viewer.state.hover_agent = 0
    hovered = viewer.render_array(hud=True)
    assert not np.array_equal(base, hovered)


def test_viewer_help_panel_changes_pixels():
    env, _ = make_env(n_agents=3)
    viewer = Viewer(env, size=(420, 320))
    base = viewer.render_array(hud=True)
    viewer.state.show_help = True
    helped = viewer.render_array(hud=True)
    assert not np.array_equal(base, helped)


def test_viewer_help_panel_changes_pixels_in_mosaic():
    env, _ = make_env(n_envs=4, n_agents=3)
    viewer = Viewer(env, size=(520, 360), mosaic=True)
    base = viewer.render_array(hud=True)
    viewer.state.show_help = True
    helped = viewer.render_array(hud=True)
    assert not np.array_equal(base, helped)


def test_viewer_write_obstacle_pos_moves_obstacle():
    env, _ = make_env(n_obstacles=1)
    viewer = Viewer(env, size=(320, 320))
    viewer._write_obstacle_pos(0, (0.25, -0.25))
    np.testing.assert_allclose(env.world.obstacle_pos[0, 0].cpu().numpy(), [0.25, -0.25])


def test_viewer_focus_env_selects_that_env():
    env, _ = make_env(n_envs=3, n_agents=3)
    viewer = Viewer(env, env_index=2)
    assert viewer.state.focus_env == 2
    g = viewer._geometry()
    np.testing.assert_allclose(g.pos, env.world.state.pos[2].cpu().numpy())


def test_period_key_requests_a_single_step():
    state, ctrl, _ = _controller()
    assert state.step_once is False
    ctrl.handle_event(_key(pygame.K_PERIOD))
    assert state.step_once is True


def test_step_gate_honors_pause_step_once_and_done():
    env, _ = make_env(n_agents=2)
    viewer = Viewer(env, size=(160, 120))

    assert viewer._steps_this_frame(done=False) == 1  # running at 1x
    viewer.state.paused = True
    assert viewer._steps_this_frame(done=False) == 0  # paused, no request

    viewer.state.step_once = True
    assert viewer._steps_this_frame(done=False) == 1  # one-shot honored
    assert viewer._steps_this_frame(done=False) == 0  # ... and consumed

    viewer.state.step_once = True
    assert viewer._steps_this_frame(done=True) == 0  # max_steps wins over a step request


def test_speed_multiplier_sets_steps_per_frame():
    """Fast speeds batch steps into one frame; slow ones spread one step over several."""
    env, _ = make_env(n_agents=2)
    viewer = Viewer(env, size=(160, 120))

    viewer.state.speed_index = SPEED_LADDER.index(4.0)
    assert [viewer._steps_this_frame(done=False) for _ in range(3)] == [4, 4, 4]

    viewer.state.speed_index = SPEED_LADDER.index(0.25)
    # 0.25x: exactly one step every fourth frame, and no drift over many frames.
    assert [viewer._steps_this_frame(done=False) for _ in range(8)] == [0, 0, 0, 1, 0, 0, 0, 1]

    # A one-shot step while paused always advances exactly one step, whatever the speed.
    viewer.state.speed_index = SPEED_LADDER.index(8.0)
    viewer.state.paused = True
    viewer.state.step_once = True
    assert viewer._steps_this_frame(done=False) == 1


def test_steps_per_frame_base_rate_decouples_playback_from_the_frame_rate():
    """A fractional base rate lets the window redraw faster than the sim advances.

    Push-T renders at 60 fps with dt=0.05, i.e. one step every third frame at real time;
    the ladder then multiplies that base instead of replacing it.
    """
    env, _ = make_env(n_agents=2)
    viewer = Viewer(env, size=(160, 120), fps=60, steps_per_frame=1 / 3)

    assert [viewer._steps_this_frame(done=False) for _ in range(6)] == [0, 0, 1, 0, 0, 1]

    viewer.state.speed_index = SPEED_LADDER.index(4.0)  # 4x real time -> 4 steps per 3 frames
    # The carry keeps the count on the ideal to within the one step still in the accumulator
    # (binary 1/3 rounds down, so an exact == would be off by one after enough frames).
    assert sum(viewer._steps_this_frame(done=False) for _ in range(30)) in (39, 40)


def test_up_down_keys_walk_the_speed_ladder_and_clamp():
    state, ctrl, _ = _controller()
    assert state.speed == 1.0

    ctrl.handle_event(_key(pygame.K_UP))
    assert state.speed == 2.0
    ctrl.handle_event(_key(pygame.K_DOWN))
    ctrl.handle_event(_key(pygame.K_DOWN))
    assert state.speed == 0.5

    for _ in range(20):  # clamped, not wrapped, at both ends
        ctrl.handle_event(_key(pygame.K_DOWN))
    assert state.speed == SPEED_LADDER[0]
    for _ in range(40):
        ctrl.handle_event(_key(pygame.K_UP))
    assert state.speed == SPEED_LADDER[-1]


def test_speed_badge_shows_only_off_1x():
    import pygame as pg

    from wmas.render.hud import draw_speed_badge
    from wmas.render.style import Style

    style = Style()
    state = ViewState(n_envs=1, enabled=set())

    def frame():
        surf = pg.Surface((320, 240))
        surf.fill(style.background)
        height = draw_speed_badge(surf, state, style)
        return height, pg.surfarray.array3d(surf).copy()

    h1, blank = frame()
    assert h1 == 0  # 1x draws nothing and reserves no space for the help panel

    state.speed_index = SPEED_LADDER.index(0.25)
    h2, slow = frame()
    assert h2 > 0 and not np.array_equal(blank, slow)


def test_plain_l_toggles_the_lidar_overlay_while_shift_l_cycles_the_mode():
    """Regression: the mode-cycle branch used to swallow "l", making the overlay untoggleable."""
    cycled = 0

    def on_cycle():
        nonlocal cycled
        cycled += 1

    state = ViewState(n_envs=2, enabled=set(DEFAULT_ENABLED))
    cam = Camera((-1.0, 1.0, -1.0, 1.0), (0, 0, 400, 400))
    ctrl = InteractionController(state, cam, on_cycle_lidar=on_cycle)

    assert "lidar" in state.enabled
    ctrl.handle_event(_key("l"))
    assert "lidar" not in state.enabled and cycled == 0  # plain l reaches the overlay
    ctrl.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l, mod=pygame.KMOD_SHIFT))
    assert cycled == 1 and "lidar" not in state.enabled  # shift+l only cycles the mode


def test_reward_hud_draws_and_no_ops_on_thin_data():
    import pygame as pg

    from wmas.render.hud import draw_reward_hud
    from wmas.render.style import Style

    env, scenario = make_env(n_agents=3)
    g = extract_geometry(env.world, 0, scenario=scenario)
    style = Style()
    rng = np.random.default_rng(0)

    def frame(rewards):
        surf = pg.Surface((320, 240))
        surf.fill(style.background)
        draw_reward_hud(surf, rewards, g, style)
        return pg.surfarray.array3d(surf).copy()

    blank = frame(None)
    assert np.array_equal(blank, frame(rng.normal(size=(1, 3))))  # one sample has no line
    assert not np.array_equal(blank, frame(rng.normal(size=(20, 3))))
    assert not np.array_equal(
        blank, frame(np.full((20, 3), 0.5))
    )  # flat window must not divide by 0


def test_viewer_reward_hud_changes_pixels_once_samples_accumulate():
    from wmas.render.style import Style

    env, scenario = make_env(n_agents=3, n_envs=2)
    viewer = Viewer(env, size=(320, 240), scenario=scenario, style=Style(reward_hud=True))
    empty = viewer.render_array(hud=True)
    for _ in range(5):
        viewer._append_reward_sample(torch.randn(env.n_envs, env.n_agents))
    assert len(viewer._reward_history) == 5
    assert not np.array_equal(empty, viewer.render_array(hud=True))


def test_focus_env_change_clears_the_trail_and_reward_buffers():
    """Both buffers hold one env's history, so switching env must not smear them together."""
    from wmas.render.style import Style

    env, scenario = make_env(n_envs=3, n_agents=2)
    viewer = Viewer(
        env,
        size=(200, 200),
        scenario=scenario,
        style=Style(reward_hud=True, trajectory_mode="trail"),
    )
    viewer._append_trail_sample()
    viewer._append_reward_sample(torch.randn(env.n_envs, env.n_agents))
    assert viewer._trail_history and viewer._reward_history

    viewer.state.focus_env = 2
    viewer.render_array()
    assert viewer._trail_history == [] and viewer._reward_history == []
    assert viewer._buffer_env == 2


def test_viewer_run_loop_paused_does_not_advance_the_sim(monkeypatch):
    """Pausing must freeze the simulation, not just the ``paused`` flag.

    ``_steps_this_frame`` is only consulted by :meth:`Viewer.run`, so a caller that drives its
    own ``env.step`` loop and calls ``render(mode="human")`` steps regardless of the flag —
    the agents keep moving with the HUD claiming "paused". Examples therefore hand the loop
    to the viewer (see ``examples/pusht_eval.py --window``), and this pins the behaviour.
    """
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env, _ = make_env(n_agents=3, n_envs=2)
    viewer = Viewer(env, size=(160, 120), fps=0)
    viewer.state.paused = True
    before = env.world.state.pos.clone()
    # max_steps counts *taken* steps, which never reach 3 while paused; close_when_done
    # still fires because `done` is evaluated on the step count, so bound the wait with a
    # quit request injected after a few frames.
    viewer.state.quit = False

    original = viewer._steps_this_frame
    frames = {"n": 0}

    def counting_should_step(done):
        frames["n"] += 1
        if frames["n"] >= 5:
            viewer.state.quit = True
        return original(done)

    viewer._steps_this_frame = counting_should_step
    viewer.run(max_steps=3, close_when_done=True)

    assert viewer._step_count == 0, "paused viewer advanced the simulation"
    assert torch.equal(env.world.state.pos, before), "paused viewer moved the agents"

    # ...and a one-shot step request while paused advances exactly one step.
    viewer._steps_this_frame = original
    viewer.state.quit = False
    viewer.state.step_once = True
    frames["n"] = 0
    viewer._steps_this_frame = counting_should_step
    viewer.run(max_steps=3, close_when_done=True)
    assert viewer._step_count == 1


def test_viewer_run_loop_smoke_headless(monkeypatch):
    # Drive the real run() loop against SDL's dummy video driver (no display needed) to
    # catch loop-wiring bugs; close_when_done makes a bounded run terminate.
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env, _ = make_env(n_agents=3, n_envs=2)
    viewer = Viewer(env, size=(160, 120), fps=0)
    viewer.run(max_steps=3, close_when_done=True)
    assert viewer._step_count == 3
