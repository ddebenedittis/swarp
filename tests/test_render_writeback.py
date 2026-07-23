"""M4 write-back: drag an agent to reposition it, right-click to move its goal."""

import numpy as np
import pygame
import torch

from wmas import Environment, NavigationScenario
from wmas.render.camera import Camera
from wmas.render.geometry import extract_geometry
from wmas.render.input import InteractionController, ViewState
from wmas.render.overlays import DEFAULT_ENABLED
from wmas.render.viewer import Viewer


def make_env(n_envs=2, n_agents=2, n_obstacles=0, device="cpu"):
    scenario = NavigationScenario(n_agents=n_agents, n_obstacles=n_obstacles)
    env = Environment(scenario, n_envs=n_envs, device=device, dt=0.1, seed=0)
    env.reset()
    return env, scenario


def _controller(geometry, camera, **callbacks):
    state = ViewState(n_envs=4, enabled=set(DEFAULT_ENABLED))
    ctrl = InteractionController(state, camera, geometry_getter=lambda: geometry, **callbacks)
    return state, ctrl


def _md(button, pos):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=button, pos=pos)


# ---------------------------------------------------------- controller wiring


def test_left_press_on_agent_selects_and_drag_reports_world_pos():
    env, scenario = make_env(n_agents=2)
    g = extract_geometry(env.world, 0, scenario=scenario)
    cam = Camera(g.bounds, (0, 0, 400, 400))
    calls = []
    state, ctrl = _controller(g, cam, on_drag_agent=lambda i, xy: calls.append((i, xy)))

    cx, cy = cam.world_to_screen(g.pos[1])
    ctrl.handle_event(_md(1, (float(cx), float(cy))))
    assert state.selected_agent == 1

    new_pt = (float(cx) + 30, float(cy) - 20)
    ctrl.handle_event(
        pygame.event.Event(pygame.MOUSEMOTION, pos=new_pt, rel=(30, -20), buttons=(1, 0, 0))
    )
    assert len(calls) == 1 and calls[0][0] == 1
    np.testing.assert_allclose(calls[0][1], cam.screen_to_world(new_pt), atol=1e-6)

    # releasing ends the drag: subsequent motion must not report
    ctrl.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=new_pt))
    ctrl.handle_event(
        pygame.event.Event(
            pygame.MOUSEMOTION, pos=(new_pt[0] + 5, new_pt[1]), rel=(5, 0), buttons=(0, 0, 0)
        )
    )
    assert len(calls) == 1


def test_left_press_on_empty_space_selects_nothing():
    env, scenario = make_env(n_agents=2)
    g = extract_geometry(env.world, 0, scenario=scenario)
    cam = Camera(g.bounds, (0, 0, 400, 400))
    state, ctrl = _controller(g, cam, on_drag_agent=lambda i, xy: None)
    ctrl.handle_event(_md(1, (-50.0, -50.0)))
    assert state.selected_agent is None


def test_right_click_places_goal_for_selected_agent():
    env, scenario = make_env(n_agents=2)
    g = extract_geometry(env.world, 0, scenario=scenario)
    cam = Camera(g.bounds, (0, 0, 400, 400))
    calls = []
    state, ctrl = _controller(g, cam, on_place_goal=lambda i, xy: calls.append((i, xy)))

    cx, cy = cam.world_to_screen(g.pos[0])
    ctrl.handle_event(_md(1, (float(cx), float(cy))))  # select agent 0
    target = (250.0, 90.0)
    ctrl.handle_event(_md(3, target))  # right-click -> goal there
    assert len(calls) == 1 and calls[0][0] == 0
    np.testing.assert_allclose(calls[0][1], cam.screen_to_world(target), atol=1e-6)


# ------------------------------------------------------------- viewer writes


def test_viewer_write_agent_pos_persists_into_next_step():
    env, _ = make_env(n_envs=2, n_agents=1)  # single agent => no collision nudges
    viewer = Viewer(env)
    viewer.state.focus_env = 1
    viewer._write_agent_pos(0, (0.2, -0.3))
    np.testing.assert_allclose(env.world.state.pos[1, 0].cpu().numpy(), [0.2, -0.3], atol=1e-6)

    with torch.no_grad():
        env.step(torch.zeros(env.n_envs, env.n_agents, env.world.act_dim, dtype=env.dtype))
    # velocity was zeroed, so a zero-action step leaves the agent where we dropped it
    np.testing.assert_allclose(env.world.state.pos[1, 0].cpu().numpy(), [0.2, -0.3], atol=1e-4)


def test_viewer_write_goal_updates_world_goals():
    env, _ = make_env(n_envs=2, n_agents=2)
    viewer = Viewer(env)
    viewer.state.focus_env = 0
    viewer._write_goal(1, (0.5, 0.4))
    np.testing.assert_allclose(env.world.goals[0, 1].cpu().numpy(), [0.5, 0.4], atol=1e-6)
