"""Read-only XY diagnostics retain metric geometry and the replay time boundary."""

from itertools import combinations

import numpy as np
import pytest

from crazyflow.balloon.xy_map import (
    Bounds,
    Projection,
    Rect,
    episode_bounds,
    layout,
    motion_kind,
    trajectory_prefix,
    velocity_arrow,
)


def _states(positions: list[list[float]]) -> np.ndarray:
    result = np.zeros((len(positions), 12))
    result[:, :3] = positions
    return result


def test_projection_keeps_metres_equal_and_xy_axes_unambiguous() -> None:
    bounds = Bounds(-4.0, 4.0, -2.0, 2.0)
    viewport = Rect(17, 29, 600, 420)
    projection = Projection(bounds, viewport)
    points = projection.project(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    right, up = points[1] - points[0], points[2] - points[0]
    assert right[0] > 0 and up[1] > 0
    assert right[1] == pytest.approx(0)
    assert up[0] == pytest.approx(0)
    assert right[0] == pytest.approx(up[1])
    assert right[0] == pytest.approx(projection.pixels_per_m)
    plot = projection.plot_rect
    corners = projection.project(np.array([[-4.0, -2.0], [4.0, 2.0]]))
    np.testing.assert_allclose(
        corners, [[plot.left, plot.bottom], [plot.left + plot.width, plot.bottom + plot.height]]
    )
    assert plot.left >= viewport.left and plot.bottom >= viewport.bottom
    assert plot.left + plot.width <= viewport.left + viewport.width
    assert plot.bottom + plot.height <= viewport.bottom + viewport.height
    assert plot.width / plot.height == pytest.approx(2)


def test_fixed_episode_bounds_cover_target_future_and_one_second_arrow() -> None:
    states = _states([[0.0, 0.0, 1.4], [0.5, -0.4, 1.6], [1.0, -1.0, 2.0]])
    states[-1, 6:8] = [4.0, -3.0]
    initial = np.array([2.0, 1.0, 2.0])
    velocity = np.array([-1.2, 0.7, -0.2])
    original_states, original_target = states.copy(), initial.copy()
    bounds = episode_bounds(states, initial, velocity, 3.0)
    points = np.vstack(
        (
            states[:, :2],
            states[:, :2] + states[:, 6:8],
            initial[:2],
            (initial + 3 * velocity)[:2],
            (initial + 4 * velocity)[:2],
        )
    )
    assert np.all(points[:, 0] >= bounds.xmin) and np.all(points[:, 0] <= bounds.xmax)
    assert np.all(points[:, 1] >= bounds.ymin) and np.all(points[:, 1] <= bounds.ymax)
    # Pausing, scrubbing and rewinding affect the visible prefix, never map scale.
    for elapsed in (0.0, 0.15, 0.4, 0.1):
        trajectory_prefix(states, 0.2, elapsed)
        assert episode_bounds(states, initial, velocity, 3.0) == bounds
    np.testing.assert_array_equal(states, original_states)
    np.testing.assert_array_equal(initial, original_target)


def test_stationary_coincident_geometry_still_has_a_finite_map_scale() -> None:
    states = _states([[1.0, -2.0, 3.0], [1.0, -2.0, 3.0]])
    bounds = episode_bounds(states, np.array([1.0, -2.0, 3.0]), np.zeros(3), 3.0)
    assert bounds.xmax > bounds.xmin and bounds.ymax > bounds.ymin
    projection = Projection(bounds, Rect(0, 0, 240, 180))
    assert np.isfinite(projection.pixels_per_m) and projection.pixels_per_m > 0
    assert np.isfinite(projection.project(states[:, :2])).all()


def test_failed_flight_records_do_not_poison_the_fixed_map_bounds() -> None:
    states = _states([[0.0, 0.0, 1.4], [1.0, -1.0, 1.5], [np.nan, np.nan, np.nan]])
    args = (np.array([2.0, 0.0, 2.0]), np.array([0.3, 0.1, 0.2]), 3.0)
    assert episode_bounds(states, *args) == episode_bounds(states[:2], *args)


@pytest.mark.parametrize(
    "velocity,kind",
    [
        ([0.0, 0.0, 0.0], "stationary"),
        ([0.0, 0.0, 1.5], "vertical"),
        ([0.0, 0.0, -1.5], "vertical"),
        ([0.3, -0.4, 1.0], "moving"),
    ],
)
def test_motion_labels_do_not_confuse_vertical_motion_with_stationary(
    velocity: list[float], kind: str
) -> None:
    assert motion_kind(np.asarray(velocity)) == kind


@pytest.mark.parametrize("velocity", [[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 0.0, -1.5]])
def test_stationary_and_vertical_targets_do_not_invent_an_xy_arrow(velocity: list[float]) -> None:
    arrow = velocity_arrow(np.array([2.0, -1.0, 3.0]), np.asarray(velocity))
    assert arrow.size == 0


def test_velocity_arrow_is_one_second_displacement_and_keeps_signs() -> None:
    position = np.array([2.0, 1.0, 4.0])
    velocity = np.array([-0.3, 0.4, -1.0])
    arrow = velocity_arrow(position, velocity)
    assert arrow.shape == (5, 2)
    np.testing.assert_allclose(arrow[0], position[:2])
    np.testing.assert_allclose(arrow[1], (position + velocity)[:2])
    assert np.linalg.norm(arrow[1] - arrow[0]) == pytest.approx(0.5)
    longer = velocity_arrow(position, velocity, seconds=2)
    np.testing.assert_allclose(longer[1] - longer[0], 2 * (arrow[1] - arrow[0]))
    # Vertical velocity belongs to its text label; it must not change XY geometry.
    horizontal = velocity.copy()
    horizontal[2] = 0
    np.testing.assert_allclose(velocity_arrow(position, horizontal), arrow)


def test_trajectory_prefix_interpolates_now_without_showing_future() -> None:
    states = _states([[0.0, 0.0, 1.0], [1.0, -2.0, 2.0], [5.0, -4.0, 4.0]])
    original = states.copy()
    start = trajectory_prefix(states, 0.2, 0)
    np.testing.assert_allclose(start, np.broadcast_to(states[0, :3], start.shape))
    exact = trajectory_prefix(states, 0.2, 0.2)
    np.testing.assert_allclose(exact[:2], states[:2, :3])
    np.testing.assert_allclose(exact[-1], states[1, :3])
    assert not np.any(np.all(exact == states[-1, :3], axis=1))
    prefix = trajectory_prefix(states, 0.2, 0.3)
    np.testing.assert_allclose(prefix, [[0, 0, 1], [1, -2, 2], [3, -3, 3]])
    assert not np.any(np.all(prefix == states[-1, :3], axis=1))
    # Going back to an earlier time must shrink history instead of retaining a trail.
    np.testing.assert_allclose(trajectory_prefix(states, 0.2, 0.1), [[0, 0, 1], [0.5, -1, 1.5]])
    end = trajectory_prefix(states, 0.2, 2)
    np.testing.assert_allclose(end[: len(states)], states[:, :3])
    np.testing.assert_allclose(end[-1], states[-1, :3])
    np.testing.assert_array_equal(states, original)


@pytest.mark.parametrize("width,height", [(960, 720), (1440, 960), (1920, 1080)])
def test_resized_layout_preserves_disjoint_panels_and_world_mouse_region(
    width: int, height: int
) -> None:
    panels = layout(width, height)
    rectangles = (panels.world, panels.map, panels.fpv, panels.telemetry)
    for rect in rectangles:
        assert rect.width > 0 and rect.height > 0
        assert rect.left >= 0 and rect.bottom >= 0
        assert rect.left + rect.width <= width
        assert rect.bottom + rect.height <= height
    for a, b in combinations(rectangles, 2):
        overlap_x = min(a.left + a.width, b.left + b.width) - max(a.left, b.left)
        overlap_y = min(a.bottom + a.height, b.bottom + b.height) - max(a.bottom, b.bottom)
        assert overlap_x <= 0 or overlap_y <= 0
    for rect in rectangles:
        x, y = rect.left + rect.width / 2, rect.bottom + rect.height / 2
        assert rect.contains(x, y)
        assert panels.world.contains(x, y) is (rect is panels.world)
    assert not panels.world.contains(-1, -1)
    assert not panels.world.contains(width + 1, height + 1)


def test_mouse_routing_uses_exclusive_top_and_right_edges() -> None:
    rect = Rect(10, 20, 30, 40)
    assert rect.contains(10, 20)
    assert rect.contains(39.9, 59.9)
    assert not rect.contains(rect.right, 30)
    assert not rect.contains(20, rect.top)


def test_resize_changes_only_projection_not_frozen_metric_bounds() -> None:
    states = _states([[0, 0, 1.4], [2, 1, 2]])
    bounds = episode_bounds(states, np.array([3.0, -1.0, 2.0]), np.array([0.3, 0.4, 0.1]), 3)
    small = Projection(bounds, layout(960, 720).map)
    large = Projection(bounds, layout(1920, 1080).map)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    for projection in (small, large):
        pixels = projection.project(points)
        np.testing.assert_allclose(pixels[1:] - pixels[0], projection.pixels_per_m * np.eye(2))
    assert large.pixels_per_m > small.pixels_per_m
