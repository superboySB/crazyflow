"""Evaluation tests exercise failure accounting and continuous event geometry."""

import numpy as np
import pytest

from crazyflow.balloon.evaluate import summarize_trajectories


def _states(positions: list, velocities: list | None = None) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    states = np.zeros((*positions.shape[:2], 12))
    states[..., :3] = positions
    if velocities is not None:
        states[..., 6:9] = velocities
    return states


def test_swept_contact_detects_tunneling_and_speed_limit():
    states = _states([[[0, 0, 1]], [[2, 0, 1]]], velocities=[[[2, 0, 0]], [[2, 0, 0]]])
    result = summarize_trajectories(
        states, [[1, 0, 1]], [0.1], dt=1, drone_radius=0.05, collision_offset=(0, 0, 0)
    )
    assert result["successes"] == 1
    assert result["per_episode"][0]["event_time_s"] == pytest.approx(0.425)
    assert result["per_episode"][0]["minimum_distance_m"] == pytest.approx(0.15)
    fast = summarize_trajectories(
        states,
        [[1, 0, 1]],
        [0.1],
        dt=1,
        drone_radius=0.05,
        max_impact_speed=1.5,
        collision_offset=(0, 0, 0),
    )
    assert fast["successes"] == 1
    assert fast["outcomes"]["too_fast_contact"] == 1
    assert fast["geometric_contact_rate"] == 1
    assert fast["safe_contact_rate"] == 0
    assert fast["per_episode"][0]["success"]


def test_all_failures_remain_in_denominator_and_post_failure_hits_do_not_count():
    positions = [
        [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
        [[1, 0, 1], [0, 0, 0], [0, 0, 1], [0, 0, 1]],
        [[2, 0, 1], [1, 0, 1], [0, 0, 1], [0, 0, 1]],
    ]
    result = summarize_trajectories(
        _states(positions),
        [[1, 0, 1]] * 4,
        [0.1] * 4,
        dt=0.5,
        visible=[[True, True, False, True], [True, True, False, True]],
        visibility_dt=0.5,
        tracking_timeout=0.5,
    )
    assert result["episodes"] == 4
    assert result["successes"] == 1
    assert result["success_rate"] == 0.25
    assert result["outcomes"]["floor"] == 1
    assert result["outcomes"]["tracking_lost"] == 1
    assert result["outcomes"]["timeout"] == 1
    assert sum(result["outcomes"].values()) == 4
    assert result["distance_metric_episodes"] == 4


def test_floor_and_bounds_are_detected_between_substeps():
    positions = [[[0, 0, 1], [0, 0, 1]], [[0, 0, -1], [4, 0, 1]]]
    result = summarize_trajectories(
        _states(positions), [[3, 0, 1]] * 2, [0.1, 0.1], dt=1, min_altitude=0.2, max_range=2
    )
    assert result["per_episode"][0]["outcome"] == "floor"
    assert result["per_episode"][0]["event_time_s"] == pytest.approx(0.4)
    assert result["per_episode"][1]["outcome"] == "out_of_bounds"
    assert result["per_episode"][1]["event_time_s"] == pytest.approx(0.5)


def test_nonfinite_and_initial_overlap_are_not_silently_dropped():
    states = _states([[[0, 0, 1], [1, 0, 1]], [[np.nan, 0, 1], [2, 0, 1]]])
    result = summarize_trajectories(states, [[2, 0, 1], [1, 0, 1]], [0.1, 0.1], dt=0.1)
    assert result["outcomes"]["nonfinite"] == 1
    assert result["outcomes"]["hit"] == 1
    assert result["success_rate"] == 0.5
    assert result["per_episode"][1]["event_time_s"] == 0


def test_short_dropout_recovers_and_contact_after_tracking_failure_does_not_count():
    states = _states([[[0, 0, 1]], [[0, 0, 1]], [[0, 0, 1]], [[1, 0, 1]]])
    recovered = summarize_trajectories(
        states,
        [[1, 0, 1]],
        [0.1],
        dt=0.2,
        visible=[[False], [True], [True]],
        visibility_dt=0.2,
        tracking_timeout=0.3,
    )
    lost = summarize_trajectories(
        states,
        [[1, 0, 1]],
        [0.1],
        dt=0.2,
        visible=[[False], [False], [True]],
        visibility_dt=0.2,
        tracking_timeout=0.3,
    )
    assert recovered["successes"] == 1
    assert lost["successes"] == 0
    assert lost["per_episode"][0]["outcome"] == "tracking_lost"
    assert lost["per_episode"][0]["event_time_s"] == pytest.approx(0.3)


def test_geometry_rejects_invalid_target_or_time():
    states = _states([[[0, 0, 1]], [[1, 0, 1]]])
    with pytest.raises(ValueError):
        summarize_trajectories(states, [[1, 0, 1]], [-0.1], dt=0.1)
    with pytest.raises(ValueError):
        summarize_trajectories(states, [[1, 0, 1]], [0.1], dt=0)


def test_official_collision_offset_rotates_with_attitude():
    states = _states([[[0, 0, 1]], [[0, 0, 1]]])
    states[..., 3] = np.pi / 2
    target = [[0, -0.22, 1]]
    offset = summarize_trajectories(states, target, [0.05], dt=0.1, drone_radius=0.16)
    no_offset = summarize_trajectories(
        states, target, [0.05], dt=0.1, drone_radius=0.16, collision_offset=(0, 0, 0)
    )
    assert offset["outcomes"]["hit"] == 1
    assert no_offset["outcomes"]["timeout"] == 1


def test_visibility_and_distances_stop_at_first_contact():
    states = _states([[[0, 0, 1]], [[1, 0, 1]], [[2, 0, 1]]])
    result = summarize_trajectories(
        states,
        [[1, 0, 1]],
        [0.1],
        dt=0.5,
        visible=[[True], [False]],
        visibility_dt=0.5,
        tracking_timeout=0.1,
    )
    assert result["outcomes"]["hit"] == 1
    assert result["per_episode"][0]["visible_fraction"] == 1
    assert result["per_episode"][0]["minimum_surface_clearance_m"] == pytest.approx(0)
    assert 0 < result["success_rate_wilson95"][0] < 1
    assert result["success_rate_wilson95"][1] == pytest.approx(1)


def test_failure_wins_a_simultaneous_contact():
    states = _states([[[0, 0, 0.1]], [[0, 0, 0.1]]])
    result = summarize_trajectories(states, [[0, 0, 0.1]], [0.1], dt=0.1)
    assert result["outcomes"]["floor"] == 1
    assert result["successes"] == 0
