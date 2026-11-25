"""Near-field clipping and real geometry, without a distance-based success gate."""

import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.balloon.config import Config
from crazyflow.balloon.evaluate import summarize_trajectories
from crazyflow.balloon.observations import ideal_bbox


def test_on_axis_contact_precedes_optical_plane_invalidation_and_tracking_loss():
    cfg = Config()
    dt = cfg.dt / cfg.substeps
    states = np.zeros((int(round(1.6 / dt)) + 1, 1, 12))
    states[:, 0, 0] = np.arange(len(states)) * dt
    states[:, 0, 2] = 1.4
    states[:, 0, 6] = 1.0
    target = np.array([[1.0, 0.0, 1.4]])
    radius = np.array([0.15])
    boxes, visible = ideal_bbox(
        jnp.asarray(states[:, :, :3]),
        jnp.asarray(states[:, :, 3:6]),
        jnp.asarray(target),
        jnp.asarray(radius),
        cfg,
    )
    visible = np.asarray(visible)
    invalid_time = np.flatnonzero(~visible[:, 0])[0] * dt
    result = summarize_trajectories(
        states,
        target,
        radius,
        dt=dt,
        visible=visible[: -1 : cfg.substeps],
        visibility_dt=cfg.dt,
        tracking_timeout=cfg.max_unseen_seconds,
    )
    episode = result["per_episode"][0]
    assert episode["outcome"] == "hit"
    assert episode["event_time_s"] < invalid_time
    assert invalid_time + cfg.max_unseen_seconds < (len(states) - 1) * dt
    contact_index = int(episode["event_time_s"] / dt)
    assert visible[contact_index, 0]
    # The target fills/clips the image near contact, which remains a valid box.
    assert np.asarray(boxes)[contact_index, 0, 4] == 1


def test_near_edge_clipped_box_is_valid_even_with_center_outside_image():
    cfg = Config()
    position = jnp.array([0.0, 0.0, 1.4])
    depth = 0.35 - cfg.camera_offset[0]
    target = jnp.array([0.35, -(depth * np.tan(np.deg2rad(cfg.hfov_deg) / 2) + 0.04), 1.4])
    optical_center_x = -target[1] / (target[0] - cfg.camera_offset[0])
    assert optical_center_x > np.tan(np.deg2rad(cfg.hfov_deg) / 2)
    box, visible = ideal_bbox(position, jnp.zeros(3), target, jnp.array(0.15), cfg)
    assert bool(visible)
    box = np.asarray(box)
    assert box[4] == 1
    assert 0 < box[2] < 2
    assert box[0] + box[2] / 2 == pytest.approx(1.0)


@pytest.mark.parametrize("depth", [0.4, 4.0])
def test_empty_out_of_fov_box_is_invalid_at_near_and_far_distances(depth: float):
    cfg = Config()
    box, visible = ideal_bbox(
        jnp.array([0.0, 0.0, 1.4]),
        jnp.zeros(3),
        jnp.array([depth, depth * 3, 1.4]),
        jnp.array(0.15),
        cfg,
    )
    assert not bool(visible)
    np.testing.assert_array_equal(box, np.zeros(5))


def test_less_than_two_metres_without_contact_is_not_a_success():
    cfg = Config()
    states = np.zeros((2, 1, 12))
    states[..., 2] = 1.4
    target = np.array([[0.5, 0.0, 1.4]])
    radius = np.array([0.15])
    _, visible = ideal_bbox(
        jnp.asarray(states[:1, :, :3]),
        jnp.asarray(states[:1, :, 3:6]),
        jnp.asarray(target),
        jnp.asarray(radius),
        cfg,
    )
    assert bool(visible[0, 0])
    result = summarize_trajectories(
        states, target, radius, dt=cfg.dt, visible=np.asarray(visible), visibility_dt=cfg.dt
    )
    assert result["successes"] == 0
    assert result["outcomes"]["timeout"] == 1
    assert result["per_episode"][0]["minimum_surface_clearance_m"] > 0


def test_optical_plane_crossing_remains_invalid_without_truth_distance_exemption():
    cfg = Config()
    box, visible = ideal_bbox(
        jnp.array([0.0, 0.0, 1.4]), jnp.zeros(3), jnp.array([0.17, 0.22, 1.4]), jnp.array(0.15), cfg
    )
    assert not bool(visible)
    np.testing.assert_array_equal(box, np.zeros(5))


def test_off_axis_near_contact_uses_only_fixed_temporal_tracking_buffer():
    cfg = Config()
    dt = cfg.dt / cfg.substeps
    states = np.zeros((int(round(1.6 / dt)) + 1, 1, 12))
    states[:, 0, 0] = np.arange(len(states)) * dt
    states[:, 0, 2] = 1.4
    states[:, 0, 6] = 1.0
    target = np.array([[1.0, 0.2, 1.4]])
    radius = np.array([0.15])
    _, visible = ideal_bbox(
        jnp.asarray(states[:, :, :3]),
        jnp.asarray(states[:, :, 3:6]),
        jnp.asarray(target),
        jnp.asarray(radius),
        cfg,
    )
    visible = np.asarray(visible)
    invalid_time = np.flatnonzero(~visible[:, 0])[0] * dt
    settings = dict(dt=dt, visible=visible[: -1 : cfg.substeps], visibility_dt=cfg.dt)
    normal = summarize_trajectories(
        states, target, radius, tracking_timeout=cfg.max_unseen_seconds, **settings
    )
    short_buffer = summarize_trajectories(states, target, radius, tracking_timeout=0.02, **settings)
    contact_time = normal["per_episode"][0]["event_time_s"]
    assert invalid_time < contact_time < invalid_time + cfg.max_unseen_seconds
    assert normal["successes"] == 1
    assert short_buffer["outcomes"]["tracking_lost"] == 1
    assert short_buffer["successes"] == 0
