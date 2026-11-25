"""Geometry and causal sensor-only contract checks for the BPTT policy."""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.balloon.observations import actor_features, clip_bbox, feature_dim, ideal_bbox


@dataclass(frozen=True)
class SensorConfig:
    """Minimal immutable sensor configuration used independently of training."""

    dt: float = 0.02
    history: int = 5
    hfov_deg: float = 90.0
    image_width: int = 640
    image_height: int = 480
    camera_offset: tuple[float, float, float] = (0.03, 0.0, 0.0)


def test_noisy_tracker_clips_edges_and_preserves_partial_intersections():
    boxes = jnp.array(
        [
            [0.98, 0.0, 0.2, 0.4, 1.0],
            [1.1, 0.0, 0.4, 0.4, 1.0],
            [1.3, 0.0, 0.2, 0.4, 1.0],
            [0.0, 0.0, -0.1, 0.4, 1.0],
            [0.0, 0.0, 0.2, 0.4, 0.0],
        ]
    )
    clipped = clip_bbox(boxes)
    np.testing.assert_allclose(
        clipped[:2], [[0.94, 0.0, 0.12, 0.4, 1.0], [0.95, 0.0, 0.1, 0.4, 1.0]], atol=1e-6
    )
    np.testing.assert_array_equal(clipped[2:], 0.0)
    np.testing.assert_allclose(clip_bbox(clipped), clipped, atol=1e-7)
    assert np.isfinite(jax.jacrev(clip_bbox)(boxes)).all()


def test_exact_on_axis_sphere_and_body_image_conventions():
    cfg = SensorConfig()
    bbox, visible = ideal_bbox(jnp.zeros(3), jnp.zeros(3), jnp.array([2.0, 0.0, 0.0]), 0.2, cfg)
    width = 2 * 0.2 / np.sqrt(1.97**2 - 0.2**2)
    assert bool(visible)
    np.testing.assert_allclose(bbox, [0.0, 0.0, width, width * 640 / 480, 1.0], rtol=1e-5)
    upper_left, _ = ideal_bbox(jnp.zeros(3), jnp.zeros(3), jnp.array([2.0, 0.2, 0.2]), 0.1, cfg)
    assert upper_left[0] < 0 and upper_left[1] < 0


def test_rotated_camera_and_world_translation_invariance():
    cfg = SensorConfig()
    position = jnp.array([0.3, -0.5, 1.0])
    target = position + jnp.array([0.0, 2.0, 0.0])
    rpy = jnp.array([0.0, 0.0, jnp.pi / 2])
    bbox, valid = ideal_bbox(position, rpy, target, 0.15, cfg)
    shifted, shifted_valid = ideal_bbox(position + 7.0, rpy, target + 7.0, 0.15, cfg)
    assert bool(valid) and bool(shifted_valid)
    np.testing.assert_allclose(bbox[:2], 0.0, atol=1e-6)
    np.testing.assert_allclose(bbox, shifted, atol=1e-6)


def test_clipped_intersection_and_invalid_geometry_remain_finite():
    cfg = SensorConfig()
    targets = jnp.array([[2.0, -1.95, 0.0], [-2.0, 0.0, 0.0], [0.08, 0.0, 0.0], [2.0, -5.0, 0.0]])
    boxes, visible = ideal_bbox(jnp.zeros(3), jnp.zeros(3), targets, 0.2, cfg)
    np.testing.assert_array_equal(visible, [True, False, False, False])
    assert np.isfinite(boxes).all()
    np.testing.assert_array_equal(boxes[1:], 0.0)
    np.testing.assert_allclose(boxes[0, 0] + boxes[0, 2] / 2, 1.0, atol=1e-6)
    jacobian = jax.jacrev(lambda p: ideal_bbox(p, jnp.zeros(3), targets, 0.2, cfg)[0])(jnp.zeros(3))
    assert np.isfinite(jacobian).all()
    np.testing.assert_array_equal(jacobian[1:], 0.0)


def test_projection_keeps_useful_position_and_attitude_gradients():
    cfg = SensorConfig()

    def project(p: jax.Array, r: jax.Array) -> jax.Array:
        return ideal_bbox(p, r, jnp.array([2.0, 0.1, 0.1]), 0.15, cfg)[0]

    position_jacobian, attitude_jacobian = jax.jacrev(project, argnums=(0, 1))(
        jnp.zeros(3), jnp.zeros(3)
    )
    assert np.isfinite(position_jacobian).all() and np.isfinite(attitude_jacobian).all()
    assert position_jacobian[2, 0] > 0  # Forward motion grows the apparent sphere.
    assert position_jacobian[0, 1] > 0  # Moving left shifts its image right.
    assert abs(attitude_jacobian[0, 2]) > 0.1


def test_actor_feature_contract_causal_log_scale_and_command_history():
    cfg = SensorConfig()
    times = jnp.arange(cfg.history) * cfg.dt
    growth_rate = 0.7
    extents = 0.12 * jnp.exp(growth_rate * times)
    boxes = jnp.stack((times * 0.2, times * -0.3, extents, extents, jnp.ones_like(times)), -1)
    commands = jnp.arange(4 * cfg.history, dtype=jnp.float32).reshape(cfg.history, 4)
    rates = jnp.array([0.1, -0.2, 0.3])
    features = actor_features(boxes, jnp.eye(3), rates, commands, cfg)
    assert features.shape == (feature_dim(cfg),)
    np.testing.assert_allclose(features[:9], np.eye(3).ravel())
    np.testing.assert_allclose(features[9:12], rates / 5.0)
    np.testing.assert_allclose(features[12 : 12 + 5 * cfg.history], boxes.ravel())
    np.testing.assert_allclose(
        features[12 + 5 * cfg.history : 12 + 9 * cfg.history], commands.ravel()
    )
    expected_ray = np.array([1.0, -boxes[-1, 0], -boxes[-1, 1] * 480 / 640])
    expected_ray /= np.linalg.norm(expected_ray)
    np.testing.assert_allclose(features[-8:-5], expected_ray, atol=1e-6)
    np.testing.assert_allclose(features[-4], growth_rate, rtol=2e-4)
    np.testing.assert_allclose(features[-3:], [0.2, -0.3, growth_rate], rtol=2e-4)


def test_invalid_current_packet_has_no_truth_fallback_or_spurious_difference():
    cfg = SensorConfig()
    boxes = jnp.tile(jnp.array([0.2, -0.1, 0.2, 0.2, 1.0]), (cfg.history, 1))
    boxes = boxes.at[-1].set(jnp.array([0.8, 0.7, 0.4, 0.3, 0.0]))
    features = actor_features(boxes, jnp.eye(3), jnp.zeros(3), jnp.zeros((cfg.history, 4)), cfg)
    np.testing.assert_array_equal(features[-8:], 0.0)
    assert np.isfinite(features).all()
    grad = jax.jacrev(
        lambda box: actor_features(box, jnp.eye(3), jnp.zeros(3), jnp.zeros((cfg.history, 4)), cfg)
    )(jnp.zeros((cfg.history, 5)))
    assert np.isfinite(grad).all()


def test_actor_features_are_batched_and_jittable():
    cfg = SensorConfig()
    boxes = jnp.tile(jnp.array([0.0, 0.0, 0.1, 0.1, 1.0]), (3, cfg.history, 1))
    rotations = jnp.tile(jnp.eye(3), (3, 1, 1))
    rates, commands = jnp.zeros((3, 3)), jnp.zeros((3, cfg.history, 4))
    batched = jax.jit(partial(actor_features, cfg=cfg))(boxes, rotations, rates, commands)
    single = actor_features(boxes[0], rotations[0], rates[0], commands[0], cfg)
    assert batched.shape == (3, feature_dim(cfg))
    np.testing.assert_allclose(batched, jnp.broadcast_to(single, batched.shape), atol=1e-6)
