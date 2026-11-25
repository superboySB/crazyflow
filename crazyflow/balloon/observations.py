"""Causal visual and inertial features for the balloon BPTT policy.

The simulator-only projection accepts geometry. ``actor_features`` accepts only
received bounding boxes, attitude, gyroscope measurements and issued commands.
The image convention is right/down, while body axes are forward/left/up.
"""

from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp

from crazyflow.balloon.camera import intrinsics, rotation_body_to_optical

GYRO_SCALE_RAD_S = 5.0


class ObservationConfig(Protocol):
    """Configuration fields required by the sensor and feature functions."""

    dt: float
    history: int
    hfov_deg: float
    image_width: int
    image_height: int
    camera_offset: tuple[float, float, float]
    camera_rpy: tuple[float, float, float]
    camera_intrinsics: tuple[float, float, float, float] | None


def _rpy_matrix(rpy: jnp.ndarray) -> jnp.ndarray:
    """Return body-to-world Rz(yaw) Ry(pitch) Rx(roll), in radians."""
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.stack(
        (
            jnp.stack((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), -1),
            jnp.stack((sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), -1),
            jnp.stack((-sp, cp * sr, cp * cr), -1),
        ),
        -2,
    )


def camera_relative(
    position: jnp.ndarray, rpy: jnp.ndarray, target: jnp.ndarray, cfg: ObservationConfig
) -> jnp.ndarray:
    """Simulator-only target vector in the configured optical frame."""
    relative = jnp.einsum("...ji,...j->...i", _rpy_matrix(rpy), target - position)
    relative = relative - jnp.asarray(cfg.camera_offset)
    return jnp.einsum("ij,...j->...i", jnp.asarray(rotation_body_to_optical(cfg)), relative)


def soft_fov_cost(
    position: jnp.ndarray, rpy: jnp.ndarray, target: jnp.ndarray, cfg: ObservationConfig
) -> jnp.ndarray:
    """Differentiable rectangular-frustum margin from the same intrinsics/extrinsics.

    This is privileged geometric supervision, not an actor observation. The
    10% center margin retains gradients when the actual tracker box is empty.
    """
    relative = camera_relative(position, rpy, target, cfg)
    direction = relative / jnp.sqrt(jnp.sum(relative**2, axis=-1, keepdims=True) + 1e-6)
    x, y, z = direction[..., 0], direction[..., 1], direction[..., 2]
    k = intrinsics(cfg)
    left, right = 0.9 * k[0, 2] / k[0, 0], 0.9 * (cfg.image_width - k[0, 2]) / k[0, 0]
    top, bottom = 0.9 * k[1, 2] / k[1, 1], 0.9 * (cfg.image_height - k[1, 2]) / k[1, 1]
    violations = jnp.stack(
        (-x - left * z, x - right * z, -y - top * z, y - bottom * z, 0.05 - z), axis=-1
    )
    return jnp.sum(jnp.maximum(violations, 0.0) ** 2, axis=-1)


def ideal_bbox(
    position: jnp.ndarray,
    rpy: jnp.ndarray,
    target: jnp.ndarray,
    radius: jnp.ndarray,
    cfg: ObservationConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project a sphere into an ideal, image-clipped bounding box.

    Inputs have trailing shapes 3, 3, 3 and scalar respectively; leading batch
    axes broadcast. The default camera points along body +X. Mount RPY and
    calibrated K are explicit.
    Its translation is ``cfg.camera_offset`` in body metres. Position/target
    use the same world frame, and roll/pitch/yaw are in radians.

    Returns ``bbox[..., 5] = [cx, cy, width, height, valid]`` and a boolean
    visibility mask. Centers use [-1, 1]; extents use fractions of the image
    half-width/half-height, so a full-image extent equals 2. The projected
    sphere is an ellipse; bounds are taken over its actual intersection with
    the image, including near-field and corner clipping. Rear spheres, empty
    intersections and spheres intersecting the optical plane produce an
    all-zero invalid box.

    Noise and delivery delays belong to the external sensor queue, so this
    function is deterministic and differentiable within visibility branches.
    """
    position, rpy, target = map(jnp.asarray, (position, rpy, target))
    optical = camera_relative(position, rpy, target, cfg)
    optical_x, optical_y, optical_z = optical[..., 0], optical[..., 1], optical[..., 2]
    sphere_radius = jnp.asarray(radius)
    safe_radius = jnp.maximum(sphere_radius, 1e-6)
    # Keep every branch finite before selecting the visibility mask: invalid
    # branches are still evaluated by JAX's reverse-mode differentiation.
    safe_depth = jnp.maximum(optical_z, safe_radius + 1e-4)
    denominator = jnp.maximum(safe_depth**2 - safe_radius**2, 1e-6)
    k = jnp.asarray(intrinsics(cfg))

    # In the normalized optical image plane, the silhouette has center m and
    # shape matrix B: (p-m).T @ inv(B) @ (p-m) <= 1. Clipping the full ellipse's
    # axis-aligned box would wrongly report some empty image-corner overlaps.
    radius_squared = safe_radius**2
    shape_factor = radius_squared / denominator**2
    center_x, center_y = optical_x * safe_depth / denominator, optical_y * safe_depth / denominator
    variance_x = shape_factor * (denominator + optical_x**2)
    variance_y = shape_factor * (denominator + optical_y**2)
    covariance = shape_factor * optical_x * optical_y
    distance_squared = denominator + optical_x**2 + optical_y**2
    image_left, image_right = -k[0, 2] / k[0, 0], (cfg.image_width - k[0, 2]) / k[0, 0]
    image_top, image_bottom = -k[1, 2] / k[1, 1], (cfg.image_height - k[1, 2]) / k[1, 1]

    def bounds_in_strip(
        center: jnp.ndarray,
        other_center: jnp.ndarray,
        variance: jnp.ndarray,
        other_variance: jnp.ndarray,
        conditional_variance: jnp.ndarray,
        strip_lower: jnp.ndarray,
        strip_upper: jnp.ndarray,
        image_lower: jnp.ndarray,
        image_upper: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Ellipse extrema within the other coordinate's image strip.

        A convex ellipse cut by a strip reaches each coordinate extremum at
        its unconstrained tangent or at one of the two strip edges. Taking
        these six candidates before clipping the remaining coordinate gives
        the bounding box of the visible silhouette, rather than its amodal box.
        """
        axis_radius = jnp.sqrt(jnp.maximum(variance, 1e-12))
        signs = jnp.array([-1.0, 1.0], dtype=axis_radius.dtype)
        extrema = center[..., None] + axis_radius[..., None] * signs
        other_at_extrema = (
            other_center[..., None] + covariance[..., None] / axis_radius[..., None] * signs
        )
        extrema_valid = (other_at_extrema >= strip_lower) & (other_at_extrema <= strip_upper)
        strip_edges = jnp.stack((strip_lower, strip_upper))
        offset = strip_edges - other_center[..., None]
        remaining = 1 - offset**2 / jnp.maximum(other_variance[..., None], 1e-12)
        edge_center = (
            center[..., None]
            + covariance[..., None] / jnp.maximum(other_variance[..., None], 1e-12) * offset
        )
        radicand = conditional_variance[..., None] * jnp.maximum(remaining, 0.0)
        # sqrt(0) has an infinite derivative even if its branch is masked.
        # Select a finite root first, then restore the exact zero tangent.
        edge_radius = jnp.where(radicand > 0, jnp.sqrt(jnp.maximum(radicand, 1e-12)), 0.0)
        intersections = edge_center[..., None] + edge_radius[..., None] * signs
        intersections = intersections.reshape(intersections.shape[:-2] + (4,))
        intersections_valid = jnp.repeat(remaining >= 0, 2, axis=-1)
        candidates = jnp.concatenate((extrema, intersections), -1)
        valid = jnp.concatenate((extrema_valid, intersections_valid), -1)
        lower = jnp.maximum(jnp.min(jnp.where(valid, candidates, image_upper), -1), image_lower)
        upper = jnp.minimum(jnp.max(jnp.where(valid, candidates, image_lower), -1), image_upper)
        return lower, upper

    # These Schur complements avoid subtracting nearly equal covariance terms
    # for the long ellipses produced by wide-angle, off-axis observations.
    conditional_x = radius_squared * distance_squared / (denominator * (denominator + optical_y**2))
    conditional_y = radius_squared * distance_squared / (denominator * (denominator + optical_x**2))
    left, right = bounds_in_strip(
        center_x,
        center_y,
        variance_x,
        variance_y,
        conditional_x,
        image_top,
        image_bottom,
        image_left,
        image_right,
    )
    top, bottom = bounds_in_strip(
        center_y,
        center_x,
        variance_y,
        variance_x,
        conditional_y,
        image_left,
        image_right,
        image_top,
        image_bottom,
    )
    optical_lower = jnp.stack((left, top), -1)
    optical_upper = jnp.stack((right, bottom), -1)
    scale = jnp.array((2 * k[0, 0] / cfg.image_width, 2 * k[1, 1] / cfg.image_height))
    offset = jnp.array((2 * k[0, 2] / cfg.image_width - 1, 2 * k[1, 2] / cfg.image_height - 1))
    lower, upper = optical_lower * scale + offset, optical_upper * scale + offset
    extent = jnp.maximum(upper - lower, 0.0)
    visible = (
        (sphere_radius > 0.0) & (optical_z > safe_radius + 1e-4) & jnp.all(extent > 0.0, axis=-1)
    )
    center = 0.5 * (lower + upper)
    values = jnp.concatenate((center, extent), axis=-1)
    values = jnp.where(visible[..., None], values, 0.0)
    return jnp.concatenate((values, visible[..., None].astype(values.dtype)), -1), visible


def feature_dim(cfg: ObservationConfig) -> int:
    """Return the flat policy observation dimension, 20 + 9 * history."""
    return 20 + 9 * cfg.history


def clip_bbox(bbox: jnp.ndarray) -> jnp.ndarray:
    """Clip normalized center/extent boxes by their edges, as the pixel interface does.

    Clipping centers and extents separately can leave noisy corners outside the
    image. Both simulated tracker packets and deployment use this edge-clipped
    representation; empty intersections and invalid packets are all zero.
    """
    bbox = jnp.asarray(bbox)
    lower = jnp.clip(bbox[..., :2] - bbox[..., 2:4] / 2, -1.0, 1.0)
    upper = jnp.clip(bbox[..., :2] + bbox[..., 2:4] / 2, -1.0, 1.0)
    extent = jnp.maximum(upper - lower, 0.0)
    valid = (
        (bbox[..., 4] > 0.5)
        & jnp.all(bbox[..., 2:4] > 0.0, axis=-1)
        & jnp.all(extent > 0.0, axis=-1)
    )
    values = jnp.concatenate(((lower + upper) / 2, extent), axis=-1)
    values = jnp.where(valid[..., None], values, 0.0)
    return jnp.concatenate((values, valid[..., None].astype(bbox.dtype)), axis=-1)


def actor_features(
    bbox_history: jnp.ndarray,
    rotation_matrix: jnp.ndarray,
    body_rates: jnp.ndarray,
    action_history: jnp.ndarray,
    cfg: ObservationConfig,
) -> jnp.ndarray:
    """Build deployable features, with history ordered oldest to newest.

    Flat field order: R9, body gyro3 divided by 5 rad/s, bbox history (5H), issued-command
    history (4H), current body bearing3, log optical bbox diagonal1, causal OLS
    log-scale slope1 (s^-1), latest bbox center derivative2 (s^-1), latest
    log-scale derivative1 (s^-1). Commands retain the caller's policy scale.

    The scale uses width/fx and height/fy and contains no physical target size
    or metric range. Its slope fits valid samples at regular policy times
    ``dt`` over the received history, including held measurements. Thus this
    is a causal feature of the mailbox contents; it does not use hidden
    capture timestamps or delivery latency. At least three valid samples
    are required for OLS. Latest differences require two valid frames.

    The caller owns noise, dropout, delivery and queue storage. Invalid boxes
    are zeroed, and empty/invalid current measurements yield zero derived
    features without any fallback to simulated geometry.
    """
    boxes = jnp.asarray(bbox_history)
    actions = jnp.asarray(action_history)
    rotation_matrix, body_rates = map(jnp.asarray, (rotation_matrix, body_rates))
    if boxes.shape[-2:] != (cfg.history, 5):
        raise ValueError("bbox_history must have trailing shape (history, 5)")
    if actions.shape[-2:] != (cfg.history, 4):
        raise ValueError("action_history must have trailing shape (history, 4)")
    if rotation_matrix.shape[-2:] != (3, 3) or body_rates.shape[-1:] != (3,):
        raise ValueError("attitude must end in (3, 3) and body_rates in (3,)")

    valid = (boxes[..., 4] > 0.5) & jnp.all(boxes[..., 2:4] > 0.0, axis=-1)
    boxes = jnp.where(valid[..., None], boxes, 0.0)
    k = jnp.asarray(intrinsics(cfg))
    optical_extent = boxes[..., 2:4] * jnp.array(
        (cfg.image_width / (2 * k[0, 0]), cfg.image_height / (2 * k[1, 1]))
    )
    # Positive epsilon under the square root avoids NaN gradients at an
    # invalid or empty box, including when that branch is later masked.
    diagonal = jnp.sqrt(jnp.sum(optical_extent**2, axis=-1) + 1e-12)
    log_scale = jnp.where(valid, jnp.log(diagonal), 0.0)
    current_valid = valid[..., -1]
    center = boxes[..., -1, :2]
    pixel_x = (center[..., 0] + 1) * cfg.image_width / 2
    pixel_y = (center[..., 1] + 1) * cfg.image_height / 2
    ray = jnp.stack(
        ((pixel_x - k[0, 2]) / k[0, 0], (pixel_y - k[1, 2]) / k[1, 1], jnp.ones_like(pixel_x)),
        axis=-1,
    )
    ray = jnp.einsum("ji,...j->...i", jnp.asarray(rotation_body_to_optical(cfg)), ray)
    ray = ray / jnp.sqrt(jnp.sum(ray**2, axis=-1, keepdims=True))
    ray = jnp.where(current_valid[..., None], ray, 0.0)

    times = (jnp.arange(cfg.history, dtype=boxes.dtype) - cfg.history + 1) * cfg.dt
    count = jnp.sum(valid, axis=-1)
    divisor = jnp.maximum(count, 1)
    mean_time = jnp.sum(times * valid, axis=-1) / divisor
    mean_scale = jnp.sum(log_scale, axis=-1) / divisor
    centered_time = (times - mean_time[..., None]) * valid
    numerator = jnp.sum(centered_time * (log_scale - mean_scale[..., None]), axis=-1)
    denominator = jnp.maximum(jnp.sum(centered_time**2, axis=-1), 1e-8)
    slope = jnp.where((count >= 3) & current_valid, numerator / denominator, 0.0)
    if cfg.history >= 2:
        pair_valid = current_valid & valid[..., -2]
        center_rate = (boxes[..., -1, :2] - boxes[..., -2, :2]) / cfg.dt
        scale_rate = (log_scale[..., -1] - log_scale[..., -2]) / cfg.dt
        center_rate = jnp.where(pair_valid[..., None], center_rate, 0.0)
        scale_rate = jnp.where(pair_valid, scale_rate, 0.0)
    else:
        center_rate = jnp.zeros_like(center)
        scale_rate = jnp.zeros_like(log_scale[..., -1])

    batch_shape = boxes.shape[:-2]
    return jnp.concatenate(
        (
            rotation_matrix.reshape(batch_shape + (9,)),
            body_rates / GYRO_SCALE_RAD_S,
            boxes.reshape(batch_shape + (5 * cfg.history,)),
            actions.reshape(batch_shape + (4 * cfg.history,)),
            ray,
            log_scale[..., -1, None],
            slope[..., None],
            center_rate,
            scale_rate[..., None],
        ),
        axis=-1,
    )
