"""Visible sphere silhouettes at wide-angle image edges and corners."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.balloon.camera import intrinsics
from crazyflow.balloon.config import Config
from crazyflow.balloon.observations import ideal_bbox


def _project(optical: np.ndarray, radius: float, cfg: Config) -> tuple:
    target = np.asarray(cfg.camera_offset) + np.array([optical[2], -optical[0], -optical[1]])
    return ideal_bbox(jnp.zeros(3), jnp.zeros(3), jnp.asarray(target), jnp.asarray(radius), cfg)


def _sample_visible_silhouette(optical: np.ndarray, radius: float, cfg: Config) -> np.ndarray:
    """Independent reference: sample the 3-D tangent circle and image-edge rays."""
    k = intrinsics(cfg)
    distance = np.linalg.norm(optical)
    direction = optical / distance
    first = np.cross(direction, [0, 1, 0])
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    phase = np.arange(65536) * (2 * np.pi / 65536)
    circle_center = optical * (1 - radius**2 / distance**2)
    circle_radius = radius * np.sqrt(1 - radius**2 / distance**2)
    points = circle_center + circle_radius * (
        np.cos(phase)[:, None] * first + np.sin(phase)[:, None] * second
    )
    projected = points[:, :2] / points[:, 2:]
    lower = -k[:2, 2] / np.diag(k)[:2]
    upper = (np.array([cfg.image_width, cfg.image_height]) - k[:2, 2]) / np.diag(k)[:2]
    projected = projected[np.all((projected >= lower) & (projected <= upper), axis=-1)]
    fraction = np.linspace(0, 1, 16385)
    x = lower[0] + fraction * (upper[0] - lower[0])
    y = lower[1] + fraction * (upper[1] - lower[1])
    edge_pixels = np.concatenate(
        (
            np.column_stack((x, np.full_like(x, lower[1]))),
            np.column_stack((x, np.full_like(x, upper[1]))),
            np.column_stack((np.full_like(y, lower[0]), y)),
            np.column_stack((np.full_like(y, upper[0]), y)),
        )
    )
    rays = np.column_stack((edge_pixels, np.ones(len(edge_pixels))))
    ray_hits = (np.sum(np.cross(optical, rays) ** 2, -1) <= radius**2 * np.sum(rays**2, -1)) & (
        rays @ optical > 0
    )
    visible = np.concatenate((projected, edge_pixels[ray_hits]))
    assert len(visible)
    pixels = visible * np.diag(k)[:2] + k[:2, 2]
    return np.r_[pixels.min(0), pixels.max(0)]


@pytest.mark.parametrize("hfov", [72.0, 120.0])
def test_amodal_corner_overlap_without_any_visible_sphere_is_invalid(hfov: float):
    cfg = replace(Config(), hfov_deg=hfov, camera_intrinsics=None)
    k = intrinsics(cfg)
    hx, hy = cfg.image_width / (2 * k[0, 0]), cfg.image_height / (2 * k[1, 1])
    # Each frustum plane individually intersects the sphere, but their corner
    # does not: an overlap of the amodal rectangle cannot establish visibility.
    optical = np.array(
        [2 * hx + 0.14 * np.sqrt(1 + hx**2), 2 * hy + 0.14 * np.sqrt(1 + hy**2), 2.0]
    )
    corner_ray = np.array([hx, hy, 1.0])
    closest_corner = corner_ray * (corner_ray @ optical) / (corner_ray @ corner_ray)
    assert np.linalg.norm(closest_corner - optical) > 0.15
    box, visible = _project(optical, 0.15, cfg)
    assert not bool(visible)
    np.testing.assert_array_equal(box, 0.0)


@pytest.mark.parametrize(
    "optical,radius",
    [
        ([0.0, 0.0, 0.16], 0.15),
        ([0.6, 0.04, 0.32], 0.15),
        ([7.0, 0.4, 4.0], 0.15),
        ([3.68, 2.76, 2.0], 0.15),
        ([0.25, -0.12, 3.0], 0.15),
    ],
)
def test_clipped_bounds_match_visible_three_dimensional_tangent_circle(
    optical: list, radius: float
):
    cfg = replace(Config(), hfov_deg=120.0, camera_intrinsics=None)
    optical = np.asarray(optical)
    expected = _sample_visible_silhouette(optical, radius, cfg)
    box, visible = _project(optical, radius, cfg)
    assert bool(visible)
    cx, cy, width, height = np.asarray(box)[:4]
    actual = (np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]) + 1) * [
        320,
        240,
        320,
        240,
    ]
    np.testing.assert_allclose(actual, expected, atol=0.08)


def test_partial_wide_angle_silhouette_has_finite_useful_gradients():
    cfg = replace(Config(), hfov_deg=120.0, camera_intrinsics=None)
    target = jnp.array([0.35, -0.6, -0.04])

    def project(p: jax.Array) -> jax.Array:
        return ideal_bbox(p, jnp.zeros(3), target, jnp.asarray(0.15), cfg)[0]

    actual = jax.jacrev(project)(jnp.zeros(3))
    assert np.isfinite(actual).all()
    assert np.linalg.norm(actual[:4]) > 0.1
    step = 1e-4
    numerical = np.column_stack(
        [
            (
                np.asarray(project(jnp.eye(3)[axis] * step))
                - np.asarray(project(-jnp.eye(3)[axis] * step))
            )
            / (2 * step)
            for axis in range(3)
        ]
    )
    np.testing.assert_allclose(actual, numerical, atol=0.004, rtol=0.004)


def test_corner_and_optical_plane_invalid_branches_have_finite_zero_jacobians():
    cfg = replace(Config(), hfov_deg=120.0, camera_intrinsics=None)
    targets = jnp.array([[2.03, -3.7441016, -2.8275866], [0.17, 0, 0], [-2, 0, 0]])

    def project(p: jax.Array) -> jax.Array:
        return ideal_bbox(p, jnp.zeros(3), targets, jnp.asarray(0.15), cfg)[0]

    boxes, jacobian = project(jnp.zeros(3)), jax.jacrev(project)(jnp.zeros(3))
    np.testing.assert_array_equal(boxes, 0.0)
    assert np.isfinite(jacobian).all()
    np.testing.assert_array_equal(jacobian, 0.0)
