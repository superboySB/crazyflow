"""Calibrated pinhole geometry, real camera pose and rendered-silhouette agreement."""

import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from crazyflow.balloon.camera import (
    camera_pose,
    camera_to_body,
    frustum_corners,
    intrinsics,
    parameters,
    rotation_body_to_optical,
)
from crazyflow.balloon.config import Config
from crazyflow.balloon.evaluate import _quaternion, make_replay_model
from crazyflow.balloon.observations import ideal_bbox


def test_editable_camera_defaults_match_direct_python_training():
    settings = json.loads((Path(__file__).resolve().parents[1] / "config/camera.json").read_text())
    assert Config(**settings) == Config()
    assert Config().hfov_deg == 120.0


@pytest.mark.parametrize("hfov", [72.0, 120.0])
def test_ov5647_horizontal_fov_is_not_a_vertical_fov(hfov: float):
    cfg = Config(hfov_deg=hfov)
    k = intrinsics(cfg)
    expected_focal = 640 / (2 * np.tan(np.deg2rad(hfov) / 2))
    np.testing.assert_allclose(k, [[expected_focal, 0, 320], [0, expected_focal, 240], [0, 0, 1]])
    p = parameters(cfg)
    assert p["hfov_deg"] == pytest.approx(hfov)
    assert p["vfov_deg"] == pytest.approx(np.rad2deg(2 * np.arctan(240 / expected_focal)))
    assert p["vfov_deg"] < hfov
    assert p["offset_body_m"] == [0.03, 0, 0]


def test_explicit_non_square_intrinsics_override_hfov_and_principal_point():
    cfg = replace(Config(), camera_intrinsics=(470, 430, 312, 246))
    np.testing.assert_array_equal(intrinsics(cfg), [[470, 0, 312], [0, 430, 246], [0, 0, 1]])
    assert parameters(cfg)["hfov_deg"] != pytest.approx(cfg.hfov_deg)


def test_forward_flu_mount_and_optical_axes_are_unambiguous():
    cfg = Config()
    r = rotation_body_to_optical(cfg)
    np.testing.assert_array_equal(r @ [1, 0, 0], [0, 0, 1])
    np.testing.assert_array_equal(r @ [0, 1, 0], [-1, 0, 0])
    np.testing.assert_array_equal(r @ [0, 0, 1], [0, -1, 0])
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1)
    turned = replace(cfg, camera_rpy=(0, 0, np.pi / 2))
    np.testing.assert_allclose(rotation_body_to_optical(turned) @ [0, 1, 0], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(camera_to_body(turned)[:, 0], [0, 1, 0], atol=1e-12)


def test_offset_and_nonzero_mount_match_actual_mujoco_camera_pose():
    cfg = replace(Config(), camera_offset=(0.08, -0.03, 0.02), camera_rpy=(0.1, -0.12, 0.2))
    position, rpy = np.array([0.2, -0.1, 1.4]), np.array([0.08, -0.11, 0.25])
    center, optical_to_world = camera_pose(position, rpy, cfg)
    model = make_replay_model(cfg, np.array([2, 0, 1.4]), 0.15)
    data = mujoco.MjData(model)
    data.mocap_pos[0], data.mocap_quat[0] = position, _quaternion(rpy)
    mujoco.mj_forward(model, data)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv")
    np.testing.assert_allclose(data.cam_xpos[camera_id], center, atol=1e-9)
    optical_axes = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
    np.testing.assert_allclose(optical_axes, optical_to_world, atol=1e-9)
    corners = frustum_corners(position, rpy, cfg, depth=1.3)
    optical = (optical_to_world.T @ (corners - center).T).T
    pixels = (intrinsics(cfg) @ optical.T).T
    pixels = pixels[:, :2] / pixels[:, 2:]
    np.testing.assert_allclose(pixels, [[0, 0], [640, 0], [640, 480], [0, 480]], atol=1e-9)


@pytest.mark.render
def test_near_contact_camera_keeps_balloon_front_surface_solid():
    cfg = Config()
    target = np.array([3.0, 0.0, 1.4])
    model = make_replay_model(cfg, target, 0.15)
    data = mujoco.MjData(model)
    # The world extent stays large while the aircraft approaches within centimetres.
    data.mocap_pos[0] = target - np.array([0.24, 0, 0])
    data.mocap_quat[0] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    assert model.vis.map.znear * model.stat.extent < 0.005
    renderer = mujoco.Renderer(model, height=cfg.image_height, width=cfg.image_width)
    try:
        renderer.update_scene(data, camera="fpv")
        image = renderer.render()
        center = image[235:245, 315:325].astype(float)
        # The old scene-scale near plane cut away the front cap, revealing sky/ground.
        assert np.all(center[..., 0] > 2 * center[..., 1])
        assert np.all(center[..., 0] > 2 * center[..., 2])
    finally:
        renderer.close()


@pytest.mark.render
@pytest.mark.parametrize("calibrated", [False, True])
def test_rendered_balloon_matches_analytic_bbox_with_calibration(calibrated: bool):
    cfg = replace(
        Config(),
        camera_offset=(0.08, -0.03, 0.02),
        camera_rpy=(0.1, -0.12, 0.2),
        camera_intrinsics=(470, 430, 312, 246) if calibrated else None,
    )
    position, rpy = np.array([0.2, -0.1, 1.4]), np.array([0.08, -0.11, 0.25])
    center, optical_to_world = camera_pose(position, rpy, cfg)
    target = center + optical_to_world @ [0.3, -0.18, 2]
    model = make_replay_model(cfg, target, 0.15)
    data = mujoco.MjData(model)
    data.mocap_pos[0], data.mocap_quat[0] = position, _quaternion(rpy)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=cfg.image_height, width=cfg.image_width)
    try:
        renderer.update_scene(data, camera="fpv")
        rgb = renderer.render().astype(float)
    finally:
        renderer.close()
    red = (rgb[..., 0] > 50) & (rgb[..., 0] > 1.5 * rgb[..., 1]) & (rgb[..., 0] > 1.5 * rgb[..., 2])
    yy, xx = np.where(red)
    assert len(xx) > 200
    rendered = np.array([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1])
    box, visible = ideal_bbox(
        jnp.asarray(position), jnp.asarray(rpy), jnp.asarray(target), jnp.array(0.15), cfg
    )
    assert bool(visible)
    cx, cy, width, height = np.asarray(box)[:4]
    expected = (
        np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]) + 1
    ) * [320, 240, 320, 240]
    np.testing.assert_allclose(rendered, expected, atol=1.5)
