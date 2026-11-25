"""One pinhole calibration shared by the policy sensor, MuJoCo and the viewer.

Pixel coordinates use right/down with continuous image boundaries [0,W]×[0,H].
The optical frame is right/down/forward; the drone body and unrotated camera mount
are forward/left/up. Mount RPY is in radians, Rz(yaw) Ry(pitch) Rx(roll).
"""

from __future__ import annotations

from typing import Any

import numpy as np

FLU_TO_OPTICAL = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def intrinsics(cfg: Any) -> np.ndarray:
    """Return K; explicit (fx,fy,cx,cy) overrides the centered square-pixel HFOV fit."""
    calibrated = getattr(cfg, "camera_intrinsics", None)
    if calibrated is None:
        fx = cfg.image_width / (2 * np.tan(np.deg2rad(cfg.hfov_deg) / 2))
        fy, cx, cy = fx, cfg.image_width / 2, cfg.image_height / 2
    else:
        fx, fy, cx, cy = calibrated
    if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise ValueError("Camera focal lengths must be positive and calibration finite")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """Return body-to-parent Rz Ry Rx rotation, allowing leading batch dimensions."""
    rpy = np.asarray(rpy)
    roll, pitch, yaw = np.moveaxis(rpy, -1, 0)
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    return np.stack(
        [
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ],
        axis=-1,
    ).reshape((*rpy.shape[:-1], 3, 3))


def camera_to_body(cfg: Any) -> np.ndarray:
    """Return the camera FLU mount-to-body rotation (identity for a forward mount)."""
    return rpy_matrix(np.asarray(getattr(cfg, "camera_rpy", (0.0, 0.0, 0.0))))


def rotation_body_to_optical(cfg: Any) -> np.ndarray:
    """Return R in p_optical = R @ (p_body - camera_offset)."""
    return FLU_TO_OPTICAL @ camera_to_body(cfg).T


def camera_pose(position: np.ndarray, rpy: np.ndarray, cfg: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return camera world position and optical-to-world rotation."""
    world_rotation = rpy_matrix(np.asarray(rpy))
    center = np.asarray(position) + np.einsum(
        "...ij,j->...i", world_rotation, np.asarray(cfg.camera_offset)
    )
    return center, world_rotation @ rotation_body_to_optical(cfg).T


def frustum_corners(
    position: np.ndarray, rpy: np.ndarray, cfg: Any, depth: float = 1.0
) -> np.ndarray:
    """Return TL,TR,BR,BL world corners at a positive optical depth, from the same K."""
    if depth <= 0:
        raise ValueError("Frustum depth must be positive")
    center, optical_to_world = camera_pose(position, rpy, cfg)
    pixels = np.array(
        [
            [0, 0, 1],
            [cfg.image_width, 0, 1],
            [cfg.image_width, cfg.image_height, 1],
            [0, cfg.image_height, 1],
        ]
    )
    rays = (np.linalg.inv(intrinsics(cfg)) @ pixels.T).T * depth
    return center[..., None, :] + np.einsum("...ij,kj->...ki", optical_to_world, rays)


def mujoco_camera_attributes(cfg: Any) -> dict[str, str]:
    """Return MJCF extrinsics/intrinsics, including MuJoCo's centered offset convention.

    MuJoCo's camera axes are right/up/backward. ``principalpixel`` is a signed
    offset, so it is not the usual top-left-origin principal-point coordinate.
    """
    k = intrinsics(cfg)
    optical_to_body = rotation_body_to_optical(cfg).T
    axes = np.r_[optical_to_body[:, 0], -optical_to_body[:, 1]]
    return {
        "pos": " ".join(map(str, cfg.camera_offset)),
        "xyaxes": " ".join(map(str, axes)),
        "resolution": f"{cfg.image_width} {cfg.image_height}",
        "sensorsize": "1 1",
        "focalpixel": f"{k[0, 0]} {k[1, 1]}",
        "principalpixel": f"{cfg.image_width / 2 - k[0, 2]} {cfg.image_height / 2 - k[1, 2]}",
    }


def parameters(cfg: Any) -> dict[str, Any]:
    """Serialize the effective calibration used by every projection and rendered view."""
    k = intrinsics(cfg)
    hfov = np.arctan(k[0, 2] / k[0, 0]) + np.arctan((cfg.image_width - k[0, 2]) / k[0, 0])
    vfov = np.arctan(k[1, 2] / k[1, 1]) + np.arctan((cfg.image_height - k[1, 2]) / k[1, 1])
    return {
        "model": "undistorted pinhole; rectified tracker pixels",
        "resolution": [cfg.image_width, cfg.image_height],
        "K": k.tolist(),
        "hfov_deg": float(np.rad2deg(hfov)),
        "vfov_deg": float(np.rad2deg(vfov)),
        "offset_body_m": list(cfg.camera_offset),
        "mount_rpy_rad": list(getattr(cfg, "camera_rpy", (0.0, 0.0, 0.0))),
        "rotation_body_to_optical": rotation_body_to_optical(cfg).tolist(),
    }
