# ruff: noqa: E402
# Configure scientific backends and the source checkout path before importing them.
"""Reproduce the November upstream controllers, MuJoCo 4 and synthetic splat updates.

Run from the repository root in the research Docker environment::

    MUJOCO_GL=egl python usage_assets/upstream_features_20261124.py --device gpu
    JAX_PLATFORMS=cpu python usage_assets/upstream_features_20261124.py --device cpu

Outputs are separated by device and section. Existing JSON records require --overwrite.
The synthetic scene uses seed 20261124. No policy training or asset download is needed.
These are software experiments; they do not validate physical drones.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

os.environ["SCIPY_ARRAY_API"] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import casadi as cs
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from crazyflow.control import Control, parametrize
from crazyflow.control.core import load_fn_params
from crazyflow.control.mellinger import (
    attitude2force_torque,
    body_rate2force_torque,
    force_torque2rotor_vel,
    state2attitude,
)
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.drones import Drone
from crazyflow.dynamics.utils import rotation
from crazyflow.sim import Dynamics, Sim
from crazyflow.sim.sim import rotor_vel_limits
from crazyflow.sim.visualize import change_material

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

SEED = 20261124


def metadata(device: str) -> dict[str, Any]:
    """Record runtime versions, selected devices, seed and the exact source revision."""
    versions = {}
    for name in (
        "crazyflow",
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
        "mujoco",
        "mujoco-mjx",
        "casadi",
        "splax",
        "warp-lang",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "versions": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "device_requested": device,
        "devices": [str(d) for d in jax.devices(device)],
        "seed": SEED,
        "mujoco_gl": os.environ["MUJOCO_GL"],
        "command": " ".join(sys.argv),
        "upstream_commit": subprocess.check_output(
            ["git", "rev-parse", "upstream/main"], cwd=ROOT, text=True
        ).strip(),
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }


def control_experiment(device: str, output: Path) -> dict[str, Any]:
    """Check motor allocation, zero-thrust integral reset and complete control chains."""
    drone = Drone.cf21B_500
    params = load_fn_params(force_torque2rotor_vel, drone)
    minimum, maximum = float(params["thrust_min"]), float(params["thrust_max"])
    raw = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [minimum / 2] * 4,
            [minimum * 1.01] * 4,
            [0.10, 0.15, 0.25, 0.30],
            [-0.10, 0.05, 0.30, 0.45],
        ]
    )
    scale = np.array([params["L"], params["L"], params["thrust2torque"]])
    force = raw.sum(axis=-1, keepdims=True)
    torque = (raw @ params["mixing_matrix"].T) * scale
    rpm = np.asarray(force_torque2rotor_vel(force, torque, **params))
    expected = np.where(raw < minimum, 0, raw)
    expected = np.maximum(
        expected - np.maximum(expected.max(axis=-1, keepdims=True) - maximum, 0), 0
    )
    np.testing.assert_allclose(
        rpm, motor_force2rotor_vel(expected, params["rpm2thrust"]), atol=1e-8
    )
    np.testing.assert_allclose(expected[3], [0, 0.05, 0.15, 0.20], atol=1e-8)
    quat, zeros = np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3)
    command = np.zeros(16)
    command[12], command[:3] = 1, [0.1, -0.1, 0.1]
    state_fn = parametrize(state2attitude, drone)
    _, active = state_fn(zeros, quat, zeros, command, pos_err_i=np.array([0.1, 0.2, 0.3]))
    _, stopped = state_fn(
        zeros, R.from_euler("x", np.pi).as_quat(), zeros, command, pos_err_i=active
    )
    assert np.any(active != 0)
    np.testing.assert_array_equal(stopped, np.zeros(3))
    integrals = {"position": {"before": active.tolist(), "after": stopped.tolist()}}
    for fn in (attitude2force_torque, body_rate2force_torque):
        controller = parametrize(fn, drone)
        cmd = np.array([0.15, -0.1, 0.2, 0.3])
        _, _, active = controller(quat, zeros, cmd, rot_err_i=np.array([0.1, 0.2, 0.3]))
        cmd[3] = 0
        f, t, stopped = controller(quat, zeros, cmd, rot_err_i=active)
        np.testing.assert_allclose(f, 0, atol=1e-12)
        np.testing.assert_allclose(t, 0, atol=1e-12)
        np.testing.assert_array_equal(stopped, np.zeros(3))
        integrals[fn.__name__] = {"before": active.tolist(), "after": stopped.tolist()}
    zcmd = np.zeros(16)
    zcmd[12] = 1
    baseline, _ = state_fn(zeros, quat, zeros, zcmd)
    zcmd[5] = 0.2
    changed, _ = state_fn(zeros, quat, zeros, zcmd)
    sp = load_fn_params(state2attitude, drone)
    delta = sp["kd"][2] * zcmd[5] * sp["mass_thrust"] * 4 * sp["thrust_max"] / sp["pwm_max"]
    np.testing.assert_allclose(changed[3] - baseline[3], delta, rtol=1e-10)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        gimbal_quat = R.from_euler("y", np.pi / 2).as_quat()
        gimbal_cmd = command.copy()
        gimbal_cmd[9:13] = gimbal_quat
        state_fn(zeros, gimbal_quat, zeros, gimbal_cmd)
        parametrize(body_rate2force_torque, drone)(
            gimbal_quat, zeros, np.array([0.1, -0.1, 0.2, 0.3])
        )
    assert not [w for w in captured if "Gimbal lock" in str(w.message)]
    pipelines = {}
    for mode in Control:
        sim = Sim(control=mode, device=device)
        try:
            lower, upper = rotor_vel_limits(sim.dynamics, sim.drone)
            cmd = np.zeros((1, 1, 16 if mode == Control.state else 4), np.float32)
            if mode == Control.state:
                cmd[..., :3], cmd[..., 12] = [10, -10, 20], 1
            elif mode in (Control.attitude, Control.body_rate):
                cmd[..., :3], cmd[..., 3] = [10, -10, 20], 100
            elif mode == Control.force_torque:
                cmd[...] = [2.0, 0.1, -0.1, 0.1]
            else:
                cmd[...] = [-100, 0, upper / 2, upper * 2]
                assert sim.control_freq == sim.freq
                assert lower == 0
            q0 = np.asarray(sim.data.states.quat)
            w0 = np.asarray(sim.data.states.ang_vel)
            ft = cmd
            if mode == Control.state:
                ft, _ = state_fn(
                    np.asarray(sim.data.states.pos), q0, np.asarray(sim.data.states.vel), cmd
                )
            if mode in (Control.state, Control.attitude):
                f, t, _ = parametrize(attitude2force_torque, drone)(q0, w0, ft)
            elif mode == Control.body_rate:
                f, t, _ = parametrize(body_rate2force_torque, drone)(
                    q0, w0, cmd, prev_cmd=np.zeros_like(cmd)
                )
            elif mode == Control.force_torque:
                f, t = cmd[..., :1], cmd[..., 1:]
            if mode == Control.rotor_vel:
                pure_rpm = cmd
            else:
                pure_rpm = parametrize(force_torque2rotor_vel, drone)(f, t)
            expected_chain = np.clip(pure_rpm, lower, upper)
            getattr(sim, f"{mode.value}_control")(cmd)
            sim.step()
            measured = np.asarray(sim.data.controls.rotor_vel)
            assert np.isfinite(measured).all()
            np.testing.assert_allclose(measured, expected_chain, atol=0.02, rtol=1e-5)
            assert measured.min() >= lower - 1e-4 and measured.max() <= upper + 1e-2
            if mode == Control.rotor_vel:
                np.testing.assert_allclose(
                    measured, np.clip(cmd, lower, upper), rtol=1e-6, atol=1e-3
                )
            pipelines[mode.value] = {
                "frequency_hz": sim.control_freq,
                "simulation_frequency_hz": sim.freq,
                "pipeline": list(sim.step_pipeline),
                "rpm_limits": [lower, upper],
                "rpm_command": measured.tolist(),
                "pure_chain_matches": True,
            }
        finally:
            sim.close()
    np.savez_compressed(
        output / "control.npz",
        requested_motor_force_N=raw,
        expected_motor_force_N=expected,
        measured_rpm=rpm,
    )
    return {
        "requested_motor_force_N": raw.tolist(),
        "redistributed_motor_force_N": expected.tolist(),
        "measured_rpm": rpm.tolist(),
        "integral_reset": integrals,
        "zero_force_rpm_note": "The fitted thrust curve maps zero force to its positive RPM root; "
        "the simulator independently permits RPM=0.",
        "kd_z": float(sp["kd"][2]),
        "z_velocity_thrust_delta_N": float(changed[3] - baseline[3]),
        "pipelines": pipelines,
        "gimbal_lock_warnings": 0,
    }


def rotation_experiment(_device: str, output: Path) -> dict[str, Any]:
    """Check symbolic angles in radians/degrees and unique CasADi function names."""
    rng = np.random.default_rng(SEED)
    angles = rng.uniform([-1.2, -1.1, -2.5], [1.2, 1.1, 2.5], size=(64, 3))
    quats = R.from_euler("xyz", angles).as_quat()
    rad = np.stack([np.asarray(rotation.cs_quat2euler(q)).reshape(3) for q in quats])
    deg = np.stack([np.asarray(rotation.cs_quat2euler(q, degrees=True)).reshape(3) for q in quats])
    np.testing.assert_allclose(rad, angles, atol=1e-10)
    np.testing.assert_allclose(deg, np.rad2deg(angles), atol=1e-8)
    functions = (
        rotation.cs_ang_vel2rpy_rates,
        rotation.cs_rpy_rates2ang_vel,
        rotation.cs_ang_vel_deriv2rpy_rates_deriv,
        rotation.cs_rpy_rates_deriv2ang_vel_deriv,
    )
    names = [fn.name() for fn in functions]
    assert names == [
        "cs_ang_vel2rpy_rates",
        "cs_rpy_rates2ang_vel",
        "cs_ang_vel_deriv2rpy_rates_deriv",
        "cs_rpy_rates_deriv2ang_vel_deriv",
    ]
    symbolic = cs.MX.sym("quat", 4)
    fn = cs.Function(
        "november_degrees", [symbolic], [rotation.cs_quat2euler(symbolic, degrees=True)]
    )
    np.testing.assert_allclose(np.asarray(fn(quats[0])).reshape(3), deg[0], atol=1e-8)
    np.savez_compressed(
        output / "rotation.npz", angles_rad=angles, measured_rad=rad, measured_deg=deg
    )
    return {
        "samples": len(quats),
        "max_radian_error": float(np.max(np.abs(rad - angles))),
        "max_degree_error": float(np.max(np.abs(deg - np.rad2deg(angles)))),
        "casadi_function_names": names,
        "symbolic_degrees_function": fn.name(),
    }


def contacts_experiment(device: str, output: Path) -> dict[str, Any]:
    """Check MuJoCo 4 names, per-drone materials and filtered contacts in three worlds."""
    sim = Sim(n_worlds=3, n_drones=3, device=device)
    try:
        assert int(mujoco.__version__.split(".")[0]) >= 4
        positions = np.array(
            [
                [[0, 0, 1], [0, 0, 1], [2, 0, 1]],
                [[0, 0, 1], [1, 0, 1], [2, 0, -0.05]],
                [[0, 0, 1], [1, 0, 1], [2, 0, 1]],
            ],
            dtype=np.float32,
        )
        sim.data = sim.data.replace(
            states=sim.data.states.replace(pos=jnp.asarray(positions, device=sim.device)),
            core=sim.data.core.replace(mjx_synced=False),
        )
        names = [f"{d}:drone" for d in range(3)]
        assert all(sim.mj_model.body(name).id > 0 for name in names)
        cameras = [f"{d}:{name}" for d in range(3) for name in ("fpv_cam", "track_cam")]
        assert all(sim.mj_model.camera(name).id >= 0 for name in cameras)
        material_ids = [sim.mj_model.mat(f"{d}:led_top").id for d in range(3)]
        middle_before = sim.mj_model.mat_rgba[material_ids[1]].copy()
        change_material(sim, "led_top", np.array([0, 2]), rgba=np.array([1.0, 0.2, 0.1, 1.0]))
        np.testing.assert_allclose(
            sim.mj_model.mat_rgba[np.asarray(material_ids)[[0, 2]]],
            np.array([[1.0, 0.2, 0.1, 1.0]] * 2),
            atol=1e-7,
        )
        np.testing.assert_array_equal(sim.mj_model.mat_rgba[material_ids[1]], middle_before)
        scenarios = [
            ("default", (), [[1, 1, 0], [0, 0, 1], [0, 0, 0]]),
            ("ignore_pair", (("0:drone", "1:drone"),), [[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
            ("ignore_world", ("world",), [[1, 1, 0], [0, 0, 0], [0, 0, 0]]),
            ("ignore_drone", ("0:drone",), [[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
        ]
        flags = {}
        for label, ignored, expected in scenarios:
            sim.build_contact_fn(ignore=ignored)
            actual = np.asarray(sim.contacts())
            np.testing.assert_array_equal(actual, np.asarray(expected, bool))
            flags[label] = actual.tolist()
        b1, b2, active = (np.asarray(a) for a in sim.contact_bodies())
        pairs = [
            [
                [
                    mujoco.mj_id2name(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, int(b1[w, k])),
                    mujoco.mj_id2name(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, int(b2[w, k])),
                ]
                for k in np.flatnonzero(active[w])
            ]
            for w in range(3)
        ]
        assert {frozenset(p) for p in pairs[0]} == {frozenset(("0:drone", "1:drone"))}
        assert {frozenset(p) for p in pairs[1]} == {frozenset(("world", "2:drone"))}
        assert pairs[2] == []
        np.savez_compressed(
            output / "contacts.npz", positions_m=positions, touching=active, body1=b1, body2=b2
        )
        return {
            "body_names": names,
            "camera_names": cameras,
            "contact_flags": flags,
            "active_body_pairs": pairs,
            "mjx_implementation": type(sim.mjx_data._impl).__module__,
        }
    finally:
        sim.close()


def props_experiment(device: str, output: Path) -> dict[str, Any]:
    """Render actual EGL frames and measure RPM-dependent propeller rotations."""
    import imageio.v3 as iio
    import matplotlib.pyplot as plt

    results, pictures, video = {}, [], {}
    for dynamics in Dynamics:
        sim = Sim(dynamics=dynamics, device=device)
        try:
            ids = [sim.mj_model.body(f"0:prop{i}").id for i in range(4)]
            sim.data = sim.data.replace(
                states=sim.data.states.replace(pos=sim.data.states.pos.at[..., 2].set(0.15)),
                core=sim.data.core.replace(mjx_synced=False),
            )
            image_before = sim.render(
                mode="rgb_array",
                width=480,
                height=320,
                camera=-1,
                cam_config={
                    "distance": 0.28,
                    "lookat": np.array([0.0, 0.0, 0.15]),
                    "azimuth": 35.0,
                    "elevation": -55.0,
                },
            )
            before = sim.mj_data.xquat[ids].copy()
            sim.data = sim.data.replace(
                states=sim.data.states.replace(rotor_vel=sim.data.states.rotor_vel * 0 + 7.5),
                core=sim.data.core.replace(steps=sim.data.core.steps + sim.freq, mjx_synced=False),
            )
            image_after = sim.render(mode="rgb_array")
            actual = (
                R.from_quat(before, scalar_first=True).inv()
                * R.from_quat(sim.mj_data.xquat[ids], scalar_first=True)
            ).as_rotvec()
            expected = np.zeros((4, 3))
            if dynamics == Dynamics.first_principles:
                expected[:, 2] = -np.asarray(sim.data.params.mixing_matrix[2]) * np.pi / 4
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            pictures.extend([image_before, image_after])
            results[dynamics.value] = {
                "rotation_rad": actual.tolist(),
                "expected_rad": expected.tolist(),
            }
            if dynamics == Dynamics.first_principles:
                frames = [image_after]
                for frame in range(1, 24):
                    sim.data = sim.data.replace(
                        core=sim.data.core.replace(
                            steps=sim.data.core.steps * 0 + sim.freq + round(frame * sim.freq / 12),
                            mjx_synced=False,
                        )
                    )
                    frames.append(sim.render(mode="rgb_array"))
                iio.imwrite(output / "props.mp4", np.asarray(frames), fps=12, codec="libx264")
                decoded = list(iio.imiter(output / "props.mp4"))
                assert len(decoded) == 24
                assert all(frame.shape == (320, 480, 3) for frame in decoded)
                assert all(np.isfinite(frame).all() for frame in decoded)
                changed_frames = sum(
                    not np.array_equal(a, b) for a, b in zip(decoded[:-1], decoded[1:], strict=True)
                )
                assert changed_frames > 0, "Encoded propeller animation is static"
                video = {
                    "path": "props.mp4",
                    "decoded_frames": len(decoded),
                    "fps": 12,
                    "frame_shape": list(decoded[0].shape),
                    "nonstatic_frame_transitions": changed_frames,
                    "sha256": hashlib.sha256((output / "props.mp4").read_bytes()).hexdigest(),
                }
                sim.reset()
                previous = sim.mj_model.body_quat[ids].copy()
                sim.render(mode="rgb_array")
                np.testing.assert_allclose(sim.mj_model.body_quat[ids], previous, atol=1e-8)
        finally:
            sim.close()
    fig, axes = plt.subplots(len(Dynamics), 2, figsize=(10, 3 * len(Dynamics)))
    for row, dynamics in enumerate(Dynamics):
        for col in range(2):
            axes[row, col].imshow(pictures[2 * row + col])
            axes[row, col].set_title(f"{dynamics.value}, t={col}s")
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output / "props.png", dpi=140)
    plt.close(fig)
    return {
        "rpm": 7.5,
        "elapsed_s": 1.0,
        "method": "Explicit RPM/time changes; visual API check, not controlled flight",
        "dynamics": results,
        "video": video,
    }


def write_synthetic_splats(path: Path) -> None:
    """Create an asymmetric colored Gaussian scene with the upstream PLY writer."""
    import splax

    rng, n = np.random.default_rng(SEED), 192
    means = rng.uniform([-1.2, -1.2, -0.5], [1.2, 1.2, 1.5], size=(n, 3)).astype(np.float32)
    scales = np.full((n, 3), np.log(0.09), np.float32)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1))
    colors = np.asarray(splax.io.rgb_to_sh(rng.uniform(0.15, 0.85, (n, 3)).astype(np.float32)))[
        :, None
    ]
    splax.io.write_ply(path, means, scales, quats, colors, np.full(n, 2.0, np.float32))


def pose_gradient_measurements(
    sim: Sim, render: Callable, weights: Array, epsilons: tuple[float, ...]
) -> dict[str, Any]:
    """Measure autodiff and a complete finite-difference step-size sweep."""
    base = sim.data

    def objective(offset: Array) -> Array:
        quat = R.from_euler("z", offset[1]).as_quat()
        states = base.states.replace(
            pos=base.states.pos.at[0, 0, 2].add(offset[0]), quat=base.states.quat.at[0, 0].set(quat)
        )
        data = base.replace(states=states, core=base.core.replace(mjx_synced=False))
        return jnp.mean(render(data) * weights)

    point = jnp.asarray([0.013, 0.007], device=sim.device)
    evaluate = jax.jit(objective)
    gradient = np.asarray(jax.jit(jax.grad(objective))(point))
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 1e-6
    measurements = []
    for epsilon in epsilons:
        delta = jnp.eye(2, device=sim.device) * epsilon
        finite_diff = np.array(
            [
                (float(evaluate(point + d)) - float(evaluate(point - d))) / (2 * epsilon)
                for d in delta
            ]
        )
        assert np.isfinite(finite_diff).all()
        error = np.linalg.norm(gradient - finite_diff) / max(np.linalg.norm(finite_diff), 1e-7)
        measurements.append(
            {
                "epsilon": epsilon,
                "gradient": finite_diff.tolist(),
                "relative_l2_error": float(error),
            }
        )
    return {
        "parameters": ["z_m", "yaw_rad"],
        "point": np.asarray(point).tolist(),
        "objective": float(evaluate(point)),
        "autodiff_gradient": gradient.tolist(),
        "finite_differences": measurements,
    }


def smooth_gradient_experiment(device: str, output: Path) -> dict[str, Any]:
    """Check gradients on four isolated, ordered Gaussians with a fine-step convergence sweep."""
    import splax

    from crazyflow.sim.sensors.splat import build_render_splat_fn
    from crazyflow.sim.splat import attach_splats

    sim = Sim(device=device)
    try:
        camera_id = sim.mj_model.camera("0:fpv_cam").id
        rot = R.from_quat(sim.mj_model.cam_quat[camera_id], scalar_first=True).as_matrix()
        camera_points = np.array([[-0.4, 0.2, 2], [0.5, -0.3, 3], [0.2, 0.5, 4], [-0.5, -0.4, 5]])
        means = (
            sim.mj_model.cam_pos[camera_id] + (rot @ (camera_points * np.array([1, -1, -1])).T).T
        ).astype(np.float32)
        scales = np.log(np.array([[0.18] * 3, [0.22] * 3, [0.28] * 3, [0.3] * 3], np.float32))
        quats = np.tile(np.array([1, 0, 0, 0], np.float32), (4, 1))
        rgb = np.array(
            [[0.8, 0.2, 0.3], [0.2, 0.8, 0.4], [0.4, 0.2, 0.8], [0.7, 0.6, 0.2]], np.float32
        )
        colors = np.asarray(splax.io.rgb_to_sh(rgb))[:, None]
        path = output / "smooth_gradient_scene.ply"
        splax.io.write_ply(path, means, scales, quats, colors, np.full(4, 2.0, np.float32))
        attach_splats(sim, scene=path)
        render = build_render_splat_fn(
            sim, drones=0, resolution=(48, 32), background=(0.0, 0.0, 0.0)
        )
        weights = (
            jnp.linspace(0.2, 1.0, 48, device=sim.device)[None, :]
            * jnp.linspace(0.2, 1.0, 32, device=sim.device)[:, None]
        )[None, None, :, :, None]
        result = pose_gradient_measurements(
            sim, render, weights, (0.0002, 0.0001, 0.00005, 0.00002)
        )
        assert max(x["relative_l2_error"] for x in result["finite_differences"]) < 0.02, result
        result["scene"] = "Four isolated Gaussians, distinct depths, black background"
        result["asserted_relative_l2_tolerance"] = 0.02
        result["scene_gaussians"] = 4
        np.savez_compressed(
            output / "splat_gradient.npz",
            means=means,
            autodiff_gradient=result["autodiff_gradient"],
            finite_difference_gradients=[x["gradient"] for x in result["finite_differences"]],
            epsilons=[x["epsilon"] for x in result["finite_differences"]],
        )
        return result
    finally:
        sim.close()


def splats_experiment(device: str, output: Path) -> dict[str, Any]:
    """Render synthetic RGB-D and compare pure-renderer pose gradients to finite differences."""
    import matplotlib.pyplot as plt

    from crazyflow.sim.sensors.splat import (
        build_render_splat_fn,
        build_render_splat_rgbd_fn,
        render_splat_rgb,
        render_splat_rgbd,
    )
    from crazyflow.sim.splat import attach_splats

    path = output / "synthetic_scene.ply"
    write_synthetic_splats(path)
    sim = Sim(n_worlds=1, n_drones=2, device=device)
    try:
        sim.data = sim.data.replace(
            states=sim.data.states.replace(pos=sim.data.states.pos.at[0, 1, 1].set(0.3)),
            core=sim.data.core.replace(mjx_synced=False),
        )
        attach_splats(sim, scene=path, drone=path)
        resolution = (48, 32)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            rgb = np.asarray(render_splat_rgb(sim, resolution=resolution))
            rgbd = np.asarray(render_splat_rgbd(sim, resolution=resolution, max_range=5.0))
            rgb_fn = build_render_splat_fn(sim, resolution=resolution)
            rgbd_fn = build_render_splat_rgbd_fn(sim, resolution=resolution, max_range=5.0)
        cpu_warnings = [str(w.message) for w in captured if "on the CPU" in str(w.message)]
        assert len(cpu_warnings) == (4 if device == "cpu" else 0)
        assert rgb.shape == (1, 2, 32, 48, 3) and rgbd.shape == (1, 2, 32, 48, 4)
        assert np.isfinite(rgbd).all() and rgb.min() >= 0 and rgb.max() <= 1
        assert rgbd[..., 3].min() < 5 and rgbd[..., 3].max() <= 5
        np.testing.assert_allclose(rgb, rgbd[..., :3], atol=1e-5)
        np.testing.assert_allclose(rgb, np.asarray(rgb_fn(sim.data)), atol=1e-5)
        np.testing.assert_allclose(rgbd, np.asarray(rgbd_fn(sim.data)), atol=1e-5)
        one = np.asarray(render_splat_rgb(sim, drones=1, resolution=resolution))
        np.testing.assert_allclose(one[:, 0], rgb[:, 1], atol=1e-5)
        track = np.asarray(render_splat_rgb(sim, resolution=resolution, camera="track_cam"))
        assert not np.allclose(track, rgb)
        excluded = np.asarray(render_splat_rgb(sim, resolution=resolution, exclude_self=True))
        assert not np.allclose(excluded, rgb)
        empty = np.asarray(
            render_splat_rgbd(sim, resolution=resolution, max_range=5.0, alpha_threshold=1.1)
        )
        np.testing.assert_array_equal(empty[..., 3], 5.0)
        render = build_render_splat_fn(sim, drones=0, resolution=resolution, exclude_self=True)
        weights = jnp.linspace(0.2, 1.0, resolution[0], device=sim.device)[
            None, None, None, :, None
        ]
        dense_gradient = pose_gradient_measurements(
            sim, render, weights, (0.01, 0.001, 0.0005, 0.0001)
        )
        dense_gradient["interpretation"] = (
            "Dense-scene finite differences vary across step sizes. This is a diagnostic; "
            "no exact dense-scene pose-gradient agreement is asserted."
        )
        smooth_gradient = smooth_gradient_experiment(device, output)
        np.savez_compressed(
            output / "splats.npz",
            rgb=rgb,
            rgbd=rgbd,
            track=track,
            excluded=excluded,
            dense_autodiff_gradient=dense_gradient["autodiff_gradient"],
            dense_finite_difference_gradients=[
                x["gradient"] for x in dense_gradient["finite_differences"]
            ],
        )
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for i in range(2):
            axes[i, 0].imshow(rgb[0, i])
            axes[i, 0].set_title(f"Drone {i}: synthetic FPV RGB")
            image = axes[i, 1].imshow(rgbd[0, i, ..., 3], vmin=0, vmax=5, cmap="viridis")
            axes[i, 1].set_title("Optical-axis depth (m)")
            axes[i, 2].imshow(track[0, i])
            axes[i, 2].set_title("track_cam")
            for ax in axes[i]:
                ax.axis("off")
        fig.colorbar(image, ax=axes[:, 1], shrink=0.7)
        fig.savefig(output / "splats.png", dpi=150)
        plt.close(fig)
        return {
            "synthetic_gaussians_per_asset": 192,
            "rgb_shape": list(rgb.shape),
            "rgbd_shape": list(rgbd.shape),
            "depth_min_m": float(rgbd[..., 3].min()),
            "depth_max_m": float(rgbd[..., 3].max()),
            "max_range_m": 5.0,
            "cpu_warning_count": len(cpu_warnings),
            "jit_rgb_match": True,
            "jit_rgbd_match": True,
            "drone_selection_match": True,
            "camera_api": "camera='track_cam' resolves to i:track_cam",
            "exclude_self_changes_image": True,
            "alpha_above_one_reports_max_range": True,
            "smooth_pose_gradient": smooth_gradient,
            "dense_pose_gradient_diagnostic": dense_gradient,
            "scope": "Synthetic fixed-seed rasterization; no real-world visual fidelity claim.",
        }
    finally:
        sim.close()


def main() -> None:
    """Run requested sections and save individually attributable measurements."""
    experiments: dict[str, Callable[[str, Path], dict[str, Any]]] = {
        "control": control_experiment,
        "rotation": rotation_experiment,
        "contacts": contacts_experiment,
        "props": props_experiment,
        "splats": splats_experiment,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--section", choices=("all", *experiments), default="all")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(""))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output / args.device
    output.mkdir(parents=True, exist_ok=True)
    selected = experiments if args.section == "all" else {args.section: experiments[args.section]}
    for name, experiment in selected.items():
        target = output / f"{name}.json"
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"{target} exists; use --overwrite or a fresh --output directory")
        context, start = metadata(args.device), time.perf_counter()
        print(f"Running {name} on {args.device}", flush=True)
        result = experiment(args.device, output)
        record = {
            "metadata": context,
            "experiment": name,
            "result": result,
            "wall_seconds_including_compilation": time.perf_counter() - start,
        }
        target.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"PASS {name}: {target}", flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    main()
