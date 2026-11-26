"""Frozen all-episode evaluation with Crazyflow or native MuJoCo flight integration.

The MuJoCo backend advances a free rigid body with mj_step and uses the shared
official controller, motor/propeller wrench and sensor pipeline. Video rendering
replays whichever flight backend produced the saved trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


def _sphere_entry(start: np.ndarray, end: np.ndarray, radius: float) -> float:
    """Return the first segment/sphere intersection, including an initial overlap."""
    if np.dot(start, start) <= radius * radius:
        return 0.0
    direction = end - start
    quadratic = float(np.dot(direction, direction))
    if quadratic <= 1e-20:
        return float("inf")
    linear = 2.0 * float(np.dot(start, direction))
    constant = float(np.dot(start, start) - radius * radius)
    discriminant = linear * linear - 4.0 * quadratic * constant
    if discriminant < 0.0:
        return float("inf")
    fraction = (-linear - np.sqrt(max(discriminant, 0.0))) / (2.0 * quadratic)
    return float(fraction) if 0.0 <= fraction <= 1.0 else float("inf")


def _sphere_exit(start: np.ndarray, end: np.ndarray, radius: float) -> float:
    if np.dot(start, start) >= radius * radius:
        return 0.0
    if np.dot(end, end) < radius * radius:
        return float("inf")
    direction = end - start
    quadratic = float(np.dot(direction, direction))
    if quadratic <= 1e-20:
        return float("inf")
    linear = 2.0 * float(np.dot(start, direction))
    constant = float(np.dot(start, start) - radius * radius)
    discriminant = max(linear * linear - 4.0 * quadratic * constant, 0.0)
    fraction = (-linear + np.sqrt(discriminant)) / (2.0 * quadratic)
    return float(fraction) if 0.0 <= fraction <= 1.0 else float("inf")


def _first_tracking_loss(visible: np.ndarray, dt: float, patience: float) -> float:
    missing_since = None
    for step, present in enumerate(visible):
        if present:
            missing_since = None
        elif missing_since is None:
            missing_since = step * dt
        if missing_since is not None and (step + 1) * dt - missing_since >= patience - 1e-9:
            return float(missing_since + patience)
    return float("inf")


def _rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """Return body-to-world rotations for intrinsic xyz RPY in radians."""
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


def summarize_trajectories(
    states: np.ndarray,
    targets: np.ndarray,
    target_radii: np.ndarray,
    *,
    dt: float,
    target_velocities: np.ndarray | None = None,
    drone_radius: float = 0.086,
    collision_offset: tuple = (0.0, 0.0, 0.02),
    min_altitude: float = 0.12,
    max_range: float = 12.0,
    max_impact_speed: float = 2.5,
    visible: np.ndarray | None = None,
    visibility_dt: float | None = None,
    tracking_timeout: float = 0.5,
    objective: float | None = None,
) -> dict[str, Any]:
    """Score every episode until its first physical/sensor event without resets.

    ``states`` contains every integration substep and the initial state, with shape
    ``(time, episodes, 12)`` and state order position, RPY, velocity, RPY rates.
    Targets move linearly from ``targets`` with constant world-frame
    ``target_velocities`` in m/s; omission preserves the static task. The
    collision envelope matches the official cf21B_500 sphere, including its
    body-frame offset. Tracking loss means
    ``tracking_timeout`` consecutive seconds without a valid box. Flight range is
    measured from each episode's initial position. Every physical contact is a
    task success; impact speed is reported separately as a training preference.
    """
    states = np.asarray(states, dtype=float)
    targets = np.asarray(targets, dtype=float)
    target_radii = np.asarray(target_radii, dtype=float)
    if states.ndim != 3 or states.shape[-1] != 12 or states.shape[0] < 2:
        raise ValueError("states must have shape (substeps + 1, episodes, 12)")
    count = states.shape[1]
    if count < 1 or targets.shape != (count, 3) or target_radii.shape != (count,):
        raise ValueError("targets/radii must match a nonempty episode dimension")
    target_velocities = (
        np.zeros((count, 3), dtype=float)
        if target_velocities is None
        else np.asarray(target_velocities, dtype=float)
    )
    if target_velocities.shape != (count, 3) or not np.isfinite(target_velocities).all():
        raise ValueError(
            "target_velocities must be finite world velocities with shape (episodes,3)"
        )
    if dt <= 0 or drone_radius <= 0 or np.any(target_radii <= 0):
        raise ValueError("time step and collision radii must be positive")
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(target_radii)):
        raise ValueError("target geometry must be finite")
    if max_range <= 0 or max_impact_speed <= 0 or tracking_timeout <= 0:
        raise ValueError("range, impact speed and tracking timeout must be positive")
    if visible is not None:
        visible = np.asarray(visible, dtype=bool)
        if visible.ndim != 2 or visible.shape[0] < 1 or visible.shape[1] != count:
            raise ValueError("visible must have shape (sensor steps, episodes)")
        visibility_dt = dt if visibility_dt is None else visibility_dt
        if visibility_dt <= 0:
            raise ValueError("visibility_dt must be positive")
    horizon = (states.shape[0] - 1) * dt
    rotations = _rotation_matrix(states[..., 3:6])
    collision_displacement = np.einsum("...ij,j->...i", rotations, np.asarray(collision_offset))
    collision_centers = states[..., :3] + collision_displacement
    sample_times = np.arange(states.shape[0], dtype=float) * dt
    relative_centers = (
        collision_centers - targets[None] - sample_times[:, None, None] * target_velocities[None]
    )
    roll, pitch = states[..., 3], states[..., 4]
    roll_dot, pitch_dot, yaw_dot = np.moveaxis(states[..., 9:12], -1, 0)
    body_rates = np.stack(
        [
            roll_dot - np.sin(pitch) * yaw_dot,
            np.cos(roll) * pitch_dot + np.sin(roll) * np.cos(pitch) * yaw_dot,
            -np.sin(roll) * pitch_dot + np.cos(roll) * np.cos(pitch) * yaw_dot,
        ],
        axis=-1,
    )
    angular_velocity = np.einsum("...ij,...j->...i", rotations, body_rates)
    collision_velocities = states[..., 6:9] + np.cross(angular_velocity, collision_displacement)
    records = []
    for episode in range(count):
        trajectory = states[:, episode]
        collision_radius = float(target_radii[episode] + drone_radius)
        finite = np.all(np.isfinite(trajectory), axis=1)
        invalid = np.flatnonzero(~finite)
        nonfinite_time = float(invalid[0] * dt) if len(invalid) else float("inf")
        tracking_time = (
            _first_tracking_loss(visible[:, episode], float(visibility_dt), tracking_timeout)
            if visible is not None
            else float("inf")
        )
        # Failures win simultaneous ties. A trajectory after its first event never
        # converts a failed episode into a success.
        event_time = min(horizon, nonfinite_time, tracking_time)
        outcome = "timeout"
        if nonfinite_time <= event_time:
            outcome = "nonfinite"
        elif tracking_time <= event_time:
            outcome = "tracking_lost"
        impact_speed = None
        drone_impact_speed = None
        contact_time = None
        origin = trajectory[0, :3]
        for step in range(states.shape[0] - 1):
            start_time = step * dt
            if start_time > event_time or not (finite[step] and finite[step + 1]):
                break
            start, end = trajectory[step, :3], trajectory[step + 1, :3]
            floor_fraction = float("inf")
            if start[2] <= min_altitude:
                floor_fraction = 0.0
            elif end[2] <= min_altitude:
                floor_fraction = float((start[2] - min_altitude) / (start[2] - end[2]))
            bound_fraction = _sphere_exit(start - origin, end - origin, max_range)
            hit_fraction = _sphere_entry(
                relative_centers[step, episode],
                relative_centers[step + 1, episode],
                collision_radius,
            )
            for fraction, candidate in (
                (floor_fraction, "floor"),
                (bound_fraction, "out_of_bounds"),
                (hit_fraction, "hit"),
            ):
                candidate_time = start_time + fraction * dt
                if not np.isfinite(candidate_time) or candidate_time > horizon:
                    continue
                if candidate_time < event_time or (
                    candidate_time == event_time and outcome == "timeout"
                ):
                    event_time, outcome = candidate_time, candidate
                    impact_speed = None
                    drone_impact_speed = None
                    contact_time = None
                    if candidate == "hit":
                        velocity = (
                            collision_velocities[step, episode] * (1.0 - fraction)
                            + collision_velocities[step + 1, episode] * fraction
                        )
                        drone_impact_speed = float(np.linalg.norm(velocity))
                        impact_speed = float(np.linalg.norm(velocity - target_velocities[episode]))
                        contact_time = float(candidate_time)
                        if impact_speed > max_impact_speed:
                            outcome = "too_fast_contact"
        minimum_distance = (
            float(np.linalg.norm(relative_centers[0, episode])) if finite[0] else float("inf")
        )
        terminal_position = relative_centers[0, episode]
        for step in range(states.shape[0] - 1):
            if step * dt > event_time or not (finite[step] and finite[step + 1]):
                break
            fraction = float(np.clip((event_time - step * dt) / dt, 0.0, 1.0))
            start = relative_centers[step, episode]
            end = start + fraction * (relative_centers[step + 1, episode] - start)
            direction = end - start
            length_squared = float(np.dot(direction, direction))
            closest_fraction = (
                float(np.clip(np.dot(-start, direction) / length_squared, 0.0, 1.0))
                if length_squared > 1e-20
                else 0.0
            )
            minimum_distance = min(
                minimum_distance, float(np.linalg.norm(start + closest_fraction * direction))
            )
            terminal_position = end
        terminal_distance = float(np.linalg.norm(terminal_position))
        observed_steps = (
            max(1, int(np.ceil(event_time / float(visibility_dt)))) if visible is not None else 0
        )
        records.append(
            {
                "episode": episode,
                "outcome": outcome,
                "success": outcome in ("hit", "too_fast_contact"),
                "event_time_s": float(event_time),
                "contact_time_s": contact_time,
                "impact_speed_m_s": impact_speed,
                "drone_collision_speed_m_s": drone_impact_speed,
                "target_speed_m_s": float(np.linalg.norm(target_velocities[episode])),
                "minimum_distance_m": minimum_distance if np.isfinite(minimum_distance) else None,
                "minimum_surface_clearance_m": (
                    max(0.0, minimum_distance - collision_radius)
                    if np.isfinite(minimum_distance)
                    else None
                ),
                "terminal_distance_m": terminal_distance
                if np.isfinite(terminal_distance)
                else None,
                "visible_fraction": float(visible[:observed_steps, episode].mean())
                if visible is not None
                else None,
            }
        )
    outcome_names = (
        "hit",
        "too_fast_contact",
        "floor",
        "out_of_bounds",
        "tracking_lost",
        "nonfinite",
        "timeout",
    )
    outcomes = {
        name: sum(record["outcome"] == name for record in records) for name in outcome_names
    }
    valid_distances = [
        r["minimum_distance_m"] for r in records if r["minimum_distance_m"] is not None
    ]
    surface_distances = [
        r["minimum_surface_clearance_m"]
        for r in records
        if r["minimum_surface_clearance_m"] is not None
    ]
    successes = outcomes["hit"] + outcomes["too_fast_contact"]
    rate = successes / count
    # Wilson's interval remains informative for zero and all-success suites.
    z = 1.959963984540054
    denominator = 1 + z * z / count
    interval_center = (rate + z * z / (2 * count)) / denominator
    interval_halfwidth = (
        z * np.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    )
    result = {
        "episodes": count,
        "successes": successes,
        "success_rate": rate,
        "success_rate_wilson95": [
            max(0.0, float(interval_center - interval_halfwidth)),
            min(1.0, float(interval_center + interval_halfwidth)),
        ],
        "geometric_contact_rate": rate,
        "safe_contacts": outcomes["hit"],
        "safe_contact_rate": outcomes["hit"] / count,
        "high_speed_contact_rate": outcomes["too_fast_contact"] / count,
        "outcomes": outcomes,
        "mean_event_time_s": float(np.mean([r["event_time_s"] for r in records])),
        "mean_minimum_distance_m": float(np.mean(valid_distances)) if valid_distances else None,
        "mean_minimum_surface_clearance_m": (
            float(np.mean(surface_distances)) if surface_distances else None
        ),
        "distance_metric_episodes": len(valid_distances),
        "objective": float(objective) if objective is not None and np.isfinite(objective) else None,
        "per_episode": records,
    }
    contact = np.asarray([record["success"] for record in records], dtype=bool)
    tracking_lost = np.asarray([record["outcome"] == "tracking_lost" for record in records])

    def subset(mask: np.ndarray) -> dict:
        episodes = int(mask.sum())
        contacts = int(np.count_nonzero(contact & mask))
        lost = int(np.count_nonzero(tracking_lost & mask))
        return {
            "episodes": episodes,
            "contacts": contacts,
            "contact_rate": contacts / episodes if episodes else None,
            "tracking_lost": lost,
            "tracking_lost_rate": lost / episodes if episodes else None,
        }

    def bins(
        values: np.ndarray, edges: tuple[float, ...], eligible: np.ndarray | None = None
    ) -> dict:
        eligible = np.ones(count, dtype=bool) if eligible is None else eligible
        included = np.zeros(count, dtype=bool)
        buckets = {}
        for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            last = index == len(edges) - 2
            mask = (
                eligible
                & (values >= low - (1e-7 if index == 0 else 0.0))
                & (values <= high + 1e-7 if last else values < high)
            )
            label = f"[{low:g},{high:g}{']' if last else ')'}"
            buckets[label] = subset(mask)
            included |= mask
        buckets["outside_bins"] = subset(eligible & ~included)
        return buckets

    speeds = np.linalg.norm(target_velocities, axis=-1)
    result["stratified"] = {
        "motion": {"static": subset(speeds == 0), "moving": subset(speeds > 0)},
        "target_speed_m_s": bins(speeds, (0.0, 0.5, 1.0, 1.5)),
        "moving_target_speed_m_s": bins(speeds, (0.0, 0.5, 1.0, 1.5), speeds > 0),
        "target_diameter_m": bins(2 * target_radii, (0.03, 0.1, 0.3, 0.6)),
        "diameter_by_motion": {
            name: bins(2 * target_radii, (0.03, 0.1, 0.3, 0.6), mask)
            for name, mask in (("static", speeds == 0), ("moving", speeds > 0))
        },
        "denominator": (
            "all sampled episodes; moving groups exclude exactly stationary targets; "
            "outside_bins retains nonstandard task settings; empty rates are null"
        ),
    }
    assert sum(outcomes.values()) == count
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _versions() -> dict[str, str]:
    result = {}
    for package in ("crazyflow", "jax", "numpy", "mujoco", "imageio"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def _quaternion(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy / 2.0
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def render_replay(
    states: np.ndarray,
    target: np.ndarray,
    radius: float,
    cfg: Any,
    path: Path,
    *,
    fps: int = 25,
    stop_time: float | None = None,
    plant: str = "crazyflow",
    target_velocity: np.ndarray | None = None,
) -> dict[str, Any]:
    """Render a frozen episode and its ideal FPV box in a MuJoCo scene.

    ``mj_forward`` synchronizes the frozen flight trajectory; ``mj_step`` is never
    used. The official cf21B_500 visual mesh and the task collision sphere are
    placed at the frozen state; they do not affect the flight dynamics.
    """
    # The launcher selects EGL/OSMesa before MuJoCo is imported. Changing the
    # environment here would not change its already-loaded OpenGL backend.
    import imageio.v2 as imageio
    import jax
    import jax.numpy as jnp
    import mujoco

    from crazyflow.balloon.camera import parameters
    from crazyflow.balloon.observations import ideal_bbox

    target = np.asarray(target, dtype=float)
    target_velocity = (
        np.zeros(3) if target_velocity is None else np.asarray(target_velocity, dtype=float)
    )
    if target_velocity.shape != (3,) or not np.isfinite(target_velocity).all():
        raise ValueError("target_velocity must be a finite world velocity with shape (3,)")
    model = make_replay_model(cfg, target, radius)
    render_width, render_height = int(cfg.image_width), int(cfg.image_height)
    data = mujoco.MjData(model)
    balloon_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "balloon")
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (states[0, :3] + target) / 2
    camera.distance = max(3.0, float(np.linalg.norm(target - states[0, :3])) * 1.25)
    camera.azimuth = 135
    camera.elevation = -25
    renderer = mujoco.Renderer(model, height=render_height, width=render_width)
    from OpenGL import GL

    graphics = {
        "backend": os.environ.get("MUJOCO_GL", "glfw"),
        "vendor": GL.glGetString(GL.GL_VENDOR).decode(),
        "renderer": GL.glGetString(GL.GL_RENDERER).decode(),
        "version": GL.glGetString(GL.GL_VERSION).decode(),
    }
    project = jax.jit(
        lambda state, center: ideal_bbox(state[:3], state[3:6], center, jnp.asarray(radius), cfg)
    )
    step_dt = float(cfg.dt) / int(cfg.substeps)
    duration = (len(states) - 1) * step_dt
    if stop_time is not None:
        duration = min(duration, max(float(stop_time), 0.0))
    frame_times = np.r_[np.arange(0.0, duration, 1.0 / fps), duration]
    contact_frames = 0
    boxes = []
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=7) as writer:
            for frame_time in frame_times:
                index = min(int(frame_time / step_dt), len(states) - 1)
                fraction = float(frame_time / step_dt - index)
                state = (
                    states[index] * (1 - fraction)
                    + states[min(index + 1, len(states) - 1)] * fraction
                )
                if not np.all(np.isfinite(state)):
                    break
                data.mocap_pos[0] = state[:3]
                data.mocap_quat[0] = _quaternion(state[3:6])
                current_target = target + target_velocity * frame_time
                model.body_pos[balloon_body] = current_target
                mujoco.mj_forward(model, data)
                balloon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "balloon_surface")
                drone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "drone_collision")
                contact_frames += int(
                    any(
                        {int(c.geom1), int(c.geom2)} == {balloon_id, drone_id}
                        for c in data.contact[: data.ncon]
                    )
                )
                renderer.update_scene(data, camera=camera)
                world = renderer.render().copy()
                renderer.update_scene(data, camera="fpv")
                fpv = renderer.render().copy()
                box, present = project(jnp.asarray(state), jnp.asarray(current_target))
                box = np.asarray(box)
                boxes.append(np.r_[frame_time, box.ravel()])
                if bool(present):
                    cx, cy, width, height = box[:4]
                    x0, x1 = np.clip(
                        np.rint((cx + np.array([-width, width]) / 2 + 1) * render_width / 2),
                        0,
                        render_width - 1,
                    ).astype(int)
                    y0, y1 = np.clip(
                        np.rint((cy + np.array([-height, height]) / 2 + 1) * render_height / 2),
                        0,
                        render_height - 1,
                    ).astype(int)
                    fpv[y0 : min(y0 + 2, render_height), x0 : x1 + 1] = [40, 255, 70]
                    fpv[max(y1 - 1, 0) : y1 + 1, x0 : x1 + 1] = [40, 255, 70]
                    fpv[y0 : y1 + 1, x0 : min(x0 + 2, render_width)] = [40, 255, 70]
                    fpv[y0 : y1 + 1, max(x1 - 1, 0) : x1 + 1] = [40, 255, 70]
                writer.append_data(np.concatenate([world, fpv], axis=1))
    finally:
        renderer.close()
    np.save(path.with_suffix(".boxes.npy"), np.asarray(boxes))
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "frames": len(boxes),
        "first_event_time_s": duration,
        "sampled_mujoco_contact_frames": contact_frames,
        "role": f"MuJoCo kinematic scene rendering; recorded {plant} flight integration",
        "visual_model": "official assets/drones/cf21B_500.xml visual meshes",
        "opengl": graphics,
        "camera": parameters(cfg),
        "target_initial_center_m": target.tolist(),
        "target_velocity_m_s": target_velocity.tolist(),
        "contact_note": "Render contacts are diagnostics; success uses swept substep contact.",
    }


def make_replay_model(cfg: Any, target: np.ndarray, radius: float) -> Any:
    """Build the shared replay scene with official visuals and calibrated pinhole camera."""
    import mujoco

    from crazyflow.balloon.camera import mujoco_camera_attributes
    from crazyflow.paths import resource_path

    drone_radius = float(cfg.drone_radius)
    render_width, render_height = int(cfg.image_width), int(cfg.image_height)
    collision_offset = " ".join(map(str, cfg.collision_offset))
    xml = f"""<mujoco model="balloon_replay">
      <visual><global offwidth="{render_width}" offheight="{render_height}"/>
        <headlight ambient=".45 .45 .45" diffuse=".7 .7 .7"/>
        <map znear=".0001"/></visual>
      <asset><texture type="skybox" builtin="gradient" width="512" height="3072"
        rgb1=".3 .4 .55" rgb2=".08 .12 .2"/>
        <texture name="grid" type="2d" builtin="checker" width="256" height="256"
        rgb1=".24 .28 .32" rgb2=".34 .38 .42"/>
        <material name="ground" texture="grid" texrepeat="8 8"/></asset>
      <worldbody><light pos="1 -3 6"/><geom name="floor" type="plane" size="20 20 .1"
        material="ground"/><body name="balloon" pos="{" ".join(map(str, target))}">
        <geom name="balloon_surface" type="sphere" size="{radius}" rgba=".95 .16 .12 1"/>
        </body></worldbody><contact><pair geom1="drone_collision"
        geom2="balloon_surface"/></contact></mujoco>"""
    scene = ET.fromstring(xml)
    drone_directory = resource_path("assets/drones")
    official = ET.parse(drone_directory / "cf21B_500.xml").getroot()
    compiler = official.find("compiler")
    compiler.set("meshdir", str(drone_directory / "assets"))
    scene.insert(0, compiler)
    scene.insert(1, official.find("default"))
    scene.find("asset").extend(list(official.find("asset")))
    drone = official.find("worldbody/body[@name='drone']")
    drone.set("pos", "0 0 0")
    drone.set("mocap", "true")
    collision = drone.find("geom[@name='col_sphere']")
    collision.set("name", "drone_collision")
    collision.set("size", str(drone_radius))
    collision.set("pos", collision_offset)
    collision.set("rgba", "0 0 0 0")
    drone.remove(drone.find("geom[@name='col_box']"))
    for camera_element in list(drone.findall("camera")):
        drone.remove(camera_element)
    ET.SubElement(drone, "camera", {"name": "fpv", **mujoco_camera_attributes(cfg)})
    scene.find("worldbody").append(drone)
    return mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))


def control_telemetry(
    trajectory: dict, cfg: Any, rollout_key: Any, *, noise: bool
) -> dict[str, np.ndarray]:
    """Read recorded hardware timing, or recover the legacy fixed FIFOs.

    Sampling matches ``rollout``'s first key split. Actions and received boxes at
    index k belong to the beginning of control tick k; state/gyro/RPM index k is
    the corresponding boundary state, before executing command k.
    """
    import jax
    import jax.numpy as jnp

    from crazyflow.balloon.model import command, nominal_params

    if cfg.hardware.enabled:
        return {
            name: np.asarray(trajectory[name])
            for name in (
                "issued_commands",
                "executed_commands",
                "camera_age_ms",
                "gyro_age_ms",
                "attitude_age_ms",
                "command_delay_ms",
            )
        }
    issued = np.asarray(trajectory["actions"])
    if issued.ndim != 3 or issued.shape[-1] != 4:
        raise ValueError("actions must have shape (control steps, episodes, 4)")
    steps, episodes = issued.shape[:2]
    keys = jax.random.split(rollout_key, 3)
    sensor_delay = (
        np.asarray(jax.random.randint(keys[0], (episodes,), 0, cfg.max_delay_steps + 1))
        if noise
        else np.zeros(episodes, dtype=int)
    )
    action_delay = (
        np.asarray(jax.random.randint(keys[1], (episodes,), 0, cfg.action_delay_steps + 1))
        if noise
        else np.zeros(episodes, dtype=int)
    )
    source = np.arange(steps)[:, None] - action_delay[None]
    executed = issued[np.maximum(source, 0), np.arange(episodes)[None]]
    # Initial FIFO entries contain normalized zero actions: nominal hover commands.
    executed = np.where((source >= 0)[..., None], executed, 0.0)
    physical = nominal_params(cfg)
    return {
        "sensor_delay_steps": sensor_delay,
        "action_delay_steps": action_delay,
        "issued_commands": np.asarray(command(jnp.asarray(issued), cfg, physical)),
        "executed_commands": np.asarray(command(jnp.asarray(executed), cfg, physical)),
    }


def main() -> None:
    """Evaluate a frozen checkpoint on reproducible heldout scenario suites."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument(
        "--plant",
        choices=("crazyflow", "mujoco"),
        default="crazyflow",
        help="flight integrator: differentiable training plant or native MuJoCo mj_step",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/balloon/evaluation"))
    parser.add_argument(
        "--video", action="store_true", help="render one episode per suite in MuJoCo"
    )
    parser.add_argument(
        "--gui", action="store_true", help="open the native calibrated-camera player"
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    import jax

    if args.device != "auto":
        jax.config.update("jax_platforms", "cuda" if args.device == "gpu" else "cpu")

    from crazyflow.balloon.camera import parameters as camera_parameters
    from crazyflow.balloon.model import sample_batch
    from crazyflow.balloon.observations import ideal_bbox
    from crazyflow.balloon.policy import load_checkpoint
    from crazyflow.balloon.rollout import rollout

    if args.plant == "mujoco":
        from crazyflow.balloon.mujoco_rollout import rollout

    params, cfg, metadata = load_checkpoint(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "balloon-heldout-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_metadata": metadata,
        "seed": args.seed,
        "episodes_per_scenario": args.episodes,
        "paired_scenarios": "same underlying initial-state, target, radius and noise draws",
        "versions": _versions(),
        "requested_device": args.device,
        "devices": [str(device) for device in jax.devices()],
        "plant": args.plant,
        "training_dynamics": metadata.get("plant", "crazyflow.first_principles.cf21B_500"),
        "dynamics": (
            "MuJoCo native free rigid body + official cf21B_500 controller/actuator wrench"
            if args.plant == "mujoco"
            else metadata.get("plant", "crazyflow.first_principles.cf21B_500")
        ),
        "integrator": (
            "native MuJoCo Euler (semi-implicit rigid body); official explicit Euler RPM"
            if args.plant == "mujoco"
            else metadata.get("integrator", "official Euler;500Hz plant/attitude/motor control")
        ),
        "execution": (
            "native MuJoCo CPU worlds; official controller, wrench and actor on selected JAX device"
            if args.plant == "mujoco"
            else "Crazyflow controller, plant and actor on selected JAX device"
        ),
        "camera": camera_parameters(cfg),
        "telemetry_contract": {
            "bbox": "received policy box at tick k, with capture noise and causal delivery",
            "actions": "normalized issued policy action at start of tick k",
            "issued_commands": "issued [roll,pitch,yaw radians,total thrust N] at tick k",
            "executed_commands": (
                "actual command at first 500Hz substep of tick k; channels may change later in tick"
                if cfg.hardware.enabled
                else "command after episode-fixed action FIFO at tick k"
            ),
            "states_body_gyro_rotor_rpm": "simulation truth at boundaries j*dt, initial sample j=0",
            "target_velocity": (
                "constant world m/s; target(t)=initial_target+target_velocity*t; "
                "simulation truth only"
            ),
            "sensor_delay_steps_action_delay_steps": "simulation debug only; never actor inputs",
            "hardware_timing": (
                "camera/gyro/attitude capture age at policy tick; scheduled RPY/T delay in ms; "
                "debug only, never actor inputs"
                if cfg.hardware.enabled
                else "legacy fixed policy-tick FIFOs"
            ),
        },
        "policy_inputs": "bounding box history, attitude, angular rates and previous command",
        "metric_protocol": {
            "denominator": "all sampled episodes, including every failure; no resets",
            "collision": "first relative swept sphere entry at every integration substep",
            "drone_collision_radius_m": cfg.drone_radius,
            "drone_collision_offset_body_m": cfg.collision_offset,
            "min_center_altitude_m": cfg.min_altitude,
            "max_distance_from_start_m": cfg.max_range,
            "impact_speed_diagnostic_threshold_m_s": cfg.max_impact_speed,
            "impact_speed_m_s": "collision sphere-center velocity relative to moving target",
            "drone_collision_speed_m_s": (
                "absolute world speed of the drone collision sphere center"
            ),
            "success": "any first physical sphere contact; speed is only a diagnostic",
            "tracking_timeout_s": cfg.max_unseen_seconds,
            "engine_role": (
                "MuJoCo mj_step integrates position/quaternion/linear and angular velocity; "
                "shared official controller and motor/propeller model generate external wrench. "
                "A separate integrator, not independently identified aircraft/actuator dynamics."
                if args.plant == "mujoco"
                else "Crazyflow integrates flight; MuJoCo only renders frozen geometry"
            ),
        },
        "scenarios": {},
    }
    batch_key, rollout_key = jax.random.split(jax.random.PRNGKey(args.seed))
    for scenario in ("nominal", "noisy", "perturbed"):
        batch = sample_batch(batch_key, cfg, args.episodes, scenario=scenario)
        trajectory = rollout(params, batch, cfg, rollout_key, noise=scenario != "nominal")
        trajectory = jax.device_get(trajectory)
        batch = jax.device_get(batch)
        metrics = summarize_trajectories(
            trajectory["substep_states"],
            batch["target"],
            batch["radius"],
            dt=cfg.dt / cfg.substeps,
            target_velocities=batch.get("target_velocity"),
            drone_radius=cfg.drone_radius,
            collision_offset=cfg.collision_offset,
            min_altitude=cfg.min_altitude,
            max_range=cfg.max_range,
            max_impact_speed=cfg.max_impact_speed,
            visible=np.asarray(trajectory["bbox"])[..., 4] > 0.5,
            visibility_dt=cfg.dt,
            tracking_timeout=cfg.max_unseen_seconds,
            objective=float(trajectory["loss"]),
        )
        initial_boxes, _ = ideal_bbox(
            batch["state"][:, :3], batch["state"][:, 3:6], batch["target"], batch["radius"], cfg
        )
        initial_boxes = np.asarray(initial_boxes)
        for episode, record in enumerate(metrics["per_episode"]):
            record["target_diameter_m"] = float(2 * batch["radius"][episode])
            record["initial_ideal_bbox_width_px"] = float(
                initial_boxes[episode, 2] * cfg.image_width / 2
            )
            record["initial_ideal_bbox_height_px"] = float(
                initial_boxes[episode, 3] * cfg.image_height / 2
            )
            record["initial_ideal_bbox_valid"] = bool(initial_boxes[episode, 4] > 0.5)
        if "native_contacts" in trajectory:
            native_contacts = np.asarray(trajectory["native_contacts"], dtype=bool)
            for episode, record in enumerate(metrics["per_episode"]):
                indices = np.flatnonzero(native_contacts[:, episode])
                contact_time = float(indices[0] * cfg.dt / cfg.substeps) if len(indices) else None
                record["native_contact_time_s"] = contact_time
                record["native_contact_by_first_event"] = (
                    contact_time is not None
                    and contact_time <= record["event_time_s"] + cfg.dt / cfg.substeps + 1e-9
                )
            metrics["native_contact_diagnostic"] = {
                "episodes_contacting_by_first_event": sum(
                    r["native_contact_by_first_event"] for r in metrics["per_episode"]
                ),
                "note": "native collision cache at 500Hz poses; primary metric uses swept geometry",
            }
        arrays = {name: np.asarray(value) for name, value in trajectory.items()}
        arrays.update(control_telemetry(trajectory, cfg, rollout_key, noise=scenario != "nominal"))
        arrays.update(
            {f"initial_{name}": np.asarray(batch[name]) for name in ("state", "target", "radius")}
        )
        arrays["target_velocity"] = np.asarray(
            batch.get("target_velocity", np.zeros_like(batch["target"]))
        )
        arrays.update(
            {f"domain_{name}": np.asarray(value) for name, value in batch["domain"].items()}
        )
        array_path = args.output / f"{scenario}.npz"
        np.savez_compressed(array_path, **arrays)
        metrics["arrays"] = {"path": str(array_path), "sha256": _sha256(array_path)}
        if args.video:
            metrics["video"] = render_replay(
                np.asarray(trajectory["substep_states"])[:, 0],
                np.asarray(batch["target"])[0],
                float(batch["radius"][0]),
                cfg,
                args.output / f"{scenario}.mp4",
                stop_time=metrics["per_episode"][0]["event_time_s"],
                plant=args.plant,
                target_velocity=arrays["target_velocity"][0],
            )
            metrics["video"]["episode"] = 0
            metrics["video"]["outcome"] = metrics["per_episode"][0]["outcome"]
        report["scenarios"][scenario] = metrics
        print(f"{scenario}: {metrics['successes']}/{args.episodes} contacts; {metrics['outcomes']}")
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.output / 'metrics.json'}")
    if args.gui:
        subprocess.run(
            [sys.executable, "-m", "crazyflow.balloon.play", "--evaluation", str(args.output)],
            env={**os.environ, "MUJOCO_GL": "glfw"},
            check=True,
        )


if __name__ == "__main__":
    main()
