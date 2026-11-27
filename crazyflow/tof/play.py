"""Live Crazyflow ToF flight and MuJoCo visualization of saved trajectories.

The aircraft poses are computed by Crazyflow. MuJoCo renders the same map, goal
and recorded poses with the bundled Crazyflie visual mesh; no second controller
or alternative flight integrator is hidden in this command. Independent plant
validation is provided by the frozen Isaac Sim evaluation harness.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import mujoco


def _numbers(values: Any) -> str:
    return " ".join(f"{x:.9g}" for x in np.asarray(values).ravel())


def sensor_packet(
    front: np.ndarray, auxiliary: np.ndarray, actor_observation: np.ndarray
) -> dict[str, np.ndarray]:
    """Keep clean ranges distinct from the actual noisy inputs before normalization.

    The actor has ten 28-channel history entries, ending with L/R/back/down,
    followed by its current 64 front pixels. Both range groups are scaled by 30
    in the task observation. No new observation or random sensor sample is made.
    """
    observation = np.asarray(actor_observation).reshape(-1)
    if observation.shape != (344,):
        raise ValueError("the ToF actor observation must have 344 channels")
    return {
        "clean_front": np.asarray(front).reshape(8, 8).copy(),
        "clean_auxiliary": np.asarray(auxiliary).reshape(4).copy(),
        "policy_front": observation[-64:].reshape(8, 8).copy() / 30,
        "policy_auxiliary": observation[276:280].copy() / 30,
    }


class LiveFlight:
    """Advance one genuine Crazyflow world; retain exact terminal poses until resumed."""

    def __init__(self, actor: Any, env: Any, seed: int = 42):
        import jax

        if env.num_envs != 1:
            raise ValueError("live play requires exactly one environment")
        self.jax, self.env = jax, env
        self.actor = jax.jit(actor)
        self.seed = seed
        self.generation = 0
        self.completed = 0
        self.manual_resets = 0
        self.restarts = 0
        self.last_outcome = "none"
        self._scene = jax.device_get(env.scene)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        """Sample a new pose/goal on the existing maps without changing task physics."""
        if seed is None:
            self.seed += 1
            self.manual_resets += 1
        else:
            self.seed = seed
        self.state = self.env.reset(self.jax.random.key(self.seed))
        self._begin_episode()

    def restart(self) -> None:
        """Repeat this episode's exact initial state, including all random streams."""
        self.state = self.initial_state
        self.restarts += 1
        self._begin_episode()

    def _begin_episode(self) -> None:
        state = self.jax.device_get(self.state)
        self.initial_state = self.state
        self.elapsed = 0.0
        self.episode_return = 0.0
        self.steps = 0
        self.pending_episode = False
        self.status = "flying"
        self.generation += 1
        map_id = int(np.asarray(state.map_id)[0])
        count = int(np.sum(np.asarray(self._scene.counts)[map_id]))
        self.episode = {
            "positions": np.asarray(state.sim.states.pos[0, :1]).copy(),
            "quaternions_xyzw": np.asarray(state.sim.states.quat[0, :1]).copy(),
            "goal": np.asarray(state.goal[0]).copy(),
            "centers": np.asarray(self._scene.centers[map_id, :count]),
            "half_sizes": np.asarray(self._scene.half_sizes[map_id, :count]),
            "kinds": np.asarray(self._scene.kinds[map_id, :count]),
            "rotations": np.asarray(self._scene.rotations[map_id, :count]),
            "room_half_size": np.asarray(self._scene.room_half_size),
            "floor_height": np.asarray(self._scene.floor),
            "ceiling_height": np.asarray(self._scene.ceiling),
            "dt": np.asarray(self.env.config.dt),
            "front_ray_directions": np.asarray(self.jax.device_get(self.env.ray_directions)),
            "front_depth_m": np.asarray(state.depth[:1]).copy(),
            "auxiliary_depth_m": np.asarray(state.auxiliary_depth[:1]).copy(),
        }
        self.map_id = map_id
        self.packet = sensor_packet(state.depth[0], state.auxiliary_depth[0], state.obs[0])
        self.velocity = np.asarray(state.sim.states.vel[0, 0])

    def continue_episode(self) -> None:
        """Use the environment's own already-sampled autoreset state."""
        if self.pending_episode:
            self._begin_episode()

    def advance(self) -> None:
        """Issue the frozen actor mean and display the resulting physical transition."""
        if self.pending_episode:
            return
        action = self.actor(self.state.obs)
        if not np.isfinite(np.asarray(self.jax.device_get(action))).all():
            raise ValueError("the selected actor returned nonfinite live actions")
        self.state, transition = self.env.step(self.state, action)
        state, result = self.jax.device_get((self.state, transition))
        terminal = bool(np.asarray(result.done)[0])
        if terminal:
            pos, quat, velocity = (
                result.terminal_pos[0],
                result.terminal_quat[0],
                result.terminal_vel[0],
            )
            front = result.terminal_critic_obs[0, -64:] / 30
            # Critic history has 37 channels per tick, and the same four clean
            # auxiliary ranges at offsets 24..27 of the most recent history row.
            auxiliary = result.terminal_critic_obs[0, 9 * 37 + 24 : 9 * 37 + 28] / 30
            observation = result.terminal_obs[0]
        else:
            pos, quat = state.sim.states.pos[0, 0], state.sim.states.quat[0, 0]
            velocity = state.sim.states.vel[0, 0]
            front, auxiliary, observation = state.depth[0], state.auxiliary_depth[0], state.obs[0]
        if not np.isfinite(np.r_[pos, quat, velocity]).all():
            raise ValueError("the live Crazyflow transition contains nonfinite physical state")
        self.steps += 1
        self.elapsed = self.steps * self.env.config.dt
        self.episode_return += float(np.asarray(result.reward)[0])
        self.velocity = np.asarray(velocity)
        self.packet = sensor_packet(front, auxiliary, observation)
        for name, value in (
            ("positions", pos),
            ("quaternions_xyzw", quat),
            ("front_depth_m", front),
            ("auxiliary_depth_m", auxiliary),
        ):
            self.episode[name] = np.concatenate([self.episode[name], np.asarray(value)[None]])
        if terminal:
            self.completed += 1
            self.pending_episode = True
            if bool(np.asarray(result.collision)[0]):
                self.status = "collision"
            elif bool(np.asarray(result.success)[0]):
                self.status = "success"
            elif bool(np.asarray(result.instability)[0]):
                self.status = "unstable"
            elif bool(np.asarray(result.truncated)[0]):
                self.status = "timeout"
            else:
                self.status = "unstable"
            self.last_outcome = self.status


def make_visual_model(
    centers: np.ndarray,
    half_sizes: np.ndarray,
    kinds: np.ndarray,
    goal: np.ndarray,
    *,
    rotations: np.ndarray | None = None,
    room_half_size: float = 8.0,
    floor_height: float = 0.01,
    ceiling_height: float = 2.0,
    width: int = 960,
    height: int = 720,
) -> mujoco.MjModel:
    """Build a standalone scene using the official Crazyflow CF21B visual assets."""
    import mujoco

    from crazyflow.paths import resource_path

    centers = np.asarray(centers, dtype=float)
    half_sizes = np.asarray(half_sizes, dtype=float)
    kinds = np.asarray(kinds, dtype=int)
    goal = np.asarray(goal, dtype=float)
    if centers.ndim != 2 or centers.shape[-1] != 3 or half_sizes.shape != centers.shape:
        raise ValueError("obstacle centers and half_sizes must both have shape (obstacles, 3)")
    if kinds.shape != centers.shape[:1] or goal.shape != (3,):
        raise ValueError("kinds and goal do not match the scene geometry")
    if not (np.isfinite(centers).all() and np.isfinite(half_sizes).all()):
        raise ValueError("obstacle geometry must be finite")
    if np.any(half_sizes <= 0) or not np.isfinite(goal).all():
        raise ValueError("obstacle sizes must be positive and goal must be finite")
    if np.any(~np.isin(kinds, (0, 1, 2))):
        raise ValueError("obstacle kinds must be 0 (box), 1 (cylinder) or 2 (cone)")
    if rotations is None:
        rotations = np.broadcast_to(np.eye(3), (*centers.shape[:1], 3, 3))
    rotations = np.asarray(rotations)
    if rotations.shape != (*centers.shape[:1], 3, 3) or not np.isfinite(rotations).all():
        raise ValueError("rotations must have shape (obstacles, 3, 3) and be finite")
    from scipy.spatial.transform import Rotation

    path = resource_path("assets/drones/cf21B_500.xml")
    aircraft = ET.parse(path).getroot()
    root = ET.Element("mujoco", model="crazyflow_tof_visualization")
    ET.SubElement(root, "compiler", angle="degree", inertiafromgeom="false")
    ET.SubElement(root, "option", timestep="0.00666666667", gravity="0 0 -9.81")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7")
    ET.SubElement(visual, "map", znear="0.001", zfar="100")
    asset = copy.deepcopy(aircraft.find("asset"))
    for mesh in asset.findall("mesh"):
        mesh.set("file", str((path.parent / "assets" / mesh.attrib["file"]).resolve()))
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1=".18 .22 .27",
        rgb2=".23 .28 .33",
        width="256",
        height="256",
    )
    ET.SubElement(
        asset,
        "material",
        name="floor",
        texture="grid",
        texrepeat="16 16",
        texuniform="true",
        reflectance=".05",
    )
    # MuJoCo has no cone primitive. A unit cone visual mesh represents the exact
    # analytic task shape, with 64 radial facets and its full radius/half-height.
    cone_vertices = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    cone_vertices.extend(
        [[np.cos(t), np.sin(t), -1.0] for t in np.linspace(0, 2 * np.pi, 64, endpoint=False)]
    )
    cone_faces = []
    for i in range(64):
        cone_faces.extend([[0, 2 + i, 2 + (i + 1) % 64], [1, 2 + (i + 1) % 64, 2 + i]])
    for index in np.flatnonzero(kinds == 2):
        ET.SubElement(
            asset,
            "mesh",
            name=f"cone_{index}",
            vertex=_numbers(cone_vertices),
            face=_numbers(cone_faces),
            scale=_numbers(half_sizes[index]),
        )
    root.append(asset)
    root.append(copy.deepcopy(aircraft.find("default")))
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 6", dir="0 0 -1", directional="true")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size=f"{room_half_size} {room_half_size} .1",
        pos=f"0 0 {floor_height}",
        material="floor",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        world,
        "geom",
        name="ceiling",
        type="plane",
        size=f"{room_half_size} {room_half_size} .1",
        pos=f"0 0 {ceiling_height}",
        quat="0 1 0 0",
        rgba=".8 .9 1 .04",
        contype="0",
        conaffinity="0",
    )
    wall_z = (floor_height + ceiling_height) / 2
    wall_height = (ceiling_height - floor_height) / 2
    for axis in range(2):
        for sign in (-1, 1):
            pos = np.array([0.0, 0.0, wall_z])
            pos[axis] = sign * room_half_size
            size = np.array([room_half_size, room_half_size, wall_height])
            size[axis] = 0.01
            ET.SubElement(
                world,
                "geom",
                name=f"wall_{axis}_{sign}",
                type="box",
                pos=_numbers(pos),
                size=_numbers(size),
                rgba=".4 .6 .8 .12",
                contype="0",
                conaffinity="0",
            )
    for index, (center, size, kind, rotation) in enumerate(
        zip(centers, half_sizes, kinds, rotations, strict=True)
    ):
        geom_type = ("box", "cylinder", "mesh")[kind]
        dimensions = size if kind == 0 else size[[0, 2]]
        attributes = {"size": _numbers(dimensions)} if kind != 2 else {"mesh": f"cone_{index}"}
        ET.SubElement(
            world,
            "geom",
            name=f"obstacle_{index}",
            type=geom_type,
            pos=_numbers(center),
            **attributes,
            quat=_numbers(Rotation.from_matrix(rotation).as_quat(scalar_first=True)),
            rgba=(".35 .48 .58 1" if kind == 0 else ".53 .42 .32 1"),
            contype="0",
            conaffinity="0",
        )
    ET.SubElement(
        world, "site", name="goal", type="sphere", pos=_numbers(goal), size=".16", rgba=".2 1 .3 .7"
    )
    drone = copy.deepcopy(aircraft.find("worldbody/body[@name='drone']"))
    drone.set("mocap", "true")
    drone.set("pos", "0 0 1")
    for geom in drone.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    # Optical axis +X, +Z up. The camera follows aircraft pose, not world yaw.
    ET.SubElement(
        drone, "camera", name="tof_view", pos=".03 0 0", xyaxes="0 -1 0 0 0 1", fovy="54.5"
    )
    world.append(drone)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def selected_episode(trajectories: dict[str, np.ndarray], episode: int) -> dict[str, np.ndarray]:
    """Select a completed trajectory, excluding frozen post-termination frames."""
    positions = trajectories["positions"]
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (frames, episodes, 3)")
    if episode < 0 or episode >= positions.shape[1]:
        raise ValueError(f"episode must be in [0, {positions.shape[1] - 1}]")
    length = min(int(trajectories["lengths"][episode]) + 1, len(positions))
    if length < 1:
        raise ValueError("trajectory has no initial frame")
    map_id = int(trajectories["map_ids"][episode])
    obstacle_count = (
        int(np.sum(trajectories["scene_counts"][map_id]))
        if "scene_counts" in trajectories
        else trajectories["scene_centers"].shape[1]
    )
    result = {
        "positions": np.asarray(positions[:length, episode]),
        "quaternions_xyzw": np.asarray(trajectories["quaternions_xyzw"][:length, episode]),
        "goal": np.asarray(trajectories["goals"][episode]),
        "centers": np.asarray(trajectories["scene_centers"][map_id, :obstacle_count]),
        "half_sizes": np.asarray(trajectories["scene_half_sizes"][map_id, :obstacle_count]),
        "kinds": np.asarray(trajectories["scene_kinds"][map_id, :obstacle_count]),
        "rotations": np.asarray(trajectories["scene_rotations"][map_id, :obstacle_count]),
        "room_half_size": np.asarray(trajectories["room_half_size"]),
        "floor_height": np.asarray(trajectories.get("floor_height", 0.01)),
        "ceiling_height": np.asarray(trajectories.get("ceiling_height", 2.0)),
        "dt": np.asarray(trajectories["dt"]),
    }
    if "front_depth_m" in trajectories:
        result["front_depth_m"] = np.asarray(trajectories["front_depth_m"][:length, episode])
        result["front_ray_directions"] = np.asarray(trajectories["front_ray_directions"])
    return result


class TrajectoryViewer:
    """Small MuJoCo renderer with follow, overview and aircraft camera modes."""

    def __init__(self, episode: dict[str, np.ndarray], width: int, height: int, camera: str):
        import mujoco

        self.mujoco = mujoco
        self.episode = episode
        self.model = make_visual_model(
            episode["centers"],
            episode["half_sizes"],
            episode["kinds"],
            episode["goal"],
            rotations=episode.get("rotations"),
            room_half_size=float(episode["room_half_size"]),
            floor_height=float(episode.get("floor_height", 0.01)),
            ceiling_height=float(episode.get("ceiling_height", 2.0)),
            width=width,
            height=height,
        )
        self.data = mujoco.MjData(self.model)
        self.camera_mode = camera
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        if camera == "fpv":
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.camera.fixedcamid = self.model.camera("tof_view").id
        else:
            self.camera.azimuth = 135
            self.camera.elevation = -40 if camera == "overview" else -22
            self.camera.distance = (
                float(episode["room_half_size"]) * 2.6 if camera == "overview" else 2.2
            )
            self.camera.lookat[:] = (0, 0, 0.7)
        self.renderer = None
        self.width, self.height = width, height
        self.set_frame(0)

    def set_frame(self, index: int) -> None:
        """Transfer one exact recorded pose to the official visual mesh."""
        pos = self.episode["positions"][index]
        quat = self.episode["quaternions_xyzw"][index]
        if not np.isfinite(np.r_[pos, quat]).all() or np.linalg.norm(quat) < 1e-8:
            raise ValueError("cannot render a nonfinite or invalid recorded pose")
        self.data.mocap_pos[0] = pos
        self.data.mocap_quat[0] = quat[[3, 0, 1, 2]] / np.linalg.norm(quat)
        self.data.time = index * float(self.episode["dt"])
        self.mujoco.mj_kinematics(self.model, self.data)
        self.mujoco.mj_comPos(self.model, self.data)
        self.mujoco.mj_camlight(self.model, self.data)
        if self.camera_mode == "follow":
            self.camera.lookat[:] = pos
        self.index = index

    def _decorate(self, scene: mujoco.MjvScene) -> None:
        """Draw the recent realized path with the same scale as obstacle geometry."""
        points = self.episode["positions"][max(0, self.index - 150) : self.index + 1]
        for start, end in zip(points[:-1], points[1:], strict=True):
            if scene.ngeom >= scene.maxgeom:
                break
            if np.linalg.norm(end - start) < 1e-8:
                continue
            geom = scene.geoms[scene.ngeom]
            self.mujoco.mjv_initGeom(
                geom,
                self.mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.array([1.0, 0.65, 0.12, 1.0]),
            )
            self.mujoco.mjv_connector(geom, self.mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, start, end)
            scene.ngeom += 1

        if "front_depth_m" in self.episode:
            from scipy.spatial.transform import Rotation

            packet = min(self.index, len(self.episode["front_depth_m"]) - 1)
            distances = self.episode["front_depth_m"][packet].reshape(-1)
            directions = self.episode["front_ray_directions"].reshape(-1, 3)
            # Sparse rays show the exact noise-free ToF geometry at this pose.
            rotation = Rotation.from_quat(self.episode["quaternions_xyzw"][self.index])
            origin = self.episode["positions"][self.index] + rotation.apply([0.02, 0, 0])
            for ray in (0, 7, 27, 28, 35, 36, 56, 63):
                if scene.ngeom >= scene.maxgeom or not np.isfinite(distances[ray]):
                    continue
                endpoint = origin + rotation.apply(directions[ray] * distances[ray])
                geom = scene.geoms[scene.ngeom]
                self.mujoco.mjv_initGeom(
                    geom,
                    self.mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).ravel(),
                    np.array([0.2, 0.9, 1.0, 0.6]),
                )
                self.mujoco.mjv_connector(
                    geom, self.mujoco.mjtGeom.mjGEOM_LINE, 1.0, origin, endpoint
                )
                scene.ngeom += 1

            if "auxiliary_depth_m" in self.episode:
                auxiliary = self.episode["auxiliary_depth_m"][packet]
                axes = np.array([[0, 1, 0], [0, -1, 0], [-1, 0, 0], [0, 0, -1]])
                colors = ([0.4, 1, 0.4, 1], [1, 0.5, 0.3, 1], [0.8, 0.5, 1, 1], [1, 1, 0.3, 1])
                for axis, distance, color in zip(axes, auxiliary, colors, strict=True):
                    if scene.ngeom >= scene.maxgeom or not np.isfinite(distance):
                        continue
                    start = self.episode["positions"][self.index] + rotation.apply(axis * 0.02)
                    end = start + rotation.apply(axis * distance)
                    geom = scene.geoms[scene.ngeom]
                    self.mujoco.mjv_initGeom(
                        geom,
                        self.mujoco.mjtGeom.mjGEOM_LINE,
                        np.zeros(3),
                        np.zeros(3),
                        np.eye(3).ravel(),
                        np.asarray(color),
                    )
                    self.mujoco.mjv_connector(
                        geom, self.mujoco.mjtGeom.mjGEOM_LINE, 2.0, start, end
                    )
                    scene.ngeom += 1

    def render(self) -> np.ndarray:
        """Return one RGB frame; OpenGL backend is chosen before importing MuJoCo."""
        if self.renderer is None:
            self.renderer = self.mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.renderer.update_scene(self.data, camera=self.camera)
        self._decorate(self.renderer.scene)
        return self.renderer.render()

    def close(self) -> None:
        """Release the OpenGL context when video rendering finishes."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def render_trajectory(
    episode: dict[str, np.ndarray],
    output: Path,
    *,
    width: int = 960,
    height: int = 720,
    camera: str = "follow",
    fps: float | None = None,
) -> dict[str, Any]:
    """Render PNG, GIF or MP4 with frame times sampled from the recorded policy clock."""
    import imageio.v2 as imageio

    if width < 16 or height < 16:
        raise ValueError("image dimensions must be at least 16 pixels")
    dt = float(episode["dt"])
    fps = 1 / dt if fps is None else fps
    if fps <= 0:
        raise ValueError("fps must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    viewer = TrajectoryViewer(episode, width, height, camera)
    duration = max((len(episode["positions"]) - 1) * dt, dt)
    indices = np.minimum(
        np.rint(np.arange(0, duration + 1e-8, 1 / fps) / dt).astype(int),
        len(episode["positions"]) - 1,
    )
    try:
        if output.suffix.lower() == ".png":
            viewer.set_frame(len(episode["positions"]) - 1)
            imageio.imwrite(output, viewer.render())
            frame_count = 1
        elif output.suffix.lower() in (".mp4", ".gif"):
            kwargs = (
                {"fps": fps}
                if output.suffix.lower() == ".mp4"
                else {"duration": 1000 / fps, "loop": 0}
            )
            if output.suffix.lower() == ".mp4":
                kwargs.update(codec="libx264", macro_block_size=1, quality=8)
            with imageio.get_writer(output, **kwargs) as writer:
                for index in indices:
                    viewer.set_frame(int(index))
                    writer.append_data(viewer.render())
            frame_count = len(indices)
        else:
            raise ValueError("--output must end in .png, .mp4 or .gif")
    finally:
        viewer.close()
    return {
        "output": str(output.resolve()),
        "frames": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "flight_backend": "crazyflow",
        "visualizer": "mujoco",
        "camera": camera,
        "trajectory_duration_s": duration,
    }


def interactive_trajectory(
    episode: dict[str, np.ndarray],
    *,
    camera: str = "follow",
    loop: bool = False,
    speed: float = 1.0,
    max_frames: int | None = None,
) -> None:
    """Show recorded Crazyflow flight in an interactive native MuJoCo window."""
    import mujoco.viewer

    if speed <= 0:
        raise ValueError("playback speed must be positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    view = TrajectoryViewer(episode, 960, 720, camera)
    dt = float(episode["dt"])
    paused = [False]

    def on_key(keycode: int) -> None:
        if keycode == 32:  # GLFW_KEY_SPACE
            paused[0] = not paused[0]

    viewer_threads = []
    existing_threads = set(threading.enumerate())
    try:
        with mujoco.viewer.launch_passive(view.model, view.data, key_callback=on_key) as handle:
            viewer_threads = [t for t in threading.enumerate() if t not in existing_threads]
            handle.cam.type = view.camera.type
            handle.cam.fixedcamid = view.camera.fixedcamid
            handle.cam.azimuth = view.camera.azimuth
            handle.cam.elevation = view.camera.elevation
            handle.cam.distance = view.camera.distance
            handle.cam.lookat[:] = view.camera.lookat
            rendered = 0
            while handle.is_running():
                for index in range(len(episode["positions"])):
                    if not handle.is_running():
                        return
                    while paused[0] and handle.is_running():
                        handle.sync()
                        time.sleep(0.02)
                    if not handle.is_running():
                        return
                    start = time.monotonic()
                    with handle.lock():
                        view.set_frame(index)
                        if camera == "follow":
                            handle.cam.lookat[:] = view.camera.lookat
                        handle.user_scn.ngeom = 0
                        view._decorate(handle.user_scn)
                    handle.sync()
                    rendered += 1
                    if max_frames is not None and rendered >= max_frames:
                        return
                    time.sleep(max(0.0, dt / speed - (time.monotonic() - start)))
                if not loop:
                    return
    finally:
        # MuJoCo 4 launch_passive owns a daemon OpenGL thread. Wait for it to
        # destroy GLFW before Python teardown; otherwise finite GUI runs can
        # race glfw.terminate and crash even after a clean context-manager exit.
        for thread in viewer_threads:
            thread.join(timeout=5.0)
        view.close()


def interactive_live(
    flight: LiveFlight,
    *,
    camera: str = "follow",
    speed: float = 1.0,
    max_frames: int | None = None,
    width: int = 1280,
    height: int = 800,
    screenshot: Path | None = None,
    report_path: Path | None = None,
    self_test: bool = False,
) -> dict[str, Any]:
    """Run a closed loop with a native GLFW world and actual ToF observation HUD."""
    import glfw
    import mujoco

    if speed <= 0 or (max_frames is not None and max_frames < 1):
        raise ValueError("speed and max_frames must be positive")
    print("Compiling live Crazyflow actor and physics (no complete flight rollout)...", flush=True)
    flight.jax.block_until_ready(flight.env.step(flight.state, flight.actor(flight.state.obs)))
    if not glfw.init():
        raise RuntimeError("GLFW could not open a display; use --headless for file rendering")
    window = glfw.create_window(
        max(width, 960), max(height, 720), "Crazyflow | Live ToF", None, None
    )
    if window is None:
        glfw.terminate()
        raise RuntimeError("GLFW could not create the live ToF window")
    glfw.make_context_current(window)
    glfw.set_window_size_limits(window, 960, 720, glfw.DONT_CARE, glfw.DONT_CARE)
    glfw.swap_interval(1)
    context = None
    view = None
    scene = None
    generation = -1
    options = mujoco.MjvOption()
    ui = {
        "paused": False,
        "policy": False,
        "camera": camera,
        "speed": speed,
        "follow": camera == "follow",
        "cursor": (0.0, 0.0),
        "buttons": [],
        "drag": None,
        "test_cursor": None,
        "world_width": max(width, 960) - 340,
    }
    checks: list[str] = []
    frames = 0
    next_tick = time.monotonic()
    terminal_until = 0.0
    self_test_stage = 0
    paused_state = None

    def set_camera(mode: str) -> None:
        ui["camera"], ui["follow"] = mode, mode == "follow"
        if view is None:
            return
        view.camera_mode = mode
        view.camera.type = (
            mujoco.mjtCamera.mjCAMERA_FIXED if mode == "fpv" else mujoco.mjtCamera.mjCAMERA_FREE
        )
        if mode == "fpv":
            view.camera.fixedcamid = view.model.camera("tof_view").id
        else:
            view.camera.azimuth = 135
            view.camera.elevation = -40 if mode == "overview" else -22
            view.camera.distance = (
                float(flight.episode["room_half_size"]) * 2.6 if mode == "overview" else 2.2
            )
            view.camera.lookat[:] = (
                [0, 0, 0.7] if mode == "overview" else flight.episode["positions"][-1]
            )

    def action(name: str) -> None:
        nonlocal next_tick, terminal_until
        if name == "pause":
            ui["paused"] = not ui["paused"]
        elif name == "reset":
            flight.reset()
            terminal_until = 0.0
        elif name == "restart":
            flight.restart()
            terminal_until = 0.0
        elif name == "sensor":
            ui["policy"] = not ui["policy"]
        elif name == "camera":
            modes = ("follow", "overview", "fpv")
            set_camera(modes[(modes.index(ui["camera"]) + 1) % len(modes)])
        next_tick = time.monotonic()

    def key_callback(_window: Any, key: int, _scan: int, pressed: int, _mods: int) -> None:
        if pressed != glfw.PRESS:
            return
        controls = {
            glfw.KEY_SPACE: "pause",
            glfw.KEY_R: "reset",
            glfw.KEY_BACKSPACE: "restart",
            glfw.KEY_T: "sensor",
            glfw.KEY_C: "camera",
        }
        if key in controls:
            action(controls[key])
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_F:
            set_camera("follow")
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            ui["speed"] = min(4.0, ui["speed"] * 2)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            ui["speed"] = max(0.125, ui["speed"] / 2)

    def cursor_pixels() -> tuple[float, float]:
        if ui["test_cursor"] is not None:
            return ui["test_cursor"]
        x, y = glfw.get_cursor_pos(window)
        ww, wh = glfw.get_window_size(window)
        fw, fh = glfw.get_framebuffer_size(window)
        return x * fw / max(ww, 1), fh - y * fh / max(wh, 1)

    def mouse_callback(_window: Any, button: int, pressed: int, _mods: int) -> None:
        x, y = cursor_pixels()
        ui["cursor"] = (x, y)
        if pressed == glfw.RELEASE:
            ui["drag"] = None
        elif pressed == glfw.PRESS:
            if x < ui["world_width"]:
                ui["drag"] = button
            elif button == glfw.MOUSE_BUTTON_LEFT:
                for name, rect in ui["buttons"]:
                    if (
                        rect.left <= x < rect.left + rect.width
                        and rect.bottom <= y < rect.bottom + rect.height
                    ):
                        action(name)
                        break

    def motion_callback(_window: Any, _x: float, _y: float) -> None:
        x, y = cursor_pixels()
        old_x, old_y = ui["cursor"]
        ui["cursor"] = (x, y)
        if ui["drag"] is None or view is None or ui["camera"] == "fpv":
            return
        movement = (
            mujoco.mjtMouse.mjMOUSE_ROTATE_V
            if ui["drag"] == glfw.MOUSE_BUTTON_LEFT
            else mujoco.mjtMouse.mjMOUSE_MOVE_V
        )
        if movement == mujoco.mjtMouse.mjMOUSE_MOVE_V:
            ui["follow"] = False
        mujoco.mjv_moveCamera(
            view.model, movement, (x - old_x) / 700, -(y - old_y) / 700, view.camera
        )

    def scroll_callback(_window: Any, _x: float, y: float) -> None:
        if view is not None and cursor_pixels()[0] < ui["world_width"]:
            mujoco.mjv_moveCamera(
                view.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y, view.camera
            )

    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_callback)
    glfw.set_cursor_pos_callback(window, motion_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    def text(rect: Any, message: str) -> None:
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            rect,
            message,
            "",
            context,
        )

    def exercise_controls() -> None:
        """Drive real key callbacks and an actual environment terminal/autoreset transition."""
        nonlocal terminal_until
        set_camera("follow")
        key_callback(window, glfw.KEY_SPACE, 0, glfw.PRESS, 0)
        assert ui["paused"]
        before = flight.steps
        assert flight.steps == before
        key_callback(window, glfw.KEY_SPACE, 0, glfw.PRESS, 0)
        assert not ui["paused"]
        checks.append("pause_resume_keys")
        initial = flight.episode["positions"][0].copy()
        key_callback(window, glfw.KEY_R, 0, glfw.PRESS, 0)
        assert flight.manual_resets == 1
        assert not np.array_equal(initial, flight.episode["positions"][0])
        checks.append("new_pose_goal_same_maps")
        reset_obs = np.asarray(flight.jax.device_get(flight.state.obs)).copy()
        reset_pos = flight.episode["positions"][0].copy()
        flight.advance()
        key_callback(window, glfw.KEY_BACKSPACE, 0, glfw.PRESS, 0)
        np.testing.assert_array_equal(flight.episode["positions"][0], reset_pos)
        np.testing.assert_array_equal(flight.jax.device_get(flight.state.obs), reset_obs)
        checks.append("exact_episode_restart_key")
        key_callback(window, glfw.KEY_T, 0, glfw.PRESS, 0)
        assert ui["policy"]
        key_callback(window, glfw.KEY_T, 0, glfw.PRESS, 0)
        assert not ui["policy"]
        checks.append("clean_policy_sensor_toggle")
        for mode in ("overview", "fpv", "follow"):
            key_callback(window, glfw.KEY_C, 0, glfw.PRESS, 0)
            assert ui["camera"] == mode
        checks.append("three_camera_modes")
        for button, label in (
            (glfw.MOUSE_BUTTON_LEFT, "mouse_orbit"),
            (glfw.MOUSE_BUTTON_RIGHT, "mouse_pan"),
        ):
            before_camera = np.r_[view.camera.azimuth, view.camera.elevation, view.camera.lookat]
            ui["test_cursor"] = (200.0, 300.0)
            mouse_callback(window, button, glfw.PRESS, 0)
            ui["test_cursor"] = (245.0, 325.0)
            motion_callback(window, 0, 0)
            mouse_callback(window, button, glfw.RELEASE, 0)
            after_camera = np.r_[view.camera.azimuth, view.camera.elevation, view.camera.lookat]
            assert not np.allclose(before_camera, after_camera), label
            checks.append(label)
        assert not ui["follow"]
        panned_target = view.camera.lookat.copy()
        view.set_frame(len(flight.episode["positions"]) - 1)
        view.camera.lookat[:] = panned_target
        np.testing.assert_array_equal(view.camera.lookat, panned_target)
        checks.append("pan_target_preserved_after_pose_update")
        distance = view.camera.distance
        scroll_callback(window, 0, 1)
        assert view.camera.distance != distance
        checks.append("mouse_wheel_zoom")
        pause_rect = next(rect for name, rect in ui["buttons"] if name == "pause")
        ui["test_cursor"] = (pause_rect.left + 5, pause_rect.bottom + 5)
        for expected in (True, False):
            mouse_callback(window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
            mouse_callback(window, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0)
            assert ui["paused"] == expected
        checks.append("clickable_pause_resume")
        ui["test_cursor"] = None
        key_callback(window, glfw.KEY_F, 0, glfw.PRESS, 0)
        # Only this explicit diagnostic moves the episode counter to its final
        # tick. Physics, actor, random streams and termination/autoreset are real.
        flight.state = flight.state.replace(
            steps=flight.state.steps * 0 + flight.env.config.max_episode_steps - 1
        )
        flight.advance()
        assert flight.pending_episode
        np.testing.assert_array_equal(
            flight.packet["clean_front"].reshape(-1), flight.episode["front_depth_m"][-1]
        )
        reset_position = np.asarray(flight.jax.device_get(flight.state.sim.states.pos[0, 0]))
        assert not np.array_equal(flight.episode["positions"][-1], reset_position)
        checks.append("terminal_pose_and_packet_before_autoreset")
        completed = flight.completed
        flight.continue_episode()
        np.testing.assert_array_equal(flight.episode["positions"][0], reset_position)
        assert flight.completed == completed and not flight.pending_episode
        checks.append("continue_real_environment_autoreset")
        terminal_until = 0.0

    try:
        while not glfw.window_should_close(window):
            now = time.monotonic()
            if not ui["paused"] and now >= next_tick:
                if flight.pending_episode:
                    if now >= terminal_until:
                        flight.continue_episode()
                else:
                    flight.advance()
                    if flight.pending_episode:
                        terminal_until = time.monotonic() + 0.8
                next_tick = now + flight.env.config.dt / ui["speed"]
            if generation != flight.generation:
                if context is not None:
                    context.free()
                if view is not None:
                    view.close()
                view = TrajectoryViewer(
                    flight.episode, max(width, 960), max(height, 720), ui["camera"]
                )
                context = mujoco.MjrContext(view.model, mujoco.mjtFontScale.mjFONTSCALE_100)
                scene = mujoco.MjvScene(view.model, maxgeom=4000)
                generation = flight.generation
                set_camera(ui["camera"])
            view.episode = flight.episode
            lookat_before = view.camera.lookat.copy()
            view.set_frame(len(flight.episode["positions"]) - 1)
            if ui["camera"] == "follow" and not ui["follow"]:
                # set_frame normally follows; preserve a user's explicit pan.
                view.camera.lookat[:] = lookat_before
            w, h = glfw.get_framebuffer_size(window)
            if w < 640 or h < 480:
                glfw.poll_events()
                time.sleep(0.02)
                continue
            panel_width = 340
            ui["world_width"] = w - panel_width
            world = mujoco.MjrRect(0, 0, w - panel_width, h)
            mujoco.mjv_updateScene(
                view.model, view.data, options, None, view.camera, mujoco.mjtCatBit.mjCAT_ALL, scene
            )
            view._decorate(scene)
            mujoco.mjr_render(world, scene, context)
            mujoco.mjr_rectangle(
                mujoco.MjrRect(w - panel_width, 0, panel_width, h), 0.07, 0.10, 0.14, 1
            )
            x = w - panel_width + 12
            title = "ACTOR INPUT (noisy)" if ui["policy"] else "CLEAN SENSOR RANGES"
            text(
                mujoco.MjrRect(x, h - 64, 320, 52),
                f"LIVE {flight.status.upper()} | seed {flight.seed} map {flight.map_id}\n"
                f"t {flight.elapsed:.2f}s ret {flight.episode_return:.1f} "
                f"done {flight.completed} | {ui['speed']:g}x",
            )
            ui["buttons"] = []
            for index, (name, label) in enumerate(
                (
                    ("pause", "Resume" if ui["paused"] else "Pause"),
                    ("reset", "Reset [R]"),
                    ("camera", "Camera [C]"),
                )
            ):
                rect = mujoco.MjrRect(x + 106 * index, h - 100, 100, 30)
                mujoco.mjr_rectangle(rect, 0.15, 0.24, 0.32, 1)
                text(rect, label)
                ui["buttons"].append((name, rect))
            prefix = "policy" if ui["policy"] else "clean"
            front, aux = flight.packet[prefix + "_front"], flight.packet[prefix + "_auxiliary"]
            text(
                mujoco.MjrRect(x, h - 160, 320, 52),
                f"{title} [T]\nFront +X 8x8 [m] | up=top, left=+Y",
            )
            cell = 34
            grid_top = h - 166
            for row in range(8):
                for col in range(8):
                    value = float(front[row, col])
                    normalized = np.clip(value / flight.env.config.camera_max_distance, 0, 1)
                    rect = mujoco.MjrRect(
                        x + col * cell, grid_top - (row + 1) * cell, cell - 1, cell - 1
                    )
                    mujoco.mjr_rectangle(
                        rect,
                        0.55 * (1 - normalized) + 0.08,
                        0.15 + 0.35 * normalized,
                        0.15 + 0.5 * normalized,
                        1,
                    )
                    text(rect, f"{value:.1f}" if abs(value) >= 1 else f"{value:.2f}")
            minimum, maximum = float(np.min(front)), float(np.max(front))
            center = float(np.mean(front[3:5, 3:5]))
            grid_bottom = grid_top - 8 * cell
            text(
                mujoco.MjrRect(x, grid_bottom - 62, 320, 52),
                f"Front min / center 2x2 mean / max [m]\n"
                f"{minimum:.2f} / {center:.2f} / {maximum:.2f}",
            )
            text(
                mujoco.MjrRect(x, grid_bottom - 170, 320, 100),
                f"LEFT  +Y   {aux[0]:6.2f} m\nRIGHT -Y   {aux[1]:6.2f} m\n"
                f"BACK  -X   {aux[2]:6.2f} m\nDOWN  -Z   {aux[3]:6.2f} m",
            )
            text(
                mujoco.MjrRect(x, 5, 320, 100),
                "Space pause | R reset | Backspace restart\n"
                "C camera | F follow | T clean/noisy\n"
                "+/- speed | Esc close\n"
                "Mouse: left orbit / right pan / wheel",
            )
            text(
                mujoco.MjrRect(12, h - 64, max(w - panel_width - 24, 1), 52),
                "LIVE closed loop | "
                + ui["camera"]
                + (" | PAUSED" if ui["paused"] else "")
                + f" | last {flight.last_outcome}"
                + "\nClean rays: cyan front | green/red/purple/yellow L/R/back/down",
            )
            if self_test and self_test_stage < 3:
                if self_test_stage == 0:
                    key_callback(window, glfw.KEY_SPACE, 0, glfw.PRESS, 0)
                    paused_state = flight.state
                else:
                    assert flight.state is paused_state
                    if self_test_stage == 2:
                        key_callback(window, glfw.KEY_SPACE, 0, glfw.PRESS, 0)
                        checks.append("physics_held_across_paused_gui_frames")
                        exercise_controls()
                # Make the next loop's physics deadline due, so these frames
                # verify pause rather than merely falling between policy ticks.
                next_tick = time.monotonic() - 1
                self_test_stage += 1
            frames += 1
            finished = max_frames is not None and frames >= max_frames
            if screenshot is not None and finished:
                import imageio.v2 as imageio

                rgb = np.empty((h, w, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, w, h), context)
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(screenshot, np.flipud(rgb))
            glfw.swap_buffers(window)
            glfw.poll_events()
            if finished:
                break
            time.sleep(0.005)
    finally:
        if context is not None:
            context.free()
        if view is not None:
            view.close()
        glfw.destroy_window(window)
        glfw.terminate()
    report = {
        "mode": "live_closed_loop",
        "flight_backend": "crazyflow",
        "visualizer": "mujoco",
        "frames": frames,
        "seed": flight.seed,
        "map_id": flight.map_id,
        "steps_current_episode": flight.steps,
        "completed_episodes": flight.completed,
        "manual_resets": flight.manual_resets,
        "exact_restarts": flight.restarts,
        "last_outcome": flight.last_outcome,
        "camera": ui["camera"],
        "sensor_display": "policy_noisy_before_normalization" if ui["policy"] else "clean_ranges",
        "sensor_packet": {name: value.tolist() for name, value in flight.packet.items()},
        "control_checks": checks,
        "self_test_forced_final_episode_tick": self_test,
        "physical_config": flight.env.config.to_dict(),
    }
    if report_path is None and screenshot is not None:
        report_path = screenshot.with_suffix(".json")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("sensor_packet", "physical_config")}
        )
    )
    return report


def main() -> None:
    """Run live checkpoint flight, replay saved flight, or export a frozen rollout."""
    from crazyflow.tof.evaluate import build_parser, load_actor_environment, rollout

    parser = build_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == "checkpoint":
            action.required = False
        if action.dest == "episodes":
            action.default = 1
        if action.dest == "output":
            action.default = None
            action.help = "Headless .png, .gif or .mp4; omit for interactive MuJoCo"
    parser.add_argument("--trajectory", type=Path, help="Saved evaluation trajectories.npz")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", choices=("follow", "overview", "fpv"), default="follow")
    parser.add_argument("--width", type=int, help="Width (live GUI 1280; file export 960)")
    parser.add_argument("--height", type=int, help="Height (live GUI 800; file export 720)")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--headless", action="store_true", help="Render without a display")
    parser.add_argument("--record", type=Path, dest="output", help="Alias for --output")
    parser.add_argument(
        "--gui-frames", type=int, help="Exit GUI after this many frames (smoke check)"
    )
    parser.add_argument("--loop", action="store_true", help="Loop interactive playback")
    parser.add_argument("--speed", type=float, default=1.0, help="Interactive playback speed")
    parser.add_argument("--screenshot", type=Path, help="Save final live GUI framebuffer as PNG")
    parser.add_argument(
        "--gui-report", type=Path, help="Save live GUI state/control checks as JSON"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Exercise live GUI input and autoreset"
    )
    args = parser.parse_args()
    if (args.trajectory is None) == (args.checkpoint is None):
        parser.error("provide exactly one of --checkpoint and --trajectory")
    if args.trajectory is not None and args.map_seed is not None:
        parser.error("--map-seed applies to a new rollout from --checkpoint")
    if args.headless and args.output is None:
        args.output = Path("/outputs/tof-preview.png")
    live = args.checkpoint is not None and args.output is None
    args.width = args.width if args.width is not None else (1280 if live else 960)
    args.height = args.height if args.height is not None else (800 if live else 720)
    if live and args.max_steps is not None:
        parser.error("--max-steps applies to file export; live GUI continues until closed")
    if args.self_test and (not live or args.gui_frames is None or args.gui_frames < 12):
        parser.error("--self-test requires live --checkpoint and --gui-frames 12 or greater")
    if (args.screenshot or args.gui_report) and not live:
        parser.error("--screenshot and --gui-report apply to live checkpoint GUI")
    if args.screenshot and (args.screenshot.suffix.lower() != ".png" or not args.gui_frames):
        parser.error("--screenshot requires a .png path and positive --gui-frames")
    if args.gui_frames is not None and args.gui_frames < 1:
        parser.error("--gui-frames must be positive")
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.output:
        # Must happen before Crazyflow imports MuJoCo while creating the environment.
        os.environ.setdefault("MUJOCO_GL", "egl" if args.device == "gpu" else "osmesa")
    else:
        os.environ["MUJOCO_GL"] = "glfw"
    if live:
        if args.episodes != 1 or args.episode != 0:
            parser.error("live GUI uses --episodes 1 and --episode 0")
        actor, env, _ = load_actor_environment(args)
        interactive_live(
            LiveFlight(actor, env, args.seed),
            camera=args.camera,
            speed=args.speed,
            max_frames=args.gui_frames,
            width=args.width,
            height=args.height,
            screenshot=args.screenshot,
            report_path=args.gui_report,
            self_test=args.self_test,
        )
        return
    if args.trajectory:
        with np.load(args.trajectory, allow_pickle=False) as archive:
            trajectories = dict(archive)
    else:
        if args.episodes < 1:
            parser.error("--episodes must be positive")
        actor, env, _ = load_actor_environment(args)
        report, trajectories = rollout(actor, env, seed=args.seed, max_steps=args.max_steps)
        print(json.dumps({"episodes": report["episodes"], "success_rate": report["success_rate"]}))
    episode = selected_episode(trajectories, args.episode)
    if args.output:
        print(
            json.dumps(
                render_trajectory(
                    episode,
                    args.output,
                    width=args.width,
                    height=args.height,
                    camera=args.camera,
                    fps=args.fps,
                ),
                indent=2,
            )
        )
    else:
        interactive_trajectory(
            episode,
            camera=args.camera,
            loop=args.loop,
            speed=args.speed,
            max_frames=args.gui_frames,
        )


if __name__ == "__main__":
    main()
