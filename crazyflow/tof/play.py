"""MuJoCo visualization of exact Crazyflow ToF obstacle-avoidance trajectories.

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


def main() -> None:
    """Load a policy or saved rollout and visualize the exact Crazyflow flight."""
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
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--headless", action="store_true", help="Render without a display")
    parser.add_argument("--record", type=Path, dest="output", help="Alias for --output")
    parser.add_argument(
        "--gui-frames", type=int, help="Exit GUI after this many frames (smoke check)"
    )
    parser.add_argument("--loop", action="store_true", help="Loop interactive playback")
    parser.add_argument("--speed", type=float, default=1.0, help="Interactive playback speed")
    args = parser.parse_args()
    if (args.trajectory is None) == (args.checkpoint is None):
        parser.error("provide exactly one of --checkpoint and --trajectory")
    if args.trajectory is not None and args.map_seed is not None:
        parser.error("--map-seed applies to a new rollout from --checkpoint")
    if args.headless and args.output is None:
        args.output = Path("/outputs/tof-preview.png")
    if args.output:
        # Must happen before Crazyflow imports MuJoCo while creating the environment.
        os.environ.setdefault("MUJOCO_GL", "egl" if args.device == "gpu" else "osmesa")
    else:
        os.environ["MUJOCO_GL"] = "glfw"
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
