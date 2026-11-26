"""Native MuJoCo/GLFW world, metric XY map and calibrated FPV for frozen flights.

The player displays saved or newly generated flights from the selected native
MuJoCo or Crazyflow plant. New seeds retain that plant; the window replays its
recorded states and separates policy measurements from simulation truth.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, Callable

import numpy as np

from crazyflow.balloon.camera import camera_pose, frustum_corners, intrinsics, parameters
from crazyflow.balloon.config import Config
from crazyflow.balloon.evaluate import (
    _quaternion,
    control_telemetry,
    make_replay_model,
    summarize_trajectories,
)
from crazyflow.balloon.xy_map import draw_map, episode_bounds, layout

SUITES = ("nominal", "noisy", "perturbed")
GIF_SEMANTIC_COLORS = (
    (38, 224, 255),  # aircraft on the XY map
    (255, 92, 31),  # target on the XY map
    (166, 76, 43),  # prescribed target future
    (38, 255, 76),  # ideal box
    (255, 166, 13),  # received box
    (255, 204, 51),  # aircraft world trail
    (255, 255, 255),
    (0, 0, 0),
    # Keep a tiny distant red balloon red before it dominates the image.
    (255, 0, 0),
    (255, 32, 24),
    (224, 24, 20),
    (160, 16, 16),
    (96, 8, 8),
    (255, 80, 72),
    (255, 128, 120),
    (255, 192, 184),
)
REPLAY_ARRAYS = (
    "substep_states",
    "states",
    "bbox",
    "actions",
    "body_gyro",
    "rotor_rpm",
    "issued_commands",
    "executed_commands",
    "sensor_delay_steps",
    "action_delay_steps",
    "camera_age_ms",
    "gyro_age_ms",
    "attitude_age_ms",
    "command_delay_ms",
    "initial_target",
    "initial_radius",
    "target_velocity",
    "domain_mass_scale",
    "domain_thrust_scale",
    "domain_response_scale",
)


class _GifRecorder:
    """Keep one scene palette with reserved semantic colors across all GIF frames."""

    def __init__(self, path: Path, durations: np.ndarray) -> None:
        self.path, self.durations = path, durations
        self.palette = None
        self.frames: list[Any] = []

    def append_data(self, rgb: np.ndarray) -> None:
        """Quantize the original RGB frame before GIF's automatic palette conversion."""
        from PIL import Image

        image = Image.fromarray(np.ascontiguousarray(rgb))
        if self.palette is None:
            scene = image.quantize(colors=240, method=Image.Quantize.MEDIANCUT)
            self.palette = Image.new("P", (1, 1))
            colors = [value for color in GIF_SEMANTIC_COLORS for value in color]
            self.palette.putpalette(colors + scene.getpalette()[: 240 * 3])
        self.frames.append(image.quantize(palette=self.palette, dither=Image.Dither.NONE))

    def close(self) -> None:
        """Preserve the existing 10 ms frame durations, including the final hold."""
        if not self.frames:
            return
        self.frames[0].save(
            self.path,
            save_all=True,
            append_images=self.frames[1:],
            duration=self.durations[: len(self.frames)].tolist(),
            loop=0,
            optimize=False,
            palette=self.palette.getpalette(),
        )
        self.frames.clear()


@dataclass
class FrozenReplay:
    """Saved trajectories and a small, deterministic playback state."""

    cfg: Config
    arrays: dict[str, dict[str, np.ndarray]]
    scores: dict[str, dict]
    seed: int
    checkpoint: Path | None = None
    plant: str = "crazyflow"
    suite: int = 0
    episode: int = 0
    elapsed: float = 0.0
    paused: bool = False
    speed: float = 1.0

    @classmethod
    def load(cls, directory: Path) -> FrozenReplay:
        """Load the exact checkpoint calibration and all stored flight outcomes."""
        report = json.loads((directory / "metrics.json").read_text())
        cfg = Config(**report["checkpoint_metadata"]["config"])
        arrays = {}
        for name in SUITES:
            with np.load(directory / f"{name}.npz", allow_pickle=False) as data:
                arrays[name] = {key: np.asarray(data[key]) for key in REPLAY_ARRAYS if key in data}
        # Older paired evaluation files contain enough seed information to recover
        # the same sampled FIFO delays. Never infer them from simulated distance.
        if report.get("paired_scenarios") and any(
            "executed_commands" not in value for value in arrays.values()
        ):
            import jax

            _, rollout_key = jax.random.split(jax.random.PRNGKey(report["seed"]))
            for name in SUITES:
                arrays[name].update(
                    control_telemetry(arrays[name], cfg, rollout_key, noise=name != "nominal")
                )
        candidates = [Path(report["checkpoint"]), directory.parent / "policy.npz"]
        checkpoint = next((p for p in candidates if p.is_file()), None)
        return cls(
            cfg,
            arrays,
            report["scenarios"],
            report["seed"],
            checkpoint,
            plant=report.get("plant", "crazyflow"),
        )

    @property
    def name(self) -> str:
        """Current frozen suite."""
        return SUITES[self.suite]

    @property
    def current(self) -> dict[str, np.ndarray]:
        """Current trajectory arrays."""
        return self.arrays[self.name]

    @property
    def record(self) -> dict:
        """First-event outcome of the selected episode."""
        return self.scores[self.name]["per_episode"][self.episode]

    @property
    def duration(self) -> float:
        """Physical first-event time, excluding all post-event rollout states."""
        return float(self.record["event_time_s"])

    @property
    def target(self) -> np.ndarray:
        """Current balloon center at the same replay time as the displayed aircraft."""
        return self.target_at()

    @property
    def initial_target(self) -> np.ndarray:
        """Initial world center; retained independently of replay time."""
        return self.current["initial_target"][self.episode]

    @property
    def target_velocity(self) -> np.ndarray:
        """Prescribed world velocity; older static evaluation arrays imply zero."""
        velocities = self.current.get("target_velocity")
        return np.zeros(3) if velocities is None else velocities[self.episode]

    def target_at(self, time_s: float | None = None) -> np.ndarray:
        """Use absolute episode time, avoiding cumulative integration drift."""
        when = min(self.elapsed, self.duration) if time_s is None else time_s
        return self.initial_target + self.target_velocity * when

    @property
    def radius(self) -> float:
        """Current balloon radius."""
        return float(self.current["initial_radius"][self.episode])

    def state(self) -> np.ndarray:
        """Interpolate a recorded state at the current playback time."""
        states = self.current["substep_states"][:, self.episode]
        value = min(self.elapsed, self.duration) * self.cfg.substeps / self.cfg.dt
        index = min(int(value), len(states) - 1)
        fraction = value - index
        state = states[index] * (1 - fraction) + states[min(index + 1, len(states) - 1)] * fraction
        return state if np.isfinite(state).all() else states[max(0, index - 1)]

    def reset(self) -> None:
        """Restart the same flight."""
        self.elapsed, self.paused = 0.0, False

    def select_suite(self, suite: int) -> None:
        """Select a suite while preserving the selected episode where possible."""
        self.suite = suite % len(SUITES)
        self.episode %= self.current["initial_radius"].shape[0]
        self.reset()

    def next_episode(self) -> None:
        """Select the next saved flight."""
        self.episode = (self.episode + 1) % self.current["initial_radius"].shape[0]
        self.reset()

    def scrub(self, fraction: float) -> None:
        """Pause at a selected fraction of the first episode."""
        self.elapsed = float(np.clip(fraction, 0, 1)) * self.duration
        self.paused = True

    def telemetry(self) -> dict[str, Any]:
        """Return aligned control-tick inputs/outputs and separately labelled state truth.

        Commands and boxes at k are used during [k*dt,(k+1)*dt). Raw plant samples
        include the initial state, so gyro/RPM/state k is the boundary before tick k.
        The displayed truth is the latest saved 50 Hz sample, not the next boundary.
        """
        tick = int(np.floor(min(self.elapsed, self.duration) / self.cfg.dt + 1e-8))
        command_tick = min(tick, len(self.current["actions"]) - 1)
        boxes = self.current["bbox"][:, self.episode]
        received = boxes[command_tick]
        present = bool(received[4] > 0.5)
        missing_since = command_tick
        if not present:
            while missing_since > 0 and boxes[missing_since - 1, 4] <= 0.5:
                missing_since -= 1
        unseen = 0.0 if present else max(0.0, self.elapsed - missing_since * self.cfg.dt)
        states = self.current.get("states")
        if states is None:
            states = self.current["substep_states"][:: self.cfg.substeps]
        truth_tick = min(tick, len(states) - 1)

        def episode_sample(name: str, sample_tick: int | None = None) -> Any:
            values = self.current.get(name)
            if values is None:
                return None
            value = (
                values[self.episode] if sample_tick is None else values[sample_tick, self.episode]
            )
            return np.asarray(value).tolist()

        return {
            "control_tick": command_tick,
            "control_time_s": command_tick * self.cfg.dt,
            "received_bbox": received.tolist(),
            "received_valid": present,
            "unseen_seconds": unseen,
            "tracking_timeout_s": self.cfg.max_unseen_seconds,
            "issued_action": self.current["actions"][command_tick, self.episode].tolist(),
            "issued_command": episode_sample("issued_commands", command_tick),
            "executed_command": episode_sample("executed_commands", command_tick),
            "sensor_delay_steps": episode_sample("sensor_delay_steps"),
            "action_delay_steps": episode_sample("action_delay_steps"),
            "camera_age_ms": episode_sample("camera_age_ms", command_tick),
            "gyro_age_ms": episode_sample("gyro_age_ms", command_tick),
            "attitude_age_ms": episode_sample("attitude_age_ms", command_tick),
            "command_delay_ms": episode_sample("command_delay_ms", command_tick),
            "truth_time_s": truth_tick * self.cfg.dt,
            "truth_rpy_rad": states[truth_tick, self.episode, 3:6].tolist(),
            "truth_body_gyro_rad_s": episode_sample("body_gyro", truth_tick),
            "truth_rotor_rpm": episode_sample("rotor_rpm", truth_tick),
            "truth_mass_scale": episode_sample("domain_mass_scale"),
            "truth_thrust_scale": episode_sample("domain_thrust_scale"),
            "truth_response_scale": episode_sample("domain_response_scale"),
            "truth_target_position_m": self.target_at(truth_tick * self.cfg.dt).tolist(),
            "truth_target_velocity_m_s": self.target_velocity.tolist(),
            "truth_target_diameter_m": 2 * self.radius,
            "first_event_time_s": self.duration,
            "first_event_outcome": self.record["outcome"],
        }


def generate(
    checkpoint: Path, seed: int, device: str = "auto", plant: str = "crazyflow"
) -> FrozenReplay:
    """Generate new frozen flights; training and optimizer state are never used."""
    import jax

    if device != "auto":
        jax.config.update("jax_platforms", "cuda" if device == "gpu" else "cpu")
    from crazyflow.balloon.model import sample_batch
    from crazyflow.balloon.policy import load_checkpoint
    from crazyflow.balloon.rollout import rollout

    if plant == "mujoco":
        from crazyflow.balloon.mujoco_rollout import rollout

    weights, cfg, _ = load_checkpoint(checkpoint)
    batch_key, rollout_key = jax.random.split(jax.random.PRNGKey(seed))
    arrays, scores = {}, {}
    for name in SUITES:
        batch = sample_batch(batch_key, cfg, 1, scenario=name)
        result = jax.device_get(rollout(weights, batch, cfg, rollout_key, noise=name != "nominal"))
        batch = jax.device_get(batch)
        arrays[name] = {key: np.asarray(result[key]) for key in REPLAY_ARRAYS if key in result}
        arrays[name].update(control_telemetry(result, cfg, rollout_key, noise=name != "nominal"))
        arrays[name].update(initial_target=batch["target"], initial_radius=batch["radius"])
        arrays[name]["target_velocity"] = np.asarray(
            batch.get("target_velocity", np.zeros_like(batch["target"]))
        )
        arrays[name].update({f"domain_{key}": value for key, value in batch["domain"].items()})
        scores[name] = summarize_trajectories(
            result["substep_states"],
            batch["target"],
            batch["radius"],
            dt=cfg.dt / cfg.substeps,
            target_velocities=batch.get("target_velocity"),
            drone_radius=cfg.drone_radius,
            collision_offset=cfg.collision_offset,
            min_altitude=cfg.min_altitude,
            max_range=cfg.max_range,
            max_impact_speed=cfg.max_impact_speed,
            visible=np.asarray(result["bbox"])[..., 4] > 0.5,
            visibility_dt=cfg.dt,
            tracking_timeout=cfg.max_unseen_seconds,
            objective=float(result["loss"]),
        )
    return FrozenReplay(cfg, arrays, scores, seed, checkpoint, plant=plant)


def _generate_in_process(checkpoint: str, seed: int, device: str, plant: str, sender: Any) -> None:
    """Spawn a separate JAX runtime; send host arrays before the parent reaps us."""
    try:
        sender.send(("ok", generate(Path(checkpoint), seed, device, plant)))
    except Exception as error:
        sender.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        sender.close()


class AsyncFlightGenerator:
    """Generate one replacement flight outside the GLFW process and callbacks.

    Spawn, rather than fork, isolates a fresh JAX/CUDA runtime. Only the main
    process polls and replaces the displayed replay. Injectable unit-test
    functions run in a controlled daemon thread and do not create a JAX runtime.
    """

    def __init__(self, generate_fn: Callable[..., FrozenReplay] | None = None) -> None:
        self.generate_fn = generate_fn
        self.future: Future[FrozenReplay] | None = None
        self.requested_seed: int | None = None
        self.error: str | None = None
        self.closed = False
        self._process: Any = None
        self._receiver: Any = None
        self._retired: list[Any] = []

    def _reap(self) -> None:
        """Reap finished workers without waiting for their interpreter cleanup."""
        alive = []
        for process in self._retired:
            if process.is_alive():
                alive.append(process)
            else:
                process.join()
                process.close()
        self._retired = alive

    @property
    def busy(self) -> bool:
        """Keep reset disabled until the main thread consumes the result."""
        return self.future is not None

    def request(self, replay: FrozenReplay) -> bool:
        """Start one new seed; duplicate requests leave the current job untouched."""
        if self.busy or self.closed:
            return False
        self._reap()
        if replay.checkpoint is None:
            self.error = "Random reset needs the original policy.npz; reopen with --checkpoint."
            return False
        checkpoint, plant = replay.checkpoint, replay.plant
        seed = (replay.seed + 1) % (2**32)
        pending: Future[FrozenReplay] = Future()
        self.future, self.requested_seed, self.error = pending, seed, None

        if self.generate_fn is None:
            import jax

            device = "gpu" if jax.default_backend() == "gpu" else "cpu"
            context = multiprocessing.get_context("spawn")
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_generate_in_process,
                args=(str(checkpoint), seed, device, plant, sender),
                name="balloon-random-reset",
            )
            try:
                process.start()
            except Exception as error:
                receiver.close()
                sender.close()
                self.future = None
                self.error = f"Random reset failed ({type(error).__name__}): {error}"
                return False
            sender.close()
            pending.set_running_or_notify_cancel()
            self._process, self._receiver = process, receiver
            return True

        def work() -> None:
            if not pending.set_running_or_notify_cancel():
                return
            try:
                pending.set_result(self.generate_fn(checkpoint, seed, plant=plant))
            except Exception as error:
                pending.set_exception(error)

        try:
            Thread(target=work, name="balloon-random-reset", daemon=True).start()
        except Exception as error:
            self.future = None
            self.error = f"Random reset failed ({type(error).__name__}): {error}"
            return False
        return True

    def poll(self) -> FrozenReplay | None:
        """Return a completed replacement or retain a visible, recoverable error."""
        self._reap()
        if self._process is not None:
            if self._receiver.poll():
                try:
                    status, result = self._receiver.recv()
                    if status == "ok":
                        self.future.set_result(result)
                    else:
                        self.future.set_exception(RuntimeError(result))
                except (EOFError, OSError) as error:
                    self.future.set_exception(RuntimeError(f"Worker connection closed: {error}"))
            elif not self._process.is_alive():
                self.future.set_exception(
                    RuntimeError(f"Generation worker exited with code {self._process.exitcode}")
                )
            if self.future.done():
                self._receiver.close()
                self._retired.append(self._process)
                self._process = self._receiver = None
        if self.future is None or not self.future.done():
            return None
        finished, self.future = self.future, None
        try:
            return finished.result()
        except Exception as error:
            self.error = f"Random reset failed ({type(error).__name__}): {error}"
            return None

    def close(self) -> None:
        """Terminate isolated simulation workers instead of aborting an active JAX thread."""
        self.closed = True
        workers = self._retired + ([self._process] if self._process is not None else [])
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join(timeout=0.2)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.2)
            if not process.is_alive():
                process.close()
        self._retired, self._process = [], None
        if self._receiver is not None:
            self._receiver.close()
            self._receiver = None
        if self.future is not None:
            self.future.cancel()


def _add_camera_geometry(scene: Any, state: np.ndarray, cfg: Config) -> None:
    """Draw the calibrated frustum and camera origin in the world viewport only."""
    import mujoco

    center, _ = camera_pose(state[:3], state[3:6], cfg)
    corners = frustum_corners(state[:3], state[3:6], cfg, depth=0.25)
    color = np.array([0.15, 0.9, 1, 1], dtype=np.float32)
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE, [0.008] * 3, center, np.eye(3).ravel(), color
    )
    scene.ngeom += 1
    for i in range(4):
        for start, end in ((center, corners[i]), (corners[i], corners[(i + 1) % 4])):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), color
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1.5, start, end)
            scene.ngeom += 1


def _draw_box(viewport: Any, box: np.ndarray, color: tuple, *, dashed: bool = False) -> None:
    """Draw a received box without changing its stored coordinates or validity."""
    import mujoco

    if box[4] <= 0.5:
        return
    cx, cy, bw, bh = box[:4]
    # Clip only drawing coordinates to this viewport, never the actor's saved box.
    lo = np.clip([cx - bw / 2, cy - bh / 2], -1, 1)
    hi = np.clip([cx + bw / 2, cy + bh / 2], -1, 1)
    x0 = viewport.left + int((lo[0] + 1) * viewport.width / 2)
    x1 = viewport.left + int((hi[0] + 1) * viewport.width / 2)
    y0 = viewport.bottom + int((1 - hi[1]) * viewport.height / 2)
    y1 = viewport.bottom + int((1 - lo[1]) * viewport.height / 2)
    if x1 <= x0 or y1 <= y0:
        return
    stride = 12 if dashed else max(viewport.width, viewport.height) + 1
    stroke = 7 if dashed else stride
    for x in range(x0, x1, stride):
        for y in (y0, max(y1 - 2, y0)):
            mujoco.mjr_rectangle(mujoco.MjrRect(x, y, min(stroke, x1 - x), 2), *color, 1)
    for y in range(y0, y1, stride):
        for x in (x0, max(x1 - 2, x0)):
            mujoco.mjr_rectangle(mujoco.MjrRect(x, y, 2, min(stroke, y1 - y)), *color, 1)


def debug_panel(replay: FrozenReplay, telemetry: dict, *, ideal_visible: bool) -> str:
    """Compact labels keep policy observations separate from simulator-only diagnostics."""

    def vector(values: list | None, *, scale: float = 1, digits: int = 2) -> str:
        if values is None:
            return "unavailable"
        return "/".join(f"{value * scale:.{digits}f}" for value in values)

    def command_line(name: str, values: list | None) -> str:
        if values is None:
            return f"{name}: unavailable"
        angles = vector(values[:3], scale=180 / np.pi, digits=1)
        return f"{name} RPY[deg] {angles} | T {values[3]:.3f} N"

    def delay(name: str) -> str:
        steps = telemetry[name]
        return "n/a" if steps is None else f"{steps * replay.cfg.dt * 1000:.0f}ms"

    seen = "VALID" if telemetry["received_valid"] else "LOST"
    fov = "visible" if ideal_visible else "empty"
    if telemetry.get("command_delay_ms") is not None:
        timing = (
            f"age camera/gyro/attitude [ms] {telemetry['camera_age_ms']:.0f} / "
            f"{telemetry['gyro_age_ms']:.0f} / {telemetry['attitude_age_ms']:.0f}\n"
            f"Scheduled RPY/T delay [ms]: {vector(telemetry['command_delay_ms'], digits=0)}"
        )
    else:
        timing = f"sensor {delay('sensor_delay_steps')} / action {delay('action_delay_steps')}"
    executed_label = "Executed@tick start" if replay.cfg.hardware.enabled else "Executed"
    return (
        f"CTRL k{telemetry['control_tick']} @{telemetry['control_time_s']:.2f}s | "
        f"{timing}\n"
        f"Policy box {seen} | unseen {telemetry['unseen_seconds']:.2f}/"
        f"{telemetry['tracking_timeout_s']:.2f}s | ideal FOV {fov}\n"
        f"{command_line('Issued', telemetry['issued_command'])}\n"
        f"{command_line(executed_label, telemetry['executed_command'])}\n"
        "SIM TRUTH (NOT actor inputs) RPY[deg]: "
        f"{vector(telemetry['truth_rpy_rad'], scale=180 / np.pi, digits=1)}\n"
        f"Body gyro[rad/s]: {vector(telemetry['truth_body_gyro_rad_s'])}\n"
        f"Rotor RPM: {vector(telemetry['truth_rotor_rpm'], digits=0)}\n"
        f"Recorded first event: {telemetry['first_event_outcome']} "
        f"@ {telemetry['first_event_time_s']:.3f}s\n"
        f"TARGET SIM TRUTH: D {2 * replay.radius:.3f}m | "
        f"{'moving' if np.any(replay.target_velocity != 0) else 'static'} "
        f"{np.linalg.norm(replay.target_velocity):.2f}m/s | {replay.speed:g}x"
    )


def launch(
    replay: FrozenReplay,
    *,
    frames: int = 0,
    screenshot: Path | None = None,
    self_test: bool = False,
    record: Path | None = None,
    fps: int = 20,
) -> None:
    """Open native world, XY map and FPV views, optionally exiting after validation."""
    import glfw
    import jax
    import jax.numpy as jnp
    import mujoco

    from crazyflow.balloon.observations import ideal_bbox

    if self_test and replay.checkpoint is None:
        raise ValueError("--self-test needs an accessible policy.npz to verify random reset")

    project = jax.jit(
        lambda state, target, radius: ideal_bbox(state[:3], state[3:6], target, radius, replay.cfg)
    )

    if not glfw.init():
        raise RuntimeError("GLFW needs a display. Use DISPLAY/X11 or xvfb-run for a smoke test.")
    window = glfw.create_window(
        1440, 960, "Crazyflow | world + calibrated FPV | frozen replay", None, None
    )
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create the GLFW OpenGL window")
    glfw.make_context_current(window)
    glfw.set_window_size_limits(window, 960, 720, glfw.DONT_CARE, glfw.DONT_CARE)
    glfw.swap_interval(1)
    from OpenGL import GL

    gl_info = {
        name: (GL.glGetString(code) or b"unknown").decode(errors="replace")
        for name, code in (
            ("vendor", GL.GL_VENDOR),
            ("renderer", GL.GL_RENDERER),
            ("version", GL.GL_VERSION),
        )
    }
    print("OpenGL: " + " | ".join(f"{name}={value}" for name, value in gl_info.items()), flush=True)
    model = make_replay_model(replay.cfg, replay.target, replay.radius)
    data = mujoco.MjData(model)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
    world_scene, fpv_scene = (mujoco.MjvScene(model, maxgeom=2000) for _ in range(2))
    options = mujoco.MjvOption()
    world_camera, fpv_camera = mujoco.MjvCamera(), mujoco.MjvCamera()
    world_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    world_camera.azimuth, world_camera.elevation = 135, -25
    world_camera.distance = max(
        2.5, float(np.linalg.norm(replay.target - replay.state()[:3])) * 1.3
    )
    world_camera.lookat[:] = (replay.target + replay.state()[:3]) / 2
    fpv_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    fpv_camera.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv")
    balloon_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "balloon")
    balloon_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "balloon_surface")
    ui = {
        "buttons": [],
        "timeline": (24, 26, 1392, 16),
        "drag": False,
        "cursor": (0.0, 0.0),
        "camera_buttons": set(),
        "follow": True,
        "test_cursor": None,
        "telemetry": True,
        "layout": layout(1440, 960, replay.cfg.image_width / replay.cfg.image_height),
    }
    map_key, map_bounds, xy_map_info = None, None, {}
    generator = AsyncFlightGenerator()

    def text(rect: Any, message: str) -> None:
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            rect,
            message,
            "",
            context,
        )

    def action(name: str) -> None:
        if record is not None and name != "Debug view":
            return
        if name == "Pause":
            replay.paused = not replay.paused
        elif name == "Reset":
            generator.request(replay)
        elif name == "Replay":
            replay.reset()
        elif name == "Next flight":
            if replay.current["initial_radius"].shape[0] == 1:
                generator.request(replay)
            else:
                replay.next_episode()
        elif name == "Debug view":
            ui["telemetry"] = not ui["telemetry"]
        elif name in SUITES:
            replay.select_suite(SUITES.index(name))

    def key_callback(_window: Any, key: int, _scan: int, pressed: int, _mods: int) -> None:
        if pressed not in (glfw.PRESS, glfw.REPEAT):
            return
        if record is not None and key not in (glfw.KEY_ESCAPE, glfw.KEY_T, glfw.KEY_F):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            action("Pause")
        elif key == glfw.KEY_R:
            action("Reset")
        elif key == glfw.KEY_N:
            action("Reset")
        elif key == glfw.KEY_BACKSPACE:
            action("Replay")
        elif key == glfw.KEY_E:
            action("Next flight")
        elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3):
            replay.select_suite(key - glfw.KEY_1)
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            direction = 1 if key == glfw.KEY_RIGHT else -1
            replay.scrub((replay.elapsed + direction * 0.02) / max(replay.duration, 1e-9))
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            replay.speed = min(4.0, replay.speed * 2)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            replay.speed = max(0.125, replay.speed / 2)
        elif key == glfw.KEY_F:
            ui["follow"] = not ui["follow"]
        elif key == glfw.KEY_T:
            action("Debug view")

    def cursor_pixels() -> tuple[float, float]:
        if ui["test_cursor"] is not None:
            return ui["test_cursor"]
        x, y = glfw.get_cursor_pos(window)
        w, h = glfw.get_window_size(window)
        fw, fh = glfw.get_framebuffer_size(window)
        return x * fw / max(w, 1), fh - y * fh / max(h, 1)

    def mouse_callback(_window: Any, button: int, pressed: int, _mods: int) -> None:
        x, y = cursor_pixels()
        ui["cursor"] = (x, y)
        if pressed == glfw.RELEASE:
            ui["camera_buttons"].discard(button)
        elif pressed == glfw.PRESS:
            if ui["layout"].world.contains(x, y):
                ui["camera_buttons"].add(button)
        if record is not None:
            return
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        ui["drag"] = False
        if pressed == glfw.PRESS:
            tx, ty, tw, th = ui["timeline"]
            if tx <= x <= tx + tw and ty - 10 <= y <= ty + th + 10:
                ui["drag"] = True
                replay.scrub((x - tx) / tw)
            for name, rect in ui["buttons"]:
                if (
                    rect.left <= x <= rect.left + rect.width
                    and rect.bottom <= y <= rect.bottom + rect.height
                ):
                    action(name)

    def motion_callback(_window: Any, _x: float, _y: float) -> None:
        x, y = cursor_pixels()
        last_x, last_y = ui["cursor"]
        ui["cursor"] = (x, y)
        if ui["drag"]:
            tx, _, tw, _ = ui["timeline"]
            replay.scrub((x - tx) / tw)
        else:
            if not ui["layout"].world.contains(x, y):
                return
            if glfw.MOUSE_BUTTON_LEFT in ui["camera_buttons"]:
                mujoco.mjv_moveCamera(
                    model,
                    mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                    (x - last_x) / 900,
                    -(y - last_y) / 900,
                    world_camera,
                )
            elif glfw.MOUSE_BUTTON_RIGHT in ui["camera_buttons"]:
                ui["follow"] = False
                mujoco.mjv_moveCamera(
                    model,
                    mujoco.mjtMouse.mjMOUSE_MOVE_V,
                    (x - last_x) / 900,
                    -(y - last_y) / 900,
                    world_camera,
                )

    def scroll_callback(_window: Any, _x: float, y: float) -> None:
        if not ui["layout"].world.contains(*cursor_pixels()):
            return
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y, world_camera)

    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_callback)
    glfw.set_cursor_pos_callback(window, motion_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    control_checks, debug_checks = [], []
    test = {"stage": 0, "reset_started": None}

    def check_replay_exact() -> None:
        seed, target = replay.seed, replay.initial_target.copy()
        velocity, radius = replay.target_velocity.copy(), replay.radius
        states = replay.current["substep_states"][:, replay.episode].copy()
        replay.scrub(0.5)
        key_callback(window, glfw.KEY_BACKSPACE, 0, glfw.PRESS, 0)
        assert not replay.paused and replay.elapsed == 0 and replay.seed == seed
        np.testing.assert_array_equal(replay.target, target)
        np.testing.assert_array_equal(replay.target_velocity, velocity)
        assert replay.radius == radius
        np.testing.assert_array_equal(replay.current["substep_states"][:, replay.episode], states)

    def exercise_controls() -> None:
        stage = test["stage"]
        if stage == 0:
            assert ui["follow"]
            key_callback(window, glfw.KEY_F, 0, glfw.PRESS, 0)
            assert not ui["follow"]
            key_callback(window, glfw.KEY_F, 0, glfw.PRESS, 0)
            assert ui["follow"]
            control_checks.append("follow_toggle")
            key_callback(window, glfw.KEY_T, 0, glfw.PRESS, 0)
            assert not ui["telemetry"]
            debug_checks.append("calibration_panel")
            key_callback(window, glfw.KEY_T, 0, glfw.PRESS, 0)
            assert ui["telemetry"]
            debug_checks.append("telemetry_panel")
            key_callback(window, glfw.KEY_SPACE, 0, glfw.PRESS, 0)
            assert replay.paused
            control_checks.append("space_pause")
            glfw.set_window_size(window, 1100, 800)
        elif stage == 1:
            assert glfw.get_window_size(window) == (1100, 800)
            control_checks.append("xy_map_resize")
            check_replay_exact()
            control_checks.append("replay_exact")
            glfw.set_window_size(window, 1440, 960)
        elif stage == 2:
            # Exercise actual button -> cursor-motion -> MuJoCo callbacks, including
            # the MuJoCo 4 five-argument API. Synthetic cursor coordinates avoid
            # desktop pointer clamping under Xvfb or a small physical display.
            assert ui["follow"]
            world = ui["layout"].world
            center = (world.left + world.width / 2, world.bottom + world.height / 2)
            for button, label in (
                (glfw.MOUSE_BUTTON_LEFT, "mouse_orbit"),
                (glfw.MOUSE_BUTTON_RIGHT, "mouse_pan"),
            ):
                before = np.r_[world_camera.azimuth, world_camera.elevation, world_camera.lookat]
                ui["test_cursor"] = center
                mouse_callback(window, button, glfw.PRESS, 0)
                ui["test_cursor"] = (center[0] + 45, center[1] + 25)
                motion_callback(window, 0, 0)
                mouse_callback(window, button, glfw.RELEASE, 0)
                after = np.r_[world_camera.azimuth, world_camera.elevation, world_camera.lookat]
                assert not np.allclose(before, after), label
                assert ui["follow"] == (button == glfw.MOUSE_BUTTON_LEFT)
                control_checks.append(label)
            key_callback(window, glfw.KEY_F, 0, glfw.PRESS, 0)
            assert ui["follow"]
            before = world_camera.distance
            scroll_callback(window, 0, 1)
            assert not np.isclose(before, world_camera.distance)
            control_checks.append("mouse_scroll")
            camera_before = np.r_[
                world_camera.azimuth,
                world_camera.elevation,
                world_camera.distance,
                world_camera.lookat,
            ]
            area = ui["layout"].map
            for button in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
                ui["test_cursor"] = (area.left + area.width / 2, area.bottom + area.height / 2)
                mouse_callback(window, button, glfw.PRESS, 0)
                ui["test_cursor"] = (ui["test_cursor"][0] + 10, ui["test_cursor"][1] + 10)
                motion_callback(window, 0, 0)
                scroll_callback(window, 0, 1)
                mouse_callback(window, button, glfw.RELEASE, 0)
            np.testing.assert_array_equal(
                camera_before,
                np.r_[
                    world_camera.azimuth,
                    world_camera.elevation,
                    world_camera.distance,
                    world_camera.lookat,
                ],
            )
            control_checks.append("xy_map_mouse_isolated")
            ui["test_cursor"] = None
            key_callback(window, glfw.KEY_2, 0, glfw.PRESS, 0)
            assert replay.name == "noisy"
            control_checks.append("suite_noisy")
        elif stage == 3:
            test["noisy_map_bounds"] = xy_map_info["bounds_m"]
            tx, ty, tw, th = ui["timeline"]
            # Inject framebuffer coordinates into the real click callback. OS pointer
            # warping may be clamped when the test window exceeds the desktop size.
            ui["test_cursor"] = (tx + tw / 2, ty + th / 2)
            mouse_callback(window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
            mouse_callback(window, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0)
            ui["test_cursor"] = None
            assert replay.paused and np.isclose(replay.elapsed, replay.duration / 2)
            control_checks.append("mouse_timeline_scrub")
        elif stage == 4:
            np.testing.assert_array_equal(xy_map_info["bounds_m"], test["noisy_map_bounds"])
            np.testing.assert_allclose(xy_map_info["target_position_m"], replay.target)
            assert np.isclose(xy_map_info["time_s"], min(replay.elapsed, replay.duration))
            control_checks.append("xy_map_time_and_fixed_bounds")
            np.testing.assert_allclose(
                world_camera.lookat, (replay.state()[:3] + replay.target) / 2, atol=1e-7
            )
            key_callback(window, glfw.KEY_3, 0, glfw.PRESS, 0)
            assert replay.name == "perturbed"
            control_checks.append("suite_perturbed")
        elif stage == 5:
            test.update(seed=replay.seed, target=replay.target.copy(), arrays=replay.arrays)
            button = next(rect for name, rect in ui["buttons"] if name == "Reset")
            ui["test_cursor"] = (button.left + button.width / 2, button.bottom + button.height / 2)
            mouse_callback(window, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
            mouse_callback(window, glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE, 0)
            ui["test_cursor"] = None
            assert generator.busy, generator.error
            pending = generator.future
            key_callback(window, glfw.KEY_R, 0, glfw.PRESS, 0)
            assert generator.future is pending
            assert replay.arrays is test["arrays"] and replay.seed == test["seed"]
            np.testing.assert_array_equal(replay.target, test["target"])
            control_checks.extend(("random_reset_started", "reset_keeps_current_until_ready"))
            test["reset_started"] = time.monotonic()
        elif stage == 6:
            assert generator.error is None, generator.error
            if generator.busy:
                assert time.monotonic() - test["reset_started"] < 180, "Random reset timed out"
                return
            assert replay.seed == (test["seed"] + 1) % (2**32)
            assert replay.arrays is not test["arrays"]
            assert not np.array_equal(replay.target, test["target"])
            control_checks.extend(("random_reset_seed", "random_reset_target_changed"))
        elif stage == 7:
            key_callback(window, glfw.KEY_1, 0, glfw.PRESS, 0)
            assert replay.name == "nominal"
            control_checks.append("suite_nominal")
        elif stage == 8:
            key_callback(window, glfw.KEY_RIGHT, 0, glfw.PRESS, 0)
            assert replay.paused
            control_checks.append("keyboard_scrub")
        elif stage == 9:
            np.testing.assert_allclose(
                world_camera.lookat, (replay.state()[:3] + replay.target) / 2, atol=1e-7
            )
            control_checks.append("follow_moving_midpoint")
            check_replay_exact()
            control_checks.append("new_flight_replay_exact")
        test["stage"] += 1

    frame_count, last_time = 0, time.monotonic()
    recording = None
    if record is not None:
        record.parent.mkdir(parents=True, exist_ok=True)
        frames = frames or int(np.ceil(replay.duration * fps / replay.speed)) + fps
        # GIF has a 10 ms timebase. Distribute rounding so e.g. 30 FPS stays real time.
        durations = (np.diff(np.rint(np.arange(frames + 1) * 100 / fps)) * 10).astype(int)
        recording = _GifRecorder(record, durations)
    try:
        while not glfw.window_should_close(window):
            now = time.monotonic()
            controls_complete = not self_test or test["stage"] >= 10
            fresh = generator.poll()
            if fresh is not None:
                if fresh.cfg.to_dict() != replay.cfg.to_dict():
                    generator.error = "Checkpoint configuration changed; reopen the player."
                else:
                    fresh.suite, fresh.speed = replay.suite, replay.speed
                    replay = fresh
                    last_time = now
            if generator.busy:
                status = (
                    f"Generating seed {generator.requested_seed} in background; "
                    "current flight stays available."
                )
            elif generator.error:
                status = "ERROR: " + " ".join(generator.error.split())[:150]
            else:
                status = (
                    "Random reset [R]: new balloon/initial state/noise | "
                    "Replay [Backspace]: same flight"
                )
            glfw.set_window_title(window, "Crazyflow | " + status)
            if recording is not None:
                replay.elapsed = min(replay.duration, frame_count * replay.speed / fps)
                replay.paused = replay.elapsed >= replay.duration
            elif not replay.paused:
                replay.elapsed = min(
                    replay.duration, replay.elapsed + min(now - last_time, 0.1) * replay.speed
                )
                if replay.elapsed >= replay.duration:
                    replay.paused = True
            last_time = now
            state = replay.state()
            model.body_pos[balloon_body] = replay.target
            model.geom_size[balloon_geom, 0] = replay.radius
            model.geom_rbound[balloon_geom] = replay.radius
            data.mocap_pos[0], data.mocap_quat[0] = state[:3], _quaternion(state[3:6])
            mujoco.mj_forward(model, data)
            w, h = glfw.get_framebuffer_size(window)
            if w < 320 or h < 320:
                glfw.poll_events()
                time.sleep(0.01)
                continue
            views = layout(w, h, replay.cfg.image_width / replay.cfg.image_height)
            ui["layout"] = views
            world_rect = mujoco.MjrRect(*views.world.as_tuple())
            fpv_rect = mujoco.MjrRect(*views.fpv.as_tuple())
            mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, w, h), 0.055, 0.07, 0.09, 1)
            if ui["follow"]:
                world_camera.lookat[:] = (state[:3] + replay.target) / 2
            mujoco.mjv_updateScene(
                model, data, options, None, world_camera, mujoco.mjtCatBit.mjCAT_ALL, world_scene
            )
            _add_camera_geometry(world_scene, state, replay.cfg)
            # A truth-only trail makes the miniature aircraft's motion visible at world scale.
            end = int(replay.elapsed * replay.cfg.substeps / replay.cfg.dt)
            stride = max(10, int(np.ceil(end / 500)))
            path = replay.current["substep_states"][: end + 1 : stride, replay.episode, :3]
            for start, finish in zip(path[:-1], path[1:]):
                geom = world_scene.geoms[world_scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).ravel(),
                    np.array([1.0, 0.8, 0.2, 1.0], dtype=np.float32),
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2, start, finish)
                world_scene.ngeom += 1
            mujoco.mjr_render(world_rect, world_scene, context)
            mujoco.mjv_updateScene(
                model, data, options, None, fpv_camera, mujoco.mjtCatBit.mjCAT_ALL, fpv_scene
            )
            mujoco.mjr_render(fpv_rect, fpv_scene, context)
            current_map_key = (id(replay.arrays), replay.suite, replay.episode)
            if current_map_key != map_key:
                records = replay.current["substep_states"][:, replay.episode]
                map_bounds = episode_bounds(
                    records,
                    replay.initial_target,
                    replay.target_velocity,
                    (len(records) - 1) * replay.cfg.dt / replay.cfg.substeps,
                )
                map_key = current_map_key
            xy_map_info = draw_map(replay, views.map, map_bounds, context, (w, h))
            box, valid = project(
                jnp.asarray(state), jnp.asarray(replay.target), jnp.asarray(replay.radius)
            )
            telemetry = replay.telemetry()
            _draw_box(fpv_rect, np.asarray(box), (0.15, 1.0, 0.3))
            _draw_box(
                fpv_rect, np.asarray(telemetry["received_bbox"]), (1.0, 0.65, 0.05), dashed=True
            )
            text(
                world_rect,
                f"WORLD | {replay.plant} truth replay | {replay.name} | seed {replay.seed}\n"
                "Cyan: camera | Yellow: aircraft path"
                + ("\n" + status if generator.busy or generator.error else ""),
            )
            text(
                fpv_rect,
                "Green solid: current ideal projection (SIM TRUTH)\n"
                "Amber dashed: received policy box (delay/noise/dropout)",
            )
            ui["buttons"] = []
            x = 12
            compact = w < 1280
            buttons = (
                ("Pause", 70 if compact else 100),
                ("Reset", 125 if compact else 180),
                ("Replay", 130 if compact else 200),
                ("Next flight", 100 if compact else 140),
                ("nominal", 80 if compact else 130),
                ("noisy", 75 if compact else 120),
                ("perturbed", 100 if compact else 140),
                ("Debug view", 110 if compact else 150),
            )
            for name, width in buttons:
                rect = mujoco.MjrRect(x, h - 54, width, 40)
                active = name == replay.name
                random_action = name == "Reset" or (
                    name == "Next flight" and replay.current["initial_radius"].shape[0] == 1
                )
                disabled = (random_action and generator.busy) or (
                    record is not None and name != "Debug view"
                )
                mujoco.mjr_rectangle(
                    rect,
                    0.08 if disabled else 0.12,
                    0.12 if disabled else (0.32 if active else 0.17),
                    0.16 if disabled else (0.4 if active else 0.23),
                    1,
                )
                label = "Resume" if name == "Pause" and replay.paused else name
                if name == "Reset":
                    label = (
                        "Generating..."
                        if generator.busy
                        else ("Random [R]" if compact else "Random reset [R]")
                    )
                elif name == "Replay":
                    label = "Replay [Bksp]" if compact else "Replay [Backspace]"
                elif name == "Next flight":
                    label = (
                        "Next [E]"
                        if compact
                        else (
                            "Next random [E]"
                            if replay.current["initial_radius"].shape[0] == 1
                            else "Next flight [E]"
                        )
                    )
                elif name == "Debug view":
                    label = (
                        "View [T]"
                        if compact
                        else ("Calibration [T]" if ui["telemetry"] else "Telemetry [T]")
                    )
                text(rect, label)
                ui["buttons"].append((name, rect))
                x += width + 8
            k, calibration = intrinsics(replay.cfg), parameters(replay.cfg)
            physical = replay.cfg.model_params or {}
            mass = float(physical.get("mass", 0.0434))
            inertia = np.asarray(physical.get("J", np.eye(3) * np.nan))
            inertia_text = "/".join(f"{v:.2e}" for v in np.diag(inertia))
            mount = "/".join(f"{v:.3f}" for v in replay.cfg.camera_offset)
            mount_rpy = "/".join(f"{v:.3f}" for v in replay.cfg.camera_rpy)
            attitude = (replay.cfg.controller_params or {}).get("attitude", {})
            kr = "/".join(f"{v:.0f}" for v in attitude.get("kR", []))
            kw = "/".join(f"{v:.0f}" for v in attitude.get("kw", []))
            panel = (
                f"OV5647 {replay.cfg.image_width}x{replay.cfg.image_height} | "
                f"H {calibration['hfov_deg']:.2f} / V {calibration['vfov_deg']:.2f} deg\n"
                f"fx/fy {k[0, 0]:.3f}/{k[1, 1]:.3f}  cx/cy {k[0, 2]:.1f}/{k[1, 2]:.1f}\n"
                f"Mount xyz [m] {mount} | RPY [rad] {mount_rpy}\n"
                f"Nominal mass {mass * 1000:.2f} g | Jdiag {inertia_text}\n"
                f"Attitude kR {kr}\n"
                f"Body-rate kw {kw}\n"
                f"{replay.plant} rigid body / official Mellinger "
                f"{replay.cfg.substeps / replay.cfg.dt:.0f} Hz\n"
                f"Policy {1 / replay.cfg.dt:.0f} Hz | MuJoCo frozen geometry replay\n"
                f"{replay.name} | seed {replay.seed} | flight {replay.episode + 1}\n"
                f"Outcome {replay.record['outcome']} | "
                f"{replay.elapsed:.3f}/{replay.duration:.3f} s | {replay.speed:g}x\n"
                "Space pause | R/N random reset | Backspace replay\n"
                "E next flight (new random sample when only one is saved)\n"
                "1/2/3 suite | Left/Right scrub | +/- speed\n"
                "F follow | mouse orbit/pan/wheel | Esc close"
            )
            if ui["telemetry"]:
                panel = debug_panel(replay, telemetry, ideal_visible=bool(valid))
            text(mujoco.MjrRect(*views.telemetry.as_tuple()), panel)
            tx, ty, tw, th = 24, 26, max(w - 48, 1), 16
            ui["timeline"] = (tx, ty, tw, th)
            mujoco.mjr_rectangle(mujoco.MjrRect(tx, ty, tw, th), 0.17, 0.22, 0.28, 1)
            progress = int(tw * replay.elapsed / max(replay.duration, 1e-9))
            mujoco.mjr_rectangle(mujoco.MjrRect(tx, ty, max(progress, 1), th), 0.1, 0.75, 0.85, 1)
            text(
                mujoco.MjrRect(20, 45, w - 40, 26),
                "Timeline click/drag | R random reset | Backspace replay | "
                "Mouse orbit/pan/wheel | F follow | Esc close",
            )
            if self_test and not controls_complete:
                exercise_controls()
            frame_count += 1
            finished = frames and frame_count >= frames and controls_complete
            if recording is not None:
                rgb = np.empty((h, w, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, w, h), context)
                recording.append_data(np.flipud(rgb))
            if screenshot is not None and finished:
                import imageio.v2 as imageio

                rgb = np.empty((h, w, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, w, h), context)
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(screenshot, np.flipud(rgb))
                screenshot.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "frames": frame_count,
                            "suite": replay.name,
                            "episode": replay.episode,
                            "seed": replay.seed,
                            "time_s": replay.elapsed,
                            "devices": [str(device) for device in jax.devices()],
                            "opengl": gl_info,
                            "xy_map": xy_map_info,
                            "flight_plant": replay.plant,
                            "camera": calibration,
                            "physical_params": replay.cfg.model_params,
                            "controller_params": replay.cfg.controller_params,
                            "control_checks": control_checks,
                            "debug_checks": debug_checks,
                            "random_reset": {
                                "generating": generator.busy,
                                "requested_seed": generator.requested_seed,
                                "error": generator.error,
                            },
                            "telemetry": telemetry,
                            "target_initial_center_m": replay.initial_target.tolist(),
                            "target_velocity_m_s": replay.target_velocity.tolist(),
                            "target_diameter_m": 2 * replay.radius,
                            "bbox_legend": {
                                "green_solid": "current ideal projection, simulation truth",
                                "amber_dashed": "received policy box held for this control tick",
                            },
                            "role": (
                                "native GLFW calibrated GUI; "
                                f"frozen {replay.plant} flight trajectory replay"
                            ),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            glfw.swap_buffers(window)
            glfw.poll_events()
            if finished:
                break
    finally:
        generator.close()
        if recording is not None:
            recording.close()
            record.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "fps": fps,
                        "gif_palette": "240 adaptive scene colors + 16 fixed semantic colors",
                        "frames": frame_count,
                        "last_sim_time_s": replay.elapsed,
                        "reached_first_event": replay.elapsed >= replay.duration - 1e-9,
                        "truncated": replay.elapsed < replay.duration - 1e-9,
                        "suite": replay.name,
                        "episode": replay.episode,
                        "seed": replay.seed,
                        "flight_plant": replay.plant,
                        "devices": [str(device) for device in jax.devices()],
                        "opengl": gl_info,
                        "xy_map": xy_map_info,
                        "first_event": replay.record,
                        "checkpoint": str(replay.checkpoint),
                        "hardware": replay.cfg.to_dict()["hardware"],
                        "target_initial_center_m": replay.initial_target.tolist(),
                        "target_velocity_m_s": replay.target_velocity.tolist(),
                        "target_diameter_m": 2 * replay.radius,
                        "role": "native GUI framebuffer recording of the saved trajectory",
                    },
                    indent=2,
                )
                + "\n"
            )
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()


def main() -> None:
    """Load saved evaluation flights or generate a fresh checkpoint rollout for the GUI."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--evaluation", type=Path, help="directory containing metrics.json + suite NPZs"
    )
    source.add_argument(
        "--checkpoint", type=Path, help="generate fresh frozen flights from this policy"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument(
        "--plant",
        choices=("crazyflow", "mujoco"),
        default="crazyflow",
        help="flight backend for --checkpoint; --evaluation uses its saved backend",
    )
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument(
        "--frames", type=int, default=0, help="auto-close after N frames; 0 runs until closed"
    )
    parser.add_argument("--screenshot", type=Path, help="save the last frame when --frames is set")
    parser.add_argument("--record", type=Path, help="record the native GUI to a looping GIF")
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per simulated second")
    parser.add_argument("--suite", choices=SUITES, default="nominal")
    parser.add_argument("--episode", type=int, default=0, help="saved flight index, starting at 0")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="exercise mouse, replay and asynchronous random reset (at least 10 frames)",
    )
    args = parser.parse_args()
    if args.frames < 0 or (args.screenshot and not args.frames):
        parser.error("--frames must be nonnegative; --screenshot requires positive --frames")
    if args.self_test and args.frames < 10:
        parser.error("--self-test requires --frames 10 or greater")
    if not 5 <= args.fps <= 60 or args.episode < 0:
        parser.error("--fps must be 5..60 and --episode must be nonnegative")
    if args.record and (args.record.suffix.lower() != ".gif" or args.self_test):
        parser.error("--record requires a .gif path and cannot be used with --self-test")
    # crazyflow imports MuJoCo before this module. Re-exec once to select the native
    # GL backend before those imports, including inside the headless-default image.
    if os.environ.get("MUJOCO_GL") != "glfw":
        env = {**os.environ, "MUJOCO_GL": "glfw"}
        os.execve(
            sys.executable, [sys.executable, "-m", "crazyflow.balloon.play", *sys.argv[1:]], env
        )
    if args.evaluation:
        replay = FrozenReplay.load(args.evaluation)
        print(
            "Loaded saved flight states; the window replays them without advancing physics.",
            flush=True,
        )
    else:
        if args.plant == "mujoco":
            print(
                "Flight generation: native MuJoCo rigid-body integration on CPU; "
                "policy/controller/wrench use the selected JAX device.",
                flush=True,
            )
        else:
            print(
                "Flight generation: Crazyflow rigid-body integration and policy use JAX.",
                flush=True,
            )
        print("Generating initial frozen flights; the GUI opens when ready...", flush=True)
        started = time.monotonic()
        replay = generate(args.checkpoint, args.seed, args.device, args.plant)
        print(
            f"Initial generation: {time.monotonic() - started:.2f} s for {len(SUITES)} suites, "
            f"{len(SUITES) * replay.cfg.steps * replay.cfg.substeps} physics substeps; "
            "includes JIT/cache initialization and complete flights.",
            flush=True,
        )
    import jax

    print(f"JAX devices: {jax.devices()} | saved flight plant: {replay.plant}", flush=True)
    if replay.plant == "mujoco":
        print(
            "MuJoCo native rigid-body physics runs on CPU; GUI rendering uses OpenGL.", flush=True
        )
    replay.select_suite(SUITES.index(args.suite))
    if args.episode >= replay.current["initial_radius"].shape[0]:
        parser.error("--episode exceeds the number of saved flights")
    replay.episode = args.episode
    launch(
        replay,
        frames=args.frames,
        screenshot=args.screenshot,
        self_test=args.self_test,
        record=args.record,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
