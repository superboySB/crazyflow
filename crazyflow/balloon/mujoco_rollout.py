"""Closed-loop native MuJoCo rigid-body evaluation of the frozen BPTT policy.

MuJoCo ``mj_step`` advances every position, orientation and rigid-body velocity.
The unchanged official Mellinger controller and first-principles motor/propeller
model supply the applied wrench; only the four RPM states use their official
explicit Euler update. This is a second rigid-body integrator, not an independent
identification of the real aircraft or an independently fitted actuator model.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax.scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as NumpyRotation

from crazyflow.balloon import hardware as H
from crazyflow.balloon.camera import mujoco_camera_attributes
from crazyflow.balloon.model import Batch, command, initialize, nominal_params, target_at_time
from crazyflow.balloon.observations import actor_features
from crazyflow.balloon.policy import PolicyParams, apply_policy
from crazyflow.control import Control
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.dynamics import Dynamics
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from crazyflow.sim import functional as F
from crazyflow.sim.dynamics import first_principles_dynamics
from crazyflow.sim.sim import build_control_fns, clip_rotor_vel_cmd, increment_steps

if TYPE_CHECKING:
    from crazyflow.balloon.config import Config
    from crazyflow.sim.data import SimData


def _values(value: Any) -> str:
    return " ".join(f"{x:.17g}" for x in np.asarray(value).ravel())


def make_flight_model(
    cfg: Config,
    mass_scale: float = 1.0,
    target: np.ndarray | None = None,
    radius: float = 0.15,
    inertia_scale: float = 1.0,
    target_velocity: np.ndarray | None = None,
) -> mujoco.MjModel:
    """Build a free aircraft with checkpoint mass/full inertia, never visual XML inertia.

    Body origin and center of mass coincide, as in the official equations. The
    displaced task collision sphere changes contact geometry, not the inertia.
    MuJoCo supplies gravity, rigid-body gyroscopic motion and contact response;
    its additional damping, fluid, wind and actuator models are disabled.
    """
    physical = jax.device_get(nominal_params(cfg))
    mass = float(physical["mass"]) * float(mass_scale)
    inertia = np.asarray(physical["J"], dtype=float) * float(inertia_scale)
    if mass <= 0 or not np.isfinite(mass):
        raise ValueError("MuJoCo flight mass must be finite and positive")
    if (
        not np.isfinite(inertia).all()
        or not np.allclose(inertia, inertia.T)
        or np.linalg.eigvalsh(inertia).min() <= 0
    ):
        raise ValueError("MuJoCo flight inertia must be symmetric positive definite")
    root = ET.Element("mujoco", model="cf21B_500_native_flight")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "map", znear=".0001")
    ET.SubElement(
        root,
        "option",
        timestep=_values(cfg.dt / cfg.substeps),
        gravity=_values(physical["gravity_vec"]),
        integrator="Euler",
        density="0",
        viscosity="0",
        wind="0 0 0",
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0", frictionloss="0", armature="0")
    ET.SubElement(default, "geom", friction="0 0 0", margin="0", gap="0")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="20 20 .1")
    drone = ET.SubElement(world, "body", name="drone")
    ET.SubElement(drone, "freejoint", name="flight")
    ET.SubElement(
        drone,
        "inertial",
        pos="0 0 0",
        mass=_values(mass),
        fullinertia=_values(
            [
                inertia[0, 0],
                inertia[1, 1],
                inertia[2, 2],
                inertia[0, 1],
                inertia[0, 2],
                inertia[1, 2],
            ]
        ),
    )
    ET.SubElement(
        drone,
        "geom",
        name="drone_collision",
        type="sphere",
        size=_values(cfg.drone_radius),
        pos=_values(cfg.collision_offset),
    )
    ET.SubElement(drone, "camera", name="fpv", **mujoco_camera_attributes(cfg))
    if target is not None:
        balloon = ET.SubElement(world, "body", name="balloon", pos=_values(target))
        if target_velocity is not None and np.any(np.asarray(target_velocity) != 0):
            # Only the prescribed moving target is mocap; the aircraft remains a
            # free rigid body integrated by mj_step, with its actual mass/inertia.
            balloon.set("mocap", "true")
        ET.SubElement(balloon, "geom", name="balloon_surface", type="sphere", size=_values(radius))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _wrench_and_derivative(data: SimData) -> tuple[Any, jax.Array, jax.Array]:
    derivative = first_principles_dynamics(data)
    params, states = data.params, data.states
    force = params.mass * (derivative.acc - params.gravity_vec)
    angular_momentum = (params.J @ states.ang_vel[..., None])[..., 0]
    # Restore the rigid-body gyro term: MuJoCo already subtracts it internally.
    torque_body = (params.J @ derivative.ang_acc[..., None])[..., 0] + jnp.cross(
        states.ang_vel, angular_momentum
    )
    torque_world = Rotation.from_quat(states.quat).apply(torque_body)
    return derivative, force[:, 0], torque_world[:, 0]


def world_wrench(data: SimData) -> tuple[jax.Array, jax.Array]:
    """Return external world force and torque at the CoM, excluding gravity.

    The official derivative includes thrust/reaction curves, rotor acceleration
    and gyroscopic effects, body drag and world disturbances. Subtracting gravity
    and restoring ``omega cross J omega`` avoids applying either effect twice.
    """
    _, force, torque = _wrench_and_derivative(data)
    return force, torque


class NativeKernels(NamedTuple):
    """Stable JIT identities; episode arrays and policy weights are dynamic inputs."""

    actuators: Callable[..., Any]
    policy_tick: Callable[..., Any]
    hardware_policy_tick: Callable[..., Any]
    hardware_substep: Callable[..., Any]
    after_physics: Callable[..., Any]


def _config_signature(cfg: Config) -> str:
    """Freeze nested dictionaries by value, independent of their insertion order."""

    def array_value(value: Any) -> Any:
        if isinstance(value, (np.ndarray, np.generic, jax.Array)):
            return value.tolist()
        raise TypeError(f"Unsupported configuration value: {type(value).__name__}")

    return json.dumps(
        cfg.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False, default=array_value
    )


def _native_kernels(cfg: Config, noise: bool = False) -> NativeKernels:
    """Reuse process-local kernels; JAX still specializes each array shape/device."""
    return _cached_native_kernels(_config_signature(cfg), noise)


@lru_cache(maxsize=8)
def _cached_native_kernels(signature: str, noise: bool) -> NativeKernels:
    """Capture only immutable configuration, never a policy, batch or native world."""
    from crazyflow.balloon.config import Config
    from crazyflow.balloon.rollout import observe

    cfg = Config(**json.loads(signature))
    physical = nominal_params(cfg)
    stages = build_control_fns(Control.attitude, Dynamics.default)
    upper = motor_force2rotor_vel(physical["thrust_max"], physical["rpm2thrust"])

    def actuator_step(current: SimData) -> tuple[SimData, jax.Array, jax.Array]:
        for _, stage in stages:
            current = stage(current)
        current = clip_rotor_vel_cmd(current, lower=0, upper=upper, dynamics=Dynamics.default)
        derivative, force, torque = _wrench_and_derivative(current)
        current = current.replace(
            states=current.states.replace(
                rotor_vel=current.states.rotor_vel + derivative.rotor_acc * cfg.dt / cfg.substeps
            )
        )
        return increment_steps(current), force, torque

    @jax.jit
    def policy_tick(
        params: PolicyParams,
        batch: Batch,
        data: SimData,
        rng: jax.Array,
        boxes: jax.Array,
        acts: jax.Array,
        camera_fifo: jax.Array,
        action_fifo: jax.Array,
        delay: jax.Array,
        action_delay: jax.Array,
        time_s: jax.Array,
    ) -> tuple[jax.Array, ...]:
        n = batch["state"].shape[0]
        observation_batch = {**batch, "target": target_at_time(batch, time_s)}
        obs, boxes, camera_fifo, received = observe(
            data, observation_batch, cfg, rng, boxes, acts, camera_fifo, delay, noise
        )
        issued = apply_policy(params, obs)
        acts = jnp.concatenate((acts[:, 1:], issued[:, None]), 1)
        action_fifo = jnp.concatenate((action_fifo[1:], issued[None]), 0)
        executed = action_fifo[cfg.action_delay_steps - action_delay, jnp.arange(n)]
        cmd = command(executed, cfg, physical)
        return boxes, acts, camera_fifo, action_fifo, received, issued, cmd

    @jax.jit
    def hardware_policy_tick(
        params: PolicyParams,
        batch: Batch,
        data: SimData,
        hardware: H.HardwareState,
        boxes: jax.Array,
        acts: jax.Array,
        rng: jax.Array,
        time_s: jax.Array,
    ) -> tuple:
        tick_keys = jax.random.split(rng, 4)
        observation_batch = {**batch, "target": target_at_time(batch, time_s)}
        hardware, received, rotation, gyro, telemetry = H.measure(
            hardware, data, observation_batch, cfg, tick_keys[0], noise
        )
        boxes = jnp.concatenate((boxes[:, 1:], received[:, None]), axis=1)
        issued = apply_policy(params, actor_features(boxes, rotation, gyro, acts, cfg))
        acts = jnp.concatenate((acts[:, 1:], issued[:, None]), axis=1)
        hardware = H.issue(hardware, issued, cfg, tick_keys[1], noise)
        telemetry["command_delay_ms"] = H.command_delay_ms(hardware, cfg)
        return hardware, boxes, acts, received, issued, command(issued, cfg, physical), telemetry

    @jax.jit
    def hardware_substep(
        data: SimData, hardware: H.HardwareState, wind: jax.Array, rng: jax.Array
    ) -> tuple[SimData, H.HardwareState, jax.Array, jax.Array, jax.Array]:
        hardware, executed, gust = H.actuate(hardware, cfg, rng, noise)
        cmd = command(executed, cfg, physical)
        force = data.params.mass * (wind + gust)[:, None]
        data = data.replace(states=data.states.replace(force=force))
        data = F.attitude_control(data, cmd[:, None])
        data, force, torque = actuator_step(data)
        return data, hardware, cmd, force, torque

    @jax.jit
    def after_physics(hardware: H.HardwareState, data: SimData, rng: jax.Array) -> H.HardwareState:
        return H.sample_imu(hardware, data, cfg, rng, noise)

    return NativeKernels(
        jax.jit(actuator_step), policy_tick, hardware_policy_tick, hardware_substep, after_physics
    )


class NativePlant:
    """Persistent native worlds and official controller/RPM states for one batch."""

    def __init__(self, batch: Batch, cfg: Config):
        self.cfg = cfg
        self.physical = nominal_params(cfg)
        self.data = initialize(batch, cfg, self.physical)
        host = jax.device_get(batch)
        self.initial_targets = np.asarray(host["target"], dtype=float)
        self.target_velocities = np.asarray(
            host.get("target_velocity", np.zeros_like(host["target"])), dtype=float
        )
        if (
            self.target_velocities.shape != self.initial_targets.shape
            or not np.isfinite(self.target_velocities).all()
        ):
            raise ValueError("target_velocity must be finite with shape (episodes,3)")
        self.steps = 0
        inertia_scales = (
            host["domain"]["inertia_scale"] if cfg.hardware.enabled else np.ones(len(host["state"]))
        )
        self.models = [
            make_flight_model(
                cfg, scale, target, radius, inertia_scale=inertia_scale, target_velocity=velocity
            )
            for scale, target, radius, inertia_scale, velocity in zip(
                host["domain"]["mass_scale"],
                host["target"],
                host["radius"],
                inertia_scales,
                self.target_velocities,
                strict=True,
            )
        ]
        self.worlds = [mujoco.MjData(model) for model in self.models]
        self.target_mocap_ids = [
            int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "balloon")])
            for model in self.models
        ]
        self.failed = np.zeros(len(self.worlds), dtype=bool)
        initial = jax.device_get(self.data.states)
        for model, world, pos, quat, vel, omega in zip(
            self.models,
            self.worlds,
            initial.pos[:, 0],
            initial.quat[:, 0],
            initial.vel[:, 0],
            initial.ang_vel[:, 0],
            strict=True,
        ):
            world.qpos[:3] = pos
            world.qpos[3:7] = np.roll(quat, 1)
            world.qvel[:3] = vel
            world.qvel[3:] = omega
            mujoco.mj_forward(model, world)
        self.body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone") for model in self.models
        ]
        self.contact_ids = [
            {
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "drone_collision"),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "balloon_surface"),
            }
            for model in self.models
        ]
        self._actuators = _native_kernels(cfg).actuators

    def contacts(self) -> np.ndarray:
        """Native balloon contact at the current integrated pose; no image gate."""
        return np.asarray(
            [
                any(
                    {int(c.geom1), int(c.geom2)} == pair and c.dist <= 0
                    for c in world.contact[: world.ncon]
                )
                if not failed
                else False
                for world, pair, failed in zip(
                    self.worlds, self.contact_ids, self.failed, strict=True
                )
            ],
            dtype=bool,
        )

    def _synchronize(self) -> np.ndarray:
        pos = np.stack([world.qpos[:3] for world in self.worlds])
        quat = np.stack([np.roll(world.qpos[3:7], -1) for world in self.worlds])
        vel = np.stack([world.qvel[:3] for world in self.worlds])
        omega = np.stack([world.qvel[3:] for world in self.worlds])
        if self.failed.any():
            previous = jax.device_get(self.data.states)
            pos[self.failed] = previous.pos[self.failed, 0]
            quat[self.failed] = previous.quat[self.failed, 0]
            vel[self.failed] = previous.vel[self.failed, 0]
            omega[self.failed] = previous.ang_vel[self.failed, 0]
        states = self.data.states.replace(
            pos=jnp.asarray(pos[:, None]),
            quat=jnp.asarray(quat[:, None]),
            vel=jnp.asarray(vel[:, None]),
            ang_vel=jnp.asarray(omega[:, None]),
        )
        self.data = self.data.replace(states=states)
        rpy = NumpyRotation.from_quat(quat).as_euler("xyz")
        rates = ang_vel2rpy_rates(quat, omega)
        result = np.concatenate((pos, rpy, vel, rates), axis=-1)
        result[self.failed] = np.nan  # Record the first fault, never a silent simulator reset.
        return result

    def advance(self, cmd: jax.Array) -> tuple[np.ndarray, np.ndarray]:
        """Execute one held 50 Hz command through real 500 Hz MuJoCo steps."""
        self.data = F.attitude_control(self.data, cmd[:, None])
        trajectory, contacts = [], []
        for _ in range(self.cfg.substeps):
            states, touching = self._substep()
            trajectory.append(states)
            contacts.append(touching)
        return np.stack(trajectory), np.stack(contacts)

    def _substep(self) -> tuple[np.ndarray, np.ndarray]:
        """Integrate one native step after the caller sets command/external force."""
        self.data, force, torque = self._actuators(self.data)
        return self._integrate(force, torque)

    def _integrate(self, force: jax.Array, torque: jax.Array) -> tuple[np.ndarray, np.ndarray]:
        """Transfer one wrench and integrate native worlds before sampling their IMU."""
        force, torque = jax.device_get((force, torque))
        end_time = (self.steps + 1) * self.cfg.dt / self.cfg.substeps
        for episode, (model, world, body, f, t) in enumerate(
            zip(self.models, self.worlds, self.body_ids, force, torque, strict=True)
        ):
            if self.failed[episode]:
                continue
            warning_ids = (
                mujoco.mjtWarning.mjWARN_BADQPOS,
                mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC,
            )
            before = [world.warning[warning].number for warning in warning_ids]
            if not np.isfinite(np.r_[f, t]).all():
                self.failed[episode] = True
                continue
            world.xfrc_applied[body, :3] = f
            world.xfrc_applied[body, 3:] = t
            mujoco.mj_step(model, world)
            if not np.isfinite(np.r_[world.qpos, world.qvel]).all() or any(
                world.warning[warning].number > count
                for warning, count in zip(warning_ids, before, strict=True)
            ):
                self.failed[episode] = True
                continue
            # mj_step updates qpos but its collision cache is pre-integration.
            mocap = self.target_mocap_ids[episode]
            if mocap >= 0:
                world.mocap_pos[mocap] = (
                    self.initial_targets[episode] + self.target_velocities[episode] * end_time
                )
            mujoco.mj_forward(model, world)
        self.steps += 1
        return self._synchronize(), self.contacts()


def _hardware_rollout(
    params: PolicyParams, batch: Batch, cfg: Config, key: jax.Array, noise: bool, plant: NativePlant
) -> dict[str, Any]:
    """Use the identical JAX mailboxes around each actual native rigid-body step."""
    from crazyflow.balloon.rollout import trajectory_loss

    keys = jax.random.split(key, 3)
    hardware = H.initialize_hardware(
        plant.data, batch, cfg, jax.random.fold_in(keys[0], 0x4857504C), noise
    )
    n = batch["state"].shape[0]
    boxes, acts = jnp.zeros((n, cfg.history, 5)), jnp.zeros((n, cfg.history, 4))
    kernels = _native_kernels(cfg, noise)

    initial = jax.device_get(plant.data.states)
    states = [np.asarray(batch["state"])]
    substeps, contacts = [states[0]], [plant.contacts()]
    quaternion, gyro, rotor = (
        [initial.quat[:, 0]],
        [initial.ang_vel[:, 0]],
        [initial.rotor_vel[:, 0]],
    )
    received_boxes, actions, issued_commands, executed_commands = [], [], [], []
    telemetry = {
        name: [] for name in ("camera_age_ms", "gyro_age_ms", "attitude_age_ms", "command_delay_ms")
    }
    for step, rng in enumerate(jax.random.split(keys[2], cfg.steps)):
        hardware, boxes, acts, received, issued, issued_cmd, debug = kernels.hardware_policy_tick(
            params, batch, plant.data, hardware, boxes, acts, rng, jnp.asarray(step * cfg.dt)
        )
        inner_base = jax.random.split(rng, 4)[2]
        for index, subkey in enumerate(jax.random.split(inner_base, cfg.substeps)):
            actuate_key, imu_key = jax.random.split(subkey, 2)
            plant.data, hardware, executed_cmd, force, torque = kernels.hardware_substep(
                plant.data, hardware, batch["domain"]["wind"], actuate_key
            )
            sub, touching = plant._integrate(force, torque)
            hardware = kernels.after_physics(hardware, plant.data, imu_key)
            substeps.append(sub)
            contacts.append(touching)
            if index == 0:
                executed_commands.append(np.asarray(executed_cmd))
        states.append(sub)
        current = jax.device_get(plant.data.states)
        quaternion.append(current.quat[:, 0])
        gyro.append(current.ang_vel[:, 0])
        rotor.append(current.rotor_vel[:, 0])
        received_boxes.append(np.asarray(received))
        actions.append(np.asarray(issued))
        issued_commands.append(np.asarray(issued_cmd))
        for name in telemetry:
            telemetry[name].append(np.asarray(debug[name]))
    states, substeps, actions = np.stack(states), np.stack(substeps), np.stack(actions)
    return {
        "states": states,
        "substep_states": substeps,
        "bbox": np.stack(received_boxes),
        "actions": actions,
        "target_positions": np.asarray(target_at_time(batch, jnp.arange(cfg.steps + 1) * cfg.dt)),
        "loss": np.asarray(trajectory_loss(states, substeps, actions, batch, cfg)),
        "quaternion_xyzw": np.stack(quaternion),
        "body_gyro": np.stack(gyro),
        "rotor_rpm": np.stack(rotor),
        "native_contacts": np.stack(contacts),
        "issued_commands": np.stack(issued_commands),
        "executed_commands": np.stack(executed_commands),
        **{name: np.stack(values) for name, values in telemetry.items()},
    }


def rollout(
    params: PolicyParams, batch: Batch, cfg: Config, key: jax.Array, noise: bool = True
) -> dict[str, Any]:
    """Frozen policy, shared sensor pipeline, and native MuJoCo flight integration."""
    from crazyflow.balloon.rollout import trajectory_loss

    plant = NativePlant(batch, cfg)
    if cfg.hardware.enabled:
        return _hardware_rollout(params, batch, cfg, key, noise, plant)
    n = batch["state"].shape[0]
    keys = jax.random.split(key, 3)
    delay = (
        jax.random.randint(keys[0], (n,), 0, cfg.max_delay_steps + 1)
        if noise
        else jnp.zeros(n, int)
    )
    action_delay = (
        jax.random.randint(keys[1], (n,), 0, cfg.action_delay_steps + 1)
        if noise
        else jnp.zeros(n, int)
    )
    boxes, acts = jnp.zeros((n, cfg.history, 5)), jnp.zeros((n, cfg.history, 4))
    camera_fifo = jnp.zeros((cfg.max_delay_steps + 1, n, 5))
    action_fifo = jnp.zeros((cfg.action_delay_steps + 1, n, 4))
    kernels = _native_kernels(cfg, noise)

    initial = jax.device_get(plant.data.states)
    states = [np.asarray(batch["state"])]
    substeps = [states[0]]
    contacts = [plant.contacts()]
    quaternions = [initial.quat[:, 0]]
    gyro, rotor = [initial.ang_vel[:, 0]], [initial.rotor_vel[:, 0]]
    received_boxes, actions = [], []
    for step, rng in enumerate(jax.random.split(keys[2], cfg.steps)):
        boxes, acts, camera_fifo, action_fifo, received, issued, cmd = kernels.policy_tick(
            params,
            batch,
            plant.data,
            rng,
            boxes,
            acts,
            camera_fifo,
            action_fifo,
            delay,
            action_delay,
            jnp.asarray(step * cfg.dt),
        )
        sub, contact = plant.advance(cmd)
        states.append(sub[-1])
        substeps.extend(sub)
        contacts.extend(contact)
        current = jax.device_get(plant.data.states)
        quaternions.append(current.quat[:, 0])
        gyro.append(current.ang_vel[:, 0])
        rotor.append(current.rotor_vel[:, 0])
        received_boxes.append(np.asarray(received))
        actions.append(np.asarray(issued))
    states, substeps, actions = np.stack(states), np.stack(substeps), np.stack(actions)
    return {
        "states": states,
        "substep_states": substeps,
        "bbox": np.stack(received_boxes),
        "actions": actions,
        "target_positions": np.asarray(target_at_time(batch, jnp.arange(cfg.steps + 1) * cfg.dt)),
        "loss": np.asarray(trajectory_loss(states, substeps, actions, batch, cfg)),
        "quaternion_xyzw": np.stack(quaternions),
        "body_gyro": np.stack(gyro),
        "rotor_rpm": np.stack(rotor),
        "native_contacts": np.stack(contacts),
    }
