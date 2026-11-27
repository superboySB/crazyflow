"""CrazyRL ToF navigation on Crazyflow's functional JAX simulation pipeline.

``reset`` and ``step`` are pure, JIT compiled batch functions.  ``step`` returns
an auto-reset state and pre-reset terminal observations for PPO time-limit
bootstrapping. Simulation outputs are stopped at the autodiff boundary.

Parity is defined by CrazyRL/e2e_drone/tasks/baseline/quad_baseline_env.py:
344 actor inputs, 434 critic inputs, raw Gaussian actions, the static_goals
reward, 15 Hz policy, 150 Hz physics and the same sensor/physical noise laws.
Crazyflow retains its own official first-principles drone/motor parameters.
"""

from __future__ import annotations

import dataclasses
import math
import tomllib
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.scipy.spatial.transform import Rotation

from crazyflow.control import Control
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData, SimState
from crazyflow.sim.integration import Integrator
from crazyflow.sim.sim import clip_rotor_vel_cmd
from crazyflow.tof.config import EnvConfig
from crazyflow.tof.scene import Scene, build_scene, nearest_obstacle, raycast


@struct.dataclass
class EnvState:
    sim: SimData
    key: jax.Array
    goal: jax.Array
    map_id: jax.Array
    desired_speed_init: jax.Array
    desired_speed: jax.Array
    action_buffer: jax.Array
    last_action: jax.Array
    wind: jax.Array
    wind_mean: jax.Array
    thrust_effectiveness: jax.Array
    degradation_rate: jax.Array
    actor_history: jax.Array
    critic_history: jax.Array
    obs: jax.Array
    critic_obs: jax.Array
    depth: jax.Array
    auxiliary_depth: jax.Array
    steps: jax.Array
    episode_return: jax.Array
    hover_counter: jax.Array
    recent_outcomes: jax.Array
    outcome_cursor: jax.Array
    episode_count: jax.Array
    map_reset_batches: jax.Array


@struct.dataclass
class StepResult:
    reward: jax.Array
    done: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    success: jax.Array
    collision: jax.Array
    instability: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    terminal_obs: jax.Array
    terminal_critic_obs: jax.Array
    terminal_pos: jax.Array
    terminal_quat: jax.Array
    terminal_vel: jax.Array
    terminal_distance: jax.Array
    reward_components: jax.Array


def squash_action(raw_action: jax.Array) -> jax.Array:
    """Apply the reference tanh and collective-thrust channel mapping."""
    action = jnp.tanh(raw_action)
    return action.at[..., 3].set((action[..., 3] + 1) * 0.5)


def huber(value: jax.Array, delta: float) -> jax.Array:
    """Compute the reference quadratic-to-linear penalty with unit slope."""
    value = jnp.abs(value)
    return jnp.where(value <= delta, 0.5 * value**2 / delta, value - 0.5 * delta)


def reward_terms(
    cfg: EnvConfig,
    pos: jax.Array,
    previous_pos: jax.Array,
    quat: jax.Array,
    vel_b: jax.Array,
    action: jax.Array,
    previous_action: jax.Array,
    goal: jax.Array,
    desired_speed_init: jax.Array,
    obstacle_distance: jax.Array,
    collision: jax.Array,
    success: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The eleven nonzero-capable CrazyRL static_goals terms, before weighting."""
    relative = goal - pos
    distance = jnp.linalg.norm(relative, axis=-1)
    delta = jnp.linalg.norm(goal - previous_pos, axis=-1) - distance
    forward = Rotation.from_quat(quat).as_matrix()[:, :2, 0]
    forward /= jnp.linalg.norm(forward, axis=-1, keepdims=True) + 1e-6
    goal_dir = relative[:, :2] / (jnp.linalg.norm(relative[:, :2], axis=-1, keepdims=True) + 1e-6)
    slowdown = jnp.clip(distance, 0, 1)
    yaw = (jnp.sum(forward * goal_dir, -1) - 1) * slowdown
    yaw = jnp.where(distance < 0.2, 0, yaw)
    speed = jnp.linalg.norm(vel_b, axis=-1)
    desired_speed = desired_speed_init * slowdown
    misalignment = speed * (1 - jnp.clip(vel_b[:, 0] / jnp.maximum(speed, 1e-6), -1, 1))
    excess = -jnp.minimum(jnp.expm1(jnp.clip((speed - desired_speed) * 5, 0, 20)), 5)
    clearance = -jnp.expm1(5 * jnp.maximum(0.5 - obstacle_distance, 0))
    components = jnp.stack(
        [
            jnp.minimum(delta, 0.01),
            yaw,
            -jnp.linalg.norm(action, axis=-1),
            -jnp.linalg.norm(action - previous_action, axis=-1),
            -huber(misalignment, 0.2),
            excess,
            2 * jnp.exp(-5 * jnp.abs(speed - desired_speed)),
            -huber(pos[:, 2] - goal[:, 2], 0.3),
            -collision.astype(jnp.float32),
            clearance,
            success.astype(jnp.float32),
        ],
        -1,
    )
    weights = jnp.array(
        [
            cfg.reward_distance,
            cfg.reward_yaw,
            cfg.reward_action,
            cfg.reward_action_change,
            cfg.reward_velocity_direction,
            cfg.reward_speed_excess,
            cfg.reward_speed_match,
            cfg.reward_height,
            cfg.reward_collision,
            cfg.reward_esdf,
            cfg.reward_success,
        ],
        jnp.float32,
    )
    return jnp.sum(components * weights, -1), components * weights, desired_speed


class ToFEnv:
    action_size = 4

    def __init__(self, config: EnvConfig | None = None, num_envs: int = 1024, device: str = "gpu"):
        self.config = self.cfg = config or EnvConfig()
        cfg = self.config
        self.num_envs = num_envs
        self.device = jax.devices(device)[0]
        self.observation_size = cfg.observation_size
        self.critic_observation_size = cfg.critic_observation_size
        self.scene = build_scene(cfg, self.device)
        self._active_map = min(1, cfg.num_maps - 1)
        self._map_changes = []
        self._generation = 0
        # CrazyRL samples this ONCE at construction. It changes controller
        # inertia and the wind-force multiplier, not the actual plant mass.
        mass_key = jax.device_put(jax.random.PRNGKey(cfg.map_seed + 7919), self.device)
        ratio = (
            1
            + (jax.random.uniform(mass_key, (num_envs,)) * 2 - 1)
            * cfg.mass_randomization_percent
            * cfg.domain_randomization
        )
        self.controller_mass = 0.049 * ratio
        # Construct the official simulator once. Only its small functional SimData
        # is expanded: training does not allocate n_worlds copies of MJX scene data.
        sim = Sim(
            n_worlds=1,
            n_drones=1,
            drone=cfg.drone,
            dynamics=Dynamics.first_principles,
            control=Control.rotor_vel,
            integrator=Integrator[cfg.integrator],
            freq=cfg.physics_hz,
            device=device,
            fused_mjx_model=True,
        )
        self._physical_override = None
        if cfg.model_params is not None or cfg.model_params_path:
            from crazyflow.identify_physics import validate_physical_params

            if cfg.model_params is not None:
                values = cfg.model_params
            else:
                table = tomllib.loads(Path(cfg.model_params_path).read_text())
                values = table.get(cfg.drone, table)
            validate_physical_params(values)
            from crazyflow.dynamics import load_params

            defaults = load_params(Dynamics.first_principles, cfg.drone)
            values = {name: value for name, value in defaults.items() if name != "J_inv"} | values
            self._physical_override = {
                name: np.asarray(value).tolist() for name, value in values.items()
            }
            updates = {}
            scalar = {"mass", "L", "prop_inertia"}
            motor = {"rpm2thrust", "rpm2torque", "rotor_dyn_coef"}
            for name in sim.data.params.__dataclass_fields__:
                if name not in values or name == "J_inv":
                    continue
                value = np.asarray(values[name], dtype=np.float32)
                if name in scalar:
                    value = np.atleast_1d(value)
                if name in motor and value.ndim == 1:
                    value = value[None]
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"Non-finite physical parameter: {name}")
                updates[name] = jax.device_put(jnp.asarray(value), self.device)
            updated = sim.data.params.replace(**updates)
            inertia = np.asarray(updated.J)
            if (
                float(updated.mass[0]) <= 0
                or not np.allclose(inertia, inertia.T)
                or np.min(np.linalg.eigvalsh(inertia)) <= 0
            ):
                raise ValueError("Physical mass and symmetric inertia must be positive")
            updated = updated.replace(J_inv=jnp.linalg.inv(updated.J))
            sim.data = sim.data.replace(params=updated)
            max_force = values.get("thrust_max", 0.2)
            # Preserve the official loader's coefficient precision when deriving
            # the command limit; converting through float32 params can move the
            # maximum by an ULP when a default model is exported and reloaded.
            rpm_max = np.asarray(
                motor_force2rotor_vel(np.asarray(max_force), np.atleast_2d(values["rpm2thrust"]))
            )
            if not np.all(np.isfinite(rpm_max)) or np.any(rpm_max <= 0):
                raise ValueError("Invalid identified thrust curve/limit")
            sim.step_pipeline["clip_rotor_vel_cmd"] = partial(
                clip_rotor_vel_cmd,
                lower=0.0,
                upper=jax.device_put(jnp.asarray(rpm_max), self.device),
                dynamics=Dynamics.first_principles,
            )
        self.nominal_params = sim.data.params
        if self._physical_override is None:
            from crazyflow.dynamics import load_params

            values = load_params(Dynamics.first_principles, cfg.drone)
            self._physical_override = {
                name: np.asarray(value).tolist()
                for name, value in values.items()
                if name != "J_inv"
            }
        base = sim.data
        self.template = base.replace(
            states=SimState.create(num_envs, 1, self.device),
            controls=base.controls.replace(
                rotor_vel=jnp.zeros((num_envs, 1, 4), device=self.device)
            ),
            params=base.params.replace(
                mass=jnp.full((num_envs, 1, 1), float(base.params.mass[0]), device=self.device)
            ),
            core=base.core.replace(
                n_worlds=num_envs,
                rng_key=jax.random.key_data(base.core.rng_key),
                steps=jnp.zeros((num_envs, 1), jnp.int32, device=self.device),
            ),
        )
        self._sim_step = sim.build_step_fn()
        # Controller physical constants reproduce CrazyRL controller.py exactly;
        # motor mixing is performed with the actual Crazyflow motor convention.
        omega_max = 24000 * math.pi / 30
        self.reference_max_thrust = 1.358 * (
            9.96063125e-8 * omega_max**2 - 2.55003087e-5 * omega_max + 5.84422691e-3
        )
        p = self.nominal_params
        mixing = np.asarray(p.mixing_matrix)
        arm = float(np.asarray(p.L)[0])
        kf = np.asarray(p.rpm2thrust)[0, 2]
        kt = np.asarray(p.rpm2torque)[0, 2]
        allocation = np.concatenate([np.ones((1, 4)), mixing[:2] * arm, mixing[2:3] * (kt / kf)], 0)
        self.allocation_inverse = jax.device_put(
            jnp.asarray(np.linalg.inv(allocation), jnp.float32), self.device
        )
        self.controller_inertia = jnp.array([2.16e-5, 2.16e-5, 4.33e-5], device=self.device)
        pixel = (np.arange(8) + 0.5 - 4) * cfg.camera_aperture / (8 * cfg.camera_focal_length)
        u, v = np.meshgrid(pixel, pixel, indexing="xy")
        front = np.stack([np.ones_like(u), -u, -v], -1).reshape(64, 3)
        self.ray_directions = jax.device_put(jnp.asarray(front, jnp.float32), self.device)
        auxiliary = np.array([[0, 1, 0], [0, -1, 0], [-1, 0, 0], [0, 0, -1]], np.float32)
        self._rays = jax.device_put(jnp.asarray(np.concatenate([front, auxiliary])), self.device)
        offsets = np.concatenate([np.tile([0.02, 0, 0], (64, 1)), auxiliary * 0.02])
        self._offsets = jax.device_put(jnp.asarray(offsets, jnp.float32), self.device)
        self.reset = jax.jit(self._reset)
        self.step = jax.jit(self._step)

    def _reset(self, key: jax.Array) -> EnvState:
        cfg, n = self.cfg, self.num_envs
        keys = jax.random.split(key, 8)
        map_id = jnp.full((n,), self._active_map, jnp.int32)
        free_count = (
            self.scene.free_counts[map_id]
            if self.scene.free_counts is not None
            else jnp.full((n,), self.scene.free_points.shape[1])
        )
        indices = jax.random.randint(keys[1], (n, 2), 0, free_count[:, None])
        pos = self.scene.free_points[map_id, indices[:, 0]]
        goal = self.scene.free_points[map_id, indices[:, 1]]
        goal += jax.random.uniform(keys[2], (n, 3), minval=-cfg.goal_noise, maxval=cfg.goal_noise)
        yaw = jax.random.uniform(keys[3], (n,), minval=-jnp.pi, maxval=jnp.pi)
        quat = jnp.stack(
            [jnp.zeros_like(yaw), jnp.zeros_like(yaw), jnp.sin(yaw / 2), jnp.cos(yaw / 2)], -1
        )
        randomness = jax.random.uniform(keys[4], (n, 4))
        mass = jnp.full((n,), self.nominal_params.mass[0])
        eta = cfg.thrust_initial_min + randomness[:, 1] * (
            cfg.thrust_initial_max - cfg.thrust_initial_min
        )
        eta = jnp.where(cfg.domain_randomization, eta, 1.0)
        degradation = (
            cfg.thrust_degradation_min
            + randomness[:, 2] * (cfg.thrust_degradation_max - cfg.thrust_degradation_min)
        ) / cfg.max_episode_steps
        degradation = jnp.where(cfg.domain_randomization, degradation, 0.0)
        speed = cfg.desired_speed_min + randomness[:, 3] * (
            cfg.desired_speed_max - cfg.desired_speed_min
        )
        sim = self.template.replace(
            states=self.template.states.replace(pos=pos[:, None], quat=quat[:, None]),
            params=self.template.params.replace(mass=mass[:, None, None]),
        )
        zeros = jnp.zeros((n,), jnp.float32)
        history = jnp.zeros((n, cfg.history_length, 28), jnp.float32)
        critic_history = jnp.zeros((n, cfg.history_length, 37), jnp.float32)
        # The reference buffer initially contains [0,0,.5,0]. Reset each world
        # independently to avoid carrying the previous episode's command.
        buffer = jnp.zeros((n, cfg.action_delay_steps + 1, 4), jnp.float32).at[..., 2].set(0.5)
        state = EnvState(
            sim,
            keys[5],
            goal,
            map_id,
            speed,
            speed,
            buffer,
            jnp.zeros((n, 4)),
            jnp.zeros((n, 3)),
            jnp.zeros((n, 3)),
            eta,
            degradation,
            history,
            critic_history,
            jnp.zeros((n, cfg.observation_size)),
            jnp.zeros((n, cfg.critic_observation_size)),
            jnp.zeros((n, 64)),
            jnp.zeros((n, 4)),
            jnp.zeros((n,), jnp.int32),
            zeros,
            zeros,
            jnp.zeros((100,), jnp.float32),
            jnp.array(0, jnp.int32),
            jnp.array(0, jnp.int32),
            jnp.array(0, jnp.int32),
        )
        return self._observe(state)

    def _controller(self, sim: SimData, action: jax.Array) -> jax.Array:
        quat = sim.states.quat[:, 0]
        euler = Rotation.from_quat(quat).as_euler("xyz")
        omega = sim.states.ang_vel[:, 0]
        error = (action[:, :2] * (jnp.pi / 6) - euler[:, :2] + jnp.pi) % (2 * jnp.pi) - jnp.pi
        rates = 20 * error - 0.5 * omega[:, :2]
        desired = jnp.concatenate([rates, action[:, 2:3] * (jnp.pi / 1.5)], -1)
        angular_acc = jnp.array([20.0, 20.0, 15.0]) * (desired - omega)
        inertia = self.controller_inertia[None] * (self.controller_mass / 0.049)[:, None]
        torque = inertia * angular_acc + jnp.cross(omega, inertia * omega)
        wrench = jnp.concatenate([jnp.zeros((self.num_envs, 1)), torque], -1)
        differential = wrench @ self.allocation_inverse.T
        collective = action[:, 3:4] * self.reference_max_thrust / 4
        positive = self.reference_max_thrust / 4 - collective
        negative = collective
        ratios = jnp.where(
            differential > 0,
            differential / jnp.maximum(positive, 1e-10),
            -differential / jnp.maximum(negative, 1e-10),
        )
        differential /= jnp.maximum(1, jnp.max(ratios, -1, keepdims=True))
        thrust = jnp.maximum(collective + differential, 0)
        coeff = self.nominal_params.rpm2thrust
        k0, k1, k2 = coeff[..., 0], coeff[..., 1], coeff[..., 2]
        rpm = (-k1 + jnp.sqrt(jnp.maximum(k1**2 + 4 * k2 * (thrust - k0), 0))) / (2 * k2)
        return jnp.where(thrust > 0, rpm, 0)[:, None]

    def _physics(self, state: EnvState, action: jax.Array) -> tuple[EnvState, jax.Array]:
        cfg = self.cfg
        alpha = math.exp(-1 / (cfg.physics_hz * cfg.wind_tau))
        sigma = cfg.wind_sigma * math.sqrt(1 - alpha**2) * cfg.domain_randomization

        def tick(carry: Any, _: None) -> tuple[Any, None]:
            sim, buffer, wind, eta, key, collision, wind_sum = carry
            key, sample = jax.random.split(key)
            buffer = jnp.concatenate([buffer[:, 1:], action[:, None]], 1)
            delayed = buffer[:, 0]
            rpm = self._controller(sim, delayed)
            wind = alpha * wind + sigma * jax.random.normal(sample, wind.shape)
            eta = jnp.maximum(eta * (1 - state.degradation_rate), cfg.thrust_min_effectiveness)
            p = self.nominal_params
            rotor = sim.states.rotor_vel[:, 0]
            coeff = p.rpm2thrust
            total = jnp.sum(coeff[..., 0] + coeff[..., 1] * rotor + coeff[..., 2] * rotor**2, -1)
            rotation = Rotation.from_quat(sim.states.quat[:, 0]).as_matrix()
            thrust_error = rotation[:, :, 2] * ((eta - 1) * total)[:, None]
            force = wind * self.controller_mass[:, None] + thrust_error
            sim = sim.replace(
                controls=sim.controls.replace(rotor_vel=rpm),
                states=sim.states.replace(force=force[:, None]),
            )
            sim = self._sim_step(sim, n_steps=1)
            distance, _ = nearest_obstacle(self.scene, sim.states.pos[:, 0], state.map_id)
            collision |= distance < cfg.drone_radius
            return (sim, buffer, wind, eta, key, collision, wind_sum + wind), None

        initial = (
            state.sim,
            state.action_buffer,
            state.wind,
            state.thrust_effectiveness,
            state.key,
            jnp.zeros((self.num_envs,), bool),
            jnp.zeros_like(state.wind),
        )
        (sim, buffer, wind, eta, key, collision, wind_sum), _ = jax.lax.scan(
            tick, initial, None, length=cfg.physics_hz // cfg.policy_hz
        )
        sim = jax.tree.map(jax.lax.stop_gradient, sim)
        return state.replace(
            sim=sim,
            action_buffer=buffer,
            wind=wind,
            wind_mean=wind_sum / (cfg.physics_hz // cfg.policy_hz),
            thrust_effectiveness=eta,
            key=key,
        ), collision

    def _observe(self, state: EnvState) -> EnvState:
        cfg, n = self.cfg, self.num_envs
        keys = jax.random.split(state.key, 12)
        pos, quat = state.sim.states.pos[:, 0], state.sim.states.quat[:, 0]
        rotation = Rotation.from_quat(quat).as_matrix()
        origins = pos[:, None] + jnp.einsum("nij,rj->nri", rotation, self._offsets)
        directions = jnp.einsum("nij,rj->nri", rotation, self._rays)
        sensed = raycast(self.scene, origins, directions, state.map_id, cfg.camera_max_distance)
        depth, auxiliary = sensed[:, :64], sensed[:, 64:]
        # One invalid probability is drawn per observation update in CrazyRL.
        invalid_rate = (
            jax.random.uniform(keys[0], ()) * cfg.depth_invalid_rate_max * cfg.observation_noise
        )
        invalid = jax.random.uniform(keys[1], depth.shape) < invalid_rate
        policy_depth = jnp.where(invalid, cfg.camera_max_distance, depth)
        vel_b = jnp.einsum("nji,nj->ni", rotation, state.sim.states.vel[:, 0])
        gyro = state.sim.states.ang_vel[:, 0]
        relative = state.goal - pos
        direction_xy = relative[:, :2]
        if cfg.observation_noise:
            direction_xy *= 1 + 0.2 * jax.random.normal(keys[2], direction_xy.shape)
        direction_xy /= jnp.linalg.norm(direction_xy, axis=-1, keepdims=True) + 1e-6
        # The reference critic reuses this noisy goal direction: preserve that detail.
        direction_goal = jnp.concatenate([direction_xy, state.goal[:, 2:3]], -1)
        z_error = pos[:, 2:3] - state.goal[:, 2:3]

        def noise(value: jax.Array, std: float, key: jax.Array) -> jax.Array:
            return value * (1 + std * jax.random.normal(key, value.shape) * cfg.observation_noise)

        common = [rotation.reshape(n, 9) * 127]
        actor = jnp.concatenate(
            [
                noise(vel_b, 0.3, keys[3]) * 25,
                *common,
                noise(gyro, 0.1, keys[4]) * 30,
                direction_goal * 127,
                noise(z_error, 0.1, keys[5]) * 30,
                state.desired_speed[:, None] * 25,
                state.last_action * 127,
                noise(auxiliary, 0.1, keys[6]) * 30,
            ],
            -1,
        )
        distance, obstacle_vec = nearest_obstacle(self.scene, pos, state.map_id)
        relative_norm = jnp.linalg.norm(relative, axis=-1, keepdims=True)
        rel_goal = relative / (relative_norm + 1e-5) * jnp.minimum(relative_norm, 5)
        obstacle_norm = jnp.linalg.norm(obstacle_vec, axis=-1, keepdims=True)
        rel_obstacle = obstacle_vec / (obstacle_norm + 1e-5) * jnp.minimum(obstacle_norm, 1)
        critic = jnp.concatenate(
            [
                vel_b * 25,
                *common,
                gyro * 30,
                direction_goal * 127,
                z_error * 30,
                state.desired_speed[:, None] * 25,
                state.last_action * 127,
                auxiliary * 30,
                rel_goal * 127,
                rel_obstacle * 127,
                state.wind_mean * 30,
            ],
            -1,
        )
        actor_history = jnp.concatenate([state.actor_history[:, 1:], actor[:, None]], 1)
        critic_history = jnp.concatenate([state.critic_history[:, 1:], critic[:, None]], 1)
        obs = jnp.concatenate(
            [actor_history.reshape(n, -1), noise(policy_depth, 0.1, keys[7]) * 30], -1
        )
        critic_obs = jnp.concatenate([critic_history.reshape(n, -1), depth * 30], -1)
        return state.replace(
            key=keys[8],
            actor_history=actor_history,
            critic_history=critic_history,
            obs=obs,
            critic_obs=critic_obs,
            depth=depth,
            auxiliary_depth=auxiliary,
        )

    def _step(self, state: EnvState, raw_action: jax.Array) -> tuple[EnvState, StepResult]:
        cfg = self.cfg
        raw_action = jax.lax.stop_gradient(raw_action)
        action = squash_action(raw_action)
        previous_pos = state.sim.states.pos[:, 0]
        previous_action = state.last_action
        state, collision = self._physics(state, action)
        pos, quat = state.sim.states.pos[:, 0], state.sim.states.quat[:, 0]
        vel, gyro = state.sim.states.vel[:, 0], state.sim.states.ang_vel[:, 0]
        vel_b = Rotation.from_quat(quat).apply(vel, inverse=True)
        distance, _ = nearest_obstacle(self.scene, pos, state.map_id)
        goal_distance = jnp.linalg.norm(state.goal - pos, axis=-1)
        hover = (goal_distance < cfg.hover_distance) & (
            jnp.linalg.norm(vel_b, axis=-1) < cfg.hover_speed
        )
        hold = jnp.where(hover, state.hover_counter + cfg.dt, 0)
        success = hover & (hold >= cfg.hover_hold_seconds)
        finite = jnp.all(jnp.isfinite(jnp.concatenate([pos, quat, vel, gyro], -1)), -1)
        instability = ~finite | jnp.any(jnp.abs(gyro) > 3.14 * 2 * 20, -1)
        reward, components, desired = reward_terms(
            cfg,
            pos,
            previous_pos,
            quat,
            vel_b,
            action,
            previous_action,
            state.goal,
            state.desired_speed_init,
            distance,
            collision,
            success,
        )
        reward = jnp.nan_to_num(
            reward,
            nan=-cfg.reward_collision,
            neginf=-cfg.reward_collision,
            posinf=cfg.reward_success,
        )
        steps = state.steps + 1
        truncated = steps >= cfg.max_episode_steps
        terminated = success | collision | instability
        done = terminated | truncated
        episode_return = state.episode_return + reward
        completions = jnp.sum(done.astype(jnp.int32))
        rank = jnp.cumsum(done.astype(jnp.int32)) - 1
        keep = done & (rank >= completions - 100)
        slots = jnp.where(keep, (state.outcome_cursor + rank) % 100, 100)
        outcomes = state.recent_outcomes.at[slots].set(success.astype(jnp.float32), mode="drop")
        state = self._observe(
            state.replace(
                last_action=action,
                desired_speed=desired,
                steps=steps,
                episode_return=episode_return,
                hover_counter=hold,
                recent_outcomes=outcomes,
                outcome_cursor=(state.outcome_cursor + completions) % 100,
                episode_count=state.episode_count + completions,
                map_reset_batches=state.map_reset_batches + jnp.any(done).astype(jnp.int32),
            )
        )
        result = StepResult(
            reward,
            done,
            terminated,
            truncated,
            success,
            collision,
            instability,
            episode_return,
            steps,
            state.obs,
            state.critic_obs,
            pos,
            quat,
            vel,
            goal_distance,
            components,
        )

        # A conditional batch reset avoids expensive raycasts on every regular step.
        def reset_done(current: EnvState) -> EnvState:
            key, reset_key = jax.random.split(current.key)
            fresh = self._reset(reset_key)

            def select(old: jax.Array, new: jax.Array) -> jax.Array:
                if old.ndim > 0 and old.shape[0] == self.num_envs:
                    mask = done.reshape((self.num_envs,) + (1,) * (old.ndim - 1))
                    return jnp.where(mask, new, old)
                return old

            selected = jax.tree.map(select, current, fresh)
            return selected.replace(
                key=key,
                recent_outcomes=current.recent_outcomes,
                outcome_cursor=current.outcome_cursor,
                episode_count=current.episode_count,
                map_reset_batches=current.map_reset_batches,
            )

        state = jax.lax.cond(jnp.any(done), reset_done, lambda current: current, state)
        return jax.tree.map(jax.lax.stop_gradient, (state, result))

    def _replace_map(self, index: int, replacement: Scene) -> None:
        capacity = max(self.scene.centers.shape[1], replacement.centers.shape[1])

        def merge(old: jax.Array, new: jax.Array, fill: float) -> jax.Array:
            old = np.asarray(old)
            padded = np.full((old.shape[0], capacity, *old.shape[2:]), fill, old.dtype)
            padded[:, : old.shape[1]] = old
            padded[index] = fill
            padded[index, : new.shape[1]] = np.asarray(new)[0]
            return jax.device_put(jnp.asarray(padded), self.device)

        centers = merge(self.scene.centers, replacement.centers, 1e5)
        sizes = merge(self.scene.half_sizes, replacement.half_sizes, 0.01)
        kinds = merge(self.scene.kinds, replacement.kinds, 0)
        rotations = merge(self.scene.rotations, replacement.rotations, 0)
        # Padded objects are far away; their orientation must remain invertible.
        valid = np.linalg.norm(np.asarray(rotations), axis=(-2, -1)) > 0
        rotations = jnp.where(jnp.asarray(valid)[..., None, None], rotations, jnp.eye(3))
        free = self.scene.free_points.at[index].set(replacement.free_points[0])
        counts = self.scene.counts.copy()
        counts[index] = replacement.counts[0]
        self.scene = Scene(
            centers,
            sizes,
            kinds,
            rotations,
            free,
            counts,
            self.scene.room_half_size,
            self.scene.floor,
            self.scene.ceiling,
            self.scene.free_counts.at[index].set(replacement.free_counts[0]),
        )

    def maybe_regenerate(self, state: EnvState) -> tuple[EnvState, bool]:
        """Host-side map maintenance after a PPO rollout; returns (state, changed).

        The reference increments its 3000 counter once per batch of episode
        resets. This counter is maintained inside the compiled step. Geometry
        generation stays outside the device rollout and is checked at update
        boundaries. An occupied inactive map is never overwritten.
        """
        cfg = self.cfg
        if cfg.num_maps < 2 or int(state.map_reset_batches) < cfg.map_generation_reset_batches:
            return state, False
        success_rate = float(
            jnp.sum(state.recent_outcomes) / jnp.maximum(1, jnp.minimum(state.episode_count, 100))
        )
        changes = {}
        if cfg.enable_difficulty_curriculum and success_rate > 0.9:
            target = min(
                cfg.hover_hold_max_seconds,
                self.config.hover_hold_seconds
                + (cfg.hover_hold_max_seconds - self.config.hover_hold_seconds) * success_rate,
            )
            if target > cfg.hover_hold_seconds:
                changes["hover_hold_seconds"] = cfg.hover_hold_seconds + min(
                    cfg.hover_hold_increment_seconds, target - cfg.hover_hold_seconds
                )
            if cfg.obstacle_min_distance > cfg.obstacle_min_distance_min:
                changes["obstacle_min_distance"] = max(
                    cfg.obstacle_min_distance / 1.1, cfg.obstacle_min_distance_min
                )
            else:
                changes.update(reward_speed_excess=0.6, reward_speed_match=0.2)
        if changes:
            self.cfg = dataclasses.replace(cfg, **changes)
        inactive = 1 - self._active_map
        if bool(jnp.any(state.map_id == inactive)):
            if changes:
                self.reset, self.step = jax.jit(self._reset), jax.jit(self._step)
            return state, bool(changes)
        self._generation += 1
        seed = self.config.map_seed + self._generation * 1009
        replacement_cfg = dataclasses.replace(self.cfg, num_maps=1, map_seed=seed)
        replacement = build_scene(replacement_cfg, self.device)
        self._replace_map(inactive, replacement)
        self._active_map = inactive
        self._map_changes.append(
            {
                "index": inactive,
                "seed": seed,
                "obstacle_min_distance": self.cfg.obstacle_min_distance,
            }
        )
        self.reset, self.step = jax.jit(self._reset), jax.jit(self._step)
        return state.replace(map_reset_batches=jnp.array(0, jnp.int32)), True

    def effective_config(self) -> dict[str, Any]:
        """Initial task configuration with the complete actual physics embedded.

        This also freezes the shipped nominal parameters, so exported models
        remain independent of later TOML edits. Adaptive stages live in runtime_state.
        """
        config = self.config.to_dict()
        config["model_params_path"] = None
        config["model_params"] = self._physical_override
        return config

    def runtime_state(self) -> dict[str, Any]:
        """Serializable host state needed to resume the identical map curriculum."""
        config = self.cfg.to_dict()
        config["model_params_path"] = None
        config["model_params"] = self._physical_override
        return {
            "active_map": self._active_map,
            "generation": self._generation,
            "map_changes": self._map_changes,
            "config": config,
        }

    def restore_runtime(self, values: dict[str, Any]) -> None:
        """Restore before compiling the trainer update or stepping saved EnvState."""
        self.scene = build_scene(self.config, self.device)
        for change in values["map_changes"]:
            cfg = dataclasses.replace(
                self.config,
                num_maps=1,
                map_seed=change["seed"],
                obstacle_min_distance=change["obstacle_min_distance"],
            )
            self._replace_map(change["index"], build_scene(cfg, self.device))
        self.cfg = EnvConfig(**values["config"])
        self._active_map = values["active_map"]
        self._generation = values["generation"]
        self._map_changes = list(values["map_changes"])
        self.reset, self.step = jax.jit(self._reset), jax.jit(self._step)
