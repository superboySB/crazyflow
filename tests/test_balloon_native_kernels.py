"""Cached native kernels keep episode inputs dynamic and preserve control staging."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.balloon import hardware as H
from crazyflow.balloon import model, mujoco_rollout
from crazyflow.balloon.config import Config
from crazyflow.balloon.noise_config import HardwareNoise
from crazyflow.balloon.observations import actor_features
from crazyflow.balloon.policy import PolicyParams, apply_policy, init_policy
from crazyflow.sim import functional as F
from crazyflow.sim.data import SimData


def _config() -> Config:
    return Config(
        steps=3,
        hidden=8,
        hardware=HardwareNoise(
            enabled=True,
            camera_delay_ms=(0, 0),
            attitude_command_delay_ms=(0, 0),
            thrust_command_delay_ms=(0, 0),
            gyro_scale_std=0.1,
            gyro_drift_std=0.003,
            attitude_bias_std=0.002,
            gust_std=0.04,
        ),
    )


def _assert_tree_close(actual: object, expected: object) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=2e-6, rtol=2e-6)


def _changed_batch(batch: model.Batch) -> model.Batch:
    changed = jax.tree.map(lambda value: value, batch)
    changed["target"] = batch["target"] + jnp.array([0.0, 0.8, 0.2])
    changed["target_velocity"] = jnp.array([[0.2, -0.3, 0.1], [-0.1, 0.4, -0.2]])
    changed["radius"] = jnp.array([0.04, 0.27])
    changed["domain"] = {
        **batch["domain"],
        "mass_scale": jnp.array([0.91, 1.12]),
        "wind": jnp.array([[0.3, -0.1, 0.2], [-0.2, 0.1, -0.1]]),
    }
    return changed


def test_factory_reuses_value_equal_config_and_separates_changed_config_and_noise() -> None:
    cfg = _config()
    reordered = replace(cfg, loss_weights=dict(reversed(tuple(cfg.loss_weights.items()))))
    kernels = mujoco_rollout._native_kernels(cfg, True)
    assert mujoco_rollout._native_kernels(reordered, True) is kernels
    assert mujoco_rollout._native_kernels(Config(**cfg.to_dict()), True) is kernels
    assert mujoco_rollout._native_kernels(replace(cfg, max_yaw=0.9), True) is not kernels
    assert mujoco_rollout._native_kernels(cfg, False) is not kernels
    assert mujoco_rollout._cached_native_kernels.cache_info().maxsize == 8


def _policy_reference(
    params: PolicyParams,
    batch: model.Batch,
    data: SimData,
    hardware: H.HardwareState,
    boxes: jax.Array,
    actions: jax.Array,
    cfg: Config,
    key: jax.Array,
    time_s: float,
) -> tuple:
    keys = jax.random.split(key, 4)
    observation_batch = {**batch, "target": model.target_at_time(batch, time_s)}
    hardware, received, rotation, gyro, telemetry = H.measure(
        hardware, data, observation_batch, cfg, keys[0], True
    )
    boxes = jnp.concatenate((boxes[:, 1:], received[:, None]), axis=1)
    issued = apply_policy(params, actor_features(boxes, rotation, gyro, actions, cfg))
    actions = jnp.concatenate((actions[:, 1:], issued[:, None]), axis=1)
    hardware = H.issue(hardware, issued, cfg, keys[1], True)
    telemetry["command_delay_ms"] = H.command_delay_ms(hardware, cfg)
    cmd = model.command(issued, cfg, model.nominal_params(cfg))
    return hardware, boxes, actions, received, issued, cmd, telemetry


def test_cached_policy_accepts_new_weights_targets_velocity_radius_and_domain() -> None:
    cfg, key = _config(), jax.random.key(41)
    kernels = mujoco_rollout._native_kernels(cfg, True)
    batch = model.sample_batch(key, cfg, 2, "noisy")
    changed = _changed_batch(batch)
    params = init_policy(key, cfg)
    other_params = jax.tree.map(lambda value: value, params)
    other_params[-1]["b"] = other_params[-1]["b"] + jnp.array([0.2, -0.1, 0.15, -0.2])
    boxes, actions = jnp.zeros((2, cfg.history, 5)), jnp.zeros((2, cfg.history, 4))
    outputs = []
    for current_batch, current_params in (
        (batch, params),
        (changed, params),
        (changed, other_params),
    ):
        data = model.initialize(current_batch, cfg)
        hardware = H.initialize_hardware(data, current_batch, cfg, key, True)
        assert mujoco_rollout._native_kernels(cfg, True) is kernels
        output = kernels.hardware_policy_tick(
            current_params, current_batch, data, hardware, boxes, actions, key, jnp.array(0.4)
        )
        expected = _policy_reference(
            current_params, current_batch, data, hardware, boxes, actions, cfg, key, 0.4
        )
        _assert_tree_close(output, expected)
        outputs.append(output)
    assert not np.allclose(outputs[0][3], outputs[1][3])
    assert not np.allclose(outputs[1][4], outputs[2][4])
    data = model.initialize(batch, cfg)
    hardware = H.initialize_hardware(data, batch, cfg, key, True)
    repeated = kernels.hardware_policy_tick(
        params, batch, data, hardware, boxes, actions, key, jnp.array(0.4)
    )
    _assert_tree_close(repeated, outputs[0])


@pytest.mark.parametrize("noise", [False, True])
def test_fused_substep_matches_old_split_controller_actuator_and_sensor_order(noise: bool) -> None:
    cfg, key = _config(), jax.random.key(52)
    batch = _changed_batch(model.sample_batch(key, cfg, 2, "noisy"))
    plant = mujoco_rollout.NativePlant(batch, cfg)
    hardware = H.initialize_hardware(plant.data, batch, cfg, key, noise)
    issued = jnp.array([[0.2, -0.3, 0.4, 0.1], [-0.3, 0.1, -0.2, 0.3]])
    hardware = H.issue(hardware, issued, cfg, key, noise)
    kernels = mujoco_rollout._native_kernels(cfg, noise)
    physical = model.nominal_params(cfg)

    @jax.jit
    def old_before(
        data: SimData, hw: H.HardwareState, wind: jax.Array, rng: jax.Array
    ) -> tuple[SimData, H.HardwareState, jax.Array]:
        hw, executed, gust = H.actuate(hw, cfg, rng, noise)
        cmd = model.command(executed, cfg, physical)
        force = data.params.mass * (wind + gust)[:, None]
        data = data.replace(states=data.states.replace(force=force))
        return F.attitude_control(data, cmd[:, None]), hw, cmd

    command_key, imu_key = jax.random.split(key)
    old_data, old_hardware, old_cmd = old_before(
        plant.data, hardware, batch["domain"]["wind"], command_key
    )
    old_data, old_force, old_torque = kernels.actuators(old_data)
    data, updated_hardware, cmd, force, torque = kernels.hardware_substep(
        plant.data, hardware, batch["domain"]["wind"], command_key
    )
    _assert_tree_close((data, updated_hardware, cmd), (old_data, old_hardware, old_cmd))
    np.testing.assert_allclose(force, old_force, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(torque, old_torque, atol=2e-8, rtol=2e-5)
    assert int(updated_hardware.tick) == 1
    # The cache must not freeze the first world's disturbance or its actual mass.
    changed_wind = batch["domain"]["wind"] + jnp.array([0.1, -0.2, 0.3])
    changed_data, *_ = kernels.hardware_substep(plant.data, hardware, changed_wind, command_key)
    np.testing.assert_allclose(
        changed_data.states.force - data.states.force,
        plant.data.params.mass * jnp.array([0.1, -0.2, 0.3]),
        atol=2e-8,
    )
    # Native integration synchronizes the new pose before the same IMU key is sampled.
    plant.data = data
    plant._integrate(force, torque)
    sampled = kernels.after_physics(updated_hardware, plant.data, imu_key)
    expected_sample = H.sample_imu(old_hardware, plant.data, cfg, imu_key, noise)
    _assert_tree_close(sampled, expected_sample)
    np.testing.assert_allclose(
        model.state_vector(plant.data)[:, :3], [world.qpos[:3] for world in plant.worlds], atol=1e-7
    )


def test_native_plants_reuse_the_actuator_function_without_sharing_controller_state() -> None:
    cfg, key = Config(steps=3, hidden=8), jax.random.key(61)
    batch = model.sample_batch(key, cfg, 2, "nominal")
    first = mujoco_rollout.NativePlant(batch, cfg)
    second = mujoco_rollout.NativePlant(_changed_batch(batch), cfg)
    assert first._actuators is second._actuators
    _assert_tree_close(first.data.controls, second.data.controls)
    original = jax.tree.map(lambda value: value.copy(), second.data)
    first.advance(model.command(jnp.ones((2, 4)) * 0.2, cfg, model.nominal_params(cfg)))
    _assert_tree_close(second.data, original)
    assert second.worlds[0].time == 0
