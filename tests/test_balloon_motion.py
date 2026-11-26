"""Moving targets use one absolute clock without privileged actor inputs."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from crazyflow.balloon.config import Config
from crazyflow.balloon.model import sample_batch, target_at_time
from crazyflow.balloon.noise_config import HardwareNoise
from crazyflow.balloon.observations import feature_dim, ideal_bbox
from crazyflow.balloon.policy import init_policy
from crazyflow.balloon.rollout import rollout, trajectory_loss
from crazyflow.balloon.settings import load_settings


def test_canonical_size_and_partial_settings_preserve_legacy_radius_contract(tmp_path: Path):
    cfg = Config(target_diameter_m=0.3, target_scale_range=[0.1, 2])
    assert cfg.radius_min == pytest.approx(0.015) and cfg.radius_max == pytest.approx(0.3)
    assert Config(**cfg.to_dict()) == cfg
    path = tmp_path / "custom.yaml"
    path.write_text("task: {target_scale_range: [0.2, 1]}\n")
    _, values = load_settings(path)
    assert values["radius_min"] == pytest.approx(0.03)
    assert values["radius_max"] == pytest.approx(0.15)
    path.write_text("task: {radius_min: 0.13, radius_max: 0.18}\n")
    _, values = load_settings(path)
    assert values["target_diameter_m"] is None and values["target_scale_range"] is None
    assert (values["radius_min"], values["radius_max"]) == (0.13, 0.18)
    path.write_text("task: {radius_min: 0.1, target_diameter_m: 0.3}\n")
    with pytest.raises(ValueError, match="canonical"):
        load_settings(path)


@pytest.mark.parametrize(
    "values",
    [
        {"target_speed_min": -1},
        {"target_speed_max": float("nan")},
        {"target_speed_min": 2, "target_speed_max": 1},
        {"target_stationary_prob": 1.1},
        {"target_spawn_margin": -0.1},
        {"target_diameter_m": 0.3},
        {"target_diameter_m": 0, "target_scale_range": [0.1, 2]},
        {"target_diameter_m": 0.3, "target_scale_range": [2, 0.1]},
    ],
)
def test_invalid_motion_or_size_parameters_are_rejected(values: dict):
    with pytest.raises(ValueError):
        Config(**values)


def test_motion_sampling_is_isotropic_and_spawn_lift_preserves_relative_start_geometry():
    cfg = Config(**load_settings()[1])
    key = jax.random.key(411)
    batch = sample_batch(key, cfg, 4096, "nominal")
    velocity = np.asarray(batch["target_velocity"])
    speed = np.linalg.norm(velocity, axis=-1)
    moving = speed > 1e-6
    assert 0.17 < np.mean(~moving) < 0.23
    assert speed.max() <= 1.5 + 1e-6
    directions = velocity[moving] / speed[moving, None]
    np.testing.assert_allclose(directions.mean(axis=0), 0, atol=0.035)
    np.testing.assert_allclose((directions**2).mean(axis=0), 1 / 3, atol=0.025)
    log_radius = np.log(np.asarray(batch["radius"]))
    assert log_radius.min() >= np.log(0.015) - 1e-6
    assert log_radius.max() <= np.log(0.3) + 1e-6
    assert abs(log_radius.mean() - (np.log(0.015) + np.log(0.3)) / 2) < 0.04
    end = target_at_time(batch, cfg.steps * cfg.dt)
    minimum_surface = jnp.minimum(batch["target"][:, 2], end[:, 2]) - batch["radius"]
    assert np.asarray(minimum_surface).min() >= cfg.min_altitude + cfg.target_spawn_margin - 1e-6
    assert (velocity[:, 2] < -0.5).any() and (velocity[:, 2] > 0.5).any()
    old = sample_batch(key, Config(), 4096, "nominal")
    np.testing.assert_allclose(
        batch["target"] - batch["state"][:, :3], old["target"] - old["state"][:, :3], atol=1e-6
    )


def test_absolute_target_clock_and_legacy_missing_velocity():
    batch = sample_batch(jax.random.key(1), Config(), 2, "nominal")
    batch["target_velocity"] = jnp.array([[0.2, -0.3, 0.4], [-0.4, 0.1, -0.2]])
    times = jnp.array([0, 0.02, 3.0])
    positions = target_at_time(batch, times)
    np.testing.assert_allclose(positions[0], batch["target"])
    np.testing.assert_allclose(
        positions[-1] - positions[0], 3 * batch["target_velocity"], atol=2e-7
    )
    legacy = {key: value for key, value in batch.items() if key != "target_velocity"}
    np.testing.assert_array_equal(target_at_time(legacy, 3.0), batch["target"])


@pytest.mark.parametrize("hardware_enabled", [False, True])
def test_bbox_capture_uses_moving_center_at_start_of_the_same_policy_tick(hardware_enabled: bool):
    cfg = Config(steps=4, hidden=8, hardware=HardwareNoise(enabled=hardware_enabled))
    batch = sample_batch(jax.random.key(2), cfg, 1, "nominal")
    batch["target_velocity"] = jnp.array([[0.1, 0.3, -0.2]])
    params = init_policy(jax.random.key(3), cfg)
    result = jax.jit(lambda: rollout(params, batch, cfg, jax.random.key(4), noise=False))()
    for tick in range(cfg.steps):
        state = result["states"][tick]
        expected, _ = ideal_bbox(
            state[:, :3],
            state[:, 3:6],
            batch["target"] + batch["target_velocity"] * (tick * cfg.dt),
            batch["radius"],
            cfg,
        )
        np.testing.assert_allclose(result["bbox"][tick], expected, atol=2e-7)
    np.testing.assert_allclose(
        result["target_positions"][-1],
        batch["target"] + batch["target_velocity"] * cfg.steps * cfg.dt,
    )
    assert feature_dim(cfg) == feature_dim(Config())


def test_loss_matches_moving_target_endpoints_and_relative_impact_velocity():
    weights = {name: 0.0 for name in Config().loss_weights}
    weights.update(distance=1.0, fov=0.8, impact=1.0)
    cfg = Config(steps=4, collision_offset=(0, 0, 0), max_impact_speed=0.1, loss_weights=weights)
    batch = sample_batch(jax.random.key(5), cfg, 1, "nominal")
    batch["target"] = jnp.array([[2.0, 0, 1.4]])
    times = jnp.arange(cfg.steps + 1) * cfg.dt
    still = jnp.zeros((cfg.steps + 1, 1, 12)).at[:, :, :3].set(jnp.array([1.0, 0, 1.4]))
    actions = jnp.zeros((cfg.steps, 1, 4))
    static_loss = trajectory_loss(still, still, actions, batch, cfg)
    velocity = jnp.array([[0.0, 0.3, 0.4]])
    moving_batch = {**batch, "target_velocity": velocity}
    moving = still.at[:, :, :3].add(times[:, None, None] * velocity)
    moving = moving.at[:, :, 6:9].set(velocity)
    np.testing.assert_allclose(
        trajectory_loss(moving, moving, actions, moving_batch, cfg), static_loss, atol=2e-6
    )
    lagging = moving.at[:, :, 6:9].set(0)
    # A stationary CoM relative to a 0.5 m/s target exceeds the 0.1 m/s limit.
    expected_penalty = (0.5**2 - cfg.max_impact_speed**2) ** 2
    np.testing.assert_allclose(
        trajectory_loss(lagging, lagging, actions, moving_batch, cfg) - static_loss,
        expected_penalty,
        atol=2e-6,
    )


def test_smallest_target_remains_valid_below_two_pixels():
    cfg = Config(target_diameter_m=0.3, target_scale_range=(0.1, 2))
    box, visible = ideal_bbox(
        jnp.array([[0.0, 0, 1.4]]),
        jnp.zeros((1, 3)),
        jnp.array([[3.0, 0, 1.4]]),
        jnp.array([cfg.radius_min]),
        cfg,
    )
    assert bool(visible[0]) and float(box[0, 4]) == 1
    assert 0 < float(box[0, 2]) * cfg.image_width / 2 < 2


def test_moving_target_full_bptt_has_finite_frozen_rng_directional_gradient():
    cfg = Config(
        steps=6,
        hidden=8,
        target_speed_min=0.2,
        target_speed_max=0.6,
        target_diameter_m=0.3,
        target_scale_range=(0.5, 2),
        hardware=HardwareNoise(enabled=True, camera_delay_ms=(0, 4), gyro_scale_std=0.1),
    )
    batch = sample_batch(jax.random.key(31), cfg, 2, "noisy")
    params = init_policy(jax.random.key(32), cfg)
    flat, unpack = ravel_pytree(params)
    noise_key = jax.random.key(33)
    loss = jax.jit(lambda vector: rollout(unpack(vector), batch, cfg, noise_key)["loss"])
    value, gradient = jax.jit(jax.value_and_grad(loss))(flat)
    direction = jax.random.normal(jax.random.key(34), flat.shape)
    direction /= jnp.linalg.norm(direction)
    epsilon = 0.003
    finite_difference = (loss(flat + epsilon * direction) - loss(flat - epsilon * direction)) / (
        2 * epsilon
    )
    assert np.isfinite(value) and np.isfinite(gradient).all()
    assert float(jnp.linalg.norm(gradient)) > 1e-5
    np.testing.assert_allclose(
        jnp.dot(gradient, direction), finite_difference, atol=0.01, rtol=0.05
    )
