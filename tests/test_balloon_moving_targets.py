"""Moving targets use one absolute clock for projection, native geometry and contact."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.balloon.config import Config
from crazyflow.balloon.evaluate import summarize_trajectories
from crazyflow.balloon.model import command, nominal_params, sample_batch
from crazyflow.balloon.mujoco_rollout import NativePlant, rollout
from crazyflow.balloon.noise_config import HardwareNoise
from crazyflow.balloon.observations import ideal_bbox
from crazyflow.balloon.play import SUITES, FrozenReplay
from crazyflow.balloon.policy import init_policy


def _stationary(samples: int = 3, episodes: int = 1) -> np.ndarray:
    states = np.zeros((samples, episodes, 12))
    states[..., 2] = 1
    return states


def test_moving_target_contacts_stationary_aircraft_at_relative_entry_time():
    result = summarize_trajectories(
        _stationary(),
        [[2, 0, 1]],
        [0.1],
        dt=1,
        target_velocities=[[-1, 0, 0]],
        drone_radius=0.1,
        collision_offset=(0, 0, 0),
    )
    record = result["per_episode"][0]
    assert result["successes"] == 1
    assert record["contact_time_s"] == pytest.approx(1.8)
    assert record["impact_speed_m_s"] == pytest.approx(1)
    assert record["drone_collision_speed_m_s"] == 0
    assert record["terminal_distance_m"] == pytest.approx(0.2)
    assert record["minimum_surface_clearance_m"] == pytest.approx(0)


def test_equal_target_and_aircraft_velocity_does_not_create_contact():
    states = _stationary()
    states[:, 0, 0] = [0, 1, 2]
    states[..., 6] = 1
    result = summarize_trajectories(
        states, [[1, 0, 1]], [0.1], dt=1, target_velocities=[[1, 0, 0]], collision_offset=(0, 0, 0)
    )
    record = result["per_episode"][0]
    assert record["outcome"] == "timeout"
    assert record["minimum_distance_m"] == pytest.approx(1)
    assert record["terminal_distance_m"] == pytest.approx(1)


def test_target_tunneling_is_swept_and_relative_speed_remains_diagnostic():
    result = summarize_trajectories(
        _stationary(samples=2),
        [[-1, 0, 1]],
        [0.1],
        dt=1,
        target_velocities=[[2, 0, 0]],
        drone_radius=0.1,
        collision_offset=(0, 0, 0),
        max_impact_speed=1.5,
    )
    record = result["per_episode"][0]
    assert record["outcome"] == "too_fast_contact" and record["success"]
    assert record["contact_time_s"] == pytest.approx(0.4)
    assert record["impact_speed_m_s"] == pytest.approx(2)
    assert record["drone_collision_speed_m_s"] == 0


def test_tracking_failure_precedes_later_moving_contact_and_truncates_distances():
    result = summarize_trajectories(
        _stationary(),
        [[2, 0, 1]],
        [0.1],
        dt=1,
        target_velocities=[[-1, 0, 0]],
        drone_radius=0.1,
        collision_offset=(0, 0, 0),
        visible=np.zeros((20, 1), dtype=bool),
        visibility_dt=0.1,
        tracking_timeout=0.5,
    )
    record = result["per_episode"][0]
    assert record["outcome"] == "tracking_lost" and not record["success"]
    assert record["event_time_s"] == pytest.approx(0.5)
    assert record["contact_time_s"] is None
    assert record["minimum_distance_m"] == pytest.approx(1.5)
    assert record["terminal_distance_m"] == pytest.approx(1.5)


def test_omitted_velocity_and_explicit_zero_velocity_score_identically():
    arguments = (_stationary(), [[0.5, 0, 1]], [0.1])
    assert summarize_trajectories(*arguments, dt=0.1) == summarize_trajectories(
        *arguments, dt=0.1, target_velocities=np.zeros((1, 3))
    )


def test_stratification_preserves_all_episodes_including_nonstandard_ranges():
    velocities = np.zeros((5, 3))
    velocities[:, 0] = [0, 0.2, 0.75, 1.25, 2]
    result = summarize_trajectories(
        _stationary(episodes=5),
        np.tile([0, 0, 1], (5, 1)),
        np.array([0.03, 0.1, 0.3, 0.6, 0.8]) / 2,
        dt=0.1,
        target_velocities=velocities,
    )
    grouped = result["stratified"]
    assert grouped["motion"]["static"]["episodes"] == 1
    assert grouped["motion"]["moving"]["episodes"] == 4
    assert grouped["target_speed_m_s"]["[0,0.5)"]["episodes"] == 2
    assert grouped["target_diameter_m"]["[0.3,0.6]"]["episodes"] == 2
    for name in ("motion", "target_speed_m_s", "target_diameter_m"):
        assert sum(bucket["episodes"] for bucket in grouped[name].values()) == 5
        assert sum(bucket["contacts"] for bucket in grouped[name].values()) == 5
    assert grouped["target_diameter_m"]["outside_bins"]["episodes"] == 1


def test_playback_target_uses_elapsed_time_and_replay_keeps_same_random_sample():
    cfg = Config(steps=3, hidden=8)
    values = {
        "states": _stationary(samples=4),
        "substep_states": _stationary(samples=31),
        "actions": np.zeros((3, 1, 4)),
        "bbox": np.tile([0, 0, 0.1, 0.1, 1], (3, 1, 1)),
        "initial_target": np.array([[2.0, 0.0, 1.4]]),
        "initial_radius": np.array([0.1]),
        "target_velocity": np.array([[0.5, -0.2, 0.1]]),
    }
    scores = {
        name: {"per_episode": [{"event_time_s": 0.06, "outcome": "timeout"}]} for name in SUITES
    }
    replay = FrozenReplay(cfg, {name: values.copy() for name in SUITES}, scores, 42)
    replay.elapsed = 0.04
    np.testing.assert_allclose(replay.target, values["initial_target"][0] + [0.02, -0.008, 0.004])
    np.testing.assert_allclose(replay.telemetry()["truth_target_position_m"], replay.target)
    arrays = replay.arrays
    replay.reset()
    assert replay.seed == 42 and replay.arrays is arrays
    np.testing.assert_array_equal(replay.target, replay.initial_target)
    del replay.current["target_velocity"]
    replay.elapsed = 0.04
    np.testing.assert_array_equal(replay.target, replay.initial_target)


@pytest.mark.parametrize("hardware", [False, True])
def test_native_target_pose_and_policy_box_follow_the_same_absolute_clock(hardware: bool):
    cfg = replace(Config(steps=3, hidden=8), hardware=HardwareNoise(enabled=hardware))
    key = jax.random.key(13)
    batch = sample_batch(key, cfg, 1, "nominal")
    batch["state"] = jnp.zeros((1, 12)).at[:, 2].set(1.5)
    batch["target"] = jnp.array([[2.0, 0.0, 1.5]])
    batch["target_velocity"] = jnp.array([[0.3, 0.4, -0.1]])
    batch["radius"] = jnp.array([0.1])
    plant = NativePlant(batch, cfg)
    assert (plant.models[0].nmocap, plant.models[0].nq, plant.models[0].nv) == (1, 7, 6)
    plant.advance(command(jnp.zeros((1, 4)), cfg, nominal_params(cfg)))
    balloon_id = plant.models[0].body("balloon").id
    expected = np.asarray(batch["target"] + batch["target_velocity"] * cfg.dt)[0]
    np.testing.assert_allclose(plant.worlds[0].xpos[balloon_id], expected, atol=1e-7)
    assert plant.worlds[0].time == pytest.approx(cfg.dt)

    trajectory = rollout(init_policy(key, cfg), batch, cfg, key, noise=False)
    for tick in range(cfg.steps):
        state = trajectory["states"][tick, 0]
        target = batch["target"][0] + batch["target_velocity"][0] * (tick * cfg.dt)
        box, _ = ideal_bbox(state[:3], state[3:6], target, batch["radius"][0], cfg)
        np.testing.assert_allclose(trajectory["bbox"][tick, 0], box, atol=1e-6)
