"""Frozen replay diagnostics retain actor data and align the two sample timelines."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from crazyflow.balloon.config import Config
from crazyflow.balloon.evaluate import control_telemetry
from crazyflow.balloon.noise_config import HardwareNoise
from crazyflow.balloon.play import SUITES, FrozenReplay, debug_panel


def _replay() -> FrozenReplay:
    cfg = Config(steps=3)
    states = np.zeros((4, 1, 12))
    states[:, 0, 3] = np.arange(4) / 10
    data = {
        "states": states,
        "substep_states": np.repeat(states, cfg.substeps, axis=0),
        "bbox": np.array([[[0.99, 0, 0.2, 0.2, 1]], [[0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0]]]),
        "actions": np.repeat(np.arange(3, dtype=float)[:, None, None] / 10, 4, axis=-1),
        "issued_commands": np.repeat(np.arange(3, dtype=float)[:, None, None], 4, axis=-1),
        "executed_commands": np.repeat((np.arange(3, dtype=float) - 1)[:, None, None], 4, axis=-1),
        "body_gyro": np.repeat((100 + np.arange(4))[:, None, None], 3, axis=-1),
        "rotor_rpm": np.repeat((1000 + 100 * np.arange(4))[:, None, None], 4, axis=-1),
        "sensor_delay_steps": np.array([2]),
        "action_delay_steps": np.array([1]),
        "initial_target": np.array([[2, 0, 1.4]]),
        "initial_radius": np.array([0.15]),
    }
    scores = {
        name: {"per_episode": [{"event_time_s": 0.06, "outcome": "timeout"}]} for name in SUITES
    }
    return FrozenReplay(cfg, {name: data.copy() for name in SUITES}, scores, 7)


@pytest.mark.parametrize("when,tick", [(0, 0), (0.01999, 0), (0.02, 1), (0.03999, 1), (0.04, 2)])
def test_debug_sampling_uses_current_tick_and_pre_command_state(when: float, tick: int):
    replay = _replay()
    replay.elapsed = when
    info = replay.telemetry()
    assert info["control_tick"] == tick
    assert info["control_time_s"] == pytest.approx(tick * 0.02)
    assert info["truth_time_s"] == pytest.approx(tick * 0.02)
    assert info["truth_rpy_rad"][0] == pytest.approx(tick / 10)
    assert info["truth_body_gyro_rad_s"] == [100 + tick] * 3
    assert info["truth_rotor_rpm"] == [1000 + 100 * tick] * 4
    assert info["issued_action"] == pytest.approx([tick / 10] * 4)
    assert info["executed_command"] == pytest.approx([tick - 1] * 4)


def test_endpoint_truth_and_last_held_command_remain_distinct():
    replay = _replay()
    replay.elapsed = replay.duration
    info = replay.telemetry()
    assert info["control_tick"] == 2
    assert info["truth_time_s"] == pytest.approx(0.06)
    assert info["truth_rotor_rpm"] == [1300] * 4
    assert info["issued_action"] == pytest.approx([0.2] * 4)


def test_received_box_is_exact_and_unseen_status_uses_time_only():
    replay = _replay()
    info = replay.telemetry()
    np.testing.assert_array_equal(info["received_bbox"], replay.current["bbox"][0, 0])
    assert info["received_bbox"][0] + info["received_bbox"][2] / 2 > 1
    replay.elapsed = 0.03
    info = replay.telemetry()
    assert not info["received_valid"]
    assert info["unseen_seconds"] == pytest.approx(0.01)
    label = debug_panel(replay, info, ideal_visible=True)
    assert "Policy box LOST" in label and "ideal FOV visible" in label
    assert "SIM TRUTH" in label and "NOT actor inputs" in label
    assert "Issued RPY" in label and "Executed RPY" in label


def test_derived_command_fifo_covers_zero_one_and_two_tick_delays():
    cfg = replace(Config(steps=3), action_delay_steps=2, max_delay_steps=2)
    actions = np.zeros((3, 64, 4), dtype=np.float32)
    actions[:, :, 0] = np.array([0.1, 0.2, 0.3])[:, None]
    untouched = actions.copy()
    info = control_telemetry({"actions": actions}, cfg, jax.random.PRNGKey(7), noise=True)
    np.testing.assert_array_equal(actions, untouched)
    assert set(info["action_delay_steps"].tolist()) == {0, 1, 2}
    for delay in (0, 1, 2):
        episode = np.flatnonzero(info["action_delay_steps"] == delay)[0]
        if delay:
            np.testing.assert_allclose(info["executed_commands"][:delay, episode, :3], 0)
            # Empty FIFO entries mean hover, not zero newtons.
            assert np.all(info["executed_commands"][:delay, episode, 3] > 0.3)
        np.testing.assert_allclose(
            info["executed_commands"][delay:, episode],
            info["issued_commands"][: 3 - delay, episode],
        )
    nominal = control_telemetry({"actions": actions}, cfg, jax.random.PRNGKey(7), noise=False)
    np.testing.assert_array_equal(nominal["executed_commands"], nominal["issued_commands"])
    np.testing.assert_array_equal(nominal["sensor_delay_steps"], 0)
    np.testing.assert_array_equal(nominal["action_delay_steps"], 0)


def test_hardware_telemetry_uses_recorded_delivery_and_current_tick():
    replay = _replay()
    replay.cfg = replace(replay.cfg, hardware=HardwareNoise(enabled=True))
    current = replay.current
    current.update(
        camera_age_ms=np.array([[np.inf], [20.0], [40.0]]),
        gyro_age_ms=np.array([[0.0], [2.0], [4.0]]),
        attitude_age_ms=np.array([[0.0], [4.0], [6.0]]),
        command_delay_ms=np.broadcast_to([2.0, 4.0, 6.0, 8.0], (3, 1, 4)).copy(),
    )
    recorded = control_telemetry(current, replay.cfg, jax.random.key(99), noise=True)
    assert "action_delay_steps" not in recorded
    for key, value in recorded.items():
        np.testing.assert_array_equal(value, current[key])
    replay.elapsed = 0.02
    info = replay.telemetry()
    assert (info["camera_age_ms"], info["gyro_age_ms"], info["attitude_age_ms"]) == (20, 2, 4)
    assert info["command_delay_ms"] == [2, 4, 6, 8]
    panel = debug_panel(replay, info, ideal_visible=True)
    assert "age camera/gyro/attitude [ms] 20 / 2 / 4" in panel
    assert "Scheduled RPY/T delay" in panel
