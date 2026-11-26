"""Frozen replay diagnostics retain actor data and align the two sample timelines."""

from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from threading import Event
from time import monotonic, sleep

import jax
import numpy as np
import pytest

from crazyflow.balloon.config import Config
from crazyflow.balloon.evaluate import control_telemetry
from crazyflow.balloon.noise_config import HardwareNoise
from crazyflow.balloon.play import (
    SUITES,
    AsyncFlightGenerator,
    FrozenReplay,
    _GifRecorder,
    debug_panel,
)


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


def test_gif_keeps_tiny_role_colors_and_final_hold(tmp_path: Path):
    """Small orange/red objects must survive a palette dominated by blue-grey scenery."""
    from PIL import Image

    shades = np.linspace(15, 200, 512 * 256, dtype=np.uint8).reshape(256, 512)
    rgb = np.stack((shades // 2, shades * 3 // 4, shades), axis=-1)
    roles = np.array([[38, 224, 255], [255, 92, 31], [255, 0, 0]], dtype=np.uint8)
    rgb[10, 10:13] = roles
    path = tmp_path / "roles.gif"
    recorder = _GifRecorder(path, np.array([50, 50, 1000]))
    recorder.append_data(rgb)
    second = rgb.copy()
    second[20, 20] = [255, 255, 255]
    recorder.append_data(second)
    recorder.append_data(second)
    recorder.close()
    with Image.open(path) as encoded:
        durations = []
        for frame in range(encoded.n_frames):
            encoded.seek(frame)
            np.testing.assert_array_equal(np.asarray(encoded.convert("RGB"))[10, 10:13], roles)
            durations.append(encoded.info["duration"])
    assert sum(durations) == 1100
    assert durations[-1] == 1050  # Identical tail frames merge without losing the hold.


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


def test_random_reset_runs_in_background_without_mutating_the_displayed_flight(tmp_path: Path):
    replay = _replay()
    replay.checkpoint = tmp_path / "policy.npz"
    replay.plant = "mujoco"
    replay.elapsed = 0.03
    entered, release = Event(), Event()
    calls = []

    def delayed_generate(checkpoint: Path, seed: int, *, plant: str) -> FrozenReplay:
        calls.append((checkpoint, seed, plant))
        entered.set()
        assert release.wait(5)
        fresh = _replay()
        fresh.seed = seed
        fresh.arrays[fresh.name]["initial_target"] = np.array([[2.5, 0.1, 1.4]])
        return fresh

    loader = AsyncFlightGenerator(delayed_generate)
    target = replay.target.copy()
    try:
        assert loader.request(replay)
        assert entered.wait(5)
        assert loader.busy and loader.poll() is None
        assert not loader.request(replay)
        np.testing.assert_array_equal(replay.target, target)
        assert replay.elapsed == 0.03 and replay.seed == 7
        release.set()
        loader.future.result(timeout=5)
        fresh = loader.poll()
        assert fresh.seed == 8 and not np.array_equal(fresh.target, target)
        assert not loader.busy and loader.error is None
        assert calls == [(replay.checkpoint, 8, "mujoco")]
    finally:
        release.set()
        loader.close()


def test_random_reset_reports_worker_errors_without_replacing_a_good_flight(tmp_path: Path):
    replay = _replay()
    replay.checkpoint = tmp_path / "policy.npz"

    def fail_generate(checkpoint: Path, seed: int, *, plant: str) -> FrozenReplay:
        raise ValueError("invalid checkpoint fixture")

    loader = AsyncFlightGenerator(fail_generate)
    try:
        assert loader.request(replay)
        with pytest.raises(ValueError, match="fixture"):
            loader.future.result(timeout=5)
        assert loader.poll() is None
        assert not loader.busy and "invalid checkpoint fixture" in loader.error
        assert replay.seed == 7
    finally:
        loader.close()


def test_random_reset_requires_a_checkpoint_and_close_does_not_wait_for_rollout(tmp_path: Path):
    replay = _replay()
    loader = AsyncFlightGenerator()
    assert not loader.request(replay)
    assert "policy.npz" in loader.error and not loader.busy
    # A running rollout cannot be cancelled; close still returns immediately.
    pending = Future()
    pending.set_running_or_notify_cancel()
    loader.future = pending
    loader.close()
    replay.checkpoint = tmp_path / "policy.npz"
    assert not loader.request(replay)
    assert not pending.done()


def test_spawned_reset_reports_load_failure_and_can_retry(tmp_path: Path):
    replay = _replay()
    replay.checkpoint = tmp_path / "missing-policy.npz"
    target = replay.target.copy()
    loader = AsyncFlightGenerator()
    try:
        for _ in range(2):
            assert loader.request(replay)
            deadline = monotonic() + 30
            while loader.busy and monotonic() < deadline:
                assert loader.poll() is None
                sleep(0.01)
            assert not loader.busy
            assert "FileNotFoundError" in loader.error
            assert "missing-policy.npz" in loader.error
            assert replay.seed == 7
            np.testing.assert_array_equal(replay.target, target)
    finally:
        loader.close()


def test_replay_rewinds_identical_target_without_resampling():
    replay = _replay()
    target, state = replay.target.copy(), replay.state().copy()
    replay.elapsed, replay.paused = replay.duration, True
    replay.reset()
    assert replay.elapsed == 0 and not replay.paused and replay.seed == 7
    np.testing.assert_array_equal(replay.target, target)
    np.testing.assert_array_equal(replay.state(), state)
