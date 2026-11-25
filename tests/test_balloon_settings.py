"""Reject ambiguous training YAML while preserving complete official task defaults."""

import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from crazyflow.balloon.config import Config
from crazyflow.balloon.noise_config import HardwareNoise
from crazyflow.balloon.settings import (
    DEFAULT_SETTINGS_PATH,
    NOISE_FIELDS,
    OBJECTIVE_FIELDS,
    TASK_FIELDS,
    load_settings,
    resolve_batch_size,
)


def _settings(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "training.yaml"
    path.write_text(text)
    return path


def test_shipped_yaml_covers_config_and_explicitly_enables_the_new_hardware_pipeline():
    training, overrides = load_settings()
    assert training == {
        "iterations": 1500,
        "batch_size": {"gpu": 1024, "cpu": 32},
        "seed": 42,
        "lr": 0.001,
    }
    excluded = {
        "hfov_deg",
        "image_width",
        "image_height",
        "camera_offset",
        "camera_rpy",
        "camera_intrinsics",
        "model_params",
        "controller_params",
    }
    assert set(overrides) == {field.name for field in fields(Config)} - excluded
    assert set(overrides) == TASK_FIELDS | NOISE_FIELDS | OBJECTIVE_FIELDS
    defaults = Config()
    assert {key: value for key, value in overrides.items() if key != "hardware"} == {
        name: getattr(defaults, name) for name in overrides if name != "hardware"
    }
    # A legacy checkpoint has no hardware field; new training opts in via YAML.
    assert not defaults.hardware.enabled
    assert overrides["hardware"] == HardwareNoise(
        enabled=True,
        camera_delay_ms=(0, 40),
        attitude_command_delay_ms=(0, 20),
        thrust_command_delay_ms=(0, 20),
    )
    assert load_settings(DEFAULT_SETTINGS_PATH) == (training, overrides)


def test_partial_hardware_settings_merge_by_field_and_survive_checkpoint_json(tmp_path: Path):
    _, overrides = load_settings(
        _settings(
            tmp_path,
            """
noise:
  hardware:
    camera_delay_ms: [4, 18]
    gyro_delay_ms: [2, 6]
    gyro_scale_std: 0.01
    inertia_fraction: 0.07
""",
        )
    )
    hardware = overrides["hardware"]
    assert hardware.enabled
    assert hardware.camera_delay_ms == (4, 18)
    assert hardware.gyro_delay_ms == (2, 6)
    assert hardware.gyro_scale_std == 0.01 and hardware.inertia_fraction == 0.07
    assert hardware.thrust_command_delay_ms == (0, 20)
    cfg = Config(**overrides)
    assert Config(**json.loads(json.dumps(cfg.to_dict()))) == cfg
    with pytest.raises(FrozenInstanceError):
        hardware.gyro_scale_std = 1
    assert load_settings()[1]["hardware"].gyro_scale_std == 0


@pytest.mark.parametrize(
    "values,match",
    [
        ({"enabled": "false"}, "boolean"),
        ({"gyro_scale_std": True}, "finite"),
        ({"gust_std": float("nan")}, "finite"),
        ({"gyro_drift_std": -0.1}, "nonnegative"),
        ({"camera_delay_ms": [2, 1]}, "low <= high"),
        ({"gyro_delay_ms": [-1, 2]}, "0 <= low"),
        ({"attitude_delay_ms": [1]}, "minimum and maximum"),
        ({"thrust_command_delay_ms": [0, float("inf")]}, "finite"),
        ({"camera_delay_ms": [0, 101]}, "camera_stale_ms"),
        ({"gyro_packet_loss_prob": 1}, r"\[0,1\)"),
        ({"attitude_packet_loss_prob": -0.1}, r"\[0,1\)"),
        ({"inertia_fraction": 2 / 3}, "2/3"),
        ({"motor_response_fraction": float("inf")}, "finite"),
        ({"bias_tau_s": 0}, "positive"),
        ({"command_jitter_tau_s": -1}, "positive"),
        ({"gust_tau_s": 0}, "positive"),
    ],
)
def test_invalid_hardware_contracts_are_rejected(values: dict, match: str):
    with pytest.raises(ValueError, match=match):
        Config(hardware=values)


def test_custom_yaml_merges_only_declared_values_and_nested_batch_or_loss_keys(tmp_path: Path):
    training, overrides = load_settings(
        _settings(
            tmp_path,
            """
training:
  iterations: 17
  batch_size: {cpu: 7}
task: {steps: 25, collision_offset: [0.01, -0.02, 0.03]}
noise: {gyro_noise_std: 0.03}
objective:
  loss_weights: {fov: 1.2, action: 0}
  ground_margin: 0.2
""",
        )
    )
    assert training["iterations"] == 17
    assert resolve_batch_size(training, "cpu") == 7
    assert resolve_batch_size(training, "gpu") == 1024
    assert overrides["steps"] == 25
    assert overrides["collision_offset"] == (0.01, -0.02, 0.03)
    assert overrides["gyro_noise_std"] == 0.03
    assert overrides["loss_weights"]["fov"] == 1.2
    assert overrides["loss_weights"]["action"] == 0
    assert overrides["loss_weights"]["contact"] == Config().loss_weights["contact"]
    assert overrides["ground_margin"] == 0.2
    # One caller's result cannot mutate subsequent defaults.
    overrides["loss_weights"]["fov"] = 999
    assert load_settings()[1]["loss_weights"]["fov"] == Config().loss_weights["fov"]


@pytest.mark.parametrize(
    "text,match",
    [
        ("camera: {hfov_deg: 120}", "Unknown"),
        ("task: {camera_offset: [0, 0, 0]}", "Unknown"),
        ("task: {model_params: {mass: 0.04}}", "Unknown"),
        ("training: {controller_params: 'x.toml'}", "Unknown"),
        ("noise: {unknown: 1}", "Unknown"),
        ("noise: {hardware: {unknown: 1}}", "Unknown"),
        ("noise: {hardware: null}", "mapping"),
        ("noise: {hardware: {enabled: 1}}", "boolean"),
        ("noise: {hardware: {gyro_scale_std: true}}", "finite"),
        ("objective: {loss_weights: {unknown: 1}}", "Unknown"),
        ("training: {batch_size: {auto: 3}}", "Unknown"),
        ("training: {iterations: true}", "integer"),
        ("training: {iterations: 4.5}", "integer"),
        ("training: {batch_size: {cpu: 0}}", "integer"),
        ("training: {batch_size: 32}", "mapping"),
        ("training: {lr: .nan}", "finite"),
        ("training: {lr: 0}", "positive"),
        ("task: {dt: .inf}", "finite"),
        ("task: {dt: 0.01}", "500 Hz"),
        ("task: {hidden: false}", "integer"),
        ("task: {collision_offset: [0, 0]}", "three"),
        ("task: {collision_offset: [0, true, 0]}", "finite"),
        ("noise: {max_delay_steps: 1.5}", "integer"),
        ("noise: {bbox_noise_std: -0.1}", "nonnegative"),
        ("noise: {dropout_prob: 1}", "dropout"),
        ("noise: {mass_fraction: 0.9}", "2/3"),
        ("objective: {softmin_temperature: 0}", "positive"),
        ("objective: {loss_weights: {distance: -.inf}}", "finite"),
        ("objective: {loss_weights: {distance: -1}}", "nonnegative"),
        ("training: {seed: '42'}", "integer"),
        ("training: {seed: 1, seed: 2}", "Duplicate"),
        ("task: {}\ntask: {}", "Duplicate"),
        ("objective: {loss_weights: {fov: 1, fov: 2}}", "Duplicate"),
        ("task: !!python/object:builtins.object {}", "Invalid"),
        ("[1, 2]", "mapping"),
        ("training: null", "mapping"),
    ],
)
def test_invalid_or_ambiguous_settings_are_rejected(tmp_path: Path, text: str, match: str):
    with pytest.raises(ValueError, match=match):
        load_settings(_settings(tmp_path, text))


def test_device_batch_is_selected_after_actual_device_resolution():
    training, _ = load_settings()
    assert resolve_batch_size(training, "cpu") == 32
    assert resolve_batch_size(training, "gpu") == 1024
    with pytest.raises(ValueError, match="Resolve device"):
        resolve_batch_size(training, "auto")
