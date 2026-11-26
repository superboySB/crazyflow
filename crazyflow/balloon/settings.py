"""Validated YAML training settings, separate from camera and physical calibration."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import yaml

from crazyflow.balloon.config import Config
from crazyflow.balloon.noise_config import HARDWARE_FIELDS, HardwareNoise
from crazyflow.paths import resource_path

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_SETTINGS_PATH = resource_path("config/training.yaml")
TRAINING_FIELDS = frozenset({"iterations", "batch_size", "seed", "lr"})
TASK_FIELDS = frozenset(
    {
        "dt",
        "substeps",
        "steps",
        "history",
        "hidden",
        "collision_offset",
        "drone_radius",
        "radius_min",
        "radius_max",
        "target_diameter_m",
        "target_scale_range",
        "target_speed_min",
        "target_speed_max",
        "target_stationary_prob",
        "target_spawn_margin",
        "distance_min",
        "distance_max",
        "max_impact_speed",
        "min_altitude",
        "max_range",
        "max_unseen_seconds",
        "max_tilt",
        "max_yaw",
    }
)
NOISE_FIELDS = frozenset(
    {
        "bbox_noise_std",
        "gyro_noise_std",
        "gyro_bias_std",
        "attitude_noise_std",
        "dropout_prob",
        "max_delay_steps",
        "action_delay_steps",
        "mass_fraction",
        "thrust_fraction",
        "response_fraction",
        "wind_std",
        "hardware",
    }
)
OBJECTIVE_FIELDS = frozenset(
    {
        "loss_weights",
        "softmin_temperature",
        "impact_temperature",
        "contact_temperature",
        "ground_margin",
        "tilt_soft_limit",
    }
)
LOSS_FIELDS = frozenset(
    {
        "distance",
        "contact",
        "speed",
        "impact",
        "fov",
        "ground",
        "tilt",
        "angular_rate",
        "action",
        "smoothness",
    }
)
SECTION_FIELDS = {
    "training": TRAINING_FIELDS,
    "task": TASK_FIELDS,
    "noise": NOISE_FIELDS,
    "objective": OBJECTIVE_FIELDS,
}
INTEGER_FIELDS = frozenset(
    {"substeps", "steps", "history", "hidden", "max_delay_steps", "action_delay_steps"}
)
NONNEGATIVE_FIELDS = (NOISE_FIELDS - {"hardware"}) | {
    "min_altitude",
    "max_unseen_seconds",
    "ground_margin",
    "target_speed_min",
    "target_speed_max",
    "target_stationary_prob",
    "target_spawn_margin",
}


class _UniqueSafeLoader(yaml.SafeLoader):
    """Keep SafeLoader's restricted tags and reject silent mapping-key replacement."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Settings YAML mapping keys must be strings")
            if key in result:
                raise ValueError(f"Duplicate YAML key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _keys(
    value: dict,
    allowed: frozenset | set,
    name: str,
    complete: bool,
    optional: frozenset = frozenset(),
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
    if complete and (missing := allowed - set(value) - optional):
        raise ValueError(f"Missing default {name} fields: {sorted(missing)}")


def _read(path: Path, complete: bool = False) -> dict:
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueSafeLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid settings YAML in {path}: {error}") from error
    document = _mapping(document, "Settings")
    _keys(document, set(SECTION_FIELDS), "settings sections", complete)
    for section, values in document.items():
        values = _mapping(values, section)
        optional = frozenset({"radius_min", "radius_max"}) if section == "task" else frozenset()
        _keys(values, SECTION_FIELDS[section], section, complete, optional)
        if section == "training" and "batch_size" in values:
            batch = _mapping(values["batch_size"], "training.batch_size")
            _keys(batch, {"cpu", "gpu"}, "training.batch_size", complete)
        if section == "objective" and "loss_weights" in values:
            weights = _mapping(values["loss_weights"], "objective.loss_weights")
            _keys(weights, LOSS_FIELDS, "objective.loss_weights", complete)
        if section == "noise" and "hardware" in values:
            hardware = _mapping(values["hardware"], "noise.hardware")
            _keys(hardware, HARDWARE_FIELDS, "noise.hardware", complete)
    return document


def _integer(value: Any, name: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be an integer{suffix}")
    return value


def _finite(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _number(value: Any, name: str, nonnegative: bool = False) -> float:
    value = _finite(value, name)
    if value < 0 if nonnegative else value <= 0:
        requirement = "nonnegative" if nonnegative else "positive"
        raise ValueError(f"{name} must be finite and {requirement}")
    return value


def _validate(document: dict) -> tuple[dict, dict]:
    raw = document["training"]
    training = {
        "iterations": _integer(raw["iterations"], "training.iterations", 1),
        "batch_size": {
            device: _integer(raw["batch_size"][device], f"training.batch_size.{device}", 1)
            for device in ("cpu", "gpu")
        },
        "seed": _integer(raw["seed"], "training.seed"),
        "lr": _number(raw["lr"], "training.lr"),
    }
    overrides = {}
    for section in ("task", "noise", "objective"):
        for key, value in document[section].items():
            name = f"{section}.{key}"
            if key == "loss_weights":
                overrides[key] = {
                    weight: _number(amount, f"{name}.{weight}", nonnegative=True)
                    for weight, amount in value.items()
                }
            elif key == "hardware":
                overrides[key] = HardwareNoise(**value)
            elif key == "target_diameter_m":
                overrides[key] = None if value is None else _number(value, name)
            elif key == "target_scale_range":
                if value is None:
                    overrides[key] = None
                elif not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError(f"{name} must contain two positive scale factors")
                else:
                    overrides[key] = tuple(_number(amount, name) for amount in value)
            elif key == "collision_offset":
                if not isinstance(value, (list, tuple)) or len(value) != 3:
                    raise ValueError(f"{name} must contain three finite metre offsets")
                overrides[key] = tuple(_finite(x, name) for x in value)
            elif key in INTEGER_FIELDS:
                overrides[key] = _integer(value, name, 0 if key in NOISE_FIELDS else 1)
            else:
                overrides[key] = _number(value, name, nonnegative=key in NONNEGATIVE_FIELDS)
    validated = Config(**overrides)  # Includes derived target radius and official 500 Hz rate.
    overrides["radius_min"], overrides["radius_max"] = validated.radius_min, validated.radius_max
    return training, overrides


def load_settings(path: Path | None = None) -> tuple[dict, dict]:
    """Read defaults plus optional partial YAML overrides.

    Returns ``(training, Config keyword arguments)``. Missing custom fields
    inherit ``config/training.yaml``. Device batch sizes and individual loss
    weights and hardware settings merge by key; camera, physical and controller
    settings are rejected.
    """
    defaults = _read(DEFAULT_SETTINGS_PATH, complete=True)
    validated_defaults = _validate(defaults)
    if path is None or path.resolve() == DEFAULT_SETTINGS_PATH.resolve():
        return validated_defaults
    custom = _read(path)
    merged = {section: dict(values) for section, values in defaults.items()}
    for section, values in custom.items():
        for key, value in values.items():
            if key in {"batch_size", "loss_weights", "hardware"}:
                merged[section][key] = defaults[section][key] | value
            else:
                merged[section][key] = value
    custom_task = custom.get("task", {})
    if {"radius_min", "radius_max"} & custom_task.keys():
        if {"target_diameter_m", "target_scale_range"} & custom_task.keys():
            raise ValueError("Choose canonical target diameter/scale or legacy radius bounds")
        # A legacy custom YAML remains usable without silently applying the new scale range.
        merged["task"].update(target_diameter_m=None, target_scale_range=None)
        for key in ("radius_min", "radius_max"):
            merged["task"].setdefault(key, validated_defaults[1][key])
    return _validate(merged)


def resolve_batch_size(training: dict, device: str) -> int:
    """Choose a batch after resolving the actual CPU/GPU device."""
    if device not in {"cpu", "gpu"}:
        raise ValueError("Resolve device to cpu or gpu before choosing the batch size")
    return _integer(training["batch_size"][device], f"training.batch_size.{device}", 1)
