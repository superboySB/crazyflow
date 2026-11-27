"""Configuration of the CrazyRL task and the native Crazyflow simulator.

Task constants come from CrazyRL's ``quad_baseline_env_cfg.py``.  The drone
parameters remain Crazyflow's distributed cf21B_500 defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EnvConfig:
    drone: str = "cf21B_500"
    integrator: str = "rk4"
    model_params_path: str | None = None
    model_params: dict | None = None
    policy_hz: int = 15
    physics_hz: int = 150
    action_delay_steps: int = 4
    history_length: int = 10
    episode_seconds: float = 80.0
    camera_max_distance: float = 4.0
    camera_focal_length: float = 20.0
    camera_aperture: float = 20.955
    depth_invalid_rate_max: float = 0.15
    observation_noise: bool = True
    domain_randomization: bool = True
    mass_randomization_percent: float = 0.2
    wind_tau: float = 1.0
    wind_sigma: float = 0.5
    thrust_initial_min: float = 0.85
    thrust_initial_max: float = 1.0
    thrust_degradation_min: float = 0.10
    thrust_degradation_max: float = 0.15
    thrust_min_effectiveness: float = 0.7
    desired_speed_min: float = 0.5
    desired_speed_max: float = 1.5
    goal_noise: float = 0.15
    map_seed: int = 42
    num_maps: int = 2
    enable_difficulty_curriculum: bool = True
    map_generation_reset_batches: int = 3000
    obstacle_min_distance_min: float = 0.4
    hover_hold_max_seconds: float = 2.0
    hover_hold_increment_seconds: float = 0.1
    map_size: float = 16.0
    obstacle_min_distance: float = 1.5
    obstacle_size_min: float = 0.4
    obstacle_size_max: float = 0.8
    floater_size_min: float = 0.4
    floater_size_max: float = 0.8
    spawn_clearance: float = 0.20
    drone_radius: float = 0.055
    hover_hold_seconds: float = 0.0
    hover_distance: float = 0.5
    hover_speed: float = 0.5
    reward_distance: float = 8.0
    reward_yaw: float = 0.3
    reward_action: float = 0.3
    reward_action_change: float = 0.5
    reward_velocity_direction: float = 0.15
    reward_speed_excess: float = 0.5
    reward_speed_match: float = 0.0
    reward_height: float = 0.25
    reward_collision: float = 80.0
    reward_esdf: float = 0.1
    reward_success: float = 80.0

    def __post_init__(self) -> None:
        """Reject task settings that would change timing or invalid distributions."""
        if self.integrator not in {"euler", "rk4"}:
            raise ValueError("integrator must be euler or rk4")
        if self.physics_hz % self.policy_hz:
            raise ValueError("physics_hz must be divisible by policy_hz")
        if self.physics_hz != 150 or self.policy_hz != 15:
            raise ValueError("The CrazyRL parity task requires 150 Hz physics and 15 Hz policy")
        if self.history_length < 1 or self.num_maps < 1:
            raise ValueError("history_length and num_maps must be positive")
        if self.obstacle_min_distance <= 0 or self.map_generation_reset_batches < 1:
            raise ValueError("Map separation and regeneration interval must be positive")
        if not 0 <= self.mass_randomization_percent < 1:
            raise ValueError("mass_randomization_percent must be in [0, 1)")
        if not 0 <= self.depth_invalid_rate_max <= 1:
            raise ValueError("depth_invalid_rate_max must be in [0, 1]")

    @property
    def observation_size(self) -> int:
        return self.history_length * 28 + 64

    @property
    def critic_observation_size(self) -> int:
        return self.history_length * 37 + 64

    @property
    def max_episode_steps(self) -> int:
        return round(self.episode_seconds * self.policy_hz)

    @property
    def dt(self) -> float:
        return 1.0 / self.policy_hz

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def env_config(values: dict[str, Any] | None = None) -> EnvConfig:
    """Validate a plain environment mapping, rejecting unknown settings."""
    values = values or {}
    valid = {field.name for field in fields(EnvConfig)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown environment settings: {sorted(unknown)}")
    return EnvConfig(**values)


def load_config(path: str | Path = "config/training.yaml") -> dict[str, Any]:
    """Load plain YAML sections; ``env_config(config['environment'])`` validates the task."""
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a mapping")
    env_config(config.get("environment", {}))
    return config
