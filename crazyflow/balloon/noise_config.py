"""Frozen hardware timing, sensor and physical uncertainty settings in SI units."""

from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real


def _number(value: Real, name: str) -> float:
    """Reject booleans, nonnumeric values and nonfinite hardware settings."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"hardware.{name} must be a finite number")
    try:
        amount = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"hardware.{name} must be a finite number") from error
    if not isfinite(amount):
        raise ValueError(f"hardware.{name} must be a finite number")
    return amount


@dataclass(frozen=True)
class HardwareNoise:
    """Opt-in hardware pipeline; zero amplitudes do not assert measured accuracy.

    Delay ranges use milliseconds; bias/drift amplitudes use normalized bbox,
    rad/s gyro and rad attitude units. Physical fractions are uniform half-widths
    with a 1.5 multiplier in the perturbed suite. Time constants use seconds.
    """

    enabled: bool = False
    camera_delay_ms: tuple[float, float] = (0.0, 0.0)
    camera_packet_loss_prob: float = 0.0
    camera_stale_ms: float = 100.0
    bbox_bias_std: float = 0.0
    gyro_delay_ms: tuple[float, float] = (0.0, 0.0)
    attitude_delay_ms: tuple[float, float] = (0.0, 0.0)
    gyro_packet_loss_prob: float = 0.0
    attitude_packet_loss_prob: float = 0.0
    gyro_drift_std: float = 0.0
    gyro_scale_std: float = 0.0
    attitude_bias_std: float = 0.0
    bias_tau_s: float = 30.0
    attitude_command_delay_ms: tuple[float, float] = (0.0, 0.0)
    thrust_command_delay_ms: tuple[float, float] = (0.0, 0.0)
    command_jitter_std_ms: float = 0.0
    command_jitter_tau_s: float = 0.1
    inertia_fraction: float = 0.0
    drag_fraction: float = 0.0
    motor_thrust_fraction: float = 0.0
    motor_response_fraction: float = 0.0
    gust_std: float = 0.0
    gust_tau_s: float = 0.5

    def __post_init__(self) -> None:
        """Validate serialized/YAML inputs before entering a JAX rollout."""
        if type(self.enabled) is not bool:
            raise ValueError("hardware.enabled must be a boolean")
        for definition in fields(self):
            name = definition.name
            if name == "enabled":
                continue
            value = getattr(self, name)
            if name.endswith("_delay_ms"):
                if not isinstance(value, (tuple, list)) or len(value) != 2:
                    raise ValueError(f"hardware.{name} must contain a minimum and maximum delay")
                low, high = (_number(amount, name) for amount in value)
                if not 0 <= low <= high:
                    raise ValueError(f"hardware.{name} must satisfy 0 <= low <= high")
                object.__setattr__(self, name, (low, high))
                continue
            amount = _number(value, name)
            if name.endswith("_tau_s"):
                if amount <= 0:
                    raise ValueError(f"hardware.{name} must be positive")
            elif name.endswith("_prob"):
                if not 0 <= amount < 1:
                    raise ValueError(f"hardware.{name} must be in [0,1)")
            elif name.endswith("_fraction"):
                if not 0 <= amount < 2 / 3:
                    raise ValueError(f"hardware.{name} must be in [0,2/3)")
            elif amount < 0:
                raise ValueError(f"hardware.{name} must be nonnegative")
            object.__setattr__(self, name, amount)
        if self.camera_stale_ms < self.camera_delay_ms[1]:
            raise ValueError("hardware.camera_stale_ms must be >= camera_delay_ms maximum")


HARDWARE_FIELDS = frozenset(definition.name for definition in fields(HardwareNoise))
