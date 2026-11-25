"""One serialized contract for training, simulation and sensor-only inference."""

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class Config:
    """SI units; domain/noise magnitudes are assumptions until measured on hardware."""

    dt: float = 0.02
    substeps: int = 10
    steps: int = 150
    history: int = 5
    hidden: int = 64
    hfov_deg: float = 120.0
    image_width: int = 640
    image_height: int = 480
    camera_offset: tuple = (0.03, 0.0, 0.0)
    camera_rpy: tuple = (0.0, 0.0, 0.0)
    camera_intrinsics: tuple | None = None
    collision_offset: tuple = (0.0, 0.0, 0.02)
    drone_radius: float = 0.086
    radius_min: float = 0.13
    radius_max: float = 0.18
    distance_min: float = 1.5
    distance_max: float = 3.0
    max_impact_speed: float = 2.5
    min_altitude: float = 0.12
    max_range: float = 12.0
    max_unseen_seconds: float = 0.5
    max_tilt: float = 0.5
    max_yaw: float = 0.8
    bbox_noise_std: float = 0.006
    gyro_noise_std: float = 0.015
    gyro_bias_std: float = 0.005
    attitude_noise_std: float = 0.005
    dropout_prob: float = 0.015
    max_delay_steps: int = 2
    mass_fraction: float = 0.05
    thrust_fraction: float = 0.08
    response_fraction: float = 0.1
    wind_std: float = 0.12
    action_delay_steps: int = 1
    loss_weights: dict = field(
        default_factory=lambda: {
            "distance": 1.0,
            "contact": 5.0,
            "speed": 0.04,
            "impact": 0.15,
            "fov": 0.8,
            "ground": 12.0,
            "tilt": 3.0,
            "angular_rate": 0.015,
            "action": 0.025,
            "smoothness": 0.08,
        }
    )
    softmin_temperature: float = 0.08
    impact_temperature: float = 0.15
    contact_temperature: float = 0.12
    ground_margin: float = 0.15
    tilt_soft_limit: float = 0.85
    model_params: dict | None = None
    controller_params: dict | None = None

    def __post_init__(self):
        """Reject contracts that cannot represent the documented task."""
        expected_weights = {
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
        if set(self.loss_weights) != expected_weights or any(
            not np.isfinite(value) or value < 0 for value in self.loss_weights.values()
        ):
            raise ValueError("loss_weights must contain all documented finite nonnegative weights")
        for key in (
            "softmin_temperature",
            "impact_temperature",
            "contact_temperature",
            "tilt_soft_limit",
        ):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not np.isfinite(self.ground_margin) or self.ground_margin < 0:
            raise ValueError("ground_margin must be finite and nonnegative")
        for key in ("camera_offset", "collision_offset", "camera_rpy"):
            value = tuple(getattr(self, key))
            if len(value) != 3 or not np.isfinite(value).all():
                raise ValueError(f"{key} must contain three finite values")
            object.__setattr__(self, key, value)
        if self.steps < 2 or self.substeps < 1 or self.dt <= 0 or self.history < 3:
            raise ValueError("Need positive dt/substeps, steps >= 2 and history >= 3")
        if not 0 < self.hfov_deg < 170 or min(self.image_width, self.image_height) < 1:
            raise ValueError("Invalid pinhole camera dimensions/FOV")
        if not 0 < self.radius_min <= self.radius_max < self.distance_min <= self.distance_max:
            raise ValueError("Need 0 < radius_min <= radius_max < distance_min <= distance_max")
        if not 0 <= self.dropout_prob < 1 or min(self.max_delay_steps, self.action_delay_steps) < 0:
            raise ValueError("Invalid sensor/control delay or dropout probability")
        for key in ("mass_fraction", "thrust_fraction", "response_fraction"):
            if not 0 <= getattr(self, key) < 2 / 3:
                raise ValueError(f"{key} must be in [0,2/3) for positive perturbed-suite samples")
        for key in (
            "bbox_noise_std",
            "gyro_noise_std",
            "gyro_bias_std",
            "attitude_noise_std",
            "wind_std",
        ):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if abs(self.substeps / self.dt - 500) > 1e-6:
            raise ValueError("The official plant and onboard controller run at 500 Hz")
        if self.camera_intrinsics is not None:
            k = tuple(self.camera_intrinsics)
            if len(k) != 4 or not np.isfinite(k).all() or min(k[:2]) <= 0:
                raise ValueError("camera_intrinsics must be finite (fx,fy,cx,cy), fx/fy > 0")
            if not (0 < k[2] < self.image_width and 0 < k[3] < self.image_height):
                raise ValueError("Principal point must lie inside the image")
            object.__setattr__(self, "camera_intrinsics", k)
        if self.model_params is not None:
            from crazyflow.drones import Drone
            from crazyflow.dynamics import Dynamics, load_params

            reference = load_params(Dynamics.first_principles, Drone.cf21B_500)
            for key, value in self.model_params.items():
                if key not in reference or key == "J_inv" or not np.isfinite(value).all():
                    raise ValueError(f"Invalid first_principles parameter {key}")
                shapes = {np.shape(reference[key])}
                if key in ("rpm2thrust", "rpm2torque"):
                    shapes |= {(1, 3), (4, 3)}
                if key == "rotor_dyn_coef":
                    shapes |= {(1, 4), (4, 4)}
                if key in ("L", "prop_inertia"):
                    shapes |= {(1,), (4,)}
                if np.shape(value) not in shapes:
                    raise ValueError(f"Wrong shape for {key}: {np.shape(value)}")
                array = np.asarray(value)
                if key in ("mass", "L", "prop_inertia", "thrust_max") and np.any(array <= 0):
                    raise ValueError(f"{key} must be positive")
                if key in ("thrust_min", "rotor_dyn_coef") and np.any(array < 0):
                    raise ValueError(f"{key} must be nonnegative")
                if key == "J" and (
                    not np.allclose(array, array.T) or np.linalg.eigvalsh(array).min() <= 0
                ):
                    raise ValueError("J must be symmetric positive definite")
                if key == "drag_matrix" and (
                    not np.allclose(array, array.T) or np.linalg.eigvalsh(array).max() > 1e-8
                ):
                    raise ValueError("drag_matrix must be symmetric negative semidefinite")
            effective = reference | self.model_params
            if effective["thrust_min"] >= effective["thrust_max"]:
                raise ValueError("thrust_min must be below thrust_max")
            curve = np.atleast_2d(effective["rpm2thrust"])
            c, b, a = curve[:, 0], curve[:, 1], curve[:, 2]
            forces = np.array([0.0, effective["thrust_max"]])
            discriminant = b[:, None] ** 2 - 4 * a[:, None] * (c[:, None] - forces)
            if np.any(a <= 0) or np.any(discriminant < 0):
                raise ValueError(
                    "rpm2thrust must have a real quadratic inverse from zero to max thrust"
                )
            rpm = (-b[:, None] + np.sqrt(discriminant)) / (2 * a[:, None])
            if not np.isfinite(rpm).all() or np.any(rpm < 0):
                raise ValueError("rpm2thrust inverse must produce finite nonnegative RPM")
        if self.controller_params is not None:
            from crazyflow.control import load_fn_params
            from crazyflow.control.mellinger import attitude2force_torque, force_torque2rotor_vel
            from crazyflow.drones import Drone

            functions = {"attitude": attitude2force_torque, "force_torque": force_torque2rotor_vel}
            for section, values in self.controller_params.items():
                if section not in functions or not isinstance(values, dict):
                    raise ValueError(f"Invalid controller section {section}")
                reference = load_fn_params(functions[section], Drone.cf21B_500)
                for key, value in values.items():
                    if (
                        key not in reference
                        or np.shape(value) != np.shape(reference[key])
                        or not np.isfinite(value).all()
                    ):
                        raise ValueError(f"Invalid controller parameter {section}.{key}")

    def to_dict(self) -> dict:
        """Return the complete checkpoint contract."""
        return asdict(self)
