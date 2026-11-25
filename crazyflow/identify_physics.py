"""Measured RPM-domain calibration for Crazyflow's first-principles plant.

These measurement adapters are separate from the official ``so_rpy`` closed-loop fit in
``crazyflow.identify``. They never interpret PWM, DShot values or electrical RPM as mechanical RPM,
and never overwrite official defaults or tune the flight controller.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from crazyflow.drones import Drone
from crazyflow.dynamics import Dynamics, load_params

RPM_SCALE = 10000.0
STATIC_COLUMNS = ("motor", "rpm", "thrust_n", "torque_nm", "voltage_v")
ROTOR_COLUMNS = ("time", "motor", "target_rpm", "actual_rpm")
MEASURED_KEYS = {
    "mass",
    "J",
    "L",
    "prop_inertia",
    "mixing_matrix",
    "drag_matrix",
    "thrust_min",
    "thrust_max",
}


def _table(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read finite numeric CSV fields without guessing field names or units."""
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float, ndmin=1)
    missing = set(required) - set(table.dtype.names or ())
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    result = {name: np.asarray(table[name], dtype=float) for name in required}
    if len(table) < 10 or not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"{path}: require at least 10 rows of finite measurements")
    return result


def _rpm(values: np.ndarray) -> None:
    """Reject invalid telemetry; the caller must first convert eRPM using actual pole pairs."""
    if np.any(values < 0) or np.any(values == 65535) or np.any(values > 200000):
        raise ValueError("Expected valid mechanical RPM; reject telemetry sentinel 65535/eRPM/PWM")


def _least_squares(design: np.ndarray, measured: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Require identifiable, reasonably conditioned measurement excitation."""
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Measurements do not excite all fitted coefficients (rank deficient)")
    condition = float(np.linalg.cond(design))
    if condition > 1e6:
        raise ValueError("Poorly conditioned excitation; collect a wider measurement range")
    params = np.linalg.lstsq(design, measured, rcond=None)[0]
    residual = design @ params - measured
    return params, {
        "samples": len(measured),
        "condition": condition,
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def _check_thrust_inverse(curve: list | np.ndarray, thrust_max: float) -> None:
    """Reject curves the unchanged official force-to-RPM quadratic inverse cannot use."""
    from crazyflow.control.transform import motor_force2rotor_vel

    curve = np.asarray(curve)
    if curve.shape != (3,) or curve[2] <= 0:
        raise ValueError("Official RPM inverse requires a positive quadratic thrust coefficient")
    with np.errstate(invalid="ignore", divide="ignore"):
        rpm = motor_force2rotor_vel(np.array([0.0, thrust_max]), curve)
    if not np.isfinite(rpm).all() or np.any(rpm < 0):
        raise ValueError(
            "Thrust curve is incompatible with official RPM inverse over [0,thrust_max]"
        )


def fit_static(path: Path, validation: Path | None = None) -> tuple[dict, dict]:
    """Fit shared single-motor thrust and optional reaction-torque curves from actual RPM."""
    data = _table(path, ("motor", "rpm", "thrust_n"))
    _rpm(data["rpm"])
    if np.any(~np.isin(data["motor"], [1, 2, 3, 4])) or np.any(data["thrust_n"] < 0):
        raise ValueError("motor IDs must be 1..4; thrust_n is nonnegative single-motor thrust")
    normalized = data["rpm"] / RPM_SCALE
    design = np.column_stack((np.ones_like(normalized), normalized, normalized**2))
    coefficient, metrics = _least_squares(design, data["thrust_n"])
    curve = coefficient / np.array([1.0, RPM_SCALE, RPM_SCALE**2])
    grid = np.linspace(data["rpm"].min(), data["rpm"].max(), 200)
    if np.any(curve[0] + curve[1] * grid + curve[2] * grid**2 < -1e-5):
        raise ValueError("Fitted thrust is negative within the measured RPM range")
    if np.any(curve[1] + 2 * curve[2] * grid <= 0):
        raise ValueError("Thrust must increase monotonically in the measured operating range")
    _check_thrust_inverse(curve, float(data["thrust_n"].max()))
    params = {"rpm2thrust": curve.tolist()}
    report: dict[str, Any] = {
        "thrust": metrics,
        "rpm_interval": [float(grid[0]), float(grid[-1])],
        "shared_curve": True,
        "motor_ids": sorted(set(data["motor"].astype(int))),
        "note": "Shared fit assumes equivalent motors/props; inspect each motor residual.",
    }
    residual = design @ coefficient - data["thrust_n"]
    report["per_motor_thrust_rmse_n"] = {
        str(int(motor)): float(np.sqrt(np.mean(residual[data["motor"] == motor] ** 2)))
        for motor in np.unique(data["motor"])
    }
    names = np.genfromtxt(path, delimiter=",", names=True, max_rows=1).dtype.names or ()
    if "torque_nm" in names:
        torque = _table(path, ("torque_nm",))["torque_nm"]
        if np.any(torque < 0):
            raise ValueError(
                "torque_nm is reaction-torque magnitude; spin direction lives in mixing_matrix"
            )
        coefficient_torque, report["torque"] = _least_squares(design, torque)
        params["rpm2torque"] = (
            coefficient_torque / np.array([1.0, RPM_SCALE, RPM_SCALE**2])
        ).tolist()
    if validation is not None:
        if validation.resolve() == path.resolve():
            raise ValueError("Validation must be a separate recording")
        heldout = _table(validation, ("rpm", "thrust_n"))
        _rpm(heldout["rpm"])
        prediction = sum(curve[i] * heldout["rpm"] ** i for i in range(3))
        report["validation_thrust_rmse_n"] = float(
            np.sqrt(np.mean((prediction - heldout["thrust_n"]) ** 2))
        )
        validation_names = (
            np.genfromtxt(validation, delimiter=",", names=True, max_rows=1).dtype.names or ()
        )
        if "rpm2torque" in params and "torque_nm" in validation_names:
            measured_torque = _table(validation, ("torque_nm",))["torque_nm"]
            predicted_torque = sum(params["rpm2torque"][i] * heldout["rpm"] ** i for i in range(3))
            report["validation_torque_rmse_nm"] = float(
                np.sqrt(np.mean((predicted_torque - measured_torque) ** 2))
            )
    return params, report


def fit_rotor(path: Path, validation: Path | None = None) -> tuple[dict, dict]:
    """Fit the official separate up/down RPM law using left-aligned target and actual RPM."""
    data = _table(path, ROTOR_COLUMNS)
    _rpm(data["target_rpm"])
    _rpm(data["actual_rpm"])
    features, derivatives, directions, intervals = [], [], [], []
    for motor in np.unique(data["motor"]):
        if motor not in (1, 2, 3, 4):
            raise ValueError("motor IDs must be 1..4")
        mask = data["motor"] == motor
        time, target, actual = (data[name][mask] for name in ("time", "target_rpm", "actual_rpm"))
        dt = np.diff(time)
        if len(dt) < 10 or np.any(dt <= 0) or np.max(dt) > 3 * np.median(dt):
            raise ValueError("Each motor needs increasing timestamps and no >3-period gaps")
        target, current = target[:-1], actual[:-1]
        delta = target - current
        features.append(
            np.column_stack((delta / RPM_SCALE, (target**2 - current**2) / RPM_SCALE**2))
        )
        derivatives.append(np.diff(actual) / dt / RPM_SCALE)
        directions.append(delta)
        intervals.extend(dt.tolist())
    design, measured, delta = map(np.concatenate, (features, derivatives, directions))
    coefficients, diagnostics = [], {}
    for name, mask in (("spin_up", delta > 50), ("spin_down", delta < -50)):
        if np.count_nonzero(mask) < 10:
            raise ValueError(f"Need at least 10 excited {name} intervals with >50 RPM error")
        a, b = design[mask], measured[mask]
        _, diagnostics[name] = _least_squares(a, b)
        fit = lsq_linear(a, b, bounds=(0.0, np.inf), tol=1e-12)
        if not fit.success:
            raise ValueError(f"{name} bounded fit did not converge")
        coefficients.extend([float(fit.x[0]), float(fit.x[1] / RPM_SCALE)])
        diagnostics[name]["rmse_rpm_per_second"] = float(
            np.sqrt(np.mean((a @ fit.x - b) ** 2)) * RPM_SCALE
        )
    diagnostics.update(
        mean_dt_s=float(np.mean(intervals)),
        max_dt_s=float(np.max(intervals)),
        method="Left-aligned finite differences; commanded target RPM held over each interval.",
        note=(
            "Requires independently calibrated target-RPM mapping, "
            "sufficient telemetry rate, and time alignment."
        ),
    )
    params = {"rotor_dyn_coef": coefficients}
    if validation is not None:
        if validation.resolve() == path.resolve():
            raise ValueError("Validation must be a separate recording")
        heldout = _table(validation, ROTOR_COLUMNS)
        _rpm(heldout["target_rpm"])
        _rpm(heldout["actual_rpm"])
        errors = []
        for motor in np.unique(heldout["motor"]):
            mask = heldout["motor"] == motor
            time, target, actual = (
                heldout[name][mask] for name in ("time", "target_rpm", "actual_rpm")
            )
            dt = np.diff(time)
            if np.any(dt <= 0):
                raise ValueError("Validation timestamps must increase per motor")
            current = actual[:-1]
            up = target[:-1] > current
            c1 = np.where(up, coefficients[0], coefficients[2])
            c2 = np.where(up, coefficients[1], coefficients[3])
            predicted = current + dt * (
                c1 * (target[:-1] - current) + c2 * (target[:-1] ** 2 - current**2)
            )
            errors.extend((predicted - actual[1:]).tolist())
        diagnostics["validation_one_step_rmse_rpm"] = float(
            np.sqrt(np.mean(np.asarray(errors) ** 2))
        )
    return params, diagnostics


def fit_flight(
    path: Path, measurements: dict[str, Any], validation: Path | None = None
) -> tuple[dict, dict]:
    """Fit shared RPM curves from CoM IMU signals on explicitly negligible-drag segments.

    This is a local adapter of the retained rigid-body equations, not the upstream PWM-domain
    lol_sysid estimator. Mass/J and IMU-to-CoM correction must already be measured. Drag, inertia,
    controller gains and command-to-target-RPM mapping are not identified by this fit.
    """
    required = (
        "time",
        "rpm1",
        "rpm2",
        "rpm3",
        "rpm4",
        "specific_force_z",
        "omega_x",
        "omega_y",
        "omega_z",
        "alpha_x",
        "alpha_y",
        "alpha_z",
    )
    data = _table(path, required)
    dt = np.diff(data["time"])
    if np.any(dt <= 0) or np.max(dt) > 3 * np.median(dt):
        raise ValueError("Flight timestamps must increase without >3-period gaps")
    rpm = np.column_stack([data[f"rpm{index}"] for index in range(1, 5)])
    _rpm(rpm)
    n = rpm / RPM_SCALE
    thrust_design = np.column_stack((np.full(len(rpm), 4.0), n.sum(-1), (n * n).sum(-1)))
    thrust_coefficient, thrust_diagnostics = _least_squares(
        thrust_design, measurements["mass"] * data["specific_force_z"]
    )
    thrust_curve = thrust_coefficient / np.array([1.0, RPM_SCALE, RPM_SCALE**2])
    interval = np.linspace(rpm.min(), rpm.max(), 200)
    if (
        np.min(sum(thrust_curve[i] * interval**i for i in range(3))) < -1e-5
        or np.min(thrust_curve[1] + 2 * thrust_curve[2] * interval) <= 0
    ):
        raise ValueError(
            "Flight thrust fit is nonphysical; inspect drag, IMU frame/bias, RPM and mass"
        )
    defaults = load_params(Dynamics.first_principles, Drone.cf21B_500)
    spin = np.asarray(measurements.get("mixing_matrix", defaults["mixing_matrix"]))[-1]
    prop_inertia = np.asarray(measurements.get("prop_inertia", defaults["prop_inertia"]))
    omega = np.column_stack([data[f"omega_{axis}"] for axis in "xyz"])
    alpha = np.column_stack([data[f"alpha_{axis}"] for axis in "xyz"])
    inertia = np.asarray(measurements["J"])
    torque = alpha @ inertia.T + np.cross(omega, omega @ inertia.T)
    rotor_momentum_derivative = np.sum(
        np.gradient(rpm, data["time"], axis=0) * spin * prop_inertia * (2 * np.pi / 60), axis=-1
    )
    yaw_torque = torque[:, 2] - rotor_momentum_derivative
    # Four balanced alternating spin signs cancel a shared constant torque term.
    if not np.isclose(spin.sum(), 0):
        raise ValueError("Flight torque fit assumes four balanced alternating spin directions")
    torque_design = np.column_stack(((n * spin).sum(-1), (n * n * spin).sum(-1)))
    torque_coefficient, torque_diagnostics = _least_squares(torque_design[2:-2], yaw_torque[2:-2])
    torque_curve = np.array(
        [0.0, torque_coefficient[0] / RPM_SCALE, torque_coefficient[1] / RPM_SCALE**2]
    )
    if np.min(sum(torque_curve[i] * interval**i for i in range(3))) < 0:
        raise ValueError(
            "Reaction torque fit is negative; check measured J, gyro units, motor order/spins"
        )
    _check_thrust_inverse(
        thrust_curve, float(np.max(measurements["mass"] * data["specific_force_z"]) / 4)
    )
    params = {"rpm2thrust": thrust_curve.tolist(), "rpm2torque": torque_curve.tolist()}
    report: dict[str, Any] = {
        "thrust": thrust_diagnostics,
        "yaw_torque": torque_diagnostics,
        "actual_rpm_interval": [float(rpm.min()), float(rpm.max())],
        "assumptions": [
            "CoM specific force in body FLU, m/s^2, includes +g when level at rest",
            "Measured J, body omega/alpha in rad/s and rad/s^2",
            "Negligible translational drag/external force and aerodynamic yaw torque",
            "Equivalent motor/prop curves; target-RPM mapping fitted separately",
        ],
        "torque_intercept": "Fixed zero: shared constant is unobservable with balanced motor spins",
        "prop_inertia_used": prop_inertia.tolist(),
        "not_fitted": ["J", "drag_matrix", "rotor_dyn_coef", "controller gains", "driver mapping"],
    }
    if validation is not None:
        if validation.resolve() == path.resolve():
            raise ValueError("Validation must be a separate recording")
        heldout = _table(validation, required)
        rpm_valid = np.column_stack([heldout[f"rpm{index}"] for index in range(1, 5)])
        _rpm(rpm_valid)
        predicted = np.sum(sum(thrust_curve[i] * rpm_valid**i for i in range(3)), axis=-1)
        observed = measurements["mass"] * heldout["specific_force_z"]
        report["validation_collective_thrust_rmse_n"] = float(
            np.sqrt(np.mean((predicted - observed) ** 2))
        )
        validation_time = heldout["time"]
        if np.any(np.diff(validation_time) <= 0):
            raise ValueError("Validation flight timestamps must strictly increase")
        omega_valid = np.column_stack([heldout[f"omega_{axis}"] for axis in "xyz"])
        alpha_valid = np.column_stack([heldout[f"alpha_{axis}"] for axis in "xyz"])
        torque_valid = alpha_valid @ inertia.T + np.cross(omega_valid, omega_valid @ inertia.T)
        hdot_valid = np.sum(
            np.gradient(rpm_valid, validation_time, axis=0)
            * spin
            * prop_inertia
            * (2 * np.pi / 60),
            axis=-1,
        )
        observed_torque = torque_valid[:, 2] - hdot_valid
        predicted_torque = np.sum(
            sum(torque_curve[i] * rpm_valid**i for i in range(3)) * spin, axis=-1
        )
        report["validation_yaw_reaction_torque_rmse_nm"] = float(
            np.sqrt(np.mean((predicted_torque[2:-2] - observed_torque[2:-2]) ** 2))
        )
    return params, report


def measured_values(path: Path, allow_synthetic: bool = False) -> dict[str, Any]:
    """Require confirmed measurements rather than treating template values as measured."""
    raw = tomllib.loads(path.read_text())
    synthetic = raw.get("synthetic") is True
    if synthetic and not allow_synthetic:
        raise ValueError(
            "Synthetic fixtures require --allow-synthetic and are not hardware measurements"
        )
    if raw.get("confirmed_measurements") is not True and not (synthetic and allow_synthetic):
        raise ValueError(
            "Set confirmed_measurements=true only after replacing/verifying the measurement fields"
        )
    values = raw.get("measurements", {})
    if {"mass", "J", "L"} - set(values):
        raise ValueError(
            "Measured mass, J, L are required; so_rpy coefficients are not physical parameters"
        )
    if set(values) - MEASURED_KEYS:
        raise ValueError(f"Unknown physical measurements: {sorted(set(values) - MEASURED_KEYS)}")
    if not all(np.isfinite(np.asarray(value)).all() for value in values.values()):
        raise ValueError("Physical measurements must be finite")
    inertia = np.asarray(values["J"])
    if (
        inertia.shape != (3, 3)
        or not np.allclose(inertia, inertia.T)
        or np.linalg.eigvalsh(inertia).min() <= 0
    ):
        raise ValueError("J must be a measured symmetric positive definite 3x3 inertia in kg m^2")
    if values["mass"] <= 0 or np.any(np.asarray(values["L"]) <= 0):
        raise ValueError("mass and motor arm L must be positive SI quantities")
    return values


def write_bundle(args: argparse.Namespace) -> None:
    """Export a physical table with per-key provenance and unchanged controller settings."""
    params = {
        key: np.asarray(value).tolist()
        for key, value in load_params(Dynamics.first_principles, Drone.cf21B_500).items()
        if key != "J_inv"
    }
    provenance = {key: "inherited official cf21B_500 default; not measured" for key in params}
    synthetic = tomllib.loads(args.measurements.read_text()).get("synthetic") is True
    measurements = measured_values(args.measurements, getattr(args, "allow_synthetic", False))
    params.update(measurements)
    provenance.update({key: f"provided measurement: {args.measurements}" for key in measurements})
    diagnostics = {}
    if args.static:
        fitted, diagnostics["static"] = fit_static(args.static, args.static_validation)
        params.update(fitted)
        provenance.update(
            {key: f"actual mechanical RPM/static fit: {args.static}" for key in fitted}
        )
    if args.flight:
        fitted, diagnostics["flight"] = fit_flight(
            args.flight, measurements, args.flight_validation
        )
        params.update(fitted)
        provenance.update({key: f"actual RPM/CoM IMU flight fit: {args.flight}" for key in fitted})
    if args.rotor:
        fitted, diagnostics["rotor"] = fit_rotor(args.rotor, args.rotor_validation)
        params.update(fitted)
        provenance.update(
            {key: f"target/mechanical RPM transient fit: {args.rotor}" for key in fitted}
        )
    if params["thrust_min"] < 0 or params["thrust_min"] >= params["thrust_max"]:
        raise ValueError("Invalid measured thrust limits")
    from crazyflow.balloon.config import Config

    Config(model_params=params)  # Same complete physical contract that training consumes.
    _check_thrust_inverse(params["rpm2thrust"], params["thrust_max"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# First-principles physical plant only; controller parameters are separate.",
        "# See the JSON sidecar for inherited, measured and fitted quantities.",
        "[cf21B_500]",
    ]
    lines.extend(f"{key} = {json.dumps(value)}" for key, value in params.items())
    if synthetic:
        lines.insert(0, "# SYNTHETIC PIPELINE CHECK: these are not hardware-identified parameters.")
        provenance = {
            key: f"SYNTHETIC pipeline fixture: {value}" for key, value in provenance.items()
        }
    args.output.write_text("\n".join(lines) + "\n")
    report = {
        "synthetic": synthetic,
        "model": "first_principles",
        "drone": "cf21B_500",
        "provenance": provenance,
        "diagnostics": diagnostics,
        "controller_tuned": False,
        "not_identified_by_this_tool": [
            "controller gains",
            "N-to-firmware actuator mapping",
            "camera calibration",
            "contact geometry",
        ],
        "note": (
            "Inherited fields remain assumptions; validate independent full flight-command replays."
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Physical plant: {args.output}; provenance: {args.output.with_suffix('.json')}")


def templates(output: Path) -> None:
    """Write empty CSV schemas and an explicitly unconfirmed measurement checklist."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "static.csv").write_text(",".join(STATIC_COLUMNS) + "\n")
    (output / "flight.csv").write_text(
        "time,rpm1,rpm2,rpm3,rpm4,specific_force_z,omega_x,omega_y,omega_z,alpha_x,alpha_y,alpha_z\n"
    )
    (output / "rotor.csv").write_text(",".join(ROTOR_COLUMNS) + "\n")
    (
        output / "measurements.toml"
    ).write_text("""# Replace each listed value with measurements of the final flight configuration.
# Template numbers are official defaults and are NOT measurements of your modified drone.
confirmed_measurements = false
[measurements]
mass = 0.0434
J = [[0.000025, 0.0, 0.0], [0.0, 0.000028, 0.0], [0.0, 0.0, 0.000049]]
L = 0.035355
# Optional measured fields: prop_inertia, thrust_min, thrust_max, mixing_matrix, drag_matrix.
# Remaining fields inherit official defaults and are marked unmeasured in the JSON sidecar.
""")
    print(output)


def synthetic_fixtures(output: Path) -> None:
    """Generate marked synthetic measurements solely for an executable file-to-export check."""
    templates(output)
    params = load_params(Dynamics.first_principles, Drone.cf21B_500)
    measurement_file = output / "measurements.toml"
    text = measurement_file.read_text().replace(
        "confirmed_measurements = false", "synthetic = true\nconfirmed_measurements = false"
    )
    text += f"prop_inertia = {float(params['prop_inertia'])}\n"
    measurement_file.write_text(text)

    def save(name: str, columns: list[str], values: np.ndarray) -> None:
        np.savetxt(output / name, values, delimiter=",", header=",".join(columns), comments="")

    for seed, name in ((0, "train"), (1, "valid")):
        phase = seed * 0.41
        rpm = np.tile(np.linspace(7000.0 + seed * 100, 23000.0 - seed * 100, 40), 4)
        force = sum(params["rpm2thrust"][i] * rpm**i for i in range(3))
        torque = sum(params["rpm2torque"][i] * rpm**i for i in range(3))
        save(
            f"static_{name}.csv",
            ["motor", "rpm", "thrust_n", "torque_nm"],
            np.column_stack((np.repeat(np.arange(1, 5), 40), rpm, force, torque)),
        )
        time = np.arange(2000) * 0.002
        target = np.array([18000.0, 10000.0, 20000.0, 14000.0, 21000.0, 9000.0])[
            ((np.arange(len(time)) // 100) + seed) % 6
        ]
        actual = np.empty(len(time))
        actual[0] = 13000.0
        for i in range(len(time) - 1):
            c1, c2 = (
                params["rotor_dyn_coef"][:2]
                if target[i] > actual[i]
                else params["rotor_dyn_coef"][2:]
            )
            actual[i + 1] = actual[i] + 0.002 * (
                c1 * (target[i] - actual[i]) + c2 * (target[i] ** 2 - actual[i] ** 2)
            )
        save(
            f"rotor_{name}.csv",
            list(ROTOR_COLUMNS),
            np.column_stack((time, np.ones(len(time)), target, actual)),
        )
        collective = 15000.0 + 2500.0 * np.sin(time * 2.3 + phase)
        differential = 200.0 * np.sin(time * 4.1 + 0.3 + phase)
        rpm = np.column_stack(
            (
                collective + differential,
                collective - differential,
                collective + differential,
                collective - differential,
            )
        )
        force = sum(params["rpm2thrust"][i] * rpm**i for i in range(3))
        torque = sum(params["rpm2torque"][i] * rpm**i for i in range(3))
        spin = np.asarray(params["mixing_matrix"])[-1]
        hdot = np.sum(
            np.gradient(rpm, time, axis=0) * spin * params["prop_inertia"] * 2 * np.pi / 60, axis=-1
        )
        alpha = np.zeros((len(time), 3))
        alpha[:, 2] = ((torque * spin).sum(-1) + hdot) / params["J"][2, 2]
        omega = np.zeros_like(alpha)
        omega[1:, 2] = np.cumsum(alpha[:-1, 2]) * 0.002
        save(
            f"flight_{name}.csv",
            [
                "time",
                "rpm1",
                "rpm2",
                "rpm3",
                "rpm4",
                "specific_force_z",
                "omega_x",
                "omega_y",
                "omega_z",
                "alpha_x",
                "alpha_y",
                "alpha_z",
            ],
            np.column_stack((time, rpm, force.sum(-1) / params["mass"], omega, alpha)),
        )
    print(f"SYNTHETIC fixtures only, not hardware measurements: {output}")


def controller_template(path: Path) -> None:
    """Export unchanged active official controller tables for explicit review and later editing."""
    from crazyflow.control import load_fn_params
    from crazyflow.control.mellinger import attitude2force_torque, force_torque2rotor_vel

    lines = [
        "# Official defaults, NOT identified gains or a calibrated firmware mapping.",
        "# Physical plant parameters are supplied independently with --model-params.",
    ]
    for section, function in (
        ("attitude", attitude2force_torque),
        ("force_torque", force_torque2rotor_vel),
    ):
        lines.append(f"[{section}]")
        values = load_fn_params(function, Drone.cf21B_500)
        lines.extend(
            f"{name} = {json.dumps(np.asarray(value).tolist())}" for name, value in values.items()
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(path)


def main() -> None:
    """CLI for measurement templates, bifilar inertia and physical parameter export."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser(
        "template", help="Write unconfirmed measurement and actual-RPM CSV templates"
    )
    template.add_argument("--output", type=Path, default=Path("outputs/physical_measurements"))
    synthetic = commands.add_parser(
        "synthetic", help="Create marked synthetic CSV fixtures for a complete software check"
    )
    synthetic.add_argument("--output", type=Path, default=Path("outputs/physical_synthetic"))
    controller = commands.add_parser(
        "controller-template", help="Export unchanged active official controller defaults"
    )
    controller.add_argument("--output", type=Path, default=Path("outputs/controller.toml"))
    inertia = commands.add_parser("inertia", help="Bifilar suspended-axis inertia, SI units")
    inertia.add_argument(
        "--suspended-mass", type=float, required=True, help="Drone plus fixture, kg"
    )
    inertia.add_argument(
        "--spacing", type=float, required=True, help="Full distance between two strings, m"
    )
    inertia.add_argument("--length", type=float, required=True, help="String length, m")
    inertia.add_argument("--period", type=float, required=True, help="Mean small-angle period, s")
    inertia.add_argument(
        "--fixture-inertia", type=float, default=0.0, help="Same-axis fixture inertia, kg m^2"
    )
    inertia.add_argument(
        "--offset", type=float, default=0.0, help="Drone CoM offset from suspended axis, m"
    )
    inertia.add_argument(
        "--drone-mass", type=float, help="Drone only kg; required for nonzero CoM offset"
    )
    export = commands.add_parser(
        "export", help="Export first-principles parameters with per-key provenance"
    )
    export.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow only explicitly marked synthetic fixtures",
    )
    export.add_argument("--measurements", type=Path, required=True)
    export.add_argument(
        "--static", type=Path, help="Single-motor actual RPM, thrust N, optional torque Nm"
    )
    export.add_argument("--static-validation", type=Path)
    export.add_argument(
        "--flight", type=Path, help="Actual RPM + CoM IMU flight log, negligible-drag segments"
    )
    export.add_argument("--flight-validation", type=Path)
    export.add_argument(
        "--negligible-drag",
        action="store_true",
        help="Explicitly acknowledge low-airspeed/external-force assumptions for --flight",
    )
    export.add_argument("--rotor", type=Path, help="Target/actual mechanical RPM transient log")
    export.add_argument("--rotor-validation", type=Path)
    export.add_argument("--output", type=Path, default=Path("outputs/physical.toml"))
    args = parser.parse_args()
    if args.command == "template":
        templates(args.output)
    elif args.command == "synthetic":
        synthetic_fixtures(args.output)
    elif args.command == "controller-template":
        controller_template(args.output)
    elif args.command == "inertia":
        if min(args.suspended_mass, args.spacing, args.length, args.period) <= 0:
            parser.error("mass, spacing, length and period must be positive SI quantities")
        if args.offset and (args.drone_mass is None or args.drone_mass <= 0):
            parser.error("Nonzero offset requires measured --drone-mass")
        result = (
            args.suspended_mass
            * 9.81
            * args.spacing**2
            * args.period**2
            / (16 * np.pi**2 * args.length)
            - args.fixture_inertia
            - (args.drone_mass or 0) * args.offset**2
        )
        if not np.isfinite(result) or result <= 0:
            parser.error("Corrected inertia is nonpositive; inspect geometry, fixture and units")
        print(json.dumps({"suspended_axis_inertia_kg_m2": result}))
    else:
        if args.static and args.flight:
            parser.error("Choose --static or --flight curve identification, not both")
        if args.flight and not args.negligible_drag:
            parser.error(
                "--flight requires explicit --negligible-drag and correctly selected segments"
            )
        if args.flight_validation and not args.flight:
            parser.error("--flight-validation requires --flight")
        if args.static_validation and not args.static:
            parser.error("--static-validation requires --static")
        if args.rotor_validation and not args.rotor:
            parser.error("--rotor-validation requires --rotor")
        write_bundle(args)


if __name__ == "__main__":
    main()
