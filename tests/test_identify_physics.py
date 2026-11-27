"""Independent measurement-equation checks for physical RPM-domain calibration."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from crazyflow.drones import Drone
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.identify_physics import (
    fit_flight,
    fit_rotor,
    fit_static,
    measured_values,
    templates,
    write_bundle,
)


def _csv(path: Path, columns: list[str], values: np.ndarray) -> None:
    np.savetxt(path, values, delimiter=",", header=",".join(columns), comments="")


def test_static_fit_recovers_mechanical_rpm_force_and_torque(tmp_path: Path) -> None:
    params = load_params(Dynamics.first_principles, Drone.cf21B_500)
    rpm = np.linspace(7000.0, 23000.0, 40)
    thrust = sum(params["rpm2thrust"][i] * rpm**i for i in range(3))
    torque = sum(params["rpm2torque"][i] * rpm**i for i in range(3))
    path = tmp_path / "static.csv"
    _csv(
        path,
        ["motor", "rpm", "thrust_n", "torque_nm"],
        np.column_stack((np.ones(len(rpm)), rpm, thrust, torque)),
    )
    validation = tmp_path / "static_valid.csv"
    validation.write_text(path.read_text())
    fitted, report = fit_static(path, validation)
    assert report["validation_thrust_rmse_n"] < 1e-12
    assert report["validation_torque_rmse_nm"] < 1e-12
    np.testing.assert_allclose(fitted["rpm2thrust"], params["rpm2thrust"], atol=1e-14)
    np.testing.assert_allclose(fitted["rpm2torque"], params["rpm2torque"], atol=1e-14)
    assert report["thrust"]["rmse"] < 1e-12


def test_rotor_fit_recovers_distinct_up_and_down_laws(tmp_path: Path) -> None:
    expected = np.array([14.0, 0.00011, 6.0, 0.00032])
    dt = 0.002
    count = 3000
    actual = np.empty(count)
    actual[0] = 10000.0
    target = np.array([18000.0, 10000.0, 20000.0, 14000.0, 21000.0, 9000.0])[
        (np.arange(count) // 100) % 6
    ]
    for i in range(count - 1):
        c1, c2 = expected[:2] if target[i] > actual[i] else expected[2:]
        actual[i + 1] = actual[i] + dt * (
            c1 * (target[i] - actual[i]) + c2 * (target[i] ** 2 - actual[i] ** 2)
        )
    path = tmp_path / "rotor.csv"
    _csv(
        path,
        ["time", "motor", "target_rpm", "actual_rpm"],
        np.column_stack((np.arange(count) * dt, np.ones(count), target, actual)),
    )
    fitted, _ = fit_rotor(path)
    np.testing.assert_allclose(fitted["rotor_dyn_coef"], expected, rtol=1e-9, atol=1e-10)


def test_no_bench_imu_fit_recovers_curves_with_measured_inertia(tmp_path: Path) -> None:
    params = load_params(Dynamics.first_principles, Drone.cf21B_500)
    time = np.arange(1500) * 0.002
    collective = 15000.0 + 2500.0 * np.sin(time * 2.3)
    differential = 600.0 * np.sin(time * 4.1 + 0.3)
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
    columns = [
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
    ]
    path = tmp_path / "flight.csv"
    _csv(path, columns, np.column_stack((time, rpm, force.sum(-1) / params["mass"], omega, alpha)))
    measured = {key: params[key] for key in ("mass", "J", "L", "prop_inertia")}
    validation = tmp_path / "flight_valid.csv"
    validation.write_text(path.read_text())
    fitted, report = fit_flight(path, measured, validation)
    assert report["validation_collective_thrust_rmse_n"] < 1e-12
    assert report["validation_yaw_reaction_torque_rmse_nm"] < 1e-12
    np.testing.assert_allclose(fitted["rpm2thrust"], params["rpm2thrust"], atol=1e-14)
    np.testing.assert_allclose(fitted["rpm2torque"], params["rpm2torque"], atol=1e-14)
    assert "drag_matrix" in report["not_fitted"]


def test_unconfirmed_templates_cannot_be_exported_as_measurements(tmp_path: Path) -> None:
    templates(tmp_path)
    with pytest.raises(ValueError, match="confirmed_measurements"):
        measured_values(tmp_path / "measurements.toml")


def test_export_keeps_unmeasured_parameters_and_controller_provenance(tmp_path: Path) -> None:
    import json
    import tomllib

    templates(tmp_path)
    measurements = tmp_path / "measurements.toml"
    measurements.write_text(
        measurements.read_text().replace(
            "confirmed_measurements = false", "confirmed_measurements = true"
        )
    )
    output = tmp_path / "physical.toml"
    write_bundle(
        Namespace(
            measurements=measurements,
            static=None,
            static_validation=None,
            flight=None,
            flight_validation=None,
            rotor=None,
            rotor_validation=None,
            output=output,
        )
    )
    params = tomllib.loads(output.read_text())["cf21B_500"]
    assert "rpm2thrust" in params and "rpy_coef" not in params and "J_inv" not in params
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["controller_tuned"] is False
    assert "inherited" in report["provenance"]["rpm2thrust"]
    assert "provided measurement" in report["provenance"]["mass"]


def test_rejects_invalid_telemetry_and_unexcited_measurements(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    _csv(path, ["motor", "rpm", "thrust_n"], np.tile([1.0, 65535.0, 0.1], (20, 1)))
    with pytest.raises(ValueError, match="65535"):
        fit_static(path)
    _csv(path, ["motor", "rpm", "thrust_n"], np.tile([1.0, 15000.0, 0.1], (20, 1)))
    with pytest.raises(ValueError, match="rank deficient"):
        fit_static(path)


def test_rejects_thrust_curve_incompatible_with_official_inverse(tmp_path: Path) -> None:
    rpm = np.linspace(7000.0, 23000.0, 40)
    # Positive/monotone measured curve, but inverse at a zero-force command has no real root.
    path = tmp_path / "bad_inverse.csv"
    _csv(
        path,
        ["motor", "rpm", "thrust_n"],
        np.column_stack((np.ones(len(rpm)), rpm, 0.02 + 1e-10 * rpm**2)),
    )
    with pytest.raises(ValueError, match="official RPM inverse"):
        fit_static(path)


def test_synthetic_full_export_is_explicitly_marked_and_not_auto_confirmed(tmp_path: Path) -> None:
    import json
    import tomllib

    from crazyflow.identify_physics import synthetic_fixtures

    synthetic_fixtures(tmp_path)
    source = tmp_path / "measurements.toml"
    with pytest.raises(ValueError, match="Synthetic fixtures require"):
        measured_values(source)
    output = tmp_path / "fit.toml"
    write_bundle(
        Namespace(
            measurements=source,
            allow_synthetic=True,
            static=None,
            static_validation=None,
            flight=tmp_path / "flight_train.csv",
            flight_validation=tmp_path / "flight_valid.csv",
            rotor=tmp_path / "rotor_train.csv",
            rotor_validation=tmp_path / "rotor_valid.csv",
            output=output,
        )
    )
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["synthetic"] is True and "SYNTHETIC" in output.read_text()
    assert all("SYNTHETIC" in value for value in report["provenance"].values())
    physical = tomllib.loads(output.read_text())["cf21B_500"]
    assert physical["mass"] > 0
    assert np.all(np.linalg.eigvalsh(physical["J"]) > 0)
