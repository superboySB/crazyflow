"""File-adapter checks for the retained official identification pipeline."""

import json
from argparse import Namespace
from pathlib import Path

import jax
import numpy as np
import pytest

from crazyflow.identify import CSV_COLUMNS, fit, load_log, synthetic_log


def _log() -> dict[str, np.ndarray]:
    time = np.arange(301, dtype=float) / 100
    return {
        "time": time,
        "pos": np.column_stack((time, time * 0.5, np.ones_like(time))),
        "quat": np.tile([0.0, 0.0, 0.0, 1.0], (len(time), 1)),
        "cmd_rpy": np.zeros((len(time), 3)),
        "cmd_f": np.full(len(time), 0.45),
    }


def test_csv_matches_official_npz_schema(tmp_path: Path) -> None:
    data = _log()
    source = tmp_path / "flight.csv"
    table = np.column_stack([data[name] for name in ("time", "pos", "quat", "cmd_rpy", "cmd_f")])
    np.savetxt(source, table, delimiter=",", header=",".join(CSV_COLUMNS), comments="")
    loaded = load_log(source)
    for name, expected in data.items():
        np.testing.assert_allclose(loaded[name], expected)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("cmd_f", 32000.0, "collective thrust"),
        ("cmd_rpy", 30.0, "radians"),
        ("time", 0.0, "strictly increasing"),
        ("quat", 0.0, "zero quaternion"),
    ],
)
def test_rejects_common_log_unit_and_time_errors(
    tmp_path: Path, field: str, value: float, message: str
) -> None:
    data = _log()
    data[field][:] = value
    source = tmp_path / "flight.npz"
    np.savez(source, **data)
    with pytest.raises(ValueError, match=message):
        load_log(source)


def test_drag_synthetic_csv_matches_npz_and_known_official_parameters(tmp_path: Path) -> None:
    """Both log formats preserve identical physical observations and provenance."""
    csv = tmp_path / "drag.csv"
    npz = tmp_path / "drag.npz"
    synthetic_log(csv, 6.0, 7, "so_rpy_rotor_drag")
    synthetic_log(npz, 6.0, 7, "so_rpy_rotor_drag")
    for key, value in load_log(csv).items():
        np.testing.assert_allclose(value, load_log(npz)[key], rtol=1e-10, atol=1e-10)
    metadata = json.loads(csv.with_suffix(".json").read_text())
    assert metadata["synthetic"] is True
    assert metadata["model"] == "so_rpy_rotor_drag"
    assert metadata["seed"] == 7
    known = metadata["known_parameters"]
    assert known["thrust_dyn_coef"] > 0
    assert np.all(np.diag(known["drag_matrix"]) < 0)
    with np.load(npz, allow_pickle=False) as data:
        assert bool(data["synthetic"])
        assert json.loads(str(data["known_parameters_json"])) == known


def test_known_nonzero_drag_and_lag_fit_converges_on_independent_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The end-to-end example must converge and recover an independently known model."""
    train, valid = tmp_path / "train.csv", tmp_path / "valid.csv"
    caplog.set_level("INFO", logger="crazyflow.dynamics.utils.identification")
    with jax.enable_x64():
        synthetic_log(train, 15.0, 0, "so_rpy_rotor_drag")
        synthetic_log(valid, 15.0, 1, "so_rpy_rotor_drag")
        fit(
            Namespace(
                mass=0.0434,
                data=train,
                validation=valid,
                dynamics="so_rpy_rotor_drag",
                verbose=0,
                plot=False,
                output=tmp_path / "fitted.toml",
            )
        )
    known = json.loads(train.with_suffix(".json").read_text())["known_parameters"]
    report = json.loads((tmp_path / "fitted.json").read_text())
    estimated = report["identified"]
    assert report["marked_synthetic"] is True
    assert "SYNTHETIC" in (tmp_path / "fitted.toml").read_text()
    assert report["training_log"] != report["validation_log"]
    np.testing.assert_allclose(estimated["cmd_f_coef"], known["cmd_f_coef"], rtol=0.01)
    np.testing.assert_allclose(estimated["thrust_dyn_coef"], known["thrust_dyn_coef"], rtol=0.15)
    np.testing.assert_allclose(estimated["drag_xy_coef"], known["drag_matrix"][0][0], rtol=0.1)
    np.testing.assert_allclose(estimated["drag_z_coef"], known["drag_matrix"][2][2], rtol=0.1)
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert messages.count("Training success=True") == 2
    assert "Training success=False" not in messages
