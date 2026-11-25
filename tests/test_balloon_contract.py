"""Reject obsolete plants and keep calibrated cameras and native controls portable."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from crazyflow.balloon.camera import intrinsics
from crazyflow.balloon.config import Config
from crazyflow.balloon.model import control_params, nominal_params
from crazyflow.balloon.policy import init_policy, load_checkpoint, save_checkpoint


def test_old_surrogate_schema_cannot_be_used_at_matching_observation_size(tmp_path: Path):
    path = tmp_path / "old.npz"
    metadata = {"schema": "cf21b_bbox_bptt_v1", "algorithm": "BPTT", "feature_dim": 65}
    np.savez(path, metadata=np.array(json.dumps(metadata)))
    with pytest.raises(ValueError, match="compatible"):
        load_checkpoint(path)


def test_calibrated_camera_and_partial_controller_override_are_frozen(tmp_path: Path):
    defaults = control_params()
    kw = (np.asarray(defaults["attitude"]["kw"]) * 1.01).tolist()
    cfg = Config(
        hidden=8,
        camera_intrinsics=(430.0, 450.0, 310.0, 247.0),
        camera_rpy=(0.01, -0.03, 0.02),
        controller_params={"attitude": {"kw": kw}},
    )
    path = tmp_path / "current.npz"
    save_checkpoint(path, init_policy(jax.random.key(0), cfg), cfg)
    _, loaded, metadata = load_checkpoint(path)
    assert metadata["plant"] == "crazyflow.first_principles.cf21B_500"
    np.testing.assert_allclose(intrinsics(loaded), intrinsics(cfg))
    assert loaded.camera_rpy == cfg.camera_rpy
    np.testing.assert_allclose(control_params(loaded)["attitude"]["kw"], kw)
    for key in defaults["force_torque"]:
        np.testing.assert_allclose(
            control_params(loaded)["force_torque"][key], defaults["force_torque"][key]
        )
    assert "rotor_dyn_coef" in loaded.model_params
    for key, value in nominal_params().items():
        np.testing.assert_allclose(nominal_params(loaded)[key], value)


def test_closed_loop_fit_is_not_a_first_principles_parameter_file():
    with pytest.raises(ValueError, match="first_principles"):
        Config(model_params={"rpy_coef": [-100.0, -100.0, -100.0]})


def test_measured_curve_must_work_with_native_zero_thrust_inverse():
    # A positive slope in a bench RPM interval does not guarantee that the
    # official allocator's inverse exists when it turns a motor off.
    with pytest.raises(ValueError, match="quadratic inverse"):
        Config(model_params={"rpm2thrust": [0.02, 0.0, 1e-10]})
