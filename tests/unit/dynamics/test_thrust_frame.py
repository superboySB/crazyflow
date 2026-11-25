"""Physical body-to-world thrust direction across fitted and rigid-body models."""

import importlib

import casadi as cs
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from crazyflow.drones import Drone
from crazyflow.dynamics import Dynamics, load_params, parametrize
from crazyflow.dynamics.first_principles import dynamics as rigid_body


@pytest.mark.parametrize("model", ["so_rpy", "so_rpy_rotor"])
@pytest.mark.parametrize("rpy", [(0.0, 0.2, 0.0), (0.2, 0.0, 0.0), (0.17, -0.23, 0.9)])
def test_world_thrust_matches_rotation_symbolic_and_rigid_body(
    model: str, rpy: tuple[float, float, float]
) -> None:
    """Positive pitch accelerates +x, positive roll -y; yaw rotates horizontal thrust."""
    angles = np.asarray(rpy)
    rotation = Rotation.from_euler("xyz", angles)
    module = importlib.import_module(f"crazyflow.dynamics.{model}.dynamics")
    params = load_params(Dynamics(model), Drone.cf21B_500)
    total_thrust = 0.5
    raw_thrust = (total_thrust - params["acc_coef"]) / params["cmd_f_coef"]
    command = np.array([0.0, 0.0, 0.0, raw_thrust])
    zeros = np.zeros(3)
    extra = {"rotor_vel": np.full(4, raw_thrust)} if model == "so_rpy_rotor" else {}
    actual = parametrize(module.dynamics_euler, Drone.cf21B_500)(
        zeros, angles, zeros, zeros, command, **extra
    )[2]
    expected = rotation.apply([0.0, 0.0, total_thrust]) / params["mass"] + params["gravity_vec"]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    if rpy == (0.0, 0.2, 0.0):
        assert actual[0] > 0
    elif rpy == (0.2, 0.0, 0.0):
        assert actual[1] < 0

    symbolic_extra = {"model_rotor_vel": True} if extra else {}
    derivative, state, action, _ = parametrize(module.symbolic_dynamics_euler, Drone.cf21B_500)(
        **symbolic_extra
    )
    symbolic = cs.Function("world_thrust", [state, action], [derivative])
    state_value = np.concatenate((zeros, angles, zeros, zeros, *extra.values()))
    symbolic_acc = np.asarray(symbolic(state_value, command)).ravel()[6:9]
    np.testing.assert_allclose(symbolic_acc, expected, atol=1e-12)

    # Give the physical rotor model the same collective thrust, independently of the fitted gains.
    rigid_params = load_params(Dynamics.first_principles, Drone.cf21B_500)
    k0, k1, k2 = rigid_params["rpm2thrust"]
    rpm = (-k1 + np.sqrt(k1**2 - 4 * k2 * (k0 - total_thrust / 4))) / (2 * k2)
    motors = np.full(4, rpm)
    rigid_acc = parametrize(rigid_body, Drone.cf21B_500)(
        zeros, rotation.as_quat(), zeros, zeros, motors, rotor_vel=motors
    )[2]
    np.testing.assert_allclose(rigid_acc, expected, atol=1e-12)
