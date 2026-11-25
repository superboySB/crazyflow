"""Physical adapter, full gradient and deployment contract regression tests."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.balloon.config import Config
from crazyflow.balloon.model import (
    advance,
    body_rates,
    command,
    control_params,
    initialize,
    nominal_params,
    sample_batch,
    state_vector,
)
from crazyflow.balloon.observations import actor_features, clip_bbox, ideal_bbox
from crazyflow.balloon.policy import Controller, init_policy, load_checkpoint, save_checkpoint
from crazyflow.balloon.rollout import rollout
from crazyflow.control import Control
from crazyflow.drones import Drone
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator


def test_hover_and_gyro_conversion():
    cfg = Config(steps=8)
    params = nominal_params(cfg)
    b = sample_batch(jax.random.key(0), cfg, 1, "nominal")
    state = jnp.zeros((1, 12)).at[:, 2].set(1.5)
    b["state"] = state
    data = initialize(b, cfg, params)
    result, _ = advance(data, command(jnp.zeros((1, 4)), cfg, params), cfg)
    np.testing.assert_allclose(state_vector(result), state, atol=1e-6)
    spin = state.at[:, 4].set(0.3).at[:, 11].set(1)
    np.testing.assert_allclose(body_rates(spin)[0], [-np.sin(0.3), 0, np.cos(0.3)], atol=1e-6)


def test_full_bptt_directional_derivative():
    cfg = Config(steps=12, hidden=8)
    key = jax.random.key(8)
    b = sample_batch(key, cfg, 2, "nominal")
    p = init_policy(key, cfg)
    objective = jax.jit(lambda p: rollout(p, b, cfg, key, noise=False)["loss"])
    grad = jax.grad(objective)(p)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad))
    # Perturb the output pitch bias, whose effect travels through command,
    # physical attitude, position, next bbox and subsequent policy calls.
    eps = 0.002
    plus = [*p[:-1], {**p[-1], "b": p[-1]["b"].at[1].add(eps)}]
    minus = [*p[:-1], {**p[-1], "b": p[-1]["b"].at[1].add(-eps)}]
    finite_difference = (objective(plus) - objective(minus)) / (2 * eps)
    np.testing.assert_allclose(grad[-1]["b"][1], finite_difference, atol=0.003, rtol=0.03)
    assert abs(float(grad[-1]["b"][1])) > 1e-4


def test_position_and_velocity_are_absent_from_actor():
    cfg = Config()
    state = jnp.zeros((2, 12))
    state = state.at[1, :3].set(jnp.array([10, 2, 3])).at[1, 6:9].set(jnp.array([4, -2, 3]))
    target = state[:, :3] + jnp.array([2, 0.1, 0])
    bbox, _ = ideal_bbox(state[:, :3], state[:, 3:6], target, jnp.full(2, 0.15), cfg)
    hist = jnp.repeat(bbox[:, None], cfg.history, 1)
    obs = actor_features(
        hist,
        jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
        body_rates(state),
        jnp.zeros((2, cfg.history, 4)),
        cfg,
    )
    np.testing.assert_allclose(obs[0], obs[1], atol=1e-6)


def test_checkpoint_and_sensor_only_controller(tmp_path: Path):
    cfg = Config(steps=8, hidden=8)
    params = init_policy(jax.random.key(4), cfg)
    path = tmp_path / "policy.npz"
    save_checkpoint(path, params, cfg)
    loaded, contract, metadata = load_checkpoint(path)
    assert (
        replace(contract, model_params=None, controller_params=None, camera_intrinsics=None) == cfg
        and metadata["algorithm"] == "BPTT"
    )
    for key, value in nominal_params(cfg).items():
        np.testing.assert_allclose(value, nominal_params(contract)[key])
    for stage, values in control_params(cfg).items():
        for key, value in values.items():
            np.testing.assert_allclose(value, control_params(contract)[stage][key])
    assert all(
        np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(loaded))
    )
    controller = Controller(path)
    output = controller.step([280, 200, 360, 280], [0, 0, 0, 1], [0, 0, 0])
    assert output.shape == (4,) and np.isfinite(output).all()
    assert 4 * 0.021 < output[3] < 4 * 0.2
    pixels = np.array([620.0, -10.0, 710.0, 110.0])
    raw_box = np.r_[
        (pixels[:2] + pixels[2:]) / [cfg.image_width, cfg.image_height] - 1,
        2 * (pixels[2:] - pixels[:2]) / [cfg.image_width, cfg.image_height],
        1.0,
    ]
    controller.step(pixels, [0, 0, 0, 1], [0, 0, 0])
    np.testing.assert_allclose(controller.boxes[-1], clip_bbox(raw_box), atol=2e-7)
    with pytest.raises(ValueError):
        controller.step(None, [0, 0, 0, 0], [0, 0, 0])


def test_randomized_rollout_reproducible():
    cfg = replace(Config(), steps=8, hidden=8, bbox_noise_std=1.0, dropout_prob=0.0)
    key = jax.random.key(12)
    batch = sample_batch(key, cfg, 2)
    params = init_policy(key, cfg)
    fn = jax.jit(lambda k: rollout(params, batch, cfg, k))
    a, b = fn(key), fn(key)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)
    valid = np.asarray(a["bbox"])[..., 4] > 0.5
    assert valid.any()
    corners = a["bbox"][..., None, :2] + jnp.array([-0.5, 0.5])[:, None] * a["bbox"][..., None, 2:4]
    assert np.max(np.abs(np.asarray(corners))) <= 1.000001


def test_nominal_first_principles_uses_all_original_parameters():
    cfg = Config(steps=8)
    assert Dynamics.default == Dynamics.first_principles
    original = load_params(Dynamics.default, Drone.cf21B_500, xp=jnp)
    current = nominal_params(cfg)
    assert set(current) == set(original)
    for key, value in original.items():
        np.testing.assert_array_equal(current[key], value)
    assert {
        "rpm2thrust",
        "rpm2torque",
        "rotor_dyn_coef",
        "prop_inertia",
        "J",
        "drag_matrix",
    } <= set(current)
    assert "cmd_f_coef" not in current and "rpy_coef" not in current
    batch = sample_batch(jax.random.key(91), cfg, 2, "nominal")
    data = initialize(batch, cfg, current)
    assert data.core.freq == 500 and data.controls.attitude.freq == 500
    assert data.controls.force_torque.freq == 500
    assert np.all(np.asarray(data.states.rotor_vel) > 0)
    np.testing.assert_array_equal(data.params.J, original["J"])
    np.testing.assert_array_equal(data.params.J_inv, original["J_inv"])
    for key in ("rpm2thrust", "rpm2torque", "rotor_dyn_coef"):
        np.testing.assert_array_equal(getattr(data.params, key)[0, 0, 0], original[key])


def test_persistent_adapter_matches_official_sim_every_state_and_controller():
    cfg = Config(steps=8)
    physical = nominal_params(cfg)
    batch = sample_batch(jax.random.key(68), cfg, 2, "nominal")
    data = initialize(batch, cfg, physical)
    official = Sim(
        n_worlds=2,
        n_drones=1,
        drone=Drone.cf21B_500,
        dynamics=Dynamics.default,
        control=Control.attitude,
        integrator=Integrator.default,
        freq=500,
        attitude_freq=500,
        force_torque_freq=500,
    )
    official.data = data
    for tick in range(2):
        actions = jnp.array([[0.12, -0.18, 0.2, 0.03], [-0.07, 0.1, -0.15, -0.02]]) * (tick + 1)
        controls = command(actions, cfg, physical)
        data, substeps = advance(data, controls, cfg)
        official.attitude_control(controls[:, None])
        official.step(n_steps=cfg.substeps)
        assert substeps.shape == (cfg.substeps, 2, 12)
        for actual, expected in zip(jax.tree.leaves(data), jax.tree.leaves(official.data)):
            if hasattr(actual, "dtype") and jax.dtypes.issubdtype(
                actual.dtype, jax.dtypes.prng_key
            ):
                continue
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=2e-6)
    np.testing.assert_array_equal(data.core.steps, 20)
    assert np.max(np.abs(np.asarray(data.controls.attitude.rot_err_i))) > 1e-5
    assert np.max(np.abs(np.asarray(data.controls.attitude.prev_ang_vel))) > 1e-3
