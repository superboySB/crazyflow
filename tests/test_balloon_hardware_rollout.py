"""Whole-trajectory gradients and native physics with the shared hardware pipeline."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from crazyflow.balloon import model, mujoco_rollout
from crazyflow.balloon.config import Config
from crazyflow.balloon.noise_config import HardwareNoise
from crazyflow.balloon.policy import init_policy
from crazyflow.balloon.rollout import rollout
from crazyflow.sim.dynamics import first_principles_dynamics


def _hardware_config() -> Config:
    # Nonzero amplitudes below are numerical test fixtures, not hardware estimates
    # or proposed defaults. Exercise every new stochastic/timing mechanism.
    return Config(
        steps=8,
        hidden=8,
        hardware=HardwareNoise(
            enabled=True,
            camera_delay_ms=(0, 22),
            camera_packet_loss_prob=0.08,
            bbox_bias_std=0.004,
            gyro_delay_ms=(0, 6),
            attitude_delay_ms=(0, 8),
            gyro_packet_loss_prob=0.04,
            attitude_packet_loss_prob=0.03,
            gyro_drift_std=0.008,
            gyro_scale_std=0.02,
            attitude_bias_std=0.003,
            bias_tau_s=0.35,
            attitude_command_delay_ms=(2, 10),
            thrust_command_delay_ms=(0, 6),
            command_jitter_std_ms=0.8,
            command_jitter_tau_s=0.06,
            inertia_fraction=0.02,
            drag_fraction=0.02,
            motor_thrust_fraction=0.02,
            motor_response_fraction=0.03,
            gust_std=0.04,
            gust_tau_s=0.15,
        ),
    )


def test_full_bptt_with_frozen_nonzero_hardware_noise_matches_directional_derivative():
    cfg = _hardware_config()
    key = jax.random.key(81)
    batch = model.sample_batch(key, cfg, 2, "noisy")
    policy = init_policy(key, cfg)
    objective = jax.jit(lambda p: rollout(p, batch, cfg, key, noise=True)["loss"])
    loss, gradient = jax.value_and_grad(objective)(policy)
    assert np.isfinite(loss)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
    # Perturb all four output biases. Effects must traverse delayed commands,
    # controller/RPM/body integration, delayed sensors and later actor calls.
    direction = jnp.array([0.2, 0.8, -0.3, 0.1])
    direction /= jnp.linalg.norm(direction)
    analytic = jnp.vdot(gradient[-1]["b"], direction)
    epsilon = 0.002
    plus = [*policy[:-1], {**policy[-1], "b": policy[-1]["b"] + epsilon * direction}]
    minus = [*policy[:-1], {**policy[-1], "b": policy[-1]["b"] - epsilon * direction}]
    finite_difference = (objective(plus) - objective(minus)) / (2 * epsilon)
    assert abs(float(analytic)) > 1e-4
    np.testing.assert_allclose(analytic, finite_difference, rtol=0.05, atol=0.004)
    # An identical key freezes the complete noise realization, including OU paths.
    np.testing.assert_array_equal(loss, objective(policy))


def test_hardware_noise_false_preserves_the_legacy_nominal_flight_trajectory():
    hardware_cfg = _hardware_config()
    legacy_cfg = replace(hardware_cfg, hardware=HardwareNoise())
    key = jax.random.key(82)
    legacy_batch = model.sample_batch(key, legacy_cfg, 2, "nominal")
    hardware_batch = model.sample_batch(key, hardware_cfg, 2, "nominal")
    policy = init_policy(key, legacy_cfg)
    legacy = rollout(policy, legacy_batch, legacy_cfg, key, noise=False)
    hardware = rollout(policy, hardware_batch, hardware_cfg, key, noise=False)
    for name in (
        "states",
        "substep_states",
        "bbox",
        "actions",
        "loss",
        "quaternion_xyzw",
        "body_gyro",
        "rotor_rpm",
    ):
        np.testing.assert_allclose(hardware[name], legacy[name], rtol=2e-6, atol=2e-6, err_msg=name)
    for name in ("camera_age_ms", "gyro_age_ms", "attitude_age_ms", "command_delay_ms"):
        np.testing.assert_array_equal(hardware[name], 0)
    np.testing.assert_array_equal(hardware["issued_commands"], hardware["executed_commands"])


def test_native_and_jax_hardware_share_first_tick_commands_and_all_mailbox_timing():
    cfg = _hardware_config()
    key = jax.random.key(83)
    batch = model.sample_batch(key, cfg, 2, "noisy")
    policy = init_policy(key, cfg)
    native = mujoco_rollout.rollout(policy, batch, cfg, key, noise=True)
    differentiable = rollout(policy, batch, cfg, key, noise=True)
    for name in ("bbox", "actions", "issued_commands", "executed_commands"):
        np.testing.assert_allclose(
            native[name][0], differentiable[name][0], rtol=3e-7, atol=3e-7, err_msg=name
        )
    for name in ("camera_age_ms", "gyro_age_ms", "attitude_age_ms", "command_delay_ms"):
        np.testing.assert_array_equal(native[name], differentiable[name], err_msg=name)
    assert native["states"].shape == (cfg.steps + 1, 2, 12)
    assert native["native_contacts"].shape == (cfg.steps * cfg.substeps + 1, 2)
    assert np.isfinite(native["states"]).all()
    # Native semi-implicit rigid-body integration is allowed to produce a
    # different trajectory; this test pins the shared mailbox/RNG contract.


def test_native_rigid_body_uses_randomized_full_inertia_in_its_actual_acceleration():
    cfg = replace(
        _hardware_config(),
        model_params={
            "J": [[25e-6, 1e-6, -0.7e-6], [1e-6, 28e-6, 0.8e-6], [-0.7e-6, 0.8e-6, 49e-6]]
        },
    )
    key = jax.random.key(84)
    batch = model.sample_batch(key, cfg, 2, "noisy")
    plant = mujoco_rollout.NativePlant(batch, cfg)
    derivative = first_principles_dynamics(plant.data)
    force, torque = mujoco_rollout.world_wrench(plant.data)
    assert np.any(np.asarray(batch["domain"]["inertia_scale"]) != 1)
    for episode, (native, world) in enumerate(zip(plant.models, plant.worlds, strict=True)):
        body = native.body("drone").id
        axes = Rotation.from_quat(native.body_iquat[body][[1, 2, 3, 0]]).as_matrix()
        actual_inertia = axes @ np.diag(native.body_inertia[body]) @ axes.T
        np.testing.assert_allclose(
            actual_inertia, plant.data.params.J[episode, 0], rtol=2e-7, atol=2e-12
        )
        np.testing.assert_allclose(native.body_mass[body], plant.data.params.mass[episode, 0, 0])
        world.xfrc_applied[body, :3] = force[episode]
        world.xfrc_applied[body, 3:] = torque[episode]
        mujoco.mj_forward(native, world)
        np.testing.assert_allclose(world.qacc[:3], derivative.acc[episode, 0], atol=3e-5)
        np.testing.assert_allclose(
            world.qacc[3:6], derivative.ang_acc[episode, 0], rtol=3e-5, atol=3e-4
        )
