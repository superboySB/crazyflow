"""Native MuJoCo owns the flight state; official motor/controller laws supply forces."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from crazyflow.balloon import model, mujoco_rollout
from crazyflow.balloon.config import Config
from crazyflow.balloon.evaluate import summarize_trajectories
from crazyflow.balloon.policy import init_policy
from crazyflow.balloon.rollout import rollout as jax_rollout
from crazyflow.sim.data import SimData
from crazyflow.sim.dynamics import first_principles_dynamics


def _physical_config() -> Config:
    return Config(
        steps=3,
        hidden=8,
        model_params={
            "mass": 0.047,
            "gravity_vec": [0.2, -0.1, -9.7],
            "J": [[26e-6, 2e-6, -1e-6], [2e-6, 30e-6, 1.5e-6], [-1e-6, 1.5e-6, 45e-6]],
        },
    )


def _rigid_inertia(native: mujoco.MjModel, body: int) -> np.ndarray:
    # MuJoCo diagonalizes fullinertia into a separate principal-axis frame.
    xyzw = np.asarray(native.body_iquat[body])[[1, 2, 3, 0]]
    frame = Rotation.from_quat(xyzw).as_matrix()
    return frame @ np.diag(native.body_inertia[body]) @ frame.T


def _excited_data(cfg: Config) -> tuple[model.Batch, SimData]:
    batch = model.sample_batch(jax.random.key(21), cfg, 2, "nominal")
    batch["state"] = jnp.array(
        [
            [0.1, -0.2, 1.4, 0.4, -0.25, 0.7, 0.7, -0.4, 0.3, 0.5, -0.6, 0.2],
            [-0.3, 0.1, 1.7, -0.2, 0.3, -0.5, -0.5, 0.2, -0.4, -0.4, 0.3, -0.5],
        ]
    )
    batch["domain"]["mass_scale"] = jnp.array([0.91, 1.13])
    batch["domain"]["thrust_scale"] = jnp.array([1.07, 0.94])
    batch["domain"]["response_scale"] = jnp.array([0.88, 1.09])
    data = model.initialize(batch, cfg)
    data = data.replace(
        states=data.states.replace(
            rotor_vel=jnp.array([[[15000, 17000, 16000, 18000]], [[18100, 14900, 16900, 15700]]]),
            force=jnp.array([[[0.02, -0.01, 0.015]], [[-0.01, 0.025, -0.005]]]),
            torque=jnp.array([[[1e-5, -2e-5, 3e-5]], [[-2e-5, 1.5e-5, -1e-5]]]),
        ),
        controls=data.controls.replace(
            rotor_vel=jnp.array([[[18100, 15200, 17100, 15500]], [[15300, 18000, 15800, 17700]]])
        ),
    )
    return batch, data


def test_native_model_freejoint_uses_frozen_full_inertia_mass_and_gravity():
    cfg = _physical_config()
    mass_scale = 1.17
    native = mujoco_rollout.make_flight_model(cfg, mass_scale=mass_scale)
    body = native.body("drone").id
    params = model.nominal_params(cfg)
    assert native.nq == 7 and native.nv == 6 and native.nmocap == 0
    assert native.jnt_type[native.body_jntadr[body]] == mujoco.mjtJoint.mjJNT_FREE
    np.testing.assert_allclose(native.body_mass[body], float(params["mass"]) * mass_scale)
    np.testing.assert_allclose(_rigid_inertia(native, body), params["J"], atol=1e-12, rtol=2e-7)
    np.testing.assert_allclose(native.body_ipos[body], 0, atol=1e-12)
    np.testing.assert_allclose(native.opt.gravity, params["gravity_vec"])
    assert native.opt.timestep == pytest.approx(cfg.dt / cfg.substeps)
    # Official aerodynamic drag is already included in the supplied wrench.
    np.testing.assert_array_equal(native.dof_damping, np.zeros(native.nv))
    np.testing.assert_array_equal(native.dof_armature, np.zeros(native.nv))


def test_world_wrench_retains_motor_propeller_drag_and_external_effects_once():
    cfg = _physical_config()
    _, data = _excited_data(cfg)
    derivative = first_principles_dynamics(data)
    force, torque = mujoco_rollout.world_wrench(data)
    force, torque = np.asarray(force), np.asarray(torque)
    assert force.shape == (2, 3) and torque.shape == (2, 3)
    expected_force = data.params.mass * (derivative.acc - data.params.gravity_vec)
    omega = np.asarray(data.states.ang_vel)
    inertia = np.asarray(data.params.J)
    angular_momentum = (inertia @ omega[..., None])[..., 0]
    expected_body_torque = (inertia @ np.asarray(derivative.ang_acc)[..., None])[..., 0]
    expected_body_torque += np.cross(omega, angular_momentum)
    rotation = Rotation.from_quat(np.asarray(data.states.quat[:, 0])).as_matrix()
    expected_world_torque = np.einsum("nij,nj->ni", rotation, expected_body_torque[:, 0])
    np.testing.assert_allclose(force, expected_force[:, 0], atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(torque, expected_world_torque, atol=2e-8, rtol=2e-5)


def test_native_acceleration_matches_official_derivative_in_world_and_body_frames():
    cfg = _physical_config()
    batch, data = _excited_data(cfg)
    derivative = first_principles_dynamics(data)
    force, torque = mujoco_rollout.world_wrench(data)
    for episode in range(2):
        native = mujoco_rollout.make_flight_model(
            cfg, mass_scale=float(batch["domain"]["mass_scale"][episode])
        )
        world = mujoco.MjData(native)
        body = native.body("drone").id
        quat = np.asarray(data.states.quat[episode, 0])
        world.qpos[:3] = data.states.pos[episode, 0]
        world.qpos[3:7] = quat[[3, 0, 1, 2]]
        world.qvel[:3] = data.states.vel[episode, 0]
        world.qvel[3:6] = data.states.ang_vel[episode, 0]
        world.xfrc_applied[body, :3] = force[episode]
        world.xfrc_applied[body, 3:] = torque[episode]
        mujoco.mj_forward(native, world)
        np.testing.assert_allclose(world.qacc[:3], derivative.acc[episode, 0], atol=3e-5)
        np.testing.assert_allclose(
            world.qacc[3:6], derivative.ang_acc[episode, 0], atol=3e-4, rtol=3e-5
        )


def test_native_hover_runs_persistent_controller_without_rigid_state_drift():
    cfg = Config(steps=3, hidden=8)
    batch = model.sample_batch(jax.random.key(4), cfg, 1, "nominal")
    batch["state"] = jnp.zeros((1, 12)).at[:, 2].set(1.5)
    plant = mujoco_rollout.NativePlant(batch, cfg)
    hover = model.command(jnp.zeros((1, 4)), cfg, model.nominal_params(cfg))
    for _ in range(10):
        states, contacts = plant.advance(hover)
        assert states.shape == (cfg.substeps, 1, 12)
        assert contacts.shape == (cfg.substeps, 1)
        assert not np.asarray(contacts).any()
    np.testing.assert_allclose(states[-1], batch["state"], atol=5e-6)
    np.testing.assert_allclose(model.state_vector(plant.data), states[-1], atol=1e-6)
    assert plant.worlds[0].time == pytest.approx(10 * cfg.dt)
    np.testing.assert_allclose(np.linalg.norm(plant.worlds[0].qpos[3:7]), 1.0, atol=1e-12)


def test_native_balloon_contact_is_created_by_flight_and_excludes_floor_contacts():
    cfg = Config(steps=5, hidden=8)
    batch = model.sample_batch(jax.random.key(8), cfg, 2, "nominal")
    batch["state"] = jnp.zeros((2, 12)).at[:, 2].set(jnp.array([1.5, 0.05]))
    batch["state"] = batch["state"].at[0, 6].set(1.0)
    batch["target"] = jnp.array([[0.3, 0, 1.52], [2, 0, 1.5]])
    batch["radius"] = jnp.full((2,), 0.15)
    plant = mujoco_rollout.NativePlant(batch, cfg)
    assert plant.worlds[1].ncon > 0  # Native floor contact exists from reset.
    np.testing.assert_array_equal(plant.contacts(), [False, False])
    hover = model.command(jnp.zeros((2, 4)), cfg, model.nominal_params(cfg))
    trajectories, contacts = [], []
    for _ in range(5):
        states, touching = plant.advance(hover)
        trajectories.extend(states)
        contacts.extend(touching)
    contacts, trajectories = np.stack(contacts), np.stack(trajectories)
    assert contacts[:, 0].any()
    assert not contacts[:, 1].any()
    first_contact = np.flatnonzero(contacts[:, 0])[0]
    assert first_contact > 0  # Flight closes an initial positive sphere gap.
    hit = trajectories[first_contact, 0]
    center = hit[:3] + Rotation.from_euler("xyz", hit[3:6]).apply(cfg.collision_offset)
    separation = np.linalg.norm(center - np.asarray(batch["target"][0]))
    assert separation <= cfg.drone_radius + float(batch["radius"][0]) + 1e-8


def test_native_and_jax_first_noisy_measurement_and_policy_command_agree():
    cfg = Config(steps=3, hidden=8)
    key = jax.random.key(1)  # Sensor delays are [0, 2], exercising valid and empty packets.
    batch = model.sample_batch(key, cfg, 2, "noisy")
    policy = init_policy(key, cfg)
    native = mujoco_rollout.rollout(policy, batch, cfg, key, noise=True)
    differentiable = jax_rollout(policy, batch, cfg, key, noise=True)
    # The shared observe runs inside scan for BPTT and in a separate native-policy
    # jit; float32 fusion may differ by one or two final bits.
    np.testing.assert_allclose(native["bbox"][0], differentiable["bbox"][0], atol=2e-7, rtol=0)
    np.testing.assert_allclose(
        native["actions"][0], differentiable["actions"][0], atol=2e-8, rtol=0
    )
    np.testing.assert_array_equal(np.asarray(native["bbox"])[0, :, 4], [1, 0])


@pytest.mark.parametrize("fault", ["nonfinite_wrench", "nonfinite_qpos"])
def test_native_fault_is_persistent_visible_and_isolated_to_its_episode(
    fault: str, monkeypatch: pytest.MonkeyPatch
):
    cfg = Config(steps=3, hidden=8)
    batch = model.sample_batch(jax.random.key(20), cfg, 2, "nominal")
    batch["state"] = jnp.zeros((2, 12)).at[:, 2].set(1.5)
    batch["target"] = jnp.array([[3, 0, 1.5], [3, 0, 1.5]])
    plant = mujoco_rollout.NativePlant(batch, cfg)
    if fault == "nonfinite_wrench":
        actuators = plant._actuators

        def bad_force(current: SimData) -> tuple[SimData, jax.Array, jax.Array]:
            updated, force, torque = actuators(current)
            return updated, force.at[0, 0].set(jnp.nan), torque

        monkeypatch.setattr(plant, "_actuators", bad_force)
    else:
        native_step = mujoco.mj_step

        def bad_position(native: mujoco.MjModel, world: mujoco.MjData) -> None:
            native_step(native, world)
            if world is plant.worlds[0]:
                world.qpos[0] = np.nan

        monkeypatch.setattr(mujoco, "mj_step", bad_position)
    hover = model.command(jnp.zeros((2, 4)), cfg, model.nominal_params(cfg))
    states, contacts = plant.advance(hover)
    assert np.isnan(states[:, 0]).all()
    assert np.isfinite(states[:, 1]).all()
    np.testing.assert_array_equal(plant.failed, [True, False])
    assert not contacts.any()
    failed_time = plant.worlds[0].time
    later, later_contacts = plant.advance(hover)
    assert np.isnan(later[:, 0]).all()
    assert np.isfinite(later[:, 1]).all()
    assert plant.worlds[0].time == failed_time
    assert plant.worlds[1].time == pytest.approx(2 * cfg.dt)
    assert not later_contacts.any()
    # Keep last finite internal data for batched control; output faults must
    # remain explicit so the failed episode stays in the evaluation denominator.
    assert np.isfinite(np.asarray(model.state_vector(plant.data))).all()
    result = summarize_trajectories(
        np.concatenate((np.asarray(batch["state"])[None], states, later)),
        batch["target"],
        batch["radius"],
        dt=cfg.dt / cfg.substeps,
    )
    assert result["episodes"] == 2 and result["successes"] == 0
    assert result["per_episode"][0]["outcome"] == "nonfinite"
    assert result["per_episode"][1]["outcome"] == "timeout"


def test_rollout_calls_native_step_for_every_substep_and_never_jax_rigid_integration(
    monkeypatch: pytest.MonkeyPatch,
):
    import crazyflow.sim.integration as integration

    cfg = replace(Config(), steps=3, hidden=8)
    key = jax.random.key(6)
    batch = model.sample_batch(key, cfg, 2, "noisy")
    policy = init_policy(key, cfg)
    original_step = mujoco.mj_step
    calls = 0

    def native_step(
        native: mujoco.MjModel, world: mujoco.MjData, *args: object, **kwargs: object
    ) -> None:
        nonlocal calls
        calls += 1
        return original_step(native, world, *args, **kwargs)

    def forbid_jax_integration(*_args, **_kwargs):
        raise AssertionError("Native evaluation must not integrate the rigid state with JAX")

    monkeypatch.setattr(mujoco, "mj_step", native_step)
    monkeypatch.setattr(model, "advance", forbid_jax_integration)
    monkeypatch.setattr(integration, "integrate", forbid_jax_integration)
    result = mujoco_rollout.rollout(policy, batch, cfg, key, noise=True)
    assert calls == cfg.steps * cfg.substeps * 2
    expected_shapes = {
        "states": (cfg.steps + 1, 2, 12),
        "substep_states": (cfg.steps * cfg.substeps + 1, 2, 12),
        "bbox": (cfg.steps, 2, 5),
        "actions": (cfg.steps, 2, 4),
        "quaternion_xyzw": (cfg.steps + 1, 2, 4),
        "body_gyro": (cfg.steps + 1, 2, 3),
        "rotor_rpm": (cfg.steps + 1, 2, 4),
        "native_contacts": (cfg.steps * cfg.substeps + 1, 2),
        "loss": (),
    }
    for name, shape in expected_shapes.items():
        assert np.shape(result[name]) == shape, name
        assert np.isfinite(np.asarray(result[name])).all(), name
    np.testing.assert_array_equal(result["states"][0], batch["state"])
    np.testing.assert_array_equal(result["substep_states"][0], batch["state"])
    np.testing.assert_allclose(
        result["states"][1:], result["substep_states"][cfg.substeps :: cfg.substeps], atol=1e-6
    )
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(result["quaternion_xyzw"]), axis=-1), 1.0, atol=2e-6
    )
    np.testing.assert_allclose(result["body_gyro"][0], model.body_rates(batch["state"]), atol=1e-6)
    assert np.max(np.abs(np.asarray(result["actions"]))) <= 1
    assert np.min(np.asarray(result["rotor_rpm"])) >= 0
