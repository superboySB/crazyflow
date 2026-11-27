"""Numerical task-contract checks independent of training or an Isaac installation."""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.tof.config import EnvConfig
from crazyflow.tof.env import ToFEnv, reward_terms, squash_action
from crazyflow.tof.scene import (
    Scene,
    _ray_primitive,
    nearest_obstacle,
    primitive_distances,
    raycast,
)


def test_actor_critic_dimensions_and_action_units():
    cfg = EnvConfig()
    assert cfg.integrator == "rk4"
    assert EnvConfig(integrator="euler").integrator == "euler"
    with pytest.raises(ValueError, match="integrator"):
        EnvConfig(integrator="unknown")
    assert cfg.observation_size == 344
    assert cfg.critic_observation_size == 434
    assert cfg.max_episode_steps == 1200
    np.testing.assert_allclose(squash_action(jnp.zeros((2, 4))), [[0, 0, 0, 0.5]] * 2)
    transformed = np.array(
        squash_action(jnp.array([[np.arctanh(0.5), -np.arctanh(0.5), 0, np.arctanh(0.5)]]))
    )
    np.testing.assert_allclose(transformed, [[0.5, -0.5, 0, 0.75]], atol=1e-6)


def test_reward_matches_reference_static_goal_equations():
    cfg = EnvConfig()
    pos = jnp.array([[0.0, 0.0, 1.0]])
    previous = jnp.array([[-0.04, 0.0, 1.0]])
    quat = jnp.array([[0.0, 0.0, 0.0, 1.0]])
    vel = jnp.array([[0.8, 0.2, 0.0]])
    action = jnp.array([[0.1, -0.1, 0.2, 0.5]])
    previous_action = jnp.array([[0.0, 0.0, 0.0, 0.4]])
    goal = jnp.array([[2.0, 0.0, 1.0]])
    reward, components, desired = reward_terms(
        cfg,
        pos,
        previous,
        quat,
        vel,
        action,
        previous_action,
        goal,
        jnp.array([0.7]),
        jnp.array([0.3]),
        jnp.array([False]),
        jnp.array([False]),
    )
    speed = np.sqrt(0.8**2 + 0.2**2)
    misalignment = speed - 0.8
    expected = np.array(
        [
            0.01 * 8,
            (1 / (1 + 1e-6) * 2 / (2 + 1e-6) - 1) * 0.3,
            -np.sqrt(0.1**2 + 0.1**2 + 0.2**2 + 0.5**2) * 0.3,
            -np.sqrt(0.1**2 + 0.1**2 + 0.2**2 + 0.1**2) * 0.5,
            -0.5 * misalignment**2 / 0.2 * 0.15,
            -min(np.exp((speed - 0.7) * 5) - 1, 5) * 0.5,
            0,
            0,
            0,
            -(np.exp(5 * (0.5 - 0.3)) - 1) * 0.1,
            0,
        ]
    )
    np.testing.assert_allclose(components[0], expected, atol=1e-6)
    np.testing.assert_allclose(reward[0], expected.sum(), atol=1e-6)
    np.testing.assert_allclose(desired, [0.7])


def test_reward_progress_is_only_upper_clipped_and_terminal_bonus_is_unscaled():
    cfg = EnvConfig(
        reward_yaw=0,
        reward_action=0,
        reward_action_change=0,
        reward_velocity_direction=0,
        reward_speed_excess=0,
        reward_height=0,
        reward_esdf=0,
    )
    arguments = (
        cfg,
        jnp.array([[0.0, 0.0, 1.0]]),
        jnp.array([[0.2, 0.0, 1.0]]),
        jnp.array([[0.0, 0.0, 0.0, 1.0]]),
        jnp.zeros((1, 3)),
        jnp.zeros((1, 4)),
        jnp.zeros((1, 4)),
        jnp.array([[2.0, 0.0, 1.0]]),
        jnp.ones((1,)),
        jnp.ones((1,)),
        jnp.array([True]),
        jnp.array([True]),
    )
    reward, terms, _ = reward_terms(*arguments)
    np.testing.assert_allclose(terms[0, 0], -0.2 * 8, atol=2e-6)
    assert float(terms[0, 8]) == -80
    assert float(terms[0, 10]) == 80
    np.testing.assert_allclose(reward, [-1.6], atol=2e-6)


def test_primitive_signed_distances_and_closest_vectors():
    points = jnp.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.9], [0.0, 0.0, 2.0]])
    sizes = jnp.ones((3, 3))
    distance, vector = primitive_distances(points, sizes, jnp.array([0, 1, 2]))
    np.testing.assert_allclose(distance, [1, 1, 1], atol=1e-6)
    np.testing.assert_allclose(vector, [[-1, 0, 0], [-1, 0, 0], [0, 0, -1]], atol=1e-6)


def test_oriented_image_plane_raycast_and_missing_depth():
    scene = Scene(
        jnp.array([[[2.0, 0.0, 1.0]]]),
        jnp.array([[[0.5, 0.5, 0.5]]]),
        jnp.zeros((1, 1), jnp.int32),
        jnp.eye(3)[None, None],
        jnp.zeros((1, 1, 3)),
        np.array([[1, 0]]),
        room_half_size=11,
        ceiling=8,
    )
    origin = jnp.array([[[0.0, 0.0, 1.0]] * 4])
    rays = jnp.array([[[1.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]])
    depth = raycast(scene, origin, rays, jnp.array([0]), 4)
    # Oblique image-plane depth is unchanged, rather than Euclidean range.
    np.testing.assert_allclose(depth, [[1.5, 1.5, 4, 0.99]], atol=1e-5)
    distance, vector = nearest_obstacle(scene, jnp.array([[1.0, 0.0, 1.0]]), jnp.array([0]))
    np.testing.assert_allclose(distance, [0.5])
    np.testing.assert_allclose(vector, [[0.5, 0, 0]])


@pytest.fixture(scope="module")
def environment() -> ToFEnv:
    return ToFEnv(
        EnvConfig(num_maps=1, domain_randomization=False, observation_noise=False), 8, "cpu"
    )


def test_native_vectorized_reset_history_and_repeatability(environment: ToFEnv):
    env = environment
    state = env.reset(jax.random.PRNGKey(42))
    duplicate = env.reset(jax.random.PRNGKey(42))
    assert state.obs.shape == (8, 344)
    assert state.critic_obs.shape == (8, 434)
    np.testing.assert_array_equal(state.obs, duplicate.obs)
    np.testing.assert_array_equal(state.actor_history[:, :-1], 0)
    np.testing.assert_array_equal(state.critic_history[:, :-1], 0)
    np.testing.assert_allclose(state.sim.params.mass, env.nominal_params.mass[0])
    np.testing.assert_array_equal(state.thrust_effectiveness, 1)
    assert state.sim.core.n_worlds == 8
    assert state.sim.core.rng_key.dtype == jnp.uint32
    assert np.min(np.asarray(state.depth)) >= 0
    assert np.max(np.asarray(state.depth)) <= 4
    assert np.all(np.abs(np.asarray(state.sim.states.pos[:, 0, :2])) <= 7)
    np.testing.assert_allclose(np.asarray(state.sim.states.pos[:, 0, 2]) * 5 % 1, 0.5, atol=1e-5)


def test_time_limit_bootstrap_observation_and_autoreset(environment: ToFEnv):
    env = environment
    state = env.reset(jax.random.PRNGKey(7))
    state = state.replace(steps=jnp.full((8,), env.config.max_episode_steps - 1, jnp.int32))
    following, result = env.step(state, jnp.zeros((8, 4)))
    assert np.all(result.truncated)
    assert np.all(result.done)
    assert np.all(np.asarray(following.steps) == 0)
    assert result.terminal_critic_obs.shape == (8, 434)
    assert np.all(np.isfinite(np.asarray(result.reward)))
    assert np.all(result.episode_length == env.config.max_episode_steps)
    assert not np.array_equal(following.obs, result.terminal_obs)


def test_physics_has_no_action_gradient(environment: ToFEnv):
    env = environment
    state = env.reset(jax.random.PRNGKey(10))
    grad = jax.grad(lambda action: env.step(state, action)[1].reward.sum())(jnp.zeros((8, 4)))
    np.testing.assert_array_equal(grad, 0)


def test_domain_randomization_ranges(environment: ToFEnv):
    # Controller mass/inertia, not the actual physical mass, are randomized in
    # the reference. Construction samples this once, rather than on each reset.
    env = ToFEnv(dataclasses.replace(environment.config, domain_randomization=True), 32, "cpu")
    ratio = np.asarray(env.controller_mass) / 0.049
    assert np.all((ratio >= 0.8) & (ratio <= 1.2))
    assert np.ptp(ratio) > 0.1
    state = env.reset(jax.random.PRNGKey(1))
    again = env.reset(jax.random.PRNGKey(2))
    np.testing.assert_allclose(state.sim.params.mass, env.nominal_params.mass[0])
    np.testing.assert_array_equal(state.sim.params.mass, again.sim.params.mass)
    eta = np.asarray(state.thrust_effectiveness)
    assert np.all((eta >= 0.85) & (eta <= 1))
    degradation = np.asarray(state.degradation_rate) * env.cfg.max_episode_steps
    assert np.all((degradation >= 0.1) & (degradation <= 0.15))


def test_dual_map_curriculum_preserves_occupied_map_and_can_resume():
    cfg = EnvConfig(
        num_maps=2,
        map_generation_reset_batches=1,
        domain_randomization=False,
        observation_noise=False,
    )
    env = ToFEnv(cfg, 4, "cpu")
    state = env.reset(jax.random.PRNGKey(3))
    occupied = int(state.map_id[0])
    old_map = np.array(env.scene.centers[occupied])
    state = state.replace(
        map_reset_batches=jnp.array(1, jnp.int32),
        recent_outcomes=jnp.ones((100,)),
        episode_count=jnp.array(100, jnp.int32),
    )
    state, changed = env.maybe_regenerate(state)
    assert changed
    assert int(state.map_reset_batches) == 0
    assert env.cfg.hover_hold_seconds == pytest.approx(0.1)
    assert env.cfg.obstacle_min_distance == pytest.approx(1.5 / 1.1)
    np.testing.assert_array_equal(np.array(env.scene.centers[occupied])[: len(old_map)], old_map)
    fresh = env.reset(jax.random.PRNGKey(4))
    assert np.all(np.array(fresh.map_id) == 1 - occupied)
    runtime = env.runtime_state()
    restored = ToFEnv(cfg, 4, "cpu")
    restored.restore_runtime(runtime)
    np.testing.assert_array_equal(restored.scene.centers, env.scene.centers)
    np.testing.assert_array_equal(restored.scene.half_sizes, env.scene.half_sizes)
    assert restored.runtime_state() == runtime
    # New active actors still occupy the old inactive map, so regeneration must
    # leave its geometry untouched until those actors finish.
    before = np.array(env.scene.centers)
    state = state.replace(map_reset_batches=jnp.array(1, jnp.int32))
    env.maybe_regenerate(state)
    np.testing.assert_array_equal(env.scene.centers, before)


def test_identified_physics_are_embedded_in_portable_metadata(tmp_path: Path):
    path = tmp_path / "physical.toml"
    path.write_text("[cf21B_500]\nmass = 0.05\n")
    cfg = EnvConfig(num_maps=1, model_params_path=str(path))
    env = ToFEnv(cfg, 2, "cpu")
    metadata = env.effective_config()
    assert metadata["integrator"] == "rk4"
    assert metadata["model_params_path"] is None
    assert metadata["model_params"]["mass"] == 0.05
    assert "J" in metadata["model_params"]
    assert "rpm2thrust" in metadata["model_params"]
    path.unlink()
    restored = ToFEnv(EnvConfig(**metadata), 2, "cpu")
    np.testing.assert_allclose(restored.nominal_params.mass, [0.05])
    np.testing.assert_allclose(restored.nominal_params.J, env.nominal_params.J)


def test_native_rk4_controller_attitude_response_is_numerically_stable(environment: ToFEnv):
    """The default 150 Hz Euler discretization oscillates for this native motor/PD pair."""
    from jax.scipy.spatial.transform import Rotation

    from crazyflow.control.transform import motor_force2rotor_vel

    env = environment
    state = env.reset(jax.random.PRNGKey(123))
    p = env.nominal_params
    mass = p.mass[0]
    hover_rpm = motor_force2rotor_vel(jnp.array([mass * 9.81 / 4]), p.rpm2thrust)
    attitude = Rotation.from_euler("xyz", jnp.array([0.2, -0.1, 0.0])).as_quat()
    states = state.sim.states.replace(
        pos=jnp.broadcast_to(jnp.array([0.0, 0.0, 10.0]), (env.num_envs, 1, 3)),
        quat=jnp.broadcast_to(attitude, (env.num_envs, 1, 4)),
        rotor_vel=jnp.broadcast_to(hover_rpm, (env.num_envs, 1, 4)),
    )
    action = jnp.broadcast_to(
        jnp.array([0.0, 0.0, 0.0, mass * 9.81 / env.reference_max_thrust]), (env.num_envs, 4)
    )
    state = state.replace(
        sim=state.sim.replace(states=states),
        action_buffer=jnp.broadcast_to(action[:, None], state.action_buffer.shape),
    )

    def one(current: object, _: None) -> tuple[object, jax.Array]:
        following, _ = env._physics(current, action)
        angles = Rotation.from_quat(following.sim.states.quat[:, 0]).as_euler("xyz")
        return following, angles

    _, angles = jax.jit(lambda initial: jax.lax.scan(one, initial, None, length=120))(state)
    rms = np.sqrt(np.mean(np.asarray(angles[-15:, :, 0]) ** 2))
    assert rms < 0.01


def test_nominal_physics_export_roundtrip_keeps_native_dynamics(environment: ToFEnv):
    """Freezing nominal TOML data must retain zero, hover and saturated commands."""
    env = environment
    frozen = env.effective_config()
    assert frozen["model_params"]["mass"] == pytest.approx(0.0434)
    assert "rpm2thrust" in frozen["model_params"]
    assert "rotor_dyn_coef" in frozen["model_params"]
    assert frozen["model_params_path"] is None
    assert env.runtime_state()["config"]["model_params"] == frozen["model_params"]
    restored = ToFEnv(EnvConfig(**frozen), env.num_envs, "cpu")
    initial = env.reset(jax.random.PRNGKey(456))
    duplicate = restored.reset(jax.random.PRNGKey(456))
    hover = float(env.nominal_params.mass[0]) * 9.81 / env.reference_max_thrust
    command = (
        jnp.zeros((env.num_envs, 4))
        .at[1, 3]
        .set(hover)
        .at[2, :]
        .set(jnp.array([1.0, -1.0, 1.0, 1.0]))
    )

    def simulate(task: ToFEnv, state: object) -> object:
        def one(current: object, _: None) -> tuple[object, None]:
            following, _ = task._physics(current, command)
            return following, None

        return jax.jit(lambda value: jax.lax.scan(one, value, None, length=10)[0])(state)

    following = simulate(env, initial)
    replay = simulate(restored, duplicate)
    for original, reloaded in zip(jax.tree.leaves(following.sim), jax.tree.leaves(replay.sim)):
        np.testing.assert_array_equal(original, reloaded)


def test_axis_parallel_cylinder_rays_use_caps_and_reject_missed_sides():
    origins = jnp.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 0.0],
            [1.0, -2.0, 0.0],
        ]
    )
    rays = jnp.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    depth = _ray_primitive(origins, rays, jnp.ones_like(origins), jnp.ones((6,), jnp.int32))
    np.testing.assert_allclose(depth, [np.inf, 1.0, 1.0, np.inf, 0.0, 2.0], atol=1e-6)


def test_downward_tof_ignores_distant_upright_cylinder():
    scene = Scene(
        jnp.array([[[5.0, 5.0, 1.0]]]),
        jnp.array([[[0.5, 0.5, 1.0]]]),
        jnp.ones((1, 1), jnp.int32),
        jnp.eye(3)[None, None],
        jnp.zeros((1, 1, 3)),
        np.array([[1, 0]]),
    )
    depth = raycast(
        scene, jnp.array([[[0.0, 0.0, 1.0]]]), jnp.array([[[0.0, 0.0, -1.0]]]), jnp.array([0])
    )
    np.testing.assert_allclose(depth, [[0.99]], atol=1e-6)


def test_cone_generatrix_linear_and_coincident_intersections():
    origins = jnp.array(
        [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 2.0], [-2.0, 0.0, 2.0], [0.0, 0.0, 2.0]]
    )
    rays = jnp.array(
        [[-0.5, 0.0, -1.0], [0.5, 0.0, -1.0], [0.5, 0.0, -1.0], [0.5, 0.0, -1.0], [0.5, 0.0, 1.0]]
    )
    depth = _ray_primitive(origins, rays, jnp.ones_like(origins), jnp.full((5,), 2, jnp.int32))
    np.testing.assert_allclose(depth, [0.5, 0.0, 1.0, 2.5, np.inf], atol=1e-6)
