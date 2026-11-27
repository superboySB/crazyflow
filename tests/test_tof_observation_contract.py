"""Named physical channels, temporal ordering and image orientation of CrazyRL inputs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import crazyflow.tof.env as environment_module
from crazyflow.tof.config import EnvConfig
from crazyflow.tof.env import ToFEnv


@pytest.fixture(scope="module")
def environment() -> ToFEnv:
    return ToFEnv(
        EnvConfig(num_maps=1, domain_randomization=False, observation_noise=False), 2, "cpu"
    )


def test_named_actor_and_critic_channels_preserve_baseline_order(
    environment: ToFEnv, monkeypatch: pytest.MonkeyPatch
):
    """Distinct values expose frame/channel transposes and swapped physical channels.

    The expected order comes from frozen quad_baseline_env.py:641-711. Sensor
    packets are supplied directly so this checks the interface independently of
    either simulator's scene geometry and ray intersection implementation.
    """
    env = environment
    state = env.reset(jax.random.PRNGKey(205))
    count = env.num_envs
    depth = np.arange(1, 65, dtype=np.float32) * 0.05
    auxiliary = np.array([0.11, 0.22, 0.33, 0.44], np.float32)
    packet = jnp.broadcast_to(jnp.asarray(np.r_[depth, auxiliary]), (count, 68))
    monkeypatch.setattr(environment_module, "raycast", lambda *args: packet)
    monkeypatch.setattr(
        environment_module,
        "nearest_obstacle",
        lambda *args: (jnp.full((count,), 0.5), jnp.tile(jnp.array([0.3, -0.4, 0.0]), (count, 1))),
    )
    old_actor = np.arange(280, dtype=np.float32).reshape(10, 28) + 1000
    old_critic = np.arange(370, dtype=np.float32).reshape(10, 37) + 2000
    states = state.sim.states.replace(
        pos=jnp.tile(jnp.array([1.0, 2.0, 0.9]), (count, 1, 1)),
        quat=jnp.tile(jnp.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]), (count, 1, 1)),
        vel=jnp.tile(jnp.array([3.0, 4.0, 5.0]), (count, 1, 1)),
        ang_vel=jnp.tile(jnp.array([0.1, -0.2, 0.3]), (count, 1, 1)),
    )
    state = state.replace(
        sim=state.sim.replace(states=states),
        goal=jnp.tile(jnp.array([4.0, 6.0, 1.2]), (count, 1)),
        desired_speed=jnp.full((count,), 1.25),
        last_action=jnp.tile(jnp.array([0.1, -0.2, 0.3, 0.4]), (count, 1)),
        wind_mean=jnp.tile(jnp.array([0.2, -0.3, 0.4]), (count, 1)),
        actor_history=jnp.tile(jnp.asarray(old_actor)[None], (count, 1, 1)),
        critic_history=jnp.tile(jnp.asarray(old_critic)[None], (count, 1, 1)),
    )
    observed = env._observe(state)
    frame = np.empty(28, np.float32)
    frame[0:3] = [100, -75, 125]  # body velocity is R_b2w.T @ world velocity, times25
    frame[3:12] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).ravel() * 127
    frame[12:15] = [3, -6, 9]  # body angular velocity, rad/s times30
    frame[15:17] = np.array([3, 4]) / (5 + 1e-6) * 127  # WORLD XY goal direction
    frame[17] = 1.2 * 127  # absolute goal altitude, not direction-z or relative-z
    frame[18] = -0.3 * 30  # current-z minus goal-z
    frame[19] = 1.25 * 25
    frame[20:24] = np.array([0.1, -0.2, 0.3, 0.4]) * 127  # processed preceding action
    frame[24:28] = auxiliary * 30  # left, right, back, down
    np.testing.assert_allclose(observed.obs[:, 252:280], np.tile(frame, (count, 1)), atol=3e-5)
    # Each entire frame precedes the next one; only the oldest frame is discarded.
    np.testing.assert_array_equal(observed.obs[:, :252], np.tile(old_actor[1:].ravel(), (count, 1)))
    # Only CURRENT front depth is appended, with no 10-frame depth interleaving.
    np.testing.assert_array_equal(observed.obs[:, 280:], np.tile(depth * 30, (count, 1)))
    critic = np.asarray(observed.critic_obs[:, 333:370])
    np.testing.assert_allclose(critic[:, :28], np.tile(frame, (count, 1)), atol=3e-5)
    relative_goal = np.array([3, 4, 0.3])
    relative_goal *= 5 / (np.linalg.norm(relative_goal) + 1e-5)
    relative_obstacle = np.array([0.3, -0.4, 0]) * 0.5 / (0.5 + 1e-5)
    expected_privileged = np.r_[relative_goal * 127, relative_obstacle * 127, [6, -9, 12]]
    np.testing.assert_allclose(critic[:, 28:], np.tile(expected_privileged, (count, 1)), atol=3e-5)
    np.testing.assert_array_equal(
        observed.critic_obs[:, :333], np.tile(old_critic[1:].ravel(), (count, 1))
    )
    np.testing.assert_array_equal(observed.critic_obs[:, 370:], np.tile(depth * 30, (count, 1)))


def test_front_pixels_are_rows_then_columns_and_auxiliary_rays_keep_body_axes(environment: ToFEnv):
    """Camera x is forward, image right is body -y, image down is body -z."""
    env = environment
    corner = 3.5 * 20.955 / (8 * 20)
    one_pixel = 20.955 / (8 * 20)
    expected = np.array(
        [
            [1, corner, corner],
            [1, -corner, corner],
            [1, corner, corner - one_pixel],
            [1, -corner, -corner],
        ]
    )
    np.testing.assert_allclose(env.ray_directions[np.array([0, 7, 8, 63])], expected, atol=1e-7)
    auxiliary = np.array([[0, 1, 0], [0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    np.testing.assert_array_equal(env._rays[64:], auxiliary)
    np.testing.assert_allclose(env._offsets[:64], np.tile([0.02, 0, 0], (64, 1)))
    np.testing.assert_allclose(env._offsets[64:], auxiliary * 0.02)
