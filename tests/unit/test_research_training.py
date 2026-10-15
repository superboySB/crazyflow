"""CPU regression checks for the research trainer's episode accounting."""

import gymnasium
import jax.numpy as jnp
import numpy as np
import pytest

# PyTorch is an optional research dependency, not a Crazyflow installation requirement.
torch = pytest.importorskip("torch")
trainer = pytest.importorskip("usage_assets.train_gymnasium")

pytestmark = pytest.mark.unit


def test_gae_terminal_reset_and_time_limit() -> None:
    # Hand calculation: the first episode terminates with return 2; the ignored reset
    # reward 999 contributes nothing; the truncated episode bootstraps 3 + 0.9 * 10.
    rewards = torch.tensor([[2.0], [999.0], [3.0]])
    values = torch.tensor([[1.0], [100.0], [4.0]])
    done_after = torch.tensor([[True], [False], [True]])
    valid = torch.tensor([[True], [False], [True]])
    advantages, returns = trainer.compute_gae(
        rewards,
        values,
        torch.tensor([500.0]),
        done_after,
        valid,
        torch.tensor([[0.0], [0.0], [10.0]]),
        gamma=0.9,
        gae_lambda=0.95,
    )
    torch.testing.assert_close(advantages[:, 0], torch.tensor([1.0, 0.0, 8.0]))
    torch.testing.assert_close(returns[[0, 2], 0], torch.tensor([2.0, 12.0]))


def test_gae_continuation_does_not_cross_reset() -> None:
    # At the time limit: delta = 2 + .9 * 5 - 1 = 5.5.
    # One step earlier: A = 1 + .9 * 1 - .5 + .9 * .8 * 5.5 = 5.36.
    # A later episode's terminal return is 3 regardless of the reset reward/value.
    rewards = torch.tensor([[1.0], [2.0], [999.0], [3.0]])
    values = torch.tensor([[0.5], [1.0], [100.0], [2.0]])
    done_after = torch.tensor([[False], [True], [False], [True]])
    valid = torch.tensor([[True], [True], [False], [True]])
    args = (
        rewards,
        values,
        torch.tensor([1000.0]),
        done_after,
        valid,
        torch.tensor([[0.0], [5.0], [0.0], [0.0]]),
        0.9,
        0.8,
    )
    advantages, returns = trainer.compute_gae(*args)
    torch.testing.assert_close(advantages[:, 0], torch.tensor([5.36, 5.5, 0.0, 1.0]))
    torch.testing.assert_close(returns[[0, 1, 3], 0], torch.tensor([5.86, 6.5, 3.0]))
    rewards[2] = -1_000_000.0
    values[2] = -500_000.0
    changed_advantages, changed_returns = trainer.compute_gae(*args)
    torch.testing.assert_close(changed_advantages, advantages)
    torch.testing.assert_close(changed_returns[valid], returns[valid])


def make_cpu_env() -> gymnasium.vector.VectorEnv:
    """Create a tiny official environment whose time limits exercise NEXT_STEP resets."""
    return trainer.NormalizeActions(
        gymnasium.make_vec(
            "DroneReachVel-v0", num_envs=2, freq=50, max_episode_time=0.04, device="cpu"
        )
    )


def test_official_next_step_reset_rewards_are_excluded() -> None:
    env = make_cpu_env()
    try:
        trainer.reset_env(env, seed=123)
        previous_done = np.zeros(2, dtype=bool)
        valid, rewards, done = [], [], []
        reset_calls = 0
        for _ in range(8):
            _, reward, terminated, truncated, _ = env.step(jnp.zeros((2, 4)))
            valid.append(~previous_done)
            rewards.append(np.asarray(reward))
            previous_done = np.asarray(terminated | truncated)
            done.append(previous_done)
            if not valid[-1].all():
                reset_calls += 1
                # Crazyflow returns a nonzero initial-state reward on the reset call,
                # although the sampled action did not produce that returned state.
                assert np.all(rewards[-1][~valid[-1]] > 0)
                np.testing.assert_array_equal(np.asarray(env.unwrapped.sim.time), 0)
        assert reset_calls >= 2
        valid_tensor = torch.tensor(np.stack(valid))
        reward_tensor = torch.tensor(np.stack(rewards))
        advantages, returns = trainer.compute_gae(
            reward_tensor,
            torch.zeros_like(reward_tensor),
            torch.zeros(2),
            torch.tensor(np.stack(done)),
            valid_tensor,
            torch.zeros_like(reward_tensor),
            gamma=0.0,
            gae_lambda=0.95,
        )
        assert valid_tensor.sum() < valid_tensor.numel()
        torch.testing.assert_close(
            advantages[~valid_tensor], torch.zeros_like(advantages[~valid_tensor])
        )
        torch.testing.assert_close(returns[valid_tensor], reward_tensor[valid_tensor])
    finally:
        env.close()


def test_environment_recovery_preserves_pending_reset_and_goal_rng() -> None:
    env = make_cpu_env()
    try:
        trainer.reset_env(env, seed=321)
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(jnp.zeros((2, 4)))
            if np.all(np.asarray(terminated | truncated)):
                break
        else:
            pytest.fail("Tiny episode did not reach its official time limit")
        state = trainer.snapshot_environment(env)
        expected = env.step(jnp.zeros((2, 4)))
        trainer.restore_environment(env, state)
        recovered = env.step(jnp.zeros((2, 4)))
        for key, value in expected[0].items():
            np.testing.assert_array_equal(value, recovered[0][key])
        for value, restored in zip(expected[1:4], recovered[1:4], strict=True):
            np.testing.assert_array_equal(value, restored)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ("DroneReachPos-v0", "DroneReachVel-v0"))
def test_goal_rng_is_independent_reproducible_and_survives_autoreset(env_id: str) -> None:
    """Catch affine goal/initial-position coupling across all worlds and later episode resets."""
    env = trainer.NormalizeActions(
        gymnasium.make_vec(env_id, num_envs=1024, freq=50, max_episode_time=0.04, device="cpu")
    )
    try:
        first, _ = trainer.reset_env(env, seed=31001)
        first = {key: np.asarray(value).copy() for key, value in first.items()}
        # The initial simulator state must retain the old seed rather than being changed to
        # hide the coupling. Its world 0 initial position is fixed by the official simulator.
        np.testing.assert_allclose(first["pos"][0], [0.18708324, 0.06038117, 1.1732959], atol=1e-6)
        reset, _ = trainer.reset_env(env, seed=31001)
        for key in first:
            np.testing.assert_array_equal(first[key], reset[key])
        for phase in ("initial", "autoreset"):
            obs = reset
            position = np.asarray(obs["pos"])
            key = (
                "difference_to_goal" if env_id == "DroneReachPos-v0" else "difference_to_target_vel"
            )
            target = np.asarray(
                obs["pos"] if env_id == "DroneReachPos-v0" else obs["vel"]
            ) + np.asarray(obs[key])
            # Old coupled keys produce perfect component-wise correlation and, for ReachPos,
            # exactly zero x/y goal offsets in every single world.
            joint = np.corrcoef(position.T, target.T)
            assert np.max(np.abs(joint[:3, 3:])) < 0.15, (phase, joint)
            assert np.all(np.ptp(target - position, axis=0) > 1.0), phase
            if phase == "initial":
                for _ in range(8):
                    obs, _, terminated, truncated, _ = env.step(jnp.zeros((1024, 4)))
                    if np.asarray(env.unwrapped.sim.time).max() == 0:
                        reset = obs
                        break
                else:
                    pytest.fail("Official NEXT_STEP autoreset was not reached")
        # Starting from the identical seed reproduces the entire sequence through an autoreset.
        sequences = []
        for _ in range(2):
            trainer.reset_env(env, seed=31001)
            sequence = []
            for _ in range(4):
                outputs = env.step(jnp.zeros((1024, 4)))
                sequence.append(
                    {key: np.asarray(value).copy() for key, value in outputs[0].items()}
                )
            sequences.append(sequence)
        for a, b in zip(*sequences, strict=True):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
    finally:
        env.close()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_calibrated_teacher_inverts_official_fitted_dynamics(dtype: torch.dtype) -> None:
    """Commands must realize desired acceleration at the fitted attitude equilibrium."""
    from crazyflow.control import Control
    from crazyflow.drones import Drone
    from crazyflow.dynamics import Dynamics, load_params
    from crazyflow.dynamics.so_rpy.dynamics import dynamics_euler
    from crazyflow.envs.drone_env import action_space

    rng = np.random.default_rng(931)
    desired = np.concatenate((rng.uniform(-4, 4, (512, 3)), [[4, 4, 4], [-4, -4, -4], [0, 0, 0]]))
    acceleration = torch.as_tensor(desired, dtype=dtype)
    normalized = trainer.acceleration_to_action(acceleration, "calibrated")
    assert torch.isfinite(normalized).all()
    assert (normalized.abs() <= 1).all()
    bounds = action_space(Control.attitude, Dynamics.so_rpy, Drone.cf2x_L250)
    physical = trainer.NormalizeActions._scale_actions(
        jnp.asarray(normalized.numpy()),
        jnp.asarray((bounds.high - bounds.low) / 2),
        jnp.asarray((bounds.high + bounds.low) / 2),
    )
    physical = np.asarray(physical, dtype=np.float64)
    params = load_params(Dynamics.so_rpy, Drone.cf2x_L250)
    gain = -np.asarray(params["cmd_rpy_coef"]) / np.asarray(params["rpy_coef"])
    steady_rpy = physical[:, :3] * gain
    zeros = np.zeros_like(steady_rpy)
    names = (
        "mass",
        "gravity_vec",
        "acc_coef",
        "cmd_f_coef",
        "rpy_coef",
        "rpy_rates_coef",
        "cmd_rpy_coef",
    )
    _, _, actual_acceleration, attitude_acceleration = dynamics_euler(
        zeros, steady_rpy, zeros, zeros, physical, **{key: params[key] for key in names}
    )
    np.testing.assert_allclose(actual_acceleration, desired, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(attitude_acceleration, 0, atol=1e-10)
    # Extreme requests still honor the original +/-4 m/s^2 acceleration and actuator bounds.
    saturated = trainer.acceleration_to_action(
        torch.tensor([[100.0, -100.0, 100.0]], dtype=dtype), "calibrated"
    )
    clipped = trainer.acceleration_to_action(
        torch.tensor([[4.0, -4.0, 4.0]], dtype=dtype), "calibrated"
    )
    torch.testing.assert_close(saturated, clipped)
