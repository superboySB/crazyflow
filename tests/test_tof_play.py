"""Live play controls and clean/noisy sensor packets without an OpenGL dependency."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.tof.play import LiveFlight, sensor_packet


def observation(front: np.ndarray, auxiliary: np.ndarray) -> np.ndarray:
    """Encode only the two range groups in the real 344-channel actor layout."""
    result = np.zeros((1, 344), np.float32)
    result[0, -64:] = np.asarray(front).reshape(-1) * 30
    result[0, -68:-64] = np.asarray(auxiliary) * 30
    return result


class LiveEnvironment:
    """Return visibly unrelated autoreset packets to expose terminal HUD mistakes."""

    num_envs = 1
    config = SimpleNamespace(dt=1 / 15)
    scene = SimpleNamespace(
        centers=np.zeros((1, 1, 3)),
        half_sizes=np.ones((1, 1, 3)),
        kinds=np.zeros((1, 1), int),
        rotations=np.eye(3)[None, None],
        counts=np.array([[1, 0]]),
        room_half_size=11.0,
        floor=0.01,
        ceiling=2.0,
    )
    ray_directions = np.tile([1.0, 0, 0], (64, 1))

    def reset(self, key: jax.Array) -> SimpleNamespace:
        """Give each explicit reset seed its own pose, goal and sensor packet."""
        seed = int(np.asarray(jax.random.key_data(key))[-1])
        return self.state(seed, 0)

    @staticmethod
    def state(seed: int, steps: int) -> SimpleNamespace:
        """Create one immutable test tick with clean and policy ranges differing."""
        pos = np.array([[[seed + steps, 0.0, 1.0]]], np.float32)
        return SimpleNamespace(
            sim=SimpleNamespace(
                states=SimpleNamespace(
                    pos=pos, quat=np.array([[[0.0, 0.0, 0.0, 1.0]]]), vel=np.zeros((1, 1, 3))
                )
            ),
            goal=np.array([[seed + 10.0, 0, 1]]),
            map_id=np.array([0]),
            depth=np.full((1, 64), 0.4),
            auxiliary_depth=np.array([[1.0, 2.0, 3.0, 4.0]]),
            obs=observation(np.full(64, 0.6), np.array([1.1, 2.2, 3.3, 4.4])),
            seed=seed,
            steps=steps,
        )

    def step(self, state: SimpleNamespace, action: jax.Array) -> tuple:
        """End after two steps and preserve sensor packets before the reset."""
        del action
        step = state.steps + 1
        done = step == 2
        terminal = self.state(state.seed, step)
        critic = np.zeros((1, 434), np.float32)
        critic[0, -64:] = 0.4 * 30
        critic[0, 9 * 37 + 24 : 9 * 37 + 28] = np.array([1, 2, 3, 4]) * 30
        result = SimpleNamespace(
            done=np.array([done]),
            reward=np.array([2.0]),
            terminal_pos=terminal.sim.states.pos[:, 0],
            terminal_quat=terminal.sim.states.quat[:, 0],
            terminal_vel=terminal.sim.states.vel[:, 0],
            terminal_critic_obs=critic,
            terminal_obs=terminal.obs,
            collision=np.array([False]),
            success=np.array([True]),
            instability=np.array([False]),
            truncated=np.array([False]),
        )
        return (self.state(999, 0) if done else terminal), result


def test_sensor_panel_decodes_actual_actor_inputs_without_resampling():
    front = np.arange(64, dtype=np.float32).reshape(8, 8) / 20
    auxiliary = np.array([0.1, 0.2, 0.3, 0.4])
    actual_front, actual_auxiliary = front + 0.5, auxiliary + 0.7
    actor = observation(actual_front, actual_auxiliary)[0]
    packet = sensor_packet(front, auxiliary, actor)
    np.testing.assert_array_equal(packet["clean_front"], front)
    np.testing.assert_array_equal(packet["clean_auxiliary"], auxiliary)
    np.testing.assert_allclose(packet["policy_front"], actual_front)
    np.testing.assert_allclose(packet["policy_auxiliary"], actual_auxiliary)
    with pytest.raises(ValueError, match="344"):
        sensor_packet(front, auxiliary, np.zeros(343))


def test_live_terminal_pose_goal_and_hud_precede_real_autoreset():
    flight = LiveFlight(lambda obs: jnp.zeros((len(obs), 4)), LiveEnvironment(), seed=42)
    initial_goal = flight.episode["goal"].copy()
    flight.advance()
    flight.advance()
    assert flight.pending_episode and flight.status == "success" and flight.completed == 1
    assert flight.steps == 2 and flight.episode_return == 4
    np.testing.assert_array_equal(flight.episode["positions"][-1], [44, 0, 1])
    np.testing.assert_array_equal(flight.episode["goal"], initial_goal)
    np.testing.assert_allclose(flight.packet["clean_auxiliary"], [1, 2, 3, 4])
    np.testing.assert_allclose(flight.packet["policy_auxiliary"], [1.1, 2.2, 3.3, 4.4])
    np.testing.assert_allclose(flight.packet["clean_front"], 0.4)
    np.testing.assert_allclose(flight.packet["policy_front"], 0.6)
    flight.advance()
    assert flight.steps == 2  # terminal display holds; it must not advance the new episode
    flight.continue_episode()
    assert not flight.pending_episode and flight.status == "flying" and flight.steps == 0
    assert flight.completed == 1
    np.testing.assert_array_equal(flight.episode["positions"][0], [999, 0, 1])
    np.testing.assert_array_equal(flight.episode["goal"], [1009, 0, 1])


def test_live_manual_reset_and_exact_restart_keep_initial_random_streams():
    flight = LiveFlight(lambda obs: jnp.zeros((len(obs), 4)), LiveEnvironment(), seed=42)
    flight.reset()
    assert flight.seed == 43 and flight.manual_resets == 1
    initial = flight.state
    flight.advance()
    flight.restart()
    assert flight.state is initial
    assert flight.restarts == 1 and flight.steps == 0
    np.testing.assert_array_equal(flight.episode["positions"][0], [43, 0, 1])
    np.testing.assert_array_equal(flight.state.obs, initial.obs)


def test_checkpoint_gui_starts_live_without_a_frozen_rollout(monkeypatch: pytest.MonkeyPatch):
    from crazyflow.tof import evaluate, play

    calls = []
    actor = lambda obs: jnp.zeros((len(obs), 4))  # noqa: E731
    monkeypatch.setattr(
        evaluate, "load_actor_environment", lambda args: (actor, LiveEnvironment(), {})
    )
    monkeypatch.setattr(evaluate, "rollout", lambda *a, **k: pytest.fail("GUI must be live"))
    monkeypatch.setattr(
        play, "interactive_live", lambda flight, **kwargs: calls.append((flight, kwargs))
    )
    monkeypatch.setattr(
        "sys.argv", ["play", "--checkpoint", "unused.npz", "--device", "cpu", "--gui-frames", "3"]
    )
    play.main()
    assert len(calls) == 1 and isinstance(calls[0][0], LiveFlight)
    assert calls[0][1]["max_frames"] == 3
