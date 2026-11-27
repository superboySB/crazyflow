"""First-episode accounting and truthful MuJoCo replay regression tests."""

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from flax import struct
from scipy.spatial.transform import Rotation

from crazyflow.tof.config import EnvConfig
from crazyflow.tof.evaluate import build_parser, load_actor_environment, rollout, summarize_episodes
from crazyflow.tof.play import TrajectoryViewer, make_visual_model, selected_episode


def test_statistics_keep_all_failures_in_denominator():
    records = [
        {"outcome": outcome, "return": reward, "duration_s": 1.0, "final_distance_m": distance}
        for outcome, reward, distance in (
            ("success", 80.0, 0.1),
            ("collision", -80.0, 2.0),
            ("nonfinite", 0.0, None),
            ("timeout", 0.0, 4.0),
        )
    ]
    report = summarize_episodes(records)
    assert report["episodes"] == 4
    assert report["success_rate"] == 0.25
    assert report["collisions"] == report["timeouts"] == report["other_failures"] == 1
    lo, hi = report["success_rate_wilson_95"]
    assert 0 < lo < 0.25 < hi < 1
    assert report["mean_final_distance_m"] == pytest.approx(6.1 / 3)


def test_new_map_seed_preserves_frozen_checkpoint_physics(monkeypatch: pytest.MonkeyPatch):
    import crazyflow.tof.env as environment_module
    import crazyflow.tof.policy as policy_module

    original = EnvConfig(
        map_seed=42,
        integrator="rk4",
        wind_sigma=0.37,
        model_params={"mass": 0.0434, "gravity_vec": [0.0, 0.0, -9.81]},
    )
    metadata = {"env_config": original.to_dict()}
    monkeypatch.setattr(policy_module, "load_policy", lambda path: (metadata, {}))

    def make_environment(config: EnvConfig, num_envs: int, device: str) -> SimpleNamespace:
        return SimpleNamespace(config=config, num_envs=num_envs, device=device)

    monkeypatch.setattr(environment_module, "ToFEnv", make_environment)
    args = build_parser().parse_args(
        ["--checkpoint", "unused.npz", "--map-seed", "14001", "--episodes", "2", "--device", "cpu"]
    )
    _, env, returned_metadata = load_actor_environment(args)
    expected = original.to_dict()
    expected["map_seed"] = 14001
    assert env.config.to_dict() == expected
    assert returned_metadata["env_config"] == original.to_dict()
    assert env.num_envs == 2 and env.device == "cpu"


def test_saved_trajectory_replay_rejects_new_map_seed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from crazyflow.tof.play import main

    monkeypatch.setattr("sys.argv", ["play", "--trajectory", "unused.npz", "--map-seed", "14001"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--map-seed applies to a new rollout" in capsys.readouterr().err


@struct.dataclass
class FlightStates:
    """Minimal simulator state for terminal accounting."""

    pos: jax.Array
    quat: jax.Array


@struct.dataclass
class Flight:
    """Simulator state container."""

    states: FlightStates


@struct.dataclass
class EvalState:
    """Minimal environment state."""

    sim: Flight
    goal: jax.Array
    map_id: jax.Array
    obs: jax.Array
    depth: jax.Array
    ticks: jax.Array


@struct.dataclass
class Result:
    """Pre-reset transition outcome."""

    reward: jax.Array
    done: jax.Array
    terminated: jax.Array
    success: jax.Array
    collision: jax.Array
    terminal_pos: jax.Array
    terminal_quat: jax.Array
    terminal_critic_obs: jax.Array


@dataclass
class EvalConfig:
    """Short evaluation timing."""

    max_episode_steps: int = 3
    dt: float = 1 / 15


class AutoResetEnvironment:
    """Expose unrelated reset poses so scoring the reset instead of terminal fails."""

    config = EvalConfig()
    scene = SimpleNamespace(
        centers=jnp.zeros((1, 1, 3)),
        half_sizes=jnp.ones((1, 1, 3)),
        kinds=jnp.zeros((1, 1), dtype=jnp.int32),
        rotations=jnp.eye(3)[None, None],
        room_half_size=8.0,
    )
    ray_directions = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (64, 1))

    def reset(self, key: jax.Array) -> EvalState:
        """Start two distinct fixed episodes."""
        del key
        return EvalState(
            Flight(FlightStates(jnp.zeros((2, 1, 3)), jnp.array([[[0.0, 0.0, 0.0, 1.0]]] * 2))),
            jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            jnp.zeros(2, dtype=jnp.int32),
            jnp.zeros((2, 344)),
            jnp.ones((2, 64)),
            jnp.array(0),
        )

    def step(self, state: EvalState, action: jax.Array) -> tuple[EvalState, Result]:
        """End worlds at distinct times and move reset poses far away."""
        del action
        ticks = state.ticks + 1
        done = jnp.array([ticks >= 1, ticks >= 2])
        terminal = jnp.array([[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        pos = jnp.where(done[:, None], 999.0, terminal)
        next_state = state.replace(
            sim=Flight(state.sim.states.replace(pos=pos[:, None])), ticks=ticks
        )
        result = Result(
            jnp.array([3.0, 5.0]),
            done,
            done,
            jnp.array([True, False]),
            jnp.array([False, True]),
            terminal,
            state.sim.states.quat[:, 0],
            jnp.ones((2, 434)) * 30,
        )
        return next_state, result


def test_first_episode_uses_terminal_pose_and_freezes_finished_world():
    report, trajectory = rollout(lambda obs: jnp.zeros((len(obs), 4)), AutoResetEnvironment())
    assert report["episodes"] == 2 and report["successes"] == 1
    first, second = report["per_episode"]
    assert first["steps"] == 1 and first["return"] == 3.0
    assert first["final_distance_m"] == 0.0
    assert second["steps"] == 2 and second["return"] == 10.0
    assert second["final_distance_m"] == 0.5
    assert second["outcome"] == "collision"
    np.testing.assert_array_equal(trajectory["positions"][:, 0, 0], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(trajectory["lengths"], [1, 2])
    assert trajectory["front_depth_m"].shape == (3, 2, 64)
    assert selected_episode(trajectory, 0)["front_depth_m"].shape == (2, 64)
    assert selected_episode(trajectory, 0)["positions"].shape == (2, 3)
    assert selected_episode(trajectory, 1)["positions"].shape == (3, 3)


def test_nonfinite_actor_is_counted_as_failure():
    report, _ = rollout(
        lambda obs: jnp.zeros((len(obs), 4)).at[0, 0].set(jnp.nan), AutoResetEnvironment()
    )
    assert report["episodes"] == 2 and report["successes"] == 0
    assert report["other_failures"] == 1 and report["collisions"] == 1
    assert report["per_episode"][0]["outcome"] == "nonfinite"


def test_visual_scene_keeps_rotated_obstacles_and_bundled_drone_mesh():
    rotation = Rotation.from_euler("xyz", [0.2, 0.3, 0.4]).as_matrix()
    centers = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 0.7], [-1.0, -2.0, 0.9]])
    sizes = np.array([[0.2, 0.3, 0.4], [0.25, 0.25, 0.6], [0.3, 0.3, 0.5]])
    model = make_visual_model(
        centers,
        sizes,
        np.arange(3),
        np.array([4.0, 5.0, 1.0]),
        rotations=np.broadcast_to(rotation, (3, 3, 3)),
    )
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    for i in range(2):
        geom = model.geom(f"obstacle_{i}").id
        np.testing.assert_allclose(data.geom_xpos[geom], centers[i])
        np.testing.assert_allclose(data.geom_xmat[geom].reshape(3, 3), rotation, atol=1e-8)
    # MuJoCo recenters a mesh around its volume centroid on compilation. Verify
    # the visible cone apex in world space, rather than its internal geom origin.
    cone = model.geom("obstacle_2").id
    mesh = int(model.geom_dataid[cone])
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    vertices = model.mesh_vert[start : start + count] @ data.geom_xmat[cone].reshape(3, 3).T
    vertices += data.geom_xpos[cone]
    apex = centers[2] + rotation @ np.array([0.0, 0.0, sizes[2, 2]])
    assert np.min(np.linalg.norm(vertices - apex, axis=1)) < 1e-6
    assert model.geom("obstacle_0").type == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom("obstacle_1").type == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert model.geom("obstacle_2").type == mujoco.mjtGeom.mjGEOM_MESH
    assert model.nmocap == 1 and model.nq == 0
    assert model.nmesh > 10


def test_viewer_transfers_recorded_pose_without_running_physics(monkeypatch: pytest.MonkeyPatch):
    def forbid_step(*args: object, **kwargs: object) -> None:
        raise AssertionError("MuJoCo visualization must not change the Crazyflow trajectory")

    monkeypatch.setattr(mujoco, "mj_step", forbid_step)
    quat = Rotation.from_euler("xyz", [0.2, 0.3, 0.4]).as_quat()
    episode = {
        "positions": np.array([[1.0, 2.0, 1.0], [2.0, 3.0, 1.5]]),
        "quaternions_xyzw": np.array([quat, quat]),
        "dt": np.array(1 / 15),
        "centers": np.array([[5.0, 5.0, 1.0]]),
        "half_sizes": np.full((1, 3), 0.4),
        "kinds": np.array([0]),
        "goal": np.array([6.0, 6.0, 1.0]),
        "room_half_size": np.array(8.0),
    }
    viewer = TrajectoryViewer(episode, 320, 240, "follow")
    viewer.set_frame(1)
    body = viewer.model.body("drone").id
    np.testing.assert_allclose(viewer.data.xpos[body], episode["positions"][1])
    np.testing.assert_allclose(viewer.data.xquat[body], quat[[3, 0, 1, 2]])
    np.testing.assert_allclose(viewer.camera.lookat, episode["positions"][1])
    viewer.close()
