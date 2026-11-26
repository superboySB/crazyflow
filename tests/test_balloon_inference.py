"""The compiled deployment tick keeps the training actor's sensor contract."""

from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from crazyflow.balloon.config import Config
from crazyflow.balloon.model import command, nominal_params
from crazyflow.balloon.observations import actor_features
from crazyflow.balloon.policy import Controller, apply_policy, init_policy, save_checkpoint


@pytest.mark.parametrize("history", (3, 5))
def test_compiled_sensor_tick_matches_training_actor_and_reset(tmp_path: Path, history: int):
    """Compare multiple sensor packets, history and reset against the training API."""
    cfg = Config(
        history=history,
        hidden=8,
        camera_intrinsics=(390.0, 410.0, 300.0, 235.0),
        camera_rpy=(0.03, -0.04, 0.02),
    )
    path = tmp_path / "policy.npz"
    save_checkpoint(path, init_policy(jax.random.key(70), cfg), cfg)
    controller = Controller(path)
    cfg = controller.cfg
    physical = nominal_params(cfg)
    boxes = np.zeros((history, 5), dtype=np.float32)
    actions = np.zeros((history, 4), dtype=np.float32)
    actor = jax.jit(
        lambda b, r, g, a: apply_policy(controller.params, actor_features(b, r, g, a, cfg))
    )
    to_command = jax.jit(lambda action: command(action, cfg, physical))
    packets = (
        [250, 170, 370, 290],
        [620, -10, 710, 110],
        None,
        [-90, 40, -10, 120],
        [270, 195, 355, 275],
        [280, 205, 345, 265],
    )
    for tick, pixels in enumerate(packets * 2):
        if tick == len(packets):
            controller.reset()
            boxes[:] = 0
            actions[:] = 0
        quaternion = Rotation.from_euler("xyz", [0.02 * tick, -0.03, 0.04 * tick]).as_quat()
        gyro = np.array([0.2, -0.07 * tick, 0.3], dtype=np.float32)
        observation = np.zeros(5, dtype=np.float32)
        if pixels is not None:
            corners = np.clip(np.asarray(pixels).reshape(2, 2), [0, 0], [640, 480])
            if np.all(corners[1] > corners[0]):
                observation[:2] = corners.sum(axis=0) / [640, 480] - 1
                observation[2:4] = 2 * (corners[1] - corners[0]) / [640, 480]
                observation[4] = 1
        boxes = np.concatenate((boxes[1:], observation[None]))
        expected_action = np.asarray(
            actor(boxes, Rotation.from_quat(quaternion).as_matrix(), gyro, actions)
        )
        expected_command = np.asarray(to_command(expected_action))
        actions = np.concatenate((actions[1:], expected_action[None]))
        actual = controller.step(pixels, quaternion * 3, gyro)
        np.testing.assert_allclose(actual, expected_command, rtol=3e-5, atol=2e-6)
        np.testing.assert_allclose(controller.boxes, boxes, atol=2e-7)
        np.testing.assert_allclose(controller.actions, actions, rtol=3e-5, atol=2e-6)
