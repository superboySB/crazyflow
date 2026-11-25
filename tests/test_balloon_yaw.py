"""Verify yaw remains learnable from output through physics and camera feedback."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.balloon.config import Config
from crazyflow.balloon.model import (
    advance,
    body_rates,
    command,
    initialize,
    nominal_params,
    sample_batch,
    state_vector,
)
from crazyflow.balloon.observations import ideal_bbox
from crazyflow.balloon.policy import init_policy
from crazyflow.balloon.rollout import rollout
from crazyflow.sim.data import SimData


def test_yaw_command_is_preserved_with_physical_angle_units():
    cfg = Config()
    physical = nominal_params(cfg)
    actions = jnp.array([[0.1, -0.2, 0.5, 0.0], [-0.1, 0.2, -0.75, 0.0]])
    physical_commands = command(actions, cfg, physical)
    np.testing.assert_allclose(physical_commands[:, 2], [0.5 * cfg.max_yaw, -0.75 * cfg.max_yaw])
    assert np.all(np.abs(physical_commands[:, 2]) > 0.1)


@pytest.mark.parametrize("yaw_sign", [-1.0, 1.0])
def test_yaw_turn_changes_inertial_and_bbox_feedback_with_correct_sign(yaw_sign: float):
    cfg = Config(steps=10)
    physical = nominal_params(cfg)
    batch = sample_batch(jax.random.key(29), cfg, 1, "nominal")
    state = jnp.zeros((1, 12)).at[:, 2].set(1.4)
    target = jnp.array([[2.0, 0.0, 1.4]])
    action = jnp.array([[0.0, 0.0, yaw_sign * 0.5, 0.0]])
    physical_command = command(action, cfg, physical)

    batch["state"] = state
    data = initialize(batch, cfg, physical)

    def step(s: SimData, _: None) -> tuple[SimData, None]:
        nxt, _ = advance(s, physical_command, cfg)
        return nxt, None

    data, _ = jax.lax.scan(step, data, None, length=10)
    turned = state_vector(data)
    box, visible = ideal_bbox(turned[:, :3], turned[:, 3:6], target, 0.15, cfg)
    assert bool(visible[0])
    assert float(turned[0, 5]) * yaw_sign > 0.05
    assert float(body_rates(turned)[0, 2]) * yaw_sign > 0.05
    # Body +Z yaw turns left. A stationary target ahead moves right in the
    # image after a positive turn, and left after a negative turn.
    assert float(box[0, 0]) * yaw_sign > 0.05
    np.testing.assert_allclose(turned[0, :2], 0.0, atol=1e-6)


def test_off_axis_target_full_bptt_yaw_gradient_matches_finite_difference():
    cfg = Config(steps=32, hidden=8)
    key = jax.random.key(15)
    batch = sample_batch(key, cfg, 1, "nominal")
    batch["state"] = jnp.zeros((1, 12)).at[:, 2].set(1.4)
    batch["target"] = jnp.array([[2.0, 0.6, 1.4]])
    params = init_policy(key, cfg)
    # Start with forward acceleration so changing yaw changes the flight
    # direction relative to this off-axis balloon within the short rollout.
    params[-1]["b"] = jnp.array([0.0, 0.25, 0.0, 0.0])

    @jax.jit
    def objective(yaw_bias: jax.Array) -> jax.Array:
        output_layer = {**params[-1], "b": params[-1]["b"].at[2].set(yaw_bias)}
        policy = [*params[:-1], output_layer]
        return rollout(policy, batch, cfg, key, noise=False)["loss"]

    analytic = jax.grad(objective)(jnp.array(0.0))
    epsilon = 0.003
    numerical = (objective(jnp.array(epsilon)) - objective(jnp.array(-epsilon))) / (2 * epsilon)
    assert np.isfinite(analytic) and np.isfinite(numerical)
    assert abs(float(analytic)) > 1e-3
    # Positive yaw steers the ongoing forward acceleration toward a left-side
    # balloon; the local loss gradient must favor that turn.
    assert float(analytic) < 0.0
    np.testing.assert_allclose(analytic, numerical, atol=0.001, rtol=0.03)
