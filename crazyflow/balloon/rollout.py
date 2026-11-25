"""Full BPTT graph: measurements -> history policy -> official dynamics -> loss."""

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from crazyflow.balloon.config import Config
from crazyflow.balloon.model import (
    Batch,
    advance,
    command,
    initialize,
    nominal_params,
    rotation_matrix,
    state_vector,
)
from crazyflow.balloon.observations import actor_features, clip_bbox, ideal_bbox, soft_fov_cost
from crazyflow.balloon.policy import PolicyParams, apply_policy
from crazyflow.sim.data import SimData


def trajectory_loss(
    states: jax.Array, substep_states: jax.Array, actions: jax.Array, batch: Batch, cfg: Config
) -> jax.Array:
    """Smooth training objective; hard first-contact outcomes belong to evaluation.

    Privileged geometry/velocity here are supervision through physics, and never
    policy inputs. Soft best approach supplies gradient even before any hits.
    """
    pos, rpy, vel = states[1:, :, :3], states[1:, :, 3:6], states[1:, :, 6:9]
    center = pos + jnp.einsum("tnij,j->tni", rotation_matrix(rpy), jnp.array(cfg.collision_offset))
    delta = batch["target"][None] - center
    distance = jnp.sqrt(jnp.sum(delta**2, -1) + 1e-6)
    surface = distance - (batch["radius"][None] + cfg.drone_radius)
    # Soft minimum with a normalized logsumexp is smooth and independent of batch size.
    best = -cfg.softmin_temperature * jax.scipy.special.logsumexp(
        -surface / cfg.softmin_temperature, axis=0
    )
    best = best + cfg.softmin_temperature * jnp.log(float(cfg.steps))
    weights = jax.nn.softmax(-surface / cfg.impact_temperature, axis=0)
    speed2 = jnp.sum(vel**2, -1)
    impact = jnp.sum(weights * jax.nn.relu(speed2 - cfg.max_impact_speed**2) ** 2, axis=0)
    fov = soft_fov_cost(pos, rpy, batch["target"][None], cfg)
    # Ground/large-tilt barriers use every substep; official floor handling is retained.
    ground = jnp.mean(
        jax.nn.relu(cfg.min_altitude + cfg.ground_margin - substep_states[..., 2]) ** 2
    )
    tilt = jnp.mean(jax.nn.relu(jnp.abs(substep_states[..., 3:5]) - cfg.tilt_soft_limit) ** 2)
    previous = jnp.concatenate((jnp.zeros_like(actions[:1]), actions[:-1]), axis=0)
    return (
        cfg.loss_weights["distance"] * jnp.mean(distance)
        + cfg.loss_weights["contact"]
        * jnp.mean(jax.nn.softplus(best / cfg.contact_temperature) * cfg.contact_temperature)
        + cfg.loss_weights["speed"] * jnp.mean(speed2)
        + cfg.loss_weights["impact"] * jnp.mean(impact)
        + cfg.loss_weights["fov"] * jnp.mean(fov)
        + cfg.loss_weights["ground"] * ground
        + cfg.loss_weights["tilt"] * tilt
        + cfg.loss_weights["angular_rate"] * jnp.mean(states[1:, :, 9:12] ** 2)
        + cfg.loss_weights["action"] * jnp.mean(actions**2)
        + cfg.loss_weights["smoothness"] * jnp.mean((actions - previous) ** 2)
    )


def observe(
    data: SimData,
    batch: Batch,
    cfg: Config,
    rng: jax.Array,
    boxes: jax.Array,
    acts: jax.Array,
    camera_fifo: jax.Array,
    delay: jax.Array,
    noise: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Share the exact camera, sensor noise and mailbox contract across flight plants.

    Geometry is used solely to synthesize measurements. The returned actor input
    contains only received boxes, measured attitude/gyro and issued-action history.
    Both plants call this at the beginning of the same 50 Hz control tick.
    """
    n = batch["state"].shape[0]
    state = state_vector(data)
    k = jax.random.split(rng, 4)
    bbox, _ = ideal_bbox(state[:, :3], state[:, 3:6], batch["target"], batch["radius"], cfg)
    if noise:
        jitter = cfg.bbox_noise_std * jax.random.normal(k[0], (n, 4))
        noisy_bounds = bbox[:, :4] + jitter
        valid = bbox[:, 4] * (jax.random.uniform(k[1], (n,)) >= cfg.dropout_prob)
        bbox = clip_bbox(jnp.concatenate((noisy_bounds, valid[:, None]), -1))
    camera_fifo = jnp.concatenate((camera_fifo[1:], bbox[None]), 0)
    received = camera_fifo[cfg.max_delay_steps - delay, jnp.arange(n)]
    boxes = jnp.concatenate((boxes[:, 1:], received[:, None]), 1)
    rot = Rotation.from_quat(data.states.quat[:, 0]).as_matrix()
    gyro = data.states.ang_vel[:, 0]
    if noise:
        rot = (
            rot
            @ Rotation.from_rotvec(
                cfg.attitude_noise_std * jax.random.normal(k[2], (n, 3))
            ).as_matrix()
        )
        gyro = (
            gyro
            + batch["domain"]["gyro_bias"]
            + cfg.gyro_noise_std * jax.random.normal(k[3], (n, 3))
        )
    obs = actor_features(boxes, rot, gyro, acts, cfg)
    return obs, boxes, camera_fifo, received


def rollout(
    params: PolicyParams, batch: Batch, cfg: Config, key: jax.Array, noise: bool = True
) -> dict[str, jax.Array]:
    """One uninterrupted rollout; random samples remain fixed during differentiation."""
    physical = nominal_params(cfg)
    n = batch["state"].shape[0]
    keys = jax.random.split(key, 3)
    delay = (
        jax.random.randint(keys[0], (n,), 0, cfg.max_delay_steps + 1)
        if noise
        else jnp.zeros(n, int)
    )
    action_delay = (
        jax.random.randint(keys[1], (n,), 0, cfg.action_delay_steps + 1)
        if noise
        else jnp.zeros(n, int)
    )
    boxes = jnp.zeros((n, cfg.history, 5))
    acts = jnp.zeros((n, cfg.history, 4))
    camera_fifo = jnp.zeros((cfg.max_delay_steps + 1, n, 5))
    action_fifo = jnp.zeros((cfg.action_delay_steps + 1, n, 4))

    def step(carry: tuple, rng: jax.Array) -> tuple[tuple, tuple[jax.Array, ...]]:
        data, boxes, acts, camera_fifo, action_fifo = carry
        obs, boxes, camera_fifo, received = observe(
            data, batch, cfg, rng, boxes, acts, camera_fifo, delay, noise
        )
        issued = apply_policy(params, obs)
        acts = jnp.concatenate((acts[:, 1:], issued[:, None]), 1)
        action_fifo = jnp.concatenate((action_fifo[1:], issued[None]), 0)
        executed = action_fifo[cfg.action_delay_steps - action_delay, jnp.arange(n)]
        data, sub = advance(data, command(executed, cfg, physical), cfg)
        state = state_vector(data)
        return (data, boxes, acts, camera_fifo, action_fifo), (
            state,
            sub,
            received,
            issued,
            data.states.quat[:, 0],
            data.states.ang_vel[:, 0],
            data.states.rotor_vel[:, 0],
        )

    initial_data = initialize(batch, cfg, physical)
    _, (states, sub, bbox, actions, quaternions, gyro, rotor_vel) = jax.lax.scan(
        jax.checkpoint(step),
        (initial_data, boxes, acts, camera_fifo, action_fifo),
        jax.random.split(keys[2], cfg.steps),
    )
    states = jnp.concatenate((batch["state"][None], states), 0)
    sub = jnp.concatenate((batch["state"][None], sub.reshape(-1, n, 12)), 0)
    loss = trajectory_loss(states, sub, actions, batch, cfg)
    return {
        "states": states,
        "substep_states": sub,
        "bbox": bbox,
        "actions": actions,
        "loss": loss,
        "quaternion_xyzw": jnp.concatenate((initial_data.states.quat[:, 0][None], quaternions)),
        "body_gyro": jnp.concatenate((initial_data.states.ang_vel[:, 0][None], gyro)),
        "rotor_rpm": jnp.concatenate((initial_data.states.rotor_vel[:, 0][None], rotor_vel)),
    }
