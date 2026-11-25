"""Causal camera, IMU, command and gust timing shared by both flight plants.

All mutable state is an array PyTree suitable for ``jax.lax.scan``. Call
``measure`` then ``issue`` at each policy boundary. For each 500 Hz physics
step, call ``actuate``, integrate using its command/gust, then ``sample_imu``.
Configuration and the ``noise`` switch are static during an episode.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from crazyflow.balloon.observations import clip_bbox, ideal_bbox

if TYPE_CHECKING:
    from crazyflow.balloon.config import Config
    from crazyflow.balloon.model import Batch
    from crazyflow.sim.data import SimData


class HardwareState(NamedTuple):
    """Private simulation mailbox state; never pass it to the actor."""

    tick: jax.Array
    camera_seq: jax.Array
    camera_packets: jax.Array
    camera_packet_seq: jax.Array
    camera_capture_tick: jax.Array
    camera_arrival_ms: jax.Array
    camera_received: jax.Array
    camera_last_seq: jax.Array
    camera_last_tick: jax.Array
    bbox_bias: jax.Array
    gyro_history: jax.Array
    gyro_capture_tick: jax.Array
    gyro_received: jax.Array
    gyro_last_tick: jax.Array
    gyro_delay_ticks: jax.Array
    gyro_episode_bias: jax.Array
    gyro_drift: jax.Array
    attitude_history: jax.Array
    attitude_capture_tick: jax.Array
    attitude_received: jax.Array
    attitude_last_tick: jax.Array
    attitude_delay_ticks: jax.Array
    attitude_bias: jax.Array
    held_issued: jax.Array
    command_packets: jax.Array
    command_capture_tick: jax.Array
    command_arrival_tick: jax.Array
    command_last_tick: jax.Array
    command_received: jax.Array
    command_base_delay_ms: jax.Array
    command_jitter_ms: jax.Array
    command_delay_ms: jax.Array
    gust: jax.Array


def _step_ms(cfg: Config) -> float:
    return 1000.0 * cfg.dt / cfg.substeps


def _push(history: jax.Array, value: jax.Array) -> jax.Array:
    return jnp.concatenate((history[1:], value[None]), axis=0)


def _uniform_delay(key: jax.Array, shape: tuple, bounds: tuple) -> jax.Array:
    return jax.random.uniform(key, shape, minval=bounds[0], maxval=bounds[1])


def _ou(value: jax.Array, std: float, key: jax.Array, dt: float, tau: float) -> jax.Array:
    if std == 0.0:
        return jnp.zeros_like(value)
    rho = math.exp(-dt / tau)
    gain = std * math.sqrt(max(0.0, 1.0 - rho * rho))
    return rho * value + gain * jax.random.normal(key, value.shape, dtype=value.dtype)


def _imu_capture(
    data: SimData,
    cfg: Config,
    key: jax.Array,
    gyro_drift: jax.Array,
    attitude_bias: jax.Array,
    gyro_episode_bias: jax.Array,
    active: bool,
) -> tuple[jax.Array, jax.Array]:
    rotation = Rotation.from_quat(data.states.quat[:, 0]).as_matrix()
    gyro = data.states.ang_vel[:, 0]
    if active:
        scale_key, gyro_key, attitude_key = jax.random.split(key, 3)
        gyro = (
            gyro * (1.0 + cfg.hardware.gyro_scale_std * jax.random.normal(scale_key, gyro.shape))
            + gyro_episode_bias
            + gyro_drift
            + cfg.gyro_noise_std * jax.random.normal(gyro_key, gyro.shape)
        )
        error = attitude_bias + cfg.attitude_noise_std * jax.random.normal(attitude_key, gyro.shape)
        rotation = rotation @ Rotation.from_rotvec(error).as_matrix()
    return rotation, gyro


def initialize_hardware(
    data: SimData, batch: Batch, cfg: Config, key: jax.Array, noise: bool
) -> HardwareState:
    """Sample episode constants and warm-start IMU once from the true reset state.

    The camera starts without a packet. IMU history is padded with the same
    initial noisy sample, so startup does not invent zero gyro or identity
    attitude measurements. Later deliveries retain their capture timestamps.
    """
    hw = cfg.hardware
    active = hw.enabled and noise
    n = batch["state"].shape[0]
    keys = jax.random.split(key, 12)
    step_ms = _step_ms(cfg)
    camera_size = math.ceil(hw.camera_delay_ms[1] / (1000.0 * cfg.dt)) + 2
    gyro_size = math.ceil(hw.gyro_delay_ms[1] / step_ms) + 1
    attitude_size = math.ceil(hw.attitude_delay_ms[1] / step_ms) + 1
    command_size = (
        math.ceil(max(hw.attitude_command_delay_ms[1], hw.thrust_command_delay_ms[1]) / step_ms) + 1
    )
    zero3 = jnp.zeros((n, 3))
    zero4 = jnp.zeros((n, 4))
    zero_tick = jnp.zeros(n, dtype=jnp.int32)
    bbox_bias = hw.bbox_bias_std * jax.random.normal(keys[0], (n, 4)) if active else zero4
    gyro_drift = hw.gyro_drift_std * jax.random.normal(keys[1], (n, 3)) if active else zero3
    attitude_bias = hw.attitude_bias_std * jax.random.normal(keys[2], (n, 3)) if active else zero3
    gyro_delay = (
        jnp.rint(_uniform_delay(keys[3], (n,), hw.gyro_delay_ms) / step_ms).astype(jnp.int32)
        if active
        else zero_tick
    )
    attitude_delay = (
        jnp.rint(_uniform_delay(keys[4], (n,), hw.attitude_delay_ms) / step_ms).astype(jnp.int32)
        if active
        else zero_tick
    )
    low = jnp.array([hw.attitude_command_delay_ms[0]] * 3 + [hw.thrust_command_delay_ms[0]])
    high = jnp.array([hw.attitude_command_delay_ms[1]] * 3 + [hw.thrust_command_delay_ms[1]])
    base_delay = jax.random.uniform(keys[5], (n, 4), minval=low, maxval=high) if active else zero4
    jitter = hw.command_jitter_std_ms * jax.random.normal(keys[6], (n, 4)) if active else zero4
    delay = jnp.clip(base_delay + jitter, low, high) if active else zero4
    gust = hw.gust_std * jax.random.normal(keys[7], (n, 3)) if active else zero3
    episode_bias = batch["domain"]["gyro_bias"] if active else zero3
    rotation, gyro = _imu_capture(
        data, cfg, keys[8], gyro_drift, attitude_bias, episode_bias, active
    )
    return HardwareState(
        tick=jnp.array(0, dtype=jnp.int32),
        camera_seq=jnp.array(0, dtype=jnp.int32),
        camera_packets=jnp.zeros((camera_size, n, 5)),
        camera_packet_seq=jnp.full((camera_size, n), -1, dtype=jnp.int32),
        camera_capture_tick=jnp.full((camera_size, n), -1, dtype=jnp.int32),
        camera_arrival_ms=jnp.full((camera_size, n), jnp.inf),
        camera_received=jnp.zeros((n, 5)),
        camera_last_seq=jnp.full(n, -1, dtype=jnp.int32),
        camera_last_tick=jnp.full(n, -1, dtype=jnp.int32),
        bbox_bias=bbox_bias,
        gyro_history=jnp.broadcast_to(gyro, (gyro_size, n, 3)),
        gyro_capture_tick=jnp.zeros((gyro_size, n), dtype=jnp.int32),
        gyro_received=gyro,
        gyro_last_tick=zero_tick,
        gyro_delay_ticks=gyro_delay,
        gyro_episode_bias=episode_bias,
        gyro_drift=gyro_drift,
        attitude_history=jnp.broadcast_to(rotation, (attitude_size, n, 3, 3)),
        attitude_capture_tick=jnp.zeros((attitude_size, n), dtype=jnp.int32),
        attitude_received=rotation,
        attitude_last_tick=zero_tick,
        attitude_delay_ticks=attitude_delay,
        attitude_bias=attitude_bias,
        held_issued=zero4,
        command_packets=jnp.zeros((command_size, n, 4)),
        command_capture_tick=jnp.full((command_size, n, 4), -1, dtype=jnp.int32),
        command_arrival_tick=jnp.full((command_size, n, 4), jnp.iinfo(jnp.int32).max),
        command_last_tick=jnp.full((n, 4), -1, dtype=jnp.int32),
        command_received=zero4,
        command_base_delay_ms=base_delay,
        command_jitter_ms=jitter,
        command_delay_ms=delay,
        gust=gust,
    )


def measure(
    state: HardwareState, data: SimData, batch: Batch, cfg: Config, key: jax.Array, noise: bool
) -> tuple[HardwareState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Capture one 50 Hz camera packet and poll the latest causal sensor samples.

    Detector failure is an invalid delivered observation; transport loss is
    an absent packet. Neither holding nor polling adds fresh measurement noise.
    Out-of-order packets can never replace a newer capture already received.
    """
    hw = cfg.hardware
    active = hw.enabled and noise
    n = batch["state"].shape[0]
    jitter_key, detector_key, transport_key, delay_key = jax.random.split(key, 4)
    rotation = Rotation.from_quat(data.states.quat[:, 0])
    bbox, _ = ideal_bbox(
        data.states.pos[:, 0], rotation.as_euler("xyz"), batch["target"], batch["radius"], cfg
    )
    if active:
        values = (
            bbox[:, :4]
            + state.bbox_bias
            + cfg.bbox_noise_std * jax.random.normal(jitter_key, (n, 4))
        )
        valid = bbox[:, 4] * (jax.random.uniform(detector_key, (n,)) >= cfg.dropout_prob)
        bbox = clip_bbox(jnp.concatenate((values, valid[:, None]), axis=-1))
        keep = jax.random.uniform(transport_key, (n,)) >= hw.camera_packet_loss_prob
        delay = _uniform_delay(delay_key, (n,), hw.camera_delay_ms)
    else:
        keep, delay = jnp.ones(n, bool), jnp.zeros(n)
    now_ms = state.tick * _step_ms(cfg)
    packets = _push(state.camera_packets, bbox)
    sequence = _push(state.camera_packet_seq, jnp.where(keep, state.camera_seq, -1))
    captured = _push(state.camera_capture_tick, jnp.full(n, state.tick, dtype=jnp.int32))
    arrival = _push(state.camera_arrival_ms, now_ms + delay)
    eligible = (arrival <= now_ms) & (sequence > state.camera_last_seq[None])
    index = jnp.argmax(jnp.where(eligible, sequence, -1), axis=0)
    world = jnp.arange(n)
    found = jnp.any(eligible, axis=0)
    received = jnp.where(found[:, None], packets[index, world], state.camera_received)
    last_seq = jnp.where(found, sequence[index, world], state.camera_last_seq)
    last_tick = jnp.where(found, captured[index, world], state.camera_last_tick)
    age = jnp.where(last_seq >= 0, (state.tick - last_tick) * _step_ms(cfg), jnp.inf)
    delivered = jnp.where((age <= hw.camera_stale_ms)[:, None], received, 0.0)
    state = state._replace(
        camera_seq=state.camera_seq + 1,
        camera_packets=packets,
        camera_packet_seq=sequence,
        camera_capture_tick=captured,
        camera_arrival_ms=arrival,
        camera_received=received,
        camera_last_seq=last_seq,
        camera_last_tick=last_tick,
    )
    if not active:
        state = state._replace(
            gyro_received=data.states.ang_vel[:, 0],
            attitude_received=rotation.as_matrix(),
            gyro_last_tick=jnp.full(n, state.tick, dtype=jnp.int32),
            attitude_last_tick=jnp.full(n, state.tick, dtype=jnp.int32),
        )
    telemetry = {
        "camera_age_ms": age,
        "gyro_age_ms": (state.tick - state.gyro_last_tick) * _step_ms(cfg),
        "attitude_age_ms": (state.tick - state.attitude_last_tick) * _step_ms(cfg),
        "command_delay_ms": command_delay_ms(state, cfg),
    }
    return state, delivered, state.attitude_received, state.gyro_received, telemetry


def issue(
    state: HardwareState, issued: jax.Array, cfg: Config, key: jax.Array, noise: bool
) -> HardwareState:
    """Hold a new policy action and update bounded per-channel latency jitter."""
    hw = cfg.hardware
    if hw.enabled and noise:
        jitter = _ou(
            state.command_jitter_ms, hw.command_jitter_std_ms, key, cfg.dt, hw.command_jitter_tau_s
        )
        low = jnp.array([hw.attitude_command_delay_ms[0]] * 3 + [hw.thrust_command_delay_ms[0]])
        high = jnp.array([hw.attitude_command_delay_ms[1]] * 3 + [hw.thrust_command_delay_ms[1]])
        delay = jnp.clip(state.command_base_delay_ms + jitter, low, high)
    else:
        jitter = jnp.zeros_like(state.command_jitter_ms)
        delay = jnp.zeros_like(state.command_delay_ms)
    return state._replace(held_issued=issued, command_jitter_ms=jitter, command_delay_ms=delay)


def command_delay_ms(state: HardwareState, cfg: Config) -> jax.Array:
    """Return scheduled four-channel delay after 500 Hz quantization, for debug only."""
    step_ms = _step_ms(cfg)
    return jnp.rint(state.command_delay_ms / step_ms) * step_ms


def actuate(
    state: HardwareState, cfg: Config, key: jax.Array, noise: bool
) -> tuple[HardwareState, jax.Array, jax.Array]:
    """Deliver monotone command packets for this substep, then increment the clock.

    Commands retain their JAX gradient through the queue. Zero before the first
    arrival is the normalized nominal-hover command, not zero physical thrust.
    Returned gust is world-frame acceleration; the flight plant applies its own
    actual mass when converting acceleration to external force.
    """
    hw = cfg.hardware
    active = hw.enabled and noise
    delay = (
        jnp.rint(state.command_delay_ms / _step_ms(cfg)).astype(jnp.int32)
        if active
        else jnp.zeros_like(state.command_last_tick)
    )
    packets = _push(state.command_packets, state.held_issued)
    capture = _push(state.command_capture_tick, jnp.full_like(state.command_last_tick, state.tick))
    arrival = _push(state.command_arrival_tick, state.tick + delay)
    eligible = (arrival <= state.tick) & (capture > state.command_last_tick[None])
    index = jnp.argmax(jnp.where(eligible, capture, -1), axis=0)
    world = jnp.arange(state.held_issued.shape[0])[:, None]
    channel = jnp.arange(4)[None]
    found = jnp.any(eligible, axis=0)
    executed = jnp.where(found, packets[index, world, channel], state.command_received)
    last_tick = jnp.where(found, capture[index, world, channel], state.command_last_tick)
    gust = (
        _ou(state.gust, hw.gust_std, key, cfg.dt / cfg.substeps, hw.gust_tau_s)
        if active
        else jnp.zeros_like(state.gust)
    )
    state = state._replace(
        tick=state.tick + 1,
        command_packets=packets,
        command_capture_tick=capture,
        command_arrival_tick=arrival,
        command_last_tick=last_tick,
        command_received=executed,
        gust=gust,
    )
    return state, executed, gust


def sample_imu(
    state: HardwareState, data: SimData, cfg: Config, key: jax.Array, noise: bool
) -> HardwareState:
    """Capture post-integration 500 Hz IMU samples and deliver fixed-delay packets."""
    hw = cfg.hardware
    active = hw.enabled and noise
    drift_key, bias_key, capture_key, gyro_loss_key, attitude_loss_key = jax.random.split(key, 5)
    step_s = cfg.dt / cfg.substeps
    drift = (
        _ou(state.gyro_drift, hw.gyro_drift_std, drift_key, step_s, hw.bias_tau_s)
        if active
        else jnp.zeros_like(state.gyro_drift)
    )
    bias = (
        _ou(state.attitude_bias, hw.attitude_bias_std, bias_key, step_s, hw.bias_tau_s)
        if active
        else jnp.zeros_like(state.attitude_bias)
    )
    rotation, gyro = _imu_capture(
        data, cfg, capture_key, drift, bias, state.gyro_episode_bias, active
    )
    n = gyro.shape[0]
    capture_tick = jnp.full(n, state.tick, dtype=jnp.int32)
    gyro_history = _push(state.gyro_history, gyro)
    gyro_captured = _push(state.gyro_capture_tick, capture_tick)
    attitude_history = _push(state.attitude_history, rotation)
    attitude_captured = _push(state.attitude_capture_tick, capture_tick)
    gyro_delay = state.gyro_delay_ticks if active else jnp.zeros(n, dtype=jnp.int32)
    attitude_delay = state.attitude_delay_ticks if active else jnp.zeros(n, dtype=jnp.int32)
    world = jnp.arange(n)
    gyro_index = gyro_history.shape[0] - 1 - gyro_delay
    attitude_index = attitude_history.shape[0] - 1 - attitude_delay
    gyro_tick = gyro_captured[gyro_index, world]
    attitude_tick = attitude_captured[attitude_index, world]
    gyro_keep = gyro_tick > state.gyro_last_tick
    attitude_keep = attitude_tick > state.attitude_last_tick
    if active:
        gyro_keep &= jax.random.uniform(gyro_loss_key, (n,)) >= hw.gyro_packet_loss_prob
        attitude_keep &= jax.random.uniform(attitude_loss_key, (n,)) >= hw.attitude_packet_loss_prob
    return state._replace(
        gyro_history=gyro_history,
        gyro_capture_tick=gyro_captured,
        gyro_received=jnp.where(
            gyro_keep[:, None], gyro_history[gyro_index, world], state.gyro_received
        ),
        gyro_last_tick=jnp.where(gyro_keep, gyro_tick, state.gyro_last_tick),
        gyro_drift=drift,
        attitude_history=attitude_history,
        attitude_capture_tick=attitude_captured,
        attitude_received=jnp.where(
            attitude_keep[:, None, None],
            attitude_history[attitude_index, world],
            state.attitude_received,
        ),
        attitude_last_tick=jnp.where(attitude_keep, attitude_tick, state.attitude_last_tick),
        attitude_bias=bias,
    )
