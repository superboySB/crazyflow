"""Causal sensor delivery and differentiable commands at the physical clock."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation

from crazyflow.balloon import hardware
from crazyflow.balloon.config import Config
from crazyflow.balloon.model import Batch, initialize, nominal_params, sample_batch
from crazyflow.balloon.noise_config import HardwareNoise
from crazyflow.balloon.observations import ideal_bbox
from crazyflow.sim.data import SimData


def _config(**hardware_values: object) -> Config:
    return Config(
        steps=3,
        hidden=8,
        bbox_noise_std=0,
        gyro_noise_std=0,
        gyro_bias_std=0,
        attitude_noise_std=0,
        dropout_prob=0,
        wind_std=0,
        hardware=HardwareNoise(enabled=True, **hardware_values),
    )


def _flight(cfg: Config, n: int = 1) -> tuple[Batch, SimData, hardware.HardwareState]:
    batch = sample_batch(jax.random.key(12), cfg, n, "nominal")
    batch["state"] = jnp.zeros((n, 12)).at[:, 2].set(1.4)
    batch["target"] = jnp.broadcast_to(jnp.array([2.0, 0.0, 1.4]), (n, 3))
    batch["radius"] = jnp.full(n, 0.15)
    data = initialize(batch, cfg)
    mailbox = hardware.initialize_hardware(data, batch, cfg, jax.random.key(13), noise=True)
    return batch, data, mailbox


def test_camera_delivers_the_captured_box_after_latency_instead_of_current_geometry():
    cfg = _config(camera_delay_ms=(20, 20))
    batch, data, mailbox = _flight(cfg)
    expected, _ = ideal_bbox(
        data.states.pos[:, 0], jnp.zeros((1, 3)), batch["target"], batch["radius"], cfg
    )
    mailbox, first, *_ = hardware.measure(mailbox, data, batch, cfg, jax.random.key(0), noise=True)
    np.testing.assert_array_equal(first, 0)
    moved = data.replace(states=data.states.replace(pos=data.states.pos.at[:, :, 0].set(0.7)))
    mailbox = mailbox._replace(tick=jnp.array(10))  # 10 * 2 ms = one camera period.
    mailbox, delivered, _, _, telemetry = hardware.measure(
        mailbox, moved, batch, cfg, jax.random.key(1), noise=True
    )
    np.testing.assert_allclose(delivered, expected, atol=1e-7)
    np.testing.assert_array_equal(telemetry["camera_age_ms"], 20)
    assert int(mailbox.camera_last_tick[0]) == 0


def test_camera_newest_capture_wins_while_late_packets_hold_and_stale_packets_expire():
    cfg = replace(_config(camera_delay_ms=(40, 40), camera_stale_ms=40), bbox_noise_std=0.04)
    batch, data, mailbox = _flight(cfg)
    older = jnp.array([[0.1, 0.0, 0.2, 0.2, 1.0]])
    newer = jnp.array([[-0.2, 0.1, 0.3, 0.3, 1.0]])
    # Captures 0 and 1 arrive at 30 and 20 ms: a known out-of-order network trace.
    mailbox = mailbox._replace(
        tick=jnp.array(10),
        camera_seq=jnp.array(2),
        camera_packets=mailbox.camera_packets.at[-2].set(older).at[-1].set(newer),
        camera_packet_seq=mailbox.camera_packet_seq.at[-2].set(0).at[-1].set(1),
        camera_capture_tick=mailbox.camera_capture_tick.at[-2].set(0).at[-1].set(10),
        camera_arrival_ms=mailbox.camera_arrival_ms.at[-2].set(30).at[-1].set(20),
    )
    mailbox, delivered, *_ = hardware.measure(
        mailbox, data, batch, cfg, jax.random.key(2), noise=True
    )
    np.testing.assert_array_equal(delivered, newer)
    # The older packet now arrives. Polling must neither regress capture order
    # nor redraw the noise on the held measurement.
    mailbox = mailbox._replace(tick=jnp.array(20))
    mailbox, held, _, _, telemetry = hardware.measure(
        mailbox, data, batch, cfg, jax.random.key(3), noise=True
    )
    np.testing.assert_array_equal(held, newer)
    assert int(mailbox.camera_last_seq[0]) == 1
    np.testing.assert_array_equal(telemetry["camera_age_ms"], 20)
    # Both pending captures are older than the 40 ms deadline when polled.
    mailbox = mailbox._replace(tick=jnp.array(50))
    _, stale, _, _, telemetry = hardware.measure(
        mailbox, data, batch, cfg, jax.random.key(4), noise=True
    )
    np.testing.assert_array_equal(stale, 0)
    assert float(telemetry["camera_age_ms"][0]) > cfg.hardware.camera_stale_ms


def test_gyro_and_attitude_have_independent_500hz_capture_delays_and_warm_start():
    cfg = _config(gyro_delay_ms=(4, 4), attitude_delay_ms=(2, 2))
    batch, data, mailbox = _flight(cfg)
    for tick in range(1, 5):
        mailbox, _, _ = hardware.actuate(mailbox, cfg, jax.random.key(tick), noise=True)
        data = data.replace(
            states=data.states.replace(
                ang_vel=jnp.array([[[float(tick), 0, 0]]]),
                quat=Rotation.from_euler("xyz", jnp.array([[[0, 0, 0.1 * tick]]])).as_quat(),
            )
        )
        mailbox = hardware.sample_imu(mailbox, data, cfg, jax.random.key(tick + 10), noise=True)
        expected_gyro = [[max(0, tick - 2), 0, 0]]
        expected_attitude = Rotation.from_euler("xyz", jnp.array([[0, 0, 0.1 * max(0, tick - 1)]]))
        np.testing.assert_allclose(mailbox.gyro_received, expected_gyro, atol=1e-7)
        np.testing.assert_allclose(
            mailbox.attitude_received, expected_attitude.as_matrix(), atol=1e-7
        )
    _, _, _, _, telemetry = hardware.measure(
        mailbox, data, batch, cfg, jax.random.key(30), noise=True
    )
    np.testing.assert_array_equal(telemetry["gyro_age_ms"], 4)
    np.testing.assert_array_equal(telemetry["attitude_age_ms"], 2)


def test_independent_attitude_and_thrust_command_delays_preserve_causal_gradients():
    cfg = _config(attitude_command_delay_ms=(4, 4), thrust_command_delay_ms=(0, 0))
    _, _, initial = _flight(cfg)

    def deliveries(action: jax.Array) -> jax.Array:
        mailbox = hardware.issue(initial, action[None], cfg, jax.random.key(0), noise=True)
        result = []
        for tick in range(3):
            mailbox, executed, _ = hardware.actuate(
                mailbox, cfg, jax.random.key(tick + 1), noise=True
            )
            result.append(executed[0])
        return jnp.stack(result)

    action = jnp.array([0.1, -0.2, 0.3, 0.4])
    actual = deliveries(action)
    np.testing.assert_allclose(actual[:2], [[0, 0, 0, 0.4], [0, 0, 0, 0.4]], atol=1e-7)
    np.testing.assert_allclose(actual[2], action, atol=1e-7)
    derivative = jax.jacrev(deliveries)(action)
    early = np.zeros((4, 4))
    early[3, 3] = 1
    np.testing.assert_array_equal(derivative[0], early)
    np.testing.assert_array_equal(derivative[1], early)
    np.testing.assert_array_equal(derivative[2], np.eye(4))


def test_noise_false_removes_hardware_delay_noise_drift_and_gust():
    cfg = replace(
        _config(
            camera_delay_ms=(20, 40),
            camera_packet_loss_prob=0.9,
            gyro_delay_ms=(4, 8),
            attitude_delay_ms=(2, 6),
            gyro_packet_loss_prob=0.9,
            attitude_packet_loss_prob=0.9,
            bbox_bias_std=0.1,
            gyro_drift_std=0.2,
            gyro_scale_std=0.1,
            attitude_bias_std=0.1,
            attitude_command_delay_ms=(10, 20),
            thrust_command_delay_ms=(4, 8),
            command_jitter_std_ms=5,
            gust_std=0.5,
        ),
        bbox_noise_std=0.3,
        gyro_noise_std=0.3,
        attitude_noise_std=0.3,
        dropout_prob=0.9,
    )
    batch, data, _ = _flight(cfg)
    data = data.replace(
        states=data.states.replace(
            quat=Rotation.from_euler("xyz", jnp.array([[[0.1, -0.1, 0.2]]])).as_quat(),
            ang_vel=jnp.array([[[0.3, -0.4, 0.2]]]),
        )
    )
    mailbox = hardware.initialize_hardware(data, batch, cfg, jax.random.key(9), noise=False)
    mailbox, received, attitude, gyro, telemetry = hardware.measure(
        mailbox, data, batch, cfg, jax.random.key(10), noise=False
    )
    expected, _ = ideal_bbox(
        data.states.pos[:, 0],
        Rotation.from_quat(data.states.quat[:, 0]).as_euler("xyz"),
        batch["target"],
        batch["radius"],
        cfg,
    )
    np.testing.assert_allclose(received, expected, atol=1e-7)
    np.testing.assert_allclose(attitude, Rotation.from_quat(data.states.quat[:, 0]).as_matrix())
    np.testing.assert_array_equal(gyro, data.states.ang_vel[:, 0])
    np.testing.assert_array_equal(telemetry["command_delay_ms"], 0)
    action = jnp.array([[0.1, -0.2, 0.3, 0.4]])
    mailbox = hardware.issue(mailbox, action, cfg, jax.random.key(11), noise=False)
    mailbox, executed, gust = hardware.actuate(mailbox, cfg, jax.random.key(12), noise=False)
    mailbox = hardware.sample_imu(mailbox, data, cfg, jax.random.key(13), noise=False)
    np.testing.assert_array_equal(executed, action)
    for value in (gust, mailbox.gyro_drift, mailbox.attitude_bias, mailbox.command_jitter_ms):
        np.testing.assert_array_equal(value, 0)


def test_physical_randomization_preserves_reset_rng_and_valid_inertia_drag_and_motor_axes():
    legacy = _config()
    legacy = replace(legacy, hardware=HardwareNoise())
    cfg = replace(
        legacy,
        model_params={"J": [[25e-6, 2e-6, 0], [2e-6, 28e-6, 1e-6], [0, 1e-6, 49e-6]]},
        hardware=HardwareNoise(
            enabled=True,
            inertia_fraction=0.2,
            drag_fraction=0.2,
            motor_thrust_fraction=0.2,
            motor_response_fraction=0.2,
        ),
    )
    key = jax.random.key(72)
    before, after = sample_batch(key, legacy, 4), sample_batch(key, cfg, 4)
    for name in ("state", "target", "radius"):
        np.testing.assert_array_equal(before[name], after[name])
    for name in before["domain"]:
        np.testing.assert_array_equal(before["domain"][name], after["domain"][name])
    data = initialize(after, cfg)
    inertia, inverse = np.asarray(data.params.J), np.asarray(data.params.J_inv)
    assert inertia.shape == (4, 1, 3, 3)
    assert np.linalg.eigvalsh(inertia).min() > 0
    np.testing.assert_allclose(
        inertia @ inverse, np.broadcast_to(np.eye(3), (4, 1, 3, 3)), atol=2e-7
    )
    base = nominal_params(cfg)
    np.testing.assert_allclose(
        inertia[:, 0], base["J"][None] * after["domain"]["inertia_scale"][:, None, None]
    )
    assert np.linalg.eigvalsh(np.asarray(data.params.drag_matrix)).max() <= 0
    assert data.params.rpm2thrust.shape == data.params.rpm2torque.shape == (4, 1, 4, 3)
    thrust_ratio = data.params.rpm2thrust[:, 0, :, 2] / base["rpm2thrust"][2]
    torque_ratio = data.params.rpm2torque[:, 0, :, 2] / base["rpm2torque"][2]
    np.testing.assert_allclose(thrust_ratio, torque_ratio, rtol=2e-7)
    assert np.any(np.ptp(np.asarray(thrust_ratio), axis=1) > 0.01)
    response = after["domain"]["motor_response_scale"]
    assert response.shape == (4, 4, 4)
    np.testing.assert_array_equal(response[..., 0], response[..., 1])
    np.testing.assert_array_equal(response[..., 2], response[..., 3])
    assert not np.array_equal(response[..., 0], response[..., 2])
    for name in ("inertia_scale", "drag_scale", "motor_thrust_scale", "motor_response_scale"):
        assert np.asarray(after["domain"][name]).min() > 0
        nominal = sample_batch(key, cfg, 4, "nominal")
        np.testing.assert_array_equal(nominal["domain"][name], 1)
