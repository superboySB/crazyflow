"""Deterministic history policy and portable, sensor-only deployment adapter."""

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation
from jax.typing import ArrayLike

from crazyflow.balloon.camera import intrinsics
from crazyflow.balloon.camera import parameters as camera_parameters
from crazyflow.balloon.config import Config
from crazyflow.balloon.model import command, control_params, nominal_params
from crazyflow.balloon.observations import actor_features, clip_bbox, feature_dim

SCHEMA = "cf21b_first_principles_visible_bbox_bptt_v3"
PolicyParams = list[dict[str, jax.Array]]


def init_policy(key: jax.Array, cfg: Config) -> PolicyParams:
    """Initialize a small history MLP; no learned state estimator receives truth."""
    dims = (feature_dim(cfg), cfg.hidden, cfg.hidden, 4)
    layers = []
    for i, k in enumerate(jax.random.split(key, 3)):
        gain = 0.03 if i == 2 else 1.0
        layers.append(
            {
                "w": gain * jax.random.normal(k, (dims[i], dims[i + 1])) / jnp.sqrt(float(dims[i])),
                "b": jnp.zeros(dims[i + 1]),
            }
        )
    return layers


def apply_policy(params: PolicyParams, obs: jax.Array) -> jax.Array:
    """Map only the explicit observation vector to normalized issued commands."""
    x = obs
    for layer in params:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x


def save_checkpoint(
    path: str | Path, params: PolicyParams, cfg: Config, metadata: dict | None = None
) -> None:
    """Store contract + arrays without pickle or dependencies on adjacent files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    physical = {k: np.asarray(v).tolist() for k, v in nominal_params(cfg).items()}
    effective = {k: v for k, v in physical.items() if k != "J_inv"}
    controller = jax.tree.map(lambda value: np.asarray(value).tolist(), control_params(cfg))
    k = intrinsics(cfg)
    frozen_config = replace(
        cfg,
        model_params=effective,
        controller_params=controller,
        camera_intrinsics=(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])),
    )
    meta = {
        "schema": SCHEMA,
        "algorithm": "BPTT",
        "config": frozen_config.to_dict(),
        "feature_dim": feature_dim(cfg),
        "plant": "crazyflow.first_principles.cf21B_500",
        "integrator": "official Euler;500Hz plant/attitude/motor control",
        "hardware_pipeline": (
            "causal_mailboxes_v1" if cfg.hardware.enabled else "legacy_control_tick_fifo"
        ),
        "controller_params": controller,
        "camera": camera_parameters(frozen_config),
        "physical_params": physical,
        "bbox_preprocessing": "xywh jitter then edge clipping; invalid packets zeroed",
        "bbox_projection": "visible sphere silhouette intersected with calibrated image rectangle",
        **(metadata or {}),
    }
    arrays = {
        f"{i}_{name}": np.asarray(value)
        for i, layer in enumerate(params)
        for name, value in layer.items()
    }
    np.savez_compressed(path, metadata=np.array(json.dumps(meta)), **arrays)


def load_checkpoint(path: str | Path) -> tuple[PolicyParams, Config, dict]:
    """Reject incompatible sensors/actions instead of padding old RL policies."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        if meta.get("schema") != SCHEMA or meta.get("algorithm") != "BPTT":
            raise ValueError(
                "Incompatible bbox projection contract; retrain with the current camera"
            )
        cfg = Config(**meta["config"])
        physical = meta.get("physical_params", {})
        required = set(nominal_params()) - {"J_inv"}
        effective = {k: v for k, v in physical.items() if k != "J_inv"}
        if set(effective) != required:
            raise ValueError(
                "Checkpoint must freeze the complete first_principles physical parameter set"
            )
        controller = meta.get("controller_params", {})
        reference_controller = control_params()
        if set(controller) != set(reference_controller) or any(
            set(controller[section]) != set(reference_controller[section])
            for section in reference_controller
        ):
            raise ValueError("Checkpoint must freeze the complete official controller parameters")
        cfg = replace(cfg, model_params=effective, controller_params=controller)
        if meta["feature_dim"] != feature_dim(cfg):
            raise ValueError("Observation feature contract differs")
        params = [{k: jnp.asarray(data[f"{i}_{k}"]) for k in ("w", "b")} for i in range(3)]
    expected = (feature_dim(cfg), cfg.hidden, cfg.hidden, 4)
    for i, layer in enumerate(params):
        if layer["w"].shape != (expected[i], expected[i + 1]) or layer["b"].shape != (
            expected[i + 1],
        ):
            raise ValueError("Policy weight shape differs from checkpoint contract")
        if not all(np.isfinite(np.asarray(v)).all() for v in layer.values()):
            raise ValueError("Nonfinite policy weights")
    return params, cfg, meta


class Controller:
    """50 Hz inference using a tracked pixel bbox, measured attitude and gyro only.

    step accepts [left,top,right,bottom] pixels, xyzw body-to-world attitude and
    body-FLU angular rates in rad/s. None means a failed tracker sample. The
    output is [roll_rad,pitch_rad,yaw_rad,total_thrust_N], not firmware PWM.
    """

    def __init__(self, checkpoint: str | Path) -> None:
        self.params, self.cfg, self.metadata = load_checkpoint(checkpoint)
        self.physical = nominal_params(self.cfg)
        self.reset()

        @jax.jit
        def forward(
            boxes: jax.Array,
            bbox: jax.Array,
            quaternion: jax.Array,
            gyro: jax.Array,
            actions: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            # Keep history, attitude conversion, the actor and command mapping in
            # one compiled device call rather than dispatching tiny GPU kernels.
            boxes = jnp.concatenate((boxes[1:], clip_bbox(bbox)[None]), 0)
            rotation = Rotation.from_quat(quaternion).as_matrix()
            action = apply_policy(
                self.params, actor_features(boxes, rotation, gyro, actions, self.cfg)
            )
            actions = jnp.concatenate((actions[1:], action[None]), 0)
            return boxes, actions, command(action, self.cfg, self.physical)

        self._forward = forward

    def reset(self) -> None:
        """Clear all episode history; call when starting a new flight."""
        self.boxes = jnp.zeros((self.cfg.history, 5))
        self.actions = jnp.zeros((self.cfg.history, 4))

    def step(
        self,
        bbox_xyxy: ArrayLike | None,
        quaternion_xyzw: ArrayLike,
        angular_velocity_body: ArrayLike,
    ) -> np.ndarray:
        """Advance one real control tick without position/velocity arguments."""
        q = np.asarray(quaternion_xyzw, dtype=np.float32)
        gyro = np.asarray(angular_velocity_body, dtype=np.float32)
        if (
            q.shape != (4,)
            or gyro.shape != (3,)
            or not np.isfinite(q).all()
            or not np.isfinite(gyro).all()
        ):
            raise ValueError("Need finite xyzw quaternion(4) and body gyro(3)")
        if np.linalg.norm(q) < 1e-6:
            raise ValueError("Quaternion norm must be nonzero")
        bbox = np.zeros(5, dtype=np.float32)
        if bbox_xyxy is not None:
            raw = np.asarray(bbox_xyxy, dtype=np.float32)
            if raw.shape != (4,) or not np.isfinite(raw).all():
                raise ValueError("bbox must be finite [left,top,right,bottom] pixels or None")
            w, h = self.cfg.image_width, self.cfg.image_height
            lo = np.clip(raw[:2], 0, (w, h))
            hi = np.clip(raw[2:], 0, (w, h))
            if np.all(hi > lo):
                bbox[:2] = (lo + hi) / np.array((w, h)) - 1
                bbox[2:4] = 2 * (hi - lo) / np.array((w, h))
                bbox[4] = 1
        self.boxes, self.actions, output = self._forward(
            self.boxes,
            jnp.asarray(bbox),
            jnp.asarray(q / np.linalg.norm(q)),
            jnp.asarray(gyro),
            self.actions,
        )
        return np.asarray(output)
