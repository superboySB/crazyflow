"""Flax actor/critic, native NPZ checkpoints and matching Torch deployment exports.

The training stack follows crazyflow_experiments' Flax/Optax PPO. The network
widths, ELU, asymmetric critic and raw Gaussian outputs follow CrazyRL baseline.
Environment tanh, sensor history and control scaling belong to the environment.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct


class MLP(nn.Module):
    hidden_dims: tuple[int, ...] = (256, 256, 128, 64)
    output_dim: int = 4
    output_scale: float = 0.01

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs
        for index, size in enumerate((*self.hidden_dims, self.output_dim)):
            x = nn.Dense(
                size,
                kernel_init=nn.initializers.orthogonal(
                    self.output_scale if index == len(self.hidden_dims) else np.sqrt(2.0)
                ),
                name=f"layer_{index}",
            )(x)
            if index < len(self.hidden_dims):
                x = nn.elu(x)
        return x


@struct.dataclass
class Moments:
    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @classmethod
    def create(cls, dim: int) -> Moments:
        return cls(jnp.zeros(dim), jnp.ones(dim), jnp.array(0.0))

    def update(self, x: jax.Array) -> Moments:
        """Parallel population moments; retain normalizer across PPO updates."""
        batch = x.reshape((-1, x.shape[-1]))
        n = jnp.asarray(batch.shape[0], dtype=jnp.float32)
        mean, var = jnp.mean(batch, axis=0), jnp.var(batch, axis=0)
        total = self.count + n
        delta = mean - self.mean
        combined = (self.var * self.count + var * n + delta**2 * self.count * n / total) / total
        return self.replace(mean=self.mean + delta * n / total, var=combined, count=total)

    def normalize(self, x: jax.Array) -> jax.Array:
        return (x - self.mean) / (jnp.sqrt(self.var) + 0.01)


def gaussian_logprob(mean: jax.Array, logstd: jax.Array, action: jax.Array) -> jax.Array:
    """Return the joint log probability of a raw diagonal Gaussian action."""
    return -0.5 * jnp.sum(
        ((action - mean) * jnp.exp(-logstd)) ** 2 + 2 * logstd + jnp.log(2 * jnp.pi), axis=-1
    )


def save_policy(
    path: str | Path,
    metadata: dict[str, Any],
    actor_params: dict[str, Any],
    moments: Moments,
    logstd: jax.Array,
) -> None:
    """Write the exact actor plus normalization shared with Isaac's Torch adapter."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": np.asarray(
            json.dumps(
                {
                    **metadata,
                    "format_version": 1,
                    "activation": "elu",
                    "normalization": "sqrt_var_plus_epsilon",
                    "normalization_epsilon": 0.01,
                },
                ensure_ascii=False,
            )
        ),
        "obs_mean": np.asarray(moments.mean),
        "obs_var": np.asarray(moments.var),
        "obs_count": np.asarray(moments.count),
        "logstd": np.asarray(logstd),
    }
    for name, layer in actor_params["params"].items():
        index = name.removeprefix("layer_")
        for param_name in ("kernel", "bias"):
            payload[f"actor_{index}_{param_name}"] = np.asarray(layer[param_name])
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load_policy(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (metadata, params); no pickle, optimizer or training runtime required."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        if meta.get("format_version") != 1 or meta["activation"] != "elu":
            raise ValueError("Unsupported policy format")
        # Initial local diagnostic v1 exports predate the explicit integrator field.
        # They used Euler; preserve that plant instead of silently adopting RK4.
        if "env_config" in meta:
            meta["env_config"].setdefault("integrator", "euler")
        layers = tuple(
            {
                "kernel": jnp.asarray(data[f"actor_{i}_kernel"]),
                "bias": jnp.asarray(data[f"actor_{i}_bias"]),
            }
            for i in range(len(meta["hidden_dims"]) + 1)
        )
        params = {
            "actor": layers,
            "obs_mean": jnp.asarray(data["obs_mean"]),
            "obs_var": jnp.asarray(data["obs_var"]),
        }
    return meta, params


def export_torch_policy(
    path: str | Path,
    metadata: dict[str, Any],
    actor_params: dict[str, Any],
    moments: Moments,
    logstd: jax.Array,
    *,
    critic_params: dict[str, Any] | None = None,
    critic_moments: Moments | None = None,
    source_npz_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export this snapshot with RSL tensor names; Torch runs only on the CPU.

    The NPZ and PT contain the same raw Gaussian actor and observation statistics.
    A full training snapshot can also export its real asymmetric critic. Historical
    actor-only NPZ exports remain actor-only, without fabricated critic weights.
    Optax state stays in checkpoint.msgpack and is never presented as Torch Adam.
    """
    import torch

    if (critic_params is None) != (critic_moments is None):
        raise ValueError("critic_params and critic_moments must be supplied together")

    def tensor(value: Any) -> Any:
        array = np.array(jax.device_get(value), dtype=np.float32, copy=True)
        if not np.isfinite(array).all():
            raise ValueError("Cannot export non-finite policy parameters")
        return torch.from_numpy(array)

    def normalizer(stats: Moments) -> dict[str, Any]:
        mean, var = tensor(stats.mean), tensor(stats.var)
        count = float(jax.device_get(stats.count))
        if torch.any(var < 0) or not np.isfinite(count) or count < 0:
            raise ValueError("Invalid normalization statistics")
        return {
            "_mean": mean.reshape(1, -1),
            "_var": var.reshape(1, -1),
            "_std": torch.sqrt(var).reshape(1, -1),
            "count": torch.tensor(round(count), dtype=torch.int64),
        }

    model = {}
    for prefix, params in (("actor", actor_params), ("critic", critic_params)):
        if params is None:
            continue
        for index in range(len(metadata["hidden_dims"]) + 1):
            layer = params["params"][f"layer_{index}"]
            model[f"{prefix}.{2 * index}.weight"] = tensor(layer["kernel"]).T.contiguous()
            model[f"{prefix}.{2 * index}.bias"] = tensor(layer["bias"])
    model["std"] = tensor(jnp.exp(jnp.clip(logstd, -5, 2)))
    meta = {
        **metadata,
        "format_version": 1,
        "activation": "elu",
        "normalization": "sqrt_var_plus_epsilon",
        "normalization_epsilon": 0.01,
        "export_format": "rsl_rl",
        "actor_only": critic_params is None,
    }
    if source_npz_path is not None:
        meta["source_npz_sha256"] = hashlib.sha256(Path(source_npz_path).read_bytes()).hexdigest()

    # Exercise both numerical implementations before publishing each PT, without
    # consuming training/evaluation RNG streams. GPU JAX may otherwise use TF32
    # for matmuls; check the weight mapping in full precision, locally to export.
    probe = np.random.default_rng(0).normal(size=(32, int(metadata["obs_dim"]))).astype(np.float32)
    jax_input = moments.mean + probe * (jnp.sqrt(moments.var) + 0.01)
    with jax.default_matmul_precision("highest"):
        jax_output = MLP(tuple(metadata["hidden_dims"])).apply(
            actor_params, moments.normalize(jax_input)
        )
    norm = normalizer(moments)
    x = (tensor(jax_input) - norm["_mean"]) / (norm["_std"] + 0.01)
    with torch.inference_mode():
        for index in range(len(metadata["hidden_dims"]) + 1):
            x = torch.nn.functional.linear(
                x, model[f"actor.{2 * index}.weight"], model[f"actor.{2 * index}.bias"]
            )
            if index < len(metadata["hidden_dims"]):
                x = torch.nn.functional.elu(x)
    expected = np.asarray(jax.device_get(jax_output))
    np.testing.assert_allclose(x.numpy(), expected, rtol=2e-5, atol=2e-5)
    meta["export_max_abs_error"] = float(np.max(np.abs(x.numpy() - expected)))
    payload = {
        "model_state_dict": model,
        "obs_norm_state_dict": norm,
        "iter": int(metadata.get("iteration", 0)),
        "infos": {},
        # Restrict the metadata to JSON primitives so weights_only=True can load it.
        "crazyflow_metadata": json.loads(json.dumps(meta)),
    }
    if critic_moments is not None:
        payload["privileged_obs_norm_state_dict"] = normalizer(critic_moments)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(path)
    return meta


def save_policy_pair(
    directory: str | Path,
    metadata: dict[str, Any],
    actor_params: dict[str, Any],
    moments: Moments,
    logstd: jax.Array,
    *,
    critic_params: dict[str, Any],
    critic_moments: Moments,
) -> None:
    """Stage and verify both formats before replacing the previous saved pair.

    A failed conversion leaves the previous NPZ and PT intact. Final per-file
    renames share a filesystem; the PT's source hash also identifies its NPZ.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".policy-export-", dir=directory) as staging:
        staged_npz, staged_pt = Path(staging) / "policy.npz", Path(staging) / "policy.pt"
        save_policy(staged_npz, metadata, actor_params, moments, logstd)
        export_torch_policy(
            staged_pt,
            metadata,
            actor_params,
            moments,
            logstd,
            critic_params=critic_params,
            critic_moments=critic_moments,
            source_npz_path=staged_npz,
        )
        staged_npz.replace(directory / "policy.npz")
        staged_pt.replace(directory / "policy.pt")


@jax.jit
def act(params: dict[str, Any], obs: jax.Array) -> jax.Array:
    """Raw deterministic Gaussian mean; apply environment tanh exactly once."""
    x = (obs - params["obs_mean"]) / (jnp.sqrt(params["obs_var"]) + 0.01)
    for layer in params["actor"][:-1]:
        x = jax.nn.elu(x @ layer["kernel"] + layer["bias"])
    last = params["actor"][-1]
    return x @ last["kernel"] + last["bias"]
