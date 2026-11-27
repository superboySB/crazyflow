"""Flax actor/critic and portable, pickle-free inference checkpoints.

The training stack follows crazyflow_experiments' Flax/Optax PPO. The network
widths, ELU, asymmetric critic and raw Gaussian outputs follow CrazyRL baseline.
Environment tanh, sensor history and control scaling belong to the environment.
"""

from __future__ import annotations

import json
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


@jax.jit
def act(params: dict[str, Any], obs: jax.Array) -> jax.Array:
    """Raw deterministic Gaussian mean; apply environment tanh exactly once."""
    x = (obs - params["obs_mean"]) / (jnp.sqrt(params["obs_var"]) + 0.01)
    for layer in params["actor"][:-1]:
        x = jax.nn.elu(x @ layer["kernel"] + layer["bias"])
    last = params["actor"][-1]
    return x @ last["kernel"] + last["bias"]
