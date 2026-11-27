"""Load JAX-exported RSL tensors into the existing native inference runner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from pathlib import Path

HIDDEN_DIMS = (256, 256, 128, 64)


def read_crazyflow_checkpoint(path: str | Path) -> dict[str, Any] | None:
    """Read tensors safely on CPU and identify an explicitly marked export."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "crazyflow_metadata" not in checkpoint:
        return None
    return checkpoint


def _tensor(value: Any, shape: tuple[int, ...], name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != shape:
        raise ValueError(f"{name} must be a tensor with shape {shape}")
    if value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite float32 values")
    return value


def _normalizer(state: Any, width: int, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict) or set(state) != {"_mean", "_var", "_std", "count"}:
        raise ValueError(f"{name} has invalid normalization keys")
    for key in ("_mean", "_var", "_std"):
        _tensor(state[key], (1, width), f"{name}.{key}")
    if (state["_var"] < 0).any() or (state["_std"] < 0).any():
        raise ValueError(f"{name} variance and standard deviation must be nonnegative")
    if not torch.allclose(state["_std"], torch.sqrt(state["_var"]), rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name} standard deviation disagrees with variance")
    count = state["count"]
    if (
        not isinstance(count, torch.Tensor)
        or count.shape != ()
        or count.dtype != torch.int64
        or count.item() < 0
    ):
        raise ValueError(f"{name}.count must be a nonnegative scalar int64 tensor")
    return state


def _layer_shapes(prefix: str, input_dim: int, output_dim: int) -> dict[str, tuple[int, ...]]:
    widths = (input_dim, *HIDDEN_DIMS, output_dim)
    shapes: dict[str, tuple[int, ...]] = {}
    for index, (inputs, outputs) in enumerate(zip(widths[:-1], widths[1:])):
        shapes[f"{prefix}.{2 * index}.weight"] = (outputs, inputs)
        shapes[f"{prefix}.{2 * index}.bias"] = (outputs,)
    return shapes


def load_into_runner(
    runner: Any, path: str | Path, *, checkpoint: dict[str, Any] | None = None
) -> bool:
    """Load a marked export for inference; leave native baseline loading to RSL.

    Actor-only exports intentionally omit critic tensors and its normalizer.
    Validate the complete payload before copying anything into the runner. This
    reuses its existing modules, preserving network-initialization RNG draws.
    """
    if checkpoint is None:
        checkpoint = read_crazyflow_checkpoint(path)
    if checkpoint is None:
        return False
    metadata = checkpoint.get("crazyflow_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("crazyflow_metadata must be a mapping")
    hidden = metadata.get("hidden_dims")
    if (
        metadata.get("export_format") != "rsl_rl"
        or metadata.get("obs_dim") != 344
        or metadata.get("critic_obs_dim") != 434
        or metadata.get("activation") != "elu"
        or not isinstance(hidden, (list, tuple))
        or tuple(hidden) != HIDDEN_DIMS
        or metadata.get("normalization") != "sqrt_var_plus_epsilon"
        or metadata.get("normalization_epsilon") != 0.01
        or type(metadata.get("actor_only")) is not bool
    ):
        raise ValueError("Unsupported Crazyflow RSL checkpoint contract")
    actor_only = metadata["actor_only"]
    shapes = {"std": (4,), **_layer_shapes("actor", 344, 4)}
    critic_shapes = _layer_shapes("critic", 434, 1)
    if not actor_only:
        shapes.update(critic_shapes)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError("Checkpoint actor/critic keys disagree with actor_only metadata")
    for key, shape in shapes.items():
        _tensor(state[key], shape, key)
    if (state["std"] <= 0).any():
        raise ValueError("std must contain positive standard deviations, not logstd")
    actor_norm = _normalizer(checkpoint.get("obs_norm_state_dict"), 344, "actor normalizer")
    if actor_only:
        if "privileged_obs_norm_state_dict" in checkpoint:
            raise ValueError("An actor-only export must omit the critic normalizer")
        critic_norm = None
    else:
        critic_norm = _normalizer(
            checkpoint.get("privileged_obs_norm_state_dict"), 434, "critic normalizer"
        )
    iteration = checkpoint.get("iter")
    if type(iteration) is not int or iteration < 0:
        raise ValueError("iter must be a nonnegative integer")
    # Inspect the actual target modules before any state is changed.
    policy = runner.alg.policy
    target = policy.state_dict()
    expected = {"std": (4,), **_layer_shapes("actor", 344, 4), **critic_shapes}
    if set(target) != set(expected) or getattr(policy, "noise_std_type", None) != "scalar":
        raise ValueError("Native RSL runner must use the matching actor/critic and scalar std mode")
    for key, shape in expected.items():
        _tensor(target[key], shape, f"runner.{key}")
    _normalizer(runner.obs_normalizer.state_dict(), 344, "runner actor normalizer")
    if getattr(runner.obs_normalizer, "eps", None) != 0.01:
        raise ValueError("Native actor normalizer must use epsilon 0.01 outside sqrt")
    if critic_norm is not None:
        _normalizer(runner.privileged_obs_normalizer.state_dict(), 434, "runner critic normalizer")
        if getattr(runner.privileged_obs_normalizer, "eps", None) != 0.01:
            raise ValueError("Native critic normalizer must use epsilon 0.01 outside sqrt")
    # The pinned RSL override returns bool, so use Module's result to verify
    # that the only allowed missing tensors are the deliberately absent critic.
    incompatible = torch.nn.Module.load_state_dict(policy, state, strict=not actor_only)
    allowed_missing = set(critic_shapes) if actor_only else set()
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise ValueError("Unexpected missing or extra native RSL tensors")
    runner.obs_normalizer.load_state_dict(actor_norm, strict=True)
    if critic_norm is not None:
        runner.privileged_obs_normalizer.load_state_dict(critic_norm, strict=True)
    runner.current_learning_iteration = iteration
    return True
