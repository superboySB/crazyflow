#!/usr/bin/env python3
"""Export the existing baseline actor for adapter parity checks; no training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    """Export baseline weights and verify the independent adapter numerically."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = state["model_state_dict"]
    normalizer = state["obs_norm_state_dict"]
    metadata = {
        "format": "crazyflow-crazyrl-actor-v1",
        "obs_dim": 344,
        "critic_obs_dim": 434,
        "hidden_dims": [256, 256, 128, 64],
        "activation": "elu",
        "normalization": "sqrt_var_plus_epsilon",
        "normalization_epsilon": 0.01,
        "source": "Frozen CrazyRL model_36825.pt; adapter validation only",
    }
    arrays = {
        "metadata": np.asarray(json.dumps(metadata)),
        "obs_mean": normalizer["_mean"].numpy().reshape(-1),
        "obs_var": normalizer["_var"].numpy().reshape(-1),
    }
    for i in range(5):
        arrays[f"actor_{i}_kernel"] = weights[f"actor.{i * 2}.weight"].numpy().T
        arrays[f"actor_{i}_bias"] = weights[f"actor.{i * 2}.bias"].numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    from npz_policy import NpzPolicy

    policy = NpzPolicy(args.output, "cpu")
    rng = torch.Generator().manual_seed(20261127)
    observations = torch.randn((128, 344), generator=rng) * 30
    expected = (observations - normalizer["_mean"]) / (normalizer["_std"] + 0.01)
    for i in range(5):
        expected = torch.nn.functional.linear(
            expected, weights[f"actor.{i * 2}.weight"], weights[f"actor.{i * 2}.bias"]
        )
        if i < 4:
            expected = torch.nn.functional.elu(expected)
    actual = policy(observations)
    error = float((actual - expected).abs().max())
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    result = {"samples": 128, "max_abs_error": error, "rtol": 2e-5, "atol": 2e-5}
    args.output.with_suffix(".parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
