"""Export an existing native actor NPZ for Torch/RSL inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from crazyflow.tof.policy import Moments, export_torch_policy


def main() -> None:
    """Convert explicit paths without changing the source NPZ or its policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Native policy.npz")
    parser.add_argument("--output", required=True, type=Path, help="Matching Torch policy.pt")
    args = parser.parse_args()
    if args.checkpoint.resolve() == args.output.resolve():
        parser.error("Source and output paths must differ")
    if args.checkpoint.suffix != ".npz" or args.output.suffix != ".pt":
        parser.error("Use an NPZ source and a PT output")
    with np.load(args.checkpoint, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if (
            metadata.get("format_version") != 1
            or metadata.get("activation") != "elu"
            or metadata.get("normalization") != "sqrt_var_plus_epsilon"
            or metadata.get("normalization_epsilon") != 0.01
        ):
            raise ValueError("Only the current native actor/normalization format is supported")
        actor = {
            "params": {
                f"layer_{index}": {
                    name: jnp.asarray(data[f"actor_{index}_{name}"]) for name in ("kernel", "bias")
                }
                for index in range(len(metadata["hidden_dims"]) + 1)
            }
        }
        moments = Moments(
            jnp.asarray(data["obs_mean"]),
            jnp.asarray(data["obs_var"]),
            jnp.asarray(data["obs_count"]),
        )
        meta = export_torch_policy(
            args.output,
            metadata,
            actor,
            moments,
            jnp.asarray(data["logstd"]),
            source_npz_path=args.checkpoint,
        )
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "output": str(args.output),
                "actor_only": meta["actor_only"],
                "source_npz_sha256": meta["source_npz_sha256"],
                "max_abs_error": meta["export_max_abs_error"],
            }
        )
    )


if __name__ == "__main__":
    main()
