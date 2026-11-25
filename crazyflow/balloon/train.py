"""Train the deterministic bbox policy with full backpropagation through time."""

import argparse
import json
import time
import tomllib
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import optax

from crazyflow.balloon.config import Config
from crazyflow.balloon.logs import write_training_plot
from crazyflow.balloon.model import Batch, sample_batch
from crazyflow.balloon.policy import PolicyParams, init_policy, save_checkpoint
from crazyflow.balloon.rollout import rollout
from crazyflow.balloon.settings import load_settings, resolve_batch_size


def main() -> None:
    """Run reset-to-end BPTT, with independently sampled worlds each update."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, help="Training/task/noise/objective YAML")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int, help="Override rollout length from Config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--output", type=Path, default=Path("outputs/balloon"))
    parser.add_argument("--config", type=Path, help="Camera intrinsics/extrinsics JSON")
    parser.add_argument("--controller-params", type=Path, help="Explicit measured controller TOML")
    parser.add_argument(
        "--nominal", action="store_true", help="Disable domain/sensor randomization"
    )
    parser.add_argument(
        "--model-params", type=Path, help="first_principles TOML from crazyflow.identify_physics"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="4 tiny updates; only a pipeline check"
    )
    args = parser.parse_args()
    if args.device != "auto":
        jax.config.update("jax_platforms", "cuda" if args.device == "gpu" else "cpu")
    settings, overrides = load_settings(args.settings)
    actual_device = "gpu" if jax.default_backend() == "gpu" else "cpu"
    args.iterations = settings["iterations"] if args.iterations is None else args.iterations
    args.batch_size = (
        resolve_batch_size(settings, actual_device) if args.batch_size is None else args.batch_size
    )
    args.seed = settings["seed"] if args.seed is None else args.seed
    args.lr = settings["lr"] if args.lr is None else args.lr
    if min(args.iterations, args.batch_size) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("iterations/batch-size/lr must be positive")
    if args.config:
        camera = json.loads(args.config.read_text())
        camera_fields = {
            "image_width",
            "image_height",
            "hfov_deg",
            "camera_intrinsics",
            "camera_offset",
            "camera_rpy",
        }
        if not isinstance(camera, dict) or set(camera) - camera_fields:
            parser.error("Camera JSON accepts camera fields only; use training YAML or model TOML")
        overrides.update(camera)
    cfg = Config(**overrides)
    if args.steps is not None:
        cfg = replace(cfg, steps=args.steps)
    if args.smoke:
        args.iterations, args.batch_size = 4, 4
        cfg = replace(cfg, steps=32)
    if args.model_params:
        with args.model_params.open("rb") as f:
            values = tomllib.load(f)
        cfg = replace(cfg, model_params=values.get("cf21B_500", values))
    if args.controller_params:
        with args.controller_params.open("rb") as f:
            values = tomllib.load(f)
        cfg = replace(cfg, controller_params=values)
    scenario = "nominal" if args.nominal else "noisy"
    key = jax.random.key(args.seed)
    key, init_key, validation_key, validation_noise = jax.random.split(key, 4)
    params = init_policy(init_key, cfg)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = optimizer.init(params)

    def loss(parameters: PolicyParams, batch: Batch, rng: jax.Array) -> jax.Array:
        return rollout(parameters, batch, cfg, rng, noise=not args.nominal)["loss"]

    @jax.jit
    def update(
        parameters: PolicyParams, state: optax.OptState, rng: jax.Array
    ) -> tuple[PolicyParams, optax.OptState, jax.Array, jax.Array]:
        reset_key, noise_key = jax.random.split(rng)
        batch = sample_batch(reset_key, cfg, args.batch_size, scenario)
        value, grad = jax.value_and_grad(loss)(parameters, batch, noise_key)
        updates, state = optimizer.update(grad, state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return parameters, state, value, optax.global_norm(grad)

    validation = sample_batch(validation_key, cfg, max(16, args.batch_size), scenario)
    validate = jax.jit(lambda p: loss(p, validation, validation_noise))
    initial = float(validate(params))
    best = initial
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": args.seed,
        "initial_validation_loss": initial,
        "iterations": 0,
        "training_scenario": scenario,
        "smoke_only": args.smoke,
        "training_settings": {
            "iterations": args.iterations,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "lr": args.lr,
            "device": actual_device,
        },
        "parameter_origin": {
            "physical": str(args.model_params)
            if args.model_params
            else (
                f"inline model_params in {args.config}"
                if cfg.model_params is not None
                else "official cf21B_500 defaults"
            ),
            "controller": str(args.controller_params)
            if args.controller_params
            else (
                f"inline controller_params in {args.config}"
                if cfg.controller_params is not None
                else "official cf21B_500 defaults"
            ),
        },
    }
    save_checkpoint(args.output / "policy.npz", params, cfg, metadata)
    print(
        json.dumps({"iteration": 0, "validation_loss": initial, "device": str(jax.devices())}),
        flush=True,
    )
    start = time.monotonic()
    records = []
    best_iteration = 0
    with (args.output / "training.jsonl").open("w") as log:
        for i in range(1, args.iterations + 1):
            key, rng = jax.random.split(key)
            params, opt_state, value, grad_norm = update(params, opt_state, rng)
            if not np.isfinite(float(value)) or not np.isfinite(float(grad_norm)):
                raise FloatingPointError(f"Nonfinite BPTT objective/gradient at iteration {i}")
            if i == 1 or i % 25 == 0 or i == args.iterations:
                valid = float(validate(params))
                record = {
                    "iteration": i,
                    "loss": float(value),
                    "validation_loss": valid,
                    "gradient_norm": float(grad_norm),
                    "seconds": time.monotonic() - start,
                }
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + "\n")
                log.flush()
                records.append(record)
                if valid < best:
                    best = valid
                    best_iteration = i
                    save_checkpoint(
                        args.output / "policy.npz",
                        params,
                        cfg,
                        {**metadata, "iterations": i, "validation_loss": best},
                    )
    save_checkpoint(
        args.output / "last.npz",
        params,
        cfg,
        {**metadata, "iterations": args.iterations, "validation_loss": float(validate(params))},
    )
    # Keep the raw scalars portable and provide a plot without a logging service.
    write_training_plot(args.output, records, initial, best_iteration)
    device = jax.devices()[0]
    summary = {
        "training_settings": metadata["training_settings"],
        "policy_steps": cfg.steps,
        "physics_substeps": cfg.substeps,
        "device_kind": device.device_kind,
        "initial_validation_loss": initial,
        "best_validation_loss": best,
        "best_iteration": best_iteration,
        "last_validation_loss": records[-1]["validation_loss"],
        "training_seconds": records[-1]["seconds"],
        "device_memory_stats_bytes": device.memory_stats(),
    }
    (args.output / "run-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Saved {args.output / 'policy.npz'}; held-out loss {initial:.4f} -> {best:.4f}", flush=True
    )


if __name__ == "__main__":
    main()
