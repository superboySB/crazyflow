"""Save portable training curves from scalar logs, without a logging service."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def write_training_plot(
    output: Path,
    records: list[dict],
    initial_validation_loss: float,
    best_iteration: int | None = None,
) -> None:
    """Write training.png, including the untrained policy's validation loss."""
    iterations = np.asarray([record["iteration"] for record in records], dtype=float)
    validation_iterations = np.concatenate(([0.0], iterations))
    validation = np.asarray(
        [initial_validation_loss, *[record["validation_loss"] for record in records]], dtype=float
    )
    figure = Figure(figsize=(10, 8), layout="constrained")
    FigureCanvasAgg(figure)
    loss_axis, gradient_axis, time_axis = figure.subplots(3, 1, sharex=True)
    loss_axis.plot(iterations, [record["loss"] for record in records], label="Train", linewidth=1.5)
    loss_axis.plot(
        validation_iterations,
        validation,
        label="Fixed validation",
        linewidth=1.5,
        marker=".",
        markersize=4,
    )
    if best_iteration is not None:
        loss_axis.axvline(
            best_iteration, color="0.4", linestyle=":", label=f"Best checkpoint: {best_iteration}"
        )
    loss_axis.set_ylabel("Loss")
    loss_axis.legend()
    gradient_axis.plot(
        iterations,
        [record["gradient_norm"] for record in records],
        color="tab:purple",
        marker=".",
        markersize=4,
    )
    gradient_axis.set_ylabel("Raw gradient norm")
    time_axis.plot(
        validation_iterations,
        [0.0, *[record["seconds"] for record in records]],
        color="tab:green",
        marker=".",
        markersize=4,
    )
    time_axis.set_ylabel("Elapsed seconds")
    time_axis.set_xlabel("Iteration")
    for axis in (loss_axis, gradient_axis, time_axis):
        axis.grid(alpha=0.25)
    figure.suptitle("BPTT training")
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "training.png", dpi=150)
    figure.clear()


def main() -> None:
    """Redraw a run's curves using JSONL scalars and checkpoint JSON metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Directory containing training.jsonl/policy.npz"
    )
    args = parser.parse_args()
    log_path = args.directory / "training.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    # Read only the scalar JSON string: inference and device initialization are unnecessary.
    with np.load(args.directory / "policy.npz", allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata"]))
    write_training_plot(
        args.directory,
        records,
        float(metadata["initial_validation_loss"]),
        best_iteration=metadata.get("iterations"),
    )
    print(f"Saved {args.directory / 'training.png'}")


if __name__ == "__main__":
    main()
