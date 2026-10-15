"""Summarize completed and partial Gymnasium runs without choosing models by test outcomes.

Run ``python usage_assets/summarize_fulltrain_20261015.py`` inside the research image.
The JSON distinguishes progress, completed budgets, validation-only model selection, held-out
performance, and an explicitly heuristic convergence check. It also includes historical weights
reevaluated under the same first-episode protocol. No training or evaluation is started here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("reach_pos", "reach_vel", "landing", "figure8")
ENV_IDS = (
    "DroneReachPos-v0",
    "DroneReachVel-v0",
    "DroneLanding-v0",
    "DroneFigureEightTrajectory-v0",
)
ID_TO_TASK = dict(zip(ENV_IDS, TASKS, strict=True))
LABELS = {
    "reach_pos": "Reach position",
    "reach_vel": "Reach velocity (default 3D)",
    "landing": "Landing (0.1 m hover)",
    "figure8": "Figure-eight tracking",
}
COLORS = {7: "#2276b9", 19: "#d76522", 43: "#399348"}


def read_json(path: Path) -> dict:
    """Read an available result, allowing a live run to have no JSON yet."""
    return json.loads(path.read_text()) if path.exists() else {}


def relative(path: Path) -> str:
    """Use repository-relative artifact references where possible."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def digest(path: Path) -> str:
    """Hash the exact weights or numeric artifact read by this report."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value: object) -> object:
    """Make numeric artifacts strict JSON, with unavailable values represented by null."""
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def wilson(successes: int, count: int) -> list[float] | None:
    """Compute an episode-level Wilson interval for a single fixed trained policy."""
    if count == 0:
        return None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def dispersion(values: Sequence[float]) -> dict:
    """Report spread across independently trained models separately from episode intervals."""
    array = np.asarray(values, dtype=float)
    return {
        "count": len(array),
        "mean": float(array.mean()) if len(array) else None,
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else None,
        "min": float(array.min()) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def plateau(history: dict, completed: bool) -> dict:
    """Describe recent validation changes; this heuristic never proves a global upper bound."""
    steps = np.asarray(history.get("validation_timesteps", []), dtype=float)
    success = np.asarray(history.get("validation_success", []), dtype=float)
    survival = np.asarray(history.get("validation_survival", []), dtype=float)
    rmse = np.asarray(history.get("validation_rmse", []), dtype=float)
    result = {
        "criterion": (
            "Last 3 validations: success/survival gain <= 0.005 and RMSE gain <= 0.02 relative"
        ),
        "validation_points": len(steps),
        "status": "not_assessed",
        "plateau": None,
        "recommend_extend": None,
        "limitation": (
            "A fixed validation seed and a finite window cannot establish a global optimum."
        ),
    }
    if len(steps) < 5 or any(len(array) != len(steps) for array in (success, survival, rmse)):
        result["reason"] = "Need at least 5 complete validation records."
        return result
    if not all(np.isfinite(array).all() for array in (steps, success, survival, rmse)):
        result["reason"] = "Nonfinite validation values."
        return result
    gain_success = float(success[-3:].max() - success[-3])
    gain_survival = float(survival[-3:].max() - survival[-3])
    gain_rmse = float((rmse[-3] - rmse[-3:].min()) / max(rmse[-3], 1e-12))
    result.update(
        tail_start_vector_step_slots=int(steps[-3]),
        tail_end_vector_step_slots=int(steps[-1]),
        success_absolute_gain=gain_success,
        survival_absolute_gain=gain_survival,
        rmse_relative_gain=gain_rmse,
        tail_success=success[-3:],
        tail_survival=survival[-3:],
        tail_rmse=rmse[-3:],
    )
    if not completed:
        result["reason"] = "Run still incomplete; the final validation window is not available."
        return result
    stable = gain_success <= 0.005 and gain_survival <= 0.005 and gain_rmse <= 0.02
    best_index = max(range(len(steps)), key=lambda i: (success[i], survival[i], -rmse[i]))
    regressing = (
        success[-1] < success[best_index] - 0.005
        or (
            abs(success[-1] - success[best_index]) <= 0.005
            and survival[-1] < survival[best_index] - 0.005
        )
        or (
            abs(success[-1] - success[best_index]) <= 0.005
            and abs(survival[-1] - survival[best_index]) <= 0.005
            and rmse[-1] > rmse[best_index] * 1.02
        )
    )
    if regressing:
        result.update(
            status="regressing_from_best",
            plateau=False,
            recommend_extend=False,
            recommend_recipe_review=True,
            reason="Final validation regresses; keep best weights and review the training recipe.",
            best_record_vector_step_slots=int(steps[best_index]),
        )
        return result
    result.update(
        status="stable_window" if stable else "still_improving",
        plateau=stable,
        recommend_extend=not stable,
        reason=(
            "No improvement larger than the declared thresholds in the final 3 validations."
            if stable
            else "Validation still improves; extend or compare an independent longer-budget run."
        ),
    )
    return result


def npz_data(path: Path) -> dict[str, np.ndarray]:
    """Read complete behavior artifacts without permitting pickled data."""
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def summarize_tests(source: dict, prefix: str, directory: Path) -> tuple[dict, dict]:
    """Pool observed first-episode samples, with raw per-episode files as count evidence."""
    rows, arrays = [], {}
    squared_error, sample_count, last_squared_error, last_sample_count = 0.0, 0, 0.0, 0
    successes, survivors, episodes = 0, 0, 0
    config = source.get("config", {})
    for test in source.get("tests", []):
        seed = int(test["seed"])
        path = directory / f"{prefix}_test{seed}.npz"
        behavior = npz_data(path)
        arrays[seed] = behavior
        n = len(behavior.get("per_env_success", [])) or int(
            test.get("episodes", config.get("eval_envs", 0))
        )
        count = (
            int(behavior["per_env_success"].sum())
            if "per_env_success" in behavior
            else int(test.get("successes", round(test["metrics"]["success_rate"] * n)))
        )
        survived = (
            int(behavior["per_env_survival"].sum())
            if "per_env_survival" in behavior
            else round(test["metrics"]["survival_rate"] * n)
        )
        row = dict(test) | {
            "episodes": n,
            "successes": count,
            "survivors": survived,
            "success_rate": count / n if n else None,
            "success_95pct_wilson": wilson(count, n),
            "behavior_artifact": relative(path) if behavior else None,
            "behavior_sha256": digest(path) if behavior else None,
        }
        if behavior:
            lengths = behavior["per_env_episode_steps"].astype(int)
            tail_lengths = np.minimum(lengths, config.get("freq", 50))
            squared_error += float(np.sum(behavior["per_env_rmse"].astype(float) ** 2 * lengths))
            sample_count += int(lengths.sum())
            last_squared_error += float(
                np.sum(behavior["per_env_last_rmse"].astype(float) ** 2 * tail_lengths)
            )
            last_sample_count += int(tail_lengths.sum())
            if config.get("env_id") == "DroneReachVel-v0":
                target_z = behavior["initial_target_vel"][:, 2]
                row["target_vz_strata"] = {}
                for name, lower, upper in (
                    ("below_minus_0.5", -math.inf, -0.5),
                    ("minus_0.5_to_zero", -0.5, 0.0),
                    ("zero_to_0.5", 0.0, 0.5),
                    ("at_least_0.5", 0.5, math.inf),
                ):
                    mask = (target_z >= lower) & (target_z < upper)
                    size = int(mask.sum())
                    won = int(behavior["per_env_success"][mask].sum())
                    row["target_vz_strata"][name] = {
                        "episodes": size,
                        "successes": won,
                        "success_rate": won / size if size else None,
                    }
        rows.append(row)
        episodes += n
        successes += count
        survivors += survived
    strata = {}
    for row in rows:
        for name, subgroup in row.get("target_vz_strata", {}).items():
            total = strata.setdefault(name, {"episodes": 0, "successes": 0})
            total["episodes"] += subgroup["episodes"]
            total["successes"] += subgroup["successes"]
    for total in strata.values():
        total["success_rate"] = (
            total["successes"] / total["episodes"] if total["episodes"] else None
        )
    quantiles = {}
    for name in ("per_env_rmse", "per_env_last_rmse"):
        measurements = [raw[name] for raw in arrays.values() if name in raw]
        if measurements:
            values = np.concatenate(measurements).astype(float)
            quantiles[name] = {
                label: float(np.quantile(values, q))
                for label, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("maximum", 1.0))
            }
    return {
        "test_seeds": [row["seed"] for row in rows],
        "per_episode_rmse_quantiles": quantiles or None,
        "target_vz_strata": strata or None,
        "per_test_seed": rows,
        "episodes": episodes,
        "successes": successes,
        "survivors": survivors,
        "success_rate": successes / episodes if episodes else None,
        "survival_rate": survivors / episodes if episodes else None,
        "success_95pct_wilson": wilson(successes, episodes),
        "first_episode_sample_weighted_rmse": math.sqrt(squared_error / sample_count)
        if sample_count
        else None,
        "first_episode_observed_steps": sample_count or None,
        "last_observed_second_sample_weighted_rmse": (
            math.sqrt(last_squared_error / last_sample_count) if last_sample_count else None
        ),
        "rmse_mean_across_test_seeds": (
            float(np.mean([row["metrics"]["rmse"] for row in rows])) if rows else None
        ),
    }, arrays


def paired_comparison(current: dict, baseline: dict) -> dict:
    """Count paired successes only when the held-out initial states and goals actually match."""
    gained, lost, both, neither, episodes = 0, 0, 0, 0, 0
    matched_seeds = []
    for seed, a in current.items():
        b = baseline.get(seed, {})
        if not a or not b:
            continue
        for key in ("initial_pos", "initial_vel", "initial_target_vel"):
            if key in a and key in b and not np.array_equal(a[key], b[key], equal_nan=True):
                return {
                    "status": "unavailable",
                    "reason": f"Initial conditions differ at seed {seed}: {key}",
                }
        new, old = a["per_env_success"].astype(bool), b["per_env_success"].astype(bool)
        if new.shape != old.shape:
            return {"status": "unavailable", "reason": "Episode counts differ."}
        gained += int((new & ~old).sum())
        lost += int((~new & old).sum())
        both += int((new & old).sum())
        neither += int((~new & ~old).sum())
        episodes += len(new)
        matched_seeds.append(seed)
    return {
        "status": "matched" if episodes else "unavailable",
        "test_seeds": matched_seeds,
        "episodes": episodes,
        "new_success_old_failure": gained,
        "new_failure_old_success": lost,
        "both_success": both,
        "both_failure": neither,
        "paired_success_rate_difference": (gained - lost) / episodes if episodes else None,
    }


def collect_run(tag: str, directory: Path, weights: Path) -> tuple[dict, dict, dict]:
    """Read immutable checkpoint metadata and optional completed held-out results."""
    checkpoint = weights / f"gym_{tag}.pt"
    last_checkpoint = weights / f"gym_{tag}_last.pt"
    source = read_json(directory / f"{tag}.json")
    best = (
        torch.load(checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.exists()
        else {}
    )
    final = (
        torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        if last_checkpoint.exists()
        else {}
    )
    config = final.get("config", best.get("config", source.get("config", {})))
    state, best_state = final.get("trainer_state", {}), best.get("trainer_state", {})
    history = final.get("history", source.get("history", best.get("history", {})))
    score = best_state.get("best_score")
    batch = int(config.get("num_envs", 0)) * int(config.get("num_steps", 0))
    requested = int(config.get("total_timesteps", 0))
    expected_slots = requested // batch * batch if batch else 0
    completed_slots = int(state.get("global_step", source.get("completed_vector_step_slots", 0)))
    complete_budget = bool(expected_slots and completed_slots >= expected_slots)
    actual_hash = digest(checkpoint) if best else None
    hash_matches = not source or source.get("checkpoint_sha256") == actual_hash
    full_tests = bool(source) and len(source.get("tests", [])) >= 3 and hash_matches
    validation_only = "validation only" in source.get("protocol", "").lower() or (
        requested == 50_000_000 and config.get("gamma") in (0.995, 0.999)
    )
    completed = complete_budget and full_tests
    tests, arrays = summarize_tests(source, tag, directory) if hash_matches else ({}, {})
    horizontal_summary = None
    if hash_matches and source.get("horizontal_tests"):
        horizontal_summary, _ = summarize_tests(
            source | {"tests": source["horizontal_tests"]}, tag + "_horizontal", directory
        )
        horizontal_summary["scope"] = (
            "Separate horizontal-target subset, vz=0; not default 3D goals."
        )
    selected_update = best_state.get("best_iteration", source.get("selected_update"))
    run = {
        "tag": tag,
        "task": ID_TO_TASK.get(config.get("env_id")),
        "training_seed": config.get("seed"),
        "env_backend": source.get("env_backend", "pending_worker_result"),
        "goal_rng_protocol": source.get(
            "goal_rng_protocol", final.get("reset_protocol", best.get("reset_protocol"))
        ),
        "env_backend_note": source.get("env_backend_note"),
        "official_main": source.get("official_main"),
        "crazyflow_version": source.get("crazyflow_version"),
        "gamma": config.get("gamma"),
        "teacher_model": config.get("teacher_model", "legacy"),
        "status": "tuning_complete"
        if complete_budget and validation_only
        else "complete"
        if completed
        else "training_complete_tests_pending"
        if complete_budget
        else "partial",
        "role": "validation_only_tuning" if validation_only else "trained_candidate",
        "config": config,
        "source_result": relative(directory / f"{tag}.json") if source else None,
        "checkpoint": relative(checkpoint) if best else None,
        "checkpoint_sha256": actual_hash,
        "last_checkpoint": relative(last_checkpoint) if final else None,
        "result_matches_checkpoint_hash": hash_matches if source else None,
        "requested_vector_step_slots": requested,
        "expected_complete_batch_slots": expected_slots,
        "completed_updates": state.get("iteration", source.get("completed_updates")),
        "completed_vector_step_slots": completed_slots,
        "completed_valid_transitions": state.get(
            "valid_step", source.get("completed_valid_transitions")
        ),
        "ignored_reset_slots": completed_slots - int(state.get("valid_step", completed_slots)),
        "selected_update": selected_update,
        "selected_vector_step_slots": best_state.get("global_step"),
        "selected_validation_score": score,
        "validation_seed": config.get("validation_seed"),
        "training_seconds_including_validation": source.get(
            "training_seconds_including_validation"
        ),
        "behavior_cloning_samples": (
            int(config.get("position_bc_steps", 0)) * int(config.get("num_envs", 0))
            if config.get("env_id") in ("DroneReachPos-v0", "DroneLanding-v0")
            else int(config.get("velocity_bc_steps", 0)) * int(config.get("num_envs", 0))
        ),
        "heldout": tests,
        "horizontal_heldout": source.get("horizontal_tests", []),
        "horizontal_heldout_summary": horizontal_summary,
        "validation_plateau": plateau(history, completed or (complete_budget and validation_only)),
        "numeric_integrity": {
            "validation_score_finite": bool(
                score is not None and len(score) == 3 and np.isfinite(score).all()
            ),
            "validation_records_finite": all(
                np.isfinite(history.get(name, [])).all()
                for name in ("validation_success", "validation_survival", "validation_rmse")
            ),
        },
    }
    return run, history, arrays


def plot_overview(tasks: dict, histories: dict, destination: Path) -> None:
    """Compare independent training seeds, validation histories, and frozen held-out episodes."""
    figure, axes = plt.subplots(4, 4, figsize=(21, 15), constrained_layout=True)
    for row, task in enumerate(TASKS):
        entry = tasks[task]
        unit = "m/s" if task == "reach_vel" else "m"
        for run in entry["runs"]:
            seed = run["training_seed"]
            color = COLORS.get(seed, "#7857aa")
            line = (
                ("--" if run["teacher_model"] == "calibrated" else "-")
                if run["status"] == "complete"
                else ":"
            )
            label = f"s{seed}, g={run['gamma']:g}, {run['requested_vector_step_slots'] / 1e6:g}M"
            if task in ("reach_pos", "landing"):
                label += ", " + run["teacher_model"]
            history = histories[run["tag"]]
            for column, x_name, y_name in (
                (0, "timesteps", "mean_distance"),
                (1, "validation_timesteps", "validation_rmse"),
            ):
                x, y = history.get(x_name, []), history.get(y_name, [])
                if len(x) and len(x) == len(y):
                    axes[row, column].plot(
                        np.asarray(x) / 1e6, y, line, color=color, label=label, alpha=0.85
                    )
            tests = run.get("heldout", {}).get("per_test_seed", [])
            if not tests:
                continue
            index = entry["runs"].index(run)
            offsets = np.linspace(-0.12, 0.12, len(tests))
            rmses = [test["metrics"]["rmse"] for test in tests]
            rates = [test["success_rate"] * 100 for test in tests]
            axes[row, 2].scatter(index + offsets, rmses, color=color, s=28)
            axes[row, 2].scatter(index, np.mean(rmses), color=color, marker="_", s=180)
            axes[row, 3].scatter(index + offsets, rates, color=color, s=28)
            interval = run["heldout"]["success_95pct_wilson"]
            rate = run["heldout"]["success_rate"] * 100
            if interval:
                axes[row, 3].errorbar(
                    index,
                    rate,
                    yerr=[[rate - interval[0] * 100], [interval[1] * 100 - rate]],
                    color=color,
                    fmt="_",
                    capsize=3,
                    markersize=11,
                )
        baseline = entry.get("historical_baseline")
        if baseline and baseline.get("validation"):
            axes[row, 1].axhline(
                baseline["validation"]["rmse"],
                color="#555555",
                linestyle="--",
                label="old: re-eval",
            )
        tick_labels = [
            f"s{run['training_seed']}, g{run['gamma']:g}\n"
            f"{run['requested_vector_step_slots'] / 1e6:g}M "
            f"{'cal' if run['teacher_model'] == 'calibrated' else ''}"
            for run in entry["runs"]
        ]
        if baseline and baseline["heldout"]["episodes"]:
            index = len(tick_labels)
            tick_labels.append("old\nre-eval")
            tests = baseline["heldout"]["per_test_seed"]
            offsets = np.linspace(-0.12, 0.12, len(tests))
            axes[row, 2].scatter(
                index + offsets,
                [t["metrics"]["rmse"] for t in tests],
                color="#555555",
                marker="D",
                s=25,
            )
            axes[row, 3].scatter(
                index + offsets,
                [t["success_rate"] * 100 for t in tests],
                color="#555555",
                marker="D",
                s=25,
            )
            interval = baseline["heldout"]["success_95pct_wilson"]
            rate = baseline["heldout"]["success_rate"] * 100
            if interval:
                axes[row, 3].errorbar(
                    index,
                    rate,
                    yerr=[[rate - interval[0] * 100], [interval[1] * 100 - rate]],
                    color="#555555",
                    fmt="_",
                    capsize=3,
                    markersize=11,
                )
        for column in range(4):
            axis = axes[row, column]
            axis.grid(alpha=0.22)
            if column < 2:
                axis.set_xlabel("Vector-step slots (million)")
                axis.set_ylabel(f"Error ({unit})")
                if axis.lines:
                    axis.legend(fontsize=7, loc="upper right")
            else:
                axis.set_xticks(range(len(tick_labels)), tick_labels, fontsize=7)
                axis.set_xlim(-0.5, max(0.5, len(tick_labels) - 0.5))
                axis.set_ylabel(f"RMSE ({unit})" if column == 2 else "Success (%)")
        axes[row, 3].set_ylim(-2, 102)
        axes[row, 0].set_title(f"{LABELS[task]}: training")
        axes[row, 1].set_title("Independent validation seed: RMSE")
        axes[row, 2].set_title("Held-out first-episode RMSE")
        axes[row, 3].set_title("Held-out success; 95% episode intervals")
    figure.suptitle(
        "Crazyflow: 3 training seeds per task; models selected only by validation\n"
        "Dots: 3 held-out seeds per model (1024 worlds each); dotted curves: unfinished runs",
        fontsize=16,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def main() -> None:
    """Read current artifacts and atomically save a strict partial or final research summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "usage_assets/fulltrain_20261015")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "trained_models")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input_dir
    protocol = read_json(args.input_dir / "protocol.json")
    tags = {
        path.stem.removeprefix("gym_").removesuffix("_last")
        for path in args.weights_dir.glob("gym_*.pt")
    }
    tags |= {path.stem for path in args.input_dir.glob("*.json") if path.stem.startswith(TASKS)}
    runs, histories, behaviors = [], {}, {}
    for tag in sorted(tags):
        run, history, arrays = collect_run(tag, args.input_dir, args.weights_dir)
        if run["task"] in TASKS:
            runs.append(run)
            histories[tag], behaviors[tag] = history, arrays
    expected_budget = protocol.get("requested_transitions_per_run", 200_000_000)
    expected = [
        f"{task}_s{seed}_g{gamma:g}_{expected_budget}"
        for task, seed, gamma in protocol.get(
            "jobs", [(t, s, 0.99) for t in TASKS for s in (7, 19, 43)]
        )
    ]
    calibrated_expected = [
        f"{task}_s{seed}_g0.99_{expected_budget}_calibrated"
        for task in ("reach_pos", "landing")
        for seed in (7, 19, 43)
    ]
    promoted_gamma_expected = [
        f"reach_vel_s{seed}_g0.995_{expected_budget}" for seed in (7, 19, 43)
    ]
    runs.sort(
        key=lambda run: (
            TASKS.index(run["task"]),
            run["training_seed"],
            run["requested_vector_step_slots"],
            run["gamma"],
        )
    )
    tasks = {}
    for task in TASKS:
        task_runs = [run for run in runs if run["task"] == task]
        baseline_source = read_json(args.input_dir / f"baseline_{task}.json")
        analytic_reference = read_json(args.input_dir / f"teacher_reference_{task}.json")
        if analytic_reference.get("kind") != "analytic_controller_reference":
            analytic_reference = None
        baseline = None
        baseline_candidate = None
        if baseline_source:
            heldout, baseline_arrays = summarize_tests(
                baseline_source, f"baseline_{task}", args.input_dir
            )
            baseline_checkpoint = args.weights_dir / f"baseline_20261009_{task}.pt"
            legacy = (
                torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
                if baseline_checkpoint.exists()
                else {}
            )
            legacy_steps = legacy.get("history", {}).get("timesteps", [])
            baseline = {
                "status": "complete"
                if len(baseline_source.get("tests", [])) == 3 and heldout["episodes"]
                else "partial",
                "validation": baseline_source.get("validation"),
                "training_goal_rng_protocol": "legacy_shared_key",
                "evaluation_goal_rng_protocol": baseline_source.get(
                    "goal_rng_protocol", protocol.get("goal_rng_protocol")
                ),
                "validation_worlds": 256 if baseline_source.get("validation") else None,
                "historical_completed_updates": len(legacy_steps) if legacy_steps else None,
                "historical_vector_step_slots": legacy_steps[-1] if legacy_steps else None,
                "historical_valid_transitions": None,
                "historical_valid_transitions_note": "Old trainer did not record valid counts.",
                "checkpoint": relative(baseline_checkpoint)
                if baseline_checkpoint.exists()
                else None,
                "result_matches_checkpoint_hash": (
                    digest(baseline_checkpoint) == baseline_source.get("checkpoint_sha256")
                    if baseline_checkpoint.exists()
                    else None
                ),
                "source_result": relative(args.input_dir / f"baseline_{task}.json"),
                "checkpoint_sha256": baseline_source.get("checkpoint_sha256"),
                "config": baseline_source.get("config"),
                "protocol": baseline_source.get("protocol"),
                "heldout": heldout,
            }
            validation = baseline_source.get("validation", {})
            if validation:
                score = [
                    validation["success_rate"],
                    validation["survival_rate"],
                    -validation["rmse"],
                ]
                if (
                    np.isfinite(score).all()
                    and baseline["result_matches_checkpoint_hash"] is not False
                ):
                    baseline_candidate = {
                        "tag": f"baseline_20261009_{task}",
                        "role": "historical_baseline",
                        "status": baseline["status"],
                        "checkpoint": baseline["checkpoint"],
                        "checkpoint_sha256": baseline["checkpoint_sha256"],
                        "selected_update": baseline["historical_completed_updates"],
                        "selected_validation_score": score,
                        "heldout": baseline["heldout"],
                    }
            for run in task_runs:
                run["paired_success_vs_historical"] = paired_comparison(
                    behaviors[run["tag"]], baseline_arrays
                )
        complete = [run for run in task_runs if run["status"] == "complete"]
        primary = [run for run in complete if run["tag"] in expected]
        task_expected = {
            tag
            for tag in expected + calibrated_expected + promoted_gamma_expected
            if tag.startswith(task + "_")
        }
        task_complete = task_expected <= {run["tag"] for run in complete}
        eligible = [
            run
            for run in task_runs
            if run["numeric_integrity"]["validation_score_finite"]
            and run["result_matches_checkpoint_hash"] is not False
            and run["role"] != "validation_only_tuning"
        ]
        if baseline_candidate is not None:
            eligible.append(baseline_candidate)
        recommendation = None
        if eligible:
            best_score = max(tuple(run["selected_validation_score"]) for run in eligible)
            selected = min(
                (run for run in eligible if tuple(run["selected_validation_score"]) == best_score),
                key=lambda run: run["tag"],
            )
            recommendation = {
                "tag": selected["tag"],
                "role": selected["role"],
                "checkpoint": selected["checkpoint"],
                "checkpoint_sha256": selected["checkpoint_sha256"],
                "selected_update": selected["selected_update"],
                "validation_score": selected["selected_validation_score"],
                "status": "provisional"
                if not task_complete or selected["status"] != "complete"
                else "complete_study_validation_choice",
                "selection_rule": (
                    "Lexicographic checkpoint trainer_state.best_score only; ties by tag. No "
                    "test metrics used."
                ),
                "heldout_for_reporting_only": selected["heldout"],
                "horizontal_heldout_for_reporting_only": selected.get("horizontal_heldout_summary"),
            }
        ranked = sorted(
            [run for run in task_runs if run["numeric_integrity"]["validation_score_finite"]]
            + ([baseline_candidate] if baseline_candidate is not None else []),
            key=lambda run: (
                tuple(-value for value in run["selected_validation_score"]) + (run["tag"],)
            ),
        )
        cohorts = {}
        for run in complete:
            key = (run["teacher_model"], run["gamma"], run["requested_vector_step_slots"])
            cohorts.setdefault(key, []).append(run)
        tasks[task] = {
            "completed_recipe_cohorts": [
                {
                    "teacher_model": key[0],
                    "gamma": key[1],
                    "requested_vector_step_slots": key[2],
                    "training_seeds": [run["training_seed"] for run in group],
                    "run_tags": [run["tag"] for run in group],
                    "success_rate": dispersion([run["heldout"]["success_rate"] for run in group]),
                    "rmse": dispersion(
                        [
                            run["heldout"]["first_episode_sample_weighted_rmse"]
                            for run in group
                            if run["heldout"]["first_episode_sample_weighted_rmse"] is not None
                        ]
                    ),
                }
                for key, group in sorted(cohorts.items())
            ],
            "validation_ranked_candidates": [
                {
                    key: run[key]
                    for key in (
                        "tag",
                        "role",
                        "status",
                        "selected_validation_score",
                        "selected_update",
                    )
                }
                for run in ranked
            ],
            "environment_id": dict(zip(TASKS, ENV_IDS, strict=True))[task],
            "unit": "m/s" if task == "reach_vel" else "m",
            "completed_runs": len(complete),
            "completed_preregistered_runs": len(primary),
            "available_runs": len(task_runs),
            "runs": task_runs,
            "historical_baseline": baseline,
            "analytic_controller_reference": analytic_reference,
            "recommended_demonstration": recommendation,
            "across_completed_training_runs": {
                "success_rate": dispersion([run["heldout"]["success_rate"] for run in primary]),
                "rmse": dispersion(
                    [
                        run["heldout"]["first_episode_sample_weighted_rmse"]
                        for run in primary
                        if run["heldout"]["first_episode_sample_weighted_rmse"] is not None
                    ]
                ),
                "note": (
                    "Preregistered runs only; training seeds are the replication unit. "
                    "The same cases recur "
                    "across models. No pooled IID interval across training seeds."
                ),
            },
            "runs_requiring_longer_budget": [
                run["tag"] for run in complete if run["validation_plateau"]["recommend_extend"]
            ],
        }
    complete_tags = {run["tag"] for run in runs if run["status"] == "complete"}
    missing = sorted(set(expected) - complete_tags)
    missing_calibrated = sorted(set(calibrated_expected) - complete_tags)
    missing_promoted_gamma = sorted(set(promoted_gamma_expected) - complete_tags)
    missing_baselines = [
        task
        for task, entry in tasks.items()
        if not entry["historical_baseline"] or entry["historical_baseline"]["status"] != "complete"
    ]
    summary = {
        "schema": 1,
        "summary_script_sha256": digest(Path(__file__)),
        "study_date": "2026-10-15",
        "status": "complete"
        if not missing
        and not missing_calibrated
        and not missing_promoted_gamma
        and not missing_baselines
        else "partial",
        "expected_promoted_gamma_runs": promoted_gamma_expected,
        "missing_or_incomplete_promoted_gamma_runs": missing_promoted_gamma,
        "completed_promoted_gamma_runs": len(set(promoted_gamma_expected) & complete_tags),
        "expected_runs_including_calibrated_and_promoted_gamma": (
            len(expected) + len(calibrated_expected) + len(promoted_gamma_expected)
        ),
        "promoted_gamma_recipe": (
            "Gamma .995 selected from 50M validation-only screening; three fresh 200M seeds. "
            "Fresh runs use the full-budget learning-rate schedule. "
            "Held-out tests do not select recipes."
        ),
        "expected_calibrated_runs": calibrated_expected,
        "missing_or_incomplete_calibrated_runs": missing_calibrated,
        "completed_calibrated_runs": len(set(calibrated_expected) & complete_tags),
        "expected_runs_including_calibrated": len(expected) + len(calibrated_expected),
        "missing_or_incomplete_baselines": missing_baselines,
        "protocol": protocol,
        "goal_rng_protocol": protocol.get("goal_rng_protocol"),
        "goal_rng_audit": "usage_assets/goal_rng_audit_20261015.json",
        "expected_runs": len(expected),
        "completed_runs": len(set(expected) & complete_tags),
        "missing_or_incomplete_runs": missing,
        "additional_completed_runs": sorted(complete_tags - set(expected)),
        "completed_tuning_runs": [run["tag"] for run in runs if run["status"] == "tuning_complete"],
        "completed_vector_step_slots_main_study": sum(
            run["completed_vector_step_slots"]
            for run in runs
            if run["tag"] in expected and run["status"] == "complete"
        ),
        "completed_valid_transitions_main_study": sum(
            run["completed_valid_transitions"] or 0
            for run in runs
            if run["tag"] in expected and run["status"] == "complete"
        ),
        "completed_vector_step_slots_all_runs": sum(
            run["completed_vector_step_slots"]
            for run in runs
            if run["status"] in ("complete", "tuning_complete")
        ),
        "completed_valid_transitions_all_runs": sum(
            run["completed_valid_transitions"] or 0
            for run in runs
            if run["status"] in ("complete", "tuning_complete")
        ),
        "metric_definitions": {
            "rmse": (
                "Only valid samples from each world's first episode; failures included, "
                "reset episodes excluded."
            ),
            "last_second_rmse": (
                "Each first episode's final observed second, including shortened failed episodes."
            ),
            "success": (
                "No termination, complete 10 s horizon, final observed second per-world "
                "RMSE < 0.10 (Figure8 < 0.15); Landing also speed RMSE < 0.10 m/s."
            ),
            "sample_budget": (
                "Vector-step slots count API slots; valid transitions exclude NEXT_STEP "
                "reset slots; BC samples separate."
            ),
            "comparability": (
                "Historical baseline is reevaluated on the same test seeds/initial "
                "states; historical published cross-reset RMSE is not compared directly."
            ),
            "validation_score": (
                "(success_rate, survival_rate, -first_episode_rmse), lexicographically maximized."
            ),
            "uncertainty": (
                "Per-policy Wilson intervals concern episodes; train-seed variation is "
                "separately reported as sample standard deviation."
            ),
        },
        "scope": {
            "gym_dynamics": (
                "so_rpy; cf2x_L250; 4D roll/pitch/yaw/collective-thrust policy with "
                "internal attitude response"
            ),
            "underactuation": (
                "Quadrotor horizontal force is coupled to tilt; these runs do not "
                "establish direct rotor control, fixed-wing support, or real-world "
                "transfer."
            ),
            "landing": (
                "Official target is 0.1 m above ground, a near-ground hover rather than "
                "contact touchdown."
            ),
            "optimality": (
                "Finite budgets and a plateau heuristic provide observed evidence, not a "
                "global upper-bound proof."
            ),
        },
        "tasks": tasks,
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "summary.json.tmp"
    temporary.write_text(
        json.dumps(clean(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(output / "summary.json")
    plot_overview(tasks, histories, output / "overview.png")
    print(
        json.dumps(
            {
                "summary": relative(output / "summary.json"),
                "status": summary["status"],
                "completed_runs": summary["completed_runs"],
                "recommendations": {
                    task: entry["recommended_demonstration"]["tag"]
                    if entry["recommended_demonstration"]
                    else None
                    for task, entry in tasks.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
