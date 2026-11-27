#!/usr/bin/env python3
"""Aggregate the five paired Isaac evaluations and retain their audit records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    """Return the linearly interpolated percentile of a nonempty sample."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def distribution(values: list[float]) -> dict[str, float]:
    """Summarize episode-level measurements without mixing per-step medians."""
    return {
        "mean": statistics.fmean(values),
        "median": percentile(values, 0.5),
        "p10": percentile(values, 0.1),
        "p90": percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
        "total": sum(values),
    }


def read_episodes(directory: Path, policy: str, seed: int) -> list[dict[str, Any]]:
    """Read completed episodes from one frozen-evaluator invocation."""
    paths = list(directory.glob("*.episodes.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one episode CSV in {directory}")
    with paths[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    converted = []
    for row in rows:
        parsed: dict[str, Any] = {"policy": policy, "seed": seed}
        for key, value in row.items():
            if key in ("env_id", "episode_index"):
                parsed[key] = int(value)
            elif key in ("success", "collision", "numerical_instability", "timeout"):
                if value not in ("True", "False"):
                    raise ValueError(f"Invalid episode flag {key}={value}")
                parsed[key] = value == "True"
            elif key == "outcome":
                parsed[key] = value
            else:
                parsed[key] = float(value)
        converted.append(parsed)
    return converted


def aggregate(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine accepted episodes with time-weighted speed and explicit outcomes."""
    count = len(rows)
    total_duration = sum(row["duration_s"] for row in rows)
    result: dict[str, Any] = {"episodes": count}
    for flag in ("success", "collision", "numerical_instability", "timeout"):
        result[flag + "_count"] = sum(row[flag] for row in rows)
        result[flag + "_rate"] = result[flag + "_count"] / count
    result["outcome_counts"] = {
        outcome: sum(row["outcome"] == outcome for row in rows)
        for outcome in ("success", "collision", "numerical_instability", "timeout", "other_death")
    }
    result["actual_speed_m_s"] = {
        "time_weighted_mean": sum(row["mean_speed_m_s"] * row["duration_s"] for row in rows)
        / total_duration,
        "maximum_sampled": max(row["max_speed_m_s"] for row in rows),
    }
    cruising = [report["cruising_speed_m_s"] for report in summaries]
    cruise_samples = sum(report["samples"] for report in cruising)
    result["cruising_speed_m_s"] = {
        "samples": cruise_samples,
        "time_weighted_mean": sum(
            report["time_weighted_mean"] * report["samples"]
            for report in cruising
            if report["samples"]
        )
        / cruise_samples
        if cruise_samples
        else None,
    }
    for field in ("path_length_m", "duration_s", "start_goal_distance_m", "commanded_speed_m_s"):
        result[field] = distribution([row[field] for row in rows])
    result["evaluation_wall_time_s"] = sum(report["evaluation_wall_time_s"] for report in summaries)
    return result


def summarize(
    input_dir: Path, output_dir: Path, expected_seeds: list[int], num_envs: int
) -> dict[str, Any]:
    """Validate paired starts and policies, aggregate results, and copy audit data."""
    reports = [
        json.loads(path.read_text()) for path in sorted(input_dir.glob("**/comparison.json"))
    ]
    seeds = sorted(report["seed"] for report in reports)
    if seeds != sorted(expected_seeds):
        raise ValueError(f"Expected seeds {expected_seeds}, found {seeds}")
    by_seed = {report["seed"]: report for report in reports}
    paths = {
        json.loads(path.read_text())["seed"]: path.parent
        for path in input_dir.glob("**/comparison.json")
    }
    hashes = {
        policy: {report[policy]["checkpoint_sha256"] for report in reports}
        for policy in ("baseline", "crazyflow")
    }
    if any(len(values) != 1 for values in hashes.values()):
        raise ValueError("A checkpoint changed between seeds")
    all_rows: list[dict[str, Any]] = []
    scene_hashes = {}
    paired = {
        "both_success": 0,
        "baseline_only_success": 0,
        "crazyflow_only_success": 0,
        "both_failure": 0,
    }
    per_seed = []
    for seed in seeds:
        report, path = by_seed[seed], paths[seed]
        pair_rows = {}
        scenes = {}
        seed_result: dict[str, Any] = {"seed": seed}
        for policy in ("baseline", "crazyflow"):
            summary = report[policy]
            if (
                summary["num_envs"] != num_envs
                or summary["episodes_per_env"] != 1
                or summary["num_episodes"] != num_envs
            ):
                raise ValueError(f"Incomplete or unpaired episode quota for {policy}, seed {seed}")
            rows = read_episodes(path / policy, policy, seed)
            if len(rows) != num_envs or any(row["episode_index"] != 1 for row in rows):
                raise ValueError("Each environment must contribute exactly one episode")
            pair_rows[policy] = {row["env_id"]: row for row in rows}
            if set(pair_rows[policy]) != set(range(num_envs)):
                raise ValueError("Missing or duplicated environment identifiers")
            all_rows.extend(rows)
            scenes[policy] = json.loads((path / policy / "initial_scene.json").read_text())
            seed_result[policy] = {
                "successes": sum(row["success"] for row in rows),
                "collisions": sum(row["collision"] for row in rows),
                "episodes": len(rows),
            }
        required_scene_fields = {
            "root_state_w",
            "goal_positions_w",
            "desired_speed",
            "controller_mass",
            "controller_inertia",
            "wind_acceleration",
            "thrust_effectiveness",
            "thrust_degradation_rate",
            "map_occupied_points_sha256",
            "scene_sha256",
        }
        for scene in scenes.values():
            if not required_scene_fields.issubset(scene):
                raise ValueError("This suite requires the expanded initial randomization audit")
            payload = {key: value for key, value in scene.items() if key != "scene_sha256"}
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            if hashlib.sha256(canonical.encode()).hexdigest() != scene["scene_sha256"]:
                raise ValueError("Initial scene contents disagree with their saved hash")
        if scenes["baseline"] != scenes["crazyflow"]:
            raise ValueError(
                f"Initial map, state or randomization parameters differ for seed {seed}"
            )
        scene_hashes[str(seed)] = scenes["baseline"]["scene_sha256"]
        for env_id in range(num_envs):
            baseline, candidate = pair_rows["baseline"][env_id], pair_rows["crazyflow"][env_id]
            for field in ("start_goal_distance_m", "commanded_speed_m_s"):
                if abs(baseline[field] - candidate[field]) > 1e-6:
                    raise ValueError(
                        f"Episode initial conditions differ for seed {seed}, env {env_id}"
                    )
            outcome = (
                "both_success"
                if baseline["success"] and candidate["success"]
                else "baseline_only_success"
                if baseline["success"]
                else "crazyflow_only_success"
                if candidate["success"]
                else "both_failure"
            )
            paired[outcome] += 1
        per_seed.append(seed_result)
        destination = output_dir / "seeds" / f"seed_{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        for source in path.rglob("*"):
            if source.is_file() and source.suffix in (".json", ".csv"):
                target = destination / source.relative_to(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    result = {
        "date": "2026-11-27",
        "timezone": "Asia/Shanghai",
        "task": "baseline",
        "seeds": seeds,
        "num_envs_per_seed": num_envs,
        "episodes_per_env": 1,
        "episodes_per_policy": len(seeds) * num_envs,
        "checkpoint_sha256": {key: next(iter(value)) for key, value in hashes.items()},
        "initial_scene_sha256": scene_hashes,
        "protocol": "Same seeded map, initial body states, goals, commanded speeds, controller "
        "mass/inertia, initial wind and thrust uncertainty; deterministic actor means; "
        "unchanged CrazyRL Isaac task. Later stochastic draws follow each policy's "
        "episode/reset progression under the same randomization laws.",
        "sampling_note": f"{len(seeds)} shared layouts with paired randomized starts/goals. "
        "Episodes "
        "within a layout are not independent map draws.",
        "per_seed": per_seed,
        "paired_success_outcomes": paired,
    }
    for policy in ("baseline", "crazyflow"):
        result[policy] = aggregate(
            [row for row in all_rows if row["policy"] == policy],
            [report[policy] for report in reports],
        )
    result["success_rate_delta_percentage_points"] = (
        result["crazyflow"]["success_rate"] - result["baseline"]["success_rate"]
    ) * 100
    result["collision_rate_delta_percentage_points"] = (
        result["crazyflow"]["collision_rate"] - result["baseline"]["collision_rate"]
    ) * 100
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output_dir / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    return result


def main() -> None:
    """Summarize a complete paired suite from the in-repository evaluation outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--num-envs", type=int, default=32)
    args = parser.parse_args()
    result = summarize(
        args.input_dir,
        args.output_dir,
        [int(seed) for seed in args.seeds.split(",")],
        args.num_envs,
    )
    print(json.dumps({policy: result[policy] for policy in ("baseline", "crazyflow")}, indent=2))


if __name__ == "__main__":
    main()
