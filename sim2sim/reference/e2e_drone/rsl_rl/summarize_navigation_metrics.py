"""Pool completed navigation episodes across seeds without loading the simulator."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics


FLAGS = ("success", "collision", "numerical_instability", "timeout")
OUTCOMES = (*FLAGS, "other_death")
MEASUREMENTS = (
    "duration_s", "path_length_m", "mean_speed_m_s", "max_speed_m_s",
    "start_goal_distance_m", "commanded_speed_m_s",
)
REQUIRED_COLUMNS = {"env_id", "episode_index", "outcome", *FLAGS, *MEASUREMENTS}


def number(value, name: str, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def integer(value, name: str, *, positive: bool = False) -> int:
    result = number(value, name, positive=positive)
    if not result.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(result)


def boolean(value: str, name: str) -> bool:
    choices = {"true": True, "false": False, "1": True, "0": False}
    if value.strip().lower() not in choices:
        raise ValueError(f"{name} must be True/False or 1/0")
    return choices[value.strip().lower()]


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "mean": statistics.fmean(ordered), "median": percentile(.5),
        "p10": percentile(.1), "p90": percentile(.9),
        "min": ordered[0], "max": ordered[-1], "total": math.fsum(ordered),
    }


def episode_statistics(rows: list[dict]) -> dict:
    count = len(rows)
    result = {"num_episodes": count}
    for flag in FLAGS:
        total = sum(row[flag] for row in rows)
        result[flag + "_count"] = total
        result[flag + "_rate"] = total / count
    outcomes = Counter(row["outcome"] for row in rows)
    result["outcome_counts"] = {name: outcomes[name] for name in OUTCOMES}
    duration = math.fsum(row["duration_s"] for row in rows)
    result["speed_m_s"] = {
        "time_weighted_mean": math.fsum(row["mean_speed_m_s"] * row["duration_s"] for row in rows) / duration,
        "max": max(row["max_speed_m_s"] for row in rows),
    }
    result["episode_mean_speed_m_s"] = distribution([row["mean_speed_m_s"] for row in rows])
    for name in ("duration_s", "path_length_m", "start_goal_distance_m", "commanded_speed_m_s"):
        result[name] = distribution([row[name] for row in rows])
    return result


def cruising_statistics(reports: list[dict]) -> dict:
    samples, weighted, maxima = [], [], []
    for report in reports:
        speed = report["cruising_speed_m_s"]
        count = integer(speed["samples"], "cruising speed samples")
        samples.append(count)
        if count:
            mean = number(speed["time_weighted_mean"], "cruising mean speed")
            maximum = number(speed["max"], "cruising maximum speed")
            if maximum + 1e-6 < mean:
                raise ValueError("Cruising maximum speed is below its mean")
            weighted.append(mean * count)
            maxima.append(maximum)
    total = sum(samples)
    return {"samples": total, "time_weighted_mean": math.fsum(weighted) / total if total else None,
            "max": max(maxima) if maxima else None}


def read_seed(path: Path) -> tuple[dict, list[dict], list[str]]:
    report = json.loads(path.read_text())
    seed = integer(report["seed"], "seed")
    if path.parent.name != f"seed_{seed}":
        raise ValueError(f"{path}: seed does not match its directory")
    num_envs = integer(report["num_envs"], "num_envs", positive=True)
    quota = integer(report["episodes_per_env"], "episodes_per_env", positive=True)
    count = integer(report["num_episodes"], "num_episodes", positive=True)
    csv_path = path.with_name(path.name.removesuffix(".summary.json") + ".episodes.csv")
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        missing = REQUIRED_COLUMNS - set(columns)
        if missing:
            raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if len(rows) != count or count != num_envs * quota:
        raise ValueError(f"{path}: episode row count disagrees with summary or environment quotas")
    for row in rows:
        if None in row or any(row[name] is None for name in REQUIRED_COLUMNS):
            raise ValueError(f"{csv_path}: malformed episode row")
        row["env_id"] = integer(row["env_id"], "env_id")
        row["episode_index"] = integer(row["episode_index"], "episode_index", positive=True)
        for flag in FLAGS:
            row[flag] = boolean(row[flag], flag)
        for name in MEASUREMENTS:
            row[name] = number(row[name], name, positive=name == "duration_s")
        if row["max_speed_m_s"] + 1e-6 < row["mean_speed_m_s"]:
            raise ValueError(f"{csv_path}: maximum speed is below episode mean speed")
        outcome = next((name for name in FLAGS if row[name]), "other_death")
        if row["outcome"] != outcome:
            raise ValueError(f"{csv_path}: outcome disagrees with flag priority")
        if "seed" in row and integer(row["seed"], "row seed") != seed:
            raise ValueError(f"{csv_path}: row seed disagrees with summary")
        row["seed"] = seed
    actual = Counter((row["env_id"], row["episode_index"]) for row in rows)
    expected = Counter((env, episode) for env in range(num_envs) for episode in range(1, quota + 1))
    if actual != expected:
        raise ValueError(f"{csv_path}: each environment must have exactly episode indices 1..{quota}")
    computed = episode_statistics(rows)
    for flag in FLAGS:
        rate_key, count_key = flag + "_rate", flag + "_count"
        if rate_key in report or flag in ("success", "collision"):
            rate = number(report[rate_key], rate_key)
            if not math.isclose(rate, computed[rate_key], rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError(f"{path}: {rate_key} disagrees with episode flags")
        if count_key in report and integer(report[count_key], count_key) != computed[count_key]:
            raise ValueError(f"{path}: {count_key} disagrees with episode flags")
    if "outcome_counts" in report and report["outcome_counts"] != computed["outcome_counts"]:
        raise ValueError(f"{path}: exclusive outcome counts disagree with episode flags")
    if "evaluator_success_rate" in report and not math.isclose(
        number(report["evaluator_success_rate"], "evaluator_success_rate"), computed["success_rate"], abs_tol=1e-8
    ):
        raise ValueError(f"{path}: evaluator success rate disagrees with episode flags")
    environment_path = path.with_name(path.name.removesuffix(".summary.json") + ".environment.json")
    environment = json.loads(environment_path.read_text())
    report.update({
        "seed": seed, "num_envs": num_envs, "episodes_per_env": quota,
        "sampling_frequency_hz": number(report["sampling_frequency_hz"], "sampling_frequency_hz", positive=True),
        "contact_force_threshold_n": number(environment["contact_force_threshold"], "contact_force_threshold"),
        "simulation_steps": integer(report["simulation_steps"], "simulation_steps", positive=True),
        "evaluation_wall_time_s": number(report["evaluation_wall_time_s"], "evaluation_wall_time_s", positive=True),
        "source_summary": str(path),
    })
    return report, rows, columns


def summarize(input_dir: Path) -> dict:
    paths = sorted(input_dir.glob("seed_*/model_*.summary.json"))
    if not paths:
        raise ValueError(f"No seed_*/model_*.summary.json files found under {input_dir}")
    loaded = sorted((read_seed(path) for path in paths), key=lambda item: item[0]["seed"])
    reports = [item[0] for item in loaded]
    if len({report["seed"] for report in reports}) != len(reports):
        raise ValueError("Found multiple checkpoint summaries for the same seed")
    for key in ("checkpoint_sha256", "task", "sampling_frequency_hz", "contact_force_threshold_n"):
        if len({report[key] for report in reports}) != 1:
            raise ValueError(f"Cannot pool different {key} values")
    reference = reports[0]
    digest = reference["checkpoint_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 hex digest")
    if not isinstance(reference["task"], str) or not reference["task"]:
        raise ValueError("task must be a nonempty string")
    rows = [row for _, episodes, _ in loaded for row in episodes]
    result = episode_statistics(rows)
    result.update({
        "schema": "navigation_metrics_pooled_v1", "task": reference["task"], "checkpoint_sha256": digest,
        "num_seeds": len(reports), "seeds": [report["seed"] for report in reports],
        "sampling_frequency_hz": reference["sampling_frequency_hz"],
        "contact_force_threshold_n": reference["contact_force_threshold_n"],
        "simulation_steps_total": sum(report["simulation_steps"] for report in reports),
        "evaluation_wall_time_s_total": math.fsum(report["evaluation_wall_time_s"] for report in reports),
        "per_seed": [
            {
                **{key: report[key] for key in ("seed", "num_envs", "episodes_per_env", "simulation_steps", "evaluation_wall_time_s")},
                **episode_statistics(episodes), "source_summary": str(Path(report["source_summary"]).relative_to(input_dir)),
            }
            for report, episodes, _ in loaded
        ],
        "definitions": {
            "rates": "Raw flagged episode counts divided by all accepted episodes; collision can overlap success.",
            "outcome": "Exclusive priority: success, collision, numerical instability, timeout, other death.",
            "speed": "3D linear speed sampled before reset; pooled mean weights each episode mean by its duration.",
            "episode_mean_speed": "Distribution of episode mean speeds, with each completed episode weighted equally.",
            "instantaneous_speed_percentiles": "Not pooled: instantaneous samples are not stored in episode CSV files.",
            "path_length": f"3D displacement summed at {reference['sampling_frequency_hz']:g} Hz; reset teleports excluded.",
            "collision": f"Any task-detected body contact above {reference['contact_force_threshold_n']:g} N, including simultaneous success.",
            "sampling": "Completed episodes within each environment quota. Episodes share maps; no independence or confidence interval is assumed.",
            "evaluation_wall_time_s_total": "Sum of individual evaluation wall times, excluding startup and loading.",
        },
    })
    if all("cruising_speed_m_s" in report for report in reports):
        result["cruising_speed_m_s"] = cruising_statistics(reports)
        result["definitions"]["cruising_speed"] = (
            "Speed samples with more than 1 m remaining to the goal; pooled mean weights seed means by sample count. "
            "Sampling frequencies match; instantaneous percentiles are not pooled."
        )
        for seed_result, report in zip(result["per_seed"], reports):
            seed_result["cruising_speed_m_s"] = cruising_statistics([report])
    columns = list(dict.fromkeys(["seed", *(column for _, _, names in loaded for column in names)]))
    with (input_dir / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (input_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing seed_*/model_*.summary.json exports.")
    args = parser.parse_args()
    try:
        result = summarize(args.input_dir)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f"Invalid navigation metrics: {error}\n")
    print("seed  episodes  success  collision  mean speed (m/s)")
    for report in result["per_seed"]:
        print(f"{report['seed']:4d}  {report['num_episodes']:8d}  {report['success_rate']:7.2%}  "
              f"{report['collision_rate']:9.2%}  {report['speed_m_s']['time_weighted_mean']:16.3f}")
    print(f"Pooled {result['num_episodes']} episodes: success={result['success_rate']:.2%}, "
          f"collision={result['collision_rate']:.2%}, mean speed={result['speed_m_s']['time_weighted_mean']:.3f} m/s")
    print(f"Saved {args.input_dir / 'summary.json'} and {args.input_dir / 'episodes.csv'}")


if __name__ == "__main__":
    main()
