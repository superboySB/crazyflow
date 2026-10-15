"""Independently check reported scores against saved evaluation arrays and checkpoint hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "usage_assets"
GYM = ASSETS / "fulltrain_20261015"


def read_json(path: Path) -> object:
    """Reject nonstandard NaN/Infinity JSON rather than silently accepting failed scores."""

    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite JSON token {value}: {path}")

    return json.loads(path.read_text(), parse_constant=reject)


def near(actual: float, expected: float, label: str) -> None:
    """Allow float32 reduction rounding while detecting stale or mismatched evaluation files."""
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-7, err_msg=label)


def check_hash(record: dict) -> None:
    """Require the exact weights identified by the result to be present."""
    path = ROOT / record["checkpoint"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == record["checkpoint_sha256"], f"Checkpoint hash mismatch: {path}"


def derive_success(raw: object, config: dict) -> np.ndarray:
    """Rebuild the task definition without trusting the saved success boolean."""
    task = config["env_id"]
    horizon = int(config["episode_time"] * config["freq"])
    threshold = 0.15 if task == "DroneFigureEightTrajectory-v0" else 0.10
    success = (
        raw["per_env_survival"]
        & (raw["per_env_episode_steps"] == horizon)
        & (raw["per_env_last_rmse"] < threshold)
    )
    if task == "DroneLanding-v0":
        success &= raw["per_env_last_speed_rmse"] < 0.10
    np.testing.assert_array_equal(success, raw["per_env_success"])
    return success


def verify_gym() -> dict:
    """Recompute first-episode RMSE and success from all saved per-world measurements."""
    completed, calibrated, promoted, checked = 0, 0, 0, 0
    for path in sorted(GYM.glob("*.json")):
        record = read_json(path)
        if not isinstance(record, dict) or "tests" not in record:
            continue
        check_hash(record)
        assert len(record["tests"]) == 3, f"Missing test seeds: {path}"
        for label in ("tests", "horizontal_tests"):
            suffix = "_horizontal" if label == "horizontal_tests" else ""
            for test in record.get(label, []):
                raw_path = GYM / f"{path.stem}{suffix}_test{test['seed']}.npz"
                with np.load(raw_path) as raw:
                    lengths = raw["per_env_episode_steps"]
                    rmse = np.sqrt(np.sum(raw["per_env_rmse"] ** 2 * lengths) / lengths.sum())
                    near(rmse, test["metrics"]["rmse"], str(raw_path))
                    success = derive_success(raw, record["config"])
                    near(success.mean(), test["metrics"]["success_rate"], str(raw_path))
                    near(
                        raw["per_env_survival"].mean(),
                        test["metrics"]["survival_rate"],
                        str(raw_path),
                    )
                    assert lengths.shape == (1024,), f"Unexpected test population: {raw_path}"
                    if "successes" in test:
                        assert int(success.sum()) == test["successes"]
                checked += 1
        if not path.name.startswith("baseline_"):
            import torch

            checkpoint = ROOT / record["checkpoint"]
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert saved.get("reset_protocol") == "independent_goal_rng_v1", path
            assert saved.get("format_version", 0) >= 3, path
            assert all(torch.isfinite(v).all().item() for v in saved["agent"].values()), path
            assert saved["config"].get("teacher_model", "legacy") == record["config"].get(
                "teacher_model", "legacy"
            ), path
            last_path = checkpoint.with_name(checkpoint.stem + "_last.pt")
            assert (
                hashlib.sha256(last_path.read_bytes()).hexdigest()
                == record["last_checkpoint_sha256"]
            ), path
            last = torch.load(last_path, map_location="cpu", weights_only=False)
            assert all(torch.isfinite(v).all().item() for v in last["agent"].values()), path
            cfg = record["config"]
            batch = cfg["num_envs"] * cfg["num_steps"]
            updates = cfg["total_timesteps"] // batch
            assert len(record["history"]["timesteps"]) == updates, path
            assert record["history"]["timesteps"][-1] == updates * batch, path
            assert last["trainer_state"]["iteration"] == updates, path
            assert last["trainer_state"]["valid_step"] == record["completed_valid_transitions"], (
                path
            )
            if cfg["total_timesteps"] == 200_000_000:
                if cfg.get("teacher_model", "legacy") == "calibrated":
                    calibrated += 1
                elif cfg["gamma"] == 0.99:
                    completed += 1
                elif cfg["gamma"] == 0.995 and cfg["env_id"] == "DroneReachVel-v0":
                    promoted += 1
    return {
        "completed_main_runs": completed,
        "completed_calibrated_runs": calibrated,
        "completed_promoted_velocity_runs": promoted,
        "checked_gym_test_populations": checked,
    }


def verify_official() -> dict:
    """Recompute all 64-world official policy metrics, retaining every failed world."""
    records, scenarios = 0, 0
    for path in sorted(ASSETS.glob("official_*_seed*.json")):
        record = read_json(path)
        if not isinstance(record, dict) or "evaluation" not in record or "checkpoint" not in record:
            continue
        check_hash(record)
        for label, metrics in record["evaluation"].items():
            raw_path = ASSETS / f"{path.stem}_{label}.npz"
            with np.load(raw_path) as raw:
                error = raw["full_error_sq"].astype(np.float64)
                active = raw["full_active"]
                crashes = raw["full_terminated"].any(axis=0)
                successes = raw["full_success"].any(axis=0)
                assert active.shape[1] == 64, raw_path
                assert np.isfinite(error).all(), raw_path
                near(
                    np.sqrt((error * active).sum() / max(active.sum(), 1)),
                    metrics["tracking_rmse_m"],
                    str(raw_path),
                )
                near(crashes.mean(), metrics["crash_rate"], str(raw_path))
                if "goal_success_rate" in metrics:
                    near(successes.mean(), metrics["goal_success_rate"], str(raw_path))
                else:
                    assert not successes.any(), raw_path
                if "alive_without_crash_rate" in metrics:
                    near((~crashes).mean(), metrics["alive_without_crash_rate"], str(raw_path))
                complete = (~crashes) & (active.sum(axis=0) >= len(active) - 1)
                key = "full_horizon_completion_rate"
                if key in metrics:
                    near(complete.mean(), metrics[key], str(raw_path))
            scenarios += 1
        records += 1
    return {"official_training_records": records, "checked_official_scenarios": scenarios}


def main() -> None:
    """Check complete experiments without running additional training or tuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = verify_gym() | verify_official()
    if args.require_complete:
        assert result["completed_main_runs"] == 12, result
        assert result["completed_calibrated_runs"] == 6, result
        assert result["completed_promoted_velocity_runs"] == 3, result
        assert result["official_training_records"] == 30, result
    result["all_checked_hashes_and_scores_match"] = True
    output = GYM / "artifact_verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
