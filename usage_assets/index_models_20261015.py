"""Index delivered weights and distinguish selected from completed training budgets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "trained_models"
GYM = ROOT / "usage_assets" / "fulltrain_20261015"


def main() -> None:
    """Create a compact index after all declared experiments pass verification."""
    verification = json.loads((GYM / "artifact_verification.json").read_text())
    assert verification["completed_main_runs"] == 12
    assert verification["completed_calibrated_runs"] == 6
    assert verification["completed_promoted_velocity_runs"] == 3
    summary = json.loads((GYM / "summary.json").read_text())
    assert summary["status"] == "complete"
    import torch

    gym = []
    for path in sorted(GYM.glob("*.json")):
        record = json.loads(path.read_text())
        if "checkpoint" not in record or "config" not in record:
            continue
        baseline = path.name.startswith("baseline_")
        screening = "tests" not in record
        item = {
            "kind": (
                "historical_baseline_reevaluated"
                if baseline
                else "validation_only_screening"
                if screening
                else "full_training_run"
            ),
            "result": str(path.relative_to(ROOT)),
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "config": record["config"],
            "selected_update": record.get("selected_update"),
            "completed_updates": record.get("completed_updates"),
            "completed_vector_step_slots": record.get("completed_vector_step_slots"),
            "completed_valid_transitions": record.get("completed_valid_transitions"),
            "goal_rng_protocol": record.get(
                "goal_rng_protocol", record.get("evaluation_goal_rng_protocol")
            ),
        }
        weight = ROOT / record["checkpoint"]
        saved = torch.load(weight, map_location="cpu", weights_only=False)
        item["config"] = saved["config"]
        item["config"].setdefault("teacher_model", "legacy")
        item["selected_vector_step_slots"] = saved.get("trainer_state", {}).get("global_step")
        item["checkpoint_format_version"] = saved.get("format_version", 0)
        if baseline:
            history = saved.get("history", {}).get("timesteps", [])
            item["historical_budget_from_checkpoint_history"] = history[-1] if history else None
        else:
            last = weight.with_name(weight.stem + "_last.pt")
            recovery = torch.load(last, map_location="cpu", weights_only=False)
            state = recovery["trainer_state"]
            item["completed_updates"] = state["iteration"]
            item["completed_vector_step_slots"] = state["global_step"]
            item["completed_valid_transitions"] = state["valid_step"]
            item["recovery_checkpoint"] = str(last.relative_to(ROOT))
            item["recovery_checkpoint_sha256"] = hashlib.sha256(last.read_bytes()).hexdigest()
        gym.append(item)
    recommended = {
        task: {
            "checkpoint": result["recommended_demonstration"]["checkpoint"],
            "sha256": result["recommended_demonstration"]["checkpoint_sha256"],
            "selected_update": result["recommended_demonstration"]["selected_update"],
            "rule": result["recommended_demonstration"]["selection_rule"],
        }
        for task, result in summary["tasks"].items()
    }
    racing_path = ROOT / "usage_assets" / "racing_fulltrain_20261015.json"
    racing = json.loads(racing_path.read_text())
    weights = []
    for path in sorted(MODELS.iterdir()):
        if path.suffix not in (".pt", ".ckpt"):
            continue
        weights.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    output = {
        "schema": 1,
        "date": "2026-10-15",
        "official_main": "c95f69bc54f830e859fee4ac5734bf571acd81f3",
        "scope": (
            "Local simulation-trained policies. Completed run budget differs from "
            "selected checkpoint budget. Analytic physics/controller references have no weights. "
            "Invalid correlated-RNG runs remain outside this directory in ignored saves."
        ),
        "gym": gym,
        "validation_selected_gym": recommended,
        "official_manifest": "trained_models/official_manifest_20261015.json",
        "racing_result": str(racing_path.relative_to(ROOT)),
        "validation_selected_racing": racing["selected"],
        "all_weight_files": weights,
        "weight_count": len(weights),
        "total_weight_bytes": sum(item["bytes"] for item in weights),
    }
    path = MODELS / "manifest_20261015.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"manifest": str(path.relative_to(ROOT)), "weights": len(weights)}))


if __name__ == "__main__":
    main()
