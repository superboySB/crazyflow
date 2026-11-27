"""Deployment weights and launcher paths remain usable from a fresh checkout."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_only_selected_policy_and_official_baseline_are_shipped():
    expected = {
        "crazyflow/policy.npz": "ac8dda90bc960a1e683b63a6bd9697fe0a40e98735dc82e7a09811392b08c206",
        "baseline/model_36825.pt": (
            "7c3ecc53bf468989789b1deee0c3efd6856438604aa5b0f19e35693527d15cc3"
        ),
    }
    model_root = ROOT / "models"
    deployed = {
        str(path.relative_to(model_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in model_root.rglob("*")
        if path.suffix in {".npz", ".pt", ".pth", ".msgpack", ".ckpt", ".onnx"}
    }
    assert deployed == expected


def test_bundled_policy_launch_does_not_create_another_model(tmp_path: Path):
    shell = """
source "$1/scripts/common.sh"
CF_OUTPUTS=$2
cf_default_checkpoint
cf_checkpoint_to_container models/crazyflow/policy.npz
"""
    result = subprocess.run(
        ["bash", "-c", shell, "check-release", str(ROOT), str(tmp_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.splitlines() == ["/opt/crazyflow/models/crazyflow/policy.npz"] * 2
    assert list(tmp_path.iterdir()) == []


def test_vendored_evaluation_files_match_the_packaged_manifest():
    provenance = json.loads((ROOT / "sim2sim/provenance.json").read_text())
    assert provenance["files"]
    for record in provenance["files"]:
        path = (ROOT / record["destination"]).resolve()
        path.relative_to(ROOT)
        assert path.is_file(), record["destination"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        assert path.stat().st_size == record["bytes"]
