#!/usr/bin/env python3
"""Stage only CrazyRL's frozen Isaac evaluation dependencies and local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SOURCES = [
    "LICENSE",
    "e2e_drone/__init__.py",
    "e2e_drone/utils/__init__.py",
    "e2e_drone/rsl_rl/__init__.py",
    *[
        f"e2e_drone/rsl_rl/{name}.py"
        for name in (
            "cli_args",
            "play",
            "evaluate_success_rate",
            "navigation_metrics",
            "summarize_navigation_metrics",
        )
    ],
    *[
        f"e2e_drone/tasks/baseline/{name}"
        for name in (
            "__init__.py",
            "quad_baseline_env.py",
            "quad_baseline_env_cfg.py",
            "agents/__init__.py",
            "agents/rsl_rl_ppo_cfg.py",
        )
    ],
    *[
        f"e2e_drone/utils/{name}.py"
        for name in (
            "asset_paths",
            "bev_utils",
            "controller",
            "death_replay_collector",
            "death_replay_viz",
            "height_randomizer",
            "map_generator",
            "reloadable_raycaster_camera",
            "thrust_uncertainty",
            "wind_gen",
        )
    ],
    *[
        f"assets/Assets/Isaac/4.5/Isaac/{name}"
        for name in (
            "Robots/Crazyflie/cf2x.usd",
            "Props/UIElements/arrow_x.usd",
            "Props/UIElements/frame_prim.usd",
        )
    ],
]
ARTIFACTS = [
    "assets/replementation/rl/model_36825.pt",
    "assets/replementation/rl/events.out.tfevents.1768936229.dcj.32.0",
    "assets/replementation/rl/params/env.yaml",
    "assets/replementation/rl/params/agent.yaml",
]


def sha256(path: Path) -> str:
    """Hash the exact bytes of a staged source or artifact file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_file(
    source: Path, relative: str, destination: Path, records: list[dict[str, Any]]
) -> None:
    """Copy a required file and append its provenance record."""
    src = source / relative
    dst = destination / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    records.append(
        {
            "source": relative,
            "destination": str(dst.relative_to(ROOT)),
            "sha256": sha256(dst),
            "bytes": dst.stat().st_size,
        }
    )


def main() -> None:
    """Stage the minimal evaluation bundle and exact reference measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.home() / "projects/CrazyRL")
    args = parser.parse_args()
    source = args.source.resolve()
    records = []
    for relative in SOURCES:
        copy_file(source, relative, ROOT / "reference", records)
    # Register only the copied evaluation task. The upstream package imports an
    # unrelated maze task; no task behavior, physics or evaluation code changes.
    task_init = ROOT / "reference/e2e_drone/tasks/__init__.py"
    task_init.parent.mkdir(parents=True, exist_ok=True)
    task_init.write_text(
        '"""Frozen CrazyRL baseline evaluation task only."""\nfrom . import baseline\n'
    )
    for relative in ARTIFACTS:
        copy_file(source, relative, ROOT / "artifacts", records)
    for src in sorted((source / "docs/evaluation/20261127").rglob("*")):
        if src.is_file():
            rel = src.relative_to(source / "docs/evaluation/20261127")
            dst = ROOT / "results/reference-20261127" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            records.append(
                {
                    "source": str(src.relative_to(source)),
                    "destination": str(dst.relative_to(ROOT)),
                    "sha256": sha256(dst),
                    "bytes": dst.stat().st_size,
                }
            )
    copy_file(source, "docker/simulation.dockerfile", ROOT / "artifacts/upstream", records)
    # Keep an exact, tracked Dockerfile for rebuilding the pinned base image.
    shutil.copy2(source / "docker/simulation.dockerfile", ROOT / "docker/isaac45.dockerfile")
    manifest = {
        "source_repository": str(source),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(source), "status", "--porcelain"], text=True
            ).strip()
        ),
        "source_note": (
            "note.md, verified 2026-11-27; evaluation only, no training entrypoint copied"
        ),
        "local_changes": [
            {
                "path": "reference/e2e_drone/tasks/__init__.py",
                "reason": "register copied baseline only; exclude maze task",
            }
        ],
        "files": records,
    }
    (ROOT / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Staged {len(records)} files from {source}; "
        "ignored weights/TensorBoard in sim2sim/artifacts"
    )


if __name__ == "__main__":
    main()
