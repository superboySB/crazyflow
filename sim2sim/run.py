#!/usr/bin/env python3
"""Self-contained, dedicated-container Isaac evaluation and playback launcher."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
IMAGE = os.environ.get("CRAZYFLOW_EVAL_IMAGE", "crazyflow:dani-crazyrl-eval")
BASELINE = ROOT / "models/baseline/model_36825.pt"
CRAZYFLOW = ROOT / "models/crazyflow/policy.npz"
ISAAC_PYTHON = "/workspace/isaaclab/_isaac_sim/python.sh"


def container_path(path: str | Path) -> str:
    """Resolve an in-repository host file to the dedicated container mount."""
    return "/workspace/crazyflow/" + str(Path(path).resolve().relative_to(ROOT))


def build() -> None:
    """Build Isaac evaluation directly from its independent pinned Docker recipe."""
    subprocess.run(
        [
            "docker",
            "build",
            "--network=host",
            "--progress=plain",
            "-f",
            str(ROOT / "docker/sim2sim.Dockerfile"),
            "-t",
            IMAGE,
            str(ROOT),
        ],
        check=True,
    )


def docker_command(gui: bool = False, *, gpu: bool = True) -> list[str]:
    """Prepare an isolated container with local source, assets and ignored caches."""
    check = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    if check.returncode:
        build()
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--shm-size=2g",
        "--entrypoint",
        "/bin/bash",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "PRIVACY_CONSENT=Y",
        "-e",
        "OMNI_KIT_ALLOW_ROOT=1",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-v",
        f"{ROOT}:/workspace/crazyflow",
    ]
    if gpu:
        command += [
            "--gpus",
            os.environ.get("CRAZYFLOW_EVAL_GPUS", "all"),
            "-e",
            "CUDA_VISIBLE_DEVICES=" + os.environ.get("GPU_ID", "0"),
        ]
    cache_dirs = {
        "kit": "/isaac-sim/kit/cache",
        "ov": "/root/.cache/ov",
        "pip": "/root/.cache/pip",
        "glcache": "/root/.cache/nvidia/GLCache",
        "computecache": "/root/.nv/ComputeCache",
    }
    for name, destination in cache_dirs.items():
        path = ROOT / "outputs/sim2sim/cache" / name
        path.mkdir(parents=True, exist_ok=True)
        command += ["-v", f"{path}:{destination}:rw"]
    if gui:
        command += [
            "-e",
            "DISPLAY=" + os.environ.get("DISPLAY", ":0"),
            "-e",
            "QT_X11_NO_MITSHM=1",
            "-v",
            "/tmp/.X11-unix:/tmp/.X11-unix:rw",
        ]
    return command + [IMAGE, ISAAC_PYTHON]


def run(
    command: list[str], output: Path | None = None, policy: Path | None = None
) -> subprocess.CompletedProcess[Any] | None:
    """Execute inference and preserve its exact command, image and console log."""
    if output is None:
        return subprocess.run(command, cwd=ROOT, check=True)
    output.mkdir(parents=True, exist_ok=True)
    info = {
        "command": command,
        "image": IMAGE,
        "evaluation_source_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted((ROOT / "sim2sim").glob("*.py")))
            + (ROOT / "sim2sim/provenance.json").read_bytes()
        ).hexdigest(),
        "image_id": subprocess.check_output(
            ["docker", "image", "inspect", "-f", "{{.Id}}", IMAGE], text=True
        ).strip(),
        "checkpoint_sha256": hashlib.sha256(policy.read_bytes()).hexdigest() if policy else None,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with (output / "console.log").open("w") as logfile:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            logfile.write(line)
            logfile.flush()
            print(line, end="", flush=True)
        info["exit_code"] = process.wait()
    info["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (output / "invocation.json").write_text(json.dumps(info, indent=2) + "\n")
    if info["exit_code"]:
        raise subprocess.CalledProcessError(info["exit_code"], command)


def policy_path(value: str) -> Path:
    """Resolve a bundled policy alias or another in-repository checkpoint."""
    aliases = {"baseline": BASELINE, "crazyflow": CRAZYFLOW}
    path = aliases.get(value, Path(value).expanduser().resolve())
    path.relative_to(ROOT)
    if not path.is_file():
        raise FileNotFoundError(f"Missing in-repository policy: {path}")
    return path


def evaluate(
    args: argparse.Namespace, extra: list[str], value: str | None = None, output: Path | None = None
) -> dict[str, Any]:
    """Evaluate a checkpoint and require a completed per-episode metric summary."""
    policy = policy_path(value or args.policy)
    output = (output or Path(args.output)).resolve()
    command = docker_command(args.gui) + [
        "/workspace/crazyflow/sim2sim/evaluate.py",
        "--task",
        "baseline",
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--checkpoint_path",
        container_path(policy),
        "--max_episodes_per_env",
        str(args.episodes),
        "--metrics_dir",
        container_path(output),
        "--output_csv",
        container_path(output / "success.csv"),
    ]
    if not args.gui:
        command += ["--headless"]
    command += ["env.enable_difficulty_curriculum=False", "env.debug_vis=False", *extra]
    run(command, output, policy)
    summaries = list(output.glob("*.summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(
            "Expected exactly one completed evaluation summary; use a fresh output directory"
        )
    return json.loads(summaries[0].read_text())


def main() -> None:
    """Launch evaluation, paired comparison, playback or baseline TensorBoard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "evaluate", "compare", "play", "tensorboard"])
    parser.add_argument(
        "--policy",
        default="baseline",
        help="baseline, crazyflow, or an in-repository .npz/.pt path",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--output", default="outputs/sim2sim/evaluation")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--port", type=int, default=6006)
    args, extra = parser.parse_known_args()
    if args.num_envs <= 0 or args.episodes <= 0 or args.max_steps <= 0:
        parser.error("Environment count, episode count and step limit must be positive")
    if args.mode == "build":
        build()
    elif args.mode == "evaluate":
        evaluate(args, extra)
    elif args.mode == "compare":
        if args.policy == "baseline":
            parser.error("compare requires --policy /path/to/your/actor.npz")
        if args.episodes != 1:
            parser.error(
                "Fair paired comparison uses one episode per environment "
                "to preserve identical starts and goals"
            )
        output = Path(args.output).resolve()
        baseline = evaluate(args, extra, "baseline", output / "baseline")
        candidate = evaluate(args, extra, args.policy, output / "crazyflow")
        base_scene = json.loads((output / "baseline/initial_scene.json").read_text())
        candidate_scene = json.loads((output / "crazyflow/initial_scene.json").read_text())
        if base_scene["scene_sha256"] != candidate_scene["scene_sha256"]:
            raise RuntimeError("Initial scenes differ; comparison rejected")
        report = {
            "seed": args.seed,
            "scene_sha256": base_scene["scene_sha256"],
            "protocol": (
                "Identical generated map, initial states, goals, speeds, "
                "environment count, seed and settings; one episode per environment"
            ),
            "baseline": baseline,
            "crazyflow": candidate,
            "success_rate_delta": candidate["success_rate"] - baseline["success_rate"],
            "collision_rate_delta": candidate["collision_rate"] - baseline["collision_rate"],
        }
        (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "seed",
                        "scene_sha256",
                        "success_rate_delta",
                        "collision_rate_delta",
                    )
                },
                indent=2,
            )
        )
    elif args.mode == "tensorboard":
        run(
            docker_command(gpu=False)
            + [
                "-m",
                "tensorboard.main",
                "--logdir",
                "/workspace/crazyflow/outputs/sim2sim/tensorboard",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ]
        )
    elif args.mode == "play":
        policy = policy_path(args.policy)
        command = docker_command(args.gui) + [
            "/workspace/crazyflow/sim2sim/play.py",
            "--task",
            "baseline",
            "--num_envs",
            str(args.num_envs),
            "--seed",
            str(args.seed),
            "--checkpoint",
            container_path(policy),
            "--skip_export",
            "--max_steps",
            str(args.max_steps),
            "env.enable_difficulty_curriculum=False",
            "env.debug_vis=False",
            "env.viewer.eye=[12.0,1.6,16.0]",
            "env.viewer.lookat=[0.0,12.8,1.0]",
        ]
        if not args.gui:
            command += ["--headless"]
        if args.video:
            command += ["--video", "--video_length", str(args.max_steps)]
        run(command + extra)


if __name__ == "__main__":
    main()
