#!/usr/bin/env python3
"""Measure real GPU PPO batches, stop before 10 GiB, retain every trial's evidence."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def memory() -> float:
    """Return summed CUDA compute-process memory in MiB."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return sum(float(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit())


def main() -> None:
    """Run increasing full PPO batches under the GPU memory watchdog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--envs",
        nargs="+",
        type=int,
        default=[1024, 4096, 8192, 10240, 11264, 12288, 14336, 16384, 32768],
    )
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=120)
    parser.add_argument(
        "--config",
        default="config/training.yaml",
        help="training YAML inside this repository, resolved from its root",
    )
    parser.add_argument("--output", default="outputs/tof-capacity")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = (root / Path(args.config).expanduser()).resolve()
    try:
        config_path = config.relative_to(root).as_posix()
    except ValueError:
        parser.error("--config must resolve to a file inside this repository")
    if not config.is_file():
        parser.error(f"--config file does not exist: {config_path}")
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    image = os.environ.get("CRAZYFLOW_IMAGE", "crazyflow:dani-crazyrl")
    baseline = memory()
    if baseline > 512:
        raise SystemExit("Stop other CUDA compute workloads before the capacity test.")
    for n in args.envs:
        if hashlib.sha256(config.read_bytes()).hexdigest() != config_sha256:
            raise SystemExit("Training configuration changed during the capacity test.")
        trial = output / str(n)
        trial.mkdir(exist_ok=True)
        name = f"crazyflow-tof-capacity-{os.getpid()}-{n}"
        container_output = "/outputs/" + str(trial.relative_to(root / "outputs"))
        command = [
            "docker",
            "run",
            "--rm",
            "--init",
            "--name",
            name,
            "--gpus",
            "all",
            "--shm-size=2g",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "JAX_PLATFORMS=cuda",
            "-e",
            "XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "-e",
            "XLA_PYTHON_CLIENT_MEM_FRACTION=0.55",
            "-e",
            "MUJOCO_GL=egl",
            "-e",
            "JAX_COMPILATION_CACHE_DIR=/outputs/.cache/jax-capacity",
            "-v",
            f"{root}:/opt/crazyflow:ro",
            "-v",
            f"{root}/outputs:/outputs",
            "-w",
            "/opt/crazyflow",
            image,
            "python",
            "-m",
            "crazyflow.tof.train",
            "--config",
            config_path,
            "--output",
            container_output,
            "--num-envs",
            str(n),
            "--iterations",
            str(args.iterations),
            "--benchmark",
            "--num-steps",
            str(args.num_steps),
        ]
        peak, killed = baseline, False
        start = time.monotonic()
        with (trial / "console.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                peak = max(peak, memory())
                if peak - baseline > 9728 or time.monotonic() - start > 1200:
                    killed = True
                    subprocess.run(["docker", "stop", "-t", "1", name], capture_output=True)
                    break
                time.sleep(0.25)
            status = process.wait()
        record = {
            "num_envs": n,
            "exit_code": status,
            "watchdog_stopped": killed,
            "sampled_compute_peak_mib": peak - baseline,
            "poll_seconds": 0.25,
            "wall_seconds": time.monotonic() - start,
            "command": command,
            "config_path": config_path,
            "config_sha256": config_sha256,
        }
        if status == 0:
            result = json.loads((trial / "result.json").read_text())
            record.update(
                {
                    k: result[k]
                    for k in [
                        "steady_update_seconds",
                        "steady_samples_per_second",
                        "first_update_seconds",
                        "source_sha256",
                    ]
                }
            )
            training = result["training_config"]
            record.update(
                ppo_variant=training.get("ppo_variant", "legacy"),
                num_minibatches=training.get("num_minibatches", 32),
                update_epochs=training.get("update_epochs", training.get("epochs", 5)),
                num_steps=training["num_steps"],
            )
        records.append(record)
        (output / "capacity.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if status != 0:
            break
    passing = [x for x in records if x["exit_code"] == 0]
    if not passing:
        raise SystemExit("No full PPO batch passed; inspect capacity logs.")
    print(
        json.dumps(
            {
                "largest_passing_envs": max(x["num_envs"] for x in passing),
                "fastest_envs": max(passing, key=lambda x: x["steady_samples_per_second"])[
                    "num_envs"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
