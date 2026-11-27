#!/usr/bin/env python3
"""Run one named training container and record/limit sampled CUDA process memory."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def memory() -> float:
    """Read CUDA compute memory in MiB, excluding graphics-only desktop processes."""
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=used_gpu_memory", "--format=csv,noheader,nounits"],
        text=True,
    )
    return sum(float(x) for x in output.splitlines() if x.strip().isdigit())


def main() -> None:
    """Terminate the named Docker container if its allocation approaches 10 GiB."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    baseline = memory()
    if baseline > 512:
        raise SystemExit("End other CUDA compute jobs before starting guarded training.")
    process = subprocess.Popen(command)
    peak = baseline
    stopped = False
    try:
        while process.poll() is None:
            peak = max(peak, memory())
            if peak - baseline > 9728:
                stopped = True
                subprocess.run(["docker", "stop", "-t", "2", args.name], check=False)
                break
            time.sleep(0.25)
        status = process.wait()
    except BaseException:
        subprocess.run(["docker", "stop", "-t", "2", args.name], check=False)
        process.wait()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "sampled_compute_peak_mib": peak - baseline,
                    "memory_limit_mib": 9728,
                    "poll_seconds": 0.25,
                    "watchdog_stopped": stopped,
                    "command": command,
                },
                indent=2,
            )
            + "\n"
        )
    raise SystemExit(status if not stopped else 137)


if __name__ == "__main__":
    main()
