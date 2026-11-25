"""Exercise launcher decisions without requiring a host GPU or Docker daemon."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

FAKE_DOCKER = r"""#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
    log.write(json.dumps(args) + "\n")
if args[0] in ("info", "image", "build"):
    sys.exit(0)
assert args[0] == "run", args
if "--gpus" in args and os.environ.get("FAKE_GPU", "0") == "0":
    print("No GPU runtime", file=sys.stderr)
    sys.exit(1)
if "-m" not in args:
    print("[CudaDevice(id=0)]" if "--gpus" in args else "[CpuDevice(id=0)]")
    sys.exit(0)
module = args[args.index("-m") + 1]
mount = args[args.index("-v") + 1].rsplit(":/outputs", 1)[0]
def value(flag):
    return args[args.index(flag) + 1]
def host(path):
    assert path.startswith("/outputs/"), path
    return pathlib.Path(mount, path[len("/outputs/"):])
if module.endswith("train"):
    output = host(value("--output"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "policy.npz").write_bytes(b"policy")
elif module.endswith("evaluate"):
    output = host(value("--output"))
    output.mkdir(parents=True, exist_ok=True)
    if os.environ.get("FAKE_INCOMPLETE") != "1":
        for name in ("metrics.json", "nominal.npz", "noisy.npz", "perturbed.npz"):
            (output / name).write_bytes(b"{}")
if os.environ.get("FAKE_FAIL_MODULE") == module:
    sys.exit(7)
"""


class LauncherTests(unittest.TestCase):
    """Check backend selection and preservation of successful experiment pointers."""

    def setUp(self) -> None:
        """Copy launchers into an isolated repository and provide a Docker test double."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "project with spaces"
        self.root.mkdir()
        source = Path(__file__).resolve().parents[1]
        shutil.copy2(source / "dani", self.root / "dani")
        shutil.copytree(source / "scripts", self.root / "scripts")
        shutil.copytree(source / "config", self.root / "config")
        self.example_policy = self.root / "examples/balloon/policy.npz"
        self.example_policy.parent.mkdir(parents=True)
        self.example_policy.write_bytes(b"bundled-example-policy")
        self.bin = Path(self.tmp.name) / "bin"
        self.bin.mkdir()
        fake = self.bin / "docker"
        fake.write_text(FAKE_DOCKER)
        fake.chmod(0o755)
        self.log = Path(self.tmp.name) / "docker.jsonl"
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "FAKE_DOCKER_LOG": str(self.log),
            "FAKE_GPU": "0",
        }
        self.env.pop("CF_DEVICE_REQUEST", None)
        self.env.pop("CRAZYFLOW_IMAGE", None)

    def run_dani(self, *args: str, ok: bool = True) -> subprocess.CompletedProcess:
        """Invoke the real Bash wrapper from outside the repository."""
        result = subprocess.run(
            [str(self.root / "dani"), *args],
            cwd=self.tmp.name,
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def calls(self) -> list[list[str]]:
        """Read the Docker invocations observed by the test double."""
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_help_does_not_require_docker(self) -> None:
        """Display help before contacting Docker."""
        for command in ("setup", "train", "evaluate", "play", "identify"):
            self.run_dani(command, "--help")
        self.run_dani("identify", "inertia", "--help")
        self.assertFalse(self.log.exists())

    def test_auto_cpu_and_default_config_snapshot(self) -> None:
        """Fall back to CPU and preserve all four editable configurations."""
        result = self.run_dani("train", "--iterations", "1", "--output", "outputs/cpu")
        self.assertIn("自动使用 CPU", result.stdout)
        call = next(c for c in self.calls() if "crazyflow.balloon.train" in c)
        self.assertNotIn("--gpus", call)
        self.assertNotIn("--batch-size", call)  # Device-specific defaults live in YAML.
        self.assertEqual(call[call.index("--device") + 1], "cpu")
        snapshot = self.root / "outputs/cpu/config/camera.json"
        self.assertEqual(json.loads(snapshot.read_text())["hfov_deg"], 120)
        for name in ("training.yaml", "camera.json", "vehicle.toml", "controller.toml"):
            self.assertTrue((self.root / "outputs/cpu/config" / name).is_file())
        pointer = self.root / "outputs/.latest-policy"
        self.assertEqual(pointer.read_text(), "cpu\n")

    def test_auto_gpu_and_forced_cpu(self) -> None:
        """Prefer working GPU computation while respecting forced CPU."""
        self.env["FAKE_GPU"] = "1"
        self.run_dani("train", "--output", "outputs/gpu")
        call = next(c for c in self.calls() if "crazyflow.balloon.train" in c)
        self.assertIn("--gpus", call)
        self.assertIn("--settings", call)
        self.run_dani("train", "--device", "cpu", "--output", "outputs/cpu")
        call = [c for c in self.calls() if "crazyflow.balloon.train" in c][-1]
        self.assertNotIn("--gpus", call)

    def test_forced_gpu_failure_never_starts_training(self) -> None:
        """Fail explicit GPU selection before starting a training job."""
        self.run_dani("train", "--device", "gpu", ok=False)
        self.assertFalse(any("crazyflow.balloon.train" in c for c in self.calls()))
        self.assertFalse((self.root / "outputs/.latest-policy").exists())

    def test_failed_or_smoke_training_preserves_latest(self) -> None:
        """Preserve the previous successful experiment after failure or smoke."""
        self.run_dani("train", "--output", "outputs/good")
        pointer = self.root / "outputs/.latest-policy"
        self.env["FAKE_FAIL_MODULE"] = "crazyflow.balloon.train"
        result = self.run_dani("train", "--output", "outputs/failed", ok=False)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(pointer.read_text(), "good\n")
        self.env.pop("FAKE_FAIL_MODULE")
        self.run_dani("train", "--smoke", "--output", "outputs/smoke")
        self.assertEqual(pointer.read_text(), "good\n")

    def test_output_overwrite_and_symlink_escape_are_rejected(self) -> None:
        """Reject overwrites and output paths escaping through symlinks."""
        self.run_dani("train", "--output", "outputs/good")
        count = len(self.calls())
        self.run_dani("train", "--output", "outputs/good", ok=False)
        self.assertEqual(len(self.calls()), count)
        (self.root / "outputs/escape").symlink_to(self.tmp.name, target_is_directory=True)
        self.run_dani("train", "--output", "outputs/escape/outside", ok=False)
        self.assertEqual(len(self.calls()), count)

    def test_evaluation_chain_and_incomplete_results_preserve_latest(self) -> None:
        """Evaluate the selected policy and reject incomplete successful exits."""
        self.run_dani("train", "--output", "outputs/good")
        self.run_dani("evaluate", "--episodes", "2", "--output", "outputs/eval")
        call = next(c for c in self.calls() if "crazyflow.balloon.evaluate" in c)
        self.assertEqual(call[call.index("--checkpoint") + 1], "/outputs/good/policy.npz")
        self.assertEqual(call[call.index("--plant") + 1], "mujoco")
        pointer = self.root / "outputs/.latest-evaluation"
        self.assertEqual(pointer.read_text(), "eval\n")
        self.env["FAKE_INCOMPLETE"] = "1"
        self.run_dani("evaluate", "--output", "outputs/incomplete", ok=False)
        self.assertEqual(pointer.read_text(), "eval\n")

    def test_abbreviations_and_raw_forwarding_cannot_override_routing(self) -> None:
        """Reject argument forwarding that could override output or device routing."""
        for command in ("evaluate", "play"):
            for argument in ("--out", "--", "--dev"):
                self.run_dani(command, argument, "/tmp/outside", ok=False)
        self.assertFalse(self.log.exists())

    def test_headless_play_uses_latest_evaluation(self) -> None:
        """Open the successful evaluation using the headless GUI self-test."""
        self.run_dani("train", "--output", "outputs/good")
        self.run_dani("evaluate", "--output", "outputs/eval")
        self.run_dani("play", "--headless")
        call = next(c for c in self.calls() if "crazyflow.balloon.play" in c)
        self.assertIn("xvfb-run", call)
        self.assertIn("--self-test", call)
        self.assertIn("--init", call)
        self.assertEqual(call[call.index("--evaluation") + 1], "/outputs/eval")

    def test_example_evaluation_works_without_training_and_mounts_only_outputs(self) -> None:
        """Stage the bundled checkpoint without inventing a latest trained policy."""
        self.run_dani("evaluate", "--example", "--device", "cpu", "--episodes", "2", "--no-video")
        call = next(c for c in self.calls() if "crazyflow.balloon.evaluate" in c)
        checkpoint = call[call.index("--checkpoint") + 1]
        self.assertTrue(checkpoint.startswith("/outputs/example-policy-"))
        staged = self.root / "outputs" / checkpoint.removeprefix("/outputs/")
        self.assertEqual(staged.read_bytes(), self.example_policy.read_bytes())
        self.assertEqual(call[call.index("--plant") + 1], "mujoco")
        mounts = [call[i + 1] for i, value in enumerate(call) if value == "-v"]
        self.assertEqual(mounts, [f"{self.root / 'outputs'}:/outputs"])
        self.assertFalse((self.root / "outputs/.latest-policy").exists())

    def test_example_play_uses_fresh_copies_and_preserves_trained_policy(self) -> None:
        """Each example flight is isolated from the user's successful training."""
        self.run_dani("train", "--device", "cpu", "--output", "outputs/user-policy")
        pointer = self.root / "outputs/.latest-policy"
        original_pointer = pointer.read_text()
        for _ in range(2):
            self.run_dani("play", "--example", "--headless", "--device", "cpu")
        calls = [c for c in self.calls() if "crazyflow.balloon.play" in c]
        checkpoints = [c[c.index("--checkpoint") + 1] for c in calls]
        self.assertEqual(len(set(checkpoints)), 2)
        for call, checkpoint in zip(calls, checkpoints, strict=True):
            self.assertTrue(checkpoint.startswith("/outputs/example-policy-"))
            staged = self.root / "outputs" / checkpoint.removeprefix("/outputs/")
            self.assertEqual(staged.read_bytes(), b"bundled-example-policy")
            self.assertEqual(call[call.index("--plant") + 1], "mujoco")
            self.assertNotIn("--evaluation", call)
            self.assertIn("xvfb-run", call)
        self.assertEqual(pointer.read_text(), original_pointer)
        self.assertEqual(self.example_policy.read_bytes(), b"bundled-example-policy")

    def test_example_sources_are_mutually_exclusive_before_docker(self) -> None:
        """Reject ambiguous source selection before staging files or starting Docker."""
        for command, source in (
            ("evaluate", "--checkpoint"),
            ("play", "--checkpoint"),
            ("play", "--evaluation"),
        ):
            with self.subTest(command=command, source=source):
                self.run_dani(command, "--example", source, "outputs/other", ok=False)
                self.run_dani(command, "--example", f"{source}=", ok=False)
        self.assertFalse(self.log.exists())
        self.assertFalse((self.root / "outputs").exists())

    def test_missing_example_never_falls_back_to_the_latest_policy(self) -> None:
        """A missing bundled file must not silently run the user's own checkpoint."""
        self.run_dani("train", "--device", "cpu", "--output", "outputs/user-policy")
        count = len(self.calls())
        self.example_policy.unlink()
        for command in ("evaluate", "play"):
            result = self.run_dani(command, "--example", ok=False)
            self.assertIn("示例策略", result.stderr)
        self.assertEqual(len(self.calls()), count)
        self.assertEqual((self.root / "outputs/.latest-policy").read_text(), "user-policy\n")


if __name__ == "__main__":
    unittest.main()
