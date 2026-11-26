"""Balloon entry points do not require the optional MJX/Warp backend."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module",
    ("crazyflow.balloon.train", "crazyflow.balloon.play", "crazyflow.balloon.mujoco_rollout"),
)
def test_balloon_import_does_not_load_mjx_or_warn_about_warp(module: str):
    """Check fresh interpreters so another test cannot hide an eager import."""
    script = (
        "import importlib, sys\n"
        f"importlib.import_module({module!r})\n"
        "assert 'mujoco' in sys.modules\n"
        "assert not any(name == 'mujoco.mjx' or name.startswith('mujoco.mjx.') "
        "for name in sys.modules), 'Unused MJX backend was imported'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert "Failed to import warp:" not in output
    assert "Failed to import mujoco_warp:" not in output
