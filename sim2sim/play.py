#!/usr/bin/env python3
"""Use the native CrazyRL playback loop for .pt or JAX-exported .npz policies."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "reference/e2e_drone/rsl_rl"
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("crazyrl_frozen_play", SCRIPTS / "play.py")
play = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = play
spec.loader.exec_module(play)
original_load = play.OnPolicyRunner.load
original_policy = play.OnPolicyRunner.get_inference_policy


def load(self: Any, path: str, *args: Any, **kwargs: Any) -> Any:
    """Load the requested actor while preserving the native playback runner."""
    if Path(path).suffix != ".npz":
        return original_load(self, path, *args, **kwargs)
    if not play.args_cli.skip_export:
        raise ValueError("NPZ playback requires --skip_export; actor is already exported")
    from npz_policy import NpzPolicy

    self._crazyflow_npz_policy = NpzPolicy(path, self.device)


def get_policy(self: Any, device: str | None = None) -> Any:
    """Select NPZ inference or the original RSL policy callable."""
    if hasattr(self, "_crazyflow_npz_policy"):
        return self._crazyflow_npz_policy
    return original_policy(self, device=device)


play.OnPolicyRunner.load = load
play.OnPolicyRunner.get_inference_policy = get_policy
if __name__ == "__main__":
    try:
        play.main()
    finally:
        play.simulation_app.close()
