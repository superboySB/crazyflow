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
original_record_video = play.gym.wrappers.RecordVideo


def record_video(env: Any, **kwargs: Any) -> Any:
    """Keep native recordings in the ignored output tree beside other run data."""
    checkpoint_name = Path(play.args_cli.checkpoint).stem
    kwargs["video_folder"] = str(ROOT.parent / "outputs/sim2sim/videos" / checkpoint_name)
    return original_record_video(env, **kwargs)


def load(self: Any, path: str, *args: Any, **kwargs: Any) -> Any:
    """Load the requested actor while preserving the native playback runner."""
    if Path(path).suffix != ".npz":
        from torch_policy import load_into_runner

        if load_into_runner(self, path):
            return {}
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
play.gym.wrappers.RecordVideo = record_video
if __name__ == "__main__":
    try:
        play.main()
    finally:
        play.simulation_app.close()
