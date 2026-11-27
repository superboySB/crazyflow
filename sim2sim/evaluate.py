#!/usr/bin/env python3
"""Run the frozen CrazyRL evaluator with baseline or JAX-exported actors.

All arguments are those of reference/e2e_drone/rsl_rl/evaluate_success_rate.py.
PT exports use native RSL inference; legacy NPZ exports use NpzPolicy.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "reference/e2e_drone/rsl_rl"
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "crazyrl_frozen_evaluator", SCRIPTS / "evaluate_success_rate.py"
)
evaluator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluator
spec.loader.exec_module(evaluator)

original_load = evaluator.SuccessRateEvaluator.load_checkpoint
original_reset = evaluator.SuccessRateEvaluator.reset_evaluation_state


def load_checkpoint(self: Any, checkpoint_path: str) -> Any:
    """Load marked PT exports or legacy NPZ, preserving baseline RSL loading."""
    from torch_policy import load_into_runner, read_crazyflow_checkpoint

    is_npz = Path(checkpoint_path).suffix == ".npz"
    exported = None if is_npz else read_crazyflow_checkpoint(checkpoint_path)
    if not is_npz and exported is None:
        return original_load(self, checkpoint_path)
    # Match the original baseline's network initialization and Torch RNG draws
    # before the evaluation reset. This runner is never trained or stepped.
    runner = evaluator.OnPolicyRunner(
        self.env, self.agent_cfg.to_dict(), log_dir=None, device=self.agent_cfg.device
    )
    if exported is not None:
        load_into_runner(runner, checkpoint_path, checkpoint=exported)
        print(f"[INFO] Loaded JAX-exported RSL PT actor: {checkpoint_path}", flush=True)
        return runner
    from npz_policy import NpzPolicy

    policy = NpzPolicy(checkpoint_path, self.env.unwrapped.device)
    runner.get_inference_policy = lambda device=None: policy
    print(f"[INFO] Loaded JAX NPZ actor: {checkpoint_path}", flush=True)
    return runner


def reset_evaluation_state(self: Any) -> None:
    """Preserve the original reset and record its initial scene for comparison."""
    original_reset(self)
    if evaluator.args_cli.metrics_dir:
        env = self.env.unwrapped
        scene = {
            "seed": int(self.env_cfg.seed),
            "num_envs": self.num_envs,
            "root_state_w": env._robot.data.root_state_w.detach().cpu().tolist(),
            "goal_positions_w": env._desired_pos_w.detach().cpu().tolist(),
            "desired_speed": env._desired_speed_init.detach().cpu().tolist(),
            "map_id": int(env._active_map_id),
            "controller_mass": env._robot_mass.detach().cpu().tolist(),
            "controller_inertia": env._robot_inertia.detach().cpu().tolist(),
            "wind_acceleration": env._wind_gen.wind_acc.detach().cpu().tolist(),
        }
        if env._thrust_uncertainty is not None:
            uncertainty = env._thrust_uncertainty
            scene["thrust_effectiveness"] = uncertainty.thrust_effectiveness.detach().cpu().tolist()
            scene["thrust_degradation_rate"] = uncertainty.degradation_rate.detach().cpu().tolist()
        # Occupied points describe the generated map independent of USD naming.
        if env.occ_kdtree is not None:
            scene["map_occupied_points_sha256"] = hashlib.sha256(
                env.occ_kdtree.data.tobytes()
            ).hexdigest()
        canonical = json.dumps(scene, sort_keys=True, separators=(",", ":"))
        scene["scene_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        out = Path(evaluator.args_cli.metrics_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "initial_scene.json").write_text(json.dumps(scene, indent=2) + "\n")


evaluator.SuccessRateEvaluator.load_checkpoint = load_checkpoint
evaluator.SuccessRateEvaluator.reset_evaluation_state = reset_evaluation_state
if __name__ == "__main__":
    try:
        evaluator.main()
    finally:
        evaluator.simulation_app.close()
