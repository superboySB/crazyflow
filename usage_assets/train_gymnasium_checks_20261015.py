"""Opt-in CUDA regression: exact PPO continuation, autoreset masking, and legacy checkpoints.

Run inside the research image with ``python usage_assets/train_gymnasium_checks_20261015.py``.
This uses only eight training worlds and two PPO updates. It tests accounting/recovery rather
than learning performance; it is intentionally separate from the CPU-only upstream test suite.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from train_gymnasium import Config, checkpoint_last_path, evaluate, load_checkpoint, train


def main() -> None:
    """Check identical uninterrupted/recovered updates and optional old-checkpoint evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-checkpoint", type=Path)
    parser.add_argument("--teacher-model", choices=("legacy", "calibrated"), default="legacy")
    parser.add_argument("--env-backend", choices=("python", "fused"), default="python")
    parser.add_argument(
        "--task", choices=("reach_pos", "reach_vel", "landing"), default="reach_vel"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This opt-in verification requires the research CUDA image")
    if args.env_backend == "fused":
        import fast_gymnasium_20261015 as fast
        import train_gymnasium as trainer

        trainer.make_env = fast.make_fused_env
    report = {
        "purpose": "regression verification; not a trained policy demonstration",
        "teacher_model": args.teacher_model,
        "env_backend": args.env_backend,
        "task": args.task,
    }
    with tempfile.TemporaryDirectory(prefix="crazyflow-trainer-check-") as tmp:
        config = Config(
            env_id={
                "reach_pos": "DroneReachPos-v0",
                "reach_vel": "DroneReachVel-v0",
                "landing": "DroneLanding-v0",
            }[args.task],
            seed=123,
            total_timesteps=256,
            num_envs=8,
            num_steps=16,
            num_minibatches=2,
            update_epochs=1,
            anneal_lr=False,
            eval_envs=4,
            episode_time=0.1,
            eval_interval=1,
            validation_seed=20123,
            position_bc_steps=2 if args.task in ("reach_pos", "landing") else 0,
            teacher_model=args.teacher_model,
        )
        continuous, recovered = Path(tmp) / "continuous.pt", Path(tmp) / "recovered.pt"
        train(config, continuous)
        train(replace(config, total_timesteps=128), recovered)
        train(config, recovered, resume=checkpoint_last_path(recovered))
        a = torch.load(checkpoint_last_path(continuous), map_location="cpu", weights_only=False)
        b = torch.load(checkpoint_last_path(recovered), map_location="cpu", weights_only=False)
        for key, value in a["agent"].items():
            assert torch.equal(value, b["agent"][key]), key
        for left, right in zip(
            a["trainer_state"]["environment"]["sim_leaves"],
            b["trainer_state"]["environment"]["sim_leaves"],
            strict=True,
        ):
            np.testing.assert_array_equal(left, right)
        assert a["history"] == b["history"]
        other_teacher = "legacy" if args.teacher_model == "calibrated" else "calibrated"
        try:
            train(
                replace(config, teacher_model=other_teacher),
                recovered,
                resume=checkpoint_last_path(recovered),
            )
        except ValueError as exc:
            assert "teacher" in str(exc).lower(), str(exc)
        else:
            raise AssertionError("Resume mixed legacy and calibrated teacher recipes")
        report["teacher_recipe_resume_mismatch_rejected"] = True
        state = a["trainer_state"]
        assert state["valid_step"] < state["global_step"]
        report.update(
            exact_policy_and_simulator_recovery=True,
            vector_step_slots=state["global_step"],
            valid_transitions=state["valid_step"],
            ignored_reset_slots=state["global_step"] - state["valid_step"],
        )
    if args.legacy_checkpoint is not None:
        agent, config, keys, _ = load_checkpoint(args.legacy_checkpoint)
        training_seed = config.seed
        config.eval_envs = 8
        metrics, behavior = evaluate(agent, config, keys, eval_seed=30123)
        assert config.seed == training_seed
        assert behavior["initial_pos"].shape == (8, 3)
        if config.env_id == "DroneReachVel-v0":
            assert behavior["initial_target_vel"].shape == (8, 3)
        assert np.all(behavior["per_env_episode_steps"] <= config.episode_time * config.freq)
        report.update(legacy_checkpoint_readable=True, legacy_metrics=metrics)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
