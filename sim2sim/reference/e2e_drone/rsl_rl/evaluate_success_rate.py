# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate policy success rates from RSL-RL checkpoints."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import re
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
import gymnasium as gym
from typing import Dict, List, Tuple

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate RL agent success rates with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, required=True, 
                   help="Single checkpoint file or directory containing checkpoints.")
parser.add_argument("--max_episodes_per_env", type=int, default=1, 
                   help="Maximum episodes to run per environment for evaluation.")
parser.add_argument("--output_csv", type=str, default=None, 
                   help="Output CSV file path. If not specified, saves to checkpoint directory.")
parser.add_argument("--metrics_dir", type=str, default=None,
                   help="Also save per-episode navigation metrics and a JSON summary in this directory.")
parser.add_argument("--start_epoch", type=int, default=99999999, 
                   help="Start epoch number. Only evaluate checkpoints with epoch >= this value.")
parser.add_argument("--skip", type=int, default=1, 
                   help="Skip interval for checkpoint evaluation. E.g., skip=2 evaluates every 2nd checkpoint.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.max_episodes_per_env <= 0:
    parser.error("--max_episodes_per_env must be a positive integer")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from e2e_drone import tasks
from navigation_metrics import NavigationMetrics


class SuccessRateEvaluator:
    """Evaluates policy success rates across multiple checkpoints."""
    
    def __init__(self, env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg, max_episodes_per_env: int = 2):
        self.env_cfg = env_cfg
        self.agent_cfg = agent_cfg
        self.max_episodes_per_env = max_episodes_per_env
        
        # Create environment
        self.env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(self.env.unwrapped, DirectMARLEnv):
            self.env = multi_agent_to_single_agent(self.env)
        self.env = RslRlVecEnvWrapper(self.env, clip_actions=agent_cfg.clip_actions)
        
        self.num_envs = self.env.unwrapped.num_envs
        print(f"[INFO] Evaluator initialized with {self.num_envs} environments")
        print(f"[INFO] Max episodes per environment: {max_episodes_per_env}")
        
        # Initialize tracking variables
        self.reset_evaluation_state()
        
    def reset_evaluation_state(self):
        """Reset all evaluation tracking variables."""
        # Reset environment outside of inference mode
        with torch.no_grad():
            self.env.reset()
        self.env_episode_counts = torch.zeros(self.num_envs, dtype=torch.int32, device=self.env.unwrapped.device)
        self.env_success_counts = torch.zeros(self.num_envs, dtype=torch.int32, device=self.env.unwrapped.device)
        self.env_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.unwrapped.device)
        self.total_episodes = 0
        self.total_successes = 0
        
    def load_checkpoint(self, checkpoint_path: str) -> OnPolicyRunner:
        """Load a checkpoint and return the runner."""
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        
        # Create new runner for this checkpoint
        ppo_runner = OnPolicyRunner(self.env, self.agent_cfg.to_dict(), log_dir=None, device=self.agent_cfg.device)
        ppo_runner.load(checkpoint_path)
        
        return ppo_runner
        
    def evaluate_checkpoint(self, checkpoint_path: str) -> float:
        """Evaluate a single checkpoint and return success rate."""
        # Load checkpoint first
        ppo_runner = self.load_checkpoint(checkpoint_path)
        policy = ppo_runner.get_inference_policy(device=self.env.unwrapped.device)
        
        # Reset evaluation state after loading checkpoint
        self.reset_evaluation_state()
        
        # Get initial observations
        obs, _ = self.env.get_observations()
        
        print(f"[INFO] Starting evaluation for checkpoint: {os.path.basename(checkpoint_path)}")
        start_time = time.time()
        step_count = 0
        telemetry = NavigationMetrics(self.env.unwrapped, self.max_episodes_per_env) if args_cli.metrics_dir else None
        if telemetry is not None:
            telemetry.start()
        
        # Run evaluation until all environments complete required episodes
        try:
            while not torch.all(self.env_completed):
                step_count += 1

                # Run inference
                with torch.inference_mode():
                    actions = policy(obs)

                # Step environment outside inference mode to allow state updates
                obs, _, _, extras = self.env.step(actions)
                if telemetry is not None:
                    telemetry.update(~self.env_completed)

                # Check for completed episodes
                if "episode_status" in extras:
                    self._process_episode_completions(extras["episode_status"])

                # Progress reporting every 1000 steps
                if step_count % 1000 == 0:
                    completed_envs = torch.sum(self.env_completed).item()
                    total_episodes_so_far = torch.sum(self.env_episode_counts).item()
                    print(f"[INFO] Step {step_count}: {completed_envs}/{self.num_envs} envs completed, "
                          f"{total_episodes_so_far} total episodes", flush=True)
        finally:
            if telemetry is not None:
                telemetry.close()
        
        # Calculate final success rate
        success_rate = self.total_successes / self.total_episodes if self.total_episodes > 0 else 0.0
        
        elapsed_time = time.time() - start_time
        print(f"[INFO] Evaluation completed in {elapsed_time:.2f}s")
        print(f"[INFO] Total episodes: {self.total_episodes}, Successes: {self.total_successes}")
        print(f"[INFO] Success rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
        if telemetry is not None:
            self.save_navigation_metrics(telemetry, checkpoint_path, step_count, elapsed_time)
        
        return success_rate

    def save_navigation_metrics(self, telemetry, checkpoint_path, step_count, elapsed_time):
        """Save measurements separately from the legacy epoch/success CSV."""
        output_dir = Path(args_cli.metrics_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(checkpoint_path).stem
        report = telemetry.summary()
        if report["num_episodes"] != self.total_episodes:
            raise RuntimeError("Navigation metrics and evaluator episode counts disagree")
        report.update({
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
            "task": args_cli.task,
            "seed": self.env_cfg.seed,
            "num_envs": self.num_envs,
            "episodes_per_env": self.max_episodes_per_env,
            "headless": args_cli.headless,
            "simulation_steps": step_count,
            "evaluation_wall_time_s": elapsed_time,
            "evaluator_success_rate": self.total_successes / self.total_episodes,
        })
        (output_dir / f"{stem}.summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (output_dir / f"{stem}.environment.json").write_text(
            json.dumps(self.env_cfg.to_dict(), indent=2, default=str) + "\n"
        )
        with (output_dir / f"{stem}.episodes.csv").open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(telemetry.episodes[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(telemetry.episodes)
        print(f"[INFO] Collision rate: {report['collision_rate']:.2%}", flush=True)
        print(f"[INFO] Navigation metrics saved to: {output_dir}", flush=True)
        
    def _process_episode_completions(self, episode_status: Dict):
        """Process episode completions and update tracking variables."""
        success_mask = episode_status.get("success_mask", torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.unwrapped.device))
        died_mask = episode_status.get("died_mask", torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.unwrapped.device))
        timeout_mask = episode_status.get("timeout_mask", torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.unwrapped.device))
        
        # Find environments that completed an episode
        completed_episode_mask = (success_mask | died_mask | timeout_mask) & ~self.env_completed
        completed_env_ids = torch.where(completed_episode_mask)[0]
        
        if len(completed_env_ids) > 0:
            # Update episode counts for completed environments
            self.env_episode_counts[completed_env_ids] += 1
            self.total_episodes += len(completed_env_ids)
            
            # Update success counts
            success_env_ids = completed_env_ids[success_mask[completed_env_ids]]
            if len(success_env_ids) > 0:
                self.env_success_counts[success_env_ids] += 1
                self.total_successes += len(success_env_ids)
            
            # Mark environments as completed if they've reached max episodes
            reached_max = self.env_episode_counts[completed_env_ids] >= self.max_episodes_per_env
            self.env_completed[completed_env_ids[reached_max]] = True
            
    def close(self):
        """Close the environment."""
        self.env.close()


def find_checkpoint_files(checkpoint_path: str, start_epoch: int = 99999999) -> List[Tuple[int, str]]:
    """Find checkpoint files in the given path, filtered by start epoch and sorted in descending order."""
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.is_file():
        # Single checkpoint file
        match = re.fullmatch(r'model_(\d+)\.pt', checkpoint_path.name)
        epoch = int(match.group(1)) if match else 0
        return [(epoch, str(checkpoint_path))]
    elif checkpoint_path.is_dir():
        # Directory containing checkpoints
        checkpoint_files = []
        pattern = re.compile(r'model_(\d+)\.pt')
        
        for file_path in checkpoint_path.glob("model_*.pt"):
            match = pattern.match(file_path.name)
            if match:
                epoch = int(match.group(1))
                # Filter by start epoch
                if epoch <= start_epoch:
                    checkpoint_files.append((epoch, str(file_path)))
        
        # Sort by epoch number in descending order
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        return checkpoint_files
    else:
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

def save_results_to_csv(results: List[Tuple[int, float]], output_path: str):
    """Save evaluation results to CSV file."""
    with open(output_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, lineterminator="\n")
        # Only write header if file is new
        if csvfile.tell() == 0:
            writer.writerow(['epoch', 'success_rate'])
        if results:
            epoch, success_rate = results[-1]
            writer.writerow([epoch, success_rate])
    print(f"[INFO] Results appended to: {output_path}")


def main_evaluate(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate with RSL-RL agent."""
    # override configurations with CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    if args_cli.seed is not None:
        env_cfg.seed = agent_cfg.seed = args_cli.seed
    
    return evaluate_impl(env_cfg, agent_cfg)


def evaluate_impl(env_cfg, agent_cfg):
    """Implementation of the evaluation functionality."""
    # Find checkpoint files
    try:
        checkpoint_files = find_checkpoint_files(args_cli.checkpoint_path, args_cli.start_epoch)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    
    if not checkpoint_files:
        print(f"[ERROR] No checkpoint files found in: {args_cli.checkpoint_path} with epoch >= {args_cli.start_epoch}")
        return
    
    print(f"[INFO] Found {len(checkpoint_files)} checkpoint(s) to evaluate (epochs >= {args_cli.start_epoch})")
    
    # Apply skip interval to checkpoint files
    if args_cli.skip > 1:
        checkpoint_files = checkpoint_files[::args_cli.skip]
        print(f"[INFO] Applied skip interval of {args_cli.skip}, evaluating {len(checkpoint_files)} checkpoint(s)")
    
    print(f"[INFO] Checkpoints will be evaluated in descending epoch order: {[epoch for epoch, _ in checkpoint_files]}")
    
    # Determine output CSV path
    if args_cli.output_csv:
        output_csv = args_cli.output_csv
    else:
        if Path(args_cli.checkpoint_path).is_dir():
            output_csv = os.path.join(args_cli.checkpoint_path, "metrics.csv")
        else:
            output_csv = os.path.join(os.path.dirname(args_cli.checkpoint_path), "metrics.csv")
    
    # Create evaluator
    evaluator = SuccessRateEvaluator(env_cfg, agent_cfg, args_cli.max_episodes_per_env)
    
    # Evaluate each checkpoint
    results = []
    try:
        for i, (epoch, checkpoint_file) in enumerate(checkpoint_files):
            print(f"\n[INFO] ========== Evaluating checkpoint {i+1}/{len(checkpoint_files)} ==========")
            print(f"[INFO] Checkpoint: {os.path.basename(checkpoint_file)} (Epoch: {epoch})")
            
            success_rate = evaluator.evaluate_checkpoint(checkpoint_file)
            results.append((epoch, success_rate))
            
            # Save intermediate results
            save_results_to_csv(results, output_csv)
            
            print(f"[INFO] Epoch {epoch}: Success rate = {success_rate:.4f} ({success_rate*100:.2f}%)")
            
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user")
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        raise
    finally:
        # Clean up
        evaluator.close()
        
        # Print final results
        if results:
            print(f"\n[INFO] ========== Final Results ==========")
            for epoch, success_rate in results:
                print(f"[INFO] Epoch {epoch}: {success_rate:.4f} ({success_rate*100:.2f}%)")


def main():
    """Main entry point - uses Hydra configuration system."""
    # Use Hydra configuration system
    hydra_main = hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")(main_evaluate)
    return hydra_main()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
