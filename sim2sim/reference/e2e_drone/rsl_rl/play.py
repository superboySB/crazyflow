# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL with comprehensive features."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL with config overrides and metrics.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--max_steps", type=int, default=None, help="Stop inference after this many steps.")
parser.add_argument("--skip_export", action="store_true", help="Skip exporting the policy to JIT and ONNX.")
# Metrics options
parser.add_argument("--show_metrics", action="store_true", default=True, help="Display success rate and other metrics during inference.")
parser.add_argument("--metrics_interval", type=int, default=300, help="Interval (in steps) to display metrics.")
parser.add_argument("--metrics_only", action="store_true", default=False, help="Only display metrics without visual rendering.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.max_steps is not None and args_cli.max_steps <= 0:
    parser.error("--max_steps must be a positive integer")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Override headless mode if metrics_only is True
if args_cli.metrics_only:
    args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np
from collections import defaultdict, deque

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from e2e_drone import tasks
# PLACEHOLDER: Extension template (do not remove this comment)


class MetricsTracker:
    """Track and display environment metrics during inference."""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.step_count = 0
        self.last_display_time = time.time()

    def update(self, info_dict):
        """Update metrics from environment info."""
        if "log" in info_dict:
            for key, value in info_dict["log"].items():
                if isinstance(value, (int, float)):
                    self.metrics_history[key].append(value)
        self.step_count += 1

    def get_latest_metrics(self):
        """Get the latest available metrics."""
        latest_metrics = {}
        for key, values in self.metrics_history.items():
            if len(values) > 0:
                latest_metrics[key] = values[-1]
        return latest_metrics

    def get_averaged_metrics(self, window=None):
        """Get averaged metrics over a window."""
        if window is None:
            window = min(self.window_size, 50)  # Default to smaller window for more responsive averages

        averaged_metrics = {}
        for key, values in self.metrics_history.items():
            if len(values) > 0:
                # Take the last 'window' values
                recent_values = list(values)[-window:]
                averaged_metrics[key] = np.mean(recent_values)
        return averaged_metrics

    def display_metrics(self, force=False):
        """Display formatted metrics."""
        current_time = time.time()
        if not force and current_time - self.last_display_time < 2.0:  # Display every 2 seconds
            return

        self.last_display_time = current_time
        latest = self.get_latest_metrics()
        averaged = self.get_averaged_metrics()

        if not latest:
            return

        print(f"\n{'='*80}")
        print(f"INFERENCE METRICS (Step: {self.step_count})")
        print(f"{'='*80}")

        # Success rate metrics
        success_metrics = {k: v for k, v in latest.items() if 'success' in k.lower() or 'Success' in k}
        if success_metrics:
            print("📊 SUCCESS METRICS:")
            for key, value in success_metrics.items():
                avg_value = averaged.get(key, value)
                if 'rate' in key.lower():
                    print(f"  {key:35}: {value:8.2f}% (avg: {avg_value:6.2f}%)")
                else:
                    print(f"  {key:35}: {value:8.0f} (avg: {avg_value:6.1f})")

        # Episode termination statistics
        termination_metrics = {k: v for k, v in latest.items() if 'Episode_Termination' in k}
        if termination_metrics:
            print("\n📈 EPISODE OUTCOMES:")
            for key, value in termination_metrics.items():
                metric_name = key.split('/')[-1]
                avg_value = averaged.get(key, value)
                print(f"  {metric_name:35}: {value:8.2f}% (avg: {avg_value:6.2f}%)")

        # Failure reason analysis
        failure_metrics = {k: v for k, v in latest.items() if 'Failure_Reasons' in k or 'Died' in k}
        if failure_metrics:
            print("\n💥 FAILURE ANALYSIS:")
            for key, value in failure_metrics.items():
                reason = key.split('/')[-1]
                avg_value = averaged.get(key, value)
                print(f"  {reason:35}: {value:8.2f}% (avg: {avg_value:6.2f}%)")

        # Task queue metrics
        goal_metrics = {k: v for k, v in latest.items() if 'goal' in k.lower() and 'Metrics' in k}
        if goal_metrics:
            print("\n🎯 TASK QUEUE PROGRESS:")
            for key, value in goal_metrics.items():
                metric_name = key.split('/')[-1]
                avg_value = averaged.get(key, value)
                if 'progress_percent' in key.lower():
                    print(f"  {metric_name:35}: {value:8.2f}% (avg: {avg_value:6.2f}%)")
                elif 'goals_remaining' in key.lower():
                    print(f"  {metric_name:35}: {value:8.2f} (avg: {avg_value:6.2f})")

        # Performance metrics
        perf_metrics = {k: v for k, v in latest.items() if 'final_distance' in k or 'episodes_completed' in k or 'episodes_succeeded' in k}
        if perf_metrics:
            print("\n📊 PERFORMANCE:")
            for key, value in perf_metrics.items():
                metric_name = key.split('/')[-1]
                avg_value = averaged.get(key, value)
                if 'distance' in key.lower():
                    print(f"  {metric_name:35}: {value:8.3f}m (avg: {avg_value:6.3f}m)")
                else:
                    print(f"  {metric_name:35}: {value:8.0f} (avg: {avg_value:6.1f})")

        # Thrust effectiveness (if available)
        thrust_metrics = {k: v for k, v in latest.items() if 'thrust_effectiveness' in k}
        if thrust_metrics:
            print("\n🚁 THRUST EFFECTIVENESS:")
            for key, value in thrust_metrics.items():
                metric_name = key.split('/')[-1]
                avg_value = averaged.get(key, value)
                print(f"  {metric_name:35}: {value:8.3f} (avg: {avg_value:6.3f})")

        print(f"{'='*80}\n")


def main_play(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # override configurations with CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    if args_cli.seed is not None:
        env_cfg.seed = agent_cfg.seed = args_cli.seed

    return play_impl(env_cfg, agent_cfg)


def play_impl(env_cfg, agent_cfg):
    """Implementation of the play functionality."""
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    render_mode = None
    if args_cli.video:
        render_mode = "rgb_array"
    elif not args_cli.metrics_only and not args_cli.headless:
        render_mode = "human"

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    if not args_cli.skip_export:
        # extract the neural network module
        try:
            # version 2.3 onwards
            policy_nn = ppo_runner.alg.policy
        except AttributeError:
            # version 2.2 and below
            policy_nn = ppo_runner.alg.actor_critic

        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

    dt = env.unwrapped.step_dt

    # Initialize metrics tracker
    metrics_tracker = None
    if args_cli.show_metrics or args_cli.metrics_only:
        metrics_tracker = MetricsTracker(window_size=100)
        print(f"[INFO] Metrics tracking enabled - display interval: {args_cli.metrics_interval} steps")

    print(f"[INFO] Hydra configuration mode - overrides applied successfully!")

    print(f"\n[INFO] Starting inference...")
    print(f"[INFO] Environment: {args_cli.task}")
    print(f"[INFO] Number of environments: {env.unwrapped.num_envs}")
    if args_cli.metrics_only:
        print(f"[INFO] Running in headless mode (metrics only)")
    print(f"[INFO] Press Ctrl+C to stop...")

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    start_time = time.time()

    # simulate environment
    try:
        while simulation_app.is_running():
            step_start_time = time.time()

            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, rewards, dones, info = env.step(actions)

            # Update metrics tracker
            if metrics_tracker and info:
                metrics_tracker.update(info)

            # Display metrics at specified intervals
            if metrics_tracker and timestep % args_cli.metrics_interval == 0 and timestep > 0:
                metrics_tracker.display_metrics(force=True)

            timestep += 1
            # Exit after recording one video or reaching the requested step limit.
            if args_cli.video and timestep == args_cli.video_length:
                break
            if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
                break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - step_start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n[INFO] Inference interrupted by user")

    # Display final metrics summary
    total_time = time.time() - start_time
    print(f"\n[INFO] Inference completed")
    print(f"[INFO] Total time: {total_time:.2f}s")
    print(f"[INFO] Total steps: {timestep}")
    print(f"[INFO] Average FPS: {timestep/total_time:.2f}")

    if metrics_tracker:
        print(f"\n[INFO] FINAL METRICS SUMMARY:")
        metrics_tracker.display_metrics(force=True)

    # close the simulator
    env.close()


def main():
    """Main entry point - uses Hydra configuration system."""
    # Use Hydra configuration system
    hydra_main = hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")(main_play)
    return hydra_main()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
