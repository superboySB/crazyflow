"""Train and evaluate CUDA PPO policies on the Crazyflow Gymnasium environments.

The trainer intentionally leaves the environments and their rewards unchanged. JAX runs parallel
Crazyflow environments on CUDA while PyTorch trains the policy on the same GPU.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers.vector import JaxToTorch
from torch import Tensor
from torch.distributions.normal import Normal

if TYPE_CHECKING:
    from collections.abc import Callable

import crazyflow.envs  # noqa: F401  # register the Gymnasium environments
from crazyflow.drones import Drone
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.envs import NormalizeActions

ENVIRONMENT_IDS = (
    "DroneReachPos-v0",
    "DroneReachVel-v0",
    "DroneLanding-v0",
    "DroneFigureEightTrajectory-v0",
)
ENVIRONMENT_SLUGS = {
    "DroneReachPos-v0": "reach_pos",
    "DroneReachVel-v0": "reach_vel",
    "DroneLanding-v0": "landing",
    "DroneFigureEightTrajectory-v0": "figure8",
}
BASE_OBS_KEYS = ("pos", "quat", "vel", "ang_vel")
TASK_OBS_KEYS = {
    "DroneReachPos-v0": ("difference_to_goal",),
    "DroneReachVel-v0": ("difference_to_target_vel",),
    "DroneLanding-v0": ("difference_to_goal",),
    "DroneFigureEightTrajectory-v0": ("local_samples",),
}


@dataclass
class Config:
    """PPO and environment configuration."""

    env_id: str = "DroneFigureEightTrajectory-v0"
    seed: int = 7
    total_timesteps: int = 20_000_000
    num_envs: int = 4_096
    num_steps: int = 64
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 16
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.002
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    freq: int = 50
    episode_time: float = 10.0
    n_samples: int = 10
    samples_dt: float = 0.1
    eval_envs: int = 256
    position_bc_steps: int = 1_000
    position_bc_coef: float = 10.0
    velocity_bc_steps: int = 0
    velocity_bc_coef: float = 0.0

    @property
    def batch_size(self) -> int:
        """Number of transitions collected for each PPO update."""
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        """Number of transitions in an optimization minibatch."""
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        """Number of complete PPO updates."""
        return self.total_timesteps // self.batch_size


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(config: Config, num_envs: int) -> gymnasium.vector.VectorEnv:
    """Create the unmodified official environment with normalized actions."""
    kwargs = {
        "num_envs": num_envs,
        "freq": config.freq,
        "max_episode_time": config.episode_time,
        "device": "gpu",
    }
    if config.env_id == "DroneFigureEightTrajectory-v0":
        kwargs |= {
            "trajectory_time": config.episode_time,
            "n_samples": config.n_samples,
            "samples_dt": config.samples_dt,
        }
    env = gymnasium.make_vec(config.env_id, **kwargs)
    env = NormalizeActions(env)
    return JaxToTorch(env, device=torch.device("cuda"))


def reset_env(
    env: gymnasium.vector.VectorEnv, seed: int, options: dict[str, object] | None = None
) -> tuple[dict[str, Tensor], dict]:
    """Reset both the simulator RNG and task-goal RNG reproducibly."""
    base_env = env.unwrapped
    if hasattr(base_env, "jax_key"):
        base_env.jax_key = jax.device_put(jax.random.key(seed), base_env.device)
    return env.reset(seed=seed, options=options)


def observation_keys(env_id: str) -> tuple[str, ...]:
    """Return a fixed observation order for a registered environment."""
    return BASE_OBS_KEYS + TASK_OBS_KEYS[env_id]


def flatten_obs(obs: dict[str, Tensor], obs_keys: tuple[str, ...]) -> Tensor:
    """Flatten the observation dict in a fixed, checkpoint-stable order."""
    return torch.cat([obs[key].reshape(obs[key].shape[0], -1) for key in obs_keys], dim=-1)


def task_error(obs: dict[str, Tensor], env_id: str) -> Tensor:
    """Return the primary position or velocity tracking-error vector."""
    if env_id == "DroneReachVel-v0":
        return obs["difference_to_target_vel"]
    if env_id == "DroneFigureEightTrajectory-v0":
        return obs["local_samples"][:, :3]
    return obs["difference_to_goal"]


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    """Use the orthogonal initialization common in PPO implementations."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """Separate actor and critic MLPs for continuous attitude control."""

    def __init__(self, obs_dim: int, action_dim: int):
        """Initialize actor and critic networks."""
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor_body = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )
        self.actor_output = layer_init(nn.Linear(128, action_dim), std=0.01)

        # The first three normalized actions command roll/pitch/yaw.  A normalized thrust of about
        # 0.22 corresponds to hover for cf2x_L250, so exploration starts near a useful operating
        # point instead of at the midpoint of the physical thrust range.
        hover_normalized = 0.22067
        self.actor_output.bias.data[-1] = np.arctanh(hover_normalized)
        self.actor_logstd = nn.Parameter(torch.tensor([[-1.2, -1.2, -2.0, -1.0]]))

    def get_value(self, obs: Tensor) -> Tensor:
        """Estimate the state value."""
        return self.critic(obs)

    def deterministic_action(self, obs: Tensor) -> Tensor:
        """Return the bounded mean action."""
        return torch.tanh(self.actor_output(self.actor_body(obs)))

    def get_action_and_value(
        self, obs: Tensor, action: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample an action and return its log probability, entropy, and value."""
        mean = self.deterministic_action(obs)
        std = torch.exp(self.actor_logstd).expand_as(mean)
        distribution = Normal(mean, std)
        if action is None:
            action = mean if deterministic else distribution.sample()
        return (
            action,
            distribution.log_prob(action).sum(dim=-1),
            distribution.entropy().sum(dim=-1),
            self.critic(obs),
        )


def acceleration_to_action(acceleration: Tensor) -> Tensor:
    """Convert desired world acceleration to a normalized attitude/thrust action."""
    params = load_params(Dynamics.so_rpy, Drone.cf2x_L250)
    mass = float(params["mass"])
    thrust_low = 4.0 * float(params["thrust_min"])
    thrust_high = 4.0 * float(params["thrust_max"])
    gravity = 9.81

    acceleration = torch.clamp(acceleration, min=-4.0, max=4.0)
    roll = torch.atan2(-acceleration[:, 1], gravity + acceleration[:, 2])
    pitch = torch.atan2(
        acceleration[:, 0],
        torch.sqrt(acceleration[:, 1] ** 2 + (gravity + acceleration[:, 2]) ** 2),
    )
    thrust = mass * torch.sqrt(
        acceleration[:, 0] ** 2 + acceleration[:, 1] ** 2 + (gravity + acceleration[:, 2]) ** 2
    )
    normalized_thrust = 2.0 * (thrust - thrust_low) / (thrust_high - thrust_low) - 1.0
    return torch.stack(
        (
            roll / (torch.pi / 2.0),
            pitch / (torch.pi / 2.0),
            torch.zeros_like(roll),
            normalized_thrust,
        ),
        dim=-1,
    ).clamp(-1.0, 1.0)


def position_teacher_from_state(position_error: Tensor, velocity: Tensor) -> Tensor:
    """Compute a critically damped position-controller action."""
    return acceleration_to_action(4.0 * position_error - 4.0 * velocity)


def velocity_teacher_from_state(velocity_error: Tensor) -> Tensor:
    """Compute a proportional velocity-controller action."""
    return acceleration_to_action(4.0 * velocity_error)


def position_teacher(obs: dict[str, Tensor]) -> Tensor:
    """Compute the position-controller teacher action from a dictionary observation."""
    return position_teacher_from_state(obs["difference_to_goal"], obs["vel"])


def position_teacher_from_flat_obs(obs: Tensor) -> Tensor:
    """Compute the teacher action from the fixed position-task observation layout."""
    # Layout: pos[0:3], quat[3:7], vel[7:10], ang_vel[10:13], goal error[13:16].
    return position_teacher_from_state(obs[:, 13:16], obs[:, 7:10])


def velocity_teacher(obs: dict[str, Tensor]) -> Tensor:
    """Compute the velocity-controller teacher action from a dictionary observation."""
    return velocity_teacher_from_state(obs["difference_to_target_vel"])


def velocity_teacher_from_flat_obs(obs: Tensor) -> Tensor:
    """Compute the teacher action from the fixed velocity-task observation layout."""
    return velocity_teacher_from_state(obs[:, 13:16])


def pretrain_actor(
    agent: Agent,
    env: gymnasium.vector.VectorEnv,
    obs_dict: dict[str, Tensor],
    obs_keys: tuple[str, ...],
    steps: int,
    teacher_fn: Callable[[dict[str, Tensor]], Tensor],
    label: str,
) -> None:
    """Behavior-clone a controller before PPO fine-tuning."""
    if steps <= 0:
        return
    optimizer = torch.optim.Adam(
        [*agent.actor_body.parameters(), *agent.actor_output.parameters()], lr=1e-3
    )
    final_loss = 0.0
    for _ in range(steps):
        flat_obs = flatten_obs(obs_dict, obs_keys)
        with torch.no_grad():
            teacher_action = teacher_fn(obs_dict)
        predicted_action = agent.deterministic_action(flat_obs)
        loss = torch.mean((predicted_action - teacher_action) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        obs_dict, _, _, _, _ = env.step(teacher_action)
    with torch.no_grad():
        agent.actor_logstd.copy_(torch.tensor([[-2.0, -2.0, -2.5, -1.5]], device="cuda"))
    print(f"{label} behavior cloning: {steps:,} batches, final MSE {final_loss:.6f}")


def save_checkpoint(
    path: Path,
    agent: Agent,
    config: Config,
    obs_keys: tuple[str, ...],
    history: dict[str, list[float]],
) -> None:
    """Save weights and enough metadata to reproduce evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
            "config": asdict(config),
            "obs_keys": obs_keys,
            "history": history,
        },
        path,
    )


def train(
    config: Config, checkpoint: Path
) -> tuple[Agent, tuple[str, ...], dict[str, list[float]]]:
    """Train PPO on parallel JAX/CUDA environments."""
    if config.batch_size % config.num_minibatches:
        raise ValueError("batch_size must be divisible by num_minibatches")
    if config.num_iterations < 1:
        raise ValueError("total_timesteps must contain at least one complete PPO batch")

    set_seeds(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    env = make_env(config, config.num_envs)
    first_obs, _ = reset_env(env, config.seed)
    obs_keys = observation_keys(config.env_id)
    next_obs = flatten_obs(first_obs, obs_keys)
    obs_dim = next_obs.shape[-1]
    action_dim = int(np.prod(env.single_action_space.shape))
    agent = Agent(obs_dim, action_dim).to(device)
    if config.env_id in ("DroneReachPos-v0", "DroneLanding-v0"):
        pretrain_actor(
            agent,
            env,
            first_obs,
            obs_keys,
            steps=config.position_bc_steps,
            teacher_fn=position_teacher,
            label="Position",
        )
        first_obs, _ = reset_env(env, config.seed)
        next_obs = flatten_obs(first_obs, obs_keys)
    elif config.env_id == "DroneReachVel-v0":
        pretrain_actor(
            agent,
            env,
            first_obs,
            obs_keys,
            steps=config.velocity_bc_steps,
            teacher_fn=velocity_teacher,
            label="Velocity",
        )
        first_obs, _ = reset_env(env, config.seed)
        next_obs = flatten_obs(first_obs, obs_keys)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    obs = torch.zeros((config.num_steps, config.num_envs, obs_dim), device=device)
    actions = torch.zeros((config.num_steps, config.num_envs, action_dim), device=device)
    logprobs = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards = torch.zeros((config.num_steps, config.num_envs), device=device)
    dones = torch.zeros((config.num_steps, config.num_envs), device=device)
    values = torch.zeros((config.num_steps, config.num_envs), device=device)

    next_done = torch.zeros(config.num_envs, dtype=torch.bool, device=device)
    episode_returns = torch.zeros(config.num_envs, device=device)
    history: dict[str, list[float]] = {
        "timesteps": [],
        "mean_reward": [],
        "mean_distance": [],
        "episode_return": [],
        "termination_rate": [],
    }
    completed_returns: list[float] = []
    global_step = 0
    start_time = time.perf_counter()

    print(
        f"Training {config.env_id}: {config.num_envs} CUDA envs, "
        f"{config.num_steps} rollout steps, "
        f"{config.num_iterations} PPO updates ({config.num_iterations * config.batch_size:,} "
        "transitions)"
    )

    for iteration in range(1, config.num_iterations + 1):
        if config.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / config.num_iterations
            optimizer.param_groups[0]["lr"] = fraction * config.learning_rate

        rollout_distance = 0.0
        rollout_terminated = 0.0
        completed_this_iteration: list[float] = []

        for step in range(config.num_steps):
            global_step += config.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value.flatten()

            next_obs_dict, reward, terminated, truncated, _ = env.step(action)
            next_obs = flatten_obs(next_obs_dict, obs_keys)
            rewards[step] = reward
            episode_returns += reward
            next_done = terminated | truncated

            if next_done.any():
                finished = episode_returns[next_done].detach().cpu().tolist()
                completed_returns.extend(finished)
                completed_this_iteration.extend(finished)
                episode_returns = torch.where(next_done, 0.0, episode_returns)

            rollout_distance += (
                torch.linalg.vector_norm(task_error(next_obs_dict, config.env_id), dim=-1)
                .mean()
                .item()
            )
            rollout_terminated += terminated.float().mean().item()

        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rewards)
            last_advantage = torch.zeros(config.num_envs, device=device)
            for step in reversed(range(config.num_steps)):
                if step == config.num_steps - 1:
                    next_nonterminal = 1.0 - next_done.float()
                    following_value = next_value
                else:
                    next_nonterminal = 1.0 - dones[step + 1]
                    following_value = values[step + 1]
                delta = (
                    rewards[step] + config.gamma * following_value * next_nonterminal - values[step]
                )
                last_advantage = (
                    delta + config.gamma * config.gae_lambda * next_nonterminal * last_advantage
                )
                advantages[step] = last_advantage
            returns = advantages + values

        batch_obs = obs.reshape(-1, obs_dim)
        batch_actions = actions.reshape(-1, action_dim)
        batch_logprobs = logprobs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        batch_values = values.reshape(-1)

        clip_fractions: list[float] = []
        approx_kl = torch.tensor(0.0, device=device)
        for _ in range(config.update_epochs):
            indices = torch.randperm(config.batch_size, device=device)
            for start in range(0, config.batch_size, config.minibatch_size):
                index = indices[start : start + config.minibatch_size]
                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    batch_obs[index], batch_actions[index]
                )
                log_ratio = new_logprob - batch_logprobs[index]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                    )

                minibatch_advantages = batch_advantages[index]
                minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (
                    minibatch_advantages.std() + 1e-8
                )
                policy_loss = torch.maximum(
                    -minibatch_advantages * ratio,
                    -minibatch_advantages
                    * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef),
                ).mean()

                new_value = new_value.flatten()
                value_unclipped = (new_value - batch_returns[index]) ** 2
                value_clipped = batch_values[index] + torch.clamp(
                    new_value - batch_values[index], -config.clip_coef, config.clip_coef
                )
                value_loss = (
                    0.5
                    * torch.maximum(
                        value_unclipped, (value_clipped - batch_returns[index]) ** 2
                    ).mean()
                )
                loss = policy_loss - config.ent_coef * entropy.mean() + config.vf_coef * value_loss
                if config.env_id in ("DroneReachPos-v0", "DroneLanding-v0"):
                    teacher_action = position_teacher_from_flat_obs(batch_obs[index])
                    imitation_loss = torch.mean(
                        (agent.deterministic_action(batch_obs[index]) - teacher_action) ** 2
                    )
                    loss += config.position_bc_coef * imitation_loss
                elif config.env_id == "DroneReachVel-v0":
                    teacher_action = velocity_teacher_from_flat_obs(batch_obs[index])
                    imitation_loss = torch.mean(
                        (agent.deterministic_action(batch_obs[index]) - teacher_action) ** 2
                    )
                    loss += config.velocity_bc_coef * imitation_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            if approx_kl.item() > config.target_kl:
                break

        mean_reward = rewards.mean().item()
        mean_distance = rollout_distance / config.num_steps
        termination_rate = rollout_terminated / config.num_steps
        if completed_this_iteration:
            mean_episode_return = float(np.mean(completed_this_iteration))
        elif completed_returns:
            mean_episode_return = float(np.mean(completed_returns[-config.num_envs :]))
        else:
            mean_episode_return = float("nan")
        history["timesteps"].append(global_step)
        history["mean_reward"].append(mean_reward)
        history["mean_distance"].append(mean_distance)
        history["episode_return"].append(mean_episode_return)
        history["termination_rate"].append(termination_rate)

        elapsed = time.perf_counter() - start_time
        print(
            f"update {iteration:03d}/{config.num_iterations} | steps {global_step:>10,} | "
            f"reward {mean_reward:.3f} | distance {mean_distance:.3f} m | "
            f"episode {mean_episode_return:.1f} | term {termination_rate:.3%} | "
            f"KL {approx_kl.item():.4f} | clip {np.mean(clip_fractions):.3f} | "
            f"{global_step / elapsed:,.0f} SPS"
        )

    elapsed = time.perf_counter() - start_time
    print(f"Training finished in {elapsed:.2f} s ({global_step / elapsed:,.0f} SPS)")
    save_checkpoint(checkpoint, agent, config, obs_keys, history)
    env.close()
    return agent, obs_keys, history


def load_checkpoint(path: Path) -> tuple[Agent, Config, tuple[str, ...], dict[str, list[float]]]:
    """Load a policy checkpoint."""
    data = torch.load(path, map_location="cuda", weights_only=False)
    config_data = data["config"]
    # Checkpoints produced by the earlier Figure-eight-only version predate env_id.
    config_data.setdefault("env_id", "DroneFigureEightTrajectory-v0")
    if "landing_bc_steps" in config_data:
        config_data.setdefault("position_bc_steps", config_data.pop("landing_bc_steps"))
    if "landing_bc_coef" in config_data:
        config_data.setdefault("position_bc_coef", config_data.pop("landing_bc_coef"))
    config = Config(**config_data)
    obs_keys = tuple(data["obs_keys"])
    if obs_keys != observation_keys(config.env_id):
        raise ValueError("Checkpoint observation order does not match this script")
    obs_dim = sum(
        3 * config.n_samples if key == "local_samples" else 4 if key == "quat" else 3
        for key in obs_keys
    )
    agent = Agent(obs_dim, action_dim=4).cuda()
    agent.load_state_dict(data["agent"])
    agent.eval()
    return agent, config, obs_keys, data["history"]


@torch.no_grad()
def evaluate(
    agent: Agent,
    config: Config,
    obs_keys: tuple[str, ...],
    reset_options: dict[str, object] | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluate deterministic policies over randomized initial states."""
    env = make_env(config, config.eval_envs)
    obs_dict, _ = reset_env(env, config.seed + 10_000, options=reset_options)
    obs = flatten_obs(obs_dict, obs_keys)
    distances: list[Tensor] = []
    rewards: list[Tensor] = []
    speeds: list[Tensor] = []
    terminated_once = torch.zeros(config.eval_envs, dtype=torch.bool, device="cuda")
    actual: list[np.ndarray] = []
    reference: list[np.ndarray] = []
    episode_steps = int(config.episode_time * config.freq)

    for _ in range(episode_steps):
        action, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
        obs_dict, reward, terminated, truncated, _ = env.step(action)
        obs = flatten_obs(obs_dict, obs_keys)
        error = task_error(obs_dict, config.env_id)
        distances.append(torch.linalg.vector_norm(error, dim=-1))
        rewards.append(reward)
        speeds.append(torch.linalg.vector_norm(obs_dict["vel"], dim=-1))
        terminated_once |= terminated
        state_key = "vel" if config.env_id == "DroneReachVel-v0" else "pos"
        actual.append(obs_dict[state_key][0].cpu().numpy())
        reference.append((obs_dict[state_key][0] + error[0]).cpu().numpy())

    distance = torch.stack(distances)
    reward = torch.stack(rewards)
    speed = torch.stack(speeds)
    per_env_rmse = torch.sqrt(torch.mean(distance**2, dim=0))
    last_steps = min(config.freq, episode_steps)
    per_env_last_rmse = torch.sqrt(torch.mean(distance[-last_steps:] ** 2, dim=0))
    threshold = 0.15 if config.env_id == "DroneFigureEightTrajectory-v0" else 0.10
    successful = (~terminated_once) & (per_env_last_rmse < threshold)
    if config.env_id == "DroneLanding-v0":
        per_env_last_speed_rmse = torch.sqrt(torch.mean(speed[-last_steps:] ** 2, dim=0))
        successful &= per_env_last_speed_rmse < 0.10
    metrics = {
        "mean_reward": reward.mean().item(),
        "episode_return": reward.sum(dim=0).mean().item(),
        "mean_error": distance.mean().item(),
        "rmse": torch.sqrt(torch.mean(distance**2)).item(),
        "last_second_rmse": torch.sqrt(torch.mean(distance[-last_steps:] ** 2)).item(),
        "final_mean_error": distance[-1].mean().item(),
        "median_env_rmse": per_env_rmse.median().item(),
        "p95_error": torch.quantile(distance, 0.95).item(),
        "max_error": distance.max().item(),
        "survival_rate": (~terminated_once).float().mean().item(),
        "success_rate": successful.float().mean().item(),
    }
    env.close()
    behavior = {
        "actual": np.asarray(actual),
        "reference": np.asarray(reference),
        "mean_error": distance.mean(dim=1).cpu().numpy(),
    }
    return metrics, behavior


def plot_result(
    path: Path,
    config: Config,
    history: dict[str, list[float]],
    metrics: dict[str, float],
    behavior: dict[str, np.ndarray],
) -> None:
    """Plot learning progress and deterministic evaluation behavior."""
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.asarray(history["timesteps"]) / 1e6
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    unit = "m/s" if config.env_id == "DroneReachVel-v0" else "m"

    axes[0].plot(steps, history["mean_distance"], color="#d95f02")
    axes[0].set(xlabel="environment steps (million)", ylabel=f"mean error ({unit})")
    axes[0].set_title("PPO training")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, history["mean_reward"], color="#1b9e77")
    axes[1].set(xlabel="environment steps (million)", ylabel="mean reward / step")
    axes[1].set_title("Official environment reward")
    axes[1].grid(alpha=0.3)

    actual, reference = behavior["actual"], behavior["reference"]
    if config.env_id == "DroneReachVel-v0":
        time_axis = np.arange(len(actual)) / config.freq
        for index, name in enumerate(("vx", "vy", "vz")):
            (line,) = axes[2].plot(time_axis, actual[:, index], label=name)
            axes[2].plot(
                time_axis, reference[:, index], linestyle="--", color=line.get_color(), alpha=0.7
            )
        axes[2].set(xlabel="time (s)", ylabel="velocity (m/s)")
        axes[2].legend(ncol=3)
    elif config.env_id == "DroneFigureEightTrajectory-v0":
        axes[2].plot(reference[:, 0], reference[:, 2], label="reference", linewidth=3)
        axes[2].plot(actual[:, 0], actual[:, 2], label="PPO policy", linewidth=2)
        axes[2].scatter(actual[0, 0], actual[0, 2], s=35, label="start")
        axes[2].set(xlabel="x (m)", ylabel="z (m)", aspect="equal")
        axes[2].legend()
    else:
        axes[2].plot(actual[:, 0], actual[:, 2], label="PPO policy", linewidth=2)
        axes[2].scatter(
            reference[0, 0],
            reference[0, 2],
            s=150,
            marker="*",
            color="#d95f02",
            label="goal",
            zorder=3,
        )
        axes[2].scatter(actual[0, 0], actual[0, 2], s=35, label="start", zorder=3)
        axes[2].set(xlabel="x (m)", ylabel="z (m)")
        axes[2].legend()
    axes[2].set_title(
        f"Deterministic evaluation\nlast-second RMSE "
        f"{metrics['last_second_rmse']:.3f} {unit}, "
        f"success {metrics['success_rate']:.1%}"
    )
    axes[2].grid(alpha=0.3)

    figure.suptitle(f"{config.env_id} — JAX/CUDA environments + PyTorch/CUDA PPO")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", choices=ENVIRONMENT_IDS, default=Config.env_id)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--total-timesteps", type=int, default=Config.total_timesteps)
    parser.add_argument("--num-envs", type=int, default=Config.num_envs)
    parser.add_argument("--num-steps", type=int, default=Config.num_steps)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--eval-envs", type=int, default=Config.eval_envs)
    parser.add_argument("--position-bc-steps", type=int, default=Config.position_bc_steps)
    parser.add_argument("--position-bc-coef", type=float, default=Config.position_bc_coef)
    parser.add_argument("--velocity-bc-steps", type=int, default=Config.velocity_bc_steps)
    parser.add_argument("--velocity-bc-coef", type=float, default=Config.velocity_bc_coef)
    parser.add_argument(
        "--horizontal-velocity-eval",
        action="store_true",
        help="Evaluate ReachVel with target vz fixed to zero.",
    )
    return parser.parse_args()


def main() -> None:
    """Train (unless disabled), evaluate, and plot."""
    args = parse_args()
    slug = ENVIRONMENT_SLUGS[args.env_id]
    checkpoint = args.checkpoint or Path(f"saves/{slug}_ppo.pt")
    output = args.output or Path(f"usage_assets/{slug}_ppo_result.png")
    if args.eval_only:
        agent, config, obs_keys, history = load_checkpoint(checkpoint)
        if config.env_id != args.env_id:
            raise ValueError(
                f"Checkpoint contains {config.env_id}, but --env-id selected {args.env_id}"
            )
        config.seed = args.seed
        config.eval_envs = args.eval_envs
    else:
        config = Config(
            env_id=args.env_id,
            seed=args.seed,
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            learning_rate=args.learning_rate,
            eval_envs=args.eval_envs,
            position_bc_steps=args.position_bc_steps,
            position_bc_coef=args.position_bc_coef,
            velocity_bc_steps=args.velocity_bc_steps,
            velocity_bc_coef=args.velocity_bc_coef,
        )
        agent, obs_keys, history = train(config, checkpoint)

    reset_options = None
    if args.horizontal_velocity_eval:
        if config.env_id != "DroneReachVel-v0":
            raise ValueError("--horizontal-velocity-eval only applies to DroneReachVel-v0")
        reset_options = {
            "vel_min": jnp.array([-1.0, -1.0, 0.0]),
            "vel_max": jnp.array([1.0, 1.0, 0.0]),
        }
    metrics, behavior = evaluate(agent, config, obs_keys, reset_options=reset_options)
    for name, value in metrics.items():
        print(f"eval/{name}={value:.6f}")
    plot_result(output, config, history, metrics, behavior)
    print(f"plot saved to {output}")


if __name__ == "__main__":
    main()
