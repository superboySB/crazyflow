"""Train and evaluate CUDA PPO policies on the Crazyflow Gymnasium environments.

The trainer intentionally leaves the environments and their rewards unchanged. JAX runs parallel
Crazyflow environments on CUDA while PyTorch trains the policy on the same GPU.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, dataclass
from functools import partial
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
RESET_PROTOCOL = "independent_goal_rng_v1"
GOAL_RNG_FOLD_IN = 0x474F414C
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
    eval_interval: int = 50
    validation_seed: int = 20_007
    teacher_model: str = "legacy"

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
    """Seed simulator and goals reproducibly using independent, domain-separated RNG streams.

    Reusing the simulator key for task goals makes ReachPos goals share their initial x/y and
    ReachVel targets become an affine transform of initial position, including after autoreset.
    The fixed GOAL tag changes only the goal stream; the simulator's original seed is preserved.
    """
    base_env = env.unwrapped
    if hasattr(base_env, "jax_key"):
        goal_key = jax.random.fold_in(jax.random.key(seed), GOAL_RNG_FOLD_IN)
        base_env.jax_key = jax.device_put(goal_key, base_env.device)
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


def acceleration_to_action(acceleration: Tensor, teacher_model: str = "legacy") -> Tensor:
    """Map world acceleration to bounded commands, retaining the historical recipe by default.

    The calibrated teacher inverts the official so_rpy thrust fit, steady-state attitude gain,
    and the xyz Euler geometry. It is a distinct training recipe; rewards/dynamics are unchanged.
    """
    params = load_params(Dynamics.so_rpy, Drone.cf2x_L250)
    mass = float(params["mass"])
    thrust_low = 4.0 * float(params["thrust_min"])
    thrust_high = 4.0 * float(params["thrust_max"])
    gravity = 9.81
    acceleration = torch.clamp(acceleration, min=-4.0, max=4.0)
    if teacher_model == "legacy":
        # Preserve all arithmetic in the historical teacher for the main study matrix.
        roll = torch.atan2(-acceleration[:, 1], gravity + acceleration[:, 2])
        pitch = torch.atan2(
            acceleration[:, 0],
            torch.sqrt(acceleration[:, 1] ** 2 + (gravity + acceleration[:, 2]) ** 2),
        )
        thrust = mass * torch.sqrt(
            acceleration[:, 0] ** 2 + acceleration[:, 1] ** 2 + (gravity + acceleration[:, 2]) ** 2
        )
    elif teacher_model == "calibrated":
        gravity_vec = torch.as_tensor(
            params["gravity_vec"], dtype=acceleration.dtype, device=acceleration.device
        )
        specific_force = acceleration - gravity_vec
        ax, ay, az = specific_force.unbind(dim=-1)
        # R.from_euler("xyz", [r,p,0]) e_z = [sin(p)cos(r), -sin(r), cos(p)cos(r)].
        roll = torch.atan2(-ay, torch.sqrt(ax**2 + az**2))
        pitch = torch.atan2(ax, az)
        steady_gain = -np.asarray(params["cmd_rpy_coef"]) / np.asarray(params["rpy_coef"])
        roll = roll / float(steady_gain[0])
        pitch = pitch / float(steady_gain[1])
        force = mass * torch.linalg.vector_norm(specific_force, dim=-1)
        thrust = (force - float(params["acc_coef"])) / float(params["cmd_f_coef"])
    else:
        raise ValueError(f"Unknown teacher model: {teacher_model}")
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


def position_teacher_from_state(
    position_error: Tensor, velocity: Tensor, teacher_model: str = "legacy"
) -> Tensor:
    """Compute a critically damped position-controller action."""
    return acceleration_to_action(4.0 * position_error - 4.0 * velocity, teacher_model)


def velocity_teacher_from_state(velocity_error: Tensor, teacher_model: str = "legacy") -> Tensor:
    """Compute a proportional velocity-controller action."""
    return acceleration_to_action(4.0 * velocity_error, teacher_model)


def position_teacher(obs: dict[str, Tensor], teacher_model: str = "legacy") -> Tensor:
    """Compute the configured position-controller teacher action."""
    return position_teacher_from_state(obs["difference_to_goal"], obs["vel"], teacher_model)


def position_teacher_from_flat_obs(obs: Tensor, teacher_model: str = "legacy") -> Tensor:
    """Compute the teacher action from the fixed position-task observation layout."""
    # Layout: pos[0:3], quat[3:7], vel[7:10], ang_vel[10:13], goal error[13:16].
    return position_teacher_from_state(obs[:, 13:16], obs[:, 7:10], teacher_model)


def velocity_teacher(obs: dict[str, Tensor], teacher_model: str = "legacy") -> Tensor:
    """Compute the configured velocity-controller teacher action."""
    return velocity_teacher_from_state(obs["difference_to_target_vel"], teacher_model)


def velocity_teacher_from_flat_obs(obs: Tensor, teacher_model: str = "legacy") -> Tensor:
    """Compute the teacher action from the fixed velocity-task observation layout."""
    return velocity_teacher_from_state(obs[:, 13:16], teacher_model)


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


def checkpoint_last_path(path: Path) -> Path:
    """Return the independent recovery checkpoint path."""
    return path.with_name(f"{path.stem}_last{path.suffix}")


def snapshot_environment(env: gymnasium.vector.VectorEnv) -> dict:
    """Serialize simulator leaves, task goals, and reset bookkeeping without device objects."""
    base = env.unwrapped
    leaves = jax.tree.leaves(base.sim.data)
    is_key = [jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key) for leaf in leaves]
    state = {
        "sim_leaves": [
            np.asarray(jax.device_get(jax.random.key_data(leaf) if key else leaf)).copy()
            for leaf, key in zip(leaves, is_key, strict=True)
        ],
        "key_leaves": is_key,
        "marked_for_reset": np.asarray(base._marked_for_reset).copy(),
    }
    for name in ("_goal", "_goal_vel"):
        if hasattr(base, name):
            state[name] = np.asarray(getattr(base, name)).copy()
    if hasattr(base, "jax_key"):
        state["jax_key"] = np.asarray(jax.random.key_data(base.jax_key)).copy()
    return state


def restore_environment(env: gymnasium.vector.VectorEnv, state: dict) -> None:
    """Restore a checkpoint into a freshly constructed environment with matching configuration."""
    base = env.unwrapped
    template, structure = jax.tree.flatten(base.sim.data)
    if len(template) != len(state["sim_leaves"]):
        raise ValueError("Checkpoint simulator structure does not match the installed Crazyflow")
    leaves = [
        jax.device_put(jax.random.wrap_key_data(jnp.asarray(value)) if key else value, base.device)
        for value, key in zip(state["sim_leaves"], state["key_leaves"], strict=True)
    ]
    base.sim.data = jax.tree.unflatten(structure, leaves)
    base._marked_for_reset = jax.device_put(state["marked_for_reset"], base.device)
    for name in ("_goal", "_goal_vel"):
        if name in state:
            setattr(base, name, jax.device_put(state[name], base.device))
    if "jax_key" in state:
        base.jax_key = jax.device_put(
            jax.random.wrap_key_data(jnp.asarray(state["jax_key"])), base.device
        )


def save_checkpoint(
    path: Path,
    agent: Agent,
    config: Config,
    obs_keys: tuple[str, ...],
    history: dict[str, list[float]],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    trainer_state: dict | None = None,
) -> None:
    """Atomically save policy, configuration, and optional complete training recovery state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format_version": 3,
        "reset_protocol": RESET_PROTOCOL,
        "goal_rng_fold_in": GOAL_RNG_FOLD_IN,
        "agent": agent.state_dict(),
        "config": asdict(config),
        "obs_keys": obs_keys,
        "history": history,
    }
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
        data["rng"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        }
    if trainer_state is not None:
        data["trainer_state"] = trainer_state
    temporary = path.with_name(path.name + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    done_after: Tensor,
    valid: Tensor,
    timeout_values: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Stop GAE at episode boundaries, bootstrap time limits, and exclude NEXT_STEP reset calls."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for step in reversed(range(len(rewards))):
        following_value = next_values if step == len(rewards) - 1 else values[step + 1]
        continues = (~done_after[step]).float()
        delta = (
            rewards[step]
            + gamma * (following_value * continues + timeout_values[step])
            - values[step]
        )
        carry = torch.where(valid[step], delta + gamma * gae_lambda * continues * carry, 0.0)
        advantages[step] = carry
    return advantages, advantages + values


def train(
    config: Config, checkpoint: Path, resume: Path | None = None
) -> tuple[Agent, tuple[str, ...], dict[str, list[float]]]:
    """Train PPO with held-out checkpoint selection and recoverable simulator/optimizer state.

    ``checkpoint`` contains the best validation policy; ``*_last.pt`` contains the latest update.
    total_timesteps counts vector-step slots. history.valid_transitions excludes autoreset calls.
    Resume preserves RNG, simulator and optimizer state; changing the total budget changes future LR
    annealing, so an extended run is a continuation rather than an identical uninterrupted schedule.
    """
    if config.teacher_model not in ("legacy", "calibrated"):
        raise ValueError("teacher_model must be legacy or calibrated")
    if config.batch_size % config.num_minibatches:
        raise ValueError("batch_size must be divisible by num_minibatches")
    if config.num_iterations < 1 or config.eval_interval < 1:
        raise ValueError("Require at least one complete PPO batch and eval_interval >= 1")
    if config.validation_seed == config.seed:
        raise ValueError("validation_seed must differ from the training seed")

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
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    next_done = torch.zeros(config.num_envs, dtype=torch.bool, device=device)
    episode_returns = torch.zeros(config.num_envs, device=device)
    history: dict[str, list[float]] = {
        key: []
        for key in (
            "timesteps",
            "valid_transitions",
            "mean_reward",
            "mean_distance",
            "episode_return",
            "termination_rate",
            "approx_kl",
            "clip_fraction",
            "validation_timesteps",
            "validation_success",
            "validation_survival",
            "validation_rmse",
        )
    }
    completed_returns: list[float] = []
    global_step = 0
    valid_step = 0
    start_iteration = 1
    best_score = (float("-inf"), float("-inf"), float("-inf"))
    best_iteration = 0

    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved.get("reset_protocol") != RESET_PROTOCOL:
            raise ValueError(
                "Resume reset protocol differs; restart training for independent goals"
            )
        if "optimizer" not in saved or "trainer_state" not in saved:
            raise ValueError(
                "Resume requires a recovery checkpoint; old checkpoints remain evaluable"
            )
        if saved["config"].get("teacher_model", "legacy") != config.teacher_model:
            raise ValueError("Resume teacher model differs; do not mix training recipes")
        mutable = {"total_timesteps", "eval_envs", "eval_interval"}
        for name, value in saved["config"].items():
            if name not in mutable and getattr(config, name) != value:
                raise ValueError(f"Resume configuration differs at {name}: {value!r}")
        if tuple(saved["obs_keys"]) != obs_keys:
            raise ValueError("Resume observation layout differs")
        agent.load_state_dict(saved["agent"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["trainer_state"]
        restore_environment(env, state["environment"])
        next_obs = state["next_obs"].to(device)
        next_done = state["next_done"].to(device)
        episode_returns = state["episode_returns"].to(device)
        completed_returns = state["completed_returns"]
        history = saved["history"]
        global_step, valid_step = state["global_step"], state["valid_step"]
        start_iteration = state["iteration"] + 1
        best_score, best_iteration = tuple(state["best_score"]), state["best_iteration"]
        # A recovery checkpoint is portable: restore its archived best weights even at a new path.
        best_agent = state.get("best_agent", saved["agent"])
        best_policy = Agent(obs_dim, action_dim).to(device)
        best_policy.load_state_dict(best_agent)
        save_checkpoint(checkpoint, best_policy, config, obs_keys, history)
        del best_policy
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        print(f"Resuming after update {start_iteration - 1}, {global_step:,} vector-step slots")
    else:
        if config.env_id in ("DroneReachPos-v0", "DroneLanding-v0"):
            pretrain_actor(
                agent,
                env,
                first_obs,
                obs_keys,
                config.position_bc_steps,
                partial(position_teacher, teacher_model=config.teacher_model),
                "Position",
            )
        elif config.env_id == "DroneReachVel-v0":
            pretrain_actor(
                agent,
                env,
                first_obs,
                obs_keys,
                config.velocity_bc_steps,
                partial(velocity_teacher, teacher_model=config.teacher_model),
                "Velocity",
            )
        first_obs, _ = reset_env(env, config.seed)
        next_obs = flatten_obs(first_obs, obs_keys)
        best_agent = None

    shape = (config.num_steps, config.num_envs)
    obs = torch.zeros((*shape, obs_dim), device=device)
    actions = torch.zeros((*shape, action_dim), device=device)
    logprobs = torch.zeros(shape, device=device)
    rewards = torch.zeros(shape, device=device)
    values = torch.zeros(shape, device=device)
    valid = torch.zeros(shape, dtype=torch.bool, device=device)
    done_after = torch.zeros_like(valid)
    timeout_values = torch.zeros(shape, device=device)
    validation_env = make_env(config, config.eval_envs)
    start_time = time.perf_counter()
    initial_step = global_step

    print(
        f"Training {config.env_id}: {config.num_envs} CUDA envs, {config.num_steps} rollout steps, "
        f"{config.num_iterations} PPO updates ({config.num_iterations * config.batch_size:,} slots)"
    )
    try:
        for iteration in range(start_iteration, config.num_iterations + 1):
            agent.train()
            if config.anneal_lr:
                optimizer.param_groups[0]["lr"] = (
                    1.0 - (iteration - 1.0) / config.num_iterations
                ) * config.learning_rate
            rollout_distance = 0.0
            rollout_terminated = 0
            completed_this_iteration: list[float] = []
            for step in range(config.num_steps):
                global_step += config.num_envs
                obs[step] = next_obs
                # Official NEXT_STEP resets after simulating an ignored action. Its returned reset
                # observation/reward is not a transition generated by the sampled policy action.
                valid[step] = ~next_done
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                actions[step], logprobs[step], values[step] = action, logprob, value.flatten()
                next_obs_dict, reward, terminated, truncated, _ = env.step(action)
                next_obs = flatten_obs(next_obs_dict, obs_keys)
                rewards[step] = reward
                done_after[step] = terminated | truncated
                timeout_values[step].zero_()
                time_limit = valid[step] & truncated & ~terminated
                if time_limit.any():
                    with torch.no_grad():
                        timeout_values[step, time_limit] = agent.get_value(
                            next_obs[time_limit]
                        ).flatten()
                episode_returns += torch.where(valid[step], reward, 0.0)
                finished_mask = valid[step] & done_after[step]
                if finished_mask.any():
                    finished = episode_returns[finished_mask].detach().cpu().tolist()
                    completed_this_iteration.extend(finished)
                    completed_returns.extend(finished)
                    episode_returns[finished_mask] = 0.0
                next_done = done_after[step].clone()
                error = torch.linalg.vector_norm(task_error(next_obs_dict, config.env_id), dim=-1)
                rollout_distance += error[valid[step]].sum().item()
                rollout_terminated += (terminated & valid[step]).sum().item()

            with torch.no_grad():
                advantages, returns = compute_gae(
                    rewards,
                    values,
                    agent.get_value(next_obs).flatten(),
                    done_after,
                    valid,
                    timeout_values,
                    config.gamma,
                    config.gae_lambda,
                )
            mask = valid.reshape(-1)
            valid_count = int(mask.sum().item())
            if valid_count < config.num_minibatches:
                raise RuntimeError("Too few valid transitions for PPO minibatches")
            valid_step += valid_count
            batch_obs = obs.reshape(-1, obs_dim)[mask]
            batch_actions = actions.reshape(-1, action_dim)[mask]
            batch_logprobs = logprobs.reshape(-1)[mask]
            batch_advantages = advantages.reshape(-1)[mask]
            batch_returns = returns.reshape(-1)[mask]
            batch_values = values.reshape(-1)[mask]
            clip_fractions: list[float] = []
            kls: list[float] = []
            approx_kl = torch.tensor(0.0, device=device)
            for _ in range(config.update_epochs):
                indices = torch.randperm(valid_count, device=device)
                epoch_kls = []
                # tensor_split keeps every valid sample and avoids a final singleton minibatch.
                for index in torch.tensor_split(indices, config.num_minibatches):
                    _, new_logprob, entropy, new_value = agent.get_action_and_value(
                        batch_obs[index], batch_actions[index]
                    )
                    log_ratio = new_logprob - batch_logprobs[index]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        epoch_kls.append(approx_kl.item())
                        clip_fractions.append(
                            ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                        )
                    mb_adv = batch_advantages[index]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)
                    policy_loss = torch.maximum(
                        -mb_adv * ratio,
                        -mb_adv
                        * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef),
                    ).mean()
                    new_value = new_value.flatten()
                    value_clipped = batch_values[index] + torch.clamp(
                        new_value - batch_values[index], -config.clip_coef, config.clip_coef
                    )
                    value_loss = (
                        0.5
                        * torch.maximum(
                            (new_value - batch_returns[index]) ** 2,
                            (value_clipped - batch_returns[index]) ** 2,
                        ).mean()
                    )
                    loss = (
                        policy_loss - config.ent_coef * entropy.mean() + config.vf_coef * value_loss
                    )
                    if (
                        config.env_id in ("DroneReachPos-v0", "DroneLanding-v0")
                        and config.position_bc_coef
                    ):
                        teacher_action = position_teacher_from_flat_obs(
                            batch_obs[index], teacher_model=config.teacher_model
                        )
                        loss += (
                            config.position_bc_coef
                            * (
                                (agent.deterministic_action(batch_obs[index]) - teacher_action) ** 2
                            ).mean()
                        )
                    elif config.env_id == "DroneReachVel-v0" and config.velocity_bc_coef:
                        teacher_action = velocity_teacher_from_flat_obs(
                            batch_obs[index], teacher_model=config.teacher_model
                        )
                        loss += (
                            config.velocity_bc_coef
                            * (
                                (agent.deterministic_action(batch_obs[index]) - teacher_action) ** 2
                            ).mean()
                        )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    optimizer.step()
                kls.extend(epoch_kls)
                if config.target_kl and float(np.mean(epoch_kls)) > config.target_kl:
                    break

            mean_reward = rewards[valid].mean().item()
            mean_distance = rollout_distance / valid_count
            termination_rate = rollout_terminated / valid_count
            completed_returns = completed_returns[-config.num_envs :]
            recent = completed_this_iteration or completed_returns
            mean_episode_return = float(np.mean(recent)) if recent else float("nan")
            for key, value in (
                ("timesteps", global_step),
                ("valid_transitions", valid_step),
                ("mean_reward", mean_reward),
                ("mean_distance", mean_distance),
                ("episode_return", mean_episode_return),
                ("termination_rate", termination_rate),
                ("approx_kl", float(np.mean(kls))),
                ("clip_fraction", float(np.mean(clip_fractions))),
            ):
                history[key].append(value)
            elapsed = time.perf_counter() - start_time
            print(
                f"update {iteration:04d}/{config.num_iterations} | steps {global_step:>12,} | "
                f"valid {valid_step:,} | reward {mean_reward:.4f} | distance {mean_distance:.4f} | "
                f"episode {mean_episode_return:.1f} | term {termination_rate:.3%} | "
                f"KL {np.mean(kls):.4f} | clip {np.mean(clip_fractions):.3f} | "
                f"{(global_step - initial_step) / elapsed:,.0f} SPS",
                flush=True,
            )
            if (
                iteration == start_iteration
                or iteration % config.eval_interval == 0
                or iteration == config.num_iterations
            ):
                agent.eval()
                metrics, _ = evaluate(
                    agent, config, obs_keys, eval_seed=config.validation_seed, env=validation_env
                )
                history["validation_timesteps"].append(global_step)
                history["validation_success"].append(metrics["success_rate"])
                history["validation_survival"].append(metrics["survival_rate"])
                history["validation_rmse"].append(metrics["rmse"])
                # Success/survival are hard constraints; RMSE resolves equally reliable policies.
                score = (metrics["success_rate"], metrics["survival_rate"], -metrics["rmse"])
                improved = score > best_score
                if improved:
                    best_score, best_iteration = score, iteration
                    best_agent = {
                        name: value.detach().cpu().clone()
                        for name, value in agent.state_dict().items()
                    }
                state = {
                    "iteration": iteration,
                    "global_step": global_step,
                    "valid_step": valid_step,
                    "next_obs": next_obs.detach().cpu(),
                    "next_done": next_done.detach().cpu(),
                    "episode_returns": episode_returns.detach().cpu(),
                    "completed_returns": completed_returns,
                    "best_score": best_score,
                    "best_iteration": best_iteration,
                    "best_agent": best_agent,
                    "environment": snapshot_environment(env),
                }
                save_checkpoint(
                    checkpoint_last_path(checkpoint),
                    agent,
                    config,
                    obs_keys,
                    history,
                    optimizer=optimizer,
                    trainer_state=state,
                )
                if improved:
                    save_checkpoint(
                        checkpoint,
                        agent,
                        config,
                        obs_keys,
                        history,
                        optimizer=optimizer,
                        trainer_state=state,
                    )
                print(
                    f"validation/{iteration} seed={config.validation_seed} "
                    f"success={metrics['success_rate']:.6f} "
                    f"survival={metrics['survival_rate']:.6f} rmse={metrics['rmse']:.6f} "
                    f"best_update={best_iteration}",
                    flush=True,
                )
    finally:
        env.close()
        validation_env.close()
    elapsed = time.perf_counter() - start_time
    print(f"Training finished in {elapsed:.2f} s; best validation update {best_iteration}")
    if best_agent is not None:
        agent.load_state_dict(best_agent)
    agent.eval()
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
    eval_seed: int | None = None,
    env: gymnasium.vector.VectorEnv | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluate first episodes only, keeping failures and excluding all post-reset samples.

    The final-second error is measured over each first episode's final observed second. Success
    additionally requires no failure and a complete requested horizon. Landing is the official
    0.1 m near-ground hover task, not contact landing. A supplied env is reset and kept open.
    """
    owns_env = env is None
    if env is None:
        env = make_env(config, config.eval_envs)
    count = env.num_envs
    actual_seed = config.seed + 10_000 if eval_seed is None else eval_seed
    obs_dict, _ = reset_env(env, actual_seed, options=reset_options)
    initial_pos = obs_dict["pos"].cpu().numpy().copy()
    initial_vel = obs_dict["vel"].cpu().numpy().copy()
    initial_target = (
        (obs_dict["vel"] + task_error(obs_dict, config.env_id)).cpu().numpy().copy()
        if config.env_id == "DroneReachVel-v0"
        else np.full_like(initial_pos, np.nan)
    )
    obs = flatten_obs(obs_dict, obs_keys)
    distances, rewards, speeds, masks = [], [], [], []
    terminated_once = torch.zeros(count, dtype=torch.bool, device="cuda")
    active = torch.ones(count, dtype=torch.bool, device="cuda")
    actual, reference = [], []
    episode_steps = int(config.episode_time * config.freq)
    try:
        for _ in range(episode_steps):
            action, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
            obs_dict, reward, terminated, truncated, _ = env.step(action)
            obs = flatten_obs(obs_dict, obs_keys)
            error = task_error(obs_dict, config.env_id)
            distances.append(torch.linalg.vector_norm(error, dim=-1))
            rewards.append(reward)
            speeds.append(torch.linalg.vector_norm(obs_dict["vel"], dim=-1))
            masks.append(active.clone())
            terminated_once |= active & terminated
            if active[0]:
                state_key = "vel" if config.env_id == "DroneReachVel-v0" else "pos"
                actual.append(obs_dict[state_key][0].cpu().numpy().copy())
                reference.append((obs_dict[state_key][0] + error[0]).cpu().numpy().copy())
            active &= ~(terminated | truncated)
    finally:
        if owns_env:
            env.close()
    distance, reward, speed = torch.stack(distances), torch.stack(rewards), torch.stack(speeds)
    valid = torch.stack(masks)
    lengths = valid.sum(dim=0)
    per_env_rmse = torch.sqrt((distance.square() * valid).sum(dim=0) / lengths.clamp_min(1))
    indices = torch.arange(episode_steps, device="cuda")[:, None]
    last_valid = valid & (indices >= (lengths - config.freq).clamp_min(0)[None, :])
    last_counts = last_valid.sum(dim=0).clamp_min(1)
    per_env_last_rmse = torch.sqrt((distance.square() * last_valid).sum(dim=0) / last_counts)
    per_env_last_speed = torch.sqrt((speed.square() * last_valid).sum(dim=0) / last_counts)
    threshold = 0.15 if config.env_id == "DroneFigureEightTrajectory-v0" else 0.10
    successful = (~terminated_once) & (lengths == episode_steps) & (per_env_last_rmse < threshold)
    if config.env_id == "DroneLanding-v0":
        successful &= per_env_last_speed < 0.10
    final_distance = distance.gather(0, (lengths - 1).clamp_min(0)[None, :]).squeeze(0)
    metrics = {
        "mean_reward": reward[valid].mean().item(),
        "episode_return": (reward * valid).sum(dim=0).mean().item(),
        "mean_error": distance[valid].mean().item(),
        "rmse": torch.sqrt(distance[valid].square().mean()).item(),
        "last_second_rmse": torch.sqrt(distance[last_valid].square().mean()).item(),
        "final_mean_error": final_distance.mean().item(),
        "median_env_rmse": per_env_rmse.median().item(),
        "p95_error": torch.quantile(distance[valid], 0.95).item(),
        "max_error": distance[valid].max().item(),
        "survival_rate": (~terminated_once).float().mean().item(),
        "success_rate": successful.float().mean().item(),
        "mean_episode_steps": lengths.float().mean().item(),
        "evaluation_seed": float(actual_seed),
    }
    behavior = {
        "actual": np.asarray(actual),
        "reference": np.asarray(reference),
        "mean_error": ((distance * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)).cpu().numpy(),
        "per_env_rmse": per_env_rmse.cpu().numpy(),
        "per_env_last_rmse": per_env_last_rmse.cpu().numpy(),
        "per_env_last_speed_rmse": per_env_last_speed.cpu().numpy(),
        "per_env_success": successful.cpu().numpy(),
        "per_env_survival": (~terminated_once).cpu().numpy(),
        "per_env_episode_steps": lengths.cpu().numpy(),
        "initial_pos": initial_pos,
        "initial_vel": initial_vel,
        "initial_target_vel": initial_target,
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
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-seed", type=int)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--total-timesteps", type=int, default=Config.total_timesteps)
    parser.add_argument("--num-envs", type=int, default=Config.num_envs)
    parser.add_argument("--num-steps", type=int, default=Config.num_steps)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--eval-envs", type=int, default=Config.eval_envs)
    parser.add_argument("--eval-interval", type=int, default=Config.eval_interval)
    parser.add_argument("--validation-seed", type=int, default=Config.validation_seed)
    parser.add_argument("--teacher-model", choices=("legacy", "calibrated"), default="legacy")
    parser.add_argument("--gamma", type=float, default=Config.gamma)
    parser.add_argument("--gae-lambda", type=float, default=Config.gae_lambda)
    parser.add_argument("--num-minibatches", type=int, default=Config.num_minibatches)
    parser.add_argument("--update-epochs", type=int, default=Config.update_epochs)
    parser.add_argument("--ent-coef", type=float, default=Config.ent_coef)
    parser.add_argument("--target-kl", type=float, default=Config.target_kl)
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
            eval_interval=args.eval_interval,
            validation_seed=args.validation_seed,
            teacher_model=args.teacher_model,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            num_minibatches=args.num_minibatches,
            update_epochs=args.update_epochs,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
        )
        agent, obs_keys, history = train(config, checkpoint, resume=args.resume)

    reset_options = None
    if args.horizontal_velocity_eval:
        if config.env_id != "DroneReachVel-v0":
            raise ValueError("--horizontal-velocity-eval only applies to DroneReachVel-v0")
        reset_options = {
            "vel_min": jnp.array([-1.0, -1.0, 0.0]),
            "vel_max": jnp.array([1.0, 1.0, 0.0]),
        }
    metrics, behavior = evaluate(
        agent,
        config,
        obs_keys,
        reset_options=reset_options,
        eval_seed=args.eval_seed if args.eval_seed is not None else args.seed + 10_000,
    )
    for name, value in metrics.items():
        print(f"eval/{name}={value:.6f}")
    plot_result(output, config, history, metrics, behavior)
    print(f"plot saved to {output}")


if __name__ == "__main__":
    main()
