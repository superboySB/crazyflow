"""Train and audit the pinned racing PPO with held-out trajectories and a real gate course.

Run inside the research image from /workspace/crazyflow:
    python usage_assets/racing_fulltrain_20261015.py
Use --smoke for a short implementation check, never as evidence of training quality.

The policy and PPO optimizer are the official implementation. The adapter fixes episode
boundaries, resets observation history, bootstraps time limits, applies the configured action
disturbance, and samples fresh splines after automatic resets. Source instrumentation happens
only in memory; the pinned lsy_drone_racing installation is never edited.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import jax
import jax.numpy as jp
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch
from jax.scipy.spatial.transform import Rotation
from lsy_drone_racing.control import train_rl as official
from lsy_drone_racing.utils import load_config
from scipy.interpolate import CubicSpline

from crazyflow.sim import functional as sim_functional

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "usage_assets"
MODELS = ROOT / "trained_models"
CONFIG = Path(official.__file__).parents[2] / "config" / "level0.toml"
TRAJECTORY_STREAM_SALT = 0x5452414A  # "TRAJ": separate from the simulator's action/dynamics RNG.
COEFS = {
    k: getattr(official.Args(), k)
    for k in ("n_obs", "rpy_coef", "d_act_th_coef", "d_act_xy_coef", "act_coef")
}


@contextlib.contextmanager
def preserve_rng():
    """Prevent validation from changing the training random stream."""
    state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all(),
    )
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        torch.cuda.set_rng_state_all(state[3])


class RacingTrainingEnv(VectorEnv):
    """Official random-spline dynamics and rewards with explicit SAME_STEP boundaries."""

    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(self, num_envs=1024, jax_device="gpu", coefs=None):
        self.coefs = COEFS if coefs is None else coefs
        config = load_config(CONFIG)
        self.base = official.RandTrajEnv(
            num_envs=num_envs,
            freq=config.env.freq,
            drone=config.sim.drone,
            dynamics=config.sim.dynamics,
            disturbances=config.env.disturbances,
            device=jax_device,
        )
        self.num_envs = num_envs
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (73,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.history = jp.zeros((num_envs, 2, 13), device=self.base.sim.device)
        self.last_action = jp.zeros((num_envs, 4), device=self.base.sim.device)
        self.last_distance = None
        self.last_position = None
        self.episode_count = 0
        self.simultaneous_terminal_time_limits = 0
        self.completed_returns = []
        self.returns = np.zeros(num_envs)
        # Linear spline basis reproduces SciPy's official boundary condition while allowing
        # fresh, per-world trajectories to be sampled on the JAX device after an auto reset.
        knots = np.linspace(0, self.base.trajectory_time, self.base.num_waypoints)
        ts = np.linspace(0, self.base.trajectory_time, self.base.n_steps)
        self.basis = jp.array(
            CubicSpline(
                knots,
                np.eye(self.base.num_waypoints),
                axis=0,
                bc_type=((1, np.zeros(self.base.num_waypoints)), "not-a-knot"),
            )(ts),
            dtype=jp.float32,
            device=self.base.sim.device,
        )
        self.offset = jp.array(
            CubicSpline(
                knots,
                np.zeros((self.base.num_waypoints, 3)),
                axis=0,
                bc_type=((1, np.array([0.0, 0.0, 0.4])), "not-a-knot"),
            )(ts),
            dtype=jp.float32,
            device=self.base.sim.device,
        )
        self._sample = jax.jit(self._sample_trajectories)
        self._step = self._build_step()

    def _sample_trajectories(self, key):
        waypoints = jax.random.uniform(key, (self.num_envs, 10, 3), minval=-1, maxval=1)
        waypoints = (
            waypoints * jp.array([1.2, 1.2, 0.5])
            + 0.3 * jp.array(self.base.takeoff_pos)
            + jp.array([0.0, 0.0, 0.7])
        )
        waypoints = waypoints.at[:, :3].set(
            jp.array([[-1.5, 1.0, 0.07], [-1.0, 0.55, 0.4], [0.3, 0.35, 0.7]])
        )
        return jp.einsum("tk,nkc->ntc", self.basis, waypoints) + self.offset

    @staticmethod
    def basic(raw):
        return jp.concatenate([raw[k] for k in ("pos", "quat", "vel", "ang_vel")], axis=-1)

    @staticmethod
    def flatten(raw, history, last_action):
        return jp.concatenate(
            [
                RacingTrainingEnv.basic(raw),
                raw["local_samples"],
                history.reshape(history.shape[0], -1),
                last_action,
            ],
            axis=-1,
        ).astype(jp.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.base.reset(seed=seed, options=options)
        self.key = jax.device_put(
            jax.random.fold_in(
                jax.random.PRNGKey(0 if seed is None else seed), TRAJECTORY_STREAM_SALT
            ),
            self.base.sim.device,
        )
        self.key, sample_key = jax.random.split(self.key)
        self.base.trajectories = self._sample(sample_key)
        raw = self.base.obs()
        self.history = jp.repeat(self.basic(raw)[:, None], 2, axis=1)
        self.last_action = jp.zeros((self.num_envs, 4), device=self.base.sim.device)
        self.returns[:] = 0
        return self.flatten(raw, self.history, self.last_action), {}

    def _build_step(self):
        """Fuse physics, rewards and reset handling to avoid dozens of tiny GPU launches."""
        sim = self.base.sim
        low = jp.array(self.base.single_action_space.low, device=sim.device)
        high = jp.array(self.base.single_action_space.high, device=sim.device)
        offsets = jp.array(self.base.sample_offsets, device=sim.device)
        action_noise = self.base.disturbances.get("action")

        def observations(data, trajectories):
            states = data.states
            index = data.core.steps // self.base.n_substeps - 1
            index = jp.clip(index + offsets[None], 0, self.base.n_steps - 1)
            samples = trajectories[jp.arange(self.num_envs)[:, None], index]
            return {
                "pos": states.pos[:, 0],
                "quat": states.quat[:, 0],
                "vel": states.vel[:, 0],
                "ang_vel": states.ang_vel[:, 0],
                "local_samples": (samples - states.pos).reshape(self.num_envs, -1),
            }

        @jax.jit
        def step(data, trajectories, key, history, last_action, action):
            applied = action.at[:, 2].set(0)
            physical = jp.clip(applied, -1, 1) * (high - low) / 2 + (high + low) / 2
            physical = physical.reshape(self.num_envs, 1, 4)
            if action_noise is not None:
                rng_key, noise_key = jax.random.split(data.core.rng_key)
                physical += action_noise(noise_key, physical.shape)
                data = data.replace(core=data.core.replace(rng_key=rng_key))
            data = sim_functional.attitude_control(data, physical)
            data = sim._step(data, n_steps=self.base.n_substeps)
            raw = observations(data, trajectories)
            terminated = self.base._terminated(data.states.pos)
            truncated = data.core.steps[:, 0] / data.core.freq >= self.base.max_episode_time
            distance = jp.linalg.norm(raw["local_samples"][:, :3], axis=-1)
            position = raw["pos"]
            reward = jp.where(terminated, -1.0, jp.exp(-2.0 * distance))
            reward -= self.coefs["rpy_coef"] * jp.linalg.norm(
                Rotation.from_quat(raw["quat"]).as_euler("xyz"), axis=-1
            )
            delta = action - last_action
            reward -= self.coefs["act_coef"] * action[:, -1] ** 2
            reward -= self.coefs["d_act_th_coef"] * delta[:, -1] ** 2
            reward -= self.coefs["d_act_xy_coef"] * jp.sum(delta[:, :3] ** 2, axis=-1)
            final_obs = self.flatten(raw, history, action)
            history_next = jp.concatenate([history[:, 1:], self.basic(raw)[:, None]], axis=1)
            done = terminated | truncated
            data = sim._reset(data, sim.default_data, done)

            def resample(carry):
                trajectories, key = carry
                key, sample_key = jax.random.split(key)
                trajectories = jp.where(done[:, None, None], self._sample(sample_key), trajectories)
                return trajectories, key

            trajectories, key = jax.lax.cond(
                jp.any(done), resample, lambda carry: carry, (trajectories, key)
            )
            raw = observations(data, trajectories)
            initial_history = jp.repeat(self.basic(raw)[:, None], 2, axis=1)
            output_history = jp.where(done[:, None, None], initial_history, history)
            history_next = jp.where(done[:, None, None], initial_history, history_next)
            output_action = jp.where(done[:, None], jp.zeros_like(action), action)
            obs = self.flatten(raw, output_history, output_action)
            return (
                data,
                trajectories,
                key,
                history_next,
                output_action,
                obs,
                reward,
                terminated,
                truncated,
                final_obs,
                distance,
                position,
            )

        return step

    def step(self, action):
        (
            self.base.sim.data,
            self.base.trajectories,
            self.key,
            self.history,
            self.last_action,
            obs,
            reward,
            terminated,
            truncated,
            final_obs,
            self.last_distance,
            self.last_position,
        ) = self._step(
            self.base.sim.data,
            self.base.trajectories,
            self.key,
            self.history,
            self.last_action,
            action,
        )
        done_cpu = np.asarray(terminated | truncated)
        self.simultaneous_terminal_time_limits += int(np.asarray(terminated & truncated).sum())
        self.returns += np.asarray(reward)
        if done_cpu.any():
            self.completed_returns.extend(self.returns[done_cpu].tolist())
            self.episode_count += int(done_cpu.sum())
            self.returns[done_cpu] = 0
        return (obs, reward, terminated, truncated, {"final_observation": final_obs})

    def close(self):
        self.base.close()


def make_envs(config="level0.toml", num_envs=1024, jax_device="gpu", torch_device=None, coefs=None):
    return JaxToTorch(RacingTrainingEnv(num_envs, jax_device, coefs), torch_device)


def verify_rng_protocol():
    """Assert disjoint named RNG streams, repeatability, diversity and auto-reset semantics."""
    env = RacingTrainingEnv(16, "cpu")
    env.base.max_episode_time = 0.02
    initial, _ = env.reset(seed=123)
    trajectories = np.asarray(env.base.trajectories).copy()
    sim_key = env.base.sim.data.core.rng_key
    sim_key, action_key = jax.random.split(sim_key)
    _, dynamics_key = jax.random.split(sim_key)
    _, trajectory_key = jax.random.split(
        jax.random.fold_in(jax.random.PRNGKey(123), TRAJECTORY_STREAM_SALT)
    )
    keys = [
        tuple(np.asarray(jax.random.key_data(k)).tolist())
        for k in (trajectory_key, action_key, dynamics_key)
    ]
    assert len(set(keys)) == 3, "trajectory, action noise and dynamics noise keys must differ"

    # Check many future subkeys too, so the regression covers repeated episode resets.
    def subkeys(key):
        values = set()
        for _ in range(128):
            key, sample = jax.random.split(key)
            values.add(tuple(np.asarray(jax.random.key_data(sample)).tolist()))
        return values

    assert not subkeys(jax.random.PRNGKey(123)) & subkeys(
        jax.random.fold_in(jax.random.PRNGKey(123), TRAJECTORY_STREAM_SALT)
    )
    first = env.step(jp.zeros((16, 4)))
    auto_trajectories = np.asarray(env.base.trajectories).copy()
    assert np.asarray(first[3]).all() and not np.asarray(first[2]).any()
    assert np.asarray(env.base.sim.time).max() == 0
    assert np.allclose(np.asarray(first[0])[:, -4:], 0)
    assert np.allclose(np.asarray(first[0])[:, 43:56], np.asarray(first[0])[:, :13])
    assert not np.array_equal(auto_trajectories, trajectories)
    second_initial, _ = env.reset(seed=123)
    np.testing.assert_array_equal(initial, second_initial)
    np.testing.assert_array_equal(trajectories, env.base.trajectories)
    second = env.step(jp.zeros((16, 4)))
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(auto_trajectories, env.base.trajectories)
    env.reset(seed=124)
    assert not np.array_equal(trajectories, env.base.trajectories)
    diversity = np.std(trajectories[:, 500], axis=0)
    assert np.all(diversity > 0.05), "random tail waypoints must vary between worlds"
    env.close()
    return {
        "passed": True,
        "trajectory_stream_salt": TRAJECTORY_STREAM_SALT,
        "first_trajectory_action_dynamics_keys": keys,
        "future_subkeys_checked_per_stream": 128,
        "same_seed_reproduces_initial_and_auto_reset": True,
        "different_seeds_change_trajectories": True,
        "across_world_tail_position_std_m": diversity.tolist(),
    }


def evaluate(checkpoint, seed, num_envs, device, *, save_trajectory=None):
    """Each world contributes exactly its first independently seeded episode."""
    with preserve_rng():
        official.set_seeds(seed)
        env = make_envs(num_envs=num_envs, jax_device="gpu", torch_device=device, coefs=COEFS)
        core = env.unwrapped
        agent = official.Agent((73,), (4,)).to(device)
        agent.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        agent.eval()
        obs, _ = env.reset(seed=seed)
        targets = np.asarray(core.base.trajectories).copy()
        goals = targets[0]
        below_ground = targets[:, :, 2].min(axis=1) < 0
        active = np.ones(num_envs, dtype=bool)
        returns = np.zeros(num_envs)
        counts = np.zeros(num_envs, dtype=int)
        distance_sum = np.zeros(num_envs)
        distance_squared = np.zeros(num_envs)
        crashes = np.zeros(num_envs, dtype=bool)
        simultaneous_terminal_time_limits = 0
        positions = []
        with torch.no_grad():
            for _ in range(750):
                action, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                errors = np.asarray(core.last_distance)
                returns += np.asarray(reward.cpu()) * active
                counts += active
                distance_sum += errors * active
                distance_squared += errors**2 * active
                term, trunc = np.asarray(terminated.cpu()), np.asarray(truncated.cpu())
                simultaneous_terminal_time_limits += int(np.count_nonzero(term & trunc & active))
                crashes |= term & active
                if active[0]:
                    positions.append(np.asarray(core.last_position[0]).copy())
                active &= ~(term | trunc)
                if not active.any():
                    break
        result = {
            "seed": seed,
            "episodes": num_envs,
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "return_ci95": float(1.96 * returns.std(ddof=1) / np.sqrt(num_envs)),
            "survival_rate": float((~crashes & (counts == 750)).mean()),
            "crashes": int(crashes.sum()),
            "simultaneous_terminal_time_limits": simultaneous_terminal_time_limits,
            "crashed_episode_indices": np.flatnonzero(crashes).tolist(),
            "references_below_ground": int(below_ground.sum()),
            "reference_below_ground_indices": np.flatnonzero(below_ground).tolist(),
            "reference_min_height_m": float(targets[:, :, 2].min()),
            "ground_valid_reference_survival_rate": float(
                (~crashes & (counts == 750))[~below_ground].mean()
            ),
            "ground_valid_reference_position_rmse_m": float(
                np.sqrt(distance_squared[~below_ground].sum() / counts[~below_ground].sum())
            ),
            "mean_episode_steps": float(counts.mean()),
            "position_mae_m": float(np.mean(distance_sum / counts)),
            "position_rmse_m": float(np.sqrt(distance_squared.sum() / counts.sum())),
            "episode_return_quantiles": np.quantile(returns, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        }
        if save_trajectory:
            np.savez_compressed(
                save_trajectory,
                positions=np.array(positions),
                target=goals,
                returns=returns,
                episode_steps=counts,
                distance_sum_m=distance_sum,
                distance_squared_sum_m2=distance_squared,
                crashes=crashes,
                reference_below_ground=below_ground,
                reference_min_height_per_world_m=targets[:, :, 2].min(axis=1),
            )
        env.close()
        del env, agent
        torch.cuda.empty_cache()
        return result


def verify_inertia_distribution(samples=4096):
    """Validate the late racing patch against the actual model and configured distribution."""
    from lsy_drone_racing.envs import randomize as race_randomize
    from lsy_drone_racing.envs.race_core import rng_spec2fn

    config = load_config(CONFIG.parent / "level1.toml")
    env = official.RandTrajEnv(
        num_envs=1,
        freq=config.env.freq,
        drone=config.sim.drone,
        dynamics=config.sim.dynamics,
        device="cpu",
    )
    data = env.sim.data
    nominal = np.asarray(data.params.J).reshape(-1, 3, 3)[0].copy()
    params = data.params.replace(
        J=jp.broadcast_to(data.params.J, (samples, 1, 3, 3)),
        J_inv=jp.broadcast_to(data.params.J_inv, (samples, 1, 3, 3)),
    )
    randomize = jax.jit(
        race_randomize.randomize_drone_inertia_fn(
            rng_spec2fn(config.env.randomizations.drone_inertia)
        )
    )
    result = randomize(
        data.replace(params=params), jp.ones(samples, dtype=bool, device=params.J.device)
    )
    inertia, inverse = np.asarray(result.params.J), np.asarray(result.params.J_inv)
    np.testing.assert_array_equal(inertia, inertia.swapaxes(-1, -2))
    eigenvalues = np.linalg.eigvalsh(inertia)
    triangle_margin = eigenvalues.sum(axis=-1) - 2 * eigenvalues.max(axis=-1)
    assert eigenvalues.min() > 0 and triangle_margin.min() > 0
    inverse_error = float(np.max(np.abs(inertia @ inverse - np.eye(3))))
    assert inverse_error < 1e-6
    noise = np.diagonal(inertia - nominal, axis1=-2, axis2=-1)
    assert noise.min() >= -1.00001e-6 and noise.max() <= 1.00001e-6
    nominal_eigenvalues = np.linalg.eigvalsh(nominal)
    nominal_margin = nominal_eigenvalues.sum() - 2 * nominal_eigenvalues.max()
    summary = {
        "passed": True,
        "samples": samples,
        "drone": config.sim.drone,
        "nominal_principal_moments_kg_m2": nominal_eigenvalues.tolist(),
        "nominal_triangle_margin_kg_m2": float(nominal_margin),
        "analytic_worst_case_triangle_margin_kg_m2": float(nominal_margin - 3e-6),
        "sample_min_eigenvalue_kg_m2": float(eigenvalues.min()),
        "sample_min_triangle_margin_kg_m2": float(triangle_margin.min()),
        "max_inverse_error": inverse_error,
        "sample_diagonal_noise_range_kg_m2": [float(noise.min()), float(noise.max())],
        "distribution": (
            "Independent uniform [-1e-6, 1e-6] noise on three principal moments; "
            "no rejection or clipping."
        ),
        "randomize_source_sha256": hashlib.sha256(
            Path(race_randomize.__file__).read_bytes()
        ).hexdigest(),
    }
    env.close()
    return summary


def smoke_current_image(checkpoint=None):
    """Check the patched seeded race reset and three frozen-policy steps on CPU."""
    from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv

    torch.set_num_threads(2)
    config = load_config(CONFIG)
    config.sim.render = False
    race = VecDroneRaceEnv(
        num_envs=2,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        control_mode="attitude",
        seed=123,
        device="cpu",
    )
    first, _ = race.reset(seed=123)
    second, _ = race.reset(seed=123)
    np.testing.assert_array_equal(first["pos"], second["pos"])
    assert np.isfinite(np.asarray(second["pos"])).all()
    race.close()
    if checkpoint is None:
        checkpoint = json.loads((ASSETS / "racing_fulltrain_20261015.json").read_text())[
            "selected"
        ]["checkpoint"]
    checkpoint = ROOT / checkpoint
    agent = official.Agent((73,), (4,)).to("cpu")
    agent.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    env = RacingTrainingEnv(2, "cpu")
    obs, _ = env.reset(seed=123)
    with torch.no_grad():
        for _ in range(3):
            action, _, _, _ = agent.get_action_and_value(
                torch.tensor(np.asarray(obs)), deterministic=True
            )
            obs, reward, term, trunc, _ = env.step(jp.array(action.numpy()))
            assert np.isfinite(np.asarray(obs)).all() and np.isfinite(np.asarray(reward)).all()
            assert not np.asarray(term | trunc).any()
    env.close()
    return {
        "seeded_race_reset_worlds": 2,
        "frozen_policy_steps": 3,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "passed": True,
    }


def instrument_trainer(hook, legacy_terminal_bootstrap=False):
    """Execute the official PPO function with checkpoint and correct timeout hooks."""
    source = inspect.getsource(official.train_ppo)
    timeout_mask = "truncations" if legacy_terminal_bootstrap else "(truncations & ~terminations)"
    source = source.replace(
        "            rewards[step] = reward",
        "            with torch.no_grad():\n"
        "                reward = reward + args.gamma * agent.get_value(\n"
        f"                    infos['final_observation']).flatten() * {timeout_mask}\n"
        "            rewards[step] = reward",
    )
    source = source.replace(
        "        end_time = time.time()",
        "        checkpoint_hook(agent, optimizer, args, iteration, global_step, envs,\n"
        "                        float(explained_var), float(approx_kl))\n"
        "        end_time = time.time()",
    )
    # Per-iteration logs make long runs unnecessarily noisy; keep progress every 25 updates.
    source = source.replace(
        '        print(f"Iter {iteration}/{args.num_iterations} '
        'took {end_time - start_time:.2f} seconds")',
        "        if iteration % 25 == 0:\n"
        '            print(f"Iter {iteration}/{args.num_iterations}, steps={global_step}, "\n'
        '                  f"seconds={end_time - train_start_time:.1f}", flush=True)',
    )
    namespace = vars(official).copy()
    namespace.update(make_envs=make_envs, checkpoint_hook=hook)
    exec(compile(source, "<instrumented-official-racing-train_ppo>", "exec"), namespace)
    return namespace["train_ppo"], hashlib.sha256(source.encode()).hexdigest()


def course_evaluate(
    checkpoint,
    seed,
    num_envs,
    device,
    *,
    save_trajectory=None,
    config_name="level0.toml",
    reference_time_s=15.0,
):
    """Evaluate the official fixed course with frames, obstacles and collision detection."""
    from lsy_drone_racing.control.attitude_rl import AttitudeRL
    from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv

    config = load_config(CONFIG.parent / config_name)
    config.sim.render = False
    config.env.control_mode = "attitude"
    env = VecDroneRaceEnv(
        num_envs=num_envs,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode="attitude",
        disturbances=config.env.disturbances,
        randomizations=config.env.get("randomizations"),
        seed=seed,
        device="gpu",
    )
    # Disable autoreset so a completed race cannot be overwritten by a new episode.
    env.settings = env.settings.replace(autoreset=False)
    env._step = env.build_step_fn()
    # Explicit seeding also exercises the maintained SimCore.device compatibility fix.
    obs, info = env.reset(seed=seed)
    first_obs = {k: np.asarray(v[0]) for k, v in obs.items()}
    first_info = {k: np.asarray(v[0]) for k, v in info.items()}
    controller = AttitudeRL(first_obs, first_info, config)
    reference_steps = int(round(reference_time_s * 50))
    compressed = CubicSpline(np.linspace(0, 15, len(controller.trajectory)), controller.trajectory)(
        np.linspace(0, 15, reference_steps)
    )
    # Match the official single-world planner's observed starting pose in every world.
    # Later waypoints remain fixed nominal course waypoints, so this is not online planning.
    start_basis = CubicSpline(np.linspace(0, 15, 10), np.eye(10)[:, 0])(
        np.linspace(0, 15, reference_steps)
    )
    start_offset = np.asarray(obs["pos"]) - first_obs["pos"][None]
    per_world = compressed[None] + start_basis[None, :, None] * start_offset[:, None]
    trajectory = jp.array(per_world, device=jax.devices("gpu")[0])
    agent = official.Agent((73,), (4,)).to(device)
    agent.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    basic = jp.concatenate([obs[k] for k in ("pos", "quat", "vel", "ang_vel")], axis=-1)
    history = jp.repeat(basic[:, None], 2, axis=1)
    last_action = jp.zeros((num_envs, 4), device=trajectory.device)
    low, high = env.single_action_space.low, env.single_action_space.high
    active = np.ones(num_envs, dtype=bool)
    final_gates = np.zeros(num_envs, dtype=int)
    finish_steps = np.full(num_envs, reference_steps, dtype=int)
    positions, quaternions = [], []
    with torch.no_grad():
        for tick in range(reference_steps):
            idx = np.clip(tick + controller.sample_offsets, 0, reference_steps - 1)
            dpos = trajectory[:, idx] - obs["pos"][:, None]
            flat = jp.concatenate(
                [basic, dpos.reshape(num_envs, -1), history.reshape(num_envs, -1), last_action],
                axis=-1,
            )
            tensor = torch.from_dlpack(flat).to(device)
            act, _, _, _ = agent.get_action_and_value(tensor, deterministic=True)
            normalized = jax.dlpack.from_dlpack(act.contiguous())
            applied = normalized.at[:, 2].set(0)
            physical = jp.clip(applied, -1, 1) * jp.array((high - low) / 2) + jp.array(
                (high + low) / 2
            )
            obs, _, terminated, truncated, _ = env.step(physical)
            gates = np.asarray(obs["n_gates_passed"])
            final_gates[active] = gates[active]
            if active[0]:
                positions.append(np.asarray(obs["pos"][0]).copy())
                quaternions.append(np.asarray(obs["quat"][0]).copy())
            done = np.asarray(terminated | truncated) | (gates >= 4)
            finish_steps[active & done] = tick + 1
            active &= ~done
            history = jp.concatenate([history[:, 1:], basic[:, None]], axis=1)
            basic = jp.concatenate([obs[k] for k in ("pos", "quat", "vel", "ang_vel")], axis=-1)
            last_action = normalized
            if not active.any():
                break
    success = final_gates >= 4
    result = {
        "seed": seed,
        "episodes": num_envs,
        "success_rate": float(success.mean()),
        "gates_mean": float(final_gates.mean()),
        "gates_histogram": np.bincount(final_gates, minlength=5).tolist(),
        "successful_time_mean_s": float(finish_steps[success].mean() / 50)
        if success.any()
        else None,
        "config": config_name,
        "reference_time_s": reference_time_s,
        "disturbances": "official configured action/dynamics disturbance",
        "randomizations": list(config.env.get("randomizations", {}).keys()),
    }
    if save_trajectory:
        np.savez_compressed(
            save_trajectory,
            positions=np.array(positions),
            target=np.asarray(trajectory[0]),
            quaternions=np.array(quaternions),
            gates=final_gates,
            finish_steps=finish_steps,
        )
    env.close()
    del env, agent
    torch.cuda.empty_cache()
    return result


def evaluate_speed_envelope(checkpoint, device):
    """Choose the fastest reference on a small validation set, then test it independently."""
    sweep = [
        course_evaluate(checkpoint, 322019, 16, device, reference_time_s=duration)
        for duration in (15.0, 12.0, 10.0, 7.5, 5.0, 4.0, 3.0, 2.0)
    ]
    passed = [item for item in sweep if item["success_rate"] == 1.0]
    duration = min(item["reference_time_s"] for item in passed) if passed else 15.0
    holdout = course_evaluate(
        checkpoint,
        827450,
        64,
        device,
        reference_time_s=duration,
        save_trajectory=ASSETS / "racing_fulltrain_20261015_fast_course.npz",
    )
    return {
        "validation_seed": 322019,
        "validation_worlds": 16,
        "selection_rule": "Fastest 16/16 validation reference; fresh 64-world holdout.",
        "sweep": sweep,
        "selected_reference_time_s": duration,
        "holdout": holdout,
    }


def render_course(path, output_stem="racing_fulltrain_20261015_course"):
    """Replay the first predeclared course episode in MuJoCo, with its measured poses."""
    import imageio.v2 as imageio
    import mujoco
    from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial.transform import Rotation as ScipyRotation

    config = load_config(CONFIG)
    config.sim.render = False
    env = VecDroneRaceEnv(
        num_envs=1,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        control_mode="attitude",
        seed=827451,
        device="cpu",
    )
    env.reset(seed=827451)
    model = env.sim.mj_model
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    data = mujoco.MjData(model)
    gate_ids, obstacle_ids = env.mocap_ids
    for mocap_id, gate in zip(gate_ids, config.env.track.gates):
        data.mocap_pos[mocap_id] = gate["pos"]
        data.mocap_quat[mocap_id] = np.roll(
            ScipyRotation.from_euler("xyz", gate["rpy"]).as_quat(), 1
        )
    for mocap_id, obstacle in zip(obstacle_ids, config.env.track.obstacles):
        data.mocap_pos[mocap_id] = obstacle["pos"]
    drone_id = int(np.asarray(env.sim.data.core.drone_mocap_ids).reshape(-1)[0])
    record = np.load(path)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.65]
    camera.distance, camera.azimuth, camera.elevation = 4.8, 125, -25
    renderer = mujoco.Renderer(model, height=720, width=1280)
    output = ASSETS / f"{output_stem}.mp4"
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    with imageio.get_writer(output, fps=25, codec="libx264", quality=8) as writer:
        for index in range(0, len(record["positions"]), 2):
            data.mocap_pos[drone_id] = record["positions"][index]
            data.mocap_quat[drone_id] = np.roll(record["quaternions"][index], 1)
            # The simulator's mocap bodies intentionally use distinct weld IDs for MJX
            # collision detection. Rendering measured poses needs only kinematics/cameras;
            # the native constraint solver rejects contacts between those static bodies.
            mujoco.mj_kinematics(model, data)
            mujoco.mj_camlight(model, data)
            renderer.update_scene(data, camera)
            for start, end in zip(record["target"][::25][:-1], record["target"][::25][1:]):
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).ravel(),
                    np.array([0.15, 0.8, 0.95, 0.7]),
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2, start, end)
                renderer.scene.ngeom += 1
            flown = record["positions"][: index + 1 : 5]
            for start, end in zip(flown[:-1], flown[1:]):
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).ravel(),
                    np.array([1.0, 0.42, 0.08, 1.0]),
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 4, start, end)
                renderer.scene.ngeom += 1
            frame = renderer.render()
            canvas = Image.fromarray(frame)
            text = ImageDraw.Draw(canvas, "RGBA")
            text.rectangle((18, 18, 850, 92), fill=(10, 20, 35, 205))
            text.text((32, 27), "PPO drone racing | recorded simulation", font=font, fill="white")
            text.text(
                (32, 58),
                f"t = {(index + 1) / 50:.2f} s | 50 Hz control | cyan: reference | orange: flight",
                font=font,
                fill="white",
            )
            frame = np.asarray(canvas)
            writer.append_data(frame)
            if index == 2 * (min(250, len(record["positions"]) - 1) // 2):
                imageio.imwrite(ASSETS / f"{output_stem}.png", frame)
    renderer.close()
    env.close()
    return str(output.relative_to(ROOT))


def plot_results(results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for seed in results["seeds"]:
        items = [x for x in results["checkpoints"] if x["training_seed"] == seed]
        axes[0].errorbar(
            [x["actual_transitions"] / 1e6 for x in items],
            [x["validation"]["return_mean"] for x in items],
            yerr=[x["validation"]["return_ci95"] for x in items],
            marker="o",
            label=f"seed {seed}",
        )
        axes[1].plot(
            [x["actual_transitions"] / 1e6 for x in items],
            [x["validation"]["position_rmse_m"] for x in items],
            marker="o",
            label=f"seed {seed}",
        )
    axes[0].set(
        xlabel="Transitions (millions)",
        ylabel="Validation return / 15 s",
        title="128 fixed validation splines",
    )
    axes[1].set(
        xlabel="Transitions (millions)",
        ylabel="Position RMSE (m)",
        title="Longer training comparison",
    )
    axes[0].axhline(
        results["previous_validation"]["return_mean"], color="gray", ls="--", label="Previous 1.5 M"
    )
    axes[1].axhline(
        results["previous_validation"]["position_rmse_m"],
        color="gray",
        ls="--",
        label="Previous 1.5 M",
    )
    paths = np.load(ASSETS / "racing_fulltrain_20261015_course.npz")
    axes[2].plot(paths["target"][:, 0], paths["target"][:, 1], "--", label="Reference")
    axes[2].plot(paths["positions"][:, 0], paths["positions"][:, 1], label="Selected policy")
    config = load_config(CONFIG)
    for i, gate in enumerate(config.env.track.gates):
        axes[2].scatter(gate["pos"][0], gate["pos"][1], marker="s", s=50, color="#e86f1b")
        axes[2].annotate(str(i + 1), (gate["pos"][0], gate["pos"][1]))
    axes[2].set(
        xlabel="x (m)",
        ylabel="y (m)",
        title=f"Independent gate course: {results['course']['success_rate']:.1%} success",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    axes[2].axis("equal")
    fig.tight_layout()
    fig.savefig(ASSETS / "racing_fulltrain_20261015.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", default="41,42")
    parser.add_argument("--budgets", default="1500000,5000000,15000000")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--legacy-terminal-bootstrap", action="store_true")
    options = parser.parse_args()
    MODELS.mkdir(exist_ok=True)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    seeds = [int(x) for x in options.seeds.split(",")]
    budgets = [int(x) for x in options.budgets.split(",")]
    if options.smoke:
        seeds, budgets = [41], [16384]
    implementation_suffix = "_legacy_timeout" if options.legacy_terminal_bootstrap else ""
    smoke_suffix = "_smoke" if options.smoke else ""
    destination = ASSETS / f"racing_fulltrain_20261015{implementation_suffix}{smoke_suffix}.json"
    results = {
        "date": "2026-10-15",
        "racing_commit": "065a6ecb7a7621f156bdd0bedc70020e3331a3e9",
        "crazyflow_version": "0.4.0",
        "seeds": seeds,
        "requested_budgets": budgets,
        "protocol": (
            "Continuous training to the largest budget, official linear LR schedule over that "
            "full horizon; each earlier checkpoint is a point on that same run. SAME_STEP "
            "corrected episode handling, timeout bootstrap, reset histories, fresh splines "
            "and configured action noise."
        ),
        "validation": {"seed": 123401, "episodes": 8 if options.smoke else 128},
        "holdout": {"seed": 987651, "episodes": 8 if options.smoke else 256},
        "args": vars(official.Args.create(total_timesteps=max(budgets))),
        "checkpoints": [],
        "training_curve": [],
        "official_trainer_sha256": hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(inspect.getsource(RacingTrainingEnv).encode()).hexdigest(),
        "trajectory_stream_salt": TRAJECTORY_STREAM_SALT,
        "timeout_bootstrap": (
            "truncations-including-terminal"
            if options.legacy_terminal_bootstrap
            else "truncations-and-not-terminations"
        ),
    }

    def save():
        destination.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")

    rng_check = verify_rng_protocol()
    (ASSETS / "racing_fulltrain_20261015_adapter_check.txt").write_text(
        json.dumps(rng_check, indent=2) + "\n"
    )
    results["rng_regression"] = rng_check
    print("RNG_REGRESSION", rng_check, flush=True)
    previous = MODELS / "racing_previous_20261009.ckpt"
    if not previous.exists():
        previous.write_bytes(
            (Path(official.__file__).parent / "ppo_drone_racing.ckpt").read_bytes()
        )
    if options.eval_only:
        results = json.loads(destination.read_text())
    else:
        results["previous_checkpoint_sha256"] = hashlib.sha256(previous.read_bytes()).hexdigest()
        results["previous_validation"] = evaluate(
            previous, 123401, results["validation"]["episodes"], device
        )
        print("PREVIOUS_VALIDATION", results["previous_validation"], flush=True)
        for seed in seeds:
            args = official.Args.create(total_timesteps=max(budgets), seed=seed)
            milestones = {max(1, b // args.batch_size): b for b in budgets}

            def hook(
                agent, optimizer, args, iteration, global_step, envs, explained_var, approx_kl
            ):
                if iteration % 25 == 0 or iteration in milestones:
                    core = envs.unwrapped
                    if iteration == args.num_iterations:
                        results.setdefault("simultaneous_terminal_time_limits_by_seed", {})[
                            str(seed)
                        ] = core.simultaneous_terminal_time_limits
                    results["training_curve"].append(
                        {
                            "seed": seed,
                            "steps": global_step,
                            "completed_episodes": core.episode_count,
                            "simultaneous_terminal_time_limits": (
                                core.simultaneous_terminal_time_limits
                            ),
                            "recent_episode_return": float(np.mean(core.completed_returns[-1024:]))
                            if core.completed_returns
                            else None,
                            "explained_variance": explained_var
                            if math.isfinite(explained_var)
                            else None,
                            "approx_kl": approx_kl,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )
                    save()
                if iteration not in milestones:
                    return
                checkpoint = MODELS / (
                    f"racing{implementation_suffix}_s{seed}_{milestones[iteration]}_20261015.ckpt"
                )
                torch.save(agent.state_dict(), checkpoint)
                validation = evaluate(checkpoint, 123401, results["validation"]["episodes"], device)
                item = {
                    "training_seed": seed,
                    "requested_budget": milestones[iteration],
                    "actual_transitions": global_step,
                    "iterations": iteration,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "validation": validation,
                    "timeout_bootstrap": results["timeout_bootstrap"],
                }
                results["checkpoints"].append(item)
                save()
                print("CHECKPOINT", json.dumps(item), flush=True)

            trainer, source_hash = instrument_trainer(hook, options.legacy_terminal_bootstrap)
            results["instrumented_trainer_sha256"] = source_hash
            started = time.time()
            trainer(args, None, device, "gpu", False)
            results.setdefault("training_seconds", {})[str(seed)] = time.time() - started
            save()
    legacy_path = ASSETS / "racing_fulltrain_20261015_legacy_timeout.json"
    legacy = []
    if not options.legacy_terminal_bootstrap and legacy_path.exists():
        legacy = json.loads(legacy_path.read_text())["checkpoints"]
        results["legacy_timeout_comparison"] = legacy
        results["legacy_timeout_checkpoint_equivalence"] = [
            {
                "training_seed": strict["training_seed"],
                "requested_budget": strict["requested_budget"],
                "strict_sha256": strict["sha256"],
                "legacy_sha256": old["sha256"],
                "identical": strict["sha256"] == old["sha256"],
            }
            for strict in results["checkpoints"]
            for old in legacy
            if (strict["training_seed"], strict["requested_budget"])
            == (old["training_seed"], old["requested_budget"])
        ]
    candidates = (
        results["checkpoints"]
        + legacy
        + [
            {
                "checkpoint": str(previous.relative_to(ROOT)),
                "validation": results["previous_validation"],
                "provenance": "Previous official 2026-10-09 default 1.5 M training run",
            }
        ]
    )
    best = max(candidates, key=lambda item: item["validation"]["return_mean"])
    results["selected"] = {
        "checkpoint": best["checkpoint"],
        "sha256": hashlib.sha256((ROOT / best["checkpoint"]).read_bytes()).hexdigest(),
        "training_seed": best.get("training_seed"),
        "actual_transitions": best.get("actual_transitions"),
        "timeout_bootstrap": best.get("timeout_bootstrap", "official-20261009-baseline"),
        "rule": "Maximum mean return on 128 fixed validation splines, before running holdout.",
    }
    selected = ROOT / best["checkpoint"]
    results["holdout_result"] = evaluate(
        selected,
        987651,
        results["holdout"]["episodes"],
        device,
        save_trajectory=ASSETS / "racing_fulltrain_20261015_holdout.npz",
    )
    results["previous_holdout"] = evaluate(previous, 987651, results["holdout"]["episodes"], device)
    save()
    print("HOLDOUT", results["holdout_result"], flush=True)
    results["course"] = course_evaluate(
        selected,
        827451,
        8 if options.smoke else 64,
        device,
        save_trajectory=ASSETS / "racing_fulltrain_20261015_course.npz",
    )
    results["previous_course"] = course_evaluate(
        previous, 827451, 8 if options.smoke else 64, device
    )
    if not options.smoke:
        results["inertia_randomization_regression"] = verify_inertia_distribution()
        results["stress_courses"] = {}
        for config_name, seed in (("level1.toml", 493517), ("level2.toml", 693517)):
            results["stress_courses"][config_name] = {
                "selected": course_evaluate(selected, seed, 64, device, config_name=config_name),
                "previous": course_evaluate(previous, seed, 64, device, config_name=config_name),
            }
            save()
        results["speed_envelope"] = evaluate_speed_envelope(selected, device)
        save()
        results["fast_course_video"] = render_course(
            ASSETS / "racing_fulltrain_20261015_fast_course.npz",
            "racing_fulltrain_20261015_fast_course",
        )
    if not options.smoke:
        results["course_video"] = render_course(ASSETS / "racing_fulltrain_20261015_course.npz")
    save()
    if not options.smoke:
        plot_results(results)
    print("COURSE", results["course"], flush=True)


if __name__ == "__main__":
    main()
