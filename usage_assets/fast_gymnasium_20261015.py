"""Fused Gymnasium step used by the independent-goal full training study.

The full study runner selects this backend explicitly and records its source hash per run.
It traces the official wrapper and environment step together, preserving action normalization,
rewards, dynamics, goal RNG,
NEXT_STEP behavior and truncation. Figure-eight's constant lookup arrays are placed on JAX before
tracing. Float32 fusion can change rounding; all states/observations/rewards are checked explicitly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import torch
from gymnasium.wrappers.vector import JaxToTorch
from train_gymnasium import (
    ENVIRONMENT_IDS,
    Agent,
    Config,
    flatten_obs,
    observation_keys,
    reset_env,
    restore_environment,
    snapshot_environment,
)

from crazyflow.envs import NormalizeActions


class FusedGymStep(gymnasium.vector.VectorWrapper):
    """Trace the official JAX environment's step with explicit state inputs/outputs."""

    def __init__(self, env: gymnasium.vector.VectorEnv):
        """Wrap an already normalized JAX environment, before the JaxToTorch bridge."""
        super().__init__(env)
        base = env.unwrapped
        self._state_names = ["_marked_for_reset"] + [
            name for name in ("jax_key", "_goal", "_goal_vel") if hasattr(base, name)
        ]
        # NumPy cannot index a lookup table with a traced JAX index. These arrays are
        # immutable constants; both paths perform the same float32 trajectory arithmetic.
        for name in ("trajectory", "sample_offsets"):
            if hasattr(base, name):
                setattr(base, name, jax.device_put(jnp.asarray(getattr(base, name)), base.device))
        self._compiled_step = jax.jit(self._trace_step)

    def _trace_step(self, data: object, state: tuple, action: jax.Array) -> tuple:
        """Trace official methods with temporary fields and restore all Python state afterwards."""
        base = self.env.unwrapped
        saved_data = base.sim.data
        saved_state = tuple(getattr(base, name) for name in self._state_names)
        try:
            base.sim.data = data
            for name, value in zip(self._state_names, state, strict=True):
                setattr(base, name, value)
            outputs = self.env.step(action)
            return base.sim.data, tuple(getattr(base, name) for name in self._state_names), outputs
        finally:
            base.sim.data = saved_data
            for name, value in zip(self._state_names, saved_state, strict=True):
                setattr(base, name, value)

    def step(self, action: jax.Array) -> tuple:
        """Execute one fused step and expose the resulting state through the official object."""
        base = self.env.unwrapped
        state = tuple(getattr(base, name) for name in self._state_names)
        data, state, outputs = self._compiled_step(base.sim.data, state, action)
        base.sim.data = data
        for name, value in zip(self._state_names, state, strict=True):
            setattr(base, name, value)
        return outputs


def make_raw(config: Config, count: int, device: str, fused: bool) -> gymnasium.vector.VectorEnv:
    """Construct an official normalized environment with optional whole-step fusion."""
    kwargs = dict(
        num_envs=count, freq=config.freq, max_episode_time=config.episode_time, device=device
    )
    if config.env_id == "DroneFigureEightTrajectory-v0":
        kwargs.update(
            trajectory_time=config.episode_time,
            n_samples=config.n_samples,
            samples_dt=config.samples_dt,
        )
    env = NormalizeActions(gymnasium.make_vec(config.env_id, **kwargs))
    return FusedGymStep(env) if fused else env


def make_fused_env(config: Config, num_envs: int) -> gymnasium.vector.VectorEnv:
    """Drop-in make_env(config, num_envs) for explicitly configured CUDA workers."""
    return JaxToTorch(make_raw(config, num_envs, "gpu", True), device=torch.device("cuda"))


def host_leaf(value: jax.Array) -> np.ndarray:
    """Convert ordinary arrays and typed PRNG keys to comparable host buffers."""
    if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
        value = jax.random.key_data(value)
    return np.asarray(value)


def verify(env_id: str, device: str, steps: int) -> dict:
    """Compare every transition, all simulator leaves, task RNG and observations across resets."""
    config = Config(env_id=env_id, episode_time=0.12)
    reference = make_raw(config, 16, device, False)
    fused = make_raw(config, 16, device, True)
    maximum, terminations, truncations = 0.0, 0, 0
    try:
        reset_env(reference, 123)
        base = reference.unwrapped
        # Force one immediate floor termination while other worlds reach time limits.
        pos = base.sim.data.states.pos.at[0, 0, 2].set(0.001)
        vel = base.sim.data.states.vel.at[0, 0, 2].set(-1.0)
        base.sim.data = base.sim.data.replace(states=base.sim.data.states.replace(pos=pos, vel=vel))
        restore_environment(fused, snapshot_environment(reference))
        actions = np.random.default_rng(123).uniform(-0.15, 0.15, (steps, 16, 4)).astype(np.float32)
        actions[..., -1] = 0.22067 + 0.1 * actions[..., -1]
        for action in actions:
            a = reference.step(jax.device_put(action, base.device))
            b = fused.step(jax.device_put(action, base.device))
            pairs = list(zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))
            pairs += list(
                zip(
                    jax.tree.leaves(reference.unwrapped.sim.data),
                    jax.tree.leaves(fused.unwrapped.sim.data),
                    strict=True,
                )
            )
            for name in ("_marked_for_reset", "jax_key", "_goal", "_goal_vel"):
                if hasattr(base, name):
                    pairs.append((getattr(base, name), getattr(fused.unwrapped, name)))
            for left, right in pairs:
                left, right = host_leaf(left), host_leaf(right)
                if left.dtype.kind in "biu":
                    np.testing.assert_array_equal(left, right)
                else:
                    np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-6)
                    maximum = max(maximum, float(np.max(np.abs(left - right))))
            terminations += int(np.asarray(a[2]).sum())
            truncations += int(np.asarray(a[3]).sum())
        assert terminations > 0 and truncations > 0
        return dict(
            env_id=env_id,
            device=device,
            steps=steps,
            worlds=16,
            maximum_absolute_float_difference=maximum,
            exact_rng_and_flags=True,
            terminations=terminations,
            truncations=truncations,
        )
    finally:
        reference.close()
        fused.close()


def benchmark(count: int, steps: int, fused: bool) -> dict:
    """Measure synchronized env-only, realistic rollout and four-epoch PyTorch update costs."""
    config = Config(env_id="DroneReachPos-v0", num_envs=count, position_bc_steps=0)
    env = JaxToTorch(make_raw(config, count, "gpu", fused), device=torch.device("cuda"))
    keys = observation_keys(config.env_id)
    obs, _ = reset_env(env, 456)
    flat = flatten_obs(obs, keys)
    agent = Agent(flat.shape[-1], 4).cuda()
    action = torch.zeros((count, 4), device="cuda")
    action[:, -1] = 0.22067
    for _ in range(5):
        obs, *_ = env.step(action)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        obs, *_ = env.step(action)
    torch.cuda.synchronize()
    env_seconds = time.perf_counter() - start
    obs, _ = reset_env(env, 456)
    flat = flatten_obs(obs, keys)
    buffers = [[], [], []]
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        with torch.no_grad():
            action, logprob, _, _ = agent.get_action_and_value(flat)
        buffers[0].append(flat)
        buffers[1].append(action)
        buffers[2].append(logprob)
        obs, reward, terminated, truncated, _ = env.step(action)
        flat = flatten_obs(obs, keys)
        # Match the current trainer's host synchronization for boundary checks and error logging.
        bool((terminated | truncated).any())
        bool(truncated.any())
        torch.linalg.vector_norm(obs["difference_to_goal"], dim=-1).sum().item()
    torch.cuda.synchronize()
    rollout_seconds = time.perf_counter() - start
    batch_obs, batch_actions, batch_logprob = [torch.cat(values) for values in buffers]
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    indices = torch.arange(len(batch_obs), device="cuda")
    advantages = torch.randn(len(batch_obs), device="cuda")
    returns = torch.randn(len(batch_obs), device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(4):
        for index in torch.tensor_split(indices, 16):
            _, logprob, entropy, value = agent.get_action_and_value(
                batch_obs[index], batch_actions[index]
            )
            ratio = (logprob - batch_logprob[index]).exp()
            loss = -torch.minimum(
                ratio * advantages[index], ratio.clamp(0.8, 1.2) * advantages[index]
            ).mean()
            loss += (
                0.5 * (value.flatten() - returns[index]).square().mean() - 0.002 * entropy.mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
    torch.cuda.synchronize()
    ppo_seconds = time.perf_counter() - start
    env.close()
    return dict(
        fused=fused,
        worlds=count,
        rollout_steps=steps,
        env_only_seconds=env_seconds,
        env_only_slots_per_second=count * steps / env_seconds,
        rollout_including_inference_and_boundary_sync_seconds=rollout_seconds,
        ppo_four_epochs_seconds=ppo_seconds,
        rollout_plus_ppo_slots_per_second=count * steps / (rollout_seconds + ppo_seconds),
        env_fraction_of_rollout=env_seconds / rollout_seconds,
        ppo_fraction_of_rollout_plus_ppo=ppo_seconds / (rollout_seconds + ppo_seconds),
    )


def main() -> None:
    """Validate fusion first and then run a bounded profile without changing live workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--verify-steps", type=int, default=64)
    parser.add_argument("--benchmark-envs", type=int, default=16)
    parser.add_argument("--benchmark-steps", type=int, default=64)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = {
        "adapter_used_by_formal_study": False,
        "conditions": (
            "Shared GPU with ongoing workers; diagnostic wall times, not hardware rankings."
        ),
        "verification": [
            verify(env_id, args.verify_device, args.verify_steps) for env_id in ENVIRONMENT_IDS
        ],
    }
    if not args.skip_benchmark:
        report["timing"] = [
            benchmark(args.benchmark_envs, args.benchmark_steps, fused) for fused in (False, True)
        ]
        a, b = report["timing"]
        report["env_speedup"] = a["env_only_seconds"] / b["env_only_seconds"]
        report["rollout_plus_ppo_speedup"] = (
            a["rollout_including_inference_and_boundary_sync_seconds"]
            + a["ppo_four_epochs_seconds"]
        ) / (
            b["rollout_including_inference_and_boundary_sync_seconds"]
            + b["ppo_four_epochs_seconds"]
        )
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
