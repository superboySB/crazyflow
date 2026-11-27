"""On-policy JAX/Flax/Optax PPO with batched Crazyflow physics on the same device.

Follows the functional JIT/scan design in the official crazyflow_experiments PPO
used by research. Network, tanh location, observation history and rewards follow
CrazyRL. Gradients are calculated only for the PPO network loss, never physics.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

    from crazyflow.tof.env import EnvState, ToFEnv

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, struct
from flax.training.train_state import TrainState

from crazyflow.tof.config import env_config, load_config
from crazyflow.tof.policy import MLP, Moments, gaussian_logprob, save_policy_pair


@struct.dataclass
class Agent:
    train: TrainState
    actor_norm: Moments
    critic_norm: Moments
    rng: jax.Array
    lr: jax.Array


class Transition(NamedTuple):
    obs: jax.Array
    critic_obs: jax.Array
    actions: jax.Array
    logprob: jax.Array
    values: jax.Array
    rewards: jax.Array
    done: jax.Array
    next_values: jax.Array
    means: jax.Array


def ppo_variant(cfg: dict[str, Any]) -> str:
    """Select explicit semantics; old checkpoint configurations remain legacy."""
    variant = cfg.get("ppo_variant", "legacy")
    if variant not in ("legacy", "rsl_aligned"):
        raise ValueError("ppo_variant must be legacy or rsl_aligned")
    return variant


def invariant_training_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize backwards-compatible defaults before checking a full resume."""
    values = {
        key: value
        for key, value in cfg.items()
        if key not in ("iterations", "save_interval", "log_interval")
    }
    variant = ppo_variant(values)
    values.setdefault("ppo_variant", variant)
    values.setdefault("normalization_until", 100_000_000 if variant == "rsl_aligned" else None)
    return values


def gaussian_kl(
    old_mean: jax.Array, old_logstd: jax.Array, mean: jax.Array, logstd: jax.Array
) -> jax.Array:
    """Old-to-new diagonal Gaussian KL, including the pinned RSL log epsilon."""
    old_std, std = jnp.exp(old_logstd), jnp.exp(logstd)
    return jnp.mean(
        jnp.sum(
            jnp.log(std / old_std + 1e-5)
            + (old_std**2 + (old_mean - mean) ** 2) / (2 * std**2)
            - 0.5,
            axis=-1,
        )
    )


def normalize_advantages(advantage: jax.Array) -> jax.Array:
    """Standardize the complete rollout with RSL's sample standard deviation."""
    return (advantage - advantage.mean()) / (advantage.std(ddof=1) + 1e-8)


def adaptive_learning_rate(lr: jax.Array, kl: jax.Array, desired_kl: float) -> jax.Array:
    """Apply pinned RSL's KL thresholds and learning-rate bounds."""
    return jnp.where(
        kl > desired_kl * 2,
        jnp.maximum(1e-5, lr / 1.5),
        jnp.where((kl < desired_kl / 2) & (kl > 0), jnp.minimum(1e-2, lr * 1.5), lr),
    )


def update_normalizer(moments: Moments, observations: jax.Array, until: int | None) -> Moments:
    """Freeze empirical statistics after the configured sample-count threshold."""
    if until is None:
        return moments.update(observations)
    return jax.lax.cond(
        moments.count < until,
        lambda _: moments.update(observations),
        lambda _: moments,
        operand=None,
    )


def advantages(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    done: jax.Array,
    gamma: float,
    lam: float,
) -> tuple[jax.Array, jax.Array]:
    """GAE: timeout bootstraps terminal state, while all resets stop the recursion."""

    def one(gae: jax.Array, transition: tuple[jax.Array, ...]) -> tuple[jax.Array, jax.Array]:
        reward, value, next_value, terminal = transition
        delta = reward + gamma * next_value - value
        gae = delta + gamma * lam * (1.0 - terminal) * gae
        return gae, gae

    _, adv = jax.lax.scan(
        one, jnp.zeros_like(rewards[0]), (rewards, values, next_values, done), reverse=True
    )
    return adv, adv + values


def make_agent(
    seed: int, actor_dim: int, critic_dim: int, cfg: dict[str, Any]
) -> tuple[Agent, MLP, MLP]:
    """Initialize the actor, asymmetric critic, optimizer and observation moments."""
    hidden = tuple(cfg.get("hidden_dims", [256, 256, 128, 64]))
    actor, critic = MLP(hidden), MLP(hidden, output_dim=1, output_scale=1.0)
    key, k1, k2 = jax.random.split(jax.random.PRNGKey(seed), 3)
    params = {
        "actor": actor.init(k1, jnp.zeros((1, actor_dim))),
        "critic": critic.init(k2, jnp.zeros((1, critic_dim))),
        "logstd": jnp.full((4,), float(cfg.get("init_logstd", 0.0)), dtype=jnp.float32),
    }
    # Scaling by the adaptive learning rate happens after Adam, as in RSL's PPO.
    tx = optax.chain(
        optax.clip_by_global_norm(float(cfg.get("max_grad_norm", 1.0))),
        optax.scale_by_adam(eps=1e-8 if ppo_variant(cfg) == "rsl_aligned" else 1e-5),
        optax.scale(-1.0),
    )
    train = TrainState.create(apply_fn=actor.apply, params=params, tx=tx)
    train = train.replace(step=jnp.array(0, jnp.int32))
    return (
        Agent(
            train,
            Moments.create(actor_dim),
            Moments.create(critic_dim),
            key,
            jnp.array(cfg.get("learning_rate", 5e-4)),
        ),
        actor,
        critic,
    )


def make_update(
    env: ToFEnv, actor: MLP, critic: MLP, cfg: dict[str, Any]
) -> Callable[[Agent, EnvState], tuple[Agent, EnvState, dict[str, jax.Array]]]:
    """Build PPO with explicit RSL-aligned or legacy optimization semantics.

    Both variants deliberately freeze normalization throughout each rollout and
    its gradient epochs. RSL instead updates normalization at every env step.
    """
    aligned = ppo_variant(cfg) == "rsl_aligned"
    normalization_until = cfg.get("normalization_until", 100_000_000 if aligned else None)
    if normalization_until is not None and (
        not isinstance(normalization_until, int) or normalization_until <= 0
    ):
        raise ValueError("normalization_until must be a positive integer or null")
    horizon = int(cfg.get("num_steps", 120))
    minibatches = int(cfg.get("num_minibatches", 32))
    epochs = int(cfg.get("update_epochs", cfg.get("epochs", 5)))
    gamma, lam = float(cfg.get("gamma", 0.99)), float(cfg.get("gae_lambda", 0.95))
    clip = float(cfg.get("clip_coef", cfg.get("clip", 0.2)))
    entropy_coef = float(cfg.get("entropy_coef", 0.005))
    value_coef = float(cfg.get("value_coef", 1.0))
    desired_kl = float(cfg.get("desired_kl", 0.01))
    batch_size = env.num_envs * horizon
    if batch_size % minibatches:
        raise ValueError("num_envs × num_steps must be divisible by num_minibatches")
    if aligned and batch_size < 2:
        raise ValueError("RSL-aligned advantage normalization requires at least two samples")

    def update(agent: Agent, state: EnvState) -> tuple[Agent, EnvState, dict[str, jax.Array]]:
        actor_norm, critic_norm = agent.actor_norm, agent.critic_norm
        params = agent.train.params
        old_logstd = jnp.clip(params["logstd"], -5, 2)

        def collect(
            carry: tuple[EnvState, jax.Array], _
        ) -> tuple[tuple[EnvState, jax.Array], tuple[Transition, jax.Array]]:
            state, key = carry
            key, noise_key = jax.random.split(key)
            mean = actor.apply(params["actor"], actor_norm.normalize(state.obs))
            logstd = jnp.clip(params["logstd"], -5, 2)
            action = mean + jnp.exp(logstd) * jax.random.normal(noise_key, mean.shape)
            logprob = gaussian_logprob(mean, logstd, action)
            value = critic.apply(params["critic"], critic_norm.normalize(state.critic_obs))[..., 0]
            next_state, info = env.step(state, jax.lax.stop_gradient(action))
            # terminal_critic_obs is the pre-reset next state for every world.
            next_value = critic.apply(
                params["critic"], critic_norm.normalize(info.terminal_critic_obs)
            )[..., 0]
            next_value = jnp.where(info.terminated, 0, next_value)
            transition = Transition(
                state.obs,
                state.critic_obs,
                action,
                logprob,
                value,
                info.reward,
                info.done,
                next_value,
                mean,
            )
            statistics = jnp.array(
                [
                    jnp.sum(info.done),
                    jnp.sum(info.success),
                    jnp.sum(info.collision),
                    jnp.sum(jnp.where(info.done, info.episode_return, 0)),
                    jnp.sum(jnp.where(info.done, info.episode_length, 0)),
                ]
            )
            return (next_state, key), (transition, statistics)

        (state, key), (transitions, episode_stats) = jax.lax.scan(
            collect, (state, agent.rng), None, length=horizon
        )
        transitions = jax.tree.map(jax.lax.stop_gradient, transitions)
        adv, returns = advantages(
            transitions.rewards,
            transitions.values,
            transitions.next_values,
            transitions.done,
            gamma,
            lam,
        )
        if aligned:
            adv = normalize_advantages(adv)
        data = (*transitions[:5], adv, returns, transitions.means)
        flat = jax.tree.map(lambda x: x.reshape((batch_size, *x.shape[2:])), data)

        def epoch(
            carry: tuple[TrainState, jax.Array, jax.Array], _
        ) -> tuple[tuple[TrainState, jax.Array, jax.Array], jax.Array]:
            train, key, lr = carry
            key, shuffle_key = jax.random.split(key)
            indices = jax.random.permutation(shuffle_key, batch_size).reshape((minibatches, -1))

            def minibatch(
                carry: tuple[TrainState, jax.Array], indices: jax.Array
            ) -> tuple[tuple[TrainState, jax.Array], jax.Array]:
                train, lr = carry
                obs, critic_obs, actions, old_logprob, old_value, advantage, target, old_mean = (
                    x[indices] for x in flat
                )
                if not aligned:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                def loss_fn(params: dict[str, Any]) -> tuple[jax.Array, tuple[jax.Array, ...]]:
                    mean = actor.apply(params["actor"], actor_norm.normalize(obs))
                    logstd = jnp.clip(params["logstd"], -5, 2)
                    logprob = gaussian_logprob(mean, logstd, actions)
                    logratio = logprob - old_logprob
                    ratio = jnp.exp(logratio)
                    policy_loss = -jnp.mean(
                        jnp.minimum(
                            ratio * advantage, jnp.clip(ratio, 1 - clip, 1 + clip) * advantage
                        )
                    )
                    value = critic.apply(params["critic"], critic_norm.normalize(critic_obs))[
                        ..., 0
                    ]
                    clipped_value = old_value + jnp.clip(value - old_value, -clip, clip)
                    value_loss = jnp.mean(
                        jnp.maximum((value - target) ** 2, (clipped_value - target) ** 2)
                    )
                    entropy = jnp.sum(logstd + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
                    kl = (
                        gaussian_kl(old_mean, old_logstd, mean, logstd)
                        if aligned
                        else jnp.mean((ratio - 1) - logratio)
                    )
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                    return loss, (policy_loss, value_loss, entropy, kl)

                (_, statistics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(train.params)
                if aligned:
                    lr = adaptive_learning_rate(lr, statistics[-1], desired_kl)
                updates, opt_state = train.tx.update(gradients, train.opt_state, train.params)
                params = optax.apply_updates(train.params, jax.tree.map(lambda x: x * lr, updates))
                train = train.replace(step=train.step + 1, params=params, opt_state=opt_state)
                if not aligned:
                    kl = statistics[-1]
                    lr = jnp.where(
                        kl > desired_kl * 2,
                        jnp.maximum(1e-5, lr / 1.5),
                        jnp.where(kl < desired_kl / 2, jnp.minimum(1e-3, lr * 1.5), lr),
                    )
                return (train, lr), jnp.array(statistics)

            (train, lr), losses = jax.lax.scan(minibatch, (train, lr), indices)
            return (train, key, lr), losses.mean(axis=0)

        (train, key, lr), losses = jax.lax.scan(
            epoch, (agent.train, key, agent.lr), None, length=epochs
        )
        agent = agent.replace(
            train=train,
            rng=key,
            lr=lr,
            actor_norm=update_normalizer(actor_norm, transitions.obs, normalization_until),
            critic_norm=update_normalizer(critic_norm, transitions.critic_obs, normalization_until),
        )
        return (
            agent,
            state,
            {
                "episode_stats": episode_stats.sum(axis=0),
                "losses": losses.mean(axis=0),
                "mean_step_reward": transitions.rewards.mean(),
                "lr": lr,
                "action_std": jnp.exp(jnp.clip(train.params["logstd"], -5, 2)).mean(),
            },
        )

    return jax.jit(update, donate_argnums=(0, 1))


def source_hash() -> str:
    """Hash the task, simulator, dynamics and controller source used for this run."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for folder in (root / "tof", root / "dynamics", root / "sim", root / "control"):
        for path in sorted(folder.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    """Train PPO, save portable actors, and preserve complete resume checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--warm-start", help="Load PPO optimizer/policy/normalizers; start fresh environments"
    )
    parser.add_argument("--resume", help="Full training checkpoint.msgpack, including Adam and RNG")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--benchmark", action="store_true", help="Skip eval; measure full PPO updates"
    )
    args = parser.parse_args()
    if args.resume and args.warm_start:
        parser.error("Choose either --resume or --warm-start")
    raw = load_config(args.config)
    cfg = dict(raw.get("training", {}))
    cfg.setdefault("ppo_variant", "legacy")
    ppo_variant(cfg)
    environment = env_config(raw.get("environment", {}))
    for key in ("num_envs", "iterations", "num_steps", "seed"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.smoke:
        cfg.update(num_envs=32, iterations=3, num_steps=16, num_minibatches=4, update_epochs=2)
    num_envs = cfg.get("num_envs", 8192 if jax.default_backend() == "gpu" else 32)
    if isinstance(num_envs, dict):
        num_envs = num_envs["gpu" if jax.default_backend() == "gpu" else "cpu"]
    cfg["num_envs"] = int(num_envs)
    iterations = int(cfg.get("iterations", 1000))
    seed = int(cfg.get("seed", 42))
    if min(int(num_envs), iterations, int(cfg.get("num_steps", 120))) < 1:
        parser.error("num-envs, iterations and num-steps must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    from crazyflow.tof.env import ToFEnv

    resume_payload = (
        serialization.msgpack_restore(Path(args.resume).read_bytes()) if args.resume else None
    )
    simulation_config = environment
    if resume_payload is not None:
        simulation_config = dataclasses.replace(
            environment,
            model_params_path=None,
            model_params=resume_payload["env_runtime"]["config"]["model_params"],
        )
    env = ToFEnv(
        simulation_config,
        num_envs=int(num_envs),
        device="gpu" if jax.default_backend() == "gpu" else "cpu",
    )
    agent, actor, critic = make_agent(
        seed, environment.observation_size, environment.critic_observation_size, cfg
    )
    state = env.reset(jax.random.PRNGKey(seed + 1))
    # Seed empirical moments with an untrained random rollout, including motion and
    # action history. An initial stationary frame has zero variance in those inputs.
    # This calibration collects no policy gradients and is excluded from PPO steps.
    if not args.resume and not args.warm_start:

        @jax.jit
        def calibrate(
            state: EnvState, actor_norm: Moments, critic_norm: Moments, key: jax.Array
        ) -> tuple[EnvState, Moments, Moments, jax.Array]:
            def one(
                carry: tuple[EnvState, Moments, Moments, jax.Array], _
            ) -> tuple[tuple[EnvState, Moments, Moments, jax.Array], None]:
                state, actor_norm, critic_norm, key = carry
                key, subkey = jax.random.split(key)
                state, _ = env.step(state, jax.random.normal(subkey, (int(num_envs), 4)))
                return (
                    state,
                    actor_norm.update(state.obs),
                    critic_norm.update(state.critic_obs),
                    key,
                ), None

            return jax.lax.scan(
                one,
                (state, actor_norm, critic_norm, key),
                None,
                length=int(cfg.get("num_steps", 120)),
            )[0]

        state, anorm, cnorm, key = calibrate(state, agent.actor_norm, agent.critic_norm, agent.rng)
        agent = agent.replace(actor_norm=anorm, critic_norm=cnorm, rng=key)
    start_iteration = 0
    warm_start = None
    if args.warm_start:
        restored = serialization.msgpack_restore(Path(args.warm_start).read_bytes())
        agent = serialization.from_state_dict(agent, restored["agent"])
        previous = restored["training_config"]
        previous_envs = previous["num_envs"]
        if isinstance(previous_envs, dict):
            previous_envs = previous_envs["gpu" if jax.default_backend() == "gpu" else "cpu"]
        warm_start = {
            "checkpoint_sha256": hashlib.sha256(Path(args.warm_start).read_bytes()).hexdigest(),
            "iterations": int(restored["iteration"]),
            "transitions": int(restored["iteration"]) * int(previous_envs) * previous["num_steps"],
        }
    if args.resume:
        restored = resume_payload
        warm_start = restored.get("warm_start")
        agent = serialization.from_state_dict(agent, restored["agent"])
        state = serialization.from_state_dict(state, restored["env_state"])
        env.restore_runtime(restored["env_runtime"])
        start_iteration = int(restored["iteration"])

        if restored["env_config"] != environment.to_dict() or invariant_training_config(
            restored["training_config"]
        ) != invariant_training_config(cfg):
            raise ValueError(
                "Resume requires identical environment, batch size and PPO configuration"
            )
        if start_iteration >= iterations:
            raise ValueError("Resume target --iterations must exceed the saved iteration")
    update = make_update(env, actor, critic, cfg)
    batch = int(num_envs) * int(cfg.get("num_steps", 120))
    metadata = {
        "env_config": dataclasses.asdict(environment),
        "training_config": cfg,
        "hidden_dims": cfg.get("hidden_dims", [256, 256, 128, 64]),
        "obs_dim": environment.observation_size,
        "critic_obs_dim": environment.critic_observation_size,
        "seed": seed,
        "num_envs": int(num_envs),
        "device": str(jax.devices()),
        "source_sha256": source_hash(),
        "algorithm": "PPO",
        "warm_start": warm_start,
        "physics_gradients": False,
        "research_commit": "595b426",
        "official_experiments_commit": "203ad4e31b9380fec985f17008bb7a68ad5cf768",
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "devices": metadata["device"],
                "envs": num_envs,
                "transitions_per_update": batch,
                "iterations": iterations,
            }
        ),
        flush=True,
    )
    durations = []
    t0 = time.perf_counter()
    save_every = int(cfg.get("save_interval", 25))
    with (output / "training.jsonl").open("a") as log:
        for iteration in range(start_iteration, iterations):
            begin = time.perf_counter()
            agent, state, metrics = update(agent, state)
            host = jax.device_get(metrics)
            state, changed = env.maybe_regenerate(state)
            if changed:
                update = make_update(env, actor, critic, cfg)
                print(
                    json.dumps(
                        {"map_maintenance_iteration": iteration + 1, "runtime": env.runtime_state()}
                    ),
                    flush=True,
                )
            duration = time.perf_counter() - begin
            durations.append(duration)
            episodes, successes, collisions, returns, lengths = host["episode_stats"]
            p_loss, v_loss, entropy, kl = host["losses"]
            record = {
                "iteration": iteration + 1,
                "transitions": (iteration + 1) * batch,
                "seconds": time.perf_counter() - t0,
                "update_seconds": duration,
                "samples_per_second": batch / duration,
                "episodes": int(episodes),
                "success_rate": float(successes / max(episodes, 1)),
                "collision_rate": float(collisions / max(episodes, 1)),
                "episode_return": float(returns / max(episodes, 1)),
                "episode_length": float(lengths / max(episodes, 1)),
                "step_reward": float(host["mean_step_reward"]),
                "policy_loss": float(p_loss),
                "value_loss": float(v_loss),
                "entropy": float(entropy),
                "kl": float(kl),
                "lr": float(host["lr"]),
                "action_std": float(host["action_std"]),
            }
            if not all(np.isfinite(x) for x in record.values()):
                raise FloatingPointError(f"Non-finite training metrics: {record}")
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if (iteration + 1) % save_every == 0 or iteration + 1 == iterations:
                meta = {
                    **metadata,
                    "iteration": iteration + 1,
                    "transitions": (iteration + 1) * batch,
                    "env_config": env.effective_config(),
                    "env_runtime": env.runtime_state(),
                }
                save_policy_pair(
                    output,
                    meta,
                    agent.train.params["actor"],
                    agent.actor_norm,
                    agent.train.params["logstd"],
                    critic_params=agent.train.params["critic"],
                    critic_moments=agent.critic_norm,
                )
                if not args.benchmark:
                    for suffix in ("npz", "pt"):
                        snapshot = output / "policies" / f"policy_{iteration + 1:06d}.{suffix}"
                        snapshot.parent.mkdir(exist_ok=True)
                        shutil.copy2(output / f"policy.{suffix}", snapshot)
                    payload = {
                        "agent": serialization.to_state_dict(agent),
                        "env_state": serialization.to_state_dict(state),
                        "env_runtime": env.runtime_state(),
                        "iteration": iteration + 1,
                        "env_config": environment.to_dict(),
                        "training_config": cfg,
                        "warm_start": warm_start,
                    }
                    temporary = output / "checkpoint.tmp"
                    temporary.write_bytes(serialization.msgpack_serialize(payload))
                    temporary.replace(output / "checkpoint.msgpack")
    result = {
        **metadata,
        "iterations": iterations,
        "transitions": iterations * batch,
        "wall_seconds": time.perf_counter() - t0,
        "first_update_seconds": durations[0],
        "steady_update_seconds": float(np.median(durations[1:] or durations)),
        "steady_samples_per_second": batch / float(np.median(durations[1:] or durations)),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if not args.benchmark:
        import matplotlib.pyplot as plt

        records = [
            json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()
        ]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, key in zip(
            axes.flat, ["episode_return", "success_rate", "collision_rate", "value_loss"]
        ):
            ax.plot([r["transitions"] for r in records], [r[key] for r in records])
            ax.set(xlabel="Transitions", ylabel=key)
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "training.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
