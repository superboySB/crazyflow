"""Independent numerical checks for explicit RSL-aligned PPO semantics."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization, struct

if TYPE_CHECKING:
    from pathlib import Path

from crazyflow.tof.policy import Moments
from crazyflow.tof.train import (
    adaptive_learning_rate,
    gaussian_kl,
    invariant_training_config,
    make_agent,
    make_update,
    normalize_advantages,
    ppo_variant,
    update_normalizer,
)


def test_analytic_kl_matches_hand_computed_old_to_new_gaussian() -> None:
    """A changed mean and scale distinguish the two directions of Gaussian KL."""
    actual = gaussian_kl(
        jnp.array([[0.0]]), jnp.array([0.0]), jnp.array([[1.0]]), jnp.array([np.log(2.0)])
    )
    # KL(N(0,1) || N(1,4)) = log(2) - 1/4. RSL adds1e-5 inside log.
    expected = np.log(2.0 + 1e-5) - 0.25
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    reverse = gaussian_kl(
        jnp.array([[1.0]]), jnp.array([np.log(2.0)]), jnp.array([[0.0]]), jnp.array([0.0])
    )
    np.testing.assert_allclose(reverse, np.log(0.5 + 1e-5) + 2.0, rtol=2e-6)
    assert not np.isclose(float(actual), float(reverse))


def test_analytic_kl_reduces_channels_then_samples() -> None:
    """State-independent exploration scales broadcast over rollout samples."""
    old_mean = jnp.array([[0.0, 1.0], [1.0, 3.0]])
    new_mean = jnp.array([[1.0, 1.0], [-1.0, 2.0]])
    actual = gaussian_kl(
        old_mean, jnp.log(jnp.array([1.0, 2.0])), new_mean, jnp.log(jnp.array([2.0, 1.0]))
    )
    first = np.log(2.0 + 1e-5) - 0.25 + np.log(0.5 + 1e-5) + 1.5
    second = np.log(2.0 + 1e-5) + 0.125 + np.log(0.5 + 1e-5) + 2.0
    np.testing.assert_allclose(actual, (first + second) / 2, rtol=2e-6)


def test_advantage_normalization_uses_whole_rollout_sample_variance() -> None:
    """Time and environment dimensions share one normalization population."""
    advantage = jnp.array([[1.0, 2.0], [4.0, 9.0]])
    actual = normalize_advantages(advantage)
    expected = (np.array([[1.0, 2.0], [4.0, 9.0]]) - 4.0) / np.sqrt(38.0 / 3.0)
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    np.testing.assert_allclose(np.var(actual, ddof=1), 1.0, rtol=2e-6)
    # A per-minibatch centering would incorrectly zero this subset's mean.
    assert abs(float(actual[0].mean())) > 0.5
    np.testing.assert_array_equal(normalize_advantages(jnp.ones((2, 2))), np.zeros((2, 2)))


@pytest.mark.parametrize(
    ("lr", "kl", "expected"),
    [
        (5e-4, 0.03, 5e-4 / 1.5),
        (5e-4, 0.002, 7.5e-4),
        (5e-4, 0.0, 5e-4),
        (5e-4, 0.01, 5e-4),
        (1e-5, 0.03, 1e-5),
        (9e-3, 0.002, 1e-2),
    ],
)
def test_adaptive_schedule_matches_rsl_thresholds(lr: float, kl: float, expected: float) -> None:
    """The schedule respects zero-KL guard, floor and the RSL1e-2 ceiling."""
    np.testing.assert_allclose(adaptive_learning_rate(jnp.array(lr), jnp.array(kl), 0.01), expected)


def test_normalizer_cap_freezes_statistics_but_legacy_keeps_updating() -> None:
    """Crossing the count limit freezes both mean and variance on later batches."""
    initial = Moments.create(2)
    first = update_normalizer(initial, jnp.array([[1.0, 2.0], [3.0, 4.0]]), 2)
    frozen = update_normalizer(first, jnp.array([[100.0, -100.0]]), 2)
    continued = update_normalizer(first, jnp.array([[100.0, -100.0]]), None)
    for actual, expected in zip(jax.tree.leaves(frozen), jax.tree.leaves(first)):
        np.testing.assert_array_equal(actual, expected)
    assert float(continued.count) == 3
    assert not np.allclose(continued.mean, first.mean)


def test_old_resume_config_canonicalizes_to_legacy_only() -> None:
    """Missing historical fields remain reproducible without mixing new semantics."""
    original = {"num_steps": 120, "iterations": 25, "save_interval": 25}
    explicit = {
        **original,
        "ppo_variant": "legacy",
        "normalization_until": None,
        "iterations": 50,
        "save_interval": 50,
    }
    assert invariant_training_config(original) == invariant_training_config(explicit)
    aligned = {**original, "ppo_variant": "rsl_aligned"}
    assert invariant_training_config(original) != invariant_training_config(aligned)
    assert invariant_training_config(aligned)["normalization_until"] == 100_000_000
    with pytest.raises(ValueError, match="ppo_variant"):
        ppo_variant({"ppo_variant": "unknown"})


def test_absent_variant_and_explicit_legacy_initialize_identically() -> None:
    """Old configurations preserve seeded parameters, normalizers and Adam state."""
    implicit, _, _ = make_agent(17, 2, 3, {"hidden_dims": [8]})
    explicit, _, _ = make_agent(17, 2, 3, {"hidden_dims": [8], "ppo_variant": "legacy"})
    for actual, expected in zip(jax.tree.leaves(implicit), jax.tree.leaves(explicit)):
        np.testing.assert_array_equal(actual, expected)


@struct.dataclass
class TinyState:
    """Minimal observation carrier for a real compiled network/Adam update."""

    obs: jax.Array
    critic_obs: jax.Array


class ConstantRewardEnv:
    """Keep transitions deterministic so the first critic update is hand-predictable."""

    num_envs = 2

    def step(self, state: TinyState, _action: jax.Array) -> tuple[TinyState, Any]:
        """Return reward1 and no termination without invoking any simulator."""
        return state, SimpleNamespace(
            reward=jnp.ones(2),
            done=jnp.zeros(2, dtype=bool),
            terminated=jnp.zeros(2, dtype=bool),
            success=jnp.zeros(2, dtype=bool),
            collision=jnp.zeros(2, dtype=bool),
            episode_return=jnp.ones(2),
            episode_length=jnp.ones(2),
            terminal_critic_obs=state.critic_obs,
        )


@pytest.mark.parametrize(("variant", "expected_bias"), [("legacy", 5e-4), ("rsl_aligned", 7.5e-4)])
def test_first_real_update_uses_adapted_lr_only_in_aligned_variant(
    variant: str, expected_bias: float
) -> None:
    """A zero critic and gamma0 give exactly negative unit clipped bias gradient."""
    cfg = {
        "hidden_dims": [8],
        "ppo_variant": variant,
        "num_steps": 2,
        "num_minibatches": 1,
        "update_epochs": 1,
        "learning_rate": 5e-4,
        "gamma": 0.0,
        "entropy_coef": 0.0,
    }
    agent, actor, critic = make_agent(17, 2, 3, cfg)
    # An all-zero actor/critic makes every advantage equal1: the policy gradient
    # is zero, while critic output-bias gradient -2 is globally clipped to -1.
    agent = agent.replace(
        train=agent.train.replace(params=jax.tree.map(jnp.zeros_like, agent.train.params))
    )
    state = TinyState(jnp.zeros((2, 2)), jnp.zeros((2, 3)))
    agent, _, metrics = make_update(ConstantRewardEnv(), actor, critic, cfg)(agent, state)
    actual = agent.train.params["critic"]["params"]["layer_1"]["bias"][0]
    np.testing.assert_allclose(actual, expected_bias, rtol=3e-5, atol=1e-8)
    np.testing.assert_allclose(metrics["lr"], 7.5e-4, rtol=1e-6)
    assert int(agent.train.step) == 1


@pytest.mark.parametrize("variant", ["legacy", "rsl_aligned"])
def test_msgpack_resume_matches_uninterrupted_update(tmp_path: Path, variant: str) -> None:
    """Restore both PPO variants; historical legacy configs omit the new fields."""
    original = {
        "hidden_dims": [8],
        "num_steps": 2,
        "num_minibatches": 1,
        "update_epochs": 1,
        "gamma": 0.0,
    }
    if variant == "rsl_aligned":
        original["ppo_variant"] = variant
    explicit = {
        **original,
        "ppo_variant": variant,
        "normalization_until": 100_000_000 if variant == "rsl_aligned" else None,
    }
    env = ConstantRewardEnv()
    agent, actor, critic = make_agent(21, 2, 3, original)
    state = TinyState(jnp.zeros((2, 2)), jnp.zeros((2, 3)))
    update = make_update(env, actor, critic, original)
    agent, state, _ = update(agent, state)
    path = tmp_path / "checkpoint.msgpack"
    path.write_bytes(serialization.to_bytes((agent, state)))
    template, resumed_actor, resumed_critic = make_agent(21, 2, 3, explicit)
    fresh_state = TinyState(jnp.zeros((2, 2)), jnp.zeros((2, 3)))
    resumed, resumed_state = serialization.from_bytes((template, fresh_state), path.read_bytes())
    expected = update(agent, state)
    actual = make_update(env, resumed_actor, resumed_critic, explicit)(resumed, resumed_state)
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
