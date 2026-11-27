"""PPO boundary conditions and portable actor numerical equivalence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.tof.policy import MLP, Moments, act, load_policy, save_policy
from crazyflow.tof.train import advantages

if TYPE_CHECKING:
    from pathlib import Path


def test_time_limit_bootstraps_but_next_episode_cannot_leak_backwards():
    # At t=1 the episode times out with V(terminal)=5; t=2 belongs to a new
    # episode. The timeout gets a bootstrap value but stops GAE recursion.
    rewards = jnp.array([[1.0], [2.0], [3.0]])
    values = jnp.array([[0.2], [0.4], [0.8]])
    next_values = jnp.array([[0.4], [5.0], [0.0]])
    done = jnp.array([[False], [True], [True]])
    adv, returns = advantages(rewards, values, next_values, done, 0.9, 0.95)
    np.testing.assert_allclose(adv[:, 0], [6.3755, 6.1, 2.2], rtol=1e-6)
    np.testing.assert_allclose(returns, adv + values)


def test_moments_merge_matches_population_statistics():
    data = jnp.array([[1.0, 3.0], [2.0, -3.0], [8.0, 4.0], [5.0, 9.0], [-1.0, 2.0]])
    moments = Moments.create(2).update(data[:2]).update(data[2:])
    np.testing.assert_allclose(moments.mean, np.mean(data, axis=0), rtol=1e-6)
    np.testing.assert_allclose(moments.var, np.var(data, axis=0), rtol=1e-6)
    assert float(moments.count) == 5


def test_pickle_free_export_preserves_raw_gaussian_mean(tmp_path: Path) -> None:
    model = MLP((16, 8))
    obs = jax.random.normal(jax.random.PRNGKey(5), (12, 344))
    params = model.init(jax.random.PRNGKey(6), obs)
    moments = Moments.create(344).update(obs)
    expected = model.apply(params, moments.normalize(obs))
    path = tmp_path / "policy.npz"
    save_policy(path, {"hidden_dims": [16, 8]}, params, moments, jnp.zeros(4))
    metadata, portable = load_policy(path)
    assert metadata["normalization"] == "sqrt_var_plus_epsilon"
    np.testing.assert_allclose(act(portable, obs), expected, atol=1e-6)
    with np.load(path, allow_pickle=False) as archive:
        assert json.loads(str(archive["metadata"]))["activation"] == "elu"
        assert all(archive[k].dtype != object for k in archive.files)


def test_actor_noise_and_adam_moments_start_with_strong_float32():
    # Weakly typed logstd used to force a second executable after update one,
    # retaining enough GPU allocations to exceed the requested memory budget.
    from crazyflow.tof.train import make_agent

    agent, _, _ = make_agent(42, 344, 434, {"hidden_dims": [16, 8]})
    for value in (
        agent.train.params["logstd"],
        agent.train.opt_state[1].mu["logstd"],
        agent.train.opt_state[1].nu["logstd"],
    ):
        assert value.dtype == jnp.float32
        assert not jax.typeof(value).weak_type
