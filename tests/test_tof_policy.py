"""PPO boundary conditions and portable actor numerical equivalence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import unfreeze

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


@pytest.mark.parametrize("include_critic", [False, True], ids=["actor_only", "trained_critic"])
def test_torch_export_preserves_native_weights_normalizers_and_raw_outputs(
    tmp_path: Path, include_critic: bool
) -> None:
    """Load a real weights-only Torch file, including both truthful export variants."""
    import torch
    from torch.nn import functional as functional

    from crazyflow.tof.policy import export_torch_policy

    hidden = (16, 8)
    actor, critic = MLP(hidden), MLP(hidden, output_dim=1, output_scale=1.0)
    obs = jax.random.normal(jax.random.PRNGKey(75), (13, 344)) * 5
    critic_obs = jax.random.normal(jax.random.PRNGKey(76), (13, 434)) * 3
    actor_params = unfreeze(actor.init(jax.random.PRNGKey(77), obs))
    # Means outside [-1,1] make an accidental output tanh plainly detectable.
    actor_params["params"]["layer_2"]["bias"] = jnp.array([0.4, -1.2, 2.3, -3.4])
    critic_params = critic.init(jax.random.PRNGKey(78), critic_obs)
    moments = Moments.create(344).update(obs)
    critic_moments = Moments.create(434).update(critic_obs)
    logstd = jnp.array([-6.0, 0.1, 1.0, 3.0])
    metadata = {
        "hidden_dims": list(hidden),
        "obs_dim": 344,
        "critic_obs_dim": 434,
        "iteration": 75,
        "activation": "elu",
        "env_config": {"integrator": "rk4", "model_params": {"mass": 0.0434}},
    }
    source = tmp_path / "policy.npz"
    save_policy(source, metadata, actor_params, moments, logstd)
    source_bytes = source.read_bytes()
    exported = tmp_path / "policy.pt"
    export_torch_policy(
        exported,
        metadata,
        actor_params,
        moments,
        logstd,
        critic_params=critic_params if include_critic else None,
        critic_moments=critic_moments if include_critic else None,
        source_npz_path=source,
    )
    payload = torch.load(exported, map_location="cpu", weights_only=True)
    assert source.read_bytes() == source_bytes
    exported_metadata = payload["crazyflow_metadata"]
    assert exported_metadata["actor_only"] is not include_critic
    assert exported_metadata["export_format"] == "rsl_rl"
    assert exported_metadata["source_npz_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert payload["iter"] == 75
    assert payload["infos"] == {}
    weights = payload["model_state_dict"]
    expected_keys = {"std"}
    modules = {"actor": actor_params}
    if include_critic:
        modules["critic"] = critic_params
    for prefix, params in modules.items():
        for index in range(len(hidden) + 1):
            for field, original in (
                ("weight", params["params"][f"layer_{index}"]["kernel"].T),
                ("bias", params["params"][f"layer_{index}"]["bias"]),
            ):
                name = f"{prefix}.{index * 2}.{field}"
                expected_keys.add(name)
                assert weights[name].dtype == torch.float32
                assert weights[name].device.type == "cpu"
                np.testing.assert_array_equal(weights[name].numpy(), np.asarray(original))
    assert set(weights) == expected_keys
    np.testing.assert_allclose(
        weights["std"].numpy(), np.exp(np.clip(np.asarray(logstd), -5, 2)), rtol=1e-6
    )

    def check_normalizer(name: str, reference: Moments, width: int) -> None:
        normalizer = payload[name]
        assert set(normalizer) == {"_mean", "_var", "_std", "count"}
        for field in ("_mean", "_var", "_std"):
            assert normalizer[field].shape == (1, width)
            assert normalizer[field].dtype == torch.float32
            assert normalizer[field].device.type == "cpu"
        np.testing.assert_array_equal(normalizer["_mean"].numpy()[0], reference.mean)
        np.testing.assert_array_equal(normalizer["_var"].numpy()[0], reference.var)
        np.testing.assert_allclose(normalizer["_std"].numpy()[0], np.sqrt(reference.var), rtol=1e-6)
        assert normalizer["count"].shape == ()
        assert normalizer["count"].dtype == torch.int64
        assert int(normalizer["count"]) == int(reference.count)

    check_normalizer("obs_norm_state_dict", moments, 344)
    assert ("privileged_obs_norm_state_dict" in payload) is include_critic
    if include_critic:
        check_normalizer("privileged_obs_norm_state_dict", critic_moments, 434)

    def forward(prefix: str, observations: jax.Array, normalizer_name: str) -> np.ndarray:
        normalizer = payload[normalizer_name]
        value = torch.from_numpy(np.array(observations, copy=True))
        value = (value - normalizer["_mean"]) / (normalizer["_std"] + 0.01)
        for index in range(len(hidden) + 1):
            value = functional.linear(
                value,
                weights[f"{prefix}.{index * 2}.weight"],
                weights[f"{prefix}.{index * 2}.bias"],
            )
            if index < len(hidden):
                value = functional.elu(value)
        return value.numpy()

    expected = actor.apply(actor_params, moments.normalize(obs))
    actual = forward("actor", obs, "obs_norm_state_dict")
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    assert np.max(np.abs(actual)) > 3
    if include_critic:
        np.testing.assert_allclose(
            forward("critic", critic_obs, "privileged_obs_norm_state_dict"),
            critic.apply(critic_params, critic_moments.normalize(critic_obs)),
            atol=2e-6,
            rtol=2e-6,
        )


def test_training_savepoints_keep_npz_and_torch_snapshots_paired(tmp_path: Path) -> None:
    """Two real CPU PPO saves must produce corresponding latest and snapshot files."""
    import torch

    config = tmp_path / "training.yaml"
    config.write_text(
        """training:
  iterations: 2
  num_envs: 4
  num_steps: 16
  num_minibatches: 4
  update_epochs: 1
  hidden_dims: [16, 8]
  seed: 471
  save_interval: 1
environment:
  num_maps: 1
  domain_randomization: false
  observation_noise: false
"""
    )
    output = tmp_path / "trained"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crazyflow.tof.train",
            "--config",
            str(config),
            "--output",
            str(output),
        ],
        env=os.environ
        | {
            "JAX_PLATFORMS": "cpu",
            "MUJOCO_GL": "osmesa",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "checkpoint.msgpack").is_file()
    for stem, iteration in (
        ("policy", 2),
        ("policies/policy_000001", 1),
        ("policies/policy_000002", 2),
    ):
        npz, pt = output / f"{stem}.npz", output / f"{stem}.pt"
        assert npz.is_file() and pt.is_file()
        payload = torch.load(pt, map_location="cpu", weights_only=True)
        assert payload["iter"] == iteration
        assert not payload["crazyflow_metadata"]["actor_only"]
        assert (
            payload["crazyflow_metadata"]["source_npz_sha256"]
            == hashlib.sha256(npz.read_bytes()).hexdigest()
        )
        assert "privileged_obs_norm_state_dict" in payload
        assert any(key.startswith("critic.") for key in payload["model_state_dict"])


def test_failed_torch_export_keeps_the_previous_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A conversion error must not publish a new NPZ beside an older PT."""
    from crazyflow.tof import policy

    (tmp_path / "policy.npz").write_bytes(b"previous-npz")
    (tmp_path / "policy.pt").write_bytes(b"previous-pt")

    def stage_npz(path: Path, *_args: object) -> None:
        path.write_bytes(b"new-npz")

    def failed_export(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected conversion failure")

    monkeypatch.setattr(policy, "save_policy", stage_npz)
    monkeypatch.setattr(policy, "export_torch_policy", failed_export)
    with pytest.raises(RuntimeError, match="injected conversion failure"):
        policy.save_policy_pair(
            tmp_path,
            {},
            {},
            Moments.create(1),
            jnp.zeros(4),
            critic_params={},
            critic_moments=Moments.create(1),
        )
    assert (tmp_path / "policy.npz").read_bytes() == b"previous-npz"
    assert (tmp_path / "policy.pt").read_bytes() == b"previous-pt"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["policy.npz", "policy.pt"]
