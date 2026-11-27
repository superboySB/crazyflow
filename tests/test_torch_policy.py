"""CPU-only interoperability and rejection checks for native RSL PT inference."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

torch = pytest.importorskip("torch")
loader = pytest.importorskip("sim2sim.torch_policy")


@pytest.fixture
def runner() -> Any:
    """Construct the actual pinned RSL modules without a simulator or GPU."""
    modules = pytest.importorskip("rsl_rl.modules")
    normalizers = pytest.importorskip("rsl_rl.modules.normalizer")
    runners = pytest.importorskip("rsl_rl.runners")
    with contextlib.redirect_stdout(io.StringIO()):
        policy = modules.ActorCritic(
            344,
            434,
            4,
            actor_hidden_dims=[256, 256, 128, 64],
            critic_hidden_dims=[256, 256, 128, 64],
            activation="elu",
        )
    native = runners.OnPolicyRunner.__new__(runners.OnPolicyRunner)
    native.alg = SimpleNamespace(policy=policy, rnd=None)
    native.obs_normalizer = normalizers.EmpiricalNormalization(344)
    native.privileged_obs_normalizer = normalizers.EmpiricalNormalization(434)
    native.empirical_normalization = True
    native.cfg = {"empirical_normalization": True}
    native.current_learning_iteration = 0
    return native


def checkpoint(runner: Any, *, actor_only: bool = True) -> dict[str, Any]:
    """Build an independent native tensor payload with detectable raw outputs."""
    state = {
        key: value.clone()
        for key, value in runner.alg.policy.state_dict().items()
        if not actor_only or not key.startswith("critic.")
    }
    state["actor.8.weight"].zero_()
    state["actor.8.bias"] = torch.tensor([2.0, -2.0, 0.25, 0.75])

    def norm(width: int) -> dict[str, Any]:
        return {
            "_mean": torch.full((1, width), 5.0),
            "_var": torch.full((1, width), 4.0),
            "_std": torch.full((1, width), 2.0),
            "count": torch.tensor(1234, dtype=torch.int64),
        }

    result = {
        "model_state_dict": state,
        "obs_norm_state_dict": norm(344),
        "iter": 75,
        "infos": {},
        "crazyflow_metadata": {
            "export_format": "rsl_rl",
            "actor_only": actor_only,
            "obs_dim": 344,
            "critic_obs_dim": 434,
            "hidden_dims": [256, 256, 128, 64],
            "activation": "elu",
            "normalization": "sqrt_var_plus_epsilon",
            "normalization_epsilon": 0.01,
        },
    }
    if not actor_only:
        result["privileged_obs_norm_state_dict"] = norm(434)
    return result


def test_actor_only_native_inference_preserves_rng_and_critic(runner: Any, tmp_path: Path) -> None:
    """Loading changes only deployment tensors and leaves raw outputs unsquashed."""
    payload = checkpoint(runner)
    path = tmp_path / "actor.pt"
    torch.save(payload, path)
    critic_before = {
        key: value.clone()
        for key, value in runner.alg.policy.state_dict().items()
        if key.startswith("critic.")
    }
    rng_before = torch.random.get_rng_state().clone()
    assert loader.load_into_runner(runner, path)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for key, value in critic_before.items():
        assert torch.equal(runner.alg.policy.state_dict()[key], value)
    result = runner.get_inference_policy(device="cpu")(torch.zeros(3, 344))
    torch.testing.assert_close(result, torch.tensor([[2.0, -2.0, 0.25, 0.75]]).expand(3, -1))
    assert runner.current_learning_iteration == 75
    assert runner.obs_normalizer.count.item() == 1234
    assert runner.privileged_obs_normalizer.count.item() == 0


def test_full_export_loads_real_critic_and_its_normalizer(runner: Any, tmp_path: Path) -> None:
    """Future full exports load both networks and distinct observation widths."""
    payload = checkpoint(runner, actor_only=False)
    payload["model_state_dict"]["critic.8.bias"].fill_(7.0)
    path = tmp_path / "actor-critic.pt"
    torch.save(payload, path)
    assert loader.load_into_runner(runner, path)
    assert torch.equal(runner.alg.policy.critic[-1].bias, torch.tensor([7.0]))
    assert runner.privileged_obs_normalizer.count.item() == 1234
    assert runner.privileged_obs_normalizer._mean.shape == (1, 434)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["model_state_dict"].pop("actor.0.weight"),
        lambda p: p["model_state_dict"]["std"].fill_(-1.0),
        lambda p: p["model_state_dict"]["actor.8.bias"].fill_(float("nan")),
        lambda p: p["obs_norm_state_dict"]["_std"].fill_(3.0),
        lambda p: p["obs_norm_state_dict"].update(count=torch.tensor(1234.0)),
        lambda p: p.update(privileged_obs_norm_state_dict=p["obs_norm_state_dict"]),
    ],
)
def test_invalid_exports_rejected_before_runner_changes(
    runner: Any, tmp_path: Path, mutate: Callable[[dict[str, Any]], Any]
) -> None:
    """Malformed state and metadata cannot silently retain incomplete actor weights."""
    payload = checkpoint(runner)
    mutate(payload)
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    before = {key: value.clone() for key, value in runner.alg.policy.state_dict().items()}
    with pytest.raises(ValueError):
        loader.load_into_runner(runner, path)
    for key, value in before.items():
        assert torch.equal(runner.alg.policy.state_dict()[key], value)
    assert runner.obs_normalizer.count.item() == 0


def test_original_baseline_is_delegated_to_native_loader() -> None:
    """The real bundled baseline remains outside the marked-export loading branch."""
    path = Path(__file__).resolve().parents[1] / "models/baseline/model_36825.pt"
    assert loader.read_crazyflow_checkpoint(path) is None
