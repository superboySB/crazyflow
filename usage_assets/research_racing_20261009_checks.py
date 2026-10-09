"""Reproduce the racing JAX-to-Torch CPU conversion checks inside the research Docker image.

Run from the Crazyflow checkout:
    python usage_assets/research_racing_20261009_checks.py

The image supplies the pinned, patched lsy_drone_racing package. Both CPU and CUDA JAX
backends are checked; policy tensors stay on CPU in both cases.
"""

import torch
from lsy_drone_racing.control.train_rl import make_envs


def main() -> None:
    """Check CPU policy tensors from both JAX backends."""
    for jax_device in ("cpu", "gpu"):
        env = make_envs(
            num_envs=2, jax_device=jax_device, torch_device=torch.device("cpu"), coefs={"n_obs": 2}
        )
        try:
            obs, _ = env.reset(seed=42)
            assert isinstance(obs, torch.Tensor) and obs.device.type == "cpu"
            assert obs.shape == (2, *env.single_observation_space.shape)
            assert torch.isfinite(obs).all()
            action = torch.zeros(env.action_space.shape)
            obs, reward, terminated, truncated, _ = env.step(action)
            assert isinstance(reward, torch.Tensor) and reward.device.type == "cpu"
            assert reward.shape == terminated.shape == truncated.shape == (2,)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            print(
                f"JAX {jax_device} -> Torch CPU passed: "
                f"obs={tuple(obs.shape)}, reward={tuple(reward.shape)}",
                flush=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
