"""Render the saved Gym policies without training or updating their weights.

Example: python usage_assets/replay_fulltrain_20261015.py --checkpoint trained_models/gym.pt
The displayed episode is always world 0 of the predetermined test seed 31001.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import train_gymnasium as trainer
from PIL import Image, ImageDraw, ImageFont

from crazyflow.sim.visualize import draw_line, draw_points


@torch.no_grad()
def render(
    checkpoint: Path,
    output: Path,
    seed: int = 31001,
    horizontal: bool = False,
    env_backend: str = "fused",
    allow_tf32: bool = True,
) -> None:
    """Record one fixed, unselected held-out episode in MuJoCo."""
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.set_num_threads(4)
    agent, config, keys, _ = trainer.load_checkpoint(checkpoint)
    config.eval_envs = 1024
    if env_backend == "fused":
        from fast_gymnasium_20261015 import make_fused_env

        env = make_fused_env(config, config.eval_envs)
    else:
        env = trainer.make_env(config, config.eval_envs)
    options = None
    if horizontal:
        if config.env_id != "DroneReachVel-v0":
            raise ValueError("Horizontal targets only apply to ReachVel")
        import jax.numpy as jnp

        options = {"vel_min": jnp.array([-1.0, -1.0, 0.0]), "vel_max": jnp.array([1.0, 1.0, 0.0])}
    obs, _ = trainer.reset_env(env, seed, options=options)
    sim = env.unwrapped.sim
    font = ImageFont.truetype("DejaVuSans.ttf", 19)
    small_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    actual = []
    output.parent.mkdir(parents=True, exist_ok=True)
    unit = "m/s" if config.env_id == "DroneReachVel-v0" else "m"
    try:
        with imageio.get_writer(output, fps=25, codec="libx264", quality=8) as writer:
            for step in range(round(config.episode_time * config.freq)):
                action = agent.deterministic_action(trainer.flatten_obs(obs, keys))
                obs, _, terminated, truncated, _ = env.step(action)
                pos = obs["pos"][0].cpu().numpy()
                actual.append(pos)
                if step % 2 == 0:
                    if sim.viewer is not None:
                        sim.viewer.viewer._markers.clear()
                    if len(actual) > 1:
                        draw_line(sim, np.asarray(actual)[::2], rgba=np.array([0.2, 0.6, 1.0, 1.0]))
                    if config.env_id == "DroneFigureEightTrajectory-v0":
                        draw_line(
                            sim,
                            np.asarray(env.unwrapped.trajectory)[::2],
                            rgba=np.array([1.0, 0.4, 0.1, 0.45]),
                        )
                    elif config.env_id != "DroneReachVel-v0":
                        goal = (obs["pos"] + trainer.task_error(obs, config.env_id))[0]
                        draw_points(
                            sim,
                            goal.cpu().numpy()[None],
                            size=0.015,
                            rgba=np.array([1.0, 0.4, 0.1, 0.45]),
                        )
                    if sim.viewer is not None and config.env_id == "DroneReachVel-v0":
                        sim.viewer.viewer.cam.lookat[:] = pos
                    frame = Image.fromarray(
                        sim.render(
                            mode="rgb_array",
                            camera=-1,
                            width=640,
                            height=480,
                            cam_config={
                                "distance": 3.2,
                                "elevation": -20,
                                "azimuth": 90
                                if config.env_id == "DroneFigureEightTrajectory-v0"
                                else 45,
                                "lookat": np.array([0.0, 0.0, 1.0]),
                            },
                        )
                    )
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 640, 81), fill=(19, 24, 35))
                    error = float(
                        torch.linalg.vector_norm(trainer.task_error(obs, config.env_id)[0])
                    )
                    title = config.env_id
                    if config.env_id == "DroneReachVel-v0":
                        title += " | horizontal subset" if horizontal else " | default 3D"
                    draw.text((12, 8), title, font=font, fill="white")
                    draw.text(
                        (12, 35),
                        f"t={(step + 1) / config.freq:.2f}s   error={error:.3f} {unit}",
                        font=small_font,
                        fill=(125, 215, 255),
                    )
                    draw.text(
                        (12, 58),
                        f"Frozen weights | held-out seed {seed} | actual MuJoCo render",
                        font=small_font,
                        fill=(210, 210, 210),
                    )
                    if config.env_id == "DroneReachVel-v0":
                        velocity = obs["vel"][0].cpu().numpy()
                        target = velocity + trainer.task_error(obs, config.env_id)[0].cpu().numpy()
                        draw.rectangle((0, 430, 640, 480), fill=(19, 24, 35))
                        draw.text(
                            (12, 435),
                            f"target velocity: {target.round(3)} m/s",
                            font=small_font,
                            fill=(255, 180, 100),
                        )
                        draw.text(
                            (12, 457),
                            f"actual velocity: {velocity.round(3)} m/s",
                            font=small_font,
                            fill=(125, 215, 255),
                        )
                    writer.append_data(np.asarray(frame))
                if terminated[0] or truncated[0]:
                    if terminated[0]:
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 200, 640, 270), fill=(125, 20, 20))
                        draw.text(
                            (12, 224),
                            "Episode terminated: failure retained",
                            font=font,
                            fill="white",
                        )
                        for _ in range(25):
                            writer.append_data(np.asarray(frame))
                    break
    finally:
        env.close()
    print(
        f"Video: {output}; checkpoint SHA256={hashlib.sha256(checkpoint.read_bytes()).hexdigest()}"
    )


def main() -> None:
    """Parse the frozen checkpoint replay command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=31001)
    parser.add_argument("--horizontal-velocity", action="store_true")
    parser.add_argument("--env-backend", choices=("python", "fused"), default="fused")
    parser.add_argument("--no-tf32", action="store_true")
    args = parser.parse_args()
    output = args.output or Path("usage_assets/fulltrain_20261015") / f"{args.checkpoint.stem}.mp4"
    render(
        args.checkpoint,
        output,
        args.seed,
        args.horizontal_velocity,
        args.env_backend,
        not args.no_tf32,
    )


if __name__ == "__main__":
    main()
