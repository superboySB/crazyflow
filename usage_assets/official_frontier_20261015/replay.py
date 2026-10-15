"""Render a selected dedicated-speed policy in its recorded held-out world.

MUJOCO_GL=egl python usage_assets/official_frontier_20261015/replay.py
Replays fixed seed14001/world0, with no learning or hardware deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import imageio.v2 as imageio
import jax
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from run import DIRECTORY, load_agent, official

from crazyflow.sim.visualize import draw_line, draw_points


def render(checkpoint: Path, output: Path, source: Path) -> None:
    """Replay the frozen actor and compare states against its complete held-out raw record."""
    official.bootstrap(source)
    record_path = DIRECTORY / f"{checkpoint.stem}.json"
    record = json.loads(record_path.read_text())
    evaluation_path = DIRECTORY / f"{checkpoint.stem}_test_position_holdout.json"
    evaluation = json.loads(evaluation_path.read_text())
    if evaluation["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError("Held-out evaluation does not match the frozen checkpoint")
    module = importlib.import_module(f"crazyflow_experiments.figure8.train_{record['algorithm']}")
    args = module.Args(**record["config"])
    ppo = importlib.import_module("crazyflow_experiments.figure8.train_ppo")
    with jax.default_device(jax.devices(args.jax_device)[0]):
        env = ppo.make_jitted_envs(
            evaluation["worlds"],
            module.reward_coefs(args),
            args.jax_device,
            reset_rotor=True,
            reset_random=True,
            dr_scale=0.0,
        )
        agent = load_agent(module, args, env, checkpoint, record["algorithm"])
        env, (obs, _) = env.reset(env, seed=evaluation["seed"])
    raw_path = evaluation_path.with_suffix(".npz")
    with np.load(raw_path) as raw:
        recorded_pos = raw["pos"][:, 0].copy()
        recorded_active = raw["active"][:, 0].copy()
    reference = np.asarray(env.unwrapped.trajectories)[0]
    sim = env.unwrapped.sim
    step_fn = jax.jit(
        lambda state, observation: state.step(
            state, agent.mean_action(agent.actor_states.params, observation)
        )
    )
    camera = {
        "distance": 3.8,
        "elevation": -20.0,
        "azimuth": 118.0,
        "lookat": np.array([0.0, 0.0, 1.2]),
    }
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    small = ImageFont.truetype("DejaVuSans.ttf", 15)
    actual, actions, velocities, quaternions = [], [], [], []
    frame = None
    outcome = "evaluation horizon"
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=25, codec="libx264", quality=8) as writer:
        for step in range(1000):
            action = np.asarray(agent.mean_action(agent.actor_states.params, obs))[0]
            env, (obs, _, terminated, truncated, _) = step_fn(env, obs)
            pos = np.asarray(env.unwrapped.data.states.pos)[0, 0]
            if bool(np.asarray(terminated | truncated)[0]):
                completed = bool(np.asarray(truncated)[0])
                outcome = "completed evaluation horizon" if completed else "out of flight volume"
                if frame is not None:
                    canvas = ImageDraw.Draw(frame)
                    canvas.rectangle(
                        (0, 400, 800, 455), fill=(25, 90, 55) if completed else (100, 25, 30)
                    )
                    canvas.text((14, 415), outcome, fill="white", font=font)
                    for _ in range(25):
                        writer.append_data(np.asarray(frame))
                break
            actual.append(pos)
            actions.append(action)
            velocities.append(np.asarray(env.unwrapped.data.states.vel)[0, 0])
            quaternions.append(np.asarray(env.unwrapped.data.states.quat)[0, 0])
            if step % 2:
                continue
            sim.data = env.unwrapped.data
            if sim.viewer is not None:
                sim.viewer.viewer._markers.clear()
            draw_line(
                sim,
                reference[:: max(1, len(reference) // 180)],
                rgba=np.array([0.95, 0.44, 0.12, 0.9]),
                start_size=2.5,
                end_size=2.5,
            )
            if len(actual) > 1:
                draw_line(
                    sim,
                    np.asarray(actual)[:: max(1, len(actual) // 200)],
                    rgba=np.array([0.1, 0.65, 1.0, 0.9]),
                    start_size=2.0,
                    end_size=2.0,
                )
            target = np.asarray(env.unwrapped.target_pos)[0]
            draw_points(sim, target, size=0.035, rgba=np.array([0.95, 0.44, 0.12, 1.0]))
            frame = Image.fromarray(
                sim.render(
                    mode="rgb_array", world=0, camera=-1, cam_config=camera, width=800, height=528
                )
            )
            canvas = ImageDraw.Draw(frame)
            canvas.rectangle((0, 0, 800, 103), fill=(18, 25, 39))
            canvas.text(
                (14, 8),
                f"Local speed-trained {record['algorithm'].upper()} | "
                f"T={args.trajectory_time:g}s | {record['actual_transitions']:,} transitions",
                font=font,
                fill="white",
            )
            error = float(np.linalg.norm(pos - target))
            canvas.text(
                (14, 38),
                f"t={(step + 1) / 50:.2f}s | position error {error * 1000:.1f} mm",
                font=small,
                fill=(150, 220, 255),
            )
            canvas.text(
                (14, 61),
                f"Frozen policy | held-out seed {evaluation['seed']}, world 0 | MuJoCo simulation",
                font=small,
                fill=(215, 215, 225),
            )
            canvas.text(
                (14, 82),
                "Orange: reference     Blue: flown path     Position randomized +/-0.1 m",
                font=small,
                fill=(215, 215, 225),
            )
            writer.append_data(np.asarray(frame))
    env.close()
    actual = np.asarray(actual)
    valid = recorded_active[: len(actual)]
    max_diff = float(np.max(np.abs(actual[valid] - recorded_pos[: len(actual)][valid])))
    if max_diff > 0.001:
        raise ValueError(f"Replay diverges from fixed held-out raw data: {max_diff}m")
    official.write_json(
        output.with_name(output.stem + "_replay.json"),
        {
            "checkpoint": str(checkpoint.relative_to(official.ROOT)),
            "checkpoint_sha256": record["checkpoint_sha256"],
            "scenario": "position_holdout",
            "seed": evaluation["seed"],
            "world": 0,
            "outcome": outcome,
            "simulation_seconds": len(actual) / 50,
            "replay_position_max_diff_m": max_diff,
            "evaluation_record": str(evaluation_path.relative_to(official.ROOT)),
            "evaluation_raw": str(raw_path.relative_to(official.ROOT)),
            "trained_and_test_reference_period_s": args.trajectory_time,
            "selection": "Validation only; world0 fixed before rendering, no best-world selection.",
        },
    )
    np.savez_compressed(
        output.with_name(output.stem + "_replay_states.npz"),
        pos=actual,
        action=np.asarray(actions),
        vel=np.asarray(velocities),
        quat=np.asarray(quaternions),
    )
    print(f"Rendered {output}; outcome={outcome}; raw comparison {max_diff:.3g}m")


def main() -> None:
    """Render the fastest complete and precise validation-selected model by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=DIRECTORY / "best_dedicated_speed.mp4")
    parser.add_argument("--experiments-dir", type=Path, default=Path("/tmp/crazyflow_experiments"))
    cli = parser.parse_args()
    chosen = json.loads((DIRECTORY / "fastest_reliable_demo.json").read_text())
    checkpoint = (cli.checkpoint or official.ROOT / chosen["checkpoint"]).resolve()
    render(checkpoint, cli.output, cli.experiments_dir)


if __name__ == "__main__":
    main()
