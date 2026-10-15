"""Replay frozen official policies through MuJoCo, using held-out seed 10001/world 0.

Example:
    MUJOCO_GL=egl python usage_assets/official_replay_20261015.py \
        --checkpoint trained_models/official_ppo_5_5s_budget2_20261015_seed42.ckpt \
        --scenario holdout_initial_state
No optimization or hardware deployment occurs during replay.
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
import official_policy_20261015 as experiment
from PIL import Image, ImageDraw, ImageFont

from crazyflow.sim.visualize import draw_line, draw_points


def render(
    checkpoint: Path,
    scenario: str,
    output: Path,
    source: Path,
    trajectory_time: float | None = None,
) -> None:
    """Load the recorded training config and replay its predetermined held-out world."""
    experiment.bootstrap(source)
    record_path = experiment.ASSETS / f"{checkpoint.stem}.json"
    record = json.loads(record_path.read_text())
    algorithm = record["algorithm"]
    module_name = f"figure8.train_{algorithm}" if algorithm != "td3" else "learn2fly.train"
    module = importlib.import_module(f"crazyflow_experiments.{module_name}")
    args = module.Args(**record["config"])
    if trajectory_time is None:
        n_envs = record["evaluation"][scenario]["worlds"]
        seed = record["evaluation"][scenario]["seed"]
        evaluation_path = record_path
    else:
        if algorithm == "td3":
            raise ValueError("Reference-period override is for Figure8 policies")
        scenario = f"trained{args.trajectory_time:g}s / test{trajectory_time:g}s"
        evaluation_path = experiment.ASSETS / (
            f"official_speed_sweep_20261015_{algorithm}_T{trajectory_time:g}s.json"
        )
        evaluation = json.loads(evaluation_path.read_text())
        if evaluation["checkpoint_sha256"] != record["checkpoint_sha256"]:
            raise ValueError("Speed sweep belongs to another checkpoint")
        n_envs, seed = evaluation["worlds"], evaluation["seed"]
    if algorithm == "td3":
        period = 0.0 if scenario.endswith("hover") else float(scenario.split("_")[-1][:-1])
        env = (
            module.make_reach_pos_envs(n_envs, args)
            if period == 0
            else module.make_figure8_env(n_envs, args, period)
        )
        agent = module.load_actor(checkpoint)
        seconds, frequency = (7.0 if period == 0 else period), 100
    else:
        ppo = importlib.import_module("crazyflow_experiments.figure8.train_ppo")
        scale = (
            0.2
            if scenario == "holdout_dynamics_0.2"
            else 1.0
            if scenario == "boundary_dynamics_1.0"
            else 0.0
        )
        coefs = module.reward_coefs(args)
        if trajectory_time is not None:
            coefs = {**coefs, "trajectory_time": trajectory_time}
        env = ppo.make_jitted_envs(
            n_envs,
            coefs,
            args.jax_device,
            reset_rotor=True,
            reset_random=trajectory_time is None and not scenario.startswith("validation_"),
            dr_scale=scale,
        )
        if algorithm == "ppo":
            agent = module.load_agent(args, env, checkpoint)
        else:
            import pickle

            agent = module.Agent.create(
                key=jax.random.PRNGKey(0),
                obs_dim=env.single_observation_space.shape[0],
                act_dim=env.single_action_space.shape[0],
                hidden_size=args.hidden_size,
                init_logstd=args.init_logstd,
            )
            payload = pickle.loads(checkpoint.read_bytes())
            agent = agent.replace(actor_states=agent.actor_states.replace(params=payload["actor"]))
        seconds, frequency = 20.0, 50

    raw_file = (
        experiment.ASSETS / f"{checkpoint.stem}_{scenario}.npz"
        if trajectory_time is None
        else evaluation_path.with_suffix(".npz")
    )
    with np.load(raw_file) as raw:
        reference = raw["target"][:, 0].copy()
        recorded_pos = raw["pos"][:, 0].copy()
        recorded_active = raw["active"][:, 0].copy()
    if trajectory_time is not None:
        reference = np.asarray(env.unwrapped.trajectories)[0]
    agent = jax.device_put(agent, jax.devices(args.jax_device)[0])
    env, (obs, _) = env.reset(env, seed=seed)
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
    actual = []
    actions = []
    velocities = []
    quaternions = []
    frame = None
    outcome = "time limit"
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=25, codec="libx264", quality=8) as writer:
        for step in range(round(seconds * frequency)):
            action = np.asarray(agent.mean_action(agent.actor_states.params, obs))[0]
            env, (obs, _, terminated, truncated, info) = step_fn(env, obs)
            pos = np.asarray(env.unwrapped.data.states.pos)[0, 0]
            done = bool(np.asarray(terminated | truncated)[0])
            success = bool(np.asarray(info.get("success", np.zeros(n_envs, dtype=bool)))[0])
            if done:
                completed_horizon = bool(np.asarray(truncated)[0]) and (
                    algorithm != "td3" or not scenario.endswith("hover")
                )
                outcome = (
                    "goal reached"
                    if success
                    else (
                        "out of flight volume"
                        if bool(np.asarray(terminated)[0])
                        else "completed evaluation horizon"
                        if completed_horizon
                        else "time limit"
                    )
                )
                if frame is not None:
                    canvas = ImageDraw.Draw(frame)
                    canvas.rectangle(
                        (0, 400, 800, 455),
                        fill=(25, 90, 55) if success or completed_horizon else (100, 25, 30),
                    )
                    canvas.text((14, 415), outcome, fill="white", font=font)
                    for _ in range(25):
                        writer.append_data(np.asarray(frame))
                break
            actual.append(pos)
            actions.append(action)
            velocities.append(np.asarray(env.unwrapped.data.states.vel)[0, 0])
            quaternions.append(np.asarray(env.unwrapped.data.states.quat)[0, 0])
            if step % (frequency // 25) != 0:
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
            target = (
                env.unwrapped.target_pos
                if hasattr(env.unwrapped, "target_pos")
                else env.unwrapped.goal_pos
            )
            draw_points(
                sim, np.asarray(target)[0], size=0.035, rgba=np.array([0.95, 0.44, 0.12, 1.0])
            )
            frame = Image.fromarray(
                sim.render(
                    mode="rgb_array", world=0, camera=-1, cam_config=camera, width=800, height=528
                )
            )
            canvas = ImageDraw.Draw(frame)
            canvas.rectangle((0, 0, 800, 103), fill=(18, 25, 39))
            canvas.text(
                (14, 8),
                f"Official {algorithm.upper()} | {record['control']} | "
                f"{record['actual_transitions']:,} transitions",
                font=font,
                fill="white",
            )
            error = float(np.linalg.norm(pos - np.asarray(target)[0]))
            canvas.text(
                (14, 38),
                f"{scenario} | t={(step + 1) / frequency:.2f}s | "
                f"position error {error * 1000:.1f} mm",
                font=small,
                fill=(150, 220, 255),
            )
            canvas.text(
                (14, 61),
                f"Frozen policy | held-out seed {seed}, world 0 | actual MuJoCo simulation",
                font=small,
                fill=(215, 215, 225),
            )
            canvas.text(
                (14, 82),
                "Orange: reference / goal     Blue: flown path",
                font=small,
                fill=(215, 215, 225),
            )
            writer.append_data(np.asarray(frame))
    env.close()
    actual = np.asarray(actual)
    valid = recorded_active[: len(actual)]
    max_diff = float(np.max(np.abs(actual[valid] - recorded_pos[: len(actual)][valid])))
    experiment.write_json(
        output.with_name(output.stem + "_replay.json"),
        {
            "checkpoint": str(checkpoint.relative_to(experiment.ROOT)),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "scenario": scenario,
            "seed": seed,
            "world": 0,
            "outcome": outcome,
            "simulation_seconds": len(actual) / frequency,
            "replay_position_max_diff_m": max_diff,
            "evaluation_record": str(evaluation_path.relative_to(experiment.ROOT)),
            "evaluation_raw": str(raw_file.relative_to(experiment.ROOT)),
            "trained_trajectory_time": getattr(args, "trajectory_time", None),
            "test_trajectory_time": trajectory_time,
        },
    )
    np.savez_compressed(
        output.with_name(output.stem + "_replay_states.npz"),
        pos=actual,
        action=np.asarray(actions),
        vel=np.asarray(velocities),
        quat=np.asarray(quaternions),
    )
    print(f"Rendered {output}; outcome={outcome}; replay max difference {max_diff:.3g}m")


def main() -> None:
    """Parse a reproducible frozen-policy replay request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario", default="holdout_initial_state")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trajectory-time", type=float)
    parser.add_argument("--experiments-dir", type=Path, default=Path("/tmp/crazyflow_experiments"))
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output or experiment.ASSETS / f"{checkpoint.stem}_{args.scenario}.mp4"
    if args.trajectory_time is not None and args.output is None:
        record = json.loads((experiment.ASSETS / f"{checkpoint.stem}.json").read_text())
        output = experiment.ASSETS / (
            f"official_speed_sweep_20261015_{record['algorithm']}_T{args.trajectory_time:g}s.mp4"
        )
    render(checkpoint, args.scenario, output, args.experiments_dir, args.trajectory_time)


if __name__ == "__main__":
    main()
