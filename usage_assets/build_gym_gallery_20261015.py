"""Build reproducible frozen-policy movies, first-episode checks, posters and gallery.

Run during or after training: python usage_assets/build_gym_gallery_20261015.py watch.
The validation-only summary chooses checkpoints. Every replay uses holdout seed 31001,
world zero of 1024; no world or weight is chosen for appearance. The optional horizontal
velocity panel changes only target vz to zero and is explicitly labelled separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium
    from torch import Tensor
    from train_gymnasium import Config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "usage_assets/fulltrain_20261015"


def sha(p: str | Path) -> str:
    """Hash an artifact without modifying it."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p: str | Path, d: dict) -> None:
    """Atomically save strict, finite JSON."""
    p = Path(p)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def render_one(task: str, checkpoint: str, video: str, tag: str, horizontal: bool) -> None:
    """Render frozen weights and verify their first episode against the saved holdout."""
    import gymnasium
    import numpy as np

    sys.path.insert(0, str(ROOT / "usage_assets"))
    import fast_gymnasium_20261015 as fast
    import replay_fulltrain_20261015 as replay

    original = replay.trainer.make_env
    original_fused = fast.make_fused_env
    recorders = []

    class Recorder(gymnasium.vector.VectorWrapper):
        def __init__(self, env: gymnasium.vector.VectorEnv):
            super().__init__(env)
            self.actual = []
            self.reference = []
            self.errors = []
            self.initial = {}

        def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
            obs, info = self.env.reset(seed=seed, options=options)
            self.initial = {k: obs[k][0].cpu().numpy().tolist() for k in ("pos", "vel", "quat")}
            if task == "reach_vel":
                self.initial["target_vel"] = (
                    (obs["vel"][0] + replay.trainer.task_error(obs, "DroneReachVel-v0")[0])
                    .cpu()
                    .numpy()
                    .tolist()
                )
            return obs, info

        def step(self, action: Tensor) -> tuple:
            outputs = self.env.step(action)
            obs = outputs[0]
            state = obs["vel" if task == "reach_vel" else "pos"][0].cpu().numpy().copy()
            error = (
                replay.trainer.task_error(
                    obs,
                    replay.trainer.ENVIRONMENT_IDS[
                        {"reach_pos": 0, "reach_vel": 1, "landing": 2, "figure8": 3}[task]
                    ],
                )[0]
                .cpu()
                .numpy()
                .copy()
            )
            self.actual.append(state)
            self.reference.append(state + error)
            self.errors.append(float(np.linalg.norm(error)))
            return outputs

    def make(config: Config, count: int) -> gymnasium.vector.VectorEnv:
        wrapped = Recorder(original(config, count))
        recorders.append(wrapped)
        return wrapped

    def make_fused(config: Config, count: int) -> gymnasium.vector.VectorEnv:
        wrapped = Recorder(original_fused(config, count))
        recorders.append(wrapped)
        return wrapped

    replay.trainer.make_env = make
    fast.make_fused_env = make_fused
    backend = "python" if tag.startswith("baseline_") else "fused"
    tf32 = not tag.startswith("baseline_")
    helper = ROOT / "usage_assets/replay_fulltrain_20261015.py"
    trainer = ROOT / "usage_assets/train_gymnasium.py"
    identity = {
        "checkpoint": str(Path(checkpoint).relative_to(ROOT)),
        "checkpoint_sha256": sha(checkpoint),
        "replay_helper_sha256": sha(helper),
        "trainer_source_sha256_at_render": sha(trainer),
        "env_backend": backend,
        "torch_cuda_matmul_allow_tf32": tf32,
        "adapter_source_sha256": sha(fast.__file__),
    }
    replay.render(
        Path(checkpoint), Path(video), 31001, horizontal, env_backend=backend, allow_tf32=tf32
    )
    rec = recorders[0]
    a = np.asarray(rec.actual)
    ref = np.asarray(rec.reference)
    assert np.isfinite(a).all() and np.isfinite(ref).all()
    source = OUT / (
        f"baseline_{task}_test31001.npz" if tag.startswith("baseline_") else f"{tag}_test31001.npz"
    )
    matching_horizontal = OUT / f"{tag}_horizontal_test31001.npz"
    exact_scope = not horizontal or matching_horizontal.exists()
    if horizontal and exact_scope:
        source = matching_horizontal
    comparison = {
        "source_npz": str(source.relative_to(ROOT)),
        "source_npz_sha256": sha(source),
        "initial": rec.initial,
        "traced_steps": len(a),
        "flight_simulation_duration_seconds": len(a) / 50,
        "max_observed_error": max(rec.errors),
        "final_observed_error": rec.errors[-1],
        "trajectory_sha256": hashlib.sha256(a.tobytes() + ref.tobytes()).hexdigest(),
    }
    with np.load(source, allow_pickle=False) as saved:
        np.testing.assert_allclose(
            np.asarray(rec.initial["pos"]), saved["initial_pos"][0], atol=1e-6, rtol=1e-6
        )
        comparison["same_initial_position_as_saved_holdout_world0"] = True
        if exact_scope:
            assert a.shape == saved["actual"].shape, (a.shape, saved["actual"].shape)
            np.testing.assert_allclose(a, saved["actual"], atol=2e-5, rtol=1e-5)
            np.testing.assert_allclose(ref, saved["reference"], atol=2e-5, rtol=1e-5)
            comparison["max_abs_actual_vs_npz"] = float(np.max(abs(a - saved["actual"])))
            comparison["max_abs_reference_vs_npz"] = float(np.max(abs(ref - saved["reference"])))
        else:
            comparison["scope"] = (
                "Horizontal-target visualization with vz=0; raw comparison file describes "
                "default 3D goals."
            )
        comparison["behavior_scope"] = (
            "horizontal_velocity_subset" if horizontal else "default_task"
        )
        comparison["matches_saved_behavior_scope"] = exact_scope
    write(
        "/tmp/gym_replay_" + task + ("_horizontal" if horizontal else "") + ".json",
        identity | comparison,
    )


def finish_video(task: str, video: Path, selected: dict, horizontal: bool) -> dict:
    """Decode every movie frame, extract the 4-second poster, and record provenance."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    stem = video.stem
    reader = imageio.get_reader(video)
    meta = reader.get_meta_data()
    frames = []
    minimum_std = 255.0
    low = 255
    high = 0
    first = None
    last = None
    for frame in reader:
        assert frame.shape == (480, 640, 3) and np.isfinite(frame).all()
        if first is None:
            first = frame.copy()
        last = frame.copy()
        frames.append(frame) if len(frames) <= 100 else None
        minimum_std = min(minimum_std, float(frame[82:430].std()))
        low = min(low, int(frame.min()))
        high = max(high, int(frame.max()))
    count = reader.count_frames()
    reader.close()
    fps = float(meta["fps"])
    index = min(100, count - 1)
    frame = frames[index] if index < len(frames) else last
    assert minimum_std > 1 and fps == 25 and count > 0
    teacher_model = "calibrated" if selected["tag"].endswith("_calibrated") else "legacy"
    method = "BC + PPO + imitation" if task in ("reach_pos", "landing") else "PPO"
    if teacher_model == "calibrated":
        method = "BC + PPO + calibrated teacher"
    unit = "m/s" if task == "reach_vel" else "m"
    scenario = {
        "reach_pos": "Independent random 3D position goals",
        "landing": "Official target: 0.1 m above ground (near-ground hover)",
        "figure8": "Official fixed 10 s figure-eight trajectory",
        "reach_vel": "Default 3D velocity targets; failures retained",
    }[task]
    if horizontal:
        scenario = "Horizontal-target subset (vz=0); separate from default 3D task"
    poster = Image.new("RGB", (640, 584), (19, 24, 35))
    poster.paste(Image.fromarray(frame), (0, 0))
    draw = ImageDraw.Draw(poster)
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    lines = [
        f"{task} | {method} | error: {unit}",
        "Validation selection | test 31001 / world 0 of 1024",
        scenario,
        f"Frozen weights | video t={index / fps:.2f}s | independent goal RNG",
    ]
    for i, line in enumerate(lines):
        draw.text(
            (12, 490 + 24 * i),
            line,
            font=font,
            fill="white" if i < 2 else (255, 180, 100) if horizontal else (125, 215, 255),
        )
    poster_path = video.with_name(stem + "_poster.png")
    poster.save(poster_path)
    trace = json.loads(
        Path(
            "/tmp/gym_replay_" + task + ("_horizontal" if horizontal else "") + ".json"
        ).read_text()
    )
    policy_frames = (trace["traced_steps"] + 1) // 2
    trace.update(
        policy_frame_count=policy_frames,
        failure_overlay_hold_seconds=max(0, count - policy_frames) / fps,
        poster_simulation_time_seconds=(min(index, policy_frames - 1) * 2 + 1) / 50,
    )
    return trace | {
        "status": "complete",
        "goal_rng_protocol": "independent_goal_rng_v1",
        "selected_tag": selected["tag"],
        "selected_validation_score": selected["validation_score"],
        "video": str(video.relative_to(ROOT)),
        "video_sha256": sha(video),
        "seed": 31001,
        "world_index": 0,
        "simulated_worlds": 1024,
        "scenario": scenario,
        "horizontal_velocity_subset": horizontal,
        "method": method,
        "teacher_model": teacher_model,
        "error_unit": unit,
        "fps": fps,
        "readback_frames": count,
        "actual_duration_seconds": count / fps,
        "metadata_duration_seconds": float(meta["duration"]),
        "resolution": [640, 480],
        "codec": meta["codec"],
        "all_frames_finite": True,
        "pixel_min": low,
        "pixel_max": high,
        "minimum_scene_pixel_std": minimum_std,
        "poster": str(poster_path.relative_to(ROOT)),
        "poster_sha256": sha(poster_path),
        "requested_poster_video_time_seconds": 4.0,
        "poster_video_time_seconds": index / fps,
        "poster_note": "Episode ended before requested 4 s; last actual video frame retained."
        if index < 100
        else None,
    }


def gallery(manifest: dict) -> None:
    """Assemble explicitly labelled panels from the four canonical frozen demonstrations."""
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1280, 1232), (19, 24, 35))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 21)
    draw.text(
        (18, 14),
        "Crazyflow frozen policies | validation choice | fixed test 31001 / world 0",
        font=font,
        fill="white",
    )
    draw.text(
        (18, 40),
        "ReachPos: independent 3D goals | Velocity panel: horizontal subset (vz=0)",
        font=ImageFont.truetype("DejaVuSans.ttf", 17),
        fill=(125, 215, 255),
    )
    for i, key in enumerate(("reach_pos", "landing", "figure8", "reach_vel_horizontal")):
        poster = ROOT / manifest["videos"][key]["poster"]
        canvas.paste(Image.open(poster).convert("RGB"), ((i % 2) * 640, 64 + (i // 2) * 584))
    path = OUT / "gallery.png"
    canvas.save(path)
    manifest["gallery"] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "resolution": [1280, 1232],
        "panels": ["reach_pos", "landing", "figure8", "reach_vel_horizontal"],
    }


def watch(once: bool = False) -> None:
    """Refresh validation-selected replays as full runs complete, including recipe comparisons."""
    manifest = {
        "schema": 3,
        "status": "partial",
        "selection_rule": (
            "Checkpoint selection solely by validation; fixed test seed 31001 / world 0 of 1024. "
            "No selection by held-out outcomes or appearance."
        ),
        "videos": {},
    }
    previous = OUT / "replay_manifest.json"
    if previous.exists():
        saved = json.loads(previous.read_text())
        if saved.get("schema", 0) >= 2 and all(
            v.get("goal_rng_protocol") == "independent_goal_rng_v1"
            for v in saved.get("videos", {}).values()
        ):
            manifest["videos"] = saved["videos"]
            for v in manifest["videos"].values():
                frames = (v["traced_steps"] + 1) // 2
                v.update(
                    flight_simulation_duration_seconds=v["traced_steps"] / 50,
                    policy_frame_count=frames,
                    failure_overlay_hold_seconds=max(0, v["readback_frames"] - frames) / v["fps"],
                    poster_simulation_time_seconds=(
                        min(round(v["poster_video_time_seconds"] * v["fps"]), frames - 1) * 2 + 1
                    )
                    / 50,
                )
    last_progress = None
    while True:
        if not OUT.exists():
            time.sleep(15)
            continue
        files = {p.stem for p in OUT.glob("*_s*_g0.99_200000000.json")}
        ready = [
            task
            for task in ("reach_pos", "reach_vel", "landing", "figure8")
            if all(f"{task}_s{seed}_g0.99_200000000" in files for seed in (7, 19, 43))
        ]
        if len(files) != last_progress:
            print(
                "New protocol completed main-run JSON:",
                len(files),
                "ready tasks:",
                ready,
                flush=True,
            )
            last_progress = len(files)
        if ready:
            subprocess.run(
                [sys.executable, str(ROOT / "usage_assets/summarize_fulltrain_20261015.py")],
                cwd=ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            summary = json.loads((OUT / "summary.json").read_text())
            for task in ready:
                selected = summary["tasks"][task]["recommended_demonstration"]
                if not selected or selected["status"] != "complete_study_validation_choice":
                    continue
                checkpoint = ROOT / selected["checkpoint"]
                for horizontal in (False, True) if task == "reach_vel" else (False,):
                    key = task + ("_horizontal" if horizontal else "")
                    existing = manifest["videos"].get(key, {})
                    source = OUT / (
                        f"baseline_{task}_test31001.npz"
                        if selected["tag"].startswith("baseline_")
                        else f"{selected['tag']}_test31001.npz"
                    )
                    if (
                        horizontal
                        and (OUT / f"{selected['tag']}_horizontal_test31001.npz").exists()
                    ):
                        source = OUT / f"{selected['tag']}_horizontal_test31001.npz"
                    source_matches = (
                        existing.get("source_npz") == str(source.relative_to(ROOT))
                        and source.exists()
                        and existing.get("source_npz_sha256") == sha(source)
                    )
                    if (
                        source_matches
                        and existing.get("checkpoint_sha256") == selected["checkpoint_sha256"]
                        and existing.get("replay_helper_sha256")
                        == sha(ROOT / "usage_assets/replay_fulltrain_20261015.py")
                    ):
                        continue
                    video = OUT / ("best_" + key + ".mp4")
                    print("Rendering frozen", key, selected["tag"], flush=True)
                    subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "render",
                            "--task",
                            task,
                            "--checkpoint",
                            str(checkpoint),
                            "--video",
                            str(video),
                            "--tag",
                            selected["tag"],
                        ]
                        + (["--horizontal"] if horizontal else []),
                        cwd=ROOT,
                        env=os.environ
                        | {
                            "MUJOCO_GL": "egl",
                            "PYOPENGL_PLATFORM": "egl",
                            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                            "MPLBACKEND": "Agg",
                        },
                        check=True,
                    )
                    manifest["videos"][key] = finish_video(task, video, selected, horizontal)
                    write(OUT / "replay_manifest.json", manifest)
                    print(
                        "Video verified",
                        key,
                        manifest["videos"][key]["readback_frames"],
                        "frames",
                        manifest["videos"][key]["actual_duration_seconds"],
                        "s",
                        flush=True,
                    )
            if (
                all(
                    key in manifest["videos"]
                    for key in (
                        "reach_pos",
                        "reach_vel",
                        "reach_vel_horizontal",
                        "landing",
                        "figure8",
                    )
                )
                and summary["status"] == "complete"
            ):
                gallery(manifest)
                manifest["status"] = "complete"
                manifest["summary_sha256"] = sha(OUT / "summary.json")
                manifest["producer_source_sha256"] = sha(__file__)
                write(OUT / "replay_manifest.json", manifest)
                print("All frozen replays and gallery complete", flush=True)
                return
        if once:
            write(OUT / "replay_manifest.json", manifest)
            return
        time.sleep(30)


def main() -> None:
    """Run the movie builder CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["watch", "render"])
    p.add_argument(
        "--once", action="store_true", help="Build currently available recommendations, then exit."
    )
    p.add_argument("--task", choices=["reach_pos", "reach_vel", "landing", "figure8"])
    p.add_argument("--checkpoint")
    p.add_argument("--video")
    p.add_argument("--tag")
    p.add_argument("--horizontal", action="store_true")
    args = p.parse_args()
    if args.mode == "watch":
        watch(args.once)
    else:
        if not all((args.task, args.checkpoint, args.video, args.tag)):
            p.error("render requires task, checkpoint, video and tag")
        render_one(args.task, args.checkpoint, args.video, args.tag, args.horizontal)


if __name__ == "__main__":
    main()
