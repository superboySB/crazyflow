"""Build local visual evidence from recorded simulations and a frozen policy replay.

Only adds labels and layout to real recorded frames. The reinforcement-learning replay advances
Crazyflow physics using the saved policy; it is not a plot animation or synthetic illustration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / "peibiao_assets"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
SWARM_COLORS = ("#e63326", "#198cf2", "#19bf59", "#bf4cd9")


def font(size: int) -> ImageFont.FreeTypeFont:
    """Load the CJK font used in evidence annotations."""
    return ImageFont.truetype(FONT, size)


def sha256(path: Path) -> str:
    """Hash a source or output for traceability."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(image: Image.Image, title: str, subtitle: str, width: int = 720) -> Image.Image:
    """Add a clearly separated label to an unchanged-aspect-ratio frame."""
    image = image.convert("RGB")
    image = image.resize((width, round(image.height * width / image.width)))
    out = Image.new("RGB", (width, image.height + 76), "#122033")
    out.paste(image, (0, 76))
    draw = ImageDraw.Draw(out)
    draw.text((16, 8), title, font=font(22), fill="white")
    draw.text((16, 42), subtitle, font=font(16), fill="#bfd1e5")
    return out


def project_swarm_positions() -> np.ndarray:
    """Project recorded drone centres through the exact free-camera configuration."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="10 10 .1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    scene = mujoco.MjvScene(model, maxgeom=100)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 8.0, 90.0, -65.0
    camera.lookat[:] = [0.0, 0.0, 0.5]
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    gl_camera = scene.camera[0]
    eye = np.mean([value.pos for value in scene.camera], axis=0)
    forward, up = gl_camera.forward, gl_camera.up
    right = np.cross(forward, up)
    scale = 360 / (gl_camera.frustum_top / gl_camera.frustum_near)
    # The source MP4 is sampled every ten 100 Hz state samples, starting at time zero.
    positions = np.load(ASSETS / "swarm_data.npz")["actual"][::10]
    relative = positions - eye
    depth = relative @ forward
    return np.stack(
        (480 + relative @ right / depth * scale, 360 - relative @ up / depth * scale), axis=-1
    )


def annotate_swarm(frame: Image.Image, pixels: np.ndarray) -> Image.Image:
    """Label each physical drone at its measured and camera-projected centre."""
    frame = frame.copy()
    draw = ImageDraw.Draw(frame)
    for index, ((x, y), color) in enumerate(zip(pixels, SWARM_COLORS, strict=True)):
        offset = -76 if index < 2 else 24
        anchor = (x + offset, y - 48)
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), outline=color, width=3)
        draw.line((x, y - 11, anchor[0] + 24, anchor[1] + 30), fill=color, width=2)
        draw.rounded_rectangle(
            (anchor[0] - 5, anchor[1], anchor[0] + 53, anchor[1] + 33),
            radius=5,
            fill="#122033",
            outline=color,
            width=2,
        )
        draw.text(anchor, f"{index + 1}号", font=font(21), fill=color)
    return frame


def label_goal(frame: Image.Image, goal: np.ndarray, scene: object) -> Image.Image:
    """Place a small off-body goal label using the renderer's actual GL camera."""
    camera = scene.camera[0]
    eye = np.mean([value.pos for value in scene.camera], axis=0)
    relative = goal - eye
    depth = relative @ camera.forward
    right = np.cross(camera.forward, camera.up)
    scale = frame.height / 2 / (camera.frustum_top / camera.frustum_near)
    x = frame.width / 2 + relative @ right / depth * scale
    y = frame.height / 2 - relative @ camera.up / depth * scale
    draw = ImageDraw.Draw(frame)
    anchor = (x + 60, y - 65)
    draw.line((x + 6, y - 6, anchor[0], anchor[1] + 29), fill="#ffb26f", width=2)
    draw.rounded_rectangle(
        (anchor[0] - 5, anchor[1] - 2, anchor[0] + 77, anchor[1] + 32),
        radius=5,
        fill="#122033",
        outline="#ffb26f",
        width=2,
    )
    draw.text(anchor, "目标点", font=font(22), fill="#ffb26f")
    return frame


def video_evidence(name: str, source: Path, seconds: list[float], title: str) -> dict:
    """Extract measured-time frames, a contact sheet, and a full-duration GIF."""
    reader = imageio.get_reader(source)
    meta = reader.get_meta_data()
    fps = float(meta["fps"])
    count = reader.count_frames()
    chosen = [min(count - 1, int(round(t * fps))) for t in seconds]
    pixels = project_swarm_positions() if name == "swarm" else None
    crop = {"neural_planner": (0, 70, 800, 528), "rl_policy": (0, 76, 960, 716)}.get(name)

    def prepare(index: int, frame: np.ndarray, width: int) -> Image.Image:
        image = Image.fromarray(frame)
        if pixels is not None:
            image = annotate_swarm(image, pixels[index])
        if crop:
            image = image.crop(crop)
        key = {
            "swarm": "1—4号为实际机体位置",
            "neural_planner": "橙色：规划路径 · 青色：实际飞行轨迹",
            "rl_policy": "橙色：目标点 · 青色：实际飞行轨迹",
        }[name]
        return panel(image, title, f"视频时间 {index / fps:.2f} 秒 · {key}", width=width)

    thumbnails = []
    for index in chosen:
        thumbnails.append(prepare(index, reader.get_data(index), width=720))
    reader.close()
    width, height = thumbnails[0].size
    rows = (len(thumbnails) + 1) // 2
    sheet = Image.new("RGB", (width * 2, height * rows), "white")
    for number, thumbnail in enumerate(thumbnails):
        sheet.paste(thumbnail, ((number % 2) * width, (number // 2) * height))
    sheet.save(HERE / f"{name}_sequence.png")

    full_video = None
    if name in ("swarm", "neural_planner"):
        video_path = HERE / f"{name}_zh.mp4"
        poster_path = HERE / (
            "swarm_replay.png" if name == "swarm" else "neural_planner_replay.png"
        )
        poster_seconds = 15.5 if name == "swarm" else 7.0
        poster_index = min(count - 1, round(poster_seconds * fps))
        reader = imageio.get_reader(source)
        writer = imageio.get_writer(
            video_path, fps=fps, codec="libx264", quality=8, macro_block_size=1
        )
        try:
            for index, frame in enumerate(reader):
                labelled = prepare(index, frame, width=frame.shape[1])
                writer.append_data(np.asarray(labelled))
                if index == poster_index:
                    labelled.save(poster_path)
        finally:
            reader.close()
            writer.close()
        full_video = {
            "path": str(video_path.relative_to(ROOT)),
            "sha256": sha256(video_path),
            "fps": fps,
            "frames": count,
            "duration_seconds": count / fps,
            "frame_rule": "every source frame retained in its original order and original fps",
            "poster": str(poster_path.relative_to(ROOT)),
            "poster_frame": poster_index,
            "poster_video_seconds": poster_index / fps,
        }

    stride = max(1, math.ceil(fps / 4))
    gif_frames = []
    reader = imageio.get_reader(source)
    for index, frame in enumerate(reader):
        if index % stride == 0:
            labelled = prepare(index, frame, width=640)
            gif_frames.append(labelled.quantize(colors=96))
    reader.close()
    gif_frames[0].save(
        HERE / f"{name}.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=[
            round(min(stride, count - index) / fps * 1000) for index in range(0, count, stride)
        ],
        loop=0,
        optimize=False,
    )
    return {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "fps": fps,
        "source_frames": count,
        "source_duration_seconds": count / fps,
        "full_chinese_video": full_video,
        "contact_sheet_frames": chosen,
        "contact_sheet_seconds": [index / fps for index in chosen],
        "gif_stride_frames": stride,
        "gif_speed": "source duration preserved; fixed-rate subsampling",
        "crop_rectangle_xyxy": crop,
        "transform": (
            "CJK label bar and aspect-preserving resize. Neural video: remove only the existing "
            "70-pixel English header. RL: remove existing 76-pixel label header "
            "before adding one CJK header. Swarm: measured centre labels only."
        ),
        "swarm_label_projection": (
            {
                "source_data": "peibiao_assets/swarm_data.npz",
                "source_data_sha256": sha256(ASSETS / "swarm_data.npz"),
                "rule": "actual[::10], 100 Hz states to 10 fps source video",
                "source_frame_size": [960, 720],
                "camera": {
                    "distance": 8.0,
                    "azimuth": 90.0,
                    "elevation": -65.0,
                    "lookat": [0.0, 0.0, 0.5],
                    "vertical_fov_degrees": 45.0,
                },
                "projection": "MuJoCo mjv_updateScene GL camera, mono eye averaged from stereo",
                "labels": "1..4 map to actual drone indices 0..3",
                "check_frames": {str(index): pixels[index].tolist() for index in chosen},
            }
            if pixels is not None
            else None
        ),
    }


def rl_replay() -> dict:
    """Run and record the fixed BC+PPO policy on the independent evaluation seed."""
    import torch

    sys.path.insert(0, str(ROOT / "usage_assets"))
    import train_gymnasium as trainer

    from crazyflow.sim.visualize import draw_line, draw_points

    torch.set_num_threads(1)
    checkpoint = ASSETS / "rl_cpu_medium.pt"
    agent, config, obs_keys, _ = trainer.load_checkpoint(checkpoint, device="cpu")
    config.eval_envs = 32
    env = trainer.make_env(config, config.eval_envs)
    sim = env.unwrapped.sim
    step = env.step
    recorded = []
    frames = []
    n_step = 0
    camera = None
    try:

        def record_step(action: torch.Tensor) -> tuple:
            nonlocal n_step, camera
            result = step(action)
            n_step += 1
            observation = result[0]
            position = observation["pos"][0].cpu().numpy().copy()
            goal = (
                (observation["pos"][0] + trainer.task_error(observation, config.env_id)[0])
                .cpu()
                .numpy()
                .copy()
            )
            recorded.append(position)
            if n_step % 5 == 1:
                if camera is None:
                    look = (position + goal) / 2
                    camera = {"distance": 1.8, "azimuth": 125.0, "elevation": -28.0, "lookat": look}
                    sim.render(mode="rgb_array", world=0, width=960, height=640, cam_config=camera)
                sim.viewer.viewer._markers.clear()
                draw_points(sim, goal[None], np.array([1, 0.4, 0.1, 0.5]), size=0.02)
                if len(recorded) > 1:
                    draw_line(sim, np.asarray(recorded)[::2], np.array([0.05, 0.85, 1, 1]), 2, 2)
                frame = Image.fromarray(
                    sim.render(mode="rgb_array", world=0, width=960, height=640)
                )
                frame = label_goal(frame, goal, sim.viewer.viewer.scn)
                frame = panel(
                    frame,
                    "冻结强化学习策略：随机初态到目标点闭环飞行",
                    (
                        f"t = {n_step / config.freq:.2f} s · 橙色：目标点 · 青色：实测轨迹"
                        f" · 误差 {np.linalg.norm(goal - position):.3f} m"
                    ),
                    width=960,
                )
                frames.append(np.asarray(frame))
            return result

        env.step = record_step
        metrics, behavior = trainer.evaluate(agent, config, obs_keys, eval_seed=10007, env=env)
    finally:
        env.close()
    output = HERE / "rl_policy_replay.mp4"
    imageio.mimsave(output, frames, fps=10, codec="libx264", quality=8, macro_block_size=1)
    Image.fromarray(frames[-1]).save(HERE / "rl_policy_replay.png")
    np.savez_compressed(HERE / "rl_replay_trace.npz", **behavior)
    (HERE / "rl_replay_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "seed": 10007,
        "worlds": 32,
        "rendered_world": 0,
        "frames": len(frames),
        "fps": 10,
        "first_video_sample_seconds": 0.02,
        "main_still_frame": len(frames) - 1,
        "main_still_simulation_seconds": 0.02 + (len(frames) - 1) / 10,
        "goal_marker": {
            "radius_m": 0.02,
            "alpha": 0.5,
            "label": "目标点",
            "label_position": "off-body leader line from renderer GL camera projection",
        },
        "metrics": metrics,
        "source": "fresh physics rollout of the saved frozen policy, own observed state feedback",
    }


def main() -> None:
    """Create the complete set of local dynamic visual evidence."""
    manifest = {
        "swarm": video_evidence(
            "swarm",
            ASSETS / "swarm_simulation.mp4",
            [0, 10, 20, 31],
            "四机自主规划：障碍环境内交叉飞行",
        ),
        "neural_planner": video_evidence(
            "neural_planner",
            ASSETS / "neural_planner_flight.mp4",
            [0, 2, 6, 9.95],
            "神经轨迹规划：预测路径及实际闭环飞行",
        ),
        "rl_replay": rl_replay(),
    }
    manifest["rl_video"] = video_evidence(
        "rl_policy",
        HERE / "rl_policy_replay.mp4",
        [0, 0.5, 2, 9.9],
        "强化学习策略：独立测试场景飞行过程",
    )
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
