"""Rebuild real-flight evidence from the two saved, attributed source videos.

This script performs frame extraction, crop, layout and video encoding only. It does not generate,
retouch or synthesize any flight imagery. All selected timestamps and source hashes are explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT_DEFAULT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
SOURCES = [
    {
        "file": "trajectory_tracking.mp4",
        "url": "https://www.youtube.com/watch?v=isHuQqZSapo",
        "title": "Crazyflow - Sim-to-Real",
        "publisher": "Learning Systems and Robotics Lab",
        "upload_date": "20260521",
        "sha256": "e51f5558811d47929e6be411664b8b882049179b19907e7d86076f3ac56d0c24",
    },
    {
        "file": "recovery_training.mp4",
        "url": "https://www.youtube.com/watch?v=684uZqaAmuU",
        "title": "Crazyflow - Throw",
        "publisher": "Learning Systems and Robotics Lab",
        "upload_date": "20260521",
        "sha256": "438550b965170b26173ad72110371fb1fd080e4fdd570bef80c75f2b4fe37a79",
    },
]


def sha256(path: Path) -> str:
    """Calculate a file hash without changing its contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(path: Path) -> dict[str, Any]:
    """Read a local source video's dimensions, frame rate and duration."""
    reader = imageio.get_reader(path)
    try:
        return reader.get_meta_data()
    finally:
        reader.close()


def make_sequence(
    source: str,
    times: tuple[float, float, float],
    crop: tuple[int, int, int, int],
    title: str,
    labels: tuple[str, str, str],
    output: str,
    font_path: Path,
    panel_height: int,
) -> None:
    """Extract three actual video frames and add captions outside their image areas."""
    reader = imageio.get_reader(HERE / source)
    fps = reader.get_meta_data()["fps"]
    top, bottom, width = 110, 125, 600
    canvas = Image.new("RGB", (1800, top + panel_height + bottom), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 16), title, fill="#102943", font=ImageFont.truetype(str(font_path), 32))
    draw.text(
        (35, 60),
        "原作者公开实飞视频帧 · 学习系统与机器人实验室（LSY）",
        fill="#3b4b5b",
        font=ImageFont.truetype(str(font_path), 24),
    )
    try:
        for index, (time_s, label) in enumerate(zip(times, labels, strict=True)):
            frame = reader.get_data(round(time_s * fps))
            image = (
                Image.fromarray(frame)
                .crop(crop)
                .resize((width, panel_height), Image.Resampling.LANCZOS)
            )
            canvas.paste(image, (index * width, top))
            draw.text(
                (index * width + 20, top + panel_height + 12),
                f"视频 {time_s:.1f} s｜{label}",
                font=ImageFont.truetype(str(font_path), 24),
                fill="#132f4a",
            )
    finally:
        reader.close()
    draw.text(
        (35, top + panel_height + 61),
        "说明：从原片定时截取；仅裁切与排版，保留真实机体及场地；非本项目实机试验。",
        font=ImageFont.truetype(str(font_path), 22),
        fill="#3b4b5b",
    )
    canvas.save(HERE / output)


def make_gif(
    source: str,
    start_s: int,
    end_s: int,
    output: str,
    crop: tuple[int, int, int, int] | None = None,
) -> None:
    """Sample video at nominal six fps and save actual frames as an animated GIF."""
    reader = imageio.get_reader(HERE / source)
    fps = reader.get_meta_data()["fps"]
    frames = []
    try:
        for index in range(start_s * 6, end_s * 6):
            image = Image.fromarray(reader.get_data(round(index / 6 * fps)))
            if crop:
                image = image.crop(crop).resize((640, 360), Image.Resampling.LANCZOS)
            else:
                image.thumbnail((800, 450), Image.Resampling.LANCZOS)
            frames.append(image)
    finally:
        reader.close()
    # GIF stores duration in 10-ms units; the nominal 167-ms interval is quantized by Pillow.
    frames[0].save(
        HERE / output, save_all=True, append_images=frames[1:], duration=167, loop=0, optimize=True
    )


def make_excerpt() -> None:
    """Encode the original recovery video's 21-to-29-second segment without audio."""
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-ss",
            "21",
            "-i",
            str(HERE / "recovery_training.mp4"),
            "-t",
            "8",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(HERE / "deployment_excerpt.mp4"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def update_manifest(font_path: Path) -> None:
    """Write source attribution, exact transformations and the regenerated file hashes."""
    videos = []
    for source in SOURCES:
        details = source.copy()
        details["duration_s"] = metadata(HERE / source["file"])["duration"]
        videos.append(details)
    derived = [
        {
            "file": "tracking_sequence.png",
            "source": "trajectory_tracking.mp4",
            "timestamps_s": [3, 9, 15],
            "processing": (
                "Crop bottom-left PPO quadrant [0,540,960,1080]; resize and add separately "
                "positioned captions. Pixel content unaltered."
            ),
        },
        {
            "file": "deployment_sequence.png",
            "source": "recovery_training.mp4",
            "timestamps_s": [21.8, 24, 27],
            "processing": (
                "Crop central physical camera area [600,0,1320,1080]; resize and add separately "
                "positioned captions. Pixel content unaltered."
            ),
        },
        {
            "file": "deployment_excerpt.mp4",
            "source": "recovery_training.mp4",
            "time_range_s": [21, 29],
            "processing": "Extract silent 8-second excerpt, full field of view retained.",
        },
        {
            "file": "tracking_flight.gif",
            "source": "trajectory_tracking.mp4",
            "time_range_s": [0, 18],
            "processing": (
                "Sample at nominal 6 fps; crop bottom-left PPO panel [0,540,960,1080], resize to "
                "640x360. Pixel content otherwise unaltered. GIF frame delays are quantized."
            ),
        },
        {
            "file": "deployment_flight.gif",
            "source": "recovery_training.mp4",
            "time_range_s": [21, 29],
            "processing": (
                "Sample at nominal 6 fps; resize to width800, full field of view retained. "
                "GIF frame delays are quantized."
            ),
        },
    ]
    for record in derived:
        path = HERE / record["file"]
        record["sha256"] = sha256(path)
        if path.suffix == ".gif":
            with Image.open(path) as image:
                duration_ms = 0
                for index in range(image.n_frames):
                    image.seek(index)
                    duration_ms += image.info.get("duration", 0)
                record["gif_frames"] = image.n_frames
                record["gif_playback_duration_s"] = duration_ms / 1000
    manifest = {
        "material_type": (
            "Third-party publicly released real-flight evidence, not a local hardware experiment"
        ),
        "retrieved_on": "2026-10-28",
        "generator": "build_frames.py",
        "font": {"name": font_path.name, "sha256": sha256(font_path)},
        "videos": videos,
        "derived": derived,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    """Verify saved source videos and regenerate all derived flight evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, default=FONT_DEFAULT)
    args = parser.parse_args()
    if not args.font.is_file():
        raise FileNotFoundError("Install Noto CJK fonts or pass --font with a CJK-capable font")
    for source in SOURCES:
        if sha256(HERE / source["file"]) != source["sha256"]:
            raise ValueError(
                f"Saved source hash does not match the attributed original: {source['file']}"
            )
    make_sequence(
        "trajectory_tracking.mp4",
        (3.0, 9.0, 15.0),
        (0, 540, 960, 1080),
        "实物八字轨迹跟踪：PPO 策略的连续时刻画面",
        ("轨迹右上方", "轨迹左上方", "轨迹右下方"),
        "tracking_sequence.png",
        args.font,
        338,
    )
    make_sequence(
        "recovery_training.mp4",
        (21.8, 24.0, 27.0),
        (600, 0, 1320, 1080),
        "实物抛掷后恢复与空中稳定：连续时刻画面",
        ("离手飞行", "姿态恢复", "空中稳定"),
        "deployment_sequence.png",
        args.font,
        900,
    )
    make_excerpt()
    make_gif("trajectory_tracking.mp4", 0, 18, "tracking_flight.gif", (0, 540, 960, 1080))
    make_gif("recovery_training.mp4", 21, 29, "deployment_flight.gif")
    update_manifest(args.font)
    print("Rebuilt two frame sequences, two GIFs, an 8-second excerpt and the evidence manifest")


if __name__ == "__main__":
    main()
