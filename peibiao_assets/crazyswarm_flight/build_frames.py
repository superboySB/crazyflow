"""Extract attributed, actual Crazyswarm flight frames without altering the scene."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_crazyswarm.mp4"
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
TIMES = (12.0, 15.5, 18.5, 20.0)
START, END = 12.0, 21.0
CROP = (250, 0, 1700, 1080)
SCENE_HEIGHT = 716
PANEL_HEIGHT = SCENE_HEIGHT + 66


def sha256(path: Path) -> str:
    """Return the SHA-256 of a local artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def title_frame(pixels: np.ndarray, time_s: float) -> Image.Image:
    """Resize an actual source frame and add a separately positioned Chinese bar."""
    frame = Image.fromarray(pixels).crop(CROP).resize((960, SCENE_HEIGHT), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (960, PANEL_HEIGHT), "#f3f6fa")
    result.paste(frame, (0, 66))
    draw = ImageDraw.Draw(result)
    draw.text(
        (16, 4), "实物机群升空与分层编队", font=ImageFont.truetype(str(FONT), 24), fill="#12304a"
    )
    draw.text(
        (16, 37),
        f"第三方实飞示例｜原片 {time_s:.2f} 秒",
        font=ImageFont.truetype(str(FONT), 17),
        fill="#43566c",
    )
    return result


def main() -> None:
    """Build a four-frame sheet, a local flight clip and an animation from the source."""
    reader = imageio.get_reader(SOURCE)
    meta = reader.get_meta_data()
    fps = float(meta["fps"])
    first, last = round(START * fps), round(END * fps)
    try:
        sheet = Image.new("RGB", (1920, PANEL_HEIGHT * 2), "#f3f6fa")
        for index, time_s in enumerate(TIMES):
            frame = title_frame(reader.get_data(round(time_s * fps)), time_s)
            sheet.paste(frame, ((index % 2) * 960, (index // 2) * PANEL_HEIGHT))
        sheet.save(HERE / "crazyswarm_sequence.png")
        gif_frames = []
        gif_indices = list(range(first, last, 8))
        writer = imageio.get_writer(
            HERE / "crazyswarm_flight.mp4", fps=fps, codec="libx264", quality=6, macro_block_size=2
        )
        try:
            for frame_index in range(first, last):
                frame = title_frame(reader.get_data(frame_index), frame_index / fps)
                writer.append_data(np.asarray(frame))
                if frame_index in gif_indices:
                    gif_frames.append(
                        frame.resize(
                            (640, round(PANEL_HEIGHT * 640 / 960)), Image.Resampling.LANCZOS
                        ).quantize(colors=128)
                    )
        finally:
            writer.close()
    finally:
        reader.close()
    # Match each selected frame's source interval, quantized to GIF's 10-ms time units.
    boundaries = [round((index - first) / fps * 100) * 10 for index in gif_indices]
    boundaries.append(round((last - first) / fps * 100) * 10)
    durations = [b - a for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)]
    gif_frames[0].save(
        HERE / "crazyswarm_flight.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    manifest = {
        "material_type": "third-party real-flight example; not this project's hardware experiment",
        "source": {
            "file": SOURCE.name,
            "url": "https://www.youtube.com/watch?v=D0CrjoYDt9w",
            "official_documentation": "https://crazyswarm.readthedocs.io/en/latest/",
            "official_repository_documentation": "https://raw.githubusercontent.com/IMRCLab/crazyswarm2/main/docs/index.rst",
            "title": "Crazyswarm: a large nano-quadcopter swarm (Sep. 2016)",
            "publisher": "ACT Lab at Brown",
            "source_duration_s": meta["duration"],
            "fps": fps,
            "size": meta["size"],
            "sha256": sha256(SOURCE),
        },
        "selected_timestamps_s": TIMES,
        "source_crop_xyxy": CROP,
        "no_people": "Entire selected source interval inspected; no people present, none erased",
        "excerpt_requested_time_s": [START, END],
        "excerpt_actual_frame_range": [first, last],
        "excerpt_duration_s": (last - first) / fps,
        "processing": (
            "Fixed source rectangle [250,0,1700,1080], resized with Chinese bar outside "
            "source frame. No scene objects synthesized, erased, moved or retouched. MP4 "
            "preserves source frame order and fps; GIF subsamples every eighth source frame "
            "with matched duration."
        ),
        "observable_content": (
            "Real indoor arena without people in the selected source interval; physical "
            "quadcopters change height and form layers. Footage alone does not validate "
            "any algorithm or hardware of the local project."
        ),
        "gif_frames": len(gif_frames),
        "gif_duration_s": sum(durations) / 1000,
        "outputs": {
            name: sha256(HERE / name)
            for name in [
                "crazyswarm_sequence.png",
                "crazyswarm_flight.mp4",
                "crazyswarm_flight.gif",
            ]
        },
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {key: manifest[key] for key in ["excerpt_duration_s", "gif_frames", "gif_duration_s"]}
        )
    )


if __name__ == "__main__":
    main()
