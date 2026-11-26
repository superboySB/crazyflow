"""Select graphics backends before importing MuJoCo in each container."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("runner", "device", "backend", "gpu"),
    (
        ("cf_container", "gpu", "egl", True),
        ("cf_container", "cpu", "osmesa", False),
        ("cf_container_cpu", "gpu", "osmesa", False),
        ("cf_container_gui", "gpu", "glfw", True),
        ("cf_container_gui", "cpu", "glfw", False),
    ),
)
def test_runtime_graphics_matches_container_device(
    runner: str, device: str, backend: str, gpu: bool
):
    """Keep CPU fallback independent of the GPU headless/desktop backends."""
    common = Path(__file__).resolve().parents[1] / "scripts/common.sh"
    script = 'source "$1"\nCF_DEVICE="$2"\ndocker() { printf "%s\\0" "$@"; }\n"$3" python -c pass\n'
    result = subprocess.run(
        ["bash", "-c", script, "render-test", str(common), device, runner],
        env={**os.environ, "DISPLAY": ":99", "CF_COMMON_LOADED": "0"},
        capture_output=True,
        text=True,
        check=True,
    )
    args = result.stdout.rstrip("\0").split("\0")
    assert f"JAX_COMPILATION_CACHE_DIR=/outputs/.cache/jax-{os.getuid()}" in args
    assert "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0" in args
    assert f"MUJOCO_GL={backend}" in args
    assert f"JAX_PLATFORMS={'cuda' if gpu else 'cpu'}" in args
    assert ("--gpus" in args) == gpu
    if runner == "cf_container_gui" and gpu:
        assert "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display" in args


@pytest.mark.render
def test_offscreen_video_records_the_active_opengl_renderer(tmp_path: Path):
    """Exercise real MuJoCo pixels and report the driver that produced them."""
    import imageio.v2 as imageio
    import numpy as np

    from crazyflow.balloon.config import Config
    from crazyflow.balloon.evaluate import render_replay

    cfg = Config(image_width=160, image_height=128)
    states = np.zeros((2, 12))
    states[:, 2] = 1.4
    path = tmp_path / "flight.mp4"
    report = render_replay(states, np.array([2.0, 0.0, 1.4]), 0.2, cfg, path)
    assert report["opengl"]["backend"] == os.environ.get("MUJOCO_GL", "glfw")
    assert all(report["opengl"][key] for key in ("vendor", "renderer", "version"))
    assert report["frames"] == 2 and path.stat().st_size > 0
    with imageio.get_reader(path) as reader:
        frame = reader.get_data(0)
    assert frame.shape == (128, 320, 3) and frame.std() > 10
