"""Run paired near-ground and two-drone downwash simulations with Chinese evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

import crazyflow  # noqa: F401, I001  # Enable SciPy array API before scientific imports.
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.sim import Sim
from crazyflow.sim.pipeline import insert_fn_before
from examples.plugins.downwash import downwash_fn
from examples.plugins.ground_effect import ground_effect_fn

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREQ = 500
CONTROL_FREQ = 100
DT = 1 / CONTROL_FREQ
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False})


def initialize(sim: Sim, positions: np.ndarray) -> float:
    """Initialize matched level airborne conditions and hovering motor speed."""
    weight = float(sim.data.params.mass[0] * -sim.data.params.gravity_vec[2])
    rpm = motor_force2rotor_vel(jnp.asarray(weight / 4), sim.data.params.rpm2thrust)
    states = sim.data.states.replace(
        pos=jnp.asarray(positions), rotor_vel=jnp.broadcast_to(rpm, sim.data.states.rotor_vel.shape)
    )
    sim.data = sim.data.replace(states=states)
    sim.build_default_data()
    return float(rpm[0]) if rpm.ndim else float(rpm)


def masked_downwash(data: SimData) -> SimData:
    """Apply the upstream downwash plugin only in world 1; world 0 is the baseline."""
    evaluated = downwash_fn(data)
    mask = jnp.array([False, True])[:, None, None]
    return evaluated.replace(
        states=evaluated.states.replace(
            force=jnp.where(mask, evaluated.states.force, 0),
            torque=jnp.where(mask, evaluated.states.torque, 0),
        )
    )


def masked_ground_effect(data: SimData) -> SimData:
    """Apply the upstream near-ground plugin only in world 1."""
    evaluated = ground_effect_fn(data)
    mask = jnp.array([False, True])[:, None, None]
    return evaluated.replace(
        states=evaluated.states.replace(force=jnp.where(mask, evaluated.states.force, 0))
    )


def composite(images: list[np.ndarray], time_s: float, label: str) -> np.ndarray:
    """Compose unaltered simulator frames with Chinese labels outside the image regions."""
    width, height = images[0].shape[1], images[0].shape[0]
    canvas = Image.new("RGB", (2 * width, height + 66), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 24)
    for i, img in enumerate(images):
        canvas.paste(Image.fromarray(img), (i * width, 66))
    draw.text((14, 6), f"基线：未启用{label}插件", fill=(25, 35, 45), font=font)
    draw.text((width + 14, 6), f"对照：已启用{label}插件", fill=(25, 35, 45), font=font)
    draw.text(
        (14, 34),
        f"实际仿真时刻 t = {time_s:.2f} s；左右相同初态、"
        + ("相同位置指令" if label == "下洗" else "相同固定转速"),
        fill=(25, 35, 45),
        font=font,
    )
    return np.asarray(canvas)


def frames(sim: Sim, camera: dict) -> list[np.ndarray]:
    """Render each paired world independently from its actual state."""
    return [
        sim.render(mode="rgb_array", world=w, width=640, height=480, cam_config=camera)
        for w in range(2)
    ]


def ground_experiment() -> dict:
    """Compare fixed-RPM near-ground lift with matched initial states."""
    sim = Sim(n_worlds=2, drone="cf21B_500", control="rotor_vel", device="cpu", freq=FREQ)
    pos = np.zeros((2, 1, 3), dtype=np.float32)
    pos[..., 2] = 0.05
    rpm = initialize(sim, pos)
    insert_fn_before(
        sim.step_pipeline, "integration", masked_ground_effect, name="paired_ground_effect"
    )
    sim.build_step_fn()
    sim.rotor_vel_control(np.full((2, 1, 4), rpm))
    records = []
    scene = None
    camera = {"lookat": [0, 0, 0.13], "distance": 0.6, "elevation": -5, "azimuth": 90}
    writer = imageio.get_writer(HERE / "ground_effect.mp4", fps=10, macro_block_size=2)
    try:
        for i in range(201):
            t = i * DT
            z = np.asarray(sim.data.states.pos)[:, 0, 2]
            vz = np.asarray(sim.data.states.vel)[:, 0, 2]
            force = np.asarray(sim.data.states.force)[:, 0, 2]
            records.append([t, z[0], z[1], vz[0], vz[1], force[0], force[1]])
            if i % 10 == 0:
                img = composite(frames(sim, camera), t, "地效")
                writer.append_data(img)
                if i == 200:
                    scene = img
            if i < 200:
                sim.step(FREQ // CONTROL_FREQ)
    finally:
        writer.close()
        sim.close()
    data = np.asarray(records)
    np.savetxt(
        HERE / "ground_effect_trace.csv",
        data,
        delimiter=",",
        header="t_s,baseline_z_m,ground_z_m,baseline_vz_m_s,ground_vz_m_s,baseline_external_fz_N,ground_external_fz_N",
        comments="",
    )
    assert np.isfinite(data).all()
    assert abs(data[-1, 1] - 0.05) < 1e-4 and data[-1, 2] > data[-1, 1] + 0.02
    imageio.imwrite(HERE / "ground_effect_scene.png", scene)
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    ax = fig.add_subplot(grid[0, :])
    ax.imshow(scene)
    ax.axis("off")
    ax = fig.add_subplot(grid[1, 0])
    ax.plot(data[:, 0], data[:, 1], label="未启用地效")
    ax.plot(data[:, 0], data[:, 2], label="启用地效")
    ax.set(xlabel="时间（s）", ylabel="实际高度（m）", title="相同悬停转速输入下的高度响应")
    ax.legend()
    ax.grid(alpha=0.25)
    ax = fig.add_subplot(grid[1, 1])
    ax.plot(data[:, 0], data[:, 6], color="#d55e00")
    ax.set(xlabel="时间（s）", ylabel="附加升力（N）", title="地效插件实际输出的附加升力")
    ax.grid(alpha=0.25)
    fig.savefig(HERE / "ground_effect_evidence.png", dpi=160)
    plt.close(fig)
    return {
        "duration_s": 2,
        "initial_height_m": 0.05,
        "constant_rotor_command_RPM": rpm,
        "baseline_final_height_m": float(data[-1, 1]),
        "ground_effect_final_height_m": float(data[-1, 2]),
        "final_height_difference_m": float(data[-1, 2] - data[-1, 1]),
        "max_added_lift_N": float(data[:, 6].max()),
        "samples": len(data),
        "plugin": "examples.plugins.ground_effect.ground_effect_fn; world1 only",
        "parameters": {
            "propeller_diameter_m": 0.055,
            "mu": 2,
            "minimum_height_m": 0.02,
            "max_gain": 2,
        },
        "scope": (
            "Open-loop fixed RPM, no real-flight validation; "
            "near-ground formula only within its stated range"
        ),
    }


def downwash_experiment() -> dict:
    """Run matched two-drone closed loops, comparing no downwash to the enabled plugin."""
    sim = Sim(n_worlds=2, n_drones=2, drone="cf21B_500", control="state", device="cpu", freq=FREQ)
    positions = np.array([[[0, 0, 1.2], [-0.5, 0, 0.5]]] * 2, dtype=np.float32)
    initialize(sim, positions)
    insert_fn_before(sim.step_pipeline, "integration", masked_downwash, name="paired_downwash")
    sim.build_step_fn()
    cmd = np.zeros((2, 2, 16), dtype=np.float32)
    cmd[..., 12] = 1
    cmd[:, 0, :3] = [0, 0, 1.2]
    cmd[:, 1, 2] = 0.5
    records = []
    poses, forces, torques = [], [], []
    camera = {"lookat": [0, 0, 0.84], "distance": 1.5, "elevation": -5, "azimuth": 90}
    selected = None
    selected_time = 0.0
    best_difference = -1.0
    writer = imageio.get_writer(HERE / "downwash_comparison.mp4", fps=10, macro_block_size=2)
    try:
        for i in range(1001):
            t = i * DT
            ref_x = -0.5 if t <= 4 else -0.5 + 0.25 * min(t - 4, 4)
            pos = np.asarray(sim.data.states.pos)
            force = np.asarray(sim.data.states.force)
            torque = np.asarray(sim.data.states.torque)
            poses.append(pos.copy())
            forces.append(force.copy())
            torques.append(torque.copy())
            records.append(
                [
                    t,
                    ref_x,
                    pos[0, 0, 2],
                    pos[0, 1, 0],
                    pos[0, 1, 2],
                    pos[1, 0, 2],
                    pos[1, 1, 0],
                    pos[1, 1, 2],
                    force[0, 1, 2],
                    force[1, 1, 2],
                    torque[1, 1, 1],
                ]
            )
            if i % 10 == 0:
                img = composite(frames(sim, camera), t, "下洗")
                writer.append_data(img)
                difference = abs(pos[1, 1, 2] - pos[0, 1, 2])
                if difference > best_difference:
                    selected = img
                    selected_time = t
                    best_difference = difference
            if i < 1000:
                cmd[:, 1, 0] = ref_x
                sim.state_control(cmd)
                sim.step(FREQ // CONTROL_FREQ)
    finally:
        writer.close()
        sim.close()
    data = np.asarray(records)
    assert np.isfinite(data).all() and data[:, 9].min() < -0.02
    np.savetxt(
        HERE / "downwash_trace.csv",
        data,
        delimiter=",",
        header="t_s,lower_x_reference_m,baseline_upper_z_m,baseline_lower_x_m,baseline_lower_z_m,downwash_upper_z_m,downwash_lower_x_m,downwash_lower_z_m,baseline_lower_fz_N,downwash_lower_fz_N,downwash_lower_ty_Nm",
        comments="",
    )
    np.savez_compressed(
        HERE / "downwash_states.npz",
        time_s=data[:, 0],
        pos=np.asarray(poses),
        force=np.asarray(forces),
        torque=np.asarray(torques),
    )
    imageio.imwrite(HERE / "downwash_scene.png", selected)
    fig = plt.figure(figsize=(13, 9), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1])
    ax = fig.add_subplot(grid[0, :])
    ax.imshow(selected)
    ax.axis("off")
    ax = fig.add_subplot(grid[1, 0])
    ax.plot(data[:, 0], data[:, 3], label="下机：未启用下洗")
    ax.plot(data[:, 0], data[:, 6], label="下机：启用下洗")
    ax.plot(data[:, 0], data[:, 1], "--", label="下机位置指令")
    ax.set(xlabel="时间（s）", ylabel="水平位置 x（m）", title="两机上下交叉时的水平运动")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax = fig.add_subplot(grid[1, 1])
    ax.plot(data[:, 0], data[:, 4], label="下机：未启用下洗")
    ax.plot(data[:, 0], data[:, 7], label="下机：启用下洗")
    ax.axhline(0.5, ls="--", color="gray", label="下机高度指令")
    ax.set(xlabel="时间（s）", ylabel="实际高度 z（m）", title="独立状态反馈下的下机高度响应")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax = fig.add_subplot(grid[2, 0])
    ax.plot(data[:, 0], data[:, 9], color="#d55e00")
    ax.set(xlabel="时间（s）", ylabel="下机外力 z（N）", title="下洗附加外力：负值沿世界坐标系向下")
    ax.grid(alpha=0.25)
    ax = fig.add_subplot(grid[2, 1])
    ax.plot(data[:, 0], data[:, 10], color="#0072b2")
    ax.set(xlabel="时间（s）", ylabel="下机外力矩 y（N·m）", title="转子采样产生的附加外力矩")
    ax.grid(alpha=0.25)
    fig.savefig(HERE / "downwash_evidence.png", dpi=160)
    plt.close(fig)
    return {
        "duration_s": 10,
        "upper_height_command_m": 1.2,
        "lower_height_command_m": 0.5,
        "lower_pass": "hold x=-0.5 m for 4 s; pass to x=0.5 m at 0.25 m/s over 4 s; hold 2 s",
        "scene_time_s": selected_time,
        "minimum_lower_external_fz_N": float(data[:, 9].min()),
        "maximum_abs_lower_external_ty_Nm": float(np.max(np.abs(data[:, 10]))),
        "max_lower_height_difference_vs_baseline_m": float(np.max(np.abs(data[:, 7] - data[:, 4]))),
        "lower_height_range_baseline_m": [float(data[:, 4].min()), float(data[:, 4].max())],
        "lower_height_range_downwash_m": [float(data[:, 7].min()), float(data[:, 7].max())],
        "samples": len(data),
        "plugin": "examples.plugins.downwash.downwash_fn; world1 only",
        "parameters": {
            "air_density_kg_m3": 1.225,
            "propeller_radius_m": 0.0275,
            "motor_distance_m": 0.1,
            "thrust_decay_s_m": 0.07,
            "BD": 10.11,
            "S": 0.07668,
            "S0": -5.817,
        },
        "scope": (
            "Matched known path, two independent state-feedback simulations with common commands; "
            "far-field empirical plugin; not real-flight validation"
        ),
    }


def main() -> None:
    """Generate both paired actual simulations and their traceable evidence."""
    results = {
        "drone": "cf21B_500",
        "dynamics": "first_principles",
        "physics_hz": FREQ,
        "sampling_hz": CONTROL_FREQ,
        "backend": "cpu",
        "jax": jax.__version__,
        "model_parameters_sha256": hashlib.sha256(
            (ROOT / "crazyflow/dynamics/first_principles/params.toml").read_bytes()
        ).hexdigest(),
        "plugin_sha256": {
            name: hashlib.sha256((ROOT / f"examples/plugins/{name}.py").read_bytes()).hexdigest()
            for name in ("ground_effect", "downwash")
        },
        "ground_effect": ground_experiment(),
        "downwash": downwash_experiment(),
    }
    (HERE / "aerodynamics_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
