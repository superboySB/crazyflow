"""Closed-loop vertical tracking in Crazyflow and independent MuJoCo dynamics.

Both simulators run the same fixed state-feedback controller, rotor command limits,
and motor/thrust coefficients. The MuJoCo plant advances a free rigid body using
mj_step; a separate scalar motor model supplies force. This is a restricted 1-D
scenario, not a validation of full 3-D drone dynamics or real-world transfer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

import crazyflow  # noqa: I001  # Configure SciPy array API before scipy is imported.
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Dynamics, Sim

HERE = Path(__file__).resolve().parent
DT = 1 / 500
CONTROL_STEPS = 10
DURATION = 8.0
MASS = 0.0434
GRAVITY = 9.81
COEFF = np.array([0.0, -3.133427287299859e-7, 4.407354891648379e-10])
ROTOR_DYN = np.array(
    [13.996001897562685, 0.00011093207920685363, 5.933168530682111, 0.00031951312393561264]
)
DRAG_Z = -0.02477138502714662
MIN_FORCE = 0.02136263065537499
MAX_FORCE = 0.2

plt.rcParams.update({"font.family": "Noto Sans CJK SC", "axes.unicode_minus": False})

XML = """<mujoco model="independent_drone">
<option timestep="0.002" gravity="0 0 -9.81" integrator="Euler"/>
<visual><global offwidth="640" offheight="480"/></visual>
<worldbody>
  <light pos="0 -2 5" dir="0 0 -1"/>
  <geom name="floor" type="plane" size="3 3 .1" contype="0" conaffinity="0" rgba=".3 .35 .4 1"/>
  <body name="drone" pos="0 0 1">
    <freejoint/>
    <inertial pos="0 0 0" mass=".0434" diaginertia=".000025 .000028 .000049"/>
    <geom type="box" size=".07 .07 .015" rgba=".1 .45 .8 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" pos=".035 .035 .019" size=".028 .002"
          rgba=".1 .8 .4 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" pos="-.035 .035 .019" size=".028 .002"
          rgba=".1 .8 .4 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" pos=".035 -.035 .019" size=".028 .002"
          rgba=".1 .8 .4 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" pos="-.035 -.035 .019" size=".028 .002"
          rgba=".1 .8 .4 1" contype="0" conaffinity="0"/>
  </body>
</worldbody>
</mujoco>"""


def force_to_rpm(force: float) -> float:
    """Invert the matching per-rotor thrust polynomial over its positive branch."""
    force = float(np.clip(force, MIN_FORCE, MAX_FORCE))
    return float(
        (-COEFF[1] + np.sqrt(COEFF[1] ** 2 - 4 * COEFF[2] * (COEFF[0] - force))) / (2 * COEFF[2])
    )


def reference(t: float) -> tuple[float, float]:
    """Continuous shared 4 s periodic height target and feedforward velocity."""
    omega = 2 * np.pi / 4
    return 1.0 + 0.25 * np.sin(omega * t), 0.25 * omega * np.cos(omega * t)


def controller(t: float, z: float, vz: float) -> float:
    """The same frozen altitude controller, evaluated from each plant's own state."""
    target, target_v = reference(t)
    thrust = MASS * (GRAVITY + 5.0 * (target - z) + 3.0 * (target_v - vz))
    return force_to_rpm(thrust / 4)


def main() -> None:
    """Run and record both independent closed loops and produce evidence image."""
    sim = Sim(control=Control.rotor_vel, dynamics=Dynamics.first_principles, device="cpu", freq=500)
    sim.reset()
    hover_rpm = force_to_rpm(MASS * GRAVITY / 4)
    states = sim.data.states
    states = states.replace(
        pos=states.pos.at[0, 0, 2].set(1.0), rotor_vel=jnp.full_like(states.rotor_vel, hover_rpm)
    )
    sim.data = sim.data.replace(states=states)
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    motor_rpm = hover_rpm
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    records = []
    screenshot_cf = screenshot_mj = None
    try:
        for n in range(int(DURATION / (DT * CONTROL_STEPS)) + 1):
            t = n * DT * CONTROL_STEPS
            z_cf = float(sim.data.states.pos[0, 0, 2])
            v_cf = float(sim.data.states.vel[0, 0, 2])
            z_mj, v_mj = float(data.qpos[2]), float(data.qvel[2])
            target, _ = reference(t)
            rpm_cf, rpm_mj = controller(t, z_cf, v_cf), controller(t, z_mj, v_mj)
            records.append([t, target, z_cf, z_mj, v_cf, v_mj, rpm_cf, rpm_mj])
            if n == 100:
                screenshot_cf = sim.render(
                    mode="rgb_array",
                    width=640,
                    height=480,
                    cam_config={
                        "distance": 0.8,
                        "elevation": -25.0,
                        "azimuth": -135.0,
                        "lookat": [0.0, 0.0, 1.0],
                    },
                )
                renderer = mujoco.Renderer(model, height=480, width=640)
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [0.0, 0.0, 1.0]
                camera.distance = 0.8
                camera.azimuth = -135
                camera.elevation = -25
                renderer.update_scene(data, camera=camera)
                screenshot_mj = renderer.render().copy()
                renderer.close()
            if n == int(DURATION / (DT * CONTROL_STEPS)):
                break
            sim.rotor_vel_control(jnp.full((1, 1, 4), rpm_cf))
            sim.step(CONTROL_STEPS)
            for _ in range(CONTROL_STEPS):
                thrust = 4 * (COEFF[0] + COEFF[1] * motor_rpm + COEFF[2] * motor_rpm**2)
                data.xfrc_applied[body, 2] = thrust + DRAG_Z * float(data.qvel[2])
                mujoco.mj_step(model, data)
                a, b = ROTOR_DYN[:2] if rpm_mj > motor_rpm else ROTOR_DYN[2:]
                motor_rpm += DT * (a * (rpm_mj - motor_rpm) + b * (rpm_mj**2 - motor_rpm**2))
    finally:
        sim.close()
    samples = np.asarray(records)
    assert np.isfinite(samples).all()
    np.savetxt(
        HERE / "sim2sim_trace.csv",
        samples,
        delimiter=",",
        comments="",
        header=(
            "time_s,target_z_m,crazyflow_z_m,mujoco_z_m,crazyflow_vz_mps,"
            "mujoco_vz_mps,crazyflow_rpm,mujoco_rpm"
        ),
    )
    summary = {
        "scenario": (
            "independent closed-loop vertical tracking, one drone, 8 s, "
            "50 Hz controller, 500 Hz plant"
        ),
        "crazyflow_version": getattr(crazyflow, "__version__", "unknown"),
        "jax_version": jax.__version__,
        "mujoco_version": mujoco.__version__,
        "control": (
            "same frozen altitude position and velocity feedback; "
            "each plant uses its own observation"
        ),
        "mujoco_physics": (
            "mj_step free rigid body, gravity, external vertical thrust and "
            "linear drag; Python motor ODE"
        ),
        "crazyflow_track_rmse_m": float(np.sqrt(np.mean((samples[:, 2] - samples[:, 1]) ** 2))),
        "mujoco_track_rmse_m": float(np.sqrt(np.mean((samples[:, 3] - samples[:, 1]) ** 2))),
        "cross_engine_rmse_m": float(np.sqrt(np.mean((samples[:, 2] - samples[:, 3]) ** 2))),
        "cross_engine_max_abs_m": float(np.max(np.abs(samples[:, 2] - samples[:, 3]))),
        "samples": len(samples),
        "seed": "deterministic; no random sampling",
        "limits": (
            "Only 1-D vertical, zero attitude, no contact/disturbance, shared coefficients; "
            "not Sim2Real or trained neural policy transfer."
        ),
    }
    (HERE / "sim2sim_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(screenshot_cf)
    ax1.axis("off")
    ax1.set_title("Crazyflow 动力学闭环｜2.0 秒")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(screenshot_mj)
    ax2.axis("off")
    ax2.set_title("独立 MuJoCo 动力学闭环｜2.0 秒")
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(samples[:, 0], samples[:, 1], "k--", label="目标高度")
    ax3.plot(samples[:, 0], samples[:, 2], label="Crazyflow 实际高度")
    ax3.plot(samples[:, 0], samples[:, 3], label="MuJoCo 实际高度", alpha=0.8)
    ax3.set(
        xlabel="时间（s）",
        ylabel="高度（m）",
        title=(
            "各自使用状态反馈的独立闭环记录｜"
            f"跨引擎高度差 RMSE = {summary['cross_engine_rmse_m']:.6f} m"
        ),
    )
    ax3.grid(alpha=0.25)
    ax3.legend()
    fig.savefig(HERE / "sim2sim_evidence.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
