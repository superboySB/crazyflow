"""Reproduce the small feature experiments added when research synchronized main.

Run from the repository root, for example::

    python usage_assets/research_update_experiments.py --device gpu
    XLA_FLAGS=--xla_force_host_platform_device_count=2 \
        python usage_assets/research_update_experiments.py --mode sharding --device cpu

The CPU device count must be set before Python starts. These experiments check
software behavior and model predictions; they do not validate physical drones or
measure multi-GPU performance. Existing example plugins are imported unchanged.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Support both an editable install and direct execution from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as R

import crazyflow
import crazyflow.sim.functional as F
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.sim import Sim
from crazyflow.sim.data import SimStateDeriv
from crazyflow.sim.pipeline import append_fn, insert_fn_before, remove_fn, replace_fn
from crazyflow.sim.sharding import world_mesh
from crazyflow.sim.sim import rotor_vel_limits
from crazyflow.utils import CORE_NDIM_KEY, world_mask
from examples.jax.gradient_clipping import clip_rotor_vel_cmd_nonblocking
from examples.plugins.derivatives import dynamics_deriv, finite_diff_deriv
from examples.plugins.downwash import downwash_fn
from examples.plugins.ground_effect import ground_effect_fn

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.sim.data import SimData

ROOT = Path(__file__).resolve().parents[1]


@flax.struct.dataclass
class Counter:
    """Declare which plugin field belongs to individual worlds."""

    count: Array = flax.struct.field(metadata={CORE_NDIM_KEY: 1})
    lookup: Array


def count_step(data: SimData) -> SimData:
    """Increment the per-world count, preserving the shared lookup."""
    counter = data.plugins["counter"]
    return data.replace(
        plugins=data.plugins | {"counter": counter.replace(count=counter.count + 1)}
    )


def airborne(sim: Sim, height: float | np.ndarray = 1.0) -> float:
    """Initialize level flight with the model's weight-balancing motor speed."""
    weight = float(np.asarray(sim.data.params.mass).reshape(-1)[0]) * float(
        -sim.data.params.gravity_vec[2]
    )
    if sim.dynamics == "first_principles":
        rotor = motor_force2rotor_vel(
            jnp.asarray(weight / 4, device=sim.device), sim.data.params.rpm2thrust
        )
    else:
        rotor = (weight - sim.data.params.acc_coef) / sim.data.params.cmd_f_coef
    states = sim.data.states
    height_array = jnp.asarray(height, device=sim.device)
    if height_array.ndim:
        height_array = height_array[:, None]
    states = states.replace(
        pos=states.pos.at[..., 2].set(height_array),
        rotor_vel=jnp.broadcast_to(rotor, states.rotor_vel.shape),
    )
    sim.data = sim.data.replace(states=states)
    return weight


def state_command(sim: Sim, height: float | np.ndarray = 1.0) -> np.ndarray:
    """Create a 16D command using the documented xyzw quaternion convention."""
    cmd = np.zeros((sim.n_worlds, sim.n_drones, 16), dtype=np.float32)
    cmd[..., 2] = np.asarray(height).reshape(-1, 1) if np.ndim(height) else height
    cmd[..., 12] = 1.0
    return cmd


def hover_experiment(device: str) -> dict[str, Any]:
    """Compare default Crazyflie and newly added X500 model hovering."""
    results = {}
    for drone in ("cf21B_500", "hb_x500"):
        dynamics = "so_rpy_rotor_drag" if drone == "hb_x500" else "first_principles"
        sim = Sim(drone=drone, dynamics=dynamics, control="state", device=device)
        weight = airborne(sim)
        command = state_command(sim)
        sim.state_control(command)
        sim.step(7 * sim.freq)
        samples = []
        for _ in range(sim.control_freq):  # 8 s, sample the final second at 100 Hz.
            sim.step(sim.freq // sim.control_freq)
            thrust = (
                sim.data.controls.force_torque.cmd[0, 0, 0]
                if dynamics == "first_principles"
                else sim.data.controls.attitude.cmd[0, 0, 3]
            )
            samples.append([float(sim.data.states.pos[0, 0, 2]), float(thrust)])
        samples = np.asarray(samples)
        assert np.isfinite(samples).all()
        assert np.max(np.abs(samples[:, 0] - 1.0)) < 0.1
        results[drone] = {
            "dynamics": dynamics,
            "state_command_shape": list(command.shape),
            "model_mass_kg": float(sim.data.params.mass[0]),
            "controller_mass_kg": float(sim.data.controls.state.params["mass"]),
            "model_weight_N": weight,
            "mean_height_m": float(samples[:, 0].mean()),
            "max_height_error_final_second_m": float(np.max(np.abs(samples[:, 0] - 1.0))),
            "mean_thrust_command_N": float(samples[:, 1].mean()),
            "final_rotor_state": np.asarray(sim.data.states.rotor_vel[0, 0]).tolist(),
            "rotor_state_unit": "RPM" if dynamics == "first_principles" else "N",
        }
        sim.close()
    return {"duration_s": 8.0, "sample_s": 1.0, "models": results}


def state_feedforward_experiment(device: str) -> dict[str, Any]:
    """Exercise all 16 fields with analytic circular motion and yaw references."""
    sim = Sim(control="state", device=device)
    airborne(sim)
    errors = []
    for i in range(600):
        t = i / sim.control_freq
        command = state_command(sim)
        omega, radius = 0.5, 0.2
        command[..., :3] = [radius * np.sin(omega * t), radius * (1 - np.cos(omega * t)), 1.0]
        command[..., 3:6] = [
            radius * omega * np.cos(omega * t),
            radius * omega * np.sin(omega * t),
            0.0,
        ]
        command[..., 6:9] = [
            -radius * omega**2 * np.sin(omega * t),
            radius * omega**2 * np.cos(omega * t),
            0.0,
        ]
        command[..., 9:13] = np.asarray(R.from_euler("z", 0.1 * t).as_quat())
        command[..., 13:16] = [0.0, 0.0, 0.1]
        sim.state_control(command)
        sim.step(sim.freq // sim.control_freq)
        errors.append(np.asarray(sim.data.states.pos[0, 0]) - command[0, 0, :3])
    errors = np.asarray(errors)
    assert np.isfinite(errors).all()
    assert np.max(np.linalg.norm(errors, axis=1)) < 0.2
    result = {
        "duration_s": 6.0,
        "radius_m": 0.2,
        "angular_frequency_rad_s": 0.5,
        "yaw_rate_reference_rad_s": 0.1,
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
        "max_position_error_m": float(np.max(np.linalg.norm(errors, axis=1))),
        "final_yaw_rad": float(R.from_quat(sim.data.states.quat[0, 0]).as_euler("xyz")[2]),
    }
    sim.close()
    return result


def body_rate_experiment(device: str) -> dict[str, Any]:
    """Track a yaw body rate using the example's disabled attitude terms."""
    sim = Sim(control="body_rate", body_rate_freq=250, device=device)
    weight = airborne(sim)
    body_rate = sim.data.controls.body_rate
    params = body_rate.params | {"kR": jnp.zeros(3), "ki_m": jnp.zeros(3)}
    sim.data = sim.data.replace(
        controls=sim.data.controls.replace(body_rate=body_rate.replace(params=params))
    )
    command = np.asarray([[[0.0, 0.0, 0.4, weight]]], dtype=np.float32)
    samples = []
    for _ in range(500):  # 2 s at 250 Hz.
        sim.body_rate_control(command)
        sim.step(sim.freq // sim.control_freq)
        samples.append(np.asarray(sim.data.states.ang_vel[0, 0]).copy())
    samples = np.asarray(samples)
    measured = samples[-125:].mean(axis=0)
    assert np.isfinite(samples).all()
    assert abs(measured[2] - 0.4) < 0.05
    result = {
        "duration_s": 2.0,
        "body_rate_hz": sim.control_freq,
        "command": command[0, 0].tolist(),
        "mean_final_half_second_rates_rad_s": measured.tolist(),
        "final_yaw_rad": float(R.from_quat(sim.data.states.quat[0, 0]).as_euler("xyz")[2]),
        "final_height_m": float(sim.data.states.pos[0, 0, 2]),
    }
    sim.close()
    return result


def reset_experiment(device: str) -> dict[str, Any]:
    """Check masked plugin reset and the handling of shared parameters."""
    sim = Sim(n_worlds=3, device=device)
    airborne(sim)
    counter = Counter(jnp.zeros((3, 1), dtype=jnp.int32), jnp.asarray([42]))
    sim.data = sim.data.replace(plugins={"counter": counter})
    append_fn(sim.step_pipeline, count_step)
    sim.build_default_data()
    sim.build_step_fn()
    sim.step(7)
    sim.data = sim.data.replace(
        plugins={"counter": sim.data.plugins["counter"].replace(lookup=jnp.asarray([99]))},
        params=sim.data.params.replace(gravity_vec=jnp.asarray([0.0, 0.0, -8.0])),
    )
    declarations = world_mask(sim.data)
    sim.reset(jnp.asarray([True, False, True]))
    counts = np.asarray(sim.data.plugins["counter"].count).flatten()
    steps = np.asarray(sim.data.core.steps).flatten()
    assert np.array_equal(counts, [0, 7, 0])
    assert np.array_equal(steps, [0, 7, 0])
    assert int(sim.data.plugins["counter"].lookup[0]) == 99
    assert float(sim.data.params.gravity_vec[2]) == -8.0
    result = {
        "mask": [True, False, True],
        "counts_after_masked_reset": counts.tolist(),
        "steps_after_masked_reset": steps.tolist(),
        "shared_lookup_after_masked_reset": 99,
        "shared_gravity_z_after_masked_reset_m_s2": -8.0,
        "counter_has_world_axis": bool(declarations.plugins["counter"].count),
        "lookup_has_world_axis": bool(declarations.plugins["counter"].lookup),
        "gravity_has_world_axis": bool(declarations.params.gravity_vec),
    }
    sim.reset()
    assert np.all(np.asarray(sim.data.plugins["counter"].count) == 0)
    assert int(sim.data.plugins["counter"].lookup[0]) == 42
    assert np.isclose(float(sim.data.params.gravity_vec[2]), -9.81)
    result["full_reset_restores_shared_fields"] = True
    sim.close()
    return result


def ground_effect_experiment(device: str) -> dict[str, Any]:
    """Run matched hover setpoints with and without the upstream ground plugin."""
    heights = np.asarray([0.50, 0.20, 0.10, 0.05], dtype=np.float32)
    results = {}
    for name, plugin in (("baseline", None), ("ground_effect", ground_effect_fn)):
        sim = Sim(n_worlds=len(heights), control="state", device=device)
        weight = airborne(sim, heights)
        if plugin is not None:
            insert_fn_before(sim.step_pipeline, "integration", plugin)
            sim.build_step_fn()
        command = state_command(sim, heights)
        sim.state_control(command)
        sim.step(9 * sim.freq)
        samples = []
        for _ in range(sim.control_freq):  # 10 s; last second has 100 samples per world.
            sim.step(sim.freq // sim.control_freq)
            samples.append(
                np.stack(
                    [
                        np.asarray(sim.data.states.pos[:, 0, 2]),
                        np.asarray(sim.data.controls.force_torque.cmd[:, 0, 0]),
                        np.asarray(sim.data.states.force[:, 0, 2]),
                    ],
                    axis=-1,
                )
            )
        samples = np.asarray(samples)
        assert np.isfinite(samples).all()
        means = samples.mean(axis=0)
        results[name] = {
            "mean_measured_heights_m": means[:, 0].tolist(),
            "height_std_m": samples[:, :, 0].std(axis=0).tolist(),
            "mean_thrust_commands_N": means[:, 1].tolist(),
            "mean_external_force_z_N": means[:, 2].tolist(),
        }
        sim.close()
    baseline = np.asarray(results["baseline"]["mean_thrust_commands_N"])
    ground = np.asarray(results["ground_effect"]["mean_thrust_commands_N"])
    assert np.all(ground < baseline)
    return {
        "setpoint_heights_m": heights.tolist(),
        "duration_s": 10.0,
        "sample_s": 1.0,
        "weight_N": weight,
        "thrust_reduction_percent": (100 * (baseline - ground) / baseline).tolist(),
        "runs": results,
    }


def downwash_experiment(device: str) -> dict[str, Any]:
    """Evaluate the same hover rotor state at controlled vertical and radial offsets."""
    offsets = np.asarray([0.0, 0.10, 0.25, 0.50], dtype=np.float32)
    sim = Sim(n_worlds=len(offsets), n_drones=2, device=device)
    airborne(sim)
    pos = jnp.zeros_like(sim.data.states.pos)
    pos = pos.at[:, 0, 2].set(1.2).at[:, 1, 2].set(0.5).at[:, 1, 0].set(offsets)
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))
    baseline_force = np.asarray(sim.data.states.force).copy()
    evaluated = jax.jit(downwash_fn)(sim.data)
    force = np.asarray(evaluated.states.force)
    torque = np.asarray(evaluated.states.torque)
    assert np.isfinite(force).all() and np.isfinite(torque).all()
    assert np.array_equal(baseline_force, np.zeros_like(baseline_force))
    assert np.max(np.abs(force[:, 0])) == 0.0
    assert np.all(force[:, 1, 2] < 0)
    assert np.all(np.diff(np.abs(force[:, 1, 2])) < 0)
    result = {
        "method": "Frozen equal hover RPM; plugin evaluation, no trajectory integration",
        "upper_height_m": 1.2,
        "lower_height_m": 0.5,
        "horizontal_offsets_m": offsets.tolist(),
        "upper_force_z_N": force[:, 0, 2].tolist(),
        "lower_force_z_N": force[:, 1, 2].tolist(),
        "lower_torque_Nm": torque[:, 1].tolist(),
        "baseline_force_z_N": baseline_force[:, 1, 2].tolist(),
    }
    sim.close()
    return result


def clipping_experiment(device: str) -> dict[str, Any]:
    """Compare clipping values and acceleration gradients from identical initial states."""
    sim = Sim(control="rotor_vel", device=device)
    airborne(sim, 2.0)
    lower, upper = rotor_vel_limits(sim.dynamics, sim.drone)
    commands = np.asarray([0.0, (lower + upper) / 2, upper + 10000], dtype=np.float32)
    results = {}
    for name in ("default_clip", "straight_through_clip", "no_clip"):
        if name == "straight_through_clip":
            replace_fn(
                sim.step_pipeline,
                partial(clip_rotor_vel_cmd_nonblocking, lower=lower, upper=upper),
                "clip_rotor_vel_cmd",
            )
        elif name == "no_clip":
            remove_fn(sim.step_pipeline, "clip_rotor_vel_cmd")
        step = sim.build_step_fn()

        def acceleration(cmd: Array) -> tuple[Array, tuple[Array, Array]]:
            data = F.rotor_vel_control(sim.data, jnp.full((1, 1, 4), cmd))
            first = step(data, 1)
            second = step(first, 1)
            acc_z = (second.states.vel[0, 0, 2] - first.states.vel[0, 0, 2]) * sim.freq
            return acc_z, (first.states.rotor_vel[0, 0, 0], first.controls.rotor_vel[0, 0, 0])

        evaluate = jax.jit(jax.value_and_grad(acceleration, has_aux=True))
        rows = []
        for cmd in commands:
            ((acc_z, (rpm, actual)), gradient) = evaluate(jnp.asarray(cmd, device=sim.device))
            rows.append([float(actual), float(rpm), float(acc_z), float(gradient)])
        rows = np.asarray(rows)
        assert np.isfinite(rows).all()
        results[name] = {
            "effective_motor_commands_rpm": rows[:, 0].tolist(),
            "next_rotor_state_rpm": rows[:, 1].tolist(),
            "lookahead_vertical_acceleration_m_s2": rows[:, 2].tolist(),
            "d_acc_z_d_command": rows[:, 3].tolist(),
        }
    assert np.allclose(
        results["default_clip"]["next_rotor_state_rpm"],
        results["straight_through_clip"]["next_rotor_state_rpm"],
        atol=1e-3,
    )
    assert results["default_clip"]["d_acc_z_d_command"][2] == 0.0
    assert results["straight_through_clip"]["d_acc_z_d_command"][2] > 0.0
    assert results["no_clip"]["effective_motor_commands_rpm"][2] > upper
    sim.close()
    return {"probe_commands_rpm": commands.tolist(), "limits_rpm": [lower, upper], "runs": results}


def derivatives_experiment(device: str) -> dict[str, Any]:
    """Compare pre-step dynamics derivatives with post-step finite differences."""
    results = {}
    for integrator in ("euler", "rk4"):
        sim = Sim(control="rotor_vel", integrator=integrator, device=device)
        airborne(sim, 2.0)
        states = sim.data.states.replace(
            vel=jnp.asarray([[[0.3, -0.1, 0.2]]], device=sim.device),
            ang_vel=jnp.asarray([[[0.03, -0.02, 0.04]]], device=sim.device),
        )
        sim.data = sim.data.replace(
            states=states,
            plugins={
                "states_deriv": SimStateDeriv.create(1, 1, sim.device),
                "fd_states_deriv": SimStateDeriv.create(1, 1, sim.device),
                "prev_states": states,
            },
        )
        insert_fn_before(sim.step_pipeline, "integration", dynamics_deriv)
        append_fn(sim.step_pipeline, finite_diff_deriv)
        sim.build_step_fn()
        command = np.asarray(states.rotor_vel) * np.asarray([1.05, 0.95, 1.02, 0.98])
        sim.rotor_vel_control(command)
        logs = {key: [] for key in ("states_deriv", "fd_states_deriv")}
        for _ in range(50):
            sim.step()
            for key in logs:
                logs[key].append(
                    jax.tree.map(lambda x: np.asarray(x).copy(), sim.data.plugins[key])
                )
        summary = {}
        for field in ("vel", "ang_vel", "acc", "ang_acc", "rotor_acc"):
            direct = np.stack([getattr(x, field) for x in logs["states_deriv"]])
            finite = np.stack([getattr(x, field) for x in logs["fd_states_deriv"]])
            error = float(np.max(np.abs(direct - finite)))
            scale = float(np.max(np.abs(direct)))
            relative = error / max(scale, 1e-12)
            assert np.isfinite(relative)
            if integrator == "euler":
                assert relative < 0.002, (field, relative)
            summary[field] = {"max_abs_difference": error, "max_relative_difference": relative}
        results[integrator] = summary
        sim.close()
    return {"steps": 50, "duration_s": 0.1, "runs": results}


def sharding_experiment(device: str) -> dict[str, Any]:
    """Check two or more real JAX device placements against unsharded values."""
    devices = jax.devices(device)
    if len(devices) < 2:
        raise RuntimeError("Sharding requires >=2 devices; set XLA_FLAGS before Python starts.")
    n_worlds = 2 * len(devices)
    sim = Sim(n_worlds=n_worlds, control="attitude", device=device)
    weight = airborne(sim)
    counter = Counter(jnp.zeros((n_worlds, 1), jnp.int32), jnp.asarray([42]))
    sim.data = sim.data.replace(plugins={"counter": counter})
    append_fn(sim.step_pipeline, count_step)
    sim.build_default_data()
    sim.build_step_fn()
    command = np.zeros((n_worlds, 1, 4), dtype=np.float32)
    command[..., :2] = [0.02, -0.03]
    command[:, 0, 2] = np.arange(n_worlds) * 0.1
    command[..., 3] = weight
    sim.attitude_control(command)
    sim.step(50)
    reference = jax.tree.map(lambda x: np.asarray(x).copy(), sim.data.states)
    sim.reset()
    sim.shard(world_mesh(devices))
    sim.attitude_control(command)
    sim.step(50)
    fields = {}
    relative_fields = {}
    for field in ("pos", "quat", "vel", "ang_vel", "rotor_vel", "force", "torque"):
        actual = np.asarray(getattr(sim.data.states, field))
        expected = getattr(reference, field)
        difference = float(np.max(np.abs(actual - expected)))
        # RPM states are O(1e4), where one float32 ULP is about 1e-3 RPM.
        # Use a scale-aware comparison while retaining the actual absolute error.
        assert np.allclose(actual, expected, atol=1e-6, rtol=1e-6), (field, difference)
        fields[field] = difference
        relative_fields[field] = difference / max(float(np.max(np.abs(expected))), 1e-12)
    assert fields["pos"] < 1e-6
    counter = sim.data.plugins["counter"]
    result = {
        "device_kind": device,
        "devices": [str(d) for d in devices],
        "n_worlds": n_worlds,
        "steps": 50,
        "max_abs_state_differences": fields,
        "max_relative_state_differences": relative_fields,
        "comparison_atol": 1e-6,
        "comparison_rtol": 1e-6,
        "position_partition_spec": str(sim.data.states.pos.sharding.spec),
        "gravity_partition_spec": str(sim.data.params.gravity_vec.sharding.spec),
        "plugin_counter_partition_spec": str(counter.count.sharding.spec),
        "shared_lookup_partition_spec": str(counter.lookup.sharding.spec),
        "worlds_per_shard": [s.data.shape[0] for s in sim.data.states.pos.addressable_shards],
        "performance_claim": "Functional correctness only; no multi-GPU timing measurement",
    }
    mask = jnp.arange(n_worlds) % 2 == 0
    sim.reset(mask)
    expected = np.where(np.arange(n_worlds) % 2 == 0, 0, 50)
    actual = np.asarray(sim.data.plugins["counter"].count).flatten()
    assert np.array_equal(actual, expected)
    result["counts_after_sharded_masked_reset"] = actual.tolist()
    sim.close()
    return result


def plot_results(results: dict[str, Any], path: Path) -> None:
    """Write a standalone comparison figure from the recorded numeric results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    ground = results["ground_effect"]
    for name, run in ground["runs"].items():
        axes[0].plot(
            run["mean_measured_heights_m"], run["mean_thrust_commands_N"], "o-", label=name
        )
    axes[0].set(
        xlabel="Measured hover height (m)", ylabel="Thrust command (N)", title="Ground effect"
    )
    axes[0].legend()
    downwash = results["downwash"]
    axes[1].plot(downwash["horizontal_offsets_m"], downwash["lower_force_z_N"], "o-", label="lower")
    axes[1].plot(
        downwash["horizontal_offsets_m"], downwash["upper_force_z_N"], "o--", label="upper"
    )
    axes[1].set(
        xlabel="Horizontal offset (m)",
        ylabel="External force z (N)",
        title="Downwash, frozen hover",
    )
    axes[1].legend()
    clipping = results["clipping"]
    x = np.arange(len(clipping["probe_commands_rpm"]))
    width = 0.25
    for i, (name, run) in enumerate(clipping["runs"].items()):
        axes[2].bar(x + (i - 1) * width, run["d_acc_z_d_command"], width=width, label=name)
    axes[2].set(
        xticks=x,
        xticklabels=["below min", "in range", "above max"],
        ylabel="d acceleration z / d RPM",
        title="Clipping gradient",
    )
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(results: dict[str, Any], path: Path) -> None:
    """Write a concise Markdown companion without changing the existing manuals."""
    hover = results["hover"]["models"]
    body = results["body_rate"]
    ground = results["ground_effect"]
    downwash = results["downwash"]
    lines = [
        "# 官方 main 增量同步：小规模可复现实验",
        "",
        "原始值、软件版本和所有断言见 `research_update_results.json`；"
        "脚本为 `research_update_experiments.py`。这些是模型/API 实验，"
        "没有实机验证或多 GPU 速度结论。",
        "",
        "| 实验 | 口径 | 实测结果 |",
        "|---|---|---|",
    ]
    for drone, run in hover.items():
        lines.append(
            f"| {drone} 悬停 | 8 s，末 1 s 100 个样本 | 高度 {run['mean_height_m']:.6f} m，"
            f"指令推力 {run['mean_thrust_command_N']:.6f} N |"
        )
    state = results["state_feedforward"]
    lines.extend(
        [
            f"| 16D state | 6 s 圆轨迹；位置、速度、加速度、四元数、角速度均赋值 | "
            f"位置 RMSE {state['position_rmse_m']:.6f} m |",
            f"| body_rate | 0.4 rad/s yaw + 模型重力推力，2 s，末 0.5 s | "
            f"yaw rate {body['mean_final_half_second_rates_rad_s'][2]:.6f} rad/s |",
            "| world axis/reset | 3 worlds，7 步后 reset [True, False, True] | "
            "计数 [0, 7, 0]；共享 lookup/gravity 保持，full reset 恢复 |",
            "",
            "## 地效对照",
            "",
            "每个 setpoint 独立 world，10 s 后取末 1 s；保留官方默认控制质量。"
            "表中同时给实际高度，不能将 setpoint 当作收敛高度。",
            "",
            "| setpoint (m) | baseline 高度 (m) | 地效高度 (m) | "
            "baseline 推力 (N) | 地效推力 (N) | 减少 (%) |",
            "|---|---|---|---|---|---|",
        ]
    )
    for i, height in enumerate(ground["setpoint_heights_m"]):
        baseline, plugin = ground["runs"]["baseline"], ground["runs"]["ground_effect"]
        lines.append(
            f"| {height:.2f} | {baseline['mean_measured_heights_m'][i]:.6f} | "
            f"{plugin['mean_measured_heights_m'][i]:.6f} | "
            f"{baseline['mean_thrust_commands_N'][i]:.6f} | "
            f"{plugin['mean_thrust_commands_N'][i]:.6f} | "
            f"{ground['thrust_reduction_percent'][i]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 下洗、梯度与导数",
            "",
            "下洗使用相同悬停 RPM 的冻结状态，上机 1.2 m、下机 0.5 m，直接计算插件 force/torque；"
            "这是受力对照，不是闭环轨迹结果。",
            "",
            "| 水平偏移 (m) | 上机 force z (N) | 下机 force z (N) |",
            "|---|---|---|",
        ]
    )
    for i, offset in enumerate(downwash["horizontal_offsets_m"]):
        lines.append(
            f"| {offset:.2f} | {downwash['upper_force_z_N'][i]:.6f} | "
            f"{downwash['lower_force_z_N'][i]:.6f} |"
        )
    clipping = results["clipping"]["runs"]
    lines.extend(
        [
            "",
            "从同一初始状态执行一个动力学步，再向前看一拍计算加速度。超过最大 RPM 时：",
            "",
            "| pipeline | 实际 motor command (RPM) | d acc_z / d command |",
            "|---|---|---|",
        ]
    )
    for name, run in clipping.items():
        lines.append(
            f"| {name} | {run['effective_motor_commands_rpm'][2]:.3f} | "
            f"{run['d_acc_z_d_command'][2]:.9f} |"
        )
    lines.extend(
        [
            "",
            "straight-through 的前向裁剪结果与默认一致，反向梯度是替代梯度；"
            "不能解释为饱和模型的真实导数。取消 clipping 的 motor command 超限。",
            "",
            "导数插件在 integration 前计算动力学导数，在步后计算有限差分。"
            "非对称 motor command 连续运行 50 步（0.1 s）：",
            "",
            "| quantity | Euler 最大相对差 | RK4 最大相对差 |",
            "|---|---|---|",
        ]
    )
    for field in ("vel", "ang_vel", "acc", "ang_acc", "rotor_acc"):
        runs = results["derivatives"]["runs"]
        lines.append(
            f"| {field} | {runs['euler'][field]['max_relative_difference']:.3e} | "
            f"{runs['rk4'][field]['max_relative_difference']:.3e} |"
        )
    lines.extend(
        [
            "",
            "Euler 中两者相同至 float32 数值误差；RK4 比较的是步前导数与跨步平均变化，"
            "其差值有离散化含义。CPU 分片结果另见 `research_update_sharding.json`。",
            "",
            "![地效、下洗与 clipping 梯度](research_update_20261009.png)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run selected checks and save inspectable numeric artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--mode", choices=("features", "sharding"), default="features")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "usage_assets")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "environment": {
            "python": platform.python_version(),
            "execution_environment": "Docker" if Path("/.dockerenv").exists() else "host",
            "hostname": platform.node(),
            "crazyflow": crazyflow.__version__,
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "jax",
                    "jaxlib",
                    "numpy",
                    "scipy",
                    "mujoco",
                    "mujoco-mjx",
                    "flax",
                    "matplotlib",
                )
            },
            "requested_device": args.device,
            "devices": [str(d) for d in jax.devices(args.device)],
            "device_kinds": [d.device_kind for d in jax.devices(args.device)],
            "dtype": "float32",
            "dynamics_hz": 500,
            "seed": 0,
            "official_main_commit": "70d09e4",
        }
    }
    if args.mode == "sharding":
        with jax.default_device(jax.devices(args.device)[0]):
            results["sharding"] = sharding_experiment(args.device)
        filename = "research_update_sharding.json"
    else:
        experiments = {
            "hover": hover_experiment,
            "state_feedforward": state_feedforward_experiment,
            "body_rate": body_rate_experiment,
            "world_axis_reset": reset_experiment,
            "ground_effect": ground_effect_experiment,
            "downwash": downwash_experiment,
            "clipping": clipping_experiment,
            "derivatives": derivatives_experiment,
        }
        for name, experiment in experiments.items():
            print(f"Running {name} on {args.device}...", flush=True)
            with jax.default_device(jax.devices(args.device)[0]):
                results[name] = experiment(args.device)
            print(json.dumps({name: results[name]}, ensure_ascii=False), flush=True)
        plot_results(results, args.output_dir / "research_update_20261009.png")
        write_report(results, args.output_dir / "research_update_results.md")
        filename = "research_update_results.json"
    results["all_assertions_passed"] = True
    output = args.output_dir / filename
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
