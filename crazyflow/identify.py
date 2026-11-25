"""Run the retained official dynamics-identification pipeline on synchronized flight logs.

This module only adapts files and validates units/shapes. The fitted models, State Variable
Filter, and least-squares routines remain in ``crazyflow.dynamics.utils``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from crazyflow.drones import Drone
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.so_rpy.dynamics import dynamics_euler
from crazyflow.dynamics.utils.data_utils import derivatives_svf, preprocessing
from crazyflow.dynamics.utils.identification import sys_id_rotation, sys_id_translation

CSV_COLUMNS = (
    "time",
    "pos_x",
    "pos_y",
    "pos_z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
    "cmd_roll",
    "cmd_pitch",
    "cmd_yaw",
    "cmd_f",
)


def load_log(path: Path) -> dict[str, np.ndarray]:
    """Read NPZ or CSV, enforcing the official flight-log schema and SI units."""
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as source:
            missing = {"time", "pos", "quat", "cmd_rpy", "cmd_f"} - set(source.files)
            if missing:
                raise ValueError(f"{path}: missing arrays {sorted(missing)}")
            data = {
                name: np.array(source[name], dtype=np.float64, copy=True)
                for name in ("time", "pos", "quat", "cmd_rpy", "cmd_f")
            }
    elif path.suffix.lower() == ".csv":
        table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
        missing = set(CSV_COLUMNS) - set(table.dtype.names or ())
        if missing:
            raise ValueError(f"{path}: missing CSV columns {sorted(missing)}")
        data = {
            "time": np.atleast_1d(table["time"]),
            "pos": np.column_stack([table[f"pos_{axis}"] for axis in "xyz"]),
            "quat": np.column_stack([table[f"quat_{axis}"] for axis in "xyzw"]),
            "cmd_rpy": np.column_stack([table[f"cmd_{axis}"] for axis in ("roll", "pitch", "yaw")]),
            "cmd_f": np.atleast_1d(table["cmd_f"]),
        }
    else:
        raise ValueError("Flight log must be .npz or .csv")
    count = len(data["time"])
    expected = {
        "time": (count,),
        "pos": (count, 3),
        "quat": (count, 4),
        "cmd_rpy": (count, 3),
        "cmd_f": (count,),
    }
    for name, shape in expected.items():
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(f"{name}: expected finite array of shape {shape}")
    if count < 100:
        raise ValueError("Record at least 100 samples; use longer excited flight logs for hardware")
    dt = np.diff(data["time"])
    if np.any(dt <= 0):
        raise ValueError("time must be strictly increasing after synchronization")
    if data["time"][-1] - data["time"][0] < 2:
        raise ValueError("Record at least 2 seconds; use longer excited flight logs for hardware")
    if np.mean(dt) >= 1 / 20:
        raise ValueError(
            "Mean sampling rate must exceed 20 Hz for the official 8 Hz attitude filter"
        )
    if np.max(dt) > 3 * np.median(dt):
        raise ValueError("Timestamp gaps exceed 3 sample periods; segment/resample the log first")
    norms = np.linalg.norm(data["quat"], axis=-1)
    if np.any(norms < 1e-6):
        raise ValueError("quat contains a zero quaternion; expected xyzw")
    data["quat"] /= norms[:, None]
    # Remove equivalent quaternion sign flips before official averaging/interpolation.
    for index in range(1, count):
        if np.dot(data["quat"][index - 1], data["quat"][index]) < 0:
            data["quat"][index] *= -1
    if np.max(np.abs(data["cmd_rpy"])) > np.pi:
        raise ValueError("cmd_rpy must contain attitude angles in radians, not degrees/yaw rates")
    if np.any(data["cmd_f"] < 0) or np.max(data["cmd_f"]) > 5:
        raise ValueError("cmd_f must be collective thrust in Newtons, not PWM or per-motor thrust")
    return data


def synthetic_drag_log(path: Path, seconds: float, seed: int) -> None:
    """Generate known nonzero drag/lag using the unchanged official forward model."""
    from crazyflow.dynamics.so_rpy_rotor_drag.dynamics import dynamics_euler as forward

    if not np.isfinite(seconds) or seconds < 6:
        raise ValueError("Drag excitation needs at least 6 seconds; use 15 seconds for the example")
    if path.suffix.lower() not in {".csv", ".npz"}:
        raise ValueError("Synthetic drag output must end in .csv or .npz")
    params = load_params(Dynamics.so_rpy_rotor_drag, Drone.cf21B_500)
    keys = (
        "mass",
        "gravity_vec",
        "thrust_dyn_coef",
        "acc_coef",
        "cmd_f_coef",
        "rpy_coef",
        "rpy_rates_coef",
        "cmd_rpy_coef",
        "drag_matrix",
    )
    physical = {name: np.asarray(params[name]) for name in keys}
    dt = 0.01
    time = np.arange(0, seconds + dt / 2, dt)
    phase = np.random.default_rng(seed).uniform(-np.pi, np.pi, 6)
    elapsed = np.maximum(time - 0.5, 0.0)
    rise = 0.5 - 0.5 * np.cos(np.pi * np.clip(elapsed / 0.75, 0, 1))
    fall = 0.5 - 0.5 * np.cos(np.pi * np.clip((seconds - 0.5 - time) / 1.5, 0, 1))
    window = rise * fall
    cmd_rpy = window[:, None] * np.column_stack(
        (
            0.08 * np.sin(2 * np.pi * 0.17 * elapsed + phase[0])
            + 0.04 * np.sin(2 * np.pi * 0.37 * elapsed + phase[1]),
            0.09 * np.sin(2 * np.pi * 0.21 * elapsed + phase[2])
            + 0.03 * np.sin(2 * np.pi * 0.41 * elapsed + phase[3]),
            0.06 * np.sin(2 * np.pi * 0.37 * elapsed + phase[4]),
        )
    )
    hover = float(physical["mass"] * 9.81 / physical["cmd_f_coef"])
    cmd_f = hover * (
        1
        + window
        * (
            0.15 * np.sin(2 * np.pi * 0.33 * elapsed + phase[4])
            + 0.08 * np.sin(2 * np.pi * 0.73 * elapsed + phase[5])
        )
    )
    commands = np.column_stack((cmd_rpy, cmd_f))
    state = np.zeros((4, 3))
    state[0, 2] = 1.0
    thrust = np.array([hover])
    states = [state.copy()]
    for action in commands[:-1]:
        derivative = forward(*state, action, rotor_vel=thrust, **physical)
        state = state + dt * np.stack(derivative[:4])
        thrust = thrust + dt * derivative[4]
        states.append(state.copy())
    trajectory = np.stack(states)
    quat = Rotation.from_euler("xyz", trajectory[:, 1]).as_quat()
    known = {name: np.asarray(value).tolist() for name, value in physical.items()}
    provenance = {
        "synthetic": True,
        "source": "official cf21B_500 so_rpy_rotor_drag forward model; not hardware data",
        "model": "so_rpy_rotor_drag",
        "seed": seed,
        "duration_s": seconds,
        "dt_s": dt,
        "integrator": "explicit Euler, matching the official identification discretization",
        "known_parameters": known,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        table = np.column_stack((time, trajectory[:, 0], quat, cmd_rpy, cmd_f))
        np.savetxt(
            path, table, delimiter=",", header=",".join(CSV_COLUMNS), comments="", fmt="%.12g"
        )
    elif path.suffix.lower() == ".npz":
        np.savez_compressed(
            path,
            time=time,
            pos=trajectory[:, 0],
            quat=quat,
            cmd_rpy=cmd_rpy,
            cmd_f=cmd_f,
            synthetic=np.array(True),
            source=np.array(provenance["source"]),
            known_parameters_json=np.array(json.dumps(known)),
        )
    else:
        raise ValueError("Synthetic drag output must end in .csv or .npz")
    path.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote SYNTHETIC nonzero-drag/lag data: {path}")


def synthetic_log(path: Path, seconds: float, seed: int, dynamics: str = "so_rpy") -> None:
    """Generate explicitly synthetic so_rpy data to exercise the complete file-to-fit pipeline."""
    if dynamics == "so_rpy_rotor_drag":
        synthetic_drag_log(path, seconds, seed)
        return
    if dynamics != "so_rpy":
        raise ValueError(f"Unsupported synthetic model: {dynamics}")
    if seconds < 3:
        raise ValueError("Synthetic trajectory must last at least 3 seconds")
    params = load_params(Dynamics.so_rpy, Drone.cf21B_500, xp=jnp)
    keys = (
        "mass",
        "gravity_vec",
        "acc_coef",
        "cmd_f_coef",
        "rpy_coef",
        "rpy_rates_coef",
        "cmd_rpy_coef",
    )
    params = {name: params[name] for name in keys}
    dt = 0.01
    time = np.arange(0, seconds + dt / 2, dt)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, size=4)
    elapsed = np.maximum(time - 0.5, 0.0)
    window = 1 - np.exp(-4 * elapsed)
    # Held independent attitude steps excite both transient and steady response.
    command_index = (elapsed / 0.4).astype(int)
    levels = rng.uniform(-0.06, 0.06, size=(command_index.max() + 1, 3))
    cmd_rpy = levels[command_index] * window[:, None]
    hover = float(params["mass"] * 9.81 / params["cmd_f_coef"])
    cmd_f = hover * (1 + 0.12 * window * np.sin(2 * np.pi * 0.63 * elapsed + phase[3]))
    cmd = jnp.asarray(np.column_stack((cmd_rpy, cmd_f)))

    def step(state: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        deriv = jnp.stack(dynamics_euler(*state, action, **params))
        state = state + dt * deriv  # Matches official identification's Euler discretization.
        return state, state

    state0 = jnp.zeros((4, 3)).at[0, 2].set(1.0)
    _, states = jax.jit(lambda: jax.lax.scan(step, state0, cmd[:-1]))()
    states = np.concatenate((np.asarray(state0)[None], np.asarray(states)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        time=time,
        pos=states[:, 0],
        quat=Rotation.from_euler("xyz", states[:, 1]).as_quat(),
        cmd_rpy=cmd_rpy,
        cmd_f=cmd_f,
        synthetic=np.array(True),
        source=np.array("official cf21B_500 so_rpy; not hardware data"),
    )
    print(f"Wrote SYNTHETIC pipeline-check data: {path}")


def _marked_synthetic(path: Path) -> bool:
    """Recognize explicit example metadata without asserting unmarked logs are hardware."""
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return (
                "synthetic" in data
                and np.asarray(data["synthetic"]).shape == ()
                and bool(data["synthetic"])
            )
    metadata = path.with_suffix(".json")
    if metadata.is_file():
        try:
            return json.loads(metadata.read_text()).get("synthetic") is True
        except (json.JSONDecodeError, AttributeError):
            return False
    return False


def fit(args: argparse.Namespace) -> None:
    """Call official preprocess/SVF and identification, then write a reviewable parameter file."""
    if not np.isfinite(args.mass) or args.mass <= 0:
        raise ValueError("--mass must be measured takeoff mass in kg")
    if args.validation and args.validation.resolve() == args.data.resolve():
        raise ValueError("--validation must be a separate held-out flight, not the training file")
    marked_synthetic = _marked_synthetic(args.data) or bool(
        args.validation and _marked_synthetic(args.validation)
    )
    data = derivatives_svf(preprocessing(load_log(args.data)))
    validation = (
        derivatives_svf(preprocessing(load_log(args.validation))) if args.validation else None
    )
    translation = sys_id_translation(
        args.dynamics, args.mass, data, validation, verbose=args.verbose, plot=args.plot
    )
    rotation = sys_id_rotation(data, validation, verbose=args.verbose, plot=args.plot)
    fitted = {name: np.asarray(value).tolist() for name, value in (translation | rotation).items()}
    if not all(np.isfinite(np.asarray(value)).all() for value in fitted.values()):
        raise ValueError("Identification returned nonfinite coefficients")
    if (
        fitted["cmd_f_coef"] <= 0
        or np.any(np.asarray(fitted["rpy_coef"]) >= 0)
        or np.any(np.asarray(fitted["rpy_rates_coef"]) >= 0)
        or np.any(np.asarray(fitted["cmd_rpy_coef"]) <= 0)
    ):
        raise ValueError(
            "Fitted model is not stable/positive; inspect excitation, units and time alignment"
        )
    if "thrust_dyn_coef" in fitted and fitted["thrust_dyn_coef"] <= 0:
        raise ValueError(
            "Nonpositive thrust response coefficient; inspect excitation/time alignment"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.dynamics,
        "drone": "cf21B_500",
        "marked_synthetic": marked_synthetic,
        "acquisition_provenance": (
            "Explicit SYNTHETIC input; not hardware measurements"
            if marked_synthetic
            else "Provided logs; hardware acquisition provenance not verified"
        ),
        "measured_mass_kg": args.mass,
        "training_log": str(args.data),
        "validation_log": str(args.validation) if args.validation else None,
        "samples": len(data["time"]),
        "identified": fitted,
        "inherited_defaults": ["J", "thrust_min", "thrust_max", "acc_coef"],
        "note": (
            "Validate on an independent hardware flight; "
            "synthetic smoke is not a hardware identification."
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    # Export the full nominal table so it can be consumed without changing the official defaults.
    params = load_params(Dynamics(args.dynamics), Drone.cf21B_500)
    params = {
        name: np.asarray(value).tolist()
        for name, value in params.items()
        if name not in {"J_inv", "gravity_vec"}
    }
    params.update(mass=args.mass, **fitted)
    if "drag_xy_coef" in params:
        xy, z = params.pop("drag_xy_coef"), params.pop("drag_z_coef")
        params["drag_matrix"] = np.diag([xy, xy, z]).tolist()
    lines = [
        "# Official least-squares identification; SI units.",
        "# J, thrust limits, acc_coef inherit cf21B_500 defaults; measure separately if changed.",
        "# No hardware parameters are validated by the synthetic smoke test.",
        "[cf21B_500]",
    ]
    if marked_synthetic:
        lines.insert(0, "# SYNTHETIC input logs: not hardware-identified parameters.")
    lines += [f"{name} = {json.dumps(value)}" for name, value in params.items()]
    args.output.write_text("\n".join(lines) + "\n")
    print(f"Parameters: {args.output}; provenance: {args.output.with_suffix('.json')}")


def main() -> None:
    """Entry point for the data adapter and retained official system identification."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser(
        "template", help="Write the synchronized CSV schema (header only)"
    )
    template.add_argument("--output", type=Path, default=Path("outputs/flight_template.csv"))
    synthetic = commands.add_parser(
        "synthetic", help="Generate synthetic pipeline-check data, not measurements"
    )
    synthetic.add_argument("--output", type=Path, default=Path("outputs/sysid_synthetic.npz"))
    synthetic.add_argument("--seconds", type=float, default=8.0)
    synthetic.add_argument("--seed", type=int, default=0)
    synthetic.add_argument(
        "--dynamics",
        choices=("so_rpy", "so_rpy_rotor_drag"),
        default="so_rpy",
        help="Forward model; rotor_drag includes known nonzero drag and thrust lag",
    )
    fitting = commands.add_parser(
        "fit", help="Fit the official attitude/collective-thrust dynamics"
    )
    fitting.add_argument("--data", type=Path, required=True)
    fitting.add_argument("--validation", type=Path)
    fitting.add_argument("--mass", type=float, required=True, help="Measured takeoff mass, kg")
    fitting.add_argument(
        "--dynamics", choices=("so_rpy", "so_rpy_rotor", "so_rpy_rotor_drag"), default="so_rpy"
    )
    fitting.add_argument("--output", type=Path, default=Path("outputs/identified.toml"))
    fitting.add_argument("--plot", action="store_true")
    fitting.add_argument("--verbose", choices=(0, 1, 2), type=int, default=0)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if args.command == "template":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(",".join(CSV_COLUMNS) + "\n")
        print(args.output)
    elif args.command == "synthetic":
        synthetic_log(args.output, args.seconds, args.seed, args.dynamics)
    else:
        fit(args)


if __name__ == "__main__":
    main()
