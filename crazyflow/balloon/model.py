"""Official cf21B_500 first-principles plant and persistent Mellinger state.

Training uses the same control, RPM limiting, integration and floor stages as
``Sim(dynamics=Dynamics.default, control=Control.attitude)``. MuJoCo scene creation
is unnecessary for these pure JAX stages. The 12-value vectors emitted for plots
are views of the full quaternion/RPM/controller state, never the training plant.
"""

from typing import TypedDict

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from crazyflow.balloon.config import Config
from crazyflow.control import Control
from crazyflow.control.core import load_fn_params as load_control_params
from crazyflow.control.mellinger import attitude2force_torque, force_torque2rotor_vel
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.drones import Drone
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.first_principles import Params as FirstPrinciplesParams
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from crazyflow.sim import functional as F
from crazyflow.sim.data import SimControls, SimCore, SimData, SimState
from crazyflow.sim.integration import Integrator
from crazyflow.sim.sim import (
    build_control_fns,
    clip_floor_pos,
    clip_rotor_vel_cmd,
    increment_steps,
    select_dynamics_fn,
    select_integrate_fn,
)

PhysicalParams = dict[str, jax.Array]
Domain = dict[str, jax.Array]
ControlParams = dict[str, dict[str, jax.Array]]


class Batch(TypedDict):
    """Privileged reset state, task geometry and sampled physical conditions."""

    state: jax.Array
    target: jax.Array
    radius: jax.Array
    domain: Domain


def nominal_params(cfg: Config | None = None) -> PhysicalParams:
    """Load every original default first-principles parameter without retuning."""
    params = load_params(Dynamics.default, Drone.cf21B_500, xp=jnp)
    if cfg is not None and cfg.model_params is not None:
        params.update({k: jnp.asarray(v) for k, v in cfg.model_params.items()})
    params["J_inv"] = jnp.linalg.inv(params["J"])
    return params


def control_params(cfg: Config | None = None) -> ControlParams:
    """Load the official attitude and allocation parameters for checkpointing."""
    params = {
        "attitude": load_control_params(attitude2force_torque, Drone.cf21B_500, xp=jnp),
        "force_torque": load_control_params(force_torque2rotor_vel, Drone.cf21B_500, xp=jnp),
    }
    if cfg is not None and getattr(cfg, "controller_params", None) is not None:
        for stage, values in cfg.controller_params.items():
            params[stage].update({k: jnp.asarray(v) for k, v in values.items()})
    return params


def rotation_matrix(rpy: jax.Array) -> jax.Array:
    """Body FLU to world ENU rotation matrix."""
    return Rotation.from_euler("xyz", rpy).as_matrix()


def body_rates(state: jax.Array) -> jax.Array:
    """Convert the legacy Euler-rate view to actual body gyroscope rates."""
    roll, pitch = state[..., 3], state[..., 4]
    dr, dp, dy = (state[..., i] for i in (9, 10, 11))
    return jnp.stack(
        (
            dr - jnp.sin(pitch) * dy,
            jnp.cos(roll) * dp + jnp.sin(roll) * jnp.cos(pitch) * dy,
            -jnp.sin(roll) * dp + jnp.cos(roll) * jnp.cos(pitch) * dy,
        ),
        -1,
    )


def state_vector(data: SimData) -> jax.Array:
    """Export position/RPY/velocity/Euler rates for existing evaluation tools."""
    states = data.states
    rpy = Rotation.from_quat(states.quat).as_euler("xyz")
    rates = ang_vel2rpy_rates(states.quat, states.ang_vel)
    return jnp.concatenate((states.pos, rpy, states.vel, rates), axis=-1)[:, 0]


def command(action: jax.Array, cfg: Config, params: PhysicalParams) -> jax.Array:
    """Normalized action to absolute RPY radians and actual total thrust N."""
    hover = params["mass"] * -params["gravity_vec"][2]
    thrust = jnp.clip(
        hover * (1 + 0.65 * action[..., 3]), 4 * params["thrust_min"], 4 * params["thrust_max"]
    )
    return jnp.concatenate(
        (action[..., :2] * cfg.max_tilt, action[..., 2:3] * cfg.max_yaw, thrust[..., None]), -1
    )


def initialize(batch: Batch, cfg: Config, physical: PhysicalParams | None = None) -> SimData:
    """Initialize the full official plant once per uninterrupted rollout.

    The flight begins with nominal hover RPM, zero controller integral and
    previous-rate memory, and the sampled airborne reset state. Randomization
    affects physical mass, force/torque curves and rotor response. The optional
    hardware configuration also varies inertia, drag and per-motor curves and
    response. The onboard controller keeps its nominal gains and allocation.
    """
    if physical is None:
        physical = nominal_params(cfg)
    state, domain = batch["state"], batch["domain"]
    n = state.shape[0]
    frequency = int(round(cfg.substeps / cfg.dt))
    if frequency < 500 or abs(frequency * cfg.dt - cfg.substeps) > 1e-6:
        raise ValueError("Use an integral plant frequency >= 500 Hz with the 500 Hz controller")
    params = FirstPrinciplesParams.create(Drone.cf21B_500, device=None)
    replacement = {}
    for name in params.__dict__:
        value = jnp.asarray(physical[name])
        if name in ("mass", "L", "prop_inertia"):
            value = jnp.atleast_1d(value)
        elif name in ("rpm2thrust", "rpm2torque", "rotor_dyn_coef"):
            value = jnp.atleast_2d(value)
        replacement[name] = value
    params = params.replace(**replacement)
    params = params.replace(
        mass=params.mass * domain["mass_scale"][:, None, None],
        rpm2thrust=params.rpm2thrust * domain["thrust_scale"][:, None, None, None],
        rpm2torque=params.rpm2torque * domain["thrust_scale"][:, None, None, None],
        rotor_dyn_coef=params.rotor_dyn_coef * domain["response_scale"][:, None, None, None],
    )
    if cfg.hardware.enabled:
        inertia = params.J * domain["inertia_scale"][:, None, None, None]
        motor_thrust = domain["motor_thrust_scale"][:, None, :, None]
        params = params.replace(
            J=inertia,
            J_inv=jnp.linalg.inv(inertia),
            drag_matrix=params.drag_matrix * domain["drag_scale"][:, None, None, None],
            rpm2thrust=params.rpm2thrust * motor_thrust,
            rpm2torque=params.rpm2torque * motor_thrust,
            rotor_dyn_coef=params.rotor_dyn_coef * domain["motor_response_scale"][:, None],
        )
    states = SimState.create(n, 1, device=None)
    hover_force = jnp.broadcast_to(physical["mass"] * -physical["gravity_vec"][2] / 4, (n, 1, 4))
    hover_rpm = motor_force2rotor_vel(hover_force, physical["rpm2thrust"])
    states = states.replace(
        pos=state[:, None, :3],
        quat=Rotation.from_euler("xyz", state[:, None, 3:6]).as_quat(),
        vel=state[:, None, 6:9],
        ang_vel=body_rates(state)[:, None],
        rotor_vel=hover_rpm,
        force=params.mass * domain["wind"][:, None],
    )
    controls = SimControls.create(n, 1, Control.attitude, Drone.cf21B_500, 100, 500, 500, 500, None)
    original_control = control_params(cfg)
    controls = controls.replace(
        attitude=controls.attitude.replace(params=original_control["attitude"]),
        force_torque=controls.force_torque.replace(params=original_control["force_torque"]),
    )
    core = SimCore.create(frequency, n, 1, [0], 0, None)
    return SimData(states=states, controls=controls, params=params, core=core)


def advance(data: SimData, cmd: jax.Array, cfg: Config) -> tuple[SimData, jax.Array]:
    """Advance all official control and physics stages, preserving their memory."""
    data = F.attitude_control(data, cmd[:, None])
    stages = build_control_fns(Control.attitude, Dynamics.default)
    integrate = select_integrate_fn(Integrator.default, select_dynamics_fn(Dynamics.default))
    physical = nominal_params(cfg)
    upper = motor_force2rotor_vel(physical["thrust_max"], physical["rpm2thrust"])

    def step(current: SimData, _: None) -> tuple[SimData, jax.Array]:
        for _, stage in stages:
            current = stage(current)
        current = clip_rotor_vel_cmd(current, lower=0.0, upper=upper, dynamics=Dynamics.default)
        current = integrate(current)
        current = clip_floor_pos(current)
        current = increment_steps(current)
        return current, state_vector(current)

    data, states = jax.lax.scan(step, data, None, length=cfg.substeps)
    data = data.replace(core=data.core.replace(mjx_synced=False))
    return data, states


def sample_batch(key: jax.Array, cfg: Config, n: int, scenario: str = "noisy") -> Batch:
    """Static indoor balloons; truth is available to plant/loss/evaluation only."""
    if scenario not in ("nominal", "noisy", "perturbed"):
        raise ValueError(f"Unknown scenario {scenario}")
    keys = jax.random.split(key, 12)

    def uniform(k: jax.Array, shape: tuple[int, ...], low: float, high: float) -> jax.Array:
        return jax.random.uniform(k, shape, minval=low, maxval=high)

    state = jnp.zeros((n, 12)).at[:, 2].set(1.4)
    state = state.at[:, 3:6].set(uniform(keys[0], (n, 3), -0.06, 0.06))
    state = state.at[:, 6:9].set(uniform(keys[1], (n, 3), -0.12, 0.12))
    state = state.at[:, 9:12].set(uniform(keys[2], (n, 3), -0.06, 0.06))
    d = uniform(keys[3], (n,), cfg.distance_min, cfg.distance_max)
    spread = 0.45 if scenario != "perturbed" else 0.8
    target = jnp.stack(
        (d, uniform(keys[4], (n,), -spread, spread), 1.4 + uniform(keys[5], (n,), -0.3, 0.3)), -1
    )
    radius = uniform(keys[6], (n,), cfg.radius_min, cfg.radius_max)
    scale = 0.0 if scenario == "nominal" else (1.5 if scenario == "perturbed" else 1.0)
    if scenario == "perturbed":
        state = state.at[:, 6:9].multiply(3)
        state = state.at[:, 3:6].multiply(2)
    domain = {
        "mass_scale": 1 + scale * uniform(keys[7], (n,), -cfg.mass_fraction, cfg.mass_fraction),
        "thrust_scale": 1
        + scale * uniform(keys[8], (n,), -cfg.thrust_fraction, cfg.thrust_fraction),
        "response_scale": 1
        + scale * uniform(keys[9], (n,), -cfg.response_fraction, cfg.response_fraction),
        "wind": scale * cfg.wind_std * jax.random.normal(keys[10], (n, 3)),
        "gyro_bias": scale * cfg.gyro_bias_std * jax.random.normal(keys[11], (n, 3)),
    }
    if cfg.hardware.enabled:
        # Separate namespace: the original twelve reset/domain draws stay unchanged.
        hardware_keys = jax.random.split(jax.random.fold_in(key, 0x48574452), 4)

        def multiplier(index: int, shape: tuple[int, ...], fraction: float) -> jax.Array:
            return 1 + scale * uniform(hardware_keys[index], shape, -fraction, fraction)

        hardware = cfg.hardware
        response = multiplier(3, (n, 4, 2), hardware.motor_response_fraction)
        domain.update(
            inertia_scale=multiplier(0, (n,), hardware.inertia_fraction),
            drag_scale=multiplier(1, (n,), hardware.drag_fraction),
            motor_thrust_scale=multiplier(2, (n, 4), hardware.motor_thrust_fraction),
            motor_response_scale=jnp.repeat(response, 2, axis=-1),
        )
    return {"state": state, "target": target, "radius": radius, "domain": domain}
