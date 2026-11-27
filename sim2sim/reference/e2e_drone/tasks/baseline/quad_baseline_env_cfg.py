# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the quadcopter baseline environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationCfg, RenderCfg
from isaaclab.utils import configclass
from isaaclab_assets import CRAZYFLIE_CFG
from isaaclab.assets import ArticulationCfg

from e2e_drone.utils.reloadable_raycaster_camera import ReloadableRayCasterCameraCfg
from e2e_drone.utils.asset_paths import get_crazyflie_usd_path
from isaaclab.sensors.ray_caster import patterns

def create_camera_cfg(pos, rot, width=1, height=1, max_distance=4.0, focal_length=20.0, aperture=20.955):
    """Create a camera configuration for raycast sensing."""
    return ReloadableRayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        offset=ReloadableRayCasterCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
        mesh_prim_paths=["/map_mesh"],
        max_distance=max_distance,
        depth_clipping_behavior="max",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=focal_length, width=width, height=height,
            horizontal_aperture=aperture, vertical_aperture=aperture,
        ),
        data_types=["distance_to_image_plane"],
        update_period=0.0, debug_vis=False,
    )

@configclass
class QuadcopterSceneCfg(InteractiveSceneCfg):
    """Configuration for the Quadcopter scene."""
    num_envs: int = None
    env_spacing: float = 16.0
    replicate_physics: bool = True
    filter_collisions: bool = True

@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg(asset_name="robot")

    # ========================================================================
    # Basic Environment Configuration
    # ========================================================================

    # Log comments
    log_comments: str = "new seed"

    # Initial curriculum stage
    curriculum_stage = "static_goals"  # Options: "yaw_alignment", "static_goals"

    # Spawn mode configuration
    spawn_mode = "free_points"  # Options: "edges", "free_points"

    # Task queue configuration
    num_goals = 1  # Number of goals in the task queue for each robot

    # ========================================================================
    # Observation Space Configuration
    # ========================================================================

    # History configuration
    history_length = 10
    depth_cam_history = 1  # Number of previous depth frames to include in observation

    # Observation space dimensions
    frame_observation_space = 3 + 9 + 3 + 3 + 1 + 1 + 4 + 4  # velocity + rot + gyro + goal_dir + z_err + speed + actions + tof_depths
    frame_observation_space_critic = frame_observation_space + 9

    # ========================================================================
    # Data Recording Configuration
    # ========================================================================

    # DeathReplay configuration
    enable_death_replay = False
    death_replay_history_capacity = 5000
    death_replay_visualization_num = 10
    death_replay_data_dir = "/workspace/isaaclab/logs/death_replay"

    # ========================================================================
    # Sensor Configuration
    # ========================================================================

    # Enable/disable sensors
    enable_omni_tof = True
    enable_debug_camera = False

    # Camera sensor parameters
    camera_max_distance = 4.0  # Maximum sensing distance in meters
    camera_focal_length = 20.0  # Camera focal length
    camera_aperture = 20.955  # Camera aperture size
    debug_camera_offset = (0.05, 0.0, 0.0)  # Debug camera position offset

    # TOF sensor FOV
    MP_TOF_FOV = 45.0  # Field of view for the multi-point TOF sensors
    SP_TOF_FOV = 10.0  # Field of view for the single-point TOF sensors

    # Sensor noise parameters
    depth_invalid_rate_max = 0.15  # Maximum rate of invalid depth readings

    # ========================================================================
    # Environment Timing Configuration
    # ========================================================================

    # Environment timing
    episode_length_s = 80 * num_goals  # seconds
    step_freq = 15  # Hz
    decimation = 10
    action_space = 4  # [roll, pitch, yaw_rate, thrust]
    state_space = 0
    debug_vis = True

    # Action delay configuration
    action_delay_steps = 4  # action_delay_steps = delay time / self.cfg.sim.dt

    # ========================================================================
    # Simulation Configuration
    # ========================================================================

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / (step_freq * decimation),
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            gpu_collision_stack_size=2**28,  # Increase from default 2**26 to 2**28 (256MB)
        ),
        render=RenderCfg(
            enable_dl_denoiser=False,
            dlss_mode=2,
        )
    )

    # ========================================================================
    # Controller Configuration
    # ========================================================================

    # Controller parameters
    controller_Kang = [20.0, 20.0]  # Roll and pitch angle controller gains
    controller_Kdang = [0.5, 0.5]
    controller_Kang_vel = [20.0, 20.0, 15.0]  # Roll, pitch, and yaw angular velocity controller gains

    # ========================================================================
    # Robot Physical Parameters
    # ========================================================================

    # Robot physical parameters
    robot_mass = 0.049  # kg
    robot_arm_length = 0.046  # m
    robot_inertia = [2.16e-5, 2.16e-5, 4.33e-5]  # kg*m^2 [Ixx, Iyy, Izz]

    # Mass randomization configuration
    enable_mass_randomization = True
    mass_randomization_percent = 0.2  # +/- percentage of mass randomization

    # Range of desired velocities in m/s
    des_vel_range = [0.5, 1.5]  # [min, max]

    # ========================================================================
    # Environmental Effects Configuration
    # ========================================================================

    # Wind generation parameters
    wind_tau = 1.0  # Wind correlation time constant
    wind_sigma = 0.5  # Wind disturbance intensity

    # Thrust uncertainty configuration
    enable_thrust_uncertainty = True
    thrust_uncertainty_initial_range = (0.85, 1.0)  # Initial effectiveness range
    thrust_uncertainty_degradation_range = (0.10, 0.15)  # Degradation factor range
    thrust_uncertainty_min_effectiveness = 0.7  # Minimum thrust effectiveness

    # Height randomization configuration
    enable_height_randomization = False
    height_randomization_range = (0.5, 1.5)  # Height range in meters (min, max)
    height_randomization_climb_rate = 0.2  # Maximum climb/descent rate in m/s
    height_randomization_waypoint_distance = 8.0  # Distance between waypoint changes in meters
    height_randomization_noise_scale = 0.1  # Amplitude of natural pilot variation
    height_randomization_debug_dir = None

    # ========================================================================
    # Map Generation Configuration
    # ========================================================================

    # Map generation parameters
    map_generation_step_threshold = 3000 * num_goals
    obstacle_min_distance_init = 1.5
    obstacle_min_distance_min = 0.4
    obstacle_query_workers = -1  # SciPy workers per training rank; -1 uses all visible CPU cores.
    map_spacing_factor = 0.8  # Factor for map spacing relative to env_spacing
    obstacle_size_range = (0.4, 0.8)  # Size range for obstacles
    floater_size_range = (0.4, 0.8)  # Size range for floating obstacles
    goal_sampling_noise_range = 0.15  # Noise range for goal position sampling

    # ========================================================================
    # Scene Configuration
    # ========================================================================

    # Scene
    scene: InteractiveSceneCfg = QuadcopterSceneCfg()

    # Robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=CRAZYFLIE_CFG.spawn.replace(usd_path=get_crazyflie_usd_path()),
    )

    # Contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body", history_length=decimation, update_period=0.0,
        track_air_time=False,
        debug_vis=False,
    )

    # Debug camera (only created if enabled)
    if enable_debug_camera:
        tiled_camera: TiledCameraCfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/body/Camera",
            offset=TiledCameraCfg.OffsetCfg(pos=debug_camera_offset, rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            data_types=["depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=camera_focal_length, focus_distance=400.0, horizontal_aperture=camera_aperture,
                clipping_range=(0.01, camera_max_distance)
            ),
            width=8,
            height=8,
            update_period=0.0,
            depth_clipping_behavior="max",
        )

    # Depth camera - using ReloadableRayCasterCamera only
    raycast_camera: ReloadableRayCasterCameraCfg = create_camera_cfg(
        pos=(0.02, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), width=8, height=8,
        max_distance=camera_max_distance, focal_length=camera_focal_length, aperture=camera_aperture)

    # Omnidirectional TOF cameras (only created if enabled)
    if enable_omni_tof:
        left_tof_camera: ReloadableRayCasterCameraCfg = create_camera_cfg(
            pos=(0.0, 0.02, 0.0), rot=(0.7071, 0.0, 0.0, 0.7071),
            max_distance=camera_max_distance, focal_length=camera_focal_length, aperture=camera_aperture)

        right_tof_camera: ReloadableRayCasterCameraCfg = create_camera_cfg(
            pos=(0.0, -0.02, 0.0), rot=(0.7071, 0.0, 0.0, -0.7071),
            max_distance=camera_max_distance, focal_length=camera_focal_length, aperture=camera_aperture)

        back_tof_camera: ReloadableRayCasterCameraCfg = create_camera_cfg(
            pos=(-0.02, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0),
            max_distance=camera_max_distance, focal_length=camera_focal_length, aperture=camera_aperture)

        down_tof_camera: ReloadableRayCasterCameraCfg = create_camera_cfg(
            pos=(0.0, 0.0, -0.02), rot=(0.7071, 0.0, 0.7071, 0.0),
            max_distance=camera_max_distance, focal_length=camera_focal_length, aperture=camera_aperture)

    # Calculate total observation space with history
    depth_cam_dims = raycast_camera.pattern_cfg.height * raycast_camera.pattern_cfg.width
    observation_space = history_length * frame_observation_space + depth_cam_history * depth_cam_dims

    # ========================================================================
    # Curriculum Configuration
    # ========================================================================

    if curriculum_stage == "yaw_alignment":
        c_distance_reward = 0.0
        c_z_position_penalty = 0.0
        scene.num_envs = 45000
        spawn_mode = "edges"  # Spawn from edges for yaw alignment stage
    elif curriculum_stage == "static_goals":
        c_distance_reward = 8.0
        c_z_position_penalty = 0.25
        scene.num_envs = 26000

    # Hover stabilization curriculum
    enable_difficulty_curriculum: bool = True
    hover_hold_initial_s = 0.0
    hover_hold_max_s = 2.0
    hover_hold_increment_s = 0.1
    hover_hold_distance_threshold = 0.5
    hover_hold_speed_threshold = 0.5
    hover_yaw_penalty_distance = 0.2

    # ========================================================================
    # Reward Configuration
    # ========================================================================

    # Reward coefficients
    reward_coef_distance_reward: float = c_distance_reward
    reward_coef_yaw_direction_penalty: float = 0.3
    reward_coef_action_magnitude_penalty: float = 0.3
    reward_coef_action_change_penalty: float = 0.5
    reward_coef_vel_direction_penalty: float = 0.15
    reward_coef_vel_speed_excess_penalty: float = 0.5
    reward_coef_vel_speed_match_reward: float = 0.0
    reward_coef_z_position_penalty: float = c_z_position_penalty
    reward_coef_obstacle_collision_penalty: float = 80.0
    reward_coef_esdf_reward: float = 0.1
    reward_coef_succeed_reward: float = 80.0
    reward_coef_max_ang_vel_penalty: float = 0.0
    reward_coef_max_angle_penalty: float = 0.0
    reward_coef_alive_reward: float = 0.0
    reward_coef_z_vel_penalty: float = 0.0

    # Position control rewards
    reward_coef_lin_vel_reward_scale: float = 0
    reward_coef_ang_vel_reward_scale: float = 0
    reward_coef_distance_to_goal_reward_scale: float = 0

    # ========================================================================
    # Physical Limits and Safety Configuration
    # ========================================================================

    # Angular limits
    max_angular_velocity_check = 3.14 * 2.0 * 20.0  # rad/s for stability check
    max_angular_velocity_penalty = 3.14 / 4.0  # rad/s for penalty
    max_angle_penalty = 3.14 / 4.0  # rad for angle penalty

    # Reward calculation parameters
    distance_goal_mapping_scale = 0.8  # Scale factor for distance-to-goal mapping
    speed_adjustment_distance = 1.0  # Distance to start speed adjustment
    contact_force_threshold = 0.01  # Minimum contact force for collision detection
    z_position_huber_delta = 0.3  # Transition point between quadratic and linear z penalty
    vel_direction_huber_delta = 0.2  # Transition point for velocity direction Huber penalty

    # ========================================================================
    # Metrics and Logging Configuration
    # ========================================================================

    # Success rate tracking
    success_rate_window_size = 100  # Number of episodes for success rate calculation

    # Light configuration
    light_intensity = 2000.0  # Dome light intensity
    light_color = (0.75, 0.75, 0.75)  # Dome light color
