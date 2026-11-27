import numpy as np
import time
import os
import textwrap
import torch
import random
from collections import deque

def save_points_to_pcd(points, filename):
    """Save point cloud to PCD file format."""
    if os.path.exists(filename):
        os.remove(filename)
    num_points = len(points)
    header = textwrap.dedent(f"""\
        # .PCD v0.7 - Point Cloud Data file format
        VERSION 0.7
        FIELDS x y z
        SIZE 4 4 4
        TYPE F F F
        COUNT 1 1 1
        WIDTH {num_points}
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS {num_points}
        DATA ascii
        """)
    with open(filename, 'w') as file:
        file.write(header)
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

class DeathReplayCollector:
    """
    Collects drone death replay data including trajectories, ToF sensors, and outcomes.
    Records data only for completed episodes (success or failure) for debugging purposes.
    """

    def __init__(self,
                 num_envs: int,
                 tof_width: int = 8,
                 tof_height: int = 8,
                 history_capacity: int = 1000,
                 data_dir: str = "/workspace/isaaclab/logs/death_replay",
                 visualization_num: int = 10,
                 stats_window: int = 100,
                 device: str = "cuda",
                 enable: bool = True,
                 dt: float = 0.01):
        """
        Initialize the death replay data collector.

        Args:
            num_envs: Number of environments to track
            tof_width: Width of ToF depth image
            tof_height: Height of ToF depth image
            history_capacity: Maximum number of frames to store per environment
            data_dir: Directory to save data
            visualization_num: Number of environments to randomly select for recording
            stats_window: Number of recent episodes to track for success statistics
            device: Device for tensor computations
            enable: Whether data collection is enabled
            dt: Simulation timestep for virtual time calculation
        """
        self.num_envs = num_envs
        self.tof_width = tof_width
        self.tof_height = tof_height
        self.history_capacity = history_capacity
        self.data_dir = data_dir + "/" + str(int(time.time()))
        self.visualization_num = min(visualization_num, num_envs)
        self.device = device
        self.enable = enable
        self.dt = dt
        self.map_id = -1
        self.map_dir = None

        # Statistics tracking
        self.stats_window = stats_window
        self.recent_episode_outcomes = deque(maxlen=stats_window)  # 1 for success, 0 for failure
        self.total_episodes = 0
        self.total_successes = 0

        # Arrays to track environment data
        self.env_dirs = np.array([None] * num_envs)
        self.episode_count = np.zeros(num_envs, dtype=np.int32)
        self.episode_data = [[] for _ in range(num_envs)]
        self.virtual_time = np.zeros(num_envs, dtype=np.float32)

        # Select environments to track (randomly)
        self.selected_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._select_environments()

        # State tracking
        self.is_recording = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Store target positions and map for visualization
        self.target_positions = None
        self.global_map = None

        if self.enable:
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"Death replay collector initialized with {self.visualization_num} selected environments.")
            print(f"Data will be saved to {self.data_dir}")
            print(f"Tracking success statistics over the last {stats_window} episodes")

    def _select_environments(self):
        """Randomly select environments to track."""
        self.selected_envs.fill_(False)
        selected_indices = random.sample(range(self.num_envs), self.visualization_num)
        self.selected_envs[selected_indices] = True

    def new_environment(self, map_3d):
        """
        Initialize a new environment for death replay collection.

        Args:
            map_3d: Point cloud of the environment (N, 3)
        """
        if not self.enable:
            return

        # Store previous environment data
        if self.map_id >= 0:
            valid_indices = np.where([len(data) > 0 for data in self.episode_data])[0]
            for env_idx in valid_indices:
                self._save_episode_data(env_idx)

        # Prepare for new environment
        self.map_id += 1
        self.map_dir = os.path.join(self.data_dir, f"map_{self.map_id}")
        os.makedirs(self.map_dir, exist_ok=True)

        # Prepare environment directory paths but don't create them yet
        # They will be created on-demand when actually saving data
        self.env_dirs = np.array([os.path.join(self.map_dir, f"env_{i}") for i in range(self.num_envs)])

        # Reset counters and data arrays
        self.episode_count = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_data = [[] for _ in range(self.num_envs)]
        self.virtual_time = np.zeros(self.num_envs, dtype=np.float32)

        # Save the 3D map
        map_file = os.path.join(self.map_dir, "map.pcd")
        save_points_to_pcd(map_3d, map_file)

        # Store map for internal use
        if isinstance(map_3d, torch.Tensor):
            self.global_map = map_3d.clone().cpu()
        else:
            self.global_map = torch.tensor(map_3d, dtype=torch.float32)

    def reset_episode(self, env_ids):
        """
        Reset history for the given environment IDs and start recording.

        Args:
            env_ids: Tensor of environment IDs to reset
        """
        if not self.enable:
            return

        # Print success statistics if we have data
        if len(self.recent_episode_outcomes) > 0:
            recent_success_count = sum(self.recent_episode_outcomes)
            recent_success_rate = recent_success_count / len(self.recent_episode_outcomes) * 100
            overall_success_rate = (self.total_successes / self.total_episodes) * 100 if self.total_episodes > 0 else 0

            print(f"Success statistics: {recent_success_rate:.2f}% over last {len(self.recent_episode_outcomes)} episodes "
                  f"({recent_success_count}/{len(self.recent_episode_outcomes)}), "
                  f"{overall_success_rate:.2f}% overall ({self.total_successes}/{self.total_episodes} episodes)")

        # Only process selected environments that are being reset
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env_ids] = True
        selected_reset_mask = torch.logical_and(mask, self.selected_envs)
        selected_reset_indices = torch.nonzero(selected_reset_mask).flatten().cpu().tolist()

        for env_id in selected_reset_indices:
            # Clear history for this environment
            self.episode_data[env_id] = []
            # Reset virtual time for this environment
            self.virtual_time[env_id] = 0.0

        # Set recording flag for all reset environments
        self.is_recording[env_ids] = True

    def record_frame(self, pos_w, quat_w, lin_vel_b, ang_vel_b, tof_data, left_depth=None, right_depth=None, back_depth=None, top_depth=None, down_depth=None):
        """
        Record a frame of data only for selected active environments.

        Args:
            pos_w: Position tensor of shape (num_envs, 3) in world frame
            quat_w: Quaternion tensor of shape (num_envs, 4) as [w, x, y, z] in world frame
            lin_vel_b: Linear velocity tensor of shape (num_envs, 3) in body frame
            ang_vel_b: Angular velocity tensor of shape (num_envs, 3) in body frame
            tof_data: ToF depth data tensor of shape (num_envs, H, W)
            left_depth: Left ToF single point measurement (num_envs,), optional
            right_depth: Right ToF single point measurement (num_envs,), optional
            back_depth: Back ToF single point measurement (num_envs,), optional
            top_depth: Top ToF single point measurement (num_envs,), optional
            down_depth: Down ToF single point measurement (num_envs,), optional
        """
        if not self.enable:
            return

        # Default values for optional ToF data
        if left_depth is None:
            left_depth = torch.zeros(pos_w.shape[0], device=self.device)
        if right_depth is None:
            right_depth = torch.zeros(pos_w.shape[0], device=self.device)
        if back_depth is None:
            back_depth = torch.zeros(pos_w.shape[0], device=self.device)
        if top_depth is None:
            top_depth = torch.zeros(pos_w.shape[0], device=self.device)
        if down_depth is None:
            down_depth = torch.zeros(pos_w.shape[0], device=self.device)

        # Combine selection mask with recording mask
        active_selection_mask = torch.logical_and(self.selected_envs, self.is_recording)
        active_env_indices = torch.nonzero(active_selection_mask).flatten().cpu().tolist()

        # Update virtual time for all selected and active environments
        for env_id in active_env_indices:
            self.virtual_time[env_id] += self.dt

        # Only process selected and active environments
        for env_id in active_env_indices:
            # Check for NaN values to avoid storing corrupt data
            if (torch.isnan(pos_w[env_id]).any() or
                torch.isnan(quat_w[env_id]).any() or
                torch.isnan(lin_vel_b[env_id]).any() or
                torch.isnan(ang_vel_b[env_id]).any() or
                torch.isnan(tof_data[env_id]).any()):
                print(f"Warning: NaN values detected for env {env_id}, skipping this frame")
                continue

            # Create frame data tuple with efficient copy operations
            frame_data = (
                self.virtual_time[env_id],                   # Virtual timestamp
                pos_w[env_id].clone().cpu().numpy(),        # Position in world frame
                quat_w[env_id].clone().cpu().numpy(),       # Quaternion in world frame
                lin_vel_b[env_id].clone().cpu().numpy(),    # Linear velocity in body frame
                ang_vel_b[env_id].clone().cpu().numpy(),    # Angular velocity in body frame
                tof_data[env_id].clone().cpu().numpy(),     # ToF depth image
                left_depth[env_id].clone().cpu().numpy(),   # Left ToF single point
                right_depth[env_id].clone().cpu().numpy(),  # Right ToF single point
                back_depth[env_id].clone().cpu().numpy(),   # Back ToF single point
                top_depth[env_id].clone().cpu().numpy(),    # Top ToF single point
                down_depth[env_id].clone().cpu().numpy()    # Down ToF single point
            )

            self.episode_data[env_id].append(frame_data)

    def end_episodes(self, completed_mask, success_mask):
        """
        End recording for completed episodes and save their data.

        Args:
            completed_mask: Boolean tensor indicating which environments completed an episode
            success_mask: Boolean tensor indicating which environments completed successfully
        """
        if not self.enable:
            return

        # Count all completed episodes for statistics
        all_completed_indices = torch.nonzero(completed_mask).flatten().cpu().tolist()

        # Update global statistics
        for env_id in all_completed_indices:
            self.total_episodes += 1
            if success_mask[env_id]:
                self.total_successes += 1
                self.recent_episode_outcomes.append(1)
            else:
                self.recent_episode_outcomes.append(0)

        # Identify completed environments that we're tracking
        selected_completed_mask = torch.logical_and(completed_mask, self.selected_envs)
        selected_completed_indices = torch.nonzero(selected_completed_mask).flatten().cpu().tolist()

        # Save data for completed tracked environments
        for env_id in selected_completed_indices:
            self.is_recording[env_id] = False
            if len(self.episode_data[env_id]) > 0:
                self._save_episode_data(env_id, success_mask[env_id].item())

    def _save_episode_data(self, env_idx, is_success=None):
        """
        Save episode data for a specific environment.

        Args:
            env_idx: Environment index
            is_success: Whether the episode was successful (None if unknown)
        """
        if len(self.episode_data[env_idx]) == 0:
            return

        # Create environment directory only when actually saving data
        env_dir = self.env_dirs[env_idx]
        os.makedirs(env_dir, exist_ok=True)

        # Determine success status for filename
        success_suffix = ""
        if is_success is not None:
            success_suffix = "_success" if is_success else "_failure"

        episode_file = os.path.join(
            env_dir,
            f"episode_{self.episode_count[env_idx]}{success_suffix}.npz"
        )

        # Extract data components efficiently
        episode_frames = self.episode_data[env_idx]

        # Use list comprehensions for better performance
        timestamps = np.array([frame[0] for frame in episode_frames])
        positions = np.array([frame[1] for frame in episode_frames])
        quaternions = np.array([frame[2] for frame in episode_frames])
        lin_velocities = np.array([frame[3] for frame in episode_frames])
        ang_velocities = np.array([frame[4] for frame in episode_frames])
        tof_images = np.array([frame[5] for frame in episode_frames])
        left_depths = np.array([frame[6] for frame in episode_frames])
        right_depths = np.array([frame[7] for frame in episode_frames])
        back_depths = np.array([frame[8] for frame in episode_frames])
        top_depths = np.array([frame[9] for frame in episode_frames])
        down_depths = np.array([frame[10] for frame in episode_frames])

        # Save additional metadata
        metadata = {
            'episode_id': self.episode_count[env_idx],
            'env_id': env_idx,
            'map_id': self.map_id,
            'is_success': is_success if is_success is not None else False,
            'num_frames': len(episode_frames),
            'target_position': self.target_positions[env_idx].numpy() if self.target_positions is not None and env_idx < len(self.target_positions) else None
        }

        # Use npz format for storing multiple arrays in one file
        np.savez_compressed(
            episode_file,
            timestamps=timestamps,
            positions=positions,
            quaternions=quaternions,
            lin_velocities=lin_velocities,
            ang_velocities=ang_velocities,
            tof_images=tof_images,
            left_depths=left_depths,
            right_depths=right_depths,
            back_depths=back_depths,
            top_depths=top_depths,
            down_depths=down_depths,
            metadata=metadata
        )

        print(f"Saved death replay data: {episode_file}")

        # Update episode count
        self.episode_count[env_idx] += 1

        # Clear episode data
        self.episode_data[env_idx] = []

    def set_target_positions(self, positions):
        """
        Set target positions for episodes.

        Args:
            positions: Tensor of shape (num_envs, 3) containing target x,y,z positions
        """
        if positions is None:
            self.target_positions = None
            return

        if isinstance(positions, torch.Tensor):
            positions = positions.clone().cpu()
        self.target_positions = positions

    def get_success_rate(self):
        """
        Get current success rate statistics.

        Returns:
            Dict with recent and overall success rates
        """
        if len(self.recent_episode_outcomes) == 0:
            return {"recent_rate": 0.0, "overall_rate": 0.0, "recent_count": 0, "total_count": 0}

        recent_success_count = sum(self.recent_episode_outcomes)
        recent_success_rate = recent_success_count / len(self.recent_episode_outcomes) * 100
        overall_success_rate = (self.total_successes / self.total_episodes) * 100 if self.total_episodes > 0 else 0

        return {
            "recent_rate": recent_success_rate,
            "overall_rate": overall_success_rate,
            "recent_count": f"{recent_success_count}/{len(self.recent_episode_outcomes)}",
            "total_count": f"{self.total_successes}/{self.total_episodes}"
        }