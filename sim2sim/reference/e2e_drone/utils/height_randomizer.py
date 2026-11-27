"""
Realistic pilot-like height control for quadcopter goal management.

Simulates how real pilots adjust altitude: smooth transitions between waypoints,
realistic climb/descent rates, and purposeful height changes based on flight phases.
No random jumps - just smooth, pilot-like altitude control.
"""

import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple, Optional, List
from enum import Enum

class HeightRandomizer:
    """
    Realistic pilot-like height control that simulates smooth altitude transitions.

    Design principles:
    - Smooth interpolation between waypoints (like real pilots)
    - Realistic climb/descent rates
    - Flight phase-aware behavior patterns
    - Perlin-like noise for natural variation
    - Zero special cases, pure tensor operations
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        height_range: Tuple[float, float] = (0.5, 2.5),
        dt: float = 0.01,
        climb_rate: float = 0.5,  # m/s
        waypoint_distance: float = 8.0,  # meters between waypoints
        noise_scale: float = 0.1,  # amplitude of height variation
        debug_save_dir: Optional[str] = None  # directory to save debug plots, None to disable
    ):
        """
        Initialize realistic pilot-like height controller.

        Args:
            num_envs: Number of parallel environments
            device: Torch device for tensor operations
            height_range: (min_height, max_height) in meters
            dt: Simulation timestep in seconds
            climb_rate: Maximum climb/descent rate in m/s
            waypoint_distance: Distance traveled between waypoint changes
            noise_scale: Amplitude of natural height variation
            debug_save_dir: Directory to save debug visualization plots, None to disable debug
        """
        self.num_envs = num_envs
        self.device = device
        self.height_range = height_range
        self.dt = dt
        self.climb_rate = climb_rate
        self.waypoint_distance = waypoint_distance
        self.noise_scale = noise_scale
        self.debug_save_dir = debug_save_dir

        # Create debug directory and initialize debug data only if debug is enabled
        if debug_save_dir is not None:
            os.makedirs(debug_save_dir, exist_ok=True)
            # Debug data collection for env 0
            self._episode_counter = 0
            self._env0_actual_heights: List[float] = []
            self._env0_target_heights: List[float] = []
            self._env0_timestamps: List[float] = []
            self._current_time = 0.0

        # Pilot state: current and target heights with smooth transitions
        self._current_heights = torch.full(
            (num_envs,),
            (height_range[0] + height_range[1]) / 2,
            device=device
        )
        self._target_waypoints = self._current_heights.clone()
        self._distance_traveled = torch.zeros(num_envs, device=device)
        self._last_position = torch.zeros((num_envs, 3), device=device)  # Track last 3D position for distance calculation
        self._noise_phase = torch.rand(num_envs, device=device) * 2 * math.pi

    def step(self, position: torch.Tensor) -> None:
        """
        Update pilot-like height control with smooth transitions.

        Args:
            position: Current 3D position of each env (num_envs, 3) in meters.
        """
        # Collect debug data for env 0 only if debug is enabled
        if self.debug_save_dir is not None:
            self._current_time += self.dt
            self._env0_actual_heights.append(position[0, 2].item())  # Use Z coordinate from position
            self._env0_target_heights.append(self._current_heights[0].item())
            self._env0_timestamps.append(self._current_time)

        # Update distance traveled based on actual position displacement
        distance_step = torch.norm(position - self._last_position, dim=1)
        self._distance_traveled += distance_step

        # Update last position for next iteration
        self._last_position = position.clone()

        # Check which environments need new waypoints
        waypoint_mask = self._distance_traveled >= self.waypoint_distance

        if torch.any(waypoint_mask):
            # Generate new waypoints for environments that have traveled far enough
            self._generate_new_waypoints(waypoint_mask)
            # Reset distance for environments that got new waypoints
            self._distance_traveled[waypoint_mask] = 0.0

        # Smooth interpolation towards target waypoints
        self._update_current_heights()

        # Add natural pilot variation using noise
        self._add_pilot_variation()

    def apply_to_goals(self, goal_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply pilot-like height control to goal positions.

        Args:
            goal_positions: Tensor of shape (num_envs, 3) with [x, y, z] positions

        Returns:
            Modified goal positions with smooth, realistic Z coordinates
        """
        modified_goals = goal_positions.clone()
        modified_goals[:, 2] = self._current_heights
        return modified_goals

    def reset(self, env_ids: torch.Tensor, initial_positions: Optional[torch.Tensor] = None) -> None:
        """Reset pilot controller state for specified environments.

        Args:
            env_ids: Environment IDs to reset
            initial_positions: Initial 3D positions for reset environments (num_reset, 3).
                              If None, _last_position will be set to zero.
        """
        # Save debug plot for env 0 if it's being reset and has data (only if debug is enabled)
        if self.debug_save_dir is not None and 0 in env_ids and len(self._env0_actual_heights) > 0:
            self._save_episode_plot()
            self._clear_debug_data()

        # Reset to starting heights. Prefer provided initial positions' Z component.
        num_reset = len(env_ids)
        if initial_positions is not None:
            # Ensure tensor is on correct device and clamp to allowed range.
            new_heights = initial_positions[:, 2]
            new_heights = torch.clamp(new_heights, self.height_range[0], self.height_range[1])
        else:
            new_heights = torch.empty(num_reset, device=self.device).uniform_(*self.height_range)

        self._current_heights[env_ids] = new_heights
        self._target_waypoints[env_ids] = new_heights
        self._distance_traveled[env_ids] = 0.0
        self._noise_phase[env_ids] = torch.rand(num_reset, device=self.device) * 2 * math.pi

        # Reset last position for distance calculation
        if initial_positions is not None:
            self._last_position[env_ids] = initial_positions
        else:
            self._last_position[env_ids] = 0.0

    def get_current_heights(self) -> torch.Tensor:
        """Get current smoothed heights for all environments."""
        return self._current_heights.clone()

    def _generate_new_waypoints(self, mask: torch.Tensor) -> None:
        """Generate new height waypoints for masked environments.

        Uses Beta(2,2) distribution to bias selection toward center of height range,
        matching real pilot behavior of preferring middle altitudes.
        """
        num_update = int(mask.sum().item())
        if num_update == 0:
            return

        # Use proper Beta(2,2) distribution
        with torch.no_grad():  # No gradients needed for waypoint generation
            dist = torch.distributions.Beta(2.0, 2.0)
            random_vals = dist.sample((num_update,)).to(self.device)

            # Map [0,1] to height range
            height_span = self.height_range[1] - self.height_range[0]
            new_waypoints = self.height_range[0] + random_vals * height_span

            self._target_waypoints[mask] = new_waypoints

    def _update_current_heights(self) -> None:
        """Smoothly interpolate current heights toward target waypoints."""
        # Calculate maximum height change this step based on climb rate
        max_height_change = self.climb_rate * self.dt

        # Calculate desired height change
        height_diff = self._target_waypoints - self._current_heights

        # Limit change to realistic climb/descent rate
        height_change = torch.clamp(height_diff, -max_height_change, max_height_change)

        # Apply smooth height change
        self._current_heights += height_change

    def _add_pilot_variation(self) -> None:
        """Add natural pilot variation using smooth noise."""
        # Update noise phase (like Perlin noise time evolution)
        self._noise_phase += self.dt * 0.5  # Slow variation

        # Generate smooth noise for natural height variation
        noise = torch.sin(self._noise_phase) * self.noise_scale

        # Apply variation while respecting bounds
        varied_heights = self._current_heights + noise
        self._current_heights = torch.clamp(
            varied_heights,
            self.height_range[0],
            self.height_range[1]
        )

    def _save_episode_plot(self) -> None:
        """Save debug visualization plot for completed episode."""
        if self.debug_save_dir is None or len(self._env0_actual_heights) == 0:
            return

        try:
            plt.figure(figsize=(12, 6))

            # Convert data to numpy arrays for plotting
            timestamps = np.array(self._env0_timestamps)
            actual_heights = np.array(self._env0_actual_heights)
            target_heights = np.array(self._env0_target_heights)

            # Plot actual vs target heights
            plt.plot(timestamps, actual_heights, 'b-', linewidth=2, label='Actual Height', alpha=0.8)
            plt.plot(timestamps, target_heights, 'r--', linewidth=2, label='Target Height', alpha=0.8)

            # Add height range boundaries
            plt.axhline(y=self.height_range[0], color='gray', linestyle=':', alpha=0.5, label='Height Bounds')
            plt.axhline(y=self.height_range[1], color='gray', linestyle=':', alpha=0.5)

            # Formatting
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Height (m)', fontsize=12)
            plt.title(f'Height Control Debug - Episode {self._episode_counter} (Env 0)', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)

            # Calculate error statistics
            height_error = actual_heights - target_heights
            mean_error = np.mean(np.abs(height_error))
            max_error = np.max(np.abs(height_error))

            # Add error statistics as text
            stats_text = f'Mean Error: {mean_error:.3f}m\nMax Error: {max_error:.3f}m'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()

            # Save plot
            filename = f"height_debug_episode_{self._episode_counter:06d}.png"
            filepath = os.path.join(self.debug_save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Height debug plot saved: {filepath}")

        except Exception as e:
            print(f"Failed to save height debug plot: {e}")

    def _clear_debug_data(self) -> None:
        """Clear debug data for next episode."""
        if self.debug_save_dir is None:
            return
        self._env0_actual_heights.clear()
        self._env0_target_heights.clear()
        self._env0_timestamps.clear()
        self._current_time = 0.0
        self._episode_counter += 1