import torch
from typing import Optional, Dict, Any, Tuple


class ThrustUncertaintySimulator:
    """
    Simulates battery-related thrust effectiveness degradation in drones.

    This class provides two key features:
    1. Episode-to-episode randomization of initial thrust effectiveness
       (representing different battery charge levels)
    2. Gradual thrust degradation over time within an episode
       (representing battery discharge during flight)

    The thrust effectiveness is modeled as a coefficient that scales the
    commanded thrust values, similar to how a real battery's voltage drop
    affects motor performance.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        dt: float,
        episode_length_steps: int,
        initial_effectiveness_range: Tuple[float, float] = (0.85, 1.0),
        degradation_factor_range: Tuple[float, float] = (0.05, 0.15),
        min_effectiveness: float = 0.7,
        training_preset: str = "default"
    ):
        """
        Initialize the thrust uncertainty simulator.

        Parameters
        ----------
        num_envs : int
            Number of parallel environments.
        device : torch.device
            Device to run computations on (CPU or GPU).
        dt : float
            Simulation timestep in seconds.
        episode_length_steps : int
            Maximum number of steps in an episode.
        initial_effectiveness_range : Tuple[float, float], default=(0.85, 1.0)
            Range for randomizing initial thrust effectiveness at episode start.
            (0.85, 1.0) means the drone starts with 85% to 100% thrust effectiveness.
        degradation_factor_range : Tuple[float, float], default=(0.05, 0.15)
            Range for randomizing the degradation factor per episode.
            This determines how much the thrust effectiveness decreases over the episode.
            A value of 0.1 means effectiveness will decrease by 10% over the full episode.
        min_effectiveness : float, default=0.7
            Minimum allowed thrust effectiveness (prevents physically unrealistic values).
        training_preset : str, default="default"
            Preset configuration name. Available options:
            - "none": No uncertainty (always 100% effective)
            - "mild": Slight initial randomness, minimal degradation
            - "moderate": Medium initial randomness, noticeable degradation
            - "severe": Large initial randomness, significant degradation
            - "default": Same as "moderate"
        """
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.episode_length_steps = episode_length_steps
        self.min_effectiveness = min_effectiveness

        # Apply preset if specified
        if training_preset != "default":
            initial_effectiveness_range, degradation_factor_range = self._get_preset_params(training_preset)

        self.initial_effectiveness_min = initial_effectiveness_range[0]
        self.initial_effectiveness_max = initial_effectiveness_range[1]
        self.degradation_factor_min = degradation_factor_range[0]
        self.degradation_factor_max = degradation_factor_range[1]

        # Initialize thrust effectiveness for each environment
        self.thrust_effectiveness = torch.ones(num_envs, device=device)

        # Per-environment degradation rate (different for each env)
        self.degradation_rate = torch.zeros(num_envs, device=device)

        # Step counter for each environment
        self.step_counter = torch.zeros(num_envs, device=device, dtype=torch.long)

        # Initial random reset for all environments
        self.reset()

    def _get_preset_params(self, preset_name: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get parameter ranges for a named preset configuration.

        Parameters
        ----------
        preset_name : str
            Name of the preset configuration.

        Returns
        -------
        Tuple[Tuple[float, float], Tuple[float, float]]
            A tuple containing (initial_effectiveness_range, degradation_factor_range)
        """
        presets = {
            "none": ((1.0, 1.0), (0.0, 0.0)),
            "mild": ((0.95, 1.0), (0.02, 0.05)),
            "moderate": ((0.85, 1.0), (0.05, 0.15)),
            "severe": ((0.75, 1.0), (0.15, 0.25))
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {list(presets.keys())}")

        return presets[preset_name]

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """
        Reset thrust effectiveness parameters for specified environments.

        Parameters
        ----------
        env_ids : torch.Tensor, optional
            Indices of environments to reset. If None, all environments are reset.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Randomize initial thrust effectiveness for each environment
        initial_effectiveness = torch.rand(
            len(env_ids), device=self.device
        ) * (self.initial_effectiveness_max - self.initial_effectiveness_min) + self.initial_effectiveness_min

        # Randomize degradation factors for each environment
        # This determines how much effectiveness will degrade over the full episode
        degradation_factor = torch.rand(
            len(env_ids), device=self.device
        ) * (self.degradation_factor_max - self.degradation_factor_min) + self.degradation_factor_min

        # Convert total degradation to per-step rate
        # Formula ensures we reach (initial_value * (1-degradation_factor)) by the end of episode
        step_degradation_rate = degradation_factor / self.episode_length_steps

        # Set values for specified environments
        self.thrust_effectiveness[env_ids] = initial_effectiveness
        self.degradation_rate[env_ids] = step_degradation_rate
        self.step_counter[env_ids] = 0

    def step(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Advance the thrust effectiveness simulation by one time step.

        Parameters
        ----------
        env_ids : torch.Tensor, optional
            Indices of environments to update. If None, all environments are updated.

        Returns
        -------
        torch.Tensor
            A tensor of thrust effectiveness coefficients for each environment.
            These should be multiplied with the commanded thrust to simulate battery effects.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Increment step counter
        self.step_counter[env_ids] += 1

        # Apply degradation based on linear decay model
        # Each environment can have a different degradation rate
        self.thrust_effectiveness[env_ids] = torch.clamp(
            self.thrust_effectiveness[env_ids] * (1.0 - self.degradation_rate[env_ids]),
            min=self.min_effectiveness
        )

        return self.thrust_effectiveness

    def get_effectiveness(self) -> torch.Tensor:
        """
        Get the current thrust effectiveness coefficients.

        Returns
        -------
        torch.Tensor
            A tensor of thrust effectiveness coefficients for each environment.
        """
        return self.thrust_effectiveness

    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get the current state of the simulator for saving/checkpointing.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the simulator state.
        """
        return {
            "thrust_effectiveness": self.thrust_effectiveness.clone(),
            "degradation_rate": self.degradation_rate.clone(),
            "step_counter": self.step_counter.clone()
        }

    def set_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Restore the simulator state from a dictionary.

        Parameters
        ----------
        state_dict : Dict[str, Any]
            Dictionary containing the simulator state.
        """
        self.thrust_effectiveness = state_dict["thrust_effectiveness"].clone()
        self.degradation_rate = state_dict["degradation_rate"].clone()
        self.step_counter = state_dict["step_counter"].clone()

    def set_attr(self, **kwargs) -> None:
        """
        Set simulator parameters during runtime.

        This method allows external programs to modify generator parameters
        while the simulator is running, enabling dynamic difficulty adjustment
        or parameter sweeps.

        Parameters
        ----------
        **kwargs
            Keyword arguments for parameters to modify. Available parameters:
            - initial_effectiveness_range: Tuple[float, float] - Range for initial thrust effectiveness
            - degradation_factor_range: Tuple[float, float] - Range for degradation factors
            - min_effectiveness: float - Minimum allowed thrust effectiveness
            - training_preset: str - Apply a preset configuration ("none", "mild", "moderate", "severe")

        Raises
        ------
        ValueError
            If a parameter name is invalid or a value is out of valid range.
        TypeError
            If a value type doesn't match the expected type for the parameter.

        Examples
        --------
        >>> simulator.set_attr(min_effectiveness=0.8)
        >>> simulator.set_attr(training_preset="severe")
        >>> simulator.set_attr(
        ...     initial_effectiveness_range=(0.9, 1.0),
        ...     degradation_factor_range=(0.1, 0.2)
        ... )
        """
        available_attrs = [
            "initial_effectiveness_range",
            "degradation_factor_range",
            "min_effectiveness",
            "training_preset"
        ]

        for attr_name, value in kwargs.items():
            if attr_name == "initial_effectiveness_range":
                if not isinstance(value, (tuple, list)) or len(value) != 2:
                    raise TypeError("initial_effectiveness_range must be a tuple or list of length 2")
                if not all(isinstance(x, (int, float)) for x in value):
                    raise TypeError("initial_effectiveness_range values must be numeric")
                if value[0] < 0 or value[1] > 1 or value[0] > value[1]:
                    raise ValueError("initial_effectiveness_range must be in [0, 1] with min <= max")

                self.initial_effectiveness_min = float(value[0])
                self.initial_effectiveness_max = float(value[1])

            elif attr_name == "degradation_factor_range":
                if not isinstance(value, (tuple, list)) or len(value) != 2:
                    raise TypeError("degradation_factor_range must be a tuple or list of length 2")
                if not all(isinstance(x, (int, float)) for x in value):
                    raise TypeError("degradation_factor_range values must be numeric")
                if value[0] < 0 or value[1] > 1 or value[0] > value[1]:
                    raise ValueError("degradation_factor_range must be in [0, 1] with min <= max")

                self.degradation_factor_min = float(value[0])
                self.degradation_factor_max = float(value[1])

            elif attr_name == "min_effectiveness":
                if not isinstance(value, (int, float)):
                    raise TypeError("min_effectiveness must be numeric")
                if value < 0 or value > 1:
                    raise ValueError("min_effectiveness must be in [0, 1]")

                self.min_effectiveness = float(value)
                # Clamp current effectiveness values to new minimum
                self.thrust_effectiveness = torch.clamp(
                    self.thrust_effectiveness, min=self.min_effectiveness
                )

            elif attr_name == "training_preset":
                if not isinstance(value, str):
                    raise TypeError("training_preset must be a string")

                try:
                    initial_range, degradation_range = self._get_preset_params(value)
                    self.initial_effectiveness_min = initial_range[0]
                    self.initial_effectiveness_max = initial_range[1]
                    self.degradation_factor_min = degradation_range[0]
                    self.degradation_factor_max = degradation_range[1]
                except ValueError as e:
                    raise ValueError(f"Invalid training_preset: {e}")

            else:
                raise ValueError(f"Unknown attribute '{attr_name}'. Available attributes: {available_attrs}")

    def get_attr(self, attr_name: str) -> Any:
        """
        Get current value of a simulator parameter.

        Parameters
        ----------
        attr_name : str
            Name of the parameter to retrieve.

        Returns
        -------
        Any
            Current value of the specified parameter.

        Raises
        ------
        ValueError
            If the parameter name is invalid.
        """
        if attr_name == "initial_effectiveness_range":
            return (self.initial_effectiveness_min, self.initial_effectiveness_max)
        elif attr_name == "degradation_factor_range":
            return (self.degradation_factor_min, self.degradation_factor_max)
        elif attr_name == "min_effectiveness":
            return self.min_effectiveness
        elif attr_name == "training_preset":
            # Try to determine current preset based on parameters
            for preset_name in ["none", "mild", "moderate", "severe"]:
                initial_range, degradation_range = self._get_preset_params(preset_name)
                if (abs(self.initial_effectiveness_min - initial_range[0]) < 1e-6 and
                    abs(self.initial_effectiveness_max - initial_range[1]) < 1e-6 and
                    abs(self.degradation_factor_min - degradation_range[0]) < 1e-6 and
                    abs(self.degradation_factor_max - degradation_range[1]) < 1e-6):
                    return preset_name
            return "custom"
        else:
            available_attrs = [
                "initial_effectiveness_range",
                "degradation_factor_range",
                "min_effectiveness",
                "training_preset"
            ]
            raise ValueError(f"Unknown attribute '{attr_name}'. Available attributes: {available_attrs}")

# Basic tests for the ThrustUncertaintySimulator
def _test_thrust_simulator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulator = ThrustUncertaintySimulator(
        num_envs=5,
        device=device,
        dt=0.01,
        episode_length_steps=200,
        initial_effectiveness_range=(0.9, 1.0),
        degradation_factor_range=(0.1, 0.2)
    )

    # Test initial state
    effectiveness = simulator.get_effectiveness()
    assert effectiveness.shape == (5,)
    assert torch.all(effectiveness >= 0.9)
    assert torch.all(effectiveness <= 1.0)

    # Test degradation over time
    initial_values = effectiveness.clone()

    # Simulate 50 steps
    for _ in range(50):
        simulator.step()

    # Check that values have decreased
    mid_values = simulator.get_effectiveness()
    assert torch.all(mid_values < initial_values)

    # Test reset function with specific indices
    test_indices = torch.tensor([0, 2], device=device)
    simulator.reset(test_indices)

    # Only test_indices should be reset
    new_values = simulator.get_effectiveness()
    for i in range(5):
        if i in test_indices:
            assert new_values[i] >= 0.9
            assert new_values[i] <= 1.0
        else:
            assert new_values[i] == mid_values[i]

    # Test presets
    simulator_mild = ThrustUncertaintySimulator(
        num_envs=2,
        device=device,
        dt=0.01,
        episode_length_steps=200,
        training_preset="mild"
    )

    simulator_severe = ThrustUncertaintySimulator(
        num_envs=2,
        device=device,
        dt=0.01,
        episode_length_steps=200,
        training_preset="severe"
    )

    # Run 100 steps
    for _ in range(100):
        simulator_mild.step()
        simulator_severe.step()

    # Severe should degrade more than mild
    mild_effectiveness = simulator_mild.get_effectiveness()
    severe_effectiveness = simulator_severe.get_effectiveness()

    assert torch.mean(mild_effectiveness) > torch.mean(severe_effectiveness)

    print("All tests passed!")


# Visualization test - plots thrust effectiveness over time for different presets
def _visualize_thrust_degradation():
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib is required for visualization. Install with: pip install matplotlib")
        return

    print("Visualizing thrust degradation patterns...")

    # Create device and simulators with different presets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episode_length = 300  # 300 steps
    num_envs_per_preset = 5

    # Initialize simulators with different presets
    simulators = {
        "none": ThrustUncertaintySimulator(
            num_envs=num_envs_per_preset, device=device, dt=0.01,
            episode_length_steps=episode_length, training_preset="none"
        ),
        "mild": ThrustUncertaintySimulator(
            num_envs=num_envs_per_preset, device=device, dt=0.01,
            episode_length_steps=episode_length, training_preset="mild"
        ),
        "moderate": ThrustUncertaintySimulator(
            num_envs=num_envs_per_preset, device=device, dt=0.01,
            episode_length_steps=episode_length, training_preset="moderate"
        ),
        "severe": ThrustUncertaintySimulator(
            num_envs=num_envs_per_preset, device=device, dt=0.01,
            episode_length_steps=episode_length, training_preset="severe"
        )
    }

    # Store effectiveness values over time
    effectiveness_history = {preset: [] for preset in simulators.keys()}

    # Run simulation for each preset
    for step in range(episode_length):
        for preset, simulator in simulators.items():
            simulator.step()
            effectiveness = simulator.get_effectiveness().cpu().numpy()
            effectiveness_history[preset].append(effectiveness)

    # Convert to numpy arrays for easier plotting
    for preset in effectiveness_history:
        effectiveness_history[preset] = np.array(effectiveness_history[preset])

    # Calculate mean and std for each preset at each timestep
    mean_effectiveness = {preset: np.mean(effectiveness_history[preset], axis=1) for preset in simulators.keys()}
    std_effectiveness = {preset: np.std(effectiveness_history[preset], axis=1) for preset in simulators.keys()}

    # Create figure
    plt.figure(figsize=(12, 8))

    # Time steps for x-axis
    time_steps = np.arange(episode_length)

    # Plot mean with shaded std area for each preset
    colors = {"none": "blue", "mild": "green", "moderate": "orange", "severe": "red"}
    for preset in simulators.keys():
        mean = mean_effectiveness[preset]
        std = std_effectiveness[preset]

        plt.plot(time_steps, mean, label=f"{preset.capitalize()}", color=colors[preset], linewidth=2)
        plt.fill_between(
            time_steps,
            mean - std,
            mean + std,
            alpha=0.2,
            color=colors[preset]
        )

    # Set plot labels and title
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Thrust Effectiveness (0.0-1.0)', fontsize=12)
    plt.title('Thrust Effectiveness Degradation Patterns', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Show percentage y-axis
    plt.yticks(np.arange(0.6, 1.05, 0.05), [f"{int(x*100)}%" for x in np.arange(0.6, 1.05, 0.05)])

    # Set y-axis limits
    plt.ylim(0.65, 1.02)

    # Save plot
    plt.savefig('thrust_degradation_patterns.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'thrust_degradation_patterns.png'")

    # Display if interactive environment
    plt.show()


if __name__ == "__main__":
    _test_thrust_simulator()
    # Comment out the following line if matplotlib is not available
    _visualize_thrust_degradation()