import torch
from typing import Optional, Tuple


class WindGustGenerator:
    """
    Generates realistic wind disturbances using an Ornstein-Uhlenbeck process.
    
    The generator models continuously changing wind forces that affect the drone,
    creating a more realistic simulation of flight conditions.
    
    Wind presets:
    # Indoor weak wind:     tau = 0.5,  sigma = 0.05
    # Outdoor mild wind:    tau = 1.0,  sigma = 0.2
    # Outdoor strong wind:  tau = 1.5,  sigma = 0.5
    # Gusty wind:           tau = 0.3,  sigma = 0.8
    """

    def __init__(
        self, 
        num_envs: int, 
        device: torch.device, 
        dt: float, 
        tau: float = 0.5, 
        sigma: float = 0.05
    ):
        """
        Initialize the wind gust generator.
        
        Parameters
        ----------
        num_envs : int
            Number of parallel environments.
        device : torch.device
            Device to run computations on (CPU or GPU).
        dt : float
            Simulation timestep in seconds.
        tau : float, default=0.5
            Correlation time in seconds. Controls how quickly the wind changes.
        sigma : float, default=0.05
            Stationary standard deviation of wind acceleration in m/s².
            Controls the intensity of the wind.
        """
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.tau = tau
        self.sigma = sigma
        
        # Calculate discretization parameters for the OU process
        self.alpha = torch.exp(torch.tensor(-dt / tau, device=device))
        self.var = sigma * torch.sqrt(torch.tensor(1.0, device=device) - self.alpha**2)
        
        # Initialize wind acceleration state
        self.wind_acc = torch.zeros((num_envs, 3), device=device)
        
        # For directional bias (optional)
        self.directional_bias = torch.zeros(3, device=device)

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """
        Reset wind state for specified environments.
        
        Parameters
        ----------
        env_ids : torch.Tensor, optional
            Indices of environments to reset. If None, all environments are reset.
        """
        if env_ids is None:
            self.wind_acc.zero_()
        else:
            self.wind_acc[env_ids] = 0.0

    def set_directional_bias(self, bias: torch.Tensor) -> None:
        """
        Set a directional bias to the wind (e.g., prevailing wind direction).
        
        Parameters
        ----------
        bias : torch.Tensor
            A (3,) tensor representing the bias in [x, y, z] directions.
        """
        if bias.shape != (3,):
            raise ValueError("Bias must be a 3-element tensor for [x, y, z] directions")
        self.directional_bias = bias.to(self.device)

    def set_attr(self, **kwargs) -> None:
        """
        Set wind generator attributes during runtime.
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for parameters to update. Supported parameters:
            - tau : float
                Correlation time in seconds. Must be positive.
            - sigma : float  
                Stationary standard deviation of wind acceleration in m/s². Must be non-negative.
        
        Raises
        ------
        ValueError
            If invalid parameter values are provided.
        """
        # Validate and update tau
        if 'tau' in kwargs:
            tau = kwargs['tau']
            if tau <= 0:
                raise ValueError("tau must be positive")
            self.tau = tau
        
        # Validate and update sigma
        if 'sigma' in kwargs:
            sigma = kwargs['sigma']
            if sigma < 0:
                raise ValueError("sigma must be non-negative")
            self.sigma = sigma
        
        # Recalculate dependent parameters if tau or sigma changed
        if 'tau' in kwargs or 'sigma' in kwargs:
            self.alpha = torch.exp(torch.tensor(-self.dt / self.tau, device=self.device))
            self.var = self.sigma * torch.sqrt(torch.tensor(1.0, device=self.device) - self.alpha**2)

    def step(self) -> torch.Tensor:
        """
        Advance the Ornstein-Uhlenbeck process by one time step.
        
        Returns
        -------
        torch.Tensor
            A (num_envs, 3) tensor of wind accelerations [ax, ay, az] in m/s².
        """
        # Generate random noise
        noise = torch.randn_like(self.wind_acc) * self.var
        
        # Update the OU process
        self.wind_acc = self.wind_acc * self.alpha + noise + self.directional_bias * self.dt
        
        return self.wind_acc
    
    def get_state(self) -> torch.Tensor:
        """
        Get the current wind acceleration state.
        
        Returns
        -------
        torch.Tensor
            A (num_envs, 3) tensor of wind accelerations [ax, ay, az] in m/s².
        """
        return self.wind_acc


# Basic tests for the WindGustGenerator
def _test_wind_generator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wind_gen = WindGustGenerator(num_envs=10, device=device, dt=0.01)
    
    # Test initial state
    assert torch.all(wind_gen.wind_acc == 0.0)
    
    # Test step function
    wind_acc = wind_gen.step()
    assert wind_acc.shape == (10, 3)
    assert not torch.all(wind_acc == 0.0)  # Should have changed
    
    # Test reset function with specific indices
    test_indices = torch.tensor([0, 2, 4], device=device)
    wind_gen.reset(test_indices)
    assert torch.all(wind_gen.wind_acc[test_indices] == 0.0)
    
    # Test set_attr function
    initial_tau = wind_gen.tau
    initial_sigma = wind_gen.sigma
    wind_gen.set_attr(tau=1.0, sigma=0.3)
    assert wind_gen.tau == 1.0
    assert wind_gen.sigma == 0.3
    assert wind_gen.tau != initial_tau or wind_gen.sigma != initial_sigma
    
    # Test invalid parameter validation
    try:
        wind_gen.set_attr(tau=-1.0)
        assert False, "Should have raised ValueError for negative tau"
    except ValueError:
        pass
    
    try:
        wind_gen.set_attr(sigma=-0.1)
        assert False, "Should have raised ValueError for negative sigma"
    except ValueError:
        pass
    
    print("All tests passed!")


if __name__ == "__main__":
    _test_wind_generator()
