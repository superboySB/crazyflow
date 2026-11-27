"""Inference-only NumPy-exported JAX actor for the frozen Isaac task (no JAX)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


class NpzPolicy:
    """Load dense ELU actor weights once and execute on Isaac's Torch device."""

    def __init__(self, path: str | Path, device: str) -> None:
        """Validate and load actor weights plus observation normalization."""
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"].item()))
            if self.metadata.get("obs_dim") != 344 or self.metadata.get("activation") != "elu":
                raise ValueError("Expected CrazyRL actor: 344 observations and ELU activation")
            self.mean = torch.as_tensor(data["obs_mean"].copy(), dtype=torch.float32, device=device)
            var = torch.as_tensor(data["obs_var"].copy(), dtype=torch.float32, device=device)
            if self.mean.shape != (344,) or var.shape != (344,):
                raise ValueError("Observation normalization must have shape (344,)")
            self.epsilon = float(self.metadata.get("normalization_epsilon", 1e-8))
            normalization = self.metadata.get("normalization", "sqrt_var_epsilon_inside")
            if normalization == "sqrt_var_plus_epsilon":
                self.denominator = torch.sqrt(var) + self.epsilon
            elif normalization == "sqrt_var_epsilon_inside":
                self.denominator = torch.sqrt(var + self.epsilon)
            else:
                raise ValueError(f"Unknown observation normalization: {normalization}")
            dims = [344, *self.metadata["hidden_dims"], 4]
            self.layers = []
            for i, (inputs, outputs) in enumerate(zip(dims[:-1], dims[1:])):
                kernel = data[f"actor_{i}_kernel"].copy()
                bias = data[f"actor_{i}_bias"].copy()
                if kernel.shape != (inputs, outputs) or bias.shape != (outputs,):
                    raise ValueError(f"Actor layer {i} has invalid shape")
                self.layers.append(
                    (torch.as_tensor(kernel, device=device), torch.as_tensor(bias, device=device))
                )
        if (
            not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.denominator).all()
            or (self.denominator <= 0).any()
        ):
            raise ValueError("Invalid observation normalization")
        for kernel, bias in self.layers:
            if not torch.isfinite(kernel).all() or not torch.isfinite(bias).all():
                raise ValueError("Nonfinite actor parameters")

    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        """Return raw deterministic action means on the observation device."""
        x = (observations - self.mean) / self.denominator
        for kernel, bias in self.layers[:-1]:
            x = F.elu(x @ kernel + bias)
        kernel, bias = self.layers[-1]
        return x @ kernel + bias  # raw means; the untouched task applies tanh
