"""
Privacy Mechanisms
==================

Noise mechanisms for differential privacy.
"""

from typing import Union, Tuple
from abc import ABC, abstractmethod
import math

import torch
import numpy as np


class NoiseMechanism(ABC):
    """Base class for noise mechanisms."""
    
    @abstractmethod
    def add_noise(self, value: torch.Tensor) -> torch.Tensor:
        """Add noise to value."""
        pass
    
    @abstractmethod
    def get_noise_scale(self) -> float:
        """Get the noise scale (sigma or b)."""
        pass


class GaussianMechanism(NoiseMechanism):
    """
    Gaussian (Normal) mechanism for approximate DP.
    
    Adds N(0, σ²) noise where σ = sensitivity * noise_multiplier.
    Provides (ε, δ)-differential privacy.
    
    Example:
        >>> mechanism = GaussianMechanism(sensitivity=1.0, noise_multiplier=1.0)
        >>> noisy_value = mechanism.add_noise(torch.tensor([1.0, 2.0, 3.0]))
    """
    
    def __init__(
        self,
        sensitivity: float,
        noise_multiplier: float,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Args:
            sensitivity: L2 sensitivity of the query
            noise_multiplier: Multiplier for noise scale (sigma = sensitivity * noise_multiplier)
            device: Device for tensor operations
        """
        self.sensitivity = sensitivity
        self.noise_multiplier = noise_multiplier
        self.device = torch.device(device)
    
    @property
    def sigma(self) -> float:
        """Standard deviation of the noise."""
        return self.sensitivity * self.noise_multiplier
    
    def get_noise_scale(self) -> float:
        return self.sigma
    
    def add_noise(self, value: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to value."""
        noise = torch.normal(
            mean=0,
            std=self.sigma,
            size=value.shape,
            device=self.device,
            dtype=value.dtype,
        )
        return value + noise.to(value.device)
    
    def sample_noise(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Sample noise without adding to value."""
        return torch.normal(
            mean=0,
            std=self.sigma,
            size=shape,
            device=self.device,
        )


class LaplaceMechanism(NoiseMechanism):
    """
    Laplace mechanism for pure DP.
    
    Adds Laplace(0, b) noise where b = sensitivity / epsilon.
    Provides ε-differential privacy.
    
    Example:
        >>> mechanism = LaplaceMechanism(sensitivity=1.0, epsilon=1.0)
        >>> noisy_value = mechanism.add_noise(torch.tensor([1.0, 2.0, 3.0]))
    """
    
    def __init__(
        self,
        sensitivity: float,
        epsilon: float,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Args:
            sensitivity: L1 sensitivity of the query
            epsilon: Privacy budget
            device: Device for tensor operations
        """
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.device = torch.device(device)
    
    @property
    def b(self) -> float:
        """Scale parameter of the Laplace distribution."""
        return self.sensitivity / self.epsilon
    
    def get_noise_scale(self) -> float:
        return self.b
    
    def add_noise(self, value: torch.Tensor) -> torch.Tensor:
        """Add Laplace noise to value."""
        # Use the inverse CDF method
        u = torch.rand(value.shape, device=self.device, dtype=value.dtype) - 0.5
        noise = -self.b * torch.sign(u) * torch.log(1 - 2 * torch.abs(u))
        return value + noise.to(value.device)
    
    def sample_noise(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Sample Laplace noise."""
        u = torch.rand(shape, device=self.device) - 0.5
        return -self.b * torch.sign(u) * torch.log(1 - 2 * torch.abs(u))


def add_gaussian_noise(
    tensor: torch.Tensor,
    noise_multiplier: float,
    max_norm: float,
) -> torch.Tensor:
    """
    Add calibrated Gaussian noise to a tensor.
    
    Args:
        tensor: Input tensor
        noise_multiplier: Noise multiplier (sigma / max_norm)
        max_norm: Clipping norm / sensitivity
    
    Returns:
        Noisy tensor
    """
    sigma = noise_multiplier * max_norm
    noise = torch.randn_like(tensor) * sigma
    return tensor + noise


def add_laplace_noise(
    tensor: torch.Tensor,
    epsilon: float,
    sensitivity: float,
) -> torch.Tensor:
    """
    Add calibrated Laplace noise to a tensor.
    
    Args:
        tensor: Input tensor
        epsilon: Privacy budget
        sensitivity: L1 sensitivity
    
    Returns:
        Noisy tensor
    """
    b = sensitivity / epsilon
    u = torch.rand_like(tensor) - 0.5
    noise = -b * torch.sign(u) * torch.log(1 - 2 * torch.abs(u))
    return tensor + noise


def clip_gradients(
    gradients: torch.Tensor,
    max_norm: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Clip gradients by L2 norm per sample.
    
    Args:
        gradients: Tensor of shape (batch_size, ...) or flat gradients
        max_norm: Maximum L2 norm
    
    Returns:
        Tuple of (clipped_gradients, norms)
    """
    # Compute per-sample norms
    if gradients.dim() == 1:
        # Flat gradients - single sample
        norm = torch.norm(gradients, p=2)
        scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
        return gradients * scale, norm.unsqueeze(0)
    else:
        # Batch of gradients
        batch_size = gradients.shape[0]
        flat = gradients.view(batch_size, -1)
        norms = torch.norm(flat, p=2, dim=1)
        
        scales = torch.clamp(max_norm / (norms + 1e-8), max=1.0)
        scales = scales.view(batch_size, *([1] * (gradients.dim() - 1)))
        
        return gradients * scales, norms


def compute_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: int,
    dataset_size: int,
    batch_size: int,
) -> float:
    """
    Compute noise multiplier to achieve target privacy budget.
    
    Uses binary search to find the noise multiplier.
    
    Args:
        target_epsilon: Target epsilon
        target_delta: Target delta
        sample_rate: Subsampling rate
        epochs: Number of training epochs
        dataset_size: Size of training dataset
        batch_size: Batch size
    
    Returns:
        Required noise multiplier
    """
    from fl_research.privacy.accountant import get_privacy_spent
    
    steps = epochs * (dataset_size // batch_size)
    
    # Binary search for noise multiplier
    low, high = 0.01, 100.0
    
    for _ in range(100):  # Max iterations
        mid = (low + high) / 2
        budget = get_privacy_spent(
            steps=steps,
            noise_multiplier=mid,
            sample_rate=sample_rate,
            delta=target_delta,
        )
        
        if abs(budget.epsilon - target_epsilon) < 0.01:
            return mid
        elif budget.epsilon > target_epsilon:
            low = mid
        else:
            high = mid
    
    return mid
