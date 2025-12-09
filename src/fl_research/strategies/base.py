"""
Base Strategy Utilities
=======================

Common utilities and configuration for FL strategies.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import torch
import numpy as np

try:
    import flwr as fl
    from flwr.common import (
        Parameters,
        Scalar,
        FitRes,
        EvaluateRes,
        NDArrays,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False


class StrategyType(Enum):
    """Available FL strategies."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"


@dataclass
class StrategyConfig:
    """
    Configuration for FL strategies.
    
    Example:
        >>> config = StrategyConfig(
        ...     strategy_type="fedprox",
        ...     fraction_fit=0.1,
        ...     fraction_evaluate=0.1,
        ...     min_fit_clients=2,
        ...     proximal_mu=0.1,  # FedProx parameter
        ... )
    """
    # Strategy type
    strategy_type: str = "fedavg"
    
    # Client selection
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    
    # FedProx parameter
    proximal_mu: float = 0.1
    
    # SCAFFOLD parameters
    server_lr: float = 1.0
    
    # Training parameters
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Differential privacy
    use_dp: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    target_epsilon: Optional[float] = None
    target_delta: float = 1e-5
    
    # Extra parameters
    extra: Dict[str, Any] = field(default_factory=dict)


def create_strategy(config: StrategyConfig):
    """
    Create a Flower strategy from configuration.
    
    Args:
        config: Strategy configuration
    
    Returns:
        Flower Strategy instance
    
    Example:
        >>> config = StrategyConfig(strategy_type="fedprox", proximal_mu=0.1)
        >>> strategy = create_strategy(config)
    """
    if not FLWR_AVAILABLE:
        raise ImportError("Flower is required. Install with: pip install flwr")
    
    from fl_research.strategies.fedavg import FedAvgStrategy
    from fl_research.strategies.fedprox import FedProxStrategy
    from fl_research.strategies.scaffold import SCAFFOLDStrategy
    
    strategy_type = StrategyType(config.strategy_type.lower())
    
    base_kwargs = {
        "fraction_fit": config.fraction_fit,
        "fraction_evaluate": config.fraction_evaluate,
        "min_fit_clients": config.min_fit_clients,
        "min_evaluate_clients": config.min_evaluate_clients,
        "min_available_clients": config.min_available_clients,
    }
    
    if strategy_type == StrategyType.FEDAVG:
        return FedAvgStrategy(**base_kwargs)
    elif strategy_type == StrategyType.FEDPROX:
        return FedProxStrategy(
            proximal_mu=config.proximal_mu,
            **base_kwargs,
        )
    elif strategy_type == StrategyType.SCAFFOLD:
        return SCAFFOLDStrategy(
            server_lr=config.server_lr,
            **base_kwargs,
        )
    else:
        raise ValueError(f"Unknown strategy type: {config.strategy_type}")


def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """Extract model parameters as numpy arrays."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Set model parameters from numpy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def aggregate_parameters(
    results: List[Tuple[int, List[np.ndarray]]],
) -> List[np.ndarray]:
    """
    Weighted average of model parameters.
    
    Args:
        results: List of (num_samples, parameters) tuples
    
    Returns:
        Aggregated parameters
    """
    total_samples = sum(n for n, _ in results)
    
    if total_samples == 0:
        # Return first set of parameters
        return results[0][1] if results else []
    
    # Weighted average
    aggregated = []
    num_params = len(results[0][1])
    
    for param_idx in range(num_params):
        weighted_sum = sum(
            n * params[param_idx] for n, params in results
        )
        aggregated.append(weighted_sum / total_samples)
    
    return aggregated


def compute_model_delta(
    old_params: List[np.ndarray],
    new_params: List[np.ndarray],
) -> List[np.ndarray]:
    """Compute difference between model parameters."""
    return [new - old for old, new in zip(old_params, new_params)]


def add_model_delta(
    params: List[np.ndarray],
    delta: List[np.ndarray],
    scale: float = 1.0,
) -> List[np.ndarray]:
    """Add scaled delta to parameters."""
    return [p + scale * d for p, d in zip(params, delta)]
