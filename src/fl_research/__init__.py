"""
FL Research Library
====================

A professional-grade federated learning research library built on Flower.

Modules:
    models: Reusable neural network architectures
    data: Dataset loading and partitioning strategies
    privacy: Differential privacy utilities and accounting
    strategies: FL aggregation strategies (FedAvg, FedProx, SCAFFOLD)
    utils: Configuration, logging, metrics, checkpointing
"""

__version__ = "0.1.0"
__author__ = "FL Research Team"

# Core utilities (always available)
from fl_research.utils.config import Config, load_config
from fl_research.utils.reproducibility import set_seed, get_device
from fl_research.utils.metrics import MetricsTracker
from fl_research.utils.logging import ExperimentLogger

__all__ = [
    # Version
    "__version__",
    # Config
    "Config",
    "load_config",
    # Reproducibility
    "set_seed",
    "get_device",
    # Metrics
    "MetricsTracker",
    # Logging
    "ExperimentLogger",
]
