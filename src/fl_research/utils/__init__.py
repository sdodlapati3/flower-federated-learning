"""
Utilities Module
=================

Common utilities for FL experiments:
    config: Configuration management
    logging: Structured logging
    metrics: Metrics tracking and aggregation
    checkpointing: Model checkpoint management
    reproducibility: Seed and reproducibility utilities
"""

from fl_research.utils.config import (
    Config,
    ModelConfig,
    DataConfig,
    PrivacyConfig,
    TrainingConfig,
    LoggingConfig,
    load_config,
    save_config,
)
from fl_research.utils.logging import ExperimentLogger
from fl_research.utils.metrics import (
    MetricsTracker,
    RoundMetrics,
    compute_accuracy,
    aggregate_weighted_average,
)
from fl_research.utils.checkpointing import Checkpointer, save_checkpoint, load_checkpoint
from fl_research.utils.reproducibility import set_seed, get_device

__all__ = [
    # Config
    "Config",
    "ModelConfig",
    "DataConfig",
    "PrivacyConfig",
    "TrainingConfig",
    "LoggingConfig",
    "load_config",
    "save_config",
    # Logging
    "ExperimentLogger",
    # Metrics
    "MetricsTracker",
    "RoundMetrics",
    "compute_accuracy",
    "aggregate_weighted_average",
    # Checkpointing
    "Checkpointer",
    "save_checkpoint",
    "load_checkpoint",
    # Reproducibility
    "set_seed",
    "get_device",
]
