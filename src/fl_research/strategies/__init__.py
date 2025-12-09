"""
Strategies Module
=================

Federated Learning strategies and algorithms.

This module provides:
- FedAvg (baseline)
- FedProx (for heterogeneous data)
- SCAFFOLD (for variance reduction)
- Custom strategy builders
"""

from fl_research.strategies.fedavg import (
    FedAvgStrategy,
    weighted_average,
)
from fl_research.strategies.fedprox import (
    FedProxStrategy,
    FedProxClient,
)
from fl_research.strategies.scaffold import (
    SCAFFOLDStrategy,
    SCAFFOLDClient,
)
from fl_research.strategies.base import (
    StrategyConfig,
    create_strategy,
)

__all__ = [
    # FedAvg
    "FedAvgStrategy",
    "weighted_average",
    # FedProx
    "FedProxStrategy", 
    "FedProxClient",
    # SCAFFOLD
    "SCAFFOLDStrategy",
    "SCAFFOLDClient",
    # Base
    "StrategyConfig",
    "create_strategy",
]
