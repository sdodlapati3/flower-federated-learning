"""Differential Privacy Experiments with Multiple Epsilon Settings.

This script runs FL experiments with different DP configurations:
- Multiple epsilon values (ε ≈ 2, 4, 8)
- IID vs Non-IID data partitioning
- Tracks accuracy vs privacy budget tradeoff
"""

import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


# Experiment configurations
EXPERIMENTS = [
    # (name, epsilon_target, noise_multiplier, clipping_norm, is_iid, alpha)
    # IID experiments with different epsilon
    ("iid_eps2", 2.0, 2.0, 1.0, True, None),
    ("iid_eps4", 4.0, 1.0, 1.0, True, None),
    ("iid_eps8", 8.0, 0.5, 1.0, True, None),
    ("iid_eps50", 50.0, 0.1, 1.0, True, None),  # Low privacy (baseline)
    
    # Non-IID experiments (Dirichlet alpha=0.5) with different epsilon
    ("noniid_eps2", 2.0, 2.0, 1.0, False, 0.5),
    ("noniid_eps4", 4.0, 1.0, 1.0, False, 0.5),
    ("noniid_eps8", 8.0, 0.5, 1.0, False, 0.5),
    ("noniid_eps50", 50.0, 0.1, 1.0, False, 0.5),
]

# Number of rounds and clients
NUM_ROUNDS = 5
NUM_CLIENTS = 5
TARGET_DELTA = 1e-5


def create_experiment_config(exp_name, noise_mult, clip_norm, is_iid, alpha=None):
    """Create pyproject.toml for experiment."""
    
    config = f'''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "dp_experiment"
version = "1.0.0"
description = "DP Experiment: {exp_name}"

dependencies = [
    "flwr[simulation]>=1.24.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "opacus>=1.4.0",
    "tqdm",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "experiment"

[tool.flwr.app.components]
serverapp = "dp_experiment.server_app:app"
clientapp = "dp_experiment.client_app:app"

[tool.flwr.app.config]
num-server-rounds = {NUM_ROUNDS}
target-delta = {TARGET_DELTA}
max-grad-norm = {clip_norm}
noise-multiplier = {noise_mult}
is-iid = {"true" if is_iid else "false"}
dirichlet-alpha = {alpha if alpha else 1.0}
experiment-name = "{exp_name}"

[tool.flwr.federations]
default = "local-simulation"

[tool.flwr.federations.local-simulation]
options.num-supernodes = {NUM_CLIENTS}
'''
    return config


def print_summary():
    """Print experiment summary."""
    print("\n" + "="*70)
    print("DIFFERENTIAL PRIVACY EXPERIMENTS")
    print("="*70)
    print(f"\nExperiments to run: {len(EXPERIMENTS)}")
    print(f"Rounds per experiment: {NUM_ROUNDS}")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Target δ: {TARGET_DELTA}")
    print("\nExperiment Matrix:")
    print("-"*70)
    print(f"{'Name':<15} {'Target ε':<10} {'Noise σ':<10} {'Clip':<8} {'IID':<6} {'Alpha':<8}")
    print("-"*70)
    for name, eps, noise, clip, iid, alpha in EXPERIMENTS:
        print(f"{name:<15} {eps:<10.1f} {noise:<10.2f} {clip:<8.1f} {str(iid):<6} {str(alpha):<8}")
    print("-"*70)
    print("\nExpected findings:")
    print("  • Lower ε → stronger privacy → lower accuracy")
    print("  • Non-IID data → harder learning → lower accuracy")
    print("  • ε ≈ 2 is considered strong privacy")
    print("  • ε ≈ 8 is moderate privacy")
    print("  • ε > 10 provides weak privacy guarantees")
    print("="*70)


if __name__ == "__main__":
    print_summary()
    print("\nTo run experiments, use the SLURM script: run_dp_experiments.slurm")
