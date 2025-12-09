"""
Neural Network Model Architectures
===================================

Consolidated model definitions for federated learning experiments.
All models are designed to be:
- Opacus-compatible (GroupNorm instead of BatchNorm)
- Configurable (flexible input/output dimensions)
- Well-documented

Available Models:
    SimpleCNN: Basic CNN for MNIST/Fashion-MNIST (28x28 grayscale)
    CIFAR10CNN: CNN for CIFAR-10 (32x32 RGB)
    CIFAR10CNNOpacus: DP-compatible CIFAR-10 CNN
    ResNetSmall: Small ResNet variant for FL experiments
    MLP: Multi-layer perceptron for tabular data
    TwoLayerMLP: Simple two-layer MLP
"""

from fl_research.models.cnn import SimpleCNN, CIFAR10CNN, CIFAR10CNNOpacus, ResNetSmall
from fl_research.models.mlp import MLP, TwoLayerMLP, LogisticRegression
from fl_research.models.registry import ModelRegistry, get_model, register_model

__all__ = [
    # CNN Models
    "SimpleCNN",
    "CIFAR10CNN", 
    "CIFAR10CNNOpacus",
    "ResNetSmall",
    # MLP Models
    "MLP",
    "TwoLayerMLP",
    "LogisticRegression",
    # Registry
    "ModelRegistry",
    "get_model",
    "register_model",
]
