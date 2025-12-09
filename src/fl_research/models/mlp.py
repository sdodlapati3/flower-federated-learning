"""
Multi-Layer Perceptron (MLP) Architectures
==========================================

MLP models for tabular data and simple classification tasks.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerMLP(nn.Module):
    """
    Simple two-layer MLP for basic classification.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension (default: 128)
        output_dim: Output dimension / number of classes (default: 10)
        dropout: Dropout probability (default: 0.0)
    
    Example:
        >>> model = TwoLayerMLP(input_dim=784, hidden_dim=128, output_dim=10)
        >>> x = torch.randn(32, 784)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 10])
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 10,
        dropout: float = 0.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten if needed
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class MLP(nn.Module):
    """
    Flexible Multi-Layer Perceptron with configurable layers.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension / number of classes
        dropout: Dropout probability between layers (default: 0.0)
        activation: Activation function ('relu', 'gelu', 'tanh') (default: 'relu')
        batch_norm: Whether to use batch normalization (default: False)
    
    Example:
        >>> model = MLP(
        ...     input_dim=784,
        ...     hidden_dims=[256, 128, 64],
        ...     output_dim=10,
        ...     dropout=0.2
        ... )
        >>> x = torch.randn(32, 784)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 10])
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
        batch_norm: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Select activation function
        self.activation = self._get_activation(activation)
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "silu": nn.SiLU(),
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name.lower()]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten if needed
        x = self.hidden_layers(x)
        return self.output_layer(x)
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LogisticRegression(nn.Module):
    """
    Logistic Regression as a single-layer neural network.
    
    Useful for federated learning baselines and simple classification.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Number of classes (default: 2 for binary)
    
    Example:
        >>> model = LogisticRegression(input_dim=784, output_dim=10)
        >>> x = torch.randn(32, 784)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 10])
    """
    
    def __init__(self, input_dim: int, output_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.linear(x)
