"""
Model Registry
==============

Registry pattern for managing and instantiating models by name.
Allows easy model switching via configuration.
"""

from typing import Dict, Type, Optional, Any, Callable
import torch.nn as nn

# Global model registry
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable:
    """
    Decorator to register a model class.
    
    Args:
        name: Name to register the model under
    
    Example:
        >>> @register_model("my_cnn")
        ... class MyCNN(nn.Module):
        ...     def __init__(self, num_classes=10):
        ...         super().__init__()
        ...         # ...
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs) -> nn.Module:
    """
    Instantiate a model by name.
    
    Args:
        name: Registered model name
        **kwargs: Arguments to pass to model constructor
    
    Returns:
        Instantiated model
    
    Raises:
        ValueError: If model name not found in registry
    
    Example:
        >>> model = get_model("cifar10_cnn", num_classes=10, dp_compatible=True)
    """
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found. Available: {available}")
    
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list:
    """Return list of all registered model names."""
    return list(_MODEL_REGISTRY.keys())


class ModelRegistry:
    """
    Class-based model registry for more control.
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.register("my_model", MyModel)
        >>> model = registry.get("my_model", num_classes=10)
    """
    
    def __init__(self):
        self._registry: Dict[str, Type[nn.Module]] = {}
    
    def register(self, name: str, model_class: Type[nn.Module]) -> None:
        """Register a model class with given name."""
        if name in self._registry:
            raise ValueError(f"Model '{name}' already registered")
        self._registry[name] = model_class
    
    def get(self, name: str, **kwargs) -> nn.Module:
        """Instantiate a model by name with given kwargs."""
        if name not in self._registry:
            raise ValueError(f"Model '{name}' not found. Available: {list(self._registry.keys())}")
        return self._registry[name](**kwargs)
    
    def list(self) -> list:
        """Return list of registered model names."""
        return list(self._registry.keys())
    
    def __contains__(self, name: str) -> bool:
        return name in self._registry


# Register built-in models
def _register_builtin_models():
    """Register all built-in models."""
    from fl_research.models.cnn import SimpleCNN, CIFAR10CNN, CIFAR10CNNOpacus, ResNetSmall
    from fl_research.models.mlp import MLP, TwoLayerMLP, LogisticRegression
    
    # CNN models
    _MODEL_REGISTRY["simple_cnn"] = SimpleCNN
    _MODEL_REGISTRY["mnist_cnn"] = SimpleCNN
    _MODEL_REGISTRY["fmnist_cnn"] = SimpleCNN
    _MODEL_REGISTRY["cifar10_cnn"] = CIFAR10CNN
    _MODEL_REGISTRY["cifar10_cnn_opacus"] = CIFAR10CNNOpacus
    _MODEL_REGISTRY["resnet_small"] = ResNetSmall
    
    # MLP models
    _MODEL_REGISTRY["mlp"] = MLP
    _MODEL_REGISTRY["two_layer_mlp"] = TwoLayerMLP
    _MODEL_REGISTRY["logistic_regression"] = LogisticRegression


# Auto-register on import
try:
    _register_builtin_models()
except ImportError:
    # Models not yet available during initial import
    pass
