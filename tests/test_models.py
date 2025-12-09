"""
Tests for fl_research.models module.
"""

import pytest
import torch
import torch.nn as nn

from fl_research.models import (
    SimpleCNN,
    CIFAR10CNN,
    CIFAR10CNNOpacus,
    MLP,
    TwoLayerMLP,
)
from fl_research.models.registry import ModelRegistry, get_model


class TestSimpleCNN:
    """Tests for SimpleCNN model."""
    
    def test_forward_pass(self, sample_image_data, device):
        """Test forward pass with sample data."""
        X, y = sample_image_data
        model = SimpleCNN(num_classes=10).to(device)
        X = X.to(device)
        
        output = model(X)
        
        assert output.shape == (X.shape[0], 10)
    
    def test_different_num_classes(self):
        """Test with different number of classes."""
        for num_classes in [2, 5, 10, 100]:
            model = SimpleCNN(num_classes=num_classes)
            x = torch.randn(1, 1, 28, 28)
            output = model(x)
            assert output.shape == (1, num_classes)


class TestCIFAR10CNN:
    """Tests for CIFAR10CNN model."""
    
    def test_forward_pass(self, sample_cifar_data, device):
        """Test forward pass with CIFAR-like data."""
        X, y = sample_cifar_data
        model = CIFAR10CNN().to(device)
        X = X.to(device)
        
        output = model(X)
        
        assert output.shape == (X.shape[0], 10)
    
    def test_3_channel_input(self):
        """Test that model accepts 3-channel input."""
        model = CIFAR10CNN()
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)


class TestCIFAR10CNNOpacus:
    """Tests for CIFAR10CNNOpacus (DP-compatible) model."""
    
    def test_forward_pass(self, sample_cifar_data, device):
        """Test forward pass."""
        X, y = sample_cifar_data
        model = CIFAR10CNNOpacus().to(device)
        X = X.to(device)
        
        output = model(X)
        
        assert output.shape == (X.shape[0], 10)
    
    def test_groupnorm_for_dp(self):
        """Test that model uses GroupNorm (DP-compatible)."""
        model = CIFAR10CNNOpacus()
        
        # Check for GroupNorm, not BatchNorm
        has_groupnorm = False
        has_batchnorm = False
        
        for module in model.modules():
            if isinstance(module, nn.GroupNorm):
                has_groupnorm = True
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                has_batchnorm = True
        
        assert has_groupnorm, "DP model should use GroupNorm"
        assert not has_batchnorm, "DP model should not have BatchNorm"


class TestMLP:
    """Tests for MLP model."""
    
    def test_forward_pass(self, sample_data, device):
        """Test forward pass with sample data."""
        X, y = sample_data
        model = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10).to(device)
        X = X.to(device)
        
        output = model(X)
        
        assert output.shape == (X.shape[0], 10)
    
    def test_custom_hidden_dims(self):
        """Test with custom hidden dimensions."""
        model = MLP(input_dim=100, hidden_dims=[256, 128, 64], output_dim=5)
        x = torch.randn(8, 100)
        output = model(x)
        assert output.shape == (8, 5)


class TestTwoLayerMLP:
    """Tests for TwoLayerMLP model."""
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = TwoLayerMLP(input_dim=100, hidden_dim=64, output_dim=10)
        x = torch.randn(4, 100)
        output = model(x)
        assert output.shape == (4, 10)


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    def test_list_models(self):
        """Test listing available models."""
        from fl_research.models.registry import list_models
        models = list_models()
        
        assert "simple_cnn" in models or len(models) > 0
    
    def test_get_model(self):
        """Test getting model from registry."""
        model = get_model("simple_cnn", num_classes=10)
        
        assert isinstance(model, nn.Module)
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        assert output.shape == (1, 10)
