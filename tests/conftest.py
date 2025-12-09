"""
FL Research Test Suite
=======================

Comprehensive tests for the fl_research library.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Generate random data
    num_samples = 100
    input_size = 784  # MNIST-like
    num_classes = 10
    
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y


@pytest.fixture
def sample_image_data():
    """Create sample image data for testing."""
    num_samples = 32
    channels = 1
    height = 28
    width = 28
    num_classes = 10
    
    X = torch.randn(num_samples, channels, height, width)
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y


@pytest.fixture
def sample_cifar_data():
    """Create CIFAR-like data for testing."""
    num_samples = 32
    channels = 3
    height = 32
    width = 32
    num_classes = 10
    
    X = torch.randn(num_samples, channels, height, width)
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for tests."""
    return tmp_path


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for partitioner testing."""
    from torch.utils.data import TensorDataset
    
    num_samples = 1000
    X = torch.randn(num_samples, 28, 28)
    y = torch.randint(0, 10, (num_samples,))
    
    return TensorDataset(X, y)


# Markers for different test categories
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
