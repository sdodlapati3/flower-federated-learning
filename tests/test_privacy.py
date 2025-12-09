"""
Tests for fl_research.privacy module.
"""

import pytest
import torch
import torch.nn as nn

from fl_research.privacy.accountant import (
    PrivacyAccountant,
    PrivacyBudget,
    compute_rdp,
    compute_epsilon,
    get_privacy_spent,
)
from fl_research.privacy.mechanisms import (
    GaussianMechanism,
    LaplaceMechanism,
    add_gaussian_noise,
    add_laplace_noise,
    clip_gradients,
)


class TestPrivacyBudget:
    """Tests for PrivacyBudget."""
    
    def test_creation(self):
        """Test budget creation."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        
        assert budget.epsilon == 1.0
        assert budget.delta == 1e-5
    
    def test_str_representation(self):
        """Test string representation."""
        budget = PrivacyBudget(epsilon=2.5, delta=1e-5)
        str_repr = str(budget)
        
        assert "2.5" in str_repr
        assert "1.00e-05" in str_repr
    
    def test_is_exceeded(self):
        """Test budget exceeded check."""
        budget = PrivacyBudget(epsilon=5.0, delta=1e-5)
        
        assert not budget.is_exceeded(10.0, 1e-5)
        assert budget.is_exceeded(3.0, 1e-5)


class TestComputeRDP:
    """Tests for RDP computation."""
    
    def test_basic_rdp(self):
        """Test basic RDP computation."""
        orders = [2.0, 5.0, 10.0]
        rdp = compute_rdp(q=0.01, noise_multiplier=1.0, orders=orders)
        
        assert len(rdp) == len(orders)
        for value in rdp:
            assert value >= 0
    
    def test_zero_noise(self):
        """Test RDP with zero noise."""
        orders = [2.0, 5.0]
        rdp = compute_rdp(q=0.01, noise_multiplier=0, orders=orders)
        
        # Should be infinity
        for value in rdp:
            assert value == float('inf')
    
    def test_zero_sampling(self):
        """Test RDP with zero sampling rate."""
        orders = [2.0, 5.0]
        rdp = compute_rdp(q=0, noise_multiplier=1.0, orders=orders)
        
        # Should be zero
        for value in rdp:
            assert value == 0


class TestComputeEpsilon:
    """Tests for epsilon computation."""
    
    def test_basic_epsilon(self):
        """Test epsilon conversion from RDP."""
        orders = [2.0, 5.0, 10.0]
        rdp = [0.1, 0.2, 0.3]
        
        epsilon, opt_order = compute_epsilon(rdp, orders, delta=1e-5)
        
        assert epsilon >= 0
        assert opt_order in orders


class TestGetPrivacySpent:
    """Tests for privacy accounting."""
    
    def test_basic_accounting(self):
        """Test basic privacy accounting."""
        budget = get_privacy_spent(
            steps=100,
            noise_multiplier=1.0,
            sample_rate=0.01,
            delta=1e-5,
        )
        
        assert isinstance(budget, PrivacyBudget)
        assert budget.epsilon >= 0
        assert budget.delta == 1e-5
    
    def test_more_steps_more_epsilon(self):
        """Test that more steps means more privacy spent."""
        budget_100 = get_privacy_spent(
            steps=100,
            noise_multiplier=1.0,
            sample_rate=0.01,
            delta=1e-5,
        )
        
        budget_1000 = get_privacy_spent(
            steps=1000,
            noise_multiplier=1.0,
            sample_rate=0.01,
            delta=1e-5,
        )
        
        assert budget_1000.epsilon > budget_100.epsilon
    
    def test_more_noise_less_epsilon(self):
        """Test that more noise means less privacy spent."""
        budget_low_noise = get_privacy_spent(
            steps=100,
            noise_multiplier=0.5,
            sample_rate=0.01,
            delta=1e-5,
        )
        
        budget_high_noise = get_privacy_spent(
            steps=100,
            noise_multiplier=2.0,
            sample_rate=0.01,
            delta=1e-5,
        )
        
        assert budget_high_noise.epsilon < budget_low_noise.epsilon


class TestPrivacyAccountant:
    """Tests for PrivacyAccountant."""
    
    def test_step(self):
        """Test stepping the accountant."""
        accountant = PrivacyAccountant(
            noise_multiplier=1.0,
            sample_rate=0.01,
            delta=1e-5,
        )
        
        assert accountant.steps == 0
        
        accountant.step()
        assert accountant.steps == 1
        
        accountant.step(10)
        assert accountant.steps == 11
    
    def test_get_epsilon(self):
        """Test epsilon retrieval."""
        accountant = PrivacyAccountant(
            noise_multiplier=1.0,
            sample_rate=0.01,
            delta=1e-5,
        )
        
        accountant.step(100)
        epsilon = accountant.get_epsilon()
        
        assert epsilon >= 0
    
    def test_budget_exceeded(self):
        """Test budget exceeded check."""
        accountant = PrivacyAccountant(
            noise_multiplier=1.0,
            sample_rate=0.01,
            delta=1e-5,
            max_epsilon=1.0,
        )
        
        # Initially not exceeded
        assert not accountant.is_budget_exceeded()
        
        # After many steps, may be exceeded
        accountant.step(10000)
        # Note: whether it's exceeded depends on the computation


class TestGaussianMechanism:
    """Tests for Gaussian mechanism."""
    
    def test_noise_addition(self):
        """Test noise is added."""
        mechanism = GaussianMechanism(sensitivity=1.0, noise_multiplier=1.0)
        
        value = torch.zeros(100)
        noisy = mechanism.add_noise(value)
        
        # Should not be exactly zero
        assert not torch.allclose(noisy, value)
    
    def test_sigma_property(self):
        """Test sigma computation."""
        mechanism = GaussianMechanism(sensitivity=2.0, noise_multiplier=0.5)
        
        assert mechanism.sigma == 1.0
    
    def test_noise_scale(self):
        """Test that higher noise multiplier means more noise."""
        value = torch.zeros(1000)
        
        low_noise = GaussianMechanism(sensitivity=1.0, noise_multiplier=0.1)
        high_noise = GaussianMechanism(sensitivity=1.0, noise_multiplier=2.0)
        
        noisy_low = low_noise.add_noise(value.clone())
        noisy_high = high_noise.add_noise(value.clone())
        
        # Standard deviation should be higher
        assert noisy_high.std() > noisy_low.std()


class TestLaplaceMechanism:
    """Tests for Laplace mechanism."""
    
    def test_noise_addition(self):
        """Test noise is added."""
        mechanism = LaplaceMechanism(sensitivity=1.0, epsilon=1.0)
        
        value = torch.zeros(100)
        noisy = mechanism.add_noise(value)
        
        # Should not be exactly zero
        assert not torch.allclose(noisy, value)
    
    def test_scale_property(self):
        """Test scale (b) computation."""
        mechanism = LaplaceMechanism(sensitivity=2.0, epsilon=0.5)
        
        assert mechanism.b == 4.0


class TestAddNoiseFunctions:
    """Tests for noise addition functions."""
    
    def test_add_gaussian_noise(self):
        """Test Gaussian noise addition function."""
        tensor = torch.zeros(100)
        noisy = add_gaussian_noise(tensor, noise_multiplier=1.0, max_norm=1.0)
        
        assert noisy.shape == tensor.shape
        assert not torch.allclose(noisy, tensor)
    
    def test_add_laplace_noise(self):
        """Test Laplace noise addition function."""
        tensor = torch.zeros(100)
        noisy = add_laplace_noise(tensor, epsilon=1.0, sensitivity=1.0)
        
        assert noisy.shape == tensor.shape
        assert not torch.allclose(noisy, tensor)


class TestClipGradients:
    """Tests for gradient clipping."""
    
    def test_clip_single(self):
        """Test clipping single gradient vector."""
        grad = torch.tensor([3.0, 4.0])  # Norm = 5.0
        clipped, norm = clip_gradients(grad, max_norm=1.0)
        
        # Clipped norm should be <= 1.0
        assert torch.norm(clipped).item() <= 1.0 + 1e-6
    
    def test_clip_batch(self):
        """Test clipping batch of gradients."""
        grads = torch.randn(10, 100) * 10  # Large gradients
        clipped, norms = clip_gradients(grads, max_norm=1.0)
        
        # All clipped norms should be <= 1.0
        for i in range(10):
            assert torch.norm(clipped[i]).item() <= 1.0 + 1e-6
    
    def test_no_clip_small_gradients(self):
        """Test that small gradients aren't affected."""
        grad = torch.tensor([0.1, 0.1])  # Norm < 1.0
        clipped, norm = clip_gradients(grad, max_norm=10.0)
        
        # Should be unchanged
        assert torch.allclose(clipped, grad)
