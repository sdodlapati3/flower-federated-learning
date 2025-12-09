"""
Tests for fl_research.utils module.
"""

import pytest
import torch
import torch.nn as nn
import json
from pathlib import Path

from fl_research.utils.config import Config, load_config, save_config
from fl_research.utils.metrics import MetricsTracker, RoundMetrics, compute_accuracy, aggregate_weighted_average
from fl_research.utils.checkpointing import Checkpointer, save_checkpoint, load_checkpoint
from fl_research.utils.reproducibility import set_seed, get_device, SeedContext


class TestConfig:
    """Tests for Config."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        
        assert config.experiment_name is not None
        assert config.data.num_clients > 0
        assert config.training.num_rounds > 0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = Config(
            experiment_name="test_exp",
        )
        
        assert config.experiment_name == "test_exp"


class TestConfigSaveLoad:
    """Tests for config save/load."""
    
    def test_save_and_load_yaml(self, temp_dir):
        """Test saving and loading YAML config."""
        config = Config(
            experiment_name="yaml_test",
        )
        
        path = temp_dir / "config.yaml"
        save_config(config, str(path))
        
        loaded = load_config(str(path))
        
        assert loaded.experiment_name == "yaml_test"
    
    def test_save_and_load_json(self, temp_dir):
        """Test saving and loading JSON config."""
        config = Config(
            experiment_name="json_test",
        )
        
        path = temp_dir / "config.json"
        save_config(config, str(path))
        
        loaded = load_config(str(path))
        
        assert loaded.experiment_name == "json_test"


class TestMetricsTracker:
    """Tests for MetricsTracker."""
    
    def test_add_round(self):
        """Test adding round metrics."""
        tracker = MetricsTracker()
        
        tracker.add_round(1, accuracy=0.85, loss=0.5)
        tracker.add_round(2, accuracy=0.90, loss=0.3)
        
        assert len(tracker.rounds) == 2
    
    def test_get_best(self):
        """Test getting best metric."""
        tracker = MetricsTracker()
        
        tracker.add_round(1, accuracy=0.85, loss=0.5)
        tracker.add_round(2, accuracy=0.92, loss=0.3)
        tracker.add_round(3, accuracy=0.88, loss=0.4)
        
        best_round, best_acc = tracker.get_best("accuracy")
        
        assert best_round == 2
        assert best_acc == 0.92
    
    def test_get_best_loss(self):
        """Test getting best (minimum) loss."""
        tracker = MetricsTracker()
        
        tracker.add_round(1, accuracy=0.85, loss=0.5)
        tracker.add_round(2, accuracy=0.90, loss=0.2)
        tracker.add_round(3, accuracy=0.88, loss=0.4)
        
        best_round, best_loss = tracker.get_best("loss")
        
        assert best_round == 2
        assert best_loss == 0.2
    
    def test_get_history(self):
        """Test getting metric history."""
        tracker = MetricsTracker()
        
        tracker.add_round(1, accuracy=0.80, loss=0.6)
        tracker.add_round(2, accuracy=0.85, loss=0.4)
        tracker.add_round(3, accuracy=0.90, loss=0.3)
        
        history = tracker.get_history("accuracy")
        
        assert history == [0.80, 0.85, 0.90]
    
    def test_save_and_load(self, temp_dir):
        """Test saving and loading metrics."""
        tracker = MetricsTracker(experiment_name="test")
        tracker.add_round(1, accuracy=0.85, loss=0.5)
        tracker.add_round(2, accuracy=0.90, loss=0.3)
        
        path = temp_dir / "metrics.json"
        tracker.save(str(path))
        
        loaded = MetricsTracker.load(str(path))
        
        assert len(loaded.rounds) == 2
        assert loaded.rounds[0].accuracy == 0.85
    
    def test_get_summary(self):
        """Test summary generation."""
        tracker = MetricsTracker(experiment_name="test")
        tracker.add_round(1, accuracy=0.80, loss=0.5)
        tracker.add_round(2, accuracy=0.90, loss=0.3)
        
        summary = tracker.get_summary()
        
        assert summary["total_rounds"] == 2
        assert summary["best_accuracy"] == 0.90
        assert summary["final_accuracy"] == 0.90


class TestComputeAccuracy:
    """Tests for compute_accuracy function."""
    
    def test_compute_accuracy(self, device):
        """Test accuracy computation."""
        # Create simple model and data
        model = nn.Linear(10, 5)
        X = torch.randn(20, 10)
        y = torch.randint(0, 5, (20,))
        
        from torch.utils.data import TensorDataset, DataLoader
        loader = DataLoader(TensorDataset(X, y), batch_size=10)
        
        accuracy, loss = compute_accuracy(model, loader, device)
        
        assert 0.0 <= accuracy <= 1.0
        assert loss >= 0.0


class TestAggregateWeightedAverage:
    """Tests for weighted average aggregation."""
    
    def test_basic_aggregation(self):
        """Test basic weighted averaging."""
        metrics = [
            (100, {"accuracy": 0.8, "loss": 0.5}),
            (200, {"accuracy": 0.9, "loss": 0.3}),
        ]
        
        result = aggregate_weighted_average(metrics)
        
        # Expected: (100*0.8 + 200*0.9) / 300 = 0.867
        assert abs(result["accuracy"] - 0.867) < 0.01
    
    def test_empty_metrics(self):
        """Test with empty input."""
        result = aggregate_weighted_average([])
        assert result == {}


class TestCheckpointer:
    """Tests for Checkpointer."""
    
    def test_save_best(self, temp_dir):
        """Test saving best checkpoint."""
        model = nn.Linear(10, 5)
        checkpointer = Checkpointer(
            checkpoint_dir=str(temp_dir),
            experiment_name="test",
            metric="accuracy",
            mode="max",
        )
        
        # First save
        is_best = checkpointer.step(
            model=model,
            epoch=1,
            metrics={"accuracy": 0.8, "loss": 0.5},
        )
        assert is_best
        
        # Better save
        is_best = checkpointer.step(
            model=model,
            epoch=2,
            metrics={"accuracy": 0.9, "loss": 0.3},
        )
        assert is_best
        
        # Worse save
        is_best = checkpointer.step(
            model=model,
            epoch=3,
            metrics={"accuracy": 0.85, "loss": 0.4},
        )
        assert not is_best
        
        # Check best info
        best_info = checkpointer.get_best_info()
        assert best_info["best_value"] == 0.9
        assert best_info["best_epoch"] == 2
    
    def test_load_best(self, temp_dir):
        """Test loading best checkpoint."""
        model = nn.Linear(10, 5)
        checkpointer = Checkpointer(
            checkpoint_dir=str(temp_dir),
            experiment_name="test",
        )
        
        # Save some checkpoints
        checkpointer.step(model, 1, {"accuracy": 0.8})
        checkpointer.step(model, 2, {"accuracy": 0.9})
        
        # Create new model and load
        new_model = nn.Linear(10, 5)
        checkpoint = checkpointer.load_best(new_model)
        
        assert checkpoint["epoch"] == 2


class TestSaveLoadCheckpoint:
    """Tests for save/load checkpoint functions."""
    
    def test_save_and_load(self, temp_dir):
        """Test basic save and load."""
        model = nn.Linear(10, 5)
        path = temp_dir / "checkpoint.pt"
        
        save_checkpoint(model, str(path), epoch=10)
        
        new_model = nn.Linear(10, 5)
        checkpoint = load_checkpoint(str(path), new_model)
        
        assert checkpoint["epoch"] == 10


class TestReproducibility:
    """Tests for reproducibility utilities."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        val1 = torch.randn(10).tolist()
        
        set_seed(42)
        val2 = torch.randn(10).tolist()
        
        assert val1 == val2
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_seed_context(self):
        """Test SeedContext."""
        # Get some random values
        set_seed(100)
        outside1 = torch.randn(5).tolist()
        
        with SeedContext(42):
            inside1 = torch.randn(5).tolist()
        
        with SeedContext(42):
            inside2 = torch.randn(5).tolist()
        
        # Inside values should match
        assert inside1 == inside2
