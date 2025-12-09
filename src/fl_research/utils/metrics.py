"""
Metrics Tracking
=================

Utilities for tracking, aggregating, and analyzing FL metrics.
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

import torch
import numpy as np


@dataclass
class RoundMetrics:
    """Metrics for a single FL round."""
    round_num: int
    accuracy: float
    loss: float
    num_clients: int = 0
    num_samples: int = 0
    epsilon: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)


class MetricsTracker:
    """
    Track and aggregate metrics across FL rounds.
    
    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.add_round(1, accuracy=0.85, loss=0.5)
        >>> tracker.add_round(2, accuracy=0.87, loss=0.4)
        >>> tracker.get_best("accuracy")
        (2, 0.87)
    """
    
    def __init__(self, experiment_name: str = "experiment"):
        self.experiment_name = experiment_name
        self.rounds: List[RoundMetrics] = []
        self.client_metrics: Dict[int, List[Dict[str, float]]] = defaultdict(list)
        self.start_time = datetime.now()
    
    def add_round(
        self,
        round_num: int,
        accuracy: float,
        loss: float,
        num_clients: int = 0,
        num_samples: int = 0,
        epsilon: Optional[float] = None,
        **extra
    ) -> None:
        """Add metrics for a round."""
        self.rounds.append(RoundMetrics(
            round_num=round_num,
            accuracy=accuracy,
            loss=loss,
            num_clients=num_clients,
            num_samples=num_samples,
            epsilon=epsilon,
            extra=extra,
        ))
    
    def add_client_metrics(
        self,
        round_num: int,
        client_id: int,
        metrics: Dict[str, float]
    ) -> None:
        """Add metrics for a specific client."""
        entry = {"round": round_num, **metrics}
        self.client_metrics[client_id].append(entry)
    
    def get_round(self, round_num: int) -> Optional[RoundMetrics]:
        """Get metrics for a specific round."""
        for r in self.rounds:
            if r.round_num == round_num:
                return r
        return None
    
    def get_best(self, metric: str = "accuracy") -> Tuple[int, float]:
        """Get round number and value of best metric."""
        if not self.rounds:
            return (0, 0.0)
        
        if metric == "accuracy":
            best = max(self.rounds, key=lambda r: r.accuracy)
            return (best.round_num, best.accuracy)
        elif metric == "loss":
            best = min(self.rounds, key=lambda r: r.loss)
            return (best.round_num, best.loss)
        else:
            # Check extra fields
            rounds_with_metric = [r for r in self.rounds if metric in r.extra]
            if rounds_with_metric:
                best = max(rounds_with_metric, key=lambda r: r.extra[metric])
                return (best.round_num, best.extra[metric])
            return (0, 0.0)
    
    def get_final(self) -> Optional[RoundMetrics]:
        """Get final round metrics."""
        return self.rounds[-1] if self.rounds else None
    
    def get_history(self, metric: str = "accuracy") -> List[float]:
        """Get history of a metric across all rounds."""
        if metric == "accuracy":
            return [r.accuracy for r in self.rounds]
        elif metric == "loss":
            return [r.loss for r in self.rounds]
        elif metric == "epsilon":
            return [r.epsilon for r in self.rounds if r.epsilon is not None]
        else:
            return [r.extra.get(metric, 0) for r in self.rounds if metric in r.extra]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.rounds:
            return {}
        
        accuracies = self.get_history("accuracy")
        losses = self.get_history("loss")
        
        summary = {
            "experiment_name": self.experiment_name,
            "total_rounds": len(self.rounds),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "final_accuracy": accuracies[-1] if accuracies else None,
            "best_accuracy": max(accuracies) if accuracies else None,
            "best_accuracy_round": self.get_best("accuracy")[0],
            "final_loss": losses[-1] if losses else None,
            "avg_accuracy": np.mean(accuracies) if accuracies else None,
        }
        
        epsilons = self.get_history("epsilon")
        if epsilons:
            summary["final_epsilon"] = epsilons[-1]
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "rounds": [
                {
                    "round": r.round_num,
                    "accuracy": r.accuracy,
                    "loss": r.loss,
                    "num_clients": r.num_clients,
                    "num_samples": r.num_samples,
                    "epsilon": r.epsilon,
                    **r.extra,
                }
                for r in self.rounds
            ],
            "summary": self.get_summary(),
        }
    
    def save(self, path: str) -> None:
        """Save metrics to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
    
    @classmethod
    def load(cls, path: str) -> "MetricsTracker":
        """Load metrics from JSON file."""
        data = json.loads(Path(path).read_text())
        tracker = cls(data.get("experiment_name", "experiment"))
        
        for r in data.get("rounds", []):
            tracker.add_round(
                round_num=r["round"],
                accuracy=r["accuracy"],
                loss=r["loss"],
                num_clients=r.get("num_clients", 0),
                num_samples=r.get("num_samples", 0),
                epsilon=r.get("epsilon"),
            )
        
        return tracker


def compute_accuracy(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Compute accuracy and loss on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to use
    
    Returns:
        Tuple of (accuracy, average_loss)
    """
    model.eval()
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    
    return accuracy, avg_loss


def aggregate_weighted_average(
    metrics: List[Tuple[int, Dict[str, float]]]
) -> Dict[str, float]:
    """
    Weighted average of metrics from multiple clients.
    
    Args:
        metrics: List of (num_samples, metrics_dict) tuples
    
    Returns:
        Aggregated metrics
    
    Example:
        >>> metrics = [
        ...     (100, {"accuracy": 0.8, "loss": 0.5}),
        ...     (200, {"accuracy": 0.9, "loss": 0.3}),
        ... ]
        >>> aggregate_weighted_average(metrics)
        {"accuracy": 0.867, "loss": 0.367}
    """
    if not metrics:
        return {}
    
    total_samples = sum(n for n, _ in metrics)
    if total_samples == 0:
        return {}
    
    result = {}
    all_keys = set()
    for _, m in metrics:
        all_keys.update(m.keys())
    
    for key in all_keys:
        weighted_sum = sum(
            n * m.get(key, 0) for n, m in metrics
        )
        result[key] = weighted_sum / total_samples
    
    return result
