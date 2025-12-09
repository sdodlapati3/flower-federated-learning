"""
Checkpointing Utilities
========================

Save and restore model checkpoints during training.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import numpy as np


def save_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a model checkpoint.
    
    Args:
        model: Model to save
        path: Path to save checkpoint
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        metrics: Optional metrics dictionary
        extra: Optional extra data
    
    Example:
        >>> save_checkpoint(
        ...     model=net,
        ...     path="checkpoints/model_epoch_10.pt",
        ...     epoch=10,
        ...     metrics={"accuracy": 0.95},
        ... )
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint["epoch"] = epoch
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    if extra is not None:
        checkpoint["extra"] = extra
    
    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """
    Load a model checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Optional model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load tensors to
    
    Returns:
        Checkpoint dictionary
    
    Example:
        >>> checkpoint = load_checkpoint(
        ...     path="checkpoints/model_epoch_10.pt",
        ...     model=net,
        ... )
        >>> print(checkpoint["epoch"])
        10
    """
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint


class Checkpointer:
    """
    Manage checkpoints during training.
    
    Supports:
    - Saving best model based on metric
    - Periodic checkpoint saving
    - Checkpoint rotation (keep N most recent)
    
    Example:
        >>> checkpointer = Checkpointer(
        ...     checkpoint_dir="checkpoints",
        ...     metric="accuracy",
        ...     mode="max",
        ... )
        >>> for epoch in range(100):
        ...     # Train...
        ...     checkpointer.step(
        ...         model=net,
        ...         epoch=epoch,
        ...         metrics={"accuracy": acc, "loss": loss},
        ...     )
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        experiment_name: str = "experiment",
        metric: str = "accuracy",
        mode: str = "max",
        save_every: Optional[int] = None,
        keep_n_checkpoints: int = 5,
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name for this experiment
            metric: Metric to track for best model
            mode: "max" or "min" for metric optimization
            save_every: Save checkpoint every N epochs (None = only best)
            keep_n_checkpoints: Keep N most recent periodic checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.metric = metric
        self.mode = mode
        self.save_every = save_every
        self.keep_n_checkpoints = keep_n_checkpoints
        
        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best
        self.best_value = float('-inf') if mode == "max" else float('inf')
        self.best_epoch = 0
        
        # Track saved checkpoints for rotation
        self._saved_checkpoints = []
    
    def step(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> bool:
        """
        Step the checkpointer with current metrics.
        
        Args:
            model: Model to potentially save
            epoch: Current epoch
            metrics: Current metrics
            optimizer: Optional optimizer to save
        
        Returns:
            True if this was the best model
        """
        is_best = False
        current_value = metrics.get(self.metric, 0)
        
        # Check if best
        if self.mode == "max" and current_value > self.best_value:
            is_best = True
        elif self.mode == "min" and current_value < self.best_value:
            is_best = True
        
        if is_best:
            self.best_value = current_value
            self.best_epoch = epoch
            self._save_best(model, epoch, metrics, optimizer)
        
        # Periodic save
        if self.save_every is not None and epoch % self.save_every == 0:
            self._save_periodic(model, epoch, metrics, optimizer)
        
        return is_best
    
    def _save_best(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Save best model checkpoint."""
        path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
        save_checkpoint(
            model=model,
            path=str(path),
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            extra={"is_best": True},
        )
    
    def _save_periodic(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Save periodic checkpoint with rotation."""
        path = self.checkpoint_dir / f"{self.experiment_name}_epoch_{epoch:04d}.pt"
        save_checkpoint(
            model=model,
            path=str(path),
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
        )
        
        self._saved_checkpoints.append(path)
        
        # Rotate old checkpoints
        while len(self._saved_checkpoints) > self.keep_n_checkpoints:
            old_path = self._saved_checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """Load best checkpoint."""
        path = self.checkpoint_dir / f"{self.experiment_name}_best.pt"
        return load_checkpoint(str(path), model, optimizer)
    
    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load most recent periodic checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob(f"{self.experiment_name}_epoch_*.pt"))
        if not checkpoints:
            return None
        return load_checkpoint(str(checkpoints[-1]), model, optimizer)
    
    def get_best_info(self) -> Dict[str, Any]:
        """Get info about best checkpoint."""
        return {
            "metric": self.metric,
            "mode": self.mode,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
        }


def save_fl_round(
    model: nn.Module,
    round_num: int,
    checkpoint_dir: str,
    metrics: Optional[Dict[str, float]] = None,
) -> str:
    """
    Save FL round checkpoint.
    
    Args:
        model: Model to save
        round_num: FL round number
        checkpoint_dir: Directory for checkpoints
        metrics: Optional metrics
    
    Returns:
        Path to saved checkpoint
    """
    path = Path(checkpoint_dir) / f"round_{round_num:04d}.pt"
    save_checkpoint(model, str(path), epoch=round_num, metrics=metrics)
    return str(path)


def load_fl_round(
    model: nn.Module,
    round_num: int,
    checkpoint_dir: str,
) -> Dict[str, Any]:
    """
    Load FL round checkpoint.
    
    Args:
        model: Model to load state into
        round_num: FL round number
        checkpoint_dir: Directory for checkpoints
    
    Returns:
        Checkpoint dictionary
    """
    path = Path(checkpoint_dir) / f"round_{round_num:04d}.pt"
    return load_checkpoint(str(path), model)
