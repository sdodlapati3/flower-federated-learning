"""
Opacus Integration Utilities
============================

Helper functions for using Opacus with Flower FL.
"""

from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False


def make_private(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    noise_multiplier: float,
    max_grad_norm: float,
    target_epsilon: Optional[float] = None,
    target_delta: Optional[float] = None,
    epochs: Optional[int] = None,
    grad_sample_mode: str = "hooks",
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader]:
    """
    Make model, optimizer, and data_loader private using Opacus.
    
    Args:
        model: PyTorch model (must be Opacus-compatible)
        optimizer: PyTorch optimizer
        data_loader: Training data loader
        noise_multiplier: Noise multiplier for DP-SGD
        max_grad_norm: Max gradient norm for clipping
        target_epsilon: Target epsilon (optional, for auto noise calibration)
        target_delta: Target delta (required if target_epsilon is set)
        epochs: Number of epochs (required if target_epsilon is set)
        grad_sample_mode: Gradient computation mode ("hooks" or "ew")
    
    Returns:
        Tuple of (private_model, private_optimizer, private_data_loader)
    
    Example:
        >>> model, optimizer, loader = make_private(
        ...     model=net,
        ...     optimizer=torch.optim.SGD(net.parameters(), lr=0.01),
        ...     data_loader=train_loader,
        ...     noise_multiplier=1.0,
        ...     max_grad_norm=1.0,
        ... )
    """
    if not OPACUS_AVAILABLE:
        raise ImportError("Opacus is required. Install with: pip install opacus")
    
    # Validate and fix model if needed
    model = validate_model_for_dp(model)
    
    # Create privacy engine
    privacy_engine = PrivacyEngine()
    
    if target_epsilon is not None:
        if target_delta is None or epochs is None:
            raise ValueError("target_delta and epochs required when target_epsilon is set")
        
        # Use Opacus to calibrate noise
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            epochs=epochs,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
            grad_sample_mode=grad_sample_mode,
        )
    else:
        # Use provided noise multiplier
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            grad_sample_mode=grad_sample_mode,
        )
    
    return model, optimizer, data_loader


def validate_model_for_dp(model: nn.Module) -> nn.Module:
    """
    Validate and fix model for differential privacy.
    
    Checks for DP-incompatible layers (BatchNorm, etc.) and replaces them.
    
    Args:
        model: PyTorch model
    
    Returns:
        DP-compatible model
    """
    if not OPACUS_AVAILABLE:
        raise ImportError("Opacus is required. Install with: pip install opacus")
    
    # Check if model is valid
    errors = ModuleValidator.validate(model, strict=False)
    
    if errors:
        # Fix the model
        model = ModuleValidator.fix(model)
    
    return model


def convert_batchnorm_to_groupnorm(
    model: nn.Module,
    num_groups: int = 32,
) -> nn.Module:
    """
    Convert all BatchNorm layers to GroupNorm.
    
    GroupNorm is compatible with differential privacy training.
    
    Args:
        model: Model with BatchNorm layers
        num_groups: Number of groups for GroupNorm
    
    Returns:
        Model with GroupNorm layers
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            num_features = module.num_features
            # Adjust num_groups if necessary
            groups = min(num_groups, num_features)
            while num_features % groups != 0:
                groups -= 1
            
            setattr(model, name, nn.GroupNorm(groups, num_features))
        else:
            convert_batchnorm_to_groupnorm(module, num_groups)
    
    return model


def get_privacy_engine_state(
    model: nn.Module,
) -> Optional[Dict[str, Any]]:
    """
    Get privacy engine state from a private model.
    
    Args:
        model: Model wrapped by PrivacyEngine
    
    Returns:
        Dict with privacy state, or None if not a private model
    """
    if not OPACUS_AVAILABLE:
        return None
    
    # Check if model has _module (wrapped by Opacus)
    if hasattr(model, '_module'):
        # This is a GradSampleModule
        return {
            "is_private": True,
            "grad_sample_mode": getattr(model, 'grad_sample_mode', 'unknown'),
        }
    
    return None


def train_with_dp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
    max_physical_batch_size: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Train model with differential privacy using Opacus.
    
    Handles gradient accumulation for large logical batch sizes.
    
    Args:
        model: DP-wrapped model
        optimizer: DP-wrapped optimizer
        data_loader: DP-wrapped data loader
        epochs: Number of epochs
        device: Training device
        max_physical_batch_size: Max batch size for memory efficiency
    
    Returns:
        Tuple of (average_loss, epsilon)
    """
    if not OPACUS_AVAILABLE:
        raise ImportError("Opacus is required")
    
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        if max_physical_batch_size is not None:
            with BatchMemoryManager(
                data_loader=data_loader,
                max_physical_batch_size=max_physical_batch_size,
                optimizer=optimizer,
            ) as memory_safe_loader:
                for inputs, labels in memory_safe_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
        else:
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Get epsilon from optimizer's privacy engine
    epsilon = float('inf')
    if hasattr(optimizer, 'privacy_engine'):
        epsilon = optimizer.privacy_engine.get_epsilon(delta=1e-5)
    
    return avg_loss, epsilon


class DPTrainer:
    """
    High-level trainer for differential privacy training.
    
    Example:
        >>> trainer = DPTrainer(
        ...     model=net,
        ...     noise_multiplier=1.0,
        ...     max_grad_norm=1.0,
        ...     target_delta=1e-5,
        ... )
        >>> trainer.fit(train_loader, epochs=5)
        >>> epsilon = trainer.get_epsilon()
    """
    
    def __init__(
        self,
        model: nn.Module,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        target_delta: float = 1e-5,
        lr: float = 0.01,
        device: Union[str, torch.device] = "auto",
    ):
        if not OPACUS_AVAILABLE:
            raise ImportError("Opacus is required")
        
        self.model = validate_model_for_dp(model)
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_delta = target_delta
        self.lr = lr
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.privacy_engine: Optional[PrivacyEngine] = None
        self._is_initialized = False
    
    def _init_privacy(self, data_loader: torch.utils.data.DataLoader) -> None:
        """Initialize privacy engine with data loader."""
        if self._is_initialized:
            return
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.privacy_engine = PrivacyEngine()
        
        self.model, self.optimizer, self.data_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )
        
        self._is_initialized = True
    
    def fit(
        self,
        data_loader: torch.utils.data.DataLoader,
        epochs: int = 1,
    ) -> Dict[str, float]:
        """Train for specified epochs."""
        self._init_privacy(data_loader)
        
        self.model.to(self.device)
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(epochs):
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += inputs.size(0)
        
        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "epsilon": self.get_epsilon(),
        }
    
    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        if self.privacy_engine is None:
            return 0.0
        return self.privacy_engine.get_epsilon(delta=self.target_delta)
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get underlying model state dict."""
        if hasattr(self.model, '_module'):
            return self.model._module.state_dict()
        return self.model.state_dict()
