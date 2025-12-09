"""
Reproducibility Utilities
==========================

Utilities for ensuring reproducible experiments.
"""

from typing import Optional, Union
import random
import os

import torch
import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility.
    
    Sets seed for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms (slower)
    
    Example:
        >>> set_seed(42)
        >>> # Now training will be reproducible
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # PyTorch 1.8+
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                pass


def get_device(
    device: Optional[Union[str, torch.device]] = None,
    prefer_cuda: bool = True,
) -> torch.device:
    """
    Get the best available device.
    
    Args:
        device: Specific device to use (overrides auto-detection)
        prefer_cuda: If True, prefer CUDA over MPS
    
    Returns:
        torch.device object
    
    Example:
        >>> device = get_device()
        >>> model.to(device)
    """
    if device is not None:
        if isinstance(device, str):
            if device == "auto":
                device = None
            else:
                return torch.device(device)
        else:
            return device
    
    # Auto-detect
    if torch.cuda.is_available() and prefer_cuda:
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_num_workers() -> int:
    """
    Get recommended number of DataLoader workers.
    
    Returns:
        Number of workers based on CPU count
    """
    try:
        num_cpus = os.cpu_count() or 1
        # Use at most 4 workers or half the CPUs
        return min(4, max(1, num_cpus // 2))
    except Exception:
        return 0


def ensure_reproducibility(
    seed: int = 42,
    deterministic: bool = True,
) -> torch.device:
    """
    Full reproducibility setup.
    
    Args:
        seed: Random seed
        deterministic: Use deterministic algorithms
    
    Returns:
        Best available device
    
    Example:
        >>> device = ensure_reproducibility(seed=42)
        >>> model = MyModel().to(device)
    """
    set_seed(seed, deterministic=deterministic)
    return get_device()


class SeedContext:
    """
    Context manager for temporary seed setting.
    
    Useful for ensuring specific operations are reproducible
    while allowing the rest of the code to be stochastic.
    
    Example:
        >>> with SeedContext(42):
        ...     # This will always produce the same result
        ...     x = torch.randn(10)
        >>> # Back to normal randomness
    """
    
    def __init__(self, seed: int):
        self.seed = seed
        self._saved_state = None
    
    def __enter__(self):
        # Save current state
        self._saved_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            self._saved_state["cuda"] = torch.cuda.get_rng_state_all()
        
        # Set new seed
        set_seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore state
        random.setstate(self._saved_state["python"])
        np.random.set_state(self._saved_state["numpy"])
        torch.set_rng_state(self._saved_state["torch"])
        if "cuda" in self._saved_state:
            torch.cuda.set_rng_state_all(self._saved_state["cuda"])
        return False


def print_environment_info() -> dict:
    """
    Print and return environment information.
    
    Useful for logging experiment environment for reproducibility.
    
    Returns:
        Dictionary with environment info
    """
    import sys
    import platform
    
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["gpu_names"] = [
            torch.cuda.get_device_name(i) 
            for i in range(torch.cuda.device_count())
        ]
    
    # Try to get package versions
    try:
        import flwr
        info["flower_version"] = flwr.__version__
    except ImportError:
        pass
    
    try:
        import opacus
        info["opacus_version"] = opacus.__version__
    except ImportError:
        pass
    
    try:
        import numpy
        info["numpy_version"] = numpy.__version__
    except ImportError:
        pass
    
    return info
