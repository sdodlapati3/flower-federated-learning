"""
Configuration Management
=========================

Flexible configuration system for FL experiments.
Supports YAML, TOML, and Python dict configurations.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import os


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "cifar10_cnn"
    num_classes: int = 10
    dp_compatible: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "cifar10"
    data_dir: str = "./data"
    batch_size: int = 32
    num_workers: int = 0
    
    # Partitioning
    partition_strategy: str = "dirichlet"
    partition_alpha: float = 0.5
    num_clients: int = 10


@dataclass
class PrivacyConfig:
    """Differential privacy configuration."""
    enabled: bool = False
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: Optional[float] = None  # Auto-calculated if None


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_rounds: int = 50
    clients_per_round: int = 5
    local_epochs: int = 1
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    
    # Strategy
    strategy: str = "fedavg"
    fedprox_mu: float = 0.01  # For FedProx


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "./logs"
    log_level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = "fl-research"
    wandb_entity: Optional[str] = None
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10


@dataclass
class Config:
    """
    Main configuration class for FL experiments.
    
    Example:
        >>> config = Config(
        ...     experiment_name="dp_fedavg_eps4",
        ...     model=ModelConfig(name="cifar10_cnn"),
        ...     data=DataConfig(partition_alpha=0.5),
        ...     privacy=PrivacyConfig(enabled=True, target_epsilon=4.0),
        ... )
        >>> config.to_dict()
    """
    experiment_name: str = "fl_experiment"
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, path: Optional[str] = None) -> str:
        """Convert to JSON string, optionally save to file."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(json_str)
        return json_str
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        return cls(
            experiment_name=d.get("experiment_name", "fl_experiment"),
            seed=d.get("seed", 42),
            device=d.get("device", "auto"),
            model=ModelConfig(**d.get("model", {})),
            data=DataConfig(**d.get("data", {})),
            privacy=PrivacyConfig(**d.get("privacy", {})),
            training=TrainingConfig(**d.get("training", {})),
            logging=LoggingConfig(**d.get("logging", {})),
        )
    
    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load Config from JSON file."""
        d = json.loads(Path(path).read_text())
        return cls.from_dict(d)


def load_config(path: str) -> Config:
    """
    Load configuration from file.
    
    Supports:
        - .json files
        - .yaml/.yml files (requires PyYAML)
        - .toml files (requires tomli/tomllib)
    
    Args:
        path: Path to configuration file
    
    Returns:
        Config object
    """
    path = Path(path)
    suffix = path.suffix.lower()
    
    if suffix == ".json":
        return Config.from_json(str(path))
    
    elif suffix in [".yaml", ".yml"]:
        try:
            import yaml
            with open(path) as f:
                d = yaml.safe_load(f)
            return Config.from_dict(d)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files: pip install pyyaml")
    
    elif suffix == ".toml":
        try:
            try:
                import tomllib  # Python 3.11+
            except ImportError:
                import tomli as tomllib  # Fallback
            with open(path, "rb") as f:
                d = tomllib.load(f)
            return Config.from_dict(d)
        except ImportError:
            raise ImportError("tomli required for TOML config files: pip install tomli")
    
    else:
        raise ValueError(f"Unsupported config file format: {suffix}")


def save_config(config: Config, path: str) -> None:
    """
    Save configuration to file.
    
    Supports:
        - .json files
        - .yaml/.yml files (requires PyYAML)
        - .toml files (requires toml)
    
    Args:
        config: Config object to save
        path: Path to save configuration file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    
    data = config.to_dict()
    
    if suffix == ".json":
        path.write_text(json.dumps(data, indent=2))
    
    elif suffix in [".yaml", ".yml"]:
        try:
            import yaml
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files: pip install pyyaml")
    
    elif suffix == ".toml":
        try:
            import toml
            with open(path, 'w') as f:
                toml.dump(data, f)
        except ImportError:
            raise ImportError("toml required for TOML config files: pip install toml")
    
    else:
        raise ValueError(f"Unsupported config file format: {suffix}")


def create_experiment_config(
    name: str,
    dataset: str = "cifar10",
    num_clients: int = 10,
    num_rounds: int = 50,
    partition_strategy: str = "dirichlet",
    partition_alpha: float = 0.5,
    dp_enabled: bool = False,
    target_epsilon: float = 8.0,
    strategy: str = "fedavg",
    seed: int = 42,
) -> Config:
    """
    Factory function to create common experiment configurations.
    
    Args:
        name: Experiment name
        dataset: Dataset name
        num_clients: Number of FL clients
        num_rounds: Number of FL rounds
        partition_strategy: Data partitioning strategy
        partition_alpha: Dirichlet alpha (if using dirichlet)
        dp_enabled: Enable differential privacy
        target_epsilon: Target epsilon for DP
        strategy: FL strategy ('fedavg', 'fedprox', 'scaffold')
        seed: Random seed
    
    Returns:
        Configured Config object
    """
    return Config(
        experiment_name=name,
        seed=seed,
        model=ModelConfig(
            name=f"{dataset}_cnn",
            dp_compatible=dp_enabled,
        ),
        data=DataConfig(
            dataset=dataset,
            num_clients=num_clients,
            partition_strategy=partition_strategy,
            partition_alpha=partition_alpha,
        ),
        privacy=PrivacyConfig(
            enabled=dp_enabled,
            target_epsilon=target_epsilon,
        ),
        training=TrainingConfig(
            num_rounds=num_rounds,
            strategy=strategy,
        ),
    )
