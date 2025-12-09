"""
Logging Utilities
==================

Structured logging for FL experiments with support for
console, file, and optional W&B integration.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


# Global logger cache
_loggers: Dict[str, logging.Logger] = {}


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    experiment_name: str = "experiment",
    use_json: bool = False,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Set up logging for an experiment.
    
    Args:
        log_dir: Directory for log files (None for console only)
        log_level: Logging level
        experiment_name: Name for the experiment/logger
        use_json: Use JSON format for file logs
        use_colors: Use colored output for console
    
    Returns:
        Configured logger
    
    Example:
        >>> logger = setup_logging(
        ...     log_dir="./logs",
        ...     experiment_name="dp_fedavg",
        ...     log_level="DEBUG"
        ... )
        >>> logger.info("Starting experiment", extra={"round": 1})
    """
    logger = logging.getLogger(experiment_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []  # Clear existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    if use_colors and sys.stdout.isatty():
        console_format = "%(asctime)s | %(levelname)s | %(message)s"
        console_handler.setFormatter(ColoredFormatter(console_format, datefmt="%H:%M:%S"))
    else:
        console_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        console_handler.setFormatter(logging.Formatter(console_format, datefmt="%Y-%m-%d %H:%M:%S"))
    
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{experiment_name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        if use_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            file_handler.setFormatter(logging.Formatter(file_format))
        
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")
    
    _loggers[experiment_name] = logger
    return logger


def get_logger(name: str = "fl_research") -> logging.Logger:
    """
    Get or create a logger.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    # Create a basic logger
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    _loggers[name] = logger
    return logger


class ExperimentLogger:
    """
    High-level experiment logger with structured logging.
    
    Example:
        >>> exp_logger = ExperimentLogger("dp_fedavg", log_dir="./logs")
        >>> exp_logger.log_round(1, {"accuracy": 0.85, "loss": 0.5})
        >>> exp_logger.log_privacy(epsilon=4.2, delta=1e-5)
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.logger = setup_logging(log_dir, experiment_name=experiment_name)
        self.metrics_history: list = []
        
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                if wandb_project:
                    wandb.init(project=wandb_project, name=experiment_name)
            except ImportError:
                self.logger.warning("W&B not installed, skipping wandb logging")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        if self.wandb:
            self.wandb.config.update(config)
    
    def log_round(
        self,
        round_num: int,
        metrics: Dict[str, float],
        phase: str = "aggregated"
    ) -> None:
        """Log metrics for a round."""
        entry = {"round": round_num, "phase": phase, **metrics}
        self.metrics_history.append(entry)
        
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Round {round_num} [{phase}]: {metrics_str}")
        
        if self.wandb:
            self.wandb.log(metrics, step=round_num)
    
    def log_privacy(
        self,
        epsilon: float,
        delta: float,
        round_num: Optional[int] = None
    ) -> None:
        """Log privacy budget."""
        msg = f"Privacy: ε={epsilon:.4f}, δ={delta:.2e}"
        if round_num:
            msg = f"Round {round_num} {msg}"
        self.logger.info(msg)
        
        if self.wandb and round_num:
            self.wandb.log({"epsilon": epsilon, "delta": delta}, step=round_num)
    
    def log_client(
        self,
        client_id: int,
        round_num: int,
        metrics: Dict[str, float]
    ) -> None:
        """Log client-level metrics."""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.debug(f"Round {round_num}, Client {client_id}: {metrics_str}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all logged metrics."""
        if not self.metrics_history:
            return {}
        
        final = self.metrics_history[-1]
        accuracies = [m.get("accuracy", 0) for m in self.metrics_history if "accuracy" in m]
        
        return {
            "total_rounds": len(self.metrics_history),
            "final_accuracy": final.get("accuracy"),
            "best_accuracy": max(accuracies) if accuracies else None,
            "final_loss": final.get("loss"),
        }
    
    def close(self) -> None:
        """Close the logger and any integrations."""
        summary = self.get_summary()
        self.logger.info(f"Experiment complete. Summary: {summary}")
        
        if self.wandb:
            self.wandb.finish()
