"""
FedAvg Strategy
===============

Federated Averaging implementation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import flwr as fl
    from flwr.common import (
        Parameters,
        Scalar,
        FitRes,
        EvaluateRes,
        NDArrays,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
        FitIns,
        EvaluateIns,
        GetParametersIns,
    )
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import Strategy
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False
    # Dummy classes for type hints
    class Strategy:
        pass


def weighted_average(
    metrics: List[Tuple[int, Dict[str, float]]],
) -> Dict[str, float]:
    """
    Compute weighted average of metrics.
    
    Args:
        metrics: List of (num_samples, metrics_dict) tuples
    
    Returns:
        Aggregated metrics
    
    Example:
        >>> metrics = [
        ...     (100, {"accuracy": 0.8, "loss": 0.5}),
        ...     (200, {"accuracy": 0.9, "loss": 0.3}),
        ... ]
        >>> weighted_average(metrics)
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
        weighted_sum = sum(n * m.get(key, 0) for n, m in metrics)
        result[key] = weighted_sum / total_samples
    
    return result


class FedAvgStrategy(Strategy):
    """
    Federated Averaging Strategy.
    
    Implements the FedAvg algorithm from McMahan et al. (2017).
    
    Example:
        >>> strategy = FedAvgStrategy(
        ...     fraction_fit=0.1,
        ...     fraction_evaluate=0.1,
        ...     min_fit_clients=2,
        ... )
        >>> fl.server.start_server(strategy=strategy)
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
    ):
        """
        Args:
            fraction_fit: Fraction of clients for training
            fraction_evaluate: Fraction of clients for evaluation
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum available clients to start
            initial_parameters: Initial global parameters
            fit_metrics_aggregation_fn: Function to aggregate fit metrics
            evaluate_metrics_aggregation_fn: Function to aggregate evaluate metrics
        """
        if not FLWR_AVAILABLE:
            raise ImportError("Flower is required. Install with: pip install flwr")
        
        super().__init__()
        
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn or weighted_average
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn or weighted_average
        
        self._current_parameters: Optional[Parameters] = initial_parameters
    
    def initialize_parameters(
        self, client_manager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure clients for training."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )
        
        # Create fit instructions
        fit_ins = FitIns(parameters, {"round": server_round})
        
        return [(client, fit_ins) for client in clients]
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure clients for evaluation."""
        sample_size, min_num_clients = self.num_evaluate_clients(
            client_manager.num_available()
        )
        
        if sample_size == 0:
            return []
        
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )
        
        evaluate_ins = EvaluateIns(parameters, {"round": server_round})
        
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""
        if not results:
            return None, {}
        
        # Convert results to (num_samples, parameters) format
        weights_results = [
            (fit_res.num_examples, parameters_to_ndarrays(fit_res.parameters))
            for _, fit_res in results
        ]
        
        # Aggregate parameters
        aggregated = self._aggregate_weights(weights_results)
        parameters = ndarrays_to_parameters(aggregated)
        
        # Aggregate metrics
        metrics_list = [
            (fit_res.num_examples, fit_res.metrics)
            for _, fit_res in results
            if fit_res.metrics
        ]
        aggregated_metrics = self.fit_metrics_aggregation_fn(metrics_list)
        
        # Add failure count
        aggregated_metrics["failures"] = len(failures)
        
        return parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}
        
        # Aggregate loss
        loss_results = [
            (eval_res.num_examples, eval_res.loss)
            for _, eval_res in results
        ]
        total_samples = sum(n for n, _ in loss_results)
        aggregated_loss = (
            sum(n * loss for n, loss in loss_results) / total_samples
            if total_samples > 0 else 0.0
        )
        
        # Aggregate metrics
        metrics_list = [
            (eval_res.num_examples, eval_res.metrics)
            for _, eval_res in results
            if eval_res.metrics
        ]
        aggregated_metrics = self.evaluate_metrics_aggregation_fn(metrics_list)
        
        return aggregated_loss, aggregated_metrics
    
    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation (optional)."""
        return None
    
    def num_fit_clients(self, num_available: int) -> Tuple[int, int]:
        """Return sample size and minimum clients for fit."""
        sample_size = max(int(num_available * self.fraction_fit), self.min_fit_clients)
        return sample_size, self.min_fit_clients
    
    def num_evaluate_clients(self, num_available: int) -> Tuple[int, int]:
        """Return sample size and minimum clients for evaluate."""
        sample_size = max(int(num_available * self.fraction_evaluate), self.min_evaluate_clients)
        return sample_size, self.min_evaluate_clients
    
    def _aggregate_weights(
        self,
        results: List[Tuple[int, List[np.ndarray]]],
    ) -> List[np.ndarray]:
        """Weighted average of model parameters."""
        total_samples = sum(n for n, _ in results)
        
        if total_samples == 0:
            return results[0][1] if results else []
        
        # Initialize with zeros
        aggregated = [np.zeros_like(p) for p in results[0][1]]
        
        # Weighted sum
        for num_samples, params in results:
            weight = num_samples / total_samples
            for i, param in enumerate(params):
                aggregated[i] += weight * param
        
        return aggregated
