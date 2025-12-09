"""
FedProx Strategy
================

Federated Proximal optimization implementation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

try:
    import flwr as fl
    from flwr.common import (
        Parameters,
        Scalar,
        FitRes,
        NDArrays,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
        FitIns,
    )
    from flwr.server.client_proxy import ClientProxy
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False

from fl_research.strategies.fedavg import FedAvgStrategy, weighted_average


class FedProxStrategy(FedAvgStrategy):
    """
    Federated Proximal (FedProx) Strategy.
    
    Implements FedProx from Li et al. (2020) which adds a proximal term
    to handle heterogeneous data across clients.
    
    The proximal term penalizes local updates that deviate too far from
    the global model:
    
        L_prox = μ/2 * ||w - w_global||²
    
    Example:
        >>> strategy = FedProxStrategy(
        ...     proximal_mu=0.1,
        ...     fraction_fit=0.1,
        ...     min_fit_clients=2,
        ... )
    """
    
    def __init__(
        self,
        proximal_mu: float = 0.1,
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
            proximal_mu: Proximal regularization parameter (μ)
            fraction_fit: Fraction of clients for training
            fraction_evaluate: Fraction of clients for evaluation
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum available clients to start
            initial_parameters: Initial global parameters
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        
        self.proximal_mu = proximal_mu
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure clients for training with proximal term."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )
        
        # Include proximal_mu in config for clients
        config = {
            "round": server_round,
            "proximal_mu": self.proximal_mu,
        }
        fit_ins = FitIns(parameters, config)
        
        return [(client, fit_ins) for client in clients]


class FedProxClient:
    """
    Client implementation for FedProx.
    
    Adds proximal regularization to the local training objective.
    
    Example:
        >>> client = FedProxClient(
        ...     model=net,
        ...     train_loader=train_loader,
        ...     proximal_mu=0.1,
        ... )
        >>> loss = client.train(epochs=5)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        proximal_mu: float = 0.1,
        lr: float = 0.01,
        device: Union[str, torch.device] = "auto",
    ):
        self.model = model
        self.train_loader = train_loader
        self.proximal_mu = proximal_mu
        self.lr = lr
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.global_params: Optional[List[torch.Tensor]] = None
    
    def set_global_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set global parameters for proximal term."""
        # Convert to tensors and store
        self.global_params = [
            torch.tensor(p, device=self.device)
            for p in parameters
        ]
        
        # Also set as model parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def compute_proximal_term(self) -> torch.Tensor:
        """Compute proximal regularization term."""
        if self.global_params is None:
            return torch.tensor(0.0, device=self.device)
        
        proximal_term = torch.tensor(0.0, device=self.device)
        
        for local_param, global_param in zip(
            self.model.parameters(),
            self.global_params,
        ):
            proximal_term += torch.sum((local_param - global_param) ** 2)
        
        return (self.proximal_mu / 2.0) * proximal_term
    
    def train(self, epochs: int = 1) -> Dict[str, float]:
        """
        Train with FedProx objective.
        
        Loss = CrossEntropy + (μ/2) * ||w - w_global||²
        """
        self.model.to(self.device)
        self.model.train()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(epochs):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Add proximal term
                proximal_term = self.compute_proximal_term()
                total_loss_batch = loss + proximal_term
                
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += inputs.size(0)
        
        return {
            "loss": total_loss / total_samples if total_samples > 0 else 0.0,
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
            "num_samples": total_samples,
        }
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays."""
        return [
            val.cpu().numpy()
            for val in self.model.state_dict().values()
        ]


def create_fedprox_client_fn(
    model_fn,
    train_loaders: Dict[int, torch.utils.data.DataLoader],
    proximal_mu: float = 0.1,
    local_epochs: int = 1,
    lr: float = 0.01,
):
    """
    Factory function to create FedProx client instances.
    
    Args:
        model_fn: Function that returns a new model instance
        train_loaders: Dict mapping client_id to DataLoader
        proximal_mu: Proximal regularization parameter
        local_epochs: Number of local training epochs
        lr: Learning rate
    
    Returns:
        Function that creates clients by ID
    
    Example:
        >>> client_fn = create_fedprox_client_fn(
        ...     model_fn=lambda: SimpleCNN(),
        ...     train_loaders={0: loader1, 1: loader2},
        ...     proximal_mu=0.1,
        ... )
        >>> client = client_fn("0")
    """
    def client_fn(cid: str):
        client_id = int(cid)
        model = model_fn()
        train_loader = train_loaders[client_id]
        
        return FedProxFlowerClient(
            model=model,
            train_loader=train_loader,
            proximal_mu=proximal_mu,
            local_epochs=local_epochs,
            lr=lr,
        )
    
    return client_fn


class FedProxFlowerClient:
    """
    Flower-compatible FedProx client.
    
    Can be used directly with fl.client.start_client or simulation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        proximal_mu: float = 0.1,
        local_epochs: int = 1,
        lr: float = 0.01,
    ):
        self.client = FedProxClient(
            model=model,
            train_loader=train_loader,
            proximal_mu=proximal_mu,
            lr=lr,
        )
        self.local_epochs = local_epochs
    
    def get_parameters(self, config=None):
        """Get model parameters."""
        if FLWR_AVAILABLE:
            return ndarrays_to_parameters(self.client.get_parameters())
        return self.client.get_parameters()
    
    def fit(self, parameters, config):
        """Train model."""
        # Get proximal_mu from config if provided
        if "proximal_mu" in config:
            self.client.proximal_mu = config["proximal_mu"]
        
        # Set global parameters
        if FLWR_AVAILABLE:
            params = parameters_to_ndarrays(parameters)
        else:
            params = parameters
        
        self.client.set_global_parameters(params)
        
        # Train
        results = self.client.train(epochs=self.local_epochs)
        
        # Return updated parameters and metrics
        new_params = self.client.get_parameters()
        
        if FLWR_AVAILABLE:
            return ndarrays_to_parameters(new_params), results["num_samples"], results
        return new_params, results["num_samples"], results
    
    def evaluate(self, parameters, config):
        """Evaluate model."""
        # Set parameters
        if FLWR_AVAILABLE:
            params = parameters_to_ndarrays(parameters)
        else:
            params = parameters
        
        self.client.set_global_parameters(params)
        
        # Evaluate
        self.client.model.eval()
        device = self.client.device
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in self.client.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.client.model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += inputs.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, total_samples, {"accuracy": accuracy}
