"""
SCAFFOLD Strategy
=================

SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.
"""

from typing import Dict, List, Optional, Tuple, Union
from copy import deepcopy
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


class SCAFFOLDStrategy(FedAvgStrategy):
    """
    SCAFFOLD Strategy for variance reduction in FL.
    
    Implements SCAFFOLD from Karimireddy et al. (2020) which uses control
    variates to correct for client drift in heterogeneous settings.
    
    Key idea: Maintain server control variate c and client control variates c_i
    that track the difference between local and global gradients.
    
    Example:
        >>> strategy = SCAFFOLDStrategy(
        ...     server_lr=1.0,
        ...     fraction_fit=0.1,
        ...     min_fit_clients=2,
        ... )
    """
    
    def __init__(
        self,
        server_lr: float = 1.0,
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
            server_lr: Server learning rate for control variate update
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
        
        self.server_lr = server_lr
        
        # Server control variate (initialized later)
        self._server_control: Optional[List[np.ndarray]] = None
        self._num_clients = 0
    
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
        """Configure clients for training with control variates."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )
        
        # Convert server control to serializable format
        server_control_bytes = None
        if self._server_control is not None:
            # Serialize control variates
            import pickle
            server_control_bytes = pickle.dumps(self._server_control).hex()
        
        config = {
            "round": server_round,
            "server_lr": self.server_lr,
            "server_control": server_control_bytes,
        }
        fit_ins = FitIns(parameters, config)
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results with SCAFFOLD updates."""
        if not results:
            return None, {}
        
        # Initialize server control if needed
        if self._server_control is None:
            first_params = parameters_to_ndarrays(results[0][1].parameters)
            self._server_control = [np.zeros_like(p) for p in first_params]
        
        # Aggregate model updates
        weights_results = []
        control_deltas = []
        
        for _, fit_res in results:
            num_examples = fit_res.num_examples
            new_params = parameters_to_ndarrays(fit_res.parameters)
            weights_results.append((num_examples, new_params))
            
            # Extract control delta from metrics if provided
            if "control_delta" in fit_res.metrics:
                import pickle
                delta_bytes = bytes.fromhex(fit_res.metrics["control_delta"])
                delta = pickle.loads(delta_bytes)
                control_deltas.append((num_examples, delta))
        
        # Aggregate parameters (standard FedAvg)
        aggregated = self._aggregate_weights(weights_results)
        parameters = ndarrays_to_parameters(aggregated)
        
        # Update server control variate
        if control_deltas:
            self._update_server_control(control_deltas)
        
        # Update client count
        self._num_clients = max(self._num_clients, len(results))
        
        # Aggregate metrics
        metrics_list = [
            (fit_res.num_examples, {
                k: v for k, v in fit_res.metrics.items()
                if k != "control_delta"
            })
            for _, fit_res in results
            if fit_res.metrics
        ]
        aggregated_metrics = self.fit_metrics_aggregation_fn(metrics_list)
        aggregated_metrics["failures"] = len(failures)
        
        return parameters, aggregated_metrics
    
    def _update_server_control(
        self,
        control_deltas: List[Tuple[int, List[np.ndarray]]],
    ) -> None:
        """Update server control variate with client deltas."""
        if not control_deltas or self._server_control is None:
            return
        
        total_samples = sum(n for n, _ in control_deltas)
        if total_samples == 0:
            return
        
        # Average the control deltas
        avg_delta = [np.zeros_like(c) for c in self._server_control]
        
        for num_samples, delta in control_deltas:
            weight = num_samples / total_samples
            for i, d in enumerate(delta):
                avg_delta[i] += weight * d
        
        # Update: c = c + (1/n) * sum(c_i+ - c_i)
        # Simplified: c = c + delta_c
        for i in range(len(self._server_control)):
            self._server_control[i] += avg_delta[i] / len(control_deltas)


class SCAFFOLDClient:
    """
    Client implementation for SCAFFOLD.
    
    Maintains local control variate and corrects gradients during training.
    
    Example:
        >>> client = SCAFFOLDClient(
        ...     model=net,
        ...     train_loader=train_loader,
        ...     lr=0.01,
        ... )
        >>> results = client.train(server_control=server_c, epochs=5)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        lr: float = 0.01,
        device: Union[str, torch.device] = "auto",
    ):
        self.model = model
        self.train_loader = train_loader
        self.lr = lr
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Local control variate
        self.local_control: Optional[List[torch.Tensor]] = None
        
        # Store global params for delta computation
        self.global_params: Optional[List[torch.Tensor]] = None
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters."""
        self.global_params = [
            torch.tensor(p, device=self.device)
            for p in parameters
        ]
        
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def _init_control(self) -> None:
        """Initialize local control variate to zeros."""
        if self.local_control is None:
            self.local_control = [
                torch.zeros_like(p, device=self.device)
                for p in self.model.parameters()
            ]
    
    def train(
        self,
        epochs: int = 1,
        server_control: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, any]:
        """
        Train with SCAFFOLD correction.
        
        Gradient correction: g_corrected = g - c_i + c
        where c is server control and c_i is local control.
        """
        self.model.to(self.device)
        self.model.train()
        self._init_control()
        
        # Convert server control to tensors
        if server_control is not None:
            server_c = [torch.tensor(c, device=self.device) for c in server_control]
        else:
            server_c = [torch.zeros_like(c) for c in self.local_control]
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_steps = 0
        
        for epoch in range(epochs):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # SCAFFOLD gradient correction
                with torch.no_grad():
                    for param, c_local, c_server in zip(
                        self.model.parameters(),
                        self.local_control,
                        server_c,
                    ):
                        if param.grad is not None:
                            # Correct gradient: g - c_i + c
                            param.grad.add_(c_server - c_local)
                
                optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += inputs.size(0)
                num_steps += 1
        
        # Update local control variate
        # c_i+ = c_i - c + (1/K*η) * (x - y)
        # where K is local steps, η is learning rate, x is global params, y is local params
        control_delta = self._update_local_control(server_c, num_steps)
        
        return {
            "loss": total_loss / total_samples if total_samples > 0 else 0.0,
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
            "num_samples": total_samples,
            "control_delta": control_delta,
        }
    
    def _update_local_control(
        self,
        server_control: List[torch.Tensor],
        num_steps: int,
    ) -> List[np.ndarray]:
        """Update local control variate and return delta."""
        if self.global_params is None or self.local_control is None:
            return []
        
        control_delta = []
        
        with torch.no_grad():
            for i, (global_p, local_p, c_local, c_server) in enumerate(zip(
                self.global_params,
                self.model.parameters(),
                self.local_control,
                server_control,
            )):
                # New control: c_i+ = c_i - c + (1/K*η) * (x - y)
                param_delta = (global_p - local_p) / (num_steps * self.lr)
                new_control = c_local - c_server + param_delta
                
                # Control delta: c_i+ - c_i
                delta = (new_control - c_local).cpu().numpy()
                control_delta.append(delta)
                
                # Update local control
                self.local_control[i] = new_control
        
        return control_delta
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters as numpy arrays."""
        return [
            val.cpu().numpy()
            for val in self.model.state_dict().values()
        ]


def create_scaffold_client_fn(
    model_fn,
    train_loaders: Dict[int, torch.utils.data.DataLoader],
    local_epochs: int = 1,
    lr: float = 0.01,
):
    """
    Factory function to create SCAFFOLD client instances.
    
    Args:
        model_fn: Function that returns a new model instance
        train_loaders: Dict mapping client_id to DataLoader
        local_epochs: Number of local training epochs
        lr: Learning rate
    
    Returns:
        Function that creates clients by ID
    """
    # Store persistent clients for state preservation
    clients: Dict[int, SCAFFOLDFlowerClient] = {}
    
    def client_fn(cid: str):
        client_id = int(cid)
        
        # Return existing client to preserve control variate state
        if client_id in clients:
            return clients[client_id]
        
        model = model_fn()
        train_loader = train_loaders[client_id]
        
        client = SCAFFOLDFlowerClient(
            model=model,
            train_loader=train_loader,
            local_epochs=local_epochs,
            lr=lr,
        )
        clients[client_id] = client
        
        return client
    
    return client_fn


class SCAFFOLDFlowerClient:
    """
    Flower-compatible SCAFFOLD client.
    
    Can be used directly with fl.client.start_client or simulation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        local_epochs: int = 1,
        lr: float = 0.01,
    ):
        self.client = SCAFFOLDClient(
            model=model,
            train_loader=train_loader,
            lr=lr,
        )
        self.local_epochs = local_epochs
    
    def get_parameters(self, config=None):
        """Get model parameters."""
        if FLWR_AVAILABLE:
            return ndarrays_to_parameters(self.client.get_parameters())
        return self.client.get_parameters()
    
    def fit(self, parameters, config):
        """Train model with SCAFFOLD."""
        # Set parameters
        if FLWR_AVAILABLE:
            params = parameters_to_ndarrays(parameters)
        else:
            params = parameters
        
        self.client.set_parameters(params)
        
        # Get server control from config
        server_control = None
        if "server_control" in config and config["server_control"] is not None:
            import pickle
            control_bytes = bytes.fromhex(config["server_control"])
            server_control = pickle.loads(control_bytes)
        
        # Train
        results = self.client.train(
            epochs=self.local_epochs,
            server_control=server_control,
        )
        
        # Serialize control delta
        import pickle
        control_delta_bytes = pickle.dumps(results["control_delta"]).hex()
        
        # Return updated parameters and metrics
        new_params = self.client.get_parameters()
        metrics = {
            "loss": results["loss"],
            "accuracy": results["accuracy"],
            "control_delta": control_delta_bytes,
        }
        
        if FLWR_AVAILABLE:
            return ndarrays_to_parameters(new_params), results["num_samples"], metrics
        return new_params, results["num_samples"], metrics
    
    def evaluate(self, parameters, config):
        """Evaluate model."""
        # Set parameters
        if FLWR_AVAILABLE:
            params = parameters_to_ndarrays(parameters)
        else:
            params = parameters
        
        self.client.set_parameters(params)
        
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
