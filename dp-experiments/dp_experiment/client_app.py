"""DP Experiment Client Application with configurable DP settings."""

import warnings

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from opacus import PrivacyEngine
from dp_experiment.task import Net, get_weights, load_data, set_weights, test, train

warnings.filterwarnings("ignore", category=UserWarning)


class DPFlowerClient(NumPyClient):
    """Flower client with Differential Privacy via Opacus."""
    
    def __init__(
        self,
        train_loader,
        test_loader,
        target_delta,
        noise_multiplier,
        max_grad_norm,
        experiment_name,
    ) -> None:
        super().__init__()
        self.model = Net()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.experiment_name = experiment_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.round_epsilon = 0.0

    def fit(self, parameters, config):
        model = self.model
        set_weights(model, parameters)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Wrap with Opacus PrivacyEngine
        privacy_engine = PrivacyEngine(secure_mode=False)
        (
            model,
            optimizer,
            self.train_loader,
        ) = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        epsilon = train(
            model,
            self.train_loader,
            privacy_engine,
            optimizer,
            self.target_delta,
            device=self.device,
        )

        self.round_epsilon = epsilon
        print(f"[{self.experiment_name}] ε = {epsilon:.4f} (δ = {self.target_delta})")

        return (
            get_weights(model), 
            len(self.train_loader.dataset), 
            {"epsilon": float(epsilon)}
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader, self.device)
        return loss, len(self.test_loader.dataset), {
            "accuracy": accuracy,
            "epsilon": float(self.round_epsilon),
        }


def client_fn(context: Context):
    """Create a DP-enabled Flower client."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Get experiment configuration
    noise_multiplier = context.run_config["noise-multiplier"]
    max_grad_norm = context.run_config["max-grad-norm"]
    target_delta = context.run_config["target-delta"]
    is_iid = context.run_config.get("is-iid", True)
    alpha = context.run_config.get("dirichlet-alpha", 0.5)
    experiment_name = context.run_config.get("experiment-name", "default")
    
    # Load data with appropriate partitioning
    train_loader, test_loader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        is_iid=is_iid,
        alpha=alpha,
    )
    
    return DPFlowerClient(
        train_loader,
        test_loader,
        target_delta,
        noise_multiplier,
        max_grad_norm,
        experiment_name,
    ).to_client()


app = ClientApp(client_fn=client_fn)
