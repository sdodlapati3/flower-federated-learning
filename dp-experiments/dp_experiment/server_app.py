"""DP Experiment Server Application with metrics tracking."""

import logging
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from dp_experiment.task import Net, get_weights

logging.getLogger("flwr").setLevel(logging.INFO)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate accuracy and epsilon from all clients."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    epsilons = [m.get("epsilon", 0) for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "avg_epsilon": sum(epsilons) / len(epsilons) if epsilons else 0,
        "max_epsilon": max(epsilons) if epsilons else 0,
    }


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit metrics including epsilon tracking."""
    epsilons = [m.get("epsilon", 0) for _, m in metrics]
    return {
        "avg_epsilon": sum(epsilons) / len(epsilons) if epsilons else 0,
        "max_epsilon": max(epsilons) if epsilons else 0,
    }


def server_fn(context: Context) -> ServerAppComponents:
    """Configure FL server with experiment settings."""
    num_rounds = context.run_config["num-server-rounds"]
    experiment_name = context.run_config.get("experiment-name", "default")
    
    print(f"\n{'='*60}")
    print(f"Starting Experiment: {experiment_name}")
    print(f"Rounds: {num_rounds}")
    print(f"Noise Multiplier: {context.run_config['noise-multiplier']}")
    print(f"Max Grad Norm: {context.run_config['max-grad-norm']}")
    print(f"IID: {context.run_config.get('is-iid', True)}")
    print(f"{'='*60}\n")

    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        initial_parameters=parameters,
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
