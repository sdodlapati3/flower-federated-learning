"""flower-tutorial: A Flower / PyTorch app.

Part 2: Use a Federated Learning Strategy
- Switch from FedAvg to FedAdagrad (adaptive gradient strategy)
- Add server-side (centralized) evaluation
- Use custom strategy with learning rate decay
"""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

# Import custom strategy and centralized evaluation function
from flower_tutorial.custom_strategy import CustomFedAdagrad
from flower_tutorial.task import Net, central_evaluate

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize CustomFedAdagrad strategy (with LR decay)
    # Using FedAdagrad instead of FedAvg for adaptive learning rates
    strategy = CustomFedAdagrad(fraction_train=fraction_train)

    # Start strategy, run FedAdagrad for `num_rounds`
    # Now with server-side evaluation via evaluate_fn callback
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=central_evaluate,  # Server-side evaluation callback
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
    print("Model saved to final_model.pt")
