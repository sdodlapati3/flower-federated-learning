"""pytorch-example: A Flower / PyTorch app.

Advanced ServerApp with:
- Custom FedAvg strategy with W&B logging
- Model checkpointing (saves best models)
- Server-side evaluation on full test set
- Learning rate scheduling
"""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from pytorch_example.strategy import CustomFedAvg
from pytorch_example.task import Net, create_run_dir, get_testloader, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_eval = context.run_config["fraction-evaluate"]
    device = context.run_config["server-device"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize CustomFedAvg strategy
    strategy = CustomFedAvg(
        fraction_train=fraction_train, fraction_evaluate=fraction_eval
    )

    # Define directory for results and save config
    save_path, run_dir = create_run_dir(config=context.run_config)
    strategy.set_save_path_and_run_dir(save_path, run_dir)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": 0.1}),
        num_rounds=num_rounds,
        evaluate_fn=get_global_evaluate_fn(device=device),
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, save_path / "final_model.pt")
    print(f"Final model saved to {save_path / 'final_model.pt'}")


def get_global_evaluate_fn(device: str):
    """Return an evaluation function for server-side evaluation."""

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on the full Fashion-MNIST test set."""
        testloader = get_testloader()

        net = Net()
        net.load_state_dict(arrays.to_torch_state_dict())
        net.to(device)
        loss, accuracy = test(net, testloader, device=device)
        
        print(f"[Round {server_round}] Server-side eval: loss={loss:.4f}, accuracy={accuracy:.4f}")
        
        return MetricRecord({"accuracy": accuracy, "loss": loss})

    return global_evaluate
