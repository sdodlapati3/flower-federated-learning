"""flower-tutorial: A Flower / PyTorch app.

Part 3: Build a Strategy from Scratch
- Custom start() method with full control over FL rounds
- Model checkpointing (save best model)
- Weights & Biases integration for experiment tracking
- Learning rate decay (from Part 2)
"""

from datetime import datetime
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

# Import advanced strategy with W&B logging and checkpointing
from flower_tutorial.advanced_strategy import AdvancedFedAdagrad
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

    # Initialize AdvancedFedAdagrad strategy
    # This custom strategy includes:
    # - W&B logging
    # - Model checkpointing
    # - Learning rate decay
    strategy = AdvancedFedAdagrad(fraction_train=fraction_train)

    # Create output directory for this run
    # Each run gets a unique directory based on timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Set the path where results and model checkpoints will be saved
    strategy.set_save_path(save_path)

    # Start strategy with custom start() method
    # This will:
    # 1. Initialize W&B
    # 2. Run federated learning for num_rounds
    # 3. Log metrics to W&B after each round
    # 4. Save model checkpoints when new best accuracy is found
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
    final_model_path = save_path / "final_model.pt"
    torch.save(state_dict, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Part 3 - Advanced Strategy Summary")
    print(f"{'='*60}")
    print(f"Output directory: {save_path}")
    print(f"Checkpoints saved: {list(save_path.glob('model_state_*.pth'))}")
    print(f"W&B logs: Check your W&B dashboard at https://wandb.ai")
    print(f"{'='*60}")
