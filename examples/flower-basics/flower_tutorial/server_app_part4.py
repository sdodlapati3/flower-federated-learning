"""flower-tutorial: A Flower / PyTorch app.

Part 4: Communicate Custom Messages
- Custom client app sends TrainProcessMetadata
- Strategy deserializes and processes per-client metadata
- Logs detailed training information
- W&B integration (from Part 3)
"""

from datetime import datetime
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from flower_tutorial.custom_strategy_part4 import CustomFedAdagradWithMessages
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

    # Initialize custom strategy with message handling
    strategy = CustomFedAdagradWithMessages(fraction_train=fraction_train)

    # Create output directory for this run
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    strategy.set_save_path(save_path)

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=central_evaluate,
    )

    # Save final model
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    final_model_path = save_path / "final_model.pt"
    torch.save(state_dict, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Part 4 - Custom Messages Summary")
    print(f"{'='*60}")
    print(f"Output directory: {save_path}")
    print(f"Features demonstrated:")
    print(f"  - Custom TrainProcessMetadata sent from clients")
    print(f"  - Per-client training time and convergence status logged")
    print(f"  - Model checkpointing on best accuracy")
    print(f"  - W&B offline logging")
    print(f"{'='*60}")
