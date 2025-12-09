"""Part 3: Custom FedAdagrad Strategy with W&B Logging.

This module implements a fully customized strategy that:
1. Saves the best global model when a new best accuracy is found
2. Logs all metrics to Weights & Biases (W&B)
3. Implements learning rate decay (from Part 2)
"""

import io
import os
import time
from datetime import datetime
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import wandb
from dotenv import load_dotenv
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.common.logger import logger
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

# Load environment variables from .env file
load_dotenv()

# W&B Project configuration
PROJECT_NAME = os.getenv("WANDB_PROJECT", "flower-federated-learning")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "sdodlapa")


class CustomFedAdagradWithWandB(FedAdagrad):
    """Custom FedAdagrad strategy with W&B logging and model checkpointing.
    
    Features:
    - Learning rate decay every 5 rounds (from Part 2)
    - Automatic logging of all metrics to Weights & Biases
    - Saves model checkpoint when new best accuracy is found
    - Customized start() method with full control over federated rounds
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path: Optional[Path] = None
        self.best_acc_so_far: float = 0.0

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training with LR decay."""
        # Decrease learning rate by a factor of 0.5 every 5 rounds
        if server_round % 5 == 0 and server_round > 0:
            config["lr"] *= 0.5
            print(f"[Round {server_round}] Learning rate decreased to: {config['lr']}")
        # Pass the updated config and the rest of arguments to the parent class
        return super().configure_train(server_round, arrays, config, grid)

    def set_save_path(self, path: Path) -> None:
        """Set the path where W&B logs and model checkpoints will be saved."""
        self.save_path = path
        path.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {path}")

    def _update_best_acc(
        self, current_round: int, accuracy: float, arrays: ArrayRecord
    ) -> None:
        """Update best accuracy and save model checkpoint if current accuracy is higher."""
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            
            if self.save_path is not None:
                # Save the PyTorch model
                file_name = f"model_state_acc_{accuracy:.4f}_round_{current_round}.pth"
                torch.save(arrays.to_torch_state_dict(), self.save_path / file_name)
                logger.log(INFO, "💾 New best model saved to disk: %s", file_name)
                
                # Log model artifact to W&B
                wandb.save(str(self.save_path / file_name))

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]
        ] = None,
    ) -> Result:
        """Execute the federated learning strategy with W&B logging.
        
        This customized start() method:
        1. Initializes a W&B run for experiment tracking
        2. Logs training and evaluation metrics after each round
        3. Saves model checkpoints when new best accuracy is found
        4. Provides detailed logging of the federated learning process
        """
        
        # Initialize W&B
        run_name = f"FL-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if self.save_path:
            run_name = f"{self.save_path.parent.name}/{self.save_path.name}"
        
        wandb.init(
            project=PROJECT_NAME,
            entity=WANDB_ENTITY,
            name=run_name,
            config={
                "num_rounds": num_rounds,
                "strategy": self.__class__.__name__,
                "fraction_train": self.fraction_train,
                "fraction_evaluate": self.fraction_evaluate,
                "initial_lr": train_config.get("lr", 0.01) if train_config else 0.01,
            }
        )
        
        # Keep track of best acc
        self.best_acc_so_far = 0.0

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        
        # Evaluate starting global parameters (Round 0)
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res
                # Log initial metrics to W&B
                wandb.log({"server/accuracy": res["accuracy"], "server/loss": res["loss"]}, step=0)

        arrays = initial_arrays

        # Main federated learning loop
        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate training results
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics
                # Log training metrics to W&B
                wandb.log(
                    {f"train/{k}": v for k, v in dict(agg_train_metrics).items()},
                    step=current_round
                )

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluation results
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log evaluation metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                # Log client evaluation metrics to W&B
                wandb.log(
                    {f"eval/{k}": v for k, v in dict(agg_evaluate_metrics).items()},
                    step=current_round
                )

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE / CENTRALIZED) -------------------
            # -----------------------------------------------------------------

            # Centralized evaluation on server
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    # Maybe save to disk if new best is found
                    self._update_best_acc(current_round, res["accuracy"], arrays)
                    # Log server-side evaluation to W&B
                    wandb.log(
                        {"server/accuracy": res["accuracy"], "server/loss": res["loss"]},
                        step=current_round
                    )

        # Log final summary
        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        # Log final metrics to W&B summary
        wandb.summary["best_accuracy"] = self.best_acc_so_far
        wandb.summary["total_rounds"] = num_rounds
        wandb.summary["total_time_seconds"] = time.time() - t_start

        # Finish W&B run
        wandb.finish()

        return result
