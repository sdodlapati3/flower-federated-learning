"""Part 4: Custom Strategy with Message Deserialization.

This module extends the advanced strategy from Part 3 to:
1. Deserialize TrainProcessMetadata from client messages
2. Log detailed per-client training info
3. Use metadata for custom aggregation logic
"""

import io
import pickle
import time
from dataclasses import asdict
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad, Result
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

from flower_tutorial.task import TrainProcessMetadata

PROJECT_NAME = "flower-tutorial-part4"


class CustomFedAdagradWithMessages(FedAdagrad):
    """Custom FedAdagrad that handles custom messages from clients.
    
    This strategy extends FedAdagrad with:
    - Deserializes TrainProcessMetadata from client replies
    - Logs per-client training details
    - Can use convergence status for custom aggregation logic
    - Model checkpointing (from Part 3)
    - W&B logging (from Part 3)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path: Optional[Path] = None
        self.best_acc_so_far: float = 0.0
        self.client_metadata: list = []  # Store metadata from all clients per round

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training with LR decay."""
        # Decrease learning rate by a factor of 0.5 every 5 rounds
        if server_round % 5 == 0 and server_round > 0:
            config["lr"] *= 0.5
            print(f"[Round {server_round}] Learning rate decreased to: {config['lr']}")
        return super().configure_train(server_round, arrays, config, grid)

    def set_save_path(self, path: Path):
        """Set the path where wandb logs and model checkpoints will be saved."""
        self.save_path = path
        log(INFO, "Save path set to: %s", path)

    def _update_best_acc(
        self, current_round: int, accuracy: float, arrays: ArrayRecord
    ) -> None:
        """Update best accuracy and save model checkpoint if current accuracy is higher."""
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            log(INFO, "💡 New best global model found: %.4f", accuracy)
            file_name = f"model_state_acc_{accuracy:.4f}_round_{current_round}.pth"
            torch.save(arrays.to_torch_state_dict(), self.save_path / file_name)
            log(INFO, "💾 New best model saved to disk: %s", file_name)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate training results and process custom metadata from clients.
        
        This method:
        1. Extracts TrainProcessMetadata from each client's reply
        2. Logs detailed training information
        3. Calls parent aggregation for model weights and metrics
        """
        # Convert to list so we can iterate multiple times
        replies_list = list(replies)
        
        # Clear previous round's metadata
        self.client_metadata = []
        
        log(INFO, "")
        log(INFO, "📊 Processing client training metadata:")
        
        converged_count = 0
        total_training_time = 0.0
        
        for i, reply in enumerate(replies_list):
            if reply.has_content() and "train_metadata" in reply.content:
                # Retrieve the ConfigRecord containing serialized metadata
                config_record = reply.content["train_metadata"]
                metadata_bytes = config_record["meta"]
                
                # Deserialize the TrainProcessMetadata object
                train_meta: TrainProcessMetadata = pickle.loads(metadata_bytes)
                self.client_metadata.append(train_meta)
                
                # Log client-specific training info
                log(INFO, "  Client %d:", i + 1)
                log(INFO, "    - Training time: %.2fs", train_meta.training_time)
                log(INFO, "    - Final loss: %.4f", train_meta.final_loss)
                log(INFO, "    - Converged: %s", train_meta.converged)
                log(INFO, "    - Batches processed: %d", train_meta.num_batches)
                log(INFO, "    - Epoch losses: %s", train_meta.training_losses)
                
                if train_meta.converged:
                    converged_count += 1
                total_training_time += train_meta.training_time
        
        if self.client_metadata:
            log(INFO, "")
            log(INFO, "📈 Round %d Summary:", server_round)
            log(INFO, "  - Clients converged: %d/%d", converged_count, len(self.client_metadata))
            log(INFO, "  - Total client training time: %.2fs", total_training_time)
            log(INFO, "  - Avg training time per client: %.2fs", total_training_time / len(self.client_metadata))
        
        # Call parent class to aggregate model weights and metrics
        return super().aggregate_train(server_round, replies_list)

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
        """Execute federated learning with custom message handling and W&B logging."""
        
        # Initialize W&B
        name = f"{str(self.save_path.parent.name)}/{str(self.save_path.name)}-Part4"
        wandb.init(project=PROJECT_NAME, name=name, dir=str(self.save_path))

        self.best_acc_so_far = 0.0

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        
        # Initial evaluation
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res
                wandb.log({"initial_accuracy": res["accuracy"], "initial_loss": res["loss"]}, step=0)

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # Training
            train_replies = grid.send_and_receive(
                messages=self.configure_train(current_round, arrays, train_config, grid),
                timeout=timeout,
            )

            # Aggregate train results (includes custom metadata processing)
            agg_arrays, agg_train_metrics = self.aggregate_train(current_round, train_replies)

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics
                
                # Log to W&B with client metadata
                metrics_dict = dict(agg_train_metrics)
                if self.client_metadata:
                    metrics_dict["converged_clients"] = sum(1 for m in self.client_metadata if m.converged)
                    metrics_dict["avg_client_training_time"] = sum(m.training_time for m in self.client_metadata) / len(self.client_metadata)
                wandb.log(metrics_dict, step=current_round)

            # Client evaluation
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(current_round, arrays, evaluate_config, grid),
                timeout=timeout,
            )

            agg_evaluate_metrics = self.aggregate_evaluate(current_round, evaluate_replies)

            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                wandb.log(dict(agg_evaluate_metrics), step=current_round)

            # Server-side evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    self._update_best_acc(current_round, res["accuracy"], arrays)
                    wandb.log(dict(res), step=current_round)

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))

        # Final W&B summary
        wandb.log({
            "final_accuracy": self.best_acc_so_far,
            "total_rounds": num_rounds,
            "total_time_seconds": time.time() - t_start,
        })
        
        wandb.finish()

        return result
