"""Custom Federated Learning Strategy with Learning Rate Decay.

This module implements a custom FedAdagrad strategy that decreases
the learning rate by a factor of 0.5 every 5 rounds.
"""

from typing import Iterable
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad
from flwr.app import ArrayRecord, ConfigRecord, Message


class CustomFedAdagrad(FedAdagrad):
    """FedAdagrad strategy with learning rate decay.
    
    This custom strategy extends FedAdagrad to implement learning rate
    decay during federated training. The learning rate is decreased by
    a factor of 0.5 every 5 rounds.
    """
    
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training with LR decay.
        
        Args:
            server_round: The current round of federated learning.
            arrays: The current model parameters.
            config: Configuration to send to clients.
            grid: The grid of available clients.
            
        Returns:
            An iterable of Messages to send to selected clients.
        """
        # Decrease learning rate by a factor of 0.5 every 5 rounds
        if server_round % 5 == 0 and server_round > 0:
            config["lr"] *= 0.5
            print(f"[Round {server_round}] Learning rate decreased to: {config['lr']}")
        
        # Pass the updated config and the rest of arguments to the parent class
        return super().configure_train(server_round, arrays, config, grid)
