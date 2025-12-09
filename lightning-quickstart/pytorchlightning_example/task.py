"""pytorchlightning_example: A Flower / PyTorch Lightning app.

Modified to use torchvision MNIST instead of HuggingFace for faster downloads.
"""

import logging
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28),
        )

    def forward(self, x) -> Any:
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self) -> Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        x, _ = train_batch  # (images, labels)
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        self._evaluate(batch, "val")

    def test_step(self, batch, batch_idx) -> None:
        self._evaluate(batch, "test")

    def _evaluate(self, batch, stage=None) -> None:
        x, _ = batch  # (images, labels)
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)


# Global cache for dataset
_trainset = None

transform = transforms.Compose([
    transforms.ToTensor(),
])


def load_data(partition_id, num_partitions):
    """Load partition of MNIST data using torchvision."""
    global _trainset
    
    # Download and cache dataset
    if _trainset is None:
        _trainset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    
    # Partition the training data IID
    total_size = len(_trainset)
    partition_size = total_size // num_partitions
    
    # Create deterministic partition indices
    np.random.seed(42)
    indices = np.random.permutation(total_size)
    
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size
    partition_indices = indices[start_idx:end_idx]
    
    # Split: 60% train, 20% validation, 20% test
    n = len(partition_indices)
    train_size = int(0.6 * n)
    val_size = int(0.2 * n)
    
    train_indices = partition_indices[:train_size]
    val_indices = partition_indices[train_size:train_size + val_size]
    test_indices = partition_indices[train_size + val_size:]
    
    # Create subset datasets
    train_subset = Subset(_trainset, train_indices)
    val_subset = Subset(_trainset, val_indices)
    test_subset = Subset(_trainset, test_indices)
    
    # Create dataloaders
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
    valloader = DataLoader(val_subset, batch_size=32, num_workers=2)
    testloader = DataLoader(test_subset, batch_size=32, num_workers=1)
    
    return trainloader, valloader, testloader
