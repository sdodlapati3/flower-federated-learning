"""pytorch-example: A Flower / PyTorch app.

Advanced example with:
- Non-IID data partitioning (Dirichlet)
- Stateful clients (personalized classification head)
- Model checkpointing
- W&B logging
- Learning rate scheduling

Modified to use torchvision Fashion-MNIST for faster downloads on HPC.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.typing import UserConfig
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

FM_NORMALIZATION = ((0.1307,), (0.3081,))
EVAL_TRANSFORMS = Compose([ToTensor(), Normalize(*FM_NORMALIZATION)])
TRAIN_TRANSFORMS = Compose(
    [
        RandomCrop(28, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(*FM_NORMALIZATION),
    ]
)


class Net(nn.Module):
    """Model (simple CNN adapted for Fashion-MNIST)"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


# Global cache for datasets and partitions
_trainset = None
_testset = None
_partition_indices = None


def dirichlet_partition(labels, num_partitions, alpha=1.0, seed=42):
    """
    Partition data using Dirichlet distribution for non-IID split.
    
    Args:
        labels: Array of labels for each sample
        num_partitions: Number of partitions to create
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed for reproducibility
    
    Returns:
        List of indices for each partition
    """
    np.random.seed(seed)
    num_classes = len(np.unique(labels))
    num_samples = len(labels)
    
    # Get indices for each class
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}
    
    # Initialize empty partitions
    partitions = [[] for _ in range(num_partitions)]
    
    # For each class, distribute samples according to Dirichlet distribution
    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_partitions))
        
        # Calculate number of samples per partition for this class
        proportions = (proportions * len(indices)).astype(int)
        # Handle rounding errors
        proportions[-1] = len(indices) - proportions[:-1].sum()
        
        # Assign samples to partitions
        start = 0
        for i, num in enumerate(proportions):
            partitions[i].extend(indices[start:start + num].tolist())
            start += num
    
    # Shuffle within each partition
    for i in range(num_partitions):
        np.random.shuffle(partitions[i])
    
    return partitions


def load_data(partition_id: int, num_partitions: int):
    """Load partition Fashion-MNIST data with non-IID (Dirichlet) partitioning."""
    global _trainset, _testset, _partition_indices
    
    # Download and cache datasets
    if _trainset is None:
        _trainset = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=TRAIN_TRANSFORMS
        )
    if _testset is None:
        _testset = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=EVAL_TRANSFORMS
        )
    
    # Create Dirichlet partitions if not cached
    if _partition_indices is None:
        labels = np.array(_trainset.targets)
        _partition_indices = dirichlet_partition(labels, num_partitions, alpha=1.0, seed=42)
    
    # Get this partition's indices
    partition_indices = _partition_indices[partition_id]
    
    # Split into train (80%) and validation (20%)
    split_idx = int(len(partition_indices) * 0.8)
    train_indices = partition_indices[:split_idx]
    val_indices = partition_indices[split_idx:]
    
    # Create subset datasets
    train_subset = Subset(_trainset, train_indices)
    val_subset = Subset(_trainset, val_indices)
    
    # Create dataloaders
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(val_subset, batch_size=32)
    
    return trainloader, testloader


def get_testloader():
    """Get the full test set loader for server-side evaluation."""
    global _testset
    if _testset is None:
        _testset = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=EVAL_TRANSFORMS
        )
    return DataLoader(_testset, batch_size=64)


def create_run_dir(config: UserConfig) -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(dict(config), fp)

    return save_path, run_dir
