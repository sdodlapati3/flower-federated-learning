"""DP Experiment Task Module.

Supports both IID and Non-IID (Dirichlet) partitioning with Opacus DP.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# Cache datasets
_train_dataset = None
_test_dataset = None
_partitions_cache = {}


class Net(nn.Module):
    """Simple CNN for CIFAR-10 classification."""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def dirichlet_partition(dataset, num_partitions, alpha=0.5):
    """Partition data using Dirichlet distribution for non-IID split.
    
    Args:
        dataset: The dataset to partition
        num_partitions: Number of partitions (clients)
        alpha: Dirichlet concentration parameter (lower = more non-IID)
               alpha=0.1 -> very non-IID
               alpha=0.5 -> moderately non-IID  
               alpha=1.0 -> slightly non-IID
               alpha=100 -> nearly IID
    
    Returns:
        List of index lists for each partition
    """
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    
    # Initialize partition indices
    partition_indices = [[] for _ in range(num_partitions)]
    
    # For each class, distribute samples according to Dirichlet distribution
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        np.random.shuffle(class_indices)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_partitions)
        
        # Ensure minimum samples per partition
        proportions = np.maximum(proportions, 0.01)
        proportions = proportions / proportions.sum()
        
        # Assign samples to partitions
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)
        split_points[-1] = len(class_indices)
        
        start = 0
        for partition_id, end in enumerate(split_points):
            partition_indices[partition_id].extend(class_indices[start:end].tolist())
            start = end
    
    return partition_indices


def load_data(partition_id: int, num_partitions: int, is_iid: bool = True, alpha: float = 0.5):
    """Load CIFAR-10 data with IID or Non-IID partitioning."""
    global _train_dataset, _test_dataset, _partitions_cache
    
    data_dir = Path(__file__).parent.parent / "data"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if _train_dataset is None:
        _train_dataset = datasets.CIFAR10(
            root=str(data_dir), train=True, download=True, transform=transform
        )
    if _test_dataset is None:
        _test_dataset = datasets.CIFAR10(
            root=str(data_dir), train=False, download=True, transform=transform
        )
    
    cache_key = f"{is_iid}_{alpha}_{num_partitions}"
    
    if cache_key not in _partitions_cache:
        if is_iid:
            # IID partitioning - simple sequential split
            total_size = len(_train_dataset)
            partition_size = total_size // num_partitions
            _partitions_cache[cache_key] = [
                list(range(i * partition_size, (i + 1) * partition_size if i < num_partitions - 1 else total_size))
                for i in range(num_partitions)
            ]
        else:
            # Non-IID partitioning using Dirichlet
            np.random.seed(42)  # For reproducibility
            _partitions_cache[cache_key] = dirichlet_partition(_train_dataset, num_partitions, alpha)
    
    partition_indices = _partitions_cache[cache_key][partition_id]
    
    # Split into train (80%) and validation (20%)
    split_idx = int(0.8 * len(partition_indices))
    train_indices = partition_indices[:split_idx]
    val_indices = partition_indices[split_idx:]
    
    train_subset = Subset(_train_dataset, train_indices)
    val_subset = Subset(_train_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(val_subset, batch_size=32)
    
    return train_loader, test_loader


def train(net, train_loader, privacy_engine, optimizer, target_delta, device, epochs=1):
    """Train with differential privacy."""
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    
    for _ in range(epochs):
        for images, labels in tqdm(train_loader, "Training", leave=False):
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    return epsilon


def test(net, test_loader, device):
    """Evaluate model accuracy."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    
    accuracy = correct / len(test_loader.dataset)
    return loss, accuracy
