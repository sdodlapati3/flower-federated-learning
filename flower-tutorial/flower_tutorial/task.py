"""flower-tutorial: A Flower / PyTorch app.

Modified to use torchvision CIFAR10 instead of HuggingFace datasets
for faster downloads on HPC clusters.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np


@dataclass
class TrainProcessMetadata:
    """Metadata about the training process.
    
    This dataclass captures detailed information about what happened
    during local training on each client. It's used in Part 4 to
    demonstrate custom message communication.
    """
    training_time: float           # Time taken for training in seconds
    converged: bool                # Whether training converged (loss < threshold)
    training_losses: dict          # Loss per epoch, e.g. {"epoch_1": 0.5, "epoch_2": 0.3}
    final_loss: float              # Final training loss
    num_batches: int               # Number of batches processed


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

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


# Global cache for datasets
_trainset = None
_testset = None

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data using torchvision.
    
    Partitions the training data IID across num_partitions clients.
    Each client gets a unique subset of the training data.
    """
    global _trainset, _testset
    
    # Download and cache datasets
    if _trainset is None:
        _trainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=pytorch_transforms
        )
    if _testset is None:
        _testset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=pytorch_transforms
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
    
    # Split partition into train (80%) and validation (20%)
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


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
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


def central_evaluate(server_round: int, arrays) -> dict:
    """Evaluate model on the server side (centralized evaluation).
    
    This callback is called by the strategy after every round of federated
    learning. It evaluates the aggregated model on the entire test dataset.
    """
    from flwr.app import MetricRecord
    
    global _testset
    
    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the entire CIFAR10 test dataset using torchvision
    if _testset is None:
        _testset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=pytorch_transforms
        )
    
    testloader = DataLoader(_testset, batch_size=64)

    # Evaluate the model on the test set
    loss, accuracy = test(model, testloader, device)

    print(f"[Round {server_round}] Server-side evaluation: loss={loss:.4f}, accuracy={accuracy:.4f}")

    # Return the evaluation metrics
    return MetricRecord({"accuracy": accuracy, "loss": loss})
