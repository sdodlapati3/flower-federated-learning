"""fl_dp_sa: Flower Example using Differential Privacy and Secure Aggregation.

Modified to use torchvision.datasets.MNIST instead of HuggingFace FederatedDataset
for faster downloads on HPC clusters.
"""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Cache datasets
_train_dataset = None
_test_dataset = None


class Net(nn.Module):
    """Simple CNN for MNIST classification."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_data(partition_id: int, num_partitions: int):
    """Load partition MNIST data using torchvision."""
    global _train_dataset, _test_dataset
    
    # Data directory
    data_dir = Path(__file__).parent.parent / "data"
    
    # Transforms (MNIST is grayscale, single channel)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load full datasets (cached)
    if _train_dataset is None:
        _train_dataset = datasets.MNIST(
            root=str(data_dir), train=True, download=True, transform=transform
        )
    if _test_dataset is None:
        _test_dataset = datasets.MNIST(
            root=str(data_dir), train=False, download=True, transform=transform
        )
    
    # Partition the training data (IID partitioning)
    total_size = len(_train_dataset)
    partition_size = total_size // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else total_size
    
    train_indices = list(range(start_idx, end_idx))
    
    # Split partition into train (80%) and validation (20%)
    partition_train_size = int(0.8 * len(train_indices))
    train_subset = Subset(_train_dataset, train_indices[:partition_train_size])
    val_subset = Subset(_train_dataset, train_indices[partition_train_size:])
    
    # Create DataLoaders
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(val_subset, batch_size=32)
    
    return trainloader, testloader


def train(net, trainloader, valloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    train_loss, train_acc = test(net, trainloader, device)
    val_loss, val_acc = test(net, valloader, device)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


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
    return loss, accuracy


def get_weights(net):
    """Extract model weights as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from a list of numpy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
