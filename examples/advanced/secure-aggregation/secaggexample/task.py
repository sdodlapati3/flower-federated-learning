"""secaggexample: A Flower with SecAgg+ app.

Modified to use torchvision.datasets.CIFAR10 instead of HuggingFace FederatedDataset.
"""

import random
from collections import OrderedDict
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Cache datasets
_train_dataset = None
_test_dataset = None


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


def make_net(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return Net()


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def load_data(partition_id: int, num_partitions: int, batch_size: int, is_demo: bool):
    """Load partition CIFAR10 data using torchvision."""
    if is_demo:
        trainloader, testloader = Mock(dataset=[0]), Mock(dataset=[0])
        return trainloader, testloader
    
    global _train_dataset, _test_dataset
    
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
    
    # Partition the training data
    total_size = len(_train_dataset)
    partition_size = total_size // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else total_size
    
    train_indices = list(range(start_idx, end_idx))
    partition_train_size = int(0.8 * len(train_indices))
    
    train_subset = Subset(_train_dataset, train_indices[:partition_train_size])
    val_subset = Subset(_train_dataset, train_indices[partition_train_size:])
    
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_subset, batch_size=batch_size)
    
    return trainloader, testloader


def train(net, trainloader, valloader, epochs, learning_rate, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()

    val_loss, val_acc = test(net, valloader, device)

    results = {
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
    loss = loss / len(testloader)
    return loss, accuracy
