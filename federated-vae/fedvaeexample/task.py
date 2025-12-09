"""fedvae: A Flower app for Federated Variational Autoencoder.

Modified to use torchvision.datasets.CIFAR10 instead of HuggingFace FederatedDataset.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Cache datasets
_train_dataset = None
_test_dataset = None


class Flatten(nn.Module):
    """Flattens input by reshaping it into a one-dimensional tensor."""

    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    """Unflattens a tensor converting it to a desired shape."""

    def forward(self, input):
        return input.view(-1, 16, 6, 6)


class Net(nn.Module):
    def __init__(self, h_dim=576, z_dim=10) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=6, kernel_size=4, stride=2
            ),  # [batch, 6, 15, 15]
            nn.ReLU(),
            nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=5, stride=2
            ),  # [batch, 16, 6, 6]
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=4, stride=2),
            nn.Tanh(),
        )

    def reparametrize(self, h):
        """Reparametrization layer of VAE."""
        mu, logvar = self.fc1(h), self.fc2(h)
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar

    def encode(self, x):
        """Encoder of the VAE."""
        h = self.encoder(x)
        z, mu, logvar = self.reparametrize(h)
        return z, mu, logvar

    def decode(self, z):
        """Decoder of the VAE."""
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z_decode = self.decode(z)
        return z_decode, mu, logvar


def load_data(partition_id, num_partitions):
    """Load partition CIFAR10 data using torchvision."""
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
    
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(val_subset, batch_size=32)
    
    return trainloader, testloader


def trainer(net, trainloader, epochs, learning_rate, device):
    """Train the network on the training set."""
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    for _ in range(epochs):
        for images, _ in trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.05 * kld_loss
            loss.backward()
            optimizer.step()


def test(net, testloader, device):
    """Validate the network on the entire test set."""
    total, loss = 0, 0.0
    with torch.no_grad():
        for images, _ in testloader:
            images = images.to(device)
            recon_images, mu, logvar = net(images)
            recon_loss = F.mse_loss(recon_images, images)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += recon_loss + kld_loss
            total += len(images)
    return loss / total


def generate(net, image):
    """Reproduce the input with trained VAE."""
    with torch.no_grad():
        return net.forward(image)
