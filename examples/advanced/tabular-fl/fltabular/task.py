"""fltabular: Flower Example on Adult Census Income Tabular Dataset.

Modified to use sklearn synthetic data instead of HuggingFace dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Cache the synthetic dataset
_synthetic_data = None


def get_synthetic_data():
    """Generate a synthetic income-like classification dataset."""
    global _synthetic_data
    
    if _synthetic_data is not None:
        return _synthetic_data
    
    # Generate synthetic data similar to Adult Census Income
    X, y = make_classification(
        n_samples=32561,  # Same as Adult Census
        n_features=14,
        n_informative=10,
        n_redundant=2,
        n_clusters_per_class=3,
        weights=[0.76, 0.24],  # Income imbalance similar to real data
        random_state=42
    )
    
    _synthetic_data = (X, y)
    return _synthetic_data


def load_data(partition_id: int, num_partitions: int):
    """Load partitioned synthetic income data."""
    
    X_full, y_full = get_synthetic_data()
    
    # Partition the data
    total_size = len(X_full)
    partition_size = total_size // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else total_size
    
    X_partition = X_full[start_idx:end_idx]
    y_partition = y_full[start_idx:end_idx]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_partition, y_partition, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


class IncomeClassifier(nn.Module):
    def __init__(self, input_dim: int = 14):
        super(IncomeClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x


def trainer(model, train_loader, num_epochs=1):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()


def evaluator(model, test_loader):
    model.eval()
    criterion = nn.BCELoss()
    loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            batch_loss = criterion(outputs, y_batch)
            loss += batch_loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    loss = loss / len(test_loader)
    return loss, accuracy
