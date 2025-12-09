"""Standalone DP Experiments - Direct Python Implementation.

Runs FL experiments with multiple epsilon settings comparing IID vs Non-IID.
Uses Opacus for differential privacy.
"""

import json
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


# Configuration
NUM_ROUNDS = 3
NUM_CLIENTS = 5
EPOCHS_PER_ROUND = 1
BATCH_SIZE = 32
TARGET_DELTA = 1e-5


class Net(nn.Module):
    """Simple CNN for CIFAR-10 - Opacus compatible."""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, 32)  # Use GroupNorm for Opacus
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_data():
    """Load and return CIFAR-10 datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    data_dir = Path("./data")
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset


def iid_partition(dataset, num_partitions):
    """Create IID partition of dataset."""
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    partition_size = len(indices) // num_partitions
    
    partitions = []
    for i in range(num_partitions):
        start = i * partition_size
        end = start + partition_size if i < num_partitions - 1 else len(indices)
        partitions.append(indices[start:end])
    
    return partitions


def dirichlet_partition(dataset, num_partitions, alpha=0.5):
    """Create Non-IID partition using Dirichlet distribution."""
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = 10
    
    partition_indices = [[] for _ in range(num_partitions)]
    
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0].tolist()
        np.random.shuffle(class_indices)
        
        proportions = np.random.dirichlet([alpha] * num_partitions)
        proportions = proportions / proportions.sum()
        
        cumsum = np.cumsum(proportions) * len(class_indices)
        cumsum = [0] + cumsum.astype(int).tolist()
        
        for i in range(num_partitions):
            start, end = cumsum[i], cumsum[i + 1]
            partition_indices[i].extend(class_indices[start:end])
    
    for indices in partition_indices:
        np.random.shuffle(indices)
    
    return partition_indices


def get_weights(net):
    """Get model weights as list of numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from list of numpy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train_client_dp(client_id, partition_indices, train_dataset, global_weights, 
                    noise_multiplier, max_grad_norm, epochs=1):
    """Train a client with differential privacy using Opacus."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model and set global weights
    net = Net().to(device)
    net = ModuleValidator.fix(net)
    set_weights(net, global_weights)
    
    # Create dataloader
    subset = Subset(train_dataset, partition_indices)
    trainloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    if len(trainloader) == 0:
        return get_weights(net), len(subset), 0.0
    
    # Setup optimizer and privacy engine
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Wrap with Opacus PrivacyEngine
    privacy_engine = PrivacyEngine()
    net, optimizer, trainloader = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    
    # Train
    net.train()
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
    
    # Get epsilon for this client
    epsilon = privacy_engine.get_epsilon(delta=TARGET_DELTA)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    return get_weights(net._module), len(subset), epsilon


def federated_average(client_weights_list, sample_counts):
    """Compute weighted average of client weights."""
    total_samples = sum(sample_counts)
    
    avg_weights = []
    for layer_idx in range(len(client_weights_list[0])):
        layer_sum = np.zeros_like(client_weights_list[0][layer_idx])
        for client_idx, (weights, count) in enumerate(zip(client_weights_list, sample_counts)):
            layer_sum += weights[layer_idx] * (count / total_samples)
        avg_weights.append(layer_sum)
    
    return avg_weights


def evaluate(weights, test_dataset):
    """Evaluate model on test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    net = ModuleValidator.fix(net)
    set_weights(net, weights)
    
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total if total > 0 else 0.0


def run_experiment(name, noise_multiplier, max_grad_norm, is_iid, alpha, 
                   train_dataset, test_dataset):
    """Run a single FL experiment with DP."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"  Noise Multiplier: {noise_multiplier}")
    print(f"  Max Grad Norm: {max_grad_norm}")
    print(f"  IID: {is_iid}, Alpha: {alpha}")
    print(f"{'='*60}")
    
    # Partition data
    if is_iid:
        partitions = iid_partition(train_dataset, NUM_CLIENTS)
    else:
        partitions = dirichlet_partition(train_dataset, NUM_CLIENTS, alpha)
    
    # Print partition sizes
    sizes = [len(p) for p in partitions]
    print(f"  Partition sizes: {sizes}")
    
    # Initialize global model
    net = Net()
    net = ModuleValidator.fix(net)
    global_weights = get_weights(net)
    
    # Federated training
    all_epsilons = []
    
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n  Round {round_num}/{NUM_ROUNDS}")
        
        client_weights = []
        sample_counts = []
        round_epsilons = []
        
        for client_id in range(NUM_CLIENTS):
            weights, count, epsilon = train_client_dp(
                client_id, 
                partitions[client_id],
                train_dataset,
                global_weights,
                noise_multiplier,
                max_grad_norm,
                epochs=EPOCHS_PER_ROUND
            )
            client_weights.append(weights)
            sample_counts.append(count)
            round_epsilons.append(epsilon)
        
        # Aggregate
        global_weights = federated_average(client_weights, sample_counts)
        
        # Evaluate
        accuracy = evaluate(global_weights, test_dataset)
        avg_epsilon = np.mean(round_epsilons)
        all_epsilons.append(avg_epsilon)
        
        print(f"    Accuracy: {accuracy:.4f}, Avg ε: {avg_epsilon:.2f}")
    
    final_accuracy = evaluate(global_weights, test_dataset)
    final_epsilon = np.mean(all_epsilons)
    
    print(f"\n  FINAL: Accuracy={final_accuracy:.4f}, ε={final_epsilon:.2f}")
    
    return {
        "name": name,
        "accuracy": final_accuracy,
        "epsilon": final_epsilon,
        "noise_multiplier": noise_multiplier,
        "is_iid": is_iid,
        "alpha": alpha
    }


def main():
    print("="*70)
    print("DIFFERENTIAL PRIVACY EXPERIMENTS")
    print("IID vs Non-IID with Multiple Epsilon Settings")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Epochs per round: {EPOCHS_PER_ROUND}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Target δ: {TARGET_DELTA}")
    
    # Load data
    print("\nLoading CIFAR-10...")
    train_dataset, test_dataset = load_data()
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Experiment configurations
    # (name, noise_multiplier, max_grad_norm, is_iid, alpha)
    experiments = [
        # IID with different epsilon targets
        ("iid_eps2", 2.0, 1.0, True, 1.0),    # Strong privacy (ε ≈ 2)
        ("iid_eps4", 1.0, 1.0, True, 1.0),    # Moderate privacy (ε ≈ 4)
        ("iid_eps8", 0.5, 1.0, True, 1.0),    # Weak privacy (ε ≈ 8)
        
        # Non-IID with different epsilon targets
        ("noniid_eps2", 2.0, 1.0, False, 0.5),    # Strong privacy + Non-IID
        ("noniid_eps4", 1.0, 1.0, False, 0.5),    # Moderate privacy + Non-IID
        ("noniid_eps8", 0.5, 1.0, False, 0.5),    # Weak privacy + Non-IID
    ]
    
    results = []
    
    for name, noise_mult, grad_norm, is_iid, alpha in experiments:
        result = run_experiment(
            name, noise_mult, grad_norm, is_iid, alpha,
            train_dataset, test_dataset
        )
        results.append(result)
    
    # Print final summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Experiment':<15} {'Accuracy':>10} {'Epsilon':>10} {'IID':>6} {'Alpha':>7}")
    print("-"*50)
    
    for r in results:
        iid_str = "Yes" if r["is_iid"] else "No"
        print(f"{r['name']:<15} {r['accuracy']:>10.4f} {r['epsilon']:>10.2f} {iid_str:>6} {r['alpha']:>7.1f}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Compare IID vs Non-IID at same epsilon
    for eps_level in ["eps2", "eps4", "eps8"]:
        iid_result = next((r for r in results if r["name"] == f"iid_{eps_level}"), None)
        noniid_result = next((r for r in results if r["name"] == f"noniid_{eps_level}"), None)
        
        if iid_result and noniid_result:
            acc_diff = iid_result["accuracy"] - noniid_result["accuracy"]
            print(f"\n{eps_level.upper()}:")
            print(f"  IID Accuracy:     {iid_result['accuracy']:.4f} (ε={iid_result['epsilon']:.2f})")
            print(f"  Non-IID Accuracy: {noniid_result['accuracy']:.4f} (ε={noniid_result['epsilon']:.2f})")
            print(f"  Difference:       {acc_diff:+.4f} (IID advantage)")
    
    # Save results
    results_file = f"dp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print("""
    Expected patterns:
    1. Lower ε (stronger privacy) → Lower accuracy
       Due to more noise added during training
       
    2. Non-IID → Lower accuracy than IID at same ε
       Heterogeneous data distribution makes learning harder
       
    3. Privacy-utility tradeoff is MORE SEVERE for Non-IID
       Already difficult learning + DP noise = compounded challenge
    """)


if __name__ == "__main__":
    main()
