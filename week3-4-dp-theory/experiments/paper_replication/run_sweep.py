"""
Paper Replication Framework: DP in Federated Learning
======================================================

Replicating key findings from:
"A Differential Privacy Approach in Federated Learning" (Tayyeh et al., 2024)

Key Hyperparameters to Sweep:
- ε (epsilon): Privacy budget [1, 2, 4, 8, 16]
- C (clipping norm): Gradient clipping [0.5, 1.0, 2.0, 5.0]
- K (clients per round): Client fraction [2, 5, 10]
- Data split: IID vs Non-IID (Dirichlet α)

Metrics:
- Final accuracy
- Convergence speed
- Privacy budget consumption
- Gradient norm statistics
"""

import json
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# Configuration
# ============================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    epsilon_target: float
    noise_multiplier: float
    clipping_norm: float
    clients_per_round: int
    total_clients: int
    num_rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    is_iid: bool
    dirichlet_alpha: float
    target_delta: float = 1e-5
    seed: int = 42


# Default sweep configurations
EPSILON_VALUES = [1.0, 2.0, 4.0, 8.0, 16.0]
CLIPPING_NORMS = [0.5, 1.0, 2.0, 5.0]
CLIENTS_PER_ROUND = [2, 5, 10]
DIRICHLET_ALPHAS = [0.1, 0.5, 1.0, 10.0]  # 0.1 = very non-IID, 10.0 ≈ IID


def noise_multiplier_for_epsilon(target_epsilon: float, 
                                  num_steps: int,
                                  sampling_rate: float,
                                  delta: float = 1e-5) -> float:
    """
    Estimate noise multiplier needed to achieve target epsilon.
    
    This is a rough approximation. In practice, use Opacus's 
    privacy analysis tools for precise calibration.
    """
    # Rough formula: ε ≈ q × √(T × log(1/δ)) / σ
    # σ ≈ q × √(T × log(1/δ)) / ε
    
    q = sampling_rate
    T = num_steps
    
    sigma = q * np.sqrt(T * np.log(1 / delta)) / target_epsilon
    
    # Clip to reasonable range
    return max(0.1, min(10.0, sigma))


# ============================================================
# Model
# ============================================================

class CNN(nn.Module):
    """CNN for CIFAR-10 - Opacus compatible."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)  # GroupNorm for Opacus
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.gn3 = nn.GroupNorm(8, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = self.pool(F.relu(self.gn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ============================================================
# Data Partitioning
# ============================================================

def iid_partition(dataset, num_clients: int) -> List[List[int]]:
    """Create IID partition."""
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    partition_size = len(indices) // num_clients
    
    partitions = []
    for i in range(num_clients):
        start = i * partition_size
        end = start + partition_size if i < num_clients - 1 else len(indices)
        partitions.append(indices[start:end])
    
    return partitions


def dirichlet_partition(dataset, num_clients: int, alpha: float) -> List[List[int]]:
    """Create non-IID partition using Dirichlet distribution."""
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = 10
    
    partition_indices = [[] for _ in range(num_clients)]
    
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0].tolist()
        np.random.shuffle(class_indices)
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = proportions / proportions.sum()
        
        cumsum = np.cumsum(proportions) * len(class_indices)
        cumsum = [0] + cumsum.astype(int).tolist()
        
        for i in range(num_clients):
            start, end = cumsum[i], cumsum[i + 1]
            partition_indices[i].extend(class_indices[start:end])
    
    for indices in partition_indices:
        np.random.shuffle(indices)
    
    return partition_indices


# ============================================================
# Training Functions
# ============================================================

def get_weights(net) -> List[np.ndarray]:
    """Get model weights as numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, weights: List[np.ndarray]):
    """Set model weights from numpy arrays."""
    params_dict = zip(net.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train_client_dp(
    client_id: int,
    partition_indices: List[int],
    train_dataset,
    global_weights: List[np.ndarray],
    config: ExperimentConfig,
    device: torch.device
) -> Tuple[List[np.ndarray], int, float, Dict]:
    """
    Train a client with differential privacy.
    
    Returns:
        weights: Updated model weights
        num_samples: Number of samples used
        epsilon: Privacy budget consumed
        stats: Training statistics (grad norms, loss, etc.)
    """
    # Create model
    net = CNN().to(device)
    net = ModuleValidator.fix(net)
    set_weights(net, global_weights)
    
    # Create dataloader
    subset = Subset(train_dataset, partition_indices)
    trainloader = DataLoader(
        subset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    if len(trainloader) == 0:
        return get_weights(net), len(subset), 0.0, {}
    
    # Setup optimizer
    optimizer = torch.optim.SGD(
        net.parameters(), 
        lr=config.learning_rate, 
        momentum=0.9
    )
    criterion = nn.CrossEntropyLoss()
    
    # Wrap with Opacus
    privacy_engine = PrivacyEngine()
    net, optimizer, trainloader = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=config.noise_multiplier,
        max_grad_norm=config.clipping_norm,
    )
    
    # Training stats
    stats = {
        'grad_norms_before_clip': [],
        'grad_norms_after_clip': [],
        'losses': []
    }
    
    # Train
    net.train()
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(config.local_epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Record stats (before optimizer step which clips)
            total_norm = 0.0
            for p in net.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            stats['grad_norms_before_clip'].append(np.sqrt(total_norm))
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            stats['losses'].append(loss.item())
    
    # Get epsilon
    epsilon = privacy_engine.get_epsilon(delta=config.target_delta)
    
    return get_weights(net._module), len(subset), epsilon, stats


def federated_average(
    client_weights: List[List[np.ndarray]], 
    sample_counts: List[int]
) -> List[np.ndarray]:
    """Weighted average of client weights."""
    total_samples = sum(sample_counts)
    
    avg_weights = []
    for layer_idx in range(len(client_weights[0])):
        layer_sum = np.zeros_like(client_weights[0][layer_idx])
        for weights, count in zip(client_weights, sample_counts):
            layer_sum += weights[layer_idx] * (count / total_samples)
        avg_weights.append(layer_sum)
    
    return avg_weights


def evaluate(
    weights: List[np.ndarray], 
    test_dataset, 
    device: torch.device,
    batch_size: int = 128
) -> Tuple[float, float]:
    """Evaluate model on test set."""
    net = CNN().to(device)
    net = ModuleValidator.fix(net)
    set_weights(net, weights)
    
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    net.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    
    return accuracy, avg_loss


# ============================================================
# Run Single Experiment
# ============================================================

def run_experiment(config: ExperimentConfig, train_dataset, test_dataset) -> Dict:
    """Run a single FL experiment with DP."""
    
    print(f"\n{'='*70}")
    print(f"Experiment: {config.name}")
    print(f"  ε target: {config.epsilon_target}, noise: {config.noise_multiplier:.2f}")
    print(f"  Clipping: {config.clipping_norm}, K: {config.clients_per_round}")
    print(f"  IID: {config.is_iid}, α: {config.dirichlet_alpha}")
    print(f"{'='*70}")
    
    # Set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Partition data
    if config.is_iid:
        partitions = iid_partition(train_dataset, config.total_clients)
    else:
        partitions = dirichlet_partition(
            train_dataset, config.total_clients, config.dirichlet_alpha
        )
    
    # Initialize model
    net = CNN()
    net = ModuleValidator.fix(net)
    global_weights = get_weights(net)
    
    # Results tracking
    results = {
        'config': asdict(config),
        'round_metrics': [],
        'final_accuracy': 0.0,
        'final_epsilon': 0.0,
        'grad_norm_stats': {'mean': [], 'std': [], 'max': []}
    }
    
    all_epsilons = []
    
    for round_num in range(1, config.num_rounds + 1):
        # Select clients for this round
        selected_clients = np.random.choice(
            config.total_clients, 
            config.clients_per_round, 
            replace=False
        )
        
        client_weights = []
        sample_counts = []
        round_epsilons = []
        round_grad_norms = []
        
        for client_id in selected_clients:
            weights, count, epsilon, stats = train_client_dp(
                client_id,
                partitions[client_id],
                train_dataset,
                global_weights,
                config,
                device
            )
            client_weights.append(weights)
            sample_counts.append(count)
            round_epsilons.append(epsilon)
            
            if 'grad_norms_before_clip' in stats:
                round_grad_norms.extend(stats['grad_norms_before_clip'])
        
        # Aggregate
        global_weights = federated_average(client_weights, sample_counts)
        
        # Evaluate
        accuracy, loss = evaluate(global_weights, test_dataset, device)
        avg_epsilon = np.mean(round_epsilons)
        all_epsilons.append(avg_epsilon)
        
        # Record metrics
        round_metrics = {
            'round': round_num,
            'accuracy': accuracy,
            'loss': loss,
            'epsilon': avg_epsilon,
            'num_clients': len(selected_clients)
        }
        results['round_metrics'].append(round_metrics)
        
        # Gradient norm stats
        if round_grad_norms:
            results['grad_norm_stats']['mean'].append(np.mean(round_grad_norms))
            results['grad_norm_stats']['std'].append(np.std(round_grad_norms))
            results['grad_norm_stats']['max'].append(np.max(round_grad_norms))
        
        print(f"  Round {round_num}/{config.num_rounds}: "
              f"Acc={accuracy:.4f}, Loss={loss:.4f}, ε={avg_epsilon:.2f}")
    
    # Final results
    final_accuracy, _ = evaluate(global_weights, test_dataset, device)
    results['final_accuracy'] = final_accuracy
    results['final_epsilon'] = np.mean(all_epsilons)
    
    print(f"\n  FINAL: Accuracy={final_accuracy:.4f}, ε={results['final_epsilon']:.2f}")
    
    return results


# ============================================================
# Hyperparameter Sweep
# ============================================================

def generate_sweep_configs(
    base_config: Dict,
    sweep_epsilon: bool = True,
    sweep_clipping: bool = True,
    sweep_clients: bool = True,
    sweep_noniid: bool = True
) -> List[ExperimentConfig]:
    """Generate all configurations for hyperparameter sweep."""
    
    configs = []
    
    # Base parameters
    num_rounds = base_config.get('num_rounds', 10)
    local_epochs = base_config.get('local_epochs', 1)
    batch_size = base_config.get('batch_size', 32)
    learning_rate = base_config.get('learning_rate', 0.01)
    total_clients = base_config.get('total_clients', 10)
    
    # Sweep dimensions
    epsilons = EPSILON_VALUES if sweep_epsilon else [4.0]
    clip_norms = CLIPPING_NORMS if sweep_clipping else [1.0]
    k_values = CLIENTS_PER_ROUND if sweep_clients else [5]
    
    # IID/Non-IID
    if sweep_noniid:
        data_splits = [
            (True, 1.0, "iid"),
            (False, 0.5, "noniid_0.5"),
            (False, 0.1, "noniid_0.1")
        ]
    else:
        data_splits = [(True, 1.0, "iid")]
    
    # Generate all combinations
    for eps, clip, k, (is_iid, alpha, split_name) in itertools.product(
        epsilons, clip_norms, k_values, data_splits
    ):
        name = f"eps{eps}_clip{clip}_k{k}_{split_name}"
        
        # Estimate noise multiplier for target epsilon
        steps_per_round = 50000 // (total_clients * batch_size) * local_epochs
        total_steps = num_rounds * steps_per_round * k
        sampling_rate = batch_size / (50000 // total_clients)
        
        noise_mult = noise_multiplier_for_epsilon(
            eps, total_steps, sampling_rate
        )
        
        config = ExperimentConfig(
            name=name,
            epsilon_target=eps,
            noise_multiplier=noise_mult,
            clipping_norm=clip,
            clients_per_round=k,
            total_clients=total_clients,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            is_iid=is_iid,
            dirichlet_alpha=alpha
        )
        configs.append(config)
    
    return configs


# ============================================================
# Main
# ============================================================

def main():
    """Run paper replication experiments."""
    
    print("=" * 70)
    print("PAPER REPLICATION: DP in Federated Learning")
    print("Tayyeh et al., 2024")
    print("=" * 70)
    
    # Load data
    print("\nLoading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Base configuration
    base_config = {
        'num_rounds': 5,
        'local_epochs': 1,
        'batch_size': 32,
        'learning_rate': 0.01,
        'total_clients': 10
    }
    
    # Generate configs for a focused sweep
    # For quick testing: just epsilon and IID/non-IID
    configs = generate_sweep_configs(
        base_config,
        sweep_epsilon=True,
        sweep_clipping=False,  # Fix clipping for now
        sweep_clients=False,   # Fix K for now
        sweep_noniid=True
    )
    
    print(f"\nRunning {len(configs)} experiments...")
    
    # Run experiments
    all_results = []
    
    for config in configs:
        result = run_experiment(config, train_dataset, test_dataset)
        all_results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"sweep_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Config':<30} {'Accuracy':>10} {'Epsilon':>10}")
    print("-" * 52)
    
    for result in all_results:
        name = result['config']['name']
        acc = result['final_accuracy']
        eps = result['final_epsilon']
        print(f"{name:<30} {acc:>10.4f} {eps:>10.2f}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Group by epsilon
    eps_groups = {}
    for result in all_results:
        eps = result['config']['epsilon_target']
        is_iid = result['config']['is_iid']
        key = f"eps{eps}"
        
        if key not in eps_groups:
            eps_groups[key] = {'iid': None, 'noniid': []}
        
        if is_iid:
            eps_groups[key]['iid'] = result['final_accuracy']
        else:
            eps_groups[key]['noniid'].append(result['final_accuracy'])
    
    print("\nIID vs Non-IID Comparison:")
    for key in sorted(eps_groups.keys()):
        iid_acc = eps_groups[key]['iid']
        noniid_accs = eps_groups[key]['noniid']
        if iid_acc and noniid_accs:
            noniid_avg = np.mean(noniid_accs)
            diff = iid_acc - noniid_avg
            print(f"  {key}: IID={iid_acc:.4f}, Non-IID avg={noniid_avg:.4f}, Diff={diff:+.4f}")


if __name__ == "__main__":
    main()
