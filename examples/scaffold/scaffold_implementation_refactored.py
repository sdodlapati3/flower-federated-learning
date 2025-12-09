"""
SCAFFOLD: Stochastic Controlled Averaging for Federated Learning (Refactored)
==============================================================================

This script demonstrates SCAFFOLD implementation using the fl_research library.
SCAFFOLD (Karimireddy et al., 2020) addresses client drift through variance 
reduction using control variates.

Refactored from original scaffold_implementation.py to use:
- fl_research.models.CIFAR10CNN (replaces custom CNN class)
- fl_research.data.DirichletPartitioner (replaces custom function)
- fl_research.data.load_cifar10 (replaces manual data loading)
- fl_research.strategies.StandaloneSCAFFOLDClient, SCAFFOLDServer (from library)
- fl_research.utils (set_seed, get_device, MetricsTracker)

Key SCAFFOLD concepts:
  - Control variates: c_k (client) and c (global) correct for drift
  - Corrected gradient: y_k = ∇f_k(x) - c_k + c
  - Reduces variance in heterogeneous settings
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# Import from fl_research library
from fl_research.models import CIFAR10CNN, ModelRegistry
from fl_research.data import load_cifar10, DirichletPartitioner
from fl_research.strategies import StandaloneSCAFFOLDClient, SCAFFOLDServer, FedAvgStrategy
from fl_research.utils import set_seed, get_device, MetricsTracker


# ============================================================
# SCAFFOLD Training (Using Library Components)
# ============================================================

def run_scaffold_experiment(
    train_dataset,
    test_dataset,
    num_rounds: int = 20,
    num_clients: int = 10,
    clients_per_round: int = 5,
    local_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.1,
    alpha: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Run SCAFFOLD experiment using fl_research library components.
    
    Uses:
    - CIFAR10CNN from fl_research.models
    - DirichletPartitioner from fl_research.data
    - StandaloneSCAFFOLDClient, SCAFFOLDServer from fl_research.strategies
    """
    set_seed(seed)
    device = get_device()
    
    print(f"\n{'='*60}")
    print("SCAFFOLD Algorithm (Using fl_research library)")
    print(f"α = {alpha}, lr = {lr}")
    print(f"{'='*60}")
    
    # Use library partitioner
    partitioner = DirichletPartitioner(num_clients=num_clients, alpha=alpha)
    partitions = partitioner.partition(train_dataset)
    
    # Print partition statistics
    sizes = [len(p) for p in partitions]
    print(f"  Partition sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.0f}")
    
    # Create client dataloaders
    client_loaders = []
    for indices in partitions:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Initialize global model using library
    global_model = ModelRegistry.create('cifar10cnn').to(device)
    print(f"  Using model: CIFAR10CNN from ModelRegistry")
    
    # Initialize SCAFFOLD server from library
    server = SCAFFOLDServer(global_model, device)
    
    # Create SCAFFOLD clients from library
    clients = []
    for i, loader in enumerate(client_loaders):
        client = StandaloneSCAFFOLDClient(client_id=i, dataloader=loader, device=device)
        client.initialize_control_variate(global_model)
        clients.append(client)
    
    # Metrics tracking using library
    metrics = MetricsTracker()
    results = {'accuracies': [], 'losses': []}
    
    for round_num in range(1, num_rounds + 1):
        # Select clients for this round
        selected_ids = np.random.choice(num_clients, clients_per_round, replace=False)
        
        delta_weights_list = []
        delta_controls_list = []
        sample_counts = []
        
        for client_id in selected_ids:
            client = clients[client_id]
            
            # Create local model with global weights
            client_model = ModelRegistry.create('cifar10cnn').to(device)
            with torch.no_grad():
                for cp, gp in zip(client_model.parameters(), server.get_weights()):
                    cp.data.copy_(gp)
            
            # Client training with SCAFFOLD
            delta_weights, delta_control, count = client.train(
                client_model,
                server.global_control,
                local_epochs,
                lr
            )
            
            delta_weights_list.append(delta_weights)
            delta_controls_list.append(delta_control)
            sample_counts.append(count)
        
        # Server aggregation
        server.aggregate(
            delta_weights_list,
            delta_controls_list,
            sample_counts,
            num_clients
        )
        
        # Evaluate
        accuracy, loss = server.evaluate(testloader)
        results['accuracies'].append(accuracy)
        results['losses'].append(loss)
        
        # Log metrics
        metrics.log_round({
            'round': round_num,
            'accuracy': accuracy,
            'loss': loss
        })
        
        if round_num % 5 == 0 or round_num == 1:
            print(f"  Round {round_num:2d}: Accuracy={accuracy:.4f}, Loss={loss:.4f}")
    
    print(f"\nFinal Accuracy: {results['accuracies'][-1]:.4f}")
    
    return results


# ============================================================
# FedAvg Baseline (Using Library Components)
# ============================================================

def run_fedavg_experiment(
    train_dataset,
    test_dataset,
    num_rounds: int = 20,
    num_clients: int = 10,
    clients_per_round: int = 5,
    local_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.1,
    alpha: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Run FedAvg experiment for comparison using fl_research components.
    """
    set_seed(seed)
    device = get_device()
    
    print(f"\n{'='*60}")
    print("FedAvg Algorithm (Baseline - Using fl_research library)")
    print(f"α = {alpha}, lr = {lr}")
    print(f"{'='*60}")
    
    # Use library partitioner
    partitioner = DirichletPartitioner(num_clients=num_clients, alpha=alpha)
    partitions = partitioner.partition(train_dataset)
    
    # Create dataloaders
    client_loaders = []
    for indices in partitions:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Global model using library
    global_model = ModelRegistry.create('cifar10cnn').to(device)
    
    # Metrics
    metrics = MetricsTracker()
    results = {'accuracies': [], 'losses': []}
    
    for round_num in range(1, num_rounds + 1):
        selected_ids = np.random.choice(num_clients, clients_per_round, replace=False)
        
        all_weights = []
        sample_counts = []
        
        for client_id in selected_ids:
            # Create local model with global weights
            local_model = ModelRegistry.create('cifar10cnn').to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            # Standard local training (no control variates)
            local_model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
            
            for epoch in range(local_epochs):
                for images, labels in client_loaders[client_id]:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            weights = [p.data.clone() for p in local_model.parameters()]
            all_weights.append(weights)
            sample_counts.append(len(client_loaders[client_id].dataset))
        
        # FedAvg aggregation
        total = sum(sample_counts)
        with torch.no_grad():
            for layer_idx, param in enumerate(global_model.parameters()):
                weighted_avg = torch.zeros_like(param)
                for weights, count in zip(all_weights, sample_counts):
                    weighted_avg += (count / total) * weights[layer_idx]
                param.data.copy_(weighted_avg)
        
        # Evaluate
        global_model.eval()
        correct = 0
        total_test = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item() * labels.size(0)
        
        accuracy = correct / total_test
        avg_loss = total_loss / total_test
        
        results['accuracies'].append(accuracy)
        results['losses'].append(avg_loss)
        
        metrics.log_round({
            'round': round_num,
            'accuracy': accuracy,
            'loss': avg_loss
        })
        
        if round_num % 5 == 0 or round_num == 1:
            print(f"  Round {round_num:2d}: Accuracy={accuracy:.4f}, Loss={avg_loss:.4f}")
    
    print(f"\nFinal Accuracy: {results['accuracies'][-1]:.4f}")
    
    return results


# ============================================================
# Main Comparison
# ============================================================

def main():
    """Compare SCAFFOLD vs FedAvg using fl_research library."""
    
    print("=" * 70)
    print("SCAFFOLD vs FedAvg Comparison (Refactored)")
    print("Using fl_research library components")
    print("=" * 70)
    
    # Load data using library
    print("\nLoading CIFAR-10 using fl_research.data.load_cifar10...")
    train_dataset, test_dataset = load_cifar10()
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # List available models
    print(f"\nAvailable models: {ModelRegistry.list()}")
    
    params = {
        'num_rounds': 25,
        'num_clients': 10,
        'clients_per_round': 5,
        'local_epochs': 5,
        'batch_size': 32,
        'lr': 0.1,
        'seed': 42
    }
    
    alphas = [0.1, 0.5]  # Different heterogeneity levels
    
    all_results = {}
    
    for alpha in alphas:
        print(f"\n{'#'*70}")
        print(f"Heterogeneity Level: α = {alpha}")
        print(f"{'#'*70}")
        
        # Run both algorithms
        scaffold_results = run_scaffold_experiment(
            train_dataset, test_dataset, alpha=alpha, **params
        )
        
        fedavg_results = run_fedavg_experiment(
            train_dataset, test_dataset, alpha=alpha, **params
        )
        
        all_results[f'scaffold_alpha{alpha}'] = scaffold_results
        all_results[f'fedavg_alpha{alpha}'] = fedavg_results
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        
        scaffold = all_results[f'scaffold_alpha{alpha}']
        fedavg = all_results[f'fedavg_alpha{alpha}']
        
        rounds = range(1, len(scaffold['accuracies']) + 1)
        
        ax.plot(rounds, scaffold['accuracies'], 'b-', label='SCAFFOLD', linewidth=2)
        ax.plot(rounds, fedavg['accuracies'], 'r--', label='FedAvg', linewidth=2)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'α = {alpha}')
        ax.legend()
        ax.grid(True)
    
    plt.suptitle('SCAFFOLD vs FedAvg: Variance Reduction in Heterogeneous FL (Refactored)')
    plt.tight_layout()
    plt.savefig('scaffold_vs_fedavg_refactored.png', dpi=150)
    print(f"\nPlot saved to: scaffold_vs_fedavg_refactored.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Algorithm':<15} {'α':<8} {'Final Acc':<12} {'Best Acc':<12}")
    print("-" * 50)
    
    for key, result in all_results.items():
        parts = key.split('_')
        alg = parts[0].upper()
        alpha = parts[1].replace('alpha', '')
        final = result['accuracies'][-1]
        best = max(result['accuracies'])
        print(f"{alg:<15} {alpha:<8} {final:<12.4f} {best:<12.4f}")
    
    print("\n" + "=" * 70)
    print("REFACTORING BENEFITS")
    print("=" * 70)
    print("""
    This refactored version uses fl_research library components:
    
    1. Models:
       - CIFAR10CNN: Standard CNN for CIFAR-10
       - ModelRegistry: Factory pattern for model creation
       
    2. Data Partitioning:
       - DirichletPartitioner: Non-IID with configurable alpha
       
    3. Strategies (from fl_research.strategies):
       - StandaloneSCAFFOLDClient: Client with control variate management
       - SCAFFOLDServer: Server with global control variate aggregation
       
    4. Utilities:
       - set_seed: Reproducibility
       - get_device: Automatic device detection
       - MetricsTracker: Experiment logging
       
    Original: ~675 lines (including FedAvg baseline)
    Refactored: ~350 lines (cleaner, more documented)
    
    Key improvements:
    - Reusable SCAFFOLD components
    - Consistent interface with FedProx strategy
    - Shared model registry across experiments
    """)
    
    print("\n" + "=" * 70)
    print("SCAFFOLD ALGORITHM SUMMARY")
    print("=" * 70)
    print("""
    1. Control variates reduce client drift:
       - c_k: Client control variate (tracks expected gradient)
       - c: Global control variate (average of client controls)
       
    2. Corrected gradient: ∇f_k(x) - c_k + c
       - Subtracts client bias: -c_k
       - Adds global direction: +c
       
    3. Control variate update (Option I):
       c_k+ = c_k - c + (x - y) / (K * η)
       
    4. When to use SCAFFOLD:
       - Very heterogeneous data (low α)
       - Many local updates
       - Full or high client participation
    """)


if __name__ == "__main__":
    main()
