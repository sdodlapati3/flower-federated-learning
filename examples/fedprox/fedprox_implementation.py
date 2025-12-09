"""
FedProx: Federated Optimization in Heterogeneous Networks
==========================================================

This implements FedProx (Li et al., 2020) which adds a proximal term
to handle heterogeneous clients and partial work.

Key idea: Instead of standard local objective f_k(w),
optimize: f_k(w) + (μ/2)||w - w_global||²

This proximal term:
1. Keeps local updates close to global model
2. Reduces client drift
3. Handles stragglers (partial work is still useful)
"""

from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# Model
# ============================================================

class CNN(nn.Module):
    """Simple CNN for CIFAR-10."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ============================================================
# FedProx Optimizer
# ============================================================

class FedProxOptimizer(torch.optim.Optimizer):
    """
    FedProx optimizer that adds proximal term to gradients.
    
    Instead of: w = w - η∇f(w)
    We do:      w = w - η(∇f(w) + μ(w - w_global))
    
    This is equivalent to optimizing:
        min_w f(w) + (μ/2)||w - w_global||²
    """
    
    def __init__(self, params, lr: float = 0.01, mu: float = 0.01):
        """
        Args:
            params: Model parameters
            lr: Learning rate
            mu: Proximal term coefficient (higher = more regularization)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid mu: {mu}")
            
        defaults = dict(lr=lr, mu=mu)
        super().__init__(params, defaults)
        
        # Store global parameters for proximal term
        self.global_params = None
    
    def set_global_params(self, global_params: List[torch.Tensor]):
        """Set the global model parameters for proximal term."""
        self.global_params = [p.clone().detach() for p in global_params]
    
    def step(self, closure=None):
        """Perform a single optimization step with proximal term."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']
            
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                # Standard gradient
                d_p = p.grad.data
                
                # Add proximal term: μ(w - w_global)
                if self.global_params is not None and idx < len(self.global_params):
                    d_p = d_p + mu * (p.data - self.global_params[idx])
                
                # Update: w = w - η * (∇f + μ(w - w_global))
                p.data.add_(d_p, alpha=-lr)
        
        return loss


# ============================================================
# FedProx Client Training
# ============================================================

def train_fedprox_client(
    model: nn.Module,
    trainloader: DataLoader,
    global_weights: List[torch.Tensor],
    epochs: int,
    lr: float,
    mu: float,
    device: torch.device
) -> Tuple[List[np.ndarray], int, float]:
    """
    Train a client using FedProx.
    
    Args:
        model: The neural network model
        trainloader: Client's local data
        global_weights: Global model weights for proximal term
        epochs: Number of local epochs
        lr: Learning rate
        mu: Proximal term coefficient
        device: Device to train on
    
    Returns:
        weights: Updated model weights
        num_samples: Number of training samples
        loss: Average training loss
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Create FedProx optimizer
    optimizer = FedProxOptimizer(model.parameters(), lr=lr, mu=mu)
    optimizer.set_global_params(global_weights)
    
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    # Get weights
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    return weights, len(trainloader.dataset), avg_loss


def train_fedavg_client(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device
) -> Tuple[List[np.ndarray], int, float]:
    """
    Train a client using standard FedAvg (no proximal term).
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    return weights, len(trainloader.dataset), avg_loss


# ============================================================
# Utility Functions
# ============================================================

def get_weights(model: nn.Module) -> List[np.ndarray]:
    """Get model weights as numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights: List[np.ndarray]):
    """Set model weights from numpy arrays."""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_weights_as_tensors(model: nn.Module) -> List[torch.Tensor]:
    """Get model weights as tensors."""
    return [val.clone() for _, val in model.state_dict().items()]


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
    model: nn.Module, 
    testloader: DataLoader, 
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    
    return accuracy, avg_loss


def dirichlet_partition(
    dataset, 
    num_clients: int, 
    alpha: float
) -> List[List[int]]:
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
# Main Experiment: FedAvg vs FedProx
# ============================================================

def run_experiment(
    algorithm: str,  # "fedavg" or "fedprox"
    train_dataset,
    test_dataset,
    num_rounds: int = 20,
    num_clients: int = 10,
    clients_per_round: int = 5,
    local_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.01,
    mu: float = 0.01,  # FedProx proximal coefficient
    alpha: float = 0.1,  # Dirichlet alpha (lower = more heterogeneous)
    seed: int = 42
) -> Dict:
    """Run FL experiment with specified algorithm."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Dirichlet α: {alpha} (lower = more heterogeneous)")
    if algorithm == "fedprox":
        print(f"Proximal μ: {mu}")
    print(f"{'='*60}")
    
    # Partition data (non-IID)
    partitions = dirichlet_partition(train_dataset, num_clients, alpha)
    
    # Print partition statistics
    sizes = [len(p) for p in partitions]
    print(f"Partition sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.0f}")
    
    # Create dataloaders
    client_loaders = []
    for indices in partitions:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Initialize global model
    global_model = CNN().to(device)
    global_weights = get_weights(global_model)
    
    # Training loop
    results = {
        'algorithm': algorithm,
        'alpha': alpha,
        'mu': mu if algorithm == 'fedprox' else 0,
        'accuracies': [],
        'losses': []
    }
    
    for round_num in range(1, num_rounds + 1):
        # Select clients
        selected = np.random.choice(num_clients, clients_per_round, replace=False)
        
        client_weights_list = []
        sample_counts = []
        round_losses = []
        
        for client_id in selected:
            # Create fresh model with global weights
            model = CNN().to(device)
            set_weights(model, global_weights)
            
            if algorithm == "fedprox":
                global_tensors = get_weights_as_tensors(model)
                weights, count, loss = train_fedprox_client(
                    model, client_loaders[client_id],
                    global_tensors, local_epochs, lr, mu, device
                )
            else:  # fedavg
                weights, count, loss = train_fedavg_client(
                    model, client_loaders[client_id],
                    local_epochs, lr, device
                )
            
            client_weights_list.append(weights)
            sample_counts.append(count)
            round_losses.append(loss)
        
        # Aggregate
        global_weights = federated_average(client_weights_list, sample_counts)
        set_weights(global_model, global_weights)
        
        # Evaluate
        accuracy, test_loss = evaluate(global_model, testloader, device)
        
        results['accuracies'].append(accuracy)
        results['losses'].append(test_loss)
        
        if round_num % 5 == 0 or round_num == 1:
            print(f"  Round {round_num:2d}: Accuracy={accuracy:.4f}, "
                  f"Loss={test_loss:.4f}, Train Loss={np.mean(round_losses):.4f}")
    
    print(f"\nFinal Accuracy: {results['accuracies'][-1]:.4f}")
    
    return results


def main():
    """Compare FedAvg and FedProx on heterogeneous data."""
    
    print("=" * 70)
    print("FedAvg vs FedProx Comparison")
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
    
    # Experiment parameters
    params = {
        'num_rounds': 20,
        'num_clients': 10,
        'clients_per_round': 5,
        'local_epochs': 5,
        'batch_size': 32,
        'lr': 0.01,
        'seed': 42
    }
    
    # Different heterogeneity levels
    alphas = [0.1, 0.5, 1.0]  # Lower = more heterogeneous
    
    all_results = []
    
    for alpha in alphas:
        print(f"\n{'#'*70}")
        print(f"Heterogeneity Level: α = {alpha}")
        print(f"{'#'*70}")
        
        # FedAvg
        fedavg_results = run_experiment(
            "fedavg", train_dataset, test_dataset,
            alpha=alpha, **params
        )
        all_results.append(fedavg_results)
        
        # FedProx with different μ values
        for mu in [0.01, 0.1]:
            fedprox_results = run_experiment(
                "fedprox", train_dataset, test_dataset,
                alpha=alpha, mu=mu, **params
            )
            all_results.append(fedprox_results)
    
    # Plot comparison
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        
        for result in all_results:
            if result['alpha'] == alpha:
                label = result['algorithm']
                if result['algorithm'] == 'fedprox':
                    label += f" (μ={result['mu']})"
                ax.plot(result['accuracies'], label=label, linewidth=2)
        
        ax.set_xlabel('Round')
        ax.set_ylabel('Test Accuracy')
        ax.set_title(f'α = {alpha} ({"High" if alpha > 0.5 else "Low"} Heterogeneity)')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('fedavg_vs_fedprox.png', dpi=150)
    plt.show()
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Algorithm':<20} {'α':<8} {'μ':<8} {'Final Acc':<12}")
    print("-" * 48)
    
    for result in all_results:
        alg = result['algorithm']
        if result['algorithm'] == 'fedprox':
            alg += f" (μ={result['mu']})"
        print(f"{alg:<20} {result['alpha']:<8} {result['mu']:<8.2f} "
              f"{result['accuracies'][-1]:<12.4f}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
    1. FedProx adds proximal term: f_k(w) + (μ/2)||w - w_global||²
       This keeps local updates close to global model
       
    2. Benefits of FedProx:
       - Reduces client drift in heterogeneous settings
       - Handles stragglers (partial work is still useful)
       - More stable convergence
       
    3. When to use FedProx:
       - Highly heterogeneous data (low α)
       - Unreliable clients that may drop out
       - Large number of local epochs
       
    4. μ selection:
       - Higher μ = more regularization, less drift
       - Too high = slow convergence (too conservative)
       - Typical values: 0.001 to 0.1
    """)


if __name__ == "__main__":
    main()
