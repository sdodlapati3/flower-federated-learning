"""
SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
=================================================================

SCAFFOLD (Karimireddy et al., 2020) addresses client drift through
variance reduction using control variates.

Key idea: Track per-client and global control variates that correct
for the drift in local updates.

Update rule:
  y_k = ∇f_k(x) - c_k + c  (corrected gradient)
  
where c_k is client control variate and c is global control variate.
"""

from collections import OrderedDict
from typing import List, Tuple, Dict
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
# SCAFFOLD Client
# ============================================================

class SCAFFOLDClient:
    """
    SCAFFOLD client that maintains local control variate.
    
    The control variate tracks the "expected" gradient, allowing
    correction for client drift.
    """
    
    def __init__(self, client_id: int, dataloader: DataLoader, device: torch.device):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        
        # Control variate c_k - initialized to zero
        # Will be updated after each training round
        self.control_variate = None
    
    def initialize_control_variate(self, model: nn.Module):
        """Initialize control variate to zeros matching model shape."""
        self.control_variate = [
            torch.zeros_like(p, device=self.device) 
            for p in model.parameters()
        ]
    
    def train(
        self,
        model: nn.Module,
        global_control: List[torch.Tensor],
        epochs: int,
        lr: float
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """
        Train using SCAFFOLD algorithm.
        
        Args:
            model: Model with current global weights
            global_control: Global control variate c
            epochs: Local epochs
            lr: Learning rate
            
        Returns:
            delta_weights: Change in weights (y - x)
            delta_control: Change in control variate (c_k+ - c_k)
            num_samples: Number of samples
        """
        if self.control_variate is None:
            self.initialize_control_variate(model)
        
        model.train()
        criterion = nn.CrossEntropyLoss()
        
        # Store initial weights
        initial_weights = [p.data.clone() for p in model.parameters()]
        
        # Training with corrected gradients
        num_batches = 0
        
        for epoch in range(epochs):
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Zero gradients
                model.zero_grad()
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # SCAFFOLD gradient correction:
                # y_k = ∇f_k(x) - c_k + c
                for param, c_k, c in zip(
                    model.parameters(), 
                    self.control_variate,
                    global_control
                ):
                    if param.grad is not None:
                        # Corrected gradient: grad - c_k + c
                        param.grad.data = param.grad.data - c_k + c
                
                # Update with corrected gradient
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.data -= lr * param.grad.data
                
                num_batches += 1
        
        # Compute delta weights: y - x
        delta_weights = [
            p.data - init_w 
            for p, init_w in zip(model.parameters(), initial_weights)
        ]
        
        # Update control variate using Option I from paper:
        # c_k+ = c_k - c + (x - y) / (K * η)
        # where K = total gradient steps, η = learning rate
        total_steps = epochs * len(self.dataloader)
        
        new_control = []
        for c_k, c, delta in zip(self.control_variate, global_control, delta_weights):
            c_new = c_k - c - delta / (total_steps * lr)
            new_control.append(c_new)
        
        # Delta control: c_k+ - c_k
        delta_control = [
            c_new - c_k 
            for c_new, c_k in zip(new_control, self.control_variate)
        ]
        
        # Update stored control variate
        self.control_variate = new_control
        
        return delta_weights, delta_control, len(self.dataloader.dataset)


# ============================================================
# SCAFFOLD Server
# ============================================================

class SCAFFOLDServer:
    """
    SCAFFOLD server that maintains global model and control variate.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.device = device
        self.model = model.to(device)
        
        # Global control variate c - average of client controls
        self.global_control = [
            torch.zeros_like(p, device=device) 
            for p in model.parameters()
        ]
    
    def get_weights(self) -> List[torch.Tensor]:
        """Get current global model weights."""
        return [p.data.clone() for p in self.model.parameters()]
    
    def set_weights(self, weights: List[torch.Tensor]):
        """Set global model weights."""
        with torch.no_grad():
            for p, w in zip(self.model.parameters(), weights):
                p.data.copy_(w)
    
    def aggregate(
        self,
        delta_weights_list: List[List[torch.Tensor]],
        delta_controls_list: List[List[torch.Tensor]],
        sample_counts: List[int],
        total_clients: int
    ):
        """
        Aggregate client updates.
        
        Server update:
        x = x + Δx  where Δx = (1/|S|) Σ (y_k - x)
        c = c + Δc  where Δc = (|S|/N) * (1/|S|) Σ (c_k+ - c_k)
        """
        num_participating = len(delta_weights_list)
        total_samples = sum(sample_counts)
        
        # Update global model: x = x + avg(delta_weights)
        with torch.no_grad():
            for layer_idx, param in enumerate(self.model.parameters()):
                weighted_delta = torch.zeros_like(param)
                
                for deltas, count in zip(delta_weights_list, sample_counts):
                    weight = count / total_samples
                    weighted_delta += weight * deltas[layer_idx]
                
                param.data.add_(weighted_delta)
        
        # Update global control: c = c + (|S|/N) * avg(delta_controls)
        scaling = num_participating / total_clients
        
        for layer_idx in range(len(self.global_control)):
            avg_delta_control = torch.zeros_like(self.global_control[layer_idx])
            
            for delta_controls in delta_controls_list:
                avg_delta_control += delta_controls[layer_idx] / num_participating
            
            self.global_control[layer_idx].add_(scaling * avg_delta_control)
    
    def evaluate(self, testloader: DataLoader) -> Tuple[float, float]:
        """Evaluate global model."""
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item() * labels.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        
        return accuracy, avg_loss


# ============================================================
# Standard FedAvg for Comparison
# ============================================================

def train_fedavg(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device
) -> Tuple[List[torch.Tensor], int]:
    """Train using standard FedAvg (no control variates)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    weights = [p.data.clone() for p in model.parameters()]
    return weights, len(dataloader.dataset)


# ============================================================
# Utility Functions
# ============================================================

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


def copy_model(model: nn.Module) -> nn.Module:
    """Create a copy of the model."""
    new_model = type(model)()
    new_model.load_state_dict(model.state_dict())
    return new_model


# ============================================================
# Main Experiment
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
    alpha: float = 0.1,  # Dirichlet
    seed: int = 42
) -> Dict:
    """Run SCAFFOLD experiment."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print("SCAFFOLD Algorithm")
    print(f"α = {alpha}, lr = {lr}")
    print(f"{'='*60}")
    
    # Create partitions
    partitions = dirichlet_partition(train_dataset, num_clients, alpha)
    
    # Create clients
    clients = []
    for i, indices in enumerate(partitions):
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client = SCAFFOLDClient(i, loader, device)
        clients.append(client)
    
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Initialize server
    global_model = CNN().to(device)
    server = SCAFFOLDServer(global_model, device)
    
    # Initialize client control variates
    for client in clients:
        client.initialize_control_variate(global_model)
    
    results = {'accuracies': [], 'losses': []}
    
    for round_num in range(1, num_rounds + 1):
        # Select clients
        selected_ids = np.random.choice(num_clients, clients_per_round, replace=False)
        
        delta_weights_list = []
        delta_controls_list = []
        sample_counts = []
        
        for client_id in selected_ids:
            client = clients[client_id]
            
            # Give client the global model
            client_model = copy_model(global_model).to(device)
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
        
        if round_num % 5 == 0 or round_num == 1:
            print(f"  Round {round_num:2d}: Accuracy={accuracy:.4f}, Loss={loss:.4f}")
    
    print(f"\nFinal Accuracy: {results['accuracies'][-1]:.4f}")
    
    return results


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
    """Run FedAvg experiment for comparison."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print("FedAvg Algorithm (Baseline)")
    print(f"α = {alpha}, lr = {lr}")
    print(f"{'='*60}")
    
    # Create partitions
    partitions = dirichlet_partition(train_dataset, num_clients, alpha)
    
    # Create dataloaders
    client_loaders = []
    for indices in partitions:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Global model
    global_model = CNN().to(device)
    
    results = {'accuracies': [], 'losses': []}
    
    for round_num in range(1, num_rounds + 1):
        selected_ids = np.random.choice(num_clients, clients_per_round, replace=False)
        
        all_weights = []
        sample_counts = []
        
        for client_id in selected_ids:
            # Create local model with global weights
            local_model = copy_model(global_model).to(device)
            
            # Train
            weights, count = train_fedavg(
                local_model,
                client_loaders[client_id],
                local_epochs,
                lr,
                device
            )
            
            all_weights.append(weights)
            sample_counts.append(count)
        
        # Aggregate
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
        
        if round_num % 5 == 0 or round_num == 1:
            print(f"  Round {round_num:2d}: Accuracy={accuracy:.4f}, Loss={avg_loss:.4f}")
    
    print(f"\nFinal Accuracy: {results['accuracies'][-1]:.4f}")
    
    return results


def main():
    """Compare SCAFFOLD vs FedAvg."""
    
    print("=" * 70)
    print("SCAFFOLD vs FedAvg Comparison")
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
    import matplotlib.pyplot as plt
    
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
    
    plt.suptitle('SCAFFOLD vs FedAvg: Variance Reduction in Heterogeneous FL')
    plt.tight_layout()
    plt.savefig('scaffold_vs_fedavg.png', dpi=150)
    plt.show()
    
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
    print("KEY INSIGHTS: SCAFFOLD")
    print("=" * 70)
    print("""
    1. SCAFFOLD uses control variates to reduce client drift:
       - c_k: Client control variate (tracks expected gradient)
       - c: Global control variate (average of client controls)
       
    2. Corrected gradient: ∇f_k(x) - c_k + c
       - Subtracts client bias: -c_k
       - Adds global direction: +c
       
    3. Control variate update (Option I):
       c_k+ = c_k - c + (x - y) / (K * η)
       
    4. Benefits:
       - Variance reduction → faster convergence
       - Provably better communication complexity
       - Works well with full client participation
       
    5. Limitations:
       - Extra communication: send Δc along with Δw
       - Memory: store c_k per client
       - Partial participation degrades benefits
       
    6. When to use SCAFFOLD:
       - Very heterogeneous data
       - Many local updates (large K)
       - Full or high client participation rate
    """)


if __name__ == "__main__":
    main()
