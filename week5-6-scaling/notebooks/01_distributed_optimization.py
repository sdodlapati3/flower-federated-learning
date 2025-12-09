"""
Distributed Optimization Foundations
=====================================

This notebook covers:
1. Gradient descent in distributed settings
2. Consensus optimization
3. How FedAvg relates to distributed optimization
4. Communication-computation tradeoffs

Key Takeaways:
- FL algorithms are special cases of distributed optimization
- Understanding the theory helps design better algorithms
- Communication cost vs convergence speed tradeoff
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PART 1: Centralized vs Distributed Optimization
# ============================================================

def explain_distributed_optimization():
    """
    Distributed Optimization Overview
    ==================================
    
    CENTRALIZED OPTIMIZATION:
    - All data at one location
    - min_w f(w) = (1/n) Σᵢ fᵢ(w)
    - Standard gradient descent: w ← w - η∇f(w)
    
    DISTRIBUTED OPTIMIZATION:
    - Data spread across K workers/clients
    - Each worker k has local data: fₖ(w) = (1/nₖ) Σᵢ∈Dₖ fᵢ(w)
    - Goal: min_w (1/K) Σₖ fₖ(w)
    
    Challenges:
    1. COMMUNICATION: Workers must exchange information
    2. HETEROGENEITY: fₖ(w) may differ significantly across k
    3. SYNCHRONIZATION: Wait for slowest worker?
    4. PRIVACY: Can't share raw data
    
    Key Algorithms:
    - Synchronous SGD: All workers compute, then average
    - Asynchronous SGD: Workers update without waiting
    - FedAvg: Local SGD + periodic averaging
    - Consensus: Peer-to-peer averaging without central server
    """
    print(explain_distributed_optimization.__doc__)


# ============================================================
# PART 2: Gradient Descent Variants
# ============================================================

def quadratic_function(w: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
    """Quadratic objective: f(w) = 0.5 * w'Aw - b'w"""
    return 0.5 * w @ A @ w - b @ w


def quadratic_gradient(w: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gradient of quadratic: ∇f(w) = Aw - b"""
    return A @ w - b


def demo_gradient_descent_variants():
    """Compare different distributed GD approaches on a simple problem."""
    
    print("=" * 60)
    print("Distributed Gradient Descent Variants")
    print("=" * 60)
    
    # Problem setup: min_w 0.5 * w'Aw - b'w
    # Optimal solution: w* = A^(-1)b
    np.random.seed(42)
    d = 10  # Dimension
    
    # Create positive definite A (well-conditioned for easier optimization)
    A = np.random.randn(d, d)
    A = A.T @ A + 0.1 * np.eye(d)  # Make positive definite
    b = np.random.randn(d)
    
    w_optimal = np.linalg.solve(A, b)
    f_optimal = quadratic_function(w_optimal, A, b)
    
    # Split data across K workers (simulate different local objectives)
    K = 5  # Number of workers
    
    # Each worker has local A_k and b_k (slightly different)
    local_As = []
    local_bs = []
    for k in range(K):
        # Add some noise to simulate heterogeneity
        noise_A = 0.1 * np.random.randn(d, d)
        noise_A = noise_A.T @ noise_A
        local_As.append(A + noise_A)
        local_bs.append(b + 0.1 * np.random.randn(d))
    
    # Learning rate
    eta = 0.01
    n_iters = 100
    
    # Method 1: Centralized GD (baseline)
    w_central = np.zeros(d)
    central_losses = []
    
    for t in range(n_iters):
        grad = quadratic_gradient(w_central, A, b)
        w_central = w_central - eta * grad
        central_losses.append(quadratic_function(w_central, A, b) - f_optimal)
    
    # Method 2: Synchronous Distributed GD
    w_sync = np.zeros(d)
    sync_losses = []
    
    for t in range(n_iters):
        # Each worker computes local gradient
        grads = [quadratic_gradient(w_sync, local_As[k], local_bs[k]) for k in range(K)]
        # Average gradients
        avg_grad = np.mean(grads, axis=0)
        w_sync = w_sync - eta * avg_grad
        sync_losses.append(quadratic_function(w_sync, A, b) - f_optimal)
    
    # Method 3: Local SGD (like FedAvg) - multiple local steps between syncs
    w_local = np.zeros(d)
    local_losses = []
    local_steps = 5  # Local updates between syncs
    
    for t in range(n_iters // local_steps):
        # Each worker does local_steps of local GD
        worker_weights = []
        for k in range(K):
            w_k = w_local.copy()
            for _ in range(local_steps):
                grad = quadratic_gradient(w_k, local_As[k], local_bs[k])
                w_k = w_k - eta * grad
            worker_weights.append(w_k)
        
        # Average worker weights
        w_local = np.mean(worker_weights, axis=0)
        local_losses.append(quadratic_function(w_local, A, b) - f_optimal)
    
    # Interpolate local_losses for fair comparison
    local_losses_interp = []
    for i, loss in enumerate(local_losses):
        local_losses_interp.extend([loss] * local_steps)
    local_losses_interp = local_losses_interp[:n_iters]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Loss vs iterations
    axes[0].semilogy(central_losses, label='Centralized GD', linewidth=2)
    axes[0].semilogy(sync_losses, label='Sync Distributed GD', linewidth=2)
    axes[0].semilogy(local_losses_interp, label=f'Local SGD (H={local_steps})', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('f(w) - f(w*) [log scale]')
    axes[0].set_title('Convergence Comparison')
    axes[0].legend()
    axes[0].grid(True)
    
    # Right: Communication rounds
    comm_central = np.arange(n_iters)  # Every iteration
    comm_sync = np.arange(n_iters)     # Every iteration
    comm_local = np.arange(0, n_iters, local_steps)  # Every H iterations
    
    axes[1].semilogy(comm_central, central_losses, 'o-', label='Centralized', 
                     markersize=3, alpha=0.5)
    axes[1].semilogy(comm_local, local_losses, 's-', label=f'Local SGD (H={local_steps})', 
                     markersize=6, linewidth=2)
    axes[1].set_xlabel('Communication Rounds')
    axes[1].set_ylabel('f(w) - f(w*) [log scale]')
    axes[1].set_title('Loss vs Communication Cost')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('distributed_gd_comparison.png', dpi=150)
    plt.show()
    
    print(f"\nFinal losses (f(w) - f(w*)):")
    print(f"  Centralized GD:     {central_losses[-1]:.6f}")
    print(f"  Sync Distributed:   {sync_losses[-1]:.6f}")
    print(f"  Local SGD (H={local_steps}):    {local_losses[-1]:.6f}")
    
    print(f"\nCommunication rounds to reach loss < 0.01:")
    for name, losses in [('Centralized', central_losses), 
                         ('Sync Dist', sync_losses)]:
        for i, l in enumerate(losses):
            if l < 0.01:
                print(f"  {name}: {i+1} rounds")
                break
    
    for i, l in enumerate(local_losses):
        if l < 0.01:
            print(f"  Local SGD: {i+1} rounds ({(i+1)*local_steps} local iters)")
            break


# ============================================================
# PART 3: Consensus Optimization
# ============================================================

def demo_consensus():
    """Demonstrate consensus-based distributed optimization."""
    
    print("=" * 60)
    print("Consensus Optimization")
    print("=" * 60)
    
    print("""
    CONSENSUS ALGORITHM:
    
    Each node k maintains local estimate w_k.
    In each round:
    1. Compute local gradient: g_k = ∇f_k(w_k)
    2. Share w_k with neighbors
    3. Update: w_k ← Σⱼ W_kj * w_j - η * g_k
    
    W is the mixing matrix (doubly stochastic):
    - W_kj > 0 if nodes k and j are connected
    - Σⱼ W_kj = 1 (rows sum to 1)
    - W = W' (symmetric) ensures convergence
    
    Properties:
    - No central server needed (peer-to-peer)
    - Converges to consensus: all w_k → w*
    - Works on arbitrary network topologies
    """)
    
    # Network topology: ring of 6 nodes
    K = 6  # Number of nodes
    
    # Create ring topology mixing matrix
    W = np.zeros((K, K))
    for i in range(K):
        W[i, i] = 0.5  # Self-weight
        W[i, (i-1) % K] = 0.25  # Left neighbor
        W[i, (i+1) % K] = 0.25  # Right neighbor
    
    print("Ring network mixing matrix W:")
    print(W)
    
    # Simple 1D problem: each node k wants to minimize (w - target_k)^2
    # Global optimum: average of all targets
    np.random.seed(42)
    targets = np.random.randn(K) * 5
    optimal = np.mean(targets)
    
    print(f"\nLocal targets: {targets.round(2)}")
    print(f"Optimal consensus value: {optimal:.2f}")
    
    # Consensus + gradient descent
    eta = 0.1
    n_iters = 50
    
    # Initialize with different values
    w = np.random.randn(K) * 2
    
    history = [w.copy()]
    consensus_error = [np.std(w)]
    optimization_error = [np.mean((w - optimal)**2)]
    
    for t in range(n_iters):
        # Compute local gradients: ∇f_k(w_k) = 2(w_k - target_k)
        grads = 2 * (w - targets)
        
        # Consensus + gradient step
        w_new = W @ w - eta * grads
        w = w_new
        
        history.append(w.copy())
        consensus_error.append(np.std(w))
        optimization_error.append(np.mean((w - optimal)**2))
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: Trajectories of each node
    history = np.array(history)
    for k in range(K):
        axes[0].plot(history[:, k], label=f'Node {k}', alpha=0.7)
    axes[0].axhline(y=optimal, color='red', linestyle='--', label='Optimal')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('w_k')
    axes[0].set_title('Node Values Over Time')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)
    
    # Middle: Consensus error
    axes[1].semilogy(consensus_error, 'b-', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('std(w_k) [log scale]')
    axes[1].set_title('Consensus Error')
    axes[1].grid(True)
    
    # Right: Optimization error
    axes[2].semilogy(optimization_error, 'g-', linewidth=2)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Mean (w_k - w*)² [log scale]')
    axes[2].set_title('Optimization Error')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('consensus_optimization.png', dpi=150)
    plt.show()
    
    print(f"\nFinal node values: {w.round(3)}")
    print(f"Consensus achieved: std = {np.std(w):.6f}")
    print(f"Optimization error: {np.mean((w - optimal)**2):.6f}")


# ============================================================
# PART 4: FedAvg as Distributed Optimization
# ============================================================

def explain_fedavg_theory():
    """
    FedAvg from Optimization Theory Perspective
    ============================================
    
    FEDAVG ALGORITHM:
    
    For each round t:
        1. Server sends global model w^t to selected clients
        2. Each client k performs H steps of local SGD:
           w_k^{t,h+1} = w_k^{t,h} - η ∇f_k(w_k^{t,h})
        3. Server aggregates: w^{t+1} = Σ_k (n_k/n) w_k^{t,H}
    
    THEORETICAL VIEW:
    
    FedAvg can be seen as:
    - Inexact gradient descent on global objective
    - Local SGD with periodic averaging
    - Consensus optimization with star topology
    
    KEY INSIGHT (Local SGD convergence):
    
    After H local steps, the "pseudo-gradient" is:
    
    g^t ≈ (1/K) Σ_k Σ_{h=0}^{H-1} ∇f_k(w_k^{t,h})
    
    This is a biased estimate of ∇F(w^t) because:
    - Local iterates w_k^{t,h} drift from global w^t
    - Called "client drift" or "gradient divergence"
    
    CONVERGENCE RATE:
    
    For convex, smooth objectives:
    - FedAvg: O(1/T) + O(H²σ²) where σ² = heterogeneity
    - More local steps H → faster per-round but more drift
    - Heterogeneity σ² → worse convergence
    
    WHY IT WORKS:
    
    1. Averaging reduces variance (like mini-batch SGD)
    2. Multiple local steps amortize communication cost
    3. Works well when clients are "similar enough"
    
    LIMITATIONS:
    
    1. Client drift increases with heterogeneity
    2. May not converge to exact optimum
    3. Learning rate must be small for stability
    """
    print(explain_fedavg_theory.__doc__)


def demo_client_drift():
    """Visualize client drift in FedAvg."""
    
    print("=" * 60)
    print("Client Drift Visualization")
    print("=" * 60)
    
    # 2D problem for visualization
    np.random.seed(42)
    
    # Two clients with different local optima
    # Client 0: min (w - [2, 0])²
    # Client 1: min (w - [-2, 0])²
    # Global optimum: [0, 0]
    
    optima = np.array([[2.0, 0.0], [-2.0, 0.0]])
    global_optimum = np.mean(optima, axis=0)
    
    def local_gradient(w, client_id):
        return 2 * (w - optima[client_id])
    
    # FedAvg simulation
    eta = 0.1
    H_values = [1, 5, 10]  # Different local steps
    n_rounds = 20
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, H in enumerate(H_values):
        w_global = np.array([0.0, 1.0])  # Start away from optima
        trajectory = [w_global.copy()]
        
        for round in range(n_rounds):
            # Each client does H local steps
            w_clients = []
            client_trajectories = []
            
            for k in range(2):
                w_k = w_global.copy()
                traj = [w_k.copy()]
                
                for h in range(H):
                    grad = local_gradient(w_k, k)
                    w_k = w_k - eta * grad
                    traj.append(w_k.copy())
                
                w_clients.append(w_k)
                client_trajectories.append(np.array(traj))
            
            # Average
            w_global = np.mean(w_clients, axis=0)
            trajectory.append(w_global.copy())
        
        trajectory = np.array(trajectory)
        
        # Plot
        ax = axes[idx]
        
        # Plot local optima
        ax.scatter(*optima[0], s=100, c='blue', marker='*', label='Client 0 opt')
        ax.scatter(*optima[1], s=100, c='orange', marker='*', label='Client 1 opt')
        ax.scatter(*global_optimum, s=150, c='green', marker='X', label='Global opt')
        
        # Plot global trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-o', 
                markersize=4, alpha=0.7, label='Global model')
        
        # Plot last round's local trajectories
        for k, (traj, color) in enumerate(zip(client_trajectories, ['blue', 'orange'])):
            ax.plot(traj[:, 0], traj[:, 1], '--', color=color, alpha=0.5, linewidth=2)
            ax.scatter(traj[-1, 0], traj[-1, 1], s=50, c=color, marker='s')
        
        ax.set_xlabel('w₁')
        ax.set_ylabel('w₂')
        ax.set_title(f'H = {H} local steps')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 2)
        ax.grid(True)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('client_drift.png', dpi=150)
    plt.show()
    
    print("\nObservations:")
    print("  - H=1: Global model moves toward global optimum smoothly")
    print("  - H=5: Some drift, but still converges")
    print("  - H=10: Significant drift, may oscillate")
    print("\nKey Insight: More local steps = more communication efficiency")
    print("             But also more client drift (especially with heterogeneous data)")


# ============================================================
# PART 5: Communication-Computation Tradeoffs
# ============================================================

def demo_comm_computation_tradeoff():
    """Analyze the communication-computation tradeoff."""
    
    print("=" * 60)
    print("Communication-Computation Tradeoff Analysis")
    print("=" * 60)
    
    # Simulated convergence curves for different H
    np.random.seed(42)
    
    # Problem parameters
    n_total_compute = 1000  # Total local gradient computations
    K = 10  # Number of clients
    
    H_values = [1, 2, 5, 10, 20, 50]
    
    # Simulate convergence (using a simplified model)
    # Loss ~ a * exp(-b * rounds) + c * H * heterogeneity
    
    heterogeneity = 0.1  # How different clients are
    base_convergence_rate = 0.05
    
    results = {}
    
    for H in H_values:
        n_rounds = n_total_compute // (K * H)
        
        losses = []
        comm_costs = []
        
        for t in range(n_rounds):
            # Loss decreases exponentially, but with floor due to drift
            base_loss = np.exp(-base_convergence_rate * t * H)
            drift_penalty = heterogeneity * H * 0.1
            loss = base_loss + drift_penalty + 0.01 * np.random.randn()
            loss = max(0.001, loss)
            
            losses.append(loss)
            comm_costs.append(t)  # Each round = 1 communication
        
        results[H] = {
            'losses': losses,
            'comm_costs': comm_costs,
            'n_rounds': n_rounds,
            'final_loss': losses[-1] if losses else 1.0
        }
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: Loss vs communication rounds
    for H in H_values:
        axes[0].semilogy(results[H]['comm_costs'], results[H]['losses'], 
                        label=f'H={H}', linewidth=2)
    axes[0].set_xlabel('Communication Rounds')
    axes[0].set_ylabel('Loss [log scale]')
    axes[0].set_title('Convergence vs Communication')
    axes[0].legend()
    axes[0].grid(True)
    
    # Middle: Loss vs total computation
    for H in H_values:
        compute = np.array(results[H]['comm_costs']) * K * H
        axes[1].semilogy(compute, results[H]['losses'], 
                        label=f'H={H}', linewidth=2)
    axes[1].set_xlabel('Total Local Gradient Computations')
    axes[1].set_ylabel('Loss [log scale]')
    axes[1].set_title('Convergence vs Computation')
    axes[1].legend()
    axes[1].grid(True)
    
    # Right: Pareto frontier (final loss vs comm cost)
    final_losses = [results[H]['final_loss'] for H in H_values]
    n_rounds_list = [results[H]['n_rounds'] for H in H_values]
    
    axes[2].scatter(n_rounds_list, final_losses, s=100, c='blue')
    for i, H in enumerate(H_values):
        axes[2].annotate(f'H={H}', (n_rounds_list[i], final_losses[i]),
                        xytext=(5, 5), textcoords='offset points')
    axes[2].set_xlabel('Communication Rounds')
    axes[2].set_ylabel('Final Loss')
    axes[2].set_title('Communication vs Accuracy Tradeoff')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('comm_computation_tradeoff.png', dpi=150)
    plt.show()
    
    print("\nTradeoff Summary:")
    print(f"{'H':<6} {'Comm Rounds':<15} {'Final Loss':<15}")
    print("-" * 36)
    for H in H_values:
        print(f"{H:<6} {results[H]['n_rounds']:<15} {results[H]['final_loss']:<15.4f}")
    
    print("\nKey Insights:")
    print("  - Small H: More communication, better convergence")
    print("  - Large H: Less communication, but client drift hurts accuracy")
    print("  - Sweet spot depends on heterogeneity and communication cost")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DISTRIBUTED OPTIMIZATION FOUNDATIONS")
    print("=" * 70 + "\n")
    
    # Part 1: Overview
    explain_distributed_optimization()
    
    # Part 2: GD variants
    print("\n")
    demo_gradient_descent_variants()
    
    # Part 3: Consensus
    print("\n")
    demo_consensus()
    
    # Part 4: FedAvg theory
    print("\n")
    explain_fedavg_theory()
    print("\n")
    demo_client_drift()
    
    # Part 5: Tradeoffs
    print("\n")
    demo_comm_computation_tradeoff()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. DISTRIBUTED OPTIMIZATION:
       - Data spread across workers
       - Need to balance communication and computation
       
    2. THREE MAIN APPROACHES:
       - Synchronous: All workers compute, then average
       - Local SGD: Multiple local steps between syncs (FedAvg)
       - Consensus: Peer-to-peer averaging without server
       
    3. FEDAVG AS LOCAL SGD:
       - H local steps → O(1/H) less communication
       - But client drift → biased gradients
       - Heterogeneity makes drift worse
       
    4. COMMUNICATION-COMPUTATION TRADEOFF:
       - More local steps = less communication cost
       - But also worse convergence (due to drift)
       - Optimal H depends on problem structure
       
    5. NEXT: ADMM and FedProx (handling heterogeneity better)
    """)
