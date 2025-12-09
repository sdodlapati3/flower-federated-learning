"""
Approximate Differential Privacy (ε,δ)-DP
==========================================

This notebook covers:
1. Pure DP (ε-DP) vs Approximate DP (ε,δ)-DP
2. Laplace vs Gaussian mechanisms
3. Composition theorems (basic and advanced)
4. Privacy budget visualization

Key Takeaways:
- δ represents probability of "catastrophic" privacy failure
- Gaussian mechanism requires (ε,δ)-DP (not pure ε-DP)
- Advanced composition gives tighter bounds for many queries
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PART 1: Pure DP vs Approximate DP Definitions
# ============================================================

def explain_dp_definitions():
    """
    Differential Privacy Definitions
    =================================
    
    PURE DP (ε-DP):
    For all neighboring databases D, D' and all outputs S:
        Pr[M(D) ∈ S] ≤ e^ε × Pr[M(D') ∈ S]
    
    - ε controls privacy: smaller ε = stronger privacy
    - Holds for ALL possible outputs (worst-case guarantee)
    - Achieved by Laplace mechanism
    
    APPROXIMATE DP (ε,δ)-DP:
    For all neighboring databases D, D' and all outputs S:
        Pr[M(D) ∈ S] ≤ e^ε × Pr[M(D') ∈ S] + δ
    
    - δ is the probability of a "catastrophic" privacy failure
    - δ should be cryptographically small (e.g., 1e-5, < 1/n)
    - Allows Gaussian mechanism (more practical for ML)
    
    Why (ε,δ)-DP is useful:
    - Tighter composition bounds
    - Works with Gaussian noise (useful for DP-SGD)
    - More practical for deep learning
    """
    print(explain_dp_definitions.__doc__)


# ============================================================
# PART 2: Laplace Mechanism (Pure ε-DP)
# ============================================================

def laplace_mechanism(true_value: float, sensitivity: float, epsilon: float) -> float:
    """
    Laplace Mechanism for ε-DP.
    
    Args:
        true_value: The true answer to the query
        sensitivity: L1 sensitivity of the query (max change for neighboring DBs)
        epsilon: Privacy parameter
    
    Returns:
        Noisy answer satisfying ε-DP
    
    The scale parameter b = sensitivity / epsilon
    Larger ε → less noise → less privacy
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return true_value + noise


def demo_laplace_mechanism():
    """Demonstrate Laplace mechanism for counting query."""
    
    print("=" * 60)
    print("Laplace Mechanism Demo: Counting Query")
    print("=" * 60)
    
    # True count (e.g., "how many people have disease X?")
    true_count = 100
    sensitivity = 1  # Adding/removing one person changes count by 1
    
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Distribution of noisy answers
    for eps in epsilons:
        scale = sensitivity / eps
        samples = [laplace_mechanism(true_count, sensitivity, eps) for _ in range(1000)]
        axes[0].hist(samples, bins=50, alpha=0.5, label=f'ε={eps}', density=True)
    
    axes[0].axvline(x=true_count, color='red', linestyle='--', label='True value')
    axes[0].set_xlabel('Noisy Count')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Laplace Mechanism: Effect of ε on Noise')
    axes[0].legend()
    
    # Right plot: Error vs epsilon
    n_trials = 1000
    errors = []
    for eps in np.linspace(0.1, 5, 20):
        trial_errors = [abs(laplace_mechanism(true_count, sensitivity, eps) - true_count) 
                       for _ in range(n_trials)]
        errors.append(np.mean(trial_errors))
    
    axes[1].plot(np.linspace(0.1, 5, 20), errors, 'b-o')
    axes[1].set_xlabel('Epsilon (ε)')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('Privacy-Utility Tradeoff')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('laplace_mechanism_demo.png', dpi=150)
    plt.show()
    
    print("\nKey Insight: Lower ε = More noise = More privacy = Less accuracy")


# ============================================================
# PART 3: Gaussian Mechanism (Approximate (ε,δ)-DP)
# ============================================================

def gaussian_mechanism(true_value: float, sensitivity: float, 
                       epsilon: float, delta: float) -> float:
    """
    Gaussian Mechanism for (ε,δ)-DP.
    
    Args:
        true_value: The true answer to the query
        sensitivity: L2 sensitivity of the query
        epsilon: Privacy parameter
        delta: Probability of privacy failure
    
    Returns:
        Noisy answer satisfying (ε,δ)-DP
    
    The standard deviation σ = sensitivity × √(2 ln(1.25/δ)) / ε
    """
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma)
    return true_value + noise


def compare_laplace_gaussian():
    """Compare Laplace and Gaussian mechanisms."""
    
    print("=" * 60)
    print("Laplace vs Gaussian Mechanism Comparison")
    print("=" * 60)
    
    true_value = 100
    sensitivity = 1
    epsilon = 1.0
    delta = 1e-5
    
    n_samples = 10000
    
    # Laplace samples
    laplace_samples = [laplace_mechanism(true_value, sensitivity, epsilon) 
                      for _ in range(n_samples)]
    
    # Gaussian samples
    gaussian_samples = [gaussian_mechanism(true_value, sensitivity, epsilon, delta) 
                       for _ in range(n_samples)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(laplace_samples, bins=100, alpha=0.5, label='Laplace (ε-DP)', density=True)
    ax.hist(gaussian_samples, bins=100, alpha=0.5, label=f'Gaussian ((ε,δ)-DP, δ={delta})', density=True)
    ax.axvline(x=true_value, color='red', linestyle='--', label='True value')
    
    ax.set_xlabel('Noisy Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Laplace vs Gaussian Mechanism (ε={epsilon})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('laplace_vs_gaussian.png', dpi=150)
    plt.show()
    
    print(f"\nLaplace: Mean={np.mean(laplace_samples):.2f}, Std={np.std(laplace_samples):.2f}")
    print(f"Gaussian: Mean={np.mean(gaussian_samples):.2f}, Std={np.std(gaussian_samples):.2f}")
    print(f"\nKey Insight: Gaussian has lighter tails (less extreme outliers)")
    print("But Gaussian requires δ > 0 (not pure DP)")


# ============================================================
# PART 4: Composition Theorems
# ============================================================

def basic_composition(epsilons: List[float], deltas: List[float]) -> Tuple[float, float]:
    """
    Basic Composition Theorem.
    
    If we run k mechanisms with privacy (ε_i, δ_i), the total privacy is:
        (Σε_i, Σδ_i)
    
    Simple but loose bound!
    """
    total_eps = sum(epsilons)
    total_delta = sum(deltas)
    return total_eps, total_delta


def advanced_composition(epsilon: float, delta: float, k: int, delta_prime: float) -> Tuple[float, float]:
    """
    Advanced Composition Theorem.
    
    For k applications of (ε,δ)-DP mechanism:
        Total privacy: (ε', kδ + δ')
        where ε' = ε × √(2k × ln(1/δ'))
    
    Much tighter for large k!
    """
    eps_prime = epsilon * np.sqrt(2 * k * np.log(1 / delta_prime))
    delta_total = k * delta + delta_prime
    return eps_prime, delta_total


def demo_composition():
    """Demonstrate basic vs advanced composition."""
    
    print("=" * 60)
    print("Composition Theorems Comparison")
    print("=" * 60)
    
    # Single query privacy
    eps_single = 0.1
    delta_single = 1e-6
    delta_prime = 1e-5  # For advanced composition
    
    # Number of queries
    k_values = [1, 5, 10, 20, 50, 100, 200]
    
    basic_eps = []
    advanced_eps = []
    
    for k in k_values:
        # Basic: just sum up
        basic_total, _ = basic_composition([eps_single] * k, [delta_single] * k)
        basic_eps.append(basic_total)
        
        # Advanced: sublinear growth
        adv_total, _ = advanced_composition(eps_single, delta_single, k, delta_prime)
        advanced_eps.append(adv_total)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, basic_eps, 'r-o', label='Basic Composition (linear)', linewidth=2)
    ax.plot(k_values, advanced_eps, 'b-s', label='Advanced Composition (√k)', linewidth=2)
    
    ax.set_xlabel('Number of Queries (k)')
    ax.set_ylabel('Total Privacy Budget (ε)')
    ax.set_title(f'Composition Theorems: Basic vs Advanced (single query ε={eps_single})')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('composition_comparison.png', dpi=150)
    plt.show()
    
    print("\nResults:")
    print(f"{'k':<10} {'Basic ε':<15} {'Advanced ε':<15} {'Improvement':<15}")
    print("-" * 55)
    for i, k in enumerate(k_values):
        improvement = basic_eps[i] / advanced_eps[i]
        print(f"{k:<10} {basic_eps[i]:<15.2f} {advanced_eps[i]:<15.2f} {improvement:<15.2f}x")
    
    print("\nKey Insight: Advanced composition grows as √k instead of k")
    print("Critical for many-round training like DP-SGD!")


# ============================================================
# PART 5: Privacy Budget for DP-SGD Training
# ============================================================

def simulate_dpsgd_privacy_budget(epochs: int, steps_per_epoch: int, 
                                   epsilon_per_step: float, delta: float):
    """Simulate privacy budget consumption during DP-SGD training."""
    
    print("=" * 60)
    print("DP-SGD Privacy Budget Consumption")
    print("=" * 60)
    
    total_steps = epochs * steps_per_epoch
    delta_prime = delta / 10
    
    # Track cumulative epsilon
    steps = list(range(1, total_steps + 1))
    basic_cumulative = []
    advanced_cumulative = []
    
    for step in steps:
        # Basic composition
        basic_eps = step * epsilon_per_step
        basic_cumulative.append(basic_eps)
        
        # Advanced composition
        adv_eps, _ = advanced_composition(epsilon_per_step, delta / total_steps, 
                                          step, delta_prime)
        advanced_cumulative.append(adv_eps)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Full trajectory
    axes[0].plot(steps, basic_cumulative, 'r-', label='Basic Composition', alpha=0.7)
    axes[0].plot(steps, advanced_cumulative, 'b-', label='Advanced Composition', alpha=0.7)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Cumulative ε')
    axes[0].set_title('Privacy Budget Over Training')
    axes[0].legend()
    axes[0].grid(True)
    
    # Epoch markers
    for e in range(epochs):
        axes[0].axvline(x=e * steps_per_epoch, color='gray', linestyle='--', alpha=0.3)
    
    # Right: Final epsilon vs target epsilon
    target_epsilons = [1, 2, 4, 8]
    achievable_steps_basic = []
    achievable_steps_advanced = []
    
    for target in target_epsilons:
        # How many steps can we do before exceeding target?
        steps_basic = int(target / epsilon_per_step)
        achievable_steps_basic.append(min(steps_basic, total_steps))
        
        # For advanced, we need to search
        for s in range(1, total_steps + 1):
            adv_eps, _ = advanced_composition(epsilon_per_step, delta / total_steps, 
                                              s, delta_prime)
            if adv_eps > target:
                achievable_steps_advanced.append(s - 1)
                break
        else:
            achievable_steps_advanced.append(total_steps)
    
    x = np.arange(len(target_epsilons))
    width = 0.35
    
    axes[1].bar(x - width/2, achievable_steps_basic, width, label='Basic Composition')
    axes[1].bar(x + width/2, achievable_steps_advanced, width, label='Advanced Composition')
    axes[1].set_xlabel('Target ε')
    axes[1].set_ylabel('Achievable Training Steps')
    axes[1].set_title('Training Steps Achievable Within Budget')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'ε={e}' for e in target_epsilons])
    axes[1].legend()
    axes[1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('dpsgd_privacy_budget.png', dpi=150)
    plt.show()
    
    print(f"\nSimulation: {epochs} epochs × {steps_per_epoch} steps = {total_steps} total steps")
    print(f"Per-step privacy: ε={epsilon_per_step}, δ={delta}")
    print(f"\nFinal ε after all training:")
    print(f"  Basic Composition:    {basic_cumulative[-1]:.2f}")
    print(f"  Advanced Composition: {advanced_cumulative[-1]:.2f}")
    print(f"\nAdvanced composition allows {basic_cumulative[-1]/advanced_cumulative[-1]:.1f}x more training!")


# ============================================================
# PART 6: Practical Implications for FL
# ============================================================

def fl_privacy_analysis():
    """Analyze privacy in federated learning context."""
    
    print("=" * 60)
    print("Privacy Analysis for Federated Learning")
    print("=" * 60)
    
    print("""
    In Federated Learning with DP:
    
    1. CLIENT-LEVEL DP:
       - Protect participation of entire client
       - Clip and noise aggregated gradient from each client
       - Used in cross-device FL (many small clients)
    
    2. SAMPLE-LEVEL DP (DP-SGD):
       - Protect each training sample
       - Clip per-sample gradients, add noise
       - Used in cross-silo FL (few large clients)
    
    3. COMPOSITION ACROSS ROUNDS:
       - Each FL round consumes privacy budget
       - After T rounds: total ε accumulates
       - Advanced composition helps with many rounds
    
    4. KEY PARAMETERS:
       - ε (epsilon): Privacy loss per round/step
       - δ (delta): Probability of failure (usually 1/n)
       - C (clipping norm): Max gradient norm
       - σ (noise scale): Controls noise magnitude
    
    Relationship: σ = C × √(2 log(1.25/δ)) / ε
    
    For strong privacy (ε ≈ 1-2):
       - Need high noise (low accuracy)
       - Limit training rounds
       - Use tight composition (RDP)
    
    For weak privacy (ε ≈ 8-10):
       - Moderate noise (reasonable accuracy)
       - More training rounds possible
       - Good for many applications
    """)


# ============================================================
# MAIN: Run all demos
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("APPROXIMATE DIFFERENTIAL PRIVACY (ε,δ)-DP TUTORIAL")
    print("=" * 70 + "\n")
    
    # Theory explanation
    explain_dp_definitions()
    
    # Demo 1: Laplace mechanism
    print("\n")
    demo_laplace_mechanism()
    
    # Demo 2: Compare Laplace vs Gaussian
    print("\n")
    compare_laplace_gaussian()
    
    # Demo 3: Composition theorems
    print("\n")
    demo_composition()
    
    # Demo 4: DP-SGD privacy budget
    print("\n")
    simulate_dpsgd_privacy_budget(
        epochs=10, 
        steps_per_epoch=100,
        epsilon_per_step=0.01,
        delta=1e-5
    )
    
    # FL context
    print("\n")
    fl_privacy_analysis()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. (ε,δ)-DP relaxes pure DP by allowing δ probability of failure
    
    2. Gaussian mechanism (used in DP-SGD) requires (ε,δ)-DP
    
    3. Composition matters: 
       - Basic: ε_total = Σε_i (linear)
       - Advanced: ε_total ∝ √k (sublinear)
    
    4. For DP-SGD/FL:
       - Use advanced/RDP composition for tight bounds
       - Trade off ε (privacy) vs accuracy
       - δ should be cryptographically small (< 1/n)
    
    5. Next: Rényi DP provides even tighter composition!
    """)
