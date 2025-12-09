"""
Rényi Differential Privacy (RDP)
================================

This notebook covers:
1. Rényi divergence definition
2. RDP definition and properties
3. Composition under RDP (much tighter than basic!)
4. Conversion from RDP to (ε,δ)-DP
5. Why Opacus uses RDP internally

Key Takeaways:
- RDP provides tighter composition bounds
- Sampled Gaussian mechanism has nice RDP properties
- Opacus privacy accountant uses RDP under the hood
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PART 1: Rényi Divergence
# ============================================================

def renyi_divergence(p: np.ndarray, q: np.ndarray, alpha: float) -> float:
    """
    Compute Rényi divergence of order α between distributions P and Q.
    
    D_α(P || Q) = (1/(α-1)) × log(Σ p(x)^α × q(x)^(1-α))
    
    Special cases:
    - α → 1: KL divergence
    - α = 2: Chi-squared divergence related
    - α → ∞: Max divergence
    """
    if alpha == 1:
        # KL divergence (limit as α → 1)
        mask = (p > 0) & (q > 0)
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))
    
    if alpha == np.inf:
        # Max divergence
        mask = p > 0
        return np.max(np.log(p[mask] / q[mask]))
    
    # General case
    mask = (p > 0) & (q > 0)
    term = np.sum(np.power(p[mask], alpha) * np.power(q[mask], 1 - alpha))
    return np.log(term) / (alpha - 1)


def demo_renyi_divergence():
    """Visualize Rényi divergence for different α values."""
    
    print("=" * 60)
    print("Rényi Divergence Visualization")
    print("=" * 60)
    
    # Two Gaussian distributions with different means
    x = np.linspace(-5, 10, 1000)
    
    mu_p, sigma_p = 0, 1
    mu_q, sigma_q = 2, 1  # Shifted mean
    
    p = np.exp(-0.5 * ((x - mu_p) / sigma_p) ** 2) / (sigma_p * np.sqrt(2 * np.pi))
    q = np.exp(-0.5 * ((x - mu_q) / sigma_q) ** 2) / (sigma_q * np.sqrt(2 * np.pi))
    
    # Normalize to sum to 1 (discrete approximation)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Compute Rényi divergence for various α
    alphas = [0.5, 1, 1.5, 2, 3, 5, 10, 20]
    divergences = []
    
    for alpha in alphas:
        if alpha == 1:
            div = renyi_divergence(p, q, 1.001)  # Approximate limit
        else:
            div = renyi_divergence(p, q, alpha)
        divergences.append(div)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: The distributions
    axes[0].plot(x, p * len(x), 'b-', label='P (original)', linewidth=2)
    axes[0].plot(x, q * len(x), 'r-', label='Q (neighboring)', linewidth=2)
    axes[0].fill_between(x, 0, p * len(x), alpha=0.3)
    axes[0].fill_between(x, 0, q * len(x), alpha=0.3)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Two Distributions P and Q')
    axes[0].legend()
    
    # Right: Rényi divergence vs α
    axes[1].plot(alphas, divergences, 'g-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('α (order)')
    axes[1].set_ylabel('D_α(P || Q)')
    axes[1].set_title('Rényi Divergence vs Order α')
    axes[1].grid(True)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('renyi_divergence.png', dpi=150)
    plt.show()
    
    print("\nRényi Divergence D_α(P || Q) for different α:")
    for a, d in zip(alphas, divergences):
        print(f"  α = {a:5.1f}: D_α = {d:.4f}")
    
    print("\nKey Insight: Higher α = stricter requirement (upper bound on privacy loss)")


# ============================================================
# PART 2: RDP Definition
# ============================================================

def explain_rdp():
    """
    Rényi Differential Privacy Definition
    ======================================
    
    A mechanism M satisfies (α, ρ)-RDP if for all neighboring D, D':
    
        D_α(M(D) || M(D')) ≤ ρ
    
    where D_α is the Rényi divergence of order α.
    
    Key Properties:
    
    1. COMPOSITION: If M₁ is (α, ρ₁)-RDP and M₂ is (α, ρ₂)-RDP,
       then M₁ ∘ M₂ is (α, ρ₁ + ρ₂)-RDP.
       
       This is EXACT composition (no looseness)!
    
    2. FROM (ε,δ)-DP TO RDP:
       (ε, 0)-DP → (α, αε²/2)-RDP for α > 1
       
    3. FROM RDP TO (ε,δ)-DP:
       (α, ρ)-RDP → (ρ + log(1/δ)/(α-1), δ)-DP
       
       This conversion can be optimized over α!
    
    Why RDP is better for composition:
    - (ε,δ)-DP composition: δ's add up, ε's grow ∝ √k or k
    - RDP composition: Just add ρ values, convert to (ε,δ) at the end
    - Optimizing over α gives tighter final bounds
    """
    print(explain_rdp.__doc__)


# ============================================================
# PART 3: Gaussian Mechanism under RDP
# ============================================================

def gaussian_rdp(sigma: float, alpha: float, sensitivity: float = 1.0) -> float:
    """
    Compute RDP guarantee for Gaussian mechanism.
    
    For Gaussian mechanism with noise N(0, σ²) and sensitivity Δ:
        (α, α × (Δ/σ)² / 2)-RDP
    
    Args:
        sigma: Noise standard deviation
        alpha: RDP order
        sensitivity: L2 sensitivity of the query
    
    Returns:
        RDP parameter ρ such that mechanism is (α, ρ)-RDP
    """
    return alpha * (sensitivity / sigma) ** 2 / 2


def subsampled_gaussian_rdp(sigma: float, alpha: float, sampling_rate: float,
                            sensitivity: float = 1.0) -> float:
    """
    Compute RDP for SUBSAMPLED Gaussian mechanism (key for DP-SGD!).
    
    When we subsample with probability q before adding noise,
    we get privacy amplification by subsampling.
    
    This is the key insight that makes DP-SGD practical!
    
    Uses the analytical formula from Mironov et al.
    """
    if sampling_rate == 0:
        return 0
    if sampling_rate == 1:
        return gaussian_rdp(sigma, alpha, sensitivity)
    
    # Simplified approximation (actual formula is more complex)
    # For small q, the RDP is approximately:
    # ρ ≈ log(1 + q² × (exp((α-1) × Δ²/σ²) - 1) / α) / (α - 1)
    
    q = sampling_rate
    if alpha <= 1:
        return 0
    
    # Use the simple upper bound for demonstration
    # Actual Opacus uses the exact formula
    base_rdp = gaussian_rdp(sigma, alpha, sensitivity)
    
    # Privacy amplification by subsampling
    # Approximately: ρ_subsampled ≈ 2 × q² × ρ_full for small q
    if q < 0.1:
        return 2 * q ** 2 * base_rdp
    else:
        return q * base_rdp  # Loose upper bound


def demo_subsampled_gaussian():
    """Show privacy amplification by subsampling."""
    
    print("=" * 60)
    print("Privacy Amplification by Subsampling")
    print("=" * 60)
    
    sigma = 1.0  # Noise multiplier
    alpha = 10   # RDP order
    sensitivity = 1.0
    
    sampling_rates = np.linspace(0.001, 1.0, 100)
    
    full_rdp = gaussian_rdp(sigma, alpha, sensitivity)
    subsampled_rdps = [subsampled_gaussian_rdp(sigma, alpha, q, sensitivity) 
                       for q in sampling_rates]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sampling_rates, subsampled_rdps, 'b-', linewidth=2, 
            label='Subsampled Gaussian RDP')
    ax.axhline(y=full_rdp, color='r', linestyle='--', 
               label=f'Full Gaussian RDP = {full_rdp:.2f}')
    
    ax.set_xlabel('Sampling Rate (q)')
    ax.set_ylabel('RDP (ρ)')
    ax.set_title(f'Privacy Amplification by Subsampling (σ={sigma}, α={alpha})')
    ax.legend()
    ax.grid(True)
    
    # Mark typical DP-SGD sampling rates
    typical_rates = [0.01, 0.05, 0.1]
    for q in typical_rates:
        rdp_val = subsampled_gaussian_rdp(sigma, alpha, q, sensitivity)
        ax.scatter([q], [rdp_val], s=100, zorder=5)
        ax.annotate(f'q={q}\nρ={rdp_val:.4f}', (q, rdp_val), 
                   xytext=(10, 10), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('subsampling_amplification.png', dpi=150)
    plt.show()
    
    print(f"\nFull Gaussian RDP: ρ = {full_rdp:.4f}")
    print("\nWith subsampling:")
    for q in [0.01, 0.05, 0.1]:
        rdp = subsampled_gaussian_rdp(sigma, alpha, q, sensitivity)
        reduction = full_rdp / rdp if rdp > 0 else float('inf')
        print(f"  q = {q}: ρ = {rdp:.4f} ({reduction:.1f}x reduction)")
    
    print("\nKey Insight: Small batches (low q) give massive privacy amplification!")
    print("This is why DP-SGD uses small batch fractions.")


# ============================================================
# PART 4: RDP Composition
# ============================================================

def rdp_composition(rdp_values: List[float]) -> float:
    """
    Compose multiple RDP mechanisms (same α).
    
    Key insight: RDP composes exactly by addition!
    No looseness, no δ accumulation.
    """
    return sum(rdp_values)


def rdp_to_dp(rdp_value: float, alpha: float, delta: float) -> float:
    """
    Convert from (α, ρ)-RDP to (ε, δ)-DP.
    
    ε = ρ + log(1/δ) / (α - 1)
    
    Can optimize over α to get tightest bound!
    """
    if alpha <= 1:
        return float('inf')
    return rdp_value + np.log(1 / delta) / (alpha - 1)


def optimize_rdp_to_dp(rdp_func, delta: float, alpha_range: np.ndarray) -> Tuple[float, float]:
    """
    Find optimal α that minimizes ε when converting RDP to (ε,δ)-DP.
    
    Args:
        rdp_func: Function that returns ρ for given α
        delta: Target δ
        alpha_range: Range of α values to search
    
    Returns:
        (optimal_epsilon, optimal_alpha)
    """
    best_eps = float('inf')
    best_alpha = None
    
    for alpha in alpha_range:
        rdp = rdp_func(alpha)
        eps = rdp_to_dp(rdp, alpha, delta)
        if eps < best_eps:
            best_eps = eps
            best_alpha = alpha
    
    return best_eps, best_alpha


def demo_rdp_composition():
    """Compare RDP composition to basic/advanced (ε,δ) composition."""
    
    print("=" * 60)
    print("RDP vs (ε,δ) Composition Comparison")
    print("=" * 60)
    
    # Simulation: T training steps
    sigma = 1.0  # Noise multiplier
    sampling_rate = 0.01  # Batch fraction
    delta = 1e-5
    
    steps_range = [10, 50, 100, 200, 500, 1000, 2000]
    
    # For each number of steps, compute total ε
    rdp_epsilons = []
    basic_epsilons = []
    
    # Alpha values to optimize over
    alphas = np.linspace(2, 100, 50)
    
    for T in steps_range:
        # RDP approach: compose, then convert
        def total_rdp(alpha):
            single_step_rdp = subsampled_gaussian_rdp(sigma, alpha, sampling_rate)
            return T * single_step_rdp  # Exact composition!
        
        eps_rdp, _ = optimize_rdp_to_dp(total_rdp, delta, alphas)
        rdp_epsilons.append(eps_rdp)
        
        # Basic (ε,δ) approach (simplified)
        # Each step: ε_step ≈ sampling_rate / sigma (rough approximation)
        eps_step = sampling_rate / sigma
        eps_basic = T * eps_step  # Linear composition
        basic_epsilons.append(eps_basic)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(steps_range, rdp_epsilons, 'b-o', label='RDP (optimized)', linewidth=2)
    ax.plot(steps_range, basic_epsilons, 'r--s', label='Basic Composition', linewidth=2)
    
    ax.set_xlabel('Number of Training Steps')
    ax.set_ylabel('Total ε')
    ax.set_title(f'Composition: RDP vs Basic (σ={sigma}, q={sampling_rate}, δ={delta})')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('rdp_vs_basic_composition.png', dpi=150)
    plt.show()
    
    print(f"\nSettings: σ={sigma}, sampling_rate={sampling_rate}, δ={delta}")
    print(f"\n{'Steps':<10} {'RDP ε':<15} {'Basic ε':<15} {'Improvement':<15}")
    print("-" * 55)
    for i, T in enumerate(steps_range):
        improvement = basic_epsilons[i] / rdp_epsilons[i]
        print(f"{T:<10} {rdp_epsilons[i]:<15.2f} {basic_epsilons[i]:<15.2f} {improvement:<15.1f}x")
    
    print("\nKey Insight: RDP gives MUCH tighter bounds for many steps!")


# ============================================================
# PART 5: How Opacus Uses RDP
# ============================================================

def explain_opacus_rdp():
    """
    How Opacus Uses RDP Internally
    ==============================
    
    Opacus's PrivacyEngine does the following:
    
    1. EACH STEP:
       - Clip per-sample gradients to norm C
       - Add Gaussian noise: N(0, σ²C²)
       - Track RDP at multiple α values
    
    2. RDP TRACKING:
       - Maintains ρ(α) for many α ∈ [1.5, 2, 3, ..., 256]
       - Each step adds: ρ_step(α) = subsampled_gaussian_rdp(σ, α, q)
       - Uses exact formula for subsampled Gaussian mechanism
    
    3. GET EPSILON:
       - When you call get_epsilon(delta):
         - For each α: ε_α = ρ(α) + log(1/δ) / (α - 1)
         - Return min over all α
    
    This is why Opacus gives tight privacy bounds!
    
    Code example:
    ```python
    from opacus import PrivacyEngine
    from opacus.accountants import RDPAccountant
    
    # Create privacy engine
    privacy_engine = PrivacyEngine()
    
    # Wrap model, optimizer, dataloader
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=1.0,  # σ
        max_grad_norm=1.0,     # C
    )
    
    # Train...
    
    # Get epsilon
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    ```
    """
    print(explain_opacus_rdp.__doc__)


def simulate_opacus_accounting():
    """Simulate Opacus-style RDP accounting."""
    
    print("=" * 60)
    print("Simulating Opacus Privacy Accounting")
    print("=" * 60)
    
    # Training parameters
    n_samples = 50000
    batch_size = 500
    epochs = 10
    
    sigma = 1.0  # noise_multiplier
    delta = 1e-5
    
    sampling_rate = batch_size / n_samples
    steps_per_epoch = n_samples // batch_size
    total_steps = epochs * steps_per_epoch
    
    print(f"Training setup:")
    print(f"  Samples: {n_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sampling rate: {sampling_rate:.4f}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Noise multiplier: {sigma}")
    
    # Track RDP at multiple α values (like Opacus)
    alphas = np.concatenate([
        np.linspace(1.5, 10, 20),
        np.linspace(10, 100, 20),
        np.linspace(100, 256, 10)
    ])
    
    # Accumulate RDP over training
    rdp_per_alpha = {alpha: 0.0 for alpha in alphas}
    
    epsilon_history = []
    steps_history = []
    
    for step in range(1, total_steps + 1):
        # Add RDP for this step
        for alpha in alphas:
            rdp_per_alpha[alpha] += subsampled_gaussian_rdp(sigma, alpha, sampling_rate)
        
        # Compute epsilon (optimize over α)
        if step % 100 == 0 or step == total_steps:
            best_eps = float('inf')
            for alpha in alphas:
                eps = rdp_to_dp(rdp_per_alpha[alpha], alpha, delta)
                best_eps = min(best_eps, eps)
            
            epsilon_history.append(best_eps)
            steps_history.append(step)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Epsilon over training
    epochs_axis = np.array(steps_history) / steps_per_epoch
    axes[0].plot(epochs_axis, epsilon_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('ε')
    axes[0].set_title('Privacy Budget Consumption Over Training')
    axes[0].grid(True)
    
    # Mark final epsilon
    final_eps = epsilon_history[-1]
    axes[0].axhline(y=final_eps, color='r', linestyle='--', alpha=0.5)
    axes[0].annotate(f'Final ε = {final_eps:.2f}', 
                    xy=(epochs, final_eps), xytext=(-50, 10),
                    textcoords='offset points')
    
    # Right: RDP at different α (final)
    final_rdps = [rdp_per_alpha[alpha] for alpha in alphas]
    final_eps_per_alpha = [rdp_to_dp(rho, alpha, delta) for rho, alpha in zip(final_rdps, alphas)]
    
    axes[1].plot(alphas, final_eps_per_alpha, 'g-', linewidth=2)
    axes[1].set_xlabel('α')
    axes[1].set_ylabel('ε (from RDP conversion)')
    axes[1].set_title('RDP to (ε,δ)-DP Conversion at Different α')
    axes[1].grid(True)
    axes[1].set_xscale('log')
    
    # Mark optimal α
    best_idx = np.argmin(final_eps_per_alpha)
    best_alpha = alphas[best_idx]
    axes[1].scatter([best_alpha], [final_eps], s=100, c='red', zorder=5)
    axes[1].annotate(f'Optimal α = {best_alpha:.1f}', 
                    xy=(best_alpha, final_eps), xytext=(10, 10),
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('opacus_accounting.png', dpi=150)
    plt.show()
    
    print(f"\nFinal results after {epochs} epochs:")
    print(f"  Total ε = {final_eps:.2f}")
    print(f"  Optimal α = {best_alpha:.1f}")
    print(f"  δ = {delta}")


# ============================================================
# MAIN: Run all demos
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RÉNYI DIFFERENTIAL PRIVACY (RDP) TUTORIAL")
    print("=" * 70 + "\n")
    
    # Part 1: Rényi divergence
    demo_renyi_divergence()
    
    # Part 2: RDP definition
    print("\n")
    explain_rdp()
    
    # Part 3: Subsampled Gaussian
    print("\n")
    demo_subsampled_gaussian()
    
    # Part 4: Composition comparison
    print("\n")
    demo_rdp_composition()
    
    # Part 5: Opacus internals
    print("\n")
    explain_opacus_rdp()
    print("\n")
    simulate_opacus_accounting()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. RDP uses Rényi divergence (parameterized by order α)
    
    2. RDP composes EXACTLY by addition (no looseness!)
    
    3. Privacy amplification by subsampling:
       - Small batches (q << 1) give massive privacy gains
       - This is why DP-SGD uses small batch fractions
    
    4. RDP → (ε,δ)-DP conversion:
       - Optimize over α for tightest bound
       - Opacus tracks RDP at many α values
    
    5. Opacus internally:
       - Tracks ρ(α) for α ∈ [1.5, 256]
       - get_epsilon() optimizes conversion to (ε,δ)-DP
       - Gives much tighter bounds than basic composition
    
    6. Next: Local DP (randomize BEFORE sending to server)
    """)
