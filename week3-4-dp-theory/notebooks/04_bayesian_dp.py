"""
Bayesian Differential Privacy
==============================

This notebook covers:
1. Bayesian view of differential privacy
2. DP through posterior sampling
3. Bayesian estimation of privacy leakage
4. Connection to PAC-Bayes and generalization
5. Practical Bayesian DP for ML

Key Takeaways:
- Bayesian DP provides probabilistic privacy guarantees
- Posterior sampling can inherently provide DP
- Can estimate ε from observed mechanism outputs
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List, Callable
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PART 1: Bayesian View of Privacy
# ============================================================

def explain_bayesian_dp():
    """
    Bayesian Differential Privacy
    ==============================
    
    Classical DP (Frequentist View):
    - Worst-case guarantee over ALL neighboring databases
    - Pr[M(D) ∈ S] ≤ e^ε × Pr[M(D') ∈ S]
    - Must hold for worst-case D, D', S
    
    Bayesian DP:
    - Consider a PRIOR distribution over databases
    - Privacy measured as posterior uncertainty about data
    - "Given the output, how much can an adversary learn?"
    
    Key Concepts:
    
    1. PRIOR KNOWLEDGE:
       - Adversary has prior belief Pr(D) about the database
       - After seeing output o = M(D), updates to posterior Pr(D|o)
       
    2. PRIVACY LOSS (Bayesian):
       - How much does posterior differ from prior?
       - If Pr(D|o) ≈ Pr(D), mechanism is private
       
    3. EXPECTED VS WORST-CASE:
       - Standard DP: worst-case privacy loss
       - Bayesian DP: expected privacy loss under prior
       
    Why Bayesian DP?
    - More realistic: adversaries have prior knowledge
    - Can give tighter bounds for "typical" cases
    - Natural for Bayesian ML pipelines
    - Can estimate ε empirically from data
    """
    print(explain_bayesian_dp.__doc__)


# ============================================================
# PART 2: Differential Privacy through Posterior Sampling
# ============================================================

def explain_posterior_sampling_dp():
    """
    DP through Posterior Sampling
    =============================
    
    Key Insight (Dimitrakakis et al., 2014):
    If we sample from the posterior distribution P(θ|D) and the 
    posterior is "stable" (doesn't change much when we change one record),
    then the sampling mechanism satisfies DP!
    
    Setup:
    - Data: D = {x₁, ..., xₙ}
    - Prior: π(θ)
    - Likelihood: P(D|θ) = ∏ᵢ P(xᵢ|θ)
    - Posterior: P(θ|D) ∝ π(θ) × P(D|θ)
    
    Privacy Analysis:
    For neighboring D, D' (differ in one record):
        P(θ|D) / P(θ|D') = P(xᵢ|θ) / P(x'ᵢ|θ)
    
    If the likelihood ratio is bounded:
        e^(-ε) ≤ P(xᵢ|θ) / P(x'ᵢ|θ) ≤ e^ε for all θ
    
    Then posterior sampling is ε-DP!
    
    Practical Implications:
    - Bounded likelihoods → automatic DP
    - Stronger priors → better privacy (less sensitivity to data)
    - Works naturally with Bayesian neural networks
    """
    print(explain_posterior_sampling_dp.__doc__)


def demo_posterior_sampling_dp():
    """Demonstrate DP through posterior sampling for mean estimation."""
    
    print("=" * 60)
    print("DP through Posterior Sampling: Mean Estimation")
    print("=" * 60)
    
    # Problem: Estimate mean of bounded data in [0, 1]
    # True data
    np.random.seed(42)
    n = 100
    true_mean = 0.6
    data = np.clip(np.random.normal(true_mean, 0.15, n), 0, 1)
    
    # Neighboring dataset (change one point)
    data_neighbor = data.copy()
    data_neighbor[0] = 1.0 - data_neighbor[0]  # Flip one point
    
    # Bayesian inference with different priors
    # Prior: θ ~ Beta(α, β) (conjugate for bounded data)
    
    def posterior_samples_beta(data: np.ndarray, prior_alpha: float, 
                                prior_beta: float, n_samples: int) -> np.ndarray:
        """Sample from Beta posterior for Bernoulli-like data."""
        # Approximate: treat data mean as success rate
        successes = np.sum(data)
        failures = len(data) - successes
        
        post_alpha = prior_alpha + successes
        post_beta = prior_beta + failures
        
        return np.random.beta(post_alpha, post_beta, n_samples)
    
    n_samples = 10000
    
    # Different prior strengths
    priors = [
        (1, 1, "Weak prior (α=1, β=1)"),
        (10, 10, "Moderate prior (α=10, β=10)"),
        (100, 100, "Strong prior (α=100, β=100)")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (alpha, beta, label) in enumerate(priors):
        # Posterior samples for D and D'
        samples_D = posterior_samples_beta(data, alpha, beta, n_samples)
        samples_D_prime = posterior_samples_beta(data_neighbor, alpha, beta, n_samples)
        
        # Plot histograms
        axes[idx].hist(samples_D, bins=50, alpha=0.5, density=True, label='D')
        axes[idx].hist(samples_D_prime, bins=50, alpha=0.5, density=True, label="D'")
        axes[idx].axvline(x=np.mean(samples_D), color='blue', linestyle='--')
        axes[idx].axvline(x=np.mean(samples_D_prime), color='orange', linestyle='--')
        
        axes[idx].set_xlabel('θ (estimated mean)')
        axes[idx].set_ylabel('Density')
        axes[idx].set_title(label)
        axes[idx].legend()
        
        # Compute "privacy": how different are the posteriors?
        mean_diff = abs(np.mean(samples_D) - np.mean(samples_D_prime))
        print(f"{label}:")
        print(f"  Mean difference between D and D': {mean_diff:.4f}")
        print(f"  Std of posterior: {np.std(samples_D):.4f}")
    
    plt.tight_layout()
    plt.savefig('posterior_sampling_dp.png', dpi=150)
    plt.show()
    
    print("\nKey Insight: Stronger prior = Less sensitivity to individual data points")
    print("This is how posterior sampling provides DP!")


# ============================================================
# PART 3: Bayesian Estimation of Privacy Leakage
# ============================================================

def explain_bayesian_estimation():
    """
    Bayesian Estimation of Privacy Leakage
    =======================================
    
    Problem: Given outputs of a mechanism, estimate ε empirically.
    
    Why useful?
    - Verify that implementation achieves claimed ε
    - Estimate actual privacy for adaptive mechanisms
    - Audit privacy of existing systems
    
    Approach (Zanella-Béguelin et al., 2020):
    
    1. Run mechanism on neighboring datasets D, D'
    2. Observe outputs o₁, ..., oₖ from each
    3. Train a classifier to distinguish D vs D' from outputs
    4. Privacy loss relates to classifier success rate
    
    Bayesian Connection:
    - Put prior on ε: P(ε)
    - Likelihood: P(classifier results | ε)  
    - Posterior: P(ε | classifier results)
    - Estimate ε from posterior
    
    Key Insight:
    If classifier achieves accuracy p on distinguishing D vs D':
        ε ≈ log(p / (1-p)) for random guess as baseline
    
    This gives EMPIRICAL lower bound on ε!
    """
    print(explain_bayesian_estimation.__doc__)


def demo_epsilon_estimation():
    """Demonstrate empirical epsilon estimation."""
    
    print("=" * 60)
    print("Empirical Estimation of ε")
    print("=" * 60)
    
    # Create a mechanism with known epsilon
    true_epsilon = 2.0
    sensitivity = 1.0
    
    def laplace_mechanism(value: float, epsilon: float) -> float:
        """Laplace mechanism with known epsilon."""
        scale = sensitivity / epsilon
        return value + np.random.laplace(0, scale)
    
    # Generate outputs from neighboring databases
    n_trials = 1000
    
    value_D = 50.0  # Value for database D
    value_D_prime = 51.0  # Value for neighboring D' (differs by sensitivity)
    
    outputs_D = [laplace_mechanism(value_D, true_epsilon) for _ in range(n_trials)]
    outputs_D_prime = [laplace_mechanism(value_D_prime, true_epsilon) for _ in range(n_trials)]
    
    # Simple classifier: output > threshold suggests D'
    threshold = (value_D + value_D_prime) / 2
    
    # Classify
    correct_D = sum(1 for o in outputs_D if o < threshold)
    correct_D_prime = sum(1 for o in outputs_D_prime if o >= threshold)
    
    accuracy = (correct_D + correct_D_prime) / (2 * n_trials)
    
    # Estimate epsilon from accuracy
    # accuracy = (e^ε + 1) / (2 * (e^ε + 1)) for symmetric case... 
    # Simplified: ε ≈ 2 * (accuracy - 0.5) * C for some constant C
    
    # Better formula: use likelihood ratio
    # For Laplace: Pr(o|D) / Pr(o|D') = exp(-|o-D|/b + |o-D'|/b)
    
    # Compute average log likelihood ratio
    log_ratios = []
    for o in outputs_D:
        scale = sensitivity / true_epsilon
        log_ratio = (abs(o - value_D_prime) - abs(o - value_D)) / scale
        log_ratios.append(log_ratio)
    
    estimated_epsilon = np.mean([abs(lr) for lr in log_ratios])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Output distributions
    axes[0].hist(outputs_D, bins=50, alpha=0.5, density=True, label='D')
    axes[0].hist(outputs_D_prime, bins=50, alpha=0.5, density=True, label="D'")
    axes[0].axvline(x=threshold, color='red', linestyle='--', label='Decision boundary')
    axes[0].set_xlabel('Output')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Output Distributions from D and D\'')
    axes[0].legend()
    
    # Right: Log likelihood ratios
    axes[1].hist(log_ratios, bins=50, alpha=0.7, density=True)
    axes[1].axvline(x=true_epsilon, color='red', linestyle='--', 
                    label=f'True ε = {true_epsilon}')
    axes[1].axvline(x=estimated_epsilon, color='green', linestyle='--',
                    label=f'Estimated ε ≈ {estimated_epsilon:.2f}')
    axes[1].set_xlabel('Log Likelihood Ratio')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution of Privacy Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('epsilon_estimation.png', dpi=150)
    plt.show()
    
    print(f"\nTrue ε: {true_epsilon}")
    print(f"Classifier accuracy: {accuracy:.3f}")
    print(f"Estimated ε: {estimated_epsilon:.2f}")
    
    print("\nThis is how privacy auditing works!")
    print("Run many trials, estimate worst-case distinguishability.")


# ============================================================
# PART 4: Bayesian DP for Machine Learning
# ============================================================

def explain_bayesian_dp_ml():
    """
    Bayesian DP for Machine Learning
    =================================
    
    Key Paper: Triastcyn & Faltings, "Bayesian DP for Machine Learning"
    
    Main Ideas:
    
    1. PERSONALIZED PRIVACY:
       - Different data points may need different ε
       - Bayesian framework naturally handles this
       - Spend more privacy budget on "typical" examples
    
    2. DATA-DEPENDENT BOUNDS:
       - Standard DP: worst-case over all D, D'
       - Bayesian DP: expected case under data distribution
       - Can give much tighter bounds for real data
    
    3. PAC-BAYES CONNECTION:
       - Privacy and generalization are related!
       - Both measure sensitivity to individual examples
       - Bayesian posteriors satisfy both
    
    Practical Implementation:
    
    1. BAYESIAN NEURAL NETWORKS:
       - Posterior over weights P(w|D)
       - Sample weights from posterior for predictions
       - Posterior concentration gives privacy
    
    2. DPSGD AS APPROXIMATE BAYES:
       - DP-SGD noise ≈ temperature scaling in SGLD
       - Can view DP training as approximate Bayesian inference
       - Privacy budget ≈ posterior concentration
    
    3. ADAPTIVE PRIVACY BUDGET:
       - Allocate more ε to hard examples
       - Less privacy loss for typical examples
       - Better utility for same total privacy
    
    Connection to Uncertainty:
    - Bayesian DP naturally provides uncertainty estimates
    - Higher posterior variance → less certain predictions
    - Privacy noise increases uncertainty (appropriately!)
    """
    print(explain_bayesian_dp_ml.__doc__)


def demo_bayesian_nn_privacy():
    """Demonstrate privacy properties of Bayesian inference."""
    
    print("=" * 60)
    print("Bayesian Neural Network Privacy Properties")
    print("=" * 60)
    
    # Simple example: Linear regression with posterior uncertainty
    np.random.seed(42)
    
    # Generate data
    n = 50
    x = np.random.uniform(-3, 3, n)
    true_slope = 2.0
    true_intercept = 1.0
    noise_std = 0.5
    y = true_slope * x + true_intercept + np.random.normal(0, noise_std, n)
    
    # Neighboring dataset (remove one point)
    idx_remove = 0
    x_neighbor = np.delete(x, idx_remove)
    y_neighbor = np.delete(y, idx_remove)
    
    # Bayesian linear regression
    def bayesian_linear_regression(x: np.ndarray, y: np.ndarray, 
                                   prior_std: float = 1.0,
                                   n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Simple Bayesian linear regression with normal prior."""
        # Design matrix
        X = np.column_stack([np.ones_like(x), x])
        
        # Prior: N(0, prior_std^2 * I)
        prior_cov = prior_std ** 2 * np.eye(2)
        prior_mean = np.zeros(2)
        
        # Posterior (conjugate for known noise variance)
        data_precision = X.T @ X / noise_std ** 2
        prior_precision = np.linalg.inv(prior_cov)
        
        post_cov = np.linalg.inv(prior_precision + data_precision)
        post_mean = post_cov @ (X.T @ y / noise_std ** 2)
        
        # Sample from posterior
        samples = np.random.multivariate_normal(post_mean, post_cov, n_samples)
        
        return samples[:, 0], samples[:, 1]  # intercepts, slopes
    
    # Different prior strengths
    prior_stds = [0.5, 2.0, 10.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, prior_std in enumerate(prior_stds):
        # Posterior for D
        intercepts_D, slopes_D = bayesian_linear_regression(x, y, prior_std)
        
        # Posterior for D' (neighboring)
        intercepts_D_prime, slopes_D_prime = bayesian_linear_regression(
            x_neighbor, y_neighbor, prior_std)
        
        # Plot slope posteriors
        axes[idx].hist(slopes_D, bins=50, alpha=0.5, density=True, label='D')
        axes[idx].hist(slopes_D_prime, bins=50, alpha=0.5, density=True, label="D'")
        axes[idx].axvline(x=true_slope, color='red', linestyle='--', label='True slope')
        
        axes[idx].set_xlabel('Slope')
        axes[idx].set_ylabel('Density')
        axes[idx].set_title(f'Prior std = {prior_std}')
        axes[idx].legend()
        
        # Compute "privacy": posterior shift
        mean_shift = abs(np.mean(slopes_D) - np.mean(slopes_D_prime))
        std_D = np.std(slopes_D)
        snr = mean_shift / std_D  # Signal-to-noise ratio
        
        print(f"\nPrior std = {prior_std}:")
        print(f"  Mean slope (D): {np.mean(slopes_D):.3f}")
        print(f"  Mean slope (D'): {np.mean(slopes_D_prime):.3f}")
        print(f"  Mean shift: {mean_shift:.4f}")
        print(f"  Posterior std: {std_D:.4f}")
        print(f"  SNR (privacy proxy): {snr:.4f}")
    
    plt.tight_layout()
    plt.savefig('bayesian_nn_privacy.png', dpi=150)
    plt.show()
    
    print("\nKey Insight: Weaker prior (higher std) → more sensitivity to data")
    print("Stronger prior → less sensitivity → better privacy!")


# ============================================================
# PART 5: Comparison with Standard DP
# ============================================================

def compare_dp_paradigms():
    """Compare different DP paradigms."""
    
    print("=" * 60)
    print("DP Paradigms Comparison")
    print("=" * 60)
    
    comparison = """
    ┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
    │ Aspect          │ Standard (ε,δ)   │ Rényi DP         │ Bayesian DP      │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ Privacy measure │ Worst-case       │ Rényi divergence │ Posterior change │
    │                 │ likelihood ratio │ at order α       │ under prior      │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ Composition     │ Linear/sublinear │ Exact addition   │ Depends on       │
    │                 │ (loose bounds)   │ (tight)          │ prior strength   │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ Mechanism       │ Add calibrated   │ Add calibrated   │ Sample from      │
    │                 │ Laplace/Gaussian │ Gaussian noise   │ posterior        │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ Adversary       │ Knows all data   │ Knows all data   │ Has prior belief │
    │ assumption      │ except one       │ except one       │ about data       │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ Typical use     │ DP-SGD, Central  │ Privacy account- │ Bayesian ML,     │
    │                 │ mechanisms       │ ing, Opacus      │ auditing         │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ Pros            │ Clear worst-case │ Tight compos-    │ Data-dependent,  │
    │                 │ guarantees       │ ition bounds     │ natural for BNN  │
    ├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ Cons            │ Can be loose,    │ Needs conversion │ Depends on prior │
    │                 │ conservative     │ to (ε,δ)         │ assumptions      │
    └─────────────────┴──────────────────┴──────────────────┴──────────────────┘
    
    When to use each:
    
    Standard (ε,δ)-DP:
    - Regulatory compliance requiring worst-case guarantees
    - Simple mechanisms (counting, histograms)
    - When you need to compose few mechanisms
    
    Rényi DP:
    - DP-SGD training (many gradient steps)
    - When composition matters (Opacus uses this!)
    - Converting to (ε,δ) at the end
    
    Bayesian DP:
    - Bayesian neural networks
    - Privacy auditing (estimating ε empirically)
    - When you have strong prior knowledge
    - Personalized/adaptive privacy
    """
    print(comparison)


# ============================================================
# MAIN: Run all demos
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BAYESIAN DIFFERENTIAL PRIVACY TUTORIAL")
    print("=" * 70 + "\n")
    
    # Part 1: Bayesian view
    explain_bayesian_dp()
    
    # Part 2: Posterior sampling
    print("\n")
    explain_posterior_sampling_dp()
    print("\n")
    demo_posterior_sampling_dp()
    
    # Part 3: Epsilon estimation
    print("\n")
    explain_bayesian_estimation()
    print("\n")
    demo_epsilon_estimation()
    
    # Part 4: Bayesian DP for ML
    print("\n")
    explain_bayesian_dp_ml()
    print("\n")
    demo_bayesian_nn_privacy()
    
    # Part 5: Comparison
    print("\n")
    compare_dp_paradigms()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. Bayesian DP: Privacy as posterior uncertainty
       - How much does output reveal about individual data?
       - Natural for Bayesian ML pipelines
    
    2. Posterior Sampling provides DP:
       - If likelihood ratios are bounded
       - Stronger priors → better privacy
       - Works naturally with Bayesian neural networks
    
    3. Empirical ε estimation:
       - Run mechanism on neighboring datasets
       - Use classifier or likelihood ratios
       - Useful for privacy auditing
    
    4. DP-SGD as approximate Bayes:
       - Noise injection ≈ temperature in SGLD
       - Privacy and generalization related
    
    5. Three paradigms:
       - Standard (ε,δ): Worst-case, regulatory
       - RDP: Tight composition (Opacus)
       - Bayesian: Data-dependent, auditing
    
    6. Complete! You now understand all major DP variants:
       - Approximate (ε,δ)-DP
       - Rényi DP
       - Local DP  
       - Bayesian DP
    """)
