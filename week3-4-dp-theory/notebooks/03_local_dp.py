"""
Local Differential Privacy (LDP)
================================

This notebook covers:
1. LDP vs Central DP - key differences
2. Randomized Response (classic LDP technique)
3. RAPPOR (Google's LDP deployment)
4. Frequency estimation under LDP
5. LDP in Federated Learning context

Key Takeaways:
- LDP: No trusted server, randomize BEFORE sending
- Much higher noise than Central DP for same privacy
- Used by Google, Apple, Microsoft for telemetry
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PART 1: Central DP vs Local DP
# ============================================================

def explain_ldp_vs_cdp():
    """
    Local DP vs Central DP
    ======================
    
    CENTRAL DP (what we've been doing):
    - Clients send RAW data to server
    - Server aggregates and adds noise
    - Requires TRUSTED server/aggregator
    - Privacy: ε-DP or (ε,δ)-DP for the aggregated output
    
    LOCAL DP (LDP):
    - Clients RANDOMIZE data BEFORE sending
    - Server receives noisy data, aggregates
    - NO trusted party needed!
    - Privacy: ε-LDP for each user's report
    
    Key Differences:
    
    |                    | Central DP      | Local DP           |
    |--------------------|-----------------|--------------------| 
    | Trust model        | Trusted server  | No trust needed    |
    | Where noise added  | At server       | At each client     |
    | Noise level        | Lower           | Much higher        |
    | Accuracy           | Better          | Worse (for same ε) |
    | Real-world use     | DP-SGD, FL      | Chrome, iOS, etc.  |
    
    Privacy Amplification:
    - Central DP: Can use aggregation to reduce noise
    - LDP: Each individual report already satisfies ε-LDP
    - For n users, Central DP with ε needs noise ∝ 1/ε
    - LDP needs noise ∝ √n/ε for same aggregate accuracy
    
    When to use LDP:
    - When you can't trust the data collector
    - For aggregate statistics (counts, frequencies, means)
    - When individual privacy is paramount
    """
    print(explain_ldp_vs_cdp.__doc__)


# ============================================================
# PART 2: Randomized Response
# ============================================================

def randomized_response(true_value: bool, epsilon: float) -> bool:
    """
    Classic Randomized Response for binary data.
    
    Mechanism:
    1. With probability p = e^ε / (e^ε + 1): Report TRUE value
    2. With probability 1-p: Report RANDOM bit
    
    This satisfies ε-LDP!
    
    Args:
        true_value: The true binary value (True/False)
        epsilon: Privacy parameter
    
    Returns:
        Randomized response (may differ from true_value)
    """
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    
    if np.random.random() < p:
        # Report true value
        return true_value
    else:
        # Report random bit
        return np.random.random() < 0.5


def estimate_proportion_rr(responses: List[bool], epsilon: float) -> float:
    """
    Estimate true proportion from randomized responses.
    
    If true proportion is π, expected response proportion is:
        E[response] = p × π + (1-p) × 0.5
        where p = e^ε / (e^ε + 1)
    
    Solving for π:
        π = (E[response] - 0.5(1-p)) / p
          = (E[response] - 0.5 + 0.5p) / p
    """
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    observed_proportion = np.mean(responses)
    
    # Unbiased estimator
    estimated = (observed_proportion - 0.5 * (1 - p)) / p
    
    # Clip to valid range
    return np.clip(estimated, 0, 1)


def demo_randomized_response():
    """Demonstrate randomized response for surveys."""
    
    print("=" * 60)
    print("Randomized Response Demo")
    print("=" * 60)
    
    # Scenario: Survey asking "Have you ever done X?"
    # True proportion: 30% have done X
    true_proportion = 0.3
    n_users = 1000
    
    # Generate true answers
    true_answers = [np.random.random() < true_proportion for _ in range(n_users)]
    
    epsilons = [0.5, 1.0, 2.0, 4.0, 8.0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    estimates = []
    stds = []
    
    for epsilon in epsilons:
        # Run multiple trials
        trial_estimates = []
        for _ in range(100):
            # Randomize responses
            responses = [randomized_response(ans, epsilon) for ans in true_answers]
            estimate = estimate_proportion_rr(responses, epsilon)
            trial_estimates.append(estimate)
        
        estimates.append(np.mean(trial_estimates))
        stds.append(np.std(trial_estimates))
    
    # Left plot: Estimates vs epsilon
    axes[0].errorbar(epsilons, estimates, yerr=stds, fmt='o-', capsize=5, 
                     linewidth=2, markersize=8)
    axes[0].axhline(y=true_proportion, color='r', linestyle='--', 
                    label=f'True proportion = {true_proportion}')
    axes[0].set_xlabel('Epsilon (ε)')
    axes[0].set_ylabel('Estimated Proportion')
    axes[0].set_title('Randomized Response Estimation Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Right plot: Example of response distribution for one epsilon
    epsilon = 2.0
    responses_low_eps = [randomized_response(ans, 0.5) for ans in true_answers]
    responses_high_eps = [randomized_response(ans, 4.0) for ans in true_answers]
    
    obs_low = np.mean(responses_low_eps)
    obs_high = np.mean(responses_high_eps)
    true_obs = np.mean(true_answers)
    
    categories = ['True', f'ε=0.5\n(obs={obs_low:.2f})', f'ε=4.0\n(obs={obs_high:.2f})']
    values = [true_proportion, obs_low, obs_high]
    colors = ['green', 'blue', 'orange']
    
    axes[1].bar(categories, values, color=colors, alpha=0.7)
    axes[1].axhline(y=0.5, color='gray', linestyle=':', label='Random (0.5)')
    axes[1].set_ylabel('Proportion Saying "Yes"')
    axes[1].set_title('Observed vs True Responses')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('randomized_response.png', dpi=150)
    plt.show()
    
    print(f"\nTrue proportion: {true_proportion}")
    print(f"Number of users: {n_users}")
    print(f"\nEstimates (with 95% CI):")
    for eps, est, std in zip(epsilons, estimates, stds):
        print(f"  ε = {eps}: {est:.3f} ± {1.96*std:.3f}")
    
    print("\nKey Insight: Lower ε = More privacy = More noise = Higher estimation error")


# ============================================================
# PART 3: RAPPOR (Google's LDP)
# ============================================================

def rappor_encode(value: str, bloom_size: int, num_hashes: int, 
                  epsilon: float) -> np.ndarray:
    """
    Simplified RAPPOR encoding.
    
    RAPPOR = Randomized Aggregatable Privacy-Preserving Ordinal Response
    
    Steps:
    1. Hash value into Bloom filter (k hash functions → k bits set)
    2. Apply permanent randomization (PRR)
    3. Apply instantaneous randomization (IRR)
    
    Args:
        value: String to encode (e.g., homepage URL)
        bloom_size: Size of Bloom filter
        num_hashes: Number of hash functions
        epsilon: Privacy parameter (split between PRR and IRR)
    
    Returns:
        Randomized bit array
    """
    # Step 1: Create Bloom filter
    bloom = np.zeros(bloom_size, dtype=int)
    
    # Simple hash functions (in practice, use proper hash functions)
    for i in range(num_hashes):
        h = hash(f"{value}_{i}") % bloom_size
        bloom[h] = 1
    
    # Step 2: Permanent randomization (simplified)
    # Each bit is flipped with some probability
    f = 0.5  # Probability of setting to random
    permanent = np.zeros(bloom_size, dtype=int)
    for i in range(bloom_size):
        if np.random.random() < f:
            permanent[i] = np.random.randint(0, 2)
        else:
            permanent[i] = bloom[i]
    
    # Step 3: Instantaneous randomization
    p = np.exp(epsilon / 2) / (np.exp(epsilon / 2) + 1)
    instantaneous = np.zeros(bloom_size, dtype=int)
    for i in range(bloom_size):
        if np.random.random() < p:
            instantaneous[i] = permanent[i]
        else:
            instantaneous[i] = 1 - permanent[i]
    
    return instantaneous


def demo_rappor_frequency():
    """Demonstrate RAPPOR for frequency estimation."""
    
    print("=" * 60)
    print("RAPPOR-style Frequency Estimation")
    print("=" * 60)
    
    # Scenario: Users report which browser they use
    # True distribution
    browsers = ['Chrome', 'Firefox', 'Safari', 'Edge', 'Other']
    true_distribution = [0.65, 0.12, 0.10, 0.08, 0.05]  # Chrome dominates
    
    n_users = 5000
    
    # Generate true answers
    true_answers = np.random.choice(browsers, size=n_users, p=true_distribution)
    true_counts = Counter(true_answers)
    
    # Apply LDP using simplified RAPPOR-like mechanism
    # For this demo, we'll use direct randomized response on categories
    
    def categorical_ldp(true_value: str, categories: List[str], epsilon: float) -> str:
        """Categorical LDP: report true value or random category."""
        p = np.exp(epsilon) / (np.exp(epsilon) + len(categories) - 1)
        
        if np.random.random() < p:
            return true_value
        else:
            # Return random category (not the true one)
            other_cats = [c for c in categories if c != true_value]
            return np.random.choice(other_cats)
    
    epsilons = [1.0, 2.0, 4.0]
    
    fig, axes = plt.subplots(1, len(epsilons) + 1, figsize=(16, 5))
    
    # True distribution
    x = np.arange(len(browsers))
    true_props = [true_counts[b] / n_users for b in browsers]
    axes[0].bar(x, true_props, color='green', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(browsers, rotation=45)
    axes[0].set_ylabel('Proportion')
    axes[0].set_title('True Distribution')
    axes[0].set_ylim(0, 0.8)
    
    # LDP estimates for each epsilon
    for idx, epsilon in enumerate(epsilons):
        # Collect LDP responses
        ldp_responses = [categorical_ldp(ans, browsers, epsilon) for ans in true_answers]
        ldp_counts = Counter(ldp_responses)
        
        # Estimate true counts (simplified unbiased estimator)
        k = len(browsers)
        p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
        
        estimated_props = []
        for b in browsers:
            observed = ldp_counts[b] / n_users
            # Unbiased estimator
            est = (k * observed - 1) / (k * p - 1)
            estimated_props.append(max(0, est))
        
        # Normalize
        total = sum(estimated_props)
        estimated_props = [e / total for e in estimated_props]
        
        axes[idx + 1].bar(x - 0.2, true_props, width=0.4, color='green', 
                          alpha=0.5, label='True')
        axes[idx + 1].bar(x + 0.2, estimated_props, width=0.4, color='blue', 
                          alpha=0.7, label='Estimated')
        axes[idx + 1].set_xticks(x)
        axes[idx + 1].set_xticklabels(browsers, rotation=45)
        axes[idx + 1].set_title(f'ε = {epsilon}')
        axes[idx + 1].set_ylim(0, 0.8)
        axes[idx + 1].legend()
    
    plt.tight_layout()
    plt.savefig('rappor_frequency.png', dpi=150)
    plt.show()
    
    print("\nGoogle uses RAPPOR for:")
    print("  - Chrome homepage tracking")
    print("  - Default search engine statistics")
    print("  - Software crash reports")
    print("\nApple uses similar LDP for:")
    print("  - Emoji popularity")
    print("  - Health data aggregates")
    print("  - Safari suggestions")


# ============================================================
# PART 4: LDP vs Central DP Accuracy
# ============================================================

def compare_ldp_cdp_accuracy():
    """Compare accuracy of LDP vs Central DP for mean estimation."""
    
    print("=" * 60)
    print("LDP vs Central DP: Accuracy Comparison")
    print("=" * 60)
    
    # Problem: Estimate mean of values in [0, 1]
    true_mean = 0.6
    true_std = 0.2
    n_users_range = [100, 500, 1000, 5000, 10000]
    epsilon = 2.0
    sensitivity = 1.0  # Range is [0, 1]
    
    n_trials = 100
    
    ldp_errors = []
    cdp_errors = []
    
    for n_users in n_users_range:
        ldp_trial_errors = []
        cdp_trial_errors = []
        
        for _ in range(n_trials):
            # Generate true values
            true_values = np.clip(np.random.normal(true_mean, true_std, n_users), 0, 1)
            
            # LDP: Each user adds their own noise
            # Using Laplace mechanism for each user
            ldp_scale = sensitivity / epsilon
            ldp_noisy_values = true_values + np.random.laplace(0, ldp_scale, n_users)
            ldp_estimate = np.mean(ldp_noisy_values)
            ldp_trial_errors.append(abs(ldp_estimate - true_mean))
            
            # Central DP: Aggregate first, then add noise
            true_sum = np.sum(true_values)
            cdp_scale = sensitivity / epsilon  # Sensitivity of sum is n × range / n = range
            cdp_noisy_sum = true_sum + np.random.laplace(0, cdp_scale)
            cdp_estimate = cdp_noisy_sum / n_users
            cdp_trial_errors.append(abs(cdp_estimate - true_mean))
        
        ldp_errors.append(np.mean(ldp_trial_errors))
        cdp_errors.append(np.mean(cdp_trial_errors))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(n_users_range, ldp_errors, 'r-o', label='LDP', linewidth=2, markersize=8)
    ax.plot(n_users_range, cdp_errors, 'b-s', label='Central DP', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Users')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title(f'LDP vs Central DP Accuracy (ε={epsilon})')
    ax.legend()
    ax.grid(True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('ldp_vs_cdp_accuracy.png', dpi=150)
    plt.show()
    
    print(f"\nMean estimation (ε={epsilon}):")
    print(f"\n{'n_users':<10} {'LDP Error':<15} {'CDP Error':<15} {'Ratio':<10}")
    print("-" * 50)
    for n, ldp_err, cdp_err in zip(n_users_range, ldp_errors, cdp_errors):
        ratio = ldp_err / cdp_err if cdp_err > 0 else float('inf')
        print(f"{n:<10} {ldp_err:<15.4f} {cdp_err:<15.4f} {ratio:<10.1f}x")
    
    print("\nKey Insight: LDP error ∝ 1/√n, Central DP error ∝ 1/n")
    print("LDP needs n² more users for same accuracy as Central DP!")


# ============================================================
# PART 5: LDP in Federated Learning
# ============================================================

def explain_ldp_in_fl():
    """
    LDP in Federated Learning Context
    ==================================
    
    Two main approaches to privacy in FL:
    
    1. CENTRAL DP (Client-Level DP):
       - Server clips + adds noise to aggregated updates
       - Requires trusted aggregator
       - What Opacus + Flower do
       - Better accuracy for same privacy
    
    2. LOCAL DP:
       - Each client adds noise BEFORE sending update
       - No trusted aggregator needed
       - Much more noise required
       - Used when server is untrusted
    
    LDP-FL Trade-offs:
    
    Pros:
    - No trust in server required
    - Privacy guaranteed even if server is compromised
    - Each client controls their own privacy
    
    Cons:
    - Much worse accuracy (√n factor)
    - Need many more clients or rounds
    - Convergence is slower
    
    When to use LDP in FL:
    - Cross-device FL with millions of clients
    - When server cannot be trusted
    - For aggregate statistics, not full models
    
    Hybrid approaches:
    - Secure Aggregation + Central DP
    - TEE (Trusted Execution Environment) + Central DP
    - Shuffled Model (between LDP and CDP)
    
    The Shuffled Model:
    - Clients send LDP reports
    - A trusted shuffler randomizes the order
    - Provides privacy amplification
    - Privacy between LDP and Central DP
    """
    print(explain_ldp_in_fl.__doc__)


def demo_ldp_gradient():
    """Demonstrate LDP for gradient sharing in FL."""
    
    print("=" * 60)
    print("LDP Gradient Perturbation in FL")
    print("=" * 60)
    
    # Simulate: 100 clients each compute a gradient (scalar for simplicity)
    n_clients = 100
    true_gradient = 0.5  # True "global" gradient direction
    
    # Each client has slightly different local gradient (heterogeneity)
    local_gradients = np.random.normal(true_gradient, 0.1, n_clients)
    
    # Clip gradients (common in DP-SGD)
    clip_norm = 1.0
    clipped_gradients = np.clip(local_gradients, -clip_norm, clip_norm)
    
    epsilons = [0.5, 1.0, 2.0, 4.0, 8.0]
    
    ldp_means = []
    cdp_means = []
    
    for epsilon in epsilons:
        # LDP: Each client adds noise before sending
        ldp_scale = 2 * clip_norm / epsilon  # Sensitivity is 2C for range [-C, C]
        ldp_noisy = clipped_gradients + np.random.laplace(0, ldp_scale, n_clients)
        ldp_agg = np.mean(ldp_noisy)
        ldp_means.append(ldp_agg)
        
        # Central DP: Aggregate first, noise once
        mean_gradient = np.mean(clipped_gradients)
        cdp_scale = 2 * clip_norm / (epsilon * n_clients)  # Sensitivity of mean
        cdp_noisy = mean_gradient + np.random.laplace(0, cdp_scale)
        cdp_means.append(cdp_noisy)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(epsilons))
    width = 0.35
    
    ax.bar(x - width/2, np.abs(np.array(ldp_means) - true_gradient), width, 
           label='LDP Error', color='red', alpha=0.7)
    ax.bar(x + width/2, np.abs(np.array(cdp_means) - true_gradient), width, 
           label='Central DP Error', color='blue', alpha=0.7)
    
    ax.set_xlabel('Epsilon (ε)')
    ax.set_ylabel('Absolute Error from True Gradient')
    ax.set_title(f'LDP vs Central DP Gradient Aggregation ({n_clients} clients)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'ε={e}' for e in epsilons])
    ax.legend()
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('ldp_gradient.png', dpi=150)
    plt.show()
    
    print(f"\nTrue gradient: {true_gradient}")
    print(f"Number of clients: {n_clients}")
    print(f"\n{'Epsilon':<10} {'LDP Mean':<15} {'CDP Mean':<15}")
    print("-" * 40)
    for eps, ldp, cdp in zip(epsilons, ldp_means, cdp_means):
        print(f"{eps:<10} {ldp:<15.4f} {cdp:<15.4f}")
    
    print("\nKey Insight: LDP adds noise at each client → much noisier aggregate")


# ============================================================
# MAIN: Run all demos
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LOCAL DIFFERENTIAL PRIVACY (LDP) TUTORIAL")
    print("=" * 70 + "\n")
    
    # Part 1: LDP vs CDP
    explain_ldp_vs_cdp()
    
    # Part 2: Randomized Response
    print("\n")
    demo_randomized_response()
    
    # Part 3: RAPPOR
    print("\n")
    demo_rappor_frequency()
    
    # Part 4: Accuracy comparison
    print("\n")
    compare_ldp_cdp_accuracy()
    
    # Part 5: LDP in FL
    print("\n")
    explain_ldp_in_fl()
    print("\n")
    demo_ldp_gradient()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. LDP: Randomize BEFORE sending (no trusted server needed)
       CDP: Randomize AFTER aggregation (needs trusted server)
    
    2. Randomized Response: Classic LDP technique for surveys
       - Flip your answer with some probability
       - Server can still estimate true statistics
    
    3. RAPPOR (Google) and Apple's LDP:
       - Used for telemetry at massive scale
       - Bloom filters + randomization
       - Estimate frequencies privately
    
    4. Accuracy tradeoff:
       - LDP error ∝ 1/√n (need n² more users!)
       - For same ε, LDP is MUCH noisier than CDP
    
    5. In Federated Learning:
       - LDP: Each client adds noise to their update
       - CDP: Server adds noise to aggregate
       - LDP is rarely used for model training (too noisy)
       - LDP works for aggregate statistics
    
    6. Next: Bayesian DP (probabilistic view of privacy)
    """)
