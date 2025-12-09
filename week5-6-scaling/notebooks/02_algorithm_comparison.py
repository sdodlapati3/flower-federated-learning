"""
Federated Optimization Algorithms Comparison
=============================================

This notebook provides a comprehensive comparison of federated
optimization algorithms:

1. FedAvg (McMahan et al., 2017) - Baseline averaging
2. FedProx (Li et al., 2020) - Proximal regularization  
3. SCAFFOLD (Karimireddy et al., 2020) - Control variates

We analyze:
- Convergence speed under different heterogeneity levels
- Communication efficiency
- Client drift analysis
- Trade-offs and recommendations
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


# ============================================================
# Algorithm Comparison Framework
# ============================================================

class FederatedAlgorithmAnalysis:
    """
    Framework for analyzing federated optimization algorithms.
    """
    
    def __init__(self):
        self.algorithms = {
            'FedAvg': {
                'name': 'Federated Averaging',
                'paper': 'McMahan et al., 2017',
                'idea': 'Average local updates weighted by sample count',
                'update_rule': 'w_{t+1} = Σ (n_k/n) * w_k^t',
                'pros': ['Simple', 'No extra communication', 'Works well with IID data'],
                'cons': ['Client drift', 'Slow with heterogeneous data', 'May diverge'],
                'communication': 'O(d) per round',  # d = model dimension
                'computation': 'O(K·E) per client',  # K batches, E epochs
                'when_to_use': 'IID or mildly heterogeneous data'
            },
            'FedProx': {
                'name': 'Federated Proximal',
                'paper': 'Li et al., 2020',
                'idea': 'Add proximal term to keep updates close to global',
                'update_rule': 'min_w f_k(w) + (μ/2)||w - w_global||²',
                'pros': ['Handles heterogeneity', 'Handles stragglers', 'Simple modification'],
                'cons': ['Extra hyperparameter μ', 'Still has some drift'],
                'communication': 'O(d) per round',
                'computation': 'O(K·E) per client',
                'when_to_use': 'Heterogeneous data, unreliable clients'
            },
            'SCAFFOLD': {
                'name': 'SCAFFOLD',
                'paper': 'Karimireddy et al., 2020',
                'idea': 'Use control variates to reduce variance',
                'update_rule': 'Use corrected gradient: ∇f_k - c_k + c',
                'pros': ['Variance reduction', 'Fast convergence', 'Provable guarantees'],
                'cons': ['2x communication', 'Memory for control variates'],
                'communication': 'O(2d) per round',
                'computation': 'O(K·E) per client',
                'when_to_use': 'Very heterogeneous data, full participation'
            }
        }
    
    def print_comparison_table(self):
        """Print a comparison table of algorithms."""
        print("=" * 90)
        print("FEDERATED OPTIMIZATION ALGORITHMS COMPARISON")
        print("=" * 90)
        
        for alg_key, alg in self.algorithms.items():
            print(f"\n{'─'*90}")
            print(f"  {alg['name']} ({alg_key})")
            print(f"  Paper: {alg['paper']}")
            print(f"{'─'*90}")
            print(f"  Core Idea: {alg['idea']}")
            print(f"  Update Rule: {alg['update_rule']}")
            print(f"\n  Communication: {alg['communication']}")
            print(f"  Computation: {alg['computation']}")
            print(f"\n  ✓ Pros:")
            for pro in alg['pros']:
                print(f"    - {pro}")
            print(f"  ✗ Cons:")
            for con in alg['cons']:
                print(f"    - {con}")
            print(f"\n  When to Use: {alg['when_to_use']}")
    
    def analyze_client_drift(self):
        """
        Analyze client drift phenomenon.
        """
        print("\n" + "=" * 70)
        print("CLIENT DRIFT ANALYSIS")
        print("=" * 70)
        
        print("""
Client drift occurs when:
- Local objectives f_k differ across clients
- Multiple local updates move away from global optimum
- Aggregated model doesn't minimize any single objective well

Mathematically:
- Let x* be global optimum: x* = argmin (1/K)Σf_k(x)
- Let x*_k be client k optimum: x*_k = argmin f_k(x)
- Drift = ||x*_k - x*||

Impact of heterogeneity (Dirichlet α):
┌──────────────────────────────────────────────────────────┐
│  α     │ Heterogeneity │ Drift    │ FedAvg Performance  │
├──────────────────────────────────────────────────────────┤
│  0.01  │ Extreme       │ Very High│ May diverge         │
│  0.1   │ High          │ High     │ Slow, suboptimal    │
│  0.5   │ Moderate      │ Medium   │ Works reasonably    │
│  1.0   │ Low           │ Low      │ Works well          │
│  10+   │ Near IID      │ Minimal  │ Works excellently   │
└──────────────────────────────────────────────────────────┘

How algorithms address drift:
1. FedAvg: Does nothing - hopes averaging reduces drift
2. FedProx: Regularizes toward global - limits how far clients can go
3. SCAFFOLD: Corrects gradient direction - removes drift before it happens
""")
    
    def convergence_theory(self):
        """
        Theoretical convergence analysis.
        """
        print("\n" + "=" * 70)
        print("CONVERGENCE THEORY")
        print("=" * 70)
        
        print("""
Convergence rates (for strongly convex, smooth functions):

1. FedAvg (with bounded gradient dissimilarity):
   - Rate: O(1/(μT)) + O(σ²/(μK)) + O(Γ)
   - Γ = gradient dissimilarity term (heterogeneity impact)
   - May not converge to optimum if Γ is large!

2. FedProx (Li et al., 2020):
   - Rate: O(1/(T)) for μ-strongly convex
   - Handles partial participation
   - Converges but may be slow with large heterogeneity

3. SCAFFOLD (Karimireddy et al., 2020):
   - Rate: O(1/(μT)) + O(σ²/(μTK))
   - No gradient dissimilarity term!
   - Variance reduction eliminates heterogeneity impact on convergence

Communication complexity to reach ε accuracy:
┌───────────────────────────────────────────────────────────┐
│  Algorithm  │ Rounds    │ Bits per round │ Total bits     │
├───────────────────────────────────────────────────────────┤
│  FedAvg     │ O(Γ/ε)    │ O(d)           │ O(dΓ/ε)        │
│  FedProx    │ O(1/ε)    │ O(d)           │ O(d/ε)         │
│  SCAFFOLD   │ O(1/ε)    │ O(2d)          │ O(d/ε)         │
└───────────────────────────────────────────────────────────┘

Key insight: SCAFFOLD has 2x communication per round but fewer rounds,
so total communication is often lower for heterogeneous settings.
""")
    
    def practical_recommendations(self):
        """
        Practical recommendations for choosing algorithms.
        """
        print("\n" + "=" * 70)
        print("PRACTICAL RECOMMENDATIONS")
        print("=" * 70)
        
        print("""
Decision tree for algorithm selection:

                    Is data IID?
                    /          \\
                  YES           NO
                   |             |
              Use FedAvg    Is participation rate high?
                             /                    \\
                           YES                     NO
                            |                      |
                    Use SCAFFOLD           Can you add regularization?
                                             /              \\
                                           YES               NO
                                            |                |
                                      Use FedProx      Use FedAvg + 
                                                       gradient clipping

Hyperparameter guidelines:

FedProx μ selection:
- Start with μ = 0.01
- If still diverging: increase to μ = 0.1
- If converging too slowly: decrease to μ = 0.001
- Higher μ → more stable but slower

SCAFFOLD considerations:
- Best with full participation or high rate (>50%)
- Control variates take rounds to stabilize
- Warmup period may be needed

Local epochs E:
- FedAvg: E ∈ [1, 5] for heterogeneous, E ∈ [5, 20] for IID
- FedProx: Can use larger E due to regularization
- SCAFFOLD: Can use larger E due to variance reduction

Learning rate:
- Start with standard SGD tuning
- FedProx: may need slightly lower due to proximal term
- SCAFFOLD: can often use higher due to variance reduction
""")
    
    def generate_synthetic_comparison(self):
        """
        Generate synthetic comparison plots.
        """
        np.random.seed(42)
        
        # Simulated convergence curves
        rounds = np.arange(1, 51)
        
        # High heterogeneity (α = 0.1)
        fedavg_high = 0.7 * (1 - np.exp(-0.05 * rounds)) + np.random.normal(0, 0.02, len(rounds))
        fedprox_high = 0.75 * (1 - np.exp(-0.08 * rounds)) + np.random.normal(0, 0.015, len(rounds))
        scaffold_high = 0.8 * (1 - np.exp(-0.12 * rounds)) + np.random.normal(0, 0.01, len(rounds))
        
        # Low heterogeneity (α = 1.0)
        fedavg_low = 0.82 * (1 - np.exp(-0.1 * rounds)) + np.random.normal(0, 0.01, len(rounds))
        fedprox_low = 0.83 * (1 - np.exp(-0.1 * rounds)) + np.random.normal(0, 0.01, len(rounds))
        scaffold_low = 0.85 * (1 - np.exp(-0.11 * rounds)) + np.random.normal(0, 0.01, len(rounds))
        
        # Make curves monotonic (smoothed)
        from scipy.ndimage import uniform_filter1d
        fedavg_high = uniform_filter1d(np.maximum.accumulate(fedavg_high), 3)
        fedprox_high = uniform_filter1d(np.maximum.accumulate(fedprox_high), 3)
        scaffold_high = uniform_filter1d(np.maximum.accumulate(scaffold_high), 3)
        fedavg_low = uniform_filter1d(np.maximum.accumulate(fedavg_low), 3)
        fedprox_low = uniform_filter1d(np.maximum.accumulate(fedprox_low), 3)
        scaffold_low = uniform_filter1d(np.maximum.accumulate(scaffold_low), 3)
        
        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: High heterogeneity
        ax = axes[0]
        ax.plot(rounds, fedavg_high, 'r-', label='FedAvg', linewidth=2)
        ax.plot(rounds, fedprox_high, 'g--', label='FedProx (μ=0.1)', linewidth=2)
        ax.plot(rounds, scaffold_high, 'b-', label='SCAFFOLD', linewidth=2)
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('High Heterogeneity (α=0.1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Plot 2: Low heterogeneity
        ax = axes[1]
        ax.plot(rounds, fedavg_low, 'r-', label='FedAvg', linewidth=2)
        ax.plot(rounds, fedprox_low, 'g--', label='FedProx (μ=0.1)', linewidth=2)
        ax.plot(rounds, scaffold_low, 'b-', label='SCAFFOLD', linewidth=2)
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Low Heterogeneity (α=1.0)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Plot 3: Communication efficiency
        ax = axes[2]
        # Total communication: SCAFFOLD sends 2x per round
        comm_fedavg = rounds
        comm_fedprox = rounds
        comm_scaffold = 2 * rounds
        
        ax.plot(comm_fedavg, fedavg_high, 'r-', label='FedAvg', linewidth=2)
        ax.plot(comm_fedprox, fedprox_high, 'g--', label='FedProx', linewidth=2)
        ax.plot(comm_scaffold, scaffold_high, 'b-', label='SCAFFOLD', linewidth=2)
        ax.set_xlabel('Total Communication (model uploads)')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Communication Efficiency (α=0.1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=150)
        print("Saved: algorithm_comparison.png")
        plt.show()


def main():
    """Run algorithm comparison analysis."""
    
    analysis = FederatedAlgorithmAnalysis()
    
    print("=" * 70)
    print("FEDERATED LEARNING OPTIMIZATION ALGORITHMS")
    print("A Comprehensive Comparison Guide")
    print("=" * 70)
    
    # Print comparison
    analysis.print_comparison_table()
    
    # Client drift analysis
    analysis.analyze_client_drift()
    
    # Convergence theory
    analysis.convergence_theory()
    
    # Practical recommendations
    analysis.practical_recommendations()
    
    # Generate comparison plot
    try:
        from scipy.ndimage import uniform_filter1d
        analysis.generate_synthetic_comparison()
    except ImportError:
        print("\n[Skipping plot - scipy not available]")
    
    # Summary table
    print("\n" + "=" * 70)
    print("QUICK REFERENCE SUMMARY")
    print("=" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEDERATED OPTIMIZATION CHEAT SHEET                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ALGORITHM     │ BEST FOR              │ KEY PARAMETER   │ OVERHEAD     │
│  ─────────────────────────────────────────────────────────────────────  │
│  FedAvg        │ IID data              │ Local epochs E  │ None         │
│  FedProx       │ Heterogeneous/dropout │ Proximal μ      │ None         │
│  SCAFFOLD      │ Very heterogeneous    │ Control var     │ 2x comm      │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  HETEROGENEITY (Dirichlet α)                                            │
│  α < 0.1  : Extreme - Use SCAFFOLD or FedProx with high μ               │
│  α ∈ [0.1, 0.5]: High - FedProx recommended, SCAFFOLD if full part.     │
│  α ∈ [0.5, 1.0]: Moderate - FedAvg works, FedProx for stability         │
│  α > 1.0  : Low/IID - FedAvg is sufficient                              │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  COMMON ISSUES AND FIXES                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  Issue: Model diverges                                                   │
│  Fix: Reduce lr, increase μ (FedProx), use gradient clipping            │
│                                                                          │
│  Issue: Slow convergence                                                 │
│  Fix: Increase local epochs, try SCAFFOLD, tune μ down                  │
│                                                                          │
│  Issue: Accuracy gap between rounds                                      │
│  Fix: Increase clients per round, reduce heterogeneity (sampling)       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
