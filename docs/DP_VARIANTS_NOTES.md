# Differential Privacy Variants - Theory Notes

## Overview

This document summarizes the key DP variants relevant to federated learning, as studied during Weeks 3-4 of the upskilling plan.

---

## 1. Pure Differential Privacy (ε-DP)

### Definition
A randomized mechanism M satisfies ε-differential privacy if for all neighboring datasets D, D' (differing in one record) and all outputs S:

$$P[M(D) \in S] \leq e^\varepsilon \cdot P[M(D') \in S]$$

### Key Properties
- **Privacy parameter ε**: Lower = more private (ε=0 is perfect privacy)
- **Composition**: ε values add under sequential composition
- **Group privacy**: For k records differing, multiply ε by k

### Mechanisms
- **Laplace Mechanism**: Add Lap(Δf/ε) noise for sensitivity Δf
- **Exponential Mechanism**: For non-numeric outputs

### Limitation
Pure DP is often too restrictive for deep learning (requires too much noise).

---

## 2. Approximate Differential Privacy ((ε,δ)-DP)

### Definition
A mechanism M satisfies (ε,δ)-DP if for all neighboring D, D' and outputs S:

$$P[M(D) \in S] \leq e^\varepsilon \cdot P[M(D') \in S] + \delta$$

### Interpretation
- **δ**: Probability of "catastrophic" privacy failure
- Typically set δ < 1/n where n = dataset size
- Common choice: δ = 1e-5 or 1e-6

### Key Properties
- Allows **Gaussian mechanism**: Add N(0, σ²) where σ = Δf·√(2ln(1.25/δ))/ε
- **Advanced composition**: After k compositions:
  - Naïve: (kε, kδ)
  - Advanced: (ε√(2k·ln(1/δ')), kδ + δ')

### Use in FL
- Standard for DP-SGD (Abadi et al., 2016)
- Opacus reports (ε, δ) after training

---

## 3. Rényi Differential Privacy (RDP)

### Definition
A mechanism M satisfies (α, ε)-RDP if for all neighboring D, D':

$$D_\alpha(M(D) \| M(D')) \leq \varepsilon$$

where D_α is the Rényi divergence of order α:

$$D_\alpha(P \| Q) = \frac{1}{\alpha-1} \log \mathbb{E}_{x \sim Q}\left[\left(\frac{P(x)}{Q(x)}\right)^\alpha\right]$$

### Why RDP?
1. **Tighter composition**: RDP composes linearly (just add ε values for same α)
2. **Subsampling amplification**: Privacy amplifies under Poisson subsampling
3. **Conversion to (ε,δ)-DP**: Can convert RDP → (ε,δ)-DP at end

### RDP Composition
For k mechanisms each satisfying (α, ε_i)-RDP:
$$\text{Composed mechanism satisfies } (\alpha, \sum_{i=1}^k \varepsilon_i)\text{-RDP}$$

### Conversion to (ε,δ)-DP
Given (α, ε)-RDP, for any δ > 0:
$$\varepsilon_{(\varepsilon,\delta)} = \varepsilon + \frac{\log(1/\delta)}{\alpha - 1}$$

Optimize over α to get tightest bound.

### Use in FL
- **Opacus** uses RDP internally for privacy accounting
- **Subsampled Gaussian mechanism**: Key for DP-SGD with mini-batches
- Our `PrivacyAccountant` implements RDP-based accounting

---

## 4. Local Differential Privacy (LDP)

### Definition
Each user applies randomization locally before sending data:

$$P[M(x) \in S] \leq e^\varepsilon \cdot P[M(x') \in S]$$

for any two possible user inputs x, x'.

### Key Difference from Central DP
| Aspect | Central DP | Local DP |
|--------|-----------|----------|
| Trust model | Trusted curator | No trusted party |
| Data collection | Raw data collected | Noisy data collected |
| Accuracy | Higher | Lower (more noise needed) |
| Use case | ML training | Telemetry, surveys |

### Common LDP Mechanisms
1. **Randomized Response** (binary data):
   - With prob. p, report true value
   - With prob. 1-p, flip the bit
   - ε = ln(p/(1-p))

2. **RAPPOR** (Google Chrome):
   - Bloom filter + randomized response
   - Used for telemetry

3. **Apple's LDP**:
   - Count Mean Sketch for frequency estimation

### LDP in FL Context
- **Shuffle Model**: Middle ground between local and central
- LDP per-client → central DP after aggregation
- Higher communication cost but stronger trust model

---

## 5. Concentrated Differential Privacy (zCDP)

### Definition
A mechanism M satisfies ρ-zCDP if for all neighboring D, D' and all α > 1:

$$D_\alpha(M(D) \| M(D')) \leq \rho \cdot \alpha$$

### Properties
- **Linear in α**: Simplifies analysis
- **Composition**: ρ values add directly
- **Gaussian mechanism**: Adding N(0, σ²) to f with sensitivity Δf satisfies (Δf²/(2σ²))-zCDP

### Conversion
- zCDP → RDP: (α, ρα)-RDP for all α
- zCDP → (ε,δ)-DP: ε = ρ + 2√(ρ·log(1/δ))

---

## 6. Bayesian Differential Privacy

### Motivation
Standard DP is worst-case over all datasets. Bayesian DP considers:
- Prior beliefs about data
- Expected privacy loss

### Approaches

#### 6.1 Posterior Sampling (Dimitrakakis et al.)
- Release sample from posterior P(θ|D)
- Privacy depends on sensitivity of posterior

#### 6.2 Bayesian DP for ML (Triastcyn & Faltings)
Data-dependent privacy bounds:
- Compute privacy loss on actual data
- Often tighter than worst-case

#### 6.3 Bayesian Privacy Accounting
- Model ε as random variable
- Estimate distribution of privacy loss
- Report credible intervals

### Use Cases
- When worst-case bounds are too pessimistic
- Uncertainty quantification in DP
- Adaptive composition strategies

---

## Comparison Summary

| Variant | Composition | Tightness | Trust Model | Primary Use |
|---------|-------------|-----------|-------------|-------------|
| Pure ε-DP | Linear (add ε) | Loose | Central | Theory baseline |
| (ε,δ)-DP | Advanced comp. | Moderate | Central | Standard ML |
| RDP | Linear in ε | Tight | Central | DP-SGD accounting |
| zCDP | Linear in ρ | Tight | Central | Gaussian mechanisms |
| LDP | Linear | Very loose | Local | Telemetry |
| Bayesian | Data-dependent | Varies | Central | Adaptive methods |

---

## Implementation in fl_research

```python
from fl_research.privacy import (
    PrivacyAccountant,      # RDP-based accounting
    get_privacy_spent,       # Quick (ε,δ) calculation
    GaussianMechanism,       # (ε,δ)-DP noise
    LaplaceMechanism,        # ε-DP noise
)

# RDP-based accounting (recommended for DP-SGD)
accountant = PrivacyAccountant(
    noise_multiplier=1.0,
    sample_rate=0.01,
    delta=1e-5,
    max_epsilon=10.0,
)

# Track privacy over training
for epoch in range(100):
    accountant.step(num_steps=len(dataloader))
    if accountant.is_budget_exceeded():
        break
        
epsilon = accountant.get_epsilon()
```

---

## Key References

1. **Dwork & Roth (2014)**: *The Algorithmic Foundations of Differential Privacy*
2. **Mironov (2017)**: *Rényi Differential Privacy*
3. **Abadi et al. (2016)**: *Deep Learning with Differential Privacy*
4. **Bun & Steinke (2016)**: *Concentrated Differential Privacy*
5. **Duchi et al. (2013)**: *Local Privacy and Statistical Minimax Rates*
6. **Triastcyn & Faltings (2019)**: *Bayesian Differential Privacy for Machine Learning*
