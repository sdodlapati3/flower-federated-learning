# Weeks 3-4: DP Theory Deep Dive & Paper Replication

## Overview
This module covers differential privacy variants (Approximate, Rényi, Local, Bayesian DP) and replicates a DP-FL research paper with extensions.

---

## Week 3: DP Variants Deep Dive

### 3.1 Approximate DP (ε,δ)-DP [Day 1-2]
**Objective**: Understand relaxed DP and composition theorems

**Key Concepts**:
- Pure DP (ε-DP) vs Approximate DP (ε,δ)-DP
- δ represents probability of "catastrophic" privacy failure
- Basic composition: (ε₁+ε₂, δ₁+δ₂)
- Advanced composition: (ε√(2k·ln(1/δ')), kδ+δ')

**Implementation**: `notebooks/01_approximate_dp.py`
- Demonstrate Gaussian mechanism
- Show composition theorem in practice
- Visualize privacy budget consumption

**Reading**:
- Dwork & Roth Ch. 3.5 (Approximate DP)
- Kamath's lecture notes on approximate DP

---

### 3.2 Rényi DP (RDP) [Day 3-4]
**Objective**: Understand why Opacus uses RDP internally

**Key Concepts**:
- Rényi divergence of order α
- RDP gives tighter composition bounds
- Conversion: RDP → (ε,δ)-DP
- Sampled Gaussian mechanism + RDP

**Implementation**: `notebooks/02_renyi_dp.py`
- Implement RDP accountant from scratch
- Compare RDP vs naive composition
- Show Opacus's internal RDP calculations

**Reading**:
- Mironov, "Rényi Differential Privacy" (2017)
- Wang et al., "Subsampled Rényi Differential Privacy"

---

### 3.3 Local DP (LDP) [Day 5]
**Objective**: Understand extreme privacy (randomize before sending)

**Key Concepts**:
- Client-side randomization (no trusted server)
- Randomized Response (classic technique)
- RAPPOR (Google), Apple's LDP deployment
- Higher noise than central DP for same ε

**Implementation**: `notebooks/03_local_dp.py`
- Implement randomized response
- RAPPOR-style frequency estimation
- Compare LDP vs CDP accuracy

**Reading**:
- Bebensee, "Local Differential Privacy: A Tutorial"
- Cormode, "Privacy at Scale: LDP in Practice"

---

### 3.4 Bayesian DP [Day 6-7]
**Objective**: Understand probabilistic view of DP

**Key Concepts**:
- DP through posterior sampling
- Bayesian estimation of privacy leakage
- Connection to PAC-Bayes bounds
- "Probably Approximately Correct" privacy

**Implementation**: `notebooks/04_bayesian_dp.py`
- Simple Bayesian inference with DP guarantees
- Posterior sampling mechanism
- Estimate ε from observed outputs

**Reading**:
- Dimitrakakis et al., "DP for Bayesian Inference through Posterior Sampling"
- Triastcyn & Faltings, "Bayesian DP for Machine Learning"

---

## Week 4: Paper Replication & Extension

### 4.1 Paper Selection & Analysis [Day 1]
**Target Paper**: "A Differential Privacy Approach in Federated Learning" (Tayyeh et al., 2024)

**Key Findings to Replicate**:
1. Impact of ε on accuracy: Lower ε → Lower accuracy
2. Impact of clipping norm C: Too small clips useful gradients, too large adds noise
3. Impact of client fraction K: More clients per round → Better convergence
4. IID vs Non-IID: Non-IID amplifies accuracy degradation

---

### 4.2 Experimental Framework [Day 2-3]
**Implementation**: `experiments/paper_replication/`

**Hyperparameter Grid**:
```
ε ∈ {1, 2, 4, 8, 16}
C (clipping norm) ∈ {0.5, 1.0, 2.0, 5.0}
K (clients/round) ∈ {2, 5, 10}
Data split: IID, Non-IID (α=0.1, 0.5, 1.0)
```

**Metrics to Track**:
- Final accuracy
- Convergence speed (rounds to X% accuracy)
- Per-round ε consumption
- Gradient norm distribution (before/after clipping)

---

### 4.3 Extension: Uncertainty Analysis [Day 4-5]
**Objective**: How does DP affect model uncertainty?

**Metrics**:
- Expected Calibration Error (ECE)
- Predictive entropy
- Brier score
- Confidence vs accuracy correlation

**Hypothesis**: DP noise should increase uncertainty but may improve calibration (prevent overconfidence)

---

### 4.4 Write-up [Day 6-7]
**Deliverable**: 4-6 page technical report

**Structure**:
1. Introduction & Paper Summary
2. Methodology (FL + DP-SGD setup)
3. Replication Results (tables, plots)
4. Extension: Uncertainty Analysis
5. Discussion & Insights
6. Conclusion

---

## Directory Structure
```
week3-4-dp-theory/
├── PLAN.md                          # This file
├── notebooks/
│   ├── 01_approximate_dp.py         # (ε,δ)-DP theory & demos
│   ├── 02_renyi_dp.py               # RDP & privacy accountants
│   ├── 03_local_dp.py               # LDP mechanisms
│   ├── 04_bayesian_dp.py            # Bayesian DP concepts
│   └── 05_dp_comparison.py          # Compare all variants
├── experiments/
│   ├── paper_replication/
│   │   ├── run_sweep.py             # Hyperparameter sweep script
│   │   ├── run_sweep.slurm          # SLURM job script
│   │   ├── analyze_results.py       # Analysis & plotting
│   │   └── configs/                 # Experiment configurations
│   └── uncertainty_analysis/
│       ├── calibration_metrics.py   # ECE, Brier score
│       └── run_uncertainty.py       # DP + uncertainty experiments
└── notes/
    ├── dp_variants_comparison.md    # 2-3 page comparison notes
    └── paper_replication_report.md  # 4-6 page writeup
```

---

## Key Deliverables

### Week 3
- [ ] `01_approximate_dp.py` - Working notebook with demos
- [ ] `02_renyi_dp.py` - RDP implementation + Opacus internals
- [ ] `03_local_dp.py` - LDP mechanisms
- [ ] `04_bayesian_dp.py` - Bayesian DP concepts
- [ ] `dp_variants_comparison.md` - 2-3 page comparison notes

### Week 4
- [ ] `run_sweep.py` - Paper replication framework
- [ ] Complete hyperparameter sweep results
- [ ] Uncertainty analysis extension
- [ ] `paper_replication_report.md` - 4-6 page technical report
