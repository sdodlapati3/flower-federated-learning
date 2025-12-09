# Week 1-2: Federated Learning Foundations

## Overview
**Duration**: 2 weeks (completed)  
**Status**: ✅ COMPLETE  
**Focus**: FL fundamentals, Flower framework, DP integration

---

## Week 1: Flower Framework & FL Basics

### Day 1-2: Core Flower Tutorial
**Status**: ✅ Complete

| Part | Topic | Location | Status |
|------|-------|----------|--------|
| Part 1 | FedAvg with Simulation | `flower-tutorial/` | ✅ |
| Part 2 | Custom Strategy | `flower-tutorial/` | ✅ |
| Part 3 | W&B Integration | `flower-tutorial/` | ✅ |
| Part 4 | Custom Messages | `flower-tutorial/` | ✅ |

**Key Concepts Learned**:
- Flower's `ClientApp` and `ServerApp` architecture
- `flwr run` simulation mode
- Custom strategy implementation (subclassing `FedAvg`)
- Metrics aggregation and logging
- Client-server message customization

### Day 3-4: Framework Quickstarts
**Status**: ✅ Complete

| Framework | Location | Key Learning |
|-----------|----------|--------------|
| XGBoost | `xgboost-quickstart/` | Federated gradient boosting |
| Scikit-learn | `sklearn-quickstart/` | Federated logistic regression |
| JAX | `jax-quickstart/` | Functional FL with JAX |
| Pandas | `pandas-quickstart/` | Federated data analytics |
| Lightning | `lightning-quickstart/` | PyTorch Lightning integration |

### Day 5: Additional Tutorials
**Status**: ✅ Complete

| Tutorial | Location | Key Learning |
|----------|----------|--------------|
| TensorFlow | `tensorflow-quickstart/` | Keras model federation |
| Vertical FL | `vertical-fl/` | Split learning across features |
| Federated VAE | `federated-vae/` | Generative models in FL |
| Tabular FL | `tabular-fl/` | Structured data federation |
| Secure Aggregation | `secure-aggregation/` | SecAgg+ protocol |
| Kaplan-Meier | `kaplan-meier-fl/` | Federated survival analysis |

---

## Week 2: Non-IID Data & Differential Privacy

### Day 1-2: Advanced PyTorch & Non-IID
**Status**: ✅ Complete

| Topic | Location | Details |
|-------|----------|---------|
| Stateful Clients | `advanced-pytorch/` | Client state persistence |
| Dirichlet Partitioning | `advanced-pytorch/` | Non-IID data simulation |
| Simulation PyTorch | `simulation-pytorch/` | Large-scale simulation |

**Key Concepts**:
- Dirichlet distribution for label imbalance: $p \sim Dir(\alpha)$
- Lower α → more heterogeneous data
- Client drift in heterogeneous settings

### Day 3-4: Differential Privacy Integration
**Status**: ✅ Complete

| Implementation | Location | Key Features |
|----------------|----------|--------------|
| Opacus DP-SGD | `opacus-dp/` | Sample-level DP |
| FL-DP-SA | `fl-dp-sa/` | DP + Secure Aggregation |

**Key Concepts**:
- (ε, δ)-Differential Privacy
- Gaussian mechanism: $\sigma = \frac{\Delta f \sqrt{2\ln(1.25/\delta)}}{\varepsilon}$
- Gradient clipping: `max_grad_norm`
- Noise multiplier calibration
- Privacy accounting with Opacus

### Day 5: Multi-Epsilon Experiments
**Status**: ✅ Complete

**Location**: `dp-experiments/`

**Experiment Design**:
```
Configurations:
├── IID + ε=2 (high privacy, noise_mult=2.0)
├── IID + ε=4 (medium privacy, noise_mult=1.0)
├── IID + ε=8 (low privacy, noise_mult=0.5)
├── Non-IID + ε=2
├── Non-IID + ε=4
└── Non-IID + ε=8
```

**Results** (Job 1206):

| Configuration | Accuracy | Epsilon | Privacy Level |
|---------------|----------|---------|---------------|
| IID, ε=2 | 17.16% | 0.11 | High |
| IID, ε=4 | 31.62% | 0.33 | Medium |
| IID, ε=8 | 42.21% | 4.63 | Low |
| Non-IID, ε=2 | 15.55% | 0.11 | High |
| Non-IID, ε=4 | 29.47% | 0.36 | Medium |
| Non-IID, ε=8 | 36.61% | 4.81 | Low |

**Key Findings**:
1. **Privacy-Utility Tradeoff**: Clear inverse relationship between privacy (ε) and accuracy
2. **Non-IID Penalty**: 1.6% - 5.6% accuracy drop compared to IID
3. **Practical Threshold**: ε ≈ 4-8 provides reasonable utility

---

## Skills Acquired

### Technical Skills
- [x] Flower framework architecture
- [x] Custom FL strategy implementation
- [x] Non-IID data partitioning (Dirichlet)
- [x] Opacus DP-SGD integration
- [x] Privacy budget tracking
- [x] SLURM job management on HPC

### Theoretical Understanding
- [x] FedAvg algorithm and convergence
- [x] Client drift in heterogeneous settings
- [x] (ε, δ)-Differential Privacy definition
- [x] Gaussian mechanism and sensitivity
- [x] Composition theorems (basic)

---

## Directory Structure

```
flower-federated-learning/
├── flower-tutorial/          # Parts 1-4 of main tutorial
├── xgboost-quickstart/       # XGBoost federation
├── sklearn-quickstart/       # Scikit-learn federation
├── jax-quickstart/           # JAX federation
├── pandas-quickstart/        # Pandas analytics
├── lightning-quickstart/     # PyTorch Lightning
├── tensorflow-quickstart/    # TensorFlow/Keras
├── vertical-fl/              # Vertical federated learning
├── federated-vae/            # Federated VAE
├── tabular-fl/               # Tabular data FL
├── secure-aggregation/       # SecAgg+ protocol
├── kaplan-meier-fl/          # Survival analysis
├── advanced-pytorch/         # Stateful clients, non-IID
├── simulation-pytorch/       # Large-scale simulation
├── opacus-dp/                # Opacus DP-SGD
├── fl-dp-sa/                 # DP + Secure Aggregation
└── dp-experiments/           # Multi-epsilon experiments
```

---

## SLURM Jobs Completed

| Job ID | Description | Status |
|--------|-------------|--------|
| 1190-1195 | Quickstart tutorials | ✅ |
| 1196 | Advanced PyTorch | ✅ |
| 1199-1204 | Additional tutorials | ✅ |
| 1206 | DP experiments | ✅ |

---

## Next Steps → Week 3-4

See `week3-4-dp-theory/PLAN.md` for:
- DP variants deep dive (Approximate, Rényi, Local, Bayesian)
- Paper replication (Tayyeh et al., 2024)
- Advanced privacy accounting
