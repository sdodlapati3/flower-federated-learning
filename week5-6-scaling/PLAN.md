# Weeks 5-6: HPC Scaling & Distributed Optimization

## Overview
This module covers scaling FL to HPC systems, distributed optimization theory (ADMM, consensus), and hyperparameter optimization for FL systems.

---

## Week 5: Distributed Optimization Theory & FedML

### 5.1 Distributed Optimization Foundations [Day 1-2]
**Objective**: Understand mathematical backbone of FL algorithms

**Key Concepts**:
- Gradient descent in distributed setting
- Consensus optimization: reaching agreement across nodes
- Communication-computation tradeoffs
- Convergence analysis with partial participation

**Implementation**: `notebooks/01_distributed_optimization.py`
- Visualize consensus convergence
- Compare centralized vs decentralized optimization
- Implement basic distributed gradient descent

**Reading**:
- Nedić & Ozdaglar, "Distributed Subgradient Methods"
- Boyd et al., "Distributed Optimization via ADMM" (Ch. 1-3)

---

### 5.2 ADMM for FL [Day 3-4]
**Objective**: Understand ADMM and its FL variants (FedADMM, FedProx)

**Key Concepts**:
- Augmented Lagrangian method
- ADMM splitting: x-update, z-update, dual update
- FedProx: proximal term for client drift
- FedADMM: consensus ADMM in federated setting

**Implementation**: `experiments/fedprox/`
- Implement FedProx from scratch
- Compare FedAvg vs FedProx on Non-IID data
- Analyze convergence speed

**Reading**:
- Boyd et al., "Distributed Optimization via ADMM" (Ch. 7-8)
- Li et al., "Federated Optimization in Heterogeneous Networks" (FedProx)

---

### 5.3 FedML Integration [Day 5-7]
**Objective**: Use FedML for multi-GPU/distributed FL

**Tasks**:
1. Install and configure FedML
2. Run multi-process simulation
3. Compare with Flower simulation
4. Measure communication overhead

**Implementation**: `experiments/fedml_scaling/`
- FedML + PyTorch setup
- Multi-GPU configuration
- Benchmark: clients vs wall-clock time

**Reading**:
- FedML documentation and tutorials
- AWS + FedML blog for production patterns

---

## Week 6: HPO & Scaling Analysis

### 6.1 Hyperparameter Optimization for FL [Day 1-3]
**Objective**: Systematically tune FL hyperparameters

**Key HPO Methods**:
- Grid search (baseline)
- Random search (Bergstra & Bengio)
- Bayesian optimization (Optuna, Ray Tune)
- Multi-fidelity: Successive Halving, Hyperband

**FL Hyperparameters to Tune**:
- Client learning rate
- Number of local epochs E
- Batch size
- Client fraction K per round
- DP parameters: ε, clipping norm C, noise multiplier

**Implementation**: `experiments/hpo/`
- Optuna integration with Flower
- Multi-objective: accuracy + privacy (ε)
- Parallel HPO on SLURM

**Reading**:
- Bergstra & Bengio, "Random Search for HPO"
- "Hyperparameter Optimization in ML" (2024 survey)

---

### 6.2 Multi-GPU Scaling Experiments [Day 4-5]
**Objective**: Measure scaling behavior on HPC

**Experiments**:
1. Strong scaling: Fixed total work, vary workers
2. Weak scaling: Fixed work per worker, vary workers
3. Communication analysis: time breakdown per round
4. Straggler impact: heterogeneous client speeds

**Metrics**:
- Wall-clock time per round
- GPU utilization
- Communication time vs computation time
- Speedup curves

**Implementation**: `experiments/scaling/`
- Multi-GPU Flower simulation
- SLURM job arrays for scaling experiments
- Timing instrumentation

---

### 6.3 Advanced FL Algorithms [Day 6]
**Objective**: Implement 2-3 SOTA FL algorithms

**Algorithms**:
- **SCAFFOLD**: Variance reduction for client drift
- **FedNova**: Normalized averaging for heterogeneous local steps
- **FedOpt (FedAdam/FedYogi)**: Server-side adaptive optimization

**Implementation**: `experiments/advanced_algorithms/`
- SCAFFOLD with control variates
- Compare on IID vs Non-IID

---

### 6.4 Systems Report [Day 7]
**Deliverable**: 2-3 page systems report

**Structure**:
1. Introduction & Motivation
2. Experimental Setup (HPC, GPUs, Flower/FedML)
3. Scaling Results (plots, analysis)
4. Communication Analysis
5. HPO Findings
6. Discussion: Path to Exascale FL

---

## Directory Structure
```
week5-6-scaling/
├── PLAN.md                          # This file
├── notebooks/
│   ├── 01_distributed_optimization.py  # Distributed opt theory
│   ├── 02_admm_tutorial.py             # ADMM walkthrough
│   └── 03_consensus_algorithms.py      # Consensus convergence
├── experiments/
│   ├── fedprox/
│   │   ├── fedprox_client.py           # FedProx client implementation
│   │   ├── fedprox_server.py           # FedProx strategy
│   │   └── run_fedprox.slurm           # SLURM job
│   ├── fedml_scaling/
│   │   ├── setup_fedml.sh              # FedML installation
│   │   ├── fedml_config.yaml           # Configuration
│   │   └── run_fedml.slurm             # Multi-GPU experiment
│   ├── hpo/
│   │   ├── optuna_flower.py            # Optuna + Flower integration
│   │   ├── ray_tune_fl.py              # Ray Tune alternative
│   │   └── run_hpo.slurm               # HPO SLURM job
│   ├── scaling/
│   │   ├── scaling_experiment.py       # Timing instrumentation
│   │   ├── run_scaling.slurm           # SLURM job array
│   │   └── analyze_scaling.py          # Plot scaling curves
│   └── advanced_algorithms/
│       ├── scaffold/                   # SCAFFOLD implementation
│       ├── fednova/                    # FedNova implementation
│       └── fedopt/                     # FedAdam/FedYogi
└── notes/
    └── systems_report.md              # 2-3 page systems report
```

---

## Key Deliverables

### Week 5
- [ ] `01_distributed_optimization.py` - Distributed opt demos
- [ ] `02_admm_tutorial.py` - ADMM walkthrough
- [ ] FedProx implementation + comparison with FedAvg
- [ ] FedML setup and basic experiments

### Week 6
- [ ] Optuna/Ray Tune integration with Flower
- [ ] Multi-GPU scaling experiments + plots
- [ ] 2-3 advanced algorithm implementations (SCAFFOLD, FedNova, FedOpt)
- [ ] `systems_report.md` - 2-3 page systems report

---

## Connection to JD Requirements

| JD Requirement | Implementation |
|---------------|----------------|
| Distributed optimization | ADMM, consensus algorithms, FedProx |
| Consensus algorithms | Decentralized FL, consensus convergence |
| Large models & HPO | Optuna/Ray Tune for FL hyperparameters |
| HPC applications | Multi-GPU scaling, SLURM integration, timing analysis |
