# Distributed Optimization & FL Algorithms - Theory Notes

## Overview

This document connects federated learning algorithms to their distributed optimization foundations, as studied during Weeks 5-6 of the upskilling plan.

---

## 1. FL as Distributed Optimization

### The FL Objective

Federated learning solves:

$$\min_w F(w) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(w)$$

where:
- K = number of clients
- n_k = samples on client k
- n = total samples
- F_k(w) = local objective on client k

This is **distributed empirical risk minimization** with:
- Data partitioned across clients
- Communication constraints (can't share raw data)
- Heterogeneity (non-IID data, varying compute)

---

## 2. FedAvg: Distributed SGD with Local Steps

### Algorithm
```
For each round t:
    1. Server sends w_t to selected clients
    2. Each client k:
       - Initialize w_k = w_t
       - Run E epochs of SGD: w_k ← w_k - η∇F_k(w_k)
    3. Server aggregates: w_{t+1} = Σ (n_k/n) w_k
```

### Connection to Distributed Optimization

FedAvg is **local SGD** (also called **parallel SGD with periodic averaging**):
- Clients do multiple local updates between syncs
- Reduces communication by factor of E (local epochs)

### Convergence Analysis
For smooth, strongly convex objectives:
- Converges at rate O(1/T) with proper learning rate decay
- **Client drift**: Local steps cause models to diverge
- Trade-off: More local steps = less communication but more drift

### Key Papers
- McMahan et al. (2017): *Communication-Efficient Learning of Deep Networks*
- Stich (2019): *Local SGD Converges Fast and Communicates Little*

---

## 3. FedProx: Proximal Regularization

### Motivation
FedAvg struggles with:
- **Statistical heterogeneity**: Non-IID data across clients
- **Systems heterogeneity**: Varying compute capabilities

### Algorithm
Each client solves a **proximal subproblem**:

$$\min_w F_k(w) + \frac{\mu}{2}\|w - w_t\|^2$$

where μ is the proximal parameter.

### Connection to Proximal Gradient Methods
This is **proximal point algorithm** applied to FL:
- Proximal term prevents local models from drifting too far
- μ controls trade-off between local fitting and staying near global

### Implementation
```python
# FedProx local training
for batch in dataloader:
    loss = criterion(model(x), y)
    
    # Proximal term: μ/2 * ||w - w_global||²
    proximal_term = 0
    for w, w_global in zip(model.parameters(), global_params):
        proximal_term += (w - w_global).norm(2) ** 2
    
    total_loss = loss + (mu / 2) * proximal_term
    total_loss.backward()
    optimizer.step()
```

### Key Paper
- Li et al. (2020): *Federated Optimization in Heterogeneous Networks*

---

## 4. SCAFFOLD: Variance Reduction

### Motivation
FedAvg suffers from **client drift** due to:
- Local gradients ∇F_k(w) ≠ global gradient ∇F(w)
- Drift compounds over local steps

### Key Idea: Control Variates
Maintain **control variates** c (global) and c_k (per-client) that estimate gradient differences.

### Algorithm
```
For each round t:
    1. Server sends (w_t, c) to clients
    
    2. Each client k:
       # Corrected local update
       For each step:
           g_k = ∇F_k(w_k)
           w_k ← w_k - η(g_k - c_k + c)  # Gradient correction
       
       # Update control variate (Option I)
       c_k^+ = c_k - c + (w_t - w_k) / (Kη)
       Δc_k = c_k^+ - c_k
       Δw_k = w_k - w_t
    
    3. Server aggregates:
       w_{t+1} = w_t + Σ (n_k/n) Δw_k
       c = c + (|S|/K) * avg(Δc_k)
```

### Why It Works
- c ≈ ∇F(w): Global control variate estimates global gradient
- c_k ≈ ∇F_k(w): Client control estimates local gradient
- Correction (g_k - c_k + c) removes local bias

### Connection to Variance Reduction
SCAFFOLD is related to:
- **SVRG**: Stochastic variance-reduced gradient
- **SAGA**: Uses stored gradients to reduce variance

The control variates play the role of "stored gradients" in variance reduction.

### Key Paper
- Karimireddy et al. (2020): *SCAFFOLD: Stochastic Controlled Averaging for FL*

---

## 5. Consensus Algorithms

### Decentralized FL
Instead of star topology (server ↔ clients), use peer-to-peer:

```
Each node i:
    1. Local SGD step
    2. Average with neighbors: w_i ← Σ_j W_ij w_j
```

where W is a mixing matrix (doubly stochastic).

### Gossip Averaging
- Random pairwise averaging
- Converges to global average over time

### Connection to FL
- FedAvg = single-round consensus (all-to-one-to-all)
- Decentralized FL = multi-round gossip consensus
- Trade-off: Communication topology vs convergence speed

### Key Papers
- Nedić & Ozdaglar (2009): *Distributed Subgradient Methods*
- Lian et al. (2017): *Can Decentralized Algorithms Outperform Centralized Algorithms?*

---

## 6. ADMM-Based FL

### Alternating Direction Method of Multipliers

ADMM solves constrained problems by splitting:

$$\min_w f(w) \text{ s.t. } Aw = b$$

Update rules:
```
w^{k+1} = argmin_w f(w) + (ρ/2)||Aw - b + u^k||²
u^{k+1} = u^k + Aw^{k+1} - b
```

### ADMM for FL (FedADMM)
Reformulate FL as:
$$\min \sum_k F_k(w_k) \text{ s.t. } w_k = z \text{ (consensus)}$$

Each client optimizes locally while dual variables enforce consensus.

### Advantages
- Natural handling of constraints
- Can incorporate regularization
- Robust to partial participation

### Key Paper
- Boyd et al. (2011): *Distributed Optimization via ADMM*

---

## 7. Algorithm Comparison

| Algorithm | Communication | Heterogeneity | Convergence | Complexity |
|-----------|--------------|---------------|-------------|------------|
| FedAvg | O(1/E) reduced | Sensitive | O(1/T) | Low |
| FedProx | O(1/E) reduced | Robust | O(1/T) | Low |
| SCAFFOLD | O(1/E) reduced | Very robust | O(1/T) faster | Medium |
| FedADMM | Higher | Robust | O(1/T) | Higher |
| Decentralized | Peer-to-peer | Varies | O(1/T) | Network-dependent |

---

## 8. Hyperparameter Tuning in FL

### Key Hyperparameters

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| **Local epochs (E)** | More = less communication, more drift | 1-10 |
| **Learning rate (η)** | Step size for SGD | 0.001-0.1 |
| **Batch size (B)** | Local mini-batch size | 32-128 |
| **Client fraction (C)** | Fraction sampled per round | 0.1-1.0 |
| **Proximal μ (FedProx)** | Drift penalty | 0.001-1.0 |

### Tuning Strategies
1. **Grid/Random Search**: Standard but expensive
2. **Bayesian Optimization**: Efficient for expensive evaluations
3. **Adaptive methods**: FedOpt (FedAdam, FedYogi) adapt server LR

### FL-Specific Considerations
- Can't do per-client HP tuning (privacy)
- Must account for heterogeneity in tuning
- Communication budget constraints

---

## 9. Implementation in fl_research

```python
from fl_research.strategies import (
    FedAvgStrategy,           # Basic federated averaging
    FedProxStrategy,          # With proximal term
    SCAFFOLDServer,           # Server with control variates
    StandaloneSCAFFOLDClient, # Client with gradient correction
)

# FedProx example
strategy = FedProxStrategy(
    proximal_mu=0.1,          # Proximal parameter
    fraction_fit=0.1,         # 10% clients per round
    min_fit_clients=2,
)

# SCAFFOLD example
server = SCAFFOLDServer(model, device)
clients = [StandaloneSCAFFOLDClient(i, loader, device) for i, loader in enumerate(loaders)]

for round in range(num_rounds):
    deltas = [client.train(model, server.global_control, epochs=5, lr=0.01) 
              for client in clients]
    server.aggregate(*zip(*deltas), total_clients=len(clients))
```

---

## Key References

1. **McMahan et al. (2017)**: *Communication-Efficient Learning* (FedAvg)
2. **Li et al. (2020)**: *Federated Optimization in Heterogeneous Networks* (FedProx)
3. **Karimireddy et al. (2020)**: *SCAFFOLD: Stochastic Controlled Averaging*
4. **Boyd et al. (2011)**: *Distributed Optimization via ADMM*
5. **Bertsekas & Tsitsiklis (1989)**: *Parallel and Distributed Computation*
6. **Nedić & Ozdaglar (2009)**: *Distributed Subgradient Methods*
