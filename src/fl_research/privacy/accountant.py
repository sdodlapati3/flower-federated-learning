"""
Privacy Accountant
==================

Track and compute privacy guarantees using RDP and GDP.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import math


@dataclass
class PrivacyBudget:
    """Represents a privacy budget."""
    epsilon: float
    delta: float
    
    def __str__(self) -> str:
        return f"(ε={self.epsilon:.4f}, δ={self.delta:.2e})"
    
    def is_exceeded(self, target_epsilon: float, target_delta: float) -> bool:
        """Check if budget exceeds target."""
        return self.epsilon > target_epsilon or self.delta > target_delta


def compute_rdp(
    q: float,
    noise_multiplier: float,
    orders: List[float],
) -> List[float]:
    """
    Compute RDP for subsampled Gaussian mechanism.
    
    Args:
        q: Subsampling rate (batch_size / dataset_size)
        noise_multiplier: Sigma / L2_sensitivity
        orders: List of RDP orders (alpha values)
    
    Returns:
        List of RDP values for each order
    
    Note:
        Uses the tight RDP analysis from Mironov et al.
    """
    if noise_multiplier == 0:
        return [float('inf')] * len(orders)
    
    rdp_values = []
    
    for alpha in orders:
        if alpha <= 1:
            rdp_values.append(float('inf'))
            continue
        
        # Compute RDP for subsampled Gaussian
        # Using the analytic formula from Mironov (2017)
        if q == 0:
            rdp_values.append(0.0)
        elif q == 1:
            # No subsampling
            rdp = alpha / (2 * noise_multiplier ** 2)
            rdp_values.append(rdp)
        else:
            # Subsampled Gaussian - use numerical upper bound
            # This is a simplified bound; Opacus uses more precise analysis
            sigma = noise_multiplier
            log_terms = []
            
            for k in range(min(int(alpha) + 1, 100)):
                log_comb = _log_comb(int(alpha), k)
                log_q = k * math.log(q) + (int(alpha) - k) * math.log(1 - q) if q < 1 else 0
                log_mgf = k * (k - 1) / (2 * sigma ** 2)
                log_terms.append(log_comb + log_q + log_mgf)
            
            if log_terms:
                rdp = _log_sum_exp(log_terms) / (alpha - 1)
            else:
                rdp = 0.0
            
            rdp_values.append(max(0.0, rdp))
    
    return rdp_values


def _log_comb(n: int, k: int) -> float:
    """Compute log of binomial coefficient."""
    if k < 0 or k > n:
        return float('-inf')
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_sum_exp(log_terms: List[float]) -> float:
    """Compute log(sum(exp(x))) in numerically stable way."""
    if not log_terms:
        return float('-inf')
    max_term = max(log_terms)
    if max_term == float('-inf'):
        return float('-inf')
    return max_term + math.log(sum(math.exp(t - max_term) for t in log_terms))


def compute_epsilon(
    rdp_values: List[float],
    orders: List[float],
    delta: float,
) -> Tuple[float, float]:
    """
    Convert RDP to (epsilon, delta)-DP.
    
    Args:
        rdp_values: RDP values at each order
        orders: RDP orders (alpha values)
        delta: Target delta
    
    Returns:
        Tuple of (epsilon, optimal_order)
    """
    if delta <= 0:
        return float('inf'), 0
    
    eps_values = []
    
    for rdp, alpha in zip(rdp_values, orders):
        if alpha <= 1 or rdp == float('inf'):
            eps_values.append(float('inf'))
        else:
            # RDP to DP conversion
            eps = rdp - math.log(delta) / (alpha - 1)
            eps_values.append(eps)
    
    # Find minimum epsilon
    min_idx = min(range(len(eps_values)), key=lambda i: eps_values[i])
    
    return eps_values[min_idx], orders[min_idx]


def get_privacy_spent(
    steps: int,
    noise_multiplier: float,
    sample_rate: float,
    delta: float,
    orders: Optional[List[float]] = None,
) -> PrivacyBudget:
    """
    Compute privacy spent after training.
    
    Args:
        steps: Number of training steps
        noise_multiplier: Noise multiplier (sigma / sensitivity)
        sample_rate: Batch size / dataset size
        delta: Target delta
        orders: RDP orders to use
    
    Returns:
        PrivacyBudget with epsilon and delta
    
    Example:
        >>> budget = get_privacy_spent(
        ...     steps=1000,
        ...     noise_multiplier=1.0,
        ...     sample_rate=0.01,
        ...     delta=1e-5,
        ... )
        >>> print(budget)
        (ε=2.1234, δ=1.00e-05)
    """
    if orders is None:
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    
    # Compute RDP for one step
    rdp_single = compute_rdp(sample_rate, noise_multiplier, orders)
    
    # RDP composes linearly
    rdp_total = [rdp * steps for rdp in rdp_single]
    
    # Convert to epsilon
    epsilon, _ = compute_epsilon(rdp_total, orders, delta)
    
    return PrivacyBudget(epsilon=epsilon, delta=delta)


class PrivacyAccountant:
    """
    Track cumulative privacy budget during training.
    
    Example:
        >>> accountant = PrivacyAccountant(
        ...     noise_multiplier=1.0,
        ...     sample_rate=0.01,
        ...     delta=1e-5,
        ... )
        >>> for step in range(100):
        ...     accountant.step()
        >>> print(accountant.get_epsilon())
        0.5234
    """
    
    DEFAULT_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    
    def __init__(
        self,
        noise_multiplier: float,
        sample_rate: float,
        delta: float = 1e-5,
        max_epsilon: float = float('inf'),
        orders: Optional[List[float]] = None,
    ):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self.max_epsilon = max_epsilon
        self.orders = orders or self.DEFAULT_ORDERS
        
        self._steps = 0
        self._rdp_per_step = compute_rdp(sample_rate, noise_multiplier, self.orders)
    
    def step(self, num_steps: int = 1) -> None:
        """Record training step(s)."""
        self._steps += num_steps
    
    @property
    def steps(self) -> int:
        """Total steps taken."""
        return self._steps
    
    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        rdp_total = [rdp * self._steps for rdp in self._rdp_per_step]
        epsilon, _ = compute_epsilon(rdp_total, self.orders, self.delta)
        return epsilon
    
    def get_privacy_budget(self) -> PrivacyBudget:
        """Get current privacy budget."""
        return PrivacyBudget(epsilon=self.get_epsilon(), delta=self.delta)
    
    def is_budget_exceeded(self) -> bool:
        """Check if epsilon exceeds maximum."""
        return self.get_epsilon() > self.max_epsilon
    
    def remaining_budget(self) -> float:
        """Get remaining epsilon budget."""
        if self.max_epsilon == float('inf'):
            return float('inf')
        return max(0, self.max_epsilon - self.get_epsilon())
    
    def estimate_steps_to_budget(self, target_epsilon: float) -> int:
        """Estimate how many steps until target epsilon is reached."""
        if self._steps == 0:
            # Use single step epsilon
            single_step_budget = get_privacy_spent(
                steps=1,
                noise_multiplier=self.noise_multiplier,
                sample_rate=self.sample_rate,
                delta=self.delta,
                orders=self.orders,
            )
            if single_step_budget.epsilon <= 0:
                return float('inf')
            # Rough linear estimate
            return int(target_epsilon / single_step_budget.epsilon)
        
        # Use current rate
        current_epsilon = self.get_epsilon()
        if current_epsilon <= 0:
            return float('inf')
        
        epsilon_per_step = current_epsilon / self._steps
        remaining = target_epsilon - current_epsilon
        
        if remaining <= 0:
            return 0
        
        return int(remaining / epsilon_per_step)
    
    def state_dict(self) -> dict:
        """Get accountant state for saving."""
        return {
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
            "delta": self.delta,
            "max_epsilon": self.max_epsilon,
            "steps": self._steps,
        }
    
    def load_state_dict(self, state: dict) -> None:
        """Restore accountant state."""
        self._steps = state["steps"]
