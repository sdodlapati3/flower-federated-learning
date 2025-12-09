"""
Privacy Module
==============

Differential Privacy utilities for Federated Learning.

This module provides:
- Privacy accounting (RDP, GDP)
- Noise mechanisms
- Opacus integration helpers
- Privacy budget management
"""

from fl_research.privacy.accountant import (
    PrivacyAccountant,
    compute_rdp,
    compute_epsilon,
    get_privacy_spent,
)
from fl_research.privacy.mechanisms import (
    GaussianMechanism,
    LaplaceMechanism,
    add_gaussian_noise,
    add_laplace_noise,
    clip_gradients,
)
from fl_research.privacy.opacus_utils import (
    make_private,
    validate_model_for_dp,
    convert_batchnorm_to_groupnorm,
    get_privacy_engine_state,
)

__all__ = [
    # Accountant
    "PrivacyAccountant",
    "compute_rdp",
    "compute_epsilon",
    "get_privacy_spent",
    # Mechanisms
    "GaussianMechanism",
    "LaplaceMechanism",
    "add_gaussian_noise",
    "add_laplace_noise",
    "clip_gradients",
    # Opacus utils
    "make_private",
    "validate_model_for_dp",
    "convert_batchnorm_to_groupnorm",
    "get_privacy_engine_state",
]
