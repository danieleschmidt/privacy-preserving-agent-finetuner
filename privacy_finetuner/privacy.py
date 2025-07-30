"""
Privacy configuration and differential privacy mechanisms.

This module defines privacy parameters and implements core DP algorithms
for privacy-preserving machine learning.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AccountingMode(Enum):
    """Privacy accounting methods."""
    RDP = "rdp"  # Rényi Differential Privacy
    GDP = "gdp"  # Gaussian Differential Privacy


@dataclass
class PrivacyConfig:
    """
    Configuration for differential privacy parameters.
    
    Args:
        epsilon: Privacy loss parameter (smaller = more private)
        delta: Failure probability (typically 1e-5 to 1e-8)
        max_grad_norm: Maximum L2 norm for gradient clipping
        noise_multiplier: Gaussian noise multiplier for DP-SGD
        accounting_mode: Method for privacy accounting
        target_delta: Target delta for privacy analysis
    """
    
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 0.5
    accounting_mode: AccountingMode = AccountingMode.RDP
    target_delta: Optional[float] = None
    
    def __post_init__(self):
        """Validate privacy parameters."""
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError("Delta must be in (0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError("Max gradient norm must be positive")
        if self.noise_multiplier <= 0:
            raise ValueError("Noise multiplier must be positive")
    
    @property
    def privacy_budget_remaining(self) -> float:
        """Calculate remaining privacy budget."""
        # Simplified calculation - real implementation would track consumption
        return max(0.0, self.epsilon - 0.1)  # Placeholder
    
    def is_privacy_budget_exhausted(self, threshold: float = 0.1) -> bool:
        """Check if privacy budget is near exhaustion."""
        return self.privacy_budget_remaining < threshold