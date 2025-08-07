"""Privacy configuration management for differential privacy training."""

from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class PrivacyConfig:
    """Configuration for differential privacy parameters.
    
    This class manages privacy budgets and parameters for DP-SGD training,
    ensuring formal privacy guarantees while maintaining model utility.
    """
    
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 0.5
    accounting_mode: Literal["rdp", "gdp"] = "rdp"
    target_delta: Optional[float] = None
    
    # Federated learning settings
    federated_enabled: bool = False
    aggregation_method: str = "secure_sum"
    min_clients: int = 5
    
    # Hardware security settings
    secure_compute_provider: Optional[str] = None
    attestation_required: bool = False
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "PrivacyConfig":
        """Load privacy configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML configuration loading. Install with: pip install PyYAML")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        privacy_config = config_data.get('privacy', {})
        return cls(**privacy_config)
    
    def validate(self) -> None:
        """Validate privacy parameters for mathematical correctness."""
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError("Delta must be between 0 and 1")
        if self.max_grad_norm <= 0:
            raise ValueError("Max gradient norm must be positive")
        if self.noise_multiplier < 0:
            raise ValueError("Noise multiplier must be non-negative")
        if self.accounting_mode not in ["rdp", "gdp"]:
            raise ValueError("Accounting mode must be 'rdp' or 'gdp'")
    
    def get_effective_noise_scale(self, sample_rate: float) -> float:
        """Calculate effective noise scale for given sample rate."""
        return self.noise_multiplier * self.max_grad_norm * sample_rate
    
    def estimate_privacy_cost(self, steps: int, sample_rate: float) -> float:
        """Estimate privacy cost for given training parameters."""
        import math
        
        if self.accounting_mode == "rdp":
            # Simplified RDP bound for Gaussian mechanism
            sigma = self.noise_multiplier
            q = sample_rate
            alpha = 10  # Common RDP order
            
            rdp = (alpha * (alpha - 1) * q * q) / (2 * sigma * sigma)
            epsilon = rdp * steps + math.log(1 / self.delta) / (alpha - 1)
            return epsilon
        else:
            # Basic composition bound
            return steps * sample_rate / self.noise_multiplier
    
    def adaptive_noise_scaling(self, gradient_norm: float, target_norm: float) -> float:
        """Calculate adaptive noise scaling based on gradient norms.
        
        Args:
            gradient_norm: Current gradient L2 norm
            target_norm: Target clipping norm
            
        Returns:
            Adaptive noise multiplier
        """
        if gradient_norm <= target_norm:
            # No clipping needed, reduce noise
            return self.noise_multiplier * 0.8
        else:
            # Clipping applied, use full noise
            return self.noise_multiplier
    
    def privacy_amplification_factor(self, subsampling_rate: float) -> float:
        """Calculate privacy amplification from subsampling.
        
        Args:
            subsampling_rate: Fraction of data used per batch
            
        Returns:
            Privacy amplification factor
        """
        import math
        return math.sqrt(subsampling_rate)
    
    def remaining_budget(self, steps: int, sample_rate: float) -> float:
        """Calculate remaining privacy budget.
        
        Args:
            steps: Training steps completed
            sample_rate: Subsampling rate
            
        Returns:
            Remaining epsilon budget
        """
        spent = self.estimate_privacy_cost(steps, sample_rate)
        return max(0, self.epsilon - spent)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "max_grad_norm": self.max_grad_norm,
            "noise_multiplier": self.noise_multiplier,
            "accounting_mode": self.accounting_mode,
            "target_delta": self.target_delta,
            "federated_enabled": self.federated_enabled,
            "aggregation_method": self.aggregation_method,
            "min_clients": self.min_clients,
            "secure_compute_provider": self.secure_compute_provider,
            "attestation_required": self.attestation_required
        }