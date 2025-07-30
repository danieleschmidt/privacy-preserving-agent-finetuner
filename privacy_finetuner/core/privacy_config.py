"""Privacy configuration management for differential privacy training."""

from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path
import yaml


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