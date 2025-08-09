"""Privacy configuration management for differential privacy training."""

import os
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
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
    accounting_mode: Literal["rdp", "gdp", "basic"] = "rdp"
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
    
    @classmethod
    def from_env(cls, prefix: str = "PRIVACY_") -> "PrivacyConfig":
        """Load privacy configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            PrivacyConfig instance with values from environment
        """
        config = {}
        
        env_mapping = {
            f"{prefix}EPSILON": ("epsilon", float),
            f"{prefix}DELTA": ("delta", float),
            f"{prefix}MAX_GRAD_NORM": ("max_grad_norm", float),
            f"{prefix}NOISE_MULTIPLIER": ("noise_multiplier", float),
            f"{prefix}ACCOUNTING_MODE": ("accounting_mode", str),
            f"{prefix}FEDERATED_ENABLED": ("federated_enabled", lambda x: x.lower() == 'true'),
            f"{prefix}MIN_CLIENTS": ("min_clients", int),
            f"{prefix}SECURE_PROVIDER": ("secure_compute_provider", str),
            f"{prefix}ATTESTATION_REQUIRED": ("attestation_required", lambda x: x.lower() == 'true'),
        }
        
        for env_var, (field_name, converter) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    config[field_name] = converter(value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid value for {env_var}: {value} ({e})")
        
        return cls(**config)
    
    def validate(self) -> None:
        """Validate privacy configuration parameters with detailed feedback."""
        errors = []
        warnings = []
        
        # Epsilon validation
        if self.epsilon <= 0:
            errors.append(f"Epsilon must be positive, got {self.epsilon}")
        elif self.epsilon > 10:
            warnings.append(f"Large epsilon ({self.epsilon}) provides weak privacy guarantees")
        elif self.epsilon < 0.1:
            warnings.append(f"Very small epsilon ({self.epsilon}) may severely impact model utility")
            
        # Delta validation
        if self.delta <= 0 or self.delta >= 1:
            errors.append(f"Delta must be in (0, 1), got {self.delta}")
        elif self.delta > 1e-3:
            warnings.append(f"Large delta ({self.delta}) may not provide strong privacy")
            
        # Gradient norm validation
        if self.max_grad_norm <= 0:
            errors.append(f"Max gradient norm must be positive, got {self.max_grad_norm}")
        elif self.max_grad_norm > 10:
            warnings.append(f"Very large gradient norm ({self.max_grad_norm}) may reduce noise effectiveness")
            
        # Noise multiplier validation
        if self.noise_multiplier < 0:
            errors.append(f"Noise multiplier must be non-negative, got {self.noise_multiplier}")
        elif self.noise_multiplier < 0.1:
            warnings.append(f"Small noise multiplier ({self.noise_multiplier}) provides weak privacy")
            
        # Accounting mode validation
        if self.accounting_mode not in ["rdp", "gdp"]:
            errors.append(f"Unknown accounting mode: {self.accounting_mode}")
            
        if errors:
            raise ValueError("Privacy configuration validation failed:\n" + "\n".join(errors))
        
        if warnings:
            import warnings as warn_module
            for warning in warnings:
                warn_module.warn(f"Privacy Config Warning: {warning}")
    
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
    
    def save_yaml(self, config_path: Path) -> None:
        """Save privacy configuration to YAML file.
        
        Args:
            config_path: Path to save configuration
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML configuration saving. Install with: pip install PyYAML")
        
        config_data = {"privacy": self.to_dict()}
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def get_recommendations(self) -> Dict[str, str]:
        """Get configuration recommendations based on current settings.
        
        Returns:
            Dictionary of recommendations for improving privacy/utility trade-off
        """
        recommendations = {}
        
        if self.epsilon > 5:
            recommendations['epsilon'] = "Consider reducing epsilon for stronger privacy (try 1.0-3.0)"
        elif self.epsilon < 0.5:
            recommendations['epsilon'] = "Epsilon is very small - model utility may be significantly impacted"
        
        if self.delta > 1e-4:
            recommendations['delta'] = "Consider reducing delta for stronger privacy guarantees (try 1e-5 or smaller)"
        
        if self.noise_multiplier < 0.5:
            recommendations['noise_multiplier'] = "Low noise multiplier - consider increasing for better privacy"
        elif self.noise_multiplier > 2.0:
            recommendations['noise_multiplier'] = "High noise multiplier may significantly impact model convergence"
        
        if not self.federated_enabled and self.min_clients == 5:
            recommendations['federated'] = "Consider enabling federated learning for enhanced privacy"
        
        return recommendations
    
    def privacy_risk_assessment(self) -> Dict[str, Any]:
        """Assess privacy risk level based on configuration.
        
        Returns:
            Dictionary with risk assessment and mitigation suggestions
        """
        risk_score = 0
        factors = []
        
        # Epsilon risk
        if self.epsilon > 10:
            risk_score += 3
            factors.append("Very high epsilon")
        elif self.epsilon > 3:
            risk_score += 2
            factors.append("High epsilon")
        elif self.epsilon > 1:
            risk_score += 1
            factors.append("Moderate epsilon")
        
        # Delta risk
        if self.delta > 1e-3:
            risk_score += 2
            factors.append("Large delta")
        elif self.delta > 1e-4:
            risk_score += 1
            factors.append("Moderate delta")
        
        # Noise adequacy
        if self.noise_multiplier < 0.5:
            risk_score += 2
            factors.append("Low noise multiplier")
        
        # Determine overall risk level
        if risk_score >= 5:
            risk_level = "HIGH"
        elif risk_score >= 3:
            risk_level = "MEDIUM"
        elif risk_score >= 1:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": factors,
            "mitigation_needed": risk_score >= 3
        }