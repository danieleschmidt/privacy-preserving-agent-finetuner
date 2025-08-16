"""
Advanced Privacy-Preserving Mechanisms

This module implements cutting-edge privacy-preserving techniques beyond standard
differential privacy, including novel algorithms for enhanced privacy-utility tradeoffs.
"""

import asyncio
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expit
from scipy.stats import laplace, norm
import secrets

logger = logging.getLogger(__name__)


class PrivacyMechanism(Enum):
    """Advanced privacy-preserving mechanisms."""
    ADAPTIVE_DIFFERENTIAL_PRIVACY = "adaptive_dp"
    PERSONALIZED_DIFFERENTIAL_PRIVACY = "personalized_dp"
    CONCENTRATED_DIFFERENTIAL_PRIVACY = "concentrated_dp"
    RELAXED_DIFFERENTIAL_PRIVACY = "relaxed_dp"
    FUNCTIONAL_DIFFERENTIAL_PRIVACY = "functional_dp"
    METRIC_DIFFERENTIAL_PRIVACY = "metric_dp"
    LOCAL_DIFFERENTIAL_PRIVACY = "local_dp"
    SHUFFLE_DIFFERENTIAL_PRIVACY = "shuffle_dp"
    AMPLIFIED_PRIVACY = "amplified_privacy"
    PRIVACY_ODOMETER = "privacy_odometer"


class NoiseDistribution(Enum):
    """Noise distributions for privacy mechanisms."""
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    EXPONENTIAL = "exponential"
    STUDENT_T = "student_t"
    SKEWED_LAPLACE = "skewed_laplace"
    TRUNCATED_LAPLACE = "truncated_laplace"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"


@dataclass
class PrivacyParameters:
    """Advanced privacy parameters."""
    epsilon: float
    delta: float
    sensitivity: float
    concentration: Optional[float] = None
    personalization_factor: Optional[float] = None
    functional_sensitivity: Optional[float] = None
    metric_distance: Optional[float] = None
    amplification_factor: Optional[float] = None
    privacy_odometer_budget: Optional[float] = None


@dataclass
class PrivacyBudgetState:
    """Dynamic privacy budget state."""
    total_budget: float
    consumed_budget: float
    remaining_budget: float
    allocation_history: List[Dict[str, Any]] = field(default_factory=list)
    prediction_accuracy: float = 0.0
    optimization_factor: float = 1.0
    last_updated: float = field(default_factory=time.time)


class AdvancedNoiseGenerator:
    """Advanced noise generation for privacy mechanisms."""
    
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
    def generate_adaptive_gaussian_noise(self, 
                                       shape: Tuple[int, ...],
                                       sigma: float,
                                       adaptive_factor: float = 1.0) -> torch.Tensor:
        """Generate adaptive Gaussian noise with data-dependent variance."""
        base_noise = torch.randn(shape) * sigma
        
        # Adaptive component based on gradient norms or data characteristics
        adaptive_component = torch.randn(shape) * sigma * adaptive_factor * 0.1
        
        return base_noise + adaptive_component
    
    def generate_skewed_laplace_noise(self, 
                                    shape: Tuple[int, ...],
                                    scale: float,
                                    skewness: float = 0.0) -> torch.Tensor:
        """Generate skewed Laplace noise for asymmetric privacy."""
        # Standard Laplace noise
        u = torch.rand(shape) - 0.5
        laplace_noise = -scale * torch.sign(u) * torch.log(1 - 2 * torch.abs(u))
        
        # Apply skewness
        if skewness != 0:
            skew_factor = torch.where(laplace_noise > 0, 
                                    1 + skewness, 
                                    1 - skewness)
            laplace_noise = laplace_noise * skew_factor
        
        return laplace_noise
    
    def generate_truncated_laplace_noise(self,
                                       shape: Tuple[int, ...],
                                       scale: float,
                                       bounds: Tuple[float, float]) -> torch.Tensor:
        """Generate truncated Laplace noise with bounded support."""
        lower, upper = bounds
        
        # Generate standard Laplace noise
        noise = self.generate_skewed_laplace_noise(shape, scale)
        
        # Truncate to bounds
        truncated_noise = torch.clamp(noise, lower, upper)
        
        return truncated_noise
    
    def generate_student_t_noise(self,
                               shape: Tuple[int, ...],
                               df: float,
                               scale: float) -> torch.Tensor:
        """Generate Student's t-distributed noise for heavy-tailed privacy."""
        # Generate using inverse transform sampling
        u = torch.rand(shape)
        
        # Approximate Student's t using normal approximation for large df
        if df > 30:
            return torch.randn(shape) * scale * math.sqrt(df / (df - 2))
        
        # For smaller df, use more complex generation
        chi2_sample = torch.distributions.chi2.Chi2(df).sample(shape)
        normal_sample = torch.randn(shape)
        
        t_sample = normal_sample / torch.sqrt(chi2_sample / df)
        
        return t_sample * scale


class AdaptiveDifferentialPrivacy:
    """Adaptive differential privacy with dynamic privacy budget allocation."""
    
    def __init__(self, 
                 total_budget: float,
                 adaptation_rate: float = 0.1,
                 prediction_window: int = 10):
        self.total_budget = total_budget
        self.adaptation_rate = adaptation_rate
        self.prediction_window = prediction_window
        self.budget_state = PrivacyBudgetState(
            total_budget=total_budget,
            consumed_budget=0.0,
            remaining_budget=total_budget
        )
        self.noise_generator = AdvancedNoiseGenerator()
        self.utility_history: List[float] = []
        
    def allocate_budget_adaptively(self, 
                                 current_utility: float,
                                 predicted_future_utility: List[float]) -> float:
        """Adaptively allocate privacy budget based on utility predictions."""
        
        # Update utility history
        self.utility_history.append(current_utility)
        if len(self.utility_history) > self.prediction_window:
            self.utility_history.pop(0)
        
        # Calculate adaptive allocation factor
        utility_trend = self._calculate_utility_trend()
        future_utility_sum = sum(predicted_future_utility)
        
        # Allocate more budget if utility is increasing or future utility is high
        allocation_factor = 1.0
        if utility_trend > 0:
            allocation_factor += self.adaptation_rate * utility_trend
        
        if future_utility_sum > 0:
            allocation_factor += self.adaptation_rate * (future_utility_sum / len(predicted_future_utility))
        
        # Calculate budget allocation
        base_allocation = self.budget_state.remaining_budget / 100  # Conservative base
        adaptive_allocation = base_allocation * allocation_factor
        
        # Ensure we don't exceed remaining budget
        final_allocation = min(adaptive_allocation, self.budget_state.remaining_budget * 0.1)
        
        # Update budget state
        self.budget_state.consumed_budget += final_allocation
        self.budget_state.remaining_budget -= final_allocation
        
        self.budget_state.allocation_history.append({
            "timestamp": time.time(),
            "allocation": final_allocation,
            "utility": current_utility,
            "utility_trend": utility_trend,
            "adaptation_factor": allocation_factor
        })
        
        return final_allocation
    
    def _calculate_utility_trend(self) -> float:
        """Calculate utility trend over recent history."""
        if len(self.utility_history) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(self.utility_history)
        x = np.arange(n)
        y = np.array(self.utility_history)
        
        # Calculate slope of linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
    
    def add_adaptive_noise(self, 
                          gradients: torch.Tensor,
                          epsilon: float,
                          sensitivity: float) -> torch.Tensor:
        """Add adaptive noise based on gradient characteristics."""
        
        # Calculate gradient-dependent noise scale
        grad_norm = torch.norm(gradients).item()
        adaptive_scale = self._calculate_adaptive_scale(grad_norm, sensitivity)
        
        # Generate noise with adaptive distribution
        noise_scale = sensitivity / epsilon * adaptive_scale
        
        # Use different noise distributions based on gradient characteristics
        if grad_norm > sensitivity * 2:
            # Use truncated noise for large gradients
            noise = self.noise_generator.generate_truncated_laplace_noise(
                gradients.shape, noise_scale, (-sensitivity, sensitivity)
            )
        else:
            # Use adaptive Gaussian for normal gradients
            noise = self.noise_generator.generate_adaptive_gaussian_noise(
                gradients.shape, noise_scale, adaptive_scale
            )
        
        return gradients + noise
    
    def _calculate_adaptive_scale(self, grad_norm: float, sensitivity: float) -> float:
        """Calculate adaptive noise scale based on gradient characteristics."""
        # Scale noise inversely with gradient norm (more noise for smaller gradients)
        normalized_norm = grad_norm / (sensitivity + 1e-8)
        
        # Sigmoid-based adaptive scaling
        adaptive_factor = 2.0 / (1.0 + math.exp(normalized_norm - 1.0))
        
        return adaptive_factor


class PersonalizedDifferentialPrivacy:
    """Personalized differential privacy with individual privacy preferences."""
    
    def __init__(self):
        self.user_privacy_preferences: Dict[str, PrivacyParameters] = {}
        self.noise_generator = AdvancedNoiseGenerator()
        
    def set_user_privacy_preference(self, 
                                  user_id: str,
                                  epsilon: float,
                                  delta: float,
                                  personalization_factor: float = 1.0):
        """Set privacy preferences for individual user."""
        self.user_privacy_preferences[user_id] = PrivacyParameters(
            epsilon=epsilon,
            delta=delta,
            sensitivity=1.0,  # Default sensitivity
            personalization_factor=personalization_factor
        )
    
    def add_personalized_noise(self,
                             user_data: Dict[str, torch.Tensor],
                             global_sensitivity: float) -> Dict[str, torch.Tensor]:
        """Add personalized noise based on individual privacy preferences."""
        
        noisy_data = {}
        
        for user_id, data in user_data.items():
            if user_id in self.user_privacy_preferences:
                params = self.user_privacy_preferences[user_id]
                
                # Calculate personalized noise scale
                noise_scale = self._calculate_personalized_noise_scale(
                    params, global_sensitivity
                )
                
                # Generate personalized noise
                noise = self.noise_generator.generate_adaptive_gaussian_noise(
                    data.shape, noise_scale, params.personalization_factor
                )
                
                noisy_data[user_id] = data + noise
            else:
                # Use default privacy for users without preferences
                default_noise_scale = global_sensitivity / 1.0  # Default epsilon = 1.0
                noise = torch.randn_like(data) * default_noise_scale
                noisy_data[user_id] = data + noise
        
        return noisy_data
    
    def _calculate_personalized_noise_scale(self,
                                          params: PrivacyParameters,
                                          global_sensitivity: float) -> float:
        """Calculate personalized noise scale."""
        base_scale = global_sensitivity / params.epsilon
        
        # Apply personalization factor
        personalized_scale = base_scale / (params.personalization_factor or 1.0)
        
        return personalized_scale


class ConcentratedDifferentialPrivacy:
    """Concentrated differential privacy for tighter composition bounds."""
    
    def __init__(self, rho: float, delta: float):
        """
        Initialize CDP with concentration parameter rho and failure probability delta.
        
        Args:
            rho: Concentration parameter (roughly rho = epsilon^2 / 2)
            delta: Failure probability
        """
        self.rho = rho
        self.delta = delta
        self.noise_generator = AdvancedNoiseGenerator()
        
    def gaussian_mechanism(self, 
                          query_result: torch.Tensor,
                          sensitivity: float) -> torch.Tensor:
        """Apply Gaussian mechanism for CDP."""
        # Calculate noise scale for CDP
        sigma = math.sqrt(sensitivity**2 / (2 * self.rho))
        
        # Generate Gaussian noise
        noise = torch.randn_like(query_result) * sigma
        
        return query_result + noise
    
    def compose_privacy_parameters(self, 
                                 operations: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Compose privacy parameters under CDP."""
        # operations: List of (rho_i, delta_i) tuples
        
        total_rho = sum(rho for rho, _ in operations)
        total_delta = sum(delta for _, delta in operations)
        
        # CDP composition is simply additive for rho
        return total_rho, total_delta
    
    def convert_to_approximate_dp(self) -> Tuple[float, float]:
        """Convert CDP parameters to approximate DP."""
        # Conversion from CDP to (epsilon, delta)-DP
        epsilon = self.rho + 2 * math.sqrt(self.rho * math.log(1 / self.delta))
        
        return epsilon, self.delta


class FunctionalDifferentialPrivacy:
    """Functional differential privacy for specific function classes."""
    
    def __init__(self, function_class: str = "linear"):
        """
        Initialize functional DP for specific function classes.
        
        Args:
            function_class: Type of functions to preserve privacy for
        """
        self.function_class = function_class
        self.noise_generator = AdvancedNoiseGenerator()
        
    def calculate_functional_sensitivity(self,
                                       function: Callable,
                                       data_domain: Tuple[float, float],
                                       num_samples: int = 1000) -> float:
        """Calculate functional sensitivity for given function."""
        
        if self.function_class == "linear":
            return self._linear_functional_sensitivity(function, data_domain)
        elif self.function_class == "polynomial":
            return self._polynomial_functional_sensitivity(function, data_domain, num_samples)
        else:
            return self._general_functional_sensitivity(function, data_domain, num_samples)
    
    def _linear_functional_sensitivity(self,
                                     function: Callable,
                                     data_domain: Tuple[float, float]) -> float:
        """Calculate sensitivity for linear functions."""
        # For linear functions f(x) = ax + b, sensitivity is |a|
        lower, upper = data_domain
        
        # Sample function at domain boundaries
        f_lower = function(torch.tensor([lower]))
        f_upper = function(torch.tensor([upper]))
        
        # Calculate slope
        slope = abs((f_upper - f_lower).item() / (upper - lower))
        
        return slope
    
    def _polynomial_functional_sensitivity(self,
                                         function: Callable,
                                         data_domain: Tuple[float, float],
                                         num_samples: int) -> float:
        """Calculate sensitivity for polynomial functions."""
        lower, upper = data_domain
        
        # Sample function at multiple points
        x_samples = torch.linspace(lower, upper, num_samples)
        f_samples = function(x_samples)
        
        # Calculate maximum derivative (sensitivity)
        derivatives = torch.diff(f_samples) / (x_samples[1] - x_samples[0])
        max_derivative = torch.max(torch.abs(derivatives)).item()
        
        return max_derivative
    
    def _general_functional_sensitivity(self,
                                      function: Callable,
                                      data_domain: Tuple[float, float],
                                      num_samples: int) -> float:
        """Calculate sensitivity for general functions using sampling."""
        lower, upper = data_domain
        step_size = (upper - lower) / num_samples
        
        max_sensitivity = 0.0
        
        for i in range(num_samples):
            x1 = lower + i * step_size
            x2 = min(x1 + step_size, upper)
            
            # Calculate function difference
            f1 = function(torch.tensor([x1]))
            f2 = function(torch.tensor([x2]))
            
            sensitivity = abs((f2 - f1).item() / (x2 - x1))
            max_sensitivity = max(max_sensitivity, sensitivity)
        
        return max_sensitivity
    
    def add_functional_noise(self,
                           query_result: torch.Tensor,
                           function: Callable,
                           epsilon: float,
                           data_domain: Tuple[float, float]) -> torch.Tensor:
        """Add noise calibrated to functional sensitivity."""
        
        functional_sensitivity = self.calculate_functional_sensitivity(
            function, data_domain
        )
        
        # Generate noise scaled to functional sensitivity
        noise_scale = functional_sensitivity / epsilon
        noise = torch.randn_like(query_result) * noise_scale
        
        return query_result + noise


class PrivacyOdometer:
    """Privacy odometer for tracking and optimizing privacy budget consumption."""
    
    def __init__(self, total_budget: float):
        self.total_budget = total_budget
        self.consumption_history: List[Dict[str, Any]] = []
        self.prediction_model = None
        self.optimization_strategies: List[str] = []
        
    def record_privacy_consumption(self,
                                 operation: str,
                                 epsilon_used: float,
                                 delta_used: float,
                                 utility_gained: float,
                                 context: Dict[str, Any]):
        """Record privacy budget consumption."""
        
        consumption_record = {
            "timestamp": time.time(),
            "operation": operation,
            "epsilon_used": epsilon_used,
            "delta_used": delta_used,
            "utility_gained": utility_gained,
            "efficiency_ratio": utility_gained / (epsilon_used + 1e-8),
            "context": context,
            "remaining_budget": self.get_remaining_budget()
        }
        
        self.consumption_history.append(consumption_record)
        
        # Trigger optimization if needed
        if len(self.consumption_history) % 10 == 0:
            self._optimize_future_allocations()
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        consumed = sum(record["epsilon_used"] for record in self.consumption_history)
        return max(0.0, self.total_budget - consumed)
    
    def predict_budget_depletion(self, 
                               future_operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict when privacy budget will be depleted."""
        
        current_consumption_rate = self._calculate_consumption_rate()
        remaining_budget = self.get_remaining_budget()
        
        # Simple prediction based on current rate
        if current_consumption_rate > 0:
            time_to_depletion = remaining_budget / current_consumption_rate
        else:
            time_to_depletion = float('inf')
        
        # Predict impact of future operations
        future_consumption = sum(op.get("expected_epsilon", 0.0) for op in future_operations)
        
        return {
            "current_consumption_rate": current_consumption_rate,
            "time_to_depletion_hours": time_to_depletion / 3600,
            "future_consumption": future_consumption,
            "budget_sufficient": remaining_budget >= future_consumption,
            "optimization_recommendations": self._generate_optimization_recommendations()
        }
    
    def _calculate_consumption_rate(self) -> float:
        """Calculate current privacy budget consumption rate."""
        if len(self.consumption_history) < 2:
            return 0.0
        
        recent_records = self.consumption_history[-10:]  # Last 10 operations
        
        total_consumption = sum(record["epsilon_used"] for record in recent_records)
        time_span = recent_records[-1]["timestamp"] - recent_records[0]["timestamp"]
        
        if time_span > 0:
            return total_consumption / time_span
        
        return 0.0
    
    def _optimize_future_allocations(self):
        """Optimize future privacy budget allocations."""
        
        # Analyze efficiency patterns
        efficiency_scores = [record["efficiency_ratio"] for record in self.consumption_history]
        avg_efficiency = np.mean(efficiency_scores)
        
        # Identify high-efficiency operations
        high_efficiency_ops = [
            record["operation"] for record in self.consumption_history
            if record["efficiency_ratio"] > avg_efficiency * 1.2
        ]
        
        # Generate optimization strategies
        self.optimization_strategies = [
            f"Prioritize {op} operations (high efficiency)" for op in set(high_efficiency_ops)
        ]
        
        # Add budget allocation recommendations
        if self.get_remaining_budget() < self.total_budget * 0.2:
            self.optimization_strategies.append("Conservative budget allocation recommended")
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate privacy budget optimization recommendations."""
        recommendations = []
        
        remaining_ratio = self.get_remaining_budget() / self.total_budget
        
        if remaining_ratio < 0.1:
            recommendations.append("Critical: Less than 10% budget remaining")
            recommendations.append("Consider increasing privacy parameters")
        elif remaining_ratio < 0.2:
            recommendations.append("Warning: Less than 20% budget remaining")
            recommendations.append("Prioritize high-utility operations")
        
        # Add efficiency-based recommendations
        recommendations.extend(self.optimization_strategies)
        
        return recommendations


class AdvancedPrivacyFramework:
    """Comprehensive framework for advanced privacy mechanisms."""
    
    def __init__(self):
        self.mechanisms = {
            PrivacyMechanism.ADAPTIVE_DIFFERENTIAL_PRIVACY: AdaptiveDifferentialPrivacy(total_budget=10.0),
            PrivacyMechanism.PERSONALIZED_DIFFERENTIAL_PRIVACY: PersonalizedDifferentialPrivacy(),
            PrivacyMechanism.CONCENTRATED_DIFFERENTIAL_PRIVACY: ConcentratedDifferentialPrivacy(rho=0.5, delta=1e-5),
            PrivacyMechanism.FUNCTIONAL_DIFFERENTIAL_PRIVACY: FunctionalDifferentialPrivacy()
        }
        self.privacy_odometer = PrivacyOdometer(total_budget=10.0)
        self.active_mechanisms: Set[PrivacyMechanism] = set()
        
    def enable_mechanism(self, mechanism: PrivacyMechanism):
        """Enable specific privacy mechanism."""
        self.active_mechanisms.add(mechanism)
        logger.info(f"Enabled privacy mechanism: {mechanism.value}")
    
    def apply_privacy_mechanism(self,
                              mechanism: PrivacyMechanism,
                              data: torch.Tensor,
                              parameters: PrivacyParameters,
                              context: Dict[str, Any]) -> torch.Tensor:
        """Apply specified privacy mechanism to data."""
        
        if mechanism not in self.mechanisms:
            raise ValueError(f"Unsupported privacy mechanism: {mechanism}")
        
        mechanism_impl = self.mechanisms[mechanism]
        
        # Record privacy consumption
        self.privacy_odometer.record_privacy_consumption(
            operation=mechanism.value,
            epsilon_used=parameters.epsilon,
            delta_used=parameters.delta,
            utility_gained=context.get("utility", 0.0),
            context=context
        )
        
        # Apply mechanism-specific processing
        if mechanism == PrivacyMechanism.ADAPTIVE_DIFFERENTIAL_PRIVACY:
            return mechanism_impl.add_adaptive_noise(
                data, parameters.epsilon, parameters.sensitivity
            )
        elif mechanism == PrivacyMechanism.CONCENTRATED_DIFFERENTIAL_PRIVACY:
            return mechanism_impl.gaussian_mechanism(data, parameters.sensitivity)
        elif mechanism == PrivacyMechanism.FUNCTIONAL_DIFFERENTIAL_PRIVACY:
            function = context.get("function", lambda x: x)
            data_domain = context.get("data_domain", (-1.0, 1.0))
            return mechanism_impl.add_functional_noise(
                data, function, parameters.epsilon, data_domain
            )
        else:
            # Default Gaussian mechanism
            noise_scale = parameters.sensitivity / parameters.epsilon
            noise = torch.randn_like(data) * noise_scale
            return data + noise
    
    def get_privacy_status(self) -> Dict[str, Any]:
        """Get comprehensive privacy status."""
        return {
            "active_mechanisms": [m.value for m in self.active_mechanisms],
            "privacy_budget_status": {
                "remaining_budget": self.privacy_odometer.get_remaining_budget(),
                "total_budget": self.privacy_odometer.total_budget,
                "consumption_rate": self.privacy_odometer._calculate_consumption_rate()
            },
            "optimization_recommendations": self.privacy_odometer._generate_optimization_recommendations(),
            "mechanism_count": len(self.mechanisms)
        }


# Utility functions
def create_advanced_privacy_framework() -> AdvancedPrivacyFramework:
    """Factory function to create advanced privacy framework."""
    return AdvancedPrivacyFramework()


def create_privacy_parameters(
    epsilon: float,
    delta: float,
    sensitivity: float = 1.0,
    **kwargs
) -> PrivacyParameters:
    """Helper function to create privacy parameters."""
    return PrivacyParameters(
        epsilon=epsilon,
        delta=delta,
        sensitivity=sensitivity,
        **kwargs
    )