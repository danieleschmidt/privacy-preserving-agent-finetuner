"""Adaptive privacy budget scheduler with dynamic optimization."""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

from .privacy_config import PrivacyConfig

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    """Privacy scheduling strategies."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    ADAPTIVE = "adaptive"
    QUANTUM_INSPIRED = "quantum_inspired"


@dataclass
class PrivacySchedule:
    """Privacy schedule configuration."""
    strategy: SchedulingStrategy
    initial_epsilon: float
    final_epsilon: float
    total_steps: int
    warmup_steps: int = 0
    decay_factor: float = 0.9
    adaptation_rate: float = 0.1
    min_epsilon: float = 0.01
    max_epsilon: float = 10.0


@dataclass
class AdaptationMetrics:
    """Metrics for adaptive privacy scheduling."""
    gradient_variance: float = 0.0
    loss_improvement: float = 0.0
    privacy_efficiency: float = 1.0
    convergence_rate: float = 0.0
    model_utility: float = 0.0
    risk_assessment: float = 0.0
    temporal_correlation: float = 0.0
    memory_consumption: float = 0.0


class AdaptivePrivacyScheduler:
    """Advanced adaptive privacy scheduler with multiple strategies.
    
    Dynamically adjusts privacy parameters based on:
    - Training dynamics (gradient variance, loss improvement)
    - Model utility metrics
    - Privacy-utility tradeoff analysis
    - Temporal correlations in data
    - Memory and computational constraints
    """
    
    def __init__(
        self,
        initial_config: PrivacyConfig,
        schedule: Optional[PrivacySchedule] = None,
        adaptation_window: int = 10,
        risk_threshold: float = 0.8
    ):
        """Initialize adaptive privacy scheduler.
        
        Args:
            initial_config: Initial privacy configuration
            schedule: Privacy schedule configuration
            adaptation_window: Window size for adaptation metrics
            risk_threshold: Risk threshold for privacy adjustments
        """
        self.initial_config = initial_config
        self.current_config = initial_config
        
        # Default schedule if not provided
        if schedule is None:
            schedule = PrivacySchedule(
                strategy=SchedulingStrategy.ADAPTIVE,
                initial_epsilon=initial_config.epsilon,
                final_epsilon=initial_config.epsilon * 0.5,
                total_steps=1000,
                warmup_steps=100
            )
        
        self.schedule = schedule
        self.adaptation_window = adaptation_window
        self.risk_threshold = risk_threshold
        
        # Tracking variables
        self.step = 0
        self.adaptation_history = []
        self.metrics_buffer = []
        self.privacy_spent = 0.0
        
        # Adaptive components
        self.utility_tracker = UtilityTracker()
        self.risk_assessor = PrivacyRiskAssessor()
        self.quantum_scheduler = QuantumInspiredScheduler(initial_config)
        
        logger.info(f"Initialized AdaptivePrivacyScheduler with {schedule.strategy.value} strategy")
    
    def update_privacy_config(
        self,
        metrics: AdaptationMetrics,
        training_state: Dict[str, Any]
    ) -> PrivacyConfig:
        """Update privacy configuration based on current metrics.
        
        Args:
            metrics: Current adaptation metrics
            training_state: Current training state
            
        Returns:
            Updated privacy configuration
        """
        self.step += 1
        self.metrics_buffer.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_buffer) > self.adaptation_window:
            self.metrics_buffer.pop(0)
        
        # Calculate new privacy parameters
        new_epsilon = self._calculate_adaptive_epsilon(metrics, training_state)
        new_delta = self._calculate_adaptive_delta(metrics, training_state)
        new_noise_multiplier = self._calculate_adaptive_noise(metrics, training_state)
        new_grad_norm = self._calculate_adaptive_grad_norm(metrics, training_state)
        
        # Risk assessment and bounds checking
        risk_score = self.risk_assessor.assess_risk(
            metrics, self.current_config, self.privacy_spent
        )
        
        if risk_score > self.risk_threshold:
            logger.warning(f"High privacy risk detected: {risk_score:.3f}")
            new_epsilon = min(new_epsilon, self.current_config.epsilon * 0.9)
            new_noise_multiplier = max(new_noise_multiplier, self.current_config.noise_multiplier * 1.1)
        
        # Apply bounds
        new_epsilon = np.clip(new_epsilon, self.schedule.min_epsilon, self.schedule.max_epsilon)
        new_delta = np.clip(new_delta, 1e-8, 1e-3)
        new_noise_multiplier = np.clip(new_noise_multiplier, 0.1, 5.0)
        new_grad_norm = np.clip(new_grad_norm, 0.1, 10.0)
        
        # Create updated configuration
        updated_config = PrivacyConfig(
            epsilon=new_epsilon,
            delta=new_delta,
            max_grad_norm=new_grad_norm,
            noise_multiplier=new_noise_multiplier,
            accounting_mode=self.current_config.accounting_mode,
            federated_enabled=self.current_config.federated_enabled,
            aggregation_method=self.current_config.aggregation_method,
            min_clients=self.current_config.min_clients
        )
        
        # Validate and apply
        updated_config.validate()
        self.current_config = updated_config
        
        # Update privacy spent
        self.privacy_spent += new_epsilon * 0.1  # Simplified accounting
        
        # Record adaptation
        adaptation_record = {
            "step": self.step,
            "epsilon": new_epsilon,
            "delta": new_delta,
            "noise_multiplier": new_noise_multiplier,
            "grad_norm": new_grad_norm,
            "risk_score": risk_score,
            "metrics": metrics,
            "timestamp": datetime.now()
        }
        self.adaptation_history.append(adaptation_record)
        
        logger.debug(f"Step {self.step}: ε={new_epsilon:.4f}, σ={new_noise_multiplier:.4f}, risk={risk_score:.3f}")
        
        return updated_config
    
    def _calculate_adaptive_epsilon(
        self,
        metrics: AdaptationMetrics,
        training_state: Dict[str, Any]
    ) -> float:
        """Calculate adaptive epsilon based on current metrics."""
        base_epsilon = self._get_scheduled_epsilon()
        
        if self.schedule.strategy == SchedulingStrategy.ADAPTIVE:
            # Adapt based on gradient variance and loss improvement
            variance_factor = 1.0 + 0.2 * np.tanh(metrics.gradient_variance - 1.0)
            improvement_factor = 1.0 - 0.1 * np.tanh(metrics.loss_improvement)
            utility_factor = 1.0 + 0.15 * (metrics.model_utility - 0.5)
            
            adaptation_factor = variance_factor * improvement_factor * utility_factor
            adapted_epsilon = base_epsilon * adaptation_factor
            
        elif self.schedule.strategy == SchedulingStrategy.QUANTUM_INSPIRED:
            # Use quantum scheduler
            adapted_epsilon = self.quantum_scheduler.get_quantum_epsilon(
                self.step, metrics, training_state
            )
            
        else:
            adapted_epsilon = base_epsilon
        
        # Smooth adaptation to prevent oscillations
        if len(self.adaptation_history) > 0:
            prev_epsilon = self.adaptation_history[-1]["epsilon"]
            smoothing_factor = 0.8
            adapted_epsilon = (
                smoothing_factor * prev_epsilon + 
                (1 - smoothing_factor) * adapted_epsilon
            )
        
        return adapted_epsilon
    
    def _calculate_adaptive_delta(
        self,
        metrics: AdaptationMetrics,
        training_state: Dict[str, Any]
    ) -> float:
        """Calculate adaptive delta based on privacy requirements."""
        base_delta = self.current_config.delta
        
        # Adapt delta based on risk assessment and data sensitivity
        risk_factor = 1.0 + 0.5 * metrics.risk_assessment
        temporal_factor = 1.0 + 0.2 * metrics.temporal_correlation
        
        adapted_delta = base_delta * risk_factor * temporal_factor
        
        return adapted_delta
    
    def _calculate_adaptive_noise(
        self,
        metrics: AdaptationMetrics,
        training_state: Dict[str, Any]
    ) -> float:
        """Calculate adaptive noise multiplier."""
        base_noise = self.current_config.noise_multiplier
        
        # Adapt noise based on gradient variance and convergence
        if metrics.gradient_variance > 2.0:
            # High variance - reduce noise to allow faster convergence
            noise_factor = 0.9
        elif metrics.gradient_variance < 0.5:
            # Low variance - increase noise for better privacy
            noise_factor = 1.1
        else:
            noise_factor = 1.0
        
        # Consider convergence rate
        if metrics.convergence_rate < 0.1:
            # Slow convergence - reduce noise
            convergence_factor = 0.95
        else:
            convergence_factor = 1.0
        
        adapted_noise = base_noise * noise_factor * convergence_factor
        
        return adapted_noise
    
    def _calculate_adaptive_grad_norm(
        self,
        metrics: AdaptationMetrics,
        training_state: Dict[str, Any]
    ) -> float:
        """Calculate adaptive gradient clipping norm."""
        base_norm = self.current_config.max_grad_norm
        
        # Adapt based on gradient variance
        if metrics.gradient_variance > 3.0:
            # High variance - increase clipping
            norm_factor = 1.2
        elif metrics.gradient_variance < 0.3:
            # Low variance - allow larger gradients
            norm_factor = 0.9
        else:
            norm_factor = 1.0
        
        adapted_norm = base_norm * norm_factor
        
        return adapted_norm
    
    def _get_scheduled_epsilon(self) -> float:
        """Get epsilon value according to schedule strategy."""
        if self.step >= self.schedule.total_steps:
            return self.schedule.final_epsilon
        
        progress = self.step / self.schedule.total_steps
        
        if self.step < self.schedule.warmup_steps:
            # Warmup phase - linear increase
            warmup_progress = self.step / self.schedule.warmup_steps
            return self.schedule.initial_epsilon * warmup_progress
        
        if self.schedule.strategy == SchedulingStrategy.LINEAR:
            epsilon = (
                self.schedule.initial_epsilon + 
                progress * (self.schedule.final_epsilon - self.schedule.initial_epsilon)
            )
            
        elif self.schedule.strategy == SchedulingStrategy.EXPONENTIAL:
            epsilon = (
                self.schedule.initial_epsilon * 
                (self.schedule.decay_factor ** progress)
            )
            
        elif self.schedule.strategy == SchedulingStrategy.COSINE:
            epsilon = (
                self.schedule.final_epsilon + 
                0.5 * (self.schedule.initial_epsilon - self.schedule.final_epsilon) *
                (1 + np.cos(np.pi * progress))
            )
            
        else:  # ADAPTIVE or QUANTUM_INSPIRED
            epsilon = self.schedule.initial_epsilon
        
        return epsilon
    
    def get_privacy_efficiency(self) -> float:
        """Calculate current privacy efficiency score."""
        if len(self.metrics_buffer) == 0:
            return 1.0
        
        recent_metrics = self.metrics_buffer[-5:]  # Last 5 measurements
        
        # Privacy efficiency based on utility vs. privacy spent
        avg_utility = np.mean([m.model_utility for m in recent_metrics])
        privacy_cost = self.privacy_spent / self.schedule.initial_epsilon
        
        efficiency = avg_utility / (privacy_cost + 1e-10)
        return min(1.0, efficiency)
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get comprehensive adaptation summary."""
        if not self.adaptation_history:
            return {"status": "no_adaptations"}
        
        recent_adaptations = self.adaptation_history[-10:]
        
        summary = {
            "total_adaptations": len(self.adaptation_history),
            "current_step": self.step,
            "privacy_spent": self.privacy_spent,
            "privacy_efficiency": self.get_privacy_efficiency(),
            "current_config": self.current_config.to_dict(),
            "adaptation_trends": {
                "epsilon_trend": [a["epsilon"] for a in recent_adaptations],
                "noise_trend": [a["noise_multiplier"] for a in recent_adaptations],
                "risk_trend": [a["risk_score"] for a in recent_adaptations]
            },
            "schedule_info": {
                "strategy": self.schedule.strategy.value,
                "progress": min(1.0, self.step / self.schedule.total_steps),
                "remaining_steps": max(0, self.schedule.total_steps - self.step)
            }
        }
        
        return summary


class UtilityTracker:
    """Tracks model utility metrics for privacy-utility tradeoff analysis."""
    
    def __init__(self):
        self.utility_history = []
        self.baseline_utility = None
    
    def update_utility(self, accuracy: float, loss: float, **kwargs) -> float:
        """Update utility metrics and return current utility score."""
        utility_score = accuracy - 0.1 * loss  # Simple utility metric
        
        utility_record = {
            "accuracy": accuracy,
            "loss": loss,
            "utility_score": utility_score,
            "timestamp": datetime.now(),
            **kwargs
        }
        
        self.utility_history.append(utility_record)
        
        # Set baseline on first measurement
        if self.baseline_utility is None:
            self.baseline_utility = utility_score
        
        return utility_score
    
    def get_utility_degradation(self) -> float:
        """Calculate utility degradation from baseline."""
        if not self.utility_history or self.baseline_utility is None:
            return 0.0
        
        current_utility = self.utility_history[-1]["utility_score"]
        degradation = max(0.0, self.baseline_utility - current_utility)
        
        return degradation / (self.baseline_utility + 1e-10)


class PrivacyRiskAssessor:
    """Assesses privacy risks and provides risk scores."""
    
    def __init__(self):
        self.risk_factors = {
            "gradient_leakage": 0.3,
            "model_memorization": 0.25,
            "inference_attacks": 0.2,
            "temporal_correlation": 0.15,
            "budget_exhaustion": 0.1
        }
    
    def assess_risk(
        self,
        metrics: AdaptationMetrics,
        config: PrivacyConfig,
        privacy_spent: float
    ) -> float:
        """Assess overall privacy risk score (0-1)."""
        risk_scores = {}
        
        # Gradient leakage risk
        risk_scores["gradient_leakage"] = min(1.0, metrics.gradient_variance / 3.0)
        
        # Model memorization risk
        risk_scores["model_memorization"] = 1.0 - metrics.model_utility
        
        # Inference attack risk based on convergence
        risk_scores["inference_attacks"] = min(1.0, metrics.convergence_rate / 0.5)
        
        # Temporal correlation risk
        risk_scores["temporal_correlation"] = metrics.temporal_correlation
        
        # Budget exhaustion risk
        budget_ratio = privacy_spent / config.epsilon
        risk_scores["budget_exhaustion"] = min(1.0, budget_ratio)
        
        # Weighted overall risk
        overall_risk = sum(
            self.risk_factors[factor] * score
            for factor, score in risk_scores.items()
        )
        
        return overall_risk


class QuantumInspiredScheduler:
    """Quantum-inspired privacy scheduling using superposition principles."""
    
    def __init__(self, initial_config: PrivacyConfig):
        self.initial_config = initial_config
        self.quantum_state = np.random.rand(8) + 1j * np.random.rand(8)
        self.quantum_state /= np.linalg.norm(self.quantum_state)
    
    def get_quantum_epsilon(
        self,
        step: int,
        metrics: AdaptationMetrics,
        training_state: Dict[str, Any]
    ) -> float:
        """Calculate epsilon using quantum superposition of multiple strategies."""
        # Evolve quantum state
        time_factor = step * 0.01
        evolution = np.exp(-1j * time_factor)
        self.quantum_state *= evolution
        
        # Renormalize
        self.quantum_state /= np.linalg.norm(self.quantum_state)
        
        # Superposition of epsilon values
        epsilon_strategies = [
            self.initial_config.epsilon,  # Constant
            self.initial_config.epsilon * (1 - step / 1000),  # Linear decay
            self.initial_config.epsilon * np.exp(-step / 500),  # Exponential decay
            self.initial_config.epsilon * (1 + 0.1 * np.sin(step / 100)),  # Oscillating
        ]
        
        # Quantum amplitudes determine strategy weights
        amplitudes = np.abs(self.quantum_state[:len(epsilon_strategies)]) ** 2
        amplitudes /= np.sum(amplitudes)
        
        # Superposition epsilon
        quantum_epsilon = np.sum(amplitudes * epsilon_strategies)
        
        return quantum_epsilon