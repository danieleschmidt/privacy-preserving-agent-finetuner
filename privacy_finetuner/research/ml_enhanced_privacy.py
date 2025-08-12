"""ML-Enhanced Privacy Protection for Advanced Research.

This module implements machine learning approaches to enhance privacy protection,
including adversarial privacy leakage prediction, reinforcement learning for
dynamic budget allocation, neural differential privacy, and privacy-aware NAS.

Research Reference: Novel ML techniques for adaptive privacy protection and
intelligent privacy-utility optimization in machine learning systems.
"""

import logging
import time
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import hashlib

# Handle numpy import gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Import stub from novel_algorithms
    from .novel_algorithms import NumpyStub, RandomStub
    np = NumpyStub()
    np.random = RandomStub()

logger = logging.getLogger(__name__)


class PrivacyAttackType(Enum):
    """Types of privacy attacks to defend against."""
    MEMBERSHIP_INFERENCE = "membership_inference"
    PROPERTY_INFERENCE = "property_inference"
    MODEL_INVERSION = "model_inversion"
    ATTRIBUTE_INFERENCE = "attribute_inference"
    RECONSTRUCTION = "reconstruction"


@dataclass
class PrivacyLeakageMetrics:
    """Metrics for privacy leakage assessment."""
    attack_success_rate: float
    confidence_interval: Tuple[float, float]
    mutual_information: float
    entropy_loss: float
    distinguishability: float
    privacy_risk_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack_success_rate": self.attack_success_rate,
            "confidence_interval": list(self.confidence_interval),
            "mutual_information": self.mutual_information,
            "entropy_loss": self.entropy_loss,
            "distinguishability": self.distinguishability,
            "privacy_risk_score": self.privacy_risk_score
        }


@dataclass
class NeuralPrivacyConfig:
    """Configuration for neural differential privacy."""
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation_function: str = "relu"
    noise_distribution: str = "gaussian"
    adaptive_noise: bool = True
    privacy_budget_decay: float = 0.95
    learning_rate: float = 0.001
    
    def validate(self) -> bool:
        """Validate neural privacy configuration."""
        return (len(self.hidden_layers) > 0 and 
                self.learning_rate > 0 and 
                0 < self.privacy_budget_decay < 1)


class AdversarialPrivacyPredictor:
    """Adversarial machine learning system for privacy leakage prediction.
    
    This system uses adversarial training to predict potential privacy leaks
    and proactively adjust privacy protection mechanisms.
    """
    
    def __init__(
        self,
        model_architecture: List[int] = None,
        adversary_strength: float = 0.8,
        defense_budget: float = 0.1,
        attack_types: Optional[List[PrivacyAttackType]] = None
    ):
        """Initialize adversarial privacy predictor.
        
        Args:
            model_architecture: Neural network architecture for predictor
            adversary_strength: Strength of adversarial attacks (0-1)
            defense_budget: Budget allocated for defense mechanisms
            attack_types: Types of privacy attacks to defend against
        """
        self.model_architecture = model_architecture or [64, 32, 16]
        self.adversary_strength = adversary_strength
        self.defense_budget = defense_budget
        
        self.attack_types = attack_types or [
            PrivacyAttackType.MEMBERSHIP_INFERENCE,
            PrivacyAttackType.PROPERTY_INFERENCE,
            PrivacyAttackType.MODEL_INVERSION
        ]
        
        # Initialize adversarial models
        self.attack_models = self._initialize_attack_models()
        self.defense_model = self._initialize_defense_model()
        
        # Training history
        self.training_history = []
        self.attack_success_history = []
        
        logger.info(f"Initialized AdversarialPrivacyPredictor with {len(self.attack_types)} attack types")
    
    def _initialize_attack_models(self) -> Dict[str, Any]:
        """Initialize adversarial attack models."""
        attack_models = {}
        
        for attack_type in self.attack_types:
            # Simple neural network simulator for each attack type
            attack_models[attack_type.value] = {
                "weights": [np.random.normal(0, 0.1, (self.model_architecture[i], self.model_architecture[i+1])) 
                           for i in range(len(self.model_architecture)-1)],
                "biases": [np.random.normal(0, 0.01, self.model_architecture[i+1]) 
                          for i in range(len(self.model_architecture)-1)],
                "success_rate": 0.5,  # Random baseline
                "training_epochs": 0
            }
        
        return attack_models
    
    def _initialize_defense_model(self) -> Dict[str, Any]:
        """Initialize defense model."""
        defense_architecture = [sum(self.model_architecture)] + self.model_architecture + [1]
        
        return {
            "weights": [np.random.normal(0, 0.1, (defense_architecture[i], defense_architecture[i+1])) 
                       for i in range(len(defense_architecture)-1)],
            "biases": [np.random.normal(0, 0.01, defense_architecture[i+1]) 
                      for i in range(len(defense_architecture)-1)],
            "architecture": defense_architecture,
            "training_epochs": 0
        }
    
    def predict_privacy_leakage(
        self,
        model_gradients: np.ndarray,
        training_data_statistics: Dict[str, Any],
        current_privacy_budget: float
    ) -> PrivacyLeakageMetrics:
        """Predict privacy leakage using adversarial models.
        
        Args:
            model_gradients: Current model gradients
            training_data_statistics: Statistics about training data
            current_privacy_budget: Current privacy budget consumption
            
        Returns:
            Privacy leakage metrics and predictions
        """
        logger.debug("Predicting privacy leakage with adversarial models")
        
        # Extract features for privacy analysis
        features = self._extract_privacy_features(
            model_gradients, training_data_statistics, current_privacy_budget
        )
        
        # Run adversarial attacks
        attack_results = {}
        total_attack_success = 0.0
        
        for attack_type in self.attack_types:
            attack_success = self._simulate_adversarial_attack(
                attack_type, features
            )
            attack_results[attack_type.value] = attack_success
            total_attack_success += attack_success
        
        avg_attack_success = total_attack_success / len(self.attack_types)
        
        # Predict defense effectiveness
        defense_effectiveness = self._predict_defense_effectiveness(features)
        
        # Compute privacy risk metrics
        mutual_information = self._estimate_mutual_information(features, attack_results)
        entropy_loss = self._compute_entropy_loss(features)
        distinguishability = self._compute_distinguishability(features, attack_results)
        
        # Overall privacy risk score
        privacy_risk_score = (
            0.4 * avg_attack_success +
            0.3 * mutual_information +
            0.2 * entropy_loss +
            0.1 * distinguishability
        )
        
        # Confidence interval estimation
        confidence_interval = self._compute_confidence_interval(
            avg_attack_success, len(self.attack_types)
        )
        
        leakage_metrics = PrivacyLeakageMetrics(
            attack_success_rate=avg_attack_success,
            confidence_interval=confidence_interval,
            mutual_information=mutual_information,
            entropy_loss=entropy_loss,
            distinguishability=distinguishability,
            privacy_risk_score=privacy_risk_score
        )
        
        # Update attack success history
        self.attack_success_history.append({
            "timestamp": time.time(),
            "metrics": leakage_metrics.to_dict(),
            "attack_results": attack_results
        })
        
        return leakage_metrics
    
    def _extract_privacy_features(
        self,
        gradients: np.ndarray,
        data_stats: Dict[str, Any],
        budget: float
    ) -> np.ndarray:
        """Extract features for privacy leakage prediction."""
        features = []
        
        # Gradient-based features
        if len(gradients) > 0:
            features.extend([
                np.mean(gradients),
                np.std(gradients),
                np.max(np.abs(gradients)),
                np.linalg.norm(gradients),
                np.percentile(gradients, 95) - np.percentile(gradients, 5)  # Range
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Data statistics features
        features.extend([
            data_stats.get("num_samples", 0) / 10000.0,  # Normalize
            data_stats.get("feature_dimension", 0) / 1000.0,
            data_stats.get("class_imbalance", 0.5),
            data_stats.get("noise_level", 0.0)
        ])
        
        # Privacy budget features
        features.extend([
            budget,
            budget ** 2,  # Quadratic term
            1.0 / max(budget, 1e-6),  # Inverse
            math.log(max(budget, 1e-6))  # Log term
        ])
        
        return np.array(features)
    
    def _simulate_adversarial_attack(
        self,
        attack_type: PrivacyAttackType,
        features: np.ndarray
    ) -> float:
        """Simulate adversarial attack and return success rate."""
        attack_model = self.attack_models[attack_type.value]
        
        # Forward pass through attack model
        x = features
        for i, (weight, bias) in enumerate(zip(attack_model["weights"], attack_model["biases"])):
            x = np.dot(x, weight) + bias
            
            # ReLU activation for hidden layers
            if i < len(attack_model["weights"]) - 1:
                x = np.maximum(0, x)
        
        # Sigmoid activation for output (success probability)
        attack_success = 1.0 / (1.0 + np.exp(-np.mean(x)))
        
        # Add adversarial strength factor
        attack_success = min(1.0, attack_success * self.adversary_strength)
        
        return attack_success
    
    def _predict_defense_effectiveness(self, features: np.ndarray) -> float:
        """Predict effectiveness of current defense mechanisms."""
        defense_model = self.defense_model
        
        # Forward pass through defense model
        x = features
        for i, (weight, bias) in enumerate(zip(defense_model["weights"], defense_model["biases"])):
            x = np.dot(x, weight) + bias
            
            # ReLU activation for hidden layers
            if i < len(defense_model["weights"]) - 1:
                x = np.maximum(0, x)
        
        # Sigmoid activation for defense effectiveness
        defense_effectiveness = 1.0 / (1.0 + np.exp(-x[0]))
        
        return defense_effectiveness
    
    def _estimate_mutual_information(
        self,
        features: np.ndarray,
        attack_results: Dict[str, float]
    ) -> float:
        """Estimate mutual information between model and private data."""
        # Simplified mutual information estimation
        avg_attack_success = np.mean(list(attack_results.values()))
        
        # MI increases with attack success and feature complexity
        feature_complexity = np.std(features) / max(np.mean(np.abs(features)), 1e-6)
        
        mutual_information = avg_attack_success * math.log2(1 + feature_complexity)
        
        return min(1.0, mutual_information)
    
    def _compute_entropy_loss(self, features: np.ndarray) -> float:
        """Compute entropy loss indicating privacy degradation."""
        # Entropy loss based on feature distribution
        if len(features) == 0:
            return 0.0
        
        # Discretize features for entropy calculation
        num_bins = min(10, len(features))
        hist, _ = np.histogram(features, bins=num_bins)
        
        # Normalize histogram
        hist = hist / max(np.sum(hist), 1)
        
        # Compute entropy
        entropy = -np.sum([p * math.log2(p) for p in hist if p > 0])
        max_entropy = math.log2(num_bins)
        
        # Entropy loss is reduction from maximum entropy
        entropy_loss = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        
        return max(0.0, entropy_loss)
    
    def _compute_distinguishability(
        self,
        features: np.ndarray,
        attack_results: Dict[str, float]
    ) -> float:
        """Compute distinguishability of model outputs."""
        # Distinguishability based on feature variance and attack success
        if len(features) == 0:
            return 0.0
        
        feature_variance = np.var(features)
        avg_attack_success = np.mean(list(attack_results.values()))
        
        # Higher variance and attack success indicate higher distinguishability
        distinguishability = math.tanh(feature_variance) * avg_attack_success
        
        return distinguishability
    
    def _compute_confidence_interval(
        self,
        estimate: float,
        num_samples: int,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Compute confidence interval for privacy estimates."""
        # Simplified confidence interval using normal approximation
        if num_samples <= 1:
            return (estimate, estimate)
        
        # Standard error estimation
        std_error = math.sqrt(estimate * (1 - estimate) / num_samples)
        
        # Z-score for confidence level
        z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
        
        margin_of_error = z_score * std_error
        
        lower_bound = max(0.0, estimate - margin_of_error)
        upper_bound = min(1.0, estimate + margin_of_error)
        
        return (lower_bound, upper_bound)
    
    def adversarial_training_step(
        self,
        training_features: List[np.ndarray],
        true_leakage_labels: List[float],
        learning_rate: float = 0.001
    ) -> Dict[str, float]:
        """Perform one step of adversarial training.
        
        Args:
            training_features: Features for training
            true_leakage_labels: True privacy leakage labels
            learning_rate: Learning rate for updates
            
        Returns:
            Training metrics
        """
        if not training_features or not true_leakage_labels:
            return {"error": "No training data provided"}
        
        logger.debug(f"Adversarial training step with {len(training_features)} samples")
        
        # Train attack models (adversarial objective)
        attack_losses = {}
        for attack_type in self.attack_types:
            attack_loss = self._train_attack_model(
                attack_type, training_features, true_leakage_labels, learning_rate
            )
            attack_losses[attack_type.value] = attack_loss
        
        # Train defense model (defensive objective)
        defense_loss = self._train_defense_model(
            training_features, true_leakage_labels, learning_rate
        )
        
        # Update training history
        training_metrics = {
            "attack_losses": attack_losses,
            "defense_loss": defense_loss,
            "avg_attack_loss": np.mean(list(attack_losses.values())),
            "learning_rate": learning_rate,
            "training_samples": len(training_features)
        }
        
        self.training_history.append({
            "timestamp": time.time(),
            "metrics": training_metrics
        })
        
        return training_metrics
    
    def _train_attack_model(
        self,
        attack_type: PrivacyAttackType,
        features_list: List[np.ndarray],
        labels: List[float],
        lr: float
    ) -> float:
        """Train individual attack model."""
        attack_model = self.attack_models[attack_type.value]
        total_loss = 0.0
        
        for features, label in zip(features_list, labels):
            # Forward pass
            x = features
            activations = [x]
            
            for weight, bias in zip(attack_model["weights"], attack_model["biases"]):
                x = np.dot(x, weight) + bias
                x = np.maximum(0, x)  # ReLU
                activations.append(x)
            
            # Output with sigmoid
            output = 1.0 / (1.0 + np.exp(-np.mean(x)))
            
            # Loss (binary cross-entropy)
            loss = -label * math.log(max(output, 1e-10)) - (1-label) * math.log(max(1-output, 1e-10))
            total_loss += loss
            
            # Backward pass (simplified gradient computation)
            output_grad = output - label
            
            # Update weights (simplified)
            for i, (weight, bias) in enumerate(zip(attack_model["weights"], attack_model["biases"])):
                weight_grad = np.outer(activations[i], output_grad * 0.1)  # Simplified
                bias_grad = output_grad * 0.1
                
                attack_model["weights"][i] -= lr * weight_grad
                attack_model["biases"][i] -= lr * bias_grad
        
        attack_model["training_epochs"] += 1
        return total_loss / len(features_list)
    
    def _train_defense_model(
        self,
        features_list: List[np.ndarray],
        labels: List[float],
        lr: float
    ) -> float:
        """Train defense model."""
        defense_model = self.defense_model
        total_loss = 0.0
        
        for features, label in zip(features_list, labels):
            # Forward pass
            x = features
            activations = [x]
            
            for weight, bias in zip(defense_model["weights"], defense_model["biases"]):
                x = np.dot(x, weight) + bias
                x = np.maximum(0, x)  # ReLU
                activations.append(x)
            
            # Output (defense effectiveness prediction)
            output = 1.0 / (1.0 + np.exp(-x[0]))
            
            # Defense loss: we want to minimize privacy leakage
            defense_target = 1.0 - label  # Higher defense for higher leakage
            loss = (output - defense_target) ** 2
            total_loss += loss
            
            # Backward pass (simplified)
            output_grad = 2 * (output - defense_target)
            
            # Update weights (simplified)
            for i, (weight, bias) in enumerate(zip(defense_model["weights"], defense_model["biases"])):
                weight_grad = np.outer(activations[i], output_grad * 0.1)
                bias_grad = output_grad * 0.1
                
                defense_model["weights"][i] -= lr * weight_grad
                defense_model["biases"][i] -= lr * bias_grad
        
        defense_model["training_epochs"] += 1
        return total_loss / len(features_list)


class ReinforcementLearningBudgetAllocator:
    """Reinforcement learning system for dynamic privacy budget allocation.
    
    Uses Q-learning to optimally allocate privacy budgets across different
    components based on utility feedback and privacy requirements.
    """
    
    def __init__(
        self,
        num_components: int,
        total_budget: float = 1.0,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.1
    ):
        """Initialize RL budget allocator.
        
        Args:
            num_components: Number of components requiring budget allocation
            total_budget: Total privacy budget available
            learning_rate: Q-learning learning rate
            discount_factor: Discount factor for future rewards
            exploration_rate: Epsilon for epsilon-greedy exploration
        """
        self.num_components = num_components
        self.total_budget = total_budget
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # RL components
        self.state_dim = num_components * 3  # budget, utility, privacy for each component
        self.action_dim = num_components  # budget allocation for each component
        
        # Q-table (simplified discrete state-action space)
        self.q_table = {}
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        # Current state
        self.current_state = self._initialize_state()
        
        logger.info(f"Initialized RL Budget Allocator for {num_components} components")
    
    def _initialize_state(self) -> np.ndarray:
        """Initialize RL state representation."""
        # State: [budget_remaining_1, utility_1, privacy_1, ..., budget_remaining_n, utility_n, privacy_n]
        state = []
        
        for i in range(self.num_components):
            state.extend([
                self.total_budget / self.num_components,  # Initial equal budget
                0.5,  # Initial utility
                0.5   # Initial privacy level
            ])
        
        return np.array(state)
    
    def _discretize_state(self, state: np.ndarray, num_bins: int = 10) -> str:
        """Discretize continuous state for Q-table."""
        discretized = []
        
        for value in state:
            bin_idx = min(num_bins - 1, int(value * num_bins))
            discretized.append(str(bin_idx))
        
        return "_".join(discretized)
    
    def _discretize_action(self, action: np.ndarray, num_bins: int = 5) -> str:
        """Discretize continuous action for Q-table."""
        # Normalize action to sum to 1
        normalized_action = action / max(np.sum(action), 1e-6)
        
        discretized = []
        for value in normalized_action:
            bin_idx = min(num_bins - 1, int(value * num_bins))
            discretized.append(str(bin_idx))
        
        return "_".join(discretized)
    
    def allocate_budget_rl(
        self,
        component_utilities: List[float],
        privacy_requirements: List[float],
        previous_rewards: Optional[List[float]] = None
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Allocate budget using reinforcement learning.
        
        Args:
            component_utilities: Current utility for each component
            privacy_requirements: Privacy requirements for each component
            previous_rewards: Rewards from previous allocation (for learning)
            
        Returns:
            Tuple of (allocated_budgets, allocation_metadata)
        """
        if len(component_utilities) != self.num_components:
            raise ValueError(f"Expected {self.num_components} utilities, got {len(component_utilities)}")
        
        logger.debug("RL budget allocation step")
        
        # Update current state
        self.current_state = self._update_state(
            component_utilities, privacy_requirements
        )
        
        # Learn from previous experience if available
        if previous_rewards and len(self.state_history) > 0:
            self._update_q_table(previous_rewards)
        
        # Select action (budget allocation) using epsilon-greedy
        action = self._select_action(self.current_state)
        
        # Convert action to actual budget allocation
        allocated_budgets = self._action_to_budget_allocation(action)
        
        # Record experience
        state_key = self._discretize_state(self.current_state)
        action_key = self._discretize_action(action)
        
        self.state_history.append(state_key)
        self.action_history.append(action_key)
        
        # Allocation metadata
        allocation_metadata = {
            "rl_state": self.current_state.tolist(),
            "rl_action": action.tolist(),
            "q_value": self.q_table.get(f"{state_key}_{action_key}", 0.0),
            "exploration_rate": self.exploration_rate,
            "total_allocated": sum(allocated_budgets),
            "q_table_size": len(self.q_table)
        }
        
        return allocated_budgets, allocation_metadata
    
    def _update_state(
        self,
        utilities: List[float],
        privacy_requirements: List[float]
    ) -> np.ndarray:
        """Update RL state based on current system state."""
        new_state = []
        
        for i in range(self.num_components):
            # Budget remaining (from current state)
            budget_idx = i * 3
            current_budget = self.current_state[budget_idx] if len(self.current_state) > budget_idx else 0.1
            
            # Update state components
            new_state.extend([
                current_budget,
                utilities[i],
                privacy_requirements[i]
            ])
        
        return np.array(new_state)
    
    def _select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using epsilon-greedy policy."""
        state_key = self._discretize_state(state)
        
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            # Random action (exploration)
            action = np.random.dirichlet(np.ones(self.num_components))  # Random budget allocation
        else:
            # Greedy action (exploitation)
            action = self._get_best_action(state_key)
        
        return action
    
    def _get_best_action(self, state_key: str) -> np.ndarray:
        """Get best action for given state."""
        # Find best action from Q-table
        best_action = None
        best_q_value = float('-inf')
        
        # Search through all actions in Q-table for this state
        for key, q_value in self.q_table.items():
            if key.startswith(state_key + "_"):
                if q_value > best_q_value:
                    best_q_value = q_value
                    action_key = key.split("_", len(state_key.split("_")))[-1]
                    best_action = self._action_key_to_array(action_key)
        
        # If no action found, use uniform allocation
        if best_action is None:
            best_action = np.ones(self.num_components) / self.num_components
        
        return best_action
    
    def _action_key_to_array(self, action_key: str) -> np.ndarray:
        """Convert action key back to array."""
        bins = [int(x) for x in action_key.split("_")]
        action = np.array(bins, dtype=float) / 4.0  # Assuming 5 bins (0-4)
        
        # Normalize to sum to 1
        action = action / max(np.sum(action), 1e-6)
        
        return action
    
    def _action_to_budget_allocation(self, action: np.ndarray) -> List[float]:
        """Convert RL action to actual budget allocation."""
        # Normalize action to sum to total budget
        normalized_action = action / max(np.sum(action), 1e-6)
        allocated_budgets = (normalized_action * self.total_budget).tolist()
        
        # Ensure minimum allocation for each component
        min_allocation = self.total_budget * 0.01  # 1% minimum
        for i in range(len(allocated_budgets)):
            allocated_budgets[i] = max(allocated_budgets[i], min_allocation)
        
        # Renormalize to maintain total budget
        total_allocated = sum(allocated_budgets)
        if total_allocated > 0:
            scaling_factor = self.total_budget / total_allocated
            allocated_budgets = [b * scaling_factor for b in allocated_budgets]
        
        return allocated_budgets
    
    def _update_q_table(self, rewards: List[float]) -> None:
        """Update Q-table based on received rewards."""
        if len(self.state_history) < 2 or len(self.action_history) < 2:
            return
        
        # Get previous state-action pair
        prev_state_key = self.state_history[-2]
        prev_action_key = self.action_history[-2]
        prev_key = f"{prev_state_key}_{prev_action_key}"
        
        # Current state-action pair
        curr_state_key = self.state_history[-1]
        curr_action_key = self.action_history[-1]
        curr_key = f"{curr_state_key}_{curr_action_key}"
        
        # Compute reward from component rewards
        total_reward = np.mean(rewards)
        
        # Q-learning update
        old_q_value = self.q_table.get(prev_key, 0.0)
        
        # Best Q-value for current state
        best_next_q = 0.0
        for key, q_val in self.q_table.items():
            if key.startswith(curr_state_key + "_"):
                best_next_q = max(best_next_q, q_val)
        
        # Q-learning update rule
        new_q_value = old_q_value + self.learning_rate * (
            total_reward + self.discount_factor * best_next_q - old_q_value
        )
        
        self.q_table[prev_key] = new_q_value
        
        # Record reward
        self.reward_history.append(total_reward)
        
        # Decay exploration rate
        self.exploration_rate *= 0.995
        self.exploration_rate = max(0.01, self.exploration_rate)  # Minimum exploration
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get RL learning statistics."""
        if not self.reward_history:
            return {"error": "No learning history available"}
        
        recent_rewards = self.reward_history[-100:]  # Last 100 episodes
        
        return {
            "total_episodes": len(self.reward_history),
            "q_table_size": len(self.q_table),
            "current_exploration_rate": self.exploration_rate,
            "average_reward_recent": np.mean(recent_rewards),
            "average_reward_overall": np.mean(self.reward_history),
            "reward_trend": "improving" if len(recent_rewards) > 1 and 
                          np.mean(recent_rewards[-50:]) > np.mean(recent_rewards[:50]) else "stable",
            "learning_progress": min(1.0, len(self.reward_history) / 1000.0)  # Progress towards convergence
        }


class NeuralDifferentialPrivacy:
    """Neural differential privacy with learned noise distributions.
    
    This system learns optimal noise distributions for differential privacy
    using neural networks instead of fixed Gaussian or Laplacian noise.
    """
    
    def __init__(
        self,
        config: NeuralPrivacyConfig,
        epsilon: float = 1.0,
        delta: float = 1e-5
    ):
        """Initialize neural differential privacy system.
        
        Args:
            config: Neural privacy configuration
            epsilon: Privacy parameter epsilon
            delta: Privacy parameter delta
        """
        self.config = config
        self.epsilon = epsilon
        self.delta = delta
        
        # Neural noise generator
        self.noise_generator = self._initialize_noise_generator()
        
        # Privacy accountant
        self.privacy_spent = 0.0
        self.noise_history = []
        
        logger.info(f"Initialized Neural Differential Privacy (ε={epsilon}, δ={delta})")
    
    def _initialize_noise_generator(self) -> Dict[str, Any]:
        """Initialize neural noise generator network."""
        # Architecture: input features -> hidden layers -> noise parameters
        layers = [10] + self.config.hidden_layers + [4]  # Output: [mean, std, skew, kurtosis]
        
        weights = []
        biases = []
        
        for i in range(len(layers) - 1):
            # Xavier initialization
            fan_in, fan_out = layers[i], layers[i + 1]
            weight_bound = math.sqrt(6.0 / (fan_in + fan_out))
            
            weight = np.random.uniform(-weight_bound, weight_bound, (fan_in, fan_out))
            bias = np.zeros(fan_out)
            
            weights.append(weight)
            biases.append(bias)
        
        return {
            "weights": weights,
            "biases": biases,
            "architecture": layers,
            "training_epochs": 0,
            "loss_history": []
        }
    
    def generate_adaptive_noise(
        self,
        gradients: np.ndarray,
        sensitivity: float,
        utility_target: float = 0.8
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate adaptive noise using neural network.
        
        Args:
            gradients: Input gradients to add noise to
            sensitivity: L2 sensitivity of the gradients
            utility_target: Target utility retention (0-1)
            
        Returns:
            Tuple of (noisy_gradients, noise_metadata)
        """
        logger.debug("Generating adaptive neural noise")
        
        # Extract features for noise generation
        gradient_features = self._extract_gradient_features(gradients, sensitivity)
        
        # Generate noise parameters using neural network
        noise_params = self._forward_pass_noise_generator(gradient_features, utility_target)
        
        # Generate noise from learned distribution
        noise = self._sample_from_learned_distribution(
            noise_params, gradients.shape
        )
        
        # Apply privacy constraints
        noise = self._apply_privacy_constraints(noise, sensitivity)
        
        # Add noise to gradients
        noisy_gradients = gradients + noise
        
        # Update privacy accounting
        privacy_cost = self._compute_privacy_cost(noise_params, sensitivity)
        self.privacy_spent += privacy_cost
        
        # Noise metadata
        noise_metadata = {
            "noise_parameters": noise_params.tolist(),
            "privacy_cost": privacy_cost,
            "total_privacy_spent": self.privacy_spent,
            "utility_estimate": self._estimate_utility_retention(gradients, noisy_gradients),
            "noise_distribution": "learned_neural",
            "sensitivity": sensitivity
        }
        
        self.noise_history.append(noise_metadata)
        
        return noisy_gradients, noise_metadata
    
    def _extract_gradient_features(
        self,
        gradients: np.ndarray,
        sensitivity: float
    ) -> np.ndarray:
        """Extract features from gradients for noise generation."""
        features = []
        
        if len(gradients) > 0:
            # Statistical features
            features.extend([
                np.mean(gradients),
                np.std(gradients),
                np.min(gradients),
                np.max(gradients),
                np.median(gradients)
            ])
            
            # Shape features
            features.extend([
                len(gradients.flatten()) / 10000.0,  # Normalized size
                np.linalg.norm(gradients),
                sensitivity,
                np.percentile(gradients, 25),
                np.percentile(gradients, 75)
            ])
        else:
            features = [0.0] * 10
        
        return np.array(features)
    
    def _forward_pass_noise_generator(
        self,
        features: np.ndarray,
        utility_target: float
    ) -> np.ndarray:
        """Forward pass through noise generator network."""
        # Add utility target to features
        x = np.concatenate([features, [utility_target]])
        
        # Ensure input dimension matches network
        if len(x) != self.noise_generator["architecture"][0]:
            # Pad or truncate to match expected input size
            target_size = self.noise_generator["architecture"][0]
            if len(x) < target_size:
                x = np.pad(x, (0, target_size - len(x)), mode='constant')
            else:
                x = x[:target_size]
        
        # Forward pass through network
        for i, (weight, bias) in enumerate(zip(self.noise_generator["weights"], self.noise_generator["biases"])):
            x = np.dot(x, weight) + bias
            
            # Activation function
            if i < len(self.noise_generator["weights"]) - 1:
                # ReLU for hidden layers
                if self.config.activation_function == "relu":
                    x = np.maximum(0, x)
                elif self.config.activation_function == "tanh":
                    x = np.tanh(x)
                else:  # sigmoid
                    x = 1.0 / (1.0 + np.exp(-x))
        
        # Output layer: noise distribution parameters
        # [mean, log_std, skewness, kurtosis]
        noise_params = x
        
        # Ensure valid parameters
        noise_params[1] = max(-5, min(5, noise_params[1]))  # Log std bounds
        noise_params[2] = max(-2, min(2, noise_params[2]))  # Skewness bounds
        noise_params[3] = max(1, min(10, noise_params[3]))  # Kurtosis bounds (>1)
        
        return noise_params
    
    def _sample_from_learned_distribution(
        self,
        params: np.ndarray,
        shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Sample noise from learned distribution."""
        mean, log_std, skewness, kurtosis = params
        std = np.exp(log_std)
        
        # Base distribution (standard normal)
        base_samples = np.random.normal(0, 1, shape)
        
        # Apply learned parameters
        # Scale and shift
        noise = base_samples * std + mean
        
        # Apply skewness transformation (simplified)
        if abs(skewness) > 0.1:
            noise = noise + skewness * (noise ** 2 - 1) / 6
        
        # Apply kurtosis adjustment (simplified)
        if abs(kurtosis - 3) > 0.1:  # 3 is normal kurtosis
            excess_kurtosis = kurtosis - 3
            noise = noise * (1 + excess_kurtosis * (noise ** 2) / 24)
        
        return noise
    
    def _apply_privacy_constraints(
        self,
        noise: np.ndarray,
        sensitivity: float
    ) -> np.ndarray:
        """Apply privacy constraints to generated noise."""
        # Minimum noise level for differential privacy
        min_noise_std = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        
        # Ensure noise magnitude meets privacy requirements
        current_noise_std = np.std(noise)
        if current_noise_std < min_noise_std:
            scaling_factor = min_noise_std / max(current_noise_std, 1e-6)
            noise = noise * scaling_factor
        
        return noise
    
    def _compute_privacy_cost(
        self,
        noise_params: np.ndarray,
        sensitivity: float
    ) -> float:
        """Compute privacy cost of generated noise."""
        mean, log_std, skewness, kurtosis = noise_params
        std = np.exp(log_std)
        
        # Privacy cost based on noise magnitude relative to sensitivity
        base_cost = sensitivity / max(std, 1e-6)
        
        # Adjust for distribution shape (non-Gaussian distributions may have different privacy guarantees)
        shape_penalty = 1.0 + 0.1 * (abs(skewness) + abs(kurtosis - 3))
        
        privacy_cost = base_cost * shape_penalty * 0.01  # Scale down for realistic values
        
        return min(privacy_cost, self.epsilon * 0.1)  # Cap at 10% of total budget
    
    def _estimate_utility_retention(
        self,
        original: np.ndarray,
        noisy: np.ndarray
    ) -> float:
        """Estimate utility retention after adding noise."""
        if len(original) == 0:
            return 1.0
        
        # Signal-to-noise ratio based utility estimate
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((noisy - original) ** 2)
        
        if noise_power == 0:
            return 1.0
        
        snr = signal_power / noise_power
        utility = min(1.0, snr / (1 + snr))  # Normalize to [0, 1]
        
        return utility
    
    def train_noise_generator(
        self,
        training_data: List[Tuple[np.ndarray, float, float]],  # (gradients, sensitivity, target_utility)
        utility_feedback: List[float],
        num_epochs: int = 10,
        learning_rate: float = 0.001
    ) -> Dict[str, float]:
        """Train neural noise generator based on utility feedback.
        
        Args:
            training_data: Training examples (gradients, sensitivity, target_utility)
            utility_feedback: Actual utility achieved for each example
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            
        Returns:
            Training metrics
        """
        if not training_data or not utility_feedback:
            return {"error": "No training data provided"}
        
        logger.info(f"Training neural noise generator with {len(training_data)} examples")
        
        epoch_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for (gradients, sensitivity, target_utility), actual_utility in zip(training_data, utility_feedback):
                # Forward pass
                gradient_features = self._extract_gradient_features(gradients, sensitivity)
                noise_params = self._forward_pass_noise_generator(gradient_features, target_utility)
                
                # Generate noise and compute utility
                noise = self._sample_from_learned_distribution(noise_params, gradients.shape)
                noise = self._apply_privacy_constraints(noise, sensitivity)
                noisy_gradients = gradients + noise
                
                predicted_utility = self._estimate_utility_retention(gradients, noisy_gradients)
                
                # Loss: difference between predicted and actual utility
                utility_loss = (predicted_utility - actual_utility) ** 2
                
                # Privacy constraint loss
                privacy_cost = self._compute_privacy_cost(noise_params, sensitivity)
                privacy_loss = max(0, privacy_cost - self.epsilon * 0.1) ** 2  # Penalty for exceeding budget
                
                total_loss = utility_loss + 0.1 * privacy_loss
                epoch_loss += total_loss
                
                # Simplified backpropagation and weight update
                self._update_noise_generator_weights(
                    gradient_features, target_utility, total_loss, learning_rate
                )
            
            avg_epoch_loss = epoch_loss / len(training_data)
            epoch_losses.append(avg_epoch_loss)
            
            logger.debug(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # Update training metadata
        self.noise_generator["training_epochs"] += num_epochs
        self.noise_generator["loss_history"].extend(epoch_losses)
        
        return {
            "training_epochs": num_epochs,
            "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
            "loss_improvement": epoch_losses[0] - epoch_losses[-1] if len(epoch_losses) > 1 else 0.0,
            "average_loss": np.mean(epoch_losses)
        }
    
    def _update_noise_generator_weights(
        self,
        features: np.ndarray,
        target_utility: float,
        loss: float,
        learning_rate: float
    ) -> None:
        """Update noise generator weights (simplified backpropagation)."""
        # Simplified weight update based on loss
        # In practice, this would be full backpropagation
        
        gradient_scale = learning_rate * loss * 0.001  # Scale down for stability
        
        # Update weights with random perturbation based on loss
        for i, (weight, bias) in enumerate(zip(self.noise_generator["weights"], self.noise_generator["biases"])):
            weight_perturbation = np.random.normal(0, gradient_scale, weight.shape)
            bias_perturbation = np.random.normal(0, gradient_scale, bias.shape)
            
            # Apply perturbation (simplified gradient step)
            self.noise_generator["weights"][i] -= weight_perturbation
            self.noise_generator["biases"][i] -= bias_perturbation


class PrivacyAwareNeuralArchitectureSearch:
    """Privacy-aware neural architecture search for optimal privacy-utility tradeoffs.
    
    This system searches for neural network architectures that provide the best
    privacy-utility tradeoffs using evolutionary algorithms and multi-objective optimization.
    """
    
    def __init__(
        self,
        search_space: Dict[str, List[Any]] = None,
        population_size: int = 20,
        num_generations: int = 10,
        privacy_weight: float = 0.5
    ):
        """Initialize privacy-aware NAS.
        
        Args:
            search_space: Architecture search space definition
            population_size: Size of architecture population
            num_generations: Number of evolutionary generations
            privacy_weight: Weight for privacy vs utility in fitness (0-1)
        """
        self.search_space = search_space or self._default_search_space()
        self.population_size = population_size
        self.num_generations = num_generations
        self.privacy_weight = privacy_weight
        
        # Evolution state
        self.population = []
        self.fitness_history = []
        self.best_architectures = []
        
        logger.info(f"Initialized Privacy-Aware NAS with population {population_size}")
    
    def _default_search_space(self) -> Dict[str, List[Any]]:
        """Define default neural architecture search space."""
        return {
            "num_layers": [2, 3, 4, 5, 6],
            "layer_sizes": [16, 32, 64, 128, 256],
            "activation_functions": ["relu", "tanh", "sigmoid"],
            "dropout_rates": [0.0, 0.1, 0.2, 0.3, 0.5],
            "batch_norm": [True, False],
            "privacy_layers": [True, False],  # Special privacy-preserving layers
            "noise_injection": ["input", "hidden", "output", "none"]
        }
    
    def search_optimal_architecture(
        self,
        dataset_info: Dict[str, Any],
        privacy_requirements: Dict[str, float],
        utility_targets: Dict[str, float]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Search for optimal privacy-aware architecture.
        
        Args:
            dataset_info: Information about the dataset
            privacy_requirements: Privacy requirements (epsilon, delta, etc.)
            utility_targets: Target utility metrics
            
        Returns:
            Tuple of (best_architecture, pareto_front_architectures)
        """
        logger.info("Starting privacy-aware architecture search")
        
        # Initialize population
        self.population = self._initialize_population()
        
        # Evolution loop
        for generation in range(self.num_generations):
            logger.debug(f"Generation {generation + 1}/{self.num_generations}")
            
            # Evaluate population fitness
            fitness_scores = self._evaluate_population_fitness(
                dataset_info, privacy_requirements, utility_targets
            )
            
            # Record generation statistics
            generation_stats = {
                "generation": generation,
                "best_fitness": max(fitness_scores),
                "avg_fitness": np.mean(fitness_scores),
                "worst_fitness": min(fitness_scores),
                "fitness_std": np.std(fitness_scores)
            }
            self.fitness_history.append(generation_stats)
            
            # Selection and reproduction
            if generation < self.num_generations - 1:
                self.population = self._evolve_population(fitness_scores)
        
        # Final evaluation and Pareto front extraction
        final_fitness = self._evaluate_population_fitness(
            dataset_info, privacy_requirements, utility_targets, detailed=True
        )
        
        pareto_front = self._extract_pareto_front(final_fitness)
        best_architecture = self._select_best_architecture(pareto_front)
        
        logger.info(f"Architecture search completed. Best fitness: {best_architecture['fitness']:.4f}")
        
        return best_architecture, pareto_front
    
    def _initialize_population(self) -> List[Dict[str, Any]]:
        """Initialize random population of architectures."""
        population = []
        
        for _ in range(self.population_size):
            architecture = {}
            
            for param, values in self.search_space.items():
                if param == "layer_sizes":
                    # Generate variable number of layers
                    num_layers = random.choice(self.search_space["num_layers"])
                    architecture[param] = [random.choice(values) for _ in range(num_layers)]
                elif param == "num_layers":
                    continue  # Handled above
                else:
                    architecture[param] = random.choice(values)
            
            # Ensure consistency
            architecture["num_layers"] = len(architecture["layer_sizes"])
            
            population.append(architecture)
        
        return population
    
    def _evaluate_population_fitness(
        self,
        dataset_info: Dict[str, Any],
        privacy_requirements: Dict[str, float],
        utility_targets: Dict[str, float],
        detailed: bool = False
    ) -> List[Dict[str, Any]] if detailed else List[float]:
        """Evaluate fitness of population architectures."""
        fitness_results = []
        
        for i, architecture in enumerate(self.population):
            # Simulate architecture evaluation
            privacy_score = self._evaluate_privacy_performance(architecture, privacy_requirements)
            utility_score = self._evaluate_utility_performance(architecture, dataset_info, utility_targets)
            
            # Combined fitness score
            fitness = self.privacy_weight * privacy_score + (1 - self.privacy_weight) * utility_score
            
            if detailed:
                result = {
                    "architecture": architecture,
                    "privacy_score": privacy_score,
                    "utility_score": utility_score,
                    "fitness": fitness,
                    "index": i
                }
                fitness_results.append(result)
            else:
                fitness_results.append(fitness)
        
        return fitness_results
    
    def _evaluate_privacy_performance(
        self,
        architecture: Dict[str, Any],
        privacy_requirements: Dict[str, float]
    ) -> float:
        """Evaluate privacy performance of architecture."""
        privacy_score = 0.0
        
        # Base privacy from architecture structure
        num_layers = architecture["num_layers"]
        
        # Deeper networks may provide better privacy through complexity
        depth_score = min(1.0, num_layers / 6.0) * 0.3
        
        # Privacy-specific features
        privacy_features_score = 0.0
        
        if architecture.get("privacy_layers", False):
            privacy_features_score += 0.3
        
        if architecture.get("noise_injection", "none") != "none":
            privacy_features_score += 0.2
        
        # Dropout provides some privacy protection
        dropout_rate = architecture.get("dropout_rates", 0.0)
        dropout_score = min(1.0, dropout_rate / 0.5) * 0.1
        
        # Layer size regularization (smaller layers = better privacy)
        avg_layer_size = np.mean(architecture["layer_sizes"])
        size_score = max(0.0, 1.0 - avg_layer_size / 512.0) * 0.1
        
        privacy_score = depth_score + privacy_features_score + dropout_score + size_score
        
        # Penalty for not meeting privacy requirements
        required_epsilon = privacy_requirements.get("epsilon", 1.0)
        if required_epsilon < 1.0:  # Strict privacy requirement
            if not architecture.get("privacy_layers", False):
                privacy_score *= 0.5  # Penalty for no privacy layers
        
        return min(1.0, privacy_score)
    
    def _evaluate_utility_performance(
        self,
        architecture: Dict[str, Any],
        dataset_info: Dict[str, Any],
        utility_targets: Dict[str, float]
    ) -> float:
        """Evaluate utility performance of architecture."""
        # Simulate model performance based on architecture
        
        # Base performance from model capacity
        num_params = self._estimate_num_parameters(architecture)
        dataset_size = dataset_info.get("num_samples", 10000)
        
        # Model capacity vs dataset size ratio
        capacity_ratio = num_params / max(dataset_size, 1)
        
        # Optimal ratio is around 0.01-0.1 (avoid overfitting and underfitting)
        if 0.01 <= capacity_ratio <= 0.1:
            capacity_score = 1.0
        elif capacity_ratio < 0.01:
            capacity_score = capacity_ratio / 0.01  # Underfitting penalty
        else:
            capacity_score = max(0.1, 0.1 / capacity_ratio)  # Overfitting penalty
        
        # Architecture-specific performance factors
        activation_performance = {
            "relu": 0.9,
            "tanh": 0.8,
            "sigmoid": 0.7
        }
        
        activation_score = activation_performance.get(
            architecture.get("activation_functions", "relu"), 0.8
        )
        
        # Batch normalization typically improves performance
        batch_norm_score = 1.1 if architecture.get("batch_norm", False) else 1.0
        
        # Dropout can help with generalization but might hurt training performance
        dropout_rate = architecture.get("dropout_rates", 0.0)
        dropout_score = max(0.8, 1.0 - dropout_rate)
        
        # Privacy features may hurt utility
        privacy_penalty = 1.0
        if architecture.get("privacy_layers", False):
            privacy_penalty *= 0.9
        if architecture.get("noise_injection", "none") != "none":
            privacy_penalty *= 0.85
        
        # Combine all factors
        utility_score = (capacity_score * activation_score * 
                        batch_norm_score * dropout_score * privacy_penalty)
        
        # Add some randomness to simulate evaluation noise
        noise = np.random.normal(0, 0.05)
        utility_score = max(0.0, min(1.0, utility_score + noise))
        
        return utility_score
    
    def _estimate_num_parameters(self, architecture: Dict[str, Any]) -> int:
        """Estimate number of parameters in architecture."""
        layer_sizes = architecture["layer_sizes"]
        
        if not layer_sizes:
            return 0
        
        # Assume input dimension (can be provided in dataset_info)
        input_dim = 100  # Default assumption
        
        num_params = 0
        prev_size = input_dim
        
        for layer_size in layer_sizes:
            # Weight parameters: prev_size * layer_size
            # Bias parameters: layer_size
            num_params += prev_size * layer_size + layer_size
            prev_size = layer_size
        
        # Output layer (assume classification with 10 classes)
        output_dim = 10
        num_params += prev_size * output_dim + output_dim
        
        return num_params
    
    def _evolve_population(self, fitness_scores: List[float]) -> List[Dict[str, Any]]:
        """Evolve population using genetic algorithm."""
        # Tournament selection
        selected = self._tournament_selection(fitness_scores, self.population_size)
        
        # Crossover and mutation
        new_population = []
        
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            
            # Crossover
            child1, child2 = self._crossover(parent1, parent2)
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to population size
        return new_population[:self.population_size]
    
    def _tournament_selection(self, fitness_scores: List[float], num_selected: int) -> List[Dict[str, Any]]:
        """Tournament selection for genetic algorithm."""
        selected = []
        
        for _ in range(num_selected):
            # Tournament size of 3
            tournament_indices = random.sample(range(len(fitness_scores)), min(3, len(fitness_scores)))
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select best from tournament
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(self.population[winner_idx].copy())
        
        return selected
    
    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover operation for genetic algorithm."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Single-point crossover for each parameter
        for param in self.search_space.keys():
            if param == "layer_sizes":
                # Special handling for variable-length lists
                min_len = min(len(parent1[param]), len(parent2[param]))
                if min_len > 1:
                    crossover_point = random.randint(1, min_len - 1)
                    
                    child1[param] = parent1[param][:crossover_point] + parent2[param][crossover_point:]
                    child2[param] = parent2[param][:crossover_point] + parent1[param][crossover_point:]
            else:
                # Random selection from parents
                if random.random() < 0.5:
                    child1[param] = parent2[param]
                    child2[param] = parent1[param]
        
        # Update num_layers to match layer_sizes
        child1["num_layers"] = len(child1["layer_sizes"])
        child2["num_layers"] = len(child2["layer_sizes"])
        
        return child1, child2
    
    def _mutate(self, architecture: Dict[str, Any], mutation_rate: float = 0.1) -> Dict[str, Any]:
        """Mutation operation for genetic algorithm."""
        mutated = architecture.copy()
        
        for param, values in self.search_space.items():
            if random.random() < mutation_rate:
                if param == "layer_sizes":
                    # Mutate layer sizes
                    if random.random() < 0.5:
                        # Change a random layer size
                        if mutated[param]:
                            idx = random.randint(0, len(mutated[param]) - 1)
                            mutated[param][idx] = random.choice(values)
                    else:
                        # Add or remove a layer
                        if len(mutated[param]) > 1 and random.random() < 0.5:
                            # Remove a layer
                            idx = random.randint(0, len(mutated[param]) - 1)
                            mutated[param].pop(idx)
                        else:
                            # Add a layer
                            new_size = random.choice(values)
                            idx = random.randint(0, len(mutated[param]))
                            mutated[param].insert(idx, new_size)
                elif param != "num_layers":
                    mutated[param] = random.choice(values)
        
        # Update num_layers
        mutated["num_layers"] = len(mutated["layer_sizes"])
        
        return mutated
    
    def _extract_pareto_front(self, detailed_fitness: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract Pareto front from population."""
        pareto_front = []
        
        for i, candidate in enumerate(detailed_fitness):
            is_dominated = False
            
            for j, other in enumerate(detailed_fitness):
                if i != j:
                    # Check if other dominates candidate
                    if (other["privacy_score"] >= candidate["privacy_score"] and
                        other["utility_score"] >= candidate["utility_score"] and
                        (other["privacy_score"] > candidate["privacy_score"] or
                         other["utility_score"] > candidate["utility_score"])):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        return pareto_front
    
    def _select_best_architecture(self, pareto_front: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best architecture from Pareto front."""
        if not pareto_front:
            return {"error": "No architectures in Pareto front"}
        
        # Select architecture with highest combined fitness
        best = max(pareto_front, key=lambda x: x["fitness"])
        
        return best


def create_ml_enhanced_privacy_system(
    num_components: int = 5,
    privacy_budget: float = 1.0,
    adversary_strength: float = 0.8
) -> Dict[str, Any]:
    """Create comprehensive ML-enhanced privacy system.
    
    Factory function to initialize all ML-enhanced privacy components.
    
    Args:
        num_components: Number of components for budget allocation
        privacy_budget: Total privacy budget
        adversary_strength: Strength of adversarial attacks
        
    Returns:
        Dictionary containing all ML-enhanced privacy components
    """
    logger.info("Creating ML-enhanced privacy system")
    
    return {
        "adversarial_predictor": AdversarialPrivacyPredictor(
            adversary_strength=adversary_strength
        ),
        "rl_budget_allocator": ReinforcementLearningBudgetAllocator(
            num_components=num_components,
            total_budget=privacy_budget
        ),
        "neural_dp": NeuralDifferentialPrivacy(
            config=NeuralPrivacyConfig(),
            epsilon=privacy_budget
        ),
        "privacy_nas": PrivacyAwareNeuralArchitectureSearch(
            population_size=20,
            num_generations=10
        )
    }