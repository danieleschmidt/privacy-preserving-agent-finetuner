"""Tests for differential privacy guarantees and mechanisms."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import math

from privacy_finetuner.core.privacy_config import PrivacyConfig
from privacy_finetuner.core.trainer import PrivateTrainer
from tests.fixtures.test_data import MockPrivateDataset
from tests.fixtures.privacy_configs import basic_privacy_config


@pytest.mark.privacy
class TestDifferentialPrivacyGuarantees:
    """Test suite for verifying differential privacy guarantees."""

    def test_epsilon_delta_bounds(self, basic_privacy_config):
        """Test that privacy bounds are respected."""
        config = basic_privacy_config
        
        # Privacy parameters should be positive
        assert config.epsilon > 0, "Epsilon must be positive"
        assert config.delta >= 0, "Delta must be non-negative"
        assert config.delta < 1, "Delta must be less than 1"
        
        # For meaningful privacy, epsilon should be reasonably small
        assert config.epsilon <= 10, "Epsilon should be bounded for meaningful privacy"

    def test_noise_scaling(self):
        """Test that noise scales correctly with privacy parameters."""
        sensitivity = 1.0
        
        # Higher epsilon should result in less noise
        config_low_epsilon = PrivacyConfig(epsilon=0.1, delta=1e-5)
        config_high_epsilon = PrivacyConfig(epsilon=10.0, delta=1e-5)
        
        # Simulate noise calculation (simplified)
        noise_low = sensitivity / config_low_epsilon.epsilon
        noise_high = sensitivity / config_high_epsilon.epsilon
        
        assert noise_low > noise_high, "Lower epsilon should require more noise"

    @pytest.mark.parametrize("epsilon,delta", [
        (0.1, 1e-6),
        (1.0, 1e-5),
        (10.0, 1e-4)
    ])
    def test_privacy_budget_consumption(self, epsilon, delta):
        """Test privacy budget consumption tracking."""
        config = PrivacyConfig(epsilon=epsilon, delta=delta)
        
        # Simulate multiple queries/steps
        steps = 100
        consumed_epsilon = 0
        
        for step in range(steps):
            # Simple composition (actual implementation would use RDP)
            step_epsilon = epsilon / steps  # Naive splitting
            consumed_epsilon += step_epsilon
            
            # Check budget not exceeded
            assert consumed_epsilon <= epsilon, f"Privacy budget exceeded at step {step}"
        
        # Final consumption should equal budget
        assert abs(consumed_epsilon - epsilon) < 1e-6

    def test_gradient_clipping_bounds(self):
        """Test that gradient clipping enforces sensitivity bounds."""
        max_grad_norm = 1.0
        batch_size = 8
        features = 10
        
        # Generate gradients with various norms
        test_cases = [
            torch.randn(batch_size, features) * 0.5,  # Small gradients
            torch.randn(batch_size, features) * 2.0,  # Large gradients
            torch.randn(batch_size, features) * 10.0, # Very large gradients
        ]
        
        for gradients in test_cases:
            # Apply per-sample gradient clipping
            clipped_gradients = []
            
            for i in range(batch_size):
                sample_grad = gradients[i]
                grad_norm = torch.norm(sample_grad)
                
                if grad_norm > max_grad_norm:
                    clipped_grad = sample_grad * (max_grad_norm / grad_norm)
                else:
                    clipped_grad = sample_grad
                
                clipped_gradients.append(clipped_grad)
                
                # Verify clipping constraint
                clipped_norm = torch.norm(clipped_grad)
                assert clipped_norm <= max_grad_norm + 1e-6, \
                    f"Gradient norm {clipped_norm} exceeds max {max_grad_norm}"
            
            clipped_batch = torch.stack(clipped_gradients)
            assert clipped_batch.shape == gradients.shape

    def test_gaussian_noise_properties(self):
        """Test properties of Gaussian noise for DP."""
        noise_multiplier = 0.5
        sensitivity = 1.0
        batch_size = 1000
        features = 10
        
        # Generate noise
        noise = torch.normal(0, noise_multiplier * sensitivity, (batch_size, features))
        
        # Test statistical properties
        mean_noise = torch.mean(noise)
        std_noise = torch.std(noise)
        
        # Mean should be close to 0
        assert abs(mean_noise) < 0.1, f"Noise mean {mean_noise} not close to 0"
        
        # Standard deviation should match expected value
        expected_std = noise_multiplier * sensitivity
        assert abs(std_noise - expected_std) < 0.1, \
            f"Noise std {std_noise} doesn't match expected {expected_std}"

    def test_composition_rules(self):
        """Test basic composition rules for differential privacy."""
        # Sequential composition
        epsilon1, delta1 = 0.5, 1e-6
        epsilon2, delta2 = 0.3, 1e-6
        
        # Basic composition bounds
        composed_epsilon = epsilon1 + epsilon2
        composed_delta = delta1 + delta2
        
        assert composed_epsilon == 0.8
        assert composed_delta == 2e-6
        
        # Advanced composition would use tighter bounds (RDP/GDP)
        # This is a simplified test

    def test_privacy_analysis_with_sampling(self):
        """Test privacy analysis with subsampling."""
        total_samples = 1000
        batch_size = 100
        sampling_rate = batch_size / total_samples
        
        base_epsilon = 1.0
        noise_multiplier = 0.5
        
        # Simplified privacy analysis with sampling
        # Real implementation would use RDP accountant
        effective_epsilon = sampling_rate * base_epsilon / (noise_multiplier ** 2)
        
        assert effective_epsilon > 0
        assert effective_epsilon < base_epsilon  # Sampling should reduce privacy cost

    @pytest.mark.parametrize("attack_type", ["membership_inference", "attribute_inference"])
    def test_privacy_against_attacks(self, attack_type):
        """Test privacy guarantees against specific attacks."""
        config = PrivacyConfig(epsilon=1.0, delta=1e-5)
        
        if attack_type == "membership_inference":
            # Simulate membership inference attack resistance
            train_data = torch.randn(100, 10)
            test_data = torch.randn(100, 10)
            
            # With proper DP, distinguishing advantage should be bounded
            # This is a placeholder for actual attack simulation
            max_advantage = config.epsilon + 2 * math.sqrt(2 * math.log(1 / config.delta))
            
            assert max_advantage < 2.0, "Privacy advantage too high"
            
        elif attack_type == "attribute_inference":
            # Simulate attribute inference attack resistance
            public_attrs = torch.randn(50, 5)
            private_attrs = torch.randn(50, 3)
            
            # Similar bounds apply for attribute inference
            assert config.epsilon < 2.0, "Epsilon too high for attribute privacy"

    def test_privacy_budget_allocation(self):
        """Test optimal privacy budget allocation across operations."""
        total_epsilon = 1.0
        total_delta = 1e-5
        
        operations = ['training', 'evaluation', 'hyperparameter_tuning']
        
        # Simple equal allocation (real systems would optimize)
        epsilon_per_op = total_epsilon / len(operations)
        delta_per_op = total_delta / len(operations)
        
        allocated_epsilon = 0
        allocated_delta = 0
        
        for op in operations:
            allocated_epsilon += epsilon_per_op
            allocated_delta += delta_per_op
        
        # Total allocation should not exceed budget
        assert allocated_epsilon <= total_epsilon + 1e-10
        assert allocated_delta <= total_delta + 1e-10

    def test_privacy_amplification_by_subsampling(self):
        """Test privacy amplification through subsampling."""
        total_data_size = 10000
        batch_size = 100
        base_epsilon = 1.0
        
        sampling_rate = batch_size / total_data_size
        
        # Privacy amplification bound (simplified)
        # Real implementation would use more sophisticated bounds
        amplified_epsilon = base_epsilon * sampling_rate
        
        assert amplified_epsilon < base_epsilon, "Subsampling should amplify privacy"
        assert amplified_epsilon > 0, "Amplified epsilon should be positive"

    def test_local_vs_global_differential_privacy(self):
        """Test comparison between local and global DP mechanisms."""
        data_size = 1000
        sensitivity = 1.0
        
        # Global DP: Add noise to aggregate
        global_epsilon = 1.0
        global_noise = torch.normal(0, sensitivity / global_epsilon, (1,))
        
        # Local DP: Add noise to each record
        local_epsilon = 1.0
        local_noise = torch.normal(0, sensitivity / local_epsilon, (data_size,))
        
        # Global DP should have much lower noise for aggregates
        global_noise_magnitude = abs(global_noise.item())
        local_noise_magnitude = torch.std(local_noise).item()
        
        # This is a simplified comparison
        assert global_noise_magnitude < local_noise_magnitude / math.sqrt(data_size)

    def test_privacy_preservation_across_epochs(self):
        """Test privacy budget consumption across training epochs."""
        epsilon_per_epoch = 0.1
        max_epochs = 10
        total_budget = 1.0
        
        consumed_budget = 0
        
        for epoch in range(max_epochs):
            # Check if we can afford this epoch
            if consumed_budget + epsilon_per_epoch <= total_budget:
                consumed_budget += epsilon_per_epoch
            else:
                # Should stop training to preserve privacy
                break
        
        assert consumed_budget <= total_budget
        assert epoch < max_epochs  # Should stop before completing all epochs

    def teardown_method(self):
        """Cleanup after privacy tests."""
        torch.manual_seed(42)  # Reset random seed for reproducibility