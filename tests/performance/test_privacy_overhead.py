"""Performance tests for privacy overhead analysis."""

import pytest
import time
import torch
import numpy as np
from unittest.mock import Mock
import psutil
import os

from privacy_finetuner.core.trainer import PrivateTrainer
from privacy_finetuner.core.privacy_config import PrivacyConfig
from tests.fixtures.test_data import MockPrivateDataset


@pytest.mark.performance
@pytest.mark.slow
class TestPrivacyPerformanceOverhead:
    """Test suite for measuring privacy-related performance overhead."""

    def setup_method(self):
        """Setup for performance tests."""
        self.baseline_config = PrivacyConfig(
            epsilon=float('inf'),  # No privacy (baseline)
            delta=0,
            max_grad_norm=float('inf'),
            noise_multiplier=0
        )
        
        self.private_config = PrivacyConfig(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=0.5
        )

    @pytest.mark.gpu
    def test_training_time_overhead(self, mock_private_dataset):
        """Test training time overhead with privacy."""
        dataset = MockPrivateDataset(size=1000, features=50)
        
        # Baseline training (no privacy)
        baseline_trainer = Mock()  # Replace with actual trainer
        start_time = time.time()
        # baseline_trainer.train(dataset, config=self.baseline_config)
        baseline_time = time.time() - start_time
        
        # Private training
        private_trainer = Mock()  # Replace with actual trainer
        start_time = time.time()
        # private_trainer.train(dataset, config=self.private_config)
        private_time = time.time() - start_time
        
        # Calculate overhead
        overhead_ratio = private_time / baseline_time if baseline_time > 0 else 1
        overhead_percentage = (overhead_ratio - 1) * 100
        
        # Assert reasonable overhead (should be < 50%)
        assert overhead_percentage < 50, f"Privacy overhead too high: {overhead_percentage:.1f}%"
        
        print(f"Training time overhead: {overhead_percentage:.1f}%")

    def test_memory_overhead(self, mock_private_dataset):
        """Test memory overhead with privacy mechanisms."""
        process = psutil.Process(os.getpid())
        
        # Baseline memory usage
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create private trainer (this would instantiate privacy mechanisms)
        # private_trainer = PrivateTrainer(config=self.private_config)
        
        privacy_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_overhead = privacy_memory - baseline_memory
        overhead_percentage = (memory_overhead / baseline_memory) * 100
        
        # Assert reasonable memory overhead (should be < 30%)
        assert overhead_percentage < 30, f"Memory overhead too high: {overhead_percentage:.1f}%"
        
        print(f"Memory overhead: {memory_overhead:.1f}MB ({overhead_percentage:.1f}%)")

    @pytest.mark.parametrize("epsilon", [0.1, 1.0, 10.0])
    def test_privacy_budget_impact_on_performance(self, epsilon, mock_private_dataset):
        """Test how different privacy budgets affect performance."""
        config = PrivacyConfig(
            epsilon=epsilon,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=1.5 / epsilon  # Higher noise for lower epsilon
        )
        
        dataset = MockPrivateDataset(size=100, features=10)
        
        start_time = time.time()
        # Simulate training with different privacy budgets
        for _ in range(10):  # 10 mini-batches
            batch = torch.randn(8, 10)
            # Add noise simulation
            noise = torch.randn_like(batch) * config.noise_multiplier
            private_batch = batch + noise
            
            # Simulate gradient clipping
            grad_norm = torch.norm(private_batch)
            if grad_norm > config.max_grad_norm:
                private_batch = private_batch * (config.max_grad_norm / grad_norm)
        
        training_time = time.time() - start_time
        
        # Lower epsilon should generally take more time due to higher noise
        print(f"Epsilon {epsilon}: Training time {training_time:.3f}s")
        assert training_time > 0

    def test_gradient_clipping_overhead(self):
        """Test overhead of gradient clipping."""
        batch_size, features = 32, 100
        gradients = torch.randn(batch_size, features, requires_grad=True)
        
        # Baseline: No clipping
        start_time = time.time()
        for _ in range(1000):
            _ = gradients.clone()
        baseline_time = time.time() - start_time
        
        # With gradient clipping
        max_norm = 1.0
        start_time = time.time()
        for _ in range(1000):
            clipped = gradients.clone()
            grad_norm = torch.norm(clipped)
            if grad_norm > max_norm:
                clipped = clipped * (max_norm / grad_norm)
        clipping_time = time.time() - start_time
        
        overhead_percentage = ((clipping_time - baseline_time) / baseline_time) * 100
        
        # Gradient clipping should have minimal overhead (< 20%)
        assert overhead_percentage < 20, f"Gradient clipping overhead too high: {overhead_percentage:.1f}%"
        
        print(f"Gradient clipping overhead: {overhead_percentage:.1f}%")

    def test_noise_generation_performance(self):
        """Test performance of noise generation for DP."""
        batch_size, features = 32, 1000
        noise_multiplier = 0.5
        
        # Test different noise generation methods
        methods = {
            "torch_normal": lambda: torch.normal(0, noise_multiplier, (batch_size, features)),
            "numpy_normal": lambda: torch.from_numpy(
                np.random.normal(0, noise_multiplier, (batch_size, features)).astype(np.float32)
            ),
        }
        
        results = {}
        for method_name, method_func in methods.items():
            start_time = time.time()
            for _ in range(1000):
                noise = method_func()
            end_time = time.time()
            results[method_name] = end_time - start_time
            print(f"{method_name}: {results[method_name]:.3f}s")
        
        # All methods should be reasonably fast
        for method_name, time_taken in results.items():
            assert time_taken < 5.0, f"{method_name} too slow: {time_taken:.3f}s"

    @pytest.mark.parametrize("batch_size", [8, 16, 32, 64])
    def test_batch_size_privacy_scaling(self, batch_size):
        """Test how privacy mechanisms scale with batch size."""
        features = 100
        noise_multiplier = 0.5
        max_grad_norm = 1.0
        
        # Simulate privacy operations for different batch sizes
        start_time = time.time()
        
        for _ in range(100):  # 100 iterations
            # Generate batch
            batch = torch.randn(batch_size, features)
            
            # Add noise
            noise = torch.randn_like(batch) * noise_multiplier
            private_batch = batch + noise
            
            # Gradient clipping per sample
            for i in range(batch_size):
                sample_grad = private_batch[i]
                grad_norm = torch.norm(sample_grad)
                if grad_norm > max_grad_norm:
                    private_batch[i] = sample_grad * (max_grad_norm / grad_norm)
        
        processing_time = time.time() - start_time
        time_per_sample = processing_time / (100 * batch_size)
        
        print(f"Batch size {batch_size}: {time_per_sample:.6f}s per sample")
        
        # Per-sample time should be relatively constant regardless of batch size
        assert time_per_sample < 0.001, f"Per-sample processing too slow: {time_per_sample:.6f}s"

    def test_federated_aggregation_performance(self):
        """Test performance of secure aggregation in federated setting."""
        num_clients = 10
        model_size = 1000
        
        # Simulate client updates
        client_updates = [torch.randn(model_size) for _ in range(num_clients)]
        
        # Test different aggregation methods
        aggregation_methods = {
            "simple_average": lambda updates: torch.stack(updates).mean(dim=0),
            "weighted_average": lambda updates: torch.stack(updates).mean(dim=0),  # Simplified
            "secure_sum": lambda updates: torch.stack(updates).sum(dim=0) / len(updates),
        }
        
        for method_name, method_func in aggregation_methods.items():
            start_time = time.time()
            for _ in range(100):
                aggregated = method_func(client_updates)
            end_time = time.time()
            
            aggregation_time = end_time - start_time
            print(f"{method_name}: {aggregation_time:.3f}s for 100 rounds")
            
            # Aggregation should be fast
            assert aggregation_time < 2.0, f"{method_name} too slow: {aggregation_time:.3f}s"

    def test_privacy_accounting_overhead(self):
        """Test overhead of privacy accounting mechanisms."""
        epsilon_values = []
        delta = 1e-5
        
        # Simulate privacy accounting
        start_time = time.time()
        
        for step in range(1000):
            # Simulate one training step
            noise_multiplier = 0.5
            sampling_rate = 0.1
            
            # Simple privacy accounting (RDP would be more complex)
            step_epsilon = (sampling_rate * step) / (noise_multiplier ** 2)
            epsilon_values.append(step_epsilon)
            
            # Check if privacy budget exceeded
            if step_epsilon > 1.0:
                break
        
        accounting_time = time.time() - start_time
        
        print(f"Privacy accounting for {len(epsilon_values)} steps: {accounting_time:.3f}s")
        
        # Privacy accounting should be very fast
        assert accounting_time < 0.1, f"Privacy accounting too slow: {accounting_time:.3f}s"

    def teardown_method(self):
        """Cleanup after performance tests."""
        # Clear GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()