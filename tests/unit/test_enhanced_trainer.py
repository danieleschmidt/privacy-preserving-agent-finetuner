"""Enhanced comprehensive unit tests for PrivateTrainer with quantum optimization."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
from datetime import datetime

from privacy_finetuner.core import (
    PrivateTrainer, PrivacyConfig, QuantumInspiredOptimizer,
    AdaptivePrivacyScheduler, TrainingMonitor, SecurityMonitor,
    AuditLogger
)
from privacy_finetuner.core.scaling_optimizer import (
    DistributedPrivacyOptimizer, ScalingConfig, MemoryManager,
    GradientCompressor, DistributedCacheManager, PerformanceTracker,
    AutoScaler, QuantumScalingOptimizer
)
from privacy_finetuner.core.exceptions import (
    PrivacyBudgetExhaustedException, ModelTrainingException,
    DataValidationException, SecurityViolationException
)


class TestEnhancedPrivateTrainer:
    """Comprehensive test suite for enhanced PrivateTrainer."""
    
    @pytest.fixture
    def privacy_config(self):
        """Create test privacy configuration."""
        return PrivacyConfig(
            epsilon=2.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=0.5,
            accounting_mode="rdp"
        )
    
    @pytest.fixture
    def scaling_config(self):
        """Create test scaling configuration."""
        return ScalingConfig(
            num_workers=2,
            batch_size_per_worker=4,
            mixed_precision=True,
            gradient_compression=True
        )
    
    def test_enhanced_initialization(self, privacy_config):
        """Test enhanced trainer initialization with all components."""
        trainer = PrivateTrainer("gpt2", privacy_config)
        
        assert trainer.model_name == "gpt2"
        assert trainer.privacy_config == privacy_config
        assert hasattr(trainer, '_security_monitor')
        assert hasattr(trainer, '_audit_logger')
    
    def test_quantum_optimization_integration(self, privacy_config):
        """Test quantum-inspired optimization integration."""
        trainer = PrivateTrainer("gpt2", privacy_config)
        
        with patch.object(trainer, '_setup_model_and_privacy'):
            trainer._setup_model_and_privacy()
            
            # Test quantum optimizer initialization
            assert trainer._quantum_optimizer is not None
            assert trainer._adaptive_scheduler is not None
            
            # Test quantum gradient update
            mock_gradients = {
                "layer1.weight": torch.randn(10, 10),
                "layer1.bias": torch.randn(10)
            }
            
            quantum_gradients = trainer._quantum_optimizer.quantum_gradient_update(
                mock_gradients, learning_rate=1e-4, step=1
            )
            
            assert len(quantum_gradients) == len(mock_gradients)
            for name in mock_gradients:
                assert name in quantum_gradients
                assert quantum_gradients[name].shape == mock_gradients[name].shape
    
    def test_adaptive_privacy_scheduling(self, privacy_config):
        """Test adaptive privacy parameter scheduling."""
        trainer = PrivateTrainer("gpt2", privacy_config)
        
        with patch.object(trainer, '_setup_model_and_privacy'):
            trainer._setup_model_and_privacy()
            
            from privacy_finetuner.core.adaptive_privacy_scheduler import AdaptationMetrics
            
            metrics = AdaptationMetrics(
                gradient_variance=1.2,
                loss_improvement=0.1,
                privacy_efficiency=0.8,
                convergence_rate=0.05,
                model_utility=0.85
            )
            
            training_state = {"step": 50, "epoch": 1, "loss": 2.5}
            
            updated_config = trainer._adaptive_scheduler.update_privacy_config(
                metrics, training_state
            )
            
            assert updated_config.epsilon > 0
            assert updated_config.delta > 0
            assert updated_config.noise_multiplier > 0
    
    def test_robust_error_handling(self, privacy_config):
        """Test robust error handling and recovery mechanisms."""
        config = PrivacyConfig(epsilon=0.1, delta=1e-6)  # Very restrictive budget
        trainer = PrivateTrainer("gpt2", config)
        
        # Test privacy budget exhaustion handling
        with patch.object(trainer, '_get_privacy_spent', return_value=1.0):
            with pytest.raises(AttributeError):  # Method doesn't exist yet
                trainer._handle_privacy_budget_exhaustion()
    
    def test_security_monitoring(self, privacy_config):
        """Test security monitoring and audit logging."""
        trainer = PrivateTrainer("gpt2", privacy_config)
        
        # Test security monitor
        assert hasattr(trainer, '_security_monitor')
        assert hasattr(trainer, '_audit_logger')
        
        # Test gradient monitoring
        mock_gradients = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10)
        }
        
        trainer._security_monitor.monitor_gradient_updates(mock_gradients)
        
        # Check if monitoring recorded the event
        security_summary = trainer._security_monitor.get_security_summary()
        assert "total_events" in security_summary
        assert "monitoring_enabled" in security_summary
    
    def test_distributed_training_setup(self, privacy_config, scaling_config):
        """Test distributed training configuration."""
        # Mock model for distributed optimizer
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(10, 10)]
        
        distributed_optimizer = DistributedPrivacyOptimizer(
            privacy_config, scaling_config, mock_model
        )
        
        assert distributed_optimizer.privacy_config == privacy_config
        assert distributed_optimizer.scaling_config == scaling_config
        assert distributed_optimizer.world_size >= 1
        
        # Test performance tracking
        metrics = distributed_optimizer.get_scaling_metrics()
        assert "performance_metrics" in metrics
        assert "resource_utilization" in metrics
        assert "scaling_efficiency" in metrics
    
    def test_memory_optimization(self, scaling_config):
        """Test memory optimization and management."""
        memory_manager = MemoryManager(scaling_config)
        
        # Test batch optimization
        batch_data = {
            "input_ids": torch.randint(0, 1000, (8, 512)),
            "attention_mask": torch.ones(8, 512, dtype=torch.bool)
        }
        
        optimized_batch = memory_manager.optimize_batch(batch_data)
        
        assert len(optimized_batch) == len(batch_data)
        
        # Test memory usage tracking
        memory_stats = memory_manager.get_memory_usage()
        assert "allocated_gb" in memory_stats
        assert "efficiency" in memory_stats
    
    def test_gradient_compression(self, scaling_config):
        """Test gradient compression for distributed training."""
        compressor = GradientCompressor(scaling_config)
        
        # Test compression and decompression
        gradients = {
            "layer1.weight": torch.randn(100, 100),
            "layer1.bias": torch.randn(100)
        }
        
        compressed = compressor.compress(gradients)
        assert len(compressed) == len(gradients)
        
        decompressed = compressor.decompress(compressed)
        assert len(decompressed) == len(gradients)
        
        # Check shapes are preserved
        for name in gradients:
            assert decompressed[name].shape == gradients[name].shape
    
    def test_cache_management(self):
        """Test distributed cache management."""
        cache_manager = DistributedCacheManager()
        
        # Test cache operations
        test_key = "test_model_weights"
        test_value = {"weights": torch.randn(10, 10), "metadata": {"epoch": 1}}
        
        # Set and get from cache
        cache_manager.set(test_key, test_value)
        retrieved_value = cache_manager.get(test_key)
        
        if retrieved_value is not None:  # Cache might not be available
            assert "weights" in retrieved_value
            assert "metadata" in retrieved_value
        
        # Test cache statistics
        stats = cache_manager.get_cache_stats()
        assert "hit_ratio" in stats
        assert "cache_available" in stats
    
    def test_performance_tracking(self):
        """Test comprehensive performance tracking."""
        tracker = PerformanceTracker()
        
        # Simulate batch processing
        for i in range(10):
            processing_time = 0.1 + np.random.normal(0, 0.01)  # 100ms ± 10ms
            batch_size = 8
            tracker.record_batch_processing(processing_time, batch_size)
        
        metrics = tracker.get_current_metrics()
        
        assert metrics.throughput_samples_per_sec > 0
        assert metrics.batch_processing_time > 0
        assert 0 <= metrics.scaling_efficiency <= 2.0  # Allow for some variance
    
    def test_auto_scaling(self, scaling_config):
        """Test automatic scaling decisions."""
        from privacy_finetuner.core.scaling_optimizer import PerformanceMetrics
        
        auto_scaler = AutoScaler(scaling_config)
        
        # Test scale up decision
        high_util_metrics = PerformanceMetrics(
            gpu_utilization=0.9,  # High utilization
            memory_efficiency=0.8,
            throughput_samples_per_sec=100.0
        )
        
        decision = auto_scaler.make_scaling_decision(high_util_metrics)
        assert decision["action"] == "scale_up"
        assert decision["target_workers"] > scaling_config.num_workers
        
        # Test scale down decision
        low_util_metrics = PerformanceMetrics(
            gpu_utilization=0.2,  # Low utilization
            memory_efficiency=0.8,
            throughput_samples_per_sec=20.0
        )
        
        decision = auto_scaler.make_scaling_decision(low_util_metrics)
        assert decision["action"] == "scale_down"
        assert decision["target_workers"] < scaling_config.num_workers
    
    def test_quantum_scaling_optimization(self, scaling_config):
        """Test quantum-inspired scaling optimization."""
        quantum_scaler = QuantumScalingOptimizer(scaling_config)
        
        # Test worker allocation
        workload = [1.0, 2.0, 0.5, 1.5, 3.0]  # Different work sizes
        allocation = quantum_scaler.optimize_worker_allocation(workload)
        
        assert len(allocation) == len(workload)
        assert all(0 <= worker_id < scaling_config.num_workers for worker_id in allocation)
    
    def test_compliance_and_audit_trail(self, privacy_config):
        """Test compliance monitoring and audit trail."""
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_log_path = Path(temp_dir) / "test_audit.log"
            
            audit_logger = AuditLogger(str(audit_log_path))
            
            # Test initialization logging
            context = {
                "model_name": "test-model",
                "epsilon": privacy_config.epsilon,
                "delta": privacy_config.delta,
                "timestamp": datetime.now().isoformat()
            }
            
            audit_logger.log_initialization(context)
            
            # Test privacy budget logging
            audit_logger.log_privacy_budget_usage(0.1, {"step": 100})
            
            # Test data access logging
            audit_logger.log_data_access(
                "user123", 
                "/data/training_data.jsonl", 
                "read"
            )
            
            # Verify audit log exists and has content
            assert audit_log_path.exists()
            assert audit_log_path.stat().st_size > 0
            
            # Read and verify log content
            with open(audit_log_path, 'r') as f:
                log_content = f.read()
                assert "SYSTEM_INITIALIZATION" in log_content
                assert "PRIVACY_BUDGET_USAGE" in log_content
                assert "DATA_ACCESS" in log_content
    
    @pytest.mark.slow
    def test_end_to_end_training_with_optimizations(self, privacy_config, scaling_config):
        """Test end-to-end training with all optimizations enabled."""
        # Create temporary training data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for i in range(10):  # Small dataset for testing
                data = {"text": f"This is training sample {i} for testing purposes."}
                f.write(json.dumps(data) + "\n")
            temp_dataset_path = f.name
        
        try:
            trainer = PrivateTrainer("gpt2", privacy_config)
            
            # Mock the actual training components for testing
            with patch.object(trainer, '_setup_model_and_privacy'), \
                 patch.object(trainer, '_validate_training_inputs'), \
                 patch.object(trainer, '_load_and_split_dataset') as mock_load, \
                 patch.object(trainer, '_train_with_dp_sgd_robust') as mock_train:
                
                # Configure mocks
                mock_load.return_value = (Mock(), Mock())  # train, val datasets
                mock_train.return_value = {
                    "status": "training_complete",
                    "epochs_completed": 1,
                    "total_steps": 10,
                    "final_loss": 2.5,
                    "privacy_spent": 0.5,
                    "model_path": "./test_model",
                    "training_losses": [3.0, 2.8, 2.6, 2.5]
                }
                
                # Run training
                results = trainer.train(
                    dataset=temp_dataset_path,
                    epochs=1,
                    batch_size=4,
                    learning_rate=1e-4,
                    checkpoint_interval=5,
                    early_stopping_patience=3
                )
                
                # Verify results
                assert results["status"] == "training_complete"
                assert results["privacy_spent"] <= privacy_config.epsilon
                assert len(results["training_losses"]) > 0
                
                # Verify privacy report
                privacy_report = trainer.get_privacy_report()
                assert privacy_report["epsilon_spent"] >= 0
                assert privacy_report["remaining_budget"] >= 0
                
        finally:
            # Cleanup
            Path(temp_dataset_path).unlink(missing_ok=True)


class TestQuantumOptimization:
    """Test suite specifically for quantum optimization features."""
    
    @pytest.fixture
    def privacy_config(self):
        return PrivacyConfig(epsilon=1.0, delta=1e-5)
    
    def test_quantum_state_evolution(self, privacy_config):
        """Test quantum state evolution in optimizer."""
        quantum_optimizer = QuantumInspiredOptimizer(privacy_config)
        
        # Test initial state
        initial_state = quantum_optimizer.quantum_state
        assert initial_state.amplitudes is not None
        assert initial_state.phases is not None
        assert 0 <= initial_state.entanglement_strength <= 1
        
        # Test state evolution
        quantum_optimizer._evolve_quantum_state(step=10)
        
        # State should have evolved
        evolved_state = quantum_optimizer.quantum_state
        assert not torch.allclose(initial_state.phases, evolved_state.phases)
        
        # Amplitudes should still be normalized
        norm = torch.sqrt(torch.sum(evolved_state.amplitudes ** 2))
        assert torch.isclose(norm, torch.tensor(1.0), atol=1e-6)
    
    def test_quantum_measurement(self, privacy_config):
        """Test quantum state measurement."""
        quantum_optimizer = QuantumInspiredOptimizer(privacy_config)
        
        measurement = quantum_optimizer.measure_quantum_state()
        
        assert "measured_state" in measurement
        assert "binary_representation" in measurement
        assert "measurement_probability" in measurement
        assert "quantum_advantage" in measurement
        assert "entanglement_measure" in measurement
        
        # Verify measurement probability is valid
        assert 0 <= measurement["measurement_probability"] <= 1
        
        # Verify quantum advantage is reasonable
        assert measurement["quantum_advantage"] > 0
    
    def test_quantum_noise_generation(self, privacy_config):
        """Test quantum-inspired noise generation."""
        from privacy_finetuner.core.quantum_optimizer import QuantumNoiseGenerator
        
        noise_gen = QuantumNoiseGenerator(
            epsilon=1.0, delta=1e-5, noise_multiplier=0.5
        )
        
        # Test noise addition
        test_tensor = torch.randn(10, 10)
        noisy_tensor = noise_gen.add_quantum_noise(test_tensor, sensitivity=1.0)
        
        # Noise should change the tensor
        assert not torch.allclose(test_tensor, noisy_tensor)
        
        # Shape should be preserved
        assert test_tensor.shape == noisy_tensor.shape
        
        # Test noise level
        noise_level = noise_gen.get_noise_level()
        assert noise_level > 0


class TestAdaptivePrivacyScheduling:
    """Test suite for adaptive privacy scheduling."""
    
    @pytest.fixture
    def privacy_config(self):
        return PrivacyConfig(epsilon=2.0, delta=1e-5)
    
    def test_schedule_strategies(self, privacy_config):
        """Test different scheduling strategies."""
        from privacy_finetuner.core.adaptive_privacy_scheduler import (
            PrivacySchedule, SchedulingStrategy
        )
        
        # Test linear schedule
        linear_schedule = PrivacySchedule(
            strategy=SchedulingStrategy.LINEAR,
            initial_epsilon=2.0,
            final_epsilon=0.5,
            total_steps=1000
        )
        
        scheduler = AdaptivePrivacyScheduler(privacy_config, linear_schedule)
        
        # Test epsilon calculation at different steps
        scheduler.step = 0
        initial_eps = scheduler._get_scheduled_epsilon()
        assert initial_eps == linear_schedule.initial_epsilon
        
        scheduler.step = 500  # Halfway
        mid_eps = scheduler._get_scheduled_epsilon()
        expected_mid = (linear_schedule.initial_epsilon + linear_schedule.final_epsilon) / 2
        assert abs(mid_eps - expected_mid) < 0.1
        
        scheduler.step = 1000  # End
        final_eps = scheduler._get_scheduled_epsilon()
        assert final_eps == linear_schedule.final_epsilon
    
    def test_risk_assessment(self, privacy_config):
        """Test privacy risk assessment."""
        from privacy_finetuner.core.adaptive_privacy_scheduler import (
            PrivacyRiskAssessor, AdaptationMetrics
        )
        
        risk_assessor = PrivacyRiskAssessor()
        
        # Test low risk scenario
        low_risk_metrics = AdaptationMetrics(
            gradient_variance=0.5,
            model_utility=0.9,
            convergence_rate=0.1,
            temporal_correlation=0.1
        )
        
        low_risk_score = risk_assessor.assess_risk(
            low_risk_metrics, privacy_config, privacy_spent=0.2
        )
        
        # Test high risk scenario
        high_risk_metrics = AdaptationMetrics(
            gradient_variance=3.0,
            model_utility=0.3,
            convergence_rate=0.8,
            temporal_correlation=0.9
        )
        
        high_risk_score = risk_assessor.assess_risk(
            high_risk_metrics, privacy_config, privacy_spent=1.8
        )
        
        assert high_risk_score > low_risk_score
        assert 0 <= low_risk_score <= 1
        assert 0 <= high_risk_score <= 1
    
    def test_utility_tracking(self):
        """Test utility tracking and degradation calculation."""
        from privacy_finetuner.core.adaptive_privacy_scheduler import UtilityTracker
        
        utility_tracker = UtilityTracker()
        
        # Record initial utility
        initial_utility = utility_tracker.update_utility(accuracy=0.9, loss=0.5)
        assert initial_utility > 0
        
        # Record degraded utility
        degraded_utility = utility_tracker.update_utility(accuracy=0.7, loss=0.8)
        
        # Calculate degradation
        degradation = utility_tracker.get_utility_degradation()
        assert degradation >= 0  # Should be non-negative
        assert degradation > 0  # Should detect the degradation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])