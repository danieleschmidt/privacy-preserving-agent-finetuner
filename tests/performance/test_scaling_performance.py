"""Performance tests for scaling optimization components."""

import pytest
import time
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
import threading
from typing import List, Dict, Any

from privacy_finetuner.core.scaling_optimizer import (
    DistributedPrivacyOptimizer, MemoryManager, GradientCompressor,
    DistributedCacheManager, PerformanceTracker, ScalingConfig,
    PerformanceMetrics, DynamicLoadBalancer, AutoScaler
)
from privacy_finetuner.core import PrivacyConfig


class TestScalingPerformance:
    """Performance tests for scaling components."""
    
    @pytest.fixture
    def privacy_config(self):
        """Create privacy configuration for testing."""
        return PrivacyConfig(
            epsilon=2.0,
            delta=1e-5,
            noise_multiplier=0.5,
            max_grad_norm=1.0
        )
    
    @pytest.fixture
    def scaling_config(self):
        """Create scaling configuration for testing."""
        return ScalingConfig(
            num_workers=4,
            batch_size_per_worker=8,
            mixed_precision=True,
            gradient_compression=True,
            memory_efficient_attention=True
        )
    
    def test_memory_manager_performance(self, scaling_config):
        """Test memory manager performance with large batches."""
        memory_manager = MemoryManager(scaling_config)
        
        # Create large batch data
        batch_sizes = [16, 32, 64, 128]
        sequence_length = 512
        vocab_size = 50000
        
        performance_results = {}
        
        for batch_size in batch_sizes:
            batch_data = {
                "input_ids": torch.randint(0, vocab_size, (batch_size, sequence_length)),
                "attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.bool),
                "labels": torch.randint(0, vocab_size, (batch_size, sequence_length))
            }
            
            # Measure optimization time
            start_time = time.time()
            optimized_batch = memory_manager.optimize_batch(batch_data)
            end_time = time.time()
            
            optimization_time = end_time - start_time
            memory_usage = memory_manager.get_memory_usage()
            
            performance_results[batch_size] = {
                "optimization_time": optimization_time,
                "memory_usage": memory_usage,
                "memory_efficiency": memory_usage["efficiency"]
            }
            
            # Verify optimization didn't change batch structure
            assert len(optimized_batch) == len(batch_data)
            for key in batch_data:
                assert optimized_batch[key].shape == batch_data[key].shape
        
        # Performance assertions
        for batch_size, results in performance_results.items():
            # Optimization should be fast (< 100ms even for large batches)
            assert results["optimization_time"] < 0.1, f"Optimization too slow for batch size {batch_size}"
            
            # Memory efficiency should be reasonable
            assert results["memory_efficiency"] >= 0.0, f"Invalid memory efficiency for batch size {batch_size}"
    
    def test_gradient_compression_performance(self, scaling_config):
        """Test gradient compression performance and ratio."""
        compressor = GradientCompressor(scaling_config)
        
        # Test with different gradient sizes
        gradient_shapes = [
            (1000, 1000),    # 1M parameters
            (2048, 4096),    # 8M parameters
            (4096, 4096),    # 16M parameters
        ]
        
        compression_results = {}
        
        for shape in gradient_shapes:
            gradients = {
                f"layer_{i}": torch.randn(shape) 
                for i in range(3)  # Multiple layers
            }
            
            # Measure compression performance
            start_time = time.time()
            compressed = compressor.compress(gradients)
            compression_time = time.time() - start_time
            
            # Measure decompression performance
            start_time = time.time()
            decompressed = compressor.decompress(compressed)
            decompression_time = time.time() - start_time
            
            # Calculate compression ratio
            original_size = sum(grad.numel() * 4 for grad in gradients.values())  # 4 bytes per float32
            compressed_size = sum(
                len(comp["indices"]) * 4 + len(comp["values"]) * 4 
                for comp in compressed.values()
            )
            compression_ratio = original_size / compressed_size
            
            compression_results[shape] = {
                "compression_time": compression_time,
                "decompression_time": decompression_time,
                "compression_ratio": compression_ratio,
                "original_size_mb": original_size / (1024 * 1024),
                "compressed_size_mb": compressed_size / (1024 * 1024)
            }
            
            # Verify shapes are preserved
            for key in gradients:
                assert decompressed[key].shape == gradients[key].shape
        
        # Performance assertions
        for shape, results in compression_results.items():
            # Compression should achieve significant reduction
            assert results["compression_ratio"] > 5.0, f"Poor compression ratio for shape {shape}"
            
            # Operations should be reasonably fast
            assert results["compression_time"] < 1.0, f"Compression too slow for shape {shape}"
            assert results["decompression_time"] < 1.0, f"Decompression too slow for shape {shape}"
    
    def test_cache_manager_concurrent_performance(self):
        """Test cache manager performance under concurrent access."""
        cache_manager = DistributedCacheManager()
        
        num_threads = 10
        operations_per_thread = 100
        cache_keys = [f"key_{i}" for i in range(operations_per_thread)]
        cache_values = [{"data": torch.randn(100, 100), "metadata": {"id": i}} for i in range(operations_per_thread)]
        
        def cache_operations(thread_id):
            """Perform cache operations in a thread."""
            thread_results = {"hits": 0, "misses": 0, "operations": 0}
            
            for i in range(operations_per_thread // 2):
                key = f"thread_{thread_id}_key_{i}"
                value = {"thread_id": thread_id, "data": torch.randn(50, 50)}
                
                # Set operation
                cache_manager.set(key, value)
                thread_results["operations"] += 1
                
                # Get operation
                retrieved = cache_manager.get(key)
                if retrieved is not None:
                    thread_results["hits"] += 1
                else:
                    thread_results["misses"] += 1
                thread_results["operations"] += 1
            
            return thread_results
        
        # Run concurrent cache operations
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(cache_operations, i) for i in range(num_threads)]
            thread_results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Aggregate results
        total_operations = sum(result["operations"] for result in thread_results)
        total_hits = sum(result["hits"] for result in thread_results)
        total_misses = sum(result["misses"] for result in thread_results)
        
        operations_per_second = total_operations / total_time
        hit_ratio = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
        
        # Performance assertions
        assert operations_per_second > 1000, f"Cache operations too slow: {operations_per_second:.2f} ops/sec"
        
        # Get final cache statistics
        cache_stats = cache_manager.get_cache_stats()
        assert "hit_ratio" in cache_stats
        assert "total_hits" in cache_stats
        assert "total_misses" in cache_stats
    
    def test_performance_tracker_overhead(self):
        """Test performance tracker overhead."""
        tracker = PerformanceTracker()
        
        num_measurements = 10000
        batch_sizes = [8, 16, 32, 64]
        
        # Measure overhead of performance tracking
        start_time = time.time()
        
        for i in range(num_measurements):
            batch_size = batch_sizes[i % len(batch_sizes)]
            processing_time = 0.01 + np.random.normal(0, 0.001)  # Simulate processing time
            
            tracker.record_batch_processing(processing_time, batch_size)
        
        end_time = time.time()
        tracking_overhead = end_time - start_time
        
        # Get current metrics
        metrics = tracker.get_current_metrics()
        
        # Performance assertions
        overhead_per_measurement = tracking_overhead / num_measurements
        assert overhead_per_measurement < 0.0001, f"Performance tracking overhead too high: {overhead_per_measurement:.6f}s per measurement"
        
        # Verify metrics are reasonable
        assert metrics.throughput_samples_per_sec > 0
        assert metrics.batch_processing_time > 0
        assert len(tracker.batch_times) <= 1000  # History should be limited
    
    def test_load_balancer_performance(self, scaling_config):
        """Test load balancer performance under high load."""
        load_balancer = DynamicLoadBalancer(scaling_config)
        load_balancer.update_worker_count(scaling_config.num_workers)
        
        num_assignments = 10000
        work_sizes = np.random.exponential(1.0, num_assignments)  # Exponential distribution of work sizes
        
        # Measure assignment performance
        start_time = time.time()
        
        assignments = []
        for work_size in work_sizes:
            worker_id = load_balancer.assign_work(work_size)
            assignments.append(worker_id)
        
        end_time = time.time()
        assignment_time = end_time - start_time
        
        # Measure load distribution quality
        worker_loads = {}
        for i, work_size in enumerate(work_sizes):
            worker_id = assignments[i]
            if worker_id not in worker_loads:
                worker_loads[worker_id] = 0
            worker_loads[worker_id] += work_size
        
        load_values = list(worker_loads.values())
        load_variance = np.var(load_values)
        load_balance = min(load_values) / max(load_values) if max(load_values) > 0 else 1.0
        
        assignments_per_second = num_assignments / assignment_time
        
        # Performance assertions
        assert assignments_per_second > 50000, f"Load balancing too slow: {assignments_per_second:.2f} assignments/sec"
        assert load_balance > 0.8, f"Poor load balancing: {load_balance:.3f}"
        
        # Verify all workers were used
        assert len(worker_loads) == scaling_config.num_workers
        
        # Get load statistics
        load_stats = load_balancer.get_load_stats()
        assert "balance_factor" in load_stats
        assert load_stats["balance_factor"] > 0.5  # Reasonable balance
    
    def test_auto_scaler_decision_speed(self, scaling_config):
        """Test auto scaler decision making speed."""
        auto_scaler = AutoScaler(scaling_config)
        
        # Generate various performance scenarios
        scenarios = []
        for _ in range(1000):
            metrics = PerformanceMetrics(
                gpu_utilization=np.random.uniform(0.1, 1.0),
                memory_efficiency=np.random.uniform(0.3, 1.0),
                throughput_samples_per_sec=np.random.uniform(10, 1000),
                cache_hit_ratio=np.random.uniform(0.5, 1.0)
            )
            scenarios.append(metrics)
        
        # Measure decision making performance
        start_time = time.time()
        
        decisions = []
        for metrics in scenarios:
            decision = auto_scaler.make_scaling_decision(metrics)
            decisions.append(decision)
        
        end_time = time.time()
        decision_time = end_time - start_time
        
        decisions_per_second = len(scenarios) / decision_time
        
        # Analyze decision patterns
        action_counts = {}
        for decision in decisions:
            action = decision["action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Performance assertions
        assert decisions_per_second > 10000, f"Auto scaling decisions too slow: {decisions_per_second:.2f} decisions/sec"
        
        # Verify decision variety (shouldn't always be the same action)
        assert len(action_counts) > 1, "Auto scaler not responding to different scenarios"
        
        # Verify all decisions have required fields
        for decision in decisions:
            assert "action" in decision
            assert "target_workers" in decision
            assert decision["target_workers"] > 0
    
    @pytest.mark.slow
    def test_distributed_optimizer_scalability(self, privacy_config, scaling_config):
        """Test distributed optimizer performance with different worker counts."""
        # Mock model for testing
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(1000, 1000) for _ in range(5)]
        
        worker_counts = [1, 2, 4, 8]
        performance_results = {}
        
        for num_workers in worker_counts:
            test_scaling_config = ScalingConfig(
                num_workers=num_workers,
                batch_size_per_worker=scaling_config.batch_size_per_worker,
                mixed_precision=scaling_config.mixed_precision,
                gradient_compression=scaling_config.gradient_compression
            )
            
            optimizer = DistributedPrivacyOptimizer(
                privacy_config, test_scaling_config, mock_model
            )
            
            # Simulate batch processing
            batch_data = {
                "input_ids": torch.randint(0, 1000, (num_workers * 8, 512)),
                "attention_mask": torch.ones(num_workers * 8, 512, dtype=torch.bool)
            }
            
            gradients = {
                f"layer_{i}": torch.randn(100, 100) 
                for i in range(10)
            }
            
            # Measure optimization performance
            start_time = time.time()
            
            for _ in range(10):  # Multiple iterations
                optimized_gradients = optimizer.optimize_batch_processing(batch_data, gradients)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Get scaling metrics
            scaling_metrics = optimizer.get_scaling_metrics()
            
            performance_results[num_workers] = {
                "total_time": total_time,
                "time_per_iteration": total_time / 10,
                "scaling_efficiency": scaling_metrics.get("scaling_efficiency", 1.0),
                "bottlenecks": scaling_metrics.get("bottlenecks", [])
            }
        
        # Analyze scaling efficiency
        baseline_time = performance_results[1]["time_per_iteration"]
        
        for num_workers in worker_counts[1:]:  # Skip single worker
            results = performance_results[num_workers]
            
            # Ideal speedup would be linear
            ideal_speedup = num_workers
            actual_speedup = baseline_time / results["time_per_iteration"]
            efficiency = actual_speedup / ideal_speedup
            
            # Performance should improve with more workers (allowing for overhead)
            assert efficiency > 0.5, f"Poor scaling efficiency with {num_workers} workers: {efficiency:.3f}"
            
            # Time per iteration should decrease with more workers
            assert results["time_per_iteration"] < baseline_time * 1.5, f"No performance improvement with {num_workers} workers"
    
    def test_memory_pressure_handling(self, scaling_config):
        """Test system behavior under memory pressure."""
        memory_manager = MemoryManager(scaling_config)
        
        # Simulate increasing memory pressure
        batch_sizes = [16, 32, 64, 128, 256]
        memory_stats = {}
        
        for batch_size in batch_sizes:
            # Create increasingly large batches
            batch_data = {
                "input_ids": torch.randint(0, 50000, (batch_size, 1024)),
                "attention_mask": torch.ones(batch_size, 1024, dtype=torch.bool),
                "labels": torch.randint(0, 50000, (batch_size, 1024))
            }
            
            try:
                # Process batch and measure memory
                optimized_batch = memory_manager.optimize_batch(batch_data)
                memory_manager.optimize_memory_usage()
                
                current_memory = memory_manager.get_memory_usage()
                memory_stats[batch_size] = {
                    "success": True,
                    "memory_usage": current_memory,
                    "allocated_gb": current_memory.get("allocated_gb", 0)
                }
                
            except Exception as e:
                # Handle memory exhaustion gracefully
                memory_stats[batch_size] = {
                    "success": False,
                    "error": str(e),
                    "memory_usage": memory_manager.get_memory_usage()
                }
        
        # Verify graceful handling of memory pressure
        success_count = sum(1 for stats in memory_stats.values() if stats["success"])
        assert success_count >= len(batch_sizes) // 2, "System should handle reasonable memory pressure"
        
        # Memory usage should be tracked accurately
        for batch_size, stats in memory_stats.items():
            if stats["success"]:
                assert "allocated_gb" in stats["memory_usage"]
                assert stats["memory_usage"]["allocated_gb"] >= 0


class TestConcurrencyAndThreadSafety:
    """Test thread safety and concurrency of scaling components."""
    
    def test_cache_manager_thread_safety(self):
        """Test cache manager thread safety."""
        cache_manager = DistributedCacheManager()
        
        results = []
        errors = []
        
        def concurrent_cache_access(thread_id):
            """Perform concurrent cache operations."""
            try:
                for i in range(100):
                    key = f"thread_{thread_id}_key_{i}"
                    value = {"thread_id": thread_id, "iteration": i, "data": list(range(100))}
                    
                    # Set and immediately get
                    cache_manager.set(key, value)
                    retrieved = cache_manager.get(key)
                    
                    # Verify data integrity
                    if retrieved is not None:
                        assert retrieved["thread_id"] == thread_id
                        assert retrieved["iteration"] == i
                        assert len(retrieved["data"]) == 100
                
                results.append(f"Thread {thread_id} completed successfully")
                
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {str(e)}")
        
        # Run concurrent threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_cache_access, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 5, f"Not all threads completed: {results}"
    
    def test_performance_tracker_concurrent_updates(self):
        """Test performance tracker under concurrent updates."""
        tracker = PerformanceTracker()
        
        errors = []
        
        def concurrent_tracking(thread_id):
            """Perform concurrent performance tracking."""
            try:
                for i in range(200):
                    processing_time = 0.01 + np.random.normal(0, 0.002)
                    batch_size = 8 + (i % 4) * 4  # Varying batch sizes
                    
                    tracker.record_batch_processing(processing_time, batch_size)
                    
                    # Occasionally get metrics
                    if i % 50 == 0:
                        metrics = tracker.get_current_metrics()
                        assert metrics.throughput_samples_per_sec >= 0
                        assert metrics.batch_processing_time >= 0
                
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {str(e)}")
        
        # Run concurrent tracking
        threads = []
        for i in range(8):
            thread = threading.Thread(target=concurrent_tracking, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)
        
        # Verify no errors and reasonable final state
        assert len(errors) == 0, f"Concurrent tracking errors: {errors}"
        
        final_metrics = tracker.get_current_metrics()
        assert final_metrics.throughput_samples_per_sec > 0
        assert tracker.step_count == 8 * 200  # Total steps from all threads


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])