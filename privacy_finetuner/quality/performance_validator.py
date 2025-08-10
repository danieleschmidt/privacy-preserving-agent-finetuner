"""Performance validation and benchmarking module."""

import time
import psutil
import gc
import sys
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark result."""
    name: str
    duration: float
    memory_usage: float
    cpu_usage: float
    throughput: Optional[float] = None
    latency: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThresholds:
    """Performance thresholds for validation."""
    max_duration: Optional[float] = None
    max_memory_mb: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    min_throughput: Optional[float] = None
    max_latency: Optional[float] = None


class PerformanceValidator:
    """Performance validation for privacy-preserving ML operations."""
    
    def __init__(self):
        self.benchmarks = []
        self.thresholds = {}
        self.monitoring_enabled = True
        
    def set_thresholds(self, operation: str, thresholds: PerformanceThresholds):
        """Set performance thresholds for an operation."""
        self.thresholds[operation] = thresholds
        
    def benchmark_operation(
        self, 
        operation_name: str, 
        operation_func: Callable,
        *args,
        **kwargs
    ) -> PerformanceBenchmark:
        """Benchmark a specific operation."""
        logger.info(f"Benchmarking operation: {operation_name}")
        
        # Collect initial metrics
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run garbage collection before benchmark
        gc.collect()
        
        start_time = time.time()
        cpu_times_start = process.cpu_times()
        
        success = True
        error_msg = None
        result = None
        
        try:
            # Execute the operation
            result = operation_func(*args, **kwargs)
        except Exception as e:
            success = False
            error_msg = str(e)
            logger.error(f"Benchmark failed for {operation_name}: {e}")
        
        end_time = time.time()
        cpu_times_end = process.cpu_times()
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        duration = end_time - start_time
        memory_usage = final_memory - initial_memory
        cpu_usage = ((cpu_times_end.user - cpu_times_start.user) + 
                    (cpu_times_end.system - cpu_times_start.system)) / duration * 100
        
        # Calculate throughput and latency if applicable
        throughput = None
        latency = None
        
        if hasattr(result, '__len__') and duration > 0:
            throughput = len(result) / duration
            latency = duration / len(result) if len(result) > 0 else None
        
        benchmark = PerformanceBenchmark(
            name=operation_name,
            duration=duration,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            throughput=throughput,
            latency=latency,
            success=success,
            error=error_msg,
            metadata={"result_type": type(result).__name__ if result else None}
        )
        
        self.benchmarks.append(benchmark)
        return benchmark
    
    def validate_performance(self, benchmark: PerformanceBenchmark) -> Dict[str, Any]:
        """Validate benchmark against thresholds."""
        operation_name = benchmark.name
        thresholds = self.thresholds.get(operation_name)
        
        if not thresholds:
            return {
                "operation": operation_name,
                "validation_passed": True,
                "message": "No thresholds defined",
                "violations": []
            }
        
        violations = []
        
        # Check duration threshold
        if thresholds.max_duration and benchmark.duration > thresholds.max_duration:
            violations.append({
                "metric": "duration",
                "value": benchmark.duration,
                "threshold": thresholds.max_duration,
                "severity": "high"
            })
        
        # Check memory threshold
        if thresholds.max_memory_mb and benchmark.memory_usage > thresholds.max_memory_mb:
            violations.append({
                "metric": "memory_usage",
                "value": benchmark.memory_usage,
                "threshold": thresholds.max_memory_mb,
                "severity": "high"
            })
        
        # Check CPU threshold
        if thresholds.max_cpu_percent and benchmark.cpu_usage > thresholds.max_cpu_percent:
            violations.append({
                "metric": "cpu_usage",
                "value": benchmark.cpu_usage,
                "threshold": thresholds.max_cpu_percent,
                "severity": "medium"
            })
        
        # Check throughput threshold
        if (thresholds.min_throughput and benchmark.throughput and 
            benchmark.throughput < thresholds.min_throughput):
            violations.append({
                "metric": "throughput",
                "value": benchmark.throughput,
                "threshold": thresholds.min_throughput,
                "severity": "medium"
            })
        
        # Check latency threshold
        if thresholds.max_latency and benchmark.latency and benchmark.latency > thresholds.max_latency:
            violations.append({
                "metric": "latency",
                "value": benchmark.latency,
                "threshold": thresholds.max_latency,
                "severity": "medium"
            })
        
        return {
            "operation": operation_name,
            "validation_passed": len(violations) == 0,
            "violations": violations,
            "benchmark": {
                "duration": benchmark.duration,
                "memory_usage": benchmark.memory_usage,
                "cpu_usage": benchmark.cpu_usage,
                "throughput": benchmark.throughput,
                "latency": benchmark.latency,
                "success": benchmark.success
            }
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.benchmarks:
            return {"error": "No benchmarks available"}
        
        successful_benchmarks = [b for b in self.benchmarks if b.success]
        failed_benchmarks = [b for b in self.benchmarks if not b.success]
        
        # Calculate statistics
        total_duration = sum(b.duration for b in successful_benchmarks)
        avg_duration = total_duration / len(successful_benchmarks) if successful_benchmarks else 0
        max_duration = max((b.duration for b in successful_benchmarks), default=0)
        
        total_memory = sum(abs(b.memory_usage) for b in successful_benchmarks)
        avg_memory = total_memory / len(successful_benchmarks) if successful_benchmarks else 0
        max_memory = max((abs(b.memory_usage) for b in successful_benchmarks), default=0)
        
        avg_cpu = sum(b.cpu_usage for b in successful_benchmarks) / len(successful_benchmarks) if successful_benchmarks else 0
        max_cpu = max((b.cpu_usage for b in successful_benchmarks), default=0)
        
        return {
            "total_benchmarks": len(self.benchmarks),
            "successful_benchmarks": len(successful_benchmarks),
            "failed_benchmarks": len(failed_benchmarks),
            "success_rate": len(successful_benchmarks) / len(self.benchmarks) if self.benchmarks else 0,
            "performance_summary": {
                "total_duration": total_duration,
                "average_duration": avg_duration,
                "max_duration": max_duration,
                "average_memory_usage": avg_memory,
                "max_memory_usage": max_memory,
                "average_cpu_usage": avg_cpu,
                "max_cpu_usage": max_cpu
            },
            "failed_operations": [
                {"name": b.name, "error": b.error} for b in failed_benchmarks
            ],
            "timestamp": datetime.now().isoformat()
        }


class BenchmarkRunner:
    """Advanced benchmark runner with concurrency and stress testing."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.validator = PerformanceValidator()
        self.stress_test_results = []
        
    def run_stress_test(
        self,
        operation_name: str,
        operation_func: Callable,
        num_iterations: int = 100,
        concurrency_levels: List[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run stress test with varying concurrency levels."""
        if concurrency_levels is None:
            concurrency_levels = [1, 2, 4, 8]
        
        logger.info(f"Running stress test for {operation_name}")
        
        stress_results = []
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency}")
            
            start_time = time.time()
            successful_operations = 0
            failed_operations = 0
            
            with ThreadPoolExecutor(max_workers=min(concurrency, self.max_workers)) as executor:
                futures = []
                
                for i in range(num_iterations):
                    future = executor.submit(self._run_single_operation, operation_func, **kwargs)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)  # 30 second timeout
                        successful_operations += 1
                    except Exception as e:
                        failed_operations += 1
                        logger.warning(f"Operation failed: {e}")
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            stress_results.append({
                "concurrency": concurrency,
                "total_duration": total_duration,
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "success_rate": successful_operations / num_iterations,
                "operations_per_second": successful_operations / total_duration if total_duration > 0 else 0
            })
        
        self.stress_test_results.append({
            "operation": operation_name,
            "num_iterations": num_iterations,
            "results": stress_results,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "operation": operation_name,
            "stress_test_results": stress_results,
            "best_performance": max(stress_results, key=lambda x: x["operations_per_second"]),
            "recommendations": self._generate_recommendations(stress_results)
        }
    
    def _run_single_operation(self, operation_func: Callable, **kwargs):
        """Run a single operation for stress testing."""
        return operation_func(**kwargs)
    
    def _generate_recommendations(self, stress_results: List[Dict[str, Any]]) -> List[str]:
        """Generate performance recommendations based on stress test results."""
        recommendations = []
        
        # Find optimal concurrency
        best_result = max(stress_results, key=lambda x: x["operations_per_second"])
        recommendations.append(
            f"Optimal concurrency level: {best_result['concurrency']} "
            f"({best_result['operations_per_second']:.2f} ops/sec)"
        )
        
        # Check for performance degradation
        single_thread_ops = next((r["operations_per_second"] for r in stress_results if r["concurrency"] == 1), 0)
        max_ops = max(r["operations_per_second"] for r in stress_results)
        
        if max_ops > single_thread_ops * 2:
            recommendations.append("Operation benefits significantly from concurrency")
        elif max_ops < single_thread_ops * 1.5:
            recommendations.append("Operation shows limited concurrency benefits - consider bottlenecks")
        
        # Check success rates
        min_success_rate = min(r["success_rate"] for r in stress_results)
        if min_success_rate < 0.95:
            recommendations.append(f"Reliability concerns: minimum success rate {min_success_rate:.2%}")
        
        return recommendations
    
    def run_memory_profile(self, operation_func: Callable, **kwargs) -> Dict[str, Any]:
        """Profile memory usage of an operation."""
        import tracemalloc
        
        # Start memory tracing
        tracemalloc.start()
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        try:
            # Run the operation
            start_time = time.time()
            result = operation_func(**kwargs)
            end_time = time.time()
            
            # Get final memory
            final_memory = process.memory_info().rss
            memory_growth = final_memory - initial_memory
            
            # Get tracemalloc stats
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            return {
                "duration": end_time - start_time,
                "memory_growth_bytes": memory_growth,
                "memory_growth_mb": memory_growth / (1024 * 1024),
                "traced_current": current,
                "traced_peak": peak,
                "traced_peak_mb": peak / (1024 * 1024),
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            tracemalloc.stop()
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }