#!/usr/bin/env python3
"""
Performance Profiling and Optimization Tool

Comprehensive performance analysis for privacy-preserving ML training:
- Training performance profiling
- Memory usage analysis
- Privacy computation overhead measurement
- GPU utilization monitoring
- Optimization recommendations
"""

import cProfile
import io
import json
import os
import pstats
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import psutil
    import GPUtil
    import torch
    HAS_GPU_MONITORING = True
except ImportError:
    HAS_GPU_MONITORING = False


class PerformanceProfiler:
    """Comprehensive performance profiling for privacy-preserving ML."""

    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "profiles": {},
            "benchmarks": {},
            "recommendations": []
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        info = {
            "cpu_count": os.cpu_count(),
            "python_version": sys.version,
            "platform": sys.platform,
        }
        
        if HAS_GPU_MONITORING:
            info.update({
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "gpu_available": torch.cuda.is_available() if 'torch' in sys.modules else False,
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            })
            
            if torch.cuda.is_available():
                try:
                    gpus = GPUtil.getGPUs()
                    info["gpu_info"] = [
                        {
                            "name": gpu.name,
                            "memory_total_mb": gpu.memoryTotal,
                            "driver_version": gpu.driver
                        } for gpu in gpus
                    ]
                except Exception:
                    info["gpu_info"] = "Could not retrieve GPU information"
        
        return info

    @contextmanager
    def profile_context(self, profile_name: str):
        """Context manager for profiling code blocks."""
        print(f"🔍 Starting profiling: {profile_name}")
        
        profiler = cProfile.Profile()
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        profiler.enable()
        try:
            yield profiler
        finally:
            profiler.disable()
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Generate profile stats
            stats_buffer = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_buffer)
            stats.sort_stats('cumulative')
            stats.print_stats(50)  # Top 50 functions
            
            self.metrics["profiles"][profile_name] = {
                "duration_seconds": round(end_time - start_time, 4),
                "memory_start_mb": start_memory,
                "memory_end_mb": end_memory,
                "memory_peak_mb": self._get_peak_memory(),
                "memory_delta_mb": round(end_memory - start_memory, 2),
                "profile_stats": stats_buffer.getvalue(),
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ Completed profiling: {profile_name} ({end_time - start_time:.2f}s)")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if HAS_GPU_MONITORING:
            try:
                process = psutil.Process(os.getpid())
                return round(process.memory_info().rss / (1024 * 1024), 2)
            except Exception:
                return 0.0
        return 0.0

    def _get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if HAS_GPU_MONITORING:
            try:
                process = psutil.Process(os.getpid())
                return round(process.memory_info().peak_wss / (1024 * 1024), 2)
            except Exception:
                return 0.0
        return 0.0

    def benchmark_privacy_operations(self) -> Dict[str, Any]:
        """Benchmark privacy-specific operations."""
        print("🔒 Benchmarking privacy operations...")
        
        benchmarks = {}
        
        # Benchmark differential privacy noise generation
        try:
            import torch
            import numpy as np
            
            # Test noise generation performance
            with self.profile_context("dp_noise_generation"):
                sizes = [1000, 10000, 100000, 1000000]
                noise_times = []
                
                for size in sizes:
                    start_time = time.time()
                    # Simulate DP noise generation
                    noise = torch.normal(0, 1.0, (size,))
                    noise_times.append(time.time() - start_time)
                
                benchmarks["noise_generation"] = {
                    "sizes": sizes,
                    "times_seconds": noise_times,
                    "throughput_elements_per_second": [s/t for s, t in zip(sizes, noise_times)]
                }
        
        except ImportError:
            benchmarks["noise_generation"] = {"status": "skipped", "reason": "PyTorch not available"}
        
        # Benchmark gradient clipping
        try:
            with self.profile_context("gradient_clipping"):
                clip_times = []
                for size in [1000, 10000, 100000]:
                    start_time = time.time()
                    # Simulate gradient clipping
                    gradients = torch.randn(size, requires_grad=True)
                    torch.nn.utils.clip_grad_norm_([gradients], max_norm=1.0)
                    clip_times.append(time.time() - start_time)
                
                benchmarks["gradient_clipping"] = {
                    "sizes": [1000, 10000, 100000],
                    "times_seconds": clip_times
                }
                
        except Exception as e:
            benchmarks["gradient_clipping"] = {"status": "error", "error": str(e)}
        
        self.metrics["benchmarks"]["privacy_operations"] = benchmarks
        return benchmarks

    def benchmark_model_operations(self) -> Dict[str, Any]:
        """Benchmark model training and inference operations."""
        print("🧠 Benchmarking model operations...")
        
        benchmarks = {}
        
        try:
            import torch
            import torch.nn as nn
            
            # Create a simple model for benchmarking
            class SimpleModel(nn.Module):
                def __init__(self, input_size=768, hidden_size=256, output_size=2):
                    super().__init__()
                    self.linear1 = nn.Linear(input_size, hidden_size)
                    self.relu = nn.ReLU()
                    self.linear2 = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    return self.linear2(self.relu(self.linear1(x)))
            
            model = SimpleModel()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            # Benchmark forward pass
            with self.profile_context("forward_pass"):
                batch_sizes = [8, 16, 32, 64]
                forward_times = []
                
                for batch_size in batch_sizes:
                    input_data = torch.randn(batch_size, 768).to(device)
                    
                    start_time = time.time()
                    with torch.no_grad():
                        output = model(input_data)
                    forward_times.append(time.time() - start_time)
                
                benchmarks["forward_pass"] = {
                    "batch_sizes": batch_sizes,
                    "times_seconds": forward_times,
                    "device": str(device)
                }
            
            # Benchmark backward pass
            with self.profile_context("backward_pass"):
                backward_times = []
                optimizer = torch.optim.Adam(model.parameters())
                criterion = nn.CrossEntropyLoss()
                
                for batch_size in batch_sizes:
                    input_data = torch.randn(batch_size, 768).to(device)
                    target = torch.randint(0, 2, (batch_size,)).to(device)
                    
                    start_time = time.time()
                    optimizer.zero_grad()
                    output = model(input_data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    backward_times.append(time.time() - start_time)
                
                benchmarks["backward_pass"] = {
                    "batch_sizes": batch_sizes,
                    "times_seconds": backward_times,
                    "device": str(device)
                }
                
        except ImportError:
            benchmarks = {"status": "skipped", "reason": "PyTorch not available"}
        except Exception as e:
            benchmarks = {"status": "error", "error": str(e)}
        
        self.metrics["benchmarks"]["model_operations"] = benchmarks
        return benchmarks

    def benchmark_data_loading(self) -> Dict[str, Any]:
        """Benchmark data loading and preprocessing operations."""
        print("📊 Benchmarking data operations...")
        
        benchmarks = {}
        
        # Simulate data loading performance
        with self.profile_context("data_loading"):
            import random
            import json
            
            data_sizes = [1000, 5000, 10000, 50000]
            loading_times = []
            
            for size in data_sizes:
                start_time = time.time()
                # Simulate loading structured data
                data = [
                    {"text": f"Sample text {i}", "label": random.randint(0, 1)}
                    for i in range(size)
                ]
                loading_times.append(time.time() - start_time)
            
            benchmarks["synthetic_data_loading"] = {
                "data_sizes": data_sizes,
                "times_seconds": loading_times,
                "throughput_items_per_second": [s/t for s, t in zip(data_sizes, loading_times)]
            }
        
        self.metrics["benchmarks"]["data_operations"] = benchmarks
        return benchmarks

    def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze performance profiles to identify bottlenecks."""
        print("🎯 Analyzing performance bottlenecks...")
        
        bottlenecks = []
        
        for profile_name, profile_data in self.metrics["profiles"].items():
            duration = profile_data["duration_seconds"]
            memory_delta = profile_data["memory_delta_mb"]
            
            # Memory bottlenecks
            if memory_delta > 1000:  # More than 1GB memory increase
                bottlenecks.append({
                    "type": "memory",
                    "severity": "high" if memory_delta > 5000 else "medium",
                    "profile": profile_name,
                    "description": f"High memory usage increase: {memory_delta:.1f}MB",
                    "recommendation": "Consider memory optimization techniques"
                })
            
            # Time bottlenecks
            if duration > 10.0:  # More than 10 seconds
                bottlenecks.append({
                    "type": "performance",
                    "severity": "high" if duration > 60 else "medium",
                    "profile": profile_name,
                    "description": f"Long execution time: {duration:.2f}s",
                    "recommendation": "Consider algorithmic or parallel processing optimizations"
                })
        
        # Analyze benchmark results
        for benchmark_category, benchmark_data in self.metrics["benchmarks"].items():
            if isinstance(benchmark_data, dict):
                for operation, results in benchmark_data.items():
                    if isinstance(results, dict) and "times_seconds" in results:
                        max_time = max(results["times_seconds"])
                        if max_time > 5.0:
                            bottlenecks.append({
                                "type": "operation",
                                "severity": "medium",
                                "profile": f"{benchmark_category}.{operation}",
                                "description": f"Slow operation: {max_time:.2f}s max time",
                                "recommendation": "Consider optimization or caching"
                            })
        
        self.metrics["bottlenecks"] = bottlenecks
        return bottlenecks

    def generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # System-level recommendations
        system_info = self.metrics["system_info"]
        
        if system_info.get("memory_total_gb", 0) < 16:
            recommendations.append(
                "Consider increasing system memory to at least 16GB for optimal ML performance"
            )
        
        if not system_info.get("gpu_available", False):
            recommendations.append(
                "Consider using GPU acceleration for significant performance improvements"
            )
        
        # Profile-based recommendations
        for profile_name, profile_data in self.metrics["profiles"].items():
            memory_delta = profile_data["memory_delta_mb"]
            duration = profile_data["duration_seconds"]
            
            if memory_delta > 2000:
                recommendations.append(
                    f"Optimize memory usage in {profile_name} - consider batch processing or memory pooling"
                )
            
            if duration > 30:
                recommendations.append(
                    f"Optimize execution time in {profile_name} - consider parallel processing or algorithm improvements"
                )
        
        # Privacy-specific recommendations
        if "privacy_operations" in self.metrics["benchmarks"]:
            privacy_benchmarks = self.metrics["benchmarks"]["privacy_operations"]
            if "noise_generation" in privacy_benchmarks:
                noise_data = privacy_benchmarks["noise_generation"]
                if "throughput_elements_per_second" in noise_data:
                    min_throughput = min(noise_data["throughput_elements_per_second"])
                    if min_throughput < 100000:  # Less than 100k elements/second
                        recommendations.append(
                            "Consider optimizing differential privacy noise generation - use vectorized operations"
                        )
        
        # General ML recommendations
        recommendations.extend([
            "Use mixed precision training (FP16) to reduce memory usage and increase speed",
            "Implement gradient accumulation for large effective batch sizes",
            "Consider model pruning or quantization for inference optimization",
            "Use data parallelism for multi-GPU training",
            "Implement efficient data loading with multiple workers",
            "Profile memory usage during training to identify peak usage patterns"
        ])
        
        self.metrics["recommendations"] = recommendations
        return recommendations

    def run_comprehensive_benchmark(self) -> None:
        """Run comprehensive performance benchmarking."""
        print("🚀 Starting comprehensive performance benchmark...")
        
        # Run all benchmarks
        self.benchmark_privacy_operations()
        self.benchmark_model_operations()
        self.benchmark_data_loading()
        
        # Analyze results
        self.analyze_bottlenecks()
        self.generate_recommendations()
        
        # Generate reports
        self.generate_reports()

    def generate_reports(self) -> None:
        """Generate comprehensive performance reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate JSON report
        json_report_path = self.output_dir / f"performance_report_{timestamp}.json"
        with open(json_report_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        # Generate human-readable report
        readable_report = self._generate_readable_report()
        md_report_path = self.output_dir / f"performance_report_{timestamp}.md"
        with open(md_report_path, "w") as f:
            f.write(readable_report)
        
        print(f"\n📊 Performance benchmarking completed!")
        print(f"📄 Readable report: {md_report_path}")
        print(f"📋 Detailed report: {json_report_path}")

    def _generate_readable_report(self) -> str:
        """Generate human-readable performance report."""
        report = f"""
# Performance Analysis Report

**Generated**: {self.metrics['timestamp']}

## System Information

- **CPU Cores**: {self.metrics['system_info'].get('cpu_count', 'Unknown')}
- **Total Memory**: {self.metrics['system_info'].get('memory_total_gb', 'Unknown')} GB
- **GPU Available**: {self.metrics['system_info'].get('gpu_available', False)}
- **GPU Count**: {self.metrics['system_info'].get('gpu_count', 0)}
- **Platform**: {self.metrics['system_info'].get('platform', 'Unknown')}

## Performance Profiles

"""
        
        for profile_name, profile_data in self.metrics["profiles"].items():
            report += f"### {profile_name.replace('_', ' ').title()}\n"
            report += f"- **Duration**: {profile_data['duration_seconds']}s\n"
            report += f"- **Memory Delta**: {profile_data['memory_delta_mb']}MB\n"
            report += f"- **Peak Memory**: {profile_data.get('memory_peak_mb', 'N/A')}MB\n\n"
        
        report += "## Benchmark Results\n\n"
        
        for benchmark_category, benchmark_data in self.metrics["benchmarks"].items():
            report += f"### {benchmark_category.replace('_', ' ').title()}\n"
            if isinstance(benchmark_data, dict):
                for operation, results in benchmark_data.items():
                    if isinstance(results, dict) and "times_seconds" in results:
                        avg_time = sum(results["times_seconds"]) / len(results["times_seconds"])
                        report += f"- **{operation}**: Average {avg_time:.4f}s\n"
            report += "\n"
        
        if self.metrics.get("bottlenecks"):
            report += "## Performance Bottlenecks\n\n"
            for bottleneck in self.metrics["bottlenecks"]:
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(bottleneck["severity"], "")
                report += f"- {severity_emoji} **{bottleneck['profile']}**: {bottleneck['description']}\n"
                report += f"  - *Recommendation*: {bottleneck['recommendation']}\n\n"
        
        report += "## Optimization Recommendations\n\n"
        for i, recommendation in enumerate(self.metrics.get("recommendations", []), 1):
            report += f"{i}. {recommendation}\n"
        
        report += """
## Next Steps

1. **Address Critical Bottlenecks**: Focus on high-severity performance issues first
2. **Implement Recommendations**: Start with the most impactful optimizations
3. **Monitor Performance**: Set up continuous performance monitoring
4. **Regular Benchmarking**: Run benchmarks after major changes
5. **Hardware Optimization**: Consider hardware upgrades if needed

---
*This report was generated automatically by the performance profiler.*
"""
        
        return report.strip()


def main():
    """Main entry point for performance profiling."""
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "performance_reports"
    
    profiler = PerformanceProfiler(output_dir)
    
    try:
        profiler.run_comprehensive_benchmark()
    except Exception as e:
        print(f"Error during performance benchmarking: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()