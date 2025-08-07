"""Advanced resource optimization for privacy-preserving training."""

import logging
import time
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings

# Handle optional dependencies gracefully
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Resource monitoring will be limited.")

# Handle optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. Resource optimization will be limited.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. GPU optimization will be limited.")

from ..utils.logging_config import performance_monitor

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ResourceProfile:
    """Profile of available computational resources."""
    cpu_cores: int
    cpu_frequency: float  # GHz
    total_memory: int     # GB
    available_memory: int # GB
    gpu_count: int
    gpu_memory: List[int] = field(default_factory=list)  # GB per GPU
    storage_type: str = "ssd"  # "ssd" or "hdd"
    network_bandwidth: float = 1.0  # Gbps
    
    # Performance characteristics
    memory_bandwidth: float = 100.0  # GB/s
    storage_bandwidth: float = 5.0   # GB/s
    
    # Optimization hints
    numa_nodes: int = 1
    cache_sizes: Dict[str, int] = field(default_factory=dict)  # L1, L2, L3 in KB
    
    @classmethod
    def detect_system_profile(cls) -> "ResourceProfile":
        """Automatically detect system resource profile."""
        if PSUTIL_AVAILABLE:
            # CPU information
            cpu_cores = psutil.cpu_count(logical=False) or 4
            cpu_freq = psutil.cpu_freq()
            cpu_frequency = cpu_freq.max / 1000.0 if cpu_freq else 2.5  # Convert MHz to GHz
            
            # Memory information
            memory = psutil.virtual_memory()
            total_memory = int(memory.total / (1024**3))  # Convert to GB
            available_memory = int(memory.available / (1024**3))
        else:
            # Fallback values when psutil is not available
            cpu_cores = os.cpu_count() or 4
            cpu_frequency = 2.5  # Default assumption
            total_memory = 8  # Default 8GB
            available_memory = 6  # Default 6GB available
        
        # GPU information
        gpu_count = 0
        gpu_memory = []
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_mem = torch.cuda.get_device_properties(i).total_memory
                gpu_memory.append(int(gpu_mem / (1024**3)))  # Convert to GB
        
        # Storage type detection (simplified)
        storage_type = "ssd"  # Default assumption for modern systems
        
        # Network bandwidth estimation (simplified)
        network_bandwidth = 1.0  # Default 1 Gbps
        
        return cls(
            cpu_cores=cpu_cores,
            cpu_frequency=cpu_frequency,
            total_memory=total_memory,
            available_memory=available_memory,
            gpu_count=gpu_count,
            gpu_memory=gpu_memory,
            storage_type=storage_type,
            network_bandwidth=network_bandwidth
        )
    
    def get_resource_score(self) -> float:
        """Calculate overall resource capability score."""
        cpu_score = self.cpu_cores * self.cpu_frequency
        memory_score = self.total_memory * 0.1
        gpu_score = sum(self.gpu_memory) * 2.0 if self.gpu_memory else 0
        storage_score = 10.0 if self.storage_type == "ssd" else 5.0
        
        return cpu_score + memory_score + gpu_score + storage_score


class ResourceOptimizer:
    """Advanced resource optimizer for privacy-preserving training.
    
    Optimizes resource utilization for:
    - Memory management and caching
    - CPU/GPU workload balancing  
    - I/O optimization
    - Network bandwidth utilization
    - Power efficiency
    """
    
    def __init__(
        self,
        resource_profile: Optional[ResourceProfile] = None,
        optimization_target: str = "throughput"  # "throughput", "latency", "power"
    ):
        """Initialize resource optimizer.
        
        Args:
            resource_profile: System resource profile (auto-detected if None)
            optimization_target: Primary optimization objective
        """
        self.resource_profile = resource_profile or ResourceProfile.detect_system_profile()
        self.optimization_target = optimization_target
        
        # Resource monitoring
        self.resource_history: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Adaptive parameters
        self.batch_size_recommendations: Dict[str, int] = {}
        self.memory_usage_targets: Dict[str, float] = {}
        self.parallelism_settings: Dict[str, int] = {}
        
        logger.info(f"Initialized resource optimizer for {self.resource_profile.cpu_cores} CPU cores, "
                   f"{self.resource_profile.gpu_count} GPUs, {self.resource_profile.total_memory}GB RAM")
    
    def optimize_training_configuration(
        self,
        model_size: int,
        dataset_size: int,
        target_privacy_budget: float,
        time_constraint: Optional[float] = None
    ) -> Dict[str, Any]:
        """Optimize training configuration for given constraints.
        
        Args:
            model_size: Model parameter count
            dataset_size: Number of training samples
            target_privacy_budget: Privacy budget (epsilon)
            time_constraint: Maximum training time in hours (optional)
            
        Returns:
            Optimized configuration parameters
        """
        timer_id = performance_monitor.start_timer("configuration_optimization")
        
        try:
            logger.info("Optimizing training configuration")
            
            # Estimate memory requirements
            memory_requirements = self._estimate_memory_requirements(model_size, dataset_size)
            
            # Optimize batch size
            optimal_batch_size = self._optimize_batch_size(
                model_size, dataset_size, memory_requirements
            )
            
            # Optimize parallelism
            parallelism_config = self._optimize_parallelism(
                model_size, optimal_batch_size
            )
            
            # Optimize I/O settings
            io_config = self._optimize_io_settings(dataset_size)
            
            # Optimize privacy-performance trade-offs
            privacy_config = self._optimize_privacy_parameters(
                target_privacy_budget, time_constraint
            )
            
            # Combine all optimizations
            optimized_config = {
                "batch_size": optimal_batch_size,
                "gradient_accumulation_steps": parallelism_config["gradient_accumulation"],
                "dataloader_workers": io_config["num_workers"],
                "prefetch_factor": io_config["prefetch_factor"],
                "pin_memory": io_config["pin_memory"],
                "mixed_precision": parallelism_config["mixed_precision"],
                "gradient_checkpointing": parallelism_config["gradient_checkpointing"],
                "model_parallel": parallelism_config["model_parallel"],
                "data_parallel": parallelism_config["data_parallel"],
                "noise_multiplier": privacy_config["noise_multiplier"],
                "max_grad_norm": privacy_config["max_grad_norm"],
                "expected_training_time": self._estimate_training_time(
                    model_size, dataset_size, optimal_batch_size
                ),
                "memory_usage_estimate": memory_requirements,
                "optimization_target": self.optimization_target
            }
            
            # Store optimization for future reference
            self.optimization_history.append({
                "timestamp": time.time(),
                "model_size": model_size,
                "dataset_size": dataset_size,
                "config": optimized_config,
                "resource_profile": self.resource_profile.__dict__
            })
            
            performance_monitor.end_timer(timer_id)
            logger.info("Training configuration optimization completed")
            
            return optimized_config
            
        except Exception as e:
            performance_monitor.end_timer(timer_id)
            logger.error(f"Configuration optimization failed: {e}")
            raise
    
    def _estimate_memory_requirements(self, model_size: int, dataset_size: int) -> Dict[str, float]:
        """Estimate memory requirements for training."""
        # Model parameters (fp16/fp32)
        model_memory = model_size * 4 / (1024**3)  # 4 bytes per param -> GB
        
        # Gradients (same size as parameters)
        gradient_memory = model_memory
        
        # Optimizer states (Adam: 2x parameters for momentum + variance)
        optimizer_memory = model_memory * 2
        
        # Activations (estimated based on model size and batch size)
        activation_memory = model_size * 0.1 / (1024**3)  # Rough estimate
        
        # Dataset caching (partial)
        cache_memory = min(dataset_size * 0.001, 2.0)  # Max 2GB for dataset cache
        
        total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory + cache_memory
        
        return {
            "model": model_memory,
            "gradients": gradient_memory,
            "optimizer": optimizer_memory,
            "activations": activation_memory,
            "cache": cache_memory,
            "total": total_memory
        }
    
    def _optimize_batch_size(
        self, 
        model_size: int, 
        dataset_size: int, 
        memory_requirements: Dict[str, float]
    ) -> int:
        """Optimize batch size based on memory constraints and performance."""
        
        # Calculate maximum batch size based on memory
        available_memory = self.resource_profile.available_memory * 0.8  # Use 80% of available
        memory_per_sample = memory_requirements["activations"] * 100  # Rough estimate
        max_batch_size = int(available_memory / memory_per_sample) if memory_per_sample > 0 else 64
        
        # Performance-optimal batch size (powers of 2 are usually best)
        performance_batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
        
        # Choose largest performance batch size that fits in memory
        optimal_batch_size = 32  # Default
        for batch_size in performance_batch_sizes:
            if batch_size <= max_batch_size:
                optimal_batch_size = batch_size
            else:
                break
        
        # Adjust for GPU memory if available
        if self.resource_profile.gpu_count > 0 and self.resource_profile.gpu_memory:
            min_gpu_memory = min(self.resource_profile.gpu_memory)
            gpu_batch_limit = int(min_gpu_memory * 0.6 / (memory_per_sample or 0.1))
            optimal_batch_size = min(optimal_batch_size, gpu_batch_limit)
        
        # Ensure minimum viable batch size
        optimal_batch_size = max(optimal_batch_size, 2)
        
        logger.debug(f"Optimal batch size: {optimal_batch_size} (max: {max_batch_size})")
        return optimal_batch_size
    
    def _optimize_parallelism(self, model_size: int, batch_size: int) -> Dict[str, Any]:
        """Optimize parallelism settings."""
        config = {
            "gradient_accumulation": 1,
            "mixed_precision": False,
            "gradient_checkpointing": False,
            "model_parallel": False,
            "data_parallel": False
        }
        
        # Enable mixed precision for large models
        if model_size > 100_000_000:  # > 100M parameters
            config["mixed_precision"] = True
            logger.debug("Enabled mixed precision for large model")
        
        # Enable gradient checkpointing for memory efficiency
        if self.resource_profile.available_memory < 16:  # < 16GB RAM
            config["gradient_checkpointing"] = True
            logger.debug("Enabled gradient checkpointing for memory efficiency")
        
        # Configure data parallelism for multiple GPUs
        if self.resource_profile.gpu_count > 1:
            config["data_parallel"] = True
            config["gradient_accumulation"] = max(1, batch_size // (self.resource_profile.gpu_count * 8))
            logger.debug(f"Enabled data parallelism across {self.resource_profile.gpu_count} GPUs")
        
        # Configure model parallelism for very large models
        if model_size > 1_000_000_000:  # > 1B parameters
            config["model_parallel"] = True
            logger.debug("Enabled model parallelism for very large model")
        
        return config
    
    def _optimize_io_settings(self, dataset_size: int) -> Dict[str, Any]:
        """Optimize I/O settings for data loading."""
        # Base settings
        num_workers = min(self.resource_profile.cpu_cores, 8)  # Cap at 8 workers
        prefetch_factor = 2
        pin_memory = self.resource_profile.gpu_count > 0
        
        # Adjust for large datasets
        if dataset_size > 1_000_000:  # > 1M samples
            num_workers = min(self.resource_profile.cpu_cores, 12)
            prefetch_factor = 4
        
        # Adjust for fast storage
        if self.resource_profile.storage_type == "ssd":
            prefetch_factor = max(prefetch_factor, 4)
        
        logger.debug(f"I/O config: {num_workers} workers, prefetch={prefetch_factor}")
        
        return {
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "pin_memory": pin_memory
        }
    
    def _optimize_privacy_parameters(
        self, 
        target_epsilon: float, 
        time_constraint: Optional[float]
    ) -> Dict[str, float]:
        """Optimize privacy parameters for performance."""
        
        # Base privacy settings
        noise_multiplier = 1.0
        max_grad_norm = 1.0
        
        # Adjust noise multiplier based on target epsilon
        if target_epsilon < 1.0:  # High privacy
            noise_multiplier = 1.5
            max_grad_norm = 0.8
        elif target_epsilon > 5.0:  # Lower privacy for better utility
            noise_multiplier = 0.5
            max_grad_norm = 1.5
        
        # Consider time constraints
        if time_constraint and time_constraint < 24:  # Less than 24 hours
            # Reduce noise for faster convergence (trade privacy for time)
            noise_multiplier *= 0.8
        
        return {
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": max_grad_norm
        }
    
    def _estimate_training_time(
        self, 
        model_size: int, 
        dataset_size: int, 
        batch_size: int
    ) -> float:
        """Estimate training time in hours."""
        
        # Base calculation factors
        base_time_per_sample = 0.001  # seconds per sample (very rough estimate)
        
        # Adjust for model size
        if model_size > 1_000_000_000:  # > 1B parameters
            base_time_per_sample *= 10
        elif model_size > 100_000_000:  # > 100M parameters
            base_time_per_sample *= 3
        
        # Adjust for hardware
        if self.resource_profile.gpu_count > 0:
            gpu_speedup = min(self.resource_profile.gpu_count, 4) * 10
            base_time_per_sample /= gpu_speedup
        
        # Calculate total time
        samples_per_batch = batch_size
        batches_per_epoch = dataset_size // samples_per_batch
        epochs = 3  # Typical number of epochs
        
        total_time_seconds = batches_per_epoch * epochs * base_time_per_sample * samples_per_batch
        total_time_hours = total_time_seconds / 3600
        
        return max(0.1, total_time_hours)  # Minimum 0.1 hours
    
    def monitor_resource_usage(self) -> Dict[str, Any]:
        """Monitor current resource usage."""
        
        if PSUTIL_AVAILABLE:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = (memory.total - memory.available) / (1024**3)
            
            # Storage I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024**2) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024**2) if disk_io else 0
        else:
            # Fallback values when psutil is not available
            cpu_percent = 50.0  # Mock moderate usage
            memory_percent = 60.0  # Mock moderate usage
            memory_used_gb = 4.0  # Mock usage
            disk_read_mb = 0
            disk_write_mb = 0
        
        # GPU usage
        gpu_usage = []
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory_used = torch.cuda.memory_allocated(i) / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_usage.append({
                    "device": i,
                    "memory_used_gb": gpu_memory_used,
                    "memory_total_gb": gpu_memory_total,
                    "memory_percent": (gpu_memory_used / gpu_memory_total) * 100
                })
        
        usage_data = {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_used_gb": memory_used_gb,
            "memory_total_gb": self.resource_profile.total_memory,
            "gpu_usage": gpu_usage,
            "disk_read_mb": disk_read_mb,
            "disk_write_mb": disk_write_mb
        }
        
        # Store for history
        self.resource_history.append(usage_data)
        
        # Keep only recent history (last 100 measurements)
        if len(self.resource_history) > 100:
            self.resource_history = self.resource_history[-100:]
        
        return usage_data
    
    def get_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on monitoring data."""
        recommendations = []
        
        if not self.resource_history:
            return [{"type": "info", "message": "No monitoring data available yet"}]
        
        recent_data = self.resource_history[-10:] if len(self.resource_history) >= 10 else self.resource_history
        
        # Analyze CPU usage
        avg_cpu = sum(data["cpu_percent"] for data in recent_data) / len(recent_data)
        if avg_cpu < 30:
            recommendations.append({
                "type": "performance",
                "message": f"Low CPU usage ({avg_cpu:.1f}%) - consider increasing batch size or parallelism"
            })
        elif avg_cpu > 90:
            recommendations.append({
                "type": "warning",
                "message": f"High CPU usage ({avg_cpu:.1f}%) - consider reducing batch size or workers"
            })
        
        # Analyze memory usage
        avg_memory = sum(data["memory_percent"] for data in recent_data) / len(recent_data)
        if avg_memory > 85:
            recommendations.append({
                "type": "warning", 
                "message": f"High memory usage ({avg_memory:.1f}%) - consider reducing batch size or enabling gradient checkpointing"
            })
        elif avg_memory < 50:
            recommendations.append({
                "type": "performance",
                "message": f"Low memory usage ({avg_memory:.1f}%) - consider increasing batch size for better performance"
            })
        
        # Analyze GPU usage
        if recent_data[-1]["gpu_usage"]:
            for gpu_data in recent_data[-1]["gpu_usage"]:
                gpu_memory_percent = gpu_data["memory_percent"]
                if gpu_memory_percent > 90:
                    recommendations.append({
                        "type": "warning",
                        "message": f"GPU {gpu_data['device']} memory usage high ({gpu_memory_percent:.1f}%) - risk of OOM"
                    })
                elif gpu_memory_percent < 30:
                    recommendations.append({
                        "type": "performance",
                        "message": f"GPU {gpu_data['device']} underutilized ({gpu_memory_percent:.1f}%) - consider larger batch size"
                    })
        
        return recommendations if recommendations else [{"type": "info", "message": "Resource usage looks optimal"}]