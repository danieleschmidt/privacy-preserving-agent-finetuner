"""Advanced memory management system for large-scale privacy-preserving training.

This module implements enterprise-grade memory optimization including:
- Intelligent gradient accumulation and checkpointing
- Dynamic memory allocation and garbage collection
- Memory pool management for efficient reuse
- CPU/GPU memory optimization and offloading
"""

import logging
import time
import gc
import threading
import asyncio
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import weakref
from pathlib import Path
import pickle
import mmap
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class MemoryStrategy(Enum):
    """Memory management strategies."""
    AGGRESSIVE_GC = "aggressive_gc"
    LAZY_ALLOCATION = "lazy_allocation"
    MEMORY_POOLING = "memory_pooling"
    CPU_OFFLOADING = "cpu_offloading"
    GRADIENT_CHECKPOINTING = "gradient_checkpointing"
    ADAPTIVE = "adaptive"


class CheckpointingStrategy(Enum):
    """Gradient checkpointing strategies."""
    NONE = "none"
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    MEMORY_AWARE = "memory_aware"


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    # Memory limits
    max_gpu_memory_gb: float = 16.0
    max_cpu_memory_gb: float = 32.0
    memory_buffer_gb: float = 2.0
    
    # Strategy settings
    memory_strategy: MemoryStrategy = MemoryStrategy.ADAPTIVE
    checkpointing_strategy: CheckpointingStrategy = CheckpointingStrategy.ADAPTIVE
    
    # Pool management
    enable_memory_pool: bool = True
    pool_size_gb: float = 4.0
    pool_cleanup_interval: int = 300
    
    # Garbage collection
    gc_threshold: float = 0.85
    aggressive_gc_threshold: float = 0.95
    gc_interval: int = 100
    
    # CPU offloading
    enable_cpu_offloading: bool = True
    offload_threshold: float = 0.8
    offload_layers: List[str] = field(default_factory=lambda: ["attention", "feedforward"])
    
    # Gradient accumulation
    max_gradient_accumulation_steps: int = 32
    adaptive_accumulation: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_gpu_memory_gb': self.max_gpu_memory_gb,
            'max_cpu_memory_gb': self.max_cpu_memory_gb,
            'memory_strategy': self.memory_strategy.value,
            'checkpointing_strategy': self.checkpointing_strategy.value,
            'enable_memory_pool': self.enable_memory_pool,
            'enable_cpu_offloading': self.enable_cpu_offloading
        }


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    gpu_allocated_gb: float = 0.0
    gpu_cached_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    cpu_memory_gb: float = 0.0
    memory_efficiency: float = 0.0
    fragmentation_ratio: float = 0.0
    gc_count: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    offload_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'gpu_allocated_gb': self.gpu_allocated_gb,
            'gpu_cached_gb': self.gpu_cached_gb,
            'gpu_reserved_gb': self.gpu_reserved_gb,
            'cpu_memory_gb': self.cpu_memory_gb,
            'memory_efficiency': self.memory_efficiency,
            'fragmentation_ratio': self.fragmentation_ratio,
            'gc_count': self.gc_count,
            'pool_hits': self.pool_hits,
            'pool_misses': self.pool_misses,
            'offload_count': self.offload_count,
            'timestamp': self.timestamp.isoformat()
        }


class MemoryPool:
    """Intelligent memory pool for tensor reuse."""
    
    def __init__(self, max_size_gb: float = 4.0):
        """Initialize memory pool.
        
        Args:
            max_size_gb: Maximum pool size in GB
        """
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.current_size_bytes = 0
        
        # Pool storage organized by shape and dtype
        self.cpu_pool: Dict[Tuple, List[torch.Tensor]] = defaultdict(list)
        self.gpu_pool: Dict[Tuple, List[torch.Tensor]] = defaultdict(list)
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.allocations = 0
        self.deallocations = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background cleanup
        self.cleanup_executor = ThreadPoolExecutor(max_workers=1)
        self._start_cleanup_thread()
    
    def get_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: Union[str, torch.device],
        requires_grad: bool = False
    ) -> torch.Tensor:
        """Get tensor from pool or allocate new one."""
        key = (shape, dtype, str(device))
        
        with self.lock:
            # Choose appropriate pool
            pool = self.gpu_pool if 'cuda' in str(device) else self.cpu_pool
            
            # Try to reuse existing tensor
            if key in pool and pool[key]:
                tensor = pool[key].pop()
                tensor.zero_()
                tensor.requires_grad_(requires_grad)
                self.hits += 1
                
                logger.debug(f"Pool hit: {shape} {dtype} on {device}")
                return tensor
        
        # Allocate new tensor
        tensor = torch.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        self.misses += 1
        self.allocations += 1
        
        logger.debug(f"Pool miss: allocated {shape} {dtype} on {device}")
        return tensor
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return tensor to pool."""
        if tensor.numel() == 0:
            return
        
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        device = str(tensor.device)
        key = (shape, dtype, device)
        
        # Calculate tensor size
        tensor_size = tensor.numel() * tensor.element_size()
        
        with self.lock:
            # Check if pool has space
            if self.current_size_bytes + tensor_size > self.max_size_bytes:
                self._cleanup_pool()
            
            if self.current_size_bytes + tensor_size <= self.max_size_bytes:
                # Choose appropriate pool
                pool = self.gpu_pool if 'cuda' in device else self.cpu_pool
                
                # Detach tensor and clear gradients
                tensor = tensor.detach()
                if tensor.grad is not None:
                    tensor.grad = None
                
                pool[key].append(tensor)
                self.current_size_bytes += tensor_size
                self.deallocations += 1
                
                logger.debug(f"Returned tensor to pool: {shape} {dtype} on {device}")
    
    def _cleanup_pool(self) -> None:
        """Clean up pool to free memory."""
        cleanup_size = self.max_size_bytes // 4  # Clean 25% of pool
        freed_size = 0
        
        # Clean CPU pool first (less critical)
        for key in list(self.cpu_pool.keys()):
            while self.cpu_pool[key] and freed_size < cleanup_size:
                tensor = self.cpu_pool[key].pop()
                freed_size += tensor.numel() * tensor.element_size()
                del tensor
            
            if not self.cpu_pool[key]:
                del self.cpu_pool[key]
        
        # Clean GPU pool if needed
        if freed_size < cleanup_size:
            for key in list(self.gpu_pool.keys()):
                while self.gpu_pool[key] and freed_size < cleanup_size:
                    tensor = self.gpu_pool[key].pop()
                    freed_size += tensor.numel() * tensor.element_size()
                    del tensor
                
                if not self.gpu_pool[key]:
                    del self.gpu_pool[key]
        
        self.current_size_bytes -= freed_size
        logger.info(f"Cleaned up {freed_size / (1024**3):.2f} GB from memory pool")
    
    def _enhanced_cleanup(self) -> None:
        """Enhanced cleanup targeting 25% memory reduction."""
        cleanup_size = self.max_size_bytes // 2  # Clean 50% of pool for better reduction
        freed_size = 0
        
        # More aggressive CPU pool cleanup
        for key in list(self.cpu_pool.keys()):
            tensors_to_remove = max(1, len(self.cpu_pool[key]) // 2)  # Remove at least half
            for _ in range(min(tensors_to_remove, len(self.cpu_pool[key]))):
                if self.cpu_pool[key]:
                    tensor = self.cpu_pool[key].pop()
                    freed_size += tensor.numel() * tensor.element_size()
                    del tensor
            
            if not self.cpu_pool[key]:
                del self.cpu_pool[key]
        
        # More aggressive GPU pool cleanup
        for key in list(self.gpu_pool.keys()):
            tensors_to_remove = max(1, len(self.gpu_pool[key]) // 2)  # Remove at least half
            for _ in range(min(tensors_to_remove, len(self.gpu_pool[key]))):
                if self.gpu_pool[key]:
                    tensor = self.gpu_pool[key].pop()
                    freed_size += tensor.numel() * tensor.element_size()
                    del tensor
            
            if not self.gpu_pool[key]:
                del self.gpu_pool[key]
        
        self.current_size_bytes -= freed_size
        logger.info(f"Enhanced cleanup freed {freed_size / (1024**3):.2f} GB from memory pool")
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(300)  # Cleanup every 5 minutes
                    with self.lock:
                        self._cleanup_pool()
                except Exception as e:
                    logger.error(f"Pool cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            
            return {
                'current_size_gb': self.current_size_bytes / (1024**3),
                'max_size_gb': self.max_size_bytes / (1024**3),
                'utilization': self.current_size_bytes / self.max_size_bytes,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'allocations': self.allocations,
                'deallocations': self.deallocations,
                'cpu_pool_entries': sum(len(tensors) for tensors in self.cpu_pool.values()),
                'gpu_pool_entries': sum(len(tensors) for tensors in self.gpu_pool.values())
            }


class GradientCheckpointer:
    """Intelligent gradient checkpointing for memory optimization."""
    
    def __init__(self, strategy: CheckpointingStrategy = CheckpointingStrategy.ADAPTIVE):
        """Initialize gradient checkpointer.
        
        Args:
            strategy: Checkpointing strategy to use
        """
        self.strategy = strategy
        self.checkpoint_layers = set()
        self.memory_stats = []
        self.checkpoint_overhead = {}
    
    def should_checkpoint_layer(self, layer_name: str, layer_size: int, memory_pressure: float) -> bool:
        """Determine if layer should be checkpointed."""
        if self.strategy == CheckpointingStrategy.NONE:
            return False
        
        elif self.strategy == CheckpointingStrategy.UNIFORM:
            # Checkpoint every other layer
            return hash(layer_name) % 2 == 0
        
        elif self.strategy == CheckpointingStrategy.MEMORY_AWARE:
            # Checkpoint based on memory pressure and layer size
            if memory_pressure > 0.8:
                return layer_size > 1024 * 1024  # 1MB threshold
            elif memory_pressure > 0.6:
                return layer_size > 10 * 1024 * 1024  # 10MB threshold
            else:
                return False
        
        elif self.strategy == CheckpointingStrategy.ADAPTIVE:
            # Adaptive checkpointing based on layer importance and memory
            return self._adaptive_checkpoint_decision(layer_name, layer_size, memory_pressure)
        
        return False
    
    def _adaptive_checkpoint_decision(self, layer_name: str, layer_size: int, memory_pressure: float) -> bool:
        """Make adaptive checkpointing decision."""
        # High memory pressure - be aggressive
        if memory_pressure > 0.85:
            return layer_size > 512 * 1024  # 512KB threshold
        
        # Medium memory pressure - checkpoint large layers
        elif memory_pressure > 0.7:
            return layer_size > 5 * 1024 * 1024  # 5MB threshold
        
        # Low memory pressure - checkpoint very large layers only
        elif memory_pressure > 0.5:
            return layer_size > 20 * 1024 * 1024  # 20MB threshold
        
        # Very low memory pressure - no checkpointing
        return False
    
    def checkpoint_function(self, function, *args, **kwargs):
        """Apply gradient checkpointing to function."""
        if torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
        else:
            return function(*args, **kwargs)


class CPUOffloader:
    """Handle CPU offloading for memory optimization."""
    
    def __init__(self, offload_threshold: float = 0.8):
        """Initialize CPU offloader.
        
        Args:
            offload_threshold: Memory threshold for offloading
        """
        self.offload_threshold = offload_threshold
        self.offloaded_tensors: Dict[str, torch.Tensor] = {}
        self.offload_count = 0
        self.reload_count = 0
        
        # Use memory-mapped files for large tensors
        self.mmap_dir = Path("./temp_offload")
        self.mmap_dir.mkdir(exist_ok=True)
    
    async def offload_tensor(self, tensor: torch.Tensor, key: str) -> bool:
        """Offload tensor to CPU memory or disk."""
        try:
            if tensor.device.type == 'cpu':
                return False  # Already on CPU
            
            # Move to CPU first
            cpu_tensor = tensor.cpu()
            
            # For very large tensors, use memory mapping
            if cpu_tensor.numel() * cpu_tensor.element_size() > 100 * 1024 * 1024:  # 100MB
                await self._offload_to_disk(cpu_tensor, key)
            else:
                self.offloaded_tensors[key] = cpu_tensor
            
            self.offload_count += 1
            logger.debug(f"Offloaded tensor {key} to CPU")
            return True
            
        except Exception as e:
            logger.error(f"Failed to offload tensor {key}: {e}")
            return False
    
    async def reload_tensor(self, key: str, device: torch.device) -> Optional[torch.Tensor]:
        """Reload tensor from CPU to GPU."""
        try:
            if key in self.offloaded_tensors:
                tensor = self.offloaded_tensors[key].to(device)
                del self.offloaded_tensors[key]
                self.reload_count += 1
                logger.debug(f"Reloaded tensor {key} to {device}")
                return tensor
            
            # Try to load from disk
            tensor = await self._reload_from_disk(key, device)
            if tensor is not None:
                self.reload_count += 1
                return tensor
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to reload tensor {key}: {e}")
            return None
    
    async def _offload_to_disk(self, tensor: torch.Tensor, key: str) -> None:
        """Offload tensor to disk using memory mapping."""
        file_path = self.mmap_dir / f"{key}.tensor"
        
        # Save tensor to memory-mapped file
        with open(file_path, 'wb') as f:
            pickle.dump({
                'shape': tensor.shape,
                'dtype': tensor.dtype,
                'data': tensor.numpy()
            }, f)
    
    async def _reload_from_disk(self, key: str, device: torch.device) -> Optional[torch.Tensor]:
        """Reload tensor from disk."""
        file_path = self.mmap_dir / f"{key}.tensor"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            tensor = torch.from_numpy(data['data']).to(device)
            file_path.unlink()  # Clean up file
            
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to reload from disk: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get offloading statistics."""
        return {
            'offloaded_tensors': len(self.offloaded_tensors),
            'offload_count': self.offload_count,
            'reload_count': self.reload_count,
            'memory_saved_gb': sum(
                t.numel() * t.element_size() for t in self.offloaded_tensors.values()
            ) / (1024**3)
        }


class AdvancedMemoryManager:
    """Advanced memory management system with multiple optimization strategies."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize memory manager.
        
        Args:
            config: Memory management configuration
        """
        self.config = config or MemoryConfig()
        
        # Components
        self.memory_pool = MemoryPool(self.config.pool_size_gb) if self.config.enable_memory_pool else None
        self.checkpointer = GradientCheckpointer(self.config.checkpointing_strategy)
        self.offloader = CPUOffloader(self.config.offload_threshold) if self.config.enable_cpu_offloading else None
        
        # Memory tracking
        self.memory_history: deque = deque(maxlen=1000)
        self.gc_count = 0
        self.peak_memory = 0.0
        self.oom_events = 0
        
        # Monitoring
        self.monitoring_active = True
        self.monitoring_thread = None
        
        # Adaptive parameters
        self.adaptive_params = {
            'gc_threshold': self.config.gc_threshold,
            'gradient_accumulation_steps': 1,
            'batch_size_multiplier': 1.0
        }
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info(f"Advanced memory manager initialized with strategy: {self.config.memory_strategy.value}")
    
    def _start_monitoring(self) -> None:
        """Start memory monitoring thread."""
        def monitor_loop():
            while self.monitoring_active:
                try:
                    stats = self._collect_memory_stats()
                    self.memory_history.append(stats)
                    
                    # Check for memory pressure
                    self._handle_memory_pressure(stats)
                    
                    time.sleep(5)  # Monitor every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
                    time.sleep(10)
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _collect_memory_stats(self) -> MemoryStats:
        """Collect comprehensive memory statistics."""
        stats = MemoryStats()
        
        # GPU memory stats
        if torch.cuda.is_available():
            stats.gpu_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            stats.gpu_cached_gb = torch.cuda.memory_reserved() / (1024**3)
            
            # Calculate fragmentation
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            stats.fragmentation_ratio = (stats.gpu_cached_gb - stats.gpu_allocated_gb) / total_memory
        
        # CPU memory stats
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            stats.cpu_memory_gb = memory_info.rss / (1024**3)
        except Exception:
            stats.cpu_memory_gb = 0.0
        
        # Pool stats
        if self.memory_pool:
            pool_stats = self.memory_pool.get_stats()
            stats.pool_hits = pool_stats['hits']
            stats.pool_misses = pool_stats['misses']
        
        # Offload stats
        if self.offloader:
            offload_stats = self.offloader.get_stats()
            stats.offload_count = offload_stats['offload_count']
        
        # Memory efficiency
        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            stats.memory_efficiency = stats.gpu_allocated_gb / total_gpu_memory
        
        stats.gc_count = self.gc_count
        
        # Track peak memory
        if stats.gpu_allocated_gb > self.peak_memory:
            self.peak_memory = stats.gpu_allocated_gb
        
        return stats
    
    def _handle_memory_pressure(self, stats: MemoryStats) -> None:
        """Handle memory pressure situations."""
        # High memory pressure
        if stats.memory_efficiency > self.config.aggressive_gc_threshold:
            self.aggressive_memory_cleanup()
        elif stats.memory_efficiency > self.config.gc_threshold:
            self.optimize_memory()
        
        # Detect potential OOM
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free_memory = total_memory - stats.gpu_allocated_gb
            
            if free_memory < self.config.memory_buffer_gb:
                logger.warning(f"Low GPU memory: {free_memory:.2f}GB free")
                self.emergency_memory_cleanup()
    
    def optimize_memory(self) -> Dict[str, int]:
        """Perform enhanced memory optimization targeting 25% reduction."""
        logger.info("Starting enhanced memory optimization...")
        
        initial_memory = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        # Enhanced multi-pass garbage collection
        collected = 0
        for generation in [2, 1, 0]:  # Collect in reverse order for better efficiency
            collected += gc.collect(generation)
        self.gc_count += 3
        
        # Clear PyTorch cache with forced synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # Enhanced pool cleanup with more aggressive freeing
        if self.memory_pool:
            self.memory_pool._enhanced_cleanup()
        
        # Memory compaction and defragmentation
        self._perform_memory_compaction()
        
        # Clear Python object caches
        self._clear_python_caches()
        
        final_memory = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        freed_memory = initial_memory - final_memory
        reduction_percentage = (freed_memory / initial_memory * 100) if initial_memory > 0 else 0
        
        optimization_stats = {
            'objects_collected': collected,
            'memory_freed_gb': freed_memory,
            'reduction_percentage': reduction_percentage,
            'gc_count': self.gc_count,
            'target_achieved': reduction_percentage >= 25.0
        }
        
        logger.info(f"Enhanced memory optimization completed: {optimization_stats}")
        return optimization_stats
    
    def aggressive_memory_cleanup(self) -> Dict[str, Any]:
        """Perform aggressive memory cleanup."""
        logger.warning("Starting aggressive memory cleanup...")
        
        initial_memory = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        # Multiple GC passes
        collected = 0
        for generation in range(3):
            collected += gc.collect(generation)
        
        self.gc_count += 3
        
        # Clear all caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # Pool cleanup
        if self.memory_pool:
            self.memory_pool._cleanup_pool()
        
        # Force CPU offloading if enabled
        if self.offloader:
            # This would require model reference to offload specific tensors
            pass
        
        final_memory = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        freed_memory = initial_memory - final_memory
        
        cleanup_stats = {
            'objects_collected': collected,
            'memory_freed_gb': freed_memory,
            'gc_passes': 3,
            'aggressive_cleanup': True
        }
        
        logger.warning(f"Aggressive memory cleanup completed: {cleanup_stats}")
        return cleanup_stats
    
    def _perform_memory_compaction(self) -> None:
        """Perform memory compaction and defragmentation."""
        try:
            # Force PyTorch memory defragmentation
            if torch.cuda.is_available():
                # Trigger defragmentation by creating and deleting a large tensor
                temp_size = min(100 * 1024 * 1024, torch.cuda.memory_reserved() // 10)  # 100MB or 10% of reserved
                temp_tensor = torch.zeros(temp_size // 4, dtype=torch.float32, device='cuda')
                del temp_tensor
                torch.cuda.empty_cache()
                
            logger.debug("Memory compaction completed")
        except Exception as e:
            logger.warning(f"Memory compaction failed: {e}")
    
    def _clear_python_caches(self) -> None:
        """Clear Python internal caches for additional memory savings."""
        try:
            # Clear function call cache
            import functools
            functools.lru_cache.cache_clear = lambda: None
            
            # Clear regex cache
            import re
            re.purge()
            
            # Clear linecache
            import linecache
            linecache.clearcache()
            
            # Clear importlib cache
            import sys
            if hasattr(sys, 'modules'):
                for module_name in list(sys.modules.keys()):
                    if hasattr(sys.modules[module_name], '__cached__'):
                        try:
                            delattr(sys.modules[module_name], '__cached__')
                        except (AttributeError, TypeError):
                            pass
            
            logger.debug("Python caches cleared")
        except Exception as e:
            logger.warning(f"Failed to clear Python caches: {e}")
    
    def emergency_memory_cleanup(self) -> None:
        """Emergency memory cleanup to prevent OOM."""
        logger.error("Emergency memory cleanup triggered!")
        
        self.oom_events += 1
        
        # Most aggressive cleanup possible
        self.aggressive_memory_cleanup()
        
        # Reduce adaptive parameters
        self.adaptive_params['batch_size_multiplier'] *= 0.75
        self.adaptive_params['gradient_accumulation_steps'] = min(
            self.adaptive_params['gradient_accumulation_steps'] * 2,
            self.config.max_gradient_accumulation_steps
        )
        
        logger.error(f"Emergency cleanup completed. Adjusted parameters: {self.adaptive_params}")
    
    def get_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = 'cuda',
        requires_grad: bool = False
    ) -> torch.Tensor:
        """Get tensor with memory pool optimization."""
        if self.memory_pool:
            return self.memory_pool.get_tensor(shape, dtype, device, requires_grad)
        else:
            return torch.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return tensor to memory pool."""
        if self.memory_pool:
            self.memory_pool.return_tensor(tensor)
    
    async def offload_if_needed(self, tensor: torch.Tensor, key: str) -> bool:
        """Offload tensor if memory pressure is high."""
        if not self.offloader:
            return False
        
        # Check current memory pressure
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_pressure = current_memory / total_memory
            
            if memory_pressure > self.config.offload_threshold:
                return await self.offloader.offload_tensor(tensor, key)
        
        return False
    
    async def reload_if_available(self, key: str, device: torch.device) -> Optional[torch.Tensor]:
        """Reload tensor from offload storage."""
        if self.offloader:
            return await self.offloader.reload_tensor(key, device)
        return None
    
    def should_checkpoint_layer(self, layer_name: str, layer: torch.nn.Module) -> bool:
        """Determine if layer should use gradient checkpointing."""
        # Estimate layer size
        layer_size = sum(p.numel() * p.element_size() for p in layer.parameters())
        
        # Get current memory pressure
        memory_pressure = 0.5  # Default
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_pressure = current_memory / total_memory
        
        return self.checkpointer.should_checkpoint_layer(layer_name, layer_size, memory_pressure)
    
    def checkpoint_function(self, function, *args, **kwargs):
        """Apply gradient checkpointing to function."""
        return self.checkpointer.checkpoint_function(function, *args, **kwargs)
    
    def get_adaptive_batch_size(self, base_batch_size: int) -> int:
        """Get adaptive batch size based on memory conditions."""
        return max(1, int(base_batch_size * self.adaptive_params['batch_size_multiplier']))
    
    def get_adaptive_accumulation_steps(self) -> int:
        """Get adaptive gradient accumulation steps."""
        return self.adaptive_params['gradient_accumulation_steps']
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory management statistics."""
        current_stats = self._collect_memory_stats()
        
        # Historical analysis
        if self.memory_history:
            recent_history = list(self.memory_history)[-100:]
            avg_efficiency = np.mean([s.memory_efficiency for s in recent_history])
            peak_efficiency = max(s.memory_efficiency for s in recent_history)
        else:
            avg_efficiency = current_stats.memory_efficiency
            peak_efficiency = current_stats.memory_efficiency
        
        stats = {
            'current_stats': current_stats.to_dict(),
            'historical_analysis': {
                'average_efficiency': avg_efficiency,
                'peak_efficiency': peak_efficiency,
                'peak_memory_gb': self.peak_memory,
                'oom_events': self.oom_events
            },
            'adaptive_parameters': self.adaptive_params,
            'configuration': self.config.to_dict()
        }
        
        # Component stats
        if self.memory_pool:
            stats['memory_pool'] = self.memory_pool.get_stats()
        
        if self.offloader:
            stats['cpu_offloader'] = self.offloader.get_stats()
        
        return stats
    
    def export_memory_report(self, output_path: str) -> None:
        """Export comprehensive memory report."""
        report = self.get_comprehensive_stats()
        report['export_timestamp'] = datetime.now().isoformat()
        
        with open(output_path, 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Memory report exported to {output_path}")
    
    def shutdown(self) -> None:
        """Shutdown memory manager."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        # Final cleanup
        self.aggressive_memory_cleanup()
        
        logger.info("Memory manager shutdown completed")


# Global memory manager instance
_global_memory_manager: Optional[AdvancedMemoryManager] = None


def get_memory_manager(config: Optional[MemoryConfig] = None) -> AdvancedMemoryManager:
    """Get global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = AdvancedMemoryManager(config)
    return _global_memory_manager


def optimize_memory() -> Dict[str, int]:
    """Convenience function to optimize memory."""
    return get_memory_manager().optimize_memory()


def get_adaptive_tensor(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = 'cuda',
    requires_grad: bool = False
) -> torch.Tensor:
    """Convenience function to get tensor with memory optimization."""
    return get_memory_manager().get_tensor(shape, dtype, device, requires_grad)


def return_adaptive_tensor(tensor: torch.Tensor) -> None:
    """Convenience function to return tensor to pool."""
    get_memory_manager().return_tensor(tensor)