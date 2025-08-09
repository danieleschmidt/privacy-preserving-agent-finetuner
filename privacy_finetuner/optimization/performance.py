"""Advanced performance optimization and scalability enhancements.

This module provides comprehensive performance optimizations including intelligent caching,
concurrent processing, memory optimization, and adaptive resource management.
"""

import asyncio
import threading
import multiprocessing
import time
import hashlib
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import weakref
import gc
import logging

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class ProcessingPriority(Enum):
    """Task processing priorities."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry is expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)
    
    def access(self) -> None:
        """Record access to entry."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class IntelligentCache:
    """High-performance intelligent caching system with multiple eviction strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 500,
        default_ttl: Optional[int] = 3600,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        enable_persistence: bool = False,
        persistence_path: Optional[str] = None
    ):
        """Initialize intelligent cache.
        
        Args:
            max_size: Maximum number of cache entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds
            strategy: Cache eviction strategy
            enable_persistence: Enable cache persistence to disk
            persistence_path: Path for cache persistence
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.enable_persistence = enable_persistence
        self.persistence_path = Path(persistence_path or "./cache/intelligent_cache.pkl")
        
        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._access_frequency: Dict[str, int] = defaultdict(int)
        self._current_memory_usage = 0
        self._lock = threading.RLock()
        
        # Performance metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._last_cleanup = datetime.now()
        
        # Auto-tuning parameters
        self._adaptive_params = {
            'hit_rate_threshold': 0.8,
            'memory_pressure_threshold': 0.9,
            'cleanup_interval': 300  # 5 minutes
        }
        
        # Load persisted cache if enabled
        if self.enable_persistence:
            self._load_cache()
        
        # Start background maintenance
        self._start_maintenance_thread()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with intelligent access tracking."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if entry.is_expired():
                    self._remove_entry(key)
                    self._misses += 1
                    return default
                
                # Update access information
                entry.access()
                self._access_frequency[key] += 1
                
                # Move to end for LRU
                if self.strategy in (CacheStrategy.LRU, CacheStrategy.ADAPTIVE):
                    self._cache.move_to_end(key)
                
                self._hits += 1
                logger.debug(f"Cache hit for key: {key}")
                return entry.value
            else:
                self._misses += 1
                logger.debug(f"Cache miss for key: {key}")
                return default
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        priority: ProcessingPriority = ProcessingPriority.NORMAL
    ) -> bool:
        """Put value in cache with intelligent eviction."""
        with self._lock:
            # Calculate value size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024  # Fallback estimate
            
            # Check if value is too large
            if size_bytes > self.max_memory_bytes * 0.5:
                logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return False
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                ttl_seconds=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Ensure space is available
            while (len(self._cache) >= self.max_size or 
                   self._current_memory_usage + size_bytes > self.max_memory_bytes):
                if not self._evict_entry(priority):
                    logger.warning("Cache eviction failed - cache may be at capacity")
                    return False
            
            # Add new entry
            self._cache[key] = entry
            self._current_memory_usage += size_bytes
            self._access_frequency[key] = 1
            
            logger.debug(f"Cache put for key: {key}, size: {size_bytes} bytes")
            return True
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_frequency.clear()
            self._current_memory_usage = 0
            self._evictions = 0
    
    def _remove_entry(self, key: str) -> None:
        """Remove cache entry and update metrics."""
        if key in self._cache:
            entry = self._cache[key]
            self._current_memory_usage -= entry.size_bytes
            del self._cache[key]
            if key in self._access_frequency:
                del self._access_frequency[key]
    
    def _evict_entry(self, preserve_priority: ProcessingPriority = ProcessingPriority.NORMAL) -> bool:
        """Evict entry based on strategy."""
        if not self._cache:
            return False
        
        victim_key = None
        
        if self.strategy == CacheStrategy.LRU:
            # Least recently used (first item in OrderedDict)
            victim_key = next(iter(self._cache))
            
        elif self.strategy == CacheStrategy.LFU:
            # Least frequently used
            min_frequency = min(self._access_frequency.values())
            for key, freq in self._access_frequency.items():
                if freq == min_frequency and key in self._cache:
                    victim_key = key
                    break
                    
        elif self.strategy == CacheStrategy.TTL:
            # Expired entries first, then oldest
            now = datetime.now()
            for key, entry in self._cache.items():
                if entry.is_expired():
                    victim_key = key
                    break
            
            if victim_key is None:
                victim_key = next(iter(self._cache))
                
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy based on access patterns
            victim_key = self._adaptive_eviction()
        
        if victim_key:
            self._remove_entry(victim_key)
            self._evictions += 1
            logger.debug(f"Evicted cache entry: {victim_key}")
            return True
        
        return False
    
    def _adaptive_eviction(self) -> Optional[str]:
        """Adaptive eviction based on access patterns and memory pressure."""
        if not self._cache:
            return None
        
        # Calculate memory pressure
        memory_pressure = self._current_memory_usage / self.max_memory_bytes
        
        # If high memory pressure, prefer size-based eviction
        if memory_pressure > self._adaptive_params['memory_pressure_threshold']:
            return max(self._cache.keys(), key=lambda k: self._cache[k].size_bytes)
        
        # Calculate hit rate
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        
        # If low hit rate, use LFU strategy
        if hit_rate < self._adaptive_params['hit_rate_threshold']:
            min_frequency = min(self._access_frequency.values())
            for key, freq in self._access_frequency.items():
                if freq == min_frequency and key in self._cache:
                    return key
        
        # Default to LRU
        return next(iter(self._cache))
    
    def _cleanup_expired(self) -> int:
        """Clean up expired entries."""
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def _start_maintenance_thread(self) -> None:
        """Start background maintenance thread."""
        def maintenance_loop():
            while True:
                try:
                    time.sleep(self._adaptive_params['cleanup_interval'])
                    
                    with self._lock:
                        # Clean up expired entries
                        self._cleanup_expired()
                        
                        # Save cache if persistence is enabled
                        if self.enable_persistence:
                            self._save_cache()
                        
                        # Update last cleanup time
                        self._last_cleanup = datetime.now()
                        
                except Exception as e:
                    logger.error(f"Cache maintenance error: {e}")
        
        maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
        maintenance_thread.start()
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            
            cache_data = {
                'entries': dict(self._cache),
                'frequency': dict(self._access_frequency),
                'metrics': {
                    'hits': self._hits,
                    'misses': self._misses,
                    'evictions': self._evictions
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            if not self.persistence_path.exists():
                return
            
            with open(self.persistence_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Restore cache entries (skip expired ones)
            now = datetime.now()
            for key, entry in cache_data['entries'].items():
                if not entry.is_expired():
                    self._cache[key] = entry
                    self._current_memory_usage += entry.size_bytes
            
            # Restore access frequency
            self._access_frequency.update(cache_data['frequency'])
            
            # Restore metrics
            metrics = cache_data.get('metrics', {})
            self._hits = metrics.get('hits', 0)
            self._misses = metrics.get('misses', 0)
            self._evictions = metrics.get('evictions', 0)
            
            logger.info(f"Loaded {len(self._cache)} cache entries from disk")
            
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            self.clear()  # Clear corrupted cache
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'entries': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_bytes': self._current_memory_usage,
                'memory_usage_mb': self._current_memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_utilization': self._current_memory_usage / self.max_memory_bytes,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'strategy': self.strategy.value,
                'last_cleanup': self._last_cleanup.isoformat()
            }


class ConcurrentProcessor:
    """High-performance concurrent processing system with adaptive scaling."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        thread_pool_size: Optional[int] = None,
        process_pool_size: Optional[int] = None,
        enable_auto_scaling: bool = True,
        queue_size_limit: int = 1000
    ):
        """Initialize concurrent processor.
        
        Args:
            max_workers: Maximum total workers (threads + processes)
            thread_pool_size: Size of thread pool
            process_pool_size: Size of process pool
            enable_auto_scaling: Enable automatic scaling based on load
            queue_size_limit: Maximum queued tasks
        """
        self.max_workers = max_workers or (multiprocessing.cpu_count() * 2)
        self.thread_pool_size = thread_pool_size or min(32, self.max_workers)
        self.process_pool_size = process_pool_size or multiprocessing.cpu_count()
        self.enable_auto_scaling = enable_auto_scaling
        self.queue_size_limit = queue_size_limit
        
        # Executor pools
        self.thread_executor = ThreadPoolExecutor(max_workers=self.thread_pool_size)
        self.process_executor = ProcessPoolExecutor(max_workers=self.process_pool_size)
        
        # Task tracking
        self._pending_tasks: Dict[str, Future] = {}
        self._completed_tasks: Dict[str, Any] = {}
        self._task_stats = defaultdict(int)
        self._lock = threading.Lock()
        
        # Performance metrics
        self._start_time = datetime.now()
        self._total_tasks = 0
        self._successful_tasks = 0
        self._failed_tasks = 0
        
        logger.info(f"Concurrent processor initialized: {self.thread_pool_size} threads, {self.process_pool_size} processes")
    
    def submit_task(
        self,
        func: Callable,
        *args,
        use_process: bool = False,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        timeout: Optional[float] = None,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """Submit task for concurrent execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            use_process: Use process pool instead of thread pool
            priority: Task priority
            timeout: Task timeout in seconds
            callback: Callback function for completion
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID for tracking
        """
        # Generate unique task ID
        task_id = hashlib.md5(
            f"{func.__name__}_{time.time()}_{id(args)}".encode()
        ).hexdigest()[:12]
        
        # Check queue limit
        with self._lock:
            if len(self._pending_tasks) >= self.queue_size_limit:
                raise RuntimeError(f"Task queue full (limit: {self.queue_size_limit})")
        
        # Choose executor
        executor = self.process_executor if use_process else self.thread_executor
        
        # Submit task
        try:
            future = executor.submit(self._execute_task, func, args, kwargs, task_id, timeout)
            
            # Add completion callback if provided
            if callback:
                future.add_done_callback(lambda f: self._handle_completion(f, callback, task_id))
            
            with self._lock:
                self._pending_tasks[task_id] = future
                self._total_tasks += 1
                self._task_stats['submitted'] += 1
            
            logger.debug(f"Submitted task {task_id} to {'process' if use_process else 'thread'} pool")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise
    
    def _execute_task(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        task_id: str,
        timeout: Optional[float]
    ) -> Any:
        """Execute task with error handling and timeout."""
        start_time = time.time()
        
        try:
            # Set timeout if specified
            if timeout:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Clear timeout
            if timeout:
                signal.alarm(0)
            
            execution_time = time.time() - start_time
            
            with self._lock:
                self._completed_tasks[task_id] = {
                    'result': result,
                    'execution_time': execution_time,
                    'status': 'success',
                    'timestamp': datetime.now()
                }
                self._successful_tasks += 1
            
            logger.debug(f"Task {task_id} completed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            with self._lock:
                self._completed_tasks[task_id] = {
                    'error': str(e),
                    'execution_time': execution_time,
                    'status': 'failed',
                    'timestamp': datetime.now()
                }
                self._failed_tasks += 1
            
            logger.error(f"Task {task_id} failed after {execution_time:.2f}s: {e}")
            raise
        finally:
            # Clean up timeout
            if timeout:
                try:
                    signal.alarm(0)
                except:
                    pass
    
    def _handle_completion(self, future: Future, callback: Callable, task_id: str) -> None:
        """Handle task completion callback."""
        try:
            result = future.result()
            callback(task_id, result, None)
        except Exception as e:
            callback(task_id, None, e)
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result (blocking)."""
        with self._lock:
            if task_id in self._pending_tasks:
                future = self._pending_tasks[task_id]
            else:
                # Check completed tasks
                if task_id in self._completed_tasks:
                    task_result = self._completed_tasks[task_id]
                    if task_result['status'] == 'success':
                        return task_result['result']
                    else:
                        raise RuntimeError(f"Task failed: {task_result['error']}")
                else:
                    raise KeyError(f"Task not found: {task_id}")
        
        # Wait for completion
        try:
            result = future.result(timeout=timeout)
            
            # Clean up
            with self._lock:
                if task_id in self._pending_tasks:
                    del self._pending_tasks[task_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get result for task {task_id}: {e}")
            raise
    
    def get_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status."""
        with self._lock:
            if task_id in self._pending_tasks:
                future = self._pending_tasks[task_id]
                return {
                    'task_id': task_id,
                    'status': 'running' if future.running() else 'pending',
                    'done': future.done()
                }
            elif task_id in self._completed_tasks:
                return {
                    'task_id': task_id,
                    'status': self._completed_tasks[task_id]['status'],
                    'done': True,
                    'completion_time': self._completed_tasks[task_id]['timestamp'].isoformat(),
                    'execution_time': self._completed_tasks[task_id]['execution_time']
                }
            else:
                return {'task_id': task_id, 'status': 'not_found'}
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel pending task."""
        with self._lock:
            if task_id in self._pending_tasks:
                future = self._pending_tasks[task_id]
                success = future.cancel()
                if success:
                    del self._pending_tasks[task_id]
                    self._task_stats['cancelled'] += 1
                return success
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._lock:
            uptime = (datetime.now() - self._start_time).total_seconds()
            throughput = self._successful_tasks / uptime if uptime > 0 else 0
            
            return {
                'uptime_seconds': uptime,
                'total_tasks': self._total_tasks,
                'successful_tasks': self._successful_tasks,
                'failed_tasks': self._failed_tasks,
                'pending_tasks': len(self._pending_tasks),
                'success_rate': self._successful_tasks / self._total_tasks if self._total_tasks > 0 else 0,
                'throughput_per_second': throughput,
                'thread_pool_size': self.thread_pool_size,
                'process_pool_size': self.process_pool_size,
                'queue_utilization': len(self._pending_tasks) / self.queue_size_limit
            }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown all executors."""
        logger.info("Shutting down concurrent processor...")
        
        self.thread_executor.shutdown(wait=wait)
        self.process_executor.shutdown(wait=wait)
        
        with self._lock:
            self._pending_tasks.clear()
        
        logger.info("Concurrent processor shutdown complete")


class MemoryOptimizer:
    """Advanced memory optimization and management."""
    
    def __init__(self, gc_threshold: float = 0.8, optimization_interval: int = 60):
        """Initialize memory optimizer.
        
        Args:
            gc_threshold: Trigger GC when memory usage exceeds this threshold
            optimization_interval: Optimization check interval in seconds
        """
        self.gc_threshold = gc_threshold
        self.optimization_interval = optimization_interval
        self._running = True
        
        # Memory tracking
        self._peak_memory = 0
        self._gc_count = 0
        self._optimization_count = 0
        
        # Weak references for auto-cleanup
        self._tracked_objects = weakref.WeakSet()
        
        # Start monitoring thread
        self._start_monitoring()
    
    def _start_monitoring(self) -> None:
        """Start memory monitoring thread."""
        def monitor_loop():
            while self._running:
                try:
                    self._check_memory_pressure()
                    time.sleep(self.optimization_interval)
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _check_memory_pressure(self) -> None:
        """Check memory pressure and optimize if needed."""
        try:
            # Get memory usage (fallback if psutil not available)
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent / 100
            except ImportError:
                memory_percent = 0.5  # Assume moderate usage
            
            # Update peak memory
            if memory_percent > self._peak_memory:
                self._peak_memory = memory_percent
            
            # Trigger optimization if threshold exceeded
            if memory_percent > self.gc_threshold:
                logger.info(f"Memory pressure detected: {memory_percent:.1%}")
                self.optimize_memory()
                
        except Exception as e:
            logger.error(f"Memory pressure check failed: {e}")
    
    def optimize_memory(self) -> Dict[str, int]:
        """Optimize memory usage with comprehensive cleanup."""
        logger.info("Starting memory optimization...")
        
        initial_objects = len(gc.get_objects())
        
        # Force garbage collection
        collected = gc.collect()
        self._gc_count += 1
        
        # Clear weak references
        self._tracked_objects.clear()
        
        # Additional optimization passes
        for generation in range(3):
            gc.collect(generation)
        
        final_objects = len(gc.get_objects())
        freed_objects = initial_objects - final_objects
        
        self._optimization_count += 1
        
        optimization_stats = {
            'objects_collected': collected,
            'objects_freed': freed_objects,
            'gc_cycles': 3,
            'optimization_count': self._optimization_count
        }
        
        logger.info(f"Memory optimization complete: {optimization_stats}")
        return optimization_stats
    
    def track_object(self, obj: Any) -> None:
        """Track object for automatic cleanup."""
        self._tracked_objects.add(obj)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            # Get detailed memory info if available
            try:
                import psutil
                memory = psutil.virtual_memory()
                memory_stats = {
                    'total_mb': memory.total / (1024 * 1024),
                    'available_mb': memory.available / (1024 * 1024),
                    'used_mb': memory.used / (1024 * 1024),
                    'percent': memory.percent
                }
            except ImportError:
                memory_stats = {
                    'total_mb': 0,
                    'available_mb': 0,
                    'used_mb': 0,
                    'percent': 0
                }
            
            # GC statistics
            gc_stats = gc.get_stats()
            
            return {
                'memory': memory_stats,
                'gc_stats': gc_stats,
                'tracked_objects': len(self._tracked_objects),
                'peak_memory_percent': self._peak_memory * 100,
                'gc_count': self._gc_count,
                'optimization_count': self._optimization_count,
                'total_objects': len(gc.get_objects())
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {'error': str(e)}
    
    def shutdown(self) -> None:
        """Shutdown memory optimizer."""
        self._running = False
        self.optimize_memory()  # Final cleanup


# Global instances for easy access
_global_cache: Optional[IntelligentCache] = None
_global_processor: Optional[ConcurrentProcessor] = None
_global_optimizer: Optional[MemoryOptimizer] = None


def get_cache() -> IntelligentCache:
    """Get or create global intelligent cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCache()
    return _global_cache


def get_processor() -> ConcurrentProcessor:
    """Get or create global concurrent processor."""
    global _global_processor
    if _global_processor is None:
        _global_processor = ConcurrentProcessor()
    return _global_processor


def get_memory_optimizer() -> MemoryOptimizer:
    """Get or create global memory optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = MemoryOptimizer()
    return _global_optimizer


# Convenience functions
def cache_result(key: str, func: Callable, *args, ttl: int = 3600, **kwargs) -> Any:
    """Cache function result with intelligent caching."""
    cache = get_cache()
    result = cache.get(key)
    
    if result is None:
        result = func(*args, **kwargs)
        cache.put(key, result, ttl=ttl)
    
    return result


def submit_concurrent_task(
    func: Callable,
    *args,
    use_process: bool = False,
    priority: ProcessingPriority = ProcessingPriority.NORMAL,
    **kwargs
) -> str:
    """Submit task for concurrent execution."""
    processor = get_processor()
    return processor.submit_task(func, *args, use_process=use_process, priority=priority, **kwargs)


def optimize_memory() -> Dict[str, int]:
    """Optimize memory usage."""
    optimizer = get_memory_optimizer()
    return optimizer.optimize_memory()


def get_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary."""
    return {
        'cache': get_cache().get_stats(),
        'processor': get_processor().get_performance_stats(),
        'memory': get_memory_optimizer().get_memory_stats(),
        'timestamp': datetime.now().isoformat()
    }