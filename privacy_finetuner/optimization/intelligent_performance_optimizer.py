"""
Intelligent Performance Optimizer for Privacy-Preserving ML
Advanced performance optimization with privacy-aware resource management
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import defaultdict, deque
import statistics
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Handle optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


class OptimizationTarget(Enum):
    """Optimization targets for the performance optimizer."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY_EFFICIENCY = "memory_efficiency"
    ENERGY_CONSUMPTION = "energy_consumption"
    PRIVACY_UTILITY_TRADEOFF = "privacy_utility_tradeoff"
    COST_EFFICIENCY = "cost_efficiency"
    SCALABILITY = "scalability"
    BALANCED = "balanced"


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    HEURISTIC_SEARCH = "heuristic_search"
    MULTI_OBJECTIVE = "multi_objective"
    ADAPTIVE_HYBRID = "adaptive_hybrid"


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    timestamp: float
    throughput: float  # samples/sec
    latency: float     # milliseconds
    memory_usage: float  # MB
    cpu_utilization: float  # percentage
    gpu_utilization: float  # percentage
    energy_consumption: float  # watts
    privacy_budget_efficiency: float  # utility/epsilon ratio
    cost_per_sample: float  # dollars per sample
    accuracy: float  # model accuracy
    privacy_loss: float  # epsilon consumed


@dataclass
class OptimizationConfiguration:
    """Configuration for performance optimization."""
    batch_size: int
    learning_rate: float
    gradient_clipping_norm: float
    privacy_noise_multiplier: float
    num_workers: int
    memory_limit_mb: int
    cache_size_mb: int
    prefetch_factor: int
    mixed_precision: bool
    gradient_accumulation_steps: int
    optimization_level: str  # O0, O1, O2, O3


@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    config: OptimizationConfiguration
    metrics: PerformanceMetrics
    optimization_score: float
    improvement_percentage: float
    feasible: bool
    validation_passed: bool


class PerformanceProfiler:
    """
    Advanced performance profiler for privacy-preserving ML workloads.
    
    Provides detailed profiling of computation, memory, I/O, and privacy overhead.
    """
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.profiling_active = False
        self.profile_data = []
        self.memory_snapshots = deque(maxlen=1000)
        self.cpu_samples = deque(maxlen=1000)
        self.privacy_overhead_samples = deque(maxlen=500)
        
    async def start_profiling(self, duration: Optional[float] = None):
        """Start performance profiling."""
        self.profiling_active = True
        self.profile_data.clear()
        
        logger.info(f"Starting performance profiling (interval: {self.sampling_interval}s)")
        
        start_time = time.time()
        while self.profiling_active:
            if duration and (time.time() - start_time) > duration:
                break
            
            # Collect performance sample
            sample = await self._collect_performance_sample()
            self.profile_data.append(sample)
            
            # Update rolling statistics
            self._update_rolling_statistics(sample)
            
            await asyncio.sleep(self.sampling_interval)
        
        logger.info(f"Performance profiling completed ({len(self.profile_data)} samples)")
    
    def stop_profiling(self):
        """Stop performance profiling."""
        self.profiling_active = False
    
    async def _collect_performance_sample(self) -> Dict[str, Any]:
        """Collect a single performance sample."""
        timestamp = time.time()
        
        # System metrics
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()
            memory_usage_mb = memory_info.used / (1024 * 1024)
            memory_percent = memory_info.percent
        else:
            cpu_percent = random.uniform(20, 80)  # Simulated
            memory_usage_mb = random.uniform(1000, 4000)
            memory_percent = random.uniform(40, 85)
        
        # Simulated ML-specific metrics
        gpu_utilization = random.uniform(60, 95)
        throughput = random.uniform(800, 1200)  # samples/sec
        latency = random.uniform(15, 35)  # ms
        privacy_overhead = random.uniform(0.1, 0.3)  # 10-30% overhead
        
        return {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'memory_usage_mb': memory_usage_mb,
            'memory_percent': memory_percent,
            'gpu_utilization': gpu_utilization,
            'throughput': throughput,
            'latency': latency,
            'privacy_overhead': privacy_overhead,
            'energy_estimate': cpu_percent * 2.5 + gpu_utilization * 3.2  # Watts estimate
        }
    
    def _update_rolling_statistics(self, sample: Dict[str, Any]):
        """Update rolling statistics with new sample."""
        self.cpu_samples.append(sample['cpu_percent'])
        self.memory_snapshots.append(sample['memory_usage_mb'])
        self.privacy_overhead_samples.append(sample['privacy_overhead'])
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get comprehensive profile summary."""
        if not self.profile_data:
            return {'status': 'no_data'}
        
        # Extract time series data
        timestamps = [s['timestamp'] for s in self.profile_data]
        cpu_usage = [s['cpu_percent'] for s in self.profile_data]
        memory_usage = [s['memory_usage_mb'] for s in self.profile_data]
        throughput = [s['throughput'] for s in self.profile_data]
        latency = [s['latency'] for s in self.profile_data]
        privacy_overhead = [s['privacy_overhead'] for s in self.profile_data]
        
        # Calculate statistics
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        
        summary = {
            'profiling_duration': duration,
            'samples_collected': len(self.profile_data),
            'metrics': {
                'cpu_usage': {
                    'mean': statistics.mean(cpu_usage),
                    'std': statistics.stdev(cpu_usage) if len(cpu_usage) > 1 else 0,
                    'min': min(cpu_usage),
                    'max': max(cpu_usage),
                    'p95': self._percentile(cpu_usage, 0.95)
                },
                'memory_usage': {
                    'mean': statistics.mean(memory_usage),
                    'std': statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0,
                    'min': min(memory_usage),
                    'max': max(memory_usage),
                    'p95': self._percentile(memory_usage, 0.95)
                },
                'throughput': {
                    'mean': statistics.mean(throughput),
                    'std': statistics.stdev(throughput) if len(throughput) > 1 else 0,
                    'min': min(throughput),
                    'max': max(throughput)
                },
                'latency': {
                    'mean': statistics.mean(latency),
                    'std': statistics.stdev(latency) if len(latency) > 1 else 0,
                    'min': min(latency),
                    'max': max(latency),
                    'p99': self._percentile(latency, 0.99)
                },
                'privacy_overhead': {
                    'mean': statistics.mean(privacy_overhead),
                    'std': statistics.stdev(privacy_overhead) if len(privacy_overhead) > 1 else 0,
                    'impact_on_throughput': statistics.mean(privacy_overhead) * 100
                }
            },
            'performance_bottlenecks': self._identify_bottlenecks(),
            'optimization_opportunities': self._identify_optimization_opportunities()
        }
        
        return summary
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile * (len(sorted_data) - 1))
        return sorted_data[index]
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify performance bottlenecks from profile data."""
        bottlenecks = []
        
        if not self.profile_data:
            return bottlenecks
        
        # Analyze CPU bottlenecks
        cpu_usage = [s['cpu_percent'] for s in self.profile_data]
        if statistics.mean(cpu_usage) > 80:
            bottlenecks.append('high_cpu_utilization')
        
        # Analyze memory bottlenecks
        memory_usage = [s['memory_percent'] for s in self.profile_data]
        if statistics.mean(memory_usage) > 85:
            bottlenecks.append('high_memory_utilization')
        
        # Analyze throughput bottlenecks
        throughput = [s['throughput'] for s in self.profile_data]
        if statistics.mean(throughput) < 500:
            bottlenecks.append('low_throughput')
        
        # Analyze latency bottlenecks
        latency = [s['latency'] for s in self.profile_data]
        if statistics.mean(latency) > 50:
            bottlenecks.append('high_latency')
        
        # Analyze privacy overhead
        privacy_overhead = [s['privacy_overhead'] for s in self.profile_data]
        if statistics.mean(privacy_overhead) > 0.4:
            bottlenecks.append('high_privacy_overhead')
        
        return bottlenecks
    
    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []
        
        if not self.profile_data:
            return opportunities
        
        # CPU optimization opportunities
        cpu_usage = [s['cpu_percent'] for s in self.profile_data]
        cpu_variance = statistics.variance(cpu_usage) if len(cpu_usage) > 1 else 0
        if cpu_variance > 400:  # High variance suggests uneven load
            opportunities.append('cpu_load_balancing')
        
        # Memory optimization opportunities
        memory_usage = [s['memory_usage_mb'] for s in self.profile_data]
        if max(memory_usage) - min(memory_usage) > 1000:
            opportunities.append('memory_usage_optimization')
        
        # Throughput optimization
        throughput = [s['throughput'] for s in self.profile_data]
        if max(throughput) - statistics.mean(throughput) > 200:
            opportunities.append('throughput_consistency_improvement')
        
        # Privacy efficiency
        privacy_overhead = [s['privacy_overhead'] for s in self.profile_data]
        if statistics.mean(privacy_overhead) > 0.2:
            opportunities.append('privacy_mechanism_optimization')
        
        return opportunities


class AdaptiveConfigurationOptimizer:
    """
    Adaptive configuration optimizer using multiple optimization strategies.
    
    Automatically tunes system parameters for optimal performance while
    maintaining privacy guarantees.
    """
    
    def __init__(self, optimization_target: OptimizationTarget = OptimizationTarget.BALANCED):
        self.optimization_target = optimization_target
        self.optimization_history = []
        self.best_configurations = deque(maxlen=10)
        self.current_generation = 0
        
        # Optimization parameters
        self.population_size = 20
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.convergence_threshold = 0.01
        
        logger.info(f"Initialized Adaptive Configuration Optimizer (target: {optimization_target.value})")
    
    async def optimize_configuration(
        self, 
        baseline_config: OptimizationConfiguration,
        evaluation_function: Callable,
        optimization_budget: int = 100
    ) -> OptimizationResult:
        """
        Optimize configuration using adaptive multi-strategy approach.
        
        Args:
            baseline_config: Starting configuration
            evaluation_function: Function to evaluate configurations
            optimization_budget: Number of optimization iterations
        
        Returns:
            Best optimization result found
        """
        logger.info(f"Starting adaptive configuration optimization (budget: {optimization_budget})")
        
        # Initialize population
        population = await self._initialize_population(baseline_config)
        
        # Evaluate baseline
        baseline_result = await evaluation_function(baseline_config)
        best_result = baseline_result
        
        # Optimization loop
        for generation in range(optimization_budget // self.population_size):
            self.current_generation = generation
            
            # Evaluate population
            evaluation_tasks = [evaluation_function(config) for config in population]
            results = await asyncio.gather(*evaluation_tasks)
            
            # Update best result
            for result in results:
                if result.optimization_score > best_result.optimization_score:
                    best_result = result
                    self.best_configurations.append(result)
            
            # Record generation statistics
            generation_stats = {
                'generation': generation,
                'best_score': max(r.optimization_score for r in results),
                'mean_score': statistics.mean(r.optimization_score for r in results),
                'population_diversity': self._calculate_diversity(population)
            }
            self.optimization_history.append(generation_stats)
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Optimization converged at generation {generation}")
                break
            
            # Evolve population
            population = await self._evolve_population(population, results)
            
            logger.debug(f"Generation {generation}: best_score={generation_stats['best_score']:.4f}")
        
        improvement = ((best_result.optimization_score - baseline_result.optimization_score) / 
                      baseline_result.optimization_score * 100)
        
        logger.info(f"Optimization completed: {improvement:.2f}% improvement achieved")
        return best_result
    
    async def _initialize_population(self, baseline_config: OptimizationConfiguration) -> List[OptimizationConfiguration]:
        """Initialize optimization population."""
        population = [baseline_config]  # Include baseline
        
        for _ in range(self.population_size - 1):
            # Create variations of baseline configuration
            variant = self._mutate_configuration(baseline_config)
            population.append(variant)
        
        return population
    
    def _mutate_configuration(self, config: OptimizationConfiguration) -> OptimizationConfiguration:
        """Create a mutated version of a configuration."""
        # Convert to dictionary for easier manipulation
        config_dict = asdict(config)
        
        # Define mutation ranges for each parameter
        mutation_ranges = {
            'batch_size': [8, 16, 32, 64, 128, 256],
            'learning_rate': (1e-5, 1e-2),
            'gradient_clipping_norm': (0.1, 10.0),
            'privacy_noise_multiplier': (0.1, 2.0),
            'num_workers': [1, 2, 4, 8, 16],
            'memory_limit_mb': (1024, 8192),
            'cache_size_mb': (128, 1024),
            'prefetch_factor': [1, 2, 4, 8],
            'mixed_precision': [True, False],
            'gradient_accumulation_steps': [1, 2, 4, 8],
            'optimization_level': ['O0', 'O1', 'O2', 'O3']
        }
        
        # Mutate random parameters
        num_mutations = random.randint(1, 3)
        parameters_to_mutate = random.sample(list(config_dict.keys()), num_mutations)
        
        for param in parameters_to_mutate:
            if param in mutation_ranges:
                if isinstance(mutation_ranges[param], list):
                    config_dict[param] = random.choice(mutation_ranges[param])
                elif isinstance(mutation_ranges[param], tuple):
                    if isinstance(config_dict[param], bool):
                        config_dict[param] = random.choice([True, False])
                    elif isinstance(config_dict[param], int):
                        min_val, max_val = mutation_ranges[param]
                        config_dict[param] = random.randint(int(min_val), int(max_val))
                    else:
                        min_val, max_val = mutation_ranges[param]
                        config_dict[param] = random.uniform(min_val, max_val)
        
        return OptimizationConfiguration(**config_dict)
    
    def _calculate_diversity(self, population: List[OptimizationConfiguration]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0
        
        # Calculate pairwise differences
        differences = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                config1_dict = asdict(population[i])
                config2_dict = asdict(population[j])
                
                diff = sum(
                    1 for k in config1_dict 
                    if config1_dict[k] != config2_dict[k]
                )
                differences.append(diff / len(config1_dict))
        
        return statistics.mean(differences)
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.optimization_history) < 5:
            return False
        
        # Check if improvement rate has stagnated
        recent_scores = [gen['best_score'] for gen in self.optimization_history[-5:]]
        if len(set(recent_scores)) == 1:  # No improvement
            return True
        
        # Check if improvement is below threshold
        improvement_rate = (recent_scores[-1] - recent_scores[0]) / recent_scores[0]
        return improvement_rate < self.convergence_threshold
    
    async def _evolve_population(
        self, 
        population: List[OptimizationConfiguration], 
        results: List[OptimizationResult]
    ) -> List[OptimizationConfiguration]:
        """Evolve population for next generation."""
        # Sort by optimization score
        sorted_pairs = sorted(zip(population, results), 
                            key=lambda x: x[1].optimization_score, reverse=True)
        
        # Select top performers (elitism)
        elite_size = max(1, self.population_size // 4)
        new_population = [pair[0] for pair in sorted_pairs[:elite_size]]
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, results)
            parent2 = self._tournament_selection(population, results)
            
            # Crossover
            if random.random() < self.crossover_rate:
                offspring = self._crossover(parent1, parent2)
            else:
                offspring = random.choice([parent1, parent2])
            
            # Mutation
            if random.random() < self.mutation_rate:
                offspring = self._mutate_configuration(offspring)
            
            new_population.append(offspring)
        
        return new_population
    
    def _tournament_selection(
        self, 
        population: List[OptimizationConfiguration], 
        results: List[OptimizationResult],
        tournament_size: int = 3
    ) -> OptimizationConfiguration:
        """Select configuration using tournament selection."""
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        tournament_results = [results[i] for i in tournament_indices]
        
        winner_index = max(range(len(tournament_results)), 
                          key=lambda i: tournament_results[i].optimization_score)
        
        return population[tournament_indices[winner_index]]
    
    def _crossover(
        self, 
        parent1: OptimizationConfiguration, 
        parent2: OptimizationConfiguration
    ) -> OptimizationConfiguration:
        """Create offspring through crossover."""
        config1_dict = asdict(parent1)
        config2_dict = asdict(parent2)
        offspring_dict = {}
        
        # Uniform crossover
        for key in config1_dict:
            offspring_dict[key] = random.choice([config1_dict[key], config2_dict[key]])
        
        return OptimizationConfiguration(**offspring_dict)


class IntelligentCacheManager:
    """
    Intelligent cache manager for privacy-preserving ML workloads.
    
    Implements adaptive caching strategies with privacy-aware cache policies.
    """
    
    def __init__(self, max_cache_size_mb: int = 1024, privacy_aware: bool = True):
        self.max_cache_size_mb = max_cache_size_mb
        self.privacy_aware = privacy_aware
        self.cache_storage = {}
        self.cache_metadata = {}
        self.access_history = deque(maxlen=10000)
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_mb': 0.0
        }
        
        # Privacy-aware parameters
        self.privacy_decay_factor = 0.95  # Cache entries lose privacy value over time
        self.max_privacy_age = 3600.0     # 1 hour max age for sensitive data
        
        logger.info(f"Initialized Intelligent Cache Manager (size: {max_cache_size_mb}MB, "
                   f"privacy_aware: {privacy_aware})")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        current_time = time.time()
        
        if key in self.cache_storage:
            metadata = self.cache_metadata[key]
            
            # Check privacy-aware expiration
            if self.privacy_aware and self._is_privacy_expired(metadata, current_time):
                await self._remove_item(key)
                self.cache_stats['misses'] += 1
                self._record_access(key, 'miss', 'privacy_expired')
                return None
            
            # Update access information
            metadata['last_accessed'] = current_time
            metadata['access_count'] += 1
            
            self.cache_stats['hits'] += 1
            self._record_access(key, 'hit')
            
            return self.cache_storage[key]
        else:
            self.cache_stats['misses'] += 1
            self._record_access(key, 'miss', 'not_found')
            return None
    
    async def put(self, key: str, value: Any, privacy_level: float = 1.0, 
                  size_estimate_mb: Optional[float] = None) -> bool:
        """Put item in cache with privacy awareness."""
        if size_estimate_mb is None:
            size_estimate_mb = self._estimate_size(value)
        
        # Check if item fits in cache
        if size_estimate_mb > self.max_cache_size_mb:
            logger.warning(f"Item too large for cache: {size_estimate_mb}MB")
            return False
        
        # Ensure cache has space
        await self._ensure_cache_space(size_estimate_mb)
        
        # Store item with metadata
        current_time = time.time()
        self.cache_storage[key] = value
        self.cache_metadata[key] = {
            'inserted_at': current_time,
            'last_accessed': current_time,
            'access_count': 0,
            'size_mb': size_estimate_mb,
            'privacy_level': privacy_level,
            'privacy_decay_start': current_time
        }
        
        self.cache_stats['size_mb'] += size_estimate_mb
        self._record_access(key, 'insert')
        
        logger.debug(f"Cached item: {key} (size: {size_estimate_mb:.2f}MB, "
                    f"privacy: {privacy_level:.2f})")
        return True
    
    async def _ensure_cache_space(self, required_mb: float):
        """Ensure cache has enough space by evicting items if necessary."""
        while self.cache_stats['size_mb'] + required_mb > self.max_cache_size_mb:
            # Find item to evict using intelligent policy
            eviction_candidate = self._select_eviction_candidate()
            
            if eviction_candidate:
                await self._remove_item(eviction_candidate)
                self.cache_stats['evictions'] += 1
            else:
                logger.warning("Could not find eviction candidate")
                break
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select item for eviction using privacy-aware LRU policy."""
        if not self.cache_metadata:
            return None
        
        current_time = time.time()
        candidates = []
        
        for key, metadata in self.cache_metadata.items():
            # Calculate eviction score
            age_factor = current_time - metadata['last_accessed']
            access_factor = 1.0 / (metadata['access_count'] + 1)
            privacy_factor = 1.0
            
            if self.privacy_aware:
                # Higher privacy level items are less likely to be evicted
                privacy_factor = 1.0 / (metadata['privacy_level'] + 0.1)
                
                # Items past privacy expiration are high priority for eviction
                if self._is_privacy_expired(metadata, current_time):
                    privacy_factor *= 10.0
            
            eviction_score = age_factor * access_factor * privacy_factor
            candidates.append((key, eviction_score))
        
        # Select candidate with highest eviction score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _is_privacy_expired(self, metadata: Dict[str, Any], current_time: float) -> bool:
        """Check if cache entry has expired due to privacy constraints."""
        if not self.privacy_aware:
            return False
        
        # Check absolute privacy age limit
        age = current_time - metadata['privacy_decay_start']
        if age > self.max_privacy_age:
            return True
        
        # Check privacy decay
        decayed_privacy = metadata['privacy_level'] * (self.privacy_decay_factor ** age)
        return decayed_privacy < 0.1  # Privacy threshold
    
    async def _remove_item(self, key: str):
        """Remove item from cache."""
        if key in self.cache_storage:
            metadata = self.cache_metadata[key]
            self.cache_stats['size_mb'] -= metadata['size_mb']
            
            del self.cache_storage[key]
            del self.cache_metadata[key]
            
            logger.debug(f"Evicted cache item: {key}")
    
    def _estimate_size(self, value: Any) -> float:
        """Estimate size of value in MB."""
        try:
            # Rough size estimation
            if hasattr(value, '__sizeof__'):
                size_bytes = value.__sizeof__()
            else:
                size_bytes = len(str(value)) * 2  # Rough estimate
            
            return size_bytes / (1024 * 1024)
        except:
            return 0.1  # Default estimate
    
    def _record_access(self, key: str, access_type: str, details: Optional[str] = None):
        """Record cache access for analysis."""
        self.access_history.append({
            'timestamp': time.time(),
            'key': key,
            'type': access_type,
            'details': details
        })
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_accesses = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_accesses if total_accesses > 0 else 0.0
        
        # Analyze recent access patterns
        recent_accesses = list(self.access_history)[-1000:] if self.access_history else []
        recent_hit_rate = 0.0
        if recent_accesses:
            recent_hits = sum(1 for access in recent_accesses if access['type'] == 'hit')
            recent_hit_rate = recent_hits / len(recent_accesses)
        
        return {
            'total_items': len(self.cache_storage),
            'size_mb': self.cache_stats['size_mb'],
            'max_size_mb': self.max_cache_size_mb,
            'utilization': self.cache_stats['size_mb'] / self.max_cache_size_mb,
            'hit_rate': hit_rate,
            'recent_hit_rate': recent_hit_rate,
            'total_hits': self.cache_stats['hits'],
            'total_misses': self.cache_stats['misses'],
            'total_evictions': self.cache_stats['evictions'],
            'privacy_aware': self.privacy_aware,
            'average_item_size_mb': (self.cache_stats['size_mb'] / len(self.cache_storage) 
                                   if self.cache_storage else 0.0)
        }
    
    async def optimize_cache_policy(self):
        """Optimize cache policy based on access patterns."""
        logger.info("Optimizing cache policy based on access patterns...")
        
        if len(self.access_history) < 100:
            logger.info("Insufficient access history for optimization")
            return
        
        # Analyze access patterns
        access_analysis = self._analyze_access_patterns()
        
        # Adjust cache parameters based on analysis
        if access_analysis['privacy_sensitive_ratio'] > 0.7:
            # High privacy workload - reduce privacy age limit
            self.max_privacy_age = min(self.max_privacy_age, 1800)  # 30 minutes
            logger.info("Adjusted for high-privacy workload")
        
        if access_analysis['access_frequency_variance'] > 0.5:
            # High variance in access patterns - adjust decay factor
            self.privacy_decay_factor = 0.98  # Slower decay
            logger.info("Adjusted for variable access patterns")
        
        logger.info(f"Cache policy optimization completed")
    
    def _analyze_access_patterns(self) -> Dict[str, Any]:
        """Analyze cache access patterns."""
        recent_accesses = list(self.access_history)[-1000:]
        
        # Calculate metrics
        privacy_sensitive_accesses = sum(
            1 for access in recent_accesses 
            if access.get('details') == 'privacy_expired'
        )
        privacy_sensitive_ratio = privacy_sensitive_accesses / len(recent_accesses)
        
        # Access frequency analysis
        key_frequencies = defaultdict(int)
        for access in recent_accesses:
            key_frequencies[access['key']] += 1
        
        frequencies = list(key_frequencies.values())
        access_frequency_variance = statistics.variance(frequencies) if len(frequencies) > 1 else 0
        
        return {
            'privacy_sensitive_ratio': privacy_sensitive_ratio,
            'access_frequency_variance': access_frequency_variance,
            'unique_keys': len(key_frequencies),
            'total_accesses': len(recent_accesses)
        }


class IntelligentPerformanceOptimizer:
    """
    Main orchestrator for intelligent performance optimization.
    
    Coordinates profiling, configuration optimization, and caching
    to achieve optimal performance for privacy-preserving ML workloads.
    """
    
    def __init__(self, optimization_target: OptimizationTarget = OptimizationTarget.BALANCED):
        self.optimization_target = optimization_target
        
        # Initialize components
        self.profiler = PerformanceProfiler()
        self.config_optimizer = AdaptiveConfigurationOptimizer(optimization_target)
        self.cache_manager = IntelligentCacheManager()
        
        # Optimization state
        self.optimization_active = False
        self.optimization_history = []
        self.current_configuration = None
        
        logger.info(f"Initialized Intelligent Performance Optimizer (target: {optimization_target.value})")
    
    async def optimize_system_performance(
        self, 
        baseline_config: OptimizationConfiguration,
        optimization_duration: float = 300.0,  # 5 minutes
        optimization_budget: int = 50
    ) -> Dict[str, Any]:
        """
        Perform comprehensive system performance optimization.
        
        Args:
            baseline_config: Starting configuration
            optimization_duration: How long to run optimization
            optimization_budget: Number of optimization iterations
        
        Returns:
            Comprehensive optimization results
        """
        logger.info(f"Starting comprehensive performance optimization (duration: {optimization_duration}s)")
        
        optimization_results = {
            'optimization_id': f"perf_opt_{int(time.time())}",
            'start_time': time.time(),
            'baseline_config': asdict(baseline_config),
            'optimization_target': self.optimization_target.value,
            'phases': {}
        }
        
        # Phase 1: Baseline Performance Profiling
        logger.info("📊 Phase 1: Baseline Performance Profiling")
        baseline_profile = await self._profile_baseline_performance(baseline_config)
        optimization_results['phases']['baseline_profiling'] = baseline_profile
        
        # Phase 2: Configuration Optimization
        logger.info("⚙️ Phase 2: Configuration Optimization")
        config_optimization = await self._optimize_configuration(
            baseline_config, optimization_budget
        )
        optimization_results['phases']['configuration_optimization'] = config_optimization
        
        # Phase 3: Cache Optimization
        logger.info("🗄️ Phase 3: Cache Optimization")
        cache_optimization = await self._optimize_caching_strategy()
        optimization_results['phases']['cache_optimization'] = cache_optimization
        
        # Phase 4: Integrated Performance Validation
        logger.info("✅ Phase 4: Integrated Performance Validation")
        validation_results = await self._validate_optimized_performance(
            config_optimization.get('best_config')
        )
        optimization_results['phases']['performance_validation'] = validation_results
        
        # Phase 5: Scalability Analysis
        logger.info("📈 Phase 5: Scalability Analysis")
        scalability_results = await self._analyze_scalability_characteristics()
        optimization_results['phases']['scalability_analysis'] = scalability_results
        
        # Compile final results
        optimization_results['summary'] = self._compile_optimization_summary(optimization_results)
        optimization_results['end_time'] = time.time()
        optimization_results['total_duration'] = optimization_results['end_time'] - optimization_results['start_time']
        
        logger.info(f"Performance optimization completed in {optimization_results['total_duration']:.2f}s")
        return optimization_results
    
    async def _profile_baseline_performance(
        self, 
        baseline_config: OptimizationConfiguration
    ) -> Dict[str, Any]:
        """Profile baseline performance."""
        logger.info("Profiling baseline performance...")
        
        # Start profiling
        profiling_task = asyncio.create_task(
            self.profiler.start_profiling(duration=30.0)
        )
        
        # Simulate workload with baseline configuration
        await self._simulate_workload(baseline_config, duration=30.0)
        
        # Stop profiling and get results
        self.profiler.stop_profiling()
        await asyncio.sleep(0.1)  # Allow profiling to complete
        
        profile_summary = self.profiler.get_profile_summary()
        
        return {
            'profiling_completed': True,
            'profile_summary': profile_summary,
            'baseline_metrics': await self._extract_baseline_metrics(profile_summary),
            'bottlenecks_identified': profile_summary.get('performance_bottlenecks', []),
            'optimization_opportunities': profile_summary.get('optimization_opportunities', [])
        }
    
    async def _simulate_workload(
        self, 
        config: OptimizationConfiguration, 
        duration: float = 30.0
    ):
        """Simulate ML workload for performance testing."""
        start_time = time.time()
        
        # Simulate training/inference workload
        while time.time() - start_time < duration:
            # Simulate batch processing
            batch_processing_time = random.uniform(0.01, 0.05)  # 10-50ms per batch
            await asyncio.sleep(batch_processing_time)
            
            # Simulate privacy computations
            privacy_overhead_time = batch_processing_time * random.uniform(0.1, 0.3)
            await asyncio.sleep(privacy_overhead_time)
            
            # Simulate I/O operations
            if random.random() < 0.1:  # 10% chance of I/O
                io_time = random.uniform(0.001, 0.01)
                await asyncio.sleep(io_time)
    
    async def _extract_baseline_metrics(self, profile_summary: Dict[str, Any]) -> PerformanceMetrics:
        """Extract baseline performance metrics."""
        if 'metrics' not in profile_summary:
            return PerformanceMetrics(
                timestamp=time.time(),
                throughput=500.0,
                latency=30.0,
                memory_usage=2048.0,
                cpu_utilization=60.0,
                gpu_utilization=70.0,
                energy_consumption=200.0,
                privacy_budget_efficiency=0.8,
                cost_per_sample=0.001,
                accuracy=0.90,
                privacy_loss=1.0
            )
        
        metrics = profile_summary['metrics']
        return PerformanceMetrics(
            timestamp=time.time(),
            throughput=metrics.get('throughput', {}).get('mean', 500.0),
            latency=metrics.get('latency', {}).get('mean', 30.0),
            memory_usage=metrics.get('memory_usage', {}).get('mean', 2048.0),
            cpu_utilization=metrics.get('cpu_usage', {}).get('mean', 60.0),
            gpu_utilization=75.0,  # Estimated
            energy_consumption=200.0,  # Estimated
            privacy_budget_efficiency=1.0 - metrics.get('privacy_overhead', {}).get('mean', 0.2),
            cost_per_sample=0.001,  # Estimated
            accuracy=0.90,  # Estimated
            privacy_loss=1.0  # Estimated
        )
    
    async def _optimize_configuration(
        self, 
        baseline_config: OptimizationConfiguration, 
        optimization_budget: int
    ) -> Dict[str, Any]:
        """Optimize system configuration."""
        logger.info("Optimizing system configuration...")
        
        # Define evaluation function
        async def evaluate_config(config: OptimizationConfiguration) -> OptimizationResult:
            # Simulate configuration evaluation
            await self._simulate_workload(config, duration=5.0)
            
            # Calculate optimization score based on target
            score = await self._calculate_optimization_score(config)
            
            return OptimizationResult(
                config=config,
                metrics=await self._measure_config_performance(config),
                optimization_score=score,
                improvement_percentage=0.0,  # Will be calculated later
                feasible=True,
                validation_passed=True
            )
        
        # Run optimization
        best_result = await self.config_optimizer.optimize_configuration(
            baseline_config, evaluate_config, optimization_budget
        )
        
        # Calculate improvement
        baseline_score = await self._calculate_optimization_score(baseline_config)
        improvement = ((best_result.optimization_score - baseline_score) / baseline_score * 100)
        best_result.improvement_percentage = improvement
        
        return {
            'optimization_completed': True,
            'baseline_score': baseline_score,
            'best_score': best_result.optimization_score,
            'improvement_percentage': improvement,
            'best_config': asdict(best_result.config),
            'optimization_history': self.config_optimizer.optimization_history,
            'convergence_generation': len(self.config_optimizer.optimization_history)
        }
    
    async def _calculate_optimization_score(self, config: OptimizationConfiguration) -> float:
        """Calculate optimization score for a configuration."""
        # Simulate performance metrics based on configuration
        
        # Batch size impact
        throughput_factor = min(2.0, config.batch_size / 32.0)
        latency_factor = max(0.5, 32.0 / config.batch_size)
        
        # Learning rate impact (optimal around 1e-4)
        lr_factor = max(0.5, min(1.5, -abs(config.learning_rate - 1e-4) * 1000 + 1.0))
        
        # Privacy noise impact
        privacy_factor = max(0.3, 2.0 - config.privacy_noise_multiplier)
        
        # Memory efficiency
        memory_factor = max(0.5, min(1.5, 4096.0 / config.memory_limit_mb))
        
        # Calculate composite score based on optimization target
        if self.optimization_target == OptimizationTarget.THROUGHPUT:
            score = throughput_factor * 0.6 + lr_factor * 0.2 + privacy_factor * 0.2
        elif self.optimization_target == OptimizationTarget.LATENCY:
            score = latency_factor * 0.6 + lr_factor * 0.2 + privacy_factor * 0.2
        elif self.optimization_target == OptimizationTarget.MEMORY_EFFICIENCY:
            score = memory_factor * 0.6 + throughput_factor * 0.2 + privacy_factor * 0.2
        elif self.optimization_target == OptimizationTarget.PRIVACY_UTILITY_TRADEOFF:
            score = privacy_factor * 0.6 + throughput_factor * 0.3 + lr_factor * 0.1
        else:  # BALANCED
            score = (throughput_factor + latency_factor + privacy_factor + memory_factor) / 4.0
        
        # Add some noise for realism
        score += random.uniform(-0.1, 0.1)
        
        return max(0.0, score)
    
    async def _measure_config_performance(self, config: OptimizationConfiguration) -> PerformanceMetrics:
        """Measure performance metrics for a configuration."""
        # Simulate performance measurement
        base_throughput = 500.0 * min(2.0, config.batch_size / 32.0)
        base_latency = 30.0 * max(0.5, 32.0 / config.batch_size)
        base_memory = config.memory_limit_mb * random.uniform(0.6, 0.9)
        
        return PerformanceMetrics(
            timestamp=time.time(),
            throughput=base_throughput + random.uniform(-50, 50),
            latency=base_latency + random.uniform(-5, 5),
            memory_usage=base_memory,
            cpu_utilization=random.uniform(40, 80),
            gpu_utilization=random.uniform(60, 90),
            energy_consumption=200.0 + random.uniform(-50, 50),
            privacy_budget_efficiency=max(0.3, 1.2 - config.privacy_noise_multiplier),
            cost_per_sample=0.001 * random.uniform(0.8, 1.2),
            accuracy=0.90 + random.uniform(-0.05, 0.05),
            privacy_loss=config.privacy_noise_multiplier * random.uniform(0.8, 1.2)
        )
    
    async def _optimize_caching_strategy(self) -> Dict[str, Any]:
        """Optimize caching strategy."""
        logger.info("Optimizing caching strategy...")
        
        # Simulate cache workload
        await self._simulate_cache_workload()
        
        # Optimize cache policy
        await self.cache_manager.optimize_cache_policy()
        
        # Get cache statistics
        cache_stats = self.cache_manager.get_cache_statistics()
        
        return {
            'cache_optimization_completed': True,
            'cache_statistics': cache_stats,
            'optimization_impact': {
                'hit_rate_improvement': max(0.0, cache_stats['hit_rate'] - 0.5),
                'memory_efficiency': cache_stats['utilization'],
                'privacy_compliance': cache_stats['privacy_aware']
            },
            'recommendations': self._generate_cache_recommendations(cache_stats)
        }
    
    async def _simulate_cache_workload(self):
        """Simulate cache workload for optimization."""
        logger.info("Simulating cache workload...")
        
        # Simulate various cache access patterns
        for i in range(100):
            key = f"data_batch_{i % 20}"  # Create access pattern with some locality
            
            # Try to get from cache
            cached_value = await self.cache_manager.get(key)
            
            if cached_value is None:
                # Simulate data generation/loading
                value = f"simulated_data_{i}"
                privacy_level = random.uniform(0.3, 1.0)
                
                await self.cache_manager.put(key, value, privacy_level=privacy_level)
            
            # Small delay between accesses
            await asyncio.sleep(0.01)
    
    def _generate_cache_recommendations(self, cache_stats: Dict[str, Any]) -> List[str]:
        """Generate cache optimization recommendations."""
        recommendations = []
        
        if cache_stats['hit_rate'] < 0.7:
            recommendations.append('increase_cache_size')
        
        if cache_stats['utilization'] > 0.9:
            recommendations.append('optimize_eviction_policy')
        
        if cache_stats['privacy_aware'] and cache_stats['hit_rate'] < 0.8:
            recommendations.append('adjust_privacy_expiration_policy')
        
        if not recommendations:
            recommendations.append('maintain_current_cache_configuration')
        
        return recommendations
    
    async def _validate_optimized_performance(
        self, 
        optimized_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate performance of optimized configuration."""
        logger.info("Validating optimized performance...")
        
        if not optimized_config:
            return {'validation_skipped': True, 'reason': 'no_optimized_config'}
        
        # Convert back to configuration object
        config = OptimizationConfiguration(**optimized_config)
        
        # Run validation workload
        validation_start = time.time()
        await self._simulate_workload(config, duration=20.0)
        validation_duration = time.time() - validation_start
        
        # Measure validation metrics
        validation_metrics = await self._measure_config_performance(config)
        
        # Performance validation criteria
        validation_criteria = {
            'throughput_acceptable': validation_metrics.throughput >= 400.0,
            'latency_acceptable': validation_metrics.latency <= 50.0,
            'memory_efficient': validation_metrics.memory_usage <= 4096.0,
            'privacy_compliant': validation_metrics.privacy_budget_efficiency >= 0.6,
            'cost_effective': validation_metrics.cost_per_sample <= 0.002
        }
        
        validation_passed = sum(validation_criteria.values()) >= 4  # 4 out of 5 criteria
        
        return {
            'validation_completed': True,
            'validation_duration': validation_duration,
            'validation_metrics': asdict(validation_metrics),
            'validation_criteria': validation_criteria,
            'validation_passed': validation_passed,
            'performance_regression': False,  # Would compare with baseline in real implementation
            'recommendations': self._generate_validation_recommendations(validation_criteria)
        }
    
    def _generate_validation_recommendations(self, criteria: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if not criteria['throughput_acceptable']:
            recommendations.append('increase_batch_size_or_parallelism')
        
        if not criteria['latency_acceptable']:
            recommendations.append('optimize_inference_pipeline')
        
        if not criteria['memory_efficient']:
            recommendations.append('implement_memory_optimization')
        
        if not criteria['privacy_compliant']:
            recommendations.append('adjust_privacy_parameters')
        
        if not criteria['cost_effective']:
            recommendations.append('optimize_resource_allocation')
        
        if not recommendations:
            recommendations.append('configuration_meets_all_criteria')
        
        return recommendations
    
    async def _analyze_scalability_characteristics(self) -> Dict[str, Any]:
        """Analyze system scalability characteristics."""
        logger.info("Analyzing scalability characteristics...")
        
        # Test different load levels
        load_levels = [0.5, 1.0, 2.0, 4.0]  # Multipliers of baseline load
        scalability_results = []
        
        for load_multiplier in load_levels:
            # Simulate scaled workload
            scaled_duration = 10.0 / load_multiplier  # Maintain total work constant
            await self._simulate_scaled_workload(load_multiplier, scaled_duration)
            
            # Measure performance at this scale
            performance = await self._measure_scaled_performance(load_multiplier)
            scalability_results.append({
                'load_multiplier': load_multiplier,
                'performance': performance
            })
        
        # Analyze scalability trends
        scalability_analysis = self._analyze_scalability_trends(scalability_results)
        
        return {
            'scalability_analysis_completed': True,
            'load_levels_tested': len(load_levels),
            'scalability_results': scalability_results,
            'scalability_analysis': scalability_analysis,
            'scalability_recommendations': self._generate_scalability_recommendations(scalability_analysis)
        }
    
    async def _simulate_scaled_workload(self, load_multiplier: float, duration: float):
        """Simulate workload at different scales."""
        # Simulate increased load by reducing sleep times
        base_batch_time = 0.02  # 20ms base batch processing time
        scaled_batch_time = base_batch_time / load_multiplier
        
        start_time = time.time()
        while time.time() - start_time < duration:
            await asyncio.sleep(scaled_batch_time)
            
            # Simulate additional overhead for higher loads
            if load_multiplier > 2.0:
                overhead = random.uniform(0.001, 0.005)
                await asyncio.sleep(overhead)
    
    async def _measure_scaled_performance(self, load_multiplier: float) -> Dict[str, float]:
        """Measure performance at a specific scale."""
        # Simulate performance measurements that degrade with scale
        base_throughput = 500.0
        base_latency = 30.0
        base_memory = 2048.0
        base_cpu = 60.0
        
        # Performance typically degrades non-linearly with scale
        throughput_efficiency = max(0.3, 1.0 - (load_multiplier - 1.0) * 0.2)
        latency_overhead = 1.0 + (load_multiplier - 1.0) * 0.3
        memory_overhead = 1.0 + (load_multiplier - 1.0) * 0.4
        cpu_overhead = 1.0 + (load_multiplier - 1.0) * 0.5
        
        return {
            'throughput': base_throughput * load_multiplier * throughput_efficiency,
            'latency': base_latency * latency_overhead,
            'memory_usage': base_memory * memory_overhead,
            'cpu_utilization': min(100.0, base_cpu * cpu_overhead),
            'load_multiplier': load_multiplier,
            'efficiency_score': throughput_efficiency
        }
    
    def _analyze_scalability_trends(self, scalability_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze scalability trends from test results."""
        load_multipliers = [r['load_multiplier'] for r in scalability_results]
        throughputs = [r['performance']['throughput'] for r in scalability_results]
        latencies = [r['performance']['latency'] for r in scalability_results]
        efficiency_scores = [r['performance']['efficiency_score'] for r in scalability_results]
        
        # Calculate scaling characteristics
        max_throughput = max(throughputs)
        optimal_load = load_multipliers[throughputs.index(max_throughput)]
        
        # Linear scalability would maintain efficiency_score = 1.0
        avg_efficiency = statistics.mean(efficiency_scores)
        
        # Determine scalability classification
        if avg_efficiency >= 0.9:
            scalability_class = 'excellent'
        elif avg_efficiency >= 0.7:
            scalability_class = 'good'
        elif avg_efficiency >= 0.5:
            scalability_class = 'acceptable'
        else:
            scalability_class = 'poor'
        
        return {
            'max_throughput': max_throughput,
            'optimal_load_multiplier': optimal_load,
            'average_efficiency': avg_efficiency,
            'scalability_class': scalability_class,
            'linear_scalability': avg_efficiency >= 0.9,
            'bottleneck_threshold': next((lm for lm, es in zip(load_multipliers, efficiency_scores) 
                                        if es < 0.7), load_multipliers[-1])
        }
    
    def _generate_scalability_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate scalability recommendations."""
        recommendations = []
        
        if analysis['scalability_class'] == 'poor':
            recommendations.extend([
                'investigate_bottlenecks',
                'implement_horizontal_scaling',
                'optimize_resource_contention'
            ])
        elif analysis['scalability_class'] == 'acceptable':
            recommendations.extend([
                'profile_performance_at_scale',
                'consider_load_balancing_improvements'
            ])
        
        if not analysis['linear_scalability']:
            recommendations.append('address_scalability_bottlenecks')
        
        if analysis['optimal_load_multiplier'] < 2.0:
            recommendations.append('investigate_early_performance_degradation')
        
        if not recommendations:
            recommendations.append('scalability_characteristics_acceptable')
        
        return recommendations
    
    def _compile_optimization_summary(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive optimization summary."""
        phases = optimization_results['phases']
        
        # Extract key metrics
        baseline_profiling = phases.get('baseline_profiling', {})
        config_optimization = phases.get('configuration_optimization', {})
        cache_optimization = phases.get('cache_optimization', {})
        validation = phases.get('performance_validation', {})
        scalability = phases.get('scalability_analysis', {})
        
        return {
            'optimization_target': self.optimization_target.value,
            'phases_completed': len(phases),
            'overall_improvement': config_optimization.get('improvement_percentage', 0.0),
            'configuration_optimized': config_optimization.get('optimization_completed', False),
            'cache_optimized': cache_optimization.get('cache_optimization_completed', False),
            'validation_passed': validation.get('validation_passed', False),
            'scalability_class': scalability.get('scalability_analysis', {}).get('scalability_class', 'unknown'),
            'key_achievements': [
                f"Configuration improvement: {config_optimization.get('improvement_percentage', 0.0):.1f}%",
                f"Cache hit rate: {cache_optimization.get('cache_statistics', {}).get('hit_rate', 0.0):.1%}",
                f"Validation: {'PASSED' if validation.get('validation_passed', False) else 'FAILED'}",
                f"Scalability: {scalability.get('scalability_analysis', {}).get('scalability_class', 'unknown').upper()}"
            ],
            'performance_ready': (
                config_optimization.get('improvement_percentage', 0.0) > 5.0 and
                validation.get('validation_passed', False) and
                cache_optimization.get('cache_optimization_completed', False)
            )
        }


# Example usage and demonstration
async def demonstrate_intelligent_performance_optimizer():
    """Demonstrate intelligent performance optimizer."""
    print("⚡ Intelligent Performance Optimizer Demonstration")
    
    # Create baseline configuration
    baseline_config = OptimizationConfiguration(
        batch_size=32,
        learning_rate=1e-4,
        gradient_clipping_norm=1.0,
        privacy_noise_multiplier=1.0,
        num_workers=4,
        memory_limit_mb=2048,
        cache_size_mb=512,
        prefetch_factor=2,
        mixed_precision=True,
        gradient_accumulation_steps=1,
        optimization_level='O2'
    )
    
    # Initialize optimizer
    optimizer = IntelligentPerformanceOptimizer(OptimizationTarget.BALANCED)
    
    # Run comprehensive optimization
    results = await optimizer.optimize_system_performance(
        baseline_config=baseline_config,
        optimization_duration=60.0,  # 1 minute for demo
        optimization_budget=20  # Smaller budget for demo
    )
    
    print("\n📊 Performance Optimization Results:")
    summary = results['summary']
    print(f"  • Optimization Target: {summary['optimization_target']}")
    print(f"  • Phases Completed: {summary['phases_completed']}/5")
    print(f"  • Overall Improvement: {summary['overall_improvement']:.2f}%")
    print(f"  • Validation Passed: {'✅' if summary['validation_passed'] else '❌'}")
    print(f"  • Scalability Class: {summary['scalability_class'].upper()}")
    print(f"  • Performance Ready: {'✅' if summary['performance_ready'] else '❌'}")
    
    print(f"\n🔑 Key Achievements:")
    for achievement in summary['key_achievements']:
        print(f"  • {achievement}")
    
    print(f"\n⏱️ Total Optimization Time: {results['total_duration']:.2f}s")
    print("✅ Intelligent performance optimization demonstration completed!")
    
    return results


if __name__ == "__main__":
    # Run demonstration
    result = asyncio.run(demonstrate_intelligent_performance_optimizer())
    print("🎯 Intelligent Performance Optimizer demonstration completed!")