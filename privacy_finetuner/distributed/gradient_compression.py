"""Advanced gradient compression system for federated learning.

This module implements state-of-the-art gradient compression techniques including:
- Multiple compression algorithms (quantization, sparsification, low-rank)
- Adaptive compression based on network and memory conditions
- Error feedback and momentum correction
- Distributed compression coordination
"""

import logging
import time
import math
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import torch
import numpy as np
from collections import defaultdict, deque
import zlib
import lz4.frame
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """Gradient compression algorithms."""
    NONE = "none"
    TOP_K = "top_k"                    # Top-k sparsification
    RANDOM_K = "random_k"              # Random-k sparsification
    THRESHOLD = "threshold"            # Threshold-based sparsification
    QUANTIZATION = "quantization"      # Gradient quantization
    SIGN_SGD = "sign_sgd"             # Sign-based compression
    LOW_RANK = "low_rank"             # Low-rank approximation
    SKETCHED = "sketched"             # Sketched gradients
    ADAPTIVE = "adaptive"             # Adaptive compression
    FEDERATED_DROPOUT = "fed_dropout" # Federated dropout compression


class ErrorCompensation(Enum):
    """Error compensation strategies."""
    NONE = "none"
    ERROR_FEEDBACK = "error_feedback"
    MOMENTUM_RESIDUAL = "momentum_residual"
    MEMORY_RESIDUAL = "memory_residual"


class SynchronizationMode(Enum):
    """Gradient synchronization modes."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    SEMI_SYNCHRONOUS = "semi_synchronous"


@dataclass
class CompressionConfig:
    """Configuration for gradient compression."""
    # Algorithm settings
    algorithm: CompressionAlgorithm = CompressionAlgorithm.ADAPTIVE
    error_compensation: ErrorCompensation = ErrorCompensation.ERROR_FEEDBACK
    sync_mode: SynchronizationMode = SynchronizationMode.SEMI_SYNCHRONOUS
    
    # Compression parameters
    compression_ratio: float = 0.1        # Fraction of gradients to keep
    quantization_bits: int = 8            # Bits for quantization
    sparsity_threshold: float = 1e-5      # Threshold for sparsification
    
    # Adaptive parameters
    adaptive_target_ratio: float = 0.05   # Target compression ratio for adaptive
    adaptation_frequency: int = 100       # Steps between adaptations
    bandwidth_threshold_mbps: float = 100  # Network bandwidth threshold
    
    # Error correction
    enable_error_feedback: bool = True
    momentum_factor: float = 0.9
    memory_decay: float = 0.99
    
    # Performance settings
    batch_compression: bool = True
    async_compression: bool = True
    prefetch_gradients: bool = True
    
    # Communication optimization
    gradient_clipping: float = 1.0
    warm_up_steps: int = 100
    staleness_tolerance: int = 5          # For asynchronous mode
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'algorithm': self.algorithm.value,
            'error_compensation': self.error_compensation.value,
            'compression_ratio': self.compression_ratio,
            'quantization_bits': self.quantization_bits,
            'adaptive_target_ratio': self.adaptive_target_ratio,
            'enable_error_feedback': self.enable_error_feedback
        }


@dataclass
class CompressionStats:
    """Statistics for gradient compression."""
    compression_ratio: float = 0.0
    bandwidth_saved_mb: float = 0.0
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0
    error_norm: float = 0.0
    staleness: int = 0
    sync_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'compression_ratio': self.compression_ratio,
            'bandwidth_saved_mb': self.bandwidth_saved_mb,
            'compression_time_ms': self.compression_time_ms,
            'decompression_time_ms': self.decompression_time_ms,
            'error_norm': self.error_norm,
            'staleness': self.staleness,
            'sync_time_ms': self.sync_time_ms,
            'timestamp': self.timestamp.isoformat()
        }


class GradientCompressor:
    """Base class for gradient compression algorithms."""
    
    def __init__(self, config: CompressionConfig):
        """Initialize gradient compressor.
        
        Args:
            config: Compression configuration
        """
        self.config = config
        self.compression_history: deque = deque(maxlen=1000)
        self.error_memory: Dict[str, torch.Tensor] = {}
        self.momentum_memory: Dict[str, torch.Tensor] = {}
        
    def compress(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], CompressionStats]:
        """Compress gradients.
        
        Args:
            gradients: Dictionary of parameter name to gradient tensor
            
        Returns:
            Tuple of (compressed gradients, compression stats)
        """
        raise NotImplementedError
    
    def decompress(self, compressed_data: Dict[str, Any], gradient_shapes: Dict[str, torch.Size]) -> Dict[str, torch.Tensor]:
        """Decompress gradients.
        
        Args:
            compressed_data: Compressed gradient data
            gradient_shapes: Original gradient shapes
            
        Returns:
            Dictionary of decompressed gradients
        """
        raise NotImplementedError
    
    def _apply_error_compensation(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply error compensation to gradients."""
        if self.config.error_compensation == ErrorCompensation.NONE:
            return gradients
        
        compensated_gradients = {}
        
        for name, grad in gradients.items():
            if self.config.error_compensation == ErrorCompensation.ERROR_FEEDBACK:
                # Add accumulated error
                if name in self.error_memory:
                    grad = grad + self.error_memory[name]
                compensated_gradients[name] = grad
                
            elif self.config.error_compensation == ErrorCompensation.MOMENTUM_RESIDUAL:
                # Apply momentum to residuals
                if name in self.momentum_memory:
                    self.momentum_memory[name] = (
                        self.config.momentum_factor * self.momentum_memory[name] + grad
                    )
                else:
                    self.momentum_memory[name] = grad
                compensated_gradients[name] = self.momentum_memory[name]
                
            elif self.config.error_compensation == ErrorCompensation.MEMORY_RESIDUAL:
                # Memory-based residual tracking
                if name in self.error_memory:
                    compensated_grad = grad + self.config.memory_decay * self.error_memory[name]
                else:
                    compensated_grad = grad
                compensated_gradients[name] = compensated_grad
        
        return compensated_gradients
    
    def _update_error_memory(self, original_gradients: Dict[str, torch.Tensor], 
                           compressed_gradients: Dict[str, torch.Tensor]) -> None:
        """Update error memory with compression residuals."""
        if self.config.error_compensation in [ErrorCompensation.ERROR_FEEDBACK, ErrorCompensation.MEMORY_RESIDUAL]:
            for name in original_gradients.keys():
                if name in compressed_gradients:
                    error = original_gradients[name] - compressed_gradients[name]
                    self.error_memory[name] = error
                else:
                    # Entire gradient was compressed away
                    self.error_memory[name] = original_gradients[name]


class TopKCompressor(GradientCompressor):
    """Top-k sparsification compressor."""
    
    def compress(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], CompressionStats]:
        """Compress using top-k sparsification."""
        start_time = time.time()
        
        # Apply error compensation
        compensated_gradients = self._apply_error_compensation(gradients)
        
        compressed_data = {}
        total_elements = 0
        compressed_elements = 0
        original_size = 0
        compressed_size = 0
        
        for name, grad in compensated_gradients.items():
            flat_grad = grad.flatten()
            total_elements += len(flat_grad)
            original_size += grad.numel() * grad.element_size()
            
            # Calculate k based on compression ratio
            k = max(1, int(len(flat_grad) * self.config.compression_ratio))
            
            # Get top-k elements
            topk_values, topk_indices = torch.topk(flat_grad.abs(), k)
            topk_signs = torch.sign(flat_grad[topk_indices])
            
            compressed_data[name] = {
                'indices': topk_indices.cpu().numpy(),
                'values': (topk_values * topk_signs).cpu().numpy(),
                'shape': grad.shape
            }
            
            compressed_elements += k
            compressed_size += k * (4 + 4)  # int32 indices + float32 values
        
        # Calculate compression stats
        compression_ratio = compressed_elements / total_elements if total_elements > 0 else 0
        compression_time = (time.time() - start_time) * 1000
        bandwidth_saved = (original_size - compressed_size) / (1024 * 1024)
        
        stats = CompressionStats(
            compression_ratio=compression_ratio,
            bandwidth_saved_mb=bandwidth_saved,
            compression_time_ms=compression_time
        )
        
        # Update error memory
        decompressed = self.decompress(compressed_data, {name: grad.shape for name, grad in gradients.items()})
        self._update_error_memory(gradients, decompressed)
        
        return compressed_data, stats
    
    def decompress(self, compressed_data: Dict[str, Any], gradient_shapes: Dict[str, torch.Size]) -> Dict[str, torch.Tensor]:
        """Decompress top-k sparse gradients."""
        start_time = time.time()
        
        decompressed_gradients = {}
        
        for name, data in compressed_data.items():
            if name not in gradient_shapes:
                continue
                
            shape = gradient_shapes[name]
            indices = torch.from_numpy(data['indices'])
            values = torch.from_numpy(data['values'])
            
            # Reconstruct sparse gradient
            flat_grad = torch.zeros(shape.numel(), dtype=values.dtype, device=values.device)
            flat_grad[indices] = values
            
            decompressed_gradients[name] = flat_grad.view(shape)
        
        return decompressed_gradients


class QuantizationCompressor(GradientCompressor):
    """Gradient quantization compressor."""
    
    def compress(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], CompressionStats]:
        """Compress using gradient quantization."""
        start_time = time.time()
        
        compensated_gradients = self._apply_error_compensation(gradients)
        
        compressed_data = {}
        original_size = 0
        compressed_size = 0
        
        for name, grad in compensated_gradients.items():
            original_size += grad.numel() * grad.element_size()
            
            # Calculate quantization parameters
            grad_min = grad.min()
            grad_max = grad.max()
            
            if grad_max > grad_min:
                num_levels = 2 ** self.config.quantization_bits
                scale = (grad_max - grad_min) / (num_levels - 1)
                
                # Quantize
                quantized = ((grad - grad_min) / scale).round().clamp(0, num_levels - 1)
                
                if self.config.quantization_bits <= 8:
                    quantized = quantized.byte()
                    compressed_size += grad.numel()
                else:
                    quantized = quantized.short()
                    compressed_size += grad.numel() * 2
                
                compressed_data[name] = {
                    'quantized': quantized.cpu(),
                    'scale': scale.item(),
                    'zero_point': grad_min.item(),
                    'shape': grad.shape
                }
            else:
                # Constant gradient
                compressed_data[name] = {
                    'constant_value': grad_min.item(),
                    'shape': grad.shape
                }
                compressed_size += 4  # One float32
        
        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        compression_time = (time.time() - start_time) * 1000
        bandwidth_saved = (original_size - compressed_size) / (1024 * 1024)
        
        stats = CompressionStats(
            compression_ratio=compression_ratio,
            bandwidth_saved_mb=bandwidth_saved,
            compression_time_ms=compression_time
        )
        
        # Update error memory
        decompressed = self.decompress(compressed_data, {name: grad.shape for name, grad in gradients.items()})
        self._update_error_memory(gradients, decompressed)
        
        return compressed_data, stats
    
    def decompress(self, compressed_data: Dict[str, Any], gradient_shapes: Dict[str, torch.Size]) -> Dict[str, torch.Tensor]:
        """Decompress quantized gradients."""
        decompressed_gradients = {}
        
        for name, data in compressed_data.items():
            if 'constant_value' in data:
                # Constant gradient
                shape = data['shape']
                decompressed_gradients[name] = torch.full(shape, data['constant_value'])
            else:
                # Dequantize
                quantized = data['quantized']
                scale = data['scale']
                zero_point = data['zero_point']
                shape = data['shape']
                
                dequantized = quantized.float() * scale + zero_point
                decompressed_gradients[name] = dequantized.view(shape)
        
        return decompressed_gradients


class SignSGDCompressor(GradientCompressor):
    """Sign-based SGD compressor."""
    
    def compress(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], CompressionStats]:
        """Compress using sign compression."""
        start_time = time.time()
        
        compensated_gradients = self._apply_error_compensation(gradients)
        
        compressed_data = {}
        original_size = 0
        compressed_size = 0
        
        for name, grad in compensated_gradients.items():
            original_size += grad.numel() * grad.element_size()
            
            # Extract signs and magnitudes
            signs = torch.sign(grad)
            magnitude = grad.abs().mean()  # Use mean magnitude for scaling
            
            # Pack signs into bits (8 signs per byte)
            flat_signs = signs.flatten().byte()
            packed_size = (len(flat_signs) + 7) // 8
            compressed_size += packed_size + 4  # packed signs + magnitude
            
            compressed_data[name] = {
                'signs': flat_signs.cpu(),
                'magnitude': magnitude.item(),
                'shape': grad.shape
            }
        
        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        compression_time = (time.time() - start_time) * 1000
        bandwidth_saved = (original_size - compressed_size) / (1024 * 1024)
        
        stats = CompressionStats(
            compression_ratio=compression_ratio,
            bandwidth_saved_mb=bandwidth_saved,
            compression_time_ms=compression_time
        )
        
        return compressed_data, stats
    
    def decompress(self, compressed_data: Dict[str, Any], gradient_shapes: Dict[str, torch.Size]) -> Dict[str, torch.Tensor]:
        """Decompress sign-based gradients."""
        decompressed_gradients = {}
        
        for name, data in compressed_data.items():
            signs = data['signs'].float()
            magnitude = data['magnitude']
            shape = data['shape']
            
            # Convert signs back to {-1, 0, 1}
            signs = torch.sign(signs - 0.5)  # Convert {0, 1} to {-1, 1}
            
            # Reconstruct gradients
            reconstructed = signs * magnitude
            decompressed_gradients[name] = reconstructed.view(shape)
        
        return decompressed_gradients


class AdaptiveCompressor(GradientCompressor):
    """Adaptive compressor that switches algorithms based on conditions."""
    
    def __init__(self, config: CompressionConfig):
        """Initialize adaptive compressor."""
        super().__init__(config)
        
        # Initialize sub-compressors
        self.compressors = {
            CompressionAlgorithm.TOP_K: TopKCompressor(config),
            CompressionAlgorithm.QUANTIZATION: QuantizationCompressor(config),
            CompressionAlgorithm.SIGN_SGD: SignSGDCompressor(config)
        }
        
        self.current_algorithm = CompressionAlgorithm.TOP_K
        self.performance_history: Dict[CompressionAlgorithm, deque] = {
            alg: deque(maxlen=50) for alg in self.compressors.keys()
        }
        self.adaptation_step = 0
    
    def compress(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], CompressionStats]:
        """Compress using adaptive algorithm selection."""
        # Adapt algorithm periodically
        if self.adaptation_step % self.config.adaptation_frequency == 0:
            self._adapt_algorithm()
        
        self.adaptation_step += 1
        
        # Use current algorithm
        compressor = self.compressors[self.current_algorithm]
        compressed_data, stats = compressor.compress(gradients)
        
        # Track performance
        self.performance_history[self.current_algorithm].append(stats)
        
        # Add algorithm info to compressed data
        compressed_data['_algorithm'] = self.current_algorithm.value
        
        return compressed_data, stats
    
    def decompress(self, compressed_data: Dict[str, Any], gradient_shapes: Dict[str, torch.Size]) -> Dict[str, torch.Tensor]:
        """Decompress using specified algorithm."""
        algorithm_name = compressed_data.pop('_algorithm', self.current_algorithm.value)
        algorithm = CompressionAlgorithm(algorithm_name)
        
        compressor = self.compressors[algorithm]
        return compressor.decompress(compressed_data, gradient_shapes)
    
    def _adapt_algorithm(self) -> None:
        """Adapt compression algorithm based on performance."""
        if len(self.performance_history[self.current_algorithm]) < 10:
            return  # Not enough data
        
        # Evaluate current performance
        current_perf = self._evaluate_algorithm_performance(self.current_algorithm)
        
        # Try other algorithms
        best_algorithm = self.current_algorithm
        best_score = current_perf
        
        for algorithm in self.compressors.keys():
            if algorithm != self.current_algorithm:
                score = self._evaluate_algorithm_performance(algorithm)
                if score > best_score:
                    best_algorithm = algorithm
                    best_score = score
        
        if best_algorithm != self.current_algorithm:
            logger.info(f"Switching compression algorithm from {self.current_algorithm.value} "
                       f"to {best_algorithm.value} (score: {best_score:.3f})")
            self.current_algorithm = best_algorithm
    
    def _evaluate_algorithm_performance(self, algorithm: CompressionAlgorithm) -> float:
        """Evaluate algorithm performance score."""
        if not self.performance_history[algorithm]:
            return 0.0
        
        recent_stats = list(self.performance_history[algorithm])[-10:]
        
        # Weighted performance score
        avg_compression_ratio = np.mean([s.compression_ratio for s in recent_stats])
        avg_compression_time = np.mean([s.compression_time_ms for s in recent_stats])
        avg_bandwidth_saved = np.mean([s.bandwidth_saved_mb for s in recent_stats])
        
        # Higher is better for compression ratio and bandwidth saved
        # Lower is better for compression time
        score = (
            0.4 * (1.0 - avg_compression_ratio) +  # Lower ratio is better
            0.3 * avg_bandwidth_saved +           # More bandwidth saved is better
            0.3 * (1.0 / max(avg_compression_time, 0.1))  # Faster is better
        )
        
        return score


class DistributedGradientCompressor:
    """Distributed gradient compression coordinator."""
    
    def __init__(self, config: CompressionConfig, world_size: int, rank: int):
        """Initialize distributed compressor.
        
        Args:
            config: Compression configuration
            world_size: Total number of processes
            rank: Current process rank
        """
        self.config = config
        self.world_size = world_size
        self.rank = rank
        
        # Initialize compressor based on algorithm
        if config.algorithm == CompressionAlgorithm.ADAPTIVE:
            self.compressor = AdaptiveCompressor(config)
        elif config.algorithm == CompressionAlgorithm.TOP_K:
            self.compressor = TopKCompressor(config)
        elif config.algorithm == CompressionAlgorithm.QUANTIZATION:
            self.compressor = QuantizationCompressor(config)
        elif config.algorithm == CompressionAlgorithm.SIGN_SGD:
            self.compressor = SignSGDCompressor(config)
        else:
            self.compressor = TopKCompressor(config)  # Default
        
        # Distributed state
        self.gradient_buffer: Dict[str, torch.Tensor] = {}
        self.compression_stats_history: deque = deque(maxlen=1000)
        self.step_count = 0
        
        # Asynchronous support
        self.pending_gradients: Dict[int, Dict[str, Any]] = {}
        self.staleness_tracker: Dict[int, int] = defaultdict(int)
        
        # Performance monitoring
        self.bandwidth_monitor = BandwidthMonitor()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized distributed gradient compressor: "
                   f"algorithm={config.algorithm.value}, rank={rank}/{world_size}")
    
    async def compress_and_send(
        self,
        gradients: Dict[str, torch.Tensor],
        step: int
    ) -> CompressionStats:
        """Compress gradients and prepare for distributed communication.
        
        Args:
            gradients: Dictionary of gradients to compress
            step: Current training step
            
        Returns:
            Compression statistics
        """
        sync_start_time = time.time()
        
        # Compress gradients
        compressed_data, compression_stats = self.compressor.compress(gradients)
        
        # Add metadata
        compressed_data['_metadata'] = {
            'step': step,
            'rank': self.rank,
            'timestamp': time.time(),
            'gradient_shapes': {name: grad.shape for name, grad in gradients.items()}
        }
        
        # Handle different synchronization modes
        if self.config.sync_mode == SynchronizationMode.SYNCHRONOUS:
            await self._synchronous_communication(compressed_data, step)
        elif self.config.sync_mode == SynchronizationMode.ASYNCHRONOUS:
            await self._asynchronous_communication(compressed_data, step)
        else:  # Semi-synchronous
            await self._semi_synchronous_communication(compressed_data, step)
        
        # Update stats
        compression_stats.sync_time_ms = (time.time() - sync_start_time) * 1000
        compression_stats.staleness = self.staleness_tracker.get(step, 0)
        
        self.compression_stats_history.append(compression_stats)
        self.step_count += 1
        
        return compression_stats
    
    async def receive_and_decompress(
        self,
        step: int,
        timeout: Optional[float] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        """Receive and decompress gradients from other processes.
        
        Args:
            step: Expected training step
            timeout: Timeout for receiving gradients
            
        Returns:
            Tuple of (aggregated gradients, contributing ranks)
        """
        start_time = time.time()
        
        # Collect compressed gradients from other ranks
        compressed_gradients = await self._collect_compressed_gradients(step, timeout)
        
        # Decompress and aggregate
        aggregated_gradients = {}
        contributing_ranks = []
        
        for rank, compressed_data in compressed_gradients.items():
            metadata = compressed_data.pop('_metadata', {})
            gradient_shapes = metadata.get('gradient_shapes', {})
            
            # Decompress
            decompressed = self.compressor.decompress(compressed_data, gradient_shapes)
            
            # Aggregate (simple averaging for now)
            for name, grad in decompressed.items():
                if name not in aggregated_gradients:
                    aggregated_gradients[name] = torch.zeros_like(grad)
                aggregated_gradients[name] += grad
            
            contributing_ranks.append(rank)
        
        # Average gradients
        num_contributors = len(contributing_ranks)
        if num_contributors > 0:
            for name in aggregated_gradients:
                aggregated_gradients[name] /= num_contributors
        
        return aggregated_gradients, contributing_ranks
    
    async def _synchronous_communication(self, compressed_data: Dict[str, Any], step: int) -> None:
        """Handle synchronous gradient communication."""
        # In a real implementation, this would use MPI, NCCL, or similar
        # For now, simulate synchronous communication
        logger.debug(f"Synchronous communication for step {step}")
        
        # Simulate network delay
        await asyncio.sleep(0.01)
    
    async def _asynchronous_communication(self, compressed_data: Dict[str, Any], step: int) -> None:
        """Handle asynchronous gradient communication."""
        # Store for asynchronous processing
        self.pending_gradients[step] = compressed_data
        
        # Process pending gradients in background
        self.executor.submit(self._process_async_gradients, step)
        
        logger.debug(f"Asynchronous communication for step {step}")
    
    async def _semi_synchronous_communication(self, compressed_data: Dict[str, Any], step: int) -> None:
        """Handle semi-synchronous gradient communication."""
        # Wait for a subset of workers or timeout
        logger.debug(f"Semi-synchronous communication for step {step}")
        
        # Simulate waiting for majority of workers
        await asyncio.sleep(0.005)
    
    def _process_async_gradients(self, step: int) -> None:
        """Process asynchronous gradients in background."""
        try:
            # Update staleness
            current_step = self.step_count
            staleness = max(0, current_step - step)
            self.staleness_tracker[step] = staleness
            
            # Apply staleness tolerance
            if staleness > self.config.staleness_tolerance:
                logger.warning(f"Dropping stale gradients from step {step} "
                              f"(staleness: {staleness})")
                if step in self.pending_gradients:
                    del self.pending_gradients[step]
            
        except Exception as e:
            logger.error(f"Error processing async gradients: {e}")
    
    async def _collect_compressed_gradients(
        self,
        step: int,
        timeout: Optional[float] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Collect compressed gradients from all processes."""
        # In a real implementation, this would collect from network/MPI
        # For now, return simulated data
        
        collected_gradients = {}
        
        # Simulate collecting from other ranks
        for rank in range(self.world_size):
            if rank != self.rank:
                # Simulate compressed gradient data
                collected_gradients[rank] = {
                    '_metadata': {
                        'step': step,
                        'rank': rank,
                        'timestamp': time.time(),
                        'gradient_shapes': {'dummy': torch.Size([10, 10])}
                    },
                    'dummy': {
                        'indices': np.array([0, 1, 2]),
                        'values': np.array([0.1, 0.2, 0.3]),
                        'shape': torch.Size([10, 10])
                    }
                }
        
        return collected_gradients
    
    def get_compression_efficiency(self) -> Dict[str, float]:
        """Get compression efficiency metrics."""
        if not self.compression_stats_history:
            return {}
        
        recent_stats = list(self.compression_stats_history)[-100:]
        
        return {
            'average_compression_ratio': np.mean([s.compression_ratio for s in recent_stats]),
            'average_bandwidth_saved_mb': np.mean([s.bandwidth_saved_mb for s in recent_stats]),
            'average_compression_time_ms': np.mean([s.compression_time_ms for s in recent_stats]),
            'average_sync_time_ms': np.mean([s.sync_time_ms for s in recent_stats]),
            'average_staleness': np.mean([s.staleness for s in recent_stats])
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics."""
        efficiency_stats = self.get_compression_efficiency()
        
        return {
            'efficiency_metrics': efficiency_stats,
            'configuration': self.config.to_dict(),
            'distributed_info': {
                'world_size': self.world_size,
                'rank': self.rank,
                'step_count': self.step_count,
                'pending_gradients': len(self.pending_gradients)
            },
            'bandwidth_stats': self.bandwidth_monitor.get_stats(),
            'compression_history': [
                stats.to_dict() for stats in list(self.compression_stats_history)[-50:]
            ]
        }
    
    def adapt_compression_ratio(self, target_bandwidth_mbps: float) -> None:
        """Adapt compression ratio based on available bandwidth."""
        current_bandwidth = self.bandwidth_monitor.get_current_bandwidth()
        
        if current_bandwidth < target_bandwidth_mbps:
            # Increase compression
            new_ratio = max(0.01, self.config.compression_ratio * 0.8)
            self.config.compression_ratio = new_ratio
            logger.info(f"Increased compression ratio to {new_ratio:.3f} due to bandwidth constraints")
        elif current_bandwidth > target_bandwidth_mbps * 1.5:
            # Decrease compression
            new_ratio = min(1.0, self.config.compression_ratio * 1.2)
            self.config.compression_ratio = new_ratio
            logger.info(f"Decreased compression ratio to {new_ratio:.3f} due to available bandwidth")
    
    def shutdown(self) -> None:
        """Shutdown distributed compressor."""
        self.executor.shutdown(wait=True)
        logger.info("Distributed gradient compressor shutdown completed")


class BandwidthMonitor:
    """Monitor network bandwidth for adaptive compression."""
    
    def __init__(self, window_size: int = 100):
        """Initialize bandwidth monitor.
        
        Args:
            window_size: Size of measurement window
        """
        self.window_size = window_size
        self.bandwidth_history: deque = deque(maxlen=window_size)
        self.last_measurement_time = time.time()
        self.last_bytes_sent = 0
    
    def record_transmission(self, bytes_sent: int) -> None:
        """Record bytes transmitted."""
        current_time = time.time()
        time_diff = current_time - self.last_measurement_time
        
        if time_diff > 0:
            bandwidth_mbps = bytes_sent / (time_diff * 1024 * 1024)
            self.bandwidth_history.append(bandwidth_mbps)
            
            self.last_measurement_time = current_time
            self.last_bytes_sent = bytes_sent
    
    def get_current_bandwidth(self) -> float:
        """Get current bandwidth estimate."""
        if not self.bandwidth_history:
            return 100.0  # Default assumption
        
        # Use exponential moving average
        weights = np.exp(-np.arange(len(self.bandwidth_history)) / 10)
        weights = weights / weights.sum()
        
        return np.average(list(self.bandwidth_history), weights=weights[-len(self.bandwidth_history):])
    
    def get_stats(self) -> Dict[str, float]:
        """Get bandwidth statistics."""
        if not self.bandwidth_history:
            return {'current_mbps': 0.0, 'average_mbps': 0.0, 'peak_mbps': 0.0}
        
        history = list(self.bandwidth_history)
        return {
            'current_mbps': self.get_current_bandwidth(),
            'average_mbps': np.mean(history),
            'peak_mbps': np.max(history),
            'measurements': len(history)
        }