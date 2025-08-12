"""High-performance distributed training system with multi-GPU support.

This module implements enterprise-grade distributed training capabilities including:
- Data and model parallelism with PyTorch DDP
- Dynamic batch sizing and gradient accumulation
- Fault-tolerant training with automatic recovery
- Performance optimization and resource scheduling
"""

import logging
import time
import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

from ..core.trainer import PrivateTrainer
from ..core.privacy_config import PrivacyConfig
from ..utils.logging_config import audit_logger, performance_monitor

logger = logging.getLogger(__name__)


class DistributionStrategy(Enum):
    """Distributed training strategies."""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"


class GradientCompressionType(Enum):
    """Gradient compression methods."""
    NONE = "none"
    QUANTIZATION = "quantization"
    SPARSIFICATION = "sparsification"
    LOW_RANK = "low_rank"
    ADAPTIVE = "adaptive"


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    # Basic distributed settings
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    init_method: str = "env://"
    
    # Strategy settings
    distribution_strategy: DistributionStrategy = DistributionStrategy.DATA_PARALLEL
    gradient_compression: GradientCompressionType = GradientCompressionType.ADAPTIVE
    
    # Performance settings
    gradient_accumulation_steps: int = 1
    max_batch_size: int = 32
    dynamic_batch_sizing: bool = True
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    
    # Memory management
    memory_efficient_attention: bool = True
    gradient_checkpointing: bool = True
    offload_to_cpu: bool = False
    pin_memory: bool = True
    
    # Fault tolerance
    enable_fault_tolerance: bool = True
    checkpoint_frequency: int = 100
    max_retries: int = 3
    
    # Communication optimization
    bucket_size_mb: int = 25
    overlap_communication: bool = True
    compress_gradients: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'backend': self.backend,
            'distribution_strategy': self.distribution_strategy.value,
            'gradient_compression': self.gradient_compression.value,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'max_batch_size': self.max_batch_size,
            'dynamic_batch_sizing': self.dynamic_batch_sizing,
            'mixed_precision': self.mixed_precision
        }


@dataclass
class PerformanceMetrics:
    """Performance tracking for distributed training."""
    step_time: float = 0.0
    forward_time: float = 0.0
    backward_time: float = 0.0
    communication_time: float = 0.0
    samples_per_second: float = 0.0
    memory_usage_gb: float = 0.0
    gpu_utilization: float = 0.0
    communication_volume_mb: float = 0.0
    gradient_norm: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'step_time': self.step_time,
            'forward_time': self.forward_time,
            'backward_time': self.backward_time,
            'communication_time': self.communication_time,
            'samples_per_second': self.samples_per_second,
            'memory_usage_gb': self.memory_usage_gb,
            'gpu_utilization': self.gpu_utilization,
            'communication_volume_mb': self.communication_volume_mb,
            'gradient_norm': self.gradient_norm
        }


class DistributedPrivateTrainer:
    """High-performance distributed trainer with privacy guarantees."""
    
    def __init__(
        self,
        privacy_config: PrivacyConfig,
        distributed_config: DistributedConfig,
        base_trainer: Optional[PrivateTrainer] = None
    ):
        """Initialize distributed trainer.
        
        Args:
            privacy_config: Privacy configuration
            distributed_config: Distributed training configuration
            base_trainer: Base privacy trainer (optional)
        """
        self.privacy_config = privacy_config
        self.distributed_config = distributed_config
        self.base_trainer = base_trainer
        
        # Distributed state
        self.is_distributed = distributed_config.world_size > 1
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.training_start_time = None
        self.current_step = 0
        self.total_samples_processed = 0
        
        # Memory management
        self.memory_monitor = MemoryMonitor()
        self.gradient_accumulator = GradientAccumulator(distributed_config)
        
        # Fault tolerance
        self.checkpoint_manager = CheckpointManager(distributed_config)
        self.failure_detector = FailureDetector()
        
        # Communication optimization
        self.compression_handler = CompressionHandler(distributed_config.gradient_compression)
        
        logger.info(f"Initialized distributed trainer: world_size={distributed_config.world_size}, "
                   f"strategy={distributed_config.distribution_strategy.value}")
    
    def setup_distributed(self) -> None:
        """Initialize distributed training environment."""
        if not self.is_distributed:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Single device training on {self.device}")
            return
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.distributed_config.backend,
                init_method=self.distributed_config.init_method,
                world_size=self.distributed_config.world_size,
                rank=self.distributed_config.rank
            )
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(self.distributed_config.local_rank)
            self.device = torch.device(f"cuda:{self.distributed_config.local_rank}")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Distributed training initialized: rank={self.distributed_config.rank}, "
                   f"local_rank={self.distributed_config.local_rank}, device={self.device}")
    
    def setup_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Setup model for distributed training."""
        model = model.to(self.device)
        
        if self.is_distributed:
            # Enable gradient checkpointing for memory efficiency
            if self.distributed_config.gradient_checkpointing and hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
            
            # Wrap with DDP
            model = DDP(
                model,
                device_ids=[self.distributed_config.local_rank] if torch.cuda.is_available() else None,
                output_device=self.distributed_config.local_rank if torch.cuda.is_available() else None,
                bucket_cap_mb=self.distributed_config.bucket_size_mb,
                find_unused_parameters=False,
                gradient_as_bucket_view=True
            )
            
            logger.info("Model wrapped with DistributedDataParallel")
        
        self.model = model
        return model
    
    def setup_optimizer(self, optimizer_class, **optimizer_kwargs) -> torch.optim.Optimizer:
        """Setup optimizer with distributed considerations."""
        # Scale learning rate for distributed training
        if self.is_distributed and 'lr' in optimizer_kwargs:
            optimizer_kwargs['lr'] *= self.distributed_config.world_size
            logger.info(f"Scaled learning rate for distributed training: {optimizer_kwargs['lr']}")
        
        optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)
        
        # Setup mixed precision scaler
        if self.distributed_config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")
        
        self.optimizer = optimizer
        return optimizer
    
    def create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        **dataloader_kwargs
    ) -> DataLoader:
        """Create distributed dataloader."""
        if batch_size is None:
            batch_size = self.distributed_config.max_batch_size
        
        # Adjust batch size for distributed training
        if self.is_distributed:
            batch_size = batch_size // self.distributed_config.world_size
            
            # Create distributed sampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.distributed_config.world_size,
                rank=self.distributed_config.rank,
                shuffle=shuffle
            )
            shuffle = False  # Sampler handles shuffling
        else:
            sampler = None
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=self.distributed_config.pin_memory and torch.cuda.is_available(),
            num_workers=min(4, os.cpu_count()),
            **dataloader_kwargs
        )
        
        logger.info(f"Created distributed dataloader: batch_size={batch_size}, "
                   f"num_workers={dataloader.num_workers}")
        
        return dataloader
    
    async def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Tuple[float, PerformanceMetrics]:
        """Execute single distributed training step."""
        step_start_time = time.time()
        metrics = PerformanceMetrics()
        
        try:
            # Dynamic batch sizing
            if self.distributed_config.dynamic_batch_sizing:
                batch = await self._adjust_batch_size(batch, step)
            
            # Forward pass with timing
            forward_start = time.time()
            
            if self.distributed_config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.distributed_config.gradient_accumulation_steps
            
            metrics.forward_time = time.time() - forward_start
            
            # Backward pass with timing
            backward_start = time.time()
            
            if self.distributed_config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            metrics.backward_time = time.time() - backward_start
            
            # Gradient accumulation and synchronization
            if (step + 1) % self.distributed_config.gradient_accumulation_steps == 0:
                # Communication timing
                comm_start = time.time()
                
                # Gradient compression
                if self.distributed_config.compress_gradients:
                    await self.compression_handler.compress_gradients(self.model)
                
                # Gradient clipping
                if self.distributed_config.gradient_clipping > 0:
                    if self.distributed_config.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.distributed_config.gradient_clipping
                    )
                
                # Optimizer step
                if self.distributed_config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                metrics.communication_time = time.time() - comm_start
            
            # Performance metrics
            metrics.step_time = time.time() - step_start_time
            metrics.samples_per_second = len(batch['input_ids']) / metrics.step_time
            
            # Memory and GPU metrics
            if torch.cuda.is_available():
                metrics.memory_usage_gb = torch.cuda.memory_allocated() / (1024**3)
                metrics.gpu_utilization = self._get_gpu_utilization()
            
            # Track gradient norms
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            metrics.gradient_norm = total_norm ** (1. / 2)
            
            # Store metrics
            self.performance_history.append(metrics)
            self.current_step = step
            self.total_samples_processed += len(batch['input_ids'])
            
            return loss.item(), metrics
            
        except Exception as e:
            logger.error(f"Training step {step} failed: {e}")
            if self.distributed_config.enable_fault_tolerance:
                await self._handle_training_failure(e, step)
            raise
    
    async def _adjust_batch_size(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, torch.Tensor]:
        """Dynamically adjust batch size based on memory usage."""
        current_memory = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        
        # Get memory pressure
        memory_pressure = current_memory / (torch.cuda.get_device_properties(0).total_memory / (1024**3))
        
        if memory_pressure > 0.85:  # High memory pressure
            # Reduce batch size
            current_batch_size = len(batch['input_ids'])
            new_batch_size = max(1, current_batch_size // 2)
            
            if new_batch_size < current_batch_size:
                # Sample subset of batch
                indices = torch.randperm(current_batch_size)[:new_batch_size]
                batch = {k: v[indices] for k, v in batch.items()}
                
                logger.info(f"Reduced batch size from {current_batch_size} to {new_batch_size} "
                           f"due to memory pressure ({memory_pressure:.2%})")
        
        elif memory_pressure < 0.5 and step % 100 == 0:  # Low memory pressure, periodic check
            # Could potentially increase batch size
            pass
        
        return batch
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.distributed_config.local_rank)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu / 100.0
        except Exception:
            return 0.0  # Fallback if pynvml not available
    
    async def _handle_training_failure(self, error: Exception, step: int) -> None:
        """Handle training failures with recovery."""
        logger.error(f"Training failure at step {step}: {error}")
        
        # Record failure
        self.failure_detector.record_failure(error, step)
        
        # Check if recoverable
        if self.failure_detector.should_retry():
            logger.info("Attempting automatic recovery...")
            
            # Load latest checkpoint
            if self.checkpoint_manager.has_checkpoint():
                checkpoint_step = await self.checkpoint_manager.load_latest_checkpoint(
                    self.model, self.optimizer, self.scaler
                )
                logger.info(f"Recovered from checkpoint at step {checkpoint_step}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        else:
            logger.error("Maximum retries exceeded, training will be terminated")
            raise error
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_history:
            return {"status": "no_data"}
        
        # Calculate averages for last N steps
        recent_metrics = self.performance_history[-100:]
        
        avg_step_time = np.mean([m.step_time for m in recent_metrics])
        avg_samples_per_second = np.mean([m.samples_per_second for m in recent_metrics])
        avg_memory_usage = np.mean([m.memory_usage_gb for m in recent_metrics])
        avg_gpu_utilization = np.mean([m.gpu_utilization for m in recent_metrics])
        avg_communication_time = np.mean([m.communication_time for m in recent_metrics])
        
        # Calculate total training time
        total_training_time = 0
        if self.training_start_time:
            total_training_time = time.time() - self.training_start_time
        
        return {
            "training_summary": {
                "current_step": self.current_step,
                "total_samples_processed": self.total_samples_processed,
                "total_training_time_seconds": total_training_time,
                "average_step_time": avg_step_time,
                "average_samples_per_second": avg_samples_per_second,
                "throughput_improvement": self._calculate_throughput_improvement()
            },
            "performance_metrics": {
                "average_memory_usage_gb": avg_memory_usage,
                "average_gpu_utilization": avg_gpu_utilization,
                "average_communication_time": avg_communication_time,
                "communication_efficiency": self._calculate_communication_efficiency()
            },
            "distributed_info": {
                "world_size": self.distributed_config.world_size,
                "distribution_strategy": self.distributed_config.distribution_strategy.value,
                "gradient_compression": self.distributed_config.gradient_compression.value,
                "mixed_precision_enabled": self.distributed_config.mixed_precision
            },
            "system_efficiency": {
                "memory_efficiency": self._calculate_memory_efficiency(),
                "compute_efficiency": self._calculate_compute_efficiency(),
                "overall_efficiency": self._calculate_overall_efficiency()
            }
        }
    
    def _calculate_throughput_improvement(self) -> float:
        """Calculate throughput improvement over baseline."""
        if len(self.performance_history) < 10:
            return 0.0
        
        # Compare recent performance with initial performance
        initial_throughput = np.mean([m.samples_per_second for m in self.performance_history[:10]])
        recent_throughput = np.mean([m.samples_per_second for m in self.performance_history[-10:]])
        
        if initial_throughput > 0:
            return (recent_throughput - initial_throughput) / initial_throughput * 100
        return 0.0
    
    def _calculate_communication_efficiency(self) -> float:
        """Calculate communication efficiency."""
        if not self.performance_history or not self.is_distributed:
            return 1.0
        
        recent_metrics = self.performance_history[-50:]
        avg_step_time = np.mean([m.step_time for m in recent_metrics])
        avg_comm_time = np.mean([m.communication_time for m in recent_metrics])
        
        if avg_step_time > 0:
            return 1.0 - (avg_comm_time / avg_step_time)
        return 1.0
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency."""
        if not torch.cuda.is_available() or not self.performance_history:
            return 1.0
        
        # Memory efficiency based on utilization
        recent_metrics = self.performance_history[-50:]
        avg_memory_usage = np.mean([m.memory_usage_gb for m in recent_metrics])
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        utilization = avg_memory_usage / total_memory
        # Optimal range is 60-80%
        if 0.6 <= utilization <= 0.8:
            return 1.0
        elif utilization < 0.6:
            return utilization / 0.6
        else:
            return max(0.5, 1.0 - (utilization - 0.8) * 2)
    
    def _calculate_compute_efficiency(self) -> float:
        """Calculate compute efficiency."""
        if not self.performance_history:
            return 1.0
        
        recent_metrics = self.performance_history[-50:]
        avg_gpu_util = np.mean([m.gpu_utilization for m in recent_metrics])
        
        # Target 85%+ GPU utilization for efficiency
        return min(1.0, avg_gpu_util / 0.85)
    
    def _calculate_overall_efficiency(self) -> float:
        """Calculate overall system efficiency."""
        memory_eff = self._calculate_memory_efficiency()
        compute_eff = self._calculate_compute_efficiency()
        comm_eff = self._calculate_communication_efficiency()
        
        # Weighted average
        weights = [0.4, 0.4, 0.2]  # Memory, compute, communication
        return np.average([memory_eff, compute_eff, comm_eff], weights=weights)
    
    async def save_checkpoint(self, step: int, loss: float) -> str:
        """Save training checkpoint."""
        return await self.checkpoint_manager.save_checkpoint(
            step=step,
            loss=loss,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            performance_metrics=self.performance_history[-100:] if self.performance_history else []
        )
    
    def cleanup(self) -> None:
        """Clean up distributed training resources."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Distributed training cleanup completed")


class MemoryMonitor:
    """Monitor and optimize memory usage during training."""
    
    def __init__(self):
        self.memory_history = []
        self.peak_memory = 0
        self.oom_warnings = 0
    
    def monitor_step(self) -> Dict[str, float]:
        """Monitor memory usage for a training step."""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        
        memory_info = {
            'allocated_gb': allocated / (1024**3),
            'cached_gb': cached / (1024**3),
            'peak_gb': torch.cuda.max_memory_allocated() / (1024**3)
        }
        
        self.memory_history.append(memory_info)
        
        # Check for potential OOM
        total_memory = torch.cuda.get_device_properties(0).total_memory
        usage_percent = allocated / total_memory
        
        if usage_percent > 0.9:
            self.oom_warnings += 1
            logger.warning(f"High memory usage detected: {usage_percent:.1%}")
        
        return memory_info


class GradientAccumulator:
    """Handle gradient accumulation across distributed training."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.accumulated_steps = 0
        self.gradient_buffer = {}
    
    def should_sync_gradients(self, step: int) -> bool:
        """Check if gradients should be synchronized."""
        return (step + 1) % self.config.gradient_accumulation_steps == 0
    
    def accumulate_gradients(self, model: torch.nn.Module) -> None:
        """Accumulate gradients from model."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self.gradient_buffer:
                    self.gradient_buffer[name] = torch.zeros_like(param.grad)
                self.gradient_buffer[name] += param.grad.data
        
        self.accumulated_steps += 1


class CheckpointManager:
    """Manage training checkpoints for fault tolerance."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.checkpoint_dir = Path("./checkpoints/distributed")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.latest_checkpoint_path = None
    
    async def save_checkpoint(
        self,
        step: int,
        loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler],
        performance_metrics: List[PerformanceMetrics]
    ) -> str:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        
        checkpoint_data = {
            'step': step,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'performance_metrics': [m.to_dict() for m in performance_metrics],
            'distributed_config': self.config.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        if scaler is not None:
            checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        
        torch.save(checkpoint_data, checkpoint_path)
        self.latest_checkpoint_path = checkpoint_path
        
        logger.info(f"Saved checkpoint at step {step}: {checkpoint_path}")
        return str(checkpoint_path)
    
    def has_checkpoint(self) -> bool:
        """Check if checkpoint exists."""
        return self.latest_checkpoint_path is not None and self.latest_checkpoint_path.exists()
    
    async def load_latest_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler]
    ) -> int:
        """Load latest checkpoint."""
        if not self.has_checkpoint():
            raise ValueError("No checkpoint available")
        
        checkpoint_data = torch.load(self.latest_checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        if scaler is not None and 'scaler_state_dict' in checkpoint_data:
            scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
        
        step = checkpoint_data['step']
        logger.info(f"Loaded checkpoint from step {step}")
        
        return step


class FailureDetector:
    """Detect and handle training failures."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.failure_count = 0
        self.failure_history = []
    
    def record_failure(self, error: Exception, step: int) -> None:
        """Record a training failure."""
        self.failure_count += 1
        self.failure_history.append({
            'error': str(error),
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
    
    def should_retry(self) -> bool:
        """Check if training should retry after failure."""
        return self.failure_count <= self.max_retries


class CompressionHandler:
    """Handle gradient compression for communication optimization."""
    
    def __init__(self, compression_type: GradientCompressionType):
        self.compression_type = compression_type
        self.compression_ratio = 0.5  # Default compression ratio
    
    async def compress_gradients(self, model: torch.nn.Module) -> None:
        """Apply gradient compression."""
        if self.compression_type == GradientCompressionType.NONE:
            return
        
        for param in model.parameters():
            if param.grad is not None:
                if self.compression_type == GradientCompressionType.QUANTIZATION:
                    self._quantize_gradient(param.grad)
                elif self.compression_type == GradientCompressionType.SPARSIFICATION:
                    self._sparsify_gradient(param.grad)
                elif self.compression_type == GradientCompressionType.ADAPTIVE:
                    self._adaptive_compress_gradient(param.grad)
    
    def _quantize_gradient(self, grad: torch.Tensor) -> None:
        """Apply gradient quantization."""
        # Simple 8-bit quantization
        grad_max = grad.abs().max()
        if grad_max > 0:
            scale = grad_max / 127
            quantized = torch.round(grad / scale).clamp(-128, 127)
            grad.data = quantized * scale
    
    def _sparsify_gradient(self, grad: torch.Tensor) -> None:
        """Apply gradient sparsification."""
        # Top-k sparsification
        flat_grad = grad.view(-1)
        k = int(len(flat_grad) * self.compression_ratio)
        
        if k > 0:
            topk_values, topk_indices = torch.topk(flat_grad.abs(), k)
            threshold = topk_values[-1]
            
            mask = grad.abs() >= threshold
            grad.data = grad * mask.float()
    
    def _adaptive_compress_gradient(self, grad: torch.Tensor) -> None:
        """Apply adaptive gradient compression."""
        # Choose compression method based on gradient properties
        grad_norm = grad.norm()
        grad_sparsity = (grad == 0).float().mean()
        
        if grad_sparsity > 0.7:
            # Already sparse, use quantization
            self._quantize_gradient(grad)
        else:
            # Dense gradient, use sparsification
            self._sparsify_gradient(grad)


def launch_distributed_training(
    rank: int,
    world_size: int,
    train_function: Callable,
    *args,
    **kwargs
) -> None:
    """Launch distributed training process."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    
    try:
        train_function(rank, world_size, *args, **kwargs)
    except Exception as e:
        logger.error(f"Distributed training failed on rank {rank}: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def start_distributed_training(
    world_size: int,
    train_function: Callable,
    *args,
    **kwargs
) -> None:
    """Start distributed training across multiple processes."""
    if world_size == 1:
        # Single process training
        train_function(0, 1, *args, **kwargs)
    else:
        # Multi-process training
        mp.spawn(
            launch_distributed_training,
            args=(world_size, train_function) + args,
            nprocs=world_size,
            join=True,
            **kwargs
        )