"""Real-time performance monitoring and bottleneck detection system.

This module implements comprehensive performance monitoring including:
- Real-time metrics collection and analysis
- Bottleneck detection and root cause analysis
- Performance regression detection and alerting
- Resource utilization tracking and optimization recommendations
"""

import logging
import time
import asyncio
import threading
import psutil
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import warnings

# Optional GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False
    warnings.warn("NVML not available. GPU monitoring will be disabled.")

# Optional PyTorch profiler
try:
    import torch
    import torch.profiler
    TORCH_PROFILER_AVAILABLE = True
except ImportError:
    TORCH_PROFILER_AVAILABLE = False
    warnings.warn("PyTorch not available. Advanced profiling will be limited.")

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    SYSTEM = "system"
    TRAINING = "training"
    NETWORK = "network" 
    MEMORY = "memory"
    GPU = "gpu"
    CUSTOM = "custom"


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    GPU_BOUND = "gpu_bound"
    IO_BOUND = "io_bound"
    NETWORK_BOUND = "network_bound"
    SYNCHRONIZATION = "synchronization"
    GRADIENT_COMPUTATION = "gradient_computation"
    DATA_LOADING = "data_loading"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    metric_name: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric_name': self.metric_name,
            'metric_type': self.metric_type.value,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class BottleneckDetection:
    """Detected performance bottleneck."""
    bottleneck_id: str
    bottleneck_type: BottleneckType
    severity: AlertSeverity
    description: str
    confidence: float  # 0.0 to 1.0
    affected_components: List[str]
    recommendations: List[str]
    metrics_evidence: List[PerformanceMetric]
    detected_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'bottleneck_id': self.bottleneck_id,
            'bottleneck_type': self.bottleneck_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'confidence': self.confidence,
            'affected_components': self.affected_components,
            'recommendations': self.recommendations,
            'detected_at': self.detected_at.isoformat(),
            'metrics_count': len(self.metrics_evidence)
        }


@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""
    # Collection settings
    collection_interval: float = 1.0  # seconds
    metrics_retention_hours: int = 24
    enable_gpu_monitoring: bool = True
    enable_network_monitoring: bool = True
    enable_detailed_profiling: bool = False
    
    # Bottleneck detection
    enable_bottleneck_detection: bool = True
    bottleneck_detection_window: int = 30  # seconds
    bottleneck_confidence_threshold: float = 0.7
    
    # Alerting
    enable_alerting: bool = True
    alert_cooldown_seconds: int = 300
    email_notifications: bool = False
    webhook_notifications: bool = False
    
    # Thresholds
    cpu_high_threshold: float = 85.0
    memory_high_threshold: float = 90.0
    gpu_high_threshold: float = 90.0
    disk_io_high_threshold: float = 100.0  # MB/s
    network_high_threshold: float = 100.0  # MB/s
    
    # Performance regression detection
    enable_regression_detection: bool = True
    regression_window_minutes: int = 60
    regression_threshold_percent: float = 20.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class SystemMetricsCollector:
    """Collect system-level performance metrics."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize system metrics collector."""
        self.config = config
        self.process = psutil.Process()
        
        # GPU initialization
        self.gpu_available = GPU_AVAILABLE
        self.gpu_count = 0
        
        if self.gpu_available:
            try:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except Exception as e:
                logger.warning(f"GPU monitoring initialization failed: {e}")
                self.gpu_available = False
    
    def collect_cpu_metrics(self) -> List[PerformanceMetric]:
        """Collect CPU performance metrics."""
        metrics = []
        current_time = datetime.now()
        
        try:
            # System-wide CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            metrics.extend([
                PerformanceMetric(
                    metric_name="cpu_utilization",
                    metric_type=MetricType.SYSTEM,
                    value=cpu_percent,
                    unit="percent",
                    timestamp=current_time
                ),
                PerformanceMetric(
                    metric_name="cpu_count",
                    metric_type=MetricType.SYSTEM,
                    value=cpu_count,
                    unit="cores",
                    timestamp=current_time
                )
            ])
            
            # Per-core metrics
            per_cpu = psutil.cpu_percent(percpu=True)
            for i, cpu_usage in enumerate(per_cpu):
                metrics.append(
                    PerformanceMetric(
                        metric_name=f"cpu_core_{i}_utilization",
                        metric_type=MetricType.SYSTEM,
                        value=cpu_usage,
                        unit="percent",
                        timestamp=current_time,
                        metadata={"core_id": i}
                    )
                )
            
            # Process-specific CPU metrics
            process_cpu = self.process.cpu_percent()
            metrics.append(
                PerformanceMetric(
                    metric_name="process_cpu_utilization",
                    metric_type=MetricType.SYSTEM,
                    value=process_cpu,
                    unit="percent",
                    timestamp=current_time
                )
            )
            
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {e}")
        
        return metrics
    
    def collect_memory_metrics(self) -> List[PerformanceMetric]:
        """Collect memory performance metrics."""
        metrics = []
        current_time = datetime.now()
        
        try:
            # System memory
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics.extend([
                PerformanceMetric(
                    metric_name="memory_total",
                    metric_type=MetricType.MEMORY,
                    value=memory.total / (1024**3),  # GB
                    unit="GB",
                    timestamp=current_time
                ),
                PerformanceMetric(
                    metric_name="memory_used",
                    metric_type=MetricType.MEMORY,
                    value=memory.used / (1024**3),
                    unit="GB",
                    timestamp=current_time
                ),
                PerformanceMetric(
                    metric_name="memory_utilization",
                    metric_type=MetricType.MEMORY,
                    value=memory.percent,
                    unit="percent",
                    timestamp=current_time
                ),
                PerformanceMetric(
                    metric_name="swap_utilization",
                    metric_type=MetricType.MEMORY,
                    value=swap.percent,
                    unit="percent",
                    timestamp=current_time
                )
            ])
            
            # Process memory
            process_memory = self.process.memory_info()
            metrics.extend([
                PerformanceMetric(
                    metric_name="process_memory_rss",
                    metric_type=MetricType.MEMORY,
                    value=process_memory.rss / (1024**3),
                    unit="GB",
                    timestamp=current_time
                ),
                PerformanceMetric(
                    metric_name="process_memory_vms",
                    metric_type=MetricType.MEMORY,
                    value=process_memory.vms / (1024**3),
                    unit="GB",
                    timestamp=current_time
                )
            ])
            
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")
        
        return metrics
    
    def collect_gpu_metrics(self) -> List[PerformanceMetric]:
        """Collect GPU performance metrics."""
        metrics = []
        current_time = datetime.now()
        
        if not self.gpu_available or not self.config.enable_gpu_monitoring:
            return metrics
        
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Memory information
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power = 0.0
                
                metrics.extend([
                    PerformanceMetric(
                        metric_name=f"gpu_{i}_utilization",
                        metric_type=MetricType.GPU,
                        value=utilization.gpu,
                        unit="percent",
                        timestamp=current_time,
                        metadata={"gpu_id": i}
                    ),
                    PerformanceMetric(
                        metric_name=f"gpu_{i}_memory_utilization",
                        metric_type=MetricType.GPU,
                        value=(memory_info.used / memory_info.total) * 100,
                        unit="percent",
                        timestamp=current_time,
                        metadata={"gpu_id": i}
                    ),
                    PerformanceMetric(
                        metric_name=f"gpu_{i}_memory_used",
                        metric_type=MetricType.GPU,
                        value=memory_info.used / (1024**3),
                        unit="GB",
                        timestamp=current_time,
                        metadata={"gpu_id": i}
                    ),
                    PerformanceMetric(
                        metric_name=f"gpu_{i}_temperature",
                        metric_type=MetricType.GPU,
                        value=temp,
                        unit="celsius",
                        timestamp=current_time,
                        metadata={"gpu_id": i}
                    ),
                    PerformanceMetric(
                        metric_name=f"gpu_{i}_power",
                        metric_type=MetricType.GPU,
                        value=power,
                        unit="watts",
                        timestamp=current_time,
                        metadata={"gpu_id": i}
                    )
                ])
                
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
        
        return metrics
    
    def collect_disk_metrics(self) -> List[PerformanceMetric]:
        """Collect disk I/O metrics."""
        metrics = []
        current_time = datetime.now()
        
        try:
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            
            # Disk I/O stats
            disk_io = psutil.disk_io_counters()
            
            if disk_io:
                metrics.extend([
                    PerformanceMetric(
                        metric_name="disk_read_bytes_per_sec",
                        metric_type=MetricType.SYSTEM,
                        value=disk_io.read_bytes / (1024**2),  # MB
                        unit="MB/s",
                        timestamp=current_time
                    ),
                    PerformanceMetric(
                        metric_name="disk_write_bytes_per_sec",
                        metric_type=MetricType.SYSTEM,
                        value=disk_io.write_bytes / (1024**2),
                        unit="MB/s",
                        timestamp=current_time
                    ),
                    PerformanceMetric(
                        metric_name="disk_utilization",
                        metric_type=MetricType.SYSTEM,
                        value=disk_usage.percent,
                        unit="percent",
                        timestamp=current_time
                    )
                ])
            
        except Exception as e:
            logger.error(f"Error collecting disk metrics: {e}")
        
        return metrics
    
    def collect_network_metrics(self) -> List[PerformanceMetric]:
        """Collect network performance metrics."""
        metrics = []
        current_time = datetime.now()
        
        if not self.config.enable_network_monitoring:
            return metrics
        
        try:
            # Network I/O stats
            network_io = psutil.net_io_counters()
            
            if network_io:
                metrics.extend([
                    PerformanceMetric(
                        metric_name="network_bytes_sent_per_sec",
                        metric_type=MetricType.NETWORK,
                        value=network_io.bytes_sent / (1024**2),  # MB
                        unit="MB/s",
                        timestamp=current_time
                    ),
                    PerformanceMetric(
                        metric_name="network_bytes_recv_per_sec",
                        metric_type=MetricType.NETWORK,
                        value=network_io.bytes_recv / (1024**2),
                        unit="MB/s",
                        timestamp=current_time
                    ),
                    PerformanceMetric(
                        metric_name="network_packets_sent_per_sec",
                        metric_type=MetricType.NETWORK,
                        value=network_io.packets_sent,
                        unit="packets/s",
                        timestamp=current_time
                    ),
                    PerformanceMetric(
                        metric_name="network_packets_recv_per_sec",
                        metric_type=MetricType.NETWORK,
                        value=network_io.packets_recv,
                        unit="packets/s",
                        timestamp=current_time
                    )
                ])
            
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
        
        return metrics


class TrainingMetricsCollector:
    """Collect training-specific performance metrics."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize training metrics collector."""
        self.config = config
        self.training_step = 0
        self.batch_processing_times: deque = deque(maxlen=100)
        self.gradient_computation_times: deque = deque(maxlen=100)
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def record_batch_time(self, batch_time: float) -> None:
        """Record batch processing time."""
        self.batch_processing_times.append(batch_time)
    
    def record_gradient_time(self, gradient_time: float) -> None:
        """Record gradient computation time."""
        self.gradient_computation_times.append(gradient_time)
    
    def record_custom_metric(self, metric_name: str, value: float) -> None:
        """Record custom training metric."""
        self.custom_metrics[metric_name].append(value)
    
    def collect_training_metrics(self) -> List[PerformanceMetric]:
        """Collect training performance metrics."""
        metrics = []
        current_time = datetime.now()
        
        try:
            # Batch processing metrics
            if self.batch_processing_times:
                avg_batch_time = np.mean(list(self.batch_processing_times))
                max_batch_time = np.max(list(self.batch_processing_times))
                
                metrics.extend([
                    PerformanceMetric(
                        metric_name="avg_batch_processing_time",
                        metric_type=MetricType.TRAINING,
                        value=avg_batch_time * 1000,  # Convert to ms
                        unit="ms",
                        timestamp=current_time
                    ),
                    PerformanceMetric(
                        metric_name="max_batch_processing_time",
                        metric_type=MetricType.TRAINING,
                        value=max_batch_time * 1000,
                        unit="ms",
                        timestamp=current_time
                    ),
                    PerformanceMetric(
                        metric_name="batch_throughput",
                        metric_type=MetricType.TRAINING,
                        value=1.0 / avg_batch_time if avg_batch_time > 0 else 0,
                        unit="batches/s",
                        timestamp=current_time
                    )
                ])
            
            # Gradient computation metrics
            if self.gradient_computation_times:
                avg_grad_time = np.mean(list(self.gradient_computation_times))
                
                metrics.append(
                    PerformanceMetric(
                        metric_name="avg_gradient_computation_time",
                        metric_type=MetricType.TRAINING,
                        value=avg_grad_time * 1000,
                        unit="ms",
                        timestamp=current_time
                    )
                )
            
            # Custom metrics
            for metric_name, values in self.custom_metrics.items():
                if values:
                    avg_value = np.mean(list(values))
                    metrics.append(
                        PerformanceMetric(
                            metric_name=f"custom_{metric_name}",
                            metric_type=MetricType.CUSTOM,
                            value=avg_value,
                            unit="custom",
                            timestamp=current_time
                        )
                    )
            
            # Training step rate
            metrics.append(
                PerformanceMetric(
                    metric_name="training_step",
                    metric_type=MetricType.TRAINING,
                    value=self.training_step,
                    unit="step",
                    timestamp=current_time
                )
            )
            
        except Exception as e:
            logger.error(f"Error collecting training metrics: {e}")
        
        return metrics


class BottleneckDetector:
    """Detect performance bottlenecks using various analysis techniques."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize bottleneck detector."""
        self.config = config
        self.detection_rules: Dict[BottleneckType, Callable] = {
            BottleneckType.CPU_BOUND: self._detect_cpu_bottleneck,
            BottleneckType.MEMORY_BOUND: self._detect_memory_bottleneck,
            BottleneckType.GPU_BOUND: self._detect_gpu_bottleneck,
            BottleneckType.IO_BOUND: self._detect_io_bottleneck,
            BottleneckType.NETWORK_BOUND: self._detect_network_bottleneck,
            BottleneckType.DATA_LOADING: self._detect_data_loading_bottleneck,
            BottleneckType.GRADIENT_COMPUTATION: self._detect_gradient_bottleneck
        }
    
    def detect_bottlenecks(self, metrics: List[PerformanceMetric]) -> List[BottleneckDetection]:
        """Detect bottlenecks from performance metrics."""
        detections = []
        
        # Group metrics by type
        metrics_by_type: Dict[MetricType, List[PerformanceMetric]] = defaultdict(list)
        for metric in metrics:
            metrics_by_type[metric.metric_type].append(metric)
        
        # Run detection rules
        for bottleneck_type, detection_func in self.detection_rules.items():
            try:
                detection = detection_func(metrics_by_type)
                if detection and detection.confidence >= self.config.bottleneck_confidence_threshold:
                    detections.append(detection)
            except Exception as e:
                logger.error(f"Error in bottleneck detection for {bottleneck_type}: {e}")
        
        return detections
    
    def _detect_cpu_bottleneck(self, metrics_by_type: Dict[MetricType, List[PerformanceMetric]]) -> Optional[BottleneckDetection]:
        """Detect CPU bottlenecks."""
        system_metrics = metrics_by_type.get(MetricType.SYSTEM, [])
        cpu_utilization_metrics = [m for m in system_metrics if 'cpu_utilization' in m.metric_name]
        
        if not cpu_utilization_metrics:
            return None
        
        max_cpu_utilization = max(m.value for m in cpu_utilization_metrics)
        avg_cpu_utilization = np.mean([m.value for m in cpu_utilization_metrics])
        
        if max_cpu_utilization > self.config.cpu_high_threshold:
            confidence = min(1.0, (max_cpu_utilization - self.config.cpu_high_threshold) / 20.0)
            
            return BottleneckDetection(
                bottleneck_id=f"cpu_bottleneck_{int(time.time())}",
                bottleneck_type=BottleneckType.CPU_BOUND,
                severity=AlertSeverity.WARNING if max_cpu_utilization < 95 else AlertSeverity.ERROR,
                description=f"High CPU utilization detected: {max_cpu_utilization:.1f}%",
                confidence=confidence,
                affected_components=["cpu", "training_loop"],
                recommendations=[
                    "Consider reducing batch size",
                    "Enable mixed precision training",
                    "Optimize data preprocessing",
                    "Scale to multiple CPUs/machines"
                ],
                metrics_evidence=cpu_utilization_metrics,
                detected_at=datetime.now()
            )
        
        return None
    
    def _detect_memory_bottleneck(self, metrics_by_type: Dict[MetricType, List[PerformanceMetric]]) -> Optional[BottleneckDetection]:
        """Detect memory bottlenecks."""
        memory_metrics = metrics_by_type.get(MetricType.MEMORY, [])
        memory_utilization_metrics = [m for m in memory_metrics if 'memory_utilization' in m.metric_name]
        
        if not memory_utilization_metrics:
            return None
        
        max_memory_utilization = max(m.value for m in memory_utilization_metrics)
        
        if max_memory_utilization > self.config.memory_high_threshold:
            confidence = min(1.0, (max_memory_utilization - self.config.memory_high_threshold) / 10.0)
            
            severity = AlertSeverity.WARNING
            if max_memory_utilization > 95:
                severity = AlertSeverity.ERROR
            elif max_memory_utilization > 98:
                severity = AlertSeverity.CRITICAL
            
            return BottleneckDetection(
                bottleneck_id=f"memory_bottleneck_{int(time.time())}",
                bottleneck_type=BottleneckType.MEMORY_BOUND,
                severity=severity,
                description=f"High memory utilization detected: {max_memory_utilization:.1f}%",
                confidence=confidence,
                affected_components=["memory", "model", "data_loader"],
                recommendations=[
                    "Reduce batch size",
                    "Enable gradient checkpointing",
                    "Use memory-efficient attention",
                    "Offload model parameters to CPU",
                    "Clear unused variables and caches"
                ],
                metrics_evidence=memory_utilization_metrics,
                detected_at=datetime.now()
            )
        
        return None
    
    def _detect_gpu_bottleneck(self, metrics_by_type: Dict[MetricType, List[PerformanceMetric]]) -> Optional[BottleneckDetection]:
        """Detect GPU bottlenecks."""
        gpu_metrics = metrics_by_type.get(MetricType.GPU, [])
        gpu_utilization_metrics = [m for m in gpu_metrics if 'gpu_' in m.metric_name and 'utilization' in m.metric_name and 'memory' not in m.metric_name]
        
        if not gpu_utilization_metrics:
            return None
        
        max_gpu_utilization = max(m.value for m in gpu_utilization_metrics)
        
        if max_gpu_utilization > self.config.gpu_high_threshold:
            confidence = min(1.0, (max_gpu_utilization - self.config.gpu_high_threshold) / 10.0)
            
            return BottleneckDetection(
                bottleneck_id=f"gpu_bottleneck_{int(time.time())}",
                bottleneck_type=BottleneckType.GPU_BOUND,
                severity=AlertSeverity.WARNING if max_gpu_utilization < 95 else AlertSeverity.ERROR,
                description=f"High GPU utilization detected: {max_gpu_utilization:.1f}%",
                confidence=confidence,
                affected_components=["gpu", "model_computation"],
                recommendations=[
                    "Consider model parallelism",
                    "Optimize kernel efficiency",
                    "Use tensor cores when available",
                    "Pipeline model execution",
                    "Scale to multiple GPUs"
                ],
                metrics_evidence=gpu_utilization_metrics,
                detected_at=datetime.now()
            )
        
        return None
    
    def _detect_io_bottleneck(self, metrics_by_type: Dict[MetricType, List[PerformanceMetric]]) -> Optional[BottleneckDetection]:
        """Detect I/O bottlenecks."""
        system_metrics = metrics_by_type.get(MetricType.SYSTEM, [])
        disk_metrics = [m for m in system_metrics if 'disk_' in m.metric_name and 'bytes_per_sec' in m.metric_name]
        
        if not disk_metrics:
            return None
        
        max_disk_io = max(m.value for m in disk_metrics)
        
        if max_disk_io > self.config.disk_io_high_threshold:
            confidence = min(1.0, (max_disk_io - self.config.disk_io_high_threshold) / 50.0)
            
            return BottleneckDetection(
                bottleneck_id=f"io_bottleneck_{int(time.time())}",
                bottleneck_type=BottleneckType.IO_BOUND,
                severity=AlertSeverity.WARNING,
                description=f"High disk I/O detected: {max_disk_io:.1f} MB/s",
                confidence=confidence,
                affected_components=["disk", "data_loading"],
                recommendations=[
                    "Use faster storage (SSD)",
                    "Increase data loading parallelism",
                    "Enable data caching",
                    "Optimize data format (HDF5, Parquet)",
                    "Use memory mapping for large datasets"
                ],
                metrics_evidence=disk_metrics,
                detected_at=datetime.now()
            )
        
        return None
    
    def _detect_network_bottleneck(self, metrics_by_type: Dict[MetricType, List[PerformanceMetric]]) -> Optional[BottleneckDetection]:
        """Detect network bottlenecks."""
        network_metrics = metrics_by_type.get(MetricType.NETWORK, [])
        network_throughput_metrics = [m for m in network_metrics if 'bytes_' in m.metric_name and '_per_sec' in m.metric_name]
        
        if not network_throughput_metrics:
            return None
        
        max_network_throughput = max(m.value for m in network_throughput_metrics)
        
        if max_network_throughput > self.config.network_high_threshold:
            confidence = min(1.0, (max_network_throughput - self.config.network_high_threshold) / 50.0)
            
            return BottleneckDetection(
                bottleneck_id=f"network_bottleneck_{int(time.time())}",
                bottleneck_type=BottleneckType.NETWORK_BOUND,
                severity=AlertSeverity.WARNING,
                description=f"High network throughput detected: {max_network_throughput:.1f} MB/s",
                confidence=confidence,
                affected_components=["network", "distributed_training"],
                recommendations=[
                    "Enable gradient compression",
                    "Use faster network interfaces",
                    "Optimize communication patterns",
                    "Reduce communication frequency",
                    "Use local gradients accumulation"
                ],
                metrics_evidence=network_throughput_metrics,
                detected_at=datetime.now()
            )
        
        return None
    
    def _detect_data_loading_bottleneck(self, metrics_by_type: Dict[MetricType, List[PerformanceMetric]]) -> Optional[BottleneckDetection]:
        """Detect data loading bottlenecks."""
        training_metrics = metrics_by_type.get(MetricType.TRAINING, [])
        batch_time_metrics = [m for m in training_metrics if 'batch_processing_time' in m.metric_name]
        
        if not batch_time_metrics:
            return None
        
        max_batch_time = max(m.value for m in batch_time_metrics)
        
        # Consider it a bottleneck if batch processing is taking too long
        if max_batch_time > 1000:  # More than 1 second
            confidence = min(1.0, max_batch_time / 5000)  # Scale confidence
            
            return BottleneckDetection(
                bottleneck_id=f"data_loading_bottleneck_{int(time.time())}",
                bottleneck_type=BottleneckType.DATA_LOADING,
                severity=AlertSeverity.WARNING,
                description=f"Slow data loading detected: {max_batch_time:.1f}ms per batch",
                confidence=confidence,
                affected_components=["data_loader", "preprocessing"],
                recommendations=[
                    "Increase number of data loading workers",
                    "Enable data prefetching",
                    "Cache preprocessed data",
                    "Optimize data preprocessing pipeline",
                    "Use pin_memory for GPU training"
                ],
                metrics_evidence=batch_time_metrics,
                detected_at=datetime.now()
            )
        
        return None
    
    def _detect_gradient_bottleneck(self, metrics_by_type: Dict[MetricType, List[PerformanceMetric]]) -> Optional[BottleneckDetection]:
        """Detect gradient computation bottlenecks."""
        training_metrics = metrics_by_type.get(MetricType.TRAINING, [])
        gradient_time_metrics = [m for m in training_metrics if 'gradient_computation_time' in m.metric_name]
        
        if not gradient_time_metrics:
            return None
        
        max_gradient_time = max(m.value for m in gradient_time_metrics)
        
        # Consider it a bottleneck if gradient computation is taking too long
        if max_gradient_time > 500:  # More than 500ms
            confidence = min(1.0, max_gradient_time / 2000)  # Scale confidence
            
            return BottleneckDetection(
                bottleneck_id=f"gradient_bottleneck_{int(time.time())}",
                bottleneck_type=BottleneckType.GRADIENT_COMPUTATION,
                severity=AlertSeverity.WARNING,
                description=f"Slow gradient computation detected: {max_gradient_time:.1f}ms",
                confidence=confidence,
                affected_components=["model", "backward_pass"],
                recommendations=[
                    "Enable gradient checkpointing",
                    "Use mixed precision training",
                    "Optimize model architecture",
                    "Reduce model complexity",
                    "Use gradient accumulation"
                ],
                metrics_evidence=gradient_time_metrics,
                detected_at=datetime.now()
            )
        
        return None


class RealTimePerformanceMonitor:
    """Real-time performance monitoring and bottleneck detection system."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        """Initialize performance monitor.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()
        
        # Collectors
        self.system_collector = SystemMetricsCollector(self.config)
        self.training_collector = TrainingMetricsCollector(self.config)
        self.bottleneck_detector = BottleneckDetector(self.config)
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=10000)
        self.bottleneck_history: deque = deque(maxlen=1000)
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Callbacks
        self.alert_callbacks: Dict[str, Callable] = {}
        self.metrics_callbacks: Dict[str, Callable] = {}
        
        # Performance regression tracking
        self.baseline_metrics: Dict[str, float] = {}
        self.regression_history: deque = deque(maxlen=100)
        
        logger.info("Real-time performance monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Started real-time performance monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Stopped performance monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_all_metrics()
                
                # Store metrics
                for metric in metrics:
                    self.metrics_history.append(metric)
                
                # Detect bottlenecks
                if self.config.enable_bottleneck_detection:
                    recent_metrics = list(self.metrics_history)[-100:]  # Last 100 metrics
                    bottlenecks = self.bottleneck_detector.detect_bottlenecks(recent_metrics)
                    
                    for bottleneck in bottlenecks:
                        self._handle_bottleneck_detection(bottleneck)
                
                # Check for performance regressions
                if self.config.enable_regression_detection:
                    self._check_performance_regressions(metrics)
                
                # Trigger metrics callbacks
                for callback in self.metrics_callbacks.values():
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"Metrics callback failed: {e}")
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(self.config.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(self.config.collection_interval * 2)
    
    def _collect_all_metrics(self) -> List[PerformanceMetric]:
        """Collect all available metrics."""
        all_metrics = []
        
        try:
            # System metrics
            all_metrics.extend(self.system_collector.collect_cpu_metrics())
            all_metrics.extend(self.system_collector.collect_memory_metrics())
            all_metrics.extend(self.system_collector.collect_gpu_metrics())
            all_metrics.extend(self.system_collector.collect_disk_metrics())
            all_metrics.extend(self.system_collector.collect_network_metrics())
            
            # Training metrics
            all_metrics.extend(self.training_collector.collect_training_metrics())
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        return all_metrics
    
    def _handle_bottleneck_detection(self, bottleneck: BottleneckDetection) -> None:
        """Handle detected bottleneck."""
        # Check alert cooldown
        cooldown_key = f"{bottleneck.bottleneck_type.value}_{bottleneck.severity.value}"
        current_time = datetime.now()
        
        if cooldown_key in self.alert_cooldowns:
            time_since_last = (current_time - self.alert_cooldowns[cooldown_key]).total_seconds()
            if time_since_last < self.config.alert_cooldown_seconds:
                logger.debug(f"Bottleneck alert in cooldown: {cooldown_key}")
                return
        
        # Record bottleneck
        self.bottleneck_history.append(bottleneck)
        self.alert_cooldowns[cooldown_key] = current_time
        
        # Log bottleneck
        logger.warning(f"Bottleneck detected: {bottleneck.description} "
                      f"(confidence: {bottleneck.confidence:.2f})")
        
        # Trigger alert callbacks
        if self.config.enable_alerting:
            for callback in self.alert_callbacks.values():
                try:
                    callback(bottleneck)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def _check_performance_regressions(self, metrics: List[PerformanceMetric]) -> None:
        """Check for performance regressions."""
        try:
            # Get key performance metrics
            key_metrics = {
                'avg_batch_processing_time': None,
                'cpu_utilization': None,
                'memory_utilization': None,
                'gpu_0_utilization': None
            }
            
            for metric in metrics:
                if metric.metric_name in key_metrics:
                    key_metrics[metric.metric_name] = metric.value
            
            # Compare with baseline
            for metric_name, current_value in key_metrics.items():
                if current_value is None or metric_name not in self.baseline_metrics:
                    continue
                
                baseline_value = self.baseline_metrics[metric_name]
                
                # Calculate regression percentage
                if baseline_value > 0:
                    change_percent = ((current_value - baseline_value) / baseline_value) * 100
                    
                    # Check for significant regression
                    if abs(change_percent) > self.config.regression_threshold_percent:
                        regression_info = {
                            'metric_name': metric_name,
                            'baseline_value': baseline_value,
                            'current_value': current_value,
                            'change_percent': change_percent,
                            'timestamp': datetime.now()
                        }
                        
                        self.regression_history.append(regression_info)
                        
                        logger.warning(f"Performance regression detected in {metric_name}: "
                                     f"{change_percent:.1f}% change from baseline")
                        
                        # Could trigger regression alerts here
            
        except Exception as e:
            logger.error(f"Error checking performance regressions: {e}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old monitoring data."""
        cutoff_time = datetime.now() - timedelta(hours=self.config.metrics_retention_hours)
        
        # Clean metrics history (already handled by deque maxlen)
        
        # Clean alert cooldowns
        expired_cooldowns = [
            key for key, timestamp in self.alert_cooldowns.items()
            if timestamp < cutoff_time
        ]
        
        for key in expired_cooldowns:
            del self.alert_cooldowns[key]
    
    def set_baseline_metrics(self, metrics: Dict[str, float]) -> None:
        """Set baseline metrics for regression detection."""
        self.baseline_metrics.update(metrics)
        logger.info(f"Updated baseline metrics: {list(metrics.keys())}")
    
    def record_training_batch(self, batch_time: float, gradient_time: float = 0.0) -> None:
        """Record training batch metrics."""
        self.training_collector.record_batch_time(batch_time)
        if gradient_time > 0:
            self.training_collector.record_gradient_time(gradient_time)
        self.training_collector.training_step += 1
    
    def record_custom_metric(self, metric_name: str, value: float) -> None:
        """Record custom metric."""
        self.training_collector.record_custom_metric(metric_name, value)
    
    def register_alert_callback(self, name: str, callback: Callable[[BottleneckDetection], None]) -> None:
        """Register callback for bottleneck alerts."""
        self.alert_callbacks[name] = callback
        logger.info(f"Registered alert callback: {name}")
    
    def register_metrics_callback(self, name: str, callback: Callable[[List[PerformanceMetric]], None]) -> None:
        """Register callback for metrics updates."""
        self.metrics_callbacks[name] = callback
        logger.info(f"Registered metrics callback: {name}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics summary."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 metrics
        
        # Group by metric name
        metrics_by_name: Dict[str, List[float]] = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_name[metric.metric_name].append(metric.value)
        
        # Calculate summaries
        summary = {}
        for metric_name, values in metrics_by_name.items():
            if values:
                summary[metric_name] = {
                    'current': values[-1],
                    'average': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values)
                }
        
        return {
            'metrics_summary': summary,
            'active_bottlenecks': len([b for b in self.bottleneck_history if 
                                     (datetime.now() - b.detected_at).total_seconds() < 300]),
            'total_metrics_collected': len(self.metrics_history),
            'monitoring_active': self.monitoring_active,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_bottleneck_summary(self) -> Dict[str, Any]:
        """Get bottleneck detection summary."""
        recent_bottlenecks = [
            b for b in self.bottleneck_history
            if (datetime.now() - b.detected_at).total_seconds() < 3600  # Last hour
        ]
        
        # Group by type
        bottlenecks_by_type: Dict[str, int] = defaultdict(int)
        for bottleneck in recent_bottlenecks:
            bottlenecks_by_type[bottleneck.bottleneck_type.value] += 1
        
        return {
            'recent_bottlenecks': len(recent_bottlenecks),
            'bottlenecks_by_type': dict(bottlenecks_by_type),
            'total_bottlenecks_detected': len(self.bottleneck_history),
            'detection_enabled': self.config.enable_bottleneck_detection,
            'confidence_threshold': self.config.bottleneck_confidence_threshold
        }
    
    def export_monitoring_report(self, output_path: str) -> None:
        """Export comprehensive monitoring report."""
        report = {
            'monitoring_config': self.config.to_dict(),
            'current_metrics': self.get_current_metrics(),
            'bottleneck_summary': self.get_bottleneck_summary(),
            'recent_bottlenecks': [
                bottleneck.to_dict() 
                for bottleneck in list(self.bottleneck_history)[-20:]
            ],
            'performance_regressions': [
                regression for regression in list(self.regression_history)[-10:]
            ],
            'baseline_metrics': self.baseline_metrics,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'gpu_count': self.system_collector.gpu_count,
                'gpu_available': self.system_collector.gpu_available
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Monitoring report exported to {output_path}")
    
    def shutdown(self) -> None:
        """Shutdown performance monitor."""
        logger.info("Shutting down performance monitor...")
        self.stop_monitoring()
        self.executor.shutdown(wait=True)
        logger.info("Performance monitor shutdown completed")


# Global monitor instance
_global_monitor: Optional[RealTimePerformanceMonitor] = None


def get_performance_monitor(config: Optional[MonitoringConfig] = None) -> RealTimePerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RealTimePerformanceMonitor(config)
    return _global_monitor


def start_monitoring() -> None:
    """Convenience function to start monitoring."""
    get_performance_monitor().start_monitoring()


def record_batch_time(batch_time: float, gradient_time: float = 0.0) -> None:
    """Convenience function to record batch time."""
    get_performance_monitor().record_training_batch(batch_time, gradient_time)


def record_custom_metric(metric_name: str, value: float) -> None:
    """Convenience function to record custom metric."""
    get_performance_monitor().record_custom_metric(metric_name, value)