"""Comprehensive resource management system for privacy-preserving ML training.

This module provides advanced resource management capabilities including dynamic scaling,
memory optimization, resource exhaustion handling, and intelligent resource allocation
for production-grade privacy-preserving machine learning systems.
"""

import os
import time
import logging
import threading
import gc
import warnings

# Import psutil with graceful fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Resource monitoring will use fallback methods.")
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import math
import json
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available, some GPU resource management features will be disabled")

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
    nvml.nvmlInit()
except ImportError:
    NVML_AVAILABLE = False
    warnings.warn("nvidia-ml-py not available, GPU monitoring will be limited")

from ..core.exceptions import ResourceExhaustedException, ModelTrainingException
from ..utils.logging_config import audit_logger

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    GPU_MEMORY = "gpu_memory"
    DISK = "disk"
    NETWORK = "network"
    SWAP = "swap"


class ResourceState(Enum):
    """Resource allocation states."""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    EXHAUSTED = "exhausted"
    WARNING = "warning"
    CRITICAL = "critical"


class ScalingPolicy(Enum):
    """Resource scaling policies."""
    CONSERVATIVE = "conservative"  # Scale slowly, prefer stability
    AGGRESSIVE = "aggressive"     # Scale quickly, prefer performance
    BALANCED = "balanced"         # Balance between stability and performance
    MANUAL = "manual"            # No automatic scaling


@dataclass
class ResourceThreshold:
    """Resource usage thresholds for scaling decisions."""
    warning_threshold: float = 0.8    # 80% usage warning
    critical_threshold: float = 0.9   # 90% usage critical
    scale_up_threshold: float = 0.85  # Scale up at 85% usage
    scale_down_threshold: float = 0.3 # Scale down below 30% usage
    max_threshold: float = 0.95       # Maximum safe usage (95%)


@dataclass
class ResourceAllocation:
    """Represents a resource allocation."""
    resource_id: str
    resource_type: ResourceType
    allocated_amount: float
    allocated_at: datetime
    owner: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1 (low) to 10 (high)
    
    def is_expired(self, max_age_seconds: int = 3600) -> bool:
        """Check if allocation has expired."""
        return (datetime.now() - self.allocated_at).total_seconds() > max_age_seconds


@dataclass
class ResourceUsageMetrics:
    """Resource usage metrics over time."""
    resource_type: ResourceType
    timestamp: datetime
    usage_percent: float
    available_amount: float
    used_amount: float
    total_amount: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingAction:
    """Represents a scaling action."""
    action_id: str
    resource_type: ResourceType
    action_type: str  # "scale_up", "scale_down", "optimize"
    requested_by: str
    current_allocation: float
    target_allocation: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    executed: bool = False
    result: Optional[str] = None


class ResourceMonitor:
    """Monitors system resource usage with predictive capabilities."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Resource usage history
        self.usage_history: Dict[ResourceType, deque] = {
            resource_type: deque(maxlen=100) for resource_type in ResourceType
        }
        
        # Resource thresholds
        self.thresholds: Dict[ResourceType, ResourceThreshold] = {
            resource_type: ResourceThreshold() for resource_type in ResourceType
        }
        
        # Callbacks for resource events
        self.resource_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # GPU monitoring setup
        self.gpu_count = 0
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
        
        self.lock = threading.RLock()
        
        logger.info(f"Resource monitor initialized with {self.gpu_count} GPUs detected")
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect resource metrics
                metrics = self._collect_resource_metrics()
                
                # Store metrics history
                self._store_metrics(metrics)
                
                # Check thresholds and trigger callbacks
                self._check_thresholds(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.monitoring_interval * 2)
    
    def _collect_resource_metrics(self) -> Dict[ResourceType, ResourceUsageMetrics]:
        """Collect current resource usage metrics."""
        metrics = {}
        timestamp = datetime.now()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            metrics[ResourceType.CPU] = ResourceUsageMetrics(
                resource_type=ResourceType.CPU,
                timestamp=timestamp,
                usage_percent=cpu_percent,
                used_amount=cpu_percent * cpu_count / 100,
                available_amount=cpu_count * (100 - cpu_percent) / 100,
                total_amount=cpu_count,
                metadata={'cpu_count': cpu_count, 'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None}
            )
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics[ResourceType.MEMORY] = ResourceUsageMetrics(
                resource_type=ResourceType.MEMORY,
                timestamp=timestamp,
                usage_percent=memory.percent,
                used_amount=memory.used / (1024**3),  # GB
                available_amount=memory.available / (1024**3),  # GB
                total_amount=memory.total / (1024**3),  # GB
                metadata={'swap': psutil.swap_memory()._asdict()}
            )
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics[ResourceType.DISK] = ResourceUsageMetrics(
                resource_type=ResourceType.DISK,
                timestamp=timestamp,
                usage_percent=disk.percent,
                used_amount=disk.used / (1024**3),  # GB
                available_amount=disk.free / (1024**3),  # GB
                total_amount=disk.total / (1024**3),  # GB
                metadata={'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else None}
            )
            
            # GPU metrics
            if self.gpu_count > 0:
                gpu_metrics = self._collect_gpu_metrics()
                metrics.update(gpu_metrics)
            
        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")
        
        return metrics
    
    def _collect_gpu_metrics(self) -> Dict[ResourceType, ResourceUsageMetrics]:
        """Collect GPU resource metrics."""
        gpu_metrics = {}
        timestamp = datetime.now()
        
        try:
            total_gpu_memory = 0
            used_gpu_memory = 0
            gpu_utilizations = []
            
            if NVML_AVAILABLE:
                # Use NVML for detailed GPU metrics
                for i in range(self.gpu_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilizations.append(util.gpu)
                    
                    # GPU memory
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    total_gpu_memory += mem_info.total
                    used_gpu_memory += mem_info.used
            
            elif TORCH_AVAILABLE:
                # Use PyTorch for basic GPU metrics
                for i in range(self.gpu_count):
                    torch.cuda.set_device(i)
                    total = torch.cuda.get_device_properties(i).total_memory
                    reserved = torch.cuda.memory_reserved(i)
                    
                    total_gpu_memory += total
                    used_gpu_memory += reserved
            
            # GPU utilization metrics
            if gpu_utilizations:
                avg_gpu_util = sum(gpu_utilizations) / len(gpu_utilizations)
                gpu_metrics[ResourceType.GPU] = ResourceUsageMetrics(
                    resource_type=ResourceType.GPU,
                    timestamp=timestamp,
                    usage_percent=avg_gpu_util,
                    used_amount=avg_gpu_util * self.gpu_count / 100,
                    available_amount=self.gpu_count * (100 - avg_gpu_util) / 100,
                    total_amount=self.gpu_count,
                    metadata={'individual_utilizations': gpu_utilizations}
                )
            
            # GPU memory metrics
            if total_gpu_memory > 0:
                gpu_memory_percent = (used_gpu_memory / total_gpu_memory) * 100
                gpu_metrics[ResourceType.GPU_MEMORY] = ResourceUsageMetrics(
                    resource_type=ResourceType.GPU_MEMORY,
                    timestamp=timestamp,
                    usage_percent=gpu_memory_percent,
                    used_amount=used_gpu_memory / (1024**3),  # GB
                    available_amount=(total_gpu_memory - used_gpu_memory) / (1024**3),  # GB
                    total_amount=total_gpu_memory / (1024**3),  # GB
                    metadata={'gpu_count': self.gpu_count}
                )
        
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
        
        return gpu_metrics
    
    def _store_metrics(self, metrics: Dict[ResourceType, ResourceUsageMetrics]):
        """Store metrics in history."""
        with self.lock:
            for resource_type, metric in metrics.items():
                if resource_type in self.usage_history:
                    self.usage_history[resource_type].append(metric)
    
    def _check_thresholds(self, metrics: Dict[ResourceType, ResourceUsageMetrics]):
        """Check resource thresholds and trigger callbacks."""
        for resource_type, metric in metrics.items():
            if resource_type not in self.thresholds:
                continue
            
            threshold = self.thresholds[resource_type]
            usage_percent = metric.usage_percent / 100.0
            
            # Check various threshold levels
            if usage_percent >= threshold.critical_threshold:
                self._trigger_resource_event('critical', resource_type, metric)
            elif usage_percent >= threshold.warning_threshold:
                self._trigger_resource_event('warning', resource_type, metric)
            elif usage_percent >= threshold.scale_up_threshold:
                self._trigger_resource_event('scale_up', resource_type, metric)
            elif usage_percent <= threshold.scale_down_threshold:
                self._trigger_resource_event('scale_down', resource_type, metric)
    
    def _trigger_resource_event(self, event_type: str, resource_type: ResourceType, metric: ResourceUsageMetrics):
        """Trigger resource event callbacks."""
        event_key = f"{event_type}_{resource_type.value}"
        
        for callback in self.resource_callbacks[event_key]:
            try:
                callback(event_type, resource_type, metric)
            except Exception as e:
                logger.error(f"Resource event callback error: {e}")
    
    def register_resource_callback(self, event_type: str, resource_type: ResourceType, callback: Callable):
        """Register callback for resource events."""
        event_key = f"{event_type}_{resource_type.value}"
        self.resource_callbacks[event_key].append(callback)
        
        logger.info(f"Registered callback for {event_key}")
    
    def get_current_usage(self, resource_type: ResourceType) -> Optional[ResourceUsageMetrics]:
        """Get current usage for a resource type."""
        with self.lock:
            history = self.usage_history.get(resource_type, deque())
            return history[-1] if history else None
    
    def get_usage_trend(self, resource_type: ResourceType, window_minutes: int = 15) -> Dict[str, float]:
        """Get usage trend for a resource type."""
        with self.lock:
            history = self.usage_history.get(resource_type, deque())
            if len(history) < 2:
                return {'trend': 0.0, 'confidence': 0.0}
            
            # Filter to window
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_metrics = [m for m in history if m.timestamp > cutoff_time]
            
            if len(recent_metrics) < 2:
                return {'trend': 0.0, 'confidence': 0.0}
            
            # Calculate trend (linear regression)
            x_values = [(m.timestamp - recent_metrics[0].timestamp).total_seconds() for m in recent_metrics]
            y_values = [m.usage_percent for m in recent_metrics]
            
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            if n * sum_x2 - sum_x * sum_x == 0:
                return {'trend': 0.0, 'confidence': 0.0}
            
            trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            confidence = min(1.0, len(recent_metrics) / 10.0)  # More samples = higher confidence
            
            return {'trend': trend, 'confidence': confidence}
    
    def predict_resource_exhaustion(self, resource_type: ResourceType) -> Optional[datetime]:
        """Predict when a resource might be exhausted based on current trend."""
        current = self.get_current_usage(resource_type)
        trend_data = self.get_usage_trend(resource_type)
        
        if not current or trend_data['trend'] <= 0 or trend_data['confidence'] < 0.5:
            return None
        
        threshold = self.thresholds.get(resource_type, ResourceThreshold())
        remaining_capacity = threshold.max_threshold * 100 - current.usage_percent
        
        if remaining_capacity <= 0:
            return datetime.now()  # Already exhausted
        
        # Calculate time to exhaustion based on trend
        trend_per_second = trend_data['trend'] / 60  # Convert per-minute to per-second
        if trend_per_second <= 0:
            return None
        
        seconds_to_exhaustion = remaining_capacity / trend_per_second
        exhaustion_time = datetime.now() + timedelta(seconds=seconds_to_exhaustion)
        
        return exhaustion_time
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource usage summary."""
        summary = {}
        
        for resource_type in ResourceType:
            current = self.get_current_usage(resource_type)
            trend = self.get_usage_trend(resource_type)
            exhaustion_time = self.predict_resource_exhaustion(resource_type)
            
            if current:
                summary[resource_type.value] = {
                    'current_usage_percent': current.usage_percent,
                    'used_amount': current.used_amount,
                    'available_amount': current.available_amount,
                    'total_amount': current.total_amount,
                    'trend': trend['trend'],
                    'trend_confidence': trend['confidence'],
                    'predicted_exhaustion': exhaustion_time.isoformat() if exhaustion_time else None,
                    'state': self._determine_resource_state(current, resource_type)
                }
        
        return summary
    
    def _determine_resource_state(self, metric: ResourceUsageMetrics, resource_type: ResourceType) -> str:
        """Determine resource state based on usage."""
        threshold = self.thresholds.get(resource_type, ResourceThreshold())
        usage_ratio = metric.usage_percent / 100.0
        
        if usage_ratio >= threshold.max_threshold:
            return ResourceState.EXHAUSTED.value
        elif usage_ratio >= threshold.critical_threshold:
            return ResourceState.CRITICAL.value
        elif usage_ratio >= threshold.warning_threshold:
            return ResourceState.WARNING.value
        else:
            return ResourceState.AVAILABLE.value


class ResourceAllocator:
    """Manages resource allocation and optimization."""
    
    def __init__(self, resource_monitor: ResourceMonitor):
        self.resource_monitor = resource_monitor
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        self.allocation_history: List[ResourceAllocation] = []
        self.allocation_policies: Dict[str, Dict[str, Any]] = {}
        
        # Resource pools
        self.resource_pools: Dict[ResourceType, Dict[str, Any]] = {}
        
        self.lock = threading.RLock()
        
        logger.info("Resource allocator initialized")
    
    def allocate_resource(
        self,
        resource_type: ResourceType,
        amount: float,
        owner: str,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Allocate resources with priority and policies."""
        with self.lock:
            # Check current availability
            current_usage = self.resource_monitor.get_current_usage(resource_type)
            if not current_usage:
                logger.warning(f"Cannot check availability for {resource_type.value}")
                return None
            
            # Calculate if allocation is possible
            available_amount = current_usage.available_amount
            if amount > available_amount:
                # Try optimization first
                if not self._optimize_allocations(resource_type, amount):
                    logger.warning(f"Cannot allocate {amount} {resource_type.value}, only {available_amount} available")
                    return None
            
            # Create allocation
            allocation_id = f"{resource_type.value}_{owner}_{int(time.time())}"
            allocation = ResourceAllocation(
                resource_id=allocation_id,
                resource_type=resource_type,
                allocated_amount=amount,
                allocated_at=datetime.now(),
                owner=owner,
                metadata=metadata or {},
                priority=priority
            )
            
            self.active_allocations[allocation_id] = allocation
            self.allocation_history.append(allocation)
            
            # Log allocation for audit
            audit_logger.log_privacy_event(
                'resource_allocation',
                {'resource_type': resource_type.value, 'amount': amount},
                {'owner': owner, 'priority': priority},
                {'allocation_id': allocation_id}
            )
            
            logger.info(f"Allocated {amount} {resource_type.value} to {owner} (ID: {allocation_id})")
            return allocation_id
    
    def deallocate_resource(self, allocation_id: str) -> bool:
        """Deallocate a resource."""
        with self.lock:
            if allocation_id not in self.active_allocations:
                logger.warning(f"Allocation {allocation_id} not found")
                return False
            
            allocation = self.active_allocations[allocation_id]
            del self.active_allocations[allocation_id]
            
            # Log deallocation
            audit_logger.log_privacy_event(
                'resource_deallocation',
                {'resource_type': allocation.resource_type.value, 'amount': allocation.allocated_amount},
                {'owner': allocation.owner},
                {'allocation_id': allocation_id}
            )
            
            logger.info(f"Deallocated {allocation.allocated_amount} {allocation.resource_type.value} from {allocation.owner}")
            return True
    
    def _optimize_allocations(self, resource_type: ResourceType, needed_amount: float) -> bool:
        """Optimize existing allocations to free up resources."""
        with self.lock:
            # Find low-priority or expired allocations
            candidates_for_cleanup = []
            
            for allocation_id, allocation in self.active_allocations.items():
                if allocation.resource_type != resource_type:
                    continue
                
                # Check if expired
                if allocation.is_expired():
                    candidates_for_cleanup.append((allocation_id, allocation, 'expired'))
                # Check if low priority and we need space
                elif allocation.priority <= 3:  # Low priority threshold
                    candidates_for_cleanup.append((allocation_id, allocation, 'low_priority'))
            
            # Sort by priority (lowest first) and expiration
            candidates_for_cleanup.sort(key=lambda x: (x[2] == 'expired', x[1].priority))
            
            # Clean up allocations until we have enough space
            freed_amount = 0
            for allocation_id, allocation, reason in candidates_for_cleanup:
                if freed_amount >= needed_amount:
                    break
                
                logger.info(f"Cleaning up allocation {allocation_id} ({reason})")
                self.deallocate_resource(allocation_id)
                freed_amount += allocation.allocated_amount
            
            return freed_amount >= needed_amount
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of current allocations."""
        with self.lock:
            summary = {
                'total_allocations': len(self.active_allocations),
                'allocations_by_type': defaultdict(list),
                'allocations_by_owner': defaultdict(list),
                'total_allocated_by_type': defaultdict(float)
            }
            
            for allocation in self.active_allocations.values():
                resource_type = allocation.resource_type.value
                
                summary['allocations_by_type'][resource_type].append({
                    'id': allocation.resource_id,
                    'amount': allocation.allocated_amount,
                    'owner': allocation.owner,
                    'priority': allocation.priority,
                    'allocated_at': allocation.allocated_at.isoformat()
                })
                
                summary['allocations_by_owner'][allocation.owner].append({
                    'id': allocation.resource_id,
                    'type': resource_type,
                    'amount': allocation.allocated_amount,
                    'priority': allocation.priority
                })
                
                summary['total_allocated_by_type'][resource_type] += allocation.allocated_amount
            
            return dict(summary)


class DynamicScaler:
    """Handles dynamic scaling of resources based on usage patterns."""
    
    def __init__(self, resource_monitor: ResourceMonitor, resource_allocator: ResourceAllocator):
        self.resource_monitor = resource_monitor
        self.resource_allocator = resource_allocator
        self.scaling_policies: Dict[ResourceType, ScalingPolicy] = {}
        self.scaling_history: List[ScalingAction] = []
        self.scaling_active = False
        self.scaling_thread = None
        
        # Default scaling policies
        for resource_type in ResourceType:
            self.scaling_policies[resource_type] = ScalingPolicy.BALANCED
        
        self.lock = threading.RLock()
        
        # Register for resource events
        for resource_type in ResourceType:
            self.resource_monitor.register_resource_callback(
                'scale_up', resource_type, self._handle_scale_up_event
            )
            self.resource_monitor.register_resource_callback(
                'scale_down', resource_type, self._handle_scale_down_event
            )
            self.resource_monitor.register_resource_callback(
                'critical', resource_type, self._handle_critical_event
            )
        
        logger.info("Dynamic scaler initialized")
    
    def start_scaling(self):
        """Start dynamic scaling."""
        if self.scaling_active:
            return
        
        self.scaling_active = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        logger.info("Dynamic scaling started")
    
    def stop_scaling(self):
        """Stop dynamic scaling."""
        self.scaling_active = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5.0)
        
        logger.info("Dynamic scaling stopped")
    
    def _scaling_loop(self):
        """Main scaling loop."""
        while self.scaling_active:
            try:
                self._evaluate_scaling_opportunities()
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Dynamic scaling error: {e}")
                time.sleep(60)
    
    def _evaluate_scaling_opportunities(self):
        """Evaluate opportunities for resource scaling."""
        resource_summary = self.resource_monitor.get_resource_summary()
        
        for resource_type_str, metrics in resource_summary.items():
            try:
                resource_type = ResourceType(resource_type_str)
                policy = self.scaling_policies.get(resource_type, ScalingPolicy.BALANCED)
                
                if policy == ScalingPolicy.MANUAL:
                    continue
                
                # Check if scaling action is needed
                if metrics.get('predicted_exhaustion'):
                    exhaustion_time = datetime.fromisoformat(metrics['predicted_exhaustion'])
                    time_to_exhaustion = (exhaustion_time - datetime.now()).total_seconds()
                    
                    # If exhaustion predicted within next hour, consider scaling
                    if time_to_exhaustion < 3600:  # 1 hour
                        self._initiate_scaling_action(
                            resource_type, 'scale_up', 
                            f"Predicted exhaustion in {time_to_exhaustion/60:.1f} minutes"
                        )
                
                # Check for scale-down opportunities
                elif metrics['current_usage_percent'] < 30:  # Low usage
                    trend = metrics.get('trend', 0)
                    if trend < 0:  # Usage is decreasing
                        self._initiate_scaling_action(
                            resource_type, 'scale_down',
                            f"Low usage ({metrics['current_usage_percent']:.1f}%) with decreasing trend"
                        )
                
            except ValueError:
                continue  # Skip unknown resource types
    
    def _handle_scale_up_event(self, event_type: str, resource_type: ResourceType, metric: ResourceUsageMetrics):
        """Handle scale up events."""
        policy = self.scaling_policies.get(resource_type, ScalingPolicy.BALANCED)
        if policy == ScalingPolicy.MANUAL:
            return
        
        self._initiate_scaling_action(
            resource_type, 'scale_up',
            f"Resource usage at {metric.usage_percent:.1f}%"
        )
    
    def _handle_scale_down_event(self, event_type: str, resource_type: ResourceType, metric: ResourceUsageMetrics):
        """Handle scale down events."""
        policy = self.scaling_policies.get(resource_type, ScalingPolicy.BALANCED)
        if policy in [ScalingPolicy.MANUAL, ScalingPolicy.AGGRESSIVE]:
            return  # Don't scale down aggressively
        
        self._initiate_scaling_action(
            resource_type, 'scale_down',
            f"Resource usage at {metric.usage_percent:.1f}%"
        )
    
    def _handle_critical_event(self, event_type: str, resource_type: ResourceType, metric: ResourceUsageMetrics):
        """Handle critical resource events."""
        logger.critical(f"Critical resource usage: {resource_type.value} at {metric.usage_percent:.1f}%")
        
        # Immediate optimization and emergency scaling
        self._emergency_resource_optimization(resource_type)
        
        # Log critical event
        audit_logger.log_security_event(
            'resource_critical',
            'critical',
            f"Critical {resource_type.value} usage at {metric.usage_percent:.1f}%"
        )
    
    def _initiate_scaling_action(self, resource_type: ResourceType, action_type: str, reason: str):
        """Initiate a scaling action."""
        with self.lock:
            current_metric = self.resource_monitor.get_current_usage(resource_type)
            if not current_metric:
                return
            
            # Create scaling action
            action = ScalingAction(
                action_id=f"{action_type}_{resource_type.value}_{int(time.time())}",
                resource_type=resource_type,
                action_type=action_type,
                requested_by='dynamic_scaler',
                current_allocation=current_metric.used_amount,
                target_allocation=self._calculate_target_allocation(resource_type, action_type, current_metric),
                reason=reason
            )
            
            # Execute scaling action
            success = self._execute_scaling_action(action)
            action.executed = True
            action.result = 'success' if success else 'failed'
            
            self.scaling_history.append(action)
            
            logger.info(f"Scaling action {action.action_id}: {action.result}")
    
    def _calculate_target_allocation(self, resource_type: ResourceType, action_type: str, current_metric: ResourceUsageMetrics) -> float:
        """Calculate target allocation for scaling action."""
        policy = self.scaling_policies.get(resource_type, ScalingPolicy.BALANCED)
        current_used = current_metric.used_amount
        total = current_metric.total_amount
        
        if action_type == 'scale_up':
            if policy == ScalingPolicy.AGGRESSIVE:
                return min(total * 0.9, current_used * 1.5)  # Scale up to 50% more, max 90% of total
            elif policy == ScalingPolicy.CONSERVATIVE:
                return min(total * 0.8, current_used * 1.2)  # Scale up to 20% more, max 80% of total
            else:  # BALANCED
                return min(total * 0.85, current_used * 1.3)  # Scale up to 30% more, max 85% of total
        
        elif action_type == 'scale_down':
            if policy == ScalingPolicy.CONSERVATIVE:
                return max(current_used * 0.9, total * 0.1)  # Scale down to 90% of current, min 10% of total
            else:  # BALANCED (AGGRESSIVE doesn't scale down)
                return max(current_used * 0.8, total * 0.2)  # Scale down to 80% of current, min 20% of total
        
        return current_used
    
    def _execute_scaling_action(self, action: ScalingAction) -> bool:
        """Execute a scaling action."""
        try:
            if action.resource_type == ResourceType.MEMORY:
                return self._scale_memory(action)
            elif action.resource_type in [ResourceType.GPU, ResourceType.GPU_MEMORY]:
                return self._scale_gpu(action)
            elif action.resource_type == ResourceType.CPU:
                return self._scale_cpu(action)
            else:
                logger.warning(f"Scaling not implemented for {action.resource_type.value}")
                return False
            
        except Exception as e:
            logger.error(f"Scaling action execution failed: {e}")
            return False
    
    def _scale_memory(self, action: ScalingAction) -> bool:
        """Scale memory resources."""
        if action.action_type == 'scale_up':
            # Force garbage collection
            gc.collect()
            logger.info("Executed memory garbage collection")
            return True
        
        elif action.action_type == 'scale_down':
            # Optimize memory allocations
            return self.resource_allocator._optimize_allocations(
                ResourceType.MEMORY, 
                action.current_allocation - action.target_allocation
            )
        
        return False
    
    def _scale_gpu(self, action: ScalingAction) -> bool:
        """Scale GPU resources."""
        if not TORCH_AVAILABLE:
            return False
        
        try:
            if action.action_type == 'scale_up':
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared GPU memory cache")
                return True
            
            elif action.action_type == 'scale_down':
                # Optimize GPU memory allocations
                return self.resource_allocator._optimize_allocations(
                    action.resource_type,
                    action.current_allocation - action.target_allocation
                )
            
        except Exception as e:
            logger.error(f"GPU scaling error: {e}")
            return False
        
        return False
    
    def _scale_cpu(self, action: ScalingAction) -> bool:
        """Scale CPU resources."""
        # For CPU, we mainly optimize allocations
        if action.action_type == 'scale_down':
            return self.resource_allocator._optimize_allocations(
                ResourceType.CPU,
                action.current_allocation - action.target_allocation
            )
        
        logger.info("CPU scaling requested (limited options available)")
        return True
    
    def _emergency_resource_optimization(self, resource_type: ResourceType):
        """Emergency resource optimization for critical situations."""
        logger.warning(f"Initiating emergency optimization for {resource_type.value}")
        
        # Force immediate cleanup
        if resource_type == ResourceType.MEMORY:
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Aggressive allocation cleanup
        self.resource_allocator._optimize_allocations(resource_type, float('inf'))
        
        logger.warning("Emergency optimization completed")
    
    def set_scaling_policy(self, resource_type: ResourceType, policy: ScalingPolicy):
        """Set scaling policy for a resource type."""
        self.scaling_policies[resource_type] = policy
        logger.info(f"Set scaling policy for {resource_type.value} to {policy.value}")
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get scaling system summary."""
        with self.lock:
            return {
                'scaling_active': self.scaling_active,
                'policies': {rt.value: policy.value for rt, policy in self.scaling_policies.items()},
                'recent_actions': [
                    {
                        'action_id': action.action_id,
                        'resource_type': action.resource_type.value,
                        'action_type': action.action_type,
                        'timestamp': action.timestamp.isoformat(),
                        'executed': action.executed,
                        'result': action.result,
                        'reason': action.reason
                    }
                    for action in self.scaling_history[-10:]  # Last 10 actions
                ],
                'total_actions': len(self.scaling_history)
            }


class ComprehensiveResourceManager:
    """Main resource management system that orchestrates all components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize core components
        self.resource_monitor = ResourceMonitor(
            monitoring_interval=self.config.get('monitoring_interval', 5.0)
        )
        self.resource_allocator = ResourceAllocator(self.resource_monitor)
        self.dynamic_scaler = DynamicScaler(self.resource_monitor, self.resource_allocator)
        
        # System state
        self.resource_management_active = False
        self.emergency_mode = False
        
        # Configuration
        self.auto_scaling_enabled = self.config.get('auto_scaling_enabled', True)
        self.emergency_thresholds = self.config.get('emergency_thresholds', {
            ResourceType.MEMORY: 0.95,
            ResourceType.GPU_MEMORY: 0.95,
            ResourceType.CPU: 0.98
        })
        
        logger.info("Comprehensive resource manager initialized")
    
    def start_resource_management(self):
        """Start the complete resource management system."""
        if self.resource_management_active:
            return
        
        self.resource_management_active = True
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        
        # Start dynamic scaling if enabled
        if self.auto_scaling_enabled:
            self.dynamic_scaler.start_scaling()
        
        # Register emergency callbacks
        self._register_emergency_callbacks()
        
        logger.info("Resource management system started")
    
    def stop_resource_management(self):
        """Stop the resource management system."""
        self.resource_management_active = False
        
        # Stop components
        self.resource_monitor.stop_monitoring()
        self.dynamic_scaler.stop_scaling()
        
        logger.info("Resource management system stopped")
    
    def _register_emergency_callbacks(self):
        """Register emergency callbacks for critical resource situations."""
        for resource_type, threshold in self.emergency_thresholds.items():
            self.resource_monitor.register_resource_callback(
                'critical', resource_type, self._handle_emergency_situation
            )
    
    def _handle_emergency_situation(self, event_type: str, resource_type: ResourceType, metric: ResourceUsageMetrics):
        """Handle emergency resource situations."""
        if not self.emergency_mode:
            self.emergency_mode = True
            logger.critical(f"Entering emergency mode due to {resource_type.value} exhaustion")
            
            # Emergency actions
            try:
                # Immediate resource cleanup
                if resource_type == ResourceType.MEMORY:
                    self._emergency_memory_cleanup()
                elif resource_type in [ResourceType.GPU, ResourceType.GPU_MEMORY]:
                    self._emergency_gpu_cleanup()
                
                # Force aggressive allocation cleanup
                self.resource_allocator._optimize_allocations(resource_type, float('inf'))
                
                # Log emergency event
                audit_logger.log_security_event(
                    'resource_emergency',
                    'critical',
                    f"Emergency mode activated for {resource_type.value} at {metric.usage_percent:.1f}%"
                )
                
            except Exception as e:
                logger.critical(f"Emergency resource handling failed: {e}")
            
            finally:
                # Reset emergency mode after 5 minutes
                threading.Timer(300, self._reset_emergency_mode).start()
    
    def _emergency_memory_cleanup(self):
        """Emergency memory cleanup."""
        logger.warning("Executing emergency memory cleanup")
        
        # Multiple rounds of garbage collection
        for i in range(3):
            gc.collect()
            time.sleep(1)
        
        # Clear PyTorch cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _emergency_gpu_cleanup(self):
        """Emergency GPU cleanup."""
        if not TORCH_AVAILABLE:
            return
        
        logger.warning("Executing emergency GPU cleanup")
        
        try:
            # Clear all GPU caches
            torch.cuda.empty_cache()
            
            # Force synchronization
            torch.cuda.synchronize()
            
        except Exception as e:
            logger.error(f"Emergency GPU cleanup error: {e}")
    
    def _reset_emergency_mode(self):
        """Reset emergency mode."""
        self.emergency_mode = False
        logger.info("Emergency mode deactivated")
    
    def allocate_training_resources(
        self,
        memory_gb: float,
        gpu_memory_gb: float = 0.0,
        cpu_cores: float = 1.0,
        owner: str = "training_job",
        priority: int = 5
    ) -> Dict[str, Optional[str]]:
        """Allocate resources for training job."""
        allocations = {}
        
        # Allocate memory
        if memory_gb > 0:
            allocations['memory'] = self.resource_allocator.allocate_resource(
                ResourceType.MEMORY, memory_gb, owner, priority
            )
        
        # Allocate GPU memory
        if gpu_memory_gb > 0:
            allocations['gpu_memory'] = self.resource_allocator.allocate_resource(
                ResourceType.GPU_MEMORY, gpu_memory_gb, owner, priority
            )
        
        # Allocate CPU
        if cpu_cores > 0:
            allocations['cpu'] = self.resource_allocator.allocate_resource(
                ResourceType.CPU, cpu_cores, owner, priority
            )
        
        return allocations
    
    def deallocate_training_resources(self, allocations: Dict[str, Optional[str]]) -> bool:
        """Deallocate training resources."""
        success = True
        
        for resource_type, allocation_id in allocations.items():
            if allocation_id:
                if not self.resource_allocator.deallocate_resource(allocation_id):
                    success = False
        
        return success
    
    def optimize_for_privacy_training(self) -> Dict[str, Any]:
        """Optimize resource allocation for privacy-preserving training."""
        logger.info("Optimizing resources for privacy training")
        
        # Set conservative scaling policies for stability
        self.dynamic_scaler.set_scaling_policy(ResourceType.MEMORY, ScalingPolicy.CONSERVATIVE)
        self.dynamic_scaler.set_scaling_policy(ResourceType.GPU_MEMORY, ScalingPolicy.CONSERVATIVE)
        
        # Pre-allocate resources to avoid interruptions
        base_allocations = self.allocate_training_resources(
            memory_gb=2.0,  # Base memory allocation
            gpu_memory_gb=1.0 if self.resource_monitor.gpu_count > 0 else 0.0,
            cpu_cores=1.0,
            owner="privacy_training_base",
            priority=8
        )
        
        # Force cleanup to ensure optimal starting state
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        optimization_summary = {
            'scaling_policies_set': True,
            'base_allocations': base_allocations,
            'cleanup_performed': True,
            'optimization_timestamp': datetime.now().isoformat()
        }
        
        logger.info("Privacy training optimization completed")
        return optimization_summary
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive resource management status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'resource_management_active': self.resource_management_active,
            'emergency_mode': self.emergency_mode,
            'auto_scaling_enabled': self.auto_scaling_enabled,
            'resource_usage': self.resource_monitor.get_resource_summary(),
            'allocations': self.resource_allocator.get_allocation_summary(),
            'scaling_status': self.dynamic_scaler.get_scaling_summary(),
            'system_health': self._assess_system_health()
        }
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health from resource perspective."""
        resource_summary = self.resource_monitor.get_resource_summary()
        
        critical_resources = []
        warning_resources = []
        healthy_resources = []
        
        for resource_type, metrics in resource_summary.items():
            state = metrics.get('state', 'unknown')
            if state in ['critical', 'exhausted']:
                critical_resources.append(resource_type)
            elif state == 'warning':
                warning_resources.append(resource_type)
            else:
                healthy_resources.append(resource_type)
        
        if critical_resources:
            overall_health = 'critical'
        elif warning_resources:
            overall_health = 'warning'
        else:
            overall_health = 'healthy'
        
        return {
            'overall_health': overall_health,
            'critical_resources': critical_resources,
            'warning_resources': warning_resources,
            'healthy_resources': healthy_resources,
            'resource_count': len(resource_summary)
        }


# Global resource manager instance
resource_manager = ComprehensiveResourceManager()