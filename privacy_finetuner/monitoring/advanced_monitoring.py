"""Advanced monitoring and metrics collection for privacy-preserving ML training.

This module provides comprehensive monitoring capabilities including real-time metrics,
health checks, performance monitoring, security event correlation, and alerting.
"""

import time
import logging
import threading
import json
import queue
import psutil
import statistics
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import uuid

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: datetime
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Monitoring alert."""
    id: str
    timestamp: datetime
    level: AlertLevel
    title: str
    description: str
    source: str
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class Metric:
    """Base class for metrics collection."""
    
    def __init__(self, name: str, metric_type: MetricType, description: str = "",
                 labels: Dict[str, str] = None, max_points: int = 10000):
        self.name = name
        self.metric_type = metric_type
        self.description = description
        self.labels = labels or {}
        self.max_points = max_points
        self.points: List[MetricPoint] = []
        self.lock = threading.RLock()
        self.created_at = datetime.now()
        
    def add_point(self, value: Union[int, float], labels: Dict[str, str] = None,
                  metadata: Dict[str, Any] = None):
        """Add a data point to the metric."""
        with self.lock:
            point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                labels={**self.labels, **(labels or {})},
                metadata=metadata or {}
            )
            
            self.points.append(point)
            
            # Maintain max points limit
            if len(self.points) > self.max_points:
                self.points.pop(0)
    
    def get_latest_value(self) -> Optional[float]:
        """Get the latest metric value."""
        with self.lock:
            if not self.points:
                return None
            return self.points[-1].value
    
    def get_values_in_range(self, start_time: datetime, end_time: datetime) -> List[MetricPoint]:
        """Get values within a time range."""
        with self.lock:
            return [
                point for point in self.points
                if start_time <= point.timestamp <= end_time
            ]
    
    def get_statistics(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get statistical summary of recent values."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_points = [p for p in self.points if p.timestamp >= cutoff_time]
        
        if not recent_points:
            return {'count': 0, 'duration_minutes': duration_minutes}
        
        values = [p.value for p in recent_points]
        
        stats = {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'duration_minutes': duration_minutes,
            'latest': values[-1],
            'first_timestamp': recent_points[0].timestamp.isoformat(),
            'last_timestamp': recent_points[-1].timestamp.isoformat()
        }
        
        if len(values) > 1:
            stats['std'] = statistics.stdev(values)
            stats['median'] = statistics.median(values)
        else:
            stats['std'] = 0.0
            stats['median'] = values[0]
        
        # Calculate percentiles if we have enough data
        if len(values) >= 10:
            sorted_values = sorted(values)
            stats['p95'] = sorted_values[int(len(sorted_values) * 0.95)]
            stats['p99'] = sorted_values[int(len(sorted_values) * 0.99)]
        
        return stats


class CounterMetric(Metric):
    """Counter metric that only increases."""
    
    def __init__(self, name: str, description: str = "", labels: Dict[str, str] = None):
        super().__init__(name, MetricType.COUNTER, description, labels)
        self.total = 0.0
    
    def increment(self, amount: float = 1.0, labels: Dict[str, str] = None):
        """Increment the counter."""
        with self.lock:
            self.total += amount
            self.add_point(self.total, labels)
    
    def get_rate(self, duration_minutes: int = 5) -> float:
        """Get the rate of increase per minute."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_points = [p for p in self.points if p.timestamp >= cutoff_time]
        
        if len(recent_points) < 2:
            return 0.0
        
        first_point = recent_points[0]
        last_point = recent_points[-1]
        
        value_diff = last_point.value - first_point.value
        time_diff = (last_point.timestamp - first_point.timestamp).total_seconds() / 60
        
        return value_diff / time_diff if time_diff > 0 else 0.0


class GaugeMetric(Metric):
    """Gauge metric that can go up and down."""
    
    def __init__(self, name: str, description: str = "", labels: Dict[str, str] = None):
        super().__init__(name, MetricType.GAUGE, description, labels)
    
    def set(self, value: float, labels: Dict[str, str] = None):
        """Set the gauge value."""
        self.add_point(value, labels)
    
    def increment(self, amount: float = 1.0, labels: Dict[str, str] = None):
        """Increment the gauge."""
        current = self.get_latest_value() or 0.0
        self.add_point(current + amount, labels)
    
    def decrement(self, amount: float = 1.0, labels: Dict[str, str] = None):
        """Decrement the gauge."""
        current = self.get_latest_value() or 0.0
        self.add_point(current - amount, labels)


class HistogramMetric(Metric):
    """Histogram metric for distribution analysis."""
    
    def __init__(self, name: str, buckets: List[float] = None, description: str = "",
                 labels: Dict[str, str] = None):
        super().__init__(name, MetricType.HISTOGRAM, description, labels)
        self.buckets = buckets or [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
        self.bucket_counts = {bucket: 0 for bucket in self.buckets}
        self.sum = 0.0
        self.count = 0
    
    def observe(self, value: float, labels: Dict[str, str] = None):
        """Add an observation to the histogram."""
        with self.lock:
            self.sum += value
            self.count += 1
            
            # Find the appropriate bucket
            for bucket in self.buckets:
                if value <= bucket:
                    self.bucket_counts[bucket] += 1
            
            self.add_point(value, labels, {
                'bucket_counts': self.bucket_counts.copy(),
                'sum': self.sum,
                'count': self.count
            })
    
    def get_quantile(self, quantile: float) -> float:
        """Estimate quantile from bucket data."""
        if not self.points:
            return 0.0
        
        target_count = self.count * quantile
        cumulative_count = 0
        
        for bucket in sorted(self.buckets):
            cumulative_count += self.bucket_counts[bucket]
            if cumulative_count >= target_count:
                return bucket
        
        return self.buckets[-1]


class TimerMetric(Metric):
    """Timer metric for measuring durations."""
    
    def __init__(self, name: str, description: str = "", labels: Dict[str, str] = None):
        super().__init__(name, MetricType.TIMER, description, labels)
        self.active_timers: Dict[str, datetime] = {}
    
    def start(self, timer_id: str = None) -> str:
        """Start a timer and return timer ID."""
        if timer_id is None:
            timer_id = str(uuid.uuid4())
        
        with self.lock:
            self.active_timers[timer_id] = datetime.now()
        
        return timer_id
    
    def stop(self, timer_id: str, labels: Dict[str, str] = None) -> float:
        """Stop a timer and record the duration."""
        with self.lock:
            if timer_id not in self.active_timers:
                logger.warning(f"Timer {timer_id} not found")
                return 0.0
            
            start_time = self.active_timers.pop(timer_id)
            duration = (datetime.now() - start_time).total_seconds()
            
            self.add_point(duration, labels, {'timer_id': timer_id})
            return duration
    
    def time_context(self, labels: Dict[str, str] = None):
        """Context manager for timing operations."""
        return TimerContext(self, labels)


class TimerContext:
    """Context manager for timer metrics."""
    
    def __init__(self, timer_metric: TimerMetric, labels: Dict[str, str] = None):
        self.timer_metric = timer_metric
        self.labels = labels
        self.timer_id = None
    
    def __enter__(self):
        self.timer_id = self.timer_metric.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer_id:
            self.timer_metric.stop(self.timer_id, self.labels)


class HealthChecker:
    """System health monitoring."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self.health_history: List[Dict[str, Any]] = []
        self.monitoring = False
        self.monitor_thread = None
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.health_checks['system_resources'] = self._check_system_resources
        self.health_checks['disk_space'] = self._check_disk_space
        if TORCH_AVAILABLE:
            self.health_checks['gpu_health'] = self._check_gpu_health
    
    def register_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Register a custom health check."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                health_status = self.check_health()
                self.health_history.append(health_status)
                
                # Keep history manageable
                if len(self.health_history) > 1000:
                    self.health_history.pop(0)
                
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(self.check_interval)
    
    def check_health(self) -> Dict[str, Any]:
        """Run all health checks and return status."""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        unhealthy_count = 0
        
        for check_name, check_func in self.health_checks.items():
            try:
                check_result = check_func()
                check_result['status'] = check_result.get('status', 'healthy')
                health_status['checks'][check_name] = check_result
                
                if check_result['status'] != 'healthy':
                    unhealthy_count += 1
            except Exception as e:
                health_status['checks'][check_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                unhealthy_count += 1
        
        # Determine overall status
        if unhealthy_count == 0:
            health_status['overall_status'] = 'healthy'
        elif unhealthy_count <= len(self.health_checks) // 2:
            health_status['overall_status'] = 'degraded'
        else:
            health_status['overall_status'] = 'unhealthy'
        
        return health_status
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            status = 'healthy'
            warnings = []
            
            if memory.percent > 90:
                status = 'critical'
                warnings.append(f"High memory usage: {memory.percent:.1f}%")
            elif memory.percent > 80:
                status = 'warning'
                warnings.append(f"Elevated memory usage: {memory.percent:.1f}%")
            
            if cpu_percent > 95:
                status = 'critical' if status != 'critical' else status
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 85:
                status = 'warning' if status == 'healthy' else status
                warnings.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
            
            return {
                'status': status,
                'memory_percent': memory.percent,
                'cpu_percent': cpu_percent,
                'warnings': warnings
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space usage."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            status = 'healthy'
            warnings = []
            
            if usage_percent > 95:
                status = 'critical'
                warnings.append(f"Critical disk usage: {usage_percent:.1f}%")
            elif usage_percent > 85:
                status = 'warning'
                warnings.append(f"High disk usage: {usage_percent:.1f}%")
            
            return {
                'status': status,
                'disk_usage_percent': usage_percent,
                'free_gb': disk.free / (1024**3),
                'warnings': warnings
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _check_gpu_health(self) -> Dict[str, Any]:
        """Check GPU health and memory usage."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {'status': 'not_available', 'message': 'GPU not available'}
        
        try:
            gpu_count = torch.cuda.device_count()
            gpu_stats = []
            overall_status = 'healthy'
            
            for i in range(gpu_count):
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                
                memory_usage_percent = (memory_allocated / memory_total) * 100
                
                gpu_status = 'healthy'
                if memory_usage_percent > 95:
                    gpu_status = 'critical'
                    overall_status = 'critical'
                elif memory_usage_percent > 85:
                    gpu_status = 'warning'
                    if overall_status == 'healthy':
                        overall_status = 'warning'
                
                gpu_stats.append({
                    'device_id': i,
                    'status': gpu_status,
                    'memory_usage_percent': memory_usage_percent,
                    'memory_allocated_gb': memory_allocated / (1024**3),
                    'memory_total_gb': memory_total / (1024**3)
                })
            
            return {
                'status': overall_status,
                'gpu_count': gpu_count,
                'gpus': gpu_stats
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


class AlertManager:
    """Alert management and notification system."""
    
    def __init__(self, max_alerts: int = 10000):
        self.max_alerts = max_alerts
        self.alerts: List[Alert] = []
        self.alert_handlers: Dict[AlertLevel, List[Callable[[Alert], None]]] = {
            level: [] for level in AlertLevel
        }
        self.lock = threading.RLock()
    
    def create_alert(self, level: AlertLevel, title: str, description: str,
                    source: str = "unknown", labels: Dict[str, str] = None) -> Alert:
        """Create and store a new alert."""
        alert = Alert(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            level=level,
            title=title,
            description=description,
            source=source,
            labels=labels or {}
        )
        
        with self.lock:
            self.alerts.append(alert)
            
            # Maintain max alerts limit
            if len(self.alerts) > self.max_alerts:
                self.alerts.pop(0)
        
        # Trigger alert handlers
        self._trigger_handlers(alert)
        
        logger.log(
            self._level_to_log_level(level),
            f"Alert created: {title} - {description}"
        )
        
        return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert by ID."""
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    logger.info(f"Alert resolved: {alert.title}")
                    return True
        return False
    
    def register_handler(self, level: AlertLevel, handler: Callable[[Alert], None]):
        """Register an alert handler for a specific level."""
        self.alert_handlers[level].append(handler)
        logger.info(f"Registered alert handler for {level.value}")
    
    def _trigger_handlers(self, alert: Alert):
        """Trigger handlers for an alert."""
        for handler in self.alert_handlers[alert.level]:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def _level_to_log_level(self, alert_level: AlertLevel) -> int:
        """Convert alert level to logging level."""
        mapping = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        return mapping.get(alert_level, logging.INFO)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts."""
        with self.lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by severity level."""
        with self.lock:
            return [alert for alert in self.alerts if alert.level == level]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert statistics."""
        with self.lock:
            active_alerts = [a for a in self.alerts if not a.resolved]
            
            level_counts = {level.value: 0 for level in AlertLevel}
            for alert in active_alerts:
                level_counts[alert.level.value] += 1
            
            return {
                'total_alerts': len(self.alerts),
                'active_alerts': len(active_alerts),
                'resolved_alerts': len(self.alerts) - len(active_alerts),
                'alert_counts_by_level': level_counts,
                'oldest_unresolved': min(
                    (a.timestamp for a in active_alerts), 
                    default=None
                )
            }


class ComprehensiveMonitor:
    """Main monitoring system that combines all monitoring capabilities."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics: Dict[str, Metric] = {}
        self.health_checker = HealthChecker(
            check_interval=self.config.get('health_check_interval', 30)
        )
        self.alert_manager = AlertManager(
            max_alerts=self.config.get('max_alerts', 10000)
        )
        
        # Monitoring state
        self.monitoring_active = False
        self.start_time = None
        
        # Setup default metrics
        self._setup_default_metrics()
        
        # Setup alert thresholds
        self._setup_alert_thresholds()
        
        logger.info("ComprehensiveMonitor initialized")
    
    def _setup_default_metrics(self):
        """Setup default system metrics."""
        self.register_counter('system.errors', 'Total system errors')
        self.register_counter('training.steps', 'Total training steps')
        self.register_gauge('system.memory_usage', 'System memory usage percentage')
        self.register_gauge('system.cpu_usage', 'System CPU usage percentage')
        self.register_gauge('privacy.epsilon_spent', 'Privacy epsilon spent')
        self.register_histogram('training.step_duration', 'Training step duration')
        self.register_timer('operation.duration', 'Operation duration timer')
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.register_gauge('gpu.memory_usage', 'GPU memory usage percentage')
    
    def _setup_alert_thresholds(self):
        """Setup default alert thresholds."""
        self.alert_thresholds = {
            'memory_critical': 95.0,
            'memory_warning': 85.0,
            'cpu_critical': 98.0,
            'cpu_warning': 90.0,
            'privacy_budget_critical': 0.95,
            'privacy_budget_warning': 0.80,
        }
    
    def start_monitoring(self):
        """Start all monitoring components."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.start_time = datetime.now()
        
        # Start health monitoring
        self.health_checker.start_monitoring()
        
        # Start metric collection
        self._start_metric_collection()
        
        logger.info("Comprehensive monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.monitoring_active = False
        
        # Stop health monitoring
        self.health_checker.stop_monitoring()
        
        logger.info("Comprehensive monitoring stopped")
    
    def _start_metric_collection(self):
        """Start automatic metric collection."""
        def collect_system_metrics():
            while self.monitoring_active:
                try:
                    # Collect system metrics
                    memory = psutil.virtual_memory()
                    cpu_percent = psutil.cpu_percent()
                    
                    self.get_gauge('system.memory_usage').set(memory.percent)
                    self.get_gauge('system.cpu_usage').set(cpu_percent)
                    
                    # Check thresholds and create alerts
                    if memory.percent > self.alert_thresholds['memory_critical']:
                        self.alert_manager.create_alert(
                            AlertLevel.CRITICAL,
                            "Critical Memory Usage",
                            f"Memory usage at {memory.percent:.1f}%",
                            "system_monitor"
                        )
                    elif memory.percent > self.alert_thresholds['memory_warning']:
                        self.alert_manager.create_alert(
                            AlertLevel.WARNING,
                            "High Memory Usage",
                            f"Memory usage at {memory.percent:.1f}%",
                            "system_monitor"
                        )
                    
                    # GPU metrics if available
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        try:
                            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                            self.get_gauge('gpu.memory_usage').set(gpu_memory)
                        except:
                            pass
                    
                    time.sleep(5)  # Collect every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Metric collection error: {e}")
                    time.sleep(10)
        
        metric_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        metric_thread.start()
    
    def register_counter(self, name: str, description: str = "", labels: Dict[str, str] = None) -> CounterMetric:
        """Register a counter metric."""
        metric = CounterMetric(name, description, labels)
        self.metrics[name] = metric
        return metric
    
    def register_gauge(self, name: str, description: str = "", labels: Dict[str, str] = None) -> GaugeMetric:
        """Register a gauge metric."""
        metric = GaugeMetric(name, description, labels)
        self.metrics[name] = metric
        return metric
    
    def register_histogram(self, name: str, buckets: List[float] = None, description: str = "",
                          labels: Dict[str, str] = None) -> HistogramMetric:
        """Register a histogram metric."""
        metric = HistogramMetric(name, buckets, description, labels)
        self.metrics[name] = metric
        return metric
    
    def register_timer(self, name: str, description: str = "", labels: Dict[str, str] = None) -> TimerMetric:
        """Register a timer metric."""
        metric = TimerMetric(name, description, labels)
        self.metrics[name] = metric
        return metric
    
    def get_counter(self, name: str) -> CounterMetric:
        """Get a counter metric."""
        return self.metrics.get(name)
    
    def get_gauge(self, name: str) -> GaugeMetric:
        """Get a gauge metric."""
        return self.metrics.get(name)
    
    def get_histogram(self, name: str) -> HistogramMetric:
        """Get a histogram metric."""
        return self.metrics.get(name)
    
    def get_timer(self, name: str) -> TimerMetric:
        """Get a timer metric."""
        return self.metrics.get(name)
    
    def record_training_step(self, step: int, loss: float, privacy_spent: float, duration: float):
        """Record training step metrics."""
        self.get_counter('training.steps').increment()
        self.get_gauge('privacy.epsilon_spent').set(privacy_spent)
        self.get_histogram('training.step_duration').observe(duration)
        
        # Check privacy budget alerts
        if hasattr(self, 'privacy_config') and self.privacy_config:
            budget_ratio = privacy_spent / self.privacy_config.epsilon
            if budget_ratio > self.alert_thresholds['privacy_budget_critical']:
                self.alert_manager.create_alert(
                    AlertLevel.CRITICAL,
                    "Privacy Budget Exhaustion",
                    f"Privacy budget {budget_ratio:.1%} exhausted",
                    "privacy_monitor"
                )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': 'active' if self.monitoring_active else 'inactive',
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'metrics': {},
            'health': self.health_checker.check_health(),
            'alerts': self.alert_manager.get_alert_summary()
        }
        
        # Add metric summaries
        for name, metric in self.metrics.items():
            dashboard_data['metrics'][name] = {
                'type': metric.metric_type.value,
                'latest_value': metric.get_latest_value(),
                'statistics': metric.get_statistics(duration_minutes=60)
            }
        
        return dashboard_data
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format."""
        if format == 'json':
            return json.dumps(self.get_dashboard_data(), indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def create_report(self, duration_hours: int = 24) -> Dict[str, Any]:
        """Create comprehensive monitoring report."""
        cutoff_time = datetime.now() - timedelta(hours=duration_hours)
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'report_period_hours': duration_hours,
            'monitoring_summary': {
                'active': self.monitoring_active,
                'total_metrics': len(self.metrics),
                'total_alerts': len(self.alert_manager.alerts)
            },
            'health_summary': self.health_checker.check_health(),
            'alert_summary': self.alert_manager.get_alert_summary(),
            'metric_summaries': {}
        }
        
        # Add detailed metric summaries
        for name, metric in self.metrics.items():
            report['metric_summaries'][name] = {
                'type': metric.metric_type.value,
                'description': metric.description,
                'statistics_1h': metric.get_statistics(60),
                'statistics_24h': metric.get_statistics(60 * 24)
            }
        
        return report


# Global monitoring instance
default_monitor = ComprehensiveMonitor()


def get_monitor() -> ComprehensiveMonitor:
    """Get the default monitor instance."""
    return default_monitor


def start_monitoring():
    """Start default monitoring."""
    default_monitor.start_monitoring()


def stop_monitoring():
    """Stop default monitoring."""
    default_monitor.stop_monitoring()