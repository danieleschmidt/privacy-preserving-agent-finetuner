"""Enhanced metrics collection and monitoring system.

This module provides comprehensive metrics tracking for privacy-preserving training,
including privacy budget consumption, model performance, and system resources.
"""

import time
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import os

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class MetricEvent:
    """Individual metric event with timestamp and metadata."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """Comprehensive training session metrics."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Privacy metrics
    epsilon_spent: float = 0.0
    delta_used: float = 0.0
    privacy_budget_remaining: float = 0.0
    noise_scale: float = 0.0
    
    # Training metrics
    epochs_completed: int = 0
    total_steps: int = 0
    final_loss: float = 0.0
    best_loss: float = float('inf')
    convergence_steps: Optional[int] = None
    
    # Performance metrics
    training_time: Optional[float] = None
    tokens_per_second: float = 0.0
    memory_peak_mb: float = 0.0
    cpu_utilization_avg: float = 0.0
    gpu_utilization_avg: float = 0.0
    
    # Data metrics
    samples_processed: int = 0
    validation_accuracy: Optional[float] = None
    privacy_leakage_score: float = 0.0


class MetricsCollector:
    """Advanced metrics collection system with multiple backends."""
    
    def __init__(
        self,
        enable_prometheus: bool = True,
        enable_file_export: bool = True,
        metrics_dir: str = "./metrics",
        retention_days: int = 30,
        collection_interval: float = 1.0
    ):
        """Initialize metrics collector.
        
        Args:
            enable_prometheus: Enable Prometheus metrics export
            enable_file_export: Enable file-based metrics export
            metrics_dir: Directory for metrics files
            retention_days: Days to retain metrics
            collection_interval: Metrics collection interval in seconds
        """
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.enable_file_export = enable_file_export
        self.metrics_dir = metrics_dir
        self.retention_days = retention_days
        self.collection_interval = collection_interval
        
        # Metrics storage
        self.events: List[MetricEvent] = []
        self.training_sessions: Dict[str, TrainingMetrics] = {}
        self.current_session: Optional[str] = None
        
        # Time-series data (last 1000 points)
        self.time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # System monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._system_stats = {}
        
        # Prometheus metrics
        self._prometheus_registry: Optional[CollectorRegistry] = None
        self._prometheus_metrics: Dict[str, Any] = {}
        
        # Event callbacks
        self._event_callbacks: List[Callable[[MetricEvent], None]] = []
        
        # Initialize metrics directory
        if self.enable_file_export:
            os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Setup Prometheus if available
        if self.enable_prometheus:
            self._setup_prometheus()
        
        # Start system monitoring
        self.start_system_monitoring()
    
    def _setup_prometheus(self) -> None:
        """Setup Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._prometheus_registry = CollectorRegistry()
        
        # Privacy metrics
        self._prometheus_metrics['privacy_epsilon_spent'] = Gauge(
            'privacy_epsilon_spent',
            'Total epsilon spent in current training session',
            registry=self._prometheus_registry
        )
        
        self._prometheus_metrics['privacy_budget_remaining'] = Gauge(
            'privacy_budget_remaining',
            'Remaining privacy budget',
            registry=self._prometheus_registry
        )
        
        # Training metrics
        self._prometheus_metrics['training_loss'] = Gauge(
            'training_loss',
            'Current training loss',
            registry=self._prometheus_registry
        )
        
        self._prometheus_metrics['training_steps'] = Counter(
            'training_steps_total',
            'Total training steps completed',
            registry=self._prometheus_registry
        )
        
        self._prometheus_metrics['training_duration'] = Histogram(
            'training_duration_seconds',
            'Training duration in seconds',
            registry=self._prometheus_registry
        )
        
        # System metrics
        self._prometheus_metrics['memory_usage'] = Gauge(
            'memory_usage_bytes',
            'Current memory usage',
            registry=self._prometheus_registry
        )
        
        self._prometheus_metrics['cpu_utilization'] = Gauge(
            'cpu_utilization_percent',
            'CPU utilization percentage',
            registry=self._prometheus_registry
        )
        
        # Start Prometheus HTTP server
        try:
            start_http_server(8000, registry=self._prometheus_registry)
        except Exception as e:
            print(f"Warning: Failed to start Prometheus server: {e}")
    
    def start_training_session(self, session_id: str, initial_config: Dict[str, Any]) -> None:
        """Start a new training session."""
        self.current_session = session_id
        self.training_sessions[session_id] = TrainingMetrics(
            session_id=session_id,
            start_time=datetime.now(),
            epsilon_spent=0.0,
            privacy_budget_remaining=initial_config.get('epsilon', 1.0)
        )
        
        # Record session start event
        self.record_event(
            'training_session_start',
            1.0,
            tags={'session_id': session_id},
            metadata=initial_config
        )
    
    def end_training_session(self, session_id: Optional[str] = None) -> TrainingMetrics:
        """End a training session and return final metrics."""
        target_session = session_id or self.current_session
        
        if target_session and target_session in self.training_sessions:
            session = self.training_sessions[target_session]
            session.end_time = datetime.now()
            
            if session.start_time:
                session.training_time = (session.end_time - session.start_time).total_seconds()
            
            # Record session end event
            self.record_event(
                'training_session_end',
                session.training_time or 0,
                tags={'session_id': target_session},
                metadata={'final_metrics': session.__dict__}
            )
            
            # Export session metrics
            if self.enable_file_export:
                self._export_session_metrics(session)
            
            return session
        
        raise ValueError(f"Training session not found: {target_session}")
    
    def record_event(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a metric event."""
        event = MetricEvent(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.events.append(event)
        self.time_series[name].append((event.timestamp, value))
        
        # Update Prometheus metrics
        if self.enable_prometheus and name in self._prometheus_metrics:
            metric = self._prometheus_metrics[name]
            if hasattr(metric, 'set'):
                metric.set(value)
            elif hasattr(metric, 'inc'):
                metric.inc(value)
        
        # Update current training session
        if self.current_session:
            self._update_session_metrics(name, value, tags, metadata)
        
        # Trigger callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Warning: Metrics callback failed: {e}")
    
    def _update_session_metrics(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Update current training session metrics."""
        session = self.training_sessions.get(self.current_session)
        if not session:
            return
        
        # Map metric names to session attributes
        metric_mappings = {
            'privacy_epsilon_spent': 'epsilon_spent',
            'privacy_budget_remaining': 'privacy_budget_remaining',
            'training_loss': 'final_loss',
            'training_steps': 'total_steps',
            'epochs_completed': 'epochs_completed',
            'memory_usage': 'memory_peak_mb',
            'tokens_per_second': 'tokens_per_second',
            'validation_accuracy': 'validation_accuracy',
            'privacy_leakage_score': 'privacy_leakage_score'
        }
        
        if name in metric_mappings:
            attr_name = metric_mappings[name]
            
            if name in ['memory_usage']:
                # Track peak values
                current_value = getattr(session, attr_name, 0)
                setattr(session, attr_name, max(current_value, value))
            elif name in ['training_steps', 'epochs_completed']:
                # Track cumulative values
                setattr(session, attr_name, int(value))
            else:
                # Track latest value
                setattr(session, attr_name, value)
        
        # Update best loss
        if name == 'training_loss' and value < session.best_loss:
            session.best_loss = value
    
    def start_system_monitoring(self) -> None:
        """Start background system monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._system_monitor_loop, daemon=True)
        self._monitoring_thread.start()
    
    def stop_system_monitoring(self) -> None:
        """Stop background system monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
    
    def _system_monitor_loop(self) -> None:
        """Background system monitoring loop."""
        while self._monitoring_active:
            try:
                if PSUTIL_AVAILABLE:
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=None)
                    self.record_event('cpu_utilization', cpu_percent)
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.record_event('memory_usage', memory.used)
                    self.record_event('memory_percent', memory.percent)
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    self.record_event('disk_usage', disk.used)
                    self.record_event('disk_percent', disk.percent)
                    
                    # Network metrics (if available)
                    try:
                        network = psutil.net_io_counters()
                        self.record_event('network_bytes_sent', network.bytes_sent)
                        self.record_event('network_bytes_recv', network.bytes_recv)
                    except Exception:
                        pass  # Network monitoring might not be available
                else:
                    # Fallback metrics when psutil is not available
                    self.record_event('cpu_utilization', 0.0)
                    self.record_event('memory_percent', 0.0)
                
                # GPU metrics (basic NVIDIA support)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        self.record_event(f'gpu_{i}_utilization', gpu.load * 100)
                        self.record_event(f'gpu_{i}_memory_used', gpu.memoryUsed)
                        self.record_event(f'gpu_{i}_temperature', gpu.temperature)
                except ImportError:
                    pass  # GPU monitoring not available
                
            except Exception as e:
                print(f"Warning: System monitoring error: {e}")
            
            time.sleep(self.collection_interval)
    
    def get_training_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        target_session = session_id or self.current_session
        
        if not target_session or target_session not in self.training_sessions:
            return {}
        
        session = self.training_sessions[target_session]
        
        # Calculate additional metrics
        duration = (
            (session.end_time or datetime.now()) - session.start_time
        ).total_seconds() if session.start_time else 0
        
        # Get recent system metrics
        recent_cpu = self._get_recent_metric_avg('cpu_utilization', 60)
        recent_memory = self._get_recent_metric_avg('memory_percent', 60)
        
        return {
            'session_id': session.session_id,
            'duration_seconds': duration,
            'privacy': {
                'epsilon_spent': session.epsilon_spent,
                'budget_remaining': session.privacy_budget_remaining,
                'privacy_efficiency': (
                    session.epsilon_spent / duration if duration > 0 else 0
                )
            },
            'training': {
                'epochs_completed': session.epochs_completed,
                'total_steps': session.total_steps,
                'final_loss': session.final_loss,
                'best_loss': session.best_loss,
                'convergence_rate': (
                    session.convergence_steps / session.total_steps 
                    if session.convergence_steps and session.total_steps > 0 else None
                )
            },
            'performance': {
                'tokens_per_second': session.tokens_per_second,
                'cpu_utilization_avg': recent_cpu,
                'memory_utilization_avg': recent_memory,
                'memory_peak_mb': session.memory_peak_mb
            },
            'data_quality': {
                'samples_processed': session.samples_processed,
                'validation_accuracy': session.validation_accuracy,
                'privacy_leakage_score': session.privacy_leakage_score
            }
        }
    
    def _get_recent_metric_avg(self, metric_name: str, seconds: int) -> float:
        """Get average of recent metric values."""
        if metric_name not in self.time_series:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        recent_values = [
            value for timestamp, value in self.time_series[metric_name]
            if timestamp >= cutoff_time
        ]
        
        return sum(recent_values) / len(recent_values) if recent_values else 0.0
    
    def _export_session_metrics(self, session: TrainingMetrics) -> None:
        """Export session metrics to file."""
        try:
            metrics_file = os.path.join(
                self.metrics_dir,
                f"training_session_{session.session_id}.json"
            )
            
            session_data = {
                **session.__dict__,
                'start_time': session.start_time.isoformat() if session.start_time else None,
                'end_time': session.end_time.isoformat() if session.end_time else None
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Warning: Failed to export session metrics: {e}")
    
    def add_event_callback(self, callback: Callable[[MetricEvent], None]) -> None:
        """Add callback for metric events."""
        self._event_callbacks.append(callback)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for monitoring dashboards."""
        current_time = datetime.now()
        
        # Get recent metrics for dashboard
        dashboard_data = {
            'timestamp': current_time.isoformat(),
            'system': {
                'cpu_percent': self._get_recent_metric_avg('cpu_utilization', 30),
                'memory_percent': self._get_recent_metric_avg('memory_percent', 30),
                'disk_percent': self._get_recent_metric_avg('disk_percent', 30)
            },
            'training': {
                'active_sessions': len([
                    s for s in self.training_sessions.values() 
                    if s.end_time is None
                ]),
                'total_sessions': len(self.training_sessions)
            }
        }
        
        # Add current session data if available
        if self.current_session:
            dashboard_data['current_session'] = self.get_training_summary()
        
        return dashboard_data
    
    def cleanup_old_metrics(self) -> None:
        """Clean up old metrics files and data."""
        if not self.enable_file_export:
            return
        
        try:
            cutoff_time = datetime.now() - timedelta(days=self.retention_days)
            
            # Clean up old event data
            self.events = [
                event for event in self.events 
                if event.timestamp >= cutoff_time
            ]
            
            # Clean up old files
            for filename in os.listdir(self.metrics_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.metrics_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        
        except Exception as e:
            print(f"Warning: Failed to cleanup old metrics: {e}")


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_collector
    
    if _global_collector is None:
        _global_collector = MetricsCollector()
    
    return _global_collector


def record_metric(name: str, value: float, **kwargs) -> None:
    """Convenience function to record a metric."""
    collector = get_metrics_collector()
    collector.record_event(name, value, **kwargs)


def start_training_metrics(session_id: str, config: Dict[str, Any]) -> None:
    """Start training metrics collection."""
    collector = get_metrics_collector()
    collector.start_training_session(session_id, config)


def end_training_metrics(session_id: Optional[str] = None) -> TrainingMetrics:
    """End training metrics collection."""
    collector = get_metrics_collector()
    return collector.end_training_session(session_id)