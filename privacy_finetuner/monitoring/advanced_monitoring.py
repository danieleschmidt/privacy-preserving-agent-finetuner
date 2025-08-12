"""Advanced monitoring and observability for privacy-preserving ML systems.

This module provides comprehensive monitoring capabilities including:
- Real-time privacy budget tracking with alerts
- Performance monitoring with SLA tracking
- Security event detection and response
- Distributed tracing for complex workflows
- Custom metrics and dashboards
- Anomaly detection for privacy leakage
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
from queue import Queue
import hashlib

# Handle imports gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    
    class NumpyStub:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        
        @staticmethod
        def std(data):
            if not data:
                return 0
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            return variance ** 0.5
        
        @staticmethod
        def percentile(data, p):
            if not data:
                return 0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p / 100
            f = int(k)
            c = k - f
            if f == len(sorted_data) - 1:
                return sorted_data[f]
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    np = NumpyStub()

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: datetime
    name: str
    value: float
    tags: Dict[str, str]
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Alert:
    """Alert configuration and state."""
    name: str
    condition: str
    threshold: float
    severity: str  # "low", "medium", "high", "critical"
    description: str
    enabled: bool = True
    cooldown_minutes: int = 5
    last_triggered: Optional[datetime] = None
    
    def should_trigger(self, value: float) -> bool:
        """Check if alert should trigger."""
        if not self.enabled:
            return False
        
        # Check cooldown
        if (self.last_triggered and 
            datetime.now() - self.last_triggered < timedelta(minutes=self.cooldown_minutes)):
            return False
        
        # Evaluate condition
        if self.condition == "greater_than":
            return value > self.threshold
        elif self.condition == "less_than":
            return value < self.threshold
        elif self.condition == "equals":
            return abs(value - self.threshold) < 1e-6
        
        return False


@dataclass
class PrivacyBudgetStatus:
    """Privacy budget tracking status."""
    total_budget: float
    consumed_budget: float
    remaining_budget: float
    consumption_rate: float  # per hour
    estimated_depletion: Optional[datetime]
    risk_level: str  # "low", "medium", "high"
    
    @property
    def utilization_percentage(self) -> float:
        """Calculate budget utilization percentage."""
        if self.total_budget <= 0:
            return 0.0
        return (self.consumed_budget / self.total_budget) * 100


class AdvancedMonitoringSystem:
    """Advanced monitoring system for privacy-preserving ML.
    
    Features:
    - Real-time metrics collection and aggregation
    - Privacy budget tracking with predictive alerts
    - Performance monitoring with SLA tracking
    - Security event monitoring
    - Anomaly detection
    - Custom dashboards and reporting
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        storage_backend: str = "in_memory",
        alert_handlers: Optional[List[Callable]] = None
    ):
        """Initialize advanced monitoring system.
        
        Args:
            config: Monitoring configuration
            storage_backend: Storage backend for metrics
            alert_handlers: List of alert handler functions
        """
        self.config = config or {}
        self.storage_backend = storage_backend
        self.alert_handlers = alert_handlers or []
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.privacy_tracker = PrivacyBudgetTracker()
        self.performance_monitor = PerformanceMonitor()
        self.security_monitor = SecurityMonitor()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager(self.alert_handlers)
        
        # Storage
        self.metrics_storage = MetricsStorage(storage_backend)
        
        # Background processing
        self.processing_queue = Queue()
        self.processing_thread = threading.Thread(target=self._process_metrics, daemon=True)
        self.is_running = False
        
        # Built-in alerts
        self._setup_default_alerts()
        
        logger.info("Advanced monitoring system initialized")
    
    def start(self):
        """Start monitoring system."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread.start()
        
        # Start component monitoring
        self.privacy_tracker.start()
        self.performance_monitor.start()
        self.security_monitor.start()
        self.anomaly_detector.start()
        
        logger.info("Monitoring system started")
    
    def stop(self):
        """Stop monitoring system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop components
        self.privacy_tracker.stop()
        self.performance_monitor.stop()
        self.security_monitor.stop()
        self.anomaly_detector.stop()
        
        # Wait for processing to complete
        self.processing_thread.join(timeout=5.0)
        
        logger.info("Monitoring system stopped")
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        unit: str = ""
    ):
        """Record a metric point."""
        metric = MetricPoint(
            timestamp=datetime.now(),
            name=name,
            value=value,
            tags=tags or {},
            unit=unit
        )
        
        self.processing_queue.put(("metric", metric))
    
    def record_privacy_event(
        self,
        event_type: str,
        epsilon_consumed: float,
        delta_consumed: float = 0.0,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record privacy budget consumption event."""
        event = {
            "type": "privacy_event",
            "event_type": event_type,
            "epsilon_consumed": epsilon_consumed,
            "delta_consumed": delta_consumed,
            "context": context or {},
            "timestamp": datetime.now()
        }
        
        self.processing_queue.put(("privacy_event", event))
    
    def record_performance_event(
        self,
        operation: str,
        duration: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record performance event."""
        event = {
            "type": "performance_event",
            "operation": operation,
            "duration": duration,
            "success": success,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        }
        
        self.processing_queue.put(("performance_event", event))
    
    def record_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record security event."""
        event = {
            "type": "security_event",
            "event_type": event_type,
            "severity": severity,
            "description": description,
            "source": source,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        }
        
        self.processing_queue.put(("security_event", event))
    
    def get_privacy_status(self) -> PrivacyBudgetStatus:
        """Get current privacy budget status."""
        return self.privacy_tracker.get_status()
    
    def get_performance_summary(
        self,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get performance summary."""
        return self.performance_monitor.get_summary(time_window)
    
    def get_security_events(
        self,
        severity: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """Get security events."""
        return self.security_monitor.get_events(severity, time_window)
    
    def get_anomalies(
        self,
        metric_name: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """Get detected anomalies."""
        return self.anomaly_detector.get_anomalies(metric_name, time_window)
    
    def create_dashboard(self, dashboard_config: Dict[str, Any]) -> str:
        """Create monitoring dashboard."""
        dashboard_id = hashlib.md5(
            json.dumps(dashboard_config, sort_keys=True).encode()
        ).hexdigest()[:12]
        
        # Store dashboard configuration
        self.metrics_storage.store_dashboard(dashboard_id, dashboard_config)
        
        logger.info(f"Created dashboard: {dashboard_id}")
        return dashboard_id
    
    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard data."""
        return self.metrics_storage.get_dashboard_data(dashboard_id)
    
    def export_metrics(
        self,
        format: str = "json",
        time_range: Optional[Tuple[datetime, datetime]] = None,
        metric_names: Optional[List[str]] = None
    ) -> str:
        """Export metrics in specified format."""
        return self.metrics_storage.export_metrics(format, time_range, metric_names)
    
    def _setup_default_alerts(self):
        """Setup default monitoring alerts."""
        default_alerts = [
            Alert(
                name="privacy_budget_high",
                condition="greater_than",
                threshold=0.8,
                severity="high",
                description="Privacy budget utilization above 80%"
            ),
            Alert(
                name="privacy_budget_critical",
                condition="greater_than",
                threshold=0.95,
                severity="critical",
                description="Privacy budget utilization above 95%"
            ),
            Alert(
                name="response_time_high",
                condition="greater_than",
                threshold=5.0,
                severity="medium",
                description="Response time above 5 seconds"
            ),
            Alert(
                name="error_rate_high",
                condition="greater_than",
                threshold=0.05,
                severity="high",
                description="Error rate above 5%"
            ),
            Alert(
                name="memory_usage_high",
                condition="greater_than",
                threshold=0.85,
                severity="medium",
                description="Memory usage above 85%"
            )
        ]
        
        for alert in default_alerts:
            self.alert_manager.add_alert(alert)
    
    def _process_metrics(self):
        """Process metrics in background thread."""
        while self.is_running:
            try:
                if not self.processing_queue.empty():
                    event_type, data = self.processing_queue.get(timeout=1.0)
                    
                    if event_type == "metric":
                        self._handle_metric(data)
                    elif event_type == "privacy_event":
                        self._handle_privacy_event(data)
                    elif event_type == "performance_event":
                        self._handle_performance_event(data)
                    elif event_type == "security_event":
                        self._handle_security_event(data)
                
            except Exception as e:
                logger.error(f"Error processing metric: {e}")
                continue
            
            time.sleep(0.1)  # Small delay to prevent busy waiting
    
    def _handle_metric(self, metric: MetricPoint):
        """Handle individual metric."""
        # Store metric
        self.metrics_storage.store_metric(metric)
        
        # Check for alerts
        self.alert_manager.check_metric_alerts(metric)
        
        # Update anomaly detection
        self.anomaly_detector.update(metric)
    
    def _handle_privacy_event(self, event: Dict[str, Any]):
        """Handle privacy event."""
        self.privacy_tracker.record_consumption(
            event["epsilon_consumed"],
            event["delta_consumed"],
            event["context"]
        )
        
        # Create metric for privacy consumption
        privacy_metric = MetricPoint(
            timestamp=event["timestamp"],
            name="privacy_epsilon_consumed",
            value=event["epsilon_consumed"],
            tags={"event_type": event["event_type"]},
            unit="epsilon"
        )
        
        self._handle_metric(privacy_metric)
    
    def _handle_performance_event(self, event: Dict[str, Any]):
        """Handle performance event."""
        self.performance_monitor.record_event(event)
        
        # Create metrics for performance
        duration_metric = MetricPoint(
            timestamp=event["timestamp"],
            name="operation_duration",
            value=event["duration"],
            tags={
                "operation": event["operation"],
                "success": str(event["success"])
            },
            unit="seconds"
        )
        
        self._handle_metric(duration_metric)
    
    def _handle_security_event(self, event: Dict[str, Any]):
        """Handle security event."""
        self.security_monitor.record_event(event)
        
        # Create alert for high-severity security events
        if event["severity"] in ["high", "critical"]:
            self.alert_manager.trigger_alert(
                f"security_event_{event['event_type']}",
                event["description"],
                event["severity"]
            )


class MetricsCollector:
    """Collects system and application metrics."""
    
    def __init__(self):
        self.collectors = {
            "system": self._collect_system_metrics,
            "privacy": self._collect_privacy_metrics,
            "performance": self._collect_performance_metrics
        }
    
    def collect_all(self) -> List[MetricPoint]:
        """Collect all available metrics."""
        metrics = []
        
        for collector_name, collector_func in self.collectors.items():
            try:
                collector_metrics = collector_func()
                metrics.extend(collector_metrics)
            except Exception as e:
                logger.warning(f"Failed to collect {collector_name} metrics: {e}")
        
        return metrics
    
    def _collect_system_metrics(self) -> List[MetricPoint]:
        """Collect system metrics."""
        metrics = []
        now = datetime.now()
        
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(MetricPoint(
                timestamp=now,
                name="cpu_usage",
                value=cpu_percent,
                tags={"type": "system"},
                unit="percent"
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.append(MetricPoint(
                timestamp=now,
                name="memory_usage",
                value=memory.percent,
                tags={"type": "system"},
                unit="percent"
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(MetricPoint(
                timestamp=now,
                name="disk_usage",
                value=disk_percent,
                tags={"type": "system"},
                unit="percent"
            ))
            
        except ImportError:
            # Fallback metrics
            metrics.append(MetricPoint(
                timestamp=now,
                name="system_health",
                value=1.0,
                tags={"type": "system"},
                unit="boolean"
            ))
        
        return metrics
    
    def _collect_privacy_metrics(self) -> List[MetricPoint]:
        """Collect privacy-specific metrics."""
        metrics = []
        now = datetime.now()
        
        # These would be populated by actual privacy computations
        metrics.append(MetricPoint(
            timestamp=now,
            name="privacy_epsilon_rate",
            value=0.1,  # Example value
            tags={"type": "privacy"},
            unit="epsilon/hour"
        ))
        
        return metrics
    
    def _collect_performance_metrics(self) -> List[MetricPoint]:
        """Collect performance metrics."""
        metrics = []
        now = datetime.now()
        
        # These would be populated by actual performance measurements
        metrics.append(MetricPoint(
            timestamp=now,
            name="request_count",
            value=10,  # Example value
            tags={"type": "performance"},
            unit="count"
        ))
        
        return metrics


class PrivacyBudgetTracker:
    """Tracks privacy budget consumption."""
    
    def __init__(self, initial_budget: float = 10.0):
        self.total_budget = initial_budget
        self.consumed_budget = 0.0
        self.consumption_history = []
        self.is_tracking = False
        
    def start(self):
        """Start privacy budget tracking."""
        self.is_tracking = True
        logger.info("Privacy budget tracking started")
    
    def stop(self):
        """Stop privacy budget tracking."""
        self.is_tracking = False
        logger.info("Privacy budget tracking stopped")
    
    def record_consumption(
        self,
        epsilon: float,
        delta: float = 0.0,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record privacy budget consumption."""
        if not self.is_tracking:
            return
        
        consumption_record = {
            "timestamp": datetime.now(),
            "epsilon": epsilon,
            "delta": delta,
            "context": context or {},
            "cumulative_epsilon": self.consumed_budget + epsilon
        }
        
        self.consumed_budget += epsilon
        self.consumption_history.append(consumption_record)
        
        # Keep only recent history (last 1000 records)
        if len(self.consumption_history) > 1000:
            self.consumption_history = self.consumption_history[-1000:]
        
        logger.debug(f"Recorded privacy consumption: ε={epsilon}, total={self.consumed_budget}")
    
    def get_status(self) -> PrivacyBudgetStatus:
        """Get current privacy budget status."""
        remaining = max(0, self.total_budget - self.consumed_budget)
        utilization = (self.consumed_budget / self.total_budget) * 100
        
        # Calculate consumption rate (per hour)
        consumption_rate = self._calculate_consumption_rate()
        
        # Estimate depletion time
        estimated_depletion = None
        if consumption_rate > 0 and remaining > 0:
            hours_remaining = remaining / consumption_rate
            estimated_depletion = datetime.now() + timedelta(hours=hours_remaining)
        
        # Determine risk level
        if utilization >= 95:
            risk_level = "high"
        elif utilization >= 80:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return PrivacyBudgetStatus(
            total_budget=self.total_budget,
            consumed_budget=self.consumed_budget,
            remaining_budget=remaining,
            consumption_rate=consumption_rate,
            estimated_depletion=estimated_depletion,
            risk_level=risk_level
        )
    
    def _calculate_consumption_rate(self) -> float:
        """Calculate privacy budget consumption rate per hour."""
        if len(self.consumption_history) < 2:
            return 0.0
        
        # Look at last hour of consumption
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_consumption = [
            record for record in self.consumption_history
            if record["timestamp"] >= one_hour_ago
        ]
        
        if not recent_consumption:
            return 0.0
        
        total_epsilon = sum(record["epsilon"] for record in recent_consumption)
        return total_epsilon  # Already per hour


class PerformanceMonitor:
    """Monitors system performance and SLA compliance."""
    
    def __init__(self):
        self.events = []
        self.sla_targets = {
            "response_time_p95": 2.0,  # 95th percentile < 2 seconds
            "error_rate": 0.01,        # Error rate < 1%
            "availability": 0.999      # 99.9% availability
        }
        self.is_monitoring = False
    
    def start(self):
        """Start performance monitoring."""
        self.is_monitoring = True
        logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        logger.info("Performance monitoring stopped")
    
    def record_event(self, event: Dict[str, Any]):
        """Record performance event."""
        if not self.is_monitoring:
            return
        
        self.events.append(event)
        
        # Keep only recent events (last 10000)
        if len(self.events) > 10000:
            self.events = self.events[-10000:]
    
    def get_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get performance summary."""
        if time_window:
            cutoff = datetime.now() - time_window
            relevant_events = [
                e for e in self.events
                if e["timestamp"] >= cutoff
            ]
        else:
            relevant_events = self.events
        
        if not relevant_events:
            return {"status": "no_data"}
        
        # Calculate metrics
        durations = [e["duration"] for e in relevant_events]
        success_count = sum(1 for e in relevant_events if e["success"])
        total_count = len(relevant_events)
        
        summary = {
            "total_operations": total_count,
            "success_rate": success_count / total_count if total_count > 0 else 0,
            "error_rate": (total_count - success_count) / total_count if total_count > 0 else 0,
            "avg_duration": np.mean(durations) if durations else 0,
            "p50_duration": np.percentile(durations, 50) if durations else 0,
            "p95_duration": np.percentile(durations, 95) if durations else 0,
            "p99_duration": np.percentile(durations, 99) if durations else 0,
        }
        
        # SLA compliance
        summary["sla_compliance"] = {
            "response_time_p95": summary["p95_duration"] <= self.sla_targets["response_time_p95"],
            "error_rate": summary["error_rate"] <= self.sla_targets["error_rate"],
            "availability": summary["success_rate"] >= self.sla_targets["availability"]
        }
        
        return summary


class SecurityMonitor:
    """Monitors security events and threats."""
    
    def __init__(self):
        self.events = []
        self.threat_patterns = {
            "brute_force": {"threshold": 10, "window_minutes": 5},
            "unusual_access": {"threshold": 5, "window_minutes": 1},
            "data_exfiltration": {"threshold": 1, "window_minutes": 1}
        }
        self.is_monitoring = False
    
    def start(self):
        """Start security monitoring."""
        self.is_monitoring = True
        logger.info("Security monitoring started")
    
    def stop(self):
        """Stop security monitoring."""
        self.is_monitoring = False
        logger.info("Security monitoring stopped")
    
    def record_event(self, event: Dict[str, Any]):
        """Record security event."""
        if not self.is_monitoring:
            return
        
        self.events.append(event)
        
        # Check for threat patterns
        self._check_threat_patterns(event)
        
        # Keep only recent events (last 10000)
        if len(self.events) > 10000:
            self.events = self.events[-10000:]
    
    def get_events(
        self,
        severity: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """Get security events."""
        events = self.events
        
        if time_window:
            cutoff = datetime.now() - time_window
            events = [e for e in events if e["timestamp"] >= cutoff]
        
        if severity:
            events = [e for e in events if e["severity"] == severity]
        
        return events
    
    def _check_threat_patterns(self, event: Dict[str, Any]):
        """Check for threat patterns."""
        event_type = event.get("event_type", "")
        
        for pattern_name, pattern_config in self.threat_patterns.items():
            if pattern_name in event_type.lower():
                # Count recent events of this type
                window = timedelta(minutes=pattern_config["window_minutes"])
                cutoff = datetime.now() - window
                
                recent_events = [
                    e for e in self.events
                    if (e["timestamp"] >= cutoff and 
                        pattern_name in e.get("event_type", "").lower())
                ]
                
                if len(recent_events) >= pattern_config["threshold"]:
                    logger.warning(f"Threat pattern detected: {pattern_name}")
                    # This would trigger additional security measures


class AnomalyDetector:
    """Detects anomalies in metrics and system behavior."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.metric_baselines = {}
        self.anomalies = []
        self.is_detecting = False
    
    def start(self):
        """Start anomaly detection."""
        self.is_detecting = True
        logger.info("Anomaly detection started")
    
    def stop(self):
        """Stop anomaly detection."""
        self.is_detecting = False
        logger.info("Anomaly detection stopped")
    
    def update(self, metric: MetricPoint):
        """Update anomaly detection with new metric."""
        if not self.is_detecting:
            return
        
        metric_name = metric.name
        
        # Initialize baseline if needed
        if metric_name not in self.metric_baselines:
            self.metric_baselines[metric_name] = {
                "values": [],
                "mean": 0.0,
                "std": 0.0,
                "count": 0
            }
        
        baseline = self.metric_baselines[metric_name]
        baseline["values"].append(metric.value)
        baseline["count"] += 1
        
        # Keep only recent values (last 1000)
        if len(baseline["values"]) > 1000:
            baseline["values"] = baseline["values"][-1000:]
        
        # Update statistics (need at least 10 values)
        if len(baseline["values"]) >= 10:
            baseline["mean"] = np.mean(baseline["values"])
            baseline["std"] = np.std(baseline["values"])
            
            # Check for anomaly
            if baseline["std"] > 0:
                z_score = abs(metric.value - baseline["mean"]) / baseline["std"]
                
                if z_score > self.sensitivity:
                    anomaly = {
                        "timestamp": metric.timestamp,
                        "metric_name": metric_name,
                        "value": metric.value,
                        "expected_range": (
                            baseline["mean"] - self.sensitivity * baseline["std"],
                            baseline["mean"] + self.sensitivity * baseline["std"]
                        ),
                        "z_score": z_score,
                        "tags": metric.tags
                    }
                    
                    self.anomalies.append(anomaly)
                    logger.warning(f"Anomaly detected in {metric_name}: {metric.value} (z-score: {z_score:.2f})")
        
        # Keep only recent anomalies
        if len(self.anomalies) > 1000:
            self.anomalies = self.anomalies[-1000:]
    
    def get_anomalies(
        self,
        metric_name: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """Get detected anomalies."""
        anomalies = self.anomalies
        
        if time_window:
            cutoff = datetime.now() - time_window
            anomalies = [a for a in anomalies if a["timestamp"] >= cutoff]
        
        if metric_name:
            anomalies = [a for a in anomalies if a["metric_name"] == metric_name]
        
        return anomalies


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, alert_handlers: List[Callable]):
        self.alert_handlers = alert_handlers
        self.alerts = {}
        self.alert_history = []
    
    def add_alert(self, alert: Alert):
        """Add alert configuration."""
        self.alerts[alert.name] = alert
        logger.info(f"Added alert: {alert.name}")
    
    def remove_alert(self, alert_name: str):
        """Remove alert configuration."""
        if alert_name in self.alerts:
            del self.alerts[alert_name]
            logger.info(f"Removed alert: {alert_name}")
    
    def check_metric_alerts(self, metric: MetricPoint):
        """Check metric against all applicable alerts."""
        for alert_name, alert in self.alerts.items():
            # Simple matching - in practice would be more sophisticated
            if alert_name.startswith(metric.name.split("_")[0]):
                if alert.should_trigger(metric.value):
                    self.trigger_alert(
                        alert_name,
                        f"{alert.description}: {metric.value}",
                        alert.severity
                    )
                    alert.last_triggered = datetime.now()
    
    def trigger_alert(self, alert_name: str, message: str, severity: str):
        """Trigger an alert."""
        alert_event = {
            "timestamp": datetime.now(),
            "alert_name": alert_name,
            "message": message,
            "severity": severity
        }
        
        self.alert_history.append(alert_event)
        
        # Execute alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert_event)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"ALERT [{severity.upper()}] {alert_name}: {message}")


class MetricsStorage:
    """Storage backend for metrics and monitoring data."""
    
    def __init__(self, backend: str = "in_memory"):
        self.backend = backend
        self.metrics = []
        self.dashboards = {}
        
        if backend == "file":
            self.storage_dir = Path("monitoring_data")
            self.storage_dir.mkdir(exist_ok=True)
    
    def store_metric(self, metric: MetricPoint):
        """Store metric point."""
        if self.backend == "in_memory":
            self.metrics.append(metric)
            
            # Keep only recent metrics (last 100000)
            if len(self.metrics) > 100000:
                self.metrics = self.metrics[-100000:]
        
        elif self.backend == "file":
            # Append to daily file
            date_str = metric.timestamp.strftime("%Y-%m-%d")
            metrics_file = self.storage_dir / f"metrics_{date_str}.jsonl"
            
            with open(metrics_file, "a") as f:
                f.write(json.dumps(metric.to_dict()) + "\n")
    
    def store_dashboard(self, dashboard_id: str, config: Dict[str, Any]):
        """Store dashboard configuration."""
        self.dashboards[dashboard_id] = {
            "config": config,
            "created_at": datetime.now()
        }
    
    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard data."""
        if dashboard_id not in self.dashboards:
            return {"error": "Dashboard not found"}
        
        dashboard = self.dashboards[dashboard_id]
        config = dashboard["config"]
        
        # Generate dashboard data based on configuration
        # This is a simplified implementation
        
        dashboard_data = {
            "id": dashboard_id,
            "config": config,
            "data": {
                "metrics": self._get_dashboard_metrics(config),
                "charts": self._generate_dashboard_charts(config),
                "alerts": self._get_dashboard_alerts(config),
                "last_updated": datetime.now()
            }
        }
        
        return dashboard_data
    
    def export_metrics(
        self,
        format: str = "json",
        time_range: Optional[Tuple[datetime, datetime]] = None,
        metric_names: Optional[List[str]] = None
    ) -> str:
        """Export metrics in specified format."""
        # Filter metrics
        filtered_metrics = self.metrics
        
        if time_range:
            start_time, end_time = time_range
            filtered_metrics = [
                m for m in filtered_metrics
                if start_time <= m.timestamp <= end_time
            ]
        
        if metric_names:
            filtered_metrics = [
                m for m in filtered_metrics
                if m.name in metric_names
            ]
        
        # Export in requested format
        if format == "json":
            return json.dumps([m.to_dict() for m in filtered_metrics], indent=2)
        
        elif format == "csv":
            if not filtered_metrics:
                return "timestamp,name,value,tags,unit\n"
            
            csv_lines = ["timestamp,name,value,tags,unit"]
            for metric in filtered_metrics:
                tags_str = ";".join(f"{k}:{v}" for k, v in metric.tags.items())
                csv_lines.append(
                    f"{metric.timestamp.isoformat()},{metric.name},{metric.value},{tags_str},{metric.unit}"
                )
            
            return "\n".join(csv_lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _get_dashboard_metrics(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get metrics for dashboard."""
        # Simplified implementation
        return [m.to_dict() for m in self.metrics[-100:]]  # Last 100 metrics
    
    def _generate_dashboard_charts(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate charts for dashboard."""
        # Simplified implementation
        charts = [
            {
                "type": "line",
                "title": "System Metrics",
                "data": [m.to_dict() for m in self.metrics[-50:]]
            },
            {
                "type": "gauge",
                "title": "Current Status",
                "value": 0.75,
                "max": 1.0
            }
        ]
        
        return charts
    
    def _get_dashboard_alerts(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get alerts for dashboard."""
        # Simplified implementation
        return [
            {
                "name": "System Health",
                "status": "OK",
                "last_checked": datetime.now().isoformat()
            }
        ]