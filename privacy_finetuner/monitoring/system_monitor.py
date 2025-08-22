"""System monitoring for privacy-preserving ML framework."""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io_bytes: int
    gpu_utilization: float
    timestamp: float


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self):
        """Initialize system monitor."""
        self.metrics_history = []
        self.monitoring_enabled = False
        self.alert_thresholds = {
            "cpu": 90.0,
            "memory": 85.0,
            "disk": 90.0
        }
        logger.info("SystemMonitor initialized")
    
    def start_monitoring(self) -> None:
        """Start system monitoring."""
        self.monitoring_enabled = True
        logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.monitoring_enabled = False
        logger.info("System monitoring stopped")
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # Mock metrics for now - in production would use psutil
        metrics = SystemMetrics(
            cpu_percent=45.0,
            memory_percent=60.0,
            disk_usage_percent=70.0,
            network_io_bytes=1024000,
            gpu_utilization=30.0,
            timestamp=time.time()
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        latest = self.metrics_history[-1]
        
        return {
            "status": "operational",
            "cpu_percent": latest.cpu_percent,
            "memory_percent": latest.memory_percent,
            "disk_usage_percent": latest.disk_usage_percent,
            "gpu_utilization": latest.gpu_utilization,
            "monitoring_enabled": self.monitoring_enabled,
            "last_update": latest.timestamp
        }
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for system alerts."""
        if not self.metrics_history:
            return []
        
        latest = self.metrics_history[-1]
        alerts = []
        
        if latest.cpu_percent > self.alert_thresholds["cpu"]:
            alerts.append({
                "type": "cpu_high",
                "value": latest.cpu_percent,
                "threshold": self.alert_thresholds["cpu"]
            })
        
        if latest.memory_percent > self.alert_thresholds["memory"]:
            alerts.append({
                "type": "memory_high",
                "value": latest.memory_percent,
                "threshold": self.alert_thresholds["memory"]
            })
        
        if latest.disk_usage_percent > self.alert_thresholds["disk"]:
            alerts.append({
                "type": "disk_high",
                "value": latest.disk_usage_percent,
                "threshold": self.alert_thresholds["disk"]
            })
        
        return alerts