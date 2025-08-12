"""Simple fallback resource manager without external dependencies."""

import os
import time
import logging
import threading
import warnings
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ResourceType(Enum):
    """Types of system resources to monitor."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ResourceUsageMetrics:
    """Resource usage metrics."""
    resource_type: ResourceType
    usage_percent: float
    used_amount: float
    available_amount: float
    total_amount: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ResourceManager:
    """Simple fallback resource manager."""
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._monitoring_enabled = False
        self._lock = threading.Lock()
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self._monitoring_enabled = True
        self._logger.info("Resource monitoring started (fallback mode)")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring_enabled = False
        self._logger.info("Resource monitoring stopped")
    
    def get_current_usage(self) -> Dict[ResourceType, ResourceUsageMetrics]:
        """Get current resource usage with fallback values."""
        return {
            ResourceType.CPU: ResourceUsageMetrics(
                resource_type=ResourceType.CPU,
                usage_percent=50.0,
                used_amount=2.0,
                available_amount=2.0,
                total_amount=4.0,
                metadata={'fallback': True}
            ),
            ResourceType.MEMORY: ResourceUsageMetrics(
                resource_type=ResourceType.MEMORY,
                usage_percent=60.0,
                used_amount=4.8,
                available_amount=3.2,
                total_amount=8.0,
                metadata={'fallback': True}
            )
        }
    
    def check_resource_availability(self, resource_type: ResourceType, required_amount: float) -> bool:
        """Check if resources are available."""
        current_usage = self.get_current_usage()
        if resource_type in current_usage:
            available = current_usage[resource_type].available_amount
            return available >= required_amount
        return True
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        import gc
        gc.collect()
        return {"status": "completed", "method": "garbage_collection"}


# Create global instance
resource_manager = ResourceManager()