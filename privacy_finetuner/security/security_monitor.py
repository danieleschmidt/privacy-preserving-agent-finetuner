"""Security monitoring system for privacy-preserving ML framework."""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    source: str
    details: Dict[str, Any]


class SecurityMonitor:
    """Real-time security monitoring system."""
    
    def __init__(self):
        """Initialize security monitor."""
        self.events = []
        self.threat_detectors = {}
        self.active_threats = []
        self.monitoring_enabled = False
        logger.info("SecurityMonitor initialized")
    
    def start_monitoring(self) -> None:
        """Start security monitoring."""
        self.monitoring_enabled = True
        logger.info("Security monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop security monitoring."""
        self.monitoring_enabled = False
        logger.info("Security monitoring stopped")
    
    def record_event(self, event_type: str, details: Dict[str, Any], threat_level: ThreatLevel = ThreatLevel.LOW) -> None:
        """Record a security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            source="system",
            details=details
        )
        self.events.append(event)
        
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.active_threats.append(event)
            logger.warning(f"Security threat detected: {event_type} - {threat_level.value}")
    
    def get_threat_status(self) -> Dict[str, Any]:
        """Get current threat status."""
        return {
            "active_threats": len(self.active_threats),
            "total_events": len(self.events),
            "monitoring_enabled": self.monitoring_enabled,
            "last_event": self.events[-1].timestamp if self.events else None
        }
    
    def clear_resolved_threats(self) -> None:
        """Clear resolved threats."""
        self.active_threats.clear()
        logger.info("Resolved threats cleared")