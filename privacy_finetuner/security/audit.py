"""Advanced security auditing and compliance monitoring system.

This module provides comprehensive security auditing, threat detection,
and compliance monitoring for privacy-preserving training operations.
"""

import os
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events to monitor."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    MODEL_ACCESS = "model_access"
    PRIVACY_VIOLATION = "privacy_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_INTRUSION = "system_intrusion"
    COMPLIANCE_VIOLATION = "compliance_violation"


class ThreatLevel(Enum):
    """Threat severity levels."""
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


@dataclass
class SecurityEvent:
    """Individual security event record."""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    mitigation_applied: bool = False
    investigation_required: bool = False


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    CCPA = "ccpa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    EU_AI_ACT = "eu_ai_act"


@dataclass
class ComplianceRule:
    """Individual compliance rule definition."""
    rule_id: str
    framework: ComplianceFramework
    description: str
    check_function: Callable[[Dict[str, Any]], bool]
    remediation_steps: List[str]
    severity: ThreatLevel = ThreatLevel.MEDIUM


class SecurityAuditor:
    """Advanced security auditing and monitoring system."""
    
    def __init__(
        self,
        audit_log_path: str = "./logs/security_audit.log",
        event_retention_days: int = 90,
        alert_threshold: ThreatLevel = ThreatLevel.HIGH,
        enable_real_time_monitoring: bool = True
    ):
        """Initialize security auditor.
        
        Args:
            audit_log_path: Path for security audit logs
            event_retention_days: Days to retain security events
            alert_threshold: Minimum threat level for alerts
            enable_real_time_monitoring: Enable continuous monitoring
        """
        self.audit_log_path = Path(audit_log_path)
        self.event_retention_days = event_retention_days
        self.alert_threshold = alert_threshold
        self.enable_real_time_monitoring = enable_real_time_monitoring
        
        # Event storage and tracking
        self.security_events: List[SecurityEvent] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns: Dict[str, int] = {}
        
        # Compliance rules
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self._initialize_compliance_rules()
        
        # Monitoring thread
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[SecurityEvent], None]] = []
        
        # Ensure audit log directory exists
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Start real-time monitoring if enabled
        if self.enable_real_time_monitoring:
            self.start_monitoring()
    
    def _initialize_compliance_rules(self) -> None:
        """Initialize built-in compliance rules."""
        
        # GDPR Rules
        self.add_compliance_rule(ComplianceRule(
            rule_id="gdpr_001",
            framework=ComplianceFramework.GDPR,
            description="Verify data subject consent for processing",
            check_function=self._check_gdpr_consent,
            remediation_steps=["Obtain explicit consent", "Document consent basis", "Provide opt-out mechanism"],
            severity=ThreatLevel.HIGH
        ))
        
        self.add_compliance_rule(ComplianceRule(
            rule_id="gdpr_002",
            framework=ComplianceFramework.GDPR,
            description="Ensure data minimization principles",
            check_function=self._check_data_minimization,
            remediation_steps=["Reduce data collection scope", "Implement data retention policies", "Regular data purging"],
            severity=ThreatLevel.MEDIUM
        ))
        
        # HIPAA Rules
        self.add_compliance_rule(ComplianceRule(
            rule_id="hipaa_001",
            framework=ComplianceFramework.HIPAA,
            description="Verify PHI protection measures",
            check_function=self._check_phi_protection,
            remediation_steps=["Enable encryption", "Access controls", "Audit trails"],
            severity=ThreatLevel.CRITICAL
        ))
        
        # Privacy Rules
        self.add_compliance_rule(ComplianceRule(
            rule_id="privacy_001",
            framework=ComplianceFramework.EU_AI_ACT,
            description="Verify differential privacy parameters",
            check_function=self._check_privacy_parameters,
            remediation_steps=["Adjust epsilon/delta values", "Increase noise levels", "Implement privacy accounting"],
            severity=ThreatLevel.HIGH
        ))
    
    def log_security_event(
        self,
        event_type: SecurityEventType,
        threat_level: ThreatLevel,
        details: Dict[str, Any],
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None
    ) -> str:
        """Log a security event and trigger appropriate responses.
        
        Args:
            event_type: Type of security event
            threat_level: Severity level
            details: Event details
            source_ip: Source IP address if applicable
            user_id: User identifier if applicable
            resource: Resource being accessed
            action: Action being performed
            
        Returns:
            Event ID for tracking
        """
        # Generate unique event ID
        event_id = self._generate_event_id(event_type, details)
        
        # Create security event
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            threat_level=threat_level,
            timestamp=datetime.now(),
            source_ip=source_ip,
            user_id=user_id,
            resource=resource,
            action=action,
            details=details
        )
        
        # Store event
        self.security_events.append(event)
        
        # Write to audit log
        self._write_audit_log(event)
        
        # Check for patterns and anomalies
        self._analyze_event_patterns(event)
        
        # Apply automatic mitigations if needed
        if threat_level.value >= ThreatLevel.HIGH.value:
            self._apply_automatic_mitigation(event)
        
        # Trigger alerts
        if threat_level.value >= self.alert_threshold.value:
            self._trigger_alert(event)
        
        logger.info(f"Security event logged: {event_id} ({event_type.value}, {threat_level.name})")
        
        return event_id
    
    def _generate_event_id(self, event_type: SecurityEventType, details: Dict[str, Any]) -> str:
        """Generate unique event ID."""
        timestamp = datetime.now().isoformat()
        content = f"{event_type.value}_{timestamp}_{json.dumps(details, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _write_audit_log(self, event: SecurityEvent) -> None:
        """Write event to audit log."""
        try:
            audit_entry = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.name,
                "source_ip": event.source_ip,
                "user_id": event.user_id,
                "resource": event.resource,
                "action": event.action,
                "details": event.details,
                "mitigation_applied": event.mitigation_applied,
                "investigation_required": event.investigation_required
            }
            
            with open(self.audit_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry) + "\\n")
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def _analyze_event_patterns(self, event: SecurityEvent) -> None:
        """Analyze event patterns for anomaly detection."""
        
        # Track IP-based patterns
        if event.source_ip:
            ip_key = f"ip_{event.source_ip}"
            self.suspicious_patterns[ip_key] = self.suspicious_patterns.get(ip_key, 0) + 1
            
            # Flag suspicious IPs
            if self.suspicious_patterns[ip_key] > 10:  # Threshold
                self.log_security_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    ThreatLevel.MEDIUM,
                    {"pattern": "high_frequency_requests", "ip": event.source_ip, "count": self.suspicious_patterns[ip_key]}
                )
        
        # Track user-based patterns
        if event.user_id:
            user_key = f"user_{event.user_id}"
            self.suspicious_patterns[user_key] = self.suspicious_patterns.get(user_key, 0) + 1
            
        # Track failed authentication attempts
        if event.event_type == SecurityEventType.AUTHENTICATION and event.details.get("success", False) is False:
            auth_key = f"failed_auth_{event.source_ip}"
            self.suspicious_patterns[auth_key] = self.suspicious_patterns.get(auth_key, 0) + 1
            
            if self.suspicious_patterns[auth_key] > 5:  # Failed auth threshold
                self.blocked_ips.add(event.source_ip)
                self.log_security_event(
                    SecurityEventType.SYSTEM_INTRUSION,
                    ThreatLevel.HIGH,
                    {"action": "ip_blocked", "ip": event.source_ip, "reason": "multiple_failed_auth"}
                )
    
    def _apply_automatic_mitigation(self, event: SecurityEvent) -> None:
        """Apply automatic security mitigations."""
        
        # Block suspicious IPs
        if event.threat_level == ThreatLevel.CRITICAL and event.source_ip:
            self.blocked_ips.add(event.source_ip)
            event.mitigation_applied = True
            logger.warning(f"Automatically blocked IP: {event.source_ip}")
        
        # Rate limiting for suspicious activity
        if event.event_type == SecurityEventType.SUSPICIOUS_ACTIVITY:
            # Implement rate limiting logic here
            event.mitigation_applied = True
        
        # Mark for investigation
        if event.threat_level.value >= ThreatLevel.HIGH.value:
            event.investigation_required = True
    
    def _trigger_alert(self, event: SecurityEvent) -> None:
        """Trigger security alerts."""
        
        # Call registered alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Log critical events to system log
        if event.threat_level == ThreatLevel.CRITICAL:
            logger.critical(f"SECURITY ALERT: {event.event_type.value} - {event.details}")
    
    def add_compliance_rule(self, rule: ComplianceRule) -> None:
        """Add a compliance rule."""
        self.compliance_rules[rule.rule_id] = rule
        logger.info(f"Added compliance rule: {rule.rule_id} ({rule.framework.value})")
    
    def run_compliance_check(self, framework: Optional[ComplianceFramework] = None) -> Dict[str, Any]:
        """Run compliance checks against current system state.
        
        Args:
            framework: Specific framework to check (all if None)
            
        Returns:
            Compliance check results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "framework_filter": framework.value if framework else "all",
            "rules_checked": 0,
            "rules_passed": 0,
            "rules_failed": 0,
            "violations": [],
            "recommendations": []
        }
        
        # Current system context for compliance checks
        system_context = {
            "blocked_ips": len(self.blocked_ips),
            "active_sessions": len(self.active_sessions),
            "recent_events": len([e for e in self.security_events if e.timestamp > datetime.now() - timedelta(hours=24)]),
            "high_threat_events": len([e for e in self.security_events if e.threat_level.value >= ThreatLevel.HIGH.value])
        }
        
        # Check applicable rules
        for rule in self.compliance_rules.values():
            if framework is None or rule.framework == framework:
                results["rules_checked"] += 1
                
                try:
                    if rule.check_function(system_context):
                        results["rules_passed"] += 1
                    else:
                        results["rules_failed"] += 1
                        results["violations"].append({
                            "rule_id": rule.rule_id,
                            "framework": rule.framework.value,
                            "description": rule.description,
                            "severity": rule.severity.name,
                            "remediation_steps": rule.remediation_steps
                        })
                        
                        # Add recommendations
                        results["recommendations"].extend(rule.remediation_steps)
                        
                except Exception as e:
                    logger.error(f"Compliance check failed for rule {rule.rule_id}: {e}")
                    results["rules_failed"] += 1
        
        # Log compliance check result
        self.log_security_event(
            SecurityEventType.COMPLIANCE_VIOLATION if results["rules_failed"] > 0 else SecurityEventType.AUTHORIZATION,
            ThreatLevel.HIGH if results["rules_failed"] > 0 else ThreatLevel.INFO,
            {"compliance_results": results}
        )
        
        return results
    
    def start_monitoring(self) -> None:
        """Start real-time security monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Security monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time security monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        logger.info("Security monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Check for system anomalies
                self._check_system_health()
                
                # Clean up old events
                self._cleanup_old_events()
                
                # Reset suspicious pattern counters periodically
                self._reset_pattern_counters()
                
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
    
    def _check_system_health(self) -> None:
        """Check overall system security health."""
        
        # Check for excessive blocked IPs
        if len(self.blocked_ips) > 100:
            self.log_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                ThreatLevel.MEDIUM,
                {"alert": "high_blocked_ip_count", "count": len(self.blocked_ips)}
            )
        
        # Check for excessive high-threat events
        recent_high_threats = [
            e for e in self.security_events
            if e.timestamp > datetime.now() - timedelta(hours=1) and e.threat_level.value >= ThreatLevel.HIGH.value
        ]
        
        if len(recent_high_threats) > 10:
            self.log_security_event(
                SecurityEventType.SUSPICIOUS_ACTIVITY,
                ThreatLevel.HIGH,
                {"alert": "high_threat_event_spike", "count": len(recent_high_threats)}
            )
    
    def _cleanup_old_events(self) -> None:
        """Clean up old security events."""
        cutoff_date = datetime.now() - timedelta(days=self.event_retention_days)
        self.security_events = [e for e in self.security_events if e.timestamp >= cutoff_date]
    
    def _reset_pattern_counters(self) -> None:
        """Reset suspicious pattern counters periodically."""
        # Reset counters every hour to prevent permanent blocking
        self.suspicious_patterns.clear()
    
    def add_alert_callback(self, callback: Callable[[SecurityEvent], None]) -> None:
        """Add callback for security alerts."""
        self._alert_callbacks.append(callback)
    
    def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get data for security monitoring dashboard."""
        now = datetime.now()
        
        # Recent events by type
        recent_events = [e for e in self.security_events if e.timestamp > now - timedelta(hours=24)]
        events_by_type = {}
        for event in recent_events:
            events_by_type[event.event_type.value] = events_by_type.get(event.event_type.value, 0) + 1
        
        # Threat level distribution
        threats_by_level = {}
        for event in recent_events:
            threats_by_level[event.threat_level.name] = threats_by_level.get(event.threat_level.name, 0) + 1
        
        return {
            "timestamp": now.isoformat(),
            "summary": {
                "total_events_24h": len(recent_events),
                "blocked_ips": len(self.blocked_ips),
                "active_sessions": len(self.active_sessions),
                "investigations_required": len([e for e in recent_events if e.investigation_required])
            },
            "events_by_type": events_by_type,
            "threats_by_level": threats_by_level,
            "top_blocked_ips": list(self.blocked_ips)[:10],
            "recent_high_threats": [
                {
                    "event_id": e.event_id,
                    "type": e.event_type.value,
                    "level": e.threat_level.name,
                    "timestamp": e.timestamp.isoformat(),
                    "source_ip": e.source_ip
                }
                for e in recent_events
                if e.threat_level.value >= ThreatLevel.HIGH.value
            ][:10]
        }
    
    # Compliance check functions
    def _check_gdpr_consent(self, context: Dict[str, Any]) -> bool:
        """Check GDPR consent compliance."""
        # Implement specific GDPR consent checks
        return context.get("consent_documented", False)
    
    def _check_data_minimization(self, context: Dict[str, Any]) -> bool:
        """Check data minimization compliance."""
        # Implement data minimization checks
        return context.get("data_retention_policy", False)
    
    def _check_phi_protection(self, context: Dict[str, Any]) -> bool:
        """Check PHI protection compliance."""
        # Implement HIPAA PHI protection checks
        return context.get("encryption_enabled", False) and context.get("access_controls", False)
    
    def _check_privacy_parameters(self, context: Dict[str, Any]) -> bool:
        """Check differential privacy parameter compliance."""
        # Implement privacy parameter validation
        epsilon = context.get("epsilon", float('inf'))
        delta = context.get("delta", 1.0)
        
        # Basic privacy parameter validation
        return epsilon <= 10.0 and delta <= 1e-3


# Global security auditor instance
_global_auditor: Optional[SecurityAuditor] = None


def get_security_auditor() -> SecurityAuditor:
    """Get or create global security auditor."""
    global _global_auditor
    
    if _global_auditor is None:
        _global_auditor = SecurityAuditor()
    
    return _global_auditor


def log_security_event(
    event_type: SecurityEventType,
    threat_level: ThreatLevel,
    details: Dict[str, Any],
    **kwargs
) -> str:
    """Convenience function to log security events."""
    auditor = get_security_auditor()
    return auditor.log_security_event(event_type, threat_level, details, **kwargs)


def run_compliance_check(framework: Optional[ComplianceFramework] = None) -> Dict[str, Any]:
    """Convenience function to run compliance checks."""
    auditor = get_security_auditor()
    return auditor.run_compliance_check(framework)