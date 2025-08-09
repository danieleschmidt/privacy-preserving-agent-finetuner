"""Advanced threat detection system for privacy-preserving ML training.

This module implements real-time security monitoring, anomaly detection,
and automated threat response for protecting privacy-sensitive training workflows.
"""

import time
import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
import queue

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    PRIVACY_BUDGET_EXHAUSTION = "privacy_budget_exhaustion"
    MODEL_INVERSION_ATTACK = "model_inversion_attack"
    MEMBERSHIP_INFERENCE_ATTACK = "membership_inference_attack"
    DATA_POISONING = "data_poisoning"
    GRADIENT_LEAKAGE = "gradient_leakage"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    ABNORMAL_TRAINING_BEHAVIOR = "abnormal_training_behavior"
    COMPLIANCE_VIOLATION = "compliance_violation"


@dataclass
class SecurityAlert:
    """Security alert with threat details."""
    threat_id: str
    threat_type: ThreatType
    threat_level: ThreatLevel
    description: str
    detection_time: str
    affected_components: List[str]
    recommended_actions: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def is_critical(self) -> bool:
        return self.threat_level == ThreatLevel.CRITICAL


class ThreatDetector:
    """Advanced threat detection system with real-time monitoring."""
    
    def __init__(
        self,
        alert_threshold: float = 0.7,
        monitoring_interval: float = 1.0,
        max_alert_queue_size: int = 1000,
        enable_automated_response: bool = True
    ):
        """Initialize threat detection system.
        
        Args:
            alert_threshold: Threshold for triggering alerts (0-1)
            monitoring_interval: Seconds between monitoring checks
            max_alert_queue_size: Maximum alerts to queue
            enable_automated_response: Enable automatic threat response
        """
        self.alert_threshold = alert_threshold
        self.monitoring_interval = monitoring_interval
        self.max_alert_queue_size = max_alert_queue_size
        self.enable_automated_response = enable_automated_response
        
        # Alert management
        self.alert_queue = queue.Queue(maxsize=max_alert_queue_size)
        self.active_alerts = {}
        self.alert_handlers = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Threat detection state
        self.baseline_metrics = {}
        self.anomaly_detectors = {}
        self.threat_patterns = {}
        
        # Security metrics tracking
        self.security_metrics = {
            "total_threats_detected": 0,
            "threats_by_type": {},
            "threats_by_level": {},
            "response_times": [],
            "false_positive_rate": 0.0
        }
        
        self._initialize_threat_patterns()
        logger.info("ThreatDetector initialized with real-time monitoring")
    
    def _initialize_threat_patterns(self) -> None:
        """Initialize known threat detection patterns."""
        self.threat_patterns = {
            ThreatType.PRIVACY_BUDGET_EXHAUSTION: {
                "indicators": ["rapid_epsilon_consumption", "budget_near_limit"],
                "threshold": 0.8
            },
            ThreatType.MODEL_INVERSION_ATTACK: {
                "indicators": ["gradient_correlation_high", "reconstruction_quality_high"],
                "threshold": 0.7
            },
            ThreatType.MEMBERSHIP_INFERENCE_ATTACK: {
                "indicators": ["output_confidence_variance", "loss_distribution_anomaly"],
                "threshold": 0.6
            },
            ThreatType.DATA_POISONING: {
                "indicators": ["loss_spike", "gradient_norm_anomaly", "accuracy_drop"],
                "threshold": 0.75
            },
            ThreatType.GRADIENT_LEAKAGE: {
                "indicators": ["gradient_l2_norm_high", "gradient_cosine_similarity_high"],
                "threshold": 0.8
            },
            ThreatType.UNAUTHORIZED_ACCESS: {
                "indicators": ["invalid_auth_attempts", "unusual_access_patterns"],
                "threshold": 0.9
            },
            ThreatType.ABNORMAL_TRAINING_BEHAVIOR: {
                "indicators": ["convergence_anomaly", "resource_usage_spike"],
                "threshold": 0.65
            }
        }
        
        logger.debug(f"Initialized {len(self.threat_patterns)} threat patterns")
    
    def start_monitoring(self) -> None:
        """Start real-time threat monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Started real-time threat monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop threat monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        logger.info("Stopped threat monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        while self.monitoring_active:
            try:
                # Check for active threats
                self._scan_for_threats()
                
                # Process queued alerts
                self._process_alert_queue()
                
                # Clean up resolved threats
                self._cleanup_resolved_threats()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(self.monitoring_interval * 2)  # Back off on errors
    
    def detect_threat(
        self, 
        training_metrics: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[SecurityAlert]:
        """Detect threats based on training metrics and context.
        
        Args:
            training_metrics: Current training metrics
            context: Additional context information
            
        Returns:
            List of detected security alerts
        """
        alerts = []
        context = context or {}
        
        # Check each threat pattern
        for threat_type, pattern in self.threat_patterns.items():
            threat_score = self._calculate_threat_score(threat_type, training_metrics, context)
            
            if threat_score >= pattern["threshold"]:
                alert = self._create_alert(threat_type, threat_score, training_metrics, context)
                alerts.append(alert)
                
                # Add to queue for processing
                try:
                    self.alert_queue.put_nowait(alert)
                except queue.Full:
                    logger.warning("Alert queue full, dropping oldest alert")
                    try:
                        self.alert_queue.get_nowait()
                        self.alert_queue.put_nowait(alert)
                    except queue.Empty:
                        pass
        
        if alerts:
            self.security_metrics["total_threats_detected"] += len(alerts)
            logger.warning(f"Detected {len(alerts)} security threats")
        
        return alerts
    
    def _calculate_threat_score(
        self,
        threat_type: ThreatType,
        metrics: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Calculate threat score for specific threat type."""
        
        if threat_type == ThreatType.PRIVACY_BUDGET_EXHAUSTION:
            return self._detect_budget_exhaustion(metrics, context)
        
        elif threat_type == ThreatType.MODEL_INVERSION_ATTACK:
            return self._detect_model_inversion(metrics, context)
        
        elif threat_type == ThreatType.MEMBERSHIP_INFERENCE_ATTACK:
            return self._detect_membership_inference(metrics, context)
        
        elif threat_type == ThreatType.DATA_POISONING:
            return self._detect_data_poisoning(metrics, context)
        
        elif threat_type == ThreatType.GRADIENT_LEAKAGE:
            return self._detect_gradient_leakage(metrics, context)
        
        elif threat_type == ThreatType.UNAUTHORIZED_ACCESS:
            return self._detect_unauthorized_access(metrics, context)
        
        elif threat_type == ThreatType.ABNORMAL_TRAINING_BEHAVIOR:
            return self._detect_abnormal_behavior(metrics, context)
        
        else:
            return 0.0
    
    def _detect_budget_exhaustion(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Detect privacy budget exhaustion threats."""
        epsilon_used = metrics.get("privacy_epsilon_used", 0.0)
        epsilon_total = metrics.get("privacy_epsilon_total", 1.0)
        
        if epsilon_total <= 0:
            return 0.0
        
        budget_ratio = epsilon_used / epsilon_total
        
        # High threat if budget > 90% consumed
        if budget_ratio > 0.9:
            return 1.0
        elif budget_ratio > 0.8:
            return 0.8
        elif budget_ratio > 0.7:
            return 0.6
        else:
            return max(0.0, (budget_ratio - 0.5) * 2)
    
    def _detect_model_inversion(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Detect model inversion attack patterns."""
        # Look for high gradient norms that could enable inversion
        gradient_norm = metrics.get("gradient_l2_norm", 0.0)
        loss_variance = metrics.get("loss_variance", 0.0)
        
        # Normalize and combine indicators
        gradient_score = min(1.0, gradient_norm / 10.0)  # Assuming max normal gradient norm ~10
        variance_score = min(1.0, loss_variance / 5.0)   # Assuming max normal variance ~5
        
        return (gradient_score * 0.6 + variance_score * 0.4)
    
    def _detect_membership_inference(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Detect membership inference attack indicators."""
        # Look for suspicious confidence patterns
        output_confidence_mean = metrics.get("output_confidence_mean", 0.5)
        output_confidence_std = metrics.get("output_confidence_std", 0.1)
        
        # High confidence with low variance can indicate memorization
        confidence_score = output_confidence_mean if output_confidence_std < 0.05 else 0.0
        
        # Check for loss distribution anomalies
        loss_distribution_score = metrics.get("loss_distribution_anomaly", 0.0)
        
        return max(confidence_score, loss_distribution_score)
    
    def _detect_data_poisoning(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Detect data poisoning attacks."""
        current_loss = metrics.get("current_loss", 0.0)
        baseline_loss = self.baseline_metrics.get("loss", current_loss)
        
        # Sudden loss spikes can indicate poisoned data
        if baseline_loss > 0:
            loss_ratio = current_loss / baseline_loss
            if loss_ratio > 2.0:  # Loss doubled
                return 1.0
            elif loss_ratio > 1.5:
                return 0.7
            elif loss_ratio > 1.2:
                return 0.4
        
        # Check gradient norm anomalies
        gradient_norm = metrics.get("gradient_l2_norm", 0.0)
        baseline_grad_norm = self.baseline_metrics.get("gradient_l2_norm", gradient_norm)
        
        if baseline_grad_norm > 0 and gradient_norm / baseline_grad_norm > 3.0:
            return 0.8
        
        return 0.0
    
    def _detect_gradient_leakage(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Detect gradient leakage vulnerabilities."""
        gradient_norm = metrics.get("gradient_l2_norm", 0.0)
        noise_scale = metrics.get("noise_scale", 1.0)
        
        # High gradient norm with low noise indicates potential leakage
        if noise_scale > 0:
            leakage_score = gradient_norm / noise_scale
            return min(1.0, leakage_score / 10.0)
        
        return min(1.0, gradient_norm / 5.0)
    
    def _detect_unauthorized_access(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Detect unauthorized access attempts."""
        # Check for unusual access patterns
        failed_auth_attempts = context.get("failed_auth_attempts", 0)
        unusual_access_time = context.get("unusual_access_time", False)
        unknown_client_ip = context.get("unknown_client_ip", False)
        
        score = 0.0
        if failed_auth_attempts > 3:
            score += 0.5
        if failed_auth_attempts > 10:
            score += 0.3
        if unusual_access_time:
            score += 0.2
        if unknown_client_ip:
            score += 0.3
        
        return min(1.0, score)
    
    def _detect_abnormal_behavior(self, metrics: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Detect abnormal training behavior."""
        # Check convergence anomalies
        accuracy = metrics.get("accuracy", 0.0)
        expected_accuracy = context.get("expected_accuracy", 0.8)
        
        convergence_score = 0.0
        if accuracy < expected_accuracy * 0.5:  # Very poor performance
            convergence_score = 0.8
        elif accuracy < expected_accuracy * 0.7:
            convergence_score = 0.4
        
        # Check resource usage spikes
        memory_usage = metrics.get("memory_usage_gb", 0.0)
        cpu_usage = metrics.get("cpu_usage_percent", 0.0)
        
        resource_score = 0.0
        if memory_usage > 50:  # > 50GB
            resource_score += 0.3
        if cpu_usage > 95:
            resource_score += 0.3
        
        return max(convergence_score, resource_score)
    
    def _create_alert(
        self,
        threat_type: ThreatType,
        threat_score: float,
        metrics: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SecurityAlert:
        """Create security alert with threat details."""
        # Determine threat level based on score
        if threat_score >= 0.9:
            level = ThreatLevel.CRITICAL
        elif threat_score >= 0.7:
            level = ThreatLevel.HIGH
        elif threat_score >= 0.5:
            level = ThreatLevel.MEDIUM
        else:
            level = ThreatLevel.LOW
        
        # Generate threat ID
        threat_data = f"{threat_type.value}_{threat_score}_{time.time()}"
        threat_id = hashlib.sha256(threat_data.encode()).hexdigest()[:16]
        
        # Create descriptions and recommendations
        descriptions = {
            ThreatType.PRIVACY_BUDGET_EXHAUSTION: f"Privacy budget near exhaustion (score: {threat_score:.2f})",
            ThreatType.MODEL_INVERSION_ATTACK: f"Potential model inversion attack detected (score: {threat_score:.2f})",
            ThreatType.MEMBERSHIP_INFERENCE_ATTACK: f"Membership inference attack indicators (score: {threat_score:.2f})",
            ThreatType.DATA_POISONING: f"Data poisoning attack suspected (score: {threat_score:.2f})",
            ThreatType.GRADIENT_LEAKAGE: f"Gradient leakage vulnerability detected (score: {threat_score:.2f})",
            ThreatType.UNAUTHORIZED_ACCESS: f"Unauthorized access attempt detected (score: {threat_score:.2f})",
            ThreatType.ABNORMAL_TRAINING_BEHAVIOR: f"Abnormal training behavior observed (score: {threat_score:.2f})"
        }
        
        recommendations = {
            ThreatType.PRIVACY_BUDGET_EXHAUSTION: [
                "Reduce learning rate", "Increase noise multiplier", "Stop training early"
            ],
            ThreatType.MODEL_INVERSION_ATTACK: [
                "Increase gradient clipping", "Add more noise", "Reduce batch size"
            ],
            ThreatType.MEMBERSHIP_INFERENCE_ATTACK: [
                "Apply output perturbation", "Use confidence masking", "Increase regularization"
            ],
            ThreatType.DATA_POISONING: [
                "Validate input data", "Remove anomalous samples", "Reset to last clean checkpoint"
            ],
            ThreatType.GRADIENT_LEAKAGE: [
                "Increase noise scale", "Apply gradient compression", "Use secure aggregation"
            ],
            ThreatType.UNAUTHORIZED_ACCESS: [
                "Block suspicious IPs", "Require re-authentication", "Enable MFA"
            ],
            ThreatType.ABNORMAL_TRAINING_BEHAVIOR: [
                "Check data quality", "Validate hyperparameters", "Monitor system resources"
            ]
        }
        
        return SecurityAlert(
            threat_id=threat_id,
            threat_type=threat_type,
            threat_level=level,
            description=descriptions.get(threat_type, f"Security threat detected: {threat_type.value}"),
            detection_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            affected_components=["trainer", "privacy_engine"],
            recommended_actions=recommendations.get(threat_type, ["Investigate threat", "Apply security measures"]),
            metadata={
                "threat_score": threat_score,
                "metrics": metrics,
                "context": context,
                "detection_method": "pattern_based"
            }
        )
    
    def register_alert_handler(self, threat_type: ThreatType, handler: Callable[[SecurityAlert], None]) -> None:
        """Register custom alert handler for specific threat type."""
        self.alert_handlers[threat_type] = handler
        logger.info(f"Registered alert handler for {threat_type.value}")
    
    def _scan_for_threats(self) -> None:
        """Scan for active threats (placeholder for integration with training system)."""
        # This would be called by the main monitoring loop
        # In practice, this would collect current metrics and detect threats
        pass
    
    def _process_alert_queue(self) -> None:
        """Process queued security alerts."""
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                self._handle_alert(alert)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    def _handle_alert(self, alert: SecurityAlert) -> None:
        """Handle individual security alert."""
        logger.warning(f"Processing security alert: {alert.threat_type.value} ({alert.threat_level.value})")
        
        # Store active alert
        self.active_alerts[alert.threat_id] = alert
        
        # Update metrics
        self.security_metrics["threats_by_type"][alert.threat_type.value] = (
            self.security_metrics["threats_by_type"].get(alert.threat_type.value, 0) + 1
        )
        self.security_metrics["threats_by_level"][alert.threat_level.value] = (
            self.security_metrics["threats_by_level"].get(alert.threat_level.value, 0) + 1
        )
        
        # Call custom handler if registered
        if alert.threat_type in self.alert_handlers:
            try:
                self.alert_handlers[alert.threat_type](alert)
            except Exception as e:
                logger.error(f"Error in custom alert handler: {e}")
        
        # Automated response if enabled
        if self.enable_automated_response and alert.is_critical():
            self._execute_automated_response(alert)
    
    def _execute_automated_response(self, alert: SecurityAlert) -> None:
        """Execute automated response to critical threats."""
        logger.critical(f"Executing automated response to {alert.threat_type.value}")
        
        # Basic automated responses
        if alert.threat_type == ThreatType.PRIVACY_BUDGET_EXHAUSTION:
            logger.info("Automated response: Reducing learning rate and increasing noise")
        elif alert.threat_type == ThreatType.UNAUTHORIZED_ACCESS:
            logger.info("Automated response: Blocking suspicious access")
        elif alert.threat_type == ThreatType.DATA_POISONING:
            logger.info("Automated response: Flagging for manual review")
    
    def _cleanup_resolved_threats(self) -> None:
        """Clean up resolved or expired threat alerts."""
        current_time = time.time()
        expired_alerts = []
        
        for threat_id, alert in self.active_alerts.items():
            # Remove alerts older than 1 hour
            alert_time = time.mktime(time.strptime(alert.detection_time, "%Y-%m-%d %H:%M:%S"))
            if current_time - alert_time > 3600:  # 1 hour
                expired_alerts.append(threat_id)
        
        for threat_id in expired_alerts:
            del self.active_alerts[threat_id]
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security summary."""
        return {
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            "active_alerts_count": len(self.active_alerts),
            "critical_alerts": len([a for a in self.active_alerts.values() if a.is_critical()]),
            "security_metrics": self.security_metrics.copy(),
            "threat_patterns_loaded": len(self.threat_patterns),
            "alert_handlers_registered": len(self.alert_handlers)
        }
    
    def export_alerts(self, output_path: str, format: str = "json") -> None:
        """Export security alerts to file."""
        alerts_data = [alert.to_dict() for alert in self.active_alerts.values()]
        
        output_file = Path(output_path)
        
        if format.lower() == "json":
            with open(output_file.with_suffix('.json'), 'w') as f:
                json.dump(alerts_data, f, indent=2)
        elif format.lower() == "csv":
            import csv
            with open(output_file.with_suffix('.csv'), 'w', newline='') as f:
                if alerts_data:
                    fieldnames = alerts_data[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(alerts_data)
        
        logger.info(f"Exported {len(alerts_data)} alerts to {output_file}")
    
    def update_baseline_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update baseline metrics for anomaly detection."""
        self.baseline_metrics.update(metrics)
        logger.debug("Updated baseline metrics for threat detection")