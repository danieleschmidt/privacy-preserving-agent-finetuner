"""Advanced privacy analytics and monitoring for differential privacy systems."""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PrivacyEvent:
    """Single privacy-affecting event in the system."""
    timestamp: float
    event_type: str
    epsilon_cost: float
    delta_cost: float
    context: Dict[str, Any]


class PrivacyBudgetTracker:
    """Advanced privacy budget tracking with real-time monitoring."""
    
    def __init__(self, total_epsilon: float, total_delta: float):
        """Initialize privacy budget tracker.
        
        Args:
            total_epsilon: Total epsilon budget available
            total_delta: Total delta budget available
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.events: List[PrivacyEvent] = []
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.alert_thresholds = [0.5, 0.8, 0.9, 0.95]  # Alert at these percentages
        self.alerts_sent = set()
        
        logger.info(f"Initialized privacy budget tracker: ε={total_epsilon}, δ={total_delta}")
    
    def record_event(
        self, 
        event_type: str, 
        epsilon_cost: float, 
        delta_cost: float = 0.0,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record a privacy-affecting event.
        
        Args:
            event_type: Type of privacy event (e.g., 'training_step', 'evaluation')
            epsilon_cost: Epsilon budget consumed
            delta_cost: Delta budget consumed
            context: Additional context information
            
        Returns:
            True if event was recorded, False if budget exceeded
        """
        if self.spent_epsilon + epsilon_cost > self.total_epsilon:
            logger.error(f"Privacy budget exceeded! Requested: {epsilon_cost}, Available: {self.remaining_epsilon}")
            return False
        
        event = PrivacyEvent(
            timestamp=time.time(),
            event_type=event_type,
            epsilon_cost=epsilon_cost,
            delta_cost=delta_cost,
            context=context or {}
        )
        
        self.events.append(event)
        self.spent_epsilon += epsilon_cost
        self.spent_delta += delta_cost
        
        # Check for alerts
        self._check_budget_alerts()
        
        logger.debug(f"Recorded privacy event: {event_type}, ε_cost={epsilon_cost:.6f}")
        return True
    
    @property
    def remaining_epsilon(self) -> float:
        """Remaining epsilon budget."""
        return max(0, self.total_epsilon - self.spent_epsilon)
    
    @property
    def remaining_delta(self) -> float:
        """Remaining delta budget."""
        return max(0, self.total_delta - self.spent_delta)
    
    @property
    def epsilon_utilization(self) -> float:
        """Percentage of epsilon budget used."""
        return (self.spent_epsilon / self.total_epsilon) * 100 if self.total_epsilon > 0 else 0
    
    def _check_budget_alerts(self) -> None:
        """Check if budget usage crosses alert thresholds."""
        utilization = self.epsilon_utilization / 100
        
        for threshold in self.alert_thresholds:
            if utilization >= threshold and threshold not in self.alerts_sent:
                self.alerts_sent.add(threshold)
                logger.warning(f"Privacy budget alert: {threshold*100}% of epsilon budget used!")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive budget usage summary."""
        event_types = defaultdict(int)
        event_costs = defaultdict(float)
        
        for event in self.events:
            event_types[event.event_type] += 1
            event_costs[event.event_type] += event.epsilon_cost
        
        return {
            "total_budget": {
                "epsilon": self.total_epsilon,
                "delta": self.total_delta
            },
            "spent_budget": {
                "epsilon": self.spent_epsilon,
                "delta": self.spent_delta
            },
            "remaining_budget": {
                "epsilon": self.remaining_epsilon,
                "delta": self.remaining_delta
            },
            "utilization": {
                "epsilon_percent": self.epsilon_utilization,
                "delta_percent": (self.spent_delta / self.total_delta) * 100 if self.total_delta > 0 else 0
            },
            "event_summary": {
                "total_events": len(self.events),
                "events_by_type": dict(event_types),
                "costs_by_type": dict(event_costs)
            }
        }


class PrivacyAttackDetector:
    """Detects potential privacy attacks and vulnerabilities."""
    
    def __init__(self, window_size: int = 100):
        """Initialize attack detector.
        
        Args:
            window_size: Size of sliding window for analysis
        """
        self.window_size = window_size
        self.recent_queries = deque(maxlen=window_size)
        self.model_outputs = deque(maxlen=window_size)
        
    def analyze_membership_inference_risk(
        self, 
        query: str, 
        model_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze risk of membership inference attacks.
        
        Args:
            query: Input query/prompt
            model_output: Model's output including confidence scores
            
        Returns:
            Risk analysis results
        """
        self.recent_queries.append(query)
        self.model_outputs.append(model_output)
        
        # Check for high-confidence outputs (potential memorization)
        confidence = model_output.get('confidence', 0.0)
        high_confidence_risk = confidence > 0.95
        
        # Check for repeated similar queries
        query_similarity_risk = self._check_query_similarity(query)
        
        # Check for unusual output patterns
        output_anomaly_risk = self._detect_output_anomalies(model_output)
        
        overall_risk = "high" if any([
            high_confidence_risk,
            query_similarity_risk,
            output_anomaly_risk
        ]) else "low"
        
        return {
            "overall_risk": overall_risk,
            "high_confidence_risk": high_confidence_risk,
            "query_similarity_risk": query_similarity_risk,
            "output_anomaly_risk": output_anomaly_risk,
            "confidence_score": confidence,
            "recommendations": self._get_risk_recommendations(overall_risk)
        }
    
    def _check_query_similarity(self, query: str) -> bool:
        """Check for similar queries that might indicate probing."""
        if len(self.recent_queries) < 10:
            return False
        
        # Simple similarity check based on word overlap
        query_words = set(query.lower().split())
        similar_count = 0
        
        for recent_query in list(self.recent_queries)[-10:]:
            recent_words = set(recent_query.lower().split())
            overlap = len(query_words.intersection(recent_words))
            similarity = overlap / max(len(query_words), len(recent_words))
            
            if similarity > 0.7:  # 70% word overlap threshold
                similar_count += 1
        
        return similar_count >= 3  # 3 or more similar queries
    
    def _detect_output_anomalies(self, model_output: Dict[str, Any]) -> bool:
        """Detect anomalous output patterns."""
        if len(self.model_outputs) < 20:
            return False
        
        # Check for consistent high confidence (potential memorization)
        recent_confidences = [
            output.get('confidence', 0.0) 
            for output in list(self.model_outputs)[-20:]
        ]
        
        if NUMPY_AVAILABLE:
            avg_confidence = np.mean(recent_confidences)
            confidence_std = np.std(recent_confidences)
        else:
            avg_confidence = sum(recent_confidences) / len(recent_confidences)
            mean_conf = avg_confidence
            confidence_std = (sum((x - mean_conf) ** 2 for x in recent_confidences) / len(recent_confidences)) ** 0.5
        
        # Anomaly if very high average confidence with low variance
        return avg_confidence > 0.9 and confidence_std < 0.05
    
    def _get_risk_recommendations(self, risk_level: str) -> List[str]:
        """Get recommendations based on risk level."""
        if risk_level == "high":
            return [
                "Increase noise multiplier for future training",
                "Apply additional output sanitization",
                "Consider retraining with stricter privacy budget",
                "Implement query rate limiting",
                "Monitor for repeated similar queries"
            ]
        else:
            return [
                "Continue monitoring for unusual patterns",
                "Maintain current privacy parameters"
            ]


class PrivacyComplianceChecker:
    """Validates compliance with privacy regulations and standards."""
    
    def __init__(self):
        """Initialize compliance checker."""
        self.regulations = {
            "GDPR": {
                "max_epsilon": 1.0,
                "requires_consent": True,
                "data_minimization": True,
                "right_to_erasure": True
            },
            "HIPAA": {
                "max_epsilon": 0.5,
                "requires_encryption": True,
                "audit_trail": True,
                "access_controls": True
            },
            "CCPA": {
                "max_epsilon": 2.0,
                "requires_notice": True,
                "opt_out_rights": True
            }
        }
    
    def check_compliance(
        self, 
        privacy_config: Dict[str, Any],
        regulation: str = "GDPR"
    ) -> Dict[str, Any]:
        """Check compliance with specific regulation.
        
        Args:
            privacy_config: Privacy configuration to check
            regulation: Regulation to check against
            
        Returns:
            Compliance check results
        """
        if regulation not in self.regulations:
            return {"error": f"Unknown regulation: {regulation}"}
        
        reg_requirements = self.regulations[regulation]
        compliance_results = {
            "regulation": regulation,
            "compliant": True,
            "violations": [],
            "recommendations": []
        }
        
        # Check epsilon limits
        epsilon = privacy_config.get("epsilon", float('inf'))
        max_epsilon = reg_requirements.get("max_epsilon", float('inf'))
        
        if epsilon > max_epsilon:
            compliance_results["compliant"] = False
            compliance_results["violations"].append(
                f"Epsilon budget {epsilon} exceeds maximum {max_epsilon} for {regulation}"
            )
            compliance_results["recommendations"].append(
                f"Reduce epsilon to {max_epsilon} or lower"
            )
        
        # Check other requirements
        if reg_requirements.get("requires_encryption") and not privacy_config.get("encryption_enabled"):
            compliance_results["violations"].append("Encryption required but not enabled")
            compliance_results["recommendations"].append("Enable data encryption")
        
        if reg_requirements.get("audit_trail") and not privacy_config.get("audit_enabled"):
            compliance_results["violations"].append("Audit trail required but not enabled")
            compliance_results["recommendations"].append("Enable comprehensive audit logging")
        
        return compliance_results
    
    def generate_compliance_report(
        self, 
        privacy_config: Dict[str, Any],
        regulations: List[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report.
        
        Args:
            privacy_config: Privacy configuration
            regulations: List of regulations to check (default: all)
            
        Returns:
            Comprehensive compliance report
        """
        if regulations is None:
            regulations = list(self.regulations.keys())
        
        report = {
            "timestamp": time.time(),
            "privacy_config": privacy_config,
            "compliance_results": {},
            "overall_compliant": True
        }
        
        for regulation in regulations:
            result = self.check_compliance(privacy_config, regulation)
            report["compliance_results"][regulation] = result
            
            if not result.get("compliant", False):
                report["overall_compliant"] = False
        
        return report


def create_privacy_dashboard_data(
    budget_tracker: PrivacyBudgetTracker,
    attack_detector: PrivacyAttackDetector,
    compliance_checker: PrivacyComplianceChecker,
    privacy_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create comprehensive privacy dashboard data.
    
    Args:
        budget_tracker: Privacy budget tracker instance
        attack_detector: Attack detector instance
        compliance_checker: Compliance checker instance
        privacy_config: Current privacy configuration
        
    Returns:
        Dashboard data for monitoring interface
    """
    return {
        "budget_status": budget_tracker.get_usage_summary(),
        "compliance_status": compliance_checker.generate_compliance_report(privacy_config),
        "security_metrics": {
            "recent_queries": len(attack_detector.recent_queries),
            "total_risk_assessments": len(attack_detector.model_outputs)
        },
        "system_health": {
            "privacy_engine_active": True,
            "monitoring_enabled": True,
            "last_update": time.time()
        }
    }