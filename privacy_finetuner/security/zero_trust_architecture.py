"""
Zero Trust Architecture for Privacy-Preserving ML Systems

This module implements a comprehensive zero-trust security architecture
specifically designed for privacy-preserving machine learning environments.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import logging
import hashlib
import secrets
import numpy as np
from cryptography.fernet import Fernet
import jwt

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust levels in zero-trust architecture."""
    UNTRUSTED = 0.0
    MINIMAL = 0.2
    LIMITED = 0.4
    MODERATE = 0.6
    HIGH = 0.8
    VERIFIED = 1.0


class EntityType(Enum):
    """Types of entities in the system."""
    USER = "user"
    SERVICE = "service"
    DEVICE = "device"
    NETWORK = "network"
    DATA_SOURCE = "data_source"
    MODEL = "model"
    COMPUTE_NODE = "compute_node"


class AuthenticationMethod(Enum):
    """Supported authentication methods."""
    PASSWORD = "password"
    MFA = "multi_factor"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    HARDWARE_TOKEN = "hardware_token"
    ZERO_KNOWLEDGE = "zero_knowledge"


@dataclass
class TrustContext:
    """Context information for trust evaluation."""
    entity_id: str
    entity_type: EntityType
    authentication_method: AuthenticationMethod
    network_location: str
    device_fingerprint: str
    behavioral_patterns: Dict[str, Any]
    historical_trust: float = 0.5
    risk_factors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AccessRequest:
    """Request for system access under zero-trust model."""
    request_id: str
    entity_id: str
    resource_id: str
    action: str
    context: TrustContext
    privacy_sensitivity: str  # low, medium, high
    requested_permissions: List[str]
    justification: Optional[str] = None


@dataclass
class TrustDecision:
    """Trust-based access decision."""
    request_id: str
    decision: bool  # granted or denied
    trust_score: float
    confidence: float
    reasoning: List[str]
    conditions: List[str] = field(default_factory=list)
    expiry_time: Optional[float] = None
    monitoring_required: bool = False


class TrustEvaluator(ABC):
    """Abstract base class for trust evaluation algorithms."""
    
    @abstractmethod
    async def evaluate_trust(self, context: TrustContext) -> Tuple[float, float]:
        """Evaluate trust score and confidence level."""
        pass


class BehavioralTrustEvaluator(TrustEvaluator):
    """Trust evaluator based on behavioral analysis."""
    
    def __init__(self):
        self.behavioral_models: Dict[str, Any] = {}
        self.baseline_patterns: Dict[str, Dict[str, float]] = {}
    
    async def evaluate_trust(self, context: TrustContext) -> Tuple[float, float]:
        """Evaluate trust based on behavioral patterns."""
        base_score = 0.5
        confidence = 0.5
        
        entity_id = context.entity_id
        patterns = context.behavioral_patterns
        
        # Load baseline if available
        if entity_id in self.baseline_patterns:
            baseline = self.baseline_patterns[entity_id]
            deviation_score = self._calculate_deviation(patterns, baseline)
            
            # Lower deviation means higher trust
            behavioral_score = max(0.0, 1.0 - deviation_score)
            base_score = (base_score + behavioral_score) / 2
            confidence = min(confidence + 0.3, 1.0)
        
        # Time-based trust decay
        time_factor = self._calculate_time_factor(context.timestamp)
        final_score = base_score * time_factor
        
        return final_score, confidence
    
    def _calculate_deviation(self, current: Dict[str, Any], baseline: Dict[str, float]) -> float:
        """Calculate behavioral deviation from baseline."""
        if not baseline or not current:
            return 0.5  # Neutral when no data
        
        deviations = []
        for key, baseline_value in baseline.items():
            if key in current:
                current_value = float(current[key])
                deviation = abs(current_value - baseline_value) / max(baseline_value, 1.0)
                deviations.append(deviation)
        
        return np.mean(deviations) if deviations else 0.5
    
    def _calculate_time_factor(self, timestamp: float) -> float:
        """Calculate time-based trust decay factor."""
        age_hours = (time.time() - timestamp) / 3600
        # Trust decays over time, but slowly
        return max(0.3, 1.0 - (age_hours / 168))  # Decay over a week


class NetworkTrustEvaluator(TrustEvaluator):
    """Trust evaluator based on network security."""
    
    def __init__(self):
        self.trusted_networks: Set[str] = set()
        self.suspicious_networks: Set[str] = set()
        self.network_reputation: Dict[str, float] = {}
    
    async def evaluate_trust(self, context: TrustContext) -> Tuple[float, float]:
        """Evaluate trust based on network location and security."""
        network = context.network_location
        base_score = 0.5
        confidence = 0.7
        
        # Check trusted networks
        if network in self.trusted_networks:
            base_score = 0.9
            confidence = 0.9
        elif network in self.suspicious_networks:
            base_score = 0.1
            confidence = 0.9
        elif network in self.network_reputation:
            base_score = self.network_reputation[network]
            confidence = 0.8
        
        # Apply network security factors
        security_score = self._evaluate_network_security(network)
        final_score = (base_score + security_score) / 2
        
        return final_score, confidence
    
    def _evaluate_network_security(self, network: str) -> float:
        """Evaluate network security characteristics."""
        # Simplified network security evaluation
        # In practice, this would integrate with network security tools
        
        security_factors = {
            "encryption": 0.3,
            "firewall": 0.2,
            "intrusion_detection": 0.2,
            "vpn": 0.3
        }
        
        # Mock evaluation - in practice, query network security status
        return 0.7  # Assume moderate network security


class DeviceTrustEvaluator(TrustEvaluator):
    """Trust evaluator based on device characteristics."""
    
    def __init__(self):
        self.known_devices: Dict[str, Dict[str, Any]] = {}
        self.device_reputation: Dict[str, float] = {}
    
    async def evaluate_trust(self, context: TrustContext) -> Tuple[float, float]:
        """Evaluate trust based on device fingerprint and characteristics."""
        fingerprint = context.device_fingerprint
        base_score = 0.5
        confidence = 0.6
        
        if fingerprint in self.known_devices:
            device_info = self.known_devices[fingerprint]
            # Higher trust for known devices
            base_score = 0.8
            confidence = 0.9
            
            # Check for device integrity
            integrity_score = self._check_device_integrity(device_info)
            base_score = (base_score + integrity_score) / 2
        
        elif fingerprint in self.device_reputation:
            base_score = self.device_reputation[fingerprint]
            confidence = 0.7
        
        return base_score, confidence
    
    def _check_device_integrity(self, device_info: Dict[str, Any]) -> float:
        """Check device integrity and security posture."""
        integrity_factors = {
            "os_updated": device_info.get("os_updated", False),
            "antivirus_active": device_info.get("antivirus_active", False),
            "encryption_enabled": device_info.get("encryption_enabled", False),
            "secure_boot": device_info.get("secure_boot", False)
        }
        
        score = sum(0.25 for factor in integrity_factors.values() if factor)
        return score


class PrivacyAwareTrustEvaluator(TrustEvaluator):
    """Trust evaluator that considers privacy sensitivity."""
    
    def __init__(self):
        self.privacy_clearance: Dict[str, str] = {}  # entity_id -> clearance level
        self.data_classification: Dict[str, str] = {}  # resource_id -> classification
    
    async def evaluate_trust(self, context: TrustContext) -> Tuple[float, float]:
        """Evaluate trust considering privacy requirements."""
        entity_id = context.entity_id
        base_score = 0.5
        confidence = 0.8
        
        # Check privacy clearance
        if entity_id in self.privacy_clearance:
            clearance = self.privacy_clearance[entity_id]
            clearance_score = self._map_clearance_to_score(clearance)
            base_score = (base_score + clearance_score) / 2
            confidence = 0.9
        
        # Consider risk factors
        privacy_risk_score = self._evaluate_privacy_risks(context.risk_factors)
        final_score = base_score * (1.0 - privacy_risk_score)
        
        return final_score, confidence
    
    def _map_clearance_to_score(self, clearance: str) -> float:
        """Map privacy clearance level to trust score."""
        clearance_mapping = {
            "public": 0.3,
            "internal": 0.5,
            "confidential": 0.7,
            "restricted": 0.9,
            "top_secret": 1.0
        }
        return clearance_mapping.get(clearance.lower(), 0.3)
    
    def _evaluate_privacy_risks(self, risk_factors: List[str]) -> float:
        """Evaluate privacy-related risk factors."""
        risk_weights = {
            "data_breach_history": 0.3,
            "compliance_violations": 0.4,
            "suspicious_access_patterns": 0.2,
            "unauthorized_data_sharing": 0.5,
            "privacy_policy_violations": 0.3
        }
        
        total_risk = sum(risk_weights.get(factor, 0.1) for factor in risk_factors)
        return min(total_risk, 1.0)


class ZeroTrustPolicyEngine:
    """Policy engine for zero-trust access decisions."""
    
    def __init__(self):
        self.policies: Dict[str, Dict[str, Any]] = {}
        self.default_policy = {
            "min_trust_score": 0.7,
            "require_mfa": True,
            "max_session_duration": 3600,
            "continuous_monitoring": True
        }
    
    def add_policy(self, resource_id: str, policy: Dict[str, Any]):
        """Add resource-specific access policy."""
        self.policies[resource_id] = policy
    
    def evaluate_policy(self, request: AccessRequest, trust_score: float) -> TrustDecision:
        """Evaluate access request against policies."""
        resource_policy = self.policies.get(request.resource_id, self.default_policy)
        
        # Check minimum trust threshold
        min_trust = resource_policy.get("min_trust_score", 0.7)
        if trust_score < min_trust:
            return TrustDecision(
                request_id=request.request_id,
                decision=False,
                trust_score=trust_score,
                confidence=0.9,
                reasoning=[f"Trust score {trust_score:.2f} below minimum {min_trust}"]
            )
        
        # Check MFA requirement
        if resource_policy.get("require_mfa", False):
            if request.context.authentication_method != AuthenticationMethod.MFA:
                return TrustDecision(
                    request_id=request.request_id,
                    decision=False,
                    trust_score=trust_score,
                    confidence=0.9,
                    reasoning=["Multi-factor authentication required"]
                )
        
        # Check privacy sensitivity
        privacy_requirements = self._check_privacy_requirements(request, resource_policy)
        if not privacy_requirements["allowed"]:
            return TrustDecision(
                request_id=request.request_id,
                decision=False,
                trust_score=trust_score,
                confidence=0.9,
                reasoning=privacy_requirements["reasons"]
            )
        
        # Grant access with conditions
        conditions = []
        expiry_time = None
        
        if "max_session_duration" in resource_policy:
            expiry_time = time.time() + resource_policy["max_session_duration"]
            conditions.append(f"Session expires in {resource_policy['max_session_duration']} seconds")
        
        monitoring_required = resource_policy.get("continuous_monitoring", False)
        if monitoring_required:
            conditions.append("Continuous monitoring enabled")
        
        return TrustDecision(
            request_id=request.request_id,
            decision=True,
            trust_score=trust_score,
            confidence=0.9,
            reasoning=["Access granted based on trust evaluation"],
            conditions=conditions,
            expiry_time=expiry_time,
            monitoring_required=monitoring_required
        )
    
    def _check_privacy_requirements(self, request: AccessRequest, policy: Dict[str, Any]) -> Dict[str, Any]:
        """Check privacy-specific access requirements."""
        allowed = True
        reasons = []
        
        # Check data classification compatibility
        required_clearance = policy.get("required_clearance", "public")
        entity_clearance = policy.get("entity_clearances", {}).get(request.entity_id, "public")
        
        clearance_levels = ["public", "internal", "confidential", "restricted", "top_secret"]
        required_level = clearance_levels.index(required_clearance)
        entity_level = clearance_levels.index(entity_clearance)
        
        if entity_level < required_level:
            allowed = False
            reasons.append(f"Insufficient clearance: {entity_clearance} < {required_clearance}")
        
        # Check privacy sensitivity
        if request.privacy_sensitivity == "high":
            if not policy.get("allow_high_privacy_access", False):
                allowed = False
                reasons.append("High privacy data access not permitted")
        
        return {"allowed": allowed, "reasons": reasons}


class ZeroTrustAccessManager:
    """Main zero-trust access management system."""
    
    def __init__(self):
        self.evaluators: List[TrustEvaluator] = [
            BehavioralTrustEvaluator(),
            NetworkTrustEvaluator(),
            DeviceTrustEvaluator(),
            PrivacyAwareTrustEvaluator()
        ]
        self.policy_engine = ZeroTrustPolicyEngine()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.access_log: List[Dict[str, Any]] = []
        
    async def evaluate_access_request(self, request: AccessRequest) -> TrustDecision:
        """Evaluate access request using zero-trust principles."""
        # Collect trust scores from all evaluators
        trust_scores = []
        confidences = []
        
        for evaluator in self.evaluators:
            try:
                score, confidence = await evaluator.evaluate_trust(request.context)
                trust_scores.append(score)
                confidences.append(confidence)
            except Exception as e:
                logger.error(f"Trust evaluator error: {e}")
                trust_scores.append(0.0)  # Conservative scoring on error
                confidences.append(0.5)
        
        # Calculate composite trust score
        weighted_trust = self._calculate_composite_trust(trust_scores, confidences)
        
        # Apply policy evaluation
        decision = self.policy_engine.evaluate_policy(request, weighted_trust)
        
        # Log access attempt
        self._log_access_attempt(request, decision)
        
        # Create session if access granted
        if decision.decision:
            self._create_session(request, decision)
        
        return decision
    
    def _calculate_composite_trust(self, scores: List[float], confidences: List[float]) -> float:
        """Calculate weighted composite trust score."""
        if not scores:
            return 0.0
        
        # Weight scores by confidence levels
        weighted_sum = sum(score * confidence for score, confidence in zip(scores, confidences))
        confidence_sum = sum(confidences)
        
        if confidence_sum == 0:
            return 0.0
        
        return weighted_sum / confidence_sum
    
    def _log_access_attempt(self, request: AccessRequest, decision: TrustDecision):
        """Log access attempt for audit purposes."""
        log_entry = {
            "timestamp": time.time(),
            "request_id": request.request_id,
            "entity_id": request.entity_id,
            "resource_id": request.resource_id,
            "action": request.action,
            "decision": decision.decision,
            "trust_score": decision.trust_score,
            "reasoning": decision.reasoning,
            "network_location": request.context.network_location,
            "authentication_method": request.context.authentication_method.value
        }
        self.access_log.append(log_entry)
        
        # Keep only last 10000 entries
        if len(self.access_log) > 10000:
            self.access_log = self.access_log[-10000:]
    
    def _create_session(self, request: AccessRequest, decision: TrustDecision):
        """Create active session for granted access."""
        session_id = secrets.token_hex(16)
        session_info = {
            "session_id": session_id,
            "entity_id": request.entity_id,
            "resource_id": request.resource_id,
            "start_time": time.time(),
            "expiry_time": decision.expiry_time,
            "trust_score": decision.trust_score,
            "monitoring_required": decision.monitoring_required,
            "permissions": request.requested_permissions
        }
        
        self.active_sessions[session_id] = session_info
        logger.info(f"Created session {session_id} for entity {request.entity_id}")
    
    def validate_session(self, session_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate active session."""
        if session_id not in self.active_sessions:
            return False, {"reason": "Session not found"}
        
        session = self.active_sessions[session_id]
        current_time = time.time()
        
        # Check expiry
        if session.get("expiry_time") and current_time > session["expiry_time"]:
            del self.active_sessions[session_id]
            return False, {"reason": "Session expired"}
        
        return True, session
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke active session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Revoked session {session_id}")
            return True
        return False
    
    def get_access_statistics(self) -> Dict[str, Any]:
        """Get access control statistics."""
        if not self.access_log:
            return {"total_requests": 0}
        
        total_requests = len(self.access_log)
        granted_requests = sum(1 for entry in self.access_log if entry["decision"])
        
        recent_hour = [entry for entry in self.access_log 
                      if time.time() - entry["timestamp"] < 3600]
        
        avg_trust_score = np.mean([entry["trust_score"] for entry in self.access_log])
        
        return {
            "total_requests": total_requests,
            "granted_requests": granted_requests,
            "denied_requests": total_requests - granted_requests,
            "success_rate": granted_requests / total_requests if total_requests > 0 else 0,
            "requests_last_hour": len(recent_hour),
            "average_trust_score": avg_trust_score,
            "active_sessions": len(self.active_sessions)
        }


# Utility functions
def create_zero_trust_manager() -> ZeroTrustAccessManager:
    """Factory function to create zero-trust access manager."""
    return ZeroTrustAccessManager()


def create_access_request(
    entity_id: str,
    resource_id: str,
    action: str,
    context: TrustContext,
    privacy_sensitivity: str = "medium",
    permissions: Optional[List[str]] = None
) -> AccessRequest:
    """Helper function to create access request."""
    return AccessRequest(
        request_id=secrets.token_hex(16),
        entity_id=entity_id,
        resource_id=resource_id,
        action=action,
        context=context,
        privacy_sensitivity=privacy_sensitivity,
        requested_permissions=permissions or []
    )


def create_trust_context(
    entity_id: str,
    entity_type: EntityType,
    auth_method: AuthenticationMethod,
    network_location: str,
    device_fingerprint: str,
    behavioral_patterns: Optional[Dict[str, Any]] = None
) -> TrustContext:
    """Helper function to create trust context."""
    return TrustContext(
        entity_id=entity_id,
        entity_type=entity_type,
        authentication_method=auth_method,
        network_location=network_location,
        device_fingerprint=device_fingerprint,
        behavioral_patterns=behavioral_patterns or {}
    )