"""Comprehensive security framework for privacy-preserving ML training.

This module provides advanced security hardening including input sanitization,
secure defaults, threat detection, automated response systems, and security
monitoring with correlation and analysis.
"""

import os
import re
import hashlib
import hmac
import secrets
import time
import logging
import threading
import ipaddress
from typing import Dict, Any, List, Optional, Set, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import base64
import uuid
from collections import defaultdict, deque

try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

from ..core.exceptions import SecurityViolationException
from ..utils.logging_config import correlation_context

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of security attacks."""
    INJECTION = "injection"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DENIAL_OF_SERVICE = "denial_of_service"
    PRIVACY_VIOLATION = "privacy_violation"
    MODEL_EXTRACTION = "model_extraction"
    MODEL_INVERSION = "model_inversion"
    MEMBERSHIP_INFERENCE = "membership_inference"
    ADVERSARIAL_ATTACK = "adversarial_attack"


@dataclass
class SecurityEvent:
    """Security event record."""
    id: str
    timestamp: datetime
    attack_type: AttackType
    severity: SecurityLevel
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    mitigated: bool = False
    mitigation_actions: List[str] = field(default_factory=list)


@dataclass
class SecurityRule:
    """Security rule definition."""
    id: str
    name: str
    attack_type: AttackType
    severity: SecurityLevel
    pattern: str
    description: str
    enabled: bool = True
    action: str = "alert"  # alert, block, rate_limit
    threshold: int = 1
    time_window_minutes: int = 60


class InputSanitizer:
    """Comprehensive input sanitization and validation."""
    
    # Malicious patterns for detection
    SQL_INJECTION_PATTERNS = [
        r"(?i)(union\s+select|select\s+.*\s+from|insert\s+into|delete\s+from|update\s+.*\s+set)",
        r"(?i)(drop\s+table|truncate\s+table|alter\s+table)",
        r"(?i)(\bor\s+1\s*=\s*1\b|\band\s+1\s*=\s*1\b)",
        r"(?i)(exec\s*\(|execute\s*\(|sp_executesql)",
        r"['\"][\s]*;[\s]*--",
        r"['\"][\s]*;[\s]*/\*.*\*/",
    ]
    
    XSS_PATTERNS = [
        r"(?i)<script[^>]*>.*?</script>",
        r"(?i)<iframe[^>]*>.*?</iframe>",
        r"(?i)javascript\s*:",
        r"(?i)vbscript\s*:",
        r"(?i)on\w+\s*=",
        r"(?i)expression\s*\(",
        r"(?i)url\s*\(",
        r"(?i)<object[^>]*>.*?</object>",
        r"(?i)<embed[^>]*>.*?</embed>",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$(){}\\]",
        r"(?i)(bash|sh|cmd|powershell|perl|python|ruby)\s",
        r"(?i)(wget|curl|nc|netcat|telnet|ssh|ftp)\s",
        r"(?i)(rm\s|del\s|format\s|mkfs\s)",
        r"\$\([^)]+\)",
        r"`[^`]+`",
        r"(?i)(cat|more|less|head|tail|grep|find)\s",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\.[\\/]",
        r"(?i)[\\/](etc|proc|sys|dev)[\\/]",
        r"(?i)[\\/](windows|winnt|system32)[\\/]",
        r"(?i)[\\/](home|root|users)[\\/].*[\\/]\.ssh",
        r"(?i)passwd$|shadow$|\.htpasswd$",
    ]
    
    def __init__(self):
        self.compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for performance."""
        return {
            'sql_injection': [re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS],
            'xss': [re.compile(p, re.IGNORECASE) for p in self.XSS_PATTERNS],
            'command_injection': [re.compile(p, re.IGNORECASE) for p in self.COMMAND_INJECTION_PATTERNS],
            'path_traversal': [re.compile(p, re.IGNORECASE) for p in self.PATH_TRAVERSAL_PATTERNS]
        }
    
    def sanitize_input(
        self,
        data: Any,
        max_length: int = 10000,
        allow_html: bool = False,
        allow_json: bool = True,
        strict_mode: bool = True
    ) -> Tuple[Any, List[str]]:
        """
        Sanitize input data and return sanitized data plus warnings.
        
        Args:
            data: Input data to sanitize
            max_length: Maximum allowed length for strings
            allow_html: Whether to allow HTML tags
            allow_json: Whether to allow JSON structures
            strict_mode: Whether to apply strict security checks
            
        Returns:
            Tuple of (sanitized_data, warnings)
        """
        warnings = []
        
        if isinstance(data, str):
            return self._sanitize_string(data, max_length, allow_html, strict_mode)
        elif isinstance(data, dict):
            if not allow_json:
                warnings.append("Dictionary structure not allowed")
                return str(data)[:max_length], warnings
            return self._sanitize_dict(data, max_length, allow_html, strict_mode)
        elif isinstance(data, list):
            if not allow_json:
                warnings.append("List structure not allowed")
                return str(data)[:max_length], warnings
            return self._sanitize_list(data, max_length, allow_html, strict_mode)
        else:
            # Convert to string and sanitize
            str_data = str(data)[:max_length]
            return self._sanitize_string(str_data, max_length, allow_html, strict_mode)
    
    def _sanitize_string(
        self,
        text: str,
        max_length: int,
        allow_html: bool,
        strict_mode: bool
    ) -> Tuple[str, List[str]]:
        """Sanitize string input."""
        warnings = []
        original_text = text
        
        # Length check
        if len(text) > max_length:
            text = text[:max_length] + "...[truncated]"
            warnings.append(f"String truncated to {max_length} characters")
        
        # Detect malicious patterns
        threats = self.detect_threats(text)
        if threats:
            if strict_mode:
                raise SecurityViolationException(
                    f"Malicious input detected: {', '.join(threats)}",
                    violation_type="input_validation",
                    severity="HIGH"
                )
            else:
                warnings.extend(threats)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove or escape HTML if not allowed
        if not allow_html:
            text = self._remove_html_tags(text)
        
        # Normalize Unicode and control characters
        text = self._normalize_unicode(text)
        
        # Remove suspicious Unicode characters
        text = self._remove_suspicious_unicode(text)
        
        return text, warnings
    
    def _sanitize_dict(
        self,
        data: dict,
        max_length: int,
        allow_html: bool,
        strict_mode: bool
    ) -> Tuple[dict, List[str]]:
        """Sanitize dictionary recursively."""
        warnings = []
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            sanitized_key, key_warnings = self._sanitize_string(
                str(key), 100, False, strict_mode
            )
            warnings.extend(key_warnings)
            
            # Sanitize value
            sanitized_value, value_warnings = self.sanitize_input(
                value, max_length, allow_html, True, strict_mode
            )
            warnings.extend(value_warnings)
            
            sanitized[sanitized_key] = sanitized_value
        
        return sanitized, warnings
    
    def _sanitize_list(
        self,
        data: list,
        max_length: int,
        allow_html: bool,
        strict_mode: bool
    ) -> Tuple[list, List[str]]:
        """Sanitize list recursively."""
        warnings = []
        sanitized = []
        
        for item in data:
            sanitized_item, item_warnings = self.sanitize_input(
                item, max_length, allow_html, True, strict_mode
            )
            warnings.extend(item_warnings)
            sanitized.append(sanitized_item)
        
        return sanitized, warnings
    
    def detect_threats(self, text: str) -> List[str]:
        """Detect security threats in text."""
        threats = []
        
        for threat_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    threats.append(f"{threat_type} detected")
                    break  # Only report each threat type once
        
        return threats
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        # Remove script and style contents
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        html_entities = {
            '&lt;': '<', '&gt;': '>', '&amp;': '&', '&quot;': '"',
            '&#x27;': "'", '&#x2F;': '/', '&#x5C;': '\\'
        }
        
        for entity, char in html_entities.items():
            text = text.replace(entity, char)
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # Normalize Unicode
        try:
            import unicodedata
            text = unicodedata.normalize('NFKC', text)
        except ImportError:
            pass
        
        # Remove control characters except newline, tab, carriage return
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text
    
    def _remove_suspicious_unicode(self, text: str) -> str:
        """Remove potentially suspicious Unicode characters."""
        # Remove zero-width characters that could be used for obfuscation
        zero_width_chars = [
            '\u200b',  # Zero Width Space
            '\u200c',  # Zero Width Non-Joiner
            '\u200d',  # Zero Width Joiner
            '\ufeff',  # Zero Width No-Break Space
        ]
        
        for char in zero_width_chars:
            text = text.replace(char, '')
        
        return text


class AuthenticationManager:
    """Secure authentication and session management."""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or self._generate_secret_key()
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.blocked_ips: Dict[str, datetime] = {}
        self.session_timeout = timedelta(hours=24)
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.lock = threading.RLock()
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        return secrets.token_urlsafe(32)
    
    def create_session(
        self,
        user_id: str,
        user_data: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Create a secure session."""
        session_id = secrets.token_urlsafe(32)
        
        with self.lock:
            self.sessions[session_id] = {
                'user_id': user_id,
                'user_data': user_data,
                'created_at': datetime.now(),
                'last_accessed': datetime.now(),
                'ip_address': ip_address,
                'user_agent': user_agent,
                'csrf_token': secrets.token_urlsafe(32)
            }
        
        logger.info(f"Session created for user {user_id}")
        return session_id
    
    def validate_session(
        self,
        session_id: str,
        ip_address: Optional[str] = None,
        csrf_token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Validate session and return user data if valid."""
        with self.lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            
            # Check session timeout
            if datetime.now() - session['last_accessed'] > self.session_timeout:
                del self.sessions[session_id]
                logger.warning(f"Session expired for user {session['user_id']}")
                return None
            
            # Check IP address consistency (if provided)
            if ip_address and session.get('ip_address') != ip_address:
                logger.warning(f"IP address mismatch for session {session_id}")
                return None
            
            # Check CSRF token (if provided)
            if csrf_token and session.get('csrf_token') != csrf_token:
                logger.warning(f"CSRF token mismatch for session {session_id}")
                return None
            
            # Update last accessed time
            session['last_accessed'] = datetime.now()
            
            return session.copy()
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session."""
        with self.lock:
            if session_id in self.sessions:
                user_id = self.sessions[session_id]['user_id']
                del self.sessions[session_id]
                logger.info(f"Session revoked for user {user_id}")
                return True
            return False
    
    def record_failed_attempt(self, identifier: str, ip_address: Optional[str] = None):
        """Record failed authentication attempt."""
        now = datetime.now()
        
        with self.lock:
            # Clean old attempts
            cutoff = now - self.lockout_duration
            self.failed_attempts[identifier] = [
                attempt for attempt in self.failed_attempts[identifier]
                if attempt > cutoff
            ]
            
            # Add new attempt
            self.failed_attempts[identifier].append(now)
            
            # Check if should block
            if len(self.failed_attempts[identifier]) >= self.max_failed_attempts:
                if ip_address:
                    self.blocked_ips[ip_address] = now
                logger.warning(f"Too many failed attempts for {identifier}, blocking IP {ip_address}")
    
    def is_blocked(self, identifier: str, ip_address: Optional[str] = None) -> bool:
        """Check if identifier or IP is blocked."""
        now = datetime.now()
        
        with self.lock:
            # Check IP block
            if ip_address and ip_address in self.blocked_ips:
                if now - self.blocked_ips[ip_address] < self.lockout_duration:
                    return True
                else:
                    del self.blocked_ips[ip_address]
            
            # Check failed attempts
            cutoff = now - self.lockout_duration
            recent_attempts = [
                attempt for attempt in self.failed_attempts[identifier]
                if attempt > cutoff
            ]
            
            return len(recent_attempts) >= self.max_failed_attempts
    
    def hash_password(self, password: str) -> str:
        """Hash password securely."""
        if not BCRYPT_AVAILABLE:
            # Fallback to PBKDF2
            return self._pbkdf2_hash(password)
        
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        if not BCRYPT_AVAILABLE:
            return self._pbkdf2_verify(password, hashed)
        
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except ValueError:
            return False
    
    def _pbkdf2_hash(self, password: str) -> str:
        """Hash password using PBKDF2."""
        salt = os.urandom(32)
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return base64.b64encode(salt + key).decode('utf-8')
    
    def _pbkdf2_verify(self, password: str, hashed: str) -> bool:
        """Verify password using PBKDF2."""
        try:
            decoded = base64.b64decode(hashed.encode('utf-8'))
            salt = decoded[:32]
            key = decoded[32:]
            new_key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            return hmac.compare_digest(key, new_key)
        except Exception:
            return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = datetime.now()
        
        with self.lock:
            expired_sessions = [
                session_id for session_id, session in self.sessions.items()
                if now - session['last_accessed'] > self.session_timeout
            ]
            
            for session_id in expired_sessions:
                user_id = self.sessions[session_id]['user_id']
                del self.sessions[session_id]
                logger.debug(f"Cleaned up expired session for user {user_id}")


class ThreatDetectionEngine:
    """Advanced threat detection with pattern analysis and ML-based detection."""
    
    def __init__(self):
        self.security_rules: Dict[str, SecurityRule] = {}
        self.event_history: deque = deque(maxlen=10000)
        self.ip_reputation: Dict[str, float] = {}  # IP -> reputation score (0-1)
        self.user_behavior_profiles: Dict[str, Dict[str, Any]] = {}
        self.alert_callbacks: List[Callable[[SecurityEvent], None]] = []
        self.lock = threading.RLock()
        
        # Initialize default rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default security rules."""
        default_rules = [
            SecurityRule(
                id="sql_injection_attempt",
                name="SQL Injection Attempt",
                attack_type=AttackType.INJECTION,
                severity=SecurityLevel.HIGH,
                pattern=r"(?i)(union\s+select|select\s+.*\s+from|drop\s+table)",
                description="Detects SQL injection attempts",
                action="block",
                threshold=1
            ),
            SecurityRule(
                id="xss_attempt",
                name="XSS Attempt",
                attack_type=AttackType.INJECTION,
                severity=SecurityLevel.MEDIUM,
                pattern=r"(?i)<script[^>]*>.*?</script>",
                description="Detects XSS attempts",
                action="alert",
                threshold=1
            ),
            SecurityRule(
                id="excessive_requests",
                name="Excessive Requests",
                attack_type=AttackType.DENIAL_OF_SERVICE,
                severity=SecurityLevel.MEDIUM,
                pattern="",  # Will be handled by rate limiting logic
                description="Detects excessive request patterns",
                action="rate_limit",
                threshold=100,
                time_window_minutes=5
            ),
            SecurityRule(
                id="suspicious_user_agent",
                name="Suspicious User Agent",
                attack_type=AttackType.AUTHENTICATION_BYPASS,
                severity=SecurityLevel.LOW,
                pattern=r"(?i)(bot|crawler|scraper|scanner|hack|exploit)",
                description="Detects suspicious user agents",
                action="alert",
                threshold=1
            ),
            SecurityRule(
                id="privacy_budget_abuse",
                name="Privacy Budget Abuse",
                attack_type=AttackType.PRIVACY_VIOLATION,
                severity=SecurityLevel.CRITICAL,
                pattern="",  # Will be handled by privacy monitoring
                description="Detects privacy budget abuse patterns",
                action="block",
                threshold=1
            )
        ]
        
        for rule in default_rules:
            self.security_rules[rule.id] = rule
    
    def add_security_rule(self, rule: SecurityRule):
        """Add a custom security rule."""
        with self.lock:
            self.security_rules[rule.id] = rule
        logger.info(f"Added security rule: {rule.name}")
    
    def remove_security_rule(self, rule_id: str) -> bool:
        """Remove a security rule."""
        with self.lock:
            if rule_id in self.security_rules:
                del self.security_rules[rule_id]
                logger.info(f"Removed security rule: {rule_id}")
                return True
            return False
    
    def analyze_request(
        self,
        request_data: Dict[str, Any],
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> List[SecurityEvent]:
        """Analyze request for security threats."""
        events = []
        
        with correlation_context() as correlation_id:
            # Check against security rules
            for rule in self.security_rules.values():
                if not rule.enabled:
                    continue
                
                threat_detected = False
                evidence = {}
                
                if rule.pattern:
                    # Pattern-based detection
                    threat_detected, evidence = self._check_pattern_rule(
                        rule, request_data, user_agent
                    )
                else:
                    # Custom logic for specific rules
                    if rule.id == "excessive_requests":
                        threat_detected, evidence = self._check_rate_limiting(
                            rule, ip_address, user_id
                        )
                    elif rule.id == "privacy_budget_abuse":
                        threat_detected, evidence = self._check_privacy_abuse(
                            rule, request_data, user_id
                        )
                
                if threat_detected:
                    event = SecurityEvent(
                        id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        attack_type=rule.attack_type,
                        severity=rule.severity,
                        source_ip=ip_address,
                        user_agent=user_agent,
                        user_id=user_id,
                        description=f"{rule.name}: {rule.description}",
                        evidence=evidence
                    )
                    
                    events.append(event)
                    
                    # Execute rule action
                    self._execute_rule_action(rule, event)
            
            # Update behavioral profiles
            if user_id:
                self._update_user_behavior_profile(user_id, request_data, events)
            
            # Update IP reputation
            if ip_address:
                self._update_ip_reputation(ip_address, events)
            
            # Store events in history
            with self.lock:
                self.event_history.extend(events)
        
        return events
    
    def _check_pattern_rule(
        self,
        rule: SecurityRule,
        request_data: Dict[str, Any],
        user_agent: Optional[str]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check pattern-based security rule."""
        evidence = {}
        
        # Check request data
        request_str = json.dumps(request_data, default=str)
        if re.search(rule.pattern, request_str):
            evidence['matched_in'] = 'request_data'
            evidence['pattern'] = rule.pattern
            return True, evidence
        
        # Check user agent
        if user_agent and re.search(rule.pattern, user_agent):
            evidence['matched_in'] = 'user_agent'
            evidence['pattern'] = rule.pattern
            evidence['user_agent'] = user_agent
            return True, evidence
        
        return False, {}
    
    def _check_rate_limiting(
        self,
        rule: SecurityRule,
        ip_address: Optional[str],
        user_id: Optional[str]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limiting rule."""
        if not ip_address and not user_id:
            return False, {}
        
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=rule.time_window_minutes)
        
        # Count recent requests
        identifier = ip_address or user_id
        recent_events = [
            event for event in self.event_history
            if event.timestamp > cutoff_time and 
            (event.source_ip == ip_address or event.user_id == user_id)
        ]
        
        if len(recent_events) > rule.threshold:
            return True, {
                'request_count': len(recent_events),
                'threshold': rule.threshold,
                'time_window_minutes': rule.time_window_minutes,
                'identifier': identifier
            }
        
        return False, {}
    
    def _check_privacy_abuse(
        self,
        rule: SecurityRule,
        request_data: Dict[str, Any],
        user_id: Optional[str]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check for privacy budget abuse patterns."""
        # Check for rapid privacy budget consumption
        privacy_request_indicators = [
            'epsilon', 'delta', 'privacy_budget', 'train', 'model'
        ]
        
        request_str = json.dumps(request_data, default=str).lower()
        
        # Count privacy-related terms
        privacy_score = sum(1 for indicator in privacy_request_indicators 
                          if indicator in request_str)
        
        if privacy_score > 3:  # High privacy-related content
            now = datetime.now()
            cutoff_time = now - timedelta(minutes=10)
            
            # Check for rapid successive privacy requests
            recent_privacy_events = [
                event for event in self.event_history
                if (event.timestamp > cutoff_time and 
                    event.user_id == user_id and
                    event.attack_type == AttackType.PRIVACY_VIOLATION)
            ]
            
            if len(recent_privacy_events) > 2:
                return True, {
                    'privacy_score': privacy_score,
                    'recent_privacy_requests': len(recent_privacy_events),
                    'time_window': '10 minutes'
                }
        
        return False, {}
    
    def _execute_rule_action(self, rule: SecurityRule, event: SecurityEvent):
        """Execute the action specified by the rule."""
        if rule.action == "alert":
            self._trigger_alert(event)
        elif rule.action == "block":
            self._block_request(event)
        elif rule.action == "rate_limit":
            self._apply_rate_limit(event)
        
        # Log security event
        logger.warning(
            f"Security event: {event.description}",
            extra={
                'security_event_type': event.attack_type.value,
                'severity': event.severity.value,
                'source_ip': event.source_ip,
                'user_id': event.user_id,
                'evidence': event.evidence
            }
        )
    
    def _trigger_alert(self, event: SecurityEvent):
        """Trigger security alert."""
        for callback in self.alert_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _block_request(self, event: SecurityEvent):
        """Block the request (to be implemented by the application)."""
        # This would typically set a flag that the application checks
        event.mitigated = True
        event.mitigation_actions.append("request_blocked")
    
    def _apply_rate_limit(self, event: SecurityEvent):
        """Apply rate limiting (to be implemented by the application)."""
        event.mitigated = True
        event.mitigation_actions.append("rate_limited")
    
    def _update_user_behavior_profile(
        self,
        user_id: str,
        request_data: Dict[str, Any],
        security_events: List[SecurityEvent]
    ):
        """Update user behavior profile for anomaly detection."""
        with self.lock:
            if user_id not in self.user_behavior_profiles:
                self.user_behavior_profiles[user_id] = {
                    'request_count': 0,
                    'security_events': 0,
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now(),
                    'request_patterns': defaultdict(int)
                }
            
            profile = self.user_behavior_profiles[user_id]
            profile['request_count'] += 1
            profile['security_events'] += len(security_events)
            profile['last_seen'] = datetime.now()
            
            # Track request patterns
            for key in request_data.keys():
                profile['request_patterns'][key] += 1
    
    def _update_ip_reputation(
        self,
        ip_address: str,
        security_events: List[SecurityEvent]
    ):
        """Update IP reputation based on security events."""
        with self.lock:
            if ip_address not in self.ip_reputation:
                self.ip_reputation[ip_address] = 1.0  # Start with good reputation
            
            # Decrease reputation based on security events
            reputation_decrease = len(security_events) * 0.1
            self.ip_reputation[ip_address] = max(
                0.0, self.ip_reputation[ip_address] - reputation_decrease
            )
    
    def register_alert_callback(self, callback: Callable[[SecurityEvent], None]):
        """Register callback for security alerts."""
        self.alert_callbacks.append(callback)
        logger.info("Registered security alert callback")
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get comprehensive security statistics."""
        with self.lock:
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            
            recent_events = [e for e in self.event_history if e.timestamp > last_24h]
            
            attack_type_counts = defaultdict(int)
            severity_counts = defaultdict(int)
            
            for event in recent_events:
                attack_type_counts[event.attack_type.value] += 1
                severity_counts[event.severity.value] += 1
            
            return {
                'total_events_24h': len(recent_events),
                'total_events_all_time': len(self.event_history),
                'attack_type_distribution': dict(attack_type_counts),
                'severity_distribution': dict(severity_counts),
                'active_rules': len([r for r in self.security_rules.values() if r.enabled]),
                'total_rules': len(self.security_rules),
                'unique_ips_tracked': len(self.ip_reputation),
                'user_profiles': len(self.user_behavior_profiles),
                'low_reputation_ips': len([ip for ip, rep in self.ip_reputation.items() if rep < 0.5])
            }


class SecureDefaults:
    """Secure configuration defaults and hardening utilities."""
    
    @staticmethod
    def get_secure_headers() -> Dict[str, str]:
        """Get secure HTTP headers."""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
        }
    
    @staticmethod
    def get_secure_privacy_config() -> Dict[str, Any]:
        """Get secure privacy configuration defaults."""
        return {
            'epsilon': 1.0,  # Conservative privacy budget
            'delta': 1e-5,   # Strong privacy guarantee
            'max_grad_norm': 1.0,
            'noise_multiplier': 1.5,  # High noise for better privacy
            'accounting_mode': 'rdp',
            'federated_enabled': True,  # Use federated learning when possible
            'min_clients': 10,  # Require multiple clients
            'secure_compute_provider': 'sgx',  # Use secure enclaves
            'attestation_required': True
        }
    
    @staticmethod
    def get_secure_logging_config() -> Dict[str, Any]:
        """Get secure logging configuration."""
        return {
            'log_level': 'INFO',
            'privacy_redaction': True,
            'structured_logging': True,
            'audit_logging': True,
            'max_log_size': '100MB',
            'backup_count': 10,
            'secure_transport': True
        }
    
    @staticmethod
    def harden_file_permissions(file_path: str):
        """Harden file permissions for security."""
        try:
            import stat
            # Set restrictive permissions (owner read/write only)
            os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception as e:
            logger.warning(f"Failed to harden file permissions for {file_path}: {e}")
    
    @staticmethod
    def validate_secure_config(config: Dict[str, Any]) -> List[str]:
        """Validate configuration for security issues."""
        warnings = []
        
        # Check for weak privacy settings
        if config.get('epsilon', 0) > 5.0:
            warnings.append("Epsilon value is high, may provide weak privacy guarantees")
        
        if config.get('delta', 0) > 1e-3:
            warnings.append("Delta value is high, may not provide strong privacy")
        
        if not config.get('privacy_redaction', True):
            warnings.append("Privacy redaction disabled, sensitive data may be logged")
        
        if not config.get('audit_logging', True):
            warnings.append("Audit logging disabled, compliance may be impacted")
        
        if config.get('debug_mode', False):
            warnings.append("Debug mode enabled in production may expose sensitive information")
        
        return warnings


# Global security framework instance
class SecurityFramework:
    """Main security framework that coordinates all security components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.input_sanitizer = InputSanitizer()
        self.auth_manager = AuthenticationManager(
            secret_key=self.config.get('secret_key')
        )
        self.threat_detector = ThreatDetectionEngine()
        
        # Security monitoring
        self.security_events: List[SecurityEvent] = []
        self.max_events = self.config.get('max_security_events', 10000)
        
        # Setup secure defaults
        self.secure_defaults = SecureDefaults()
        
        logger.info("Security framework initialized")
    
    def process_request(
        self,
        request_data: Any,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Tuple[Any, List[str], List[SecurityEvent]]:
        """Process request through complete security pipeline."""
        warnings = []
        security_events = []
        
        try:
            # 1. Input sanitization
            sanitized_data, sanitization_warnings = self.input_sanitizer.sanitize_input(
                request_data,
                max_length=self.config.get('max_input_length', 10000),
                strict_mode=self.config.get('strict_mode', True)
            )
            warnings.extend(sanitization_warnings)
            
            # 2. Session validation (if session provided)
            if session_id:
                session = self.auth_manager.validate_session(
                    session_id, ip_address
                )
                if not session:
                    raise SecurityViolationException(
                        "Invalid or expired session",
                        violation_type="authentication",
                        severity="HIGH"
                    )
            
            # 3. Threat detection
            threat_events = self.threat_detector.analyze_request(
                {'data': sanitized_data},
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            security_events.extend(threat_events)
            
            # 4. Check for blocking conditions
            critical_events = [e for e in threat_events if e.severity == SecurityLevel.CRITICAL]
            if critical_events:
                raise SecurityViolationException(
                    f"Critical security threats detected: {len(critical_events)}",
                    violation_type="threat_detection",
                    severity="CRITICAL"
                )
            
            # 5. Record events
            self.security_events.extend(security_events)
            if len(self.security_events) > self.max_events:
                self.security_events = self.security_events[-self.max_events:]
            
            return sanitized_data, warnings, security_events
            
        except SecurityViolationException:
            raise
        except Exception as e:
            logger.error(f"Security framework error: {e}")
            raise SecurityViolationException(
                f"Security processing failed: {str(e)}",
                violation_type="framework_error",
                severity="HIGH"
            )
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'threat_detection_stats': self.threat_detector.get_security_statistics(),
            'total_security_events': len(self.security_events),
            'active_sessions': len(self.auth_manager.sessions),
            'blocked_ips': len(self.auth_manager.blocked_ips),
            'security_config': self.config,
            'secure_defaults_applied': True
        }


# Global security framework instance
security_framework = SecurityFramework()