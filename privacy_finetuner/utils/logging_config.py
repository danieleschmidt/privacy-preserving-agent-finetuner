"""Enhanced logging configuration for privacy-preserving training."""

import logging
import logging.handlers
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class PrivacyAwareFormatter(logging.Formatter):
    """Enhanced formatter with comprehensive privacy-aware redaction."""
    
    def __init__(self, *args, enable_privacy_redaction=True, custom_patterns=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_privacy_redaction = enable_privacy_redaction
        self.custom_patterns = custom_patterns or []
        
    SENSITIVE_PATTERNS = [
        'password', 'secret', 'key', 'token', 'auth', 'credential',
        'email', 'phone', 'ssn', 'card', 'account', 'api_key'
    ]


class ColoredFormatter(PrivacyAwareFormatter):
    """Colored console formatter for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format with colors if terminal supports it."""
        formatted = super().format(record)
        if sys.stdout.isatty():  # Only colorize if output is a terminal
            color = self.COLORS.get(record.levelname, '')
            formatted = f"{color}{formatted}{self.RESET}"
        return formatted
    
    def format(self, record):
        """Format log record while redacting sensitive information."""
        # Create a copy to avoid modifying the original record
        record_copy = logging.LogRecord(
            record.name, record.levelno, record.pathname, record.lineno,
            record.getMessage(), record.args, record.exc_info, record.funcName,
            record.stack_info
        )
        
        # Redact sensitive information from message
        message = record_copy.getMessage()
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern.lower() in message.lower():
                # Simple redaction - replace with placeholder
                message = message.replace(pattern, '[REDACTED]')
        
        record_copy.msg = message
        record_copy.args = None
        
        return super().format(record_copy)

class StructuredJsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'privacy_budget'):
            log_entry['privacy_budget'] = record.privacy_budget
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        
        if hasattr(record, 'training_job_id'):
            log_entry['training_job_id'] = record.training_job_id
        
        return json.dumps(log_entry)

def setup_privacy_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    structured_logging: bool = True,
    privacy_redaction: bool = True
) -> None:
    """Setup comprehensive logging for privacy-preserving training.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        structured_logging: Use JSON structured logging
        privacy_redaction: Enable privacy-aware log redaction
    """
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Choose formatter
    if structured_logging:
        formatter = StructuredJsonFormatter()
    elif privacy_redaction:
        formatter = PrivacyAwareFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    privacy_logger = logging.getLogger('privacy_finetuner')
    privacy_logger.setLevel(logging.DEBUG)
    
    # Add audit logger for compliance
    audit_logger = logging.getLogger('privacy_finetuner.audit')
    if log_file:
        audit_file = str(log_path.parent / 'audit.log')
        audit_handler = logging.handlers.RotatingFileHandler(
            audit_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10
        )
        audit_handler.setFormatter(StructuredJsonFormatter())
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
    
    # Security event logger
    security_logger = logging.getLogger('privacy_finetuner.security')
    if log_file:
        security_file = str(log_path.parent / 'security.log')
        security_handler = logging.handlers.RotatingFileHandler(
            security_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        security_handler.setFormatter(StructuredJsonFormatter())
        security_logger.addHandler(security_handler)
        security_logger.setLevel(logging.WARNING)

class AuditLogger:
    """Specialized audit logger for compliance and privacy events."""
    
    def __init__(self):
        self.logger = logging.getLogger('privacy_finetuner.audit')
    
    def log_privacy_event(
        self,
        event_type: str,
        privacy_cost: Dict[str, float],
        user_context: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None
    ):
        """Log privacy-affecting events for audit trail."""
        audit_entry = {
            'event_type': 'privacy_event',
            'privacy_event_type': event_type,
            'privacy_cost': privacy_cost,
            'user_context': user_context,
            'details': details or {},
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        self.logger.info(
            f"Privacy event: {event_type}",
            extra={
                'audit_entry': audit_entry,
                'privacy_budget': privacy_cost.get('epsilon', 0)
            }
        )
    
    def log_training_event(
        self,
        event_type: str,
        job_id: str,
        model_name: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log training lifecycle events."""
        audit_entry = {
            'event_type': 'training_event',
            'training_event_type': event_type,
            'job_id': job_id,
            'model_name': model_name,
            'user_id': user_id,
            'details': details or {},
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        self.logger.info(
            f"Training event: {event_type}",
            extra={
                'audit_entry': audit_entry,
                'training_job_id': job_id,
                'user_id': user_id
            }
        )
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log security events and potential threats."""
        security_logger = logging.getLogger('privacy_finetuner.security')
        
        security_entry = {
            'event_type': 'security_event',
            'security_event_type': event_type,
            'severity': severity,
            'description': description,
            'source_ip': source_ip,
            'user_agent': user_agent,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        log_level = getattr(logging, severity.upper(), logging.WARNING)
        security_logger.log(
            log_level,
            f"Security event: {event_type} - {description}",
            extra={'security_entry': security_entry}
        )

class PerformanceMonitor:
    """Monitor performance metrics for training operations."""
    
    def __init__(self):
        self.logger = logging.getLogger('privacy_finetuner.performance')
        self.metrics = {}
    
    def start_timer(self, operation: str) -> str:
        """Start timing an operation."""
        import uuid
        timer_id = str(uuid.uuid4())
        self.metrics[timer_id] = {
            'operation': operation,
            'start_time': datetime.utcnow(),
            'start_timestamp': datetime.utcnow().timestamp()
        }
        return timer_id
    
    def end_timer(self, timer_id: str) -> Dict[str, Any]:
        """End timing and log performance metrics."""
        if timer_id not in self.metrics:
            self.logger.warning(f"Timer {timer_id} not found")
            return {}
        
        metric = self.metrics[timer_id]
        end_time = datetime.utcnow()
        duration = end_time - metric['start_time']
        
        performance_data = {
            'operation': metric['operation'],
            'duration_seconds': duration.total_seconds(),
            'start_time': metric['start_time'].isoformat() + 'Z',
            'end_time': end_time.isoformat() + 'Z'
        }
        
        self.logger.info(
            f"Performance: {metric['operation']} took {duration.total_seconds():.2f}s",
            extra={'performance_data': performance_data}
        )
        
        del self.metrics[timer_id]
        return performance_data

# Global instances
audit_logger = AuditLogger()
performance_monitor = PerformanceMonitor()

# Configure logging on module import if not already configured
if not logging.getLogger().handlers:
    setup_privacy_logging(
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        log_file=os.getenv('LOG_FILE'),
        structured_logging=os.getenv('STRUCTURED_LOGGING', 'true').lower() == 'true',
        privacy_redaction=os.getenv('PRIVACY_REDACTION', 'true').lower() == 'true'
    )