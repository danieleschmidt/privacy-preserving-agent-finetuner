"""Enhanced logging configuration for privacy-preserving training."""

import logging
import logging.handlers
import sys
import os
import json
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid
import traceback
from contextlib import contextmanager

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class PrivacyAwareFormatter(logging.Formatter):
    """Enhanced formatter with comprehensive privacy-aware redaction and correlation."""
    
    def __init__(self, *args, enable_privacy_redaction=True, custom_patterns=None, 
                 enable_correlation=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_privacy_redaction = enable_privacy_redaction
        self.custom_patterns = custom_patterns or []
        self.enable_correlation = enable_correlation
        self.correlation_id = None
        
    SENSITIVE_PATTERNS = [
        r'password[\s=:]+[^\s]+',
        r'secret[\s=:]+[^\s]+',
        r'key[\s=:]+[^\s]+',
        r'token[\s=:]+[^\s]+',
        r'auth[\s=:]+[^\s]+',
        r'credential[\s=:]+[^\s]+',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email pattern
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card pattern
        r'api[_-]?key[\s=:]+[^\s]+',
    ]
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracing."""
        self.correlation_id = correlation_id
    
    def format(self, record):
        """Enhanced format with privacy redaction and correlation."""
        # Create a copy to avoid modifying the original record
        record_copy = logging.LogRecord(
            record.name, record.levelno, record.pathname, record.lineno,
            record.getMessage(), record.args, record.exc_info, record.funcName,
            record.stack_info
        )
        
        # Add correlation ID if available
        if self.enable_correlation:
            correlation_id = getattr(record, 'correlation_id', None) or self.correlation_id
            if correlation_id:
                record_copy.correlation_id = correlation_id
        
        # Add thread and process information
        record_copy.thread_name = threading.current_thread().name
        record_copy.process_id = os.getpid()
        
        # Redact sensitive information from message
        if self.enable_privacy_redaction:
            message = record_copy.getMessage()
            message = self._redact_sensitive_data(message)
            record_copy.msg = message
            record_copy.args = None
        
        return super().format(record_copy)
    
    def _redact_sensitive_data(self, message: str) -> str:
        """Redact sensitive data from log messages."""
        import re
        
        redacted_message = message
        
        # Apply built-in patterns
        for pattern in self.SENSITIVE_PATTERNS:
            redacted_message = re.sub(pattern, '[REDACTED]', redacted_message, flags=re.IGNORECASE)
        
        # Apply custom patterns
        for pattern in self.custom_patterns:
            redacted_message = re.sub(pattern, '[REDACTED]', redacted_message, flags=re.IGNORECASE)
        
        return redacted_message


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
    """Enhanced JSON formatter for structured logging with comprehensive metadata."""
    
    def __init__(self, include_system_info=True, include_performance_metrics=True):
        super().__init__()
        self.include_system_info = include_system_info
        self.include_performance_metrics = include_performance_metrics
        self.hostname = os.getenv('HOSTNAME', 'unknown')
        self.application = 'privacy-finetuner'
    
    def format(self, record):
        """Format log record as comprehensive JSON."""
        # Base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_name': getattr(record, 'thread_name', threading.current_thread().name),
            'process_id': getattr(record, 'process_id', os.getpid()),
            'hostname': self.hostname,
            'application': self.application
        }
        
        # Add correlation ID for request tracing
        if hasattr(record, 'correlation_id'):
            log_entry['correlation_id'] = record.correlation_id
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add stack trace if available
        if record.stack_info:
            log_entry['stack_trace'] = record.stack_info
        
        # Add privacy-specific fields
        privacy_fields = ['privacy_budget', 'epsilon_spent', 'delta_value', 'privacy_risk_level']
        for field in privacy_fields:
            if hasattr(record, field):
                log_entry['privacy'][field] = getattr(record, field)
        
        # Add training-specific fields
        training_fields = ['training_job_id', 'epoch', 'step', 'loss', 'accuracy', 'model_name']
        for field in training_fields:
            if hasattr(record, field):
                if 'training' not in log_entry:
                    log_entry['training'] = {}
                log_entry['training'][field] = getattr(record, field)
        
        # Add security-specific fields
        security_fields = ['user_id', 'session_id', 'ip_address', 'user_agent', 'security_event_type']
        for field in security_fields:
            if hasattr(record, field):
                if 'security' not in log_entry:
                    log_entry['security'] = {}
                log_entry['security'][field] = getattr(record, field)
        
        # Add system information
        if self.include_system_info:
            system_info = self._get_system_info()
            if system_info:
                log_entry['system'] = system_info
        
        # Add performance metrics
        if self.include_performance_metrics:
            perf_metrics = self._get_performance_metrics()
            if perf_metrics:
                log_entry['performance'] = perf_metrics
        
        # Add custom fields from record
        for key, value in record.__dict__.items():
            if key.startswith('custom_') and key not in log_entry:
                log_entry[key] = value
        
        # Add error context if this is an error/warning
        if record.levelno >= logging.WARNING:
            log_entry['error_context'] = {
                'severity': record.levelname,
                'filename': record.filename,
                'pathname': record.pathname,
                'created': record.created
            }
        
        return json.dumps(log_entry, default=self._json_serializer)
    
    def _get_system_info(self) -> Optional[Dict[str, Any]]:
        """Get current system information."""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            return {
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_percent': psutil.cpu_percent(),
                'disk_usage_percent': psutil.disk_usage('/').percent
            }
        except:
            return None
    
    def _get_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """Get performance metrics."""
        try:
            import time
            return {
                'timestamp_ns': time.time_ns(),
                'uptime_seconds': time.time() - psutil.boot_time() if PSUTIL_AVAILABLE else None
            }
        except:
            return None
    
    def _json_serializer(self, obj):
        """JSON serializer for non-standard types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

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

class CorrelationContextManager:
    """Context manager for correlation ID tracking."""
    
    def __init__(self, correlation_id: str = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.old_correlation_id = None
    
    def __enter__(self):
        # Store old correlation ID if any
        self.old_correlation_id = getattr(threading.current_thread(), 'correlation_id', None)
        # Set new correlation ID
        threading.current_thread().correlation_id = self.correlation_id
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old correlation ID
        if self.old_correlation_id:
            threading.current_thread().correlation_id = self.old_correlation_id
        else:
            delattr(threading.current_thread(), 'correlation_id')


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler for high-performance logging."""
    
    def __init__(self, target_handler, queue_size=10000):
        super().__init__()
        self.target_handler = target_handler
        self.log_queue = queue.Queue(maxsize=queue_size)
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.start_worker()
    
    def start_worker(self):
        """Start the worker thread."""
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def _worker(self):
        """Worker thread that processes log records."""
        while not self.stop_event.is_set():
            try:
                record = self.log_queue.get(timeout=1.0)
                if record is None:  # Sentinel to stop
                    break
                self.target_handler.emit(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Log worker error: {e}", file=sys.stderr)
    
    def emit(self, record):
        """Emit a log record asynchronously."""
        try:
            self.log_queue.put_nowait(record)
        except queue.Full:
            # Drop the record if queue is full
            print("Log queue full, dropping record", file=sys.stderr)
    
    def stop(self):
        """Stop the async log handler."""
        self.stop_event.set()
        self.log_queue.put(None)  # Sentinel
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)


class MetricsLogHandler(logging.Handler):
    """Log handler that extracts metrics from log records."""
    
    def __init__(self, metrics_collector=None):
        super().__init__()
        self.metrics_collector = metrics_collector
        self.log_counts = {'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0}
    
    def emit(self, record):
        """Extract metrics from log record."""
        try:
            # Count log levels
            self.log_counts[record.levelname] += 1
            
            # Extract custom metrics if present
            if hasattr(record, 'metric_name') and hasattr(record, 'metric_value'):
                if self.metrics_collector:
                    self.metrics_collector.record_metric(
                        record.metric_name,
                        record.metric_value,
                        getattr(record, 'metric_labels', {})
                    )
            
            # Record error rates
            if record.levelno >= logging.ERROR:
                if self.metrics_collector:
                    self.metrics_collector.increment_counter(
                        'logs.errors.total',
                        {'level': record.levelname, 'logger': record.name}
                    )
                    
        except Exception as e:
            print(f"Metrics extraction error: {e}", file=sys.stderr)
    
    def get_log_counts(self):
        """Get current log counts by level."""
        return self.log_counts.copy()


class AuditLogger:
    """Enhanced audit logger for compliance and privacy events with correlation tracking."""
    
    def __init__(self):
        self.logger = logging.getLogger('privacy_finetuner.audit')
        self.audit_events = []
        self.max_events = 10000
        self.lock = threading.RLock()
    
    def log_privacy_event(
        self,
        event_type: str,
        privacy_cost: Dict[str, float],
        user_context: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        """Log privacy-affecting events for audit trail with correlation."""
        correlation_id = correlation_id or getattr(threading.current_thread(), 'correlation_id', str(uuid.uuid4()))
        
        audit_entry = {
            'event_id': str(uuid.uuid4()),
            'event_type': 'privacy_event',
            'privacy_event_type': event_type,
            'privacy_cost': privacy_cost,
            'user_context': user_context,
            'details': details or {},
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'correlation_id': correlation_id
        }
        
        # Store audit event
        with self.lock:
            self.audit_events.append(audit_entry)
            if len(self.audit_events) > self.max_events:
                self.audit_events.pop(0)
        
        self.logger.info(
            f"Privacy event: {event_type}",
            extra={
                'audit_entry': audit_entry,
                'privacy_budget': privacy_cost.get('epsilon', 0),
                'correlation_id': correlation_id,
                'privacy_risk_level': self._assess_privacy_risk(privacy_cost)
            }
        )
    
    def _assess_privacy_risk(self, privacy_cost: Dict[str, float]) -> str:
        """Assess privacy risk level based on cost."""
        epsilon = privacy_cost.get('epsilon', 0)
        if epsilon > 5.0:
            return 'high'
        elif epsilon > 1.0:
            return 'medium'
        else:
            return 'low'
    
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

class LogAggregator:
    """Aggregates and analyzes log patterns."""
    
    def __init__(self, window_size_minutes: int = 60):
        self.window_size_minutes = window_size_minutes
        self.log_buffer = []
        self.patterns = {}
        self.lock = threading.RLock()
    
    def add_log_record(self, record: logging.LogRecord):
        """Add log record for pattern analysis."""
        with self.lock:
            timestamp = datetime.now()
            
            # Clean old records
            cutoff_time = timestamp - timedelta(minutes=self.window_size_minutes)
            self.log_buffer = [r for r in self.log_buffer if r['timestamp'] > cutoff_time]
            
            # Add new record
            self.log_buffer.append({
                'timestamp': timestamp,
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'function': record.funcName,
                'line': record.lineno
            })
    
    def get_error_patterns(self) -> Dict[str, Any]:
        """Get error patterns and anomalies."""
        with self.lock:
            error_records = [r for r in self.log_buffer if r['level'] in ['ERROR', 'CRITICAL']]
            
            # Group by logger and function
            patterns = {}
            for record in error_records:
                key = f"{record['logger']}.{record['function']}"
                if key not in patterns:
                    patterns[key] = {'count': 0, 'messages': set()}
                patterns[key]['count'] += 1
                patterns[key]['messages'].add(record['message'][:100])  # Truncate for grouping
            
            # Convert sets to lists for JSON serialization
            for pattern in patterns.values():
                pattern['messages'] = list(pattern['messages'])
            
            return patterns
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get log statistics for the current window."""
        with self.lock:
            level_counts = {}
            logger_counts = {}
            
            for record in self.log_buffer:
                level = record['level']
                logger_name = record['logger']
                
                level_counts[level] = level_counts.get(level, 0) + 1
                logger_counts[logger_name] = logger_counts.get(logger_name, 0) + 1
            
            return {
                'total_records': len(self.log_buffer),
                'window_size_minutes': self.window_size_minutes,
                'level_distribution': level_counts,
                'logger_distribution': logger_counts,
                'error_rate': (level_counts.get('ERROR', 0) + level_counts.get('CRITICAL', 0)) / max(len(self.log_buffer), 1)
            }


# Enhanced global instances
audit_logger = AuditLogger()
performance_monitor = PerformanceMonitor()
log_aggregator = LogAggregator()


@contextmanager
def correlation_context(correlation_id: str = None):
    """Context manager for correlation ID tracking."""
    with CorrelationContextManager(correlation_id) as cid:
        yield cid


def log_with_correlation(logger, level, message, correlation_id=None, **extra):
    """Log message with correlation ID."""
    correlation_id = correlation_id or getattr(threading.current_thread(), 'correlation_id', None)
    if correlation_id:
        extra['correlation_id'] = correlation_id
    
    logger.log(level, message, extra=extra)

def setup_enhanced_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    structured_logging: bool = True,
    privacy_redaction: bool = True,
    async_logging: bool = False,
    enable_metrics: bool = True,
    correlation_enabled: bool = True
) -> None:
    """Setup enhanced logging with all advanced features."""
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Choose formatter
    if structured_logging:
        formatter = StructuredJsonFormatter(
            include_system_info=True,
            include_performance_metrics=True
        )
    else:
        formatter = PrivacyAwareFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s' if correlation_enabled 
            else '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            enable_privacy_redaction=privacy_redaction,
            enable_correlation=correlation_enabled
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Wrap in async handler if requested
    if async_logging:
        console_handler = AsyncLogHandler(console_handler)
    
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10
        )
        file_handler.setFormatter(formatter)
        
        if async_logging:
            file_handler = AsyncLogHandler(file_handler)
        
        root_logger.addHandler(file_handler)
    
    # Add metrics handler if enabled
    if enable_metrics:
        metrics_handler = MetricsLogHandler()
        root_logger.addHandler(metrics_handler)
    
    # Setup specialized loggers with enhanced configuration
    _setup_specialized_loggers(log_file, formatter, async_logging)
    
    root_logger.info(f"Enhanced logging configured: level={log_level}, structured={structured_logging}, async={async_logging}")


def _setup_specialized_loggers(log_file: Optional[str], formatter, async_logging: bool):
    """Setup specialized loggers for different components."""
    log_path = Path(log_file).parent if log_file else Path('./logs')
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Enhanced audit logger
    audit_logger_instance = logging.getLogger('privacy_finetuner.audit')
    audit_file = log_path / 'audit.log'
    audit_handler = logging.handlers.RotatingFileHandler(
        audit_file, maxBytes=200 * 1024 * 1024, backupCount=20
    )
    audit_handler.setFormatter(formatter)
    
    if async_logging:
        audit_handler = AsyncLogHandler(audit_handler)
    
    audit_logger_instance.addHandler(audit_handler)
    audit_logger_instance.setLevel(logging.INFO)
    
    # Enhanced security logger
    security_logger_instance = logging.getLogger('privacy_finetuner.security')
    security_file = log_path / 'security.log'
    security_handler = logging.handlers.RotatingFileHandler(
        security_file, maxBytes=100 * 1024 * 1024, backupCount=15
    )
    security_handler.setFormatter(formatter)
    
    if async_logging:
        security_handler = AsyncLogHandler(security_handler)
    
    security_logger_instance.addHandler(security_handler)
    security_logger_instance.setLevel(logging.WARNING)
    
    # Performance logger
    performance_logger_instance = logging.getLogger('privacy_finetuner.performance')
    performance_file = log_path / 'performance.log'
    performance_handler = logging.handlers.RotatingFileHandler(
        performance_file, maxBytes=100 * 1024 * 1024, backupCount=10
    )
    performance_handler.setFormatter(formatter)
    
    if async_logging:
        performance_handler = AsyncLogHandler(performance_handler)
    
    performance_logger_instance.addHandler(performance_handler)
    performance_logger_instance.setLevel(logging.DEBUG)


# Configure enhanced logging on module import if not already configured
if not logging.getLogger().handlers:
    setup_enhanced_logging(
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        log_file=os.getenv('LOG_FILE'),
        structured_logging=os.getenv('STRUCTURED_LOGGING', 'true').lower() == 'true',
        privacy_redaction=os.getenv('PRIVACY_REDACTION', 'true').lower() == 'true',
        async_logging=os.getenv('ASYNC_LOGGING', 'false').lower() == 'true',
        enable_metrics=os.getenv('ENABLE_LOG_METRICS', 'true').lower() == 'true',
        correlation_enabled=os.getenv('CORRELATION_ENABLED', 'true').lower() == 'true'
    )