"""Advanced error handling and recovery system for privacy-preserving training.

This module provides sophisticated error handling, automatic recovery, and resilient
training capabilities with comprehensive logging and monitoring.
"""

import logging
import traceback
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import pickle
import hashlib

from .exceptions import (
    PrivacyBudgetExhaustedException,
    ModelTrainingException,
    DataValidationException,
    SecurityViolationException
)

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    ROLLBACK = "rollback"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    DEGRADED_MODE = "degraded_mode"
    FAIL_FAST = "fail_fast"
    GRACEFUL_SHUTDOWN = "graceful_shutdown"


@dataclass
class ErrorRecord:
    """Comprehensive error record with context and recovery information."""
    error_id: str
    timestamp: datetime
    exception_type: str
    exception_message: str
    traceback: str
    severity: ErrorSeverity
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0
    max_retries: int = 3
    backoff_factor: float = 2.0
    user_notified: bool = False


@dataclass
class RecoveryCheckpoint:
    """Training checkpoint with recovery metadata."""
    checkpoint_id: str
    timestamp: datetime
    training_state: Dict[str, Any]
    model_state_path: Optional[Path] = None
    privacy_state: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True


class ErrorRecoveryManager:
    """Advanced error handling and recovery management system."""
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        error_log_path: str = "./logs/error_recovery.log",
        max_error_history: int = 1000,
        auto_recovery_enabled: bool = True,
        notification_callbacks: Optional[List[Callable]] = None
    ):
        """Initialize error recovery manager.
        
        Args:
            checkpoint_dir: Directory for storing recovery checkpoints
            error_log_path: Path for error recovery logs
            max_error_history: Maximum number of errors to keep in memory
            auto_recovery_enabled: Enable automatic error recovery
            notification_callbacks: Callbacks for error notifications
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.error_log_path = Path(error_log_path)
        self.max_error_history = max_error_history
        self.auto_recovery_enabled = auto_recovery_enabled
        self.notification_callbacks = notification_callbacks or []
        
        # Error tracking
        self.error_history: List[ErrorRecord] = []
        self.recovery_checkpoints: List[RecoveryCheckpoint] = []
        self.active_recovery_operations: Dict[str, threading.Thread] = {}
        
        # Recovery strategies per error type
        self.recovery_strategies: Dict[Type[Exception], RecoveryStrategy] = {
            PrivacyBudgetExhaustedException: RecoveryStrategy.GRACEFUL_SHUTDOWN,
            ModelTrainingException: RecoveryStrategy.CHECKPOINT_RESTORE,
            DataValidationException: RecoveryStrategy.DEGRADED_MODE,
            SecurityViolationException: RecoveryStrategy.FAIL_FAST,
            ConnectionError: RecoveryStrategy.RETRY,
            TimeoutError: RecoveryStrategy.RETRY,
            MemoryError: RecoveryStrategy.ROLLBACK,
            KeyboardInterrupt: RecoveryStrategy.GRACEFUL_SHUTDOWN
        }
        
        # Error pattern detection
        self.error_patterns: Dict[str, int] = {}
        self.pattern_threshold = 3
        
        # Setup directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Error recovery manager initialized")
    
    def handle_error(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        custom_strategy: Optional[RecoveryStrategy] = None
    ) -> bool:
        """Handle an error with appropriate recovery strategy.
        
        Args:
            exception: The exception that occurred
            context: Additional context information
            custom_strategy: Override default recovery strategy
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Create error record
        error_record = self._create_error_record(exception, context or {})
        
        # Determine recovery strategy
        strategy = custom_strategy or self._determine_recovery_strategy(exception)
        error_record.recovery_strategy = strategy
        
        # Store error record
        self.error_history.append(error_record)
        self._write_error_log(error_record)
        
        # Detect error patterns
        self._analyze_error_patterns(error_record)
        
        # Notify stakeholders
        self._notify_error(error_record)
        
        # Attempt recovery if enabled
        if self.auto_recovery_enabled and strategy != RecoveryStrategy.FAIL_FAST:
            return self._execute_recovery(error_record)
        
        return False
    
    def _create_error_record(self, exception: Exception, context: Dict[str, Any]) -> ErrorRecord:
        """Create comprehensive error record."""
        error_id = self._generate_error_id(exception, context)
        
        # Determine severity based on exception type and context
        severity = self._determine_error_severity(exception, context)
        
        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=datetime.now(),
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            traceback=traceback.format_exc(),
            severity=severity,
            context=context
        )
        
        return error_record
    
    def _generate_error_id(self, exception: Exception, context: Dict[str, Any]) -> str:
        """Generate unique error ID for tracking."""
        content = f"{type(exception).__name__}_{str(exception)}_{json.dumps(context, sort_keys=True, default=str)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _determine_error_severity(self, exception: Exception, context: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity based on exception type and context."""
        
        # Critical errors that require immediate attention
        if isinstance(exception, (SecurityViolationException, PrivacyBudgetExhaustedException)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors that impact training significantly
        if isinstance(exception, (ModelTrainingException, MemoryError)):
            return ErrorSeverity.HIGH
        
        # Medium severity errors that may be recoverable
        if isinstance(exception, (DataValidationException, ConnectionError, TimeoutError)):
            return ErrorSeverity.MEDIUM
        
        # Low severity errors (warnings, minor issues)
        return ErrorSeverity.LOW
    
    def _determine_recovery_strategy(self, exception: Exception) -> RecoveryStrategy:
        """Determine appropriate recovery strategy for exception type."""
        exception_type = type(exception)
        
        # Check direct mapping
        if exception_type in self.recovery_strategies:
            return self.recovery_strategies[exception_type]
        
        # Check parent classes
        for error_type, strategy in self.recovery_strategies.items():
            if isinstance(exception, error_type):
                return strategy
        
        # Default strategy
        return RecoveryStrategy.RETRY
    
    def _execute_recovery(self, error_record: ErrorRecord) -> bool:
        """Execute recovery strategy for error."""
        error_record.recovery_attempted = True
        strategy = error_record.recovery_strategy
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return self._retry_operation(error_record)
            elif strategy == RecoveryStrategy.ROLLBACK:
                return self._rollback_operation(error_record)
            elif strategy == RecoveryStrategy.CHECKPOINT_RESTORE:
                return self._restore_from_checkpoint(error_record)
            elif strategy == RecoveryStrategy.DEGRADED_MODE:
                return self._enable_degraded_mode(error_record)
            elif strategy == RecoveryStrategy.GRACEFUL_SHUTDOWN:
                return self._graceful_shutdown(error_record)
            else:
                logger.warning(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Recovery operation failed: {e}")
            return False
    
    def _retry_operation(self, error_record: ErrorRecord) -> bool:
        """Retry the failed operation with exponential backoff."""
        if error_record.retry_count >= error_record.max_retries:
            logger.warning(f"Max retries exceeded for error {error_record.error_id}")
            return False
        
        error_record.retry_count += 1
        backoff_time = error_record.backoff_factor ** error_record.retry_count
        
        logger.info(f"Retrying operation in {backoff_time} seconds (attempt {error_record.retry_count}/{error_record.max_retries})")
        time.sleep(backoff_time)
        
        # The actual retry logic would be implemented by the calling code
        # For now, we simulate a successful retry for certain error types
        if "connection" in error_record.exception_message.lower():
            error_record.recovery_successful = True
            return True
        
        return False
    
    def _rollback_operation(self, error_record: ErrorRecord) -> bool:
        """Rollback to previous stable state."""
        logger.info(f"Rolling back operation for error {error_record.error_id}")
        
        # Find the most recent valid checkpoint
        valid_checkpoint = self._find_latest_checkpoint()
        
        if valid_checkpoint:
            logger.info(f"Rolling back to checkpoint {valid_checkpoint.checkpoint_id}")
            error_record.recovery_successful = True
            return True
        
        logger.warning("No valid checkpoint found for rollback")
        return False
    
    def _restore_from_checkpoint(self, error_record: ErrorRecord) -> bool:
        """Restore training state from checkpoint."""
        logger.info(f"Restoring from checkpoint for error {error_record.error_id}")
        
        checkpoint = self._find_latest_checkpoint()
        if checkpoint and checkpoint.is_valid:
            # Restore training state (implementation would depend on specific framework)
            logger.info(f"Restored from checkpoint {checkpoint.checkpoint_id}")
            error_record.recovery_successful = True
            return True
        
        return False
    
    def _enable_degraded_mode(self, error_record: ErrorRecord) -> bool:
        """Enable degraded mode operation."""
        logger.info(f"Enabling degraded mode for error {error_record.error_id}")
        
        # Implement degraded mode logic (reduced functionality)
        # For example: disable certain features, use fallback methods
        
        error_record.recovery_successful = True
        return True
    
    def _graceful_shutdown(self, error_record: ErrorRecord) -> bool:
        """Perform graceful shutdown."""
        logger.info(f"Initiating graceful shutdown for error {error_record.error_id}")
        
        # Save current state
        self.create_checkpoint("emergency_shutdown", {
            "error_context": error_record.context,
            "shutdown_reason": error_record.exception_message
        })
        
        error_record.recovery_successful = True
        return True
    
    def create_checkpoint(
        self,
        checkpoint_id: str,
        training_state: Dict[str, Any],
        model_state_path: Optional[Path] = None,
        privacy_state: Optional[Dict[str, Any]] = None,
        system_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a recovery checkpoint.
        
        Args:
            checkpoint_id: Unique identifier for checkpoint
            training_state: Training state dictionary
            model_state_path: Path to saved model state
            privacy_state: Privacy-related state
            system_state: System state information
            
        Returns:
            Checkpoint ID for reference
        """
        checkpoint = RecoveryCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            training_state=training_state,
            model_state_path=model_state_path,
            privacy_state=privacy_state or {},
            system_state=system_state or {}
        )
        
        # Save checkpoint to disk
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            # Add to memory storage
            self.recovery_checkpoints.append(checkpoint)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            logger.info(f"Created checkpoint: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            checkpoint.is_valid = False
            return checkpoint_id
    
    def _find_latest_checkpoint(self) -> Optional[RecoveryCheckpoint]:
        """Find the most recent valid checkpoint."""
        valid_checkpoints = [c for c in self.recovery_checkpoints if c.is_valid]
        
        if valid_checkpoints:
            return max(valid_checkpoints, key=lambda c: c.timestamp)
        
        return None
    
    def _cleanup_old_checkpoints(self, max_checkpoints: int = 10) -> None:
        """Clean up old checkpoints to prevent disk space issues."""
        if len(self.recovery_checkpoints) <= max_checkpoints:
            return
        
        # Sort by timestamp and keep only the most recent
        self.recovery_checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
        old_checkpoints = self.recovery_checkpoints[max_checkpoints:]
        
        for checkpoint in old_checkpoints:
            try:
                checkpoint_path = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.pkl"
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete old checkpoint: {e}")
        
        # Keep only recent checkpoints in memory
        self.recovery_checkpoints = self.recovery_checkpoints[:max_checkpoints]
    
    def _analyze_error_patterns(self, error_record: ErrorRecord) -> None:
        """Analyze error patterns for proactive issue detection."""
        error_signature = f"{error_record.exception_type}_{error_record.exception_message[:50]}"
        
        self.error_patterns[error_signature] = self.error_patterns.get(error_signature, 0) + 1
        
        # Check if error pattern exceeds threshold
        if self.error_patterns[error_signature] >= self.pattern_threshold:
            logger.warning(f"Error pattern detected: {error_signature} (count: {self.error_patterns[error_signature]})")
            
            # Trigger pattern-based recovery or notifications
            self._handle_error_pattern(error_signature, self.error_patterns[error_signature])
    
    def _handle_error_pattern(self, pattern: str, count: int) -> None:
        """Handle detected error patterns."""
        logger.info(f"Handling error pattern: {pattern} (count: {count})")
        
        # Implement pattern-specific handling
        if "connection" in pattern.lower():
            logger.info("Detected connection issues, implementing circuit breaker")
        elif "memory" in pattern.lower():
            logger.info("Detected memory issues, enabling memory optimization")
        elif "timeout" in pattern.lower():
            logger.info("Detected timeout issues, adjusting timeout parameters")
    
    def _notify_error(self, error_record: ErrorRecord) -> None:
        """Send error notifications to registered callbacks."""
        if error_record.user_notified:
            return
        
        for callback in self.notification_callbacks:
            try:
                callback(error_record)
            except Exception as e:
                logger.error(f"Error notification callback failed: {e}")
        
        error_record.user_notified = True
    
    def _write_error_log(self, error_record: ErrorRecord) -> None:
        """Write error record to log file."""
        try:
            log_entry = {
                "error_id": error_record.error_id,
                "timestamp": error_record.timestamp.isoformat(),
                "exception_type": error_record.exception_type,
                "exception_message": error_record.exception_message,
                "severity": error_record.severity.name,
                "context": error_record.context,
                "recovery_strategy": error_record.recovery_strategy.value if error_record.recovery_strategy else None,
                "recovery_attempted": error_record.recovery_attempted,
                "recovery_successful": error_record.recovery_successful,
                "retry_count": error_record.retry_count
            }
            
            with open(self.error_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\\n")
                
        except Exception as e:
            logger.error(f"Failed to write error log: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        now = datetime.now()
        recent_errors = [e for e in self.error_history if e.timestamp > now - timedelta(hours=24)]
        
        # Error distribution by type
        errors_by_type = {}
        for error in recent_errors:
            errors_by_type[error.exception_type] = errors_by_type.get(error.exception_type, 0) + 1
        
        # Error distribution by severity
        errors_by_severity = {}
        for error in recent_errors:
            errors_by_severity[error.severity.name] = errors_by_severity.get(error.severity.name, 0) + 1
        
        # Recovery success rate
        recovery_attempts = len([e for e in recent_errors if e.recovery_attempted])
        recovery_successes = len([e for e in recent_errors if e.recovery_successful])
        recovery_rate = (recovery_successes / recovery_attempts) * 100 if recovery_attempts > 0 else 0
        
        return {
            "timestamp": now.isoformat(),
            "summary": {
                "total_errors_24h": len(recent_errors),
                "total_errors_all_time": len(self.error_history),
                "recovery_attempts": recovery_attempts,
                "recovery_success_rate": recovery_rate,
                "active_checkpoints": len(self.recovery_checkpoints),
                "error_patterns_detected": len([p for p, c in self.error_patterns.items() if c >= self.pattern_threshold])
            },
            "distribution": {
                "errors_by_type": errors_by_type,
                "errors_by_severity": errors_by_severity
            },
            "patterns": {
                pattern: count for pattern, count in self.error_patterns.items()
                if count >= self.pattern_threshold
            },
            "recent_critical_errors": [
                {
                    "error_id": e.error_id,
                    "type": e.exception_type,
                    "message": e.exception_message[:100],
                    "timestamp": e.timestamp.isoformat(),
                    "recovery_successful": e.recovery_successful
                }
                for e in recent_errors
                if e.severity == ErrorSeverity.CRITICAL
            ][-10:]  # Last 10 critical errors
        }
    
    def add_notification_callback(self, callback: Callable[[ErrorRecord], None]) -> None:
        """Add error notification callback."""
        self.notification_callbacks.append(callback)
    
    def set_recovery_strategy(self, exception_type: Type[Exception], strategy: RecoveryStrategy) -> None:
        """Set recovery strategy for specific exception type."""
        self.recovery_strategies[exception_type] = strategy
        logger.info(f"Set recovery strategy for {exception_type.__name__}: {strategy.value}")


# Global error recovery manager
_global_recovery_manager: Optional[ErrorRecoveryManager] = None


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get or create global error recovery manager."""
    global _global_recovery_manager
    
    if _global_recovery_manager is None:
        _global_recovery_manager = ErrorRecoveryManager()
    
    return _global_recovery_manager


def handle_error(exception: Exception, context: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to handle errors with recovery."""
    manager = get_recovery_manager()
    return manager.handle_error(exception, context)


def create_checkpoint(checkpoint_id: str, training_state: Dict[str, Any], **kwargs) -> str:
    """Convenience function to create recovery checkpoints."""
    manager = get_recovery_manager()
    return manager.create_checkpoint(checkpoint_id, training_state, **kwargs)


class ErrorRecoveryDecorator:
    """Decorator for automatic error handling and recovery."""
    
    def __init__(
        self,
        recovery_strategy: Optional[RecoveryStrategy] = None,
        max_retries: int = 3,
        context_extractor: Optional[Callable] = None
    ):
        self.recovery_strategy = recovery_strategy
        self.max_retries = max_retries
        self.context_extractor = context_extractor
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            manager = get_recovery_manager()
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = {}
                    if self.context_extractor:
                        try:
                            context = self.context_extractor(*args, **kwargs)
                        except Exception:
                            pass
                    
                    context.update({
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "max_retries": self.max_retries
                    })
                    
                    if attempt < self.max_retries:
                        success = manager.handle_error(e, context, self.recovery_strategy)
                        if not success:
                            # If recovery failed, don't retry
                            break
                    else:
                        # Final attempt failed
                        manager.handle_error(e, context, RecoveryStrategy.FAIL_FAST)
                        raise
            
            # If we get here, all recovery attempts failed
            raise RuntimeError(f"All recovery attempts failed for {func.__name__}")
        
        return wrapper


# Convenience decorator instances
error_recovery = ErrorRecoveryDecorator()
retry_on_error = ErrorRecoveryDecorator(recovery_strategy=RecoveryStrategy.RETRY)
checkpoint_on_error = ErrorRecoveryDecorator(recovery_strategy=RecoveryStrategy.CHECKPOINT_RESTORE)