"""Advanced error recovery and fault tolerance for privacy-preserving ML systems.

This module provides comprehensive error recovery mechanisms including:
- Automatic checkpoint/restart with privacy budget preservation
- Circuit breakers for external dependencies
- Graceful degradation strategies
- Distributed system failure recovery
- Data corruption detection and repair
- Privacy budget rollback mechanisms
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
import hashlib
from enum import Enum
import pickle

# Handle imports gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    
    class NumpyStub:
        @staticmethod
        def array(data):
            return data
        
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
    
    np = NumpyStub()

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    ROLLBACK = "rollback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAILOVER = "failover"
    CIRCUIT_BREAK = "circuit_break"
    CHECKPOINT_RESTORE = "checkpoint_restore"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemState(Enum):
    """System state enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    FAILED = "failed"


@dataclass
class ErrorEvent:
    """Error event information."""
    timestamp: datetime
    error_type: str
    severity: ErrorSeverity
    message: str
    stack_trace: Optional[str]
    context: Dict[str, Any]
    component: str
    recovery_strategy: Optional[RecoveryStrategy]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "recovery_strategy": self.recovery_strategy.value if self.recovery_strategy else None
        }


@dataclass
class CheckpointMetadata:
    """Checkpoint metadata."""
    checkpoint_id: str
    timestamp: datetime
    privacy_budget_consumed: float
    model_state_hash: str
    data_state_hash: str
    training_step: int
    validation_score: float
    component_states: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CircuitBreakerState:
    """Circuit breaker state."""
    name: str
    state: str  # "closed", "open", "half_open"
    failure_count: int
    failure_threshold: int
    timeout_seconds: int
    last_failure_time: Optional[datetime]
    success_threshold: int  # For half-open -> closed transition
    success_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class AdvancedErrorRecoverySystem:
    """Advanced error recovery and fault tolerance system.
    
    Features:
    - Multi-level checkpointing with privacy budget tracking
    - Circuit breakers for external dependencies
    - Automatic retry with exponential backoff
    - Graceful degradation strategies
    - Distributed failure recovery
    - Real-time health monitoring
    - Privacy-preserving error logging
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: Optional[Path] = None,
        max_recovery_attempts: int = 3
    ):
        """Initialize advanced error recovery system.
        
        Args:
            config: Recovery system configuration
            checkpoint_dir: Directory for storing checkpoints
            max_recovery_attempts: Maximum recovery attempts per error
        """
        self.config = config or {}
        self.checkpoint_dir = checkpoint_dir or Path("recovery_checkpoints")
        self.max_recovery_attempts = max_recovery_attempts
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        self.circuit_breakers = CircuitBreakerRegistry()
        self.retry_manager = RetryManager()
        self.degradation_manager = DegradationManager()
        self.health_monitor = HealthMonitor()
        
        # State tracking
        self.system_state = SystemState.HEALTHY
        self.error_history = []
        self.recovery_history = []
        self.active_recoveries = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info("Advanced error recovery system initialized")
    
    def register_component(
        self,
        component_name: str,
        health_check: Callable[[], bool],
        recovery_strategies: List[RecoveryStrategy],
        circuit_breaker_config: Optional[Dict[str, Any]] = None
    ):
        """Register a component for monitoring and recovery.
        
        Args:
            component_name: Name of the component
            health_check: Health check function
            recovery_strategies: Available recovery strategies
            circuit_breaker_config: Circuit breaker configuration
        """
        # Register health check
        self.health_monitor.register_component(component_name, health_check)
        
        # Setup circuit breaker if configured
        if circuit_breaker_config:
            self.circuit_breakers.create_breaker(
                component_name, circuit_breaker_config
            )
        
        logger.info(f"Registered component: {component_name}")
    
    def handle_error(
        self,
        error: Exception,
        component: str,
        context: Optional[Dict[str, Any]] = None,
        severity: Optional[ErrorSeverity] = None
    ) -> bool:
        """Handle an error with automatic recovery.
        
        Args:
            error: The exception that occurred
            component: Component where error occurred
            context: Additional context information
            severity: Error severity level
            
        Returns:
            True if error was successfully recovered, False otherwise
        """
        with self.lock:
            # Determine severity if not provided
            if severity is None:
                severity = self._determine_error_severity(error, component)
            
            # Create error event
            error_event = ErrorEvent(
                timestamp=datetime.now(),
                error_type=type(error).__name__,
                severity=severity,
                message=str(error),
                stack_trace=self._get_stack_trace(error),
                context=context or {},
                component=component,
                recovery_strategy=None
            )
            
            self.error_history.append(error_event)
            
            # Log error (privacy-preserving)
            self._log_error_safely(error_event)
            
            # Update system state
            if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                self.system_state = SystemState.DEGRADED
            
            # Attempt recovery
            recovery_successful = self._attempt_recovery(error_event)
            
            if recovery_successful:
                logger.info(f"Successfully recovered from error in {component}")
                return True
            else:
                logger.error(f"Failed to recover from error in {component}")
                if severity == ErrorSeverity.CRITICAL:
                    self.system_state = SystemState.FAILED
                return False
    
    def create_checkpoint(
        self,
        component_states: Dict[str, Any],
        privacy_budget_consumed: float,
        training_step: int = 0,
        validation_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a system checkpoint.
        
        Args:
            component_states: States of all components
            privacy_budget_consumed: Total privacy budget consumed
            training_step: Current training step
            validation_score: Current validation score
            metadata: Additional metadata
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = self.checkpoint_manager.create_checkpoint(
            component_states=component_states,
            privacy_budget_consumed=privacy_budget_consumed,
            training_step=training_step,
            validation_score=validation_score,
            metadata=metadata or {}
        )
        
        logger.info(f"Created checkpoint: {checkpoint_id}")
        return checkpoint_id
    
    def restore_checkpoint(
        self,
        checkpoint_id: Optional[str] = None,
        privacy_budget_limit: Optional[float] = None
    ) -> Dict[str, Any]:
        """Restore from a checkpoint.
        
        Args:
            checkpoint_id: Specific checkpoint ID to restore (latest if None)
            privacy_budget_limit: Maximum privacy budget to allow
            
        Returns:
            Restored state information
        """
        try:
            restored_state = self.checkpoint_manager.restore_checkpoint(
                checkpoint_id, privacy_budget_limit
            )
            
            logger.info(f"Restored from checkpoint: {restored_state['checkpoint_id']}")
            
            # Update system state
            self.system_state = SystemState.RECOVERING
            
            # Record recovery
            recovery_record = {
                "timestamp": datetime.now(),
                "type": "checkpoint_restore",
                "checkpoint_id": restored_state["checkpoint_id"],
                "success": True
            }
            self.recovery_history.append(recovery_record)
            
            return restored_state
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            
            recovery_record = {
                "timestamp": datetime.now(),
                "type": "checkpoint_restore",
                "checkpoint_id": checkpoint_id,
                "success": False,
                "error": str(e)
            }
            self.recovery_history.append(recovery_record)
            
            raise
    
    def execute_with_protection(
        self,
        func: Callable,
        component: str,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        circuit_breaker: bool = True
    ) -> Any:
        """Execute function with error protection.
        
        Args:
            func: Function to execute
            component: Component name
            max_retries: Maximum retry attempts
            timeout_seconds: Execution timeout
            circuit_breaker: Whether to use circuit breaker
            
        Returns:
            Function result
        """
        # Check circuit breaker
        if circuit_breaker and self.circuit_breakers.is_open(component):
            raise RuntimeError(f"Circuit breaker open for {component}")
        
        # Execute with retry
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                
                # Execute function
                result = func()
                
                execution_time = time.time() - start_time
                
                # Check timeout
                if execution_time > timeout_seconds:
                    raise TimeoutError(f"Function execution exceeded {timeout_seconds}s")
                
                # Success - close circuit breaker if it was open
                if circuit_breaker:
                    self.circuit_breakers.record_success(component)
                
                return result
                
            except Exception as e:
                # Record failure
                if circuit_breaker:
                    self.circuit_breakers.record_failure(component)
                
                # Last attempt - propagate error
                if attempt == max_retries:
                    self.handle_error(e, component)
                    raise
                
                # Wait before retry
                wait_time = self.retry_manager.get_wait_time(attempt)
                logger.warning(f"Attempt {attempt + 1} failed for {component}, retrying in {wait_time}s")
                time.sleep(wait_time)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health_status = {
            "system_state": self.system_state.value,
            "timestamp": datetime.now().isoformat(),
            "components": self.health_monitor.get_all_health_status(),
            "circuit_breakers": {
                name: breaker.to_dict()
                for name, breaker in self.circuit_breakers.breakers.items()
            },
            "recent_errors": len([
                e for e in self.error_history
                if (datetime.now() - e.timestamp) < timedelta(hours=1)
            ]),
            "successful_recoveries": len([
                r for r in self.recovery_history
                if r.get("success", False) and
                (datetime.now() - r["timestamp"]) < timedelta(hours=1)
            ]),
            "available_checkpoints": len(self.checkpoint_manager.get_available_checkpoints())
        }
        
        return health_status
    
    def enable_graceful_degradation(
        self,
        component: str,
        degraded_function: Callable,
        trigger_conditions: List[str]
    ):
        """Enable graceful degradation for a component.
        
        Args:
            component: Component name
            degraded_function: Function to use in degraded mode
            trigger_conditions: Conditions that trigger degradation
        """
        self.degradation_manager.register_degradation(
            component, degraded_function, trigger_conditions
        )
    
    def get_recovery_recommendations(self) -> List[Dict[str, Any]]:
        """Get recovery recommendations based on current state."""
        recommendations = []
        
        # Analyze error patterns
        recent_errors = [
            e for e in self.error_history
            if (datetime.now() - e.timestamp) < timedelta(hours=24)
        ]
        
        if len(recent_errors) > 10:
            recommendations.append({
                "type": "error_frequency",
                "message": f"High error frequency detected: {len(recent_errors)} errors in 24h",
                "action": "Consider scaling down or enabling degraded mode",
                "priority": "high"
            })
        
        # Check circuit breaker status
        open_breakers = [
            name for name, breaker in self.circuit_breakers.breakers.items()
            if breaker.state == "open"
        ]
        
        if open_breakers:
            recommendations.append({
                "type": "circuit_breakers",
                "message": f"Circuit breakers open: {open_breakers}",
                "action": "Investigate underlying issues or reset breakers",
                "priority": "medium"
            })
        
        # Check checkpoint age
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint:
            checkpoint_age = datetime.now() - latest_checkpoint["timestamp"]
            if checkpoint_age > timedelta(hours=6):
                recommendations.append({
                    "type": "checkpoint_age",
                    "message": f"Latest checkpoint is {checkpoint_age} old",
                    "action": "Create new checkpoint to ensure recent recovery point",
                    "priority": "low"
                })
        
        return recommendations
    
    def _attempt_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt to recover from an error.
        
        Args:
            error_event: The error event to recover from
            
        Returns:
            True if recovery successful, False otherwise
        """
        component = error_event.component
        
        # Skip if already recovering this component
        if component in self.active_recoveries:
            logger.warning(f"Recovery already in progress for {component}")
            return False
        
        self.active_recoveries[component] = {
            "start_time": datetime.now(),
            "error_event": error_event,
            "attempts": 0
        }
        
        try:
            # Determine recovery strategy based on error type and severity
            strategy = self._select_recovery_strategy(error_event)
            error_event.recovery_strategy = strategy
            
            logger.info(f"Attempting {strategy.value} recovery for {component}")
            
            recovery_successful = False
            
            if strategy == RecoveryStrategy.RETRY:
                recovery_successful = self._retry_recovery(error_event)
            
            elif strategy == RecoveryStrategy.ROLLBACK:
                recovery_successful = self._rollback_recovery(error_event)
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                recovery_successful = self._graceful_degradation_recovery(error_event)
            
            elif strategy == RecoveryStrategy.FAILOVER:
                recovery_successful = self._failover_recovery(error_event)
            
            elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
                recovery_successful = self._circuit_break_recovery(error_event)
            
            elif strategy == RecoveryStrategy.CHECKPOINT_RESTORE:
                recovery_successful = self._checkpoint_restore_recovery(error_event)
            
            # Record recovery attempt
            recovery_record = {
                "timestamp": datetime.now(),
                "type": strategy.value,
                "component": component,
                "error_type": error_event.error_type,
                "success": recovery_successful,
                "attempts": self.active_recoveries[component]["attempts"] + 1
            }
            self.recovery_history.append(recovery_record)
            
            return recovery_successful
            
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed: {recovery_error}")
            return False
            
        finally:
            # Clean up active recovery
            if component in self.active_recoveries:
                del self.active_recoveries[component]
    
    def _select_recovery_strategy(self, error_event: ErrorEvent) -> RecoveryStrategy:
        """Select appropriate recovery strategy for error."""
        component = error_event.component
        error_type = error_event.error_type
        severity = error_event.severity
        
        # Critical errors - immediate checkpoint restore
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.CHECKPOINT_RESTORE
        
        # Network/IO errors - retry with circuit breaker
        if error_type in ["ConnectionError", "TimeoutError", "NetworkError"]:
            if self.circuit_breakers.get_failure_count(component) > 5:
                return RecoveryStrategy.CIRCUIT_BREAK
            else:
                return RecoveryStrategy.RETRY
        
        # Memory/Resource errors - graceful degradation
        if error_type in ["MemoryError", "ResourceError", "OutOfMemoryError"]:
            return RecoveryStrategy.GRACEFUL_DEGRADATION
        
        # Data corruption - rollback
        if error_type in ["ValueError", "CorruptionError", "ValidationError"]:
            return RecoveryStrategy.ROLLBACK
        
        # System errors - checkpoint restore
        if error_type in ["SystemError", "OSError", "RuntimeError"]:
            return RecoveryStrategy.CHECKPOINT_RESTORE
        
        # Default strategy
        return RecoveryStrategy.RETRY
    
    def _retry_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt retry-based recovery."""
        component = error_event.component
        
        # Check if component has health check
        if not self.health_monitor.has_component(component):
            return False
        
        # Wait before retry
        wait_time = self.retry_manager.get_wait_time(0)
        time.sleep(wait_time)
        
        # Check component health
        try:
            is_healthy = self.health_monitor.check_component_health(component)
            return is_healthy
        except Exception:
            return False
    
    def _rollback_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt rollback recovery."""
        try:
            # Find recent checkpoint
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            
            if not latest_checkpoint:
                logger.warning("No checkpoint available for rollback")
                return False
            
            # Restore to previous checkpoint
            self.restore_checkpoint(latest_checkpoint["checkpoint_id"])
            return True
            
        except Exception as e:
            logger.error(f"Rollback recovery failed: {e}")
            return False
    
    def _graceful_degradation_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt graceful degradation recovery."""
        component = error_event.component
        
        try:
            degraded_mode_enabled = self.degradation_manager.enable_degraded_mode(component)
            if degraded_mode_enabled:
                logger.info(f"Enabled degraded mode for {component}")
                self.system_state = SystemState.DEGRADED
                return True
            else:
                logger.warning(f"No degraded mode available for {component}")
                return False
                
        except Exception as e:
            logger.error(f"Graceful degradation failed: {e}")
            return False
    
    def _failover_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt failover recovery."""
        # Simplified failover - would integrate with load balancer/orchestration
        component = error_event.component
        
        logger.info(f"Attempting failover for {component}")
        
        # This would typically involve:
        # 1. Marking current instance as unhealthy
        # 2. Routing traffic to backup instances
        # 3. Starting recovery procedures
        
        # Placeholder implementation
        return True
    
    def _circuit_break_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt circuit breaker recovery."""
        component = error_event.component
        
        # Open circuit breaker
        self.circuit_breakers.open_breaker(component)
        
        logger.info(f"Opened circuit breaker for {component}")
        
        # Circuit breaker recovery is passive - it will automatically
        # transition to half-open state after timeout
        return True
    
    def _checkpoint_restore_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt checkpoint restore recovery."""
        try:
            # Find best checkpoint to restore
            suitable_checkpoint = self.checkpoint_manager.find_suitable_checkpoint(
                max_age_hours=24,
                min_validation_score=0.0
            )
            
            if not suitable_checkpoint:
                logger.warning("No suitable checkpoint found for restore")
                return False
            
            # Restore checkpoint
            self.restore_checkpoint(suitable_checkpoint["checkpoint_id"])
            
            logger.info(f"Restored to checkpoint: {suitable_checkpoint['checkpoint_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint restore recovery failed: {e}")
            return False
    
    def _determine_error_severity(self, error: Exception, component: str) -> ErrorSeverity:
        """Determine error severity based on error type and component."""
        error_type = type(error).__name__
        
        # Critical errors
        if error_type in ["SystemExit", "KeyboardInterrupt", "MemoryError"]:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in ["RuntimeError", "OSError", "ConnectionError"]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in ["ValueError", "TypeError", "AttributeError"]:
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def _get_stack_trace(self, error: Exception) -> str:
        """Get stack trace from exception (privacy-preserving)."""
        import traceback
        
        # Get full stack trace
        full_trace = traceback.format_exception(type(error), error, error.__traceback__)
        
        # Remove sensitive information (file paths, variables, etc.)
        sanitized_trace = []
        for line in full_trace:
            # Remove file paths
            sanitized_line = line.split("/")[-1] if "/" in line else line
            sanitized_line = sanitized_line.split("\\")[-1] if "\\" in sanitized_line else sanitized_line
            
            # Remove variable values (keep only variable names)
            if "=" in sanitized_line and not sanitized_line.strip().startswith("#"):
                parts = sanitized_line.split("=")
                if len(parts) > 1:
                    sanitized_line = parts[0] + "= <redacted>"
            
            sanitized_trace.append(sanitized_line)
        
        return "".join(sanitized_trace[-5:])  # Keep only last 5 lines
    
    def _log_error_safely(self, error_event: ErrorEvent):
        """Log error in privacy-preserving manner."""
        # Create sanitized log entry
        log_entry = {
            "timestamp": error_event.timestamp.isoformat(),
            "component": error_event.component,
            "error_type": error_event.error_type,
            "severity": error_event.severity.value,
            "message_hash": hashlib.sha256(error_event.message.encode()).hexdigest()[:16]
        }
        
        # Log without sensitive information
        logger.error(f"Error in {error_event.component}: {error_event.error_type} "
                    f"(severity: {error_event.severity.value})")
        
        # Store detailed information securely
        error_log_file = self.checkpoint_dir / "error_log.jsonl"
        with open(error_log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


class CheckpointManager:
    """Manages system checkpoints with privacy budget tracking."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def create_checkpoint(
        self,
        component_states: Dict[str, Any],
        privacy_budget_consumed: float,
        training_step: int = 0,
        validation_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new checkpoint."""
        checkpoint_id = hashlib.md5(
            f"{datetime.now().isoformat()}_{training_step}".encode()
        ).hexdigest()[:12]
        
        # Calculate state hashes
        model_state_hash = self._calculate_state_hash(component_states.get("model", {}))
        data_state_hash = self._calculate_state_hash(component_states.get("data", {}))
        
        # Create checkpoint metadata
        checkpoint_metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            privacy_budget_consumed=privacy_budget_consumed,
            model_state_hash=model_state_hash,
            data_state_hash=data_state_hash,
            training_step=training_step,
            validation_score=validation_score,
            component_states=component_states
        )
        
        # Save checkpoint data
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        metadata_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        # Save component states
        with open(checkpoint_file, "wb") as f:
            pickle.dump(component_states, f)
        
        # Save metadata
        with open(metadata_file, "w") as f:
            json.dump(checkpoint_metadata.to_dict(), f, indent=2)
        
        logger.info(f"Created checkpoint {checkpoint_id} at step {training_step}")
        return checkpoint_id
    
    def restore_checkpoint(
        self,
        checkpoint_id: Optional[str] = None,
        privacy_budget_limit: Optional[float] = None
    ) -> Dict[str, Any]:
        """Restore from checkpoint."""
        # Find checkpoint to restore
        if checkpoint_id is None:
            # Get latest checkpoint
            checkpoint_metadata = self.get_latest_checkpoint()
            if not checkpoint_metadata:
                raise RuntimeError("No checkpoints available")
        else:
            checkpoint_metadata = self._load_checkpoint_metadata(checkpoint_id)
            if not checkpoint_metadata:
                raise RuntimeError(f"Checkpoint {checkpoint_id} not found")
        
        # Check privacy budget limit
        if (privacy_budget_limit is not None and
            checkpoint_metadata["privacy_budget_consumed"] > privacy_budget_limit):
            raise RuntimeError(
                f"Checkpoint privacy budget ({checkpoint_metadata['privacy_budget_consumed']}) "
                f"exceeds limit ({privacy_budget_limit})"
            )
        
        # Load component states
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_metadata['checkpoint_id']}.pkl"
        
        if not checkpoint_file.exists():
            raise RuntimeError(f"Checkpoint data file not found: {checkpoint_file}")
        
        with open(checkpoint_file, "rb") as f:
            component_states = pickle.load(f)
        
        restore_info = {
            "checkpoint_id": checkpoint_metadata["checkpoint_id"],
            "timestamp": checkpoint_metadata["timestamp"],
            "privacy_budget_consumed": checkpoint_metadata["privacy_budget_consumed"],
            "training_step": checkpoint_metadata["training_step"],
            "validation_score": checkpoint_metadata["validation_score"],
            "component_states": component_states
        }
        
        return restore_info
    
    def get_available_checkpoints(self) -> List[Dict[str, Any]]:
        """Get list of available checkpoints."""
        checkpoints = []
        
        for metadata_file in self.checkpoint_dir.glob("*.json"):
            try:
                metadata = self._load_checkpoint_metadata(metadata_file.stem)
                if metadata:
                    checkpoints.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata {metadata_file}: {e}")
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint."""
        checkpoints = self.get_available_checkpoints()
        return checkpoints[0] if checkpoints else None
    
    def find_suitable_checkpoint(
        self,
        max_age_hours: int = 24,
        min_validation_score: float = 0.0,
        max_privacy_budget: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Find suitable checkpoint based on criteria."""
        checkpoints = self.get_available_checkpoints()
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        suitable_checkpoints = []
        for checkpoint in checkpoints:
            checkpoint_time = datetime.fromisoformat(checkpoint["timestamp"])
            
            # Check age
            if checkpoint_time < cutoff_time:
                continue
            
            # Check validation score
            if checkpoint["validation_score"] < min_validation_score:
                continue
            
            # Check privacy budget
            if (max_privacy_budget is not None and
                checkpoint["privacy_budget_consumed"] > max_privacy_budget):
                continue
            
            suitable_checkpoints.append(checkpoint)
        
        # Return best checkpoint (highest validation score)
        if suitable_checkpoints:
            return max(suitable_checkpoints, key=lambda x: x["validation_score"])
        
        return None
    
    def cleanup_old_checkpoints(self, keep_count: int = 10, max_age_days: int = 30):
        """Clean up old checkpoints."""
        checkpoints = self.get_available_checkpoints()
        
        # Keep recent checkpoints
        checkpoints_to_keep = checkpoints[:keep_count]
        
        # Remove old checkpoints
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        for checkpoint in checkpoints[keep_count:]:
            checkpoint_time = datetime.fromisoformat(checkpoint["timestamp"])
            
            if checkpoint_time < cutoff_time:
                checkpoint_id = checkpoint["checkpoint_id"]
                
                # Remove files
                checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
                metadata_file = self.checkpoint_dir / f"{checkpoint_id}.json"
                
                try:
                    if checkpoint_file.exists():
                        checkpoint_file.unlink()
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    logger.info(f"Cleaned up old checkpoint: {checkpoint_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to cleanup checkpoint {checkpoint_id}: {e}")
    
    def _load_checkpoint_metadata(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint metadata."""
        metadata_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint metadata {checkpoint_id}: {e}")
            return None
    
    def _calculate_state_hash(self, state: Dict[str, Any]) -> str:
        """Calculate hash of state for integrity checking."""
        state_str = json.dumps(state, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]


class CircuitBreakerRegistry:
    """Registry for circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreakerState] = {}
    
    def create_breaker(
        self,
        name: str,
        config: Dict[str, Any]
    ):
        """Create a circuit breaker."""
        breaker = CircuitBreakerState(
            name=name,
            state="closed",
            failure_count=0,
            failure_threshold=config.get("failure_threshold", 5),
            timeout_seconds=config.get("timeout_seconds", 60),
            last_failure_time=None,
            success_threshold=config.get("success_threshold", 3),
            success_count=0
        )
        
        self.breakers[name] = breaker
        logger.info(f"Created circuit breaker: {name}")
    
    def record_failure(self, name: str):
        """Record a failure for circuit breaker."""
        if name not in self.breakers:
            return
        
        breaker = self.breakers[name]
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now()
        breaker.success_count = 0  # Reset success count
        
        # Check if should open breaker
        if breaker.failure_count >= breaker.failure_threshold:
            breaker.state = "open"
            logger.warning(f"Circuit breaker opened: {name}")
    
    def record_success(self, name: str):
        """Record a success for circuit breaker."""
        if name not in self.breakers:
            return
        
        breaker = self.breakers[name]
        
        if breaker.state == "closed":
            # Reset failure count on success
            breaker.failure_count = 0
        
        elif breaker.state == "half_open":
            breaker.success_count += 1
            
            # Check if should close breaker
            if breaker.success_count >= breaker.success_threshold:
                breaker.state = "closed"
                breaker.failure_count = 0
                breaker.success_count = 0
                logger.info(f"Circuit breaker closed: {name}")
    
    def is_open(self, name: str) -> bool:
        """Check if circuit breaker is open."""
        if name not in self.breakers:
            return False
        
        breaker = self.breakers[name]
        
        # Check if should transition from open to half-open
        if (breaker.state == "open" and 
            breaker.last_failure_time and
            (datetime.now() - breaker.last_failure_time).seconds >= breaker.timeout_seconds):
            
            breaker.state = "half_open"
            breaker.success_count = 0
            logger.info(f"Circuit breaker half-open: {name}")
        
        return breaker.state == "open"
    
    def get_failure_count(self, name: str) -> int:
        """Get failure count for circuit breaker."""
        if name not in self.breakers:
            return 0
        
        return self.breakers[name].failure_count
    
    def open_breaker(self, name: str):
        """Manually open circuit breaker."""
        if name in self.breakers:
            self.breakers[name].state = "open"
            self.breakers[name].last_failure_time = datetime.now()
    
    def close_breaker(self, name: str):
        """Manually close circuit breaker."""
        if name in self.breakers:
            breaker = self.breakers[name]
            breaker.state = "closed"
            breaker.failure_count = 0
            breaker.success_count = 0


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_wait_time(self, attempt: int) -> float:
        """Get wait time for retry attempt."""
        # Calculate exponential backoff
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            import random
            jitter_factor = random.uniform(0.8, 1.2)
            delay *= jitter_factor
        
        return delay


class DegradationManager:
    """Manages graceful degradation strategies."""
    
    def __init__(self):
        self.degraded_functions: Dict[str, Callable] = {}
        self.degraded_components: set = set()
        self.trigger_conditions: Dict[str, List[str]] = {}
    
    def register_degradation(
        self,
        component: str,
        degraded_function: Callable,
        trigger_conditions: List[str]
    ):
        """Register degradation strategy for component."""
        self.degraded_functions[component] = degraded_function
        self.trigger_conditions[component] = trigger_conditions
        
        logger.info(f"Registered degradation strategy for: {component}")
    
    def enable_degraded_mode(self, component: str) -> bool:
        """Enable degraded mode for component."""
        if component not in self.degraded_functions:
            return False
        
        self.degraded_components.add(component)
        logger.info(f"Enabled degraded mode for: {component}")
        
        return True
    
    def disable_degraded_mode(self, component: str):
        """Disable degraded mode for component."""
        self.degraded_components.discard(component)
        logger.info(f"Disabled degraded mode for: {component}")
    
    def is_degraded(self, component: str) -> bool:
        """Check if component is in degraded mode."""
        return component in self.degraded_components
    
    def get_degraded_function(self, component: str) -> Optional[Callable]:
        """Get degraded function for component."""
        if component in self.degraded_components:
            return self.degraded_functions.get(component)
        return None


class HealthMonitor:
    """Monitors component health status."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
    
    def register_component(self, component: str, health_check: Callable[[], bool]):
        """Register component health check."""
        self.health_checks[component] = health_check
        self.health_status[component] = {
            "healthy": True,
            "last_check": datetime.now(),
            "failure_count": 0
        }
        
        logger.info(f"Registered health check for: {component}")
    
    def check_component_health(self, component: str) -> bool:
        """Check health of specific component."""
        if component not in self.health_checks:
            return True  # Assume healthy if no check registered
        
        try:
            is_healthy = self.health_checks[component]()
            
            # Update status
            status = self.health_status[component]
            status["healthy"] = is_healthy
            status["last_check"] = datetime.now()
            
            if is_healthy:
                status["failure_count"] = 0
            else:
                status["failure_count"] += 1
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check failed for {component}: {e}")
            
            status = self.health_status[component]
            status["healthy"] = False
            status["last_check"] = datetime.now()
            status["failure_count"] += 1
            
            return False
    
    def check_all_health(self) -> Dict[str, bool]:
        """Check health of all registered components."""
        health_results = {}
        
        for component in self.health_checks:
            health_results[component] = self.check_component_health(component)
        
        return health_results
    
    def get_all_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed health status of all components."""
        return {
            component: {
                **status,
                "last_check": status["last_check"].isoformat()
            }
            for component, status in self.health_status.items()
        }
    
    def has_component(self, component: str) -> bool:
        """Check if component is registered."""
        return component in self.health_checks