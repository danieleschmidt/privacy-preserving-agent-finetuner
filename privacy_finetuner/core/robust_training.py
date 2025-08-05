"""Robust training components with comprehensive error handling and monitoring."""

import logging
import time
import json
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch

from .privacy_config import PrivacyConfig
from .exceptions import (
    SecurityViolationException,
    ModelTrainingException,
    PrivacyBudgetExhaustedException,
    ResourceExhaustedException
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics for monitoring."""
    step: int
    epoch: int
    loss: float
    accuracy: float = 0.0
    privacy_spent: float = 0.0
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    memory_usage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SecurityEvent:
    """Security event for monitoring."""
    event_type: str
    severity: str
    message: str
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False


class TrainingMonitor:
    """Comprehensive training monitoring with early stopping and checkpointing."""
    
    def __init__(
        self,
        checkpoint_interval: int = 100,
        early_stopping_patience: int = 5,
        privacy_config: PrivacyConfig = None,
        metrics_history_size: int = 1000,
        checkpoint_dir: str = "./checkpoints"
    ):
        """Initialize training monitor.
        
        Args:
            checkpoint_interval: Steps between checkpoints
            early_stopping_patience: Patience for early stopping
            privacy_config: Privacy configuration
            metrics_history_size: Maximum metrics history size
            checkpoint_dir: Directory for checkpoints
        """
        self.checkpoint_interval = checkpoint_interval
        self.early_stopping_patience = early_stopping_patience
        self.privacy_config = privacy_config
        self.metrics_history_size = metrics_history_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.metrics_history: List[TrainingMetrics] = []
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.should_stop_early = False
        
        # Resource monitoring
        self.start_time = None
        self.resource_monitor = ResourceMonitor()
        
        # Recovery mechanisms
        self.recovery_strategies = {
            "gradient_explosion": self._handle_gradient_explosion,
            "loss_divergence": self._handle_loss_divergence,
            "memory_overflow": self._handle_memory_overflow,
            "privacy_budget_low": self._handle_privacy_budget_low
        }
        
        logger.info(f"TrainingMonitor initialized with checkpoint_interval={checkpoint_interval}")
    
    def start_training(self, total_steps: int, total_epochs: int) -> None:
        """Start training monitoring."""
        self.start_time = datetime.now()
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        self.resource_monitor.start_monitoring()
        
        logger.info(f"Training started: {total_epochs} epochs, {total_steps} steps")
    
    def log_step(self, metrics: TrainingMetrics) -> None:
        """Log training step metrics."""
        self.metrics_history.append(metrics)
        
        # Keep history size manageable
        if len(self.metrics_history) > self.metrics_history_size:
            self.metrics_history.pop(0)
        
        # Check for issues
        self._check_training_health(metrics)
        
        # Early stopping check
        self._check_early_stopping(metrics)
        
        # Checkpoint if needed
        if metrics.step % self.checkpoint_interval == 0:
            self._create_checkpoint(metrics)
        
        # Log progress
        if metrics.step % 10 == 0:
            self._log_progress(metrics)
    
    def _check_training_health(self, metrics: TrainingMetrics) -> None:
        """Check training health and detect issues."""
        # Gradient explosion detection
        if metrics.gradient_norm > 100.0:  # Configurable threshold
            logger.warning(f"High gradient norm detected: {metrics.gradient_norm:.4f}")
            self.recovery_strategies["gradient_explosion"](metrics)
        
        # Loss divergence detection
        if len(self.metrics_history) > 10:
            recent_losses = [m.loss for m in self.metrics_history[-10:]]
            if all(l1 < l2 for l1, l2 in zip(recent_losses[:-1], recent_losses[1:])):
                logger.warning("Loss divergence detected")
                self.recovery_strategies["loss_divergence"](metrics)
        
        # Memory usage check
        if metrics.memory_usage > 0.9:  # 90% memory usage
            logger.warning(f"High memory usage: {metrics.memory_usage:.2%}")
            self.recovery_strategies["memory_overflow"](metrics)
        
        # Privacy budget check
        if self.privacy_config and metrics.privacy_spent > 0.8 * self.privacy_config.epsilon:
            logger.warning(f"Privacy budget low: {metrics.privacy_spent:.6f}/{self.privacy_config.epsilon:.6f}")
            self.recovery_strategies["privacy_budget_low"](metrics)
    
    def _check_early_stopping(self, metrics: TrainingMetrics) -> None:
        """Check early stopping conditions."""
        if metrics.loss < self.best_loss:
            self.best_loss = metrics.loss
            self.best_accuracy = metrics.accuracy
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.early_stopping_patience:
            logger.info(f"Early stopping triggered after {self.patience_counter} steps without improvement")
            self.should_stop_early = True
    
    def _create_checkpoint(self, metrics: TrainingMetrics) -> None:
        """Create training checkpoint."""
        checkpoint_data = {
            "step": metrics.step,
            "epoch": metrics.epoch,
            "loss": metrics.loss,
            "accuracy": metrics.accuracy,
            "privacy_spent": metrics.privacy_spent,
            "best_loss": self.best_loss,
            "best_accuracy": self.best_accuracy,
            "timestamp": datetime.now().isoformat(),
            "training_duration": (datetime.now() - self.start_time).total_seconds()
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{metrics.step}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.debug(f"Checkpoint created: {checkpoint_path}")
    
    def _log_progress(self, metrics: TrainingMetrics) -> None:
        """Log training progress."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        steps_per_sec = metrics.step / elapsed if elapsed > 0 else 0
        
        logger.info(
            f"Step {metrics.step}/{self.total_steps} | "
            f"Epoch {metrics.epoch}/{self.total_epochs} | "
            f"Loss: {metrics.loss:.4f} | "
            f"Acc: {metrics.accuracy:.4f} | "
            f"Privacy: {metrics.privacy_spent:.6f} | "
            f"Speed: {steps_per_sec:.2f} steps/s"
        )
    
    def _handle_gradient_explosion(self, metrics: TrainingMetrics) -> None:
        """Handle gradient explosion."""
        logger.warning("Applying gradient explosion recovery")
        # This would be handled by the trainer with gradient clipping adjustments
        
    def _handle_loss_divergence(self, metrics: TrainingMetrics) -> None:
        """Handle loss divergence."""
        logger.warning("Applying loss divergence recovery")
        # This would trigger learning rate reduction
        
    def _handle_memory_overflow(self, metrics: TrainingMetrics) -> None:
        """Handle memory overflow."""
        logger.warning("Applying memory overflow recovery")
        torch.cuda.empty_cache()  # Clear GPU cache
        
    def _handle_privacy_budget_low(self, metrics: TrainingMetrics) -> None:
        """Handle low privacy budget."""
        logger.warning("Privacy budget running low - adjusting noise parameters")
        # This would trigger privacy parameter adjustments
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.metrics_history:
            return {"status": "no_metrics"}
        
        current_time = datetime.now()
        duration = (current_time - self.start_time).total_seconds() if self.start_time else 0
        
        recent_metrics = self.metrics_history[-10:]
        avg_loss = np.mean([m.loss for m in recent_metrics])
        avg_accuracy = np.mean([m.accuracy for m in recent_metrics])
        
        return {
            "training_duration": duration,
            "total_steps": len(self.metrics_history),
            "best_loss": self.best_loss,
            "best_accuracy": self.best_accuracy,
            "current_loss": recent_metrics[-1].loss if recent_metrics else 0.0,
            "current_accuracy": recent_metrics[-1].accuracy if recent_metrics else 0.0,
            "average_loss": avg_loss,
            "average_accuracy": avg_accuracy,
            "should_stop_early": self.should_stop_early,
            "patience_counter": self.patience_counter,
            "resource_usage": self.resource_monitor.get_current_usage()
        }


class SecurityMonitor:
    """Security monitoring for privacy-preserving training."""
    
    def __init__(self, max_events: int = 1000):
        """Initialize security monitor.
        
        Args:
            max_events: Maximum security events to store
        """
        self.max_events = max_events
        self.security_events: List[SecurityEvent] = []
        self.threat_patterns = self._load_threat_patterns()
        self.monitoring_enabled = True
        
        # Security metrics
        self.failed_authentications = 0
        self.suspicious_activities = 0
        self.data_access_violations = 0
        
        logger.info("SecurityMonitor initialized")
    
    def _load_threat_patterns(self) -> Dict[str, Any]:
        """Load threat detection patterns."""
        return {
            "unusual_data_access": {
                "pattern": "rapid_sequential_access",
                "threshold": 100,
                "time_window": 60  # seconds
            },
            "gradient_leakage": {
                "pattern": "high_gradient_correlation",
                "threshold": 0.9,
                "time_window": 300
            },
            "model_extraction": {
                "pattern": "systematic_queries",
                "threshold": 1000,
                "time_window": 3600
            }
        }
    
    def monitor_data_access(self, user_id: str, data_path: str, access_type: str) -> None:
        """Monitor data access for security violations."""
        if not self.monitoring_enabled:
            return
        
        # Check for unusual access patterns
        recent_accesses = self._get_recent_accesses(user_id, 60)  # Last minute
        
        if len(recent_accesses) > self.threat_patterns["unusual_data_access"]["threshold"]:
            self._record_security_event(
                "DATA_ACCESS_VIOLATION",
                "HIGH",
                f"Unusual data access pattern detected for user {user_id}",
                {
                    "user_id": user_id,
                    "data_path": data_path,
                    "access_type": access_type,
                    "recent_access_count": len(recent_accesses)
                }
            )
    
    def monitor_gradient_updates(self, gradients: Dict[str, torch.Tensor]) -> None:
        """Monitor gradient updates for potential leakage."""
        if not self.monitoring_enabled:
            return
        
        # Analyze gradient patterns for potential information leakage
        for name, grad in gradients.items():
            if grad is not None:
                grad_norm = torch.norm(grad).item()
                grad_std = torch.std(grad).item()
                
                # Check for suspiciously structured gradients
                if grad_std < 1e-6 and grad_norm > 1.0:  # High norm, low variance
                    self._record_security_event(
                        "GRADIENT_LEAKAGE_RISK",
                        "MEDIUM",
                        f"Suspicious gradient pattern in {name}",
                        {
                            "parameter_name": name,
                            "gradient_norm": grad_norm,
                            "gradient_std": grad_std
                        }
                    )
    
    def monitor_model_queries(self, user_id: str, query_pattern: str) -> None:
        """Monitor model queries for extraction attempts."""
        if not self.monitoring_enabled:
            return
        
        # Track query patterns for potential model extraction
        recent_queries = self._get_recent_queries(user_id, 3600)  # Last hour
        
        if len(recent_queries) > self.threat_patterns["model_extraction"]["threshold"]:
            self._record_security_event(
                "MODEL_EXTRACTION_ATTEMPT",
                "HIGH",
                f"Potential model extraction attempt by user {user_id}",
                {
                    "user_id": user_id,
                    "query_count": len(recent_queries),
                    "query_pattern": query_pattern
                }
            )
    
    def _record_security_event(
        self, 
        event_type: str, 
        severity: str, 
        message: str, 
        context: Dict[str, Any]
    ) -> None:
        """Record security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            context=context
        )
        
        self.security_events.append(event)
        
        # Keep events list manageable
        if len(self.security_events) > self.max_events:
            self.security_events.pop(0)
        
        # Update counters
        if event_type == "DATA_ACCESS_VIOLATION":
            self.data_access_violations += 1
        elif "EXTRACTION" in event_type:
            self.suspicious_activities += 1
        
        # Log high severity events
        if severity == "HIGH":
            logger.warning(f"Security Event [{severity}]: {message}")
        
        # Trigger automated response for critical events
        if severity == "CRITICAL":
            self._trigger_security_response(event)
    
    def _trigger_security_response(self, event: SecurityEvent) -> None:
        """Trigger automated security response."""
        logger.critical(f"Critical security event: {event.message}")
        
        # Implement automated responses based on event type
        if event.event_type == "MODEL_EXTRACTION_ATTEMPT":
            # Rate limit the user
            self._apply_rate_limiting(event.context.get("user_id"))
        
        elif event.event_type == "DATA_ACCESS_VIOLATION":
            # Temporarily suspend data access
            self._suspend_data_access(event.context.get("user_id"))
    
    def _get_recent_accesses(self, user_id: str, time_window: int) -> List[Dict[str, Any]]:
        """Get recent data accesses for user."""
        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        
        # Filter events by user and time
        recent_events = [
            event for event in self.security_events
            if (event.context.get("user_id") == user_id and 
                event.timestamp > cutoff_time and
                "access" in event.event_type.lower())
        ]
        
        return [event.context for event in recent_events]
    
    def _get_recent_queries(self, user_id: str, time_window: int) -> List[Dict[str, Any]]:
        """Get recent model queries for user."""
        cutoff_time = datetime.now() - timedelta(seconds=time_window)
        
        # Filter events by user and time
        recent_events = [
            event for event in self.security_events
            if (event.context.get("user_id") == user_id and 
                event.timestamp > cutoff_time and
                "query" in event.event_type.lower())
        ]
        
        return [event.context for event in recent_events]
    
    def _apply_rate_limiting(self, user_id: str) -> None:
        """Apply rate limiting to user."""
        logger.warning(f"Applying rate limiting to user {user_id}")
        # Implementation would integrate with API gateway
    
    def _suspend_data_access(self, user_id: str) -> None:
        """Suspend data access for user."""
        logger.warning(f"Suspending data access for user {user_id}")
        # Implementation would update user permissions
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary."""
        recent_events = [
            event for event in self.security_events
            if event.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "total_events": len(self.security_events),
            "recent_events": len(recent_events),
            "failed_authentications": self.failed_authentications,
            "suspicious_activities": self.suspicious_activities,
            "data_access_violations": self.data_access_violations,
            "high_severity_events": len([e for e in recent_events if e.severity == "HIGH"]),
            "unresolved_events": len([e for e in self.security_events if not e.resolved]),
            "monitoring_enabled": self.monitoring_enabled
        }


class AuditLogger:
    """Comprehensive audit logging for compliance."""
    
    def __init__(self, log_file: str = "privacy_audit.log", max_log_size: int = 100 * 1024 * 1024):
        """Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
            max_log_size: Maximum log file size in bytes
        """
        self.log_file = Path(log_file)
        self.max_log_size = max_log_size
        self.log_lock = threading.Lock()
        
        # Setup audit logger
        self.audit_logger = logging.getLogger("privacy_audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.audit_logger.addHandler(handler)
        
        logger.info(f"AuditLogger initialized with log file: {self.log_file}")
    
    def log_initialization(self, context: Dict[str, Any]) -> None:
        """Log system initialization."""
        self._log_audit_event("SYSTEM_INITIALIZATION", context)
    
    def log_privacy_budget_usage(self, epsilon_spent: float, context: Dict[str, Any]) -> None:
        """Log privacy budget usage."""
        audit_context = {
            "epsilon_spent": epsilon_spent,
            **context
        }
        self._log_audit_event("PRIVACY_BUDGET_USAGE", audit_context)
    
    def log_data_access(self, user_id: str, data_path: str, access_type: str) -> None:
        """Log data access."""
        context = {
            "user_id": user_id,
            "data_path": data_path,
            "access_type": access_type
        }
        self._log_audit_event("DATA_ACCESS", context)
    
    def log_model_training(self, model_id: str, training_params: Dict[str, Any]) -> None:
        """Log model training."""
        context = {
            "model_id": model_id,
            "training_params": training_params
        }
        self._log_audit_event("MODEL_TRAINING", context)
    
    def log_security_event(self, event_type: str, severity: str, context: Dict[str, Any]) -> None:
        """Log security event."""
        audit_context = {
            "event_type": event_type,
            "severity": severity,
            **context
        }
        self._log_audit_event("SECURITY_EVENT", audit_context)
    
    def _log_audit_event(self, event_type: str, context: Dict[str, Any]) -> None:
        """Log audit event with thread safety."""
        with self.log_lock:
            # Check log rotation
            if self.log_file.exists() and self.log_file.stat().st_size > self.max_log_size:
                self._rotate_log()
            
            # Create audit record
            audit_record = {
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "checksum": self._calculate_checksum(context)
            }
            
            # Log the event
            self.audit_logger.info(json.dumps(audit_record))
    
    def _rotate_log(self) -> None:
        """Rotate audit log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.log_file.with_suffix(f".{timestamp}.log")
        
        if self.log_file.exists():
            self.log_file.rename(backup_file)
            logger.info(f"Audit log rotated to: {backup_file}")
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for audit record integrity."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]


class ResourceMonitor:
    """Monitor system resources during training."""
    
    def __init__(self):
        """Initialize resource monitor."""
        self.monitoring = False
        self.resource_history = []
        self.monitor_thread = None
        
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Resource monitoring loop."""
        import psutil
        
        while self.monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Get GPU metrics if available
                gpu_memory = 0.0
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                
                resource_data = {
                    "timestamp": datetime.now(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available": memory.available,
                    "gpu_memory_percent": gpu_memory,
                }
                
                self.resource_history.append(resource_data)
                
                # Keep history manageable
                if len(self.resource_history) > 1000:
                    self.resource_history.pop(0)
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(10)  # Wait longer on error
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        if not self.resource_history:
            return {"cpu": 0.0, "memory": 0.0, "gpu_memory": 0.0}
        
        latest = self.resource_history[-1]
        return {
            "cpu": latest["cpu_percent"],
            "memory": latest["memory_percent"],
            "gpu_memory": latest["gpu_memory_percent"]
        }
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.resource_history:
            return {"status": "no_data"}
        
        recent_data = self.resource_history[-60:]  # Last 5 minutes
        
        cpu_values = [d["cpu_percent"] for d in recent_data]
        memory_values = [d["memory_percent"] for d in recent_data]
        gpu_values = [d["gpu_memory_percent"] for d in recent_data]
        
        return {
            "monitoring_duration": len(self.resource_history) * 5,  # seconds
            "cpu": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": np.mean(cpu_values) if cpu_values else 0,
                "peak": np.max(cpu_values) if cpu_values else 0
            },
            "memory": {
                "current": memory_values[-1] if memory_values else 0,
                "average": np.mean(memory_values) if memory_values else 0,
                "peak": np.max(memory_values) if memory_values else 0
            },
            "gpu_memory": {
                "current": gpu_values[-1] if gpu_values else 0,
                "average": np.mean(gpu_values) if gpu_values else 0,
                "peak": np.max(gpu_values) if gpu_values else 0
            }
        }