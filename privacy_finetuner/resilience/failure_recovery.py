"""Comprehensive failure recovery system for privacy-preserving training.

This module implements advanced recovery strategies for handling various failure
scenarios while preserving privacy guarantees and training progress.
"""

import time
import logging
import json
import pickle
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur during training."""
    SYSTEM_CRASH = "system_crash"
    NETWORK_FAILURE = "network_failure"
    GPU_MEMORY_ERROR = "gpu_memory_error"
    DATA_CORRUPTION = "data_corruption"
    PRIVACY_VIOLATION = "privacy_violation"
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"
    DISTRIBUTED_NODE_FAILURE = "distributed_node_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure types."""
    RESTART_FROM_CHECKPOINT = "restart_from_checkpoint"
    ROLLBACK_TO_SAFE_STATE = "rollback_to_safe_state"
    REDISTRIBUTE_COMPUTATION = "redistribute_computation"
    REDUCE_RESOURCE_USAGE = "reduce_resource_usage"
    EMERGENCY_STOP = "emergency_stop"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class RecoveryPoint:
    """Recovery point containing training state and metadata."""
    recovery_id: str
    timestamp: str
    epoch: int
    step: int
    model_state: Optional[Dict[str, Any]]
    optimizer_state: Optional[Dict[str, Any]]
    privacy_state: Dict[str, Any]
    training_metrics: Dict[str, Any]
    system_state: Dict[str, Any]
    data_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def is_valid(self) -> bool:
        """Check if recovery point is valid."""
        required_fields = ['recovery_id', 'timestamp', 'privacy_state']
        return all(getattr(self, field, None) is not None for field in required_fields)


@dataclass
class FailureEvent:
    """Record of a failure event and recovery attempt."""
    failure_id: str
    failure_type: FailureType
    failure_time: str
    description: str
    affected_components: List[str]
    recovery_strategy: RecoveryStrategy
    recovery_success: bool
    recovery_time: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FailureRecoverySystem:
    """Advanced failure recovery system with privacy preservation."""
    
    def __init__(
        self,
        checkpoint_dir: str = "recovery_checkpoints",
        max_recovery_attempts: int = 3,
        auto_recovery_enabled: bool = True,
        privacy_threshold: float = 0.8
    ):
        """Initialize failure recovery system.
        
        Args:
            checkpoint_dir: Directory for recovery checkpoints
            max_recovery_attempts: Maximum recovery attempts per failure
            auto_recovery_enabled: Enable automatic recovery
            privacy_threshold: Privacy budget threshold for recovery decisions
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_recovery_attempts = max_recovery_attempts
        self.auto_recovery_enabled = auto_recovery_enabled
        self.privacy_threshold = privacy_threshold
        
        # Recovery state
        self.recovery_points = {}
        self.failure_history = []
        self.recovery_callbacks = {}
        self.active_recoveries = {}
        
        # Recovery strategies for different failure types
        self.recovery_strategies = {
            FailureType.SYSTEM_CRASH: [
                RecoveryStrategy.RESTART_FROM_CHECKPOINT,
                RecoveryStrategy.ROLLBACK_TO_SAFE_STATE
            ],
            FailureType.NETWORK_FAILURE: [
                RecoveryStrategy.REDISTRIBUTE_COMPUTATION,
                RecoveryStrategy.RESTART_FROM_CHECKPOINT
            ],
            FailureType.GPU_MEMORY_ERROR: [
                RecoveryStrategy.REDUCE_RESOURCE_USAGE,
                RecoveryStrategy.RESTART_FROM_CHECKPOINT
            ],
            FailureType.DATA_CORRUPTION: [
                RecoveryStrategy.ROLLBACK_TO_SAFE_STATE,
                RecoveryStrategy.EMERGENCY_STOP
            ],
            FailureType.PRIVACY_VIOLATION: [
                RecoveryStrategy.EMERGENCY_STOP,
                RecoveryStrategy.ROLLBACK_TO_SAFE_STATE
            ],
            FailureType.CHECKPOINT_CORRUPTION: [
                RecoveryStrategy.ROLLBACK_TO_SAFE_STATE,
                RecoveryStrategy.RESTART_FROM_CHECKPOINT
            ],
            FailureType.DISTRIBUTED_NODE_FAILURE: [
                RecoveryStrategy.REDISTRIBUTE_COMPUTATION,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            FailureType.RESOURCE_EXHAUSTION: [
                RecoveryStrategy.REDUCE_RESOURCE_USAGE,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ]
        }
        
        logger.info(f"FailureRecoverySystem initialized with checkpoint dir: {checkpoint_dir}")
    
    def create_recovery_point(
        self,
        epoch: int,
        step: int,
        model_state: Optional[Dict[str, Any]] = None,
        optimizer_state: Optional[Dict[str, Any]] = None,
        privacy_state: Dict[str, Any] = None,
        training_metrics: Dict[str, Any] = None,
        system_state: Dict[str, Any] = None
    ) -> str:
        """Create a recovery point for current training state.
        
        Args:
            epoch: Current epoch
            step: Current step
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary  
            privacy_state: Privacy budget and configuration state
            training_metrics: Current training metrics
            system_state: System resource and configuration state
            
        Returns:
            Recovery point ID
        """
        # Generate recovery point ID
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        recovery_data = f"{epoch}_{step}_{timestamp}"
        recovery_id = hashlib.sha256(recovery_data.encode()).hexdigest()[:16]
        
        # Create data hash for integrity checking
        state_data = {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "privacy_state": privacy_state or {},
            "training_metrics": training_metrics or {},
            "system_state": system_state or {}
        }
        data_hash = hashlib.sha256(json.dumps(state_data, sort_keys=True).encode()).hexdigest()
        
        # Create recovery point
        recovery_point = RecoveryPoint(
            recovery_id=recovery_id,
            timestamp=timestamp,
            epoch=epoch,
            step=step,
            model_state=model_state,
            optimizer_state=optimizer_state,
            privacy_state=privacy_state or {},
            training_metrics=training_metrics or {},
            system_state=system_state or {},
            data_hash=data_hash
        )
        
        # Save recovery point
        self._save_recovery_point(recovery_point)
        self.recovery_points[recovery_id] = recovery_point
        
        logger.info(f"Created recovery point {recovery_id} at epoch {epoch}, step {step}")
        return recovery_id
    
    def handle_failure(
        self,
        failure_type: FailureType,
        description: str = "",
        affected_components: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Handle a training failure with appropriate recovery strategy.
        
        Args:
            failure_type: Type of failure that occurred
            description: Description of the failure
            affected_components: List of affected system components
            metadata: Additional failure metadata
            
        Returns:
            True if recovery was successful, False otherwise
        """
        failure_time = time.strftime("%Y-%m-%d %H:%M:%S")
        failure_id = hashlib.sha256(f"{failure_type.value}_{failure_time}".encode()).hexdigest()[:16]
        
        logger.error(f"Handling failure {failure_id}: {failure_type.value} - {description}")
        
        # Record failure event
        failure_event = FailureEvent(
            failure_id=failure_id,
            failure_type=failure_type,
            failure_time=failure_time,
            description=description,
            affected_components=affected_components or [],
            recovery_strategy=RecoveryStrategy.RESTART_FROM_CHECKPOINT,  # Will be updated
            recovery_success=False,
            recovery_time=0.0,
            metadata=metadata or {}
        )
        
        recovery_success = False
        recovery_start_time = time.time()
        
        if self.auto_recovery_enabled:
            # Attempt recovery strategies in order of preference
            strategies = self.recovery_strategies.get(failure_type, [RecoveryStrategy.RESTART_FROM_CHECKPOINT])
            
            for attempt, strategy in enumerate(strategies):
                if attempt >= self.max_recovery_attempts:
                    break
                
                logger.info(f"Attempting recovery with strategy: {strategy.value} (attempt {attempt + 1})")
                failure_event.recovery_strategy = strategy
                
                try:
                    if self._execute_recovery_strategy(strategy, failure_type, metadata or {}):
                        recovery_success = True
                        logger.info(f"Recovery successful with strategy: {strategy.value}")
                        break
                    else:
                        logger.warning(f"Recovery attempt {attempt + 1} failed with strategy: {strategy.value}")
                
                except Exception as e:
                    logger.error(f"Recovery attempt {attempt + 1} raised exception: {e}")
        
        # Record recovery results
        recovery_time = time.time() - recovery_start_time
        failure_event.recovery_success = recovery_success
        failure_event.recovery_time = recovery_time
        
        self.failure_history.append(failure_event)
        self._save_failure_event(failure_event)
        
        if recovery_success:
            logger.info(f"Failure {failure_id} recovered successfully in {recovery_time:.2f}s")
        else:
            logger.error(f"Failed to recover from failure {failure_id} after {len(strategies)} attempts")
        
        return recovery_success
    
    def _execute_recovery_strategy(
        self,
        strategy: RecoveryStrategy,
        failure_type: FailureType,
        metadata: Dict[str, Any]
    ) -> bool:
        """Execute specific recovery strategy.
        
        Args:
            strategy: Recovery strategy to execute
            failure_type: Original failure type
            metadata: Failure metadata
            
        Returns:
            True if recovery strategy succeeded
        """
        try:
            if strategy == RecoveryStrategy.RESTART_FROM_CHECKPOINT:
                return self._restart_from_checkpoint()
            
            elif strategy == RecoveryStrategy.ROLLBACK_TO_SAFE_STATE:
                return self._rollback_to_safe_state()
            
            elif strategy == RecoveryStrategy.REDISTRIBUTE_COMPUTATION:
                return self._redistribute_computation(metadata)
            
            elif strategy == RecoveryStrategy.REDUCE_RESOURCE_USAGE:
                return self._reduce_resource_usage(metadata)
            
            elif strategy == RecoveryStrategy.EMERGENCY_STOP:
                return self._emergency_stop()
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return self._graceful_degradation(metadata)
            
            else:
                logger.warning(f"Unknown recovery strategy: {strategy}")
                return False
        
        except Exception as e:
            logger.error(f"Recovery strategy {strategy.value} failed with exception: {e}")
            return False
    
    def _restart_from_checkpoint(self) -> bool:
        """Restart training from most recent valid checkpoint."""
        if not self.recovery_points:
            logger.error("No recovery points available for restart")
            return False
        
        # Find most recent valid recovery point
        latest_recovery = None
        for recovery_point in sorted(self.recovery_points.values(), 
                                   key=lambda rp: rp.timestamp, reverse=True):
            if recovery_point.is_valid() and self._validate_recovery_point(recovery_point):
                latest_recovery = recovery_point
                break
        
        if not latest_recovery:
            logger.error("No valid recovery points found")
            return False
        
        # Restore from recovery point
        success = self._restore_from_recovery_point(latest_recovery)
        if success:
            logger.info(f"Successfully restarted from checkpoint {latest_recovery.recovery_id}")
        
        return success
    
    def _rollback_to_safe_state(self) -> bool:
        """Rollback to a safe state before privacy violations."""
        # Find safe recovery points (below privacy threshold)
        safe_recovery_points = []
        
        for recovery_point in self.recovery_points.values():
            privacy_state = recovery_point.privacy_state
            epsilon_used = privacy_state.get("epsilon_spent", 0.0)
            epsilon_total = privacy_state.get("epsilon_total", 1.0)
            
            if epsilon_total > 0 and epsilon_used / epsilon_total < self.privacy_threshold:
                safe_recovery_points.append(recovery_point)
        
        if not safe_recovery_points:
            logger.error("No safe recovery points found below privacy threshold")
            return False
        
        # Select most recent safe recovery point
        safe_recovery = max(safe_recovery_points, key=lambda rp: rp.timestamp)
        
        success = self._restore_from_recovery_point(safe_recovery)
        if success:
            logger.info(f"Successfully rolled back to safe state {safe_recovery.recovery_id}")
        
        return success
    
    def _redistribute_computation(self, metadata: Dict[str, Any]) -> bool:
        """Redistribute computation after node failures."""
        failed_nodes = metadata.get("failed_nodes", [])
        available_nodes = metadata.get("available_nodes", [])
        
        if not available_nodes:
            logger.error("No available nodes for redistribution")
            return False
        
        # Simulate redistribution logic
        logger.info(f"Redistributing computation from {len(failed_nodes)} failed nodes to {len(available_nodes)} available nodes")
        
        # In a real implementation, this would:
        # 1. Recalculate data sharding
        # 2. Update distributed training configuration
        # 3. Restart training with new node configuration
        
        return len(available_nodes) > 0
    
    def _reduce_resource_usage(self, metadata: Dict[str, Any]) -> bool:
        """Reduce resource usage to recover from resource exhaustion."""
        current_batch_size = metadata.get("batch_size", 32)
        current_model_size = metadata.get("model_parameters", 1000000)
        
        # Reduce batch size
        new_batch_size = max(1, current_batch_size // 2)
        
        # Consider gradient accumulation to maintain effective batch size
        gradient_accumulation_steps = current_batch_size // new_batch_size
        
        logger.info(f"Reducing batch size from {current_batch_size} to {new_batch_size}")
        logger.info(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
        
        # In a real implementation, this would update training configuration
        return True
    
    def _emergency_stop(self) -> bool:
        """Execute emergency stop to prevent privacy violations."""
        logger.critical("Executing emergency stop due to privacy violation")
        
        # Save current state for forensic analysis
        emergency_checkpoint_id = self.create_recovery_point(
            epoch=-1,  # Special marker for emergency checkpoint
            step=-1,
            privacy_state={"emergency_stop": True, "stop_reason": "privacy_violation"}
        )
        
        logger.critical(f"Emergency checkpoint saved: {emergency_checkpoint_id}")
        
        # In a real implementation, this would:
        # 1. Immediately stop all training processes
        # 2. Save forensic data
        # 3. Alert security team
        # 4. Generate compliance report
        
        return True  # Emergency stop is considered successful
    
    def _graceful_degradation(self, metadata: Dict[str, Any]) -> bool:
        """Implement graceful degradation for partial failures."""
        failed_components = metadata.get("failed_components", [])
        
        logger.info(f"Implementing graceful degradation for failed components: {failed_components}")
        
        # Disable non-essential components
        degraded_mode = {
            "monitoring_disabled": "monitoring" in failed_components,
            "logging_reduced": "logging" in failed_components,
            "checkpointing_reduced": "storage" in failed_components
        }
        
        # Continue training with reduced functionality
        logger.info(f"Operating in degraded mode: {degraded_mode}")
        
        return True
    
    def _validate_recovery_point(self, recovery_point: RecoveryPoint) -> bool:
        """Validate integrity of recovery point."""
        try:
            # Check data hash integrity
            state_data = {
                "model_state": recovery_point.model_state,
                "optimizer_state": recovery_point.optimizer_state,
                "privacy_state": recovery_point.privacy_state,
                "training_metrics": recovery_point.training_metrics,
                "system_state": recovery_point.system_state
            }
            expected_hash = hashlib.sha256(json.dumps(state_data, sort_keys=True).encode()).hexdigest()
            
            if expected_hash != recovery_point.data_hash:
                logger.error(f"Data hash mismatch for recovery point {recovery_point.recovery_id}")
                return False
            
            # Check if checkpoint file exists and is readable
            checkpoint_file = self.checkpoint_dir / f"{recovery_point.recovery_id}.pkl"
            if not checkpoint_file.exists():
                logger.error(f"Checkpoint file missing: {checkpoint_file}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Recovery point validation failed: {e}")
            return False
    
    def _restore_from_recovery_point(self, recovery_point: RecoveryPoint) -> bool:
        """Restore training state from recovery point."""
        try:
            # Load full checkpoint data
            checkpoint_file = self.checkpoint_dir / f"{recovery_point.recovery_id}.pkl"
            
            if checkpoint_file.exists():
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                # Validate loaded data matches recovery point
                if checkpoint_data.get("recovery_id") != recovery_point.recovery_id:
                    logger.error("Checkpoint data mismatch")
                    return False
                
                logger.info(f"Restored training state to epoch {recovery_point.epoch}, step {recovery_point.step}")
                
                # Trigger recovery callbacks
                for callback in self.recovery_callbacks.values():
                    try:
                        callback(recovery_point, checkpoint_data)
                    except Exception as e:
                        logger.error(f"Recovery callback failed: {e}")
                
                return True
            
            else:
                logger.error(f"Checkpoint file not found: {checkpoint_file}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restore from recovery point: {e}")
            return False
    
    def _save_recovery_point(self, recovery_point: RecoveryPoint) -> None:
        """Save recovery point to disk."""
        try:
            checkpoint_file = self.checkpoint_dir / f"{recovery_point.recovery_id}.pkl"
            
            checkpoint_data = {
                "recovery_point": recovery_point.to_dict(),
                "recovery_id": recovery_point.recovery_id,
                "model_state": recovery_point.model_state,
                "optimizer_state": recovery_point.optimizer_state,
                "privacy_state": recovery_point.privacy_state,
                "training_metrics": recovery_point.training_metrics,
                "system_state": recovery_point.system_state
            }
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Save metadata
            metadata_file = self.checkpoint_dir / f"{recovery_point.recovery_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(recovery_point.to_dict(), f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save recovery point: {e}")
    
    def _save_failure_event(self, failure_event: FailureEvent) -> None:
        """Save failure event for analysis."""
        try:
            failure_file = self.checkpoint_dir / "failure_history.jsonl"
            
            with open(failure_file, 'a') as f:
                f.write(json.dumps(failure_event.to_dict()) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save failure event: {e}")
    
    def register_recovery_callback(self, name: str, callback: Callable) -> None:
        """Register callback to be called during recovery."""
        self.recovery_callbacks[name] = callback
        logger.info(f"Registered recovery callback: {name}")
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery system statistics."""
        failure_by_type = {}
        successful_recoveries = 0
        
        for failure in self.failure_history:
            failure_type = failure.failure_type.value
            failure_by_type[failure_type] = failure_by_type.get(failure_type, 0) + 1
            
            if failure.recovery_success:
                successful_recoveries += 1
        
        recovery_rate = (successful_recoveries / len(self.failure_history) 
                        if self.failure_history else 0.0)
        
        return {
            "total_failures": len(self.failure_history),
            "successful_recoveries": successful_recoveries,
            "recovery_rate": recovery_rate,
            "failure_by_type": failure_by_type,
            "recovery_points_available": len(self.recovery_points),
            "auto_recovery_enabled": self.auto_recovery_enabled
        }
    
    def cleanup_old_recovery_points(self, max_age_hours: int = 24) -> None:
        """Clean up old recovery points to save disk space."""
        current_time = time.time()
        removed_count = 0
        
        recovery_points_to_remove = []
        
        for recovery_id, recovery_point in self.recovery_points.items():
            # Parse timestamp
            try:
                point_time = time.mktime(time.strptime(recovery_point.timestamp, "%Y%m%d_%H%M%S"))
                age_hours = (current_time - point_time) / 3600
                
                if age_hours > max_age_hours:
                    recovery_points_to_remove.append(recovery_id)
            
            except ValueError:
                logger.warning(f"Invalid timestamp format in recovery point {recovery_id}")
        
        # Remove old recovery points
        for recovery_id in recovery_points_to_remove:
            try:
                # Remove files
                checkpoint_file = self.checkpoint_dir / f"{recovery_id}.pkl"
                metadata_file = self.checkpoint_dir / f"{recovery_id}.json"
                
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                if metadata_file.exists():
                    metadata_file.unlink()
                
                # Remove from memory
                del self.recovery_points[recovery_id]
                removed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to remove recovery point {recovery_id}: {e}")
        
        logger.info(f"Cleaned up {removed_count} old recovery points (older than {max_age_hours}h)")
    
    def test_recovery_system(self) -> Dict[str, Any]:
        """Test recovery system functionality."""
        logger.info("Testing recovery system...")
        
        # Create test recovery point
        test_recovery_id = self.create_recovery_point(
            epoch=0,
            step=0,
            privacy_state={"epsilon_spent": 0.1, "epsilon_total": 1.0},
            training_metrics={"loss": 2.5, "accuracy": 0.6}
        )
        
        # Test failure handling
        test_results = {}
        
        for failure_type in FailureType:
            logger.info(f"Testing recovery for {failure_type.value}")
            
            # Simulate failure and recovery
            recovery_success = self.handle_failure(
                failure_type=failure_type,
                description=f"Test failure: {failure_type.value}",
                affected_components=["test_component"],
                metadata={"test": True}
            )
            
            test_results[failure_type.value] = recovery_success
        
        logger.info("Recovery system test completed")
        return {
            "test_recovery_point": test_recovery_id,
            "recovery_tests": test_results,
            "system_statistics": self.get_recovery_statistics()
        }