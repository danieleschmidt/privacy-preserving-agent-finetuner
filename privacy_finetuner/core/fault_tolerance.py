"""Comprehensive fault tolerance system for privacy-preserving ML training.

This module provides advanced fault tolerance capabilities including automatic failover,
graceful degradation, distributed system resilience, and comprehensive error recovery
mechanisms for production-grade privacy-preserving machine learning systems.
"""

import time
import logging
import threading
import asyncio
import json
import pickle
import hashlib
from typing import Dict, Any, List, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import uuid
import random

from ..core.circuit_breaker import RobustExecutor, CircuitBreakerConfig, RetryConfig
from ..core.exceptions import (
    ModelTrainingException, ResourceExhaustedException, 
    ValidationException, SecurityViolationException
)

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Types of system failures."""
    TRANSIENT = "transient"           # Temporary failures that may resolve
    PERSISTENT = "persistent"         # Long-lasting failures requiring intervention
    CASCADING = "cascading"          # Failures that propagate to other components
    BYZANTINE = "byzantine"          # Unpredictable or malicious failures
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # System resource failures


class SystemState(Enum):
    """System operational states."""
    HEALTHY = "healthy"              # All systems operating normally
    DEGRADED = "degraded"           # Some components impaired but system functional
    CRITICAL = "critical"           # Significant impairment, limited functionality
    FAILING = "failing"             # System failing, emergency protocols active
    FAILED = "failed"               # System has failed, requires recovery


class ComponentStatus(Enum):
    """Status of individual system components."""
    ACTIVE = "active"               # Component is active and healthy
    STANDBY = "standby"             # Component is ready but not active
    DEGRADED = "degraded"           # Component is working but impaired
    FAILED = "failed"               # Component has failed
    RECOVERING = "recovering"       # Component is in recovery process
    MAINTENANCE = "maintenance"     # Component is under maintenance


@dataclass
class SystemComponent:
    """Represents a system component with fault tolerance capabilities."""
    id: str
    name: str
    component_type: str
    status: ComponentStatus = ComponentStatus.ACTIVE
    health_score: float = 1.0  # 0.0 (failed) to 1.0 (perfect health)
    last_health_check: datetime = field(default_factory=datetime.now)
    failure_count: int = 0
    recovery_attempts: int = 0
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if component is in a healthy state."""
        return self.status in [ComponentStatus.ACTIVE, ComponentStatus.STANDBY]
    
    def is_operational(self) -> bool:
        """Check if component is operational (can perform its function)."""
        return self.status in [ComponentStatus.ACTIVE, ComponentStatus.DEGRADED]


@dataclass
class FailoverTarget:
    """Represents a failover target for a component."""
    component_id: str
    priority: int  # Lower number = higher priority
    readiness_check: Optional[Callable[[], bool]] = None
    activation_callback: Optional[Callable[[], bool]] = None
    cost_factor: float = 1.0  # Relative cost of using this target
    capacity_factor: float = 1.0  # Relative capacity compared to primary


@dataclass
class FailureEvent:
    """Records a failure event in the system."""
    id: str
    timestamp: datetime
    component_id: str
    failure_mode: FailureMode
    severity: int  # 1 (low) to 10 (critical)
    description: str
    root_cause: Optional[str] = None
    affected_components: Set[str] = field(default_factory=set)
    recovery_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class HealthMonitor:
    """Monitors system and component health with predictive capabilities."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.components: Dict[str, SystemComponent] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.monitoring_active = False
        self.monitor_thread = None
        self.health_checks: Dict[str, Callable[[], float]] = {}
        self.predictive_models: Dict[str, Any] = {}
        self.lock = threading.RLock()
        
    def register_component(
        self, 
        component_id: str, 
        name: str, 
        component_type: str,
        health_check: Optional[Callable[[], float]] = None,
        dependencies: Set[str] = None
    ) -> SystemComponent:
        """Register a component for monitoring."""
        with self.lock:
            component = SystemComponent(
                id=component_id,
                name=name,
                component_type=component_type,
                dependencies=dependencies or set()
            )
            
            self.components[component_id] = component
            
            if health_check:
                self.health_checks[component_id] = health_check
            
            # Update dependency graph
            if dependencies:
                for dep_id in dependencies:
                    if dep_id in self.components:
                        self.components[dep_id].dependents.add(component_id)
            
            logger.info(f"Registered component {name} ({component_id}) for health monitoring")
            return component
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._perform_health_checks()
                self._update_predictive_models()
                self._detect_health_trends()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval * 2)
    
    def _perform_health_checks(self):
        """Perform health checks on all registered components."""
        with self.lock:
            for component_id, component in self.components.items():
                try:
                    if component_id in self.health_checks:
                        # Use custom health check
                        health_score = self.health_checks[component_id]()
                    else:
                        # Use default health check
                        health_score = self._default_health_check(component)
                    
                    # Update component health
                    component.health_score = max(0.0, min(1.0, health_score))
                    component.last_health_check = datetime.now()
                    
                    # Record health history
                    self.health_history[component_id].append({
                        'timestamp': component.last_health_check,
                        'health_score': health_score,
                        'status': component.status.value
                    })
                    
                    # Update component status based on health score
                    self._update_component_status(component)
                    
                except Exception as e:
                    logger.warning(f"Health check failed for {component_id}: {e}")
                    component.failure_count += 1
                    component.health_score = max(0.0, component.health_score - 0.1)
    
    def _default_health_check(self, component: SystemComponent) -> float:
        """Default health check for components."""
        # Basic health check based on failure count and last update
        age_minutes = (datetime.now() - component.last_health_check).total_seconds() / 60
        
        health_score = 1.0
        
        # Penalize for failures
        if component.failure_count > 0:
            health_score -= min(0.8, component.failure_count * 0.1)
        
        # Penalize for staleness
        if age_minutes > 60:  # More than 1 hour old
            health_score -= min(0.5, (age_minutes - 60) / 120 * 0.5)
        
        return max(0.0, health_score)
    
    def _update_component_status(self, component: SystemComponent):
        """Update component status based on health score."""
        old_status = component.status
        
        if component.health_score >= 0.8:
            component.status = ComponentStatus.ACTIVE
        elif component.health_score >= 0.5:
            component.status = ComponentStatus.DEGRADED
        elif component.health_score >= 0.2:
            component.status = ComponentStatus.RECOVERING
        else:
            component.status = ComponentStatus.FAILED
        
        # Log status changes
        if old_status != component.status:
            logger.warning(f"Component {component.id} status changed: {old_status.value} -> {component.status.value}")
    
    def _update_predictive_models(self):
        """Update predictive models for failure prediction."""
        # Simple trend-based prediction
        for component_id, history in self.health_history.items():
            if len(history) >= 10:
                recent_scores = [h['health_score'] for h in list(history)[-10:]]
                trend = self._calculate_trend(recent_scores)
                
                self.predictive_models[component_id] = {
                    'trend': trend,
                    'predicted_failure_time': self._predict_failure_time(recent_scores, trend)
                }
    
    def _calculate_trend(self, scores: List[float]) -> float:
        """Calculate health trend (positive = improving, negative = degrading)."""
        if len(scores) < 2:
            return 0.0
        
        # Simple linear trend
        n = len(scores)
        sum_x = sum(range(n))
        sum_y = sum(scores)
        sum_xy = sum(i * score for i, score in enumerate(scores))
        sum_x2 = sum(i * i for i in range(n))
        
        trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return trend
    
    def _predict_failure_time(self, scores: List[float], trend: float) -> Optional[datetime]:
        """Predict when component might fail based on trend."""
        if trend >= 0:  # Health improving or stable
            return None
        
        current_score = scores[-1]
        if current_score <= 0.2:  # Already critical
            return datetime.now()
        
        # Estimate time to reach critical threshold (0.2)
        steps_to_failure = (current_score - 0.2) / abs(trend)
        failure_time = datetime.now() + timedelta(seconds=steps_to_failure * self.check_interval)
        
        return failure_time
    
    def _detect_health_trends(self):
        """Detect concerning health trends and trigger alerts."""
        for component_id, model in self.predictive_models.items():
            predicted_failure = model.get('predicted_failure_time')
            
            if predicted_failure and predicted_failure < datetime.now() + timedelta(hours=1):
                logger.warning(f"Component {component_id} predicted to fail within 1 hour")
                # Could trigger preemptive failover here
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        with self.lock:
            total_components = len(self.components)
            if total_components == 0:
                return {'overall_health': 1.0, 'status': 'unknown', 'components': 0}
            
            healthy_count = sum(1 for c in self.components.values() if c.is_healthy())
            operational_count = sum(1 for c in self.components.values() if c.is_operational())
            
            overall_health = sum(c.health_score for c in self.components.values()) / total_components
            
            if overall_health >= 0.8 and healthy_count == total_components:
                status = SystemState.HEALTHY
            elif overall_health >= 0.6 and operational_count >= total_components * 0.8:
                status = SystemState.DEGRADED
            elif operational_count >= total_components * 0.5:
                status = SystemState.CRITICAL
            elif operational_count > 0:
                status = SystemState.FAILING
            else:
                status = SystemState.FAILED
            
            return {
                'overall_health': overall_health,
                'status': status.value,
                'total_components': total_components,
                'healthy_components': healthy_count,
                'operational_components': operational_count,
                'failed_components': total_components - operational_count,
                'component_details': {
                    cid: {
                        'name': c.name,
                        'health_score': c.health_score,
                        'status': c.status.value,
                        'failure_count': c.failure_count
                    } for cid, c in self.components.items()
                }
            }


class FailoverManager:
    """Manages automatic failover and recovery operations."""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        self.failover_targets: Dict[str, List[FailoverTarget]] = {}
        self.active_failovers: Dict[str, str] = {}  # component_id -> active_target_id
        self.failover_history: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        
        # Failover policies
        self.auto_failover_enabled = True
        self.failover_threshold = 0.3  # Health score threshold for failover
        self.recovery_threshold = 0.7  # Health score threshold for failback
        self.max_failover_attempts = 3
    
    def register_failover_target(
        self,
        primary_component_id: str,
        target: FailoverTarget
    ):
        """Register a failover target for a component."""
        with self.lock:
            if primary_component_id not in self.failover_targets:
                self.failover_targets[primary_component_id] = []
            
            self.failover_targets[primary_component_id].append(target)
            # Sort by priority (lower number = higher priority)
            self.failover_targets[primary_component_id].sort(key=lambda x: x.priority)
            
            logger.info(f"Registered failover target {target.component_id} for {primary_component_id}")
    
    def check_failover_conditions(self) -> List[str]:
        """Check all components for failover conditions."""
        components_needing_failover = []
        
        with self.lock:
            for component_id, component in self.health_monitor.components.items():
                # Skip if already failed over
                if component_id in self.active_failovers:
                    continue
                
                # Check if component needs failover
                if (component.health_score < self.failover_threshold and
                    component_id in self.failover_targets and
                    self.auto_failover_enabled):
                    
                    components_needing_failover.append(component_id)
        
        return components_needing_failover
    
    def execute_failover(self, component_id: str) -> bool:
        """Execute failover for a specific component."""
        with self.lock:
            if component_id not in self.failover_targets:
                logger.error(f"No failover targets configured for {component_id}")
                return False
            
            if component_id in self.active_failovers:
                logger.warning(f"Failover already active for {component_id}")
                return False
            
            # Try failover targets in priority order
            for target in self.failover_targets[component_id]:
                try:
                    if self._attempt_failover(component_id, target):
                        self.active_failovers[component_id] = target.component_id
                        
                        # Record failover event
                        self.failover_history.append({
                            'timestamp': datetime.now(),
                            'primary_component': component_id,
                            'target_component': target.component_id,
                            'reason': 'automatic_health_check',
                            'success': True
                        })
                        
                        logger.info(f"Successful failover: {component_id} -> {target.component_id}")
                        return True
                
                except Exception as e:
                    logger.error(f"Failover attempt failed for {component_id} -> {target.component_id}: {e}")
                    continue
            
            # Record failed failover
            self.failover_history.append({
                'timestamp': datetime.now(),
                'primary_component': component_id,
                'target_component': None,
                'reason': 'automatic_health_check',
                'success': False,
                'error': 'All failover targets failed'
            })
            
            logger.error(f"All failover attempts failed for {component_id}")
            return False
    
    def _attempt_failover(self, component_id: str, target: FailoverTarget) -> bool:
        """Attempt failover to a specific target."""
        # Check target readiness
        if target.readiness_check and not target.readiness_check():
            logger.warning(f"Failover target {target.component_id} not ready")
            return False
        
        # Activate target
        if target.activation_callback:
            if not target.activation_callback():
                logger.error(f"Failed to activate failover target {target.component_id}")
                return False
        
        # Update component status
        if target.component_id in self.health_monitor.components:
            self.health_monitor.components[target.component_id].status = ComponentStatus.ACTIVE
        
        return True
    
    def check_failback_conditions(self) -> List[str]:
        """Check for components that can fail back to primary."""
        components_for_failback = []
        
        with self.lock:
            for component_id, active_target_id in self.active_failovers.items():
                primary_component = self.health_monitor.components.get(component_id)
                
                if (primary_component and 
                    primary_component.health_score >= self.recovery_threshold):
                    components_for_failback.append(component_id)
        
        return components_for_failback
    
    def execute_failback(self, component_id: str) -> bool:
        """Execute failback to primary component."""
        with self.lock:
            if component_id not in self.active_failovers:
                logger.warning(f"No active failover for {component_id}")
                return False
            
            active_target_id = self.active_failovers[component_id]
            
            try:
                # Reactivate primary component
                primary_component = self.health_monitor.components.get(component_id)
                if primary_component:
                    primary_component.status = ComponentStatus.ACTIVE
                
                # Deactivate failover target
                target_component = self.health_monitor.components.get(active_target_id)
                if target_component:
                    target_component.status = ComponentStatus.STANDBY
                
                # Remove from active failovers
                del self.active_failovers[component_id]
                
                # Record failback event
                self.failover_history.append({
                    'timestamp': datetime.now(),
                    'primary_component': component_id,
                    'target_component': active_target_id,
                    'reason': 'automatic_failback',
                    'success': True,
                    'action': 'failback'
                })
                
                logger.info(f"Successful failback: {active_target_id} -> {component_id}")
                return True
            
            except Exception as e:
                logger.error(f"Failback failed for {component_id}: {e}")
                return False
    
    def get_failover_status(self) -> Dict[str, Any]:
        """Get current failover status."""
        with self.lock:
            return {
                'auto_failover_enabled': self.auto_failover_enabled,
                'active_failovers': self.active_failovers.copy(),
                'total_failover_targets': sum(len(targets) for targets in self.failover_targets.values()),
                'failover_history_count': len(self.failover_history),
                'recent_failovers': self.failover_history[-10:] if self.failover_history else []
            }


class GracefulDegradationManager:
    """Manages graceful degradation of system functionality."""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        self.degradation_policies: Dict[str, Dict[str, Any]] = {}
        self.active_degradations: Dict[str, Dict[str, Any]] = {}
        self.feature_toggles: Dict[str, bool] = {}
        self.performance_limits: Dict[str, Dict[str, float]] = {}
        self.lock = threading.RLock()
    
    def register_degradation_policy(
        self,
        policy_id: str,
        trigger_conditions: Dict[str, Any],
        degradation_actions: List[Dict[str, Any]],
        recovery_conditions: Dict[str, Any]
    ):
        """Register a graceful degradation policy."""
        with self.lock:
            self.degradation_policies[policy_id] = {
                'trigger_conditions': trigger_conditions,
                'degradation_actions': degradation_actions,
                'recovery_conditions': recovery_conditions,
                'created_at': datetime.now()
            }
            
            logger.info(f"Registered degradation policy: {policy_id}")
    
    def check_degradation_conditions(self) -> List[str]:
        """Check for conditions that trigger graceful degradation."""
        policies_to_activate = []
        
        with self.lock:
            system_health = self.health_monitor.get_system_health()
            
            for policy_id, policy in self.degradation_policies.items():
                if policy_id in self.active_degradations:
                    continue  # Already active
                
                if self._evaluate_conditions(policy['trigger_conditions'], system_health):
                    policies_to_activate.append(policy_id)
        
        return policies_to_activate
    
    def activate_degradation(self, policy_id: str) -> bool:
        """Activate a graceful degradation policy."""
        with self.lock:
            if policy_id not in self.degradation_policies:
                logger.error(f"Unknown degradation policy: {policy_id}")
                return False
            
            if policy_id in self.active_degradations:
                logger.warning(f"Degradation policy already active: {policy_id}")
                return True
            
            policy = self.degradation_policies[policy_id]
            
            try:
                # Execute degradation actions
                executed_actions = []
                for action in policy['degradation_actions']:
                    if self._execute_degradation_action(action):
                        executed_actions.append(action)
                
                # Record active degradation
                self.active_degradations[policy_id] = {
                    'activated_at': datetime.now(),
                    'executed_actions': executed_actions,
                    'policy': policy
                }
                
                logger.warning(f"Activated graceful degradation policy: {policy_id}")
                return True
            
            except Exception as e:
                logger.error(f"Failed to activate degradation policy {policy_id}: {e}")
                return False
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], system_health: Dict[str, Any]) -> bool:
        """Evaluate whether conditions are met."""
        for condition_type, condition_value in conditions.items():
            if condition_type == 'overall_health_below':
                if system_health['overall_health'] >= condition_value:
                    return False
            elif condition_type == 'operational_components_below':
                if system_health['operational_components'] >= condition_value:
                    return False
            elif condition_type == 'failed_components_above':
                if system_health['failed_components'] <= condition_value:
                    return False
            elif condition_type == 'component_health_below':
                component_id, threshold = condition_value
                component = self.health_monitor.components.get(component_id)
                if not component or component.health_score >= threshold:
                    return False
        
        return True
    
    def _execute_degradation_action(self, action: Dict[str, Any]) -> bool:
        """Execute a specific degradation action."""
        action_type = action['type']
        
        try:
            if action_type == 'disable_feature':
                feature_name = action['feature']
                self.feature_toggles[feature_name] = False
                logger.info(f"Disabled feature: {feature_name}")
                
            elif action_type == 'reduce_performance':
                component = action['component']
                limits = action['limits']
                self.performance_limits[component] = limits
                logger.info(f"Applied performance limits to {component}: {limits}")
                
            elif action_type == 'increase_cache_ttl':
                # Placeholder for cache TTL increase
                logger.info(f"Increased cache TTL for {action.get('cache', 'default')}")
                
            elif action_type == 'reduce_precision':
                # Placeholder for precision reduction
                logger.info(f"Reduced precision for {action.get('component', 'default')}")
                
            elif action_type == 'enable_batch_processing':
                # Placeholder for batch processing
                logger.info("Enabled batch processing mode")
                
            else:
                logger.warning(f"Unknown degradation action type: {action_type}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute degradation action {action_type}: {e}")
            return False
    
    def check_recovery_conditions(self) -> List[str]:
        """Check for conditions that allow recovery from degradation."""
        policies_to_recover = []
        
        with self.lock:
            system_health = self.health_monitor.get_system_health()
            
            for policy_id, active_degradation in self.active_degradations.items():
                recovery_conditions = active_degradation['policy']['recovery_conditions']
                
                if self._evaluate_conditions(recovery_conditions, system_health):
                    policies_to_recover.append(policy_id)
        
        return policies_to_recover
    
    def recover_from_degradation(self, policy_id: str) -> bool:
        """Recover from graceful degradation."""
        with self.lock:
            if policy_id not in self.active_degradations:
                logger.warning(f"No active degradation for policy: {policy_id}")
                return False
            
            active_degradation = self.active_degradations[policy_id]
            
            try:
                # Reverse degradation actions
                for action in active_degradation['executed_actions']:
                    self._reverse_degradation_action(action)
                
                # Remove active degradation
                del self.active_degradations[policy_id]
                
                logger.info(f"Recovered from graceful degradation policy: {policy_id}")
                return True
            
            except Exception as e:
                logger.error(f"Failed to recover from degradation policy {policy_id}: {e}")
                return False
    
    def _reverse_degradation_action(self, action: Dict[str, Any]):
        """Reverse a degradation action."""
        action_type = action['type']
        
        if action_type == 'disable_feature':
            feature_name = action['feature']
            self.feature_toggles[feature_name] = True
            logger.info(f"Re-enabled feature: {feature_name}")
            
        elif action_type == 'reduce_performance':
            component = action['component']
            if component in self.performance_limits:
                del self.performance_limits[component]
                logger.info(f"Removed performance limits from {component}")
        
        # Add other reversal logic as needed
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is currently enabled."""
        return self.feature_toggles.get(feature_name, True)
    
    def get_performance_limits(self, component: str) -> Optional[Dict[str, float]]:
        """Get current performance limits for a component."""
        return self.performance_limits.get(component)
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        with self.lock:
            return {
                'active_degradations': len(self.active_degradations),
                'disabled_features': [k for k, v in self.feature_toggles.items() if not v],
                'performance_limited_components': list(self.performance_limits.keys()),
                'degradation_details': {
                    pid: {
                        'activated_at': deg['activated_at'].isoformat(),
                        'actions_count': len(deg['executed_actions'])
                    } for pid, deg in self.active_degradations.items()
                }
            }


class ComprehensiveFaultToleranceSystem:
    """Main fault tolerance system that orchestrates all components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize core components
        self.health_monitor = HealthMonitor(
            check_interval=self.config.get('health_check_interval', 30.0)
        )
        self.failover_manager = FailoverManager(self.health_monitor)
        self.degradation_manager = GracefulDegradationManager(self.health_monitor)
        
        # System state
        self.system_state = SystemState.HEALTHY
        self.last_state_change = datetime.now()
        self.fault_tolerance_active = False
        
        # Fault tolerance policies
        self.auto_recovery_enabled = self.config.get('auto_recovery_enabled', True)
        self.failover_enabled = self.config.get('failover_enabled', True)
        self.degradation_enabled = self.config.get('degradation_enabled', True)
        
        # Monitoring and control thread
        self.control_thread = None
        self.control_active = False
        
        logger.info("Comprehensive fault tolerance system initialized")
    
    def start_fault_tolerance(self):
        """Start the fault tolerance system."""
        if self.fault_tolerance_active:
            return
        
        self.fault_tolerance_active = True
        
        # Start health monitoring
        self.health_monitor.start_monitoring()
        
        # Start control loop
        self.control_active = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        logger.info("Fault tolerance system started")
    
    def stop_fault_tolerance(self):
        """Stop the fault tolerance system."""
        self.fault_tolerance_active = False
        self.control_active = False
        
        # Stop health monitoring
        self.health_monitor.stop_monitoring()
        
        # Stop control thread
        if self.control_thread:
            self.control_thread.join(timeout=5.0)
        
        logger.info("Fault tolerance system stopped")
    
    def _control_loop(self):
        """Main control loop for fault tolerance operations."""
        while self.control_active:
            try:
                # Update system state
                self._update_system_state()
                
                # Check for failover conditions
                if self.failover_enabled:
                    components_needing_failover = self.failover_manager.check_failover_conditions()
                    for component_id in components_needing_failover:
                        self.failover_manager.execute_failover(component_id)
                    
                    # Check for failback conditions
                    components_for_failback = self.failover_manager.check_failback_conditions()
                    for component_id in components_for_failback:
                        self.failover_manager.execute_failback(component_id)
                
                # Check for degradation conditions
                if self.degradation_enabled:
                    policies_to_activate = self.degradation_manager.check_degradation_conditions()
                    for policy_id in policies_to_activate:
                        self.degradation_manager.activate_degradation(policy_id)
                    
                    # Check for recovery conditions
                    policies_to_recover = self.degradation_manager.check_recovery_conditions()
                    for policy_id in policies_to_recover:
                        self.degradation_manager.recover_from_degradation(policy_id)
                
                time.sleep(10)  # Control loop interval
                
            except Exception as e:
                logger.error(f"Fault tolerance control loop error: {e}")
                time.sleep(30)  # Back off on errors
    
    def _update_system_state(self):
        """Update overall system state based on component health."""
        system_health = self.health_monitor.get_system_health()
        new_state = SystemState(system_health['status'])
        
        if new_state != self.system_state:
            old_state = self.system_state
            self.system_state = new_state
            self.last_state_change = datetime.now()
            
            logger.warning(f"System state changed: {old_state.value} -> {new_state.value}")
            
            # Trigger state change callbacks if configured
            self._handle_state_change(old_state, new_state)
    
    def _handle_state_change(self, old_state: SystemState, new_state: SystemState):
        """Handle system state changes."""
        if new_state in [SystemState.CRITICAL, SystemState.FAILING, SystemState.FAILED]:
            logger.critical(f"System entering critical state: {new_state.value}")
            
            # Could trigger additional emergency procedures here
            if new_state == SystemState.FAILED and self.auto_recovery_enabled:
                self._initiate_emergency_recovery()
    
    def _initiate_emergency_recovery(self):
        """Initiate emergency recovery procedures."""
        logger.critical("Initiating emergency recovery procedures")
        
        # Emergency recovery actions
        emergency_actions = [
            "Save current state to persistent storage",
            "Activate all available backup systems",
            "Enable maximum graceful degradation",
            "Alert system administrators",
            "Prepare for system restart if necessary"
        ]
        
        for action in emergency_actions:
            logger.critical(f"Emergency action: {action}")
            # Implement actual emergency actions here
    
    def register_component_with_failover(
        self,
        component_id: str,
        name: str,
        component_type: str,
        health_check: Optional[Callable[[], float]] = None,
        failover_targets: Optional[List[FailoverTarget]] = None,
        dependencies: Set[str] = None
    ) -> SystemComponent:
        """Register a component with failover capabilities."""
        # Register with health monitor
        component = self.health_monitor.register_component(
            component_id, name, component_type, health_check, dependencies
        )
        
        # Register failover targets
        if failover_targets:
            for target in failover_targets:
                self.failover_manager.register_failover_target(component_id, target)
        
        return component
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive fault tolerance system status."""
        system_health = self.health_monitor.get_system_health()
        failover_status = self.failover_manager.get_failover_status()
        degradation_status = self.degradation_manager.get_degradation_status()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'fault_tolerance_active': self.fault_tolerance_active,
            'system_state': self.system_state.value,
            'last_state_change': self.last_state_change.isoformat(),
            'system_health': system_health,
            'failover_status': failover_status,
            'degradation_status': degradation_status,
            'configuration': {
                'auto_recovery_enabled': self.auto_recovery_enabled,
                'failover_enabled': self.failover_enabled,
                'degradation_enabled': self.degradation_enabled
            }
        }
    
    def simulate_failure(self, component_id: str, failure_type: str = "health_degradation") -> bool:
        """Simulate a failure for testing purposes."""
        if component_id not in self.health_monitor.components:
            logger.error(f"Component {component_id} not found for failure simulation")
            return False
        
        component = self.health_monitor.components[component_id]
        
        if failure_type == "health_degradation":
            component.health_score = 0.1
            component.failure_count += 1
        elif failure_type == "complete_failure":
            component.health_score = 0.0
            component.status = ComponentStatus.FAILED
            component.failure_count += 1
        
        logger.warning(f"Simulated {failure_type} for component {component_id}")
        return True


# Global fault tolerance system instance
fault_tolerance_system = ComprehensiveFaultToleranceSystem()