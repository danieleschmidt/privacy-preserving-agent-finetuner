"""Intelligent auto-scaling system for privacy-preserving ML workloads.

This module provides advanced auto-scaling capabilities that are privacy-aware:
- Dynamic resource allocation based on privacy budget consumption
- Predictive scaling using workload patterns
- Multi-dimensional scaling (CPU, memory, GPU, privacy budget)
- Cost-optimization with privacy constraints
- Adaptive batch sizing for optimal privacy-utility tradeoff
- Federated learning node orchestration
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
from queue import Queue
import hashlib
from enum import Enum
import math

# Handle imports gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    
    class NumpyStub:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        
        @staticmethod
        def std(data):
            if not data:
                return 0
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            return variance ** 0.5
        
        @staticmethod
        def percentile(data, p):
            if not data:
                return 0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p / 100
            f = int(k)
            c = k - f
            if f == len(sorted_data) - 1:
                return sorted_data[f]
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    np = NumpyStub()

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ResourceType(Enum):
    """Resource types for scaling."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    PRIVACY_BUDGET = "privacy_budget"
    NETWORK = "network"
    STORAGE = "storage"


class ScalingTrigger(Enum):
    """Scaling trigger types."""
    THRESHOLD = "threshold"
    PREDICTIVE = "predictive"
    PRIVACY_BUDGET = "privacy_budget"
    COST_OPTIMIZATION = "cost_optimization"
    SLA_VIOLATION = "sla_violation"
    CUSTOM = "custom"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    network_io: float
    storage_io: float
    privacy_budget_consumption_rate: float
    request_rate: float
    response_time_p95: float
    error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""
    name: str
    resource_type: ResourceType
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    cooldown_minutes: int = 5
    min_instances: int = 1
    max_instances: int = 10
    privacy_aware: bool = True
    cost_weight: float = 1.0
    sla_weight: float = 1.0
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "resource_type": self.resource_type.value,
            "trigger": self.trigger.value
        }


@dataclass
class ScalingAction:
    """Scaling action to be executed."""
    timestamp: datetime
    rule_name: str
    resource_type: ResourceType
    direction: ScalingDirection
    current_capacity: int
    target_capacity: int
    reason: str
    confidence: float
    estimated_cost_impact: float
    privacy_budget_impact: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat(),
            "resource_type": self.resource_type.value,
            "direction": self.direction.value
        }


@dataclass
class WorkloadPrediction:
    """Workload prediction for proactive scaling."""
    timestamp: datetime
    prediction_horizon_minutes: int
    predicted_cpu_usage: float
    predicted_memory_usage: float
    predicted_request_rate: float
    predicted_privacy_consumption: float
    confidence_score: float
    factors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


class IntelligentAutoScaler:
    """Intelligent auto-scaling system for privacy-preserving ML workloads.
    
    Features:
    - Multi-dimensional resource scaling with privacy awareness
    - Predictive scaling using ML-based workload forecasting
    - Cost-optimization with SLA constraints
    - Adaptive batch sizing for privacy-utility optimization
    - Federated learning node orchestration
    - Real-time monitoring and alerting
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        resource_manager: Optional[Any] = None,
        privacy_tracker: Optional[Any] = None
    ):
        """Initialize intelligent auto-scaler.
        
        Args:
            config: Scaling configuration
            resource_manager: Resource management interface
            privacy_tracker: Privacy budget tracker
        """
        self.config = config or {}
        self.resource_manager = resource_manager or MockResourceManager()
        self.privacy_tracker = privacy_tracker
        
        # Initialize components
        self.workload_predictor = WorkloadPredictor()
        self.cost_optimizer = CostOptimizer()
        self.privacy_aware_scaler = PrivacyAwareScaler(privacy_tracker)
        self.batch_size_optimizer = BatchSizeOptimizer()
        self.federated_orchestrator = FederatedOrchestrator()
        
        # State management
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.current_metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_usage=0.0, memory_usage=0.0, gpu_usage=0.0,
            network_io=0.0, storage_io=0.0,
            privacy_budget_consumption_rate=0.0,
            request_rate=0.0, response_time_p95=0.0, error_rate=0.0
        )
        
        # History tracking
        self.metrics_history = []
        self.scaling_history = []
        self.predictions = []
        
        # Threading
        self.is_running = False
        self.monitoring_thread = None
        self.scaling_thread = None
        self.prediction_thread = None
        
        # Default scaling rules
        self._setup_default_rules()
        
        logger.info("Intelligent auto-scaler initialized")
    
    def start(self):
        """Start auto-scaling system."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring threads
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        
        self.monitoring_thread.start()
        self.scaling_thread.start()
        self.prediction_thread.start()
        
        logger.info("Auto-scaler started")
    
    def stop(self):
        """Stop auto-scaling system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for threads to complete
        for thread in [self.monitoring_thread, self.scaling_thread, self.prediction_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        
        logger.info("Auto-scaler stopped")
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a scaling rule."""
        self.scaling_rules[rule.name] = rule
        logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str):
        """Remove a scaling rule."""
        if rule_name in self.scaling_rules:
            del self.scaling_rules[rule_name]
            logger.info(f"Removed scaling rule: {rule_name}")
    
    def get_scaling_recommendations(self) -> List[ScalingAction]:
        """Get current scaling recommendations."""
        recommendations = []
        
        current_time = datetime.now()
        
        for rule_name, rule in self.scaling_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if self._is_in_cooldown(rule_name):
                continue
            
            # Evaluate scaling rule
            action = self._evaluate_scaling_rule(rule)
            
            if action and action.direction != ScalingDirection.STABLE:
                recommendations.append(action)
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        return recommendations
    
    def execute_scaling_action(self, action: ScalingAction) -> bool:
        """Execute a scaling action."""
        try:
            logger.info(f"Executing scaling action: {action.rule_name} - {action.direction.value}")
            
            # Execute through resource manager
            success = self.resource_manager.scale_resource(
                resource_type=action.resource_type.value,
                target_capacity=action.target_capacity,
                current_capacity=action.current_capacity
            )
            
            if success:
                # Record scaling action
                self.scaling_history.append(action)
                
                # Update privacy budget if needed
                if action.privacy_budget_impact != 0 and self.privacy_tracker:
                    self.privacy_tracker.adjust_budget(action.privacy_budget_impact)
                
                logger.info(f"Scaling action executed successfully: {action.rule_name}")
                return True
            else:
                logger.error(f"Scaling action failed: {action.rule_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing scaling action: {e}")
            return False
    
    def get_workload_prediction(
        self,
        horizon_minutes: int = 30
    ) -> Optional[WorkloadPrediction]:
        """Get workload prediction for specified horizon."""
        try:
            prediction = self.workload_predictor.predict(
                metrics_history=self.metrics_history,
                horizon_minutes=horizon_minutes
            )
            
            if prediction:
                self.predictions.append(prediction)
                
                # Keep only recent predictions
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.predictions = [
                    p for p in self.predictions
                    if p.timestamp >= cutoff_time
                ]
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating workload prediction: {e}")
            return None
    
    def optimize_batch_size(
        self,
        current_batch_size: int,
        privacy_budget_remaining: float,
        target_accuracy: float = 0.9
    ) -> int:
        """Optimize batch size for privacy-utility tradeoff."""
        try:
            optimal_batch_size = self.batch_size_optimizer.optimize(
                current_batch_size=current_batch_size,
                privacy_budget_remaining=privacy_budget_remaining,
                target_accuracy=target_accuracy,
                current_metrics=self.current_metrics
            )
            
            logger.info(f"Optimized batch size: {current_batch_size} -> {optimal_batch_size}")
            return optimal_batch_size
            
        except Exception as e:
            logger.error(f"Error optimizing batch size: {e}")
            return current_batch_size
    
    def get_federated_scaling_plan(
        self,
        num_clients: int,
        data_distribution: Dict[str, float],
        privacy_requirements: Dict[str, float]
    ) -> Dict[str, Any]:
        """Get federated learning scaling plan."""
        try:
            scaling_plan = self.federated_orchestrator.create_scaling_plan(
                num_clients=num_clients,
                data_distribution=data_distribution,
                privacy_requirements=privacy_requirements,
                current_metrics=self.current_metrics
            )
            
            return scaling_plan
            
        except Exception as e:
            logger.error(f"Error creating federated scaling plan: {e}")
            return {}
    
    def get_cost_optimization_report(self) -> Dict[str, Any]:
        """Get cost optimization report."""
        try:
            report = self.cost_optimizer.generate_report(
                scaling_history=self.scaling_history,
                current_metrics=self.current_metrics,
                predictions=self.predictions
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating cost optimization report: {e}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "is_running": self.is_running,
            "timestamp": datetime.now().isoformat(),
            "current_metrics": self.current_metrics.to_dict(),
            "active_rules": len([r for r in self.scaling_rules.values() if r.enabled]),
            "recent_actions": len([
                a for a in self.scaling_history
                if (datetime.now() - a.timestamp) < timedelta(hours=1)
            ]),
            "prediction_accuracy": self._calculate_prediction_accuracy(),
            "cost_efficiency": self._calculate_cost_efficiency(),
            "privacy_budget_status": self._get_privacy_budget_status(),
            "scaling_rules": {
                name: rule.to_dict() for name, rule in self.scaling_rules.items()
            }
        }
        
        return status
    
    def _setup_default_rules(self):
        """Setup default scaling rules."""
        default_rules = [
            ScalingRule(
                name="cpu_threshold",
                resource_type=ResourceType.CPU,
                trigger=ScalingTrigger.THRESHOLD,
                scale_up_threshold=0.8,
                scale_down_threshold=0.3,
                cooldown_minutes=5
            ),
            ScalingRule(
                name="memory_threshold",
                resource_type=ResourceType.MEMORY,
                trigger=ScalingTrigger.THRESHOLD,
                scale_up_threshold=0.85,
                scale_down_threshold=0.4,
                cooldown_minutes=5
            ),
            ScalingRule(
                name="privacy_budget_aware",
                resource_type=ResourceType.PRIVACY_BUDGET,
                trigger=ScalingTrigger.PRIVACY_BUDGET,
                scale_up_threshold=0.7,  # Scale up when 70% consumed
                scale_down_threshold=0.3,
                cooldown_minutes=10,
                privacy_aware=True
            ),
            ScalingRule(
                name="predictive_scaling",
                resource_type=ResourceType.CPU,
                trigger=ScalingTrigger.PREDICTIVE,
                scale_up_threshold=0.6,
                scale_down_threshold=0.2,
                cooldown_minutes=15
            ),
            ScalingRule(
                name="sla_protection",
                resource_type=ResourceType.CPU,
                trigger=ScalingTrigger.SLA_VIOLATION,
                scale_up_threshold=2.0,  # Response time > 2s
                scale_down_threshold=0.5,
                cooldown_minutes=3
            )
        ]
        
        for rule in default_rules:
            self.scaling_rules[rule.name] = rule
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                self.current_metrics = metrics
                
                # Add to history
                self.metrics_history.append(metrics)
                
                # Keep only recent history
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(10)  # Collect metrics every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _scaling_loop(self):
        """Main scaling decision loop."""
        while self.is_running:
            try:
                # Get scaling recommendations
                recommendations = self.get_scaling_recommendations()
                
                # Execute top recommendation if confidence is high enough
                if recommendations:
                    top_recommendation = recommendations[0]
                    
                    if top_recommendation.confidence > 0.7:
                        self.execute_scaling_action(top_recommendation)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _prediction_loop(self):
        """Main prediction loop."""
        while self.is_running:
            try:
                # Generate predictions for different horizons
                for horizon in [15, 30, 60]:  # 15min, 30min, 1hr
                    prediction = self.get_workload_prediction(horizon)
                
                time.sleep(300)  # Generate predictions every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                time.sleep(600)  # Wait longer on error
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        try:
            # Get metrics from resource manager
            raw_metrics = self.resource_manager.get_metrics()
            
            # Get privacy budget metrics
            privacy_rate = 0.0
            if self.privacy_tracker:
                privacy_rate = self.privacy_tracker.get_consumption_rate()
            
            metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_usage=raw_metrics.get("cpu_usage", 0.0),
                memory_usage=raw_metrics.get("memory_usage", 0.0),
                gpu_usage=raw_metrics.get("gpu_usage", 0.0),
                network_io=raw_metrics.get("network_io", 0.0),
                storage_io=raw_metrics.get("storage_io", 0.0),
                privacy_budget_consumption_rate=privacy_rate,
                request_rate=raw_metrics.get("request_rate", 0.0),
                response_time_p95=raw_metrics.get("response_time_p95", 0.0),
                error_rate=raw_metrics.get("error_rate", 0.0)
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            # Return default metrics
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0, memory_usage=0.0, gpu_usage=0.0,
                network_io=0.0, storage_io=0.0,
                privacy_budget_consumption_rate=0.0,
                request_rate=0.0, response_time_p95=0.0, error_rate=0.0
            )
    
    def _evaluate_scaling_rule(self, rule: ScalingRule) -> Optional[ScalingAction]:
        """Evaluate a scaling rule and return action if needed."""
        try:
            current_value = self._get_metric_value(rule.resource_type)
            current_capacity = self.resource_manager.get_current_capacity(rule.resource_type.value)
            
            # Determine scaling direction
            direction = ScalingDirection.STABLE
            confidence = 0.0
            reason = ""
            
            if rule.trigger == ScalingTrigger.THRESHOLD:
                direction, confidence, reason = self._evaluate_threshold_rule(rule, current_value)
            
            elif rule.trigger == ScalingTrigger.PREDICTIVE:
                direction, confidence, reason = self._evaluate_predictive_rule(rule)
            
            elif rule.trigger == ScalingTrigger.PRIVACY_BUDGET:
                direction, confidence, reason = self._evaluate_privacy_rule(rule, current_value)
            
            elif rule.trigger == ScalingTrigger.SLA_VIOLATION:
                direction, confidence, reason = self._evaluate_sla_rule(rule, current_value)
            
            elif rule.trigger == ScalingTrigger.COST_OPTIMIZATION:
                direction, confidence, reason = self._evaluate_cost_rule(rule)
            
            if direction == ScalingDirection.STABLE:
                return None
            
            # Calculate target capacity
            if direction == ScalingDirection.UP:
                target_capacity = max(
                    rule.min_instances,
                    min(rule.max_instances, int(current_capacity * rule.scale_up_factor))
                )
            else:
                target_capacity = max(
                    rule.min_instances,
                    int(current_capacity * rule.scale_down_factor)
                )
            
            # Calculate impacts
            cost_impact = self._estimate_cost_impact(
                rule.resource_type, current_capacity, target_capacity
            )
            
            privacy_impact = self._estimate_privacy_impact(
                rule.resource_type, current_capacity, target_capacity
            )
            
            # Apply privacy awareness
            if rule.privacy_aware:
                confidence = self.privacy_aware_scaler.adjust_confidence(
                    confidence, rule.resource_type, direction, privacy_impact
                )
            
            action = ScalingAction(
                timestamp=datetime.now(),
                rule_name=rule.name,
                resource_type=rule.resource_type,
                direction=direction,
                current_capacity=current_capacity,
                target_capacity=target_capacity,
                reason=reason,
                confidence=confidence,
                estimated_cost_impact=cost_impact,
                privacy_budget_impact=privacy_impact
            )
            
            return action
            
        except Exception as e:
            logger.error(f"Error evaluating scaling rule {rule.name}: {e}")
            return None
    
    def _evaluate_threshold_rule(
        self,
        rule: ScalingRule,
        current_value: float
    ) -> Tuple[ScalingDirection, float, str]:
        """Evaluate threshold-based scaling rule."""
        if current_value > rule.scale_up_threshold:
            confidence = min(1.0, (current_value - rule.scale_up_threshold) / 0.2)
            return (
                ScalingDirection.UP,
                confidence,
                f"{rule.resource_type.value} usage {current_value:.2f} > {rule.scale_up_threshold}"
            )
        
        elif current_value < rule.scale_down_threshold:
            confidence = min(1.0, (rule.scale_down_threshold - current_value) / 0.2)
            return (
                ScalingDirection.DOWN,
                confidence,
                f"{rule.resource_type.value} usage {current_value:.2f} < {rule.scale_down_threshold}"
            )
        
        return ScalingDirection.STABLE, 0.0, "Within thresholds"
    
    def _evaluate_predictive_rule(
        self,
        rule: ScalingRule
    ) -> Tuple[ScalingDirection, float, str]:
        """Evaluate predictive scaling rule."""
        # Get prediction for next 30 minutes
        prediction = self.workload_predictor.predict(
            self.metrics_history, horizon_minutes=30
        )
        
        if not prediction:
            return ScalingDirection.STABLE, 0.0, "No prediction available"
        
        # Get predicted value for resource type
        if rule.resource_type == ResourceType.CPU:
            predicted_value = prediction.predicted_cpu_usage
        elif rule.resource_type == ResourceType.MEMORY:
            predicted_value = prediction.predicted_memory_usage
        else:
            return ScalingDirection.STABLE, 0.0, "Resource type not supported for prediction"
        
        # Evaluate against thresholds
        if predicted_value > rule.scale_up_threshold:
            confidence = prediction.confidence_score * 0.8  # Reduce confidence for predictions
            return (
                ScalingDirection.UP,
                confidence,
                f"Predicted {rule.resource_type.value} usage {predicted_value:.2f} > {rule.scale_up_threshold}"
            )
        
        elif predicted_value < rule.scale_down_threshold:
            confidence = prediction.confidence_score * 0.8
            return (
                ScalingDirection.DOWN,
                confidence,
                f"Predicted {rule.resource_type.value} usage {predicted_value:.2f} < {rule.scale_down_threshold}"
            )
        
        return ScalingDirection.STABLE, 0.0, "Predicted values within thresholds"
    
    def _evaluate_privacy_rule(
        self,
        rule: ScalingRule,
        current_value: float
    ) -> Tuple[ScalingDirection, float, str]:
        """Evaluate privacy budget-aware scaling rule."""
        if not self.privacy_tracker:
            return ScalingDirection.STABLE, 0.0, "No privacy tracker available"
        
        # Get privacy budget status
        budget_status = self.privacy_tracker.get_status()
        utilization = budget_status.utilization_percentage / 100.0
        
        # Scale up if budget consumption is high (need more resources to be more efficient)
        if utilization > rule.scale_up_threshold:
            confidence = min(1.0, (utilization - rule.scale_up_threshold) / 0.2)
            return (
                ScalingDirection.UP,
                confidence,
                f"Privacy budget utilization {utilization:.2f} > {rule.scale_up_threshold}"
            )
        
        # Scale down if budget consumption is low
        elif utilization < rule.scale_down_threshold:
            confidence = min(1.0, (rule.scale_down_threshold - utilization) / 0.2)
            return (
                ScalingDirection.DOWN,
                confidence,
                f"Privacy budget utilization {utilization:.2f} < {rule.scale_down_threshold}"
            )
        
        return ScalingDirection.STABLE, 0.0, "Privacy budget utilization within thresholds"
    
    def _evaluate_sla_rule(
        self,
        rule: ScalingRule,
        current_value: float
    ) -> Tuple[ScalingDirection, float, str]:
        """Evaluate SLA violation-based scaling rule."""
        # Check response time SLA
        response_time = self.current_metrics.response_time_p95
        error_rate = self.current_metrics.error_rate
        
        # SLA violation if response time > threshold or error rate > 5%
        sla_violated = response_time > rule.scale_up_threshold or error_rate > 0.05
        
        if sla_violated:
            confidence = 0.9  # High confidence for SLA violations
            return (
                ScalingDirection.UP,
                confidence,
                f"SLA violated: response_time={response_time:.2f}s, error_rate={error_rate:.3f}"
            )
        
        # Scale down if performance is very good
        elif response_time < rule.scale_down_threshold and error_rate < 0.01:
            confidence = 0.6  # Lower confidence for scale down
            return (
                ScalingDirection.DOWN,
                confidence,
                f"Performance excellent: response_time={response_time:.2f}s, error_rate={error_rate:.3f}"
            )
        
        return ScalingDirection.STABLE, 0.0, "SLA within bounds"
    
    def _evaluate_cost_rule(self, rule: ScalingRule) -> Tuple[ScalingDirection, float, str]:
        """Evaluate cost optimization-based scaling rule."""
        # Get cost optimization recommendation
        cost_recommendation = self.cost_optimizer.get_recommendation(
            current_metrics=self.current_metrics,
            resource_type=rule.resource_type
        )
        
        if not cost_recommendation:
            return ScalingDirection.STABLE, 0.0, "No cost recommendation available"
        
        direction = cost_recommendation.get("direction", "stable")
        confidence = cost_recommendation.get("confidence", 0.5)
        savings = cost_recommendation.get("estimated_savings", 0.0)
        
        if direction == "scale_down" and savings > 0:
            return (
                ScalingDirection.DOWN,
                confidence,
                f"Cost optimization: estimated savings ${savings:.2f}/hour"
            )
        elif direction == "scale_up":
            return (
                ScalingDirection.UP,
                confidence,
                "Cost optimization: scale up for better efficiency"
            )
        
        return ScalingDirection.STABLE, 0.0, "No cost optimization needed"
    
    def _get_metric_value(self, resource_type: ResourceType) -> float:
        """Get current metric value for resource type."""
        if resource_type == ResourceType.CPU:
            return self.current_metrics.cpu_usage
        elif resource_type == ResourceType.MEMORY:
            return self.current_metrics.memory_usage
        elif resource_type == ResourceType.GPU:
            return self.current_metrics.gpu_usage
        elif resource_type == ResourceType.PRIVACY_BUDGET:
            if self.privacy_tracker:
                status = self.privacy_tracker.get_status()
                return status.utilization_percentage / 100.0
            return 0.0
        elif resource_type == ResourceType.NETWORK:
            return self.current_metrics.network_io
        elif resource_type == ResourceType.STORAGE:
            return self.current_metrics.storage_io
        
        return 0.0
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Check if scaling rule is in cooldown period."""
        if rule_name not in self.scaling_rules:
            return False
        
        rule = self.scaling_rules[rule_name]
        cooldown_duration = timedelta(minutes=rule.cooldown_minutes)
        
        # Find last scaling action for this rule
        for action in reversed(self.scaling_history):
            if action.rule_name == rule_name:
                if datetime.now() - action.timestamp < cooldown_duration:
                    return True
                break
        
        return False
    
    def _estimate_cost_impact(
        self,
        resource_type: ResourceType,
        current_capacity: int,
        target_capacity: int
    ) -> float:
        """Estimate cost impact of scaling action."""
        # Simplified cost calculation
        cost_per_unit = {
            ResourceType.CPU: 0.10,  # $0.10/hour per CPU unit
            ResourceType.MEMORY: 0.02,  # $0.02/hour per GB
            ResourceType.GPU: 2.50,  # $2.50/hour per GPU
            ResourceType.STORAGE: 0.001,  # $0.001/hour per GB
        }.get(resource_type, 0.05)
        
        capacity_change = target_capacity - current_capacity
        hourly_cost_change = capacity_change * cost_per_unit
        
        return hourly_cost_change
    
    def _estimate_privacy_impact(
        self,
        resource_type: ResourceType,
        current_capacity: int,
        target_capacity: int
    ) -> float:
        """Estimate privacy budget impact of scaling action."""
        # Scaling up generally allows for better privacy (more noise, larger batches)
        # Scaling down may require more privacy budget consumption
        
        capacity_ratio = target_capacity / max(current_capacity, 1)
        
        if resource_type in [ResourceType.CPU, ResourceType.MEMORY]:
            # More compute resources -> better privacy efficiency
            if capacity_ratio > 1:
                return -0.1 * (capacity_ratio - 1)  # Negative = better privacy
            else:
                return 0.1 * (1 - capacity_ratio)  # Positive = worse privacy
        
        return 0.0
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy over recent history."""
        if not self.predictions or len(self.metrics_history) < 10:
            return 0.0
        
        # Compare predictions with actual values
        accurate_predictions = 0
        total_predictions = 0
        
        for prediction in self.predictions[-10:]:  # Last 10 predictions
            # Find actual metrics at prediction time
            prediction_target_time = prediction.timestamp + timedelta(
                minutes=prediction.prediction_horizon_minutes
            )
            
            # Find closest actual metrics
            closest_actual = None
            min_time_diff = timedelta(hours=1)
            
            for actual in self.metrics_history:
                time_diff = abs(actual.timestamp - prediction_target_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_actual = actual
            
            if closest_actual and min_time_diff < timedelta(minutes=10):
                # Calculate accuracy for CPU prediction
                predicted = prediction.predicted_cpu_usage
                actual = closest_actual.cpu_usage
                
                # Consider prediction accurate if within 20%
                error = abs(predicted - actual) / max(actual, 0.1)
                if error < 0.2:
                    accurate_predictions += 1
                
                total_predictions += 1
        
        return accurate_predictions / max(total_predictions, 1)
    
    def _calculate_cost_efficiency(self) -> float:
        """Calculate cost efficiency score."""
        if not self.scaling_history:
            return 1.0
        
        # Calculate cost savings from scaling decisions
        total_savings = 0.0
        
        for action in self.scaling_history[-20:]:  # Last 20 actions
            if action.direction == ScalingDirection.DOWN and action.estimated_cost_impact < 0:
                total_savings += abs(action.estimated_cost_impact)
        
        # Normalize to 0-1 scale
        return min(1.0, total_savings / 100.0)  # $100/hour = perfect efficiency
    
    def _get_privacy_budget_status(self) -> Dict[str, Any]:
        """Get privacy budget status."""
        if not self.privacy_tracker:
            return {"available": False}
        
        status = self.privacy_tracker.get_status()
        return {
            "available": True,
            "utilization_percentage": status.utilization_percentage,
            "remaining_budget": status.remaining_budget,
            "consumption_rate": status.consumption_rate,
            "risk_level": status.risk_level
        }


class WorkloadPredictor:
    """Predicts future workload patterns using time series analysis."""
    
    def __init__(self):
        self.models = {}
        self.feature_extractors = {
            "time_of_day": self._extract_time_features,
            "trend": self._extract_trend_features,
            "seasonality": self._extract_seasonal_features
        }
    
    def predict(
        self,
        metrics_history: List[ResourceMetrics],
        horizon_minutes: int
    ) -> Optional[WorkloadPrediction]:
        """Predict workload for specified horizon."""
        if len(metrics_history) < 10:
            return None
        
        try:
            # Extract features
            features = self._extract_features(metrics_history)
            
            # Simple trend-based prediction
            recent_metrics = metrics_history[-5:]
            
            # Calculate trends
            cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
            memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
            request_trend = self._calculate_trend([m.request_rate for m in recent_metrics])
            privacy_trend = self._calculate_trend([m.privacy_budget_consumption_rate for m in recent_metrics])
            
            # Project trends forward
            current_cpu = recent_metrics[-1].cpu_usage
            current_memory = recent_metrics[-1].memory_usage
            current_requests = recent_metrics[-1].request_rate
            current_privacy = recent_metrics[-1].privacy_budget_consumption_rate
            
            # Apply trend for horizon
            time_factor = horizon_minutes / 60.0  # Convert to hours
            
            predicted_cpu = max(0.0, min(1.0, current_cpu + cpu_trend * time_factor))
            predicted_memory = max(0.0, min(1.0, current_memory + memory_trend * time_factor))
            predicted_requests = max(0.0, current_requests + request_trend * time_factor)
            predicted_privacy = max(0.0, current_privacy + privacy_trend * time_factor)
            
            # Calculate confidence based on trend stability
            confidence = self._calculate_confidence(features, horizon_minutes)
            
            # Identify factors
            factors = []
            if abs(cpu_trend) > 0.1:
                factors.append("cpu_trend")
            if abs(memory_trend) > 0.1:
                factors.append("memory_trend")
            if abs(request_trend) > 10:
                factors.append("request_trend")
            
            prediction = WorkloadPrediction(
                timestamp=datetime.now(),
                prediction_horizon_minutes=horizon_minutes,
                predicted_cpu_usage=predicted_cpu,
                predicted_memory_usage=predicted_memory,
                predicted_request_rate=predicted_requests,
                predicted_privacy_consumption=predicted_privacy,
                confidence_score=confidence,
                factors=factors
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating workload prediction: {e}")
            return None
    
    def _extract_features(self, metrics_history: List[ResourceMetrics]) -> Dict[str, Any]:
        """Extract features from metrics history."""
        features = {}
        
        for name, extractor in self.feature_extractors.items():
            try:
                features.update(extractor(metrics_history))
            except Exception as e:
                logger.warning(f"Failed to extract {name} features: {e}")
        
        return features
    
    def _extract_time_features(self, metrics_history: List[ResourceMetrics]) -> Dict[str, Any]:
        """Extract time-based features."""
        if not metrics_history:
            return {}
        
        latest = metrics_history[-1]
        hour = latest.timestamp.hour
        day_of_week = latest.timestamp.weekday()
        
        return {
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "is_business_hours": 9 <= hour <= 17,
            "is_weekend": day_of_week >= 5
        }
    
    def _extract_trend_features(self, metrics_history: List[ResourceMetrics]) -> Dict[str, Any]:
        """Extract trend features."""
        if len(metrics_history) < 3:
            return {}
        
        cpu_values = [m.cpu_usage for m in metrics_history[-10:]]
        memory_values = [m.memory_usage for m in metrics_history[-10:]]
        
        return {
            "cpu_trend": self._calculate_trend(cpu_values),
            "memory_trend": self._calculate_trend(memory_values),
            "cpu_volatility": np.std(cpu_values),
            "memory_volatility": np.std(memory_values)
        }
    
    def _extract_seasonal_features(self, metrics_history: List[ResourceMetrics]) -> Dict[str, Any]:
        """Extract seasonal features."""
        if len(metrics_history) < 24:  # Need at least 24 hours of data
            return {}
        
        # Group by hour of day
        hourly_patterns = {}
        for metric in metrics_history:
            hour = metric.timestamp.hour
            if hour not in hourly_patterns:
                hourly_patterns[hour] = []
            hourly_patterns[hour].append(metric.cpu_usage)
        
        # Calculate average usage per hour
        hourly_averages = {}
        for hour, values in hourly_patterns.items():
            hourly_averages[hour] = np.mean(values)
        
        current_hour = metrics_history[-1].timestamp.hour
        expected_usage = hourly_averages.get(current_hour, 0.5)
        
        return {
            "seasonal_expected_cpu": expected_usage,
            "seasonal_deviation": abs(metrics_history[-1].cpu_usage - expected_usage)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope
    
    def _calculate_confidence(
        self,
        features: Dict[str, Any],
        horizon_minutes: int
    ) -> float:
        """Calculate prediction confidence."""
        base_confidence = 0.8
        
        # Reduce confidence for longer horizons
        horizon_factor = max(0.3, 1.0 - (horizon_minutes / 120.0))
        
        # Reduce confidence for high volatility
        volatility = features.get("cpu_volatility", 0.1)
        volatility_factor = max(0.5, 1.0 - volatility * 2)
        
        # Increase confidence for stable trends
        trend_magnitude = abs(features.get("cpu_trend", 0.0))
        trend_factor = min(1.2, 1.0 + trend_magnitude)
        
        confidence = base_confidence * horizon_factor * volatility_factor * trend_factor
        
        return max(0.1, min(1.0, confidence))


class CostOptimizer:
    """Optimizes costs while maintaining SLA and privacy requirements."""
    
    def __init__(self):
        self.cost_models = {
            "compute": self._compute_cost_model,
            "storage": self._storage_cost_model,
            "network": self._network_cost_model
        }
    
    def get_recommendation(
        self,
        current_metrics: ResourceMetrics,
        resource_type: ResourceType
    ) -> Dict[str, Any]:
        """Get cost optimization recommendation."""
        try:
            # Analyze current utilization
            utilization = self._get_utilization(current_metrics, resource_type)
            
            # Calculate cost efficiency
            efficiency = self._calculate_efficiency(current_metrics, resource_type)
            
            # Determine recommendation
            if utilization < 0.3 and efficiency > 0.8:
                return {
                    "direction": "scale_down",
                    "confidence": 0.8,
                    "estimated_savings": self._calculate_savings(resource_type, 0.7),
                    "reason": "Low utilization with good efficiency"
                }
            
            elif utilization > 0.9 and efficiency < 0.6:
                return {
                    "direction": "scale_up",
                    "confidence": 0.7,
                    "estimated_savings": -self._calculate_savings(resource_type, 1.5),
                    "reason": "High utilization with poor efficiency"
                }
            
            return {
                "direction": "stable",
                "confidence": 0.6,
                "estimated_savings": 0.0,
                "reason": "Current configuration is cost-optimal"
            }
            
        except Exception as e:
            logger.error(f"Error generating cost recommendation: {e}")
            return {}
    
    def generate_report(
        self,
        scaling_history: List[ScalingAction],
        current_metrics: ResourceMetrics,
        predictions: List[WorkloadPrediction]
    ) -> Dict[str, Any]:
        """Generate comprehensive cost optimization report."""
        try:
            # Calculate current costs
            current_hourly_cost = self._calculate_current_cost(current_metrics)
            
            # Calculate potential savings from scaling history
            realized_savings = sum(
                abs(action.estimated_cost_impact) for action in scaling_history
                if action.direction == ScalingDirection.DOWN and action.estimated_cost_impact < 0
            )
            
            # Calculate optimization opportunities
            opportunities = self._identify_optimization_opportunities(
                current_metrics, predictions
            )
            
            report = {
                "current_hourly_cost": current_hourly_cost,
                "realized_savings_last_24h": realized_savings,
                "potential_savings": sum(op["savings"] for op in opportunities),
                "optimization_opportunities": opportunities,
                "cost_efficiency_score": self._calculate_cost_efficiency_score(current_metrics),
                "recommendations": self._generate_cost_recommendations(current_metrics),
                "generated_at": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating cost optimization report: {e}")
            return {}
    
    def _get_utilization(self, metrics: ResourceMetrics, resource_type: ResourceType) -> float:
        """Get utilization for resource type."""
        if resource_type == ResourceType.CPU:
            return metrics.cpu_usage
        elif resource_type == ResourceType.MEMORY:
            return metrics.memory_usage
        elif resource_type == ResourceType.GPU:
            return metrics.gpu_usage
        
        return 0.5  # Default
    
    def _calculate_efficiency(self, metrics: ResourceMetrics, resource_type: ResourceType) -> float:
        """Calculate resource efficiency."""
        utilization = self._get_utilization(metrics, resource_type)
        
        # Efficiency based on utilization curve
        if utilization < 0.2:
            return 0.3  # Very inefficient
        elif utilization < 0.5:
            return 0.6  # Moderate efficiency
        elif utilization < 0.8:
            return 1.0  # High efficiency
        else:
            return 0.7  # Overutilized
    
    def _calculate_savings(self, resource_type: ResourceType, scale_factor: float) -> float:
        """Calculate potential savings from scaling."""
        base_costs = {
            ResourceType.CPU: 10.0,  # $10/hour baseline
            ResourceType.MEMORY: 5.0,
            ResourceType.GPU: 25.0,
            ResourceType.STORAGE: 2.0
        }
        
        base_cost = base_costs.get(resource_type, 5.0)
        return base_cost * (1.0 - scale_factor)  # Savings from scaling down
    
    def _calculate_current_cost(self, metrics: ResourceMetrics) -> float:
        """Calculate current hourly cost."""
        # Simplified cost calculation
        cpu_cost = metrics.cpu_usage * 10.0
        memory_cost = metrics.memory_usage * 5.0
        gpu_cost = metrics.gpu_usage * 25.0
        
        return cpu_cost + memory_cost + gpu_cost
    
    def _identify_optimization_opportunities(
        self,
        current_metrics: ResourceMetrics,
        predictions: List[WorkloadPrediction]
    ) -> List[Dict[str, Any]]:
        """Identify cost optimization opportunities."""
        opportunities = []
        
        # Low CPU utilization opportunity
        if current_metrics.cpu_usage < 0.3:
            opportunities.append({
                "type": "cpu_rightsizing",
                "description": "CPU utilization is low, consider scaling down",
                "savings": 5.0,  # $5/hour
                "confidence": 0.8
            })
        
        # Memory overprovisioning
        if current_metrics.memory_usage < 0.4:
            opportunities.append({
                "type": "memory_rightsizing",
                "description": "Memory utilization is low, consider reducing allocation",
                "savings": 3.0,  # $3/hour
                "confidence": 0.7
            })
        
        # Predictive optimization
        if predictions:
            latest_prediction = predictions[-1]
            if latest_prediction.predicted_cpu_usage < 0.5:
                opportunities.append({
                    "type": "predictive_scaling",
                    "description": "Predicted low utilization, schedule scale-down",
                    "savings": 7.0,
                    "confidence": latest_prediction.confidence_score
                })
        
        return opportunities
    
    def _calculate_cost_efficiency_score(self, metrics: ResourceMetrics) -> float:
        """Calculate overall cost efficiency score."""
        cpu_efficiency = self._calculate_efficiency(metrics, ResourceType.CPU)
        memory_efficiency = self._calculate_efficiency(metrics, ResourceType.MEMORY)
        gpu_efficiency = self._calculate_efficiency(metrics, ResourceType.GPU)
        
        # Weighted average
        total_efficiency = (
            cpu_efficiency * 0.4 +
            memory_efficiency * 0.3 +
            gpu_efficiency * 0.3
        )
        
        return total_efficiency
    
    def _generate_cost_recommendations(self, metrics: ResourceMetrics) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        if metrics.cpu_usage < 0.3:
            recommendations.append("Consider reducing CPU allocation to save costs")
        
        if metrics.memory_usage < 0.4:
            recommendations.append("Memory appears overprovisioned, consider rightsizing")
        
        if metrics.gpu_usage < 0.5 and metrics.gpu_usage > 0:
            recommendations.append("GPU utilization is low, consider switching to CPU-only instances")
        
        if metrics.response_time_p95 < 0.5:
            recommendations.append("Excellent performance allows for potential cost optimization")
        
        return recommendations
    
    def _compute_cost_model(self, resources: Dict[str, float]) -> float:
        """Compute cost model for compute resources."""
        return resources.get("cpu", 0) * 0.10 + resources.get("memory", 0) * 0.02
    
    def _storage_cost_model(self, resources: Dict[str, float]) -> float:
        """Storage cost model."""
        return resources.get("storage", 0) * 0.001
    
    def _network_cost_model(self, resources: Dict[str, float]) -> float:
        """Network cost model."""
        return resources.get("network_gb", 0) * 0.05


class PrivacyAwareScaler:
    """Scales resources with privacy budget considerations."""
    
    def __init__(self, privacy_tracker: Optional[Any]):
        self.privacy_tracker = privacy_tracker
    
    def adjust_confidence(
        self,
        base_confidence: float,
        resource_type: ResourceType,
        direction: ScalingDirection,
        privacy_impact: float
    ) -> float:
        """Adjust scaling confidence based on privacy considerations."""
        if not self.privacy_tracker:
            return base_confidence
        
        try:
            budget_status = self.privacy_tracker.get_status()
            
            # If privacy budget is running low, be more conservative about scaling down
            if budget_status.risk_level == "high":
                if direction == ScalingDirection.DOWN:
                    return base_confidence * 0.5  # Reduce confidence
                else:
                    return base_confidence * 1.2  # Increase confidence for scaling up
            
            # If privacy impact is negative (good for privacy), increase confidence
            if privacy_impact < 0:
                return min(1.0, base_confidence * 1.1)
            elif privacy_impact > 0:
                return base_confidence * 0.9
            
            return base_confidence
            
        except Exception as e:
            logger.error(f"Error adjusting privacy-aware confidence: {e}")
            return base_confidence


class BatchSizeOptimizer:
    """Optimizes batch sizes for privacy-utility tradeoffs."""
    
    def optimize(
        self,
        current_batch_size: int,
        privacy_budget_remaining: float,
        target_accuracy: float,
        current_metrics: ResourceMetrics
    ) -> int:
        """Optimize batch size for current conditions."""
        try:
            # Base optimization on available resources and privacy budget
            memory_factor = 1.0 - current_metrics.memory_usage  # Available memory
            privacy_factor = privacy_budget_remaining / 10.0  # Normalize privacy budget
            
            # Larger batches are generally better for privacy but require more memory
            optimal_factor = math.sqrt(memory_factor * privacy_factor)
            
            # Apply factor with bounds
            optimal_batch_size = int(current_batch_size * (0.5 + optimal_factor))
            
            # Apply reasonable bounds
            optimal_batch_size = max(8, min(1024, optimal_batch_size))
            
            return optimal_batch_size
            
        except Exception as e:
            logger.error(f"Error optimizing batch size: {e}")
            return current_batch_size


class FederatedOrchestrator:
    """Orchestrates federated learning node scaling."""
    
    def create_scaling_plan(
        self,
        num_clients: int,
        data_distribution: Dict[str, float],
        privacy_requirements: Dict[str, float],
        current_metrics: ResourceMetrics
    ) -> Dict[str, Any]:
        """Create federated learning scaling plan."""
        try:
            # Calculate optimal number of participants
            optimal_clients = self._calculate_optimal_clients(
                num_clients, data_distribution, privacy_requirements
            )
            
            # Determine resource allocation per client
            resource_per_client = self._calculate_resource_allocation(
                optimal_clients, current_metrics
            )
            
            # Create communication topology
            topology = self._design_communication_topology(optimal_clients)
            
            scaling_plan = {
                "optimal_clients": optimal_clients,
                "current_clients": num_clients,
                "scaling_direction": "up" if optimal_clients > num_clients else "down",
                "resource_per_client": resource_per_client,
                "communication_topology": topology,
                "estimated_training_time": self._estimate_training_time(optimal_clients),
                "privacy_efficiency": self._calculate_privacy_efficiency(
                    optimal_clients, privacy_requirements
                ),
                "cost_per_round": self._estimate_cost_per_round(optimal_clients, resource_per_client)
            }
            
            return scaling_plan
            
        except Exception as e:
            logger.error(f"Error creating federated scaling plan: {e}")
            return {}
    
    def _calculate_optimal_clients(
        self,
        current_clients: int,
        data_distribution: Dict[str, float],
        privacy_requirements: Dict[str, float]
    ) -> int:
        """Calculate optimal number of federated clients."""
        # More clients generally improve privacy but increase communication overhead
        privacy_weight = sum(privacy_requirements.values()) / len(privacy_requirements)
        
        # Base on data distribution and privacy needs
        if privacy_weight > 0.8:  # High privacy requirements
            optimal = min(100, current_clients * 2)
        elif privacy_weight > 0.5:  # Medium privacy requirements
            optimal = min(50, int(current_clients * 1.5))
        else:  # Low privacy requirements
            optimal = current_clients
        
        return max(5, optimal)  # Minimum 5 clients for meaningful federation
    
    def _calculate_resource_allocation(
        self,
        num_clients: int,
        current_metrics: ResourceMetrics
    ) -> Dict[str, float]:
        """Calculate resource allocation per client."""
        # Distribute available resources among clients
        available_cpu = 1.0 - current_metrics.cpu_usage
        available_memory = 1.0 - current_metrics.memory_usage
        
        return {
            "cpu_cores": max(0.5, available_cpu / num_clients * 8),  # Assuming 8 core baseline
            "memory_gb": max(1.0, available_memory / num_clients * 16),  # Assuming 16GB baseline
            "network_mbps": 100 / num_clients  # Distribute 100 Mbps
        }
    
    def _design_communication_topology(self, num_clients: int) -> Dict[str, Any]:
        """Design communication topology for federated learning."""
        if num_clients <= 10:
            return {"type": "star", "aggregators": 1}
        elif num_clients <= 50:
            return {"type": "tree", "aggregators": max(2, num_clients // 10)}
        else:
            return {"type": "hierarchical", "aggregators": max(3, num_clients // 20)}
    
    def _estimate_training_time(self, num_clients: int) -> float:
        """Estimate training time in minutes."""
        # More clients increase communication overhead
        base_time = 10.0  # 10 minutes base
        communication_overhead = num_clients * 0.5  # 0.5 min per client
        
        return base_time + communication_overhead
    
    def _calculate_privacy_efficiency(
        self,
        num_clients: int,
        privacy_requirements: Dict[str, float]
    ) -> float:
        """Calculate privacy efficiency score."""
        # More clients generally improve privacy
        client_factor = min(1.0, num_clients / 50.0)  # Normalize to 50 clients
        privacy_factor = np.mean(list(privacy_requirements.values()))
        
        return client_factor * privacy_factor
    
    def _estimate_cost_per_round(
        self,
        num_clients: int,
        resource_per_client: Dict[str, float]
    ) -> float:
        """Estimate cost per federated learning round."""
        compute_cost = resource_per_client["cpu_cores"] * 0.10  # $0.10/core/hour
        memory_cost = resource_per_client["memory_gb"] * 0.02  # $0.02/GB/hour
        network_cost = resource_per_client["network_mbps"] * 0.001  # $0.001/Mbps/hour
        
        cost_per_client = compute_cost + memory_cost + network_cost
        return cost_per_client * num_clients / 60  # Convert to per-minute


class MockResourceManager:
    """Mock resource manager for testing purposes."""
    
    def __init__(self):
        self.capacities = {
            "cpu": 4,
            "memory": 8,
            "gpu": 1,
            "storage": 100
        }
        
        self.utilizations = {
            "cpu": 0.5,
            "memory": 0.6,
            "gpu": 0.3,
            "storage": 0.4
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current resource metrics."""
        return {
            "cpu_usage": self.utilizations["cpu"],
            "memory_usage": self.utilizations["memory"],
            "gpu_usage": self.utilizations["gpu"],
            "network_io": 0.2,
            "storage_io": 0.3,
            "request_rate": 10.0,
            "response_time_p95": 1.5,
            "error_rate": 0.02
        }
    
    def get_current_capacity(self, resource_type: str) -> int:
        """Get current capacity for resource type."""
        return self.capacities.get(resource_type, 1)
    
    def scale_resource(
        self,
        resource_type: str,
        target_capacity: int,
        current_capacity: int
    ) -> bool:
        """Scale resource to target capacity."""
        logger.info(f"Scaling {resource_type} from {current_capacity} to {target_capacity}")
        
        # Update mock capacity
        self.capacities[resource_type] = target_capacity
        
        # Simulate scaling effects
        if target_capacity > current_capacity:
            # Scaling up reduces utilization
            self.utilizations[resource_type] *= 0.8
        else:
            # Scaling down increases utilization
            self.utilizations[resource_type] *= 1.2
        
        # Keep utilization in bounds
        self.utilizations[resource_type] = min(1.0, self.utilizations[resource_type])
        
        return True