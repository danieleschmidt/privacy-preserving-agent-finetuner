"""Distributed training and federated learning components."""

from .federated_trainer import FederatedPrivateTrainer, FederatedConfig, AggregationMethod

# TODO: Implement additional distributed components
# from .distributed_optimizer import DistributedPrivacyOptimizer
# from .secure_aggregation import SecureAggregationProtocol
# from .performance_monitor import DistributedPerformanceMonitor

__all__ = [
    "FederatedPrivateTrainer",
    "FederatedConfig",
    "AggregationMethod"
]