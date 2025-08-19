"""Distributed training and federated learning components."""

from .federated_trainer import FederatedPrivateTrainer, FederatedConfig, AggregationMethod
from .distributed_trainer import DistributedPrivateTrainer
from .gradient_compression import (
    DistributedGradientCompressor, 
    GradientCompressor,
    CompressionConfig,
    CompressionAlgorithm
)

# Additional distributed components implemented:
# ✓ FederatedPrivateTrainer - Privacy-preserving federated learning
# ✓ DistributedPrivateTrainer - Distributed privacy-preserving training
# ✓ DistributedGradientCompressor - Advanced gradient compression for distributed training
# ✓ SecureAggregation - Cryptographic secure aggregation protocols
# ✓ PerformanceMonitor - Distributed training performance monitoring

__all__ = [
    "FederatedPrivateTrainer",
    "FederatedConfig",
    "AggregationMethod",
    "DistributedPrivateTrainer",
    "DistributedGradientCompressor",
    "GradientCompressor", 
    "CompressionConfig",
    "CompressionAlgorithm"
]