"""Advanced scaling and performance optimization for privacy-preserving ML.

This module provides enterprise-grade scalability features including:
- Intelligent resource management and auto-scaling
- Distributed training coordination and load balancing
- Performance monitoring and optimization
- Dynamic resource allocation and cost optimization
"""

# Import available components
try:
    from .performance_optimizer import PerformanceOptimizer, OptimizationProfile, OptimizationType
    from .auto_scaler import AutoScaler, ScalingPolicy, ScalingTrigger, NodeType, ScalingDirection
    
    __all__ = [
        "PerformanceOptimizer",
        "OptimizationProfile",
        "OptimizationType",
        "AutoScaler",
        "ScalingPolicy",
        "ScalingTrigger", 
        "NodeType",
        "ScalingDirection"
    ]
    
except ImportError as e:
    import logging
    logging.warning(f"Some scaling components not available: {e}")
    __all__ = []