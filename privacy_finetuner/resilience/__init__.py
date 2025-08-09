"""Advanced resilience and recovery systems for privacy-preserving ML.

This module provides enterprise-grade reliability features including:
- Automatic checkpoint recovery and rollback
- Distributed training failure handling
- Privacy budget emergency protocols  
- System health monitoring and auto-healing
"""

# Import available components
try:
    from .failure_recovery import FailureRecoverySystem, RecoveryStrategy, FailureType, RecoveryPoint
    
    __all__ = [
        "FailureRecoverySystem", 
        "RecoveryStrategy",
        "FailureType",
        "RecoveryPoint"
    ]
    
except ImportError as e:
    import logging
    logging.warning(f"Some resilience components not available: {e}")
    __all__ = []