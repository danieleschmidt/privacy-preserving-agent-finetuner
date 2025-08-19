"""Performance optimization and resource management components."""

from .resource_optimizer import ResourceOptimizer, ResourceProfile
from .memory_manager import MemoryManager
from .performance import PerformanceOptimizer
from .recommendation_engine import RecommendationEngine
from .quantum_performance_optimizer import QuantumPerformanceOptimizer
from .quantum_performance_engine import QuantumPerformanceEngine

# Additional optimization components implemented:
# ✓ ResourceOptimizer - Intelligent resource allocation and optimization
# ✓ MemoryManager - Advanced memory management and caching strategies
# ✓ PerformanceOptimizer - Comprehensive performance optimization
# ✓ RecommendationEngine - ML-powered optimization recommendations
# ✓ QuantumPerformanceOptimizer - Quantum-inspired optimization algorithms
# ✓ QuantumPerformanceEngine - Advanced quantum performance techniques

__all__ = [
    "ResourceOptimizer",
    "ResourceProfile", 
    "MemoryManager",
    "PerformanceOptimizer",
    "RecommendationEngine",
    "QuantumPerformanceOptimizer",
    "QuantumPerformanceEngine"
]