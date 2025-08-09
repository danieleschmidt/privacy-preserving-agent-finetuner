"""Research-oriented privacy-preserving machine learning components.

This module provides advanced research capabilities including:
- Comprehensive benchmarking frameworks
- Novel privacy algorithm implementations  
- Experimental privacy-utility tradeoff analysis
- Research reproducibility utilities
"""

# Import available components
try:
    from .benchmark_suite import PrivacyBenchmarkSuite
    from .novel_algorithms import AdaptiveDPAlgorithm, HybridPrivacyMechanism
    
    __all__ = [
        "PrivacyBenchmarkSuite",
        "AdaptiveDPAlgorithm", 
        "HybridPrivacyMechanism"
    ]
    
except ImportError as e:
    import logging
    logging.warning(f"Some research components not available: {e}")
    __all__ = []