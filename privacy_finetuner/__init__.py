"""
Privacy-Preserving Agent Finetuner

Enterprise-grade framework for fine-tuning LLMs with differential privacy guarantees.
Ensures sensitive data never leaves your trust boundary while maintaining model performance.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon-labs.com"

from .core import PrivateTrainer
from .privacy import PrivacyConfig
from .context_guard import ContextGuard, RedactionStrategy

__all__ = [
    "PrivateTrainer",
    "PrivacyConfig", 
    "ContextGuard",
    "RedactionStrategy",
]