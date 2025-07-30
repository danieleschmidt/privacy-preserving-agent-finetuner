"""Privacy-Preserving Agent Finetuner Framework.

Enterprise-grade framework for fine-tuning LLMs with differential privacy guarantees.
Ensures sensitive data never leaves your trust boundary while maintaining model performance.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt <daniel@terragon-labs.com>"
__license__ = "MIT"

from .core.trainer import PrivateTrainer
from .core.context_guard import ContextGuard
from .core.privacy_config import PrivacyConfig
from .api.server import create_app
from .utils.monitoring import PrivacyBudgetMonitor

__all__ = [
    "PrivateTrainer",
    "ContextGuard", 
    "PrivacyConfig",
    "PrivacyBudgetMonitor",
    "create_app",
]