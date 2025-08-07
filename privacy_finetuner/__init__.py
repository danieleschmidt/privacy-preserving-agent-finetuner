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

# Import API server with graceful fallback
try:
    from .api.server import create_app
except ImportError:
    def create_app():
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")

# Import monitoring with graceful fallback
try:
    from .utils.monitoring import PrivacyBudgetMonitor
except ImportError:
    class PrivacyBudgetMonitor:
        def __init__(self, *args, **kwargs):
            import warnings
            warnings.warn("Monitoring utilities not fully available")

__all__ = [
    "PrivateTrainer",
    "ContextGuard", 
    "PrivacyConfig",
    "PrivacyBudgetMonitor",
    "create_app",
]