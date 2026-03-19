"""
Privacy-Preserving Agent Fine-Tuner
====================================
A framework for fine-tuning AI agents with formal differential privacy guarantees.
No Opacus dependency — pure PyTorch implementation of DP-SGD.
"""

from .dataset import PrivateDataset
from .trainer import PrivateTrainer
from .budget import PrivacyBudgetManager
from .federated import FederatedAggregator
from .report import PrivacyReport

__version__ = "1.0.0"
__all__ = [
    "PrivateDataset",
    "PrivateTrainer",
    "PrivacyBudgetManager",
    "FederatedAggregator",
    "PrivacyReport",
]
