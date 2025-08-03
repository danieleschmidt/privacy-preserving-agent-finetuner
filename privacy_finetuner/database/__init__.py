"""Database layer for privacy-preserving agent finetuner."""

from .connection import DatabaseManager, get_database
from .models import (
    Base,
    TrainingJob,
    PrivacyBudgetEntry,
    Dataset,
    Model,
    User,
    AuditLog
)
from .repositories import (
    TrainingJobRepository,
    PrivacyBudgetRepository,
    DatasetRepository,
    ModelRepository,
    UserRepository,
    AuditLogRepository
)

__all__ = [
    "DatabaseManager",
    "get_database",
    "Base",
    "TrainingJob",
    "PrivacyBudgetEntry",
    "Dataset",
    "Model",
    "User",
    "AuditLog",
    "TrainingJobRepository",
    "PrivacyBudgetRepository",
    "DatasetRepository",
    "ModelRepository",
    "UserRepository",
    "AuditLogRepository",
]