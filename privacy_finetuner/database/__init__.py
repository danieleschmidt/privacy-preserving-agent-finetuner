"""Database layer for privacy-preserving agent finetuner."""

from .connection import DatabaseManager, get_database, get_db_session
from .models import (
    Base,
    TrainingJob,
    PrivacyBudgetEntry,
    Dataset,
    Model,
    User,
    AuditLog,
    TimestampMixin
)
from .repositories import (
    TrainingJobRepository,
    PrivacyBudgetRepository,
    DatasetRepository,
    ModelRepository,
    UserRepository,
    AuditLogRepository,
    BaseRepository
)
from .query_optimizer import QueryOptimizer, PrivacyQueryMixin, QueryMetrics
from .advanced_operations import (
    AdvancedPrivacyOperations,
    PrivacyBudgetAnalysis,
    ModelPerformanceMetrics
)

__all__ = [
    # Connection management
    "DatabaseManager",
    "get_database",
    "get_db_session",
    
    # Models
    "Base",
    "TrainingJob",
    "PrivacyBudgetEntry",
    "Dataset",
    "Model",
    "User",
    "AuditLog",
    "TimestampMixin",
    
    # Repositories
    "TrainingJobRepository",
    "PrivacyBudgetRepository",
    "DatasetRepository",
    "ModelRepository",
    "UserRepository",
    "AuditLogRepository",
    "BaseRepository",
    
    # Query optimization
    "QueryOptimizer",
    "PrivacyQueryMixin",
    "QueryMetrics",
    
    # Advanced operations
    "AdvancedPrivacyOperations",
    "PrivacyBudgetAnalysis",
    "ModelPerformanceMetrics",
]