"""Repository layer for data access patterns and business logic."""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, and_, or_, func
from uuid import UUID

from .models import (
    TrainingJob, PrivacyBudgetEntry, Dataset, Model, User, AuditLog
)

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common CRUD operations."""
    
    def __init__(self, session: Session, model_class):
        self.session = session
        self.model_class = model_class
    
    def create(self, **kwargs) -> Any:
        """Create a new record."""
        instance = self.model_class(**kwargs)
        self.session.add(instance)
        self.session.flush()
        return instance
    
    def get_by_id(self, id: UUID) -> Optional[Any]:
        """Get record by ID."""
        return self.session.query(self.model_class).filter(
            self.model_class.id == id
        ).first()
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """Get all records with pagination."""
        return self.session.query(self.model_class).offset(offset).limit(limit).all()
    
    def update(self, id: UUID, **kwargs) -> Optional[Any]:
        """Update record by ID."""
        instance = self.get_by_id(id)
        if instance:
            for key, value in kwargs.items():
                setattr(instance, key, value)
            self.session.flush()
        return instance
    
    def delete(self, id: UUID) -> bool:
        """Delete record by ID."""
        instance = self.get_by_id(id)
        if instance:
            self.session.delete(instance)
            self.session.flush()
            return True
        return False
    
    def count(self) -> int:
        """Count total records."""
        return self.session.query(self.model_class).count()


class UserRepository(BaseRepository):
    """Repository for user management."""
    
    def __init__(self, session: Session):
        super().__init__(session, User)
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.session.query(User).filter(User.username == username).first()
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.session.query(User).filter(User.email == email).first()
    
    def get_active_users(self) -> List[User]:
        """Get all active users."""
        return self.session.query(User).filter(User.is_active == True).all()
    
    def search_users(self, query: str, limit: int = 20) -> List[User]:
        """Search users by username or email."""
        return self.session.query(User).filter(
            or_(
                User.username.ilike(f"%{query}%"),
                User.email.ilike(f"%{query}%"),
                User.full_name.ilike(f"%{query}%")
            )
        ).filter(User.is_active == True).limit(limit).all()


class DatasetRepository(BaseRepository):
    """Repository for dataset management."""
    
    def __init__(self, session: Session):
        super().__init__(session, Dataset)
    
    def get_by_owner(self, owner_id: UUID, limit: int = 50) -> List[Dataset]:
        """Get datasets by owner."""
        return self.session.query(Dataset).filter(
            Dataset.owner_id == owner_id
        ).order_by(desc(Dataset.created_at)).limit(limit).all()
    
    def get_public_datasets(self, limit: int = 50) -> List[Dataset]:
        """Get public datasets."""
        return self.session.query(Dataset).filter(
            Dataset.is_public == True
        ).order_by(desc(Dataset.created_at)).limit(limit).all()
    
    def search_datasets(self, query: str, owner_id: Optional[UUID] = None) -> List[Dataset]:
        """Search datasets by name or description."""
        base_query = self.session.query(Dataset).filter(
            or_(
                Dataset.name.ilike(f"%{query}%"),
                Dataset.description.ilike(f"%{query}%")
            )
        )
        
        if owner_id:
            base_query = base_query.filter(Dataset.owner_id == owner_id)
        else:
            base_query = base_query.filter(Dataset.is_public == True)
        
        return base_query.order_by(desc(Dataset.created_at)).limit(20).all()
    
    def get_by_privacy_level(self, privacy_level: str) -> List[Dataset]:
        """Get datasets by privacy level."""
        return self.session.query(Dataset).filter(
            Dataset.privacy_level == privacy_level
        ).all()
    
    def get_datasets_with_pii(self) -> List[Dataset]:
        """Get datasets containing PII."""
        return self.session.query(Dataset).filter(Dataset.contains_pii == True).all()


class ModelRepository(BaseRepository):
    """Repository for model management."""
    
    def __init__(self, session: Session):
        super().__init__(session, Model)
    
    def get_by_owner(self, owner_id: UUID, limit: int = 50) -> List[Model]:
        """Get models by owner."""
        return self.session.query(Model).filter(
            Model.owner_id == owner_id
        ).order_by(desc(Model.created_at)).limit(limit).all()
    
    def get_by_name_version(self, name: str, version: str, owner_id: UUID) -> Optional[Model]:
        """Get model by name and version."""
        return self.session.query(Model).filter(
            and_(
                Model.name == name,
                Model.version == version,
                Model.owner_id == owner_id
            )
        ).first()
    
    def get_deployed_models(self) -> List[Model]:
        """Get all deployed models."""
        return self.session.query(Model).filter(Model.is_deployed == True).all()
    
    def get_models_by_base_model(self, base_model: str) -> List[Model]:
        """Get models derived from a base model."""
        return self.session.query(Model).filter(
            Model.base_model == base_model
        ).order_by(desc(Model.created_at)).all()
    
    def get_models_by_privacy_budget(self, max_epsilon: float) -> List[Model]:
        """Get models within privacy budget range."""
        return self.session.query(Model).filter(
            Model.epsilon_spent <= max_epsilon
        ).order_by(asc(Model.epsilon_spent)).all()


class TrainingJobRepository(BaseRepository):
    """Repository for training job management."""
    
    def __init__(self, session: Session):
        super().__init__(session, TrainingJob)
    
    def get_by_user(self, user_id: UUID, limit: int = 50) -> List[TrainingJob]:
        """Get training jobs by user."""
        return self.session.query(TrainingJob).filter(
            TrainingJob.user_id == user_id
        ).order_by(desc(TrainingJob.created_at)).limit(limit).all()
    
    def get_by_status(self, status: str, limit: int = 100) -> List[TrainingJob]:
        """Get training jobs by status."""
        return self.session.query(TrainingJob).filter(
            TrainingJob.status == status
        ).order_by(desc(TrainingJob.created_at)).limit(limit).all()
    
    def get_active_jobs(self) -> List[TrainingJob]:
        """Get currently running or queued jobs."""
        return self.session.query(TrainingJob).filter(
            TrainingJob.status.in_(["queued", "running"])
        ).order_by(asc(TrainingJob.created_at)).all()
    
    def get_jobs_by_dataset(self, dataset_id: UUID) -> List[TrainingJob]:
        """Get training jobs using a specific dataset."""
        return self.session.query(TrainingJob).filter(
            TrainingJob.dataset_id == dataset_id
        ).order_by(desc(TrainingJob.created_at)).all()
    
    def get_recent_completed_jobs(self, days: int = 7, limit: int = 20) -> List[TrainingJob]:
        """Get recently completed jobs."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return self.session.query(TrainingJob).filter(
            and_(
                TrainingJob.status == "completed",
                TrainingJob.completed_at >= cutoff_date
            )
        ).order_by(desc(TrainingJob.completed_at)).limit(limit).all()
    
    def get_privacy_budget_summary(self, user_id: Optional[UUID] = None) -> Dict[str, float]:
        """Get privacy budget summary statistics."""
        query = self.session.query(
            func.sum(TrainingJob.epsilon_spent).label("total_epsilon"),
            func.avg(TrainingJob.epsilon_spent).label("avg_epsilon"),
            func.max(TrainingJob.epsilon_spent).label("max_epsilon"),
            func.count(TrainingJob.id).label("job_count")
        )
        
        if user_id:
            query = query.filter(TrainingJob.user_id == user_id)
        
        result = query.first()
        
        return {
            "total_epsilon_spent": float(result.total_epsilon or 0),
            "average_epsilon_per_job": float(result.avg_epsilon or 0),
            "max_epsilon_single_job": float(result.max_epsilon or 0),
            "total_jobs": int(result.job_count or 0)
        }
    
    def update_job_progress(self, job_id: UUID, progress: float, **kwargs) -> Optional[TrainingJob]:
        """Update job progress and other fields."""
        job = self.get_by_id(job_id)
        if job:
            job.progress = max(0.0, min(1.0, progress))  # Clamp between 0 and 1
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            self.session.flush()
        return job


class PrivacyBudgetRepository(BaseRepository):
    """Repository for privacy budget tracking."""
    
    def __init__(self, session: Session):
        super().__init__(session, PrivacyBudgetEntry)
    
    def get_by_training_job(self, training_job_id: UUID) -> List[PrivacyBudgetEntry]:
        """Get privacy entries for a training job."""
        return self.session.query(PrivacyBudgetEntry).filter(
            PrivacyBudgetEntry.training_job_id == training_job_id
        ).order_by(asc(PrivacyBudgetEntry.step_number)).all()
    
    def get_by_user(self, user_id: UUID, limit: int = 100) -> List[PrivacyBudgetEntry]:
        """Get privacy entries by user."""
        return self.session.query(PrivacyBudgetEntry).filter(
            PrivacyBudgetEntry.user_id == user_id
        ).order_by(desc(PrivacyBudgetEntry.created_at)).limit(limit).all()
    
    def get_user_budget_summary(self, user_id: UUID) -> Dict[str, Any]:
        """Get privacy budget summary for a user."""
        result = self.session.query(
            func.sum(PrivacyBudgetEntry.epsilon_spent).label("total_epsilon"),
            func.count(PrivacyBudgetEntry.id).label("entry_count"),
            func.min(PrivacyBudgetEntry.created_at).label("first_entry"),
            func.max(PrivacyBudgetEntry.created_at).label("last_entry")
        ).filter(PrivacyBudgetEntry.user_id == user_id).first()
        
        return {
            "total_epsilon_spent": float(result.total_epsilon or 0),
            "total_entries": int(result.entry_count or 0),
            "first_entry_date": result.first_entry,
            "last_entry_date": result.last_entry
        }
    
    def get_budget_by_operation(self, user_id: Optional[UUID] = None) -> List[Tuple[str, float, int]]:
        """Get privacy budget grouped by operation type."""
        query = self.session.query(
            PrivacyBudgetEntry.operation,
            func.sum(PrivacyBudgetEntry.epsilon_spent).label("total_epsilon"),
            func.count(PrivacyBudgetEntry.id).label("count")
        )
        
        if user_id:
            query = query.filter(PrivacyBudgetEntry.user_id == user_id)
        
        return query.group_by(PrivacyBudgetEntry.operation).order_by(
            desc("total_epsilon")
        ).all()
    
    def get_recent_entries(self, hours: int = 24, limit: int = 50) -> List[PrivacyBudgetEntry]:
        """Get recent privacy budget entries."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return self.session.query(PrivacyBudgetEntry).filter(
            PrivacyBudgetEntry.created_at >= cutoff_time
        ).order_by(desc(PrivacyBudgetEntry.created_at)).limit(limit).all()


class AuditLogRepository(BaseRepository):
    """Repository for audit log management."""
    
    def __init__(self, session: Session):
        super().__init__(session, AuditLog)
    
    def log_event(
        self,
        event_type: str,
        resource_type: str,
        action: str,
        user_id: Optional[UUID] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ) -> AuditLog:
        """Create an audit log entry."""
        entry = AuditLog(
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            user_id=user_id,
            **kwargs
        )
        self.session.add(entry)
        self.session.flush()
        return entry
    
    def get_by_user(self, user_id: UUID, limit: int = 100) -> List[AuditLog]:
        """Get audit logs for a user."""
        return self.session.query(AuditLog).filter(
            AuditLog.user_id == user_id
        ).order_by(desc(AuditLog.created_at)).limit(limit).all()
    
    def get_by_resource(self, resource_type: str, resource_id: str) -> List[AuditLog]:
        """Get audit logs for a specific resource."""
        return self.session.query(AuditLog).filter(
            and_(
                AuditLog.resource_type == resource_type,
                AuditLog.resource_id == resource_id
            )
        ).order_by(desc(AuditLog.created_at)).all()
    
    def get_by_event_type(self, event_type: str, limit: int = 100) -> List[AuditLog]:
        """Get audit logs by event type."""
        return self.session.query(AuditLog).filter(
            AuditLog.event_type == event_type
        ).order_by(desc(AuditLog.created_at)).limit(limit).all()
    
    def get_security_events(self, hours: int = 24) -> List[AuditLog]:
        """Get security-related events in the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        security_events = [
            "login_failed", "login_success", "logout", 
            "password_change", "permission_denied", "data_access"
        ]
        
        return self.session.query(AuditLog).filter(
            and_(
                AuditLog.event_type.in_(security_events),
                AuditLog.created_at >= cutoff_time
            )
        ).order_by(desc(AuditLog.created_at)).all()
    
    def get_privacy_events(self, days: int = 7) -> List[AuditLog]:
        """Get privacy-related events."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        return self.session.query(AuditLog).filter(
            and_(
                AuditLog.contains_pii == True,
                AuditLog.created_at >= cutoff_time
            )
        ).order_by(desc(AuditLog.created_at)).all()
    
    def cleanup_old_logs(self, retention_days: int = 90) -> int:
        """Clean up old audit logs based on retention policy."""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        # Only delete logs that don't have specific retention requirements
        count = self.session.query(AuditLog).filter(
            and_(
                AuditLog.created_at < cutoff_date,
                or_(
                    AuditLog.retention_until.is_(None),
                    AuditLog.retention_until < cutoff_date
                )
            )
        ).delete()
        
        logger.info(f"Cleaned up {count} old audit log entries")
        return count