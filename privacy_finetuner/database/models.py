"""SQLAlchemy database models for privacy-preserving agent finetuner."""

from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, JSON,
    ForeignKey, UniqueConstraint, Index, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class TimestampMixin:
    """Mixin for created/updated timestamps."""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class User(Base, TimestampMixin):
    """User model for authentication and authorization."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Privacy settings
    privacy_preferences = Column(JSON, default=dict)
    consent_given = Column(Boolean, default=False, nullable=False)
    consent_date = Column(DateTime)
    
    # Relationships
    training_jobs = relationship("TrainingJob", back_populates="user")
    datasets = relationship("Dataset", back_populates="owner")
    models = relationship("Model", back_populates="owner")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class Dataset(Base, TimestampMixin):
    """Dataset model for training data management."""
    __tablename__ = "datasets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    file_path = Column(String(500), nullable=False)
    format = Column(String(50), nullable=False)  # jsonl, csv, parquet
    size_bytes = Column(Integer)
    record_count = Column(Integer)
    
    # Privacy metadata
    contains_pii = Column(Boolean, default=False, nullable=False)
    privacy_level = Column(String(20), nullable=False, default="medium")  # low, medium, high
    data_classification = Column(String(50))  # public, internal, confidential, restricted
    retention_policy = Column(String(100))
    
    # Ownership and access
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)
    
    # Checksums for integrity
    file_hash = Column(String(64))  # SHA-256 hash
    metadata_hash = Column(String(64))
    
    # Relationships
    owner = relationship("User", back_populates="datasets")
    training_jobs = relationship("TrainingJob", back_populates="dataset")
    
    __table_args__ = (
        Index("idx_dataset_owner_name", "owner_id", "name"),
        CheckConstraint("privacy_level IN ('low', 'medium', 'high')", name="valid_privacy_level"),
        CheckConstraint("format IN ('jsonl', 'csv', 'parquet', 'json')", name="valid_format"),
    )
    
    def __repr__(self):
        return f"<Dataset(name='{self.name}', format='{self.format}', size={self.size_bytes})>"


class Model(Base, TimestampMixin):
    """Model registry for tracking trained models."""
    __tablename__ = "models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    description = Column(Text)
    
    # Model metadata
    base_model = Column(String(255), nullable=False)  # HuggingFace model ID
    model_type = Column(String(50), nullable=False)  # causal_lm, seq2seq, etc.
    model_path = Column(String(500), nullable=False)
    config_path = Column(String(500))
    tokenizer_path = Column(String(500))
    
    # Training metadata
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"))
    
    # Privacy guarantees
    epsilon_spent = Column(Float, nullable=False)
    delta_value = Column(Float, nullable=False)
    noise_multiplier = Column(Float)
    max_grad_norm = Column(Float)
    
    # Performance metrics
    eval_loss = Column(Float)
    eval_accuracy = Column(Float)
    eval_perplexity = Column(Float)
    model_size_mb = Column(Float)
    
    # Deployment status
    is_deployed = Column(Boolean, default=False, nullable=False)
    deployment_url = Column(String(500))
    deployment_status = Column(String(50), default="not_deployed")
    
    # Ownership
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    owner = relationship("User", back_populates="models")
    training_job = relationship("TrainingJob", back_populates="model")
    
    __table_args__ = (
        UniqueConstraint("name", "version", "owner_id", name="unique_model_version"),
        Index("idx_model_owner_name", "owner_id", "name"),
        CheckConstraint("epsilon_spent >= 0", name="positive_epsilon"),
        CheckConstraint("delta_value >= 0 AND delta_value <= 1", name="valid_delta"),
    )
    
    def __repr__(self):
        return f"<Model(name='{self.name}', version='{self.version}', epsilon={self.epsilon_spent})>"


class TrainingJob(Base, TimestampMixin):
    """Training job tracking with privacy accounting."""
    __tablename__ = "training_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="queued", index=True)
    
    # Job configuration
    model_name = Column(String(255), nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    
    # Training parameters
    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    max_steps = Column(Integer)
    warmup_steps = Column(Integer)
    
    # Privacy configuration
    target_epsilon = Column(Float, nullable=False)
    target_delta = Column(Float, nullable=False)
    noise_multiplier = Column(Float, nullable=False)
    max_grad_norm = Column(Float, nullable=False)
    accounting_mode = Column(String(10), nullable=False)
    
    # Privacy accounting
    epsilon_spent = Column(Float, default=0.0, nullable=False)
    actual_steps = Column(Integer, default=0)
    sample_rate = Column(Float)
    
    # Job execution
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    
    # Results
    final_loss = Column(Float)
    best_eval_loss = Column(Float)
    training_time_seconds = Column(Integer)
    gpu_hours_used = Column(Float)
    
    # File paths
    checkpoint_path = Column(String(500))
    log_path = Column(String(500))
    config_path = Column(String(500))
    
    # Ownership
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="training_jobs")
    dataset = relationship("Dataset", back_populates="training_jobs")
    model = relationship("Model", back_populates="training_job", uselist=False)
    privacy_entries = relationship("PrivacyBudgetEntry", back_populates="training_job")
    
    __table_args__ = (
        Index("idx_training_job_user_status", "user_id", "status"),
        Index("idx_training_job_created", "created_at"),
        CheckConstraint("status IN ('queued', 'running', 'completed', 'failed', 'cancelled')", name="valid_status"),
        CheckConstraint("target_epsilon > 0", name="positive_target_epsilon"),
        CheckConstraint("target_delta > 0 AND target_delta < 1", name="valid_target_delta"),
        CheckConstraint("progress >= 0 AND progress <= 1", name="valid_progress"),
    )
    
    def __repr__(self):
        return f"<TrainingJob(name='{self.job_name}', status='{self.status}', epsilon={self.epsilon_spent})>"


class PrivacyBudgetEntry(Base, TimestampMixin):
    """Privacy budget consumption tracking."""
    __tablename__ = "privacy_budget_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Budget consumption
    epsilon_spent = Column(Float, nullable=False)
    delta_value = Column(Float, nullable=False)
    operation = Column(String(100), nullable=False)
    
    # Context
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("training_jobs.id"))
    session_id = Column(String(100))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Metadata
    step_number = Column(Integer)
    epoch_number = Column(Integer)
    batch_number = Column(Integer)
    metadata = Column(JSON, default=dict)
    
    # Accounting details
    noise_multiplier = Column(Float)
    sample_rate = Column(Float)
    accounting_mode = Column(String(10))
    
    # Relationships
    training_job = relationship("TrainingJob", back_populates="privacy_entries")
    user = relationship("User")
    
    __table_args__ = (
        Index("idx_privacy_entry_job_step", "training_job_id", "step_number"),
        Index("idx_privacy_entry_user_created", "user_id", "created_at"),
        CheckConstraint("epsilon_spent >= 0", name="positive_epsilon_spent"),
        CheckConstraint("delta_value >= 0 AND delta_value <= 1", name="valid_delta_value"),
    )
    
    def __repr__(self):
        return f"<PrivacyBudgetEntry(operation='{self.operation}', epsilon={self.epsilon_spent})>"


class AuditLog(Base, TimestampMixin):
    """Audit log for compliance and security tracking."""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event details
    event_type = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(100))
    action = Column(String(50), nullable=False)
    
    # Actor information
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    actor_ip = Column(String(45))  # IPv6 max length
    user_agent = Column(String(500))
    session_id = Column(String(100))
    
    # Request/response data
    request_data = Column(JSON)
    response_data = Column(JSON)
    status_code = Column(Integer)
    
    # Privacy and compliance
    contains_pii = Column(Boolean, default=False, nullable=False)
    data_classification = Column(String(50))
    retention_until = Column(DateTime)
    
    # Additional metadata
    metadata = Column(JSON, default=dict)
    tags = Column(JSON, default=list)  # For categorization
    
    # Relationships
    user = relationship("User")
    
    __table_args__ = (
        Index("idx_audit_log_event_type_created", "event_type", "created_at"),
        Index("idx_audit_log_user_created", "user_id", "created_at"),
        Index("idx_audit_log_resource", "resource_type", "resource_id"),
    )
    
    def __repr__(self):
        return f"<AuditLog(event_type='{self.event_type}', action='{self.action}')>"